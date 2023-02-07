"""Script for running inference and sampling.

Sample command:
> python scripts/run_inference.py

To run with hyperparameter sweeps:
> python scripts/run_inference.py -m

"""

from mimetypes import init
import os
import time
import tree
import numpy as np
import hydra
import torch
import subprocess
import logging
import pandas as pd
import shutil
from datetime import datetime
import GPUtil
import collections

from biotite.sequence.io import fasta
from hydra.core.hydra_config import HydraConfig
from typing import Optional

from analysis import utils as au
from analysis import metrics
from data import utils as du
from data import residue_constants
from typing import Dict
from experiments import train_se3_diffusion
from experiments import utils as eu
from torch.nn.parallel import DataParallel as DP
from omegaconf import DictConfig, OmegaConf
from openfold.utils import rigid_utils as ru
from openfold.data import data_transforms
import esm


CA_IDX = residue_constants.atom_order['CA']
PATH_TO_MPNN = '/data/rsg/chemistry/jyim/projects/protein_diffusion/ProteinMPNN'


def process_chain(design_pdb_feats):
    chain_feats = {
        'aatype': torch.tensor(design_pdb_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(design_pdb_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(design_pdb_feats['atom_mask']).double()
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
    seq_idx = design_pdb_feats['residue_index'] - np.min(design_pdb_feats['residue_index']) + 1
    chain_feats['seq_idx'] = seq_idx
    chain_feats['res_mask'] = design_pdb_feats['bb_mask']
    chain_feats['residue_index'] = design_pdb_feats['residue_index']
    return chain_feats


def create_pad_feats(pad_amt):
    pad_feats = {        
        'res_mask': torch.ones(pad_amt),
        'fixed_mask': torch.zeros(pad_amt),
        'aatype_impute': torch.zeros((pad_amt, 21)),
        'rigids_impute': torch.zeros((pad_amt, 4, 4)),
        'torsion_impute': torch.zeros((pad_amt, 7, 2)),
    }
    return pad_feats


class Sampler:

    def __init__(
            self,
            conf: DictConfig,
            conf_overrides: Dict=None
        ):
        """Initialize sampler.

        Args:
            conf: inference config.
            gpu_id: GPU device ID.
            conf_overrides: Dict of fields to override with new values.
        """
        self._log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Prepare configs.
        self._conf = conf
        self._infer_conf = conf.inference
        self._diff_conf = self._infer_conf.diffusion

        self._rng = np.random.default_rng(self._infer_conf.seed)

        # Set model hub directory for ESMFold.
        torch.hub.set_dir(self._infer_conf.esm_dir)

        # Set-up accelerator
        self._available_gpus = ''.join(
            [str(x) for x in GPUtil.getAvailable(
                order='memory', limit = 8)])
        if torch.cuda.is_available():
            if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
                self._replica_id = HydraConfig.get().job.num
            else:
                self._replica_id = 0
            self._gpu_id = self._available_gpus[self._replica_id]
            self._log.info(f"Using GPU: {self._gpu_id}")
            self.device = f'cuda:{self._gpu_id}'
        else:
            self.device = 'cpu'

        # Set-up directories
        base_dir = self._infer_conf.base_dir
        self._ckpt_dir = os.path.join(base_dir, self._infer_conf.ckpt_dir)
        if self._infer_conf.output_dir is None:
            output_dir = self._infer_conf.ckpt_dir.replace(
                'pkl_jar/ckpt/', 'samples/')
        else:
            output_dir = os.path.join(base_dir, self._infer_conf.output_dir)
        if self._infer_conf.name is None:
            dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        else:
            dt_string = self._infer_conf.name
        self._output_dir = os.path.join(output_dir, self._infer_conf.task, dt_string)
        os.makedirs(self._output_dir, exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')

        config_path = os.path.join(self._output_dir, 'inference_conf.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f'Saving config to {config_path}')

        # Load models and experiment
        self._load_ckpt(conf_overrides)
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(self.device)

    def _load_ckpt(self, conf_overrides):
        """Loads in model checkpoint."""
        ckpt_files = [
            x for x in os.listdir(self._ckpt_dir) if '.pth' in x or '.pkl' in x]
        if len(ckpt_files) != 1:
            raise ValueError(f'Ambiguous ckpt in {self._ckpt_dir}')
        ckpt_name = ckpt_files[0]
        self._ckpt_path = os.path.join(self._ckpt_dir, ckpt_name)
        self._log.info(f'Loading checkpoint from {self._ckpt_path}')

        # Read checkpoint and create experiment.
        ckpt_pkl = du.read_pkl(
            self._ckpt_path, use_torch=True,
            map_location=f'cuda:{self._available_gpus[0]}')
        ckpt_conf = ckpt_pkl['conf']
        ckpt_model = ckpt_pkl['model']

        # Merge base experiment config with checkpoint config.
        self._ckpt_conf = ckpt_conf
        self._conf = OmegaConf.merge(self._conf, ckpt_conf)
        if conf_overrides is not None:
            self._conf = OmegaConf.merge(self._conf, conf_overrides)

        # Diffuser overrides
        if self._infer_conf.override_diffuser:
            self._log.info('Overriding diffuser settings')
            self._conf.diffuser = OmegaConf.merge(
                self._conf.diffuser, self._infer_conf.diffuser)

        # Prepare model
        self._conf.experiment.ckpt_dir = None
        self._conf.experiment.warm_start = None
        self.exp = train_se3_diffusion.Experiment(
            conf=self._conf)
        self.model = self.exp.model
        # Remove module prefix if it exists.
        ckpt_model = {k.replace('module.', ''):v for k,v in ckpt_model.items()}
        self.model.load_state_dict(ckpt_model)
        if self._infer_conf.dist_mode == 'multi':
            device_ids = [
                f"cuda:{i}"
                for i in self._available_gpus[:self._infer_conf.num_multi_gpu]
            ]
            self._log.info(f"Multi-GPU inference on: {device_ids}")
            self.model = DP(
                self.model,
                device_ids=device_ids
            )

        self.model = self.model.to(self.device)
        self.model.eval()
        self.diffuser = self.exp.diffuser

    def init_data(
            self,
            *,
            rigids_impute,
            aatype_impute,
            torsion_impute,
            fixed_mask,
            res_mask,
        ):
        num_res = res_mask.shape[0]
        diffuse_mask = (1 - fixed_mask) * res_mask
        fixed_mask = fixed_mask * res_mask

        ref_sample = self.diffuser.sample_ref(
            n_samples=num_res,
            rigids_impute=rigids_impute,
            aatype_impute=aatype_impute,
            diffuse_mask=diffuse_mask,
            as_tensor_7=True,
        )
        res_idx = torch.arange(1, num_res+1)
        init_feats = {
            'res_mask': res_mask,
            'seq_idx': res_idx * res_mask,
            'aatype_probs_0': aatype_impute,
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': torsion_impute,
            'sc_aatype_probs_t': np.zeros_like(aatype_impute),
            'sc_ca_t': torch.zeros_like(rigids_impute.get_trans()),
            **ref_sample,
        }
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device), init_feats)
        return init_feats

    def sample(
            self,
            res_mask: np.ndarray,
            res_idx: np.ndarray = None,
            aatype_impute: np.ndarray = None,
            rigids_impute: ru.Rigid = None,
            torsion_impute: np.ndarray = None,
            fixed_mask: np.ndarray = None,
            smc: bool = False,
        ):
        """Sample function.

        Args:
            res_mask: [N] residue mask.
            res_idx: [N] residue indices.
            aatype_impute: [N, 21] amino acid type properties.
            rigids_impute: [N] rigid bodies.
            torsion_impute: [N, 7, 2] torsion angles.
            fixed_mask: [N] motif mask. Residues to stay fixed during sampling.

        Returns:
            Dict with items:
            - prot_traj: [T, N, 37, 3] backbone atom trajectories.
            - aatype_traj: [T, N, 21] aatype probability trajectories.
            - rigid_traj: [T, N, 7] rigid tensor trajectories.
            - trans_traj: [T, N, 3] x0 C-a trajectories.
        """
        assert res_mask.ndim == 1

        def _data_gen():
            return self.init_data(
                rigids_impute=rigids_impute,
                aatype_impute=aatype_impute,
                torsion_impute=torsion_impute,
                fixed_mask=fixed_mask,
                res_mask=res_mask
            )    

        # Run inference
        if smc:
            init_particles = {}
            for _ in range(self._infer_conf.inpainting.num_particles):
                particle_feats = _data_gen()
                for k,v in particle_feats.items():
                    if k not in init_particles:
                        init_particles[k] = []
                    init_particles[k].append(v)
            init_feats = {
                k: torch.concat(v, dim=0)
                for k,v in init_particles.items()
            }
            infer_out = self.exp.smc_inference_fn(
                init_feats,
                num_t=self._diff_conf.num_t,
                min_t=self._diff_conf.min_t,
                aux_traj=True
            )
        else:
            infer_out = self.exp.inference_fn(
                _data_gen(),
                ode=self._diff_conf.ode,
                num_t=self._diff_conf.num_t,
                min_t=self._diff_conf.min_t,
                aux_traj=True,
                self_condition=self._diff_conf.self_condition,
            )

        # Remove batch dimension.
        return tree.map_structure(lambda x: x[:, 0], infer_out)

    def process_motif_row(self, motif_row):
        """Parse row in the motif CSV."""
        motif_path = motif_row.motif_path
        motif_chain_feats = du.parse_pdb_feats(
            'motif', motif_path, chain_id=None)
        return {
            k: process_chain(v) for k,v in motif_chain_feats.items()
        }

    def create_motif_feats(self, chain_feats, start_idx, end_idx):
        """Extract subset of features in chain_feats."""
        motif_length = end_idx - start_idx + 1
        motif_aatype = self.diffuser.seq_diffuser.one_hot(chain_feats['aatype'])
        motif_rigids = chain_feats['rigidgroups_gt_frames'][:, 0]
        pad_feats = {        
            'res_mask': torch.ones(motif_length),
            'fixed_mask': torch.ones(motif_length),
            'aatype_impute': torch.tensor(motif_aatype[start_idx:(end_idx+1)]),
            'rigids_impute': motif_rigids[start_idx:(end_idx+1)],
            'torsion_impute': chain_feats['torsion_angles_sin_cos'][start_idx:(end_idx+1)],
        }
        return pad_feats

    def process_contig(self, sample_contig, all_chain_feats):
        """Creates input features based on contig.

        Args:
            sample_contig: Contig to sample.
            all_chain_feats: Dict of motif features.

        Returns:
            Input features based on contig. For instance, contig
            [5-5, A2-6, 2-2, B3-7] will concatenate padding features of length 5
            to the left of motif residues 2-6 on chain A then padding features
            of length 2 to the left of motif residues 3-7 on chain B.
            Note the padding lengths have already been sampled prior to calling
            this function.
        """

        # Parse contig.
        all_feats = []
        for segment in sample_contig.split(','):
            if segment[0].isnumeric():
                segment_feats = create_pad_feats(int(segment.split('-')[0]))
            else:
                chain_id = segment[0]
                lengths = segment[1:]
                start_idx, end_idx = lengths.split('-')
                chain_feats = all_chain_feats[chain_id]
                res_idx = chain_feats['residue_index']
                if np.all(int(start_idx) != res_idx) or np.all(int(end_idx) != res_idx):
                    raise ValueError('Failed at finding motif residue index')
                start_idx = np.argmax(int(start_idx) == res_idx)
                end_idx = np.argmax(int(end_idx) == res_idx)
                segment_feats = self.create_motif_feats(
                    chain_feats, start_idx, end_idx)
            all_feats.append(segment_feats)
        combined_dict = collections.defaultdict(list)
        for chain_dict in all_feats:
            for feat_name, feat_val in chain_dict.items():
                combined_dict[feat_name].append(feat_val)

        # Concatenate each feature
        for feat_name, feat_vals in combined_dict.items():
            combined_dict[feat_name] = torch.concat(feat_vals, dim=0)
        combined_dict = dict(combined_dict)

        # Center on motif.
        sample_rigids = ru.Rigid.from_tensor_4x4(
            combined_dict['rigids_impute'])
        motif_mask = combined_dict['fixed_mask']
        combined_dict['rigids_impute'] = du.center_on_motif(sample_rigids, motif_mask)
        return combined_dict

    def run_inpainting(self, csv_path):
        """Sets up inference run on inpainting.

        Runs inference based on unconditional config.
        - samples_per_motif: number of samples per motif.
        - target_csv: CSV with information about each motif target.

        All outputs are written to 
            {output_dir}/inpainting/{date_time}
        where {output_dir} is created at initialization.
        """
        inpaint_csv = pd.read_csv(csv_path, index_col=0)
        for _, row in inpaint_csv.iterrows():
            motif_chain_feats = self.process_motif_row(row)
            motif_length = row.length
            motif_contig = row.contig

            design_output_dir = os.path.join(
                self._output_dir, f'{row.target}_{motif_contig}')
            if isinstance(motif_length, str):
                motif_length = [int(x) for x in motif_length.split('-')]
                if len(motif_length) == 1:
                    motif_length.append(int(motif_length[0]) + 1)
            elif np.isnan(motif_length):
                motif_length = None
            else:
                raise ValueError(f'Unrecognized length: {motif_length}')

            os.makedirs(design_output_dir, exist_ok=True)
            self._log.info(f'Inpainting target: {design_output_dir}')

            # Run multiple samples for each motif
            sample_rows = []
            for sample_i in range(self._infer_conf.inpainting.samples_per_motif):

                # Samples a length for each padding length range in the contig.
                # TODO: Add RNG seed.
                sample_contig, sample_length, _ = eu.get_sampled_mask(
                    motif_contig, motif_length, rng=self._rng)

                # Create input features with sampled contig.
                sample_feats = self.process_contig(
                    sample_contig[0], motif_chain_feats)
                design_output = self.sample(
                    smc=self._infer_conf.inpainting.use_smc,
                    **sample_feats
                )
                diffuse_mask = du.move_to_np(1 - sample_feats['fixed_mask'])

                # Output directory for this particular sample.
                sample_i = str(sample_i+1)
                sampled_id = f'sample_{sample_i}_length_{str(sample_length)}'
                sample_output_dir = os.path.join(
                    design_output_dir, sampled_id)
                os.makedirs(sample_output_dir, exist_ok=True)

                # Save sample
                sampled_seq = ''.join(design_output['seq_traj'][0])
                final_aatype = np.argmax(design_output['aatype_probs_traj'][0], axis=-1)
                traj_paths = self.save_traj(
                    design_output['prot_traj'],
                    design_output['rigid_0_traj'],
                    final_aatype,
                    du.move_to_np(sample_feats['res_mask']),
                    diffuse_mask,
                    output_dir=sample_output_dir
                )

                # Run self-consistency
                seq_path = os.path.join(sample_output_dir, 'sampled_seq.fasta')
                du.save_fasta([sampled_seq], ['sampled_seq'], seq_path)
                pdb_path = traj_paths['sample_path']
                mpnn_output_dir = os.path.join(sample_output_dir, 'mpnn')
                os.makedirs(mpnn_output_dir, exist_ok=True)
                shutil.copy(pdb_path, os.path.join(
                    mpnn_output_dir, os.path.basename(pdb_path)))
                _ = self.run_self_consistency(
                    mpnn_output_dir,
                    pdb_path,
                    (1 - diffuse_mask).astype(bool)
                )
                self._log.info(f'Done with sample {sample_i}: {sample_output_dir}')

            sample_csv = pd.DataFrame(sample_rows)
            output_csv_path = os.path.join(design_output_dir, 'sample_summary.csv')
            sample_csv.to_csv(output_csv_path)
            self._log.info(f'Saved sample summary CSV to {output_csv_path}')

    def run_unconditional(self):
        """Sets up inference run on unconditional sampling.

        Runs inference based on unconditional config.
        - samples_per_length: number of samples per sequence length.
        - min_length: minimum sequence length to sample.
        - max_length: maximum sequence length to sample.
        - length_step: gap between lengths to sample.
            i.e. this script will sample all lengths in
            range(min_length, max_length, length_step)

        All outputs are written to 
            {output_dir}/unconditional/{date_time}
        where {output_dir} is created at initialization.
        """
        sample_conf = self._infer_conf.unconditional
        if sample_conf.sampling == 'uniform':
            all_sample_lengths = np.random.randint(
                sample_conf.min_length, sample_conf.max_length, sample_conf.num_lengths
            )
        elif sample_conf.sampling == 'fixed_range':
            all_sample_lengths = range(
                sample_conf.min_length, sample_conf.max_length, sample_conf.length_step
            )
        else:
            raise ValueError(
                f'Unrecognized sampling method: {sample_conf.sampling}')
        for sample_length in all_sample_lengths:
            length_dir = os.path.join(
                self._output_dir, f'length_{sample_length}')
            # if os.path.isdir(length_dir):
            #     self._log.info(f'Done with {length_dir}')
            #     continue
            os.makedirs(length_dir, exist_ok=True)
            self._log.info(f'Sampling length {sample_length}: {length_dir}')
            for sample_i in range(sample_conf.samples_per_length):
                sample_dir = os.path.join(length_dir, f'sample_{sample_i}')
                if os.path.isdir(sample_dir):
                    continue
                os.makedirs(sample_dir, exist_ok=True)
                sample_output = self.sample_unconditional(sample_length)
                final_aatype = np.argmax(
                    sample_output["aatype_probs_traj"][0], axis=-1)
                traj_paths = self.save_traj(
                    sample_output['prot_traj'],
                    sample_output['rigid_0_traj'],
                    final_aatype,
                    np.ones(sample_length),
                    np.zeros(sample_length),
                    output_dir=sample_dir
                )

                # Run ProteinMPNN
                pdb_path = traj_paths['sample_path']
                mpnn_output_dir = os.path.join(sample_dir, 'mpnn')
                os.makedirs(mpnn_output_dir, exist_ok=True)
                shutil.copy(pdb_path, os.path.join(
                    mpnn_output_dir, os.path.basename(pdb_path)))
                _ = self.run_self_consistency(
                    mpnn_output_dir,
                    pdb_path,
                    motif_mask=None
                )
                self._log.info(f'Done sample {sample_i}: {pdb_path}')

    def save_traj(
            self,
            bb_prot_traj: np.ndarray,
            x0_traj: np.ndarray,
            aatype: np.ndarray,
            res_mask: np.ndarray,
            diffuse_mask: np.ndarray,
            output_dir: str
        ):
        """Writes final sample and reverse diffusion trajectory.

        Args:
            bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
                T is number of time steps. First time step is t=eps,
                i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
                N is number of residues.
            x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
            aatype: [T, N, 21] amino acid probability vector trajectory.
            res_mask: [N] residue mask.
            diffuse_mask: [N] which residues are diffused.
            output_dir: where to save samples.

        Returns:
            Dictionary with paths to saved samples.
                'sample_path': PDB file of final state of reverse trajectory.
                'traj_path': PDB file os all intermediate diffused states.
                'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
            b_factors are set to 100 for diffused residues and 0 for motif
            residues if there are any.
        """

        # Write sample.
        final_prot = bb_prot_traj[0]
        res_mask = res_mask.astype(bool)
        diffuse_mask = diffuse_mask.astype(bool)

        prot_name = 'ode' if self._diff_conf.ode else 'sde'
        sample_path = os.path.join(output_dir, prot_name)
        prot_traj_path = os.path.join(
            output_dir, f'{prot_name}_bb_traj')
        x0_traj_path = os.path.join(
            output_dir, f'{prot_name}_x0_traj')
        prot = final_prot[res_mask]
        traj = bb_prot_traj[:, res_mask]
        prot_traj = x0_traj[:, res_mask]

        # Use b-factors to specify which residues are diffused.
        b_factors = np.tile(
            (diffuse_mask[res_mask] * 100)[:, None], (1, 37))

        int_seq = aatype[res_mask]
        sample_path = au.write_prot_to_pdb(
            prot,
            sample_path,
            aatype=int_seq,
            b_factors=b_factors
        )
        prot_traj_path = au.write_prot_to_pdb(
            traj,
            prot_traj_path,
            aatype=int_seq,
            b_factors=b_factors
        )
        x0_traj_path = au.write_prot_to_pdb(
            prot_traj,
            x0_traj_path,
            aatype=int_seq,
            b_factors=b_factors
        )
        return {
            'sample_path': sample_path,
            'traj_path': prot_traj_path,
            'x0_traj_path': x0_traj_path,
        }

    def run_self_consistency(
            self,
            decoy_pdb_dir: str,
            reference_pdb_path: str,
            motif_mask: Optional[np.ndarray]=None):
        """Run self-consistency on design proteins against reference protein.
        
        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file
            motif_mask: Optional mask of which residues are the motif.

        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf
            Writes results in decoy_pdb_dir/sc_results.csv
        """

        # Run PorteinMPNN
        output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
        process = subprocess.Popen([
            'python',
            f'{PATH_TO_MPNN}/helper_scripts/parse_multiple_chains.py',
            f'--input_path={decoy_pdb_dir}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()
        num_tries = 0
        ret = -1
        while ret < 0:
            try:
                process = subprocess.Popen([
                    'python',
                    f'{PATH_TO_MPNN}/protein_mpnn_run.py',
                    '--out_folder',
                    decoy_pdb_dir,
                    '--jsonl_path',
                    output_path,
                    '--device',
                    self.device.split(':')[1],
                    '--num_seq_per_target',
                    str(self._infer_conf.unconditional.seq_per_sample),
                    '--sampling_temp',
                    '0.1',
                    '--seed',
                    '38',
                    '--batch_size',
                    '1',
                ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                self._log.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e
        mpnn_fasta_path = os.path.join(
            decoy_pdb_dir,
            'seqs',
            os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
        )

        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        mpnn_results = {
            'tm_score': [],
            'sample_path': [],
            'header': [],
            'sequence': [],
            'rmsd': [],
        }
        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            mpnn_results['motif_rmsd'] = []
        esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
        os.makedirs(esmf_dir, exist_ok=True)
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)
        for i, (header, string) in enumerate(fasta_seqs.items()):

            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
            _ = self.run_folding(string, esmf_sample_path)
            esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path)
            sample_seq = du.aatype_to_seq(sample_feats['aatype'])

            # Calculate scTM of ESMFold outputs with reference protein
            _, tm_score = metrics.calc_tm_score(
                sample_feats['bb_positions'], esmf_feats['bb_positions'],
                sample_seq, sample_seq)
            rmsd = metrics.calc_aligned_rmsd(
                sample_feats['bb_positions'], esmf_feats['bb_positions'])
            if motif_mask is not None:
                sample_motif = sample_feats['bb_positions'][motif_mask]
                of_motif = esmf_feats['bb_positions'][motif_mask]
                motif_rmsd = metrics.calc_aligned_rmsd(
                    sample_motif, of_motif)
                mpnn_results['motif_rmsd'].append(motif_rmsd)
            mpnn_results['rmsd'].append(rmsd)
            mpnn_results['tm_score'].append(tm_score)
            mpnn_results['sample_path'].append(esmf_sample_path)
            mpnn_results['header'].append(header)
            mpnn_results['sequence'].append(string)

        # Save results to CSV
        csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output

    def sample_unconditional(self, sample_length: int):
        """Sample unconditionally based on length.

        Args:
            sample_length: length to sample

        Returns:
            Sample outputs. See train_se3_diffusion.inference_fn.
        """
        # Process motif features.
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)

        # Initialize data
        ref_sample = self.diffuser.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
            aatype_impute=np.ones((sample_length, 21))
        )
        res_idx = torch.arange(1, sample_length+1)
        zero_aatype = np.zeros((sample_length, 21))
        init_feats = {
            'res_mask': res_mask,
            'seq_idx': res_idx,
            'chain_idx': torch.ones_like(res_idx),
            'aatype_probs_0': zero_aatype,
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2)),
            'sc_aatype_probs_t': zero_aatype,
            'sc_ca_t': np.zeros((sample_length, 3)),
            **ref_sample,
        }
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device), init_feats)

        # Run inference
        sample_out = self.exp.inference_fn(
            init_feats,
            ode=self._diff_conf.ode,
            num_t=self._diff_conf.num_t,
            min_t=self._diff_conf.min_t,
            aux_traj=True,
            self_condition=self._diff_conf.self_condition,
            noise_scale=self._diff_conf.noise_scale,
        )
        return tree.map_structure(lambda x: x[:, 0], sample_out)

@hydra.main(version_base=None, config_path="../config", config_name="inference")
def run(conf: DictConfig) -> None:

    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    sampler = Sampler(conf)
    if conf.inference.task == 'inpainting':
        sampler.run_inpainting(conf.inference.inpainting.target_csv)
    elif conf.inference.task == 'unconditional':
        sampler.run_unconditional()
    else:
        raise ValueError(f'Unrecognized inference task: {conf.inference_task}')
    print(f'Finished in {time.time() - start_time}s')

if __name__ == '__main__':
    run()
