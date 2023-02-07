"""Dataset for SE(3) experiments."""
import tree
import numpy as np
import torch
import pandas as pd
import logging
import random
import functools as fn

from torch.utils import data
from data import utils as du
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def _rog_quantile_curve(df, quantile, eval_x):
    quantile = 0.96
    y_quant = pd.pivot_table(
        df,
        values='radius_gyration', 
        index='modeled_seq_len',
        aggfunc=lambda x: np.quantile(x, quantile)
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    pred_poly_features = poly.fit_transform(eval_x[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1
    return pred_y

class PdbDataset(data.Dataset):
    def __init__(
            self,
            *,
            data_conf,
            diffuser,
            is_training,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._data_conf = data_conf
        self._init_metadata()
        self._diffuser = diffuser

    @property
    def is_training(self):
        return self._is_training

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def data_conf(self):
        return self._data_conf

    def _init_metadata(self):
        """Initialize metadata."""

        # Process CSV with different filtering criterions.
        filter_conf = self.data_conf.filtering
        pdb_csv = pd.read_csv(self.data_conf.csv_path)
        self.raw_csv = pdb_csv
        if filter_conf.allowed_oligomer is not None and len(filter_conf.allowed_oligomer) > 0:
            pdb_csv = pdb_csv[pdb_csv.oligomeric_detail.isin(
                filter_conf.allowed_oligomer)]
        if filter_conf.max_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= filter_conf.max_len]
        if filter_conf.min_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= filter_conf.min_len]
        if filter_conf.max_helix_percent is not None:
            pdb_csv = pdb_csv[
                pdb_csv.helix_percent < filter_conf.max_helix_percent]
        if filter_conf.max_loop_percent is not None:
            pdb_csv = pdb_csv[
                pdb_csv.coil_percent < filter_conf.max_loop_percent]
        if filter_conf.min_beta_percent is not None:
            pdb_csv = pdb_csv[
                pdb_csv.strand_percent > filter_conf.min_beta_percent]
        if filter_conf.rog_quantile > 0.0:
            prot_rog_low_pass = _rog_quantile_curve(
                pdb_csv, 
                filter_conf.rog_quantile,
                np.arange(filter_conf.max_len))
            row_rog_cutoffs = pdb_csv.modeled_seq_len.map(
                lambda x: prot_rog_low_pass[x-1])
            pdb_csv = pdb_csv[pdb_csv.radius_gyration < row_rog_cutoffs]
        if filter_conf.subset is not None:
            pdb_csv = pdb_csv[:filter_conf.subset]
        pdb_csv = pdb_csv.sort_values('modeled_seq_len', ascending=False)

        if self.data_conf.has_splits:
            if self.is_training:
                # TODO: Correct cath data to have split instead of cath_split.
                self.csv = pdb_csv[pdb_csv.cath_split == 'train']
                self._log.info(f'CATH training: {len(self.csv)} examples')
            else:
                valid_csv = pdb_csv[pdb_csv.cath_split == 'validation']
                self.csv = valid_csv.groupby('cath_code').sample(
                    self.data_conf.samples_per_cath,
                    replace=True,
                    random_state=123
                )
                self._log.info(f'CATH validation: {len(self.csv)} examples from {self.csv.cath_code.tolist()}')
        else:
            self._create_split(pdb_csv)

    def _create_split(self, pdb_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = pdb_csv
            self._log.info(f'Training: {len(self.csv)} examples')
        else:
            all_lengths = np.sort(pdb_csv.modeled_seq_len.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self._data_conf.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = pdb_csv[pdb_csv.modeled_seq_len.isin(eval_lengths)]
            eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                self._data_conf.samples_per_eval_length, replace=True, random_state=123)
            eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
            self.csv = eval_csv
            self._log.info(f'Validation: {len(self.csv)} examples with lengths {eval_lengths}')

    @fn.lru_cache(maxsize=50000)
    def _process_csv_row(self, processed_file_path):
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_chain_feats(processed_feats)

        # Only take modeled residues.
        modeled_idx = processed_feats['modeled_idx']
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        del processed_feats['modeled_idx']
        processed_feats = tree.map_structure(
            lambda x: x[min_idx:(max_idx+1)], processed_feats)

        # Run through OpenFold data transforms.
        chain_feats = {
            'aatype': torch.tensor(processed_feats['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        # Re-number residue indices for each chain such that it starts from 1.
        # Randomize chain indices.
        chain_idx = processed_feats["chain_index"]
        res_idx = processed_feats['residue_index']
        new_res_idx = np.zeros_like(res_idx)
        new_chain_idx = np.zeros_like(res_idx)
        all_chain_idx = np.unique(chain_idx).tolist()
        shuffled_chain_idx = np.array(
            random.sample(all_chain_idx, len(all_chain_idx))) - np.min(all_chain_idx) + 1
        for i,chain_id in enumerate(all_chain_idx):
            chain_mask = (chain_idx == chain_id).astype(np.int)
            chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(np.int)
            new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

            # Shuffle chain_index
            replacement_chain_id = shuffled_chain_idx[i]
            new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

        # To speed up processing, only take necessary features
        final_feats = {
            'aatype': chain_feats['aatype'],
            'seq_idx': new_res_idx,
            'chain_idx': chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
            'residue_index': processed_feats['residue_index'],
            'res_mask': processed_feats['bb_mask'],
            'atom37_pos': chain_feats['all_atom_positions'],
            'atom37_mask': chain_feats['all_atom_mask'],
            'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'],
            # 'rigidgroups_0_exists': chain_feats['rigidgroups_gt_exists'],
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'],
            # 'torsion_angles_mask': chain_feats['torsion_angles_mask'],
        }
        return final_feats

    def _create_diffused_masks(self, atom37_pos, rng, row):
        bb_pos = atom37_pos[:, residue_constants.atom_order['CA']]
        dist2d = np.linalg.norm(bb_pos[:, None, :] - bb_pos[None, :, :], axis=-1)

        # Randomly select residue then sample a distance cutoff
        # TODO: Use a more robust diffuse mask sampling method.
        diff_mask = np.zeros_like(bb_pos)
        attempts = 0
        while np.sum(diff_mask) < 1:
            crop_seed = rng.integers(dist2d.shape[0])
            seed_dists = dist2d[crop_seed]
            max_scaffold_size = min(
                self._data_conf.scaffold_size_max,
                seed_dists.shape[0] - self._data_conf.motif_size_min
            )
            scaffold_size = rng.integers(
                low=self._data_conf.scaffold_size_min,
                high=max_scaffold_size
            )
            dist_cutoff = np.sort(seed_dists)[scaffold_size]
            diff_mask = (seed_dists < dist_cutoff).astype(float)
            attempts += 1
            if attempts > 100:
                raise ValueError(
                    f'Unable to generate diffusion mask for {row}')
        return diff_mask

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):

        # Sample data example.
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        if 'pdb_name' in csv_row:
            pdb_name = csv_row['pdb_name']
        elif 'chain_name' in csv_row:
            pdb_name = csv_row['chain_name']
        else:
            raise ValueError('Need chain identifier.')
        processed_file_path = csv_row['processed_path']
        chain_feats = self._process_csv_row(processed_file_path)

        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)

        # Inpainting or hallucination.
        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(
            chain_feats['rigidgroups_0'])[:, 0]
        if (self._data_conf.hallucination_percent is not None) and (rng.random() < self._data_conf.hallucination_percent):
            diffused_mask = np.ones_like(chain_feats['res_mask'])
            hallucination = True
        else:
            diffused_mask = self._create_diffused_masks(
                chain_feats['atom37_pos'], rng, csv_row)
            gt_bb_rigid = du.center_on_motif(gt_bb_rigid, 1 - diffused_mask)
            hallucination = False
        if np.sum(diffused_mask) < 1:
            raise ValueError('Must be diffused')
        fixed_mask = 1 - diffused_mask
        chain_feats['fixed_mask'] = fixed_mask
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        gt_aatype_probs =  self.diffuser.seq_diffuser.one_hot(
            chain_feats['aatype'])
        chain_feats['aatype_probs_0'] = gt_aatype_probs
        chain_feats['sc_aatype_probs_t'] = np.zeros_like(gt_aatype_probs)
        chain_feats['sc_ca_t'] = torch.zeros_like(gt_bb_rigid.get_trans())

        # Sample t and diffuse.
        if self.is_training:
            t_seq = rng.uniform(self._data_conf.min_t, 1.0)
            if self._data_conf.mixed_t:
                t_struct = rng.uniform(self._data_conf.min_t, 1.0)
            else:
                t_struct = t_seq
            diff_feats_t = self._diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                aatype_probs_0=gt_aatype_probs,
                t_seq=t_seq,
                t_struct=t_struct,
                diffuse_mask=None if hallucination else diffused_mask
            )
        else:
            t_seq = 1.0
            t_struct = 1.0
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=gt_bb_rigid.shape[0],
                rigids_impute=gt_bb_rigid,
                aatype_impute=gt_aatype_probs,
                diffuse_mask=None if hallucination else diffused_mask,
                as_tensor_7=True,
            )
        if self._data_conf.aatype_masking:
            mask_percent = rng.random()
            aatype_input = diff_feats_t['aatype_probs_t']
            mask_residue = rng.random((aatype_input.shape[0],)) < mask_percent
            masked_aatype = np.zeros_like(aatype_input)
            masked_aatype[:, -1] = 1.0
            diff_feats_t['aatype_probs_t'] = (
                masked_aatype * mask_residue[:, None] +
                aatype_input * (1 - mask_residue)[:, None] 
            )
        chain_feats.update(diff_feats_t)
        chain_feats['t_seq'] = t_seq
        chain_feats['t_struct'] = t_struct


        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats 
        )

        final_feats = du.pad_feats(final_feats, csv_row['modeled_seq_len'])
        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name


class LengthSampler(data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
        ):
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv

    def __iter__(self):
        return iter(range(len(self._data_csv)))

    def __len__(self):
        return len(self._data_csv)

class TrainSampler(data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            batch_size,
            sample_mode,
        ):
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        self._batch_size = batch_size
        self.epoch = 0
        self._sample_mode = sample_mode

    def __iter__(self):
        if self._sample_mode == 'length_batch':
            sampled_order = self._data_csv.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
            return iter(sampled_order['index'].tolist())
        elif self._sample_mode == 'time_batch':
            random.shuffle(self._dataset_indices)
            repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
            return iter(repeated_indices)
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        if self._sample_mode == 'length_batch':
            return len(self._data_csv['modeled_seq_len'].unique()) * self._batch_size
        elif self._sample_mode == 'time_batch':
            return len(self._dataset_indices) * self._batch_size
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')
