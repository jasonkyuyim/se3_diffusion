"""Pytorch script for training SE(3) protein diffusion.

To run:

> python experiments/train_se3_diffusion.py

Without Wandb,

> python experiments/train_se3_diffusion.py experiment.use_wandb=False

To modify config options with the command line,

> python experiments/train_se3_diffusion.py experiment.batch_size=32

"""

from collections import defaultdict
from collections import deque
import os
import torch
import GPUtil
import time
import tree
import numpy as np
import wandb
import hydra
import logging
import copy
import random
from datetime import datetime
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pandas as pd
import torch.distributed as dist

from torch.utils import data
from torch.nn.parallel import DataParallel as DP
from openfold.utils import rigid_utils as ru
from hydra.core.hydra_config import HydraConfig

from analysis import utils as au
from analysis import metrics
from data import digs_data_loader
from data import residue_constants
from data import rosetta_data_loader
from data import se3_diffuser
from data import utils as du
from data import all_atom
from model import score_network
from model import ode_utils
from experiments import utils as eu


class Experiment:

    def __init__(
            self,
            *,
            conf: DictConfig,
        ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self._log = logging.getLogger(__name__)
        self._available_gpus = ''.join(
            [str(x) for x in GPUtil.getAvailable(
                order='memory', limit = 8)])

        # Warm starting
        if conf.experiment.warm_start:
            ckpt_dir = conf.experiment.warm_start
            self._log.info(f'Warm starting from: {ckpt_dir}')
            ckpt_files = [
                x for x in os.listdir(ckpt_dir)
                if 'pkl' in x or '.pth' in x
            ]
            if len(ckpt_files) != 1:
                raise ValueError(f'Ambiguous ckpt in {ckpt_dir}')
            ckpt_name = ckpt_files[0]
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            self._log.info(f'Loading checkpoint from {ckpt_path}')
            ckpt_pkl = du.read_pkl(ckpt_path, use_torch=True)
            ckpt_model = ckpt_pkl['model']
            if conf.experiment.use_warm_start_conf:
                OmegaConf.set_struct(conf, False)
                conf = OmegaConf.merge(conf, ckpt_pkl['conf'])
                OmegaConf.set_struct(conf, True)
            conf.experiment.warm_start = ckpt_dir
        else:
            ckpt_model = None

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            self._exp_conf.name = (
                f'{self._exp_conf.name}_{HydraConfig.get().job.num}')
        self._diff_conf = conf.diffuser
        self._model_conf = conf.model
        self._data_conf = conf.data
        self._dist_mode = self._exp_conf.dist_mode
        self._use_wandb = self._exp_conf.use_wandb

        # Initialize experiment objects
        self.trained_epochs = 0
        self.trained_steps = 0
        self._diffuser = se3_diffuser.SE3Diffuser(self._diff_conf)
        self._model = score_network.ScoreNetwork(
            self._model_conf, self.diffuser)

        if ckpt_model is not None:
            ckpt_model = {k.replace('module.', ''):v for k,v in ckpt_model.items()}
            self._model.load_state_dict(ckpt_model, strict=True)

        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._exp_conf.num_parameters = num_parameters
        self._log.info(f'Number of model parameters {num_parameters}')
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._exp_conf.learning_rate)

        if self._exp_conf.use_scheduler:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, T_max=10, eta_min=0.0001)
        else:
            self._scheduler = None

        dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        if self._exp_conf.ckpt_dir is not None:
            # Set-up checkpoint location
            ckpt_dir = os.path.join(
                self._exp_conf.ckpt_dir,
                self._exp_conf.name,
                dt_string)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            self._exp_conf.ckpt_dir = ckpt_dir
            self._log.info(f'Checkpoints saved to: {ckpt_dir}')
            eval_dir = os.path.join(
                self._exp_conf.eval_dir,
                self._exp_conf.name,
                dt_string)
            self._exp_conf.eval_dir = eval_dir
            self._log.info(f'Evaluation saved to: {eval_dir}')
        else:
            self._log.info('Checkpoint not being saved.')

        self._aux_data_history = deque(maxlen=100)

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

    def create_digs_dataset(self, replica_id, num_replicas):
        # Datasets
        train_dataset = digs_data_loader.DistilledDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=True
        )

        valid_dataset = digs_data_loader.DistilledDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=False
        )

        # Samplers
        train_sampler = digs_data_loader.DistributedWeightedSampler(
            data_conf=self._data_conf,
            dataset=train_dataset,
            num_replicas=num_replicas,
            rank=replica_id
        )
        valid_sampler = data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=num_replicas,
            rank=replica_id
        )

        # Loaders
        if self._dist_mode == 'single':
            num_workers = 0
        else:
            num_workers = self._exp_conf.num_loader_workers
        train_loader = data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self._exp_conf.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            prefetch_factor=self._exp_conf.prefetch_factor if num_workers > 0 else 2,
        )
        valid_loader = data.DataLoader(
            valid_dataset,
            sampler=valid_sampler,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            prefetch_factor=self._exp_conf.prefetch_factor if num_workers > 0 else 2
        )

        # Return samplers in order to set epoch during training loop.
        return train_loader, train_sampler, valid_loader, valid_sampler

    def create_dataset(self):

        # Datasets
        train_dataset = rosetta_data_loader.PdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=True
        )

        valid_dataset = rosetta_data_loader.PdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=False
        )

        train_sampler = rosetta_data_loader.TrainSampler(
            data_conf=self._data_conf,
            dataset=train_dataset,
            batch_size=self._exp_conf.batch_size,
            sample_mode=self._exp_conf.sample_mode,
        )

        valid_sampler = None

        # Loaders
        num_workers = self._exp_conf.num_loader_workers
        train_loader = du.create_data_loader(
            train_dataset,
            sampler=train_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=self._exp_conf.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            max_squared_res=self._exp_conf.max_squared_res,
        )
        valid_loader = du.create_data_loader(
            valid_dataset,
            sampler=valid_sampler,
            np_collate=False,
            length_batch=False,
            batch_size=self._exp_conf.eval_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        return train_loader, valid_loader, train_sampler, valid_sampler

    def start_training(self, return_logs=False):
        """Start training based on distribution strategy."""

        if self._dist_mode == 'slurm':
            num_replicas = int(os.environ["SLURM_NTASKS"])
            replica_id = int(os.environ["SLURM_PROCID"])
            self._log.info(f"Training SLURM replica: {replica_id}/{num_replicas}")
            self.train(replica_id, num_replicas)

        else:
            # Set environment variables for which GPUs to use.
            if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
                chosen_gpu = int(HydraConfig.get().job.num)
            else:
                chosen_gpu = 0

            # Return logs for training single processed.
            return self.train(
                chosen_gpu, 1, return_logs=return_logs, init_wandb=True)

    def init_wandb(self):
        self._log.info('Initializing Wandb.')
        conf_dict = OmegaConf.to_container(self._conf, resolve=True)
        wandb.init(
            project='protein-diffusion-v2',
            name=self._exp_conf.name,
            config=dict(eu.flatten_dict(conf_dict)),
            dir=self._exp_conf.wandb_dir,
        )
        self._exp_conf.run_id = wandb.util.generate_id()
        self._exp_conf.wandb_dir = wandb.run.dir
        self._log.info(
            f'Wandb: run_id={self._exp_conf.run_id}, run_dir={self._exp_conf.wandb_dir}')

    def train(self, replica_id, num_replicas, return_logs=False, init_wandb=False):
        if (init_wandb or replica_id == 0) and self._use_wandb:
            self.init_wandb()

        if torch.cuda.is_available() and self._exp_conf.use_gpu:
            gpu_id = self._available_gpus[replica_id]
            device = f"cuda:{gpu_id}"
        else:
            device = 'cpu'
        self._log.info(f"Using device: {device}")

        if self._dist_mode == 'multi':
            device_ids = [
                f"cuda:{i}" for i in self._available_gpus[:self._exp_conf.multi_gpu_size]
            ]
            self._log.info(f"Multi-GPU training on GPUs: {device_ids}")
            self._model = DP(self._model, device_ids=device_ids)
        self._model = self.model.to(device)
        self._model.train()

        (
            train_loader,
            valid_loader,
            train_sampler,
            valid_sampler
        ) = self.create_dataset()

        logs = []
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)
            epoch_log = self.train_epoch(
                    train_loader, valid_loader, device, return_logs)
            if return_logs:
                logs.append(epoch_log)
            self.trained_epochs = epoch
            if self._scheduler is not None:
                self._scheduler.step()

        self._log.info('Done')
        return logs

    def update_fn(self, data):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        loss, aux_data = self.loss_fn(data)
        loss.backward()
        self._optimizer.step()
        return loss, aux_data

    def train_epoch(self, train_loader, valid_loader, device, return_logs=False):
        log_lossses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        step_time = time.time()
        for train_feats in train_loader:
            train_feats = tree.map_structure(
                lambda x: x.to(device), train_feats)
            loss, aux_data = self.update_fn(train_feats)
            if return_logs:
                global_logs.append(loss)
            for k,v in aux_data.items():
                log_lossses[k].append(du.move_to_np(v))
            self.trained_steps += 1

            # Logging to terminal
            if self.trained_steps == 1 or self.trained_steps % self._exp_conf.log_freq == 0:
                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self._exp_conf.log_freq / elapsed_time
                rolling_losses = tree.map_structure(np.mean, log_lossses)
                loss_log = ' '.join([
                    f'{k}={v[0]:.4f}'
                    for k,v in rolling_losses.items() if 'batch' not in k
                ])
                self._log.info(
                    f'[{self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}')
                log_lossses = defaultdict(list)

            # Take checkpoint
            if self._exp_conf.ckpt_dir is not None and (
                    (self.trained_steps % self._exp_conf.ckpt_freq) == 0
                    or (self._exp_conf.early_ckpt and self.trained_steps == 100)
                ):
                ckpt_path = os.path.join(
                    self._exp_conf.ckpt_dir, f'step_{self.trained_steps}.pth')
                du.write_checkpoint(
                    ckpt_path, self.model.state_dict(), self._conf,
                    logger=self._log, use_torch=True)

                # Run evaluation
                # TODO: Dispatch eval as a subprocess.
                self._log.info(f'Running evaluation of {ckpt_path}')
                start_time = time.time()
                eval_dir = os.path.join(
                    self._exp_conf.eval_dir, f'step_{self.trained_steps}')
                os.makedirs(eval_dir, exist_ok=True)
                ckpt_metrics = self.eval_fn(
                    eval_dir, valid_loader, device,
                    noise_scale=self._exp_conf.noise_scale
                )
                eval_time = time.time() - start_time
                self._log.info(f'Finished evaluation in {eval_time:.2f}s')
            else:
                ckpt_metrics = None
                eval_time = None

            # Remote log to Wandb.
            if self._use_wandb:
                step_time = time.time() - step_time
                example_per_sec = self._exp_conf.batch_size / step_time
                step_time = time.time()
                wandb_logs = {
                    'loss': loss,
                    'rotation_loss': aux_data['rot_loss'],
                    'translation_loss': aux_data['trans_loss'],
                    'psi_loss': aux_data['psi_loss'],
                    'aatype_loss': aux_data['aatype_loss'],
                    'bb_atom_loss': aux_data['bb_atom_loss'],
                    'dist_mat_loss': aux_data['batch_dist_mat_loss'],
                    'batch_size': aux_data['examples_per_step'],
                    'res_length': aux_data['res_length'],
                    'examples_per_sec': example_per_sec,
                    'num_epochs': self.trained_epochs,
                }

                # Stratified losses
                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t_struct']),
                    du.move_to_np(aux_data['batch_rot_loss']),
                    loss_name='rot_loss',
                ))

                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t_struct']),
                    du.move_to_np(aux_data['batch_trans_loss']),
                    loss_name='trans_loss',
                ))

                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t_struct']),
                    du.move_to_np(aux_data['batch_psi_loss']),
                    loss_name='psi_loss',
                ))

                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t_seq']),
                    du.move_to_np(aux_data['batch_aatype_loss']),
                    loss_name='aatype_loss',
                ))

                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t_struct']),
                    du.move_to_np(aux_data['batch_bb_atom_loss']),
                    loss_name='bb_atom_loss',
                ))

                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t_struct']),
                    du.move_to_np(aux_data['batch_dist_mat_loss']),
                    loss_name='dist_mat_loss',
                ))

                if ckpt_metrics is not None:
                    wandb_logs['eval_time'] = eval_time
                    for metric_name in metrics.ALL_METRICS:
                        wandb_logs[metric_name] = ckpt_metrics[metric_name].mean()
                    eval_table = wandb.Table(
                        columns=ckpt_metrics.columns.to_list()+['structure'])
                    for _, row in ckpt_metrics.iterrows():
                        pdb_path = row['sample_path']
                        row_metrics = row.to_list() + [wandb.Molecule(pdb_path)]
                        eval_table.add_data(*row_metrics)
                    wandb_logs['sample_metrics'] = eval_table

                wandb.log(wandb_logs, step=self.trained_steps)

            if torch.isnan(loss):
                if self._use_wandb:
                    wandb.alert(
                        title="Encountered NaN loss",
                        text=f"Loss NaN after {self.trained_epochs} epochs, {self.trained_steps} steps"
                    )
                raise Exception(f'NaN encountered')

        if return_logs:
            return global_logs

    def eval_fn(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0):
        ckpt_eval_metrics = []
        for valid_feats, pdb_names in valid_loader:
            res_mask = du.move_to_np(valid_feats['res_mask'].bool())
            fixed_mask = du.move_to_np(valid_feats['fixed_mask'].bool())
            aatype = du.move_to_np(valid_feats['aatype'])
            gt_prot = du.move_to_np(valid_feats['atom37_pos'])
            batch_size = res_mask.shape[0]
            valid_feats = tree.map_structure(
                lambda x: x.to(device), valid_feats)

            # Run inference
            infer_out = self.inference_fn(
                valid_feats, min_t=min_t, num_t=num_t, noise_scale=noise_scale)
            final_prot = infer_out['prot_traj'][0]
            final_aatype_probs = infer_out['aatype_probs_traj'][0]
            for i in range(batch_size):
                num_res = int(np.sum(res_mask[i]).item())
                unpad_fixed_mask = fixed_mask[i][res_mask[i]]
                unpad_diffused_mask = 1 - unpad_fixed_mask
                unpad_prot = final_prot[i][res_mask[i]]
                unpad_aatype_probs = final_aatype_probs[i][res_mask[i]]
                unpad_gt_prot = gt_prot[i][res_mask[i]]
                unpad_gt_aatype = aatype[i][res_mask[i]]
                percent_diffused = np.sum(unpad_diffused_mask) / num_res

                # Extract argmax predicted aatype
                saved_path = au.write_prot_to_pdb(
                    unpad_prot,
                    os.path.join(
                        eval_dir,
                        f'len_{num_res}_sample_{i}_diffused_{percent_diffused:.2f}.pdb'
                    ),
                    aatype=np.argmax(unpad_aatype_probs, axis=-1),
                    no_indexing=True,
                    b_factors=np.tile(1 - unpad_fixed_mask[..., None], 37) * 100
                )
                try:
                    sample_metrics = metrics.protein_metrics(
                        pdb_path=saved_path,
                        atom37_pos=unpad_prot,
                        aatype_probs=unpad_aatype_probs,
                        gt_atom37_pos=unpad_gt_prot,
                        gt_aatype=unpad_gt_aatype,
                        diffuse_mask=unpad_diffused_mask,
                    )
                except ValueError as e:
                    self._log.warning(
                        f'Failed evaluation of length {num_res} sample {i}: {e}')
                    continue
                sample_metrics['step'] = self.trained_steps
                sample_metrics['num_res'] = num_res
                sample_metrics['fixed_residues'] = np.sum(unpad_fixed_mask)
                sample_metrics['diffused_percentage'] = percent_diffused
                sample_metrics['sample_path'] = saved_path
                sample_metrics['gt_pdb'] = pdb_names[i]

                # Likelihood computation
                # Pull out just item i in the batch
                valid_feats_i = tree.map_structure(
                    lambda x: x[i],
                    valid_feats
                )

                # compute log likelihood twice to compare
                if self._exp_conf.likelihood_metric:
                    _, log_lik_val = self.log_likelihood(valid_feats_i, num_t=None,
                            center=True)
                    sample_metrics['log_lik'] = log_lik_val

                ckpt_eval_metrics.append(sample_metrics)

        # Save metrics as CSV.
        eval_metrics_csv_path = os.path.join(eval_dir, 'metrics.csv')
        ckpt_eval_metrics = pd.DataFrame(ckpt_eval_metrics)
        ckpt_eval_metrics.to_csv(eval_metrics_csv_path, index=False)
        return ckpt_eval_metrics

    def _self_conditioning(self, batch):
        model_sc = self.model(batch)
        batch['sc_aatype_probs_t'] = model_sc['aatype_probs']
        if self._model_conf.rigid_prediction:
            batch['sc_ca_t'] = model_sc['rigids'][..., 4:]
        return batch

    def loss_fn(self, batch):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """
        if self._model_conf.embed.embed_self_conditioning and random.random() > 0.5:
            with torch.no_grad():
                batch = self._self_conditioning(batch)
        model_out = self.model(batch)
        bb_mask = batch['res_mask']
        diffuse_mask = 1 - batch['fixed_mask']
        loss_mask = bb_mask * diffuse_mask
        batch_size, num_res = bb_mask.shape

        gt_rot_score = batch['rot_score']
        gt_trans_score = batch['trans_score']
        rot_score_scaling = batch['rot_score_scaling']
        trans_score_scaling = batch['trans_score_scaling']
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_score = model_out['rot_score'] * diffuse_mask[..., None]
        pred_trans_score = model_out['trans_score'] * diffuse_mask[..., None]

        # Translation score loss
        trans_score_mse = (gt_trans_score - pred_trans_score)**2 * loss_mask[..., None]
        trans_score_loss = torch.sum(
            trans_score_mse / trans_score_scaling[:, None, None]**2,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Translation x0 loss
        gt_trans_x0 = batch['rigids_0'][..., 4:] * self._exp_conf.coord_loss_scaling
        pred_trans_x0 = model_out['rigids'][..., 4:] * self._exp_conf.coord_loss_scaling
        trans_x0_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0)**2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        if self._exp_conf.trans_x0_threshold is not None:
            trans_loss = (
                trans_score_loss * (batch['t_struct'] > self._exp_conf.trans_x0_threshold)
                + trans_x0_loss * (batch['t_struct'] <= self._exp_conf.trans_x0_threshold)
            )
        else:
            trans_loss = trans_score_loss
        trans_loss *= self._exp_conf.trans_loss_weight
        trans_loss *= int(self._diff_conf.se3.diffuse_trans)

        # Rotation loss
        rot_mse = (gt_rot_score - pred_rot_score)**2 * loss_mask[..., None]
        rot_loss = torch.sum(
            rot_mse / rot_score_scaling[:, None, None]**2,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)
        rot_loss *= self._exp_conf.rot_loss_weight
        rot_loss *= int(self._diff_conf.se3.diffuse_rot)

        # Psi loss
        psi_mse = (model_out['psi'] - batch['torsion_angles_sin_cos'][..., 2, :])**2
        psi_loss = torch.sum(
            psi_mse * loss_mask[..., None],
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)
        psi_loss *= self._exp_conf.psi_loss_weight
        psi_loss *= self._exp_conf.psi_loss_t_filter > batch['t_struct']

        # Backbone atom loss
        pred_atom37 = model_out['atom37'][:, :, :5]
        gt_rigids = ru.Rigid.from_tensor_7(batch['rigids_0'].type(torch.float32))
        gt_psi = batch['torsion_angles_sin_cos'][..., 2, :]
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(
            gt_rigids, gt_psi)
        gt_atom37 = gt_atom37[:, :, :5]
        atom37_mask = atom37_mask[:, :, :5]

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_mse = (
            pred_atom37 - gt_atom37) * self._exp_conf.bb_atom_scaling
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            bb_atom_mse**2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3)
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss *= self._exp_conf.bb_atom_loss_weight
        bb_atom_loss *= batch['t_struct'] < self._exp_conf.bb_atom_loss_t_filter

        # Pairwise distance loss
        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res*5, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_atom37.reshape([batch_size, num_res*5, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 5))
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res*5])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, 5))
        flat_res_mask = flat_res_mask.reshape([batch_size, num_res*5])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # No loss on anything >6A
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask  = pair_dist_mask * proximity_mask

        pair_dist_diff = (gt_pair_dists - pred_pair_dists) * self._exp_conf.dist_mat_scaling
        dist_mat_loss = torch.sum(pair_dist_diff**2 * pair_dist_mask, dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)
        dist_mat_loss *= self._exp_conf.dist_mat_loss_weight
        dist_mat_loss *= batch['t_struct'] < self._exp_conf.dist_mat_loss_t_filter

        if self._model_conf.aatype_prediction:
            pred_aatype = model_out['aatype_logits']
            gt_aatype = batch['aatype']
            # Don't compute loss on unknown aatypes
            # Compute loss over all residues regardless of diffused.
            aatype_loss_mask = bb_mask * (
                gt_aatype < residue_constants.unk_restype_index)
            known_aatype = gt_aatype * (aatype_loss_mask).type(gt_aatype.dtype)
            aatype_loss = torch.nn.functional.cross_entropy(
                pred_aatype.reshape(-1, pred_aatype.shape[-1]),
                known_aatype.reshape(-1),
                reduction='none'
            ).reshape(bb_mask.shape)
            aatype_loss = torch.sum(
                aatype_loss * aatype_loss_mask, dim=-1
            ) / (torch.sum(aatype_loss_mask, dim=-1) + 1e-10)
            aatype_loss *= self._exp_conf.aatype_loss_weight
        else:
            aatype_loss = torch.zeros_like(trans_loss)

        final_loss = (
            rot_loss
            + trans_loss
            + psi_loss
            + aatype_loss
            + bb_atom_loss
            + dist_mat_loss
        )

        def normalize_loss(x):
            return x.sum() /  (batch_loss_mask.sum() + 1e-10)

        aux_data = {
            'batch_train_loss': final_loss,
            'batch_rot_loss': rot_loss,
            'batch_trans_loss': trans_loss,
            'batch_psi_loss': psi_loss,
            'batch_aatype_loss': aatype_loss,
            'batch_bb_atom_loss': bb_atom_loss,
            'batch_dist_mat_loss': dist_mat_loss,
            'total_loss': normalize_loss(final_loss),
            'rot_loss': normalize_loss(rot_loss),
            'trans_loss': normalize_loss(trans_loss),
            'psi_loss': normalize_loss(psi_loss),
            'aatype_loss': normalize_loss(aatype_loss),
            'bb_atom_loss': normalize_loss(bb_atom_loss),
            'dist_mat_loss': normalize_loss(dist_mat_loss),
            'examples_per_step': torch.tensor(batch_size),
            'res_length': torch.mean(torch.sum(bb_mask, dim=-1)),
        }

        # Maintain a history of the past N number of steps.
        # Helpful for debugging.
        self._aux_data_history.append({
            'aux_data': aux_data,
            'model_out': model_out,
            'batch': batch
        })

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return normalize_loss(final_loss), aux_data

    def _calc_trans_0(self, trans_score, trans_t, t):
        beta_t = self._diffuser._se3_diffuser._r3_diffuser.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_score * cond_var + trans_t) / torch.exp(-1/2*beta_t)

    def _set_t_feats(self, feats, t, t_placeholder):
        feats['t_seq'] = t * t_placeholder
        feats['t_struct'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.se3_score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats

    def log_likelihood(self, data_0, num_t=None, center=True):
        """log_likelihood evaluation with probability flow ODE

        Args:
            data_0: data for time t=0
            num_t: number of time steps in discretization
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_0)

        # Run model and pull out score
        sample_feats_batched = tree.map_structure(
            lambda x: x[None],
            sample_feats
        )

        assert sample_feats['rigids_t'].ndim == 2, "Can compute likelihood of only one example at a time"
        if num_t is None:
            num_t = self._data_conf.num_t
        steps = np.linspace(1e-3, 1.0, num_t)
        dt = 1/num_t
        all_bb_prots = []


        def get_tilde_f_from_6D_and_t(x_t_as_6D, t, sample_feats_batched):
            """get_tilde_f_from_6D_and_t computes the drift function for the probability flow ODE

            Args:
                x_t_as_6D: [N, 6] translations and then rotations
                t: scalar time step
                sample_feats_batched: input features (including batch dimension)
            
            returns:
                tilde_f: 
            """
            sample_feats_batched['t'] = t * torch.ones_like(sample_feats_batched['t'])
            rot_score_norm, trans_score_norm = self.diffuser.se3_score_scaling(t)
            sample_feats_batched['rot_score_norm'] = rot_score_norm * torch.ones_like(sample_feats_batched['rot_score_scaling'])
            sample_feats_batched['trans_score_norm'] = trans_score_norm * torch.ones_like(sample_feats_batched['trans_score_scaling'])

            # Assemble the rigid_t as a tensor 7
            rigids_t = ode_utils.x_t_6D_to_tensor7(x_t_as_6D)
            sample_feats_batched['rigids_t'] = rigids_t

            model_out = self.model(sample_feats_batched)
            score_t_as_6D = torch.cat([model_out['trans_score'][0], model_out['rot_score'][0]], dim=-1)

            # Compute tilde_f
            tilde_f_val = self.diffuser._se3_diffuser.tilde_f(
                x_t_as_6D, score_t_as_6D, sample_feats_batched['res_mask'][0], t)
            return tilde_f_val

        rigids_t = sample_feats_batched['rigids_0'][0] # shape [L, 7]
        mask = sample_feats_batched['res_mask'][0] # shape [L]

        log_lik_total = 0.

        # Loop for ODE
        for t in steps:
            # compute 6D representation of shape [L, 6]
            x_t_as_6D = ode_utils.tensor7_to_6D(rigids_t)

            # Compute derivative for simulating the ODE
            get_tilde_f_from_6D = lambda x_t_as_6D: get_tilde_f_from_6D_and_t(
                    x_t_as_6D, t, sample_feats_batched)
            tilde_f_val = get_tilde_f_from_6D(x_t_as_6D).detach()
            
            # Mask out masked positions corresponding to residues not being modeled.
            mask_idcs = torch.where(1-mask.clone().detach())
            tilde_f_val[mask_idcs] = 0.

            # Update 6D and rigid
            x_t_next_as_6D = x_t_as_6D.clone().detach()
            if self.diffuser._se3_diffuser._so3_diffuser.equivariant_score:
                x_t_next_as_6D[torch.where(mask)] = ode_utils.compose_6D(
                    x_t_as_6D[torch.where(mask)],
                    tilde_f_val[torch.where(mask)]*dt
                    )
            else:
                x_t_next_as_6D[torch.where(mask)] = ode_utils.compose_6D(
                    tilde_f_val[torch.where(mask)]*dt,
                    x_t_as_6D[torch.where(mask)]
                    )

            if center:
                # force center of mass to zero
                x_t_next_as_6D = ode_utils.center_x_t_6D(x_t_next_as_6D, mask)
            rigids_next = ode_utils.x_t_6D_to_tensor7(x_t_next_as_6D)

            ### Compute the divergence of tilde f with Skilling-Hutchinson's estimator
            with torch.enable_grad():
                # x_t_as_6D is [L, 6]
                x_t_as_6D.requires_grad_(True)
                tilde_f_val = get_tilde_f_from_6D(x_t_as_6D)[torch.where(mask)]
                epsilon = torch.normal(mean=0, std=1, size=tilde_f_val.shape).to(tilde_f_val.device)
                grad_tilde_f_val_e = torch.autograd.grad(tilde_f_val, x_t_as_6D, epsilon)[0]
                grad_tilde_f_val_e = grad_tilde_f_val_e[torch.where(mask)]*epsilon

                # stochastic estimate of divergence.
                div_tilde_f_val = grad_tilde_f_val_e.sum()

            log_lik_total += -div_tilde_f_val * dt

            # Update rigids_t for next iteration
            rigids_t = rigids_next

        # Compute likelihood for last term (at t=1) -- since the reference on
        # SO(3) is uniform we do this for translations only.
        trans = ru.Rigid.from_tensor_7(rigids_t).get_trans()[torch.where(mask)]
        log_lik_total += -(1/2)*(trans**2).sum()

        # Following Emile's suggestion, add in log likelihood on SO3 & Reference distribution.  These cancel out
        # Account for Jacobian of exponential map so that the divergence term is invariant to rotation.
        rot_vecs = x_t_as_6D[..., 3:]
        omega = rot_vecs.norm(dim=-1)[torch.where(mask)]
        log_lik_total += torch.log(
            (1.-torch.cos(omega)) / torch.pi # IGSO3 uniform density for rotation angle
            / (4*torch.pi*(omega**2)) # divide by surface of the sphere to get IGSO3 density on rotation vector
        ).sum()
        log_lik_total += -torch.log(
            (2.-2.*torch.cos(omega)) / omega**2 # divide by Volume of SO3
        ).sum()

        return rigids_t, log_lik_total

    def log_importance_weights(self, trans_t, score_t, i, t, dt, motif_forward_diffusion, motif_mask):
        """Computes log importance weights

        Args:

        """
        # TODO: Double check trans parameters is correct
        mu, std = self.diffuser.trans_parameters(
            trans_t, score_t, t, motif_mask, dt)
        mu_M = mu[:, motif_mask, :]
        x_t_1_m = motif_forward_diffusion[i-1][None]
        x_t_1_m = self.diffuser.se3_diffuser._r3_diffuser._scale(x_t_1_m)

        # compute un-normalized weighting factor for importance resampling step
        log_w = -(1./2)*(x_t_1_m-mu_M)**2/(std**2)
        log_w = torch.sum(log_w, axis=[-2, -1])
        log_w -= torch.logsumexp(log_w, 0)
        return log_w

    def forward_traj(self, x_0, min_t, num_t):
        forward_steps = np.linspace(min_t, 1.0, num_t)[:-1]
        x_traj = [x_0]
        for t in forward_steps:
            x_t = self.diffuser.se3_diffuser._r3_diffuser.forward(
                x_traj[-1], t, num_t)
            x_traj.append(x_t)
        x_traj = torch.stack(x_traj, axis=0)
        return x_traj

    def residual_resample(self, weights):
        """residual_resample samples from discrete distribution with probabilities `weights'
        trying to maintain diversity.

        Args:
            weights: simplex variable weights of shape [B]

        Returns:
            idcs of samples
        """
        B = len(weights)
        weights *= B/sum(weights)

        weights_floor = np.floor(weights)
        weights_remainder = weights - weights_floor
        idcs_no_replace = sum([[i]*int(w) for i, w in enumerate(weights_floor)], [])

        N_replace = sum(weights_remainder)
        N_replace = int(np.round(N_replace))
        idcs_replace = np.random.choice(B, size=N_replace, p=weights_remainder/sum(weights_remainder))
        idcs = idcs_no_replace + list(idcs_replace)
        return idcs, N_replace

    def smc_step(
            self,
            input_feats,
            model_out,
            i,
            t,
            dt,
            motif_traj,
            motif_mask,
            weights,
            greedy=False
        ):
        trans_score = model_out['trans_score']
        rigids_t = input_feats['rigids_t']
        n_samples = trans_score.shape[0] 
        trans_t = rigids_t[..., 4:]
        log_w = self.log_importance_weights(
            trans_t, trans_score, i, t, dt, motif_traj, motif_mask)
        # Update Self-normalized importance weights
        weights = weights*torch.exp(log_w).cpu().detach().numpy()
        # TODO: Check that this isn't a bug
        weights += 1e-3
        if greedy:
            max_weight = np.max(weights)
            weights = np.zeros_like(weights)
            weights[np.argmax(weights)] = max_weight
        weights /= np.sum(weights) # Re-normalize

        # Residual resample, but only if
        #   (1) weights are sufficiently non-uniform, and
        #   (2)(optionally) not too close to end of the trajectory
        departure_from_uniform = np.sum(abs(n_samples*weights-1))
        # if (departure_from_uniform > 0.75*n_sample) and t > self.n_T//10:
        if departure_from_uniform > 0.75*n_samples:

            # print(t, "resampling, departure=%0.02f"%departure_from_uniform)
            idcs, _ = self.residual_resample(weights + 0.01)

            # Apply resampling
            trans_t, trans_score = trans_t[idcs].to(trans_score.device), trans_score[idcs].to(trans_score.device)
            rigids_t[..., 4:] = trans_t
            # [BUG] TODO: Should also be updating the rotations.
            # [BUG] TODO: Should also be updating the self-conditioning.

        # Reset weights to uniform
        weights = np.ones_like(weights)/n_samples
        return rigids_t, trans_score, weights

    def smc_inference_fn(
            self,
            data_init,
            ode=False,
            num_t=None,
            min_t=None,
            aux_traj=True
        ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
            ode: set true to use probability flow ODE instead of stochastic dynamics
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats['rigids_t'].device
        if sample_feats['rigids_t'].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones(
                (sample_feats['rigids_t'].shape[0],)).to(device)
        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1/num_t

        # Forward diffuse motif
        motif_mask = sample_feats['fixed_mask'][0].bool()
        bb_pos = sample_feats['rigids_t'][0, :, 4:]
        forward_traj = self.forward_traj(bb_pos, min_t, num_t)
        motif_traj = forward_traj[:, motif_mask]
        sample_feats['fixed_mask'] = torch.zeros_like(
            sample_feats['fixed_mask']).to(motif_mask.device)

        initial_aatype_probs = du.move_to_np(copy.deepcopy(sample_feats['aatype_probs_t']))
        np_restypes = np.array(residue_constants.restypes_with_x)
        probs_to_seq = lambda x: np_restypes[np.argmax(x, axis=-1)]
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats['rigids_t']))[:1]]
        all_aatype_probs = [initial_aatype_probs[:1]]
        all_seqs = [probs_to_seq(initial_aatype_probs)[:1]]
        all_bb_prots = []
        all_trans_0_pred = []
        all_aatype_0_pred = []
        all_bb_0_pred = []
        prev_t = reverse_steps[0]
        weights = np.ones([sample_feats['fixed_mask'].shape[0]])
        with torch.no_grad():
            for i,t in enumerate(reverse_steps):
                i = num_t - i - 1
                motif_t = motif_traj[i]
                sample_feats['rigids_t'][:, motif_mask, 4:] = motif_t.float()
                if self._model_conf.embed.embed_self_conditioning:
                    sample_feats = self._set_t_feats(sample_feats, prev_t, t_placeholder)
                    # [BUG] TODO: Do this update with the model outputs at end of each iteration.
                    sample_feats = self._self_conditioning(sample_feats)
                    prev_t = t
                if t > min_t:
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                    model_out = self.model(sample_feats)

                    # SMCDiff
                    greedy = True if i == 1 else False
                    rigids_t, trans_score, weights = self.smc_step(
                        sample_feats, model_out, i, t, dt,
                        motif_traj, motif_mask, weights, greedy=greedy
                    )

                    rot_score = model_out['rot_score']
                    aatype_pred = model_out['aatype_probs']
                    rigid_pred = model_out['rigids']

                    fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                    diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                    rigids_t, aatype_sde_t, aatype_t = self.diffuser.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(rigids_t),
                        rot_score=du.move_to_np(rot_score),
                        trans_score=du.move_to_np(trans_score),
                        aatype_sde_t=du.move_to_np(sample_feats['aatype_sde_t']),
                        aatype_probs_0=du.move_to_np(aatype_pred),
                        diffuse_mask=du.move_to_np(diffuse_mask),
                        t=t,
                        dt=dt,
                        center=True,
                        ode=ode,
                    )
                    aatype_t = torch.tensor(aatype_t).to(device)
                    aatype_sde_t = torch.tensor(aatype_sde_t).to(device)
                else:
                    model_out = self.model(sample_feats)
                    aatype_t = model_out['aatype_probs']
                    rigids_t = ru.Rigid.from_tensor_7(model_out['rigids'])

                sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
                sample_feats['aatype_sde_t'] = aatype_sde_t
                sample_feats['aatype_probs_t'] = aatype_t
                if aux_traj:
                    all_rigids.append(du.move_to_np(rigids_t.to_tensor_7())[:1])
                    all_aatype_0_pred.append(du.move_to_np(aatype_pred)[:1])
                np_aatype_prob = du.move_to_np(aatype_t)
                all_aatype_probs.append(np_aatype_prob[:1])
                all_seqs.append(probs_to_seq(np_aatype_prob)[:1])

                # Calculate x0 prediction derived from score predictions.
                gt_trans_0 = sample_feats['rigids_t'][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0
                psi_pred = model_out['psi']
                if aux_traj:
                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred),
                        psi_pred
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0)[:1])
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0)[:1])
                atom37_t = all_atom.compute_backbone(
                    rigids_t, psi_pred)[0]
                all_bb_prots.append(
                    du.move_to_np(atom37_t)[:1]
                )

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        all_aatype_probs = flip(all_aatype_probs)
        all_seqs = flip(all_seqs)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_aatype_0_pred = flip(all_aatype_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            'prot_traj': all_bb_prots,
            'aatype_probs_traj': all_aatype_probs,
            'seq_traj': all_seqs
        }
        if aux_traj:
            ret['rigid_traj'] = all_rigids
            ret['trans_traj'] = all_trans_0_pred
            ret['aatype_pred_traj'] = all_aatype_0_pred
            ret['psi_pred'] = psi_pred[None]
            ret['rigid_0_traj'] = all_bb_0_pred
        return ret

    def inference_fn(
            self,
            data_init,
            num_t=None,
            min_t=None,
            center=True,
            ode=False,
            aux_traj=False,
            self_condition=True,
            noise_scale=1.0,
        ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
            ode: set true to use probability flow ODE instead of stochastic dynamics
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats['rigids_t'].device
        if sample_feats['rigids_t'].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones(
                (sample_feats['rigids_t'].shape[0],)).to(device)
        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1/num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats['rigids_t']))]
        initial_aatype_probs = du.move_to_np(copy.deepcopy(sample_feats['aatype_probs_t']))
        np_restypes = np.array(residue_constants.restypes_with_x)
        probs_to_seq = lambda x: np_restypes[np.argmax(x, axis=-1)]
        all_aatype_probs = [initial_aatype_probs]
        all_seqs = [probs_to_seq(initial_aatype_probs)]
        all_bb_prots = []
        all_trans_0_pred = []
        all_aatype_0_pred = []
        all_bb_0_pred = []
        with torch.no_grad():
            if self._model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats = self._set_t_feats(
                    sample_feats, reverse_steps[0], t_placeholder)
                sample_feats = self._self_conditioning(sample_feats)
            for t in reverse_steps:
                if t > min_t:
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                    model_out = self.model(sample_feats)
                    rot_score = model_out['rot_score']
                    trans_score = model_out['trans_score']
                    aatype_pred = model_out['aatype_probs']
                    rigid_pred = model_out['rigids']
                    if self._model_conf.embed.embed_self_conditioning:
                        sample_feats['sc_aatype_probs_t'] = aatype_pred
                        sample_feats['sc_ca_t'] = rigid_pred[..., 4:]
                    fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                    diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                    rigids_t, aatype_sde_t, aatype_t = self.diffuser.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
                        rot_score=du.move_to_np(rot_score),
                        trans_score=du.move_to_np(trans_score),
                        aatype_sde_t=du.move_to_np(sample_feats['aatype_sde_t']),
                        aatype_probs_0=du.move_to_np(aatype_pred),
                        diffuse_mask=du.move_to_np(diffuse_mask),
                        t=t,
                        dt=dt,
                        center=center,
                        ode=ode,
                        noise_scale=noise_scale,
                    )
                    aatype_t = torch.tensor(aatype_t).to(device)
                    aatype_sde_t = torch.tensor(aatype_sde_t).to(device)
                else:
                    model_out = self.model(sample_feats)
                    aatype_t = model_out['aatype_probs']
                    rigids_t = ru.Rigid.from_tensor_7(model_out['rigids'])

                sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
                sample_feats['aatype_sde_t'] = aatype_sde_t
                sample_feats['aatype_probs_t'] = aatype_t
                if aux_traj:
                    all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))
                    all_aatype_0_pred.append(du.move_to_np(aatype_pred))
                np_aatype_prob = du.move_to_np(aatype_t)
                all_aatype_probs.append(np_aatype_prob)
                all_seqs.append(probs_to_seq(np_aatype_prob))

                # Calculate x0 prediction derived from score predictions.
                gt_trans_0 = sample_feats['rigids_t'][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0
                psi_pred = model_out['psi']
                if aux_traj:
                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred),
                        psi_pred
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                atom37_t = all_atom.compute_backbone(
                    rigids_t, psi_pred)[0]
                all_bb_prots.append(
                    du.move_to_np(atom37_t)
                )

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        all_aatype_probs = flip(all_aatype_probs)
        all_seqs = flip(all_seqs)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_aatype_0_pred = flip(all_aatype_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            'prot_traj': all_bb_prots,
            'aatype_probs_traj': all_aatype_probs,
            'seq_traj': all_seqs
        }
        if aux_traj:
            ret['rigid_traj'] = all_rigids
            ret['trans_traj'] = all_trans_0_pred
            ret['aatype_pred_traj'] = all_aatype_0_pred
            ret['psi_pred'] = psi_pred[None]
            ret['rigid_0_traj'] = all_bb_0_pred
        return ret


@hydra.main(version_base=None, config_path="../config", config_name="base")
def run(conf: DictConfig) -> None:

    # Set random seed
    # torch.manual_seed(123123)
    # np.random.seed(123123)

    # multinode requires this set in submit script
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(conf.experiment.port)

    # Fixes bug in https://github.com/wandb/wandb/issues/1525
    os.environ["WANDB_START_METHOD"] = "thread"

    exp = Experiment(conf=conf)
    exp.start_training()


if __name__ == '__main__':
    run()
