"""Pytorch script for training SE(3) protein diffusion.

Instructions:

By default hydra will launch with `config/base.yaml` settings.
Specify different config with the `-cn <config_name>` flag excluding .yaml,
i.e. `-cn subset_1000`.

To run:

> python experiments/train_se3_diffusion.py

Without Wandb,

> python experiments/train_se3_diffusion.py experiment.use_wandb=False

To modify config options with the command line,

> python experiments/train_se3_diffusion.py experiment.batch_size=32

Use tmux to start experiments and have them continue running.
Another option is to send the experiments into the background.

> python experiments/train_se3_diffusion.py &

Hydra saves all logging into outputs/<date>/<time>/<train_se3_diffusion.log
as well as the configs used to run the experiment in the same directory.

Multi-run can be achieved with the `-m` flag. The config must specify the sweep.
For an example, see `psi_sweep.yaml`.

> python experiments/train_se3_diffusion.py -cn psi_sweep -m

"""

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
from datetime import datetime
from omegaconf import DictConfig
from omegaconf import OmegaConf 
import pandas as pd
import torch.distributed as dist
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP
from openfold.utils import rigid_utils as ru
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler, TestAF2LRScheduler
from hydra.core.hydra_config import HydraConfig

from analysis import utils as au
from analysis import metrics
from data import digs_data_loader
from data import rosetta_data_loader
from data import se3_diffuser
from data import utils as du
from data import all_atom
from model import reverse_se3_diffusion
from model import reverse_diffusion_SE3_tfmr
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
        self.data_mode = self._exp_conf.data_location
        if self.data_mode == 'rosetta':
            self._data_conf = self.conf.data.rosetta
        elif self.data_mode == 'digs':
            self._data_conf = self.conf.data.digs
        else:
            raise ValueError(
                f'Unrecognize data location {self.data_mode}')
        self._dist_mode = self._exp_conf.dist_mode
        self._use_wandb = self._exp_conf.use_wandb
        
        # Initialize experiment objects
        self.trained_epochs = 0
        self.trained_steps = 0
        if self._model_conf.network_type == 'ipa':
            self._model = reverse_se3_diffusion.ReverseDiffusion(
                self._model_conf)
        elif self._model_conf.network_type == 'se3_tfmr':
            self._model = reverse_diffusion_SE3_tfmr.ReverseDiffusionSE3TFMR(
                self._model_conf)
        else:
            raise ValueError(
                f'Unrecognized network: {self._model_conf.network_type}')
        if ckpt_model is not None:
            self._model.load_state_dict(ckpt_model)
        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._log.info(f'Number of model parameters {num_parameters}')
        self._diffuser = se3_diffuser.SE3Diffuser(**self._diff_conf)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._exp_conf.learning_rate)
        
        self._scheduler_conf = conf.openfold_lr_scheduler
        self._scheduler = AlphaFoldLRScheduler(self._optimizer, **self._scheduler_conf) 

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
        else:
            self._log.info('Checkpoint not being saved.')

        # Set-up evaluation save location to match ckpt_dir
        eval_dir = os.path.join(
            self._exp_conf.eval_dir,
            self._exp_conf.name,
            dt_string)
        self._exp_conf.eval_dir = eval_dir
        self._log.info(f'Evaluation saved to: {eval_dir}')

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
    
    def create_rosetta_dataset(self, replica_id, num_replicas):

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
        
        # Loaders
        num_workers = self._exp_conf.num_loader_workers
        train_loader = du.create_data_loader(
            train_dataset,
            np_collate=False,
            batch_size=self._exp_conf.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        valid_loader = du.create_data_loader(
            valid_dataset,
            np_collate=False,
            batch_size=self._exp_conf.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        return train_loader, valid_loader

    def start_training(self, return_logs=False):
        """Start training based on distribution strategy."""
        
        if self._dist_mode == 'slurm':
            num_replicas = int(os.environ["SLURM_NTASKS"])
            replica_id = int(os.environ["SLURM_PROCID"])
            self._log.info(f"Training replica: {replica_id}/{num_replicas}")
            self.train(replica_id, num_replicas)
        elif self._dist_mode == 'multi':            
            num_replicas = torch.cuda.device_count()
            self._log.info("Training distributed.")
            torch.multiprocessing.spawn(
                self.train,
                args=(num_replicas,),
                nprocs=num_replicas,
                join=True
            )
        elif self._dist_mode == 'single':
            self._log.info("Training single process.")
            # Set environment variables for which GPUs to use.
            available_gpus = ''.join(
                [str(x) for x in GPUtil.getAvailable(
                    order='memory', limit = 8)])
            if HydraConfig.initialized() and 'num' in HydraConfig.get().job: 
                chosen_gpu = available_gpus[HydraConfig.get().job.num] 
            else:
                chosen_gpu = available_gpus[0]
            self._log.info(f"Using GPU: {chosen_gpu}")

            # Return logs for training single processed.
            return self.train(
                int(chosen_gpu), 1, return_logs=return_logs, init_wandb=True)
        else:
            raise ValueError(
                f'Unrecognized distribution mode: {self._dist_mode}')

    def init_wandb(self):
        self._log.info('Initializing Wandb.')
        conf_dict = OmegaConf.to_container(self._conf, resolve=True)
        wandb.init(
            project='protein-diffusion-v2',
            name=self._exp_conf.name,
            config=dict(eu.flatten_dict(conf_dict)),
            dir=self._exp_conf.wandb_dir,
            tags=[
                'experimental',
                self._exp_conf.data_location,
            ],
        )
        self._exp_conf.run_id = wandb.util.generate_id()
        self._exp_conf.wandb_dir = wandb.run.dir
        self._log.info(
            f'Wandb: run_id={self._exp_conf.run_id}, run_dir={self._exp_conf.wandb_dir}')

    def train(self, replica_id, num_replicas, return_logs=False, init_wandb=False):

        if (init_wandb or replica_id == 0) and self._use_wandb: 
            self.init_wandb()

        if self._dist_mode != 'single':
            if replica_id > num_replicas:
                raise ValueError(
                    f'Replica ID {replica_id} greater than world size {num_replicas}')
            dist.init_process_group(
                backend="nccl", world_size=num_replicas, rank=replica_id)
        if torch.cuda.is_available():
            device = f"cuda:{replica_id}"
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        self._model = self._model.to(replica_id)

        if self._dist_mode != 'single':
            self._model = DDP(
                self._model,
                device_ids=[replica_id],
                find_unused_parameters=False)
        self._model.train()

        # Data selector based on server being ran from.
        if self.data_mode == 'digs':
            train_loader, train_sampler, valid_loader, valid_sampler = self.create_digs_dataset(
                replica_id, num_replicas)
        elif self.data_mode == 'rosetta':
            train_loader, valid_loader = self.create_rosetta_dataset(
                replica_id, num_replicas)
            train_sampler = None
            valid_sampler = None
        else:
            raise ValueError(
                f'Unrecognize data location {self.data_mode}')

        logs = []
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)
            if self._dist_mode != 'single':
                with self._model.no_sync():
                    self.train_epoch(train_loader, valid_loader, device)
            else:
                epoch_log = self.train_epoch(
                        train_loader, valid_loader, device, return_logs)
                if return_logs:
                    logs.append(epoch_log)
            self.trained_epochs = epoch

        self._log.info('Done')
        return logs

    def update_fn(self, data):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        loss, aux_data = self.loss_fn(data)
        loss.backward()
        self._optimizer.step()
        self._scheduler.step()
        return loss, aux_data

    def train_epoch(self, train_loader, valid_loader, device, return_logs=False):
        log_lossses = []
        global_logs = []
        log_time = time.time()
        step_time = time.time()
        for train_feats in train_loader:
            train_feats = tree.map_structure(
                lambda x: x.to(device, non_blocking=True), train_feats)
            loss, aux_data = self.update_fn(train_feats)
            if return_logs:
                global_logs.append(loss)
            log_lossses.append(du.move_to_np(loss))
            self.trained_steps += 1

            # Logging to terminal
            if self.trained_steps == 1 or self.trained_steps % self._exp_conf.log_freq == 0:
                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self._exp_conf.log_freq / elapsed_time
                rolling_loss = np.mean(log_lossses)
                self._log.info(
                    f'[{self.trained_steps}]: loss={rolling_loss:.5f}, steps/sec={step_per_sec:.5f}')
                log_lossses = []

            # Take checkpoint
            if self._exp_conf.ckpt_dir is not None and (
                    ((self.trained_steps % self._exp_conf.ckpt_freq) == 0)
                    or (self.trained_steps == 100)
                ):  # Force a checkpoint at step 100 for quick feedback.
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
                ckpt_metrics = self.eval_fn(eval_dir, valid_loader, device)
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
                    'examples_per_sec': example_per_sec,
                    'num_epochs': self.trained_epochs,
                }
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
                    wandb.alert(
                        title="Encountered NaN loss",
                        text=f"Loss NaN after {self.trained_epochs} epochs, {self.trained_steps} steps"
                    )
                    raise Exception(f'NaN encountered')

        if return_logs:
            return global_logs

    def eval_fn(self, eval_dir, valid_loader, device):
        ckpt_eval_metrics = []
        for valid_feats, pdb_names in valid_loader:
            res_mask = du.move_to_np(valid_feats['res_mask'].bool())
            aatype = du.move_to_np(valid_feats['aatype']) 
            gt_prot = du.move_to_np(valid_feats['atom37_pos'])
            batch_size = res_mask.shape[0]
            valid_feats = tree.map_structure(
                lambda x: x.to(device), valid_feats)

            # Run inference
            infer_out = self.inference_fn(
                valid_feats, add_noise=True)
            sampled_bb_prots = infer_out[1]
            final_prot = du.move_to_np(sampled_bb_prots[-1])
            for i in range(batch_size):
                num_res = int(np.sum(res_mask[i]).item())
                unpad_prot = final_prot[i][res_mask[i]]
                unpad_gt_prot = gt_prot[i][res_mask[i]] 
                unpad_aatype = aatype[i][res_mask[i]] 
                saved_path = au.write_prot_to_pdb(
                    unpad_prot,
                    os.path.join(eval_dir, f'len_{num_res}_sample_{i}.pdb'),
                    aatype=unpad_aatype,
                    no_indexing=True)
                try:
                    sample_metrics = metrics.protein_metrics(
                        pdb_path=saved_path,
                        atom37_pos=unpad_prot,
                        gt_atom37_pos=unpad_gt_prot,
                        gt_aatype=unpad_aatype
                    )
                except ValueError as e:
                    self._log.warning(
                        f'Failed evaluation of length {num_res} sample {i}: {e}')
                    continue
                sample_metrics['step'] = self.trained_steps
                sample_metrics['num_res'] = num_res 
                sample_metrics['sample_path'] = saved_path
                sample_metrics['gt_pdb'] = pdb_names[i]

                ckpt_eval_metrics.append(sample_metrics)

        # Save metrics as CSV.
        eval_metrics_csv_path = os.path.join(eval_dir, 'metrics.csv')
        ckpt_eval_metrics = pd.DataFrame(ckpt_eval_metrics)
        ckpt_eval_metrics.to_csv(eval_metrics_csv_path, index=False)
        return ckpt_eval_metrics 

    def loss_fn(self, batch, model_out=None):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """
        if model_out is None:
            model_out = self.model(batch)
        bb_mask = batch['res_mask']
        batch_size = bb_mask.shape[0]

        gt_rot_score = batch['rot_score']
        gt_trans_score = batch['trans_score']
        rot_score_norm = batch['rot_score_norm']
        trans_score_norm = batch['trans_score_norm']
        gt_torsions = batch['torsion_angles_sin_cos']
        gt_psi = gt_torsions[..., 2, :]
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        # Model predictions.
        pred_rot_score = model_out['rot_score']
        pred_trans_score = model_out['trans_score']
        pred_psi = model_out['psi']

        # Rotation loss
        rot_mse = (pred_rot_score - gt_rot_score)**2
        rot_loss = torch.sum(
            rot_mse / rot_score_norm[:, None, None]**2,
            dim=(-1, -2)
        ) / (bb_mask.sum(dim=-1) + 1e-10)
        rot_loss *= self._exp_conf.rot_loss_weight
        rot_loss *= int(self._diff_conf.diffuse_rot)

        # Translation loss
        trans_mse = (pred_trans_score - gt_trans_score)**2
        trans_loss = torch.sum(
            trans_mse / trans_score_norm[:, None, None]**2,
            dim=(-1, -2)
        ) / (bb_mask.sum(dim=-1) + 1e-10)
        trans_loss *= self._exp_conf.trans_loss_weight 
        trans_loss *= int(self._diff_conf.diffuse_trans)

        # Psi loss
        psi_mse = (pred_psi - gt_psi)**2
        psi_loss = torch.sum(
            psi_mse,
            dim=(-1, -2)
        ) / (bb_mask.sum(dim=-1) + 1e-10)
        psi_loss *= self._exp_conf.psi_loss_weight 
        psi_loss *= self._exp_conf.psi_loss_t_filter > batch['t'] 

        final_loss = rot_loss + trans_loss + psi_loss

        def normalize_loss(x):
            return x.sum() /  (batch_loss_mask.sum() + 1e-10)

        aux_data = {
            'total_loss': normalize_loss(final_loss),
            'rot_loss': normalize_loss(rot_loss),
            'trans_loss': normalize_loss(trans_loss),
            'psi_loss': normalize_loss(psi_loss),
        }
        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return normalize_loss(final_loss), aux_data            

    def inference_fn(self, data_init, num_t=None, add_noise=True, center=True):
        """Inference function.
        
        Args:
            data_init: Initial data values for sampling.
            add_noise: Whether to add noise during sampling.
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
        reverse_steps = np.linspace(1e-3, 1.0, num_t)[::-1]
        dt = 1/num_t
        all_rigids = [copy.deepcopy(sample_feats['rigids_t'])]
        all_bb_prots = []
        for t in reverse_steps:
            sample_feats['t'] = t * t_placeholder
            rot_score_norm, trans_score_norm = self.diffuser.exp_score_norm(t)
            sample_feats['rot_score_norm'] = rot_score_norm * t_placeholder
            sample_feats['trans_score_norm'] = trans_score_norm * t_placeholder
            with torch.no_grad():
                model_out = self.model(sample_feats)
            rot_score = model_out['rot_score']
            trans_score = model_out['trans_score']
            rigids_t = self.diffuser.reverse(
                ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
                du.move_to_np(rot_score),
                du.move_to_np(trans_score),
                t,
                dt,
                add_noise=add_noise,
                mask=du.move_to_np(torch.ones_like(sample_feats['res_mask'])),
                center=center,
            )
            sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
            rigids_t = rigids_t.apply_trans_fn(
                lambda x: x * self._data_conf.scale_factor)
            all_rigids.append(rigids_t.to_tensor_7().to(device))
            all_bb_prots.append(
                all_atom.compute_backbone(
                    rigids_t, model_out['psi']
                )[0].to(device)
            )
        return all_rigids, all_bb_prots


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
