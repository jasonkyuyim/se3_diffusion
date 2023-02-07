"""Protein diffusion data loader.

WARNING: Be future-proof. Imagine we'll be working with complexes and trying
pre-trained RF at some point.

"""
from dateutil import parser
import os
import tree
from tqdm import tqdm
import pandas as pd
import numpy as np
import functools as fn
import torch
import logging
import random
from torch.utils import data

from data import residue_constants
from data import utils as du
from functools import cached_property


def _parse_clusters(df):
    """Process clusters to calculate their size, weights, and members."""
    processed_clusters = []
    for cluster_id, cluster_df in tqdm(df.groupby('CLUSTER')):
        num_seqs = len(cluster_df)
        avg_len = cluster_df.seq_len.sum() // num_seqs
        weight = (1 / 512.) * max(min(float(avg_len), 512.), 256.)
        membership = cluster_df.CHAINID.tolist()
        processed_clusters.append((
            cluster_id,
            num_seqs,
            avg_len,
            weight,
            membership
        ))
    return processed_clusters


class DistilledDataset(data.Dataset):
    def __init__(self,
            *,
            data_conf,
            diffuser,
            is_training,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._data_conf = data_conf
        self._filter_conf = data_conf.filtering
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

    @cached_property
    def cluster_weights(self):
        pdb_weights = [x[-2] for x in self._pdb_clusters]
        fb_weights = [x[-2] for x in self._fb_clusters]
        return torch.tensor(pdb_weights), torch.tensor(fb_weights)

    @cached_property
    def valid_clusters(self):
        return set(
            [
                int(l)
                for l in open(self.data_conf.valid_clusters).readlines()
            ]
        )

    @cached_property
    def lookup(self):
        """hash table lookup for chain id to dataframe row number."""
        return dict(
            zip(
                self._train_csv.CHAINID.tolist(),
                range(len(self._train_csv))
            )
        )
        
    @property
    def num_examples(self):
        if self.is_training:
            return len(self._pdb_clusters) + len(self._fb_clusters)
        else:
            return len(self._valid_pdb_csv)

    def __len__(self):
        return self.num_examples

    def _init_metadata(self):
        """Initialize metadata."""
        loaded_data = False
        cache_path = os.path.join(
            self.data_conf.cache_dir, 
            'dataset_' + '_'.join(
                [str(x) for x in self._filter_conf.values()])+'.pkl'
        )
        if os.path.exists(cache_path):
            (
                loaded_data_conf,
                train_csv,
                pdb_clusters,
                fb_clusters,
                valid_pdb_csv,
            ) = du.read_pkl(cache_path)
            if not du.compare_conf(
                    self._filter_conf, loaded_data_conf.filtering):
                self._log.info(
                    f'Invalid conf in cached data {cache_path}')
            else:
                loaded_data = True
                self._log.info(f'Loaded data at {cache_path}')

        if not loaded_data:
            self._log.info('Processing metadata.')
            
            # Parse CSVs
            pdb_csv = self._parse_pdb_csv()
            fb_csv = self._parse_fb_csv()
            if self._filter_conf.subset is not None:
                pdb_csv = pdb_csv.iloc[:self._filter_conf.subset]
                fb_csv = fb_csv.iloc[:self._filter_conf.subset]
            train_pdb_csv = pdb_csv[pdb_csv.split == 'train']
            valid_pdb_csv = pdb_csv[pdb_csv.split == 'valid']
            train_fb_csv = fb_csv[fb_csv.split == 'train']

            # Parse train clusters
            train_csv = pd.concat([train_pdb_csv, train_fb_csv])
            pdb_clusters = _parse_clusters(train_pdb_csv)
            fb_clusters = _parse_clusters(train_fb_csv)
            cache_data = (
                self.data_conf,
                train_csv,
                pdb_clusters,
                fb_clusters,
                valid_pdb_csv,
            )
            du.write_pkl(cache_path, cache_data, create_dir=True)
            self._log.info(f'Cached data to {cache_path}')

        self._train_csv = train_csv
        self._pdb_clusters = pdb_clusters
        self._fb_clusters = fb_clusters
        self._valid_pdb_csv = valid_pdb_csv

    def _parse_pdb_csv(self):
        """Parse and filter PDB metadata csv."""
        raw_pdb_csv = pd.read_csv(
            self.data_conf.pdb_csv)
        raw_pdb_csv['seq_len'] = raw_pdb_csv.SEQUENCE.apply(lambda x: len(x))
        raw_pdb_csv['resolution'] = raw_pdb_csv.RESOLUTION.apply(lambda x: float(x))
        raw_pdb_csv['date'] = raw_pdb_csv.DEPOSITION.apply(lambda x: parser.parse(x))
        raw_pdb_csv['source'] = 'pdb'
        raw_pdb_csv['split'] = raw_pdb_csv.CLUSTER.apply(
                lambda x: 'valid' if x in self.valid_clusters else 'train')
        date_cutoff = parser.parse(self._filter_conf.date_cutoff)
        return raw_pdb_csv[
            (raw_pdb_csv.seq_len <= self._filter_conf.max_len) &
            (self._filter_conf.min_len <= raw_pdb_csv.seq_len) &
            (raw_pdb_csv.resolution <= self._filter_conf.resolution_cutoff) &
            (raw_pdb_csv.date <= date_cutoff)
        ]

    def _parse_fb_csv(self):
        """Parse and filter FB metadata csv."""
        raw_fb_csv = pd.read_csv(
            self.data_conf.fb_csv).rename(
                {'#CHAINID': 'CHAINID'}, axis='columns')
        raw_fb_csv['seq_len'] = raw_fb_csv.SEQUENCE.apply(lambda x: len(x))
        raw_fb_csv['source'] = 'fb'
        raw_fb_csv['split'] = raw_fb_csv.CLUSTER.apply(
                lambda x: 'valid' if x in self.valid_clusters else 'train')
        return raw_fb_csv[
            (raw_fb_csv.plDDT >= self._filter_conf.min_plddt) &
            (raw_fb_csv.seq_len <= self._filter_conf.max_len) &
            (self._filter_conf.min_fb_len <= raw_fb_csv.seq_len)
        ]

    @fn.lru_cache
    def load_pdb(self, chain_id):
        parsed_pdb = torch.load(
            os.path.join(
                self.data_conf.pdb_dir, chain_id[1:3], f'{chain_id}.pt'))
        xyz = parsed_pdb['xyz']
        num_res = len(xyz)
        mask = parsed_pdb['mask'].long()
        res_idx = torch.arange(num_res).long()
        xyz = torch.nan_to_num(xyz).float()
        seq = torch.tensor(
            [
                residue_constants.restype_order_with_x[i]
                for i in parsed_pdb['seq']
            ]).long()
        seq = torch.squeeze(seq)
        num_res = len(seq)
        # TODO: Remove beginning and ending tags
        res_mask = torch.any(mask, axis=-1).long()
        feats = {
            'xyz': xyz,
            'mask': mask,
            'res_mask': res_mask,
            'seq': seq,
            'res_idx': res_idx,
        }
        tree.map_structure(lambda x: len(x) == num_res, feats)
        return chain_id, feats

    @fn.lru_cache
    def load_fb(self, chain_id, chain_hash):
        file_path = os.path.join(
            self.data_conf.fb_dir, chain_hash[:2], chain_hash[2:], chain_id)
        xyz, mask, res_idx, _ = du.parse_pdb(file_path+'.pdb')
        msa_path = os.path.join(
            self.data_conf.fb_msa_dir, chain_hash[:2], chain_hash[2:], chain_id)
        msa, _ = du.parse_a3m(msa_path+'.a3m.gz')
        seq = msa[:1]
        num_res = len(seq)
        plddt = np.load(file_path+'.plddt.npy')

        # TODO: Add back sidechain pLDDT masking
        mask = np.logical_and(mask, (plddt > self.data_conf.min_plddt)[:,None])

        # TODO: Remove beginning and ending tags
        res_mask = np.any(mask, axis=-1)
        feats = {
            'xyz': torch.tensor(xyz).float(),
            'mask': torch.tensor(mask).long(),
            'res_mask': torch.tensor(res_mask).long(),
            'seq': torch.squeeze(torch.tensor(seq)).long(),
            'res_idx': torch.tensor(res_idx).long(),
        }
        tree.map_structure(lambda x: len(x) == num_res, feats)
        return os.path.basename(file_path), feats
    
    def ipa_featurize(self, item_data, t):
        # Perform cropping on raw data.
        if self._data_conf.crop_len is not None:
            item_data = tree.map_structure(
                lambda x: x[:self._data_conf.crop_len] if x.ndim > 0 else x,
                item_data)
        # zero-center
        center = torch.sum(item_data['xyz'][:, 1], dim=0) / torch.sum(item_data['res_mask'])
        centered_pos = item_data['xyz'] - center[None, :]
        centered_pos *= item_data['mask'][..., None]

        rigids_0 = du.rigid_frames_from_atom_14(centered_pos)
        rigids_0 = rigids_0.apply_trans_fn(
            lambda x: x/self._data_conf.scale_factor)
        (
            rigids_t,
            trans_score,
            rot_score,
            trans_score_norm,
            rot_score_norm,
        ) = self.diffuser.forward_marginal(rigids_0, t)
        res_mask = item_data['res_mask'][:, None]
        final_feats = {
            'rigids_0': rigids_0.to_tensor_7() * res_mask,
            'rigids_t': rigids_t.to_tensor_7() * res_mask,
            'trans_score': torch.tensor(trans_score) * res_mask,
            'rot_score': torch.tensor(rot_score) * res_mask,
            'rot_score_norm': torch.tensor(rot_score_norm),
            'trans_score_norm': torch.tensor(trans_score_norm),
            't': torch.tensor(t),
            'res_idx': item_data['res_idx'],
            'res_mask': item_data['res_mask'],
            'xyz': centered_pos, 
        }
        final_feats = du.pad_feats(final_feats, self._filter_conf.max_len)
        return final_feats

    def __getitem__(self, index):
        if self.is_training:
            num_pdb_clusters = len(self._pdb_clusters)
            if index < num_pdb_clusters:
                selected_cluster = self._pdb_clusters[index]
            else:
                selected_cluster = self._fb_clusters[index - num_pdb_clusters]
            item_id = random.sample(selected_cluster[-1], 1)[0]
            item = self._train_csv.iloc[self.lookup[item_id]]
            src = item['source']
            if self._data_conf.t_sampler == 'scheduled':
                ts = np.linspace(1e-3, 1.0, self._data_conf.num_t)
                t = np.random.choice(ts)
            elif self._data_conf.t_sampler == 'uniform': 
                t = random.uniform(1e-3, 1.0)
            else:
                raise ValueError(
                    f'Unrecognized sampler {self._data_conf.t_sampler}')
        else:
            item = self._valid_pdb_csv.iloc[index]
            src = 'pdb'
            t = 1.0

        if src == 'pdb':
            item_id, item_data = self.load_pdb(item.CHAINID)
        elif src == 'fb':
            item_id, item_data = self.load_fb(item.CHAINID, item.HASH)
        else:
            raise ValueError(f'Unrecognized source {src}')

        # TODO: Skip bad examples with too many missing residues.
        # TODO: Batch into graph format. Flag to batch into IPA format.
        return self.ipa_featurize(item_data, t)

class DistributedWeightedSampler(data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            num_replicas,
            rank,
        ):
        self._data_conf = data_conf
        self._dataset = dataset
        self._num_replices = num_replicas
        self._rank = rank
        self._num_examples_per_epoch = self._data_conf.epoch_size
        self._num_fb_per_epoch = int(
            self._num_examples_per_epoch*(self._data_conf.fraction_fb))
        self._num_pdb_per_epoch = (
            self._num_examples_per_epoch - self._num_fb_per_epoch
        )
        self._num_samples_per_replica = (
            self._num_examples_per_epoch // self._num_replices
        )
        self._pdb_weights, self._fb_weights = self._dataset.cluster_weights
        self.epoch = 0
        self._num_pdb_samples = min(self._num_pdb_per_epoch, len(self._pdb_weights))

    def _index_tile(self, example_indices):
        """Compute structured cartesian product of example and time indices.
        
        E.g. 
            example_indices=[0, 1]
            num_t=3
            first tile example indices: [0, 0, 0, 2, 2, 2]
            multiple by num_t: [0, 0, 0, 6, 6, 6]
            add tiled t indices [0, 1, 2, 6, 7, 8].
            
            A sampled index i will correspond to example (i // num_t)
            and t = i % num_t.
        """
        num_t = self._dataset.diffuser.num_t
        tiled_example_indices = torch.reshape(
            torch.tile(
                example_indices[:, None], (1, num_t)),
            (-1,)) * num_t
        tiled_t_indices = torch.tile(
            torch.arange(num_t), (len(example_indices),))
        assert tiled_example_indices.shape == tiled_t_indices.shape
        return tiled_example_indices + tiled_t_indices

    def __iter__(self):
        g = torch.Generator()
        # Each replica has a unique seed for each epoch.
        g.manual_seed(self.epoch * self._num_replices + self._rank)

        # Sample indices
        pdb_sampled = torch.multinomial(
            self._pdb_weights,
            self._num_pdb_samples,
            False,
            generator=g).repeat(self._data_conf.num_t)
        # pdb_sampled = self._index_tile(pdb_sampled)
        
        if self._data_conf.fraction_fb > 1e-5:
            fb_sampled = torch.multinomial(
                self._fb_weights,
                self._num_fb_per_epoch,
                False,
                generator=g).repeat(self._data_conf.num_t)
            # fb_sampled = self._index_tile(fb_sampled)

            # Offset FB examples with number of PDB examples.
            fb_sampled += len(self._pdb_weights)
            sampled_indices = torch.cat([pdb_sampled, fb_sampled])
        else:
            sampled_indices = pdb_sampled
        indices = sampled_indices[
            torch.randperm(len(sampled_indices), generator=g)]

        # assert torch.unique(indices).shape == indices.shape
        return iter(indices.tolist())

    def __len__(self):
        return self._num_samples_per_replica

    def set_epoch(self, epoch):
        self.epoch = epoch
