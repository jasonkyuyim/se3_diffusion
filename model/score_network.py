import torch
import math
from torch import nn
from torch.nn import functional as F
from data import utils as du
from data import all_atom
from model import ipa_pytorch
from openfold.np import residue_constants
from openfold.utils import rigid_utils as ru
import functools as fn

Tensor = torch.Tensor


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2).to(indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Embedder(nn.Module):

    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size
        if self._embed_conf.mixed_t:
            t_embed_size += index_embed_size
        node_embed_dims = t_embed_size + 1
        edge_in = (t_embed_size + 1) * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size
        edge_in += index_embed_size

        # Chain index embedding
        if self._embed_conf.embed_chain_idx:
            node_embed_dims += index_embed_size
            edge_in += index_embed_size*2

        if self._embed_conf.embed_aatype:
            aatype_embed_size = self._embed_conf.aatype_embed_size
            self.aatype_embedder = nn.Sequential(
                nn.Linear(residue_constants.restype_num+1, aatype_embed_size),
                nn.ReLU(),
                nn.Linear(aatype_embed_size, aatype_embed_size),
                nn.LayerNorm(aatype_embed_size),
            )
            node_embed_dims += aatype_embed_size
            edge_in += aatype_embed_size * 2
            if self._embed_conf.embed_self_conditioning:
                node_embed_dims += aatype_embed_size
                edge_in += aatype_embed_size * 2
        else:
            aatype_embed_size = 0

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        if self._embed_conf.embed_self_conditioning:
            edge_in += self._embed_conf.num_bins
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(
            self,
            *,
            seq_idx,
            chain_idx,
            t_seq,
            t_struct,
            aatype,
            fixed_mask,
            self_conditioning_aatype,
            self_conditioning_ca,
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_aatype: [..., N, 21] aatype probabilities of
                self-conditioning input.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_res = seq_idx.shape

        node_feats = []

        # Timestep features.
        t_seq_embed = torch.tile(
            self.timestep_embedder(t_seq)[:, None, :], (1, num_res, 1))

        # Set time step to epsilon=1e-10 for fixed residues.
        fixed_t_embed = self.timestep_embedder(
            torch.ones_like(t_seq)*1e-10)

        if self._embed_conf.mixed_t:
            t_struct_embed = torch.tile(
                self.timestep_embedder(t_struct)[:, None, :], (1, num_res, 1))
            t_embed = torch.cat([t_seq_embed, t_struct_embed], dim=-1)
            fixed_t_embed = torch.tile(fixed_t_embed[:, None, :], (1, num_res, 2))
        else:
            fixed_t_embed = torch.tile(fixed_t_embed[:, None, :], (1, num_res, 1))
            t_embed = t_seq_embed

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = (
            t_embed * (1 - fixed_mask)
            + fixed_t_embed * fixed_mask
        )
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        if self._embed_conf.embed_chain_idx:
            chain_feats = self.index_embedder(chain_idx)
            node_feats.append(chain_feats)
            pair_feats.append(
                self._cross_concat(chain_feats, num_batch, num_res))

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        if self._embed_conf.embed_chain_idx:
            # Only embed seq offsets within chains.
            rel_seq_offset *= chain_idx[:, :, None] == chain_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])
        pair_feats.append(self.index_embedder(rel_seq_offset))

        # Aatype features.
        if self._embed_conf.embed_aatype:
            aatype_embed = self.aatype_embedder(aatype)
            node_feats.append(aatype_embed)
            pair_feats.append(self._cross_concat(
                aatype_embed, num_batch, num_res))

            if self._embed_conf.embed_self_conditioning:
                aatype_embed = self.aatype_embedder(self_conditioning_aatype)
                node_feats.append(aatype_embed)
                pair_feats.append(self._cross_concat(
                    aatype_embed, num_batch, num_res))

        # Self-conditioning distogram.
        if self._embed_conf.embed_self_conditioning:
            sc_dgram = du.calc_distogram(
                self_conditioning_ca,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        return node_embed, edge_embed


class ScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser

        if self._model_conf.network_type == 'ipa':
            self.score_model = ipa_pytorch.IpaScore(model_conf, diffuser)
        else:
            raise ValueError(
                f'Unrecognized network {self._model_conf.network_type}')

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats):
        """forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """

        # Frames as [batch, res, 7] tensors.
        bb_mask = input_feats['res_mask'].type(torch.float32)  # [B, N]
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        # Ensure unknown aatypes are set to correct token.
        pad_aatype = torch.eye(residue_constants.restype_num + 1)[-1][None]
        aatype_t = (
            input_feats['aatype_probs_t'] * bb_mask[..., None]
            + pad_aatype[:, None, :].to(bb_mask.device) * (1 - bb_mask[..., None])
        ).type(torch.float32)

        # Initial embeddings of positonal and relative indices.
        init_node_embed, init_edge_embed = self.embedding_layer(
            seq_idx=input_feats['seq_idx'],
            chain_idx=input_feats['chain_idx'],
            t_seq=input_feats['t_seq'],
            t_struct=input_feats['t_struct'],
            aatype=aatype_t,
            fixed_mask=fixed_mask,
            self_conditioning_aatype=input_feats['sc_aatype_probs_t'].type(torch.float32),
            self_conditioning_ca=input_feats['sc_ca_t'],
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        # Run main network
        model_out = self.score_model(node_embed, edge_embed, input_feats)

        # Logits are of shape [..., 20] where 20 is the number of aatypes.
        if self._model_conf.aatype_prediction:
            aatype_logits = model_out['aatype']
            # Probs are of shape [..., 21] where 21 is the vocab size.
            # Last token is padding that we set to 0.
            aatype_probs = torch.nn.functional.softmax(aatype_logits, dim=-1)
        else:
            aatype_logits = input_feats['aatype_probs_t'][..., :-1]
            aatype_probs = input_feats['aatype_probs_t'][..., :-1]

        aatype_probs = torch.cat([
            aatype_probs,
            torch.zeros(aatype_probs.shape[:-1] + (1,)).to(
                aatype_probs.device)
        ], dim=-1)
        aatype_probs = self._apply_mask(
            aatype_probs, input_feats['aatype_probs_0'], 1 - fixed_mask[..., None])

        gt_psi = input_feats['torsion_angles_sin_cos'][..., 2, :]
        psi_pred = self._apply_mask(
            model_out['psi'], gt_psi, 1 - fixed_mask[..., None])

        pred_out = {
            'psi': psi_pred,
            'rot_score': model_out['rot_score'],
            'trans_score': model_out['trans_score'],
            'aatype_logits': aatype_logits,
            'aatype_probs': aatype_probs,
        }
        if self._model_conf.rigid_prediction:
            rigids_pred = model_out['final_rigids']
            pred_out['rigids'] = rigids_pred.to_tensor_7()
            bb_representations = all_atom.compute_backbone(
                rigids_pred, psi_pred)
            pred_out['atom37'] = bb_representations[0].to(rigids_pred.device)
            pred_out['atom14'] = bb_representations[-1].to(rigids_pred.device)
        return pred_out
