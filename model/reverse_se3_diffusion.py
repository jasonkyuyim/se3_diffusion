import torch
import math
from torch import nn
from torch.nn import functional as F
from data import utils as du
from model import ipa_pytorch
from model import frame_gemnet
from openfold.np import residue_constants
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
        node_embed_dims = index_embed_size
        edge_in = index_embed_size * 2

        # Sequence index embedding
        if self._embed_conf.use_res_idx_encoding:
            node_embed_dims += index_embed_size
        edge_in += index_embed_size

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

        if self._embed_conf.embed_distogram:
            edge_in += self._embed_conf.num_bins
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
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

    def forward(
            self,
            *,
            seq_idx,
            t,
            aatype,
            fixed_mask,
            ca_pos
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_res = seq_idx.shape

        init_node_embed = []

        # Embed timestep.
        t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_t_embed = self.timestep_embedder(torch.ones_like(t)*1e-5)
        fixed_t_embed = torch.tile(fixed_t_embed[:, None, :], (1, num_res, 1))
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = (
            t_embed * (1 - fixed_mask)
            + fixed_t_embed * fixed_mask
        )
        init_node_embed.append(prot_t_embed)

        # Embed 1D sequence features.
        if self._embed_conf.use_res_idx_encoding:
            init_node_embed.append(self.index_embedder(seq_idx))
        if self._embed_conf.embed_aatype:
            aatype_embed = self.aatype_embedder(aatype)
            init_node_embed.append(aatype_embed)

        node_embed = self.node_embedder(
            torch.cat(init_node_embed, dim=-1).float())

        # Embed 2D sequence features.
        edge_attr = seq_idx[:, :, None] - seq_idx[:, None, :]
        edge_attr = edge_attr.reshape([num_batch, num_res**2])
        edge_embed = self.index_embedder(edge_attr)
        cross_t_embed = torch.cat([
            torch.tile(prot_t_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(prot_t_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])
        pair_feats = [
            edge_embed,
            cross_t_embed,
        ]
        if self._embed_conf.embed_aatype:
            cross_aatype = torch.cat([
                torch.tile(aatype_embed[:, :, None, :], (1, 1, num_res, 1)),
                torch.tile(aatype_embed[:, None, :, :], (1, num_res, 1, 1)),
            ], dim=-1).float()
            pair_feats.append(cross_aatype.reshape(
                [num_batch, num_res**2, -1]))
        if self._embed_conf.embed_distogram:
            dgram = du.calc_distogram(
                ca_pos,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            pair_feats.append(dgram.reshape([num_batch, num_res**2, -1]))
        edge_embed = torch.cat(pair_feats, dim=-1).float()
        edge_embed = self.edge_embedder(edge_embed)
        edge_embed = edge_embed.reshape(
            [num_batch, num_res, num_res, -1])
        return node_embed, edge_embed


class ReverseDiffusion(nn.Module):

    def __init__(self, model_conf):
        super(ReverseDiffusion, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)

        if self._model_conf.network_type == 'ipa':
            self.score_model = ipa_pytorch.IpaScore(model_conf)
        else:
            raise ValueError(
                f'Unrecognized network {self._model_conf.network_type}')

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def _calc_trans_0(self, trans_score, trans_t, beta_t):
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_score * cond_var + trans_t) / torch.exp(-1/2*beta_t)

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

        # Padding needs to be unknown aatypes.
        pad_aatype = torch.eye(residue_constants.restype_num + 1)[-1][None]
        aatype_t = (
            input_feats['aatype_t'] * bb_mask[..., None]
            + pad_aatype[:, None, :].to(bb_mask.device) * (1 - bb_mask[..., None])
        ).type(torch.float32)

        # Initial embeddings of positonal and relative indices.
        init_node_embed, init_edge_embed = self.embedding_layer(
            seq_idx=input_feats['seq_idx'],
            t=input_feats['t'],
            aatype=aatype_t,
            fixed_mask=fixed_mask,
            ca_pos=input_feats['rigids_t'][..., 4:],
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        # Run main network
        model_out = self.score_model(node_embed, edge_embed, input_feats)

        # Rescale score predictions by the standard deviations or variances.
        trans_score = model_out['trans_score'] * input_feats['trans_score_scaling'][:, None, None]
        rot_score = model_out['rot_score'] * input_feats['rot_score_scaling'][:, None, None]

        # Logits are of shape [..., 20] where 20 is the number of aatypes.
        if self._model_conf.aatype_prediction:
            aatype_logits = model_out['aatype']
            # Probs are of shape [..., 21] where 21 is the vocab size.
            # Last token is padding that we set to 0.
            aatype_probs = torch.nn.functional.softmax(aatype_logits, dim=-1)
        else:
            aatype_logits = input_feats['aatype_t'][..., :-1]
            aatype_probs = input_feats['aatype_t'][..., :-1]

        aatype_probs = torch.cat([
            aatype_probs,
            torch.zeros(aatype_probs.shape[:-1] + (1,)).to(
                aatype_probs.device)
        ], dim=-1)
        aatype_probs = self._apply_mask(
            aatype_probs, input_feats['aatype_0'], 1 - fixed_mask[..., None])

        pred_out = {
            'psi': model_out['psi'],
            'rot_score': rot_score,
            'trans_score': trans_score,
            'aatype_logits': aatype_logits,
            'aatype_probs': aatype_probs,
        }
        if self._model_conf.direct_prediction:
            raise ValueError('Make compatible with masking')
            pred_out['final_rigids'] = model_out['final_rigids']
            pred_out['rigids_update'] = model_out['rigids_update']

        if self._model_conf.dgram_prediction:
            pred_out['dgram'] = model_out['dgram']
        return pred_out
