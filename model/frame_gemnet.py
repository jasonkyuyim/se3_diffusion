
import torch
import ipdb
import numpy as np
from data import all_atom
from model import basis_utils
from model import layers
from openfold.utils import rigid_utils as ru

Dense = layers.Dense
Linear = layers.Linear


def local_1d_gather(local_graph, feats_1d):
    batch_size, num_res, num_neighbors = local_graph.shape
    flat_local_graph = local_graph.view(batch_size, -1)
    _, res_per_batch = flat_local_graph.shape
    i_idx = torch.repeat_interleave(torch.arange(batch_size), res_per_batch)
    j_idx = flat_local_graph.view(-1)
    local_feats = feats_1d[(i_idx, j_idx)].reshape(
        batch_size, num_res, num_neighbors, feats_1d.shape[-1])
    return local_feats

def local_2d_gather(local_graph, feats_2d):
    batch_size, num_res, num_neighbors = local_graph.shape
    b_idx, i_idx, _ = torch.where(torch.ones_like(local_graph))
    j_idx = local_graph.view(-1)
    local_feats = feats_2d[(b_idx, i_idx, j_idx)].reshape(
        batch_size, num_res, num_neighbors, feats_2d.shape[-1])
    return local_feats


class EdgeEmbedding(torch.nn.Module):

    def __init__(
            self,
            gem_conf
        ):
        super().__init__()
        in_features = 2 * gem_conf.emb_size_node + gem_conf.emb_size_edge + gem_conf.emb_size_rbf
        self.dense_1 = Dense(
            in_features,
            gem_conf.hid_size_edge,
            activation=None,
            bias=False)
        self.dense_2 = Dense(
            gem_conf.hid_size_edge,
            gem_conf.emb_size_edge,
            activation=gem_conf.activation,
            bias=False)

    def forward(self, node_embed, local_node_embed, local_edge_embed, rbf_embed):
        _, _, num_neighbors, _ = local_node_embed.shape
        feats_in = torch.cat([
            torch.tile(
                node_embed[:, :, None, :], (1, 1, num_neighbors, 1)),
            local_node_embed,
            local_edge_embed,
            rbf_embed,
        ], dim=-1)
        feats_hid = self.dense_1(feats_in)
        feats_out = self.dense_2(feats_hid)
        return feats_out


class DirectionEmbedding(torch.nn.Module):

    def __init__(
            self,
            gem_conf,
        ):
        super(DirectionEmbedding, self).__init__()
        self._gem_conf = gem_conf

        self.atom_embedder = torch.nn.Embedding(3, gem_conf.emb_size_atom)
        torch.nn.init.uniform_(
            self.atom_embedder.weight, a=-np.sqrt(3), b=np.sqrt(3))

        self.cbf_basis = basis_utils.SphericalBasisLayer(
            num_spherical=gem_conf.num_spherical,
            num_radial=gem_conf.num_radial,
            cutoff=gem_conf.cutoff,
            envelope_exponent=gem_conf.envelope_exponent,
        )
        self.sbf_basis = basis_utils.TensorBasisLayer(
            num_spherical=gem_conf.num_spherical,
            num_radial=gem_conf.num_radial,
            cutoff=gem_conf.cutoff,
            envelope_exponent=gem_conf.envelope_exponent,
        )
        self.rbf_basis = basis_utils.BesselBasisLayer(
            num_radial=gem_conf.num_radial,
            cutoff=gem_conf.cutoff,
            envelope_exponent=gem_conf.envelope_exponent,
        )

        self.mlp_rbf = Dense(
            gem_conf.num_radial,
            gem_conf.emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf = Dense(
            gem_conf.num_radial * gem_conf.num_spherical + gem_conf.emb_size_atom,
            gem_conf.emb_size_cbf,
            activation=None,
            bias=False,
        )
        self.mlp_sbf = Dense(
            gem_conf.num_radial * gem_conf.num_spherical ** 2 + gem_conf.emb_size_atom*2,
            gem_conf.emb_size_sbf,
            activation=None,
            bias=False,
        )

    def compute_angles(self, bb_pos, local_bb_pos):
        # GemNet diagram:
        #  c     d
        #  |     |
        #  a-----b 

        # Extract vectors
        ca_pos = bb_pos[..., 1, :]
        non_ca_pos = bb_pos[..., [0, 2, 3], :]
        local_ca_pos = local_bb_pos[..., 1, :]
        local_non_ca_pos = local_bb_pos[..., [0, 2, 3], :]

        # VECTOR CONVENTION:
        # - first index: starting position
        # - second index: ending position
        # - third index: backbone angle of backbone atom connected to first index.
        #
        # C-alpha pairwise displacement vectors: [B, N, K, 3]
        # calpha_vecs[*, a, b]: a --> b
        # calpha_vecs[*, b, a]: a <-- b
        calpha_vecs = (ca_pos[..., None, :] - local_ca_pos) + 1e-10
        # Backbone pairwise displacements: [B, N, K, 3, 3]
        bb_vecs = (non_ca_pos[..., None, :, :] - local_non_ca_pos) + 1e-10

        # Calculate phi angles: [B, N, N, 3]
        # where 3 is the number of non-Ca atoms per residue.
        # phi_angles[a, b, c] is angle:
        # c
        # ^
        # |
        # |
        # a ----> b
        batch_size, num_res, num_neighbors, num_non_ca, _ = bb_vecs.shape
        tiled_calpha_vecs = torch.tile(calpha_vecs[..., None, :], (1, 1, 1, num_non_ca, 1))
        phi_angles = all_atom.calculate_neighbor_angles(
            tiled_calpha_vecs.reshape(-1, 3),
            bb_vecs.reshape(-1, 3)
        ).reshape(batch_size, num_res, num_neighbors, num_non_ca) 

        # Projection of backbone vectors onto C-alpha plane: [B, N, K, 3, 3]
        # proj_vecs[*, a, b, c]: 
        # projection of a --> c onto plane with norm vector a --> b
        proj_vecs_1 = all_atom.vector_projection(
            bb_vecs.reshape(-1, 3),
            tiled_calpha_vecs.reshape(-1, 3)
        ).reshape(batch_size, num_res, num_neighbors, num_non_ca, 3)
        # Flip order of C-alpha residues so that we project onto the opposite plane.
        proj_vecs_2 = all_atom.vector_projection(
            bb_vecs.transpose(0, 1).reshape(-1, 3),
            tiled_calpha_vecs.reshape(-1, 3)
        ).reshape(batch_size, num_res, num_neighbors, num_non_ca, 3)

        # Calculate dihedral angle: [*, N, K, 3, 3]
        # Should be symmetric: theta_angles[*, a, b] == theta_angles[*, b, a].T
        proj_ac = torch.tile(proj_vecs_1[:, :, :, None, :], (1, 1, 1, 3, 1))
        proj_bd = torch.tile(proj_vecs_2[:, :, None, :, :], (1, 1, 3, 1, 1))
        theta_angles = all_atom.calculate_neighbor_angles(
            proj_ac.reshape(-1, 3),
            proj_bd.reshape(-1, 3),
        ).reshape(batch_size, num_res, num_neighbors, num_non_ca, num_non_ca)

        calpha_dists = torch.linalg.norm(calpha_vecs, axis=-1)

        return calpha_dists, phi_angles, theta_angles

    def forward(self, bb_pos, local_bb_pos, node_mask, edge_mask):
        ca_dists, phi_angles, theta_angles = self.compute_angles(
            bb_pos, local_bb_pos)
        ca_dists *= edge_mask
        ca_dists += 1e-8  # Avoid dividing/sqrt by 0.
        phi_angles *= edge_mask[..., None]
        theta_angles *= edge_mask[..., None, None]

        ##########################
        #  Basis transformation. #
        ##########################
        batch_size, num_res, num_neighbors = edge_mask.shape
        non_edge_mask_idx = torch.where(1 - edge_mask)

        # CBF features
        cbf_feats = self.cbf_basis(
            torch.tile(ca_dists[..., None], (1, 1, 3)).ravel(),
            phi_angles.ravel()
        ).reshape(batch_size, num_res, num_neighbors, 3, -1)
        # Need mask out possible NaNs.
        cbf_feats[non_edge_mask_idx] = 0.0

        # SBF features
        sbf_feats = self.sbf_basis(
            torch.tile(ca_dists[..., None, None], (1, 1, 3, 3)).ravel(),
            torch.tile(phi_angles[..., None], (1, 1, 1, 3)).ravel(),
            theta_angles.ravel()
        ).reshape(batch_size, num_res, num_neighbors, 3, 3, -1)
        # Masking. Need to do indexing because of NaNs.
        sbf_feats[non_edge_mask_idx] = 0.0
        assert not torch.isnan(sbf_feats).any()

        # RBF features
        rbf_feats = self.rbf_basis(ca_dists.ravel()).reshape(
            batch_size, num_res, num_neighbors, -1)
        rbf_feats[non_edge_mask_idx] = 0.0
        assert not torch.isnan(rbf_feats).any()

        # Create atom features
        # [3, D]
        bb_atom_feats = self.atom_embedder(torch.arange(3).to(node_mask.device))
        cbf_feats = torch.cat([
            cbf_feats,
            torch.tile(
                bb_atom_feats[None, None, None, :, :],
                (batch_size, num_res, num_neighbors, 1, 1))
        ], dim=-1)

        # [3, 3, D]
        cross_atom_feats = torch.cat([
            torch.tile(bb_atom_feats[:, None, :], (1, 3, 1)),
            torch.tile(bb_atom_feats[None, :, :], (3, 1, 1))
        ], dim=-1)
        sbf_feats = torch.cat([
            sbf_feats,
            torch.tile(
                cross_atom_feats[None, None, None, :, :, :],
                (batch_size, num_res, num_neighbors, 1, 1, 1)
            )
        ], dim=-1)

        # [B, N, K, D]
        rbf_embed = self.mlp_rbf(rbf_feats)
        # [B, N, K, 3, D]
        cbf_embed = self.mlp_cbf(cbf_feats)
        # [B, N, K, 3, 3, D]
        # sbf_embed = self.mlp_sbf(sbf_feats)
        sbf_embed = None
        return cbf_embed, sbf_embed, rbf_embed


class InteractionBlock(torch.nn.Module):

    def __init__(
            self,
            gem_conf
        ):
        super(InteractionBlock, self).__init__()
        self._gem_conf = gem_conf

        # Dense transformation
        self.mlp_edge = Dense(
            gem_conf.emb_size_edge,
            gem_conf.emb_size_edge,
            activation=None, #gem_conf.activation,
            bias=False
        )

        # Up projections of basis representations, bilinear layer and scaling factors
        self.mlp_rbf = Dense(
            gem_conf.emb_size_rbf,
            gem_conf.emb_size_edge,
            activation=None,
            name='MLP_rbf4_2'
        )
        self.ln_rbf = torch.nn.LayerNorm(gem_conf.emb_size_edge)
        # self.scale_rbf = ScalingFactor(
        #     scale_file=gem_conf.scale_file,
        # )

        self.mlp_cbf = Dense(
            gem_conf.emb_size_cbf,
            gem_conf.emb_size_quad, 
            activation=None,
            name="MLP_cbf4_2",
            bias=False
        )
        self.ln_cbf = torch.nn.LayerNorm(gem_conf.emb_size_quad)
        # self.scale_cbf = ScalingFactor(scale_file=scale_file, name=name + "_had_cbf")

        self.blp_sbf = torch.nn.Bilinear(
            gem_conf.emb_size_quad,
            gem_conf.emb_size_sbf,
            gem_conf.emb_size_bilinear,
            bias=False
        )
        self.ln_sbf = torch.nn.LayerNorm(gem_conf.emb_size_bilinear)
        # self.mlp_sbf = EfficientInteractionBilinear(
        #     gem_conf.emb_size_quad,
        #     gem_conf.emb_size_sbf,
        #     gem_conf.emb_size_bilinear,
        #     name="MLP_sbf4_2"
        # )
        # self.scale_sbf_sum = ScalingFactor(
        #     scale_file=scale_file, name=name + "_sum_sbf"
        # )  # combines scaling for bilinear layer and summation

        # Down and up projections
        self.down_projection = Dense(
            gem_conf.emb_size_edge,
            gem_conf.emb_size_quad,
            activation=gem_conf.activation,
            bias=False,
            name="dense_down",
        )

        self.up_projection = Dense(
            gem_conf.emb_size_quad,
            gem_conf.emb_size_edge,
            activation=None, #gem_conf.activation,
            bias=False,
            name="dense_up",
        )
        # self.up_projection_ca = Dense(
        #     gem_conf.emb_size_bilinear,
        #     gem_conf.emb_size_edge,
        #     activation=gem_conf.activation,
        #     bias=False,
        #     name="dense_up_ca",
        # )
        # self.up_projection_ac = Dense(
        #     gem_conf.emb_size_bilinear,
        #     gem_conf.emb_size_edge,
        #     activation=gem_conf.activation,
        #     bias=False,
        #     name="dense_up_ac",
        # )

        self.inv_sqrt_2 = 1 / (2.0 ** 0.5)

    def forward(
            self,
            *,
            edge_embed,
            rbf_embed,
            cbf_embed,
            sbf_embed,
            debug
        ):
        edge_embed = self.mlp_edge(edge_embed)  # (nEdges, emb_size_edge)

        # Transform via radial bessel basis
        edge_embed = edge_embed * self.mlp_rbf(rbf_embed)  # (nEdges, emb_size_edge)
        if debug:
            ipdb.set_trace()
        # edge_embed = self.ln_rbf(edge_embed)
        # x_db = self.scale_rbf(x_db, x_db2)

        # Down project embeddings
        edge_embed = self.down_projection(edge_embed)  # (nEdges, emb_size_quad)

        # Transform via circular spherical bessel basis
        edge_embed = torch.tile(edge_embed[:, :, :, None, :], (1, 1, 1, 3, 1))
        edge_embed = edge_embed * self.mlp_cbf(cbf_embed)  # (intmTriplets, emb_size_quad)
        if debug:
            ipdb.set_trace()
        # edge_embed = self.ln_cbf(edge_embed)
        # x_db = self.scale_cbf(x_db, x_db2)

        # Transform via spherical bessel basis
        # x_db = x_db[id4_expand_abd]  # (nQuadruplets, emb_size_quad)
        # edge_embed = torch.tile(
        #     edge_embed[:, :, :, :, None, :], (1, 1, 1, 1, 3, 1))
        # edge_embed = self.blp_sbf(sbf_embed, edge_embed)  # (nEdges, emb_size_bilinear)
        # if debug:
        #     ipdb.set_trace()
        # edge_embed = self.ln_sbf(edge_embed)
        # return edge_embed
        # x = self.scale_sbf_sum(x_db, x)
        return self.up_projection(edge_embed) # * self.inv_sqrt_2

        # # Basis representation:
        # # rbf(d_db)
        # # cbf(d_ba, angle_abd)
        # # sbf(d_ca, angle_cab, angle_cabd)

        # # Upproject embeddings
        # x_ca = self.up_projection_ca(x)  # (nEdges, emb_size_edge)
        # x_ac = self.up_projection_ac(x)  # (nEdges, emb_size_edge)

        # # Merge interaction of c->a and a->c
        # x_ac = x_ac[id_swap]  # swap to add to edge a->c and not c->a
        # x4 = x_ca + x_ac
        # x4 = x4 * self.inv_sqrt_2

        # return x4

class EdgeTransition(torch.nn.Module):
    def __init__(
            self,
            gem_conf
        ):
        super(EdgeTransition, self).__init__()

        self._gem_conf = gem_conf
        self.edge_down_1 = Dense(
            self._gem_conf.emb_size_edge * 3, # 9,
            self._gem_conf.edge_trans_hid,
            activation=None, # self._gem_conf.activation,
            bias=False
        )
        self.edge_resnet = layers.ResidualLayer(
            units=self._gem_conf.edge_trans_hid,
            activation=self._gem_conf.activation,
        )
        self.edge_resnet_ln = torch.nn.LayerNorm(
            self._gem_conf.edge_trans_hid)
        self.edge_down_2 = Dense(
            self._gem_conf.edge_trans_hid,
            self._gem_conf.emb_size_edge,
            activation=self._gem_conf.activation,
            bias=False
        )
        self.edge_out_ln = torch.nn.LayerNorm(self._gem_conf.emb_size_edge)

    def forward(self, edge_embed):
        batch_size, num_res, num_neighbors = edge_embed.shape[:3]
        edge_embed = edge_embed.reshape(batch_size, num_res, num_neighbors, -1)
        edge_embed = self.edge_down_1(edge_embed)
        edge_embed = self.edge_resnet(edge_embed)
        # edge_embed = self.edge_resnet_ln(edge_embed)
        edge_embed = self.edge_down_2(edge_embed)
        # edge_embed = self.edge_out_ln(edge_embed)
        return edge_embed


class NodeUpdate(torch.nn.Module):
    def __init__(
            self,
            gem_conf
        ):
        super(NodeUpdate, self).__init__()
        self._gem_conf = gem_conf
        self.rbf_mlp = Dense(
            self._gem_conf.emb_size_rbf,
            self._gem_conf.emb_size_edge,
            activation=None,
            bias=False)
        self.init_edge_mlp = Dense(
            self._gem_conf.emb_size_edge,
            self._gem_conf.emb_size_edge,
            activation=None,
            bias=False)
        self.edge_weights_mlp = torch.nn.Bilinear(
            self._gem_conf.emb_size_edge,
            self._gem_conf.emb_size_edge,
            1,
            bias=False
        )
        self.node_out_mlp = Dense(
            self._gem_conf.emb_size_edge,
            self._gem_conf.emb_size_node,
            activation=None, # self._gem_conf.activation,
            bias=False
        )

    def forward(
            self,
            local_edge_embed,
            init_local_edge_embed,
            rbf_embed,
            local_edge_mask
        ):
        init_local_edge_embed = self.init_edge_mlp(init_local_edge_embed)
        rbf_embed = self.rbf_mlp(rbf_embed)
        edge_weights = self.edge_weights_mlp(init_local_edge_embed, rbf_embed)
        node_denom = torch.sum(local_edge_mask, dim=-1)
        node_embed = torch.sum(
            local_edge_mask[..., None] * edge_weights * local_edge_embed,
            dim=-2
        )  # / (node_denom[:, :, None] + 1e-5)
        node_embed = node_embed.to(rbf_embed.dtype)
        return self.node_out_mlp(node_embed)


class ScoreHead(torch.nn.Module):
    def __init__(
            self,
            gem_conf,
        ):
        super(ScoreHead, self).__init__()
        self.linear_1 = Linear(
            gem_conf.emb_size_node,
            gem_conf.head_hid1,
            init="relu")
        self.linear_2 = Linear(
            gem_conf.head_hid1,
            gem_conf.head_hid2,
            init="relu")
        self.linear_3 = Linear(
            gem_conf.head_hid2,
            6,
            init="final")

        self.relu = torch.nn.ReLU()

    def forward(self, s):
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        # s = self.relu(s)
        s = self.linear_3(s)
        return s


class TorsionHead(torch.nn.Module):
    def __init__(
            self,
            gem_conf,
        ):
        super(TorsionHead, self).__init__()
        self.linear_1 = Linear(
            gem_conf.emb_size_node,
            gem_conf.head_hid1,
            init="relu")
        self.linear_2 = Linear(
            gem_conf.head_hid1,
            gem_conf.head_hid2,
            init="relu")
        self.linear_3 = Linear(
            gem_conf.head_hid2,
            2,
            init="final")

        self.relu = torch.nn.ReLU()

    def forward(self, s):
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        unnormalized_s = self.linear_3(s)
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_s ** 2, dim=-1, keepdim=True),
                min=1e-8,
            )
        )
        normalized_s = unnormalized_s / norm_denom
        return normalized_s


class FrameGem(torch.nn.Module):
    def __init__(
            self,
            model_conf
        ):
        super(FrameGem, self).__init__()
        self._model_conf = model_conf
        self._gem_conf = model_conf.gemnet

        self.direction_embedder = DirectionEmbedding(self._gem_conf)
        self.edge_embedder = EdgeEmbedding(self._gem_conf)
        self.interaction_block = InteractionBlock(self._gem_conf)
        self.edge_transition = EdgeTransition(self._gem_conf)
        self.node_update = NodeUpdate(self._gem_conf)
        self.score_head = ScoreHead(self._gem_conf)
        self.torsion_head = TorsionHead(self._gem_conf)

    def local_graph(self, bb_pos, node_mask, node_embed, edge_mask, edge_embed):
        batch_size, num_res = node_mask.shape
        ca_pos = bb_pos[..., 1, :]
        dists_2d = torch.linalg.norm(
            ca_pos[:, :, None, :] - ca_pos[:, None, :, :], axis=-1)

        # Add bias to self-edges and masked residues.
        dists_2d += torch.eye(num_res).to(node_mask.device)[None]*1e5
        dists_2d += (1 - edge_mask)*1e5

        # Calculate local graph features
        _, local_graph = torch.topk(
            dists_2d, k=self._gem_conf.num_neighbors, dim=-1, largest=False)
        # TODO: Add local in sequence order.
        local_edge_mask = torch.gather(
            edge_mask * (1 - torch.eye(num_res))[None].to(edge_mask.device),
            -1,
            local_graph,
            sparse_grad=True
        )

        # Gather local features.
        flat_local_graph = local_graph.view(batch_size, -1)
        _, res_per_batch = flat_local_graph.shape
        i_idx = torch.repeat_interleave(torch.arange(batch_size), res_per_batch)
        j_idx = flat_local_graph.view(-1)
        local_bb_pos = bb_pos[(i_idx, j_idx)].reshape(
            batch_size, num_res, self._gem_conf.num_neighbors, 4, 3)

        local_edge_embed = local_2d_gather(local_graph, edge_embed)
        local_node_embed = local_1d_gather(local_graph, node_embed)
        return (
            local_edge_mask,
            local_bb_pos,
            local_edge_embed,
            local_node_embed
        )

    def forward(
            self,
            init_node_embed,  # [B, N, D]
            init_edge_embed,  # [B, N, N, D]
            input_feats,
            debug
        ):
        # Extract raw features
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        rigids_tensor = input_feats['rigids_t']
        rigids = ru.Rigid.from_tensor_7(rigids_tensor)
        rots = rigids.get_rots()
        atom_pos = all_atom.compute_backbone(
            rigids,
            torch.zeros(node_mask.shape + (2,))
        )[0].to(node_mask.device)
        bb_pos = atom_pos[..., :4, :]

        # TODO: REMOVE
        # init_node_embed[..., :3] = init_rots.invert_apply(input_feats['trans_score'])
        # init_node_embed[..., 3:6] = input_feats['rot_score']

        (
            local_edge_mask,
            local_bb_pos,
            init_local_edge_embed,
            init_local_node_embed,
        ) = self.local_graph(
            bb_pos, node_mask, init_node_embed, edge_mask, init_edge_embed)
        node_embed = torch.clone(init_node_embed)
        local_edge_embed = torch.clone(init_local_edge_embed)
        local_node_embed = torch.clone(init_local_node_embed)

        # Create basis features
        # TODO: Add chain graph.
        cbf_embed, sbf_embed, rbf_embed = self.direction_embedder(
            bb_pos, local_bb_pos, node_mask, local_edge_mask)
        # [B, N, K, D]
        rbf_embed *= local_edge_mask[..., None]
        # [B, N, K, 3, D]
        cbf_embed *= local_edge_mask[..., None, None]
        # [B, N, K, 3, 3, D]
        # sbf_embed *= local_edge_mask[..., None, None, None]

        # TODO: Multiple blocks.
        # Embed initial edge features with RBF
        if debug:
            ipdb.set_trace()
        local_edge_embed = self.edge_embedder(
            node_embed, local_node_embed, local_edge_embed, rbf_embed)
        # [B, N, K, D]
        local_edge_embed *= local_edge_mask[..., None]
        if debug:
            ipdb.set_trace()

        # Interaction layer
        local_edge_embed = self.interaction_block(
            edge_embed=local_edge_embed,
            rbf_embed=rbf_embed,
            cbf_embed=cbf_embed,
            sbf_embed=sbf_embed,
            debug=debug,
        )
        # local_edge_embed *= local_edge_mask[..., None, None, None]
        local_edge_embed *= local_edge_mask[..., None, None]
        if debug:
            ipdb.set_trace()

        # Edge update
        local_edge_embed = self.edge_transition(local_edge_embed)
        local_edge_embed *= local_edge_mask[..., None]

        if debug:
            ipdb.set_trace()

        # Node update
        node_embed = self.node_update(
            local_edge_embed, init_local_edge_embed, rbf_embed, local_edge_mask)
        node_embed *= node_mask[..., None]

        score_out = self.score_head(node_embed)
        score_out *= node_mask[..., None]
        trans_out = score_out[..., :3]
        rot_out = score_out[..., 3:]

        trans_out = rots.apply(trans_out)
        if self._model_conf.equivariant_rot_score:
            rot_out = rots.apply(
                rot_out)

        torsion_out = self.torsion_head(node_embed)
        torsion_out *= node_mask[..., None]

        if debug:
            ipdb.set_trace()

        assert not torch.isnan(trans_out).any()
        assert not torch.isnan(rot_out).any()
        assert not torch.isnan(torsion_out).any()
        return trans_out, rot_out, torsion_out
