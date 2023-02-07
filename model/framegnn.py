from torch import nn
import torch
from openfold.utils import rigid_utils as ru
from openfold.utils.rigid_utils import Rigid
from model import ipa_pytorch
import functools as fn

Linear = ipa_pytorch.Linear


def calc_rbf(dists, min_dist, max_dist, num_rbf):
    # Distance radial basis function
    dist_mu = torch.linspace(min_dist, max_dist, num_rbf).to(dists.device)
    dist_mu = dist_mu.view([1,1,1,-1])
    dist_sigma = (max_dist - min_dist) / num_rbf
    dist_expand = torch.unsqueeze(dists, -1)
    rbf = torch.exp(-((dist_expand - dist_mu) / dist_sigma)**2)
    return rbf


class EdgeLayer(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super(EdgeLayer, self).__init__()

        self.linear_1 = Linear(dim_in, dim_hid, init="relu")
        self.linear_2 = Linear(dim_hid, dim_hid, init="relu")
        self.linear_3 = Linear(dim_hid, dim_out, init="glorot")

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        return s


class ScoreLayer(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super(ScoreLayer, self).__init__()

        self.linear_1 = Linear(dim_in, dim_hid, init="relu")
        self.linear_2 = Linear(dim_hid, dim_hid, init="relu")
        self.linear_3 = Linear(dim_hid, dim_hid, init="relu")
        self.linear_4 = Linear(dim_hid, dim_out, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.linear_3(s)
        s = self.relu(s)
        s = self.linear_4(s)
        return s


class NodeLayer(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, residual=True):
        super(NodeLayer, self).__init__()

        self.linear_1 = Linear(dim_in, dim_hid, init="relu")
        self.linear_2 = Linear(dim_hid, dim_hid, init="relu")
        self.linear_3 = Linear(dim_hid, dim_out, init="glorot")

        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, s):
        init_s = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        if self.residual:
            s = s + init_s
        return s

class SE_GCL(nn.Module):
    """
    SE(3) Equivariant Convolutional Layer
    """

    def __init__(
            self,
            *,
            # Local graph features
            local_edge_hid,
            local_edge_out,
            # Sub graph features
            subgraph_edge_hid,
            subgraph_edge_out,
            attn_weight_hid,
            # Global graph features
            global_node_in,
            global_node_hid,
            global_node_out,
            global_edge_in,
            global_edge_hid,
            global_edge_out,
            # Distance features
            min_rbf_dist,
            max_rbf_dist,
            num_rbf,
            atom_embed_size,
        ):
        super(SE_GCL, self).__init__()

        self._calc_rbf = fn.partial(
            calc_rbf,
            min_dist=min_rbf_dist,
            max_dist=max_rbf_dist,
            num_rbf=num_rbf,
        )

        # Local graph embeddings
        local_edge_in = atom_embed_size*2 + num_rbf + global_edge_in
        self.proximity_local_edge_embed = EdgeLayer(
            local_edge_in,
            local_edge_hid,  # TODO: Set hidden size dynamically.
            local_edge_out,
        )
        self.chain_local_edge_embed = EdgeLayer(
            local_edge_in,
            local_edge_hid,
            local_edge_out,
        )
        self.local_layer_norm = nn.LayerNorm(local_edge_out)

        # Subgraph edge embeddings
        self.proximity_sub_edge_embed = EdgeLayer(
            local_edge_out + atom_embed_size,
            subgraph_edge_hid,
            subgraph_edge_out,
        )
        self.chain_sub_edge_embed = EdgeLayer(
            local_edge_out + atom_embed_size,
            subgraph_edge_hid,  # TODO: Set hidden size dynamically.
            subgraph_edge_out,
        )
        self.sub_layer_norm = nn.LayerNorm(subgraph_edge_out)

        # Subgraph attention weights
        self.proximity_sub_weights = EdgeLayer(
            subgraph_edge_out, attn_weight_hid, 1)
        self.chain_sub_weights = EdgeLayer(
            subgraph_edge_out, attn_weight_hid, 1)

        # Global node embeddings
        self.global_node_embed = NodeLayer(
            global_node_in + subgraph_edge_out * 2,
            global_node_hid,  # TODO: Set hidden size dynamically.
            global_node_out,
            residual=False
        )
        self.global_node_layer_norm = nn.LayerNorm(global_node_out)

        # Global edge embeddings
        self.global_edge_embed = EdgeLayer(
            global_edge_in + global_node_out*2 + num_rbf,
            global_edge_hid,  # TODO: Set hidden size dynamically.
            global_edge_out,
        )
        self.global_edge_layer_norm = nn.LayerNorm(global_edge_out)

    def create_local_graph_feats(
            self,
            *,
            subgraph,
            edge_dists,
            edge_embed,
            atom_embed
        ):
        batch_size, num_res, num_neighbors = subgraph.shape
        # [B, N, 4, S, 4, D1]
        local_graph_edge_embed = torch.tile(
            edge_embed[:, :, None, :, None, :],
            (1, 1, 4, 1, 4, 1)
        )
        # [4, 4, D2]
        cross_atom_embed = torch.concat([
            torch.tile(atom_embed[:, None, :], (1, 4, 1)),
            torch.tile(atom_embed[None, :, :], (4, 1, 1)),
        ], dim=-1)
        # [B, N, 4, S, 4, D2]
        local_graph_atom_embed = torch.tile(
            cross_atom_embed[None, None, :, None, :, :],
            (batch_size, num_res, 1, num_neighbors, 1, 1)
        )
        # [B, N, 4, S, 4, D3]
        local_graph_dist_rbf = self._calc_rbf(edge_dists)
        # [B, N, 4, S, 4, D1+D2+D3]
        local_graph_edge_init = torch.concat([
            local_graph_edge_embed, 
            local_graph_atom_embed,
            local_graph_dist_rbf,
        ], dim=-1)
        return local_graph_edge_init

    def create_sub_graph_feats(
            self,
            *,
            local_edge_feats,
            atom_embed,
        ):
        num_batch, num_res, _, num_neighbors, _ = local_edge_feats.shape
        local_atom_feats = torch.tile(
            atom_embed[None, None, :, None, :],
            (num_batch, num_res, 1, num_neighbors, 1)
        )
        # [B, N, 4, S, D]
        sub_edge_feats = torch.concat([
            local_atom_feats, local_edge_feats
        ], dim=-1)
        return sub_edge_feats

    def create_global_edge_feats(
            self,
            *,
            node_feats,
            edge_feats,
            edge_dists,
        ):
        _, num_res, _ = node_feats.shape
        cross_node_embed = torch.concat([
            torch.tile(node_feats[:, None, :, :], (1, num_res, 1, 1)),
            torch.tile(node_feats[:, :, None, :], (1, 1, num_res, 1)),
        ], dim=-1)
        global_rbf_feats = self._calc_rbf(edge_dists)
        global_edge_feats = torch.concat([
            cross_node_embed,
            edge_feats,
            global_rbf_feats,
        ], dim=-1)
        return global_edge_feats

    def extract_subgraph_edge_embed(self, subgraph, edge_embed):
        return torch.gather(
            edge_embed,
            -2,
            torch.tile(
                subgraph[..., None],
                (1, 1, 1, edge_embed.shape[-1])
            )
        )

    def forward(
            self,
            *,
            node_embed,
            edge_embed,
            atom_embed,
            proximity_graph,
            proximity_dists,
            chain_graph,
            chain_dists,
            chain_graph_mask,
            global_ca_dists,
            node_mask,
            edge_mask
        ):
        ###########################################
        # Extract subgraph level edge embeddings. #
        ###########################################
        # [B, N, S, D]
        proximity_edge_embed = self.extract_subgraph_edge_embed(
            proximity_graph, edge_embed)
        chain_edge_embed = self.extract_subgraph_edge_embed(
            chain_graph, edge_embed)

        ###########################
        # Embed each local graph. #
        ###########################
        local_graph_feats = fn.partial(
            self.create_local_graph_feats,
            atom_embed=atom_embed,
        )
        # [B, N, 4, S, 4, D]
        proximity_local_edge_init = local_graph_feats(
            subgraph=proximity_graph,
            edge_dists=proximity_dists,
            # TODO: Try not including edge embeddings in local graph.
            edge_embed=proximity_edge_embed)
        chain_local_edge_init = local_graph_feats(
            subgraph=chain_graph,
            edge_dists=chain_dists,
            edge_embed=chain_edge_embed)

        proximity_local_edge_embed = self.proximity_local_edge_embed(
            proximity_local_edge_init)
        chain_local_edge_embed = self.chain_local_edge_embed(
            chain_local_edge_init)

        # Masking and layer norm
        proximity_local_edge_embed *= node_mask[:, :, None, None, None, None]  
        chain_local_edge_embed *= chain_graph_mask[:, :, None, :, None, None]
        chain_local_edge_embed *= node_mask[:, :, None, None, None, None]
        proximity_local_edge_embed = self.local_layer_norm(proximity_local_edge_embed)
        chain_local_edge_embed = self.local_layer_norm(chain_local_edge_embed)

        proximity_local_node_embed = torch.mean(
            proximity_local_edge_embed, dim=-2)
        chain_local_node_embed = torch.mean(
            chain_local_edge_embed, dim=-2)

        ##########################
        # Embed sub graph edges. #
        ##########################
        sub_graph_feats = fn.partial(
            self.create_sub_graph_feats,
            atom_embed=atom_embed
        )
        # [B, N, 4, S, D]
        proximity_sub_edge_init = sub_graph_feats(
            local_edge_feats=proximity_local_node_embed,
        )
        chain_sub_edge_init = sub_graph_feats(
            local_edge_feats=chain_local_node_embed,
        )
        proximity_sub_edge_embed = self.proximity_sub_edge_embed(proximity_sub_edge_init)
        chain_sub_edge_embed = self.chain_sub_edge_embed(chain_sub_edge_init)

        # Masking and layer norm
        proximity_sub_edge_embed *= node_mask[:, :, None, None, None]  
        chain_sub_edge_embed *= chain_graph_mask[:, :, None, :, None]
        chain_sub_edge_embed *= node_mask[:, :, None, None, None]
        proximity_sub_edge_embed = self.sub_layer_norm(proximity_sub_edge_embed)
        chain_sub_edge_embed = self.sub_layer_norm(chain_sub_edge_embed)

        # [B, N, S, D]
        proximity_sub_node_embed = torch.mean(proximity_sub_edge_embed, dim=-3)
        chain_sub_node_embed = torch.mean(chain_sub_edge_embed, dim=-3)

        ########################
        # Embed sub graph node #
        ########################
        # [B, N, S, 1]
        proximity_attn_weights = self.proximity_sub_weights(
            proximity_sub_node_embed)
        chain_attn_weights = self.proximity_sub_weights(
            chain_sub_node_embed)

        # [B, N, D]
        proximity_node_embed = torch.mean(
            proximity_attn_weights * proximity_sub_node_embed, dim=-2)
        chain_node_embed = torch.mean(
            chain_attn_weights * chain_sub_node_embed, dim=-2)
        sub_node_embed = torch.concat(
            [proximity_node_embed, chain_node_embed], dim=-1)

        ############################
        # Embed global graph nodes #
        ############################
        # [B, N, D]
        global_node_embed = torch.concat([
            sub_node_embed, node_embed
        ], dim=-1)
        global_node_embed = self.global_node_embed(global_node_embed)
        global_node_embed *= node_mask[:, :, None]
        global_node_embed = self.global_node_layer_norm(global_node_embed)
        # [B, N, N, D]
        global_edge_init = self.create_global_edge_feats(
            node_feats=global_node_embed,
            edge_feats=edge_embed,
            edge_dists=global_ca_dists
        )
        global_edge_embed = self.global_edge_embed(global_edge_init)
        global_edge_embed *= edge_mask[:, :, :, None]
        global_edge_embed = self.global_edge_layer_norm(global_edge_embed)
        return global_node_embed, global_edge_embed


class FrameGNN(nn.Module):
    def __init__(
            self,
            model_conf
        ):

        super(FrameGNN, self).__init__()
        self._model_conf = model_conf
        self._fgnn_conf = model_conf.framegnn

        self.atom_embedder = nn.Embedding(
            4, self._fgnn_conf.atom_embed_size)

        self.trunk = nn.ModuleDict()
        for l in range(self._fgnn_conf.num_layers):
            self.trunk[f'gcl_{l}'] = SE_GCL(
                atom_embed_size=self._fgnn_conf.atom_embed_size,
                **self._fgnn_conf.gcl
            )

        self.trans_pred = ScoreLayer(
            self._fgnn_conf.gcl.global_node_out,
            self._fgnn_conf.score_hid,
            3,
        )
        self.rot_pred = ScoreLayer(
            self._fgnn_conf.gcl.global_node_out,
            self._fgnn_conf.score_hid,
            3,
        )
        self.torsion_pred = ipa_pytorch.TorsionAngles(
            self._fgnn_conf.gcl.global_node_out, 1)

    def forward(
            self,
            node_embed,
            edge_embed,
            input_feats
        ):

        node_mask = input_feats['res_mask'].type(torch.float32)  # [B, N]
        edge_mask = node_mask[..., None] * node_mask[..., None, :]

        # Embed atom types
        atom_type = torch.arange(4).to(node_embed.device)
        atom_embed = self.atom_embedder(atom_type)

        # Construct global graph
        init_rigids = Rigid.from_tensor_7(input_feats['rigids_t']) 
        init_rots = init_rigids.get_rots()
        ca_pos = init_rigids.get_trans()
        global_ca_dists = torch.linalg.norm(
            ca_pos[:, :, None, :] - ca_pos[:, None, :, :], axis=-1)
        static_gcl_args = dict(
            atom_embed=atom_embed,
            proximity_graph=input_feats['proximity_graph'],
            proximity_dists=input_feats['proximity_dists'],
            chain_graph=input_feats['chain_graph'],
            chain_dists=input_feats['chain_dists'],
            chain_graph_mask=input_feats['chain_graph_mask'],
            global_ca_dists=global_ca_dists,
            node_mask=node_mask,
            edge_mask=edge_mask
        )

        for l in range(self._fgnn_conf.num_layers):
            node_embed, edge_embed = self.trunk[f'gcl_{l}'](
                node_embed=node_embed,
                edge_embed=edge_embed,
                **static_gcl_args
            )

        local_trans_score = self.trans_pred(node_embed)
        trans_score = init_rots.apply(local_trans_score)
        rot_score = self.rot_pred(node_embed)
        if self._model_conf.equivariant_rot_score:
            rot_score = init_rots.apply(
                rot_score)
        _, psi_pred = self.torsion_pred(node_embed) 

        return trans_score, rot_score, psi_pred