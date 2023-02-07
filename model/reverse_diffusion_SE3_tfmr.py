import torch
from torch import nn
from openfold.utils import rigid_utils as ru
from openfold.utils.rigid_utils import Rigid
from openfold.np import residue_constants
from model import reverse_se3_diffusion

import dgl
from model.SE3_network import SE3TransformerWrapper



pi = 3.141592653589793
Tensor = torch.Tensor

def rigid_to_3_body(bb_rigids):
    """rigid_to_3_body returns the coordinates of N, Ca, and C using idealized geometries from the backbone rigids

    Args:
        bb_rigids of shape [...]
    
    Returns:
        bb_atoms of shape [..., 3, 3]

    """
    idealized_pos = torch.tensor(residue_constants.restype_atom14_rigid_group_positions)
    idealized_frames = ru.Rigid.from_3_points(
        idealized_pos[:, 0],
        idealized_pos[:, 1],
        idealized_pos[:, 2]
    )
    idealized_rots = idealized_frames.get_rots()

    inverted_idealized_pos = idealized_rots[:, None].invert_apply(idealized_pos)
    inverted_idealized_bb_pos = torch.stack([
        inverted_idealized_pos[:, 0],  # N
        inverted_idealized_pos[:, 1],  # Ca
        inverted_idealized_pos[:, 2],  # C
    ], dim=1)
    ideal_bb_pos = inverted_idealized_bb_pos[0]
    bb_rigids = Rigid.from_tensor_7(bb_rigids)

    ideal_bb_pos = ideal_bb_pos.to(bb_rigids.device)
    bb_atoms = bb_rigids[..., None].apply(ideal_bb_pos)
    return bb_atoms




class ReverseDiffusionSE3TFMR(nn.Module):

    def __init__(self, model_conf):
        """
        # se3tfmr: with {l0_in_features,l1_in_features,l0_out_features,l1_out_features, 
            num_layers, num_channels, num_degrees, n_heads, num_edge_features}
        """
        super(ReverseDiffusionSE3TFMR, self).__init__()
        self._model_conf = model_conf

        self.l0_scale = model_conf.l0_scale
        self.l1_scale = model_conf.l1_scale

        self.embedding_layer = reverse_se3_diffusion.IndexEmbedder(
            **model_conf.index
        )

        self.torsion_pred = reverse_se3_diffusion.TorsionAngles(
            model_conf.node_embed_size, 1,
        )

        self.se3 = SE3TransformerWrapper(**model_conf.se3_tfmr)

    def get_graph(self, rigids7, node_embed, edge_embed, bb_mask):
        """get_graph builds a DGL graph

        The batched features are used to assemble a single graph with disconnected components 
        associated with each node in the graph.
        
        The first example is indices 0 -- sum(bb_mask[0])-1 etc.

        Args:
            rigids7: rigid frames of shape [B, L, 7]
            node_embed: initial node embeddings of shape [B, L, self.node_embed_size]
            edge_embed: initial edge embeddings of shape [B, L, L, self.edge_embed_size]
            bb_mask: mask of shape [B, L]

        Returns:
            dgl.graph G with 'N' nodes and 'E' edges.
                G.edata['rel_pos']: vectors between ca's of shape [E, 3]
                G.edata['edge_features']: l0 edge features of shape [E, self.edge_embed_size, 1]
                G.ndata['type_0_features']: l0 node features of shape [N, self.node_embed_size, 1]
                G.ndata['type_1_features']: l1 node features of shape [N, 2, 3] corresponding to (c-ca) and (n-ca)
        """
        B = bb_mask.shape[0]
        num_nodes_by_example = bb_mask.sum(dim=-1)
        N = sum(num_nodes_by_example) # total nodes in the graph

        node_l0_by_b = []
        node_l1_by_b = []
        edge_rel_pos_by_b = []
        edge_l0_by_b = []
        i_by_b, j_by_b = [], [] # to and from nodes for edge list
        def all_edges(nodes):
            """all_edges creates and edge list of all directed edges
            in an N node graph excluding self-edges.
            
            Args:
                list of node indices of shape [N]
                
            Returns:
                from edges, to edges of size N*(N-1)
            """
            N = nodes.shape[0]
            nodes = torch.arange(N)
            i = nodes.repeat(N).reshape([-1])
            j = nodes.repeat([N, 1]).T.reshape([-1])
            ij = torch.stack([i, j])
            ij = ij[:, ij[0] != ij[1]]
            return ij[0], ij[1]

        num_res_so_far = 0
        for b, bb_mask_b in enumerate(bb_mask):
            res_idcs = torch.where(bb_mask_b)[0]
            num_res_b = len(res_idcs)

            # Append node features
            node_l0_by_b.append(node_embed[b,res_idcs])
            three_body_coords = rigid_to_3_body(rigids7[b, res_idcs])
            l1_feats = torch.stack([
                three_body_coords[:, 0, :]-three_body_coords[:, 1, :], # N - Ca
                three_body_coords[:, 2, :]-three_body_coords[:, 1, :]  # C - Ca
            ]) # of shape [2, num_res_b, 3]
            # normalize to unit length
            l1_feats /= l1_feats.norm(dim=-1, keepdim=True)


            l1_feats = torch.transpose(l1_feats, 0, 1)
            node_l1_by_b.append(l1_feats)

            # Assemble and append edge features and edge list
            i_b, j_b = all_edges(res_idcs)
            edge_l0_by_b.append(edge_embed[b, i_b, j_b])
            rel_pos = rigids7[b, i_b, 4:] - rigids7[b, j_b, 4:]
            edge_rel_pos_by_b.append(rel_pos)
            i_by_b.append(i_b + num_res_so_far)
            j_by_b.append(j_b + num_res_so_far)

            num_res_so_far += num_res_b


        # concatenate node and edge features across batch items
        node_l0_by_b = torch.concat(node_l0_by_b)
        node_l1_by_b = torch.concat(node_l1_by_b)
        edge_rel_pos_by_b = torch.concat(edge_rel_pos_by_b)
        edge_l0_by_b = torch.concat(edge_l0_by_b)
        i_by_b, j_by_b = torch.concat(i_by_b), torch.concat(j_by_b)
        
        # create graph
        G = dgl.graph((i_by_b, j_by_b), num_nodes=num_res_so_far).to(rigids7.device)
        G.ndata['type_0_features'] = node_l0_by_b[..., None]
        G.ndata['type_1_features'] = node_l1_by_b
        G.edata['rel_pos'] = edge_rel_pos_by_b.detach() # no gradients through basis functions
        G.edata['edge_features'] = edge_l0_by_b[..., None] # need lagging dimension for l0 features

        return G



    def reconstruct_batched_features(self, data, bb_mask):
        """reconstruct_batched_features reconstructs features from single graph into batched form

        Args:
            data: for each unmasked residue of shape [sum(bb_mask), ...]
            bb_mask: mask of shape [B, L]

        Returns:
            data_batched of shape [B, L] + data.shape[1:]
        """
        data_batched = torch.zeros(list(bb_mask.shape) + list(data.shape[1:]), device=data.device)
        num_res_so_far = 0
        for b, bb_mask_b in enumerate(bb_mask):
            num_res_b = int(bb_mask_b.sum())
            # copy data over for item b in the batch
            data_batched[b, torch.where(bb_mask_b)[0]] = \
                data[num_res_so_far:num_res_so_far + num_res_b] 
            num_res_so_far += num_res_b
        
        return data_batched


    def forward(self, input_feats):
        """forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            eps_theta_val: estimate of error for each step shape [B, N, 3]
        """

        # Frames as [batch, res, 7] tensors.
        noisy_local_frames = input_feats['rigids_t'].type(torch.float32)
        res_idx = input_feats['res_idx']
        bb_mask = input_feats['res_mask'].type(torch.float32)  # [B, N]
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        # Padding needs to be unknown aatypes.
        aatype = (
            input_feats['aatype'] * bb_mask
            + residue_constants.unk_restype_index * (1 - bb_mask)
        ).long()

        # Initial embeddings of positonal and relative indices.
        init_node_embed, init_edge_embed = self.embedding_layer(
            res_idx, input_feats['t'], aatype, bb_mask)

        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        # Build graph
        G = self.get_graph(noisy_local_frames, node_embed, edge_embed, bb_mask)

        # Run SE3 tfmr model
        out = self.se3(G=G,
                        type_0_features=G.ndata['type_0_features'],
                        type_1_features=G.ndata['type_1_features'],
                        edge_features=G.edata['edge_features'])

        # Pull out l1 features corresponding to rotation and translation score predictions
        #import pdb; pdb.set_trace()
        node_out_l1 = out['1'] # of shape [sum(bb_mask), 2, 3]
        node_embed = out['0'] # of shape [sum(bb_mask), self.node_embed_size, 0]
        node_embed = node_embed[..., 0] # cut out extra dimension

        # Scale down magnitude
        node_embed /= self._model_conf.l0_scale
        node_out_l1 /= self._model_conf.l1_scale 

        
        node_out_l1 = self.reconstruct_batched_features(node_out_l1, bb_mask)
        node_embed = self.reconstruct_batched_features(node_embed, bb_mask)

        # Pull out and scale the translation and rotation scores
        trans_score = node_out_l1[:, :, 0]
        rot_score = node_out_l1[:, :, 1]

        trans_score *= input_feats['trans_score_norm'][:, None, None]
        rot_score *= input_feats['rot_score_norm'][:, None, None]
        
        # Predict backbon torsion angles
        #if self._model_conf.stop_torsion_grad:
        #    node_embed = node_embed.detach()

        raw_psi, psi_pred = self.torsion_pred(node_embed)
        return {
            'raw_psi': raw_psi,
            'psi': psi_pred,
            'rot_score': rot_score,
            'trans_score': trans_score,
        }