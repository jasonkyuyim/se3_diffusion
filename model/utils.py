import torch

def calc_rbf(dists, min_dist, max_dist, num_rbf):
    # Distance radial basis function
    dist_mu = torch.linspace(min_dist, max_dist, num_rbf).to(dists.device)
    dist_mu = dist_mu.view([1,1,1,-1])
    dist_sigma = (max_dist - min_dist) / num_rbf
    dist_expand = torch.unsqueeze(dists, -1)
    rbf = torch.exp(-((dist_expand - dist_mu) / dist_sigma)**2)
    return rbf


def calc_neighbors(
        ca_pos, node_mask, edge_mask, num_neighbors, rigid_tensors):
    batch_size, num_res = node_mask.shape
    dists_2d = torch.linalg.norm(
        ca_pos[:, :, None, :] - ca_pos[:, None, :, :], axis=-1)

    # Add bias to self-edges and masked residues.
    dists_2d += torch.eye(num_res).to(node_mask.device)[None]*1e5
    dists_2d += (1 - edge_mask)*1e5

    # Calculate local graph indices.
    _, local_graph = torch.topk(
        dists_2d, k=num_neighbors, dim=-1, largest=False)
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
    local_rigids = rigid_tensors[(i_idx, j_idx)].reshape(
        batch_size, num_res, num_neighbors, 7)
    return local_graph, local_edge_mask, local_rigids


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
