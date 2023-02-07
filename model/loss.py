from openfold.utils import rigid_utils
import torch
import numpy as np
from data import utils as du
from typing import Optional

Rigid = rigid_utils.Rigid


def bb_fape_loss(
        pred_frames: Rigid,
        target_frames: Rigid,
        frames_mask: torch.Tensor,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        positions_mask: torch.Tensor,
        length_scale: float,
        l1_clamp_distance: Optional[float] = None,
        eps=1e-8,
        ignore_nan=True,
    ) -> torch.Tensor:
    """
        Computes FAPE loss.
        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]
    if ignore_nan:
        normed_error = torch.nan_to_num(normed_error)

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = (
        normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    )
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))
    return normed_error


def rigids_fape(pred_frame, target_frame, mask, length_scale=1.0):
    pred_frame = Rigid.from_tensor_7(pred_frame)
    target_frame = Rigid.from_tensor_7(target_frame)
    return bb_fape_loss(
        pred_frames=pred_frame,
        target_frames=target_frame,
        frames_mask=mask,
        positions_mask=mask,
        pred_positions=pred_frame.get_trans(),
        target_positions=target_frame.get_trans(),
        l1_clamp_distance=10.0,
        length_scale=length_scale,
        eps=1e-4,
        ignore_nan=True,
    )


def np_rmsd(x, y, mask=None, per_residue=False):
    """
    Args:
        x: [..., D]
        y: [..., D]
        mask: [...]
    """
    if mask is None:
        mask = np.ones(x.shape[:-1])
    delta = (x - y)**2 * mask[..., None]
    delta = np.sqrt(np.sum(delta, axis=-1))

    if per_residue:
        return delta
    return np.sum(delta) / np.sum(mask)


def torch_rmsd(x, y, mask=None, per_residue=False):
    """
    Args:
        x: [..., D]
        y: [..., D]
        mask: [...]
    """
    if mask is None:
        mask = np.ones(x.shape[:-1])
    delta = (x - y)**2 * mask[..., None]
    if per_residue:
        rmsd_loss = torch.sum(delta, dim=-1)
    else:
        rmsd_loss = torch.sum(delta) 
    return torch.sqrt(
        rmsd_loss / torch.sum(mask) 
    )


def rigids_ca_rmsd(pred_frame, target_frame, res_mask, length_scale=1.0, return_align=False):
    pred_frame = Rigid.from_tensor_7(pred_frame)
    gt_frame = Rigid.from_tensor_7(target_frame)

    pred_ca = pred_frame.get_trans()
    gt_ca = gt_frame.get_trans()

    ca_mask = du.move_to_np(res_mask).astype(bool)
    pred_ca = du.move_to_np(pred_ca)[ca_mask]
    gt_ca = du.move_to_np(gt_ca)[ca_mask]

    aligned_ca, rot, trans, reflection = du.rigid_transform_3D(
        pred_ca, gt_ca)
    rmsd = np_rmsd(aligned_ca, gt_ca) * length_scale 
    if return_align:
        return rmsd, aligned_ca, gt_ca, rot, trans, reflection
    else:
        return rmsd 
