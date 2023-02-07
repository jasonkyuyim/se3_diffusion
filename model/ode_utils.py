import torch
from openfold.utils import rigid_utils as ru
from data import utils as du
from scipy.spatial.transform import Rotation as scipy_R

def center_x_t_6D(x_t_as_6D, mask):
    """center_x_t_6D takes a structure represented as translations 
    and rotation vector and centers to have center of mass at zero.

    Args:
        x_t_as_6D: shape [..., 6], with [..., :3] as translations and [..., 3:] as rotation vector
        mask: zero where hidden.

    Returns: x_t_as_6D but centered at zero
    """
    trans = x_t_as_6D[..., :3]
    #com = torch.sum(trans, dim=-2) / torch.sum(mask, dim=-1)[..., None]
    com = torch.mean(trans[torch.where(mask)], dim=-2)
    x_t_as_6D[..., :3] -= mask[..., None]*com[..., None, :]
    return x_t_as_6D

def tensor7_to_6D(tensor_7):
    rot_mats  = ru.Rigid.from_tensor_7(tensor_7).get_rots().get_rot_mats()
    rot_vecs = torch.tensor(scipy_R.from_matrix(
        rot_mats.detach().cpu().numpy()).as_rotvec()).to(rot_mats.device)

    # compute 6D representation of shape [L, 6]
    x_t_as_6D = torch.cat([tensor_7[...,  4:], rot_vecs], dim=-1)
    return x_t_as_6D


def x_t_6D_to_tensor7(x_t_as_6D):
    """x_t_6D_to_tensor7 converts 6D frame representation with rotation vector to 7D using quaternion

    Args:
        x_t_as_6D: shape [..., 6], with [..., :3] as translations and [..., 3:] as rotation vector
    
    Returns:
        tensor7 with quats first, followed by translations (following the convention in openfold)
    """
    trans = x_t_as_6D[..., :3]
    rotvecs = x_t_as_6D[..., 3:]
    quats = rotvec_to_quaternion(rotvecs)
    tensor7 = torch.cat([quats, trans], dim=-1)
    return tensor7


def x_t_6D_to_rigid(x_t_as_6D):
    """x_t_6D_to_rigid computes 7D rigid object from 6D representation with rotation vector.
    """
    tensor7 = x_t_6D_to_tensor7(x_t_as_6D)
    rigid = ru.Rigid.from_tensor_7(tensor7)
    return rigid

def rotvec_to_quaternion(rotvec):
    """rotvec_to_quaternion

    Torch conversion

    Args:
        rotvec: with shape [... 3]
    
    Returns:
        quat with shape [..., 4], dimension 0 is scaling and quat[...,1:4] is in direction of rotation vector
    """
    # compute angle of rotation
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    rot_axis = rotvec/theta
    q0 = torch.cos(theta/2)
    q123 = rot_axis*torch.sin(theta/2)
    quat = torch.cat([q0, q123], dim=-1)
    return quat

def compose_6D(frame_6D_1, frame_6D_2):
    """compose_6D applies composes the rigids frames
    (separately for rotations and translations)

    Args:
        frame_6D_1, frame_6D_2: 6d representation of frame / perturbation with shape [..., 6],
            translations followed by rotation.

    The translations (first 3 dimensions) are added.  The order of composition for rotations
    (left vs right) depends on whether or not we use an equivariant rotation score.

    Returns:
        next_as_6D of shape [..., 6]
    """
    trans_1, rots_1 = frame_6D_1[..., :3], frame_6D_1[..., 3:]
    trans_2, rots_2 = frame_6D_2[..., :3], frame_6D_2[..., 3:]

    # compose translations
    trans_next = trans_1 + trans_2

    # compose_rotvec converts to numpy, and so must be on CPU
    rots_1, rots_2 = rots_1.cpu().detach(), rots_2.cpu().detach()
    rots_next = du.compose_rotvec(rots_1, rots_2)
    rots_next = torch.tensor(rots_next).to(frame_6D_1.device)

    next_as_6D = torch.cat([
        trans_next,
        rots_next
    ], dim=-1)
    return next_as_6D
