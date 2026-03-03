"""SE(3) flow matching methods (rectified flow)."""
import numpy as np
from data import so3_flow_matcher
from data import r3_flow_matcher
from scipy.spatial.transform import Rotation
from openfold.utils import rigid_utils as ru
from data import utils as du
import torch
import logging


def _extract_trans_rots(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    rot_shape = rot.shape
    num_rots = np.cumprod(rot_shape[:-2])[-1]
    rot = rot.reshape((num_rots, 3, 3))
    rot = Rotation.from_matrix(rot).as_rotvec().reshape(rot_shape[:-2] + (3,))
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot


def _assemble_rigid(rotvec, trans):
    rotvec_shape = rotvec.shape
    num_rotvecs = np.cumprod(rotvec_shape[:-1])[-1]
    rotvec = rotvec.reshape((num_rotvecs, 3))
    rotmat = Rotation.from_rotvec(rotvec).as_matrix().reshape(
        rotvec_shape[:-1] + (3, 3))
    return ru.Rigid(
        rots=ru.Rotation(
            rot_mats=torch.Tensor(rotmat)),
        trans=torch.tensor(trans))


class SE3FlowMatcher:
    """SE(3) flow matcher combining R(3) and SO(3) flow matching."""

    def __init__(self, se3_conf):
        self._log = logging.getLogger(__name__)
        self._se3_conf = se3_conf

        self._diffuse_rot = se3_conf.diffuse_rot
        self._so3_fm = so3_flow_matcher.SO3FlowMatcher(self._se3_conf.so3)

        self._diffuse_trans = se3_conf.diffuse_trans
        self._r3_fm = r3_flow_matcher.R3FlowMatcher(self._se3_conf.r3)

    def forward_marginal(
            self,
            rigids_0: ru.Rigid,
            t: float,
            diffuse_mask: np.ndarray = None,
            as_tensor_7: bool = True,
    ):
        """Forward interpolation at time t.

        Args:
            rigids_0: [..., N] openfold Rigid objects.
            t: continuous time in [0, 1].

        Returns:
            rigids_t: noised rigid.
            trans_velocity: [..., N, 3] translation velocity target.
            rot_velocity: [..., N, 3] rotation velocity target.
        """
        trans_0, rot_0 = _extract_trans_rots(rigids_0)

        if not self._diffuse_rot:
            rot_t, rot_velocity = rot_0, np.zeros_like(rot_0)
        else:
            rot_t, rot_velocity = self._so3_fm.forward_marginal(rot_0, t)

        if not self._diffuse_trans:
            trans_t, trans_velocity = trans_0, np.zeros_like(trans_0)
        else:
            trans_t, trans_velocity = self._r3_fm.forward_marginal(trans_0, t)

        if diffuse_mask is not None:
            rot_t = self._apply_mask(rot_t, rot_0, diffuse_mask[..., None])
            trans_t = self._apply_mask(trans_t, trans_0, diffuse_mask[..., None])
            trans_velocity = self._apply_mask(
                trans_velocity, np.zeros_like(trans_velocity),
                diffuse_mask[..., None])
            rot_velocity = self._apply_mask(
                rot_velocity, np.zeros_like(rot_velocity),
                diffuse_mask[..., None])

        rigids_t = _assemble_rigid(rot_t, trans_t)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()

        # Use same key names as SE3Diffuser for compatibility
        # rot_score/trans_score now contain velocities instead of scores
        rot_velocity_scaling = self._so3_fm.score_scaling(t)
        trans_velocity_scaling = self._r3_fm.score_scaling(t)
        return {
            'rigids_t': rigids_t,
            'rot_score': rot_velocity,
            'trans_score': trans_velocity,
            'rot_score_scaling': rot_velocity_scaling,
            'trans_score_scaling': trans_velocity_scaling,
        }

    def calc_trans_0(self, velocity, trans_t, t):
        return self._r3_fm.calc_trans_0(velocity, trans_t, t)

    def calc_trans_score(self, trans_t, trans_0, t, use_torch=False, scale=True):
        """Compute velocity between trans_t and trans_0.

        For flow matching: v = (x_t - x_0) / t
        Uses 'score' naming for backward compatibility.
        """
        return self._r3_fm.velocity(
            trans_t, trans_0, t, use_torch=use_torch, scale=scale)

    def calc_rot_score(self, rots_t, rots_0, t):
        """Compute rotation velocity between rots_t and rots_0.

        For flow matching: omega = Log(R_0^{-1} * R_t) / t
        Uses 'score' naming for backward compatibility.
        """
        rots_0_inv = rots_0.invert()
        quats_0_inv = rots_0_inv.get_quats()
        quats_t = rots_t.get_quats()
        quats_0t = ru.quat_multiply(quats_0_inv, quats_t)
        rotvec_0t = du.quat_to_rotvec(quats_0t)
        return self._so3_fm.torch_velocity(rotvec_0t, t)

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def trans_parameters(self, trans_t, velocity, t, dt, mask):
        return self._r3_fm.distribution(trans_t, velocity, t, mask, dt)

    def score(self, rigid_0, rigid_t, t):
        """Compute velocity (named 'score' for compatibility)."""
        tran_0, rot_0 = _extract_trans_rots(rigid_0)
        tran_t, rot_t = _extract_trans_rots(rigid_t)

        if not self._diffuse_rot:
            rot_velocity = np.zeros_like(rot_0)
        else:
            # omega = Log(R_0^{-1} * R_t) / t
            n = np.prod(rot_0.shape[:-1])
            R_0 = Rotation.from_rotvec(rot_0.reshape(n, 3))
            R_t = Rotation.from_rotvec(rot_t.reshape(n, 3))
            R_rel = R_0.inv() * R_t
            omega = R_rel.as_rotvec().reshape(rot_0.shape)
            rot_velocity = omega / (t + 1e-6)

        if not self._diffuse_trans:
            trans_velocity = np.zeros_like(tran_0)
        else:
            trans_velocity = self._r3_fm.velocity(
                tran_t, tran_0, t, use_torch=False, scale=True)

        return trans_velocity, rot_velocity

    def score_scaling(self, t):
        """Returns (1, 1) — no scaling needed for flow matching."""
        rot_scaling = self._so3_fm.score_scaling(t)
        trans_scaling = self._r3_fm.score_scaling(t)
        return rot_scaling, trans_scaling

    def reverse(
            self,
            rigid_t: ru.Rigid,
            rot_score: np.ndarray,
            trans_score: np.ndarray,
            t: float,
            dt: float,
            diffuse_mask: np.ndarray = None,
            center: bool = True,
            noise_scale: float = 1.0,
    ):
        """ODE reverse step from t to t-dt.

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3] rotation velocity (named 'score' for compat).
            trans_score: [..., N, 3] translation velocity.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            diffuse_mask: [..., N] which residues to update.
            center: set center of mass to zero after step.
            noise_scale: unused (deterministic ODE).
        """
        trans_t, rot_t = _extract_trans_rots(rigid_t)

        if not self._diffuse_rot:
            rot_t_1 = rot_t
        else:
            rot_t_1 = self._so3_fm.reverse(
                rot_t=rot_t,
                velocity=rot_score,
                t=t,
                dt=dt,
            )

        if not self._diffuse_trans:
            trans_t_1 = trans_t
        else:
            trans_t_1 = self._r3_fm.reverse(
                x_t=trans_t,
                velocity=trans_score,
                t=t,
                dt=dt,
                center=center,
            )

        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(
                trans_t_1, trans_t, diffuse_mask[..., None])
            rot_t_1 = self._apply_mask(
                rot_t_1, rot_t, diffuse_mask[..., None])

        return _assemble_rigid(rot_t_1, trans_t_1)

    def sample_ref(
            self,
            n_samples: int,
            impute: ru.Rigid = None,
            diffuse_mask: np.ndarray = None,
            as_tensor_7: bool = False,
    ):
        """Sample from reference distribution (t=1)."""
        if impute is not None:
            assert impute.shape[0] == n_samples
            trans_impute, rot_impute = _extract_trans_rots(impute)
            trans_impute = trans_impute.reshape((n_samples, 3))
            rot_impute = rot_impute.reshape((n_samples, 3))
            trans_impute = self._r3_fm._scale(trans_impute)

        if diffuse_mask is not None and impute is None:
            raise ValueError('Must provide imputation values.')
        if (not self._diffuse_rot) and impute is None:
            raise ValueError('Must provide imputation values.')
        if (not self._diffuse_trans) and impute is None:
            raise ValueError('Must provide imputation values.')

        if self._diffuse_rot:
            rot_ref = self._so3_fm.sample_ref(n_samples=n_samples)
        else:
            rot_ref = rot_impute

        if self._diffuse_trans:
            trans_ref = self._r3_fm.sample_ref(n_samples=n_samples)
        else:
            trans_ref = trans_impute

        if diffuse_mask is not None:
            rot_ref = self._apply_mask(
                rot_ref, rot_impute, diffuse_mask[..., None])
            trans_ref = self._apply_mask(
                trans_ref, trans_impute, diffuse_mask[..., None])

        trans_ref = self._r3_fm._unscale(trans_ref)
        rigids_t = _assemble_rigid(rot_ref, trans_ref)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {'rigids_t': rigids_t}
