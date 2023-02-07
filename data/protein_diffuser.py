"""Protein diffusion methods."""
from re import T
import numpy as np
from data import se3_diffuser
from data import simplex_diffuser
from data import utils as du
import torch
from openfold.utils import rigid_utils as ru


class ProteinDiffuser:
    """Full protein diffuser class.
    
    Diffuser over protein sequence and structure.
    """

    def __init__(self, diff_conf):
        self._diff_conf = diff_conf

        self._se3_diffuser = se3_diffuser.SE3Diffuser(diff_conf.se3)
        self._seq_diffuser = simplex_diffuser.SimplexDiffuser(diff_conf.seq)

    @property
    def conf(self):
        """Diffuser conf."""
        return self._diff_conf

    @property
    def seq_diffuser(self):
        """Sequence diffuser."""
        return self._seq_diffuser

    @property
    def se3_diffuser(self):
        """Structure diffuser."""
        return self._se3_diffuser

    def forward_marginal(
            self,
            *,
            rigids_0: ru.Rigid,
            aatype_probs_0: np.ndarray,
            t_seq: float,
            t_struct: float,
            diffuse_mask: np.ndarray=None,
        ):
        """Sample at time (t) for protein SDEs.

        Args:
            rigids_0: [..., N] true rigid bodies.
            aatype_probs_0: [..., N, 21] aatype probabilities.
            t_seq: continuous time in [0, 1] for sequence SDE.
            t_struct: continuous time in [0, 1] for structure SDE.
            diffuse_mask: [..., N] which residues to diffuse.

        Returns: dict with the following keys.
            aatype_sde_t: [..., N, 20] underlying VP-SDE values for sequence diffuser.
            aatype_probs_t: [..., N, 21] sampled amino-acid type probability vector.
            rigid_t: [..., N, 7] sampled rigid object from SE(3) diffuser.
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3] rotation score
            trans_score_scaling: [...] translation score scaling
            rot_score_scaling: [...] rotation score scaling
        """
        se3_sample = self._se3_diffuser.forward_marginal(
            rigids_0, t_struct, diffuse_mask=diffuse_mask)
        seq_sample = self._seq_diffuser.forward_marginal(
            aatype_probs_0, t_seq, diffuse_mask=diffuse_mask)
        return {**seq_sample, **se3_sample}


    def se3_score_scaling(self, t):
        """Score rescaling factors for SE(3) diffuser at time (t).

        Args:
            t: continuous time in [0, 1] for SDE.

        Returns:
            trans_score_scaling: [...] translation score scaling
            rot_score_scaling: [...] rotation score scaling
        """
        return self._se3_diffuser.score_scaling(t)

    def calc_trans_0(self, trans_score, trans_t, t):
        return self._se3_diffuser.calc_trans_0(trans_score, trans_t, t)

    def calc_trans_score(self, trans_t, trans_0, t, use_torch=False, scale=True):
        return self._se3_diffuser._r3_diffuser.score(
            trans_t, trans_0, t, use_torch=use_torch, scale=scale)

    def trans_parameters(self, trans_t, score_t, t, dt, mask):
        return self._se3_diffuser._r3_diffuser.distribution(
            trans_t, score_t, t, dt, mask)

    def calc_rot_sore(self, rots_t, rots_0, t):
        rots_0_inv = rots_0.invert()
        quats_0_inv = rots_0_inv.get_quats()
        quats_t = rots_t.get_quats()
        quats_0t = ru.quat_multiply(quats_0_inv, quats_t)
        rotvec_0t = du.quat_to_rotvec(quats_0t)
        return self._se3_diffuser._so3_diffuser.torch_score(rotvec_0t, t)

    def reverse(
            self,
            *,
            rigid_t: ru.Rigid,
            rot_score: np.ndarray,
            trans_score: np.ndarray,
            aatype_sde_t: np.ndarray,
            aatype_probs_0: np.ndarray,
            t: float,
            dt: float,
            diffuse_mask: np.ndarray=None,
            center: bool=True,
            ode: bool=False,
            noise_scale: float=1.0,
        ):
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            aatype_sde_t: [..., N, 20] current sequence SDE state.
            aatype_probs_0: [..., N, 21] amino-acid probabilities.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            diffuse_mask: [..., N] which residues to diffuse.
            add_noise: flag to add noise term when sampling.
            center: true to set center of mass to zero after step
            ode: true to us deterministic dynamics

        Returns:
            rigid_t_1: [..., N] next (t-1) rigid objects.
            aatype_sde_t_1: [..., N] next (t-1) sequence SDE state.
            aatype_t_1: [..., N] next (t-1) amino-acid probabilities.
        """
        rigids_t_1 = self._se3_diffuser.reverse(
            rigid_t,
            rot_score,
            trans_score,
            t,
            dt,
            diffuse_mask=diffuse_mask,
            center=center,
            ode=ode,
            noise_scale=noise_scale,
        )
        aatype_sde_t_1, aatype_t_1 = self._seq_diffuser.reverse(
            x_t=aatype_sde_t,
            y_0=aatype_probs_0,
            t=t,
            dt=dt,
            ode=ode,
            diffuse_mask=diffuse_mask,
        )
        return rigids_t_1, aatype_sde_t_1, aatype_t_1

    def sample_ref(
            self,
            *,
            n_samples: int,
            rigids_impute: ru.Rigid=None,
            aatype_impute: np.ndarray=None,
            diffuse_mask: np.ndarray=None,
            as_tensor_7: bool=False,
        ):
        """Samples protein from reference distribution.

        Args:
            n_samples: Number of samples.
            rigids_impute: [..., N] Rigid objects to use as imputation values if either
                translations or rotations are not diffused.
            aatype_impute: [..., N, 21] Amino-acids to use as imputation values if amino
                acids are not diffused.
            diffuse_mask: [..., N] which residues to diffuse.
            as_tensor_7: whether to return rigids as a tensor.

        Returns:
            rigid_1: [..., N] (or [..., N, 7]) time (t=1) rigid objects.
            aatype_sde_1: [..., N] time (t=1) sequence SDE state.
            aatype_1: [..., N] time (t=1) amino-acid probabilities.
        """
        ref_se3_sample = self._se3_diffuser.sample_ref(
            n_samples=n_samples,
            impute=rigids_impute,
            diffuse_mask=diffuse_mask,
            as_tensor_7=as_tensor_7
        )
        ref_aatype_sample = self._seq_diffuser.sample_ref(
            n_samples=n_samples,
            impute=aatype_impute,
            diffuse_mask=diffuse_mask
        )
        return {
            **ref_se3_sample,
            **ref_aatype_sample,
        }

