"""SO(3) diffusion methods."""
import numpy as np
import os
from data import utils as du
import logging
import torch
from data import so3_utils, igso3


class SO3Diffuser:

    def __init__(self, so3_conf):
        self.schedule = so3_conf.schedule

        self.min_sigma = so3_conf.min_sigma
        self.max_sigma = so3_conf.max_sigma

        self.num_sigma = so3_conf.num_sigma
        self._log = logging.getLogger(__name__)

        # Discretize omegas for calculating CDFs. Skip omega=0.
        self.igso3 = igso3.IGSO3(min_t=self.min_sigma**2, max_t=self.max_sigma**2,
                L=2000, num_ts=self.num_sigma, num_omegas=so3_conf.num_omega,
                cache_dir=so3_conf.cache_dir)

        self._score_scaling = np.sqrt(np.abs(
            np.sum(
                self.igso3._d_logf_d_omega**2 * self.igso3._pdf_angle, axis=-1) / np.sum(
                    self.igso3._pdf_angle, axis=-1)
        )) / np.sqrt(3)

    @property
    def discrete_sigma(self):
        return self.sigma(np.linspace(0.0, 1.0, self.num_sigma))

    def sigma_idx(self, sigma: np.ndarray):
        """Calculates the index for discretized sigma during IGSO(3) initialization."""
        return np.digitize(sigma, self.discrete_sigma) - 1

    def sigma(self, t: np.ndarray):
        """Extract \sigma(t) corresponding to chosen sigma schedule."""
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        if self.schedule == 'logarithmic':
            return np.log(t * np.exp(self.max_sigma) + (1 - t) * np.exp(self.min_sigma))
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')

    def diffusion_coef(self, t):
        """Compute diffusion coefficient (g_t)."""
        if self.schedule == 'logarithmic':
            g_t = np.sqrt(
                2 * (np.exp(self.max_sigma) - np.exp(self.min_sigma)) * self.sigma(t) / np.exp(self.sigma(t))
            )
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')
        return g_t


    def sample_ref(self, n_samples: float=1):
        return so3_utils.sample_uniform(n_samples)

    def torch_score(self, R: torch.tensor, R_0: torch.tensor, t: torch.tensor,
            eps: float=1e-6):
        """Computes the score of IGSO(3) density

        grad_R log IGSO3(R ; R_0, t)

        Args:
            R: [..., 3, 3] array of rotation matrices at which to compute the score.
            R_0: [..., 3, 3] initial rotations used for IGSO3 location
                parameter.
            t: continuous time in [0, 1].

        Returns:
            [..., 3, 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        R_0t = torch.einsum('...ji,...jk->...ik', R_0, R) # compute R_0^T R
        score = torch.einsum(
                '...ij,...jk->...ik',
                R_0, self.igso3.score(R_0t, t, eps))
        return score

    def score_scaling(self, t: np.ndarray):
        """Calculates scaling used for scores during trianing."""
        return self._score_scaling[self.igso3.t_idx(t)]

    def forward_marginal(self, R_0: np.ndarray, t: float):
        """Samples from the forward diffusion process at time index t.

        Args:
            R_0: [..., 3, 3] initial rotations.
            t: continuous time in [0, 1].

        Returns:
            R_t: [..., 3, 3] noised rotation vectors.
            rot_score: [..., 3, 3] score of rot_t as a rotation vector.
        """
        n_samples = np.cumprod(R_0.shape[:-2])[-1]
        R_0_ = torch.tensor(R_0, dtype=torch.float64) # rotvec to matrix
        sampled_rots = self.igso3.sample(t, n_samples=n_samples)
        R_t = torch.einsum('...ij,...jk->...ik', sampled_rots, R_0_)
        rot_score = self.torch_score(R_t, R_0_, torch.tensor(t)).reshape(R_0_.shape)
        rot_score = rot_score.numpy().astype(R_0.dtype)
        R_t = R_t.numpy().astype(R_0.dtype)
        return R_t, rot_score

    def reverse(
            self,
            R_t: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
            mask: np.ndarray=None,
            noise_scale: float=1.0,
            ):
        """Simulates the reverse SDE for 1 step using the Geodesic random walk.

        Args:
            rot_t: [..., 3] current rotations at time t.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            add_noise: set False to set diffusion coefficent to 0.
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] rotation vector at next step.
        """
        if not np.isscalar(t): raise ValueError(f'{t} must be a scalar.')

        # Convert to pytorch tensors
        R_t_, score_t_ = torch.tensor(R_t), torch.tensor(score_t)
        g_t = self.diffusion_coef(t)
        perturb = ( g_t ** 2 ) * dt * score_t_ + noise_scale * g_t * np.sqrt(dt) * so3_utils.tangent_gaussian(R_t_)
        if mask is not None: perturb *= mask[..., None, None]
        R_t_1 = so3_utils.expmap(R_t_, perturb.to(R_t_.dtype)).numpy().astype(R_t.dtype)
        return R_t_1
