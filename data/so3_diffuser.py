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
                L=500, num_ts=self.num_sigma, num_omegas=so3_conf.num_omega, cache_dir=so3_conf.cache_dir)

        self._score_scaling = np.sqrt(np.abs(
            np.sum(
                self.igso3._d_logf_d_omega**2 * self.igso3._pdf_angle, axis=-1) / np.sum(
                    self.igso3._pdf_angle, axis=-1)
        )) / np.sqrt(3)

    @property
    def discrete_sigma(self):
        return self.sigma(
            np.linspace(0.0, 1.0, self.num_sigma)
        )

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
        return so3_utils.Log(so3_utils.sample_uniform(n_samples)).numpy()

    def torch_score(
            self,
            vec: torch.tensor,
            t: torch.tensor,
            eps: float=1e-6
        ):
        """Computes the score of IGSO(3) density as a rotation vector.


        Same as score function but uses pytorch and performs a look-up.

        Args:
            vec: [..., 3] array of axis-angle rotation vectors.
            t: continuous time in [0, 1].

        Returns:
            [..., 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        # TODO: Change this to use self.igso3.score (which actually returns
        # something in the tangent space)
        omega = torch.linalg.norm(vec, dim=-1)
        score_norms_t = self.igso3._d_logf_d_omega[self.igso3.t_idx(du.move_to_np(t))]
        score_norms_t = torch.tensor(score_norms_t).to(vec.device)
        omega_idx = torch.bucketize(
            omega, torch.tensor(self.igso3._discrete_omegas[:-1]).to(vec.device))
        omega_score_t = torch.gather(
            score_norms_t, 1, omega_idx)
        return omega_score_t[..., None] * vec / (omega[..., None] + eps)

    def score_scaling(self, t: np.ndarray):
        """Calculates scaling used for scores during trianing."""
        return self._score_scaling[self.igso3.t_idx(t)]

    def forward_marginal(self, rots_0: np.ndarray, t: float):
        """Samples from the forward diffusion process at time index t.

        Args:
            rots_0: [..., 3] initial rotations.
            t: continuous time in [0, 1].

        Returns:
            rot_t: [..., 3] noised rotation vectors.
            rot_score: [..., 3] score of rot_t as a rotation vector.
        """
        # TODO change this to take and return 3x3 rotation matrices
        n_samples = np.cumprod(rots_0.shape[:-1])[-1]
        sampled_rots = self.sample(t, n_samples=n_samples)
        rot_score = self.score(sampled_rots, t).reshape(rots_0.shape)

        # Left multiply.
        rot_t = du.compose_rotvec(sampled_rots, rots_0).reshape(rots_0.shape)
        return rot_t, rot_score

    def reverse(
            self,
            rot_t: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
            mask: np.ndarray=None,
            noise_scale: float=1.0,
            ):
        """Simulates the reverse SDE for 1 step

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
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        g_t = self.diffusion_coef(t)
        z = noise_scale * np.random.normal(size=score_t.shape)
        perturb = (g_t ** 2) * score_t * dt + g_t * np.sqrt(dt) * z

        if mask is not None: perturb *= mask[..., None]
        n_samples = np.cumprod(rot_t.shape[:-1])[-1]

        # TODO: change to use expmap that's easily identifiable as a sigle step
        # of geodesic random walk.
        # Left multiply.
        rot_t_1 = du.compose_rotvec(
            perturb.reshape(n_samples, 3),
            rot_t.reshape(n_samples, 3),
        ).reshape(rot_t.shape)
        return rot_t_1
