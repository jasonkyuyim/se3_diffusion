"""R^3 diffusion methods."""
import numpy as np
from data import utils as du


class R3Diffuser:
    """VE-SDE diffuser class for translations."""

    def __init__(
            self,
            min_sigma,
            max_sigma,
            num_t=1000,
            align_t=False,
            schedule='linear',
        ):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma 
        self._discrete_t = np.linspace(1e-5, 1, num_t)
        self.schedule = schedule
        self.align_t = align_t

        # Simulate score of marginals
        loc = np.zeros((num_t, 3))
        exp_score_norms = []
        self._num_t = num_t
        for t in self._discrete_t:
            _, score_t = self.forward_marginal(loc, t, score_norm=False)
            scores_norm_t = np.linalg.norm(score_t, axis=-1)
            exp_score_norms.append(np.mean(scores_norm_t))
        self._exp_score_norm = np.array(exp_score_norms)

    def sigma_t(self, t):
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        if self.schedule == 'linear': 
            return self.min_sigma * (self.max_sigma/self.min_sigma)**t
        else:
            raise ValueError(f'Unknown schedule {self.schedule}')
    
    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return np.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t) * x

    def sample_ref(self, n_samples: float=1):
        return np.random.normal(size=(n_samples, 3))

    def forward(
            self,
            x_t: np.ndarray,
            t: float,
            t_1: float):
        """Samples time t-1 from the forward process given time t.

        Args:
            t: continuous time in [0, 1]. 
            dt: time step interval.
            x_t: [..., 3] current positions at time t.

        Returns:
            x_t_1: [..., 3] positions at time t-1.
        """
        z = np.random.normal(size=x_t.shape)
        return x_t + np.sqrt(self.sigma_t(t)**2 - self.sigma_t(t_1)**2)*z

    def marginal_b_t(self, t):
        if self.schedule == 'linear':
            return t*self.min_b + (1/2)*(t**2)*(self.max_b-self.min_b)
        else:
            raise ValueError(f'Unknown schedule {self.schedule}')

    def forward_marginal(self, x_0: np.ndarray, t: float, score_norm=True):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions.
            t: continuous time in [0, 1]. 

        Returns:
            x_t: [..., n, 3] positions at time t.
            score_t: [..., n, 3] score at time t.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t = np.random.normal(
            loc=x_0,
            scale=self.sigma_t(t)
        )
        if self.align_t:
            x_t, _, _, _ = du.rigid_transform_3D(x_t, x_0)
        score_t = self.score(x_t, x_0, t)
        if score_norm:
            score_norm_t = self.exp_score_norm(t)
            return x_t, score_t, score_norm_t
        return x_t, score_t

    def exp_score_norm(self, t: float):
        return self._exp_score_norm[
            min(np.sum(self._discrete_t <= t), self._num_t-1)]

    def reverse(
            self,
            x_t: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
            add_noise: bool=True,
            mask: np.ndarray=None,
            center: bool=True):
        """Simulates the reverse SDE for 1 step

        Args:
            rot_t: [..., 3] current positions at time t.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            add_noise: set False to set diffusion coefficent to 0.
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        if add_noise:
            z = np.random.normal(size=score_t.shape)
        else:
            z = np.zeros_like(score_t)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(dt) * z
        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = np.ones(x_t.shape[:-1])
        x_t_1 = x_t - perturb
        if center:
            com = np.sum(x_t_1, axis=-2) / np.sum(mask, axis=-1)[..., None]
            x_t_1 -= com[..., None, :]
        return x_t_1

    def score(self, x_t, x_0, t):
        return -(x_t - np.exp(-1/2*self.marginal_b_t(t)) * x_0) /(
            1 - np.exp(-self.marginal_b_t(t)))
