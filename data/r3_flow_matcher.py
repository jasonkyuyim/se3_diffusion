"""R^3 flow matching methods (rectified flow)."""
import numpy as np
import torch


class R3FlowMatcher:
    """Flow matching for translations in R^3.

    Uses linear interpolation between data and noise:
        x_t = (1 - t) * x_0 + t * eps,   eps ~ N(0, I)
    Model predicts velocity v = eps - x_0.
    """

    def __init__(self, r3_conf):
        self._r3_conf = r3_conf
        self.coordinate_scaling = r3_conf.coordinate_scaling

    def _scale(self, x):
        return x * self.coordinate_scaling

    def _unscale(self, x):
        return x / self.coordinate_scaling

    def sample_ref(self, n_samples: int = 1):
        return np.random.normal(size=(n_samples, 3))

    def forward_marginal(self, x_0: np.ndarray, t: float):
        """Samples from the forward interpolation at time t.

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] interpolated positions in Angstroms.
            velocity: [..., n, 3] target velocity (eps - x_0) in scaled coords.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_0 = self._scale(x_0)
        eps = np.random.normal(size=x_0.shape)
        x_t = (1 - t) * x_0 + t * eps
        velocity = eps - x_0
        x_t = self._unscale(x_t)
        return x_t, velocity

    def calc_trans_0(self, velocity, x_t, t, use_torch=True):
        """Recover x_0 from velocity prediction.

        x_t = (1-t)*x_0 + t*eps  and  v = eps - x_0
        => x_0 = x_t - t * v  (in scaled coords)
        => eps = x_0 + v
        """
        if use_torch:
            t = t[..., None, None]
        x_t = self._scale(x_t)
        x_0 = (x_t - t * velocity) / (1 - t + 1e-8)
        return self._unscale(x_0)

    def velocity(self, x_t, x_0, t, use_torch=False, scale=True):
        """Compute ground truth velocity from x_t and x_0.

        v = (eps - x_0) where eps = (x_t - (1-t)*x_0) / t
        Simplifies to: v = (x_t - x_0) / t  (but we compute directly)

        Actually: v = eps - x_0, and x_t = (1-t)*x_0 + t*eps
        => eps = (x_t - (1-t)*x_0) / t
        => v = (x_t - (1-t)*x_0) / t - x_0 = (x_t - x_0) / t
        """
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        if use_torch:
            return (x_t - x_0) / (t + 1e-8)
        else:
            return (x_t - x_0) / (t + 1e-8)

    def score_scaling(self, t: float):
        """No score scaling needed for flow matching. Returns 1."""
        if np.isscalar(t):
            return np.ones_like(np.array(t))
        return np.ones_like(t)

    def reverse(
            self,
            *,
            x_t: np.ndarray,
            velocity: np.ndarray,
            t: float,
            dt: float,
            mask: np.ndarray = None,
            center: bool = True,
            noise_scale: float = 1.0,
    ):
        """ODE step from t to t-dt.

        x_{t-dt} = x_t - velocity * dt

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            velocity: [..., 3] predicted velocity.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to update.
            center: center of mass to zero after step.
            noise_scale: unused (kept for API compatibility).
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t = self._scale(x_t)
        perturb = velocity * dt

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = np.ones(x_t.shape[:-1])

        x_t_1 = x_t - perturb
        if center:
            com = np.sum(x_t_1, axis=-2) / np.sum(mask, axis=-1)[..., None]
            x_t_1 -= com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def forward(self, x_t_1: np.ndarray, t: float, num_t: int):
        """Single forward step (for trajectory visualization)."""
        x_t_1 = self._scale(x_t_1)
        dt = 1.0 / num_t
        z = torch.tensor(np.random.normal(size=x_t_1.shape)).to(x_t_1.device)
        x_t = x_t_1 + dt * z
        return x_t

    def distribution(self, x_t, velocity, t, mask, dt):
        """Return mean and std of the reverse step distribution.

        For flow matching the reverse is deterministic (ODE), so std=0.
        We return a small std for compatibility.
        """
        x_t = self._scale(x_t)
        mu = x_t - velocity * dt
        if mask is not None:
            mu *= mask[..., None]
        std = 0.0
        return mu, std
