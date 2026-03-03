"""SO(3) flow matching methods (rectified flow with geodesic interpolation)."""
import numpy as np
import torch
import logging
from scipy.spatial.transform import Rotation


class SO3FlowMatcher:
    """Flow matching for rotations on SO(3).

    Uses geodesic (SLERP) interpolation between data and noise:
        R_t = R_0 * Exp(t * Log(R_0^{-1} * R_1)),  R_1 ~ Uniform(SO(3))
    Model predicts tangent velocity omega = Log(R_0^{-1} * R_1).
    """

    def __init__(self, so3_conf):
        self._log = logging.getLogger(__name__)
        self._so3_conf = so3_conf

    def sample_ref(self, n_samples: int = 1):
        """Sample uniform rotations as rotation vectors."""
        return Rotation.random(n_samples).as_rotvec()

    def _uniform_so3(self, n_samples: int):
        """Sample uniform rotations on SO(3)."""
        return Rotation.random(n_samples)

    def forward_marginal(self, rot_0: np.ndarray, t: float):
        """Geodesic interpolation on SO(3) from rot_0 to a random rotation.

        R_t = R_0 * Exp(t * omega)
        where omega = Log(R_0^{-1} * R_1), R_1 ~ Uniform(SO(3))

        Args:
            rot_0: [..., 3] initial rotation vectors.
            t: continuous time in [0, 1].

        Returns:
            rot_t: [..., 3] interpolated rotation vectors.
            omega: [..., 3] target velocity (tangent vector in Lie algebra).
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')

        orig_shape = rot_0.shape
        n_samples = np.prod(orig_shape[:-1])
        rot_0_flat = rot_0.reshape(n_samples, 3)

        # Convert to rotation objects
        R_0 = Rotation.from_rotvec(rot_0_flat)

        # Sample target rotations R_1 ~ Uniform(SO(3))
        R_1 = self._uniform_so3(n_samples)

        # Compute relative rotation: R_rel = R_0^{-1} * R_1
        R_rel = R_0.inv() * R_1

        # omega = Log(R_rel) — the geodesic tangent vector
        omega = R_rel.as_rotvec()  # [n_samples, 3]

        # Geodesic interpolation: R_t = R_0 * Exp(t * omega)
        R_delta = Rotation.from_rotvec(t * omega)
        R_t = R_0 * R_delta
        rot_t = R_t.as_rotvec().reshape(orig_shape)

        # Target velocity
        omega = omega.reshape(orig_shape)

        return rot_t, omega

    def torch_velocity(self, rotvec_0t, t):
        """Compute velocity in torch (for model output processing).

        For flow matching, the model directly predicts the velocity,
        so this is an identity-like operation (no analytical formula needed).
        This method exists for API compatibility with the score-matching path.

        Args:
            rotvec_0t: [..., 3] relative rotation from init to current.
            t: [B] time values.

        Returns:
            [..., 3] velocity estimate.
        """
        # In flow matching, the velocity is the relative rotation divided by t.
        # rotvec_0t = t * omega, so omega = rotvec_0t / t
        eps = 1e-6
        t_expanded = t
        while t_expanded.dim() < rotvec_0t.dim():
            t_expanded = t_expanded.unsqueeze(-1)
        return rotvec_0t / (t_expanded + eps)

    def score_scaling(self, t: np.ndarray):
        """No scaling needed for flow matching. Returns 1."""
        if np.isscalar(t):
            return np.ones_like(np.array(t))
        return np.ones_like(t)

    def reverse(
            self,
            rot_t: np.ndarray,
            velocity: np.ndarray,
            t: float,
            dt: float,
            mask: np.ndarray = None,
            noise_scale: float = 1.0,
    ):
        """ODE reverse step on SO(3) from t to t-dt.

        R_{t-dt} = R_t * Exp(-velocity * dt)

        Args:
            rot_t: [..., 3] current rotation vectors at time t.
            velocity: [..., 3] predicted velocity (angular velocity).
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: which residues to update.
            noise_scale: unused (kept for API compatibility).

        Returns:
            [..., 3] rotation vectors at time t-dt.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')

        # Perturbation in tangent space
        perturb = velocity * dt
        if mask is not None:
            perturb *= mask[..., None]

        n_samples = np.prod(rot_t.shape[:-1])
        R_t = Rotation.from_rotvec(rot_t.reshape(n_samples, 3))
        R_perturb = Rotation.from_rotvec((-perturb).reshape(n_samples, 3))
        R_t_1 = R_t * R_perturb
        return R_t_1.as_rotvec().reshape(rot_t.shape)
