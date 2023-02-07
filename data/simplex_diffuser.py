"""Simplex diffusion methods."""
import numpy as np
from data import residue_constants

def _additive_logistic_transform(x):
    norm_factor = 1 + np.sum(np.exp(x), axis=-1)[..., None]
    unnorm_probs = np.concatenate(
        [np.exp(x), np.ones_like(x)[..., :1]], axis=-1)
    return unnorm_probs / norm_factor


def _inverse_logit_transform(x):
    ratio = x[..., :-1] / (x[..., :1] + 1e-5)
    return np.log(ratio + 1e-5)

class SimplexDiffuser:
    """VP-SDE diffuser class for simplex."""

    def __init__(self, simplex_conf):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self._simplex_conf = simplex_conf
        self.diffuse_seq = simplex_conf.diffuse_seq
        self.min_b = simplex_conf.min_b
        self.max_b = simplex_conf.max_b
        self.schedule = simplex_conf.schedule
        # 20: number of canonical amino acid types
        self.num_aatype = residue_constants.restype_num
        # 21: canonical amino acids plus padding type
        self.vocab_size = residue_constants.restype_num+1

    def b_t(self, t):
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        if self.schedule == 'linear': 
            return self.min_b + t*(self.max_b - self.min_b)
        elif self.schedule == 'exponential':
            sigma = t * np.log10(self.max_b) + (1 - t) * np.log10(self.min_b)
            return 10 ** sigma
        else:
            raise ValueError(f'Unknown schedule {self.schedule}')
    
    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return np.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t) * x

    def sample_ref(
            self,
            n_samples: float=1,
            impute: np.ndarray=None,
            diffuse_mask: np.ndarray=None
        ):
        if (not self.diffuse_seq) and impute is None:
            raise ValueError('Must provide imputation values.')

        if (diffuse_mask is not None) and impute is None:
            raise ValueError('Must provide imputation values.')
        if impute is not None:
            assert impute.shape[-1] == self.vocab_size
            x_impute = _inverse_logit_transform(impute)
            y_impute = impute

        if not self.diffuse_seq:
            x_T = x_impute
            y_T = y_impute
        else:
            x_T = np.random.normal(size=(n_samples, self.num_aatype))
            y_T = _additive_logistic_transform(x_T)

        if diffuse_mask is not None:
            x_T = self._apply_mask(x_T, x_impute, diffuse_mask[..., None])
            y_T = self._apply_mask(y_T, y_impute, diffuse_mask[..., None])
        return {
            'aatype_sde_t': x_T,
            'aatype_probs_t': y_T
        }

    def one_hot(self, aatype):
        return np.eye(self.vocab_size)[aatype]

    def marginal_b_t(self, t):
        if self.schedule == 'linear':
            return t*self.min_b + (1/2)*(t**2)*(self.max_b-self.min_b)
        elif self.schedule == 'exponential': 
            return (self.max_b**t * self.min_b**(1-t) - self.min_b) / (
                np.log(self.max_b) - np.log(self.min_b))
        else:
            raise ValueError(f'Unknown schedule {self.schedule}')

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def forward_marginal(
            self,
            y_0: np.ndarray,
            t: float,
            diffuse_mask: np.ndarray=None
        ):
        """Samples marginal p(x(t) | x(0)) and its logit-normal transform.

        Args:
            y_0: [..., n, 21] one-hot aatype vectors.
            t: continuous time in [0, 1]. 

        Returns:
            aatype_sde_t (or x_t): [..., n, 20] simulated brownian motion.
            aatype_probs_t (or y_t): [..., n, 21] sampled logit-normal probability vector.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        assert y_0.shape[-1] == self.vocab_size

        x_0 = _inverse_logit_transform(y_0)

        if not self.diffuse_seq:
            return {
                'aatype_sde_t': x_0,
                'aatype_probs_t': y_0
            }

        # Remove padding aatype
        x_t = np.random.normal(
            loc=np.exp(-1/2*self.marginal_b_t(t)) * x_0,
            scale=np.sqrt(1 - np.exp(-self.marginal_b_t(t)))
        )
        y_t = _additive_logistic_transform(x_t)

        if diffuse_mask is not None:
            x_t = self._apply_mask(x_t, x_0, diffuse_mask[..., None])
            y_t = self._apply_mask(y_t, y_0, diffuse_mask[..., None])

        assert x_t.shape[-1] == self.num_aatype
        assert y_t.shape[-1] == self.vocab_size
        return {
            'aatype_sde_t': x_t,
            'aatype_probs_t': y_t
        }

    def reverse(
            self,
            *,
            x_t: np.ndarray,
            y_0: np.ndarray,
            t: float,
            dt: float,
            ode: bool=False,
            diffuse_mask: np.ndarray=None,
        ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., n, 20] aatype brownian motion at time t.
            y_0: [..., n, 21] true aatype probability vector.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            add_noise: set False to set diffusion coefficent to 0.
            mask: True indicates which residues to diffuse.

        Returns:
            x_t_1: [..., n, 20] simulated brownian motion at time t-1.
            y_t_1: [..., n, 21] logit-normal probability vector at time t-1.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        assert x_t.shape[-1] == self.num_aatype
        assert y_0.shape[-1] == self.vocab_size

        if not self.diffuse_seq:
            return x_t, y_0

        # Remove padding aatype
        x_0 = _inverse_logit_transform(y_0)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        score_t = self.score(x_t, x_0, t)
        if ode:
            # Probability flow ODE
            perturb = (f_t - (1/2)*(g_t**2) * score_t) * dt
        else:
            # Usual stochastic dynamics
            z = np.random.normal(size=score_t.shape)
            perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(dt) * z

        x_t_1 = x_t - perturb
        y_t_1 = _additive_logistic_transform(x_t)

        if diffuse_mask is not None:
            x_t_1 = self._apply_mask(x_t_1, x_t, diffuse_mask[..., None])
            y_t_1 = self._apply_mask(y_t_1, y_0, diffuse_mask[..., None])

        assert y_t_1.shape[-1] == self.vocab_size
        assert x_t_1.shape[-1] == self.num_aatype
        return x_t_1, y_t_1

    def conditional_var(self, t):
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I

        """
        return 1 - np.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t):
        return -(x_t - np.exp(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t)
