"""R^3 diffusion methods."""
import numpy as np
from openfold.utils import rigid_utils as ru
import torch


def cosine_schedule(num_t, eta_max, eta_min):
    """
    Cosine interpolation of some value between its max <eta_max> and its min <eta_min>
    from https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    
    Parameters:
        T (int, required): total number of steps 
        eta_max (float, required): Max value of some parameter eta 
        eta_min (float, required): Min value of some parameter eta 
    """
    t = torch.arange(num_t)
    return eta_max + 0.5*(eta_min-eta_max)*(1+torch.cos((t/num_t)*np.pi))

class Diffuser:

    def __init__(
            self,
            num_b,
            min_b,
            max_b,
            schedule,
        ):
        """
        Args:
            num_b: length of diffusion.
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self.num_b = num_b
        self.min_b = min_b
        self.max_b = max_b
        if schedule == 'cosine':
            self._b_schedule = cosine_schedule(num_b, max_b, min_b)
        elif schedule == 'linear':
            self._b_schedule = np.linspace(min_b, max_b, num_b)
        else:
            raise ValueError(f'Unrecognized noise schedule {schedule}.')
        self._a_schedule = 1 - self._b_schedule
        self._cum_a_schedule = np.cumprod(self._a_schedule)

    def forward_sample(self, x_0, t):
        cum_a_t = self._cum_a_schedule[t]
        noise_t = self.sample_normal(size=x_0.shape)
        x_t = np.sqrt(cum_a_t) * x_0 + np.sqrt(1 - cum_a_t) * noise_t
        return x_t, noise_t, 1

    def sample_normal(self, size, scale=1.0):
        return np.random.normal(scale=scale, size=size)

    def reverse_sample(
            self,
            rigid_t: ru.Rigid,
            e_t: torch.Tensor,
            t: torch.Tensor,
            noise_scale: int=1.0,
            mask: torch.Tensor=None,
            center: bool=True):
        """Samples next step of the reverse diffusion.
        
        Args:
            rigid_t: [..., N] rigid object of frames at time t.
            e_t: [..., N, 3] epsilon (noise) prediction at time t.
            t: scalar of current time step.
            noise_scale: adhoc multiplier of drift -- added noise.
            mask: [..., N] true indicates the residue is diffused
            center: whether to center the resulting translations.

        Returns:
            rigid_t_1: [..., N] rigid object of the next time step.
        
        """
        rigid_t = ru.Rigid.from_tensor_7(rigid_t)
        x_t = rigid_t.get_trans()
        b_t = torch.Tensor(self._b_schedule).to(x_t.device)[t][:, None, None]
        a_t = torch.Tensor(self._a_schedule).to(x_t.device)[t][:, None, None]
        cum_a_t = torch.Tensor(self._cum_a_schedule).to(x_t.device)[t][:, None, None]
        pred_noise = (1 - a_t) / torch.sqrt(1 - cum_a_t) * e_t

        z = (t > 0).float()[:, None, None] * torch.Tensor(
            self.sample_normal(size=x_t.shape)).to(x_t.device)

        x_t_1 = 1 / torch.sqrt(a_t) * (x_t - pred_noise) + z * torch.sqrt(b_t) * noise_scale
        if mask is not None:
            x_t_1 = x_t_1 * mask[..., None] + (1 - mask[..., None]) * x_t
        if center:
            com = torch.sum(x_t_1, dim=1) / torch.sum(mask, dim=1)[..., None]
            x_t_1 -= com[..., None, :]
        return ru.Rigid(
            rots=rigid_t.get_rots(),
            trans=x_t_1).to_tensor_7()

    def one_shot_sample(
            self,
            rigid_t: ru.Rigid,
            e_t: torch.Tensor,
            t: torch.Tensor,
            mask: torch.Tensor=None,
            center: bool=True):
        rigid_t = ru.Rigid.from_tensor_7(rigid_t)
        x_t = rigid_t.get_trans()
        cum_a_t = torch.Tensor(self._cum_a_schedule).to(x_t.device)[t][:, None, None]
        x_0 = (x_t - torch.sqrt(1 - cum_a_t) * e_t) / torch.sqrt(cum_a_t)
        if center:
            com = torch.sum(x_0, dim=1) / torch.sum(mask, dim=1)[..., None]
            x_0 -= com[..., None, :]
        return ru.Rigid(
            rots=rigid_t.get_rots(),
            trans=x_0).to_tensor_7()


    def process_pos(self, x):
        non_zero_mask = (torch.sum(x > 1e-6, axis=-1) > 0).to(x.device)
        x_center = torch.mean(x[torch.where(non_zero_mask)], axis=0)
        return (x - x_center)
