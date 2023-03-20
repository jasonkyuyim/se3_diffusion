import numpy as np
import torch
from scipy.spatial.transform import Rotation
import scipy.linalg

# hat map from vector space R^3 to Lie algebra so(3)
def hat(v):
    """
    v: [..., 3]
    hat_v: [..., 3, 3]
    """
    hat_v = torch.zeros([*v.shape[:-1], 3, 3])
    hat_v[..., 0, 1], hat_v[..., 0, 2], hat_v[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    return hat_v + -hat_v.transpose(-1, -2)

# vee map from Lie algebra so(3) to the vector space R^3
def vee(A):
    assert torch.allclose(A, -A.transpose(-1, -2)), "Input A must be skew symmetric"
    vee_A = torch.stack([-A[..., 1, 2], A[..., 0, 2], -A[..., 0, 1]], dim=-1)
    return vee_A

# Logarithmic map from SO(3) to R^3 (i.e. rotation vector)
def Log(R):
    shape = list(R.shape[:-2])
    R_ = R.reshape([-1, 3, 3])
    Log_R_ = torch.tensor(Rotation.from_matrix(R_.numpy()).as_rotvec())
    return Log_R_.reshape(shape + [3])

# logarithmic map from SO(3) to so(3), this is the matrix logarithm
def log(R): return hat(Log(R))

# Exponential map from so(3) to SO(3), this is the matrix exponential
def exp(A): return torch.linalg.matrix_exp(A)

# Exponential map from R^3 to SO(3)
def Exp(A): return exp(hat(A))

# Angle of rotation SO(3) to R^+, this is the norm in our chosen orthonormal basis
def Omega(R): return torch.arccos((torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)-1)/2)

# exponential map from tangent space at R0 to SO(3)
def expmap(R0, tangent):
    skew_sym = torch.einsum('...ij,...ik->...jk', R0, tangent)
    assert torch.allclose(skew_sym, -skew_sym.transpose(-1, -2), atol=1e-4), "R0.T @ tangent must be skew symmetric"
    skew_sym = (skew_sym - torch.transpose(skew_sym, -2, -1))/2.
    exp_skew_sym = exp(skew_sym)
    return torch.einsum('...ij,...jk->...ik', R0, exp_skew_sym)

# Normal sample in tangent space at R0
def tangent_gaussian(R0): return torch.einsum('...ij,...jk->...ik', R0, hat(torch.randn(*R0.shape[:-2], 3)))

# Usual log density of normal distribution in Euclidean space
def normal_log_density(x, mean, var):
    return (-(1/2)*(x-mean)**2 / var - (1/2)*torch.log(2*torch.pi*var)).sum(dim=-1)

# log density of Gaussian in the tangent space
def tangent_gaussian_log_density(R, R_mean, var):
    Log_RmeanT_R = Log(torch.einsum('Nji,Njk->Nik', R_mean, R))
    return normal_log_density(Log_RmeanT_R, torch.zeros_like(Log_RmeanT_R), var)

# sample from uniform distribution on SO(3)
def sample_uniform(N, M=1000):
    omega_grid = np.linspace(0, np.pi, M)
    cdf = np.cumsum(np.pi**-1 * (1-np.cos(omega_grid)), 0)/(M/np.pi)
    omegas = np.interp(np.random.rand(N), cdf, omega_grid)
    axes = np.random.randn(N, 3)
    axes = omegas[..., None]* axes/np.linalg.norm(axes, axis=-1, keepdims=True)
    axes_ = axes.reshape([-1, 3])
    Rs = exp(hat(torch.tensor(axes_)))
    Rs = Rs.reshape([N, 3, 3])
    return Rs
