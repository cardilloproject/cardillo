import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings


def prox_R0_nm(x):
    return np.minimum(x, 0)


def prox_R0_np(x):
    return np.maximum(x, 0)


def prox_sphere(x, radius):
    nx = np.linalg.norm(x)
    if nx > 0:
        return x if nx <= radius else radius * x / nx
    else:
        return x if nx <= radius else radius * x


def prox_sphere_x(x, radius):
    nx = np.linalg.norm(x)
    if nx > 0:
        return (
            np.ones_like(x)
            if nx <= radius
            else radius * (np.eye(len(x)) / nx - np.outer(x, x) / nx**3)
        )
    else:
        return np.ones_like(x) if nx <= radius else radius


"""
Estimation of relaxation parameter $\vr_N$ of prox function for normal contacts.
The parameter is calculated as follows, whereby $\alpha\in(0,2)$ is some scaling factor used for both normal and frictional contact.
$$
    \vr = (\alpha\vG)^{-1},
$$
where $\vG = \vW^T\vM^{-1}\vW$.


References
----------
Studer2008: https://doi.org/10.3929/ethz-a-005556821
Schweizer2015: https://doi.org/10.3929/ethz-a-010464319
"""


def prox_r(alpha, W, M):
    try:
        return alpha / csr_matrix(W.T @ spsolve(M, csc_matrix(W))).diagonal()
    except:
        return np.ones(W.shape[1], dtype=W.dtype)


def check_alpha(alpha):
    if not 0 < alpha < 2:
        warnings.warn(
            "Invalid value for alpha. alpha must be in (0,2). alpha set to 1.",
            RuntimeWarning,
        )
        return 1
    else:
        return alpha
