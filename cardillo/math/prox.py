import numpy as np
from scipy.sparse import csc_array, diags
from scipy.sparse.linalg import spsolve
import warnings


class NegativeOrthant:
    """This class collects functions related to the set `C = R_0^{-n}`
    (negative orthant).

    This includes
    - proximal point

        `y = prox_C(x) = min(x, 0)`

    - implicit nonlinear function whose zero solves the normal cone inclusion

        `x \in N_{C}(-y) <=> f(x, y) = y + prox_C(rho * x - y)`,

        where `rho` denotes a positive real number.

        An active-set strategy is utilized for solving the implicit equation.
        Hence, in addition to the nonlinear equation, the active set, a
        modified nonlinear equation and the Jacobian are implemented.
    """

    @staticmethod
    def prox(x):
        return np.minimum(x, np.zeros_like(x))

    # gen = implicit_function(0, 1, 2)
    # active_set = next(gen)
    # residual = next(gen)
    # Jx, Jy = next(gen)

    @staticmethod
    def implicit_function(x, y, rho):
        active_set = (rho * x - y) <= 0
        yield active_set

        # residual
        yield np.where(active_set, x, y)

        # Jacobian
        Jx = diags(active_set.astype(float))
        Jy = diags((~active_set).astype(float))
        yield Jx, Jy

    @staticmethod
    def active_set(x, y, rho):
        return (rho * x - y) <= 0

    @staticmethod
    def residual(x, y, active_set):
        return np.where(active_set, x, y)

    @staticmethod
    def Jacobian(active_set):
        Jg = diags(active_set.astype(float))
        Jh = diags((~active_set).astype(float))
        return Jg, Jh


class Sphere:
    """This class collects functions related to the set `C(z; r) = {y | ||y|| <= r * z}`,
    a n-dimensional ball of radius `r*z`.

    This includes
    - proximal point

        `y = prox_{C(z; r)}(x)`

    - implicit nonlinear function whose zero solves the normal cone inclusion

        `x \in N_{C(z; r)}(-y) <=> f(x, y) = y + prox_{C(z; r)}(rho * x - y)`,

        where `rho` denotes a positive real number.

        An active-set strategy is utilized for solving the implicit equation.
        Hence, in addition to the nonlinear equation, the active set, a
        modified nonlinear equation and the Jacobian are implemented.
    """

    def __init__(self, r=1) -> None:
        self.r = r

    def prox(self, x, z):
        radius = max(0, self.r * z)
        norm_x = np.linalg.norm(x)
        if norm_x <= radius:
            return x
        else:
            return radius * x / norm_x

    # def active_set(self, x, y, radius, rho):
    #     assert len(x) == len(y)

    #     inside = False
    #     outside_and_positive = False
    #     if radius > 0:
    #         arg = rho * x - y
    #         norm_arg = np.linalg.norm(arg)

    #         if norm_arg <= radius:
    #             inside = True
    #         else:
    #             if self.reformulation:
    #                 if np.linalg.norm(x) > 0:
    #                     outside_and_positive = True
    #             else:
    #                 if norm_arg > 0:
    #                     outside_and_positive = True

    #     return inside, outside_and_positive

    # @staticmethod
    # def residual(g, h, r, rho, NF_connectivity, inside, outside_and_positive):
    #     R = np.zeros_like(g, dtype=np.common_type(g, h, r))

    #     ########################
    #     # non-vectorized version
    #     ########################
    #     for i, j in enumerate(NF_connectivity):
    #         if len(j) > 0:
    #             ri = r[i]
    #             gj = g[j]

    #             if inside[i]:
    #                 R[j] = g[j]
    #             elif outside_and_positive[i]:
    #                 if ri < 0:
    #                     # raise RuntimeError("This should never be the case.")
    #                     ri = 0
    #                 if use_friction_prox_reformulation:
    #                     R[j] = h[j] + ri * gj / norm(gj)
    #                 else:
    #                     prox_argj = rho[j] * gj - h[j]
    #                     R[j] = h[j] + ri * prox_argj / norm(prox_argj)
    #             else:
    #                 R[j] = h[j]

    #     return R

    # @staticmethod
    # def Jacobian(g, h, r, rho, NF_connectivity, inside, outside_and_positive):
    #     ng = len(g)
    #     assert ng == len(h)
    #     nr = len(r)
    #     Jg = lil_matrix((ng, ng))
    #     Jh = lil_matrix((ng, ng))
    #     Jr = lil_matrix((ng, nr))

    #     if not use_friction_prox_reformulation:
    #         prox_arg = rho * g - h

    #     ########################
    #     # non-vectorized version
    #     ########################
    #     for i, j in enumerate(NF_connectivity):
    #         j = np.array(j)
    #         nj = len(j)
    #         if nj > 0:
    #             if inside[i]:
    #                 Jg[j[:, None], j] = identity(nj)
    #             elif outside_and_positive[i]:
    #                 ri = r[i]
    #                 if ri < 0:
    #                     ri = 0
    #                 if use_friction_prox_reformulation:
    #                     gj = g[j]
    #                     norm_gj = norm(gj)
    #                     dj = gj / norm_gj

    #                     Jg[j[:, None], j] = (
    #                         ri * (identity(nj) - np.outer(dj, dj)) / norm_gj
    #                     )
    #                     Jh[j[:, None], j] = identity(nj)
    #                     Jr[j[:, None], i] = dj
    #                 else:
    #                     prox_argj = prox_arg[j]
    #                     norm_prox_argj = norm(prox_argj)
    #                     dj = prox_argj / norm_prox_argj

    #                     factor = ri * (identity(nj) - np.outer(dj, dj)) / norm_prox_argj

    #                     Jg[j[:, None], j] = factor @ diags(rho[j])
    #                     Jh[j[:, None], j] = identity(nj) - factor
    #                     Jr[j[:, None], i] = dj

    #             else:
    #                 Jh[j[:, None], j] = identity(nj)

    #     return Jg, Jh, Jr


# TODO:
# - write documentation
# - Can we pass the reformulation somehow to the class without making it dynamic?
class Hyperellipsoid:
    def __init__(self, dimension=2, scaling=np.ones(2), reformulation=True):
        scaling = np.atleast_1d(scaling)
        self.dimension = dimension
        self.scaling = scaling
        assert len(scaling) == dimension
        self.inverse_scaling = 1 / scaling
        self.reformulation = reformulation

    def prox(self, x, radius):
        # scale the argument to compute the proximal point on a hypersphere
        x_scaled = x * self.inverse_scaling

        # project onto hypersphere
        norm_x = np.linalg.norm(x_scaled)
        if norm_x > 0:
            return x_scaled if norm_x <= radius else radius * x_scaled / norm_x
        else:
            return x_scaled if norm_x <= radius else radius * x_scaled


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
        if nx <= radius:
            np.ones_like(x)
        else:
            d = x / nx
            return radius * (np.eye(len(x)) - np.outer(d, d)) / nx
        # return (
        #     np.ones_like(x)
        #     if nx <= radius
        #     else radius * (np.eye(len(x)) / nx - np.outer(x, x) / nx**3)
        # )
    else:
        return np.ones_like(x) if nx <= radius else radius


"""
Estimation of relaxation parameters $r_i$ of prox function for normal contacts 
and friction. The parameters are calculated as follows, whereby $\alpha \in (0,2)$ 
is some scaling factor used for both normal and frictional contact:
$$
    r_i = \alpha / diag(\vG)_i,
$$
where $\vG = \vW^T \vM^{-1} \vW$.


References
----------
Studer2008: https://doi.org/10.3929/ethz-a-005556821
Schweizer2015: https://doi.org/10.3929/ethz-a-010464319
"""


def estimate_prox_parameter(alpha, W, M):
    if W.shape[1] > 0:
        return alpha / csc_array(W.T @ spsolve(csc_array(M), csc_array(W))).diagonal()
    else:
        return np.full(W.shape[1], alpha, dtype=W.dtype)


def validate_alpha(alpha):
    if not 0 < alpha < 2:
        warnings.warn(
            "Invalid value for alpha. alpha must be in (0,2). alpha set to 1.",
            RuntimeWarning,
        )
        return 1
    else:
        return alpha
