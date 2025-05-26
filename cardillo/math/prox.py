import numpy as np
from scipy.sparse import csc_array, diags
from scipy.sparse.linalg import spsolve


class NegativeOrthant:
    r"""This class collects functions related to the set `C = R_0^{-n}`
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

    # @staticmethod
    # def implicit_function(x, y, rho):
    #     active_set = (rho * x - y) <= 0
    #     yield active_set

    #     # residual
    #     yield np.where(active_set, x, y)

    #     # Jacobian
    #     Jx = diags(active_set.astype(float))
    #     Jy = diags((~active_set).astype(float))
    #     yield Jx, Jy

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
    r"""This class collects functions related to the set `C(z; r) = {y | ||y|| <= r * z}`,
    a n-dimensional ball of radius `r * z`.

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

    def active_set(self, x, y, z, rho):
        assert len(x) == len(y)
        if np.linalg.norm(rho * x - y) <= max(0, self.r * z):
            return True
        else:
            return False

    def residual(self, x, y, z, rho, active_set):
        if active_set:
            return x
        else:
            radius = max(0, self.r * z)
            arg = rho * x - y
            return y + radius * arg / np.linalg.norm(arg)

    def Jacobian(self, x, y, z, rho, active_set):
        nx = len(x)
        assert nx == len(y)
        nr = len(z)

        if active_set:
            Jx = np.eye(nx)
            Jy = np.zeros((nx, nx))
            Jz = np.zeros((nx, nr))
        else:
            radius = max(0, self.r * z)
            arg = rho * x - y
            norm_arg = np.linalg.norm(arg)
            direction = arg / norm_arg
            factor = radius * (np.eye(nx) - np.outer(direction, direction)) / norm_arg

            Jx = factor * rho
            Jy = np.eye(nx) - factor
            Jz = self.r * direction.reshape((nx, nr))

        return Jx, Jy, Jz


def estimate_prox_parameter(alpha, W, M):
    r"""
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
    cols = W.shape[1]
    if cols > 0:
        W = csc_array(W)
        M_inv_W = spsolve(csc_array(M), W)
        WT_M_inv_W = csc_array((W.T @ M_inv_W).reshape((cols, cols)))
        return alpha / WT_M_inv_W.diagonal()
    else:
        return np.full(cols, alpha, dtype=W.dtype)
