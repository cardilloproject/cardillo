from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import spsolve


def terminationcriteria_cardillo(x, f, fun_args, scale, options):
    # error
    error = np.linalg.norm(f / scale) / scale.size**0.5
    converged = error < 1
    return converged, error


@dataclass
class SolverOptions:
    fixed_point_atol: float = 1e-6
    fixed_point_rtol: float = 1e-6
    fixed_point_max_iter: int = int(1e3)
    newton_atol: float = 1e-6
    newton_rtol: float = 1e-6
    newton_max_iter: int = 20
    reuse_lu_decomposition: bool = True
    reuse_lu_max_iter: int = 4
    prox_scaling: float = 1.0
    continue_with_unconverged: bool = False
    linear_solver: callable = spsolve
    numerical_jacobian_method: bool | str = False
    numerical_jacobian_eps: float = 1e-6
    compute_consistent_initial_conditions: bool = True
    terminationcriteria: callable = terminationcriteria_cardillo

    def __post_init__(self):
        assert self.fixed_point_atol > 0
        assert self.fixed_point_rtol > 0
        assert self.fixed_point_max_iter > 0
        assert self.newton_atol >= 0
        assert self.newton_rtol >= 0
        assert self.newton_atol + self.newton_rtol > 0
        assert self.newton_max_iter > 0
        assert self.prox_scaling > 0
        assert self.numerical_jacobian_method in [False, "2-point", "3-point", "cs"]

    def converged(self, x, f, fun_args, scale):
        return self.terminationcriteria(x, f, fun_args, scale, self)
