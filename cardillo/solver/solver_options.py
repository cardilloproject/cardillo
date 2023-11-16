from dataclasses import dataclass

from numpy import max, abs


@dataclass
class SolverOptions:
    atol: float = 1e-8
    rtol: float = 1e-8
    fixedpoint_max_iter: int = int(1e3)
    newton_max_iter: int = 20
    error_function: callable = lambda x: max(abs(x))
    prox_scaling: float = 1.0
    continue_with_unconverged: bool = True
