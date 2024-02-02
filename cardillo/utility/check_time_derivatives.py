import numpy as np


def check_time_derivatives(f, f_t, f_tt, eps=1.0e-6):
    """Generates time callable functions and first and second time derivatives.
    Does not check for consistency.

    Parameters
    ----------
    f : np.array()
        Callable/not-callable function f.
    f_t : np.array(n) (callable/non-callable)
        Callable/not-callable time derivative f_t.
    f_tt : np.array(n) (callable/non-callable)
        Callable/not-callable second time derivative f_tt.

    Returns
    -------
    f__ : np.array() callable
        Callable function f.
    f_t__ :
        Callable time derivative f_t.
    f_tt__ :
        Callable second time derivative f_tt.

    """
    if not callable(f):
        f__ = lambda t: f
        f_t__ = lambda t: np.zeros_like(f)
        f_tt__ = lambda t: np.zeros_like(f)
    else:
        f__ = f
        if f_t is not None:
            if callable(f_t):
                f_t__ = f_t
            else:
                f_t__ = lambda t: f_t
        else:
            f_t__ = lambda t: (f__(t + eps) - f__(t - eps)) / (2 * eps)

        if f_tt is not None:
            if callable(f_tt):
                f_tt__ = f_tt
            else:
                f_tt__ = lambda t: f_tt
        else:
            f_tt__ = lambda t: (f__(t + eps) - 2 * f__(t) + f__(t - eps)) / (eps * eps)

    return f__, f_t__, f_tt__
