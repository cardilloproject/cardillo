import numpy as np
import warnings


# TODO:
# * Can we implement second derivatives as done for _tt below?
def approx_fprime(x0, f, eps=1.0e-6, method="2-point"):
    """Inspired by scipy's approx_derivative method but extended to
    multidimensional function values and arguments.

    References:
    ===========
    scipy: https://github.com/scipy/scipy/blob/4cf21e753cf937d1c6c2d2a0e372fbc1dbbeea81/scipy/optimize/_numdiff.py#L275."""
    warnings.warn(
        "'approx_fprime' is used. Be careful since this can be a performance drawback."
    )

    if method not in ["2-point", "3-point", "cs"]:
        raise ValueError(f"Unknown method '{method}'.")

    x0 = np.atleast_1d(x0)
    f0 = np.atleast_1d(f(x0))

    # reshape possible mutidimensional arguments to 1D arrays and wrap f
    # accordingly
    x_shape = x0.shape
    xx = x0.reshape(-1)
    ff = lambda x: f(x.reshape(x_shape))
    m = len(xx)

    f_shape = f0.shape
    grad = np.empty(f_shape + (m,))

    h = np.diag(eps * np.ones(m))
    for i in range(m):
        if method == "2-point":
            x = xx + h[i]
            dx = x[i] - xx[i]  # recompute dx as exactly representable number
            df = ff(x) - f0
        elif method == "3-point":
            x1 = xx + h[i]
            x2 = xx - h[i]
            dx = x2[i] - x1[i]  # recompute dx as exactly representable number
            df = ff(x2) - ff(x1)
        elif method == "cs":
            x1 = xx + h[i] * 1.0j
            f1 = ff(x1)
            df = f1.imag
            dx = (x1[i] - xx[i]).imag
        else:
            raise RuntimeError('method "{method}" is not implemented!')

        grad[..., i] = df / dx

    return np.squeeze(grad.reshape(f_shape + x_shape))
