import numpy as np
import warnings


def approx_fprime(x0, f, method="3-point", eps=1.0e-6):
    """Inspired by scipy's approx_derivative method but extended to
    multidimensional function values and arguments.

    Parameters
    ----------
    x0 : array_like of arbitrary shape or float
        Point at which to estimate the derivatives. Float will be converted
        to a 1-D array.
    f : callable
        Function of which to estimate the derivatives. The argument x
        passed to this function is ndarray of arbitrary shape (never a scalar).
    method : {'cs', '2-point', '3-point'}, optional
        Finite difference method to use:
            - '2-point' - use the first order accuracy forward or backward
                          difference.
            - '3-point' - use central difference in interior points and the
                          second order accuracy forward or backward difference
                          near the boundary.
            - 'cs' - use a complex-step finite difference scheme. This assumes
                     that the user function is real-valued and can be
                     analytically continued to the complex plane. Otherwise,
                     produces bogus results.

    References:
    ----------
    scipy: https://github.com/scipy/scipy/blob/4cf21e753cf937d1c6c2d2a0e372fbc1dbbeea81/scipy/optimize/_numdiff.py#L275.
    """
    warnings.warn(
        "'approx_fprime' is used. Be careful since this can be a performance drawback.",
        stacklevel=2,
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
