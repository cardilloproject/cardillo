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


class Numerical_derivative:
    def __init__(self, residual, order=1):
        self.residual = residual
        self.order = order

    def _dot(self, t, x, x_dot, x_ddot, eps=1.0e-6):
        f_t = self._t(t, x, x_dot, eps=eps)
        f_x = self._x(t, x, x_dot, eps=eps)
        f_x_dot = self._y(t, x, x_dot, eps=eps)
        f_dot = f_t + f_x @ x_dot + f_x_dot @ x_ddot
        return f_dot

    def _X(self, X, eps=1.0e-6):
        x = X.reshape(-1)
        Xshape = X.shape
        R = lambda t, x: self.residual(x.reshape(Xshape))
        R_x = Numerical_derivative(R, order=self.order)._x(0, x, eps=eps)

        Rshape = R_x.shape[:-1]
        return R_x.reshape(Rshape + Xshape)

    def _x(self, t, x, y=None, eps=1.0e-6):
        # evaluate residual residual and vary first x
        xPlus = np.copy(x)
        xPlus[0] += eps
        if y is None:
            RPlus = self.residual(t, xPlus)
        else:
            RPlus = self.residual(t, xPlus, y)

        nx = len(x)
        shapeR = RPlus.shape
        R_x = np.zeros(shapeR + (nx,))
        R_xi = np.zeros(shapeR)

        if self.order == 1:
            # evaluate true residual
            if y is None:
                R = self.residual(t, x)
            else:
                R = self.residual(t, x, y)
            R_xi = np.squeeze(RPlus - R) / eps
        else:
            # evaluate first residual at x[0] - eps
            xMinus = np.copy(x)
            xMinus[0] -= eps
            if y is None:
                RMinus = self.residual(t, xMinus)
            else:
                RMinus = self.residual(t, xMinus, y)
            R_xi = np.squeeze(RPlus - RMinus) / (2 * eps)
        R_x[..., 0] += R_xi

        for i in range(1, nx):
            # forward differences
            xPlus = np.copy(x)
            xPlus[i] += eps
            if y is None:
                RPlus = self.residual(t, xPlus)
            else:
                RPlus = self.residual(t, xPlus, y)

            # backward differences for central differences computation
            if self.order == 1:
                # compute forward differences
                R_xi = np.squeeze(RPlus - R) / eps
            else:
                xMinus = np.copy(x)
                xMinus[i] -= eps
                if y is None:
                    RMinus = self.residual(t, xMinus)
                else:
                    RMinus = self.residual(t, xMinus, y)

                # compute central differences
                R_xi = np.squeeze(RPlus - RMinus) / (2 * eps)
            R_x[..., i] += R_xi

        return R_x

    def _y(self, t, x, y, eps=1.0e-6):
        # evaluate residual residual and vary first x
        yPlus = np.copy(y)
        yPlus[0] += eps
        RPlus = self.residual(t, x, yPlus)

        ny = len(y)
        shapeR = RPlus.shape
        R_y = np.zeros(shapeR + (ny,))
        R_yi = np.zeros(shapeR)

        if self.order == 1:
            # evaluate true residual
            R = self.residual(t, x, y)
            R_yi = np.squeeze(RPlus - R) / eps
        else:
            # evaluate first residual at x[0] - eps
            yMinus = np.copy(y)
            yMinus[0] -= eps
            RMinus = self.residual(t, x, yMinus)
            R_yi = np.squeeze(RPlus - RMinus) / (2 * eps)
        R_y[..., 0] += R_yi

        for i in range(1, ny):
            # forward differences
            yPlus = np.copy(y)
            yPlus[i] += eps
            RPlus = self.residual(t, x, yPlus)

            # backward differences for central differences computation
            if self.order == 1:
                # compute forward differences
                R_yi = np.squeeze(RPlus - R) / eps
            else:
                yMinus = np.copy(y)
                yMinus[i] -= eps
                RMinus = self.residual(t, x, yMinus)

                # compute central differences
                R_yi = np.squeeze(RPlus - RMinus) / (2 * eps)
            R_y[..., i] += R_yi

        return R_y

    def _t(self, t, x, y=None, eps=1.0e-6):
        if y is None:
            RPlus = self.residual(t + eps, x)
        else:
            RPlus = self.residual(t + eps, x, y)

        if self.order == 1:
            # evaluate residual at t
            if y is None:
                return (RPlus - self.residual(t, x)) / eps
            else:
                return (RPlus - self.residual(t, x, y)) / eps
        else:
            # evaluate residual at t - eps
            if y is None:
                return (RPlus - self.residual(t - eps, x)) / (2 * eps)
            else:
                return (RPlus - self.residual(t - eps, x, y)) / (2 * eps)

    def _tt(self, t, x, y=None, eps=1.0e-6):
        if y is None:
            R = self.residual(t, x)
            RPlus = self.residual(t + eps, x)
            RMinus = self.residual(t - eps, x)
        else:
            R = self.residual(t, x, y)
            RPlus = self.residual(t + eps, x, y)
            RMinus = self.residual(t - eps, x, y)
        return (RPlus - 2 * R + RMinus) / (eps * eps)
