import numpy as np

class Numerical_derivative(object):
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
        return R_x.reshape(Rshape+Xshape)

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
        R_x[... , 0] += R_xi

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
            R_x[... , i] += R_xi

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
        R_y[... , 0] += R_yi

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
            R_y[... , i] += R_yi

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