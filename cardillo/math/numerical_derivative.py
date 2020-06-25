import numpy as np

class Numerical_derivative(object):
    def __init__(self, residual, order=1):
        self.residual = residual
        self.order = order

    # TODO: get a better solution for this!
    def view(self, obj, idx, value, dim):
        if dim == 1:
            obj[:, idx] += value
        elif dim == 2:
            obj[:, :, idx] += value
        elif dim == 3:
            obj[:, :, :, idx] += value
        else:
            raise RuntimeError('dim > 3 is not implemented!')

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
        dim = len(shapeR)
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
        self.view(R_x, 0, R_xi, dim)

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
            self.view(R_x, i, R_xi, dim)

        return R_x

    def _y(self, t, x, y, eps=1.0e-6):
        # evaluate residual residual and vary first x
        yPlus = np.copy(y)
        yPlus[0] += eps
        RPlus = self.residual(t, x, yPlus)
        
        nx = len(x)
        shapeR = RPlus.shape
        dim = len(shapeR)
        R_u = np.zeros(shapeR + (nx,))
        R_ui = np.zeros(shapeR)

        if self.order == 1:
            # evaluate true residual
            R = self.residual(t, x, y)
            R_ui = np.squeeze(RPlus - R) / eps
        else:
            # evaluate first residual at x[0] - eps
            yMinus = np.copy(y)
            yMinus[0] -= eps    
            RMinus = self.residual(t, x, yMinus)   
            R_ui = np.squeeze(RPlus - RMinus) / (2 * eps)
        self.view(R_u, 0, R_ui, dim)

        for i in range(1, nx):
            # forward differences
            yPlus = np.copy(y)
            yPlus[i] += eps
            RPlus = self.residual(t, x, yPlus)

            # backward differences for central differences computation
            if self.order == 1:
                # compute forward differences
                R_ui = np.squeeze(RPlus - R) / eps
            else:
                yMinus = np.copy(y)
                yMinus[i] -= eps       
                RMinus = self.residual(t, x, yMinus)            
                                        
                # compute central differences
                R_ui = np.squeeze(RPlus - RMinus) / (2 * eps)
            self.view(R_u, i, R_ui, dim)

        return R_u

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