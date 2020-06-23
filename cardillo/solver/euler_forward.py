import numpy as np
from scipy.sparse.linalg import spsolve 
from scipy.sparse import csc_matrix

class Euler_forward():
    r""" Euler forward

    Parameters
    ----------
    model : Model
        Mechanical model
    t_span : list, tuple or numpy.ndarray
        integration domain, t_span[0] and t_span[-1] are the start and end integration time, respetively; if t_span contains only two elements no dense output will be generated, otherwise the given points are used for computing a dense output
    dt : float, optional
        user given time step; if variable_step_size is chosen this is the initial step size; defualt value is None, then we use a conservative initial value as given in :cite:`Hairer1993` p. 169
    atol : float or numpy.ndarray, optional
        Absolute tolerance used for the error estimation. Let $y_{1i}$ and $\\hat{y}_{1i}$ being the $i$-th components of the solution at the next time step, computed by the generalized-$\\alpha$ scheme and the backward Euler method, respectively. The last converged solution of the generalized-$\\alpha$ method is $y_{0i}$. We estimate the error using $e = \\sqrt{\\frac{1}{n} \\sum_{i=1}^n \\left(\\frac{y_{1i} - \\hat{y}_{1i}}{sc_i}\\right)^2}$ with $sc_i = atol_i + rtol_i~\\mathrm{max}(\\lvert y_{0i}\\rvert, \\lvert y_{1i} \\rvert)$. Where $atol_i$ and $rtol_i$ are the desired tolerances prescribed by the user (relative errors are considered for $atol_i = 0$, absolute errors for $rtol_i = 0$; usually both tolerances are different from zero.
    rtol : float or numpy.ndarray, optional
        Relative tolerance used for the error estimation. Let $y_{1i}$ and $\\hat{y}_{1i}$ being the $i$-th components of the solution at the next time step, computed by the generalized-$\\alpha$ scheme and the backward Euler method, respectively. The last converged solution of the generalized-$\\alpha$ method is $y_{0i}$. We estimate the error using $e = \\sqrt{\\frac{1}{n} \\sum_{i=1}^n \\left(\\frac{y_{1i} - \\hat{y}_{1i}}{sc_i}\\right)^2}$ with $sc_i = atol_i + rtol_i~\\mathrm{max}(\\lvert y_{0i}\\rvert, \\lvert y_{1i} \\rvert)$. Where $atol_i$ and $rtol_i$ are the desired tolerances prescribed by the user (relative errors are considered for $atol_i = 0$, absolute errors for $rtol_i = 0$; usually both tolerances are different from zero.
    newton_max_iter : int, optional
        maximum number of iterations for internal Newton--Raphson steps.
    newton_tol : float, optional
        tolerance for internal Newton--Raphson steps.
    newton_error_function : :ref:`lambda<python:lambda>` with numpy.ndarray as argument, optional
        Function which is used for computing the error in the underlying Newton-Raphson method. The maximum absolute value is the default funciton.

    Notes
    -----
    

    Note
    ---- 
    """
    def __init__(self, model, t_span, dt):
        
        self.model = model

        # integration time
        self.t_span = np.asarray(t_span)
        if self.t_span.ndim != 1:
            raise ValueError("`t_span` must be 1-dimensional.")
        d = np.diff(self.t_span)
        if np.any(d <= 0):
            raise ValueError("Values in `t_span` are not an increasing sequence.")
        self.t0, self.t1 = self.t_span[[0, -1]]
        
        # constant time step
        self.dt = dt

        self.linearSolver = spsolve

    def step(self, t0, q0, u0):
        # general quantities
        dt = self.dt
        t1 = t0 + dt
        
        M = self.model.M(t0, q0, scipy_matrix=csc_matrix)
        h = self.model.h(t0, q0, u0) 
        u1 = u0 + dt * self.linearSolver(M, h) 

        B = self.model.B(t0, q0, scipy_matrix=csc_matrix)
        beta = self.model.beta(t0, q0) 
        q1 = q0 + dt * (B @ u0 + beta)
        
        return t1, q1, u1

    def solve(self): 
        
        # lists storing output variables
        tk = self.t0
        qk = self.model.q0.copy()
        uk = self.model.u0.copy()
        
        t = [tk]
        q = [qk]
        u = [uk]

        while tk <= self.t1:
            tk1, qk1, uk1 = self.step(tk, qk, uk)

            qk1, uk1 = self.model.callback(tk1, qk1, uk1)

            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            # update local variables for accepted time step
            tk, qk, uk = tk1, qk1, uk1
            
        # write solution
        return np.array(t), np.array(q), np.array(u)
    
    