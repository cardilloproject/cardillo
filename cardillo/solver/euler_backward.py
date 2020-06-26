import numpy as np
from scipy.sparse.linalg import spsolve 
from scipy.sparse import coo_matrix, csr_matrix, identity, bmat
from tqdm import tqdm

from cardillo.math import Numerical_derivative

class Euler_backward():
    r""" Euler backward

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
    def __init__(self, model, t_span, dt, newton_tol=1e-6, newton_max_iter=10, newton_error_function=lambda x: np.max(np.abs(x))):
        
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
        self.t = np.arange(self.t0, self.t1 + self.dt, self.dt)

        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.newton_error_function = newton_error_function
        self.linearSolver = spsolve

        self.nq = self.model.nq
        self.nu = self.model.nu
        self.nla_g = self.model.nla_g
        self.n = self.nq + self.nu + self.nla_g

        self.uDOF = np.arange(self.nu)
        self.qDOF = self.nu + np.arange(self.nq)
        self.la_gDOF = self.nu + self.nq + np.arange(self.nla_g)

        self.Mk1 = self.model.M(self.t0, model.q0)
        self.W_gk1 = self.model.W_g(self.t0, model.q0)

    def __R(self, qk, uk, tk1, qk1, uk1, la_gk1):
        self.Mk1 = self.model.M(tk1, qk1)
        self.W_gk1 = self.model.W_g(tk1, qk1)

        R = np.zeros(self.n)
        R[self.uDOF] = self.Mk1 @ (uk1 - uk) - self.dt * (self.model.h(tk1, qk1, uk1) + self.W_gk1 @ la_gk1 )
        R[self.qDOF] = qk1 - qk - self.dt * self.model.q_dot(tk1, qk1, uk1)
        R[self.la_gDOF] = self.model.g(tk1, qk1)

        return R

    def __R_wrapper(self, tk1, xk1, xk):
        qk1 = xk1[self.qDOF]
        uk1 = xk1[self.uDOF]
        la_gk1 = xk1[self.la_gDOF]

        qk = xk[self.qDOF]
        uk = xk[self.uDOF]

        return self.__R(qk, uk, tk1, qk1, uk1, la_gk1)

    def __R_x(self, qk, uk, tk1, qk1, uk1, la_gk1):
        Ru_u = self.Mk1 - self.dt * self.model.h_u(tk1, qk1, uk1)
        Ru_q = self.model.Mu_q(tk1, qk1, uk1 - uk) - self.dt * (self.model.h_q(tk1, qk1, uk1) + self.model.Wla_g_q(tk1, qk1, la_gk1))
        Ru_la_g = -self.dt * self.W_gk1

        Rq_u = -self.dt * self.model.B(tk1, qk1)
        Rq_q = identity(self.nq) - self.dt * self.model.q_dot_q(tk1, qk1, uk1)
        # Rq_la_g = coo_matrix((self.nq, self.nla_g))

        # Rla_g_u = coo_matrix((self.nla_g, self.nu))
        Rla_g_q = self.model.g_q(tk1, qk1)
        # Rla_g_la_g = coo_matrix((self.nla_g, self.nla_g))
        
        return bmat([[Ru_u, Ru_q, Ru_la_g], \
                     [Rq_u, Rq_q, None], \
                     [None, Rla_g_q, None]]).tocsc()
        
        # R_x = bmat([[Ru_u, Ru_q, Ru_la_g], \
        #             [Rq_u, Rq_q, None], \
        #             [None, Rla_g_q, None]])

        # xk = np.zeros(self.n)
        # xk[self.qDOF] = qk
        # xk[self.uDOF] = uk

        # xk1 = np.zeros(self.n)
        # xk1[self.qDOF] = qk1
        # xk1[self.uDOF] = uk1
        # xk1[self.la_gDOF] = la_gk1

        # R_x_num = Numerical_derivative(self.__R_wrapper, order=2)._x(tk1, xk1, xk)

        # diff = R_x_num - R_x.toarray()
        # error_uu = np.linalg.norm(diff[self.uDOF[:,None], self.uDOF])
        # error_uq = np.linalg.norm(diff[self.uDOF[:,None], self.qDOF])
        # error_ula = np.linalg.norm(diff[self.uDOF[:,None], self.la_gDOF])

        # error_qu = np.linalg.norm(diff[self.qDOF[:,None], self.uDOF])
        # error_qq = np.linalg.norm(diff[self.qDOF[:,None], self.qDOF])
        # error_qla = np.linalg.norm(diff[self.qDOF[:,None], self.la_gDOF])

        # error_lau = np.linalg.norm(diff[self.la_gDOF[:,None], self.uDOF])
        # error_laq = np.linalg.norm(diff[self.la_gDOF[:,None], self.qDOF])
        # error_lala = np.linalg.norm(diff[self.la_gDOF[:,None], self.la_gDOF])
        # print(f'error_uu jacobian: {error_uu:.5e}')
        # print(f'error_uq jacobian: {error_uq:.5e}')
        # print(f'error_ula jacobian: {error_ula:.5e}')

        # print(f'error_qu jacobian: {error_qu:.5e}')
        # print(f'error_qq jacobian: {error_qq:.5e}')
        # print(f'error_qla jacobian: {error_qla:.5e}')

        # print(f'error_lau jacobian: {error_lau:.5e}')
        # print(f'error_laq jacobian: {error_laq:.5e}')
        # print(f'error_lala jacobian: {error_lala:.5e}')

        # return csr_matrix( R_x_num )

        # return csr_matrix( Numerical_derivative(self.__R_wrapper, order=1)._x(tk1, xk1, xk) )
        # # return csr_matrix( Numerical_derivative(self.__R_wrapper, order=2)._x(tk1, xk1, xk) )

        # # hack for testing nuemrical derivative w.r.t. second argument 
        # # return csr_matrix( Numerical_derivative(lambda tk1, xk, xk1: self.__R_wrapper(tk1, xk1, xk), order=1)._y(tk1, xk, xk1) )
        # # return csr_matrix( Numerical_derivative(lambda tk1, xk, xk1: self.__R_wrapper(tk1, xk1, xk), order=2)._y(tk1, xk, xk1) )

    def step(self, tk, qk, uk, la_gk):
        dt = self.dt
        tk1 = tk + dt

        # foward Euler predictor
        la_gk1 = la_gk
        uk1 = uk + dt * spsolve(self.Mk1.tocsc(), self.model.h(tk, qk, uk) + self.W_gk1 @ la_gk)
        qk1 = qk + dt * self.model.q_dot(tk, qk, uk1)

        # initial guess for Newton-Raphson solver
        xk1 = np.zeros(self.n)
        xk1[self.qDOF] = qk1
        xk1[self.uDOF] = uk1
        xk1[self.la_gDOF] = la_gk1

        # initial residual and error
        R = self.__R(qk, uk, tk1, qk1, uk1, la_gk1)
        error = self.newton_error_function(R)
        converged = error < self.newton_tol
        j = 0
        if not converged:
            while j < self.newton_max_iter:
                # jacobian
                R_x = self.__R_x(qk, uk, tk1, qk1, uk1, la_gk1)

                # Newton update
                j += 1
                dx = spsolve(R_x, R)
                xk1 -= dx
                qk1 = xk1[self.qDOF]
                uk1 = xk1[self.uDOF]
                la_gk1 = xk1[self.la_gDOF]

                R = self.__R(qk, uk, tk1, qk1, uk1, la_gk1)
                error = self.newton_error_function(R)
                converged = error < self.newton_tol
                if converged:
                    break

            if not converged:
                raise RuntimeError(f'internal Newton-Raphson method not converged after {j} stepts with error: {error:.5e}')
            
        return (converged, j, error), tk1, qk1, uk1, la_gk1

    def solve(self): 
        # lists storing output variables
        tk = self.t0
        qk = self.model.q0.copy()
        uk = self.model.u0.copy()
        la_gk = self.model.la_g0.copy()
        
        q = [qk]
        u = [uk]
        la_g = [la_gk]

        pbar = tqdm(self.t[:-1])
        for tk in pbar:
            (converged, n_iter, error), tk1, qk1, uk1, la_gk1 = self.step(tk, qk, uk, la_gk)
            # print(f't: {tk1:.5f}; converged: {converged}; #iterations: {n_iter}; error: {error:.5e}')

            pbar.set_description(f't: {tk1:0.2e}; Newton: {n_iter}/{self.newton_max_iter} iterations; error: {error:0.2e}')
            # pbar.update(1)

            qk1, uk1 = self.model.solver_step_callback(tk1, qk1, uk1)

            # t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            la_g.append(la_gk1)

            # update local variables for accepted time step
            qk, uk, la_gk = qk1, uk1, la_gk1
            
        # write solution
        return self.t, np.array(q), np.array(u), np.array(la_g)
    
    