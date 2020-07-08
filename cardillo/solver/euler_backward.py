import numpy as np
from scipy.sparse.linalg import spsolve 
from scipy.sparse import coo_matrix, csr_matrix, identity, bmat
from tqdm import tqdm

from cardillo.math import Numerical_derivative
from cardillo.solver import Solution

class Euler_backward():
    def __init__(self, model, t1, dt, newton_tol=1e-6, newton_max_iter=10, newton_error_function=lambda x: np.max(np.abs(x)), numerical_jacobian=False, debug=False):
        
        self.model = model

        # integration time
        t0 = model.t0
        self.t1 = t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.newton_error_function = newton_error_function
        self.linearSolver = spsolve

        self.nq = self.model.nq
        self.nu = self.model.nu
        self.nla_g = self.model.nla_g
        self.nla_gamma = self.model.nla_gamma
        self.n = self.nq + self.nu + self.nla_g + self.nla_gamma

        self.uDOF = np.arange(self.nu)
        self.qDOF = self.nu + np.arange(self.nq)
        self.la_gDOF = self.nu + self.nq + np.arange(self.nla_g)
        self.la_gammaDOF = self.nu + self.nq + self.nla_g + np.arange(self.nla_gamma)

        self.Mk1 = self.model.M(t0, model.q0)
        self.W_gk1 = self.model.W_g(t0, model.q0)
        self.W_gammak1 = self.model.W_gamma(t0, model.q0)

        self.numerical_jacobian = numerical_jacobian
        if numerical_jacobian:
            self.__R_x = self.__R_x_num
        else:
            self.__R_x = self.__R_x_analytic
        self.debug = debug
        if debug:
            self.__R_x = self.__R_x_debug

    def __R(self, qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
        self.Mk1 = self.model.M(tk1, qk1)
        self.W_gk1 = self.model.W_g(tk1, qk1)
        self.W_gammak1 = self.model.W_gamma(tk1, qk1)

        R = np.zeros(self.n)
        R[self.uDOF] = self.Mk1 @ (uk1 - uk) - self.dt * (self.model.h(tk1, qk1, uk1) + self.W_gk1 @ la_gk1 + self.W_gammak1 @ la_gammak1 )
        R[self.qDOF] = qk1 - qk - self.dt * self.model.q_dot(tk1, qk1, uk1)
        R[self.la_gDOF] = self.model.g(tk1, qk1)
        R[self.la_gammaDOF] = self.model.gamma(tk1, qk1, uk1)

        return R

    def __R_wrapper(self, tk1, xk1, xk):
        qk1 = xk1[self.qDOF]
        uk1 = xk1[self.uDOF]
        la_gk1 = xk1[self.la_gDOF]
        la_gammak1 = xk1[self.la_gammaDOF]

        qk = xk[self.qDOF]
        uk = xk[self.uDOF]

        return self.__R(qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1)

    def __R_x_num(self, qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
        xk = np.zeros(self.n)
        xk[self.qDOF] = qk
        xk[self.uDOF] = uk

        xk1 = np.zeros(self.n)
        xk1[self.qDOF] = qk1
        xk1[self.uDOF] = uk1
        xk1[self.la_gDOF] = la_gk1
        xk1[self.la_gammaDOF] = la_gammak1

        R_x_num = Numerical_derivative(self.__R_wrapper, order=2)._x(tk1, xk1, xk)

        return csr_matrix( R_x_num )

    def __R_x_analytic(self, qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
        # equations of motion
        Ru_u = self.Mk1 - self.dt * self.model.h_u(tk1, qk1, uk1)
        Ru_q = self.model.Mu_q(tk1, qk1, uk1 - uk) - self.dt * (self.model.h_q(tk1, qk1, uk1) + self.model.Wla_g_q(tk1, qk1, la_gk1) + self.model.Wla_gamma_q(tk1, qk1, la_gammak1))
        Ru_la_g = -self.dt * self.W_gk1
        Ru_la_gamma = -self.dt * self.W_gammak1

        # kinematic equation
        Rq_u = -self.dt * self.model.B(tk1, qk1)
        Rq_q = identity(self.nq) - self.dt * self.model.q_dot_q(tk1, qk1, uk1)

        # constrain equations
        Rla_g_q = self.model.g_q(tk1, qk1)
        Rla_gamma_q = self.model.gamma_q(tk1, qk1, uk1)
        Rla_gamma_u = self.model.gamma_u(tk1, qk1)
        
        return bmat([[Ru_u,        Ru_q,        Ru_la_g, Ru_la_gamma], \
                     [Rq_u,        Rq_q,        None,    None], \
                     [None,        Rla_g_q,     None,    None], \
                     [Rla_gamma_u, Rla_gamma_q, None,    None]]).tocsc()

    def __R_x_debug(self, qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
        R_x_num = self.__R_x_num(qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1)
        R_x_analytic = self.__R_x_analytic(qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1)
        diff = R_x_num - R_x_analytic.toarray()

        if self.debug > 1:
            error_uu = np.linalg.norm(diff[self.uDOF[:,None], self.uDOF])
            error_uq = np.linalg.norm(diff[self.uDOF[:,None], self.qDOF])
            error_ula_g = np.linalg.norm(diff[self.uDOF[:,None], self.la_gDOF])
            error_ula_gamma = np.linalg.norm(diff[self.uDOF[:,None], self.la_gammaDOF])

            error_qu = np.linalg.norm(diff[self.qDOF[:,None], self.uDOF])
            error_qq = np.linalg.norm(diff[self.qDOF[:,None], self.qDOF])
            error_qla_g = np.linalg.norm(diff[self.qDOF[:,None], self.la_gDOF])
            error_qla_gamma = np.linalg.norm(diff[self.qDOF[:,None], self.la_gammaDOF])

            error_la_gu = np.linalg.norm(diff[self.la_gDOF[:,None], self.uDOF])
            error_la_gq = np.linalg.norm(diff[self.la_gDOF[:,None], self.qDOF])
            error_la_gla_g = np.linalg.norm(diff[self.la_gDOF[:,None], self.la_gDOF])
            error_lala_gamma = np.linalg.norm(diff[self.la_gDOF[:,None], self.la_gammaDOF])

            error_la_gammau = np.linalg.norm(diff[self.la_gammaDOF[:,None], self.uDOF])
            error_la_gammaq = np.linalg.norm(diff[self.la_gammaDOF[:,None], self.qDOF])
            error_la_gammala_g = np.linalg.norm(diff[self.la_gammaDOF[:,None], self.la_gDOF])
            error_la_gammala_gamma = np.linalg.norm(diff[self.la_gammaDOF[:,None], self.la_gammaDOF])

            print(f'error_uu jacobian: {error_uu:.5e}')
            print(f'error_uq jacobian: {error_uq:.5e}')
            print(f'error_ula_g jacobian: {error_ula_g:.5e}')
            print(f'error_ula_gamma jacobian: {error_ula_gamma:.5e}')

            print(f'error_qu jacobian: {error_qu:.5e}')
            print(f'error_qq jacobian: {error_qq:.5e}')
            print(f'error_qla_g jacobian: {error_qla_g:.5e}')
            print(f'error_qla_gamma jacobian: {error_qla_gamma:.5e}')

            print(f'error_lau jacobian: {error_la_gu:.5e}')
            print(f'error_laq jacobian: {error_la_gq:.5e}')
            print(f'error_la_gla_g jacobian: {error_la_gla_g:.5e}')
            print(f'error_lala_gamma jacobian: {error_lala_gamma:.5e}')

            print(f'error_la_gammau jacobian: {error_la_gammau:.5e}')
            print(f'error_la_gammaq jacobian: {error_la_gammaq:.5e}')
            print(f'error_la_gammala_g jacobian: {error_la_gammala_g:.5e}')
            print(f'error_la_gammala_gamma jacobian: {error_la_gammala_gamma:.5e}')

        print(f'total error jacobian: {np.linalg.norm(diff)/ self.n:.5e}')

        if self.numerical_jacobian:
            return R_x_num
        else:
            return R_x_analytic
        
    def step(self, tk, qk, uk, la_gk, la_gammak):
        dt = self.dt
        tk1 = tk + dt

        # foward Euler predictor
        la_gk1 = la_gk
        la_gammak1 = la_gammak
        uk1 = uk + dt * spsolve(self.Mk1.tocsc(), self.model.h(tk, qk, uk) + self.W_gk1 @ la_gk + self.W_gammak1 @ la_gammak)
        qk1 = qk + dt * self.model.q_dot(tk, qk, uk1)

        # initial guess for Newton-Raphson solver
        xk1 = np.zeros(self.n)
        xk1[self.qDOF] = qk1
        xk1[self.uDOF] = uk1
        xk1[self.la_gDOF] = la_gk1
        xk1[self.la_gammaDOF] = la_gammak1

        # initial residual and error
        R = self.__R(qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1)
        error = self.newton_error_function(R)
        converged = error < self.newton_tol
        j = 0
        if not converged:
            while j < self.newton_max_iter:
                # jacobian
                R_x = self.__R_x(qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1)

                # Newton update
                j += 1
                dx = spsolve(R_x, R)
                xk1 -= dx
                qk1 = xk1[self.qDOF]
                uk1 = xk1[self.uDOF]
                la_gk1 = xk1[self.la_gDOF]
                la_gammak1 = xk1[self.la_gammaDOF]

                R = self.__R(qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1)
                error = self.newton_error_function(R)
                converged = error < self.newton_tol
                if converged:
                    break

            if not converged:
                raise RuntimeError(f'internal Newton-Raphson method not converged after {j} stepts with error: {error:.5e}')
            
        return (converged, j, error), tk1, qk1, uk1, la_gk1, la_gammak1

    def solve(self): 
        # lists storing output variables
        tk = self.model.t0
        qk = self.model.q0.copy()
        uk = self.model.u0.copy()
        la_gk = self.model.la_g0.copy()
        la_gammak = self.model.la_gamma0.copy()
        
        q = [qk]
        u = [uk]
        la_g = [la_gk]
        la_gamma = [la_gammak]


        pbar = tqdm(self.t[:-1])
        for tk in pbar:
            (converged, n_iter, error), tk1, qk1, uk1, la_gk1, la_gammak1 = self.step(tk, qk, uk, la_gk, la_gammak)

            pbar.set_description(f't: {tk1:0.2e}; Newton: {n_iter}/{self.newton_max_iter} iterations; error: {error:0.2e}')

            qk1, uk1 = self.model.solver_step_callback(tk1, qk1, uk1)

            q.append(qk1)
            u.append(uk1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)

            # update local variables for accepted time step
            qk, uk, la_gk, la_gammak = qk1, uk1, la_gk1, la_gammak1
            
        # write solution
        return Solution(t=self.t, q=np.array(q), u=np.array(u), la_g=np.array(la_g), la_gamma=np.array(la_gamma))