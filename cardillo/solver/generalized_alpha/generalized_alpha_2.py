from math import inf
from numpy.core.records import fromarrays
from scipy.sparse import csc
from cardillo.math.algebra import norm2
import numpy as np
from scipy.sparse.linalg import spsolve 
from scipy.sparse import coo_matrix, csc_matrix, identity, bmat
from tqdm import tqdm

from cardillo.math.prox import prox_Rn0, prox_circle
from cardillo.math import Numerical_derivative
from cardillo.solver import Solution

class Generalized_alpha_2():
    def __init__(self, model, t1, dt, beta=0.25, gamma=0.5,\
                       newton_tol=1e-8, newton_max_iter=30, newton_error_function=lambda x: np.max(np.abs(x))):
        
        self.model = model

        # integration time
        self.t0 = t0 = model.t0
        self.t1 = t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        self.dt = dt

        # parameter
        self.beta = beta
        self.gamma = gamma

        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.newton_error_function = newton_error_function

        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = model.nla_g
        self.nla_gamma = model.nla_gamma
        self.nla_N = model.nla_N
        self.nla_T = model.nla_T

        self.nR_smooth = 3 * self.nu + self.nla_g + self.nla_gamma
        self.nR = self.nR_smooth + 3 * self.nla_N + 2 * self.nla_T

        # self.Mk1 = model.M(t0, model.q0)
        # self.W_gk1 = self.model.W_g(t0, model.q0)
        # self.W_gammak1 = self.model.W_gamma(t0, model.q0)
        # self.W_Nk1 = self.model.W_N(t0, model.q0, scipy_matrix=csc_matrix)

        self.tk = model.t0
        self.qk = model.q0 
        self.uk = model.u0 
        self.la_gk = model.la_g0
        self.la_gammak = model.la_gamma0
        self.kappa_Nk = np.zeros_like(model.la_N0)
        self.la_Nk = model.la_N0
        self.La_Nk = np.zeros_like(model.la_N0)
        self.la_Tk = model.la_T0
        self.La_Tk = np.zeros_like(model.la_T0)
        #TODO:
        self.ak = spsolve(model.M(t0, model.q0).tocsr(), self.model.h(t0, model.q0, model.u0) )#+ self.W_gk1 @ model.la_g0 + self.W_gammak1 @ model.la_gamma0 + self.W_Nk1 @ model.la_N0)
        self.Qk = np.zeros(self.nu)
        self.Uk = np.zeros(self.nu)

        # self.__R_x = self.__R_x_num
        self.__R_x = self.__R_x_analytic
        
    def __R(self, tk1, xk1):
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        nla_T = self.nla_T
        dt = self.dt
        dt2 = self.dt**2
        ak1 = xk1[:nu]
        Uk1 = xk1[nu:2*nu]
        Qk1 = xk1[2*nu:3*nu]
        la_gk1 = xk1[3*nu:3*nu+nla_g]
        la_gammak1 = xk1[3*nu+nla_g:3*nu+nla_g+nla_gamma]
        kappa_Nk1 = xk1[3*nu+nla_g+nla_gamma:3*nu+nla_g+nla_gamma+nla_N]
        La_Nk1 = xk1[3*nu+nla_g+nla_gamma+nla_N:3*nu+nla_g+nla_gamma+2*nla_N]
        la_Nk1 = xk1[3*nu+nla_g+nla_gamma+2*nla_N:3*nu+nla_g+nla_gamma+3*nla_N]
        La_Tk1 = xk1[3*nu+nla_g+nla_gamma+3*nla_N:3*nu+nla_g+nla_gamma+3*nla_N+nla_T]
        la_Tk1 = xk1[3*nu+nla_g+nla_gamma+3*nla_N+nla_T:3*nu+nla_g+nla_gamma+3*nla_N+2*nla_T]

        # update dependent variables
        uk1 = self.uk + dt * ((1-self.gamma) * self.ak + self.gamma * ak1) + Uk1
        a_beta = (0.5 - self.beta) * self.ak + self.beta * ak1 
        qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, self.uk) + dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, a_beta) + self.model.B(self.tk, self.qk) @ Qk1 
        kappa_ast = kappa_Nk1 + dt**2 * ( (0.5 - self.beta) * self.la_Nk + self.beta * la_Nk1 )
        P_N = La_Nk1 + dt * ((1-self.gamma) * self.la_Nk + self.gamma * la_Nk1)
        P_T = La_Tk1 + dt * ((1-self.gamma) * self.la_Tk + self.gamma * la_Tk1)

        Mk1 = self.model.M(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1)
        W_gammak1 = self.model.W_gamma(tk1, qk1)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csc_matrix)
        W_Tk1 = self.model.W_T(tk1, qk1, scipy_matrix=csc_matrix)

        g_N = self.model.g_N(tk1, qk1)
        # g_N_dot_post = self.model.g_N_dot(tk1, qk1, uk1)
        xi_N = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_T = self.model.xi_T(tk1, qk1, self.uk, uk1)
        I_N = (kappa_ast - self.model.prox_r_N * g_N >= 0)
        # I_N =  (g_N <= 0)
        A_N = (P_N - self.model.prox_r_N * xi_N) >=0
        # A_N = g_N_dot_post <=0

        g_N_ddot_post = self.model.g_N_ddot(tk1, qk1, uk1, ak1)
        gamma_T_dot_post = self.model.gamma_T_dot(tk1, qk1, uk1, ak1)
        gamma_T_post = self.model.gamma_T(tk1, qk1, uk1)
        # evaluate residual R(ak1, la_gk1, la_gammak1)
        R = np.zeros(self.nR)
        R[:nu] = Mk1 @ ak1 - ( self.model.h(tk1, qk1, uk1) + W_gk1 @ la_gk1 + W_gammak1 @ la_gammak1 + W_Nk1 @ la_Nk1 + W_Tk1 @ la_Tk1)
        R[nu:2*nu] = Mk1 @ Uk1 - W_Nk1 @ La_Nk1 - W_Tk1 @ La_Tk1
        R[2*nu:3*nu] = Mk1 @ Qk1 - W_Nk1 @ kappa_Nk1
        R[3*nu:3*nu+nla_g] = self.model.g(tk1, qk1)
        R[3*nu+nla_g:3*nu+nla_g+nla_gamma] = self.model.gamma(tk1, qk1, uk1)
        R[3*nu+nla_g+nla_gamma:3*nu+nla_g+nla_gamma+nla_N] = kappa_ast - prox_Rn0(kappa_ast - self.model.prox_r_N * g_N)
        for i, i_N in enumerate(I_N):
            if i_N:
                R[3*nu+nla_g+nla_gamma+nla_N+i] = P_N[i] - prox_Rn0(P_N[i] - self.model.prox_r_N[i] * xi_N[i])
                if A_N[i]:
                    R[3*nu+nla_g+nla_gamma+2*nla_N+i] = la_Nk1[i] - prox_Rn0(la_Nk1[i] - self.model.prox_r_N[i] * g_N_ddot_post[i])
                else:
                    R[3*nu+nla_g+nla_gamma+2*nla_N+i] = la_Nk1[i]
            else:
                R[3*nu+nla_g+nla_gamma+nla_N+i] = P_N[i]
                R[3*nu+nla_g+nla_gamma+2*nla_N+i] = la_Nk1[i]

        offset = 0
        for i_N, i_T in enumerate(self.model.NT_connectivity):
            nT = len(i_T)
            if nT:
                if I_N[i_N]:
                    # R[3*nu+nla_g+nla_gamma+3*nla_N+offset:3*nu+nla_g+nla_gamma+3*nla_N+offset+2] = La_Tk1[i_T] - prox_circle(La_Tk1[i_T] - self.model.prox_r_T[i_N] * xi_T[i_T], self.model.mu[i_N] * La_Nk1[i_N])
                    # R[3*nu+nla_g+nla_gamma+3*nla_N+nla_T+offset:3*nu+nla_g+nla_gamma+3*nla_N+nla_T+offset+2] = la_Tk1[i_T] - prox_circle(la_Tk1[i_T] - self.model.prox_r_T[i_N] * xi_T[i_T], self.model.mu[i_N] * la_Nk1[i_N])
                    R[3*nu+nla_g+nla_gamma+3*nla_N+offset:3*nu+nla_g+nla_gamma+3*nla_N+offset+nT] = P_T[i_T] - prox_circle(P_T[i_T] - self.model.prox_r_T[i_N] * xi_T[i_T], self.model.mu[i_N] * P_N[i_N])
                    if np.linalg.norm( P_T[i_T] - self.model.prox_r_T[i_N] * xi_T[i_T]) <= self.model.mu[i_N] * P_N[i_N]:
                        R[3*nu+nla_g+nla_gamma+3*nla_N+nla_T+offset:3*nu+nla_g+nla_gamma+3*nla_N+nla_T+offset+nT] = gamma_T_dot_post[i_T] 
                    else:
                        if np.linalg.norm(gamma_T_post[i_T]) > 0:
                            n = gamma_T_post[i_T] / np.linalg.norm(gamma_T_post[i_T])
                        else:
                            n = gamma_T_post[i_T]
                        R[3*nu+nla_g+nla_gamma+3*nla_N+nla_T+offset:3*nu+nla_g+nla_gamma+3*nla_N+nla_T+offset+nT] = la_Tk1[i_T] + self.model.mu[i_N] * la_Nk1[i_N] * n
                else:
                    # R[3*nu+nla_g+nla_gamma+3*nla_N+offset:3*nu+nla_g+nla_gamma+3*nla_N+offset+2] = La_Tk1[i_T]
                    R[3*nu+nla_g+nla_gamma+3*nla_N+offset:3*nu+nla_g+nla_gamma+3*nla_N+offset+nT] = P_T[i_T]
                    R[3*nu+nla_g+nla_gamma+3*nla_N+nla_T+offset:3*nu+nla_g+nla_gamma+3*nla_N+nla_T+offset+nT] = la_Tk1[i_T]
                offset += nT
        return R
        
    def __R_nonsmooth(self, tk1, xk1):
        return self.__R(tk1, xk1)[self.nR_smooth:]

    def __R_x_analytic(self, tk1, xk1):
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        nla_T = self.nla_T
        dt = self.dt
        dt2 = self.dt**2
        
        ak1 = xk1[:nu]
        Uk1 = xk1[nu:2*nu]
        Qk1 = xk1[2*nu:3*nu]
        la_gk1 = xk1[3*nu:3*nu+nla_g]
        la_gammak1 = xk1[3*nu+nla_g:3*nu+nla_g+nla_gamma]
        kappa_Nk1 = xk1[3*nu+nla_g+nla_gamma:3*nu+nla_g+nla_gamma+nla_N]
        La_Nk1 = xk1[3*nu+nla_g+nla_gamma+nla_N:3*nu+nla_g+nla_gamma+2*nla_N]
        la_Nk1 = xk1[3*nu+nla_g+nla_gamma+2*nla_N:3*nu+nla_g+nla_gamma+3*nla_N]
        La_Tk1 = xk1[3*nu+nla_g+nla_gamma+3*nla_N:3*nu+nla_g+nla_gamma+3*nla_N+nla_T]
        la_Tk1 = xk1[3*nu+nla_g+nla_gamma+3*nla_N+nla_T:3*nu+nla_g+nla_gamma+3*nla_N+2*nla_T]

        # update dependent variables
        uk1 = self.uk + dt * ((1-self.gamma) * self.ak + self.gamma * ak1) + Uk1
        a_beta = (0.5 - self.beta) * self.ak + self.beta * ak1 
        qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, self.uk) + dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, a_beta) + self.model.B(self.tk, self.qk) @ Qk1
        kappa_ast = kappa_Nk1 + dt**2 * ( (0.5 - self.beta) * self.la_Nk + self.beta * la_Nk1 )
        P_N = La_Nk1 + dt * ((1-self.gamma) * self.la_Nk + self.gamma * la_Nk1)
        P_T = La_Tk1 + dt * ((1-self.gamma) * self.la_Tk + self.gamma * la_Tk1)
        

        # R[3*nu+nla_g+nla_gamma:3*nu+nla_g+nla_gamma+nla_N] = kappa_ast - prox_Rn0(kappa_ast - self.model.prox_r_N * g_N)

        Mk1 = self.model.M(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1)
        W_gammak1 = self.model.W_gamma(tk1, qk1)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csc_matrix)
        W_Tk1 = self.model.W_T(tk1, qk1, scipy_matrix=csc_matrix)

        # R[:nu] = Mk1 @ ak1 - ( self.model.h(tk1, qk1, uk1) + W_gk1 @ la_gk1 + W_gammak1 @ la_gammak1 + W_Nk1 @ la_Nk1 + W_Tk1 @ la_Tk1)
        Ra_q = self.model.Mu_q(tk1, qk1, ak1) - (self.model.h_q(tk1, qk1, uk1) + self.model.Wla_g_q(tk1, qk1, la_gk1) + self.model.Wla_gamma_q(tk1, qk1, la_gammak1) + self.model.Wla_N_q(tk1, qk1, la_Nk1) + self.model.Wla_T_q(tk1, qk1, la_Tk1))
        Ra_u = -self.model.h_u(tk1, qk1, uk1)
        Ra_a = Mk1 + Ra_q @ self.q_a + Ra_u * self.u_a
        Ra_U = Ra_u
        Ra_Q = Ra_q @ self.q_Q
        
        # R[nu:2*nu] = Mk1 @ Uk1 - W_Nk1 @ La_Nk1 - W_Tk1 @ La_Tk1
        RU_q = self.model.Mu_q(tk1, qk1, Uk1) - self.model.Wla_N_q(tk1, qk1, La_Nk1) - self.model.Wla_T_q(tk1, qk1, La_Tk1)
        RU_a = RU_q @ self.q_a
        RU_Q = RU_q @ self.q_Q
        
        # R[2*nu:3*nu] = Mk1 @ Qk1 - W_Nk1 @ kappa_Nk1
        RQ_q = self.model.Mu_q(tk1, qk1, Qk1) - self.model.Wla_N_q(tk1, qk1, kappa_Nk1)
        RQ_a = RQ_q @ self.q_a
        RQ_Q = Mk1 + RQ_q @ self.q_Q

        # R[3*nu:3*nu+nla_g] = self.model.g(tk1, qk1)
        Rla_g_q = self.model.g_q(tk1, qk1)
        Rla_g_a = Rla_g_q @ self.q_a
        Rla_g_Q = Rla_g_q @ self.q_Q

        # R[3*nu+nla_g:3*nu+nla_g+nla_gamma] = self.model.gamma(tk1, qk1, uk1)
        Rla_gamma_q = self.model.gamma_q(tk1, qk1, uk1) 
        Rla_gamma_u = self.model.gamma_u(tk1, qk1) # == Rla_gamma_U
        Rla_gamma_a = Rla_gamma_q @ self.q_a + Rla_gamma_u * self.u_a
        Rla_gamma_Q = Rla_gamma_q @ self.q_Q
        
        R_x_smooth =  bmat([[Ra_a, Ra_U, Ra_Q, -W_gk1,  -W_gammak1, None,     None, -W_Nk1,   None, -W_Tk1], \
                         [RU_a,  Mk1, RU_Q,   None,        None, None,   -W_Nk1,   None, -W_Tk1,   None], \
                         [RQ_a,  None, RQ_Q,   None,        None, -W_Nk1,   None,   None,   None,   None], \
                         [Rla_g_a,  None, Rla_g_Q,   None,        None, None,   None,   None,   None,   None], \
                         [Rla_gamma_a, Rla_gamma_u, Rla_gamma_Q, None, None, None, None, None, None, None] \
                         ], format='coo')

        R_x_nonsmooth = coo_matrix(Numerical_derivative(self.__R_nonsmooth, order=2)._x(tk1, xk1))

        R_x =  bmat([[R_x_smooth], \
                     [R_x_nonsmooth]], format='csc')

        # R_x_num = Numerical_derivative(self.__R, order=2)._x(tk1, xk1)

        # diff = R_x_num - R_x.toarray()
        # error = np.linalg.norm(diff[:self.nu, :self.nu], ord=inf)
        # error = np.max(np.abs(diff[:self.nu, :self.nu]))
        # error = np.linalg.norm(diff[self.nu:], ord=inf)
        # error = np.linalg.norm(diff[:self.nR_smooth], ord=inf)
        # error = np.linalg.norm(diff[self.nR_smooth:], ord=inf)
        # print(f'error R_x: {error:.3e}')
        
        # print(f'max(abs(R_x)): {np.max(np.abs(R_x_num[:self.nu, :self.nu])):.3e}')

        # return csc_matrix(R_x_num)
        return R_x

    def __R_x_num(self, tk1, xk1):
        R_x_num = Numerical_derivative(self.__R, order=2)._x(tk1, xk1)
        return csc_matrix( R_x_num )
    
    def step(self):
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        nla_T = self.nla_T
        dt = self.dt
        tk1 = self.tk + dt

        # initial guess for Newton-Raphson solver
        xk1 = np.zeros(self.nR)
        xk1[:nu] = self.ak
        xk1[nu:2*nu] = self.Uk
        xk1[2*nu:3*nu] = self.Qk
        xk1[3*nu:3*nu+nla_g] = self.la_gk
        xk1[3*nu+nla_g:3*nu+nla_g+nla_gamma] = self.la_gammak
        xk1[3*nu+nla_g+nla_gamma:3*nu+nla_g+nla_gamma+nla_N] = self.kappa_Nk
        xk1[3*nu+nla_g+nla_gamma+nla_N:3*nu+nla_g+nla_gamma+2*nla_N] = self.La_Nk
        xk1[3*nu+nla_g+nla_gamma+2*nla_N:3*nu+nla_g+nla_gamma+3*nla_N] = self.la_Nk
        xk1[3*nu+nla_g+nla_gamma+3*nla_N:3*nu+nla_g+nla_gamma+3*nla_N+nla_T] = self.La_Tk
        xk1[3*nu+nla_g+nla_gamma+3*nla_N+nla_T:3*nu+nla_g+nla_gamma+3*nla_N+2*nla_T] = self.la_Tk

        # initial residual and error
        R = self.__R(tk1, xk1)
        error = self.newton_error_function(R)
        converged = error < self.newton_tol
        j = 0
        if not converged:
            while j < self.newton_max_iter:
                # jacobian
                R_x = self.__R_x(tk1, xk1)

                # Newton update
                j += 1
                dx = spsolve(R_x, R)
                # try:
                #     dx = spsolve(R_x, R)
                # except:
                #     print('Fehler!!!!')
                xk1 -= dx

                R = self.__R(tk1, xk1)
                # print(f'I_N = {self.I_N}; A_N = {self.A_N}; R = {R}')
                error = self.newton_error_function(R)
                converged = error < self.newton_tol
                if converged:
                    break
        ak1 = xk1[:nu]
        Uk1 = xk1[nu:2*nu]
        Qk1 = xk1[2*nu:3*nu]
        la_gk1 = xk1[3*nu:3*nu+nla_g]
        la_gammak1 = xk1[3*nu+nla_g:3*nu+nla_g+nla_gamma]
        kappa_Nk1 = xk1[3*nu+nla_g+nla_gamma:3*nu+nla_g+nla_gamma+nla_N]
        La_Nk1 = xk1[3*nu+nla_g+nla_gamma+nla_N:3*nu+nla_g+nla_gamma+2*nla_N]
        la_Nk1 = xk1[3*nu+nla_g+nla_gamma+2*nla_N:3*nu+nla_g+nla_gamma+3*nla_N]
        La_Tk1 = xk1[3*nu+nla_g+nla_gamma+3*nla_N:3*nu+nla_g+nla_gamma+3*nla_N+nla_T]
        la_Tk1 = xk1[3*nu+nla_g+nla_gamma+3*nla_N+nla_T:3*nu+nla_g+nla_gamma+3*nla_N+2*nla_T]
        # if not converged:
        #     print('')
            
        return (converged, j, error), tk1, ak1, Uk1, Qk1, la_gk1, la_gammak1, kappa_Nk1, La_Nk1, la_Nk1, La_Tk1, la_Tk1

    def solve(self): 
        dt = self.dt
        dt2 = self.dt**2

        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        a = [self.ak]
        la_g = [self.la_gk]
        la_gamma = [self.la_gammak]
        kappa_N = [self.kappa_Nk]
        La_N = [self.La_Nk]
        la_N = [self.la_Nk]
        La_T = [self.La_Tk]
        la_T = [self.la_Tk]
        P_N = [self.La_Nk + self.dt * self.la_Nk]
        P_T = [self.La_Tk + self.dt * self.la_Tk]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            self.q_a = dt2 * self.beta * self.model.B(self.tk, self.qk)
            self.q_Q = self.model.B(self.tk, self.qk)
            self.u_a = dt * self.gamma

            (converged, n_iter, error), tk1, ak1, Uk1, Qk1, la_gk1, la_gammak1, kappa_Nk1, La_Nk1, la_Nk1, La_Tk1, la_Tk1 = self.step()
            pbar.set_description(f't: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.newton_max_iter} iterations; error: {error:0.2e}')
            
            if not converged:
                raise RuntimeError(f'internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}')
            dt = self.dt
            dt2 = dt * dt
            uk1 = self.uk + dt * ((1-self.gamma) * self.ak + self.gamma * ak1) + Uk1
            a_beta = (0.5 - self.beta) * self.ak + self.beta * ak1 
            qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, self.uk) + dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, a_beta) + self.model.B(self.tk, self.qk) @ Qk1
            qk1, uk1 = self.model.solver_step_callback(tk1, qk1, uk1)
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            a.append(ak1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)
            kappa_N.append(kappa_Nk1)
            La_N.append(La_Nk1)
            la_N.append(la_Nk1)
            La_T.append(La_Tk1)
            la_T.append(la_Tk1)
            P_N.append( La_Nk1 + dt * ((1-self.gamma) * self.la_Nk + self.gamma * la_Nk1) )
            P_T.append( La_Tk1 + dt * ((1-self.gamma) * self.la_Tk + self.gamma * la_Tk1) )

            # update local variables for accepted time step
            self.tk = tk1
            self.qk = qk1
            self.uk = uk1
            self.ak = ak1
            self.Qk = Qk1
            self.la_gk = la_gk1
            self.la_gammak = la_gammak1
            self.kappa_Nk = kappa_Nk1
            self.La_Nk = La_Nk1
            self.la_Nk = la_Nk1
            self.La_Tk = La_Tk1
            self.la_Tk = la_Tk1

        # write solution
        return Solution(t=np.array(t), q=np.array(q), u=np.array(u), a=np.array(a), la_g=np.array(la_g), la_gamma=np.array(la_gamma), kappa_P=np.array(kappa_N), La_N=np.array(La_N), la_N=np.array(la_N), La_T=np.array(La_T), la_T=np.array(la_T), P_N=np.array(P_N), P_T=np.array(P_T))