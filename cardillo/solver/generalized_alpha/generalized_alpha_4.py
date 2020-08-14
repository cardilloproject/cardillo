from math import inf, isnan
from numpy.core.records import fromarrays
from scipy.sparse import csc
from cardillo.math.algebra import norm2
import numpy as np
from scipy.sparse.linalg import spsolve 
from scipy.sparse import coo_matrix, csc_matrix, identity, bmat, diags
from tqdm import tqdm

from cardillo.math.prox import prox_Rn0, prox_circle
from cardillo.math import Numerical_derivative
from cardillo.utility.coo import Coo
from cardillo.solver import Solution

class Generalized_alpha_4():
    def __init__(self, model, t1, dt, uDOF_algebraic=None, \
                    rho_inf=1, beta=None, gamma=None, alpha_m=None, alpha_f=None,\
                       newton_tol=1e-8, newton_max_iter=40, newton_error_function=lambda x: np.max(np.abs(x)),\
                           numerical_jacobian=False, debug=False):
        
        self.model = model

        # integration time
        self.t0 = t0 = model.t0
        self.t1 = t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        self.dt = dt

        # parameter
        self.rho_inf = rho_inf
        if None in [beta, gamma, alpha_m, alpha_f]:
            self.alpha_m = (2 * rho_inf - 1) / (1 + rho_inf)
            self.alpha_f = rho_inf / (1 + rho_inf)
            self.gamma = 0.5 + self.alpha_f - self.alpha_m
            self.beta = 0.25 * ((self.gamma + 0.5) ** 2)
        else:
            self.gamma = gamma
            self.beta = beta
            self.alpha_m = alpha_m
            self.alpha_f = alpha_f
        self.alpha_ratio = (1 - self.alpha_f) / (1 - self.alpha_m)

        # newton settings
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.newton_error_function = newton_error_function

        # dimensions
        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = model.nla_g
        self.nla_gamma = model.nla_gamma
        self.nla_N = model.nla_N
        self.nla_T = model.nla_T

        
        # DOF's with singular mass matrix are pure algebraic DOF's
        if uDOF_algebraic is None:
            self.uDOF_algebraic = np.array([], dtype=int)
        else:
            self.uDOF_algebraic = uDOF_algebraic
        self.uDOF = np.arange(self.nu)
        self.uDOF_dynamic = np.setdiff1d(self.uDOF, self.uDOF_algebraic)
        self.nu_algebraic = len(self.uDOF_algebraic)
        self.nu_dynamic = len(self.uDOF_dynamic)

        # self.nR = 3 * self.nu + 3 * self.nla_g + self.nla_gamma
        self.nR = self.nu + self.nla_g + self.nla_gamma

        self.tk = model.t0
        self.qk = model.q0 
        self.uk = model.u0 
        self.kappa_gk = np.zeros_like(model.la_g0)
        self.la_gk = model.la_g0
        self.La_gk = np.zeros_like(model.la_g0)
        self.la_gammak = model.la_gamma0
        self.kappa_Nk = np.zeros_like(model.la_N0)

        Mk = model.M(t0, model.q0).tocsr()
        rhsk = self.model.h(t0, model.q0, model.u0) + self.model.W_g(t0, model.q0) @ model.la_g0 + self.model.W_gamma(t0, model.q0) @ model.la_gamma0
        self.ak = np.zeros(model.nu)
        self.ak[self.uDOF_dynamic] = spsolve(Mk[self.uDOF_dynamic[:, None], self.uDOF_dynamic], rhsk[self.uDOF_dynamic] )
        self.Qk = np.zeros(self.nu)
        self.Uk = np.zeros(self.nu)
        self.a_bark = self.ak.copy()

        self.debug = debug
        if numerical_jacobian:
            self.__R_gen = self.__R_gen_num
        else:
            self.__R_gen = self.__R_gen_analytic

    def __R_gen_num(self, tk1, xk1):
        yield self.__R(tk1, xk1)
        yield csc_matrix(self.__R_x_num(tk1, xk1))
        
    def __R_gen_analytic(self, tk1, xk1):
        nq = self.nq
        nu = self.nu
        nu_dyn = self.nu_dynamic
        uDOF_alg = self.uDOF_algebraic
        uDOF_dyn = self.uDOF_dynamic
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        dt = self.dt
        dt2 = self.dt**2
        
        ak1 = np.zeros(nu)
        ak1[uDOF_dyn] = xk1[:nu_dyn]
        ak1[uDOF_alg] = xk1[nu_dyn:nu]
        la_gk1 = xk1[nu:nu+nla_g]
        la_gammak1 = xk1[nu+nla_g:nu+nla_g+nla_gamma]

        # Uk1 = xk1[nu:2*nu]
        # Qk1 = xk1[2*nu:3*nu]
        # kappa_gk1 = xk1[3*nu:3*nu+nla_g]
        # La_gk1 = xk1[3*nu+nla_g:3*nu+2*nla_g]
        # la_gk1 = xk1[3*nu+2*nla_g:3*nu+3*nla_g]
        # la_gammak1 = xk1[3*nu+3*nla_g:3*nu+3*nla_g+nla_gamma]

        # update dependent variables
        a_bark1 = (self.alpha_f * self.ak + (1-self.alpha_f) * ak1  - self.alpha_m * self.a_bark) / (1 - self.alpha_m)
        uk1 = self.uk + dt * ((1-self.gamma) * self.a_bark + self.gamma * a_bark1) # + Uk1
        a_beta = (0.5 - self.beta) * self.a_bark + self.beta * a_bark1 
        qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, self.uk) + dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, a_beta) # + self.model.B(self.tk, self.qk) @ Qk1

        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csc_matrix)[uDOF_dyn[:, None], uDOF_dyn]
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csc_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csc_matrix)

        rhs = -( self.model.h(tk1, qk1, uk1) + W_gk1 @ la_gk1 + W_gammak1 @ la_gammak1 )
        
        # evaluate residual 
        # R = np.zeros(self.nR)
        # R[:nu_dyn] = Mk1 @ ak1 + rhs[uDOF_dyn]
        # R[nu_dyn:nu] = rhs[uDOF_alg]
        # R[nu:2*nu] = Mk1 @ Uk1 - W_gk1 @ La_gk1
        # R[2*nu:3*nu] = Mk1 @ Qk1 - W_gk1 @ kappa_gk1 - W_Nk1 @ kappa_Nk1
        # R[3*nu:3*nu+nla_g] = self.model.g(tk1, qk1)
        # R[3*nu+nla_g:3*nu+2*nla_g] = self.model.g_dot(tk1, qk1, uk1)
        # R[3*nu+2*nla_g:3*nu+3*nla_g] = self.model.g_ddot(tk1, qk1, uk1, ak1)
        # R[3*nu+3*nla_g:3*nu+3*nla_g+nla_gamma] = self.model.gamma(tk1, qk1, uk1)

        R = np.zeros(self.nR)
        R[:nu_dyn] = Mk1 @ ak1[uDOF_dyn] + rhs[uDOF_dyn]
        R[nu_dyn:nu] = rhs[uDOF_alg]
        R[nu:nu+nla_g] = self.model.g(tk1, qk1)
        R[nu+nla_g:nu+nla_g+nla_gamma] = self.model.gamma(tk1, qk1, uk1)
        
        yield R


        ##################################################
        # R[:nu_dyn] = Mk1 @ ak1[uDOF_dyn] + rhs[uDOF_dyn]
        ##################################################
        # rhs = -( self.model.h(tk1, qk1, uk1) + W_gk1 @ la_gk1 + W_gammak1 @ la_gammak1 )
        rhs_q = - ( self.model.h_q(tk1, qk1, uk1) + self.model.Wla_g_q(tk1, qk1, la_gk1) + self.model.Wla_gamma_q(tk1, qk1, la_gammak1) )
        rhs_u = -self.model.h_u(tk1, qk1, uk1)
        rhs_a = rhs_q @ self.q_a + rhs_u * self.u_a

        Ra_dyn_a_dyn = Mk1 + rhs_a[uDOF_dyn[:, None], uDOF_dyn]
        Ra_dyn_a_alg = rhs_a[uDOF_dyn[:, None], uDOF_alg]
        # TODO:
        # Ra_dyn_U = rhs_u
        # Ra_dyn_Q = rhs_q @ self.q_Q
        Ra_alg_a_dyn = rhs_a[uDOF_alg[:, None], uDOF_dyn]
        Ra_alg_a_alg = rhs_a[uDOF_alg[:, None], uDOF_alg]
        # TODO:
        # Ra_alg_U = rhs_u
        # Ra_alg_Q = rhs_q @ self.q_Q
        Ra_dyn_la_g = -W_gk1[uDOF_dyn]
        Ra_alg_la_g = -W_gk1[uDOF_alg]
        Ra_dyn_la_gamma = -W_gammak1[uDOF_dyn]
        Ra_alg_la_gamma = -W_gammak1[uDOF_alg]
        
        # # R[nu:2*nu] = Mk1 @ Uk1 - W_Nk1 @ La_Nk1 - W_Tk1 @ La_Tk1
        # RU_q = self.model.Mu_q(tk1, qk1, Uk1) - self.model.Wla_g_q(tk1, qk1, La_gk1) - self.model.Wla_N_q(tk1, qk1, La_Nk1) - self.model.Wla_T_q(tk1, qk1, La_Tk1)
        # RU_a = RU_q @ self.q_a
        # RU_Q = RU_q @ self.q_Q
        
        # # R[2*nu:3*nu] = Mk1 @ Qk1 - W_Nk1 @ kappa_Nk1
        # RQ_q = self.model.Mu_q(tk1, qk1, Qk1) - self.model.Wla_g_q(tk1, qk1, kappa_gk1) - self.model.Wla_N_q(tk1, qk1, kappa_Nk1)
        # RQ_a = RQ_q @ self.q_a
        # RQ_Q = Mk1 + RQ_q @ self.q_Q


        # # R[3*nu+nla_g:3*nu+2*nla_g] = self.model.g_dot(tk1, qk1, uk1)
        # RLa_g_q = self.model.g_dot_q(tk1, qk1, uk1)
        # RLa_g_u = self.model.g_dot_u(tk1, qk1)
        # RLa_g_a = RLa_g_q @ self.q_a + RLa_g_u * self.u_a
        # RLa_g_Q = RLa_g_q @ self.q_Q


        #########################################
        # R[nu:nu+nla_g] = self.model.g(tk1, qk1)
        #########################################
        Rla_g_q = self.model.g_q(tk1, qk1)
        Rla_g_a = Rla_g_q @ self.q_a
        Rla_g_a_dyn = Rla_g_a[:, uDOF_dyn]
        Rla_g_a_alg = Rla_g_a[:, uDOF_alg]
        # Rla_g_Q = Rla_g_q @ self.q_Q

        # ######################################################################
        # # R[3*nu+2*nla_g:3*nu+3*nla_g] = self.model.g_ddot(tk1, qk1, uk1, ak1)
        # ######################################################################
        # Rla_g_q = self.model.g_ddot_q(tk1, qk1, uk1, ak1)
        # Rla_g_u = self.model.g_ddot_u(tk1, qk1, uk1, ak1)
        # Rla_g_a = self.model.g_dot_u(tk1, qk1)
        # Rla_g_a += Rla_g_q @ self.q_a + Rla_g_u * self.u_a
        # Rla_g_Q = Rla_g_q @ self.q_Q 

        ##################################################################
        # R[nu+nla_g:nu+nla_g+nla_gamma] = self.model.gamma(tk1, qk1, uk1)
        ##################################################################
        Rla_gamma_q = self.model.gamma_q(tk1, qk1, uk1) 
        Rla_gamma_u = self.model.gamma_u(tk1, qk1) # == Rla_gamma_U
        Rla_gamma_a = Rla_gamma_q @ self.q_a + Rla_gamma_u * self.u_a
        Rla_gamm_a_dyn = Rla_gamma_a[:, uDOF_dyn]
        Rla_gamm_a_alg = Rla_gamma_a[:, uDOF_alg]
        # Rla_gamma_Q = Rla_gamma_q @ self.q_Q
        
        # return bmat([[Ru1_u1,       Ru1_u2,       Ru1_q,       Ru1_la_g, Ru1_la_gamma], \
        #              [Ru2_u1,       Ru2_u2,       Ru2_q,       Ru2_la_g, Ru2_la_gamma], \
        #              [Rq_u1,        Rq_u2,        Rq_q,        None,     None], \
        #              [None,         None,         Rla_g_q,     None,     None], \
        #              [Rla_gamma_u1, Rla_gamma_u2, Rla_gamma_q, None,     None]]).tocsc()

        R_x =  bmat([ [Ra_dyn_a_dyn,      Ra_dyn_a_alg, Ra_dyn_la_g, Ra_dyn_la_gamma], \
                      [Ra_alg_a_dyn,      Ra_alg_a_alg, Ra_alg_la_g, Ra_alg_la_gamma], \
                      [Rla_g_a_dyn,        Rla_g_a_alg,        None,            None], \
                      [Rla_gamm_a_dyn,  Rla_gamm_a_alg,        None,            None], \
                    ], format='csc')

        # R_x_num = self.__R_x_num(tk1, xk1)
        # diff = R_x.toarray() - R_x_num
        # error = np.max(np.abs(diff)) / np.max(np.abs(R_x_num))
        # print(f'error = {error}')

        yield R_x
        
    def __R(self, tk1, xk1):
        return next(self.__R_gen_analytic(tk1, xk1))

    def __R_x_num(self, tk1, xk1):
        # return Numerical_derivative(self.__R, order=1)._x(tk1, xk1)
        return Numerical_derivative(self.__R, order=2)._x(tk1, xk1)
    
    def step(self):
        nu_dyn = self.nu_dynamic
        nu = self.nu
        uDOF_dyn = self.uDOF_dynamic
        uDOF_alg = self.uDOF_algebraic
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        dt = self.dt
        tk1 = self.tk + dt

        # initial guess for Newton-Raphson solver
        xk1 = np.zeros(self.nR)
        xk1[:nu_dyn] = self.ak[uDOF_dyn]
        xk1[nu_dyn:nu] = self.ak[uDOF_alg]
        xk1[nu:nu+nla_g] = self.la_gk
        xk1[nu+nla_g:nu+nla_g+nla_gamma] = self.la_gammak

        # xk1[nu:2*nu] = self.Uk
        # xk1[2*nu:3*nu] = self.Qk
        # xk1[3*nu:3*nu+nla_g] = self.kappa_gk
        # xk1[3*nu+nla_g:3*nu+2*nla_g] = self.La_gk
        # xk1[3*nu+2*nla_g:3*nu+3*nla_g] = self.la_gk
        # xk1[3*nu+nla_g:3*nu+nla_g+nla_gamma] = self.la_gammak
        # xk1[self.nR_smooth:self.nR_smooth+nla_N] = self.kappa_Nk
        # xk1[self.nR_smooth+nla_N:self.nR_smooth+2*nla_N] = self.La_Nk
        # xk1[self.nR_smooth+2*nla_N:self.nR_smooth+3*nla_N] = self.la_Nk
        # xk1[self.nR_smooth+3*nla_N:self.nR_smooth+3*nla_N+nla_T] = self.La_Tk
        # xk1[self.nR_smooth+3*nla_N+nla_T:self.nR_smooth+3*nla_N+2*nla_T] = self.la_Tk

        # initial residual and error
        R_gen = self.__R_gen(tk1, xk1)
        R = next(R_gen)
        error = self.newton_error_function(R)
        converged = error < self.newton_tol
        j = 0
        if not converged:
            while j < self.newton_max_iter:
                # jacobian
                R_x = next(R_gen)
                
                # Newton update
                j += 1
                dx = spsolve(R_x, R)
                xk1 -= dx
                R_gen = self.__R_gen(tk1, xk1)
                R = next(R_gen)
                
                error = self.newton_error_function(R)
                # print(f'  * error: {error}')
                converged = error < self.newton_tol
                if converged:
                    break

        ak1 = np.zeros(nu)
        ak1[uDOF_dyn] = xk1[:nu_dyn]
        ak1[uDOF_alg] = xk1[nu_dyn:nu]
        la_gk1 = xk1[nu:nu+nla_g]
        la_gammak1 = xk1[nu+nla_g:nu+nla_g+nla_gamma]

        # Uk1 = xk1[nu:2*nu]
        # Qk1 = xk1[2*nu:3*nu]
        # kappa_gk1 = xk1[3*nu:3*nu+nla_g]
        # La_gk1 = xk1[3*nu+nla_g:3*nu+2*nla_g]
        # la_gk1 = xk1[3*nu+2*nla_g:3*nu+3*nla_g]
        # la_gammak1 = xk1[3*nu+3*nla_g:3*nu+3*nla_g+nla_gamma]
        # kappa_Nk1 = xk1[self.nR_smooth:self.nR_smooth+nla_N]
        # La_Nk1 = xk1[self.nR_smooth+nla_N:self.nR_smooth+2*nla_N]
        # la_Nk1 = xk1[self.nR_smooth+2*nla_N:self.nR_smooth+3*nla_N]
        # La_Tk1 = xk1[self.nR_smooth+3*nla_N:self.nR_smooth+3*nla_N+nla_T]
        # la_Tk1 = xk1[self.nR_smooth+3*nla_N+nla_T:self.nR_smooth+3*nla_N+2*nla_T]
            
        # return (converged, j, error), tk1, ak1, Uk1, Qk1, kappa_gk1, La_gk1, la_gk1, la_gammak1, kappa_Nk1, La_Nk1, la_Nk1, La_Tk1, la_Tk1
        return (converged, j, error), tk1, ak1, la_gk1, la_gammak1

    def solve(self): 
        dt = self.dt
        dt2 = self.dt**2

        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        a = [self.ak]
        # kappa_g = [self.kappa_gk]
        # La_g = [self.La_gk]
        la_g = [self.la_gk]
        la_gamma = [self.la_gammak]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            Bk = self.model.B(self.tk, self.qk)
            self.q_a = dt2 * self.beta * self.alpha_ratio * Bk
            self.q_Q = Bk
            self.u_a = dt * self.gamma * self.alpha_ratio
            # self.ka_ast_la_N = dt2 * self.beta * self.alpha_ratio
            # self.P_N_la_N =  dt * self.gamma * self.alpha_ratio
            # self.P_T_la_T =  dt * self.gamma * self.alpha_ratio

            # (converged, n_iter, error), tk1, ak1, Uk1, Qk1, kappa_gk1, La_gk1, la_gk1, la_gammak1, kappa_Nk1, La_Nk1, la_Nk1, La_Tk1, la_Tk1 = self.step()
            (converged, n_iter, error), tk1, ak1, la_gk1, la_gammak1 = self.step()
            pbar.set_description(f't: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.newton_max_iter} iterations; error: {error:0.2e}')
            
            if not converged:
                raise RuntimeError(f'internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}')
            dt = self.dt
            dt2 = dt * dt
            a_bark1 = (self.alpha_f * self.ak + (1-self.alpha_f) * ak1  - self.alpha_m * self.a_bark) / (1 - self.alpha_m)
            uk1 = self.uk + dt * ((1-self.gamma) * self.a_bark + self.gamma * a_bark1) #+ Uk1
            a_beta = (0.5 - self.beta) * self.a_bark + self.beta * a_bark1 
            qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, self.uk) + dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, a_beta) # + Bk @ Qk1 

            qk1, uk1 = self.model.solver_step_callback(tk1, qk1, uk1)

            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            a.append(ak1)
            # kappa_g.append(kappa_gk1)
            # La_g.append(La_gk1)
            # la_g.append(la_gk1)
            # la_gamma.append(la_gammak1)

            # update local variables for accepted time step
            self.tk = tk1
            self.qk = qk1
            self.uk = uk1
            self.ak = ak1
            # self.Qk = Qk1
            # self.kappa_gk = kappa_gk1
            # self.La_gk = La_gk1
            self.la_gk = la_gk1
            self.la_gammak = la_gammak1

        # write solution
        # return Solution(t=np.array(t), q=np.array(q), u=np.array(u), a=np.array(a), kappa_g=np.array(kappa_g), La_g=np.array(La_g), la_g=np.array(la_g), la_gamma=np.array(la_gamma), kappa_P=np.array(kappa_N), La_N=np.array(La_N), la_N=np.array(la_N), La_T=np.array(La_T), la_T=np.array(la_T), P_N=np.array(P_N), P_T=np.array(P_T))
        return Solution(t=np.array(t), q=np.array(q), u=np.array(u), a=np.array(a), la_g=np.array(la_g), la_gamma=np.array(la_gamma))