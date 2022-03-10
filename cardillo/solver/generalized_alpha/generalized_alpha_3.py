from math import inf, isnan
from numpy.core.records import fromarrays
from scipy.sparse import csc
from cardillo.math.algebra import norm
import numpy as np
from scipy.sparse.linalg import spsolve 
from scipy.sparse import coo_matrix, csc_matrix, identity, bmat, diags
from tqdm import tqdm

from cardillo.math.prox import prox_Rn0, prox_circle
from cardillo.math import Numerical_derivative
# from cardillo.utility.coo import Coo
from cardillo.solver import Solution

class Generalized_alpha_3():
    def __init__(self, model, t1, dt, \
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

        self.nR_smooth = 3 * self.nu + 3 * self.nla_g + self.nla_gamma
        self.nR = self.nR_smooth + 3 * self.nla_N + 2 * self.nla_T

        self.tk = model.t0
        self.qk = model.q0 
        self.uk = model.u0 
        self.kappa_gk = np.zeros_like(model.la_g0)
        self.la_gk = model.la_g0
        self.La_gk = np.zeros_like(model.la_g0)
        self.la_gammak = model.la_gamma0
        self.kappa_Nk = np.zeros_like(model.la_N0)
        self.la_Nk = model.la_N0
        self.La_Nk = np.zeros_like(model.la_N0)
        self.la_Tk = model.la_T0
        self.La_Tk = np.zeros_like(model.la_T0)
        self.ak = spsolve(model.M(t0, model.q0).tocsr(), self.model.h(t0, model.q0, model.u0) + self.model.W_g(t0, model.q0) @ model.la_g0 + self.model.W_gamma(t0, model.q0) @ model.la_gamma0 + self.model.W_N(t0, model.q0, scipy_matrix=csc_matrix) @ model.la_N0 + self.model.W_T(t0, model.q0, scipy_matrix=csc_matrix) @ model.la_T0 )
        self.Qk = np.zeros(self.nu)
        self.Uk = np.zeros(self.nu)
        self.a_bark = self.ak.copy()
        self.la_gbark = self.la_Nk.copy()
        self.la_Nbark = self.la_Nk.copy()
        self.la_Tbark = self.la_Tk.copy()

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
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        nla_T = self.nla_T
        dt = self.dt
        dt2 = self.dt**2
        ak1 = xk1[:nu]
        Uk1 = xk1[nu:2*nu]
        Qk1 = xk1[2*nu:3*nu]
        kappa_gk1 = xk1[3*nu:3*nu+nla_g]
        La_gk1 = xk1[3*nu+nla_g:3*nu+2*nla_g]
        la_gk1 = xk1[3*nu+2*nla_g:3*nu+3*nla_g]
        la_gammak1 = xk1[3*nu+3*nla_g:3*nu+3*nla_g+nla_gamma]
        kappa_Nk1 = xk1[self.nR_smooth:self.nR_smooth+nla_N]
        La_Nk1 = xk1[self.nR_smooth+nla_N:self.nR_smooth+2*nla_N]
        la_Nk1 = xk1[self.nR_smooth+2*nla_N:self.nR_smooth+3*nla_N]
        La_Tk1 = xk1[self.nR_smooth+3*nla_N:self.nR_smooth+3*nla_N+nla_T]
        la_Tk1 = xk1[self.nR_smooth+3*nla_N+nla_T:self.nR_smooth+3*nla_N+2*nla_T]

        # update dependent variables
        a_bark1 = (self.alpha_f * self.ak + (1-self.alpha_f) * ak1  - self.alpha_m * self.a_bark) / (1 - self.alpha_m)
        uk1 = self.uk + dt * ((1-self.gamma) * self.a_bark + self.gamma * a_bark1) + Uk1
        a_beta = (0.5 - self.beta) * self.a_bark + self.beta * a_bark1 
        qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, self.uk) + dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, a_beta) + self.model.B(self.tk, self.qk) @ Qk1 

        la_Nbark1 = (self.alpha_f * self.la_Nk + (1-self.alpha_f) * la_Nk1  - self.alpha_m * self.la_Nbark) / (1 - self.alpha_m)
        kappa_Nast = kappa_Nk1 + dt**2 * ( (0.5 - self.beta) * self.la_Nbark + self.beta * la_Nbark1 )
        P_N = La_Nk1 + dt * ((1-self.gamma) * self.la_Nbark + self.gamma * la_Nbark1)

        la_Tbark1 = (self.alpha_f * self.la_Tk + (1-self.alpha_f) * la_Tk1  - self.alpha_m * self.la_Tbark) / (1 - self.alpha_m)
        P_T = La_Tk1 + dt * ((1-self.gamma) * self.la_Tbark + self.gamma * la_Tbark1)

        Mk1 = self.model.M(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1)
        W_gammak1 = self.model.W_gamma(tk1, qk1)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csc_matrix)
        W_Tk1 = self.model.W_T(tk1, qk1, scipy_matrix=csc_matrix)

        g_N = self.model.g_N(tk1, qk1)
        xi_N = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_T = self.model.xi_T(tk1, qk1, self.uk, uk1)
        g_N_ddot_post = self.model.g_N_ddot(tk1, qk1, uk1, ak1)
        gamma_T_dot_post = self.model.gamma_T_dot(tk1, qk1, uk1, ak1)
        gamma_T_post = self.model.gamma_T(tk1, qk1, uk1)
        
        # evaluate residual 
        R = np.zeros(self.nR)
        R[:nu] = Mk1 @ ak1 - ( self.model.h(tk1, qk1, uk1) + W_gk1 @ la_gk1 + W_gammak1 @ la_gammak1 + W_Nk1 @ la_Nk1 + W_Tk1 @ la_Tk1)
        R[nu:2*nu] = Mk1 @ Uk1 - W_gk1 @ La_gk1 - W_Nk1 @ La_Nk1 - W_Tk1 @ La_Tk1
        R[2*nu:3*nu] = Mk1 @ Qk1 - W_gk1 @ kappa_gk1 - W_Nk1 @ kappa_Nk1
        R[3*nu:3*nu+nla_g] = self.model.g(tk1, qk1)
        R[3*nu+nla_g:3*nu+2*nla_g] = self.model.g_dot(tk1, qk1, uk1)
        R[3*nu+2*nla_g:3*nu+3*nla_g] = self.model.g_ddot(tk1, qk1, uk1, ak1)
        R[3*nu+3*nla_g:3*nu+3*nla_g+nla_gamma] = self.model.gamma(tk1, qk1, uk1)

        I_N = (kappa_Nast - self.model.prox_r_N * g_N >= 0)
        I_N_ind = np.where(I_N)[0]
        _I_N_ind = np.where(~I_N)[0]
        R[self.nR_smooth+I_N_ind] = g_N[I_N]
        R[self.nR_smooth+_I_N_ind] = kappa_Nast[~I_N]

        A_N_ = (P_N - self.model.prox_r_N * xi_N) >= 0
        A_N = I_N * A_N_
        A_N_ind = np.where(A_N)[0]
        _A_N_ind = np.where(~A_N)[0]
        R[self.nR_smooth+nla_N+A_N_ind] = xi_N[A_N]
        R[self.nR_smooth+nla_N+_A_N_ind] = P_N[~A_N]

        B_N_ = (la_Nk1 - self.model.prox_r_N * g_N_ddot_post) >= 0
        B_N = A_N * B_N_
        B_N_ind = np.where(B_N)[0]
        _B_N_ind = np.where(~B_N)[0]
        R[self.nR_smooth+2*nla_N+B_N_ind] = g_N_ddot_post[B_N]
        R[self.nR_smooth+2*nla_N+_B_N_ind] = la_Nk1[~B_N]

        C_N = I_N * self.model.N_has_friction
        C_T = []
        for i_N, i_T in enumerate(self.model.NT_connectivity):
            C_T.append(np.linalg.norm(P_T[i_T] - self.model.prox_r_T[i_N] * xi_T[i_T]) <= self.model.mu[i_N] * P_N[i_N])
        C_T = np.array(C_T, dtype=bool)
        N_open = ~I_N * self.model.N_has_friction
        N_stick = C_N * C_T
        N_slip = C_N * ~C_T
        N_open_ind = np.where(N_open)[0]
        N_stick_ind = np.where(N_stick)[0]
        N_slip_ind = np.where(N_slip)[0]
        T_open_ind = np.array([j for i in N_open_ind for j in self.model.NT_connectivity[i]], dtype=int)
        T_stick_ind = np.array([j for i in N_stick_ind for j in self.model.NT_connectivity[i]], dtype=int)
        T_slip_ind = np.array([j for i in N_slip_ind for j in self.model.NT_connectivity[i]], dtype=int)
        T_slip_ind_mat = np.array([self.model.NT_connectivity[i] for i in N_slip_ind], dtype=int)

        R[self.nR_smooth+3*nla_N+T_open_ind] = P_T[T_open_ind]
        R[self.nR_smooth+3*nla_N+T_stick_ind] = xi_T[T_stick_ind]
        tmp = xi_T[T_slip_ind_mat]
        norm_xi_ = np.linalg.norm(tmp, axis=-1) 
        norm_xi = np.ones_like(norm_xi_)
        norm_xi[norm_xi_>0] = norm_xi_[norm_xi_>0]
        R[self.nR_smooth+3*nla_N+T_slip_ind] = P_T[T_slip_ind] + ((self.model.mu[N_slip_ind] * P_N[N_slip_ind] / norm_xi).reshape(-1, 1) * tmp).reshape(-1)

        R[self.nR_smooth+3*nla_N+nla_T+T_open_ind] = la_Tk1[T_open_ind]
        R[self.nR_smooth+3*nla_N+nla_T+T_stick_ind] = gamma_T_dot_post[T_stick_ind]
        tmp = gamma_T_post[T_slip_ind_mat]
        norm_xi_ = np.linalg.norm(tmp, axis=-1) 
        norm_xi = np.ones_like(norm_xi_)
        norm_xi[norm_xi_>0] = norm_xi_[norm_xi_>0]
        R[self.nR_smooth+3*nla_N+nla_T+T_slip_ind] = la_Tk1[T_slip_ind] + ((self.model.mu[N_slip_ind] * la_Nk1[N_slip_ind] / norm_xi).reshape(-1, 1) * tmp).reshape(-1)
        
        yield R
        
        g_N_q = self.model.g_N_q(tk1, qk1, scipy_matrix=csc_matrix)
        g_N_dot_u = self.model.g_N_dot_u(tk1, qk1, scipy_matrix=csc_matrix)
        xi_N_q = self.model.xi_N_q(tk1, qk1, self.uk, uk1, scipy_matrix=csc_matrix)
        g_N_ddot_post_q = self.model.g_N_ddot_q(tk1, qk1, uk1, ak1, scipy_matrix=csc_matrix)
        g_N_ddot_post_u = self.model.g_N_ddot_u(tk1, qk1, uk1, ak1, scipy_matrix=csc_matrix)
        xi_T_q = self.model.xi_T_q(tk1, qk1, self.uk, uk1, scipy_matrix=csc_matrix)
        xi_T_u = gamma_T_u = self.model.gamma_T_u(tk1, qk1, scipy_matrix=csc_matrix)
        gamma_T_q = self.model.gamma_T_q(tk1, qk1, uk1, scipy_matrix=csc_matrix)
        gamma_T_dot_post_q = self.model.gamma_T_dot_q(tk1, qk1, uk1, ak1, scipy_matrix=csc_matrix)
        gamma_T_dot_post_u = self.model.gamma_T_dot_u(tk1, qk1, uk1, ak1, scipy_matrix=csc_matrix)

        # R[:nu] = Mk1 @ ak1 - ( self.model.h(tk1, qk1, uk1) + W_gk1 @ la_gk1 + W_gammak1 @ la_gammak1 + W_Nk1 @ la_Nk1 + W_Tk1 @ la_Tk1)
        Ra_q = self.model.Mu_q(tk1, qk1, ak1) - (self.model.h_q(tk1, qk1, uk1) + self.model.Wla_g_q(tk1, qk1, la_gk1) + self.model.Wla_gamma_q(tk1, qk1, la_gammak1) + self.model.Wla_N_q(tk1, qk1, la_Nk1) + self.model.Wla_T_q(tk1, qk1, la_Tk1))
        Ra_u = -self.model.h_u(tk1, qk1, uk1)
        Ra_a = Mk1 + Ra_q @ self.q_a + Ra_u * self.u_a
        Ra_U = Ra_u
        Ra_Q = Ra_q @ self.q_Q
        
        # R[nu:2*nu] = Mk1 @ Uk1 - W_Nk1 @ La_Nk1 - W_Tk1 @ La_Tk1
        RU_q = self.model.Mu_q(tk1, qk1, Uk1) - self.model.Wla_g_q(tk1, qk1, La_gk1) - self.model.Wla_N_q(tk1, qk1, La_Nk1) - self.model.Wla_T_q(tk1, qk1, La_Tk1)
        RU_a = RU_q @ self.q_a
        RU_Q = RU_q @ self.q_Q
        
        # R[2*nu:3*nu] = Mk1 @ Qk1 - W_Nk1 @ kappa_Nk1
        RQ_q = self.model.Mu_q(tk1, qk1, Qk1) - self.model.Wla_g_q(tk1, qk1, kappa_gk1) - self.model.Wla_N_q(tk1, qk1, kappa_Nk1)
        RQ_a = RQ_q @ self.q_a
        RQ_Q = Mk1 + RQ_q @ self.q_Q

        # R[3*nu:3*nu+nla_g] = self.model.g(tk1, qk1)
        Rka_g_q = self.model.g_q(tk1, qk1)
        Rka_g_a = Rka_g_q @ self.q_a
        Rka_g_Q = Rka_g_q @ self.q_Q

        # R[3*nu+nla_g:3*nu+2*nla_g] = self.model.g_dot(tk1, qk1, uk1)
        RLa_g_q = self.model.g_dot_q(tk1, qk1, uk1)
        RLa_g_u = self.model.g_dot_u(tk1, qk1)
        RLa_g_a = RLa_g_q @ self.q_a + RLa_g_u * self.u_a
        RLa_g_Q = RLa_g_q @ self.q_Q 

        # R[3*nu+2*nla_g:3*nu+3*nla_g] = self.model.g_ddot(tk1, qk1, uk1, ak1)
        Rla_g_q = self.model.g_ddot_q(tk1, qk1, uk1, ak1)
        Rla_g_u = self.model.g_ddot_u(tk1, qk1, uk1, ak1)
        Rla_g_a = self.model.g_dot_u(tk1, qk1)
        Rla_g_a += Rla_g_q @ self.q_a + Rla_g_u * self.u_a
        Rla_g_Q = Rla_g_q @ self.q_Q 

        # R[3*nu+nla_g:3*nu+nla_g+nla_gamma] = self.model.gamma(tk1, qk1, uk1)
        Rla_gamma_q = self.model.gamma_q(tk1, qk1, uk1) 
        Rla_gamma_u = self.model.gamma_u(tk1, qk1) # == Rla_gamma_U
        Rla_gamma_a = Rla_gamma_q @ self.q_a + Rla_gamma_u * self.u_a
        Rla_gamma_Q = Rla_gamma_q @ self.q_Q

        # R[3*nu+nla_g+nla_gamma:3*nu+nla_g+nla_gamma+nla_N] = kappa_ast - prox_Rn0(kappa_ast - self.model.prox_r_N * g_N)
        row = col = np.where(~I_N)[0]
        data = np.ones_like(row)
        Rka_ka_ast = coo_matrix((data, (row, col)), shape=(nla_N, nla_N))

        # Rka_q = (diags(self.model.prox_r_N) @ g_N_q)[I_N].tocoo()
        Rka_q = g_N_q[I_N].tocoo()
        Rka_q.resize(nla_N, nq)
        Rka_q.row = np.where(I_N)[0][Rka_q.row]
        
        Rka_a = Rka_q @ self.q_a
        Rka_Q = Rka_q @ self.q_Q
        Rka_ka = Rka_ka_ast
        Rka_la_N = Rka_ka_ast * self.ka_ast_la_N

        # R[3*nu+nla_g+nla_gamma+nla_N+A_N_ind] = xi_N[A_N]
        # R[3*nu+nla_g+nla_gamma+nla_N+_A_N_ind] = P_N[~A_N]
        row = col = _A_N_ind
        data = np.ones_like(row)
        RLaN_P_N = coo_matrix((data, (row, col)), shape=(nla_N, nla_N))
        
        RLaN_u = g_N_dot_u[A_N].tocoo()
        RLaN_u.resize(nla_N, nu)
        RLaN_u.row = A_N_ind[RLaN_u.row]

        RLaN_q = xi_N_q[A_N].tocoo()
        RLaN_q.resize(nla_N, nq)
        RLaN_q.row = A_N_ind[RLaN_q.row]

        RLaN_a = RLaN_q @ self.q_a + RLaN_u * self.u_a
        RLaN_Q = RLaN_q @ self.q_Q
        RLaN_La_N = RLaN_P_N
        RLaN_la_N = RLaN_P_N * self.P_N_la_N

        # R[3*nu+nla_g+nla_gamma+2*nla_N+B_N_ind] = g_N_ddot_post[B_N]
        # R[3*nu+nla_g+nla_gamma+2*nla_N+_B_N_ind] = la_Nk1[~B_N]
        row = col = _B_N_ind
        data = np.ones_like(row)
        RlaN_la_N = coo_matrix((data, (row, col)), shape=(nla_N, nla_N))

        RlaN_a = g_N_dot_u[B_N].tocoo()
        RlaN_a.resize(nla_N, nu)
        RlaN_a.row = B_N_ind[RlaN_a.row]

        RlaN_u = g_N_ddot_post_u[B_N].tocoo()
        RlaN_u.resize(nla_N, nu)
        RlaN_u.row = B_N_ind[RlaN_u.row]

        RlaN_q = g_N_ddot_post_q[B_N].tocoo()
        RlaN_q.resize(nla_N, nq)
        RlaN_q.row = B_N_ind[RlaN_q.row]

        RlaN_a += RlaN_q @ self.q_a + RlaN_u * self.u_a
        RlaN_Q = RlaN_q @ self.q_Q
        
        # tmp = xi_T[T_slip_ind_mat]
        # norm_xi_ = np.linalg.norm(tmp, axis=-1) 
        # norm_xi = np.ones_like(norm_xi_)
        # norm_xi[norm_xi_>0] = norm_xi_[norm_xi_>0]
        # R[T_open_ind] = P_T[T_open_ind]
        # R[T_stick_ind] = xi_T[T_stick_ind]
        # R[T_slip_ind] = P_T[T_slip_ind] + ((self.model.mu[N_slip_ind] * P_N[N_slip_ind] / norm_xi).reshape(-1, 1) * tmp).reshape(-1)
        row = col = np.concatenate( (T_open_ind, T_slip_ind) )
        data = np.ones_like(row)
        RLaT_P_T = coo_matrix((data, (row, col)), shape=(nla_T, nla_T))

        RLaT_u = xi_T_u[T_stick_ind].tocoo()
        RLaT_u.resize(nla_T, nu)
        RLaT_u.row = T_stick_ind[RLaT_u.row]

        RLaT_q = xi_T_q[T_stick_ind].tocoo()
        RLaT_q.resize(nla_T, nq)
        RLaT_q.row = T_stick_ind[RLaT_q.row]

        RLaT_P_N = coo_matrix( (nla_T, nla_N) )

        u_data = []
        u_row = []
        u_col = []
        q_data = []
        q_row = []
        q_col = []
        P_N_data = []
        P_N_row = []
        P_N_col = []
        for i_N in N_slip_ind:
            i_T = self.model.NT_connectivity[i_N]
            xi_T_loc = xi_T[i_T]
            xi_T_u_loc = xi_T_u[i_T]
            xi_T_q_loc = xi_T_q[i_T]
            norm_T = np.linalg.norm(xi_T_loc)
            norm_T2 = norm_T**2

            if norm_T > 0:
                tmp_u = (self.model.mu[i_N] * P_N[i_N] / norm_T) * (xi_T_u_loc  - np.outer(xi_T_loc / norm_T2, xi_T_loc @ xi_T_u_loc) )
                tmp_q = (self.model.mu[i_N] * P_N[i_N] / norm_T) * (xi_T_q_loc  - np.outer(xi_T_loc / norm_T2, xi_T_loc @ xi_T_q_loc) )
                tmp_P_N = (self.model.mu[i_N] / norm_T) * xi_T_loc
            else:
                tmp_u = (self.model.mu[i_N] * P_N[i_N]) * xi_T_u_loc.toarray() 
                tmp_q = (self.model.mu[i_N] * P_N[i_N]) * xi_T_q_loc.toarray()
                tmp_P_N = (self.model.mu[i_N]) * xi_T_loc

            u_data.extend(np.asarray(tmp_u).reshape(-1, order='C').tolist())
            u_row.extend(np.repeat(i_T, nu).tolist())
            u_col.extend(np.tile(np.arange(nu), len(i_T)).tolist())
                
            q_data.extend(np.asarray(tmp_q).reshape(-1, order='C').tolist())
            q_row.extend(np.repeat(i_T, nq).tolist())
            q_col.extend(np.tile(np.arange(nq), len(i_T)).tolist())

            P_N_data.extend(tmp_P_N.tolist())
            P_N_row.extend(i_T)
            P_N_col.extend((i_N * np.ones_like(i_T)).tolist())

        RLaT_u.data = np.append(RLaT_u.data, u_data)
        RLaT_u.row = np.append(RLaT_u.row, u_row).astype(int)
        RLaT_u.col = np.append(RLaT_u.col, u_col).astype(int)

        RLaT_q.data = np.append(RLaT_q.data, q_data)
        RLaT_q.row = np.append(RLaT_q.row, q_row).astype(int)
        RLaT_q.col = np.append(RLaT_q.col, q_col).astype(int)

        RLaT_P_N.data = np.append(RLaT_P_N.data, P_N_data)
        RLaT_P_N.row = np.append(RLaT_P_N.row, P_N_row).astype(int)
        RLaT_P_N.col = np.append(RLaT_P_N.col, P_N_col).astype(int)

        RLaT_a = RLaT_u * self.u_a + RLaT_q @ self.q_a
        RLaT_U = RLaT_u
        RLaT_Q = RLaT_q @ self.q_Q
        RLaT_La_N = RLaT_P_N
        RLaT_la_N = RLaT_P_N * self.P_N_la_N
        RLaT_La_T = RLaT_P_T
        RLaT_la_T = RLaT_P_T * self.P_T_la_T

        # tmp = gamma_T_post[T_slip_ind_mat]
        # norm_xi_ = np.linalg.norm(tmp, axis=-1) 
        # norm_xi = np.ones_like(norm_xi_)
        # norm_xi[norm_xi_>0] = norm_xi_[norm_xi_>0]
        # R[3*nu+nla_g+nla_gamma+3*nla_N+nla_T+T_open_ind] = la_Tk1[T_open_ind]
        # R[3*nu+nla_g+nla_gamma+3*nla_N+nla_T+T_stick_ind] = gamma_T_dot_post[T_stick_ind]
        # R[3*nu+nla_g+nla_gamma+3*nla_N+nla_T+T_slip_ind] = la_Tk1[T_slip_ind] + ((self.model.mu[N_slip_ind] * la_Nk1[N_slip_ind] / norm_xi).reshape(-1, 1) * tmp).reshape(-1)
        row = col = np.concatenate( (T_open_ind, T_slip_ind) )
        data = np.ones_like(row)
        RlaT_la_T = coo_matrix((data, (row, col)), shape=(nla_T, nla_T))

        RlaT_a = gamma_T_u[T_stick_ind].tocoo()
        RlaT_a.resize(nla_T, nu)
        RlaT_a.row = T_stick_ind[RlaT_a.row]

        RlaT_u = gamma_T_dot_post_u[T_stick_ind].tocoo()
        RlaT_u.resize(nla_T, nu)
        RlaT_u.row = T_stick_ind[RlaT_u.row]

        RlaT_q = gamma_T_dot_post_q[T_stick_ind].tocoo()
        RlaT_q.resize(nla_T, nq)
        RlaT_q.row = T_stick_ind[RlaT_q.row]

        RlaT_la_N = coo_matrix( (nla_T, nla_N) )

        u_data = []
        u_row = []
        u_col = []
        q_data = []
        q_row = []
        q_col = []
        la_N_data = []
        la_N_row = []
        la_N_col = []
        for i_N in N_slip_ind:
            i_T = self.model.NT_connectivity[i_N]
            gamma_T_loc = gamma_T_post[i_T]
            gamma_T_u_loc = gamma_T_u[i_T]
            gamma_T_q_loc = gamma_T_q[i_T]
            norm_T = np.linalg.norm(gamma_T_loc)
            norm_T2 = norm_T**2

            if norm_T > 0:
                tmp_u = (self.model.mu[i_N] * la_Nk1[i_N] / norm_T) * (gamma_T_u_loc  - np.outer(gamma_T_loc / norm_T2, gamma_T_loc @ gamma_T_u_loc) )
                tmp_q = (self.model.mu[i_N] * la_Nk1[i_N] / norm_T) * (gamma_T_q_loc  - np.outer(gamma_T_loc / norm_T2, gamma_T_loc @ gamma_T_q_loc) )
                tmp_la_N = (self.model.mu[i_N] / norm_T) * gamma_T_loc
            else:
                tmp_u = (self.model.mu[i_N] * la_Nk1[i_N]) * gamma_T_u_loc.toarray()
                tmp_q = (self.model.mu[i_N] * la_Nk1[i_N]) * gamma_T_q_loc.toarray()
                tmp_la_N = (self.model.mu[i_N]) * gamma_T_loc

            u_data.extend(np.asarray(tmp_u).reshape(-1, order='C').tolist())
            u_row.extend(np.repeat(i_T, nu).tolist())
            u_col.extend(np.tile(np.arange(nu), len(i_T)).tolist())
                
            q_data.extend(np.asarray(tmp_q).reshape(-1, order='C').tolist())
            q_row.extend(np.repeat(i_T, nq).tolist())
            q_col.extend(np.tile(np.arange(nq), len(i_T)).tolist())

            la_N_data.extend(tmp_la_N.tolist())
            la_N_row.extend(i_T)
            la_N_col.extend((i_N * np.ones_like(i_T)).tolist())

        RlaT_u.data = np.append(RlaT_u.data, u_data)
        RlaT_u.row = np.append(RlaT_u.row, u_row).astype(int)
        RlaT_u.col = np.append(RlaT_u.col, u_col).astype(int)

        RlaT_q.data = np.append(RlaT_q.data, q_data)
        RlaT_q.row = np.append(RlaT_q.row, q_row).astype(int)
        RlaT_q.col = np.append(RlaT_q.col, q_col).astype(int)

        RlaT_la_N.data = np.append(RlaT_la_N.data, la_N_data)
        RlaT_la_N.row = np.append(RlaT_la_N.row, la_N_row).astype(int)
        RlaT_la_N.col = np.append(RlaT_la_N.col, la_N_col).astype(int)

        RlaT_a += RlaT_u * self.u_a + RlaT_q @ self.q_a
        RlaT_U = RlaT_u
        RlaT_Q = RlaT_q @ self.q_Q

        R_x =  bmat([[Ra_a, Ra_U, Ra_Q, None, None, -W_gk1,  -W_gammak1, None,     None, -W_Nk1,   None, -W_Tk1], \
                         [RU_a,  Mk1, RU_Q, None, -W_gk1,  None,        None, None,   -W_Nk1,   None, -W_Tk1,   None], \
                         [RQ_a,  None, RQ_Q,  -W_gk1, None, None,        None, -W_Nk1,   None,   None,   None,   None], \
                         [Rka_g_a,  None, Rka_g_Q,   None, None,None,        None, None,   None,   None,   None,   None], \
                         [RLa_g_a,  RLa_g_u, RLa_g_Q,  None, None, None,        None, None,   None,   None,   None,   None], \
                         [Rla_g_a,  Rla_g_u, Rla_g_Q,   None, None,None,        None, None,   None,   None,   None,   None], \
                         [Rla_gamma_a, Rla_gamma_u, Rla_gamma_Q, None, None,None, None, None, None, None, None, None], \
                         [Rka_a, None, Rka_Q, None, None,None, None, Rka_ka,  None, Rka_la_N, None, None], \
                         [RLaN_a, RLaN_u, RLaN_Q, None, None,None, None, None,  RLaN_La_N, RLaN_la_N, None, None], \
                         [RlaN_a, RlaN_u, RlaN_Q, None, None,None, None, None,  None, RlaN_la_N, None, None], \
                         [RLaT_a, RLaT_U, RLaT_Q, None, None,None, None, None,  RLaT_La_N, RLaT_la_N, RLaT_La_T, RLaT_la_T], \
                         [RlaT_a, RlaT_U, RlaT_Q, None, None,None, None, None,  None, RlaT_la_N, None, RlaT_la_T] \
                         ], format='csc')

        # R_x_num = self.__R_x_num(tk1, xk1)
        # diff = R_x.toarray() - R_x_num
        # # # error = np.linalg.norm(diff, ord=inf)
        # # # error = np.linalg.norm(diff[:nu], ord=inf)
        # error = np.max(np.abs(diff)) / np.max(np.abs(R_x_num))
        # print(f'error = {error}')

        yield R_x
        
    def __R(self, tk1, xk1):
        return next(self.__R_gen_analytic(tk1, xk1))

    def __R_x_num(self, tk1, xk1):
        return Numerical_derivative(self.__R, order=2)._x(tk1, xk1)
    
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
        xk1[3*nu:3*nu+nla_g] = self.kappa_gk
        xk1[3*nu+nla_g:3*nu+2*nla_g] = self.La_gk
        xk1[3*nu+2*nla_g:3*nu+3*nla_g] = self.la_gk
        xk1[3*nu+3*nla_g:3*nu+3*nla_g+nla_gamma] = self.la_gammak
        xk1[self.nR_smooth:self.nR_smooth+nla_N] = self.kappa_Nk
        xk1[self.nR_smooth+nla_N:self.nR_smooth+2*nla_N] = self.La_Nk
        xk1[self.nR_smooth+2*nla_N:self.nR_smooth+3*nla_N] = self.la_Nk
        xk1[self.nR_smooth+3*nla_N:self.nR_smooth+3*nla_N+nla_T] = self.La_Tk
        xk1[self.nR_smooth+3*nla_N+nla_T:self.nR_smooth+3*nla_N+2*nla_T] = self.la_Tk

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
                # try:
                #     dx = spsolve(R_x, R)
                # except:
                #     print('Fehler!!!!')
                xk1 -= dx
                R_gen = self.__R_gen(tk1, xk1)
                R = next(R_gen)
                
                error = self.newton_error_function(R)
                converged = error < self.newton_tol
                if converged:
                    break
        ak1 = xk1[:nu]
        Uk1 = xk1[nu:2*nu]
        Qk1 = xk1[2*nu:3*nu]
        kappa_gk1 = xk1[3*nu:3*nu+nla_g]
        La_gk1 = xk1[3*nu+nla_g:3*nu+2*nla_g]
        la_gk1 = xk1[3*nu+2*nla_g:3*nu+3*nla_g]
        la_gammak1 = xk1[3*nu+3*nla_g:3*nu+3*nla_g+nla_gamma]
        kappa_Nk1 = xk1[self.nR_smooth:self.nR_smooth+nla_N]
        La_Nk1 = xk1[self.nR_smooth+nla_N:self.nR_smooth+2*nla_N]
        la_Nk1 = xk1[self.nR_smooth+2*nla_N:self.nR_smooth+3*nla_N]
        La_Tk1 = xk1[self.nR_smooth+3*nla_N:self.nR_smooth+3*nla_N+nla_T]
        la_Tk1 = xk1[self.nR_smooth+3*nla_N+nla_T:self.nR_smooth+3*nla_N+2*nla_T]
            
        return (converged, j, error), tk1, ak1, Uk1, Qk1, kappa_gk1, La_gk1, la_gk1, la_gammak1, kappa_Nk1, La_Nk1, la_Nk1, La_Tk1, la_Tk1

    def solve(self): 
        dt = self.dt
        dt2 = self.dt**2

        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        a = [self.ak]
        kappa_g = [self.kappa_gk]
        La_g = [self.La_gk]
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
            Bk = self.model.B(self.tk, self.qk)
            self.q_a = dt2 * self.beta * self.alpha_ratio * Bk
            self.q_Q = Bk
            self.u_a = dt * self.gamma * self.alpha_ratio
            self.ka_ast_la_N = dt2 * self.beta * self.alpha_ratio
            self.P_N_la_N =  dt * self.gamma * self.alpha_ratio
            self.P_T_la_T =  dt * self.gamma * self.alpha_ratio

            (converged, n_iter, error), tk1, ak1, Uk1, Qk1, kappa_gk1, La_gk1, la_gk1, la_gammak1, kappa_Nk1, La_Nk1, la_Nk1, La_Tk1, la_Tk1 = self.step()
            pbar.set_description(f't: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.newton_max_iter} iterations; error: {error:0.2e}')
            
            if not converged:
                raise RuntimeError(f'internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}')
            dt = self.dt
            dt2 = dt * dt
            a_bark1 = (self.alpha_f * self.ak + (1-self.alpha_f) * ak1  - self.alpha_m * self.a_bark) / (1 - self.alpha_m)
            uk1 = self.uk + dt * ((1-self.gamma) * self.a_bark + self.gamma * a_bark1) + Uk1
            a_beta = (0.5 - self.beta) * self.a_bark + self.beta * a_bark1 
            qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, self.uk) + dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, a_beta) + Bk @ Qk1 

            la_Nbark1 = (self.alpha_f * self.la_Nk + (1-self.alpha_f) * la_Nk1  - self.alpha_m * self.la_Nbark) / (1 - self.alpha_m)
            P_N_ = La_Nk1 + dt * ((1-self.gamma) * self.la_Nbark + self.gamma * la_Nbark1)
            
            la_Tbark1 = (self.alpha_f * self.la_Tk + (1-self.alpha_f) * la_Tk1  - self.alpha_m * self.la_Tbark) / (1 - self.alpha_m)
            P_T_ = La_Tk1 + dt * ((1-self.gamma) * self.la_Tbark + self.gamma * la_Tbark1)

            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            a.append(ak1)
            kappa_g.append(kappa_gk1)
            La_g.append(La_gk1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)
            kappa_N.append(kappa_Nk1)
            La_N.append(La_Nk1)
            la_N.append(la_Nk1)
            La_T.append(La_Tk1)
            la_T.append(la_Tk1)
            P_N.append( P_N_ )
            P_T.append( P_T_ )

            # update local variables for accepted time step
            self.tk = tk1
            self.qk = qk1
            self.uk = uk1
            self.ak = ak1
            self.Qk = Qk1
            self.kappa_gk = kappa_gk1
            self.La_gk = La_gk1
            self.la_gk = la_gk1
            self.la_gammak = la_gammak1
            self.kappa_Nk = kappa_Nk1
            self.La_Nk = La_Nk1
            self.la_Nk = la_Nk1
            self.La_Tk = La_Tk1
            self.la_Tk = la_Tk1
            self.a_bark = a_bark1
            self.la_Nbark = la_Nbark1
            self.la_Tbark = la_Tbark1

        # write solution
        return Solution(t=np.array(t), q=np.array(q), u=np.array(u), a=np.array(a), kappa_g=np.array(kappa_g), La_g=np.array(La_g), la_g=np.array(la_g), la_gamma=np.array(la_gamma), kappa_P=np.array(kappa_N), La_N=np.array(La_N), la_N=np.array(la_N), La_T=np.array(La_T), la_T=np.array(la_T), P_N=np.array(P_N), P_T=np.array(P_T))