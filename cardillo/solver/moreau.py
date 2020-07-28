import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, bmat, eye
from tqdm import tqdm

from cardillo.math.numerical_derivative import Numerical_derivative
from cardillo.solver import Solution
from cardillo.math.prox import prox_Rn0, prox_circle
from cardillo.math.algebra import norm2

class Moreau():
    def __init__(self, model, t1, dt, fix_point_tol=1e-5, fix_point_max_iter=1000, prox_solver_method='fixed-point'):
        self.model = model

        # integration time
        t0 = model.t0
        self.t1 = t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.fix_point_error_function = lambda x: np.max(np.abs(x)) / len(x)
        self.fix_point_tol = fix_point_tol
        self.fix_point_max_iter = fix_point_max_iter

        self.newton_tol = 1e-6
        self.newton_max_iter = 100
        self.newton_error_function =lambda x: np.max(np.abs(x)) / len(x)

        self.nq = self.model.nq
        self.nu = self.model.nu
        self.nla_g = self.model.nla_g
        self.nla_gamma = self.model.nla_gamma
        self.nla_N = self.model.nla_N
        self.nla_T = self.model.nla_T
        self.nR_smooth = self.nu + self.nla_g + self.nla_gamma
        self.nR = self.nR_smooth + self.nla_N + self.nla_T

        # TODO:
        self.NT_connectivity = self.model.NT_connectivity

        self.DOFs_smooth = np.arange(self.nR_smooth)

        if prox_solver_method == 'fixed-point':
            self.step = self.__step_fixed_point
        elif prox_solver_method == 'newton':
            self.step = self.__step_newton

    # def __step_prox(self, tk, qk, uk, la_Nk, la_Tk):
    def __step_fixed_point(self, tk, qk, uk, la_gk, la_gammak, la_Nk, la_Tk):
        # general quantities
        dt = self.dt

        tk1 = tk + dt
        qk1 = qk + dt * self.model.q_dot(tk, qk, uk)

        M = self.model.M(tk1, qk1)
        h = self.model.h(tk1, qk1, uk)
        W_g = self.model.W_g(tk1, qk1)
        W_gamma = self.model.W_gamma(tk1, qk1)
        W_N = self.model.W_N(tk1, qk1, scipy_matrix=csc_matrix)
        W_T = self.model.W_T(tk1, qk1, scipy_matrix=csc_matrix)
        g_dot_u = self.model.g_dot_u(tk1, qk1)
        chi_g = self.model.chi_g(tk1, qk1)
        gamma_u = self.model.gamma_u(tk1, qk1)
        chi_gamma = self.model.chi_gamma(tk1, qk1)

        # identify active normal and tangential contacts
        g_N = self.model.g_N(tk1, qk1)
        I_N = (g_N <= 0)
        if np.any(I_N):
            I_T = np.array([c for i, I_N_i in enumerate(I_N) for c in self.model.NT_connectivity[i] if I_N_i], dtype=int)
        else:
            I_T = np.array([], dtype=int)

        # solve for new velocities and bilateral constraint forces
        # M (uk1 - uk) - dt (h + W_g la_g + W_gamma la_gamma + W_gN la_N + W_gT la_T) = 0
        # g_dot_u @ uk1 + chi_g = 0
        # gamma_u @ uk1 + chi_gamma = 0
        A =  bmat([[M      ,  -dt * W_g, -dt * W_gamma], \
                   [g_dot_u,       None,          None], \
                   [gamma_u,       None,          None]]).tocsr()

        b = np.concatenate( (M @ uk + dt * h + W_N[:, I_N] @ la_Nk[I_N] + W_T[:, I_T] @ la_Tk[I_T],\
                             -chi_g,\
                             -chi_gamma) )

        x = spsolve(A, b)
        uk1 = x[:self.nu]
        la_gk1 = x[self.nu:self.nu+self.nla_g]
        la_gammak1 = x[self.nu+self.nla_g:]

        la_Nk1 = np.zeros_like(la_Nk)
        la_Tk1 = np.zeros_like(la_Tk)

        converged = True
        error = 0
        j = 0
        if np.any(I_N):
            converged = False
            la_Nk0 = la_Nk.copy()
            la_Nk1_i = la_Nk.copy()
            la_Tk0 = la_Tk.copy()
            la_Tk1_i = la_Tk.copy()
            for j in range(self.fix_point_max_iter):
                
                # relative contact velocity and prox equation
                la_Nk1_i, la_Tk1_i = self.model.contact_force_fixpoint_update(tk1, qk1, uk, uk1, la_Nk1_i, la_Tk1_i, I_N=I_N)

                # check if velocities or contact percussions do not change
                # error = self.fix_point_error_function(uk1 - uk0)
                R = np.concatenate( (la_Nk1_i[I_N] - la_Nk0[I_N], la_Tk1_i[I_T] - la_Tk0[I_T]) )
                error = self.fix_point_error_function(R)
                converged = error < self.fix_point_tol
                if converged:
                    la_Nk1[I_N] = la_Nk1_i[I_N]
                    la_Tk1[I_T] = la_Tk1_i[I_T]
                    break
                la_Nk0 = la_Nk1_i.copy()
                la_Tk0 = la_Tk1_i.copy()

                # solve for new velocities and bilateral constraint forces
                A =  bmat([[M      ,  -dt * W_g, -dt * W_gamma], \
                        [g_dot_u,       None,          None], \
                        [gamma_u,       None,          None]]).tocsr()

                b = np.concatenate( (M @ uk + dt * h + W_N[:, I_N] @ la_Nk1_i[I_N] + W_T[:, I_T] @ la_Tk1_i[I_T],\
                                     -chi_g,\
                                     -chi_gamma) )

                x = spsolve(A, b)
                uk1 = x[:self.nu]
                la_gk1 = x[self.nu:self.nu+self.nla_g]
                la_gammak1 = x[self.nu+self.nla_g:]
                
        return (converged, j, error), tk1, qk1, uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1

    def __step_newton(self, tk, qk, uk, la_gk, la_gammak, la_Nk, la_Tk):
        # general quantities
        dt = self.dt

        tk1 = tk + dt
        qk1 = qk + dt * self.model.q_dot(tk, qk, uk)

        self.qk1 = qk1
        self.uk = uk

        # TODO: use e_N on subsystem level!
        self.e_N = e_N = 0.1
        self.e_T = e_T = 0
        r_N = 0.01
        self.r_T = r_T = 0.01
        self.mu = mu = 0.1
        # initial residual and error
        R = np.zeros(self.nR)
        uk1 = uk.copy() 
        la_gk1 = la_gk.copy()
        la_gammak1 = la_gammak.copy()
        # la_Nk1 = la_Nk.copy()
        # la_Tk1 = la_Tk.copy()
        la_Nk1 = np.zeros_like(la_Nk)
        la_Tk1 = np.zeros_like(la_Tk)
        # la_Nk1 = np.ones_like(la_Nk)
        # la_Tk1 = np.ones_like(la_Tk)
        xk1 = np.concatenate( (uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1) )
        # identify active normal and tangential contacts
        g_N = self.model.g_N(tk1, qk1)
        self.I_N = I_N = (g_N <= 0)
        if np.any(I_N):
            self.I_T = np.array([c for i, I_N_i in enumerate(I_N) for c in self.model.NT_connectivity[i] if I_N_i], dtype=int)
        else:
            self.I_T = np.array([], dtype=int)
        
        # TODO: compute xi on model and subsystem level
        g_N_dotk = self.model.g_N_dot(tk1, qk1, uk)
        gamma_Tk = self.model.gamma_T(tk1, qk1, uk)
        gamma_Tk1 = self.model.gamma_T(tk1, qk1, uk1)
        
        xi_N = self.model.g_N_dot(tk1, qk1, uk1) + e_N * g_N_dotk
        # divide active set into both parts of the prox equation
        # S_N contains the parts where the contact velocities and the gaps are <= 0
        A_N = (la_Nk1- r_N * xi_N >= 0) * I_N
        self.A_N_ = A_N_ = np.where( ~A_N )[0]
        self.A_N = A_N = np.where(A_N)[0]
        
        B_N = []
        for i_N, i_T in enumerate(self.NT_connectivity):
            if np.any(i_T):
                P_N = la_Nk1[i_N]
                P_T = la_Tk1[i_T]
                xi_T = gamma_Tk1[i_T] + e_T * gamma_Tk[i_T]
                B_N.append(norm2(P_T - r_T * xi_T) <= mu * P_N)
            else:
                B_N.append(False)

        self.B_N = np.array(B_N)
        B_N_old = np.array(B_N)


        R = self.__R_newton(tk1, uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1)

        error = self.newton_error_function(R)
        converged = error < self.newton_tol and np.all(self.B_N == B_N_old)
        j = 0
        if not converged:
            while j < self.newton_max_iter:
                # # jacobian
                # R_x =  bmat([[M      ,        -dt * W_g, -dt * W_gamma, -W_N[:, A_N], None], \
                #              [g_dot_u,        None     , None         , None        , None], \
                #              [gamma_u,        None     , None         , None        , None], \
                #              [g_N_dot_u[A_N], None     , None         , None        , None],
                #              [None          , None     , None         , None        , eye(len(A_N_), format="coo")]]).tocsc()

                R_x = Numerical_derivative(self.__R_newton_wrapper, order=2)._x(tk1, xk1)

                # Newton update
                j += 1

                try:
                    dx = spsolve(R_x, R)
                except:
                    R = self.__R_newton(tk1, uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1)
                
                xk1 -= dx
                uk1 = xk1[:self.nu]
                la_gk1 = xk1[self.nu:self.nu+self.nla_g]
                la_gammak1 = xk1[self.nu+self.nla_g:self.nu+self.nla_g+self.nla_gamma]
                la_Nk1[A_N] = xk1[self.nR_smooth:self.nR_smooth+len(A_N)]
                la_Nk1[A_N_] = xk1[self.nR_smooth+len(A_N):self.nR_smooth+self.nla_N]

                offset = 0
                for i_N, i_T in enumerate(self.NT_connectivity):
                    if np.any(i_T):
                        if self.I_N[i_N]:
                            if self.B_N[i_N]:
                                la_Tk1[i_T] = xk1[self.nR_smooth+self.nla_N+offset:self.nR_smooth+self.nla_N+offset+2]
                            else:
                                la_Tk1[i_T] = xk1[self.nR_smooth+self.nla_N+offset:self.nR_smooth+self.nla_N+offset+2]
                        else:
                            la_Tk1[i_T] = xk1[self.nR_smooth+self.nla_N+offset:self.nR_smooth+self.nla_N+offset+2]
                        offset += 2 
                
                # TODO: compute xi on model and subsystem level
                xi_N = self.model.g_N_dot(tk1, qk1, uk1) + e_N * g_N_dotk
                # divide active set into both parts of the prox equation
                # S_N contains the parts where the contact velocities and the gaps are <= 0
                A_N = (la_Nk1- r_N * xi_N >= 0) * I_N
                self.A_N_ = A_N_ = np.where( ~A_N )[0]
                self.A_N = A_N = np.where(A_N)[0]

                B_N = []
                gamma_Tk1 = self.model.gamma_T(tk1, qk1, uk1)
                for i_N, i_T in enumerate(self.NT_connectivity):
                    if np.any(i_T):
                        P_N = la_Nk1[i_N]
                        P_T = la_Tk1[i_T]
                        xi_T = gamma_Tk1[i_T] + e_T * gamma_Tk[i_T]
                        B_N.append(norm2(P_T - r_T * xi_T) <= mu * P_N)
                    else:
                        B_N.append(False)

                self.B_N = np.array(B_N) 
                
                # R(uk1, la_gk1, la_gammak1, la_Nk1)
                R = self.__R_newton(tk1, uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1)
                error = self.newton_error_function(R)
                converged = error < self.newton_tol and np.all(self.B_N == B_N_old)
                B_N_old = self.B_N.copy()
                if converged:
                    break
        
        return (converged, j, error), tk1, qk1, uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1

    def __R_newton(self, tk1, uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1):
        dt = self.dt
        uk = self.uk
        qk1 = self.qk1
        I_T = self.I_T
        A_N = self.A_N
        A_N_ = self.A_N_


        M = self.model.M(tk1, qk1)
        h = self.model.h(tk1, qk1, uk)
        W_g = self.model.W_g(tk1, qk1)
        W_gamma = self.model.W_gamma(tk1, qk1)
        W_N = self.model.W_N(tk1, qk1, scipy_matrix=csc_matrix)
        W_T = self.model.W_T(tk1, qk1, scipy_matrix=csc_matrix)
        g_N_dotk = self.model.g_N_dot(tk1, qk1, uk)
        xi_N = self.model.g_N_dot(tk1, qk1, uk1) + self.e_N * g_N_dotk

        gamma_Tk = self.model.gamma_T(tk1, qk1, uk)
        gamma_Tk1 = self.model.gamma_T(tk1, qk1, uk1)

        R = np.zeros(self.nR)
        R[:self.nu] = M @ (uk1 - uk) - dt * (h + W_g @ la_gk1 + W_gamma @ la_gammak1) - W_N[:, A_N] @ la_Nk1[A_N] - W_T[:, I_T] @ la_Tk1[I_T]
        R[self.nu:self.nu+self.nla_g] = self.model.g_dot(tk1, qk1, uk1)
        R[self.nu+self.nla_g:self.nu+self.nla_g+self.nla_gamma] = self.model.gamma(tk1, qk1, uk1)
        R[self.nR_smooth:self.nR_smooth+len(A_N)] = xi_N[A_N]
        R[self.nR_smooth+len(A_N):self.nR_smooth+self.nla_N] = la_Nk1[A_N_]

        offset = 0
        for i_N, i_T in enumerate(self.NT_connectivity):
            if np.any(i_T):
                if self.I_N[i_N]:
                    if self.B_N[i_N]:
                        R[self.nR_smooth+self.nla_N+offset:self.nR_smooth+self.nla_N+offset+2] = gamma_Tk1[i_T] + self.e_T * gamma_Tk[i_T]
                    else:
                        P_N = la_Nk1[i_N]
                        P_T = la_Tk1[i_T]
                        xi_T = gamma_Tk1[i_T] + self.e_T * gamma_Tk[i_T]
                        radius = self.mu * P_N
                        arg = P_T - self.r_T * xi_T
                        narg = norm2(arg) 
                        R[self.nR_smooth+self.nla_N+offset:self.nR_smooth+self.nla_N+offset+2] = P_T - radius * arg / narg
                else:
                    R[self.nR_smooth+self.nla_N+offset:self.nR_smooth+self.nla_N+offset+2] = la_Tk1[i_T]
                offset += 2

        return R

    def __R_newton_wrapper(self, t, x):
        uk1 = x[:self.nu]
        la_gk1 = x[self.nu:self.nu+self.nla_g]
        la_gammak1 = x[self.nu+self.nla_g:self.nu+self.nla_g+self.nla_gamma]
        la_Nk1 = np.empty(self.nla_N)
        la_Nk1[self.A_N] = x[self.nR_smooth:self.nR_smooth+len(self.A_N)]
        la_Nk1[self.A_N_] = x[self.nR_smooth+len(self.A_N):self.nR_smooth+self.nla_N]

        offset = 0
        la_Tk1 = np.empty(self.nla_T)
        for i_N, i_T in enumerate(self.NT_connectivity):
            if np.any(i_T):
                if self.I_N[i_N]:
                    if self.B_N[i_N]:
                        la_Tk1[i_T] = x[self.nR_smooth+self.nla_N+offset:self.nR_smooth+self.nla_N+offset+2]
                    else:
                        la_Tk1[i_T] = x[self.nR_smooth+self.nla_N+offset:self.nR_smooth+self.nla_N+offset+2]
                else:
                    la_Tk1[i_T] = x[self.nR_smooth+self.nla_N+offset:self.nR_smooth+self.nla_N+offset+2]
                offset += 2

        return self.__R_newton(t, uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1)


    def solve(self):
        # initial values
        tk = self.model.t0
        qk = self.model.q0.copy()
        uk = self.model.u0.copy()
        la_gk = self.model.la_g0.copy()
        la_gammak = self.model.la_gamma0.copy()
        la_Nk = self.model.la_N0.copy()
        la_Tk = self.model.la_T0.copy()
        
        # lists storing output variables
        q = [qk]
        u = [uk]
        la_g = [la_gk]
        la_gamma = [la_gammak]
        la_N = [la_Nk]
        la_T = [la_Tk]

        pbar = tqdm(self.t[:-1])
        for tk in pbar:
            # (converged, j, error), tk1, qk1, uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1 = self.step(tk, qk, uk, la_Nk, la_Tk)
            (converged, j, error), tk1, qk1, uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1 = self.step(tk, qk, uk, la_gk, la_gammak, la_Nk, la_Tk)
            pbar.set_description(f't: {tk1:0.2e}; fixed point iterations: {j+1}; error: {error:.3e}')
            if not converged:
                raise RuntimeError(f'internal fixed point iteration not converged after {j+1} iterations with error: {error:.5e}')

            qk1, uk1 = self.model.solver_step_callback(tk1, qk1, uk1)

            q.append(qk1)
            u.append(uk1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)
            la_N.append(la_Nk1)
            la_T.append(la_Tk1)

            # update local variables for accepted time step
            qk, uk, la_gk, la_gammak, la_Nk, la_Tk = qk1, uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1
            
        # write solution
        return Solution(t=self.t, q=np.array(q), u=np.array(u), la_g=np.array(la_g), la_gamma=np.array(la_gamma), la_N=np.array(la_N), la_T=np.array(la_T))

class Moreau_sym():
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
        self.n = self.nu + self.nla_g + self.nla_gamma

        self.uDOF = np.arange(self.nu)
        self.la_gDOF = self.nu + np.arange(self.nla_g)
        self.la_gammaDOF = self.nu + self.nla_g + np.arange(self.nla_gamma)

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

    def __R(self, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
        self.Mk1 = self.model.M(tk1, qk1)
        self.W_gk1 = self.model.W_g(tk1, qk1)
        self.W_gammak1 = self.model.W_gamma(tk1, qk1)

        R = np.zeros(self.n)
        R[self.uDOF] = self.Mk1 @ (uk1 - uk) - 0.5 * self.dt * (self.model.h(tk1, qk1, uk) + self.model.h(tk1, qk1, uk1) ) - self.dt * ( self.W_gk1 @ la_gk1 + self.W_gammak1 @ la_gammak1 )
        R[self.la_gDOF] = self.model.g_dot(tk1, qk1, uk1)
        R[self.la_gammaDOF] = self.model.gamma(tk1, qk1, uk1)

        return R

    def __R_wrapper(self, t, x, y):
        
        uk1 = x[self.uDOF]
        la_gk1 = x[self.la_gDOF]
        la_gammak1 = x[self.la_gammaDOF]

        qk1 = y[:self.nq]
        uk = y[self.nq:]

        return self.__R(uk, t, qk1, uk1, la_gk1, la_gammak1)

    def __R_x_num(self, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
        y = np.zeros(self.nq + self.nu)
        y[:self.nq] = qk1
        y[self.nq:] = uk

        x = np.zeros(self.n)
        x[self.uDOF] = uk1
        x[self.la_gDOF] = la_gk1
        x[self.la_gammaDOF] = la_gammak1

        R_x_num = Numerical_derivative(self.__R_wrapper, order=2)._x(tk1, x, y)

        return csr_matrix( R_x_num )

    def __R_x_analytic(self, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
        # equations of motion
        Ru_u = self.Mk1 - 0.5* self.dt * self.model.h_u(tk1, qk1, uk1)
        Ru_la_g = - self.dt * self.W_gk1
        Ru_la_gamma = - self.dt * self.W_gammak1

        # constraint equations
        Rla_g_u = self.model.g_dot_u(tk1, qk1)
        Rla_gamma_u = self.model.gamma_u(tk1, qk1)
        
        return bmat([[Ru_u       , Ru_la_g, Ru_la_gamma], \
                     [Rla_g_u    ,    None,        None], \
                     [Rla_gamma_u,    None,        None]]).tocsc()

    def __R_x_debug(self, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
        R_x_num = self.__R_x_num(uk, tk1, qk1, uk1, la_gk1, la_gammak1)
        R_x_analytic = self.__R_x_analytic(uk, tk1, qk1, uk1, la_gk1, la_gammak1)
        diff = R_x_num - R_x_analytic.toarray()

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
        qk1 = qk + dt * self.model.q_dot(tk, qk, uk)

        # initial guess for Newton-Raphson solver
        xk1 = np.zeros(self.n)
        xk1[self.uDOF] = uk1
        xk1[self.la_gDOF] = la_gk1
        xk1[self.la_gammaDOF] = la_gammak1

        # initial residual and error
        R = self.__R(uk, tk1, qk1, uk1, la_gk1, la_gammak1)
        error = self.newton_error_function(R)
        converged = error < self.newton_tol
        j = 0
        if not converged:
            while j < self.newton_max_iter:
                # jacobian
                R_x = self.__R_x(uk, tk1, qk1, uk1, la_gk1, la_gammak1)

                # Newton update
                j += 1
                dx = spsolve(R_x, R)
                xk1 -= dx
                uk1 = xk1[self.uDOF]
                la_gk1 = xk1[self.la_gDOF]
                la_gammak1 = xk1[self.la_gammaDOF]

                R = self.__R(uk, tk1, qk1, uk1, la_gk1, la_gammak1)
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