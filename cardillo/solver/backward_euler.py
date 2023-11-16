import numpy as np
from scipy.sparse import csc_array, csr_array, eye, diags, bmat
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from tqdm import tqdm

from cardillo.math import fsolve, approx_fprime, prox_R0_nm, prox_sphere
from cardillo.solver import Solution
from cardillo.utility.coo_matrix import CooMatrix


NEWTON_MAXITER = 4  # Maximum number of Newton iterations.


class BackwardEuler:
    def __init__(
        self,
        system,
        t1,
        dt,
        atol=1e-8,
        max_iter=10,
        max_iter_fixed_point=int(1e3),
        reuse_lu_decomposition=True,
    ):
        self.system = system

        #######################################################################
        # integration time
        #######################################################################
        self.t0 = t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        #######################################################################
        # newton settings
        #######################################################################
        self.atol = atol
        self.max_iter = max_iter
        self.max_iter_fixed_point = max_iter_fixed_point

        #######################################################################
        # dimensions
        #######################################################################
        self.nq = system.nq
        self.nu = system.nu
        self.nla_g = system.nla_g
        self.nla_gamma = system.nla_gamma
        self.nla_c = system.nla_c
        self.nla_N = system.nla_N
        self.nla_F = system.nla_F
        self.nla_S = self.system.nla_S

        self.nx = (
            self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_c + self.nla_S
        )
        self.ny = self.nla_N + self.nla_F
        self.split_x = np.cumsum(
            np.array(
                [
                    self.nq,
                    self.nu,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_c,
                    self.nla_S,
                ],
                dtype=int,
            )
        )[:-1]
        self.split_y = np.cumsum(
            np.array(
                [
                    self.nla_N,
                    self.nla_F,
                ],
                dtype=int,
            )
        )[:-1]

        #######################################################################
        # initial conditions
        #######################################################################
        self.tn = system.t0
        self.qn = system.q0
        self.un = system.u0
        self.q_dotn = system.q_dot0
        self.u_dotn = system.u_dot0
        self.la_gn = system.la_g0
        self.la_gamman = system.la_gamma0
        self.la_cn = system.la_c0
        self.la_Nn = system.la_N0
        self.la_Fn = system.la_F0
        self.mu_Sn = np.zeros(self.nla_S)

        #######################################################################
        # initial values
        #######################################################################
        self.zn = self.dt * np.concatenate(
            (
                self.q_dotn,
                self.u_dotn,
                self.la_gn,
                self.la_gamman,
                self.la_cn,
                self.mu_Sn,
                self.la_Nn,
                self.la_Fn,
            )
        )
        self.xn = self.zn[: self.nx].copy()
        self.yn = self.zn[self.nx :].copy()

    def R_x(self, xn1, yn1):
        (
            q_dotn1,
            u_dotn1,
            la_gn1,
            la_gamman1,
            la_cn1,
            mu_Sn1,
        ) = np.array_split(xn1, self.split_x)
        (
            la_Nn1,
            la_Fn1,
        ) = np.array_split(yn1, self.split_y)

        tn, dt, qn, un = self.tn, self.dt, self.qn, self.un
        tn1 = tn + dt
        qn1 = qn + q_dotn1
        un1 = un + u_dotn1

        ###################
        # evaluate residual
        ###################
        R_x = np.zeros_like(xn1)

        ####################
        # kinematic equation
        ####################
        g_S_q = self.system.g_S_q(tn1, qn1, scipy_matrix=csc_array)
        R_x[: self.split_x[0]] = (
            q_dotn1 - dt * self.system.q_dot(tn1, qn1, un1) - g_S_q.T @ mu_Sn1
        )

        ####################
        # equations of motion
        ####################
        R_x[self.split_x[0] : self.split_x[1]] = (
            self.system.M(tn1, qn1, scipy_matrix=csr_array) @ u_dotn1
            - dt * self.system.h(tn1, qn1, un1)
            - self.system.W_g(tn1, qn1, scipy_matrix=csr_array) @ la_gn1
            - self.system.W_gamma(tn1, qn1, scipy_matrix=csr_array) @ la_gamman1
            - self.system.W_c(tn1, qn1, scipy_matrix=csr_array) @ la_cn1
            - self.system.W_N(tn1, qn1, scipy_matrix=csr_array) @ la_Nn1
            - self.system.W_F(tn1, qn1, scipy_matrix=csr_array) @ la_Fn1
        )

        #######################
        # bilateral constraints
        #######################
        R_x[self.split_x[1] : self.split_x[2]] = self.system.g(tn1, qn1)
        R_x[self.split_x[2] : self.split_x[3]] = self.system.gamma(tn1, qn1, un1)

        ############
        # compliance
        ############
        R_x[self.split_x[3] : self.split_x[4]] = self.system.c(
            tn1, qn1, un1, la_cn1 / self.dt
        )

        ##########################
        # quaternion stabilization
        ##########################
        R_x[self.split_x[4] :] = self.system.g_S(tn1, qn1)

        return R_x

    def J_x(self, xn1, yn1):
        # return csc_array(approx_fprime(xn1, lambda x: self.R_x(x, yn1), method="2-point"))

        (
            q_dotn1,
            u_dotn1,
            la_gn1,
            la_gamman1,
            la_cn1,
            mu_Sn1,
        ) = np.array_split(xn1, self.split_x)
        (
            la_Nn1,
            la_Fn1,
        ) = np.array_split(yn1, self.split_y)

        tn, dt, qn, un = self.tn, self.dt, self.qn, self.un
        tn1 = tn + dt
        qn1 = qn + q_dotn1
        un1 = un + u_dotn1

        ####################
        # kinematic equation
        ####################
        Rq_dot_q_dot = eye(self.nq) - (
            dt * self.system.q_dot_q(tn1, qn1, un1)
            + self.system.g_S_q_T_mu_q(tn1, qn1, mu_Sn1)
        )
        Rq_dot_u_dot = -dt * self.system.q_dot_u(tn1, qn1, un1)
        g_S_q = self.system.g_S_q(tn1, qn1)
        g_S_q_dot = g_S_q

        ########################
        # equations of motion (1)
        ########################
        M = self.system.M(tn1, qn1)
        W_g = self.system.W_g(tn1, qn1)
        W_gamma = self.system.W_gamma(tn1, qn1)
        W_c = self.system.W_c(tn1, qn1)

        Ru_dot_q_dot = (
            self.system.Mu_q(tn1, qn1, u_dotn1)
            - dt * self.system.h_q(tn1, qn1, un1)
            - self.system.Wla_g_q(tn1, qn1, la_gn1)
            - self.system.Wla_gamma_q(tn1, qn1, la_gamman1)
            - self.system.Wla_c_q(tn1, qn1, la_cn1)
            - self.system.Wla_N_q(tn1, qn1, la_Nn1)
            - self.system.Wla_F_q(tn1, qn1, la_Fn1)
        )
        Ru_dot_u_dot = M - dt * self.system.h_u(tn1, qn1, un1)

        #######################
        # bilateral constraints
        #######################
        Rla_g_q_dot = self.system.g_q(tn1, qn1)
        Rla_gamma_q_dot = self.system.gamma_q(tn1, qn1, un1)
        Rla_gamma_u_dot = self.system.gamma_u(tn1, qn1, un1)

        ############
        # compliance
        ############
        Rla_c_q_dot = self.system.c_q(tn1, qn1, un1, la_cn1 / self.dt)
        Rla_c_u_dot = self.system.c_u(tn1, qn1, un1, la_cn1 / self.dt)
        Rla_c_la_c = self.system.c_la_c(tn1, qn1, un1, la_cn1 / self.dt) / self.dt

        # fmt: off
        J = bmat(
            [
                [   Rq_dot_q_dot,    Rq_dot_u_dot, None,     None,       None, -g_S_q.T],
                [   Ru_dot_q_dot,    Ru_dot_u_dot, -W_g, -W_gamma,       -W_c,     None],
                [    Rla_g_q_dot,            None, None,     None,       None,     None],
                [Rla_gamma_q_dot, Rla_gamma_u_dot, None,     None,       None,     None],
                [    Rla_c_q_dot,     Rla_c_u_dot, None,     None, Rla_c_la_c,     None],
                [      g_S_q_dot,            None, None,     None,       None,     None],
            ],
            format="csc",
        )
        # fmt: on

        return J

    def prox(self, x1, y0):
        (
            q_dotn1,
            u_dotn1,
            _,
            _,
            _,
            _,
        ) = np.array_split(x1, self.split_x)
        (
            la_Nn1,
            la_Fn1,
        ) = np.array_split(y0, self.split_y)

        tn, dt, qn, un = self.tn, self.dt, self.qn, self.un
        tn1 = tn + dt
        qn1 = qn + q_dotn1
        un1 = un + u_dotn1

        mu = self.system.mu
        prox_r_N = self.prox_r_N
        prox_r_F = self.prox_r_F

        y1 = np.zeros_like(y0)

        ##############################
        # fixed-point update Signorini
        ##############################
        g_N = self.system.g_N(tn1, qn1)
        prox_arg = (prox_r_N / self.dt) * g_N - la_Nn1
        y1[: self.split_y[0]] = -prox_R0_nm(prox_arg)

        #############################
        # fixed-point update friction
        #############################
        gamma_F = self.system.gamma_F(tn1, qn1, un1)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            if len(i_F):
                y1[self.split_y[0] + np.array(i_F)] = -prox_sphere(
                    prox_r_F[i_N] * gamma_F[i_F] - la_Fn1[i_F],
                    mu[i_N] * la_Nn1[i_N],
                )

        return y1

    def solve(self):
        # lists storing output variables
        t = [self.tn]
        q = [self.qn]
        u = [self.un]
        q_dot = [self.dt * self.q_dotn]
        u_dot = [self.dt * self.u_dotn]
        P_c = [self.dt * self.la_cn]
        P_g = [self.dt * self.la_gn]
        P_gamma = [self.dt * self.la_gamman]
        P_N = [self.dt * self.la_Nn]
        P_F = [self.dt * self.la_Fn]
        mu_S = [self.dt * self.mu_Sn]

        n_iter_list = [0]
        errors = [0.0]

        # initial Jacobian
        i_newton = 0
        lu = splu(self.J_x(self.xn.copy(), self.yn.copy()))
        n_lu = 1

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # only compute optimized proxparameters once per time step
            self.prox_r_N = self.system.prox_r_N(self.tn, self.qn)
            self.prox_r_F = self.system.prox_r_F(self.tn, self.qn)

            # perform a solver step
            tn1 = self.tn + self.dt

            # ###############
            # # smooth Newton
            # ###############
            # xn1, converged, error, n_iter, _ = fsolve(
            #     self.R_x,
            #     self.xn,
            #     jac=self.J_x,
            #     fun_args=(self.yn.copy(),),
            #     jac_args=(self.yn.copy(),),
            #     atol=self.atol,
            #     max_iter=self.max_iter,
            # )
            # yn1 = self.yn.copy()

            ########################
            # fixed-point iterations
            ########################
            # store old values
            x0 = self.xn.copy()
            y0 = self.yn.copy()

            # fixed-point loop
            xn1 = x0.copy()
            yn1 = y0.copy()
            # lu = splu(self.J_x(xn1, yn1))
            # n_lu += 1
            converged = False
            n_state = self.nx - self.nla_g - self.nla_gamma
            for n_iter in range(self.max_iter_fixed_point):
                # find proximal point
                yn1 = self.prox(xn1, yn1)

                # compute new residual and check convergence
                R_newton = self.R_x(xn1, yn1)
                error_newton = np.max(np.absolute(R_newton))
                converged_newton = error_newton < self.atol

                # Newton loop with inexact Jacobian
                i_newton = 0
                while (not converged_newton) and (i_newton < self.max_iter):
                    i_newton += 1
                    # compute new Jacobian if requested
                    if i_newton >= NEWTON_MAXITER:
                        lu = splu(self.J_x(xn1, yn1))
                        n_lu += 1
                        i_newton = 0

                    xn1 -= lu.solve(R_newton)
                    R_newton = self.R_x(xn1, yn1)
                    error_newton = np.max(np.absolute(R_newton))
                    converged_newton = error_newton < self.atol

                # # convergence percussions (note: only for reference, but
                # # useless for redundant contacts)
                # diff = yn1 - y0

                # convergence in smooth state (without Lagrange multipliers)
                diff = xn1[:n_state] - x0[:n_state]

                error = np.max(np.absolute(diff))

                converged = error < self.atol and converged_newton
                if converged:
                    break
                else:
                    # update values
                    x0 = xn1.copy()
                    y0 = yn1.copy()

            zn1 = np.concatenate((xn1, yn1))

            n_iter_list.append(n_iter)
            errors.append(error)

            # update progress bar
            pbar.set_description(
                f"t: {tn1:0.2e}s < {self.t1:0.2e}s; ||R||: {error:0.2e} ({n_iter}/{self.max_iter_fixed_point})"
            )

            # compute state
            (
                q_dotn1,
                u_dotn1,
                la_gn1,
                la_gamman1,
                la_cn1,
                mu_Sn1,
            ) = np.array_split(xn1, self.split_x)
            (
                la_Nn1,
                la_Fn1,
            ) = np.array_split(yn1, self.split_y)
            qn1 = self.qn + q_dotn1
            un1 = self.un + u_dotn1

            # modify converged quantities
            qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            # store solution fields
            t.append(tn1)
            q.append(qn1)
            u.append(un1)
            q_dot.append(q_dotn1 / self.dt)
            u_dot.append(u_dotn1 / self.dt)
            P_g.append(la_gn1)
            P_gamma.append(la_gamman1)
            P_c.append(la_cn1)
            P_N.append(la_Nn1)
            P_F.append(la_Fn1)
            mu_S.append(mu_Sn1)

            # update local variables for accepted time step
            self.xn = xn1.copy()
            self.yn = yn1.copy()
            self.zn = zn1.copy()
            self.tn = tn1
            self.qn = qn1
            self.un = un1

        print("-" * 80)
        print("Solver summary:")
        print(
            f" - iterations: max = {max(n_iter_list)}, avg={sum(n_iter_list) / float(len(n_iter_list))}"
        )
        print(f" - errors: max = {max(errors)}, avg={sum(errors) / float(len(errors))}")
        print(f" - lu-decompositions: {n_lu}")
        print("-" * 80)

        # write solution
        return Solution(
            system=self.system,
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            q_dot=np.array(q_dot),
            u_dot=np.array(u_dot),
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
            mu_S=np.array(mu_S),
            niter=np.array(n_iter_list),
        )

