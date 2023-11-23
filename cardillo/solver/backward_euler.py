import numpy as np
from scipy.sparse import csc_array, eye, bmat
from scipy.sparse.linalg import splu
from tqdm import tqdm

from cardillo.solver import SolverOptions, SolverSummary, Solution
from cardillo.math import (
    fsolve,
    prox_R0_nm,
    prox_sphere,
    estimate_prox_parameter,
    approx_fprime,
)
from cardillo.math import fsolve, approx_fprime, prox_R0_nm, prox_sphere


NEWTON_MAXITER = 4  # maximum number of Newton iterations


class BackwardEuler:
    def __init__(
        self,
        system,
        t1,
        dt,
        options=SolverOptions(),
    ):
        self.system = system
        self.options = options

        if options.numerical_jacobian_method:
            self.J_x = lambda x, y: csc_array(
                approx_fprime(
                    x,
                    lambda x: self.R_x(x, y),
                    method=options.numerical_jacobian_method,
                    eps=options.numerical_jacobian_eps,
                )
            )
        else:
            self.J_x = self._J_x

        #######################################################################
        # integration time
        #######################################################################
        self.t0 = t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

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
        self.xn = self.dt * np.concatenate(
            (
                self.q_dotn,
                self.u_dotn,
                self.la_gn,
                self.la_gamman,
                self.la_cn,
                self.mu_Sn,
            )
        )
        self.yn = self.dt * np.concatenate(
            (
                self.la_Nn,
                self.la_Fn,
            )
        )

        # initial mass matrix and force directions for prox-parameter estimation
        self.M = system.M(self.tn, self.qn)
        self.W_N = system.W_N(self.tn, self.qn)
        self.W_F = system.W_F(self.tn, self.qn)

    def R_x(self, xn1, yn1):
        (
            dqn1,
            dun1,
            dP_gn1,
            dP_gamman1,
            dP_cn1,
            dmu_Sn1,
        ) = np.array_split(xn1, self.split_x)
        (
            dP_Nn1,
            dP_Fn1,
        ) = np.array_split(yn1, self.split_y)

        tn, dt, qn, un = self.tn, self.dt, self.qn, self.un
        tn1 = tn + dt
        qn1 = qn + dqn1
        un1 = un + dun1

        ###################
        # evaluate residual
        ###################
        R_x = np.zeros_like(xn1)

        ####################
        # kinematic equation
        ####################
        g_S_q = self.system.g_S_q(tn1, qn1, format="csc")
        R_x[: self.split_x[0]] = (
            dqn1 - dt * self.system.q_dot(tn1, qn1, un1) - g_S_q.T @ dmu_Sn1
        )

        ####################
        # equations of motion
        ####################
        self.M = self.system.M(tn1, qn1, format="csr")
        self.W_N = self.system.W_N(tn1, qn1, format="csr")
        self.W_F = self.system.W_F(tn1, qn1, format="csr")
        R_x[self.split_x[0] : self.split_x[1]] = (
            self.M @ dun1
            - dt * self.system.h(tn1, qn1, un1)
            - self.system.W_g(tn1, qn1, format="csr") @ dP_gn1
            - self.system.W_gamma(tn1, qn1, format="csr") @ dP_gamman1
            - self.system.W_c(tn1, qn1, format="csr") @ dP_cn1
            - self.W_N @ dP_Nn1
            - self.W_F @ dP_Fn1
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
            tn1, qn1, un1, dP_cn1 / self.dt
        )

        ##########################
        # quaternion stabilization
        ##########################
        R_x[self.split_x[4] :] = self.system.g_S(tn1, qn1)

        return R_x

    def _J_x(self, xn1, yn1):
        (
            dqn1,
            dun1,
            dP_gn1,
            dP_gamman1,
            dP_cn1,
            dmu_Sn1,
        ) = np.array_split(xn1, self.split_x)
        (
            dP_Nn1,
            dP_Fn1,
        ) = np.array_split(yn1, self.split_y)

        tn, dt, qn, un = self.tn, self.dt, self.qn, self.un
        tn1 = tn + dt
        qn1 = qn + dqn1
        un1 = un + dun1

        ####################
        # kinematic equation
        ####################
        Rq_dot_q_dot = eye(self.nq) - (
            dt * self.system.q_dot_q(tn1, qn1, un1)
            + self.system.g_S_q_T_mu_q(tn1, qn1, dmu_Sn1)
        )
        Rq_dot_u_dot = -dt * self.system.q_dot_u(tn1, qn1)
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
            self.system.Mu_q(tn1, qn1, dun1)
            - dt * self.system.h_q(tn1, qn1, un1)
            - self.system.Wla_g_q(tn1, qn1, dP_gn1)
            - self.system.Wla_gamma_q(tn1, qn1, dP_gamman1)
            - self.system.Wla_c_q(tn1, qn1, dP_cn1)
            - self.system.Wla_N_q(tn1, qn1, dP_Nn1)
            - self.system.Wla_F_q(tn1, qn1, dP_Fn1)
        )
        Ru_dot_u_dot = M - dt * self.system.h_u(tn1, qn1, un1)

        #######################
        # bilateral constraints
        #######################
        Rla_g_q_dot = self.system.g_q(tn1, qn1)
        Rla_gamma_q_dot = self.system.gamma_q(tn1, qn1, un1)
        Rla_gamma_u_dot = self.system.gamma_u(tn1, qn1)

        ############
        # compliance
        ############
        Rla_c_q_dot = self.system.c_q(tn1, qn1, un1, dP_cn1 / self.dt)
        Rla_c_u_dot = self.system.c_u(tn1, qn1, un1, dP_cn1 / self.dt)
        Rla_c_la_c = self.system.c_la_c() / self.dt

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
            dqn1,
            dun1,
            _,
            _,
            _,
            _,
        ) = np.array_split(x1, self.split_x)
        (
            dP_Nn1,
            dP_Fn1,
        ) = np.array_split(y0, self.split_y)

        tn, dt, qn, un = self.tn, self.dt, self.qn, self.un
        tn1 = tn + dt
        qn1 = qn + dqn1
        un1 = un + dun1

        # mu = self.system.mu
        prox_r_N = self.prox_r_N
        prox_r_F = self.prox_r_F

        y1 = np.zeros_like(y0)

        ##############################
        # fixed-point update Signorini
        ##############################
        g_N = self.system.g_N(tn1, qn1)
        prox_arg = (prox_r_N / self.dt) * g_N - dP_Nn1
        y1[: self.split_y[0]] = -prox_R0_nm(prox_arg)

        #############################
        # fixed-point update friction
        #############################
        # TODO: Get gamma_f_contr without "private" atrribute
        for contr in self.system._System__gamma_F_contr:
            gamma_F_contr = contr.gamma_F(tn1, qn1[contr.qDOF], un1[contr.uDOF])
            la_FDOF = contr.la_FDOF
            dP_Fn1_contr = dP_Fn1[la_FDOF]
            prox_r_F_contr = prox_r_F[la_FDOF]
            for i_N, i_F, force_recervoir in contr.NF_connectivity2:
                if len(i_N) > 0:
                    dP_Nn1i = dP_Nn1[contr.la_NDOF[i_N]]
                else:
                    dP_Nn1i = self.dt

                y1[self.split_y[0] + la_FDOF[i_F]] = -force_recervoir.prox(
                    prox_r_F_contr[i_F] * gamma_F_contr[i_F] - dP_Fn1_contr[i_F],
                    dP_Nn1i,
                )

        return y1

    def solve(self):
        solver_summary = SolverSummary()

        # lists storing output variables
        t = [self.tn]
        q = [self.qn]
        u = [self.un]
        q_dot = [self.q_dotn]
        u_dot = [self.u_dotn]
        la_c = [self.la_cn]
        P_g = [self.dt * self.la_gn]
        P_gamma = [self.dt * self.la_gamman]
        P_N = [self.dt * self.la_Nn]
        P_F = [self.dt * self.la_Fn]
        mu_S = [self.mu_Sn]

        # initial Jacobian
        if self.options.reuse_lu_decomposition:
            i_newton = 0
            lu = splu(self.J_x(self.xn.copy(), self.yn.copy()))
            solver_summary.add_lu(1)

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # only compute optimized prox-parameters once per time step
            self.prox_r_N = estimate_prox_parameter(
                self.options.prox_scaling, self.W_N, self.M
            )
            self.prox_r_F = estimate_prox_parameter(
                self.options.prox_scaling, self.W_F, self.M
            )

            # perform a solver step
            tn1 = self.tn + self.dt

            ########################
            # fixed-point iterations
            ########################
            # store old values
            x0 = self.xn.copy()
            y0 = self.yn.copy()

            # fixed-point loop
            xn1 = x0.copy()
            yn1 = y0.copy()
            converged = False
            n_state = self.nx - self.nla_g - self.nla_gamma
            for i_fixed_point in range(self.options.fixed_point_max_iter):
                # find proximal point
                yn1 = self.prox(xn1, yn1)

                if self.options.reuse_lu_decomposition:
                    # compute new residual and check convergence
                    R_newton = self.R_x(xn1, yn1)
                    error_newton = np.max(np.absolute(R_newton))
                    converged_newton = error_newton < self.options.newton_atol

                    # Newton loop with inexact Jacobian
                    if not converged_newton:
                        i_newton = 0
                    while (not converged_newton) and (
                        i_newton < self.options.newton_max_iter
                    ):
                        i_newton += 1
                        # compute new Jacobian if requested
                        if i_newton >= NEWTON_MAXITER:
                            lu = splu(self.J_x(xn1, yn1))
                            solver_summary.add_lu(1)
                            i_newton = 0

                        xn1 -= lu.solve(R_newton)
                        R_newton = self.R_x(xn1, yn1)
                        error_newton = np.max(np.absolute(R_newton))
                        converged_newton = error_newton < self.options.newton_atol

                else:
                    xn1, converged_newton, error_newton, i_newton, _ = fsolve(
                        self.R_x,
                        self.xn,
                        jac=self.J_x,
                        fun_args=(yn1,),
                        jac_args=(yn1,),
                        atol=self.options.newton_atol,
                        max_iter=self.options.newton_max_iter,
                    )
                    solver_summary.add_lu(i_newton)

                # convergence in smooth state (without Lagrange multipliers)
                diff = xn1[:n_state] - x0[:n_state]

                # error measure, see Hairer1993, Section II.4
                sc = (
                    self.options.fixed_point_atol
                    + np.maximum(np.abs(x0[:n_state]), np.abs(xn1[:n_state]))
                    * self.options.fixed_point_rtol
                )
                error = np.linalg.norm(diff / sc) / sc.size**0.5
                converged = error < 1.0 and converged_newton

                if converged:
                    break
                else:
                    # update values
                    x0 = xn1.copy()
                    y0 = yn1.copy()

            fixed_point_absolute_error = np.max(np.abs(diff))
            solver_summary.add_fixed_point(i_fixed_point, fixed_point_absolute_error)
            solver_summary.add_newton(i_newton)

            # update progress bar
            pbar.set_description(
                f"t: {tn1:0.2e}s < {self.t1:0.2e}s; |x1 - x0|: {fixed_point_absolute_error:0.2e} (fixed-point: {i_fixed_point}/{self.options.fixed_point_max_iter}; newton: {i_newton}/{self.options.newton_max_iter})"
            )

            # compute state
            (
                dqn1,
                dun1,
                dP_gn1,
                dP_gamman1,
                dP_cn1,
                dmu_Sn1,
            ) = np.array_split(xn1, self.split_x)
            (
                dP_Nn1,
                dP_Fn1,
            ) = np.array_split(yn1, self.split_y)
            qn1 = self.qn + dqn1
            un1 = self.un + dun1

            # modify converged quantities
            qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            # store solution fields
            t.append(tn1)
            q.append(qn1)
            u.append(un1)
            q_dot.append(dqn1 / self.dt)
            u_dot.append(dun1 / self.dt)
            la_c.append(dP_cn1 / self.dt)
            P_g.append(dP_gn1)
            P_gamma.append(dP_gamman1)
            P_N.append(dP_Nn1)
            P_F.append(dP_Fn1)
            mu_S.append(dmu_Sn1 / self.dt)

            # update local variables for accepted time step
            self.xn = xn1.copy()
            self.yn = yn1.copy()
            self.tn = tn1
            self.qn = qn1
            self.un = un1

        solver_summary.print()

        # write solution
        return Solution(
            system=self.system,
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            q_dot=np.array(q_dot),
            u_dot=np.array(u_dot),
            la_c=np.array(la_c),
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
            mu_S=np.array(mu_S),
        )
