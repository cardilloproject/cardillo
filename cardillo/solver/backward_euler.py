import warnings

import numpy as np
from scipy.sparse import bmat, coo_array, csc_array, eye
from scipy.sparse.linalg import splu
from tqdm import tqdm

from cardillo.math.approx_fprime import approx_fprime
from cardillo.math.fsolve import fsolve
from cardillo.math.prox import NegativeOrthant, estimate_prox_parameter
from cardillo.solver import Solution, SolverOptions, SolverSummary


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

        self.nx = self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_c
        self.ny = self.nla_N + self.nla_F
        self.split_x = np.cumsum(
            np.array(
                [
                    self.nq,
                    self.nu,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_c,
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
        R_x[: self.split_x[0]] = dqn1 - dt * self.system.q_dot(tn1, qn1, un1)

        ####################
        # equations of motion
        ####################
        self.M = self.system.M(tn1, qn1, format="csr")
        self.W_N = self.system.W_N(tn1, qn1, format="csr")
        self.W_F = self.system.W_F(tn1, qn1, format="csr")
        R_x[self.split_x[0] : self.split_x[1]] = (
            self.M @ dun1
            - dt
            * (
                self.system.h(tn1, qn1, un1)
                + self.system.W_tau(tn1, qn1, format="csr")
                @ self.system.la_tau(tn1, qn1, un1)
            )
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
        R_x[self.split_x[3] :] = self.system.c(tn1, qn1, un1, dP_cn1 / self.dt)

        return R_x

    def _J_x(self, xn1, yn1):
        (
            dqn1,
            dun1,
            dP_gn1,
            dP_gamman1,
            dP_cn1,
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
        Rq_dot_q_dot = eye(self.nq) - dt * self.system.q_dot_q(tn1, qn1, un1)
        Rq_dot_u_dot = -dt * self.system.q_dot_u(tn1, qn1)

        ########################
        # equations of motion (1)
        ########################
        M = self.system.M(tn1, qn1)
        W_g = self.system.W_g(tn1, qn1)
        W_gamma = self.system.W_gamma(tn1, qn1)
        W_c = self.system.W_c(tn1, qn1)

        Ru_dot_q_dot = (
            self.system.Mu_q(tn1, qn1, dun1)
            - dt
            * (self.system.h_q(tn1, qn1, un1) + self.system.Wla_tau_q(tn1, qn1, un1))
            - self.system.Wla_g_q(tn1, qn1, dP_gn1)
            - self.system.Wla_gamma_q(tn1, qn1, dP_gamman1)
            - self.system.Wla_c_q(tn1, qn1, dP_cn1)
            - self.system.Wla_N_q(tn1, qn1, dP_Nn1)
            - self.system.Wla_F_q(tn1, qn1, dP_Fn1)
        )
        Ru_dot_u_dot = M - dt * (
            self.system.h_u(tn1, qn1, un1) + self.system.Wla_tau_u(tn1, qn1, un1)
        )

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
                [   Rq_dot_q_dot,    Rq_dot_u_dot, None,     None,       None],
                [   Ru_dot_q_dot,    Ru_dot_u_dot, -W_g, -W_gamma,       -W_c],
                [    Rla_g_q_dot,            None, None,     None,       None],
                [Rla_gamma_q_dot, Rla_gamma_u_dot, None,     None,       None],
                [    Rla_c_q_dot,     Rla_c_u_dot, None,     None, Rla_c_la_c],
            ],
            format="csc",
        )
        # fmt: on
        return J

    def J_y(self, xn1, yn1):
        # fmt: off
        return bmat([
            [       coo_array((self.nq, self.nla_N)),      None],
            [                              -self.W_N, -self.W_F],
            [    coo_array((self.nla_g, self.nla_N)),      None],
            [coo_array((self.nla_gamma, self.nla_N)),      None],
            [    coo_array((self.nla_c, self.nla_N)),      None],
        ])
        # fmt: on

    def prox(self, x1, y0):
        (
            dqn1,
            dun1,
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

        prox_r_N = self.prox_r_N
        prox_r_F = self.prox_r_F

        y1 = np.zeros_like(y0)

        ##############################
        # fixed-point update Signorini
        ##############################
        g_N = self.system.g_N(tn1, qn1)
        dP_Nn1 = -NegativeOrthant.prox((prox_r_N / self.dt) * g_N - dP_Nn1)
        y1[: self.split_y[0]] = dP_Nn1

        #############################
        # fixed-point update friction
        #############################
        for contr in self.system.get_contribution_list("gamma_F"):
            gamma_F_contr = contr.gamma_F(tn1, qn1[contr.qDOF], un1[contr.uDOF])
            la_FDOF = contr.la_FDOF
            dP_Fn1_contr = dP_Fn1[la_FDOF]
            prox_r_F_contr = prox_r_F[la_FDOF]
            for i_N, i_F, force_recervoir in contr.friction_laws:
                if len(i_N) > 0:
                    dP_Nn1i = dP_Nn1[contr.la_NDOF[i_N]]
                else:
                    dP_Nn1i = self.dt

                y1[self.split_y[0] + la_FDOF[i_F]] = -force_recervoir.prox(
                    prox_r_F_contr[i_F] * gamma_F_contr[i_F] - dP_Fn1_contr[i_F],
                    dP_Nn1i,
                )

        return y1

    def _solve_nonlinear_system(self, x0, y, lu):
        if self.options.reuse_lu_decomposition:
            sol = fsolve(
                lambda x, y, *args: self.R_x(x, y, *args),
                x0,
                jac=lu,
                fun_args=(y,),
                options=self.options,
            )
        else:
            sol = fsolve(
                lambda x, y, *args: self.R_x(x, y, *args),
                x0,
                jac=lambda x, y, *args: self.J_x(x, y, *args),
                fun_args=(y,),
                jac_args=(y,),
                options=self.options,
            )

        return sol

    def solve(self):
        self.solver_summary = SolverSummary("Backward Euler")

        def make_solution():
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
                solver_summary=self.solver_summary,
            )

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

        J_x = self.J_x(self.xn, self.yn)
        lu = splu(J_x)

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # only compute optimized prox-parameters once per time step
            # self.prox_r_N, self.prox_r_F = np.array_split(
            #     estimate_prox_parameter(
            #         self.options.prox_scaling, bmat([[self.W_N, self.W_F]]), self.M
            #     ),
            #     [self.nla_N],
            # )

            A = -lu.solve(self.J_y(self.xn, self.yn).toarray())
            # fmt: off
            g_x = bmat([
                [             self.system.g_N_q(self.tn, self.qn),       None],
                [self.system.gamma_F_q(self.tn, self.qn, self.un), self.W_F.T],
            ])
            # fmt: on
            prox_r = self.options.prox_scaling / np.abs(g_x @ A[: self.split_x[1]]).sum(
                axis=1
            )
            self.prox_r_N, self.prox_r_F = np.array_split(prox_r, self.split_y)
            self.prox_r_N *= self.dt

            # perform a solver step
            tn1 = self.tn + self.dt

            ########################
            # fixed-point iterations
            ########################
            # store old values
            x0 = self.xn.copy()
            y0 = self.yn.copy()

            # Jacobian and lu-decompositon
            if self.options.reuse_lu_decomposition:
                J_x = self.J_x(x0, y0)
                lu = splu(J_x)
                self.solver_summary.add_lu(1)

            # solve nonlinear system
            sol = self._solve_nonlinear_system(x0, y0, lu)
            xn1 = sol.x
            self.solver_summary.add_lu(sol.njev)

            if not sol.success:
                if self.options.continue_with_unconverged:
                    warnings.warn(
                        "Newton is not converged but integration is continued"
                    )
                else:
                    warnings.warn(
                        f"Newton is not converged. Returning solution up to t={self.tn}."
                    )
                    return make_solution()

            # fixed-point loop
            converged = False
            if self.nla_N + self.nla_F > 0:
                for i_fixed_point in range(self.options.fixed_point_max_iter):
                    # find proximal point
                    yn1 = self.prox(xn1, y0)

                    # error measure, see Hairer1993, Section II.4
                    diff = yn1 - y0
                    sc = (
                        self.options.fixed_point_atol
                        + np.maximum(np.abs(yn1), np.abs(y0))
                        * self.options.fixed_point_rtol
                    )
                    error = np.linalg.norm(diff / sc) / sc.size**0.5
                    converged = error < 1.0

                    if converged:
                        break
                    else:
                        # update values
                        x0 = xn1.copy()
                        y0 = yn1.copy()

                        # solve nonlinear system
                        sol = self._solve_nonlinear_system(x0, y0, lu)
                        xn1 = sol.x
                        self.solver_summary.add_lu(sol.njev)

                        if not sol.success:
                            if self.options.continue_with_unconverged:
                                warnings.warn(
                                    "Newton is not converged but integration is continued"
                                )
                            else:
                                RuntimeWarning(
                                    "Newton is not converged. Returning solution up to t={self.tn}."
                                )
                                return make_solution()

            else:
                converged = True
                i_fixed_point = 0
                error = 0

            if not converged:
                if self.options.continue_with_unconverged:
                    warnings.warn(
                        "fixed-point iteration is not converged but integration is continued"
                    )
                else:
                    warnings.warn(
                        f"fixed-point iteration is not converged. Returning solution up to t={self.tn}."
                    )
                    return make_solution()

            self.solver_summary.add_fixed_point(i_fixed_point, error)
            self.solver_summary.add_newton(sol.nit, sol.error)

            # update progress bar
            pbar.set_description(
                f"t: {tn1:0.2e}s < {self.t1:0.2e}s; |y1 - y0|_rel: {error:0.2e}; fixed-point: {i_fixed_point}/{self.options.fixed_point_max_iter}; |dx|_rel: {sol.error:0.2e}; newton: {sol.nit}/{self.options.newton_max_iter}"
            )

            # compute state
            (
                dqn1,
                dun1,
                dP_gn1,
                dP_gamman1,
                dP_cn1,
            ) = np.array_split(xn1, self.split_x)
            (
                dP_Nn1,
                dP_Fn1,
            ) = np.array_split(y0, self.split_y)
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

            # update local variables for accepted time step
            self.xn = xn1.copy()
            self.yn = y0.copy()
            self.tn = tn1
            self.qn = qn1
            self.un = un1

        self.solver_summary.print()

        # write solution
        return make_solution()
