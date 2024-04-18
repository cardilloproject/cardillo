import warnings
import numpy as np
from scipy.sparse import lil_array, eye, bmat
from scipy.sparse.linalg import splu
from tqdm import tqdm

from cardillo.math.prox import NegativeOrthant, estimate_prox_parameter
from cardillo.math.fsolve import fsolve
from cardillo.solver import Solution, SolverOptions, SolverSummary


class Rattle:
    def __init__(
        self,
        system,
        t1,
        dt,
        options=SolverOptions(),
    ):
        """
        A nonsmooth RATTLE algorithm for mechanical systems with frictional
        unilateral constraints, see Breuling2024.

        References:
        -----------
        Breuling2024: https://doi.org/10.1016/j.nahs.2024.101469
        """
        self.system = system
        self.options = options

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
        self.nla_c = system.nla_c
        self.nla_g = system.nla_g
        self.nla_gamma = system.nla_gamma
        self.nla_N = system.nla_N
        self.nla_F = system.nla_F

        self.nx1 = self.nq + self.nu + self.nla_c + self.nla_g + self.nla_gamma
        self.nx2 = self.nu + self.nla_g + self.nla_gamma

        self.ny = self.nla_N + self.nla_F

        self.split_x1 = np.cumsum(
            np.array(
                [self.nq, self.nu, self.nla_c, self.nla_g, self.nla_gamma],
                dtype=int,
            )
        )[:-1]

        self.split_x2 = np.cumsum(
            np.array(
                [
                    self.nu,
                    self.nla_g,
                    self.nla_gamma,
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

        #######################################################################
        # initial values
        #######################################################################
        self.x1n = np.concatenate(
            (
                system.q0,
                system.u0,
                system.la_c0,
                dt * system.la_g0,
                dt * system.la_gamma0,
            )
        )
        self.x2n = np.concatenate(
            (
                system.u0,
                np.zeros(self.nla_g),
                np.zeros(self.nla_gamma),
            )
        )
        self.y1n = np.concatenate((dt * system.la_N0, dt * system.la_F0))
        self.y2n = np.zeros_like(self.y1n)

        ###################################################
        # compute quantities for prox estimation
        ###################################################
        self.Mn = self.system.M(self.tn, self.qn, format="csr")
        self.Bn = self.system.q_dot_u(self.tn, self.qn, format="csr")
        self.betan = self.system.q_dot(self.tn, self.qn, 0 * self.un)
        self.W_taun = self.system.W_tau(self.tn, self.qn, format="csr")
        self.W_cn = self.system.W_c(self.tn, self.qn, format="csr")
        self.W_gn = self.system.W_g(self.tn, self.qn, format="csr")
        self.W_gamman = self.system.W_gamma(self.tn, self.qn, format="csr")
        self.W_Nn = self.system.W_N(self.tn, self.qn, format="csr")
        self.W_Fn = self.system.W_F(self.tn, self.qn, format="csr")

    def R_x1(self, x1n1, y1n1):
        dt = self.dt
        tn = self.tn
        qn = self.qn
        un = self.un

        tn1 = tn + dt

        qn1, un12, la_c1, P_g1, P_gamma1 = np.array_split(x1n1, self.split_x1)
        P_N1, P_F1 = np.array_split(y1n1, self.split_y)

        R = np.zeros_like(x1n1)

        ####################
        # kinematic equation
        ####################
        R[: self.split_x1[0]] = (
            qn1
            - qn
            - 0.5 * dt
            # * (self.system.q_dot(tn, qn, un12) + self.system.q_dot(tn1, qn1, un12))
            * (self.Bn @ un12 + self.betan + self.system.q_dot(tn1, qn1, un12))
        )

        ########################
        # euations of motion (1)
        ########################
        R[self.split_x1[0] : self.split_x1[1]] = (
            self.Mn @ (un12 - un)
            - 0.5
            * dt
            * (
                self.system.h(tn, qn, un12)
                + self.W_taun @ self.system.la_tau(tn, qn, un12)
                + self.W_cn @ la_c1
            )
            - (
                self.W_gn @ P_g1
                + self.W_gamman @ P_gamma1
                + self.W_Nn @ P_N1
                + self.W_Fn @ P_F1
            )
        )

        ############
        # compliance
        ############
        R[self.split_x1[1] : self.split_x1[2]] = self.system.c(tn, qn, un12, la_c1)

        #######################
        # bilateral constraints
        #######################
        R[self.split_x1[2] : self.split_x1[3]] = self.system.g(tn1, qn1)
        R[self.split_x1[3] :] = self.system.gamma(tn1, qn1, un12)

        return R

    def _J_x1(self, x1n1, y1n1):
        dt = self.dt
        tn = self.tn
        qn = self.qn
        un = self.un

        tn1 = tn + dt

        qn1, un12, la_c1, _, _ = np.array_split(x1n1, self.split_x1)

        J_x1 = lil_array((self.nx1, self.nx1))

        ####################
        # kinematic equation
        ####################
        J_x1[: self.split_x1[0], : self.split_x1[0]] = eye(
            self.nq
        ) - 0.5 * dt * self.system.q_dot_q(tn1, qn1, un12)
        J_x1[: self.split_x1[0], self.split_x1[0] : self.split_x1[1]] = (
            -0.5 * dt * (self.Bn + self.system.q_dot_u(tn1, qn1))
        )

        ########################
        # euations of motion (1)
        ########################
        J_x1[
            self.split_x1[0] : self.split_x1[1], self.split_x1[0] : self.split_x1[1]
        ] = self.Mn - 0.5 * dt * (
            self.system.h_u(tn, qn, un12)
            + self.system.Wla_tau_u(self.tn, self.qn, un12)
        )
        J_x1[
            self.split_x1[0] : self.split_x1[1], self.split_x1[1] : self.split_x1[2]
        ] = (-0.5 * dt * self.W_cn)
        J_x1[
            self.split_x1[0] : self.split_x1[1], self.split_x1[2] : self.split_x1[3]
        ] = -self.W_gn
        J_x1[self.split_x1[0] : self.split_x1[1], self.split_x1[3] :] = -self.W_gamman

        ############
        # compliance
        ############
        J_x1[
            self.split_x1[1] : self.split_x1[2], self.split_x1[0] : self.split_x1[1]
        ] = self.system.c_u(tn, qn, un12, la_c1)
        J_x1[
            self.split_x1[1] : self.split_x1[2], self.split_x1[1] : self.split_x1[2]
        ] = self.system.c_la_c()

        #######################
        # bilateral constraints
        #######################
        J_x1[self.split_x1[2] : self.split_x1[3], : self.split_x1[0]] = self.system.g_q(
            tn1, qn1
        )
        J_x1[self.split_x1[3] :, : self.split_x1[0]] = self.system.gamma_q(
            tn1, qn1, un12
        )
        J_x1[self.split_x1[3] :, self.split_x1[0] : self.split_x1[1]] = (
            self.system.gamma_u(tn1, qn1)
        )

        return J_x1.tocsc()

        # J_x1_num = csc_array(
        #     approx_fprime(x1n1, lambda x: self.R_x1(x, y1n1), method="cs")
        # )
        # diff = J_x1 - J_x1_num
        # error = np.linalg.norm(diff.toarray())
        # print(f"error J_x1: {error}")
        # return J_x1_num

    def prox1(self, x1n1, y1n1):
        tn = self.tn
        dt = self.dt
        tn1 = tn + dt
        (
            qn1,
            un12,
            _,
            _,
            _,
        ) = np.array_split(x1n1, self.split_x1)

        P_N1, P_F1 = np.array_split(y1n1, self.split_y)

        prox_r_N = self.prox_r_N
        prox_r_F = self.prox_r_F

        y1n1p = np.zeros_like(y1n1)  # initialize projected forces

        ##############################
        # fixed-point update Signorini
        ##############################
        g_N = self.system.g_N(tn1, qn1)
        prox_arg = (prox_r_N / self.dt) * g_N - P_N1
        self.I_N = prox_arg <= 0  # active set for second stage
        y1n1p[: self.split_y[0]] = -NegativeOrthant.prox(prox_arg)

        #############################
        # fixed-point update friction
        #############################
        for contr in self.system.get_contribution_list("gamma_F"):
            gamma_F_contr = contr.gamma_F(tn1, qn1[contr.qDOF], un12[contr.uDOF])
            la_FDOF = contr.la_FDOF
            P_Fn1_contr = P_F1[la_FDOF]
            prox_r_F_contr = prox_r_F[la_FDOF]
            for i_N, i_F, force_recervoir in contr.friction_laws:
                if len(i_N) > 0:
                    P_Nn1i = P_N1[contr.la_NDOF[i_N]]
                else:
                    P_Nn1i = self.dt

                y1n1p[self.split_y[0] + la_FDOF[i_F]] = -force_recervoir.prox(
                    prox_r_F_contr[i_F] * gamma_F_contr[i_F] - P_Fn1_contr[i_F],
                    P_Nn1i,
                )

        return y1n1p

    def prox2(self, x2n1, y2n1):
        dt = self.dt

        tn = self.tn
        qn = self.qn
        un = self.un

        tn1 = tn + dt

        qn1 = self.x1n[: self.nq]
        un1 = x2n1[: self.nu]

        P_N, P_F = np.array_split(self.y1n + y2n1, self.split_y)

        prox_r_N = self.prox_r_N
        prox_r_F = self.prox_r_F

        y2n1p = np.zeros_like(y2n1)  # initialize projected forces

        ##############################
        # fixed-point update Signorini
        ##############################
        xi_N = self.system.xi_N(tn, tn1, qn, qn1, un, un1)
        y2n1p[: self.split_y[0]] = self.I_N * (
            -NegativeOrthant.prox(prox_r_N * xi_N - P_N)
        )

        #############################
        # fixed-point update friction
        #############################
        xi_F = self.system.xi_F(tn, tn1, qn, qn1, un, un1)
        for contr in self.system.get_contribution_list("gamma_F"):
            la_FDOF = contr.la_FDOF
            xi_F_contr = xi_F[la_FDOF]
            P_F_contr = P_F[la_FDOF]
            prox_r_F_contr = prox_r_F[la_FDOF]
            for i_N, i_F, force_recervoir in contr.friction_laws:
                if len(i_N) > 0:
                    P_Ni = P_N[contr.la_NDOF[i_N]]
                else:
                    P_Ni = self.dt

                y2n1p[self.split_y[0] + la_FDOF[i_F]] = -force_recervoir.prox(
                    prox_r_F_contr[i_F] * xi_F_contr[i_F] - P_F_contr[i_F],
                    P_Ni,
                )

        return y2n1p - self.y1n

    def _solve_nonlinear_system(self, x0, y, lu):
        if self.options.reuse_lu_decomposition:
            sol = fsolve(
                lambda x, y, *args: self.R_x1(x, y, *args),
                x0,
                jac=lu,
                fun_args=(y,),
                options=self.options,
            )
        else:
            sol = fsolve(
                lambda x, y, *args: self.R_x1(x, y, *args),
                x0,
                jac=lambda x, y, *args: self._J_x1(x, y, *args),
                fun_args=(y,),
                jac_args=(y,),
                options=self.options,
            )

        if not sol.success:
            if self.options.continue_with_unconverged:
                warnings.warn("Newton is not converged but integration is continued")
            else:
                raise RuntimeError("Newton is not converged")

        return sol

    def _iterative_projection_method(self, x0, y0, lu=None):
        # solve nonlinear system
        sol = self._solve_nonlinear_system(x0, y0, lu)
        x0 = sol.x
        self.solver_summary.add_lu(sol.njev)

        # fixed-point loop
        converged = False
        n_state = self.nx1 - self.nla_g - self.nla_gamma
        for i_fixed_point in range(self.options.fixed_point_max_iter):
            # find proximal point
            y1 = self.prox1(x0, y0)

            # solve nonlinear system
            sol = self._solve_nonlinear_system(x0, y1, lu)
            x1 = sol.x
            self.solver_summary.add_lu(sol.njev)

            # convergence in smooth state (without Lagrange multipliers)
            diff = x1[:n_state] - x0[:n_state]

            # error measure, see Hairer1993, Section II.4
            sc = (
                self.options.fixed_point_atol
                + np.maximum(np.abs(x0[:n_state]), np.abs(x1[:n_state]))
                * self.options.fixed_point_rtol
            )
            error = np.linalg.norm(diff / sc) / sc.size**0.5
            converged = error < 1.0 and sol.success

            if converged:
                break
            else:
                # update values
                x0 = x1.copy()
                y0 = y1.copy()

        self.solver_summary.add_fixed_point(i_fixed_point, error)
        self.solver_summary.add_newton(sol.nit, sol.error)

        if not converged:
            if self.options.continue_with_unconverged:
                warnings.warn(
                    "fixed-point iteration is not converged in stage 1 but integration is continued"
                )
            else:
                raise RuntimeError("fixed-point iteration is not converged in stage 1")

        return x1, y1, i_fixed_point

    def solve(self):
        self.solver_summary = SolverSummary("RATTLE")

        # lists storing output variables
        _, _, la_c0, P_g0, P_gamma0 = np.array_split(self.x1n, self.split_x1)
        P_N0, P_F0 = np.array_split(self.y1n + self.y2n, self.split_y)

        t = [self.tn]
        q = [self.qn]
        u = [self.un]
        la_c = [la_c0]
        P_g = [P_g0]
        P_gamma = [P_gamma0]
        P_N = [P_N0]
        P_F = [P_F0]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # only compute optimized prox-parameters once per time step
            self.prox_r_N, self.prox_r_F = np.array_split(
                estimate_prox_parameter(
                    self.options.prox_scaling, bmat([[self.W_Nn, self.W_Fn]]), self.Mn
                ),
                [self.nla_N],
            )

            # perform a solver step
            tn1 = self.tn + self.dt

            #########
            # Stage 1
            #########
            # Jacobian and lu-decompositon
            if self.options.reuse_lu_decomposition:
                J_x1 = self._J_x1(self.x1n, self.y1n)
                lu = splu(J_x1)
                self.solver_summary.add_lu(1)
            else:
                lu = None

            # fixed-point iterations
            x1n1, y1n1, i1_fixed_point = self._iterative_projection_method(
                self.x1n, self.y1n, lu
            )

            # save converged quantities of first stage. Required for R_x2
            self.x1n = x1n1.copy()
            self.y1n = y1n1.copy()

            # compute constant quantities for second stage once
            qn1, un12, _, _, _ = np.array_split(x1n1, self.split_x1)
            self.Mn = self.system.M(tn1, qn1, format="csr")
            self.Bn = self.system.q_dot_u(tn1, qn1, format="csr")
            self.betan = self.system.q_dot(tn1, qn1, 0 * un12)
            self.W_taun = self.system.W_tau(tn1, qn1, format="csr")
            self.W_cn = self.system.W_c(tn1, qn1, format="csr")
            self.W_gn = self.system.W_g(tn1, qn1, format="csr")
            self.W_gamman = self.system.W_gamma(tn1, qn1, format="csr")
            self.W_Nn = self.system.W_N(tn1, qn1, format="csr")
            self.W_Fn = self.system.W_F(tn1, qn1, format="csr")
            self.W_FNn = bmat([[self.W_Nn, self.W_Fn]], format="csr")
            self.la_c2 = self.system.la_c(tn1, qn1, un12)

            # constant part of the rhs
            b0 = np.concatenate(
                [
                    -self.Mn @ un12
                    - 0.5
                    * self.dt
                    * (
                        self.system.h(tn1, qn1, un12)
                        + self.system.W_tau(tn1, qn1, format="csr")
                        @ self.system.la_tau(tn1, qn1, un12)
                        + self.W_cn @ self.la_c2
                    ),
                    self.system.chi_g(tn1, qn1),
                    self.system.chi_gamma(tn1, qn1),
                ]
            )

            # iteration matrix
            g_dot_u = self.system.g_dot_u(tn1, qn1)
            gamma_u = self.system.gamma_u(tn1, qn1)
            # fmt: off
            A = bmat([
                [self.Mn, -self.W_gn, -self.W_gamman],
                [g_dot_u,       None,           None],
                [gamma_u,       None,           None],
            ], format="csc")
            # fmt: on
            lu = splu(A)
            self.solver_summary.add_lu(1)

            #########
            # Stage 2
            #########

            # store old values
            x2n = self.x2n.copy()
            y2n = self.y2n.copy()

            # solve LGS
            b = b0.copy()  # mandatory copy
            b[: self.nu] -= self.W_FNn @ y2n
            x2n1 = -lu.solve(b)

            # fixed-point iterations
            y2n1 = y2n.copy()
            converged = False
            for i2_fixed_point in range(self.options.fixed_point_max_iter):
                # find proximal point
                y2n1 = self.prox2(x2n1, y2n1)

                # convergence in smooth state (without Lagrange multipliers)
                diff = x2n1[: self.nu] - x2n[: self.nu]

                # error measure, see Hairer1993, Section II.4
                sc = (
                    self.options.fixed_point_atol
                    + np.maximum(np.abs(x2n[: self.nu]), np.abs(x2n1[: self.nu]))
                    * self.options.fixed_point_rtol
                )
                error = np.linalg.norm(diff / sc) / sc.size**0.5
                converged = error < 1.0

                if converged:
                    break
                else:
                    # update values
                    x2n = x2n1.copy()
                    y2n = y2n1.copy()

                    # solve LGS
                    b = b0.copy()  # mandatory copy
                    b[: self.nu] -= self.W_FNn @ y2n
                    x2n1 = -lu.solve(b)

            self.solver_summary.add_fixed_point(i2_fixed_point, error)

            if not converged:
                if self.options.continue_with_unconverged:
                    warnings.warn(
                        "fixed-point iteration is not converged in stage 2 but integration is continued"
                    )
                else:
                    raise RuntimeError(
                        "fixed-point iteration is not converged in stage 2"
                    )

            # update progress bar
            pbar.set_description(
                f"t: {tn1:0.2e}s < {self.t1:0.2e}s; |x1 - x0|_rel: {error:0.2e}; fixed-point: ({i1_fixed_point}, {i2_fixed_point})/{self.options.fixed_point_max_iter}"
            )

            # compute state
            qn1, un12, la_c1, P_g1, P_gamma1 = np.array_split(x1n1, self.split_x1)
            P_N1, P_F1 = np.array_split(y1n1, self.split_y)
            un1, P_g2, P_gamma2 = np.array_split(x2n1, self.split_x2)
            P_N2, P_F2 = np.array_split(y2n1, self.split_y)

            # modify converged quantities
            qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            t.append(tn1)
            q.append(qn1)
            u.append(un1)
            la_c.append(0.5 * (la_c1 + self.la_c2))
            P_g.append(P_g1 + P_g2)
            P_gamma.append(P_gamma1 + P_gamma2)
            P_N.append(P_N1 + P_N2)
            P_F.append(P_F1 + P_F2)

            # update local variables for accepted time step
            self.x2n = x2n1.copy()
            self.y2n = y2n1.copy()
            self.tn = tn1
            self.qn = qn1
            self.un = un1

        self.solver_summary.print()
        return Solution(
            system=self.system,
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            la_c=np.array(la_c),
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
            solver_summary=self.solver_summary,
        )
