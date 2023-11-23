import numpy as np
from scipy.sparse import csc_array
from tqdm import tqdm

from cardillo.math.prox import NegativeOrthant
from cardillo.math import fsolve, approx_fprime, estimate_prox_parameter
from cardillo.solver import Solution, SolverOptions, SolverSummary


class Rattle:
    def __init__(
        self,
        system,
        t1,
        dt,
        options=SolverOptions(numerical_jacobian_method="2-point"),
    ):
        """
        Nonsmooth extension of RATTLE.

        A nice interpretation of the left and right limes are found in Hante2019.

        References:
        -----------
        Hante2019: https://doi.org/10.1016/j.cam.2019.112492
        """
        self.system = system
        self.options = options

        if options.numerical_jacobian_method:
            self.J_x1 = lambda x, y: csc_array(
                approx_fprime(
                    x,
                    lambda x: self.R_x1(x, y),
                    method=options.numerical_jacobian_method,
                    eps=options.numerical_jacobian_eps,
                )
            )
            self.J_x2 = lambda x, y: csc_array(
                approx_fprime(
                    x,
                    lambda x: self.R_x2(x, y),
                    method=options.numerical_jacobian_method,
                    eps=options.numerical_jacobian_eps,
                )
            )
        else:
            self.J_x1 = self._J_x1
            self.J_x2 = self._J_x2

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
        # self.la_cn = system.la_c0
        # self.P_gn = dt * system.la_g0
        # self.P_gamman = dt * system.la_gamma0
        # self.P_Nn = dt * system.la_N0
        # self.P_Fn = dt * system.la_F0
        # self.mu_Sn = np.zeros(self.nla_S)

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
        # TODO: precompute q_dot_u, beta for performance
        R[: self.split_x1[0]] = (
            qn1
            - qn
            - 0.5
            * dt
            * (self.system.q_dot(tn, qn, un12) + self.system.q_dot(tn1, qn1, un12))
        )

        ########################
        # euations of motion (1)
        ########################
        R[self.split_x1[0] : self.split_x1[1]] = (
            self.Mn @ (un12 - un)
            - 0.5 * dt * (self.system.h(tn, qn, un12) + self.W_cn @ la_c1)
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
        raise NotImplementedError

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

    def R_x2(self, x2n1, y2n1):
        dt = self.dt
        tn = self.tn

        un12 = self.x1n[self.nq : self.nq + self.nu]

        tn1 = tn + dt
        qn1 = self.x1n[: self.nq]
        un1, P_g2, P_gamma2 = np.array_split(x2n1, self.split_x2)
        P_N2, P_F2 = np.array_split(y2n1, self.split_y)

        R = np.zeros_like(x2n1)

        ########################
        # euations of motion (1)
        ########################
        self.Mn = self.system.M(tn1, qn1, format="csr")
        self.W_cn = self.system.W_c(tn1, qn1, format="csr")
        self.W_gn = self.system.W_g(tn1, qn1, format="csr")
        self.W_gamman = self.system.W_gamma(tn1, qn1, format="csr")
        self.W_Nn = self.system.W_N(tn1, qn1, format="csr")
        self.W_Fn = self.system.W_F(tn1, qn1, format="csr")
        self.la_c2 = self.system.la_c(tn1, qn1, un12)

        R[: self.split_x2[0]] = (
            self.Mn @ (un1 - un12)
            - 0.5 * dt * (self.system.h(tn1, qn1, un12) + self.W_cn @ self.la_c2)
            - (
                self.W_gn @ P_g2
                + self.W_gamman @ P_gamma2
                + self.W_Nn @ P_N2
                + self.W_Fn @ P_F2
            )
        )

        #######################
        # bilateral constraints
        #######################
        R[self.split_x2[0] : self.split_x2[1]] = self.system.g_dot(tn1, qn1, un1)
        R[self.split_x2[1] :] = self.system.gamma(tn1, qn1, un1)

        return R

    def _J_x2(self, x2n1, y2n1):
        raise NotImplementedError

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

    def solve(self):
        solver_summary = SolverSummary()
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
            self.prox_r_N = estimate_prox_parameter(
                self.options.prox_scaling, self.W_Nn, self.Mn
            )
            self.prox_r_F = estimate_prox_parameter(
                self.options.prox_scaling, self.W_Fn, self.Mn
            )

            # perform a solver step
            tn1 = self.tn + self.dt

            #########
            # Stage 1
            #########
            # fixed-point iterations

            # store old values
            x1n = self.x1n.copy()
            y1n = self.y1n.copy()

            # fixed-point loop
            x1n1 = x1n.copy()
            y1n1 = y1n.copy()
            converged = False
            n_state = self.nx1 - self.nla_g - self.nla_gamma
            for i_fixed_point in range(self.options.fixed_point_max_iter):
                # find proximal point
                y1n1 = self.prox1(x1n1, y1n1)

                x1n1, converged_newton, error_newton, i_newton, _ = fsolve(
                    self.R_x1,
                    x1n1,
                    jac=self.J_x1,
                    fun_args=(y1n1,),
                    jac_args=(y1n1,),
                    atol=self.options.newton_atol,
                    max_iter=self.options.newton_max_iter,
                )
                solver_summary.add_lu(i_newton)

                # convergence in smooth state (without Lagrange multipliers)
                diff = x1n1[:n_state] - x1n[:n_state]

                # error measure, see Hairer1993, Section II.4
                sc = (
                    self.options.fixed_point_atol
                    + np.maximum(np.abs(x1n[:n_state]), np.abs(x1n1[:n_state]))
                    * self.options.fixed_point_rtol
                )
                error = np.linalg.norm(diff / sc) / sc.size**0.5
                converged = error < 1.0 and converged_newton

                if converged:
                    break
                else:
                    # update values
                    x1n = x1n1.copy()
                    y1n = y1n1.copy()

            fixed_point_absolute_error = np.max(np.abs(diff))
            solver_summary.add_fixed_point(i_fixed_point, fixed_point_absolute_error)
            solver_summary.add_newton(i_newton)

            if not converged:
                raise ValueError("not converged")

            # save converged quantities of first stage. Required for R_x2
            self.x1n = x1n1.copy()
            self.y1n = y1n1.copy()

            self.I_N = y1n1[: self.split_y[0]] > 0

            #########
            # Stage 2
            #########
            # fixed-point iterations

            # store old values
            x2n = self.x2n.copy()
            y2n = self.y2n.copy()

            # fixed-point loop
            x2n1 = x2n.copy()
            y2n1 = y2n.copy()
            converged = False

            for i_fixed_point in range(self.options.fixed_point_max_iter):
                # find proximal point
                y2n1 = self.prox2(x2n1, y2n1)

                x2n1, converged_newton, error_newton, i_newton, _ = fsolve(
                    self.R_x2,
                    x2n1,
                    jac=self.J_x2,
                    fun_args=(y2n1,),
                    jac_args=(y2n1,),
                    atol=self.options.newton_atol,
                    max_iter=self.options.newton_max_iter,
                )
                solver_summary.add_lu(i_newton)

                # convergence in smooth state (without Lagrange multipliers)
                diff = x2n1[: self.nu] - x2n[: self.nu]

                # error measure, see Hairer1993, Section II.4
                sc = (
                    self.options.fixed_point_atol
                    + np.maximum(np.abs(x2n[: self.nu]), np.abs(x2n1[: self.nu]))
                    * self.options.fixed_point_rtol
                )
                error = np.linalg.norm(diff / sc) / sc.size**0.5
                converged = error < 1.0 and converged_newton

                if converged:
                    break
                else:
                    # update values
                    x2n = x2n1.copy()
                    y2n = y2n1.copy()

            fixed_point_absolute_error = np.max(np.abs(diff))
            solver_summary.add_fixed_point(i_fixed_point, fixed_point_absolute_error)
            solver_summary.add_newton(i_newton)

            if not converged:
                raise ValueError("not converged")

            # update progress bar
            pbar.set_description(
                f"t: {tn1:0.2e}s < {self.t1:0.2e}s; |x1 - x0|: {fixed_point_absolute_error:0.2e} (fixed-point: {i_fixed_point}/{self.options.fixed_point_max_iter}; newton: {i_newton}/{self.options.newton_max_iter})"
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

        solver_summary.print()
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
        )
