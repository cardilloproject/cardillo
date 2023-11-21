import numpy as np
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse import bmat, eye, diags, csc_array
from tqdm import tqdm

from cardillo.math.prox import prox_R0_nm, prox_R0_np, prox_sphere
from cardillo.math import fsolve, approx_fprime, estimate_prox_parameter
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
        else:
            self.J_x1 = self._J_x1

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
        self.nla_S = self.system.nla_S

        self.nx1 = (
            self.nq + self.nu + self.nla_c + self.nla_g + self.nla_gamma + self.nla_S
        )
        self.nx2 = self.nu + self.nla_c + self.nla_g + self.nla_gamma 

        self.ny = self.nla_N + self.nla_F

        self.split_x1 = np.cumsum(
            np.array(
                [
                    self.nq,
                    self.nu,
                    self.nla_c,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_S,
                ],
                dtype=int,
            )
        )[:-1]

        self.split_x2 = np.cumsum(
            np.array(
                [
                    self.nu,
                    self.nla_c,
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
        self.la_cn = system.la_c0
        self.P_gn = dt * system.la_g0
        self.P_gamman = dt * system.la_gamma0
        self.P_Nn = dt * system.la_N0
        self.P_Fn = dt * system.la_F0
        self.mu_Sn = np.zeros(self.nla_S)

        #######################################################################
        # initial values
        #######################################################################
        self.x1n = np.concatenate(
            (
                self.qn,
                self.un,
                self.la_cn,
                self.P_gn,
                self.P_gamman,
                self.mu_Sn,
            )
        )
        self.x2n = np.concatenate(
            (
                self.un,
                self.la_cn,
                self.P_gn,
                self.P_gamman,
            )
        )
        self.y1n = np.concatenate(
            (
                self.P_Nn,
                self.P_Fn,
            )
        )
        self.y2n = np.zeros_like(self.y1n)

        ###################################################
        # compute quantities for prox estimation
        ###################################################
        self.Mn = system.M(self.tn, self.qn, format="csr")
        self.W_Nn = system.W_N(self.tn, self.qn, format="csr")
        self.W_Fn = system.W_F(self.tn, self.qn, format="csr")

    def R_x1(self, xn1, yn1):
        tn = self.tn
        dt = self.dt
        tn1 = tn + dt
        qn = self.x1n[: self.nq]
        un = self.x2n[: self.nu]

        (qn1, un12, la_c1, P_g1, P_gamma1, mu_S1) = np.array_split(xn1, self.split_x1)

        P_N1, P_F1 = np.array_split(yn1, self.split_y)

        R = np.zeros_like(xn1)

        ####################
        # kinematic equation
        ####################
        g_S_q = self.system.g_S_q(tn1, qn1, format="csc")
        R[: self.split_x1[0]] = (
            qn1
            - qn
            - 0.5
            * dt
            * (self.system.q_dot(tn, qn, un12) + self.system.q_dot(tn1, qn1, un12))
            - g_S_q.T @ mu_S1
        )

        ########################
        # euations of motion (1)
        ########################
        R[self.split_x1[0] : self.split_x1[1]] = self.system.M(tn, qn, format="csr") @ (
            un12 - un
        ) - dt * (
            0.5 * self.system.h(tn, qn, un12)
            + self.system.W_c(tn, qn) @ la_c1) - (self.system.W_g(tn, qn) @ P_g1
            + self.system.W_gamma(tn, qn) @ P_gamma1
            + self.system.W_N(tn, qn) @ P_N1
            + self.system.W_F(tn, qn) @ P_F1
        )

        ############
        # compliance
        ############
        R[self.split_x1[1] : self.split_x1[2]] = self.system.c(tn1, qn1, un12, la_c1)

        #######################
        # bilateral constraints
        #######################
        R[self.split_x1[2] : self.split_x1[3]] = self.system.g(tn1, qn1)
        R[self.split_x1[3] : self.split_x1[4]] = self.system.gamma(tn1, qn1, un12)

        ##########################
        # quaternion stabilization
        ##########################
        R[self.split_x1[4] :] = self.system.g_S(tn1, qn1)

        return R

    def _J_x1(self, xn1, yn1):
        raise NotImplementedError

    def prox1(self, xn1, yn1):
        tn = self.tn
        dt = self.dt
        tn1 = tn + dt
        (
            qn1,
            un12,
            _,
            _,
            _,
            _,
        ) = np.array_split(xn1, self.split_x1)

        P_N1, P_F1 = np.array_split(yn1, self.split_y)

        mu = self.system.mu
        prox_r_N = self.prox_r_N
        prox_r_F = self.prox_r_F

        yn1p = np.zeros_like(yn1) # initialize projected forces

        ##############################
        # fixed-point update Signorini
        ##############################
        g_N = self.system.g_N(tn1, qn1)
        prox_arg = (prox_r_N / self.dt) * g_N - P_N1
        yn1p[: self.split_y[0]] = -prox_R0_nm(prox_arg)

        #############################
        # fixed-point update friction
        #############################
        gamma_F = self.system.gamma_F(tn1, qn1, un12)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            if len(i_F):
                yn1p[self.split_y[0] + np.array(i_F)] = -prox_sphere(
                    prox_r_F[i_N] * gamma_F[i_F] - P_F1[i_F],
                    mu[i_N] * P_N1[i_N],
                )

        return yn1p

    def solve(self):
        solver_summary = SolverSummary()
        # lists storing output variables
        t = [self.tn]
        q = [self.qn]
        u = [self.un]
        la_c = [self.la_cn]
        P_g = [self.P_gn]
        P_gamma = [self.P_gamman]
        P_N = [self.P_Nn]
        P_F = [self.P_Fn]
        mu_S = [self.mu_Sn]


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
                        self.x1n,
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
                raise ValueError('not converged')

            # update progress bar
            pbar.set_description(
                f"t: {tn1:0.2e}s < {self.t1:0.2e}s; |x1 - x0|: {fixed_point_absolute_error:0.2e} (fixed-point: {i_fixed_point}/{self.options.fixed_point_max_iter}; newton: {i_newton}/{self.options.newton_max_iter})"
            )

            # compute state
            (qn1, un12, la_c1, P_g1, P_gamma1, mu_S1) = np.array_split(x1n1, self.split_x1)
            P_N1, P_F1 = np.array_split(y1n1, self.split_y)

            un1 = un12
            # modify converged quantities
            qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            t.append(tn1)
            q.append(qn1)
            u.append(un1)
            la_c.append(la_c1)
            P_g.append(P_g1)
            P_gamma.append(P_gamma1)
            P_N.append(P_N1)
            P_F.append(P_F1)
            mu_S.append(mu_S1)

            # update local variables for accepted time step
            self.x1n = x1n1.copy()
            self.y1n = y1n1.copy()
            self.x2n[: self.nu] = un1.copy()
            self.tn = tn1

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
            mu_S=np.array(mu_S)
        )
