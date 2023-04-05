import numpy as np
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, bmat, eye, diags
from tqdm import tqdm

from cardillo.math.prox import prox_R0_nm, prox_R0_np, prox_sphere
from cardillo.math import fsolve, approx_fprime
from cardillo.solver import Solution, consistent_initial_conditions


# TODO:
# - Keep prox_r_N constant during each Newton-iteration since it is an expensive expression!
# - Improve Jacobian by passing evaluated quantities depending on (tn, qn, un)
# - Recycle already computed quantities of the LGS of step 2
class Rattle:
    def __init__(
        self,
        system,
        t1,
        dt,
        atol=1e-8,
        max_iter=50,
        fix_point_tol=1e-6,
        fix_point_max_iter=1000,
        error_function=lambda x: np.max(np.abs(x)),
        method="Newton_decoupled",
        # method="Newton_full",
        # method="fixed_point",
        # method="fixed_point_nonlinear_full",
        continue_with_unconverged=True,
    ):
        """
        Nonsmooth extension of RATTLE.

        A nice interpretation of the left and right limes are found in Hante2019.

        References:
        -----------
        Hante2019: https://doi.org/10.1016/j.cam.2019.112492
        """
        self.system = system
        self.method = method

        self.fix_point_error_function = error_function
        self.fix_point_tol = fix_point_tol
        self.fix_point_max_iter = fix_point_max_iter
        self.continue_with_unconverged = continue_with_unconverged

        # integration time
        t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.error_function = error_function
        self.atol = atol
        self.max_iter = max_iter

        self.nq = system.nq
        self.nu = system.nu
        self.nla_g = system.nla_g
        self.nla_gamma = system.nla_gamma
        self.nla_N = system.nla_N
        self.nla_F = system.nla_F

        # consistent initial conditions
        (
            self.tn,
            self.qn,
            self.un,
            self.q_dotn,
            self.u_dotn,
            self.la_gn,
            self.la_gamman,
        ) = consistent_initial_conditions(system)

        self.la_Nn = system.la_N0
        self.la_Fn = system.la_F0

        #####################
        # full coupled Newton
        #####################
        self.ny = (
            self.nq
            + 2 * self.nu
            + 2 * self.nla_g
            + 2 * self.nla_gamma
            + 2 * self.nla_N
            + 2 * system.nla_F
        )
        self.yn = np.concatenate(
            (
                self.qn,
                self.un,
                self.un,
                self.la_gn * 0.5 * dt,
                self.la_gn * 0.5 * dt,
                self.la_gamman * 0.5 * dt,
                self.la_gamman * 0.5 * dt,
                self.la_Nn * 0.5 * dt,
                self.la_Nn * 0.5 * dt,
                self.la_Fn * 0.5 * dt,
                self.la_Fn * 0.5 * dt,
            )
        )
        self.split_y = np.array(
            [
                self.nq,
                self.nq + self.nu,
                self.nq + 2 * self.nu,
                self.nq + 2 * self.nu + self.nla_g,
                self.nq + 2 * (self.nu + self.nla_g),
                self.nq + 2 * (self.nu + self.nla_g) + self.nla_gamma,
                self.nq + 2 * (self.nu + self.nla_g + self.nla_gamma),
                self.nq + 2 * (self.nu + self.nla_g + self.nla_gamma) + self.nla_N,
                self.nq + 2 * (self.nu + self.nla_g + self.nla_gamma + self.nla_N),
                self.nq
                + 2 * (self.nu + self.nla_g + self.nla_gamma + self.nla_N)
                + self.nla_F,
            ],
            dtype=int,
        )

        # # Solve for consistent initial conditions for the specific scheme
        # # TODO: Generalize this to all other implementations
        # self.prox_r_N = self.system.prox_r_N(self.tn, self.qn)
        # self.prox_r_F = self.system.prox_r_F(self.tn, self.qn)
        # y0, *_ = fsolve(self.R, self.yn, fun_args=(True,), jac_args=(False,))

        ############################
        # nonlinear fixed-point full
        ############################
        self.split_x = np.cumsum(
            np.array(
                [
                    self.nq,
                    self.nu,
                    self.nu,
                    self.nla_g,
                    self.nla_g,
                    self.nla_gamma,
                ],
                dtype=int,
            )
        )
        self.x = np.concatenate(
            (
                self.qn,
                self.un,
                self.un,
                self.la_gn * 0.5 * dt,
                self.la_gn * 0.5 * dt,
                self.la_gamman * 0.5 * dt,
                self.la_gamman * 0.5 * dt,
            )
        )
        self.nx = len(self.x)
        self.x0 = self.x.copy()
        self.split_z = np.cumsum(
            np.array(
                [self.nla_N, self.nla_N, self.nla_F],
                dtype=int,
            )
        )
        self.zn = np.concatenate(
            (
                self.la_Nn * 0.5 * dt,
                self.la_Nn * 0.5 * dt,
                self.la_Fn * 0.5 * dt,
                self.la_Fn * 0.5 * dt,
            )
        )

        ##################
        # decoupled Newton
        ##################
        self.ny1 = (
            self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_N + system.nla_F
        )
        self.y1n = np.concatenate(
            (
                self.qn,
                self.un,
                self.la_gn * 0.5 * dt,
                self.la_gamman * 0.5 * dt,
                self.la_Nn * 0.5 * dt,
                self.la_Fn * 0.5 * dt,
            )
        )
        self.split_y1 = np.array(
            [
                self.nq,
                self.nq + self.nu,
                self.nq + self.nu + self.nla_g,
                self.nq + self.nu + self.nla_g + self.nla_gamma,
                self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_N,
            ],
            dtype=int,
        )
        self.ny2 = self.nu + self.nla_g + self.nla_gamma + self.nla_N + system.nla_F
        self.y2n = np.concatenate(
            (
                self.un,
                self.la_gn * 0.5 * dt,
                self.la_gamman * 0.5 * dt,
                self.la_Nn * 0.5 * dt,
                self.la_Fn * 0.5 * dt,
            )
        )
        self.split_y2 = np.array(
            [
                self.nu,
                self.nu + self.nla_g,
                self.nu + self.nla_g + self.nla_gamma,
                self.nu + self.nla_g + self.nla_gamma + self.nla_N,
            ],
            dtype=int,
        )

        ################################
        # decouple fixed point iteration
        ################################
        self.split_x1 = np.array(
            [
                self.nq,
                self.nq + self.nu,
                self.nq + self.nu + self.nla_g,
            ],
            dtype=int,
        )
        self.x1 = np.concatenate(
            (
                self.qn,
                self.un,
                self.la_gn * 0.5 * dt,
                self.la_gamman * 0.5 * dt,
            )
        )
        self.x10 = self.x1.copy()
        self.z1n = np.concatenate((self.la_Nn * 0.5 * dt, self.la_Fn * 0.5 * dt))

        self.split_x2 = np.array(
            [
                self.nu,
                self.nu + self.nla_g,
            ],
            dtype=int,
        )
        self.x2 = np.concatenate(
            (
                self.un,
                self.la_gn * 0.5 * dt,
                self.la_gamman * 0.5 * dt,
            )
        )
        self.x20 = self.x2.copy()
        self.z2n = np.concatenate((self.la_Nn * dt, self.la_Fn * dt))

        ###################################################
        # compute constant quantities for current time step
        ###################################################
        self.Mn1 = system.M(self.tn, self.qn, scipy_matrix=csr_matrix)
        self.W_gn1 = system.W_g(self.tn, self.qn, scipy_matrix=csr_matrix)
        self.W_gamman1 = system.W_gamma(self.tn, self.qn, scipy_matrix=csr_matrix)
        self.W_Nn1 = system.W_N(self.tn, self.qn, scipy_matrix=csr_matrix)
        self.W_Fn1 = system.W_F(self.tn, self.qn, scipy_matrix=csr_matrix)

    def R(self, y, update_index=False):
        tn = self.tn
        dt = self.dt
        tn1 = tn + dt
        qn, _, un, _, _, _, _, _, _, _, _ = np.array_split(self.yn, self.split_y)
        (
            qn1,
            un12,
            un1,
            R_g1,
            R_g2,
            R_gamma1,
            R_gamma2,
            R_N1,
            R_N2,
            R_F1,
            R_F2,
        ) = np.array_split(y, self.split_y)

        P_N = 0.5 * (R_N1 + R_N2)
        P_F = 0.5 * (R_F2 + R_F1)

        R = np.zeros(self.ny, dtype=y.dtype)

        ####################
        # kinematic equation
        ####################
        R[: self.split_y[0]] = (
            qn1
            - qn
            - 0.5
            * dt
            * (self.system.q_dot(tn, qn, un12) + self.system.q_dot(tn1, qn1, un12))
        )

        ########################
        # euations of motion (1)
        ########################
        R[self.split_y[0] : self.split_y[1]] = self.system.M(
            tn,
            qn,
            scipy_matrix=csr_matrix
            # ) @ (un12 - un) - 0.5 * (
            #     dt * self.system.h(tn, qn, un12)
        ) @ (un12 - un) - 0.5 * dt * (
            self.system.h(tn, qn, un12)
            + self.system.W_g(tn, qn) @ R_g1
            + self.system.W_gamma(tn, qn) @ R_gamma1
            + self.system.W_N(tn, qn) @ R_N1
            + self.system.W_F(tn, qn) @ R_F1
        )

        ########################
        # euations of motion (2)
        ########################
        R[self.split_y[1] : self.split_y[2]] = self.system.M(
            tn1,
            qn1,
            scipy_matrix=csr_matrix
            # ) @ (un1 - un12) - 0.5 * (
            #     dt * self.system.h(tn1, qn1, un12)
        ) @ (un1 - un12) - 0.5 * dt * (
            self.system.h(tn1, qn1, un12)
            + self.system.W_g(tn1, qn1) @ R_g2
            + self.system.W_gamma(tn1, qn1) @ R_gamma2
            + self.system.W_N(tn1, qn1) @ R_N2
            + self.system.W_F(tn1, qn1) @ R_F2
        )

        #######################
        # bilateral constraints
        #######################
        R[self.split_y[2] : self.split_y[3]] = self.system.g(tn1, qn1)
        R[self.split_y[3] : self.split_y[4]] = self.system.g_dot(tn1, qn1, un1)

        R[self.split_y[4] : self.split_y[5]] = self.system.gamma(tn1, qn1, un12)
        R[self.split_y[5] : self.split_y[6]] = self.system.gamma(tn1, qn1, un1)

        ###########
        # Signorini
        ###########
        prox_r_N = self.prox_r_N
        g_Nn1 = self.system.g_N(tn1, qn1)
        prox_arg = prox_r_N * g_Nn1 - R_N1
        if update_index:
            self.I_N = prox_arg <= 0.0

        R[self.split_y[6] : self.split_y[7]] = np.where(self.I_N, g_Nn1, R_N1)

        ##################################################
        # mixed Singorini on velocity level and impact law
        ##################################################
        xi_Nn1 = self.system.xi_N(tn1, qn1, un, un1)
        R[self.split_y[7] : self.split_y[8]] = np.where(
            self.I_N,
            P_N + prox_R0_nm(prox_r_N * xi_Nn1 - P_N),
            P_N,
        )
        # R[self.split_y[7] : self.split_y[8]] = np.where(
        #     self.I_N,
        #     R_N2 + prox_R0_nm(prox_r_N * xi_Nn1 - R_N2),
        #     R_N2,
        # )

        ##############################
        # friction and tangent impacts
        ##############################
        prox_r_F = self.prox_r_F
        gamma_Fn1 = self.system.gamma_F(tn1, qn1, un12)
        xi_Fn1 = self.system.xi_F(tn1, qn1, un, un1)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                R[self.split_y[8] + i_F] = R_F1[i_F] + prox_sphere(
                    prox_r_F[i_N] * gamma_Fn1[i_F] - R_F1[i_F],
                    self.system.mu[i_N] * R_N1[i_N],
                )

                R[self.split_y[9] + i_F] = P_F[i_F] + prox_sphere(
                    prox_r_F[i_N] * xi_Fn1[i_F] - P_F[i_F],
                    self.system.mu[i_N] * P_N[i_N],
                )
                # R[self.split_y[9] + i_F] = R_F2[i_F] + prox_sphere(
                #     prox_r_F[i_N] * xi_Fn1[i_F] - R_F2[i_F],
                #     self.system.mu[i_N] * R_N2[i_N],
                # )

        return R

    def F(self, x, P_N1, P_N2, P_F1, P_F2):
        tn = self.tn
        qn = self.qn
        un = self.un
        dt = self.dt
        tn1 = tn + dt

        qn1, un12, un1, P_g1, P_g2, P_gamma1, P_gamma2 = np.array_split(x, self.split_x)

        F = np.zeros_like(x)

        ####################
        # kinematic equation
        ####################
        F[: self.split_x[0]] = (
            qn1
            - qn
            - 0.5
            * dt
            * (self.system.q_dot(tn, qn, un12) + self.system.q_dot(tn1, qn1, un12))
        )

        ######################################################
        # euations of motion (1) without contacts and friction
        ######################################################
        F[self.split_x[0] : self.split_x[1]] = self.system.M(
            tn, qn, scipy_matrix=csr_matrix
        ) @ (un12 - un) - 0.5 * (
            dt * self.system.h(tn, qn, un12)
            + self.system.W_g(tn, qn) @ P_g1
            + self.system.W_gamma(tn, qn) @ P_gamma1
            + self.system.W_N(tn, qn) @ P_N1
            + self.system.W_F(tn, qn) @ P_F1
        )

        ########################
        # euations of motion (2)
        ########################
        F[self.split_x[1] : self.split_x[2]] = self.system.M(
            tn1, qn1, scipy_matrix=csr_matrix
        ) @ (un1 - un12) - 0.5 * (
            dt * self.system.h(tn1, qn1, un12)
            + self.system.W_g(tn1, qn1) @ P_g2
            + self.system.W_gamma(tn1, qn1) @ P_gamma2
            + self.system.W_N(tn1, qn1) @ P_N2
            + self.system.W_F(tn1, qn1) @ P_F2
        )

        #######################
        # bilateral constraints
        #######################
        F[self.split_x[2] : self.split_x[3]] = self.system.g(tn1, qn1)
        F[self.split_x[3] : self.split_x[4]] = self.system.g_dot(tn1, qn1, un1)

        F[self.split_x[4] : self.split_x[5]] = self.system.gamma(tn1, qn1, un12)
        F[self.split_x[5] :] = self.system.gamma(tn1, qn1, un1)

        return F

    def p(self, z):
        P_N1, P_N_bar, P_F1, P_F_bar = np.array_split(z, self.split_z)
        P_N2 = 2 * P_N_bar - P_N1
        P_F2 = 2 * P_F_bar - P_F1

        self.x0 = self.x.copy()

        self.x, self.converged1, self.error1, self.i1, _ = fsolve(
            self.F,
            self.x0,
            jac="2-point",
            eps=1e-6,
            atol=self.atol,
            max_iter=self.max_iter,
            fun_args=(P_N1, P_N2, P_F1, P_F2),
            jac_args=(P_N1, P_N2, P_F1, P_F2),
        )
        assert self.converged1

        qn1, un12, un1, _, _, _, _ = np.array_split(self.x, self.split_x)

        tn1 = self.tn + self.dt

        prox_r_N = self.prox_r_N
        prox_r_F = self.prox_r_F
        mu = self.system.mu

        p = np.zeros_like(z)

        ##############################
        # fixed-point update Signorini
        ##############################
        prox_arg = P_N1 - (prox_r_N / self.dt) * self.system.g_N(tn1, qn1)
        self.I_N = prox_arg >= 0
        # P_N1 = prox_R0_np(prox_arg)  # Gauss-Seidel
        # p[: self.split_z[0]] = P_N1
        p[: self.split_z[0]] = prox_R0_np(prox_arg)  # Jacobi

        ############################################################
        # fixed-point update mixed Signorini and Newton's impact law
        ############################################################
        # P_N_bar = 0.5 * (P_N1 + P_N2) # Gauss-Seidel
        P_N_bar = np.where(
            self.I_N,
            prox_R0_np(P_N_bar - prox_r_N * self.system.xi_N(tn1, qn1, self.un, un1)),
            np.zeros(self.nla_N),
        )
        p[self.split_z[0] : self.split_z[1]] = P_N_bar

        #############################
        # fixed-point update friction
        #############################
        gamma_F = self.system.gamma_F(tn1, qn1, un12)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            if len(i_F):
                p[self.split_z[1] + np.array(i_F)] = prox_sphere(
                    P_F1[i_F] - prox_r_F[i_N] * gamma_F[i_F],
                    mu[i_N] * P_N1[i_N],
                )

        # P_F1 = p[self.split_z[1] : self.split_z[2]]  # Gauss-Seidel

        ###########################################################
        # fixed-point update mixed friction and Newton's impact law
        ###########################################################
        # P_F_bar = 0.5 * (P_F1 + P_F2) # Gauss-Seidel
        xi_F = self.system.xi_F(tn1, qn1, self.un, un1)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            if len(i_F):
                if self.I_N[i_N]:
                    p[self.split_z[2] + np.array(i_F)] = prox_sphere(
                        P_F_bar[i_F] - prox_r_F[i_N] * xi_F[i_F],
                        mu[i_N] * P_N_bar[i_N],
                    )

        return p

    def c(self, x, z):
        tn1 = self.tn + self.dt

        qn, un = self.qn, self.un
        qn1, un12, un1, _, _, _, _ = np.array_split(x, self.split_x)

        P_N1, P_N2, P_F1, P_F2 = np.array_split(z, self.split_z)
        P_N = 0.5 * (P_N1 + P_N2)
        P_F = 0.5 * (P_F2 + P_F1)

        c = np.zeros_like(z)

        ###########
        # Signorini
        ###########
        prox_r_N = self.prox_r_N
        g_Nn1 = self.system.g_N(tn1, qn1)
        prox_arg = prox_r_N * g_Nn1 - P_N1
        # if update_index:
        #     self.I_N = prox_arg <= 0.0
        self.I_N = prox_arg <= 0.0

        c[: self.split_z[0]] = np.where(self.I_N, g_Nn1, P_N1)

        ##################################################
        # mixed Singorini on velocity level and impact law
        ##################################################
        xi_Nn1 = self.system.xi_N(tn1, qn1, un, un1)
        c[self.split_z[0] : self.split_z[1]] = np.where(
            self.I_N,
            P_N + prox_R0_nm(prox_r_N * xi_Nn1 - P_N),
            P_N,
        )

        ##############################
        # friction and tangent impacts
        ##############################
        prox_r_F = self.prox_r_F
        gamma_Fn1 = self.system.gamma_F(tn1, qn1, un12)
        xi_Fn1 = self.system.xi_F(tn1, qn1, un, un1)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                c[self.split_z[1] + i_F] = P_F1[i_F] + prox_sphere(
                    prox_r_F[i_N] * gamma_Fn1[i_F] - P_F1[i_F],
                    self.system.mu[i_N] * P_N1[i_N],
                )

                c[self.split_z[2] + i_F] = P_F[i_F] + prox_sphere(
                    prox_r_F[i_N] * xi_Fn1[i_F] - P_F[i_F],
                    self.system.mu[i_N] * P_N[i_N],
                )

        return c

    def R1(self, y1, update_index=False):
        tn = self.tn
        qn = self.qn
        un = self.un
        dt = self.dt
        tn1 = tn + dt

        qn1, un12, R_g1, R_gamma1, R_N1, R_F1 = np.array_split(y1, self.split_y1)

        R1 = np.zeros(self.ny1, dtype=y1.dtype)

        ####################
        # kinematic equation
        ####################
        R1[: self.split_y1[0]] = (
            qn1
            - qn
            - 0.5
            * dt
            * (self.system.q_dot(tn, qn, un12) + self.system.q_dot(tn1, qn1, un12))
        )

        ########################
        # euations of motion (1)
        ########################
        # R1[self.split_y1[0] : self.split_y1[1]] = self.system.M(
        #     tn,
        #     qn,
        #     scipy_matrix=csr_matrix
        #     # ) @ (un12 - un) - 0.5 * (
        #     #     dt * self.system.h(tn, qn, un12)
        # ) @ (un12 - un) - 0.5 * dt * (
        #     self.system.h(tn, qn, un12)
        #     + self.system.W_g(tn, qn) @ R_g1
        #     + self.system.W_gamma(tn, qn) @ R_gamma1
        #     + self.system.W_N(tn, qn) @ R_N1
        #     + self.system.W_F(tn, qn) @ R_F1
        # )
        R1[self.split_y1[0] : self.split_y1[1]] = self.Mn1 @ (un12 - un) - 0.5 * dt * (
            self.system.h(tn, qn, un12)
            + self.W_gn1 @ R_g1
            + self.W_gamman1 @ R_gamma1
            + self.W_Nn1 @ R_N1
            + self.W_Fn1 @ R_F1
        )

        #######################
        # bilateral constraints
        #######################
        R1[self.split_y1[1] : self.split_y1[2]] = self.system.g(tn1, qn1)
        R1[self.split_y1[2] : self.split_y1[3]] = self.system.gamma(tn1, qn1, un12)

        ###########
        # Signorini
        ###########
        prox_r_N = self.prox_r_N
        g_Nn1 = self.system.g_N(tn1, qn1)
        prox_arg = prox_r_N * g_Nn1 - R_N1
        if update_index:
            self.I_N = prox_arg <= 0.0

        R1[self.split_y1[3] : self.split_y1[4]] = np.where(self.I_N, g_Nn1, R_N1)

        ##############################
        # friction and tangent impacts
        ##############################
        prox_r_F = self.prox_r_F
        mu = self.system.mu
        gamma_F = self.system.gamma_F(tn1, qn1, un12)

        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)
            n_F = len(i_F)
            if n_F > 0:
                R_Ni = R_N1[i_N]
                R_Fi = R_F1[i_F]
                gamma_Fi = gamma_F[i_F]
                arg_F = prox_r_F[i_F] * gamma_Fi - R_Fi
                mui = mu[i_N]
                radius = mui * R_Ni
                norm_arg_F = np.linalg.norm(arg_F)

                if norm_arg_F < radius:
                    R1[self.split_y1[4] + i_F] = gamma_Fi
                else:
                    if norm_arg_F > 0:
                        R1[self.split_y1[4] + i_F] = (
                            R_F1[i_F] + radius * arg_F / norm_arg_F
                        )
                    else:
                        R1[self.split_y1[4] + i_F] = R_F1[i_F] + radius * arg_F
        return R1

    def J1(self, y1, *args, **kwargs):
        tn = self.tn
        qn = self.qn
        un = self.un
        h = self.dt
        h2 = 0.5 * h
        tn1 = tn + h

        qn1, un12, R_g1, R_gamma1, R_N1, R_F1 = np.array_split(y1, self.split_y1)

        ####################
        # kinematic equation
        ####################
        Rq_q = eye(self.nq) - h2 * self.system.q_dot_q(tn1, qn1, un12)
        Rq_u = -h2 * (self.system.B(tn, qn) + self.system.B(tn1, qn1))

        ########################
        # euations of motion (1)
        ########################
        # TODO: Compute generalized force directions in advance and pass them
        #       as function kwargs. Same holds for the mass matrix.
        # M = self.system.M(tn, qn)
        # W_g = self.system.W_g(tn, qn)
        # W_gamma = self.system.W_gamma(tn, qn)
        # W_N = self.system.W_N(tn, qn)
        # W_F = self.system.W_F(tn, qn)
        M = self.Mn1
        W_g = self.W_gn1
        W_gamma = self.W_gamman1
        W_N = self.W_Nn1
        W_F = self.W_Fn1

        Ru_u = M - h2 * self.system.h_u(tn, qn, un12)

        #######################
        # bilateral constraints
        #######################
        Rla_g_q = self.system.g_q(tn1, qn1)
        Rla_gamma_q = self.system.gamma_q(tn1, qn1, un12)
        Rla_gamma_u = self.system.W_gamma(tn1, qn1).T

        ###########
        # Signorini
        ###########
        if np.any(self.I_N):
            # note: csr_matrix is best for row slicing, see
            # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
            g_N_q = self.system.g_N_q(tn1, qn1, scipy_matrix=csr_matrix)

        Rla_N_q = lil_matrix((self.nla_N, self.nq))
        Rla_N_la_N = lil_matrix((self.nla_N, self.nla_N))
        for i in range(self.nla_N):
            if self.I_N[i]:
                Rla_N_q[i] = g_N_q[i]
            else:
                Rla_N_la_N[i, i] = 1.0

        ##############################
        # friction and tangent impacts
        ##############################
        mu = self.system.mu
        prox_r_F = self.prox_r_F
        gamma_F = self.system.gamma_F(tn1, qn1, un12)

        # note: csr_matrix is best for row slicing, see
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
        gamma_F_q = self.system.gamma_F_q(tn1, qn1, un12, scipy_matrix=csr_matrix)

        # note: we use csc_matrix sicne its transpose is a csr_matrix that is best for row slicing, see,
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
        gamma_F_u = self.system.W_F(tn1, qn1, scipy_matrix=csc_matrix).T

        Rla_F_q = lil_matrix((self.nla_F, self.nq))
        Rla_F_u = lil_matrix((self.nla_F, self.nu))
        Rla_F_la_N = lil_matrix((self.nla_F, self.nla_N))
        Rla_F_la_F = lil_matrix((self.nla_F, self.nla_F))

        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)
            n_F = len(i_F)
            if n_F > 0:
                R_Ni = R_N1[i_N]
                R_Fi = R_F1[i_F]
                gamma_Fi = gamma_F[i_F]
                arg_F = prox_r_F[i_F] * gamma_Fi - R_Fi
                mui = mu[i_N]
                radius = mui * R_Ni
                norm_arg_F = np.linalg.norm(arg_F)

                if norm_arg_F < radius:
                    Rla_F_q[i_F] = gamma_F_q[i_F]
                    Rla_F_u[i_F] = gamma_F_u[i_F]
                else:
                    if norm_arg_F > 0:
                        slip_dir = arg_F / norm_arg_F
                        factor = (
                            np.eye(n_F) - np.outer(slip_dir, slip_dir)
                        ) / norm_arg_F
                        Rla_F_q[i_F] = (
                            radius * factor @ diags(prox_r_F[i_F]) @ gamma_F_q[i_F]
                        )
                        Rla_F_u[i_F] = (
                            radius * factor @ diags(prox_r_F[i_F]) @ gamma_F_u[i_F]
                        )
                        Rla_F_la_N[i_F[:, None], i_N] = mui * slip_dir
                        Rla_F_la_F[i_F[:, None], i_F] = np.eye(n_F) - radius * factor
                    else:
                        slip_dir = arg_F
                        Rla_F_q[i_F] = radius * diags(prox_r_F[i_F]) @ gamma_F_q[i_F]
                        Rla_F_u[i_F] = radius * diags(prox_r_F[i_F]) @ gamma_F_u[i_F]
                        Rla_F_la_N[i_F[:, None], i_N] = mui * slip_dir
                        Rla_F_la_F[i_F[:, None], i_F] = (1 - radius) * eye(n_F)

        # fmt: off
        J1 = bmat(
            [
                [Rq_q, Rq_u, None, None, None, None],
                # [None, Ru_u, -0.5 * W_g, -0.5 * W_gamma, -0.5 * W_N, -0.5 * W_F],
                [None, Ru_u, -h2 * W_g, -h2 * W_gamma, -h2 * W_N, -h2 * W_F],
                [Rla_g_q, None, None, None, None, None],
                [Rla_gamma_q, Rla_gamma_u, None, None, None, None],
                [Rla_N_q, None, None, None, Rla_N_la_N, None],
                [Rla_F_q, Rla_F_u, None, None, Rla_F_la_N, Rla_F_la_F],
            ],
            format="csr",
        )
        # fmt: on

        return J1

        J1_num = csr_matrix(approx_fprime(y1, self.R1, method="3-point", eps=1e-6))

        diff = (J1 - J1_num).toarray()
        # diff = diff[:self.split_y1[0]]
        # diff = diff[self.split_y1[0]:self.split_y1[1]]
        # diff = diff[self.split_y1[1]:self.split_y1[2]]
        # diff = diff[self.split_y1[2]:self.split_y1[3]]
        # diff = diff[self.split_y1[3]:self.split_y1[4]]
        # diff = diff[self.split_y1[4] :]
        # diff = diff[self.split_y1[4] :, :self.split_y1[0]]
        # diff = diff[self.split_y1[4] :, self.split_y1[0] : self.split_y1[1]]
        # diff = diff[self.split_y1[4] :, self.split_y1[1] : self.split_y1[2]]
        # diff = diff[self.split_y1[4] :, self.split_y1[2] : self.split_y1[3]]
        # diff = diff[self.split_y1[4] :, self.split_y1[3] : self.split_y1[4]]
        # diff = diff[self.split_y1[4] :, self.split_y1[4] :]
        error = np.linalg.norm(diff)
        if error > 1.0e-6:
            print(f"error J1: {error}")

        return J1_num

    def R2(self, y2, update_index=False):
        tn = self.tn
        un = self.un
        h = self.dt
        h2 = 0.5 * h
        tn1 = tn + h

        qn1 = self.qn1
        un12 = self.un12
        # R_N1 = self.R_N1
        # R_F1 = self.R_F1

        un1, R_g2, R_gamma2, R_N2, R_F2 = np.array_split(y2, self.split_y2)

        # # P_N = 0.5 * (R_N1 + R_N2)
        # # P_F = 0.5 * (R_F1 + R_F2)
        # P_N = h2 * (R_N1 + R_N2)
        # P_F = h2 * (R_F1 + R_F2)

        R2 = np.zeros(self.ny2, dtype=y2.dtype)

        ########################
        # euations of motion (2)
        ########################
        # R2[: self.split_y2[0]] = self.system.M(tn1, qn1, scipy_matrix=csr_matrix) @ (
        #     un1
        #     - un12
        #     # ) - 0.5 * (
        #     #     h * self.system.h(tn1, qn1, un12)
        # ) - h2 * (
        #     self.system.h(tn1, qn1, un12)
        #     + self.system.W_g(tn1, qn1) @ R_g2
        #     + self.system.W_gamma(tn1, qn1) @ R_gamma2
        #     + self.system.W_N(tn1, qn1) @ R_N2
        #     + self.system.W_F(tn1, qn1) @ R_F2
        # )
        R2[: self.split_y2[0]] = self.Mn1 @ (un1 - un12) - h2 * (
            self.system.h(tn1, qn1, un12)
            + self.W_gn1 @ R_g2
            + self.W_gamman1 @ R_gamma2
            + self.W_Nn1 @ R_N2
            + self.W_Fn1 @ R_F2
        )

        #######################
        # bilateral constraints
        #######################
        R2[self.split_y2[0] : self.split_y2[1]] = self.system.g_dot(tn1, qn1, un1)
        R2[self.split_y2[1] : self.split_y2[2]] = self.system.gamma(tn1, qn1, un1)

        ##################################################
        # mixed Singorini on velocity level and impact law
        ##################################################
        prox_r_N = self.prox_r_N
        xi_Nn1 = self.system.xi_N(tn1, qn1, un, un1)
        # prox_arg = prox_r_N * xi_Nn1 - P_N
        prox_arg = prox_r_N * xi_Nn1 - R_N2
        if update_index:
            self.B_N = self.I_N * (prox_arg <= 0)

        R2[self.split_y2[2] : self.split_y2[3]] = np.where(
            self.B_N,
            xi_Nn1,
            # P_N,
            R_N2,
        )

        ##############################
        # friction and tangent impacts
        ##############################
        prox_r_F = self.prox_r_F
        mu = self.system.mu
        xi_F = self.system.xi_F(tn1, qn1, un, un1)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            # i_F = np.array(i_F)
            # if len(i_F) > 0:
            #     R2[self.split_y2[3] + i_F] = P_F[i_F] + prox_sphere(
            #         prox_r_F[i_N] * xi_F[i_F] - P_F[i_F],
            #         self.system.mu[i_N] * P_N[i_N],
            #     )

            i_F = np.array(i_F)
            n_F = len(i_F)
            if n_F > 0:
                # P_Ni = P_N[i_N]
                # P_Fi = P_F[i_F]
                P_Ni = R_N2[i_N]
                P_Fi = R_F2[i_F]
                xi_Fi = xi_F[i_F]
                arg_F = prox_r_F[i_F] * xi_Fi - P_Fi
                mui = mu[i_N]
                radius = mui * P_Ni
                norm_arg_F = np.linalg.norm(arg_F)

                if norm_arg_F < radius:
                    R2[self.split_y2[3] + i_F] = xi_F[i_F]
                else:
                    if norm_arg_F > 0:
                        R2[self.split_y2[3] + i_F] = P_Fi + radius * arg_F / norm_arg_F
                    else:
                        R2[self.split_y2[3] + i_F] = P_Fi + radius * arg_F

        return R2

    def J2(self, y2, *args, **kwargs):
        tn = self.tn
        qn = self.qn
        un = self.un
        h = self.dt
        h2 = 0.5 * h
        tn1 = tn + h

        qn1 = self.qn1
        un12 = self.un12
        P_N1 = self.R_N1
        P_F1 = self.R_F1

        un1, R_g2, R_gamma2, R_N2, R_F2 = np.array_split(y2, self.split_y2)

        # # P_N = 0.5 * (P_N1 + R_N2)
        # # P_F = 0.5 * (P_F1 + R_F2)
        # P_N = h2 * (P_N1 + R_N2)
        # P_F = h2 * (P_F1 + R_F2)

        ########################
        # euations of motion (2)
        ########################
        # M = self.system.M(tn1, qn1)
        # W_g = self.system.W_g(tn1, qn1)
        # W_gamma = self.system.W_gamma(tn1, qn1)
        # W_N = self.system.W_N(tn1, qn1, scipy_matrix=csr_matrix)
        # W_F = self.system.W_F(tn1, qn1)
        M = self.Mn1
        W_g = self.W_gn1
        W_gamma = self.W_gamman1
        W_N = self.W_Nn1
        W_F = self.W_Fn1

        Ru_u = M - h2 * self.system.h_u(tn1, qn1, un12)

        ##################################################
        # mixed Singorini on velocity level and impact law
        ##################################################
        Rla_N_u = lil_matrix((self.nla_N, self.nu))
        Rla_N_la_N = lil_matrix((self.nla_N, self.nla_N))
        for i in range(self.nla_N):
            if self.B_N[i]:
                Rla_N_u[i] = W_N.T[i]
            else:
                # Rla_N_la_N[i, i] = h2
                Rla_N_la_N[i, i] = 1.0

        ##############################
        # friction and tangent impacts
        ##############################
        mu = self.system.mu
        prox_r_F = self.prox_r_F
        xi_F = self.system.xi_F(tn1, qn1, un, un1)
        xi_F_u = W_F.tocsr().T

        Rla_F_u = lil_matrix((self.nla_F, self.nu))
        Rla_F_la_N = lil_matrix((self.nla_F, self.nla_N))
        Rla_F_la_F = lil_matrix((self.nla_F, self.nla_F))

        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)
            n_F = len(i_F)
            if n_F > 0:
                # P_Ni = P_N[i_N]
                # P_Fi = P_F[i_F]
                P_Ni = R_N2[i_N]
                P_Fi = R_F2[i_F]
                xi_Fi = xi_F[i_F]
                arg_F = prox_r_F[i_F] * xi_Fi - P_Fi
                mui = mu[i_N]
                radius = mui * P_Ni
                norm_arg_F = np.linalg.norm(arg_F)

                if norm_arg_F < radius:
                    # print(f"stick")
                    Rla_F_u[i_F] = xi_F_u[i_F]
                else:
                    if norm_arg_F > 0:
                        # print(f"slip ||x|| > 0")
                        slip_dir = arg_F / norm_arg_F
                        factor = (
                            np.eye(n_F) - np.outer(slip_dir, slip_dir)
                        ) / norm_arg_F
                        Rla_F_u[i_F] = (
                            radius * factor @ diags(prox_r_F[i_F]) @ xi_F_u[i_F]
                        )
                        # Rla_F_la_N[i_F[:, None], i_N] = h2 * mui * slip_dir
                        # Rla_F_la_F[i_F[:, None], i_F] = h2 * (
                        #     np.eye(n_F) - radius * factor
                        # )
                        Rla_F_la_N[i_F[:, None], i_N] = mui * slip_dir
                        Rla_F_la_F[i_F[:, None], i_F] = np.eye(n_F) - radius * factor
                    else:
                        # print(f"slip ||x|| = 0")
                        slip_dir = arg_F
                        Rla_F_u[i_F] = radius * diags(prox_r_F[i_F]) @ xi_F_u[i_F]
                        # Rla_F_la_N[i_F[:, None], i_N] = h2 * mui * slip_dir
                        # Rla_F_la_F[i_F[:, None], i_F] = h2 * (1 - radius) * eye(n_F)
                        Rla_F_la_N[i_F[:, None], i_N] = mui * slip_dir
                        Rla_F_la_F[i_F[:, None], i_F] = (1 - radius) * eye(n_F)

        # fmt: off
        J2 = bmat(
            [
                # [Ru_u, -0.5 * W_g, -0.5 * W_gamma, -0.5 * W_N, -0.5 * W_F],
                [Ru_u, -h2 * W_g, -h2 * W_gamma, -h2 * W_N, -h2 * W_F],
                [W_g.T, None, None, None, None],
                [W_gamma.T, None, None, None, None],
                [Rla_N_u, None, None, Rla_N_la_N, None],
                [Rla_F_u, None, None, Rla_F_la_N, Rla_F_la_F],
            ],
            format="csr",
        )
        # fmt: on

        return J2

        J2_num = csr_matrix(approx_fprime(y2, self.R2, method="3-point", eps=1e-6))

        diff = (J2 - J2_num).toarray()
        # diff = diff[:self.split_y2[0]]
        # diff = diff[self.split_y2[0]:self.split_y2[1]]
        # diff = diff[self.split_y2[1]:self.split_y2[2]]
        # diff = diff[self.split_y2[2]:self.split_y2[3]]
        # diff = diff[self.split_y2[3]:]
        # diff = diff[self.split_y2[3]:, :self.split_y2[0]]
        # diff = diff[self.split_y2[3]:, self.split_y2[0] : self.split_y2[1]]
        # diff = diff[self.split_y2[3]:, self.split_y2[1] : self.split_y2[2]]
        # diff = diff[self.split_y2[3]:, self.split_y2[2] : self.split_y2[3]]
        # diff = diff[self.split_y2[3]:, self.split_y2[3] :]
        error = np.linalg.norm(diff)
        if error > 1.0e-6:
            print(f"error J2: {error}")

        return J2_num

    def F1(self, x1, P_N1, P_F1):
        tn = self.tn
        qn = self.qn
        un = self.un
        dt = self.dt
        tn1 = tn + dt

        qn1, un12, P_g1, P_gamma1 = np.array_split(x1, self.split_x1)

        F1 = np.zeros_like(x1)

        ####################
        # kinematic equation
        ####################
        F1[: self.split_x1[0]] = (
            qn1
            - qn
            - 0.5
            * dt
            * (self.system.q_dot(tn, qn, un12) + self.system.q_dot(tn1, qn1, un12))
        )

        ######################################################
        # euations of motion (1) without contacts and friction
        ######################################################
        F1[self.split_x1[0] : self.split_x1[1]] = self.system.M(
            tn, qn, scipy_matrix=csr_matrix
        ) @ (un12 - un) - 0.5 * (
            dt * self.system.h(tn, qn, un12)
            + self.system.W_g(tn, qn) @ P_g1
            + self.system.W_gamma(tn, qn) @ P_gamma1
            + self.system.W_N(tn, qn) @ P_N1
            + self.system.W_F(tn, qn) @ P_F1
        )

        #######################
        # bilateral constraints
        #######################
        F1[self.split_x1[1] : self.split_x1[2]] = self.system.g(tn1, qn1)
        F1[self.split_x1[2] :] = self.system.gamma(tn1, qn1, un12)

        return F1

    def p1(self, z1):
        P_N1 = z1[: self.nla_N]
        P_F1 = z1[self.nla_N :]

        self.x10 = self.x1.copy()

        self.x1, converged, error, i, _ = fsolve(
            self.F1,
            self.x10,
            jac="2-point",
            eps=1e-6,
            atol=self.atol,
            max_iter=self.max_iter,
            fun_args=(P_N1, P_F1),
            jac_args=(P_N1, P_F1),
        )
        assert converged

        # # TODO: Check convergence of bilaeral constraints
        # self.x1 -= spsolve(
        #     approx_fprime(
        #         self.x1.copy(),
        #         lambda x, P_N1=P_N1, P_F1=P_F1: self.F1(x, P_N1, P_F1),
        #         method="2-point",
        #         eps=1e-6,
        #     ),
        #     self.F1(self.x1.copy(), P_N1, P_F1),
        # )

        qn1, un12, _, _ = np.array_split(self.x1, self.split_x1)

        tn1 = self.tn + self.dt

        prox_r_N = self.prox_r_N
        prox_r_F = self.prox_r_F
        mu = self.system.mu

        p1 = np.zeros_like(z1)

        # fixed-point update normal direction
        prox_arg = P_N1 - (prox_r_N / self.dt) * self.system.g_N(tn1, qn1)
        self.I_N = prox_arg >= 0
        P_N1 = prox_R0_np(prox_arg)  # Gauss-Seidel
        p1[: self.nla_N] = P_N1
        # z1[: self.nla_N] = prox_R0_np(prox_arg) # Jacobi

        # fixed-point update friction
        gamma_F = self.system.gamma_F(tn1, qn1, un12)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            if len(i_F):
                p1[self.nla_N + np.array(i_F)] = prox_sphere(
                    P_F1[i_F] - prox_r_F[i_N] * gamma_F[i_F],
                    mu[i_N] * P_N1[i_N],
                )

        return p1

    def F2(self, x2, P_N2, P_F2):
        tn = self.tn
        dt = self.dt
        tn1 = tn + dt

        qn1, un12, _, _ = np.array_split(self.x1, self.split_x1)
        (
            un1,
            P_g2,
            P_gamma2,
        ) = np.array_split(x2, self.split_x2)

        F2 = np.zeros_like(x2)

        ########################
        # euations of motion (2)
        ########################
        F2[: self.split_x2[0]] = self.system.M(tn1, qn1, scipy_matrix=csr_matrix) @ (
            un1 - un12
        ) - 0.5 * (
            dt * self.system.h(tn1, qn1, un12)
            + self.system.W_g(tn1, qn1) @ P_g2
            + self.system.W_gamma(tn1, qn1) @ P_gamma2
            + self.system.W_N(tn1, qn1) @ P_N2
            + self.system.W_F(tn1, qn1) @ P_F2
        )

        #######################
        # bilateral constraints
        #######################
        F2[self.split_x2[0] : self.split_x2[1]] = self.system.g_dot(tn1, qn1, un1)

        F2[self.split_x2[1] :] = self.system.gamma(tn1, qn1, un1)

        return F2

    def p2(self, z1, z2, lu_A, b, W_N, W_F):
        P_N1 = z1[: self.nla_N]
        P_F1 = z1[self.nla_N :]
        P_N_bar = z2[: self.nla_N]
        P_F_bar = z2[self.nla_N :]
        P_N2 = 2 * P_N_bar - P_N1
        P_F2 = 2 * P_F_bar - P_F1

        self.x20 = self.x2.copy()

        # self.x2, converged, error, i, _ = fsolve(
        #     self.F2,
        #     self.x20,
        #     jac="2-point",
        #     eps=1e-6,
        #     fun_args=(P_N2, P_F2),
        #     jac_args=(P_N2, P_F2),
        # )
        # assert converged
        bb = b.copy()
        bb[: self.nu] += 0.5 * (W_N @ P_N2 + W_F @ P_F2)
        self.x2 = lu_A.solve(bb)

        qn1, un12, _, _ = np.array_split(self.x1, self.split_x1)
        un1, _, _ = np.array_split(self.x2, self.split_x2)

        tn1 = self.tn + self.dt

        prox_r_N = self.prox_r_N
        prox_r_F = self.prox_r_F
        mu = self.system.mu

        p2 = np.zeros_like(z2)

        ##################################################
        # mixed Singorini on velocity level and impact law
        ##################################################
        P_N_bar = np.where(
            self.I_N,
            prox_R0_np(P_N_bar - prox_r_N * self.system.xi_N(tn1, qn1, self.un, un1)),
            np.zeros(self.nla_N),
        )
        p2[: self.nla_N] = P_N_bar

        ##############################
        # friction and tangent impacts
        ##############################
        xi_F = self.system.xi_F(tn1, qn1, self.un, un1)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            if len(i_F):
                if self.I_N[i_N]:
                    p2[self.nla_N + np.array(i_F)] = prox_sphere(
                        P_F_bar[i_F] - prox_r_F[i_N] * xi_F[i_F],
                        mu[i_N] * P_N_bar[i_N],
                    )

        return p2

    def solve(self):
        # lists storing output variables
        q = [self.qn]
        u = [self.un]
        P_g = [self.la_gn]
        P_gamma = [self.la_gamman]
        P_N = [self.la_Nn]
        P_F = [self.la_Fn]

        pbar = tqdm(self.t[:-1])
        for n in pbar:
            # # only compute optimized proxparameters once per time step
            # self.prox_r_N = self.system.prox_r_N(self.tn, self.qn)
            # self.prox_r_F = self.system.prox_r_F(self.tn, self.qn)
            # print(f"prox_r_N: {self.prox_r_N}")
            # print(f"prox_r_F: {self.prox_r_F}")

            # TODO: We need a possibility to set prox parameters on system level!

            # ########################
            # # rotating bouncing ball
            # ########################
            # self.prox_r_N = np.ones(self.nla_N) * 0.5
            # self.prox_r_F = np.ones(self.nla_F) * 0.5

            # ##########
            # # tippetop
            # ##########
            # self.prox_r_N = np.ones(self.nla_N) * 0.001
            # self.prox_r_F = np.ones(self.nla_F) * 0.001

            ##############
            # slider crank
            ##############
            self.prox_r_N = np.ones(self.nla_N) * 0.001
            self.prox_r_F = np.ones(self.nla_F) * 0.001

            tn1 = self.tn + self.dt

            if self.method == "Newton_decoupled":
                y1, converged1, error1, i1, _ = fsolve(
                    self.R1,
                    self.y1n,
                    jac=self.J1,
                    atol=self.atol,
                    max_iter=self.max_iter,
                    fun_args=(True,),
                    jac_args=(False,),
                )

                (
                    self.qn1,
                    self.un12,
                    self.R_g1,
                    self.R_gamma1,
                    self.R_N1,
                    self.R_F1,
                ) = np.array_split(y1, self.split_y1)

                # compute constant quantities for next stage
                self.Mn1 = self.system.M(tn1, self.qn1, scipy_matrix=csr_matrix)
                self.W_gn1 = self.system.W_g(tn1, self.qn1, scipy_matrix=csr_matrix)
                self.W_gamman1 = self.system.W_gamma(
                    tn1, self.qn1, scipy_matrix=csr_matrix
                )
                self.W_Nn1 = self.system.W_N(tn1, self.qn1, scipy_matrix=csr_matrix)
                self.W_Fn1 = self.system.W_F(tn1, self.qn1, scipy_matrix=csr_matrix)

                y2, converged2, error2, i2, _ = fsolve(
                    self.R2,
                    self.y2n,
                    jac=self.J2,
                    atol=self.atol,
                    max_iter=self.max_iter,
                    fun_args=(True,),
                    jac_args=(False,),
                )

                un1, R_g2, R_gamma2, R_N2, R_F2 = np.array_split(y2, self.split_y2)

                converged = converged1 and converged2
                error = error1 + error2
                i = i1 + i2

                P_gn1 = 0.5 * (self.R_g1 + R_g2)
                P_gamman1 = 0.5 * (self.R_gamma1 + R_gamma2)
                P_Nn1 = 0.5 * (self.R_N1 + R_N2)
                P_Fn1 = 0.5 * (self.R_F1 + R_F2)

                # qn1 = self.qn1
                qn1, un1 = self.system.step_callback(tn1, self.qn1, un1)

            elif self.method == "Newton_full":
                y, converged, error, i, _ = fsolve(
                    self.R,
                    self.yn,
                    # jac="2-point",
                    jac="3-point",  # TODO: keep this, otherwise sinuglairites arise
                    eps=1.0e-6,
                    atol=self.atol,
                    max_iter=self.max_iter,
                    fun_args=(True,),
                    jac_args=(False,),
                )

                (
                    qn1,
                    un12,
                    un1,
                    R_g1,
                    R_g2,
                    R_gamma1,
                    R_gamma2,
                    R_N1,
                    R_N2,
                    R_F1,
                    R_F2,
                ) = np.array_split(y, self.split_y)

                P_gn1 = 0.5 * (R_g1 + R_g2)
                P_gamman1 = 0.5 * (R_gamma1 + R_gamma2)
                P_Nn1 = 0.5 * (R_N1 + R_N2)
                P_Fn1 = 0.5 * (R_F1 + R_F2)

                qn1, un1 = self.system.step_callback(tn1, qn1, un1)

                i1, i2 = i, "-"
                error1, error2 = error, np.nan

            elif self.method == "fixed_point":
                z10 = self.z1n.copy()
                for i1 in range(self.fix_point_max_iter):
                    z1 = self.p1(z10)

                    # convergence percussions
                    # error1 = self.fix_point_error_function(z1 - z10)

                    # convergence positions and velocities
                    error1 = self.fix_point_error_function(
                        self.x1[: self.nq + self.nu] - self.x10[: self.nq + self.nu]
                    )

                    converged1 = error1 < self.fix_point_tol
                    if converged1:
                        break
                    z10 = z1

                qn1, un12, R_g1, R_gamma1 = np.array_split(self.x1, self.split_x1)

                #################
                # second stage
                #################

                # get quantities from system
                Mn = self.system.M(tn1, qn1)
                h = self.system.h(tn1, qn1, un12)
                W_g = self.system.W_g(tn1, qn1)
                W_gamma = self.system.W_gamma(tn1, qn1)
                chi_g = self.system.g_dot(tn1, qn1, np.zeros_like(un12))
                chi_gamma = self.system.gamma(tn1, qn1, np.zeros_like(un12))
                # note: we use csc_matrix for efficient column slicing later,
                # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html#scipy.sparse.csc_array
                W_N = self.system.W_N(tn1, qn1, scipy_matrix=csc_matrix)
                W_F = self.system.W_F(tn1, qn1, scipy_matrix=csc_matrix)

                A = bmat(
                    [
                        [Mn, -0.5 * W_g, -0.5 * W_gamma],
                        [-W_g.T, None, None],
                        [-W_gamma.T, None, None],
                    ],
                    format="csc",
                )

                lu_A = splu(A)

                # initial right hand side
                b = np.concatenate(
                    (
                        Mn @ un12 + 0.5 * self.dt * h,
                        chi_g,
                        chi_gamma,
                    )
                )

                z20 = self.z2n.copy()
                for i2 in range(self.fix_point_max_iter):
                    # z2 = self.p2(z1, z20)
                    z2 = self.p2(z1, z20, lu_A, b, W_N, W_F)

                    # convergence percussions
                    # error2 = self.fix_point_error_function(z2 - z20)

                    # convergence positions and velocities
                    error2 = self.fix_point_error_function(
                        self.x2[: self.nu] - self.x20[: self.nu]
                    )

                    converged2 = error2 < self.fix_point_tol
                    if converged2:
                        break
                    z20 = z2

                converged = converged1 and converged2
                error = error1 + error2
                i = i1 + i2
                print(f"i: {i}, i1: {i1}, i2: {i2}")

                (
                    un1,
                    R_g2,
                    R_gamma2,
                ) = np.array_split(self.x2, self.split_x2)

                P_Nn1 = z2[: self.nla_N]
                P_Fn1 = z2[self.nla_N :]

                P_gn1 = 0.5 * (R_g1 + R_g2)
                P_gamman1 = 0.5 * (R_gamma1 + R_gamma2)

                qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            elif self.method == "fixed_point_nonlinear_full":
                #######################
                # fixed-point iteration
                #######################
                z0 = self.zn.copy()
                for i2 in range(self.fix_point_max_iter):
                    z = self.p(z0)

                    # convergence percussions
                    # error2 = self.fix_point_error_function(z - z0)

                    # convergence positions and both velocities
                    error2 = self.fix_point_error_function(
                        self.x[: self.split_x[2]] - self.x0[: self.split_x[2]]
                    )

                    converged2 = error2 < self.fix_point_tol
                    if converged2:
                        break
                    z0 = z

                qn1, un12, un1, R_g1, R_g2, R_gamma1, R_gamma2 = np.array_split(
                    self.x, self.split_x
                )
                _, P_Nn1, _, P_Fn1 = np.array_split(z, self.split_z)

                P_gn1 = 0.5 * (R_g1 + R_g2)
                P_gamman1 = 0.5 * (R_gamma1 + R_gamma2)

                i1 = self.i1
                error1 = self.error1
                converged = self.converged1 and converged2
                i = i1

                ##########################
                # constrained optimization
                ##########################
                from scipy.optimize import minimize, NonlinearConstraint

                def fun(y):
                    x = y[: self.nx]
                    z = y[self.nx :]
                    F = self.F(x, *np.array_split(z, self.split_z))
                    return F @ F

                def fun_c(y):
                    x = y[: self.nx]
                    z = y[self.nx :]
                    return self.c(x, z)

                lb = ub = np.zeros(2 * self.nla_N + 2 * self.nla_F)
                constraints = NonlinearConstraint(fun_c, lb, ub)

                y0 = self.yn
                sol = minimize(fun, y0, method="SLSQP", constraints=[constraints])
                converged = sol.success
                y = sol.x
                x = y[: self.nx]
                z = y[self.nx :]
                i1 = sol.nit
                i2 = np.nan
                error1 = sol.fun
                error2 = np.nan
            else:
                raise NotImplementedError

            pbar.set_description(
                f"t: {tn1:0.2e}; iter: {i1},{i2}; error: {error1:.3e}, {error2:.3e}"
            )
            if not converged:
                if self.continue_with_unconverged:
                    print(
                        f"step {n:.0f} is not converged after i1: {i1}, i2: {i2} iterations with error:  err1 = {error1:.3e}, err2 = {error2:.3e}"
                    )
                else:
                    raise RuntimeError(
                        f"step {n:.0f} is not converged after i1: {i1}, i2: {i2} iterations with error:  err1 = {error1:.3e}, err2 = {error2:.3e}"
                    )

            q.append(qn1)
            u.append(un1)
            P_g.append(P_gn1)
            P_gamma.append(P_gamman1)
            P_N.append(P_Nn1)
            P_F.append(P_Fn1)
            # q.append(qn1.copy())
            # u.append(un1.copy())
            # P_g.append(P_gn1.copy())
            # P_gamma.append(P_gamman1.copy())
            # P_N.append(P_Nn1.copy())
            # P_F.append(P_Fn1.copy())

            # update local variables for accepted time step
            if self.method == "Newton_decoupled":
                self.y1n = y1.copy()
                self.y2n = y2.copy()
                self.qn = qn1.copy()
                self.un = un1.copy()
            elif self.method == "Newton_full":
                self.yn = y.copy()
            elif self.method == "fixed_point":
                self.z1n = z1.copy()
                self.z2n = z2.copy()
                self.qn = qn1.copy()
                self.un = un1.copy()
            elif self.method == "fixed_point_nonlinear_full":
                self.zn = z.copy()
                self.qn = qn1.copy()
                self.un = un1.copy()
            else:
                raise NotImplementedError

        return Solution(
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
            # la_g=np.array(P_g) / self.dt,
            # la_gamma=np.array(P_gamma) / self.dt,
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
        )
