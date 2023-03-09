import numpy as np
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse import csr_matrix, csc_matrix, bmat, eye
from tqdm import tqdm

from cardillo.math.prox import prox_R0_nm, prox_R0_np, prox_sphere
from cardillo.math import fsolve, approx_fprime
from cardillo.solver import Solution, consistent_initial_conditions

import warnings


class Rattle:
    def __init__(
        self,
        system,
        t1,
        dt,
        atol=1e-8,
        max_iter=50,
        fix_point_tol=1e-6,
        fix_point_max_iter=500,
        error_function=lambda x: np.max(np.abs(x)),
        method="Newton_decoupled",
        # method="Newton_full",
        # method="fixed_point",
    ):
        """
        Nonsmooth extension of RATTLE.
        """
        self.system = system
        self.method = method

        self.fix_point_error_function = error_function
        self.fix_point_tol = fix_point_tol
        self.fix_point_max_iter = fix_point_max_iter

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

        # full coupled Newton
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

        # decoupled Newton
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

        # decouple fixed point iteration
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

    def R(self, y, update_index=False):
        tn = self.tn
        dt = self.dt
        tn1 = tn + dt
        qn, _, un, _, _, _, _, _, _, _, _ = np.array_split(self.yn, self.split_y)
        (
            qn1,
            un12,
            un1,
            P_g1,
            P_g2,
            P_gamma1,
            P_gamma2,
            P_N1,
            P_N2,
            P_F1,
            P_F2,
        ) = np.array_split(y, self.split_y)

        P_N = 0.5 * (P_N1 + P_N2)
        P_F = 0.5 * (P_F2 + P_F1)

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
        R[self.split_y[1] : self.split_y[2]] = self.system.M(
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
        R[self.split_y[2] : self.split_y[3]] = self.system.g(tn1, qn1)
        R[self.split_y[3] : self.split_y[4]] = self.system.g_dot(tn1, qn1, un1)

        R[self.split_y[4] : self.split_y[5]] = self.system.gamma(tn1, qn1, un12)
        R[self.split_y[5] : self.split_y[6]] = self.system.gamma(tn1, qn1, un1)

        ###########
        # Signorini
        ###########
        prox_r_N = self.system.prox_r_N(tn1, qn1)
        g_Nn1 = self.system.g_N(tn1, qn1)
        prox_arg = prox_r_N * g_Nn1 - P_N1
        if update_index:
            self.I_N = prox_arg <= 0.0

        R[self.split_y[6] : self.split_y[7]] = np.where(self.I_N, g_Nn1, P_N1)

        ##################################################
        # mixed Singorini on velocity level and impact law
        ##################################################
        xi_Nn1 = self.system.xi_N(tn1, qn1, un, un1)
        R[self.split_y[7] : self.split_y[8]] = np.where(
            self.I_N,
            P_N + prox_R0_nm(prox_r_N * xi_Nn1 - P_N),
            P_N,
        )

        ##############################
        # friction and tangent impacts
        ##############################
        prox_r_F = self.system.prox_r_F(tn1, qn1)
        gamma_Fn1 = self.system.gamma_F(tn1, qn1, un12)
        xi_Fn1 = self.system.xi_F(tn1, qn1, un, un1)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                R[self.split_y[8] + i_F] = P_F1[i_F] + prox_sphere(
                    prox_r_F[i_N] * gamma_Fn1[i_F] - P_F1[i_F],
                    self.system.mu[i_N] * P_N1[i_N],
                )

                R[self.split_y[9] + i_F] = P_F[i_F] + prox_sphere(
                    prox_r_F[i_N] * xi_Fn1[i_F] - P_F[i_F],
                    self.system.mu[i_N] * P_N[i_N],
                )

        return R

    def R1(self, y1, update_index=False):
        tn = self.tn
        qn = self.qn
        un = self.un
        h = self.dt
        tn1 = tn + h

        qn1, un12, P_g1, P_gamma1, P_N1, P_F1 = np.array_split(y1, self.split_y1)

        R = np.zeros(self.ny1, dtype=y1.dtype)

        ####################
        # kinematic equation
        ####################
        R[: self.split_y1[0]] = (
            qn1
            - qn
            - 0.5
            * h
            * (self.system.q_dot(tn, qn, un12) + self.system.q_dot(tn1, qn1, un12))
        )

        ########################
        # euations of motion (1)
        ########################
        R[self.split_y1[0] : self.split_y1[1]] = self.system.M(
            tn, qn, scipy_matrix=csr_matrix
        ) @ (un12 - un) - 0.5 * (
            h * self.system.h(tn, qn, un12)
            + self.system.W_g(tn, qn) @ P_g1
            + self.system.W_gamma(tn, qn) @ P_gamma1
            + self.system.W_N(tn, qn) @ P_N1
            + self.system.W_F(tn, qn) @ P_F1
        )

        #######################
        # bilateral constraints
        #######################
        R[self.split_y1[1] : self.split_y1[2]] = self.system.g(tn1, qn1)
        R[self.split_y1[2] : self.split_y1[3]] = self.system.gamma(tn1, qn1, un12)

        ###########
        # Signorini
        ###########
        prox_r_N = self.system.prox_r_N(tn1, qn1)
        g_Nn1 = self.system.g_N(tn1, qn1)
        prox_arg = prox_r_N * g_Nn1 - P_N1
        if update_index:
            self.I_N = prox_arg <= 0.0

        R[self.split_y1[3] : self.split_y1[4]] = np.where(self.I_N, g_Nn1, P_N1)

        ##############################
        # friction and tangent impacts
        ##############################
        prox_r_F = self.system.prox_r_F(tn1, qn1)
        gamma_Fn1 = self.system.gamma_F(tn1, qn1, un12)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                R[self.split_y1[4] + i_F] = P_F1[i_F] + prox_sphere(
                    prox_r_F[i_N] * gamma_Fn1[i_F] - P_F1[i_F],
                    self.system.mu[i_N] * P_N1[i_N],
                )

        return R

    def R2(self, y2):
        tn = self.tn
        un = self.un
        h = self.dt
        tn1 = tn + h

        qn1 = self.qn1
        un12 = self.un12
        P_N1 = self.P_N1
        P_F1 = self.P_F1

        un1, P_g2, P_gamma2, P_N2, P_F2 = np.array_split(y2, self.split_y2)

        P_N = 0.5 * (P_N1 + P_N2)
        P_F = 0.5 * (P_F1 + P_F2)

        R = np.zeros(self.ny2, dtype=y2.dtype)

        ########################
        # euations of motion (2)
        ########################
        R[: self.split_y2[0]] = self.system.M(tn1, qn1, scipy_matrix=csr_matrix) @ (
            un1 - un12
        ) - 0.5 * (
            h * self.system.h(tn1, qn1, un12)
            + self.system.W_g(tn1, qn1) @ P_g2
            + self.system.W_gamma(tn1, qn1) @ P_gamma2
            + self.system.W_N(tn1, qn1) @ P_N2
            + self.system.W_F(tn1, qn1) @ P_F2
        )

        #######################
        # bilateral constraints
        #######################
        R[self.split_y2[0] : self.split_y2[1]] = self.system.g_dot(tn1, qn1, un1)
        R[self.split_y2[1] : self.split_y2[2]] = self.system.gamma(tn1, qn1, un1)

        ##################################################
        # mixed Singorini on velocity level and impact law
        ##################################################
        prox_r_N = self.system.prox_r_N(tn1, qn1)
        xi_Nn1 = self.system.xi_N(tn1, qn1, un, un1)
        R[self.split_y2[2] : self.split_y2[3]] = np.where(
            self.I_N,
            P_N + prox_R0_nm(prox_r_N * xi_Nn1 - P_N),
            P_N,
        )

        ##############################
        # friction and tangent impacts
        ##############################
        prox_r_F = self.system.prox_r_F(tn1, qn1)
        xi_Fn1 = self.system.xi_F(tn1, qn1, un, un1)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                R[self.split_y2[3] + i_F] = P_F[i_F] + prox_sphere(
                    prox_r_F[i_N] * xi_Fn1[i_F] - P_F[i_F],
                    self.system.mu[i_N] * P_N[i_N],
                )

        return R

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

        prox_r_N = self.system.prox_r_N(tn1, qn1)
        prox_r_F = self.system.prox_r_F(tn1, qn1)
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

        prox_r_N = self.system.prox_r_N(tn1, qn1)
        prox_r_F = self.system.prox_r_F(tn1, qn1)
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
        P_g = [self.dt * self.la_gn]
        P_gamma = [self.dt * self.la_gamman]
        P_N = [self.dt * self.la_Nn]
        P_F = [self.dt * self.la_Fn]

        pbar = tqdm(self.t[:-1])
        for _ in pbar:
            tn1 = self.tn + self.dt
            if self.method == "Newton_decoupled":
                y1, converged1, error1, i1, _ = fsolve(
                    self.R1,
                    self.y1n,
                    jac="3-point",  # TODO: keep this, otherwise sinuglairites arise
                    eps=1.0e-6,
                    atol=self.atol,
                    fun_args=(True,),
                    jac_args=(False,),
                )

                (
                    self.qn1,
                    self.un12,
                    self.P_g1,
                    self.P_gamma1,
                    self.P_N1,
                    self.P_F1,
                ) = np.array_split(y1, self.split_y1)

                y2, converged2, error2, i2, _ = fsolve(
                    self.R2,
                    self.y2n,
                    jac="3-point",  # TODO: keep this, otherwise sinuglairites arise
                    eps=1.0e-6,
                    atol=self.atol,
                )

                un1, P_g2, P_gamma2, P_N2, P_F2 = np.array_split(y2, self.split_y2)

                converged = converged1 and converged2
                error = error1 + error2
                i = i1 + i2

                P_gn1 = 0.5 * (self.P_g1 + P_g2)
                P_gamman1 = 0.5 * (self.P_gamma1 + P_gamma2)
                P_N_bar = 0.5 * (self.P_N1 + P_N2)
                P_F_bar = 0.5 * (self.P_F1 + P_F2)

                qn1, un1 = self.system.step_callback(tn1, self.qn1, un1)

            elif self.method == "Newton_full":
                y, converged, error, i, _ = fsolve(
                    self.R,
                    self.yn,
                    jac="3-point",  # TODO: keep this, otherwise sinuglairites arise
                    eps=1.0e-6,
                    atol=self.atol,
                    fun_args=(True,),
                    jac_args=(False,),
                )

                (
                    qn1,
                    un12,
                    un1,
                    P_g1,
                    P_g2,
                    P_gamma1,
                    P_gamma2,
                    P_N1,
                    P_N2,
                    P_F1,
                    P_F2,
                ) = np.array_split(y, self.split_y)

                P_gn1 = 0.5 * (P_g1 + P_g2)
                P_gamman1 = 0.5 * (P_gamma1 + P_gamma2)
                P_N_bar = 0.5 * (P_N1 + P_N2)
                P_F_bar = 0.5 * (P_F1 + P_F2)

                qn1, un1 = self.system.step_callback(tn1, qn1, un1)

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

                qn1, un12, P_g1, P_gamma1 = np.array_split(self.x1, self.split_x1)

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
                    P_g2,
                    P_gamma2,
                ) = np.array_split(self.x2, self.split_x2)

                P_N_bar = z2[: self.nla_N]
                P_F_bar = z2[self.nla_N :]

                P_gn1 = 0.5 * (P_g1 + P_g2)
                P_gamman1 = 0.5 * (P_gamma1 + P_gamma2)

                qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            else:
                raise NotImplementedError

            pbar.set_description(f"t: {tn1:0.2e}; step: {i+1}; error: {error:.3e}")
            if not converged:
                # raise RuntimeError(
                print(
                    f"step is not converged after {i+1} iterations with error: {error:.5e}"
                )

            q.append(qn1.copy())
            u.append(un1.copy())
            P_g.append(P_gn1.copy())
            P_gamma.append(P_gamman1.copy())
            P_N.append(P_N_bar.copy())
            P_F.append(P_F_bar.copy())

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
            else:
                raise NotImplementedError

        return Solution(
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
            la_g=np.array(P_g) / self.dt,
            la_gamma=np.array(P_gamma) / self.dt,
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
        )
