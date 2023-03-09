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
        # method="Newton_decoupled",
        # method="Newton_full",
        # method="fixed point decoupled",
        method="fixed point full",
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

        # fully coupled scheme
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

        self.split_x = np.array(
            [
                self.nq,
                self.nq + self.nu,
                self.nq + 2 * self.nu,
                self.nq + 2 * self.nu + self.nla_g,
                self.nq + 2 * (self.nu + self.nla_g),
                self.nq + 2 * (self.nu + self.nla_g) + self.nla_gamma,
            ],
            dtype=int,
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
            self.I_Nn1 = prox_arg <= 0.0

        R[self.split_y[6] : self.split_y[7]] = np.where(self.I_Nn1, g_Nn1, P_N1)

        ##################################################
        # mixed Singorini on velocity level and impact law
        ##################################################
        xi_Nn1 = self.system.xi_N(tn1, qn1, un, un1)
        R[self.split_y[7] : self.split_y[8]] = np.where(
            self.I_Nn1,
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
            self.I_Nn1 = prox_arg <= 0.0

        R[self.split_y1[3] : self.split_y1[4]] = np.where(self.I_Nn1, g_Nn1, P_N1)

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
            self.I_Nn1,
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

    def f1(self, x, P_N1, P_F1):
        tn = self.tn
        qn = self.qn
        un = self.un
        h = self.dt
        tn1 = tn + h

        qn1 = x[: self.nq]
        un12 = x[self.nq : self.nq + self.nu]
        P_g1 = x[self.nq + self.nu : self.nq + self.nu + self.nla_g]

        nx = self.nq + self.nu + self.nla_g
        R = np.zeros(nx, dtype=x.dtype)

        ####################
        # kinematic equation
        ####################
        R[: self.nq] = (
            qn1
            - qn
            - 0.5
            * h
            * (self.system.q_dot(tn, qn, un12) + self.system.q_dot(tn1, qn1, un12))
        )

        ######################################################
        # euations of motion (1) without contacts and friction
        ######################################################
        R[self.nq : self.nq + self.nu] = self.system.M(
            tn, qn, scipy_matrix=csr_matrix
        ) @ (un12 - un) - 0.5 * (
            h * self.system.h(tn, qn, un12)
            + self.system.W_g(tn, qn) @ P_g1
            + self.system.W_N(tn, qn) @ P_N1
            + self.system.W_F(tn, qn) @ P_F1
        )

        #######################
        # bilateral constraints
        #######################
        R[self.nq + self.nu : self.nq + self.nu + self.nla_g] = self.system.g(tn1, qn1)

        return R

    def p1(self, z):
        P_N1 = z[: self.nla_N]
        P_F1 = z[self.nla_N :]

        self.x, converged, error, i, _ = fsolve(
            self.f1,
            self.x.copy(),
            jac="2-point",
            eps=1e-6,
            fun_args=(P_N1.copy(), P_F1.copy()),
            jac_args=(P_N1.copy(), P_F1.copy()),
        )
        assert converged

        qn1 = self.x[: self.nq]
        un12 = self.x[self.nq : self.nq + self.nu]
        P_g1 = self.x[self.nq + self.nu : self.nq + self.nu + self.nla_g]
        P_gamma1 = self.x[self.nq + self.nu + self.nla_g :]

        tn1 = self.tn + self.dt

        prox_r_N = self.system.prox_r_N(tn1, qn1)
        prox_r_F = self.system.prox_r_F(tn1, qn1)
        mu = self.system.mu

        R = np.zeros(self.nla_N + self.nla_F)

        # fixed-point update normal direction
        prox_r_N /= self.dt
        prox_arg = P_N1 - prox_r_N * self.system.g_N(tn1, qn1)
        self.I_Nn1 = prox_arg >= 0
        R[: self.nla_N] = prox_R0_np(prox_arg)

        # fixed-point update friction
        gamma_F = self.system.gamma_F(tn1, qn1, un12)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            if len(i_F):
                R[self.nla_N + np.array(i_F)] = prox_sphere(
                    P_F1[i_F] - prox_r_F[i_N] * gamma_F[i_F],
                    mu[i_N] * P_N1[i_N],
                )

        return R

    def f(self, x, P_N1, P_F1, P_N2, P_F2):
        tn = self.tn
        dt = self.dt
        tn1 = tn + dt
        qn = self.qn
        un = self.un
        (
            qn1,
            un12,
            un1,
            P_g1,
            P_g2,
            P_gamma1,
            P_gamma2,
        ) = np.array_split(x, self.split_x)

        R = np.zeros_like(x)

        ####################
        # kinematic equation
        ####################
        R[: self.split_x[0]] = (
            qn1
            - qn
            - 0.5
            * dt
            * (self.system.q_dot(tn, qn, un12) + self.system.q_dot(tn1, qn1, un12))
        )

        ########################
        # euations of motion (1)
        ########################
        R[self.split_x[0] : self.split_x[1]] = self.system.M(
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
        R[self.split_x[1] : self.split_x[2]] = self.system.M(
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
        R[self.split_x[2] : self.split_x[3]] = self.system.g(tn1, qn1)
        R[self.split_x[3] : self.split_x[4]] = self.system.g_dot(tn1, qn1, un1)

        R[self.split_x[4] : self.split_x[5]] = self.system.gamma(tn1, qn1, un12)
        R[self.split_x[5] : self.split_x[6]] = self.system.gamma(tn1, qn1, un1)

        return R

    def p(self, z):
        P_N1 = z[: self.nla_N]
        P_F1 = z[self.nla_N : self.nla_N + self.nla_F]
        P_N_bar = z[self.nla_N + self.nla_F : 2 * self.nla_N + self.nla_F]
        P_F_bar = z[2 * self.nla_N + self.nla_F :]
        P_N2 = P_N_bar - P_N1
        P_F2 = P_F_bar - P_F1

        self.x, converged, error, i, _ = fsolve(
            self.f1,
            self.x.copy(),
            jac="2-point",
            eps=1e-6,
            fun_args=(P_N1, P_F1, P_N2, P_F2),
            jac_args=(P_N1, P_F1, P_N2, P_F2),
        )
        assert converged

        (
            qn1,
            un12,
            un1,
            _,
            _,
            _,
            _,
        ) = np.array_split(self.x, self.split_x)

        tn1 = self.tn + self.dt

        prox_r_N = self.system.prox_r_N(tn1, qn1)
        prox_r_F = self.system.prox_r_F(tn1, qn1)
        mu = self.system.mu

        R = np.zeros(2 * (self.nla_N + self.nla_F), dtype=float)

        ###########
        # Signorini
        ###########
        prox_r_N /= self.dt
        prox_arg = P_N1 - prox_r_N * self.system.g_N(tn1, qn1)
        I_N = prox_arg >= 0
        R[: self.nla_N] = prox_R0_np(prox_arg)

        # identify active tangent contacts based on active normal contacts and
        # NF-connectivity lists
        if np.any(I_N):
            I_F = np.array(
                [
                    c
                    for i, I_N_i in enumerate(I_N)
                    for c in self.system.NF_connectivity[i]
                    if I_N_i
                ],
                dtype=int,
            )
        else:
            I_F = np.array([], dtype=int)

        ##################################################
        # mixed Singorini on velocity level and impact law
        ##################################################
        R[self.nla_N + I_N] = prox_R0_np(
            P_N_bar[I_N] - prox_r_N[I_N] * self.system.xi_N(tn1, qn1, self.un, un1)[I_N]
        )

        ##############################
        # friction and tangent impacts
        ##############################
        gamma_F = self.system.gamma_F(tn1, qn1, un12)
        xi_F = self.system.xi_F(tn1, qn1, self.un, un1)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            if len(i_F):
                R[2 * self.nla_N + np.array(i_F)] = prox_sphere(
                    P_F1[i_F] - prox_r_F[i_N] * gamma_F[i_F],
                    mu[i_N] * P_N1[i_N],
                )
                if I_N[i_N]:
                    R[2 * self.nla_N + self.nla_F + np.array(i_F)] = prox_sphere(
                        P_F_bar[i_F] - prox_r_F[i_N] * xi_F[i_F],
                        mu[i_N] * P_N_bar[i_N],
                    )

        return R

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

            elif self.method == "fixed point decoupled":
                self.x = np.concatenate((self.qn, self.un, np.zeros(self.nla_g)))

                z0 = np.zeros(self.nla_N + self.nla_F)
                # from scipy.optimize import fixed_point
                # z = fixed_point(self.p, z0, method="iteration", xtol=1e-6, maxiter=500)

                z = z0.copy()
                for i1 in range(self.fix_point_max_iter):
                    z = self.p1(z0)

                    error1 = self.fix_point_error_function(
                        z[: self.nq + self.nu] - z0[: self.nq + self.nu]
                    )

                    converged1 = error1 < self.fix_point_tol
                    if converged1:
                        break
                    z0 = z

                P_N1 = z[: self.nla_N]
                P_F1 = z[self.nla_N :]

                qn1 = self.x[: self.nq]
                un12 = self.x[self.nq : self.nq + self.nu]
                P_g1 = self.x[self.nq + self.nu : self.nq + self.nu + self.nla_g]
                P_gamma1 = self.x[self.nq + self.nu + self.nla_g :]

                converged1 = True
                error1 = 0
                i1 = 0

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

                I_N = self.I_Nn1

                # identify active tangent contacts based on active normal contacts and
                # NF-connectivity lists
                if np.any(I_N):
                    I_F = np.array(
                        [
                            c
                            for i, I_N_i in enumerate(I_N)
                            for c in self.system.NF_connectivity[i]
                            if I_N_i
                        ],
                        dtype=int,
                    )
                else:
                    I_F = np.array([], dtype=int)

                A = bmat(
                    [
                        [Mn, -W_g, -W_gamma],
                        [-W_g.T, None, None],
                        [-W_gamma.T, None, None],
                    ],
                    format="csc",
                )

                lu = splu(A)

                # initial right hand side
                rhs = Mn @ un12 + 0.5 * self.dt * h
                P_N2 = np.zeros_like(P_N1)
                P_F2 = np.zeros_like(P_F1)

                # update rhs
                b = np.concatenate(
                    (
                        rhs + 0.5 * (W_N[:, I_N] @ P_N2[I_N] + W_F[:, I_F] @ P_F2[I_F]),
                        chi_g,
                        chi_gamma,
                    )
                )

                # solve for initial velocities and percussions of the bilateral
                # constraints for the fixed point iteration
                x = lu.solve(b)
                un1 = x[: self.nu]
                P_g2 = 2 * x[self.nu : self.nu + self.nla_g]
                P_gamma2 = 2 * x[self.nu + self.nla_g :]

                P_N_bar = np.zeros(self.nla_N, dtype=float)
                P_F_bar = np.zeros(self.nla_F, dtype=float)

                converged2 = True
                error2 = 0
                i2 = 0
                un_fixed_point = un12.copy()

                # only enter fixed-point loop if any contact is active
                if np.any(I_N):
                    # compute new estimates for prox parameters and get friction coefficient
                    prox_r_N = self.system.prox_r_N(tn1, qn1)
                    prox_r_F = self.system.prox_r_F(tn1, qn1)
                    mu = self.system.mu
                    converged2 = False
                    P_N_bar_i1 = P_N1.copy() + P_N2.copy()
                    P_F_bar_i1 = P_F1.copy() + P_F2.copy()
                    for i2 in range(self.fix_point_max_iter):

                        # fixed-point update normal direction
                        P_N_bar_i1[I_N] = prox_R0_np(
                            P_N_bar_i1[I_N]
                            - prox_r_N[I_N]
                            * self.system.xi_N(tn1, qn1, self.un, un1)[I_N]
                        )

                        # fixed-point update friction
                        xi_F = self.system.xi_F(tn1, qn1, self.un, un1)
                        for i_N, i_F in enumerate(self.system.NF_connectivity):
                            if I_N[i_N] and len(i_F):
                                P_F_bar_i1[i_F] = prox_sphere(
                                    P_F_bar_i1[i_F] - prox_r_F[i_N] * xi_F[i_F],
                                    mu[i_N] * P_N_bar_i1[i_N],
                                )

                        P_N2 = 2 * P_N_bar_i1 - P_N1
                        P_F2 = 2 * P_F_bar_i1 - P_F1

                        # update rhs
                        b = np.concatenate(
                            (
                                rhs
                                + 0.5
                                * (W_N[:, I_N] @ P_N2[I_N] + W_F[:, I_F] @ P_F2[I_F]),
                                chi_g,
                                chi_gamma,
                            )
                        )

                        # solve for new velocities and Lagrange multipliers of bilateral constraints
                        x = lu.solve(b)
                        un1 = x[: self.nu]
                        P_g2 = 2 * x[self.nu : self.nu + self.nla_g]
                        P_gamma2 = 2 * x[self.nu + self.nla_g :]

                        # check for convergence
                        error2 = self.fix_point_error_function(un1 - un_fixed_point)
                        un_fixed_point = un1
                        converged2 = error2 < self.fix_point_tol
                        if converged2:
                            P_N_bar[I_N] = P_N_bar_i1[I_N]
                            P_F_bar[I_F] = P_F_bar_i1[I_F]
                            break

                converged = converged1 and converged2
                error = error1 + error2
                i = i1 + i2

                P_gn1 = 0.5 * (P_g1 + P_g2)

                qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            elif self.method == "fixed point full":
                z0 = np.zeros(2 * (self.nla_N + self.nla_F))
                from scipy.optimize import fixed_point
                z = fixed_point(self.p, z0, method="iteration", xtol=1e-6, maxiter=500)

                print(f"")

            else:
                raise NotImplementedError

            pbar.set_description(f"t: {tn1:0.2e}; step: {i+1}; error: {error:.3e}")
            if not converged:
                # raise RuntimeError(
                print(
                    f"step is not converged after {i+1} iterations with error: {error:.5e}"
                )

            q.append(qn1)
            u.append(un1)
            P_g.append(P_gn1)
            # P_gamma.append(P_gamman1)
            P_N.append(P_N_bar)
            P_F.append(P_F_bar)

            # update local variables for accepted time step
            if self.method == "Newton_decoupled":
                self.y1n = y1.copy()
                self.y2n = y2.copy()
                self.qn = qn1.copy()
                self.un = un1.copy()
            elif self.method == "Newton_full":
                self.yn = y.copy()
            elif self.method == "fixed point":
                self.qn = qn1.copy()
                self.un = un1.copy()

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
