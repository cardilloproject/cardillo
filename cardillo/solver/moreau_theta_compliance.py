import warnings

import numpy as np
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve
from scipy.sparse import bmat, block_diag, coo_array, csc_array, eye, diags_array
from scipy.sparse.linalg import splu, gmres, cg, LinearOperator
from tqdm import tqdm

from cardillo.math.approx_fprime import approx_fprime
from cardillo.math.fsolve import fsolve
from cardillo.math.prox import NegativeOrthant, estimate_prox_parameter
from cardillo.solver import Solution, SolverOptions, SolverSummary


def fixed_point_iteration(f, q0, atol=1e-6, rtol=1e-6, max_iter=100):
    q = q0.copy()
    scale = atol + np.abs(q0) * rtol
    for k in range(max_iter):
        q_new = f(q)
        error = np.linalg.norm((q_new - q) / scale) / len(scale) ** 0.5
        if error < 1:
            return q_new, k + 1, error
        q = q_new
    raise ValueError(
        f"Fixed-point iteration did not converge after {k + 1} iterations with error: {error}"
    )


class MoreauThetaCompliance:
    def __init__(
        self,
        system,
        t1,
        dt,
        # velocity_level_contact=False,
        velocity_level_contact=True,
        theta=0.5,
        options=SolverOptions(),
    ):
        self.theta = theta
        assert 0 < theta <= 1
        self.velocity_level_contact = velocity_level_contact
        if not velocity_level_contact:
            # otherwise an arbitrary impact law is realized
            assert np.isclose(theta, 1.0)

        self.system = system

        # simplified Newton iterations
        # options.reuse_lu_decomposition = True
        options.reuse_lu_decomposition = False
        self.options = options

        options.numerical_jacobian_method = "2-point"
        if options.numerical_jacobian_method:
            self.J_x = lambda x, y: csc_array(
                approx_fprime(
                    x,
                    lambda x: self.R_x(x, y),
                    method=options.numerical_jacobian_method,
                    eps=options.numerical_jacobian_eps,
                )
            )
            self.J_z = lambda z: csc_array(
                approx_fprime(
                    z,
                    lambda z: self.R_z(z),
                    method=options.numerical_jacobian_method,
                    eps=options.numerical_jacobian_eps,
                )
            )
        else:
            self.J_x = self._J_x
            self.J_z = self._J_z

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
        self.nla = self.nla_g + self.nla_gamma + self.nla_c
        self.nla_N = system.nla_N
        self.nla_F = system.nla_F

        # self.nx = self.nu + self.nla_g + self.nla_gamma + self.nla_c
        # self.ny = self.nla_N + self.nla_F
        self.split_la = np.cumsum(
            np.array(
                [
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_c,
                ],
                dtype=int,
            )
        )[:-1]
        self.split_x = np.cumsum(
            np.array(
                [
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
        self.split_z = np.cumsum(
            np.array(
                [
                    self.nq,
                    self.nla_N,
                    self.nu,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_c,
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
        self.zn = self.dt * np.concatenate(
            (
                self.q_dotn,
                0 * self.la_Nn,
                self.u_dotn,
                self.la_gn,
                self.la_gamman,
                self.la_cn,
                self.la_Nn,
                self.la_Fn,
            )
        )

        # # initial mass matrix and force directions for prox-parameter estimation
        # self.M = system.M(self.tn, self.qn)
        # self.W_N = system.W_N(self.tn, self.qn)
        # self.W_F = system.W_F(self.tn, self.qn)

    def R_z(self, zn1):
        (
            dqn1,
            dmu_Nn1,
            dun1,
            dP_gn1,
            dP_gamman1,
            dP_cn1,
            dP_Nn1,
            dP_Fn1,
        ) = np.array_split(zn1, self.split_z)

        # initialize residual
        R = np.empty_like(zn1)

        #############
        # integration
        #############
        theta = self.theta
        dt = self.dt
        tn = self.tn
        qn = self.qn
        un = self.un

        # - first stage
        tm = tn + (1 - theta) * dt
        qm = qn + (1 - theta) * dt * self.system.q_dot(tn, qn, un)
        # qm = qn + (1 - theta) * dqn1
        # solve for qm with fixed-point iterations
        qm, niter = fixed_point_iteration(
            lambda qm: qn + (1 - theta) * dt * self.system.q_dot(tm, qm, un),
            # qn, # naive initial guess
            qn
            + (1 - theta)
            * dt
            * self.system.q_dot(tn, qn, un),  # improved initial guess
        )
        # print(f"niter fixed-point: {niter}")

        # - second (final) stage
        tn1 = tn + dt
        # qn1 = qn + dqn1
        un1 = un + dun1
        qn1 = qm + theta * dt * self.system.q_dot(tm, qm, un1)
        # qn1 = qm + dqn1

        ####################
        # kinematic equation
        ####################
        # R[: self.split_z[0]] = dqn1 - dt * self.system.q_dot(tm, qm, un)
        # R[: self.split_z[0]] = dqn1 - theta * dt * self.system.q_dot(tn1, qn1, un1)
        R[: self.split_z[0]] = dqn1

        ##########################
        # unilateral stabilization
        ##########################
        if self.nla_N + self.nla_F > 0:
            g_N = self.system.g_N(tm, qm)
            # # I_N = np.where(g_N <= 0)[0]
            # # I_N = self.prox_r_N / dt * g_N - dmu_Nn1 <= 0
            # g_N_q = self.system.g_N_q(tm, qm)
            # R[: self.split_z[0]] -= g_N_q.T @ dmu_Nn1
            # R[self.split_z[0] : self.split_z[1]] = dmu_Nn1 + NegativeOrthant.prox(self.prox_r_N / dt * g_N - dmu_Nn1)
            R[self.split_z[0] : self.split_z[1]] = dmu_Nn1

        #####################
        # equations of motion
        #####################
        R[self.split_z[1] : self.split_z[2]] = (
            self.system.M(tm, qm) @ dun1
            # - 0.5 * self.dt * (self.system.h(tm, qm, un) + self.system.h(tm, qm, un1))
            # note: this variant is way better since only a single evaluation
            #       of h is required!
            - self.dt * self.system.h(tm, qm, 0.5 * (un + un1))
            - self.system.W_g(tm, qm, format="csr") @ dP_gn1
            - self.system.W_gamma(tm, qm, format="csr") @ dP_gamman1
            - self.system.W_c(tm, qm, format="csr") @ dP_cn1
            - self.system.W_N(tm, qm, format="csr") @ dP_Nn1
            - self.system.W_F(tm, qm, format="csr") @ dP_Fn1
        )

        #######################
        # bilateral constraints
        #######################
        R[self.split_z[2] : self.split_z[3]] = self.system.g(tn1, qn1)
        R[self.split_z[3] : self.split_z[4]] = self.system.gamma(tn1, qn1, un1)

        ############
        # compliance
        ############
        R[self.split_z[4] : self.split_z[5]] = self.system.c(
            tn1, qn1, un1, dP_cn1 / self.dt
        )

        ################
        # normal contact
        ################
        if self.nla_N + self.nla_F > 0:
            xi_N = self.system.xi_N(tm, tm, qm, qm, un, un1)
            # dP_Nn1 = np.where(
            #     g_N <= 0,
            #     -NegativeOrthant.prox(self.prox_r_N * xi_N - dP_Nn1),
            #     np.zeros_like(dP_Nn1),
            # )
            R[self.split_z[5] : self.split_z[6]] = np.where(
                g_N <= 0,
                # I_N,
                # dmu_Nn1 > 0,
                dP_Nn1 + NegativeOrthant.prox(self.prox_r_N * xi_N - dP_Nn1),
                dP_Nn1,
                # dP_Nn1 + dmu_Nn1 + NegativeOrthant.prox(self.prox_r_N * xi_N - dP_Nn1 - dmu_Nn1),
                # dP_Nn1 + dmu_Nn1,
            )

        ##########
        # friction
        ##########
        if self.nla_N + self.nla_F > 0:
            xi_F = self.system.xi_F(tn, tn1, qn, qn1, un, un1)
            for contr in self.system.get_contribution_list("gamma_F"):
                la_FDOF = contr.la_FDOF
                gamma_F_contr = xi_F[la_FDOF]
                dP_Fn1_contr = dP_Fn1[la_FDOF]
                prox_r_F_contr = self.prox_r_F[la_FDOF]
                for i_N, i_F, force_recervoir in contr.friction_laws:
                    if len(i_N) > 0:
                        dP_Nn1i = dP_Nn1[contr.la_NDOF[i_N]]
                    else:
                        dP_Nn1i = self.dt

                    R[self.split_z[6] + la_FDOF[i_F]] = dP_Fn1_contr[
                        i_F
                    ] + force_recervoir.prox(
                        prox_r_F_contr[i_F] * gamma_F_contr[i_F] - dP_Fn1_contr[i_F],
                        dP_Nn1i,
                    )

        self.qn1 = qn1.copy()
        self.un1 = un1.copy()

        return R

    def R_x(self, xn1, yn1):
        (
            dun1,
            dP_gn1,
            dP_gamman1,
            dP_cn1,
        ) = np.array_split(xn1, self.split_x)
        (
            dP_Nn1,
            dP_Fn1,
        ) = np.array_split(yn1, self.split_y)

        # initialize residual
        R_x = np.empty_like(xn1)

        # ######################################
        # # v0:
        # # strange method that is very accurate
        # ######################################

        # # TODO: Why is this method so accurate?
        # # - the method satisfies condition (2.3b) for symplectic PRK schemes,
        # #   see https://www.jstor.org/stable/2158439?seq=1. For separable
        # #   Hamiltonians H(q, p) = T(p) + U(q), the condition b_i = b^_i can
        # #   be ommited.
        # # - the method is discussed in Example 3.2 of https://www.emis.de/journals/ETNA/vol.2.1994/pp194-204.dir/pp194-204.pdf
        # #   for seperable partitioned systems.
        # # - In the end of the paper from above:
        # #   Further, a (s1,s2)–stage partitioned Runge–Kutta method with M(1,2) = 0 has
        # #   order two if Cˆ(1) and Bˆ(2) hold, order three if Cˆ(1) and Bˆ(3) hold, and order four
        # #   if it is order three and the symmetry condition in the usual sense is satisfied. Order
        # #   conditions of s–stage partitioned Runge–Kutta methods with M(1,2) = 0 are known
        # #   (see Sanz–Serna and Calvo [10, Chap. 7]). These conditions are easily modified for
        # #   (s1,s2)–stage methods.

        # # integration
        # tn1 = self.tn + self.dt
        # un1 = self.un + dun1
        # self.tn_theta = self.tn + self.dt * (1 - self.theta)
        # self.qn_theta = self.qn + self.dt * (1 - self.theta) * self.system.q_dot(
        #     self.tn, self.qn, self.un
        # )
        # # # TODO: Here we have to evaluate self.system.q_dot(tn1, qn1, un1) for a correct theta-method
        # # qn1 = self.qn_theta + self.dt * self.theta * self.system.q_dot(
        # #     tn1, self.qn_theta, un1
        # # )

        # # TODO: Symmetric methid with q = B(q) u
        # qn1 = self.qn + (1 - self.theta) * self.dt * self.system.q_dot(
        #     self.tn, self.qn, self.un
        # ) + self.theta * self.dt * self.system.q_dot(
        #     self.tn, self.qn, un1
        # )

        # # equations of motion (v1)
        # R_x[: self.split_x[0]] = (
        #     self.system.M(self.tn_theta, self.qn_theta) @ dun1
        #     - self.dt * self.system.h(self.tn_theta, self.qn_theta, self.un)
        #     # - self.dt * self.system.W_tau(self.tn_theta, self.qn_theta, format="csr") @ self.system.la_tau(self.tn_theta, self.qn_theta, self.un)
        #     - self.system.W_g(self.tn_theta, self.qn_theta, format="csr") @ dP_gn1
        #     - self.system.W_gamma(self.tn_theta, self.qn_theta, format="csr")
        #     @ dP_gamman1
        #     - self.system.W_c(self.tn_theta, self.qn_theta, format="csr") @ dP_cn1
        #     - self.system.W_N(self.tn_theta, self.qn_theta, format="csr") @ dP_Nn1
        #     - self.system.W_F(self.tn_theta, self.qn_theta, format="csr") @ dP_Fn1
        # )

        # ########################
        # # v1:
        # # theta-method for q's
        # # implicit Euler for u's
        # ########################

        # # integration
        # tn1 = self.tn + self.dt
        # un1 = self.un + dun1
        # self.tn_theta = self.tn + self.dt * (1 - self.theta)
        # self.qn_theta = self.qn + self.dt * (1 - self.theta) * self.system.q_dot(self.tn, self.qn, self.un)
        # # TODO: Here we have to evaluate self.system.q_dot(tn1, qn1, un1) for a correct theta-method
        # qn1 = self.qn_theta + self.dt * self.theta * self.system.q_dot(tn1, self.qn_theta, un1)

        # # true theta-method
        # # solve for qn1 which gives the true theta method
        # f = lambda qn1: qn1 - self.qn_theta - self.dt * self.theta * self.system.q_dot(tn1, qn1, un1)
        # sol = fsolve(f, qn1)
        # qn1 = sol.x

        # # equations of motion (v1)
        # R_x[: self.split_x[0]] = (
        #     # # why this method is so accurate?
        #     self.system.M(self.tn_theta, self.qn_theta) @ dun1
        #     - self.dt * self.system.h(self.tn_theta, self.qn_theta, self.un)
        #     # - self.dt * self.system.W_tau(self.tn_theta, self.qn_theta, format="csr") @ self.system.la_tau(self.tn_theta, self.qn_theta, self.un)
        #     - self.system.W_g(self.tn_theta, self.qn_theta, format="csr") @ dP_gn1
        #     - self.system.W_gamma(self.tn_theta, self.qn_theta, format="csr") @ dP_gamman1
        #     - self.system.W_c(self.tn_theta, self.qn_theta, format="csr") @ dP_cn1
        #     - self.system.W_N(self.tn_theta, self.qn_theta, format="csr") @ dP_Nn1
        #     - self.system.W_F(self.tn_theta, self.qn_theta, format="csr") @ dP_Fn1
        #     #
        #     # # TODO: Theta method is also required here for global damping in elastic_chain_pendulum
        #     # self.system.M(tn1, qn1) @ dun1
        #     # - self.dt * self.system.h(tn1, qn1, un1)
        #     # # - self.dt * self.system.W_tau(tn1, qn1, format="csr") @ self.system.la_tau(tn1, qn1, un1)
        #     # - self.system.W_g(tn1, qn1, format="csr") @ dP_gn1
        #     # - self.system.W_gamma(tn1, qn1, format="csr") @ dP_gamman1
        #     # - self.system.W_c(tn1, qn1, format="csr") @ dP_cn1
        #     # - self.system.W_N(tn1, qn1, format="csr") @ dP_Nn1
        #     # - self.system.W_F(tn1, qn1, format="csr") @ dP_Fn1
        # )

        ########################
        # v2:
        # Heun's method for q's
        # implicit Euler for u's
        ########################

        # # integration
        # tn1 = self.tn + self.dt
        # un1 = self.un + dun1
        # Q_dot1 = self.system.q_dot(self.tn, self.qn, self.un)
        # # # Original Heun's method
        # # T2 = tn1
        # # Q2 = self.qn + self.dt * Q_dot1
        # # qn1 = self.qn + 0.5 * self.dt * (Q_dot1 + self.system.q_dot(T2, Q2, un1))
        # # # TODO: Why is this so good?
        # # T2 = self.tn + 0.5 * self.dt
        # # Q2 = self.qn + 0.5 * self.dt * Q_dot1
        # # qn1 = self.qn + 0.5 * self.dt * (Q_dot1 + self.system.q_dot(T2, Q2, un1))
        # # generic second-order method
        # alpha = 0.5 # works for knife edge
        # # alpha = 1
        # # alpha = 2 / 3
        # b2 = 1 / (2 * alpha)
        # b1 = 1 - b2
        # T2 = self.tn + alpha * self.dt
        # Q2 = self.qn + alpha * self.dt * Q_dot1
        # qn1 = self.qn + self.dt * (b1 * Q_dot1 + b2 * self.system.q_dot(T2, Q2, un1))
        # # qn1 = self.qn + 0.5 * self.dt * (Q_dot1 + self.system.q_dot(T2, Q2, un1))

        # # # "strange" theta-method
        # # # theta = self.theta
        # # theta = 0.5
        # # # theta = 0.75
        # # # theta = 1
        # # T2 = self.tn + (1 - theta) * self.dt
        # # Q2 = self.qn + self.dt * (1 - theta) * Q_dot1
        # # qn1 = self.qn + self.dt * ((1 - theta) * Q_dot1 + theta * self.system.q_dot(T2, Q2, un1))

        # # # true theta-method
        # # # solve for qn1 which gives the true theta method
        # # f = lambda qn1: qn1 - self.qn - self.dt * ((1 - theta) * Q_dot1 + theta * self.system.q_dot(tn1, qn1, un1))
        # # sol = fsolve(f, qn1)
        # # qn1 = sol.x
        # # Q2 = qn1

        # # equations of motion (v1)
        # R_x[: self.split_x[0]] = (
        #     self.system.M(T2, Q2) @ dun1
        #     - self.dt * self.system.h(T2, Q2, un1)
        #     - self.dt
        #     * self.system.W_tau(T2, Q2, format="csr")
        #     @ self.system.la_tau(T2, Q2, un1)
        #     - self.system.W_g(T2, Q2, format="csr") @ dP_gn1
        #     - self.system.W_gamma(T2, Q2, format="csr") @ dP_gamman1
        #     - self.system.W_c(T2, Q2, format="csr") @ dP_cn1
        #     - self.system.W_N(T2, Q2, format="csr") @ dP_Nn1
        #     - self.system.W_F(T2, Q2, format="csr") @ dP_Fn1
        # )

        ########################
        # v3:
        # Störmer-Verlet (B)
        # see https://www.math.kit.edu/ianm3/lehre/geonumint2009s/media/gni_by_stoermer-verlet.pdf
        ########################

        # integration
        tn1 = self.tn + self.dt
        un1 = self.un + dun1
        tm = self.tn + 0.5 * self.dt

        # implicit mid-point positions
        # f_qm = lambda qm: qm - self.qn - 0.5 * self.dt * self.system.q_dot(tm, qm, self.un)
        # sol = fsolve(f_qm, self.qn, options=SolverOptions(numerical_jacobian_method="2-point"))
        # qm = sol.x
        # explicit mid-point position (symmetric for q_dot = u)
        qm = self.qn + 0.5 * self.dt * self.system.q_dot(tm, self.qn, self.un)
        qn1 = qm + 0.5 * self.dt * self.system.q_dot(tm, qm, un1)

        # # trapezoidal method
        # qm = self.qn + 0.5 * self.dt * self.system.q_dot(self.tn, self.qn, self.un)
        # qn1 = qm + 0.5 * self.dt * self.system.q_dot(tn1, qm, un1)
        # # f_qn1 = lambda qn1: qn1 - qm - 0.5 * self.dt * self.system.q_dot(tn1, qn1, un1)
        # # sol = fsolve(f_qn1, self.qn, options=SolverOptions(numerical_jacobian_method="2-point"))
        # # qn1 = sol.x

        # equations of motion
        R_x[: self.split_x[0]] = (
            self.system.M(tm, qm) @ dun1
            - 0.5
            * self.dt
            * (
                2
                * self.system.h(tm, qm, self.un)
                # self.system.h(tm, qm, self.un) + self.system.h(tm, qm, un1)
            )
            # - self.dt * self.system.W_tau(self.tn_theta, self.qn_theta, format="csr") @ self.system.la_tau(self.tn_theta, self.qn_theta, self.un)
            - self.system.W_g(tm, qm, format="csr") @ dP_gn1
            - self.system.W_gamma(tm, qm, format="csr") @ dP_gamman1
            - self.system.W_c(tm, qm, format="csr") @ dP_cn1
            - self.system.W_N(tm, qm, format="csr") @ dP_Nn1
            - self.system.W_F(tm, qm, format="csr") @ dP_Fn1
            #
            # # TODO: Theta method is also required here for global damping in elastic_chain_pendulum
            # self.system.M(tn1, qn1) @ dun1
            # - self.dt * self.system.h(tn1, qn1, un1)
            # # - self.dt * self.system.W_tau(tn1, qn1, format="csr") @ self.system.la_tau(tn1, qn1, un1)
            # - self.system.W_g(tn1, qn1, format="csr") @ dP_gn1
            # - self.system.W_gamma(tn1, qn1, format="csr") @ dP_gamman1
            # - self.system.W_c(tn1, qn1, format="csr") @ dP_cn1
            # - self.system.W_N(tn1, qn1, format="csr") @ dP_Nn1
            # - self.system.W_F(tn1, qn1, format="csr") @ dP_Fn1
        )

        #######################
        # bilateral constraints
        #######################
        R_x[self.split_x[0] : self.split_x[1]] = self.system.g(tn1, qn1)
        R_x[self.split_x[1] : self.split_x[2]] = self.system.gamma(tn1, qn1, un1)

        ############
        # compliance
        ############
        R_x[self.split_x[2] :] = self.system.c(tn1, qn1, un1, dP_cn1 / self.dt)

        self.qn1 = qn1.copy()
        self.un1 = un1.copy()

        return R_x

    def _J_x(self, xn1, yn1):
        # (
        #     dun1,
        #     dP_gn1,
        #     dP_gamman1,
        #     dP_cn1,
        # ) = np.array_split(xn1, self.split_x)
        # (
        #     dP_Nn1,
        #     dP_Fn1,
        # ) = np.array_split(yn1, self.split_y)

        # # integration
        # dt_th = self.dt * self.theta
        # tn1 = self.tn + self.dt
        # # un1 = self.un_theta + dun1
        # un1 = self.un + dun1
        # qn1 = self.qn_theta + self.dt * self.theta * self.system.q_dot(
        #     self.tn_theta, self.qn_theta, un1
        # )
        # q_dot_u = self.system.q_dot_u(self.tn_theta, self.qn_theta)

        # #######################
        # # bilateral constraints
        # #######################
        # g_q = self.system.g_q(tn1, qn1)
        # gamma_q = self.system.gamma_q(tn1, qn1, un1)
        # gamma_u = self.system.gamma_u(tn1, qn1)

        # ############
        # # compliance
        # ############
        # c_q = self.system.c_q(tn1, qn1, un1, dP_cn1 / self.dt)
        # c_u = self.system.c_u(tn1, qn1, un1, dP_cn1 / self.dt)
        # c_la_c = self.system.c_la_c() / self.dt

        # # note: This is a simplified Jacobian and does not coincide with the exact one!
        # # fmt: off
        # J = bmat(
        #     [
        #         [                             self.M, -self.W_g, -self.W_gamma,        -self.W_c],
        #         [              g_q @ q_dot_u * dt_th,      None,          None,             None],
        #         [gamma_q @ q_dot_u * dt_th + gamma_u,      None,          None,             None],
        #         [        c_q @ q_dot_u * dt_th + c_u,      None,          None, c_la_c / self.dt],
        #     ],
        #     format="csc",
        # )
        # # fmt: on
        # return J

        J_num = approx_fprime(xn1, lambda x: self.R_x(x, yn1), method="2-point")
        # diff = J_num - J
        # error = np.linalg.norm(diff)
        # print(f"error J: {error}")
        return J_num

    def prox(self, x1, y0):
        (
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
        un1 = self.un + dun1

        # theta-method
        self.tn_theta = self.tn + self.dt * (1 - self.theta)
        self.qn_theta = self.qn + self.dt * (1 - self.theta) * self.system.q_dot(
            self.tn, self.qn, self.un
        )
        qn1 = self.qn_theta + self.dt * self.theta * self.system.q_dot(
            self.tn_theta, self.qn_theta, un1
        )

        prox_r_N = self.prox_r_N
        prox_r_F = self.prox_r_F

        y1 = np.zeros_like(y0)

        ##############################
        # fixed-point update Signorini
        ##############################
        if self.velocity_level_contact:
            # explicit gap evaluation
            g_N = self.system.g_N(self.tn_theta, self.qn_theta)
            xi_N = self.system.xi_N(tn, tn1, qn, qn1, un, un1)
            dP_Nn1 = np.where(
                g_N <= 0,
                -NegativeOrthant.prox(prox_r_N * xi_N - dP_Nn1),
                np.zeros_like(dP_Nn1),
            )
        else:
            g_N = self.system.g_N(tn1, qn1)
            dP_Nn1 = -NegativeOrthant.prox((prox_r_N / self.dt) * g_N - dP_Nn1)
        y1[: self.split_y[0]] = dP_Nn1

        #############################
        # fixed-point update friction
        #############################
        if self.velocity_level_contact:
            gamma_F = self.system.xi_F(tn, tn1, qn, qn1, un, un1)
        else:
            gamma_F = self.system.gamma_F(tn1, qn1, un1)
        for contr in self.system.get_contribution_list("gamma_F"):
            la_FDOF = contr.la_FDOF
            gamma_F_contr = gamma_F[la_FDOF]
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

    def _make_solution(self):
        return Solution(
            system=self.system,
            t=np.array(self.sol_t),
            q=np.array(self.sol_q),
            u=np.array(self.sol_u),
            la_c=np.array(self.sol_la_c),
            P_g=np.array(self.sol_P_g),
            P_gamma=np.array(self.sol_P_gamma),
            P_N=np.array(self.sol_P_N),
            P_F=np.array(self.sol_P_F),
            solver_summary=self.solver_summary,
        )

    def _step_naive_old(self):
        # theta step
        dt = self.dt
        self.tn_theta = self.tn + dt * (1 - self.theta)
        self.qn_theta = self.qn + dt * (1 - self.theta) * self.system.q_dot(
            self.tn, self.qn, self.un
        )

        self.M = self.system.M(self.tn_theta, self.qn_theta, format="csc")
        # # # TODO: What to do with control forces?
        # # h_Wla_tau = self.system.h(
        # #     self.tn_theta, self.qn_theta, self.un
        # # ) + self.system.W_tau(
        # #     self.tn_theta, self.qn_theta, format="csr"
        # # ) @ self.system.la_tau(
        # #     self.tn_theta, self.qn_theta, self.un
        # # )
        M_inv = splu(self.M)  # TODO: Use this for prox parameter estimation
        # self.un_theta = self.un + dt * M_inv.solve(h_Wla_tau)

        # # evaluate quantities that are kept fixed during the simplified Newton iterations
        # self.W_g = self.system.W_g(self.tn_theta, self.qn_theta)
        # self.W_gamma = self.system.W_gamma(self.tn_theta, self.qn_theta)
        self.W_N = self.system.W_N(self.tn_theta, self.qn_theta)
        self.W_F = self.system.W_F(self.tn_theta, self.qn_theta)
        # self.W_c = self.system.W_c(self.tn_theta, self.qn_theta)
        # # self.c_q = self.system.c_q(self.tn_theta, self.qn_theta, self.la_cn)
        # # self.c_la_c = self.system.c_la_c(self.tn_theta, self.qn_theta, self.la_cn)

        # only compute optimized prox-parameters once per time step
        if self.nla_N + self.nla_F > 0:
            self.prox_r_N, self.prox_r_F = np.array_split(
                estimate_prox_parameter(
                    self.options.prox_scaling, bmat([[self.W_N, self.W_F]]), M_inv
                ),
                [self.nla_N],
            )

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
        else:
            lu = None

        # solve nonlinear system
        sol = self._solve_nonlinear_system(x0, y0, lu)
        xn1 = sol.x
        self.solver_summary.add_lu(sol.njev)

        if not sol.success:
            if self.options.continue_with_unconverged:
                warnings.warn("Newton is not converged but integration is continued")
            else:
                warnings.warn(
                    f"Newton is not converged. Returning solution up to t={self.tn}."
                )
                return self._make_solution()

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
                            return self._make_solution()

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
                return self._make_solution()

        self.solver_summary.add_fixed_point(i_fixed_point, error)
        self.solver_summary.add_newton(sol.nit, sol.error)

        # update progress bar
        tn1 = self.tn + self.dt
        self.pbar.set_description(
            f"t: {tn1:0.2e}s < {self.t1:0.2e}s; |y1 - y0|_rel: {error:0.2e}; fixed-point: {i_fixed_point}/{self.options.fixed_point_max_iter}; |dx|_rel: {sol.error:0.2e}; newton: {sol.nit}/{self.options.newton_max_iter}"
        )

        # compute state
        (
            dun1,
            dP_gn1,
            dP_gamman1,
            dP_cn1,
        ) = np.array_split(xn1, self.split_x)
        (
            dP_Nn1,
            dP_Fn1,
        ) = np.array_split(y0, self.split_y)

        # modify converged quantities
        qn1, un1 = self.system.step_callback(tn1, self.qn1, self.un1)

        # store solution fields
        self.sol_t.append(tn1)
        self.sol_q.append(qn1)
        self.sol_u.append(un1)
        self.sol_la_c.append(dP_cn1 / self.dt)
        self.sol_P_g.append(dP_gn1)
        self.sol_P_gamma.append(dP_gamman1)
        self.sol_P_N.append(dP_Nn1)
        self.sol_P_F.append(dP_Fn1)

        # update local variables for accepted time step
        self.xn = xn1.copy()
        self.yn = y0.copy()
        self.tn = tn1
        self.qn = qn1.copy()
        self.un = un1.copy()

    def _step_naive(self):
        M = self.system.M(self.tn, self.qn, format="csc")
        W_N = self.system.W_N(self.tn, self.qn)
        W_F = self.system.W_F(self.tn, self.qn)

        # only compute optimized prox-parameters once per time step
        if self.nla_N + self.nla_F > 0:
            self.prox_r_N, self.prox_r_F = np.array_split(
                estimate_prox_parameter(
                    self.options.prox_scaling, bmat([[W_N, W_F]]), M
                ),
                [self.nla_N],
            )

        ########################
        # fixed-point iterations
        ########################
        # solve nonlinear system
        sol = fsolve(
            lambda z: self.R_z(z),
            # self.zn.copy(),
            self.zn,
            jac="2-point",
            options=self.options,
        )
        zn1 = sol.x
        self.solver_summary.add_lu(sol.njev)
        self.solver_summary.add_newton(sol.nit, sol.error)

        if not sol.success:
            if self.options.continue_with_unconverged:
                warnings.warn("Newton is not converged but integration is continued")
            else:
                warnings.warn(
                    f"Newton is not converged. Returning solution up to t={self.tn}."
                )
                return self._make_solution()

        # update progress bar
        tn1 = self.tn + self.dt
        self.pbar.set_description(
            f"t: {tn1:0.2e}s < {self.t1:0.2e}s; |dz|_rel: {sol.error:0.2e}; newton: {sol.nit}/{self.options.newton_max_iter}"
        )

        # modify converged quantities
        qn1, un1 = self.system.step_callback(tn1, self.qn1, self.un1)

        # unpack other solution fields
        (
            dqn1,
            dmu_Nn1,
            dun1,
            dP_gn1,
            dP_gamman1,
            dP_cn1,
            dP_Nn1,
            dP_Fn1,
        ) = np.array_split(zn1, self.split_z)

        # store solution fields
        self.sol_t.append(tn1)
        self.sol_q.append(qn1)
        self.sol_u.append(un1)
        self.sol_la_c.append(dP_cn1 / self.dt)
        self.sol_P_g.append(dP_gn1)
        self.sol_P_gamma.append(dP_gamman1)
        self.sol_P_N.append(dP_Nn1)
        self.sol_P_F.append(dP_Fn1)

        # update local variables for accepted time step
        self.zn = zn1.copy()
        self.tn = tn1
        self.qn = qn1.copy()
        self.un = un1.copy()

    def c(self, t, q, u, la):
        """Combine all constraint forces in order to simplify the solver."""
        la_g, la_gamma, la_c = np.array_split(la, self.split_la)

        c = np.zeros(self.nla)
        c[: self.split_la[0]] = self.system.g(t, q)
        c[self.split_la[0] : self.split_la[1]] = self.system.gamma(t, q, u)
        c[self.split_la[1] :] = self.system.c(t, q, u, la_c)

        return c

    def c_q(self, t, q, u, la, format="coo"):
        la_g, la_gamma, la_c = np.array_split(la, self.split_la)

        g_q = self.system.g_q(t, q)
        gamma_q = self.system.gamma_q(t, q, u)
        c_q = self.system.c_q(t, q, u, la_c)

        return bmat([[g_q], [gamma_q], [c_q]], format=format)

    def c_u(self, t, q, u, la, format="coo"):
        la_g, la_gamma, la_c = np.array_split(la, self.split_la)

        g_u = np.zeros((self.nla_g, self.nu))
        gamma_u = self.system.gamma_u(t, q)
        c_u = self.system.c_u(t, q, u, la_c)

        return bmat([[g_u], [gamma_u], [c_u]], format=format)

    def c_la(self, format="coo"):
        g_la_g = np.zeros((self.nla_g, self.nla_g))
        gamma_la_gamma = np.zeros((self.nla_gamma, self.nla_gamma))
        c_la_c = self.system.c_la_c()

        return block_diag([g_la_g, gamma_la_gamma, c_la_c], format=format)

    def W(self, t, q, format="coo"):
        W_g = self.system.W_g(t, q)
        W_gamma = self.system.W_gamma(t, q)
        W_c = self.system.W_c(t, q)

        return bmat(
            [
                [W_g, W_gamma, W_c],
            ],
            format=format,
        )

    def _step_schur_old(self):
        # theta step (part 1)
        dt = self.dt
        theta = self.theta
        tn_theta = self.tn + dt * (1 - theta)
        qn_theta = self.qn + dt * (1 - theta) * self.system.q_dot(
            self.tn, self.qn, self.un
        )

        M = self.system.M(tn_theta, qn_theta, format="csc")
        # TODO: Since M is block diagonal, we can implement M_inv @ B on a
        # subsystem level or assemble M_inv directly which improves efficiency
        # alot. Morover, most system have a constant mass matrix so it gets
        # very cheap.
        M_inv = splu(M)
        self.solver_summary.add_lu(1)
        # TODO: Investigate if it is sufficient to use a diagonal apporximation
        # or a row sum lumped matrix approximation to the mass matrix. For this
        # case the Delassus matrix gets diagonal and we get rid of another
        # LU-decomposition.
        # M_diag_inv = diags_array(1 / M.diagonal())
        # M_inv = type("LU", (), {"solve": lambda self, rhs: M_diag_inv @ rhs})()
        # M_lumped_inv = diags_array(1 / M.sum(axis=1))
        # M_inv = type("LU", (), {"solve": lambda self, rhs: M_lumped_inv @ rhs})()

        # explicit h-vector and control forces
        # note: This is the only step where we do not apply the correct
        # theta-method (maybe I'm wrong).
        h_Wla_tau = self.system.h(tn_theta, qn_theta, self.un)
        h_Wla_tau += self.system.W_tau(
            tn_theta, qn_theta, format="csr"
        ) @ self.system.la_tau(tn_theta, qn_theta, self.un)

        # theta step (part 2): free velocity (without constraints and contacts)
        un_theta = self.un + dt * M_inv.solve(h_Wla_tau)

        # evaluate all other quantities that are kept fixed during the
        # simplified Newton iterations
        q_dot_u = self.system.q_dot_u(tn_theta, qn_theta)
        beta = self.system.q_dot(tn_theta, qn_theta, np.zeros_like(self.un))
        W = self.W(tn_theta, qn_theta)
        c_q = self.c_q(tn_theta, qn_theta, un_theta, self.la_cn)
        c_u = self.c_u(tn_theta, qn_theta, un_theta, self.la_cn)
        c_la = self.c_la()

        # compute iteration matrices
        A = dt * theta * c_q @ q_dot_u + c_u
        # TODO: This feels very bad since W might be sparse!
        M_inv_W = M_inv.solve(W.toarray())
        D = A @ M_inv_W + c_la / dt

        # for a moderate number of constraints D is small and dense,
        # hence we use a dense factorization
        # D_fac = lu_factor(D)
        # self.solver_summary.add_lu(1)
        # D_inv = type("LU", (), {"solve": lambda self, rhs: lu_solve(D_fac, rhs)})()
        # TODO: Investigate the performance of iterative methods for real applications
        D_inv = type("CG", (), {})()

        def solve(rhs):
            x, info = cg(D, rhs)
            # x, info = gmres(D, rhs)
            if info > 0:
                raise RuntimeError(
                    f"Iterative solver is not converged with 'info': {info}"
                )
            return x

        D_inv.solve = solve

        # - only compute optimized prox-parameters once per time step
        # - generalized force directions for constraint forces are only
        #   required if we have contacts
        if self.nla_N + self.nla_F > 0:
            W_N = self.system.W_N(tn_theta, qn_theta)
            W_F = self.system.W_F(tn_theta, qn_theta)
            self.prox_r_N, self.prox_r_F = np.array_split(
                estimate_prox_parameter(
                    self.options.prox_scaling, bmat([[W_N, W_F]]), M_inv
                ),
                [self.nla_N],
            )

        # TODO: Add outer fixed point loop for contacts

        ###################
        # newton iterations
        ###################
        if self.nla > 0:
            # compute initial positions velocities and percussions
            tn1 = self.tn + dt
            un1 = un_theta.copy()
            qn1 = qn_theta + dt * theta * (q_dot_u @ un1 + beta)
            # # TODO: Is this a good initial guess?
            # Pin1 = dt * np.concatenate([self.la_gn.copy(), self.la_gamman.copy(), self.la_cn.copy()])
            # TODO: Why is this guess much better?
            Pin1 = np.zeros(self.nla)

            # evaluate residuals
            # R1 = M @ (un1 - un_theta) - W @ Pin1
            # TODO: Implicit evaluation is required here for global damping in
            # elastic_chain_pendulum.
            R1 = M @ (un1 - un_theta) - self.W(tn1, qn1) @ Pin1
            R2 = self.c(tn1, qn1, un1, Pin1 / dt)
            R = np.concatenate((R1, R2))

            # newton scaling
            scale1 = self.options.newton_atol + np.abs(R1) * self.options.newton_rtol
            scale2 = self.options.newton_atol + np.abs(R2) * self.options.newton_rtol
            scale = np.concatenate((scale1, scale2))

            # error of initial guess
            error = np.linalg.norm(R / scale) / scale.size**0.5
            converged = error < 1
            print(f"i: {-1}; error: {error}; converged: {converged}")

            # Newton loop
            if not converged:
                for i in range(self.options.newton_max_iter):
                    # Newton updates
                    Delta_Pin1 = D_inv.solve(A @ R1 - R2)
                    Delta_Un1 = M_inv_W @ Delta_Pin1 - R1

                    # update dependent variables
                    un1 += Delta_Un1
                    Pin1 += Delta_Pin1
                    # TODO: Implicit evaluation is required here for global damping in
                    # elastic_chain_pendulum.
                    qn1 = qn_theta + dt * theta * (q_dot_u @ un1 + beta)

                    # evaluate residuals
                    # R1 = M @ (un1 - un_theta) - W @ Pin1
                    R1 = M @ (un1 - un_theta) - self.W(tn1, qn1) @ Pin1
                    R2 = self.c(tn1, qn1, un1, Pin1 / dt)
                    R = np.concatenate((R1, R2))

                    # error and convergence check
                    error = np.linalg.norm(R / scale) / scale.size**0.5
                    converged = error < 1
                    print(f"i: {i}; error: {error}; converged: {converged}")
                    if converged:
                        break

                if not converged:
                    warnings.warn(
                        f"Newton method is not converged after {i} iterations with error {error:.2e}"
                    )

                # TODO: Only used when continue_with_unconverged is implemented
                # nit = i + 1

        # modify converged quantities
        qn1, un1 = self.system.step_callback(tn1, qn1, un1)

        # unpack percussion
        Pi_gn1, Pi_gamman1, Pi_cn1 = np.array_split(Pin1, self.split_la)

        # store solution fields
        self.sol_t.append(tn1)
        self.sol_q.append(qn1)
        self.sol_u.append(un1)
        self.sol_la_c.append(Pi_cn1 / self.dt)
        self.sol_P_g.append(Pi_gn1)
        self.sol_P_gamma.append(Pi_gamman1)
        # self.sol_P_N.append(dP_Nn1)
        # self.sol_P_F.append(dP_Fn1)

        # update local variables for accepted time step
        self.tn = tn1
        self.qn = qn1.copy()
        self.un = un1.copy()
        self.la_gn = Pi_gn1.copy() / dt
        self.la_gamman = Pi_gamman1.copy() / dt
        self.la_cn = Pi_cn1.copy() / dt

    def _step_schur(self):
        theta = self.theta
        dt = self.dt
        tn = self.tn
        qn = self.qn
        un = self.un

        ######################################################################
        # 1. implicit mid-point step solved with simple fixed-point iterations
        ######################################################################
        tm = tn + (1 - theta) * dt
        # qm = qn.copy() # naive initial guess
        qm = qn + (1 - theta) * dt * self.system.q_dot(tn, qn, un)
        qm, niter, error = fixed_point_iteration(
            lambda qm: qn + (1 - theta) * dt * self.system.q_dot(tm, qm, un),
            qm,
        )
        print(f"Fixed-point:")
        print(f"i: {niter}; error: {error}")

        #########################################
        # evaluate all quantities at the midpoint
        #########################################
        M = self.system.M(tm, qm, format="csc")
        # TODO: Since M is block diagonal, we can implement M_inv @ B on a
        # subsystem level or assemble M_inv directly which improves efficiency
        # alot. Morover, most system have a constant mass matrix so it gets
        # very cheap.
        M_inv = splu(M)
        self.solver_summary.add_lu(1)

        q_dot_u = self.system.q_dot_u(tm, qm)
        beta = self.system.q_dot(tm, qm, np.zeros_like(self.un))
        W = self.W(tm, qm)
        c_q = self.c_q(tm, qm, un, self.la_cn)
        c_u = self.c_u(tm, qm, un, self.la_cn)
        C = self.c_la()

        ############################
        # compute iteration matrices
        ############################
        A = 0.5 * dt * c_q @ q_dot_u + c_u
        # TODO: This feels very bad since W might be sparse!
        M_inv_W = M_inv.solve(W.toarray())
        D = C / dt + A @ M_inv_W

        # for a moderate number of constraints D is small and dense,
        # hence we use a dense factorization
        # # - LU-decomposition
        # D_fac = lu_factor(D)
        # self.solver_summary.add_lu(1)
        # D_inv = type("LU", (), {"solve": lambda self, rhs: lu_solve(D_fac, rhs)})()

        # # - Cholesky-decomposition
        # D_fac = cho_factor(D)
        # self.solver_summary.add_lu(1)
        # D_inv = type("Cholesky", (), {"solve": lambda self, rhs: cho_solve(D_fac, rhs)})()

        # Krylov subspace methods for symetric positive definite D
        # - conjugate gradient (CG) with jacobi preconditioner
        D_inv = type("CG", (), {})()
        DD_inv = 1 / np.diag(D)
        preconditioner = LinearOperator(D.shape, lambda x: DD_inv * x)

        def solve(rhs):
            x, info = cg(D, rhs, M=preconditioner)
            # global cg_iter
            # cg_iter = 0
            # def callback(x):
            #     global cg_iter
            #     print(f"cg iteration: {cg_iter}")
            #     cg_iter += 1
            # x, info = cg(D, rhs, M=preconditioner, callback=callback)
            # x, info = cg(D, rhs, callback=callback)
            # x, info = gmres(D, rhs)
            if info > 0:
                raise RuntimeError(
                    f"Iterative solver is not converged with 'info': {info}"
                )
            return x

        D_inv.solve = solve

        # - only compute optimized prox-parameters once per time step
        # - generalized force directions for constraint forces are only
        #   required if we have contacts
        if self.nla_N + self.nla_F > 0:
            W_N = self.system.W_N(tm, qm)
            W_F = self.system.W_F(tm, qm)
            self.prox_r_N, self.prox_r_F = np.array_split(
                estimate_prox_parameter(
                    self.options.prox_scaling, bmat([[W_N, W_F]]), M_inv
                ),
                [self.nla_N],
            )

        # TODO: Add outer fixed point loop for contacts

        ###################
        # newton iterations
        ###################
        print(f"Newton:")
        # compute initial positions velocities and percussions
        tn1 = self.tn + dt
        un1 = un.copy()
        qn1 = qm + 0.5 * dt * (q_dot_u @ un1 + beta)
        # # TODO: Is this a good initial guess?
        # Pin1 = dt * np.concatenate([self.la_gn.copy(), self.la_gamman.copy(), self.la_cn.copy()])
        # TODO: Why is this guess much better?
        Pin1 = np.zeros(self.nla)

        # evaluate residuals
        R1 = M @ (un1 - un) - dt * self.system.h(tm, qm, 0.5 * (un + un1)) - W @ Pin1
        R2 = self.c(tn1, qn1, un1, Pin1 / dt)
        R = np.concatenate((R1, R2))

        # newton scaling
        scale1 = self.options.newton_atol + np.abs(R1) * self.options.newton_rtol
        scale2 = self.options.newton_atol + np.abs(R2) * self.options.newton_rtol
        scale = np.concatenate((scale1, scale2))

        # error of initial guess
        error = np.linalg.norm(R / scale) / scale.size**0.5
        converged = error < 1
        print(f"i: {-1}; error: {error}; converged: {converged}")

        # Newton loop
        if not converged:
            for i in range(self.options.newton_max_iter):
                # Newton updates
                M_inv_R1 = M_inv.solve(R1)
                Delta_Pin1 = D_inv.solve(A @ M_inv_R1 - R2)
                Delta_Un1 = M_inv_W @ Delta_Pin1 - M_inv_R1

                # update dependent variables
                un1 += Delta_Un1
                Pin1 += Delta_Pin1
                qn1 = qm + 0.5 * dt * (q_dot_u @ un1 + beta)

                # evaluate residuals
                R1 = (
                    M @ (un1 - un)
                    - dt * self.system.h(tm, qm, 0.5 * (un + un1))
                    - W @ Pin1
                )
                R2 = self.c(tn1, qn1, un1, Pin1 / dt)
                R = np.concatenate((R1, R2))

                # error and convergence check
                error = np.linalg.norm(R / scale) / scale.size**0.5
                converged = error < 1
                print(f"i: {i}; error: {error}; converged: {converged}")
                if converged:
                    break

            if not converged:
                warnings.warn(
                    f"Newton method is not converged after {i} iterations with error {error:.2e}"
                )

            # TODO: Only used when continue_with_unconverged is implemented
            # nit = i + 1

        # modify converged quantities
        qn1, un1 = self.system.step_callback(tn1, qn1, un1)

        # unpack percussion
        Pi_gn1, Pi_gamman1, Pi_cn1 = np.array_split(Pin1, self.split_la)

        # store solution fields
        self.sol_t.append(tn1)
        self.sol_q.append(qn1)
        self.sol_u.append(un1)
        self.sol_la_c.append(Pi_cn1 / self.dt)
        self.sol_P_g.append(Pi_gn1)
        self.sol_P_gamma.append(Pi_gamman1)
        # self.sol_P_N.append(dP_Nn1)
        # self.sol_P_F.append(dP_Fn1)
        self.sol_P_N.append(0 * self.la_Nn)
        self.sol_P_F.append(0 * self.la_Fn)

        # update local variables for accepted time step
        self.tn = tn1
        self.qn = qn1.copy()
        self.un = un1.copy()
        # not used and seems to be a bad initial guess for the next iteration
        # self.la_gn = Pi_gn1.copy() / dt
        # self.la_gamman = Pi_gamman1.copy() / dt
        # self.la_cn = Pi_cn1.copy() / dt

    def solve(self):
        self.solver_summary = SolverSummary("MoreauThetaCompliance")

        # lists storing output variables
        self.sol_t = [self.tn]
        self.sol_q = [self.qn]
        self.sol_u = [self.un]
        self.sol_la_c = [self.la_cn]
        self.sol_P_g = [self.dt * self.la_gn]
        self.sol_P_gamma = [self.dt * self.la_gamman]
        self.sol_P_N = [self.dt * self.la_Nn]
        self.sol_P_F = [self.dt * self.la_Fn]

        self.pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in self.pbar:
            # self._step_naive_old()
            # self._step_naive()
            self._step_schur()

        self.solver_summary.print()

        # write solution
        return self._make_solution()
