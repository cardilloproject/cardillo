# -----------------------------------------------------------------------
# A nonsmooth generalized-alpha method for mechanical systems
# with frictional contact
#
# Giuseppe Capobianco, Jonas Harsch, Simon R. Eugster, Remco I. Leine
# -----------------------------------------------------------------------
# Int J Numer Methods Eng. 2021; 1– 30. https://doi.org/10.1002/nme.6801
# -----------------------------------------------------------------------
#
# This file implements the generalized-alpha method as described in our
# paper. All equation numbers found in the comments refer to the paper.
# For this implementation, we chose readability over performance, i.e,
# we aimed at an implementation that is as close as possible to the
# equations found in our paper.
#
# Stuttgart, September 2021                      G.Capobianco, J. Harsch

import numpy as np
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from cardillo.math import prox_R0_nm, prox_R0_np, prox_sphere, approx_fprime, fsolve
from cardillo.solver import Solution


class NonsmoothGeneralizedAlpha:
    """Generalized-alpha solver for mechanical systems with frictional contact."""

    def __init__(
        self,
        system,
        t1,
        dt,
        rho_inf=0.9,
        method="newton",
        newton_tol=1e-6,
        newton_max_iter=100,
        error_function=lambda x: np.max(np.abs(x)),
        fixed_point_tol=1e-6,
        fixed_point_max_iter=1000,
    ):
        self.system = system

        # initial time, final time, time step
        self.t0 = t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        # eqn. (72): parameters
        self.rho_inf = rho_inf
        self.alpha_m = (2 * rho_inf - 1) / (rho_inf + 1)
        self.alpha_f = rho_inf / (rho_inf + 1)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m
        self.beta = 0.25 * (0.5 + self.gamma) ** 2

        # newton settings
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.newton_error_function = error_function

        # fixed point settings
        self.fixed_point_tol = fixed_point_tol
        self.fixed_point_max_iter = fixed_point_max_iter
        self.fixed_point_error_function = error_function

        # dimensions (nq = number of coordinates q, etc.)
        self.nq = system.nq
        self.nu = system.nu
        self.nla_g = system.nla_g
        self.nla_gamma = system.nla_gamma
        self.nla_N = system.nla_N
        self.nla_F = system.nla_F

        # eqn. (127): dimensions of residual
        self.nR_s = 3 * self.nu + 3 * self.nla_g + 2 * self.nla_gamma
        self.nR_c = 3 * self.nla_N + 2 * self.nla_F
        self.nR = self.nR_s + self.nR_c

        # initial conditions
        self.ti = system.t0
        self.qi = system.q0
        self.ui = system.u0
        self.q_doti = system.q_dot0
        self.ai = system.u_dot0
        self.la_gi = system.la_g0
        self.la_gammai = system.la_gamma0
        self.la_Ni = system.la_N0
        self.la_Fi = system.la_F0

        # other initial conditions
        self.kappa_gi = np.zeros_like(system.la_g0)
        self.La_gi = np.zeros_like(system.la_g0)
        self.La_gammai = np.zeros_like(system.la_gamma0)
        self.kappa_Ni = np.zeros_like(system.la_N0)
        self.La_Ni = np.zeros_like(system.la_N0)
        self.La_Fi = np.zeros_like(system.la_F0)
        self.Qi = np.zeros(self.nu)
        self.Ui = np.zeros(self.nu)

        # initialize auxilary variables
        self.a_bari = self.ai.copy()
        self.la_Nbari = self.la_Ni.copy()
        self.la_Fbari = self.la_Fi.copy()

        # initialize index sets
        self.Ai1 = np.zeros(self.nla_N, dtype=bool)
        self.Bi1 = np.zeros(self.nla_N, dtype=bool)
        self.Ci1 = np.zeros(self.nla_N, dtype=bool)
        self.Di1_st = np.zeros(self.nla_N, dtype=bool)
        self.Ei1_st = np.zeros(self.nla_N, dtype=bool)

        # initialize arrays for splitting operation
        self.split_x = np.array(
            [
                self.nu,
                2 * self.nu,
                3 * self.nu,
                3 * self.nu + self.nla_g,
                3 * self.nu + 2 * self.nla_g,
                3 * (self.nu + self.nla_g),
                3 * (self.nu + self.nla_g) + self.nla_gamma,
                3 * (self.nu + self.nla_g) + 2 * self.nla_gamma,
                3 * (self.nu + self.nla_g) + 2 * self.nla_gamma + self.nla_N,
                3 * (self.nu + self.nla_g) + 2 * (self.nla_gamma + self.nla_N),
                3 * (self.nu + self.nla_g + self.nla_N) + 2 * self.nla_gamma,
                3 * (self.nu + self.nla_g + self.nla_N)
                + 2 * self.nla_gamma
                + self.nla_F,
            ],
            dtype=int,
        )

        self.split_y = np.array(
            [
                self.nu,
                2 * self.nu,
                3 * self.nu,
                3 * self.nu + self.nla_g,
                3 * self.nu + 2 * self.nla_g,
                3 * (self.nu + self.nla_g),
                3 * (self.nu + self.nla_g) + self.nla_gamma,
            ],
            dtype=int,
        )

        self.split_z = np.array(
            [
                self.nla_N,
                2 * self.nla_N,
                3 * self.nla_N,
                3 * self.nla_N + self.nla_F,
            ],
            dtype=int,
        )

        if method == "fixed-point":
            self.step = self.step_fixed_point
            self.max_iter = self.fixed_point_max_iter
        elif method == "newton":
            self.step = self.step_newton
            self.max_iter = self.newton_max_iter

        # function called at the end of each time step. Can for example be
        # used to norm quaternions at the end of each time step.
        if hasattr(system, "step_callback"):
            self.step_callback = system.step_callback
        else:
            self.step_callback = self.__step_callback

    def R(self, x, update_index_set=False):
        """Residual R=(R_s, R_c), see eqn. (127)"""
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        nla_F = self.nla_F
        nR_s = self.nR_s

        mu = self.system.mu
        dt = self.dt
        dt2 = self.dt**2
        ti1 = self.ti + dt

        # eqn. (126): unpack vector x
        (
            ai1,
            Ui1,
            Qi1,
            kappa_gi1,
            La_gi1,
            la_gi1,
            La_gammai1,
            la_gammai1,
            kappa_Ni1,
            La_Ni1,
            la_Ni1,
            La_Fi1,
            la_Fi1,
        ) = np.array_split(x, self.split_x)

        # ----- kinematic variables -----
        # eqn. (71): compute auxiliary acceleration variables
        a_bari1 = (
            self.alpha_f * self.ai
            + (1 - self.alpha_f) * ai1
            - self.alpha_m * self.a_bari
        ) / (1 - self.alpha_m)

        # eqn. (73): velocity update formula
        ui1 = (
            self.ui + dt * ((1 - self.gamma) * self.a_bari + self.gamma * a_bari1) + Ui1
        )

        # eqn. (125): generalized position update formula
        a_beta = (1 - 2 * self.beta) * self.a_bari + 2 * self.beta * a_bari1
        qi1 = (
            self.qi
            + dt * self.system.q_dot(self.ti, self.qi, self.ui)
            + dt2 / 2 * self.system.q_ddot(self.ti, self.qi, self.ui, a_beta)
            + self.system.B(self.ti, self.qi) @ Qi1
        )

        # # TODO: Add this to all updates of generalized coordinates
        # # GAMM2022 Harsch
        # Delta_u = self.ui + dt * ((0.5 - self.beta) * self.a_bari + self.beta * a_bari1)
        # qi1 = (
        #     self.qi
        #     + dt * self.model.q_dot(self.ti, self.qi, Delta_u)
        #     + self.model.B(self.ti, self.qi) @ Qi1
        # )

        # ----- normal contact forces -----
        # eqn. (96): compute auxiliary normal contact forces
        la_Nbari1 = (
            self.alpha_f * self.la_Ni
            + (1 - self.alpha_f) * la_Ni1
            - self.alpha_m * self.la_Nbari
        ) / (1 - self.alpha_m)

        # eqn. (95): compute normal percussions
        P_Ni1 = La_Ni1 + dt * (
            (1 - self.gamma) * self.la_Nbari + self.gamma * la_Nbari1
        )

        # eqn. (102):
        kappa_hatNi1 = kappa_Ni1 + dt2 * (
            (0.5 - self.beta) * self.la_Nbari + self.beta * la_Nbari1
        )

        # ----- frictions forces -----
        # eqn. (114): compute auxiliary friction forces
        la_Fbari1 = (
            self.alpha_f * self.la_Fi
            + (1 - self.alpha_f) * la_Fi1
            - self.alpha_m * self.la_Fbari
        ) / (1 - self.alpha_m)

        # eqn. (113): compute frictional percussions
        P_Fi1 = La_Fi1 + dt * (
            (1 - self.gamma) * self.la_Fbari + self.gamma * la_Fbari1
        )

        # ----- get quantities from model -----
        # Mass matrix
        Mi1 = self.system.M(ti1, qi1)

        # generalized force directions
        W_gi1 = self.system.W_g(ti1, qi1)
        W_gammai1 = self.system.W_gamma(ti1, qi1)
        W_Ni1 = self.system.W_N(ti1, qi1)
        W_Fi1 = self.system.W_F(ti1, qi1)

        # kinematic quantities of contacts
        g_Ni1 = self.system.g_N(ti1, qi1)
        xi_Ni1 = self.system.xi_N(ti1, qi1, self.ui, ui1)
        xi_Fi1 = self.system.xi_F(ti1, qi1, self.ui, ui1)
        g_N_ddoti1 = self.system.g_N_ddot(ti1, qi1, ui1, ai1)
        gamma_Fi1 = self.system.gamma_F(ti1, qi1, ui1)
        gamma_F_doti1 = self.system.gamma_F_dot(ti1, qi1, ui1, ai1)

        # ----- compute residual -----
        R = np.zeros(self.nR)

        # eqn. (127): R_s
        R[: self.split_x[0]] = (
            Mi1 @ ai1
            - self.system.h(ti1, qi1, ui1)
            - W_gi1 @ la_gi1
            - W_gammai1 @ la_gammai1
            - W_Ni1 @ la_Ni1
            - W_Fi1 @ la_Fi1
        )
        R[self.split_x[0] : self.split_x[1]] = (
            Mi1 @ Ui1
            - W_gi1 @ La_gi1
            - W_gammai1 @ La_gammai1
            - W_Ni1 @ La_Ni1
            - W_Fi1 @ La_Fi1
        )
        R[self.split_x[1] : self.split_x[2]] = (
            Mi1 @ Qi1
            - W_gi1 @ kappa_gi1
            - W_Ni1 @ kappa_Ni1
            - 0.5 * dt * (W_gammai1 @ La_gammai1 + W_Fi1 @ La_Fi1)
        )
        R[self.split_x[2] : self.split_x[3]] = self.system.g(ti1, qi1)
        R[self.split_x[3] : self.split_x[4]] = self.system.g_dot(ti1, qi1, ui1)
        R[self.split_x[4] : self.split_x[5]] = self.system.g_ddot(ti1, qi1, ui1, ai1)
        R[self.split_x[5] : self.split_x[6]] = self.system.gamma(ti1, qi1, ui1)
        R[self.split_x[6] : self.split_x[7]] = self.system.gamma_dot(ti1, qi1, ui1, ai1)

        # update index sets
        if update_index_set:
            prox_r_N = self.system.prox_r_N(ti1, qi1)
            prox_r_F = self.system.prox_r_F(ti1, qi1)

            # eqn. (130):
            self.Ai1 = prox_r_N * g_Ni1 - kappa_hatNi1 <= 0
            # eqn. (133):
            self.Bi1 = self.Ai1 * ((prox_r_N * xi_Ni1 - P_Ni1) <= 0)
            # eqn. (136):
            self.Ci1 = self.Bi1 * ((prox_r_N * g_N_ddoti1 - la_Ni1) <= 0)

            for i_N, i_F in enumerate(self.system.NF_connectivity):
                i_F = np.array(i_F)
                if len(i_F) > 0:
                    # eqn. (139):
                    self.Di1_st[i_N] = self.Ai1[i_N] and (
                        norm(prox_r_F[i_N] * xi_Fi1[i_F] - P_Fi1[i_F])
                        <= mu[i_N] * P_Ni1[i_N]
                    )
                    # eqn. (141):
                    self.Ei1_st[i_N] = self.Di1_st[i_N] and (
                        norm(prox_r_F[i_N] * gamma_F_doti1[i_F] - la_Fi1[i_F])
                        <= mu[i_N] * la_Ni1[i_N]
                    )

        # eqn. (129):
        Ai1 = self.Ai1
        Ai1_ind = np.where(Ai1)[0]
        _Ai1_ind = np.where(~Ai1)[0]
        R[self.split_x[7] + Ai1_ind] = g_Ni1[Ai1]
        R[self.split_x[7] + _Ai1_ind] = kappa_hatNi1[~Ai1]

        # eqn. (132):
        Bi1 = self.Bi1
        Bi1_ind = np.where(Bi1)[0]
        _Bi1_ind = np.where(~Bi1)[0]
        R[self.split_x[8] + Bi1_ind] = xi_Ni1[Bi1]
        R[self.split_x[8] + _Bi1_ind] = P_Ni1[~Bi1]

        # eqn. (135):
        Ci1 = self.Ci1
        Ci1_ind = np.where(Ci1)[0]
        _Ci1_ind = np.where(~Ci1)[0]
        R[self.split_x[9] + Ci1_ind] = g_N_ddoti1[Ci1]
        R[self.split_x[9] + _Ci1_ind] = la_Ni1[~Ci1]

        # eqn. (138) and (142):
        Di1_st = self.Di1_st
        Ei1_st = self.Ei1_st

        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                if Ai1[i_N]:
                    if Di1_st[i_N]:
                        # eqn. (138a)
                        R[self.split_x[10] + i_F] = xi_Fi1[i_F]

                        if Ei1_st[i_N]:
                            # eqn. (142a)
                            R[self.split_x[11] + i_F] = gamma_F_doti1[i_F]
                        else:
                            # eqn. (142b)
                            norm_gamma_Fdoti1 = norm(gamma_F_doti1[i_F])
                            if norm_gamma_Fdoti1 > 0:
                                R[self.split_x[10] + i_F] = (
                                    la_Fi1[i_F]
                                    + mu[i_N]
                                    * la_Ni1[i_N]
                                    * gamma_F_doti1[i_F]
                                    / norm_gamma_Fdoti1
                                )
                            else:
                                R[self.split_x[11] + i_F] = (
                                    la_Fi1[i_F]
                                    + mu[i_N] * la_Ni1[i_N] * gamma_F_doti1[i_F]
                                )
                    else:
                        # eqn. (138b)
                        norm_xi_Fi1 = norm(xi_Fi1[i_F])
                        if norm_xi_Fi1 > 0:
                            R[self.split_x[10] + i_F] = (
                                P_Fi1[i_F]
                                + mu[i_N] * P_Ni1[i_N] * xi_Fi1[i_F] / norm_xi_Fi1
                            )
                        else:
                            R[self.split_x[10] + i_F] = (
                                P_Fi1[i_F] + mu[i_N] * P_Ni1[i_N] * xi_Fi1[i_F]
                            )

                        # eqn. (142c)
                        norm_gamma_Fi1 = norm(gamma_Fi1[i_F])
                        if norm_gamma_Fi1 > 0:
                            R[self.split_x[11] + i_F] = (
                                la_Fi1[i_F]
                                + mu[i_N]
                                * la_Ni1[i_N]
                                * gamma_Fi1[i_F]
                                / norm_gamma_Fi1
                            )
                        else:
                            R[self.split_x[11] + i_F] = (
                                la_Fi1[i_F] + mu[i_N] * la_Ni1[i_N] * gamma_Fi1[i_F]
                            )
                else:
                    # eqn. (138c)
                    R[self.split_x[10] + i_F] = P_Fi1[i_F]
                    # eqn. (142d)
                    R[self.split_x[11] + i_F] = la_Fi1[i_F]

        return R

    def R_s(self, y, z):
        """Residual R_s, see eqn. (127), as function of y and z given by (144)."""
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        nla_F = self.nla_F

        dt = self.dt
        dt2 = self.dt**2
        ti1 = self.ti + dt

        # eqn. (144): unpack vectors y and z
        (
            ai1,
            Ui1,
            Qi1,
            kappa_gi1,
            La_gi1,
            la_gi1,
            La_gammai1,
            la_gammai1,
        ) = np.array_split(y, self.split_y)
        kappa_Ni1, La_Ni1, la_Ni1, La_Fi1, la_Fi1 = np.array_split(z, self.split_z)

        # ----- kinematic variables -----
        # eqn. (71): compute auxiliary acceleration variables
        a_bari1 = (
            self.alpha_f * self.ai
            + (1 - self.alpha_f) * ai1
            - self.alpha_m * self.a_bari
        ) / (1 - self.alpha_m)

        # eqn. (73): velocity update formula
        ui1 = (
            self.ui + dt * ((1 - self.gamma) * self.a_bari + self.gamma * a_bari1) + Ui1
        )

        # eqn. (125): generalized position update formula
        a_beta = (1 - 2 * self.beta) * self.a_bari + 2 * self.beta * a_bari1
        qi1 = (
            self.qi
            + dt * self.system.q_dot(self.ti, self.qi, self.ui)
            + dt2 / 2 * self.system.q_ddot(self.ti, self.qi, self.ui, a_beta)
            + self.system.B(self.ti, self.qi) @ Qi1
        )

        # ----- get quantities from model -----
        # Mass matrix
        Mi1 = self.system.M(ti1, qi1)

        # generalized force directions
        W_gi1 = self.system.W_g(ti1, qi1)
        W_gammai1 = self.system.W_gamma(ti1, qi1)
        W_Ni1 = self.system.W_N(ti1, qi1)
        W_Fi1 = self.system.W_F(ti1, qi1)

        # ----- compute residual -----
        R_s = np.concatenate(
            (
                Mi1 @ ai1
                - self.system.h(ti1, qi1, ui1)
                - W_gi1 @ la_gi1
                - W_gammai1 @ la_gammai1
                - W_Ni1 @ la_Ni1
                - W_Fi1 @ la_Fi1,
                Mi1 @ Ui1
                - W_gi1 @ La_gi1
                - W_gammai1 @ La_gammai1
                - W_Ni1 @ La_Ni1
                - W_Fi1 @ La_Fi1,
                Mi1 @ Qi1
                - W_gi1 @ kappa_gi1
                - W_Ni1 @ kappa_Ni1
                - 0.5 * dt * (W_gammai1 @ La_gammai1 + W_Fi1 @ La_Fi1),
                self.system.g(ti1, qi1),
                self.system.g_dot(ti1, qi1, ui1),
                self.system.g_ddot(ti1, qi1, ui1, ai1),
                self.system.gamma(ti1, qi1, ui1),
                self.system.gamma_dot(ti1, qi1, ui1, ai1),
            )
        )

        return R_s

    def p(self, y, z):
        """map p(y,z) used in (146)."""
        nu = self.nu
        nla_N = self.nla_N
        nla_F = self.nla_F

        mu = self.system.mu

        dt = self.dt
        dt2 = self.dt**2
        ti1 = self.ti + dt

        # eqn. (144): read kinematic variables from y
        ai1, Ui1, Qi1, _, _, _, _, _ = np.array_split(y, self.split_y)

        # ----- kinematic variables -----
        # eqn. (71): compute auxiliary acceleration variables
        a_bari1 = (
            self.alpha_f * self.ai
            + (1 - self.alpha_f) * ai1
            - self.alpha_m * self.a_bari
        ) / (1 - self.alpha_m)

        # eqn. (73): velocity update formula
        ui1 = (
            self.ui + dt * ((1 - self.gamma) * self.a_bari + self.gamma * a_bari1) + Ui1
        )

        # eqn. (125): generalized position update formula
        a_beta = (1 - 2 * self.beta) * self.a_bari + 2 * self.beta * a_bari1
        qi1 = (
            self.qi
            + dt * self.system.q_dot(self.ti, self.qi, self.ui)
            + dt2 / 2 * self.system.q_ddot(self.ti, self.qi, self.ui, a_beta)
            + self.system.B(self.ti, self.qi) @ Qi1
        )

        # ----- get quantities from model -----
        # kinematic quantities of contacts
        g_N = self.system.g_N(ti1, qi1)
        xi_N = self.system.xi_N(ti1, qi1, self.ui, ui1)
        g_N_ddot = self.system.g_N_ddot(ti1, qi1, ui1, ai1)
        gamma_F = self.system.gamma_F(ti1, qi1, ui1)
        xi_F = self.system.xi_F(ti1, qi1, self.ui, ui1)
        gamma_F_dot = self.system.gamma_F_dot(ti1, qi1, ui1, ai1)

        # ----- eqn. (146): fixed point update -----
        # For convenience, we call the iteration index j instead of mu.

        # eqn. (144): unpack vector z
        kappa_Ni1_j, La_Ni1_j, la_Ni1_j, La_Fi1_j, la_Fi1_j = np.array_split(
            z, self.split_z
        )

        # eqn. (96): compute auxiliary normal contact forces
        la_Nbari1 = (
            self.alpha_f * self.la_Ni
            + (1 - self.alpha_f) * la_Ni1_j
            - self.alpha_m * self.la_Nbari
        ) / (1 - self.alpha_m)
        # eqn. (95): compute normal percussions
        P_N_j = La_Ni1_j + dt * (
            (1 - self.gamma) * self.la_Nbari + self.gamma * la_Nbari1
        )
        # eqn. (102):
        kappa_hatN_j = kappa_Ni1_j + dt2 * (
            (0.5 - self.beta) * self.la_Nbari + self.beta * la_Nbari1
        )

        # eqn. (114): compute auxiliary friction forces
        la_Fbari1 = (
            self.alpha_f * self.la_Fi
            + (1 - self.alpha_f) * la_Fi1_j
            - self.alpha_m * self.la_Fbari
        ) / (1 - self.alpha_m)
        # eqn. (113): compute frictional percussions
        P_F_j = La_Fi1_j + dt * (
            (1 - self.gamma) * self.la_Fbari + self.gamma * la_Fbari1
        )

        # evaluate prox parameters
        prox_r_N = self.system.prox_r_N(ti1, qi1)
        prox_r_F = self.system.prox_r_F(ti1, qi1)

        # -- prox normal direction --
        P_N_j1 = np.zeros(self.nla_N)
        la_Ni1_j1 = np.zeros(self.nla_N)

        # eqn. (128):
        prox_arg = prox_r_N * g_N - kappa_hatN_j
        kappa_hatN_j1 = -prox_R0_nm(prox_arg)
        # eqn. (130):
        Ai1 = prox_arg <= 0

        # eqn. (131):
        prox_arg = prox_r_N * xi_N - P_N_j
        P_N_j1[Ai1] = -prox_R0_nm(prox_arg[Ai1])
        # eqn. (133):
        Bi1 = (prox_arg <= 0) * Ai1

        # eqn. (134):
        la_Ni1_j1[Bi1] = -prox_R0_nm(prox_r_N[Bi1] * g_N_ddot[Bi1] - la_Ni1_j[Bi1])

        # -- prox friction --
        P_F_j1 = np.zeros(self.nla_F)
        la_Fi1_j1 = np.zeros(self.nla_F)

        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)
            if Ai1[i_N]:
                # eqn. (137):
                prox_arg = prox_r_F[i_N] * xi_F[i_F] - P_F_j[i_F]
                radius = mu[i_N] * P_N_j1[i_N]
                P_F_j1[i_F] = -prox_sphere(prox_arg, radius)

                # eqn. (139): if contact index is in D_st
                if norm(prox_arg) <= radius:
                    # eqn. (140a):
                    prox_arg_acc = prox_r_F[i_N] * gamma_F_dot[i_F] - la_Fi1_j[i_F]
                    radius_acc = mu[i_N] * la_Ni1_j1[i_N]
                    la_Fi1_j1[i_F] = -prox_sphere(prox_arg_acc, radius_acc)
                else:
                    # eqn. (140b):
                    norm_gamma_F = norm(gamma_F[i_F])
                    if norm_gamma_F > 0:
                        la_Fi1_j1[i_F] = (
                            -mu[i_N] * la_Ni1_j1[i_N] * gamma_F[i_F] / norm_gamma_F
                        )
                    else:
                        la_Fi1_j1[i_F] = -mu[i_N] * la_Ni1_j1[i_N] * gamma_F[i_F]

        # -- update contact forces --
        # eqn. (96): compute auxiliary normal contact forces
        la_Nbari1 = (
            self.alpha_f * self.la_Ni
            + (1 - self.alpha_f) * la_Ni1_j1
            - self.alpha_m * self.la_Nbari
        ) / (1 - self.alpha_m)
        # eqn. (95): compute normal percussions
        La_Ni1_j1 = P_N_j1 - dt * (
            (1 - self.gamma) * self.la_Nbari + self.gamma * la_Nbari1
        )
        # eqn. (102):
        kappa_Ni1_j1 = kappa_hatN_j1 - dt2 * (
            (0.5 - self.beta) * self.la_Nbari + self.beta * la_Nbari1
        )

        # eqn. (114): compute auxiliary friction forces
        la_Fbari1 = (
            self.alpha_f * self.la_Fi
            + (1 - self.alpha_f) * la_Fi1_j1
            - self.alpha_m * self.la_Fbari
        ) / (1 - self.alpha_m)
        # eqn. (113): compute frictional percussions
        La_Fi1_j1 = P_F_j1 - dt * (
            (1 - self.gamma) * self.la_Fbari + self.gamma * la_Fbari1
        )

        # eqn. (144): pack z vector
        z = np.concatenate((kappa_Ni1_j1, La_Ni1_j1, la_Ni1_j1, La_Fi1_j1, la_Fi1_j1))

        return z

    def step_newton(self):
        """Time step with semismooth Newton method"""
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        nla_F = self.nla_F
        nR_s = self.nR_s
        dt = self.dt
        ti1 = self.ti + dt

        # eqn. (126): initialize x vector with quanitites from previous time step
        x = np.concatenate(
            (
                self.ai,
                self.Ui,
                self.Qi,
                self.kappa_gi,
                self.La_gi,
                self.la_gi,
                self.La_gammai,
                self.la_gammai,
                self.kappa_Ni,
                self.La_Ni,
                self.la_Ni,
                self.La_Fi,
                self.la_Fi,
            )
        )
        # initial residual and error
        R = self.R(x, update_index_set=True)
        error = self.newton_error_function(R)
        converged = error < self.newton_tol
        j = 0
        # iterate Newton update until converged or max_iter reached
        if not converged:
            for j in range(self.newton_max_iter):
                # jacobian
                R_x = csc_matrix(approx_fprime(x, self.R, method="2-point", eps=1.0e-6))

                # eqn. (143): Newton update
                try:
                    x -= spsolve(R_x, R)
                except:
                    print(f"Failed to invert R at time t = {ti1}.")
                    converged = False
                    break

                R = self.R(x, update_index_set=True)
                error = self.newton_error_function(R)
                converged = error < self.newton_tol
                if converged:
                    break

        # eqn. (126): unpack converged vector x
        (
            ai1,
            Ui1,
            Qi1,
            kappa_gi1,
            La_gi1,
            la_gi1,
            La_gammai1,
            la_gammai1,
            kappa_Ni1,
            La_Ni1,
            la_Ni1,
            La_Fi1,
            la_Fi1,
        ) = np.array_split(x, self.split_x)

        return (
            (converged, j, error),
            ti1,
            ai1,
            Ui1,
            Qi1,
            kappa_gi1,
            La_gi1,
            la_gi1,
            La_gammai1,
            la_gammai1,
            kappa_Ni1,
            La_Ni1,
            la_Ni1,
            La_Fi1,
            la_Fi1,
        )

    def step_fixed_point(self):
        """Time step with fixed point iterations"""
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        nla_F = self.nla_F
        nR_s = self.nR_s
        dt = self.dt
        ti1 = self.ti + dt

        # eqn. (144): initialize split variables
        y = np.concatenate(
            (
                self.ai,
                self.Ui,
                self.Qi,
                self.kappa_gi,
                self.La_gi,
                self.la_gi,
                self.La_gammai,
                self.la_gammai,
            )
        )
        z = np.concatenate(
            (
                self.kappa_Ni,
                self.La_Ni,
                self.la_Ni,
                self.La_Fi,
                self.la_Fi,
            )
        )

        # eqn. (145): Newton iterations for update of non-contact variables
        fixed_point_error = None
        fixed_point_converged = False
        j = 0
        for j in range(self.fixed_point_max_iter):
            R_s = self.R_s(y, z)
            newton_error = self.newton_error_function(R_s)
            newton_converged = newton_error < self.newton_tol
            if not newton_converged:
                for _ in range(self.newton_max_iter):
                    # jacobian
                    R_s_y = csc_matrix(
                        approx_fprime(
                            y, lambda y: self.R_s(y, z), method="2-point", eps=1.0e-6
                        )
                    )

                    # Newton update
                    y -= spsolve(R_s_y, R_s)

                    R_s = self.R_s(y, z)
                    newton_error = self.newton_error_function(R_s)
                    newton_converged = newton_error < self.newton_tol
                    if newton_converged:
                        break
                if not newton_converged:
                    raise RuntimeError(
                        f"Newton method in {j}-th fixed-point iteration not converged."
                    )

            # eqn. (146): fixed point update
            z1 = self.p(y, z)
            fixed_point_error = self.fixed_point_error_function(z1 - z)
            fixed_point_converged = fixed_point_error < self.fixed_point_tol
            z = z1.copy()

            if fixed_point_converged:
                break

        # eqn. (144): unpack converged y and z vectors
        (
            ai1,
            Ui1,
            Qi1,
            kappa_gi1,
            La_gi1,
            la_gi1,
            La_gammai1,
            la_gammai1,
        ) = np.array_split(y, self.split_y)
        kappa_Ni1, La_Ni1, la_Ni1, La_Fi1, la_Fi1 = np.array_split(z, self.split_z)

        return (
            (fixed_point_converged, j, fixed_point_error),
            ti1,
            ai1,
            Ui1,
            Qi1,
            kappa_gi1,
            La_gi1,
            la_gi1,
            La_gammai1,
            la_gammai1,
            kappa_Ni1,
            La_Ni1,
            la_Ni1,
            La_Fi1,
            la_Fi1,
        )

    def solve(self):
        """Method that runs the solver"""
        dt = self.dt
        dt2 = self.dt**2

        # lists storing output variables
        t = [self.ti]
        q = [self.qi]
        u = [self.ui]
        a = [self.ai]
        Q = [self.Qi]
        U = [self.Ui]
        kappa_g = [self.kappa_gi]
        La_g = [self.La_gi]
        la_g = [self.la_gi]
        La_gamma = [self.La_gammai]
        la_gamma = [self.la_gammai]
        kappa_N = [self.kappa_Ni]
        La_N = [self.La_Ni]
        la_N = [self.la_Ni]
        La_F = [self.La_Fi]
        la_F = [self.la_Fi]
        P_N = [self.La_Ni + dt * self.la_Ni]
        P_F = [self.La_Fi + dt * self.la_Fi]

        # initialize progress bar
        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))

        iter = []
        fixpt_iter = []
        # for-loop over all time steps
        for _ in pbar:
            # try to solve time step with user-defined method (method='newton' if unspecified)
            try:
                (
                    (converged, n_iter, error),
                    ti1,
                    ai1,
                    Ui1,
                    Qi1,
                    kappa_gi1,
                    La_gi1,
                    la_gi1,
                    La_gammai1,
                    la_gammai1,
                    kappa_Ni1,
                    La_Ni1,
                    la_Ni1,
                    La_Fi1,
                    la_Fi1,
                ) = self.step()
                pbar.set_description(
                    f"t: {ti1:0.2e}s < {self.t1:0.2e}s; {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
                )
                if not converged:
                    raise RuntimeError(
                        f"step not converged after {n_iter} steps with error: {error:.5e}"
                    )
                iter.append(n_iter + 1)
            except (
                RuntimeError
            ):  # if method specified does not converge, use fixed-point iterations in time step.
                print("\nSwitched to fixed-point step.\n")
                (
                    (converged, n_iter, error),
                    ti1,
                    ai1,
                    Ui1,
                    Qi1,
                    kappa_gi1,
                    La_gi1,
                    la_gi1,
                    La_gammai1,
                    la_gammai1,
                    kappa_Ni1,
                    La_Ni1,
                    la_Ni1,
                    La_Fi1,
                    la_Fi1,
                ) = self.step_fixed_point()
                pbar.set_description(
                    f"t: {ti1:0.2e}s < {self.t1:0.2e}s; {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
                )
                if not converged:
                    raise RuntimeError(
                        f"fixed-point step not converged after {n_iter} steps with error: {error:.5e}"
                    )
                fixpt_iter.append(n_iter + 1)

            # ----- compute variables for output -----

            # eqn. (71): compute auxiliary acceleration variables
            a_bari1 = (
                self.alpha_f * self.ai
                + (1 - self.alpha_f) * ai1
                - self.alpha_m * self.a_bari
            ) / (1 - self.alpha_m)

            # eqn. (73): velocity update formula
            ui1 = (
                self.ui
                + dt * ((1 - self.gamma) * self.a_bari + self.gamma * a_bari1)
                + Ui1
            )

            # eqn. (125): generalized position update formula
            a_beta = (1 - 2 * self.beta) * self.a_bari + 2 * self.beta * a_bari1
            qi1 = (
                self.qi
                + dt * self.system.q_dot(self.ti, self.qi, self.ui)
                + dt2 / 2 * self.system.q_ddot(self.ti, self.qi, self.ui, a_beta)
                + self.system.B(self.ti, self.qi) @ Qi1
            )

            # eqn. (96): compute auxiliary normal contact forces
            la_Nbari1 = (
                self.alpha_f * self.la_Ni
                + (1 - self.alpha_f) * la_Ni1
                - self.alpha_m * self.la_Nbari
            ) / (1 - self.alpha_m)

            # eqn. (95): compute normal percussions
            P_Ni1 = La_Ni1 + dt * (
                (1 - self.gamma) * self.la_Nbari + self.gamma * la_Nbari1
            )

            # eqn. (114): compute auxiliary friction forces
            la_Fbari1 = (
                self.alpha_f * self.la_Fi
                + (1 - self.alpha_f) * la_Fi1
                - self.alpha_m * self.la_Fbari
            ) / (1 - self.alpha_m)

            # eqn. (113): compute frictional percussions
            P_Fi1 = La_Fi1 + dt * (
                (1 - self.gamma) * self.la_Fbari + self.gamma * la_Fbari1
            )

            # function called at the end of each time step. Can for example be used to norm quaternions at the end of each time step.
            qi1, ui1 = self.step_callback(ti1, qi1, ui1)

            # append solution of time step to global output vectors
            t.append(ti1)
            q.append(qi1)
            u.append(ui1)
            a.append(ai1)
            Q.append(Qi1)
            U.append(Ui1)
            kappa_g.append(kappa_gi1)
            La_g.append(La_gi1)
            la_g.append(la_gi1)
            La_gamma.append(La_gammai1)
            la_gamma.append(la_gammai1)
            kappa_N.append(kappa_Ni1)
            La_N.append(La_Ni1)
            la_N.append(la_Ni1)
            La_F.append(La_Fi1)
            la_F.append(la_Fi1)
            P_N.append(P_Ni1)
            P_F.append(P_Fi1)

            # update local variables for accepted time step
            self.ti = ti1
            self.qi = qi1
            self.ui = ui1
            self.ai = ai1
            self.Qi = Qi1
            self.kappa_gi = kappa_gi1
            self.La_gi = La_gi1
            self.la_gi = la_gi1
            self.La_gammai = La_gammai1
            self.la_gammai = la_gammai1
            self.kappa_Ni = kappa_Ni1
            self.La_Ni = La_Ni1
            self.la_Ni = la_Ni1
            self.La_Fi = La_Fi1
            self.la_Fi = la_Fi1
            self.a_bari = a_bari1
            self.la_Nbari = la_Nbari1
            self.la_Fbari = la_Fbari1

        # print statistics
        print("-----------------")
        print(
            f"Iterations per time step: max = {max(iter)}, avg={sum(iter) / float(len(iter))}"
        )
        if len(fixpt_iter) > 0:
            print("-----------------")
            print("For the time steps, where primary method did not converge:")
            print(f"Number of such time steps: {len(fixpt_iter)}")
            print(
                f"Fixed-point iterations: max = {max(fixpt_iter)}, avg={sum(fixpt_iter) / float(len(fixpt_iter))}"
            )
        print("-----------------")
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            a=np.array(a),
            Q=np.array(Q),
            U=np.array(U),
            kappa_g=np.array(kappa_g),
            La_g=np.array(La_g),
            la_g=np.array(la_g),
            La_gamma=np.array(La_gamma),
            la_gamma=np.array(la_gamma),
            kappa_N=np.array(kappa_N),
            La_N=np.array(La_N),
            la_N=np.array(la_N),
            La_F=np.array(La_F),
            la_F=np.array(la_F),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
        )


class SimplifiedNonsmoothGeneralizedAlpha:
    """Simplified version of the nonsmooth generalized-alpha solver for
    mechanical systems with frictional contact presented in Capobianco2020.

    References
    ----------
    Capobianco2020: https://doi.org/10.1002/nme.6801
    """

    def __init__(
        self,
        system,
        t1,
        h,
        rho_inf=0.8,
        atol=1e-6,
        max_iter=10,
    ):
        self.system = system

        # initial time, final time, time step
        self.t0 = t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.t = np.arange(t0, t1 + h, h)
        self.h = h

        # eqn. (72): parameters
        self.rho_inf = rho_inf
        self.alpha_m = (2 * rho_inf - 1) / (rho_inf + 1)
        self.alpha_f = rho_inf / (rho_inf + 1)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m
        self.beta = 0.25 * (0.5 + self.gamma) ** 2

        # newton settings
        self.atol = atol
        self.max_iter = max_iter

        # dimensions (nq = number of coordinates q, etc.)
        self.nq = system.nq
        self.nu = system.nu
        self.nla_g = system.nla_g
        self.nla_gamma = system.nla_gamma
        self.nla_N = system.nla_N
        self.nla_F = system.nla_F

        self.split_y = np.cumsum(
            np.array(
                [
                    self.nu,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_N,
                    self.nla_F,
                    self.nu,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_N,
                ],
                dtype=int,
            )
        )

        # initial conditions
        self.tn = system.t0
        self.qn = system.q0
        self.un = system.u0
        self.q_dotn = system.q_dot0
        self.u_dotn = system.u_dot0
        self.la_gn = system.la_g0
        self.la_gamman = system.la_gamma0
        self.la_Nn = system.la_N0
        self.la_Fn = system.la_F0
        self.R_Nn = self.la_Nn
        self.R_Fn = self.la_Fn

        # initialize auxilary variables
        self.an = self.u_dotn.copy()
        self.R_N_barn = self.R_Nn.copy()
        self.R_F_barn = self.R_Fn.copy()
        self.P_Nn = h * self.R_Nn.copy()
        self.P_Fn = h * self.R_Fn.copy()

        # initialize y vector of unknowns
        self.yn = np.concatenate(
            (
                self.u_dotn,
                h * self.la_gn,
                h * self.la_gamman,
                self.R_Nn,
                self.R_Fn,
                np.zeros(self.nu),
                0 * self.la_gn,
                0 * self.la_gamman,
                0 * self.R_Nn,
                0 * self.R_Fn,
            )
        )

        # initialize index sets
        self.I = np.zeros(self.nla_N, dtype=bool)
        self.B = np.zeros(self.nla_N, dtype=bool)

    def _update(self, y, store=False):
        h = self.h
        tn1 = self.tn + h
        gamma = self.gamma
        beta = self.beta
        alpha_f = self.alpha_f
        alpha_m = self.alpha_m

        # unpack unknowns
        (
            u_dotn1,
            R_gn1,
            R_gamman1,
            R_Nn1,
            R_Fn1,
            Delta_U,
            Delta_R_gn2,
            Delta_R_gamman2,
            Delta_R_Nn2,
            Delta_R_Fn2,
        ) = np.array_split(y, self.split_y)

        # eqn. (71): compute new auxiliary variables
        an1 = (alpha_f * self.u_dotn + (1 - alpha_f) * u_dotn1 - alpha_m * self.an) / (
            1 - alpha_m
        )
        R_N_barn1 = (
            alpha_f * self.R_Nn + (1 - alpha_f) * R_Nn1 - alpha_m * self.R_N_barn
        ) / (1 - alpha_m)
        R_F_barn1 = (
            alpha_f * self.R_Fn + (1 - alpha_f) * R_Fn1 - alpha_m * self.R_F_barn
        ) / (1 - alpha_m)

        # eqn. (73): velocity update formula
        un12 = self.un + h * ((1 - gamma) * self.an + gamma * an1)
        un1 = un12 + Delta_U
        # P_Nn12 = self.P_Nn + h * ((1 - gamma) * self.R_N_barn + gamma * R_N_barn1)
        P_Nn12 = h * ((1 - gamma) * self.R_N_barn + gamma * R_N_barn1)
        P_Nn1 = P_Nn12 + Delta_R_Nn2
        # P_Fn12 = self.P_Fn + h * ((1 - gamma) * self.R_F_barn + gamma * R_F_barn1)
        P_Fn12 = h * ((1 - gamma) * self.R_F_barn + gamma * R_F_barn1)
        P_Fn1 = P_Fn12 + Delta_R_Fn2

        # # position update Capobianco2021
        # a_beta = (1 - 2 * beta) * self.an + 2 * beta * an1
        # qn1 = (
        #     self.qn
        #     + h * self.system.q_dot(self.tn, self.qn, self.un)
        #     + h**2 / 2 * self.system.q_ddot(self.tn, self.qn, self.un, a_beta)
        # )

        # position update, see GAMM2022 Harsch
        Delta_un12 = self.un + h * ((0.5 - beta) * self.an + beta * an1)
        qn1 = self.qn + h * self.system.q_dot(self.tn, self.qn, Delta_un12)

        if store:
            # update local variables for accepted time step
            self.tn = tn1
            self.qn = qn1.copy()
            self.un = un1.copy()
            self.u_dotn = u_dotn1.copy()
            self.an = an1.copy()
            self.R_Nn = R_Nn1.copy()
            self.R_N_barn = R_N_barn1.copy()
            # self.P_Nn = P_Nn1.copy()
            self.R_Fn = R_Fn1.copy()
            self.R_F_barn = R_F_barn1.copy()
            # self.P_Fn = P_Fn1.copy()

        return (
            tn1,
            qn1,
            un12,
            un1,
            u_dotn1,
            R_gn1,
            R_gamman1,
            R_Nn1,
            R_Fn1,
            Delta_U,
            Delta_R_gn2,
            Delta_R_gamman2,
            Delta_R_Nn2,
            Delta_R_Fn2,
            P_Nn12,
            P_Fn12,
            P_Nn1,
            P_Fn1,
        )

    def _R(self, y, update_index=False):
        (
            tn1,
            qn1,
            un12,
            un1,
            u_dotn1,
            R_gn1,
            R_gamman1,
            R_Nn1,
            R_Fn1,
            Delta_U,
            Delta_R_gn2,
            Delta_R_gamman2,
            Delta_R_Nn2,
            Delta_R_Fn2,
            P_Nn12,
            P_Fn12,
            P_Nn1,
            P_Fn1,
        ) = self._update(y)

        R = np.zeros_like(y)
        R = y.copy()

        ####################
        # eqations of motion
        ####################
        M = self.system.M(tn1, qn1)
        W_g = self.system.W_g(tn1, qn1)
        W_gamma = self.system.W_gamma(tn1, qn1)
        W_N = self.system.W_N(tn1, qn1)
        W_F = self.system.W_F(tn1, qn1)
        R[: self.split_y[0]] = (
            M @ u_dotn1
            - self.system.h(tn1, qn1, un12)
            # - self.system.h(tn1, qn1, un1)
            - W_g @ R_gn1
            - W_gamma @ R_gamman1
            - W_N @ R_Nn1
            - W_F @ R_Fn1
        )

        #######################
        # bilateral constraints
        #######################
        R[self.split_y[0] : self.split_y[1]] = self.system.g(tn1, qn1)
        R[self.split_y[1] : self.split_y[2]] = self.system.gamma(tn1, qn1, un12)

        ###########
        # Signorini
        ###########
        g_N = self.system.g_N(tn1, qn1)
        prox_arg = g_N - self.prox_r_N * P_Nn12

        if update_index:
            self.I = prox_arg <= 0.0

        R[self.split_y[2] : self.split_y[3]] = np.where(
            self.I,
            g_N,
            P_Nn12,
        )

        ##################
        # Coulomb friction
        ##################
        prox_r_F = self.prox_r_F
        mu = self.system.mu
        gamma_F = self.system.gamma_F(tn1, qn1, un12)

        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                R[self.split_y[3] + i_F] = P_Fn12[i_F] + prox_sphere(
                    prox_r_F[i_N] * gamma_F[i_F] - P_Fn12[i_F],
                    mu[i_N] * P_Nn12[i_N],
                )

        #################
        # impact equation
        #################
        R[self.split_y[4] : self.split_y[5]] = (
            M @ Delta_U
            - W_g @ Delta_R_gn2
            - W_gamma @ Delta_R_gamman2
            - W_N @ Delta_R_Nn2
            - W_F @ Delta_R_Fn2
        )

        #################################
        # impulsive bilateral constraints
        #################################
        R[self.split_y[5] : self.split_y[6]] = self.system.g_dot(tn1, qn1, un1)
        R[self.split_y[6] : self.split_y[7]] = self.system.gamma(tn1, qn1, un1)

        ##################################################
        # mixed Singorini on velocity level and impact law
        ##################################################
        xi_Nn1 = self.system.xi_N(tn1, qn1, self.un, un1)
        prox_arg = xi_Nn1 - self.prox_r_N * P_Nn1
        if update_index:
            self.B = self.I * (prox_arg <= 0)

        R[self.split_y[7] : self.split_y[8]] = np.where(
            self.B,
            xi_Nn1,
            P_Nn1,
        )

        # R[self.split_y[7] : self.split_y[8]] = np.where(
        #     self.I,
        #     xi_Nn1 - prox_R0_np(xi_Nn1 - self.prox_r_N * P_Nn1),
        #     P_Nn1,
        # )

        ##################################################
        # mixed Coulomb friction and tangent impact law
        ##################################################
        xi_Fn1 = self.system.xi_F(tn1, qn1, self.un, un1)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                R[self.split_y[8] + i_F] = P_Fn1[i_F] + prox_sphere(
                    prox_r_F[i_N] * xi_Fn1[i_F] - P_Fn1[i_F],
                    mu[i_N] * P_Nn1[i_N],
                )

        return R

    def solve(self):
        # lists storing output variables
        sol_q = [self.qn]
        sol_u = [self.un]
        sol_P_g = [self.h * self.la_gn]
        sol_P_gamma = [self.h * self.la_gamman]
        sol_P_N = [self.h * self.la_Nn]
        sol_P_F = [self.h * self.la_Fn]

        # initialize progress bar
        pbar = tqdm(self.t[1:])

        iter = []
        for _ in pbar:
            # only compute optimized proxparameters once per time step
            self.prox_r_N = self.system.prox_r_N(self.tn, self.qn)
            self.prox_r_F = self.system.prox_r_F(self.tn, self.qn)
            # print(f"self.prox_r_N: {self.prox_r_N}")
            # print(f"self.prox_r_F: {self.prox_r_F}")
            # self.prox_r_N = np.ones(self.nla_N) * 0.2
            # self.prox_r_F = np.ones(self.nla_F) * 0.2

            yn1, converged, error, n_iter, _ = fsolve(
                self._R,
                self.yn,
                jac="2-point",
                eps=1.0e-6,
                fun_args=(True,),
                jac_args=(False,),
                atol=self.atol,
                max_iter=self.max_iter,
            )
            if not converged:
                raise RuntimeError(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )
            iter.append(n_iter + 1)

            (
                tn1,
                qn1,
                un12,
                un1,
                u_dotn1,
                R_gn1,
                R_gamman1,
                R_Nn1,
                R_Fn1,
                Delta_U,
                Delta_R_gn2,
                Delta_R_gamman2,
                Delta_R_Nn2,
                Delta_R_Fn2,
                P_Nn12,
                P_Fn12,
                P_Nn1,
                P_Fn1,
            ) = self._update(yn1, store=True)

            # P_gn1 = R_g1 + R_g2
            # P_gamman1 = R_gamma1 + R_gamma2
            # P_Nn1 = R_N1 + R_N2
            # P_Fn1 = R_F1 + R_F2
            # TODO:
            P_gn1 = R_gn1
            P_gamman1 = R_gamman1
            # P_Nn1 = P_Nn2
            # P_Fn1 = P_Fn2

            pbar.set_description(f"t: {tn1:0.2e}; step: {n_iter}; error: {error:.3e}")
            if not converged:
                print(
                    f"step is not converged after {n_iter} iterations with error: {error:.5e}"
                )

            qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            sol_q.append(qn1)
            sol_u.append(un1)
            sol_P_g.append(P_gn1)
            sol_P_gamma.append(P_gamman1)
            sol_P_N.append(P_Nn1)
            sol_P_F.append(P_Fn1)

            # warmstart for next iteration
            self.tn = tn1
            self.yn = yn1.copy()

        # print statistics
        print("-----------------")
        print(
            f"Iterations per time step: max = {max(iter)}, avg={sum(iter) / float(len(iter))}"
        )
        print("-----------------")
        return Solution(
            t=self.t,
            q=np.array(sol_q),
            u=np.array(sol_u),
            P_g=np.array(sol_P_g),
            P_gamma=np.array(sol_P_gamma),
            P_N=np.array(sol_P_N),
            P_F=np.array(sol_P_F),
        )


class SimplifiedNonsmoothGeneralizedAlphaFirstOrder:
    """Simplified version of the nonsmooth generalized-alpha solver for
    mechanical systems with frictional contact presented in Capobianco2020.
    The implementation applies the first order formulation of Jansen2000.

    References
    ----------
    Capobianco2020: https://doi.org/10.1002/nme.6801
    """

    def __init__(
        self,
        system,
        t1,
        h,
        rho_inf=0.8,
        atol=1e-6,
        max_iter=10,
    ):
        self.system = system

        # initial time, final time, time step
        self.t0 = t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.t = np.arange(t0, t1 + h, h)
        self.h = h

        # eqn. (72): parameters
        self.rho_inf = rho_inf
        self.alpha_m = 0.5 * (3.0 * rho_inf - 1.0) / (rho_inf + 1.0)
        self.alpha_f = rho_inf / (rho_inf + 1.0)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m

        # newton settings
        self.atol = atol
        self.max_iter = max_iter

        # dimensions (nq = number of coordinates q, etc.)
        self.nq = system.nq
        self.nu = system.nu
        self.nla_g = system.nla_g
        self.nla_gamma = system.nla_gamma
        self.nla_N = system.nla_N
        self.nla_F = system.nla_F

        self.split_y = np.cumsum(
            np.array(
                [
                    self.nq,
                    self.nu,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_N,
                    self.nla_F,
                    self.nu,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_N,
                ],
                dtype=int,
            )
        )

        # initial conditions
        self.tn = system.t0
        self.qn = system.q0
        self.un = system.u0
        self.q_dotn = system.q_dot0
        self.u_dotn = system.u_dot0
        self.la_gn = system.la_g0
        self.la_gamman = system.la_gamma0
        self.la_Nn = system.la_N0
        self.la_Fn = system.la_F0
        self.R_Nn = self.la_Nn
        self.R_Fn = self.la_Fn

        # initialize auxilary variables
        self.vn = self.q_dotn.copy()
        self.an = self.u_dotn.copy()
        self.R_N_barn = self.R_Nn.copy()
        self.R_F_barn = self.R_Fn.copy()
        self.P_Nn = h * self.R_Nn.copy()
        self.P_Fn = h * self.R_Fn.copy()

        # initialize y vector of unknowns
        self.yn = np.concatenate(
            (
                self.q_dotn,
                self.u_dotn,
                h * self.la_gn,
                h * self.la_gamman,
                self.R_Nn,
                self.R_Fn,
                np.zeros(self.nu),
                0 * self.la_gn,
                0 * self.la_gamman,
                0 * self.R_Nn,
                0 * self.R_Fn,
            )
        )

        # initialize index sets
        self.I = np.zeros(self.nla_N, dtype=bool)
        self.B = np.zeros(self.nla_N, dtype=bool)

    def _update(self, y, store=False):
        h = self.h
        tn1 = self.tn + h
        gamma = self.gamma
        alpha_f = self.alpha_f
        alpha_m = self.alpha_m

        # unpack unknowns
        (
            q_dotn1,
            u_dotn1,
            R_gn1,
            R_gamman1,
            R_Nn1,
            R_Fn1,
            Delta_U,
            Delta_R_gn2,
            Delta_R_gamman2,
            Delta_R_Nn2,
            Delta_R_Fn2,
        ) = np.array_split(y, self.split_y)

        # eqn. (71): compute auxiliary acceleration variables
        vn1 = (alpha_f * self.q_dotn + (1 - alpha_f) * q_dotn1 - alpha_m * self.vn) / (
            1 - alpha_m
        )
        an1 = (alpha_f * self.u_dotn + (1 - alpha_f) * u_dotn1 - alpha_m * self.an) / (
            1 - alpha_m
        )
        R_N_barn1 = (
            alpha_f * self.R_Nn + (1 - alpha_f) * R_Nn1 - alpha_m * self.R_N_barn
        ) / (1 - alpha_m)
        R_F_barn1 = (
            alpha_f * self.R_Fn + (1 - alpha_f) * R_Fn1 - alpha_m * self.R_F_barn
        ) / (1 - alpha_m)

        # eqn. (73): velocity update formula
        qn1 = self.qn + h * ((1 - gamma) * self.vn + gamma * vn1)
        un12 = self.un + h * ((1 - gamma) * self.an + gamma * an1)
        un1 = un12 + Delta_U
        # P_Nn12 = self.P_Nn + h * ((1 - gamma) * self.R_N_barn + gamma * R_N_barn1)
        P_Nn12 = h * ((1 - gamma) * self.R_N_barn + gamma * R_N_barn1)
        P_Nn1 = P_Nn12 + Delta_R_Nn2
        # P_Fn12 = self.P_Fn + h * ((1 - gamma) * self.R_F_barn + gamma * R_F_barn1)
        P_Fn12 = h * ((1 - gamma) * self.R_F_barn + gamma * R_F_barn1)
        P_Fn1 = P_Fn12 + Delta_R_Fn2

        if store:
            # update local variables for accepted time step
            self.tn = tn1
            self.qn = qn1.copy()
            self.un = un1.copy()
            self.q_dotn = q_dotn1.copy()
            self.u_dotn = u_dotn1.copy()
            self.vn = vn1.copy()
            self.an = an1.copy()
            self.R_Nn = R_Nn1.copy()
            self.R_N_barn = R_N_barn1.copy()
            # self.P_Nn = P_Nn1.copy()
            self.R_Fn = R_Fn1.copy()
            self.R_F_barn = R_F_barn1.copy()
            # self.P_Fn = P_Fn1.copy()

        return (
            tn1,
            qn1,
            un12,
            un1,
            q_dotn1,
            u_dotn1,
            R_gn1,
            R_gamman1,
            R_Nn1,
            R_Fn1,
            Delta_U,
            Delta_R_gn2,
            Delta_R_gamman2,
            Delta_R_Nn2,
            Delta_R_Fn2,
            P_Nn12,
            P_Fn12,
            P_Nn1,
            P_Fn1,
        )

    def _R(self, y, update_index=False):
        (
            tn1,
            qn1,
            un12,
            un1,
            q_dotn1,
            u_dotn1,
            R_gn1,
            R_gamman1,
            R_Nn1,
            R_Fn1,
            Delta_U,
            Delta_R_gn2,
            Delta_R_gamman2,
            Delta_R_Nn2,
            Delta_R_Fn2,
            P_Nn12,
            P_Fn12,
            P_Nn1,
            P_Fn1,
        ) = self._update(y)

        R = np.zeros_like(y)

        #################################
        # kinematic differential equation
        #################################
        R[: self.split_y[0]] = q_dotn1 - self.system.q_dot(tn1, qn1, un12)

        ####################
        # eqations of motion
        ####################
        M = self.system.M(tn1, qn1)
        W_g = self.system.W_g(tn1, qn1)
        W_gamma = self.system.W_gamma(tn1, qn1)
        W_N = self.system.W_N(tn1, qn1)
        W_F = self.system.W_F(tn1, qn1)
        R[self.split_y[0] : self.split_y[1]] = (
            M @ u_dotn1
            - self.system.h(tn1, qn1, un12)
            # - self.system.h(tn1, qn1, un1)
            - W_g @ R_gn1
            - W_gamma @ R_gamman1
            - W_N @ R_Nn1
            - W_F @ R_Fn1
        )

        #######################
        # bilateral constraints
        #######################
        R[self.split_y[1] : self.split_y[2]] = self.system.g(tn1, qn1)
        R[self.split_y[2] : self.split_y[3]] = self.system.gamma(tn1, qn1, un12)

        ###########
        # Signorini
        ###########
        g_N = self.system.g_N(tn1, qn1)
        prox_arg = g_N - self.prox_r_N * P_Nn12

        if update_index:
            self.I = prox_arg <= 0.0

        R[self.split_y[3] : self.split_y[4]] = np.where(
            self.I,
            g_N,
            P_Nn12,
        )

        ##################
        # Coulomb friction
        ##################
        prox_r_F = self.prox_r_F
        mu = self.system.mu
        gamma_F = self.system.gamma_F(tn1, qn1, un12)

        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                R[self.split_y[4] + i_F] = P_Fn12[i_F] + prox_sphere(
                    prox_r_F[i_N] * gamma_F[i_F] - P_Fn12[i_F],
                    mu[i_N] * P_Nn12[i_N],
                )

        #################
        # impact equation
        #################
        R[self.split_y[5] : self.split_y[6]] = (
            M @ Delta_U
            - W_g @ Delta_R_gn2
            - W_gamma @ Delta_R_gamman2
            - W_N @ Delta_R_Nn2
            - W_F @ Delta_R_Fn2
        )

        #################################
        # impulsive bilateral constraints
        #################################
        R[self.split_y[6] : self.split_y[7]] = self.system.g_dot(tn1, qn1, un1)
        R[self.split_y[7] : self.split_y[8]] = self.system.gamma(tn1, qn1, un1)

        ##################################################
        # mixed Singorini on velocity level and impact law
        ##################################################
        xi_Nn1 = self.system.xi_N(tn1, qn1, self.un, un1)
        prox_arg = xi_Nn1 - self.prox_r_N * P_Nn1
        if update_index:
            self.B = self.I * (prox_arg <= 0)

        R[self.split_y[8] : self.split_y[9]] = np.where(
            self.B,
            xi_Nn1,
            P_Nn1,
        )

        # R[self.split_y[8] : self.split_y[9]] = np.where(
        #     self.I,
        #     xi_Nn1 - prox_R0_np(xi_Nn1 - self.prox_r_N * P_Nn1),
        #     P_Nn1,
        # )

        ##################################################
        # mixed Coulomb friction and tangent impact law
        ##################################################
        xi_Fn1 = self.system.xi_F(tn1, qn1, self.un, un1)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                R[self.split_y[9] + i_F] = P_Fn1[i_F] + prox_sphere(
                    prox_r_F[i_N] * xi_Fn1[i_F] - P_Fn1[i_F],
                    mu[i_N] * P_Nn1[i_N],
                )

        return R

    def solve(self):
        # lists storing output variables
        sol_q = [self.qn]
        sol_u = [self.un]
        sol_P_g = [self.h * self.la_gn]
        sol_P_gamma = [self.h * self.la_gamman]
        sol_P_N = [self.h * self.la_Nn]
        sol_P_F = [self.h * self.la_Fn]

        # initialize progress bar
        pbar = tqdm(self.t[1:])

        iter = []
        for _ in pbar:
            # only compute optimized proxparameters once per time step
            self.prox_r_N = self.system.prox_r_N(self.tn, self.qn)
            self.prox_r_F = self.system.prox_r_F(self.tn, self.qn)
            # print(f"self.prox_r_N: {self.prox_r_N}")
            # print(f"self.prox_r_F: {self.prox_r_F}")
            # self.prox_r_N = np.ones(self.nla_N) * 0.2
            # self.prox_r_F = np.ones(self.nla_F) * 0.2

            yn1, converged, error, n_iter, _ = fsolve(
                self._R,
                self.yn,
                jac="2-point",
                eps=1.0e-6,
                fun_args=(True,),
                jac_args=(False,),
                atol=self.atol,
                max_iter=self.max_iter,
            )
            if not converged:
                raise RuntimeError(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )
            iter.append(n_iter + 1)

            (
                tn1,
                qn1,
                un12,
                un1,
                q_dotn1,
                u_dotn1,
                R_gn1,
                R_gamman1,
                R_Nn1,
                R_Fn1,
                Delta_U,
                Delta_R_gn2,
                Delta_R_gamman2,
                Delta_R_Nn2,
                Delta_R_Fn2,
                P_Nn12,
                P_Fn12,
                P_Nn1,
                P_Fn1,
            ) = self._update(yn1, store=True)

            # P_gn1 = R_g1 + R_g2
            # P_gamman1 = R_gamma1 + R_gamma2
            # P_Nn1 = R_N1 + R_N2
            # P_Fn1 = R_F1 + R_F2
            # TODO:
            P_gn1 = R_gn1
            P_gamman1 = R_gamman1
            # P_Nn1 = P_Nn2
            # P_Fn1 = P_Fn2

            pbar.set_description(f"t: {tn1:0.2e}; step: {n_iter}; error: {error:.3e}")
            if not converged:
                print(
                    f"step is not converged after {n_iter} iterations with error: {error:.5e}"
                )

            qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            sol_q.append(qn1)
            sol_u.append(un1)
            sol_P_g.append(P_gn1)
            sol_P_gamma.append(P_gamman1)
            sol_P_N.append(P_Nn1)
            sol_P_F.append(P_Fn1)

            # warmstart for next iteration
            self.tn = tn1
            self.yn = yn1.copy()

        # print statistics
        print("-----------------")
        print(
            f"Iterations per time step: max = {max(iter)}, avg={sum(iter) / float(len(iter))}"
        )
        print("-----------------")
        return Solution(
            t=self.t,
            q=np.array(sol_q),
            u=np.array(sol_u),
            P_g=np.array(sol_P_g),
            P_gamma=np.array(sol_P_gamma),
            P_N=np.array(sol_P_N),
            P_F=np.array(sol_P_F),
        )


class NonsmoothGeneralizedAlphaOriginal:
    def __init__(
        self,
        model,
        t1,
        dt,
        rho_inf=1,
        beta=None,
        gamma=None,
        alpha_m=None,
        alpha_f=None,
        newton_tol=1e-8,
        newton_max_iter=40,
        newton_error_function=lambda x: np.max(np.abs(x)),
        numerical_jacobian=False,
        debug=False,
    ):
        self.model = model

        # integration time
        self.t0 = t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        # parameter
        self.rho_inf = rho_inf
        if None in [beta, gamma, alpha_m, alpha_f]:
            self.alpha_m = (2 * rho_inf - 1) / (1 + rho_inf)
            self.alpha_f = rho_inf / (1 + rho_inf)
            self.gamma = 0.5 + self.alpha_f - self.alpha_m
            self.beta = 0.25 * ((self.gamma + 0.5) ** 2)
        else:
            self.gamma = gamma
            self.beta = beta
            self.alpha_m = alpha_m
            self.alpha_f = alpha_f
        self.alpha_ratio = (1 - self.alpha_f) / (1 - self.alpha_m)

        # newton settings
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.newton_error_function = newton_error_function

        # dimensions
        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = model.nla_g
        self.nla_gamma = model.nla_gamma
        self.nla_N = model.nla_N
        self.nla_T = model.nla_T

        self.nR_smooth = 3 * self.nu + 3 * self.nla_g + self.nla_gamma
        self.nR = self.nR_smooth + 3 * self.nla_N + 2 * self.nla_T

        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.kappa_gk = np.zeros_like(model.la_g0)
        self.la_gk = model.la_g0
        self.La_gk = np.zeros_like(model.la_g0)
        self.la_gammak = model.la_gamma0
        self.kappa_Nk = np.zeros_like(model.la_N0)
        self.la_Nk = model.la_N0
        self.La_Nk = np.zeros_like(model.la_N0)
        self.la_Tk = model.la_T0
        self.La_Tk = np.zeros_like(model.la_T0)
        self.ak = spsolve(
            model.M(t0, model.q0).tocsr(),
            self.model.h(t0, model.q0, model.u0)
            + self.model.W_g(t0, model.q0) @ model.la_g0
            + self.model.W_gamma(t0, model.q0) @ model.la_gamma0
            + self.model.W_N(t0, model.q0, scipy_matrix=csc_matrix) @ model.la_N0
            + self.model.W_T(t0, model.q0, scipy_matrix=csc_matrix) @ model.la_T0,
        )
        self.Qk = np.zeros(self.nu)
        self.Uk = np.zeros(self.nu)
        self.a_bark = self.ak.copy()
        self.la_gbark = self.la_Nk.copy()
        self.la_Nbark = self.la_Nk.copy()
        self.la_Tbark = self.la_Tk.copy()

        self.debug = debug
        if numerical_jacobian:
            self.__R_gen = self.__R_gen_num
        else:
            self.__R_gen = self.__R_gen_analytic

    def __R_gen_num(self, tk1, xk1):
        yield self.__R(tk1, xk1)
        yield csc_matrix(self.__R_x_num(tk1, xk1))

    def __R_gen_analytic(self, tk1, xk1):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        nla_T = self.nla_T
        dt = self.dt
        dt2 = self.dt**2
        ak1 = xk1[:nu]
        Uk1 = xk1[nu : 2 * nu]
        Qk1 = xk1[2 * nu : 3 * nu]
        kappa_gk1 = xk1[3 * nu : 3 * nu + nla_g]
        La_gk1 = xk1[3 * nu + nla_g : 3 * nu + 2 * nla_g]
        la_gk1 = xk1[3 * nu + 2 * nla_g : 3 * nu + 3 * nla_g]
        la_gammak1 = xk1[3 * nu + 3 * nla_g : 3 * nu + 3 * nla_g + nla_gamma]
        kappa_Nk1 = xk1[self.nR_smooth : self.nR_smooth + nla_N]
        La_Nk1 = xk1[self.nR_smooth + nla_N : self.nR_smooth + 2 * nla_N]
        la_Nk1 = xk1[self.nR_smooth + 2 * nla_N : self.nR_smooth + 3 * nla_N]
        La_Tk1 = xk1[self.nR_smooth + 3 * nla_N : self.nR_smooth + 3 * nla_N + nla_T]
        la_Tk1 = xk1[
            self.nR_smooth + 3 * nla_N + nla_T : self.nR_smooth + 3 * nla_N + 2 * nla_T
        ]

        # update dependent variables
        a_bark1 = (
            self.alpha_f * self.ak
            + (1 - self.alpha_f) * ak1
            - self.alpha_m * self.a_bark
        ) / (1 - self.alpha_m)
        uk1 = (
            self.uk + dt * ((1 - self.gamma) * self.a_bark + self.gamma * a_bark1) + Uk1
        )
        a_beta = (0.5 - self.beta) * self.a_bark + self.beta * a_bark1
        qk1 = (
            self.qk
            + dt * self.model.q_dot(self.tk, self.qk, self.uk)
            + dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, a_beta)
            + self.model.B(self.tk, self.qk) @ Qk1
        )

        la_Nbark1 = (
            self.alpha_f * self.la_Nk
            + (1 - self.alpha_f) * la_Nk1
            - self.alpha_m * self.la_Nbark
        ) / (1 - self.alpha_m)
        kappa_Nast = kappa_Nk1 + dt**2 * (
            (0.5 - self.beta) * self.la_Nbark + self.beta * la_Nbark1
        )
        P_N = La_Nk1 + dt * ((1 - self.gamma) * self.la_Nbark + self.gamma * la_Nbark1)

        la_Tbark1 = (
            self.alpha_f * self.la_Tk
            + (1 - self.alpha_f) * la_Tk1
            - self.alpha_m * self.la_Tbark
        ) / (1 - self.alpha_m)
        P_T = La_Tk1 + dt * ((1 - self.gamma) * self.la_Tbark + self.gamma * la_Tbark1)

        Mk1 = self.model.M(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1)
        W_gammak1 = self.model.W_gamma(tk1, qk1)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csc_matrix)
        W_Tk1 = self.model.W_T(tk1, qk1, scipy_matrix=csc_matrix)

        g_N = self.model.g_N(tk1, qk1)
        xi_N = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_T = self.model.xi_T(tk1, qk1, self.uk, uk1)
        g_N_ddot_post = self.model.g_N_ddot(tk1, qk1, uk1, ak1)
        gamma_T_dot_post = self.model.gamma_T_dot(tk1, qk1, uk1, ak1)
        gamma_T_post = self.model.gamma_T(tk1, qk1, uk1)

        # evaluate residual
        R = np.zeros(self.nR)
        R[:nu] = Mk1 @ ak1 - (
            self.model.h(tk1, qk1, uk1)
            + W_gk1 @ la_gk1
            + W_gammak1 @ la_gammak1
            + W_Nk1 @ la_Nk1
            + W_Tk1 @ la_Tk1
        )
        R[nu : 2 * nu] = Mk1 @ Uk1 - W_gk1 @ La_gk1 - W_Nk1 @ La_Nk1 - W_Tk1 @ La_Tk1
        R[2 * nu : 3 * nu] = Mk1 @ Qk1 - W_gk1 @ kappa_gk1 - W_Nk1 @ kappa_Nk1
        R[3 * nu : 3 * nu + nla_g] = self.model.g(tk1, qk1)
        R[3 * nu + nla_g : 3 * nu + 2 * nla_g] = self.model.g_dot(tk1, qk1, uk1)
        R[3 * nu + 2 * nla_g : 3 * nu + 3 * nla_g] = self.model.g_ddot(
            tk1, qk1, uk1, ak1
        )
        R[3 * nu + 3 * nla_g : 3 * nu + 3 * nla_g + nla_gamma] = self.model.gamma(
            tk1, qk1, uk1
        )

        I_N = kappa_Nast - self.model.prox_r_N * g_N >= 0
        I_N_ind = np.where(I_N)[0]
        _I_N_ind = np.where(~I_N)[0]
        R[self.nR_smooth + I_N_ind] = g_N[I_N]
        R[self.nR_smooth + _I_N_ind] = kappa_Nast[~I_N]

        A_N_ = (P_N - self.model.prox_r_N * xi_N) >= 0
        A_N = I_N * A_N_
        A_N_ind = np.where(A_N)[0]
        _A_N_ind = np.where(~A_N)[0]
        R[self.nR_smooth + nla_N + A_N_ind] = xi_N[A_N]
        R[self.nR_smooth + nla_N + _A_N_ind] = P_N[~A_N]

        B_N_ = (la_Nk1 - self.model.prox_r_N * g_N_ddot_post) >= 0
        B_N = A_N * B_N_
        B_N_ind = np.where(B_N)[0]
        _B_N_ind = np.where(~B_N)[0]
        R[self.nR_smooth + 2 * nla_N + B_N_ind] = g_N_ddot_post[B_N]
        R[self.nR_smooth + 2 * nla_N + _B_N_ind] = la_Nk1[~B_N]

        C_N = I_N * self.model.N_has_friction
        C_T = []
        for i_N, i_T in enumerate(self.model.NT_connectivity):
            C_T.append(
                np.linalg.norm(P_T[i_T] - self.model.prox_r_T[i_N] * xi_T[i_T])
                <= self.model.mu[i_N] * P_N[i_N]
            )
        C_T = np.array(C_T, dtype=bool)
        N_open = ~I_N * self.model.N_has_friction
        N_stick = C_N * C_T
        N_slip = C_N * ~C_T
        N_open_ind = np.where(N_open)[0]
        N_stick_ind = np.where(N_stick)[0]
        N_slip_ind = np.where(N_slip)[0]
        T_open_ind = np.array(
            [j for i in N_open_ind for j in self.model.NT_connectivity[i]], dtype=int
        )
        T_stick_ind = np.array(
            [j for i in N_stick_ind for j in self.model.NT_connectivity[i]], dtype=int
        )
        T_slip_ind = np.array(
            [j for i in N_slip_ind for j in self.model.NT_connectivity[i]], dtype=int
        )
        T_slip_ind_mat = np.array(
            [self.model.NT_connectivity[i] for i in N_slip_ind], dtype=int
        )

        R[self.nR_smooth + 3 * nla_N + T_open_ind] = P_T[T_open_ind]
        R[self.nR_smooth + 3 * nla_N + T_stick_ind] = xi_T[T_stick_ind]
        tmp = xi_T[T_slip_ind_mat]
        norm_xi_ = np.linalg.norm(tmp, axis=-1)
        norm_xi = np.ones_like(norm_xi_)
        norm_xi[norm_xi_ > 0] = norm_xi_[norm_xi_ > 0]
        R[self.nR_smooth + 3 * nla_N + T_slip_ind] = P_T[T_slip_ind] + (
            (self.model.mu[N_slip_ind] * P_N[N_slip_ind] / norm_xi).reshape(-1, 1) * tmp
        ).reshape(-1)

        R[self.nR_smooth + 3 * nla_N + nla_T + T_open_ind] = la_Tk1[T_open_ind]
        R[self.nR_smooth + 3 * nla_N + nla_T + T_stick_ind] = gamma_T_dot_post[
            T_stick_ind
        ]
        tmp = gamma_T_post[T_slip_ind_mat]
        norm_xi_ = np.linalg.norm(tmp, axis=-1)
        norm_xi = np.ones_like(norm_xi_)
        norm_xi[norm_xi_ > 0] = norm_xi_[norm_xi_ > 0]
        R[self.nR_smooth + 3 * nla_N + nla_T + T_slip_ind] = la_Tk1[T_slip_ind] + (
            (self.model.mu[N_slip_ind] * la_Nk1[N_slip_ind] / norm_xi).reshape(-1, 1)
            * tmp
        ).reshape(-1)

        yield R

        g_N_q = self.model.g_N_q(tk1, qk1, scipy_matrix=csc_matrix)
        g_N_dot_u = self.model.g_N_dot_u(tk1, qk1, scipy_matrix=csc_matrix)
        xi_N_q = self.model.xi_N_q(tk1, qk1, self.uk, uk1, scipy_matrix=csc_matrix)
        g_N_ddot_post_q = self.model.g_N_ddot_q(
            tk1, qk1, uk1, ak1, scipy_matrix=csc_matrix
        )
        g_N_ddot_post_u = self.model.g_N_ddot_u(
            tk1, qk1, uk1, ak1, scipy_matrix=csc_matrix
        )
        xi_T_q = self.model.xi_T_q(tk1, qk1, self.uk, uk1, scipy_matrix=csc_matrix)
        xi_T_u = gamma_T_u = self.model.gamma_T_u(tk1, qk1, scipy_matrix=csc_matrix)
        gamma_T_q = self.model.gamma_T_q(tk1, qk1, uk1, scipy_matrix=csc_matrix)
        gamma_T_dot_post_q = self.model.gamma_T_dot_q(
            tk1, qk1, uk1, ak1, scipy_matrix=csc_matrix
        )
        gamma_T_dot_post_u = self.model.gamma_T_dot_u(
            tk1, qk1, uk1, ak1, scipy_matrix=csc_matrix
        )

        # R[:nu] = Mk1 @ ak1 - ( self.model.h(tk1, qk1, uk1) + W_gk1 @ la_gk1 + W_gammak1 @ la_gammak1 + W_Nk1 @ la_Nk1 + W_Tk1 @ la_Tk1)
        Ra_q = self.model.Mu_q(tk1, qk1, ak1) - (
            self.model.h_q(tk1, qk1, uk1)
            + self.model.Wla_g_q(tk1, qk1, la_gk1)
            + self.model.Wla_gamma_q(tk1, qk1, la_gammak1)
            + self.model.Wla_N_q(tk1, qk1, la_Nk1)
            + self.model.Wla_T_q(tk1, qk1, la_Tk1)
        )
        Ra_u = -self.model.h_u(tk1, qk1, uk1)
        Ra_a = Mk1 + Ra_q @ self.q_a + Ra_u * self.u_a
        Ra_U = Ra_u
        Ra_Q = Ra_q @ self.q_Q

        # R[nu:2*nu] = Mk1 @ Uk1 - W_Nk1 @ La_Nk1 - W_Tk1 @ La_Tk1
        RU_q = (
            self.model.Mu_q(tk1, qk1, Uk1)
            - self.model.Wla_g_q(tk1, qk1, La_gk1)
            - self.model.Wla_N_q(tk1, qk1, La_Nk1)
            - self.model.Wla_T_q(tk1, qk1, La_Tk1)
        )
        RU_a = RU_q @ self.q_a
        RU_Q = RU_q @ self.q_Q

        # R[2*nu:3*nu] = Mk1 @ Qk1 - W_Nk1 @ kappa_Nk1
        RQ_q = (
            self.model.Mu_q(tk1, qk1, Qk1)
            - self.model.Wla_g_q(tk1, qk1, kappa_gk1)
            - self.model.Wla_N_q(tk1, qk1, kappa_Nk1)
        )
        RQ_a = RQ_q @ self.q_a
        RQ_Q = Mk1 + RQ_q @ self.q_Q

        # R[3*nu:3*nu+nla_g] = self.model.g(tk1, qk1)
        Rka_g_q = self.model.g_q(tk1, qk1)
        Rka_g_a = Rka_g_q @ self.q_a
        Rka_g_Q = Rka_g_q @ self.q_Q

        # R[3*nu+nla_g:3*nu+2*nla_g] = self.model.g_dot(tk1, qk1, uk1)
        RLa_g_q = self.model.g_dot_q(tk1, qk1, uk1)
        RLa_g_u = self.model.g_dot_u(tk1, qk1)
        RLa_g_a = RLa_g_q @ self.q_a + RLa_g_u * self.u_a
        RLa_g_Q = RLa_g_q @ self.q_Q

        # R[3*nu+2*nla_g:3*nu+3*nla_g] = self.model.g_ddot(tk1, qk1, uk1, ak1)
        Rla_g_q = self.model.g_ddot_q(tk1, qk1, uk1, ak1)
        Rla_g_u = self.model.g_ddot_u(tk1, qk1, uk1, ak1)
        Rla_g_a = self.model.g_dot_u(tk1, qk1)
        Rla_g_a += Rla_g_q @ self.q_a + Rla_g_u * self.u_a
        Rla_g_Q = Rla_g_q @ self.q_Q

        # R[3*nu+nla_g:3*nu+nla_g+nla_gamma] = self.model.gamma(tk1, qk1, uk1)
        Rla_gamma_q = self.model.gamma_q(tk1, qk1, uk1)
        Rla_gamma_u = self.model.gamma_u(tk1, qk1)  # == Rla_gamma_U
        Rla_gamma_a = Rla_gamma_q @ self.q_a + Rla_gamma_u * self.u_a
        Rla_gamma_Q = Rla_gamma_q @ self.q_Q

        # R[3*nu+nla_g+nla_gamma:3*nu+nla_g+nla_gamma+nla_N] = kappa_ast - prox_R0_nm(kappa_ast - self.model.prox_r_N * g_N)
        row = col = np.where(~I_N)[0]
        data = np.ones_like(row)
        Rka_ka_ast = coo_matrix((data, (row, col)), shape=(nla_N, nla_N))

        # Rka_q = (diags(self.model.prox_r_N) @ g_N_q)[I_N].tocoo()
        Rka_q = g_N_q[I_N].tocoo()
        Rka_q.resize(nla_N, nq)
        Rka_q.row = np.where(I_N)[0][Rka_q.row]

        Rka_a = Rka_q @ self.q_a
        Rka_Q = Rka_q @ self.q_Q
        Rka_ka = Rka_ka_ast
        Rka_la_N = Rka_ka_ast * self.ka_ast_la_N

        # R[3*nu+nla_g+nla_gamma+nla_N+A_N_ind] = xi_N[A_N]
        # R[3*nu+nla_g+nla_gamma+nla_N+_A_N_ind] = P_N[~A_N]
        row = col = _A_N_ind
        data = np.ones_like(row)
        RLaN_P_N = coo_matrix((data, (row, col)), shape=(nla_N, nla_N))

        RLaN_u = g_N_dot_u[A_N].tocoo()
        RLaN_u.resize(nla_N, nu)
        RLaN_u.row = A_N_ind[RLaN_u.row]

        RLaN_q = xi_N_q[A_N].tocoo()
        RLaN_q.resize(nla_N, nq)
        RLaN_q.row = A_N_ind[RLaN_q.row]

        RLaN_a = RLaN_q @ self.q_a + RLaN_u * self.u_a
        RLaN_Q = RLaN_q @ self.q_Q
        RLaN_La_N = RLaN_P_N
        RLaN_la_N = RLaN_P_N * self.P_N_la_N

        # R[3*nu+nla_g+nla_gamma+2*nla_N+B_N_ind] = g_N_ddot_post[B_N]
        # R[3*nu+nla_g+nla_gamma+2*nla_N+_B_N_ind] = la_Nk1[~B_N]
        row = col = _B_N_ind
        data = np.ones_like(row)
        RlaN_la_N = coo_matrix((data, (row, col)), shape=(nla_N, nla_N))

        RlaN_a = g_N_dot_u[B_N].tocoo()
        RlaN_a.resize(nla_N, nu)
        RlaN_a.row = B_N_ind[RlaN_a.row]

        RlaN_u = g_N_ddot_post_u[B_N].tocoo()
        RlaN_u.resize(nla_N, nu)
        RlaN_u.row = B_N_ind[RlaN_u.row]

        RlaN_q = g_N_ddot_post_q[B_N].tocoo()
        RlaN_q.resize(nla_N, nq)
        RlaN_q.row = B_N_ind[RlaN_q.row]

        RlaN_a += RlaN_q @ self.q_a + RlaN_u * self.u_a
        RlaN_Q = RlaN_q @ self.q_Q

        # tmp = xi_T[T_slip_ind_mat]
        # norm_xi_ = np.linalg.norm(tmp, axis=-1)
        # norm_xi = np.ones_like(norm_xi_)
        # norm_xi[norm_xi_>0] = norm_xi_[norm_xi_>0]
        # R[T_open_ind] = P_T[T_open_ind]
        # R[T_stick_ind] = xi_T[T_stick_ind]
        # R[T_slip_ind] = P_T[T_slip_ind] + ((self.model.mu[N_slip_ind] * P_N[N_slip_ind] / norm_xi).reshape(-1, 1) * tmp).reshape(-1)
        row = col = np.concatenate((T_open_ind, T_slip_ind))
        data = np.ones_like(row)
        RLaT_P_T = coo_matrix((data, (row, col)), shape=(nla_T, nla_T))

        RLaT_u = xi_T_u[T_stick_ind].tocoo()
        RLaT_u.resize(nla_T, nu)
        RLaT_u.row = T_stick_ind[RLaT_u.row]

        RLaT_q = xi_T_q[T_stick_ind].tocoo()
        RLaT_q.resize(nla_T, nq)
        RLaT_q.row = T_stick_ind[RLaT_q.row]

        RLaT_P_N = coo_matrix((nla_T, nla_N))

        u_data = []
        u_row = []
        u_col = []
        q_data = []
        q_row = []
        q_col = []
        P_N_data = []
        P_N_row = []
        P_N_col = []
        for i_N in N_slip_ind:
            i_T = self.model.NT_connectivity[i_N]
            xi_T_loc = xi_T[i_T]
            xi_T_u_loc = xi_T_u[i_T]
            xi_T_q_loc = xi_T_q[i_T]
            norm_T = np.linalg.norm(xi_T_loc)
            norm_T2 = norm_T**2

            if norm_T > 0:
                tmp_u = (self.model.mu[i_N] * P_N[i_N] / norm_T) * (
                    xi_T_u_loc - np.outer(xi_T_loc / norm_T2, xi_T_loc @ xi_T_u_loc)
                )
                tmp_q = (self.model.mu[i_N] * P_N[i_N] / norm_T) * (
                    xi_T_q_loc - np.outer(xi_T_loc / norm_T2, xi_T_loc @ xi_T_q_loc)
                )
                tmp_P_N = (self.model.mu[i_N] / norm_T) * xi_T_loc
            else:
                tmp_u = (self.model.mu[i_N] * P_N[i_N]) * xi_T_u_loc.toarray()
                tmp_q = (self.model.mu[i_N] * P_N[i_N]) * xi_T_q_loc.toarray()
                tmp_P_N = (self.model.mu[i_N]) * xi_T_loc

            u_data.extend(np.asarray(tmp_u).reshape(-1, order="C").tolist())
            u_row.extend(np.repeat(i_T, nu).tolist())
            u_col.extend(np.tile(np.arange(nu), len(i_T)).tolist())

            q_data.extend(np.asarray(tmp_q).reshape(-1, order="C").tolist())
            q_row.extend(np.repeat(i_T, nq).tolist())
            q_col.extend(np.tile(np.arange(nq), len(i_T)).tolist())

            P_N_data.extend(tmp_P_N.tolist())
            P_N_row.extend(i_T)
            P_N_col.extend((i_N * np.ones_like(i_T)).tolist())

        RLaT_u.data = np.append(RLaT_u.data, u_data)
        RLaT_u.row = np.append(RLaT_u.row, u_row).astype(int)
        RLaT_u.col = np.append(RLaT_u.col, u_col).astype(int)

        RLaT_q.data = np.append(RLaT_q.data, q_data)
        RLaT_q.row = np.append(RLaT_q.row, q_row).astype(int)
        RLaT_q.col = np.append(RLaT_q.col, q_col).astype(int)

        RLaT_P_N.data = np.append(RLaT_P_N.data, P_N_data)
        RLaT_P_N.row = np.append(RLaT_P_N.row, P_N_row).astype(int)
        RLaT_P_N.col = np.append(RLaT_P_N.col, P_N_col).astype(int)

        RLaT_a = RLaT_u * self.u_a + RLaT_q @ self.q_a
        RLaT_U = RLaT_u
        RLaT_Q = RLaT_q @ self.q_Q
        RLaT_La_N = RLaT_P_N
        RLaT_la_N = RLaT_P_N * self.P_N_la_N
        RLaT_La_T = RLaT_P_T
        RLaT_la_T = RLaT_P_T * self.P_T_la_T

        # tmp = gamma_T_post[T_slip_ind_mat]
        # norm_xi_ = np.linalg.norm(tmp, axis=-1)
        # norm_xi = np.ones_like(norm_xi_)
        # norm_xi[norm_xi_>0] = norm_xi_[norm_xi_>0]
        # R[3*nu+nla_g+nla_gamma+3*nla_N+nla_T+T_open_ind] = la_Tk1[T_open_ind]
        # R[3*nu+nla_g+nla_gamma+3*nla_N+nla_T+T_stick_ind] = gamma_T_dot_post[T_stick_ind]
        # R[3*nu+nla_g+nla_gamma+3*nla_N+nla_T+T_slip_ind] = la_Tk1[T_slip_ind] + ((self.model.mu[N_slip_ind] * la_Nk1[N_slip_ind] / norm_xi).reshape(-1, 1) * tmp).reshape(-1)
        row = col = np.concatenate((T_open_ind, T_slip_ind))
        data = np.ones_like(row)
        RlaT_la_T = coo_matrix((data, (row, col)), shape=(nla_T, nla_T))

        RlaT_a = gamma_T_u[T_stick_ind].tocoo()
        RlaT_a.resize(nla_T, nu)
        RlaT_a.row = T_stick_ind[RlaT_a.row]

        RlaT_u = gamma_T_dot_post_u[T_stick_ind].tocoo()
        RlaT_u.resize(nla_T, nu)
        RlaT_u.row = T_stick_ind[RlaT_u.row]

        RlaT_q = gamma_T_dot_post_q[T_stick_ind].tocoo()
        RlaT_q.resize(nla_T, nq)
        RlaT_q.row = T_stick_ind[RlaT_q.row]

        RlaT_la_N = coo_matrix((nla_T, nla_N))

        u_data = []
        u_row = []
        u_col = []
        q_data = []
        q_row = []
        q_col = []
        la_N_data = []
        la_N_row = []
        la_N_col = []
        for i_N in N_slip_ind:
            i_T = self.model.NT_connectivity[i_N]
            gamma_T_loc = gamma_T_post[i_T]
            gamma_T_u_loc = gamma_T_u[i_T]
            gamma_T_q_loc = gamma_T_q[i_T]
            norm_T = np.linalg.norm(gamma_T_loc)
            norm_T2 = norm_T**2

            if norm_T > 0:
                tmp_u = (self.model.mu[i_N] * la_Nk1[i_N] / norm_T) * (
                    gamma_T_u_loc
                    - np.outer(gamma_T_loc / norm_T2, gamma_T_loc @ gamma_T_u_loc)
                )
                tmp_q = (self.model.mu[i_N] * la_Nk1[i_N] / norm_T) * (
                    gamma_T_q_loc
                    - np.outer(gamma_T_loc / norm_T2, gamma_T_loc @ gamma_T_q_loc)
                )
                tmp_la_N = (self.model.mu[i_N] / norm_T) * gamma_T_loc
            else:
                tmp_u = (self.model.mu[i_N] * la_Nk1[i_N]) * gamma_T_u_loc.toarray()
                tmp_q = (self.model.mu[i_N] * la_Nk1[i_N]) * gamma_T_q_loc.toarray()
                tmp_la_N = (self.model.mu[i_N]) * gamma_T_loc

            u_data.extend(np.asarray(tmp_u).reshape(-1, order="C").tolist())
            u_row.extend(np.repeat(i_T, nu).tolist())
            u_col.extend(np.tile(np.arange(nu), len(i_T)).tolist())

            q_data.extend(np.asarray(tmp_q).reshape(-1, order="C").tolist())
            q_row.extend(np.repeat(i_T, nq).tolist())
            q_col.extend(np.tile(np.arange(nq), len(i_T)).tolist())

            la_N_data.extend(tmp_la_N.tolist())
            la_N_row.extend(i_T)
            la_N_col.extend((i_N * np.ones_like(i_T)).tolist())

        RlaT_u.data = np.append(RlaT_u.data, u_data)
        RlaT_u.row = np.append(RlaT_u.row, u_row).astype(int)
        RlaT_u.col = np.append(RlaT_u.col, u_col).astype(int)

        RlaT_q.data = np.append(RlaT_q.data, q_data)
        RlaT_q.row = np.append(RlaT_q.row, q_row).astype(int)
        RlaT_q.col = np.append(RlaT_q.col, q_col).astype(int)

        RlaT_la_N.data = np.append(RlaT_la_N.data, la_N_data)
        RlaT_la_N.row = np.append(RlaT_la_N.row, la_N_row).astype(int)
        RlaT_la_N.col = np.append(RlaT_la_N.col, la_N_col).astype(int)

        RlaT_a += RlaT_u * self.u_a + RlaT_q @ self.q_a
        RlaT_U = RlaT_u
        RlaT_Q = RlaT_q @ self.q_Q

        R_x = bmat(
            [
                [
                    Ra_a,
                    Ra_U,
                    Ra_Q,
                    None,
                    None,
                    -W_gk1,
                    -W_gammak1,
                    None,
                    None,
                    -W_Nk1,
                    None,
                    -W_Tk1,
                ],
                [
                    RU_a,
                    Mk1,
                    RU_Q,
                    None,
                    -W_gk1,
                    None,
                    None,
                    None,
                    -W_Nk1,
                    None,
                    -W_Tk1,
                    None,
                ],
                [
                    RQ_a,
                    None,
                    RQ_Q,
                    -W_gk1,
                    None,
                    None,
                    None,
                    -W_Nk1,
                    None,
                    None,
                    None,
                    None,
                ],
                [
                    Rka_g_a,
                    None,
                    Rka_g_Q,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                [
                    RLa_g_a,
                    RLa_g_u,
                    RLa_g_Q,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                [
                    Rla_g_a,
                    Rla_g_u,
                    Rla_g_Q,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                [
                    Rla_gamma_a,
                    Rla_gamma_u,
                    Rla_gamma_Q,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                [
                    Rka_a,
                    None,
                    Rka_Q,
                    None,
                    None,
                    None,
                    None,
                    Rka_ka,
                    None,
                    Rka_la_N,
                    None,
                    None,
                ],
                [
                    RLaN_a,
                    RLaN_u,
                    RLaN_Q,
                    None,
                    None,
                    None,
                    None,
                    None,
                    RLaN_La_N,
                    RLaN_la_N,
                    None,
                    None,
                ],
                [
                    RlaN_a,
                    RlaN_u,
                    RlaN_Q,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    RlaN_la_N,
                    None,
                    None,
                ],
                [
                    RLaT_a,
                    RLaT_U,
                    RLaT_Q,
                    None,
                    None,
                    None,
                    None,
                    None,
                    RLaT_La_N,
                    RLaT_la_N,
                    RLaT_La_T,
                    RLaT_la_T,
                ],
                [
                    RlaT_a,
                    RlaT_U,
                    RlaT_Q,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    RlaT_la_N,
                    None,
                    RlaT_la_T,
                ],
            ],
            format="csc",
        )

        # R_x_num = self.__R_x_num(tk1, xk1)
        # diff = R_x.toarray() - R_x_num
        # # # error = np.linalg.norm(diff, ord=inf)
        # # # error = np.linalg.norm(diff[:nu], ord=inf)
        # error = np.max(np.abs(diff)) / np.max(np.abs(R_x_num))
        # print(f'error = {error}')

        yield R_x

    def __R(self, tk1, xk1):
        return next(self.__R_gen_analytic(tk1, xk1))

    def __R_x_num(self, tk1, xk1):
        return Numerical_derivative(self.__R, order=2)._x(tk1, xk1)

    def step(self):
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        nla_T = self.nla_T
        dt = self.dt
        tk1 = self.tk + dt

        # initial guess for Newton-Raphson solver
        xk1 = np.zeros(self.nR)
        xk1[:nu] = self.ak
        xk1[nu : 2 * nu] = self.Uk
        xk1[2 * nu : 3 * nu] = self.Qk
        xk1[3 * nu : 3 * nu + nla_g] = self.kappa_gk
        xk1[3 * nu + nla_g : 3 * nu + 2 * nla_g] = self.La_gk
        xk1[3 * nu + 2 * nla_g : 3 * nu + 3 * nla_g] = self.la_gk
        xk1[3 * nu + 3 * nla_g : 3 * nu + 3 * nla_g + nla_gamma] = self.la_gammak
        xk1[self.nR_smooth : self.nR_smooth + nla_N] = self.kappa_Nk
        xk1[self.nR_smooth + nla_N : self.nR_smooth + 2 * nla_N] = self.La_Nk
        xk1[self.nR_smooth + 2 * nla_N : self.nR_smooth + 3 * nla_N] = self.la_Nk
        xk1[
            self.nR_smooth + 3 * nla_N : self.nR_smooth + 3 * nla_N + nla_T
        ] = self.La_Tk
        xk1[
            self.nR_smooth + 3 * nla_N + nla_T : self.nR_smooth + 3 * nla_N + 2 * nla_T
        ] = self.la_Tk

        # initial residual and error
        R_gen = self.__R_gen(tk1, xk1)
        R = next(R_gen)
        error = self.newton_error_function(R)
        converged = error < self.newton_tol
        j = 0
        if not converged:
            while j < self.newton_max_iter:
                # jacobian
                R_x = next(R_gen)

                # Newton update
                j += 1
                dx = spsolve(R_x, R)
                # try:
                #     dx = spsolve(R_x, R)
                # except:
                #     print('Fehler!!!!')
                xk1 -= dx
                R_gen = self.__R_gen(tk1, xk1)
                R = next(R_gen)

                error = self.newton_error_function(R)
                converged = error < self.newton_tol
                if converged:
                    break
        ak1 = xk1[:nu]
        Uk1 = xk1[nu : 2 * nu]
        Qk1 = xk1[2 * nu : 3 * nu]
        kappa_gk1 = xk1[3 * nu : 3 * nu + nla_g]
        La_gk1 = xk1[3 * nu + nla_g : 3 * nu + 2 * nla_g]
        la_gk1 = xk1[3 * nu + 2 * nla_g : 3 * nu + 3 * nla_g]
        la_gammak1 = xk1[3 * nu + 3 * nla_g : 3 * nu + 3 * nla_g + nla_gamma]
        kappa_Nk1 = xk1[self.nR_smooth : self.nR_smooth + nla_N]
        La_Nk1 = xk1[self.nR_smooth + nla_N : self.nR_smooth + 2 * nla_N]
        la_Nk1 = xk1[self.nR_smooth + 2 * nla_N : self.nR_smooth + 3 * nla_N]
        La_Tk1 = xk1[self.nR_smooth + 3 * nla_N : self.nR_smooth + 3 * nla_N + nla_T]
        la_Tk1 = xk1[
            self.nR_smooth + 3 * nla_N + nla_T : self.nR_smooth + 3 * nla_N + 2 * nla_T
        ]

        return (
            (converged, j, error),
            tk1,
            ak1,
            Uk1,
            Qk1,
            kappa_gk1,
            La_gk1,
            la_gk1,
            la_gammak1,
            kappa_Nk1,
            La_Nk1,
            la_Nk1,
            La_Tk1,
            la_Tk1,
        )

    def solve(self):
        dt = self.dt
        dt2 = self.dt**2

        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        a = [self.ak]
        kappa_g = [self.kappa_gk]
        La_g = [self.La_gk]
        la_g = [self.la_gk]
        la_gamma = [self.la_gammak]
        kappa_N = [self.kappa_Nk]
        La_N = [self.La_Nk]
        la_N = [self.la_Nk]
        La_T = [self.La_Tk]
        la_T = [self.la_Tk]
        P_N = [self.La_Nk + self.dt * self.la_Nk]
        P_T = [self.La_Tk + self.dt * self.la_Tk]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            Bk = self.model.B(self.tk, self.qk)
            self.q_a = dt2 * self.beta * self.alpha_ratio * Bk
            self.q_Q = Bk
            self.u_a = dt * self.gamma * self.alpha_ratio
            self.ka_ast_la_N = dt2 * self.beta * self.alpha_ratio
            self.P_N_la_N = dt * self.gamma * self.alpha_ratio
            self.P_T_la_T = dt * self.gamma * self.alpha_ratio

            (
                (converged, n_iter, error),
                tk1,
                ak1,
                Uk1,
                Qk1,
                kappa_gk1,
                La_gk1,
                la_gk1,
                la_gammak1,
                kappa_Nk1,
                La_Nk1,
                la_Nk1,
                La_Tk1,
                la_Tk1,
            ) = self.step()
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.newton_max_iter} iterations; error: {error:0.2e}"
            )

            if not converged:
                raise RuntimeError(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )
            dt = self.dt
            dt2 = dt * dt
            a_bark1 = (
                self.alpha_f * self.ak
                + (1 - self.alpha_f) * ak1
                - self.alpha_m * self.a_bark
            ) / (1 - self.alpha_m)
            uk1 = (
                self.uk
                + dt * ((1 - self.gamma) * self.a_bark + self.gamma * a_bark1)
                + Uk1
            )
            a_beta = (0.5 - self.beta) * self.a_bark + self.beta * a_bark1
            qk1 = (
                self.qk
                + dt * self.model.q_dot(self.tk, self.qk, self.uk)
                + dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, a_beta)
                + Bk @ Qk1
            )

            la_Nbark1 = (
                self.alpha_f * self.la_Nk
                + (1 - self.alpha_f) * la_Nk1
                - self.alpha_m * self.la_Nbark
            ) / (1 - self.alpha_m)
            P_N_ = La_Nk1 + dt * (
                (1 - self.gamma) * self.la_Nbark + self.gamma * la_Nbark1
            )

            la_Tbark1 = (
                self.alpha_f * self.la_Tk
                + (1 - self.alpha_f) * la_Tk1
                - self.alpha_m * self.la_Tbark
            ) / (1 - self.alpha_m)
            P_T_ = La_Tk1 + dt * (
                (1 - self.gamma) * self.la_Tbark + self.gamma * la_Tbark1
            )

            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            a.append(ak1)
            kappa_g.append(kappa_gk1)
            La_g.append(La_gk1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)
            kappa_N.append(kappa_Nk1)
            La_N.append(La_Nk1)
            la_N.append(la_Nk1)
            La_T.append(La_Tk1)
            la_T.append(la_Tk1)
            P_N.append(P_N_)
            P_T.append(P_T_)

            # update local variables for accepted time step
            self.tk = tk1
            self.qk = qk1
            self.uk = uk1
            self.ak = ak1
            self.Qk = Qk1
            self.kappa_gk = kappa_gk1
            self.La_gk = La_gk1
            self.la_gk = la_gk1
            self.la_gammak = la_gammak1
            self.kappa_Nk = kappa_Nk1
            self.La_Nk = La_Nk1
            self.la_Nk = la_Nk1
            self.La_Tk = La_Tk1
            self.la_Tk = la_Tk1
            self.a_bark = a_bark1
            self.la_Nbark = la_Nbark1
            self.la_Tbark = la_Tbark1

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            a=np.array(a),
            kappa_g=np.array(kappa_g),
            La_g=np.array(La_g),
            la_g=np.array(la_g),
            la_gamma=np.array(la_gamma),
            kappa_P=np.array(kappa_N),
            La_N=np.array(La_N),
            la_N=np.array(la_N),
            La_T=np.array(La_T),
            la_T=np.array(la_T),
            P_N=np.array(P_N),
            P_T=np.array(P_T),
        )
