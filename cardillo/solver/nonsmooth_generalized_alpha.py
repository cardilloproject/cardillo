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
from tqdm import tqdm

from cardillo.math import (
    prox_R0_nm,
    prox_sphere,
    fsolve,
    estimate_prox_parameter,
)
from cardillo.solver import SolverOptions, SolverSummary, Solution


class NonsmoothGeneralizedAlpha:
    """Generalized-alpha solver for mechanical systems with frictional contact."""

    def __init__(
        self,
        system,
        t1,
        dt,
        rho_inf=0.9,
        # method="newton",
        method="fixed-point",
        options=SolverOptions(),
    ):
        self.system = system
        self.method = method

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

        # options
        self.options = options

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

    def R(self, x, update_index_set=False):
        """Residual R=(R_s, R_c), see eqn. (127)"""
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

        # # eqn. (125): generalized position update formula
        # a_beta = (1 - 2 * self.beta) * self.a_bari + 2 * self.beta * a_bari1
        # qi1 = (
        #     self.qi
        #     + dt * self.system.q_dot(self.ti, self.qi, self.ui)
        #     + dt2 / 2 * self.system.q_ddot(self.ti, self.qi, self.ui, a_beta)
        #     + self.system.B(self.ti, self.qi) @ Qi1
        # )

        # position update, see GAMM2022 Harsch
        a_beta = (1 - 2 * self.beta) * self.a_bari + 2 * self.beta * a_bari1
        Delta_u_bar = self.ui + dt * a_beta
        qi1 = (
            self.qi
            + dt * self.system.q_dot(self.ti, self.qi, Delta_u_bar)
            + self.system.q_dot_u(self.ti, self.qi) @ Qi1
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
        xi_Ni1 = self.system.xi_N(self.ti, ti1, self.qi, qi1, self.ui, ui1)
        xi_Fi1 = self.system.xi_F(self.ti, ti1, self.qi, qi1, self.ui, ui1)
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
            prox_r_N = estimate_prox_parameter(self.options.prox_scaling, W_Ni1, Mi1)
            prox_r_F = estimate_prox_parameter(self.options.prox_scaling, W_Fi1, Mi1)

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
        dt = self.dt
        # dt2 = self.dt**2
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

        # # eqn. (125): generalized position update formula
        # a_beta = (1 - 2 * self.beta) * self.a_bari + 2 * self.beta * a_bari1
        # qi1 = (
        #     self.qi
        #     + dt * self.system.q_dot(self.ti, self.qi, self.ui)
        #     + dt2 / 2 * self.system.q_ddot(self.ti, self.qi, self.ui, a_beta)
        #     + self.system.B(self.ti, self.qi) @ Qi1
        # )

        # position update, see GAMM2022 Harsch
        a_beta = (1 - 2 * self.beta) * self.a_bari + 2 * self.beta * a_bari1
        Delta_u_bar = self.ui + dt * a_beta
        qi1 = (
            self.qi
            + dt * self.system.q_dot(self.ti, self.qi, Delta_u_bar)
            + self.system.q_dot_u(self.ti, self.qi) @ Qi1
        )

        # ----- get quantities from model -----
        # Mass matrix
        self.Mi1 = self.system.M(ti1, qi1)

        # generalized force directions
        W_gi1 = self.system.W_g(ti1, qi1)
        W_gammai1 = self.system.W_gamma(ti1, qi1)
        self.W_Ni1 = self.system.W_N(ti1, qi1)
        self.W_Fi1 = self.system.W_F(ti1, qi1)

        # ----- compute residual -----
        R_s = np.concatenate(
            (
                self.Mi1 @ ai1
                - self.system.h(ti1, qi1, ui1)
                - W_gi1 @ la_gi1
                - W_gammai1 @ la_gammai1
                - self.W_Ni1 @ la_Ni1
                - self.W_Fi1 @ la_Fi1,
                self.Mi1 @ Ui1
                - W_gi1 @ La_gi1
                - W_gammai1 @ La_gammai1
                - self.W_Ni1 @ La_Ni1
                - self.W_Fi1 @ La_Fi1,
                self.Mi1 @ Qi1
                - W_gi1 @ kappa_gi1
                - self.W_Ni1 @ kappa_Ni1
                - 0.5 * dt * (W_gammai1 @ La_gammai1 + self.W_Fi1 @ La_Fi1),
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

        # # eqn. (125): generalized position update formula
        # a_beta = (1 - 2 * self.beta) * self.a_bari + 2 * self.beta * a_bari1
        # qi1 = (
        #     self.qi
        #     + dt * self.system.q_dot(self.ti, self.qi, self.ui)
        #     + dt2 / 2 * self.system.q_ddot(self.ti, self.qi, self.ui, a_beta)
        #     + self.system.B(self.ti, self.qi) @ Qi1
        # )

        # position update, see GAMM2022 Harsch
        a_beta = (1 - 2 * self.beta) * self.a_bari + 2 * self.beta * a_bari1
        Delta_u_bar = self.ui + dt * a_beta
        qi1 = (
            self.qi
            + dt * self.system.q_dot(self.ti, self.qi, Delta_u_bar)
            + self.system.q_dot_u(self.ti, self.qi) @ Qi1
        )

        # ----- get quantities from model -----
        # kinematic quantities of contacts
        g_N = self.system.g_N(ti1, qi1)
        xi_N = self.system.xi_N(self.ti, ti1, self.qi, qi1, self.ui, ui1)
        g_N_ddot = self.system.g_N_ddot(ti1, qi1, ui1, ai1)
        gamma_F = self.system.gamma_F(ti1, qi1, ui1)
        xi_F = self.system.xi_F(self.ti, ti1, self.qi, qi1, self.ui, ui1)
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
        prox_r_N = estimate_prox_parameter(
            self.options.prox_scaling, self.W_Ni1, self.Mi1
        )
        prox_r_F = estimate_prox_parameter(
            self.options.prox_scaling, self.W_Fi1, self.Mi1
        )

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

        # solve nonlinear system
        x, converged, error, j, R = fsolve(
            self.R,
            x,
            fun_args=(True,),
            jac_args=(False,),
            atol=self.options.newton_atol,
            max_iter=self.options.newton_max_iter,
        )

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
        for i_fixed_point in range(self.options.fixed_point_max_iter):
            # solve nonlinear system
            y, newton_converged, newton_error, i_newton, R_s = fsolve(
                self.R_s,
                y,
                fun_args=(z,),
                jac_args=(z,),
                atol=self.options.newton_atol,
                max_iter=self.options.newton_max_iter,
            )

            # eqn. (146): fixed point update
            z1 = self.p(y, z)
            fixed_point_error = self.options.error_function(z1 - z)
            fixed_point_converged = fixed_point_error < self.options.fixed_point_atol
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
            (fixed_point_converged, i_fixed_point, fixed_point_error),
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
        solver_summary = SolverSummary()

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

        # for-loop over all time steps
        for _ in pbar:
            if self.method == "newton":
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
                solver_summary.add_newton(n_iter)
                pbar.set_description(
                    f"t: {ti1:0.2e}s < {self.t1:0.2e}s; {n_iter}/{self.options.newton_max_iter} iterations; error: {error:0.2e}"
                )
                if not converged:
                    raise RuntimeError(
                        f"step not converged after {n_iter} steps with error: {error:.5e}"
                    )
            else:
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
                solver_summary.add_fixed_point(n_iter, error)
                pbar.set_description(
                    f"t: {ti1:0.2e}s < {self.t1:0.2e}s; {n_iter}/{self.options.fixed_point_max_iter} iterations; error: {error:0.2e}"
                )
                if not converged:
                    raise RuntimeError(
                        f"fixed-point step not converged after {n_iter} steps with error: {error:.5e}"
                    )

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

            # # eqn. (125): generalized position update formula
            # a_beta = (1 - 2 * self.beta) * self.a_bari + 2 * self.beta * a_bari1
            # qi1 = (
            #     self.qi
            #     + dt * self.system.q_dot(self.ti, self.qi, self.ui)
            #     + dt2 / 2 * self.system.q_ddot(self.ti, self.qi, self.ui, a_beta)
            #     + self.system.B(self.ti, self.qi) @ Qi1
            # )

            # position update, see GAMM2022 Harsch
            a_beta = (1 - 2 * self.beta) * self.a_bari + 2 * self.beta * a_bari1
            Delta_u_bar = self.ui + dt * a_beta
            qi1 = (
                self.qi
                + dt * self.system.q_dot(self.ti, self.qi, Delta_u_bar)
                + self.system.q_dot_u(self.ti, self.qi) @ Qi1
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
            qi1, ui1 = self.system.step_callback(ti1, qi1, ui1)

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

        solver_summary.print()

        return Solution(
            system=self.system,
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
