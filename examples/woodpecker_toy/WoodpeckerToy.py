# This file implements the woodpecker toy example,
# see Glocker 1995 or Glocker 2001,

import numpy as np

from cardillo.math.prox import Sphere


class WoodpeckerToy:
    def __init__(self, q0, u0):
        # dimensions
        self.nq = 3
        self.nu = 3
        self.nla_N = 3
        self.nla_F = 3

        # initial conditions
        self.q0 = q0
        self.u0 = u0

        # inertia
        self.m_M = 0.0003  # kg
        self.J_M = 5e-9  # kg m2
        self.m_S = 0.0045  # kg
        self.J_C = 7e-7  # kg m2

        # dynamics
        self.c_phi = 0.0056  # Nm / rad
        self.gravity = 9.81  # m / s2

        # geometry
        self.r_0 = 0.0025  # m
        self.r_M = 0.0031  # m
        self.h_M = 0.0058  # m
        self.l_M = 0.01  # m
        self.l_G = 0.015  # m
        self.h_S = 0.020  # m
        self.l_S = 0.0201  # m

        # contact
        self.mu = 0.3
        self.e_N = np.array([0.5, 0, 0])
        self.e_F = np.zeros(3)
        # fmt: off
        self.friction_laws = [
            ([0], [0], Sphere(self.mu)), # Coulomb
            ([1], [1], Sphere(self.mu)), # Coulomb
            ([2], [2], Sphere(self.mu)), # Coulomb
        ]
        # fmt: on

    #####################
    # kinematic equation
    #####################
    def q_dot(self, t, q, u):
        return u

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return np.array(
            [
                [self.m_S + self.m_M, self.m_S * self.l_M, self.m_S * self.l_G],
                [
                    self.m_S * self.l_M,
                    self.J_M + self.m_S * self.l_M**2,
                    self.m_S * self.l_M * self.l_G,
                ],
                [
                    self.m_S * self.l_G,
                    self.m_S * self.l_M * self.l_G,
                    self.J_C + self.m_S * self.l_G**2,
                ],
            ]
        )

    def h(self, t, q, u):
        y, phi_M, phi_S = q
        return np.array(
            [
                -(self.m_S + self.m_M) * self.gravity,
                -self.c_phi * (phi_M - phi_S) - self.m_S * self.l_M * self.gravity,
                -self.c_phi * (phi_S - phi_M) - self.m_S * self.l_G * self.gravity,
            ]
        )

    #################
    # normal contacts
    #################
    def g_N(self, t, q):
        y, phi_M, phi_S = q
        return np.array(
            [
                self.l_M + self.l_G - self.l_S - self.r_0 - self.h_S * phi_S,
                self.r_M - self.r_0 + self.h_M * phi_M,
                self.r_M - self.r_0 - self.h_M * phi_M,
            ]
        )

    def W_N(self, t, q):
        W_N = np.zeros((self.nu, self.nla_N))
        W_N[2, 0] = -self.h_S
        W_N[1, 1] = self.h_M
        W_N[1, 2] = -self.h_M
        return W_N

    def g_N_dot(self, t, q, u):
        return self.W_N(t, q).T @ u

    ##########
    # friction
    ##########
    def W_F(self, t, q):
        W_F = np.zeros((self.nu, self.nla_F))
        W_F[0, 0] = 1
        W_F[1, 0] = self.l_M
        W_F[2, 0] = self.l_G - self.l_S
        W_F[0, 1] = 1
        W_F[1, 1] = self.r_M
        W_F[0, 2] = 1
        W_F[1, 2] = self.r_M
        return W_F

    def gamma_F(self, t, q, u):
        return self.W_F(t, q).T @ u
