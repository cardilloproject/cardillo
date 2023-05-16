import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math import approx_fprime

from cardillo import System
from cardillo.solver import (
    MoreauShifted,
    Rattle,
    MoreauClassical,
    NPIRK,
)
from cardillo.solver._butcher_tableaus import RadauIIATableau


class SliderCrankFlores:
    def __init__(self, q0=None, u0=None):
        """Flores2011, Section 4: Demonstrative Application to a Slider-Crank Mechanism.

        References:
        -----------
        Flores2011:  https://doi.org/10.1007/978-90-481-9971-6_6
        """
        self.nq = 3
        self.nu = 3
        self.nla_N = 4

        # geometric characteristics
        self.l1 = 0.1530
        self.l2 = 0.306
        self.a = 0.05
        self.b = 0.025
        self.c = 0.001
        self.d = 2 * self.c + 2 * self.b

        # inertial properties
        self.m1 = self.m2 = 0.038
        self.m3 = 0.076
        self.J1 = 7.4e-5
        self.J2 = 5.9e-4
        self.J3 = 2.7e-6

        # gravity
        self.g = 9.81

        # contact parameters
        self.e_N = 0.4 * np.ones(4)
        self.e_F = np.zeros(4)
        mu = 0.01
        # mu = 0
        self.mu = mu * np.ones(4)
        # r = 0.1
        # r = 0.01
        # r = 0.001
        r = 0.0005
        # self.prox_r_N = r * np.ones(4)
        # self.prox_r_F = r * np.ones(4)

        if mu == 0:
            self.nla_F = 0
            self.NF_connectivity = [[], [], [], []]
        else:
            self.nla_F = 4
            self.NF_connectivity = [[0], [1], [2], [3]]
            self.gamma_F = self.__gamma_F

        # initial conditions
        theta10 = 0
        theta20 = 0
        theta30 = 0
        # theta30 = 5 * np.pi / 180

        omega10 = 150
        omega20 = -75
        omega30 = 0

        self.q0 = np.array([theta10, theta20, theta30]) if q0 is None else q0
        self.u0 = np.array([omega10, omega20, omega30]) if u0 is None else u0
        self.la_N0 = np.zeros(self.nla_N)
        self.la_F0 = np.zeros(self.nla_F)

    def contour_crank(self, q):
        theta1, _, _ = q
        x = np.array([0, self.l1 * np.cos(theta1)])
        y = np.array([0, self.l1 * np.sin(theta1)])
        return x, y

    def contour_connecting_rod(self, q):
        x1, y1 = self.contour_crank(q)

        _, theta2, _ = q
        x = x1[1] + np.array([0, self.l2 * np.cos(theta2)])
        y = y1[1] + np.array([0, self.l2 * np.sin(theta2)])
        return x, y

    def contour_slider(self, q):
        x2, y2 = self.contour_connecting_rod(q)
        r_OS = np.array([x2[1], y2[1]])

        K_r_SP1 = np.array([-self.a, self.b])
        K_r_SP2 = np.array([self.a, self.b])
        K_r_SP3 = np.array([-self.a, -self.b])
        K_r_SP4 = np.array([self.a, -self.b])

        _, _, theta3 = q
        # fmt: off
        A_IK = np.array([
            [np.cos(theta3), -np.sin(theta3)], 
            [np.sin(theta3),  np.cos(theta3)]
        ])
        # fmt: on

        r_SP1 = r_OS + A_IK @ K_r_SP1
        r_SP2 = r_OS + A_IK @ K_r_SP2
        r_SP3 = r_OS + A_IK @ K_r_SP3
        r_SP4 = r_OS + A_IK @ K_r_SP4

        x = np.array([r_OS[0], r_SP1[0], r_SP2[0], r_SP4[0], r_SP3[0], r_SP1[0]])
        y = np.array([r_OS[1], r_SP1[1], r_SP2[1], r_SP4[1], r_SP3[1], r_SP1[1]])
        return x, y

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        theta1, theta2, theta3 = q

        M11 = self.J1 + (self.m1 / 4 + self.m2 + self.m3) * self.l1**2  # (90)
        M12 = M21 = (
            (self.m2 / 2 + self.m3) * self.l1 * self.l2 * np.cos(theta2 - theta1)
        )  # (91)
        M13 = M31 = M23 = M32 = 0  # (92)
        M22 = self.J2 + (self.m2 / 4 + self.m3) * self.l2**2  # (93)
        M33 = self.J3  # (94)

        # fmt: off
        return np.array([
            [M11, M12, M13], 
            [M21, M22, M23], 
            [M31, M32, M33],
        ])
        # fmt: on

    def Mu_q(self, t, q, u):
        M_q = np.zeros((3, 3, 3))

        theta1, theta2, theta3 = q
        M12_theta1 = (
            (self.m2 / 2 + self.m3)
            * self.l1
            * self.l2
            * (-np.cos(theta2) * np.sin(theta1) + np.sin(theta2) * np.cos(theta1))
        )
        M_q[0, 1, 0] = M12_theta1
        M_q[1, 0, 0] = M12_theta1
        M12_theta2 = (
            (self.m2 / 2 + self.m3)
            * self.l1
            * self.l2
            * (-np.sin(theta2) * np.cos(theta1) + np.cos(theta2) * np.sin(theta1))
        )
        M_q[0, 1, 1] = M12_theta2
        M_q[1, 0, 1] = M12_theta2

        Mu_q = np.einsum("ijk,j->ik", M_q, u)
        return Mu_q

        # Mu = lambda t, q: self.M(t, q) @ u
        # Mu_q_num = Numerical_derivative(Mu, order=2)._x(t, q)
        # error = np.linalg.norm(Mu_q_num - Mu_q)
        # print(f'error Mu_q: {error}')

    def h(self, t, q, u):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u
        factor1 = (self.m2 / 2 + self.m3) * self.l1 * self.l2 * np.sin(theta2 - theta1)
        h1 = factor1 * omega2**2 - (
            self.m1 / 2 + self.m2 + self.m3
        ) * self.g * self.l1 * np.cos(
            theta1
        )  # (95)
        h2 = -factor1 * omega1**2 - (
            self.m2 / 2 + self.m3
        ) * self.g * self.l2 * np.cos(
            theta2
        )  # (96)
        h3 = 0
        return np.array([h1, h2, h3])

    def h_q(self, t, q, u):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u

        factor1_theta1 = (
            (self.m2 / 2 + self.m3)
            * self.l1
            * self.l2
            * (-np.sin(theta2) * np.sin(theta1) - np.cos(theta2) * np.cos(theta1))
        )
        factor1_theta2 = (
            (self.m2 / 2 + self.m3)
            * self.l1
            * self.l2
            * (np.cos(theta2) * np.cos(theta1) + np.sin(theta2) * np.sin(theta1))
        )

        h_q = np.zeros((3, 3))
        h_q[0, 0] = factor1_theta1 * omega2**2 + (
            self.m1 / 2 + self.m2 + self.m3
        ) * self.g * self.l1 * np.sin(theta1)
        h_q[0, 1] = factor1_theta2 * omega2**2
        h_q[1, 0] = -factor1_theta1 * omega1**2
        h_q[1, 1] = -factor1_theta2 * omega1**2 + (
            self.m2 / 2 + self.m3
        ) * self.g * self.l2 * np.sin(theta2)

        # h_q_num = Numerical_derivative(self.f_npot, order=2)._x(t, q, u)
        # error = np.linalg.norm(h_q_num - h_q)
        # print(f'error f_npot_q: {error}')

        return h_q

    def h_u(self, t, q, u):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u
        factor1 = (self.m2 / 2 + self.m3) * self.l1 * self.l2 * np.sin(theta2 - theta1)

        h_u = np.zeros((3, 3))
        h_u[0, 1] = 2 * factor1 * omega2
        h_u[1, 0] = -2 * factor1 * omega1

        # h_u_num = Numerical_derivative(self.f_npot, order=2)._y(t, q, u)
        # error = np.linalg.norm(h_u_num - h_u)
        # print(f'error f_npot_u: {error}')

        return h_u

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q):
        return np.ones(3)

    #################
    # normal contacts
    #################
    def g_N(self, t, q):
        theta1, theta2, theta3 = q
        sth1 = np.sin(theta1)
        sth2 = np.sin(theta2)
        sth3 = np.sin(theta3)
        cth3 = np.cos(theta3)
        g_N1 = (
            self.d / 2 - self.l1 * sth1 - self.l2 * sth2 + self.a * sth3 - self.b * cth3
        )
        g_N2 = (
            self.d / 2 - self.l1 * sth1 - self.l2 * sth2 - self.a * sth3 - self.b * cth3
        )
        g_N3 = (
            self.d / 2 + self.l1 * sth1 + self.l2 * sth2 - self.a * sth3 - self.b * cth3
        )
        g_N4 = (
            self.d / 2 + self.l1 * sth1 + self.l2 * sth2 + self.a * sth3 - self.b * cth3
        )
        return np.array([g_N1, g_N2, g_N3, g_N4])

    def g_N_q(self, t, q):
        theta1, theta2, theta3 = q
        cth1 = np.cos(theta1)
        cth2 = np.cos(theta2)
        cth3 = np.cos(theta3)
        sth3 = np.sin(theta3)
        w_N1 = np.array(
            [
                -self.l1 * cth1,
                -self.l2 * cth2,
                self.a * cth3 + self.b * sth3,
            ]
        )
        w_N2 = np.array(
            [
                -self.l1 * cth1,
                -self.l2 * cth2,
                -self.a * cth3 + self.b * sth3,
            ]
        )
        w_N3 = np.array(
            [
                self.l1 * cth1,
                self.l2 * cth2,
                -self.a * cth3 + self.b * sth3,
            ]
        )
        w_N4 = np.array(
            [
                self.l1 * cth1,
                self.l2 * cth2,
                self.a * cth3 + self.b * sth3,
            ]
        )

        g_N_q_dense = np.vstack((w_N1, w_N2, w_N3, w_N4))

        return g_N_q_dense

    def g_N_dot(self, t, q, u):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u
        cth1 = np.cos(theta1)
        cth2 = np.cos(theta2)
        cth3 = np.cos(theta3)
        sth3 = np.sin(theta3)
        g_N1_dot = (
            -self.l1 * cth1 * omega1
            - self.l2 * cth2 * omega2
            + self.a * cth3 * omega3
            + self.b * sth3 * omega3
        )
        g_N2_dot = (
            -self.l1 * cth1 * omega1
            - self.l2 * cth2 * omega2
            - self.a * cth3 * omega3
            + self.b * sth3 * omega3
        )
        g_N3_dot = (
            +self.l1 * cth1 * omega1
            + self.l2 * cth2 * omega2
            - self.a * cth3 * omega3
            + self.b * sth3 * omega3
        )
        g_N4_dot = (
            +self.l1 * cth1 * omega1
            + self.l2 * cth2 * omega2
            + self.a * cth3 * omega3
            + self.b * sth3 * omega3
        )
        return np.array([g_N1_dot, g_N2_dot, g_N3_dot, g_N4_dot])

    # TODO!
    def g_N_dot_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.g_N_dot(t, q, u))

    def g_N_dot_u(self, t, q):
        return self.g_N_q(t, q)

    def xi_N_q(self, t, q, u_pre, u_post):
        g_N_q_pre = self.g_N_dot_q(t, q, u_pre)
        g_N_q_post = self.g_N_dot_q(t, q, u_post)
        return g_N_q_post + np.diag(self.e_N) @ g_N_q_pre

    def W_N(self, t, q):
        return self.g_N_dot_u(t, q).T

    def g_N_ddot(self, t, q, u, u_dot):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u
        omega1_dot, omega2_dot, omega3_dot = u_dot
        g_N1_ddot = (
            self.l1 * np.sin(theta1) * omega1**2
            - self.l1 * np.cos(theta1) * omega1_dot
            + self.l2 * np.sin(theta2) * omega2**2
            - self.l2 * np.cos(theta2) * omega2_dot
            - self.a * np.sin(theta3) * omega3**2
            + self.a * np.cos(theta3) * omega3_dot
            + self.b * np.cos(theta3) * omega3**2
            + self.b * np.sin(theta3) * omega3_dot
        )
        g_N2_ddot = (
            self.l1 * np.sin(theta1) * omega1**2
            - self.l1 * np.cos(theta1) * omega1_dot
            + self.l2 * np.sin(theta2) * omega2**2
            - self.l2 * np.cos(theta2) * omega2_dot
            + self.a * np.sin(theta3) * omega3**2
            - self.a * np.cos(theta3) * omega3_dot
            + self.b * np.cos(theta3) * omega3**2
            + self.b * np.sin(theta3) * omega3_dot
        )
        g_N3_ddot = (
            -self.l1 * np.sin(theta1) * omega1**2
            + self.l1 * np.cos(theta1) * omega1_dot
            - self.l2 * np.sin(theta2) * omega2**2
            + self.l2 * np.cos(theta2) * omega2_dot
            + self.a * np.sin(theta3) * omega3**2
            - self.a * np.cos(theta3) * omega3_dot
            + self.b * np.cos(theta3) * omega3**2
            + self.b * np.sin(theta3) * omega3_dot
        )
        g_N4_ddot = (
            -self.l1 * np.sin(theta1) * omega1**2
            + self.l1 * np.cos(theta1) * omega1_dot
            - self.l2 * np.sin(theta2) * omega2**2
            + self.l2 * np.cos(theta2) * omega2_dot
            - self.a * np.sin(theta3) * omega3**2
            + self.a * np.cos(theta3) * omega3_dot
            + self.b * np.cos(theta3) * omega3**2
            + self.b * np.sin(theta3) * omega3_dot
        )
        return np.array([g_N1_ddot, g_N2_ddot, g_N3_ddot, g_N4_ddot])

    # TODO:
    def g_N_ddot_q(self, t, q, u, u_dot):
        return approx_fprime(q, lambda q: self.g_N_ddot(t, q, u, u_dot))

    # TODO:
    def g_N_ddot_u(self, t, q, u, u_dot):
        return approx_fprime(u, lambda u: self.g_N_ddot(t, q, u, u_dot))

    # TODO:
    def Wla_N_q(self, t, q, la_N):
        Wla_N = lambda q: self.g_N_dot_u(t, q).T @ la_N
        return approx_fprime(q, Wla_N)

    #################
    # tanget contacts
    #################
    def __gamma_F(self, t, q, u):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u
        sth1 = np.sin(theta1)
        sth2 = np.sin(theta2)
        sth3 = np.sin(theta3)
        cth3 = np.cos(theta3)
        gamma_1 = (
            -self.l1 * sth1 * omega1
            - self.l2 * sth2 * omega2
            + self.a * sth3 * omega3
            - self.b * cth3 * omega3
        )
        gamma_2 = (
            -self.l1 * sth1 * omega1
            - self.l2 * sth2 * omega2
            - self.a * sth3 * omega3
            - self.b * cth3 * omega3
        )
        gamma_3 = (
            -self.l1 * sth1 * omega1
            - self.l2 * sth2 * omega2
            + self.a * sth3 * omega3
            + self.b * cth3 * omega3
        )
        gamma_4 = (
            -self.l1 * sth1 * omega1
            - self.l2 * sth2 * omega2
            - self.a * sth3 * omega3
            + self.b * cth3 * omega3
        )
        return np.array([gamma_1, gamma_2, gamma_3, gamma_4])

    def gamma_F_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.__gamma_F(t, q, u))

    def gamma_F_u(self, t, q):
        theta1, theta2, theta3 = q
        sth1 = np.sin(theta1)
        sth2 = np.sin(theta2)
        sth3 = np.sin(theta3)
        cth3 = np.cos(theta3)

        w_F1 = np.array(
            [
                -self.l1 * sth1,
                -self.l2 * sth2,
                self.a * sth3 - self.b * cth3,
            ]
        )
        w_F2 = np.array(
            [
                -self.l1 * sth1,
                -self.l2 * sth2,
                -self.a * sth3 - self.b * cth3,
            ]
        )
        w_F3 = np.array(
            [
                -self.l1 * sth1,
                -self.l2 * sth2,
                self.a * sth3 + self.b * cth3,
            ]
        )
        w_F4 = np.array(
            [
                -self.l1 * sth1,
                -self.l2 * sth2,
                -self.a * sth3 + self.b * cth3,
            ]
        )

        gamma_F_u_dense = np.vstack((w_F1, w_F2, w_F3, w_F4))
        return gamma_F_u_dense

        # num = approx_fprime(np.zeros(self.nu), lambda u: self.__gamma_F(t, q, u), method="2-point", eps=1e-6)
        # diff = gamma_F_u_dense - num
        # error = np.linalg.norm(diff)
        # print(f"error gamma_F_u_dense: {error}")
        # return num

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def Wla_F_q(self, t, q, la_T):
        Wla_T = lambda t, q: self.gamma_F_u(t, q).T @ la_T
        return approx_fprime(q, Wla_T)

    def gamma_F_dot(self, t, q, u, u_dot):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u
        omega1_dot, omega2_dot, omega3_dot = u_dot
        gamma_1_dot = (
            -self.l1 * np.cos(theta1) * omega1**2
            - self.l1 * np.sin(theta1) * omega1_dot
            - self.l2 * np.cos(theta2) * omega2**2
            - self.l2 * np.sin(theta2) * omega2_dot
            + self.a * np.cos(theta3) * omega3**2
            + self.a * np.sin(theta3) * omega3_dot
            + self.b * np.sin(theta3) * omega3**2
            - self.b * np.cos(theta3) * omega3_dot
        )
        gamma_2_dot = (
            -self.l1 * np.cos(theta1) * omega1**2
            - self.l1 * np.sin(theta1) * omega1_dot
            - self.l2 * np.cos(theta2) * omega2**2
            - self.l2 * np.sin(theta2) * omega2_dot
            - self.a * np.cos(theta3) * omega3**2
            - self.a * np.sin(theta3) * omega3_dot
            + self.b * np.sin(theta3) * omega3**2
            - self.b * np.cos(theta3) * omega3_dot
        )
        gamma_3_dot = (
            -self.l1 * np.cos(theta1) * omega1**2
            - self.l1 * np.sin(theta1) * omega1_dot
            - self.l2 * np.cos(theta2) * omega2**2
            - self.l2 * np.sin(theta2) * omega2_dot
            + self.a * np.cos(theta3) * omega3**2
            + self.a * np.sin(theta3) * omega3_dot
            - self.b * np.sin(theta3) * omega3**2
            + self.b * np.cos(theta3) * omega3_dot
        )
        gamma_4_dot = (
            -self.l1 * np.cos(theta1) * omega1**2
            - self.l1 * np.sin(theta1) * omega1_dot
            - self.l2 * np.cos(theta2) * omega2**2
            - self.l2 * np.sin(theta2) * omega2_dot
            - self.a * np.cos(theta3) * omega3**2
            - self.a * np.sin(theta3) * omega3_dot
            - self.b * np.sin(theta3) * omega3**2
            + self.b * np.cos(theta3) * omega3_dot
        )
        return np.array([gamma_1_dot, gamma_2_dot, gamma_3_dot, gamma_4_dot])

    def gamma_F_dot_q(self, t, q, u, u_dot):
        return approx_fprime(q, lambda q: self.gamma_F_dot(t, q, u, u_dot))

    def gamma_F_dot_u(self, t, q, u, u_dot):
        return approx_fprime(u, lambda u: self.gamma_F_dot(t, q, u, u_dot))


class SliderCrankDAE:
    def __init__(self):
        """Inspired by Flores2011, Section 4: Demonstrative Application to a Slider-Crank Mechanism.

        References:
        -----------
        Flores2011:  https://doi.org/10.1007/978-90-481-9971-6_6
        """
        self.nq = 9
        self.nu = 9
        self.nla_g = 6
        self.nla_N = 4

        # geometric characteristics
        self.l1 = 0.1530
        self.l2 = 0.306
        self.a = 0.05
        self.b = 0.025
        self.c = 0.001
        self.d = 2 * self.c + 2 * self.b

        # inertial properties
        self.m1 = self.m2 = 0.038
        self.m3 = 0.076
        self.J1 = 7.4e-5
        self.J2 = 5.9e-4
        self.J3 = 2.7e-6

        # gravity
        self.grav = 9.81

        # contact parameters
        self.e_N = 0.4 * np.ones(4)
        self.e_F = np.zeros(4)
        mu = 0.01
        self.mu = mu * np.ones(4)

        if mu == 0:
            self.nla_F = 0
            self.NF_connectivity = [[], [], [], []]
        else:
            self.nla_F = 4
            self.NF_connectivity = [[0], [1], [2], [3]]
            self.gamma_F = self.__gamma_F

        # initial conditions
        theta10 = 0
        theta20 = 0
        # theta30 = 0
        # theta30 = 1 * np.pi / 180
        theta30 = 0.017  # approx. 1 * np.pi / 180

        r_OP10 = self.l1 * np.array([np.cos(theta10), np.sin(theta10)])
        r_P1P20 = self.l2 * np.array([np.cos(theta20), np.sin(theta20)])
        x10, y10 = 0.5 * r_OP10
        x20, y20 = r_OP10 + 0.5 * r_P1P20
        x30, y30 = r_OP10 + r_P1P20

        omega10 = 150
        omega20 = -75
        omega30 = 0

        v_P10 = self.l1 * np.array([-np.sin(theta10), np.cos(theta10)]) * omega10
        v_P1P20 = self.l2 * np.array([-np.sin(theta20), np.cos(theta20)]) * omega20
        x1_dot0, y1_dot0 = 0.5 * v_P10
        x2_dot0, y2_dot0 = v_P10 + 0.5 * v_P1P20
        x3_dot0, y3_dot0 = v_P10 + v_P1P20

        self.q0 = np.array([x10, y10, theta10, x20, y20, theta20, x30, y30, theta30])
        self.u0 = np.array(
            [
                x1_dot0,
                y1_dot0,
                omega10,
                x2_dot0,
                y2_dot0,
                omega20,
                x3_dot0,
                y3_dot0,
                omega30,
            ]
        )
        self.la_N0 = np.zeros(self.nla_N)
        self.la_F0 = np.zeros(self.nla_F)
        self.la_g0 = np.zeros(self.nla_g)

        print(f"q0: {self.q0}")
        print(f"u0: {self.u0}")
        # exit()

    def contour_crank(self, q):
        x1, y1, theta1, _, _, _, _, _, _ = q
        x = x1 + 0.5 * self.l1 * np.array([-1, 1]) * np.cos(theta1)
        y = y1 + 0.5 * self.l1 * np.array([-1, 1]) * np.sin(theta1)
        return x, y

    def contour_connecting_rod(self, q):
        _, _, _, x2, y2, theta2, _, _, _ = q
        x = x2 + 0.5 * self.l2 * np.array([-1, 1]) * np.cos(theta2)
        y = y2 + 0.5 * self.l2 * np.array([-1, 1]) * np.sin(theta2)
        return x, y

    def contour_slider(self, q):
        K_r_SP1 = np.array([-self.a, self.b])
        K_r_SP2 = np.array([self.a, self.b])
        K_r_SP3 = np.array([-self.a, -self.b])
        K_r_SP4 = np.array([self.a, -self.b])

        _, _, _, _, _, _, x3, y3, theta3 = q
        # fmt: off
        A_IK = np.array([
            [np.cos(theta3), -np.sin(theta3)], 
            [np.sin(theta3),  np.cos(theta3)]
        ])
        # fmt: on

        r_OS = np.array([x3, y3])
        r_SP1 = r_OS + A_IK @ K_r_SP1
        r_SP2 = r_OS + A_IK @ K_r_SP2
        r_SP3 = r_OS + A_IK @ K_r_SP3
        r_SP4 = r_OS + A_IK @ K_r_SP4

        x = np.array([r_OS[0], r_SP1[0], r_SP2[0], r_SP4[0], r_SP3[0], r_SP1[0]])
        y = np.array([r_OS[1], r_SP1[1], r_SP2[1], r_SP4[1], r_SP3[1], r_SP1[1]])
        return x, y

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return np.diag(
            [
                self.m1,
                self.m1,
                self.J1,
                self.m2,
                self.m2,
                self.J2,
                self.m3,
                self.m3,
                self.J3,
            ]
        )

    def h(self, t, q, u):
        return -np.array([0, self.m1, 0, 0, self.m2, 0, 0, self.m3, 0]) * self.grav

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q):
        return np.ones(self.nq)

    #######################
    # bilateral constraints
    #######################
    def g(self, t, q):
        x1, y1, theta1, x2, y2, theta2, x3, y3, theta3 = q
        sin1 = np.sin(theta1)
        cos1 = np.cos(theta1)
        sin2 = np.sin(theta2)
        cos2 = np.cos(theta2)
        return np.array(
            [
                x1 - 0.5 * self.l1 * cos1,
                y1 - 0.5 * self.l1 * sin1,
                x1 + 0.5 * self.l1 * cos1 - (x2 - 0.5 * self.l2 * cos2),
                y1 + 0.5 * self.l1 * sin1 - (y2 - 0.5 * self.l2 * sin2),
                x2 + 0.5 * self.l2 * cos2 - x3,
                y2 + 0.5 * self.l2 * sin2 - y3,
            ]
        )

    def g_dot(self, t, q, u):
        x1, y1, theta1, x2, y2, theta2, x3, y3, theta3 = q
        u1, v1, omega1, u2, v2, omega2, u3, v3, omega3 = u
        sin1 = np.sin(theta1)
        cos1 = np.cos(theta1)
        sin2 = np.sin(theta2)
        cos2 = np.cos(theta2)
        # fmt: off
        g_dot = np.array([
            u1 + 0.5 * self.l1 * sin1 * omega1,
            v1 - 0.5 * self.l1 * cos1 * omega1,
            u1 - 0.5 * self.l1 * sin1 * omega1 - (u2 + 0.5 * self.l2 * sin2 * omega2),
            v1 + 0.5 * self.l1 * cos1 * omega1 - (v2 - 0.5 * self.l2 * cos2 * omega2),
            u2 - 0.5 * self.l2 * sin2 * omega2 - u3,
            v2 + 0.5 * self.l2 * cos2 * omega2 - v3,
        ])
        # fmt: on
        return g_dot

        # g_dot_num = self.g_q_dense(t, q) @ u
        # diff = g_dot - g_dot_num
        # error = np.linalg.norm(diff)
        # print(f"error g_dot: {error}")

        # return g_dot_num

    def g_dot_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.g_dot(t, q, u))

    def g_ddot(self, t, q, u, u_dot):
        x1, y1, theta1, x2, y2, theta2, x3, y3, theta3 = q
        u1, v1, omega1, u2, v2, omega2, u3, v3, omega3 = u
        u_dot1, v_dot1, psi1, u_dot2, v_dot2, psi2, u_dot3, v_dot3, psi3 = u_dot
        sin1 = np.sin(theta1)
        cos1 = np.cos(theta1)
        sin2 = np.sin(theta2)
        cos2 = np.cos(theta2)
        # fmt: off
        g_ddot = np.array([
            # u1 + 0.5 * self.l1 * sin1 * omega1,
            # v1 - 0.5 * self.l1 * cos1 * omega1,
            # u1 - 0.5 * self.l1 * sin1 * omega1 - (u2 + 0.5 * self.l2 * sin2 * omega2),
            # v1 + 0.5 * self.l1 * cos1 * omega1 - (v2 - 0.5 * self.l2 * cos2 * omega2),
            # u2 - 0.5 * self.l2 * sin2 * omega2 - u3,
            # v2 + 0.5 * self.l2 * cos2 * omega2 - v3,
            u_dot1 + 0.5 * self.l1 * sin1 * psi1 + 0.5 * self.l1 * cos1 * omega1**2,
            v_dot1 - 0.5 * self.l1 * cos1 * psi1 + 0.5 * self.l1 * sin1 * omega1**2,
            u_dot1 - 0.5 * self.l1 * sin1 * psi1 - 0.5 * self.l1 * cos1 * omega1**2 - (u_dot2 + 0.5 * self.l2 * sin2 * psi2 + 0.5 * self.l2 * cos2 * omega2**2),
            v_dot1 + 0.5 * self.l1 * cos1 * psi1 - 0.5 * self.l1 * sin1 * omega1**2 - (v_dot2 - 0.5 * self.l2 * cos2 * psi2 + 0.5 * self.l2 * sin2 * omega2**2),
            u_dot2 - 0.5 * self.l2 * sin2 * psi2 - 0.5 * self.l2 * cos2 * omega2**2 - u_dot3,
            v_dot2 + 0.5 * self.l2 * cos2 * psi2 - 0.5 * self.l2 * sin2 * omega2**2 - v_dot3,
        ])
        # fmt: on
        return g_ddot

        # g_ddot_num = self.g_q_dense(t, q) @ u_dot + approx_fprime(q, lambda q: self.g_dot(t, q, u), method="3-point", eps=1e-6) @ u
        # diff = g_ddot - g_ddot_num
        # error = np.linalg.norm(diff)
        # print(f"error g_ddot: {error}")

        # return g_ddot_num

    def g_q_dense(self, t, q):
        x1, y1, theta1, x2, y2, theta2, x3, y3, theta3 = q
        sin1 = np.sin(theta1)
        cos1 = np.cos(theta1)
        sin2 = np.sin(theta2)
        cos2 = np.cos(theta2)

        g_q = np.zeros((self.nla_g, self.nq), dtype=float)
        g_q[0, 0] = 1
        g_q[0, 2] = 0.5 * self.l1 * sin1

        g_q[1, 1] = 1
        g_q[1, 2] = -0.5 * self.l1 * cos1

        g_q[2, 0] = 1
        g_q[2, 2] = -0.5 * self.l1 * sin1
        g_q[2, 3] = -1
        g_q[2, 5] = -0.5 * self.l2 * sin2

        g_q[3, 1] = 1
        g_q[3, 2] = 0.5 * self.l1 * cos1
        g_q[3, 4] = -1
        g_q[3, 5] = 0.5 * self.l2 * cos2

        g_q[4, 3] = 1
        g_q[4, 5] = -0.5 * self.l2 * sin2
        g_q[4, 6] = -1

        g_q[5, 4] = 1
        g_q[5, 5] = 0.5 * self.l2 * cos2
        g_q[5, 7] = -1

        return g_q

        # g_q_num = approx_fprime(q, lambda q: self.g(t, q), method="2-point", eps=1e-6)
        # diff = g_q - g_q_num
        # error = np.linalg.norm(diff)
        # print(f"error g_q: {error}")
        # return g_q_num

    def g_q(self, t, q):
        return self.g_q_dense(t, q)

    def W_g(self, t, q):
        return self.g_q_dense(t, q).T

    # TODO:
    def Wla_g_q(self, t, q, la_g):
        Wla_g = lambda q: self.g_q_dense(t, q).T @ la_g
        return approx_fprime(q, Wla_g)

    #################
    # normal contacts
    #################
    def g_N(self, t, q):
        _, _, _, _, _, _, x3, y3, theta3 = q
        sin = np.sin(theta3)
        cos = np.cos(theta3)
        g_N1 = 0.5 * self.d - (y3 - self.a * sin + self.b * cos)
        g_N2 = 0.5 * self.d - (y3 + self.a * sin + self.b * cos)
        g_N3 = 0.5 * self.d + (y3 - self.a * sin - self.b * cos)
        g_N4 = 0.5 * self.d + (y3 + self.a * sin - self.b * cos)
        return np.array([g_N1, g_N2, g_N3, g_N4])

    def g_N_q(self, t, q):
        _, _, _, _, _, _, x3, y3, theta3 = q
        sin = np.sin(theta3)
        cos = np.cos(theta3)

        g_N_q = np.zeros((self.nla_N, self.nq), dtype=float)
        g_N_q[0, -2:] = -np.array([1, -self.a * cos - self.b * sin])
        g_N_q[1, -2:] = -np.array([1, self.a * cos - self.b * sin])
        g_N_q[2, -2:] = np.array([1, -self.a * cos + self.b * sin])
        g_N_q[3, -2:] = np.array([1, self.a * cos + self.b * sin])

        return g_N_q

    def g_N_dot(self, t, q, u):
        return self.g_N_q(t, q) @ u

    def g_N_ddot(self, t, q, u, u_dot):
        return (
            self.g_N_q(t, q) @ u_dot
            + approx_fprime(q, lambda q: self.g_N_dot(t, q, u)) @ u
        )

    def g_N_dot_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.g_N_dot(t, q, u))

    def W_N(self, t, q):
        return self.g_N_q(t, q).T

    # TODO:
    def Wla_N_q(self, t, q, la_N):
        Wla_N = lambda q: self.g_N_q(t, q).T @ la_N
        return approx_fprime(q, Wla_N)

    #################
    # tanget contacts
    #################
    def __gamma_F(self, t, q, u):
        _, _, _, _, _, _, _, _, theta3 = q
        _, _, _, _, _, _, u3, _, omega3 = u
        sin = np.sin(theta3)
        cos = np.cos(theta3)
        gamma_1 = u3 + omega3 * (self.a * sin - self.b * cos)
        gamma_2 = u3 + omega3 * (-self.a * sin - self.b * cos)
        gamma_3 = u3 + omega3 * (self.a * sin + self.b * cos)
        gamma_4 = u3 + omega3 * (-self.a * sin + self.b * cos)
        return np.array([gamma_1, gamma_2, gamma_3, gamma_4])

    def gamma_F_dot(self, t, q, u, u_dot):
        return (
            self.W_F(t, q).T @ u_dot
            + approx_fprime(q, lambda q: self.gamma_F(t, q, u)) @ u
        )

    def gamma_F_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.__gamma_F(t, q, u))

    def W_F(self, t, q):
        _, _, _, _, _, _, x3, y3, theta3 = q
        sin = np.sin(theta3)
        cos = np.cos(theta3)

        W_F = np.zeros((self.nu, self.nla_F), dtype=float)
        W_F[-2:, 0] = np.array([1, self.a * sin - self.b * cos])
        W_F[-2:, 1] = np.array([1, -self.a * sin - self.b * cos])
        W_F[-2:, 2] = np.array([1, self.a * sin + self.b * cos])
        W_F[-2:, 3] = np.array([1, -self.a * sin + self.b * cos])
        return W_F

    # TODO:
    def Wla_F_q(self, t, q, la_F):
        Wla_F = lambda q: self.W_F(t, q) @ la_F
        return approx_fprime(q, Wla_F)

    def xi_F_q(self, t, q, u_pre, u_post):
        gamma_T_q_pre = self.gamma_F_q(t, q, u_pre)
        gamma_T_q_post = self.gamma_F_q(t, q, u_post)
        return gamma_T_q_post + np.diag(self.e_F) @ gamma_T_q_pre


def run_Flores():
    animate = True
    # animate = False

    system = System()
    slider_crank = SliderCrankFlores()
    system.add(slider_crank)
    system.assemble()

    t1 = 4 * np.pi / 150
    # t1 = 0.5
    # dt = 1e-5
    dt = 1e-4
    # dt = 2.5e-4
    # dt = 5e-4
    # dt = 1e-3
    # dt = 5e-3

    solver = MoreauShifted(system, t1, dt, fix_point_max_iter=5000)

    sol = solver.solve()
    t = sol.t
    q = sol.q
    u = sol.u

    # positions
    fig, ax = plt.subplots(3, 3)
    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("theta1 [rad]")
    ax[0, 0].plot(t, q[:, 0], "-k")

    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("theta2 [rad]")
    ax[0, 1].plot(t, q[:, 1], "-k")

    ax[0, 2].set_xlabel("t [s]")
    ax[0, 2].set_ylabel("theta3 [rad]")
    ax[0, 2].plot(t, q[:, 2], "-k")

    # velocities
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel("omega1 [rad/s]")
    ax[1, 0].plot(t, u[:, 0], "-k")

    ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].set_ylabel("omega2 [rad/s]")
    ax[1, 1].plot(t, u[:, 1], "-k")

    ax[1, 2].set_xlabel("t [s]")
    ax[1, 2].set_ylabel("omega3 [rad/s]")
    ax[1, 2].plot(t, u[:, 2], "-k")

    # gaps
    nt = len(t)
    g_N = np.zeros((nt, 4))
    g_N_dot = np.zeros((nt, 4))
    # gamma_T = np.zeros(nt)

    for i, ti in enumerate(t):
        g_N[i] = slider_crank.g_N(ti, q[i])
        g_N_dot[i] = slider_crank.g_N_dot(ti, q[i], u[i])
        # gamma_T[i] = silder_crank.gamma_T(ti, q[i], u[i])

    ax[2, 0].set_xlabel("t [s]")
    ax[2, 0].set_ylabel("g_N [m]")
    ax[2, 0].plot(t, g_N[:, 0], "-k", label="g_N1")
    ax[2, 0].plot(t, g_N[:, 1], "--k", label="g_N2")
    ax[2, 0].plot(t, g_N[:, 2], "-.k", label="g_N3")
    ax[2, 0].plot(t, g_N[:, 3], ":k", label="g_N4")
    ax[2, 0].legend(shadow=True, fancybox=True)

    ax[2, 1].set_xlabel("t [s]")
    ax[2, 1].set_ylabel("g_N_dot [m/s]")
    ax[2, 1].plot(t, g_N_dot[:, 0], "-k", label="g_N1_dot")
    ax[2, 1].plot(t, g_N_dot[:, 1], "--k", label="g_N2_dot")
    ax[2, 1].plot(t, g_N_dot[:, 2], "-.k", label="g_N3_dot")
    ax[2, 1].plot(t, g_N_dot[:, 3], ":k", label="g_N4_dot")
    ax[2, 1].legend(shadow=True, fancybox=True)

    # ax[2, 2].set_xlabel('t [s]')
    # ax[2, 2].set_ylabel('gamma_T [m/s]')
    # ax[2, 2].plot(t, gamma_T, '-k')

    if not animate:
        plt.show()

    if animate:
        fig_anim, ax_anim = plt.subplots()

        ax_anim.set_xlabel("x [m]")
        ax_anim.set_ylabel("y [m]")
        ax_anim.axis("equal")
        ax_anim.set_xlim(-0.2, 0.6)
        ax_anim.set_ylim(-0.4, 0.4)

        # prepare data for animation
        slowmotion = 10
        fps = 25
        animation_time = slowmotion * t1
        target_frames = int(fps * animation_time)
        frac = max(1, int(len(t) / target_frames))
        if frac == 1:
            target_frames = len(t)
        interval = 1000 / fps

        frames = target_frames
        t = t[::frac]
        q = q[::frac]

        ax_anim.plot(
            np.array([0, slider_crank.l1 + slider_crank.l2 + 4 * slider_crank.b]),
            +slider_crank.d / 2 * np.ones(2),
            "-k",
        )
        ax_anim.plot(
            np.array([0, slider_crank.l1 + slider_crank.l2 + 4 * slider_crank.b]),
            -slider_crank.d / 2 * np.ones(2),
            "-k",
        )

        (line1,) = ax_anim.plot(
            *slider_crank.contour_crank(slider_crank.q0), "-ok", linewidth=2
        )
        (line2,) = ax_anim.plot(
            *slider_crank.contour_connecting_rod(slider_crank.q0), "-ob", linewidth=2
        )
        (line3,) = ax_anim.plot(
            *slider_crank.contour_slider(slider_crank.q0), "-or", linewidth=2
        )

        def animate(i):
            line1.set_data(*slider_crank.contour_crank(q[i]))
            line2.set_data(*slider_crank.contour_connecting_rod(q[i]))
            line3.set_data(*slider_crank.contour_slider(q[i]))
            return (
                line1,
                line2,
                line3,
            )

        anim = animation.FuncAnimation(
            fig_anim, animate, frames=frames, interval=interval, blit=False
        )
        plt.show()

        # writer = animation.writers['ffmpeg'](fps=fps, bitrate=1800)
        # anim.save('slider_crank.mp4', writer=writer)


def run_DAE(export=True):
    animate = True
    # animate = False

    system = System()
    slider_crank = SliderCrankDAE()
    system.add(slider_crank)
    system.assemble()

    # approx. two crank revolutions
    t_final = 7 * np.pi / 150
    # t_final *= 0.1
    # dt1 = 1e-4
    dt1 = 1e-3
    dt2 = 1e-3

    sol1, label1 = (
        NPIRK(system, t_final, dt1, RadauIIATableau(2)).solve(),
        "NPIRK",
    )

    # sol1, label1 = Rattle(system, t_final, dt1, atol=1e-8).solve(), "Rattle"
    # sol1, label1 = (
    #     MoreauShifted(system, t_final, dt2, fix_point_max_iter=5).solve(),
    #     "MoreauShifted",
    # )
    sol2, label2 = (
        MoreauClassical(system, t_final, dt2, max_iter=500).solve(),
        "MoreauClassical",
    )
    # sol2, label2 = (
    #     NPIRK(system, t_final, 0.1 * dt1, RadauIIATableau(2)).solve(),
    #     "NPIRK",
    # )

    t1 = sol1.t
    q1 = sol1.q
    u1 = sol1.u
    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u

    fig, ax = plt.subplots(2, 3)

    # positions
    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("theta1 [rad]")
    ax[0, 0].plot(t1, q1[:, 2], "-k", label=label1)
    ax[0, 0].plot(t2, q2[:, 2], "--r", label=label2)
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("theta2 [rad]")
    ax[0, 1].plot(t1, q1[:, 5], "-k", label=label1)
    ax[0, 1].plot(t2, q2[:, 5], "--r", label=label2)
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[0, 2].set_xlabel("t [s]")
    ax[0, 2].set_ylabel("theta3 [rad]")
    ax[0, 2].plot(t1, q1[:, 8], "-k", label=label1)
    ax[0, 2].plot(t2, q2[:, 8], "--r", label=label2)
    ax[0, 2].grid()
    ax[0, 2].legend()

    # velocities
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel("omega1 [rad/s]")
    ax[1, 0].plot(t1, u1[:, 2], "-k", label=label1)
    ax[1, 0].plot(t2, u2[:, 2], "--r", label=label2)
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].set_ylabel("omega2 [rad/s]")
    ax[1, 1].plot(t1, u1[:, 5], "-k", label=label1)
    ax[1, 1].plot(t2, u2[:, 5], "--r", label=label2)
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[1, 2].set_xlabel("t [s]")
    ax[1, 2].set_ylabel("omega3 [rad/s]")
    ax[1, 2].plot(t1, u1[:, 8], "-k", label=label1)
    ax[1, 2].plot(t2, u2[:, 8], "--r", label=label2)
    ax[1, 2].grid()
    ax[1, 2].legend()

    plt.tight_layout()

    # bilateral constraints
    fig, ax = plt.subplots(2, 3)
    g1 = np.array([system.g(ti, qi) for (ti, qi) in zip(t1, q1)])
    g2 = np.array([system.g(ti, qi) for (ti, qi) in zip(t2, q2)])
    for i in range(2):
        for j in range(3):
            idx = 3 * i + j
            ax[i, j].plot(t1, g1[:, idx], "-k", label=label1)
            ax[i, j].plot(t2, g2[:, idx], "--r", label=label2)
            ax[i, j].grid()
            ax[i, j].legend()
            ax[i, j].set_xlabel("t [s]")
            ax[i, j].set_ylabel(f"g{idx + 1} [m]")

    plt.tight_layout()

    fig, ax = plt.subplots(2, 3)
    g_dot1 = np.array([system.g_dot(ti, qi, ui) for (ti, qi, ui) in zip(t1, q1, u1)])
    g_dot2 = np.array([system.g_dot(ti, qi, ui) for (ti, qi, ui) in zip(t2, q2, u2)])
    for i in range(2):
        for j in range(3):
            idx = 3 * i + j
            ax[i, j].plot(t1, g_dot1[:, idx], "-k", label=label1)
            ax[i, j].plot(t2, g_dot2[:, idx], "--r", label=label2)
            ax[i, j].grid()
            ax[i, j].legend()
            ax[i, j].set_xlabel("t [s]")
            ax[i, j].set_ylabel(f"g_dot{idx + 1} [m]")

    plt.tight_layout()

    # contacts
    fig, ax = plt.subplots(3, 4)
    g_N1 = np.array([system.g_N(ti, qi) for (ti, qi) in zip(t1, q1)])
    g_N_dot1 = np.array(
        [system.g_N_dot(ti, qi, ui) for (ti, qi, ui) in zip(t1, q1, u1)]
    )
    gamma_F1 = np.array(
        [system.gamma_F(ti, qi, ui) for (ti, qi, ui) in zip(t1, q1, u1)]
    )
    g_N2 = np.array([system.g_N(ti, qi) for (ti, qi) in zip(t2, q2)])
    g_N_dot2 = np.array(
        [system.g_N_dot(ti, qi, ui) for (ti, qi, ui) in zip(t2, q2, u2)]
    )
    gamma_F2 = np.array(
        [system.gamma_F(ti, qi, ui) for (ti, qi, ui) in zip(t2, q2, u2)]
    )
    for i in range(4):
        ax[0, i].plot(t1, g_N1[:, i], "-k", label=label1)
        ax[0, i].plot(t2, g_N2[:, i], "--r", label=label2)
        ax[0, i].grid()
        ax[0, i].legend()
        ax[0, i].set_xlabel("t [s]")
        ax[0, i].set_ylabel(f"g_N{i + 1} [m]")

        ax[1, i].plot(t1, g_N_dot1[:, i], "-k", label=label1)
        ax[1, i].plot(t2, g_N_dot2[:, i], "--r", label=label2)
        ax[1, i].grid()
        ax[1, i].legend()
        ax[1, i].set_xlabel("t [s]")
        ax[1, i].set_ylabel(f"g_N_dot{i + 1} [m/s]")

        ax[2, i].plot(t1, gamma_F1[:, i], "-k", label=label1)
        ax[2, i].plot(t2, gamma_F2[:, i], "--r", label=label2)
        ax[2, i].grid()
        ax[2, i].legend()
        ax[2, i].set_xlabel("t [s]")
        ax[2, i].set_ylabel(f"gamma_F_dot{i + 1} [m/s]")

    plt.tight_layout()

    if export:
        path = Path(__file__)

        np.savetxt(
            path.parent / "state1.dat",
            np.hstack((sol1.t[:, None], q1, u1)),
            delimiter=", ",
            header="t, x1, y1, phi1, x2, y2, phi2, x3, y3, phi3, u1, v1, omega1, u2, v2, omega2, u3, v3, omega3",
            comments="",
        )

        np.savetxt(
            path.parent / "g1.dat",
            np.hstack((sol1.t[:, None], g1)),
            delimiter=", ",
            header="t, g1, g2, g3, g4, g5, g6",
            comments="",
        )

        np.savetxt(
            path.parent / "g_N1.dat",
            np.hstack((sol1.t[:, None], g_N1, g_N1 * 1e3)),
            delimiter=", ",
            header="t, g_N1, g_N2, g_N3, g_N4, g_N1_1000, g_N2_1000, g_N3_1000, g_N4_1000",
            comments="",
        )

        np.savetxt(
            path.parent / "g_N_dot1.dat",
            np.hstack((sol1.t[:, None], g_N_dot1)),
            delimiter=", ",
            header="t, g_N_dot1, g_N_dot2, g_N_dot3, g_N_dot4",
            comments="",
        )

        # np.savetxt(
        #     path.parent / "state2.dat",
        #     np.hstack((sol2.t[:, None], q2, u2)),
        #     delimiter=", ",
        #     header="t, x1, y1, phi1, x2, y2, phi2, x3, y3, phi3, u1, v1, omega1, u2, v2, omega2, u3, v3, omega3",
        #     comments="",
        # )

        # np.savetxt(
        #     path.parent / "g2.dat",
        #     np.hstack((sol1.t[:, None], g2)),
        #     delimiter=", ",
        #     header="t, g1, g2, g3, g4, g5, g6",
        #     comments="",
        # )

        # np.savetxt(
        #     path.parent / "g_N2.dat",
        #     np.hstack((sol1.t[:, None], g_N2)),
        #     delimiter=", ",
        #     header="t, g_N1, g_N2, g_N3, g_N4",
        #     comments="",
        # )

    if not animate:
        plt.show()

    if animate:
        fig_anim, ax_anim = plt.subplots()

        ax_anim.set_xlabel("x [m]")
        ax_anim.set_ylabel("y [m]")
        ax_anim.axis("equal")
        ax_anim.set_xlim(-0.2, 0.6)
        ax_anim.set_ylim(-0.4, 0.4)

        # prepare data for animation
        slowmotion = 100
        fps = 25
        animation_time = slowmotion * t_final
        target_frames = max(len(t1), int(fps * animation_time))
        frac = max(1, int(len(t1) / target_frames))
        if frac == 1:
            target_frames = len(t1)
        interval = 1000 / fps

        frames = target_frames
        t1 = t1[::frac]
        q1 = q1[::frac]

        ax_anim.plot(
            np.array([0, slider_crank.l1 + slider_crank.l2 + 4 * slider_crank.b]),
            +slider_crank.d / 2 * np.ones(2),
            "-k",
        )
        ax_anim.plot(
            np.array([0, slider_crank.l1 + slider_crank.l2 + 4 * slider_crank.b]),
            -slider_crank.d / 2 * np.ones(2),
            "-k",
        )

        (line1,) = ax_anim.plot(
            *slider_crank.contour_crank(slider_crank.q0), "-ok", linewidth=2
        )
        (line2,) = ax_anim.plot(
            *slider_crank.contour_connecting_rod(slider_crank.q0), "-ob", linewidth=2
        )
        (line3,) = ax_anim.plot(
            *slider_crank.contour_slider(slider_crank.q0), "-or", linewidth=2
        )

        def animate(i):
            line1.set_data(*slider_crank.contour_crank(q1[i]))
            line2.set_data(*slider_crank.contour_connecting_rod(q1[i]))
            line3.set_data(*slider_crank.contour_slider(q1[i]))
            return (
                line1,
                line2,
                line3,
            )

        anim = animation.FuncAnimation(
            fig_anim, animate, frames=frames, interval=interval, blit=False
        )
        plt.show()

        # writer = animation.writers['ffmpeg'](fps=fps, bitrate=1800)
        # anim.save('slider_crank.mp4', writer=writer)


if __name__ == "__main__":
    # run_Flores()
    run_DAE()
