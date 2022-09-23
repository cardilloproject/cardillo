from cardillo.math.numerical_derivative import Numerical_derivative
import numpy as np
from math import cos, sin

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.model import Model
from cardillo.solver import (
    Moreau,
    NonsmoothEulerBackwardsGGL_V2,
    NonsmoothTheta,
    NonsmoothGeneralizedAlpha,
    NonsmoothThetaGGL,
    NonsmoothGenAlphaFirstOrder,
)
from cardillo.solver.MoreauGGL import MoreauGGLInvertM, MoreauGGLWorkingSolution


class Slider_crank:
    def __init__(self, mu=5 / 3, q0=None, u0=None):
        """Flores2011, Section 4"""
        self.nq = 3
        self.nu = 3
        self.nla_N = 4

        # geometric characteristics
        self.l1 = 0.1530
        self.l2 = 0.306
        self.a = 0.05
        self.b = 0.025
        self.c = 0.001
        # self.c = 0.01
        self.d = 2 * self.c + 2 * self.b

        # inertial properties
        self.m1 = 0.038
        self.m2 = 0.038
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
        self.prox_r_N = r * np.ones(4)
        self.prox_r_F = r * np.ones(4)

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

        omega10 = 150
        omega20 = -75
        omega30 = 0
        # omega30 = 5

        self.q0 = np.array([theta10, theta20, theta30]) if q0 is None else q0
        self.u0 = np.array([omega10, omega20, omega30]) if u0 is None else u0
        self.la_N0 = np.zeros(self.nla_N)
        self.la_F0 = np.zeros(self.nla_F)

    def contour_crank(self, q):
        theta1, _, _ = q
        x = np.array([0, self.l1 * cos(theta1)])
        y = np.array([0, self.l1 * sin(theta1)])
        return x, y

    def contour_connecting_rod(self, q):
        x1, y1 = self.contour_crank(q)

        _, theta2, _ = q
        x = x1[1] + np.array([0, self.l2 * cos(theta2)])
        y = y1[1] + np.array([0, self.l2 * sin(theta2)])
        return x, y

    def contour_slider(self, q):
        x2, y2 = self.contour_connecting_rod(q)
        r_OS = np.array([x2[1], y2[1]])

        K_r_SP1 = np.array([-self.a, self.b])
        K_r_SP2 = np.array([self.a, self.b])
        K_r_SP3 = np.array([-self.a, -self.b])
        K_r_SP4 = np.array([self.a, -self.b])

        _, _, theta3 = q
        A_IK = np.array([[cos(theta3), -sin(theta3)], [sin(theta3), cos(theta3)]])

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
    def M_dense(self, t, q):
        theta1, theta2, theta3 = q

        M11 = self.J1 + (self.m1 / 4 + self.m2 + self.m3) * self.l1**2  # (90)
        M12 = M21 = (
            (self.m2 / 2 + self.m3) * self.l1 * self.l2 * cos(theta2 - theta1)
        )  # (91)
        M13 = M31 = M23 = M32 = 0  # (92)
        M22 = self.J2 + (self.m2 / 4 + self.m3) * self.l2**2  # (93)
        M33 = self.J3  # (94)

        return np.array([[M11, M12, M13], [M21, M22, M23], [M31, M32, M33]])

    def M(self, t, q, coo):
        coo.extend(self.M_dense(t, q), (self.uDOF, self.uDOF))

    def Mu_q(self, t, q, u, coo):

        M_q = np.zeros((3, 3, 3))

        theta1, theta2, theta3 = q
        M12_theta1 = (
            (self.m2 / 2 + self.m3)
            * self.l1
            * self.l2
            * (-cos(theta2) * sin(theta1) + sin(theta2) * cos(theta1))
        )
        M_q[0, 1, 0] = M12_theta1
        M_q[1, 0, 0] = M12_theta1
        M12_theta2 = (
            (self.m2 / 2 + self.m3)
            * self.l1
            * self.l2
            * (-sin(theta2) * cos(theta1) + cos(theta2) * sin(theta1))
        )
        M_q[0, 1, 1] = M12_theta2
        M_q[1, 0, 1] = M12_theta2

        dense = np.einsum("ijk,j->ik", M_q, u)

        # Mu = lambda t, q: self.M_dense(t, q) @ u
        # dense_num = Numerical_derivative(Mu, order=2)._x(t, q)
        # error = np.linalg.norm(dense_num - dense)
        # print(f'error Mu_q: {error}')

        coo.extend(dense, (self.uDOF, self.qDOF))

    def f_npot(self, t, q, u):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u
        factor1 = (self.m2 / 2 + self.m3) * self.l1 * self.l2 * sin(theta2 - theta1)
        h1 = factor1 * omega2**2 - (
            self.m1 / 2 + self.m2 + self.m3
        ) * self.g * self.l1 * cos(
            theta1
        )  # (95)
        h2 = -factor1 * omega1**2 - (self.m2 / 2 + self.m3) * self.g * self.l2 * cos(
            theta2
        )  # (96)
        h3 = 0
        return np.array([h1, h2, h3])

    def f_npot_q(self, t, q, u, coo):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u

        factor1_theta1 = (
            (self.m2 / 2 + self.m3)
            * self.l1
            * self.l2
            * (-sin(theta2) * sin(theta1) - cos(theta2) * cos(theta1))
        )
        factor1_theta2 = (
            (self.m2 / 2 + self.m3)
            * self.l1
            * self.l2
            * (cos(theta2) * cos(theta1) + sin(theta2) * sin(theta1))
        )

        dense = np.zeros((3, 3))
        dense[0, 0] = factor1_theta1 * omega2**2 + (
            self.m1 / 2 + self.m2 + self.m3
        ) * self.g * self.l1 * sin(theta1)
        dense[0, 1] = factor1_theta2 * omega2**2
        dense[1, 0] = -factor1_theta1 * omega1**2
        dense[1, 1] = -factor1_theta2 * omega1**2 + (
            self.m2 / 2 + self.m3
        ) * self.g * self.l2 * sin(theta2)

        # dense_num = Numerical_derivative(self.f_npot, order=2)._x(t, q, u)
        # error = np.linalg.norm(dense_num - dense)
        # print(f'error f_npot_q: {error}')

        coo.extend(dense, (self.uDOF, self.qDOF))

    def f_npot_u(self, t, q, u, coo):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u
        factor1 = (self.m2 / 2 + self.m3) * self.l1 * self.l2 * sin(theta2 - theta1)

        dense = np.zeros((3, 3))
        dense[0, 1] = 2 * factor1 * omega2
        dense[1, 0] = -2 * factor1 * omega1

        # dense_num = Numerical_derivative(self.f_npot, order=2)._y(t, q, u)
        # error = np.linalg.norm(dense_num - dense)
        # print(f'error f_npot_u: {error}')

        coo.extend(dense, (self.uDOF, self.uDOF))

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q, coo):
        coo.extend_diag(np.ones(3), (self.qDOF, self.uDOF))

    #################
    # normal contacts
    #################
    def g_N(self, t, q):
        theta1, theta2, theta3 = q
        g_N1 = (
            self.d / 2
            - self.l1 * sin(theta1)
            - self.l2 * sin(theta2)
            + self.a * sin(theta3)
            - self.b * cos(theta3)
        )
        g_N2 = (
            self.d / 2
            - self.l1 * sin(theta1)
            - self.l2 * sin(theta2)
            - self.a * sin(theta3)
            - self.b * cos(theta3)
        )
        g_N3 = (
            self.d / 2
            + self.l1 * sin(theta1)
            + self.l2 * sin(theta2)
            - self.a * sin(theta3)
            - self.b * cos(theta3)
        )
        g_N4 = (
            self.d / 2
            + self.l1 * sin(theta1)
            + self.l2 * sin(theta2)
            + self.a * sin(theta3)
            - self.b * cos(theta3)
        )
        return np.array([g_N1, g_N2, g_N3, g_N4])

    def g_N_q_dense(self, t, q):
        theta1, theta2, theta3 = q
        w_N1 = np.array(
            [
                -self.l1 * cos(theta1),
                -self.l2 * cos(theta2),
                self.a * cos(theta3) + self.b * sin(theta3),
            ]
        )
        w_N2 = np.array(
            [
                -self.l1 * cos(theta1),
                -self.l2 * cos(theta2),
                -self.a * cos(theta3) + self.b * sin(theta3),
            ]
        )
        w_N3 = np.array(
            [
                self.l1 * cos(theta1),
                self.l2 * cos(theta2),
                -self.a * cos(theta3) + self.b * sin(theta3),
            ]
        )
        w_N4 = np.array(
            [
                self.l1 * cos(theta1),
                self.l2 * cos(theta2),
                self.a * cos(theta3) + self.b * sin(theta3),
            ]
        )

        g_N_q_dense = np.vstack((w_N1, w_N2, w_N3, w_N4))

        # g_N_q_dense_num = Numerical_derivative(self.g_N, order=2)._x(t, q)
        # error = np.linalg.norm(g_N_q_dense_num - g_N_q_dense)
        # print(f'error g_N_q_dense: {error}')

        return g_N_q_dense

    def g_N_q(self, t, q, coo):
        coo.extend(self.g_N_q_dense(t, q), (self.la_NDOF, self.qDOF))

    def g_N_dot(self, t, q, u):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u
        g_N1_dot = (
            -self.l1 * cos(theta1) * omega1
            - self.l2 * cos(theta2) * omega2
            + self.a * cos(theta3) * omega3
            + self.b * sin(theta3) * omega3
        )
        g_N2_dot = (
            -self.l1 * cos(theta1) * omega1
            - self.l2 * cos(theta2) * omega2
            - self.a * cos(theta3) * omega3
            + self.b * sin(theta3) * omega3
        )
        g_N3_dot = (
            +self.l1 * cos(theta1) * omega1
            + self.l2 * cos(theta2) * omega2
            - self.a * cos(theta3) * omega3
            + self.b * sin(theta3) * omega3
        )
        g_N4_dot = (
            +self.l1 * cos(theta1) * omega1
            + self.l2 * cos(theta2) * omega2
            + self.a * cos(theta3) * omega3
            + self.b * sin(theta3) * omega3
        )
        return np.array([g_N1_dot, g_N2_dot, g_N3_dot, g_N4_dot])

    # TODO!
    def g_N_dot_q_dense(self, t, q, u):
        return Numerical_derivative(self.g_N_dot, order=2)._x(t, q, np.zeros(self.nu))

    def g_N_dot_q(self, t, q, u, coo):
        coo.extend(self.g_N_dot_q_dense(t, q, u), (self.la_NDOF, self.qDOF))

    def g_N_dot_u_dense(self, t, q):
        return self.g_N_q_dense(t, q)

    def g_N_dot_u(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q), (self.la_NDOF, self.uDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def xi_N_q(self, t, q, u_pre, u_post, coo):
        g_N_q_pre = self.g_N_dot_q_dense(t, q, u_pre)
        g_N_q_post = self.g_N_dot_q_dense(t, q, u_post)
        dense = g_N_q_post + np.diag(self.e_N) @ g_N_q_pre
        coo.extend(dense, (self.la_NDOF, self.qDOF))

    def W_N(self, t, q, coo):
        dense = self.g_N_dot_u_dense(t, q).T
        coo.extend(dense, (self.uDOF, self.la_NDOF))

    def g_N_ddot(self, t, q, u, u_dot):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u
        omega1_dot, omega2_dot, omega3_dot = u_dot
        g_N1_ddot = (
            self.l1 * sin(theta1) * omega1**2
            - self.l1 * cos(theta1) * omega1_dot
            + self.l2 * sin(theta2) * omega2**2
            - self.l2 * cos(theta2) * omega2_dot
            - self.a * sin(theta3) * omega3**2
            + self.a * cos(theta3) * omega3_dot
            + self.b * cos(theta3) * omega3**2
            + self.b * sin(theta3) * omega3_dot
        )
        g_N2_ddot = (
            self.l1 * sin(theta1) * omega1**2
            - self.l1 * cos(theta1) * omega1_dot
            + self.l2 * sin(theta2) * omega2**2
            - self.l2 * cos(theta2) * omega2_dot
            + self.a * sin(theta3) * omega3**2
            - self.a * cos(theta3) * omega3_dot
            + self.b * cos(theta3) * omega3**2
            + self.b * sin(theta3) * omega3_dot
        )
        g_N3_ddot = (
            -self.l1 * sin(theta1) * omega1**2
            + self.l1 * cos(theta1) * omega1_dot
            - self.l2 * sin(theta2) * omega2**2
            + self.l2 * cos(theta2) * omega2_dot
            + self.a * sin(theta3) * omega3**2
            - self.a * cos(theta3) * omega3_dot
            + self.b * cos(theta3) * omega3**2
            + self.b * sin(theta3) * omega3_dot
        )
        g_N4_ddot = (
            -self.l1 * sin(theta1) * omega1**2
            + self.l1 * cos(theta1) * omega1_dot
            - self.l2 * sin(theta2) * omega2**2
            + self.l2 * cos(theta2) * omega2_dot
            - self.a * sin(theta3) * omega3**2
            + self.a * cos(theta3) * omega3_dot
            + self.b * cos(theta3) * omega3**2
            + self.b * sin(theta3) * omega3_dot
        )
        return np.array([g_N1_ddot, g_N2_ddot, g_N3_ddot, g_N4_ddot])

    # TODO:
    def g_N_ddot_q(self, t, q, u, u_dot, coo):
        dense = Numerical_derivative(
            lambda t, q, u: self.g_N_ddot(t, q, u, u_dot), order=2
        )._x(t, q, u)
        coo.extend(dense, (self.la_NDOF, self.qDOF))

    # TODO:
    def g_N_ddot_u(self, t, q, u, u_dot, coo):
        dense = Numerical_derivative(
            lambda t, q, u: self.g_N_ddot(t, q, u, u_dot), order=2
        )._y(t, q, u)
        coo.extend(dense, (self.la_NDOF, self.uDOF))

    # TODO:
    def Wla_N_q(self, t, q, la_N, coo):
        Wla_N = lambda t, q: self.g_N_dot_u_dense(t, q).T @ la_N
        dense = Numerical_derivative(Wla_N, order=2)._x(t, q)
        coo.extend(dense, (self.uDOF, self.qDOF))

    #################
    # tanget contacts
    #################
    def __gamma_F(self, t, q, u):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u
        gamma_1 = (
            -self.l1 * sin(theta1) * omega1
            - self.l2 * sin(theta2) * omega2
            + self.a * sin(theta3) * omega3
            - self.b * cos(theta3) * omega3
        )
        gamma_2 = (
            -self.l1 * sin(theta1) * omega1
            - self.l2 * sin(theta2) * omega2
            - self.a * sin(theta3) * omega3
            - self.b * cos(theta3) * omega3
        )
        gamma_3 = (
            -self.l1 * sin(theta1) * omega1
            - self.l2 * sin(theta2) * omega2
            + self.a * sin(theta3) * omega3
            + self.b * cos(theta3) * omega3
        )
        gamma_4 = (
            -self.l1 * sin(theta1) * omega1
            - self.l2 * sin(theta2) * omega2
            - self.a * sin(theta3) * omega3
            + self.b * cos(theta3) * omega3
        )
        return np.array([gamma_1, gamma_2, gamma_3, gamma_4])

    def gamma_F_q_dense(self, t, q, u):
        return Numerical_derivative(self.__gamma_F, order=2)._x(t, q, u)

    def gamma_F_q(self, t, q, u, coo):
        coo.extend(self.gamma_F_q_dense(t, q, u), (self.la_FDOF, self.qDOF))

    def gamma_F_u_dense(self, t, q):
        return Numerical_derivative(self.__gamma_F, order=2)._y(t, q, np.zeros(self.nu))

    def gamma_F_u(self, t, q, coo):
        coo.extend(self.gamma_F_u_dense(t, q), (self.la_FDOF, self.uDOF))

    def W_F(self, t, q, coo):
        coo.extend(self.gamma_F_u_dense(t, q).T, (self.uDOF, self.la_FDOF))

    def Wla_F_q(self, t, q, la_T, coo):
        Wla_T = lambda t, q: self.gamma_F_u_dense(t, q).T @ la_T
        dense = Numerical_derivative(Wla_T, order=2)._x(t, q)
        coo.extend(dense, (self.uDOF, self.qDOF))

    def gamma_F_dot(self, t, q, u, u_dot):
        theta1, theta2, theta3 = q
        omega1, omega2, omega3 = u
        omega1_dot, omega2_dot, omega3_dot = u_dot
        gamma_1_dot = (
            -self.l1 * cos(theta1) * omega1**2
            - self.l1 * sin(theta1) * omega1_dot
            - self.l2 * cos(theta2) * omega2**2
            - self.l2 * sin(theta2) * omega2_dot
            + self.a * cos(theta3) * omega3**2
            + self.a * sin(theta3) * omega3_dot
            + self.b * sin(theta3) * omega3**2
            - self.b * cos(theta3) * omega3_dot
        )
        gamma_2_dot = (
            -self.l1 * cos(theta1) * omega1**2
            - self.l1 * sin(theta1) * omega1_dot
            - self.l2 * cos(theta2) * omega2**2
            - self.l2 * sin(theta2) * omega2_dot
            - self.a * cos(theta3) * omega3**2
            - self.a * sin(theta3) * omega3_dot
            + self.b * sin(theta3) * omega3**2
            - self.b * cos(theta3) * omega3_dot
        )
        gamma_3_dot = (
            -self.l1 * cos(theta1) * omega1**2
            - self.l1 * sin(theta1) * omega1_dot
            - self.l2 * cos(theta2) * omega2**2
            - self.l2 * sin(theta2) * omega2_dot
            + self.a * cos(theta3) * omega3**2
            + self.a * sin(theta3) * omega3_dot
            - self.b * sin(theta3) * omega3**2
            + self.b * cos(theta3) * omega3_dot
        )
        gamma_4_dot = (
            -self.l1 * cos(theta1) * omega1**2
            - self.l1 * sin(theta1) * omega1_dot
            - self.l2 * cos(theta2) * omega2**2
            - self.l2 * sin(theta2) * omega2_dot
            - self.a * cos(theta3) * omega3**2
            - self.a * sin(theta3) * omega3_dot
            - self.b * sin(theta3) * omega3**2
            + self.b * cos(theta3) * omega3_dot
        )
        return np.array([gamma_1_dot, gamma_2_dot, gamma_3_dot, gamma_4_dot])

    def gamma_F_dot_q(self, t, q, u, u_dot, coo):
        dense = Numerical_derivative(
            lambda t, q, u: self.gamma_F_dot(t, q, u, u_dot), order=2
        )._x(t, q, u)
        coo.extend(dense, (self.la_FDOF, self.qDOF))

    def gamma_F_dot_u(self, t, q, u, u_dot, coo):
        dense = Numerical_derivative(
            lambda t, q, u: self.gamma_F_dot(t, q, u, u_dot), order=2
        )._y(t, q, u)
        coo.extend(dense, (self.la_FDOF, self.uDOF))

    def xi_F(self, t, q, u_pre, u_post):
        return self.gamma_F(t, q, u_post) + self.e_F * self.gamma_F(t, q, u_pre)

    def xi_F_q(self, t, q, u_pre, u_post, coo):
        gamma_T_q_pre = self.gamma_F_q_dense(t, q, u_pre)
        gamma_T_q_post = self.gamma_F_q_dense(t, q, u_post)
        dense = gamma_T_q_post + np.diag(self.e_F) @ gamma_T_q_pre
        coo.extend(dense, (self.la_FDOF, self.qDOF))


if __name__ == "__main__":
    animate = True
    # animate = False

    model = Model()
    slider_crank = Slider_crank()
    model.add(slider_crank)
    model.assemble()

    t1 = 0.5
    # dt = 1e-5
    # dt = 1e-4
    # dt = 2.5e-4
    # dt = 5e-4
    dt = 1e-3

    # TODO: This example seems to be broken!
    # solver = Moreau(model, t1, dt, fix_point_max_iter=5000)
    # solver = MoreauGGL(model, t1, dt)
    # solver = MoreauGGLInvertM(model, t1, dt)
    # solver = MoreauGGLWorkingSolution(model, t1, dt)
    # solver = NonsmoothNewmarkFirstOrder(model, t1, dt, atol=1.0e-6)
    # solver = NonsmoothGeneralizedAlpha(model, t1, dt, rho_inf=0.85)
    # solver = NonsmoothTheta(model, t1, dt)
    # solver = NonsmoothGenAlphaFirstOrder(model, t1, dt, rho_inf=0.85)
    solver = NonsmoothEulerBackwardsGGL_V2(model, t1, dt)
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
