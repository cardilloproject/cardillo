import numpy as np
from math import pi

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo import System
from cardillo.solver import (
    MoreauShifted,
    NonsmoothGeneralizedAlpha,
    Rattle,
    NonsmoothPIRK,
    RadauIIATableau,
)


class Painleve_rod:
    def __init__(self, mu=5 / 3, q0=None, u0=None):
        """Glocker1995, 5.3.4 Stoß ohne Kollision"""

        self.m = 1
        self.s = 1
        self.J_S = self.m * self.s**2 / 3
        self.g = 10
        self.nu = 0

        self.mu = np.array([mu])
        self.e_N = np.array([0])
        self.e_F = np.array([0])
        # dt = 1e-3, Moreau
        self.prox_r_N = np.array([0.4])
        self.prox_r_F = np.array([0.4])
        # # dt = 1e-3, gen-alpha
        # self.prox_r_N = np.array([0.1])
        # self.prox_r_F = np.array([0.1])

        self.NF_connectivity = [[0]]

        x0 = 0
        phi0 = 31 / 180 * pi
        # y0 = 0.515
        y0 = np.sin(phi0) * self.s  # 0.5150380749100542

        x_dot0 = 30
        y_dot0 = 0
        phi_dot0 = 0

        self.q0 = np.array([x0, y0, phi0]) if q0 is None else q0
        self.u0 = np.array([x_dot0, y_dot0, phi_dot0]) if u0 is None else u0
        self.la_N0 = np.zeros(1)
        self.la_F0 = np.zeros(1)
        # self.la_F0 = np.zeros(0)

        self.nq = 3
        self.nu = 3
        self.nla_N = 1
        self.nla_F = 1
        # self.nla_F = 0

    #####################
    # equations of motion
    #####################
    def M(self, t, q, coo):
        coo.extend_diag(np.array([self.m, self.m, self.J_S]), (self.uDOF, self.uDOF))

    def h(self, t, q, u):
        return np.array([0, -self.m * self.g, 0])

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
        x, y, phi = q
        return np.array([y - self.s * np.sin(phi)])

    def g_N_q_dense(self, t, q):
        x, y, phi = q
        return np.array([[0, 1, -self.s * np.cos(phi)]])

    def g_N_q(self, t, q, coo):
        coo.extend(self.g_N_q_dense(t, q), (self.la_NDOF, self.qDOF))

    def g_N_dot(self, t, q, u):
        x, y, phi = q
        x_dot, y_dot, phi_dot = u
        return np.array([y_dot - self.s * np.cos(phi) * phi_dot])

    def g_N_dot_q_dense(self, t, q, u):
        x, y, phi = q
        x_dot, y_dot, phi_dot = u
        return np.array([0, 0, self.s * np.sin(phi) * phi_dot])

    def g_N_dot_q(self, t, q, u, coo):
        coo.extend(self.g_N_dot_q_dense(t, q, u), (self.la_NDOF, self.qDOF))

    def g_N_dot_u_dense(self, t, q):
        x, y, phi = q
        return np.array([0, 1, -self.s * np.cos(phi)])

    def g_N_dot_u(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q), (self.la_NDOF, self.uDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def xi_N_q(self, t, q, u_pre, u_post, coo):
        g_N_q_pre = self.g_N_dot_q_dense(t, q, u_pre)
        g_N_q_post = self.g_N_dot_q_dense(t, q, u_post)
        dense = g_N_q_post + self.e_N * g_N_q_pre
        coo.extend(dense, (self.la_NDOF, self.qDOF))

    def W_N(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q).T, (self.uDOF, self.la_NDOF))

    def g_N_ddot(self, t, q, u, u_dot):
        x, y, phi = q
        x_dot, y_dot, phi_dot = u
        x_ddot, y_ddot, phi_ddot = u_dot
        return np.array(
            [
                y_ddot
                + self.s * np.sin(phi) * phi_dot**2
                - self.s * np.cos(phi) * phi_ddot
            ]
        )

    def g_N_ddot_q(self, t, q, u, u_dot, coo):
        x, y, phi = q
        x_dot, y_dot, phi_dot = u
        x_ddot, y_ddot, phi_ddot = u_dot
        dense = np.array(
            [
                0,
                0,
                self.s * np.cos(phi) * phi_dot**2 + self.s * np.sin(phi) * phi_ddot,
            ]
        )
        coo.extend(dense, (self.la_NDOF, self.qDOF))

    def g_N_ddot_u(self, t, q, u, u_dot, coo):
        x, y, phi = q
        x_dot, y_dot, phi_dot = u
        dense = np.array([0, 0, 2 * self.s * np.sin(phi) * phi_dot])
        coo.extend(dense, (self.la_NDOF, self.uDOF))

    def Wla_N_q(self, t, q, la_N, coo):
        x, y, phi = q
        dense = la_N[0] * np.array([[0, 0, 0], [0, 0, 0], [0, 0, self.s * np.sin(phi)]])
        coo.extend(dense, (self.uDOF, self.qDOF))

    #################
    # tanget contacts
    #################
    def gamma_F(self, t, q, u):
        x, y, phi = q
        x_dot, y_dot, phi_dot = u
        return np.array([x_dot - self.s * np.sin(phi) * phi_dot])

    def gamma_F_q_dense(self, t, q, u):
        x, y, phi = q
        x_dot, y_dot, phi_dot = u
        return np.array([[0, 0, -self.s * np.cos(phi) * phi_dot]])

    def gamma_F_q(self, t, q, u, coo):
        coo.extend(self.gamma_F_q_dense(t, q, u), (self.la_FDOF, self.qDOF))

    def gamma_F_u_dense(self, t, q):
        x, y, phi = q
        return np.array([[1, 0, -self.s * np.sin(phi)]])

    def gamma_F_u(self, t, q, coo):
        coo.extend(self.gamma_F_u_dense(t, q), (self.la_FDOF, self.uDOF))

    def W_F(self, t, q, coo):
        coo.extend(self.gamma_F_u_dense(t, q).T, (self.uDOF, self.la_FDOF))

    def Wla_T_q(self, t, q, la_T, coo):
        x, y, phi = q
        dense = la_T[0] * np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, -self.s * np.cos(phi)]]
        )
        coo.extend(dense, (self.uDOF, self.qDOF))

    def gamma_F_dot(self, t, q, u, u_dot):
        x, y, phi = q
        x_dot, y_dot, phi_dot = u
        x_ddot, y_ddot, phi_ddot = u_dot
        return np.array(
            [
                x_ddot
                - self.s * np.cos(phi) * phi_dot**2
                - self.s * np.sin(phi) * phi_ddot
            ]
        )

    def gamma_F_dot_q(self, t, q, u, u_dot, coo):
        x, y, phi = q
        x_dot, y_dot, phi_dot = u
        x_ddot, y_ddot, phi_ddot = u_dot
        dense = np.array(
            [
                [
                    0,
                    0,
                    self.s * np.sin(phi) * phi_dot**2
                    - self.s * np.cos(phi) * phi_ddot,
                ]
            ]
        )
        coo.extend(dense, (self.la_FDOF, self.qDOF))

    def gamma_F_dot_u(self, t, q, u, u_dot, coo):
        x, y, phi = q
        x_dot, y_dot, phi_dot = u
        x_ddot, y_ddot, phi_ddot = u_dot
        dense = np.array([[0, 0, -2 * self.s * np.cos(phi) * phi_dot]])
        coo.extend(dense, (self.la_FDOF, self.uDOF))

    def xi_F(self, t, q, u_pre, u_post):
        return self.gamma_T(t, q, u_post) + self.e_F * self.gamma_T(t, q, u_pre)

    def xi_F_q(self, t, q, u_pre, u_post, coo):
        gamma_T_q_pre = self.gamma_T_q_dense(t, q, u_pre)
        gamma_T_q_post = self.gamma_T_q_dense(t, q, u_post)
        dense = gamma_T_q_post + self.e_F * gamma_T_q_pre
        coo.extend(dense, (self.la_FDOF, self.qDOF))


if __name__ == "__main__":
    animate = True

    system = System()
    rod = Painleve_rod()
    system.add(rod)
    system.assemble()

    t_final = 1.5
    dt1 = 5e-3
    dt2 = 5e-3

    sol1, label1 = (
        NonsmoothPIRK(system, t_final, dt1, RadauIIATableau(2)).solve(),
        "NPIRK",
    )

    # sol1, label1 = Rattle(system, t_final, dt1, atol=1e-10).solve(), "Rattle"

    sol2, label2 = (
        MoreauShifted(system, t_final, dt2, fix_point_tol=1e-6).solve(),
        "Moreau",
    )

    t1 = sol1.t
    q1 = sol1.q
    u1 = sol1.u
    P_N1 = sol1.P_N
    P_F1 = sol1.P_F
    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u
    P_N2 = sol2.P_N
    P_F2 = sol2.P_F

    t_a = 0.83

    #######
    # plots
    #######

    fig, ax = plt.subplots(2, 3)

    # positions
    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("x [m]")
    ax[0, 0].plot(t1, q1[:, 0], "-k", label=label1)
    ax[0, 0].plot(t2, q2[:, 0], "--r", label=label2)
    ax[0, 0].plot([t_a, t_a], [min(q1[:, 0]), max(q1[:, 0])], "--k")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("y [m]")
    ax[0, 1].plot(t1, q1[:, 1], "-k", label=label1)
    ax[0, 1].plot(t2, q2[:, 1], "--r", label=label2)
    ax[0, 1].plot([t_a, t_a], [min(q1[:, 1]), max(q1[:, 1])], "--k")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[0, 2].set_xlabel("t [s]")
    ax[0, 2].set_ylabel("phi [rad]")
    ax[0, 2].plot(t1, q1[:, 2], "-k", label=label1)
    ax[0, 2].plot(t2, q2[:, 2], "--r", label=label2)
    ax[0, 2].plot([t_a, t_a], [min(q1[:, 2]), max(q1[:, 2])], "--k")
    ax[0, 2].grid()
    ax[0, 2].legend()

    # velocities
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel("x_dot [m/s]")
    ax[1, 0].plot(t1, u1[:, 0], "-k", label=label1)
    ax[1, 0].plot(t2, u2[:, 0], "--r", label=label2)
    ax[1, 0].plot([t_a, t_a], [min(u1[:, 0]), max(u1[:, 0])], "--k")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].set_ylabel("y_dot [m/s]")
    ax[1, 1].plot(t1, u1[:, 1], "-k", label=label1)
    ax[1, 1].plot(t2, u2[:, 1], "--r", label=label2)
    ax[1, 1].plot([t_a, t_a], [min(u1[:, 1]), max(u1[:, 1])], "--k")
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[1, 2].set_xlabel("t [s]")
    ax[1, 2].set_ylabel("phi_dot [rad/s]")
    ax[1, 2].plot(t1, u1[:, 2], "-k", label=label1)
    ax[1, 2].plot(t2, u2[:, 2], "--r", label=label2)
    ax[1, 2].plot([t_a, t_a], [min(u1[:, 2]), max(u1[:, 2])], "--k")
    ax[1, 2].grid()
    ax[1, 2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(2, 3)

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

    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("g_N [m]")
    ax[0, 0].plot(t1, g_N1, "-k", label=label1)
    ax[0, 0].plot(t2, g_N2, "--r", label=label2)
    ax[0, 0].plot([t_a, t_a], [min(g_N1), max(g_N1)], "--k")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("g_N_dot [m/s]")
    ax[0, 1].plot(t1, g_N_dot1, "-k", label=label1)
    ax[0, 1].plot(t2, g_N_dot2, "--r", label=label2)
    ax[0, 1].plot([t_a, t_a], [min(g_N_dot1), max(g_N_dot1)], "--k")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[0, 2].set_xlabel("t [s]")
    ax[0, 2].set_ylabel("gamma_T [m/s]")
    ax[0, 2].plot(t1, gamma_F1, "-k", label=label1)
    ax[0, 2].plot(t2, gamma_F2, "--r", label=label2)
    ax[0, 2].plot([t_a, t_a], [min(gamma_F1), max(gamma_F1)], "--k")
    ax[0, 2].grid()
    ax[0, 2].legend()

    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel("P_N")
    ax[1, 0].plot(t1, P_N1[:, 0], "-k", label=label1)
    ax[1, 0].plot(t2, P_N2[:, 0], "--r", label=label2)
    ax[1, 0].plot([t_a, t_a], [min(P_N1), max(P_N1)], "--k")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[1, 2].set_xlabel("t [s]")
    ax[1, 2].set_ylabel("P_F1")
    ax[1, 2].plot(t1, P_F1[:, 0], "-k", label=label1)
    ax[1, 2].plot(t2, P_F2[:, 0], "--r", label=label2)
    ax[1, 2].plot([t_a, t_a], [min(P_F1[:, 0]), max(P_F1[:, 0])], "--k")
    ax[1, 2].grid()
    ax[1, 2].legend()

    plt.tight_layout()

    ###########
    # animation
    ###########
    fig_anim, ax_anim = plt.subplots()

    ax_anim.set_xlabel("x [m]")
    ax_anim.set_ylabel("y [m]")
    ax_anim.axis("equal")
    l = 30
    ax_anim.set_xlim(-1, l - 1)
    ax_anim.set_ylim(-l / 2, l / 2)

    # prepare data for animation
    slowmotion = 2
    fps = 25
    animation_time = slowmotion * t_final
    target_frames = int(fps * animation_time)
    frac = max(1, int(len(t1) / target_frames))
    if frac == 1:
        target_frames = len(t1)
    interval = 1000 / fps

    frames = target_frames
    t1 = t1[::frac]
    q1 = q1[::frac]

    (line,) = ax_anim.plot([-1, l - 1], [0, 0], "-k")

    def configuration_rod(q):
        x, y, phi = q
        s = rod.s
        x = np.array([x - s * np.cos(phi), x + s * np.cos(phi)])
        y = np.array([y + s * np.sin(phi), y - s * np.sin(phi)])
        return x, y

    (line,) = ax_anim.plot(*configuration_rod(rod.q0), "-r", linewidth=2)

    def animate(i):
        x, y = configuration_rod(q1[i])
        line.set_data(x, y)
        return (line,)

    anim = animation.FuncAnimation(
        fig_anim, animate, frames=frames, interval=interval, blit=False
    )
    plt.show()
