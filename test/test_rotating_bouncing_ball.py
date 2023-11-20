import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo import System
from cardillo.solver import (
    Moreau,
    BackwardEuler,
    NonsmoothGeneralizedAlpha,
    SolverOptions,
    SolverOptions,
    Rattle,
)


class RotatingBouncingBall:
    def __init__(self, mass, radius, gravity, e_N, e_F, mu, q0, u0):
        self.mass = mass
        self.radius = radius
        self.Theta_S = 2 / 5 * mass * radius**2
        self.gravity = gravity

        self.nq = 3
        self.nu = 3
        self.q0 = q0
        self.u0 = u0
        assert self.nq == len(q0)
        assert self.nu == len(u0)

        self.nla_N = 1
        self.nla_F = 1
        self.NF_connectivity = [[0]]
        self.mu = np.atleast_1d(mu)
        self.e_N = np.atleast_1d(e_N)
        self.e_F = np.atleast_1d(e_F)

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def q_dot_u(self, t, q):
        return np.eye(self.nq)

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return np.diag([self.mass, self.mass, self.Theta_S])

    def h(self, t, q, u):
        return np.array([0, -self.gravity * self.mass, 0])

    #################
    # normal contacts
    #################
    def g_N(self, t, q):
        return np.array([q[1] - self.radius])

    def g_N_dot(self, t, q, u):
        return np.array([u[1]])

    def g_N_ddot(self, t, q, u, u_dot):
        return np.array([u_dot[1]])

    def g_N_q(self, t, q):
        g_N_q = np.zeros((self.nla_N, self.nq), dtype=q.dtype)
        g_N_q[0, 1] = 1
        return g_N_q

    def W_N(self, t, q):
        return self.g_N_q(t, q).T

    def Wla_N_q(self, t, q, la_N):
        return np.zeros((self.nu, self.nq))

    ##########
    # friction
    ##########
    def gamma_F(self, t, q, u):
        return np.array([u[0] + self.radius * u[2]])

    def gamma_F_u(self, t, q):
        gamma_F_u = np.zeros((self.nla_F, self.nu), dtype=q.dtype)
        gamma_F_u[0, 0] = 1
        gamma_F_u[0, 2] = self.radius
        return gamma_F_u

    def gamma_F_dot(self, t, q, u, u_dot):
        return np.array([u_dot[0] + self.radius * u_dot[2]])

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def Wla_F_q(self, t, q, la_F):
        return np.zeros((self.nu, self.nq))

    ###############
    # visualization
    ###############
    def boundary(self, t, q, num=100):
        x, y, phi = q
        # fmt: off
        A_IK = np.array([
            [np.cos(phi), -np.sin(phi)], 
            [np.sin(phi),  np.cos(phi)]
        ])
        # fmt: on

        def A_IK(theta):
            # fmt: off
            return np.array([
                [np.cos(phi + theta), -np.sin(phi + theta)], 
                [np.sin(phi + theta),  np.cos(phi + theta)]
            ])
            # fmt: on

        phis = np.linspace(0, 2 * np.pi, num=num, endpoint=True)

        r_OS = np.array([x, y])
        r_OPs = np.array(
            [r_OS + A_IK(phi) @ np.array([self.radius, 0]) for phi in phis]
        ).T
        return np.concatenate((r_OS[:, None], r_OPs), axis=-1)


def run(case, export=False):
    """Example 10.1 of Capobianco2021.

    Three different cases are implemented:
    * case 1: e_N = 0.5, omega = 0 => Figure 1
    * case 2: e_N = 0, omega = 50 => Figure 2 (left)
    * case 3: e_N = 0, omega = 10 => Figure 2 (right)
    * case 4: TODO

    References:
    -----------
    Capobianco2021: https://doi.org/10.1002/nme.6801
    """

    y0 = 1
    y_dot0 = 0
    # dt = 5e-4
    # dt = 1e-3
    dt = 5e-3
    # dt = 1e-2
    # dt = 5e-2

    if case == 1:
        e_N, e_F, mu = 0.5, 0, 0
        x0 = 0
        x_dot0 = 0
        omega = 0
        t_final = 1.5
    elif case == 2:
        e_N, e_F, mu = 0, 0, 0.2
        x0 = 0
        x_dot0 = 0
        omega = 50
        t_final = 1.1
    elif case == 3:
        e_N, e_F, mu = 0, 0, 0.2
        x0 = 0
        x_dot0 = 0
        omega = 10
        t_final = 1.1
    elif case == 4:
        # more complex example for benchmarking purposes
        e_N, e_F, mu = 0.5, 0, 0.2
        x0 = -0.5
        x_dot0 = 1
        omega = 50
        # t_final = 1.5
        t_final = 2
    else:
        raise AssertionError("Case not found!")

    q0 = np.array([x0, y0, 0], dtype=float)
    u0 = np.array([x_dot0, y_dot0, omega], dtype=float)

    # # TODO: Remove this
    # q0 = np.array([x0, 0.1, 0], dtype=float)
    # # x_dot0 = 0.1
    # # omega = -1
    # x_dot0 = 1
    # omega = 0
    # u0 = np.array([x_dot0, 0, omega], dtype=float)

    mass = 1.0
    radius = 0.1
    gravity = 9.81

    ball = RotatingBouncingBall(mass, radius, gravity, e_N, e_F, mu, q0, u0)

    system = System()
    system.add(ball)

    system.assemble()

    # solver1, label1 = Moreau(system, t_final, dt), "Moreau"
    solver1, label1 = BackwardEuler(system, t_final, dt), "BackwardEuler"
    # solver1, label1 = BackwardEuler(system, t_final, dt, options=SolverOptions(reuse_lu_decomposition=False)), "BackwardEuler"
    # solver1, label1 = (
    #     NonsmoothGeneralizedAlpha(system, t_final, dt),
    #     "NonsmoothGeneralizedAlpha",
    # )

    sol1 = solver1.solve()
    t1 = sol1.t
    q1 = sol1.q
    u1 = sol1.u
    # la_N1 = sol1.la_N
    # la_F1 = sol1.la_F
    # La_N1 = sol1.La_N
    # La_F1 = sol1.La_F
    P_N1 = sol1.P_N
    P_F1 = sol1.P_F

    # solver2, label2 = (
    #     NonsmoothGeneralizedAlpha(system, t_final, dt),
    #     "Gen-alpha",
    # )
    solver2, label2 = Moreau(system, t_final, dt), "Moreau"
    # solver2, label2 = BackwardEuler(system, t_final, dt), "BackwardEuler"
    # solver2, label2 = Rattle(system, t_final, dt), "Rattle"
    sol2 = solver2.solve()
    # sol2 = sol1
    # label2 = label1
    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u
    # la_N2 = sol2.la_N
    # la_F2 = sol2.la_F
    # La_N2 = sol2.La_N
    # La_F2 = sol2.La_F
    P_N2 = sol2.P_N
    P_F2 = sol2.P_F

    if export:
        path = Path(__file__)

        ###############
        # gap functions
        ###############
        g_N1 = np.array([system.g_N(ti, qi) for ti, qi in zip(sol1.t, sol1.q)])
        np.savetxt(
            path.parent / "g_N1.dat",
            np.hstack((sol1.t[:, None], g_N1)),
            delimiter=", ",
            header="t, g_N",
            comments="",
        )

        g_N2 = np.array([system.g_N(ti, qi) for ti, qi in zip(sol2.t, sol2.q)])
        np.savetxt(
            path.parent / "g_N2.dat",
            np.hstack((sol2.t[:, None], g_N2)),
            delimiter=", ",
            header="t, g_N",
            comments="",
        )

        ################
        # contact forces
        ################
        np.savetxt(
            path.parent / "P_N1.dat",
            np.hstack((sol1.t[:, None], P_N1)),
            delimiter=", ",
            header="t, P_N",
            comments="",
        )
        np.savetxt(
            path.parent / "P_N2.dat",
            np.hstack((sol2.t[:, None], P_N2)),
            delimiter=", ",
            header="t, P_N",
            comments="",
        )
        np.savetxt(
            path.parent / "int_P_N1.dat",
            np.hstack((sol1.t[:, None], np.cumsum(P_N1, axis=0))),
            delimiter=", ",
            header="t, P_N",
            comments="",
        )
        np.savetxt(
            path.parent / "int_P_N2.dat",
            np.hstack((sol2.t[:, None], np.cumsum(P_N2, axis=0))),
            delimiter=", ",
            header="t, P_N",
            comments="",
        )

        #################
        # friction forces
        #################
        if mu > 0:
            np.savetxt(
                path.parent / "P_F1.dat",
                np.hstack((sol1.t[:, None], P_F1)),
                delimiter=", ",
                header="t, P_F1, P_F2",
                comments="",
            )
            np.savetxt(
                path.parent / "P_F2.dat",
                np.hstack((sol2.t[:, None], P_F2)),
                delimiter=", ",
                header="t, P_F1, P_F2",
                comments="",
            )
            np.savetxt(
                path.parent / "int_P_F1.dat",
                np.hstack((sol1.t[:, None], np.cumsum(P_F1, axis=0))),
                delimiter=", ",
                header="t, P_F1, P_F2",
                comments="",
            )
            np.savetxt(
                path.parent / "int_P_F2.dat",
                np.hstack((sol2.t[:, None], np.cumsum(P_F2, axis=0))),
                delimiter=", ",
                header="t, P_F1, P_F2",
                comments="",
            )

    fig, ax = plt.subplots(2, 3)

    ax[0, 0].set_title("x(t)")
    ax[0, 0].plot(t2, q2[:, 0], "-k", label=label2)
    ax[0, 0].plot(t1, q1[:, 0], "--r", label=label1)
    ax[0, 0].legend()

    ax[1, 0].set_title("u_x(t)")
    ax[1, 0].plot(t2, u2[:, 0], "-k", label=label2)
    ax[1, 0].plot(t1, u1[:, 0], "--r", label=label1)
    ax[1, 0].legend()

    ax[0, 1].set_title("y(t)")
    ax[0, 1].plot([t2[0], t2[-1]], [radius, radius], "-b", label="ground")
    ax[0, 1].plot(t2, q2[:, 1], "-k", label=label2)
    ax[0, 1].plot(t1, q1[:, 1], "--r", label=label1)
    ax[0, 1].legend()

    ax[1, 1].set_title("u_y(t)")
    ax[1, 1].plot(t2, u2[:, 1], "-k", label=label2)
    ax[1, 1].plot(t1, u1[:, 1], "--r", label=label1)
    ax[1, 1].legend()

    ax[0, 2].set_title("phi(t)")
    ax[0, 2].plot(t2, q2[:, -1], "-k", label=label2)
    ax[0, 2].plot(t1, q1[:, -1], "--r", label=label1)
    ax[0, 2].legend()

    ax[1, 2].set_title("u_phi(t)")
    ax[1, 2].plot(t2, u2[:, -1], "-k", label=label2)
    ax[1, 2].plot(t1, u1[:, -1], "--r", label=label1)
    ax[1, 2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(1, 2)

    ax[0].set_title("P_N(t)")
    ax[0].plot(t2, P_N2[:, 0], "-k", label=label2)
    ax[0].plot(t1, P_N1[:, 0], "--r", label=label1)
    ax[0].legend()

    if mu > 0:
        ax[1].set_title("P_F(t)")
        ax[1].plot(t2, P_F2[:, 0], "-k", label=label2)
        ax[1].plot(t1, P_F1[:, 0], "--r", label=label1)
        ax[1].legend()

    plt.tight_layout()

    ###########
    # animation
    ###########
    t = t1
    q = q1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    width = 1.5
    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)

    # prepare data for animation
    frames = len(t)
    target_frames = min(len(t), 200)
    frac = int(frames / target_frames)
    animation_time = 5
    interval = animation_time * 1000 / target_frames

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    # horizontal plane
    ax.plot([-2 * width, 2 * width], [0, 0], "-k")

    def create(t, q):
        (COM,) = ax.plot([], [], "ok")
        (bdry,) = ax.plot([], [], "-k")
        (d1_,) = ax.plot([], [], "-r")
        (d2_,) = ax.plot([], [], "-g")
        return COM, bdry, d1_, d2_

    COM, bdry, d1_, d2_ = create(0, q[0])

    def update(t, q, COM, bdry, d1_, d2_):
        x_S, y_S, phi = q
        d1 = np.array([np.cos(phi), np.sin(phi), 0]) * radius
        d2 = np.array([-np.sin(phi), np.cos(phi), 0]) * radius

        x_bdry, y_bdry = ball.boundary(t, q)

        COM.set_data([x_S], [y_S])
        bdry.set_data(x_bdry, y_bdry)

        d1_.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
        d2_.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])

        return COM, bdry, d1_, d2_

    def animate(i):
        update(t[i], q[i], COM, bdry, d1_, d2_)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()


if __name__ == "__main__":
    # run(1)
    run(2)
    # run(3)
    # run(4)
