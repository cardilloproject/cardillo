import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo import System
from cardillo.discrete import RigidBodyEuler
from cardillo.discrete import Frame
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane
from cardillo.solver import (
    Moreau,
    NonsmoothGeneralizedAlpha,
    Rattle,
    Moreau_new,
    Moreau_classical,
)


class Ball(RigidBodyEuler):
    def __init__(self, m, R, q0=None, u0=None):
        theta = 2 / 5 * m * R**2
        self.R = R
        super().__init__(m, theta * np.eye(3), q0=q0, u0=u0)

    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.R * np.vstack([np.sin(phi), np.cos(phi), np.zeros(n)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP


def run(case):
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
    # dt = 5e-3
    dt = 1e-2

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
        t_final = 1.5
    else:
        raise AssertionError("Case not found!")

    q0 = np.array([x0, y0, 0, 0, 0, 0], dtype=float)
    u0 = np.array([x_dot0, y_dot0, 0, 0, 0, omega], dtype=float)

    m = 1.0
    R = 0.1
    g = 9.81
    RB = Ball(m, R, q0, u0)

    e1, e2, e3 = np.eye(3, dtype=float)
    frame = Frame(A_IK=np.vstack((e3, e1, e2)).T, r_OP=np.zeros(3, dtype=float))
    plane = Sphere2Plane(frame, RB, R, mu, e_N=e_N, e_F=e_F)

    system = System()
    system.add(RB)
    system.add(Force(np.array([0, -g * m, 0]), RB))
    system.add(plane)

    system.assemble()

    # solver1, label1 = NonsmoothGeneralizedAlpha(system, t_final, dt, method="newton"), "Gen-alpha"
    solver1, label1 = Rattle(system, t_final, dt), "Rattle"
    # solver1, label1 = Moreau(system, t_final, dt), "Moreau"
    # solver1, label1 = Moreau_new(system, t_final, dt), "Moreau_new"
    # solver1, label1 = Moreau_classical(system, t_final, dt), "Moreau_classical"

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
    # solver2, label2 = Rattle(system, t_final, dt), "Rattle"
    sol2 = solver2.solve()
    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u
    # la_N2 = sol2.la_N
    # la_F2 = sol2.la_F
    # La_N2 = sol2.La_N
    # La_F2 = sol2.La_F
    P_N2 = sol2.P_N
    P_F2 = sol2.P_F

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
    ax[0, 1].plot([t2[0], t2[-1]], [R, R], "-b", label="ground")
    ax[0, 1].plot(t2, q2[:, 1], "-k", label=label2)
    ax[0, 1].plot(t1, q1[:, 1], "--r", label=label1)
    ax[0, 1].legend()

    ax[1, 1].set_title("u_y(t)")
    ax[1, 1].plot(t2, u2[:, 1], "-k", label=label2)
    ax[1, 1].plot(t1, u1[:, 1], "--r", label=label1)
    ax[1, 1].legend()

    ax[0, 2].set_title("phi(t)")
    ax[0, 2].plot(t2, q2[:, 3], "-k", label=label2)
    ax[0, 2].plot(t1, q1[:, 3], "--r", label=label1)
    ax[0, 2].legend()

    ax[1, 2].set_title("u_phi(t)")
    ax[1, 2].plot(t2, u2[:, -1], "-k", label=label2)
    ax[1, 2].plot(t1, u1[:, -1], "--r", label=label1)
    ax[1, 2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(1, 3)

    ax[0].set_title("P_N(t)")
    ax[0].plot(t2, P_N2[:, 0], "-k", label=label2)
    ax[0].plot(t1, P_N1[:, 0], "--r", label=label1)
    ax[0].legend()

    if mu > 0:
        ax[1].set_title("P_Fx(t)")
        ax[1].plot(t2, P_F2[:, 0], "-k", label=label2)
        ax[1].plot(t1, P_F1[:, 0], "--r", label=label1)
        ax[1].legend()

        ax[2].set_title("P_Fy(t)")
        ax[2].plot(t2, P_F2[:, 1], "-k", label=label2)
        ax[2].plot(t1, P_F1[:, 1], "--r", label=label1)
        ax[2].legend()

    plt.tight_layout()

    t = t1
    q = q1

    # animate configurations
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.set_xlim(-2 * y0, 2 * y0)
    ax.set_ylim(-2 * y0, 2 * y0)

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
    ax.plot([-2 * y0, 2 * y0], [0, 0], "-k")

    def create(t, q):
        x_S, y_S, _ = RB.r_OP(t, q)

        A_IK = RB.A_IK(t, q)
        d1 = A_IK[:, 0] * R
        d2 = A_IK[:, 1] * R
        # d3 = A_IK[:, 2] * r

        (COM,) = ax.plot([x_S], [y_S], "ok")
        (bdry,) = ax.plot([], [], "-k")
        (d1_,) = ax.plot([x_S, x_S + d1[0]], [y_S, y_S + d1[1]], "-r")
        (d2_,) = ax.plot([x_S, x_S + d2[0]], [y_S, y_S + d2[1]], "-g")
        return COM, bdry, d1_, d2_

    COM, bdry, d1_, d2_ = create(0, q[0])

    def update(t, q, COM, bdry, d1_, d2_):
        x_S, y_S, _ = RB.r_OP(t, q)

        x_bdry, y_bdry, _ = RB.boundary(t, q)

        A_IK = RB.A_IK(t, q)
        d1 = A_IK[:, 0] * R
        d2 = A_IK[:, 1] * R
        # d3 = A_IK[:, 2] * r

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
    # run(2)
    # run(3)
    run(4)
