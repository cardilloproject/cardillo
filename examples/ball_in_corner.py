import numpy as np
from math import pi

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math import A_IK_basic

from cardillo import System
from cardillo.discrete import RigidBodyEuler
from cardillo.discrete import Frame
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane
from cardillo.solver import (
    MoreauShifted,
    Rattle,
    NonsmoothBackwardEuler,
    NonsmoothPIRK,
    RadauIIATableau,
)


Solver1, label1, dt1, kwargs1 = (
    NonsmoothPIRK,
    "NPIRK",
    1e-2,
    {"butcher_tableau": RadauIIATableau(2)},
)
# Solver1, label1, dt1, kwargs1 = NonsmoothBackwardEuler, "Euler backward", 1e-2, {}
# Solver1, label1, dt1, kwargs1 = Rattle, "Rattle", 1e-2, {}
# Solver1, label1, dt1, kwargs1 = MoreauShifted, "MoreauShifted", 2e-2, {}
Solver2, label2, dt2, kwargs2 = MoreauShifted, "MoreauShifted", 1e-2, {}


class Ball(RigidBodyEuler):
    def __init__(self, m, r, q0=None, u0=None):
        theta = 2 / 5 * m * r**2
        self.r = r
        super().__init__(m, theta * np.eye(3), q0=q0, u0=u0)

    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.r * np.vstack([np.sin(phi), np.cos(phi), np.zeros(n)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP


def run():
    """Example 10.2 of Capobianco2021.

    References:
    -----------
    Capobianco2021: https://doi.org/10.1002/nme.6801
    """
    m = 1.25
    r = 0.1
    g = 10

    x0 = -0.5
    y0 = 1
    phi0 = 0
    x_dot0 = 0
    y_dot0 = 0
    phi_dot0 = 0

    r_OS0 = np.array([x0, y0, 0], dtype=float)
    vS0 = np.array([x_dot0, y_dot0, 0], dtype=float)
    q0 = np.concatenate([r_OS0, np.array([phi0, 0, 0], dtype=float)])
    u0 = np.concatenate([vS0, np.array([0, 0, phi_dot0], dtype=float)])
    RB = Ball(m, r, q0, u0)

    alpha = -pi / 4 * 1.1
    e1, e2, e3 = A_IK_basic(alpha).z().T
    frame_left = Frame(A_IK=np.vstack((e3, e1, e2)).T)
    mu = 0.3
    e_N = 0.0
    e_F = 0.0
    plane_left = Sphere2Plane(frame_left, RB, r, mu, e_N=e_N, e_F=e_F)

    beta = pi / 4
    e1, e2, e3 = A_IK_basic(beta).z().T
    frame_right = Frame(A_IK=np.vstack((e3, e1, e2)).T)
    mu = 0.3
    e_N = 0.5
    # e_N = 0.0  # TODO: Remove this
    e_F = 0.0
    plane_right = Sphere2Plane(frame_right, RB, r, mu, e_N=e_N, e_F=e_F)
    # plane_right2 = Sphere2Plane(frame_right, RB, r, mu, e_N=e_N, e_F=e_F)

    system = System()
    system.add(RB)
    system.add(Force(lambda t: np.array([0, -g * m, 0]), RB))
    system.add(plane_right)
    system.add(plane_left)
    # system.add(plane_right2)

    system.assemble()

    t_final = 2

    sol1 = Solver1(system, t_final, dt1, **kwargs1).solve()
    t = sol1.t
    q = sol1.q
    t1 = sol1.t
    q1 = sol1.q
    u1 = sol1.u
    P_N1 = sol1.P_N
    P_F1 = sol1.P_F

    sol2 = Solver2(system, t_final, dt2, **kwargs2).solve()
    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u
    P_N2 = sol2.P_N
    P_F2 = sol2.P_F

    fig, ax = plt.subplots(2, 3)

    ax[0, 0].set_title("x(t)")
    ax[0, 0].plot(t1, q1[:, 0], "--k", label=label1)
    ax[0, 0].plot(t2, q2[:, 0], "--r", label=label2)
    ax[0, 0].legend()

    ax[1, 0].set_title("u_x(t)")
    ax[1, 0].plot(t1, u1[:, 0], "--k", label=label1)
    ax[1, 0].plot(t2, u2[:, 0], "--r", label=label2)
    ax[1, 0].legend()

    ax[0, 1].set_title("y(t)")
    ax[0, 1].plot(t1, q1[:, 1], "--k", label=label1)
    ax[0, 1].plot(t2, q2[:, 1], "--r", label=label2)
    ax[0, 1].legend()

    ax[1, 1].set_title("u_y(t)")
    ax[1, 1].plot(t1, u1[:, 1], "--k", label=label1)
    ax[1, 1].plot(t2, u2[:, 1], "--r", label=label2)
    ax[1, 1].legend()

    ax[0, 2].set_title("phi(t)")
    ax[0, 2].plot(t1, q1[:, 3], "--k", label=label1)
    ax[0, 2].plot(t2, q2[:, 3], "--r", label=label2)
    ax[0, 2].legend()

    ax[1, 2].set_title("u_phi(t)")
    ax[1, 2].plot(t1, u1[:, 3], "--k", label=label1)
    ax[1, 2].plot(t2, u2[:, 3], "--r", label=label2)
    ax[1, 2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(2, 3)

    ax[0, 0].set_title("P_N_left(t)")
    ax[0, 0].plot(t1, P_N1[:, 0], "-k", label=label1)
    ax[0, 0].plot(t2, P_N2[:, 0], "--r", label=label2)
    ax[0, 0].legend()

    ax[1, 0].set_title("P_N_right(t)")
    ax[1, 0].plot(t1, P_N1[:, 1], "-k", label=label1)
    ax[1, 0].plot(t2, P_N2[:, 1], "--r", label=label2)
    ax[1, 0].legend()

    if mu > 0:
        ax[0, 1].set_title("P_Fx_left(t)")
        ax[0, 1].plot(t1, P_F1[:, 0], "-k", label=label1)
        ax[0, 1].plot(t2, P_F2[:, 0], "--r", label=label2)
        ax[0, 1].legend()

        ax[0, 2].set_title("P_Fy_left(t)")
        ax[0, 2].plot(t1, P_F1[:, 1], "-k", label=label1)
        ax[0, 2].plot(t2, P_F2[:, 1], "--r", label=label2)
        ax[0, 2].legend()

        ax[1, 1].set_title("P_Fx_right(t)")
        ax[1, 1].plot(t1, P_F1[:, 2], "-k", label=label1)
        ax[1, 1].plot(t2, P_F2[:, 2], "--r", label=label2)
        ax[1, 1].legend()

        ax[1, 2].set_title("P_Fy_right(t)")
        ax[1, 2].plot(t1, P_F1[:, 3], "-k", label=label1)
        ax[1, 2].plot(t2, P_F2[:, 3], "--r", label=label2)
        ax[1, 2].legend()

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

    # inclined planes
    K_r_OPs = np.array([[-y0, 0, 0], [y0, 0, 0]]).T
    r_OPs_left = A_IK_basic(alpha).z() @ K_r_OPs
    r_OPs_right = A_IK_basic(beta).z() @ K_r_OPs
    ax.plot(*r_OPs_left[:2], "-k")
    ax.plot(*r_OPs_right[:2], "--k")

    def create(t, q):
        x_S, y_S, _ = RB.r_OP(t, q)

        A_IK = RB.A_IK(t, q)
        d1 = A_IK[:, 0] * r
        d2 = A_IK[:, 1] * r

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
        d1 = A_IK[:, 0] * r
        d2 = A_IK[:, 1] * r

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
    run()
