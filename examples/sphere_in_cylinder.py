import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo import System
from cardillo.discrete import RigidBodyEuler
from cardillo.discrete import Frame
from cardillo.forces import Force
from cardillo.contacts import SphereInSphere
from cardillo.solver import Moreau, NonsmoothBackwardEulerDecoupled, Rattle


class Ball(RigidBodyEuler):
    def __init__(self, m, r, q0=None, u0=None):
        theta = 2 / 5 * m * r**2
        self.r = r
        super().__init__(m, theta * np.eye(3), q0=q0, u0=u0)

    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.r * np.vstack([np.sin(phi), np.cos(phi), np.zeros(n)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP


if __name__ == "__main__":
    animate = True

    m = 1
    r = 0.1
    R = 1
    g = 9.81
    x0 = -0.5
    y0 = 0
    x_dot0 = 0
    y_dot0 = 0
    phi0 = 0
    phi_dot0 = 0
    r_OS0 = np.array([x0, y0, 0])
    vS0 = np.array([x_dot0, y_dot0, 0])
    q0 = np.concatenate([r_OS0, np.array([phi0, 0, 0])])
    u0 = np.concatenate([vS0, np.array([0, 0, phi_dot0])])
    RB = Ball(m, r, q0, u0)

    e1, e2, e3 = np.eye(3)
    frame = Frame(A_IK=np.vstack((e3, e1, e2)).T, r_OP=np.array([0, 0, 0]))
    mu = 0.1
    r_N = 0.3
    e_N = 0
    sphere = SphereInSphere(
        frame, R, RB, r, mu, prox_r_N=r_N, prox_r_F=r_N, e_N=e_N, e_F=0
    )

    system = System()
    system.add(RB)
    system.add(Force(lambda t: np.array([0, -g * m, 0]), RB))
    system.add(sphere)
    system.assemble()

    t0 = 0
    t_final = 2
    dt = 1e-2

    sol1, label1 = Rattle(system, t_final, dt, atol=1e-8).solve(), "Rattle"
    sol2, label2 = Moreau(system, t_final, dt, fix_point_tol=1e-6).solve(), "Moreau"

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

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_title("x(t)")
    ax[0, 0].plot(t1, q1[:, 0], "-k", label=label1)
    ax[0, 0].plot(t2, q2[:, 0], "--r", label=label2)
    ax[0, 0].legend()

    ax[1, 0].set_title("u_x(t)")
    ax[1, 0].plot(t1, u1[:, 0], "-k", label=label1)
    ax[1, 0].plot(t2, u2[:, 0], "--r", label=label2)
    ax[1, 0].legend()

    ax[0, 1].set_title("y(t)")
    ax[0, 1].plot(t1, q1[:, 1], "-k", label=label1)
    ax[0, 1].plot(t2, q2[:, 1], "--r", label=label2)
    ax[0, 1].legend()

    ax[1, 1].set_title("u_y(t)")
    ax[1, 1].plot(t1, u1[:, 1], "-k", label=label1)
    ax[1, 1].plot(t2, u2[:, 1], "--r", label=label2)
    ax[1, 1].legend()

    ax[0, 2].set_title("phi(t)")
    ax[0, 2].plot(t1, q1[:, 3], "-k", label=label1)
    ax[0, 2].plot(t2, q2[:, 3], "--r", label=label2)
    ax[0, 2].legend()

    ax[1, 2].set_title("u_phi(t)")
    ax[1, 2].plot(t1, u1[:, 5], "-k", label=label1)
    ax[1, 2].plot(t2, u2[:, 5], "--r", label=label2)
    ax[1, 2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(3, 1)

    ax[0].set_title("P_N(t)")
    ax[0].plot(t1, P_N1[:, 0], "--r", label=label1)
    ax[0].plot(t2, P_N2[:, 0], "-k", label=label2)
    ax[0].legend()

    ax[1].set_title("P_Fx(t)")
    ax[1].plot(t1, P_F1[:, 0], "--r", label=label1)
    ax[1].plot(t2, P_F2[:, 0], "-k", label=label2)
    ax[1].legend()

    ax[2].set_title("P_Fy(t)")
    ax[2].plot(t1, P_F1[:, 1], "--r", label=label1)
    ax[2].plot(t2, P_F2[:, 1], "-k", label=label2)
    ax[2].legend()

    plt.tight_layout()

    # gaps
    nt1 = len(t1)
    g_N1 = np.zeros(nt1)
    g_N_dot1 = np.zeros(nt1)
    nt2 = len(t2)
    g_N2 = np.zeros(nt2)
    g_N_dot2 = np.zeros(nt2)

    for i, ti in enumerate(t1):
        g_N1[i] = system.g_N(ti, q1[i])
        g_N_dot1[i] = system.g_N_dot(ti, q1[i], u1[i])
    for i, ti in enumerate(t2):
        g_N2[i] = system.g_N(ti, q2[i])
        g_N_dot2[i] = system.g_N_dot(ti, q2[i], u2[i])

    fig, ax = plt.subplots(2, 1)

    ax[0].set_title("g_N(t)")
    ax[0].plot(t1, g_N1, "--r", label=label1)
    ax[0].plot(t2, g_N2, "-k", label=label2)
    ax[0].legend()

    ax[1].set_title("g_N_dot_Fx(t)")
    ax[1].plot(t1, g_N_dot1, "--r", label=label1)
    ax[1].plot(t2, g_N_dot2, "-k", label=label2)
    ax[1].legend()

    plt.tight_layout()

    if not animate:
        plt.show()

    if animate:
        # animate configurations
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.set_xlim(-2 * R, 2 * R)
        ax.set_ylim(-2 * R, 2 * R)

        # prepare data for animation
        frames = len(t1)
        target_frames = min(len(t1), 200)
        frac = int(frames / target_frames)
        animation_time = 5
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t1[::frac]
        q1 = q1[::frac]
        q2 = q2[::frac]

        circ_phi = np.arange(0, 2 * np.pi, step=0.01)
        ax.plot(R * np.sin(circ_phi), R * np.cos(circ_phi), "-k")

        def create(t, q_n, q_fp):
            x_S, y_S, _ = RB.r_OP(t, q_n)

            A_IK = RB.A_IK(t, q_n)
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
            # d3 = A_IK[:, 2] * r

            (COM_n,) = ax.plot([x_S], [y_S], "ok")
            (bdry_n,) = ax.plot([], [], "-k")
            (d1_n,) = ax.plot([x_S, x_S + d1[0]], [y_S, y_S + d1[1]], "-r")
            (d2_n,) = ax.plot([x_S, x_S + d2[0]], [y_S, y_S + d2[1]], "-g")

            x_S, y_S, _ = RB.r_OP(t, q_fp)

            A_IK = RB.A_IK(t, q_fp)
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
            # d3 = A_IK[:, 2] * r

            (COM_fp,) = ax.plot([x_S], [y_S], "ob")
            (bdry_fp,) = ax.plot([], [], "-b")
            (d1_fp,) = ax.plot([x_S, x_S + d1[0]], [y_S, y_S + d1[1]], "-r")
            (d2_fp,) = ax.plot([x_S, x_S + d2[0]], [y_S, y_S + d2[1]], "-g")
            return COM_n, bdry_n, d1_n, d2_n, COM_fp, bdry_fp, d1_fp, d2_fp

        COM_n, bdry_n, d1_n, d2_n, COM_fp, bdry_fp, d1_fp, d2_fp = create(
            0, q1[0], q2[0]
        )

        def update(
            t, q_n, q_fp, COM_n, bdry_n, d1_n, d2_n, COM_fp, bdry_fp, d1_fp, d2_fp
        ):
            x_S, y_S, _ = RB.r_OP(t, q_n)

            x_bdry, y_bdry, _ = RB.boundary(t, q_n)

            A_IK = RB.A_IK(t, q_n)
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
            # d3 = A_IK[:, 2] * r

            COM_n.set_data([x_S], [y_S])
            bdry_n.set_data(x_bdry, y_bdry)

            d1_n.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
            d2_n.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])

            x_S, y_S, _ = RB.r_OP(t, q_fp)

            x_bdry, y_bdry, _ = RB.boundary(t, q_fp)

            A_IK = RB.A_IK(t, q_fp)
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
            # d3 = A_IK[:, 2] * r

            COM_fp.set_data([x_S], [y_S])
            bdry_fp.set_data(x_bdry, y_bdry)

            d1_fp.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
            d2_fp.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])

            return COM_n, bdry_n, d1_n, d2_n, COM_fp, bdry_fp, d1_fp, d2_fp

        def animate(i):
            update(
                t[i],
                q1[i],
                q2[i],
                COM_n,
                bdry_n,
                d1_n,
                d2_n,
                COM_fp,
                bdry_fp,
                d1_fp,
                d2_fp,
            )

        anim = animation.FuncAnimation(
            fig, animate, frames=frames, interval=interval, blit=False
        )
        plt.show()
