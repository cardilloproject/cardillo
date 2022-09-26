import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.model import Model
from cardillo.model.rigid_body import RigidBodyEuler
from cardillo.model.frame import Frame
from cardillo.forces import Force
from cardillo.contacts import SphereInSphere
from cardillo.solver import (
    Moreau,
    NonsmoothEulerBackwardsGGL,
    NonsmoothEulerBackwardsGGL_V2,
    NonsmoothEulerBackwardsGGL_V3,
    NonsmoothTheta,
    NonsmoothGeneralizedAlpha,
    NonsmoothGenAlphaFirstOrder,
    NonsmoothNewmark,
)


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
    # q0 = np.array([r_OS0[0], r_OS0[1], phi0])
    # u0 = np.array([vS0[0], vS0[1], phi_dot0])
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

    model = Model()
    model.add(RB)
    model.add(Force(lambda t: np.array([0, -g * m, 0]), RB))
    model.add(sphere)
    model.assemble()

    t0 = 0
    # t1 = 0.6
    t1 = 2
    # t1 = 5
    # dt = 1e-1
    # dt = 5e-2
    # dt = 1e-2
    # dt = 5e-3
    dt = 1e-3

    # solver_n = NonsmoothGeneralizedAlpha(model, t1, dt, rho_inf=0.85)
    # solver_n = NonsmoothEulerBackwardsGGL(model, t1, dt)
    solver_n = NonsmoothEulerBackwardsGGL_V2(model, t1, dt)
    # solver_n = NonsmoothEulerBackwardsGGL_V3(model, t1, dt)
    # solver_n = NonsmoothTheta(model, t1, dt)
    # solver_n = NonsmoothGenAlphaFirstOrder(model, t1, dt, rho_inf=0.85)
    # solver_n = NonsmoothNewmark(model, t1, dt)
    sol_n = solver_n.solve()
    # sol_n = sol_fp
    t_n = sol_n.t
    q_n = sol_n.q
    u_n = sol_n.u
    P_N_n = sol_n.P_N
    P_F_n = sol_n.P_F
    if type(solver_n) in [
        NonsmoothEulerBackwardsGGL,
        NonsmoothEulerBackwardsGGL_V2,
        NonsmoothEulerBackwardsGGL_V3,
    ]:
        a_n = np.zeros_like(u_n)
        a_n[1:] = (u_n[1:] - u_n[:-1]) / dt
        la_N_n = np.zeros_like(P_N_n)
        la_F_n = np.zeros_like(P_F_n)
        La_N_n = np.zeros_like(P_N_n)
        La_F_n = np.zeros_like(P_F_n)
    else:
        a_n = sol_n.a
        la_N_n = sol_n.la_N
        la_F_n = sol_n.la_F
        La_N_n = sol_n.La_N
        La_F_n = sol_n.La_F

    solver_fp = Moreau(model, t1, dt)
    sol_fp = solver_fp.solve()
    t_fp = sol_fp.t
    q_fp = sol_fp.q
    u_fp = sol_fp.u
    a_fp = np.zeros_like(u_fp)
    a_fp[1:] = (u_fp[1:] - u_fp[:-1]) / dt
    P_N_fp = sol_fp.P_N
    P_F_fp = sol_fp.P_F

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title("x(t)")
    ax[0].plot(t_fp, q_fp[:, 0], "-r", label="fixed_point")
    ax[0].plot(t_n, q_n[:, 0], "--b", label="newton")
    ax[0].legend()

    ax[1].set_title("u_x(t)")
    ax[1].plot(t_fp, u_fp[:, 0], "-r", label="fixed_point")
    ax[1].plot(t_n, u_n[:, 0], "--b", label="newton")
    ax[1].legend()

    ax[2].set_title("a_x(t)")
    ax[2].plot(t_fp, a_fp[:, 0], "-r", label="fixed_point")
    ax[2].plot(t_n, a_n[:, 0], "--b", label="newton")
    ax[2].legend()

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title("y(t)")
    ax[0].plot(t_fp, q_fp[:, 1], "-r", label="fixed_point")
    ax[0].plot(t_n, q_n[:, 1], "--b", label="newton")
    ax[0].legend()

    ax[1].set_title("u_y(t)")
    ax[1].plot(t_fp, u_fp[:, 1], "-r", label="fixed_point")
    ax[1].plot(t_n, u_n[:, 1], "--b", label="newton")
    ax[1].legend()

    ax[2].set_title("a_y(t)")
    ax[2].plot(t_fp, a_fp[:, 1], "-r", label="fixed_point")
    ax[2].plot(t_n, a_n[:, 1], "--b", label="newton")
    ax[2].legend()

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title("phi(t)")
    ax[0].plot(t_fp, q_fp[:, 3], "-r", label="fixed_point")
    ax[0].plot(t_n, q_n[:, 3], "--b", label="newton")
    ax[0].legend()

    ax[1].set_title("u_phi(t)")
    ax[1].plot(t_fp, u_fp[:, -1], "-r", label="fixed_point")
    ax[1].plot(t_n, u_n[:, -1], "--b", label="newton")
    ax[1].legend()

    ax[2].set_title("a_phi(t)")
    ax[2].plot(t_fp, a_fp[:, -1], "-r", label="fixed_point")
    ax[2].plot(t_n, a_n[:, -1], "--b", label="newton")
    ax[2].legend()

    fig, ax = plt.subplots(3, 1)

    ax[0].set_title("P_N(t)")
    ax[0].plot(t_fp, P_N_fp[:, 0], "-r", label="fixed_point")
    # ax[0].plot(t_n, sol_n.la_N[:, 0]*dt, '--b', label='newton_la_N')
    # ax[0].plot(t_n, sol_n.La_N[:, 0], '--g', label='newton_La_N')
    ax[0].plot(t_n, P_N_n[:, 0], "--g", label="newton_P_N")
    ax[0].legend()

    ax[1].set_title("P_Fx(t)")
    ax[1].plot(t_fp, P_F_fp[:, 0], "-r", label="fixed_point")
    # ax[1].plot(t_n, sol_n.la_T[:, 0]*dt, '--b', label='newton_la_T')
    # ax[1].plot(t_n, sol_n.La_T[:, 0], '--g', label='newton_La_T')
    ax[1].plot(t_n, P_F_n[:, 0], "--g", label="newton_P_N")
    ax[1].legend()

    ax[2].set_title("P_Fy(t)")
    ax[2].plot(t_fp, P_F_fp[:, 1], "-r", label="fixed_point")
    # ax[2].plot(t_n, sol_n.la_T[:, 1]*dt, '--b', label='newton_la_T')
    # ax[2].plot(t_n, sol_n.La_T[:, 1], '--g', label='newton_La_T')
    ax[2].plot(t_n, P_F_n[:, 1], "--g", label="newton_P_N")
    ax[2].legend()

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
        frames = len(t_n)
        target_frames = min(len(t_n), 200)
        frac = int(frames / target_frames)
        animation_time = 5
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t_n[::frac]
        q_n = q_n[::frac]
        q_fp = q_fp[::frac]

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
            0, q_n[0], q_fp[0]
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
                q_n[i],
                q_fp[i],
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
