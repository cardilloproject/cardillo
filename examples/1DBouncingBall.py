import numpy as np
from math import pi

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.model import Model

from cardillo.solver import (
    Moreau,
    NonsmoothEulerBackwardsGGL,
    NonsmoothThetaGGL,
    NonsmoothNewmarkFirstOrder,
    NonsmoothGeneralizedAlpha,
)


class BouncingBall:
    def __init__(
        self,
        m,
        g,
        radius,
        prox_r_N,
        e_N=None,
        q0=None,
        u0=None,
    ):
        self.m = m
        self.g = g
        self.nq = 1
        self.nu = 1
        self.nla_N = 1
        self.nla_F = 0
        self.NF_connectivity = [[]]

        self.radius = radius
        self.prox_r_N = np.array([prox_r_N])
        self.prox_r_F = np.zeros(self.nla_F)
        self.mu = np.zeros(self.nla_F)

        self.e_N = np.zeros(self.nla_N) if e_N is None else np.array([e_N])
        self.e_F = np.zeros(self.nla_F)

        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0
        self.la_N0 = np.zeros(self.nla_N)
        self.la_F0 = np.zeros(self.nla_F)

    def r_OS(self, q):
        return np.array([0.0, *q, 0.0], dtype=float)

    def boundary(self, q, n=100):
        r_OS = self.r_OS(q)
        phi = np.linspace(0, 2.0 * np.pi, n, endpoint=True)
        K_r_SP = self.radius * np.vstack([np.sin(phi), np.cos(phi), np.zeros(n)])
        return np.repeat(r_OS, n).reshape(3, n) + K_r_SP

    def M(self, t, q, coo):
        coo.extend(m * np.eye(1, dtype=float), (self.uDOF, self.uDOF))

    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q, coo):
        coo.extend_diag(np.ones(self.nq), (self.qDOF, self.uDOF))

    def f_pot(self, t, q):
        return np.array([-self.m * self.g])

    def f_pot_q(self, t, q, coo):
        coo.extend(np.zeros((1, 1)), (self.uDOF, self.qDOF))

    def g_N(self, t, q):
        return q - self.radius

    def g_N_dot(self, t, q, u):
        return u

    def g_N_ddot(self, t, q, u, u_dot):
        return u_dot

    def g_N_q(self, t, q, coo):
        coo.extend(np.eye(1, dtype=float), (self.la_NDOF, self.qDOF))

    def g_N_dot_q(self, t, q, u, coo):
        coo.extend(np.zeros((1, 1)), (self.la_NDOF, self.qDOF))

    def g_N_dot_u(self, t, q, coo):
        coo.extend(np.eye(1, dtype=float), (self.la_NDOF, self.uDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def W_N(self, t, q, coo):
        coo.extend(np.eye(1, dtype=float), (self.uDOF, self.la_NDOF))


if __name__ == "__main__":
    # system definition
    m = 1
    radius = 0.1
    g = 9.81
    e_N = 0.5
    prox_r_N = 0.5

    # initial conditions
    y0 = 1
    y_dot0 = 0
    r_OS0 = np.array([y0])
    vS0 = np.array([y_dot0])
    q0 = np.array([y0])
    u0 = np.array([y_dot0])

    # create bouncing ball subsystem
    ball = BouncingBall(m, g, radius, prox_r_N=prox_r_N, e_N=e_N, q0=q0, u0=u0)

    # assemble model
    model = Model()
    model.add(ball)
    model.assemble()

    # solver setup
    t0 = 0
    t1 = 2
    # dt = 1e-1
    # dt = 5e-2
    # dt = 1e-2
    dt = 5e-3
    # dt = 1e-3

    # solve problem
    sol_moreau = Moreau(model, t1, dt).solve()
    t_moreau = sol_moreau.t
    q_moreau = sol_moreau.q
    u_moreau = sol_moreau.u
    a_moreau = np.zeros_like(u_moreau)
    a_moreau[1:] = (u_moreau[1:] - u_moreau[:-1]) / dt
    P_N_moreau = sol_moreau.P_N
    P_F_moreau = sol_moreau.P_F

    # solver_other = NonsmoothGeneralizedAlpha(model, t1, dt)
    # solver_other = Generalized_alpha_3(model, t1, dt, numerical_jacobian=True)
    solver_other = NonsmoothNewmarkFirstOrder(model, t1, dt, atol=1.0e-8)
    # solver_other = NonsmoothThetaGGL(model, t1, dt)
    # solver_other = NonsmoothEulerBackwardsGGL(model, t1, dt)
    sol_other = solver_other.solve()
    t = sol_other.t
    q = sol_other.q
    t_other = sol_other.t
    q_other = sol_other.q
    u_other = sol_other.u
    P_N_other = sol_other.P_N
    P_F_other = sol_other.P_F
    if type(solver_other) in [NonsmoothThetaGGL, NonsmoothEulerBackwardsGGL]:
        a_other = np.zeros_like(u_other)
        a_other[1:] = (u_other[1:] - u_other[:-1]) / dt
        la_N_other = np.zeros_like(P_N_other)
        la_F_other = np.zeros_like(P_F_other)
        La_N_other = np.zeros_like(P_N_other)
        La_F_other = np.zeros_like(P_F_other)
    else:
        a_other = sol_other.a
        la_N_other = sol_other.la_N
        la_F_other = sol_other.la_F
        La_N_other = sol_other.La_N
        La_F_other = sol_other.La_F

    # visualization
    fig, ax = plt.subplots(3, 1)

    ax[0].plot([t_moreau[0], t_moreau[-1]], [radius, radius], "-k", label="ground")
    ax[0].plot(t_moreau, q_moreau[:, 0], "-r", label="q - Moreau")
    ax[0].plot(t_other, q_other[:, 0], "-b", label="q - Other")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t_moreau, u_moreau[:, 0], "-r", label="u - Moreau")
    ax[1].plot(t_other, u_other[:, 0], "-b", label="u - Other")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(t_moreau, a_moreau[:, 0], "-r", label="a - Moreau")
    ax[2].plot(t_other, a_other[:, 0], "-b", label="a - Other")
    ax[2].grid()
    ax[2].legend()

    # animation
    # t = t_moreau
    # q = q_moreau
    t = t_other
    q = q_other

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
        (COM,) = ax.plot([], [], "ok")
        (bdry,) = ax.plot([], [], "-k")
        return COM, bdry

    COM, bdry = create(0, q[0])

    def update(t, q, COM, bdry):
        x_S, y_S, _ = ball.r_OS(q)
        x_bdry, y_bdry, _ = ball.boundary(q)

        COM.set_data([x_S], [y_S])
        bdry.set_data(x_bdry, y_bdry)

        return COM, bdry

    def animate(i):
        update(t[i], q[i], COM, bdry)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()

    exit()

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title("x(t)")
    ax[0].plot(t_moreau, q_moreau[:, 0], "-r", label="Moreau")
    ax[0].plot(t_other, q_other[:, 0], "--b", label="Other")
    ax[0].legend()

    ax[1].set_title("u_x(t)")
    ax[1].plot(t_moreau, u_moreau[:, 0], "-r", label="Moreau")
    ax[1].plot(t_other, u_other[:, 0], "--b", label="Other")
    ax[1].legend()

    ax[2].set_title("a_x(t)")
    ax[2].plot(t_moreau, a_moreau[:, 0], "-r", label="Moreau")
    ax[2].plot(t_other, a_other[:, 0], "--b", label="Other")
    ax[2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title("y(t)")
    ax[0].plot([t_moreau[0], t_moreau[-1]], [radius, radius], "-k", label="ground")
    ax[0].plot(t_moreau, q_moreau[:, 1], "-r", label="y Moreau")
    ax[0].plot(t_other, q_other[:, 1], "--b", label="y Other")
    ax[0].legend()

    ax[1].set_title("u_y(t)")
    ax[1].plot(t_moreau, u_moreau[:, 1], "-r", label="Moreau")
    ax[1].plot(t_other, u_other[:, 1], "--b", label="Other")
    ax[1].legend()

    ax[2].set_title("a_y(t)")
    ax[2].plot(t_moreau, a_moreau[:, 1], "-r", label="Moreau")
    ax[2].plot(t_other, a_other[:, 1], "--b", label="Other")
    ax[2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title("phi(t)")
    ax[0].plot(t_moreau, q_moreau[:, 3], "-r", label="Moreau")
    ax[0].plot(t_other, q_other[:, 3], "--b", label="Other")
    ax[0].legend()

    ax[1].set_title("u_phi(t)")
    ax[1].plot(t_moreau, u_moreau[:, -1], "-r", label="Moreau")
    ax[1].plot(t_other, u_other[:, -1], "--b", label="Other")
    ax[1].legend()

    ax[2].set_title("a_phi(t)")
    ax[2].plot(t_moreau, a_moreau[:, -1], "-r", label="Moreau")
    ax[2].plot(t_other, a_other[:, -1], "--b", label="Other")
    ax[2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(3, 1)

    ax[0].set_title("P_N(t)")
    ax[0].plot(t_moreau, P_N_moreau[:, 0], "-r", label="Moreau")
    ax[0].plot(t_other, la_N_other[:, 0], "--b", label="Other_la_N")
    ax[0].plot(t_other, La_N_other[:, 0], "--g", label="Other_La_N")
    ax[0].plot(t_other, P_N_other[:, 0], "--k", label="Other_P_N")
    ax[0].legend()

    # ax[1].set_title("P_Fx(t)")
    # ax[1].plot(t_fp, P_F_fp[:, 0], "-r", label="Moreau")
    # ax[1].plot(t_g, la_F_g[:, 0], "--b", label="Other_la_F")
    # ax[1].plot(t_g, La_F_g[:, 0], "--g", label="Other_La_F")
    # ax[1].plot(t_g, P_F_g[:, 0], "--k", label="Other_P_N")
    # ax[1].legend()

    # ax[2].set_title("P_Fy(t)")
    # ax[2].plot(t_fp, P_F_fp[:, 1], "-r", label="Moreau")
    # ax[2].plot(t_g, la_F_g[:, 1], "--b", label="Other_la_F")
    # ax[2].plot(t_g, La_F_g[:, 1], "--g", label="Other_La_F")
    # ax[2].plot(t_g, P_F_g[:, 1], "--k", label="Other_P_N")
    # ax[2].legend()

    plt.tight_layout()

    plt.show()

    if animate:

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

        # ax.plot([-2 * y0, 2 * y0], (y0-0.1)*np.array([1, 1]), '-k')

        # horizontal plane
        ax.plot([-2 * y0, 2 * y0], [0, 0], "-k")

        # # inclined planes
        # ax.plot([0, -y0 * np.cos(alpha)], [0, y0 * np.sin(alpha)], '-k')
        # ax.plot([0, y0 * np.cos(beta)], [0, - y0 * np.sin(beta)], '-k')

        def create(t, q):
            x_S, y_S, _ = ball.r_OP(t, q)

            A_IK = ball.A_IK(t, q)
            d1 = A_IK[:, 0] * radius
            d2 = A_IK[:, 1] * radius
            # d3 = A_IK[:, 2] * r

            (COM,) = ax.plot([x_S], [y_S], "ok")
            (bdry,) = ax.plot([], [], "-k")
            (d1_,) = ax.plot([x_S, x_S + d1[0]], [y_S, y_S + d1[1]], "-r")
            (d2_,) = ax.plot([x_S, x_S + d2[0]], [y_S, y_S + d2[1]], "-g")
            return COM, bdry, d1_, d2_

        COM, bdry, d1_, d2_ = create(0, q[0])

        def update(t, q, COM, bdry, d1_, d2_):

            x_S, y_S, _ = ball.r_OP(t, q)

            x_bdry, y_bdry, _ = ball.boundary(t, q)

            A_IK = ball.A_IK(t, q)
            d1 = A_IK[:, 0] * radius
            d2 = A_IK[:, 1] * radius
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
