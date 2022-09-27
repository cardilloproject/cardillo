import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.model import Model
from cardillo.model.rigid_body import RigidBodyEuler
from cardillo.model.frame import Frame
from cardillo.contacts import Sphere2Plane, Sphere2Sphere

from cardillo.solver import (
    Moreau,
    NonsmoothEulerBackwardsGGL_V2,
    NonsmoothEulerBackwardsGGL_V3,
    NonsmoothGeneralizedAlpha,
    Remco,
    NonsmoothHalfExplicitEuler,
)


class Ball(RigidBodyEuler):
    def __init__(self, m, radius, g, q0=None, u0=None):
        theta = 2 / 5 * m * radius**2
        self.radius = radius
        self.g = g
        super().__init__(m, theta * np.eye(3), q0=q0, u0=u0)

    def f_pot(self, t, q):
        return np.array([0, -self.m * self.g, 0, 0, 0, 0])

    def f_pot_q(self, t, q, coo):
        coo.extend(np.zeros((self.nu, 1)), (self.uDOF, self.qDOF))

    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.radius * np.vstack([np.sin(phi), np.cos(phi), np.zeros(n)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP


if __name__ == "__main__":
    g = 10.0 * 100
    e_N = 0.5
    prox_r_N = 0.25
    prox_r_F = 0.1

    def initalize_ball(mass, radius, r_OS0, v_S0):
        q0 = np.array([*r_OS0, 0, 0, 0])
        u0 = np.array([*v_S0, 0, 0, 0])
        return Ball(mass, radius, g, q0=q0, u0=u0)

    def add_ground_contact(ball, mu, e_N):
        e1, e2, e3 = np.eye(3)
        frame = Frame(A_IK=np.vstack((e3, e1, e2)).T)
        plane = Sphere2Plane(
            frame,
            ball,
            ball.radius,
            mu,
            prox_r_N=prox_r_N,
            prox_r_F=prox_r_F,
            e_N=e_N,
            e_F=0,
        )
        return plane

    # eps = 1.0e-1
    # ball1 = initalize_ball(1, 0.5, np.array([0.0, 0.5, 0]), np.array([0, 0, 0]))
    # ball2 = initalize_ball(1, 0.5, np.array([0.5, 1.5 - eps, 0]), np.array([-1, 0, 0]))
    # ball3 = initalize_ball(
    #     1, 0.5, np.array([0.0, 2.5 - 2 * eps, 0]), np.array([0, 0, 0])
    # )
    # ball4 = initalize_ball(
    #     1, 0.5, np.array([0.5, 3.5 - 3 * eps, 0]), np.array([-1, 0, 0])
    # )
    # balls = [ball1, ball2, ball3, ball4]

    eps = 1.0e-1
    y0 = 1

    # random
    ball1 = initalize_ball(1, 0.5, np.array([-0.5 - eps, y0, 0]), np.array([0, 0, 0]))
    ball2 = initalize_ball(1, 0.5, np.array([0.5 + eps, y0, 0]), np.array([0, 0, 0]))
    ball3 = initalize_ball(
        1, 0.5, np.array([eps, y0 + 1.0 + eps, 0]), np.array([0, 0, 0])
    )
    ball4 = initalize_ball(
        1, 0.5, np.array([-eps, y0 + 2.0 + 2 * eps, 0]), np.array([0, 0, 0])
    )

    # # tower
    # ball1 = initalize_ball(1, 0.5, np.array([0, y0, 0]), np.array([0, 0, 0]))
    # ball2 = initalize_ball(
    #     1, 0.5, np.array([0, y0 + (1.0 + eps), 0]), np.array([0, 0, 0])
    # )
    # ball3 = initalize_ball(
    #     1, 0.5, np.array([0, y0 + 2.0 * (1.0 + eps), 0]), np.array([0, 0, 0])
    # )
    # ball4 = initalize_ball(
    #     1, 0.5, np.array([0, y0 + 3.0 * (1.0 + eps), 0]), np.array([0, 0, 0])
    # )

    # # four parallel
    # ball1 = initalize_ball(1, 0.5, np.array([-1.0 * (1.0 + eps), y0, 0]), np.array([0, 0, 0]))
    # ball2 = initalize_ball(
    #     1, 0.5, np.array([0, y0, 0]), np.array([0, 0, 0])
    # )
    # ball3 = initalize_ball(
    #     1, 0.5, np.array([1.0 * (1.0 + eps), y0, 0]), np.array([0, 0, 0])
    # )
    # ball4 = initalize_ball(
    #     1, 0.5, np.array([2.0 * (1.0 + eps), y0, 0]), np.array([0, 0, 0])
    # )

    balls = [ball1, ball2, ball3, ball4]

    # contact = Sphere2Sphere(ball1, ball2, ball1.radius, ball2.radius, e_N, prox_r_N)

    # assemble model
    model = Model()
    for ball in balls:
        model.add(ball)
        model.add(add_ground_contact(ball, 0, e_N))
    n_balls = len(balls)
    for i in range(n_balls):
        for j in range(i + 1, n_balls):
            model.add(
                Sphere2Sphere(
                    balls[i], balls[j], balls[i].radius, balls[j].radius, e_N, prox_r_N
                )
            )
    model.assemble()

    # solver setup
    t0 = 0
    # t1 = 0.05
    # t1 = 0.1
    t1 = 0.5
    # t1 = 20
    # dt = 1e-1
    # dt = 5e-2
    # dt = 1e-2
    # dt = 5e-3
    dt = 1e-3
    # dt = 5e-4
    # dt = 1e-4

    # solve problem
    # solver_other = NonsmoothGeneralizedAlpha(model, t1, dt)
    # solver_other = NonsmoothEulerBackwardsGGL_V2(model, t1, dt, tol=1.0e-8)
    # solver_other = NonsmoothEulerBackwardsGGL_V3(model, t1, dt, tol=1.0e-8)
    solver_other = NonsmoothHalfExplicitEuler(model, t1, dt)
    # solver_other = Remco(model, t1, dt, tol=1.0e-6)
    sol_other = solver_other.solve()
    t = sol_other.t
    q = sol_other.q
    t_other = sol_other.t
    q_other = sol_other.q
    u_other = sol_other.u
    P_N_other = sol_other.P_N
    # P_F_other = sol_other.P_F
    if type(solver_other) in [
        NonsmoothEulerBackwardsGGL_V2,
        NonsmoothHalfExplicitEuler,
    ]:
        a_other = np.zeros_like(u_other)
        a_other[1:] = (u_other[1:] - u_other[:-1]) / dt
        # la_N_other = sol_other.mu_N
        La_N_other = np.zeros_like(P_N_other)
        P_N_other = sol_other.P_N
    else:
        a_other = sol_other.a
        la_N_other = sol_other.la_N
        La_N_other = sol_other.La_N

    sol_moreau = Moreau(model, t1, dt).solve()
    t_moreau = sol_moreau.t
    q_moreau = sol_moreau.q
    u_moreau = sol_moreau.u
    a_moreau = np.zeros_like(u_moreau)
    a_moreau[1:] = (u_moreau[1:] - u_moreau[:-1]) / dt
    P_N_moreau = sol_moreau.P_N
    P_F_moreau = sol_moreau.P_F

    # # visualization
    # fig, ax = plt.subplots(3, 1)

    # # ax[0].plot([t_moreau[0], t_moreau[-1]], [-0, 0], "-k", label="ground")
    # ax[0].plot(t_moreau, q_moreau[:, 0], "-r", label="x0 - Moreau")
    # ax[0].plot(t_moreau, q_moreau[:, 1], "-g", label="y0 - Moreau")
    # ax[0].plot(t_moreau, q_moreau[:, 6], "-b", label="x1 - Moreau")
    # ax[0].plot(t_moreau, q_moreau[:, 7], "-k", label="y1 - Moreau")

    # ax[0].plot(t_other, q_other[:, 0], "--g", label="x0 - Other")
    # ax[0].plot(t_other, q_other[:, 1], "--b", label="x1 - Other")
    # ax[0].plot(t_other, q_other[:, 6], "--k", label="y0 - Other")
    # ax[0].plot(t_other, q_other[:, 7], "--r", label="y1 - Other")
    # ax[0].grid()
    # ax[0].legend()

    # # ax[1].plot(t_moreau, u_moreau[:, 1], "-r", label="u_y0 - Moreau")
    # # ax[1].plot(t_other, u_other[:, 6], "-b", label="u_y1 - Other")
    # # ax[1].grid()
    # # ax[1].legend()

    # # ax[2].plot(t_moreau, a_moreau[:, 0], "-r", label="a - Moreau")
    # # ax[2].plot(t_other, a_other[:, 0], "-b", label="a - Other")
    # # ax[2].grid()
    # # ax[2].legend()

    # # ax[2].plot(t_moreau, P_N_moreau[:, 0], "-r", label="P_N - Moreau")
    # # ax[2].plot(t_other, la_N_other[:, 0], "-b", label="la_N - Other")
    # # ax[2].plot(t_other, La_N_other[:, 0], "-g", label="La_N - Other")
    # # ax[2].plot(t_other, P_N_other[:, 0], "--k", label="P_N - Other")
    # # ax[2].grid()
    # # ax[2].legend()

    # # plt.show()

    # animation
    # t = t_moreau
    # q = q_moreau
    t = t_other
    q = q_other

    if True:
        lim = 5.5

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.set_xlim(-lim / 2, lim / 2)
        ax.set_ylim(-0.1, lim - 0.1)

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
        ax.plot([-lim, lim], [0, 0], "-k")

        def create():
            coms = []
            bdrys = []
            for ball in balls:
                coms.extend(ax.plot([], [], "ok"))
                bdrys.extend(ax.plot([], [], "-k"))
            return coms, bdrys

        coms, bdrys = create()

        def update(t, q, coms, bdrys):
            for i, ball in enumerate(balls):
                x, y, _ = ball.r_OP(t, q[ball.qDOF])
                x_bdry, y_bdry, _ = ball.boundary(t, q[ball.qDOF])
                coms[i].set_data([x], [y])
                bdrys[i].set_data(x_bdry, y_bdry)

            return coms, bdrys

        def animate(i):
            update(t[i], q[i], coms, bdrys)

        anim = animation.FuncAnimation(
            fig, animate, frames=frames, interval=interval, blit=False
        )

    plt.show()
