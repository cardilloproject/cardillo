import numpy as np
from math import pi

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math import A_IK_basic

from cardillo.model import Model
from cardillo.model.rigid_body import RigidBodyEuler
from cardillo.model.frame import Frame
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane
from cardillo.solver import Moreau, MoreauGGL


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
    g = 9.81
    x0 = -1
    y0 = 1
    x_dot0 = 1
    y_dot0 = 0
    phi0 = 0
    phi_dot0 = 50
    r_OS0 = np.array([x0, y0, 0])
    vS0 = np.array([x_dot0, y_dot0, 0])
    # q0 = np.array([r_OS0[0], r_OS0[1], phi0])
    # u0 = np.array([vS0[0], vS0[1], phi_dot0])
    q0 = np.concatenate([r_OS0, np.array([phi0, 0, 0])])
    u0 = np.concatenate([vS0, np.array([0, 0, phi_dot0])])
    RB = Ball(m, r, q0, u0)

    e1, e2, e3 = np.eye(3)
    frame = Frame(A_IK=np.vstack((e3, e1, e2)).T, r_OP=np.array([0, 0, 0]))
    mu = 0.0  # no friction
    # mu = 0.2
    r_N = 0.1
    e_N = 0.5
    plane = Sphere2Plane(frame, RB, r, mu, prox_r_N=r_N, prox_r_F=r_N, e_N=e_N, e_F=0)

    alpha = pi / 4
    # e1, e2, e3 = A_IK_basic_z(alpha)
    e1, e2, e3 = A_IK_basic(alpha).z()
    frame1 = Frame(A_IK=np.vstack((e3, e1, e2)).T)
    mu = 0.2
    r_N = 0.2
    e_N = 1
    plane_left = Sphere2Plane(frame1, RB, r, mu, prox_r_N=r_N, prox_r_F=r_N, e_N=e_N)

    beta = -pi / 4
    # e1, e2, e3 = A_IK_basic_z(beta)
    e1, e2, e3 = A_IK_basic(beta).z()
    frame2 = Frame(A_IK=np.vstack((e3, e1, e2)).T)
    mu = 0.1
    # r_N = 0.2
    r_N = 0.01
    e_N = 1
    plane_right = Sphere2Plane(frame2, RB, r, mu, prox_r_N=r_N, prox_r_F=r_N, e_N=e_N)

    model = Model()
    model.add(RB)
    model.add(Force(lambda t: np.array([0, -g * m, 0]), RB))
    # model.add(plane)
    # model.add(plane_left)
    model.add(plane)
    # model.add(plane_right)
    # model.add(plane_left)
    model.assemble()

    t0 = 0
    t1 = 1.5
    dt = 5e-3

    # solver_fp = Moreau(model, t1, dt)
    # sol_fp = solver_fp.solve()
    # t_fp = sol_fp.t
    # q_fp = sol_fp.q
    # u_fp = sol_fp.u
    # la_N_fp = sol_fp.la_N
    # la_F_fp = sol_fp.la_F

    # solver_n = Generalized_alpha_3(model, t1, dt, rho_inf=1, numerical_jacobian=0)
    # solver_n = Moreau(model, t1, dt)
    solver_n = MoreauGGL(model, t1, dt, max_iter=50, tol=1.0e-4)
    sol_n = solver_n.solve()
    # sol_n = sol_fp
    t_n = t = sol_n.t
    q_n = q = sol_n.q
    u_n = sol_n.u
    # a_n = sol_n.a
    # la_N_n = sol_n.la_N
    # la_F_n = sol_n.la_F
    # la_N_n = sol_n.P_N
    # la_F_n = sol_n.P_F

    P_N_n = sol_n.P_N
    # P_F_n = sol_n.P_F

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

    # ax[2].set_title('a_x(t)')
    # ax[2].plot(t_fp, a_fp[:, 0], '-r', label='fixed_point')
    # ax[2].plot(t_n, a_n[:, 0], '--b', label='newton')
    # ax[2].legend()

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title("y(t)")
    ax[0].plot(t_fp, q_fp[:, 1], "-r", label="fixed_point")
    ax[0].plot(t_n, q_n[:, 1], "--b", label="newton")
    ax[0].legend()

    ax[1].set_title("u_y(t)")
    ax[1].plot(t_fp, u_fp[:, 1], "-r", label="fixed_point")
    ax[1].plot(t_n, u_n[:, 1], "--b", label="newton")
    ax[1].legend()

    # ax[2].set_title('a_y(t)')
    # ax[2].plot(t_fp, a_fp[:, 1], '-r', label='fixed_point')
    # ax[2].plot(t_n, a_n[:, 1], '--b', label='newton')
    # ax[2].legend()

    # fig, ax = plt.subplots(3, 1)
    # ax[0].set_title('phi(t)')
    # ax[0].plot(t_fp, q_fp[:, 3], '-r', label='fixed_point')
    # ax[0].plot(t_n, q_n[:, 3], '--b', label='newton')
    # ax[0].legend()

    # ax[1].set_title('u_phi(t)')
    # ax[1].plot(t_fp, u_fp[:, -1], '-r', label='fixed_point')
    # ax[1].plot(t_n, u_n[:, -1], '--b', label='newton')
    # ax[1].legend()

    # ax[2].set_title('a_phi(t)')
    # ax[2].plot(t_fp, a_fp[:, -1], '-r', label='fixed_point')
    # ax[2].plot(t_n, a_n[:, -1], '--b', label='newton')
    # ax[2].legend()

    # fig, ax = plt.subplots(3, 1)

    # ax[0].set_title('x(t)')
    # ax[0].plot(t_fp, q_fp[:, 0], '-r', label='fixed_point')
    # ax[0].plot(t_n, q_n[:, 0], '--b', label='newton')
    # ax[0].legend()

    # ax[1].set_title('y(t)')
    # ax[1].plot(t_fp, q_fp[:, 1], '-r', label='fixed_point')
    # ax[1].plot(t_n, q_n[:, 1], '--b', label='newton')
    # ax[1].legend()

    # ax[2].set_title('phi(t)')
    # ax[2].plot(t_fp, q_fp[:, 3], '-r', label='fixed_point')
    # ax[2].plot(t_n, q_n[:, 3], '--b', label='newton')
    # ax[2].legend()

    # fig, ax = plt.subplots(3, 1)

    # ax[0].set_title('u_x(t)')
    # ax[0].plot(t_fp, u_fp[:, 0], '-r', label='fixed_point')
    # ax[0].plot(t_n, u_n[:, 0], '--b', label='newton')
    # ax[0].legend()

    # ax[1].set_title('u_y(t)')
    # ax[1].plot(t_fp, u_fp[:, 1], '-r', label='fixed_point')
    # ax[1].plot(t_n, u_n[:, 1], '--b', label='newton')
    # ax[1].legend()

    # ax[2].set_title('u_phi(t)')
    # ax[2].plot(t_fp, u_fp[:, 3], '-r', label='fixed_point')
    # ax[2].plot(t_n, u_n[:, 3], '--b', label='newton')
    # ax[2].legend()

    fig, ax = plt.subplots(3, 1)

    ax[0].set_title("P_N(t)")
    ax[0].plot(t_fp, P_N_fp[:, 0], "-r", label="fixed_point")
    # ax[0].plot(t_n, sol_n.la_N[:, 0]*dt, '--b', label='newton_la_N')
    # ax[0].plot(t_n, sol_n.La_N[:, 0], '--g', label='newton_La_N')
    ax[0].plot(t_n, P_N_n[:, 0], "--g", label="newton_P_N")
    ax[0].legend()

    # ax[1].set_title("P_Fx(t)")
    # ax[1].plot(t_fp, P_F_fp[:, 0], "-r", label="fixed_point")
    # # ax[1].plot(t_n, sol_n.la_F[:, 0]*dt, '--b', label='newton_la_F')
    # # ax[1].plot(t_n, sol_n.La_F[:, 0], '--g', label='newton_La_F')
    # ax[1].plot(t_n, P_F_n[:, 0], "--g", label="newton_P_N")
    # ax[1].legend()

    # ax[2].set_title("P_Fy(t)")
    # ax[2].plot(t_fp, P_F_fp[:, 1], "-r", label="fixed_point")
    # # ax[2].plot(t_n, sol_n.la_F[:, 1]*dt, '--b', label='newton_la_F')
    # # ax[2].plot(t_n, sol_n.La_F[:, 1], '--g', label='newton_La_F')
    # ax[2].plot(t_n, P_F_n[:, 1], "--g", label="newton_P_N")
    # ax[2].legend()

    # ax[1].set_title('u_y(t)')
    # ax[1].plot(t_fp, u_fp[:, 1], '-r', label='fixed_point')
    # ax[1].plot(t_n, u_n[:, 1], '--b',  label='newton')
    # ax[1].legend()

    # fig, ax = plt.subplots(3, 1)
    # ax[0].set_title('positions')
    # ax[0].plot(t, q[:, 0], '-r', label='x(t)')
    # ax[0].plot(t, q[:, 1], '-b', label='y(t)')
    # ax[0].legend()

    # ax[1].set_title('velocities')
    # ax[1].plot(t, u[:, 0], '-r', label='x_dot(t)')
    # ax[1].plot(t, u[:, 1], '-b', label='y_dot(t)')
    # ax[1].legend()

    # ax[2].set_title('contact forces')
    # ax[2].plot(t, la_N[:, 0], '-r', label='P_N_left(t)')
    # ax[2].plot(t, la_N[:, -1], '-b', label='P_N_right(t)')
    # ax[2].legend()

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
        ax.plot([-2 * y0, 2 * y0], [0, 0], "-k")
        # ax.plot([0, -y0 * np.cos(alpha)], [0, y0 * np.sin(alpha)], '-k')
        # ax.plot([0, y0 * np.cos(beta)], [0, - y0 * np.sin(beta)], '-k')

        def create(t, q):
            x_S, y_S, _ = RB.r_OP(t, q)

            A_IK = RB.A_IK(t, q)
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
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
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
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
