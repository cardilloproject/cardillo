import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from cardillo.math.algebra import A_IK_basic_z, axis_angle2quat

from cardillo.model import System
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.explicit import (
    Revolute_joint,
    Spherical_joint,
)
from cardillo.model.rigid_body import Rigid_body_rel_kinematics
from cardillo.model.force import Force
from cardillo.solver import (
    Euler_forward,
    Scipy_ivp,
    Euler_backward,
    Generalized_alpha_1,
)
from scipy.integrate import solve_ivp

if __name__ == "__main__":
    animate = False
    m = 10
    r = 1
    l = 0.2
    g = 9.81

    A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
    C = 1 / 2 * m * r**2
    K_theta_S = np.diag(np.array([A, A, C]))

    theta_O = C + m * r**2

    # e = lambda t: 0
    # e_t = lambda t: 0
    # e_tt = lambda t: 0

    omega = 10
    amplitude = r / 10

    e = lambda t: amplitude * np.sin(omega * t)
    e_t = lambda t: amplitude * omega * np.cos(omega * t)
    e_tt = lambda t: -amplitude * omega * omega * np.sin(omega * t)

    alpha0 = np.pi / 4
    alpha_dot0 = 0

    r_OP = lambda t: np.array([e(t), 0, 0])
    v_P = lambda t: np.array([e_t(t), 0, 0])
    a_P = lambda t: np.array([e_tt(t), 0, 0])

    r_OB1 = r_OP(0)

    A_IB1 = np.eye(3)
    origin = Frame(r_OP=r_OP, r_OP_t=v_P, r_OP_tt=a_P, A_IK=A_IB1)
    # joint1 = Revolute_joint(r_OB1, A_IB1, q0=np.array([alpha0]), u0=np.array([alpha_dot0]))
    joint1 = Spherical_joint(
        r_OB1,
        A_IB1,
        q0=axis_angle2quat(np.array([0, 0, 1]), alpha0),
        u0=np.array([0, 0, alpha_dot0]),
    )
    A_IK10 = A_IK_basic_z(alpha0)
    r_OS10 = -r * A_IK10[:, 1]
    RB1 = Rigid_body_rel_kinematics(
        m, K_theta_S, joint1, origin, r_OS0=r_OS10, A_IK0=A_IK10
    )

    model = System()
    model.add(origin)
    model.add(joint1)
    model.add(RB1)
    model.add(Force(lambda t: np.array([0, -g * m, 0]), RB1))

    model.assemble()

    t0 = 0
    t1 = 5
    dt = 1e-3
    # # solver = Euler_forward(model, t1, dt)
    # solver = Scipy_ivp(model, t1, dt)
    # solver = Euler_backward(model, t1, dt)
    solver = Generalized_alpha_1(
        model, t1, atol=1e-3, rtol=0, t_eval=np.arange(t0, t1, dt), newton_tol=1.0e-6
    )

    sol = solver.solve()
    t = sol.t
    q = sol.q

    if animate:
        # animate configurations
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        scale = 2 * r
        ax.set_xlim3d(left=-scale, right=scale)
        ax.set_ylim3d(bottom=-scale, top=scale)
        ax.set_zlim3d(bottom=-scale, top=scale)

        def init(t, q):
            x_0, y_0, z_0 = origin.r_OP(t)
            x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])

            A_IK1 = RB1.A_IK(t, q[RB1.qDOF])
            d11 = A_IK1[:, 0]
            d21 = A_IK1[:, 1]
            d31 = A_IK1[:, 2]

            (COM,) = ax.plot([x_0, x_S1], [y_0, y_S1], [z_0, z_S1], "-ok")
            (d11_,) = ax.plot(
                [x_S1, x_S1 + d11[0]],
                [y_S1, y_S1 + d11[1]],
                [z_S1, z_S1 + d11[2]],
                "-r",
            )
            (d21_,) = ax.plot(
                [x_S1, x_S1 + d21[0]],
                [y_S1, y_S1 + d21[1]],
                [z_S1, z_S1 + d21[2]],
                "-g",
            )
            (d31_,) = ax.plot(
                [x_S1, x_S1 + d31[0]],
                [y_S1, y_S1 + d31[1]],
                [z_S1, z_S1 + d31[2]],
                "-b",
            )

            return COM, d11_, d21_, d31_

        def update(t, q, COM, d11_, d21_, d31_):
            x_0, y_0, z_0 = origin.r_OP(t)
            x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])

            A_IK1 = RB1.A_IK(t, q[RB1.qDOF])
            d11 = A_IK1[:, 0]
            d21 = A_IK1[:, 1]
            d31 = A_IK1[:, 2]

            COM.set_data([x_0, x_S1], [y_0, y_S1])
            COM.set_3d_properties([z_0, z_S1])

            d11_.set_data([x_S1, x_S1 + d11[0]], [y_S1, y_S1 + d11[1]])
            d11_.set_3d_properties([z_S1, z_S1 + d11[2]])

            d21_.set_data([x_S1, x_S1 + d21[0]], [y_S1, y_S1 + d21[1]])
            d21_.set_3d_properties([z_S1, z_S1 + d21[2]])

            d31_.set_data([x_S1, x_S1 + d31[0]], [y_S1, y_S1 + d31[1]])
            d31_.set_3d_properties([z_S1, z_S1 + d31[2]])

            return COM, d11_, d21_, d31_

        COM, d11_, d21_, d31_ = init(0, q[0])

        def animate(i):
            update(t[i], q[i], COM, d11_, d21_, d31_)

        # compute naimation interval according to te - ts = frames * interval / 1000
        frames = len(t)
        interval = dt * 1000
        anim = animation.FuncAnimation(
            fig, animate, frames=frames, interval=interval, blit=False
        )
        # fps = int(np.ceil(frames / (te - ts))) / 10
        # writer = animation.writers['ffmpeg'](fps=fps, bitrate=1800)
        # # anim.save('directorRigidBodyPendulum.mp4', writer=writer)

        plt.show()

    fig, ax = plt.subplots()
    # ax.plot(t, q[:, 0], '-x')
    ax.plot(t, 2 * np.arcsin(q[:, 3]), "-x")

    # reference solution
    def eqm(t, x):
        dx = np.zeros(2)
        dx[0] = x[1]
        dx[1] = -m * r * (e_tt(t) * np.cos(x[0]) + g * np.sin(x[0])) / theta_O
        return dx

    dt = 0.001

    x0 = np.array([alpha0, alpha_dot0])
    ref = solve_ivp(
        eqm,
        [t0, t1],
        x0,
        method="RK45",
        t_eval=np.arange(t0, t1 + dt, dt),
        rtol=1e-8,
        atol=1e-12,
    )  # MATLAB ode45
    x = ref.y
    t_ref = ref.t

    # plot reference solution
    ax.plot(t_ref, x[0])

    plt.show()
