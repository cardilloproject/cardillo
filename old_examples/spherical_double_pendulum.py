import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from cardillo.math.algebra import axis_angle2quat
from cardillo.model import System
from cardillo.model.rigid_body import Rigid_body_quaternion
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import (
    Spherical_joint,
    Rigid_connection,
    Revolute_joint,
)
from cardillo.model.force import Force
from cardillo.solver import Euler_backward, Moreau


class Rigid_cylinder(Rigid_body_quaternion):
    def __init__(self, m, r, l, q0=None, u0=None):
        A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
        C = 1 / 2 * m * r**2
        K_theta_S = np.diag(np.array([A, A, C]))

        super().__init__(m, K_theta_S, q0=q0, u0=u0)


if __name__ == "__main__":
    m = 10
    r = 1
    l = 0.2

    r01 = np.array([0, r, 0])
    # p0 = np.array([1, 0, 0, 0])
    p01 = axis_angle2quat(np.array([1, 0, 0]), np.pi / 2)

    r0_t = np.array([0, 0, 0])
    omega = np.array([0, 0, 0])
    u0 = np.concatenate((r0_t, omega))

    q01 = np.concatenate((r01, p01))
    RB1 = Rigid_cylinder(m, r, l, q01, u0)

    r02 = np.array([0, r, r])
    p02 = np.random.rand(4)  # axis_angle2quat(np.array([0, 0, 1]), np.pi/4)
    p02 = p02 / np.linalg.norm(p02)
    q02 = np.concatenate((r02, p02))
    RB2 = Rigid_cylinder(m, r, l, q02, u0)
    frame = Frame()

    model = System()
    model.add(RB1)
    model.add(RB2)
    model.add(Force(lambda t: np.array([0, 0, -9.81 * m]), RB1))
    model.add(Force(lambda t: np.array([0, 0, -9.81 * m]), RB2))
    model.add(frame)
    # model.add( Spherical_joint(frame, RB1, np.zeros(3)) )
    A_IB = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    model.add(Revolute_joint(frame, RB1, np.zeros(3), A_IB))
    # model.add( Spherical_joint(RB1, RB2, r01) )
    # model.add( Rigid_connection(RB1, RB2, r01) )
    model.add(Revolute_joint(RB1, RB2, r01, A_IB))

    model.assemble()

    t0 = 0
    t1 = 5
    dt = 1e-2
    solver = Euler_backward(
        model, t1, dt, newton_max_iter=50, numerical_jacobian=False, debug=False
    )
    sol = solver.solve()
    t = sol.t
    q = sol.q

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
        x_0, y_0, z_0 = frame.r_OP(t)
        x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])
        x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF])

        A_IK1 = RB1.A_IK(t, q[RB1.qDOF])
        d11 = A_IK1[:, 0]
        d21 = A_IK1[:, 1]
        d31 = A_IK1[:, 2]

        A_IK2 = RB2.A_IK(t, q[RB2.qDOF])
        d12 = A_IK2[:, 0]
        d22 = A_IK2[:, 1]
        d32 = A_IK2[:, 2]

        (COM,) = ax.plot([x_0, x_S1, x_S2], [y_0, y_S1, y_S2], [z_0, z_S1, z_S2], "-ok")
        (d11_,) = ax.plot(
            [x_S1, x_S1 + d11[0]], [y_S1, y_S1 + d11[1]], [z_S1, z_S1 + d11[2]], "-r"
        )
        (d21_,) = ax.plot(
            [x_S1, x_S1 + d21[0]], [y_S1, y_S1 + d21[1]], [z_S1, z_S1 + d21[2]], "-g"
        )
        (d31_,) = ax.plot(
            [x_S1, x_S1 + d31[0]], [y_S1, y_S1 + d31[1]], [z_S1, z_S1 + d31[2]], "-b"
        )
        (d12_,) = ax.plot(
            [x_S2, x_S2 + d12[0]], [y_S2, y_S2 + d12[1]], [z_S2, z_S2 + d12[2]], "-r"
        )
        (d22_,) = ax.plot(
            [x_S2, x_S2 + d22[0]], [y_S2, y_S2 + d22[1]], [z_S2, z_S2 + d22[2]], "-g"
        )
        (d32_,) = ax.plot(
            [x_S2, x_S2 + d32[0]], [y_S2, y_S2 + d32[1]], [z_S2, z_S2 + d32[2]], "-b"
        )

        return COM, d11_, d21_, d31_, d12_, d22_, d32_

    def update(t, q, COM, d11_, d21_, d31_, d12_, d22_, d32_):
        x_0, y_0, z_0 = frame.r_OP(t)
        x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])
        x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF])

        A_IK1 = RB1.A_IK(t, q[RB1.qDOF])
        d11 = A_IK1[:, 0]
        d21 = A_IK1[:, 1]
        d31 = A_IK1[:, 2]

        A_IK2 = RB2.A_IK(t, q[RB2.qDOF])
        d12 = A_IK2[:, 0]
        d22 = A_IK2[:, 1]
        d32 = A_IK2[:, 2]

        COM.set_data([x_0, x_S1, x_S2], [y_0, y_S1, y_S2])
        COM.set_3d_properties([z_0, z_S1, z_S2])

        d11_.set_data([x_S1, x_S1 + d11[0]], [y_S1, y_S1 + d11[1]])
        d11_.set_3d_properties([z_S1, z_S1 + d11[2]])

        d21_.set_data([x_S1, x_S1 + d21[0]], [y_S1, y_S1 + d21[1]])
        d21_.set_3d_properties([z_S1, z_S1 + d21[2]])

        d31_.set_data([x_S1, x_S1 + d31[0]], [y_S1, y_S1 + d31[1]])
        d31_.set_3d_properties([z_S1, z_S1 + d31[2]])

        d12_.set_data([x_S2, x_S2 + d12[0]], [y_S2, y_S2 + d12[1]])
        d12_.set_3d_properties([z_S2, z_S2 + d12[2]])

        d22_.set_data([x_S2, x_S2 + d22[0]], [y_S2, y_S2 + d22[1]])
        d22_.set_3d_properties([z_S2, z_S2 + d22[2]])

        d32_.set_data([x_S2, x_S2 + d32[0]], [y_S2, y_S2 + d32[1]])
        d32_.set_3d_properties([z_S2, z_S2 + d32[2]])

        return COM, d11_, d21_, d31_, d12_, d22_, d32_

    COM, d11_, d21_, d31_, d12_, d22_, d32_ = init(0, q[0])

    def animate(i):
        update(t[i], q[i], COM, d11_, d21_, d31_, d12_, d22_, d32_)

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
