import numpy as np
from math import pi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from cardillo.math.algebra import A_IK_basic_z

from cardillo.model import System
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.explicit import (
    Revolute_joint,
    Rigid_connection,
)
from cardillo.model.rigid_body import Rigid_body_rel_kinematics
from cardillo.model.force import Force
from cardillo.solver import (
    Scipy_ivp,
    Euler_backward,
    Generalized_alpha_1,
    Generalized_alpha_3,
)

if __name__ == "__main__":
    m = 1
    r = 0.1
    l = 2
    g = 9.81
    n_segments = 7
    body_indices = np.arange(2, n_segments * 3 + 1, step=3)

    A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
    C = 1 / 2 * m * r**2
    K_theta_S = np.diag(np.array([A, C, A]))

    alpha0 = pi / 2
    alpha_dot0 = 0

    # first chain segment
    r_OB1 = np.zeros(3)
    A_IB1 = np.eye(3)
    origin = Frame(r_OP=r_OB1, A_IK=A_IB1)
    joint1 = Revolute_joint(
        r_OB1, A_IB1, q0=np.array([alpha0]), u0=np.array([alpha_dot0])
    )
    A_IK10 = A_IK_basic_z(alpha0)
    r_OS10 = -0.5 * l * A_IK10[:, 1]
    RB1 = Rigid_body_rel_kinematics(
        m, K_theta_S, joint1, origin, r_OS0=r_OS10, A_IK0=A_IK10
    )

    model = System()
    model.add(origin)
    model.add(joint1)
    model.add(RB1)
    model.add(Force(lambda t: np.array([0, -g * m, 0]), RB1))

    # other chain segments
    for i in range(n_segments - 1):

        r_OB2 = -(i + 1) * l * A_IK10[:, 1]
        A_IB2 = A_IK10
        model.add(Revolute_joint(r_OB2, A_IB2))
        # model.add( Rigid_connection(r_OB2, A_IB2) )
        A_IK20 = A_IK10
        r_OS20 = r_OB2 - 0.5 * l * A_IK20[:, 1]
        model.add(
            Rigid_body_rel_kinematics(
                m,
                K_theta_S,
                model.contributions[-1],
                model.contributions[-3],
                r_OS0=r_OS20,
                A_IK0=A_IK20,
            )
        )

        model.add(Force(lambda t: np.array([0, -g * m, 0]), model.contributions[-1]))

    model.assemble()

    t0 = 0
    t1 = 2
    dt = 5e-2
    # solver = Scipy_ivp(model, t1, dt)
    # solver = Euler_backward(model, t1, dt, debug=True)
    # solver = Generalized_alpha_1(model, t1, dt)
    solver = Generalized_alpha_3(model, t1, dt)
    sol = solver.solve()
    t = sol.t
    q = sol.q

    # animate configurations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    scale = (n_segments + 1) * l
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=-scale, top=scale)

    (chain,) = ax.plot([], [], "-ok")

    def update(t, q, chain):
        x = []
        y = []
        z = []

        x_, y_, z_ = origin.r_OP(t, q[origin.qDOF])
        x.append(x_)
        y.append(y_)
        z.append(z_)

        for i in body_indices:
            body = model.contributions[i]

            x_, y_, z_ = body.r_OP(t, q[body.qDOF], K_r_SP=np.array([0, -l / 2, 0]))
            x.append(x_)
            y.append(y_)
            z.append(z_)

        chain.set_data(x, y)
        chain.set_3d_properties(z)

        return (chain,)

    def animate(i):
        update(t[i], q[i], chain)

    # compute naimation interval according to te - ts = frames * interval / 1000
    frames = len(t)
    interval = dt * 1000
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )
    plt.show()
