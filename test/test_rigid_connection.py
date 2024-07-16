import numpy as np
import pytest
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math import A_IB_basic, axis_angle2quat, cross3
from cardillo import System
from cardillo.discrete import Frame, RigidBody
from cardillo.constraints import Revolute, RigidConnection
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.forces import Force
from cardillo.solver import ScipyIVP, ScipyDAE, BackwardEuler, Moreau, Rattle

solvers_and_kwargs = [
    (ScipyIVP, {}),
    (ScipyDAE, {}),
    (BackwardEuler, {}),
    (Moreau, {}),
    (Rattle, {}),
]


@pytest.mark.parametrize("Solver, kwargs", solvers_and_kwargs)
def test_rigid_connection(Solver, kwargs, show=False):
    ############
    # parameters
    ############
    m = 1
    L = 2
    theta = 1 / 12 * m * (L**2)
    theta_O = theta + m * (L**2) / 4
    theta1 = theta2 = 1 / 12 * m / 2 * (L**2) / 4
    B_theta_C1 = B_theta_C2 = theta1 * np.diag((1, 1e-8, 1))
    g = 9.81

    k = 1e2
    d = 2e1
    # k = 0
    # d = 0

    ##############
    # moving frame
    ##############
    omega = 10
    A = L / 10
    # e = lambda t: A * np.cos(omega * t)
    # e_t = lambda t: -A * omega * np.sin(omega * t)
    # e_tt = lambda t: -A * omega * omega * np.cos(omega * t)

    e = lambda t: A * np.sin(omega * t)
    e_t = lambda t: A * omega * np.cos(omega * t)
    e_tt = lambda t: -A * omega * omega * np.sin(omega * t)

    # e = lambda t: A * t
    # e_t = lambda t: A
    # e_tt = lambda t: 0

    # e = lambda t: 0
    # e_t = lambda t: 0
    # e_tt = lambda t: 0

    r_OP = lambda t: np.array([e(t), 0, 0])
    v_P = lambda t: np.array([e_t(t), 0, 0])
    a_P = lambda t: np.array([e_tt(t), 0, 0])
    frame = Frame(r_OP=r_OP, r_OP_t=v_P, r_OP_tt=a_P)

    ##############
    # rigid bodies
    ##############
    B_r_PC1 = np.array([0, -L / 4, 0])  # center of mass half rigid body 1
    B_r_PC2 = np.array([0, -3 * L / 4, 0])  # center of mass half rigid body 2

    phi0 = 0
    phi_dot0 = 0
    B_omega0 = np.array([0, 0, phi_dot0])
    A_IB0 = A_IB_basic(phi0).z

    r_OC10 = r_OP(0) + A_IB0 @ B_r_PC1
    v_C10 = v_P(0) + A_IB0 @ (cross3(B_omega0, B_r_PC1))
    r_OC20 = r_OP(0) + A_IB0 @ B_r_PC2
    v_C20 = v_P(0) + A_IB0 @ (cross3(B_omega0, B_r_PC2))

    p0 = axis_angle2quat(np.array([0, 0, 1]), phi0)
    q10 = np.concatenate((r_OC10, p0))
    q20 = np.concatenate((r_OC20, p0))
    u10 = np.concatenate((v_C10, B_omega0))
    u20 = np.concatenate((v_C20, B_omega0))
    RB1 = RigidBody(m / 2, B_theta_C1, q0=q10, u0=u10)
    RB2 = RigidBody(m / 2, B_theta_C2, q0=q20, u0=u20)

    ##################################
    # revolute joint and spring damper
    ##################################
    joint = Revolute(frame, RB1, 2, r_OJ0=r_OP(0), A_IJ0=np.eye(3))
    spring_damper = SpringDamper(
        joint,
        k,
        d,
        l_ref=-phi0,
        compliance_form=False,
        name="spring_damper",
    )

    system = System()
    system.add(frame)
    system.add(RB1)
    system.add(RB2)
    gravity1 = Force(np.array([0, -m / 2 * g, 0]), RB1)
    system.add(gravity1)
    gravity2 = Force(np.array([0, -m / 2 * g, 0]), RB2)
    system.add(gravity2)
    system.add(joint)
    system.add(spring_damper)
    system.add(RigidConnection(RB1, RB2))
    system.assemble()

    ############################################################################
    #                   solver
    ############################################################################
    t0 = 0
    t1 = 2
    dt = 1e-2
    sol = Solver(system, t1, dt, *kwargs).solve()
    t = sol.t
    q = sol.q
    ############################################################################
    #                   animation
    ############################################################################
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        scale = L
        ax.set_xlim3d(left=-scale, right=scale)
        ax.set_ylim3d(bottom=-scale, top=scale)
        ax.set_zlim3d(bottom=-scale, top=scale)
        ax.view_init(vertical_axis="y")

        def init(t, q):
            x_0, y_0, z_0 = frame.r_OP(t, q[frame.qDOF])
            x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])
            x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF])

            A_IB1 = RB1.A_IB(t, q[RB1.qDOF])
            d11 = A_IB1[:, 0]
            d21 = A_IB1[:, 1]
            d31 = A_IB1[:, 2]

            A_IB2 = RB2.A_IB(t, q[RB2.qDOF])
            d12 = A_IB2[:, 0]
            d22 = A_IB2[:, 1]
            d32 = A_IB2[:, 2]

            (COM,) = ax.plot(
                [x_0, x_S1, x_S2], [y_0, y_S1, y_S2], [z_0, z_S1, z_S2], "-ok"
            )
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
            (d12_,) = ax.plot(
                [x_S2, x_S2 + d12[0]],
                [y_S2, y_S2 + d12[1]],
                [z_S2, z_S2 + d12[2]],
                "-r",
            )
            (d22_,) = ax.plot(
                [x_S2, x_S2 + d22[0]],
                [y_S2, y_S2 + d22[1]],
                [z_S2, z_S2 + d22[2]],
                "-g",
            )
            (d32_,) = ax.plot(
                [x_S2, x_S2 + d32[0]],
                [y_S2, y_S2 + d32[1]],
                [z_S2, z_S2 + d32[2]],
                "-b",
            )

            return COM, d11_, d21_, d31_, d12_, d22_, d32_

        def update(t, q, COM, d11_, d21_, d31_, d12_, d22_, d32_):
            x_0, y_0, z_0 = frame.r_OP(t, q[frame.qDOF])
            x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])
            x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF])

            A_IB1 = RB1.A_IB(t, q[RB1.qDOF])
            d11 = A_IB1[:, 0]
            d21 = A_IB1[:, 1]
            d31 = A_IB1[:, 2]

            A_IB2 = RB2.A_IB(t, q[RB2.qDOF])
            d12 = A_IB2[:, 0]
            d22 = A_IB2[:, 1]
            d32 = A_IB2[:, 2]

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

    # reference solution
    def eqm(t, x):
        dx = np.zeros(2)
        dx[0] = x[1]
        dx[1] = (
            -d * x[1]
            - k * x[0]
            - 0.5 * m * L * (e_tt(t) * np.cos(x[0]) + g * np.sin(x[0]))
        ) / theta_O
        return dx

    dt = 1e-3
    x0 = np.array([phi0, phi_dot0])
    ref = solve_ivp(
        eqm,
        [t0, t1],
        x0,
        method="RK45",
        t_eval=np.arange(t0, t1 + dt, dt),
        rtol=1e-8,
        atol=1e-12,
    )
    x_ = []
    y_ = []
    r_OC = np.zeros((3, len(q[:, 0])))
    for i, ti in enumerate(t):
        r_OC = RB1.r_OP(ti, q[i, RB1.qDOF], B_r_CP=B_r_PC1)
        x_.append(r_OC[0])
        y_.append(r_OC[1])

    if show:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(t, x_, "--gx", label="cardillo")
        ax[1].plot(t, y_, "--gx", label="cardillo")
        x_ref = ref.y
        t_ref = ref.t

        ax[0].plot(t_ref, e(t_ref) + L / 2 * np.sin(x_ref[0]), "-r", label="ODE")
        ax[1].plot(t_ref, -L / 2 * np.cos(x_ref[0]), "-r", label="ODE")

        ax[0].legend()
        ax[0].grid()
        ax[1].legend()
        ax[1].grid()
        plt.show()


if __name__ == "__main__":
    for p in solvers_and_kwargs:
        test_rigid_connection(*p, show=True)
