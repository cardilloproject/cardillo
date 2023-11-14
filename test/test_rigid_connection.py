import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math import A_IK_basic, axis_angle2quat, cross3
from cardillo import System
from cardillo.discrete import Frame, RigidBody
from cardillo.constraints import Revolute, RigidConnection
from cardillo.forces import Force, PDRotational, LinearSpring, LinearDamper
from cardillo.solver import BackwardEuler, ScipyIVP


def run(revolute_joint_used=False):
    # parameters
    m = 1
    L = 2
    theta = 1 / 12 * m * (L**2)
    theta_O = theta + m * (L**2) / 4
    theta1 = theta2 = 1 / 12 * m / 2 * (L**2) / 4
    # K_theta_S = theta * np.eye(3)
    K_theta_S1 = K_theta_S2 = theta1 * np.diag((1, 1e-8, 1))
    g = 9.81
    omega = 10
    A = L / 10

    k = 1e2
    d = 2e1

    e = lambda t: A * np.cos(omega * t)
    e_t = lambda t: -A * omega * np.sin(omega * t)
    e_tt = lambda t: -A * omega * omega * np.cos(omega * t)

    # e = lambda t: A * np.sin(omega * t)
    # e_t = lambda t: A * omega * np.cos(omega * t)
    # e_tt = lambda t: -A * omega * omega * np.sin(omega * t)

    # e = lambda t: A * t
    # e_t = lambda t: A
    # e_tt = lambda t: 0

    # e = lambda t: 0
    # e_t = lambda t: 0
    # e_tt = lambda t: 0

    r_OP = lambda t: np.array([e(t), 0, 0])
    v_P = lambda t: np.array([e_t(t), 0, 0])
    a_P = lambda t: np.array([e_tt(t), 0, 0])

    K_r_SP = np.array([0, L / 2, 0])  # center of mass single rigid body
    K_r_SP1 = np.array([0, L / 4, 0])  # center of mass half rigid body 1
    K_r_SP2 = np.array([0, 3 * L / 4, 0])  # center of mass half rigid body 2

    phi0 = 0.5
    phi_dot0 = 0
    K_omega0 = np.array([0, 0, phi_dot0])
    A_IK0 = A_IK_basic(phi0).z()

    # single rigid body
    r_OS0 = r_OP(0) - A_IK0 @ K_r_SP
    v_S0 = v_P(0) + A_IK0 @ (cross3(K_omega0, K_r_SP))

    # connected rigid bodies
    r_OS10 = r_OP(0) - A_IK0 @ K_r_SP1
    v_S10 = v_P(0) + A_IK0 @ (cross3(K_omega0, K_r_SP1))
    r_OS20 = r_OP(0) - A_IK0 @ K_r_SP2
    v_S20 = v_P(0) + A_IK0 @ (cross3(K_omega0, K_r_SP2))

    system = System()

    frame = Frame(r_OP=r_OP, r_OP_t=v_P, r_OP_tt=a_P)
    system.add(frame)

    p0 = axis_angle2quat(np.array([0, 0, 1]), phi0)
    q10 = np.concatenate((r_OS10, p0))
    q20 = np.concatenate((r_OS20, p0))
    u10 = np.concatenate((v_S10, K_omega0))
    u20 = np.concatenate((v_S20, K_omega0))
    RB1 = RigidBody(m / 2, K_theta_S1, q0=q10, u0=u10)
    RB2 = RigidBody(m / 2, K_theta_S2, q0=q20, u0=u20)

    if revolute_joint_used:
        joint = Revolute(frame, RB1, 2, r_OP(0), np.eye(3))
    else:
        joint = PDRotational(Revolute, LinearSpring, LinearDamper)(
            frame,
            RB1,
            2,
            r_OP(0),
            np.eye(3),
            k=k,
            d=d,
            g_ref=-phi0,
        )

    system.add(RB1)
    system.add(RB2)
    gravity1 = Force(np.array([0, -m / 2 * g, 0]), RB1)
    system.add(gravity1)
    gravity2 = Force(np.array([0, -m / 2 * g, 0]), RB2)
    system.add(gravity2)
    system.add(joint)
    system.add(RigidConnection(RB1, RB2))
    system.assemble()

    ############################################################################
    #                   solver
    ############################################################################
    t0 = 0
    t1 = 2
    dt = 1e-2

    # solver = BackwardEuler(system, t1, dt) # TODO: Fix singular jacobian.
    solver = ScipyIVP(system, t1, dt)

    sol = solver.solve()
    t = sol.t
    q = sol.q
    ############################################################################
    #                   animation
    ############################################################################
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
        x_0, y_0, z_0 = r_OP(t)
        x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])
        x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF])
        x_RC, y_RC, z_RC = RB1.r_OP(t, q[RB1.qDOF], K_r_SP=K_r_SP1)
        x_RB2, y_RB2, z_RB2 = RB2.r_OP(t, q[RB2.qDOF], K_r_SP=K_r_SP1)

        A_IK1 = RB1.A_IK(t, q[RB1.qDOF])
        d11 = A_IK1[:, 0]
        d21 = A_IK1[:, 1]
        d31 = A_IK1[:, 2]

        A_IK2 = RB2.A_IK(t, q[RB2.qDOF])
        d12 = A_IK2[:, 0]
        d22 = A_IK2[:, 1]
        d32 = A_IK2[:, 2]

        (rb,) = ax.plot(
            [x_0, x_RC, x_RB2], [y_0, y_RC, y_RB2], [z_0, z_RC, z_RB2], "-k"
        )
        (COM,) = ax.plot([x_0, x_S1, x_S2], [y_0, y_S1, y_S2], [z_0, z_S1, z_S2], "ok")
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

        return COM, rb, d11_, d21_, d31_, d12_, d22_, d32_

    def update(t, q, COM, rb, d11_, d21_, d31_, d12_, d22_, d32_):
        # def update(t, q, COM, d11_, d21_, d31_):
        x_0, y_0, z_0 = r_OP(t)
        # TODO is this -L/2?
        x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF], K_r_SP=np.array([0, -L / 2, 0]))
        x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF], K_r_SP=np.array([0, -L / 2, 0]))
        x_RC, y_RC, z_RC = RB1.r_OP(t, q[RB1.qDOF], K_r_SP=K_r_SP1)
        x_RB2, y_RB2, z_RB2 = RB2.r_OP(t, q[RB2.qDOF], K_r_SP=K_r_SP1)

        A_IK1 = RB1.A_IK(t, q[RB1.qDOF])
        d11 = A_IK1[:, 0]
        d21 = A_IK1[:, 1]
        d31 = A_IK1[:, 2]

        A_IK2 = RB2.A_IK(t, q[RB2.qDOF])
        d12 = A_IK2[:, 0]
        d22 = A_IK2[:, 1]
        d32 = A_IK2[:, 2]

        rb.set_data([x_0, x_RC, x_RB2], [y_0, y_RC, y_RB2])
        rb.set_3d_properties([z_0, z_RC, z_RB2])
        COM.set_data([x_0, x_S1, x_S2], [y_0, y_S1, y_S2])
        COM.set_3d_properties([z_0, z_S1, z_S2])
        # COM.set_data([x_0, x_S1], [y_0, y_S1])
        # COM.set_3d_properties([z_0, z_S1])

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

    COM, rb, d11_, d21_, d31_, d12_, d22_, d32_ = init(0, q[0])

    def animate(i):
        update(t[i], q[i], COM, rb, d11_, d21_, d31_, d12_, d22_, d32_)

    # compute naimation interval according to te - ts = frames * interval / 1000
    frames = len(t)
    interval = dt * 1000
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    # reference solution
    def eqm_revolute_joint(t, x):
        dx = np.zeros(2)
        dx[0] = x[1]
        dx[1] = -0.5 * m * L * (e_tt(t) * np.cos(x[0]) + g * np.sin(x[0])) / theta_O
        return dx

    def eqm_pd_rotational_joint(t, x):
        dx = np.zeros(2)
        dx[0] = x[1]
        dx[1] = (
            -d * x[1]
            - k * x[0]
            - 0.5 * m * L * (e_tt(t) * np.cos(x[0]) + g * np.sin(x[0]))
        ) / theta_O
        return dx

    eqm = eqm_revolute_joint if revolute_joint_used else eqm_pd_rotational_joint
    dt = 0.001

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
    r_OS = np.zeros((3, len(q[:, 0])))
    r_OS1 = np.zeros((3, len(q[:, 0])))
    for i, ti in enumerate(t):
        r_OS = q[i, :3] - (RB1.A_IK(ti, q[i, :7]) @ K_r_SP1)
        x_.append(r_OS[0])
        y_.append(r_OS[1])
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, x_, "--gx")
    ax[1].plot(t, y_, "--gx")
    x = ref.y
    t = ref.t

    ax[0].plot(t, e(t) + L / 2 * np.sin(x[0]), "-r")
    ax[1].plot(t, -L / 2 * np.cos(x[0]), "-r")

    # fig, ax = plt.subplots(1, 1)
    # # delta_phi = [abs(norm(quat2axis_angle(qi[RB1.qDOF[3:]])) - norm(quat2axis_angle(qi[RB2.qDOF[3:]])) ) for qi in q]
    # n1 = [np.linalg.norm(RB1.A_IK(ti, qi[RB1.qDOF])) for ti, qi in zip(sol.t, q)]
    # n2 = [np.linalg.norm(RB2.A_IK(ti, qi[RB2.qDOF])) for ti, qi in zip(sol.t, q)]
    # ax.plot(sol.t, n1)
    # ax.plot(sol.t, n2)

    plt.show()


if __name__ == "__main__":
    run(True)
    run(False)
