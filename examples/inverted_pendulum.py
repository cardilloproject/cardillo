import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from scipy.interpolate import interp1d
from scipy.optimize import minimize

from cardillo import System

from cardillo.discrete import RigidBody
from cardillo.constraints import Revolute
from cardillo.forces import Force
from cardillo.transmissions import RotationalTransmission
from cardillo.actuators import Motor, PDcontroller, PIDcontroller

from cardillo.solver import Moreau

from cardillo.math import A_IK_basic, cross3
from cardillo.math import smoothstep2

if __name__ == "__main__":
    # only compute forward dynamics of unforced pendulum
    only_unforced_dynmics = False

    # desired swing-up trajectory
    desired_trajectory = ["homogeneous", "optimal_torque"][0]

    # feed forward method
    feed_forward = [None, "inverse_dynamics"][1]

    # feed back controller
    feed_back = [None, "PD", "PID"][1]

    l = 1
    m = 1
    theta_S = m * (l**2) / 12

    phi0 = 0  # np.pi / 2
    phi_dot0 = 0

    system = System()

    K_r_OS = np.array([0, -l, 0])
    A_IK0 = A_IK_basic(phi0).z()
    r_OS0 = A_IK0 @ K_r_OS
    K_Omega0 = np.array([0, 0, phi_dot0])
    v_S0 = cross3(K_Omega0, r_OS0)  # I_Omega0 = K_Omega0

    q0 = RigidBody.pose2q(r_OS0, A_IK0)
    u0 = np.concatenate([v_S0, K_Omega0])
    pendulum = RigidBody(m, theta_S * np.eye(3), q0=q0, u0=u0)
    pendulum.name = "pendulum"

    joint = Revolute(
        system.origin, pendulum, axis=2, angle0=phi0, A_IB0=A_IK_basic(-np.pi / 2).z()
    )
    joint.name = "revolute joint"

    gravity = Force(np.array([0, -10 * m, 0]), pendulum)

    system.add(pendulum, gravity, joint)

    # # add moment
    tau = lambda t: 0
    motor = Motor(RotationalTransmission)(tau, subsystem=joint)
    system.add(motor)

    # # add spring damper
    # stiffness = 10
    # damping = 2
    # l0 = 0 #np.pi
    # force_law = lambda t, l, l_dot: stiffness * (l - l0) + damping * l_dot
    # spring = ScalarForceLaw(RotationalTransmission)(force_law, subsystem=joint)
    # system.add(spring)

    # add moment
    # force_law = lambda t, l, l_dot: - 1
    # moment = ScalarForceLaw(RotationalTransmission)(force_law, subsystem=joint)
    # system.add(moment)

    system.assemble()

    ############
    # simulation
    ############

    if only_unforced_dynmics:
        t1 = 6
        dt = 1e-2
        sol = Moreau(system, t1, dt).solve()

        joint.reset()
        angle = []
        for ti, qi in zip(sol.t, sol.q):
            angle.append(joint.angle(ti, qi))

        plt.plot(sol.t, angle)
        plt.show()
        exit()

    ####################
    # inverse kinematics
    ####################

    t0 = 0
    t1 = 1
    dt = 1e-3

    if desired_trajectory == "homogeneous":
        t = np.arange(t0, t1, dt)
        phi = np.pi * smoothstep2(t, x_min=t0, x_max=t1)
        phi_dot = (phi[1:] - phi[:-1]) / dt

        # fig, ax = plt.subplots()
        # ax.plot(t, phi, "k")
        # ax.plot(t[1:], phi_dot, "r")
        # plt.show()

    elif desired_trajectory == "optimal_torque":
        # initial unknowns
        t = np.arange(t0, t1, dt)
        phi = np.pi * smoothstep2(t, x_min=t0, x_max=t1)
        phi_dot = (phi[1:] - phi[:-1]) / dt
        A_IK = np.array([A_IK_basic(phi_).z() for phi_ in phi])
        r_OS = np.array([A_IK_ @ K_r_OS for A_IK_ in A_IK])
        K_Omega = np.array([[0, 0, phi_dot_] for phi_dot_ in phi_dot])
        v_S = np.array(
            [cross3(K_Omega_, r_OS_) for K_Omega_, r_OS_ in zip(K_Omega, r_OS[1:])]
        )  # I_Omega = K_Omega!!

        q = np.array(
            [RigidBody.pose2q(r_OS_, A_IK_) for r_OS_, A_IK_ in zip(r_OS, A_IK)]
        )
        u = np.concatenate([v_S, K_Omega], axis=1)

        tk1 = t[2:]
        nt = len(tk1)
        nx = system.nq + system.nu + system.nla_g + system.ntau

        def pack_x(q, u, la_g, tau):
            return np.concatenate([q, u, la_g, tau], axis=1).flatten(order="C")

        def unpack_x(x):
            return np.array_split(
                x.reshape((nt, nx), order="C"),
                [
                    system.nq,
                    system.nq + system.nu,
                    system.nq + system.nu + system.nla_g,
                ],
                axis=1,
            )

        x0 = pack_x(q[2:], u[1:], np.zeros((nt, system.nla_g)), np.zeros((nt, 1)))

        def constraints(x):
            qk1, uk1, la_g, tau = unpack_x(x)
            q0 = system.q0
            u0 = system.u0
            q1 = q0 + dt * system.q_dot(t0, q0, u0)
            qk = np.concatenate([q1.reshape(1, system.nq), qk1[:-1]])
            uk = np.concatenate([u0.reshape(1, system.nu), uk1[:-1]])

            system.set_tau(interp1d(tk1, tau, axis=0, fill_value="extrapolate"))

            R_q = np.array(
                [
                    qk1_ - qk_ - dt * system.q_dot(t_, qk1_, uk1_)
                    for t_, qk_, qk1_, uk1_ in zip(tk1, qk, qk1, uk1)
                ]
            )
            R_u = np.array(
                [
                    system.M(t_, qk1_) @ (uk1_ - uk_)
                    - system.h(t_, qk1_, uk1_)
                    - system.W_g(t_, qk1_) @ la_g_
                    - system.W_tau(t_, qk1_) @ system.tau(t_)
                    for t_, qk1_, uk_, uk1_, la_g_ in zip(tk1, qk1, uk, uk1, la_g)
                ]
            )
            R_g = np.array([system.g(t_, qk1_) for t_, qk1_ in zip(tk1, qk1)])

            R = np.concatenate([R_q, R_u, R_g], axis=1)
            return R.flatten(order="C")

        def cost(x):
            qk1, uk1, la_g, tau = unpack_x(x)
            k1 = 1
            k2 = 1
            return 0  # k1 * np.sum(tau) + k2 * joint.angle(t[-1], qk1[-1, joint.qDOF])

        opt_sol = minimize(cost, x0, constraints={"type": "eq", "fun": constraints})
        if opt_sol.success:
            x_opt = opt_sol.x
        else:
            ValueError("optimization ")

    if feed_forward == "inverse_dynamics":
        ####################
        # inverse kinematics
        ####################
        A_IK = np.array([A_IK_basic(phi_).z() for phi_ in phi])
        r_OS = np.array([A_IK_ @ K_r_OS for A_IK_ in A_IK])
        K_Omega = np.array([[0, 0, phi_dot_] for phi_dot_ in phi_dot])
        v_S = np.array(
            [cross3(K_Omega_, r_OS_) for K_Omega_, r_OS_ in zip(K_Omega, r_OS[1:])]
        )  # I_Omega = K_Omega!!

        q = np.array(
            [RigidBody.pose2q(r_OS_, A_IK_) for r_OS_, A_IK_ in zip(r_OS, A_IK)]
        )
        u = np.concatenate([v_S, K_Omega], axis=1)
        u_dot = (u[1:] - u[:-1]) / dt

        # fig, ax = plt.subplots()
        # ax.plot(t[2:], u_dot)
        # plt.show()

        ##################
        # inverse dynamics
        ##################
        # M @ u_dot - h = W_tau @ la_tau + W_g @ la_g
        la = np.array(
            [
                np.linalg.pinv(
                    np.concatenate(
                        (system.W_tau(t_, q_).toarray(), system.W_g(t_, q_).toarray()),
                        axis=1,
                    )
                )
                @ (system.M(t_, q_) @ u_dot_ - system.h(t_, q_, u_))
                for t_, q_, u_, u_dot_ in zip(t[2:], q[2:], u[1:], u_dot)
            ]
        )
        la_tau = la[:, 0]
        la_g = la[:, 1:]

        # fig, ax = plt.subplots()
        # ax.plot(t[2:], la_tau)
        # plt.show()

        # add motor moment as feedforward
        la_tau_interp = interp1d(t[2:], la_tau, axis=0, fill_value="extrapolate")
        motor.tau = lambda t: la_tau_interp(t) if t <= 1 else 0

    # add PD controller as feedback
    phi_interp = interp1d(t, phi, axis=0, fill_value="extrapolate")
    phi_dot_interp = interp1d(t[1:], phi_dot, axis=0, fill_value="extrapolate")
    angle_des = lambda t: np.array([phi_interp(t), phi_dot_interp(t)])
    kp = 50
    ki = 100
    kd = 10
    if feed_back == "PD":
        controller = PDcontroller(RotationalTransmission)(
            kp, kd, angle_des, subsystem=joint
        )
        system.add(controller)
    elif feed_back == "PID":
        controller = PIDcontroller(RotationalTransmission)(
            kp, ki, kd, angle_des, subsystem=joint
        )
        system.add(controller)

    system.assemble()

    t1 = 4
    dt = 1e-3
    joint.reset()
    sol = Moreau(system, t1, dt).solve()

    joint.reset()
    angle = []
    for ti, qi in zip(sol.t, sol.q[:, :7]):
        angle.append(joint.angle(ti, qi))

    fig, ax = plt.subplots()
    ax.plot(sol.t, angle)
    plt.show()

    ###########
    # animation
    ###########
    t, q = sol.t, sol.q[:, :7]

    fig, ax = plt.subplots()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    width = 1.5 * l
    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)
    ax.axis("equal")

    # prepare data for animation
    frames = len(t)
    target_frames = min(len(t), 200)
    frac = int(frames / target_frames)
    animation_time = 5
    interval = animation_time * 1000 / target_frames

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    (line,) = ax.plot([], [], "-ok")
    angles = np.linspace(0, 2 * np.pi, num=100, endpoint=True)
    ax.plot(np.cos(angles), np.sin(angles), "--k")

    def update(t, q, line):
        r_OS = pendulum.r_OP(t, q)
        line.set_data([0, r_OS[0]], [0, r_OS[1]])
        return (line,)

    def animate(i):
        update(t[i], q[i], line)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()
