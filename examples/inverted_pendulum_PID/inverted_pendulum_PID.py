import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d

from cardillo import System

from cardillo.actuators import PIDcontroller, PDcontroller
from cardillo.constraints import Revolute
from cardillo.discrete import RigidBody
from cardillo.forces import Force
from cardillo.math import A_IB_basic, cross3, smoothstep2
from cardillo.solver import ScipyIVP, ScipyDAE


if __name__ == "__main__":
    ############
    # parameters
    ############

    # control signal
    control_signal = ["Heaviside", "smoothened Heaviside"][0]

    # control strategy
    control_strategy = ["PID", "PD"][0]

    # model parameters
    l = 1  # pendulum length
    m = 1  # mass
    theta_S = m * (l**2) / 12  # inertia
    g = 10  # gravitational acceleration

    # controller gains
    kp = 30  # proportional
    ki = 0.1  # integral
    kd = 3  # differential

    # initial conditions
    t0 = 0
    phi0 = 0  # np.pi / 2
    phi_dot0 = 0

    # target angle
    phiN = np.pi

    # simulation time
    t1 = 4

    # initialize system
    system = System()

    ##########
    # pendulum
    ##########
    B_r_OC = np.array([0, -l / 2, 0])
    A_IB0 = A_IB_basic(phi0).z
    r_OC0 = A_IB0 @ B_r_OC
    B_Omega0 = np.array([0, 0, phi_dot0])
    v_C0 = cross3(B_Omega0, r_OC0)  # I_Omega0 = B_Omega0

    q0 = RigidBody.pose2q(r_OC0, A_IB0)
    u0 = np.concatenate([v_C0, B_Omega0])
    pendulum = RigidBody(m, theta_S * np.eye(3), q0=q0, u0=u0, name="pendulum")

    joint = Revolute(
        system.origin,
        pendulum,
        axis=2,
        angle0=phi0,
        A_IJ0=A_IB_basic(-np.pi / 2).z,
        name="revolute joint",
    )

    gravity = Force(np.array([0, -g * m, 0]), pendulum)

    system.add(pendulum, gravity, joint)

    ############
    # controller
    ############

    # time grid for planned trajectory
    dt = 1e-3
    t = np.arange(t0, t1, dt)

    if control_signal == "smoothened Heaviside":
        tN = 1  # swing up time
        phi = phiN * smoothstep2(t, x_min=t0, x_max=tN)
        phi_dot = (phi[1:] - phi[:-1]) / dt
    elif control_signal == "Heaviside":
        phi = phiN * np.ones_like(t)
        phi_dot = np.zeros_like(t[1:])
    else:
        raise NotImplementedError(
            f"Control signal '{control_signal}' is not implemented."
        )

    phi_interp = interp1d(t, phi, axis=0, fill_value=phiN, bounds_error=False)
    phi_dot_interp = interp1d(
        t[1:], phi_dot, axis=0, fill_value=0.0, bounds_error=False
    )
    state_des = lambda t: np.array([phi_interp(t), phi_dot_interp(t)])

    if control_strategy == "PID":
        controller = PIDcontroller(joint, kp, ki, kd, state_des)
    elif control_strategy == "PD":
        controller = PDcontroller(joint, kp, kd, state_des)
    else:
        raise NotImplementedError(
            f"Control strategy '{control_strategy}' is not implemented."
        )
    system.add(controller)

    system.assemble()

    ############
    # simulation
    ############

    dt = 1e-2
    # sol = ScipyIVP(system, t1, dt).solve()
    sol = ScipyDAE(system, t1, dt).solve()

    joint.reset()  # flush internal memory of joint angle
    angle = []
    for ti, qi in zip(sol.t, sol.q[:, joint.qDOF]):
        angle.append(joint.angle(ti, qi))

    #################
    # plot trajectory
    #################

    fig, ax = plt.subplots()
    ax.plot(sol.t, phi_interp(sol.t), "r", label="planned")
    ax.plot(sol.t, angle, "b", label="simulated")
    ax.set_title("Evolution of angle")
    ax.set_xlabel("t")
    ax.set_ylabel("$\phi$")
    ax.grid()
    ax.legend()
    plt.show()

    ###########
    # animation
    ###########
    t, q = sol.t, sol.q[:, :7]

    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
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
        r_OC = pendulum.r_OP(t, q)
        line.set_data([0, 2 * r_OC[0]], [0, 2 * r_OC[1]])
        return (line,)

    def animate(i):
        update(t[i], q[i], line)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()
