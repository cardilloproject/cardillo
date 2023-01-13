import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math import A_IK_basic
from cardillo import System
from cardillo.discrete import Frame, RigidBodyRelKinematics
from cardillo.joints import RevoluteJoint, LinearGuidance
from cardillo.forces import Force, ScalarForceTranslational, LinearSpring
from cardillo.solver import EulerBackward, ScipyIVP

# As a benchmark, the two-arm robot from DMS is used but a spring is used instead of the linear motor.
if __name__ == "__main__":
    # parameters
    m1 = 1
    theta1 = 1
    K_Theta_S1 =theta1 * np.eye(3)
    m2 = 1
    theta2 = 1
    K_Theta_S2 = theta2 * np.eye(3)
    a1 = 1
    a2 = 1
    g = 10

    stiffness = 25

    # initial conditions
    phi0 = 0
    phi_dot0 = 0

    x0 = 2 * a1
    x_dot0 = 0

    # system definition

    system = System()

    origin = Frame()
    system.add(origin)

    revolute_joint = RevoluteJoint(np.zeros(3), np.eye(3), q0=np.array([phi0]), u0=np.array([phi_dot0]))
    r_OS10 = a1 * np.array([np.cos(phi0), np.sin(phi0), 0])
    RB1 = RigidBodyRelKinematics(
        m1, K_Theta_S1, revolute_joint, origin, r_OS0=r_OS10, A_IK0=A_IK_basic(phi0).z()
    )
    system.add(revolute_joint, RB1)

    lin_guidance = LinearGuidance(np.zeros(3), A_IK_basic(phi0).z(), q0=np.array([x0]), u0=np.array([x_dot0]))
    r_OS10 = x0 * np.array([np.cos(phi0), np.sin(phi0), 0])
    RB2 = RigidBodyRelKinematics(
        m2, K_Theta_S2, lin_guidance, RB1, r_OS0=r_OS10, A_IK0=A_IK_basic(phi0).z()
    )
    system.add(lin_guidance, RB2)

    spring = ScalarForceTranslational(RB1, RB2, LinearSpring(stiffness), None)
    system.add(spring)

    
    gravity1 = Force(np.array([0, -m1 * g, 0]), RB1)
    system.add(gravity1)
    gravity2 = Force(np.array([0, -m2 * g, 0]), RB2)
    system.add(gravity2)
    system.assemble()

    ############################################################################
    #                   solver
    ############################################################################
    t0 = 0
    t1 = 4
    dt = 1e-2

    # solver = EulerBackward(system, t1, dt)
    solver = ScipyIVP(system, t1, dt)

    sol = solver.solve()
    t = sol.t
    q = sol.q
    u = sol.u

    ############################################################################
    #                   compare with reference solution
    ############################################################################
    def eqm(t, y):
        phi, x, phi_dot, x_dot = y
        dy = np.zeros(4)
        dy[0] = phi_dot
        dy[1] = x_dot
        dy[2] = - ( 2 * m2 * x * x_dot * phi_dot + g * np.cos(phi) * (a1 * m1 + x * m2)) / (m1 * a1**2 + m2 * x**2 + theta1 + theta2)
        dy[3] = - stiffness * (x - x0) / m2 + x * phi_dot**2 - g * np.sin(phi)
        return dy


    y0 = np.array([phi0, x0, phi_dot0, x_dot0])
    ref = solve_ivp(
        eqm,
        [t0, t1],
        y0,
        method="RK45",
        t_eval=np.arange(t0, t1 + dt, dt),
        rtol=1e-6,
        atol=1e-8,
    )
    y_ref = ref.y
    t_ref = ref.t

    plt.plot(t, q, 'x')
    plt.plot(t_ref, y_ref[:2].T)
    plt.show()
    
    ############################################################################
    #                   animation
    ############################################################################
    fig, ax = plt.subplots()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    scale = 2 * (x0 + a2)
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)

    def init(t, q):
        x_0, y_0, z_0 = np.zeros(3)
        x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])
        x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF])
        x_A, y_A, z_A = RB2.r_OP(t, q[RB2.qDOF], K_r_SP=np.array([a1, 0, 0]))

        A_IK1 = RB1.A_IK(t, q[RB1.qDOF])
        d11 = A_IK1[:, 0]
        d21 = A_IK1[:, 1]
        d31 = A_IK1[:, 2]

        A_IK2 = RB2.A_IK(t, q[RB2.qDOF])
        d12 = A_IK2[:, 0]
        d22 = A_IK2[:, 1]
        d32 = A_IK2[:, 2]

        (COM,) = ax.plot([x_0, x_S1, x_S2, x_A], [y_0, y_S1, y_S2, y_A], "-ok")
        (d11_,) = ax.plot(
            [x_S1, x_S1 + d11[0]],
            [y_S1, y_S1 + d11[1]],
            "-r",
        )
        (d21_,) = ax.plot(
            [x_S1, x_S1 + d21[0]],
            [y_S1, y_S1 + d21[1]],
            "-g",
        )
        (d31_,) = ax.plot(
            [x_S1, x_S1 + d31[0]],
            [y_S1, y_S1 + d31[1]],
            "-b",
        )
        (d12_,) = ax.plot(
            [x_S2, x_S2 + d12[0]],
            [y_S2, y_S2 + d12[1]],
            "-r",
        )
        (d22_,) = ax.plot(
            [x_S2, x_S2 + d22[0]],
            [y_S2, y_S2 + d22[1]],
            "-g",
        )
        (d32_,) = ax.plot(
            [x_S2, x_S2 + d32[0]],
            [y_S2, y_S2 + d32[1]],
            "-b",
        )

        return COM, d11_, d21_, d31_, d12_, d22_, d32_
    
    def update(t, q, COM, d11_, d21_, d31_, d12_, d22_, d32_):
        x_0, y_0, z_0 = np.zeros(3)
        x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])
        x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF])
        x_A, y_A, z_A = RB2.r_OP(t, q[RB2.qDOF], K_r_SP=np.array([a1, 0, 0]))

        A_IK1 = RB1.A_IK(t, q[RB1.qDOF])
        d11 = A_IK1[:, 0]
        d21 = A_IK1[:, 1]
        d31 = A_IK1[:, 2]

        A_IK2 = RB2.A_IK(t, q[RB2.qDOF])
        d12 = A_IK2[:, 0]
        d22 = A_IK2[:, 1]
        d32 = A_IK2[:, 2]

        COM.set_data([x_0, x_S1, x_S2, x_A], [y_0, y_S1, y_S2, y_A])

        d11_.set_data([x_S1, x_S1 + d11[0]], [y_S1, y_S1 + d11[1]])

        d21_.set_data([x_S1, x_S1 + d21[0]], [y_S1, y_S1 + d21[1]])

        d31_.set_data([x_S1, x_S1 + d31[0]], [y_S1, y_S1 + d31[1]])

        d12_.set_data([x_S2, x_S2 + d12[0]], [y_S2, y_S2 + d12[1]])

        d22_.set_data([x_S2, x_S2 + d22[0]], [y_S2, y_S2 + d22[1]])

        d32_.set_data([x_S2, x_S2 + d32[0]], [y_S2, y_S2 + d32[1]])

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
    plt.show()
   