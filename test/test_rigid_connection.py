import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from cardillo.math import A_IK_basic, axis_angle2quat, cross3
from cardillo import System
from cardillo.discrete import (
    Frame,
    RigidBodyQuaternion,
)
from cardillo.constraints import RevoluteJoint, RigidConnection
from cardillo.forces import Force
from cardillo.solver import EulerBackward, ScipyIVP


if __name__ == "__main__":
    # parameters
    m = 1
    L = 2
    theta = 1 / 12 * m * (L**2)
    theta_O = theta + m * (L**2) / 4
    theta1 = theta2 = 1 / 12 * m / 2 * (L**2) / 4
    K_theta_S = theta * np.eye(3)
    K_theta_S1 = K_theta_S2 = theta1 * np.eye(3)
    g = 9.81
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
    RB1 = RigidBodyQuaternion(m / 2, K_theta_S1, q0=q10, u0=u10)
    RB2 = RigidBodyQuaternion(m / 2, K_theta_S2, q0=q20, u0=u20)

    system.add(RB1)
    system.add(RB2)
    gravity1 = Force(np.array([0, -m / 2 * g, 0]), RB1)
    system.add(gravity1)
    gravity2 = Force(np.array([0, -m / 2 * g, 0]), RB2)
    system.add(gravity2)
    system.add(RevoluteJoint(frame, RB1, r_OP(0), np.eye(3)))
    system.add(RigidConnection(RB1, RB2, r_OS0))
    system.assemble()

    t0 = 0
    t1 = 2
    dt = 1e-2

    # solver = EulerBackward(system, t1, dt, method="index 1")
    # solver = EulerBackward(system, t1, dt, method="index 2")
    # solver = EulerBackward(system, t1, dt, method="index 3")
    # solver = EulerBackward(system, t1, dt, method="index 2 GGL")
    solver = ScipyIVP(system, t1, dt)

    sol = solver.solve()
    t = sol.t
    q = sol.q
    u = sol.u

    fig, ax = plt.subplots(2, 1)

    x_ = []
    y_ = []
    r_OS = np.zeros((3, len(q[:, 0])))
    r_OS1 = np.zeros((3, len(q[:, 0])))
    for i, ti in enumerate(t):
        r_OS = q[i, :3] - (RB1.A_IK(ti, q[i, :7]) @ K_r_SP1)
        x_.append(r_OS[0])
        y_.append(r_OS[1])

    ax[0].plot(t, x_, "--gx")
    ax[1].plot(t, y_, "--gx")

    # reference solution
    def eqm(t, x):
        dx = np.zeros(2)
        dx[0] = x[1]
        dx[1] = -0.5 * m * L * (e_tt(t) * np.cos(x[0]) + g * np.sin(x[0])) / theta_O
        return dx

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
    x = ref.y
    t = ref.t

    # plot reference solution
    ax[0].plot(t, e(t) + L / 2 * np.sin(x[0]), "-r")
    ax[1].plot(t, -L / 2 * np.cos(x[0]), "-r")

    plt.show()
