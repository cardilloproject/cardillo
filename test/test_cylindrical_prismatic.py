import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

from cardillo import System
from cardillo.solver import ScipyIVP, EulerBackward, RadauIIa
from cardillo.constraints import RigidConnection, Cylindrical, Prismatic
from cardillo.discrete import Frame, RigidBodyQuaternion
from cardillo.forces import Force
from cardillo.math import (
    e1,
    Exp_SO3,
    T_SO3,
    T_SO3_dot,
    Log_SO3,
    Spurrier,
    cross3,
    ax2skew,
)

# setup solver
t_span = (0.0, 2.0)
# t_span = (0.0, 0.1)
t0, t1 = t_span
dt = 1.0e-3

# parameters
# g = 9.81
g = 0
l = 10
m = 1
r = 0.2
C = 1 / 2 * m * r**2
A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
K_theta_S = np.diag(np.array([C, A, A]))


def RigidCylinder(RigidBody):
    class _RigidCylinder(RigidBody):
        def __init__(self, q0=None, u0=None):
            super().__init__(m, K_theta_S, q0=q0, u0=u0)

    return _RigidCylinder


def run_old(
    joint,
    Solver,
    **solver_kwargs,
):
    # axis origin
    r_OB0 = np.zeros(3)
    # r_OB0 = np.random.rand(3)

    # axis orientation
    # psi0 = np.zeros(3)
    # psi0 = np.pi / 2 * e1
    # psi0 = 1.25 * np.pi / 2 * e1
    # psi0 = np.pi / 4 * e1
    psi0 = np.random.rand(3)
    A_IB0 = Exp_SO3(psi0)

    e_xB, e_yB, e_zB = A_IB0.T

    z0 = 0
    z_dot0 = 10

    alpha0 = 0
    if joint == "Cylindrical":
        alpha_dot0 = 2
    elif joint == "Prismatic":
        alpha_dot0 = 0  # has to be zero for prismatic example
    else:
        raise NotImplementedError

    ##############
    # DAE solution
    ##############
    frame = Frame(r_OP=r_OB0, A_IK=A_IB0)

    q0 = np.array([*r_OB0, *Spurrier(A_IB0)])
    RB1 = RigidBodyQuaternion(m, K_theta_S, q0)

    rigid_connection = RigidConnection(frame, RB1)

    r_OS0 = r_OB0 + A_IB0 @ np.array([0.5 * l, 0, z0])
    A_IK0 = A_IB0
    p0 = Spurrier(A_IK0)
    q0 = np.array([*r_OS0, *p0])
    K0_omega_IK0 = np.array([0, 0, alpha_dot0])
    v_P0 = A_IB0 @ np.array([0, 0, z_dot0])
    v_S0 = v_P0 + A_IK0 @ cross3(K0_omega_IK0, np.array([0.5 * l, 0, 0]))
    u0 = np.array([*v_S0, *K0_omega_IK0])
    RB2 = RigidBodyQuaternion(m, K_theta_S, q0, u0)

    f_g = Force(np.array([0, 0, -m * g]), RB2)

    if joint == "Cylindrical":
        constraint = Cylindrical(
            subsystem1=RB1,
            subsystem2=RB2,
            axis=2,
        )
    elif joint == "Prismatic":
        constraint = Prismatic(
            subsystem1=RB1,
            subsystem2=RB2,
            axis=2,
        )

    system = System()
    system.add(frame, RB1, rigid_connection)
    system.add(RB2, f_g)
    system.add(constraint)
    system.assemble()

    sol = Solver(system, t1, dt, **solver_kwargs).solve()

    t, q, u = sol.t, sol.q, sol.u

    zs = np.array([A_IB0[:, 2] @ RB2.r_OP(ti, qi[RB2.qDOF]) for (ti, qi) in zip(t, q)])
    A_IK = np.array([RB2.A_IK(ti, qi[RB2.qDOF]) for (ti, qi) in zip(t, q)])
    Delta_alpha = np.zeros(len(t), dtype=float)
    Delta_alpha[0] = alpha0
    for i in range(1, len(t)):
        Delta_alpha[i] = Log_SO3(A_IK[i - 1].T @ A_IK[i])[-1]
    alphas = np.cumsum(Delta_alpha)

    ##############
    # ODE solution
    ##############
    theta = A

    def eqm(t, y):
        z, alpha = y[:2]

        dy = np.zeros_like(y)

        # trivial kinematic equation
        dy[:2] = y[2:]

        # equations of motion
        dy[2] = -g * e_zB[-1]

        if joint == "Cylindrical":
            dy[3] = (
                -0.5
                * m
                * g
                * l
                * (np.cos(alpha) * e_yB[-1] - np.sin(alpha) * e_xB[-1])
                / (m * l**2 / 4 + theta)
            )
        else:
            # note: this is a very lazy solution
            dy[3] = 0

        return dy

    y0 = np.array([z0, alpha0, z_dot0, alpha_dot0], dtype=float)
    t_eval = np.arange(t0, t1, dt)
    sol = solve_ivp(eqm, t_span, y0, t_eval=t_eval)

    t_ref, y_ref = sol.t, sol.y

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].set_title("z")
    ax[0, 0].plot(t_ref, y_ref[0], "-k", label="reference")
    ax[0, 0].plot(t, zs, "--r", label="DAE")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].set_title("alpha")
    ax[0, 1].plot(t_ref, y_ref[1], "-k", label="reference")
    ax[0, 1].plot(t, alphas, "--r", label="DAE")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 0].set_title("z_dot")
    ax[1, 0].plot(t_ref, y_ref[2], "-k", label="reference")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[1, 1].set_title("alpha_dot")
    ax[1, 1].plot(t_ref, y_ref[3], "-k", label="reference")
    ax[1, 1].grid()
    ax[1, 1].legend()

    # plt.show()

    # animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    z1 = max(l, y_ref[0, -1])
    diff = abs(z1 - z0)
    scale = diff
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=-scale, top=scale)

    ax.plot(*np.array([r_OB0 - e_zB * diff, r_OB0, r_OB0 + e_zB * diff]).T, "-ok")
    (rod,) = ax.plot([], [], [], "-ob")

    def animate(i):
        z, alpha = y_ref[:2, i]
        r_OP = r_OB0 + z * e_zB
        r_OS = r_OP + 0.5 * l * (np.cos(alpha) * e_xB + np.sin(alpha) * e_yB)
        r_OQ = r_OP + l * (np.cos(alpha) * e_xB + np.sin(alpha) * e_yB)
        x, y, z = np.array([r_OP, r_OS, r_OQ]).T

        rod.set_xdata(x)
        rod.set_ydata(y)
        rod.set_3d_properties(z)

    frames = len(t_ref)
    interval = dt * 1000
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()


def run(
    joint,
    Solver,
    **solver_kwargs,
):
    # e = lambda t: 0
    # e_t = lambda t: 0
    # e_tt = lambda t: 0

    # e = lambda t: t**2
    # e_t = lambda t: 2 * t
    # e_tt = lambda t: 2

    amplitude = l / 10
    omega = np.pi * 2
    e = lambda t: amplitude * np.cos(omega * t)
    e_t = lambda t: -amplitude * omega * np.sin(omega * t)
    e_tt = lambda t: -amplitude * omega * omega * np.cos(omega * t)

    # axis origin
    n1 = np.random.rand(3)
    r_OB = lambda t: e(t) * n1
    r_OB_t = lambda t: e_t(t) * n1
    r_OB_tt = lambda t: e_tt(t) * n1

    r_OB0 = r_OB(0)

    # # axis orientation
    # psi0 = np.random.rand(3)
    # A_IB0 = Exp_SO3(psi0)

    # phi = lambda t: np.cos(omega * t)
    # phi_t = lambda t: -omega * np.sin(omega * t)
    # phi_tt = lambda t: -omega * omega * np.cos(omega * t)

    # def A_B0B(t):
    #     _phi = phi(t)
    #     sp = np.sin(_phi)
    #     cp = np.cos(_phi)
    #     return np.array(
    #         [
    #             [cp, -sp, 0],
    #             [sp, cp, 0],
    #             [0, 0, 1],
    #         ],
    #     )

    # def A_B0B_t(t):
    #     _phi = phi(t)
    #     _phi_t = phi_t(t)
    #     sp = np.sin(_phi)
    #     cp = np.cos(_phi)
    #     return _phi_t * np.array(
    #         [
    #             [-sp, -cp, 0],
    #             [cp, -sp, 0],
    #             [0, 0, 0],
    #         ],
    #     )

    # def A_B0B_tt(t):
    #     _phi = phi(t)
    #     _phi_t = phi_t(t)
    #     _phi_tt = phi_tt(t)
    #     sp = np.sin(_phi)
    #     cp = np.cos(_phi)
    #     return _phi_tt * np.array(
    #         [
    #             [-sp, -cp, 0],
    #             [cp, -sp, 0],
    #             [0, 0, 0],
    #         ],
    #     ) + _phi_t**2 * np.array(
    #         [
    #             [-cp, sp, 0],
    #             [-sp, -cp, 0],
    #             [0, 0, 0],
    #         ],
    #     )

    # A_IB = lambda t: A_IB0 @ A_B0B(t)
    # A_IB_t = lambda t: A_IB0 @ A_B0B_t(t)
    # A_IB_tt = lambda t: A_IB0 @ A_B0B_tt(t)

    # omega_IB = lambda t: A_IB0 @ np.array([0, 0, phi_t(t)])
    # omega_IB_t = lambda t: A_IB0 @ np.array([0, 0, phi_tt(t)])
    # # B_omega_IB = lambda t: A_IB(t).T @ omega_IB(t)
    # # B_omega_IB_t = lambda t: A_IB(t).T @ omega_IB_t(t)

    n2 = np.random.rand(3)
    # n2 = np.array([0, 0, 1])
    # n2 = np.array([0, 1, 0])

    # e2 = lambda t: amplitude * np.cos(omega * t)
    # e2_t = lambda t: -amplitude * omega * np.sin(omega * t)
    # e2_tt = lambda t: -amplitude * omega * omega * np.cos(omega * t)
    # psi = lambda t: e2(t) * n2
    # psi_t = lambda t: e2_t(t) * n2
    # psi_tt = lambda t: e2_tt(t) * n2

    # psi = lambda t: n2
    # psi_t = lambda t: 0 * n2
    # psi_tt = lambda t: 0 * n2

    # psi = lambda t: t * n2
    # psi_t = lambda t: n2
    # psi_tt = lambda t: 0 * n2

    # psi = lambda t: t**2 * n2
    # psi_t = lambda t: 2 * t * n2
    # psi_tt = lambda t: 2 * n2

    psi = lambda t: t**3 * n2
    psi_t = lambda t: 3 * t**2 * n2
    psi_tt = lambda t: 6 * t * n2

    # B_omega_IB = lambda t: T_SO3(psi(t)) @ psi_t(t)
    # # from cardillo.math import T_SO3_psi
    # # def T_SO3_dot(psi, psi_dot):
    # #     return np.einsum("ijk,k->ij", T_SO3_psi(psi), psi_dot)

    # # def tmp(psi, psi_dot):
    # #     return np.einsum("ijk,j,k->i", T_SO3_psi(psi), psi_dot, psi_dot)

    # # B_omega_IB_t = lambda t: T_SO3(psi(t)) @ psi_tt(t) + T_SO3_dot(
    # #     psi(t), psi_t(t)
    # # ) @ psi_t(t)
    # from cardillo.math import T_SO3_psi
    # B_omega_IB_t = lambda t: T_SO3(psi(t)) @ psi_tt(t) + np.einsum(
    #     "ijk,j,k->i", T_SO3_psi(psi(t)), psi_t(t), psi_t(t)
    # )

    from cardillo.math import T_SO3_psi

    omega_IB = lambda t: T_SO3(psi(t)).T @ psi_t(t)
    omega_IB_t = lambda t: T_SO3(psi(t)).T @ psi_tt(t) + np.einsum(
        "jik,j,k->i", T_SO3_psi(psi(t)), psi_t(t), psi_t(t)
    )
    # omega_IB = lambda t: A_IB(t) @ B_omega_IB(t)
    # omega_IB_t = lambda t: A_IB(t) @ B_omega_IB_t(t)

    A_IB = lambda t: Exp_SO3(psi(t))
    A_IB_t = lambda t: ax2skew(omega_IB(t)) @ A_IB(t)
    A_IB_tt = lambda t: ax2skew(omega_IB_t(t)) @ A_IB(t) + ax2skew(
        omega_IB(t)
    ) @ A_IB_t(t)

    A_IB0 = A_IB(0)

    z0 = 0
    z_dot0 = 1

    alpha0 = 0
    if joint == "Cylindrical":
        alpha_dot0 = 2
    elif joint == "Prismatic":
        alpha_dot0 = 0  # has to be zero for prismatic example
    else:
        raise NotImplementedError

    ##############
    # DAE solution
    ##############
    frame = Frame(
        r_OP=r_OB,
        r_OP_t=r_OB_t,
        r_OP_tt=r_OB_tt,
        A_IK=A_IB,
        A_IK_t=A_IB_t,
        A_IK_tt=A_IB_tt,
    )

    # q0 = np.array([*r_OB0, *Spurrier(A_IB0)])
    # RB1 = RigidBodyQuaternion(m, K_theta_S, q0)

    # rigid_connection = RigidConnection(frame, RB1)

    A_IK0 = A_IB0
    r_OS0 = r_OB0 + A_IK0 @ np.array([0.5 * l, 0, z0])
    p0 = Spurrier(A_IK0)
    q0 = np.array([*r_OS0, *p0])

    K0_omega_IK0 = A_IK0.T @ omega_IB(0) + np.array([0, 0, alpha_dot0])
    v_P0 = (
        r_OB_t(0) + A_IB0 @ np.array([0, 0, z_dot0]) + A_IB_t(0) @ np.array([0, 0, z0])
    )
    v_S0 = v_P0 + A_IK0 @ cross3(K0_omega_IK0, np.array([0.5 * l, 0, 0]))

    u0 = np.array([*v_S0, *K0_omega_IK0])
    RB2 = RigidBodyQuaternion(m, K_theta_S, q0, u0)

    f_g = Force(np.array([0, 0, -m * g]), RB2)

    if joint == "Cylindrical":
        constraint = Cylindrical(
            # subsystem1=RB1,
            subsystem1=frame,
            subsystem2=RB2,
            axis=2,
        )
    elif joint == "Prismatic":
        constraint = Prismatic(
            # subsystem1=RB1,
            subsystem1=frame,
            subsystem2=RB2,
            axis=2,
        )

    system = System()
    # system.add(frame, RB1, rigid_connection)
    system.add(RB2, f_g)
    # system.add(constraint)
    system.add(frame, constraint)
    system.assemble()

    sol = Solver(system, t1, dt, **solver_kwargs).solve()

    t, q, u = sol.t, sol.q, sol.u

    zs = np.array(
        [
            A_IB(ti)[:, 2] @ (RB2.r_OP(ti, qi[RB2.qDOF]) - r_OB(ti))
            for (ti, qi) in zip(t, q)
        ]
    )

    A_BK = np.array([A_IB(ti).T @ RB2.A_IK(ti, qi[RB2.qDOF]) for (ti, qi) in zip(t, q)])
    # Delta_alpha = np.zeros(len(t), dtype=float)
    # Delta_alpha[0] = alpha0
    # for i in range(1, len(t)):
    #     Delta_psi = Log_SO3(A_BK[i - 1].T @ A_BK[i])
    #     # Delta_psi = Log_SO3(A_BK[i])
    #     # print(f"Delta_psi: {Delta_psi}")
    #     # Delta_alpha[i] = Delta_psi[-1]
    #     Delta_alpha[i] = Log_SO3(A_BK[i])[-1]
    # alphas = np.cumsum(Delta_alpha)
    # TODO: Remove 2 pi jumps
    alphas = np.array([Log_SO3(A_BKi)[-1] for A_BKi in A_BK])
    for i in range(1, len(alphas)):
        diff = alphas[i] - alphas[i - 1]
        if diff > np.pi:
            alphas[i:] += diff
        elif diff < -np.pi:
            alphas[i:] -= diff

    ##############
    # ODE solution
    ##############
    def eqm(t, y):
        z, alpha = y[:2]
        z_dot, alpha_dot = y[2:]

        sa = np.sin(alpha)
        ca = np.cos(alpha)

        e1, e2, e3 = np.eye(3)

        _A_IB = A_IB(t)
        e_x_B, e_y_B, e_z_B = _A_IB.T

        A_BK = np.array(
            [
                [ca, -sa, 0],
                [sa, ca, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        # A_KB = A_BK.T
        A_KI = A_BK.T @ _A_IB.T

        J_S = np.zeros((3, 2), dtype=float)
        J_S[:, 0] = e_z_B
        J_S[:, 1] = 0.5 * l * (ca * e_y_B - sa * e_x_B)

        K_J_R = np.zeros((3, 2), dtype=float)
        K_J_R[:, 1] = e3

        M = m * (J_S.T @ J_S) + K_J_R.T @ K_theta_S @ K_J_R

        _omega_IB = omega_IB(t)

        B_r_BS = z * e3 + 0.5 * l * (ca * e1 + sa * e2)
        r_BS = _A_IB @ B_r_BS
        B_r_BS_dot = z_dot * e3 + 0.5 * l * alpha_dot * (ca * e2 - sa * e1)

        nu_S_dot = (
            r_OB_tt(t)
            + 2 * cross3(_omega_IB, _A_IB @ B_r_BS_dot)
            - _A_IB @ (0.5 * l * alpha_dot**2 * (ca * e1 + sa * e2))
            + cross3(omega_IB_t(t), r_BS)
            + cross3(_omega_IB, cross3(_omega_IB, r_BS))
        )

        # K_nu_R_dot = A_KB @ B_omega_IB_t(t) + alpha_dot * cross3(B_omega_IB(t), e3)
        K_nu_R_dot = A_KI @ omega_IB_t(t) + alpha_dot * cross3(
            _A_IB.T @ omega_IB(t), e3
        )

        h = -J_S.T @ (m * nu_S_dot + m * g * e3) - K_J_R.T @ K_theta_S @ K_nu_R_dot

        u_dot = np.linalg.solve(M, h)

        dy = np.zeros_like(y)
        if joint == "Cylindrical":
            dy[:2] = y[2:]
            dy[2:] = u_dot
        else:
            dy[0] = y[2]
            dy[2] = u_dot[0]

        return dy

    y0 = np.array([z0, alpha0, z_dot0, alpha_dot0], dtype=float)
    t_eval = np.arange(t0, t1, dt)
    sol_ref = solve_ivp(
        eqm, t_span, y0, t_eval=t_eval, rtol=1e-10, atol=1e-10, max_step=dt
    )

    t_ref, y_ref = sol_ref.t, sol_ref.y

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].set_title("z")
    ax[0, 0].plot(t_ref, y_ref[0], "-k", label="reference")
    ax[0, 0].plot(t, zs, "--r", label="DAE")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].set_title("alpha")
    ax[0, 1].plot(t_ref, y_ref[1], "-k", label="reference")
    ax[0, 1].plot(t, alphas, "--r", label="DAE")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 0].set_title("z_dot")
    ax[1, 0].plot(t_ref, y_ref[2], "-k", label="reference")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[1, 1].set_title("alpha_dot")
    ax[1, 1].plot(t_ref, y_ref[3], "-k", label="reference")
    ax[1, 1].grid()
    ax[1, 1].legend()

    # animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    z_min = min(-l, min(y_ref[0]))
    z_max = max(l, max(y_ref[0]))
    diff = z_max - z_min
    scale = diff
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=-scale, top=scale)

    (axis,) = ax.plot([], [], [], "-ok")
    (rod,) = ax.plot([], [], [], "-ob")

    def animate(i):
        ti = t_ref[i]
        z, alpha = y_ref[:2, i]

        _r_OB = r_OB(ti)

        _A_IB = A_IB(ti)
        e_x_B, e_y_B, e_z_B = _A_IB.T

        r_OP = _r_OB + z * e_z_B
        r_OS = r_OP + 0.5 * l * (np.cos(alpha) * e_x_B + np.sin(alpha) * e_y_B)
        r_OQ = r_OP + l * (np.cos(alpha) * e_x_B + np.sin(alpha) * e_y_B)
        x, y, z = np.array([r_OP, r_OS, r_OQ]).T

        rod.set_xdata(x)
        rod.set_ydata(y)
        rod.set_3d_properties(z)

        r0 = _r_OB + diff * e_z_B
        r1 = _r_OB - diff * e_z_B
        x, y, z = np.array([r0, _r_OB, r1]).T

        axis.set_xdata(x)
        axis.set_ydata(y)
        axis.set_3d_properties(z)

    frames = len(t_ref)
    interval = dt * 1000
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()


if __name__ == "__main__":
    #############
    # Cylindrical
    #############
    run("Cylindrical", ScipyIVP, rtol=1e-10, atol=1e-10, max_step=dt)

    # run("Cylindrical", EulerBackward, method="index 1")
    # run("Cylindrical", EulerBackward, method="index 2")
    # run("Cylindrical", EulerBackward, method="index 3")
    # run("Cylindrical", EulerBackward, method="index 2 GGL")

    # run("Cylindrical", RadauIIa, dae_index=2, rtol=1e-6, atol=1e-6, max_step=dt)
    # run("Cylindrical", RadauIIa, dae_index=3, rtol=1e-2, atol=1e-2, max_step=dt) # this is not working
    # run("Cylindrical", RadauIIa, dae_index="GGL", rtol=1e-4, atol=1e-4, max_step=dt)

    ###########
    # Prismatic
    ###########
    # run("Prismatic", ScipyIVP)

    # run("Prismatic", EulerBackward, method="index 1")
    # run("Prismatic", EulerBackward, method="index 2")
    # run("Prismatic", EulerBackward, method="index 3")
    # run("Prismatic", EulerBackward, method="index 2 GGL")

    # run("Prismatic", RadauIIa, dae_index=2)
    # run("Prismatic", RadauIIa, dae_index=3)
    # run("Prismatic", RadauIIa, dae_index="GGL")
