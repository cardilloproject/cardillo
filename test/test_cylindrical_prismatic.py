import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import pytest

from cardillo import System
from cardillo.solver import ScipyIVP, EulerBackward, MoreauClassical
from cardillo.constraints import RigidConnection, Cylindrical, Prismatic
from cardillo.discrete import Frame, RigidBodyQuaternion, RigidBodyAxisAngle, Cylinder
from cardillo.forces import Force, ScalarForceTranslational, LinearSpring, LinearDamper
from cardillo.math import (
    e1,
    Exp_SO3,
    T_SO3,
    T_SO3_psi,
    Log_SO3,
    Spurrier,
    cross3,
    ax2skew,
)

# solver parameters
t_span = (0.0, 5.0)
t0, t1 = t_span
dt = 1.0e-2

# parameters
k = 10
d = 10
g = 9.81
l = 10
m = 1
r = 0.2
C = 1 / 2 * m * r**2
A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
K_theta_S = np.diag(np.array([C, A, A]))

show = False


def run(
    joint: str,
    RigidBody,
    Solver,
    **solver_kwargs,
):
    ############################################################################
    #                   system setup
    ############################################################################

    #############
    # origin axis
    #############
    amplitude = l / 10
    omega = np.pi * 2
    e = lambda t: amplitude * np.cos(omega * t)
    e_t = lambda t: -amplitude * omega * np.sin(omega * t)
    e_tt = lambda t: -amplitude * omega * omega * np.cos(omega * t)

    n1 = np.random.rand(3)
    r_OB = lambda t: e(t) * n1
    r_OB_t = lambda t: e_t(t) * n1
    r_OB_tt = lambda t: e_tt(t) * n1

    r_OB0 = r_OB(0)

    n2 = np.random.rand(3)

    ##################
    # orientation axis
    ##################
    amplitude = 1
    omega = 0.5
    e2 = lambda t: amplitude * np.cos(omega * t)
    e2_t = lambda t: -amplitude * omega * np.sin(omega * t)
    e2_tt = lambda t: -amplitude * omega * omega * np.cos(omega * t)
    psi = lambda t: e2(t) * n2
    psi_t = lambda t: e2_t(t) * n2
    psi_tt = lambda t: e2_tt(t) * n2

    omega_IB = lambda t: T_SO3(psi(t)).T @ psi_t(t)
    omega_IB_t = lambda t: T_SO3(psi(t)).T @ psi_tt(t) + np.einsum(
        "jik,j,k->i", T_SO3_psi(psi(t)), psi_t(t), psi_t(t)
    )

    A_IB = lambda t: Exp_SO3(psi(t))
    A_IB_t = lambda t: ax2skew(omega_IB(t)) @ A_IB(t)
    A_IB_tt = lambda t: ax2skew(omega_IB_t(t)) @ A_IB(t) + ax2skew(
        omega_IB(t)
    ) @ A_IB_t(t)

    A_IB0 = A_IB(0)

    ##########################
    # other initial conditions
    ##########################
    z0 = 0
    z_dot0 = 0

    alpha0 = 0
    if joint == "Cylindrical":
        alpha_dot0 = 2
    elif joint == "Prismatic":
        alpha_dot0 = 0  # has to be zero for prismatic example
    else:
        raise NotImplementedError

    ######################
    # define contributions
    ######################
    system = System()

    frame = Frame(
        r_OP=r_OB,
        r_OP_t=r_OB_t,
        r_OP_tt=r_OB_tt,
        A_IK=A_IB,
        A_IK_t=A_IB_t,
        A_IK_tt=A_IB_tt,
    )

    B0_omega_IB0 = A_IB0.T @ omega_IB(0)
    v_P0 = (
        r_OB_t(0) + A_IB0 @ np.array([0, 0, z_dot0]) + A_IB_t(0) @ np.array([0, 0, z0])
    )

    u0 = np.array([*v_P0, *B0_omega_IB0])

    if RigidBody == RigidBodyAxisAngle:
        q0 = np.array([*r_OB0, *Log_SO3(A_IB0)])
    elif RigidBody == RigidBodyQuaternion:
        q0 = np.array([*r_OB0, *Spurrier(A_IB0)])
    else:
        raise NotImplementedError
    RB1 = RigidBody(m, K_theta_S, q0, u0)

    rigid_connection = RigidConnection(frame, RB1)

    A_IK0 = A_IB0
    r_OS0 = r_OB0 + A_IK0 @ np.array([0.5 * l, 0, z0])

    K0_omega_IK0 = A_IK0.T @ omega_IB(0) + np.array([0, 0, alpha_dot0])
    v_S0 = v_P0 + A_IK0 @ cross3(K0_omega_IK0, np.array([0.5 * l, 0, 0]))

    u0 = np.array([*v_S0, *K0_omega_IK0])
    if RigidBody == RigidBodyAxisAngle:
        q0 = np.array([*r_OS0, *Log_SO3(A_IK0)])
    elif RigidBody == RigidBodyQuaternion:
        q0 = np.array([*r_OS0, *Spurrier(A_IK0)])
    else:
        raise NotImplementedError
    RB2 = RigidBody(m, K_theta_S, q0, u0)

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

    # spring force
    spring = ScalarForceTranslational(
        frame,
        RB2,
        LinearSpring(k, g_ref=0),
        LinearDamper(d),
        K_r_SP2=np.array([-0.5 * l, 0, 0]),
    )

    #################
    # assemble system
    #################

    system.add(frame, RB1, rigid_connection)
    system.add(RB2, f_g)
    system.add(constraint)
    system.add(spring)
    system.assemble()

    ############################################################################
    #                   DAE solution
    ############################################################################
    sol = Solver(system, t1, dt, **solver_kwargs).solve()
    t, q = sol.t, sol.q

    ############################################################################
    #                   post processing
    ############################################################################
    zs = np.array(
        [
            A_IB(ti)[:, 2] @ (RB2.r_OP(ti, qi[RB2.qDOF]) - r_OB(ti))
            for (ti, qi) in zip(t, q)
        ]
    )

    A_BK = np.array([A_IB(ti).T @ RB2.A_IK(ti, qi[RB2.qDOF]) for (ti, qi) in zip(t, q)])
    alphas = np.array([Log_SO3(A_BKi)[-1] for A_BKi in A_BK])
    # remove jumps of Log_SO(3)
    for i in range(1, len(alphas)):
        diff = alphas[i] - alphas[i - 1]
        if diff > np.pi:
            alphas[i:] += diff
        elif diff < -np.pi:
            alphas[i:] -= diff

    ############################################################################
    #                   reference ODE solution
    ############################################################################
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

        K_nu_R_dot = A_KI @ omega_IB_t(t) + alpha_dot * cross3(
            _A_IB.T @ omega_IB(t), e3
        )

        h = -J_S.T @ (m * nu_S_dot + m * g * e3) - K_J_R.T @ K_theta_S @ K_nu_R_dot
        h[0] -= k * (z - z0) + k * z_dot

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
    sol_ref = solve_ivp(eqm, t_span, y0, t_eval=t_eval, rtol=1e-10, atol=1e-10)

    t_ref, y_ref = sol_ref.t, sol_ref.y

    ############################################################################
    #                   plot
    ############################################################################
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

    if show:
        plt.show()


################################################################################
# test setup
################################################################################

solver_and_kwargs = [
    (ScipyIVP, {}),
    (MoreauClassical, {}),
    (EulerBackward, {"method": "index 1"}),
    (EulerBackward, {"method": "index 2"}),
    (EulerBackward, {"method": "index 3"}),
    (EulerBackward, {"method": "index 2 GGL"}),
]

rigid_bodies = [
    RigidBodyQuaternion,
]

test_parameters = []

for RB in rigid_bodies:
    for SK in solver_and_kwargs:
        test_parameters.append((RB, *SK))


@pytest.mark.parametrize("RigidBody, Solver, kwargs", test_parameters)
def test_cylindrical(RigidBody, Solver, kwargs):
    run("Cylindrical", RigidBody, Solver, **kwargs)


@pytest.mark.parametrize("RigidBody, Solver, kwargs", test_parameters)
def test_prismatic(RigidBody, Solver, kwargs):
    run("Prismatic", RigidBody, Solver, **kwargs)


if __name__ == "__main__":
    show = True

    # simulations with RigidBodyAxisAngle only within tests

    #############
    # Cylindrical
    #############
    # run("Cylindrical", RigidBodyQuaternion, ScipyIVP)
    # run("Cylindrical", rigid_bodies[0], solver_and_kwargs[6][0], **solver_and_kwargs[6][1])
    run(
        "Cylindrical",
        rigid_bodies[0],
        solver_and_kwargs[7][0],
        **solver_and_kwargs[7][1],
    )

    # run("Cylindrical", RigidBodyQuaternion, MoreauClassical)

    # run("Cylindrical", RigidBodyQuaternion, EulerBackward, method="index 1")
    # run("Cylindrical", RigidBodyQuaternion, EulerBackward, method="index 2")
    # run("Cylindrical", RigidBodyQuaternion, EulerBackward, method="index 3")
    # run("Cylindrical", RigidBodyQuaternion, EulerBackward, method="index 2 GGL")

    ###########
    # Prismatic
    ###########
    # run("Prismatic", RigidBodyQuaternion, ScipyIVP)

    # run("Prismatic", RigidBodyQuaternion, MoreauClassical)

    # run("Prismatic", RigidBodyQuaternion, EulerBackward, method="index 1")
    # run("Prismatic", RigidBodyQuaternion, EulerBackward, method="index 2")
    # run("Prismatic", RigidBodyQuaternion, EulerBackward, method="index 3")
    # run("Prismatic", RigidBodyQuaternion, EulerBackward, method="index 2 GGL")
