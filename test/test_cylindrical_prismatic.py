import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import pytest
from itertools import product

from cardillo import System
from cardillo.solver import (
    ScipyIVP,
    ScipyDAE,
    Moreau,
    BackwardEuler,
    Rattle,
)
from cardillo.constraints import RigidConnection, Cylindrical, Prismatic
from cardillo.discrete import Frame, RigidBody
from cardillo.interactions import TwoPointInteraction
from cardillo.forces import Force
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.math import (
    Exp_SO3,
    T_SO3,
    T_SO3_psi,
    Log_SO3,
    Spurrier,
    cross3,
    ax2skew,
)


def run(
    joint: str,
    RigidBody,
    Solver,
    solver_kwargs,
    show=False,
):
    ###################
    # solver parameters
    ###################
    t_span = (0.0, 1.0)
    t0, t1 = t_span
    dt = 1.0e-2

    ############
    # parameters
    ############
    k = 20
    d = 10
    g = 9.81
    l = 10
    m = 1
    r = 0.2
    C = 1 / 2 * m * r**2
    A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
    B_Theta_C = np.diag(np.array([C, A, A]))

    #############
    # origin axis
    #############
    amplitude = l / 10
    omega = np.pi * 2
    e = lambda t: amplitude * np.cos(omega * t)
    e_t = lambda t: -amplitude * omega * np.sin(omega * t)
    e_tt = lambda t: -amplitude * omega * omega * np.cos(omega * t)

    n1 = np.random.rand(3)

    r_OJ = lambda t: e(t) * n1
    r_OJ_t = lambda t: e_t(t) * n1
    r_OJ_tt = lambda t: e_tt(t) * n1

    r_OJ0 = r_OJ(0)

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

    A_IJ = lambda t: Exp_SO3(psi(t))
    A_IJ_t = lambda t: ax2skew(omega_IB(t)) @ A_IJ(t)
    A_IJ_tt = lambda t: ax2skew(omega_IB_t(t)) @ A_IJ(t) + ax2skew(
        omega_IB(t)
    ) @ A_IJ_t(t)

    A_IJ0 = A_IJ(0)

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
        r_OP=r_OJ,
        r_OP_t=r_OJ_t,
        r_OP_tt=r_OJ_tt,
        A_IB=A_IJ,
        A_IB_t=A_IJ_t,
        A_IB_tt=A_IJ_tt,
    )

    B0_omega_IB0 = A_IJ0.T @ omega_IB(0)
    v_P0 = (
        r_OJ_t(0) + A_IJ0 @ np.array([0, 0, z_dot0]) + A_IJ_t(0) @ np.array([0, 0, z0])
    )

    q0 = np.array([*r_OJ0, *Spurrier(A_IJ0)])
    u0 = np.array([*v_P0, *B0_omega_IB0])
    RB1 = RigidBody(m, B_Theta_C, q0, u0, name="rigid body 1")

    rigid_connection = RigidConnection(frame, RB1)

    A_IB0 = A_IJ0
    r_OC0 = r_OJ0 + A_IB0 @ np.array([0.5 * l, 0, z0])

    K0_omega_IK0 = A_IB0.T @ omega_IB(0) + np.array([0, 0, alpha_dot0])
    v_C0 = v_P0 + A_IB0 @ cross3(K0_omega_IK0, np.array([0.5 * l, 0, 0]))

    q0 = np.array([*r_OC0, *Spurrier(A_IB0)])
    u0 = np.array([*v_C0, *K0_omega_IK0])
    RB2 = RigidBody(m, B_Theta_C, q0, u0, name="rigid body 2")

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
    spring_damper = SpringDamper(
        TwoPointInteraction(
            frame,
            RB2,
            B_r_CP2=np.array([-0.5 * l, 0, 0.5 * l - z0]),
        ),
        k,
        d,
        l_ref=0.5 * l,
        compliance_form=False,
        name="spring_damper",
    )

    #################
    # assemble system
    #################

    system.add(frame, RB1, rigid_connection)
    system.add(RB2, f_g)
    system.add(constraint)
    system.add(spring_damper)
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
            A_IJ(ti)[:, 2] @ (RB2.r_OP(ti, qi[RB2.qDOF]) - r_OJ(ti))
            for (ti, qi) in zip(t, q)
        ]
    )

    A_BK = np.array([A_IJ(ti).T @ RB2.A_IB(ti, qi[RB2.qDOF]) for (ti, qi) in zip(t, q)])
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

        _A_IJ = A_IJ(t)
        e_x_B, e_y_B, e_z_B = _A_IJ.T

        A_BK = np.array(
            [
                [ca, -sa, 0],
                [sa, ca, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        A_KI = A_BK.T @ _A_IJ.T

        J_C = np.zeros((3, 2), dtype=float)
        J_C[:, 0] = e_z_B
        J_C[:, 1] = 0.5 * l * (ca * e_y_B - sa * e_x_B)

        B_J_R = np.zeros((3, 2), dtype=float)
        B_J_R[:, 1] = e3

        M = m * (J_C.T @ J_C) + B_J_R.T @ B_Theta_C @ B_J_R

        _omega_IB = omega_IB(t)

        B_r_BS = z * e3 + 0.5 * l * (ca * e1 + sa * e2)
        r_BS = _A_IJ @ B_r_BS
        B_r_BS_dot = z_dot * e3 + 0.5 * l * alpha_dot * (ca * e2 - sa * e1)

        nu_C_dot = (
            r_OJ_tt(t)
            + 2 * cross3(_omega_IB, _A_IJ @ B_r_BS_dot)
            - _A_IJ @ (0.5 * l * alpha_dot**2 * (ca * e1 + sa * e2))
            + cross3(omega_IB_t(t), r_BS)
            + cross3(_omega_IB, cross3(_omega_IB, r_BS))
        )

        B_nu_R_dot = A_KI @ omega_IB_t(t) + alpha_dot * cross3(
            _A_IJ.T @ omega_IB(t), e3
        )

        h = -J_C.T @ (m * nu_C_dot + m * g * e3) - B_J_R.T @ B_Theta_C @ B_nu_R_dot
        h[0] -= k * (z - z0) + d * z_dot

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

        _r_OJ = r_OJ(ti)

        _A_IJ = A_IJ(ti)
        e_x_B, e_y_B, e_z_B = _A_IJ.T

        r_OP = _r_OJ + z * e_z_B
        r_OC = r_OP + 0.5 * l * (np.cos(alpha) * e_x_B + np.sin(alpha) * e_y_B)
        r_OQ = r_OP + l * (np.cos(alpha) * e_x_B + np.sin(alpha) * e_y_B)
        x, y, z = np.array([r_OP, r_OC, r_OQ]).T

        rod.set_xdata(x)
        rod.set_ydata(y)
        rod.set_3d_properties(z)

        r0 = _r_OJ + diff * e_z_B
        r1 = _r_OJ - diff * e_z_B
        x, y, z = np.array([r0, _r_OJ, r1]).T

        axis.set_xdata(x)
        axis.set_ydata(y)
        axis.set_3d_properties(z)

    frames = len(t_ref)
    interval = dt * 1000

    if show:
        anim = animation.FuncAnimation(
            fig, animate, frames=frames, interval=interval, blit=False
        )
        plt.show()
    else:
        plt.close()


################################################################################
# test setup
################################################################################
solver_and_kwargs = [
    (ScipyIVP, {}),
    (ScipyDAE, {}),
    (Moreau, {}),
    (BackwardEuler, {}),
    (Rattle, {}),
]

rigid_bodies = [
    RigidBody,
]

joints = [
    "Cylindrical",
    "Prismatic",
]

test_parameters = product(solver_and_kwargs, rigid_bodies, joints)


@pytest.mark.parametrize("Solver_and_kwargs, RigidBody, joint", test_parameters)
def test_cylindrical_prismatic(Solver_and_kwargs, RigidBody, joint, show=False):
    Solver, solver_kwargs = Solver_and_kwargs
    run(joint, RigidBody, Solver, solver_kwargs, show)


if __name__ == "__main__":
    for p in test_parameters:
        test_cylindrical_prismatic(*p, show=True)
