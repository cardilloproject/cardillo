import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

from cardillo import System
from cardillo.solver import ScipyIVP, EulerBackward, RadauIIa
from cardillo.constraints import RigidConnection, Cylindrical, Prismatic
from cardillo.discrete import Frame, RigidBodyQuaternion
from cardillo.forces import Force
from cardillo.math import e1, Exp_SO3, Log_SO3, Spurrier, cross3

# setup solver
t_span = (0.0, 5.0)
t0, t1 = t_span
dt = 1.0e-2

# parameters
g = 9.81
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


def run(
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
            free_axis=2,
        )
    elif joint == "Prismatic":
        constraint = Prismatic(
            subsystem1=RB1,
            subsystem2=RB2,
            free_axis=2,
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


if __name__ == "__main__":
    #############
    # Cylindrical
    #############
    # run("Cylindrical", ScipyIVP)

    # run("Cylindrical", EulerBackward, method="index 1")
    # run("Cylindrical", EulerBackward, method="index 2")
    # run("Cylindrical", EulerBackward, method="index 3")
    # run("Cylindrical", EulerBackward, method="index 2 GGL")

    # run("Cylindrical", RadauIIa, dae_index=2)
    # run("Cylindrical", RadauIIa, dae_index=3, rtol=1e-2, atol=1e-2) # this is not working for alpha_dot0 != 0
    run("Cylindrical", RadauIIa, dae_index="GGL")

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
