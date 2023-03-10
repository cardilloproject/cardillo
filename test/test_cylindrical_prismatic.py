import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# from cardillo import System
# from cardillo.solver import ScipyIVP, EulerBackward, RadauIIa
# from cardillo.constraints import Revolute
from cardillo.discrete import RigidBodyAxisAngle, RigidBodyQuaternion

# from cardillo.forces import (
#     LinearSpring,
#     LinearDamper,
#     PDRotationalJoint,
# )
from cardillo.math import e1, e2, e3, Exp_SO3, axis_angle2quat, norm

# parameters of cylinder
l = 10
m = 1
r = 0.2
C = 1 / 2 * m * r**2
A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
K_theta_S = np.diag(np.array([C, A, A]))


def RigidCylinder(RigidBodyParametrization):
    class _RigidCylinder(RigidBodyParametrization):
        def __init__(self, q0=None, u0=None):
            super().__init__(m, K_theta_S, q0=q0, u0=u0)

    return _RigidCylinder


def run(
    # x0,
    # alpha_dot0,
    # RigidBodyParametrization=RigidBodyQuaternion,
    # solver_type="EulerBackward",
    # plot=True,
    # rotation_axis=2,
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
    z_dot0 = 0

    alpha0 = 0
    alpha_dot0 = 0

    ##############
    # ODE solution
    ##############
    # TODO: Use arbitrary rotation axis
    theta = A
    # theta = K_theta_S[rotation_axis, rotation_axis]

    def eqm(t, y):
        z, alpha = y[:2]

        dy = np.zeros_like(y)

        # trivial kinematic equation
        dy[:2] = y[2:]

        # equations of motion
        dy[2] = -e_zB[-1] / m
        dy[3] = (
            -0.5
            * l
            * (np.cos(alpha) * e_yB[-1] - np.sin(alpha) * e_xB[-1])
            / (m * l**2 / 4 + theta)
        )

        return dy

    t_span = (0, 5)
    t0, t1 = t_span
    dt = 1.0e-2
    t_eval = np.arange(t0, t1, dt)
    y0 = np.array([z0, alpha0, z_dot0, alpha_dot0], dtype=float)
    sol = solve_ivp(eqm, t_span, y0, t_eval=t_eval)

    t_ref, y_ref = sol.t, sol.y

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].set_title("z")
    ax[0, 0].plot(t_ref, y_ref[0])

    ax[0, 1].set_title("alpha")
    ax[0, 1].plot(t_ref, y_ref[1])

    ax[1, 0].set_title("z_dot")
    ax[1, 0].plot(t_ref, y_ref[2])

    ax[1, 1].set_title("alpha_dot")
    ax[1, 1].plot(t_ref, y_ref[3])

    # plt.show()

    # animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    z1 = y_ref[0, -1]
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
    exit()

    def update(t, q, COM, d11_, d21_, d31_, d12_, d22_, d32_):
        # def update(t, q, COM, d11_, d21_, d31_):
        x_0, y_0, z_0 = origin.r_OP(t)
        x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF], K_r_SP=np.array([0, -l / 2, 0]))
        x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF], K_r_SP=np.array([0, -l / 2, 0]))

        A_IK1 = RB1.A_IK(t, q[RB1.qDOF])
        d11 = A_IK1[:, 0]
        d21 = A_IK1[:, 1]
        d31 = A_IK1[:, 2]

        A_IK2 = RB2.A_IK(t, q[RB2.qDOF])
        d12 = A_IK2[:, 0]
        d22 = A_IK2[:, 1]
        d32 = A_IK2[:, 2]

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

    COM, d11_, d21_, d31_, d12_, d22_, d32_ = init(0, q[0])

    def animate(i):
        update(t[i], q[i], COM, d11_, d21_, d31_, d12_, d22_, d32_)

    # compute naimation interval according to te - ts = frames * interval / 1000
    frames = len(t)
    interval = dt * 1000
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    exit()

    # r_OP0 = np.zeros(3)
    # v_P0 = np.zeros(3)
    # K_Omega0 = np.array((0, 0, alpha_dot0))
    # u0 = np.hstack((v_P0, K_Omega0))

    # if type(RigidBodyParametrization) is type(RigidBodyAxisAngle):
    #     q0 = np.hstack((r_OP0, psi))
    #     rigid_body = RigidCylinder(RigidBodyAxisAngle)(q0, u0)
    # elif type(RigidBodyParametrization) is type(RigidBodyQuaternion):
    #     n_psi = norm(psi)
    #     p = axis_angle2quat(psi / n_psi, n_psi)
    #     q0 = np.hstack((r_OP0, p))
    #     rigid_body = RigidCylinder(RigidBodyQuaternion)(q0, u0)
    # else:
    #     raise (TypeError)

    # system = System()
    # joint = PDRotationalJoint(Revolute, Spring=LinearSpring, Damper=LinearDamper)(
    #     subsystem1=system.origin,
    #     subsystem2=rigid_body,
    #     r_OB0=np.zeros(3),
    #     A_IB0=A_IK0,
    #     rotation_axis=rotation_axis,
    #     k=k,
    #     d=d,
    #     g_ref=g_ref,
    # )

    # system.add(rigid_body, joint)
    # system.assemble()

    # ############################################################################
    # #                   solver
    # ############################################################################
    # t1 = 2
    # dt = 1.0e-2
    # # dt = 5.0e-3
    # match solver_type:
    #     case "ScipyIVP":
    #         solver = ScipyIVP(system, t1, dt, atol=1e-8)
    #     case "RadauIIaDAE2":
    #         solver = RadauIIa(
    #             system, t1, dt, atol=1e-2, rtol=1e-2, dae_index=2, max_step=dt
    #         )
    #     case "RadauIIaDAE3":
    #         solver = RadauIIa(
    #             system, t1, dt, atol=1e-4, rtol=1e-4, dae_index=3, max_step=dt
    #         )
    #     case "RaudauIIaGGL":
    #         solver = RadauIIa(
    #             system, t1, dt, atol=1e-3, rtol=1e-3, dae_index="GGL", max_step=dt
    #         )
    #     case "EulerBackward" | _:
    #         solver = EulerBackward(system, t1, dt)

    # sol = solver.solve()
    # t = sol.t
    # q = sol.q
    # u = sol.u

    ############################################################################
    #                   plot
    ############################################################################
    if plot:
        # joint.reset()
        # alpha_cmp = [joint.angle(ti, qi[joint.qDOF]) for ti, qi in zip(t, q)]

        theta = K_theta_S[rotation_axis, rotation_axis]

        def eqm(t, x):
            dx = np.zeros(2)
            dx[0] = x[1]
            dx[1] = -1 / theta * (d * x[1] + k * (x[0] - g_ref))
            return dx

        x0 = np.array((0, alpha_dot0))
        ref = solve_ivp(eqm, [0, t1], x0, method="RK45", rtol=1e-8, atol=1e-12)
        x = ref.y
        t_ref = ref.t
        alpha_ref = x[0]

        fig, ax = plt.subplots(1, 1)

        ax.plot(t, alpha_cmp, "-k", label="alpha")
        ax.plot(t_ref, alpha_ref, "-.r", label="alpha_ref")
        ax.legend()

        plt.show()


if __name__ == "__main__":
    run()
    exit()
    profiling = False

    # initial rotational velocity e_z^K axis
    alpha_dot0 = 0

    # axis angle rotation
    psi = np.random.rand(3)
    # psi = np.array((0, 1, 0))
    # psi = np.array((1, 0, 0))
    # Following rotations result in linear eqms, Radau Solver without max step argument set rotates more than 360° in one time step.
    # psi = np.array((0, 0, 1))
    # psi = np.zeros(3)

    A_IK0 = Exp_SO3(psi)
    print(f"A_IK0:\n{A_IK0}")

    # spring stiffness and damper parameter
    k = 1e1
    d = 0.05
    # k=d=0
    g_ref = 2 * np.pi
    rotation_axis = 1

    # Rigid body parametrization
    RigidBodyParametrization = RigidBodyQuaternion
    # RigidBodyParametrization = RigidBodyAxisAngle

    # Solver
    solver = [
        "ScipyIVP",
        "RadauIIaDAE2",
        "RadauIIaDAE3",
        "RaudauIIaGGL",
        "EulerBackward",
    ]

    if profiling:
        import cProfile, pstats

        profiler = cProfile.Profile()

        profiler.enable()
        run(
            k=k,
            d=d,
            psi=psi,
            alpha_dot0=alpha_dot0,
            g_ref=g_ref,
            RigidBodyParametrization=RigidBodyParametrization,
            solver_type=solver[2],
            plot=False,
            rotation_axis=rotation_axis,
        )
        profiler.disable()

        stats = pstats.Stats(profiler)
        # stats.print_stats(20)
        stats.sort_stats(pstats.SortKey.TIME, pstats.SortKey.CUMULATIVE).print_stats(
            0.5, "cardillo"
        )
    else:
        run(
            k=k,
            d=d,
            psi=psi,
            alpha_dot0=alpha_dot0,
            g_ref=g_ref,
            RigidBodyParametrization=RigidBodyParametrization,
            solver_type=solver[1],
            plot=True,
            rotation_axis=rotation_axis,
        )
