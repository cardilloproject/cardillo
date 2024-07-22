import numpy as np
import pytest
from itertools import product
from math import pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math import A_IB_basic, cross3, Spurrier, Exp_SO3
from cardillo import System
from cardillo.discrete import Frame
from cardillo.constraints import (
    Spherical,
    Revolute,
)
from cardillo.discrete import RigidBody
from cardillo.forces import Force
from cardillo.force_laws import KelvinVoigtElement
from cardillo.solver import ScipyIVP, ScipyDAE, BackwardEuler, Moreau, Rattle

solvers_and_kwargs = [
    (ScipyIVP, {}),
    (ScipyDAE, {}),
    (BackwardEuler, {}),
    (Moreau, {}),
    (Rattle, {}),
]

joints = [
    ("Spherical", None, None),
    ("Revolute", None, None),
    ("PDRotational", 1e2, 3e1),
]

test_parameters = product(solvers_and_kwargs, joints)


@pytest.mark.parametrize("Solver_and_kwargs, joint_and_stiffnesses", test_parameters)
def test_spherical_revolute(Solver_and_kwargs, joint_and_stiffnesses, show=False):
    Solver, solver_kwargs = Solver_and_kwargs
    joint, k, d = joint_and_stiffnesses
    m = 1
    r = 0.1
    l = 2
    g = 9.81

    A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
    C = 1 / 2 * m * r**2
    B_Theta_C = np.diag(np.array([A, C, A]))

    use_spherical_joint = use_revolute_joint = use_pdrotational_joint = False

    # random rotation
    psi = np.random.rand(3)
    # psi = np.array((np.pi / 2, 0, 0))
    # psi = np.array((0, 0, 0))
    A_IprimeI = Exp_SO3(psi)

    ############################################################################
    #                   Rigid Body 1
    ############################################################################
    alpha0 = pi / 2
    alpha_dot0 = 0

    r_OJ1 = np.zeros(3)
    A_IJ1 = np.eye(3)
    A_IB10 = A_IB_basic(alpha0).z
    r_OC10 = -0.5 * l * A_IB10[:, 1]
    B1_omega01 = np.array([0, 0, alpha_dot0])
    vS1 = cross3(B1_omega01, r_OC10)

    p01 = Spurrier(A_IprimeI @ A_IB10)
    q01 = np.concatenate([A_IprimeI @ r_OC10, p01])
    u01 = np.concatenate([A_IprimeI @ vS1, B1_omega01])
    origin = Frame(r_OP=A_IprimeI @ r_OJ1, A_IB=A_IprimeI @ A_IJ1)
    # origin = Frame(r_OP=r_OJ1, A_IB=A_IJ1)
    RB1 = RigidBody(m, B_Theta_C, q01, u01)

    if joint == "Spherical":
        use_spherical_joint = True
    elif joint == "Revolute":
        use_revolute_joint = True
    elif joint == "PDRotational":
        use_pdrotational_joint = True
        assert (k and d) is not None
    else:
        raise RuntimeError(
            'Invalid Argument.\nPossible Arguments are "Spherical", "Revolute", "PDRotational".'
        )

    if use_spherical_joint:
        joint1 = Spherical(origin, RB1, r_OJ0=A_IprimeI @ r_OJ1)
    elif use_revolute_joint:
        joint1 = Revolute(
            origin, RB1, axis=2, r_OJ0=A_IprimeI @ r_OJ1, A_IJ0=A_IprimeI @ A_IJ1
        )
    elif use_pdrotational_joint:
        joint1 = Revolute(
            origin, RB1, axis=2, r_OJ0=A_IprimeI @ r_OJ1, A_IJ0=A_IprimeI @ A_IJ1
        )
        pd_rotational1 = KelvinVoigtElement(joint1, k, d, l_ref=-alpha0)

    ############################################################################
    #                   Rigid Body 2
    ############################################################################
    beta0 = -pi / 4
    beta_dot0 = 0

    r_OJ2 = -l * A_IB10[:, 1]
    A_IJ2 = A_IB10
    A_IB20 = A_IB10 @ A_IB_basic(beta0).z
    r_B2S2 = -0.5 * l * A_IB20[:, 1]
    r_OC20 = r_OJ2 + r_B2S2
    B2_omega02 = np.array([0, 0, alpha_dot0 + beta_dot0])
    vB2 = cross3(B1_omega01, r_OJ2)
    vS2 = vB2 + cross3(B2_omega02, r_B2S2)

    p02 = Spurrier(A_IprimeI @ A_IB20)
    q02 = np.concatenate([A_IprimeI @ r_OC20, p02])
    u02 = np.concatenate([A_IprimeI @ vS2, B2_omega02])
    RB2 = RigidBody(m, B_Theta_C, q02, u02)

    if use_spherical_joint:
        joint2 = Spherical(RB1, RB2, r_OJ0=A_IprimeI @ r_OJ2)
    elif use_revolute_joint:
        joint2 = Revolute(
            RB1, RB2, axis=2, r_OJ0=A_IprimeI @ r_OJ2, A_IJ0=A_IprimeI @ A_IJ2
        )
    elif use_pdrotational_joint:
        joint2 = Revolute(
            RB1, RB2, axis=2, r_OJ0=A_IprimeI @ r_OJ2, A_IJ0=A_IprimeI @ A_IJ2
        )
        pd_rotational2 = KelvinVoigtElement(joint2, k, d, l_ref=-beta0)

    ############################################################################
    #                   model
    ############################################################################
    system = System()
    system.add(origin)
    system.add(RB1)
    system.add(joint1)
    system.add(RB2)
    system.add(joint2)
    system.add(Force(lambda t: A_IprimeI @ np.array([0, -g * m, 0]), RB1))
    system.add(Force(lambda t: A_IprimeI @ np.array([0, -g * m, 0]), RB2))
    if use_pdrotational_joint:
        system.add(pd_rotational1, pd_rotational2)

    system.assemble()

    ############################################################################
    #                   solver
    ############################################################################
    t0 = 0
    t1 = 3
    dt = 1e-2
    sol = Solver(system, t1, dt, **solver_kwargs).solve()
    t = sol.t
    q = sol.q

    ############################################################################
    #                   animation
    ############################################################################
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        scale = 2 * l
        ax.set_xlim3d(left=-scale, right=scale)
        ax.set_ylim3d(bottom=-scale, top=scale)
        ax.set_zlim3d(bottom=-scale, top=scale)
        ax.view_init(vertical_axis="y")

        def init(t, q):
            x_0, y_0, z_0 = origin.r_OP(t)
            x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])
            x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF])

            A_IB1 = RB1.A_IB(t, q[RB1.qDOF])
            d11 = A_IB1[:, 0]
            d21 = A_IB1[:, 1]
            d31 = A_IB1[:, 2]

            A_IB2 = RB2.A_IB(t, q[RB2.qDOF])
            d12 = A_IB2[:, 0]
            d22 = A_IB2[:, 1]
            d32 = A_IB2[:, 2]

            (COM,) = ax.plot(
                [x_0, x_S1, x_S2], [y_0, y_S1, y_S2], [z_0, z_S1, z_S2], "-ok"
            )
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

            return COM, d11_, d21_, d31_, d12_, d22_, d32_

        def update(t, q, COM, d11_, d21_, d31_, d12_, d22_, d32_):
            # def update(t, q, COM, d11_, d21_, d31_):
            x_0, y_0, z_0 = origin.r_OP(t)
            x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF], B_r_CP=np.array([0, -l / 2, 0]))
            x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF], B_r_CP=np.array([0, -l / 2, 0]))

            A_IB1 = RB1.A_IB(t, q[RB1.qDOF])
            d11 = A_IB1[:, 0]
            d21 = A_IB1[:, 1]
            d31 = A_IB1[:, 2]

            A_IB2 = RB2.A_IB(t, q[RB2.qDOF])
            d12 = A_IB2[:, 0]
            d22 = A_IB2[:, 1]
            d32 = A_IB2[:, 2]

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

    # compute reference solution
    def _eqm(t, x):
        thetaA = A + 5 * m * (l**2) / 4
        thetaB = A + m * (l**2) / 4

        M = np.array(
            [
                [thetaA, 0.5 * m * l * l * np.cos(x[0] - x[1])],
                [0.5 * m * l * l * np.cos(x[0] - x[1]), thetaB],
            ]
        )

        h = np.array(
            [
                -0.5 * m * l * l * (x[3] ** 2) * np.sin(x[0] - x[1])
                - 1.5 * m * l * g * np.sin(x[0]),
                0.5 * m * l * l * (x[2] ** 2) * np.sin(x[0] - x[1])
                - 0.5 * m * l * g * np.sin(x[1]),
            ]
        )

        dx = np.zeros(4)
        dx[:2] = x[2:]
        dx[2:] = np.linalg.solve(M, h)
        return dx

    def _eqm_pd_rotational(t, x):
        thetaA = A + 5 * m * (l**2) / 4
        thetaB = A + m * (l**2) / 4

        M = np.array(
            [
                [thetaA, 0.5 * m * l * l * np.cos(x[0] - x[1])],
                [0.5 * m * l * l * np.cos(x[0] - x[1]), thetaB],
            ]
        )

        h = np.array(
            [
                -0.5 * m * l * l * (x[3] ** 2) * np.sin(x[0] - x[1])
                - 1.5 * m * l * g * np.sin(x[0])
                - k * (2 * x[0] - x[1])
                - d * (2 * x[2] - x[3]),
                0.5 * m * l * l * (x[2] ** 2) * np.sin(x[0] - x[1])
                - 0.5 * m * l * g * np.sin(x[1])
                - k * (x[1] - x[0])
                - d * (x[3] - x[2]),
            ]
        )

        dx = np.zeros(4)
        dx[:2] = x[2:]
        dx[2:] = np.linalg.solve(M, h)
        return dx

    if use_pdrotational_joint:
        eqm = _eqm_pd_rotational
    else:
        eqm = _eqm

    x0 = np.array([alpha0, alpha0 + beta0, alpha_dot0, alpha_dot0 + beta_dot0])
    ref = solve_ivp(eqm, [t0, t1], x0, method="RK45", rtol=1e-8, atol=1e-12)
    x = ref.y
    t_ref = ref.t

    alpha_ref = x[0]
    phi_ref = x[1]

    # q = (A_IprimeI.T @ sol.q[:, :3].T).T
    sol_q1 = sol.q[:, :3] @ A_IprimeI
    sol_q2 = sol.q[:, 7:10] @ A_IprimeI
    alpha = np.arctan2(sol_q1[:, 0], -sol_q1[:, 1])
    # alpha = np.arctan2(sol.q[:, 0], -sol.q[:, 1])
    x_B2 = 2 * sol_q1[:, 0]
    y_B2 = 2 * sol_q1[:, 1]
    # phi = np.arctan2(sol.q[:, 7] - x_B2, -(sol.q[:, 8] - y_B2))
    phi = np.arctan2(sol_q2[:, 0] - x_B2, -(sol_q2[:, 1] - y_B2))

    if show:
        fig, ax = plt.subplots()
        ax.plot(t_ref, alpha_ref, "-.r", label="alpha ref")
        ax.plot(t, alpha, "-r", label="alpha")

        ax.plot(t_ref, phi_ref, "-.g", label="phi ref")
        ax.plot(t, phi, "-g", label="phi")

        ax.grid()
        ax.legend()

        plt.show()


if __name__ == "__main__":
    for p in test_parameters:
        test_spherical_revolute(*p, show=True)
