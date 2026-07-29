import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pytest
import warnings

from cardillo import System
from cardillo.discrete import RigidBody, Box, Sphere, Frame, Tetrahedron
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane
from cardillo.solver import Moreau, BackwardEuler, SolverOptions
from cardillo.math import A_IB_basic, Exp_SO3
from cardillo.math.approx_fprime import approx_fprime
from cardillo.constraints import RigidConnection

warnings.filterwarnings(
    "ignore", message=r".*'approx_fprime' is used.*", category=UserWarning
)


def run(solver=Moreau, VTK_export=False):
    ############################################################################
    #                   system setup
    ############################################################################

    ###################
    # solver parameters
    ###################
    t_span = (0.0, 2)
    t0, t1 = t_span
    dt = 1.0e-3

    ############
    # parameters
    ############
    radius = 0.05  # radius of ball
    mass = 1  # mass ball
    density = mass / (4 / 3 * np.pi * radius**3)  # density of ball
    g = np.array([0, 0, -10])  # gravitational acceleration
    e_N = 0.0  # restitution coefficient in normal direction
    e_F = 0.0  # restitution coefficient in tangent direction
    mu = 0.3  # frictional coefficient

    # initialize system
    system = System()
    # floor
    omega = 2 * np.pi * 0.5
    amplitude = radius
    # r_OP=lambda t: amplitude * np.array([np.sin(omega * t), 0.0, 0.0])
    # r_OP=lambda t: amplitude * np.array([0.0, np.sin(omega * t), 0.0])
    r_OP = lambda t: amplitude * np.array([0.0, 0.0, np.sin(omega * t)])
    # r_OP = lambda t: amplitude * np.array([0.0, 0.0, 0.0])

    angle = np.deg2rad(20)
    # A_IB = A_IB_basic(np.deg2rad(10)).x @ A_IB_basic(np.deg2rad(10)).y
    # A_IB=lambda t: A_IB_basic(angle * np.sin(omega * t)).x
    # A_IB=lambda t: A_IB_basic(angle * np.sin(omega * t)).y
    # A_IB = lambda t: A_IB_basic(angle * np.sin(omega * t)).z
    A_IB = (
        lambda t: A_IB_basic(angle * np.sin(omega * t)).y
        @ A_IB_basic(angle * np.sin(omega * t)).z
    )

    floor = Box(Frame)(
        dimensions=[4.5, 4.5, 0.0001],
        r_OP=r_OP,
        A_IB=A_IB,
        name="floor",
    )
    system.add(floor)  # (only for visualization purposes)

    # initial conditions ball
    initial_gap = 0.01 * radius + radius
    r_OC0 = np.array([0, 0, radius + initial_gap])
    q0 = RigidBody.pose2q(r_OC0, np.eye(3))
    u0 = np.zeros(6)

    # ball as sphere
    ball = Sphere(RigidBody)(
        radius=radius,
        density=density,
        subdivisions=3,
        q0=q0,
        u0=u0,
        name="ball",
    )

    system.add(ball)

    # gravity of ball
    system.add(Force(ball.mass * g, ball, name="gravity_" + ball.name))

    # contact between ball and plane
    system.add(
        Sphere2Plane(
            floor,
            ball,
            mu=mu,
            radius=radius,
            e_N=e_N,
            e_F=e_F,
            name="floor2" + ball.name,
        )
    )

    # add tetrahedron
    edge = 0.1
    density = 7700
    mu = 0.3
    r_OC0_tetra = np.array([10 * edge, 0, edge])
    q0_tetra = RigidBody.pose2q(r_OC0_tetra, np.eye(3))
    u0_tetra = np.zeros(6)

    tetrahedron = Tetrahedron(RigidBody)(
        edge=edge, density=density, q0=q0_tetra, u0=u0_tetra, name="tetrahedron"
    )

    system.add(tetrahedron)

    # gravity of ball
    system.add(
        Force(tetrahedron.mass * g, tetrahedron, name="gravity_" + tetrahedron.name)
    )

    for i, vertex in enumerate(tetrahedron.B_visual_mesh.vertices):
        system.add(
            Sphere2Plane(
                floor,
                tetrahedron,
                mu=mu,
                radius=0,
                e_N=e_N,
                e_F=e_F,
                B_r_CP2=vertex,
                name=f"floor2{tetrahedron.name}_{i}",
            )
        )

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    solver = solver(
        system,
        t1,
        dt,
        options=SolverOptions(prox_scaling=0.4, continue_with_unconverged=True),
    )  # create solver
    sol = solver.solve()  # simulate system

    # vtk-export
    if VTK_export:
        dir_name = Path(__file__).parent
        system.export(dir_name, "vtk", sol)


def test_with_Moreau():
    run(Moreau)


def test_with_BackwardEuler():
    run(BackwardEuler)


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
def test_implementation():
    m = 1.0
    B_Theta_C = np.diag([1.0, 1.0, 1.0])
    q01 = np.random.rand(7) * 5
    u01 = np.random.rand(6) * 3
    u0_dot1 = np.random.rand(6) * 4
    q02 = np.random.rand(7) * 5
    u02 = np.random.rand(6) * 3
    u0_dot2 = np.random.rand(6) * 4

    body1 = RigidBody(m, B_Theta_C, q01, u01, name="Body1")
    body2 = RigidBody(m, B_Theta_C, q02, u02, name="Body2")

    B_r_CP1 = np.random.rand(3)
    B_r_CP2 = np.random.rand(3)
    A_B1P = Exp_SO3(np.random.rand(3))

    B1_n = A_B1P[:, 2]

    mu = 1.0
    radius = np.random.rand()

    contact = Sphere2Plane(body1, body2, mu, radius, B_r_CP1, B_r_CP2, A_B1P)

    # assembly
    t0 = np.random.rand()
    body1.qDOF = np.arange(0, 7)
    body2.qDOF = np.arange(7, 14)
    body1.uDOF = np.arange(0, 6)
    body2.uDOF = np.arange(6, 12)
    contact.assembler_callback()

    q0 = np.array([*q01, *q02])
    u0 = np.array([*u01, *u02])
    q0_dot = np.array([*body1.q_dot(t0, q01, u01), *body2.q_dot(t0, q02, u02)])
    u0_dot = np.array([*u0_dot1, *u0_dot2])
    la_N0 = np.random.rand(1)
    la_F0 = np.random.rand(2)

    # compute contact kinematics analytically
    n = body1.A_IB(t0, q01) @ B1_n
    r_OP1 = body1.r_OP(t0, q01, B_r_CP=B_r_CP1)
    r_OP2 = body2.r_OP(t0, q02, B_r_CP=B_r_CP2)

    g_N_ana = n @ (r_OP2 - r_OP1) - radius

    ####################
    # normal direction #
    ####################
    # g_N
    g_N = contact.g_N(t0, q0)[0]
    assert np.isclose(g_N, g_N_ana), f"g_N: {np.abs(g_N - g_N_ana)}"

    # g_N_q
    g_N_q = contact.g_N_q(t0, q0)
    g_N_q_num = approx_fprime(q0, lambda q_: contact.g_N(t0, q_))
    assert np.all(
        np.isclose(g_N_q, g_N_q_num, rtol=1e-5)
    ), f"g_N_q: {np.linalg.norm(g_N_q - g_N_q_num)}"

    # g_N_dot
    g_N_dot = contact.g_N_dot(t0, q0, u0)
    g_N_dot_num = g_N_q_num @ q0_dot
    assert np.isclose(
        g_N_dot, g_N_dot_num
    ), f"g_N_dot: {np.linalg.norm(g_N_dot - g_N_dot_num)}"

    # g_N_dot_q
    g_N_dot_q = contact.g_N_dot_q(t0, q0, u0)
    g_N_dot_q_num = approx_fprime(q0, lambda q_: contact.g_N_dot(t0, q_, u0))
    assert np.all(
        np.isclose(g_N_dot_q, g_N_dot_q_num, rtol=1e-5)
    ), f"g_N_dot_q: {np.linalg.norm(g_N_dot_q - g_N_dot_q_num)}"

    # g_N_dot_u
    g_N_dot_u = contact.g_N_dot_u(t0, q0)
    g_N_dot_u_num = approx_fprime(u0, lambda u_: contact.g_N_dot(t0, q0, u_))
    assert np.all(
        np.isclose(g_N_dot_u, g_N_dot_u_num, rtol=1e-5)
    ), f"g_N_dot_u: {np.linalg.norm(g_N_dot_u - g_N_dot_u_num)}"

    # W_N
    W_N = contact.W_N(t0, q0)
    assert np.all(
        np.isclose(W_N, g_N_dot_u.T, rtol=1e-5)
    ), f"W_N: {np.linalg.norm(W_N - g_N_dot_u.T)}"

    # g_N_ddot
    g_N_ddot = contact.g_N_ddot(t0, q0, u0, u0_dot)
    g_N_ddot_num = g_N_dot_q_num @ q0_dot + g_N_dot_u @ u0_dot
    assert np.isclose(
        g_N_ddot, g_N_ddot_num
    ), f"g_N_ddot: {np.linalg.norm(g_N_ddot - g_N_ddot_num)}"

    # Wla_N_q
    Wla_N_q = contact.Wla_N_q(t0, q0, la_N0)
    Wla_N_q_num = approx_fprime(q0, lambda q_: contact.W_N(t0, q_) @ la_N0)
    assert np.all(
        np.isclose(Wla_N_q, Wla_N_q_num, rtol=1e-5)
    ), f"Wla_N_q: {np.linalg.norm(Wla_N_q - Wla_N_q_num)}"

    ########################
    # tangential direction #
    ########################
    # gamma_F_q
    gamma_F_q_num = approx_fprime(q0, lambda q_: contact.gamma_F(t0, q_, u0))
    gamma_F_q = contact.gamma_F_q(t0, q0, u0)
    assert np.all(
        np.isclose(gamma_F_q, gamma_F_q_num, rtol=1e-5)
    ), f"gamma_F_q: {np.linalg.norm(gamma_F_q - gamma_F_q_num)}"

    # gamma_F_u
    gamma_F_u_num = approx_fprime(u0, lambda u_: contact.gamma_F(t0, q0, u_))
    gamma_F_u = contact.gamma_F_u(t0, q0)
    assert np.all(
        np.isclose(gamma_F_u, gamma_F_u_num, rtol=1e-5)
    ), f"gamma_F_u: {np.linalg.norm(gamma_F_u - gamma_F_u_num)}"

    # W_F
    W_F = contact.W_F(t0, q0)
    assert np.all(
        np.isclose(W_F, gamma_F_u.T, rtol=1e-5)
    ), f"W_F: {np.linalg.norm(W_F - gamma_F_u.T)}"

    # gamma_F_dot
    gamma_F_dot = contact.gamma_F_dot(t0, q0, u0, u0_dot)
    gamma_F_dot_num = gamma_F_q @ q0_dot + gamma_F_u @ u0_dot
    assert np.all(
        np.isclose(gamma_F_dot, gamma_F_dot_num)
    ), f"gamma_F_dot: {np.linalg.norm(gamma_F_dot - gamma_F_dot_num)}"

    # Wla_N_q
    Wla_F_q = contact.Wla_F_q(t0, q0, la_F0)
    Wla_F_q_num = approx_fprime(q0, lambda q_: contact.W_F(t0, q_) @ la_F0)
    assert np.all(
        np.isclose(Wla_F_q, Wla_F_q_num, rtol=1e-5)
    ), f"Wla_F_q: {np.linalg.norm(Wla_F_q - Wla_F_q_num)}"


def test_rotating_plate_kin(show_plot=False):
    A_rig = Exp_SO3(np.random.rand(3))
    r_rig = np.random.rand(3)

    sol, gamma, gamma_theo = rotating_plate(np.eye(3), np.zeros(3))
    sol_rig, gamma_rig, gamma_theo_rig = rotating_plate(A_rig, r_rig)

    if show_plot:
        # plot relative velocities
        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].plot(sol.t, gamma[:, 0], label="gamma_1")
        ax[0, 1].plot(sol.t, gamma[:, 1], label="gamma_2")
        ax[0, 0].plot(sol.t, gamma_theo[:, 0], "--", label="theo_1")
        ax[0, 1].plot(sol.t, gamma_theo[:, 1], "--", label="theo_2")

        ax[0, 0].plot(sol_rig.t, gamma_rig[:, 0], "-.", label="gamma_1 rig")
        ax[0, 1].plot(sol_rig.t, gamma_rig[:, 1], "-.", label="gamma_2 rig")
        ax[0, 0].plot(sol_rig.t, gamma_theo_rig[:, 0], ":", label="theo_1 rig")
        ax[0, 1].plot(sol_rig.t, gamma_theo_rig[:, 1], ":", label="theo_2 rig")

        ax[0, 0].legend()
        ax[0, 1].legend()
        ax[0, 0].grid()
        ax[0, 1].grid()
        ax[0, 0].set_title("gamma_1")
        ax[0, 1].set_title("gamma_2")
        plt.show()

    assert np.all(
        np.isclose(gamma, gamma_theo, atol=1e-6)
    ), f"gamma: {gamma}, gamma_theory: {gamma_theo}"
    assert np.all(
        np.isclose(gamma, gamma_rig, atol=1e-6)
    ), f"gamma: {gamma}, gamma of transformed system: {gamma_rig}"
    assert np.all(
        np.isclose(gamma_rig, gamma_theo_rig, atol=1e-6)
    ), f"gamma of transformed system: {gamma_rig}, gamma_theory: {gamma_theo_rig}"


def test_rotating_plate_dyn(show_plot=False):
    sol, _, _ = rotating_plate(
        np.eye(3), np.zeros(3), constrained=False, blender_export=False
    )
    A_rig = Exp_SO3(np.random.rand(3))
    r_rig = np.random.rand(3)
    sol_rig, _, _ = rotating_plate(
        A_rig, r_rig, constrained=False, blender_export=False
    )

    r_OBall = sol.q[:, :3].T
    r_OBall_rig = A_rig.T @ (sol_rig.q[:, :3] - r_rig).T

    if show_plot:
        fig, ax = plt.subplots(3, 1, squeeze=False)
        ax[0, 0].plot(sol.t, r_OBall[0], label="ez^Plane up")
        ax[1, 0].plot(sol.t, r_OBall[1], label="ez^Plane up")
        ax[2, 0].plot(sol.t, r_OBall[2], label="ez^Plane up")

        ax[0, 0].plot(sol_rig.t, r_OBall_rig[0], "--", label="rigidly transformed")
        ax[1, 0].plot(sol_rig.t, r_OBall_rig[1], "--", label="rigidly transformed")
        ax[2, 0].plot(sol_rig.t, r_OBall_rig[2], "--", label="rigidly transformed")
        ax[0, 0].set_ylabel("Ball position x")
        ax[1, 0].set_ylabel("Ball position y")
        ax[2, 0].set_ylabel("Ball position z")
        [(axi.legend(), axi.grid()) for axi in ax.flatten()]
        plt.show()

    assert np.all(
        np.isclose(r_OBall, r_OBall_rig, atol=1e-6)
    ), "Position of ball is not the same in both systems!"


def rotating_plate(A_rig, r_rig, constrained=True, blender_export=False):
    radius = 0.5
    anisotropy = np.array([1.0, 0.8])

    omega = 2 * np.pi * 0.5
    vx_rel = 0.4
    vy_rel = 0.2

    z = radius + (0.0 if constrained else 0.2)

    offset = lambda t: np.array([0.1 + vx_rel * t, 0.3 + vy_rel * t, z])
    offset_dot = lambda t: np.array([vx_rel, vy_rel, 0.0])

    system = System()
    A_IB = lambda t: A_rig @ A_IB_basic(omega * t).z
    floor = Box(Frame)(
        dimensions=[8, 8, 0.0001],
        r_OP=r_rig,
        A_IB=A_IB,
        name="floor",
    )

    A_IB0 = A_IB(0.0)
    r_OP_ball = lambda t: r_rig + A_IB0 @ offset(t)
    v_P_ball = lambda t: A_IB0 @ offset_dot(t)

    frame_ball = Frame(
        r_OP=r_OP_ball,
        v_P=v_P_ball,
        A_IB=A_IB0,
    )

    q0_ball = RigidBody.pose2q(r_OP_ball(0.0), A_IB0)
    u0_ball = np.array([*v_P_ball(0.0), 0.0, 0.0, 0.0])
    ball = Sphere(RigidBody)(
        radius=radius,
        mass=1.0,
        B_Theta_C=np.diag([1.0, 1.0, 1.0]),
        q0=q0_ball,
        u0=u0_ball,
    )
    ball_constraint = RigidConnection(ball, frame_ball)

    F_gravity = lambda t: A_rig @ np.array([0.0, 0.0, -9.81 * ball.mass])
    gravity = Force(F_gravity, ball)

    contact = Sphere2Plane(floor, ball, mu=1.0, radius=radius, anisotropy=anisotropy)

    system.add(floor, frame_ball, ball, contact, gravity)
    if constrained:
        system.add(ball_constraint)

    system.assemble()

    # solver
    solver = Moreau(system, 3.0, 1.0e-2)
    sol = solver.solve()

    if type(blender_export) == str:
        # export to blender
        dir_name = Path(__file__).parent
        system.export_blender(
            dir_name, f"blender_rotating_plate_{blender_export}", sol, create_blend=True
        )

    # compute relative velocities
    gamma = np.zeros((len(sol.t), 2))
    gamma_theo = np.zeros((len(sol.t), 2))
    for i, (ti, qi, ui) in enumerate(zip(sol.t, sol.q, sol.u)):
        # from contact
        gamma[i] = contact.gamma_F(ti, qi, ui)

        # theoretical value
        r_OJ2 = A_IB0 @ offset(ti)
        v_J2 = A_IB0 @ offset_dot(ti)
        Bi_Omega1 = np.array([0.0, 0.0, omega])
        Bi_r_OJ2_dot = A_IB(ti).T @ v_J2 - np.cross(Bi_Omega1, A_IB(ti).T @ r_OJ2)

        gamma_theo[i] = anisotropy * Bi_r_OJ2_dot[:2]

    return sol, gamma, gamma_theo


if __name__ == "__main__":
    test_implementation()
    test_rotating_plate_kin(show_plot=True)
    test_rotating_plate_dyn(show_plot=True)
    run(Moreau)
    run(BackwardEuler)
