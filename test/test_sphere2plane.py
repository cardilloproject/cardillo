import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import RigidBody, Box, Sphere, Frame, Tetrahedron
from cardillo.forces import Force
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.interactions import TwoPointInteraction
from cardillo.contacts import Sphere2Plane, Sphere2Sphere
from cardillo.solver import Moreau, BackwardEuler, SolverOptions
from cardillo.math import A_IB_basic


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
            r=radius,
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
                r=0,
                e_N=e_N,
                e_F=e_F,
                B_r_CP=vertex,
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


if __name__ == "__main__":
    run(Moreau)
    run(BackwardEuler)
