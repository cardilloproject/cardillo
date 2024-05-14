import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import RigidBody, Sphere, Box, Frame
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane, Sphere2Sphere
from cardillo.solver import Moreau, BackwardEuler, SolverOptions
from cardillo.math import A_IB_basic

if __name__ == "__main__":
    ############
    # parameters
    ############

    # radius of ball
    radius = 0.05

    # contact parameters
    e_N = 0  # restitution coefficient in normal direction
    e_F = 0.0  # restitution coefficient in tangent direction
    mu = 0.3  # frictional coefficient

    # density of ball
    density = 1

    # gravitational acceleration
    g = np.array([0, 0, -10])

    # initial conditions
    height = 8 * radius
    nx = 3
    ny = 3
    nz = 5
    offset = 0.3 * radius

    # simulation parameters
    t1 = 1  # final time

    # initialize system
    system = System()

    r_OC0 = np.array([0, 0, height])
    offset_x = np.array([offset + 2 * radius, 0, 0])
    offset_y = np.array([0, offset + 2 * radius, 0])
    offset_z = np.array([0, 0, offset + 2 * radius])

    u0 = np.zeros(6)

    balls = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r_OC = r_OC0 + i * offset_x + j * offset_y + k * offset_z
                q0 = RigidBody.pose2q(r_OC, np.eye(3))
                balls.append(
                    Sphere(RigidBody)(
                        radius=radius,
                        density=density,
                        subdivisions=3,
                        q0=q0,
                        u0=u0,
                        name="ball_" + str(i) + str(j) + str(k),
                    )
                )

    system.add(*balls)

    # floor
    floor = Box(Frame)(
        dimensions=[4.5, 4.5, 0.0001],
        name="floor",
        A_IB=A_IB_basic(np.deg2rad(10)).x @ A_IB_basic(np.deg2rad(10)).y,
    )
    system.add(floor)  # (only for visualization purposes)

    for ball in balls:
        system.add(Force(ball.mass * g, ball, name="gravity_" + ball.name))
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

    idx = 0
    while balls:
        for ball in balls[1:]:
            idx += 1
            system.add(
                Sphere2Sphere(
                    balls[0],
                    ball,
                    radius,
                    radius,
                    mu,
                    e_N=e_N,
                    e_F=e_F,
                    name=f"sphere2sphere contact{idx}",
                )
            )
        balls.pop(0)

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    dt = 1.0e-3  # time step
    solver = Moreau(
        system,
        t1,
        dt,
        options=SolverOptions(fixed_point_max_iter=int(1e4), prox_scaling=0.5),
    )  # create solver
    # solver = BackwardEuler(
    #     system,
    #     t1,
    #     dt,
    #     options=SolverOptions(prox_scaling=0.5),
    # )  # create solver
    sol = solver.solve()  # simulate system

    # vtk-export
    dir_name = Path(__file__).parent
    system.export(dir_name, "vtk", sol)
