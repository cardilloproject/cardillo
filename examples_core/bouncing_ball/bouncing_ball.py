from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import RigidBody, Sphere, Box, Frame
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane
from cardillo.math import A_IK_basic, cross3
from cardillo.solver import Moreau

if __name__ == "__main__":
    ############
    # parameters
    ############
    
    # radius of ball
    radius = 0.05

    # contact parameters
    e_N = 0.75 # restitution coefficient in normal direction
    e_F = 0.0 # restitution coefficient in tangent direction
    mu = 0.5 # frictional coefficient

    # density of ball
    density = 1

    # gravitational acceleration
    g = np.array([0, 0, -10])

    # initial conditions
    r_OS0 = np.array([-.75, 0, 8 * radius])
    v_S0 = np.array([1, 0, 0])
    K_Omega0 = np.array([0, -25, 0])

    # simulation parameters
    t1 = 3  # final time

    # initialize system
    system = System()

    #####
    # top
    #####
    
    q0 = RigidBody.pose2q(r_OS0, np.eye(3))
    u0 = np.hstack([v_S0, K_Omega0])

    ball = Sphere(RigidBody)(
        radius=radius,
        density=density,
        subdivisions=3,
        q0=q0,
        u0=u0,
        name="ball",
    )
    gravity = Force(ball.mass * g, ball, name="gravity")
    system.add(ball, gravity)

    # floor (only for visualization purposes)
    floor = Box(Frame)(
        dimensions=[1.5, 4 * radius, 0.0001],
        name="floor",
    )

    tip2plane = Sphere2Plane(floor, ball, mu=mu, r=radius, e_N=e_N, e_F=e_F)
    system.add(floor, tip2plane)

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    dt = 2.0e-3  # time step
    solver = Moreau(system, t1, dt)  # create solver
    sol = solver.solve()  # simulate system

    # read solution
    t = sol.t
    q = sol.q
    u = sol.u

    #################
    # post-processing
    #################

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    # plot time evolution for x-coordinate
    x = [ball.r_OP(ti, qi)[0] for ti, qi in zip(t, q[:, ball.qDOF])]
    # TODO: can we plot the rotation angle  around y-axis?
    ax[0, 0].plot(t, x, "-r", label="$x$")
    ax[0, 0].set_title("Evolution of horizontal position")
    ax[0, 0].set_xlabel("t")
    ax[0, 0].set_ylabel("x")
    ax[0, 0].grid()

    # plot time evolution for z-coordinate
    z = [ball.r_OP(ti, qi)[2] for ti, qi in zip(t, q[:, ball.qDOF])]
    ax[0, 1].plot(t, z, "-g", label="$z$")
    ax[0, 1].set_title("Evolution of height")
    ax[0, 1].set_xlabel("t")
    ax[0, 1].set_ylabel("z")
    ax[0, 1].grid()

    # plot time evolution of x-velocity
    v_x = [ball.v_P(ti, qi, ui)[0] for ti, qi, ui in zip(t, q[:, ball.qDOF], u[:, ball.uDOF])]
    # TODO: can we plot the rotation angle  around y-axis?
    ax[1, 0].plot(t, v_x, "-r", label="$v_x$")
    ax[1, 0].set_title("Evolution of horizontal velocity")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].set_ylabel("v_x")
    ax[1, 0].grid()

    # plot time evolution of z-velocity
    v_z = [ball.v_P(ti, qi, ui)[2] for ti, qi, ui in zip(t, q[:, ball.qDOF], u[:, ball.uDOF])]
    ax[1, 1].plot(t, v_z, "-g", label="$z$")
    ax[1, 1].set_title("Evolution of vertical velocity")
    ax[1, 1].set_xlabel("t")
    ax[1, 1].set_ylabel("v_z")
    ax[1, 1].grid()

    plt.tight_layout()
    plt.show()

    # vtk-export
    dir_name = Path(__file__).parent
    system.export(dir_name, "vtk", sol)
