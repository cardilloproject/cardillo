#############
# description
#############
# 3-dimensional ball subject to gravity. The ball can come into contact with x-y-plane.

from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import RigidBody, Sphere, Box, Frame
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane
from cardillo.solver import Moreau

if __name__ == "__main__":
    ############
    # parameters
    ############

    # radius of ball
    radius = 0.05

    # contact parameters
    e_N = 0.75  # restitution coefficient in normal direction
    e_F = 0.0  # restitution coefficient in tangent direction
    mu = 0.5  # frictional coefficient

    # density of ball
    density = 1

    # gravitational acceleration
    g = np.array([0, 0, -10])

    # initial conditions
    r_OC0 = np.array([-0.75, 0, 8 * radius])  # initial position of c.o.m.
    v_C0 = np.array([1, 0, 0])  # initial velocity of c.o.m.
    B_Omega0 = np.array([0, -25, 0])  # initial angular velocity

    # simulation parameters
    t1 = 3  # final time

    # initialize system
    system = System()

    #################
    # assemble system
    #################

    # create ball
    q0 = RigidBody.pose2q(r_OC0, np.eye(3))
    u0 = np.hstack([v_C0, B_Omega0])

    ball = Sphere(RigidBody)(
        radius=radius,
        density=density,
        subdivisions=3,
        q0=q0,
        u0=u0,
        name="ball",
    )
    # gravitational force for ball
    gravity = Force(ball.mass * g, ball, name="gravity")
    # add ball and gravitational force to system
    system.add(ball, gravity)

    # create floor (Box only for visualization purposes)
    floor = Box(Frame)(
        dimensions=[1.5, 4 * radius, 0.0001],
        name="floor",
    )

    # add contact between ball and floor
    ball2plane = Sphere2Plane(floor, ball, mu=mu, r=radius, e_N=e_N, e_F=e_F)
    system.add(floor, ball2plane)

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    dt = 2.0e-3  # time step
    solver = Moreau(system, t1, dt)  # create solver
    sol = solver.solve()  # simulate system

    # read solution
    t = sol.t  # time
    q = sol.q  # position coordinates
    u = sol.u  # velocity coordinates
    P_N = sol.P_N  # discrete percussions in normal direction
    P_F = sol.P_F  # discrete percussions of friction

    #################
    # post-processing
    #################

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))
    # plot time evolution for x-coordinate
    x = [ball.r_OP(ti, qi)[0] for ti, qi in zip(t, q[:, ball.qDOF])]
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

    # plot time evolution of rotation angle around y-axis
    phi_dot = np.array(
        [
            ball.B_Omega(ti, qi, ui)[1]
            for ti, qi, ui in zip(t, q[:, ball.qDOF], u[:, ball.uDOF])
        ]
    )
    phi = np.cumsum(phi_dot * dt)
    ax[0, 2].plot(t, phi, "-g", label="$\varphi$")
    ax[0, 2].set_title("Evolution of rotation angle around y-axis")
    ax[0, 2].set_xlabel("t")
    ax[0, 2].set_ylabel("angle")
    ax[0, 2].grid()

    # plot time evolution of x-velocity
    v_x = [
        ball.v_P(ti, qi, ui)[0]
        for ti, qi, ui in zip(t, q[:, ball.qDOF], u[:, ball.uDOF])
    ]
    ax[1, 0].plot(t, v_x, "-r", label="$v_x$")
    ax[1, 0].set_title("Evolution of horizontal velocity")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].set_ylabel("v_x")
    ax[1, 0].grid()

    # plot time evolution of z-velocity
    v_z = [
        ball.v_P(ti, qi, ui)[2]
        for ti, qi, ui in zip(t, q[:, ball.qDOF], u[:, ball.uDOF])
    ]
    ax[1, 1].plot(t, v_z, "-g", label="$z$")
    ax[1, 1].set_title("Evolution of vertical velocity")
    ax[1, 1].set_xlabel("t")
    ax[1, 1].set_ylabel("v_z")
    ax[1, 1].grid()

    # plot time evolution of angular velocity around y-axis
    ax[1, 2].plot(t, phi_dot, "-g", label=r"$\dot{\varphi}$")
    ax[1, 2].set_title("Evolution of angular velocity")
    ax[1, 2].set_xlabel("t")
    ax[1, 2].set_ylabel("v_phi")
    ax[1, 2].grid()

    plt.tight_layout()
    plt.show()

    # second figure plotting percussions
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    # plot time evolution for x-coordinate
    x = [ball.r_OP(ti, qi)[0] for ti, qi in zip(t, q[:, ball.qDOF])]
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

    # plot time evolution of discrete friction percussion
    P_Fx = P_F[:, ball2plane.la_FDOF[0]]
    ax[1, 0].plot(t, P_Fx, "-r", label="$P_Fx$")
    ax[1, 0].set_title("Evolution of discrete friction percussion")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].set_ylabel("P_Fx")
    ax[1, 0].grid()

    # plot time evolution of discrete normal percussion
    # TODO: How do we name this thing? incremental normal percussion?
    ax[1, 1].plot(t, P_N, "-g", label="$P_N$")
    ax[1, 1].set_title("Evolution of discrete normal percussion")
    ax[1, 1].set_xlabel("t")
    ax[1, 1].set_ylabel("P_N")
    ax[1, 1].grid()

    plt.tight_layout()
    plt.show()

    # vtk-export
    dir_name = Path(__file__).parent
    system.export(dir_name, "vtk", sol)
