from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.constraints import Revolute
from cardillo.discrete import RigidBody, Frame, Meshed
from cardillo.forces import Force
from cardillo.math import A_IB_basic, cross3
from cardillo.solver import ScipyIVP, ScipyDAE

if __name__ == "__main__":
    ############
    # parameters
    ############

    # gravitational acceleration
    g = np.array([0, 0, -10])

    # initial conditions
    phi10 = np.pi / 4
    phi20 = np.pi / 3
    phi1_dot0 = 0
    phi2_dot0 = 0

    # simulation parameters
    t1 = 2  # final time

    # initialize system
    system = System()

    ###########
    # base link
    ###########
    # read directory name of this file
    dir_name = Path(__file__).parent

    # create base link and add it to system
    base_link = Meshed(Frame)(
        mesh_obj=Path(dir_name, "stl", "base_link.stl"),
        B_r_CP=np.array([-3.0299e-03, -2.6279e-13, -2.9120e-02]),
        A_BM=np.eye(3),
        r_OP=np.zeros(3),
        A_IB=np.eye(3),
        name="base_link",
    )
    system.add(base_link)

    ########
    # link 1
    ########

    # position of joint 1
    r_OJ1 = np.array([3.0573e-03, -2.6279e-13, 5.8800e-03])

    # kinematics and initial conditions of link 1
    B1_r_J1C1 = np.array([8.6107e-03, 2.1727e-06, 3.6012e-02])
    A_IB1 = A_IB_basic(np.pi + phi10).x
    r_J1C1 = A_IB1 @ B1_r_J1C1
    r_OC1 = r_OJ1 + r_J1C1
    B1_Omega1 = np.array([phi1_dot0, 0, 0])
    v_C1 = A_IB1 @ cross3(B1_Omega1, B1_r_J1C1)

    q0 = RigidBody.pose2q(r_OC1, A_IB1)
    u0 = np.hstack([v_C1, B1_Omega1])

    # create link 1
    link1 = Meshed(RigidBody)(
        mesh_obj=Path(dir_name, "stl", "link1.stl"),
        B_r_CP=np.array([-8.6107e-03, -2.1727e-06, -3.6012e-02]),
        A_BM=np.eye(3),
        mass=0.26703,
        B_Theta_C=np.array(
            [
                [4.0827e-04, 1.2675e-09, 1.8738e-05],
                [1.2675e-09, 3.8791e-04, 3.5443e-08],
                [1.8738e-05, 3.5443e-08, 3.6421e-05],
            ]
        ),
        q0=q0,
        u0=u0,
        name="link1",
    )

    # gravity for link 1
    gravity1 = Force(link1.mass * g, link1, name="gravity_link1")

    # add contributions to the system
    system.add(link1, gravity1)

    #########
    # joint 1
    #########

    joint1 = Revolute(
        base_link, link1, axis=0, r_OJ0=r_OJ1, angle0=phi10, name="joint1"
    )
    system.add(joint1)

    ########
    # link 2
    ########

    # position of joint 2
    B1_r_J1J2 = np.array([2.3e-02, 7.05451171e-24, 1.0e-01])
    r_OJ2 = r_OJ1 + A_IB1 @ B1_r_J1J2
    v_J2 = A_IB1 @ cross3(B1_Omega1, B1_r_J1J2)

    # kinematics of link 2
    B2_r_J2C2 = np.array([-5.0107e-03, 1.9371e-10, 1.0088e-01])
    A_IB2 = A_IB_basic(np.pi + phi20).x
    r_J2C2 = A_IB2 @ B2_r_J2C2
    r_OC2 = r_OJ2 + r_J2C2

    B_Omega2 = np.array([phi2_dot0, 0, 0])
    v_C2 = v_J2 + A_IB2 @ cross3(B_Omega2, B2_r_J2C2)

    q0 = RigidBody.pose2q(r_OC2, A_IB2)
    u0 = np.hstack([v_C2, B_Omega2])

    link2 = Meshed(RigidBody)(
        mesh_obj=Path(dir_name, "stl", "link2.stl"),
        B_r_CP=np.array([5.0107e-03, -1.9371e-10, -1.0088e-01]),
        A_BM=np.eye(3),
        mass=0.33238,
        B_Theta_C=np.array(
            [
                [1.1753e-03, -3.8540e-13, -2.9304e-08],
                [-3.8540e-13, 1.1666e-03, -5.2365e-12],
                [-2.9304e-08, -5.2365e-12, 1.4553e-05],
            ]
        ),
        q0=q0,
        u0=u0,
        name="link2",
    )

    # gravity for link 2
    gravity2 = Force(link2.mass * g, link2, name="gravity_link2")

    # add contributions to the system
    system.add(link2, gravity2)

    #########
    # joint 2
    #########

    joint2 = Revolute(link1, link2, axis=0, r_OJ0=r_OJ2, angle0=phi20, name="joint2")
    system.add(joint2)

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    dt = 1.0e-2  # time step
    # solver = ScipyIVP(system, t1, dt)  # create solver
    solver = ScipyDAE(system, t1, dt)  # create solver
    sol = solver.solve()  # simulate system

    # read solution
    t = sol.t
    q = sol.q
    u = sol.u

    #################
    # post-processing
    #################

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    # plot time evolution for angles
    phi1 = [joint1.angle(ti, qi) for ti, qi in zip(t, q[:, joint1.qDOF])]
    phi2 = [joint2.angle(ti, qi) for ti, qi in zip(t, q[:, joint2.qDOF])]
    ax[0].plot(t, phi1, "-r", label="$\\varphi_1$")
    ax[0].plot(t, phi2, "-g", label="$\\varphi_2$")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("angle")
    ax[0].legend()
    ax[0].grid()

    # plot time evolution for angular velocities
    phi1_dot = [
        joint1.angle_dot(ti, qi, ui)
        for ti, qi, ui in zip(t, q[:, joint1.qDOF], u[:, joint1.uDOF])
    ]
    phi2_dot = [
        joint2.angle_dot(ti, qi, ui)
        for ti, qi, ui in zip(t, q[:, joint2.qDOF], u[:, joint2.uDOF])
    ]
    ax[1].plot(t, phi1_dot, "-r", label="$\\dot{\\varphi}_1$")
    ax[1].plot(t, phi2_dot, "-g", label="$\\dot{\\varphi}_2$")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("angular velocity")
    ax[1].legend()
    ax[1].grid()

    plt.show()

    # vtk-export
    system.export(dir_name, "vtk", sol)
