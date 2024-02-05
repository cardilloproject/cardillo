import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import RigidBody, Frame, Meshed
from cardillo.solver import ScipyIVP
from cardillo.math import Spurrier, A_IK_basic, cross3
from cardillo.constraints import Revolute
from cardillo.forces import Force

if __name__ == "__main__":
    # read directory name of this file
    dir_name = Path(__file__).parent

    # gravitational acceleration
    g = np.array([0, 0, -10])

    # initial conditions
    phi10 = np.pi + np.pi / 6
    phi20 = np.pi + np.pi / 4
    phi1_dot0 = 0
    phi2_dot0 = 0

    # simulation parameters
    t1 = 2  # final time

    # initialize system
    system = System()

    ###########
    # base link
    ###########
    base_link = Meshed(Frame)(
        mesh_obj=Path(dir_name, "stl", "base_link.stl"),
        K_r_SP=np.array([-3.0299e-03, -2.6279e-13, -2.9120e-02]),
        A_KM=np.eye(3),
        r_OP=np.zeros(3),
        A_IK=np.eye(3),
        name="base_link",
    )

    ########
    # link 1
    ########

    # position of joint 1
    r_OJ1 = np.array([3.0573e-03, -2.6279e-13, 5.8800e-03])

    # kinematics of link 1
    K1_r_J1S1 = np.array([8.6107e-03, 2.1727e-06, 3.6012e-02])
    A_IK1 = A_IK_basic(phi10).x()
    r_J1S1 = A_IK1 @ K1_r_J1S1
    r_OS1 = r_OJ1 + r_J1S1

    K_Omega1 = np.array([phi1_dot0, 0, 0])
    v_S1 = A_IK1 @ cross3(K_Omega1, K1_r_J1S1)

    q0 = np.hstack([r_OS1, Spurrier(A_IK1)])
    u0 = np.hstack([v_S1, K_Omega1])

    link1 = Meshed(RigidBody)(
        mesh_obj=Path(dir_name, "stl", "link1.stl"),
        K_r_SP=np.array([-8.6107e-03, -2.1727e-06, -3.6012e-02]),
        A_KM=np.eye(3),
        mass=0.26703,
        K_Theta_S=np.array(
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

    #########
    # joint 1
    #########

    joint1 = Revolute(base_link, link1, axis=0, r_OB0=r_OJ1, name="joint1")

    ########
    # link 2
    ########

    # position of joint 2
    K1_r_J1J2 = np.array([2.3e-02, 7.05451171e-24, 1.0e-01])
    r_OJ2 = r_OJ1 + A_IK1 @ K1_r_J1J2
    v_J2 = A_IK1 @ cross3(K_Omega1, K1_r_J1J2)

    # kinematics of link 2
    K2_r_J2S2 = np.array([-5.0107e-03, 1.9371e-10, 1.0088e-01])
    A_IK2 = A_IK_basic(phi20).x()
    r_J2S2 = A_IK2 @ K2_r_J2S2
    r_OS2 = r_OJ2 + r_J2S2

    K_Omega2 = np.array([phi2_dot0, 0, 0])
    v_S2 = v_J2 + A_IK2 @ cross3(K_Omega2, K2_r_J2S2)

    q0 = np.hstack([r_OS2, Spurrier(A_IK2)])
    u0 = np.hstack([v_S2, K_Omega2])

    link2 = Meshed(RigidBody)(
        mesh_obj=Path(dir_name, "stl", "link2.stl"),
        K_r_SP=np.array([5.0107e-03, -1.9371e-10, -1.0088e-01]),
        A_KM=np.eye(3),
        mass=0.33238,
        K_Theta_S=np.array(
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

    #########
    # joint 2
    #########

    joint2 = Revolute(link1, link2, axis=0, r_OB0=r_OJ2, name="joint2")

    # add all contributions and assemble system
    system.add(base_link, link1, joint1, gravity1, link2, gravity2, joint2)
    system.assemble()

    # simulation
    dt = 1.0e-2  # time step
    solver = ScipyIVP(system, t1, dt)  # create solver
    sol = solver.solve()  # simulate system
    t = sol.t
    q = sol.q
    u = sol.u

    system.export(dir_name, "vtk", sol)
    exit()
