import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import RigidBody, Frame, Meshed
from cardillo.solver import ScipyIVP
from cardillo.math import Spurrier, A_IK_basic, cross3

if __name__ == "__main__":
    # read directory name of this file
    dir_name = Path(__file__).parent

    # initial conditions
    phi10 = 0
    phi20 = 0
    phi1_dot0 = 0
    phi2_dot0 = 0

    # simulation parameters
    t1 = 0.1  # final time

    # initialize system
    system = System()

    base_link = Meshed(Frame)(Path(dir_name, "stl", "base_link.stl"))

    K1_r_OS1 = np.array([1.16680000e-02, 2.17269974e-06, 3.58048000e-02])
    A_IK1 = A_IK_basic(phi10).x()
    r_OS1 = A_IK1 @ K1_r_OS1

    K_Omega1 = np.array([phi1_dot0, 0, 0])
    v_S1 = A_IK1 @ cross3(K_Omega1, K1_r_OS1)

    q0 = np.hstack([r_OS1, Spurrier(A_IK1)])
    u0 = np.hstack([v_S1, K_Omega1])

    link1 = Meshed(RigidBody)(
        mesh_obj=Path(dir_name, "stl", "link1.stl"),
        # K_r_SP=np.array([-8.6107e-03, -2.1727e-06, -3.6012e-02]),
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
    )
    system.add(base_link, link1)
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
