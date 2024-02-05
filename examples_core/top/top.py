from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import trimesh

from cardillo import System
from cardillo.discrete import RigidBody, Frame, Meshed, Box
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane
from cardillo.math import Spurrier, A_IK_basic, cross3
from cardillo.solver import Moreau, BackwardEuler

if __name__ == "__main__":
    ############
    # parameters
    ############

    # gravitational acceleration
    g = np.array([0, 0, -10])

    # initial conditions
    phi10 = np.pi / 6
    phi20 = np.pi / 4
    phi1_dot0 = 0
    phi2_dot0 = 0

    # simulation parameters
    t1 = 3  # final time

    # initialize system
    system = System()

    #####
    # top
    #####

    # load mesh using trimesh
    dir_name = Path(__file__).parent
    top_mesh = trimesh.load_mesh(Path(dir_name, "mesh", "top.stl"))
    scale = 1/1000 # m/mm
    top_mesh.apply_transform(np.diag([scale, scale, scale, 1]))

    top_mesh.density = 8850 #kg/m3 (Copper)

    # quantities in mesh/body fixed frame
    K_r_PS = top_mesh.center_mass # vector from mesh origin to center of mass
    mass = top_mesh.mass
    K_Theta_S = top_mesh.moment_inertia
    
    tip_radius = 1e-3 # 1mm
    height = 0.0001
    A_IK = np.eye(3)
    r_OS = np.array([0, 0, tip_radius + height]) + A_IK @ K_r_PS

    v_S = np.zeros(3)
    K_Omega = np.array([0, 0, 200])
    
    q0 = np.hstack([r_OS, Spurrier(A_IK)])
    u0 = np.hstack([v_S, K_Omega])

    top = Meshed(RigidBody)(
        mesh_obj=top_mesh,
        K_r_SP=-K_r_PS,
        A_KM=np.eye(3), 
        mass=mass,
        K_Theta_S=K_Theta_S,
        q0=q0,
        u0=u0,
        name="top",
    )
    gravity = Force(top.mass * g, top, name="gravity_top")
    system.add(top, gravity)

    # floor (only for visualization purposes)
    floor = Box(Frame)(
        dimensions=[1, 1, 0.0001],
        name="floor",
    )

    tip2plane = Sphere2Plane(floor, top, mu=0.01, r=tip_radius, e_N=0, K_r_SP=-K_r_PS)
    system.add(floor, tip2plane)
    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    dt = 2.0e-3  # time step
    solver = BackwardEuler(system, t1, dt)  # create solver
    sol = solver.solve()  # simulate system

    # read solution
    t = sol.t
    q = sol.q
    u = sol.u

    #################
    # post-processing
    #################

    # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    
    # # plot time evolution for angles 
    # phi1 = [joint1.angle(ti, qi) for ti, qi in zip(t, q[:, joint1.qDOF])]
    # phi2 = [joint2.angle(ti, qi) for ti, qi in zip(t, q[:, joint2.qDOF])]
    # ax[0].plot(t, phi1, "-r", label="$\\varphi_1$")
    # ax[0].plot(t, phi2, "-g", label="$\\varphi_2$")
    # ax[0].set_xlabel("t")
    # ax[0].set_ylabel("angle")
    # ax[0].legend()
    # ax[0].grid()

    # # plot time evolution for angular velocities
    # phi1_dot = [joint1.angle_dot(ti, qi, ui) for ti, qi, ui in zip(t, q[:, joint1.qDOF], u[:, joint1.uDOF])]
    # phi2_dot = [joint2.angle_dot(ti, qi, ui) for ti, qi, ui in zip(t, q[:, joint2.qDOF], u[:, joint2.uDOF])]
    # ax[1].plot(t, phi1_dot, "-r", label="$\\dot{\\varphi}_1$")
    # ax[1].plot(t, phi2_dot, "-g", label="$\\dot{\\varphi}_2$")
    # ax[1].set_xlabel("t")
    # ax[1].set_ylabel("angular velocity")
    # ax[1].legend()
    # ax[1].grid()

    # plt.show()

    # vtk-export
    system.export(dir_name, "vtk", sol)

