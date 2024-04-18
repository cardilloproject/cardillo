import numpy as np
from pathlib import Path
import trimesh

from cardillo import System
from cardillo.discrete import RigidBody, Frame, Meshed, Box
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane
from cardillo.math import A_IB_basic, cross3
from cardillo.solver import Moreau

if __name__ == "__main__":
    ############
    # parameters
    ############

    # gravitational acceleration
    g = np.array([0, 0, -10])

    # initial conditions
    phi0 = np.deg2rad(5)  # inclination angle
    omega_z0 = 200  # angular velocity in e_z^K-direction

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
    scale = 1 / 1000  # m/mm
    top_mesh.apply_transform(np.diag([scale, scale, scale, 1]))

    top_mesh.density = 8850  # kg/m3 (Copper)

    # quantities in mesh/body-fixed basis
    B_r_PC = top_mesh.center_mass  # vector from mesh origin to center of mass
    mass = top_mesh.mass
    B_Theta_C = top_mesh.moment_inertia

    tip_radius = 1e-3  # 1mm
    A_IB = A_IB_basic(phi0).y
    r_OC = np.array([0, 0, tip_radius]) + A_IB @ B_r_PC

    B_Omega = np.array([0, 0, omega_z0])
    v_C = cross3(A_IB @ B_Omega, r_OC)

    q0 = RigidBody.pose2q(r_OC, A_IB)
    u0 = np.hstack([v_C, B_Omega])

    top = Meshed(RigidBody)(
        mesh_obj=top_mesh,
        B_r_CP=-B_r_PC,
        A_BM=np.eye(3),
        mass=mass,
        B_Theta_C=B_Theta_C,
        q0=q0,
        u0=u0,
        name="top",
    )
    gravity = Force(top.mass * g, top, name="gravity_top")
    system.add(top, gravity)

    # floor (only for visualization purposes)
    floor = Box(Frame)(
        dimensions=[0.1, 0.1, 0.0001],
        name="floor",
    )

    tip2plane = Sphere2Plane(floor, top, mu=0.01, r=tip_radius, e_N=0, B_r_CP=-B_r_PC)
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

    # vtk-export
    system.export(dir_name, "vtk", sol)
