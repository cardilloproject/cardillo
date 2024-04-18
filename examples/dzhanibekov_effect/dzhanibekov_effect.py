import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import trimesh

from cardillo.discrete import RigidBody, Meshed
from cardillo.system import System
from cardillo.solver import Rattle


if __name__ == "__main__":
    ############
    # parameters
    ############

    # initial conditions
    r_OC0 = np.zeros(3)
    v_C0 = np.zeros(3)
    phi_dot0 = 20
    B_Omega_disturbance = np.array((1e-10, 0, 0))  # disturbance is required
    B_Omega0 = np.array((0, 0, phi_dot0)) + B_Omega_disturbance

    # simulation parameters
    t1 = 10  # final time

    # initialize system
    system = System()

    #################
    # assemble system
    #################

    # load mesh using trimesh
    dir_name = Path(__file__).parent
    mesh = trimesh.load_mesh(Path(dir_name, "mesh", "screwdriver.stl"))
    scale = 1 / 1000  # m/mm
    mesh.apply_transform(np.diag([scale, scale, scale, 1]))

    # quantities in mesh/body-fixed basis
    B_r_PC = mesh.center_mass  # vector from mesh origin to center of mass
    mass = mesh.mass
    B_Theta_C = mesh.moment_inertia

    q0 = RigidBody.pose2q(r_OC0, np.eye(3))
    u0 = np.hstack([v_C0, B_Omega0])
    screwdriver = Meshed(RigidBody)(
        mesh_obj=mesh,
        B_r_CP=-B_r_PC,
        A_BM=np.eye(3),
        mass=mass,
        B_Theta_C=B_Theta_C,
        q0=q0,
        u0=u0,
        name="screwdriver",
    )
    system.add(screwdriver)

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    dt = 1e-2  # time step
    solver = Rattle(system, t1, dt)  # create solver
    sol = solver.solve()  # simulate system

    # read solution
    t = sol.t  # time
    q = sol.q  # position coordinates

    #################
    # post-processing
    #################
    r_OP = np.array([screwdriver.r_OP(ti, qi) for (ti, qi) in zip(t, q)])
    e_zB = np.array([screwdriver.A_IB(ti, qi)[:, 2] for (ti, qi) in zip(t, q)])

    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    ax[0].set_title("Evolution of center of mass")
    ax[0].plot(t, r_OP[:, 0], label="x")
    ax[0].plot(t, r_OP[:, 1], label="y")
    ax[0].plot(t, r_OP[:, 2], label="z")
    ax[0].set_xlabel("t")
    ax[0].grid()
    ax[0].legend()

    ax[1].set_title("Evolution of body-fixed z-axis")
    ax[1].plot(t, e_zB[:, 0], label="$(e_{z}^B)_x$")
    ax[1].plot(t, e_zB[:, 1], label="$(e_{z}^B)_y$")
    ax[1].plot(t, e_zB[:, 2], label="$(e_{z}^B)_z$")
    ax[1].set_xlabel("t")
    ax[1].grid()
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # vtk-export
    dir_name = Path(__file__).parent
    system.export(dir_name, "vtk", sol)
