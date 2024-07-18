import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import trimesh

from cardillo.discrete import RigidBody, Meshed
from cardillo.system import System
from cardillo.solver import Rattle
from cardillo.visualization import Renderer


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
        mesh_obj=Path(dir_name, "mesh", "screwdriver.stl"),
        scale=1e-3,
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

    ###################
    # initialize render
    ###################
    render = Renderer(system)
    render.start_step_render(sync=False)

    ############
    # simulation
    ############
    dt = 1e-2  # time step
    solver = Rattle(system, t1, dt)  # create solver
    sol = solver.solve()  # simulate system
    
    render.stop_step_render()
    render.render_solution(sol, repeat=True)
    render.start_interaction(sol.t[-1], sol.q[-1], sol.u[-1])

