import numpy as np
from cardillo_urdf.urdf import load_urdf

from cardillo import System
from cardillo.math import A_IB_basic
from pathlib import Path
from cardillo.solver import Rattle, Moreau, BackwardEuler
from cardillo.discrete import Box, Frame
from cardillo.contacts import Sphere2Plane
from cardillo.actuators import PDcontroller, Motor
from cardillo.visualization import Export, Renderer
from cardillo.visualization.trimesh import show_system

from cardillo.actuators._base import BaseActuator

from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from cardillo.definitions import IS_CLOSE_ATOL
from cardillo.solver._base import compute_I_F

if __name__ == "__main__":
    from os import path

    dir_name = path.dirname(__file__)

    PD_joint_controller = False
    virtual_model_controller = True

    # Method 1
    initial_config = {}
    initial_config["panda_joint1"] = 0.0
    initial_config["panda_joint2"] = 1.0
    initial_config["panda_joint3"] = 0.0
    initial_config["panda_joint4"] = -1.5
    initial_config["panda_joint5"] = 0.0
    initial_config["panda_joint6"] = 2
    initial_config["panda_joint7"] = 0.0
    initial_config["panda_finger_joint1"] = 0.0

    joint_names = list(initial_config.keys())[:-1]



    initial_vel = {}
    # initial_vel["world_trunk"] = np.array([0.5, 0.5, 0, 0, 0, 0])
    initial_vel["panda_joint2"] = 0.0

    # Method 2
    # initial_config = (np.pi / 2, np.pi/2)
    # initial_vel = (1, 1)
    system = System()
    load_urdf(
        system,
        path.join(
            dir_name,
            "urdf",
            "panda.urdf",
        ),
        r_OC0=np.array([0, 0, 0]),
        A_IC0=A_IB_basic(0).y,
        v_C0=np.array([0, 0, 0]),
        C0_Omega_0=np.array([0, 0, 0]),
        initial_config=initial_config,
        initial_vel=initial_vel,
        base_link_is_floating=False,
        gravitational_acceleration=np.array([0, 0, -10]),
        redundant_coordinates=False,
    )
    # show_system(system, 0, system.q0)

    if PD_joint_controller:
        kp = 5000
        kd = 1
        for joint_name in joint_names:
            # controller = PDcontroller(system.contributions_map[joint_name], kp, kd, np.array([0, 0]))
            motor = PDcontroller(system.contributions_map[joint_name], kp, kd, np.array([initial_config[joint_name], 0]))
            motor.name = "PD_" + system.contributions_map[joint_name].name
            system.add(motor)
    

    system.assemble() 
    render = Renderer(system, [system.contributions_map.values, system.origin])
    solver = Rattle(system, 0.5, 1e-2).solve()
    #sol = Moreau(system, 2, 1e-2).solve()
    # from cardillo.solver import SolverOptions
    # sol = BackwardEuler(system, 1.5, 1e-2, options=SolverOptions(reuse_lu_decomposition=True)).solve()
    render.render_solution(solver, repeat=True)

    # animate_system(system, sol.t, sol.q)
    if True:
        path = Path(__file__)
        e = Export(
            path=path.parent,
            folder_name=path.stem,
            overwrite=True,
            fps=30,
            solution=sol,
        )

        for b in system.contributions:
            if hasattr(b, "export") and not ("gravity" in b.name):
                print(f"exporting {b.name}")
                # print(f"b.r_OP(0, sol.q[0, b.qDOF]) = {b.r_OP(0, sol.q[1, b.qDOF])}")
                e.export_contr(b)
