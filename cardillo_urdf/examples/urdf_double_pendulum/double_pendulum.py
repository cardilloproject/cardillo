import numpy as np
from cardillo_urdf.urdf import load_urdf

from cardillo.math import A_IB_basic
from pathlib import Path
from cardillo import System
from cardillo.actuators import PDcontroller
from cardillo.solver import Rattle, BackwardEuler
from cardillo.visualization import Export, Renderer
from cardillo.visualization.trimesh import animate_system, show_system

if __name__ == "__main__":
    from os import path

    dir_name = path.dirname(__file__)

    # Method 1
    initial_config = {}
    initial_config["joint1"] = 0 # np.pi / 2
    initial_config["joint2"] = 0 #np.pi / 2

    initial_vel = {}
    initial_vel["joint1"] = 0
    initial_vel["joint2"] = 0

    # Method 2
    # initial_config = (np.pi / 2, np.pi/2)
    # initial_vel = (1, 1)

    system = System()

    load_urdf(
        system,
        path.join(
            dir_name,
            "urdf",
            "double_pendulum.urdf",
        ),
        r_OC0=np.array([0, 0, 0]),
        A_IC0=A_IB_basic(0).y @ A_IB_basic(np.pi).x,
        v_C0=np.zeros(3),
        C0_Omega_0=np.array([0, 0, 0]),
        initial_config=initial_config,
        initial_vel=initial_vel,
        


        gravitational_acceleration=np.array([0, -1, 0]),
    )
    show_system(system, 0, system.q0)
    kp = 2
    kd = 1
    for joint_name in initial_config:
        motor = PDcontroller(system.contributions_map[joint_name], kp, kd, np.array([initial_config[joint_name], 0]))
        motor.name = "PD_" + system.contributions_map[joint_name].name
        system.add(motor)

    system.contributions_map["PD_joint1"].tau = lambda t: np.array([np.pi/2, 0])
    system.contributions_map["PD_joint2"].tau = lambda t: np.array([np.pi/4, 0])
    system.assemble()

    render = Renderer(system, system.contributions)
    sol = BackwardEuler(system, 5, 1e-2).solve()
    render.render_solution(sol, repeat=True)
    # show_system(system, sol.t[5], sol.q[5])
    # animate_system(system, sol.t, sol.q, t_factor=1)
    path = Path(__file__)
    e = Export(
        path=path.parent,
        folder_name=path.stem,
        overwrite=True,
        fps=30,
        solution=sol,
    )

    for b in system.contributions:
        if hasattr(b, "export"):
            print(f"exporting {b.name}")
            print(f"b.r_OP(0, sol.q[0, b.qDOF]) = {b.r_OP(0, sol.q[1, b.qDOF])}")
            e.export_contr(b)
