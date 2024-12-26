import numpy as np
from cardillo_urdf.urdf import load_urdf

from cardillo import System
from cardillo.math import A_IB_basic
from pathlib import Path
from cardillo.solver import Rattle, Moreau, BackwardEuler
from cardillo.discrete import Box, Frame
from cardillo.contacts import Sphere2Plane
from cardillo.actuators import PDcontroller, Motor
from cardillo.visualization import Export
from cardillo.visualization.trimesh import animate_system

from cardillo.actuators._base import BaseActuator

from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from cardillo.definitions import IS_CLOSE_ATOL
from cardillo.solver._base import compute_I_F

if __name__ == "__main__":
    from os import path

    dir_name = path.dirname(__file__)

    PD_joint_controller = False

    # Method 1
    initial_config = {}
    initial_config["FR_hip_joint"] = -np.pi / 12  # np.pi / 2
    initial_config["FR_thigh_joint"] = np.pi / 4
    initial_config["FR_calf_joint"] = -np.pi / 2

    initial_config["FL_hip_joint"] = np.pi / 12  # np.pi / 2
    initial_config["FL_thigh_joint"] = np.pi / 4
    initial_config["FL_calf_joint"] = -np.pi / 2

    initial_config["RL_hip_joint"] = np.pi / 12  # np.pi / 2
    initial_config["RL_thigh_joint"] = np.pi / 4
    initial_config["RL_calf_joint"] = -np.pi / 2

    initial_config["RR_hip_joint"] = -np.pi / 12  # np.pi / 2
    initial_config["RR_thigh_joint"] = np.pi / 4
    initial_config["RR_calf_joint"] = -np.pi / 2

    joint_names = [name for name in initial_config]



    initial_vel = {}
    # initial_vel["world_trunk"] = np.array([0.5, 0.5, 0, 0, 0, 0])
    # initial_vel["joint2"] = 5

    # Method 2
    # initial_config = (np.pi / 2, np.pi/2)
    # initial_vel = (1, 1)
    system = System()
    load_urdf(
        system,
        path.join(
            dir_name,
            "urdf",
            "go1.urdf",
        ),
        r_OC0=np.array([0, 0, 0.3]),
        A_IC0=A_IB_basic(0).y,
        v_C0=np.array([0, 0, 0]),
        C0_Omega_0=np.array([0, 0, 0]),
        initial_config=initial_config,
        initial_vel=initial_vel,
        base_link_is_floating=False,
        gravitational_acceleration=np.array([0, 0, -10]),
    )

    radius = 0.022
    mu = 0.3
    foot_contact_FR = Sphere2Plane(system.origin, system.contributions_map["FR_foot"], mu=mu, r=radius)
    foot_contact_FL = Sphere2Plane(system.origin, system.contributions_map["FL_foot"], mu=mu, r=radius)
    foot_contact_RR = Sphere2Plane(system.origin, system.contributions_map["RR_foot"], mu=mu, r=radius)
    foot_contact_RL = Sphere2Plane(system.origin, system.contributions_map["RL_foot"], mu=mu, r=radius)
    system.add(foot_contact_FR, foot_contact_FL, foot_contact_RR, foot_contact_RL)

    if PD_joint_controller:
        kp = 50
        kd = 1
        for joint_name in joint_names:
            # controller = PDcontroller(system.contributions_map[joint_name], kp, kd, np.array([0, 0]))
            motor = PDcontroller(system.contributions_map[joint_name], kp, kd, np.array([initial_config[joint_name], 0]))
            motor.name = "PD_" + system.contributions_map[joint_name].name
            system.add(motor)


        # kp_hip = 30
        # system.contributions_map["PD_FR_hip_joint"].kp = kp_hip
        # system.contributions_map["PD_FL_hip_joint"].kp = kp_hip
        # system.contributions_map["PD_RR_hip_joint"].kp = kp_hip
        # system.contributions_map["PD_RL_hip_joint"].kp = kp_hip
        frq = 1.25
        angle_d_calf = -np.deg2rad(30)
        for joint_name in ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]:
            system.contributions_map["PD_" + joint_name].tau = lambda t: np.array([initial_config[joint_name] + angle_d_calf * np.sin(2 * np.pi * frq * t), angle_d_calf * 2 * np.pi * frq * np.cos(2 * np.pi * frq * t)])

            # angle_d_thigh = np.deg2rad(2)
            # for joint_name in ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]:
            #     system.contributions_map["PD_" + joint_name].tau = lambda t: np.array([initial_config[joint_name] + angle_d_thigh * np.sin(2 * np.pi * frq * t), angle_d_thigh * 2 * np.pi * frq * np.cos(2 * np.pi * frq * t)])
                
    

    plane = Box(Frame)(dimensions=[4, 4, 0.001], axis=2)
    plane.name = 'Plane'
    system.add(plane)
    system.assemble()
    # sol = Rattle(system, 0.5, 1e-2).solve()
    sol = Moreau(system, 0.25, 1e-3).solve()
    # from cardillo.solver import SolverOptions
    # sol = BackwardEuler(system, 1.5, 1e-2, options=SolverOptions(reuse_lu_decomposition=True)).solve()

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
            if hasattr(b, "export") and not ("contr" in b.name):
                print(f"exporting {b.name}")
                # print(f"b.r_OP(0, sol.q[0, b.qDOF]) = {b.r_OP(0, sol.q[1, b.qDOF])}")
                e.export_contr(b)
