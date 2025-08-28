# from urdf_parser_py.urdf import URDF

# robot = URDF.from_xml_file("examples/urdf_unitree_go1/urdf/go1.urdf")

# exit()

import numpy as np


from cardillo import System
from cardillo.urdf import system_from_urdf
from cardillo.math import A_IB_basic
from pathlib import Path
from cardillo.solver import Moreau
from cardillo.discrete import Box, Frame
from cardillo.contacts import Sphere2Plane
from cardillo.visualization import Renderer

if __name__ == "__main__":
    from os import path

    dir_name = path.dirname(__file__)

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

    system = system_from_urdf(
        file_path=Path(dir_name, "urdf", "go1.urdf"),
        configuration=initial_config,
        gravitational_acceleration=np.array([0, 0, -9.81]),
    )

    render = Renderer(system)
    if simulate := False:
        sol = Moreau(system, 0.25, 1e-3).solve()
        render.render_solution(
            sol, repeat=True
        )
    else:
        render.render_solution(
            Solution(system, np.array([system.t0]), np.asanyarray([system.q0]), np.asanyarray([system.u0])), repeat=True
        )
    exit()



    radius = 0.022
    mu = 0.3
    foot_contact_FR = Sphere2Plane(
        system.origin, system.contributions_map["FR_foot"], mu=mu, r=radius
    )
    foot_contact_FL = Sphere2Plane(
        system.origin, system.contributions_map["FL_foot"], mu=mu, r=radius
    )
    foot_contact_RR = Sphere2Plane(
        system.origin, system.contributions_map["RR_foot"], mu=mu, r=radius
    )
    foot_contact_RL = Sphere2Plane(
        system.origin, system.contributions_map["RL_foot"], mu=mu, r=radius
    )
    system.add(foot_contact_FR, foot_contact_FL, foot_contact_RR, foot_contact_RL)

    if PD_joint_controller:
        kp = 50
        kd = 1
        for joint_name in joint_names:
            # controller = PDcontroller(system.contributions_map[joint_name], kp, kd, np.array([0, 0]))
            motor = PDcontroller(
                system.contributions_map[joint_name],
                kp,
                kd,
                np.array([initial_config[joint_name], 0]),
            )
            motor.name = "PD_" + system.contributions_map[joint_name].name
            system.add(motor)

        # kp_hip = 30
        # system.contributions_map["PD_FR_hip_joint"].kp = kp_hip
        # system.contributions_map["PD_FL_hip_joint"].kp = kp_hip
        # system.contributions_map["PD_RR_hip_joint"].kp = kp_hip
        # system.contributions_map["PD_RL_hip_joint"].kp = kp_hip
        frq = 1.25
        angle_d_calf = -np.deg2rad(30)
        for joint_name in [
            "FR_calf_joint",
            "FL_calf_joint",
            "RR_calf_joint",
            "RL_calf_joint",
        ]:
            system.contributions_map["PD_" + joint_name].tau = lambda t: np.array(
                [
                    initial_config[joint_name]
                    + angle_d_calf * np.sin(2 * np.pi * frq * t),
                    angle_d_calf * 2 * np.pi * frq * np.cos(2 * np.pi * frq * t),
                ]
            )

            # angle_d_thigh = np.deg2rad(2)
            # for joint_name in ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]:
            #     system.contributions_map["PD_" + joint_name].tau = lambda t: np.array([initial_config[joint_name] + angle_d_thigh * np.sin(2 * np.pi * frq * t), angle_d_thigh * 2 * np.pi * frq * np.cos(2 * np.pi * frq * t)])

    plane = Box(Frame)(dimensions=[4, 4, 0.001], axis=2, name= "Plane")

    system.add(plane)
    system.assemble()
    
    render = Renderer(system)
    if simulate := False:
        sol = Moreau(system, 0.25, 1e-3).solve()
        render.render_solution(
            sol, repeat=True
        )
    else:
        render.render_solution(
            Solution(system, np.array([system.t0]), np.asanyarray([system.q0]), np.asanyarray([system.u0])), repeat=True
        )