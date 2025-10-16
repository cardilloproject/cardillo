import numpy as np
from pathlib import Path
from cardillo.urdf import system_from_urdf
from cardillo.actuators import PDcontroller
from cardillo.contacts import Sphere2Plane
from cardillo.discrete import Box, Frame
from cardillo.visualization import Renderer
from cardillo.solver import Moreau, Solution

if __name__ == "__main__":
    from os import path

    # Get directory of this script
    dir_name = path.dirname(__file__)

    # Initial joint configuration for Unitree Go1 robot
    initial_config = {
        "FR_hip_joint": -np.pi / 12,
        "FR_thigh_joint": np.pi / 4,
        "FR_calf_joint": -np.pi / 2,
        "FL_hip_joint": np.pi / 12,
        "FL_thigh_joint": np.pi / 4,
        "FL_calf_joint": -np.pi / 2,
        "RL_hip_joint": np.pi / 12,
        "RL_thigh_joint": np.pi / 4,
        "RL_calf_joint": -np.pi / 2,
        "RR_hip_joint": -np.pi / 12,
        "RR_thigh_joint": np.pi / 4,
        "RR_calf_joint": -np.pi / 2,
        "world_trunk": np.array([0, 0, 0.4, 0, 0, 0]),
    }

    # List of joint names (excluding world_trunk)
    joint_names = [k for k in initial_config.keys() if k != "world_trunk"]

    # Build system from URDF
    system = system_from_urdf(
        file_path=Path(dir_name, "urdf", "go1.urdf"),
        configuration=initial_config,
        gravitational_acceleration=np.array([0, 0, -9.81]),
    )

    # Add foot contact models
    radius = 0.022
    mu = 0.3
    for foot in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
        contact = Sphere2Plane(
            system.origin, system.contributions_map[foot], mu=mu, r=radius
        )
        system.add(contact)

    # Add PD controllers to each joint
    PD_joint_controller = True
    if PD_joint_controller:
        kp = 50
        kd = 1
        for joint_name in joint_names:
            motor = PDcontroller(
                system.contributions_map[joint_name],
                kp,
                kd,
                np.array([initial_config[joint_name], 0]),
            )
            motor.name = "PD_" + system.contributions_map[joint_name].name
            system.add(motor)

        # Example: Add time-varying torque to calf joints (currently static)
        frq = 0
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

    # Add ground plane
    plane = Box(Frame)(dimensions=[4, 4, 0.001], axis=2, name="Plane")
    system.add(plane)
    system.assemble()

    # Visualization and simulation
    render = Renderer(system)
    simulate = True  # Set to False to show initial configuration only
    if simulate:
        sol = Moreau(system, 0.25, 1e-3).solve()
        render.render_solution(sol, repeat=True)
    else:
        render.render_solution(
            Solution(
                system,
                np.array([system.t0]),
                np.asanyarray([system.q0]),
                np.asanyarray([system.u0]),
            ),
            repeat=True,
        )
