import numpy as np
from pathlib import Path
from cardillo.urdf import system_from_urdf
from cardillo.solver import ScipyIVP, Solution
from cardillo.visualization import Renderer

if __name__ == "__main__":
    from os import path

    # Get directory of this script
    dir_name = path.dirname(__file__)

    # Initial joint configuration for Panda robot
    initial_config = {
        "panda_joint1": 0.0,
        "panda_joint2": 1.0,
        "panda_joint3": 0.0,
        "panda_joint4": -1.5,
        "panda_joint5": 0.0,
        "panda_joint6": 2.0,
        "panda_joint7": 0.0,
        "panda_finger_joint1": 0.0,
    }

    # Build system from URDF
    system = system_from_urdf(
        file_path=Path(dir_name, "urdf", "panda.urdf"),
        configuration=initial_config,
        gravitational_acceleration=np.array([0, 0, -9.81]),
    )

    # Visualization
    render = Renderer(system)

    simulate = False  # Set to True to run a short simulation
    if simulate:
        sol = ScipyIVP(system, t1=1, dt=1e-1).solve()
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
