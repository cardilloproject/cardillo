import numpy as np

from cardillo.urdf import system_from_urdf

from cardillo import System
from cardillo.math import A_IB_basic
from pathlib import Path
from cardillo.solver import ScipyIVP, Solution
from cardillo.discrete import Box, Frame
from cardillo.contacts import Sphere2Plane

from cardillo.visualization import Renderer

if __name__ == "__main__":
    from os import path

    dir_name = path.dirname(__file__)

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


    system = system_from_urdf(
        file_path=Path(dir_name, "urdf", "panda.urdf"),
        configuration=initial_config,
        gravitational_acceleration=np.array([0, 0, -9.81]),
    )

    render = Renderer(system)

if simulate := False:
    sol = ScipyIVP(system, t1=1, dt=1e-1).solve()
    render.render_solution(
        sol, repeat=True
    )
else:
    render.render_solution(
        Solution(system, np.array([system.t0]), np.asanyarray([system.q0]), np.asanyarray([system.u0])), repeat=True
    )