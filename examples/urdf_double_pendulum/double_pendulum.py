import numpy as np
from cardillo.urdf import system_from_urdf
from cardillo.discrete import Sphere, Frame
from cardillo.math import A_IB_basic
from cardillo.visualization import Renderer
from cardillo.solver import Solution, ScipyIVP

configuration = {
    "joint1": np.pi,
    "joint2": np.pi / 2,
}
velocity = {
    "joint1": 1.0,
    # "joint2": 5.0,
}

system = system_from_urdf(
    "examples/urdf_double_pendulum/urdf/double_pendulum.urdf",
    r_OR=np.array([0.05, 0, 0.1]),
    A_IR=A_IB_basic(np.pi / 4).x,
    configuration=configuration,
    velocities=velocity,
    root_is_floating=False,
    gravitational_acceleration=np.array([0, 0, -9.81]),
)

render = Renderer(system)

if simulate := True:
    sol = ScipyIVP(system, t1=3, dt=1e-2).solve()
    render.render_solution(sol, repeat=True)
else:
    render.render_solution(
        Solution(system, np.array([system.t0]), np.array([system.q0])), repeat=True
    )
