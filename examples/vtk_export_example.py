import numpy as np
from pathlib import Path

from cardillo.discrete import (
    Frame,
    PointMass,
    Ball,
    Cuboid,
    RigidBody,
    Rectangle,
)
from cardillo.forces import (
    Force,
    ScalarForceTranslational,
    LinearSpring,
)

from cardillo.constraints import Spherical

from cardillo.system import System
from cardillo.math import axis_angle2quat, cross3
from cardillo.solver import ScipyIVP
from cardillo.visualization import Export


class Ball(RigidBody):
    def __init__(self, m, r, q0, u0=None):
        B_Theta_C = 2 / 5 * m * r**2 * np.eye(3)
        super().__init__(m, B_Theta_C, q0, u0)


if __name__ == "__main__":
    points_cube = np.array(
        [
            [1, -0.5, -0.5],
            [1, 0.5, -0.5],
            [1, 0.5, 0.5],
            [1, -0.5, 0.5],
            [-1, -0.5, -0.5],
            [-1, 0.5, -0.5],
            [-1, 0.5, 0.5],
            [-1, -0.5, 0.5],
        ]
    )

    # create a cuboid body object
    m = 10
    r_OC = np.zeros(3)
    phi = np.pi / 6
    p = axis_angle2quat(np.array([0, 0, 1]), phi)
    phi_dot = 10
    omega = np.array([0, 0, phi_dot])
    v_C = cross3(omega, r_OC)

    q0 = np.concatenate([r_OC, p])
    u0 = np.concatenate([v_C, omega])

    # cube = ConvexRigidBody(points_cube, mass=m, u0=u0, q0=q0)
    # cube = newConvexRigidBody(RigidBodyQuaternion, points_cube, mass=m, u0=u0, q0=q0)
    cube = Cuboid(RigidBody)(dimensions=[1, 2, 3], mass=m, q0=q0, u0=u0)
    cube2 = Cuboid(RigidBody)(dimensions=[1, 2, 3], mass=m, q0=q0, u0=u0)
    r = 0.5
    # ball = Ball(RigidBodyQuaternion)(m, r, q0, u0)

    m = 1
    pm0 = PointMass(m, np.zeros(3))
    r_OC1 = np.array([1, 0, 0])
    pm1 = PointMass(m, r_OC1)
    om = 2 / 3 * np.pi
    force = Force(lambda t: np.array([np.sin(om * t), np.cos(om * t), 0]), pm0)

    k = 1e2
    spring = ScalarForceTranslational(pm0, pm1, LinearSpring(k))
    frame = Rectangle(Frame)(axis=1, r_OP=r_OC1)
    joint = Spherical(frame, pm1, r_OC1)

    system = System()
    system.add(cube)
    system.add(cube2)
    # system.add(ball)

    # system = System()
    # system.add(pm0)
    # system.add(force)
    # system.add(pm1)

    # system.add(spring)
    system.add(frame)
    # system.add(joint)

    system.assemble()

    t0 = 0
    t1 = 3
    dt = 1e-2
    solution = ScipyIVP(system, t1, dt).solve()

    path = Path(__file__)
    # # overwrite param == True leads to overwriting of previously calculated data.
    # # If it is set to false a new folder, e.g. sim_data1 is created and used instead.
    # path_vtk = create_vtk_folder(path.parent, path.stem, overwrite=True)

    # # If the simulated time is rather long or the time step small (or both) you don't want to export all data.
    # sol_export = prepare_data(solution, t1, fps=60)
    # VtkExport.convex_body(path_vtk, sol_export, cube)

    e = Export(
        path=path.parent,
        folder_name=path.stem,
        overwrite=True,
        fps=30,
        solution=solution,
    )
    # e.export_contr([pm0, pm1], file_name="points")
    # e.export_contr(pm0)
    e.export_contr([cube, cube2])
    # e.export_contr(cube)
    # e.export_contr(cube, base_export=True)
    # e.export_contr(ball, resolution=20)
    # e.export_contr(spring)
    # e.export_contr(frame, base_export=True)
    e.export_contr(frame)
