import numpy as np
from pathlib import Path

from cardillo.discrete import PointMass, ConvexRigidBody, RigidBodyQuaternion
from cardillo.forces import Force

from cardillo.system import System
from cardillo.math import axis_angle2quat, cross3
from cardillo.solver import ScipyIVP
from cardillo.utility import Export


class Ball(RigidBodyQuaternion):
    def __init__(self, m, r, q0, u0=None):
        K_theta_S = 2 / 5 * m * r**2 * np.eye(3)
        super().__init__(m, K_theta_S, q0, u0)


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

    # create a convex body object
    m = 10
    r_OS = np.zeros(3)
    phi = np.pi / 6
    p = axis_angle2quat(np.array([0, 0, 1]), phi)
    phi_dot = 10
    omega = np.array([0, 0, phi_dot])
    v_S = cross3(omega, r_OS)

    q0 = np.concatenate([r_OS, p])
    u0 = np.concatenate([v_S, omega])

    cube = ConvexRigidBody(points_cube, mass=m, u0=u0, q0=q0)
    # r = 0.5
    # ball = Ball(m, r, q0, u0)

    system = System()
    system.add(cube)
    # system.add(ball)
    system.assemble()

    # m = 1
    # pm0 = PointMass(m, np.zeros(3))
    # pm1 = PointMass(m, np.zeros(3))
    # om = 2/3*np.pi
    # force = Force(lambda t: np.array([np.sin(om*t), np.cos(om*t), 0]), pm0)
    # system = System()
    # system.add(pm0)
    # system.add(force)
    # system.add(pm1)
    # system.assemble()

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

    # Export(path.parent, path.stem, True, 30, solution).export_contr([pm0, pm1])
    # Export(path.parent, path.stem, True, 30, solution).export_contr([pm0])
    e = Export(path.parent, path.stem, True, 30, solution)
    e.export_contr(cube)
    e.export_contr(cube, base_export=True)
    # Export(path.parent, path.stem, True, 30, solution).export_contr(ball)
