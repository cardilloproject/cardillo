from numpy import array, hstack, zeros, pi
from pathlib import Path

from cardillo.discrete import (
    Box,
    Ball,
    PlaneFixed,
    RigidBodyQuaternion,
    RigidBodyAxisAngle,
)
from cardillo.math import axis_angle2quat
from cardillo.contacts import Sphere2Plane, Point2Plane
from cardillo import System
from cardillo.forces import Force
from cardillo.solver import NonsmoothBackwardEuler, MoreauShiftedNew, NPIRK
from cardillo.visualization import Export
from cardillo.solver._butcher_tableaus import RadauIIATableau

parametrization = "quaternion"
# parametrization = "axis_angle"
# primitive = "box"
# primitive = "ball"
primitive = "cylinder"

if parametrization == "quaternion":
    P = RigidBodyQuaternion
elif parametrization == "axis_angle":
    P = RigidBodyAxisAngle
else:
    raise (RuntimeError, "unknown parametrization")

if __name__ == "__main__":
    plane = PlaneFixed(array((0, 1, 1)), dim_export=(3, 5))

    ##############################################################
    #               Ridid body
    ##############################################################
    m = 1
    r_OS = array((0, 0, 2))
    phi = 5 * pi / 180
    axis = array((0, 1, 0))
    if parametrization == "quaternion":
        p = axis_angle2quat(axis, phi)
        q0 = hstack((r_OS, p))
    elif parametrization == "axis_angle":
        q0 = hstack((r_OS, phi * axis))
    v_S = zeros(3)
    K_Omega = zeros(3)
    u0 = hstack((v_S, K_Omega))
    if primitive == "box":
        dim = (2, 1, 0.5)
        RB = Box(P)(dim, m, q0=q0, u0=u0)
    elif primitive == "ball":
        r = 1
        RB = Ball(P)(m, r, q0, u0)
    F_G = Force(array((0, 0, -9.81)), RB)

    ##############################################################
    #               Contact
    ##############################################################
    mu = 0.3
    e_N = 0.5
    e_F = 0
    if primitive == "box":
        contact = [
            Sphere2Plane(plane, RB, 0, mu, e_N, e_F, K_r_SP=p) for p in RB.points
        ]
    elif primitive == "ball":
        contact = [Sphere2Plane(plane, RB, RB.radius, mu, e_N, e_F)]

    ##############################################################
    #               System
    ##############################################################
    system = System()
    system.add(plane)
    system.add(RB)
    system.add(F_G)
    system.extend(contact)
    system.assemble()

    ##############################################################
    #               Solver
    ##############################################################
    dt = 5e-2
    t1 = 1
    # solver = NonsmoothBackwardEuler(system, t1, dt)
    # solver = MoreauShiftedNew(system, t1, dt)
    butcher_tableau = RadauIIATableau(3)
    solver = NPIRK(system, t1, dt, butcher_tableau)
    sol = solver.solve()

    ##############################################################
    #               Export
    ##############################################################
    file_path = Path(__file__)
    path = Path(file_path.parents[1], "sim_data")
    folder = file_path.stem
    e = Export(path, folder, overwrite=True, fps=60, solution=sol)
    e.export_contr(plane, file_name="Plane")
    e.export_contr(RB, base_export=True, file_name=f"COM_{primitive}")
    e.export_contr(RB, base_export=False, file_name=primitive)
    e.export_contr(F_G, file_name="F_G")
    e.export_contr(contact, file_name="Sphere2Plane")
