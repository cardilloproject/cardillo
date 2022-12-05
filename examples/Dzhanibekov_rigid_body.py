import numpy as np
from pathlib import Path

from cardillo.discrete import Box, RigidBodyQuaternion
from cardillo.math import axis_angle2quat
from cardillo.system import System
from cardillo.solver import RadauIIa, EulerBackward
from cardillo.utility import Export

if __name__=="__main__":
    
    l = 1
    w = 2
    h = 3
    r_OP0 = np.zeros(3)
    v_P0 = np.zeros(3)
    phi0 = 0
    phi_dot0 = 20
    # K_Omega_disturbance = np.array((1e-5, 0, 0))
    K_Omega_disturbance = np.array((1, 1, 1)) * 1
    K_Omega = np.array((0, phi_dot0, 0))
    # K_Omega = np.array((phi_dot0, 0, 0))
    q0 = np.concatenate((r_OP0, axis_angle2quat(np.array((0, 0, 1)), phi0)))
    u0 = np.concatenate((v_P0, K_Omega + K_Omega_disturbance))
    m = 5
    box = Box(RigidBodyQuaternion)(l, w, h, q0, u0, mass=m)

    system = System()
    system.add(box)
    system.assemble()

    t1 = 4
    dt = 1e-2
    # solver = RadauIIa(system, t1, dt)
    solver = EulerBackward(system, t1, dt)
    sol = solver.solve()

    path = Path(__file__)
    e = Export(path.parent, path.stem, overwrite=True, fps=30, solution=sol)
    e.export_contr(box)
    e.export_contr(box, base_export=True)