import numpy as np
from pathlib import Path
from cardillo.discrete import RigidBodyQuaternion
from cardillo.discrete.some_rigid_bodies import Box, Ball, Cylinder, FromSTL
from cardillo import System
from cardillo.solver import Solution
from cardillo.visualization import Export

if __name__ == "__main__":
    path = Path(__file__)

    dimensions = np.array([3, 1, 2])
    box = Box(RigidBodyQuaternion)(dimensions=dimensions, density=1)
    q0 = np.array([0, 0, -dimensions[-1], 1, 0, 0, 0], dtype=float)
    cylinder = Cylinder(RigidBodyQuaternion)(length=3, radius=1, density=1, q0=q0)
    q0 = np.array([*(0.5 * dimensions), 1, 0, 0, 0], dtype=float)
    ball = Ball(RigidBodyQuaternion)(mass=1, radius=0.2, q0=q0)
    stl = FromSTL(RigidBodyQuaternion)(path.parent / "Suzanne.stl", density=1)
    # stl = FromSTL(RigidBodyQuaternion)(path.parent / "Tetrahedron.stl", density=1)

    system = System()
    system.add(ball, box, cylinder, stl)
    system.assemble()

    t = [0]
    q = [system.q0]
    u = [system.u0]
    sol = Solution(t=t, q=q, u=u)

    e = Export(path.parent, path.stem, True, 30, sol)
    e.export_contr(ball)
    e.export_contr(box)
    e.export_contr(cylinder)
    e.export_contr(stl)
