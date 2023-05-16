import numpy as np
from pathlib import Path
from cardillo.discrete import RigidBodyQuaternion
from cardillo.discrete.some_rigid_bodies import Box, Ball, Cylinder, FromSTL
from cardillo import System
from cardillo.solver import MoreauClassical
from cardillo.visualization import Export

if __name__ == "__main__":
    path = Path(__file__)

    u0 = np.random.rand(6)

    dimensions = np.array([3, 2, 1])
    box = Box(RigidBodyQuaternion)(dimensions=dimensions, density=0.0078, u0=u0)

    # identical stl box
    stl_box = FromSTL(RigidBodyQuaternion)(
        path.parent / "stl" / "box.stl",
        mass=0.0468,
        r_PS=np.array([1.5, 1.0, 0.5]),
        K_Theta_P=np.array(
            [
                [0.078, 0.0702, 0.0351],
                [0.0702, 0.156, 0.0234],
                [0.0351, 0.0234, 0.2028],
            ]
        ),
        u0=u0,
    )

    q0 = np.array([0, 0, -dimensions[-1], 1, 0, 0, 0], dtype=float)
    cylinder = Cylinder(RigidBodyQuaternion)(
        length=3, radius=1, density=1, q0=q0, u0=u0
    )

    q0 = np.array([*(0.5 * dimensions), 1, 0, 0, 0], dtype=float)
    ball = Ball(RigidBodyQuaternion)(mass=1, radius=0.2, q0=q0, u0=u0)

    system = System()
    system.add(ball, box, cylinder, stl_box)
    system.assemble()

    sol = MoreauClassical(system, 10, 1e-2).solve()

    e = Export(path.parent, path.stem, True, 30, sol)
    e.export_contr(ball)
    e.export_contr(box)
    e.export_contr(cylinder)
    e.export_contr(stl_box)
