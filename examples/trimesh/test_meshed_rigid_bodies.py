import trimesh
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.forces import Force
from cardillo.discrete import RigidBody, Frame, Meshed
from cardillo.math import A_IK_basic, Spurrier
from cardillo.solver import BackwardEuler

from cardillo.visualization import Export

from cardillo.visualization.trimesh import show_system, animate_system


if __name__ == "__main__":
    path = Path(__file__)

    cube_mesh = trimesh.creation.box(extents=[1, 1, 1])
    plane_mesh = cube_mesh.copy().apply_transform(np.diag([1, 1, 0.01, 1]))
    box_mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.1])

    part_mesh = trimesh.load_mesh(Path.joinpath(path.parent, "part.stl"))

    MeshedFrame = Meshed(Frame)
    MeshedRB = Meshed(RigidBody)

    frame = MeshedFrame(
        plane_mesh, K_r_SP=np.array([0, 0, 0]), A_KM=A_IK_basic(np.pi / 10).x()
    )

    q10 = np.concatenate([np.array([0, 0, 1]), Spurrier(A_IK_basic(np.pi / 4).x())])
    rigid_body1 = MeshedRB(box_mesh, density=2, mass=1, K_Theta_S=np.eye(3), q0=q10)

    q20 = np.concatenate([np.array([0, 1, 1]), Spurrier(A_IK_basic(np.pi / 4).x())])
    rigid_body2 = MeshedRB(part_mesh, density=1, mass=1, K_Theta_S=np.eye(3), q0=q20)

    system = System()
    system.add(frame)
    system.add(rigid_body1)
    system.add(rigid_body2)
    system.add(Force(np.array([0, 0, -1 * rigid_body1.mass]), rigid_body1))
    system.add(Force(np.array([0, 0, -1 * rigid_body2.mass]), rigid_body2))
    system.assemble()

    # show_system(system, system.t0, system.q0, origin_size=0.05)

    sol = BackwardEuler(system, 5, 1e-1).solve()

    path = Path(__file__)
    e = Export(
        path=path.parent,
        folder_name=path.stem,
        overwrite=True,
        fps=30,
        solution=sol,
    )
    e.export_contr(system.origin)
    e.export_contr(frame)
    e.export_contr(rigid_body1)
    e.export_contr(rigid_body2)

    animate_system(system, sol.t, sol.q, t_factor=1, origin_size=0.05)
