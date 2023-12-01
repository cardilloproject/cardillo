import trimesh
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.forces import Force
from cardillo.discrete import RigidBody, Frame, Meshed
from cardillo.math import A_IK_basic
from cardillo.solver import BackwardEuler

from cardillo.visualization import Export


if __name__=="__main__":

    cube_mesh = trimesh.creation.box(extents=[1,1,1])
    plane_mesh = cube_mesh.copy().apply_transform(np.diag([1,1,0.01,1]))
    box_mesh = trimesh.creation.box(extents=[1,1,1])

    MeshedFrame = Meshed(Frame)
    MeshedRB = Meshed(RigidBody)

    frame = MeshedFrame(plane_mesh, K_r_SP=np.array([0, 0, 0]), A_KM=A_IK_basic(np.pi/10).x())
    
    q0 = np.array([0,0,1,1,0,0,0])
    rigid_body = MeshedRB(box_mesh, density=1, mass=1, K_Theta_S=np.eye(3), q0=q0)#K_r_SP=np.array([0, 0, 1]), A_KM=A_IK_basic(np.pi/10).x())

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.axis(origin_size=0.05))
    scene.add_geometry(frame.get_visual_mesh_wrt_I(0, 0))
    scene.add_geometry(rigid_body.get_visual_mesh_wrt_I(0, q0))
    # scene.add_geometry(trimesh.creation.axis(origin_size=0.1, transform=box.visual_mesh.bounding_box_oriented.transform))
    scene.show()

    system = System()
    system.add(frame)
    system.add(rigid_body)
    system.add(Force(np.array([0,0,-1]), rigid_body))
    system.assemble()

    sol = BackwardEuler(system, 1, 1e-1).solve()

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
    e.export_contr(rigid_body)