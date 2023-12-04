import trimesh
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.forces import Force
from cardillo.discrete import RigidBody, Frame, Meshed
from cardillo.math import A_IK_basic, Spurrier
from cardillo.solver import BackwardEuler

from cardillo.visualization import Export

from scipy.interpolate import interp1d

def show_system(system, t, q, origin_size=0):
    fps = 30
    # TODO: this is nice for debugging and quickly get an overview. However, when the window is closed, it stops the execution of the code. 
    # If we find a solution to this, we could provide this function as a visualization utility or as part of System.py.
    scene = trimesh.Scene()
    if origin_size>0:
        scene.add_geometry(trimesh.creation.axis(origin_size=origin_size))
    for contr in system.contributions:
        if hasattr(contr, "get_visual_mesh_wrt_I"):
            scene.add_geometry(contr.get_visual_mesh_wrt_I(t, q[contr.qDOF]))
    scene.show()
    return

def animate_system(system, t, q, fps=30, t_factor=1, origin_size=0):
    # TODO: this is nice for debugging and quickly get an overview. However, when the window is closed, it stops the execution of the code. 
    # If we find a solution to this, we could provide this function as a visualization utility or as part of System.py.
    # t_factor : 1s real time = t_factor * 1s animation time (t_factor=10 means video is 10 times slower than reality)
    scene = trimesh.Scene()
    # since an empty scene would raise an arror, an origin is added. This is overridden anyways.
    scene.add_geometry(trimesh.creation.axis())

    # interpolate data
    q_interp = interp1d(t, q, axis=0)
    dt = 1/fps
    t_span = np.arange(t[0], t[-1], step=dt/t_factor)
    # this is a trick to pass variables to callback function as it is called as: callback(scene)
    scene.i = 0
    scene.t = t_span
    scene.q = q_interp
    scene.system = system
    scene.origin_size = origin_size
    scene.show(callback=update_scene, callback_period=dt)

def update_scene(scene):
    # TODO: this is not a real update but a redrawing of the scene. Using relative transformations on the objects could be faster.
    ti = scene.t[scene.i]
    qi = scene.q(ti)
    scene.i += 1
    scene.i = scene.i % len(scene.t)
    scene.geometry.clear()
    if scene.origin_size>0:
        scene.add_geometry(trimesh.creation.axis(origin_size=scene.origin_size))
    for contr in scene.system.contributions:
        if hasattr(contr, "get_visual_mesh_wrt_I"):
            scene.add_geometry(contr.get_visual_mesh_wrt_I(ti, qi[contr.qDOF]))
    

if __name__=="__main__":
    path = Path(__file__)

    cube_mesh = trimesh.creation.box(extents=[1,1,1])
    plane_mesh = cube_mesh.copy().apply_transform(np.diag([1,1,0.01,1]))
    box_mesh = trimesh.creation.box(extents=[0.2,0.2,0.1])

    part_mesh = trimesh.load_mesh(Path.joinpath(path.parent,"part.stl"))

    MeshedFrame = Meshed(Frame)
    MeshedRB = Meshed(RigidBody)

    frame = MeshedFrame(plane_mesh, K_r_SP=np.array([0, 0, 0]), A_KM=A_IK_basic(np.pi/10).x())
    
    q10 = np.concatenate([np.array([0,0,1]), Spurrier(A_IK_basic(np.pi/4).x())])
    rigid_body1 = MeshedRB(box_mesh, density=2, mass=1, K_Theta_S=np.eye(3), q0=q10)#K_r_SP=np.array([0, 0, 1]), A_KM=A_IK_basic(np.pi/10).x())
    
    q20 = np.concatenate([np.array([0,1,1]), Spurrier(A_IK_basic(np.pi/4).x())])
    rigid_body2 = MeshedRB(part_mesh, density=1, mass=1, K_Theta_S=np.eye(3), q0=q20)#K_r_SP=np.array([0, 0, 1]), A_KM=A_IK_basic(np.pi/10).x())

    system = System()
    system.add(frame)
    system.add(rigid_body1)
    system.add(rigid_body2)
    system.add(Force(np.array([0,0,-1 * rigid_body1.mass]), rigid_body1))
    system.add(Force(np.array([0,0,-1 * rigid_body2.mass]), rigid_body2))
    system.assemble()

    # show_system(system, system.t0, system.q0, origin_size=0.05)

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
    e.export_contr(rigid_body1)
    e.export_contr(rigid_body2)

    animate_system(system, sol.t, sol.q, t_factor=10, origin_size=0.05)