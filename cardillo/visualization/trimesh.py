import trimesh
import numpy as np
from scipy.interpolate import interp1d

from cardillo.math import SE3inv


def show_system(system, t, q, origin_size=0):
    # TODO: this is nice for debugging and quickly get an overview. However, when the window is closed, it stops the execution of the code.
    # If we find a solution to this, we could provide this function as a visualization utility or as part of System.py.
    scene = trimesh.Scene()
    if origin_size > 0:
        scene.add_geometry(trimesh.creation.axis(origin_size=origin_size))
    for contr in system.contributions:
        if hasattr(contr, "K_visual_mesh"):
            qc = q[contr.qDOF]
            H_IK = np.eye(4)
            H_IK[:3, 3] = contr.r_OP(t, qc)
            H_IK[:3, :3] = contr.A_IK(t, qc)
            scene.add_geometry(contr.K_visual_mesh.copy().apply_transform(H_IK))
    scene.show()
    return


def animate_system(system, t, q, fps=30, t_factor=1, origin_size=0):
    # TODO: this is nice for debugging and quickly get an overview. However, when the window is closed, it stops the execution of the code.
    # If we find a solution to this, we could provide this function as a visualization utility or as part of System.py.
    # t_factor : 1s real time = t_factor * 1s animation time (t_factor=10 means video is 10 times slower than reality)
    scene = trimesh.Scene()
    # initial configuration
    if origin_size > 0:
        scene.add_geometry(trimesh.creation.axis(origin_size=origin_size))
    contributions = {}
    transformations = {}
    for contr in system.contributions:
        if hasattr(contr, "K_visual_mesh"):
            qc = system.q0[contr.qDOF]
            H_IK = np.eye(4)
            H_IK[:3, 3] = contr.r_OP(system.t0, qc)
            H_IK[:3, :3] = contr.A_IK(system.t0, qc)
            mesh = contr.K_visual_mesh.copy().apply_transform(H_IK)
            name = scene.add_geometry(mesh)
            contributions[name] = contr
            transformations[name] = H_IK

    # interpolate data
    q_interp = interp1d(t, q, axis=0)
    dt = float(1 / fps)
    t_span = np.arange(t[0], t[-1], step=dt / t_factor)
    # this is a trick to pass variables to callback function as it is called as: callback(scene)
    scene.i = 0
    scene.t = t_span
    scene.q = q_interp
    scene.contributions = contributions
    scene.H_IK = transformations
    scene.show(callback=update_scene, callback_period=dt)


def update_scene(scene):
    # TODO: this is not a real update but a redrawing of the scene. Using relative transformations on the objects could be faster.
    ti = scene.t[scene.i]
    qi = scene.q(ti)
    scene.i += 1
    scene.i = scene.i % len(scene.t)
    for name in scene.contributions:
        contr = scene.contributions[name]
        H_I0K = scene.H_IK[name]
        qic = qi[contr.qDOF]
        H_IK = np.eye(4)
        H_IK[:3, 3] = contr.r_OP(ti, qic)
        H_IK[:3, :3] = contr.A_IK(ti, qic)
        H_update = H_IK @ SE3inv(H_I0K)
        scene.graph.update(name, matrix=H_update)
        scene.H_IK[name] = H_IK
