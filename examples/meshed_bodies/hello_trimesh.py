# [x] apply transformations

# stl files
# - read
# - write vtk

# intertia
# - center of mass
# - intertia

# contacts
# - closest point
# - outward normals
# - convex hull
# - bounding box
# - ???

# notes
# - pyglet<2 dependency

import numpy as np
import trimesh

if __name__ == "__main__":
    ##############
    # create frame
    ##############
    origin = trimesh.creation.axis()

    ############
    # primitives
    ############
    # [x] plane (degenerated box)
    # [x] box
    # [x] sphere
    # [x] cylinder
    # [x] capsule
    plane = trimesh.primitives.Box(extents=[3, 3, 0])
    box = trimesh.primitives.Box(extents=[1, 1, 1])
    box.apply_transform(trimesh.transformations.random_rotation_matrix())
    box.apply_translation([1, 1, 1])
    sphere = trimesh.primitives.Sphere(radius=0.5, center=[-1, 1, 1])
    cylinder = trimesh.primitives.Cylinder(radius=0.5, height=1)
    cylinder.apply_translation([-1, -1, 1])
    capsule = trimesh.primitives.Capsule(radius=0.5, height=1)
    capsule.apply_translation([1, -1, 1])

    # primitives = [plane, box, sphere, cylinder, capsule]
    primitives = [box]

    # check that all primitives are "watertight"
    for p in primitives:
        assert p.is_watertight

    # inertia properties
    for p in primitives:
        print(f" - {p.__class__.__name__}")

        # convert to mesh
        mesh = p.to_mesh()

        # transformation
        print(f"  * homogeneous transformation:\n{p.bounding_box_oriented.transform}")

        # the convex hull is another Trimesh object that is available as a property
        # lets compare the volume of our mesh with the volume of its convex hull
        volume = mesh.volume
        print(f"  * volume: {volume}")

        # since the mesh is watertight, it means there is a
        # volumetric center of mass which we can set as the origin for our mesh
        center_mass = mesh.center_mass
        print(f"  * center of mass: {center_mass}")
        mesh.vertices -= mesh.center_mass

        # what's the moment of inertia for the mesh?
        moment_inertia = mesh.moment_inertia
        print(f"  * moment of inertia:\n{moment_inertia}")

    # [x] apply transformations
    translation = trimesh.transformations.random_vector(3)
    rotation = trimesh.transformations.random_rotation_matrix()
    # cylinder.apply_translation(translation)
    # cylinder.apply_transform(rotation)
    transformation = rotation
    transformation[:3, -1] = translation
    cylinder.apply_transform(rotation)

    # [x] visualization
    scene = trimesh.Scene()
    scene.add_geometry(origin)
    scene.add_geometry(plane)
    scene.add_geometry(box)
    scene.add_geometry(sphere)
    scene.add_geometry(cylinder)
    scene.add_geometry(capsule)
    # scene.show()

    # [ ] TODO: vtk export
    #

    # primitives = [box, sphere, cylinder, capsule]
    # for p in primitives:
    #     p.to_mesh().show()

    # scene = trimesh.Scene()

    # # plane
    # plane = trimesh.creation.box(extents=[1, 1, 0.01])
    # plane.visual.face_colors = [0.5, 0.5, 0.5, 0.5]
    # scene.add_geometry(plane)
    # scene.add_geometry(trimesh.creation.axis())

    # # object-1 (box)
    # box = trimesh.creation.box(extents=[0.3, 0.3, 0.3])
    # box.visual.face_colors = [0, 1., 0, 0.5]
    # axis = trimesh.creation.axis(origin_color=[1., 0, 0])
    # translation = [-0.2, 0, 0.15 + 0.01]  # box offset + plane offset
    # box.apply_translation(translation)
    # axis.apply_translation(translation)
    # rotation = trimesh.transformations.rotation_matrix(
    #     np.deg2rad(30), [0, 0, 1], point=box.centroid
    # )
    # box.apply_transform(rotation)
    # axis.apply_transform(rotation)
    # scene.add_geometry(box)
    # scene.add_geometry(axis)

    # # object-2 (cylinder)
    # cylinder = trimesh.creation.cylinder(radius=0.1, height=0.3)
    # cylinder.visual.face_colors = [0, 0, 1., 0.5]
    # axis = trimesh.creation.axis(origin_color=[1., 0, 0])
    # translation = [0.1, -0.2, 0.15 + 0.01]  # cylinder offset + plane offset
    # cylinder.apply_translation(translation)
    # axis.apply_translation(translation)
    # scene.add_geometry(cylinder)
    # scene.add_geometry(axis)

    # scene.show()
