import numpy as np
from pathlib import Path
from cardillo.discrete import RigidBody, Frame
from cardillo.discrete.meshed import Meshed, Box, Cylinder, Sphere, Tetrahedron
from cardillo import System
from cardillo.solver import Moreau
from cardillo.visualization import Export


def test_some_rigid_bodies():
    path = Path(__file__)
    u0 = np.random.rand(6)

    #####################
    # rectangle primitive
    #####################
    rectangle = Box(Frame)(dimensions=[5, 8, 1e-3], axis=2)

    ###############
    # box primitive
    ###############
    dimensions = np.array([3, 2, 1])
    box = Box(RigidBody)(dimensions=dimensions, density=0.0078, u0=u0)

    #############################
    # identical stl box primitive
    #############################
    B_r_CP = -np.array([1.5, 1.0, 0.5])
    # fmt: off
    B_Theta_C = np.array([
        [0.0195,   0.0,    0.0],
        [   0.0, 0.039,    0.0],
        [   0.0,   0.0, 0.0507],
    ])
    stl_box = Meshed(RigidBody)(
        path.parent / "_data" / "box.stl",
        mass=0.0468,
        B_r_CP=B_r_CP,
        B_Theta_C=B_Theta_C,
        u0=u0,
    )

    assert np.isclose(box.mass, stl_box.mass)
    assert np.allclose(box.B_Theta_C, stl_box.B_Theta_C)

    ####################
    # cylinder primitive
    ####################
    q0 = np.array([0, 0, -dimensions[-1], 1, 0, 0, 0], dtype=float)
    cylinder = Cylinder(RigidBody)(
        height=3, radius=1, density=1, q0=q0, u0=u0
    )

    ################
    # ball primitive
    ################
    q0 = np.array([*(0.5 * dimensions), 1, 0, 0, 0], dtype=float)
    ball = Sphere(RigidBody)(density=1, radius=0.2, q0=q0, u0=u0)

    ########################
    # tetraherdron primitive
    ########################
    u0 = np.random.rand(6)
    q0 = np.array([*(-0.5 * dimensions), 1, 0, 0, 0], dtype=float)
    tetrahedron = Tetrahedron(RigidBody)(density=1, edge=1, q0=q0, u0=u0)

    ######################################
    # solve system and generate vtk export
    ######################################
    system = System()
    system.add(rectangle, ball, box, cylinder, stl_box, tetrahedron)
    system.assemble()

    sol = Moreau(system, 1, 1e-2).solve()

    e = Export(
        path=path.parent, 
        folder_name=path.stem, 
        overwrite=True, 
        fps=30, 
        solution=sol)
    # TODO: give rectangle also a name
    e.export_contr(rectangle, file_name="floor")
    e.export_contr(ball)
    e.export_contr(box)
    e.export_contr(cylinder)
    e.export_contr(stl_box)
    e.export_contr(tetrahedron)


if __name__ == "__main__":
    test_some_rigid_bodies()
