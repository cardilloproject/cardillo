import numpy as np
from pathlib import Path
from cardillo.discrete import RigidBody
from cardillo.discrete.shapes import Cuboid, Ball, Cylinder, FromSTL
from cardillo import System
from cardillo.solver import MoreauClassical
from cardillo.visualization import Export


def test_some_rigid_bodies():
    path = Path(__file__)
    u0 = np.random.rand(6)

    ###############
    # box primitive
    ###############
    dimensions = np.array([3, 2, 1])
    box = Cuboid(RigidBody)(dimensions=dimensions, density=0.0078, u0=u0)

    #############################
    # identical stl box primitive
    #############################
    K_r_SP = -np.array([1.5, 1.0, 0.5])
    # fmt: off
    K_Theta_S = np.array([
        [0.0195,   0.0,    0.0],
        [   0.0, 0.039,    0.0],
        [   0.0,   0.0, 0.0507],
    ])
    stl_box = FromSTL(RigidBody)(
        path=path.parent / ".." / "geometry" / "box" / "box.stl",
        mass=0.0468,
        K_r_SP=K_r_SP,
        K_Theta_S=K_Theta_S,
        u0=u0,
    )

    assert np.isclose(box.mass, stl_box.mass)
    assert np.allclose(box.K_Theta_S, stl_box.K_Theta_S)

    ####################
    # cylinder primitive
    ####################
    q0 = np.array([0, 0, -dimensions[-1], 1, 0, 0, 0], dtype=float)
    cylinder = Cylinder(RigidBody)(
        length=3, radius=1, density=1, axis=2, q0=q0, u0=u0
    )

    ################
    # ball primitive
    ################
    q0 = np.array([*(0.5 * dimensions), 1, 0, 0, 0], dtype=float)
    ball = Ball(RigidBody)(mass=1, radius=0.2, q0=q0, u0=u0)

    ######################################
    # solve system and generate vtk export
    ######################################
    system = System()
    system.add(ball, box, cylinder, stl_box)
    system.assemble()

    sol = MoreauClassical(system, 10, 1e-2).solve()

    e = Export(
        path=path.parent, 
        folder_name=path.stem, 
        overwrite=True, 
        fps=30, 
        solution=sol, 
        system=system)
    e.export_contr(ball)
    e.export_contr(box)
    e.export_contr(cylinder)
    e.export_contr(stl_box)


if __name__ == "__main__":
    test_some_rigid_bodies()
