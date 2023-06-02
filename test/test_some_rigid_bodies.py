import numpy as np
from pathlib import Path
from cardillo.discrete import Frame, RigidBodyQuaternion, RigidBodyRelKinematics
from cardillo.discrete.shapes import Rectangle, Cuboid, Ball, Cylinder, FromSTL
from cardillo.joints import RigidConnection
from cardillo.math import Exp_SO3
from cardillo import System
from cardillo.solver import MoreauClassical
from cardillo.visualization import Export

if __name__ == "__main__":
    path = Path(__file__)
    u0 = np.random.rand(6)

    ############################
    # plane attatched to a frame
    ############################
    rectangle = Rectangle(Frame)(
        r_OP=np.array([0, 0, -1]),
        dimensions=(3, 2),
    )

    ########################
    # stl attatched to frame
    ########################
    v_S = np.random.rand(3)
    psi = np.random.rand(3)
    disk = FromSTL(Frame)(
        scale=10,
        path=path.parent / ".." / "geometry" / "tippedisk" / "tippedisk.stl",
        K_r_SP=np.zeros(3),
        r_OP=lambda t: np.array([0, -1, 1]) + t * v_S,
        A_IK=lambda t: Exp_SO3(t * psi),
    )

    ###############
    # box primitive
    ###############
    dimensions = np.array([3, 2, 1])
    box = Cuboid(RigidBodyQuaternion)(dimensions=dimensions, density=0.0078, u0=u0)

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
    stl_box = FromSTL(RigidBodyQuaternion)(
        path=path.parent / ".." / "geometry" / "box" / "box.stl",
        mass=0.0468,
        K_r_SP=K_r_SP,
        K_Theta_S=K_Theta_S,
        u0=u0,
    )

    ###########################################################
    # identical relativ rigid body rigidly connected to the box
    ###########################################################
    joint = RigidConnection()
    relative_box = Cuboid(RigidBodyRelKinematics)(joint=joint, predecessor=box, dimensions=dimensions, density=0.0078)

    assert np.isclose(box.mass, stl_box.mass)
    assert np.isclose(box.mass, relative_box.mass)
    assert np.allclose(box.K_Theta_S, stl_box.K_Theta_S)
    assert np.allclose(box.K_Theta_S, relative_box.K_Theta_S)

    ####################
    # cylinder primitive
    ####################
    radius = 1
    q0 = np.array([0, 0, -0.5 * dimensions[-1] - radius, 1, 0, 0, 0], dtype=float)
    cylinder = Cylinder(RigidBodyQuaternion)(
        length=3, radius=radius, density=1, axis=1, q0=q0, u0=u0
    )

    ################
    # ball primitive
    ################
    q0 = np.array([*(0.5 * dimensions), 1, 0, 0, 0], dtype=float)
    ball = Ball(RigidBodyQuaternion)(mass=1, radius=0.2, q0=q0, u0=u0)

    ######################################
    # solve system and generate vtk export
    ######################################
    system = System()
    system.add(rectangle, disk, ball, box, cylinder, stl_box, joint, relative_box)
    system.assemble()

    # sol = MoreauClassical(system, 10, 1e-1).solve()
    sol = MoreauClassical(system, 1, 1e-1).solve()

    e = Export(path.parent, path.stem, True, 30, sol, system)
    e.export_contr(rectangle)
    e.export_contr(disk)
    e.export_contr(ball)
    e.export_contr(box)
    e.export_contr(cylinder)
    e.export_contr(stl_box)
    e.export_contr(relative_box)
