from cardillo.math import e1, e2
from cardillo.beams import (
    RectangularCrossSection,
    Simo1986,
)
from cardillo.discrete import Frame
from cardillo.constraints import Revolute
from cardillo.constraints._base import ProjectedPositionOrientationBase
from cardillo.beams import animate_beam

from cardillo.beams import K_R12_PetrovGalerkin_AxisAngle as Rod
from cardillo.forces import Force
from cardillo import System
from cardillo.solver import Newton, Riks
from cardillo.visualization import Export

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# number of elements
nelements = 10

# used polynomial degree
polynomial_degree = 2
basis = "Lagrange"


if __name__ == "__main__":
    # Young's and shear modulus
    E = 1.0
    G = 0.5

    # length od the rod
    L = 1

    # used cross section
    width = 0.005
    height = 0.02

    # cross section and quadratic beam material
    line_density = 1
    cross_section = RectangularCrossSection(line_density, width, height)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A]) * 10
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    q0 = Rod.straight_configuration(
        polynomial_degree,
        polynomial_degree,
        basis,
        basis,
        nelements,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
    )
    rod = Rod(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree,
        polynomial_degree,
        nelements,
        Q=q0,
        q0=q0,
        basis_r=basis,
        basis_psi=basis,
    )

    frame0 = Frame(r_OP0, A_IK0)

    # left and right joint
    joint0 = Revolute(frame0, rod, axis=2, frame_ID2=(0,))

    r_OP1 = np.array([L, 0, 0], dtype=float)
    frame1 = Frame(r_OP1, A_IK0)
    joint1 = ProjectedPositionOrientationBase(
        frame1,
        rod,
        constrained_axes_translation=[1, 2],
        projection_pairs_rotation=[(1, 2), (2, 0)],
        frame_ID2=(1,),
    )

    # force at the beam's tip
    f_crit = np.pi**2 * E * I3 / L**2
    F1 = lambda t: -t * e1 * f_crit * 1.3
    force1 = Force(F1, rod, frame_ID=(1,))

    F12 = lambda t: t * e2 * f_crit * 1e-3
    force12 = Force(F12, rod, frame_ID=(0.5,))

    # assemble the system
    system = System()
    system.add(rod)
    system.add(frame0)
    system.add(joint0)
    system.add(frame1)
    system.add(joint1)
    system.add(force1)
    system.add(force12)
    system.assemble()

    n_load_steps = 100
    sol = Newton(
        system,
        atol=1e-12,
        n_load_steps=n_load_steps,
        max_iter=30,
    ).solve()

    # sol = Riks(
    #     system,
    #     # tol=1e-10, # factor 2
    #     # tol=1e-12, # factor 4
    #     tol=1e-14, # factor 4
    #     la_arc_span=[-1, 1],
    # ).solve()

    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    ############
    # VTK export
    ############
    path = Path(__file__)
    e = Export(path.parent, path.stem, True, n_load_steps, sol)
    # e.export_contr(rod, level="centerline + directors", num=20)
    e.export_contr(rod, level="volume", n_segments=nelements, num=50)

    #########################
    # visualize displacements
    #########################
    fig, ax = plt.subplots()

    r_OPs = np.array([rod.centerline(qi, num=3) for qi in sol.q])
    forces = np.array([F1(ti) for ti in sol.t]) / f_crit

    ax.plot([-1, -1], [0, 0.6 * L], "-k", label="f_crit")
    # ax.plot(forces[:, 0], r_OPs[:, 0, 1], "-r", label="x(t, 0.5)")
    ax.plot(forces[:, 0], r_OPs[:, 1, 1], "-g", label="y(t, 0.5)")
    # ax.plot(forces[:, 0], r_OPs[:, 2, 1], "--b", label="z(t, 0.5)")
    ax.grid()
    ax.legend()

    ###########
    # animation
    ###########
    animate_beam(t, q, [rod], L, show=True)
