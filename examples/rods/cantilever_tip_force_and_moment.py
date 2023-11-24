from cardillo.rods import (
    RectangularCrossSection,
    Simo1986,
    Harsch2021,
    animate_beam,
)

from cardillo.rods.cosseratRod import (
    make_CosseratRod_SE3,
    make_CosseratRod_R12,
    make_CosseratRod_Quat,
)

from cardillo.constraints import RigidConnection
from cardillo.solver import Newton, SolverOptions
from cardillo.forces import Force, K_Moment

from cardillo.math import e2, e3

from cardillo.visualization import Export

from cardillo import System

import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

""" Cantilever beam example from

Harsch, J., Capobianco, G. and Eugster, S. R., "Finite element formulations for constrained spatial nonlinear beam theories", 2021.
https://doi.org/10.1177/10812865211000790

4.1 Elliptic integral solutions of Euler's elastica
"""


def cantilever(
    Rod,
    nelements=10,
    polynomial_degree=2,
    n_load_steps=10,
    VTK_export=False,
    reduced_integration=True,
    constitutive_law=Harsch2021,
):
    # geometry of the rod
    length = 2 * np.pi

    # cross section properties for visualization purposes
    slenderness = 1.0e2
    width = length / slenderness
    cross_section = RectangularCrossSection(width, width)
    Ei = np.array([5, 1, 1])
    Fi = np.array([0.5, 2, 2])

    material_model = constitutive_law(Ei, Fi)

    # construct system
    system = System()

    # compute straight initial configuration of cantilever
    q0 = Rod.straight_configuration(
        nelements, length, polynomial_degree=polynomial_degree
    )
    # construct cantilever
    cantilever = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        reduced_integration=reduced_integration,
    )

    clamping_left = RigidConnection(system.origin, cantilever, frame_ID2=(0,))

    # assemble the system
    system.add(cantilever)
    system.add(clamping_left)

    # spatially fixed load at cantilever tip
    P = lambda t: material_model.Fi[2] * (10 * t) / length**2
    F = lambda t: -P(t) * e2
    force = Force(F, cantilever, (1,))
    system.add(force)

    # moment at cantilever tip
    M = lambda t: 2.5 * P(t) * e3
    moment = K_Moment(M, cantilever, (1,))
    system.add(moment)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # add Newton solver
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        max_iter=30,
        atol=1e-8,
    )

    # solve nonlinear static equilibrium equations
    sol = solver.solve()

    # extract solutions
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # VTK export
    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 30, sol)
        e.export_contr(
            cantilever,
            level="centerline + directors",
            num=3 * nelements,
            file_name="cantilever_curve",
        )
        e.export_contr(
            cantilever,
            continuity="C0",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
            file_name="cantilever_volume",
        )

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        [cantilever],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever.nelement + 1,
        repeat=True,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    Y_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)

    path = Path(__file__)  # creation of a current path

    # add reference solution to the animation
    centerline_T = np.loadtxt(
        Path(
            path.parent,
            "_data_cantilever_tip_force_and_moment",
            "centerline_T_numeric.txt",
        ),
        delimiter=",",
        skiprows=1,
    )
    ax1.plot(
        centerline_T[:, -2],
        centerline_T[:, -1],
        np.zeros_like(centerline_T[:, 0]),
        "-b",
    )
    centerline_EB = np.loadtxt(
        Path(
            path.parent,
            "_data_cantilever_tip_force_and_moment",
            "centerline_EB_numeric.txt",
        ),
        delimiter=",",
        skiprows=1,
    )
    ax1.plot(
        centerline_EB[:, -2],
        centerline_EB[:, -1],
        np.zeros_like(centerline_EB[:, 0]),
        "-g",
    )
    centerline_IEB = np.loadtxt(
        Path(
            path.parent,
            "_data_cantilever_tip_force_and_moment",
            "centerline_IEB_analytic.txt",
        ),
        delimiter=",",
        skiprows=1,
    )
    ax1.plot(
        centerline_IEB[:, -2],
        centerline_IEB[:, -1],
        np.zeros_like(centerline_IEB[:, 0]),
        "-r",
    )

    # plot animation
    ax1.azim = -90
    ax1.elev = 72

    plt.show()


if __name__ == "__main__":
    #####################
    # SE3 interpolation #
    #####################
    # displacement based methods
    cantilever(
        Rod=make_CosseratRod_SE3(),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        reduced_integration=False,
        constitutive_law=Harsch2021,
    )
    cantilever(
        Rod=make_CosseratRod_SE3(constraints=[1, 2]),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        reduced_integration=False,
        constitutive_law=Harsch2021,
    )
    cantilever(
        Rod=make_CosseratRod_SE3(constraints=[0, 1, 2]),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        reduced_integration=False,
        constitutive_law=Harsch2021,
    )
    # mixed formulations
    # For shear-rigid rods Harsch2021 and Simo1986 coincide.
    cantilever(
        Rod=make_CosseratRod_SE3(mixed=True, constraints=[1, 2]),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        reduced_integration=False,
        constitutive_law=Simo1986,
    )
    cantilever(
        Rod=make_CosseratRod_SE3(mixed=True, constraints=[0, 1, 2]),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        reduced_integration=False,
        constitutive_law=Simo1986,
    )

    ############################
    # Quaternion interpolation #
    ############################
    # displacement based methods
    cantilever(
        Rod=make_CosseratRod_Quat(),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        reduced_integration=True,
        constitutive_law=Harsch2021,
    )
    cantilever(
        Rod=make_CosseratRod_Quat(constraints=[1, 2]),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        constitutive_law=Harsch2021,
    )
    cantilever(
        Rod=make_CosseratRod_Quat(constraints=[0, 1, 2]),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        constitutive_law=Harsch2021,
    )
    # mixed formulations
    # For shear-rigid rods Harsch2021 and Simo1986 coincide.
    cantilever(
        Rod=make_CosseratRod_Quat(mixed=True, constraints=[1, 2]),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        reduced_integration=False,
        constitutive_law=Simo1986,
    )
    cantilever(
        Rod=make_CosseratRod_Quat(mixed=True, constraints=[0, 1, 2]),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        reduced_integration=False,
        constitutive_law=Simo1986,
    )

    #####################
    # R12 interpolation #
    #####################
    # displacement base methods
    cantilever(
        Rod=make_CosseratRod_R12(),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        constitutive_law=Harsch2021,
    )
    cantilever(
        Rod=make_CosseratRod_R12(constraints=[1, 2]),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        constitutive_law=Harsch2021,
    )
    cantilever(
        Rod=make_CosseratRod_R12(constraints=[0, 1, 2]),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        constitutive_law=Harsch2021,
    )
    # mixed formulations
    # For shear-rigid rods Harsch2021 and Simo1986 coincide.
    cantilever(
        Rod=make_CosseratRod_R12(mixed=True, constraints=[1, 2]),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        reduced_integration=False,
        constitutive_law=Simo1986,
    )
    cantilever(
        Rod=make_CosseratRod_R12(mixed=True, constraints=[0, 1, 2]),
        nelements=10,
        polynomial_degree=2,
        n_load_steps=3,
        reduced_integration=False,
        constitutive_law=Simo1986,
    )
