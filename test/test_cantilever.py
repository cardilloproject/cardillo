from cardillo.rods import (
    RectangularCrossSection,
    Simo1986,
    Harsch2021,
    animate_beam,
)

from cardillo.rods.cosseratRod import make_CosseratRod

from cardillo.constraints import RigidConnection
from cardillo.solver import Newton, SolverOptions
from cardillo.forces import Force, B_Moment

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

It is a test for the static analysis of all different rod formulations in their different versions: 
(constrained) displacement-based and (constrained) mixed formulations.
"""


def cantilever(
    Rod,
    nelements=10,
    n_load_steps=3,
    constitutive_law=Harsch2021,
    title="set_a_plot_title",
):
    print(title)
    # geometry of the rod
    length = 2 * np.pi

    # cross section properties for visualization purposes
    slenderness = 1.0e2
    width = length / slenderness
    cross_section = RectangularCrossSection(width, width)

    # material properties
    Ei = np.array([5, 1, 1])
    Fi = np.array([0.5, 2, 2])
    material_model = constitutive_law(Ei, Fi)

    # construct system
    system = System()

    # compute straight initial configuration of cantilever
    q0 = Rod.straight_configuration(nelements, length)
    # construct cantilever
    cantilever = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
    )

    clamping_left = RigidConnection(system.origin, cantilever, xi2=(0,))

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
    moment = B_Moment(M, cantilever, (1,))
    system.add(moment)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # add Newton solver
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_max_iter=30, newton_atol=1.0e-8),
    )

    # solve nonlinear static equilibrium equations
    sol = solver.solve()

    # extract solutions
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

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
        repeat=False,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    Y_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)
    ax1.set_title(title)

    path = Path(__file__)  # creation of a current path

    # add reference solution to the animation
    centerline_T = np.loadtxt(
        Path(
            path.parent,
            "_data_cantilever",
            "centerline_T_numeric.csv",
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
            "_data_cantilever",
            "centerline_EB_numeric.csv",
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
            "_data_cantilever",
            "centerline_IEB_analytic.csv",
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

    plt.show(block=False)
    plt.pause(5)
    plt.close()


if __name__ == "__main__":
    ############################
    # Quaternion interpolation #
    ############################
    # displacement-based formulation

    cantilever(
        Rod=make_CosseratRod(interpolation="Quaternion", mixed=False),
        constitutive_law=Harsch2021,
        title="shear-deformable (blue): D-B quaternion interpolation",
    )

    cantilever(
        Rod=make_CosseratRod(
            interpolation="Quaternion", mixed=False, constraints=[1, 2]
        ),
        constitutive_law=Harsch2021,
        title="shear-rigid (green): constrained D-B quaternion interpolation",
    )

    cantilever(
        Rod=make_CosseratRod(
            interpolation="Quaternion", mixed=False, constraints=[0, 1, 2]
        ),
        constitutive_law=Harsch2021,
        title="inextensible shear-rigid (red): constrained D-B quaternion interpolation",
    )

    # mixed formulation
    # For shear-rigid rods Harsch2021 and Simo1986 coincide.
    cantilever(
        Rod=make_CosseratRod(
            interpolation="Quaternion", mixed=True, constraints=[1, 2]
        ),
        constitutive_law=Simo1986,
        title="shear-rigid (green): constrained mixed quaternion interpolation",
    )

    cantilever(
        Rod=make_CosseratRod(
            interpolation="Quaternion", mixed=True, constraints=[0, 1, 2]
        ),
        constitutive_law=Simo1986,
        title="inextensible shear-rigid (red): constrained mixed quaternion interpolation",
    )

    #####################
    # SE3 interpolation #
    #####################
    # displacement-based formulation
    cantilever(
        Rod=make_CosseratRod(interpolation="SE3", mixed=False),
        constitutive_law=Harsch2021,
        title="shear-deformable (blue): D-B SE3 interpolation",
    )

    cantilever(
        Rod=make_CosseratRod(interpolation="SE3", mixed=False, constraints=[1, 2]),
        constitutive_law=Harsch2021,
        title="shear-rigid (green): constrained D-B SE3 interpolation",
    )
    cantilever(
        Rod=make_CosseratRod(interpolation="SE3", mixed=False, constraints=[0, 1, 2]),
        constitutive_law=Harsch2021,
        title="inextensible shear-rigid (red): constrained D-B SE3 interpolation",
    )
    # mixed formulation
    # For shear-rigid rods Harsch2021 and Simo1986 coincide.
    cantilever(
        Rod=make_CosseratRod(interpolation="SE3", mixed=True, constraints=[1, 2]),
        constitutive_law=Simo1986,
        title="shear-rigid (green): constrained mixed SE3 interpolation",
    )
    cantilever(
        Rod=make_CosseratRod(interpolation="SE3", mixed=True, constraints=[0, 1, 2]),
        constitutive_law=Simo1986,
        title="inextensible shear-rigid (red): constrained mixed SE3 interpolation",
    )

    #####################
    # R12 interpolation #
    #####################
    # displacement-based formulation
    cantilever(
        Rod=make_CosseratRod(interpolation="R12", mixed=False),
        constitutive_law=Harsch2021,
        title="shear-deformable (blue): D-B R12 interpolation",
    )
    cantilever(
        Rod=make_CosseratRod(interpolation="R12", mixed=False, constraints=[1, 2]),
        constitutive_law=Harsch2021,
        title="shear-rigid (green): constrained D-B R12 interpolation",
    )
    cantilever(
        Rod=make_CosseratRod(interpolation="R12", mixed=False, constraints=[0, 1, 2]),
        constitutive_law=Harsch2021,
        title="inextensible shear-rigid (red): constrained D-B R12 interpolation",
    )
    # mixed formulation
    # For shear-rigid rods Harsch2021 and Simo1986 coincide.
    cantilever(
        Rod=make_CosseratRod(interpolation="R12", mixed=True, constraints=[1, 2]),
        constitutive_law=Simo1986,
        title="shear-rigid (green): constrained mixed R12 interpolation",
    )
    cantilever(
        Rod=make_CosseratRod(interpolation="R12", mixed=True, constraints=[0, 1, 2]),
        constitutive_law=Simo1986,
        title="inextensible shear-rigid (red): constrained mixed R12 interpolation",
    )
