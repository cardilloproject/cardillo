from cardillo.beams import (
    RectangularCrossSection,
    Simo1986,
    Harsch2021,
    animate_beam,
)

from cardillo.beams.cosseratRodPGMixed import (
    CosseratRodPG_R12Mixed,
    CosseratRodPG_QuatMixed,
    CosseratRodPG_SE3Mixed,
)

from cardillo.constraints import RigidConnection
from cardillo.solver import Newton
from cardillo.forces import Force, K_Moment

from cardillo.math import e2, e3

from cardillo.visualization import Export

from cardillo import System

import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

""" Cantilever beam examples from 
Harsch, J., Capobianco, G. and Eugster, S. R., "Finite element formulations for constrained spatial nonlinear beam theories", 2021.
load_type = "dead_load_and_moment": 4.1 Elliptic integral solutions of Euler's elastica
"""

def cantilever(
    Rod=CosseratRodPG_R12Mixed,
    nelements=10,
    polynomial_degree=2,
    n_load_steps=10,
    rod_hypothesis_penalty="shear_deformable",
    VTK_export=False,
    mixed=False,
    reduced_integration=True,
    constitutive_law=Harsch2021,
):

    # geometry of the rod
    length = 2 * np.pi
    # just for visualization purposes
    slenderness = 1.0e2
    width = length / slenderness

    # cross section
    line_density = 1
    cross_section = RectangularCrossSection(line_density, width, width)
    A = cross_section.area
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    # material model
    if rod_hypothesis_penalty == "shear_deformable":
        Ei = np.array([5, 1, 1])
    elif rod_hypothesis_penalty == "shear_rigid":
        Ei = np.array([5, 1e3, 1e3])
    elif rod_hypothesis_penalty == "inextensible_shear_rigid":
        Ei = np.array([5, 1, 1]) * 1e4

    Fi = np.array([0.5, 2, 2])

    material_model = constitutive_law(Ei, Fi)

    # position and orientation of left point
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    # construct system
    system = System()

    # compute straight initial configuration of cantilever
    q0 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        r_OP=r_OP0,
        A_IK=A_IK0,
        mixed=mixed,
    )
    # construct cantilever
    cantilever = Rod(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        mixed=mixed,
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

    system.assemble()

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

    path = Path(__file__)

    # add reference solution to the animation
    centerline_T = np.loadtxt(
        Path(path.parent, "cantilever_data", "dead_load_and_moment_centerline_T_numeric.csv"),
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
        Path(path.parent, "cantilever_data", "dead_load_and_moment_centerline_EB_numeric.csv"),
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
        Path(path.parent, "cantilever_data", "dead_load_and_moment_centerline_IEB_analytic.csv"),
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

    ################################################
    # load: dead load and moment at cantilever tip #
    ################################################

    # R12 interpolation: 
    # cantilever(Rod=CosseratRodPG_R12Mixed, nelements=10, polynomial_degree=2, n_load_steps = 3, constitutive_law=Harsch2021)
    # cantilever(Rod=CosseratRodPG_R12Mixed, nelements=10, polynomial_degree=2, n_load_steps = 14, rod_hypothesis_penalty="shear_rigid", constitutive_law=Harsch2021)
    # cantilever(Rod=CosseratRodPG_R12Mixed, nelements=10, polynomial_degree=2, n_load_steps = 3, rod_hypothesis_penalty="inextensible_shear_rigid", constitutive_law=Harsch2021)

    # For shear-rigid rods Harsch2021 and Simo1986 coincide. For Simo1986 also more robust mixed formulations are available
    # cantilever(Rod=CosseratRodPG_R12Mixed, nelements=10, polynomial_degree=2, n_load_steps = 3, rod_hypothesis_penalty="shear_rigid", mixed=True, reduced_integration=False,
    #            constitutive_law=Simo1986)
    # cantilever(Rod=CosseratRodPG_R12Mixed, nelements=10, polynomial_degree=2, n_load_steps = 3, rod_hypothesis_penalty="inextensible_shear_rigid", mixed=True, reduced_integration=False,
            #    constitutive_law=Simo1986)

    # Quaternion interpolation: 
    # cantilever(Rod=CosseratRodPG_QuatMixed, nelements=10, polynomial_degree=2, n_load_steps = 3, constitutive_law=Harsch2021)
    # cantilever(Rod=CosseratRodPG_QuatMixed, nelements=10, polynomial_degree=2, n_load_steps = 14, rod_hypothesis_penalty="shear_rigid", constitutive_law=Harsch2021)
    # cantilever(Rod=CosseratRodPG_QuatMixed, nelements=10, polynomial_degree=2, n_load_steps = 3, rod_hypothesis_penalty="inextensible_shear_rigid", constitutive_law=Harsch2021)

    # # For shear-rigid rods Harsch2021 and Simo1986 coincide. For Simo1986 also more robust mixed formulations are available
    # cantilever(Rod=CosseratRodPG_QuatMixed, nelements=10, polynomial_degree=2, n_load_steps = 3, rod_hypothesis_penalty="shear_rigid", mixed=True, reduced_integration=False,
    #            constitutive_law=Simo1986)
    # cantilever(Rod=CosseratRodPG_QuatMixed, nelements=10, polynomial_degree=2, n_load_steps = 3, rod_hypothesis_penalty="inextensible_shear_rigid", mixed=True, reduced_integration=False,
    #            constitutive_law=Simo1986)
    
    # SE3 interpolation: 
    cantilever(Rod=CosseratRodPG_SE3Mixed, nelements=10, polynomial_degree=1, n_load_steps = 3, constitutive_law=Harsch2021)
    # cantilever(Rod=CosseratRodPG_SE3Mixed, nelements=10, polynomial_degree=1, n_load_steps = 14, rod_hypothesis_penalty="shear_rigid", constitutive_law=Harsch2021)
    # cantilever(Rod=CosseratRodPG_SE3Mixed, nelements=10, polynomial_degree=1, n_load_steps = 3, rod_hypothesis_penalty="inextensible_shear_rigid", constitutive_law=Harsch2021)

    # For shear-rigid rods Harsch2021 and Simo1986 coincide. For Simo1986 also more robust mixed formulations are available
    cantilever(Rod=CosseratRodPG_SE3Mixed, nelements=10, polynomial_degree=1, n_load_steps = 3, rod_hypothesis_penalty="shear_rigid", mixed=True, reduced_integration=False,
               constitutive_law=Simo1986)
    cantilever(Rod=CosseratRodPG_SE3Mixed, nelements=10, polynomial_degree=1, n_load_steps = 3, rod_hypothesis_penalty="inextensible_shear_rigid", mixed=True, reduced_integration=False,
               constitutive_law=Simo1986)
