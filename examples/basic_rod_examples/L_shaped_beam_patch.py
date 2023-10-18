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

from cardillo.math import e2, e3, A_IK_basic

from cardillo.visualization import Export

from cardillo import System

import numpy as np

from math import pi
import matplotlib.pyplot as plt
from pathlib import Path

""" Cantilever beam example from

Harsch, J., Capobianco, G. and Eugster, S. R., "Finite element formulations for constrained spatial nonlinear beam theories", 2021.
https://doi.org/10.1177/10812865211000790
4.4. Beam patches with slope discontinuity

Greco, L., "An iso-parametric G1-conforming finite element for the nonlinear analysis of Kirchhoff rod. Part I: the 2D case"
https://doi.org/10.1007/s00161-020-00861-9
4.2 L-shaped spring
"""

def L_shaped_beam_patch(
    Rod=CosseratRodPG_R12Mixed,
    nelements=10,
    polynomial_degree=2,
    n_load_steps=10,
    rod_hypothesis_penalty="shear_deformable",
    VTK_export=False,
    mixed=True,
    reduced_integration=False,
    constitutive_law=Simo1986,
):

    # geometry of the rod
    length = 2 * np.pi

    # cross section properties for visualization purposes
    slenderness = 1.0e2
    width = length / slenderness
    line_density = 1
    cross_section = RectangularCrossSection(line_density, width, width)
    A = cross_section.area
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    # material model
    Ei = np.array([5, 1, 1])
    Fi = np.array([0.5, 2, 2])
    material_model = constitutive_law(Ei, Fi)

    # construct system
    system = System()

    # compute straight initial configuration of cantilever
    q0 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        mixed=mixed,
    )
    # construct cantilever
    cantilever1 = Rod(
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

    r_OP = np.array([length, 0, 0])
    A_IK = A_IK_basic(-pi / 2).z()

    q0 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        mixed=mixed,
        r_OP=r_OP,
        A_IK=A_IK
    )

    cantilever2 = Rod(
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

    clamping_left = RigidConnection(system.origin, cantilever1, frame_ID2=(0,))
    rod_rod_clamping = RigidConnection(cantilever1, cantilever2, frame_ID1=(1,), frame_ID2=(0,))

    # assemble the system
    system.add(cantilever1)
    system.add(cantilever2)
    system.add(clamping_left)
    system.add(rod_rod_clamping)

    # moment at cantilever tip
    m = material_model.Fi[2] * 2 * np.pi / length
    M = lambda t: t * e3 * m
    moment = K_Moment(M, cantilever2, frame_ID=(1,))
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
        rod_list = [cantilever1, cantilever2]
        for (i, rod) in enumerate(rod_list):
            e.export_contr(
                rod,
                level="centerline + directors",
                num=3 * nelements,
                file_name="rod_curve",
            )
            e.export_contr(
                rod,
                continuity="C0",
                level="volume",
                n_segments=nelements,
                num=3 * nelements,
                file_name="rod_volume",
            )

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        [cantilever1, cantilever2],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever1.nelement + 1,
        repeat=True,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    Y_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)

    path = Path(__file__)

   

    # plot animation
    ax1.azim = -90
    ax1.elev = 72

    plt.show()


if __name__ == "__main__":    

    ################################################
    # load: dead load and moment at cantilever tip #
    ################################################

    # R12 interpolation: 
    # L_shaped_beam_patch(Rod=CosseratRodPG_R12Mixed, nelements=5, polynomial_degree=2, n_load_steps = 20, mixed=False, reduced_integration=True)
    # L_shaped_beam_patch(Rod=CosseratRodPG_R12Mixed, nelements=5, polynomial_degree=2, n_load_steps = 5, mixed=True, reduced_integration=False)

    # Quaternion interpolation: 
    # L_shaped_beam_patch(Rod=CosseratRodPG_QuatMixed, nelements=5, polynomial_degree=2, n_load_steps = 20, mixed=False, reduced_integration=True)
    # L_shaped_beam_patch(Rod=CosseratRodPG_QuatMixed, nelements=5, polynomial_degree=2, n_load_steps = 5, mixed=True, reduced_integration=False)

    # SE3 interpolation:
    L_shaped_beam_patch(Rod=CosseratRodPG_SE3Mixed, nelements=10, polynomial_degree=2, n_load_steps = 20, mixed=False, reduced_integration=False, VTK_export=True)
    # L_shaped_beam_patch(Rod=CosseratRodPG_SE3Mixed, nelements=5, polynomial_degree=2, n_load_steps = 5, mixed=True, reduced_integration=False)
    