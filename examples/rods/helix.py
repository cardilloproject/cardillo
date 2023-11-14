from cardillo.math import e1, e2, e3
from cardillo.beams import (
    RectangularCrossSection,
    CircularCrossSection,
    Simo1986,
)
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.beams import animate_beam
from cardillo.beams.cosseratRodPG import (
    make_CosseratRod_R12,
    make_CosseratRod_Quat,
    make_CosseratRod_SE3,
)

from cardillo.forces import K_Moment, Force
from cardillo import System
from cardillo.solver import Newton
from cardillo.visualization import Export

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def helix(Rod, nelements=20, polynomial_degree=2, n_load_steps=10, reduced_integration=True, VTK_export=False):
    
    # geometry of the rod
    length = 1.0e3

    # cross section properties 
    # slenderness = 1.0e1
    # atol = 1.0e-8
    slenderness = 1.0e2
    atol = 1.0e-10
    # slenderness = 1.0e3
    # atol = 1.0e-12
    # slenderness = 1.0e4
    # atol = 1.0e-12

    width = length / slenderness
    line_density = 1
    cross_section = CircularCrossSection(line_density, width)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)

    # material model
    E = 1.0     # Young's modulus
    G = 0.5     # shear modulus
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

    # construct system
    system = System()

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    q0 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
    )
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
        reduced_integration=reduced_integration,
        mixed=False,
    )

    # clamping left
    clamping_left = RigidConnection(system.origin, cantilever, frame_ID2=(0,))
    system.add(cantilever)
    system.add(clamping_left)

    # moment at right end
    Fi = material_model.Fi
    M = lambda t: 2 * np.pi / length * (e1 * Fi[0] + e3 * Fi[2]) * t * 1.5
    moment = K_Moment(M, cantilever, (1,))
    system.add(moment)

    system.assemble()

    # add Newton solver
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        max_iter=30,
        atol=atol,
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
        show=False,
        repeat=False,
    )

    plt.show()


if __name__ == "__main__":
    #############################
    # helix example of Harsch2023
    #############################

    # SE3 interpolation:
    # helix(Rod=make_CosseratRod_SE3(mixed=True), nelements=10, polynomial_degree=1, n_load_steps = 10, reduced_integration=False)
    helix(Rod=make_CosseratRod_SE3(mixed=False), nelements=10, polynomial_degree=1, n_load_steps = 300, reduced_integration=True)

    # Quaternion interpolation:
    
    # R12 interpolation:
    # helix(Rod=make_CosseratRod_R12(mixed=True), nelements=10, polynomial_degree=2, n_load_steps = 3)
    # helix(Rod=make_CosseratRod_SE3(mixed=True), nelements=10, polynomial_degree=3, n_load_steps = 20, reduced_integration=False, VTK_export=True)
    # helix(Rod=make_CosseratRod_Quat(mixed=True), nelements=20, polynomial_degree=2, n_load_steps = 10, reduced_integration=False, VTK_export=False)

    


    
