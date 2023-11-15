from cardillo.math import e1, e2, e3
from cardillo.beams import (
    RectangularCrossSection,
    CircularCrossSection,
    Simo1986,
)
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.beams import animate_beam
from cardillo.beams.cosseratRod import (
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
from math import pi

def helix(Rod, nelements=20, polynomial_degree=2, n_load_steps=10, reduced_integration=True, VTK_export=False):
    
    # geometry of the rod
    n = 2       # number of coils
    R0 = 10     # radius of the helix
    h = 50      # height of the helix
    c = h / (2 * R0 * pi * n) # pitch of the helix
    length = np.sqrt(1 + c**2) * R0 * 2 * pi * n
    cc = 1 / (np.sqrt(1 + c**2))

    alpha = lambda xi: 2 * pi * n * xi
    alpha_xi = 2 * pi * n

    # cross section properties 
    # slenderness = 1.0e1
    # atol = 1.0e-8
    # slenderness = 1.0e2
    # atol = 1.0e-9
    # slenderness = 1.0e3
    # atol = 1.0e-10
    slenderness = 1.0e4
    atol = 1.0e-12

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

    alpha_0 = alpha(0)

    r_OP0 = R0 * np.array([np.sin(alpha_0), -np.cos(alpha_0), c * alpha_0])

    e_x = cc * np.array([np.cos(alpha_0), np.sin(alpha_0), c])
    e_y = np.array([- np.sin(alpha_0), np.cos(alpha_0), 0])
    e_z = cc * np.array([- c * np.cos(alpha_0), - c * np.sin(alpha_0), 1])

    A_IK0 = np.vstack((e_x, e_y, e_z))
    A_IK0 = A_IK0.T

    q0 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        r_OP=r_OP0,
        A_IK=A_IK0,
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
    M = lambda t: (R0 * alpha_xi**2) / (length**2) * (c * e1 * Fi[0] + e3 * Fi[2]) * t
    # M = lambda t: 2 * np.pi / length * (e1 * Fi[0] + e3 * Fi[2]) * t * 1.5
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
    # SE3 interpolation:
    # helix(Rod=make_CosseratRod_SE3(mixed=True), nelements=5, polynomial_degree=1, n_load_steps = 1, reduced_integration=False)
    # helix(Rod=make_CosseratRod_SE3(mixed=False), nelements=10, polynomial_degree=1, n_load_steps = 499, reduced_integration=True)

    # Quaternion interpolation:
    # helix(Rod=make_CosseratRod_Quat(mixed=True), nelements=10, polynomial_degree=2, n_load_steps = 2, reduced_integration=False)
    # helix(Rod=make_CosseratRod_Quat(mixed=False), nelements=10, polynomial_degree=2, n_load_steps = 500, reduced_integration=True)
    
    # R12 interpolation:
    helix(Rod=make_CosseratRod_R12(mixed=True), nelements=10, polynomial_degree=2, n_load_steps = 2, reduced_integration=False)
    # helix(Rod=make_CosseratRod_R12(mixed=False), nelements=10, polynomial_degree=2, n_load_steps = 500, reduced_integration=True)

    


    
