from cardillo.math import e1, e2, e3
from cardillo.beams import (
    RectangularCrossSection,
    CircularCrossSection,
    Simo1986,
)
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.beams import animate_beam
from cardillo.beams.cosseratRodPGMixed import (
    CosseratRodPG_R12Mixed,
    CosseratRodPG_QuatMixed,
    CosseratRodPG_SE3Mixed,
)

from cardillo.forces import K_Moment, Force
from cardillo import System
from cardillo.solver import Newton, Riks, EulerBackward, ScipyIVP
from cardillo.visualization import Export

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Rod = CosseratRodPG_R12Mixed
Rod = CosseratRodPG_SE3Mixed

# slenderness = 1.0e1
# atol = 1.0e-8
slenderness = 1.0e2
atol = 1.0e-10
# slenderness = 1.0e3
# atol = 1.0e-12
# slenderness = 1.0e4
# atol = 1.0e-14
# atol = 1.0e-12

# number of elements
# nelements = 20
nelements = 20
# used polynomial degree
# polynomial_degree = 1
# polynomial_degree = 2
polynomial_degree = 2
basis = "Lagrange"
# mixed=False
mixed=True
red_int=False



if __name__ == "__main__":
    #############################
    # helix example of Harsch2023
    #############################

    # Young's and shear modulus
    E = 1.0
    G = 0.5

    # length od the rod
    length = 1.0e3

    # used cross section
    width = length / slenderness

    # cross section and quadratic beam material
    line_density = 1
    # cross_section = RectangularCrossSection(line_density, width, width)
    cross_section = CircularCrossSection(line_density, width)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    q0 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        mixed=mixed,
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
        reduced_integration=red_int,
        mixed=mixed,
    )

    frame1 = Frame(r_OP0, A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, cantilever, frame_ID2=(0,))

    # moment at right end
    Fi = material_model.Fi
    M = lambda t: 2 * np.pi / length * (e1 * Fi[0] + e3 * Fi[2]) * t * 1.5
    moment = K_Moment(M, cantilever, (1,))

    # assemble the system
    system = System()
    system.add(cantilever)
    system.add(frame1)
    system.add(joint1)
    system.add(moment)
    # system.add(force)
    system.assemble()

    n_load_steps = 10
    # n_load_steps = 400
    # n_load_steps = 10
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        max_iter=30,
        atol=atol,
    )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # q1 = q[-1]
    # for i in range(cantilever.nnodes_psi - 1):
    #     Pi = q1[cantilever.nodalDOF_psi[i]]
    #     Pi1 = q1[cantilever.nodalDOF_psi[i + 1]]
    #     inner = Pi @ Pi1
    #     print(f"i: {i}")
    #     if inner < 0:
    #         print("wrong hemisphere!")
    #     else:
    #         print(f"correct hemisphere")


    # ############
    # # VTK export
    # ############
    path = Path(__file__)
    e = Export(path.parent, path.stem, True, 30, sol)
    e.export_contr(cantilever, level="centerline + directors", num=20)
    e.export_contr(cantilever, level="volume", n_segments=nelements, num=50)
    ###########
    # animation
    ###########
    animate_beam(t, q, [cantilever], length, show=True, repeat=False)


