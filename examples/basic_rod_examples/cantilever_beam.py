from cardillo.beams import (
    K_R12_PetrovGalerkin_Quaternion,
    K_SE3_PetrovGalerkin_Quaternion,
    K_PetrovGalerkinQuaternionInterpolation,
    RectangularCrossSection,
    CircularCrossSection,
    Simo1986,
    animate_beam,
)
from cardillo.constraints import RigidConnection
from cardillo.solver import Newton
from cardillo.forces import Force, K_Force

from cardillo.math import e1, e2, e3, A_IK_basic

from cardillo.visualization import Export


from cardillo import System

# from cardillo.math import e1, e2, e3
# from cardillo.beams import (
#     RectangularCrossSection,
#     CircularCrossSection,
#     Simo1986,
# )
# from cardillo.discrete import Frame
# from cardillo.constraints import RigidConnection
# from cardillo.beams import animate_beam
# from cardillo.beams import (
#     K_R12_PetrovGalerkin_AxisAngle,
#     K_R12_PetrovGalerkin_Quaternion,
#     K_R12_PetrovGalerkin_R9,
# )
# from cardillo.beams import (
#     K_SE3_PetrovGalerkin_AxisAngle,
#     K_SE3_PetrovGalerkin_Quaternion,
#     K_SE3_PetrovGalerkin_R9,
# )
# from cardillo.beams import (
#     I_R12_BubonvGalerkin_R12_Dirac,
#     I_R12_BubonvGalerkin_R12_Integral,
#     K_Cardona,
#     K_TimoshenkoLerp,
#     K_PetrovGalerkinQuaternionInterpolation,
#     HigherOrder_K_SE3_PetrovGalerkin_Quaternion,
# )
# from cardillo.forces import K_Moment, Force
# from cardillo import System
# from cardillo.solver import Newton, Riks, EulerBackward, ScipyIVP
# from cardillo.visualization import Export

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


###################
# R12 interpolation
###################
# Rod = K_R12_PetrovGalerkin_Quaternion

# #####################
# # SE(3)-interpolation
# #####################
Rod = K_SE3_PetrovGalerkin_Quaternion

# ##########################
# # Quaternion interpolation
# ##########################
# Rod = K_PetrovGalerkinQuaternionInterpolation

n_load_steps = 100

# slenderness = 1.0e1
# atol = 1.0e-8
slenderness = 1.0e2
atol = 1.0e-8
# # slenderness = 1.0e3
# # atol = 1.0e-12
# # slenderness = 1.0e4
# # atol = 1.0e-14

# number of elements
nelements = 10

# used polynomial degree
polynomial_degree = 2

if __name__ == "__main__":
    #############################
    # helix example of Harsch2023
    #############################

    # Young's and shear modulus
    E = 1.0
    G = 0.5

    # length od the rod
    L = 1.0e3

    # used cross section
    width = L / slenderness

    # cross section and quadratic beam material
    line_density = 1
    cross_section = RectangularCrossSection(line_density, width, width)
    #     cross_section = CircularCrossSection(line_density, width)

    A = cross_section.area
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

    # position and orientation of left point
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)
    # A_IK0 = A_IK_basic(np.pi / 4).y()

    if Rod is K_SE3_PetrovGalerkin_Quaternion:
        q0 = Rod.straight_configuration(
            nelements,
            L,
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
            q0,
        )
    elif Rod is K_R12_PetrovGalerkin_Quaternion:
        q0 = Rod.straight_configuration(
            polynomial_degree,
            polynomial_degree,
            "Lagrange",
            "Lagrange",
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        cantilever = Rod(
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
            basis_r="Lagrange",
            basis_psi="Lagrange",
        )
    elif Rod is K_PetrovGalerkinQuaternionInterpolation:
        q0 = Rod.straight_configuration(
            polynomial_degree,
            "Lagrange",
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        cantilever = Rod(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            polynomial_degree,
            nelements,
            Q=q0,
            q0=q0,
            basis="Lagrange",
        )
    else:
        raise NotImplementedError

    system = System()

    clamping_left = RigidConnection(system.origin, cantilever, frame_ID2=(0,))

    # # moment at the beam's tip
    # Fi = material_model.Fi
    # m = Fi[2] * 2 * np.pi / L * 1.5
    # M = lambda t: t * e3 * m
    # moment = K_Moment(M, rod, (1,))

    # # dead load at the beam's tip
    F1 = lambda t: t * (e2 * 1e-2)
    print(f"f_max: {F1(1)}")
    force = Force(F1, cantilever, frame_ID=(1,))

    # follower force at the beam's tip
    F = lambda t: t * (-e3 * 1e-2 + e2 * 0) * 6
    print(f"f_max: {F(1)}")
    follower_force = K_Force(F, cantilever, frame_ID=(1,))

    #     # moment at right end
    #     Fi = material_model.Fi
    #     M = lambda t: 2 * np.pi / L * (e1 * Fi[0] + e3 * Fi[2]) * t * 1.5
    #     moment = K_Moment(M, rod, (1,))

    #     # # force at the rght end
    #     # f = lambda t: t * e1 * 1.0e3
    #     # force = K_Force(f, beam, (1,))

    #     # # line distributed body force
    #     # if statics:
    #     #     l = lambda t, xi: t * (0.5 * e2 - e3) * 5e1
    #     # else:
    #     #     l = lambda t, xi: (0.5 * e2 - e3) * 5e0
    #     # line_force = DistributedForce1DBeam(l, beam)

    # assemble the system
    system.add(cantilever)
    system.add(clamping_left)
    system.add(follower_force)
    system.add(force)

    #     system.add(moment)
    #     # system.add(force)
    system.assemble()

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



    ############
    # VTK export
    ############
    path = Path(__file__)
    e = Export(path.parent, path.stem, True, 30, sol)
    e.export_contr(
        cantilever, continuity="C0", level="volume", n_segments=nelements, num=3 * nelements
    )

    ###########
    # animation
    ###########
    animate_beam(t, q, [cantilever], L, show=True, 
                 n_frames=cantilever.nelement + 1, scale_di=100)


    # path = Path(__file__)
    # e = Export(path.parent, path.stem, True, 30, sol)
    # e.export_contr(cantilever, level="centerline + directors", num=20)
    # e.export_contr(cantilever, level="volume", n_segments=5, num=50)
