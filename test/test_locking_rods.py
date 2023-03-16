from cardillo.math import e1, e2, e3
from cardillo.beams import (
    RectangularCrossSection,
    Simo1986,
)
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.beams import animate_beam
from cardillo.beams import (
    K_R12_PetrovGalerkin_AxisAngle,
    K_R12_PetrovGalerkin_Quaternion,
    K_R12_PetrovGalerkin_R9,
)
from cardillo.beams import (
    K_SE3_PetrovGalerkin_AxisAngle,
    K_SE3_PetrovGalerkin_Quaternion,
    K_SE3_PetrovGalerkin_R9,
)
from cardillo.beams import (
    Crisfield1999,
    TimoshenkoDirectorDirac,
    TimoshenkoDirectorIntegral,
    K_Cardona,
    K_TimoshenkoLerp,
)
from cardillo.forces import K_Moment, Force
from cardillo import System
from cardillo.solver import Newton
from cardillo.utility import Export

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

###################
# R12 interpolation
###################
# Rod = K_R12_PetrovGalerkin_AxisAngle
# Rod = K_R12_PetrovGalerkin_Quaternion
Rod = K_R12_PetrovGalerkin_R9

#####################
# SE(3)-interpolation
#####################
# Rod = K_SE3_PetrovGalerkin_AxisAngle
# Rod = K_SE3_PetrovGalerkin_Quaternion
# Rod = K_SE3_PetrovGalerkin_R9

####################
# other formulations
####################
# Rod = Crisfield1999
# Rod = TimoshenkoDirectorDirac
# Rod = TimoshenkoDirectorIntegral
# Rod = I_DirectorAxisAngle
# Rod = K_Cardona
# Rod = K_TimoshenkoLerp

###################
# chose slenderness
###################
# slenderness = 1.0e1
# atol = 1.0e-8
# slenderness = 1.0e2
# atol = 1.0e-10
# slenderness = 1.0e3
# atol = 1.0e-12
slenderness = 1.0e4
atol = 1.0e-14

####################
# number of elements
####################
nelements = 10

########################
# used polynomial degree
########################
polynomial_degree = 1
# polynomial_degree = 2
basis = "Lagrange"


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

    if Rod in [
        K_SE3_PetrovGalerkin_AxisAngle,
        K_SE3_PetrovGalerkin_Quaternion,
        K_SE3_PetrovGalerkin_R9,
    ]:
        q0 = Rod.straight_configuration(
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
            nelements,
            q0,
        )
    elif Rod in [
        K_R12_PetrovGalerkin_AxisAngle,
        K_R12_PetrovGalerkin_Quaternion,
        K_R12_PetrovGalerkin_R9,
    ]:
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
    elif Rod in [TimoshenkoDirectorDirac, TimoshenkoDirectorIntegral]:
        q0 = Rod.straight_configuration(
            polynomial_degree,
            polynomial_degree,
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
            basis=basis,
        )

        # TODO: How works a generic translation between K_S_rho0 -> B_rho0 and K_I_rho0 - > C_rho0
        A_rho0 = A_rho0
        B_rho0 = K_S_rho0
        C_rho0 = np.diag(np.array([0.0, K_I_rho0[2, 2], K_I_rho0[1, 1]]))
        nquadrature = polynomial_degree + 1
        # nquadrature = polynomial_degree

        rod = Rod(
            cross_section,
            material_model,
            A_rho0,
            B_rho0,
            C_rho0,
            polynomial_degree,
            polynomial_degree,
            nquadrature,
            nelements,
            q0,
            basis=basis,
        )
    elif Rod == Crisfield1999:
        q0 = Crisfield1999.straight_configuration(
            polynomial_degree,
            polynomial_degree,
            basis,
            basis,
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        rod = Crisfield1999(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            polynomial_degree,
            polynomial_degree,
            nelements,
            q0,
            basis_r=basis,
            basis_psi=basis,
        )
    elif Rod in [K_Cardona, K_TimoshenkoLerp]:
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
            q0,
            basis_r=basis,
            basis_psi=basis,
        )
    else:
        raise NotImplementedError

    frame1 = Frame(r_OP0, A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, rod, frame_ID2=(0,))

    # moment at the beam's tip
    Fi = material_model.Fi
    m = Fi[2] * 2 * np.pi / L * 0.25
    M = lambda t: t * e3 * m
    moment = K_Moment(M, rod, (1,))

    # force at the beam's tip
    f = m / L
    F = lambda t: t * f * e3
    print(f"f_max: {F(1)}")
    force = Force(F, rod, frame_ID=(1,))

    # assemble the system
    system = System()
    system.add(rod)
    system.add(frame1)
    system.add(joint1)
    system.add(moment)
    system.add(force)
    system.assemble()

    # solve static system
    n_load_steps = 50
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

    ###########
    # animation
    ###########
    animate_beam(t, q, [rod], L, show=True)

    # ############
    # # VTK export
    # ############
    # path = Path(__file__)
    # e = Export(path.parent, path.stem, True, 30, sol)
    # e.export_contr(beam, level="centerline + directors", num=20)
    # e.export_contr(beam, level="volume", n_segments=5, num=50)
