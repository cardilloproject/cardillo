from cardillo.math import e1, e2, e3
from cardillo.beams import (
    RectangularCrossSection,
    CircularCrossSection,
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
    I_R12_BubonvGalerkin_R12_Dirac,
    I_R12_BubonvGalerkin_R12_Integral,
    K_Cardona,
    K_TimoshenkoLerp,
    K_PetrovGalerkinQuaternionInterpolation,
    HigherOrder_K_SE3_PetrovGalerkin_Quaternion,
)
from cardillo.forces import K_Moment, Force
from cardillo import System
from cardillo.solver import Newton, Riks, EulerBackward, ScipyIVP
from cardillo.visualization import Export

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

###################
# R12 interpolation
###################
# Rod = K_R12_PetrovGalerkin_AxisAngle
# Rod = K_R12_PetrovGalerkin_Quaternion
# Rod = K_R12_PetrovGalerkin_R9

#####################
# SE(3)-interpolation
#####################
# Rod = K_SE3_PetrovGalerkin_AxisAngle
# Rod = K_SE3_PetrovGalerkin_Quaternion
# Rod = K_SE3_PetrovGalerkin_R9

Rod = K_PetrovGalerkinQuaternionInterpolation
# Rod = HigherOrder_K_SE3_PetrovGalerkin_Quaternion

statics = True
# statics = False

slenderness = 1.0e1
atol = 1.0e-8
# slenderness = 1.0e2
# atol = 1.0e-10
# slenderness = 1.0e3
# atol = 1.0e-12
# slenderness = 1.0e4
# atol = 1.0e-14

# number of elements
# nelements = 20
nelements = 5

# used polynomial degree
# polynomial_degree = 1
# polynomial_degree = 2
polynomial_degree = 3
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
    elif Rod is K_PetrovGalerkinQuaternionInterpolation:
        q0 = Rod.straight_configuration(
            polynomial_degree,
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
            nelements,
            Q=q0,
            q0=q0,
            basis=basis,
        )
    elif Rod is HigherOrder_K_SE3_PetrovGalerkin_Quaternion:
        q0 = Rod.straight_configuration(
            polynomial_degree,
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
            nelements,
            Q=q0,
            q0=q0,
        )
    else:
        raise NotImplementedError

    frame1 = Frame(r_OP0, A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, rod, frame_ID2=(0,))

    # # moment at the beam's tip
    # Fi = material_model.Fi
    # m = Fi[2] * 2 * np.pi / L * 1.5
    # M = lambda t: t * e3 * m
    # moment = K_Moment(M, rod, (1,))

    # # force at the beam's tip
    # f = m / L * 10e-1
    # F = lambda t: t * f * e3
    # print(f"f_max: {F(1)}")
    # force = Force(F, rod, frame_ID=(1,))

    # moment at right end
    Fi = material_model.Fi
    M = lambda t: 2 * np.pi / L * (e1 * Fi[0] + e3 * Fi[2]) * t * 1.5
    moment = K_Moment(M, rod, (1,))

    # # force at the rght end
    # f = lambda t: t * e1 * 1.0e3
    # force = K_Force(f, beam, (1,))

    # # line distributed body force
    # if statics:
    #     l = lambda t, xi: t * (0.5 * e2 - e3) * 5e1
    # else:
    #     l = lambda t, xi: (0.5 * e2 - e3) * 5e0
    # line_force = DistributedForce1DBeam(l, beam)

    # assemble the system
    system = System()
    system.add(rod)
    system.add(frame1)
    system.add(joint1)
    system.add(moment)
    # system.add(force)
    system.assemble()

    if statics:
        n_load_steps = 40
        solver = Newton(
            system,
            n_load_steps=n_load_steps,
            max_iter=30,
            atol=atol,
        )
        # solver = Riks(
        #     system,
        #     atol=1e-8,
        # )
    else:
        t1 = 1
        dt = 2.5e-2
        solver = EulerBackward(system, t1, dt, method="index 1")
        # solver = EulerBackward(system, t1, dt, method="index 2")
        # solver = EulerBackward(system, t1, dt, method="index 3")
        # solver = EulerBackward(system, t1, dt, method="index 2 GGL")
        # solver = ScipyIVP(system, t1, dt, rtol=1.0e-2, atol=1.0e-2)
        # solver = ScipyIVP(system, t1, dt, rtol=1.0e-2, atol=1.0e-2, method="Radau")

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    q1 = q[-1]
    for i in range(rod.nnodes_psi - 1):
        Pi = q1[rod.nodalDOF_psi[i]]
        Pi1 = q1[rod.nodalDOF_psi[i + 1]]
        inner = Pi @ Pi1
        print(f"i: {i}")
        if inner < 0:
            print("wrong hemisphere!")
        else:
            print(f"correct hemisphere")

    ###########
    # animation
    ###########
    animate_beam(t, q, [rod], L, show=True)

    # ############
    # # VTK export
    # ############
    # path = Path(__file__)
    # e = Export(path.parent, path.stem, True, 30, sol)
    # e.export_contr(rod, level="centerline + directors", num=20)
    # e.export_contr(rod, level="volume", n_segments=5, num=50)
