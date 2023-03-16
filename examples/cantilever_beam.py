from cardillo.math import e1, e2, e3
from cardillo.beams import (
    CircularCrossSection,
    RectangularCrossSection,
    Simo1986,
)
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.beams import (
    animate_beam,
    I_R12_PetrovGalerkin_AxisAngle,
    K_R12_PetrovGalerkin_AxisAngle,
    K_R12_PetrovGalerkin_Quaternion,
    K_R12_PetrovGalerkin_R9,
    K_SE3_PetrovGalerkin_AxisAngle,
    K_SE3_PetrovGalerkin_Quaternion,
    Crisfield1999,
    TimoshenkoDirectorDirac,
    TimoshenkoDirectorIntegral,
    K_Cardona,
    K_TimoshenkoLerp,
)
from cardillo.forces import K_Moment, K_Force, DistributedForce1DBeam
from cardillo import System
from cardillo.solver import Newton, EulerBackward, ScipyIVP
from cardillo.utility import Export

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

###################
# R12 interpolation
###################
# Beam = K_R12_PetrovGalerkin_AxisAngle
# Beam = K_R12_PetrovGalerkin_Quaternion
# Beam = K_R12_PetrovGalerkin_R9

#####################
# SE(3)-interpolation
#####################
# Beam = K_SE3_PetrovGalerkin_AxisAngle
Beam = K_SE3_PetrovGalerkin_Quaternion
# Beam = Crisfield1999
# Beam = TimoshenkoDirectorDirac
# Beam = TimoshenkoDirectorIntegral
# Beam = I_DirectorAxisAngle
# Beam = K_Cardona
# Beam = K_TimoshenkoLerp

statics = True
# statics = False


if __name__ == "__main__":
    # number of elements
    nelements = 10
    # nelements = 5

    # used polynomial degree
    polynomial_degree = 3
    basis = "Lagrange"
    # basis = "B-spline"

    L = np.pi
    # EA = GA = 1.0e6
    EA = GA = 1.0e4
    # EA = GA = 1.0e2
    GJ = EI = 1.0e2

    # build quadratic material model
    Ei = np.array([EA, GA, GA], dtype=float)
    Fi = np.array([GJ, EI, EI], dtype=float)
    material_model = Simo1986(Ei, Fi)

    # Note: This is never used in statics!
    line_density = 1.0
    # radius = 0.1
    # cross_section = CircularCrossSection(line_density, radius)
    width = 0.1
    height = 0.2
    cross_section = RectangularCrossSection(line_density, width, height)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    if Beam in [K_SE3_PetrovGalerkin_AxisAngle, K_SE3_PetrovGalerkin_Quaternion]:
        q0 = Beam.straight_configuration(
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        beam = Beam(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            nelements,
            q0,
        )
    elif Beam in [
        K_R12_PetrovGalerkin_AxisAngle,
        K_R12_PetrovGalerkin_Quaternion,
        K_R12_PetrovGalerkin_R9,
    ]:
        q0 = Beam.straight_configuration(
            polynomial_degree,
            polynomial_degree,
            basis,
            basis,
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        beam = Beam(
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
    elif Beam == I_R12_PetrovGalerkin_AxisAngle:
        q0 = I_R12_PetrovGalerkin_AxisAngle.straight_configuration(
            polynomial_degree,
            polynomial_degree,
            basis,
            basis,
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        beam = I_R12_PetrovGalerkin_AxisAngle(
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
    elif Beam in [TimoshenkoDirectorDirac, TimoshenkoDirectorIntegral]:
        q0 = Beam.straight_configuration(
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

        beam = Beam(
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
    elif Beam == Crisfield1999:
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
        beam = Crisfield1999(
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
    elif Beam in [K_Cardona, K_TimoshenkoLerp]:
        q0 = Beam.straight_configuration(
            polynomial_degree,
            polynomial_degree,
            basis,
            basis,
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        beam = Beam(
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

    # junctions
    from cardillo.math import Exp_SO3

    frame1 = Frame(r_OP=np.random.rand(3), A_IK=Exp_SO3(np.random.rand(3)))

    # left and right joint
    joint1 = RigidConnection(frame1, beam, frame_ID2=(0,))

    # moment at right end
    Fi = material_model.Fi
    # M = lambda t: (e3 * 2 * np.pi * Fi[2] / L * t) * 2
    M = lambda t: 2 * np.pi / L * (e1 * Fi[0] + e3 * Fi[2]) * t * 2
    # if statics:
    #     M = lambda t: (e1 * Fi[0] + e3 * Fi[2]) * 1.0 * t * 2 * np.pi / L * 0.5
    # else:
    #     M = lambda t: (e1 * Fi[0] + e3 * Fi[2]) * 1.0 * 2 * np.pi / L * 0.05
    moment = K_Moment(M, beam, (1,))

    # force at the rght end
    f = lambda t: t * e1 * 1.0e3
    force = K_Force(f, beam, (1,))

    # line distributed body force
    if statics:
        l = lambda t, xi: t * (0.5 * e2 - e3) * 5e1
    else:
        l = lambda t, xi: (0.5 * e2 - e3) * 5e0
    line_force = DistributedForce1DBeam(l, beam)

    # assemble the system
    system = System()
    system.add(beam)
    system.add(frame1)
    system.add(joint1)
    if statics:
        system.add(moment)
        # system.add(force)
    else:
        system.add(line_force)
    system.assemble()

    # animate_beam([0], [system.q0], [beam], L)
    # exit()

    if statics:
        # n_load_steps = 50
        n_load_steps = 75
        # n_load_steps = 100
        # n_load_steps = 500
        solver = Newton(
            system,
            n_load_steps=n_load_steps,
            max_iter=30,
            atol=1.0e-8,
        )
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

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam], L, show=True)

    # ############
    # # VTK export
    # ############
    # path = Path(__file__)
    # e = Export(path.parent, path.stem, True, 30, sol)
    # e.export_contr(beam, level="centerline + directors", num=20)
    # e.export_contr(beam, level="volume", n_segments=5, num=50)
