from cardillo.math import e1, e2, e3
from cardillo.beams import (
    CircularCrossSection,
    Simo1986,
)
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.beams import (
    animate_beam,
    TimoshenkoAxisAngleSE3,
    TimoshenkoAxisAngle,
    DirectorAxisAngle,
    TimoshenkoDirectorDirac,
    TimoshenkoDirectorIntegral,
)
from cardillo.forces import K_Moment, K_Force, DistributedForce1DBeam
from cardillo import System
from cardillo.solver import Newton

import numpy as np
import matplotlib.pyplot as plt

# Beam = TimoshenkoAxisAngleSE3
Beam = DirectorAxisAngle
# Beam = TimoshenkoDirectorDirac
# Beam = TimoshenkoDirectorIntegral


if __name__ == "__main__":
    # number of elements
    nelements = 100

    # used polynomial degree
    polynomial_degree = 1
    basis = "B-spline"

    # beam parameters found in Section 5.1 Ibrahimbegovic1997
    L = np.pi
    EA = GA = 1.0e4
    GJ = EI = 1.0e2

    # build quadratic material model
    Ei = np.array([EA, GA, GA], dtype=float)
    Fi = np.array([GJ, EI, EI], dtype=float)
    material_model = Simo1986(Ei, Fi)

    # Note: This is never used in statics!
    line_density = 1.0
    radius = 1.0
    cross_section = CircularCrossSection(line_density, radius)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    if Beam == TimoshenkoAxisAngleSE3:
        q0 = TimoshenkoAxisAngleSE3.straight_configuration(
            1,
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        beam = TimoshenkoAxisAngleSE3(
            1,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            nelements,
            q0,
        )
    elif Beam == DirectorAxisAngle:
        q0 = DirectorAxisAngle.straight_configuration(
            polynomial_degree,
            polynomial_degree,
            basis,
            basis,
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        beam = DirectorAxisAngle(
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

        beam = Beam(
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
    else:
        raise NotImplementedError

    # junctions
    frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, beam, frame_ID2=(0,))

    # moment at right end
    Fi = material_model.Fi
    M = lambda t: (e3 * 2 * np.pi * Fi[2] / L * t) * 0.25
    moment = K_Moment(M, beam, (1,))

    # force at the rght end
    f = lambda t: t * e1 * 1.0e3
    force = K_Force(f, beam, (1,))

    # line distributed body force
    l = lambda t, xi: t * e3 * 2e1
    line_force = DistributedForce1DBeam(l, beam)

    # assemble the model
    model = System()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    # model.add(moment)
    # model.add(force)
    model.add(line_force)
    model.assemble()

    n_load_steps = int(10)

    solver = Newton(
        model,
        n_load_steps=n_load_steps,
        max_iter=30,
        atol=1.0e-8,
        numerical_jacobian=False,
    )
    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # t = [0]
    # q = [model.q0]

    ############################
    # Visualize tip displacement
    ############################
    elDOF = beam.qDOF[beam.elDOF[-1]]
    r_OP = np.array([beam.r_OP(ti, qi[elDOF], (1,)) for (ti, qi) in zip(t, q)])

    fig, ax = plt.subplots()
    ax.plot(t, r_OP[:, 0], "-k", label="x")
    ax.plot(t, r_OP[:, 1], "--k", label="y")
    ax.plot(t, r_OP[:, 2], "-.k", label="z")
    ax.set_xlabel("t")
    ax.set_ylabel("tip displacement")
    ax.grid()
    ax.legend()

    ######################################################
    # visualize strain measures of the final configuration
    ######################################################
    num = 100
    xis = np.linspace(0, 1, num=num)

    K_Gammas = np.zeros((num, 3))
    K_Kappas = np.zeros((num, 3))
    for i in range(num):
        K_Gammas[i], K_Kappas[i] = beam.strains(xis[i], sol.q[-1])

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(xis, K_Gammas[:, 0] - 1, "-r", label="K_Gamma0 - 1")
    ax[0].plot(xis, K_Gammas[:, 1], "--g", label="K_Gamma1")
    ax[0].plot(xis, K_Gammas[:, 2], "-.b", label="K_Gamma2")
    ax[0].set_xlabel("xi")
    ax[0].set_ylabel("K_Gamma")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(xis, K_Kappas[:, 0], "-r", label="K_Kappa0")
    ax[1].plot(xis, K_Kappas[:, 1], "--g", label="K_Kappa1")
    ax[1].plot(xis, K_Kappas[:, 2], "-.b", label="K_Kappa2")
    ax[1].set_xlabel("xi")
    ax[1].set_ylabel("K_Kappa")
    ax[1].grid()
    ax[1].legend()

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam], L, show=True)
