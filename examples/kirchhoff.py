from cardillo.math import e1, e2, e3
from cardillo.beams import (
    RectangularCrossSection,
    ShearStiffQuadratic,
)
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.beams import animate_beam, Kirchhoff
from cardillo.forces import K_Moment, K_Force, DistributedForce1DBeam
from cardillo import System
from cardillo.solver import Newton, EulerBackward, ScipyIVP
from cardillo.utility import Export

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


statics = True
# statics = False


if __name__ == "__main__":
    # number of elements
    nelements = 2

    # number of quadrature points
    nquadrature = 4

    # beam parameters found in Section 5.1 Ibrahimbegovic1997
    L = np.pi
    EA = 1.0e4
    GJ = EI = 1.0e2

    # build quadratic material model
    Fi = np.array([GJ, EI, EI], dtype=float)
    material_model = ShearStiffQuadratic(EA, Fi)

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

    # initial configuration
    q0 = Kirchhoff.straight_configuration(nelements, L, r_OP0, A_IK0)

    # build rod model
    beam = Kirchhoff(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
        nquadrature,
        nquadrature,
        q0,
    )

    # junctions
    frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, beam, frame_ID2=(0,))

    # moment at right end
    Fi = material_model.Fi
    M = lambda t: (e3 * 2 * np.pi * Fi[2] / L * t) * 0.25
    # if statics:
    #     M = lambda t: (e1 * Fi[0] + e3 * Fi[2]) * 1.0 * t * 2 * np.pi / L * 0.1
    # else:
    #     M = lambda t: (e1 * Fi[0] + e3 * Fi[2]) * 1.0 * 2 * np.pi / L * 0.05
    moment = K_Moment(M, beam, (1,))

    # # force at the rght end
    # # f = lambda t: t * e1 * 1.0e3
    # f = lambda t: t * e2 * 1.0e3
    # force = K_Force(f, beam, (1,))

    # # line distributed body force
    # if statics:
    #     l = lambda t, xi: t * (0.5 * e2 - e3) * 5e1
    # else:
    #     l = lambda t, xi: (0.5 * e2 - e3) * 1e0
    # line_force = DistributedForce1DBeam(l, beam)

    # assemble the system
    system = System()
    system.add(beam)
    system.add(frame1)
    system.add(joint1)
    system.add(moment)
    # system.add(force)
    # system.add(line_force)
    system.assemble()

    # animate_beam([0], [system.q0], [beam], L)
    # exit()

    if statics:
        n_load_steps = int(10)
        solver = Newton(
            system,
            n_load_steps=n_load_steps,
            max_iter=30,
            atol=1.0e-8,
        )
    else:
        t1 = 1
        dt = 2.5e-2
        solver = EulerBackward(system, t1, dt, method="index 3")
        # solver = ScipyIVP(system, t1, dt, rtol=1.0e-2, atol=1.0e-2)

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam], L, show=True)

    ############
    # VTK export
    ############
    path = Path(__file__)
    e = Export(path.parent, path.stem, True, 30, sol)
    e.export_contr(beam, level="centerline + directors", num=20)
    e.export_contr(beam, level="volume", n_segments=5, num=50)

    exit()

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

    # ######################################################
    # # visualize strain measures of the final configuration
    # ######################################################
    # num = 100
    # xis = np.linspace(0, 1, num=num)

    # K_Gammas = np.zeros((num, 3))
    # K_Kappas = np.zeros((num, 3))
    # for i in range(num):
    #     K_Gammas[i], K_Kappas[i] = beam.strains(xis[i], sol.q[-1])

    # fig, ax = plt.subplots(2, 1)

    # ax[0].plot(xis, K_Gammas[:, 0] - 1, "-r", label="K_Gamma0 - 1")
    # ax[0].plot(xis, K_Gammas[:, 1], "--g", label="K_Gamma1")
    # ax[0].plot(xis, K_Gammas[:, 2], "-.b", label="K_Gamma2")
    # ax[0].set_xlabel("xi")
    # ax[0].set_ylabel("K_Gamma")
    # ax[0].grid()
    # ax[0].legend()

    # ax[1].plot(xis, K_Kappas[:, 0], "-r", label="K_Kappa0")
    # ax[1].plot(xis, K_Kappas[:, 1], "--g", label="K_Kappa1")
    # ax[1].plot(xis, K_Kappas[:, 2], "-.b", label="K_Kappa2")
    # ax[1].set_xlabel("xi")
    # ax[1].set_ylabel("K_Kappa")
    # ax[1].grid()
    # ax[1].legend()
