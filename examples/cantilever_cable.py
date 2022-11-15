from cardillo.math import e1, e2, e3
from cardillo.discrete import Frame
from cardillo.constraints import SphericalJoint, RigidConnectionCable
from cardillo.beams import (
    animate_beam,
    Cable,
)
from cardillo.beams.cable import QuadraticMaterial
from cardillo.forces import K_Moment, K_Force, DistributedForce1DBeam
from cardillo import System
from cardillo.solver import Newton, RadauIIa, EulerBackward

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # number of elements
    nelements = 3

    # used polynomial degree
    polynomial_degree = 3
    basis = "Hermite"

    # beam parameters found in Section 5.1 Ibrahimbegovic1997
    L = np.pi
    k_e = 1.0e4
    k_b = 1.0e2

    # build quadratic material model
    material_model = QuadraticMaterial(k_e, k_b)

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    q0 = Cable.straight_configuration(
        basis,
        polynomial_degree,
        nelements,
        L,
        r_OP0,
        A_IK0,
    )
    u0 = np.zeros_like(q0)

    # Note: This is never used in statics!
    A_rho0 = 1.0

    beam = Cable(
        material_model,
        A_rho0,
        polynomial_degree,
        nelements,
        Q=q0,
        q0=q0,
        u0=u0,
        basis=basis,
    )

    # junctions
    frame_left = Frame()
    r_OP1 = np.array([L, 0, 0])
    frame_right = Frame(r_OP=r_OP1)

    # left and right joint
    # joint_left = RigidConnectionCable(frame, beam, frame_ID2=(0,))
    # joint_left = RigidConnectionCable(beam, frame, frame_ID1=(0,))
    joint_left = SphericalJoint(frame_left, beam, r_OB=r_OP0, frame_ID2=(0,))
    joint_right = SphericalJoint(frame_right, beam, r_OB=r_OP1, frame_ID2=(1,))

    # # moment at right end
    # Fi = material_model.Fi
    # M = lambda t: (e3 * 2 * np.pi * Fi[2] / L * t) * 0.25
    # moment = K_Moment(M, beam, (1,))

    # # force at the rght end
    # f = lambda t: t * e1 * 1.0e3
    # force = K_Force(f, beam, (1,))

    # line distributed body force
    # l = lambda t, xi: t * e3 * 2e1
    l = lambda t, xi: -e3 * 2e1
    line_force = DistributedForce1DBeam(l, beam)

    # assemble the system
    system = System()
    system.add(beam)
    system.add(frame_left)
    system.add(joint_left)
    system.add(frame_right)
    system.add(joint_right)
    # system.add(moment)
    # system.add(force)
    system.add(line_force)
    system.assemble()

    t1 = 1
    dt = 1.0e-2
    rtol = 1.0e-2
    atol = 1.0e-2
    dae_index = "GGL"
    # sol = RadauIIa(system, t1, dt, rtol, atol, dae_index).solve()
    sol = EulerBackward(system, t1, dt).solve()

    t = sol.t
    q = sol.q

    # t = [0]
    # q = [model.q0]

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam], L, show=True)
    exit()

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
