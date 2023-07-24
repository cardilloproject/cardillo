from cardillo.math import e1, e2, e3, pi
from cardillo.beams import (
    RectangularCrossSection,
    Simo1986,
)
from cardillo.math.rotations import Exp_SO3
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.beams import (
    animate_beam,
    K_TimoshenkoAxisAngleSE3,
    K_Cardona,
)
from cardillo.forces import K_Force, K_Moment
from cardillo import System
from cardillo.solver import Newton

import numpy as np
import matplotlib.pyplot as plt

# Beam = K_TimoshenkoAxisAngleSE3
Beam = K_Cardona


def objectivity_quarter_circle():
    """This example examines shear and membrande locking as done in Meier2015.

    References:
    ===========
    Meier2015: https://doi.org/10.1016/j.cma.2015.02.029
    """
    polynomial_degree = 1
    basis = "Lagrange"

    # Young's and shear modulus
    E = 1.0  # Meier2015
    G = 0.5  # Meier2015
    nelements = 3
    # n = 10  # number of full rotations
    # n_load_steps = 500  # used for the paper
    # n = 1.25
    # n_load_steps = 75
    n = 2
    n_load_steps = 100
    t_star = 0.1  # fraction of deformation pseudo time
    t_star = 0.0001

    L = 1.0e3
    slenderness = 1.0e2
    # slenderness = 1.0e3
    # atol = 1.0e-9
    atol = 1.0e-10

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

    e = np.random.rand(3)

    def A_IK0(t):
        phi = (
            n * np.heaviside(t - t_star, 1.0) * (t - t_star) / (1.0 - t_star) * 2.0 * pi
        )
        return Exp_SO3(e1 * phi)
        # return Exp_SO3(e * phi)

    if Beam == K_TimoshenkoAxisAngleSE3:
        q0 = Beam.straight_configuration(
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0(0),
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
    elif Beam == K_Cardona:
        q0 = Beam.straight_configuration(
            polynomial_degree,
            polynomial_degree,
            basis,
            basis,
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0(0),
        )
        q0 = np.array(
            [
                0.00000000e00,
                3.21975275e02,
                5.57677536e02,
                6.43950551e02,
                0.00000000e00,
                8.62730151e01,
                3.21975276e02,
                6.43950551e02,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                5.23598776e-01,
                1.04719755e00,
                1.57079633e00,
            ]
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

    frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)
    joint1 = RigidConnection(frame1, beam, r_OP0, frame_ID2=(0,))

    # moment at the beam's tip
    Fi = material_model.Fi

    m = Fi[2] * 2 * np.pi / L * 0.25 * 0

    def M(t):
        M_max = m * e3
        if t <= t_star:
            return t / t_star * M_max
        else:
            return M_max

    moment = K_Moment(M, beam, (1,))

    # force at the beam's tip
    def f(t):
        f_max = (m / L) * e3 * 0
        if t <= t_star:
            return t / t_star * f_max
        else:
            return f_max

    force = K_Force(f, beam, frame_ID=(1,))

    # assemble the model
    model = System()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    model.add(moment)
    model.add(force)
    model.assemble()

    solver = Newton(
        model,
        n_load_steps=n_load_steps,
        max_iter=100,
        atol=atol,
    )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    #################################
    # visualize nodal rotation vector
    #################################
    fig, ax = plt.subplots()

    psi0 = q[:, beam.qDOF[beam.nodalDOF_psi[0]]]
    ax.plot(t, psi0[:, 0], "-r", label=f"psi0_0")
    ax.plot(t, psi0[:, 1], "--g", label=f"psi0_1")
    ax.plot(t, psi0[:, 2], "-.b", label=f"psi0_2")
    # for i, psi in enumerate(psi0.T):
    #     ax.plot(t, psi, label=f"psi0_{i}")

    ax.set_xlabel("t")
    ax.set_ylabel("nodal rotation vector left")
    ax.grid()
    ax.legend()

    fig, ax = plt.subplots()

    psi1 = q[:, beam.qDOF[beam.nodalDOF_psi[-1]]]
    for i, psi in enumerate(psi1.T):
        ax.plot(t, psi, label=f"psi1_{i}")

    ax.set_xlabel("t")
    ax.set_ylabel("nodal rotation vector right")
    ax.grid()
    ax.legend()

    #####################################
    # visualize nodal displacement vector
    #####################################
    fig, ax = plt.subplots()

    r1 = q[:, beam.qDOF[beam.nodalDOF_r[-1]]]
    for i, r in enumerate(r1.T):
        ax.plot(t, r, label=f"r1_{i}")

    ax.set_xlabel("t")
    ax.set_ylabel("nodal position vector")
    ax.grid()
    ax.legend()

    ############################
    # Visualize potential energy
    ############################
    E_pot = np.array([model.E_pot(ti, qi) for (ti, qi) in zip(t, q)])

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(t, E_pot)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("E_pot")
    ax[0].grid()

    idx = np.where(t > t_star)[0]
    ax[1].plot(t[idx], E_pot[idx])
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("E_pot")
    ax[1].grid()

    ##############
    # export plots
    ##############

    # potential energy
    header = "t, E_pot"
    export_data = np.vstack([np.arange(len(t)), E_pot]).T
    np.savetxt(
        "code/results/ObjectivityEpot_full.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    # rotation vector
    header = "t, psi0, psi1, psi2, abs_psi"
    export_data = np.vstack(
        [np.arange(len(t)), *psi1.T, np.linalg.norm(psi1, axis=1)]
    ).T
    np.savetxt(
        "code/results/ObjectivityRotationVector.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    # position vector
    header = "t, r0, r1, r2, abs_r"
    export_data = np.vstack([np.arange(len(t)), *r1.T, np.linalg.norm(r1, axis=1)]).T
    np.savetxt(
        "code/results/ObjectivityPositionVector.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    # ############
    # # VTK export
    # ############
    # from pathlib import Path
    # from cardillo.utility import Export

    # path = Path(__file__)
    # e = Export(path.parent, path.stem, True, 50, sol)
    # e.export_contr(beam, level="centerline + directors", num=50)
    # e.export_contr(beam, level="volume", n_segments=5, num=50)

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam], L, show=True)


if __name__ == "__main__":
    objectivity_quarter_circle()
