from cardillo.rods import (
    RectangularCrossSection,
    Simo1986,
    animate_beam,
)

from cardillo.rods.cosseratRod import (
    make_CosseratRod_Quat,
)

from cardillo.constraints import RigidConnection
from cardillo.solver import Newton, SolverOptions
from cardillo.forces import Force, K_Moment, K_Force

from cardillo.math import e1, e2, e3, A_IK_basic

from cardillo.visualization import Export

from cardillo import System

import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path
from math import pi

""" 4.4. Beam patches with slope discontinuity

Harsch, J. , Capobianco, G. and Eugster, S. R., "Finite element formulations for
constrained spatial nonlinear beam theories", 2021. 
https://doi.org/10.1177/10812865211000790

"""


def beam_patches_3d(
    Rod,
    nelements=5,
    polynomial_degree=2,
    n_load_steps=10,
    VTK_export=True,
    reduced_integration=True,
):
    # cross section properties
    length = 1
    width = 0.1
    cross_section = RectangularCrossSection(width, width)
    A = cross_section.area
    I1, I2, I3 = np.diag(cross_section.second_moment)

    # material model
    EE = 1.0e6
    GG = EE / 2
    Ei = np.array([EE * A, GG * A, GG * A])
    Fi = np.array([GG * I1, EE * I2, EE * I3])

    material_model = Simo1986(Ei, Fi)

    # starting positions and orientations
    r_OP0 = np.array([[0, 0, 0], [length, 0, 0], [length, length, 0]])
    A_IK0 = np.zeros((3, 3, 3))
    A_IK0[0] = np.eye(3)
    A_IK0[1] = A_IK_basic(np.pi / 2).z()
    A_IK0[2] = A_IK_basic(-np.pi / 2).y()

    # construct system
    system = System()

    rod = []
    for i in range(3):
        q0 = Rod.straight_configuration(
            nelements,
            length,
            polynomial_degree=polynomial_degree,
            r_OP=r_OP0[i],
            A_IK=A_IK0[i],
        )
        rodi = Rod(
            cross_section,
            material_model,
            nelements,
            Q=q0,
            q0=q0,
            polynomial_degree=polynomial_degree,
            reduced_integration=reduced_integration,
        )
        rod.append(rodi)
        system.add(rodi)

    clamping = RigidConnection(rod[0], system.origin, frame_ID1=(0,))
    connection_c0c1 = RigidConnection(rod[0], rod[1], frame_ID1=(1,), frame_ID2=(0,))
    connection_c1c2 = RigidConnection(rod[1], rod[2], frame_ID1=(1,), frame_ID2=(0,))

    system.add(clamping)
    system.add(connection_c0c1)
    system.add(connection_c1c2)

    # applied tip load
    tip_force = 10
    F = lambda t: -tip_force * t * (e1 + e3)
    force = Force(F, rod[2], frame_ID=(1,))
    system.add(force)

    # assemble the system
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # add Newton solver
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_max_iter=30, newton_atol=1.0e-10),
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
        for i, rodi in enumerate(rod):
            e.export_contr(
                rodi,
                level="centerline + directors",
                num=3 * nelements,
                file_name=f"rod_{i}_curve",
            )
            e.export_contr(
                rodi,
                continuity="C0",
                level="volume",
                n_segments=nelements,
                num=3 * nelements,
                file_name=f"rod_{i}_volume",
            )

    u = np.zeros((3, nt))

    q_c2 = sol.q[:, rod[2].qDOF]
    qe_c2 = q_c2[:, rod[2].local_qDOF_P((1,))]

    r_OP0 = rod[2].r_OP(0, qe_c2[0], frame_ID=(1,))
    for i in range(nt):
        r_OPi = rod[2].r_OP(0, qe_c2[i], frame_ID=(1,))
        u[:, i] = r_OPi - r_OP0

    path = Path(__file__)

    force_u_Harsch = np.loadtxt(
        Path(path.parent, "_data_beam_patches_3d", "force_u_Harsch2021.csv"),
        delimiter=",",
        skiprows=1,
    )

    _, ax = plt.subplots(1, 1)

    ax.plot(tip_force * t, u[0], "-", color="red", label="u_x: Cosserat rod")
    ax.plot(tip_force * t, u[1], "-", color="green", label="u_y: Cosserat rod")
    ax.plot(tip_force * t, u[2], "-", color="blue", label="u_z: Cosserat rod")
    # ax[0].plot(force_u_x_Simo[:, 0], forc'e_u_x_Simo[:, 1] / 40, "--", color="green", label="Simo")
    ax.plot(
        force_u_Harsch[:, 0],
        force_u_Harsch[:, 1],
        "x",
        color="red",
        label="u_x: Harsch2021",
    )
    ax.plot(
        force_u_Harsch[:, 0],
        force_u_Harsch[:, 2],
        "x",
        color="green",
        label="u_y: Harsch2021",
    )
    ax.plot(
        force_u_Harsch[:, 0],
        force_u_Harsch[:, 3],
        "x",
        color="blue",
        label="u_z: Harsch2021",
    )
    ax.legend(loc="lower left")
    ax.set_title("displacement of tip point")
    ax.set_xlabel("tip force")
    ax.set_ylabel("displacement")
    ax.grid()

    # matplotlib visualization
    # construct animation of beam
    _, ax, _ = animate_beam(
        t,
        q,
        rod,
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=rod[0].nelement + 1,
        repeat=True,
    )

    # plot animation
    ax.azim = 30
    ax.elev = 25

    plt.show()


if __name__ == "__main__":
    beam_patches_3d(Rod=make_CosseratRod_Quat(mixed=True))
