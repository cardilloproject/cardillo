from cardillo.rods import (
    RectangularCrossSection,
    CircularCrossSection,
    Simo1986,
    animate_beam,
)

from cardillo.rods.cosseratRod import (
    make_CosseratRod_SE3,
    make_CosseratRod_Quat,
    make_CosseratRod_R12,
)

from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection, Cylindrical, Revolute
from cardillo.solver import Newton, Riks, SolverOptions
from cardillo.forces import Force, K_Force, K_Moment

from cardillo.math import e1, e2, e3, A_IK_basic
from cardillo.visualization import Export

from cardillo import System

from math import pi
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

""" Lee frame: Buckling of a hinged right-angle frame under follower point
load

Harsch, J. and Eugster, S. R., "Finite element analysis of planar nonlinear classical beam theories", 2020. 
https://doi.org/10.1007/978-3-030-50460-1_10, Section 6.5

"""


def lee_frame(
    Rod,
    nelements=5,
    polynomial_degree=3,
    n_load_steps=10,
    reduced_integration=True,
    VTK_export=False,
):
    # cross section properties for visualization purposes
    width = 1.0
    cross_section = RectangularCrossSection(width, width)

    # geometry of the rod
    length = 120

    # material properties
    EE = 7.2e6
    nu = 0.3
    GG = EE / (2 + 2 * nu)
    A = 6.0
    I = 2.0

    # material model
    Ei = np.array([EE * A, GG * A, GG * A])
    Fi = np.array([2 * GG * I, EE * I, EE * I])

    material_model = Simo1986(Ei, Fi)

    # starting position and orientation of left point
    r_OP01 = np.zeros(3, dtype=float)
    A_IK01 = A_IK_basic(np.pi / 2).z()
    r_OP02 = np.array([0, length, 0])
    A_IK02 = np.eye(3)

    # construct system
    system = System()

    q01 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        r_OP=r_OP01,
        A_IK=A_IK01,
    )
    cantilever1 = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q01,
        q0=q01,
        polynomial_degree=polynomial_degree,
        reduced_integration=reduced_integration,
    )

    q02 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        r_OP=r_OP02,
        A_IK=A_IK02,
    )
    cantilever2 = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q02,
        q0=q02,
        polynomial_degree=polynomial_degree,
        reduced_integration=reduced_integration,
    )

    # rigid connection between rods
    clamping_c1_c2 = RigidConnection(
        cantilever1, cantilever2, frame_ID1=(1,), frame_ID2=(0,)
    )

    # hinge constraints
    hinge_c1 = Revolute(cantilever1, system.origin, 2, frame_ID1=(0,))
    hinge_c2 = Revolute(cantilever2, system.origin, 2, frame_ID1=(1,))

    # concentrated force
    f_max = 40.0e3
    F = lambda t: -f_max * t * e2
    force = K_Force(F, cantilever2, frame_ID=(0.2,))

    # assemble the system
    system.add(cantilever1)
    system.add(cantilever2)
    system.add(clamping_c1_c2)
    system.add(hinge_c1)
    system.add(hinge_c2)
    system.add(force)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    atol = 1e-6
    # add Newton solver
    # solver = Newton(
    #     system,
    #     n_load_steps=1,
    #     options=SolverOptions(newton_atol=atol),
    # )

    solver = Riks(
        system,
        iter_goal=3,
        la_arc0=1e-3,
        la_arc_span=np.array([-0.4, 1]),
        options=SolverOptions(newton_atol=atol),
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
        e.export_contr(
            cantilever1,
            level="centerline + directors",
            num=3 * nelements,
            file_name="cantilever1_curve",
        )
        e.export_contr(
            cantilever1,
            continuity="C0",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
            file_name="cantilever1_volume",
        )
        e.export_contr(
            cantilever2,
            level="centerline + directors",
            num=3 * nelements,
            file_name="cantilever2_curve",
        )
        e.export_contr(
            cantilever2,
            continuity="C0",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
            file_name="cantilever2_volume",
        )

    u_x = np.zeros(nt)
    u_y = np.zeros(nt)

    q_c2 = sol.q[:, cantilever2.qDOF]
    qe_c2 = q_c2[:, cantilever2.local_qDOF_P((0.2,))]

    r_OP0 = cantilever2.r_OP(0, qe_c2[0], frame_ID=(0.2,))
    print(f"r_OP0:{r_OP0}")
    for i in range(nt):
        r_OPi = cantilever2.r_OP(0, qe_c2[i], frame_ID=(0.2,))
        u = r_OPi - r_OP0
        u_x[i] = u[0]
        u_y[i] = -u[1]

    path = Path(__file__)

    force_u_x_Simo = np.loadtxt(
        Path(path.parent, "_data_Lee_frame", "force_u_x_Simo1986.csv"),
        delimiter=",",
        skiprows=1,
    )

    force_u_y_Simo = np.loadtxt(
        Path(path.parent, "_data_Lee_frame", "force_u_y_Simo2986.csv"),
        delimiter=",",
        skiprows=1,
    )

    force_u_Harsch = np.loadtxt(
        Path(path.parent, "_data_Lee_frame", "force_u_Harsch2020.csv"),
        delimiter=",",
        skiprows=1,
    )

    _, ax = plt.subplots(1, 2)

    ax[0].plot(u_x, t, "-", color="red", marker="x", label="Cosserat rod")
    ax[0].plot(
        force_u_x_Simo[:, 0],
        force_u_x_Simo[:, 1] / 40,
        "--",
        color="green",
        label="Simo",
    )
    ax[0].plot(
        force_u_Harsch[:, 2], force_u_Harsch[:, 1], "--", color="blue", label="Harsch"
    )
    ax[0].legend(loc="upper left")
    ax[0].set_title("horizontal displacements of tip point")
    ax[0].set_xlabel("u_x")
    ax[0].set_ylabel("load factor")
    ax[0].grid()

    ax[1].plot(u_y, t, "-", color="red", marker="x", label="Cosserat rod")
    ax[1].plot(
        force_u_y_Simo[:, 0],
        force_u_y_Simo[:, 1] / 40,
        "--",
        color="green",
        label="Simo",
    )
    ax[1].plot(
        force_u_Harsch[:, 3], force_u_Harsch[:, 1], "--", color="blue", label="Harsch"
    )
    ax[1].legend(loc="upper left")
    ax[1].set_title("vertical displacements of tip point")
    ax[1].set_xlabel("u_y")
    ax[1].set_ylabel("load factor")
    ax[1].grid()

    # matplotlib visualization
    # construct animation of rod
    _, ax, _ = animate_beam(
        t,
        q,
        [cantilever1, cantilever2],
        scale=1.1 * length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever1.nelement + 1,
        repeat=True,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    Y_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax.plot_surface(X_z, Y_z, Z_z, alpha=0.2)

    # camera settings for 3D plot
    ax.azim = -90
    ax.elev = 72

    plt.show()


if __name__ == "__main__":
    lee_frame(Rod=make_CosseratRod_Quat(mixed=True))
