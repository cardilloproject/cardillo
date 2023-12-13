from cardillo.rods import (
    RectangularCrossSection,
    CircularCrossSection,
    Simo1986,
    animate_beam,
)

from cardillo.rods.cosseratRod import (
    make_CosseratRod_Quat,
    make_CosseratRod_SE3,
    make_CosseratRod_R12,
)

from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection, Cylindrical, Prismatic, Revolute
from cardillo.solver import Newton, Riks, SolverOptions
from cardillo.forces import Force, Moment, K_Force, K_Moment

from cardillo.math import e1, e2, e3, A_IK_basic, norm, cross3, Log_SO3_quat

from cardillo.visualization import Export

from cardillo import System

from math import pi
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path


"""
Instability of Clamped-Hinged Circular Arches Subjected to a Point Load

D.A. DaDeppo, R. Schmidt, Instability of clamped-hinged
circular arches subjected to a point load, J. Appl. Mech.
Trans. ASME 42 (1975) 894-896
https://doi.org/10.1115/1.3423734

"A simple extrapolated predictor for overcoming the starting and tracking issues in the  arc-length method for nonlinear structural mechanics", Engineering Structures, Volume 234, 1 May 2021, 111755
https://doi.org/10.1016/j.engstruct.2020.111755

"""

def circular_arch(
    Rod,
    i_alpha=14,
    nelements=10,
    polynomial_degree=2,
    n_load_steps=10,
    reduced_integration=True,
    VTK_export=False,
):
    # cross section properties for visualization purposes
    width = 1.0
    cross_section = RectangularCrossSection(width, width)

    # material properties
    A = 2.29
    I = 1.0
    EE = 1e6  # Young's modulus
    GG = EE / (2 * (1 + 0))  # shear modulus
    EA = EE * A
    GA = GG * A
    GIp = GG * I
    EI = EE * I
    tip_force = 1000

    # material model
    Ei = np.array([EA, GA, GA])
    Fi = np.array([GIp, EI, EI])

    material_model = Simo1986(Ei, Fi)

    # curverd initial configuration
    R = 100

    path = Path(__file__)

    data_DaDeppo = np.loadtxt(
        Path(path.parent, "_data_circular_arch_215", "critical_loads_DaDeppo1975.csv"),
        delimiter=",",
        skiprows=1,
    )

    alpha = data_DaDeppo[i_alpha, 0]
    print(f"circular arch with angle:{2*alpha}")
    critical_load = data_DaDeppo[i_alpha, 1] / 10
    u_x_DaDeppo = data_DaDeppo[i_alpha, 2] * R
    u_y_DaDeppo = data_DaDeppo[i_alpha, 3] * R

    angle = 2 * alpha * pi / 180
    start_angle = (90 - alpha) * pi / 180

    # definition of the arch curve
    curve = lambda xi: np.array(
        [R * np.cos(xi + start_angle), R * np.sin(xi + start_angle), 0]
    )
    dcurve = lambda xi: np.array(
        [-R * np.sin(xi + start_angle), R * np.cos(xi + start_angle), 0]
    )
    ddcurve = lambda xi: np.array(
        [-R * np.cos(xi + start_angle), -R * np.sin(xi + start_angle), 0]
    )

    # final point
    r_OP02 = curve(angle)

    q0 = Rod.deformed_configuration(
        nelements,
        curve,
        dcurve,
        ddcurve,
        angle,
        polynomial_degree=polynomial_degree,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
    )

    arch = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        reduced_integration=reduced_integration,
    )

    # create the system
    system = System()

    # clamped point
    clamping_right = RigidConnection(system.origin, arch, frame_ID2=(0,))

    # hinged point
    frame_left = Frame(r_OP02)
    hinge_left = Revolute(frame_left, arch, 2, frame_ID2=(1,))

    system.add(arch)
    system.add(clamping_right)
    system.add(frame_left)
    system.add(hinge_left)

    # concentrated force
    F = lambda t: -tip_force * t * e2
    force = Force(F, arch, frame_ID=(0.5,))
    system.add(force)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    atol = 1e-10
    # # solver = Newton(
    # #     system,
    # #     n_load_steps=10,
    # #     options=SolverOptions(newton_atol=atol),
    # # )
    solver = Riks(
        system,
        la_arc0=1e-2,
        iter_goal=4,
        la_arc_span=np.array([-1.1*critical_load, 1.1*critical_load]),
        options=SolverOptions(newton_atol=atol),
    )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    

    u_x = np.zeros(nt)
    u_y = np.zeros(nt)

    q_arch = sol.q[:, arch.qDOF]
    qe_arch = q_arch[:, arch.local_qDOF_P((0.5,))]

    r_OP0 = arch.r_OP(0, qe_arch[0], frame_ID=(0.5,))
    print(f"r_OP0:{r_OP0}")
    for i in range(nt):
        r_OPi = arch.r_OP(0, qe_arch[i], frame_ID=(0.5,))
        u_x[i], u_y[i], _ = -(r_OPi - r_OP0)

    fig5, ax = plt.subplots(1, 2)

    ax[0].plot(u_x, t, "-", color="blue", label="Cosserat rod")
    ax[0].plot(u_x_DaDeppo, critical_load, marker="+", color="red", label="critical load DaDeppo 1975")
    ax[0].legend(loc="upper left")
    ax[0].set_title("horizontal displacements of tip point")
    ax[0].set_xlabel("u_x")
    ax[0].set_ylabel("load factor")
    
    ax[1].plot(u_y, t, "-", color="blue", label="Cosserat rod")
    ax[1].plot(u_y_DaDeppo, critical_load, "+", color="red", label="critical load DaDeppo 1975")
    ax[1].legend(loc="upper left")
    ax[1].set_title("horizontal displacements of tip point")
    ax[1].set_xlabel("u_x")
    ax[1].set_ylabel("load factor")

    if i_alpha == 14:
        force_u_x_Kadapa = np.loadtxt(
            Path(path.parent, "_data_circular_arch_215", "force_u_x_Kadapa2021.csv"),
            delimiter=",",
            skiprows=1,
        )
        force_u_y_Kadapa = np.loadtxt(
            Path(path.parent, "_data_circular_arch_215", "force_u_y_Kadapa2021.csv"),
            delimiter=",",
            skiprows=1,
        )
        ax[0].plot(force_u_x_Kadapa[:, 0] * 100, force_u_x_Kadapa[:, 1] / 10, "x", color="green")
        ax[1].plot(force_u_y_Kadapa[:, 0] * 100, force_u_y_Kadapa[:, 1] / 10, "x", color="green")
  
    # VTK export
    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 30, sol)
        e.export_contr(
            arch,
            level="centerline + directors",
            num=3 * nelements,
            file_name="cantilever_curve",
        )
        e.export_contr(
            arch,
            continuity="C0",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
            file_name="cantilever_volume",
        )

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        [arch],
        scale=2 * R,
        scale_di=0.05,
        show=False,
        n_frames=arch.nelement + 1,
        repeat=True,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-2 * R, 2 * R, num=2)
    Y_z = np.linspace(-2 * R, 2 * R, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)

    # camera settings for 3D plot
    ax1.azim = -90
    ax1.elev = 72

    plt.show()

if __name__ == "__main__":
    circular_arch(Rod=make_CosseratRod_Quat(mixed=True), i_alpha=14)
