from cardillo.beams import (
    CosseratRodPG_SE3,
    CosseratRodPG_Quat,
    CosseratRodPG_R12,
    RectangularCrossSection,
    CircularCrossSection,
    Simo1986,
    animate_beam,
)


from cardillo.beams.cosseratRodPGMixed import (
    CosseratRodPG_R12Mixed,
    CosseratRodPG_QuatMixed,
    CosseratRodPG_SE3Mixed,
)

from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection, Cylindrical, Prismatic, Revolute
from cardillo.solver import Newton, Riks
from cardillo.forces import Force, Moment, K_Force, K_Moment

from cardillo.math import e1, e2, e3, A_IK_basic, norm, cross3, Log_SO3_quat

from cardillo.visualization import Export

from cardillo import System

from math import pi
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path


"""
Example proposed by Kadapa, C. in
"A simple extrapolated predictor for overcoming the starting and tracking issues in the
 arc-length method for nonlinear structural mechanics",
Engineering Structures, Volume 234, 1 May 2021, 111755

https://doi.org/10.1016/j.engstruct.2020.111755
"""


def _215_ring(
    load_type="force", rod_hypothesis_penalty="shear_deformable", VTK_export=False
):
    Rod = CosseratRodPG_R12Mixed
    # Rod = CosseratRodPG_QuatMixed
    # Rod = CosseratRodPG_SE3Mixed

    nelements = 20
    polynomial_degree = 1

    # geometry of the rod
    width = 1.0
    line_density = 1
    cross_section = RectangularCrossSection(line_density, width, width)

    A = 2.29
    I = 1.0
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    atol = 1e-7

    EE = 1e6  # Young's modulus
    GG = EE / (2 * (1 + 0))  # shear modulus

    # material model
    Ei = np.array([EE * A, GG * A, GG * A])
    Fi = np.array([GG * I, EE * I, EE * I])

    material_model = Simo1986(Ei, Fi)

    R = 100
    angle = 215 * pi / 180
    start_angle = 17.5 * pi / 180

    # definition of the parametric curve
    curve = lambda xi: np.array(
        [R - R * np.cos(xi - start_angle), R * np.sin(xi - start_angle), 0]
    )
    dcurve = lambda xi: np.array(
        [R * np.sin(xi - start_angle), R * np.cos(xi - start_angle), 0]
    )
    ddcurve = lambda xi: np.array(
        [R * np.cos(xi - start_angle), -R * np.sin(xi - start_angle), 0]
    )

    # starting point and orientation of initial point, initial length
    r_OP01 = curve(0)
    A_IK01 = np.eye(3, dtype=float)
    A_IK01[:, 0] = dcurve(0) / norm(dcurve(0))
    A_IK01[:, 1] = ddcurve(0) / norm(ddcurve(0))
    A_IK01[:, 2] = cross3(A_IK01[:, 0], A_IK01[:, 1])

    # final point
    r_OP02 = curve(angle)
    A_IK02 = np.eye(3, dtype=float)
    A_IK02[:, 0] = dcurve(angle) / norm(dcurve(angle))
    A_IK02[:, 1] = ddcurve(angle) / norm(ddcurve(angle))
    A_IK02[:, 2] = cross3(A_IK02[:, 0], A_IK02[:, 1])

    q0 = Rod.deformed_configuration(
        nelements,
        curve,
        dcurve,
        ddcurve,
        angle,
        polynomial_degree=polynomial_degree,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
        mixed=True,
    )

    cantilever = Rod(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        reduced_integration=False,
        mixed=True,
    )

    # create the system
    system = System()

    # clamped point
    frame_left = Frame(r_OP01, A_IK01)
    clamping_left = RigidConnection(frame_left, cantilever, frame_ID2=(0,))

    # hinged point
    frame_right = Frame(r_OP02, A_IK02)
    hinge_right = Revolute(frame_right, cantilever, 2, frame_ID2=(1,))

    system.add(cantilever)
    system.add(frame_left)
    system.add(clamping_left)
    system.add(frame_right)
    system.add(hinge_right)

    # concentrated force
    F = lambda t: -5000 * t * e2
    force = Force(F, cantilever, frame_ID=(0.5,))
    system.add(force)

    system.assemble()

    # solver = Newton(
    #     system,
    #     n_load_steps=10,
    #     max_iter=30,
    #     atol=atol,
    # )
    solver = Riks(system, atol=atol, la_arc0=1e-2, la_arc_span=np.array([-0.1, 0.25]))

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    x_tip_displacement = np.zeros(len(t))
    y_tip_displacement = np.zeros(len(t))

    element = nelements // 2

    for i in range(len(t)):
        x_tip_displacement[i] = abs(
            q[i, cantilever.elDOF_r[element - 1][polynomial_degree]]
            - q0[cantilever.elDOF_r[element - 1][polynomial_degree]]
        )
        y_tip_displacement[i] = abs(
            q[i, cantilever.elDOF_r[element - 1][polynomial_degree * 2 + 1]]
            - q0[cantilever.elDOF_r[element - 1][polynomial_degree * 2 + 1]]
        )

    fig5, ax = plt.subplots()

    # ax.plot(t, x_tip_displacement, '-', color='blue', label='X Tip Displacement', marker='o')
    # ax.plot(t, y_tip_displacement, '-', color='red', label='Y Tip Displacement', marker='s')
    ax.plot(
        x_tip_displacement, t, "-", color="blue", label="X Tip Displacement", marker="o"
    )
    ax.plot(
        y_tip_displacement, t, "-", color="red", label="Y Tip Displacement", marker="s"
    )

    # Aggiungi una legenda
    ax.legend(loc="upper left")

    # Personalizza il titolo e le label degli assi se necessario
    ax.set_title("Displacements of the point B")
    ax.set_xlabel("u, v")
    ax.set_ylabel("Load Factor")

    # x_values = np.linspace(0, f_max, n_load_step)
    # for x_value in x_values:  # Assumendo che tu voglia una linea per ogni punto di dati
    #     ax.axvline(x=x_value, color='gray', linestyle='--')

    # y_values = np.linspace(-30, 60, 11)
    # for y_value in y_values:
    #     ax.axhline(y=y_value, color='gray', linestyle='--')

    # VTK export
    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 30, sol)
        e.export_contr(
            cantilever,
            level="centerline + directors",
            num=3 * nelements,
            file_name="cantilever_curve",
        )
        e.export_contr(
            cantilever,
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
        [cantilever],
        scale=2 * R,
        scale_di=0.05,
        show=False,
        n_frames=cantilever.nelement + 1,
        repeat=True,
    )

    plt.show()


if __name__ == "__main__":
    _215_ring(load_type="force", VTK_export=False)
