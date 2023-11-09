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
from cardillo.constraints import RigidConnection, Cylindrical, Prismatic
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
Example proposed by Goto, Y. et al,
"Elastic buckling phenomenon applicable to deployable rings" ,
International Journal of Solids and Structures, 29(7):893– 909, 1992,
https://sci-hub.hkvisa.net/10.1016/0020-7683(92)90024-n, 1988
"""


def deployment_of_elastic_ring(
    load_type="moment", rod_hypothesis_penalty="shear_deformable", VTK_export=False
):
    Rod = CosseratRodPG_R12Mixed
    # Rod = CosseratRodPG_QuatMixed
    # Rod = CosseratRodPG_SE3Mixed

    nelements_Lagrangian = 20
    polynomial_degree = 1

    # number of elements
    if Rod is CosseratRodPG_SE3Mixed:
        nelements = nelements_Lagrangian * polynomial_degree
    else:
        nelements = nelements_Lagrangian

    # geometry of the rod
    width = 1 / 3.0
    height = 1

    # cross section
    line_density = 1
    cross_section = RectangularCrossSection(line_density, width, height)

    A = cross_section.area
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    atol = 1e-10

    EE = 2.1 * 1e7  # Young's modulus
    GG = EE / (2 * (1 + 0.3))  # shear modulus

    # material model
    if rod_hypothesis_penalty == "shear_deformable":
        Ei = np.array([EE * A, GG * A, GG * A])
    elif rod_hypothesis_penalty == "shear_rigid":
        Ei = np.array([EE * A, 1e10 * GG * A, 1e10 * GG * A])
    elif rod_hypothesis_penalty == "inextensilbe_shear_rigid":
        Ei = np.array([EE * A, GG * A, GG * A]) * 1e10

    Fi = np.array([GG * K_I_rho0[0, 0], EE * K_I_rho0[1, 1], EE * K_I_rho0[2, 2]])

    material_model = Simo1986(Ei, Fi)

    R = 20
    angle = 2 * pi / 2

    # definition of the parametric curve
    curve = lambda xi: np.array([R - R * np.cos(xi), R * np.sin(xi), 0])
    dcurve = lambda xi: np.array([R * np.sin(xi), R * np.cos(xi), 0])
    ddcurve = lambda xi: np.array([R * np.cos(xi), -R * np.sin(xi), 0])

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    q0 = Rod.deformed_configuration(
        nelements,
        curve,
        dcurve,
        ddcurve,
        angle,
        polynomial_degree=polynomial_degree,
        r_OP=r_OP0,
        A_IK=A_IK0,
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

    # generate the constraint on the beam
    # r_OP_2_end = np.zeros(3, dtype=float)
    # r_OP_2_end[0] = 2 * R
    # A_IK_disp = lambda t:  A_IK_basic(pi).z() @ A_IK_basic(t * pi).y()
    # A_IK_c = A_IK_basic(pi).z()
    # A_IK_disp = A_IK_c @ A_IK_disp

    # clamping_point = Frame(A_IK=A_IK0)
    # clamping_point_2 = Frame(A_IK=A_IK0)
    # clamping = Frame(A_IK=A_IK_displ)

    A_IK_disp = lambda t: A_IK_basic(2 * pi * t).x()
    displ = lambda t: np.array([-4 * R / 3 * t, 0, 0])

    displ_imposed = Frame(r_OP=displ, A_IK=A_IK_disp)
    clamping_left = RigidConnection(system.origin, cantilever, frame_ID2=(0,))
    # clamping_right = RigidConnection(displ_imposed, cantilever, 0, frame_ID2=(0.5,))
    clamping_right = Cylindrical(system.origin, cantilever, 0, frame_ID2=(1,))
    # clamping_right = Cylindrical(system.origin, cantilever, 0, frame_ID2=(0.5,))
    # clamping_left_2 = RigidConnection(system.origin, cantilever, frame_ID2=(1,))

    # assemble the system
    system.add(cantilever)
    # system.add(displ_imposed)
    # system.add(clamping_point)
    system.add(clamping_left)
    system.add(clamping_right)
    # system.add(clamping_left_2)
    # system.add(clamping_point_2)

    M = lambda t: EE * K_I_rho0[1, 1] / R * t * np.array([1, 0, 0])

    moment = Moment(M, cantilever, (1,))
    # moment = Moment(M, cantilever, (0.5, ))
    system.add(moment)

    system.assemble()

    # solver = Newton(
    #     system,
    #     n_load_steps=50,
    #     max_iter=30,
    #     atol=atol,
    # )
    solver = Riks(
        system,
        atol=1e-8,
        la_arc0=2e-2,
        la_arc_span=np.array([-0.1, 1]),
    )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

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

    x_tip_displacement = np.zeros(len(t))
    y_tip_displacement = np.zeros(len(t))
    z_tip_displacement = np.zeros(len(t))
    reference_point = pi / 2 / angle
    frame_ID = (reference_point,)

    qDOF_element_of_interest = cantilever.local_qDOF_P(frame_ID)
    r0 = cantilever.r_OP(0, q[0, qDOF_element_of_interest], frame_ID=frame_ID)

    # the plotted displacements depend on how the structure is modelled in the 3D space. In this case we have that the x
    for i, ti in enumerate(t):
        ri = cantilever.r_OP(ti, q[i, qDOF_element_of_interest], frame_ID=frame_ID)
        x_tip_displacement[i] = ri[0] - r0[0]
        y_tip_displacement[i] = ri[1] - r0[1]
        z_tip_displacement[i] = ri[2] - r0[2]

    fig2, ax = plt.subplots()

    # ax.plot(t, x_tip_displacement, '-', color='blue', label='X Tip Displacement', marker='o')
    # ax.plot(t, y_tip_displacement, '-', color='red', label='Y Tip Displacement', marker='s')
    ax.plot(
        x_tip_displacement, t, "-", color="blue", label="X Tip Displacement", marker="o"
    )
    ax.plot(
        y_tip_displacement, t, "-", color="red", label="Y Tip Displacement", marker="s"
    )
    ax.plot(
        z_tip_displacement,
        t,
        "-",
        color="green",
        label="Y Tip Displacement",
        marker="^",
    )

    # Aggiungi una legenda
    ax.legend(loc="upper left")

    # Personalizza il titolo e le label degli assi se necessario
    ax.set_title("Displacements of the point B")
    ax.set_xlabel("u, v")
    ax.set_ylabel("Load Factor")
    # ax.set_ylim(-1.5, 3.0)

    plt.show()


if __name__ == "__main__":
    deployment_of_elastic_ring(load_type="moment", VTK_export=False)
