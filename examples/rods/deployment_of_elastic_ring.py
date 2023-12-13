from cardillo.rods import (
    RectangularCrossSection,
    Simo1986,
    animate_beam,
)


from cardillo.rods.cosseratRod import (
    make_CosseratRod_SE3,
    make_CosseratRod_Quat,
    make_CosseratRod_R12
)

from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection, Cylindrical, Prismatic
from Cardillo_extensions.solver.statics import Newton, Riks, SolverOptions
from cardillo.forces import Moment

import Cardillo_extensions.discrete.shapes 

from cardillo.math import e1, e2, e3, A_IK_basic, norm, cross3, Log_SO3_quat

from cardillo.visualization import Export

from cardillo import System

from math import pi
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path
from shutil import rmtree


""" Elastic buckling phenomenon applicable to deployable rings:

Goto, Y. , Watanabe, Y., Kasugai, T. and Obata, M.: "Elastic buckling phenomenon applicable to deployable rings" ,
International Journal of Solids and Structures, 29(7):893 – 909, 1992,
https://doi.org/10.1016/0020-7683(92)90024-N
"""


def deployment_of_elastic_ring(
    Rod,
    nelements=10,
    polynomial_degree=2,
    n_load_steps=40,
    VTK_export=False,
    reduced_integration=True,
    displacement_controlled=False,
):
    
    # cross section properties
    width = 1.0 / 3
    height = 1.0
    cross_section = RectangularCrossSection(width, height)
    A = cross_section.area
    I1, I2, I3 = np.diag(cross_section.second_moment)

    # material model
    EE = 2.1 * 1.e7             # Young's modulus
    GG = EE / (2 * (1 + 0.3))   # shear modulus
    Ei = np.array([EE * A, GG * A, GG * A])
    Fi = np.array([GG * 9.753 * 1e-3, EE * I2, EE * I3])

    material_model = Simo1986(Ei, Fi)

    radius = 20
    xi_end = 2 * pi

    # definition of circle
    curve = lambda xi: np.array([radius - radius * np.cos(xi), radius * np.sin(xi), 0])
    dcurve = lambda xi: np.array([radius * np.sin(xi), radius * np.cos(xi), 0])
    ddcurve = lambda xi: np.array([radius * np.cos(xi), -radius * np.sin(xi), 0])

    q0 = Rod.deformed_configuration(
        nelements,
        curve,
        dcurve,
        ddcurve,
        xi_end,
        polynomial_degree=polynomial_degree,
    )

    ring = Rod(
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

    system.add(ring)

    closing_condition = RigidConnection(ring, ring, frame_ID1=(0,), frame_ID2=(1,))
    clamping_left = RigidConnection(ring, system.origin, frame_ID1=(0,))
    system.add(closing_condition)
    system.add(clamping_left)

    if displacement_controlled:
        A_IK_rotating = lambda t: A_IK_basic(4 * pi * t).x()
        rotating_frame = Frame(A_IK=A_IK_rotating)
        guidance_right = Prismatic(ring, rotating_frame, 0, frame_ID1=(0.5,))
        
        system.add(rotating_frame)
        system.add(guidance_right)

        system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

        solver = Newton(
        system,
        n_load_steps=100,
        )

    else:
        # guidance_right = Cylindrical(ring, system.origin, 0, frame_ID1=(0.5,))
        
        # system.add(guidance_right)

        M = lambda t: EE * I2 / radius * t * np.array([1, 0, 0])

        moment = Moment(M, ring, (0.5,))
        # moment = Moment(M, cantilever, (0.5, ))
        system.add(moment)

        system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

        solver = Riks(
        system,
        n_load_step=100,
        iter_goal=3,
        la_arc0=2e-2,
        la_arc_span=np.array([-1, 1]),
        )



    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]
    la_g = sol.la_g[:nt]

    M = la_g[:, clamping_left.la_gDOF[4]]


    # VTK export
    # if VTK_export:
    #     path = Path(__file__)
    #     e = Export(path.parent, path.stem, True, 30, sol)
    #     e.export_contr(
    #         cantilever,
    #         level="centerline + directors",
    #         num=3 * nelements,
    #         file_name="cantilever_curve",
    #     )
    #     e.export_contr(
    #         cantilever,
    #         continuity="C0",
    #         level="volume",
    #         n_segments=nelements,
    #         num=3 * nelements,
    #         file_name="cantilever_volume",
    #     )

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        [ring],
        scale=2 * radius,
        scale_di=0.05,
        show=False,
        n_frames=ring.nelement + 1,
        repeat=True,
    )

    x_B_displacement = np.zeros(len(t))
    y_B_displacement = np.zeros(len(t))
    z_B_displacement = np.zeros(len(t))

    reference_point = pi / 2 / xi_end
    frame_ID = (reference_point,)

    qDOF_element_of_interest = ring.local_qDOF_P(frame_ID)
    r0 = ring.r_OP(0, q[0, qDOF_element_of_interest], frame_ID=frame_ID)

    # the plotted displacements depend on how the structure is modelled in the 3D space. In this case we have that the x
    for i, ti in enumerate(t):
        ri = ring.r_OP(ti, q[i, qDOF_element_of_interest], frame_ID=frame_ID)
        x_B_displacement[i] = ri[0] - r0[0]
        y_B_displacement[i] = ri[1] - r0[1]
        z_B_displacement[i] = ri[2] - r0[2]


    displacement_data = np.column_stack((t, x_B_displacement, y_B_displacement, z_B_displacement))

    fig2, ax = plt.subplots()

    if displacement_controlled:
        M = la_g[:, clamping_left.la_gDOF[4]]
    else:
        M = t

    ax.plot(
        x_B_displacement, M, "-", color="blue", label="B X Displacement", marker="o"
    )
    ax.plot(
        y_B_displacement, M, "-", color="red", label="B Y Displacement", marker="s"
    )
    ax.plot(
        z_B_displacement, M, "-", color="green", label="B Z Displacement", marker="^",
    )



    

    # Aggiungi una legenda
    ax.legend(loc="upper left")

    # Personalizza il titolo e le label degli assi se necessario
    ax.set_title("Displacements of the point B")
    ax.set_xlabel("u, v, w")
    ax.set_ylabel("Load Factor")
    # ax.set_ylim(-1.5, 3.0)

    plt.show()

    path = Path(__file__)
    path = Path(path.parent / path.stem)
    # if path.exists():
    #     rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    header = "t, x_B_displacement, y_B_displacement, z_B_displacement"

    np.savetxt(
        path
        / f"B_point_displacements.txt",
        displacement_data,
        delimiter=", ",
        header=header,
        comments="",
    )


if __name__ == "__main__":
    deployment_of_elastic_ring(Rod=make_CosseratRod_Quat(mixed=True))


# to obtain the curves for the displacement of point B need run the file "curve_B_deployment"