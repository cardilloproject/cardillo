from cardillo.rods import (
    RectangularCrossSection,
    Simo1986,
    animate_beam,
)


from cardillo.rods.cosseratRod import (
    make_CosseratRod_SE3,
    make_CosseratRod_Quat,
    make_CosseratRod_R12,
)

from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection, Cylindrical, Prismatic
from cardillo.solver import Newton, Riks, SolverOptions
from cardillo.forces import Moment


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
    n_load_steps=100,
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
    EE = 2.1 * 1.0e7  # Young's modulus
    GG = EE / (2 * (1 + 0.3))  # shear modulus
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
        guidance_right = Prismatic(ring, rotating_frame, 1, frame_ID1=(0.5,))

        system.add(rotating_frame)
        system.add(guidance_right)

        system.assemble(
            options=SolverOptions(compute_consistent_initial_conditions=False)
        )

        solver = Newton(
            system,
            n_load_steps=100,
        )

    else:
        M_max = EE * I2 / radius
        M = lambda t: M_max * t * np.array([1, 0, 0])
        moment = Moment(M, ring, (0.5,))
        system.add(moment)

        system.assemble(
            options=SolverOptions(compute_consistent_initial_conditions=False)
        )

        solver = Riks(
            system,
            iter_goal=3,
            la_arc0=2e-2,
            la_arc_span=np.array([-1, 1]),
            max_load_steps=100,
        )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]
    la_g = sol.la_g[:nt]

    if displacement_controlled:
        M = la_g[:, clamping_left.la_gDOF[5]]
    else:
        M = t * M_max

    # compute displacement of point B
    u = np.zeros((3, nt))
    frame_ID = (0.25,)
    qe = q[:, ring.local_qDOF_P(frame_ID)]
    r_OP0 = ring.r_OP(0, qe[0], frame_ID=frame_ID)

    for i in range(nt):
        r_OPi = ring.r_OP(0, qe[i], frame_ID=frame_ID)
        u[:, i] = r_OPi - r_OP0

    _, ax = plt.subplots()

    ax.plot(u[0], M, "-", color="blue", label="u_x")
    ax.plot(u[1], M, "-", color="red", label="u_y")
    ax.plot(u[2], M, "-", color="green", label="u_z")

    ax.grid()
    ax.legend(loc="upper left")
    ax.set_title("moment-displacement curve point B")
    ax.set_xlabel("displacement")
    ax.set_ylabel("moment")

    # VTK export
    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 30, sol)
        e.export_contr(
            ring,
            level="centerline + directors",
            num=3 * nelements,
            file_name="ring_curve",
        )
        e.export_contr(
            ring,
            continuity="C0",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
            file_name="ring_volume",
        )

    # matplotlib visualization
    # construct animation of beam
    _, ax, _ = animate_beam(
        t,
        q,
        [ring],
        scale=2 * radius,
        scale_di=0.05,
        show=False,
        n_frames=ring.nelement + 1,
        repeat=True,
    )

    plt.show()


if __name__ == "__main__":
    deployment_of_elastic_ring(
        Rod=make_CosseratRod_Quat(mixed=True), displacement_controlled=True
    )
