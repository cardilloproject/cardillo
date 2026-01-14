import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.constraints import Prismatic, RigidConnection
from cardillo.discrete import Frame, RigidBody
from cardillo.forces import Moment
from cardillo.math import A_IB_basic
from cardillo.rods import RectangularCrossSection, Simo1986, animate_beam
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.solver import Newton, SolverOptions

""" Elastic buckling phenomenon applicable to deployable rings:

Goto, Y. , Watanabe, Y., Kasugai, T. and Obata, M.: "Elastic buckling phenomenon applicable to deployable rings" ,
International Journal of Solids and Structures, 29(7):893 – 909, 1992,
https://doi.org/10.1016/0020-7683(92)90024-N

"""


def deployment_of_elastic_ring(
    Rod,
    constitutive_law,
    *,
    nelements: int = 10,
    #
    n_load_steps: int = 100,
    #
    name: str = "simulation",
    VTK_export: bool = False,
    save_B_displacement: bool = False,
):
    save_name = f'{name.replace(" ", "_")}_nel{nelements}'
    ############
    # parameters
    ############
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

    material_model = constitutive_law(Ei, Fi)

    radius = 20

    # create function of circle
    r_OP_circle = lambda alpha: radius * np.array([1 - np.cos(alpha), np.sin(alpha), 0])
    A_IB_circle = lambda alpha: A_IB_basic(np.pi / 2 - alpha).z

    # define angle
    angle = 2 * np.pi
    r_OP0 = lambda xi: r_OP_circle(xi * angle)
    A_IB0 = lambda xi: A_IB_circle(xi * angle)

    ######
    # ring
    ######
    q0 = Rod.pose_configuration(nelements, r_OP0, A_IB0)
    ring = Rod(cross_section, material_model, nelements, Q=q0)

    # create the system
    system = System()

    system.add(ring)

    closing_condition = RigidConnection(
        ring, ring, xi1=0, xi2=1, name="closing_condition"
    )
    clamping_left = RigidConnection(ring, system.origin, xi1=0, name="origin_clamping")
    system.add(closing_condition)
    system.add(clamping_left)

    # add rigid body at points B and C
    xi_B = 0.25
    q0_B = RigidBody.pose2q(r_OP0(xi_B), A_IB0(xi_B))
    body_B = RigidBody(0.0, np.zeros([3, 3], dtype=float), q0=q0_B, name="B")
    constraint_B = RigidConnection(ring, body_B, xi1=xi_B, name="PointB")
    system.add(body_B, constraint_B)

    xi_C = 0.5
    q0_C = RigidBody.pose2q(r_OP0(xi_C), A_IB0(xi_C))
    body_C = RigidBody(0.0, np.zeros([3, 3], dtype=float), q0=q0_C, name="C")
    constraint_C = RigidConnection(ring, body_C, xi1=xi_C, name="PointC")
    system.add(body_C, constraint_C)

    # rotating frame
    A_IB_rotating = lambda t: A_IB_basic(4 * np.pi * t).x
    rotating_frame = Frame(A_IB=A_IB_rotating)
    guidance_right = Prismatic(ring, rotating_frame, 1, xi1=0.5)

    system.add(rotating_frame)
    system.add(guidance_right)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # add Newton solver
    atol = 1e-6
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_atol=atol),  # rtol=0
    )
    sol = solver.solve()

    # read solution
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]
    la_g = sol.la_g[:nt]

    #################
    # post-processing
    #################

    # vtk-export
    dir_name = Path(__file__).parent
    if VTK_export:
        system.export(dir_name, f"vtk/{save_name}", sol, fps=n_load_steps)

    M = la_g[:, clamping_left.la_gDOF[4]]

    # compute displacement of point B
    r_OP0_B = body_B.r_OP(0, q0_B)
    u = np.array([body_B.r_OP(0, qi[body_B.qDOF]) - r_OP0_B for qi in q]).T

    # plot displacement of point B
    _, axes = plt.subplots(2, 2)

    axes[0, 0].plot(4 * np.pi * t, M, "-", color="black", label="theta")
    axes[0, 1].plot(u[0], M, "-", color="red", label="u_x")
    axes[1, 0].plot(u[1], M, "-", color="green", label="u_y")
    axes[1, 1].plot(u[2], M, "-", color="blue", label="u_z")

    axes[0, 0].set_ylabel("moment")
    axes[1, 0].set_ylabel("moment")
    for ax in np.array(axes).flatten():
        ax.grid()
        ax.legend(loc="upper left")
        ax.set_xlabel("displacement")

    axes[0, 0].set_xlabel("angle")

    # saving displacement of point B
    if save_B_displacement:
        delta_tip_header = "time, load_moment, delta_x, delta_y, delta_z"
        path_tip = Path(dir_name, "csv", "B_displacement")
        path_tip.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            path_tip / f"{save_name}.csv",
            np.vstack((t, M, u)).T,
            delimiter=", ",
            header=delta_tip_header,
            comments="",
        )

    ##########################
    # matplotlib visualization
    ##########################
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
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        mixed=True,
        polynomial_degree=2,
        reduced_integration=True,
    )
    deployment_of_elastic_ring(
        Rod,
        Simo1986,
        nelements=20,
        n_load_steps=120,
        name="Deployment of an elastic ring",
    )
