from cardillo.rods import (
    RectangularCrossSection,
    CrossSectionInertias,
    Simo1986,
    animate_beam,
)

from cardillo.rods.cosseratRod import (
    make_CosseratRod_SE3,
    make_CosseratRod_R12,
    make_CosseratRod_Quat,
)

from cardillo.discrete import Cuboid, RigidBody

from cardillo.contacts import Sphere2Plane

from cardillo.rods.force_line_distributed import Force_line_distributed

from cardillo.constraints import RigidConnection
from cardillo.solver import Newton, BackwardEuler, SolverOptions
from cardillo.forces import Force, K_Moment, Moment, K_Force

from cardillo.math import e1, e2, e3, smoothstep0

from cardillo.visualization import Export

from cardillo import System

import numpy as np
from math import pi

import matplotlib.pyplot as plt
from pathlib import Path

""" Rhex from youtube: https://www.youtube.com/watch?v=ntlI-pDUxPE """


def Rhex(
    Rod,
    nelements=2,
    polynomial_degree=2,
    n_load_steps=10,
    VTK_export=False,
    reduced_integration=True,
):
    # geometry of the rod
    length = 2 * np.pi

    # cross section properties for visualization purposes
    slenderness = 1.0e2
    width = length / slenderness
    density = 8  # [kg / m^3]
    # cross_section = RectangularCrossSection(density, width, width)
    cross_section = RectangularCrossSection(width, 3 * width)
    cross_section_inertias = CrossSectionInertias(density, cross_section)

    A = cross_section.area
    Ip, Iy, Iz = np.diagonal(cross_section.second_moment)

    E = 210.0e6
    G = 80.0e6
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * Iy, E * Iz])

    material_model = Simo1986(Ei, Fi)

    # construct system
    system = System()

    R = 2
    length = 4 * R
    dimensions = np.array([4*R, 2*R, 1])
    box = Cuboid(RigidBody)(dimensions=dimensions, density=0.1, q0 = np.array([0, 0, 2.5 * R, 1, 0, 0, 0]), u0=np.zeros(6))
    system.add(box)
    box_grav = Force(-box.mass * 9.81 * e3, box)
    system.add(box_grav)

    start_angle = np.pi / 2
    # definition of the arch curve
    curve = lambda xi: np.array(
        [R * np.cos(xi + start_angle), 0, R * np.sin(xi + start_angle)]
    )
    dcurve = lambda xi: np.array(
        [-R * np.sin(xi + start_angle), 0, R * np.cos(xi + start_angle)]
    )
    ddcurve = lambda xi: np.array(
        [-R * np.cos(xi + start_angle), 0, -R * np.sin(xi + start_angle)]
    )


    F_grav = lambda t, xi: - density * A * 9.81 * e3

    l = length
    r_OP0 = np.array([[0.4 * l, -R, 1.5 * R],
                      [0.4 * l,  R, 1.5 * R],
                      [0 ,  -R, 1.5 * R],
                      [0 ,  R, 1.5 * R],
                      [-0.4 * l, -R, 1.5 * R],
                      [-0.4 * l, R, 1.5 * R]])
    
    leg = []

    for i in range(6):
        q0 = Rod.deformed_configuration(
            nelements,
            curve,
            dcurve,
            ddcurve,
            np.pi,
            polynomial_degree=polynomial_degree,
            r_OP=r_OP0[i],
            A_IK=np.eye(3, dtype=float),
        )

        legi = Rod(
            cross_section,
            material_model,
            nelements,
            Q=q0,
            q0=q0,
            polynomial_degree=polynomial_degree,
            reduced_integration=reduced_integration,
        )
        leg.append(legi)
        system.add(legi)
        foot_contact = Sphere2Plane(system.origin, legi, mu=0.1, frame_ID=(1,))
        system.add(foot_contact)
        foot_contact2 = Sphere2Plane(system.origin, legi, mu=0.1, frame_ID=(0.9,))
        system.add(foot_contact2)
        gravity = Force_line_distributed(F_grav, legi)
        system.add(gravity)
        box_leg_connection = RigidConnection(box, legi, frame_ID2=(0,))
        system.add(box_leg_connection)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    dt = 5e-1
    solver = BackwardEuler(
        system,
        t1=10,
        dt=dt,
        options=SolverOptions(newton_atol=1e-2, fixed_point_atol=1e-2),
    )

    sol = solver.solve()

    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # VTK export
    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 30, sol)
        for i, legi in enumerate(leg):
            e.export_contr(
                legi,
                level="centerline + directors",
                num=3 * nelements,
                file_name=f"leg_{i}_curve",
            )
            e.export_contr(
                legi,
                continuity="C0",
                level="volume",
                n_segments=nelements,
                num=3 * nelements,
                file_name=f"leg_{i}_volume",
            ) 
        e.export_contr(box)


    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        leg,
        scale=length,
        scale_di=0.5,
        show=False,
        n_frames=leg[0].nelement + 1,
        repeat=True,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    Y_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)

    # plot animation
    ax1.azim = -90
    ax1.elev = 72

    plt.show()


if __name__ == "__main__":
    # SE3 interpolation:
    Rhex(
        Rod=make_CosseratRod_SE3(mixed=True),
        nelements=5,
        polynomial_degree=1,
        n_load_steps=10,
        VTK_export=True,
    )

