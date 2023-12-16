from cardillo.rods import (
    RectangularCrossSection,
    CrossSectionInertias,
    Simo1986,
    animate_beam,
)

from cardillo.discrete.shapes import RectangleTrimesh

from cardillo.rods.cosseratRod import (
    make_CosseratRod_SE3,
    make_CosseratRod_R12,
    make_CosseratRod_Quat,
)

from cardillo.discrete import Cuboid, RigidBody, Frame

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

def Rhex_leg_stiffnes(
    Rod,
    nelements=2,
    polynomial_degree=2,
    n_load_steps=10,
    reduced_integration=True,
):
    # geometry of the rod
    leg_R = 0.165 / 2
    l_leg = np.pi * leg_R

    # cross section properties for visualization purposes
    slenderness = 1.0e2
    h_leg = l_leg / slenderness
    w_leg = 8 * h_leg
    # cross_section = RectangularCrossSection(density, width, width)
    cross_section = RectangularCrossSection(h_leg, w_leg)
    density = 0.08 / (l_leg * cross_section.area)
    cross_section_inertias = CrossSectionInertias(density, cross_section)

    A = cross_section.area
    Ip, Iy, Iz = np.diagonal(cross_section.second_moment)

    E = 40.6e9
    G = E / 2
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * Iy, E * Iz])

    material_model = Simo1986(Ei, Fi)

    # construct system
    system = System()

    start_angle = np.pi / 2
    # definition of the arch curve
    curve = lambda xi: np.array(
        [leg_R * np.cos(xi + start_angle), 0, leg_R * np.sin(xi + start_angle)]
    )
    dcurve = lambda xi: np.array(
        [-leg_R * np.sin(xi + start_angle), 0, leg_R * np.cos(xi + start_angle)]
    )
    ddcurve = lambda xi: np.array(
        [-leg_R * np.cos(xi + start_angle), 0, -leg_R * np.sin(xi + start_angle)]
    )

    q0 = Rod.deformed_configuration(
            nelements,
            curve,
            dcurve,
            ddcurve,
            np.pi,
            polynomial_degree=polynomial_degree,
        )
    
    leg = Rod(
            cross_section,
            material_model,
            nelements,
            Q=q0,
            q0=q0,
            polynomial_degree=polynomial_degree,
            reduced_integration=reduced_integration,
        )
    
    system.add(leg)
    clamping = RigidConnection(leg, system.origin, frame_ID1 = (0,))
    system.add(clamping)

    f_max = 1.9
    tip_force = lambda t: f_max * t * e3
    force = Force(tip_force, leg, frame_ID=(1,))
    system.add(force)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_atol=1e-8),
    )

    sol = solver.solve()

    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    u = np.zeros(nt)

    q_c2 = sol.q[:, leg.qDOF]
    qe_c2 = q_c2[:, leg.local_qDOF_P((1,))]

    r_OP0 = leg.r_OP(0, qe_c2[0], frame_ID=(1,))
    for i in range(nt):
        r_OPi = leg.r_OP(0, qe_c2[i], frame_ID=(1,))
        u[i] = (r_OPi - r_OP0)[2]

    _, ax = plt.subplots()

    ax.plot(u, f_max * t, "-", color="blue", marker="x", label="force-displacement leg")
    ax.grid()
    ax.set_xlabel("displacement")
    ax.set_ylabel("force")
    ax.plot([0, 1e-3], [0, f_max], label="desired stiffness")
    ax.legend()

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        [leg],
        scale=l_leg,
        scale_di=0.05,
        show=False,
        n_frames=leg.nelement + 1,
        repeat=True,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-1.1 * l_leg, 1.1 * l_leg, num=2)
    Y_z = np.linspace(-1.1 * l_leg, 1.1 * l_leg, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)

    # # plot animation
    # ax1.azim = -90
    # ax1.elev = 72

    plt.show()

    

def Rhex(
    Rod,
    nelements=2,
    polynomial_degree=2,
    VTK_export=False,
    reduced_integration=True,
):
    # geometry of the rod
    leg_R = 0.165 / 2
    l_leg = np.pi * leg_R

    # cross section properties for visualization purposes
    slenderness = 1.0e2
    h_leg = l_leg / slenderness
    w_leg = 8 * h_leg
    # cross_section = RectangularCrossSection(density, width, width)
    cross_section = RectangularCrossSection(h_leg, w_leg)
    density = 0.08 / (l_leg * cross_section.area)
    cross_section_inertias = CrossSectionInertias(density, cross_section)

    A = cross_section.area
    Ip, Iy, Iz = np.diagonal(cross_section.second_moment)

    E = 40.6e9
    G = E / 2
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * Iy, E * Iz])

    material_model = Simo1986(Ei, Fi)

    # construct system
    system = System()

    h_drop = 0.1
    l_box = 0.5
    w_box = 0.2
    h_box = 0.15

    dimensions = np.array([l_box, w_box, h_box])
    q0_box = np.array([0, 0, 0.5 * h_box + leg_R + h_drop, 1, 0, 0, 0])
    box = Cuboid(RigidBody)(dimensions=dimensions, mass=8.5, q0 = q0_box, u0=np.zeros(6))
    system.add(box)
    box_gravity = Force(-box.mass * 9.81 * e3, box)
    system.add(box_gravity)

    start_angle = np.pi / 2
    # definition of the arch curve
    curve = lambda xi: np.array(
        [leg_R * np.cos(xi + start_angle), 0, leg_R * np.sin(xi + start_angle)]
    )
    dcurve = lambda xi: np.array(
        [-leg_R * np.sin(xi + start_angle), 0, leg_R * np.cos(xi + start_angle)]
    )
    ddcurve = lambda xi: np.array(
        [-leg_R * np.cos(xi + start_angle), 0, -leg_R * np.sin(xi + start_angle)]
    )

    leg_gravity = lambda t, xi: - density * A * 9.81 * e3

    # fmt: off
    r_OP0 = np.array([[0.4 * l_box , -w_box / 2 - 1.5 * w_leg, leg_R + h_drop],
                      [0.4 * l_box ,  w_box / 2 + 1.5 * w_leg, leg_R + h_drop],
                      [0           , -w_box / 2 - 1.5 * w_leg, leg_R + h_drop],
                      [0           ,  w_box / 2 + 1.5 * w_leg, leg_R + h_drop],
                      [-0.4 * l_box, -w_box / 2 - 1.5 * w_leg, leg_R + h_drop],
                      [-0.4 * l_box,  w_box / 2 + 1.5 * w_leg, leg_R + h_drop]])
    # fmt: on
    
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
        foot_contact = Sphere2Plane(system.origin, legi, mu=0.2, frame_ID=(1,))
        system.add(foot_contact)
        foot_contact2 = Sphere2Plane(system.origin, legi, mu=0.2, frame_ID=(0.9,))
        system.add(foot_contact2)
        foot_contact3 = Sphere2Plane(system.origin, legi, mu=0.2, frame_ID=(0.8,))
        system.add(foot_contact3)
        gravity = Force_line_distributed(leg_gravity, legi)
        system.add(gravity)
        box_leg_connection = RigidConnection(box, legi, frame_ID2=(0,))
        system.add(box_leg_connection)

    rectangle = RectangleTrimesh(Frame)(dimensions=[5, 8], axis=2)


    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    dt = 1e-2
    solver = BackwardEuler(
        system,
        t1=2,
        dt=dt,
        options=SolverOptions(newton_atol=1e-4, fixed_point_atol=1e-3),
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
                num=15 * nelements,
                file_name=f"leg_{i}_curve",
            )
            e.export_contr(
                legi,
                continuity="C0",
                level="volume",
                n_segments= 2,
                num= 10,
                file_name=f"leg_{i}_volume",
            ) 
        e.export_contr(box, file_name="box")
        e.export_contr(rectangle, file_name="floor")


    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        leg,
        scale=l_leg,
        scale_di=0.05,
        show=False,
        n_frames=leg[0].nelement + 1,
        repeat=True,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-1.1 * l_leg, 1.1 * l_leg, num=2)
    Y_z = np.linspace(-1.1 * l_leg, 1.1 * l_leg, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)

    # # plot animation
    # ax1.azim = -90
    # ax1.elev = 72

    plt.show()


if __name__ == "__main__":

    Rhex(
        Rod=make_CosseratRod_SE3(mixed=True),
        nelements=2,
        polynomial_degree=1,
        VTK_export=True,
    )


    # Rhex_leg_stiffnes(dr
    #     Rod=make_CosseratRod_SE3(mixed=True),
    #     nelements=2,
    #     polynomial_degree=1,
    #     n_load_steps=2,
    # )
