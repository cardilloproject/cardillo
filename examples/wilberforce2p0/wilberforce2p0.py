import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

from cardillo import System
from cardillo.constraints import RigidConnection, Spherical
from cardillo.discrete import RigidBody, Cylinder
from cardillo.forces import Force, Moment
from cardillo.math import e1, e2, e3, ax2skew
from cardillo.rods import (
    CircularCrossSection,
    CrossSectionInertias,
    Simo1986,
    animate_beam,
)
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.rods.force_line_distributed import Force_line_distributed
from cardillo.solver import (
    Newton,
    BackwardEuler,
    SolverOptions,
    ScipyDAE,
    DualStormerVerlet,
    Moreau,
    MoreauCompliance,
)


if __name__ == "__main__":

    nturns = 3  # number of coils

    # t1 = 20
    t1 = 5

    #########
    # gravity
    #########
    gravity = 9.81

    #######################
    # spring modeled as rod
    #######################
    polynomial_degree = 1
    elements_per_turn = 15
    # elements_per_turn = 30
    nelements = int(elements_per_turn * nturns)

    Rod = make_CosseratRod(
        interpolation="Quaternion",
        mixed=True,
        polynomial_degree=polynomial_degree,
        reduced_integration=True,
    )

    ############
    # Harsch2021
    ############
    # steel
    rho = 7850  # [kg / m^3]
    G = 81.5e9
    E = 206.0e9
    print(f"G: {G}; E: {E}")

    # 1mm cross sectional diameter
    wire_diameter = 1e-3
    wire_radius = wire_diameter / 2

    # helix parameter
    coil_diameter = 32.0e-3
    coil_radius = coil_diameter / 2
    pitch_unloaded = wire_diameter
    c = pitch_unloaded / (coil_radius * 2 * np.pi)

    # rod cross-section
    cross_section = CircularCrossSection(wire_radius)
    cross_section_inertias = CrossSectionInertias(rho, cross_section)

    A_rho0 = rho * cross_section.area
    K_S_rho0 = rho * cross_section.first_moment
    K_I_rho0 = rho * cross_section.second_moment
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)
    print(f"Ei: {Ei}")
    print(f"Fi: {Fi}")

    # helix and derivatives
    def r(xi, phi0=0):
        alpha = 2 * np.pi * nturns * xi
        return coil_radius * np.array(
            [np.sin(alpha + phi0), -np.cos(alpha + phi0), c * alpha]
        )

    def dr(xi, phi0=0):
        alpha = 2 * np.pi * nturns * xi
        return (
            coil_radius
            * 2
            * np.pi
            * nturns
            * np.array([np.cos(alpha + phi0), np.sin(alpha + phi0), c])
        )

    def ddr(xi, phi0=0):
        alpha = 2 * np.pi * nturns * xi
        return (
            coil_radius
            * (2 * np.pi * nturns) ** 2
            * np.array([-np.sin(alpha + phi0), np.cos(alpha + phi0), 0])
        )

    # definition of the parametric curve
    curve = lambda xi: r(xi, phi0=np.pi)
    dcurve = lambda xi: dr(xi, phi0=np.pi)
    ddcurve = lambda xi: ddr(xi, phi0=np.pi)

    q0 = Rod.serret_frenet_configuration(
        nelements,
        curve,
        dcurve,
        ddcurve,
        xi1=1,
        alpha=0.0,
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
    )

    rod = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        cross_section_inertias=cross_section_inertias,
    )

    ##############
    # pendulum bob
    ##############
    R = 25e-3  # radius of the main cylinder
    h = 34e-3  # height of the main cylinder
    density = 7850  # [kg / m^3]; steel
    r_OS0 = np.array([0, 0, -h / 2 - wire_radius])
    p0 = np.array([1, 0, 0, 0], dtype=float)
    q0 = np.concatenate((r_OS0, p0))
    mass_bob = density * R**2 * np.pi * h
    bob = Cylinder(RigidBody)(radius=R, height=h, density=density, q0=q0)

    print(f"bob mass: {mass_bob}")
    print(f"bob B_Theta_C:\n{bob.B_Theta_C}")

    system = System()

    joint1 = RigidConnection(system.origin, rod, xi2=1)
    joint2 = RigidConnection(bob, rod, xi2=0)

    f_g_rod = lambda t, xi: -A_rho0 * gravity * e3
    gravity_rod = Force_line_distributed(f_g_rod, rod)
    f_g_bob = lambda t: -mass_bob * gravity * e3
    gravity_bob = Force(f_g_bob, bob)

    #####################
    # assemble the system
    #####################
    system.add(
        rod,
        bob,
        joint1,
        joint2,
        gravity_rod,
        gravity_bob,
    )
    system.assemble()

    solver = ScipyDAE(
        system,
        t1=t1,
        dt=1.0e-3,
        method="Radau",
        atol=1e-1,
        rtol=1e-1,
        stages=3,
    )
    # t1 = 0.2
    # solver = DualStormerVerlet(
    #     system,
    #     t1=t1,
    #     dt=1e-4,
    #     linear_solver="LU",
    #     accelerated=False,
    # )
    # solver = MoreauCompliance(
    #     system,
    #     t1,
    #     dt=5e-4,
    #     theta=0.55,
    # )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # ################################
    # # plot characteristic quantities
    # ################################
    r_OS = np.array([bob.r_OP(ti, qi[bob.qDOF]) for (ti, qi) in zip(sol.t, sol.q)])

    ordering = "zyx"
    angles = np.array(
        [
            Rotation.from_matrix(bob.A_IB(ti, qi[bob.qDOF])).as_euler(ordering)
            for (ti, qi) in zip(sol.t, sol.q)
        ]
    )

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(t, r_OS[:, 0], label="x")
    ax[0].plot(t, r_OS[:, 1], label="y")
    ax[0].plot(t, r_OS[:, 2], label="z")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, np.rad2deg(angles[:, 0]), label="alpha")
    ax[1].plot(t, np.rad2deg(angles[:, 1]), label="beta")
    ax[1].plot(t, np.rad2deg(angles[:, 2]), label="gamma")
    ax[1].legend()
    ax[1].grid()

    ###########
    # animation
    ###########
    _ = animate_beam(t, q, [rod], 0.05, scale_di=0.01, show=False)

    plt.show()

    ############
    # VTK export
    ############
    # VTK export
    dir_name = Path(__file__).parent
    VTK_export = False
    if VTK_export:
        # fake second bob for export
        bob_glyph = RigidBody(1.0, np.eye(3, dtype=float), name="bob_glyph")
        bob_glyph.qDOF = bob.qDOF
        bob_glyph.uDOF = bob.uDOF
        system.add(bob_glyph)
        system.export(dir_name, "vtk", sol, fps=50)
