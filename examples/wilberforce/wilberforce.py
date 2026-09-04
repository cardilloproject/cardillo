import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
    MoreauTheta,
    SolverOptions,
)
# from cardillo.rods.discreteRod import DiscreteRod


if __name__ == "__main__":
    dir_name = Path(sys.argv[0]).parent

    # nturns = 3  # number of coils
    nturns = 10  # number of coils
    # nturns = 20  # number of coils Harsch2021

    t1 = 20  #
    t1 = 5

    #########
    # gravity
    #########
    gravity = 9.81

    #######################
    # spring modeled as rod
    #######################

    # elements_per_turn = 40
    elements_per_turn = 20
    nelements = int(elements_per_turn * nturns)

    ############
    # Harsch2021
    ############
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

    
    # Rod = DiscreteRod
    
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        mixed=True,
        polynomial_degree=1,
        reduced_integration=True,
        # constraints=[0, 1, 2],
    )

    q0 = Rod.serret_frenet_configuration(
        nelements,
        curve,
        dcurve,
        ddcurve,
        xi1=1,
        # polynomial_degree=polynomial_degree,
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
    )

    rod = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        # polynomial_degree=polynomial_degree,
        cross_section_inertias=cross_section_inertias,
    )

    rod._export_dict["level"] = "NodalVolume"

    ##############
    # pendulum bob
    ##############
    # R = 23e-3  # radius of the main cylinder
    # h = 36e-3  # height of the main cylinder
    R = 25e-3  # radius of the main cylinder
    h = 34e-3  # height of the main cylinder
    density = 7850  # [kg / m^3]; steel
    r_OS0 = np.array([0, 0, -h / 2 - wire_radius])
    p0 = np.array([1, 0, 0, 0], dtype=float)
    q0 = np.concatenate((r_OS0, p0))
    # mass_bob = 0.469
    mass_bob = density * R**2 * np.pi * h
    # K_Theta_S_bob = np.diag([1.468e-4, 1.468e-4, 1.247e-4])
    bob = Cylinder(RigidBody)(radius=R, height=h, density=density, q0=q0)
    # bob = RigidBody(mass_bob, K_Theta_S_bob, q0=q0)

    system = System()

    f_g_rod_statics = lambda t, xi: -t * A_rho0 * gravity * e3
    gravity_rod_statics = Force_line_distributed(f_g_rod_statics, rod)
    f_g_bob_statics = lambda t: -t * mass_bob * gravity * e3
    gravity_bob_statics = Force(f_g_bob_statics, bob)

    f_pulling = lambda t: -t * mass_bob * gravity * e3 * 0.3
    pulling_force = Force(f_pulling, bob)

    joint1 = RigidConnection(system.origin, rod, xi2=1)
    joint2 = RigidConnection(bob, rod, xi2=0)

    #####################
    # assemble the system
    #####################
    # system.add(rod, joint1, force_rod)
    system.add(
        rod,
        bob,
        joint1,
        joint2,
        # gravity_rod_statics,
        # gravity_bob_statics,
        # pulling_force,
    )
    # system.assemble()


    load_sol = False
    if load_sol:
        from cardillo.solver.solution import load_solution
        sol = load_solution(dir_name / "wilberforce2p0_sol.npy")
        sol.system = system
    else:
        # #####################
        # # solve static system
        # #####################
        # sol = Newton(
        #     system,
        #     n_load_steps=10,
        # ).solve()
        # q = sol.q
        # nt = len(q)
        # t = sol.t[:nt]

        # system.set_new_initial_state(q0=sol.q[-1], u0=sol.u[-1])

        f_g_rod = lambda t, xi: -A_rho0 * gravity * e3
        gravity_rod = Force_line_distributed(f_g_rod, rod)
        f_g_bob = lambda t: -mass_bob * gravity * e3
        gravity_bob = Force(f_g_bob, bob)

        ########################
        # solve dynamical system
        ########################
        # system.remove(gravity_bob_statics, gravity_rod_statics, pulling_force)
        system.add(gravity_bob, gravity_rod)
        system.assemble(options=SolverOptions(compute_consistent_initial_conditions=True))
        dt = 1e-3
        solver = MoreauTheta(
            system, 
            t1=t1, 
            dt=dt, 
            theta=0.75,
        )
        sol = solver.solve()
        sol.system = None
        from cardillo.solver.solution import save_solution
        save_solution(sol, dir_name / "wilberforce2p0_sol.npy")

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
    angles = np.unwrap(angles, axis=0)

    
    # Export for pgfplots
    # data = np.column_stack([t[::10], r_OS[::10, 2],  np.rad2deg(angles[::10, 0])])
    # dir_name = Path(sys.argv[0]).parent
    # np.savetxt(
    #     dir_name / ".." / "latex src" / "figures" / "data_wilberforce2p0.csv",
    #     data,
    #     delimiter=",",
    #     header="t,z,alpha",
    #     comments="",
    # )

    # compute reference and current arc length
    # l0 = 
    # q_rod = q[rod.qDOF].reshape(len(t), 7, -1)[:, :3]

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(3, 1)

    # ax[0].plot(t, r_OS[:, 0], label="x")
    # ax[0].plot(t, r_OS[:, 1], label="y")
    ax[0].plot(t, r_OS[:, 2], "-o", label="z")
    ax[0].set_ylabel("position [m]")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, np.rad2deg(angles[:, 0]), "-o", label="alpha")
    # ax[1].plot(t, np.rad2deg(angles[:, 1]), label="beta")
    # ax[1].plot(t, np.rad2deg(angles[:, 2]), label="gamma")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("angle [deg]")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t[:-1], np.diff(t), "-o", label="h")
    ax[2].set_yscale("log")
    ax[2].legend()
    ax[2].grid()


    plt.show()

    ############
    # VTK export
    ############
    VTK_export = True
    print("exporting VTK")
    if VTK_export:
        # fake second bob for export
        bob_glyph = RigidBody(1.0, np.eye(3, dtype=float), name="bob_glyph")
        bob_glyph.qDOF = bob.qDOF
        bob_glyph.uDOF = bob.uDOF
        system.add(bob_glyph)
        system.export(dir_name, f"vtk/wilberforce_pendulum", sol, fps=25)
    print("finished")
    exit()