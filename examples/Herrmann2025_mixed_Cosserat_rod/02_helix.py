import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.forces import B_Moment
from cardillo.math import e1, e3
from cardillo.rods import CircularCrossSection, animate_beam, Simo1986
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.solver import Newton, SolverOptions


def helix(
    Rod,
    constitutive_law,
    *,
    nelements: int = 10,
    #
    n_coil: float = 2,
    slenderness: float = 1.0e4,
    n_load_steps: int = 10,
    atol: float | None = None,
    #
    VTK_export: bool = False,
    name: str = "rod",
    show_plots: bool = False,
    save_stresses: bool = False,
):
    # handle name
    plot_name = name.replace("_", " ")
    save_name = f'{name.replace(" ", "_")}_nel{nelements}'
    print(f"Slenderness: {slenderness:1.0e}, Rod: {plot_name}, nel: {nelements}")

    # geometry of the rod
    R0 = 10  # radius of the helix
    h = 50  # height of the helix
    c = h / (2 * R0 * np.pi * n_coil)  # pitch of the helix
    length = np.sqrt(1 + c**2) * R0 * 2 * np.pi * n_coil
    cc = 1 / (np.sqrt(1 + c**2))

    alpha = lambda xi: 2 * np.pi * n_coil * xi
    alpha_xi = 2 * np.pi * n_coil

    # cross section properties
    width = length / slenderness
    radius = width / 2
    cross_section = CircularCrossSection(radius=radius)
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)

    # material model
    E = 1.0  # Young's modulus
    G = 0.5  # shear modulus
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = constitutive_law(Ei, Fi)

    # initialize system
    system = System()

    # initial positions and orientations at xi=0
    alpha_0 = alpha(0)

    r_OP0 = R0 * np.array([np.sin(alpha_0), -np.cos(alpha_0), c * alpha_0])

    e_x = cc * np.array([np.cos(alpha_0), np.sin(alpha_0), c])
    e_y = np.array([-np.sin(alpha_0), np.cos(alpha_0), 0])
    e_z = cc * np.array([-c * np.cos(alpha_0), -c * np.sin(alpha_0), 1])

    A_IB0 = np.vstack((e_x, e_y, e_z))
    A_IB0 = A_IB0.T

    #####
    # rod
    #####
    # generate position coordinates for straight initial configuration
    q0 = Rod.straight_configuration(
        nelements,
        length,
        r_OP0=r_OP0,
        A_IB0=A_IB0,
    )
    # create rod
    rod = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
    )
    system.add(rod)

    ##########
    # clamping
    ##########
    clamping = RigidConnection(system.origin, rod, xi2=0)
    system.add(clamping)

    ################
    # applied moment
    ################
    Fi = material_model.Fi
    M = lambda t: (R0 * alpha_xi**2) / (length**2) * (c * e1 * Fi[0] + e3 * Fi[2]) * t
    moment = B_Moment(M, rod, 1)
    system.add(moment)

    # assemble system
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # add Newton solver
    atols_dict = {1e1: 1e-8, 1e2: 1e-10, 1e3: 1e-12, 1e4: 1e-14}
    if atol == None:
        atol = atols_dict[slenderness]

    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_max_iter=30, newton_atol=atol),  # rtol=0
    )
    # create solver
    sol = solver.solve()  # solve static equilibrium equations

    # read solution
    t = sol.t
    q = sol.q
    la_c = sol.la_c
    la_g = sol.la_g

    #################
    # post-processing
    #################

    # VTK export
    dir_name = Path(sys.argv[0]).parent
    if VTK_export:
        system.export(dir_name, f"vtk/slen_{slenderness:1.0e}/{save_name}", sol)

    ##########################
    # matplotlib visualization
    ##########################
    # construct animation of beam
    if show_plots:
        fig1, ax1, anim1 = animate_beam(
            t,
            q,
            [rod],
            scale=length,
            show=False,
            repeat=False,
        )

    ########################
    # stresses along the rod
    ########################
    nxi_ges_min = 201
    nxi_el = max(11, int(np.ceil((nxi_ges_min + rod.nelement - 1) / rod.nelement)))
    stresses_header = "xi, nx, ny, nz, mx, my, mz"
    stresses = np.zeros((7, nxi_el * rod.nelement), dtype=float)
    for el in range(rod.nelement):
        xi_el = np.linspace(*rod.element_interval(el), nxi_el)
        for i in range(nxi_el):
            idx = el * nxi_el + i
            stresses[0, idx] = xi_el[i]
            B_n, B_m = rod.eval_stresses(t[-1], q[-1], la_c[-1], la_g[-1], xi_el[i], el)
            stresses[1:, idx] = *B_n, *B_m

    fig2, ax2 = plt.subplots(2, 1)
    fig2.suptitle(f"Stresses {name}")
    for i in range(2):
        ax2[i].plot(stresses[0], stresses[3 * i + 1], "r")
        ax2[i].plot(stresses[0], stresses[3 * i + 2], "g")
        ax2[i].plot(stresses[0], stresses[3 * i + 3], "b")
        ax2[i].grid()

    ax2[0].set_ylabel(r"$_B n$")
    ax2[1].set_ylabel(r"$_B m$")
    ax2[1].set_xlabel(r"$\xi$")

    if save_stresses:
        path_stresses = Path(dir_name, "csv", f"slen_{slenderness:1.0e}", "stresses")
        path_stresses.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            path_stresses / f"{save_name}.csv",
            stresses.T,
            delimiter=", ",
            header=stresses_header,
            comments="",
        )

    if show_plots:
        plt.show()


if __name__ == "__main__":
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        mixed=True,
        polynomial_degree=2,
        reduced_integration=True,
    )
    helix(
        Rod,
        Simo1986,
        nelements=16,
        slenderness=1e1,
        n_load_steps=1,
        show_plots=True,
        name="helix",
    )
