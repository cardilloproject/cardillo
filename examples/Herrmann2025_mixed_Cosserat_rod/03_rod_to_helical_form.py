from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.forces import Force, Moment
from cardillo.math import e3
from cardillo.rods import CircularCrossSection, Simo1986, animate_beam
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.solver import Newton, SolverOptions


# On the choice of finite rotation parameters
# https://doi.org/10.1016/S0045-7825(97)00059-5
# Adnan Ibrahimbegovic

# SE(3) publication
# https://www.simonreugster.com/literature/journal_publications/Harsch2023a.pdf
# Example 4.4


def rod_to_helical_form(
    Rod,
    constitutive_law,
    *,
    nelements: int = 30,
    n_load_steps: int = 53,
    #
    name: str = "simulation",
    show_plots: bool = False,
    VTK_export: bool = False,
    save_tip_displacement: bool = False,
    save_stresses: bool = False,
):
    plot_name = name.replace("_", " ")
    save_name = f'{name.replace(" ", "_")}_nel{nelements}'
    ########################################
    # ibrahimbegovic example of Harsch2023 #
    ########################################

    # length od the rod
    length = 10

    # used cross section
    width = 0.005

    # cross section is just used for the visualization
    cross_section = CircularCrossSection(width)
    cross_section_fat = CircularCrossSection(length / 100)

    # material law
    Ei = 1e4 * np.array([1, 1, 1])
    Fi = 1e2 * np.array([1, 1, 1])
    material_model = constitutive_law(Ei, Fi)

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IB0 = np.eye(3, dtype=float)

    q0 = Rod.straight_configuration(
        nelements,
        length,
        r_OP0=r_OP0,
        A_IB0=A_IB0,
    )
    rod = Rod(cross_section, material_model, nelements, Q=q0)

    # left joint
    system = System()
    joint1 = RigidConnection(system.origin, rod, xi2=0)

    # moment at right end
    Fi = material_model.Fi
    F_max = 50
    F = lambda t: e3 * F_max * t
    force = Force(F, rod, 1)
    M_max = 20 * Fi[2] * np.pi / length
    M = lambda t: e3 * (M_max * t)
    moment = Moment(M, rod, 1)

    # assemble the system
    system.add(rod)
    system.add(joint1)
    system.add(moment)
    system.add(force)
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # add Newton solver
    atol = 1e-8
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_max_iter=30, newton_atol=atol),  # rtol=0
    )
    sol = solver.solve()

    # read solution
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]
    la_c = sol.la_c
    la_g = sol.la_g

    #################
    # post-processing
    #################
    # vtk-export
    dir_name = Path(sys.argv[0]).parent
    if VTK_export:
        # add surface normals
        rod.name = "rod"
        # rod._export_dict["surface_normals"] = True # TODO: resolve this issue
        rod._export_dict["hasCap"] = True

        # add rod with larger cross-section for better visualization
        rod_fat = deepcopy(rod)
        rod_fat.name = "rod_fat"
        rod_fat.cross_section = cross_section_fat
        system.add(rod_fat)

        # add rod for centerline export
        rod_centerline = deepcopy(rod)
        rod_centerline.name = "rod_centerline"
        rod_centerline._export_dict["level"] = "centerline + directors"
        system.add(rod_centerline)

        system.export(dir_name, f"vtk/{save_name}", sol)

    # animation
    if show_plots:
        fig1, ax1, anim1 = animate_beam(
            t, q, [rod], length, n_r=5, n_frames=5, show=False, repeat=True
        )

    # tip displacement over load steps
    fig, ax = plt.subplots(1, 1)
    if len(t) == n_load_steps + 1:
        qDOF_tip = rod.elDOF_P(1)
        r_OP0_tip = rod.r_OP(0, q0[qDOF_tip], 1)
        delta_tip_header = "time, load_force, load_moment, delta_x, delta_y, delta_z"
        delta_tip = np.zeros((6, n_load_steps + 1), dtype=float)
        for i in range(n_load_steps + 1):
            delta_tip[0, i] = t[i]
            delta_tip[1, i] = t[i] * F_max
            delta_tip[2, i] = t[i] * M_max
            qe = q[i][rod.qDOF][qDOF_tip]
            delta_tip[3:, i] = rod.r_OP(t[i], qe, 1) - r_OP0_tip

        fig.suptitle(f"Tip displacement {plot_name}")
        ax.plot(delta_tip[0], delta_tip[3], "r-")
        ax.plot(delta_tip[0], delta_tip[4], "g-")
        ax.plot(delta_tip[0], delta_tip[5], "b-")
        ax.grid()
        ax.set_xlabel("Load parameter $t$")
        ax.set_ylabel("Tip displacement")

        if save_tip_displacement:
            path_tip = Path(dir_name, "csv", "tip_displacement")
            path_tip.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                path_tip / f"{save_name}.csv",
                delta_tip.T,
                delimiter=", ",
                header=delta_tip_header,
                comments="",
            )

    # stresses along the rod
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
    fig2.suptitle(f"Stresses {plot_name}")
    for i in range(2):
        ax2[i].plot(stresses[0], stresses[3 * i + 1], "r")
        ax2[i].plot(stresses[0], stresses[3 * i + 2], "g")
        ax2[i].plot(stresses[0], stresses[3 * i + 3], "b")
        ax2[i].grid()

    ax2[0].set_ylabel(r"$_B n$")
    ax2[1].set_ylabel(r"$_B m$")
    ax2[1].set_xlabel(r"$\xi$")

    if save_stresses:
        path_stresses = Path(dir_name, "csv", "stresses")
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
    rod_to_helical_form(
        Rod,
        Simo1986,
        nelements=30,
        n_load_steps=90,
        show_plots=True,
        name="Rod to helical form",
    )
