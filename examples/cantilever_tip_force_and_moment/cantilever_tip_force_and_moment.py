import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.forces import Force, B_Moment
from cardillo.math import e2, e3
from cardillo.rods import (
    RectangularCrossSection,
    Simo1986,
    Harsch2021,
    animate_beam,
)
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.solver import Newton, SolverOptions


""" Cantilever beam example from

Harsch, J., Capobianco, G. and Eugster, S. R., "Finite element formulations for constrained spatial nonlinear beam theories", 2021.
https://doi.org/10.1177/10812865211000790

4.1 Elliptic integral solutions of Euler's elastica

This is an example how to switch on and off the constraints.
"""


def cantilever(
    Rod,
    constitutive_law=Harsch2021,
    *,
    nelements: int = 10,
    polynomial_degree: int = 2,
    reduced_integration: bool = True,
    #
    n_load_steps: int = 3,
    #
    VTK_export: bool = False,
    title: str = "set_a_plot_title",
):
    print(title)
    ############
    # parameters
    ############
    # geometry of the rod
    length = 2 * np.pi

    # cross section properties for visualization purposes
    slenderness = 1.0e2
    width = length / slenderness
    cross_section = RectangularCrossSection(width, width)

    # material properties
    Ei = np.array([5, 1, 1])
    Fi = np.array([0.5, 2, 2])
    material_model = constitutive_law(Ei, Fi)

    # initialize system
    system = System()

    #####
    # rod
    #####
    # compute straight initial configuration of cantilever
    q0 = Rod.straight_configuration(
        nelements, length, polynomial_degree=polynomial_degree
    )
    # construct cantilever
    cantilever = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        reduced_integration=reduced_integration,
    )

    ##########
    # clamping
    ##########
    clamping = RigidConnection(system.origin, cantilever, xi2=0)
    system.add(cantilever, clamping)

    ###############
    # applied loads
    ###############
    # spatially fixed load at cantilever tip
    P = lambda t: material_model.Fi[2] * (10 * t) / length**2
    F = lambda t: -P(t) * e2
    force = Force(F, cantilever, 1)
    system.add(force)

    # moment at cantilever tip
    M = lambda t: 2.5 * P(t) * e3
    moment = B_Moment(M, cantilever, 1)
    system.add(moment)

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    solver = Newton(system, n_load_steps=n_load_steps)  # create solver
    sol = solver.solve()  # solve static equilibrium equations

    # read solution
    t = sol.t
    q = sol.q

    #################
    # post-processing
    #################

    # VTK export
    dir_name = Path(__file__).parent
    if VTK_export:
        system.export(dir_name, "vtk", sol)

    ##########################
    # matplotlib visualization
    ##########################
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        [cantilever],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever.nelement + 1,
        repeat=False,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    Y_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)
    ax1.set_title(title)

    path = Path(__file__)  # creation of a current path

    # add reference solution to the animation
    centerline_T = np.loadtxt(
        Path(
            path.parent,
            "_data",
            "centerline_T_numeric.csv",
        ),
        delimiter=",",
        skiprows=1,
    )
    ax1.plot(
        centerline_T[:, -2],
        centerline_T[:, -1],
        np.zeros_like(centerline_T[:, 0]),
        "-b",
    )
    centerline_EB = np.loadtxt(
        Path(
            path.parent,
            "_data",
            "centerline_EB_numeric.csv",
        ),
        delimiter=",",
        skiprows=1,
    )
    ax1.plot(
        centerline_EB[:, -2],
        centerline_EB[:, -1],
        np.zeros_like(centerline_EB[:, 0]),
        "-g",
    )
    centerline_IEB = np.loadtxt(
        Path(
            path.parent,
            "_data",
            "centerline_IEB_analytic.csv",
        ),
        delimiter=",",
        skiprows=1,
    )
    ax1.plot(
        centerline_IEB[:, -2],
        centerline_IEB[:, -1],
        np.zeros_like(centerline_IEB[:, 0]),
        "-r",
    )

    # plot animation
    ax1.azim = -90
    ax1.elev = 72

    plt.show()

    def stress_strain(rod, sol, nxi_el=10):
        xis = []
        epsilon_Gamma = []
        epsilon_Kappa = []
        B_n = []
        B_m = []

        t = sol.t[-1]
        q = sol.q[-1]
        la_c = sol.la_c[-1]
        la_g = sol.la_g[-1]

        for el in range(rod.nelement):
            xi_el = np.linspace(*rod.knot_vector_r.element_interval(el), nxi_el)
            epsilon_Gamma_el = np.zeros([nxi_el, 3])
            epsilon_Kappa_el = np.zeros([nxi_el, 3])
            B_n_el = np.zeros([nxi_el, 3])
            B_m_el = np.zeros([nxi_el, 3])
            for i, xii in enumerate(xi_el):
                B_n_el[i], B_m_el[i] = rod.eval_stresses(t, q, la_c, la_g, xii, el)
                epsilon_Gamma_el[i], epsilon_Kappa_el[i] = rod.eval_strains(
                    t, q, la_c, la_g, xii, el
                )

            xis.append(xi_el)
            epsilon_Gamma.append(epsilon_Gamma_el)
            epsilon_Kappa.append(epsilon_Kappa_el)
            B_n.append(B_n_el)
            B_m.append(B_m_el)

        xis = np.concatenate(xis)
        epsilon_Gamma = np.concatenate(epsilon_Gamma).T
        epsilon_Kappa = np.concatenate(epsilon_Kappa).T
        B_n = np.concatenate(B_n).T
        B_m = np.concatenate(B_m).T
        return xis, epsilon_Gamma, epsilon_Kappa, B_n, B_m

    fig, ax = plt.subplots(1, 4)

    xis, K_Gamma, K_Kappa, K_n, K_m = stress_strain(cantilever, sol)

    ax[0].set_title("K_Gamma - K_Gamma0")
    ax[0].plot(xis, K_Gamma[0], label="Delta K_Gamma0")
    ax[0].plot(xis, K_Gamma[1], label="Delta K_Gamma1")
    ax[0].plot(xis, K_Gamma[2], label="Delta K_Gamma2")

    ax[1].set_title("K_Kappa - K_Kappa0")
    ax[1].plot(xis, K_Kappa[0], label="Delta K_Kappa0")
    ax[1].plot(xis, K_Kappa[1], label="Delta K_Kappa1")
    ax[1].plot(xis, K_Kappa[2], label="Delta K_Kappa2")

    ax[2].set_title("K_n")
    ax[2].plot(xis, K_n[0], label="K_n0")
    ax[2].plot(xis, K_n[1], label="K_n1")
    ax[2].plot(xis, K_n[2], label="K_n2")

    ax[3].set_title("K_m")
    ax[3].plot(xis, K_m[0], label="K_m0")
    ax[3].plot(xis, K_m[1], label="K_m1")
    ax[3].plot(xis, K_m[2], label="K_m2")

    for axi in ax.flat:
        axi.grid()
        axi.legend()

    plt.show()


if __name__ == "__main__":
    ############################
    # Quaternion interpolation #
    ############################
    # displacement-based formulation
    cantilever(
        Rod=make_CosseratRod(mixed=False),
        constitutive_law=Harsch2021,
        title="shear-deformable (blue): D-B quaternion interpolation",
    )

    cantilever(
        Rod=make_CosseratRod(mixed=False, constraints=[1, 2]),
        constitutive_law=Harsch2021,
        title="shear-rigid (green): constrained D-B quaternion interpolation",
    )

    cantilever(
        Rod=make_CosseratRod(mixed=False, constraints=[0, 1, 2]),
        constitutive_law=Harsch2021,
        title="inextensible shear-rigid (red): constrained D-B quaternion interpolation",
    )

    # mixed formulation
    # For shear-rigid rods Harsch2021 and Simo1986 coincide.
    cantilever(
        Rod=make_CosseratRod(mixed=True, constraints=[1, 2]),
        constitutive_law=Simo1986,
        title="shear-rigid (green): constrained mixed quaternion interpolation",
    )

    cantilever(
        Rod=make_CosseratRod(mixed=True, constraints=[0, 1, 2]),
        constitutive_law=Simo1986,
        title="inextensible shear-rigid (red): constrained mixed quaternion interpolation",
    )
