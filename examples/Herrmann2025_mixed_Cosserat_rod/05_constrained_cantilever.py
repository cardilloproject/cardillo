import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.forces import B_Force, B_Moment, Force, Moment
from cardillo.rods import RectangularCrossSection, Simo1986, animate_beam
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.solver import Newton, SolverOptions


def cantilever(
    Rod,
    constitutive_law=Simo1986,
    *,
    nelements: int = 10,
    #
    n_load_steps: int = 10,
    load_type: str = "force",
    #
    VTK_export: bool = False,
    name: str = "simulation",
    show_plots: bool = False,
    save_centerlines: bool = False,
    save_tip_displacement: bool = False,
):
    # handle name
    plot_name = name.replace("_", " ")
    save_name = name.replace(" ", "_")

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
    q0 = Rod.straight_configuration(nelements, length)
    # construct cantilever
    cantilever = Rod(cross_section, material_model, nelements, Q=q0)

    ##########
    # clamping
    ##########
    clamping = RigidConnection(system.origin, cantilever, xi2=0)
    system.add(cantilever, clamping)

    ###############
    # applied loads
    ###############
    if load_type == "force":
        # spatially fixed load at cantilever tip
        P = lambda t: material_model.Fi[2] * (10 * t) / length**2
        F = lambda t: np.array([0.0, -P(t), 0.0])
        force = Force(F, cantilever, 1)
        system.add(force)
    elif load_type == "force+moment":
        P = lambda t: material_model.Fi[2] * (10 * t) / length**2
        F = lambda t: np.array([0.0, -P(t), 0.0])
        M = lambda t: np.array([0.0, 0.0, 2.5 * P(t)])
        force = Force(F, cantilever, 1)
        moment = B_Moment(M, cantilever, 1)
        system.add(force, moment)
    else:
        raise NotImplementedError

    # assemble system
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    ############
    # simulation
    ############
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_max_iter=30, newton_atol=1e-12),  # rtol=0
    )
    sol = solver.solve()

    # read solution
    t = sol.t
    q = sol.q

    #################
    # post-processing
    #################

    # VTK export
    dir_name = Path(__file__).parent
    if VTK_export:
        system.export(dir_name, f"vtk/{load_type}/{save_name}", sol)

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        [cantilever],
        scale=length,
        scale_di=0.05,
        show=False,
        n_frames=cantilever.nelement + 1,
        repeat=True,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    Y_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)

    path = Path(__file__)

    if True:
        # add normalized force-displacement diagram
        fig2, ax2 = plt.subplots()

        r_x_by_L = np.zeros(len(t))
        r_y_by_L = np.zeros(len(t))

        for i in range(len(t)):
            r_OP_L = cantilever.nodes(q[i])[:, -1]
            r_x_by_L[i] = r_OP_L[0] / length
            r_y_by_L[i] = -r_OP_L[1] / length

        # Kirchhoff rod
        ax2.plot(10 * t, r_x_by_L, "-", color="k", label=f"$r_x(1) / L$")
        ax2.plot(10 * t, r_y_by_L, "--", color="k", label=f"$-r_y(1) / L$")

        fig2.suptitle(f"Tip displacement {plot_name}")
        ax2.set_xlabel(r"$\alpha^2$")
        ax2.set_ylabel("Tip displacement")
        ax2.legend()
        ax2.grid()

        # plot animation
        ax1.azim = -90
        ax1.elev = 72

    ###########
    # Exports #
    ###########
    centerline_header = "xi, x, y, z"
    if save_centerlines:
        path_centerline = Path(dir_name, "csv", load_type, "centerline", save_name)
        path_centerline.mkdir(parents=True, exist_ok=True)
    xyz = np.zeros((3, n_load_steps + 1))
    xis = np.linspace(0, 1, num=201)
    for i in range(len(t)):
        points = cantilever.centerline(sol.q[i], num=len(xis))
        xyz[:, i] = points[:, -1]
        if save_centerlines:
            np.savetxt(
                path_centerline / f"{i}_of_{n_load_steps}.csv",
                np.vstack((xis, points)).T,
                delimiter=", ",
                header=centerline_header,
                comments="",
            )

    if save_tip_displacement:
        tip_header = "time, x, y, z"
        path_tip = Path(dir_name, "csv", load_type, "tip")
        path_tip.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            path_tip / f"{save_name}.csv",
            np.vstack((t, xyz)).T,
            delimiter=", ",
            header=tip_header,
            comments="",
        )

    if show_plots:
        plt.show()


if __name__ == "__main__":
    constraints = None  # Cosserat
    constraints = [1, 2]  # Euler-Bernoulli
    constraints = [0, 1, 2]  # inextensible Euler-Bernoulli

    load_type = "force"
    load_type = "force+moment"

    Rod = make_CosseratRod(
        interpolation="Quaternion",
        mixed=True,
        polynomial_degree=2,
        reduced_integration=True,
        constraints=constraints,
    )
    cantilever(
        Rod,
        Simo1986,
        nelements=4,
        n_load_steps=40,
        load_type=load_type,
        show_plots=True,
        name=f"{load_type}, constraints: {constraints}",
    )
