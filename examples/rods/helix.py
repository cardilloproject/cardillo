from cardillo.math import e1, e2, e3
from cardillo.rods import (
    CircularCrossSection,
    Simo1986,
)
from cardillo.constraints import RigidConnection
from cardillo.rods import animate_beam
from cardillo.rods.cosseratRod import (
    make_CosseratRod_SE3,
    make_CosseratRod_Quat,
    make_CosseratRod_R12,
    make_CosseratRod_R3SO3,
)

from cardillo.forces import K_Moment
from cardillo import System
from cardillo.solver import Newton, SolverOptions
from cardillo.visualization import Export

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import pi

def helix(
    Rod,
    nelements=10,
    polynomial_degree=2,
    n_load_steps=10,
    reduced_integration=True,
    VTK_export=False,
    slenderness=1.0e4,
    atol=1.0e-13,
    export_name="rod",
):
    # geometry of the rod
    n = 2  # number of coils
    R0 = 10  # radius of the helix
    h = 50  # height of the helix
    c = h / (2 * R0 * pi * n)  # pitch of the helix
    length = np.sqrt(1 + c**2) * R0 * 2 * pi * n
    cc = 1 / (np.sqrt(1 + c**2))

    alpha = lambda xi: 2 * pi * n * xi
    alpha_xi = 2 * pi * n

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
    material_model = Simo1986(Ei, Fi)

    # construct system
    system = System()

    alpha_0 = alpha(0)

    r_OP0 = R0 * np.array([np.sin(alpha_0), -np.cos(alpha_0), c * alpha_0])

    e_x = cc * np.array([np.cos(alpha_0), np.sin(alpha_0), c])
    e_y = np.array([-np.sin(alpha_0), np.cos(alpha_0), 0])
    e_z = cc * np.array([-c * np.cos(alpha_0), -c * np.sin(alpha_0), 1])

    A_IK0 = np.vstack((e_x, e_y, e_z))
    A_IK0 = A_IK0.T

    q0 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        r_OP=r_OP0,
        A_IK=A_IK0,
    )
    cantilever = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        reduced_integration=reduced_integration,
    )

    # clamping left
    clamping_left = RigidConnection(system.origin, cantilever, frame_ID2=(0,))
    system.add(cantilever)
    system.add(clamping_left)

    # moment at right end
    Fi = material_model.Fi
    M = (
        lambda t: (R0 * alpha_xi**2)
        / (length**2)
        * (c * e1 * Fi[0] + e3 * Fi[2])
        * t * 0.5
    )
    moment = K_Moment(M, cantilever, (1,))
    system.add(moment)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # add Newton solver
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_max_iter=30, newton_atol=atol)
    )

    # solve nonlinear static equilibrium equations
    sol = solver.solve()

    # extract solutions
    q = sol.q
    la_c = sol.la_c
    nt = len(q)
    t = sol.t[:nt]

    # VTK export
    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 30, sol)
        e.export_contr(
            cantilever,
            level="centerline + directors",
            num=3 * nelements,
            file_name="cantilever_curve",
        )
        e.export_contr(
            cantilever,
            continuity="C0",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
            file_name="cantilever_volume",
        )

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        [cantilever],
        scale=length,
        show=False,
        repeat=False,
    )

    plt.show()

    path = Path(__file__)
    path = Path(path.parent / path.stem)
    path.mkdir(parents=True, exist_ok=True)

    def stress_strain(rod, sol, nxi=100):
        xis = np.linspace(0, 1, num=nxi)

        Delta_K_Gamma = np.zeros((3, nxi))
        Delta_K_Kappa = np.zeros((3, nxi))
        K_n = np.zeros((3, nxi))
        K_m = np.zeros((3, nxi))

        for i, xii in enumerate(xis):
                (
                    K_n[:, i],
                    K_m[:, i]
                ) = rod.eval_stresses(sol.t[-1], sol.q[-1], sol.la_c[-1], xii)
                (
                    Delta_K_Gamma[:, i],
                    Delta_K_Kappa[:, i]
                ) = rod.eval_strains(sol.t[-1], sol.q[-1], sol.la_c[-1], xii)

        return xis, Delta_K_Gamma, Delta_K_Kappa, K_n, K_m
        
    fig, ax = plt.subplots(1, 4)

    xis, K_Gamma, K_Kappa, K_n, K_m = stress_strain(cantilever, sol)
    header = "xi, K_Gamma1, K_Gamma2, K_Gamma3, K_Kappa1, K_Kappa2, K_Kappa3, \
            K_n1, K_n2, K_n3, K_m1, K_m2, K_m3"
    export_data = np.vstack(
        [xis, *K_Gamma, *K_Kappa, *K_n, *K_m]
    ).T

    np.savetxt(
        path / f"strain_stress_helix_{export_name}.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

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
    #############################
    # robust mixed formulations #
    #############################

    # helix(
    #     Rod=make_CosseratRod_SE3(mixed=True),
    #     nelements=10,
    #     n_load_steps=1,
    #     reduced_integration=True,
    #     slenderness=1.0e4,
    #     atol=1.0e-12,
    # )

    # helix(
    #     Rod=make_CosseratRod_Quat(mixed=True),
    #     nelements=10,
    #     polynomial_degree=2,
    #     n_load_steps=1,
    #     reduced_integration=True,
    #     slenderness=1.0e4,
    #     atol=1.0e-12,
    # )

    # helix(
    #     Rod=make_CosseratRod_R12(mixed=True),
    #     nelements=10,
    #     polynomial_degree=2,
    #     n_load_steps=2,
    #     reduced_integration=True,
    #     slenderness=1.0e4,
    #     atol=1.0e-12,
    # )


    # #######################
    # # paramters that work #
    # #######################

    # helix(Rod=make_CosseratRod_SE3(mixed=False), nelements=10, n_load_steps = 1, reduced_integration=True, slenderness=1.0e1, atol=1.0e-8, export_name="SE3_DB")

    # helix(Rod=make_CosseratRod_SE3(mixed=False), nelements=5, polynomial_degree=1, n_load_steps = 109, reduced_integration=True, slenderness=1.0e2, atol=1.0e-9)
    # # strangely: n_load_steps = 110 does not work anymore

    # helix(Rod=make_CosseratRod_SE3(mixed=False), nelements=5, polynomial_degree=1, n_load_steps = 200, reduced_integration=True, slenderness=1.0e3, atol=1.0e-12)

    # helix(Rod=make_CosseratRod_SE3(mixed=False), nelements=5, polynomial_degree=1, n_load_steps = 700, reduced_integration=True, slenderness=1.0e4, atol=1.0e-14)

    # Quaternion-interpolation:
    helix(Rod=make_CosseratRod_Quat(mixed=False), nelements=10, polynomial_degree=1, n_load_steps = 10, reduced_integration=True, slenderness=1.0e1, atol=1.0e-8)

    # helix(Rod=make_CosseratRod_R3SO3(mixed=False), nelements=10, polynomial_degree=1, n_load_steps = 10, reduced_integration=True, slenderness=1.0e1, atol=1.0e-8)


    # helix(Rod=make_CosseratRod_Quat(mixed=False), nelements=10, polynomial_degree=2, n_load_steps = 119, reduced_integration=True, slenderness=1.0e2, atol=1.0e-9)

    # helix(Rod=make_CosseratRod_Quat(mixed=False), nelements=10, polynomial_degree=2, n_load_steps = 200, reduced_integration=True, slenderness=1.0e3, atol=1.0e-12)

    # helix(Rod=make_CosseratRod_Quat(mixed=False), nelements=10, polynomial_degree=2, n_load_steps = 700, reduced_integration=True, slenderness=1.0e4, atol=1.0e-14)

    # R12 interpolation
    # helix(Rod=make_CosseratRod_R12(mixed=False), nelements=20, polynomial_degree=2, n_load_steps = 100, reduced_integration=True, slenderness=1.0e1, atol=1.0e-8)

    # helix(Rod=make_CosseratRod_R12(mixed=False), nelements=10, polynomial_degree=2, n_load_steps = 129, reduced_integration=True, slenderness=1.0e2, atol=1.0e-9)

    # helix(Rod=make_CosseratRod_R12(mixed=False), nelements=10, polynomial_degree=2, n_load_steps = 200, reduced_integration=True, slenderness=1.0e3, atol=1.0e-12)

    # helix(Rod=make_CosseratRod_R12(mixed=False), nelements=10, polynomial_degree=2, n_load_steps = 700, reduced_integration=True, slenderness=1.0e4, atol=1.0e-14)
