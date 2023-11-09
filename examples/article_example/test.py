from cardillo.beams import (
    RectangularCrossSection,
    Simo1986,
    Harsch2021,
)
from cardillo.math import Log_SO3, norm, Log_SE3, SE3, SE3inv
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.beams import (
    K_SE3_PetrovGalerkin_Quaternion,
    K_R3_SO3_PetrovGalerkin_Quaternion,
    K_R12_PetrovGalerkin_Quaternion,
    HigherOrder_K_R3xSO3_PetrovGalerkin_Quaternion,
    HigherOrder_K_SE3_PetrovGalerkin_Quaternion,
)

from cardillo.beams.cosseratRodPGMixed import (
    CosseratRodPG_R12Mixed,
    CosseratRodPG_QuatMixed,
    CosseratRodPG_SE3Mixed,
)

from cardillo.forces import K_Moment, Force
from cardillo import System
from cardillo.solver import Newton

from cardillo.math import e1, e2, e3, A_IK_basic

from cardillo.visualization import Export

from math import pi
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shutil import rmtree


def make_SE3_rod_mixed(reduced_integration):
    def make(
        L,
        r_OP0,
        A_IK0,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
    ):
        q0 = CosseratRodPG_SE3Mixed.straight_configuration(
            nelements,
            L,
            polynomial_degree=1,
            r_OP=r_OP0,
            A_IK=A_IK0,
            mixed=True,
        )
        rod = CosseratRodPG_SE3Mixed(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            nelements,
            Q=q0,
            q0=q0,
            polynomial_degree=1,
            reduced_integration=reduced_integration,
            mixed=True,
        )
        return q0, rod

    return make


def make_R12_rod_mixed(polynomial_degree):
    def make(
        L,
        r_OP0,
        A_IK0,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
    ):
        q0 = CosseratRodPG_R12Mixed.straight_configuration(
            nelements,
            L,
            polynomial_degree=polynomial_degree,
            r_OP=r_OP0,
            A_IK=A_IK0,
            mixed=True,
        )

        rod = CosseratRodPG_R12Mixed(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            nelements,
            Q=q0,
            q0=q0,
            polynomial_degree=polynomial_degree,
            reduced_integration=False,
            mixed=True,
        )

        return q0, rod

    return make


def make_SE3_rod(reduced_integration):
    def make(
        L,
        r_OP0,
        A_IK0,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
    ):
        q0 = CosseratRodPG_SE3Mixed.straight_configuration(
            nelements,
            L,
            polynomial_degree=1,
            r_OP=r_OP0,
            A_IK=A_IK0,
            mixed=False,
        )
        rod = CosseratRodPG_SE3Mixed(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            nelements,
            Q=q0,
            q0=q0,
            polynomial_degree=1,
            reduced_integration=reduced_integration,
            mixed=False,
        )
        return q0, rod

    return make


def make_R12_rod(polynomial_degree, reduced_integration):
    def make(
        L,
        r_OP0,
        A_IK0,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
    ):
        q0 = CosseratRodPG_R12Mixed.straight_configuration(
            nelements,
            L,
            polynomial_degree=polynomial_degree,
            r_OP=r_OP0,
            A_IK=A_IK0,
            mixed=False,
        )

        rod = CosseratRodPG_R12Mixed(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            nelements,
            Q=q0,
            q0=q0,
            polynomial_degree=polynomial_degree,
            reduced_integration=reduced_integration,
            mixed=False,
        )

        return q0, rod

    return make


# length
L = 2 * pi

# slendernesses = [1e1, 1e2, 1e3, 1e4]
# atols = [1e-2, 1e-6, 1e-10, 1e-13]
# f_vect = [6 * 1e6, 6 * 1e2, 6 * 1e-2, 6 * 1e-6]
slendernesses = [1.0e2]
atols = [1e-8]

n_load_step = 10

# starting point and orientation of initial point, initial length
r_OP0 = np.zeros(3, dtype=float)
A_IK0 = np.eye(3, dtype=float)

# define reference rod
reference_rod = "SE3_Mixed"
# reference_rod = "R12p2_Mixed"

test_rods = ["R12p1_Mixed"]
# test_rods = ["R12p1", "R12p2", "SE3"]
# test_rods = ["R12p1", "R12p1_Mixed"]
# test_rods = ["SE3_Mixed", "SE3"]
# test_rods = ["R12p1", "R12p1_Mixed", "R12p2", "R12p2_Mixed"]
# test_rods = ["R12p1", "R12p1_Mixed", "R12p2", "R12p2_Mixed", "SE3_Mixed", "SE3"]

nnodes_list = np.array([5, 9, 17, 33, 65, 129], dtype=int)
nnodes_ref = 513
# nnodes_list = np.array([5, 9, 17], dtype=int)
# nnodes_ref = 513

# reduced_integration = False
reduced_integration = False

rod_hypothesis_penalty = "shear_deformable"
# rod_hypothesis_penalty = "shear_rigid"
# rod_hypothesis_penalty = "inextensible_shear_rigid"


def convergence():
    n_rods = len(test_rods)
    n_slenderness = len(slendernesses)
    nnodes = len(nnodes_list)
    rods = np.zeros((n_slenderness, n_rods), dtype=object)
    sols = np.zeros((n_slenderness, n_rods), dtype=object)
    position_errors = np.zeros((n_slenderness, n_rods, nnodes), dtype=float)
    rotation_errors = np.zeros((n_slenderness, n_rods, nnodes), dtype=float)
    twist_errors = np.zeros((n_slenderness, n_rods, nnodes), dtype=float)
    Newton_iterations = np.zeros((n_slenderness, n_rods, nnodes), dtype=float)

    for i in range(n_slenderness):
        slenderness = slendernesses[i]
        atol = atols[i]

        # used cross section
        width = L / slenderness

        # cross section and quadratic beam material
        line_density = 1
        cross_section = RectangularCrossSection(line_density, width, width)
        A_rho0 = line_density * cross_section.area
        K_S_rho0 = line_density * cross_section.first_moment
        K_I_rho0 = line_density * cross_section.second_moment

        # material model
        if rod_hypothesis_penalty == "shear_deformable":
            Ei = np.array([5, 1, 1])
        elif rod_hypothesis_penalty == "shear_rigid":
            Ei = np.array([5, 1e3, 1e3])
        elif rod_hypothesis_penalty == "inextensible_shear_rigid":
            Ei = np.array([5, 1, 1]) * 1e4

        Fi = np.array([0.5, 2, 2])

        material_model = Simo1986(Ei, Fi)

        def solve(nnodes, rod, reduced_integration):
            assert nnodes % 2 == 1
            if rod == "SE3_Mixed":
                nelements = nnodes - 1
                make_rod = make_SE3_rod_mixed(False)
            elif rod == "SE3":
                nelements = nnodes - 1
                make_rod = make_SE3_rod(reduced_integration)
            elif rod == "R12p1_Mixed":
                nelements = nnodes - 1
                make_rod = make_R12_rod_mixed(1)
            elif rod == "R12p2_Mixed":
                nelements = int((nnodes - 1) / 2)
                make_rod = make_R12_rod_mixed(2)
            elif rod == "R12p1":
                nelements = nnodes - 1
                make_rod = make_R12_rod(
                    1,
                    reduced_integration=reduced_integration,
                )
            elif rod == "R12p2":
                nelements = int((nnodes - 1) / 2)
                make_rod = make_R12_rod(
                    2,
                    reduced_integration=reduced_integration,
                )
            else:
                print(f"wrong rod: '{rod}'")
                raise NotImplementedError

            q0, rod = make_rod(
                L,
                r_OP0,
                A_IK0,
                cross_section,
                material_model,
                A_rho0,
                K_S_rho0,
                K_I_rho0,
                nelements,
            )

            print(f" - rod.nnode: {rod.nnodes_r}")
            print(f" - rod.nelement: {rod.nelement}")

            # creation of the system
            system = System()

            # clamping constrain
            clamping_left = RigidConnection(system.origin, rod, frame_ID2=(0,))

            # spatially fixed load at cantilever tip
            P = lambda t: material_model.Fi[2] * (10 * t) / L**2
            F = lambda t: -P(t) * e2
            force = Force(F, rod, frame_ID=(1,))

            M = lambda t: 2.5 * P(t) * e3
            moment = K_Moment(M, rod, (1,))

            # assemble the system
            system.add(rod)
            system.add(clamping_left)
            system.add(force)
            system.add(moment)
            system.assemble()

            solver = Newton(
                system,
                n_load_steps=n_load_step + 1,
                max_iter=30,
                atol=atol,
            )

            sol = solver.solve()
            n_iter_tot = sol.n_iter_tot

            return rod, sol, n_iter_tot

        # solve system with reference rod
        rod_ref, sol_ref, _ = solve(
            nnodes_ref, reference_rod, reduced_integration=False
        )
        """ Be careful to set the reduced_integration to False when we use mixed elements"""

        # sample centerline deflection of reference solution
        num = 100
        xis = np.linspace(0, 1, num=num)
        r_OP_ref = rod_ref.centerline(sol_ref.q[-1], num=num)
        A_IK_ref = np.array(rod_ref.frames(sol_ref.q[-1], num=num)[1:])

        for j, rod_name in enumerate(test_rods):
            print(f"rod: {rod_name}")
            for k, nnodes in enumerate(nnodes_list):
                rod, sol, n_iter_tot = solve(
                    nnodes,
                    rod_name,
                    reduced_integration=reduced_integration,
                )

                # Newton iteration number
                Newton_iterations[i, j, k] = n_iter_tot

                # centerline errors
                r_OPj = rod.centerline(sol.q[-1], num=num)
                diff = r_OPj - r_OP_ref
                error = np.sqrt(sum([d @ d for d in diff])) / num
                position_errors[i, j, k] = error

                # rotation errors
                A_IKj = np.array(rod.frames(sol.q[-1], num=num)[1:])
                diff = []
                for l in range(num):
                    diff.append(Log_SO3(A_IKj[:, :, l].T @ A_IK_ref[:, :, l]))
                diff = np.array(diff)
                error = np.sqrt(sum([d @ d for d in diff]))
                rotation_errors[i, j, k] = error

                # twist errors
                diff = []
                for l in range(num):
                    H_IK = SE3(A_IKj[:, :, l], r_OPj[:, l])
                    H_IK_ref = SE3(A_IK_ref[:, :, l], r_OP_ref[:, l])
                    diff.append(Log_SE3(SE3inv(H_IK) @ H_IK_ref))
                diff = np.array(diff)
                error = np.sqrt(sum([d @ d for d in diff]))
                twist_errors[i, j, k] = error

            rods[i, j] = rod
            sols[i, j] = sol

    print(f"position_errors: {position_errors}")
    print(f"rotation_errors: {rotation_errors}")
    print(f"twist_errors: {twist_errors}")
    print(f"total_number_Newton_iteration: {Newton_iterations}")

    ###############
    # export errors
    ###############
    path = Path(__file__)
    path = Path(path.parent / path.stem)
    if path.exists():
        rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    for i in range(n_slenderness):
        for j, rod_name in enumerate(test_rods):
            header = "nnodes, position_error, rotation_errors, twist_errors, total_number_Newton_iteration"
            export_data = np.vstack(
                [
                    nnodes_list,
                    position_errors[i, j],
                    rotation_errors[i, j],
                    twist_errors[i, j],
                    Newton_iterations[i, j],
                ]
            ).T
            np.savetxt(
                path
                / f"{reference_rod}_vs_{rod_name}_slenderness_{slendernesses[i]:1.0e}.txt",
                export_data,
                delimiter=", ",
                header=header,
                comments="",
            )

            # Each vectors is organized in this way: each row is related to a defined mesh for the type of element considered.
            # The row contains the position, angle and twist errors plus the total number of Newton's iterations

    ##########################
    # plot rate of convergence
    ##########################
    # for i in range(n_slenderness):
    #     fig, ax = plt.subplots(1, n_rods, sharey=True)
    #     fig.suptitle(f"slenderness: {slendernesses[i]}")
    #     for j, rod_name in enumerate(test_rods):
    #         ax[j].set_title(f"{rod_name}")
    #         # ax[j].loglog(nnodes_list, position_errors[i, j], "-ok", label="e_r^100")
    #         # ax[j].loglog(nnodes_list, rotation_errors[i, j], "-sk", label="e_psi^100")
    #         ax[j].loglog(nnodes_list, twist_errors[i, j], "-vk", label="e_theta^100")
    #         ax[j].loglog(nnodes_list, 90 / nnodes_list, "--k", label="~1 / n_el")
    #         ax[j].loglog(
    #             nnodes_list, 90 / nnodes_list**2, "-.k", label="~1 / nnodes^2"
    #         )
    #         ax[j].loglog(
    #             nnodes_list, 90 / nnodes_list**3, "-.k", label="~1 / nnodes^3"
    #         )
    #         ax[j].loglog(
    #             nnodes_list, 90 / nnodes_list**4, "-.k", label="~1 / nnodes^4"
    #         )
    #         ax[j].grid()
    #         # ax[j].legend()
    #         # ax[j].legend(loc='upper right')  # Imposta la posizione della legenda

    # Define a list of markers
    # markers = ['o', 's', '^', 'v', '<', '>', 'p', 'h']
    markers = ["o", "s", "^"]
    # Define a list of line styles
    line_styles = ["-", "-", "-."]

    for i in range(n_slenderness):
        # Create a new figure for each slenderness value
        fig, ax = plt.subplots()

        for j, rod_name in enumerate(test_rods):
            marker = markers[(i * n_rods + j) % len(markers)]
            line_style = line_styles[i % len(line_styles)]
            # Plot the twist_errors for each combination of slenderness and rod_name with a line and marker style
            ax.loglog(
                nnodes_list,
                position_errors[i, j],
                line_style + marker,
                label=f"{slendernesses[i]}, {rod_name}",
                markerfacecolor="none",
            )

        # Plot the reference lines once
        ax.loglog(nnodes_list, 90 / nnodes_list, "--k", label="~1 / n_el")
        ax.loglog(nnodes_list, 90 / nnodes_list**2, "-.k", label="~1 / nnodes^2")
        ax.loglog(nnodes_list, 90 / nnodes_list**3, ":k", label="~1 / nnodes^3")

        ax.set_title(f"Position Errors for Slenderness: {slendernesses[i]}")
        ax.set_xlabel("nnodes_list")
        ax.set_ylabel("Position Errors")
        ax.grid()
        ax.legend(loc="lower left")

    # ax[0].legend(loc='lower left')

    # for i in range(n_slenderness):
    #     fig.savefig(f"C:/Users/Domenico/Desktop/cardillo/examples/mixed_examples/45_bent_beam_convergence/slenderness_{slendernesses[i]}.eps", format='eps', bbox_inches='tight')

    plt.show()

    ####################################################
    # visualize centerline curves of final configuration
    ####################################################
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="3d"))
    ax.plot(*rod_ref.centerline(sol_ref.q[-1]), "-k", label="reference")
    styles = ["--r", "--g", "--b", "--m"]
    # for j, (rod, sol) in enumerate(zip(rods, sols)):
    #     ax.plot(*rod.centerline(sol.q[-1]), styles[j], label=test_rods[j])
    ax.legend()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim3d(left=-L, right=L)
    ax.set_ylim3d(bottom=-L, top=L)
    ax.set_zlim3d(bottom=-L, top=L)

    plt.show()
    # #################
    # # strain measures
    # #################
    # def stress_strain(rod, sol, nxi=1000):
    #     xis = np.linspace(0, 1, num=nxi)

    #     K_Gamma_bar = np.zeros((3, nxi))
    #     K_Kappa_bar = np.zeros((3, nxi))
    #     K_Gamma = np.zeros((3, nxi))
    #     K_Kappa = np.zeros((3, nxi))
    #     K_n = np.zeros((3, nxi))
    #     K_m = np.zeros((3, nxi))
    #     I_n = np.zeros((3, nxi))
    #     I_m = np.zeros((3, nxi))
    #     for i in range(nxi):
    #         frame_ID = (xis[i],)
    #         elDOF = rod.local_qDOF_P(frame_ID)

    #         # length of reference tangent vector
    #         Qe = rod.Q[elDOF]
    #         _, _, K_Gamma_bar0, K_Kappa_bar0 = rod._eval(Qe, xis[i])
    #         J = norm(K_Gamma_bar0)

    #         # current strain measures
    #         qe = sol.q[-1, rod.qDOF][elDOF]
    #         _, A_IK_i, K_Gamma_bar_i, K_Kappa_bar_i = rod._eval(qe, xis[i])

    #         K_Gamma_bar[:, i] = K_Gamma_bar_i
    #         K_Kappa_bar[:, i] = K_Kappa_bar_i
    #         K_Gamma[:, i] = K_Gamma_bar_i / J
    #         K_Kappa[:, i] = K_Kappa_bar_i / J
    #         K_n[:, i] = material_model.K_n(
    #             K_Gamma_bar_i / J, K_Gamma_bar0 / J, K_Kappa_bar_i / J, K_Kappa_bar0 / J
    #         )
    #         K_m[:, i] = material_model.K_m(
    #             K_Gamma_bar_i / J, K_Gamma_bar0 / J, K_Kappa_bar_i / J, K_Kappa_bar0 / J
    #         )
    #         I_n[:, i] = A_IK_i @ K_n[:, i]
    #         I_m[:, i] = A_IK_i @ K_m[:, i]

    #     return xis, K_Gamma_bar, K_Gamma, K_Kappa_bar, K_Kappa, K_n, K_m, I_n, I_m

    # fig, ax = plt.subplots(1, 4)

    # xis, K_Gamma_bar, K_Gamma, K_Kappa_bar, K_Kappa, K_n, K_m, I_n, I_m = stress_strain(
    #     rod_ref, sol_ref
    # )
    # header = "xi, K_Gamma1_minus_1, K_Gamma2, K_Gamma3, K_Kappa1, K_Kappa2, K_Kappa3, K_n1, K_n2, K_n3, K_m1, K_m2, K_m3"
    # export_data = np.vstack(
    #     [xis, K_Gamma[0] - 1.0, K_Gamma[1], K_Gamma[2], *K_Kappa, *K_n, *K_m]
    # ).T
    # np.savetxt(
    #     f"StrainMeasuresConvergence_Reference_{reference_rod}.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    # ax[0].set_title("K_Gamma")
    # ax[0].plot(K_Gamma[0], label="K_Gamma0 - reference")
    # ax[0].plot(K_Gamma[1], label="K_Gamma1 - reference")
    # ax[0].plot(K_Gamma[2], label="K_Gamma2 - reference")

    # ax[1].set_title("K_Gamma")
    # ax[1].plot(K_Kappa[0], label="K_Kappa0 - reference")
    # ax[1].plot(K_Kappa[1], label="K_Kappa1 - reference")
    # ax[1].plot(K_Kappa[2], label="K_Kappa2 - reference")

    # ax[2].set_title("K_n")
    # ax[2].plot(K_n[0], label="K_n0 - reference")
    # ax[2].plot(K_n[1], label="K_n1 - reference")
    # ax[2].plot(K_n[2], label="K_n2 - reference")

    # ax[3].set_title("K_m")
    # ax[3].plot(K_m[0], label="K_m0 - reference")
    # ax[3].plot(K_m[1], label="K_m1 - reference")
    # ax[3].plot(K_m[2], label="K_m2 - reference")

    # for j, (rod, sol) in enumerate(zip(rods, sols)):
    #     (
    #         xis,
    #         K_Gamma_bar,
    #         K_Gamma,
    #         K_Kappa_bar,
    #         K_Kappa,
    #         K_n,
    #         K_m,
    #         I_n,
    #         I_m,
    #     ) = stress_strain(rod, sol)
    #     export_data = np.vstack(
    #         [xis, K_Gamma[0] - 1.0, K_Gamma[1], K_Gamma[2], *K_Kappa, *K_n, *K_m]
    #     ).T
    #     np.savetxt(
    #         f"StrainMeasuresConvergence_{test_rods[j]}.txt",
    #         export_data,
    #         delimiter=", ",
    #         header=header,
    #         comments="",
    #     )

    #     ax[0].plot(K_Gamma[0], label=f"K_Gamma0 - {test_rods[j]}")
    #     ax[0].plot(K_Gamma[1], label=f"K_Gamma1 - {test_rods[j]}")
    #     ax[0].plot(K_Gamma[2], label=f"K_Gamma2 - {test_rods[j]}")

    #     ax[1].plot(K_Kappa[0], label=f"K_Kappa0 - {test_rods[j]}")
    #     ax[1].plot(K_Kappa[1], label=f"K_Kappa1 - {test_rods[j]}")
    #     ax[1].plot(K_Kappa[2], label=f"K_Kappa2 - {test_rods[j]}")

    #     ax[2].plot(K_n[0], label=f"K_n0 - {test_rods[j]}")
    #     ax[2].plot(K_n[1], label=f"K_n1 - {test_rods[j]}")
    #     ax[2].plot(K_n[2], label=f"K_n2 - {test_rods[j]}")

    #     ax[3].plot(K_m[0], label=f"K_m0 - {test_rods[j]}")
    #     ax[3].plot(K_m[1], label=f"K_m1 - {test_rods[j]}")
    #     ax[3].plot(K_m[2], label=f"K_m2 - {test_rods[j]}")

    # ax[0].grid()
    # ax[0].legend()
    # ax[1].grid()
    # ax[1].legend()

    # plt.show()


if __name__ == "__main__":
    convergence()
