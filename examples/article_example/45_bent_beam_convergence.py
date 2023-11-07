from cardillo.beams import (
    RectangularCrossSection,
    Simo1986,
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
        R,
        curve,
        dcurve,
        ddcurve,
        angle,
        r_OP0,
        A_IK0,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
    ):
        q0 = CosseratRodPG_SE3Mixed.deformed_configuration(
            nelements,
            curve,
            dcurve,
            ddcurve,
            angle,
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
            mixed=True
        )
        return q0, rod

    return make

def make_R12_rod_mixed(polynomial_degree):
    def make(
        R,
        curve,
        dcurve,
        ddcurve,
        angle,
        r_OP0,
        A_IK0,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
    ):
        q0 = CosseratRodPG_R12Mixed.deformed_configuration(
            nelements,
            curve,
            dcurve,
            ddcurve,
            angle,
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
        R,
        curve,
        dcurve,
        ddcurve,
        angle,
        r_OP0,
        A_IK0,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
    ):
        q0 = CosseratRodPG_SE3Mixed.deformed_configuration(
            nelements,
            curve,
            dcurve,
            ddcurve,
            angle,
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
            mixed=False
        )
        return q0, rod

    return make

def make_R12_rod(polynomial_degree, reduced_integration):
    def make(
        R,
        curve,
        dcurve,
        ddcurve,
        angle,
        r_OP0,
        A_IK0,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
    ):
        q0 = CosseratRodPG_R12Mixed.deformed_configuration(
            nelements,
            curve,
            dcurve,
            ddcurve,
            angle,
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

# Young's and shear modulus
E = 1e7  # Meier2015
G = 5e6  # Meier2015

L = 2 * pi * 100 / 8
R = 100
angle = pi / 4


# slendernesses = [1e1, 1e2, 1e3, 1e4]
# atols = [1e-2, 1e-6, 1e-10, 1e-13]
# f_vect = [6 * 1e6, 6 * 1e2, 6 * 1e-2, 6 * 1e-6]
slendernesses = [1e1]
atols = [1e-2]
f_vect = [6 * 1e6]

# starting point and orientation of initial point, initial length
r_OP0 = np.zeros(3, dtype=float)
A_IK0 = np.eye(3, dtype=float)

# parametric law
curve = lambda xi: np.array([R - R * np.cos(xi), R * np.sin(xi), 0])
dcurve = lambda xi: np.array([R * np.sin(xi), R * np.cos(xi), 0])
ddcurve = lambda xi: np.array([R * np.cos(xi), -R * np.sin(xi), 0])

# define reference rod
# reference_rod = "SE3_Mixed"
reference_rod = "R12p2_Mixed"

test_rods = ["R12p2_Mixed", "R12p1"]
# test_rods = ["R12p1", "R12p2", "SE3"]
# test_rods = ["R12p1", "R12p1_Mixed"]
# test_rods = ["SE3_Mixed", "SE3"]
# test_rods = ["R12p1", "R12p1_Mixed", "R12p2", "R12p2_Mixed"]
# test_rods = ["R12p1", "R12p1_Mixed", "R12p2", "R12p2_Mixed", "SE3_Mixed", "SE3"]

# nnodes_list = np.array([5, 9, 17, 33, 65, 129], dtype=int) 
# nnodes_ref = 513  
nnodes_list = np.array([5], dtype=int) 
nnodes_ref = 9                                   

volume_correction = False
# volume_correction = True

# reduced_integration = False
reduced_integration = False


def convergence(): 
    n_rods = len(test_rods)  
    n_slenderness = len(slendernesses)
    nnodes = len(nnodes_list) 
    rods = np.zeros((n_slenderness, n_rods), dtype=object)
    sols = np.zeros((n_slenderness, n_rods), dtype=object)
    position_errors = np.zeros((n_slenderness, n_rods, nnodes), dtype=float)
    rotation_errors = np.zeros((n_slenderness, n_rods, nnodes), dtype=float)
    twist_errors = np.zeros((n_slenderness, n_rods, nnodes), dtype=float)
    Newton_iterations = np.zeros((n_slenderness, n_rods, nnodes), dtype=int)

    for i in range(n_slenderness):
        slenderness = slendernesses[i]
        atol = atols[i]
        ff = f_vect[i]

        # used cross section
        width = R / slenderness

        # cross section and quadratic beam material
        line_density = 1
        cross_section = RectangularCrossSection(line_density, width, width)
        A_rho0 = line_density * cross_section.area
        K_S_rho0 = line_density * cross_section.first_moment
        K_I_rho0 = line_density * cross_section.second_moment
        A = cross_section.area
        Ip, I2, I3 = np.diag(cross_section.second_moment)
        Ei = np.array([E * A, G * A, G * A])
        Fi = np.array([G * Ip, E * I2, E * I3])
        material_model = Simo1986(Ei, Fi)


        def solve(nnodes, rod, volume_correction, reduced_integration):
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
                R,
                curve,
                dcurve,
                ddcurve,
                angle,
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

            # junctions
            frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)

            # generate the constraint on the beam
            A_IK_clamping= lambda t: A_IK_basic(0.).z()
            clamping_point = Frame(A_IK=A_IK_clamping)
            clamping_left = RigidConnection(clamping_point, rod, frame_ID2=(0,))
            
            F = lambda t: ff * t * e3
            force = Force(F, rod, frame_ID=(1,))
            
            # assemble the system
            system = System()
            system.add(rod)
            system.add(clamping_point)
            system.add(clamping_left)
            system.add(force)
            system.assemble()

            solver = Newton(
                system,
                n_load_steps=11,
                max_iter=30,
                atol=atol,
            )

            sol = solver.solve()
            n_iter_tot = sol.n_iter_tot

            return rod, sol, n_iter_tot

        # solve system with reference rod
        rod_ref, sol_ref, _ = solve(
            nnodes_ref, reference_rod, volume_correction=False, reduced_integration=False)
        ''' Be careful to set the reduced_integration to False when we use mixed elements'''

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
                    volume_correction=volume_correction,
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
    markers = ['o', 's', '^']
    # Define a list of line styles
    line_styles = ['-', '-', '-.']

    for i in range(n_slenderness):
        # Create a new figure for each slenderness value
        fig, ax = plt.subplots()

        for j, rod_name in enumerate(test_rods):
            marker = markers[(i * n_rods + j) % len(markers)]
            line_style = line_styles[i % len(line_styles)]
            # Plot the twist_errors for each combination of slenderness and rod_name with a line and marker style
            ax.loglog(nnodes_list, twist_errors[i, j], line_style + marker, label=f"{slendernesses[i]}, {rod_name}", markerfacecolor='none')

        # Plot the reference lines once
        ax.loglog(nnodes_list, 90 / nnodes_list, "--k", label="~1 / n_el")
        ax.loglog(nnodes_list, 90 / nnodes_list**2, "-.k", label="~1 / nnodes^2")
        ax.loglog(nnodes_list, 90 / nnodes_list**3, ":k", label="~1 / nnodes^3")

        ax.set_title(f"Twist Errors for Slenderness: {slendernesses[i]}")
        ax.set_xlabel("nnodes_list")
        ax.set_ylabel("Twist Errors")
        ax.grid()
        ax.legend(loc='lower left')


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


    #################
    # strain measures
    #################
    def stress_strain(rod, sol, nxi=1000):
        xis = np.linspace(0, 1, num=nxi)

        K_Gamma = np.zeros((3, nxi))
        K_Kappa = np.zeros((3, nxi))
        K_n = np.zeros((3, nxi))
        K_m = np.zeros((3, nxi))
        
        if rod.mixed is True:
            K_Gamma_DB_M = np.zeros((3, nxi))
            K_Kappa_DB_M = np.zeros((3, nxi))
            K_n_DB_M = np.zeros((3, nxi))
            K_m_DB_M = np.zeros((3, nxi))
            
            for i, xii in enumerate(xis):
                K_n[:, i], K_m[:, i], K_n_DB_M[:, i], K_m_DB_M[:, i] = rod.eval_stresses(sol.t[-1],sol.q[-1], xii, mixed=rod.mixed)
                K_Gamma[:, i], K_Kappa[:, i], K_Gamma_DB_M[:, i], K_Kappa_DB_M[:, i] = rod.eval_strains(sol.t[-1],sol.q[-1], xii, mixed=rod.mixed)
        
            return xis, K_Gamma, K_Kappa, K_Gamma_DB_M, K_Kappa_DB_M, K_n, K_m, K_n_DB_M, K_m_DB_M
        
        else:
            for i, xii in enumerate(xis):
                K_n[:, i], K_m[:, i] = rod.eval_stresses(sol.t[-1],sol.q[-1], xii, mixed=rod.mixed)
                K_Gamma[:, i], K_Kappa[:, i] = rod.eval_strains(sol.t[-1],sol.q[-1], xii, mixed=rod.mixed)

            return xis, K_Gamma, K_Kappa, K_n, K_m


    fig, ax = plt.subplots(1, 4)

    if rod_ref.mixed:
        xis, K_Gamma, K_Kappa, K_Gamma_DB_M, K_Kappa_DB_M, K_n, K_m, K_n_DB_M, K_m_DB_M = stress_strain(
        rod_ref, sol_ref)

        header = "xi, K_Gamma1_minus_1, K_Gamma2, K_Gamma3, K_Kappa1, K_Kappa2, K_Kappa3, \
              K_Gamma1_minus_1_DB_M, K_Gamma2_DB_M, K_Gamma3_DB_M, K_Kappa1_DB_M, K_Kappa2_DB_M, K_Kappa3_DB_M, \
              K_n1, K_n2, K_n3, K_m1, K_m2, K_m3,\
              K_n1_DB_M, K_n2_DB_M, K_n3_DB_M, K_m1_DB_M, K_m2_DB_M, K_m3_DB_M"
        export_data = np.vstack(
            [xis, K_Gamma[0] - 1.0, K_Gamma[1], K_Gamma[2], *K_Kappa, 
             K_Gamma_DB_M[0] - 1.0, K_Gamma_DB_M[1], K_Gamma_DB_M[2], *K_Kappa_DB_M, *K_n, *K_m, *K_n_DB_M, *K_m_DB_M]
        ).T
    else: 
        xis, K_Gamma, K_Kappa, K_n, K_m = stress_strain(rod_ref, sol_ref)
        header = "xi, K_Gamma1_minus_1, K_Gamma2, K_Gamma3, K_Kappa1, K_Kappa2, K_Kappa3, \
                K_n1, K_n2, K_n3, K_m1, K_m2, K_m3"
        export_data = np.vstack(
            [xis, K_Gamma[0] - 1.0, K_Gamma[1], K_Gamma[2], *K_Kappa, *K_n, *K_m]
        ).T

    np.savetxt(
        f"StrainMeasuresConvergence_Reference_{reference_rod}.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    ax[0].set_title("K_Gamma")
    ax[0].plot(K_Gamma[0], label="K_Gamma0 - reference")
    ax[0].plot(K_Gamma[1], label="K_Gamma1 - reference")
    ax[0].plot(K_Gamma[2], label="K_Gamma2 - reference")

    ax[1].set_title("K_Kappa")
    ax[1].plot(K_Kappa[0], label="K_Kappa0 - reference")
    ax[1].plot(K_Kappa[1], label="K_Kappa1 - reference")
    ax[1].plot(K_Kappa[2], label="K_Kappa2 - reference")

    ax[2].set_title("K_n")
    ax[2].plot(K_n[0], label="K_n0 - reference")
    ax[2].plot(K_n[1], label="K_n1 - reference")
    ax[2].plot(K_n[2], label="K_n2 - reference")

    ax[3].set_title("K_m")
    ax[3].plot(K_m[0], label="K_m0 - reference")
    ax[3].plot(K_m[1], label="K_m1 - reference")
    ax[3].plot(K_m[2], label="K_m2 - reference")

    for j, (rod, sol) in enumerate(zip(rods, sols)):

        if rod.mixed is True:
            xis, K_Gamma, K_Kappa, K_Gamma_DB_M, K_Kappa_DB_M, K_n, K_m, K_n_DB_M, K_m_DB_M = stress_strain(rod, sol)
            export_data = np.vstack(
                [xis, K_Gamma[0] - 1.0, K_Gamma[1], K_Gamma[2], *K_Kappa, K_Gamma_DB_M[0] - 1.0, K_Gamma_DB_M[1],
                K_Gamma_DB_M[2], *K_Kappa_DB_M, *K_n, *K_m, *K_n_DB_M, *K_m_DB_M]
            ).T
        else:
            xis, K_Gamma, K_Kappa, K_n, K_m = stress_strain(rod, sol)
            export_data = np.vstack(
                [xis, K_Gamma[0] - 1.0, K_Gamma[1], K_Gamma[2], *K_Kappa, *K_n, *K_m]
            ).T

        np.savetxt(
            f"StrainMeasuresConvergence_{test_rods[j]}.txt",
            export_data,
            delimiter=", ",
            header=header,
            comments="",
        )

        ax[0].plot(K_Gamma[0], label=f"K_Gamma0 - {test_rods[j]}")
        ax[0].plot(K_Gamma[1], label=f"K_Gamma1 - {test_rods[j]}")
        ax[0].plot(K_Gamma[2], label=f"K_Gamma2 - {test_rods[j]}")

        ax[1].plot(K_Kappa[0], label=f"K_Kappa0 - {test_rods[j]}")
        ax[1].plot(K_Kappa[1], label=f"K_Kappa1 - {test_rods[j]}")
        ax[1].plot(K_Kappa[2], label=f"K_Kappa2 - {test_rods[j]}")

        ax[2].plot(K_n[0], label=f"K_n0 - {test_rods[j]}")
        ax[2].plot(K_n[1], label=f"K_n1 - {test_rods[j]}")
        ax[2].plot(K_n[2], label=f"K_n2 - {test_rods[j]}")

        ax[3].plot(K_m[0], label=f"K_m0 - {test_rods[j]}")
        ax[3].plot(K_m[1], label=f"K_m1 - {test_rods[j]}")
        ax[3].plot(K_m[2], label=f"K_m2 - {test_rods[j]}")

    ax[0].grid()
    ax[0].legend()
    ax[1].grid()
    ax[1].legend()

    plt.show()


if __name__ == "__main__":
    convergence()
