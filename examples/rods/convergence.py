from cardillo.beams import (
    RectangularCrossSection,
    Simo1986,
)
from cardillo.math import e3, Log_SO3, norm, Log_SE3, SE3, SE3inv
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.beams import (
    K_SE3_PetrovGalerkin_Quaternion,
    K_R3_SO3_PetrovGalerkin_Quaternion,
    K_R12_PetrovGalerkin_Quaternion,
    HigherOrder_K_SE3_PetrovGalerkin_Quaternion,
)
from cardillo.forces import K_Moment, Force
from cardillo import System
from cardillo.solver import Newton

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shutil import rmtree


def make_SE3_rod(
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
    q0 = K_SE3_PetrovGalerkin_Quaternion.straight_configuration(
        nelements,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
    )
    rod = K_SE3_PetrovGalerkin_Quaternion(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
        q0,
    )
    return q0, rod


def make_R3xSO3_rod(
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
    q0 = K_R3_SO3_PetrovGalerkin_Quaternion.straight_configuration(
        1,
        1,
        "Lagrange",
        "Lagrange",
        nelements,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
    )

    rod = K_R3_SO3_PetrovGalerkin_Quaternion(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        1,
        1,
        nelements,
        q0.copy(),
        basis_r="Lagrange",
        basis_psi="Lagrange",
    )

    return q0, rod


def make_R12_rod(polynomial_degree, volume_correction=False):
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
        q0 = K_R12_PetrovGalerkin_Quaternion.straight_configuration(
            polynomial_degree,
            polynomial_degree,
            "Lagrange",
            "Lagrange",
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )

        rod = K_R12_PetrovGalerkin_Quaternion(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            polynomial_degree,
            polynomial_degree,
            nelements,
            q0.copy(),
            basis_r="Lagrange",
            basis_psi="Lagrange",
            volume_correction=volume_correction,
        )

        return q0, rod

    return make


def make_higher_order_SE3_rod(polynomial_degree):
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
        q0 = HigherOrder_K_SE3_PetrovGalerkin_Quaternion.straight_configuration(
            polynomial_degree,
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        rod = HigherOrder_K_SE3_PetrovGalerkin_Quaternion(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            polynomial_degree,
            nelements,
            q0,
        )
        return q0, rod

    return make


# Young's and shear modulus
E = 1.0  # Meier2015
G = 0.5  # Meier2015

L = 1.0e3

# slenderness = 1.0e1
# atol = 1.0e-8
# slenderness = 1.0e2
# atol = 1.0e-10
# slenderness = 1.0e3
# atol = 1.0e-12
slenderness = 1.0e4
atol = 1.0e-14

# used cross section
width = L / slenderness

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

# starting point and orientation of initial point, initial length
r_OP0 = np.zeros(3, dtype=float)
A_IK0 = np.eye(3, dtype=float)

# define reference rod
# reference_rod = "SE3"
# reference_rod = "R3xSO3"
# reference_rod = "R12p1"
reference_rod = "R12p2"
# test_rods = ["R3xSO3", "SE3", "R12p1", "R12p2"]
# test_rods = ["SE3", "R12p1"]
# test_rods = ["R12p1", "R12p2"]
test_rods = ["SE3p1", "SE3p2"]

# dummy parameters for testing setup
nnodes_list = np.array([5], dtype=int)
nnodes_ref = 9
# nnodes_list = np.array([5, 9], dtype=int)
# nnodes_ref = 35
# nnodes_list = np.array([5, 9, 17], dtype=int)
# nnodes_ref = 65

# # used parameters for the paper
# nnodes_list = np.array([5, 9, 17, 33, 65, 129], dtype=int)
# nnodes_ref = 513

# volume_correction = False
volume_correction = True


def convergence():
    def solve(nnodes, rod, volume_correction):
        assert nnodes % 2 == 1
        if rod == "SE3":
            nelements = nnodes - 1
            make_rod = make_SE3_rod
        elif rod == "R3xSO3":
            nelements = nnodes - 1
            make_rod = make_R3xSO3_rod
        elif rod == "R12p1":
            nelements = nnodes - 1
            make_rod = make_R12_rod(1, volume_correction=volume_correction)
        elif rod == "R12p2":
            nelements = int((nnodes - 1) / 2)
            make_rod = make_R12_rod(2, volume_correction=volume_correction)
        elif rod == "SE3p1":
            nelements = nnodes - 1
            make_rod = make_higher_order_SE3_rod(1)
        elif rod == "SE3p2":
            nelements = int((nnodes - 1) / 2)
            make_rod = make_higher_order_SE3_rod(2)
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

        # junctions
        frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)

        # left and right joint
        joint1 = RigidConnection(frame1, rod, frame_ID2=(0,))

        # moment at the beam's tip
        Fi = material_model.Fi
        m = Fi[2] * 2 * np.pi / L * 0.25
        M = lambda t: t * e3 * m
        moment = K_Moment(M, rod, (1,))

        # force at the beam's tip
        f = m / L
        F = lambda t: t * f * e3
        print(f"f_max: {F(1)}")
        force = Force(F, rod, frame_ID=(1,))

        # assemble the model
        system = System()
        system.add(rod)
        system.add(frame1)
        system.add(joint1)
        system.add(moment)
        system.add(force)
        system.assemble()

        solver = Newton(
            system,
            n_load_steps=50,
            max_iter=100,
            atol=atol,
        )

        sol = solver.solve()

        return rod, sol

    # solve system with reference rod
    rod_ref, sol_ref = solve(nnodes_ref, reference_rod, volume_correction=True)

    # sample centerline deflection of reference solution
    num = 100
    xis = np.linspace(0, 1, num=num)
    r_OP_ref = rod_ref.centerline(sol_ref.q[-1], num=num)
    A_IK_ref = np.array(rod_ref.frames(sol_ref.q[-1], num=num)[1:])

    n_rods = len(test_rods)
    nnodes = len(nnodes_list)
    rods = np.zeros(n_rods, dtype=object)
    sols = np.zeros(n_rods, dtype=object)
    position_errors = np.zeros((n_rods, nnodes), dtype=float)
    rotation_errors = np.zeros((n_rods, nnodes), dtype=float)
    twist_errors = np.zeros((n_rods, nnodes), dtype=float)
    for i, rod_name in enumerate(test_rods):
        print(f"rod: {rod_name}")
        for j, nnodes in enumerate(nnodes_list):
            rod, sol = solve(nnodes, rod_name, volume_correction=volume_correction)

            # centerline errors
            r_OPi = rod.centerline(sol.q[-1], num=num)
            diff = r_OPi - r_OP_ref
            error = np.sqrt(sum([d @ d for d in diff])) / num
            position_errors[i, j] = error

            # rotation errors
            A_IKi = np.array(rod.frames(sol.q[-1], num=num)[1:])
            diff = []
            for k in range(num):
                diff.append(Log_SO3(A_IKi[:, :, k].T @ A_IK_ref[:, :, k]))
            diff = np.array(diff)
            error = np.sqrt(sum([d @ d for d in diff]))
            rotation_errors[i, j] = error

            # twist errors
            diff = []
            for k in range(num):
                H_IK = SE3(A_IKi[:, :, k], r_OPi[:, k])
                H_IK_ref = SE3(A_IK_ref[:, :, k], r_OP_ref[:, k])
                diff.append(Log_SE3(SE3inv(H_IK) @ H_IK_ref))
            diff = np.array(diff)
            error = np.sqrt(sum([d @ d for d in diff]))
            twist_errors[i, j] = error

        rods[i] = rod
        sols[i] = sol

    print(f"position_errors: {position_errors}")
    print(f"rotation_errors: {rotation_errors}")
    print(f"twist_errors: {twist_errors}")

    ###############
    # export errors
    ###############
    path = Path(__file__)
    path = Path(path.parent / path.stem)
    if path.exists():
        rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    for i, rod_name in enumerate(test_rods):
        header = "nnodes, position_error, rotation_errors, twist_errors"
        export_data = np.vstack(
            [nnodes_list, position_errors[i], rotation_errors[i], twist_errors[i]]
        ).T
        np.savetxt(
            path / f"{reference_rod}_vs_{rod_name}_slenderness_{slenderness:1.0e}.txt",
            export_data,
            delimiter=", ",
            header=header,
            comments="",
        )

    ##########################
    # plot rate of convergence
    ##########################
    fig, ax = plt.subplots(1, n_rods, sharey=True)
    for i, rod_name in enumerate(test_rods):
        ax[i].set_title(f"{rod_name}")
        ax[i].loglog(nnodes_list, position_errors[i], "-ok", label="e_r^100")
        ax[i].loglog(nnodes_list, rotation_errors[i], "-sk", label="e_psi^100")
        ax[i].loglog(nnodes_list, twist_errors[i], "-vk", label="e_theta^100")
        ax[i].loglog(nnodes_list, 90 / nnodes_list, "--k", label="~1 / n_el")
        ax[i].loglog(nnodes_list, 90 / nnodes_list**2, "-.k", label="~1 / nnodes^2")
        ax[i].loglog(nnodes_list, 90 / nnodes_list**3, "-.k", label="~1 / nnodes^3")
        ax[i].loglog(nnodes_list, 90 / nnodes_list**4, "-.k", label="~1 / nnodes^4")
        ax[i].grid()
        ax[i].legend()

    ####################################################
    # visualize centerline curves of final configuration
    ####################################################
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="3d"))
    ax.plot(*rod_ref.centerline(sol_ref.q[-1]), "-k", label="reference")
    styles = ["--r", "--g", "--b", "--m"]
    for i, (rod, sol) in enumerate(zip(rods, sols)):
        ax.plot(*rod.centerline(sol.q[-1]), styles[i], label=test_rods[i])
    ax.legend()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim3d(left=-L, right=L)
    ax.set_ylim3d(bottom=-L, top=L)
    ax.set_zlim3d(bottom=-L, top=L)

    #################
    # strain measures
    #################
    def stress_strain(rod, sol, nxi=1000):
        xis = np.linspace(0, 1, num=nxi)

        K_Gamma_bar = np.zeros((3, nxi))
        K_Kappa_bar = np.zeros((3, nxi))
        K_Gamma = np.zeros((3, nxi))
        K_Kappa = np.zeros((3, nxi))
        K_n = np.zeros((3, nxi))
        K_m = np.zeros((3, nxi))
        I_n = np.zeros((3, nxi))
        I_m = np.zeros((3, nxi))
        for i in range(nxi):
            frame_ID = (xis[i],)
            elDOF = rod.local_qDOF_P(frame_ID)

            # length of reference tangent vector
            Qe = rod.Q[elDOF]
            _, _, K_Gamma_bar0, K_Kappa_bar0 = rod._eval(Qe, xis[i])
            J = norm(K_Gamma_bar0)

            # current strain measures
            qe = sol.q[-1, rod.qDOF][elDOF]
            _, A_IK_i, K_Gamma_bar_i, K_Kappa_bar_i = rod._eval(qe, xis[i])

            K_Gamma_bar[:, i] = K_Gamma_bar_i
            K_Kappa_bar[:, i] = K_Kappa_bar_i
            K_Gamma[:, i] = K_Gamma_bar_i / J
            K_Kappa[:, i] = K_Kappa_bar_i / J
            K_n[:, i] = material_model.K_n(
                K_Gamma_bar_i / J, K_Gamma_bar0 / J, K_Kappa_bar_i / J, K_Kappa_bar0 / J
            )
            K_m[:, i] = material_model.K_m(
                K_Gamma_bar_i / J, K_Gamma_bar0 / J, K_Kappa_bar_i / J, K_Kappa_bar0 / J
            )
            I_n[:, i] = A_IK_i @ K_n[:, i]
            I_m[:, i] = A_IK_i @ K_m[:, i]

        return xis, K_Gamma_bar, K_Gamma, K_Kappa_bar, K_Kappa, K_n, K_m, I_n, I_m

    fig, ax = plt.subplots(1, 4)

    xis, K_Gamma_bar, K_Gamma, K_Kappa_bar, K_Kappa, K_n, K_m, I_n, I_m = stress_strain(
        rod_ref, sol_ref
    )
    header = "xi, K_Gamma1_minus_1, K_Gamma2, K_Gamma3, K_Kappa1, K_Kappa2, K_Kappa3, K_n1, K_n2, K_n3, K_m1, K_m2, K_m3"
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

    ax[1].set_title("K_Gamma")
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

    for i, (rod, sol) in enumerate(zip(rods, sols)):
        (
            xis,
            K_Gamma_bar,
            K_Gamma,
            K_Kappa_bar,
            K_Kappa,
            K_n,
            K_m,
            I_n,
            I_m,
        ) = stress_strain(rod, sol)
        export_data = np.vstack(
            [xis, K_Gamma[0] - 1.0, K_Gamma[1], K_Gamma[2], *K_Kappa, *K_n, *K_m]
        ).T
        np.savetxt(
            f"StrainMeasuresConvergence_{test_rods[i]}.txt",
            export_data,
            delimiter=", ",
            header=header,
            comments="",
        )

        ax[0].plot(K_Gamma[0], label=f"K_Gamma0 - {test_rods[i]}")
        ax[0].plot(K_Gamma[1], label=f"K_Gamma1 - {test_rods[i]}")
        ax[0].plot(K_Gamma[2], label=f"K_Gamma2 - {test_rods[i]}")

        ax[1].plot(K_Kappa[0], label=f"K_Kappa0 - {test_rods[i]}")
        ax[1].plot(K_Kappa[1], label=f"K_Kappa1 - {test_rods[i]}")
        ax[1].plot(K_Kappa[2], label=f"K_Kappa2 - {test_rods[i]}")

        ax[2].plot(K_n[0], label=f"K_n0 - {test_rods[i]}")
        ax[2].plot(K_n[1], label=f"K_n1 - {test_rods[i]}")
        ax[2].plot(K_n[2], label=f"K_n2 - {test_rods[i]}")

        ax[3].plot(K_m[0], label=f"K_m0 - {test_rods[i]}")
        ax[3].plot(K_m[1], label=f"K_m1 - {test_rods[i]}")
        ax[3].plot(K_m[2], label=f"K_m2 - {test_rods[i]}")

    ax[0].grid()
    ax[0].legend()
    ax[1].grid()
    ax[1].legend()

    plt.show()


if __name__ == "__main__":
    convergence()
