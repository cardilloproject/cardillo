from cardillo.math import e1, e2, e3, sqrt, pi
from cardillo.beams import (
    CircularCrossSection,
    RectangularCrossSection,
    Simo1986,
)
from cardillo.math.algebra import ax2skew, norm, inv3D, cross3
from cardillo.math.rotations import (
    Exp_SO3,
    Log_SO3,
    T_SO3,
    A_IK_basic,
)
from cardillo.discrete import Frame
from cardillo.constraints import (
    RigidConnection,
    SphericalJoint,
)
from cardillo.beams import (
    animate_beam,
    K_TimoshenkoAxisAngleSE3,
    DirectorAxisAngle,
    Crisfield1999,
)
from cardillo.forces import Force, K_Force, K_Moment, Moment, DistributedForce1DBeam
from cardillo import System
from cardillo.solver import Newton

import numpy as np
import matplotlib.pyplot as plt


def helix():
    """This example examines membrande and shear locking similar to Harsch2020 example 4.2.

    References:
    ===========
    Harsch2020: https://doi.org/10.1177/1081286521100079
    """
    Beam = Crisfield1999
    # Beam = K_TimoshenkoAxisAngleSE3

    # number of points for normalized L2 error
    nxi = 200
    xis = np.linspace(0, 1, nxi)

    # solve for helix length
    n = 2  # number of helix coils
    scale = 1.0e1
    R0 = 1 * scale  # helix radius
    h = 5 * scale  # helix height
    c = h / (R0 * 2 * np.pi * n)
    L = np.sqrt(1 + c**2) * R0 * 2 * np.pi * n
    print(f"R0: {R0}")
    print(f"h: {h}")
    print(f"c: {c}")
    print(f"n: {n}")
    print(f"L: {L}")

    # Young's and shear modulus
    E = 1.0
    G = E / 2.0

    # reference solution
    def r_ref(xi):
        alpha = 2 * np.pi * n * xi
        return R0 * np.array([np.sin(alpha), -np.cos(alpha), c * alpha])

    ####################################
    # dummy parameters for testing setup
    ####################################
    # 5 elements is the minimal solvable number, so lets use nelements >= 8
    # nelements_list = np.array([5], dtype=int)
    # nelements_list = np.array([8], dtype=int)
    # nelements_ref = 8
    nelements_list = np.array([10], dtype=int)

    # Crisfield1999
    if Beam == Crisfield1999:
        triplets = [
            # (1.0e1, 1.0e-8, 100),
            (1e2, 1e-10, 1000),
            # (1e3, 1e-12, 1000),
            # (1e4, 1e-14, 1000),
        ]
    elif Beam == K_TimoshenkoAxisAngleSE3:
        triplets = [
            (1.0e1, 1e-8, 70),
            # (1e2, 1e-10, 200),
            # (1e3, 1e-12, 200),
            # (1e4, 1e-14, 600),
        ]

    solutions = np.zeros((len(triplets), len(nelements_list)), dtype=object)
    beams = np.zeros((len(triplets), len(nelements_list)), dtype=object)
    tip_displacement = np.zeros((len(triplets), len(nelements_list), 3), dtype=float)
    tip_orientation = np.zeros((len(triplets), len(nelements_list), 3), dtype=float)

    for triplet_idx, triplet in enumerate(triplets):
        slenderness, atol, n_load_steps = triplet

        # cross section
        d = L / slenderness
        r = radius = d / 2
        line_density = 1
        cross_section = CircularCrossSection(line_density, radius)
        A = cross_section.area
        I = cross_section.second_moment
        A_rho0 = line_density * cross_section.area
        K_S_rho0 = line_density * cross_section.first_moment
        K_I_rho0 = line_density * cross_section.second_moment
        print(f"slenderness: {slenderness}")
        print(f"r: {r} m")
        print(f"A: {A} m^2")

        # build quadratic material model
        Ei = np.array([E * A, G * A, G * A], dtype=float)
        Fi = np.array([G * I[0, 0], E * I[1, 1], E * I[2, 2]], dtype=float)
        material_model = Simo1986(Ei, Fi)
        print(f"Ei: {Ei} N")
        print(f"Fi: {Fi} N")

        # couple at right end
        alpha_xi = 2 * np.pi * n
        M1 = Fi[0] * c * R0 * alpha_xi**2 / L**2
        M3 = Fi[2] * R0 * alpha_xi**2 / L**2
        K_M = lambda t: t * np.array([M1, 0, M3])

        # compute initial configuration of beam
        r_OP0 = R0 * np.array([0, -1, 0], dtype=float)
        D1 = np.array([1, 0, c], dtype=float) / np.sqrt(1.0 + c**2)
        D2 = np.array([0, 1, 0], dtype=float)
        D3 = np.array([-c, 0, 1], dtype=float) / np.sqrt(1.0 + c**2)
        A_IK0 = np.vstack((D1, D2, D3)).T

        for nelements_idx, nelements in enumerate(nelements_list):

            if Beam == K_TimoshenkoAxisAngleSE3:
                q0 = K_TimoshenkoAxisAngleSE3.straight_configuration(
                    nelements,
                    L,
                    r_OP=r_OP0,
                    A_IK=A_IK0,
                )
                beam = K_TimoshenkoAxisAngleSE3(
                    cross_section,
                    material_model,
                    A_rho0,
                    K_S_rho0,
                    K_I_rho0,
                    nelements,
                    q0,
                )
            elif Beam == Crisfield1999:
                polynomial_degree = 1
                basis = "Lagrange"
                q0 = Crisfield1999.straight_configuration(
                    polynomial_degree,
                    polynomial_degree,
                    basis,
                    basis,
                    nelements,
                    L,
                    r_OP=r_OP0,
                    A_IK=A_IK0,
                )
                beam = Crisfield1999(
                    cross_section,
                    material_model,
                    A_rho0,
                    K_S_rho0,
                    K_I_rho0,
                    polynomial_degree,
                    polynomial_degree,
                    nelements,
                    q0,
                    basis_r=basis,
                    basis_psi=basis,
                )
            else:
                raise NotImplementedError

            frame = Frame(r_OP=r_OP0, A_IK=A_IK0)
            joint = RigidConnection(frame, beam, r_OP0, frame_ID2=(0,))

            # assemble the model
            model = System()
            model.add(beam)
            model.add(frame)
            model.add(joint)
            model.add(K_Moment(K_M, beam, (1,)))
            model.assemble()

            sol = Newton(
                model,
                n_load_steps=n_load_steps,
                max_iter=100,
                atol=atol,
            ).solve()

            beams[triplet_idx, nelements_idx] = beam
            solutions[triplet_idx, nelements_idx] = sol

            t1 = sol.t[-1]
            q1_p = sol.q[-1][beam.local_qDOF_P((1,))]
            tip_displacement[triplet_idx, nelements_idx] = beam.r_OP(t1, q1_p, (1,))
            tip_orientation[triplet_idx, nelements_idx] = Log_SO3(
                beam.A_IK(t1, q1_p, (1,))
            )

    # # sample centerline deflection of reference solution
    # num = 100
    # position_errors = np.zeros((len(triplets), len(nelements_list) + 1), dtype=float)
    # rotation_errors = np.zeros((len(triplets), len(nelements_list) + 1), dtype=float)
    # tip_position_errors = np.zeros(
    #     (len(triplets), len(nelements_list) + 1), dtype=float
    # )
    # tip_rotation_errors = np.zeros(
    #     (len(triplets), len(nelements_list) + 1), dtype=float
    # )
    # for i in range(len(triplets)):
    #     sol_ref = solutions[i, 0]
    #     beam_ref = beams[i, 0]

    #     r_OP_ref = beam_ref.centerline(sol_ref.q[-1], num=num)
    #     A_IK_ref = np.array(beam_ref.frames(sol_ref.q[-1], num=num)[1:])
    #     for j in range(len(nelements_list) + 1):
    #         beam = beams[i, j]
    #         sol = solutions[i, j]

    #         # centerline errors
    #         r_OPi = beam.centerline(sol.q[-1], num=num)
    #         diff = r_OPi - r_OP_ref
    #         error = sqrt(sum([d @ d for d in diff])) / num
    #         position_errors[i, j] = error

    #         # tip displacement errors
    #         tip_position_errors[i, j] = norm(diff[-1]) / norm(r_OP_ref[-1])

    #         # rotation errors
    #         A_IKi = np.array(beam.frames(sol.q[-1], num=num)[1:])
    #         diff = []
    #         for k in range(num):
    #             diff.append(Log_SO3(A_IKi[:, :, k].T @ A_IK_ref[:, :, k]))
    #         diff = np.array(diff)
    #         error = sqrt(sum([d @ d for d in diff]))
    #         rotation_errors[i, j] = error

    #         # tip orientation errors
    #         tip_rotation_errors[i, j] = norm(diff[-1]) / norm(r_OP_ref[-1])

    # print(f"e_r_100:\n{position_errors}")
    # print(f"e_psi_100:\n{rotation_errors}")
    # print(f"tip_position_errors:\n{tip_position_errors}")
    # print(f"tip_rotation_errors:\n{tip_rotation_errors}")

    # header = "ref, nel_1, nel_64"
    # np.savetxt(
    #     "code/results/HelixLocking_e_r_100.txt",
    #     position_errors,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )
    # np.savetxt(
    #     "code/results/HelixLocking_e_psi_100.txt",
    #     rotation_errors,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    ############
    # VTK export
    ############
    from pathlib import Path
    from cardillo.utility import Export

    path = Path(__file__)
    e = Export(path.parent, path.stem, True, 30, sol)
    e.export_contr(beam, level="centerline + directors", num=50)
    e.export_contr(beam, level="volume", n_segments=5, num=50)

    #################
    # strain measures
    #################
    xis = np.linspace(0.0, 1.0, num=nxi, dtype=float)

    K_Gamma = np.zeros((3, nxi), dtype=float)
    K_Kappa = np.zeros((3, nxi), dtype=float)
    for i, xi in enumerate(xis):
        elDOF = beam.local_qDOF_P((xi,))

        _, _, K_Gamma_bar, K_Kappa_bar = beam._eval(sol.q[0, beam.qDOF[elDOF]], xi)
        J = norm(K_Gamma_bar)

        _, _, K_Gamma_bar, K_Kappa_bar = beam._eval(sol.q[-1, beam.qDOF[elDOF]], xi)
        K_Gamma[:, i] = K_Gamma_bar / J
        K_Kappa[:, i] = K_Kappa_bar / J

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(xis, K_Gamma[0], "-r", label="K_Gamma1")
    ax[0].plot(xis, K_Gamma[1], "-g", label="K_Gamma2")
    ax[0].plot(xis, K_Gamma[2], "-b", label="K_Gamma3")
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(xis, K_Kappa[0], "-r", label="K_Kappa1")
    ax[1].plot(xis, K_Kappa[1], "-g", label="K_Kappa2")
    ax[1].plot(xis, K_Kappa[2], "-b", label="K_Kappa3")
    ax[1].grid()
    ax[1].legend()

    header = "xi, K_Gamma1, K_Gamma1_minus_1, K_Gamma2, K_Gamma3, K_Kappa1, K_Kappa2, K_Kappa3"
    export_data = np.vstack(
        [xis, K_Gamma[0], K_Gamma[0] - 1.0, K_Gamma[1], K_Gamma[2], *K_Kappa]
    ).T
    np.savetxt(
        f"Simon/helix_strain_measures_{Beam.__name__}_slenderness_{slenderness}.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    # header = "ref, nel_1, nel_64"
    # np.savetxt(
    #     "code/results/HelixLocking_e_r_100.txt",
    #     position_errors,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )
    # np.savetxt(
    #     "code/results/HelixLocking_e_psi_100.txt",
    #     rotation_errors,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    ###############
    # visualization
    ###############
    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    # scale = 1.2 * R0
    scale = 1.2 * max(R0, h / 2)
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=0, top=2 * scale)

    centerline = beam.centerline(sol.q[-1], num=nxi)
    reference = np.array([r_ref(xi) for xi in xis])
    ax.plot(*reference.T, "-k")
    ax.plot(*centerline, "--r")

    q = sol.q
    nt = len(q)
    t = sol.t[:nt]
    animate_beam(t, q, [beam], L, show=True)


def SE3_vs_Crisfield(r_OP0, psi0, r_OP1, psi1, L, cross_section):
    # Young's and shear modulus
    E = 1.0
    G = E / 2.0

    # discretization
    nelements = 1

    # cross section
    slenderness = 10
    width = L / slenderness
    line_density = 1
    if cross_section == "circle":
        cross_section = CircularCrossSection(line_density, width)
    else:
        cross_section = RectangularCrossSection(line_density, width, width)
    A = cross_section.area
    I = cross_section.second_moment
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    # build quadratic material model
    Ei = np.array([E * A, G * A, G * A], dtype=float)
    Fi = np.array([G * I[0, 0], E * I[1, 1], E * I[2, 2]], dtype=float)
    material_model = Simo1986(Ei, Fi)

    q_r = np.vstack((r_OP0, r_OP1)).T.reshape(-1, order="C")
    q_psi = np.vstack((psi0, psi1)).T.reshape(-1, order="C")

    A_IK0 = Exp_SO3(psi0)
    q0 = K_TimoshenkoAxisAngleSE3.straight_configuration(1, L, A_IK=A_IK0)

    q_rod = np.concatenate((q_r, q_psi))
    q = np.concatenate((q_rod, q_rod))

    rod_SE3 = K_TimoshenkoAxisAngleSE3(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
        q0.copy(),
    )

    rod_Crisfield = Crisfield1999(
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

    # assemble the model
    model = System()
    model.add(rod_SE3)
    model.add(rod_Crisfield)
    model.assemble()

    from cardillo.solver import Solution

    # sol = Solution([0], [model.q0], [model.u0])
    sol = Solution([0, 1], [model.q0, q], [model.u0, model.u0])

    ############
    # VTK export
    ############
    from pathlib import Path
    from cardillo.utility import Export

    path = Path(__file__)
    fps = 60
    e = Export(path.parent, path.stem, True, fps, sol)
    e.export_contr(
        rod_SE3, file_name="SE3_centerline", level="centerline + directors", num=10
    )
    e.export_contr(
        rod_SE3, file_name="SE3_volume", level="volume", n_segments=1, num=50
    )
    e.export_contr(
        rod_Crisfield,
        file_name="Crisfield_centerline",
        level="centerline + directors",
        num=10,
    )
    e.export_contr(
        rod_Crisfield,
        file_name="Crisfield_volume",
        level="volume",
        n_segments=1,
        num=50,
    )

    #########################
    # compute strain measures
    #########################
    nxi = 100
    xis = np.linspace(0.0, 1.0, num=nxi, dtype=float)

    K_Gamma_SE3 = np.zeros((3, nxi), dtype=float)
    K_Kappa_SE3 = np.zeros((3, nxi), dtype=float)
    K_Gamma_Crisfield = np.zeros((3, nxi), dtype=float)
    K_Kappa_Crisfield = np.zeros((3, nxi), dtype=float)
    for i, xi in enumerate(xis):
        # SE(3) rod
        elDOF = rod_SE3.local_qDOF_P((xi,))

        _, _, K_Gamma_bar, K_Kappa_bar = rod_SE3._eval(
            sol.q[0][rod_SE3.qDOF[elDOF]], xi
        )
        J = norm(K_Gamma_bar)

        _, _, K_Gamma_bar, K_Kappa_bar = rod_SE3._eval(
            sol.q[-1][rod_SE3.qDOF[elDOF]], xi
        )

        K_Gamma_SE3[:, i] = K_Gamma_bar / J
        K_Kappa_SE3[:, i] = K_Kappa_bar / J

        # Crisfield - R3 x SO(3)
        elDOF = rod_Crisfield.local_qDOF_P((xi,))

        _, _, K_Gamma_bar, K_Kappa_bar = rod_Crisfield._eval(
            sol.q[0][rod_Crisfield.qDOF[elDOF]], xi
        )
        J = norm(K_Gamma_bar)

        _, _, K_Gamma_bar, K_Kappa_bar = rod_Crisfield._eval(
            sol.q[-1][rod_Crisfield.qDOF[elDOF]], xi
        )

        K_Gamma_Crisfield[:, i] = K_Gamma_bar / J
        K_Kappa_Crisfield[:, i] = K_Kappa_bar / J

    header = "xi, K_Gamma1_SE3, K_Gamma2_SE3, K_Gamma3_SE3, K_Kappa1_SE3, K_Kappa2_SE3, K_Kappa3_SE3, K_Gamma1_Crisfield, K_Gamma2_Crisfield, K_Gamma3_Crisfield, K_Kappa1_Crisfield, K_Kappa2_Crisfield, K_Kappa3_Crisfield"
    export_data = np.vstack(
        [xis, *K_Gamma_SE3, *K_Kappa_SE3, *K_Gamma_Crisfield, *K_Kappa_Crisfield]
    ).T
    np.savetxt(
        "Simon/CrisfieldvsSE3StrainMeasures.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(xis, K_Gamma_Crisfield[0], "-r", label="gamma1 - Crisfield")
    ax[0].plot(xis, K_Gamma_Crisfield[1], "-g", label="gamma2 - Crisfield")
    ax[0].plot(xis, K_Gamma_Crisfield[2], "-b", label="gamma3 - Crisfield")
    ax[0].plot(xis, K_Gamma_SE3[0], "--r", label="gamma1 - SE(3)")
    ax[0].plot(xis, K_Gamma_SE3[1], "--g", label="gamma2 - SE(3)")
    ax[0].plot(xis, K_Gamma_SE3[2], "--b", label="gamma3 - SE(3)")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(xis, K_Kappa_Crisfield[0], "-r", label="kappa1 - Crisfield")
    ax[1].plot(xis, K_Kappa_Crisfield[1], "-g", label="kappa2 - Crisfield")
    ax[1].plot(xis, K_Kappa_Crisfield[2], "-b", label="kappa3 - Crisfield")
    ax[1].plot(xis, K_Kappa_SE3[0], "--r", label="kappa1 - SE(3)")
    ax[1].plot(xis, K_Kappa_SE3[1], "--g", label="kappa2 - SE(3)")
    ax[1].plot(xis, K_Kappa_SE3[2], "--b", label="kappa3 - SE(3)")
    ax[1].grid()
    ax[1].legend()

    plt.show()


if __name__ == "__main__":
    helix()

    # #######################
    # # given strain measures
    # #######################
    # r_OP0 = np.zeros(3)
    # psi0 = np.zeros(3)
    # A_IK0 = Exp_SO3(psi0)

    # L = 1

    # # K_kappa_IK = np.array([0, 0, 0])
    # K_kappa_IK = np.array([1, 0, 1]) * 0.5 * np.pi
    # K_gamma = np.array([1, 0, 0])

    # # ##############
    # # # 1. extension
    # # ##############
    # # K_kappa_IK = np.zeros(3)
    # # K_gamma = np.array([1.25, 0, 0])

    # # ##############
    # # # 2. shear2
    # # ##############
    # # K_kappa_IK = np.zeros(3)
    # # K_gamma = np.array([1.0, 0.3, 0])

    # # ##############
    # # # 3. shear3
    # # ##############
    # # K_kappa_IK = np.zeros(3)
    # # K_gamma = np.array([1.0, 0, 0.3])

    # # ##############
    # # # 4. torsion
    # # ##############
    # # K_kappa_IK = np.array([np.pi / (2 * L), 0, 0])
    # # K_gamma = np.array([1.0, 0, 0])

    # # ##############
    # # # 5. bending2
    # # ##############
    # # K_kappa_IK = np.array([0, -np.pi / (4 * L), 0])
    # # K_gamma = np.array([1.0, 0, 0])

    # ##############
    # # 6. bending3
    # ##############
    # K_kappa_IK = np.array([0, 0, -np.pi / (4 * L)])
    # K_gamma = np.array([1.0, 0, 0])

    # A_IK1 = A_IK0 @ Exp_SO3(L * K_kappa_IK)
    # psi1 = Log_SO3(A_IK1)

    # r_OP1 = r_OP0 + L * A_IK0 @ T_SO3(L * K_kappa_IK).T @ K_gamma

    # # cross_section = "circle"
    # cross_section = "square"
    # SE3_vs_Crisfield(r_OP0, psi0, r_OP1, psi1, L, cross_section)
