import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pathlib import Path

from cardillo import System
from cardillo.discrete import RigidBody, RigidBodyEuler
from cardillo.math import axis_angle2quat, Exp_SO3_quat, cross3
from cardillo.forces import Force
from cardillo.contacts import Sphere2PlaneCoulombContensouMoeller
from cardillo.solver import (
    Moreau,
    Rattle,
    BackwardEuler,
)


def make_system(RigidBodyBase):
    assert RigidBodyBase in [RigidBody, RigidBodyEuler]

    system = System()

    # Dynamics:
    m = 6e-3
    I1 = 8e-7  # = I2
    I3 = 7e-7
    B_Theta_C = np.diag([I1, I1, I3])
    g = 9.81

    # Geometry:
    a1 = 3e-3
    a2 = 1.6e-2
    R1 = 1.5e-2
    R2 = 5e-3
    K_r_CP1 = np.array([0, 0, a1])
    K_r_CP2 = np.array([0, 0, a2])

    mu = 0.3  # = mu1 = mu2
    e_N = 0  # = eN1 = eN2
    e_F = 0
    R = 5e-4

    # Leine2003
    # z0 = 1.2015e-2
    # theta0 = 0.1
    # psi_dot0 = 180

    theta0 = 0.1
    gamma = 0
    omega = 180

    # initial coordinates
    p0 = axis_angle2quat(np.array([1, 0, 0]), theta0)
    A_IK = Exp_SO3_quat(p0)
    K_r_PS = np.array([0, 0, -a1])
    r_OC = np.array([0, 0, R1]) + A_IK @ K_r_PS
    if RigidBodyBase is RigidBody:
        q0 = np.zeros(7, dtype=float)
        q0[:3] = r_OC
        q0[3:] = p0
    elif RigidBodyBase is RigidBodyEuler:
        axis = "zxz"
        q0[:3] = r_OC
        q0[4] = theta0

    # initial velocities
    K_omega_IK = np.array([gamma, 0, omega])
    v_C = A_IK @ cross3(K_omega_IK, K_r_PS)

    print(f"v_C = {v_C}")
    print(
        f"v_C = {v_C + A_IK @ cross3(K_omega_IK, np.array([0, 0, a1]) + np.array([0, 0, -R1]))}"
    )

    u0 = np.zeros(6, dtype=float)
    u0[:3] = v_C
    u0[3:] = K_omega_IK

    if RigidBodyBase is RigidBody:
        top = RigidBody(m, B_Theta_C, q0=q0, u0=u0)
    elif RigidBodyBase is RigidBodyEuler:
        top = RigidBodyEuler(m, B_Theta_C, axis=axis, q0=q0, u0=u0)

    contact1 = Sphere2PlaneCoulombContensouMoeller(
        system.origin,
        top,
        R1,
        R,
        mu,
        e_N,
        e_F,
        B_r_CP=K_r_CP1,
    )
    contact2 = Sphere2PlaneCoulombContensouMoeller(
        system.origin,
        top,
        R2,
        R,
        mu,
        e_N,
        e_F,
        B_r_CP=K_r_CP2,
    )

    gravity = Force(np.array([0, 0, -m * g]), top)

    system.add(top, contact1, contact2, gravity)
    system.assemble()

    return system, top, contact1, contact2


def run(export=True):
    """Example 10.6 of Capobianco2021.

    References:
    -----------
    Capobianco2021: https://doi.org/10.1002/nme.6801 \\
    Moeller2009: https://www.inm.uni-stuttgart.de/institut/mitarbeiter/leine/papers/proceedings/Moeller_x_Leine_x_Glocker_-_An_efficient_approximation_of_set-valued_force_laws_of_normal_cone_type_ESMC2009.pdf \\
    Leine2003: https://doi.org/10.1016/S0997-7538(03)00025-1
    """

    system, top, contact1, contact2 = make_system(RigidBody)

    # t1 = 8
    t_final = 2
    # t_final = 0.1
    # dt1 = 1e-4
    dt1 = 5e-4
    dt2 = 1e-4  # Moreau
    dt2 = 5e-5
    dt1 = 1e-4
    dt2 = 1e-4

    # sol1, label1 = Rattle(system, t_final, dt1, atol=1e-12).solve(), "Rattle"
    sol1, label1 = (
        MoreauShiftedNew(system, t_final, dt2, atol=1e-12).solve(),
        "MoreauShiftedNew",
    )

    sol2, label2 = (
        Moreau(system, t_final, dt2, atol=1e-12).solve(),
        "Moreau",
    )

    t1 = sol1.t
    q1 = sol1.q
    u1 = sol1.u
    P_N1 = sol1.P_N
    P_F1 = sol1.P_F

    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u
    P_N2 = sol2.P_N
    P_F2 = sol2.P_F

    fig, ax = plt.subplots(2, 3)

    ax[0, 0].set_title("x(t)")
    ax[0, 0].plot(t1, q1[:, 0], "-k", label=label1)
    ax[0, 0].plot(t2, q2[:, 0], "--r", label=label2)
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].set_title("y(t)")
    ax[0, 1].plot(t1, q1[:, 1], "-k", label=label1)
    ax[0, 1].plot(t2, q2[:, 1], "--r", label=label2)
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[0, 2].set_title("z(t)")
    ax[0, 2].plot(t1, q1[:, 2], "-k", label=label1)
    ax[0, 2].plot(t2, q2[:, 2], "--r", label=label2)
    ax[0, 2].grid()
    ax[0, 2].legend()

    ax[1, 0].set_title("u_x(t)")
    ax[1, 0].plot(t1, u1[:, 0], "-k", label=label1)
    ax[1, 0].plot(t2, u2[:, 0], "--r", label=label2)
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[1, 1].set_title("u_y(t)")
    ax[1, 1].plot(t1, u1[:, 1], "-k", label=label1)
    ax[1, 1].plot(t2, u2[:, 1], "--r", label=label2)
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[1, 2].set_title("u_z(t)")
    ax[1, 2].plot(t1, u1[:, 2], "-k", label=label1)
    ax[1, 2].plot(t2, u2[:, 2], "--r", label=label2)
    ax[1, 2].grid()
    ax[1, 2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(2, 3)

    nt1 = len(t1)
    angles1 = np.zeros((nt1, 3), dtype=float)
    for i in range(len(t1)):
        A_IK = top.A_IK(t1[i], q1[i])
        angles1[i] = Rotation.from_matrix(A_IK).as_euler("zxz")

    nt2 = len(t2)
    angles2 = np.zeros((nt2, 3), dtype=float)
    for i in range(len(t2)):
        A_IK = top.A_IK(t2[i], q2[i])
        angles2[i] = Rotation.from_matrix(A_IK).as_euler("zxz")

    ax[0, 0].set_title("psi(t)")
    ax[0, 0].plot(t1, angles1[:, 0], "-k", label=label1)
    ax[0, 0].plot(t2, angles2[:, 0], "--r", label=label2)
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].set_title("theta(t)")
    ax[0, 1].plot(t1, angles1[:, 1], "-k", label=label1)
    ax[0, 1].plot(t2, angles2[:, 1], "--r", label=label2)
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[0, 2].set_title("phi(t)")
    ax[0, 2].plot(t1, angles1[:, 2], "-k", label=label1)
    ax[0, 2].plot(t2, angles2[:, 2], "--r", label=label2)
    ax[0, 2].legend()
    ax[0, 2].grid()

    ax[1, 0].set_title("omega1(t)")
    ax[1, 0].plot(t1, u1[:, 3], "-k", label=label1)
    ax[1, 0].plot(t2, u2[:, 3], "--r", label=label2)
    ax[1, 0].legend()
    ax[1, 0].grid()

    ax[1, 1].set_title("omega2(t)")
    ax[1, 1].plot(t1, u1[:, 4], "-k", label=label1)
    ax[1, 1].plot(t2, u2[:, 4], "--r", label=label2)
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[1, 2].set_title("omega3(t)")
    ax[1, 2].plot(t1, u1[:, 5], "-k", label=label1)
    ax[1, 2].plot(t2, u2[:, 5], "--r", label=label2)
    ax[1, 2].grid()
    ax[1, 2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(4)

    ax[0].set_title("P_N(t)")
    ax[0].plot(t1, P_N1[:, 0], "-ok", label=label1)
    ax[0].plot(t2, P_N2[:, 0], "--xr", label=label2)
    ax[0].grid()
    ax[0].legend()

    ax[1].set_title("P_Fx(t)")
    ax[1].plot(t1, P_F1[:, 0], "-ok", label=label1)
    ax[1].plot(t2, P_F2[:, 0], "--xr", label=label2)
    ax[1].grid()
    ax[1].legend()

    ax[2].set_title("P_Fy(t)")
    ax[2].plot(t1, P_F1[:, 1], "-ok", label=label1)
    ax[2].plot(t2, P_F2[:, 1], "--xr", label=label2)
    ax[2].grid()
    ax[2].legend()

    ax[3].set_title("P_drill(t)")
    ax[3].plot(t1, P_F1[:, 2], "-ok", label=label1)
    ax[3].plot(t2, P_F2[:, 2], "--xr", label=label2)
    ax[3].grid()
    ax[3].legend()

    plt.tight_layout()

    if export:
        path = Path(__file__)

        np.savetxt(
            path.parent / "state1.dat",
            np.hstack((sol1.t[:, None], q1[:, :3], angles1)),
            delimiter=", ",
            header="t, x, y, z, psi, theta, phi",
            comments="",
        )

        np.savetxt(
            path.parent / "state2.dat",
            np.hstack((sol2.t[:, None], q2[:, :3], angles2)),
            delimiter=", ",
            header="t, x, y, z, psi, theta, phi",
            comments="",
        )

    plt.show()


def convergence(export=True):
    system, top, contact1, contact2 = make_system(RigidBody)
    # system, top, contact1, contact2 = make_system(RigidBodyEuler)

    # tol_ref = 1.0e-8
    # tol = 1.0e-8
    tol_ref = 1.0e-14
    tol = 1.0e-14

    # compute step sizes with powers of 2
    # dt_ref = 1e-4
    dt_ref = 1e-5
    # t1 = (2.0**12) * dt_ref # 0.4096 s
    # t1 = (2.0**10) * dt_ref # 0.1024 s
    # t1 = (2.0**9) * dt_ref  # 0.0512 s
    # t1 = (2.0**8) * dt_ref  # 0.0256 s
    t1 = (2.0**6) * dt_ref  # 0.0256 s
    # dts = (2.0 ** np.arange(6, 1, -1)) * dt_ref
    # dts = (2.0 ** np.arange(8, 2, -1)) * dt_ref
    dts = (2.0 ** np.arange(5, 2, -1)) * dt_ref

    print(f"t1: {t1}")
    print(f"dts: {dts}")
    # exit()

    # errors for possible solvers
    q_errors_transient = np.inf * np.ones((2, len(dts)), dtype=float)
    u_errors_transient = np.inf * np.ones((2, len(dts)), dtype=float)
    P_g_errors_transient = np.inf * np.ones((2, len(dts)), dtype=float)
    P_N_errors_transient = np.inf * np.ones((2, len(dts)), dtype=float)
    P_F_errors_transient = np.inf * np.ones((2, len(dts)), dtype=float)
    q_errors_longterm = np.inf * np.ones((2, len(dts)), dtype=float)
    u_errors_longterm = np.inf * np.ones((2, len(dts)), dtype=float)
    P_g_errors_longterm = np.inf * np.ones((2, len(dts)), dtype=float)
    P_N_errors_longterm = np.inf * np.ones((2, len(dts)), dtype=float)
    P_F_errors_longterm = np.inf * np.ones((2, len(dts)), dtype=float)

    ###################################################################
    # compute reference solution as described in Arnold2015 Section 3.3
    ###################################################################
    Solver, label, kwargs = Rattle, "Rattle", {}

    # Solver, label, kwargs = (
    #     NPIRK,
    #     "Radau IIa(2)",
    #     {"butcher_tableau": RadauIIATableau(2)},
    # )

    Solver, label, kwargs = NonsmoothGeneralizedAlpha, "Gen-alpha", {}

    print(f"compute reference solution with " + label)
    # reference = Solver(system, t1, dt_ref, atol=tol_ref, **kwargs).solve()
    reference = Solver(system, t1, dt_ref, **kwargs).solve()
    print(f"done")

    plot_state = False
    if plot_state:
        t_ref = reference.t
        q_ref = reference.q
        u_ref = reference.u
        P_N_ref = reference.P_N
        P_F_ref = reference.P_F

        ###################
        # visualize results
        ###################
        fig = plt.figure(figsize=plt.figaspect(0.5))

        # center of mass
        ax = fig.add_subplot(2, 3, 1)
        ax.plot(t_ref, q_ref[:, 0], "-r", label="x")
        ax.plot(t_ref, q_ref[:, 1], "-g", label="y")
        ax.plot(t_ref, q_ref[:, 2], "-b", label="z")
        ax.grid()
        ax.legend()

        # alpha, beta, gamma
        nt_ref = len(t_ref)
        angles_ref = np.zeros((nt_ref, 3), dtype=float)
        for i in range(nt_ref):
            A_IK = top.A_IK(t_ref[i], q_ref[i])
            angles_ref[i] = Rotation.from_matrix(A_IK).as_euler("zxz")

        ax = fig.add_subplot(2, 3, 2)
        ax.plot(t_ref, angles_ref[:, 0], "-r", label="alpha")
        ax.plot(t_ref, angles_ref[:, 1], "-g", label="beta")
        ax.plot(t_ref, angles_ref[:, 2], "-b", label="gamma")
        ax.grid()
        ax.legend()

        # x-y-z trajectory
        ax = fig.add_subplot(2, 3, 3, projection="3d")
        ax.plot3D(
            q_ref[:, 0],
            q_ref[:, 1],
            q_ref[:, 2],
            "-r",
            label="x-y-z trajectory",
        )
        ax.grid()
        ax.legend()

        # x_dot, y_dot, z_dot
        ax = fig.add_subplot(2, 3, 4)
        ax.plot(t_ref, u_ref[:, 0], "-r", label="x_dot")
        ax.plot(t_ref, u_ref[:, 1], "-g", label="y_dot")
        ax.plot(t_ref, u_ref[:, 2], "-b", label="z_dot")
        ax.grid()
        ax.legend()

        # omega_x, omega_y, omega_z
        ax = fig.add_subplot(2, 3, 5)
        ax.plot(t_ref, u_ref[:, 3], "-r", label="omega_x")
        ax.plot(t_ref, u_ref[:, 4], "-g", label="omega_y")
        ax.plot(t_ref, u_ref[:, 5], "-b", label="omega_z")
        ax.grid()
        ax.legend()

        # percussions
        ax = fig.add_subplot(2, 3, 6)
        ax.plot(t_ref, P_N_ref[:, 0], "-r", label="P_N")
        ax.plot(t_ref, P_F_ref[:, 0], "-g", label="P_F0")
        ax.plot(t_ref, P_F_ref[:, 1], "-b", label="P_F1")
        ax.grid()
        ax.legend()

        plt.show()

    # exit()

    # TODO: Adapt bounds
    # def errors(sol, sol_ref, t_transient=0.01, t_longterm=0.05):
    def errors(sol, sol_ref, t_transient=0.5 * t1, t_longterm=0.5 * t1):
        t = sol.t
        q = sol.q
        u = sol.u
        # P_g = sol.P_g
        P_N = sol.P_N
        P_F = sol.P_F

        t_ref = sol_ref.t
        q_ref = sol_ref.q
        u_ref = sol_ref.u
        # P_g_ref = sol_ref.P_g
        P_N_ref = sol_ref.P_N
        P_F_ref = sol_ref.P_F

        # distinguish between transient and long term time steps
        t_idx_transient = np.where(t <= t_transient)[0]
        t_idx_longterm = np.where(t >= t_longterm)[0]

        # compute difference between computed solution and reference solution
        # for identical time instants
        t_ref_idx_transient = np.where(
            np.abs(t[t_idx_transient, None] - t_ref) < 1.0e-8
        )[1]
        t_ref_idx_longterm = np.where(np.abs(t[t_idx_longterm, None] - t_ref) < 1.0e-8)[
            1
        ]

        # differences
        q_transient = q[t_idx_transient]
        u_transient = u[t_idx_transient]
        # P_g_transient = P_g[t_idx_transient]
        P_N_transient = P_N[t_idx_transient]
        P_F_transient = P_F[t_idx_transient]
        diff_transient_q = q_transient - q_ref[t_ref_idx_transient]
        diff_transient_u = u_transient - u_ref[t_ref_idx_transient]
        # diff_transient_P_g = P_g_transient - P_g_ref[t_ref_idx_transient]
        diff_transient_P_N = P_N_transient - P_N_ref[t_ref_idx_transient]
        diff_transient_P_F = P_F_transient - P_F_ref[t_ref_idx_transient]

        q_longterm = q[t_idx_longterm]
        u_longterm = u[t_idx_longterm]
        # P_g_longterm = P_g[t_idx_longterm]
        P_N_longterm = P_N[t_idx_longterm]
        P_F_longterm = P_F[t_idx_longterm]
        diff_longterm_q = q_longterm - q_ref[t_ref_idx_longterm]
        diff_longterm_u = u_longterm - u_ref[t_ref_idx_longterm]
        # diff_longterm_P_g = P_g_longterm - P_g_ref[t_ref_idx_longterm]
        diff_longterm_P_N = P_N_longterm - P_N_ref[t_ref_idx_longterm]
        diff_longterm_P_F = P_F_longterm - P_F_ref[t_ref_idx_longterm]

        # max relative error
        q_error_transient = np.max(
            np.linalg.norm(diff_transient_q, axis=1)
            / np.linalg.norm(q_transient, axis=1)
        )
        u_error_transient = np.max(
            np.linalg.norm(diff_transient_u, axis=1)
            / np.linalg.norm(u_transient, axis=1)
        )
        P_g_error_transient = np.zeros_like(u_error_transient)
        # P_g_error_transient = np.max(
        #     np.linalg.norm(diff_transient_P_g, axis=1)
        #     / np.linalg.norm(P_g_transient, axis=1)
        # )
        P_N_error_transient = np.max(
            np.linalg.norm(diff_transient_P_N, axis=1)
            / np.linalg.norm(P_N_transient, axis=1)
        )
        P_F_error_transient = np.max(
            np.linalg.norm(diff_transient_P_F, axis=1)
            / np.linalg.norm(P_F_transient, axis=1)
        )

        q_error_longterm = np.max(
            np.linalg.norm(diff_longterm_q, axis=1) / np.linalg.norm(q_longterm, axis=1)
        )
        u_error_longterm = np.max(
            np.linalg.norm(diff_longterm_u, axis=1) / np.linalg.norm(u_longterm, axis=1)
        )
        P_g_error_longterm = np.zeros_like(u_error_longterm)
        # P_g_error_longterm = np.max(
        #     np.linalg.norm(diff_longterm_P_g, axis=1)
        #     / np.linalg.norm(P_g_longterm, axis=1)
        # )
        P_N_error_longterm = np.max(
            np.linalg.norm(diff_longterm_P_N, axis=1)
            / np.linalg.norm(P_N_longterm, axis=1)
        )
        P_F_error_longterm = np.max(
            np.linalg.norm(diff_longterm_P_F, axis=1)
            / np.linalg.norm(P_F_longterm, axis=1)
        )

        return (
            q_error_transient,
            u_error_transient,
            P_g_error_transient,
            P_N_error_transient,
            P_F_error_transient,
            q_error_longterm,
            u_error_longterm,
            P_g_error_longterm,
            P_N_error_longterm,
            P_F_error_longterm,
        )

    for i, dt in enumerate(dts):
        print(f"i: {i}, dt: {dt:1.1e}")

        sol = Moreau(system, t1, dt, atol=tol).solve()
        (
            q_errors_transient[0, i],
            u_errors_transient[0, i],
            P_g_errors_transient[0, i],
            P_N_errors_transient[0, i],
            P_F_errors_transient[0, i],
            q_errors_longterm[0, i],
            u_errors_longterm[0, i],
            P_g_errors_longterm[0, i],
            P_N_errors_longterm[0, i],
            P_F_errors_longterm[0, i],
        ) = errors(sol, reference)

        # sol = Solver(system, t1, dt, atol=tol, **kwargs).solve()
        sol = Solver(system, t1, dt, **kwargs).solve()
        (
            q_errors_transient[1, i],
            u_errors_transient[1, i],
            P_g_errors_transient[1, i],
            P_N_errors_transient[1, i],
            P_F_errors_transient[1, i],
            q_errors_longterm[1, i],
            u_errors_longterm[1, i],
            P_g_errors_longterm[1, i],
            P_N_errors_longterm[1, i],
            P_F_errors_longterm[1, i],
        ) = errors(sol, reference)

    # #############################
    # # export errors and dt, dt**2
    # #############################
    # header = "dt, dt2, 2nd, 1st, 1st_GGL"

    # # transient errors
    # export_data = np.vstack((dts, dts**2, *q_errors_transient)).T
    # np.savetxt(
    #     "transient_error_heavy_top_q.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    # export_data = np.vstack((dts, dts**2, *u_errors_transient)).T
    # np.savetxt(
    #     "transient_error_heavy_top_u.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    # export_data = np.vstack((dts, dts**2, *P_N_errors_transient)).T
    # np.savetxt(
    #     "transient_error_heavy_top_la_g.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    # # longterm errors
    # export_data = np.vstack((dts, dts**2, *q_errors_longterm)).T
    # np.savetxt(
    #     "longterm_error_heavy_top_q.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    # export_data = np.vstack((dts, dts**2, *u_errors_longterm)).T
    # np.savetxt(
    #     "longterm_error_heavy_top_u.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    # export_data = np.vstack((dts, dts**2, *P_N_errors_longterm)).T
    # np.savetxt(
    #     "longterm_error_heavy_top_la_g.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    ##################
    # visualize errors
    ##################
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].set_title("transient: Moreau")
    ax[0, 0].loglog(dts, dts, "-k", label="dt")
    ax[0, 0].loglog(dts, dts**2, "--k", label="dt^2")
    # ax[0, 0].loglog(dts, dts**3, "-.k", label="dt^3")
    # ax[0, 0].loglog(dts, dts**4, ":k", label="dt^4")
    ax[0, 0].loglog(dts, q_errors_transient[0], "-.ro", label="q")
    ax[0, 0].loglog(dts, u_errors_transient[0], "-.go", label="u")
    ax[0, 0].loglog(dts, P_g_errors_transient[0], "-.yo", label="P_g")
    ax[0, 0].loglog(dts, P_N_errors_transient[0], "-.bo", label="P_N")
    ax[0, 0].loglog(dts, P_F_errors_transient[0], "-.mo", label="P_F")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].set_title("transient: " + label)
    ax[1, 0].loglog(dts, dts, "-k", label="dt")
    ax[1, 0].loglog(dts, dts**2, "--k", label="dt^2")
    # ax[1, 0].loglog(dts, dts**3, "-.k", label="dt^3")
    # ax[1, 0].loglog(dts, dts**4, ":k", label="dt^4")
    ax[1, 0].loglog(dts, q_errors_transient[1], "-.ro", label="q")
    ax[1, 0].loglog(dts, u_errors_transient[1], "-.go", label="u")
    ax[1, 0].loglog(dts, P_g_errors_transient[1], "-.yo", label="P_g")
    ax[1, 0].loglog(dts, P_N_errors_transient[1], "-.bo", label="P_N")
    ax[1, 0].loglog(dts, P_F_errors_transient[1], "-.mo", label="P_F")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[0, 1].set_title("long term: Moreau")
    ax[0, 1].loglog(dts, dts, "-k", label="dt")
    ax[0, 1].loglog(dts, dts**2, "--k", label="dt^2")
    # ax[0, 1].loglog(dts, dts**3, "-.k", label="dt^3")
    # ax[0, 1].loglog(dts, dts**4, ":k", label="dt^4")
    ax[0, 1].loglog(dts, q_errors_longterm[0], "-.ro", label="q")
    ax[0, 1].loglog(dts, u_errors_longterm[0], "-.go", label="u")
    ax[0, 1].loglog(dts, P_g_errors_longterm[0], "-.yo", label="P_g")
    ax[0, 1].loglog(dts, P_N_errors_longterm[0], "-.bo", label="P_N")
    ax[0, 1].loglog(dts, P_F_errors_longterm[0], "-.mo", label="P_F")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].set_title("long term: " + label)
    ax[1, 1].loglog(dts, dts, "-k", label="dt")
    ax[1, 1].loglog(dts, dts**2, "--k", label="dt^2")
    # ax[1, 1].loglog(dts, dts**3, "-.k", label="dt^3")
    # ax[1, 1].loglog(dts, dts**4, ":k", label="dt^4")
    ax[1, 1].loglog(dts, q_errors_longterm[1], "-.ro", label="q")
    ax[1, 1].loglog(dts, u_errors_longterm[1], "-.go", label="u")
    ax[1, 1].loglog(dts, P_g_errors_longterm[1], "-.yo", label="P_g")
    ax[1, 1].loglog(dts, P_N_errors_longterm[1], "-.bo", label="P_N")
    ax[1, 1].loglog(dts, P_F_errors_longterm[1], "-.mo", label="P_F")
    ax[1, 1].grid()
    ax[1, 1].legend()

    plt.show()


if __name__ == "__main__":
    run()
    # convergence()
