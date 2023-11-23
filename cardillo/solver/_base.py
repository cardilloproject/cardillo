import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import splu
from cardillo.math import (
    prox_sphere,
    prox_R0_nm,
    norm,
    estimate_prox_parameter,
)

from .solver_options import SolverOptions


def consistent_initial_conditions(
    system,
    rtol=1.0e-5,  # TODO: Rename; can they be changed by the user?
    atol=1.0e-8,  # TODO: Rename; can they be changed by the user?
    slice_active_contacts=True,
    options=SolverOptions(),
):
    t0 = system.t0
    q0 = system.q0
    u0 = system.u0

    # normalize quaternions etc.
    q0, u0 = system.step_callback(t0, q0, u0)

    q_dot0 = system.q_dot(t0, q0, u0)

    if not options.compute_consistent_initial_conditions:
        return (
            t0,
            q0,
            u0,
            q_dot0,
            np.zeros(system.nu),
            np.zeros(system.nla_g),
            np.zeros(system.nla_gamma),
            np.zeros(system.nla_c),
            np.zeros(system.nla_N),
            np.zeros(system.nla_F),
        )

    # evaluate constant quantities
    M = system.M(t0, q0)
    h = system.h(t0, q0, u0)

    W_g = system.W_g(t0, q0)
    g_dot_u = system.g_dot_u(t0, q0)
    zeta_g = system.zeta_g(t0, q0, u0)

    W_gamma = system.W_gamma(t0, q0)
    gamma_u = system.gamma_u(t0, q0)
    zeta_gamma = system.zeta_gamma(t0, q0, u0)

    W_c = system.W_c(t0, q0)
    la_c0 = system.la_c(t0, q0, u0)

    # compute constant contact quantities
    g_N = system.g_N(t0, q0)
    g_N_dot = system.g_N_dot(t0, q0, u0)
    A_N = np.isclose(g_N, np.zeros(system.nla_N), rtol, atol)
    B_N = A_N * np.isclose(g_N_dot, np.zeros(system.nla_N), rtol, atol)

    assert np.all(
        np.logical_or(g_N >= 0, A_N)
    ), "Initial conditions do not fulfill g_N0!"
    assert np.all(
        np.logical_or(A_N * g_N_dot >= 0, B_N)
    ), "Initial conditions do not fulfill g_N_dot0!"

    # get set of active normal contacts
    B_N = np.where(B_N)[0]

    # identify active tangent contacts based on active normal contacts and
    # NF-connectivity lists; compute local NF_connectivity
    B_F, global_active_NF_connectivity = compute_I_F(
        B_N, system, slice=slice_active_contacts
    )

    gamma_F = system.gamma_F(t0, q0, u0)[B_F]
    W_N = system.W_N(t0, q0, format="csc")[:, B_N]
    W_F = system.W_F(t0, q0, format="csc")[:, B_F]
    zeta_N = system.g_N_ddot(t0, q0, u0, np.zeros_like(u0))[B_N]
    zeta_F = system.gamma_F_dot(t0, q0, u0, np.zeros_like(u0))[B_F]
    prox_r_N = estimate_prox_parameter(options.prox_scaling, W_N, M)
    prox_r_F = estimate_prox_parameter(options.prox_scaling, W_F, M)

    split_x = np.cumsum(
        [
            system.nu,
            system.nla_g,
            system.nla_gamma,
        ]
    )[:-1]

    # fmt: off
    A = bmat(
        [
            [      M, -W_g, -W_gamma],
            [g_dot_u, None,     None],
            [gamma_u, None,     None],
        ],
        format="csc",
    )
    # fmt: on

    lu = splu(A)

    b0 = np.concatenate(
        [
            h + W_c @ la_c0,
            -zeta_g,
            -zeta_gamma,
        ]
    )

    def prox(x1, la_N, la_F):
        u_dot, _, _ = np.array_split(x1, split_x)

        ##############################
        # fixed-point update Signorini
        ##############################
        g_N_ddot = W_N.T @ u_dot + zeta_N
        prox_arg = prox_r_N * g_N_ddot - la_N
        la_N = -prox_R0_nm(prox_arg)

        #############################
        # fixed-point update friction
        #############################
        gamma_F_dot = W_F.T @ u_dot + zeta_F
        for i_N, i_F, force_recervoir in global_active_NF_connectivity:
            if len(i_N) > 0:
                la_Ni = la_N[i_N]
            else:
                la_Ni = 1.0

            gamma_Fi = gamma_F[i_F]
            if np.isclose(
                norm(gamma_Fi), 0, rtol, atol
            ):  # possible stick on acceleration level
                la_F[i_F] = -force_recervoir.prox(
                    prox_r_F[i_F] * gamma_F_dot[i_F] - la_F[i_F],
                    la_Ni,
                )
            else:  # slip
                la_F[i_F] = -force_recervoir.prox(
                    prox_r_F[i_F] * gamma_Fi - la_F[i_F],
                    la_Ni,
                )

        return la_N, la_F

    x0 = np.zeros(system.nu + system.nla_g + system.nla_gamma)

    # compute accelerations and constraints without contacts
    x0 = lu.solve(b0)

    # initialize zero contact forces
    la_N0 = np.zeros(system.nla_N)
    la_F0 = np.zeros(system.nla_F)
    if len(B_N) > 0:
        # fixed-point loop
        x1 = x0.copy()
        la_N1 = la_N0[B_N].copy()
        la_F1 = la_F0[B_F].copy()
        converged_fixed_point = False
        for i_fixed_point in range(options.fixed_point_max_iter):
            # find proximal point
            la_N1, la_F1 = prox(x1, la_N1, la_F1)

            # compute new rhs
            b = b0.copy()
            b[: system.nu] += W_N @ la_N1 + W_F @ la_F1

            # solve linear system
            x1 = lu.solve(b)

            # convergence in accelerations
            diff = x1[: system.nu] - x0[: system.nu]

            error_fixed_point = np.max(np.absolute(diff))

            converged_fixed_point = error_fixed_point < options.fixed_point_atol
            if converged_fixed_point:
                la_N0[B_N] = la_N1
                la_F0[B_F] = la_F1
                break
            else:
                # update values
                x0 = x1.copy()

        assert (
            converged_fixed_point
        ), f"Solving for consistent initial conditions does not converge after {i_fixed_point} fixed-point iterations with error {error_fixed_point}."
        print(
            f"consistent_initial_conditions converged after {i_fixed_point} fixed-point iterations with error: {error_fixed_point}"
        )

    u_dot0, la_g0, la_gamma0 = np.array_split(x0, split_x)

    # check if initial conditions satisfy constraints on position, velocity
    # and acceleration level
    g0 = system.g(t0, q0)
    g_dot0 = system.g_dot(t0, q0, u0)
    g_ddot0 = system.g_ddot(t0, q0, u0, u_dot0)
    gamma0 = system.gamma(t0, q0, u0)
    gamma_dot0 = system.gamma_dot(t0, q0, u0, u_dot0)
    g_S0 = system.g_S(t0, q0)

    assert np.allclose(
        g0, np.zeros(system.nla_g), rtol, atol
    ), "Initial conditions do not fulfill g0!"
    assert np.allclose(
        g_dot0, np.zeros(system.nla_g), rtol, atol
    ), "Initial conditions do not fulfill g_dot0!"
    assert np.allclose(
        g_ddot0, np.zeros(system.nla_g), rtol, atol
    ), "Initial conditions do not fulfill g_ddot0!"
    assert np.allclose(
        gamma0, np.zeros(system.nla_gamma), rtol, atol
    ), "Initial conditions do not fulfill gamma0!"
    assert np.allclose(
        gamma_dot0, np.zeros(system.nla_gamma), rtol, atol
    ), "Initial conditions do not fulfill gamma_dot0!"
    assert np.allclose(
        g_S0, np.zeros(system.nla_S), rtol, atol
    ), "Initial conditions do not fulfill g_S0!"

    return t0, q0, u0, q_dot0, u_dot0, la_g0, la_gamma0, la_c0, la_N0, la_F0


def compute_I_F(I_N, system, slice=True):
    """Compute set of active friction contacts based on active normal contacts
    and NF-connectivity list."""
    # compute set of active normal contacts if boolean array is given
    # TODO: This should not be required in future.
    if I_N.dtype == bool:
        I_N = np.where(I_N)[0]

    active_normal_contacts = len(I_N) > 0

    # compute set of active friction contacts and local connectivity
    I_F = []
    global_active_NF_connectivity = []
    nla_N = 0
    nla_F = 0
    for contr in system._System__gamma_F_contr:
        for i_N, i_F, force_reservoir in contr.NF_connectivity2:
            i_F_global = np.array(i_F, dtype=int) + nla_F

            if len(i_N) > 0:  # normal force dependence
                i_N_global = np.array(i_N, dtype=int) + nla_N
                # only add friction if normal force is active
                if not slice or (active_normal_contacts and len(I_N[i_N_global]) > 0):
                    I_F.extend(i_F_global)
                    global_active_NF_connectivity.append(
                        (i_N_global, i_F_global, force_reservoir)
                    )

            else:  # no normal force dependence
                I_F.extend(i_F_global)
                global_active_NF_connectivity.append(([], i_F_global, force_reservoir))

        if hasattr(contr, "nla_N"):
            nla_N += contr.nla_N
        nla_F += contr.nla_F

    return np.array(I_F, dtype=int), global_active_NF_connectivity
