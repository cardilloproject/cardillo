import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import splu

from cardillo.math.prox import NegativeOrthant, estimate_prox_parameter
from cardillo.math.algebra import norm
from cardillo.definitions import IS_CLOSE_ATOL
from .solver_options import SolverOptions


def consistent_initial_conditions(
    system,
    slice_active_contacts=True,
    options=SolverOptions(),
):
    """Checks consistency of initial conditions with constraints on position and velocity level and finds initial accelerations and constraint/contact forces.

    Parameters
    ----------
    system : cardillo.System
        System for which the consistent initial conditions are computed.
    slice_active_contacts : bool
        Slice friction forces to contemplate only those corresponding to active normal contact.
    options : cardillo.solver.SolverOptions
        Solver options for the computations of the constraint/contact forces.
    """
    t0 = system.t0
    q0 = system.q0
    u0 = system.u0

    # normalize quaternions etc.
    q0, u0 = system.step_callback(t0, q0, u0)

    q_dot0 = system.q_dot(t0, q0, u0)

    if (
        not options.compute_consistent_initial_conditions or system.nu == 0
    ):  # second case can happen during debugging, when only frames are added to the system
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
    A_N = np.isclose(g_N, np.zeros(system.nla_N), atol=IS_CLOSE_ATOL)
    B_N = A_N * np.isclose(g_N_dot, np.zeros(system.nla_N), atol=IS_CLOSE_ATOL)

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
    B_F, global_active_friction_laws = compute_I_F(
        B_N, system, slice=slice_active_contacts
    )
    # TODO: Is there a case where slice=False??

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
        la_N = -NegativeOrthant.prox(prox_r_N * g_N_ddot - la_N)

        #############################
        # fixed-point update friction
        #############################
        gamma_F_dot = W_F.T @ u_dot + zeta_F
        for i_N, i_F, force_recervoir in global_active_friction_laws:
            if len(i_N) > 0:
                la_Ni = la_N[i_N]
            else:
                la_Ni = 1.0

            gamma_Fi = gamma_F[i_F]
            if np.isclose(
                norm(gamma_Fi), 0, atol=IS_CLOSE_ATOL
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
        g0, np.zeros(system.nla_g), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill g0!"
    assert np.allclose(
        g_dot0, np.zeros(system.nla_g), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill g_dot0!"
    assert np.allclose(
        g_ddot0, np.zeros(system.nla_g), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill g_ddot0!"
    assert np.allclose(
        gamma0, np.zeros(system.nla_gamma), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill gamma0!"
    assert np.allclose(
        gamma_dot0, np.zeros(system.nla_gamma), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill gamma_dot0!"
    assert np.allclose(
        g_S0, np.zeros(system.nla_S), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill g_S0!"

    return t0, q0, u0, q_dot0, u_dot0, la_g0, la_gamma0, la_c0, la_N0, la_F0


def compute_I_F(I_N, system, slice=True):
    """Compute set of active friction contacts based on active normal contacts
    and NF-connectivity list."""

    # compute set of active friction contacts and local connectivity
    I_F = []
    global_active_friction_laws = []
    nla_F_local = 0
    for contr in system.get_contribution_list("gamma_F"):
        for i_N, i_F, force_reservoir in contr.friction_laws:
            i_F_global = contr.la_FDOF[i_F]
            n_F = len(i_F)
            i_F_local = np.arange(n_F) + nla_F_local

            if len(i_N) > 0:  # normal force dependence
                i_N_global = contr.la_NDOF[i_N][0]  # scalar normal force

                # only add friction if normal force is active
                if not slice or (i_N_global in I_N):
                    nla_F_local += n_F
                    i_N_local = np.where(i_N_global == I_N)[0]
                    I_F.extend(i_F_global)
                    global_active_friction_laws.append(
                        (i_N_local, i_F_local, force_reservoir)
                    )

            else:  # no normal force dependence
                nla_F_local += n_F
                I_F.extend(i_F_global)
                global_active_friction_laws.append(([], i_F_local, force_reservoir))

    return np.array(I_F, dtype=int), global_active_friction_laws
