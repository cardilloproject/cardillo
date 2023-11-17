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


# TODO: Add rtol atol error measure
def consistent_initial_conditions(
    system,
    rtol=1.0e-5,  # TODO: Rename; can they be changed by the user?
    atol=1.0e-8,  # TODO: Rename; can they be changed by the user?
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

    g_N = system.g_N(t0, q0)
    g_N_dot = system.g_N_dot(t0, q0, u0)
    gamma_F = system.gamma_F(t0, q0, u0)
    A_N = np.isclose(g_N, np.zeros(system.nla_N), rtol, atol)
    B_N = A_N * np.isclose(g_N_dot, np.zeros(system.nla_N), rtol, atol)

    assert np.all(
        np.logical_or(g_N >= 0, A_N)
    ), "Initial conditions do not fulfill g_N0!"
    assert np.all(
        np.logical_or(A_N * g_N_dot >= 0, B_N)
    ), "Initial conditions do not fulfill g_N_dot0!"

    # csr for fast matrix vector product
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
    W_N = system.W_N(t0, q0)
    W_F = system.W_F(t0, q0)
    prox_r_N = estimate_prox_parameter(options.prox_scaling, W_N, M)
    prox_r_F = estimate_prox_parameter(options.prox_scaling, W_F, M)
    mu = system.mu

    split_x = np.cumsum(
        [
            system.nu,
            system.nla_g,
            system.nla_gamma,
        ]
    )[:-1]
    split_y = np.cumsum(
        [
            system.nla_N,
            system.nla_F,
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

    def prox(x1, y0):
        u_dot, _, _ = np.array_split(x1, split_x)
        la_N, la_F = np.array_split(y0, split_y)

        y1 = np.zeros_like(y0)

        ##############################
        # fixed-point update Signorini
        ##############################
        g_N_ddot = system.g_N_ddot(t0, q0, u0, u_dot)
        prox_arg = prox_r_N * g_N_ddot - la_N
        # TODO: What do you prefer?
        # y1[: split_y[0]] = np.where(B_N, -prox_R0_nm(prox_arg), np.zeros_like(la_N))
        y1[: split_y[0]] = B_N * (-prox_R0_nm(prox_arg))

        #############################
        # fixed-point update friction
        #############################
        gamma_F_dot = system.gamma_F_dot(t0, q0, u0, u_dot)
        for i_N, i_F in enumerate(system.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                if B_N[i_N]:  # active normal contact
                    norm_gamma_Fi = norm(gamma_F[i_F])
                    if np.isclose(
                        norm_gamma_Fi, 0, rtol, atol
                    ):  # possible stick on acceleration level
                        y1[split_y[0] + i_F] = prox_sphere(
                            prox_r_F[i_F] * gamma_F_dot[i_F] - la_F[i_F],
                            mu[i_N] * la_N[i_N],
                        )
                    else:  # slip
                        y1[split_y[0] + i_F] = (
                            -mu[i_N] * la_N[i_N] * gamma_F[i_F] / norm_gamma_Fi
                        )
                else:  # open normal contact
                    y1[split_y[0] + i_F] = np.zeros_like(i_F)

        return y1

    x0 = np.zeros(system.nu + system.nla_g + system.nla_gamma)
    y0 = np.zeros(system.nla_N + system.nla_F)

    # fixed-point loop
    x1 = x0.copy()
    y1 = y0.copy()
    converged_fixed_point = False
    for i_fixed_point in range(options.fixed_point_max_iter):
        # find proximal point
        y1 = prox(x1, y1)

        # compute new rhs
        la_N, la_F = np.array_split(y1, split_y)
        b = b0.copy()
        b[: system.nu] += W_N @ la_N + W_F @ la_F

        # solve linear system
        x1 = lu.solve(b)

        # convergence in accelerations
        diff = x1[: system.nu] - x0[: system.nu]

        error_fixed_point = np.max(np.absolute(diff))

        converged_fixed_point = error_fixed_point < options.fixed_point_atol
        if converged_fixed_point:
            break
        else:
            # update values
            x0 = x1.copy()
            y0 = y1.copy()

    assert (
        converged_fixed_point
    ), f"Solving for consistent initial conditions does not converge after {i_fixed_point} fixed-point iterations with error {error_fixed_point}."
    print(
        f"consistent_initial_conditions converged after {i_fixed_point} fixed-point iterations with error: {error_fixed_point}"
    )

    u_dot0, la_g0, la_gamma0 = np.array_split(x1, split_x)
    la_N0, la_F0 = np.array_split(y1, split_y)

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


def compute_I_F(I_N, NF_connectivity):
    """identify active tangent contacts based on active normal contacts and
    NF-connectivity lists"""
    if np.any(I_N):
        I_F = np.array(
            [c for i, I_N_i in enumerate(I_N) for c in NF_connectivity[i] if I_N_i],
            dtype=int,
        )
    else:
        I_F = np.array([], dtype=int)

    return I_F
