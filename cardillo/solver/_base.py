import numpy as np
from scipy.sparse import csr_matrix, bmat, csc_matrix, lil_matrix, diags
from cardillo.math import prox_sphere, prox_R0_nm, fsolve, norm, approx_fprime


def consistent_initial_conditions(
    system,
    rtol=1.0e-5,
    atol=1.0e-8,
    newton_atol=1e-10,
    newton_max_iter=10,
    jac=None,
):
    t0 = system.t0
    q0 = system.q0
    u0 = system.u0

    # normalize quaternions etc.
    system.step_callback(t0, q0, u0)

    q_dot0 = system.q_dot(t0, q0, u0)

    # # TODO: Remove me!
    # return (
    #     t0,
    #     q0,
    #     u0,
    #     q_dot0,
    #     np.zeros(system.nu),
    #     np.zeros(system.nla_g),
    #     np.zeros(system.nla_gamma),
    #     np.zeros(system.nla_N),
    #     np.zeros(system.nla_F),
    # )

    g_N = system.g_N(t0, q0)
    g_N_dot = system.g_N_dot(t0, q0, u0)
    I_N = np.isclose(g_N, np.zeros(system.nla_N), rtol, atol)
    B_N = I_N * np.isclose(g_N_dot, np.zeros(system.nla_N), rtol, atol)

    assert np.all(g_N >= 0) or np.allclose(
        g_N, np.zeros(system.nla_N), rtol, atol
    ), "Initial conditions do not fulfill g_N0!"
    assert np.all(g_N_dot[I_N] >= 0) or np.allclose(
        g_N_dot[I_N], np.zeros(system.nla_N), rtol, atol
    ), "Initial conditions do not fulfill g_N_dot0!"

    M = system.M(t0, q0, scipy_matrix=csr_matrix)
    h = system.h(t0, q0, u0)
    W_g = system.W_g(t0, q0, scipy_matrix=csr_matrix)
    W_gamma = system.W_gamma(t0, q0, scipy_matrix=csr_matrix)
    W_N = system.W_N(t0, q0, scipy_matrix=csr_matrix)
    W_F = system.W_F(t0, q0, scipy_matrix=csr_matrix)
    gamma_F = system.gamma_F(t0, q0, u0)

    prox_r_N = system.prox_r_N(t0, q0)
    prox_r_F = system.prox_r_F(t0, q0)
    mu = system.mu

    split = np.cumsum(
        [
            system.nu,
            system.nla_g,
            system.nla_gamma,
            system.nla_N,
        ]
    )

    global C_N
    C_N = np.zeros(system.nla_N, dtype=bool)

    def _R_F(x):
        u_dot, _, _, la_N, la_F = np.array_split(x, split)
        gamma_F_dot = system.gamma_F_dot(t0, q0, u0, u_dot)
        R_la_F = np.zeros_like(la_F)

        for i_N, i_F in enumerate(system.NF_connectivity):
            i_F = np.array(i_F)
            n_F = len(i_F)
            if n_F > 0:
                gamma_Fi = gamma_F[i_F]
                norm_gamma_Fi = norm(gamma_Fi)
                if norm_gamma_Fi > 0:
                    # slip
                    R_la_F[i_F] = la_F[i_F] + prox_sphere(
                        prox_r_F[i_F] * gamma_Fi - la_F[i_F],
                        mu[i_N] * la_N[i_N],
                    )
                else:
                    # possibly stick
                    R_la_F[i_F] = la_F[i_F] + prox_sphere(
                        prox_r_F[i_F] * gamma_F_dot[i_F] - la_F[i_F],
                        mu[i_N] * la_N[i_N],
                    )

                    # arg_F = prox_r_F[i_F] * gamma_F_dot[i_F] - la_F[i_F]
                    # norm_arg_F = norm(arg_F)
                    # radius = mu[i_N] * la_N[i_N]
                    # if norm_arg_F < radius:
                    #     R[split[3] + i_F] = gamma_F_dot[i_F]
                    # else:
                    #     if norm_arg_F > 0:
                    #         R[split[3] + i_F] = la_F[i_F] + radius * arg_F / norm_arg_F
                    #     else:
                    #         R[split[3] + i_F] = la_F[i_F]
        return R_la_F

    def R(x, update_index=False):
        u_dot, la_g, la_gamma, la_N, la_F = np.array_split(x, split)

        R = np.zeros_like(x)

        #####################
        # equations of motion
        #####################
        R[: split[0]] = (
            M @ u_dot - h - W_g @ la_g - W_gamma @ la_gamma - W_N @ la_N - W_F @ la_F
        )

        #############################################
        # bilateral constraints on acceleration level
        #############################################
        R[split[0] : split[1]] = system.g_ddot(t0, q0, u0, u_dot)
        R[split[1] : split[2]] = system.gamma_dot(t0, q0, u0, u_dot)

        #################################
        # Signorini on acceleration level
        #################################
        g_N_ddot = system.g_N_ddot(t0, q0, u0, u_dot)
        prox_arg = prox_r_N * g_N_ddot - la_N
        global C_N
        if update_index:
            C_N = B_N * (prox_arg <= 0)
        R[split[2] : split[3]] = np.where(C_N, g_N_ddot, la_N)

        ################################
        # friction on acceleration level
        ################################
        R[split[3] :] = _R_F(x)

        return R

    def J(x, *args, **kwargs):
        global C_N
        # TODO: Sparse matrix matrix or sparse matrix slicing?
        Rla_N_u_dot = diags(C_N.astype(float)) @ W_N.T
        # Rla_N_u_dot = lil_matrix((system.nla_N, system.nu))
        # Rla_N_u_dot[C_N] = W_N.T[C_N]
        Rla_N_la_N = diags((~C_N).astype(float))

        Rla_F_u_dot, _, _, Rla_F_la_N, Rla_F_la_F = np.array_split(
            approx_fprime(x, lambda x: _R_F(x)), split, axis=1
        )

        # fmt: off
        _J = bmat(
            [
                [          M, -W_g, -W_gamma,       -W_N,       -W_F],
                [      W_g.T, None,     None,       None,       None],
                [  W_gamma.T, None,     None,       None,       None],
                [Rla_N_u_dot, None,     None, Rla_N_la_N,       None],
                [Rla_F_u_dot, None,     None, Rla_F_la_N, Rla_F_la_F],
            ],
            format="csc",
        )
        # fmt: on

        return _J

        J_num = csc_matrix(approx_fprime(x, R, method="3-point", eps=1.0e-6))
        diff = (_J - J_num).toarray()
        error = np.linalg.norm(diff)
        print(f"error Jacobian: {error}")
        return J_num

    x0 = np.zeros(
        system.nu + system.nla_g + system.nla_gamma + system.nla_N + system.nla_F
    )
    if jac is None:
        jac = J
    else:
        jac = "3-point"
    x0, converged, error, i, f = fsolve(
        R,
        x0,
        atol=newton_atol,
        max_iter=newton_max_iter,
        jac=jac,
        fun_args=(True,),
        jac_args=(False,),
    )
    assert (
        converged
    ), "Solving for consistent initial conditions does not converge after {i} iterations with error {error}."
    u_dot0, la_g0, la_gamma0, la_N0, la_F0 = np.array_split(x0, split)

    # check if initial conditions satisfy constraints on position, velocity
    # and acceleration level
    g0 = system.g(t0, q0)
    g_dot0 = system.g_dot(t0, q0, u0)
    g_ddot0 = system.g_ddot(t0, q0, u0, u_dot0)
    gamma0 = system.gamma(t0, q0, u0)
    gamma_dot0 = system.gamma_dot(t0, q0, u0, u_dot0)
    g_S0 = system.g_S(t0, q0)

    g_N_ddot = system.g_N_ddot(t0, q0, u0, u_dot0)
    assert np.all(g_N_ddot >= 0) or np.allclose(
        B_N * g_N_ddot, np.zeros(system.nla_N), rtol, atol
    ), "Initial conditions do not fulfill g_N_ddot0!"

    assert np.allclose(
        g0, np.zeros(system.nla_g), rtol, atol
    ), "Initial conditions do not fulfill g0!"
    # assert np.allclose(
    #     g_dot0, np.zeros(system.nla_g), rtol, atol
    # ), "Initial conditions do not fulfill g_dot0!"
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

    return t0, q0, u0, q_dot0, u_dot0, la_g0, la_gamma0, la_N0, la_F0


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
