import numpy as np
from scipy.sparse import csr_array, coo_array, lil_array, eye, diags, bmat
from cardillo.math import prox_sphere, prox_R0_nm, fsolve, norm, approx_fprime, prox_r


def consistent_initial_conditions(
    system,
    rtol=1.0e-5,
    atol=1.0e-8,
    newton_atol=1e-10,
    newton_max_iter=10,
    # jac=None,
    jac="2-point",
    error_function=lambda x: np.max(np.absolute(x)),
    alpha=1,
):
    t0 = system.t0
    q0 = system.q0
    u0 = system.u0

    # normalize quaternions etc.
    system.step_callback(t0, q0, u0)

    q_dot0 = system.q_dot(t0, q0, u0)

    # import warnings

    # warnings.warn("Wrong initial conditions are used!")
    # return (
    #     t0,
    #     q0,
    #     u0,
    #     q_dot0,
    #     np.zeros(system.nu),
    #     np.zeros(system.nla_g),
    #     np.zeros(system.nla_gamma),
    #     np.zeros(system.nla_c),
    #     np.zeros(system.nla_N),
    #     np.zeros(system.nla_F),
    # )

    g_N = system.g_N(t0, q0)
    g_N_dot = system.g_N_dot(t0, q0, u0)
    gamma_F = system.gamma_F(t0, q0, u0)
    A_N = np.isclose(g_N, np.zeros(system.nla_N), rtol, atol)
    B_N = A_N * np.isclose(g_N_dot, np.zeros(system.nla_N), rtol, atol)
    global C_N  # normal contact on acceleration level
    C_N = np.zeros(system.nla_N, dtype=bool)
    global slip_vel  # slip on velocity level
    slip_vel = np.zeros(system.nla_N, dtype=bool)
    global slip_acc  # slip on acceleration level
    slip_acc = np.zeros(system.nla_N, dtype=bool)
    global stick_acc  # stick on acceleration level
    stick_acc = np.zeros(system.nla_N, dtype=bool)

    assert np.all(
        np.logical_or(g_N >= 0, A_N)
    ), "Initial conditions do not fulfill g_N0!"
    assert np.all(
        np.logical_or(A_N * g_N_dot >= 0, B_N)
    ), "Initial conditions do not fulfill g_N_dot0!"

    # csr for fast matrix vector product
    M = system.M(t0, q0, scipy_matrix=csr_array)
    h = system.h(t0, q0, u0)
    W_g = system.W_g(t0, q0, scipy_matrix=csr_array)
    zeta_g = system.g_ddot(t0, q0, u0, np.zeros(system.nu))
    W_gamma = system.W_gamma(t0, q0, scipy_matrix=csr_array)
    zeta_gamma = system.gamma_dot(t0, q0, u0, np.zeros(system.nu))
    W_c = system.W_c(t0, q0, scipy_matrix=csr_array)
    W_N = system.W_N(t0, q0, scipy_matrix=csr_array)
    zeta_N = system.g_N_ddot(t0, q0, u0, np.zeros(system.nu))
    W_F = system.W_F(t0, q0, scipy_matrix=csr_array)
    zeta_F = system.gamma_F_dot(t0, q0, u0, np.zeros(system.nu))
    I_N = np.isclose(g_N, np.zeros(system.nla_N), rtol, atol)
    prox_r_N = prox_r(alpha, W_N[:, I_N], M)
    I_F = compute_I_F(I_N, system.NF_connectivity)
    prox_r_F = prox_r(alpha, W_F[:, I_F], M)
    mu = system.mu

    split = np.cumsum(
        [
            system.nu,
            system.nla_g,
            system.nla_gamma,
            system.nla_c,
            system.nla_N,
        ]
    )

    def _R_F(x):
        u_dot, _, _, _, la_N, la_F = np.array_split(x, split)
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
        u_dot, la_g, la_gamma, la_c, la_N, la_F = np.array_split(x, split)

        R = np.zeros_like(x)

        #####################
        # equations of motion
        #####################
        R[: split[0]] = (
            M @ u_dot
            - h
            - W_g @ la_g
            - W_gamma @ la_gamma
            - W_c @ la_c
            - W_N @ la_N
            - W_F @ la_F
        )

        #############################################
        # bilateral constraints on acceleration level
        #############################################
        R[split[0] : split[1]] = W_g.T @ u_dot + zeta_g
        R[split[1] : split[2]] = W_gamma.T @ u_dot + zeta_gamma

        ############
        # compliance
        ############
        R[split[2] : split[3]] = system.c(t0, q0, u0, la_c)

        #################################
        # Signorini on acceleration level
        #################################
        g_N_ddot = W_N.T @ u_dot + zeta_N
        prox_arg = prox_r_N * g_N_ddot - la_N
        global C_N
        if update_index:
            C_N = B_N * (prox_arg <= 0)
        R[split[3] : split[4]] = np.where(C_N, g_N_ddot, la_N)

        ################################
        # friction on acceleration level
        ################################
        # R[split[4] :] = la_F
        gamma_F_dot = W_F.T @ u_dot + zeta_F

        global D_N, E_N
        D_N = np.zeros(system.nla_N, dtype=bool)
        E_N = np.zeros(system.nla_N, dtype=bool)
        # global D_N  # stick on acceleration level
        # global E_N  # slip on velocity level
        for i_N, i_F in enumerate(system.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                if B_N[i_N]:  # active normal contact
                    norm_gamma_Fi = norm(gamma_F[i_F])
                    if np.isclose(
                        norm_gamma_Fi, 0, rtol, atol
                    ):  # possible stick on acceleration level
                        prox_arg = prox_r_F[i_F] * gamma_F_dot[i_F] - la_F[i_F]
                        norm_prox_arg = norm(prox_arg)
                        radius = mu[i_N] * la_N[i_N]
                        if norm_prox_arg <= radius:  # stick on acceleration level
                            stick_acc[i_N] = True
                            R[split[4] + i_F] = gamma_F_dot[i_F]
                        else:  # slip on acceleration level
                            slip_acc[i_N] = True
                            R[split[4] + i_F] = (
                                la_F[i_F]
                                + mu[i_N] * la_N[i_N] * prox_arg / norm_prox_arg
                            )
                    else:  # slip
                        slip_vel[i_N] = True
                        R[split[4] + i_F] = (
                            la_F[i_F]
                            + mu[i_N] * la_N[i_N] * gamma_F[i_F] / norm_gamma_Fi
                        )
                else:  # open normal contact
                    R[split[4] + i_F] = la_F[i_F]

        return R

    def J(x, *args, **kwargs):
        # return csc_matrix(approx_fprime(x, R, method="3-point", eps=1.0e-6))

        # Rla_F_u_dot, _, _, _, Rla_F_la_N, Rla_F_la_F = np.array_split(
        #     approx_fprime(x, lambda x: _R_F(x)),
        #     split,
        #     axis=0
        #     # approx_fprime(x, lambda x: _R_F(x)),
        #     # split,
        #     # axis=0,
        # )

        # coo for fast bmat
        c_la_c = system.c_la_c(
            t0, q0, u0, x[split[2] : split[3]], scipy_matrix=coo_array
        )

        global C_N
        J_N_u_dot = csr_array((system.nla_N, system.nu))
        J_N_u_dot[C_N] = W_N.T[C_N]
        J_N_la_N = diags((~C_N).astype(float))
        # eye_N = eye(system.nla_N, format="coo")

        eye_F = eye(system.nla_F, format="coo")

        # assemble jacobian
        # fmt: off
        J = bmat(
            [
                [          M, -W_g, -W_gamma,   -W_c,     -W_N,  -W_F],
                [      W_g.T, None,     None,   None,     None,  None],
                [  W_gamma.T, None,     None,   None,     None,  None],
                [       None, None,     None, c_la_c,     None,  None],
                [  J_N_u_dot, None,     None,   None, J_N_la_N,  None],
                [       None, None,     None,   None,     None, eye_F],
                # [Rla_F_u_dot, None,     None,   None, Rla_F_la_N, Rla_F_la_F],
            ],
            format="csc",
        )
        # fmt: on

        return J

        J_num = csc_matrix(approx_fprime(x, R, method="3-point", eps=1.0e-6))
        diff = (J - J_num).toarray()
        error = np.linalg.norm(diff)
        print(f"error Jacobian: {error}")
        return J_num

    x0 = np.zeros(
        system.nu
        + system.nla_g
        + system.nla_gamma
        + system.nla_c
        + system.nla_N
        + system.nla_F
    )
    if jac is None:
        jac = J

    x0, converged, error, i, _ = fsolve(
        R,
        x0,
        atol=newton_atol,
        max_iter=newton_max_iter,
        jac=jac,
        fun_args=(True,),
        jac_args=(False,),
        error_function=error_function,
    )
    assert (
        converged
    ), "Solving for consistent initial conditions does not converge after {i} iterations with error {error}."
    print(
        f"consistent_initial_conditions converged after {i + 1} iterations with error: {error}"
    )

    u_dot0, la_g0, la_gamma0, la_c0, la_N0, la_F0 = np.array_split(x0, split)

    # check if initial conditions satisfy constraints on position, velocity
    # and acceleration level
    g0 = system.g(t0, q0)
    g_dot0 = system.g_dot(t0, q0, u0)
    g_ddot0 = system.g_ddot(t0, q0, u0, u_dot0)
    gamma0 = system.gamma(t0, q0, u0)
    gamma_dot0 = system.gamma_dot(t0, q0, u0, u_dot0)
    g_S0 = system.g_S(t0, q0)

    # g_N_ddot = system.g_N_ddot(t0, q0, u0, u_dot0)
    # assert np.all(g_N_ddot >= 0) or np.allclose(
    #     B_N * g_N_ddot, np.zeros(system.nla_N), rtol, atol
    # ), "Initial conditions do not fulfill g_N_ddot0!"

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
