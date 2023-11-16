import numpy as np
from scipy.sparse import csr_array, coo_array, bmat
from cardillo.math import prox_R0_nm, prox_sphere, fsolve, norm


def consistent_initial_conditions(
    system,
    rtol=1.0e-5,
    atol=1.0e-8,
    newton_atol=1e-10,
    newton_max_iter=10,
    fixed_point_atol=1e-8,
    fixed_point_max_iter=int(1e3),
):
    t0 = system.t0
    q0 = system.q0
    u0 = system.u0

    # normalize quaternions etc.
    system.step_callback(t0, q0, u0)

    q_dot0 = system.q_dot(t0, q0, u0)

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
    M = system.M(t0, q0, scipy_matrix=csr_array)
    h = system.h(t0, q0, u0)
    W_g = system.W_g(t0, q0, scipy_matrix=csr_array)
    g_dot_u = system.g_dot_u(t0, q0, u0, scipy_matrix=csr_array)
    zeta_g = system.zeta_g(t0, q0, u0)
    W_gamma = system.W_gamma(t0, q0, scipy_matrix=csr_array)
    gamma_u = system.gamma_u(t0, q0, u0, scipy_matrix=csr_array)
    zeta_gamma = system.zeta_gamma(t0, q0, u0)
    W_c = system.W_c(t0, q0, scipy_matrix=csr_array)
    W_N = system.W_N(t0, q0, scipy_matrix=csr_array)
    W_F = system.W_F(t0, q0, scipy_matrix=csr_array)

    prox_r_N = system.prox_r_N(t0, q0)
    prox_r_F = system.prox_r_F(t0, q0)
    mu = system.mu

    split_x = np.cumsum(
        [
            system.nu,
            system.nla_g,
            system.nla_gamma,
            system.nla_c,
        ]
    )[:-1]
    split_y = np.cumsum(
        [
            system.nla_N,
            system.nla_F,
        ]
    )[:-1]

    def R_x(x, y):
        u_dot, la_g, la_gamma, la_c = np.array_split(x, split_x)
        la_N, la_F = np.array_split(y, split_y)

        R_x = np.zeros_like(x)

        #####################
        # equations of motion
        #####################
        R_x[: split_x[0]] = (
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
        R_x[split_x[0] : split_x[1]] = g_dot_u @ u_dot + zeta_g
        R_x[split_x[1] : split_x[2]] = gamma_u @ u_dot + zeta_gamma

        ############
        # compliance
        ############
        R_x[split_x[2] :] = system.c(t0, q0, u0, la_c)

        return R_x

    def J_x(x, y):
        la_c = x[split_x[2] :]
        # coo for fast bmat
        c_la_c = system.c_la_c(t0, q0, u0, la_c, scipy_matrix=coo_array)

        # assemble jacobian
        # fmt: off
        J = bmat(
            [
                [        M, -W_g, -W_gamma,   -W_c],
                [  g_dot_u, None,     None,   None],
                [  gamma_u, None,     None,   None],
                [     None, None,     None, c_la_c],
            ],
            format="csc",
        )
        # fmt: on

        return J

        # from cardillo.math import approx_fprime
        # J_num = csr_array(approx_fprime(x, lambda x: R_x(x, y)))
        # diff = (J - J_num).toarray()
        # error_fixed_point = np.linalg.norm(diff)
        # print(f"error Jacobian: {error_fixed_point}")
        # return J_num

    def prox(x1, y0):
        u_dot, _, _, _ = np.array_split(x1, split_x)
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

    x0 = np.zeros(system.nu + system.nla_g + system.nla_gamma + system.nla_c)
    y0 = np.zeros(system.nla_N + system.nla_F)

    # fixed-point loop
    x1 = x0.copy()
    y1 = y0.copy()
    converged_fixed_point = False
    for i_fixed_point in range(fixed_point_max_iter):
        # find proximal point
        y1 = prox(x1, y1)

        # solve nonlinear system
        x1, converged_newton, error_newton, i_newton, _ = fsolve(
            R_x,
            x1,
            jac=J_x,
            fun_args=(y1,),
            jac_args=(y1,),
            atol=newton_atol,
            max_iter=newton_max_iter,
        )
        assert (
            converged_newton
        ), f"Newton method in consistent_initial_conditions did not converge after {i_newton} iterations with error: {error_newton}"

        # convergence in accelerations
        diff = x1[: system.nu] - x0[: system.nu]

        error_fixed_point = np.max(np.absolute(diff))

        converged_fixed_point = error_fixed_point < fixed_point_atol
        if converged_fixed_point:
            break
        else:
            # update values
            x0 = x1.copy()
            y0 = y1.copy()

    assert (
        converged_fixed_point
    ), f"Solving for consistent initial conditions does not converge after {i_fixed_point} iterations with error {error_fixed_point}."
    print(
        f"consistent_initial_conditions converged after {i_fixed_point} iterations with error: {error_fixed_point}"
    )

    u_dot0, la_g0, la_gamma0, la_c0 = np.array_split(x1, split_x)
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
