import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, csc_matrix, bmat
from tqdm import tqdm

from cardillo.math.prox import prox_R0_np, prox_sphere
from cardillo.math import approx_fprime
from cardillo.solver import Solution


class NonsmoothPartitionedHalfExplicitEuler:
    def __init__(
        self,
        model,
        t1,
        dt,
        atol=1e-6,
        max_iter=50,
        error_function=lambda x: np.max(np.abs(x)),
    ):
        self.model = model

        # integration time
        t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.error_function = error_function
        self.atol = atol
        self.max_iter = max_iter

        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = model.nla_g
        self.nla_gamma = model.nla_gamma
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F

        # initial state
        self.tk = t0
        self.qk = model.q0
        self.uk = model.u0

        # initial accelerations
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        def consistent_initial_values(t0, q0, u0):
            """compute physically consistent initial values"""

            # initial velocites
            q_dot0 = self.model.q_dot(t0, q0, u0)

            # solve for consistent initial accelerations and Lagrange mutlipliers
            M0 = self.model.M(t0, q0, scipy_matrix=csr_matrix)
            h0 = self.model.h(t0, q0, u0)
            W_g0 = self.model.W_g(t0, q0, scipy_matrix=csr_matrix)
            W_gamma0 = self.model.W_gamma(t0, q0, scipy_matrix=csr_matrix)
            zeta_g0 = self.model.zeta_g(t0, q0, u0)
            zeta_gamma0 = self.model.zeta_gamma(t0, q0, u0)
            # fmt: off
            A = bmat(
                [
                    [        M0, -W_g0, -W_gamma0],
                    [    W_g0.T,  None,      None],
                    [W_gamma0.T,  None,      None],
                ],
                format="csc",
            )
            # fmt: on
            b = np.concatenate([h0, -zeta_g0, -zeta_gamma0])
            u_dot_la_g_la_gamma = spsolve(A, b)
            u_dot0 = u_dot_la_g_la_gamma[: self.nu]
            la_g0 = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
            la_gamma0 = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

            return q_dot0, u_dot0, la_g0, la_gamma0

        (
            self.q_dotk,
            self.u_dotk,
            self.la_gk,
            self.la_gammak,
        ) = consistent_initial_values(model.t0, model.q0, model.u0)

        # # TODO: Solve for initial Lagrange multipliers!
        # M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        # h0 = self.model.h(self.tk, self.qk, self.uk)
        # W_g0 = self.model.W_g(self.tk, self.qk, scipy_matrix=csr_matrix)
        # W_gamma0 = self.model.W_gamma(self.tk, self.qk, scipy_matrix=csr_matrix)
        # W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        # W_F0 = self.model.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)
        # self.u_dotk = spsolve(
        #     M0,
        #     h0
        #     + W_g0 @ model.la_g0
        #     + W_gamma0 @ model.la_gamma0
        #     + W_N0 @ model.la_N0
        #     + W_F0 @ model.la_F0,
        # )

        # TODO: Add solve for bilateral constraints!
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.P_gk = dt * self.la_gk
        self.mu_gk = np.zeros(self.nla_g)
        self.P_gammak = dt * self.la_gammak
        self.P_Nk = dt * model.la_N0
        self.mu_Nk = np.zeros(self.nla_N)
        self.P_Fk = dt * model.la_F0

        self.yk = np.concatenate((self.qk, self.uk))

    def g(self, t, q, u, z, update_index_set=True):
        # unpack percussions
        la_g = z[: self.nla_g]
        la_gamma = z[self.nla_g : self.nla_g + self.nla_gamma]
        la_N = z[self.nla_g + self.nla_gamma : self.nla_g + self.nla_gamma + self.nla_N]
        P_F = z[self.nla_g + self.nla_gamma + self.nla_N :]

        # bilateral constraints
        gk1 = self.model.g(t, q)
        g_dotk1 = self.model.g_dot(t, q, u)
        gammak1 = self.model.gamma(t, q, u)

        # unilateral constraints
        g_Nk1 = self.model.g_N(t, q)
        g_N_dotk1 = self.model.g_N_dot(t, q, u)
        gamma_Fk1 = self.model.gamma_F(t, q, u)
        # xi_Nk1 = self.model.xi_N(t, q, self.uk, u)
        # xi_Fk1 = self.model.xi_F(t, q, self.uk, u)

        prox_r_N = self.model.prox_r_N(t, q)
        prox_r_F = self.model.prox_r_F(t, q)

        prox_arg_pos = g_Nk1 - prox_r_N * la_N
        c_Nk1 = g_Nk1 - prox_R0_np(prox_arg_pos)

        if update_index_set:
            self.Ak1 = prox_arg_pos <= 0
            # self.Ak1 = g_Nk1 <= 0

        # c_Nk1 = np.where(
        #     self.Ak1, xi_Nk1 - prox_R0_np(xi_Nk1 - prox_r_N * la_N), la_N
        # )
        # c_Nk1 = la_N
        # c_Nk1 = np.where(
        #     self.Ak1,
        #     g_N_dotk1 - prox_R0_np(g_N_dotk1 - prox_r_N * la_N),
        #     la_N,
        # )

        # c_Nk1 = np.where(
        #     self.Ak1, xi_Nk1 - prox_R0_np(xi_Nk1 - self.model.prox_r_N * P_N), P_N
        # )

        # # c_Nk1 = np.where(
        # #     self.Ak1, xi_Nk1 - prox_R0_np(xi_Nk1 - self.model.prox_r_N * P_N), P_N
        # # )
        # alpha = 1.0 - 1.0e-1
        # g_N_bar = g_Nk1 / self.dt + (1.0 - alpha) * g_N_dotk1
        # c_Nk1 = P_N - prox_R0_np(P_N - self.model.prox_r_N * g_N_bar)
        # # c_Nk1 = P_N - prox_R0_np(P_N - self.model.prox_r_N * xi_Nk1)

        # c_Nk1 = np.where(
        #     self.Ak1, g_N_dotk1 - prox_R0_np(g_N_dotk1 - self.model.prox_r_N * P_N), P_N
        # )

        # #############
        # # no friction
        # #############
        # c_Fk1 = P_F

        ##########
        # friction
        ##########
        mu = self.model.mu
        c_Fk1 = P_F.copy()  # this is the else case => P_Fk1 = 0
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                # TODO: Is there a primal/ dual form?
                if self.Ak1[i_N]:
                    # c_Fk1[i_F] = -P_F[i_F] - prox_sphere(
                    #     -P_F[i_F] + prox_r_F[i_N] * xi_Fk1[i_F],
                    #     mu[i_N] * la_N[i_N],
                    # )
                    c_Fk1[i_F] = -P_F[i_F] - prox_sphere(
                        -P_F[i_F] + prox_r_F[i_N] * gamma_Fk1[i_F],
                        mu[i_N] * la_N[i_N],
                    )

        ck1 = np.concatenate((gk1, gammak1, c_Nk1, c_Fk1))

        return ck1

    def g_z(self, zk1):
        return csr_matrix(
            approx_fprime(zk1, lambda z: self.c(z, update_index_set=False))
        )

    def f(self, t, q, u):
        return self.model.q_dot(t, q, u)

    def k(self, t, q, u, z):
        # unpack percussions
        la_g = z[: self.nla_g]
        la_gamma = z[self.nla_g : self.nla_g + self.nla_gamma]
        la_N = z[self.nla_g + self.nla_gamma : self.nla_g + self.nla_gamma + self.nla_N]
        la_F = z[self.nla_g + self.nla_gamma + self.nla_N :]

        # evaluate quantities of previous time step
        M = self.model.M(t, q, scipy_matrix=csr_matrix)
        h = self.model.h(t, q, u)
        W_g = self.model.W_g(t, q, scipy_matrix=csr_matrix)
        W_gamma = self.model.W_gamma(t, q, scipy_matrix=csr_matrix)
        W_N = self.model.W_N(t, q, scipy_matrix=csr_matrix)
        W_F = self.model.W_F(t, q, scipy_matrix=csr_matrix)

        return spsolve(
            M,
            h + W_g @ la_g + W_gamma @ la_gamma + W_N @ la_N + W_F @ la_F,
        )

    def unpack_y(self, yk1):
        Uk1 = yk1[: self.nu]
        La_gk1 = yk1[self.nu : self.nu + self.nla_g]
        La_gammak1 = yk1[self.nu + self.nla_g : self.nu + self.nla_g + self.nla_gamma]
        La_Nk1 = yk1[
            self.nu
            + self.nla_g
            + self.nla_gamma : self.nu
            + self.nla_g
            + self.nla_gamma
            + self.nla_N
        ]
        La_Fk1 = yk1[self.nu + self.nla_g + self.nla_gamma + self.nla_N :]

        return Uk1, La_gk1, La_gammak1, La_Nk1, La_Fk1

    def Ry(self, yk1):
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        mu = self.model.mu

        # quantities of old time step
        tk1 = self.tk1
        qk1 = self.qk1
        uk1_free = self.uk1_free

        # unpack xk1
        Uk1, La_gk1, La_gammak1, La_Nk1, La_Fk1 = self.unpack_y(yk1)

        # update velocities
        uk1 = uk1_free + Uk1

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_dot = self.model.g_dot(tk1, qk1, uk1)
        gamma = self.model.gamma(tk1, qk1, uk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        ###################
        # evaluate residual
        ###################
        Ry = np.zeros(self.nu + self.nla_g + self.nla_gamma + self.nla_N + self.nla_F)

        #################
        # impact equation
        #################
        Ry[:nu] = (
            Mk1 @ Uk1
            - W_gk1 @ La_gk1
            - W_gammak1 @ La_gammak1
            - W_Nk1 @ La_Nk1
            - W_Fk1 @ La_Fk1
        )

        # bilateral constraints
        Ry[nu : nu + nla_g] = g_dot
        Ry[nu + nla_g : nu + nla_g + nla_gamma] = gamma

        ###################
        # normal impact law
        ###################
        prox_r_N = self.model.prox_r_N(tk1, qk1)
        Ry[nu + nla_g + nla_gamma : nu + nla_g + nla_gamma + nla_N] = np.where(
            self.Ak1,
            xi_Nk1 - prox_R0_np(xi_Nk1 - prox_r_N * La_Nk1),
            La_Nk1,
        )

        ####################
        # tangent impact law
        ####################
        prox_r_F = self.model.prox_r_F(tk1, qk1)
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                Ry[nu + nla_g + nla_gamma + nla_N + i_F] = np.where(
                    self.Ak1[i_N] * np.ones(len(i_F), dtype=bool),
                    -La_Fk1[i_F]
                    - prox_sphere(
                        -La_Fk1[i_F] + prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * La_Nk1[i_N],
                    ),
                    La_Fk1[i_F],
                )

        return Ry

    def step(self):
        ##############################
        # simple two stage Runge-Kutta
        ##############################
        from scipy.optimize import fsolve

        tk = self.tk
        qk = self.qk.copy()
        uk = self.uk.copy()
        h = self.dt

        u_dot = lambda z: self.k(tk, qk.copy(), uk.copy(), z)
        q_dot = lambda z: self.f(tk, qk.copy(), uk.copy() + h * u_dot(z))

        tk1 = tk + h
        g = lambda z: self.g(tk1, qk.copy() + h * q_dot(z), uk.copy() + h * u_dot(z), z)

        # solve for final Lagrange multipliers
        z0 = np.concatenate([self.la_gk, self.la_gammak, self.P_Nk, self.P_Fk])
        res = fsolve(g, z0, full_output=1)
        zk1 = res[0]
        converged = res[2]
        j = res[1]["nfev"]

        # compute final velocites
        uk1_free = uk + h * u_dot(zk1)

        # compute final positions
        qk1 = qk + h * q_dot(zk1)

        # #################
        # # compute impacts
        # #################
        # self.tk1 = tk1
        # self.qk1 = qk1
        # self.uk1_free = uk1_free.copy()

        # sk1 = np.concatenate(
        #     (
        #         np.zeros(self.nu),
        #         np.zeros(self.nla_g),
        #         np.zeros(self.nla_gamma),
        #         np.zeros(self.nla_N),
        #         np.zeros(self.nla_F),
        #     )
        # )

        # res = fsolve(self.Ry, sk1, full_output=1)
        # sk1 = res[0]
        # converged *= res[2]
        # j += res[1]["nfev"]

        # Uk1, La_gk1, La_gammak1, La_Nk1, La_Fk1 = self.unpack_y(sk1)
        # # print(f"Uk1: {Uk1}")

        # uk1 = self.uk1_free + Uk1

        ############
        # no impacts
        ############
        uk1 = uk1_free

        # unpack percussions
        P_gk1 = zk1[: self.nla_g]
        P_gammak1 = zk1[self.nla_g : self.nla_g + self.nla_gamma]
        P_Nk1 = zk1[
            self.nla_g + self.nla_gamma : self.nla_g + self.nla_gamma + self.nla_N
        ]
        P_Fk1 = zk1[self.nla_g + self.nla_gamma + self.nla_N :]

        error = 0.0
        return (
            (converged, j, error),
            tk1,
            qk1,
            uk1,
            P_gk1,
            P_gammak1,
            P_Nk1,
            P_Fk1,
        )

    def solve(self):
        # lists storing output variables
        q = [self.qk]
        u = [self.uk]
        P_g = [self.P_gk]
        P_gamma = [self.P_gammak]
        P_N = [self.P_Nk]
        P_F = [self.P_Fk]

        self.yk = np.concatenate((self.qk, self.uk))

        pbar = tqdm(self.t[:-1])
        for _ in pbar:
            (
                (converged, j, error),
                tk1,
                qk1,
                uk1,
                P_gk1,
                P_gammak1,
                P_Nk1,
                P_Fk1,
            ) = self.step()
            pbar.set_description(
                # f"t: {tk1:0.2e}; fixed-point iterations: {j+1}; error: {error:.3e}"
                f"t: {tk1:0.2e}; Newton iterations: {j+1}; error: {error:.3e}"
            )
            if not converged:
                # raise RuntimeError(
                print(
                    # f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                    f"Newton iteration not converged after {j+1} iterations with error: {error:.5e}"
                )

            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            q.append(qk1)
            u.append(uk1)
            P_g.append(P_gk1)
            P_gamma.append(P_gammak1)
            P_N.append(P_Nk1)
            P_F.append(P_Fk1)

            # update local variables for accepted time step
            (
                self.tk,
                self.qk,
                self.uk,
                self.P_gk,
                self.P_gammak,
                self.P_Nk,
                self.P_Fk,
            ) = (tk1, qk1, uk1, P_gk1, P_gammak1, P_Nk1, P_Fk1)

        return Solution(
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
        )


class HalfExplicitEuler:
    def __init__(
        self,
        model,
        t1,
        dt,
        atol=1e-6,
        max_iter=10000,
        error_function=lambda x: np.max(np.abs(x)),
    ):
        self.model = model

        # integration time
        t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.error_function = error_function
        self.atol = atol
        self.max_iter = max_iter

        self.nq = self.model.nq
        self.nu = self.model.nu
        self.nla_g = self.model.nla_g
        # self.nla_gamma = self.model.nla_gamma
        # self.nR = self.nu + self.nla_g + self.nla_gamma
        self.nR = self.nu + self.nla_g

        # initial state
        self.tk = t0
        self.qk = model.q0
        self.uk = model.u0

        def initial_values(t0, q0, u0):
            # initial velocites
            q_dot0 = self.model.q_dot(t0, q0, u0)

            # solve for consistent initial accelerations and Lagrange mutlipliers
            M0 = self.model.M(t0, q0, scipy_matrix=csr_matrix)
            h0 = self.model.h(t0, q0, u0)
            W_g0 = self.model.W_g(t0, q0, scipy_matrix=csr_matrix)
            # W_gamma0 = self.model.W_gamma(t0, q0, scipy_matrix=csr_matrix)
            zeta_g0 = self.model.zeta_g(t0, q0, u0)
            # fmt: off
            # zeta_gamma0 = self.model.zeta_gamma(t0, q0, u0)
            # A = bmat(
            #     [
            #         [M0, -W_g0, -W_gamma0],
            #         [W_g0.T, None, None],
            #         [W_gamma0.T, None, None],
            #     ],
            #     format="csc",
            # )
            A = bmat(
                [
                    [    M0, -W_g0],
                    [W_g0.T,  None],
                ],
                format="csc",
            )
            # fmt: on
            # b = np.concatenate([h0, -zeta_g0, -zeta_gamma0])
            b = np.concatenate([h0, -zeta_g0])
            u_dot_la_g_la_gamma = spsolve(A, b)
            u_dot0 = u_dot_la_g_la_gamma[: self.nu]
            la_g0 = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
            # la_gamma0 = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

            return q_dot0, u_dot0, la_g0

        # solve for consistent inital accelerations and Lagrange multipliers
        self.q_dotk, self.u_dotk, self.la_gk = initial_values(self.tk, self.qk, self.uk)
        self.mu_gk = np.zeros_like(self.la_gk)

        # if hasattr(model, "step_callback"):
        #     self.step_callback = model.step_callback
        # else:
        #     self.step_callback = self.__step_callback

        # check if initial conditions satisfy constraints on position, velocity
        # and acceleration level
        g0 = model.g(self.tk, self.qk)
        g_dot0 = model.g_dot(self.tk, self.qk, self.uk)
        g_ddot0 = model.g_ddot(self.tk, self.qk, self.uk, self.u_dotk)
        # gamma0 = model.gamma(self.tk, self.qk, self.uk)
        # gamma_dot0 = model.gamma_dot(self.tk, self.qk, self.uk, self.u_dotk)

        assert np.allclose(
            g0, np.zeros(self.nla_g)
        ), "Initial conditions do not fulfill g0!"
        assert np.allclose(
            g_dot0, np.zeros(self.nla_g)
        ), "Initial conditions do not fulfill g_dot0!"
        assert np.allclose(
            g_ddot0, np.zeros(self.nla_g)
        ), "Initial conditions do not fulfill g_ddot0!"
        # assert np.allclose(
        #     gamma0, np.zeros(self.nla_gamma)
        # ), "Initial conditions do not fulfill gamma0!"
        # assert np.allclose(
        #     gamma_dot0, np.zeros(self.nla_gamma)
        # ), "Initial conditions do not fulfill gamma_dot0!"

    def c(self, kappa_k):
        mu_gk1 = kappa_k[: self.nla_g]
        la_gk1 = kappa_k[self.nla_g :]

        # values from previous time step
        tk = self.tk
        qk = self.qk
        uk = self.uk

        # evaluate quantities of previous time step
        Mk = self.model.M(tk, qk, scipy_matrix=csc_matrix)
        W_gk = self.model.W_g(tk, qk, scipy_matrix=csc_matrix)
        g_qk = self.model.g_q(tk, qk, scipy_matrix=csr_matrix)
        hk = self.model.h(tk, qk, uk)

        # explicit Euler step
        tk1 = self.tk + self.dt
        qk1 = self.qk + self.dt * self.model.q_dot(tk, qk, uk) + g_qk.T @ mu_gk1
        uk1 = self.uk + self.dt * spsolve(Mk, hk + W_gk @ la_gk1)

        # constraint equations
        gk1 = self.model.g(tk1, qk1)
        g_dotk1 = self.model.g_dot(tk1, qk1, uk1)
        ck1 = np.concatenate((gk1, g_dotk1))
        yield ck1, tk1, qk1, uk1, mu_gk1, la_gk1

        # jacobian
        g_qk1 = self.model.g_q(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        g_muk1 = g_qk1 @ g_qk.T
        g_dot_muk1 = self.model.g_dot_q(tk1, qk1, uk1) @ g_qk.T
        g_dot_la_gk1 = self.dt * W_gk1.T @ spsolve(Mk, W_gk)

        # fmt: off
        c_kappak1 = bmat(
            [[    g_muk1,         None], 
             [g_dot_muk1, g_dot_la_gk1]], 
             format="csr"
        )
        # fmt: on

        yield c_kappak1

    def f(self, tk, qk, uk, mu_k1, la_gk):
        # evaluate quantities of previous time step
        Mk = self.model.M(tk, qk, scipy_matrix=csr_matrix)
        W_gk = self.model.W_g(tk, qk, scipy_matrix=csr_matrix)
        g_qk = self.model.g_q(tk, qk, scipy_matrix=csr_matrix)
        hk = self.model.h(tk, qk, uk)

        # explicit Euler step
        tk1 = self.tk + self.dt
        qk1 = self.qk + self.dt * self.model.q_dot(tk, qk, uk) + g_qk.T @ mu_k1
        uk1 = self.uk + self.dt * spsolve(Mk, hk + W_gk @ la_gk)

        return tk1, qk1, uk1

    def step(self):
        j = 0
        converged = False
        kappa_k1 = np.concatenate((self.mu_gk, self.la_gk))
        gen = self.c(kappa_k1)
        ck1, tk1, qk1, uk1, mu_gk1, la_gk1 = next(gen)
        error = self.error_function(ck1)
        # print(f"step: {j}; error: {error}")
        converged = error < self.atol
        while (not converged) and (j < self.max_iter):
            j += 1

            # compute Jacobian and make a Newton step
            c_kappak1 = next(gen)
            kappa_k1 -= spsolve(c_kappak1, ck1)
            # kappa_k1 -= spsolve(c_kappak1, ck1)
            # kappa_k1 -= np.linalg.lstsq(c_kappak1.toarray(), ck1)[0]

            # # fixed-point iteration
            # # self.atol = 1.0e-4
            # r = 4.0e-1
            # # r = 1.0e-2
            # kappa_k1 -= r * ck1

            # c_kappak1_num = approx_fprime(kappa_k1, lambda kappa: next(self.c(kappa))[0], method="2-point")
            # # diff = c_kappak1.toarray() - c_kappak1_num
            # # error = np.linalg.norm(diff)
            # # np.set_printoptions(3, suppress=True)
            # # print(f"diff:\n{diff}")
            # # print(f"error jacobian: {error}")
            # kappa_k1 -= np.linalg.solve(c_kappak1_num, ck1)

            # check error for new Lagrange multipliers
            gen = self.c(kappa_k1)
            ck1, tk1, qk1, uk1, mu_gk1, la_gk1 = next(gen)
            error = self.error_function(ck1)
            converged = error < self.atol
            # print(f"step: {j}; error: {error:0.3e}")
            # print(f"")

        if not converged:
            raise RuntimeError("Internal Newton-Raphson scheme is not converged!")

        return (converged, j, error), tk1, qk1, uk1, mu_gk1, la_gk1

        # exit()

        ###########
        # old tests
        ###########
        tk1, qk1, uk1 = self.f(self.tk, self.qk, self.uk, self.mu_gk, self.la_gk)
        gk1 = self.model.g(tk1, qk1)
        g_dotk1 = self.model.g_dot(tk1, qk1, uk1)
        ck1 = np.concatenate((gk1, g_dotk1))
        error = self.error_function(ck1)
        converged = error <= self.atol

        #################################################################
        # starting fixed-point iteration if constraints are not satisfied
        #################################################################
        j = 0
        mu_gk1 = self.mu_gk.copy()
        la_gk1 = self.la_gk.copy()
        while (not converged) and (j < self.max_iter):
            j += 1

            # update state
            # tk1, qk1, uk1 = self.f(self.tk, self.qk, self.uk, mu_gk1, la_gk1)
            tk1, qk1, uk1 = self.f(tk1, qk1, uk1, mu_gk1, la_gk1)

            # check if constraints are satisfied
            gk1 = self.model.g(tk1, qk1)
            g_dotk1 = self.model.g_dot(tk1, qk1, uk1)
            ck1 = np.concatenate((gk1, g_dotk1))
            error = self.error_function(ck1)
            converged = error <= self.atol
            print(f"j: {j}; error: {error:0.3e}; converged: {converged}")
            # print(f"la_gk1: {la_gk1}")
            # print(f"mu_gk1: {mu_gk1}")

            # prox equation for Lagrange multipliers using old constraints
            r = 2.0e-1
            mu_gk1 -= r * gk1
            la_gk1 -= r * g_dotk1

        # ########################
        # # Newton-Raphson version
        # ########################
        # j = 0
        # mu_gk1 = self.mu_gk.copy()
        # la_gk1 = self.la_gk.copy()
        # while (not converged) and (j < self.max_iter):
        #     j += 1

        #     # update state
        #     tk1, qk1, uk1 = self.f(self.tk, self.qk, self.uk, mu_gk1, la_gk1)

        #     # check if constraints are satisfied
        #     gk1 = self.model.g(tk1, qk1)
        #     g_dotk1 = self.model.g_dot(tk1, qk1, uk1)
        #     ck1 = np.concatenate((gk1, g_dotk1))

        #     # TODO Move on here!
        #     ck1_kappa_k1 = bmat([
        #         [g_qk1 @ g_qk1.T, ]
        #     ])

        #     kappa_k1 -= np.linalg.solve(
        #         approx_fprime(kappa_k1, ),
        #         ck1
        #     )

        #     error = self.error_function(ck1)
        #     converged = (error <= self.atol)
        #     print(f"j: {j}; error: {error:0.3e}")
        #     print(f"la_gk1: {la_gk1}")
        #     print(f"mu_gk1: {mu_gk1}")

        print(f"converged: {converged}")
        exit()

    def solve(self):
        # lists storing output variables
        q = [self.qk]
        u = [self.uk]
        mu_g = [self.mu_gk]
        la_g = [self.la_gk]
        # la_gamma = [self.la_gammak]

        pbar = tqdm(self.t[:-1])
        for _ in pbar:
            (
                (converged, j, error),
                tk1,
                qk1,
                uk1,
                mu_gk1,
                la_gk1,
            ) = self.step()
            pbar.set_description(
                # f"t: {tk1:0.2e}; fixed-point iterations: {j+1}; error: {error:.3e}"
                f"t: {tk1:0.2e}; Newton iterations: {j+1}; error: {error:.3e}"
            )
            if not converged:
                raise RuntimeError(
                    # f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                    f"Newton iteration not converged after {j+1} iterations with error: {error:.5e}"
                )

            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            q.append(qk1)
            u.append(uk1)
            mu_g.append(mu_gk1)
            la_g.append(la_gk1)

            # update local variables for accepted time step
            (
                self.tk,
                self.qk,
                self.uk,
                self.mu_gk,
                self.la_gk,
            ) = (tk1, qk1, uk1, mu_gk1, la_gk1)

        return Solution(
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
            mu_g=np.array(mu_g),
            la_g=np.array(la_g),
            # la_gamma=np.array(la_gamma),
        )


# TODO:
# - Investigate formulation by Arnold and Murua.
#   They should improve the interpolation order of the Lagrange multipliers.
# - Why is RK4 with decoupled stages working perfectly?
class NonsmoothHalfExplicitRungeKutta:
    def __init__(
        self,
        model,
        t1,
        dt,
        atol=1e-6,
        max_iter=50,
        error_function=lambda x: np.max(np.abs(x)),
    ):
        self.model = model

        # integration time
        t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.error_function = error_function
        self.atol = atol
        self.max_iter = max_iter

        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = model.nla_g
        self.nla_gamma = model.nla_gamma
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F

        # initial state
        self.tk = t0
        self.qk = model.q0
        self.uk = model.u0

        # initial accelerations
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        def consistent_initial_values(t0, q0, u0):
            """compute physically consistent initial values"""

            # initial velocites
            q_dot0 = self.model.q_dot(t0, q0, u0)

            # solve for consistent initial accelerations and Lagrange mutlipliers
            M0 = self.model.M(t0, q0, scipy_matrix=csr_matrix)
            h0 = self.model.h(t0, q0, u0)
            W_g0 = self.model.W_g(t0, q0, scipy_matrix=csr_matrix)
            W_gamma0 = self.model.W_gamma(t0, q0, scipy_matrix=csr_matrix)
            zeta_g0 = self.model.zeta_g(t0, q0, u0)
            zeta_gamma0 = self.model.zeta_gamma(t0, q0, u0)
            # fmt: off
            A = bmat(
                [
                    [        M0, -W_g0, -W_gamma0],
                    [    W_g0.T,  None,      None],
                    [W_gamma0.T,  None,      None],
                ],
                format="csc",
            )
            # fmt: on
            b = np.concatenate([h0, -zeta_g0, -zeta_gamma0])
            u_dot_la_g_la_gamma = spsolve(A, b)
            u_dot0 = u_dot_la_g_la_gamma[: self.nu]
            la_g0 = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
            la_gamma0 = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

            return q_dot0, u_dot0, la_g0, la_gamma0

        (
            self.q_dotk,
            self.u_dotk,
            self.la_gk,
            self.la_gammak,
        ) = consistent_initial_values(model.t0, model.q0, model.u0)

        # # TODO: Solve for initial Lagrange multipliers!
        # M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        # h0 = self.model.h(self.tk, self.qk, self.uk)
        # W_g0 = self.model.W_g(self.tk, self.qk, scipy_matrix=csr_matrix)
        # W_gamma0 = self.model.W_gamma(self.tk, self.qk, scipy_matrix=csr_matrix)
        # W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        # W_F0 = self.model.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)
        # self.u_dotk = spsolve(
        #     M0,
        #     h0
        #     + W_g0 @ model.la_g0
        #     + W_gamma0 @ model.la_gamma0
        #     + W_N0 @ model.la_N0
        #     + W_F0 @ model.la_F0,
        # )

        # TODO: Add solve for bilateral constraints!
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.P_gk = dt * self.la_gk
        self.mu_gk = np.zeros(self.nla_g)
        self.P_gammak = dt * self.la_gammak
        self.P_Nk = dt * model.la_N0
        self.mu_Nk = np.zeros(self.nla_N)
        self.P_Fk = dt * model.la_F0

        self.yk = np.concatenate((self.qk, self.uk))

    def gen_c(self, ti, Yi_fun):
        def c(Z, update_index_set=True):
            # unpack percussions
            P_g = Z[: self.nla_g]
            mu_g = Z[self.nla_g : 2 * self.nla_g]
            P_gamma = Z[2 * self.nla_g : 2 * self.nla_g + self.nla_gamma]
            P_N = Z[
                2 * self.nla_g
                + self.nla_gamma : 2 * self.nla_g
                + self.nla_gamma
                + self.nla_N
            ]
            mu_N = Z[
                2 * self.nla_g
                + self.nla_gamma
                + self.nla_N : 2 * self.nla_g
                + self.nla_gamma
                + 2 * self.nla_N
            ]
            P_F = Z[2 * self.nla_g + self.nla_gamma + 2 * self.nla_N :]

            # update states
            Yi = Yi_fun(Z)

            # unpack state
            qi = Yi[: self.nq]
            ui = Yi[self.nq :]

            # bilateral constraints
            gk1 = self.model.g(ti, qi)
            g_dotk1 = self.model.g_dot(ti, qi, ui)
            gammak1 = self.model.gamma(ti, qi, ui)

            # unilateral constraints
            g_Nk1 = self.model.g_N(ti, qi)
            g_N_dotk1 = self.model.g_N_dot(ti, qi, ui)
            gamma_Fk1 = self.model.gamma_F(ti, qi, ui)
            xi_Nk1 = self.model.xi_N(ti, qi, self.uk, ui)
            xi_Fk1 = self.model.xi_F(ti, qi, self.uk, ui)

            prox_r_N = self.model.prox_r_N(ti, qi)
            prox_r_F = self.model.prox_r_F(ti, qi)

            prox_arg_pos = g_Nk1 - prox_r_N * mu_N
            c_Nk1_stab = g_Nk1 - prox_R0_np(prox_arg_pos)
            # c_Nk1_stab = mu_N

            if update_index_set:
                # self.Ak1 = g_Nk1 <= 0
                self.Ak1 = prox_arg_pos <= 0

            # c_Nk1 = np.where(
            #     self.Ak1, xi_Nk1 - prox_R0_np(xi_Nk1 - prox_r_N * P_N), P_N
            # )
            c_Nk1 = np.where(
                self.Ak1,
                g_N_dotk1 - prox_R0_np(g_N_dotk1 - prox_r_N * P_N),
                P_N,
            )

            # c_Nk1 = np.where(
            #     self.Ak1, xi_Nk1 - prox_R0_np(xi_Nk1 - self.model.prox_r_N * P_N), P_N
            # )

            # # c_Nk1 = np.where(
            # #     self.Ak1, xi_Nk1 - prox_R0_np(xi_Nk1 - self.model.prox_r_N * P_N), P_N
            # # )
            # alpha = 1.0 - 1.0e-1
            # g_N_bar = g_Nk1 / self.dt + (1.0 - alpha) * g_N_dotk1
            # c_Nk1 = P_N - prox_R0_np(P_N - self.model.prox_r_N * g_N_bar)
            # # c_Nk1 = P_N - prox_R0_np(P_N - self.model.prox_r_N * xi_Nk1)

            # c_Nk1 = np.where(
            #     self.Ak1, g_N_dotk1 - prox_R0_np(g_N_dotk1 - self.model.prox_r_N * P_N), P_N
            # )

            ##########
            # friction
            ##########
            mu = self.model.mu
            c_Fk1 = P_F.copy()  # this is the else case => P_Fk1 = 0
            for i_N, i_F in enumerate(self.model.NF_connectivity):
                i_F = np.array(i_F)

                if len(i_F) > 0:
                    # TODO: Is there a primal/ dual form?
                    if self.Ak1[i_N]:
                        # c_Fk1[i_F] = -P_F[i_F] - prox_sphere(
                        #     -P_F[i_F] + prox_r_F[i_N] * xi_Fk1[i_F],
                        #     mu[i_N] * P_N[i_N],
                        # )
                        c_Fk1[i_F] = -P_F[i_F] - prox_sphere(
                            -P_F[i_F] + prox_r_F[i_N] * gamma_Fk1[i_F],
                            mu[i_N] * P_N[i_N],
                        )

            ck1 = np.concatenate((g_dotk1, gk1, gammak1, c_Nk1, c_Nk1_stab, c_Fk1))

            return ck1

        return c

    def c_y(self, zk1):
        return csr_matrix(
            approx_fprime(zk1, lambda z: self.c(z, update_index_set=False)[0])
        )

    def f(self, tk, y, z):
        # unpack state
        q = y[: self.nq]
        u = y[self.nq :]

        # unpack percussions
        P_g = z[: self.nla_g]
        mu_g = z[self.nla_g : 2 * self.nla_g]
        P_gamma = z[2 * self.nla_g : 2 * self.nla_g + self.nla_gamma]
        P_N = z[
            2 * self.nla_g
            + self.nla_gamma : 2 * self.nla_g
            + self.nla_gamma
            + self.nla_N
        ]
        mu_N = z[
            2 * self.nla_g
            + self.nla_gamma
            + self.nla_N : 2 * self.nla_g
            + self.nla_gamma
            + 2 * self.nla_N
        ]
        P_F = z[2 * self.nla_g + self.nla_gamma + 2 * self.nla_N :]

        # evaluate quantities of previous time step
        M = self.model.M(tk, q, scipy_matrix=csr_matrix)
        h = self.model.h(tk, q, u)
        g_q = self.model.g_q(tk, q, scipy_matrix=csr_matrix)
        g_N_q = self.model.g_N_q(tk, q, scipy_matrix=csr_matrix)
        W_g = self.model.W_g(tk, q, scipy_matrix=csr_matrix)
        W_gamma = self.model.W_gamma(tk, q, scipy_matrix=csr_matrix)
        W_N = self.model.W_N(tk, q, scipy_matrix=csr_matrix)
        W_F = self.model.W_F(tk, q, scipy_matrix=csr_matrix)

        dt = self.dt

        return np.concatenate(
            (
                self.model.q_dot(tk, q, u) * dt + g_q.T @ mu_g + g_N_q.T @ mu_N,
                spsolve(
                    M,
                    h * dt + W_g @ P_g + W_gamma @ P_gamma + W_N @ P_N + W_F @ P_F,
                ),
            )
        )

    def unpack_y(self, yk1):
        Uk1 = yk1[: self.nu]
        La_gk1 = yk1[self.nu : self.nu + self.nla_g]
        La_gammak1 = yk1[self.nu + self.nla_g : self.nu + self.nla_g + self.nla_gamma]
        La_Nk1 = yk1[
            self.nu
            + self.nla_g
            + self.nla_gamma : self.nu
            + self.nla_g
            + self.nla_gamma
            + self.nla_N
        ]
        La_Fk1 = yk1[self.nu + self.nla_g + self.nla_gamma + self.nla_N :]

        return Uk1, La_gk1, La_gammak1, La_Nk1, La_Fk1

    def Ry(self, yk1):
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        mu = self.model.mu

        # quantities of old time step
        tk1 = self.tk1
        qk1 = self.qk1
        uk1_free = self.uk1_free

        # unpack xk1
        Uk1, La_gk1, La_gammak1, La_Nk1, La_Fk1 = self.unpack_y(yk1)

        # update velocities
        uk1 = uk1_free + Uk1

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_dot = self.model.g_dot(tk1, qk1, uk1)
        gamma = self.model.gamma(tk1, qk1, uk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        ###################
        # evaluate residual
        ###################
        Ry = np.zeros(self.nu + self.nla_g + self.nla_gamma + self.nla_N + self.nla_F)

        #################
        # impact equation
        #################
        Ry[:nu] = (
            Mk1 @ Uk1
            - W_gk1 @ La_gk1
            - W_gammak1 @ La_gammak1
            - W_Nk1 @ La_Nk1
            - W_Fk1 @ La_Fk1
        )

        # bilateral constraints
        Ry[nu : nu + nla_g] = g_dot
        Ry[nu + nla_g : nu + nla_g + nla_gamma] = gamma

        ###################
        # normal impact law
        ###################
        prox_r_N = self.model.prox_r_N(tk1, qk1)
        Ry[nu + nla_g + nla_gamma : nu + nla_g + nla_gamma + nla_N] = np.where(
            self.Ak1,
            xi_Nk1 - prox_R0_np(xi_Nk1 - prox_r_N * La_Nk1),
            La_Nk1,
        )

        ####################
        # tangent impact law
        ####################
        prox_r_F = self.model.prox_r_F(tk1, qk1)
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                Ry[nu + nla_g + nla_gamma + nla_N + i_F] = np.where(
                    self.Ak1[i_N] * np.ones(len(i_F), dtype=bool),
                    -La_Fk1[i_F]
                    - prox_sphere(
                        -La_Fk1[i_F] + prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * La_Nk1[i_N],
                    ),
                    La_Fk1[i_F],
                )

        return Ry

    def step(self):
        j = 0
        converged = True
        error = 0.0

        ##############################
        # simple two stage Runge-Kutta
        ##############################
        from scipy.optimize import fsolve

        # # forward Euler
        # c = [1.0]
        # A = [[0.0]]
        # b = [0.0, 1.0]

        # # explicit midpoint rule
        # c = [0.5]
        # A = [[0.5]]
        # b = [0.0, 1.0]

        # # Heun
        # c = [1.0]
        # A = [[1.0]]
        # b = [0.5, 0.5]

        # # Ralston
        # c = [2/3]
        # A = [[2/3]]
        # b = [1/4, 3/4]

        # # RK3
        # c = [0.5, 1]
        # A = [
        #     [0.5],
        #     [-1, 2],
        # ]
        # b = [1/6, 2/3, 1/6]

        # # RK4
        # c = [0.5, 0.5, 1.0]
        # A = [
        #     [0.5],
        #     [0.0, 0.5],
        #     [0.0, 0.0, 1.0],
        # ]
        # b = [1/6, 1/3, 1/3, 1/6]

        # # hem3
        # c = [1.0/3.0, 1.0]
        # A = [
        #         [1.0/3.0],
        #         [-1.0, 2.0],
        #     ]
        # b = [0.0, 3.0/4.0, 1.0/4.0]

        tk = self.tk
        h = self.dt
        yk = self.yk.copy()
        zk = np.concatenate(
            (self.P_gk, self.mu_gk, self.P_gammak, self.P_Nk, self.mu_Nk, self.P_Fk)
        )

        #############
        # Brasey 1993
        #############

        # # forward Euler
        # c = [0]
        # A = [[0]]
        # b = [1]

        # # hem3
        # c = [0, 1 / 3, 1]
        # A = [
        #     [0],
        #     [1 / 3],
        #     [-1, 2],
        # ]
        # b = [0, 3 / 4, 1 / 4]

        # hem5
        from math import sqrt

        sqrt6 = sqrt(6)
        c = [0, 3 / 10, (4 - sqrt6) / 10, (4 + sqrt6) / 10, 1]
        A = [
            [0],
            [3 / 10],
            [(1 + sqrt6) / 30, (11 - 4 * sqrt6) / 30],
            [(-79 - 31 * sqrt6) / 150, (-1 - 4 * sqrt6) / 30, (24 + 11 * sqrt6) / 25],
            [
                (14 + 5 * sqrt6) / 6,
                (-8 + 7 * sqrt6) / 6,
                (-9 - 7 * sqrt6) / 4,
                (9 - sqrt6) / 4,
            ],
        ]
        b = [0, 0, (16 - sqrt6) / 36, (16 + sqrt6) / 36, 1 / 9]

        # # RK4
        # c = [0.0, 0.5, 0.5, 1.0]
        # A = [
        #     [0.0],
        #     [0.0, 0.5],
        #     [0.0, 0.0, 0.5],
        #     [0.0, 0.0, 0.0, 1.0],
        # ]
        # b = [1 / 6, 1 / 3, 1 / 3, 1 / 6]

        # first stage
        t1 = tk
        Y1 = yk.copy()

        # intermediate states
        tis = [t1]
        Yis = [Y1]
        Zis = []

        s = len(c)
        for i in range(1, s):
            ti = tk + c[i] * h

            Yi_fun = lambda Zi: (
                yk
                + sum([A[i][j] * self.f(ti, Yis[j], Zis[j]) for j in range(i - 1)])
                + A[i][i - 1] * self.f(ti, Yis[i - 1], Zi)
            )

            ci_fun = self.gen_c(ti, Yi_fun)
            res = fsolve(ci_fun, np.zeros_like(zk), full_output=1)

            Zi = res[0]
            converged *= res[2]
            j += res[1]["nfev"]
            Yi = Yi_fun(Zi)

            tis.append(ti)
            Yis.append(Yi.copy())
            Zis.append(Zi.copy())

        # final stage
        tk1 = tk + h
        yk1_fun = lambda Zs: (
            yk
            + sum([b[i] * self.f(tis[i], Yis[i], Zis[i]) for i in range(s - 1)])
            + b[-1] * self.f(tk1, Yis[-1], Zs)
        )
        ck1_fun = self.gen_c(tk1, yk1_fun)
        res = fsolve(ck1_fun, np.zeros_like(zk), full_output=1)
        Zs = res[0]
        Zis.append(Zs.copy())
        converged *= res[2]
        j += res[1]["nfev"]
        yk1 = yk1_fun(Zs)

        # zk1 = Zs
        zk1 = sum([b[i] * Zis[i] for i in range(s)])

        # # second stage
        # t2 = tk + c[1] * h
        # Y2_fun = lambda Z1: (
        #     yk
        #     + A[1][1] * self.f(t2, Y1, Z1)
        # )
        # c2_fun = self.gen_c(t2, Y2_fun)
        # res = fsolve(c2_fun, np.zeros_like(zk), full_output=1)
        # Z1 = res[0]
        # converged *= res[2]
        # j += res[1]["nfev"]
        # Y2 = Y2_fun(Z1)

        # # third stage
        # t3 = tk + c[2] * h
        # Y3_fun = lambda Z2: (
        #     yk
        #     + A[2][1] * self.f(t2, Y1, Z1)
        #     + A[2][2] * self.f(t3, Y2, Z2)
        # )
        # c3_fun = self.gen_c(t3, Y3_fun)
        # res = fsolve(c3_fun, np.zeros_like(zk), full_output=1)
        # Z2 = res[0]
        # converged *= res[2]
        # j += res[1]["nfev"]
        # Y3 = Y3_fun(Z2)

        # # fourth stage
        # t4 = tk + c[3] * h
        # Y4_fun = lambda Z3: (
        #     yk
        #     + A[3][1] * self.f(t2, Y1, Z1)
        #     + A[3][2] * self.f(t3, Y2, Z2)
        #     + A[3][3] * self.f(t4, Y3, Z3)
        # )
        # c4_fun = self.gen_c(t4, Y4_fun)
        # res = fsolve(c4_fun, np.zeros_like(zk), full_output=1)
        # Z3 = res[0]
        # converged *= res[2]
        # j += res[1]["nfev"]
        # Y4 = Y4_fun(Z3)

        # # final stage
        # tk1 = tk + h
        # yk1_fun = lambda Z4: (
        #     yk
        #     + b[0] * self.f(t1, Y1, Z1)
        #     + b[1] * self.f(t2, Y2, Z2)
        #     + b[2] * self.f(t3, Y3, Z3)
        #     + b[3] * self.f(t4, Y4, Z4)
        # )
        # ck1_fun = self.gen_c(tk1, yk1_fun)
        # res = fsolve(ck1_fun, np.zeros_like(zk), full_output=1)
        # Z4 = res[0]
        # converged *= res[2]
        # j += res[1]["nfev"]
        # yk1 = yk1_fun(Z4)

        # # zk1 = Z4
        # zk1 = (
        #     b[0] * Z1
        #     + b[1] * Z2
        #     + b[2] * Z3
        #     + b[3] * Z4
        # )

        # # intermediate states
        # tis = [tk]
        # Yis = [yk]
        # Zis = []
        # for i, (ci, ai) in enumerate(zip(c, A)):

        #     ti = tk + ci * h
        #     Yi_fun = lambda Zi: (
        #         yk + sum([ai[j] * self.f(ti, Yis[j], Zis[j]) for j in range(i)])
        #         + ai[i] * self.f(ti, Yis[i], Zi)
        #     )
        #     ci_fun = self.gen_c(ti, Yi_fun)
        #     res = fsolve(ci_fun, np.zeros_like(zk), full_output=1)
        #     Zi = res[0]
        #     converged *= res[2]
        #     j += res[1]["nfev"]
        #     Yi = Yi_fun(Zi)

        #     tis.append(ti)
        #     Yis.append(Yi.copy())
        #     Zis.append(Zi.copy())

        # # final stage
        # tk1 = tk + h
        # s = len(b)
        # yk1_fun = lambda zk1: (
        #     yk + sum([b[i] * self.f(tis[i], Yis[i], Zis[i]) for i in range(s - 1)])
        #     + b[-1] * self.f(tk1, Yis[-1], zk1)
        # )
        # ck1_fun = self.gen_c(tk1, yk1_fun)
        # res = fsolve(ck1_fun, np.zeros_like(zk), full_output=1)
        # zk1 = res[0]
        # Zis.append(zk1.copy())
        # j += res[1]["nfev"]
        # converged *= res[2]
        # yk1 = yk1_fun(zk1)

        # zk1 = sum([b[i] * Zis[i] for i in range(s)])

        # ################################################
        # # modified version according to Arnold and Murua
        # ################################################

        # # consistent initial conditions
        # t1 = tk
        # Y1 = yk.copy()
        # Z1 = zk.copy()
        # tis = [t1]
        # Yis = [Y1.copy()]
        # Zis = [Z1.copy()]

        # # perform explicit stage without constraints
        # # Arnold1995 Definition 2.1
        # Y2 = yk + A[0][0] * self.f(tk + c[0] * h, Yis[0], Zis[0])
        # Yis.append(Y2)

        # # compute intermediate states
        # for i in range(1, len(c)):
        #     # print(f"i: {i}")

        #     # new time
        #     ti = tk + c[i] * h

        #     # function for update of Yi
        #     # Arnold1995 (2.3)
        #     # Hairer1996 (6.5b)
        #     Yi_fun = lambda Zi: (
        #         yk + sum([A[i][j] * self.f(ti, Yis[j], Zis[j]) for j in range(i)])
        #         + A[i][i] * self.f(ti, Yis[i], Zi)
        #     )

        #     # constraint functions
        #     gi_fun = self.gen_c(ti, Yi_fun)

        #     # solve zeros of constraint function
        #     res = fsolve(gi_fun, np.zeros_like(Z1), full_output=1)
        #     Zi = res[0]
        #     converged *= res[2]
        #     j += res[1]["nfev"]

        #     # compute used value Yi
        #     Yi = Yi_fun(Zi)

        #     tis.append(ti)
        #     Yis.append(Yi.copy())
        #     Zis.append(Zi.copy())

        # # final stage
        # tk1 = tk + h
        # yk1_fun = lambda Zs: (
        #     yk + sum([b[i] * self.f(tis[i], Yis[i], Zis[i]) for i in range(len(b) - 1)])
        #     # + b[-1] * self.f(tk1, Yis[-1], sum(Zis) + Zs)
        #     + b[-1] * self.f(tk1, Yis[-1], Zs)
        # )
        # ck1_fun = self.gen_c(tk1, yk1_fun)
        # res = fsolve(ck1_fun, np.zeros_like(Z1), full_output=1)
        # zk1 = res[0]
        # Zis.append(zk1.copy())
        # j += res[1]["nfev"]
        # converged *= res[2]
        # yk1 = yk1_fun(zk1)

        # # # final stage (backward Euler)
        # # tk1 = tk + h
        # # yk1_fun = lambda Z: yk + self.f(tk, yk, Z)
        # # ck1_fun = self.gen_c(tk, yk1_fun)
        # # # Jk1_fun = lambda Z: csr_matrix(approx_fprime(Z, ck1_fun))
        # # # res = fsolve(ck1_fun, Z0, full_output=True, fprime=Jk1_fun)
        # # res = fsolve(ck1_fun, Z0, full_output=True)
        # # zk1 = res[0]
        # # converged *= res[2]
        # # yk1 = yk1_fun(zk1)

        # # second stage
        # t2 = tk + c[0] * h
        # Y2_fun = lambda Z1: yk + A[0][0] * self.f(t2, Y1, Z1)
        # c2_fun = self.gen_c(t2, Y2_fun)
        # res2 = fsolve(c2_fun, Z0, full_output=1)
        # Z1 = res2[0]
        # converged *= res2[2]
        # Y2 = Y2_fun(Z1)

        # # final stage
        # tk1 = tk + h
        # yk1_fun = lambda Z1: yk + b[0] * self.f(t1, Y1, Z0) + b[1] * self.f(t2, Y2, Z1)
        # ck1_fun = self.gen_c(tk1, yk1_fun)
        # res = fsolve(ck1_fun, Z0, full_output=1)
        # zk1 = res[0]
        # converged *= res[2]
        # yk1 = yk1_fun(zk1)

        # update previous state and time
        self.yk = yk1
        self.tk = tk1

        # unpack (smooth) state
        qk1 = yk1[: self.nq].copy()
        uk1_free = yk1[self.nq :].copy()

        #################
        # compute impacts
        #################
        self.tk1 = tk1
        self.qk1 = qk1
        self.uk1_free = uk1_free.copy()

        sk1 = np.concatenate(
            (
                np.zeros(self.nu),
                np.zeros(self.nla_g),
                np.zeros(self.nla_gamma),
                np.zeros(self.nla_N),
                np.zeros(self.nla_F),
            )
        )

        res = fsolve(self.Ry, sk1, full_output=1)
        sk1 = res[0]
        converged *= res[2]
        j += res[1]["nfev"]

        Uk1, La_gk1, La_gammak1, La_Nk1, La_Fk1 = self.unpack_y(sk1)
        # print(f"Uk1: {Uk1}")

        uk1 = self.uk1_free + Uk1

        # ############
        # # no impacts
        # ############
        # uk1 = uk1_free

        # use new velocity for next yk
        self.yk = np.concatenate((qk1, uk1))

        # unpack percussions
        P_gk1 = zk1[: self.nla_g]
        mu_gk1 = zk1[self.nla_g : 2 * self.nla_g]
        P_gammak1 = zk1[2 * self.nla_g : 2 * self.nla_g + self.nla_gamma]
        P_Nk1 = zk1[
            2 * self.nla_g
            + self.nla_gamma : 2 * self.nla_g
            + self.nla_gamma
            + self.nla_N
        ]
        mu_Nk1 = zk1[
            2 * self.nla_g
            + self.nla_gamma
            + self.nla_N : 2 * self.nla_g
            + self.nla_gamma
            + 2 * self.nla_N
        ]
        P_Fk1 = zk1[2 * self.nla_g + self.nla_gamma + 2 * self.nla_N :]

        error = 0.0
        return (
            (converged, j, error),
            tk1,
            qk1,
            uk1,
            P_gk1,
            mu_gk1,
            P_gammak1,
            P_Nk1,
            mu_Nk1,
            P_Fk1,
        )

    def solve(self):
        # lists storing output variables
        q = [self.qk]
        u = [self.uk]
        P_g = [self.P_gk]
        P_gamma = [self.P_gammak]
        P_N = [self.P_Nk]
        P_F = [self.P_Fk]

        self.yk = np.concatenate((self.qk, self.uk))

        pbar = tqdm(self.t[:-1])
        for _ in pbar:
            (
                (converged, j, error),
                tk1,
                qk1,
                uk1,
                P_gk1,
                mu_gk1,
                P_gammak1,
                P_Nk1,
                mu_Nk1,
                P_Fk1,
            ) = self.step()
            pbar.set_description(
                # f"t: {tk1:0.2e}; fixed-point iterations: {j+1}; error: {error:.3e}"
                f"t: {tk1:0.2e}; Newton iterations: {j+1}; error: {error:.3e}"
            )
            if not converged:
                # raise RuntimeError(
                print(
                    # f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                    f"Newton iteration not converged after {j+1} iterations with error: {error:.5e}"
                )

            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            q.append(qk1)
            u.append(uk1)
            P_g.append(P_gk1)
            P_gamma.append(P_gammak1)
            P_N.append(P_Nk1)
            P_F.append(P_Fk1)

            # update local variables for accepted time step
            (
                self.tk,
                self.qk,
                self.uk,
                self.P_gk,
                self.mu_gk,
                self.P_gammak,
                self.P_Nk,
                self.mu_Nk1,
                self.P_Fk,
            ) = (tk1, qk1, uk1, P_gk1, mu_gk1, P_gammak1, P_Nk1, mu_Nk1, P_Fk1)

        return Solution(
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
        )


class NonsmoothHalfExplicitEulerGGL:
    def __init__(
        self,
        model,
        t1,
        dt,
        atol=1e-6,
        max_iter=50,
        error_function=lambda x: np.max(np.abs(x)),
    ):
        self.model = model

        # integration time
        t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.error_function = error_function
        self.atol = atol
        self.max_iter = max_iter

        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = model.nla_g
        self.nla_gamma = model.nla_gamma
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F

        # initial state
        self.tk = t0
        self.qk = model.q0
        self.uk = model.u0

        # initial accelerations
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        # TODO: Solve for initial Lagrange multipliers!
        M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        h0 = self.model.h(self.tk, self.qk, self.uk)
        W_g0 = self.model.W_g(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_gamma0 = self.model.W_gamma(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_F0 = self.model.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)
        self.u_dotk = spsolve(
            M0,
            h0
            + W_g0 @ model.la_g0
            + W_gamma0 @ model.la_gamma0
            + W_N0 @ model.la_N0
            + W_F0 @ model.la_F0,
        )

        # TODO: Add solve for bilateral constraints!
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.P_gk = dt * model.la_g0
        self.mu_gk = np.zeros(self.nla_g)
        self.P_gammak = dt * model.la_gamma0
        self.P_Nk = dt * model.la_N0
        self.mu_Nk = np.zeros(self.nla_N)
        self.P_Fk = dt * model.la_F0

    def c(self, zk1, update_index_set=True):
        # unpack percussions
        P_gk1 = zk1[: self.nla_g]
        mu_gk1 = zk1[self.nla_g : 2 * self.nla_g]
        P_gammak1 = zk1[2 * self.nla_g : 2 * self.nla_g + self.nla_gamma]
        P_Nk1 = zk1[
            2 * self.nla_g
            + self.nla_gamma : 2 * self.nla_g
            + self.nla_gamma
            + self.nla_N
        ]
        mu_Nk1 = zk1[
            2 * self.nla_g
            + self.nla_gamma
            + self.nla_N : 2 * self.nla_g
            + self.nla_gamma
            + 2 * self.nla_N
        ]
        P_Fk1 = zk1[2 * self.nla_g + self.nla_gamma + 2 * self.nla_N :]

        #####################
        # explicit Euler step
        #####################
        tk1 = self.tk + self.dt
        yk = np.concatenate((self.qk, self.uk))
        # yk1 = yk + self.f_G(self.tk, yk, zk1)
        yk1 = yk + self.f(self.tk, yk) + self.G_z(self.tk, yk, zk1)

        # # #######################################################################
        # # # three-stage Runge-Kutta,
        # # # see https://de.wikipedia.org/wiki/Runge-Kutta-Verfahren#Beispiel
        # # # This is not so easy and requires multiple solutions of c1, c2, c3,
        # # # etc. See Hairer1996, Section VII.6, p 520.
        # # #######################################################################
        # dt = self.dt
        # tk = self.tk
        # yk = np.concatenate((self.qk, self.uk))
        # k1 = self.f(tk, yk, np.zeros_like(zk1))
        # k2 = self.f(tk + 0.5 * dt, yk + 0.5 * k1, np.zeros_like(zk1))
        # k3 = self.f(tk + 1.0 * dt, yk - 1.0 * k1 + 2.0 * k2, 6 * zk1)
        # # k1 = self.f(tk, yk, zk1 / 6)
        # # k2 = self.f(tk + 0.5 * dt, yk + 0.5 * k1, zk1 * 4 / 6)
        # # k3 = self.f(tk + 1.0 * dt, yk - 1.0 * k1 + 2.0 * k2, zk1 / 6)
        # yk1 = yk + (k1 / 6 + 4 * k2 / 6 + k3 / 6)

        # different methods, see https://de.wikipedia.org/wiki/Runge-Kutta-Verfahren#Beispiele

        # ###############
        # # forward Euler
        # ###############
        # c = [0.0]

        # A = [[]]

        # b = [1.0]

        # # ################
        # # # midpoint rule,
        # # # see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Second-order_methods_with_two_stages
        # # ################
        # # c = [
        # #     0,
        # #     0.5,
        # # ]

        # # A = [
        # #     [],
        # #     [0.5],
        # # ]

        # # b = [
        # #     0.0,
        # #     1.0
        # # ]

        # ##########################
        # # classical Runge-Kutta 4,
        # # see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Examples
        # ##########################
        # ################
        # c = [
        #     0,
        #     0.5,
        #     0.5,
        #     1.0,
        # ]

        # A = [
        #     [],
        #     [0.5],
        #     [0.0, 0.5],
        #     [0.0, 0.0, 1.0],
        # ]

        # b = [
        #     1 / 6,
        #     1 / 3,
        #     1 / 3,
        #     1 / 6,
        # ]

        # dt = self.dt
        # tk = self.tk
        # yk = np.concatenate((self.qk, self.uk))

        # k = []
        # for ci, ai in zip(c, A):
        #     ti = tk + ci * dt
        #     Yi = yk + sum([aij * kj for aij, kj in zip(ai, k)])
        #     k.append(self.f(ti, Yi))

        # yk1 = yk + sum([bj * kj for bj, kj in zip(b, k)]) + self.G_z(ti, Yi, zk1)

        # ######################################
        # # HEM4, see Brasey1993, table 5,
        # # https://www.jstor.org/stable/2158176
        # ######################################
        # # c = np.array([
        # #     3 / 10,
        # #     (4 - sqrt(6) / 10),
        # #     (4 + sqrt(6) / 10),
        # #     1,
        # # ], dtype=float)

        # # A = np.array([
        # #     [3 / 10, 0, 0, 0],
        # #     [(1 + sqrt(6)) / 10, (11 - 4 * sqrt(6)) / 30, 0, 0]
        # # ])

        ################################################
        # extract final solution for Runge-Kutta methods
        ################################################
        tk1 = self.tk + self.dt
        qk1 = yk1[: self.nq]
        uk1 = yk1[self.nq :]

        # bilateral constraints
        gk1 = self.model.g(tk1, qk1)
        g_dotk1 = self.model.g_dot(tk1, qk1, uk1)
        gammak1 = self.model.gamma(tk1, qk1, uk1)

        # unilateral constraints
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        prox_arg_pos = g_Nk1 - self.model.prox_r_N * mu_Nk1
        # c_Nk1_stab = g_Nk1 - prox_R0_np(prox_arg_pos)
        c_Nk1_stab = mu_Nk1

        if update_index_set:
            self.Ak1 = g_Nk1 <= 0
            # # TODO: This together with u_k1_free removes chattering!
            # self.Ak1 = prox_arg_pos <= 0

        c_Nk1 = np.where(
            self.Ak1, xi_Nk1 - prox_R0_np(xi_Nk1 - self.model.prox_r_N * P_Nk1), P_Nk1
        )

        ##########
        # friction
        ##########
        mu = self.model.mu
        c_Fk1 = P_Fk1.copy()  # this is the else case => P_Fk1 = 0
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                # TODO: Is there a primal/ dual form?
                if self.Ak1[i_N]:
                    c_Fk1[i_F] = -P_Fk1[i_F] - prox_sphere(
                        -P_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * P_Nk1[i_N],
                    )

        ck1 = np.concatenate((g_dotk1, gk1, gammak1, c_Nk1, c_Nk1_stab, c_Fk1))

        return ck1, tk1, qk1, uk1, P_gk1, mu_gk1, P_gammak1, P_Nk1, mu_Nk1, P_Fk1

    def c_y(self, zk1):
        return csr_matrix(
            approx_fprime(zk1, lambda z: self.c(z, update_index_set=False)[0])
            # approx_fprime(
            #     zk1, lambda z: self.c(z, update_index_set=False)[0], method="3-point"
            # )
        )

    def f_G(self, tk, yk, zk1):
        # unpack state
        qk = yk[: self.nq]
        uk = yk[self.nq : self.nq + self.nu]

        # unpack percussions
        P_gk1 = zk1[: self.nla_g]
        mu_gk1 = zk1[self.nla_g : 2 * self.nla_g]
        P_gammak1 = zk1[2 * self.nla_g : 2 * self.nla_g + self.nla_gamma]
        P_Nk1 = zk1[
            2 * self.nla_g
            + self.nla_gamma : 2 * self.nla_g
            + self.nla_gamma
            + self.nla_N
        ]
        mu_Nk1 = zk1[
            2 * self.nla_g
            + self.nla_gamma
            + self.nla_N : 2 * self.nla_g
            + self.nla_gamma
            + 2 * self.nla_N
        ]
        P_Fk1 = zk1[2 * self.nla_g + self.nla_gamma + 2 * self.nla_N :]

        # evaluate quantities of previous time step
        Mk = self.model.M(tk, qk, scipy_matrix=csr_matrix)
        hk = self.model.h(tk, qk, uk)
        W_gk = self.model.W_g(tk, qk, scipy_matrix=csr_matrix)
        W_gammak = self.model.W_gamma(tk, qk, scipy_matrix=csr_matrix)
        g_qk = self.model.g_q(tk, qk, scipy_matrix=csr_matrix)
        W_Nk = self.model.W_N(tk, qk, scipy_matrix=csr_matrix)
        g_N_qk = self.model.g_N_q(tk, qk, scipy_matrix=csr_matrix)
        W_Fk = self.model.W_F(tk, qk, scipy_matrix=csr_matrix)

        # explicit Euler step
        dt = self.dt
        # TODO: How to get compute free velocity for higher order Runge-Kutta methods?
        # uk1_free = self.uk + spsolve(Mk, hk * dt)
        return np.concatenate(
            (
                self.model.q_dot(tk, qk, uk) * dt
                + g_qk.T @ mu_gk1
                + g_N_qk.T @ mu_Nk1,  # original version
                # self.model.q_dot(tk, qk, uk1_free) * dt
                # + g_qk.T @ mu_gk1
                # + g_N_qk.T @ mu_Nk1,  # TODO: This works without chattering!
                spsolve(
                    Mk,
                    hk * dt
                    + W_gk @ P_gk1
                    + W_gammak @ P_gammak1
                    + W_Nk @ P_Nk1
                    + W_Fk @ P_Fk1,
                ),
            )
        )

    def f(self, tk, yk):
        # unpack state
        qk = yk[: self.nq]
        uk = yk[self.nq : self.nq + self.nu]

        # evaluate quantities of previous time step
        Mk = self.model.M(tk, qk, scipy_matrix=csr_matrix)
        hk = self.model.h(tk, qk, uk)

        # explicit Euler step
        dt = self.dt

        # TODO: How to get compute free velocity for higher order Runge-Kutta methods?
        u_dotk1_free = spsolve(Mk, hk * dt)
        uk1_free = uk + u_dotk1_free
        return np.concatenate(
            (
                # self.model.q_dot(tk, qk, uk) * dt,  # original version
                self.model.q_dot(tk, qk, uk1_free) * dt,
                u_dotk1_free,
            )
        )

    def G_z(self, tk, yk, zk1):
        # unpack state
        qk = yk[: self.nq]

        # unpack percussions
        P_gk1 = zk1[: self.nla_g]
        mu_gk1 = zk1[self.nla_g : 2 * self.nla_g]
        P_gammak1 = zk1[2 * self.nla_g : 2 * self.nla_g + self.nla_gamma]
        P_Nk1 = zk1[
            2 * self.nla_g
            + self.nla_gamma : 2 * self.nla_g
            + self.nla_gamma
            + self.nla_N
        ]
        mu_Nk1 = zk1[
            2 * self.nla_g
            + self.nla_gamma
            + self.nla_N : 2 * self.nla_g
            + self.nla_gamma
            + 2 * self.nla_N
        ]
        P_Fk1 = zk1[2 * self.nla_g + self.nla_gamma + 2 * self.nla_N :]

        # evaluate quantities of previous time step
        Mk = self.model.M(tk, qk, scipy_matrix=csc_matrix)
        W_gk = self.model.W_g(tk, qk, scipy_matrix=csr_matrix)
        W_gammak = self.model.W_gamma(tk, qk, scipy_matrix=csr_matrix)
        g_qk = self.model.g_q(tk, qk, scipy_matrix=csr_matrix)
        W_Nk = self.model.W_N(tk, qk, scipy_matrix=csr_matrix)
        g_N_qk = self.model.g_N_q(tk, qk, scipy_matrix=csr_matrix)
        W_Fk = self.model.W_F(tk, qk, scipy_matrix=csr_matrix)

        return np.concatenate(
            [
                g_qk.T @ mu_gk1 + g_N_qk.T @ mu_Nk1,
                spsolve(
                    Mk,
                    W_gk @ P_gk1 + W_gammak @ P_gammak1 + W_Nk @ P_Nk1 + W_Fk @ P_Fk1,
                ),
            ]
        )

    def step(self):
        # from scipy.optimize import fsolve
        # zk1 = np.concatenate(
        #     (self.P_gk, self.mu_gk, self.P_gammak, self.P_Nk, self.mu_Nk, self.P_Fk)
        # )
        # res = fsolve(lambda z: self.c(z)[0], zk1, full_output=1)

        # j = res[1]["nfev"]
        # converged = res[2]
        # zk1 = res[0]

        # ck1, tk1, qk1, uk1, P_gk1, mu_gk1, P_gammak1, P_Nk1, mu_Nk1, P_Fk1 = self.c(zk1)
        # error = self.error_function(ck1)

        # return (converged, j, error), tk1, qk1, uk1, P_gk1, mu_gk1, P_gammak1, P_Nk1, mu_Nk1, P_Fk1

        j = 0
        converged = False
        zk1 = np.concatenate(
            (self.P_gk, self.mu_gk, self.P_gammak, self.P_Nk, self.mu_Nk, self.P_Fk)
        )
        ck1, tk1, qk1, uk1, P_gk1, mu_gk1, P_gammak1, P_Nk1, mu_Nk1, P_Fk1 = self.c(zk1)

        error = self.error_function(ck1)
        converged = error < self.atol
        while (not converged) and (j < self.max_iter):
            j += 1

            # compute Jacobian and make a Newton step
            c_yk1 = self.c_y(zk1)

            zk1 -= spsolve(c_yk1, ck1)

            # from scipy.sparse.linalg import lsqr
            # zk1 -= lsqr(c_yk1, ck1)[0]

            # zk1 -= np.linalg.lstsq(c_yk1.toarray(), ck1, rcond=None)[0]

            # zk1 -= spsolve(c_yk1.T @ c_yk1, c_yk1.T @ ck1)

            # # fixed-point iteration
            # # self.atol = 1.0e-4
            # # r = 7.0e-1
            # r = 4.0e-1
            # # r = 3.0e-1
            # zk1 -= r * ck1

            # check error for new Lagrange multipliers
            ck1, tk1, qk1, uk1, P_gk1, mu_gk1, P_gammak1, P_Nk1, mu_Nk1, P_Fk1 = self.c(
                zk1
            )
            error = self.error_function(ck1)
            converged = error < self.atol

        if not converged:
            # raise RuntimeError("Internal Newton-Raphson scheme is not converged!")
            print(f"Internal Newton-Raphson scheme is not converged with error {error}")

        return (
            (converged, j, error),
            tk1,
            qk1,
            uk1,
            P_gk1,
            mu_gk1,
            P_gammak1,
            P_Nk1,
            mu_Nk1,
            P_Fk1,
        )

    def solve(self):
        # lists storing output variables
        q = [self.qk]
        u = [self.uk]
        P_g = [self.P_gk]
        mu_g = [self.mu_gk]
        P_gamma = [self.P_gammak]
        P_N = [self.P_Nk]
        mu_N = [self.mu_Nk]
        P_F = [self.P_Fk]

        pbar = tqdm(self.t[:-1])
        for _ in pbar:
            (
                (converged, j, error),
                tk1,
                qk1,
                uk1,
                P_gk1,
                mu_gk1,
                P_gammak1,
                P_Nk1,
                mu_Nk1,
                P_Fk1,
            ) = self.step()
            pbar.set_description(
                # f"t: {tk1:0.2e}; fixed-point iterations: {j+1}; error: {error:.3e}"
                f"t: {tk1:0.2e}; Newton iterations: {j+1}; error: {error:.3e}"
            )
            if not converged:
                # raise RuntimeError(
                print(
                    # f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                    f"Newton iteration not converged after {j+1} iterations with error: {error:.5e}"
                )

            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            q.append(qk1)
            u.append(uk1)
            P_g.append(P_gk1)
            mu_g.append(mu_gk1)
            P_gamma.append(P_gammak1)
            P_N.append(P_Nk1)
            mu_N.append(mu_Nk1)
            P_F.append(P_Fk1)

            # update local variables for accepted time step
            (
                self.tk,
                self.qk,
                self.uk,
                self.P_gk,
                self.mu_gk,
                self.P_gammak,
                self.P_Nk,
                self.mu_Nk,
                self.P_Fk,
            ) = (tk1, qk1, uk1, P_gk1, mu_gk1, P_gammak1, P_Nk1, mu_Nk1, P_Fk1)

        return Solution(
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
            P_g=np.array(P_g),
            mu_g=np.array(mu_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            mu_N=np.array(mu_N),
            P_F=np.array(P_F),
        )


class NonsmoothHalfExplicitEulerGGLWithImpactEquation:
    def __init__(
        self,
        model,
        t1,
        dt,
        atol=1e-6,
        max_iter=50,
        error_function=lambda x: np.max(np.abs(x)),
    ):
        self.model = model

        # integration time
        t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.error_function = error_function
        self.atol = atol
        self.max_iter = max_iter

        self.nq = model.nq
        self.nu = model.nu
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F

        # initial state
        self.tk = t0
        self.qk = model.q0
        self.uk = model.u0

        # initial accelerations
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        # TODO: Solve for initial Lagrange multipliers!
        M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        h0 = self.model.h(self.tk, self.qk, self.uk)
        W_g0 = self.model.W_g(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_F0 = self.model.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)
        self.u_dotk = spsolve(
            M0, h0 + W_g0 @ model.la_g0 + W_N0 @ model.la_N0 + W_F0 @ model.la_F0
        )

        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.mu_Nk = np.zeros(self.nla_N)
        self.P_Nk = dt * model.la_N0
        self.P_Fk = dt * model.la_F0

    def c(self, zk1, update_index_set=True):
        # unpack percussions
        P_Nk1 = zk1[: self.nla_N]
        mu_Nk1 = zk1[self.nla_N : 2 * self.nla_N]
        P_Fk1 = zk1[2 * self.nla_N :]

        #####################
        # explicit Euler step
        #####################
        tk1 = self.tk + self.dt
        yk = np.concatenate((self.qk, self.uk))
        self.Uk = np.zeros(self.nu)
        yk = np.concatenate((self.qk, self.uk, self.Uk))
        yk1 = yk + self.dt * self.f(self.tk, yk, zk1)
        qk1 = yk1[: self.nq]
        uk1 = yk1[self.nq : self.nq + self.nu]
        Uk1 = yk1[self.nq + self.nu :]

        uk1 += Uk1

        # #######################################################################
        # # three-stage Runge-Kutta,
        # # see https://de.wikipedia.org/wiki/Runge-Kutta-Verfahren#Beispiel
        # # This is not so easy and requires multiple solutions of c1, c2, c3,
        # # etc. See Hairer1996, Section VII.6, p 520.
        # #######################################################################
        # dt = self.dt
        # tk = self.tk
        # yk = np.concatenate((self.qk, self.uk))
        # k1 = self.f(tk, yk, P_Nk1)
        # k2 = self.f(tk + 0.5 * dt, yk + 0.5 * k1, P_Nk1)
        # k3 = self.f(tk + 1.0 * dt, yk - 1.0 * k1 + 2.0 * k2, P_Nk1)
        # yk1 = yk + dt * (k1 / 6 + 4 * k2 / 6 + k3 / 6)

        # tk1 = tk + dt
        # qk1 = yk1[:self.nq]
        # uk1 = yk1[self.nq:]

        # constraint equations
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        prox_arg_pos = g_Nk1 - self.model.prox_r_N * mu_Nk1
        c_Nk1_stab = g_Nk1 - prox_R0_np(prox_arg_pos)

        if update_index_set:
            # self.Ak1 = g_Nk1 <= 0
            self.Ak1 = prox_arg_pos <= 0

        c_Nk1 = np.where(
            self.Ak1, xi_Nk1 - prox_R0_np(xi_Nk1 - self.model.prox_r_N * P_Nk1), P_Nk1
        )

        ##########
        # friction
        ##########
        mu = self.model.mu
        c_Fk1 = P_Fk1.copy()  # this is the else case => P_Fk1 = 0
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                # TODO: Is there a primal/ dual form?
                if self.Ak1[i_N]:
                    c_Fk1[i_F] = -P_Fk1[i_F] - prox_sphere(
                        -P_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * P_Nk1[i_N],
                    )

        ck1 = np.concatenate((c_Nk1, c_Nk1_stab, c_Fk1))

        return ck1, tk1, qk1, uk1, P_Nk1, mu_Nk1, P_Fk1

    def c_y(self, zk1):
        return csr_matrix(
            approx_fprime(zk1, lambda z: self.c(z, update_index_set=False)[0])
        )

    def f(self, tk, yk, zk1):
        # unpack state
        qk = yk[: self.nq]
        uk = yk[self.nq : self.nq + self.nu]
        Uk = yk[self.nq + self.nu :]

        # unpack percussions
        P_Nk1 = zk1[: self.nla_N]
        mu_Nk1 = zk1[self.nla_N : 2 * self.nla_N]
        P_Fk1 = zk1[2 * self.nla_N :]

        P_Nk1 = mu_Nk1 * self.dt + P_Nk1.copy()

        # evaluate quantities of previous time step
        Mk = self.model.M(tk, qk, scipy_matrix=csr_matrix)
        hk = self.model.h(tk, qk, uk + Uk)
        W_Nk = self.model.W_N(tk, qk, scipy_matrix=csr_matrix)
        g_N_qk = self.model.g_N_q(tk, qk, scipy_matrix=csr_matrix)
        W_Fk = self.model.W_F(tk, qk, scipy_matrix=csr_matrix)

        # explicit Euler step
        dt = self.dt
        return np.concatenate(
            (
                # self.model.q_dot(tk, qk, uk) + g_N_qk.T @ mu_Nk1,
                # spsolve(Mk, hk + W_Nk @ P_Nk1 / dt + W_Fk @ P_Fk1 / dt),
                self.model.q_dot(tk, qk, uk + Uk),
                spsolve(Mk, hk + W_Nk @ mu_Nk1),
                spsolve(Mk, W_Nk @ P_Nk1 + W_Fk @ P_Fk1),
            )
        )

    def step(self):
        j = 0
        converged = False
        zk1 = np.concatenate((self.P_Nk, self.mu_Nk, self.P_Fk))
        ck1, tk1, qk1, uk1, P_Nk1, mu_Nk1, P_Fk1 = self.c(zk1)

        error = self.error_function(ck1)
        converged = error < self.atol
        while (not converged) and (j < self.max_iter):
            j += 1

            # compute Jacobian and make a Newton step
            c_yk1 = self.c_y(zk1)

            # zk1 -= spsolve(c_yk1, ck1)

            # from scipy.sparse.linalg import lsqr
            # zk1 -= lsqr(c_yk1, ck1)[0]

            # zk1 -= np.linalg.lstsq(c_yk1.toarray(), ck1, rcond=None)[0]

            zk1 -= spsolve(c_yk1.T @ c_yk1, c_yk1.T @ ck1)

            # # fixed-point iteration
            # # self.atol = 1.0e-4
            # r = 4.0e-1
            # # r = 1.0e-2
            # zk1 -= r * ck1

            # check error for new Lagrange multipliers
            ck1, tk1, qk1, uk1, P_Nk1, mu_Nk1, P_Fk1 = self.c(zk1)
            error = self.error_function(ck1)
            converged = error < self.atol

        if not converged:
            # raise RuntimeError("Internal Newton-Raphson scheme is not converged!")
            print(f"Internal Newton-Raphson scheme is not converged with error {error}")

        return (converged, j, error), tk1, qk1, uk1, P_Nk1, mu_Nk1, P_Fk1

    def solve(self):
        # lists storing output variables
        q = [self.qk]
        u = [self.uk]
        P_N = [self.P_Nk]
        mu_N = [self.mu_Nk]
        P_F = [self.P_Fk]

        pbar = tqdm(self.t[:-1])
        for _ in pbar:
            (
                (converged, j, error),
                tk1,
                qk1,
                uk1,
                P_Nk1,
                mu_Nk1,
                P_Fk1,
            ) = self.step()
            pbar.set_description(
                # f"t: {tk1:0.2e}; fixed-point iterations: {j+1}; error: {error:.3e}"
                f"t: {tk1:0.2e}; Newton iterations: {j+1}; error: {error:.3e}"
            )
            if not converged:
                # raise RuntimeError(
                print(
                    # f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                    f"Newton iteration not converged after {j+1} iterations with error: {error:.5e}"
                )

            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            q.append(qk1)
            u.append(uk1)
            P_N.append(P_Nk1)
            mu_N.append(mu_Nk1)
            P_F.append(P_Fk1)

            # update local variables for accepted time step
            (
                self.tk,
                self.qk,
                self.uk,
                self.P_Nk,
                self.mu_Nk,
                self.P_Fk,
            ) = (tk1, qk1, uk1, P_Nk1, mu_Nk1, P_Fk1)

        return Solution(
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
            P_N=np.array(P_N),
            mu_N=np.array(mu_N),
            P_F=np.array(P_F),
        )


class NonsmoothHalfExplicitEulerGGLOld:
    def __init__(
        self,
        model,
        t1,
        dt,
        atol=1e-6,
        max_iter=50,
        # max_iter=10000,
        error_function=lambda x: np.max(np.abs(x)),
    ):
        self.model = model

        # integration time
        t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.error_function = error_function
        self.atol = atol
        self.max_iter = max_iter

        self.nq = model.nq
        self.nu = model.nu
        self.nla_N = model.nla_N
        self.nR = self.nu + 2 * self.nla_N

        # initial state
        self.tk = t0
        self.qk = model.q0
        self.uk = model.u0

        # initial accelerations
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        h0 = self.model.h(self.tk, self.qk, self.uk)
        W_g0 = self.model.W_g(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_F0 = self.model.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)
        self.u_dotk = spsolve(
            M0, h0 + W_g0 @ model.la_g0 + W_N0 @ model.la_N0 + W_F0 @ model.la_F0
        )

        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.Uk = np.zeros(model.nu)
        self.P_Nk = dt * model.la_N0
        self.mu_Nk = np.zeros(self.nla_N)

    def c(self, kappa_k1, update_index_set=True):
        mu_Nk1 = kappa_k1[: self.nla_N]
        P_Nk1 = kappa_k1[self.nla_N :]

        # values from previous time step
        tk = self.tk
        qk = self.qk
        uk = self.uk

        # evaluate quantities of previous time step
        Mk = self.model.M(tk, qk, scipy_matrix=csc_matrix)
        W_gk = self.model.W_N(tk, qk, scipy_matrix=csc_matrix)
        g_qk = self.model.g_N_q(tk, qk, scipy_matrix=csr_matrix)
        hk = self.model.h(tk, qk, uk)

        # explicit Euler step
        tk1 = self.tk + self.dt
        qk1 = self.qk + self.dt * self.model.q_dot(tk, qk, uk) + g_qk.T @ mu_Nk1
        uk1 = self.uk + self.dt * spsolve(Mk, hk + W_gk @ P_Nk1)

        # constraint equations
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, uk, uk1)

        prox_r_N = 1.0e-1
        prox_N_arg_position = g_Nk1 - prox_r_N * mu_Nk1
        prox_N_arg_velocity = xi_Nk1 - prox_r_N * P_Nk1
        if update_index_set:
            self.Ak1 = prox_N_arg_position <= 0

        from cardillo.math.prox import prox_R0_np

        c1_k1 = g_Nk1 - prox_R0_np(prox_N_arg_position)
        c2_k1 = np.where(self.Ak1, xi_Nk1 - prox_R0_np(prox_N_arg_velocity), P_Nk1)

        ck1 = np.concatenate((c1_k1, c2_k1))
        return ck1, tk1, qk1, uk1, mu_Nk1, P_Nk1

    def c_kappa(self, kappa_k1):
        return csr_matrix(
            approx_fprime(
                kappa_k1, lambda kappa: self.c(kappa, update_index_set=False)[0]
            )
        )

    def f(self, tk, qk, uk, mu_k1, P_Nk1):
        # evaluate quantities of previous time step
        Mk = self.model.M(tk, qk, scipy_matrix=csr_matrix)
        W_gk = self.model.W_N(tk, qk, scipy_matrix=csr_matrix)
        g_qk = self.model.g_N_q(tk, qk, scipy_matrix=csr_matrix)
        hk = self.model.h(tk, qk, uk)

        # explicit Euler step
        tk1 = self.tk + self.dt
        qk1 = self.qk + self.dt * self.model.q_dot(tk, qk, uk) + g_qk.T @ mu_k1
        uk1 = self.uk + self.dt * spsolve(Mk, hk + W_gk @ P_Nk1)

        return tk1, qk1, uk1

    def step(self):
        j = 0
        converged = False
        kappa_k1 = np.concatenate((self.mu_Nk, self.P_Nk))
        ck1, tk1, qk1, uk1, mu_Nk1, P_Nk1 = self.c(kappa_k1)

        # ck1, tk1, qk1, uk1, mu_gk1, la_gk1 = next(gen)
        error = self.error_function(ck1)
        # print(f"step: {j}; error: {error}")
        converged = error < self.atol
        while (not converged) and (j < self.max_iter):
            j += 1

            # compute Jacobian and make a Newton step
            c_kappak1 = self.c_kappa(kappa_k1)
            # kappa_k1 -= spsolve(c_kappak1, ck1)
            from scipy.sparse.linalg import lsqr

            # kappa_k1 -= lsqr(c_kappak1, ck1)[0]
            # kappa_k1 -= np.linalg.lstsq(c_kappak1.toarray(), ck1, rcond=None)[0]
            kappa_k1 -= spsolve(c_kappak1.T @ c_kappak1, c_kappak1.T @ ck1)

            # # fixed-point iteration
            # # self.atol = 1.0e-4
            # r = 4.0e-1
            # # r = 1.0e-2
            # kappa_k1 -= r * ck1

            # c_kappak1_num = approx_fprime(kappa_k1, lambda kappa: next(self.c(kappa))[0], method="2-point")
            # # diff = c_kappak1.toarray() - c_kappak1_num
            # # error = np.linalg.norm(diff)
            # # np.set_printoptions(3, suppress=True)
            # # print(f"diff:\n{diff}")
            # # print(f"error jacobian: {error}")
            # kappa_k1 -= np.linalg.solve(c_kappak1_num, ck1)

            # check error for new Lagrange multipliers
            # gen = self.c(kappa_k1)
            # ck1, tk1, qk1, uk1, mu_gk1, la_gk1 = next(gen)
            ck1, tk1, qk1, uk1, mu_Nk1, P_Nk1 = self.c(kappa_k1)
            error = self.error_function(ck1)
            converged = error < self.atol
            # print(f"step: {j}; error: {error:0.3e}")
            # print(f"")

        if not converged:
            # raise RuntimeError("Internal Newton-Raphson scheme is not converged!")
            print(f"Internal Newton-Raphson scheme is not converged with error {error}")

        return (converged, j, error), tk1, qk1, uk1, mu_Nk1, P_Nk1

    def solve(self):
        # lists storing output variables
        q = [self.qk]
        u = [self.uk]
        mu_N = [self.mu_Nk]
        P_N = [self.P_Nk]

        pbar = tqdm(self.t[:-1])
        for _ in pbar:
            (
                (converged, j, error),
                tk1,
                qk1,
                uk1,
                mu_Nk1,
                P_Nk1,
            ) = self.step()
            pbar.set_description(
                # f"t: {tk1:0.2e}; fixed-point iterations: {j+1}; error: {error:.3e}"
                f"t: {tk1:0.2e}; Newton iterations: {j+1}; error: {error:.3e}"
            )
            if not converged:
                # raise RuntimeError(
                print(
                    # f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                    f"Newton iteration not converged after {j+1} iterations with error: {error:.5e}"
                )

            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            q.append(qk1)
            u.append(uk1)
            mu_N.append(mu_Nk1)
            P_N.append(P_Nk1)

            # update local variables for accepted time step
            (
                self.tk,
                self.qk,
                self.uk,
                self.mu_Nk,
                self.P_Nk,
            ) = (tk1, qk1, uk1, mu_Nk1, P_Nk1)

        return Solution(
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
            mu_N=np.array(mu_N),
            P_N=np.array(P_N),
        )
