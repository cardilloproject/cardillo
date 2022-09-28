import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, csc_matrix, bmat
from tqdm import tqdm

from cardillo.math.prox import prox_R0_np, prox_sphere
from cardillo.math import approx_fprime
from cardillo.solver import Solution


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
            # kappa_k1 -= spsolve(c_kappak1, ck1)
            # kappa_k1 -= spsolve(c_kappak1, ck1)
            kappa_k1 -= np.linalg.lstsq(c_kappak1.toarray(), ck1)[0]

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


class NonsmoothHalfExplicitEuler:
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
        self.P_Nk = dt * model.la_N0
        self.P_Fk = dt * model.la_F0

    def c(self, zk1, update_index_set=True):
        # unpack percussions
        P_Nk1 = zk1[: self.nla_N]
        P_Fk1 = zk1[self.nla_N :]

        #####################
        # explicit Euler step
        #####################
        tk1 = self.tk + self.dt
        yk = np.concatenate((self.qk, self.uk))
        # yk1 = yk + self.f(self.tk, yk, zk1)
        yk1 = yk + self.f(self.tk, yk, zk1)
        qk1 = yk1[: self.nq]
        uk1 = yk1[self.nq :]

        # #######################################################################
        # # three-stage Runge-Kutta,
        # # see https://de.wikipedia.org/wiki/Runge-Kutta-Verfahren#Beispiel
        # # This is not so easy and requires multiple solutions of c1, c2, c3,
        # # etc. See Hairer1996, Section VII.6, p 520.
        # #######################################################################
        # dt = self.dt
        # tk = self.tk
        # yk = np.concatenate((self.qk, self.uk))
        # k1 = self.f(tk, yk, np.zeros_like(zk1))
        # k2 = self.f(tk + 0.5 * dt, yk + 0.5 * k1, np.zeros_like(zk1))
        # k3 = self.f(tk + 1.0 * dt, yk - 1.0 * k1 + 2.0 * k2, 6 * P_Nk1)
        # # k1 = self.f(tk, yk, P_Nk1 / 6)
        # # k2 = self.f(tk + 0.5 * dt, yk + 0.5 * k1, P_Nk1 * 4 / 6)
        # # k3 = self.f(tk + 1.0 * dt, yk - 1.0 * k1 + 2.0 * k2, P_Nk1 / 6)
        # yk1 = yk + (k1 / 6 + 4 * k2 / 6 + k3 / 6)

        # tk1 = tk + dt
        # qk1 = yk1[:self.nq]
        # uk1 = yk1[self.nq:]

        # constraint equations
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        if update_index_set:
            self.Ak1 = g_Nk1 <= 0

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

        ck1 = np.concatenate((c_Nk1, c_Fk1))

        return ck1, tk1, qk1, uk1, P_Nk1, P_Fk1

    def c_y(self, zk1):
        return csr_matrix(
            approx_fprime(zk1, lambda z: self.c(z, update_index_set=False)[0])
        )

    def f(self, tk, yk, zk1):
        # unpack y and z
        qk = yk[: self.nq]
        uk = yk[self.nq :]
        P_Nk1 = zk1[: self.nla_N]
        P_Fk1 = zk1[self.nla_N :]

        # evaluate quantities of previous time step
        Mk = self.model.M(tk, qk, scipy_matrix=csr_matrix)
        hk = self.model.h(tk, qk, uk)
        W_Nk = self.model.W_N(tk, qk, scipy_matrix=csr_matrix)
        W_Fk = self.model.W_F(tk, qk, scipy_matrix=csr_matrix)

        # explicit Euler step
        dt = self.dt
        return np.concatenate(
            (
                self.model.q_dot(tk, qk, uk) * dt,
                spsolve(Mk, hk * dt + W_Nk @ P_Nk1 + W_Fk @ P_Fk1),
            )
        )

    def step(self):
        j = 0
        converged = False
        zk1 = np.concatenate((self.P_Nk, self.P_Fk))
        # zk1 = np.zeros(self.nla_N + self.nla_F)
        ck1, tk1, qk1, uk1, P_Nk1, P_Fk1 = self.c(zk1)

        error = self.error_function(ck1)
        converged = error < self.atol
        while (not converged) and (j < self.max_iter):
            j += 1

            # compute Jacobian and make a Newton step
            c_yk1 = self.c_y(zk1)

            zk1 -= spsolve(c_yk1, ck1)

            # from scipy.sparse.linalg import lsqr
            # zk1 -= lsqr(c_yk1, ck1)[0]

            # # zk1 -= np.linalg.lstsq(c_yk1.toarray(), ck1, rcond=None)[0]
            # zk1 -= spsolve(c_yk1.T @ c_yk1, c_yk1.T @ ck1)

            # # fixed-point iteration
            # # self.atol = 1.0e-4
            # r = 4.0e-1
            # # r = 1.0e-2
            # zk1 -= r * ck1

            # check error for new Lagrange multipliers
            ck1, tk1, qk1, uk1, P_Nk1, P_Fk1 = self.c(zk1)
            error = self.error_function(ck1)
            converged = error < self.atol

        if not converged:
            # raise RuntimeError("Internal Newton-Raphson scheme is not converged!")
            print(f"Internal Newton-Raphson scheme is not converged with error {error}")

        # # TODO: Perfom projection on position level
        # def R(x):
        #     q = x[:self.nq]
        #     mu = x[self.nq:]
        #     g_N = self.model.g_N(tk1, q)
        #     # g_N_qk1 = self.model.g_N_q(tk1, qk1)
        #     g_N_qk1 = self.model.g_N_q(tk1, q)
        #     return np.concatenate([
        #         q - qk1 - g_N_qk1 @ mu,
        #         g_N - prox_R0_np(g_N - self.model.prox_r_N * mu)
        #         # g_N - np.maximum(g_N - self.model.prox_r_N * mu, 1.0e-8)
        #     ])

        # from scipy.optimize import fsolve
        # x0 = np.concatenate((qk1, np.zeros(self.nla_N)))
        # res = fsolve(R, x0, full_output=1)
        # x = res[0]
        # qk1 = x[:self.nq]
        # mu_Nk1 = x[self.nq:]

        return (converged, j, error), tk1, qk1, uk1, P_Nk1, P_Fk1

    def solve(self):
        # lists storing output variables
        q = [self.qk]
        u = [self.uk]
        P_N = [self.P_Nk]
        P_F = [self.P_Fk]

        pbar = tqdm(self.t[:-1])
        for _ in pbar:
            (
                (converged, j, error),
                tk1,
                qk1,
                uk1,
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
            P_N.append(P_Nk1)
            P_F.append(P_Fk1)

            # update local variables for accepted time step
            (
                self.tk,
                self.qk,
                self.uk,
                self.P_Nk,
                self.P_Fk,
            ) = (tk1, qk1, uk1, P_Nk1, P_Fk1)

        return Solution(
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
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
        yk1 = yk + self.f(self.tk, yk, zk1)
        qk1 = yk1[: self.nq]
        uk1 = yk1[self.nq : self.nq + self.nu]

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
            # approx_fprime(zk1, lambda z: self.c(z, update_index_set=False)[0])
            approx_fprime(
                zk1, lambda z: self.c(z, update_index_set=False)[0], method="3-point"
            )
        )

    def f(self, tk, yk, zk1):
        # unpack state
        qk = yk[: self.nq]
        uk = yk[self.nq : self.nq + self.nu]

        # unpack percussions
        P_Nk1 = zk1[: self.nla_N]
        mu_Nk1 = zk1[self.nla_N : 2 * self.nla_N]
        P_Fk1 = zk1[2 * self.nla_N :]

        # evaluate quantities of previous time step
        Mk = self.model.M(tk, qk, scipy_matrix=csr_matrix)
        hk = self.model.h(tk, qk, uk)
        W_Nk = self.model.W_N(tk, qk, scipy_matrix=csr_matrix)
        g_N_qk = self.model.g_N_q(tk, qk, scipy_matrix=csr_matrix)
        W_Fk = self.model.W_F(tk, qk, scipy_matrix=csr_matrix)

        # explicit Euler step
        dt = self.dt
        return np.concatenate(
            (
                self.model.q_dot(tk, qk, uk) * dt + g_N_qk.T @ mu_Nk1,
                spsolve(Mk, hk * dt + W_Nk @ P_Nk1 + W_Fk @ P_Fk1),
            )
        )

    def step(self):
        # from scipy.optimize import fsolve
        # zk1 = np.concatenate((self.P_Nk, self.mu_Nk, self.P_Fk))
        # res = fsolve(lambda z: self.c(z)[0], zk1, full_output=1)

        # j = res[1]["nfev"]
        # converged = res[2]
        # zk1 = res[0]

        # ck1, tk1, qk1, uk1, P_Nk1, mu_Nk1, P_Fk1 = self.c(zk1)
        # error = self.error_function(ck1)

        # return (converged, j, error), tk1, qk1, uk1, P_Nk1, mu_Nk1, P_Fk1

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
