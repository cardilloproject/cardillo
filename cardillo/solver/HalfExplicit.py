import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, csc_matrix, bmat
from tqdm import tqdm

from cardillo.math import approx_fprime
from cardillo.solver import Solution


class HalfExplicitEulerFixedPoint:
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

        if hasattr(model, "step_callback"):
            self.step_callback = model.step_callback
        else:
            self.step_callback = self.__step_callback

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

    def __step_callback(self, q, u):
        return q, u

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

        c_kappak1 = bmat([[g_muk1, None], [g_dot_muk1, g_dot_la_gk1]], format="csr")

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

            # # fixed-point iteration
            # self.atol = 1.0e-4
            # r = 4.0e-1
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

            qk1, uk1 = self.step_callback(tk1, qk1, uk1)

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
