import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, identity, bmat
from tqdm import tqdm

from cardillo.math import approx_fprime, fsolve
from cardillo.solver import Solution


def consistent_initial_conditions(system, rtol=1.0e-5, atol=1.0e-8):
    t0 = system.t0
    q0 = system.q0
    u0 = system.u0

    q_dot0 = system.q_dot(t0, q0, u0)

    M0 = system.M(t0, q0, scipy_matrix=coo_matrix)
    h0 = system.h(t0, q0, u0)
    W_g0 = system.W_g(t0, q0, scipy_matrix=coo_matrix)
    W_gamma0 = system.W_gamma(t0, q0, scipy_matrix=coo_matrix)
    zeta_g0 = system.zeta_g(t0, q0, u0)
    zeta_gamma0 = system.zeta_gamma(t0, q0, u0)
    # fmt: off
    A = bmat(
        [
            [        M0, -W_g0, -W_gamma0],
            [    W_g0.T,  None,      None],
            [W_gamma0.T,  None,      None],
        ],
        format="csc",
    )
    b = np.concatenate([
        h0, 
        -zeta_g0, 
        -zeta_gamma0
    ])
    # fmt: on

    u_dot_la_g_la_gamma = spsolve(A, b)
    u_dot0 = u_dot_la_g_la_gamma[: system.nu]
    la_g0 = u_dot_la_g_la_gamma[system.nu : system.nu + system.nla_g]
    la_gamma0 = u_dot_la_g_la_gamma[system.nu + system.nla_g :]

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

    return t0, q0, u0, q_dot0, u_dot0, la_g0, la_gamma0


class EulerBackward:
    def __init__(
        self,
        system,
        t1,
        dt,
        atol=1e-6,
        max_iter=10,
        error_function=lambda x: np.max(np.abs(x)),
        method="index 2 GGL",
    ):
        self.system = system
        assert method in ["index 1", "index 2", "index 3", "index 2 GGL"]
        self.method = method
        self.atol = atol
        self.max_iter = max_iter
        self.error_function = error_function

        #######################################################################
        # integration time
        #######################################################################
        t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t_eval = np.arange(t0, self.t1 + self.dt, self.dt)

        #######################################################################
        # dimensions
        #######################################################################
        self.nq = self.system.nq
        self.nu = self.system.nu
        self.nla_g = self.system.nla_g
        self.nla_gamma = self.system.nla_gamma
        self.nla_S = self.system.nla_S
        self.ny = self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_S
        if method == "index 2 GGL":
            self.ny += self.nla_g

        #######################################################################
        # consistent initial conditions
        #######################################################################
        (
            t0,
            self.qn,
            self.un,
            q_dot0,
            u_dot0,
            la_g0,
            la_gamma0,
        ) = consistent_initial_conditions(system)

        self.y = np.zeros(self.ny, dtype=float)
        self.y[: self.nq] = q_dot0
        self.y[self.nq : self.nq + self.nu] = u_dot0
        self.y[self.nq + self.nu : self.nq + self.nu + self.nla_g] = la_g0
        self.y[
            self.nq
            + self.nu
            + self.nla_g : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
        ] = la_gamma0

    def _unpack(self, y):
        q_dot = y[: self.nq]
        u_dot = y[self.nq : self.nq + self.nu]
        la_g = y[self.nq + self.nu : self.nq + self.nu + self.nla_g]
        la_gamma = y[
            self.nq
            + self.nu
            + self.nla_g : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
        ]
        mu_S = y[
            self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
            + self.nla_S
        ]
        mu_g = y[self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_S :]
        return q_dot, u_dot, la_g, la_gamma, mu_S, mu_g

    def _update(self, y):
        q_dot = y[: self.nq]
        u_dot = y[self.nq : self.nq + self.nu]
        q = self.qn + self.dt * q_dot
        u = self.un + self.dt * u_dot
        return q, u

    def _R(self, y):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_S = self.nla_S
        nqu = nq + nu

        t = self.t
        q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = self._unpack(y)
        q, u = self._update(y)

        self.M = self.system.M(t, q, scipy_matrix=csr_matrix)
        self.W_g = self.system.W_g(t, q, scipy_matrix=csr_matrix)
        self.W_gamma = self.system.W_gamma(t, q, scipy_matrix=csr_matrix)
        R = np.zeros(self.ny, dtype=y.dtype)

        R[:nq] = q_dot - self.system.q_dot(t, q, u)
        if self.method == "index 2 GGL":
            self.g_q = self.system.g_q(t, q, scipy_matrix=csc_matrix)
            self.g_S_q = self.system.g_S_q(t, q, scipy_matrix=csc_matrix)
            R[:nq] += self.g_q.T @ mu_g + self.g_S_q.T @ mu_S

        R[nq:nqu] = self.M @ u_dot - (
            self.system.h(t, q, u) + self.W_g @ la_g + self.W_gamma @ la_gamma
        )

        if self.method == "index 1":
            R[nqu + nla_g : nqu + nla_g + nla_gamma] = self.system.gamma_dot(
                t, q, u, u_dot
            )
        else:
            R[nqu + nla_g : nqu + nla_g + nla_gamma] = self.system.gamma(t, q, u)

        if self.method == "index 2 GGL":
            R[nqu : nqu + nla_g] = self.system.g_dot(t, q, u)
            R[nqu + nla_g + nla_gamma + nla_S :] = self.system.g(t, q)
        elif self.method == "index 3":
            R[nqu : nqu + nla_g] = self.system.g(t, q)
        elif self.method == "index 2":
            R[nqu : nqu + nla_g] = self.system.g_dot(t, q, u)
        elif self.method == "index 1":
            R[nqu : nqu + nla_g] = self.system.g_ddot(t, q, u, u_dot)

        R[nqu + nla_g + nla_gamma : nqu + nla_g + nla_gamma + nla_S] = self.system.g_S(
            t, q
        )

        return R

    if False:

        def __R_wrapper(self, tk1, xk1, xk):
            qk1 = xk1[self.qDOF]
            uk1 = xk1[self.uDOF]
            la_gk1 = xk1[self.la_gDOF]
            la_gammak1 = xk1[self.la_gammaDOF]

            qk = xk[self.qDOF]
            uk = xk[self.uDOF]

            return self.__R(qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1)

        def __R_x_num(self, qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
            xk = np.zeros(self.ny)
            xk[self.qDOF] = qk
            xk[self.uDOF] = uk

            xk1 = np.zeros(self.ny)
            xk1[self.qDOF] = qk1
            xk1[self.uDOF] = uk1
            xk1[self.la_gDOF] = la_gk1
            xk1[self.la_gammaDOF] = la_gammak1

            R_x_num = approx_fprime(
                xk1, lambda xk1: self.__R_wrapper(tk1, xk1, xk), method="3-point"
            )

            return csc_matrix(R_x_num)

        def __R_x_analytic(self, qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
            # equations of motion
            Ru_u = self.Mk1 - self.dt * self.system.h_u(tk1, qk1, uk1)
            Ru_q = self.system.Mu_q(tk1, qk1, uk1 - uk) - self.dt * (
                self.system.h_q(tk1, qk1, uk1)
                + self.system.Wla_g_q(tk1, qk1, la_gk1)
                + self.system.Wla_gamma_q(tk1, qk1, la_gammak1)
            )
            Ru_la_g = -self.dt * self.W_gk1
            Ru_la_gamma = -self.dt * self.W_gammak1

            # kinematic equation
            Rq_u = -self.dt * self.system.B(tk1, qk1)
            Rq_q = identity(self.nq) - self.dt * self.system.q_dot_q(tk1, qk1, uk1)

            # constrain equations
            Rla_g_q = self.system.g_q(tk1, qk1)
            Rla_gamma_q = self.system.gamma_q(tk1, qk1, uk1)
            Rla_gamma_u = self.system.gamma_u(tk1, qk1)

            return bmat(
                [
                    [Ru_u, Ru_q, Ru_la_g, Ru_la_gamma],
                    [Rq_u, Rq_q, None, None],
                    [None, Rla_g_q, None, None],
                    [Rla_gamma_u, Rla_gamma_q, None, None],
                ]
            ).tocsc()

        def __R_x_debug(self, qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
            R_x_num = self.__R_x_num(qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1)
            R_x_analytic = self.__R_x_analytic(
                qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1
            )
            diff = R_x_num - R_x_analytic.toarray()

            if self.debug > 1:
                error_uu = np.linalg.norm(diff[self.uDOF[:, None], self.uDOF])
                error_uq = np.linalg.norm(diff[self.uDOF[:, None], self.qDOF])
                error_ula_g = np.linalg.norm(diff[self.uDOF[:, None], self.la_gDOF])
                error_ula_gamma = np.linalg.norm(
                    diff[self.uDOF[:, None], self.la_gammaDOF]
                )

                error_qu = np.linalg.norm(diff[self.qDOF[:, None], self.uDOF])
                error_qq = np.linalg.norm(diff[self.qDOF[:, None], self.qDOF])
                error_qla_g = np.linalg.norm(diff[self.qDOF[:, None], self.la_gDOF])
                error_qla_gamma = np.linalg.norm(
                    diff[self.qDOF[:, None], self.la_gammaDOF]
                )

                error_la_gu = np.linalg.norm(diff[self.la_gDOF[:, None], self.uDOF])
                error_la_gq = np.linalg.norm(diff[self.la_gDOF[:, None], self.qDOF])
                error_la_gla_g = np.linalg.norm(
                    diff[self.la_gDOF[:, None], self.la_gDOF]
                )
                error_lala_gamma = np.linalg.norm(
                    diff[self.la_gDOF[:, None], self.la_gammaDOF]
                )

                error_la_gammau = np.linalg.norm(
                    diff[self.la_gammaDOF[:, None], self.uDOF]
                )
                error_la_gammaq = np.linalg.norm(
                    diff[self.la_gammaDOF[:, None], self.qDOF]
                )
                error_la_gammala_g = np.linalg.norm(
                    diff[self.la_gammaDOF[:, None], self.la_gDOF]
                )
                error_la_gammala_gamma = np.linalg.norm(
                    diff[self.la_gammaDOF[:, None], self.la_gammaDOF]
                )

                print(f"error_uu jacobian: {error_uu:.5e}")
                print(f"error_uq jacobian: {error_uq:.5e}")
                print(f"error_ula_g jacobian: {error_ula_g:.5e}")
                print(f"error_ula_gamma jacobian: {error_ula_gamma:.5e}")

                print(f"error_qu jacobian: {error_qu:.5e}")
                print(f"error_qq jacobian: {error_qq:.5e}")
                print(f"error_qla_g jacobian: {error_qla_g:.5e}")
                print(f"error_qla_gamma jacobian: {error_qla_gamma:.5e}")

                print(f"error_lau jacobian: {error_la_gu:.5e}")
                print(f"error_laq jacobian: {error_la_gq:.5e}")
                print(f"error_la_gla_g jacobian: {error_la_gla_g:.5e}")
                print(f"error_lala_gamma jacobian: {error_lala_gamma:.5e}")

                print(f"error_la_gammau jacobian: {error_la_gammau:.5e}")
                print(f"error_la_gammaq jacobian: {error_la_gammaq:.5e}")
                print(f"error_la_gammala_g jacobian: {error_la_gammala_g:.5e}")
                print(f"error_la_gammala_gamma jacobian: {error_la_gammala_gamma:.5e}")

            print(f"\ntotal error jacobian: {np.linalg.norm(diff)/ self.ny:.5e}")

            if self.numerical_jacobian:
                return R_x_num
            else:
                return R_x_analytic

        def step(self, tk, qk, uk, la_gk, la_gammak):
            dt = self.dt
            tk1 = tk + dt

            # foward Euler predictor
            la_gk1 = la_gk
            la_gammak1 = la_gammak
            uk1 = uk + dt * spsolve(
                self.Mk1.tocsc(),
                self.system.h(tk, qk, uk)
                + self.W_gk1 @ la_gk
                + self.W_gammak1 @ la_gammak,
            )
            qk1 = qk + dt * self.system.q_dot(tk, qk, uk1)

            # initial guess for Newton-Raphson solver
            xk1 = np.zeros(self.ny)
            xk1[self.qDOF] = qk1
            xk1[self.uDOF] = uk1
            xk1[self.la_gDOF] = la_gk1
            xk1[self.la_gammaDOF] = la_gammak1

            # initial residual and error
            self.system.pre_iteration_update(tk1, qk1, uk1)
            R = self.__R(qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1)
            error = self.error_function(R)
            converged = error < self.atol
            j = 0
            if not converged:
                while j < self.max_iter:
                    # jacobian
                    R_x = self.__R_x(qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1)

                    # Newton update
                    j += 1
                    dx = spsolve(R_x, R)
                    xk1 -= dx
                    qk1 = xk1[self.qDOF]
                    uk1 = xk1[self.uDOF]
                    la_gk1 = xk1[self.la_gDOF]
                    la_gammak1 = xk1[self.la_gammaDOF]

                    self.system.pre_iteration_update(tk1, qk1, uk1)
                    R = self.__R(qk, uk, tk1, qk1, uk1, la_gk1, la_gammak1)
                    error = self.error_function(R)
                    converged = error < self.atol
                    if converged:
                        break

                if not converged:
                    raise RuntimeError(
                        f"internal Newton-Raphson method not converged after {j} steps with error: {error:.5e}"
                    )

            return (converged, j, error), tk1, qk1, uk1, la_gk1, la_gammak1

    def solve(self):
        q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = self._unpack(self.y)

        # lists storing output variables
        q_list = [self.qn]
        u_list = [self.un]
        q_dot_list = [q_dot]
        u_dot_list = [u_dot]
        la_g_list = [la_g]
        la_gamma_list = [la_gamma]
        mu_S_list = [mu_S]
        mu_g_list = [mu_g]

        pbar = tqdm(self.t_eval[:-1])
        for t in pbar:
            self.t = t
            sol = fsolve(self._R, self.y)
            self.y = sol[0]
            converged = sol[1]
            error = sol[2]
            n_iter = sol[3]
            assert converged

            pbar.set_description(
                f"t: {t:0.2e}; {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )

            q, u = self._update(self.y)
            self.qn, self.un = self.system.step_callback(t, q, u)
            q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = self._unpack(self.y)

            q_list.append(self.qn)
            u_list.append(self.un)
            q_dot_list.append(q_dot)
            u_dot_list.append(u_dot)
            la_g_list.append(la_g)
            la_gamma_list.append(la_gamma)
            mu_S_list.append(mu_S)
            mu_g_list.append(mu_g)

        # write solution
        return Solution(
            t=self.t_eval,
            q=np.array(q_list),
            u=np.array(u_list),
            q_dot=np.array(q_dot_list),
            u_dot=np.array(u_dot_list),
            la_g=np.array(la_g_list),
            la_gamma=np.array(la_gamma_list),
            mu_s=np.array(mu_S_list),
            mu_g=np.array(mu_g_list),
        )
