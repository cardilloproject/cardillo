import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, eye, bmat
from tqdm import tqdm

from cardillo.math import approx_fprime, fsolve
from cardillo.solver import Solution, consistent_initial_conditions


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

        self.g_S_q = self.system.g_S_q(t, q, scipy_matrix=csc_matrix)
        R[:nq] = q_dot - self.system.q_dot(t, q, u) - self.g_S_q.T @ mu_S
        if self.method == "index 2 GGL":
            self.g_q = self.system.g_q(t, q, scipy_matrix=csc_matrix)
            R[:nq] -= self.g_q.T @ mu_g

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

    def _J(self, y):
        t = self.t
        dt = self.dt
        q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = self._unpack(y)
        q, u = self._update(y)

        A = (
            eye(self.nq, format="coo")
            - dt * self.system.q_dot_q(t, q, u)
            - dt * self.system.g_S_q_T_mu_q(t, q, mu_S)
        )
        B = self.system.B(t, q)
        C = (
            self.system.Mu_q(t, q, u_dot)
            - self.system.h_q(t, q, u)
            - self.system.Wla_g_q(t, q, la_g)
            - self.system.Wla_gamma_q(t, q, la_gamma)
        )
        D = self.M - dt * self.system.h_u(t, q, u)

        gamma_q = self.system.gamma_q(t, q, u)
        g_S_q = self.g_S_q

        # fmt: off
        if self.method == "index 2 GGL":
            g_q = self.g_q
            g_dot_q = self.system.g_dot_q(t, q, u)
            A -= dt * self.system.g_q_T_mu_q(t, q, mu_g)
            J = bmat([
                [           A,             -dt * B,      None,          None, -g_S_q.T, -g_q.T],
                [      dt * C,                   D, -self.W_g, -self.W_gamma,     None,   None],
                [dt * g_dot_q,     dt * self.W_g.T,      None,          None,     None,   None],
                [dt * gamma_q, dt * self.W_gamma.T,      None,          None,     None,   None],
                [  dt * g_S_q,                None,      None,          None,     None,   None],
                [    dt * g_q,                None,      None,          None,     None,   None],
            ], format="csc")
        elif self.method == "index 3":
            g_q = self.system.g_q(t, q)
            J = bmat([
                [           A,             -dt * B,      None,          None, -g_S_q.T],
                [      dt * C,                   D, -self.W_g, -self.W_gamma,     None],
                [    dt * g_q,                None,      None,          None,     None],
                [dt * gamma_q, dt * self.W_gamma.T,      None,          None,     None],
                [  dt * g_S_q,                None,      None,          None,     None],
            ], format="csc")
        elif self.method == "index 2":
            g_dot_q = self.system.g_dot_q(t, q, u)
            J = bmat([
                [           A,             -dt * B,      None,          None, -g_S_q.T],
                [      dt * C,                   D, -self.W_g, -self.W_gamma,     None],
                [dt * g_dot_q,     dt * self.W_g.T,      None,          None,     None],
                [dt * gamma_q, dt * self.W_gamma.T,      None,          None,     None],
                [  dt * g_S_q,                None,      None,          None,     None],
            ], format="csc")
        elif self.method == "index 1":
            g_ddot_q = self.system.g_ddot_q(t, q, u, u_dot)
            g_ddot_u = self.system.g_ddot_u(t, q, u, u_dot)
            gamma_dot_q = self.system.gamma_dot_q(t, q, u, u_dot)
            gamma_dot_u = self.system.gamma_dot_u(t, q, u, u_dot)
            J = bmat([
                [               A,                           -dt * B,      None,          None, -g_S_q.T],
                [          dt * C,                                 D, -self.W_g, -self.W_gamma,     None],
                [   dt * g_ddot_q,        self.W_g.T + dt * g_ddot_u,      None,          None,     None],
                [dt * gamma_dot_q, self.W_gamma.T + dt * gamma_dot_u,      None,          None,     None],
                [      dt * g_S_q,                              None,      None,          None,     None],
            ], format="csc")
        else:
            raise NotImplementedError
        # fmt: on

        return J

        # J_num = csc_matrix(approx_fprime(y, self._R, method="2-point"))
        # J_num = csc_matrix(approx_fprime(y, self._R, method="3-point"))
        J_num = csc_matrix(approx_fprime(y, self._R, method="cs", eps=1.0e-12))
        diff = (J - J_num).toarray()
        # diff = diff[:self.nq]
        # diff = diff[self.nq : ]
        error = np.linalg.norm(diff)
        print(f"error Jacobian: {error}")
        return J_num

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
            sol = fsolve(self._R, self.y, jac=self._J)
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
