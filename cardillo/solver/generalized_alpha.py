import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, eye, bmat
from scipy.sparse.linalg import splu
from tqdm import tqdm

from cardillo.math import approx_fprime, fsolve
from cardillo.solver import Solution


class GeneralizedAlphaFirstOrder:
    def __init__(
        self,
        system,
        t1,
        h,
        rho_inf=0.8,
        atol=1e-8,
        max_iter=10,
        error_function=lambda x: np.max(np.abs(x)),
        method="index 3",
        jac_method=None,
        debug=False,
    ):
        self.system = system
        assert method in ["index 1", "index 2", "index 3", "index 2 GGL"]
        self.method = method
        self.rho_inf = rho_inf
        self.atol = atol
        self.max_iter = max_iter
        self.error_function = error_function
        self.jac_method = jac_method
        self.debug = debug

        # generalized alpha constants
        self.rho_inf = rho_inf
        self.alpha_m = 0.5 * (3.0 * rho_inf - 1.0) / (rho_inf + 1.0)
        self.alpha_f = rho_inf / (rho_inf + 1.0)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m

        #######################################################################
        # integration time
        #######################################################################
        self.tn = t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.h = h
        self.t_eval = np.arange(t0, self.t1 + self.h, self.h)

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
        # initial conditions
        #######################################################################
        t0 = system.t0
        self.qn = system.q0
        self.un = system.u0
        self.q_dotn = system.q_dot0
        self.u_dotn = system.u_dot0
        self.la_gn = system.la_g0
        self.la_gamman = system.la_gamma0

        self.vn = self.q_dotn.copy()
        self.an = self.u_dotn.copy()

        self.y = np.zeros(self.ny, dtype=float)
        self.y[: self.nq] = self.q_dotn.copy()
        self.y[self.nq : self.nq + self.nu] = self.u_dotn.copy()
        self.y[self.nq + self.nu : self.nq + self.nu + self.nla_g] = self.la_gn.copy()
        self.y[
            self.nq
            + self.nu
            + self.nla_g : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
        ] = self.la_gamman.copy()

        self.split_y = np.cumsum(
            np.array(
                [self.nq, self.nu, self.nla_g, self.nla_gamma, self.nla_S], dtype=int
            )
        )

    def _split_and_update(self, y, store=False):
        h = self.h
        t = self.tn + h
        gamma = self.gamma
        alpha_f = self.alpha_f
        alpha_m = self.alpha_m

        q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = np.array_split(y, self.split_y)

        # auxiliary derivatives
        v = (alpha_f * self.q_dotn + (1 - alpha_f) * q_dot - alpha_m * self.vn) / (
            1 - alpha_m
        )
        a = (alpha_f * self.u_dotn + (1 - alpha_f) * u_dot - alpha_m * self.an) / (
            1 - alpha_m
        )

        # Newmark update
        q = self.qn + h * ((1 - gamma) * self.vn + gamma * v)
        u = self.un + h * ((1 - gamma) * self.an + gamma * a)

        if store:
            self.tn = t
            self.qn = q.copy()
            self.un = u.copy()
            self.q_dotn = q_dot.copy()
            self.u_dotn = u_dot.copy()
            self.vn = v.copy()
            self.an = a.copy()

        return t, q, u, q_dot, u_dot, la_g, la_gamma, mu_S, mu_g

    def _R(self, y, store=False):
        t, q, u, q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = self._split_and_update(
            y, store
        )

        R = np.zeros(self.ny, dtype=y.dtype)

        # kinematic differential equation
        self.g_S_q = self.system.g_S_q(t, q, scipy_matrix=csc_matrix)
        R[: self.split_y[0]] = q_dot - self.system.q_dot(t, q, u) - self.g_S_q.T @ mu_S
        if self.method == "index 2 GGL":
            self.g_q = self.system.g_q(t, q, scipy_matrix=csc_matrix)
            R[: self.split_y[0]] -= self.g_q.T @ mu_g

        # equations of motion
        self.M = self.system.M(t, q, scipy_matrix=csr_matrix)
        self.W_g = self.system.W_g(t, q, scipy_matrix=csr_matrix)
        self.W_gamma = self.system.W_gamma(t, q, scipy_matrix=csr_matrix)
        R[self.split_y[0] : self.split_y[1]] = self.M @ u_dot - (
            self.system.h(t, q, u) + self.W_g @ la_g + self.W_gamma @ la_gamma
        )

        # bilateral position constraints
        if self.method == "index 2 GGL":
            R[self.split_y[1] : self.split_y[2]] = self.system.g_dot(t, q, u)
            R[self.split_y[4] :] = self.system.g(t, q)
        elif self.method == "index 3":
            R[self.split_y[1] : self.split_y[2]] = self.system.g(t, q)
        elif self.method == "index 2":
            R[self.split_y[1] : self.split_y[2]] = self.system.g_dot(t, q, u)
        elif self.method == "index 1":
            R[self.split_y[1] : self.split_y[2]] = self.system.g_ddot(t, q, u, u_dot)

        # bilateral velocity constraints
        if self.method == "index 1":
            R[self.split_y[2] : self.split_y[3]] = self.system.gamma_dot(t, q, u, u_dot)
        else:
            R[self.split_y[2] : self.split_y[3]] = self.system.gamma(t, q, u)

        # quaternion constraint
        R[self.split_y[3] : self.split_y[4]] = self.system.g_S(t, q)

        return R

    def _J(self, y):
        chain = self.h * self.gamma * (1 - self.alpha_f) / (1 - self.alpha_m)

        t, q, u, q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = self._split_and_update(y)

        # self.g_S_q = self.system.g_S_q(t, q, scipy_matrix=csc_matrix)
        # self.M = self.system.M(t, q, scipy_matrix=csr_matrix)
        # self.W_g = self.system.W_g(t, q, scipy_matrix=csr_matrix)
        # self.W_gamma = self.system.W_gamma(t, q, scipy_matrix=csr_matrix)

        A = (
            eye(self.nq, format="csc")
            - chain * self.system.q_dot_q(t, q, u)
            - chain * self.system.g_S_q_T_mu_q(t, q, mu_S)
        )
        B = self.system.B(t, q)
        C = (
            self.system.Mu_q(t, q, u_dot)
            - self.system.h_q(t, q, u)
            - self.system.Wla_g_q(t, q, la_g)
            - self.system.Wla_gamma_q(t, q, la_gamma)
        )
        D = self.M - chain * self.system.h_u(t, q, u)

        gamma_q = self.system.gamma_q(t, q, u)

        # fmt: off
        if self.method == "index 2 GGL":
            g_q = self.g_q
            g_dot_q = self.system.g_dot_q(t, q, u)
            A -= chain * self.system.g_q_T_mu_q(t, q, mu_g)
            J = bmat([
                [                   A,             -chain * B,      None,          None, -self.g_S_q.T, -g_q.T],
                [           chain * C,                      D, -self.W_g, -self.W_gamma,          None,   None],
                [     chain * g_dot_q,     chain * self.W_g.T,      None,          None,          None,   None],
                [     chain * gamma_q, chain * self.W_gamma.T,      None,          None,          None,   None],
                [  chain * self.g_S_q,                   None,      None,          None,          None,   None],
                [         chain * g_q,                   None,      None,          None,          None,   None],
            ], format="csc")
        elif self.method == "index 3":
            g_q = self.system.g_q(t, q)
            J = bmat([
                [                   A,             -chain * B,      None,          None, -self.g_S_q.T],
                [           chain * C,                      D, -self.W_g, -self.W_gamma,          None],
                [         chain * g_q,                   None,      None,          None,          None],
                [     chain * gamma_q, chain * self.W_gamma.T,      None,          None,          None],
                [  chain * self.g_S_q,                   None,      None,          None,          None],
            ], format="csc")
        elif self.method == "index 2":
            g_dot_q = self.system.g_dot_q(t, q, u)
            J = bmat([
                [                   A,             -chain * B,      None,          None, -self.g_S_q.T],
                [           chain * C,                      D, -self.W_g, -self.W_gamma,          None],
                [     chain * g_dot_q,     chain * self.W_g.T,      None,          None,          None],
                [     chain * gamma_q, chain * self.W_gamma.T,      None,          None,          None],
                [  chain * self.g_S_q,                   None,      None,          None,          None],
            ], format="csc")
        elif self.method == "index 1":
            raise NotImplementedError
            g_ddot_q = self.system.g_ddot_q(t, q, u, u_dot)
            g_ddot_u = self.system.g_ddot_u(t, q, u, u_dot)
            gamma_dot_q = self.system.gamma_dot_q(t, q, u, u_dot)
            gamma_dot_u = self.system.gamma_dot_u(t, q, u, u_dot)
            J = bmat([
                [                  A,                           -h * B,      None,          None, -self.g_S_q.T],
                [              h * C,                                D, -self.W_g, -self.W_gamma,          None],
                [       h * g_ddot_q,        self.W_g.T + h * g_ddot_u,      None,          None,          None],
                [    h * gamma_dot_q, self.W_gamma.T + h * gamma_dot_u,      None,          None,          None],
                [      h * self._S_q,                             None,      None,          None,          None],
            ], format="csc")
        else:
            raise NotImplementedError
        # fmt: on

        if not self.debug:
            return J
        else:
            J_num = csc_matrix(approx_fprime(y, self._R))
            diff = (J - J_num).toarray()
            error = np.linalg.norm(diff)
            print(f"error Jacobian: {error}")
            return J_num

    def solve(self):
        q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = np.array_split(self.y, self.split_y)

        # lists storing output variables
        t_list = [self.tn]
        q_list = [self.qn.copy()]
        u_list = [self.un.copy()]
        q_dot_list = [q_dot]
        u_dot_list = [u_dot]
        la_g_list = [la_g]
        la_gamma_list = [la_gamma]
        mu_S_list = [mu_S]
        mu_g_list = [mu_g]

        pbar = tqdm(self.t_eval[1:])
        for _ in pbar:
            ################
            # fsolve version
            ################
            sol = fsolve(
                self._R,
                self.y,
                jac=self.jac_method if self.jac_method is not None else self._J,
                error_function=self.error_function,
                atol=self.atol,
                max_iter=self.max_iter,
            )
            self.y, converged, error, n_iter = sol[:4]
            assert converged

            # ##############
            # # splu version
            # ##############
            # R = self._R(self.y)
            # error = self.error_function(R)
            # converged = error <= self.atol

            # # Newton loop
            # LU = splu(self._J(self.y))
            # n_iter = 0
            # while (not converged) and (n_iter < self.max_iter):
            #     n_iter += 1
            #     self.y -= LU.solve(R)
            #     R = self._R(self.y)
            #     error = self.error_function(R)
            #     converged = error <= self.atol

            # if not converged:
            #     print(f"fsolve is not converged after {n_iter} iterations with error {error:2.3f} => compute new LU")
            #     continue

            t, q, u, q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = self._split_and_update(
                self.y, store=True
            )

            pbar.set_description(
                f"t: {t:0.2e}; {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )

            t_list.append(t)
            q_list.append(q)
            u_list.append(u)
            q_dot_list.append(q_dot)
            u_dot_list.append(u_dot)
            la_g_list.append(la_g)
            la_gamma_list.append(la_gamma)
            mu_S_list.append(mu_S)
            mu_g_list.append(mu_g)

        # write solution
        return Solution(
            t=np.array(t_list),
            q=np.array(q_list),
            u=np.array(u_list),
            q_dot=np.array(q_dot_list),
            u_dot=np.array(u_dot_list),
            la_g=np.array(la_g_list),
            la_gamma=np.array(la_gamma_list),
            mu_s=np.array(mu_S_list),
            mu_g=np.array(mu_g_list),
        )
