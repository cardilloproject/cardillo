import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, eye, diags, bmat
from tqdm import tqdm

from cardillo.math import fsolve, approx_fprime
from cardillo.solver import Solution


# Factors taken from https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/radau.py.
# See also Hairer1993 around (4.13).
# NEWTON_MAXITER = 6  # Maximum number of Newton iterations.
NEWTON_MAXITER = 10  # Maximum number of Newton iterations.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.


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
        # method="index 2 GGL",
        method="index 3",
        # method="index 2",
        # method="index 1",
        # debug=True,
        debug=False,
    ):
        self.system = system
        assert method in ["index 1", "index 2", "index 3", "index 2 GGL"]
        self.method = method
        self.rho_inf = rho_inf
        self.atol = atol
        self.max_iter = max_iter
        self.error_function = error_function
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
        self.frac = (t1 - t0) / 101
        self.i = 0
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
            # self.tn = t
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

        # pbar = tqdm(self.t_eval[1:])
        # for _ in pbar:
        pbar = tqdm(total=100, leave=True)
        while self.tn <= self.t1:
            step_accepted = False
            while not step_accepted:
                sol = fsolve(
                    self._R,
                    self.y,
                    # jac="2-point",
                    jac=self._J,
                    error_function=self.error_function,
                    atol=self.atol,
                    max_iter=self.max_iter,
                )
                converged = sol[1]

                # halve step size if fsolve does not converge
                if not converged:
                    print(
                        f"fsolve is not converged at t: {self.tn + self.h} => halve time step"
                    )
                    self.h *= 0.5
                    continue

                self.y = sol[0]
                error = sol[2]
                n_iter = sol[3]
                assert converged

                (
                    t,
                    q,
                    u,
                    q_dot,
                    u_dot,
                    la_g,
                    la_gamma,
                    mu_S,
                    mu_g,
                ) = self._split_and_update(self.y, store=False)
                # self.qn, self.un = self.system.step_callback(t, q, u)

                ########################################
                # error estimate of Rang2013 Section 4.1
                ########################################
                # backward Euler solution
                q_Euler = self.qn + self.h * q_dot
                u_Euler = self.un + self.h * u_dot
                e = np.concatenate((q, u)) - np.concatenate((q_Euler, u_Euler))
                # q = 1
                q = 2

                # scaled tolerance of Hairer1993 (4.10)
                scale = np.ones_like(e) * 1e-5

                # error measure of Hairer1993 (4.11)
                err = np.sqrt(sum(e * e / scale) / len(e))

                # safety factor depending on required Newton iterations,
                # see https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/radau.py#L475L476
                safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

                # step size selection of Hairer1993 (4.13)
                fac = min(
                    MAX_FACTOR, max(MIN_FACTOR, safety * (1 / err) ** (1 / (q + 1)))
                )
                h_new = self.h * fac

                # accept step if err < 1, see Hairer1993 after (4.13)
                if err < 1:
                    step_accepted = True
                    self.tn = t
                else:
                    print(f"step rejected at t: {t}")

                self.h = h_new

            t, q, u, q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = self._split_and_update(
                self.y, store=True
            )

            pbar.set_description(
                f"t: {t:0.2e}; {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )

            # update progress bar
            i1 = int(t // self.frac)
            pbar.update(i1 - self.i)
            pbar.set_description(
                f"t: {t:0.2e}s < {self.t1:0.2e}s; {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )
            self.i = i1

            t_list.append(t)
            q_list.append(q)
            u_list.append(u)
            q_dot_list.append(q_dot)
            u_dot_list.append(u_dot)
            la_g_list.append(la_g)
            la_gamma_list.append(la_gamma)
            mu_S_list.append(mu_S)
            mu_g_list.append(mu_g)

            # # update step size
            # min_factor = 0.2
            # max_factor = 5
            # target_iter = 4.2
            # factor = target_iter / n_iter
            # factor = max(min_factor, min(max_factor, factor))
            # print(f"factor: {factor}")
            # self.h *= factor

            # self.tn += self.h

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
