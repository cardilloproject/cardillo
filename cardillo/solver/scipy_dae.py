import numpy as np
from scipy.sparse import eye_array, lil_array
from scipy_dae.integrate import solve_dae
from tqdm import tqdm

from cardillo.solver import Solution, SolverSummary


# TODO:
# - review events if implementation accepts signature `events(t, y, yp)`
# - Add Jacobian of GGl term if convergence problems occur
class ScipyDAE:
    """Wrapper around Radau IIA and BDF methods implementted in `scipy_dae`. 
    A stabilized index 1 formulation is used as proposed by Anantharaman and Hiller.

    References:
    -----------
    scipy_dae: https://github.com/JonasBreuling/scipy_dae \\
    Anantharaman and Hiller.: https://doi.org/10.1002/nme.1620320803
    """

    def __init__(
        self,
        system,
        t1,
        dt,
        method="Radau",
        rtol=1.0e-3,
        atol=1.0e-6,
        **kwargs,
    ):
        self.system = system
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.kwargs = kwargs

        self.nq = system.nq
        self.nu = system.nu
        self.nla_g = self.system.nla_g
        self.nla_gamma = self.system.nla_gamma
        self.nla_c = self.system.nla_c
        self.ny = self.nq + self.nu + 2 * self.nla_g + self.nla_gamma + self.nla_c
        self.split = np.cumsum(
            np.array(
                [
                    self.nq,
                    self.nu,
                    self.nla_g,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_c,
                ],
                dtype=int,
            )
        )[:-1]
        self.y0 = np.concatenate(
            (
                system.q0,
                system.u0,
                0 * system.la_g0,
                0 * system.la_g0,
                0 * system.la_gamma0,
                0 * system.la_c0,
            )
        )
        self.y_dot0 = np.concatenate(
            (
                system.q_dot0,
                system.u_dot0,
                0 * system.la_g0,  # GGL multiplier
                system.la_g0,
                system.la_gamma0,
                system.la_c0,
            )
        )

        # integration time
        t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t_eval = np.arange(t0, self.t1 + self.dt, self.dt)

        self.frac = (t1 - t0) / 101
        self.pbar = tqdm(total=100, leave=True)
        self.i = 0

    def event(self, t, y):
        q, u = np.array_split(y, self.split)[:2]
        q, u = self.system.step_callback(t, q, u)
        return 1

    def fun(self, t, y, yp):
        # update progress bar
        i1 = int(t // self.frac)
        self.pbar.update(i1 - self.i)
        self.pbar.set_description(f"t: {t:0.2e}s < {self.t1:0.2e}s")
        self.i = i1

        # unpack vectors
        q, u, _, _, _, _ = np.array_split(y, self.split)
        q_dot, u_dot, mu_g, la_g, la_gamma, la_c = np.array_split(yp, self.split)

        # residual
        F = np.zeros_like(y, dtype=np.common_type(y, yp))

        ####################
        # kinematic equation
        ####################
        F[: self.split[0]] = (
            q_dot
            - self.system.q_dot(t, q, u)
            - self.system.g_q(t, q, format="csc").T @ mu_g
        )

        ####################
        # equations of motion
        ####################
        F[self.split[0] : self.split[1]] = (
            self.system.M(t, q) @ u_dot
            - self.system.h(t, q, u)
            - self.system.W_tau(t, q, format="csr") @ self.system.la_tau(t, q, u)
            - self.system.W_g(t, q, format="csr") @ la_g
            - self.system.W_gamma(t, q, format="csr") @ la_gamma
            - self.system.W_c(t, q, format="csr") @ la_c
        )

        #######################
        # bilateral constraints
        #######################
        F[self.split[1] : self.split[2]] = self.system.g(t, q)
        F[self.split[2] : self.split[3]] = self.system.g_dot(t, q, u)
        F[self.split[3] : self.split[4]] = self.system.gamma(t, q, u)

        ############
        # compliance
        ############
        F[self.split[4] :] = self.system.c(t, q, u, la_c)

        return F

    def jac(self, t, y, yp):
        # unpack vectors
        q, u, _, _, _, _ = np.array_split(y, self.split)
        q_dot, u_dot, mu_g, la_g, la_gamma, la_c = np.array_split(yp, self.split)

        # evaluate commonly used quantities
        q_dot_q = self.system.q_dot_q(t, q, u)
        q_dot_u = self.system.q_dot_u(t, q)

        Mu_q = self.system.Mu_q(t, q, u_dot)
        h_q = self.system.h_q(t, q, u)
        h_u = self.system.h_u(t, q, u)
        Wla_tau_q = self.system.Wla_tau_q(t, q, u)
        Wla_tau_u = self.system.Wla_tau_u(t, q, u)
        Wla_g_q = self.system.Wla_g_q(t, q, la_g)
        Wla_gamma_q = self.system.Wla_gamma_q(t, q, la_gamma)
        Wla_c_q = self.system.Wla_c_q(t, q, la_c)

        g_dot_q = self.system.g_dot_q(t, q, u)
        g_dot_u = self.system.g_dot_u(t, q)

        gamma_q = self.system.gamma_q(t, q, u)
        gamma_u = self.system.gamma_u(t, q)

        c_q = self.system.c_q(t, q, u, la_c)
        c_u = self.system.c_u(t, q, u, la_c)

        eye_q = eye_array(self.nq)
        M = self.system.M(t, q)
        g_q = self.system.g_q(t, q)
        W_g = self.system.W_g(t, q)
        W_gamma = self.system.W_gamma(t, q)
        W_c = self.system.W_c(t, q)
        c_la_c = self.system.c_la_c()

        # first Jacobian w.r.t. y
        Jy = lil_array((self.ny, self.ny))

        Jy[: self.split[0], : self.split[0]] = -q_dot_q
        Jy[: self.split[0], self.split[0] : self.split[1]] = -q_dot_u
        # note: Here we ignore the derivative d((dg/dq)^T mu) / dq since
        # `solve_dae` does performs an inexact Newton.
        # Jy[:self.split[0], self.split[1]:self.split[2]] = g_q_T_mu_q

        Jy[self.split[0] : self.split[1], : self.split[0]] = (
            Mu_q - h_q - Wla_tau_q - Wla_gamma_q - Wla_g_q - Wla_c_q
        )
        Jy[self.split[0] : self.split[1], self.split[0] : self.split[1]] = (
            -h_u - Wla_tau_u
        )

        Jy[self.split[1] : self.split[2], : self.split[0]] = g_q

        Jy[self.split[2] : self.split[3], : self.split[0]] = g_dot_q
        Jy[self.split[2] : self.split[3], self.split[0] : self.split[1]] = g_dot_u

        Jy[self.split[3] : self.split[4], : self.split[0]] = gamma_q
        Jy[self.split[3] : self.split[4], self.split[0] : self.split[1]] = gamma_u

        Jy[self.split[4] :, : self.split[0]] = c_q
        Jy[self.split[4] :, self.split[0] : self.split[1]] = c_u

        # second Jacobian w.r.t. yp
        Jyp = lil_array((self.ny, self.ny))

        Jyp[: self.split[0], : self.split[0]] = eye_q
        Jyp[: self.split[0], self.split[1] : self.split[2]] = -g_q.T

        Jyp[self.split[0] : self.split[1], self.split[0] : self.split[1]] = M
        Jyp[self.split[0] : self.split[1], self.split[2] : self.split[3]] = -W_g
        Jyp[self.split[0] : self.split[1], self.split[3] : self.split[4]] = -W_gamma
        Jyp[self.split[0] : self.split[1], self.split[4] :] = -W_c

        Jyp[self.split[4] :, self.split[4] :] = c_la_c

        return Jy, Jyp

        # note: Keep this for debugging the Jacobian

        # from scipy.optimize._numdiff import approx_derivative

        # Jy_num = approx_derivative(lambda y: self.fun(t, y, yp), y, method="2-point")
        # diff_Jy = Jy - Jy_num
        # diff_Jy = diff_Jy[self.split[0]:, self.split[0]:] # ignore kinematic equations since GGL Jacobian use not implemented
        # error_Jy = np.linalg.norm(diff_Jy)
        # print(f"error_Jy: {error_Jy}")

        # Jyp_num = approx_derivative(lambda yp: self.fun(t, y, yp), yp, method="2-point")
        # diff_Jyp = Jyp - Jyp_num
        # error_Jyp = np.linalg.norm(diff_Jyp)
        # print(f"error_Jyp: {error_Jyp}")

        # return Jy_num, Jyp_num

    def solve(self):
        solver_summary = SolverSummary(f"Scipy solve_dae with method '{self.method}'")
        sol = solve_dae(
            self.fun,
            self.t_eval[[0, -1]],
            self.y0,
            self.y_dot0,
            t_eval=self.t_eval,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            events=[self.event],
            jac=self.jac,
            **self.kwargs,
        )
        solver_summary.print()

        # unpack solution
        t = sol.t
        q, u, _, _, _, _ = np.array_split(sol.y, self.split)
        q_dot, u_dot, mu_g, la_g, la_gamma, la_c = np.array_split(sol.yp, self.split)

        return Solution(
            system=self.system,
            t=t,
            q=q.T,
            u=u.T,
            q_dot=q_dot.T,
            u_dot=u_dot.T,
            mu_g=mu_g.T,
            la_g=la_g.T,
            la_gamma=la_gamma.T,
            la_c=la_c.T,
            solver_summary=solver_summary,
        )
