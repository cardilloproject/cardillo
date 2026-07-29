import numpy as np
from scipy.sparse import eye_array, lil_array
from scipy_dae.integrate import solve_dae
from tqdm import tqdm

from cardillo.solver import Solution, SolverSummary
from cardillo.utility.coo_matrix import CooMatrix


# TODO:
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

        # data allocation
        self.F = np.zeros(self.ny, dtype=float)
        self.g_q1 = self._W_tau = self.W_g1 = self.W_gamma1 = self.W_c1 = None
        self.q_dot_q = self.q_dot_u = None

        self.Mu_q = self.h_q = self.h_u = self.Wla_tau_q = self.Wla_tau_u = (
            self.Wla_g_q
        ) = self.Wla_gamma_q = self.Wla_c_q = None
        self.g_dot_q = self.g_dot_u = self.gamma_q = self.gamma_u = self.c_q = (
            self.c_u
        ) = None
        self.M1 = self.M2 = self.g_q2 = self.W_g2 = self.W_gamma2 = self.W_c2 = None

        self.Jy = CooMatrix((self.ny, self.ny))
        self.Jyp = CooMatrix((self.ny, self.ny))
        eye_q = eye_array(self.nq)
        c_la_c = self.system.c_la_c()
        self.Jyp["eye_q", : self.split[0], : self.split[0]] = eye_q
        self.Jyp["c_la_c", self.split[4] :, self.split[4] :] = c_la_c

    def event(self, t, y, yp):
        q, u = np.array_split(y, self.split)[:2]
        q, u = self.system.step_callback(t, q, u)
        return 1

    def fun(self, t, y, yp):
        # update progress bar
        i1 = int(t // self.frac)
        self.pbar.update(i1 - self.i)
        self.pbar.set_description(f"t: {t:0.2e}s < {self.t1:0.2e}s", refresh=False)
        self.i = i1

        # unpack vectors
        s1, s2, s3, s4, s5 = self.split
        q, u = y[:s1], y[s1:s2]
        q_dot, u_dot, mu_g, la_g, la_gamma, la_c = (
            yp[:s1],
            yp[s1:s2],
            yp[s2:s3],
            yp[s3:s4],
            yp[s4:s5],
            yp[s5:],
        )

        # residual
        F = self.F

        ####################
        # kinematic equation
        ####################
        self.g_q1 = self.system.g_q(t, q, format="Coo", coo=self.g_q1)
        F[: self.split[0]] = (
            q_dot - self.system.q_dot(t, q, u) - mu_g @ self.g_q1.asformat("coo")
        )
        ####################
        # equations of motion
        ####################
        M = self.M2 = self.system.M(t, q, format="Coo", coo=self.M2)
        W_tau = self._W_tau = self.system.W_tau(t, q, format="Coo", coo=self._W_tau)
        W_g = self.W_g1 = self.system.W_g(t, q, format="Coo", coo=self.W_g1)
        W_gamma = self.W_gamma1 = self.system.W_gamma(
            t, q, format="Coo", coo=self.W_gamma1
        )
        W_c = self.W_c1 = self.system.W_c(t, q, format="Coo", coo=self.W_c1)
        F[self.split[0] : self.split[1]] = (
            M.asformat("coo") @ u_dot
            - self.system.h(t, q, u)
            - W_tau.asformat("coo") @ self.system.la_tau(t, q, u)
            - W_g.asformat("coo") @ la_g
            - W_gamma.asformat("coo") @ la_gamma
            - W_c.asformat("coo") @ la_c
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
        s0, s1, s2, s3, s4 = self.split
        q, u = y[:s0], y[s0:s1]
        u_dot = yp[s0:s1]
        la_g = yp[s2:s3]
        la_gamma = yp[s3:s4]
        la_c = yp[s4:]

        # evaluate commonly used quantities
        q_dot_q = self.q_dot_q = self.system.q_dot_q(
            t, q, u, format="Coo", coo=self.q_dot_q
        )
        q_dot_u = self.q_dot_u = self.system.q_dot_u(
            t, q, format="Coo", coo=self.q_dot_u
        )

        Mu_q = self.Mu_q = self.system.Mu_q(t, q, u_dot, format="Coo", coo=self.Mu_q)
        h_q = self.h_q = self.system.h_q(t, q, u, format="Coo", coo=self.h_q)
        h_u = self.h_u = self.system.h_u(t, q, u, format="Coo", coo=self.h_u)
        Wla_tau_q = self.Wla_tau_q = self.system.Wla_tau_q(
            t, q, u, format="Coo", coo=self.Wla_tau_q
        )
        Wla_tau_u = self.Wla_tau_u = self.system.Wla_tau_u(
            t, q, u, format="Coo", coo=self.Wla_tau_u
        )
        Wla_g_q = self.Wla_g_q = self.system.Wla_g_q(
            t, q, la_g, format="Coo", coo=self.Wla_g_q
        )
        Wla_gamma_q = self.Wla_gamma_q = self.system.Wla_gamma_q(
            t, q, la_gamma, format="Coo", coo=self.Wla_gamma_q
        )
        Wla_c_q = self.Wla_c_q = self.system.Wla_c_q(
            t, q, la_c, format="Coo", coo=self.Wla_c_q
        )

        g_dot_q = self.g_dot_q = self.system.g_dot_q(
            t, q, u, format="Coo", coo=self.g_dot_q
        )
        g_dot_u = self.g_dot_u = self.system.g_dot_u(
            t, q, format="Coo", coo=self.g_dot_u
        )

        gamma_q = self.gamma_q = self.system.gamma_q(
            t, q, u, format="Coo", coo=self.gamma_q
        )
        gamma_u = self.gamma_u = self.system.gamma_u(
            t, q, format="Coo", coo=self.gamma_u
        )

        c_q = self.c_q = self.system.c_q(t, q, u, la_c, format="Coo", coo=self.c_q)
        c_u = self.c_u = self.system.c_u(t, q, u, la_c, format="Coo", coo=self.c_u)

        M = self.M1 = self.system.M(t, q, format="Coo", coo=self.M1)
        g_q = self.g_q2 = self.system.g_q(t, q, format="Coo", coo=self.g_q2)
        W_g = self.W_g2 = self.system.W_g(t, q, format="Coo", coo=self.W_g2)
        W_gamma = self.W_gamma2 = self.system.W_gamma(
            t, q, format="Coo", coo=self.W_gamma2
        )
        W_c = self.W_c2 = self.system.W_c(t, q, format="Coo", coo=self.W_c2)

        # first Jacobian w.r.t. y
        Jy = self.Jy

        Jy["q_dot_q", :s0, :s0] = -q_dot_q
        Jy["q_dot_u", :s0, s0:s1] = -q_dot_u
        # note: Here we ignore the derivative d((dg/dq)^T mu) / dq since
        # `solve_dae` already performs an inexact Newton method.
        # Jy[:self.split[0], self.split[1]:self.split[2]] = g_q_T_mu_q

        Jy["Mu_q", s0:s1, :s0] = Mu_q
        Jy["h_q", s0:s1, :s0] = -h_q
        Jy["Wla_tau_q", s0:s1, :s0] = -Wla_tau_q
        Jy["Wla_gamma_q", s0:s1, :s0] = -Wla_gamma_q
        Jy["Wla_g_q", s0:s1, :s0] = -Wla_g_q
        Jy["Wla_c_q", s0:s1, :s0] = -Wla_c_q
        Jy["h_u", s0:s1, s0:s1] = -h_u
        Jy["Wla_tau_u", s0:s1, s0:s1] = -Wla_tau_u

        Jy["g_q", s1:s2, :s0] = g_q

        Jy["g_dot_q", s2:s3, :s0] = g_dot_q
        Jy["g_dot_u", s2:s3, s0:s1] = g_dot_u

        Jy["gamma_q", s3:s4, :s0] = gamma_q
        Jy["gamma_u", s3:s4, s0:s1] = gamma_u

        Jy["c_q", s4:, :s0] = c_q
        Jy["c_u", s4:, s0:s1] = c_u

        # second Jacobian w.r.t. yp
        Jyp = self.Jyp

        Jyp["g_q_T", :s0, s1:s2] = -g_q.T

        Jyp["M", s0:s1, s0:s1] = M
        Jyp["W_g", s0:s1, s2:s3] = -W_g
        Jyp["W_gamma", s0:s1, s3:s4] = -W_gamma
        Jyp["W_c", s0:s1, s4:] = -W_c

        return Jy.asformat("coo"), Jyp.asformat("coo")

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
        self.pbar.close()
        # solver_summary.print()

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
