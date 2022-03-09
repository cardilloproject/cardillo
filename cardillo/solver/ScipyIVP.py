import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat, csr_matrix, csc_matrix
from scipy.integrate import solve_ivp
from tqdm import tqdm

from cardillo.solver import Solution


class ScipyIVP:
    def __init__(self, model, t1, dt, method="RK45", rtol=1.0e-6, atol=1.0e-10):
        self.model = model
        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.nq = model.nq
        self.nu = model.nu
        self.nx = self.nq + self.nu
        self.nla_g = self.model.nla_g
        self.nla_gamma = self.model.nla_gamma
        self.x0 = np.concatenate([model.q0, model.u0])

        # integration time
        t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.frac = (t1 - t0) / 100
        self.pbar = tqdm(total=100, leave=True)
        self.i = 0

        # check if initial state satisfies bilateral constraints on position and
        # velocity level
        g = model.g(t0, model.q0)
        assert np.allclose(g, np.zeros(len(g)))
        g_dot = model.g_dot(t0, model.q0, model.u0)
        assert np.allclose(g_dot, np.zeros(len(g_dot)))
        gamma = model.gamma(t0, model.q0, model.u0)
        assert np.allclose(gamma, np.zeros(len(gamma)))

    def eqm(self, t, x):
        if int(t // self.frac) == self.i:
            self.pbar.update(1)
            self.pbar.set_description(f"t: {t:0.2e}s < {self.t1:0.2e}s")
            self.i += 1

        q = x[: self.nq]
        u = x[self.nq :]
        q, u = self.model.step_callback(t, q, u)

        M = self.model.M(t, q)
        h = self.model.h(t, q, u)
        W_g = self.model.W_g(t, q)
        W_gamma = self.model.W_gamma(t, q)
        zeta_g = self.model.zeta_g(t, q, u)
        zeta_gamma = self.model.zeta_gamma(t, q, u)

        # TODO: Can be use a sparse ldl decomposition here as done in C++?
        # fmt: off
        A = bmat([[M,       -W_g, -W_gamma], \
                  [W_g.T, None,     None], \
                  [W_gamma.T, None,     None]], format="csc")
        # fmt: on

        ula = spsolve(A, np.concatenate([h, -zeta_g, -zeta_gamma]))

        dx = np.zeros(self.nx)
        dx[: self.nq] = self.model.q_dot(t, q, u)
        dx[self.nq :] = ula[: self.nu]
        return dx

    def la_g_la_gamma(self, t, q, u):
        W_g = self.model.W_g(t, q, scipy_matrix=csc_matrix)
        W_gamma = self.model.W_gamma(t, q, scipy_matrix=csc_matrix)
        zeta_g = self.model.zeta_g(t, q, u)
        zeta_gamma = self.model.zeta_gamma(t, q, u)
        M = self.model.M(t, q, scipy_matrix=csc_matrix)
        h = self.model.h(t, q, u)

        if self.nla_g > 0:
            MW_g = (spsolve(M, W_g)).reshape((self.nu, self.nla_g))
        else:
            MW_g = csc_matrix((self.nu, self.nla_g))
        if self.nla_gamma > 0:
            MW_gamma = (spsolve(M, W_gamma)).reshape((self.nu, self.nla_gamma))
        else:
            MW_gamma = csc_matrix((self.nu, self.nla_gamma))
        Mh = spsolve(M, h)
        
        # fmt: off
        G = bmat([[    W_g.T @ MW_g,     W_g.T @ MW_gamma], \
                  [W_gamma.T @ MW_g, W_gamma.T @ MW_gamma]], format="csc")
        # fmt: on

        mu = np.concatenate((
            zeta_g + W_g.T @ Mh,
            zeta_gamma + W_gamma.T @ Mh,
        ))
        la = spsolve(G, -mu)
        return la[:self.nla_g], la[self.nla_g:]

    def solve(self):
        sol = solve_ivp(
            self.eqm,
            (self.t[0], self.t[-1]),
            self.x0,
            t_eval=self.t,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            dense_output=True,
        )

        # compute Lagrange multipliers a posteriori at given t's
        t = sol.t
        nt = len(t)
        q = sol.y[: self.nq, :].T
        u = sol.y[self.nq :, :].T
        la_g = np.zeros((nt, self.nla_g))
        la_gamma = np.zeros((nt, self.nla_gamma))
        for i, (ti, qi, ui) in enumerate(zip(t, q, u)):
            la_g[i], la_gamma[i] = self.la_g_la_gamma(ti, qi, ui)

        return Solution(t=t, q=q, u=u, la_g=la_g, la_gamma=la_gamma)
