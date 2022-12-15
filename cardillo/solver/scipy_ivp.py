import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat, csc_matrix
from scipy.integrate import solve_ivp
from tqdm import tqdm

from cardillo.solver import Solution


class ScipyIVP:
    def __init__(
        self,
        system,
        t1,
        dt,
        method="RK45",
        rtol=1.0e-8,
        atol=1.0e-10,
        cDOF_q=np.array([], dtype=int),
        b_q=lambda t: np.array([]),
        cDOF_u=np.array([], dtype=int),
        b_u=lambda t: np.array([]),
    ):
        self.system = system
        self.rtol = rtol
        self.atol = atol
        self.method = method

        # handle redundtant coordinates using cDOF/fDOF for q and u
        z0 = system.q0.copy()
        v0 = system.u0.copy()

        self.nz = len(z0)
        self.nv = len(v0)

        nc_q = len(cDOF_q)
        nc_u = len(cDOF_u)

        self.nq = self.nz - nc_q
        self.nu = self.nv - nc_u

        self.cDOF_q = cDOF_q
        self.cDOF_u = cDOF_u
        self.zDOF = np.arange(self.nz)
        self.vDOF = np.arange(self.nv)
        self.fDOF_q = np.setdiff1d(self.zDOF, cDOF_q)
        self.fDOF_u = np.setdiff1d(self.vDOF, cDOF_u)

        q0 = z0[self.fDOF_q]
        u0 = v0[self.fDOF_u]

        if callable(b_q):
            self.b_q = b_q
        else:
            self.b_q = lambda t: b_q
        if callable(b_u):
            self.b_u = b_u
        else:
            self.b_u = lambda t: b_u

        self.x0 = np.concatenate([q0, u0])
        self.nx = len(self.x0)
        self.nla_g = self.system.nla_g
        self.nla_gamma = self.system.nla_gamma

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

        # check if initial state satisfies bilateral constraints on position and
        # velocity level
        g = system.g(t0, system.q0)
        assert np.allclose(g, np.zeros(len(g)))
        g_dot = system.g_dot(t0, system.q0, system.u0)
        assert np.allclose(g_dot, np.zeros(len(g_dot)))
        gamma = system.gamma(t0, system.q0, system.u0)
        assert np.allclose(gamma, np.zeros(len(gamma)))

    def z(self, t, q):
        z = np.zeros(self.nz)
        z[self.fDOF_q] = q
        z[self.cDOF_q] = self.b_q(t)
        return z

    def v(self, t, u):
        v = np.zeros(self.nv)
        v[self.fDOF_u] = u
        v[self.cDOF_u] = self.b_u(t)
        return v

    def eqm(self, t, x):
        # update progress bar
        i1 = int(t // self.frac)
        self.pbar.update(i1 - self.i)
        self.pbar.set_description(f"t: {t:0.2e}s < {self.t1:0.2e}s")
        self.i = i1

        q = x[: self.nq]
        z = self.z(t, q)
        u = x[self.nq :]
        v = self.v(t, u)

        z, v = self.system.step_callback(t, z, v)

        M = self.system.M(t, z)
        h = self.system.h(t, z, v)
        W_g = self.system.W_g(t, z)
        W_gamma = self.system.W_gamma(t, z)
        zeta_g = self.system.zeta_g(t, z, v)
        zeta_gamma = self.system.zeta_gamma(t, z, v)

        # TODO: Can be use a sparse ldl decomposition here as done in C++?
        # fmt: off
        A = bmat([[        M, -W_g, -W_gamma], \
                  [    W_g.T, None,     None], \
                  [W_gamma.T, None,     None]], format="csc")
        # fmt: on

        v_dot_la = spsolve(A, np.concatenate([h, -zeta_g, -zeta_gamma]))

        dx = np.zeros(self.nx)
        dx[: self.nq] = self.system.q_dot(t, z, v)[self.fDOF_q]
        dx[self.nq :] = v_dot_la[: self.nv][self.fDOF_u]
        return dx

    def la_g_la_gamma(self, t, q, u):
        W_g = self.system.W_g(t, q, scipy_matrix=csc_matrix)
        W_gamma = self.system.W_gamma(t, q, scipy_matrix=csc_matrix)
        zeta_g = self.system.zeta_g(t, q, u)
        zeta_gamma = self.system.zeta_gamma(t, q, u)
        M = self.system.M(t, q, scipy_matrix=csc_matrix)
        h = self.system.h(t, q, u)

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

        mu = np.concatenate(
            (
                zeta_g + W_g.T @ Mh,
                zeta_gamma + W_gamma.T @ Mh,
            )
        )
        la = spsolve(G, -mu)
        la_g = la[: self.nla_g]
        la_gamma = la[self.nla_g :]
        u_dot = spsolve(M, h + W_g @ la_g + W_gamma @ la_gamma)
        return u_dot, la_g, la_gamma

    def solve(self):
        sol = solve_ivp(
            self.eqm,
            self.t_eval[[0, -1]],
            self.x0,
            t_eval=self.t_eval,
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

        # reconstruct redundant coordinates
        z = np.array([self.z(t[i], q[i]) for i in range(nt)])
        v = np.array([self.v(t[i], u[i]) for i in range(nt)])

        # u_dot = np.zeros((nt, self.nu))
        # la_g = np.zeros((nt, self.nla_g))
        # la_gamma = np.zeros((nt, self.nla_gamma))
        # for i, (ti, qi, ui) in enumerate(zip(t, q, u)):
        #     u_dot[i], la_g[i], la_gamma[i] = self.la_g_la_gamma(ti, qi, ui)

        # return Solution(t=t, q=q, u=u, u_dot=u_dot, la_g=la_g, la_gamma=la_gamma)
        return Solution(t=t, q=z, u=v)
