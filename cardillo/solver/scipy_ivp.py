import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat
from scipy.integrate import solve_ivp
from tqdm import tqdm

from cardillo.solver import Solution

class Scipy_ivp(object):
    def __init__(self, model, t1, dt, method='RK45', rtol=1.0e-6, atol=1.0e-10):
        self.model = model
        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.nq = model.nq
        self.nu = model.nu
        self.nx = self.nq + self.nu
        self.x0 = np.concatenate([model.q0, model.u0])

        # integration time
        t0 = model.t0
        self.t1 = t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.frac = (t1 - t0) / 100
        self.pbar = tqdm(total=100, leave=True)
        self.i = 0

    def eqm(self, t, x):
        if int(t // self.frac) == self.i:
            self.pbar.update(1)
            self.pbar.set_description(f't: {t:0.2e}s < {self.t1:0.2e}s')
            self.i += 1

        q = x[:self.nq]
        u = x[self.nq:]

        M = self.model.M(t, q)
        h = self.model.h(t, q, u)
        W_g = self.model.W_g(t, q)
        g_dot_u = self.model.g_dot_u(t, q)
        W_gamma = self.model.W_gamma(t, q)
        gamma_u = self.model.gamma_u(t,q)
        zeta_g = self.model.zeta_g(t, q, u)
        zeta_gamma = self.model.zeta_gamma(t, q, u)

        A = bmat([[M,       -W_g, -W_gamma], \
                  [g_dot_u, None,     None], \
                  [gamma_u, None,     None]]).tocsc()

        ula = spsolve(A, np.concatenate([h, -zeta_g, -zeta_gamma]))

        dx = np.zeros(self.nx)
        dx[:self.nq] = self.model.q_dot(t, q, u)
        dx[self.nq:] = ula[:self.nu]

        return dx

    def solve(self):
        sol = solve_ivp(self.eqm, (self.t[0], self.t[-1]), self.x0,  \
                        t_eval=self.t, method=self.method, rtol=self.rtol, atol=self.atol)
        return Solution(t=sol.t, q=sol.y[:self.nq, :].T, u=sol.y[self.nq:, :].T)