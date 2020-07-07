import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat
from scipy.integrate import solve_ivp

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
        self.t_span = np.arange(model.t0, t1, dt)

    def eqm(self, t, x):
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

        A = bmat([[M,           -W_g,   -W_gamma], \
                  [g_dot_u,       None,   None], \
                  [gamma_u,   None,   None]]).tocsc()

        ula = spsolve(A, np.concatenate([h, -zeta_g, -zeta_gamma]))

        dx = np.zeros(self.nx)
        dx[:self.nq] = self.model.q_dot(t, q, u)
        dx[self.nq:] = ula[:self.nu]

        return dx

    def solve(self):
        sol = solve_ivp(self.eqm, (self.t_span[0], self.t_span[-1]), self.x0,  \
                        t_eval=self.t_span, method=self.method, rtol=self.rtol, atol=self.atol)

        return sol.t, sol.y[:self.nq, :].T, sol.y[self.nq:, :].T