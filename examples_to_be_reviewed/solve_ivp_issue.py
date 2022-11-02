import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class MathematicalPendulum:
    """Mathematical pendulum in Cartesian coordinates and with bilateral
    constraint, see Hairer1996 p. 464 - Example 2.

    References
    ----------
    Hairer1996: https://link.springer.com/book/10.1007/978-3-642-05221-7
    """

    def __init__(self, m, l, grav):
        self.m = m
        self.l = l
        self.grav = grav

        self.nq = 2
        self.nu = 2
        self.nla = 1

    def __call__(self, t, z):
        q = z[: self.nq]
        u = z[self.nq : self.nq + self.nu]
        La = z[self.nq + self.nu :]
        x, y = q
        u_x, u_y = u

        M = np.eye(self.nu, self.nu, dtype=float) * self.m
        h = np.array([0, -self.m * self.grav], dtype=float)
        g_q = np.array([[2 * x, 2 * y]], dtype=float)
        zeta = np.array([2 * (u_x * u_x) + 2 * (u_y * u_y)])

        C = np.zeros((1, 1), dtype=float)
        A = np.block([[M, g_q.T], [g_q, C]])
        b = np.concatenate([h, -zeta])
        u_dot_la = np.linalg.solve(A, b)
        u_dot = u_dot_la[: self.nu]
        la = u_dot_la[self.nu :]  # TODO: How can this be stored?

        z_dot = np.zeros(self.nq + self.nu + self.nla, dtype=float)
        z_dot[: self.nq] = u
        z_dot[self.nq : self.nq + self.nu] = u_dot
        z_dot[self.nq + self.nu :] = la
        return z_dot

    def la(self, t, q, u):
        x, y = q
        u_x, u_y = u

        M = np.eye(self.nu, self.nu, dtype=float) * self.m
        h = np.array([0, -self.m * self.grav], dtype=float)
        g_q = np.array([[2 * x, 2 * y]], dtype=float)
        W = g_q.T
        zeta = np.array([2 * (u_x * u_x) + 2 * (u_y * u_y)])

        G = W.T @ np.linalg.solve(M, W)
        eta = zeta + W.T @ np.linalg.solve(M, h)
        return np.linalg.solve(G, -eta)


if __name__ == "__main__":
    # parameters
    m = 1
    l = 1
    grav = 1

    # pendulum model
    pendulum = MathematicalPendulum(m, l, grav)

    # initial state
    q0 = np.array([l, 0], dtype=float)
    u0 = np.zeros(2, dtype=float)
    # z0 = np.concatenate((q0, u0))
    La0 = np.zeros(1, dtype=float)
    z0 = np.concatenate((q0, u0, La0))

    # time integration
    t0 = 0
    t1 = 10
    t_span = (t0, t1)
    t_eval = np.arange(t0, t1, step=1.0e-3)
    rtol = 1.0e-6
    atol = 1.0e-8
    sol = solve_ivp(
        pendulum, t_span, z0, t_eval=t_eval, rtol=rtol, atol=atol, dense_output=True
    )
    t = sol.t
    z = sol.y

    # visualization
    fig, ax = plt.subplots(1, 3)

    ax[0].plot(t, z[0], "-b", label="x")
    ax[0].plot(t, z[1], "-g", label="y")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, z[2], "-b", label="x_dot")
    ax[1].plot(t, z[3], "-g", label="y_dot")
    ax[1].grid()
    ax[1].legend()

    # evaluate Lagrange multiplicator a posteriori
    q = z[:2].T
    u = z[2:4].T
    la = z[4:].T
    # la = np.array([
    #     pendulum.la(ti, qi, ui) for (ti, qi, ui) in zip(t, q, u)
    # ])

    ax[2].plot(t, la, "-b", label="la")
    ax[2].grid()
    ax[2].legend()

    plt.show()
