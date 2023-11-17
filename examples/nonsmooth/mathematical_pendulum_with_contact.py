import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math import e1, norm, approx_fprime
from cardillo import System
from cardillo.solver import Moreau, BackwardEuler


bilateral_constrained = True
# bilateral_constrained = False

quadratic_length = True
# quadratic_length = False

with_contact = True
# with_contact = False

Solver1, label1, dt1, kwargs1 = BackwardEuler, "BackwardEuler", 1e-2, {}
Solver2, label2, dt2, kwargs2 = Moreau, "Moreau", 1e-2, {}


class MathematicalPendulumCartesianContact:
    """Mathematical pendulum in Cartesian coordinates and with bilateral
    constraint, see Hairer1996 p. 464 - Example 2. Additionally,
    unilateral with the e_x^I-plane are implemented.

    References
    ----------
    Hairer1996: https://link.springer.com/book/10.1007/978-3-642-05221-7
    """

    def __init__(
        self,
        m,
        l,
        grav,
        q0=None,
        u0=None,
        la_g0=None,
        la_N0=None,
        la_F0=None,
        e_N=None,
        e_F=None,
        prox_r_N=1.0e-1,
        prox_r_T=1.0e-1,
    ):
        self.m = m
        self.l = l
        self.grav = grav

        self.nq = 2
        self.nu = 2
        if bilateral_constrained:
            self.nla_g = 1
        else:
            self.nla_g = 0
        if with_contact:
            self.nla_N = 1
            self.nla_F = 0
        else:
            self.nla_N = 0
            self.nla_F = 0
        self.NF_connectivity = [[]]
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.la_N0 = np.zeros(self.nla_N) if la_N0 is None else la_N0
        self.la_F0 = np.zeros(self.nla_F) if la_F0 is None else la_F0
        self.e_N = np.zeros(self.nla_N) if e_N is None else np.array([e_N])
        self.e_F = np.zeros(self.nla_N) if e_F is None else np.array([e_F])
        self.prox_r_N = np.array([prox_r_N])
        self.prox_r_F = np.array([prox_r_T])
        self.mu = np.array([0.0])

        self.g0 = norm(self.q0[:3])

    def q_dot(self, t, q, u):
        return u

    def q_dot_q(self, t, q, u):
        pass

    def q_dot_u(self, t, q):
        return np.eye(self.nq)

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def M(self, t, q):
        return np.eye(self.nu) * self.m

    def h(self, t, q, u):
        f_g = np.array([0, -self.m * self.grav], dtype=np.common_type(q, u))
        r_OS = q[:3]
        g = norm(r_OS)
        n = r_OS / g
        k = 1.0e2
        # k = 1.0e3
        # k = 1.0e4
        # k = 1.0e5
        # k = 1.0e6
        la = -k * (g - self.g0)
        f_spring = n * la
        return f_g + f_spring

    def h_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.h(t, q, u))

    def h_u(self, t, q, u):
        return approx_fprime(u, lambda u: self.h(t, q, u))

    if bilateral_constrained:
        if quadratic_length:

            def g(self, t, q):
                x, y = q
                return np.array([x * x + y * y - self.l * self.l], dtype=q.dtype)

            def g_dot(self, t, q, u):
                x, y = q
                u_x, u_y = u
                return np.array([2 * x * u_x + 2 * y * u_y], dtype=np.common_type(q, u))

            def g_dot_q(self, t, q, u):
                x, y = q
                u_x, u_y = u
                return np.array([2 * u_x, 2 * u_y], dtype=np.common_type(q, u))

            def g_dot_u(self, t, q):
                return self.g_q(t, q)

            def g_ddot(self, t, q, u, a):
                x, y = q
                u_x, u_y = u
                a_x, a_y = a
                return np.array(
                    [2 * (u_x * u_x + x * a_x) + 2 * (u_y * u_y + y * a_y)],
                    dtype=np.common_type(q, u, a),
                )

            def g_q(self, t, q):
                x, y = q
                return np.array([[2 * x, 2 * y]], dtype=q.dtype)

            def g_q_T_mu_g(self, t, q, mu_g):
                x, y = q
                return 2 * mu_g[0] * np.eye(2, dtype=np.common_type(q, mu_g))

            def W_g(self, t, q):
                return self.g_q(t, q).T.reshape(self.nu, self.nla_g)

            def Wla_g_q(self, t, q, la_g):
                return approx_fprime(q, lambda q: self.W_g(t, q) @ la_g)

        else:
            # raise NotImplementedError

            def g(self, t, q):
                x, y = q
                l = np.sqrt(x * x + y * y)
                return np.array([l - self.l])

            def g_dot(self, t, q, u):
                x, y = q
                u_x, u_y = u
                l = np.sqrt(x * x + y * y)
                return np.array([x / l * u_x + y / l * u_y])

            def g_ddot(self, t, q, u, u_dot):
                x, y = q
                u_x, u_y = u
                a_x, a_y = u_dot
                l = np.sqrt(x * x + y * y)
                return np.array(
                    [(u @ u) / l + (q @ u_dot) / l + (q @ u) * (q @ u) / (l * l)]
                )

            def g_q(self, t, q):
                x, y = q
                l = np.sqrt(x * x + y * y)
                return np.array([[x / l, y / l]])

            def g_dot_u(self, t, q):
                x, y = q
                l = np.sqrt(x * x + y * y)
                return np.array([[x / l, y / l]])

            def W_g(self, t, q):
                return self.g_dot_u(t, q).T

    # def G(self, t, q):
    #     W = self.W_g_dense(t, q)
    #     M = self.M_dense(t, q)
    #     # G1 = W.T @ np.linalg.inv(M) @ W
    #     G2 = W.T @ np.linalg.solve(M, W)
    #     # error = np.linalg.norm(G1 - G2)
    #     # print(f"error G: {error}")
    #     return G2

    # def la_g(self, t, q, u):
    #     W = self.W_g_dense(t, q)
    #     M = self.M_dense(t, q)
    #     G = np.array([[W.T @ np.linalg.solve(M, W)]])
    #     zeta = self.g_ddot(t, q, u, np.zeros_like(u))
    #     h = self.f_pot(t, q)
    #     eta = zeta + W.T @ np.linalg.solve(M, h)
    #     return np.linalg.solve(G, -eta)

    if with_contact:

        def g_N(self, t, q):
            return e1[:2] @ q

        def g_N_q(self, t, q):
            return e1[np.newaxis, :2]

        def g_N_dot(self, t, q, u):
            return e1[:2] @ u

        def g_N_ddot(self, t, q, u, a):
            return e1[:2] @ a

        def W_N(self, t, q):
            return e1[:2, np.newaxis]

        def Wla_N_q(self, t, q, la_N):
            return approx_fprime(q, lambda q: self.W_N(t, q) @ la_N)


if __name__ == "__main__":
    # system parameters
    m = 1.25
    l = 1
    g = 10
    e_N = 0.5

    # initial state
    theta0 = np.pi / 2
    q0 = l * np.array([np.sin(theta0), -np.cos(theta0)])
    u0 = np.array([0.0, 0.0])

    # system definition and assemble the system
    pendulum = MathematicalPendulumCartesianContact(m, l, g, q0, u0, e_N=e_N)
    system = System()
    system.add(pendulum)
    system.assemble()

    # end time
    t_end = 5

    sol1 = Solver1(system, t_end, dt1, **kwargs1).solve()
    sol2 = Solver2(system, t_end, dt2, **kwargs2).solve()

    t1 = sol1.t
    q1 = sol1.q
    u1 = sol1.u
    if bilateral_constrained:
        try:
            R_g1 = sol1.P_g
        except:
            R_g1 = dt1 * sol1.la_g
            # P_g1 = sol1.La_g
    if with_contact:
        R_N1 = sol1.P_N

    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u
    if bilateral_constrained:
        try:
            R_g2 = sol2.P_g
        except:
            la_g2 = sol2.la_g
            La_g2 = sol2.La_g
            R_g2 = la_g2 * dt2 + La_g2
    if with_contact:
        P_N2 = sol2.P_N

    # visualize results
    fig, ax = plt.subplots(2, 2)

    # analytical solution according to Beléndez2007,
    # https://doi.org/10.1590/S1806-11172007000400024
    from scipy.special import ellipj, ellipk

    omega0 = np.sqrt(g / l)
    omega = np.pi * omega0 / (2 * ellipk(np.sin(theta0 / 2) ** 2))
    theta = 2 * np.arcsin(
        np.sin(theta0 / 2)
        * ellipj(
            ellipk(np.sin(theta0 / 2) ** 2) - omega0 * t1, np.sin(theta0 / 2) ** 2
        )[0]
    )
    x_ref = -l * np.cos(theta)
    y_ref = l * np.sin(theta)
    # TODO: What is wrong here?
    # u_x_ref = l * np.sin(theta) * omega
    # u_y_ref = l * np.cos(theta) * omega

    # generalized coordinates
    ax[0, 0].plot(t1, x_ref, "-xk", label="x - ref")
    ax[0, 0].plot(t1, y_ref, "-ok", label="y - ref")
    ax[0, 0].plot(t1, q1[:, 0], "--xb", label="x - " + label1)
    ax[0, 0].plot(t1, q1[:, 1], "-.ob", label="y - " + label1)
    ax[0, 0].plot(t2, q2[:, 0], "--xr", label="x - " + label2)
    ax[0, 0].plot(t2, q2[:, 1], "-.or", label="y - " + label2)
    ax[0, 0].grid()
    ax[0, 0].legend()

    # generalized velocities
    # ax[0, 1].plot(t1, u_x_ref, "-xk", label="x_dot - ref")
    # ax[0, 1].plot(t1, u_y_ref, "-ok", label="y_dot - ref")
    ax[0, 1].plot(t1, u1[:, 0], "--xb", label="x_dot - " + label1)
    ax[0, 1].plot(t1, u1[:, 1], "-.ob", label="y_dot - " + label1)
    ax[0, 1].plot(t2, u2[:, 0], "--xr", label="x_dot - " + label2)
    ax[0, 1].plot(t2, u2[:, 1], "-.or", label="y_dot - " + label2)
    ax[0, 1].grid()
    ax[0, 1].legend()

    # bilateral constraints
    try:
        ax[1, 0].plot(t1, R_g1[:, 0], "-xb", label="P_g - " + label1)
        ax[1, 0].plot(t2, R_g2[:, 0], "-xr", label="P_g - " + label2)
        ax[1, 0].grid()
        ax[1, 0].legend()
    except:
        pass

    # normal percussions
    if with_contact:
        ax[1, 1].plot(t1, R_N1[:, 0], "-xb", label="P_N - " + label1)
        ax[1, 1].plot(t2, P_N2[:, 0], "-xr", label="P_N - " + label2)
        ax[1, 1].grid()
        ax[1, 1].legend()

    # plt.show()

    #########
    # animate
    #########
    t = t1
    q = q1

    lim = 1.5 * l

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # prepare data for animation
    frames = len(t)
    target_frames = min(len(t), 200)
    frac = int(frames / target_frames)
    animation_time = 1
    interval = animation_time * 1000 / target_frames

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    # vertical plane
    ax.plot([0, 0], [-lim, lim], "-k")

    # point mass
    (com,) = ax.plot([], [], "-ok")

    def update(t, q, com):
        x, y = q
        com.set_data([0, x], [0, y])

        return com

    def animate(i):
        update(t[i], q[i], com)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()
