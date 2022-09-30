import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.model import Model
from cardillo.solver import (
    Moreau,
    NonsmoothThetaGGL,
    NonsmoothEulerBackwardsGGL_V2,
    NonsmoothEulerBackwardsGGL_V3,
    NonsmoothNewmarkGGL,
    NonsmoothTheta,
    NonsmoothGeneralizedAlpha,
    NonsmoothGenAlphaFirstOrder,
    NonsmoothNewmark,
    NonsmoothHalfExplicitEuler,
    NonsmoothHalfExplicitEulerGGL,
)

from cardillo.math.algebra import e1, norm


class MathematicalPendulumCartesianContact:
    """Mathematical pendulum in Cartesian coordinates and with bilateral
    constraint, see Hairer1996 p. 464 - Example 2. Additionally,
    unilateral with the e_x^I-plane re implemented.

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
        self.nla_g = 1
        self.nla_N = 1
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

    def q_dot_q(self, t, q, u, coo):
        pass

    def B(self, t, q, coo):
        coo.extend(np.eye(self.nq, self.nu), (self.qDOF, self.uDOF))

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def M_dense(self, t, q):
        return np.eye(self.nu, self.nu) * self.m

    def M(self, t, q, coo):
        coo.extend(self.M_dense(t, q), (self.uDOF, self.uDOF))

    def f_pot(self, t, q):
        f_g = np.array([0, -self.m * self.grav])
        r_OS = q[:3]
        g = norm(r_OS)
        n = r_OS / g
        k = 1.0e3
        la = -k * (g - self.g0)
        f_spring = n * la
        return f_g + f_spring

    def f_pot_q(self, t, q, coo):
        raise NotImplementedError

    def g(self, t, q):
        x, y = q
        return np.array([x * x + y * y - self.l * self.l])

    def g_dot(self, t, q, u):
        x, y = q
        u_x, u_y = u
        return np.array([2 * x * u_x + 2 * y * u_y])

    def g_dot_u(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def g_ddot(self, t, q, u, a):
        x, y = q
        u_x, u_y = u
        a_x, a_y = a
        return np.array([2 * (u_x * u_x + x * a_x) + 2 * (u_y * u_y + y * a_y)])

    def g_q_dense(self, t, q):
        x, y = q
        return np.array([2 * x, 2 * y])

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g_dense(self, t, q):
        return self.g_q_dense(t, q).T

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        coo.extend(np.eye(self.nu, self.nq) * 2 * la_g[0], (self.uDOF, self.qDOF))

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

    def g_N(self, t, q):
        return e1[:2] @ q

    def g_N_q(self, t, q, coo):
        coo.extend(e1[:2], (self.la_NDOF, self.qDOF))

    def g_N_dot(self, t, q, u):
        return e1[:2] @ u

    def g_N_ddot(self, t, q, u, a):
        return e1[:2] @ a

    def W_N_dense(self, t, q):
        return e1[:2, np.newaxis]

    def W_N(self, t, q, coo):
        coo.extend(self.W_N_dense(t, q), (self.uDOF, self.la_NDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)


if __name__ == "__main__":
    # system parameters
    m = 1
    l = 1
    g = 15
    e_N = 0.5

    # initial state
    q0 = np.array([l, 0.0])
    u0 = np.array([0.0, 0.0])

    # system definition and assemble the model
    pendulum = MathematicalPendulumCartesianContact(m, l, g, q0, u0, e_N=e_N)
    model = Model()
    model.add(pendulum)
    model.assemble()

    # end time and numerical dissipation of generalized-alpha solver
    # t_end = 0.1
    # t_end = 1
    t_end = 3
    # dt = 5.0e-2
    dt = 1.0e-2
    # dt = 5.0e-3
    # dt = 1.0e-3
    # dt = 1.0e-4

    # solve with GGL stabilized Moreau scheme
    # sol1 = NonsmoothThetaGGL(model, t_end, dt).solve()
    # sol1 = NonsmoothTheta(model, t_end, dt).solve()
    # sol1 = NonsmoothGenAlphaFirstOrder(model, t_end, dt, rho_inf=0.85).solve()
    # sol1 = NonsmoothGeneralizedAlpha(model, t_end, dt, rho_inf=0.85).solve()
    # sol1 = NonsmoothNewmark(model, t_end, dt).solve()
    # sol1 = NonsmoothEulerBackwardsGGL_V2(model, t_end, dt).solve()
    # sol1 = NonsmoothEulerBackwardsGGL_V3(model, t_end, dt).solve()
    # sol1 = NonsmoothNewmarkGGL(model, t_end, dt).solve()
    # sol1 = NonsmoothEulerBackwardsGGL_V3(model, t_end, dt).solve()
    # sol1 = NonsmoothHalfExplicitEuler(model, t_end, dt).solve()
    sol1 = NonsmoothHalfExplicitEulerGGL(model, t_end, dt).solve()
    t1 = sol1.t
    q1 = sol1.q
    u1 = sol1.u
    P_g1 = sol1.P_g
    P_N1 = sol1.P_N

    # solve with classical Moreau scheme
    # sol2 = Moreau(model, t_end, dt).solve()
    sol2 = NonsmoothGeneralizedAlpha(model, t_end, dt).solve()
    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u
    try:
        P_g2 = sol2.P_g
    except:
        la_g2 = sol2.la_g
        La_g2 = sol2.La_g
        P_g2 = la_g2 * dt + La_g2
    P_N2 = sol2.P_N

    # visualize results
    fig, ax = plt.subplots(2, 2)

    # generalized coordinates
    ax[0, 0].plot(t1, q1[:, 0], "-xr", label="x - Method1")
    ax[0, 0].plot(t1, q1[:, 1], "--or", label="y - Method1")
    ax[0, 0].plot(t2, q2[:, 0], "-xb", label="x - Method2")
    ax[0, 0].plot(t2, q2[:, 1], "--ob", label="y - Method2")
    ax[0, 0].grid()
    ax[0, 0].legend()

    # generalized velocities
    ax[0, 1].plot(t1, u1[:, 0], "-xr", label="x_dot - Method1")
    ax[0, 1].plot(t1, u1[:, 1], "--or", label="y_dot - Method1")
    ax[0, 1].plot(t2, u2[:, 0], "-xb", label="x_dot - Method2")
    ax[0, 1].plot(t2, u2[:, 1], "--ob", label="y_dot - Method2")
    ax[0, 1].grid()
    ax[0, 1].legend()

    # bilateral constraints
    ax[1, 0].plot(t1, P_g1[:, 0], "-xr", label="P_g - Method1")
    ax[1, 0].plot(t2, P_g2[:, 0], "-xb", label="P_g - Method2")
    ax[1, 0].grid()
    ax[1, 0].legend()

    # normal percussions
    ax[1, 1].plot(t1, P_N1[:, 0], "-xr", label="P_N - Method1")
    ax[1, 1].plot(t2, P_N2[:, 0], "-xb", label="P_N - Method2")
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
