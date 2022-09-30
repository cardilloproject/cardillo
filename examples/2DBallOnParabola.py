import numpy as np
from math import pi

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cardillo.math.algebra import norm
from cardillo.math.numerical_derivative import approx_fprime

from cardillo.model import Model

from cardillo.solver import (
    Moreau,
    NonsmoothEulerBackwardsGGL,
    NonsmoothEulerBackwardsGGL_V2,
    NonsmoothNewmarkGGL,
    NonsmoothEulerBackwardsGGL_V3,
    NonsmoothThetaGGL,
    NonsmoothTheta,
    NonsmoothGeneralizedAlpha,
    NonsmoothGenAlphaFirstOrder,
    NonsmoothNewmark,
    Remco,
    RemcoOriginal,
    NonsmoothHalfExplicitEuler,
    NonsmoothHalfExplicitEulerGGL,
)


class BallOnParabola:
    def __init__(
        self,
        m,
        g,
        mu,
        prox_r_N,
        prox_r_F,
        e_N=None,
        e_F=None,
        q0=None,
        u0=None,
    ):
        self.m = m
        self.g = g
        self.nq = 2
        self.nu = 2
        self.nla_N = 1
        self.nla_F = 1
        self.NF_connectivity = [[0]]

        self.a = 0.5
        self.prox_r_N = np.array([prox_r_N])
        self.prox_r_F = np.array([prox_r_F])
        self.mu = np.array([mu])

        self.e_N = np.zeros(self.nla_N) if e_N is None else np.array([e_N])
        self.e_F = np.zeros(self.nla_F) if e_F is None else np.array([e_F])

        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0
        self.la_N0 = np.zeros(self.nla_N)
        self.la_F0 = np.zeros(self.nla_F)

    def parabola(self, s):
        return np.array([s, -self.a * s**2], dtype=float)

    def parabola_s(self, s):
        return np.array([1.0, -2.0 * self.a * s], dtype=float)

    def r_OS(self, q):
        return np.array([*q, 0.0], dtype=float)

    def M(self, t, q, coo):
        coo.extend(m * np.eye(self.nu, dtype=float), (self.uDOF, self.uDOF))

    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q, coo):
        coo.extend_diag(np.ones(self.nq), (self.qDOF, self.uDOF))

    def f_pot(self, t, q):
        return np.array([0.0, -self.m * self.g])

    def f_pot_q(self, t, q, coo):
        coo.extend(np.zeros((self.nu, self.nq)), (self.uDOF, self.qDOF))

    #################
    # normal contacts
    #################
    def tangent(self, s):
        f_prime = self.parabola_s(s)
        return f_prime / norm(f_prime)

    def normal(self, s):
        t = self.tangent(s)
        return np.array([-t[1], t[0]])
        # f_prime = self.parabola_s(s)
        # f_prime_perp = np.array([-f_prime[1], f_prime[0]])
        # return f_prime_perp / norm(f_prime_perp)

    def s_star(self, q):
        return q[0]

    def g_N(self, t, q):
        s_star = self.s_star(q)
        g_N = self.normal(s_star) @ (q - self.parabola(s_star))
        return g_N

    def g_N_dot(self, t, q, u):
        s_star = self.s_star(q)
        return self.normal(s_star) @ u

    def g_N_ddot(self, t, q, u, u_dot):
        s_star = self.s_star(q)
        return self.normal(s_star) @ u_dot

    def g_N_q(self, t, q, coo):
        dense = approx_fprime(q, lambda q: self.g_N(t, q))
        coo.extend(dense, (self.la_NDOF, self.qDOF))

    # def g_N_dot_q(self, t, q, u, coo):
    #     coo.extend(np.zeros((1, 1)), (self.la_NDOF, self.qDOF))

    # def g_N_dot_u(self, t, q, coo):
    #     coo.extend(np.eye(1, dtype=float), (self.la_NDOF, self.uDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def W_N(self, t, q, coo):
        s_star = self.s_star(q)
        dense = self.normal(s_star).reshape(self.nu, self.nla_N)
        coo.extend(dense, (self.uDOF, self.la_NDOF))

    ##################
    # tangent contacts
    ##################
    def gamma_F(self, t, q, u):
        s_star = self.s_star(q)
        return self.tangent(s_star) @ u

    def xi_F(self, t, q, u_pre, u_post):
        return self.gamma_F(t, q, u_post) + self.e_N * self.g_N_gamma_Fdot(t, q, u_pre)

    def W_F(self, t, q, coo):
        s_star = self.s_star(q)
        dense = self.tangent(s_star).reshape(self.nu, self.nla_F)
        coo.extend(dense, (self.uDOF, self.la_FDOF))


if __name__ == "__main__":
    # system definition
    m = 1
    radius = 0.1
    g = 10.0
    # e_N = 0.5
    e_N = 0.0
    e_F = 0.0
    mu = 0.3
    prox_r_N = 0.5
    prox_r_F = 0.5

    # initial conditions
    x0 = 1
    y0 = 1
    x_dot0 = 0.1
    y_dot0 = 0
    q0 = np.array([x0, y0])
    u0 = np.array([x_dot0, y_dot0])

    # create bouncing ball subsystem
    ball = BallOnParabola(m, g, mu, prox_r_N, prox_r_F, e_N=e_N, e_F=e_F, q0=q0, u0=u0)

    # assemble model
    model = Model()
    model.add(ball)
    model.assemble()

    # solver setup
    t0 = 0
    t1 = 2
    # t1 = 20
    # dt = 1e-1
    # dt = 5e-2
    # dt = 1e-2
    # dt = 5e-3
    dt = 1e-3
    # dt = 5e-4

    # solve problem
    # solver_other = NonsmoothGeneralizedAlpha(model, t1, dt)
    # solver_other = NonsmoothEulerBackwardsGGL_V2(model, t1, dt)
    # solver_other = NonsmoothHalfExplicitEuler(model, t1, dt)
    solver_other = NonsmoothHalfExplicitEulerGGL(model, t1, dt)
    sol_other = solver_other.solve()
    t = sol_other.t
    q = sol_other.q
    t_other = sol_other.t
    q_other = sol_other.q
    u_other = sol_other.u
    P_N_other = sol_other.P_N
    P_F_other = sol_other.P_F
    if type(solver_other) in [
        NonsmoothThetaGGL,
        NonsmoothEulerBackwardsGGL,
        NonsmoothEulerBackwardsGGL_V2,
        NonsmoothNewmarkGGL,
        NonsmoothHalfExplicitEuler,
        NonsmoothHalfExplicitEulerGGL,
    ]:
        a_other = np.zeros_like(u_other)
        a_other[1:] = (u_other[1:] - u_other[:-1]) / dt
        la_N_other = np.zeros_like(P_N_other)
        La_N_other = np.zeros_like(P_N_other)
        P_N_other = sol_other.P_N
    else:
        a_other = sol_other.a
        la_N_other = sol_other.la_N
        La_N_other = sol_other.La_N

    sol_moreau = Moreau(model, t1, dt).solve()
    t_moreau = sol_moreau.t
    q_moreau = sol_moreau.q
    u_moreau = sol_moreau.u
    a_moreau = np.zeros_like(u_moreau)
    a_moreau[1:] = (u_moreau[1:] - u_moreau[:-1]) / dt
    P_N_moreau = sol_moreau.P_N
    P_F_moreau = sol_moreau.P_F

    ###########
    # animation
    ###########
    lim = 5
    s = np.linspace(-lim, lim, num=100)
    parabola = np.array([ball.parabola(si) for si in s])

    t = t_moreau
    q = q_moreau

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

    # parabola
    ax.plot(*parabola.T, "-k")

    # point mass
    (com,) = ax.plot([], [], "-ok")

    def update(t, q, com):
        x, y = q
        com.set_data([x], [y])

        return com

    def animate(i):
        update(t[i], q[i], com)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    # plt.show()

    # exit()

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(4, 1)

    # ax[0].plot(s, parabola, "-k", label="ground")
    ax[0].plot(t_moreau, q_moreau[:, 0], "-xr", label="x - Moreau")
    ax[0].plot(t_moreau, q_moreau[:, 1], "--or", label="y - Moreau")
    ax[0].plot(t_other, q_other[:, 0], "-xb", label="x - Other")
    ax[0].plot(t_other, q_other[:, 1], "--ob", label="y - Other")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t_moreau, u_moreau[:, 0], "-xr", label="u_x - Moreau")
    ax[1].plot(t_moreau, u_moreau[:, 1], "--or", label="u_y - Moreau")
    ax[1].plot(t_other, u_other[:, 0], "-xb", label="u_x - Other")
    ax[1].plot(t_other, u_other[:, 1], "-ob", label="u_y - Other")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(t_moreau, P_N_moreau[:, 0], "-r", label="P_N - Moreau")
    ax[2].plot(t_other, P_N_other[:, 0], "--b", label="P_N - Other")
    ax[2].grid()
    ax[2].legend()

    ax[3].plot(t_moreau, P_F_moreau[:, 0], "-r", label="P_F - Moreau")
    ax[3].plot(t_other, P_F_other[:, 0], "--b", label="P_F - Other")
    ax[3].grid()
    ax[3].legend()

    plt.show()

    exit()

    # animation
    # t = t_moreau
    # q = q_moreau
    t = t_other
    q = q_other

    if True:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.set_xlim(-2 * y0, 2 * y0)
        ax.set_ylim(-2 * y0, 2 * y0)

        # prepare data for animation
        frames = len(t)
        target_frames = min(len(t), 200)
        frac = int(frames / target_frames)
        animation_time = 5
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t[::frac]
        q = q[::frac]

        # horizontal plane
        ax.plot([-2 * y0, 2 * y0], [0, 0], "-k")

        def create(t, q):
            (COM,) = ax.plot([], [], "ok")
            (bdry,) = ax.plot([], [], "-k")
            return COM, bdry

        COM, bdry = create(0, q[0])

        def update(t, q, COM, bdry):
            x_S, y_S, _ = ball.r_OS(q)
            x_bdry, y_bdry, _ = ball.boundary(q)

            COM.set_data([x_S], [y_S])
            bdry.set_data(x_bdry, y_bdry)

            return COM, bdry

        def animate(i):
            update(t[i], q[i], COM, bdry)

        anim = animation.FuncAnimation(
            fig, animate, frames=frames, interval=interval, blit=False
        )

    plt.show()
