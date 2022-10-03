import numpy as np
from math import pi

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    DecoupledNonsmoothHalfExplicitRungeKutta,
    NonsmoothHalfExplicitEuler,
    NonsmoothHalfExplicitEulerGGL,
    NonsmoothDecoupled,
)


class BouncingBall:
    def __init__(
        self,
        m,
        g,
        radius,
        prox_r_N,
        e_N=None,
        q0=None,
        u0=None,
    ):
        self.m = m
        self.g = g
        self.nq = 1
        self.nu = 1
        self.nla_N = 1
        self.nla_F = 0
        self.NF_connectivity = [[]]

        self.radius = radius
        self.prox_r_N = np.array([prox_r_N])
        self.prox_r_F = np.zeros(self.nla_F)
        self.mu = np.zeros(self.nla_F)

        self.e_N = np.zeros(self.nla_N) if e_N is None else np.array([e_N])
        self.e_F = np.zeros(self.nla_F)

        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0
        self.la_N0 = np.zeros(self.nla_N)
        self.la_F0 = np.zeros(self.nla_F)

    def r_OS(self, q):
        return np.array([0.0, *q, 0.0], dtype=float)

    def boundary(self, q, n=100):
        r_OS = self.r_OS(q)
        phi = np.linspace(0, 2.0 * np.pi, n, endpoint=True)
        K_r_SP = self.radius * np.vstack([np.sin(phi), np.cos(phi), np.zeros(n)])
        return np.repeat(r_OS, n).reshape(3, n) + K_r_SP

    def M(self, t, q, coo):
        coo.extend(m * np.eye(1, dtype=float), (self.uDOF, self.uDOF))

    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q, coo):
        coo.extend_diag(np.ones(self.nq), (self.qDOF, self.uDOF))

    def f_pot(self, t, q):
        return np.array([-self.m * self.g])

    def f_pot_q(self, t, q, coo):
        coo.extend(np.zeros((1, 1)), (self.uDOF, self.qDOF))

    def g_N(self, t, q):
        return q - self.radius

    def g_N_dot(self, t, q, u):
        return u

    def g_N_ddot(self, t, q, u, u_dot):
        return u_dot

    def g_N_q(self, t, q, coo):
        coo.extend(np.eye(1, dtype=float), (self.la_NDOF, self.qDOF))

    def g_N_dot_q(self, t, q, u, coo):
        coo.extend(np.zeros((1, 1)), (self.la_NDOF, self.qDOF))

    def g_N_dot_u(self, t, q, coo):
        coo.extend(np.eye(1, dtype=float), (self.la_NDOF, self.uDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def W_N(self, t, q, coo):
        coo.extend(np.eye(1, dtype=float), (self.uDOF, self.la_NDOF))


if __name__ == "__main__":
    # system definition
    m = 1
    radius = 0.1
    # g = 9.81
    g = 10.0
    e_N = 0.5
    # e_N = 0.0
    prox_r_N = 0.5

    # initial conditions
    y0 = 1
    y_dot0 = 0
    r_OS0 = np.array([y0])
    vS0 = np.array([y_dot0])
    q0 = np.array([y0])
    u0 = np.array([y_dot0])

    # create bouncing ball subsystem
    ball = BouncingBall(m, g, radius, prox_r_N=prox_r_N, e_N=e_N, q0=q0, u0=u0)

    # assemble model
    model = Model()
    model.add(ball)
    model.assemble()

    # solver setup
    t0 = 0
    # t1 = 1.5
    t1 = 2
    # dt = 1e-1
    # dt = 5e-2
    dt = 1e-2
    # dt = 5e-3
    # dt = 1e-3
    # dt = 5e-4

    # solve problem
    # solver_other = NonsmoothGeneralizedAlpha(model, t1, dt)
    # solver_other = Generalized_alpha_3(model, t1, dt, numerical_jacobian=True)
    # solver_other = NonsmoothTheta(model, t1, dt, atol=1.0e-8)
    # solver_other = NonsmoothThetaGGL(model, t1, dt)
    # solver_other = NonsmoothEulerBackwardsGGL(model, t1, dt)
    # solver_other = NonsmoothEulerBackwardsGGL_V2(model, t1, dt)
    # solver_other = NonsmoothEulerBackwardsGGL_V3(model, t1, dt)
    # solver_other = NonsmoothHalfExplicitEuler(model, t1, dt)
    # solver_other = NonsmoothHalfExplicitEulerGGL(model, t1, dt)
    # solver_other = NonsmoothNewmarkGGL(model, t1, dt)
    # solver_other = NonsmoothGenAlphaFirstOrder(model, t1, dt, rho_inf=0.9)
    # solver_other = NonsmoothNewmark(model, t1, dt)
    # solver_other = RemcoOriginal(model, t1, dt)
    # solver_other = Remco(model, t1, dt)
    # solver_other = DecoupledNonsmoothHalfExplicitRungeKutta(model, t1, dt)
    solver_other = NonsmoothDecoupled(model, t1, dt)
    sol_other = solver_other.solve()
    t = sol_other.t
    q = sol_other.q
    t_other = sol_other.t
    q_other = sol_other.q
    u_other = sol_other.u
    P_N_other = sol_other.P_N
    # P_F_other = sol_other.P_F
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
        # la_N_other = sol_other.mu_N
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

    ########
    # export
    ########
    def export(file, sol):
        t = sol.t
        q = sol.q
        u = sol.u
        P_N = sol.P_N
        try:
            la_N = sol.la_N
        except:
            la_N = np.zeros_like(P_N)
        try:
            La_N = sol.La_N
        except:
            La_N = np.zeros_like(P_N)
        header = "t, q, u, la_N, La_N, P_N"
        export_data = np.vstack([t, *q.T, *u.T, *la_N.T, La_N.T, P_N.T]).T
        np.savetxt(
            file,
            export_data,
            delimiter=", ",
            header=header,
            comments="",
        )

    export("examples/1DBouncingBall_Moreau.txt", sol_moreau)
    export("examples/1DBouncingBall_Other.txt", sol_other)

    # exit()

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(3, 1)

    ax[0].plot([t_moreau[0], t_moreau[-1]], [radius, radius], "-k", label="ground")
    ax[0].plot(t_moreau, q_moreau[:, 0], "-r", label="q - Moreau")
    ax[0].plot(t_other, q_other[:, 0], "-b", label="q - Other")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t_moreau, u_moreau[:, 0], "-r", label="u - Moreau")
    ax[1].plot(t_other, u_other[:, 0], "-b", label="u - Other")
    ax[1].grid()
    ax[1].legend()

    # ax[2].plot(t_moreau, a_moreau[:, 0], "-r", label="a - Moreau")
    # ax[2].plot(t_other, a_other[:, 0], "-b", label="a - Other")
    # ax[2].grid()
    # ax[2].legend()

    ax[2].plot(t_moreau, P_N_moreau[:, 0], "-r", label="P_N - Moreau")
    ax[2].plot(t_other, dt * la_N_other[:, 0], "-b", label="dt * la_N - Other")
    ax[2].plot(t_other, La_N_other[:, 0], "-g", label="La_N - Other")
    ax[2].plot(t_other, P_N_other[:, 0], "--k", label="P_N - Other")
    ax[2].grid()
    ax[2].legend()

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
