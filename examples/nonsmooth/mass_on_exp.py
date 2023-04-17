import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math.algebra import norm
from cardillo.math import approx_fprime

from cardillo import System

from cardillo.solver import (
    convergence_analysis,
    MoreauClassical,
    MoreauShifted,
    Rattle,
    NonsmoothPIRK,
    RadauIIATableau,
    NonsmoothGeneralizedAlpha,
)


class BallOnExp:
    def __init__(
        self,
        m,
        g,
        mu,
        x0,
        x_dot0,
        e_N,
        e_F,
    ):
        self.m = m
        self.g = g
        self.nq = 2
        self.nu = 2
        self.nla_N = 1
        if mu > 0:
            self.nla_F = 1
            self.NF_connectivity = [[0]]
            self.gamma_F = self.__gamma_F
        else:
            self.nla_F = 0
            self.NF_connectivity = [[]]

        self.mu = np.array([mu])

        self.e_N = np.array([e_N])
        self.e_F = np.array([e_F])

        self.x0 = x0
        self.x_dot0 = x_dot0
        self.q0 = self.f(x0)
        self.u0 = self.tangent(x0) * self.x_dot0

        # self.q0 = np.array([0, 1.5])

    ####################
    # kinematic equation
    ####################
    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q, coo):
        coo.extend_diag(np.ones(self.nq), (self.qDOF, self.uDOF))

    #####################
    # equations of motion
    #####################
    def M(self, t, q, coo):
        coo.extend(m * np.eye(self.nu, dtype=float), (self.uDOF, self.uDOF))

    def h(self, t, q, u):
        return np.array([0.0, -self.m * self.g])

    ################
    # curve geometry
    ################
    def f(angle, x):
        return np.array([x, np.exp(-x)])

    def f_x(angle, x):
        return np.array([1, -np.exp(-x)])

    def f_xx(angle, x):
        return np.array([0, np.exp(-x)])

    def tangent(self, x):
        return np.array([1, -np.exp(-x)]) / np.sqrt(1 + np.exp(-2 * x))
        # f_x = self.f_x(x)
        # return f_x / norm(f_x)

    def tangent_x(self, x):
        ex = np.exp(-x)
        ex2 = np.exp(-2 * x)
        denom = 1 + ex2
        return np.array([1, -ex]) * ex2 / (denom) ** (3 / 2) + np.array(
            [0, ex]
        ) / np.sqrt(denom)

    def normal(self, x):
        return np.array([np.exp(-x), 1]) / np.sqrt(1 + np.exp(-2 * x))
        # t = self.tangent(x)
        # return np.array([-t[1], t[0]])

    def normal_x(self, x):
        ex = np.exp(-x)
        ex2 = np.exp(-2 * x)
        denom = 1 + ex2
        return np.array([ex, 1]) * ex2 / (denom) ** (3 / 2) + np.array(
            [-ex, 0]
        ) / np.sqrt(denom)

    #################
    # normal contacts
    #################
    def g_N(self, t, q):
        n = self.normal(q[0])
        r_OQ = self.f(q[0])
        return np.array([n @ (q - r_OQ)])

    def g_N_dot(self, t, q, u):
        n = self.normal(q[0])
        return np.array([n @ u])

        # # Note: This is unnecessary, since q - r_OQ = 0 for closed contacts!
        # r_OQ = self.f(q[0])
        # n_x = self.normal_x(q[0])
        # return n @ u + (q - r_OQ) @ n_x * u[0]

    def g_N_ddot(self, t, q, u, u_dot):
        n = self.normal(q[0])
        n_x = self.normal_x(q[0])
        return np.array([n @ u_dot + u @ n_x * u[0]])

    def g_N_q(self, t, q, coo):
        dense = approx_fprime(q, lambda q: self.g_N(t, q))
        coo.extend(dense, (self.la_NDOF, self.qDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def W_N(self, t, q, coo):
        coo.extend(self.normal(q[0]), (self.uDOF, self.la_NDOF))

    def Wla_N_q(self, t, q, la_N, coo):
        dense = approx_fprime(
            q, lambda q: la_N[0] * self.normal(q[0]).reshape(self.nu, self.nla_N)
        )
        coo.extend(dense, (self.uDOF, self.la_NDOF))

    ##################
    # tangent contacts
    ##################
    def __gamma_F(self, t, q, u):
        t = self.tangent(q[0])
        return np.array([t @ u])

    def gamma_F_dot(self, t, q, u, u_dot):
        t = self.tangent(q[0])
        return np.array([t @ u_dot])

        # # Note: This is unnecessary, since u @ t_dot = alpha * g_N_dot = 0!
        # t_x = self.tangent_x(q[0])
        # return np.array([t @ u_dot + u @ t_x * u[0]])

    def gamma_F_q(self, t, q, u, coo):
        dense = approx_fprime(q, lambda q: self.__gamma_F(t, q, u))
        coo.extend(dense, (self.la_FDOF, self.qDOF))

    def xi_F(self, t, q, u_pre, u_post):
        return self.__gamma_F(t, q, u_post) + self.e_N * self.__gamma_F(t, q, u_pre)

    def W_F(self, t, q, coo):
        coo.extend(self.tangent(q[0]), (self.uDOF, self.la_FDOF))

    def Wla_F_q(self, t, q, la_F, coo):
        dense = approx_fprime(
            q, lambda q: la_F[0] * self.tangent(q[0]).reshape(self.nu, self.nla_F)
        )
        coo.extend(dense, (self.uDOF, self.la_FDOF))


# system definition
m = np.pi
g = 10.0
e_N = 0.0
e_F = 0.0
mu = 0.3
# mu = 3
# mu = 0.0
# x0 = -2
x0 = 0
# x0 = 3
x_dot0 = 0
# x_dot0 = 1
# x_dot0 = -1

# create bouncing ball subsystem
ball = BallOnExp(m, g, mu, x0, x_dot0, e_N, e_F)

# assemble model
model = System()
model.add(ball)
model.assemble()


def plot(sol):
    t = sol.t
    q = sol.q
    u = sol.u
    P_N = sol.P_N
    try:
        P_F = sol.P_F
    except:
        pass

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(4, 1)

    # ax[0].plot(s, parabola, "-k", label="ground")
    ax[0].plot(t, q[:, 0], "-xr", label="x")
    ax[0].plot(t, q[:, 1], "--or", label="y")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, u[:, 0], "-xr", label="u_x")
    ax[1].plot(t, u[:, 1], "--or", label="u_y")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(t, P_N[:, 0], "-r", label="P_N")
    ax[2].grid()
    ax[2].legend()

    try:
        ax[3].plot(t, P_F[:, 0], "-r", label="P_F - Moreau")
        ax[3].grid()
        ax[3].legend()
    except:
        pass


def state():
    # solver setup
    # t_final = 0.1
    t_final = 3.2768
    dt = 1e-2
    # dt = 5e-3
    # dt = 1e-3

    # solve problem
    solver1, label1 = MoreauClassical(model, t_final, dt), "Moreau classical"
    # solver1, label1 = Rattle(model, t_final, dt), "Rattle"
    # solver1, label1 = NonsmoothGeneralizedAlpha(model, t_final, dt), "Gen-alpha"
    # solver1, label1 = NonsmoothPIRK(model, t_final, dt, RadauIIATableau(2)), "NPRIK"

    sol1 = solver1.solve()
    t1 = sol1.t
    q1 = sol1.q
    u1 = sol1.u
    P_N1 = sol1.P_N
    P_F1 = sol1.P_F

    solver2, label2 = MoreauShifted(model, t_final, dt), "Moreau shifted"

    sol2 = solver2.solve()
    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u
    # a2 = np.zeros_like(u2)
    # a2[1:] = (u2[1:] - u2[:-1]) / dt
    P_N_moreau = sol2.P_N
    P_F_moreau = sol2.P_F

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(4, 1)

    # # analytical solution
    # r_OP, v_P = ball.solution(t_other)

    # ax[0].plot(s, parabola, "-k", label="ground")
    ax[0].plot(t2, q2[:, 0], "-xr", label="x - " + label2)
    ax[0].plot(t2, q2[:, 1], "--or", label="y - " + label2)
    ax[0].plot(t1, q1[:, 0], "-xb", label="x - " + label1)
    ax[0].plot(t1, q1[:, 1], "--ob", label="y - " + label1)
    # ax[0].plot(t_other, r_OP[:, 0], "--k", label="x")
    # ax[0].plot(t_other, r_OP[:, 1], "-.k", label="y")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t2, u2[:, 0], "-xr", label="u_x - " + label2)
    ax[1].plot(t2, u2[:, 1], "--or", label="u_y - " + label2)
    ax[1].plot(t1, u1[:, 0], "-xb", label="u_x - " + label1)
    ax[1].plot(t1, u1[:, 1], "-ob", label="u_y - " + label1)
    # ax[1].plot(t_other, v_P[:, 0], "--k", label="x")
    # ax[1].plot(t_other, v_P[:, 1], "-.k", label="y")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(t2, P_N_moreau[:, 0], "-r", label="P_N - " + label2)
    ax[2].plot(t1, P_N1[:, 0], "--b", label="P_N - " + label1)
    ax[2].grid()
    ax[2].legend()

    try:
        ax[3].plot(t2, P_F_moreau[:, 0], "-r", label="P_F - " + label2)
        ax[3].plot(t1, P_F1[:, 0], "--b", label="P_F - " + label1)
        ax[3].grid()
        ax[3].legend()
    except:
        pass

    ###########
    # animation
    ###########
    lim0 = 1
    lim1 = 4
    s = np.linspace(-lim0, lim1, num=100)
    # plane = np.array([BallOnExp.plane(angle, si) for si in s])
    plane = np.array([ball.f(si) for si in s])

    # t = t_moreau
    # q = q_moreau
    t1 = t1
    q1 = q1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.set_xlim(-lim0, lim1)
    ax.set_ylim(-lim0, lim1)

    # prepare data for animation
    frames = len(t1)
    target_frames = min(len(t1), 200)
    frac = int(frames / target_frames)
    animation_time = 3
    interval = animation_time * 1000 / target_frames

    frames = target_frames
    t1 = t1[::frac]
    q1 = q1[::frac]

    # parabola
    ax.plot(*plane.T, "-k")

    # point mass
    (com,) = ax.plot([], [], "-or")

    def update(t, q, com):
        x, y = q
        com.set_data([x], [y])

        return com

    def animate(i):
        update(t1[i], q1[i], com)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()


def convergence():
    # get_solver = lambda t_final, dt, atol: MoreauClassical(
    #     model, t_final, dt, atol=atol
    # )
    # get_solver = lambda t_final, dt, atol: Rattle(model, t_final, dt, atol=atol)
    # get_solver = lambda t_final, dt, atol: NonsmoothGeneralizedAlpha(model, t_final, dt, newton_tol=atol)
    get_solver = lambda t_final, dt, atol: NonsmoothPIRK(
        model, t_final, dt, RadauIIATableau(2), atol=atol
    )

    errors = convergence_analysis(
        get_solver,
        # dt_ref=3.2e-3,
        # # final_power=10,
        # final_power=6,
        # power_span=(1, 4),
        # dt_ref=1.6e-3,
        # # final_power=11,
        # final_power=7,
        # power_span=(1, 5),
        dt_ref=8e-4,
        # final_power=12,
        final_power=8,
        power_span=(1, 6),
        # dt_ref=4e-4,
        # final_power=13,
        # # final_power=9,
        # power_span=(1, 7),
        # #############
        # # final setup
        # #############
        # dt_ref=1e-4,
        # final_power=15,
        # power_span=(1, 9),
        #
        states=["q", "u", "P_N", "P_F"],
        atol=1e-12,
        measure="lp",
        visualize=True,
        export=True,
        kwargs={"p": 1},
    )


if __name__ == "__main__":
    # state()
    convergence()
