import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math.algebra import norm
from cardillo.math import approx_fprime

from cardillo import System

from cardillo.solver import (
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
        # vg = g * np.array([0, -1])
        # self.la_N0 = np.array([-self.normal(x0) @ vg])
        # print(f"initial contact forces are not correct yet!")
        # if mu > 0:
        #     self.la_F0 = np.array([-self.tangent(x0) @ vg])
        # else:
        #     self.la_F0 = np.zeros(self.nla_F)

        ##################################
        # solve for initial contact forces
        ##################################
        M = m * np.eye(self.nu, dtype=float)
        w_N = self.normal(self.q0[0]).reshape((self.nu, self.nla_N))
        w_F = self.tangent(self.q0[0]).reshape((self.nu, self.nla_F))
        # w_F = np.zeros((self.nu, self.nla_F))

        from cardillo.math import prox_sphere, prox_R0_nm, fsolve

        def R(x):
            *u_dot, la_N, la_F = x
            R = np.zeros_like(x)

            #####################
            # equations of motion
            #####################
            R[:2] = (
                M @ u_dot - np.array([0, -m * g]) - w_N[:, 0] * la_N - w_F[:, 0] * la_F
            )

            #################################
            # Signorini on acceleration level
            #################################
            prox_r_N = 0.1
            g_N = self.g_N(0, self.q0)
            g_N_dot = self.g_N_dot(0, self.q0, self.u0)
            g_N_ddot = self.g_N_ddot(0, self.q0, self.u0, u_dot)
            I_N = g_N <= 0
            # B_N = I_N * (g_N_dot <= 0)
            # C_N = B_N * (g_N_ddot <= 0)
            B_N = I_N * (prox_r_N * g_N_dot - la_N <= 0)
            C_N = B_N * (prox_r_N * g_N_ddot - la_N <= 0)
            # R[2] = np.where(
            #     B_N,
            #     la_N + prox_R0_nm(prox_r_N * g_N_ddot - la_N),
            #     # np.minimum(g_N_ddot, la_N),
            #     la_N,
            # )
            R[2] = np.minimum(g_N_ddot, la_N)  # always start with contact
            # R[2] = np.where(
            #     C_N,
            #     g_N_ddot,
            #     # la_N + prox_R0_nm(prox_r_N * g_N_ddot - la_N),
            #     la_N,
            # )
            # R[2] = la_N + B_N * prox_R0_nm(prox_r_N * g_N_ddot - la_N)
            # R[2] = la_N + prox_R0_nm(prox_r_N * g_N_ddot - la_N)
            # R[2] = g_N_ddot

            ################################
            # friction on acceleration level
            ################################
            prox_r_F = 0.1
            gamma_F = self.__gamma_F(0, self.q0, self.u0)
            gamma_F_dot = self.gamma_F_dot(0, self.q0, self.u0, u_dot)

            if norm(gamma_F) > 0:
                # TODO: Why this case is wrong?
                print(f"slip")
                R[3] = la_F + mu * la_N * gamma_F / norm(gamma_F)
                # R[3] = la_F + prox_sphere(prox_r_F * gamma_F - la_F, mu * la_N)
            else:
                print(f"stick")
                R[3] = la_F + prox_sphere(prox_r_F * gamma_F_dot - la_F, mu * la_N)

            # # norm_gamma_F_dot = norm(gamma_F_dot)
            # # if norm_gamma_F_dot > 0:
            # #     R[3] = la_F + mu * la_N * gamma_F_dot / norm_gamma_F_dot
            # # else:
            # #     R[3] = la_F
            # R[3] = la_F + prox_sphere(prox_r_F * gamma_F_dot - la_F, mu * la_N)
            # R[3] = la_F + prox_sphere(prox_r_F * gamma_F - la_F, mu * la_N)
            # R[3] = la_F + mu * la_N * gamma_F_dot / norm(gamma_F_dot)
            # R[3] = gamma_F_dot

            # # eps = 1e-6
            # # R[3] = la_F + mu * la_N * gamma_F / (eps + norm(gamma_F))
            # R[3] = la_F + prox_sphere(prox_r_F * gamma_F_dot - la_F, mu * la_N)
            # # R[3] = la_F + prox_sphere(prox_r_F * gamma_F - la_F, mu * la_N)

            # radius = mu * la_N
            # prox_arg_F_velocity = prox_r_F * gamma_F - la_F
            # norm_prox_arg_F_velocity = norm(prox_arg_F_velocity)
            # prox_arg_F_acceleration = prox_r_F * gamma_F_dot - la_F
            # norm_prox_arg_F_acceleration = norm(prox_arg_F_acceleration)

            # if C_N:
            #     print(f"acceleration")
            #     R[3] = la_F + prox_sphere(prox_r_F * gamma_F_dot - la_F, mu * la_N)
            # elif B_N:
            #     print(f"velocity")
            #     R[3] = la_F + prox_sphere(prox_r_F * gamma_F - la_F, mu * la_N)
            # else:
            #     R[3] = la_F

            # # # if norm_prox_arg_F_velocity < radius:
            # # # if norm_prox_arg_F_acceleration < radius:
            # if radius < norm_prox_arg_F_velocity:
            # # if radius < norm_prox_arg_F_acceleration:
            #     print(f"slip")
            #     # R[3] = la_F + prox_sphere(prox_r_F * gamma_F - la_F, mu * la_N)
            #     # R[3] = la_F + mu * la_N * gamma_F / norm(gamma_F)
            #     R[3] = la_F + mu * la_N * prox_arg_F_velocity / norm_prox_arg_F_velocity
            # else:
            #     print(f"stick")
            #     # R[3] = gamma_F_dot
            #     R[3] = la_F + prox_sphere(prox_r_F * gamma_F_dot - la_F, mu * la_N)
            #     # R[3] = la_F + mu * la_N * gamma_F / norm(gamma_F)
            #     # R[3] = la_F + prox_sphere(prox_r_F * gamma_F - la_F, mu * la_N)

            # R[3] = la_F + prox_sphere(prox_r_F * gamma_F_dot - la_F, mu * la_N)
            # # R[3] = la_F + prox_sphere(prox_r_F * gamma_F - la_F, mu * la_N)

            # # R[3] = la_F + mu * la_N * gamma_F / norm(gamma_F)
            # # R[2] = la_N - 53.16

            return R

        # x0 = np.random.rand(4)
        x0 = np.ones(4)
        x0, converged, error, i, f = fsolve(R, x0, atol=1e-14)
        assert converged
        *u_dot, la_N, la_F = x0
        # x0 = np.random.rand(3)
        # x0, converged, error, i, f = fsolve(R, x0)
        # *u_dot, la_N = x0
        # print(f"")
        # exit()

        # gamma_F = self.__gamma_F(0, self.q0, self.u0)
        # if norm(gamma_F) > 0:
        #     W_NF = w_N - w_F * gamma_F / norm(gamma_F) * mu
        #     A = np.block([
        #         [M, -W_NF],
        #         [W_NF.T, np.zeros((self.nla_N, self.nla_N))],
        #     ])
        #     b = np.array([
        #         0,
        #         -m * g,
        #         0,
        #     ])
        #     *u_dot, la_N = np.linalg.solve(A, b)
        #     la_F = -gamma_F[0] / norm(gamma_F) * mu * la_N
        # else:
        #     A = np.block([
        #         [M, -w_N, -w_F],
        #         [w_N.T, np.zeros((self.nla_N, self.nla_N)), np.zeros((self.nla_N, self.nla_F))],
        #         [w_F.T, np.zeros((self.nla_F, self.nla_N)), np.zeros((self.nla_F, self.nla_F))],
        #     ])
        #     b = np.array([
        #         0,
        #         -m * g,
        #         0,
        #         0
        #     ])
        #     *u_dot, la_N, la_F = np.linalg.solve(A, b)
        #     la_F = -mu * la_N

        # gamma_F = w_F.T @ self.u0
        # W_NF = w_N - w_F * mu
        # # if norm(gamma_F) > 0:
        # #     W_NF = w_N - w_F * gamma_F / norm(gamma_F) * mu
        # # else:
        # #     W_NF = w_N - w_F * gamma_F * mu
        # A = np.block([
        #     [M, -W_NF],
        #     [W_NF.T, np.zeros((self.nla_N, self.nla_N))],
        # ])
        # b = np.array([
        #     0,
        #     -m * g,
        #     0,
        # ])
        # *u_dot, la_N = np.linalg.solve(A, b)
        # la_F = -mu * la_N
        # # if norm(gamma_F) > 0:
        # #     la_F = -gamma_F[0] / norm(gamma_F) * mu * la_N
        # # else:
        # #     la_F = -gamma_F[0] * mu * la_N

        self.la_N0 = np.array([la_N])
        self.la_F0 = np.array([la_F])
        # self.la_F0 = np.zeros(self.nla_F)
        print(f"")

        # self.la_N0 = np.zeros(self.nla_N)
        # self.la_F0 = np.zeros(self.nla_F)

        # self.q0 = np.array([0, 1.25])
        # self.u0 = np.array([0.0, 0.0])

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
        # return np.array([1, -np.exp(-x)]) / np.sqrt(1 + np.exp(-2 * x))
        f_x = self.f_x(x)
        return f_x / norm(f_x)

    def normal(self, x):
        # return np.array([np.exp(-x), 1]) / np.sqrt(1 + np.exp(-2 * x))
        t = self.tangent(x)
        return np.array([-t[1], t[0]])

    #################
    # normal contacts
    #################
    def r_OQ(self, t, q):
        x, y = q
        return self.f(x)

        # def fun(x):
        #     r_PQ = self.f(x[0]) - q
        #     return r_PQ @ r_PQ

        # from scipy.optimize import minimize
        # res = minimize(fun, [q[0]])
        # assert res.success
        # x = res.x[0]
        # return self.f(x)

        # def J(x):
        #     x = x[0]
        #     n = self.normal(x)
        #     r_OQ = self.f(x)
        #     r_OP = np.array([x, q[1]])
        #     return n @ (r_OQ - r_OP)

        # from cardillo.math import fsolve
        # x, converged, error, i, f = fsolve(J, q[0])
        # assert converged
        # return self.f(x[0])

    def g_N(self, t, q):
        x, y = q
        n = self.normal(x)
        r_OQ = self.r_OQ(t, q)
        g_N = n @ (q - r_OQ)
        return g_N

    def g_N_dot(self, t, q, u):
        x, y = q
        n = self.normal(x)

        return n @ u

        g_N_q = approx_fprime(q, lambda q: self.g_N(t, q), method="3-point")
        return g_N_q @ u + n @ u

    def g_N_ddot(self, t, q, u, u_dot):
        x, y = q
        n = self.normal(x)

        # return n @ u_dot

        g_N_dot_q = approx_fprime(q, lambda q: self.g_N_dot(t, q, u))
        return g_N_dot_q @ u + n @ u_dot

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

        gamma_F_q = approx_fprime(q, lambda q: self.__gamma_F(t, q, u))
        return gamma_F_q @ u + t @ u_dot

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
# x0 = -1
x0 = 1
# x0 = 3
# x_dot0 = 0
x_dot0 = -5

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
    t_final = 3
    # dt = 5e-3
    dt = 1e-2

    # solve problem
    # solver1, label1 = MoreauClassical(model, t_final, dt), "Moreau classical"
    solver1, label1 = Rattle(model, t_final, dt), "Rattle"
    # solver1, label1 = NonsmoothGeneralizedAlpha(model, t_final, dt), "Gen-alpha"
    # solver1, label1 = NonsmoothPIRK(model, t_final, dt, RadauIIATableau(2)), "NPRIK"

    print(f"la_N0: {model.la_N0}; la_F0: {model.la_F0}")
    print(f"dt * la_N0: {dt * model.la_N0}; dt * la_F0: {dt * model.la_F0}")
    # exit()

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


def convergence(use_analytical_solution=False):
    tol = 1.0e-15
    # tol = 1.0e-12
    # tol = 1.0e-10
    # tol = 1.0e-8

    #####################################
    # compute step sizes with powers of 2
    #####################################
    # dt_ref = 1.28e-2
    # dts = (2.0 ** np.arange(3, 1, -1)) * dt_ref  # [5.12e-2, ..., 2.56e-2]
    # t1 = (2.0**8) * dt_ref  # 3.2768s

    # dt_ref = 6.4e-3
    # dts = (2.0 ** np.arange(3, 1, -1)) * dt_ref  # [5.12e-2, ..., 2.56e-2]
    # t1 = (2.0**9) * dt_ref  # 3.2768s

    dt_ref = 3.2e-3
    dts = (2.0 ** np.arange(4, 1, -1)) * dt_ref  # [5.12e-2, ..., 1.28e-2]
    t1 = (2.0**10) * dt_ref  # 3.2768s

    # dt_ref = 1.6e-3
    # dts = (2.0 ** np.arange(5, 1, -1)) * dt_ref  # [5.12e-2, ..., 6.4e-3]
    # t1 = (2.0**11) * dt_ref  # 3.2768s

    # dt_ref = 8e-4
    # dts = (2.0 ** np.arange(6, 1, -1)) * dt_ref  # [5.12e-2, ..., 3.2e-3]
    # t1 = (2.0**12) * dt_ref  # 3.2768s

    # dt_ref = 4e-4
    # dts = (2.0 ** np.arange(7, 1, -1)) * dt_ref  # [5.12e-2, ..., 1.6e-3]
    # t1 = (2.0**13) * dt_ref  # 3.2768s

    # dt_ref = 2e-4
    # dts = (2.0 ** np.arange(8, 1, -1)) * dt_ref  # [5.12e-2, ..., 8e-4]
    # t1 = (2.0**14) * dt_ref  # 3.2768s

    print(f"t1: {t1}")
    print(f"dts: {dts}")
    # exit()

    ################
    # chose a solver
    ################
    # from spook.solver.lobatto import NonsmoothPIRK

    # # get_solver = lambda dt: NonsmoothPIRK(model, t1, dt, RadauIIATableau(3))
    # get_solver = lambda dt: NonsmoothPIRK(model, t1, dt, RadauIIATableau(5))
    # # solver_other = NonsmoothPIRK(model, t1, dt, AlexanderTableau())
    # # solver_other = NonsmoothPIRK(model, t1, dt, LobattoIIIDTableau(4))

    # get_solver = lambda dt: MoreauClassical(model, t1, dt, atol=tol)
    get_solver = lambda dt: Rattle(model, t1, dt)
    # get_solver = lambda dt: NonsmoothBackwardEulerDecoupled(model, t1, dt, atol=tol)
    # get_solver = lambda dt: NonsmoothGeneralizedAlpha(model, t1, dt, newton_tol=tol)
    # get_solver = lambda dt: IRK(model, t1, dt, TRBDF2Tableau(), atol=tol)
    # get_solver = lambda dt: IRK(model, t1, dt, RadauIIATableau(order=3), atol=tol)
    # get_solver = lambda dt: Rattle(model, t1, dt, atol=tol)
    # get_solver = lambda dt: SymplecticPartitionedRungeKutta(
    #     model, t1, dt, TRBDF2Tableau(), atol=tol
    # )
    # get_solver = lambda dt: SymplecticPartitionedRungeKutta(
    #     model, t1, dt, LobattoIIIATableau(order=2), atol=tol
    # )
    # get_solver = lambda dt: SymplecticPartitionedRungeKutta(
    #     model, t1, dt, LobattoIIIATableau(order=4), atol=tol
    # )
    # get_solver = lambda dt: SymplecticPartitionedRungeKutta(
    #     model, t1, dt, LobattoIIIATableau(order=6), atol=tol
    # )

    # get_solver = lambda dt: SimplifiedNonsmoothGeneralizedAlpha(model, t1, dt, atol=tol)
    # get_solver = lambda dt: SimplifiedNonsmoothGeneralizedAlphaNoAcceleration(model, t1, dt, atol=tol)
    # get_solver = lambda dt: SimplifiedNonsmoothGeneralizedFirstOrderAlphaNoAcceleration(model, t1, dt, atol=tol)

    #############################
    # errors for possible solvers
    #############################
    q_errors_transient = np.inf * np.ones(len(dts), dtype=float)
    u_errors_transient = np.inf * np.ones(len(dts), dtype=float)
    P_N_errors_transient = np.inf * np.ones(len(dts), dtype=float)
    P_F_errors_transient = np.inf * np.ones(len(dts), dtype=float)
    q_errors_longterm = np.inf * np.ones(len(dts), dtype=float)
    u_errors_longterm = np.inf * np.ones(len(dts), dtype=float)
    P_N_errors_longterm = np.inf * np.ones(len(dts), dtype=float)
    P_F_errors_longterm = np.inf * np.ones(len(dts), dtype=float)
    q_errors_integral = np.inf * np.ones(len(dts), dtype=float)
    u_errors_integral = np.inf * np.ones(len(dts), dtype=float)
    P_N_errors_integral = np.inf * np.ones(len(dts), dtype=float)
    P_F_errors_integral = np.inf * np.ones(len(dts), dtype=float)

    #####################
    # create export files
    #####################
    path = Path(__file__)

    file_q = path.parent / f"{path.stem}_q.txt"
    file_u = path.parent / f"{path.stem}_u.txt"
    file_P_N = path.parent / f"{path.stem}_P_N.txt"
    file_P_F = path.parent / f"{path.stem}_P_F.txt"
    header = "dt, dt2, error"

    def create(name):
        with open(name, "w") as file:
            file.write(header)
        with open(name, "ab") as file:
            file.write(b"\n")

    def append(name, data):
        with open(name, "ab") as file:
            np.savetxt(
                file,
                data,
                delimiter=", ",
                comments="",
            )

    # create(file_q)
    # create(file_u)
    # create(file_P_N)
    # create(file_P_F)

    ###################################################################
    # compute reference solution as described in Arnold2015 Section 3.3
    ###################################################################
    if not use_analytical_solution:
        print(f"compute reference solution:")
        reference = get_solver(dt_ref).solve()
        # reference = Moreau(model, t1, dt_ref, fix_point_tol=tol).solve()
        print(f"done")

        def errors(sol, sol_ref, t_transient=1.1, t_longterm=1.5):
            # extract solution an remove first time step
            t = sol.t[1:]
            q = sol.q[1:]
            u = sol.u[1:]
            P_N = sol.P_N[1:]
            P_F = sol.P_F[1:]

            t_ref = sol_ref.t[1:]
            q_ref = sol_ref.q[1:]
            u_ref = sol_ref.u[1:]
            P_N_ref = sol_ref.P_N[1:]
            P_F_ref = sol_ref.P_F[1:]

            # distinguish between transient and long term time steps
            t_idx_transient = np.where(t <= t_transient)[0]
            t_idx_longterm = np.where(t >= t_longterm)[0]

            # compute difference between computed solution and reference solution
            # for identical time instants
            t_ref_idx = np.where(np.abs(t[:, None] - t_ref) < 1.0e-8)[1]
            t_ref_idx_transient = np.where(
                np.abs(t[t_idx_transient, None] - t_ref) < 1.0e-8
            )[1]
            t_ref_idx_longterm = np.where(
                np.abs(t[t_idx_longterm, None] - t_ref) < 1.0e-8
            )[1]

            def max_relative_error(f, f_ref):
                diff = f - f_ref
                return np.max(
                    np.linalg.norm(diff, axis=1) / np.linalg.norm(f_ref, axis=1)
                )

            def integral_error(f, f_ref):
                diff = f - f_ref
                diff2 = np.linalg.norm(diff, axis=1) ** 2
                # diff2 = np.array([d @ d for d in diff])

                # integral = np.trapz(diff2, t, dx=dt, axis=0)
                integral = dt * np.sum(diff2)

                error = np.sqrt(integral)
                return error

            # pre stick
            q_error_transient = max_relative_error(
                q[t_idx_transient], q_ref[t_ref_idx_transient]
            )
            u_error_transient = max_relative_error(
                u[t_idx_transient], u_ref[t_ref_idx_transient]
            )
            P_N_error_transient = max_relative_error(
                P_N[t_idx_transient], P_N_ref[t_ref_idx_transient]
            )
            P_F_error_transient = max_relative_error(
                P_F[t_idx_transient], P_F_ref[t_ref_idx_transient]
            )

            # post stick
            q_error_longterm = max_relative_error(
                q[t_idx_longterm], q_ref[t_ref_idx_longterm]
            )
            u_error_longterm = max_relative_error(
                u[t_idx_longterm], u_ref[t_ref_idx_longterm]
            )
            P_N_error_longterm = max_relative_error(
                P_N[t_idx_longterm], P_N_ref[t_ref_idx_longterm]
            )
            P_F_error_longterm = max_relative_error(
                P_F[t_idx_longterm], P_F_ref[t_ref_idx_longterm]
            )

            # integral
            q_error_integral = integral_error(q, q_ref[t_ref_idx])
            u_error_integral = integral_error(u, u_ref[t_ref_idx])
            P_N_error_integral = integral_error(P_N, P_N_ref[t_ref_idx])
            P_F_error_integral = integral_error(P_F, P_F_ref[t_ref_idx])

            return (
                q_error_transient,
                u_error_transient,
                P_N_error_transient,
                P_F_error_transient,
                q_error_longterm,
                u_error_longterm,
                P_N_error_longterm,
                P_F_error_longterm,
                q_error_integral,
                u_error_integral,
                P_N_error_integral,
                P_F_error_integral,
            )

    else:
        raise NotImplementedError

        def errors(sol, t_transient=1.1, t_longterm=1.5):
            t = sol.t
            q = sol.q
            u = sol.u
            P_N = sol.P_N
            P_F = sol.P_F

            # analytical solution
            r_OP, v_P = ball.solution(t)

            # distinguish between transient and long term time steps
            t_idx_transient = np.where(t <= t_transient)[0]
            t_idx_longterm = np.where(t >= t_longterm)[0]

            # differences
            q_transient = q[t_idx_transient]
            u_transient = u[t_idx_transient]
            # P_N_transient = P_N[t_idx_transient]
            # P_F_transient = P_F[t_idx_transient]
            diff_transient_q = q_transient - r_OP[t_idx_transient]
            diff_transient_u = u_transient - v_P[t_idx_transient]
            # diff_transient_P_N = P_N_transient - P_N_ref[t_idx_transient]
            # diff_transient_P_F = P_F_transient - P_F_ref[t_idx_transient]

            q_longterm = q[t_idx_longterm]
            u_longterm = u[t_idx_longterm]
            # P_N_longterm = P_N[t_idx_longterm]
            # P_F_longterm = P_F[t_idx_longterm]
            diff_longterm_q = q_longterm - r_OP[t_idx_longterm]
            diff_longterm_u = u_longterm - v_P[t_idx_longterm]
            # diff_longterm_P_N = P_N_longterm - P_N_ref[t_idx_longterm]
            # diff_longterm_P_F = P_F_longterm - P_F_ref[t_idx_longterm]

            # max relative error
            q_error_transient = np.max(
                np.linalg.norm(diff_transient_q, axis=1)
                / np.linalg.norm(q_transient, axis=1)
            )
            u_error_transient = np.max(
                np.linalg.norm(diff_transient_u, axis=1)
                / np.linalg.norm(u_transient, axis=1)
            )
            # P_N_error_transient = np.max(
            #     np.linalg.norm(diff_transient_P_N, axis=1)
            #     / np.linalg.norm(P_N_transient, axis=1)
            # )
            # P_F_error_transient = np.max(
            #     np.linalg.norm(diff_transient_P_F, axis=1)
            #     / np.linalg.norm(P_F_transient, axis=1)
            # )

            q_error_longterm = np.max(
                np.linalg.norm(diff_longterm_q, axis=1)
                / np.linalg.norm(q_longterm, axis=1)
            )
            u_error_longterm = np.max(
                np.linalg.norm(diff_longterm_u, axis=1)
                / np.linalg.norm(u_longterm, axis=1)
            )
            # P_N_error_longterm = np.max(
            #     np.linalg.norm(diff_longterm_P_N, axis=1)
            #     / np.linalg.norm(P_N_longterm, axis=1)
            # )
            # P_F_error_longterm = np.max(
            #     np.linalg.norm(diff_longterm_P_F, axis=1)
            #     / np.linalg.norm(P_F_longterm, axis=1)
            # )

            return (
                q_error_transient,
                u_error_transient,
                # P_N_error_transient,
                # P_F_error_transient,
                q_error_longterm,
                u_error_longterm,
                # P_N_error_longterm,
                # P_F_error_longterm,
            )

    for i, dt in enumerate(dts):
        print(f"i: {i}, dt: {dt:1.1e}")

        # solve problem with current time step
        sol = get_solver(dt).solve()
        if use_analytical_solution:
            (
                q_errors_transient[i],
                u_errors_transient[i],
                q_errors_longterm[i],
                u_errors_longterm[i],
            ) = errors(sol)
        else:
            (
                q_errors_transient[i],
                u_errors_transient[i],
                P_N_errors_transient[i],
                P_F_errors_transient[i],
                q_errors_longterm[i],
                u_errors_longterm[i],
                P_N_errors_longterm[i],
                P_F_errors_longterm[i],
                q_errors_integral[i],
                u_errors_integral[i],
                P_N_errors_integral[i],
                P_F_errors_integral[i],
            ) = errors(sol, reference)

        # # append(
        # #     file_transient_q,
        # #     np.array([[dts_1[i], dts_2[i], *q_errors_transient[:, i]]]),
        # # )
        # # append(
        # #     file_transient_u,
        # #     np.array([[dts_1[i], dts_2[i], *u_errors_transient[:, i]]]),
        # # )
        # # append(
        # #     file_transient_la_g,
        # #     np.array([[dts_1[i], dts_2[i], *P_N_errors_transient[:, i]]]),
        # # )
        # # append(
        # #     file_transient_la_gamma,
        # #     np.array([[dts_1[i], dts_2[i], *P_F_errors_transient[:, i]]]),
        # # )
        # append(
        #     file_q, np.array([[dts_1[i], dts_2[i], q_errors_longterm[i]]])
        # )
        # append(
        #     file_u, np.array([[dts_1[i], dts_2[i], u_errors_longterm[i]]])
        # )
        # append(
        #     file_P_N,
        #     np.array([[dts_1[i], dts_2[i], P_N_errors_longterm[i]]]),
        # )
        # append(
        #     file_P_F,
        #     np.array([[dts_1[i], dts_2[i], P_F_errors_longterm[i]]]),
        # )

    # plot(reference)

    ##################
    # visualize errors
    ##################
    fig, ax = plt.subplots(1, 3)

    ax[0].set_title("pre stick")
    ax[0].loglog(dts, dts, "-k", label="dt")
    ax[0].loglog(dts, dts**2, "--k", label="dt^2")
    ax[0].loglog(dts, dts**3, "-.k", label="dt^3")
    ax[0].loglog(dts, dts**4, ":k", label="dt^4")
    ax[0].loglog(dts, dts**5, "--y", label="dt^5")
    ax[0].loglog(dts, dts**6, "-.y", label="dt^6")
    ax[0].loglog(dts, q_errors_transient, "-.ro", label="q")
    ax[0].loglog(dts, u_errors_transient, "-.go", label="u")
    ax[0].loglog(dts, P_N_errors_transient, "-.bo", label="P_N")
    ax[0].loglog(dts, P_F_errors_transient, "-.ko", label="P_F")
    ax[0].grid()
    ax[0].legend()

    ax[1].set_title("post stick")
    ax[1].loglog(dts, dts, "-k", label="dt")
    ax[1].loglog(dts, dts**2, "--k", label="dt^2")
    ax[1].loglog(dts, dts**3, "-.k", label="dt^3")
    ax[1].loglog(dts, dts**4, ":k", label="dt^4")
    ax[1].loglog(dts, dts**5, "--y", label="dt^5")
    ax[1].loglog(dts, dts**6, "-.y", label="dt^6")
    ax[1].loglog(dts, q_errors_longterm, "-.ro", label="q")
    ax[1].loglog(dts, u_errors_longterm, "-.go", label="u")
    ax[1].loglog(dts, P_N_errors_longterm, "-.bo", label="P_N")
    ax[1].loglog(dts, P_F_errors_longterm, "-.ko", label="P_F")
    ax[1].grid()
    ax[1].legend()

    ax[2].set_title("integral")
    ax[2].loglog(dts, dts, "-k", label="dt")
    ax[2].loglog(dts, dts**2, "--k", label="dt^2")
    ax[2].loglog(dts, dts**3, "-.k", label="dt^3")
    ax[2].loglog(dts, dts**4, ":k", label="dt^4")
    ax[2].loglog(dts, dts**5, "--y", label="dt^5")
    ax[2].loglog(dts, dts**6, "-.y", label="dt^6")
    ax[2].loglog(dts, q_errors_integral, "-.ro", label="q")
    ax[2].loglog(dts, u_errors_integral, "-.go", label="u")
    ax[2].loglog(dts, P_N_errors_integral, "-.bo", label="P_N")
    ax[2].loglog(dts, P_F_errors_integral, "-.ko", label="P_F")
    ax[2].grid()
    ax[2].legend()

    plt.show()


if __name__ == "__main__":
    state()
    # convergence()
