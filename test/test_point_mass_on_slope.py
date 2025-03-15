import numpy as np
import pytest
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate._ivp.tests.test_ivp import compute_error

from cardillo.math.prox import Sphere
from cardillo import System
from cardillo.solver import (
    BackwardEuler,
    Moreau,
    Rattle,
    DualStörmerVerlet,
    SolverOptions,
)
from cardillo.utility.convergence_analysis import convergence_analysis


solvers_names_kwargs = [
    # (BackwardEuler, "BackwardEuler", {}),
    # (Moreau, "Moreau", {}),
    # (Rattle, "Rattle", {}),
    (DualStörmerVerlet, "DualStörmerVerlet", {}),
]

# system definition
m = np.pi
g = 10.0
mu = 0.3
x0 = 0
x_dot0 = 0

case = 1
# case = 2

if case == 1:
    q0 = None
elif case == 2:
    q0 = np.array([0, 1.5])
else:
    raise NotImplementedError


class BallOnExp:
    def __init__(
        self,
        m,
        g,
        mu,
        x0,
        x_dot0,
        q0=None,
    ):
        self.m = m
        self.g = g
        self.nq = 2
        self.nu = 2

        self.nla_N = 1
        self.e_N = np.zeros(self.nla_N)

        if mu > 0:
            self.nla_F = 1
            # fmt: off
            self.friction_laws = [
                ([0], [0], Sphere(mu)), # Coulomb
            ]
            # fmt: on
            self.gamma_F = self.__gamma_F
        else:
            self.nla_F = 0
            self.friction_laws = []

        self.e_F = np.zeros(self.nla_F)

        self.x0 = x0
        self.x_dot0 = x_dot0
        if q0 is None:
            self.q0 = self.f(x0)
        else:
            self.q0 = q0
        self.u0 = self.tangent(x0) * self.x_dot0

    ####################
    # kinematic equation
    ####################x0
    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def q_dot_u(self, t, q):
        return np.eye(self.nq)

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return m * np.eye(self.nu, dtype=float)

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

    def g_N_q(self, t, q):
        x = q[0]
        g_N_q = (self.normal(x).T @ np.eye(self.nq)).reshape(self.nla_N, self.nq)
        g_N_q[0] += x * self.normal_x(x)
        return g_N_q

        # from cardillo.math.approx_fprime import approx_fprime
        # g_N_q_num = approx_fprime(q, lambda q: self.g_N(t, q), method="cs").reshape(
        #     self.nla_N, self.nq
        # )
        # diff = g_N_q - g_N_q_num
        # error = np.linalg.norm(diff)
        # print(f"error g_N_q: {error}")
        # return g_N_q_num

    def W_N(self, t, q):
        return self.normal(q[0])[:, np.newaxis]

    def Wla_N_q(self, t, q, la_N):
        Wla_N_q = np.zeros((self.nu, self.nq))
        Wla_N_q[:, 0] = la_N[0] * self.normal_x(q[0])
        return Wla_N_q

        # from cardillo.math.approx_fprime import approx_fprime
        # Wla_N_q_num = approx_fprime(
        #     q,
        #     lambda q: la_N[0] * self.normal(q[0]).reshape(self.nu, self.nla_N),
        #     method="cs",
        # )
        # diff = Wla_N_q - Wla_N_q_num
        # error = np.linalg.norm(diff)
        # print(f"error Wla_N_q: {error}")
        # return Wla_N_q_num

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

    def gamma_F_q(self, t, q, u):
        gamma_F_q = np.zeros((self.nla_F, self.nq))
        gamma_F_q[0, 0] = u.T @ self.tangent_x(q[0])
        return gamma_F_q

        # from cardillo.math.approx_fprime import approx_fprime
        # gamma_F_q_num = approx_fprime(q, lambda q: self.__gamma_F(t, q, u), method="cs").reshape(
        #     self.nla_F, self.nq
        # )
        # diff = gamma_F_q - gamma_F_q_num
        # error = np.linalg.norm(diff)
        # print(f"error gamma_F_q: {error}")
        # return gamma_F_q_num

    def W_F(self, t, q):
        return self.tangent(q[0])[:, np.newaxis]

    def Wla_F_q(self, t, q, la_F):
        Wla_F_q = np.zeros((self.nu, self.nq))
        Wla_F_q[:, 0] = la_F[0] * self.tangent_x(q[0])
        return Wla_F_q

        # Wla_F_q_num = approx_fprime(
        #     q,
        #     lambda q: la_F[0] * self.tangent(q[0]).reshape(self.nu, self.nla_F),
        #     method="cs",
        # )
        # diff = Wla_F_q - Wla_F_q_num
        # error = np.linalg.norm(diff)
        # print(f"error Wla_F_q: {error}")
        # return Wla_F_q_num


system = System()
ball = BallOnExp(m, g, mu, x0, x_dot0, q0=q0)
system.add(ball)
system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

# @pytest.mark.parametrize("Solver, kwargs", solvers_and_kwargs)
# def test_index3_problem(Solver, kwargs, show=False):
#     # create the system
#     system = System()
#     ball = BallOnExp()
#     system.add(ball)
#     system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

#     # call the solver
#     t1 = 2 * np.pi
#     t1 *= 0.1
#     dt = 1e-2
#     sol = Solver(system, t1, dt, **kwargs).solve()
#     t = sol.t
#     q = sol.q
#     u = sol.u

#     # compare with exact solution
#     q_true, u_true, la_true = sol_true(t)
#     error_q = compute_error(q, q_true, rtol=1e-6, atol=1e-3)
#     error_u = compute_error(u, u_true, rtol=1e-6, atol=1e-3)
#     print(f"error q: {error_q}")
#     print(f"error u: {error_u}")

#     # visualization
#     if show:
#         fig, ax = plt.subplots(2, 1)

#         ax[0].plot(t, q[:, 0], "-r", label="x")
#         ax[0].plot(t, q_true[:, 0], "rx", label="x_true")
#         ax[0].plot(t, q[:, 1], "-g", label="y")
#         ax[0].plot(t, q_true[:, 1], "gx", label="y_true")
#         ax[0].grid()
#         ax[0].legend()

#         ax[1].plot(t, u[:, 0], "-r", label="u")
#         ax[1].plot(t, u_true[:, 0], "rx", label="u_true")
#         ax[1].plot(t, u[:, 1], "-g", label="v")
#         ax[1].plot(t, u_true[:, 1], "gx", label="v_true")
#         ax[1].grid()
#         ax[1].legend()

#         plt.show()

#     # convergence analysis
#     global first
#     first = True

#     def get_solver(t_final, dt, atol):
#         global first
#         if first:
#             first = False
#             t_true = np.arange(0, t_final + dt, dt)
#             q_true, u_true, la_true = sol_true(t_true)
#             return type(
#                 "Solver",
#                 (),
#                 {
#                     "solve": lambda self: Solution(
#                         system,
#                         t=t_true,
#                         q=q_true,
#                         u=u_true,
#                         P_g=dt * la_true,
#                     )
#                 },
#             )()
#         else:
#             return Solver(
#                 system,
#                 t_final,
#                 dt,
#                 options=SolverOptions(
#                     fixed_point_atol=atol,
#                     fixed_point_rtol=atol,
#                     newton_atol=atol,
#                     newton_rtol=atol,
#                     reuse_lu_decomposition=False,
#                 ),
#                 **kwargs,
#             )

#     errors = convergence_analysis(
#         get_solver,
#         #
#         # dt_ref=1.6e-3,
#         # final_power=7,
#         # power_span=(1, 5),
#         #
#         # dt_ref=8e-4,
#         # final_power=8,
#         # power_span=(1, 6),
#         #
#         # dt_ref=4e-4,
#         # final_power=9,
#         # power_span=(1, 7),
#         #
#         dt_ref=2e-4,
#         final_power=10,
#         power_span=(1, 8),
#         #
#         #############
#         # other setup
#         #############
#         states=["q", "u", "P_g"],
#         measure="lp",
#         visualize=show,
#         export=True,
#         # export_name="Rattle",
#         kwargs={"p": 1},
#     )


# if __name__ == "__main__":
#     for p in solvers_and_kwargs:
#         test_index3_problem(*p, show=True)


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

    # solve problem
    solver1, label1 = (
        DualStörmerVerlet(system, t_final, dt),
        "DualStörmerVerlet",
    )

    sol1 = solver1.solve()
    t1 = sol1.t
    q1 = sol1.q
    u1 = sol1.u
    P_N1 = sol1.P_N
    P_F1 = sol1.P_F

    solver2, label2 = Moreau(system, t_final, dt), "Moreau"

    sol2 = solver2.solve()
    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u
    # a2 = np.zeros_like(u2)
    # a2[1:] = (u2[1:] - u2[:-1]) / dt
    P_N_moreau = sol2.P_N
    P_F_moreau = sol2.P_F

    # ########
    # # export
    # ########
    # export_txt(
    #     system,
    #     sol1,
    #     ["q", "u", "P_N", "P_F"],
    #     ["g_N", "g_N_dot", "gamma_F"],
    # )

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
    for Solver, label, kwargs in solvers_names_kwargs:
        # append solver options
        # kwargs.update({"options": options})

        # get_solver = lambda t_final, dt: Solver(
        #     system,
        #     t_final,
        #     dt,
        #     # solution_method="fixed-point",
        #     # solution_method="semismooth Newton",
        #     # options=options,
        #     **kwargs,
        # )
        get_solver = lambda t_final, dt, atol: Solver(
            system,
            t_final,
            dt,
            options=SolverOptions(
                fixed_point_atol=atol,
                fixed_point_rtol=atol,
                newton_atol=atol,
                newton_rtol=atol,
                newton_max_iter=200,
                reuse_lu_decomposition=False,
            ),
            **kwargs,
        )

        final_power = 9

        errors = convergence_analysis(
            get_solver,
            #
            dt_ref=1.6e-3,
            final_power=final_power,
            power_span=(1, 6),
            #
            # dt_ref=8e-4,
            # final_power=final_power + 1,
            # # power_span=(1, 7), # too large time-steps
            # power_span=(1, 6),
            #
            # dt_ref=4e-4,
            # final_power=final_power + 2,
            # # power_span=(1, 8), # too large time-steps
            # power_span=(1, 7),
            #
            # # used for pendulum example
            # dt_ref=2e-4,
            # final_power=final_power + 3,
            # power_span=(1, 9),  # TODO: Check if this step-size is too large
            #
            states=["q", "u", "P_N", "P_F"],
            measure="lp",
            visualize=True,
            # visualize=False,
            export=False,
            kwargs={"p": 1},
        )


if __name__ == "__main__":
    # state()
    convergence()
