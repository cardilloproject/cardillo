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

        # self.q0 = np.array([0, 1.25])

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
x0 = -2
# x0 = 0
# x0 = 3
# x_dot0 = 0
x_dot0 = 1
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
    t_final = 3
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
    tol = 1.0e-12

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

    # dt_ref = 1e-4
    # dts = (2.0 ** np.arange(9, 1, -1)) * dt_ref  # [5.12e-2, ..., 8e-4]
    # t1 = (2.0**15) * dt_ref  # 3.2768s

    print(f"t1: {t1}")
    print(f"dts: {dts}")
    # exit()

    ################
    # chose a solver
    ################
    # get_solver = lambda dt: MoreauClassical(model, t1, dt, atol=tol)
    get_solver = lambda dt: Rattle(model, t1, dt, atol=tol)
    # get_solver = lambda dt: NonsmoothGeneralizedAlpha(model, t1, dt, newton_tol=tol)
    # get_solver = lambda dt: NonsmoothPIRK(model, t1, dt, RadauIIATableau(2), atol=tol)
    # get_solver = lambda dt: NonsmoothPIRK(model, t1, dt, RadauIIATableau(3), atol=tol)

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
    print(f"compute reference solution:")
    reference = get_solver(dt_ref).solve()
    print(f"done")

    def errors(sol, sol_ref, t_transient=1.5, t_longterm=2.5):
        # extract solution an remove first time step
        t = sol.t  # [1:]
        q = sol.q  # [1:]
        u = sol.u  # [1:]
        P_N = sol.P_N  # [1:]
        P_F = sol.P_F  # [1:]

        t_ref = sol_ref.t  # [1:]
        q_ref = sol_ref.q  # [1:]
        u_ref = sol_ref.u  # [1:]
        P_N_ref = sol_ref.P_N  # [1:]
        P_F_ref = sol_ref.P_F  # [1:]

        # distinguish between transient and long term time steps
        t_idx_transient = np.where(t <= t_transient)[0]
        t_idx_longterm = np.where(t >= t_longterm)[0]

        # compute difference between computed solution and reference solution
        # for identical time instants
        t_ref_idx = np.where(np.abs(t[:, None] - t_ref) < 1.0e-8)[1]
        t_ref_idx_transient = np.where(
            np.abs(t[t_idx_transient, None] - t_ref) < 1.0e-8
        )[1]
        t_ref_idx_longterm = np.where(np.abs(t[t_idx_longterm, None] - t_ref) < 1.0e-8)[
            1
        ]

        def max_relative_error(f, f_ref):
            diff = f - f_ref
            return np.max(np.linalg.norm(diff, axis=1) / np.linalg.norm(f_ref, axis=1))

        # def l1_error(t, t_ref, f, f_ref):
        #     """https://de.wikipedia.org/wiki/Summennorm"""
        #     dt = t[1] - t[0]
        #     return np.sum(dt * np.abs(f - f_ref[t_ref_idx]))

        def max_lp_error(t, t_ref, f, f_ref, p=1):
            """See https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
            and https://de.wikipedia.org/wiki/Lp-Raum."""
            dt = t[1] - t[0]
            # return np.max(np.sum(dt * np.abs(f - f_ref[t_ref_idx]) ** p, axis=0) ** (1 / p))
            return np.max(np.linalg.norm(dt * (f - f_ref[t_ref_idx]), ord=p, axis=0))

        def distance_function(x, y):
            # return np.linalg.norm(x[1:] - y[1:], ord=2)
            # return max(np.abs(x[0] - y[0]), np.linalg.norm(x[1:] - y[1:], ord=2))
            return max(np.abs(x[0] - y[0]), norm(x[1:] - y[1:]))

        def hausdorff_distance_error(t, t_ref, f, f_ref):
            """See https://en.wikipedia.org/wiki/Hausdorff_distance,
            https://github.com/mavillan/py-hausdorff/blob/master/hausdorff/hausdorff.py,
            https://github.com/scipy/scipy/blob/v1.10.1/scipy/spatial/_hausdorff.pyx."""
            # dt = t[1] - t[0]
            # X = np.hstack((t[:, None], dt * f))
            # dt_ref = t_ref[1] - t_ref[0]
            # Y = np.hstack((t_ref[:, None], dt_ref * f_ref))
            # X = np.hstack((t[:, None], f))
            # Y = np.hstack((t_ref[:, None], f_ref))
            Y = np.hstack((t[:, None], f))
            X = np.hstack((t_ref[:, None], f_ref))

            # def supremal_distance(A, B):
            #     cmax = 0.0
            #     for a in A:
            #         cmin = np.inf
            #         for b in B:
            #             d = distance_function(a, b)
            #             if d < cmin:
            #                 cmin = d
            #             if cmin < cmax:
            #                 break
            #         if cmin > cmax and np.inf > cmin:
            #             cmax = cmin
            #     return cmax

            def supremal_distance(A, B):
                """See https://github.com/scipy/scipy/blob/v1.10.1/scipy/spatial/_hausdorff.pyx."""
                cmax = 0.0
                for a in A:
                    cmin = np.inf
                    for b in B:
                        d = distance_function(a, b)
                        if d < cmax:  # break out of `for j` loop
                            break

                        if d < cmin:  # always true on first iteration of for-j loop
                            cmin = d

                    # Note: The reference paper by A. A. Taha and A. Hanbury has this line
                    # (Algorithm 2, line 16) as:
                    #
                    # if cmin > cmax:
                    #
                    # That logic is incorrect, as cmin could still be np.inf if breaking early.
                    # The logic here accounts for that case.
                    if cmin >= cmax and d >= cmax:
                        cmax = cmin

                return cmax

            return max(supremal_distance(X, Y), supremal_distance(Y, X))

        # # TODO: Test this for different norms
        # # def matrix_norm_error(t, t_ref, f, f_ref, p=1):
        # def matrix_norm_error(t, t_ref, f, f_ref, p=3):
        #     """https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html"""
        #     dt = t[1] - t[0]
        #     return np.linalg.norm(dt * (f - f_ref[t_ref_idx]), ord=p)

        # def trapezoid_error(t, t_ref, f, f_ref):
        #     from scipy.integrate import trapezoid

        #     F = trapezoid(f, t, axis=0)
        #     F_ref = trapezoid(f_ref, t_ref, axis=0)
        #     return np.max(np.abs(F - F_ref))

        # def p_var_error(t, t_ref, f, f_ref, p=1):
        #     """https://en.wikipedia.org/wiki/P-variation"""

        #     def p_var(x, p):
        #         cum_p_var = np.zeros_like(x)
        #         for i in range(1, len(x)):
        #             for j in range(i):
        #                 cum_p_var[i] = np.maximum(
        #                     cum_p_var[i], cum_p_var[j] + np.abs(x[i] - x[j]) ** p
        #                 )
        #         return cum_p_var

        #     cum_p_var = p_var(f, p)
        #     cum_p_var_ref = p_var(f_ref, p)
        #     return np.max(np.abs(cum_p_var[-1] - cum_p_var_ref[-1]))

        integral_error = max_lp_error
        # integral_error = matrix_norm_error
        # integral_error = hausdorff_distance_error

        # integral_error = l1_error
        # integral_error = trapezoid_error
        # integral_error = p_var_error

        # def integral_error(f, f_ref):
        #     # diff = f - f_ref

        #     # # # diff2 = np.linalg.norm(diff, axis=1) ** 2
        #     # # diff2 = np.array([d @ d for d in diff])

        #     # # # integral = np.trapz(diff2, t, dx=dt, axis=0)
        #     # # # TODO: why dt not dt**2?
        #     # # integral = dt * np.sum(diff2)
        #     # # error = np.sqrt(integral)

        #     # # integral = np.sum(diff2)
        #     # integral = np.sum(diff)
        #     # error = dt * np.sqrt(integral)
        #     # return error

        #     return np.sum(dt * (f - f_ref))

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
        # q_error_integral = integral_error(t, t_ref[t_ref_idx], q, q_ref[t_ref_idx])
        # u_error_integral = integral_error(t, t_ref[t_ref_idx], u, u_ref[t_ref_idx])
        # P_N_error_integral = integral_error(t, t_ref[t_ref_idx], P_N, P_N_ref[t_ref_idx])
        # P_F_error_integral = integral_error(t, t_ref[t_ref_idx], P_F, P_F_ref[t_ref_idx])
        q_error_integral = integral_error(t, t_ref, q, q_ref)
        u_error_integral = integral_error(t, t_ref, u, u_ref)
        P_N_error_integral = integral_error(t, t_ref, P_N, P_N_ref)
        P_F_error_integral = integral_error(t, t_ref, P_F, P_F_ref)

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

    for i, dt in enumerate(dts):
        print(f"i: {i}, dt: {dt:1.1e}")

        # solve problem with current time step
        sol = get_solver(dt).solve()
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
    # state()
    convergence()
