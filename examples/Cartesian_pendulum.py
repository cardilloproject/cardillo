import numpy as np
import matplotlib.pyplot as plt

from cardillo.model import Model
from cardillo.solver import (
    GeneralizedAlphaFirstOrder,
    GenAlphaFirstOrderGGL2_V3,
    HalfExplicitEuler,
)


class MathematicalPendulumCartesian:
    """Mathematical pendulum in Cartesian coordinates and with bilateral
    constraint, see Hairer1996 p. 464 - Example 2.

    References
    ----------
    Hairer1996: https://link.springer.com/book/10.1007/978-3-642-05221-7
    """

    def __init__(self, m, l, grav, q0=None, u0=None, la_g0=None):
        self.m = m
        self.l = l
        self.grav = grav

        self.nq = 2
        self.nu = 2
        self.nla_g = 1
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

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
        return np.array([0, -self.m * self.grav])

    def f_pot_q(self, t, q, coo):
        pass

    def g(self, t, q):
        x, y = q
        return np.array([x * x + y * y - self.l * self.l])

    def g_dot(self, t, q, u):
        x, y = q
        u_x, u_y = u
        return np.array([2 * x * u_x + 2 * y * u_y])

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

    def g_dot_q(self, t, q, u, coo):
        dense = np.atleast_2d(2.0 * u)
        coo.extend(dense, (self.la_gDOF, self.qDOF))

    def W_g_dense(self, t, q):
        return self.g_q_dense(t, q).T

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        coo.extend(np.eye(self.nu, self.nq) * 2 * la_g[0], (self.uDOF, self.qDOF))

    def G(self, t, q):
        W = self.W_g_dense(t, q)
        M = self.M_dense(t, q)
        # G1 = W.T @ np.linalg.inv(M) @ W
        G2 = W.T @ np.linalg.solve(M, W)
        # error = np.linalg.norm(G1 - G2)
        # print(f"error G: {error}")
        return G2

    def la_g(self, t, q, u):
        W = self.W_g_dense(t, q)
        M = self.M_dense(t, q)
        G = np.array([[W.T @ np.linalg.solve(M, W)]])
        zeta = self.g_ddot(t, q, u, np.zeros_like(u))
        h = self.f_pot(t, q)
        eta = zeta + W.T @ np.linalg.solve(M, h)
        return np.linalg.solve(G, -eta)


if __name__ == "__main__":
    # system parameters
    m = 1
    l = 1
    g = 10

    # initial state
    phi0 = 0
    phi_dot0 = 0
    x0, y0 = l * np.array([np.cos(phi0), np.sin(phi0)])
    x_dot0, y_dot0 = l * np.array([-np.sin(phi0), np.cos(phi0)]) * phi_dot0
    q0 = np.array([x0, y0])
    u0 = np.array([x_dot0, y_dot0])

    # system definition and assemble the model
    pendulum = MathematicalPendulumCartesian(m, l, g, q0, u0)
    model = Model()
    model.add(pendulum)
    model.assemble()

    # end time and numerical dissipation of generalized-alpha solver
    # t1 = 0.1
    # t1 = 1.0
    t1 = 5.0

    h = 1.0e-2
    atol = 1.0e-8

    sol = HalfExplicitEuler(model, t1, h, atol).solve()

    # use GGL results for visualization
    t = sol.t
    q = sol.q
    u = sol.u
    la_g = sol.la_g

    ######################
    # visualize q, u, la_g
    ######################
    fig, ax = plt.subplots(1, 3)

    # generalized coordinates
    ax[0].plot(t, q[:, 0], "xg", label="x")
    ax[0].plot(
        t,
        q[:, 1],
        "og",
        label="y",
    )
    ax[0].grid()
    ax[0].legend()

    # generalized velocities
    ax[1].plot(
        t,
        u[:, 0],
        "xg",
        label="x_dot",
    )
    ax[1].plot(
        t,
        u[:, 1],
        "og",
        label="y_dot",
    )
    ax[1].grid()
    ax[1].legend()

    # Lagrange multipliers
    ax[2].plot(t, la_g[:, 0], "sg", label="la_g")
    ax[2].grid()
    ax[2].legend()

    plt.show()

    exit()

    # log spaced time steps
    # num = 3
    # num = 4 # num = 4 yields problems with Lagrange multipliers
    # dts = np.logspace(-1, -num, num=num, endpoint=True)
    dts = np.array([1.0e-1, 1.0e-2])
    # dts = np.array([1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4])
    dts_1 = dts
    dts_2 = dts**2
    print(f"dts: {dts}")

    # errors for all 6 possible solvers
    q_errors = np.inf * np.ones((6, len(dts)), dtype=float)
    u_errors = np.inf * np.ones((6, len(dts)), dtype=float)
    la_g_errors = np.inf * np.ones((6, len(dts)), dtype=float)

    # compute reference solution as described in Arnold2015 Section 3.3
    print(f"compute reference solution:")
    dt_ref = 1.0e-5
    # dt_ref = 1.0e-3
    reference = GeneralizedAlphaFirstOrder(model, t1, dt_ref, rho_inf=rho_inf).solve()
    t_ref = reference.t
    q_ref = reference.q
    u_ref = reference.u
    la_g_ref = reference.la_g

    def errors(sol):
        t = sol.t
        q = sol.q
        u = sol.u
        la_g = sol.la_g

        # compute difference between computed solution and reference solution
        # for identical time instants
        idx = np.where(np.abs(t[:, None] - t_ref) < 1.0e-8)[1]

        # differences
        diff_q = q - q_ref[idx]
        diff_u = u - u_ref[idx]
        diff_la_g = la_g - la_g_ref[idx]

        # relative error
        q_error = np.linalg.norm(diff_q) / np.linalg.norm(q)
        u_error = np.linalg.norm(diff_u) / np.linalg.norm(u)
        la_g_error = np.linalg.norm(diff_la_g) / np.linalg.norm(la_g)

        return q_error, u_error, la_g_error

    for i, dt in enumerate(dts):
        print(f"i: {i}, dt: {dt:1.1e}")

        # position formulation
        sol = GeneralizedAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, unknowns="positions"
        ).solve()
        q_errors[0, i], u_errors[0, i], la_g_errors[0, i] = errors(sol)

        # velocity formulation
        sol = GeneralizedAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, unknowns="velocities"
        ).solve()
        q_errors[1, i], u_errors[1, i], la_g_errors[1, i] = errors(sol)

        # auxiliary formulation
        sol = GeneralizedAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, unknowns="auxiliary"
        ).solve()
        q_errors[2, i], u_errors[2, i], la_g_errors[2, i] = errors(sol)

        # GGL formulation - positions
        sol = GeneralizedAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, unknowns="positions", GGL=True
        ).solve()
        q_errors[3, i], u_errors[3, i], la_g_errors[3, i] = errors(sol)

        # GGL formulation - velocityies
        sol = GeneralizedAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, unknowns="velocities", GGL=True
        ).solve()
        q_errors[4, i], u_errors[4, i], la_g_errors[4, i] = errors(sol)

        # GGL formulation - auxiliary
        sol = GeneralizedAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, unknowns="auxiliary", GGL=True
        ).solve()
        q_errors[5, i], u_errors[5, i], la_g_errors[5, i] = errors(sol)

    # # names = ["GenAlphaFirstOrder", "GenAlphaSecondOrder", "GenAlphaFirstOrderGGl", "Theta"]
    # # ts = [t_GenAlphaFirstOrder, t_GenAlphaSecondOrder, t_GenAlphaFirstOrderGGl, t_ThetaNewton]
    # # qs = [q_GenAlphaFirstOrder, q_GenAlphaSecondOrder, q_GenAlphaFirstOrderGGl, q_ThetaNewton]
    # # us = [u_GenAlphaFirstOrder, u_GenAlphaSecondOrder, u_GenAlphaFirstOrderGGl, u_ThetaNewton]
    # # la_gs = [la_g_GenAlphaFirstOrder, la_g_GenAlphaSecondOrder, la_g_GenAlphaFirstOrderGGl, la_g_ThetaNewton]
    # names = ["GenAlphaFirstOrderGGl"]
    # ts = [t_GenAlphaFirstOrderGGl]
    # qs = [q_GenAlphaFirstOrderGGl]
    # us = [u_GenAlphaFirstOrderGGl]
    # la_gs = [la_g_GenAlphaFirstOrderGGl]
    # for i, name in enumerate(names):
    #     filename = "sol_" + name + "_cartesian_pendulum_.txt"
    #     # export_data = np.hstack((t_GenAlphaFirstOrderGGl[:, np.newaxis], q_GenAlphaFirstOrderGGl, u_GenAlphaFirstOrderGGl, la_g_GenAlphaFirstOrderGGl))
    #     export_data = np.hstack((ts[i][:, np.newaxis], qs[i], us[i], la_gs[i]))
    #     header = "t, x, y, x_dot, y_dot, la_g"
    #     np.savetxt(filename, export_data, delimiter=", ", header=header, comments="")

    #     filename = "error_" + name + "_cartesian_pendulum_.txt"
    #     header = "dt, dt2, error_xy, error_xy_dot, error_la_g"
    #     export_data = np.vstack((dts, dts_2, x_y_errors[i], x_dot_y_dot_errors[i], la_g_errors[i])).T
    #     np.savetxt(filename, export_data, delimiter=", ", header=header, comments="")

    ##################
    # visualize errors
    ##################
    fig, ax = plt.subplots(2, 3)

    # errors position formulation
    ax[0, 0].loglog(dts, q_errors[0], "-.r", label="q")
    ax[0, 0].loglog(dts, u_errors[0], "-.g", label="u")
    ax[0, 0].loglog(dts, la_g_errors[0], "-.b", label="la_g")
    ax[0, 0].loglog(dts, dts_1, "-k", label="dt")
    ax[0, 0].loglog(dts, dts_2, "--k", label="dt^2")
    ax[0, 0].set_title("position formulation")
    ax[0, 0].grid()
    ax[0, 0].legend()

    # errors velocity formulation
    ax[0, 1].loglog(dts, q_errors[1], "-.r", label="q")
    ax[0, 1].loglog(dts, u_errors[1], "-.g", label="u")
    ax[0, 1].loglog(dts, la_g_errors[1], "-.b", label="la_g")
    ax[0, 1].loglog(dts, dts_1, "-k", label="dt")
    ax[0, 1].loglog(dts, dts_2, "--k", label="dt^2")
    ax[0, 1].set_title("velocity formulation")
    ax[0, 1].grid()
    ax[0, 1].legend()

    # errors auxiliary formulation
    ax[0, 2].loglog(dts, q_errors[2], "-.r", label="q")
    ax[0, 2].loglog(dts, u_errors[2], "-.g", label="u")
    ax[0, 2].loglog(dts, la_g_errors[2], "-.b", label="la_g")
    ax[0, 2].loglog(dts, dts_1, "-k", label="dt")
    ax[0, 2].loglog(dts, dts_2, "--k", label="dt^2")
    ax[0, 2].set_title("auxiliary formulation")
    ax[0, 2].grid()
    ax[0, 2].legend()

    # errors position formulation
    ax[1, 0].loglog(dts, q_errors[3], "-.r", label="q")
    ax[1, 0].loglog(dts, u_errors[3], "-.g", label="u")
    ax[1, 0].loglog(dts, la_g_errors[3], "-.b", label="la_g")
    ax[1, 0].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 0].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 0].set_title("position formulation GGL")
    ax[1, 0].grid()
    ax[1, 0].legend()

    # errors velocity formulation
    ax[1, 1].loglog(dts, q_errors[4], "-.r", label="q")
    ax[1, 1].loglog(dts, u_errors[4], "-.g", label="u")
    ax[1, 1].loglog(dts, la_g_errors[4], "-.b", label="la_g")
    ax[1, 1].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 1].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 1].set_title("velocity formulation GGL")
    ax[1, 1].grid()
    ax[1, 1].legend()

    # errors auxiliary formulation
    ax[1, 2].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 2].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 2].loglog(dts, q_errors[5], "-.r", label="q")
    ax[1, 2].loglog(dts, u_errors[5], "-.g", label="u")
    ax[1, 2].loglog(dts, la_g_errors[5], "-.b", label="la_g")
    ax[1, 2].set_title("auxiliary formulation GGL")
    ax[1, 2].grid()
    ax[1, 2].legend()

    plt.show()


# if __name__ == "__main__":
#     # system parameters
#     m = 1
#     l = 1
#     g = 10

#     # initial state
#     phi0 = 0
#     phi_dot0 = 0
#     x0, y0 = l * np.array([np.cos(phi0), np.sin(phi0)])
#     x_dot0, y_dot0 = l * np.array([-np.sin(phi0), np.cos(phi0)]) * phi_dot0
#     q0 = np.array([x0, y0])
#     u0 = np.array([x_dot0, y_dot0])

#     # system definition and assemble the model
#     pendulum = MathematicalPendulumCartesian(m, l, g, q0, u0)
#     model = Model()
#     model.add(pendulum)
#     model.assemble()

#     # end time and numerical dissipation of generalized-alpha solver
#     t1 = 0.01
#     # t1 = 1.0
#     rho_inf = 0.85  # numerical damping is required to reduce oszillations of the Lagrange multipliers

#     # log spaced time steps
#     # num = 3
#     # num = 4 # num = 4 yields problems with Lagrange multipliers
#     # dts = np.logspace(-1, -num, num=num, endpoint=True)
#     dts = np.array([1.0e-1, 1.0e-2])
#     # dts = np.array([1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4])
#     dts_1 = dts
#     dts_2 = dts**2
#     print(f"dts: {dts}")

#     # errors for all 6 possible solvers
#     q_errors = np.inf * np.ones((6, len(dts)), dtype=float)
#     u_errors = np.inf * np.ones((6, len(dts)), dtype=float)
#     la_g_errors = np.inf * np.ones((6, len(dts)), dtype=float)

#     # compute reference solution as described in Arnold2015 Section 3.3
#     print(f"compute reference solution:")
#     dt_ref = 1.0e-5
#     # dt_ref = 1.0e-3
#     reference = GeneralizedAlphaFirstOrder(model, t1, dt_ref, rho_inf=rho_inf).solve()
#     t_ref = reference.t
#     q_ref = reference.q
#     u_ref = reference.u
#     la_g_ref = reference.la_g

#     def errors(sol):
#         t = sol.t
#         q = sol.q
#         u = sol.u
#         la_g = sol.la_g

#         # compute difference between computed solution and reference solution
#         # for identical time instants
#         idx = np.where(np.abs(t[:, None] - t_ref) < 1.0e-8)[1]

#         # differences
#         diff_q = q - q_ref[idx]
#         diff_u = u - u_ref[idx]
#         diff_la_g = la_g - la_g_ref[idx]

#         # relative error
#         q_error = np.linalg.norm(diff_q) / np.linalg.norm(q)
#         u_error = np.linalg.norm(diff_u) / np.linalg.norm(u)
#         la_g_error = np.linalg.norm(diff_la_g) / np.linalg.norm(la_g)

#         return q_error, u_error, la_g_error

#     for i, dt in enumerate(dts):
#         print(f"i: {i}, dt: {dt:1.1e}")

#         # position formulation
#         sol = GeneralizedAlphaFirstOrder(
#             model, t1, dt, rho_inf=rho_inf, unknowns="positions"
#         ).solve()
#         q_errors[0, i], u_errors[0, i], la_g_errors[0, i] = errors(sol)

#         # velocity formulation
#         sol = GeneralizedAlphaFirstOrder(
#             model, t1, dt, rho_inf=rho_inf, unknowns="velocities"
#         ).solve()
#         q_errors[1, i], u_errors[1, i], la_g_errors[1, i] = errors(sol)

#         # auxiliary formulation
#         sol = GeneralizedAlphaFirstOrder(
#             model, t1, dt, rho_inf=rho_inf, unknowns="auxiliary"
#         ).solve()
#         q_errors[2, i], u_errors[2, i], la_g_errors[2, i] = errors(sol)

#         # GGL formulation - positions
#         sol = GeneralizedAlphaFirstOrder(
#             model, t1, dt, rho_inf=rho_inf, unknowns="positions", GGL=True
#         ).solve()
#         q_errors[3, i], u_errors[3, i], la_g_errors[3, i] = errors(sol)

#         # GGL formulation - velocityies
#         sol = GeneralizedAlphaFirstOrder(
#             model, t1, dt, rho_inf=rho_inf, unknowns="velocities", GGL=True
#         ).solve()
#         q_errors[4, i], u_errors[4, i], la_g_errors[4, i] = errors(sol)

#         # GGL formulation - auxiliary
#         sol = GeneralizedAlphaFirstOrder(
#             model, t1, dt, rho_inf=rho_inf, unknowns="auxiliary", GGL=True
#         ).solve()
#         q_errors[5, i], u_errors[5, i], la_g_errors[5, i] = errors(sol)

#     # # names = ["GenAlphaFirstOrder", "GenAlphaSecondOrder", "GenAlphaFirstOrderGGl", "Theta"]
#     # # ts = [t_GenAlphaFirstOrder, t_GenAlphaSecondOrder, t_GenAlphaFirstOrderGGl, t_ThetaNewton]
#     # # qs = [q_GenAlphaFirstOrder, q_GenAlphaSecondOrder, q_GenAlphaFirstOrderGGl, q_ThetaNewton]
#     # # us = [u_GenAlphaFirstOrder, u_GenAlphaSecondOrder, u_GenAlphaFirstOrderGGl, u_ThetaNewton]
#     # # la_gs = [la_g_GenAlphaFirstOrder, la_g_GenAlphaSecondOrder, la_g_GenAlphaFirstOrderGGl, la_g_ThetaNewton]
#     # names = ["GenAlphaFirstOrderGGl"]
#     # ts = [t_GenAlphaFirstOrderGGl]
#     # qs = [q_GenAlphaFirstOrderGGl]
#     # us = [u_GenAlphaFirstOrderGGl]
#     # la_gs = [la_g_GenAlphaFirstOrderGGl]
#     # for i, name in enumerate(names):
#     #     filename = "sol_" + name + "_cartesian_pendulum_.txt"
#     #     # export_data = np.hstack((t_GenAlphaFirstOrderGGl[:, np.newaxis], q_GenAlphaFirstOrderGGl, u_GenAlphaFirstOrderGGl, la_g_GenAlphaFirstOrderGGl))
#     #     export_data = np.hstack((ts[i][:, np.newaxis], qs[i], us[i], la_gs[i]))
#     #     header = "t, x, y, x_dot, y_dot, la_g"
#     #     np.savetxt(filename, export_data, delimiter=", ", header=header, comments="")

#     #     filename = "error_" + name + "_cartesian_pendulum_.txt"
#     #     header = "dt, dt2, error_xy, error_xy_dot, error_la_g"
#     #     export_data = np.vstack((dts, dts_2, x_y_errors[i], x_dot_y_dot_errors[i], la_g_errors[i])).T
#     #     np.savetxt(filename, export_data, delimiter=", ", header=header, comments="")

#     # plot_state = True
#     plot_state = False
#     if plot_state:
#         # use GGL results for visualization
#         t = sol.t
#         q = sol.q
#         u = sol.u
#         la_g = sol.la_g

#         ######################
#         # visualize q, u, la_g
#         ######################
#         fig, ax = plt.subplots(1, 3)

#         # generalized coordinates
#         ax[0].plot(
#             t,
#             q[:, 0],
#             "xg",
#             label="x - GGL auxiliary",
#         )
#         ax[0].plot(
#             t,
#             q[:, 1],
#             "og",
#             label="y - GGL auxiliary",
#         )
#         ax[0].plot(t_ref, q_ref[:, 0], "-k", label="x - ref")
#         ax[0].plot(t_ref, q_ref[:, 1], "--k", label="y - ref")
#         ax[0].grid()
#         ax[0].legend()

#         # generalized velocities
#         ax[1].plot(
#             t,
#             u[:, 0],
#             "xg",
#             label="x_dot - GGL auxiliary",
#         )
#         ax[1].plot(
#             t,
#             u[:, 1],
#             "og",
#             label="y_dot - GGL auxiliary",
#         )
#         ax[1].plot(t_ref, u_ref[:, 0], "-k", label="x_dot - ref")
#         ax[1].plot(t_ref, u_ref[:, 1], "--k", label="y_dot - ref")
#         ax[1].grid()
#         ax[1].legend()

#         # Lagrange multipliers
#         ax[2].plot(
#             t,
#             la_g[:, 0],
#             "sg",
#             label="la_g - GGL auxiliary",
#         )
#         ax[2].plot(t_ref, la_g_ref[:, 0], "-k", label="la_g - ref")
#         ax[2].grid()
#         ax[2].legend()

#     ##################
#     # visualize errors
#     ##################
#     fig, ax = plt.subplots(2, 3)

#     # errors position formulation
#     ax[0, 0].loglog(dts, q_errors[0], "-.r", label="q")
#     ax[0, 0].loglog(dts, u_errors[0], "-.g", label="u")
#     ax[0, 0].loglog(dts, la_g_errors[0], "-.b", label="la_g")
#     ax[0, 0].loglog(dts, dts_1, "-k", label="dt")
#     ax[0, 0].loglog(dts, dts_2, "--k", label="dt^2")
#     ax[0, 0].set_title("position formulation")
#     ax[0, 0].grid()
#     ax[0, 0].legend()

#     # errors velocity formulation
#     ax[0, 1].loglog(dts, q_errors[1], "-.r", label="q")
#     ax[0, 1].loglog(dts, u_errors[1], "-.g", label="u")
#     ax[0, 1].loglog(dts, la_g_errors[1], "-.b", label="la_g")
#     ax[0, 1].loglog(dts, dts_1, "-k", label="dt")
#     ax[0, 1].loglog(dts, dts_2, "--k", label="dt^2")
#     ax[0, 1].set_title("velocity formulation")
#     ax[0, 1].grid()
#     ax[0, 1].legend()

#     # errors auxiliary formulation
#     ax[0, 2].loglog(dts, q_errors[2], "-.r", label="q")
#     ax[0, 2].loglog(dts, u_errors[2], "-.g", label="u")
#     ax[0, 2].loglog(dts, la_g_errors[2], "-.b", label="la_g")
#     ax[0, 2].loglog(dts, dts_1, "-k", label="dt")
#     ax[0, 2].loglog(dts, dts_2, "--k", label="dt^2")
#     ax[0, 2].set_title("auxiliary formulation")
#     ax[0, 2].grid()
#     ax[0, 2].legend()

#     # errors position formulation
#     ax[1, 0].loglog(dts, q_errors[3], "-.r", label="q")
#     ax[1, 0].loglog(dts, u_errors[3], "-.g", label="u")
#     ax[1, 0].loglog(dts, la_g_errors[3], "-.b", label="la_g")
#     ax[1, 0].loglog(dts, dts_1, "-k", label="dt")
#     ax[1, 0].loglog(dts, dts_2, "--k", label="dt^2")
#     ax[1, 0].set_title("position formulation GGL")
#     ax[1, 0].grid()
#     ax[1, 0].legend()

#     # errors velocity formulation
#     ax[1, 1].loglog(dts, q_errors[4], "-.r", label="q")
#     ax[1, 1].loglog(dts, u_errors[4], "-.g", label="u")
#     ax[1, 1].loglog(dts, la_g_errors[4], "-.b", label="la_g")
#     ax[1, 1].loglog(dts, dts_1, "-k", label="dt")
#     ax[1, 1].loglog(dts, dts_2, "--k", label="dt^2")
#     ax[1, 1].set_title("velocity formulation GGL")
#     ax[1, 1].grid()
#     ax[1, 1].legend()

#     # errors auxiliary formulation
#     ax[1, 2].loglog(dts, dts_1, "-k", label="dt")
#     ax[1, 2].loglog(dts, dts_2, "--k", label="dt^2")
#     ax[1, 2].loglog(dts, q_errors[5], "-.r", label="q")
#     ax[1, 2].loglog(dts, u_errors[5], "-.g", label="u")
#     ax[1, 2].loglog(dts, la_g_errors[5], "-.b", label="la_g")
#     ax[1, 2].set_title("auxiliary formulation GGL")
#     ax[1, 2].grid()
#     ax[1, 2].legend()

#     plt.show()
