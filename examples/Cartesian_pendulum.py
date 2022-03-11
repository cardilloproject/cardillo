import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from cardillo.model import Model
from cardillo.solver import ScipyIVP, EulerBackward, GenAlphaFirstOrderVelocityGGL


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

    def cartesian_coordinates(phi):
        return l * np.array([np.cos(phi), np.sin(phi)])

    def cartesian_coordinates_dot(phi, phi_dot):
        return l * np.array([-np.sin(phi), np.cos(phi)]) * phi_dot

    def la_g_analytic(t, phi, phi_dot):
        x, y = cartesian_coordinates(phi)
        x_dot, y_dot = cartesian_coordinates_dot(phi, phi_dot)

        pendulum = MathematicalPendulumCartesian(m, l, g)
        q = np.array([x, y])
        u = np.array([x_dot, y_dot])
        return pendulum.la_g(t, q, u)

    # rhs of ODE formulation
    def eqm(t, x):
        phi = x[0]
        u = x[1]
        return np.array([u, -np.cos(phi) * g / l])

    # initial state
    phi0 = 0
    phi_dot0 = 0
    x0, y0 = cartesian_coordinates(phi0)
    x_dot0, y_dot0 = cartesian_coordinates_dot(phi0, phi_dot0)
    q0 = np.array([x0, y0])
    u0 = np.array([x_dot0, y_dot0])

    # system definition and assemble the model
    pendulum = MathematicalPendulumCartesian(m, l, g, q0, u0)
    model = Model()
    model.add(pendulum)
    model.assemble()

    # end time and numerical dissipation of generalized-alpha solver
    t1 = 1
    # rho_inf = 0.85
    rho_inf = 0.95

    # log spaced time steps
    num = 3
    dts = np.logspace(-1, -num, num=num, endpoint=True)
    dts_1 = dts
    dts_2 = dts**2
    print(f"dts: {dts}")

    # TODO: Compare error with theta method
    x_y_errors = np.inf * np.ones((4, len(dts)), dtype=float)
    x_dot_y_dot_errors = np.inf * np.ones((4, len(dts)), dtype=float)
    la_g_errors = np.inf * np.ones((4, len(dts)), dtype=float)

    for i, dt in enumerate(dts):
        print(f"i: {i}, dt: {dt:1.1e}")

        # # solve with theta-method
        # # sol_ThetaNewton = ThetaNewton(model, t1, dt).solve()
        # sol_ThetaNewton = MoreauTheta(model, t1, dt).solve()
        # t_ThetaNewton = sol_ThetaNewton.t
        # q_ThetaNewton = sol_ThetaNewton.q
        # u_ThetaNewton = sol_ThetaNewton.u
        # la_g_ThetaNewton = sol_ThetaNewton.la_g

        # # solve with first order generalized-alpha method
        # sol_GenAlphaFirstOrder = GenAlphaFirstOrderVelocity(model, t1, dt, rho_inf=rho_inf).solve()
        # t_GenAlphaFirstOrder = sol_GenAlphaFirstOrder.t
        # q_GenAlphaFirstOrder = sol_GenAlphaFirstOrder.q
        # u_GenAlphaFirstOrder = sol_GenAlphaFirstOrder.u
        # la_g_GenAlphaFirstOrder = sol_GenAlphaFirstOrder.la_g

        # # solve with second order generalized-alpha method velocity implementation
        # sol_GenAlphaSecondOrder = GenAlphaDAEAcc(model, t1, dt, rho_inf=rho_inf).solve()
        # t_GenAlphaSecondOrder = sol_GenAlphaSecondOrder.t
        # q_GenAlphaSecondOrder = sol_GenAlphaSecondOrder.q
        # u_GenAlphaSecondOrder = sol_GenAlphaSecondOrder.u
        # la_g_GenAlphaSecondOrder = sol_GenAlphaSecondOrder.la_g

        # solve with generalzed-alpha method positin implementation
        # sol_ThetaNewton = GenAlphaFirstOrderPosition(model, t1, dt, rho_inf=rho_inf).solve()
        sol_GenAlphaFirstOrderGGl = GenAlphaFirstOrderVelocityGGL(model, t1, dt, rho_inf=rho_inf).solve()
        # sol_GenAlphaFirstOrderGGl = ScipyIVP(model, t1, dt, method="RK45", atol=1.0e-10, rtol=1.0e-10).solve()
        # sol_GenAlphaFirstOrderGGl = EulerBackward(model, t1, dt, atol=1.0e-12).solve()
        t_GenAlphaFirstOrderGGl = sol_GenAlphaFirstOrderGGl.t
        q_GenAlphaFirstOrderGGl = sol_GenAlphaFirstOrderGGl.q
        u_GenAlphaFirstOrderGGl = sol_GenAlphaFirstOrderGGl.u
        la_g_GenAlphaFirstOrderGGl = sol_GenAlphaFirstOrderGGl.la_g

        # compute true solution using Runge-Kutta 4(5) method of the ODE formulation
        t_span = np.array([0, t1])
        y0 = np.array([phi0, phi_dot0])
        t_eval = np.linspace(0, t1, num=len(t_GenAlphaFirstOrderGGl))
        sol_RK45 = solve_ivp(
            eqm, t_span, y0, method="RK45", t_eval=t_eval, atol=1.0e-12, rtol=1.0e-12
        )
        t_RK45 = sol_RK45.t
        phi_RK45 = sol_RK45.y[0, :]
        omega_RK45 = sol_RK45.y[1, :]
        q_RK45 = cartesian_coordinates(phi_RK45).T
        u_RK45 = cartesian_coordinates_dot(phi_RK45, omega_RK45).T
        la_g_RK4 = np.array(
            [
                la_g_analytic(t, phi, omega)
                for (t, phi, omega) in zip(t_RK45, phi_RK45, omega_RK45)
            ]
        )

        # compute errors
        # x_y_errors[0, i] = np.linalg.norm(q_GenAlphaFirstOrder - q_RK45)
        # x_y_errors[1, i] = np.linalg.norm(q_GenAlphaSecondOrder - q_RK45)
        x_y_errors[2, i] = np.linalg.norm(q_GenAlphaFirstOrderGGl - q_RK45)
        # x_y_errors[3, i] = np.linalg.norm(q_ThetaNewton - q_RK45)
        # x_dot_y_dot_errors[0, i] = np.linalg.norm(u_GenAlphaFirstOrder - u_RK45)
        # x_dot_y_dot_errors[1, i] = np.linalg.norm(u_GenAlphaSecondOrder - u_RK45)
        x_dot_y_dot_errors[2, i] = np.linalg.norm(u_GenAlphaFirstOrderGGl - u_RK45)
        # x_dot_y_dot_errors[3, i] = np.linalg.norm(u_ThetaNewton - u_RK45)
        # la_g_errors[0, i] = np.linalg.norm(la_g_GenAlphaFirstOrder - la_g_RK4)
        # la_g_errors[1, i] = np.linalg.norm(la_g_GenAlphaSecondOrder - la_g_RK4)
        la_g_errors[2, i] = np.linalg.norm(la_g_GenAlphaFirstOrderGGl - la_g_RK4)
        # la_g_errors[3, i] = np.linalg.norm(la_g_ThetaNewton - la_g_RK4)

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

    # visualize results
    fig, ax = plt.subplots(2, 3)

    # generalized coordinates
    # ax[0, 0].plot(t_GenAlphaFirstOrder, q_GenAlphaFirstOrder[:, 0], 'xb', label="x - GenAlphaFirstOrder")
    # ax[0, 0].plot(t_GenAlphaFirstOrder, q_GenAlphaFirstOrder[:, 1], 'ob', label="y - GenAlphaFirstOrder")
    # ax[0, 0].plot(t_GenAlphaSecondOrder, q_GenAlphaSecondOrder[:, 0], 'xr', label="x - GenAlphaSecondOrder")
    # ax[0, 0].plot(t_GenAlphaSecondOrder, q_GenAlphaSecondOrder[:, 1], 'or', label="y - GenAlphaSecondOrder")
    ax[0, 0].plot(
        t_GenAlphaFirstOrderGGl,
        q_GenAlphaFirstOrderGGl[:, 0],
        "xg",
        label="x - GenAlphaFirstOrderGGl",
    )
    ax[0, 0].plot(
        t_GenAlphaFirstOrderGGl,
        q_GenAlphaFirstOrderGGl[:, 1],
        "og",
        label="y - GenAlphaFirstOrderGGl",
    )
    # ax[0, 0].plot(t_ThetaNewton, q_ThetaNewton[:, 0], 'xm', label="x - Theta")
    # ax[0, 0].plot(t_ThetaNewton, q_ThetaNewton[:, 1], 'om', label="y - Theta")
    ax[0, 0].plot(t_RK45, q_RK45[:, 0], "-k", label="x - RK45")
    ax[0, 0].plot(t_RK45, q_RK45[:, 1], "--k", label="y - RK45")
    ax[0, 0].grid()
    ax[0, 0].legend()

    # generalized velocities
    # ax[0, 1].plot(t_GenAlphaFirstOrder, u_GenAlphaFirstOrder[:, 0], 'xb', label="x_dot - GenAlphaFirstOrder")
    # ax[0, 1].plot(t_GenAlphaFirstOrder, u_GenAlphaFirstOrder[:, 1], 'ob', label="y_dot - GenAlphaFirstOrder")
    # ax[0, 1].plot(t_GenAlphaSecondOrder, u_GenAlphaSecondOrder[:, 0], 'xr', label="x_dot - GenAlphaSecondOrder")
    # ax[0, 1].plot(t_GenAlphaSecondOrder, u_GenAlphaSecondOrder[:, 1], 'or', label="y_dot - GenAlphaSecondOrder")
    ax[0, 1].plot(
        t_GenAlphaFirstOrderGGl,
        u_GenAlphaFirstOrderGGl[:, 0],
        "xg",
        label="x_dot - GenAlphaFirstOrderGGl",
    )
    ax[0, 1].plot(
        t_GenAlphaFirstOrderGGl,
        u_GenAlphaFirstOrderGGl[:, 1],
        "og",
        label="y_dot - GenAlphaFirstOrderGGl",
    )
    # ax[0, 1].plot(t_ThetaNewton, u_ThetaNewton[:, 0], 'xm', label="x_dot - Theta")
    # ax[0, 1].plot(t_ThetaNewton, u_ThetaNewton[:, 1], 'om', label="y_dot - Theta")
    ax[0, 1].plot(t_RK45, u_RK45[:, 0], "-k", label="x_dot - RK45")
    ax[0, 1].plot(t_RK45, u_RK45[:, 1], "--k", label="y_dot - RK45")
    ax[0, 1].grid()
    ax[0, 1].legend()

    # Lagrange multipliers
    # ax[0, 2].plot(t_GenAlphaFirstOrder, la_g_GenAlphaFirstOrder[:, 0], 'ob', label="la_g - GenAlphaFirstOrder")
    # ax[0, 2].plot(t_GenAlphaSecondOrder, la_g_GenAlphaSecondOrder[:, 0], 'xr', label="la_g - GenAlphaSecondOrder")
    # ax[0, 2].plot(t_GenAlphaFirstOrderGGl, la_g_GenAlphaFirstOrderGGl[:, 0], 'sg', label="la_g - GenAlphaFirstOrderGGl")
    ax[0, 2].plot(
        t_GenAlphaFirstOrderGGl,
        la_g_GenAlphaFirstOrderGGl[:, 0],
        "sg",
        label="la_g - GenAlphaFirstOrderGGl",
    )
    # ax[0, 2].plot(t_ThetaNewton, la_g_ThetaNewton[:, 0], 'sm', label="la_g - Theta")
    ax[0, 2].plot(t_RK45, la_g_RK4, "-k", label="la_g - RK45")
    ax[0, 2].grid()
    ax[0, 2].legend()

    # (x,y) - errors
    # ax[1, 0].loglog(dts, x_y_errors[0], '-b', label="(x,y) - error - GenAlphaFirstOrder")
    # ax[1, 0].loglog(dts, x_y_errors[1], '--r', label="(x,y) - error - GenAlphaSecondOrder")
    ax[1, 0].loglog(
        dts, x_y_errors[2], "-.g", label="(x,y) - error - GenAlphaFirstOrderGGl"
    )
    # ax[1, 0].loglog(dts, x_y_errors[3], '--m', label="(x,y) - error - Theta")
    ax[1, 0].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 0].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 0].grid()
    ax[1, 0].legend()

    # (x_dot,y_dot) - errors
    # ax[1, 1].loglog(dts, x_dot_y_dot_errors[0], '-b', label="(x_dot,y_dot) - error - GenAlphaFirstOrder")
    # ax[1, 1].loglog(dts, x_dot_y_dot_errors[1], '--r', label="(x_dot,y_dot) - error - GenAlphaSecondOrder")
    ax[1, 1].loglog(
        dts,
        x_dot_y_dot_errors[2],
        "-.g",
        label="(x_dot,y_dot) - error - GenAlphaFirstOrderGGl",
    )
    # ax[1, 1].loglog(dts, x_dot_y_dot_errors[3], '--m', label="(x_dot,y_dot) - error - Theta")
    ax[1, 1].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 1].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 1].grid()
    ax[1, 1].legend()

    # TODO: la_g - errors
    # ax[1, 2].loglog(dts, la_g_errors[0], '-b', label="la_g - error - GenAlphaFirstOrder")
    # ax[1, 2].loglog(dts, la_g_errors[1], '--r', label="la_g - error - GenAlphaSecondOrder")
    ax[1, 2].loglog(
        dts, la_g_errors[2], "-.g", label="la_g - error - GenAlphaFirstOrderGGl"
    )
    # ax[1, 2].loglog(dts, la_g_errors[3], '-.m', label="la_g - error - Theta")
    ax[1, 2].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 2].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 2].grid()
    ax[1, 2].legend()

    plt.show()
