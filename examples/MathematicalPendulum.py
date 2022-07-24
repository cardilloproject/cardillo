import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt

from cardillo.model.frame.frame import Frame
from cardillo.model.rigid_body import RigidBodyEuler
from cardillo.model.bilateral_constraints.implicit import SphericalJoint
from cardillo.math.algebra import cross3, ax2skew
from cardillo.math import approx_fprime
from cardillo.model import Model
from cardillo.solver import (
    GeneralizedAlphaFirstOrder,
    GenAlphaFirstOrderGGL2_V3,
    GeneralizedAlphaSecondOrder,
)


class MathematicalPendulum:
    def __init__(self, m, l, F_g, q0=None, u0=None):
        self.m = m
        self.l = l
        self.F_g = F_g

        self.nq = 2
        self.nu = 1
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

    def M(self, t, q, coo):
        M = np.array([[self.m * self.l**2]])
        coo.extend(M, (self.uDOF, self.uDOF))

    def f_pot(self, t, q):
        # return np.array([self.F_g @ q])
        return self.B_dense(q).T @ self.F_g

    def f_pot_q(self, t, q, coo):
        dense = np.array([self.F_g])
        coo.extend(dense, (self.uDOF, self.qDOF))

    def q_dot(self, t, q, u):
        return self.B_dense(q) @ u

    def q_ddot(self, t, q, u, u_dot):
        return self.B_dense(q) @ u_dot - q * (u @ u)

    def q_dot_q(self, t, q, u, coo):
        dense = np.array([[0, -1], [1, 0]], dtype=float) * u[0]
        coo.extend(dense, (self.qDOF, self.qDOF))

    def B_dense(self, q):
        x, y = q
        return np.array([[y], [-x]])

    def B(self, t, q, coo):
        coo.extend(self.B_dense(q), (self.qDOF, self.uDOF))


###################
# system parameters
###################
m = 1
l = 2
g = 9.81
F_g = np.array([0, -m * g])

###############
# initial state
###############
q0 = np.array([l, 0])
u0 = np.array([0])

#############
# build model
#############
pendulum = MathematicalPendulum(m, l, F_g, q0=q0, u0=u0)
model = Model()
model.add(pendulum)
model.assemble()


def transient():
    t1 = 10
    tol = 1.0e-8
    h = 1.0e-2

    def export_la_g(sol, name):
        header = "t, la_g0, la_g1, la_g2"
        t = sol.t
        la_g = sol.la_g
        export_data = np.vstack([t, *la_g.T]).T
        np.savetxt(
            name,
            export_data,
            delimiter=", ",
            header=header,
            comments="",
        )

    # solve index 3 problem with rho_inf = 0.9
    sol_9 = GeneralizedAlphaFirstOrder(
        model, t1, h, rho_inf=0.6, tol=tol, unknowns="velocities", GGL=False
    ).solve()
    # export_la_g(sol_9, "la_g_9.txt")

    # # solve index 3 problem with rho_inf = 0.6
    # sol_6 = GenAlphaFirstOrder(
    #     model, t1, h, rho_inf=0.6, tol=tol, unknowns="velocities", GGL=False
    # ).solve()
    # export_la_g(sol_6, "la_g_6.txt")

    # # solve GGL with rho_inf = 0.9
    # sol_9_GGL = GenAlphaFirstOrder(
    #     model, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities", GGL=True
    # ).solve()
    # export_la_g(sol_9_GGL, "la_g_9_GGL.txt")

    # # solve GGL with rho_inf = 0.6
    # sol_6_GGL = GenAlphaFirstOrder(
    #     model, t1, h, rho_inf=0.6, tol=tol, unknowns="velocities", GGL=True
    # ).solve()
    # export_la_g(sol_6_GGL, "la_g_6_GGL.txt")

    # # solve GGL with rho_inf = 0.9
    # sol_9_GGL2 = GenAlphaFirstOrderGGL2_V3(
    #     model, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities"
    # ).solve()
    # export_la_g(sol_9_GGL2, "la_g_9_GGL2.txt")

    # # solve GGL with rho_inf = 0.6
    # sol_6_GGL2 = GenAlphaFirstOrderGGL2_V3(
    #     model, t1, h, rho_inf=0.6, tol=tol, unknowns="velocities"
    # ).solve()
    # export_la_g(sol_6_GGL2, "la_g_6_GGL2.txt")

    ###################
    # visualize results
    ###################
    fig = plt.figure(figsize=plt.figaspect(1))

    ax = fig.add_subplot(3, 3, 1)
    # ax.plot(sol_6.t, sol_6.la_g[:, 0], "-k", label="la_g0_6")
    ax.plot(sol_9.t, sol_9.q[:, 0], "-k", label="x")
    ax.plot(sol_9.t, sol_9.q[:, 1], "--k", label="y")
    ax.grid()
    ax.legend()

    # # index 3
    # ax = fig.add_subplot(3, 3, 1)
    # # ax.plot(sol_6.t, sol_6.la_g[:, 0], "-k", label="la_g0_6")
    # ax.plot(sol_9.t, sol_9.la_g[:, 0], "--k", label="la_g_9")
    # ax.grid()
    # ax.legend()

    # ax = fig.add_subplot(3, 3, 4)
    # ax.plot(sol_6.t, sol_6.la_g[:, 1], "-k", label="la_g1_6")
    # ax.plot(sol_9.t, sol_9.la_g[:, 1], "--k", label="la_g1_9")
    # ax.grid()
    # ax.legend()

    # ax = fig.add_subplot(3, 3, 7)
    # ax.plot(sol_6.t, sol_6.la_g[:, 2], "-k", label="la_g2_6")
    # ax.plot(sol_9.t, sol_9.la_g[:, 2], "--k", label="la_g2_9")
    # ax.grid()
    # ax.legend()

    # # index 2
    # ax = fig.add_subplot(3, 3, 2)
    # ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 0], "-k", label="la_g0_6_GGL")
    # ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 0], "--k", label="la_g0_9_GGL")
    # ax.grid()
    # ax.legend()

    # ax = fig.add_subplot(3, 3, 5)
    # ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 1], "-k", label="la_g1_6_GGL")
    # ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 1], "--k", label="la_g1_9_GGL")
    # ax.grid()
    # ax.legend()

    # ax = fig.add_subplot(3, 3, 8)
    # ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 2], "-k", label="la_g2_6_GGL")
    # ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 2], "--k", label="la_g2_9_GGL")
    # ax.grid()
    # ax.legend()

    # # index 1
    # ax = fig.add_subplot(3, 3, 3)
    # ax.plot(sol_6_GGL2.t, sol_6_GGL2.la_g[:, 0], "-k", label="la_g0_6_GGL2")
    # ax.plot(sol_9_GGL2.t, sol_9_GGL2.la_g[:, 0], "--k", label="la_g0_9_GGL2")
    # ax.grid()
    # ax.legend()

    # ax = fig.add_subplot(3, 3, 6)
    # ax.plot(sol_6_GGL2.t, sol_6_GGL2.la_g[:, 1], "-k", label="la_g1_6_GGL2")
    # ax.plot(sol_9_GGL2.t, sol_9_GGL2.la_g[:, 1], "--k", label="la_g1_9_GGL2")
    # ax.grid()
    # ax.legend()

    # ax = fig.add_subplot(3, 3, 9)
    # ax.plot(sol_6_GGL2.t, sol_6_GGL2.la_g[:, 2], "-k", label="la_g2_6_GGL2")
    # ax.plot(sol_9_GGL2.t, sol_9_GGL2.la_g[:, 2], "--k", label="la_g2_9_GGL2")
    # ax.grid()
    # ax.legend()

    plt.show()


def gaps():
    t1 = 0.1
    tol = 1.0e-8
    h = 1.0e-3

    def export_gaps(sol, name):
        header = "t, g, g_dot, g_ddot"
        t = sol.t
        q = sol.q
        u = sol.u
        try:
            u_dot = sol.a  # GGL2 solver
        except:
            u_dot = sol.u_dot  # other solvers

        g = np.array([np.linalg.norm(model.g(ti, qi)) for ti, qi in zip(t, q)])

        g_dot = np.array(
            [np.linalg.norm(model.g_dot(ti, qi, ui)) for ti, qi, ui in zip(t, q, u)]
        )

        g_ddot = np.array(
            [
                np.linalg.norm(model.g_ddot(ti, qi, ui, u_doti))
                for ti, qi, ui, u_doti in zip(t, q, u, u_dot)
            ]
        )

        export_data = np.vstack([t, g, g_dot, g_ddot]).T
        np.savetxt(
            name,
            export_data,
            delimiter=", ",
            header=header,
            comments="",
        )

        return g, g_dot, g_ddot

    # solve index 3 problem with rho_inf = 0.9
    sol_9 = GeneralizedAlphaFirstOrder(
        model, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities", GGL=False
    ).solve()
    g_9, g_dot_9, g_ddot_9 = export_gaps(sol_9, "g_9.txt")

    # solve GGL with rho_inf = 0.9
    sol_9_GGL = GeneralizedAlphaFirstOrder(
        model, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities", GGL=True
    ).solve()
    g_9_GGL, g_dot_9_GGL, g_ddot_9_GGL = export_gaps(sol_9_GGL, "g_9_GGL.txt")

    # solve GGL2 with rho_inf = 0.9
    sol_9_GGL2 = GenAlphaFirstOrderGGL2_V3(
        model, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities"
    ).solve()
    g_9_GGL2, g_dot_9_GGL2, g_ddot_9_GGL2 = export_gaps(sol_9_GGL2, "g_9_GGL2.txt")

    ###################
    # visualize results
    ###################
    fig = plt.figure(figsize=plt.figaspect(1))

    # index 3
    ax = fig.add_subplot(3, 3, 1)
    ax.plot(sol_9.t, g_9, "-k", label="||g_9||")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 4)
    ax.plot(sol_9.t, g_dot_9, "-k", label="||g_dot_9||")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 7)
    ax.plot(sol_9.t, g_ddot_9, "-k", label="||g_ddot_9||")
    ax.grid()
    ax.legend()

    # index 2
    ax = fig.add_subplot(3, 3, 2)
    ax.plot(sol_9_GGL.t, g_9_GGL, "-k", label="||g_9_GGL||")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 5)
    ax.plot(sol_9_GGL.t, g_dot_9_GGL, "-k", label="||g_dot_9_GGL||")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 8)
    ax.plot(sol_9_GGL.t, g_ddot_9_GGL, "-k", label="||g_ddot_9_GGL||")
    ax.grid()
    ax.legend()

    # index 1
    ax = fig.add_subplot(3, 3, 3)
    ax.plot(sol_9_GGL2.t, g_9_GGL2, "-k", label="||g_9_GGL2||")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 6)
    ax.plot(sol_9_GGL2.t, g_dot_9_GGL2, "-k", label="||g_dot_9_GGL2||")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 9)
    ax.plot(sol_9_GGL2.t, g_ddot_9_GGL2, "-k", label="||g_ddot_9_GGL2||")
    ax.grid()
    ax.legend()

    plt.show()


def convergence():
    # end time and numerical dissipation of generalized-alpha solver
    # t1 = 5
    # t1 = 5 / 4
    # t1 = 5 / 8
    # t1 = 1
    # t1 = 0.25
    # t1 = 1
    t1 = 0.1  # this is used for investigating the transient behavior
    rho_inf = 0.9
    tol_ref = 1.0e-12
    tol = 1.0e-8

    # log spaced time steps
    # num = 3
    # dts = np.logspace(-1, -num, num=num, endpoint=True)
    # dts = np.logspace(-2, -num, num=num - 1, endpoint=True)
    # dts = np.array([1.0e-2])
    dts = np.array([5e-2, 1e-2])
    # dts = np.array([5e-2, 1e-2, 1.0e-3])
    # dts = np.array([1e-2, 1e-3, 1e-4])
    # dts = np.array([1e-2, 5e-3, 1e-3, 5e-4, 1e-4]) # used by Arnold2015, p. 29
    dts_1 = dts
    dts_2 = dts**2
    print(f"dts: {dts}")

    # errors for all 6 possible solvers
    q_errors = np.inf * np.ones((6, len(dts)), dtype=float)
    u_errors = np.inf * np.ones((6, len(dts)), dtype=float)
    la_g_errors = np.inf * np.ones((6, len(dts)), dtype=float)

    # compute reference solution as described in Arnold2015 Section 3.3
    print(f"compute reference solution:")
    # dt_ref = 1.0e-5
    # dt_ref = 2.5e-5 # see Arnold2015 p. 174/ Arnodl2015b p. 14
    # dt_ref = 1.0e-4
    dt_ref = 1.0e-3
    reference = GeneralizedAlphaFirstOrder(
        model, t1, dt_ref, rho_inf=rho_inf, tol=tol_ref, unknowns="velocities", GGL=True
    ).solve()
    t_ref = reference.t
    q_ref = reference.q
    u_ref = reference.u
    la_g_ref = reference.la_g

    # plot_state = True
    plot_state = False
    if plot_state:
        ###################
        # visualize results
        ###################
        fig = plt.figure(figsize=plt.figaspect(0.5))

        # center of mass
        ax = fig.add_subplot(2, 3, 1)
        ax.plot(t_ref, q_ref[:, 0], "-r", label="x")
        ax.plot(t_ref, q_ref[:, 1], "-g", label="y")
        ax.plot(t_ref, q_ref[:, 2], "-b", label="z")
        ax.grid()
        ax.legend()

        # alpha, beta, gamma
        ax = fig.add_subplot(2, 3, 2)
        ax.plot(t_ref, q_ref[:, 3], "-r", label="alpha")
        ax.plot(t_ref, q_ref[:, 4], "-g", label="beta")
        ax.plot(t_ref, q_ref[:, 5], "-b", label="gamm")
        ax.grid()
        ax.legend()

        # x-y-z trajectory
        ax = fig.add_subplot(2, 3, 3, projection="3d")
        ax.plot3D(
            q_ref[:, 0],
            q_ref[:, 1],
            q_ref[:, 2],
            "-r",
            label="x-y-z trajectory",
        )
        ax.grid()
        ax.legend()

        # x_dot, y_dot, z_dot
        ax = fig.add_subplot(2, 3, 4)
        ax.plot(t_ref, u_ref[:, 0], "-r", label="x_dot")
        ax.plot(t_ref, u_ref[:, 1], "-g", label="y_dot")
        ax.plot(t_ref, u_ref[:, 2], "-b", label="z_dot")
        ax.grid()
        ax.legend()

        # omega_x, omega_y, omega_z
        ax = fig.add_subplot(2, 3, 5)
        ax.plot(t_ref, u_ref[:, 3], "-r", label="omega_x")
        ax.plot(t_ref, u_ref[:, 4], "-g", label="omega_y")
        ax.plot(t_ref, u_ref[:, 5], "-b", label="omega_z")
        ax.grid()
        ax.legend()

        # la_g
        ax = fig.add_subplot(2, 3, 6)
        ax.plot(t_ref, la_g_ref[:, 0], "-r", label="la_g0")
        ax.plot(t_ref, la_g_ref[:, 1], "-g", label="la_g1")
        ax.plot(t_ref, la_g_ref[:, 2], "-b", label="la_g2")
        ax.grid()
        ax.legend()

        plt.show()

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

        # TODO: What error do we chose?
        # # relative error
        # q_error = np.linalg.norm(diff_q) / np.linalg.norm(q)
        # u_error = np.linalg.norm(diff_u) / np.linalg.norm(u)
        # la_g_error = np.linalg.norm(diff_la_g) / np.linalg.norm(la_g)

        # # max abs difference
        # q_error = np.max(np.abs(diff_q))
        # u_error = np.max(np.abs(diff_u))
        # la_g_error = np.max(np.abs(diff_la_g))

        # max relative error
        q_error = np.max(np.linalg.norm(diff_q, axis=0) / np.linalg.norm(q, axis=0))
        u_error = np.max(np.linalg.norm(diff_u, axis=0) / np.linalg.norm(u, axis=0))
        la_g_error = np.max(
            np.linalg.norm(diff_la_g, axis=0) / np.linalg.norm(la_g, axis=0)
        )

        return q_error, u_error, la_g_error

    # # generate files for error export that only contain the header
    # header = "dt, dt2, pos, vel, aux, pos_GGL, vel_GGL, aux_GGL"
    # header = np.array([["dt", "dt2", "pos", "vel", "aux", "pos_GGL", "vel_GGL", "aux_GGL"]])
    # data = np.empty((0, 0))
    # np.savetxt("error_heavy_top_q.txt", data, delimiter=", ", header=header, comments="")
    # np.savetxt("error_heavy_top_u.txt", export_data, delimiter=", ", header=header, comments="")
    # np.savetxt("error_heavy_top_la_g.txt", export_data, delimiter=", ", header=header, comments="")

    for i, dt in enumerate(dts):
        print(f"i: {i}, dt: {dt:1.1e}")

        # position formulation
        sol = GeneralizedAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="positions", GGL=False
        ).solve()
        q_errors[0, i], u_errors[0, i], la_g_errors[0, i] = errors(sol)

        # velocity formulation
        sol = GeneralizedAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="velocities", GGL=False
        ).solve()
        q_errors[1, i], u_errors[1, i], la_g_errors[1, i] = errors(sol)

        # auxiliary formulation
        sol = GeneralizedAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="auxiliary", GGL=False
        ).solve()
        q_errors[2, i], u_errors[2, i], la_g_errors[2, i] = errors(sol)

        # GGL formulation - positions
        sol = GeneralizedAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="positions", GGL=True
        ).solve()
        q_errors[3, i], u_errors[3, i], la_g_errors[3, i] = errors(sol)

        # GGL formulation - velocityies
        sol = GeneralizedAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="velocities", GGL=True
        ).solve()
        q_errors[4, i], u_errors[4, i], la_g_errors[4, i] = errors(sol)

        # GGL formulation - auxiliary
        sol = GeneralizedAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="auxiliary", GGL=True
        ).solve()
        q_errors[5, i], u_errors[5, i], la_g_errors[5, i] = errors(sol)

        #############################
        # export errors and dt, dt**2
        #############################
        if i == 0:
            header = "dt, dt2, pos, vel, aux, pos_GGL, vel_GGL, aux_GGL"
            export_data = np.array([[dt, dt**2, *q_errors[:, i]]])
            np.savetxt(
                "error_heavy_top_q.txt",
                export_data,
                delimiter=", ",
                header=header,
                comments="",
            )
        else:
            with open("error_heavy_top_q.txt", "ab") as f:
                # f.write(b"\n")
                export_data = np.array([[dt, dt**2, *q_errors[:, i]]])
                np.savetxt(f, export_data, delimiter=", ", comments="")

    #############################
    # export errors and dt, dt**2
    #############################
    header = "dt, dt2, pos, vel, aux, pos_GGL, vel_GGL, aux_GGL"

    export_data = np.vstack((dts, dts_2, *q_errors)).T
    np.savetxt(
        "error_heavy_top_q.txt", export_data, delimiter=", ", header=header, comments=""
    )

    export_data = np.vstack((dts, dts_2, *u_errors)).T
    np.savetxt(
        "error_heavy_top_u.txt", export_data, delimiter=", ", header=header, comments=""
    )

    export_data = np.vstack((dts, dts_2, *la_g_errors)).T
    np.savetxt(
        "error_heavy_top_la_g.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    # # names = ["GenAlphaFirstOrder", "GenAlphaSecondOrder", "GenAlphaFirstOrderGGl", "Theta"]
    # # ts = [t_GenAlphaFirstOrder, t_GenAlphaSecondOrder, t_GenAlphaFirstOrderGGl, t_ThetaNewton]
    # # qs = [q_GenAlphaFirstOrder, q_GenAlphaSecondOrder, q_GenAlphaFirstOrderGGl, q_ThetaNewton]
    # # us = [u_GenAlphaFirstOrder, u_GenAlphaSecondOrder, u_GenAlphaFirstOrderGGl, u_ThetaNewton]
    # # la_gs = [la_g_GenAlphaFirstOrder, la_g_GenAlphaSecondOrder, la_g_GenAlphaFirstOrderGGl, la_g_ThetaNewton]
    # names = ["pos", "vel", "aux", "pos-GGL", "vel-GGL", "aux-GGL"]
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


if __name__ == "__main__":
    # transient()
    gaps()
    # convergence()
