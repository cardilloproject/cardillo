import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cardillo.math.rotations import Spurrier

from cardillo.discrete import (
    RigidBodyQuaternion,
)
from cardillo.constraints import Spherical
from cardillo.math.algebra import cross3
from cardillo import System
from cardillo.solver import (
    EulerBackward,
    GeneralizedAlphaFirstOrder,
    GeneralizedAlphaSecondOrder,
)


class HeavyTopQuaternion(RigidBodyQuaternion):
    def __init__(self, A, B, grav, q0=None, u0=None):
        self.grav = grav
        self.r_OQ = r_OQ

        # initialize rigid body
        self.K_Theta_S = np.diag([A, A, B])
        RigidBodyQuaternion.__init__(self, m, self.K_Theta_S, q0=q0, u0=u0)

        # gravity
        self.f_g = np.array([0, 0, -self.m * self.grav])

    def h(self, t, q, u):
        return self.f_g @ self.J_P(t, q)

    def h_q(self, t, q, u, coo):
        dense = np.einsum("i,ijk->jk", self.f_g, self.J_P_q(t, q))
        coo.extend(dense, (self.uDOF, self.qDOF))


########################################
# Arnold2015, p. 174/ Arnold2015b, p. 13
########################################
m = 15.0
A = 0.234375
B = 0.46875
l = 1.0
grav = 9.81
alpha0 = pi
beta0 = pi / 2
gamma0 = 0

omega_x0 = 0
# omega_y0 = 0 # Arnodl2015 p. 174
omega_y0 = -4.61538  # Arnold2015b p. 13
omega_z0 = 150

#############################
# initial position and angles
#############################
phi0 = np.array([alpha0, beta0, gamma0])


def A_IK(alpha, beta, gamma):
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta), np.cos(beta)
    sg, cg = np.sin(gamma), np.cos(gamma)
    # fmt: off
    return np.array([
        [ca * cg - sa * cb * sg, - ca * sg - sa * cb * cg, sa * sb],
        [sa * cg + ca * cb * sg, -sa * sg + ca * cb * cg, -ca * sb],
        [sb * sg, sb * cg, cb]
    ])
    # fmt: on


r_OQ = np.zeros(3)
K_r_OS0 = np.array([0, 0, l])
A_IK0 = A_IK(alpha0, beta0, gamma0)
r_OS0 = A_IK0 @ K_r_OS0
q0 = np.concatenate((r_OS0, Spurrier(A_IK0)))

####################
# initial velocities
####################
K_Omega0 = np.array([omega_x0, omega_y0, omega_z0])
v_S0 = A_IK0 @ cross3(K_Omega0, K_r_OS0)
u0 = np.concatenate((v_S0, K_Omega0))

#################
# build the system
#################
system = System()

top = HeavyTopQuaternion(A, B, grav, q0, u0)
spherical_joint = Spherical(system.origin, top, np.zeros(3, dtype=float))
system.add(top)
system.add(spherical_joint)
system.assemble()


def show_animation(top, t, q, scale=1, show=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=-scale, top=scale)

    slowmotion = 1
    fps = 200
    t0 = t[0]
    t1 = t[-1]
    animation_time = slowmotion * t1
    target_frames = int(fps * animation_time)
    frac = max(1, int(len(t) / target_frames))
    if frac == 1:
        target_frames = len(t)
    interval = 1000 / fps

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    def create(t, q):
        x_S, y_S, z_S = top.r_OP(t, q)

        A_IK = top.A_IK(t, q)
        d1, d2, d3 = A_IK.T

        (COM,) = ax.plot([0.0, x_S], [0.0, y_S], [0.0, z_S], "-ok")
        (d1_,) = ax.plot(
            [x_S, x_S + d1[0]], [y_S, y_S + d1[1]], [z_S, z_S + d1[2]], "-r"
        )
        (d2_,) = ax.plot(
            [x_S, x_S + d2[0]], [y_S, y_S + d2[1]], [z_S, z_S + d2[2]], "-g"
        )
        (d3_,) = ax.plot(
            [x_S, x_S + d3[0]], [y_S, y_S + d3[1]], [z_S, z_S + d3[2]], "-b"
        )

        return COM, d1_, d2_, d3_

    COM, d1_, d2_, d3_ = create(t0, q[0])

    def update(t, q, COM, d1_, d2_, d3_):
        x_S, y_S, z_S = top.r_OP(t, q)

        A_IK = top.A_IK(t, q)
        d1, d2, d3 = A_IK.T

        COM.set_data(np.array([0.0, x_S]), np.array([0.0, y_S]))
        COM.set_3d_properties(np.array([0.0, z_S]))

        d1_.set_data(np.array([x_S, x_S + d1[0]]), np.array([y_S, y_S + d1[1]]))
        d1_.set_3d_properties(np.array([z_S, z_S + d1[2]]))

        d2_.set_data(np.array([x_S, x_S + d2[0]]), np.array([y_S, y_S + d2[1]]))
        d2_.set_3d_properties(np.array([z_S, z_S + d2[2]]))

        d3_.set_data(np.array([x_S, x_S + d3[0]]), np.array([y_S, y_S + d3[1]]))
        d3_.set_3d_properties(np.array([z_S, z_S + d3[2]]))

        return COM, d1_, d2_, d3_

    def animate(i):
        update(t[i], q[i], COM, d1_, d2_, d3_)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    if show:
        plt.show()

    return anim


def state():
    rho_inf = 0.9
    tol = 1.0e-8
    t1 = 0.25
    dt = 1.0e-3

    sol = EulerBackward(system, t1, dt, atol=tol).solve()

    # sol = GeneralizedAlphaFirstOrder(
    #     system,
    #     t1,
    #     dt,
    #     rho_inf=rho_inf,
    #     tol=tol,
    #     unknowns="velocities",
    #     GGL=False,
    #     # GGL=True,
    #     # numerical_jacobian=False,
    #     numerical_jacobian=True,
    # ).solve()

    # sol = GeneralizedAlphaSecondOrder(
    #     system, t1, dt, rho_inf=rho_inf, tol=tol
    # ).solve()

    t = sol.t
    q = sol.q
    u = sol.u
    la_g = sol.la_g

    def export_q(sol, name):
        header = "t, x, y, z, al, be, ga, la_g"
        export_data = np.vstack([sol.t, *sol.q.T, *sol.la_g.T]).T
        np.savetxt(
            name,
            export_data,
            delimiter=", ",
            header=header,
            comments="",
        )

    export_q(sol, "trajectory.txt")

    ###################
    # visualize results
    ###################
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # center of mass
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(t, q[:, 0], "-r", label="x")
    ax.plot(t, q[:, 1], "-g", label="y")
    ax.plot(t, q[:, 2], "-b", label="z")
    ax.grid()
    ax.legend()

    # alpha, beta, gamma
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(t, q[:, 3], "-r", label="alpha")
    ax.plot(t, q[:, 4], "-g", label="beta")
    ax.plot(t, q[:, 5], "-b", label="gamm")
    ax.grid()
    ax.legend()

    # x-y-z trajectory
    ax = fig.add_subplot(2, 3, 3, projection="3d")
    ax.plot3D(
        q[:, 0],
        q[:, 1],
        q[:, 2],
        "-r",
        label="x-y-z trajectory",
    )
    ax.grid()
    ax.legend()

    # x_dot, y_dot, z_dot
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(t, u[:, 0], "-r", label="x_dot")
    ax.plot(t, u[:, 1], "-g", label="y_dot")
    ax.plot(t, u[:, 2], "-b", label="z_dot")
    ax.grid()
    ax.legend()

    # omega_x, omega_y, omega_z
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(t, u[:, 3], "-r", label="omega_x")
    ax.plot(t, u[:, 4], "-g", label="omega_y")
    ax.plot(t, u[:, 5], "-b", label="omega_z")
    ax.grid()
    ax.legend()

    # la_g
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(t, la_g[:, 0], "-r", label="la_g0")
    ax.plot(t, la_g[:, 1], "-g", label="la_g1")
    ax.plot(t, la_g[:, 2], "-b", label="la_g2")
    ax.grid()
    ax.legend()

    anim = show_animation(top, t, q)

    plt.show()


def transient():
    t1 = 0.1
    tol = 1.0e-8
    h = 1.0e-3

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
    # sol_9 = GeneralizedAlphaFirstOrder(
    #     system, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities", GGL=False
    # ).solve()
    sol_9 = GeneralizedAlphaFirstOrderGGLGiuseppe(
        system, t1, h, rho_inf=0.9, tol=tol
    ).solve()
    export_la_g(sol_9, "la_g_9.txt")

    # solve index 3 problem with rho_inf = 0.6
    sol_6 = GeneralizedAlphaFirstOrder(
        system, t1, h, rho_inf=0.6, tol=tol, unknowns="velocities", GGL=False
    ).solve()
    export_la_g(sol_6, "la_g_6.txt")

    # solve GGL with rho_inf = 0.9
    sol_9_GGL = GeneralizedAlphaFirstOrder(
        system, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities", GGL=True
    ).solve()
    export_la_g(sol_9_GGL, "la_g_9_GGL.txt")

    # solve GGL with rho_inf = 0.6
    sol_6_GGL = GeneralizedAlphaFirstOrder(
        system, t1, h, rho_inf=0.6, tol=tol, unknowns="velocities", GGL=True
    ).solve()
    export_la_g(sol_6_GGL, "la_g_6_GGL.txt")

    # # solve GGL with rho_inf = 0.9
    # sol_9_GGL2 = GenAlphaFirstOrderGGL2_V3(
    #     system, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities"
    # ).solve()
    # export_la_g(sol_9_GGL2, "la_g_9_GGL2.txt")

    # # solve GGL with rho_inf = 0.6
    # sol_6_GGL2 = GenAlphaFirstOrderGGL2_V3(
    #     system, t1, h, rho_inf=0.6, tol=tol, unknowns="velocities"
    # ).solve()
    # export_la_g(sol_6_GGL2, "la_g_6_GGL2.txt")

    ###################
    # visualize results
    ###################
    fig = plt.figure(figsize=plt.figaspect(1))

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(sol_6.t, sol_6.la_g[:, 0], "-r", label="la_g0_6")
    ax.plot(sol_6.t, sol_6.la_g[:, 1], "-g", label="la_g1_6")
    ax.plot(sol_6.t, sol_6.la_g[:, 2], "-b", label="la_g2_6")
    ax.plot(sol_9.t, sol_9.la_g[:, 0], "--r", label="la_g0_9")
    ax.plot(sol_9.t, sol_9.la_g[:, 1], "--g", label="la_g1_9")
    ax.plot(sol_9.t, sol_9.la_g[:, 2], "--b", label="la_g2_9")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 0], "-r", label="la_g0_6_GGL")
    ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 1], "-g", label="la_g1_6_GGL")
    ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 2], "-b", label="la_g2_6_GGL")
    ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 0], "--r", label="la_g0_9_GGL")
    ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 1], "--g", label="la_g1_9_GGL")
    ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 2], "--b", label="la_g2_9_GGL")
    ax.grid()
    ax.legend()

    plt.show()
    exit()

    # index 3
    ax = fig.add_subplot(3, 3, 1)
    ax.plot(sol_6.t, sol_6.la_g[:, 0], "-k", label="la_g0_6")
    ax.plot(sol_9.t, sol_9.la_g[:, 0], "--k", label="la_g0_9")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 4)
    ax.plot(sol_6.t, sol_6.la_g[:, 1], "-k", label="la_g1_6")
    ax.plot(sol_9.t, sol_9.la_g[:, 1], "--k", label="la_g1_9")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 7)
    ax.plot(sol_6.t, sol_6.la_g[:, 2], "-k", label="la_g2_6")
    ax.plot(sol_9.t, sol_9.la_g[:, 2], "--k", label="la_g2_9")
    ax.grid()
    ax.legend()

    # index 2
    ax = fig.add_subplot(3, 3, 2)
    ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 0], "-k", label="la_g0_6_GGL")
    ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 0], "--k", label="la_g0_9_GGL")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 5)
    ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 1], "-k", label="la_g1_6_GGL")
    ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 1], "--k", label="la_g1_9_GGL")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 8)
    ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 2], "-k", label="la_g2_6_GGL")
    ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 2], "--k", label="la_g2_9_GGL")
    ax.grid()
    ax.legend()

    # index 1
    ax = fig.add_subplot(3, 3, 3)
    ax.plot(sol_6_GGL2.t, sol_6_GGL2.la_g[:, 0], "-k", label="la_g0_6_GGL2")
    ax.plot(sol_9_GGL2.t, sol_9_GGL2.la_g[:, 0], "--k", label="la_g0_9_GGL2")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 6)
    ax.plot(sol_6_GGL2.t, sol_6_GGL2.la_g[:, 1], "-k", label="la_g1_6_GGL2")
    ax.plot(sol_9_GGL2.t, sol_9_GGL2.la_g[:, 1], "--k", label="la_g1_9_GGL2")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 9)
    ax.plot(sol_6_GGL2.t, sol_6_GGL2.la_g[:, 2], "-k", label="la_g2_6_GGL2")
    ax.plot(sol_9_GGL2.t, sol_9_GGL2.la_g[:, 2], "--k", label="la_g2_9_GGL2")
    ax.grid()
    ax.legend()

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
        # try:
        #     u_dot = sol.a  # GGL2 solver
        # except:
        #     u_dot = sol.u_dot  # other solvers
        u_dot = sol.u_dot  # other solvers

        g = np.array([np.linalg.norm(system.g(ti, qi)) for ti, qi in zip(t, q)])

        g_dot = np.array(
            [np.linalg.norm(system.g_dot(ti, qi, ui)) for ti, qi, ui in zip(t, q, u)]
        )

        g_ddot = np.array(
            [
                np.linalg.norm(system.g_ddot(ti, qi, ui, u_doti))
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
        system, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities", GGL=False
    ).solve()
    g_9, g_dot_9, g_ddot_9 = export_gaps(sol_9, "g_9.txt")

    # solve GGL with rho_inf = 0.9
    sol_9_GGL = GeneralizedAlphaFirstOrder(
        system, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities", GGL=True
    ).solve()
    g_9_GGL, g_dot_9_GGL, g_ddot_9_GGL = export_gaps(sol_9_GGL, "g_9_GGL.txt")

    # solve GGL2 with rho_inf = 0.9
    # sol_9_GGL2 = GenAlphaFirstOrderGGL2_V3(
    #     system, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities"
    # ).solve()
    sol_9_GGL2 = GeneralizedAlphaFirstOrderGGLGiuseppe(
        system, t1, h, rho_inf=0.9, tol=tol
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
    rho_inf = 0.9
    tol_ref = 1.0e-8
    tol = 1.0e-8

    # compute step sizes with powers of 2
    dt_ref = 2.5e-5  # Arnold2015b
    dts = (2.0 ** np.arange(7, 1, -1)) * dt_ref  # [3.2e-3, ..., 2e-4, 1e-4]
    # dts = (2.0 ** np.arange(7, 4, -1)) * dt_ref  # [3.2e-3, 1.6e-3, 8e-4]

    # end time (note this has to be > 0.5, otherwise long term error throws ans error)
    # t1 = (2.0**9) * dt_ref  # this yields 0.256 for dt_ref = 5e-4
    # t1 = (2.0**10) * dt_ref  # this yields 0.512 for dt_ref = 5e-4
    # t1 = (2.0**11) * dt_ref  # this yields 0.2048 for dt_ref = 1e-4
    # t1 = (2.0**13) * dt_ref  # this yields 0.8192 for dt_ref = 1e-4
    t1 = (2.0**15) * dt_ref  # this yields 0.8192 for dt_ref = 2.5e-5
    # # t1 = (2.0**16) * dt_ref # this yields 1.6384 for dt_ref = 2.5e-5

    # TODO: Only for debugging!
    dt_ref = 2.5e-3
    dts = np.array([5.0e-3])
    t1 = (2.0**8) * dt_ref

    dts_1 = dts
    dts_2 = dts**2

    print(f"t1: {t1}")
    print(f"dts: {dts}")
    # exit()

    # errors for possible solvers
    q_errors_transient = np.inf * np.ones((3, len(dts)), dtype=float)
    u_errors_transient = np.inf * np.ones((3, len(dts)), dtype=float)
    la_g_errors_transient = np.inf * np.ones((3, len(dts)), dtype=float)
    q_errors_longterm = np.inf * np.ones((3, len(dts)), dtype=float)
    u_errors_longterm = np.inf * np.ones((3, len(dts)), dtype=float)
    la_g_errors_longterm = np.inf * np.ones((3, len(dts)), dtype=float)

    dt_ref = 2.5e-3
    t1 = 1

    ###################################################################
    # compute reference solution as described in Arnold2015 Section 3.3
    ###################################################################
    # print(f"compute reference solution with first order method:")
    # reference1 = GeneralizedAlphaFirstOrder(
    #     system, t1, dt_ref, rho_inf=rho_inf, tol=tol_ref, unknowns="velocities", GGL=False
    # ).solve()

    print(f"compute reference solution with first order method + GGL:")
    reference1_GGL = GeneralizedAlphaFirstOrder(
        system,
        t1,
        dt_ref,
        rho_inf=rho_inf,
        tol=tol_ref,
        unknowns="velocities",
        GGL=True,
    ).solve()

    # print(f"compute reference solution with second order method:")
    # reference2 = GeneralizedAlphaSecondOrder(
    #     system, t1, dt_ref, rho_inf=rho_inf, tol=tol_ref, GGL=False
    # ).solve()

    # print(f"compute reference solution with second order method + GGL:")
    # reference2_GGL = GeneralizedAlphaSecondOrder(
    #     system, t1, dt_ref, rho_inf=rho_inf, tol=tol_ref, GGL=True
    # ).solve()

    print(f"done")

    plot_state = True
    # plot_state = False
    if plot_state:
        # reference = reference1
        reference = reference1_GGL
        # reference = reference2
        # reference = reference2_GGL
        t_ref = reference.t
        q_ref = reference.q
        u_ref = reference.u
        la_g_ref = reference.la_g

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

    exit()

    def errors(sol, sol_ref, t_transient=0.1, t_longterm=0.5):
        t = sol.t
        q = sol.q
        u = sol.u
        la_g = sol.la_g

        t_ref = sol_ref.t
        q_ref = sol_ref.q
        u_ref = sol_ref.u
        # t = t_genAlphaFirstOrderVelocityGGL
        # q = q_genAlphaFirstOrderVelocityGGL
        # t = t_genAlphaFirstOrderVelocityGGL
        # q = q_genAlphaFirstOrderVelocityGGL
        # t = t_genAlphaFirstOrderVelocityGGL
        # q = q_genAlphaFirstOrderVelocityGGL
        la_g_ref = sol_ref.la_g

        # distinguish between transient and long term time steps
        t_idx_transient = np.where(t <= t_transient)[0]
        t_idx_longterm = np.where(t >= t_longterm)[0]

        # compute difference between computed solution and reference solution
        # for identical time instants
        t_ref_idx_transient = np.where(
            np.abs(t[t_idx_transient, None] - t_ref) < 1.0e-8
        )[1]
        t_ref_idx_longterm = np.where(np.abs(t[t_idx_longterm, None] - t_ref) < 1.0e-8)[
            1
        ]

        # differences
        q_transient = q[t_idx_transient]
        u_transient = u[t_idx_transient]
        la_g_transient = la_g[t_idx_transient]
        diff_transient_q = q_transient - q_ref[t_ref_idx_transient]
        diff_transient_u = u_transient - u_ref[t_ref_idx_transient]
        diff_transient_la_g = la_g_transient - la_g_ref[t_ref_idx_transient]

        q_longterm = q[t_idx_longterm]
        u_longterm = u[t_idx_longterm]
        la_g_longterm = la_g[t_idx_longterm]
        diff_longterm_q = q_longterm - q_ref[t_ref_idx_longterm]
        diff_longterm_u = u_longterm - u_ref[t_ref_idx_longterm]
        diff_longterm_la_g = la_g_longterm - la_g_ref[t_ref_idx_longterm]

        # max relative error
        q_error_transient = np.max(
            np.linalg.norm(diff_transient_q, axis=1)
            / np.linalg.norm(q_transient, axis=1)
        )
        u_error_transient = np.max(
            np.linalg.norm(diff_transient_u, axis=1)
            / np.linalg.norm(u_transient, axis=1)
        )
        la_g_error_transient = np.max(
            np.linalg.norm(diff_transient_la_g, axis=1)
            / np.linalg.norm(la_g_transient, axis=1)
        )

        q_error_longterm = np.max(
            np.linalg.norm(diff_longterm_q, axis=1) / np.linalg.norm(q_longterm, axis=1)
        )
        u_error_longterm = np.max(
            np.linalg.norm(diff_longterm_u, axis=1) / np.linalg.norm(u_longterm, axis=1)
        )
        la_g_error_longterm = np.max(
            np.linalg.norm(diff_longterm_la_g, axis=1)
            / np.linalg.norm(la_g_longterm, axis=1)
        )

        return (
            q_error_transient,
            u_error_transient,
            la_g_error_transient,
            q_error_longterm,
            u_error_longterm,
            la_g_error_longterm,
        )

    for i, dt in enumerate(dts):
        print(f"i: {i}, dt: {dt:1.1e}")

        # generalized alpha for mechanical systems in second order form
        sol = GeneralizedAlphaSecondOrder(
            system, t1, dt, rho_inf=rho_inf, tol=tol
        ).solve()
        (
            q_errors_transient[0, i],
            u_errors_transient[0, i],
            la_g_errors_transient[0, i],
            q_errors_longterm[0, i],
            u_errors_longterm[0, i],
            la_g_errors_longterm[0, i],
        ) = errors(sol, reference2)

        # generalized alpha for mechanical systems in first order form (velocity formulation)
        sol = GeneralizedAlphaFirstOrder(
            system, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="velocities", GGL=False
        ).solve()
        (
            q_errors_transient[1, i],
            u_errors_transient[1, i],
            la_g_errors_transient[1, i],
            q_errors_longterm[1, i],
            u_errors_longterm[1, i],
            la_g_errors_longterm[1, i],
        ) = errors(sol, reference1)

        # generalized alpha for mechanical systems in first order form (velocity formulation - GGL)
        sol = GeneralizedAlphaFirstOrder(
            system, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="velocities", GGL=True
        ).solve()
        (
            q_errors_transient[2, i],
            u_errors_transient[2, i],
            la_g_errors_transient[2, i],
            q_errors_longterm[2, i],
            u_errors_longterm[2, i],
            la_g_errors_longterm[2, i],
        ) = errors(sol, reference1_GGL)

    #############################
    # export errors and dt, dt**2
    #############################
    header = "dt, dt2, 2nd, 1st, 1st_GGL"

    # transient errors
    export_data = np.vstack((dts, dts_2, *q_errors_transient)).T
    np.savetxt(
        "transient_error_heavy_top_q.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    export_data = np.vstack((dts, dts_2, *u_errors_transient)).T
    np.savetxt(
        "transient_error_heavy_top_u.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    export_data = np.vstack((dts, dts_2, *la_g_errors_transient)).T
    np.savetxt(
        "transient_error_heavy_top_la_g.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    # longterm errors
    export_data = np.vstack((dts, dts_2, *q_errors_longterm)).T
    np.savetxt(
        "longterm_error_heavy_top_q.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    export_data = np.vstack((dts, dts_2, *u_errors_longterm)).T
    np.savetxt(
        "longterm_error_heavy_top_u.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    export_data = np.vstack((dts, dts_2, *la_g_errors_longterm)).T
    np.savetxt(
        "longterm_error_heavy_top_la_g.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    ##################
    # visualize errors
    ##################
    fig, ax = plt.subplots(3, 2)

    ax[0, 0].set_title("transient: gen alpha 2nd order")
    ax[0, 0].loglog(dts, dts_1, "-k", label="dt")
    ax[0, 0].loglog(dts, dts_2, "--k", label="dt^2")
    ax[0, 0].loglog(dts, q_errors_transient[0], "-.ro", label="q")
    ax[0, 0].loglog(dts, u_errors_transient[0], "-.go", label="u")
    ax[0, 0].loglog(dts, la_g_errors_transient[0], "-.bo", label="la_g")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].set_title("transient: gen alpha 1st order (velocity form.)")
    ax[1, 0].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 0].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 0].loglog(dts, q_errors_transient[1], "-.ro", label="q")
    ax[1, 0].loglog(dts, u_errors_transient[1], "-.go", label="u")
    ax[1, 0].loglog(dts, la_g_errors_transient[1], "-.bo", label="la_g")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[2, 0].set_title("transient: gen alpha 1st order (velocity form. + GGL)")
    ax[2, 0].loglog(dts, dts_1, "-k", label="dt")
    ax[2, 0].loglog(dts, dts_2, "--k", label="dt^2")
    ax[2, 0].loglog(dts, q_errors_transient[2], "-.ro", label="q")
    ax[2, 0].loglog(dts, u_errors_transient[2], "-.go", label="u")
    ax[2, 0].loglog(dts, la_g_errors_transient[2], "-.bo", label="la_g")
    ax[2, 0].grid()
    ax[2, 0].legend()

    ax[0, 1].set_title("long term: gen alpha 2nd order")
    ax[0, 1].loglog(dts, dts_1, "-k", label="dt")
    ax[0, 1].loglog(dts, dts_2, "--k", label="dt^2")
    ax[0, 1].loglog(dts, q_errors_longterm[0], "-.ro", label="q")
    ax[0, 1].loglog(dts, u_errors_longterm[0], "-.go", label="u")
    ax[0, 1].loglog(dts, la_g_errors_longterm[0], "-.bo", label="la_g")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].set_title("long term: gen alpha 1st order (velocity form.)")
    ax[1, 1].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 1].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 1].loglog(dts, q_errors_longterm[1], "-.ro", label="q")
    ax[1, 1].loglog(dts, u_errors_longterm[1], "-.go", label="u")
    ax[1, 1].loglog(dts, la_g_errors_longterm[1], "-.bo", label="la_g")
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[2, 1].set_title("long term: gen alpha 1st order (velocity form. + GGL)")
    ax[2, 1].loglog(dts, dts_1, "-k", label="dt")
    ax[2, 1].loglog(dts, dts_2, "--k", label="dt^2")
    ax[2, 1].loglog(dts, q_errors_longterm[2], "-.ro", label="q")
    ax[2, 1].loglog(dts, u_errors_longterm[2], "-.go", label="u")
    ax[2, 1].loglog(dts, la_g_errors_longterm[2], "-.bo", label="la_g")
    ax[2, 1].grid()
    ax[2, 1].legend()

    plt.show()


if __name__ == "__main__":
    state()
    # transient()
    # gaps()
    # convergence()
