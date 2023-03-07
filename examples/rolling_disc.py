import numpy as np

from math import pi, sin, cos, sqrt

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo import System
from cardillo.discrete import RigidBodyQuaternion, RigidBodyAxisAngle, RigidBodyEuler
from cardillo.math import axis_angle2quat
from cardillo.constraints import (
    RollingCondition_g_I_Frame_gamma,
    RollingCondition,
)
from cardillo.forces import Force
from cardillo.solver import (
    ScipyIVP,
    EulerBackward,
    GeneralizedAlphaFirstOrder,
    NonsmoothBackwardEulerDecoupled,
    RadauIIa,
    Rattle,
)


############
# parameters
############
g = 9.81  # gravity
m = 0.3048  # disc mass

# disc radius
r = 0.05  # used for GAMM

# radius of of the circular motion
R = 10 * r  # used for GAMM

# inertia of the disc, Lesaux2005 before (5.3)
A = B = 0.25 * m * r**2
C = 0.5 * m * r**2

# ratio between disc radius and radius of rolling
rho = r / R  # Lesaux2005 (5.10)

####################
# initial conditions
####################
# alpha0 = 0
beta0 = 5 * np.pi / 180  # initial inlination angle (0 < beta0 < pi/2)
# beta0 = 15 * np.pi / 180  # initial inlination angle (0 < beta0 < pi/2)
# Lesaux2005 before (5.8)
# gamma0 = 0

# center of mass, see DMS (22)
x0 = 0
y0 = R - r * np.sin(beta0)
z0 = r * np.cos(beta0)

# initial angles
beta_dot0 = 0  # Lesaux1005 before (5.10)
gamma_dot0_pow2 = (
    4 * (g / r) * np.sin(beta0) / ((6 - 5 * rho * np.sin(beta0)) * rho * np.cos(beta0))
)
gamma_dot0 = sqrt(gamma_dot0_pow2)  # Lesaux2005 (5.12)
alpha_dot0 = -rho * gamma_dot0  # Lesaux2005 (5.11)

# angular velocity, see DMS after (22)
K_Omega0 = np.array(
    [beta_dot0, alpha_dot0 * np.sin(beta0) + gamma_dot0, alpha_dot0 * np.cos(beta0)]
)

# center of mass velocity
# TODO: Derive these equations!
v_S0 = np.array([-R * alpha_dot0 + r * alpha_dot0 * np.sin(beta0), 0, 0])

# RigidBody = RigidBodyQuaternion
# RigidBody = RigidBodyAxisAngle
RigidBody = RigidBodyEuler

# initial conditions
u0 = np.concatenate((v_S0, K_Omega0))
if RigidBody is RigidBodyQuaternion:
    p0 = axis_angle2quat(np.array([1, 0, 0]), beta0)
elif RigidBody is RigidBodyAxisAngle:
    p0 = np.array([1, 0, 0]) * beta0
elif RigidBody is RigidBodyEuler:
    p0 = np.array([0, 1, 0]) * beta0
q0 = np.array((x0, y0, z0, *p0))


class Disc(RigidBody):
    def __init__(self, m, r, q0=None, u0=None):
        A = 1 / 4 * m * r**2
        C = 1 / 2 * m * r**2
        K_theta_S = np.diag(np.array([A, C, A]))

        self.r = r

        super().__init__(m, K_theta_S, q0=q0, u0=u0)

    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.r * np.vstack([np.sin(phi), np.zeros(n), np.cos(phi)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP


# create the model
disc = Disc(m, r, q0, u0)

Constraint = RollingCondition_g_I_Frame_gamma
# Constraint = RollingCondition

rolling = Constraint(disc)
f_g = Force(lambda t: np.array([0, 0, -m * g]), disc)

model = System()
model.add(disc)
model.add(rolling)
model.add(f_g)
model.assemble()


def state():
    """Analytical analysis of the roolling motion of a disc, see Lesaux2005
    Section 5 and 6 and DMS exercise 5.12 (g).

    References
    ==========
    Lesaux2005: https://doi.org/10.1007/s00332-004-0655-4
    """
    t0 = 0
    # t1 = 2 * np.pi / np.abs(alpha_dot0) * 0.01
    t1 = 2 * np.pi / np.abs(alpha_dot0) * 0.25
    # t1 = 2 * np.pi / np.abs(alpha_dot0) * 0.3  # used for GAMM
    # t1 = 2 * np.pi / np.abs(alpha_dot0) * 0.5
    # t1 = 2 * np.pi / np.abs(alpha_dot0) * 1.0
    # dt = 5e-3
    dt = 5e-2
    # dt = 2.5e-2
    # dt = 1.0e-2  # used for GAMM with R = 10 * r
    # dt = 1.0e-3

    # rho_inf = 0.99
    rho_inf = 0.96  # used for GAMM (high oszillations)
    # rho_inf = 0.85  # used for GAMM (low oszillations)
    # rho_inf = 0.1
    # see Arnold2016, p. 118
    tol = 1.0e-10

    # sol = ScipyIVP(model, t1, dt).solve()
    # sol = EulerBackward(model, t1, dt).solve()
    # sol = NonsmoothHalfExplicitRungeKutta(model, t1, dt).solve()
    # sol = GeneralizedAlphaFirstOrder(model, t1, dt, rho_inf=rho_inf, tol=tol).solve()
    # sol = GeneralizedAlphaFirstOrder(
    #     model, t1, dt, rho_inf=rho_inf, tol=tol, GGL=1
    # ).solve()
    # sol = NonsmoothDecoupled(model, t1, dt).solve()
    # sol = NonsmoothHalfExplicitRungeKutta(model, t1, dt).solve()
    # sol = NonsmoothPartitionedHalfExplicitEuler(model, t1, dt).solve()

    # rtol = atol = 1.0e-5
    # # dae_index = 2
    # # dae_index = 3
    # dae_index = "GGL"
    # sol = RadauIIa(model, t1, dt, rtol=rtol, atol=atol, dae_index=dae_index).solve()

    sol = Rattle(model, t1, dt, atol=tol).solve()

    t = sol.t
    q = sol.q
    u = sol.u
    try:
        u_dot = sol.u_dot
        # u_dot = np.zeros_like(u)
        # u_dot[1:] = (u[1:] - u[:-1]) / dt
        la_g = sol.la_g
        la_gamma = sol.la_gamma
    except:
        u_dot = np.zeros_like(u)
        u_dot[1:] = (u[1:] - u[:-1]) / dt
        la_g = sol.P_g
        la_gamma = sol.P_gamma

    g = np.array([model.g(ti, qi) for ti, qi in zip(t, q)])
    g_dot = np.array([model.g_dot(ti, qi, ui) for ti, qi, ui in zip(t, q, u)])
    # g_ddot = np.array(
    #     [model.g_ddot(ti, qi, ui, u_doti) for ti, qi, ui, u_doti in zip(t, q, u, u_dot)]
    # )
    gamma = np.array([model.gamma(ti, qi, ui) for ti, qi, ui in zip(t, q, u)])
    # gamma_dot = np.array(
    #     [
    #         model.gamma_dot(ti, qi, ui, u_doti)
    #         for ti, qi, ui, u_doti in zip(t, q, u, u_dot)
    #     ]
    # )

    ########
    # export
    ########
    # header = "t, x, y, z, p0, p1, p2, p3, la_g, la_ga0, la_ga1"
    # export_data = np.vstack([t, *q.T, *la_g.T, *la_gamma.T]).T
    # np.savetxt(
    #     "examples/GAMM2022/RollingDiscTrajectory.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    # header = "t, g, g_dot, g_ddot, gamma0, gamma1, gamma_dot0, gamma_dot1"
    # export_data = np.vstack([t, *g.T, *g_dot.T, *g_ddot.T, *gamma.T, *gamma_dot.T]).T
    # np.savetxt(
    #     "examples/GAMM2022/RollingDiscGaps.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    ###############
    # visualization
    ###############
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # g
    ax = fig.add_subplot(3, 2, 1)
    ax.plot(t, g)
    ax.set_xlabel("t")
    ax.set_ylabel("g")
    ax.grid()

    # g_dot
    ax = fig.add_subplot(3, 2, 3)
    ax.plot(t, g_dot)
    ax.set_xlabel("t")
    ax.set_ylabel("g_dot")
    ax.grid()

    # # g_ddot
    # ax = fig.add_subplot(3, 2, 5)
    # ax.plot(t, g_ddot)
    # ax.set_xlabel("t")
    # ax.set_ylabel("g_ddot")
    # ax.grid()

    # gamma
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(t, gamma)
    ax.set_xlabel("t")
    ax.set_ylabel("gamma")
    ax.grid()

    # # gamma_dot
    # ax = fig.add_subplot(3, 2, 4)
    # ax.plot(t, gamma_dot)
    # ax.set_xlabel("t")
    # ax.set_ylabel("gamma_dot")
    # ax.grid()

    fig = plt.figure(figsize=plt.figaspect(0.5))

    # trajectory center of mass
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(
        q[:, 0],
        q[:, 1],
        q[:, 2],
        "-r",
        label="x-y-z",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid()
    ax.legend()

    # nonpenetrating contact point
    ax = fig.add_subplot(2, 2, 2)
    if Constraint is RollingCondition_g_I_Frame_gamma:
        ax.plot(t[:], la_g[:, 0], "-r", label="la_g")
    elif Constraint is RollingCondition:
        ax.plot(t[:], la_gamma[:, 0], "--g", label="la_gamma0")
    ax.set_xlabel("t")
    ax.grid()
    ax.legend()

    # no lateral velocities 1
    ax = fig.add_subplot(2, 2, 3)
    if Constraint is RollingCondition_g_I_Frame_gamma:
        ax.plot(t[:], la_gamma[:, 0], "-r", label="la_gamma[0]")
    elif Constraint is RollingCondition:
        ax.plot(t[:], la_gamma[:, 1], "-r", label="la_gamma[1]")
    ax.set_xlabel("t")
    ax.grid()
    ax.legend()

    # no lateral velocities 2
    ax = fig.add_subplot(2, 2, 4)
    if Constraint is RollingCondition_g_I_Frame_gamma:
        ax.plot(t[:], la_gamma[:, 1], "-r", label="la_gamma[1]")
    elif Constraint is RollingCondition:
        ax.plot(t[:], la_gamma[:, 2], "-r", label="la_gamma[2]")
    ax.set_xlabel("t")
    ax.grid()
    ax.legend()

    ########################
    # animate configurations
    ########################
    t = t
    q = q

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    scale = R
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=0, top=2 * scale)

    from collections import deque

    slowmotion = 1
    fps = 200
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
        x_S, y_S, z_S = disc.r_OP(t, q)

        A_IK = disc.A_IK(t, q)
        d1 = A_IK[:, 0] * r
        d2 = A_IK[:, 1] * r
        d3 = A_IK[:, 2] * r

        (COM,) = ax.plot([x_S], [y_S], [z_S], "ok")
        (bdry,) = ax.plot([], [], [], "-k")
        (trace,) = ax.plot([], [], [], "--k")
        (d1_,) = ax.plot(
            [x_S, x_S + d1[0]], [y_S, y_S + d1[1]], [z_S, z_S + d1[2]], "-r"
        )
        (d2_,) = ax.plot(
            [x_S, x_S + d2[0]], [y_S, y_S + d2[1]], [z_S, z_S + d2[2]], "-g"
        )
        (d3_,) = ax.plot(
            [x_S, x_S + d3[0]], [y_S, y_S + d3[1]], [z_S, z_S + d3[2]], "-b"
        )

        return COM, bdry, trace, d1_, d2_, d3_

    COM, bdry, trace, d1_, d2_, d3_ = create(0, q[0])

    def update(t, q, COM, bdry, trace, d1_, d2_, d3_):
        global x_trace, y_trace, z_trace
        if t == t0:
            x_trace = deque([])
            y_trace = deque([])
            z_trace = deque([])

        x_S, y_S, z_S = disc.r_OP(t, q)

        x_bdry, y_bdry, z_bdry = disc.boundary(t, q)

        x_t, y_t, z_t = disc.r_OP(t, q) + rolling.r_SC(t, q)

        x_trace.append(x_t)
        y_trace.append(y_t)
        z_trace.append(z_t)

        A_IK = disc.A_IK(t, q)
        d1 = A_IK[:, 0] * r
        d2 = A_IK[:, 1] * r
        d3 = A_IK[:, 2] * r

        COM.set_data(np.array([x_S]), np.array([y_S]))
        COM.set_3d_properties(np.array([z_S]))

        bdry.set_data(np.array(x_bdry), np.array(y_bdry))
        bdry.set_3d_properties(np.array(z_bdry))

        # if len(x_trace) > 500:
        #     x_trace.popleft()
        #     y_trace.popleft()
        #     z_trace.popleft()
        trace.set_data(np.array(x_trace), np.array(y_trace))
        trace.set_3d_properties(np.array(z_trace))

        d1_.set_data(np.array([x_S, x_S + d1[0]]), np.array([y_S, y_S + d1[1]]))
        d1_.set_3d_properties(np.array([z_S, z_S + d1[2]]))

        d2_.set_data(np.array([x_S, x_S + d2[0]]), np.array([y_S, y_S + d2[1]]))
        d2_.set_3d_properties(np.array([z_S, z_S + d2[2]]))

        d3_.set_data(np.array([x_S, x_S + d3[0]]), np.array([y_S, y_S + d3[1]]))
        d3_.set_3d_properties(np.array([z_S, z_S + d3[2]]))

        return COM, bdry, trace, d1_, d2_, d3_

    def animate(i):
        update(t[i], q[i], COM, bdry, trace, d1_, d2_, d3_)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()


def convergence():
    rho_inf = 0.85
    # see Arnodl2016, p. 118
    tol_ref = 1.0e-10
    tol = 1.0e-10

    #####################################
    # compute step sizes with powers of 2
    #####################################
    # dt_ref = 6.4e-3
    # dts = (2.0 ** np.arange(3, 1, -1)) * dt_ref  # [5.12e-2, ..., 2.56e-2]
    # t1 = (2.0**10) * dt_ref  # 6.5536s

    # dt_ref = 3.2e-3
    # dts = (2.0 ** np.arange(4, 1, -1)) * dt_ref  # [5.12e-2, ..., 1.28e-2]
    # t1 = (2.0**11) * dt_ref  # 6.5536s

    dt_ref = 1.6e-3
    dts = (2.0 ** np.arange(5, 1, -1)) * dt_ref  # [5.12e-2, ..., 6.4e-3]
    t1 = (2.0**12) * dt_ref  # 6.5536s
    # print(f"t1: {t1}")
    # print(f"dts: {dts}")
    # exit()

    # dt_ref = 8e-4
    # dts = (2.0 ** np.arange(6, 1, -1)) * dt_ref  # [5.12e-2, ..., 3.2e-3]
    # t1 = (2.0**13) * dt_ref  # 6.5536s

    # dt_ref = 4e-4
    # dts = (2.0 ** np.arange(7, 1, -1)) * dt_ref  # [5.12e-2, ..., 1.6e-3]
    # t1 = (2.0**14) * dt_ref # 6.5536s

    # # TODO: This is used for GAMM presentation!
    # dt_ref = 2e-4
    # dts = (2.0 ** np.arange(8, 1, -1)) * dt_ref  # [5.12e-2, ..., 8e-4]
    # t1 = (2.0**15) * dt_ref  # 6.5536s

    # # TODO: Why this setup gets killed!
    # dt_ref = 1e-4
    # dts = (2.0 ** np.arange(9, 1, -1)) * dt_ref  # [5.12e-2, ..., 4e-4]
    # t1 = (2.0**16) * dt_ref  # 6.5536s

    # # TODO: Why this setup gets killed!
    # dt_ref = 5e-5
    # dts = (2.0 ** np.arange(10, 1, -1)) * dt_ref  # [5.12e-2, ..., 2e-4]
    # t1 = (2.0**17) * dt_ref # 6.5536s

    # # TODO:
    # # Final version used by Martin
    # dt_ref = 2.5e-5
    # dts = (2.0 ** np.arange(11, 1, -1)) * dt_ref  # [5.12e-2, ..., 1e-4]
    # t1 = (2.0**18) * dt_ref # 6.5536s

    dts_1 = dts
    dts_2 = dts**2

    print(f"t1: {t1}")
    print(f"dts: {dts}")

    # errors for possible solvers
    q_errors_transient = np.inf * np.ones((4, len(dts)), dtype=float)
    u_errors_transient = np.inf * np.ones((4, len(dts)), dtype=float)
    la_g_errors_transient = np.inf * np.ones((4, len(dts)), dtype=float)
    la_gamma_errors_transient = np.inf * np.ones((4, len(dts)), dtype=float)
    q_errors_longterm = np.inf * np.ones((4, len(dts)), dtype=float)
    u_errors_longterm = np.inf * np.ones((4, len(dts)), dtype=float)
    la_g_errors_longterm = np.inf * np.ones((4, len(dts)), dtype=float)
    la_gamma_errors_longterm = np.inf * np.ones((4, len(dts)), dtype=float)

    # #####################
    # # create export files
    # #####################
    # file_transient_q = "examples/GAMM2022/TransientErrorRollingDisc_q.txt"
    # file_transient_u = "examples/GAMM2022/TransientErrorRollingDisc_u.txt"
    # file_transient_la_g = "examples/GAMM2022/TransientErrorRollingDisc_la_g.txt"
    # file_transient_la_gamma = "examples/GAMM2022/TransientErrorRollingDisc_la_gamma.txt"
    # file_longterm_q = "examples/GAMM2022/LongtermErrorRollingDisc_q.txt"
    # file_longterm_u = "examples/GAMM2022/LongtermErrorRollingDisc_u.txt"
    # file_longterm_la_g = "examples/GAMM2022/LongtermErrorRollingDisc_la_g.txt"
    # file_longterm_la_gamma = "examples/GAMM2022/LongtermErrorRollingDisc_la_gamma.txt"
    # header = "dt, dt2, 2nd, 1st, 2nd_GGL, 1st_GGL"

    # def create(name):
    #     with open(name, "w") as file:
    #         file.write(header)
    #     with open(name, "ab") as file:
    #         file.write(b"\n")

    # def append(name, data):
    #     with open(name, "ab") as file:
    #         np.savetxt(
    #             file,
    #             data,
    #             delimiter=", ",
    #             comments="",
    #         )

    # create(file_transient_q)
    # create(file_transient_u)
    # create(file_transient_la_g)
    # create(file_transient_la_gamma)
    # create(file_longterm_q)
    # create(file_longterm_u)
    # create(file_longterm_la_g)
    # create(file_longterm_la_gamma)

    ###################################################################
    # compute reference solution as described in Arnold2015 Section 3.3
    ###################################################################
    # print(f"compute reference solution with second order method:")
    # reference2 = GeneralizedAlphaSecondOrder(
    #     model, t1, dt_ref, rho_inf=rho_inf, tol=tol_ref, GGL=False
    # ).solve()

    print(f"compute reference solution with first order method:")
    # reference1 = GeneralizedAlphaFirstOrder(
    #     model,
    #     t1,
    #     dt_ref,
    #     rho_inf=rho_inf,
    #     tol=tol_ref,
    #     unknowns="velocities",
    #     GGL=False,
    # ).solve()

    reference1 = Rattle(model, t1, dt_ref, atol=tol_ref).solve()

    # print(f"compute reference solution with second order method + GGL:")
    # reference2_GGL = GeneralizedAlphaSecondOrder(
    #     model, t1, dt_ref, rho_inf=rho_inf, tol=tol_ref, GGL=True
    # ).solve()

    # print(f"compute reference solution with first order method + GGL:")
    # reference1_GGL = GeneralizedAlphaFirstOrder(
    #     model, t1, dt_ref, rho_inf=rho_inf, tol=tol_ref, unknowns="velocities", GGL=True
    # ).solve()

    print(f"done")

    # plot_state = True
    plot_state = False
    # TODO:
    if plot_state:
        reference = reference1
        # reference = reference1_GGL
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

    def errors(sol, sol_ref, t_transient=2, t_longterm=2):
        # def errors(sol, sol_ref, t_transient=0.01, t_longterm=0.01):
        # def errors(sol, sol_ref, t_transient=4, t_longterm=4):
        t = sol.t
        q = sol.q
        u = sol.u
        la_g = sol.la_g
        la_gamma = sol.la_gamma

        t_ref = sol_ref.t
        q_ref = sol_ref.q
        u_ref = sol_ref.u
        la_g_ref = sol_ref.la_g
        la_gamma_ref = sol_ref.la_gamma

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
        la_gamma_transient = la_gamma[t_idx_transient]
        diff_transient_q = q_transient - q_ref[t_ref_idx_transient]
        diff_transient_u = u_transient - u_ref[t_ref_idx_transient]
        diff_transient_la_g = la_g_transient - la_g_ref[t_ref_idx_transient]
        diff_transient_la_gamma = la_gamma_transient - la_gamma_ref[t_ref_idx_transient]

        q_longterm = q[t_idx_longterm]
        u_longterm = u[t_idx_longterm]
        la_g_longterm = la_g[t_idx_longterm]
        la_gamma_longterm = la_gamma[t_idx_longterm]
        diff_longterm_q = q_longterm - q_ref[t_ref_idx_longterm]
        diff_longterm_u = u_longterm - u_ref[t_ref_idx_longterm]
        diff_longterm_la_g = la_g_longterm - la_g_ref[t_ref_idx_longterm]
        diff_longterm_la_gamma = la_gamma_longterm - la_gamma_ref[t_ref_idx_longterm]

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
        la_gamma_error_transient = np.max(
            np.linalg.norm(diff_transient_la_gamma, axis=1)
            / np.linalg.norm(la_gamma_transient, axis=1)
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
        la_gamma_error_longterm = np.max(
            np.linalg.norm(diff_longterm_la_gamma, axis=1)
            / np.linalg.norm(la_gamma_longterm, axis=1)
        )

        return (
            q_error_transient,
            u_error_transient,
            la_g_error_transient,
            la_gamma_error_transient,
            q_error_longterm,
            u_error_longterm,
            la_g_error_longterm,
            la_gamma_error_longterm,
        )

    for i, dt in enumerate(dts):
        print(f"i: {i}, dt: {dt:1.1e}")

        # generalized alpha for mechanical systems in second order form
        # sol = GeneralizedAlphaSecondOrder(
        #     model, t1, dt, rho_inf=rho_inf, tol=tol, GGL=False
        # ).solve()
        sol = Rattle(model, t1, dt, atol=tol).solve()
        (
            q_errors_transient[0, i],
            u_errors_transient[0, i],
            la_g_errors_transient[0, i],
            la_gamma_errors_transient[0, i],
            q_errors_longterm[0, i],
            u_errors_longterm[0, i],
            la_g_errors_longterm[0, i],
            la_gamma_errors_longterm[0, i],
        ) = errors(sol, reference1)

        # # generalized alpha for mechanical systems in first order form (velocity formulation)
        # sol = GeneralizedAlphaFirstOrder(
        #     model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="velocities", GGL=False
        # ).solve()
        # (
        #     q_errors_transient[1, i],
        #     u_errors_transient[1, i],
        #     la_g_errors_transient[1, i],
        #     la_gamma_errors_transient[1, i],
        #     q_errors_longterm[1, i],
        #     u_errors_longterm[1, i],
        #     la_g_errors_longterm[1, i],
        #     la_gamma_errors_longterm[1, i],
        # ) = errors(sol, reference1)

        # # generalized alpha for mechanical systems in second order form + GGL
        # sol = GeneralizedAlphaSecondOrder(
        #     model, t1, dt, rho_inf=rho_inf, tol=tol, GGL=True
        # ).solve()
        # (
        #     q_errors_transient[2, i],
        #     u_errors_transient[2, i],
        #     la_g_errors_transient[2, i],
        #     la_gamma_errors_transient[2, i],
        #     q_errors_longterm[2, i],
        #     u_errors_longterm[2, i],
        #     la_g_errors_longterm[2, i],
        #     la_gamma_errors_longterm[2, i],
        # ) = errors(sol, reference2_GGL)

        # # generalized alpha for mechanical systems in first order form (velocity formulation - GGL)
        # sol = GeneralizedAlphaFirstOrder(
        #     model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="velocities", GGL=True
        # ).solve()
        # (
        #     q_errors_transient[3, i],
        #     u_errors_transient[3, i],
        #     la_g_errors_transient[3, i],
        #     la_gamma_errors_transient[3, i],
        #     q_errors_longterm[3, i],
        #     u_errors_longterm[3, i],
        #     la_g_errors_longterm[3, i],
        #     la_gamma_errors_longterm[3, i],
        # ) = errors(sol, reference1_GGL)

        # append(
        #     file_transient_q,
        #     np.array([[dts_1[i], dts_2[i], *q_errors_transient[:, i]]]),
        # )
        # append(
        #     file_transient_u,
        #     np.array([[dts_1[i], dts_2[i], *u_errors_transient[:, i]]]),
        # )
        # append(
        #     file_transient_la_g,
        #     np.array([[dts_1[i], dts_2[i], *la_g_errors_transient[:, i]]]),
        # )
        # append(
        #     file_transient_la_gamma,
        #     np.array([[dts_1[i], dts_2[i], *la_gamma_errors_transient[:, i]]]),
        # )
        # append(
        #     file_longterm_q, np.array([[dts_1[i], dts_2[i], *q_errors_longterm[:, i]]])
        # )
        # append(
        #     file_longterm_u, np.array([[dts_1[i], dts_2[i], *u_errors_longterm[:, i]]])
        # )
        # append(
        #     file_longterm_la_g,
        #     np.array([[dts_1[i], dts_2[i], *la_g_errors_longterm[:, i]]]),
        # )
        # append(
        #     file_longterm_la_gamma,
        #     np.array([[dts_1[i], dts_2[i], *la_gamma_errors_longterm[:, i]]]),
        # )

    ##################
    # visualize errors
    ##################
    fig, ax = plt.subplots(1, 2)

    ax[0].set_title("transient")
    ax[0].loglog(dts, dts_1, "-k", label="dt")
    ax[0].loglog(dts, dts_2, "--k", label="dt^2")
    ax[0].loglog(dts, q_errors_transient[0], "-.ro", label="q")
    ax[0].loglog(dts, u_errors_transient[0], "-.go", label="u")
    ax[0].loglog(dts, la_g_errors_transient[0], "-.bo", label="la_g")
    ax[0].loglog(dts, la_gamma_errors_transient[0], "-.ko", label="la_ga")
    ax[0].grid()
    ax[0].legend()

    ax[1].set_title("long term")
    ax[1].loglog(dts, dts_1, "-k", label="dt")
    ax[1].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1].loglog(dts, q_errors_longterm[0], "-.ro", label="q")
    ax[1].loglog(dts, u_errors_longterm[0], "-.go", label="u")
    ax[1].loglog(dts, la_g_errors_longterm[0], "-.bo", label="la_g")
    ax[1].loglog(dts, la_gamma_errors_longterm[0], "-.ko", label="la_ga")
    ax[1].grid()
    ax[1].legend()

    plt.show()


if __name__ == "__main__":
    # state()
    convergence()
