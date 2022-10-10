import numpy as np

from math import pi, sin, cos, sqrt

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cardillo.math.rotations import rodriguez_inv

from cardillo.model import System
from cardillo.model.rigid_body import RigidBodyEuler, RigidBodyAxisAngle
from cardillo.math import A_IK_basic
from cardillo.model.rolling_disc import (
    Rolling_condition,
    Rolling_condition_I_frame,
    Rolling_condition_I_frame_g_gamma,
)
from cardillo.forces import Force
from cardillo.solver import (
    # GenAlphaFirstOrderVelocityGGL,
    # GenAlphaFirstOrderVelocity,
    # GenAlphaDAEAcc,
    GeneralizedAlphaFirstOrder,
    # GenAlphaFirstOrderGGL2_V1,
    # GenAlphaFirstOrderGGL2_V2,
    # GenAlphaFirstOrderGGL2_V3,
)


def rolling_disc_DMS(rigid_body_case="Euler", constraint_case="velocity_K"):
    """Analytical analysis of the roolling motion of a disc, see Lesaux2005
    Section 5 and 6 and DMS exercise 5.12 (g).

    References
    ==========
    Lesaux2005: https://doi.org/10.1007/s00332-004-0655-4
    """
    assert rigid_body_case == "Euler" or rigid_body_case == "AxisAngle"
    ############
    # parameters
    ############
    g = 9.81  # gravity
    m = 0.3048  # disc mass
    # r = 3.75e-2 # disc radius
    r = 0.05  # disc radius
    R = 0.5  # radius of of the circular motion

    # inertia of the disc, Lesaux2005 before (5.3)
    A = B = 0.25 * m * r**2
    C = 0.5 * m * r**2

    # ratio between disc radius and radius of rolling
    rho = r / R  # Lesaux2005 (5.10)

    ####################
    # initial conditions
    ####################
    x0_Lesaux = 0  # Lesaux2005 before (5.8)
    y0_Lesaux = R  # Lesaux2005 (5.8)
    x_dot0_Lesaux = 0  # Lesaux2005 before (5.8)
    y_dot0_Lesaux = 0  # Lesaux2005 (5.8)

    alpha0 = 0
    beta0 = 5 * np.pi / 180  # initial inlination angle (0 < beta0 < pi/2)
    # Lesaux2005 before (5.8)
    gamma0 = 0

    # center of mass, see DMS (22)
    # x0 = x0_Lesaux
    # y0 = y0_Lesaux - r * sin(beta0)
    x0 = 0
    y0 = R - r * sin(beta0)
    z0 = r * cos(beta0)
    r_S0 = np.array([x0, y0, z0])

    # initial angles
    beta_dot0 = 0  # Lesaux1005 before (5.10)
    gamma_dot0_pow2 = (
        4 * (g / r) * sin(beta0) / ((6 - 5 * rho * sin(beta0)) * rho * cos(beta0))
    )
    gamma_dot0 = sqrt(gamma_dot0_pow2)  # Lesaux2005 (5.12)
    alpha_dot0 = -rho * gamma_dot0  # Lesaux2005 (5.11)

    # angular velocity, see DMS after (22)
    K_Omega0 = np.array(
        [beta_dot0, alpha_dot0 * sin(beta0) + gamma_dot0, alpha_dot0 * cos(beta0)]
    )

    # center of mass velocity
    # TODO: Derive these equations!
    v_S0 = np.array([-R * alpha_dot0 + r * alpha_dot0 * sin(beta0), 0, 0])

    class DiscEuler(RigidBodyEuler):
        def __init__(self, m, r, q0=None, u0=None):
            A = 1 / 4 * m * r**2
            C = 1 / 2 * m * r**2
            K_theta_S = np.diag(np.array([A, C, A]))

            self.r = r

            super().__init__(m, K_theta_S, axis="zxy", q0=q0, u0=u0)

        def boundary(self, t, q, n=100):
            phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
            K_r_SP = self.r * np.vstack([np.sin(phi), np.zeros(n), np.cos(phi)])
            return (
                np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP
            )

    class DiscAxisAngle(RigidBodyAxisAngle):
        def __init__(self, m, r, q0=None, u0=None):
            A = 1 / 4 * m * r**2
            C = 1 / 2 * m * r**2
            K_theta_S = np.diag(np.array([A, C, A]))

            self.r = r

            super().__init__(m, K_theta_S, q0=q0, u0=u0)

        def boundary(self, t, q, n=100):
            phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
            K_r_SP = self.r * np.vstack([np.sin(phi), np.zeros(n), np.cos(phi)])
            return (
                np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP
            )

    # initial conditions
    A_IK0 = A_IK_basic(alpha0).z() @ A_IK_basic(beta0).x() @ A_IK_basic(gamma0).y()
    u0 = np.concatenate((v_S0, K_Omega0))
    if rigid_body_case == "Euler":
        q0 = np.array((x0, y0, z0, alpha0, beta0, gamma0))
        disc = DiscEuler(m, r, q0, u0)
    elif rigid_body_case == "AxisAngle":
        psi0 = rodriguez_inv(A_IK0)
        q0 = np.array([x0, y0, z0, *psi0])
        disc = DiscAxisAngle(m, r, q0, u0)
    # elif rigid_body_case == "Quaternion":
    #     p0 = axis_angle2quat(np.array([1, 0, 0]), beta0)
    #     q0 = np.array((x0, y0, z0, *p0))
    #     disc = DiscQuaternion(m, r, q0, u0)
    else:
        raise NotImplementedError("Wrong case chosen!")

    # build model
    if constraint_case == "velocity_K":
        rolling = Rolling_condition(disc)
    elif constraint_case == "velocity_I":
        rolling = Rolling_condition_I_frame(disc)
    elif constraint_case == "g_gamma":
        # TODO: We have to implement g_dot, g_ddot and gamma_dot!
        rolling = Rolling_condition_I_frame_g_gamma(disc)
    else:
        raise NotImplementedError("")
    f_g = Force(lambda t: np.array([0, 0, -m * g]), disc)

    model = System()
    model.add(disc)
    model.add(rolling)
    model.add(f_g)
    model.assemble()

    t0 = 0
    # t1 = 2 * np.pi / np.abs(alpha_dot0) * 0.25
    t1 = 2 * np.pi / np.abs(alpha_dot0) * 1.0
    # dt = 1e-1
    # dt = 5e-2
    # dt = 1e-2
    dt = 5e-3
    # dt = 1e-3
    # dt = 1e-4 # required for second order gen alpha solver
    rho_inf = 0.90
    # rho_inf = 1.0
    tol = 1.0e-8

    # sol_genAlphaFirstOrderVelocity = GenAlphaFirstOrder(
    #     model, t1, dt, rho_inf=rho_inf, tol=tol
    # ).solve()
    sol_genAlphaFirstOrderVelocity = GeneralizedAlphaFirstOrder(
        model, t1, dt, rho_inf=rho_inf, tol=tol, GGL=True
    ).solve()

    # sol_genAlphaFirstOrderVelocity = GenAlphaFirstOrderVelocityGGL(model, t1, dt, rho_inf=rho_inf, tol=tol).solve()
    # sol_genAlphaFirstOrderVelocity = GenAlphaFirstOrderVelocity(
    #     model, t1, dt, rho_inf=rho_inf, tol=tol
    # ).solve()
    # sol_genAlphaFirstOrderVelocity = GenAlphaDAEAcc(model, t1, dt, rho_inf=rho_inf, newton_tol=tol).solve()
    # sol_genAlphaFirstOrderVelocity = Generalized_alpha_3(model, t1, dt, rho_inf=rho_inf, newton_tol=tol, numerical_jacobian=True).solve()

    # solver = GenAlphaFirstOrder(model, t1, dt, tol=1.0e-10, rho_inf=1.0)
    # solver = Euler_backward(model, t1, dt, numerical_jacobian=False, debug=False)
    # solver = Moreau_sym(model, t1, dt)
    # solver = Generalized_alpha_1(model, t1, variable_dt=True, rtol=0, atol=1.0e-2)
    # solver = Generalized_alpha_1(model, t1, dt=dt, variable_dt=False)
    # solver = Generalized_alpha_2(model, t1, dt)
    # solver = Generalized_alpha_3(model, t1, dt, numerical_jacobian=True)
    # solver = Generalized_alpha_3(model, t1, dt, numerical_jacobian=False)
    # solver = Generalized_alpha_4_index3(model, t1, dt)
    # solver = Generalized_alpha_4_index1(model, t1, dt, numerical_jacobian=False)
    # solver = Generalized_alpha_5(model, t1, dt, numerical_jacobian=False)
    # solver = Moreau(model, t1, dt)
    # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='RK23')
    # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='RK45')
    # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='DOP853')
    # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='Radau')
    # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='BDF')
    # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='LSODA')
    # sol = solver.solve()

    t = sol_genAlphaFirstOrderVelocity.t
    q = sol_genAlphaFirstOrderVelocity.q
    u = sol_genAlphaFirstOrderVelocity.u
    q_dot = sol_genAlphaFirstOrderVelocity.q_dot
    u_dot = sol_genAlphaFirstOrderVelocity.u_dot
    la_g = sol_genAlphaFirstOrderVelocity.la_g
    la_gamma = sol_genAlphaFirstOrderVelocity.la_gamma
    # mu_g = sol_genAlphaFirstOrderVelocity.mu_g
    # kappa_g = sol_genAlphaFirstOrderVelocity.kappa_g
    # kappa_gamma = sol_genAlphaFirstOrderVelocity.kappa_gamma
    # a = sol_genAlphaFirstOrderVelocity.a

    ###############
    # visualization
    ###############
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # a vs u_dot
    ax = fig.add_subplot(2, 2, 1)
    for j in range(u_dot.shape[1]):
        # for j in range(a.shape[1]):
        ax.plot(t, u_dot[:, j], "-", label=f"u_dot{j}")
        # ax.plot(t, a[:, j], "--", label=f"a{j}")
    ax.grid()
    ax.legend()

    # # mu_g
    # ax = fig.add_subplot(2, 2, 2)
    # for j in range(mu_g.shape[1]):
    #     ax.plot(t, mu_g[:, j], "-", label=f"mu_g{j}")
    # ax.grid()
    # ax.legend()

    # # kappa_g, kappa_gamma
    # ax = fig.add_subplot(2, 2, 3)
    # for j in range(kappa_g.shape[1]):
    #     ax.plot(t, kappa_g[:, j], "-", label=f"kappa_g{j}")
    # for j in range(kappa_gamma.shape[1]):
    #     ax.plot(t, kappa_gamma[:, j], "-", label=f"kappa_gamma{j}")
    # ax.grid()
    # ax.legend()

    # plt.show()
    # exit()

    fig = plt.figure(figsize=plt.figaspect(0.5))

    # trajectory center of mass
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(
        q[:, 0],
        q[:, 1],
        q[:, 2],
        "-r",
        label="x-y-z - GenAlphaFirstOrderVeclotiy",
    )
    # ax.plot(q_genAlphaFirstOrderVelocityGGL[:, 0], q_genAlphaFirstOrderVelocityGGL[:, 1], q_genAlphaFirstOrderVelocityGGL[:, 2], '--g', label="x-y-z - GenAlphaFirstOrderVeclotiyGGl")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid()
    ax.legend()

    # nonpenetrating contact point
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(
        t[:],
        la_gamma[:, 0],
        "-r",
        label="la_gamma[0] - GenAlphaFirstOrderVeclotiy",
    )
    # ax.plot(t_genAlphaFirstOrderVelocityGGL[:], la_gamma_genAlphaFirstOrderVelocityGGL[:, 0], '--g', label="la_gamma[0] - GenAlphaFirstOrderVeclotiyGGL")
    ax.set_xlabel("t")
    ax.set_ylabel("la_gamma0")
    ax.grid()
    ax.legend()

    # no lateral velocities 1
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(
        t[:],
        la_gamma[:, 1],
        "-r",
        label="la_gamma[1] - GenAlphaFirstOrderVeclotiy",
    )
    # ax.plot(t_genAlphaFirstOrderVelocityGGL[:], la_gamma_genAlphaFirstOrderVelocityGGL[:, 1], '--g', label="la_gamma[1] - GenAlphaFirstOrderVeclotiyGGL")
    ax.set_xlabel("t")
    ax.set_ylabel("la_gamma1")
    ax.grid()
    ax.legend()
    if constraint_case == "velocity_K" or constraint_case == "velocity_I":
        # no lateral velocities 2
        ax = fig.add_subplot(2, 2, 4)
        ax.plot(
            t[:],
            la_gamma[:, 2],
            "-r",
            label="la_gamma[2] - GenAlphaFirstOrderVeclotiy",
        )
        # ax.plot(t_genAlphaFirstOrderVelocityGGL[:], la_gamma_genAlphaFirstOrderVelocityGGL[:, 2], '--g', label="la_gamma[2] - GenAlphaFirstOrderVeclotiyGGL")
        ax.set_xlabel("t")
        ax.set_ylabel("la_gamma2")
        ax.grid()
        ax.legend()
    elif constraint_case == "g_gamma":
        # nonpenetrating contact point
        ax = fig.add_subplot(2, 2, 4)
        ax.plot(
            t[:],
            la_g[:, 0],
            "-r",
            label="la_g - GenAlphaFirstOrderVeclotiy",
        )
        # ax.plot(t_genAlphaFirstOrderVelocityGGL[:], la_g_genAlphaFirstOrderVelocityGGL[:, 0], '--g', label="la_g - GenAlphaFirstOrderVeclotiyGGL")
        ax.set_xlabel("t")
        ax.set_ylabel("la_g")
        ax.grid()
        ax.legend()
    else:
        raise NotImplementedError("")

    # plt.show()

    ########################
    # animate configurations
    ########################
    t = t
    q = q
    # t = t_genAlphaFirstOrderVelocityGGL
    # q = q_genAlphaFirstOrderVelocityGGL

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

    x_trace = deque([])
    y_trace = deque([])
    z_trace = deque([])

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

        x_t, y_t, z_t = disc.r_OP(t, q) + rolling.r_SA(t, q)

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


if __name__ == "__main__":
    rolling_disc_DMS(rigid_body_case="Euler", constraint_case="g_gamma")
    # rolling_disc_DMS(rigid_body_case="Euler", constraint_case="velocity_I")
    # rolling_disc_DMS(rigid_body_case="Euler", constraint_case="velocity_K")
    # rolling_disc_DMS(rigid_body_case="AxisAngle", constraint_case="g_gamma")
