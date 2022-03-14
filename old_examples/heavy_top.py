import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from cardillo.math.algebra import (
    inv3D,
    A_IK_basic_x,
    A_IK_basic_y,
    A_IK_basic_z,
    cross3,
    axis_angle2quat,
    ax2skew,
)
from scipy.integrate import solve_ivp

from cardillo.model import Model
from cardillo.model.rigid_body import (
    Rigid_body_euler,
    Rigid_body_quaternion,
    Rigid_body_director,
)
from cardillo.model.bilateral_constraints.implicit import Spherical_joint
from cardillo.model.frame import Frame
from cardillo.model.force import Force
from cardillo.solver import Scipy_ivp, Generalized_alpha_1, Moreau_sym


class Heavy_top:
    def __init__(self, m, r, L):
        self.m = m
        self.r = r
        self.L = L
        self.A = 1 / 2 * m * r**2
        self.B = 1 / 4 * m * r**2

    def A_IK(self, t, q):
        A_IB = A_IK_basic_z(q[0])
        A_BC = A_IK_basic_y(-q[1])
        A_CK = A_IK_basic_x(q[2])
        return A_IB @ A_BC @ A_CK

    def r_OP(self, t, q, K_r_SP=np.zeros(3)):
        A_IK = self.A_IK(t, q)
        r_OS = A_IK @ np.array([self.L, 0, 0])
        return r_OS + A_IK @ K_r_SP

    def eqm(self, t, x):
        dx = np.zeros(6)
        beta = x[1]
        gamma = x[2]
        omega_x = x[3]
        omega_y = x[4]
        omega_z = x[5]

        g = 9.81

        m = self.m
        L = self.L
        A = self.A
        B = self.B

        Q = np.array(
            [
                [sin(beta), 0, 1],
                [cos(beta) * sin(gamma), -cos(gamma), 0],
                [cos(beta) * cos(gamma), sin(gamma), 0],
            ]
        )

        M = np.diag(np.array([A, B + m * L**2, B + m * L**2]))

        h = np.array(
            [
                0,
                (B + m * L**2 - A) * omega_x * omega_z
                + m * g * L * cos(beta) * cos(gamma),
                -(B + m * L**2 - A) * omega_x * omega_y
                - m * g * L * cos(beta) * sin(gamma),
            ]
        )

        dx[:3] = inv3D(Q) @ x[3:]
        dx[3:] = inv3D(M) @ h
        return dx

    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.r * np.vstack([np.zeros(n), np.sin(phi), np.cos(phi)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP


class Heavy_top_euler(Rigid_body_euler):
    def __init__(self, m, r, axis="zxy", q0=None, u0=None):
        A = 1 / 2 * m * r**2
        B = 1 / 4 * m * r**2
        K_theta_S = np.diag(np.array([A, B, B]))

        self.r = r

        super().__init__(m, K_theta_S, axis=axis, q0=q0, u0=u0)

    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.r * np.vstack([np.zeros(n), np.sin(phi), np.cos(phi)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP


class Heavy_top_quaternion(Rigid_body_quaternion):
    def __init__(self, m, r, q0=None, u0=None):
        A = 1 / 2 * m * r**2
        B = 1 / 4 * m * r**2
        K_theta_S = np.diag(np.array([A, B, B]))

        self.r = r

        super().__init__(m, K_theta_S, q0=q0, u0=u0)

    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.r * np.vstack([np.zeros(n), np.sin(phi), np.cos(phi)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP


class Heavy_top_director(Rigid_body_director):
    def __init__(self, m, r, q0=None, u0=None):
        A = 1 / 2 * m * r**2
        B = 1 / 4 * m * r**2
        K_theta_S = np.diag(np.array([A, B, B]))

        self.r = r

        I11 = K_theta_S[0, 0]
        I22 = K_theta_S[1, 1]
        I33 = K_theta_S[2, 2]

        # Binet inertia tensor
        i11 = 0.5 * (I22 + I33 - I11)
        i22 = 0.5 * (I11 + I33 - I22)
        i33 = 0.5 * (I11 + I22 - I33)
        B_rho0 = np.zeros(3)
        C_rho0 = np.diag(np.array([i11, i22, i33]))

        super().__init__(m, B_rho0, C_rho0, q0=q0, u0=u0)

    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.r * np.vstack([np.zeros(n), np.sin(phi), np.cos(phi)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP


def comparison_heavy_top(
    t1=1, rigid_body="Euler", plot_graphs=True, animate=True, animate_ref=False
):
    t0 = 0
    dt = 1e-4

    m = 0.1
    L = 0.2
    g = 9.81
    r = 0.1
    Omega = 2 * pi * 50

    # reference solution
    heavy_top = Heavy_top(m, r, L)

    alpha0 = 0
    beta0 = pi / 10
    gamma0 = 0

    omega_x0 = Omega
    omega_y0 = 0
    omega_z0 = 0

    x0 = np.array([alpha0, beta0, gamma0, omega_x0, omega_y0, omega_z0])
    ref = solve_ivp(
        heavy_top.eqm,
        [t0, t1],
        x0,
        method="RK45",
        t_eval=np.arange(t0, t1 + dt, dt),
        rtol=1e-8,
        atol=1e-12,
    )
    t_ref = ref.t
    q_ref = ref.y[:3].T

    # solutions with cardillo models
    r_OS0 = heavy_top.r_OP(t0, np.array([alpha0, beta0, gamma0]))
    A_IK0 = heavy_top.A_IK(t0, np.array([alpha0, beta0, gamma0]))

    K_Omega0 = np.array([omega_x0, omega_y0, omega_z0])
    v_S0 = cross3(A_IK0 @ K_Omega0, r_OS0)

    if rigid_body == "Euler":
        p0 = np.array([alpha0, -beta0, gamma0])
        q0 = np.concatenate([r_OS0, p0])
        u0 = np.concatenate([v_S0, K_Omega0])
        RB = Heavy_top_euler(m, r, axis="zyx", q0=q0, u0=u0)
    elif rigid_body == "Quaternion":
        p0 = axis_angle2quat(np.array([0, 1, 0]), -beta0)
        q0 = np.concatenate([r_OS0, p0])
        u0 = np.concatenate([v_S0, K_Omega0])
        RB = Heavy_top_quaternion(m, r, q0=q0, u0=u0)
    elif rigid_body == "Director":
        R0 = A_IK0
        p0 = np.concatenate((R0[:, 0], R0[:, 1], R0[:, 2]))
        q0 = np.concatenate([r_OS0, p0])
        omega0_tilde = R0 @ ax2skew(K_Omega0)
        u0 = np.concatenate(
            (v_S0, omega0_tilde[:, 0], omega0_tilde[:, 1], omega0_tilde[:, 2])
        )
        RB = Heavy_top_director(m, r, q0=q0, u0=u0)

    origin = Frame()
    joint = Spherical_joint(origin, RB, np.zeros(3))

    model = Model()
    model.add(origin)
    model.add(RB)
    model.add(joint)
    model.add(Force(lambda t: np.array([0, 0, -g * m]), RB))

    model.assemble()

    solver = Scipy_ivp(model, t1, dt, rtol=1e-6, atol=1.0e-7)

    sol = solver.solve()
    t = sol.t
    q = sol.q

    if plot_graphs:
        x_ref_ = []
        y_ref_ = []
        for i, ti in enumerate(t_ref):
            x_ref_.append(heavy_top.r_OP(ti, q_ref[i])[0])
            y_ref_.append(heavy_top.r_OP(ti, q_ref[i])[1])

        x_ = []
        y_ = []
        for i, ti in enumerate(t):
            x_.append(RB.r_OP(ti, q[i])[0])
            y_.append(RB.r_OP(ti, q[i])[1])

        plt.plot(x_ref_, y_ref_, "-b")
        plt.plot(x_, y_, "xb")
        scale_ = 1.2 * L
        plt.xlim(-scale_, scale_)
        plt.ylim(-scale_, scale_)
        plt.axis("equal")
        plt.xlabel("x_S [m]")
        plt.ylabel("y_S [m]")
        plt.show()

    if animate:
        if animate_ref:
            t = t_ref
            q = q_ref
            RB = heavy_top

        # animate configurations
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        scale = L
        ax.set_xlim3d(left=-scale, right=scale)
        ax.set_ylim3d(bottom=-scale, top=scale)
        ax.set_zlim3d(bottom=-scale, top=scale)

        # prepare data for animation
        frames = len(t)
        target_frames = min(frames, 100)
        frac = int(frames / target_frames)
        animation_time = t1 - t0
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t[::frac]
        q = q[::frac]

        def create(t, q):
            x_0, y_0, z_0 = np.zeros(3)
            x_S, y_S, z_S = RB.r_OP(t, q)

            A_IK = RB.A_IK(t, q)
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
            d3 = A_IK[:, 2] * r

            (COM,) = ax.plot([x_0, x_S], [y_0, y_S], [z_0, z_S], "-ok")
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

            return COM, bdry, d1_, d2_, d3_

        COM, bdry, d1_, d2_, d3_ = create(0, q[0])

        def update(t, q, COM, bdry, d1_, d2_, d3_):
            x_0, y_0, z_0 = np.zeros(3)
            x_S, y_S, z_S = RB.r_OP(t, q)

            x_bdry, y_bdry, z_bdry = RB.boundary(t, q)

            A_IK = RB.A_IK(t, q)
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
            d3 = A_IK[:, 2] * r

            COM.set_data([x_0, x_S], [y_0, y_S])
            COM.set_3d_properties([z_0, z_S])

            bdry.set_data(x_bdry, y_bdry)
            bdry.set_3d_properties(z_bdry)

            d1_.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
            d1_.set_3d_properties([z_S, z_S + d1[2]])

            d2_.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])
            d2_.set_3d_properties([z_S, z_S + d2[2]])

            d3_.set_data([x_S, x_S + d3[0]], [y_S, y_S + d3[1]])
            d3_.set_3d_properties([z_S, z_S + d3[2]])

            return COM, bdry, d1_, d2_, d3_

        def animate(i):
            update(t[i], q[i], COM, bdry, d1_, d2_, d3_)

        anim = animation.FuncAnimation(
            fig, animate, frames=frames, interval=interval, blit=False
        )
        plt.show()


if __name__ == "__main__":
    comparison_heavy_top(t1=5, rigid_body="Director", animate=True, animate_ref=True)
