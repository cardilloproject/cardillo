import numpy as np

from math import sin, cos, sqrt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from cardillo.model import System
from cardillo.model.rolling_disc import Rolling_disc
from cardillo.model.rigid_body import (
    Rigid_body_euler,
    Rigid_body_quaternion,
    Rigid_body_director,
    Rigid_body_director_angular_velocities,
)
from cardillo.model.rolling_disc import Rolling_condition
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import Rod
from cardillo.model.force import Force
from cardillo.solver import (
    Euler_backward,
    Moreau,
    Moreau_sym,
    Scipy_ivp,
    Generalized_alpha_1,
    Generalized_alpha_2,
    Generalized_alpha_3,
    Generalized_alpha_4_index3,
    Generalized_alpha_5,
    Generalized_alpha_4_index1,
)
from cardillo.math.algebra import axis_angle2quat, ax2skew, A_IK_basic_x

# rigid_body = 'Euler'
# rigid_body = 'Quaternion'
rigid_body = "Director"
# rigid_body = 'Director2'


class Rigid_disc_euler(Rigid_body_euler):
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


class Rigid_disc_Lesaux_euler(Rigid_body_euler):
    def __init__(self, m, r, q0=None, u0=None):
        assert m == 0.3048
        assert r == 3.75e-2
        self.r = r
        K_theta_S = np.diag([1.0716e-4, 2.1433e-4, 1.0716e-4])

        super().__init__(m, K_theta_S, q0=q0, u0=u0)

    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.r * np.vstack([np.sin(phi), np.zeros(n), np.cos(phi)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP


class Rigid_disc_quat(Rigid_body_quaternion):
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


class Rigid_disc_Lesaux_quat(Rigid_body_quaternion):
    def __init__(self, m, r, q0=None, u0=None):
        assert m == 0.3048
        assert r == 3.75e-2
        self.r = r
        K_theta_S = np.diag([1.0716e-4, 2.1433e-4, 1.0716e-4])

        super().__init__(m, K_theta_S, q0=q0, u0=u0)

    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.r * np.vstack([np.sin(phi), np.zeros(n), np.cos(phi)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP


class Rigid_disc_Lesaux_director(Rigid_body_director):
    def __init__(self, m, r, q0=None, u0=None):
        assert m == 0.3048
        assert r == 3.75e-2
        self.r = r
        K_theta_S = np.diag([1.0716e-4, 2.1433e-4, 1.0716e-4])

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
        K_r_SP = self.r * np.vstack([np.sin(phi), np.zeros(n), np.cos(phi)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP


class Rigid_disc_Lesaux_director2(Rigid_body_director_angular_velocities):
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


def DMS():
    m = 1
    r = 1
    g = 9.81

    # q0 = np.zeros(5)
    # u0 = np.array([0, 0, 0])
    alpha0 = 0
    beta0 = np.pi / 4
    # q0 = np.array([0, 0, alpha0, beta0, 0])
    # u0 = np.array([5, 0, 0])

    # -----------
    # Initial condition for circular trajectory
    R = 5  # radius of trajectory
    rho = r / R
    gamma_dot = 4 * g * sin(beta0) / ((6 - 5 * rho * sin(beta0)) * rho * r * cos(beta0))
    gamma_dot = sqrt(gamma_dot)
    alpha_dot = -rho * gamma_dot

    u0 = np.array([alpha_dot, 0, gamma_dot])
    q0 = np.array([0, R, alpha0, beta0, 0])
    # ------------

    RD = Rolling_disc(m, r, 9.81, q0=q0, u0=u0)

    model = System()
    model.add(RD)
    model.assemble()
    t0 = 0
    t1 = 10
    dt = 1e-2
    t_span = t0, t1
    solver = Euler_backward(
        model,
        t_span=t_span,
        dt=dt,
        newton_max_iter=50,
        numerical_jacobian=False,
        debug=False,
    )
    t, q, u, la = solver.solve()

    # animate configurations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    scale = 10 * r
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=0, top=2 * scale)

    x_trace = []
    y_trace = []
    z_trace = []

    def init(t, q):
        x_S, y_S, z_S = RD.r_OS(t, q)

        A_IK = RD.A_IK(t, q)
        d1 = A_IK[:, 0]
        d2 = A_IK[:, 1]
        d3 = A_IK[:, 2]

        (COM,) = ax.plot([x_S], [y_S], [z_S], "ok")
        (bdry,) = ax.plot([], [], [], "-k")
        (trace,) = ax.plot([], [], [], "-r")
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

    def update(t, q, COM, bdry, trace, d1_, d2_, d3_):
        x_S, y_S, z_S = RD.r_OS(t, q)

        x_bdry, y_bdry, z_bdry = RD.boundary(t, q)

        x_t, y_t, z_t = RD.r_OA(t, q)

        x_trace.append(x_t)
        y_trace.append(y_t)
        z_trace.append(z_t)

        A_IK = RD.A_IK(t, q)
        d1 = A_IK[:, 0]
        d2 = A_IK[:, 1]
        d3 = A_IK[:, 2]

        COM.set_data([x_S], [y_S])
        COM.set_3d_properties(np.array([z_S]))

        bdry.set_data(x_bdry, y_bdry)
        bdry.set_3d_properties(np.array(z_bdry))

        trace.set_data(x_trace, y_trace)
        trace.set_3d_properties(np.array(z_trace))

        # d1_.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
        # d1_.set_3d_properties([z_S, z_S + d1[2]])

        # d2_.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])
        # d2_.set_3d_properties([z_S, z_S + d2[2]])

        # d3_.set_data([x_S, x_S + d3[0]], [y_S, y_S + d3[1]])
        # d3_.set_3d_properties([z_S, z_S + d3[2]])

        return (
            COM,
            bdry,
            trace,
        )  # d1_, d2_, d3_

    # COM, bdry, trace, d1_, d2_, d3_ = init(0, q[0])
    (
        COM,
        bdry,
        trace,
    ) = init(0, q[0])

    def animate(i):
        update(t[i], q[i], COM, bdry, trace, d1_, d2_, d3_)

    frames = len(t)
    interval = dt * 100
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()

    plt.plot(x_trace, y_trace)
    plt.show()


def rolling_disc_velocity_constraints():
    g = 9.81
    m = 0.3048
    r = 3.75e-2
    R = 0.5  # radius of trajectory
    beta0 = 5 * np.pi / 180

    # -----------
    # Initial condition for circular trajectory
    rho = r / R

    r_S0 = np.array([0, R - r * sin(beta0), r * cos(beta0)])

    if rigid_body == "Euler":
        p0 = np.array([0, beta0, 0])
    elif rigid_body == "Quaternion":
        p0 = axis_angle2quat(np.array([1, 0, 0]), beta0)
    elif rigid_body == "Director" or rigid_body == "Director2":
        R0 = A_IK_basic_x(beta0)
        p0 = np.concatenate((R0[:, 0], R0[:, 1], R0[:, 2]))

    gamma_dot = 4 * g * sin(beta0) / ((6 - 5 * rho * sin(beta0)) * rho * r * cos(beta0))
    gamma_dot = sqrt(gamma_dot)
    alpha_dot = -rho * gamma_dot

    v_S0 = np.array([-R * alpha_dot + r * alpha_dot * sin(beta0), 0, 0])
    omega0 = np.array([0, alpha_dot * sin(beta0) + gamma_dot, alpha_dot * cos(beta0)])
    # omega0 = np.array([0, 1, 0]) * 10
    # v_S0 = np.array([r * omega0[1], 0, 0])

    if rigid_body == "Euler" or rigid_body == "Quaternion" or rigid_body == "Director2":
        u0 = np.concatenate((v_S0, omega0))
    elif rigid_body == "Director":
        omega0_tilde = R0 @ ax2skew(omega0)
        u0 = np.concatenate(
            (v_S0, omega0_tilde[:, 0], omega0_tilde[:, 1], omega0_tilde[:, 2])
        )
    # ------------

    q0 = np.concatenate((r_S0, p0))

    if rigid_body == "Euler":
        RD = Rigid_disc_Lesaux_euler(m, r, q0=q0, u0=u0)
    elif rigid_body == "Quaternion":
        RD = Rigid_disc_Lesaux_quat(m, r, q0=q0, u0=u0)
    elif rigid_body == "Director":
        RD = Rigid_disc_Lesaux_director(m, r, q0=q0, u0=u0)
    elif rigid_body == "Director2":
        RD = Rigid_disc_Lesaux_director2(m, r, q0=q0, u0=u0)

    RC = Rolling_condition(RD)
    f_g = Force(lambda t: np.array([0, 0, -m * g]), RD)

    model = System()
    model.add(RD)
    model.add(RC)
    model.add(f_g)
    model.assemble()

    t0 = 0
    t1 = 2 * np.pi / np.abs(alpha_dot) * 0.25
    dt = 1e-2
    # solver = Euler_backward(model, t1, dt, numerical_jacobian=False, debug=False)
    # solver = Moreau_sym(model, t1, dt)
    # solver = Generalized_alpha_1(model, t1, variable_dt=True, rtol=0, atol=1.0e-2)
    # solver = Generalized_alpha_1(model, t1, dt=dt, variable_dt=False)
    # solver = Generalized_alpha_2(model, t1, dt)
    # solver = Generalized_alpha_3(model, t1, dt, numerical_jacobian=True)
    # solver = Generalized_alpha_3(model, t1, dt, numerical_jacobian=False)
    # solver = Generalized_alpha_4_index3(model, t1, dt)
    solver = Generalized_alpha_4_index1(model, t1, dt, numerical_jacobian=False)
    # solver = Generalized_alpha_5(model, t1, dt, numerical_jacobian=False)
    # solver = Moreau(model, t1, dt)
    # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='RK23')
    # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='RK45')
    # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='DOP853')
    # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='Radau')
    # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='BDF')
    # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='LSODA')
    sol = solver.solve()
    t = sol.t
    q = sol.q

    # animate configurations
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
    fps = 50
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
        x_S, y_S, z_S = RD.r_OP(t, q)

        A_IK = RD.A_IK(t, q)
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
        # def update(t, q, COM, bdry, trace):
        global x_trace, y_trace, z_trace
        if t == t0:
            x_trace = deque([])
            y_trace = deque([])
            z_trace = deque([])

        x_S, y_S, z_S = RD.r_OP(t, q)

        x_bdry, y_bdry, z_bdry = RD.boundary(t, q)

        x_t, y_t, z_t = RD.r_OP(t, q) + RC.r_SA(t, q)

        x_trace.append(x_t)
        y_trace.append(y_t)
        z_trace.append(z_t)

        A_IK = RD.A_IK(t, q)
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

    x_trace = []
    y_trace = []
    z_trace = []
    for i, (t_i, q_i) in enumerate(zip(t, q)):
        x_t, y_t, z_t = RD.r_OP(t_i, q_i) + RC.r_SA(t_i, q_i)
        x_trace.append(x_t)
        y_trace.append(y_t)
        z_trace.append(z_t)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(x_trace, y_trace, "-k")
    ax[1].plot(t, z_trace, "-b")
    ax[0].axis("equal")
    plt.show()


if __name__ == "__main__":
    # DMS()
    rolling_disc_velocity_constraints()
