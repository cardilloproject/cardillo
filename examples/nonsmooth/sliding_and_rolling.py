import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo import System
from cardillo.solver import Moreau, BackwardEuler


class SlidingRollingSphereOnPlane:
    def __init__(self, mass, radius, gravity, mu_T, mu_R, q0, u0):
        self.mass = mass
        self.radius = radius
        self.gravity = gravity
        self.Theta_S = 2 / 5 * mass * radius**2

        self.nq = 3
        self.nu = 3
        self.q0 = q0
        self.u0 = u0
        assert self.nq == len(q0)
        assert self.nu == len(u0)

        self.nla_N = 1
        self.nla_F = 2
        self.NF_connectivity = [[0, 1]]
        from cardillo.math.prox import Sphere

        # fmt: off
        self.NF_connectivity2 = [
            # (i_N, i_F, radius, force_reservoir), # prototype
            ([0], [0], Sphere(mu_T)), # Coulomb
            # ([0], [1], Hypersphere(mu_R)), # rolling with normal force coupling
            ([], [1], Sphere(mu_R * mass * gravity)), # rolling with constant normal force
        ]
        # fmt: on
        # self.mu = np.array([mu_T, mu_R])
        self.mu = np.array([mu_T, mu_R * mass * gravity])
        self.my = np.ones(2)
        self.e_N = np.zeros(self.nla_N)
        self.e_F = np.zeros(self.nla_F)

    #####################
    # kinematic equations
    #####################
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
        return np.diag([self.mass, self.mass, self.Theta_S])

    def h(self, t, q, u):
        return np.array([0, -self.gravity * self.mass, 0])

    #################
    # normal contacts
    #################
    def g_N(self, t, q):
        return np.array([q[1] - self.radius])

    def g_N_dot(self, t, q, u):
        return np.array([u[1]])

    def g_N_ddot(self, t, q, u, u_dot):
        return np.array([u_dot[1]])

    def g_N_q(self, t, q):
        g_N_q = np.zeros((self.nla_N, self.nq), dtype=q.dtype)
        g_N_q[0, 1] = 1
        return g_N_q

    def W_N(self, t, q):
        return self.g_N_q(t, q).T

    def Wla_N_q(self, t, q, la_N):
        return np.zeros((self.nu, self.nq))

    ##########
    # friction
    ##########
    def gamma_F(self, t, q, u):
        return np.array([u[0] - self.radius * u[2], u[2]])

    def gamma_F_u(self, t, q):
        gamma_F_u = np.zeros((self.nla_F, self.nu), dtype=q.dtype)
        gamma_F_u[0, 0] = 1
        gamma_F_u[0, 2] = -self.radius
        gamma_F_u[1, 2] = 1
        return gamma_F_u

    def gamma_F_dot(self, t, q, u, u_dot):
        return np.array([u_dot[0] - self.radius * u_dot[2], u_dot[2]])

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def Wla_F_q(self, t, q, la_F):
        return np.zeros((self.nu, self.nq))

    ###############
    # visualization
    ###############
    def boundary(self, t, q, num=100):
        x, y, phi = q

        def A_IK(theta):
            # fmt: off
            return np.array([
                [ np.cos(phi + theta), np.sin(phi + theta)], 
                [-np.sin(phi + theta), np.cos(phi + theta)]
            ])
            # fmt: on

        phis = np.linspace(0, 2 * np.pi, num=num, endpoint=True)

        r_OS = np.array([x, y])
        r_OPs = np.array(
            [r_OS + A_IK(phi) @ np.array([self.radius, 0]) for phi in phis]
        ).T
        return np.concatenate((r_OS[:, None], r_OPs), axis=-1)


if __name__ == "__main__":
    mass = 1
    radius = 0.1
    gravity = 9.81
    mu_T = 0.1
    mu_R = 0.005
    # mu_R = 1e-12

    q0 = np.array([0, radius, 0])
    x_dot0 = 1
    # omega0 = -10
    omega0 = 0
    u0 = np.array([x_dot0, 0, omega0])

    sphere = SlidingRollingSphereOnPlane(mass, radius, gravity, mu_T, mu_R, q0, u0)

    #####################
    # analytical solution
    #####################
    t_star = (x_dot0 - radius * omega0) / (
        mu_T * gravity
        + (radius * mu_T - mu_R) * radius * mass * gravity / sphere.Theta_S
    )
    t_until_rolling = np.linspace(0, t_star, num=100)
    x_dot_until_rolling = x_dot0 - mu_T * gravity * t_until_rolling
    phi_dot_until_rolling = (
        omega0
        + (radius * mu_T - mu_R) * (mass * gravity / sphere.Theta_S) * t_until_rolling
    )

    phi_dot_rolling0 = phi_dot_until_rolling[-1]
    t_starstar = t_star + phi_dot_rolling0 * (sphere.Theta_S + mass * radius**2) / (
        mu_R * mass * gravity
    )

    # # c = mu_R * mass * gravity * t_star / sphere.Theta_S + (x_dot0 - mu_T * gravity * t_star) / radius
    # # c = phi_dot_until_rolling[-1] #+ mu_R * mass * gravity * t_star / sphere.Theta_S
    # # t_starstar = t_star + (c * sphere.Theta_S) / (mu_R * mass * gravity)
    # t_starstar = t_star + (max(phi_dot_until_rolling) * sphere.Theta_S) / (mu_R * mass * gravity)
    # # t_starstar = sphere.Theta_S  / (mu_R * mass * gravity) * (
    # #     mu_R * mass * gravity * t_star / sphere.Theta_S + (x_dot0 - mu_T * gravity * t_star) / radius
    # # )

    print(f"t_starstar: {t_starstar}")
    print(f"x_dot(t_star): {x_dot0 - mu_T * gravity * t_star}")
    print(f"phi_dot(t_star): {(x_dot0 - mu_T * gravity * t_star) / radius}")
    # exit()

    system = System()
    system.add(sphere)
    system.assemble()

    t_final = 2.5
    dt1 = 5e-2
    dt2 = 5e-2

    sol1, label1 = (
        BackwardEuler(system, t_final, dt1).solve(),
        "BackwardEuler",
    )

    sol2, label2 = (
        Moreau(system, t_final, dt2).solve(),
        "Moreau",
    )

    ####################
    # visualize solution
    ####################
    t1 = sol1.t
    q1 = sol1.q
    u1 = sol1.u
    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u

    # omega0 = 0.4
    # t_stick = 0.408
    # ts = np.linspace(0, t_stick, num=100)
    # omega_analytic = omega0 - omega0 / t_stick * ts
    # alpha_analytic = omega0 * ts - omega0 / (2 * t_stick) * ts**2

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("x [m]")
    # ax[0, 0].plot(ts, alpha_analytic, "-k", label="analytic")
    # ax[0, 0].plot([t_stick, t_final], [alpha_analytic[-1], alpha_analytic[-1]], "-k")
    ax[0, 0].plot([t_star, t_star], [min(q1[:, 0]), max(q1[:, 0])], "--k")
    ax[0, 0].plot([t_starstar, t_starstar], [min(q1[:, 0]), max(q1[:, 0])], "--k")
    ax[0, 0].plot(t1, q1[:, 0], "--b", label=label1)
    ax[0, 0].plot(t2, q2[:, 0], ":r", label=label2)
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel("x_dot [m / s]")
    # ax[1, 0].plot(ts, omega_analytic, "-k", label="analytic")
    # ax[1, 0].plot([t_stick, t_final], [omega_analytic[-1], omega_analytic[-1]], "-k")
    ax[1, 0].plot([t_star, t_star], [min(u1[:, 0]), max(u1[:, 0])], "--k")
    ax[1, 0].plot([t_starstar, t_starstar], [min(u1[:, 0]), max(u1[:, 0])], "--k")
    ax[1, 0].plot(t_until_rolling, x_dot_until_rolling, "-k")
    ax[1, 0].plot(t1, u1[:, 0], "--b", label=label1)
    ax[1, 0].plot(t2, u2[:, 0], ":r", label=label2)
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("phi [rad]")
    # ax[0, 1].plot(ts, alpha_analytic, "-k", label="analytic")
    ax[0, 1].plot([t_star, t_star], [min(q1[:, 2]), max(q1[:, 2])], "--k")
    ax[0, 1].plot([t_starstar, t_starstar], [min(q1[:, 2]), max(q1[:, 2])], "--k")
    ax[0, 1].plot(t1, q1[:, 2], "--b", label=label1)
    ax[0, 1].plot(t2, q2[:, 2], ":r", label=label2)
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].set_ylabel("phi_dot [rad / s]")
    # ax[1, 1].plot(ts, omega_analytic, "-k", label="analytic")
    # ax[1, 1].plot([t_stick, t_final], [omega_analytic[-1], omega_analytic[-1]], "-k")
    ax[1, 1].plot([t_star, t_star], [min(u1[:, 2]), max(u1[:, 2])], "--k")
    ax[1, 1].plot([t_starstar, t_starstar], [min(u1[:, 2]), max(u1[:, 2])], "--k")
    ax[1, 1].plot(t_until_rolling, phi_dot_until_rolling, "-k")
    ax[1, 1].plot(t1, u1[:, 2], "--b", label=label1)
    ax[1, 1].plot(t2, u2[:, 2], ":r", label=label2)
    ax[1, 1].grid()
    ax[1, 1].legend()

    ###########
    # animation
    ###########
    t = t1
    q = q1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    width = 1
    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)

    # prepare data for animation
    frames = len(t)
    target_frames = min(len(t), 200)
    frac = int(frames / target_frames)
    animation_time = 5
    interval = animation_time * 1000 / target_frames

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    # horizontal plane
    ax.plot([-2 * width, 2 * width], [0, 0], "-k")

    def create(t, q):
        (COM,) = ax.plot([], [], "ok")
        (bdry,) = ax.plot([], [], "-k")
        (d1_,) = ax.plot([], [], "-r")
        (d2_,) = ax.plot([], [], "-g")
        return COM, bdry, d1_, d2_

    COM, bdry, d1_, d2_ = create(0, q[0])

    def update(t, q, COM, bdry, d1_, d2_):
        x_S, y_S, phi = q
        # d1 = np.array([np.cos(phi), np.sin(phi), 0]) * radius
        # d2 = np.array([-np.sin(phi), np.cos(phi), 0]) * radius
        d1 = np.array([np.cos(phi), -np.sin(phi), 0]) * radius
        d2 = np.array([np.sin(phi), np.cos(phi), 0]) * radius

        x_bdry, y_bdry = sphere.boundary(t, q)

        COM.set_data([x_S], [y_S])
        bdry.set_data(x_bdry, y_bdry)

        d1_.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
        d2_.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])

        return COM, bdry, d1_, d2_

    def animate(i):
        update(t[i], q[i], COM, bdry, d1_, d2_)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()
