import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math.approx_fprime import approx_fprime
from cardillo import System
from cardillo.solver import (
    Moreau,
    BackwardEuler,
    SolverOptions,
)


def error():
    def f(phi, ab):
        return (np.cos(phi) ** 2 / (np.cos(phi) ** 2 / ab + np.sin(phi) ** 2 * ab)) / (
            1 + phi**2
        ) - 1

    num = 500
    phi = np.linspace(-np.pi / 2, np.pi / 2, num=num)
    ab = np.linspace(0.1, 10, num=num)

    X, Y = np.meshgrid(phi, ab)
    zs = np.array(f(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel("phi")
    ax.set_ylabel("a/b")
    ax.set_zlabel("error g_dot - gamma")
    plt.show()


def sim():
    class EllipseContact:
        def __init__(self, a, b, q0, u0, simplified):
            self.simplified = simplified
            self.mass = 1
            self.a = a
            self.b = b
            self.Theta_S = self.mass * (self.a**2 + self.b**2) / 4
            self.gravity = 9.81

            self.nq = 3
            self.nu = 3
            self.q0 = q0
            self.u0 = u0
            assert self.nq == len(q0)
            assert self.nu == len(u0)

            self.nla_N = 1
            self.e_N = np.zeros(self.nla_N)

        #####################
        # kinematic equations
        #####################
        def q_dot(self, t, q, u):
            return u

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
        def alpha(self, phi):
            # return np.arctan(self.a * np.sin(phi) / self.b * np.cos(phi)) - phi
            return np.arctan2(self.a * np.sin(phi), self.b * np.cos(phi)) - phi

        def g_N(self, t, q):
            x, y, phi = q
            alpha = self.alpha(phi)
            sp, cp = np.sin(phi), np.cos(phi)
            sap, cap = np.sin(alpha + phi), np.cos(alpha + phi)
            g_N = y - self.a * sap * sp - self.b * cap * cp
            return np.array([g_N])

        def g_N_dot(self, t, q, u):
            x, y, phi = q
            x_dot, y_dot, phi_dot = u

            alpha = self.alpha(phi)
            # print(f"alpha: {alpha}")
            # print(f"phi  : {phi}")
            sp, cp = np.sin(phi), np.cos(phi)
            sap, cap = np.sin(alpha + phi), np.cos(alpha + phi)

            g_N_dot_simplified = (
                y_dot
                + (
                    -self.a * sap * cp
                    + self.b * cap * sp
                    - self.a * cap * sp
                    + self.b * sap * cp
                )
                * phi_dot
            )
            # note: the -1 is already put into g_N_dot
            # alpha_phi = (self.a / self.b) / (1 + (self.a**2 / self.b**2) * np.tan(phi)**2) / (1 + phi**2)
            alpha_phi = (
                cp**2
                / ((self.b / self.a) * cp**2 + (self.a / self.b) * sp**2)
                / (1 + phi**2)
            )
            # print(f"alpha_phi: {alpha_phi}")
            g_N_dot = (
                y_dot
                + (
                    -self.a * sap * cp
                    + self.b * cap * sp
                    # note: this term in the bracket is always zero!
                    + (-self.a * cap * sp + self.b * sap * cp) * alpha_phi
                )
                * phi_dot
            )

            # bracket = (-self.a * cap * sp + self.b * sap * cp)
            # print(f"bracket: {bracket}")

            # g_N_dot_num = approx_fprime(q, lambda q: self.g_N(t, q)) @ self.q_dot(t, q, u)
            # print(f"g_N_dot - g_N_dot_num: {g_N_dot - g_N_dot_num}")

            # print(f"g_N_dot - g_N_dot_simplified: {g_N_dot - g_N_dot_simplified}")
            if self.simplified:
                return np.array([g_N_dot_simplified])
            else:
                return np.array([g_N_dot])

        def g_N_q(self, t, q):
            return approx_fprime(q, lambda q: self.g_N(t, q)).reshape(
                (self.nla_N, self.nq)
            )

        def W_N(self, t, q):
            W_N = (
                # approx_fprime(np.zeros(self.nu), lambda u: self.g_N_dot(t, q, u))
                approx_fprime(np.random.rand(self.nu), lambda u: self.g_N_dot(t, q, u))
                .reshape((self.nla_N, self.nu))
                .T
            )
            # print(f"W_N:\n{W_N}")
            return W_N

        def Wla_N_q(self, t, q, la_N):
            return approx_fprime(q, lambda q: self.W_N(t, q) @ la_N)

        ###############
        # visualization
        ###############
        def boundary(self, t, q, num=100):
            x, y, phi = q[self.qDOF]
            # fmt: off
            A_IK = np.array([
                [np.cos(phi), -np.sin(phi)], 
                [np.sin(phi),  np.cos(phi)]
            ])
            # fmt: on

            thetas = np.linspace(0, 2 * np.pi, num=num, endpoint=True)
            r_SP = A_IK @ np.array([self.a * np.sin(thetas), self.b * np.cos(thetas)])
            r_OC = np.array([x, y])
            r_OPs = r_OC[:, None] + r_SP

            return np.concatenate((r_OC[:, None], r_OPs), axis=1)

    # a = 1
    # b = 0.1
    # a = 1
    # b = 2
    a = 0.1
    b = 1
    x0 = 0
    y0 = 1.1
    phi0 = np.pi / 20  # 0 # np.pi / 2
    q0 = np.array([x0, y0, phi0])
    u0 = np.array([0, 0, 1])

    ellipse_simplified = EllipseContact(a, b, q0, u0, simplified=True)
    ellipse = EllipseContact(a, b, q0, u0, simplified=False)

    system = System()
    system.add(ellipse_simplified)
    system.add(ellipse)
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    t1 = 2
    dt = 1e-2
    sol = Moreau(
        system,
        t1,
        dt,
        options=SolverOptions(compute_consistent_initial_conditions=False),
    ).solve()
    # sol = BackwardEuler(
    #     system,
    #     t1,
    #     dt,
    #     options=SolverOptions(compute_consistent_initial_conditions=False),
    # ).solve()

    t = sol.t
    q = sol.q
    u = sol.u
    P_N = sol.P_N

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(2, 4)

    ax[0, 0].plot(t, q[:, 0], "-k", label="simplified")
    ax[0, 0].plot(t, q[:, 3], "--r", label="exact")
    ax[0, 0].set_xlabel("t")
    ax[0, 0].set_ylabel("x")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].plot(t, u[:, 0], "-k", label="simplified")
    ax[1, 0].plot(t, u[:, 3], "--r", label="exact")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].set_ylabel("x_dot")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[0, 1].plot(t, q[:, 1], "-k", label="simplified")
    ax[0, 1].plot(t, q[:, 4], "--r", label="exact")
    ax[0, 1].set_xlabel("t")
    ax[0, 1].set_ylabel("y")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].plot(t, u[:, 1], "-k", label="simplified")
    ax[1, 1].plot(t, u[:, 4], "--r", label="exact")
    ax[1, 1].set_xlabel("t")
    ax[1, 1].set_ylabel("y_dot")
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[0, 2].plot(t, q[:, 2], "-k", label="simplified")
    ax[0, 2].plot(t, q[:, 5], "--r", label="exact")
    ax[0, 2].set_xlabel("t")
    ax[0, 2].set_ylabel("phi")
    ax[0, 2].grid()
    ax[0, 2].legend()

    ax[1, 2].plot(t, u[:, 2], "-k", label="simplified")
    ax[1, 2].plot(t, u[:, 5], "--r", label="exact")
    ax[1, 2].set_xlabel("t")
    ax[1, 2].set_ylabel("phi_dot")
    ax[1, 2].grid()
    ax[1, 2].legend()

    ax[0, 3].plot(t, P_N[:, 0], "-k", label="simplified")
    ax[0, 3].plot(t, P_N[:, 1], "--r", label="exact")
    ax[0, 3].set_xlabel("t")
    ax[0, 3].set_ylabel("P_N")
    ax[0, 3].grid()
    ax[0, 3].legend()

    ###########
    # animation
    ###########
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    width = 2 * max(x0, y0)
    # ax.set_xlim(-width, width)
    # ax.set_ylim(-width, width)
    ax.axis("equal")

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
        (simplified,) = ax.plot([], [], "-k")
        (correct,) = ax.plot([], [], "--r")
        return simplified, correct

    simplified, correct = create(0, q[0])

    def update(t, q, simplified, correct):
        simplified.set_data(*ellipse_simplified.boundary(t, q))
        correct.set_data(*ellipse.boundary(t, q))

        return simplified, correct

    def animate(i):
        update(t[i], q[i], simplified, correct)

    ax.axis("equal")
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()


if __name__ == "__main__":
    # error()
    sim()
