import numpy as np
import matplotlib.pyplot as plt

from cardillo.math.prox import Sphere
from cardillo import System
from cardillo.solver import Moreau, BackwardEuler


class FrictionBelt:
    def __init__(self, q0, u0):
        self.mass = 1
        self.gravity = 10
        self.la_N = self.mass * self.gravity
        self.stiffness = 1
        self.damping = 0
        self.delta = 3
        self.u_b = 3
        self.mu_s = 0.5
        self.mu_d = 0.3

        self.nq = 1
        self.nu = 1
        self.q0 = q0
        self.u0 = u0
        assert self.nq == len(q0)
        assert self.nu == len(u0)

        # add friction force with constant force recervoir
        # fmt: off
        self.friction_laws = [
            ([], [0], Sphere(self.mu_s * self.la_N)),
        ]
        # fmt: on
        self.nla_F = 1
        self.e_F = np.zeros(self.nla_F)

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
        return np.diag([self.mass])

    def h(self, t, q, u):
        return np.array(
            [
                -self.stiffness * q[0]
                - self.damping * u[0]
                + (self.mu_s - self.mu_d)
                * self.la_N
                * (2 / np.pi)
                * np.arctan(self.delta * (u[0] - self.u_b))
            ]
        )

    def h_q(self, t, q, u):
        return np.array([[-self.stiffness]])

    def h_q(self, t, q, u):
        return np.array(
            [
                [
                    -self.damping
                    + (self.mu_s - self.mu_d)
                    * self.la_N
                    * (2 / np.pi)
                    / (1 + (self.delta * (u[0] - self.u_b)) ** 2)
                ]
            ]
        )

    ##########
    # friction
    ##########
    def gamma_F(self, t, q, u):
        return np.array([u[0] - self.u_b])

    def gamma_F_u(self, t, q):
        gamma_F_u = np.zeros((self.nla_F, self.nu), dtype=q.dtype)
        gamma_F_u[0, 0] = 1
        return gamma_F_u

    def gamma_F_dot(self, t, q, u, u_dot):
        return np.array([u_dot[0]])

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def Wla_F_q(self, t, q, la_F):
        return np.zeros((self.nu, self.nq))


if __name__ == "__main__":
    x0 = 0.2
    x_dot0 = 0.2
    q0 = np.array([x0], dtype=float)
    u0 = np.array([x_dot0], dtype=float)

    system = System()
    block_on_belt = FrictionBelt(q0, u0)
    system.add(block_on_belt)

    system.assemble()

    t_final = 2e1
    dt = 1e-2
    sol1, label1 = Moreau(system, t_final, dt).solve(), "Moreau"
    sol2, label2 = BackwardEuler(system, t_final, dt).solve(), "BackwardEuler"

    t1 = sol1.t
    q1 = sol1.q
    u1 = sol1.u
    P_N1 = sol1.P_N
    P_F1 = sol1.P_F

    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u
    P_N2 = sol2.P_N
    P_F2 = sol2.P_F

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].set_title("x(t)")
    ax[0, 0].plot(t1, q1[:, 0], "-k", label=label1)
    ax[0, 0].plot(t2, q2[:, 0], "--r", label=label2)
    ax[0, 0].legend()

    ax[1, 0].set_title("u_x(t)")
    ax[1, 0].plot(t1, u1[:, 0], "-k", label=label1)
    ax[1, 0].plot(t2, u2[:, 0], "--r", label=label2)
    ax[1, 0].legend()

    ax[0, 1].set_title("x_dot(x)")
    ax[0, 1].plot(q1[:, 0], u1[:, 0], "-k", label=label1)
    ax[0, 1].plot(q2[:, 0], u2[:, 0], "--r", label=label2)
    ax[0, 1].legend()

    ax[1, 1].set_title("P_F(t)")
    ax[1, 1].plot(t1, P_F1[:, 0], "-k", label=label1)
    ax[1, 1].plot(t2, P_F2[:, 0], "--r", label=label2)
    ax[1, 1].legend()

    plt.show()
