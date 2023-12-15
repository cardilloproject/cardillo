import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math.prox import Sphere
from cardillo import System
from cardillo.solver import (
    Moreau,
    BackwardEuler,
    SolverOptions,
)


class Reibschwinger:
    def __init__(
        self, mass, damping, stiffness, mu, gravity, amplitude, frequency, q0, u0
    ):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        self.mu = mu
        self.gravity = gravity
        self.amplitude = amplitude
        self.frequency = frequency

        self.nq = 1
        self.nu = 1
        self.q0 = q0
        self.u0 = u0
        assert self.nq == len(q0)
        assert self.nu == len(u0)

        # fmt: off
        self.friction_laws = [
            ([], [0], Sphere(mu * mass * gravity)),
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
                -self.damping * u[0]
                - self.stiffness * q[0]
                + self.amplitude * np.sin(t * self.frequency)
            ]
        )

    ##########
    # friction
    ##########
    def gamma_F(self, t, q, u):
        return np.array([u[0]])

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
    x0 = 10
    x_dot0 = 0
    q0 = np.array([x0], dtype=float)
    u0 = np.array([x_dot0], dtype=float)

    mass = 1.0
    damping = 1e-3
    stiffness = 1.0
    gravity = 1.0
    mu = 1.0
    amplitude = 1.5
    frequency = 1.0
    ball = Reibschwinger(
        mass, damping, stiffness, mu, gravity, amplitude, frequency, q0, u0
    )

    system = System()
    system.add(ball)
    system.assemble()

    t_final = 20
    dt = 1e-2

    # solver, label = Moreau(system, t_final, dt), "Moreau"
    solver, label = (
        BackwardEuler(
            system, t_final, dt, options=SolverOptions(reuse_lu_decomposition=True)
        ),
        "BackwardEuler",
    )

    sol = solver.solve()
    t = sol.t
    q = sol.q
    u = sol.u
    P_F = sol.P_F

    fig, ax = plt.subplots(1, 3)

    ax[0].set_title("x(t)")
    ax[0].set_title("x(t)")
    ax[0].plot(t, q[:, 0], "-k", label=label)
    ax[0].legend()

    ax[1].set_title("u_x(t)")
    ax[1].plot(t, u[:, 0], "-k", label=label)
    ax[1].legend()

    ax[2].set_title("P_F(t)")
    ax[2].plot(t, P_F[:, 0], "-k", label=label)
    ax[2].legend()

    plt.show()
