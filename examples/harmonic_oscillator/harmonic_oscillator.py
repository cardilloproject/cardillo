import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.discrete import Box, PointMass, Frame
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.interactions import TwoPointInteraction
from cardillo.solver import ScipyIVP, MoreauTheta


class Oscillator:
    def __init__(
        self,
        x0=1.0,
        x_dot0=0.0,
    ):
        self.nq = 1
        self.nu = 1
        self.nla_c = 1

        # self.c = self.__c
        # self.c_q = self.__c_q
        # self.c_u = self.__c_u

        self.x0 = x0
        self.x_dot0 = x_dot0
        self.q0 = np.array([x0], dtype=float)
        self.u0 = np.array([x_dot0], dtype=float)

        self.mass = np.pi
        self.k = 1e2
        # self.d = 0.25 * self.k
        self.d = 0.0

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
        return np.eye(self.nu) * self.mass

    #################
    # compliance form
    #################
    # def h(self, t, q, u):
    #     return -(self.k * q + self.d * u)

    def la_c(self, t, q, u):
        return -self.k * q - self.d * u

    def c(self, t, q, u, la_c):
        return la_c / self.k + q + self.d / self.k * u
    
    def c_q(self, t, q, u, la_c):
        return np.eye(self.nq)
    
    def c_u(self, t, q, u, la_c):
        # return np.zeros((self.nla_c, self.nu))
        return np.eye(self.nla_c) * self.d / self.k
    
    def c_la_c(self):
        return np.eye(self.nla_c) / self.k

    def W_c(self, t, q):
        return np.eye(self.nu)

    ###############
    # true solution
    ###############
    def true_sol(self, t):
        delta = self.d / (2 * self.mass)
        omega0 = np.sqrt(self.k / self.mass)
        discriminant = delta**2 - omega0**2


        if discriminant < 0:
            # underdamped
            omega_d = np.sqrt(-discriminant)
            A = self.x0
            B = (self.x_dot0 + delta * self.x0) / omega_d

            envelope = np.exp(-delta * t)
            cos, sin = np.cos(omega_d * t), np.sin(omega_d * t)
            x = envelope * (A * cos + B * sin)
            x_dot = envelope * (
                self.x_dot0 * cos - (delta * self.x_dot0 + omega0**2 * self.x0) / omega_d * sin
            )
            la_c = -self.k * x - self.d * x_dot
            return x, x_dot, la_c
        else:
            raise RuntimeError("discriminant is nonnegative")


if __name__ == "__main__":
    ###################
    # system parameters
    ###################
    m = 1  # mass
    l0 = 1  # undeformed length of spring
    k = 100  # spring stiffness
    d = 2  # damping constant

    # dimensions of boxes
    width = l0
    height = depth = 0.5 * width
    box_dim = np.array([width, height, depth])

    #######################
    # simulation parameters
    #######################
    t0 = 0  # initial time
    t1 = 5  # final time
    dt = 1e-2

    # initial condition
    stretch = 1.9  # initial stretch of spring

    #################
    # assemble system
    #################

    # initialize system
    system = System(t0=t0)

    oscillator = Oscillator()
    system.add(oscillator)
    system.assemble()

    # solver = ScipyIVP(system, t1, dt)
    solver = MoreauTheta(system, t1, dt, theta=0.5)

    sol = solver.solve()
    t, q, u, la_c = sol.t, sol.q, sol.u, sol.la_c

    q_true, u_true, la_c_true = oscillator.true_sol(t)

    fig, ax = plt.subplots(3, 1)

    ax[0].plot(t, q[:, 0], label="q")
    ax[0].plot(t, q_true, "--", label="q true")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, u[:, 0], label="u")
    ax[1].plot(t, u_true, "--", label="u true")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(t, la_c[:, 0], label="la_c")
    ax[2].plot(t, la_c_true, "--", label="la_c true")
    ax[2].grid()
    ax[2].legend()

    plt.show()
