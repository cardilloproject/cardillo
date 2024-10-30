import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.solver import SolverOptions, BackwardEuler


class JenkinsElement:
    """Pointmass with external force connected with Jenkins element to origin."""

    def __init__(self, force, mass, stiffness, yieldstress, l0, q0, u0):
        self.force = force
        self.mass = mass
        self.stiffness = stiffness
        self.yieldstress = yieldstress
        self.l0 = l0
        self.s0 = self.yieldstress
        # self.prox_r = 1 / self.stiffness
        self.prox_r = 1

        # We regularize the problem here. Otherwise solving for consistent
        # initial conditions requires a least square solver.
        # Using pseudi inverse in consistent_initial_conditions also works
        # without regulariztation.
        self.regularization_mass_matrix = 1e-12

        self.nq = 1
        self.nu = 3
        self.q0 = q0
        self.u0 = u0
        assert self.nq == len(q0)
        assert self.nu == len(u0)

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.nq)
        q_dot[0] = u[0]
        return q_dot

    def q_dot_q(self, t, q, u):
        q_dot_q = np.zeros((self.nq, self.nq))
        return q_dot_q

    def q_dot_u(self, t, q):
        q_dot_u = np.zeros((self.nq, self.nu))
        q_dot_u[0, 0] = 1
        return q_dot_u

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return np.diag([self.mass, 1 / self.stiffness, self.regularization_mass_matrix])

    def prox(self, x):
        if x > self.yieldstress:
            return self.yieldstress
        elif x < -self.yieldstress:
            return -self.yieldstress
        else:
            return x

    def prox_x(self, x):
        if x > self.yieldstress:
            return 0
        elif x < -self.yieldstress:
            return 0
        else:
            return 1

    def h(self, t, q, u):
        u_PM, sigma, la_c = u
        return np.array(
            [
                -sigma + self.force(t),
                -u_PM + la_c,
                sigma - self.prox(sigma - self.prox_r * la_c),
            ]
        )

    def h_u(self, t, q, u):
        _, sigma, la_c = u
        prox_der = self.prox_x(sigma - self.prox_r * la_c)

        h_u = np.zeros((self.nu, self.nu))
        h_u[0, 1] = -1

        h_u[1, 0] = -1
        h_u[1, 2] = 1

        h_u[2, 1] = 1 - prox_der
        h_u[2, 2] = prox_der * self.prox_r

        return h_u


# TODO: Make this a test!
if __name__ == "__main__":
    mass = 1
    stiffness = 1
    yieldstress = 10
    l0 = 1.5
    eps0 = 1.5
    sigma0 = 0
    eps_dot0 = 0
    eps_p0 = 0

    def force(t):
        if t < 1.5:
            return 10
        else:
            if t < 4:
                return -30
            else:
                return 20

    q0 = np.array([eps0], dtype=float)
    u0 = np.array([eps_dot0, sigma0, eps_p0], dtype=float)
    jenkins_element = JenkinsElement(force, mass, stiffness, yieldstress, l0, q0, u0)

    system = System()
    system.add(jenkins_element)
    system.assemble()
    jenkins_element.regularization_mass_matrix = 0

    t0 = 0
    t1 = 10
    dt = 5e-2

    sol = BackwardEuler(
        system,
        t1,
        dt,
        options=SolverOptions(reuse_lu_decomposition=False),
    ).solve()

    t, q, u, eps_p_dot = sol.t, sol.q, sol.u, sol.la_c

    fig, ax = plt.subplots(1, 4)
    ax[0].plot(t, q[:, 0], label=r"$\epsilon$")
    ax[0].grid()
    ax[0].legend()
    plt.xlabel("$t$")

    ax[1].plot(t, -u[:, 1], label=r"$-\sigma$")
    ax[1].grid()
    ax[1].legend()
    plt.xlabel("$t$")

    ax[2].plot(t, u[:, 0], label=r"$\dot\epsilon$")
    # if eps_p_dot is not None:
    #     ax[2].plot(t, eps_p_dot[:, 0], "-r", label="$\dot\epsilon_p$")
    ax[2].grid()
    ax[2].legend()
    plt.xlabel("$t$")

    # TODO: add axis labels
    ax[3].plot(q[:, 0], -u[:, 1])
    ax[3].grid()
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$-\sigma$")

    plt.show()
