import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.discrete import PointMass
from cardillo.forces import MaxwellElement as MaxwellElementFL
from cardillo.solver import EulerBackward


class MaxwellElement:
    """Pointmass connected with Maxwell element to origin."""

    def __init__(self, mass, stiffness, damping, q0, u0):
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping

        self.nq = 2
        self.nu = 1
        self.q0 = q0
        self.u0 = u0
        assert self.nq == len(q0)
        assert self.nu == len(u0)

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        x, x_D = q
        q_dot = np.zeros(self.nq)
        q_dot[0] = u[0]
        q_dot[1] = (self.stiffness / self.damping) * (x - x_D)
        return q_dot

    def q_ddot(self, t, q, u, u_dot):
        x_dot, x_D_dot = self.q_dot(t, q, u)
        q_ddot = np.zeros(self.nq)
        q_ddot[0] = u_dot[0]
        q_ddot[1] = (self.stiffness / self.damping) * (x_dot - x_D_dot)
        return q_ddot

    def q_dot_q(self, t, q, u):
        q_dot_q = np.zeros((self.nq, self.nq))
        ratio = self.stiffness / self.damping
        q_dot_q[1, 0] = ratio
        q_dot_q[1, 1] = -ratio
        return q_dot_q

    def q_dot_u(self, t, q, u):
        q_dot_u = np.zeros((self.nq, self.nu))
        q_dot_u[0, 0] = 1
        return q_dot_u

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return self.mass * np.ones(self.nu)

    def h(self, t, q, u):
        x, x_D = q
        return -self.stiffness * (x - x_D)


class MaxwellElementCompliance:
    """Pointmass connected with Maxwell element to origin."""

    def __init__(self, mass, stiffness, damping, q0, u0):
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping

        self.nq = 2
        self.nu = 1
        self.nla_c = 1
        self.q0 = q0
        self.u0 = u0
        assert self.nq == len(q0)
        assert self.nu == len(u0)

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        x, x_D = q
        q_dot = np.zeros(self.nq)
        q_dot[0] = u[0]
        q_dot[1] = (self.stiffness / self.damping) * (x - x_D)
        return q_dot

    def q_ddot(self, t, q, u, u_dot):
        x_dot, x_D_dot = self.q_dot(t, q, u)
        q_ddot = np.zeros(self.nq)
        q_ddot[0] = u_dot[0]
        q_ddot[1] = (self.stiffness / self.damping) * (x_dot - x_D_dot)
        return q_ddot

    def q_dot_q(self, t, q, u):
        q_dot_q = np.zeros((self.nq, self.nq))
        ratio = self.stiffness / self.damping
        q_dot_q[1, 0] = ratio
        q_dot_q[1, 1] = -ratio
        return q_dot_q

    def q_dot_u(self, t, q, u):
        q_dot_u = np.zeros((self.nq, self.nu))
        q_dot_u[0, 0] = 1
        return q_dot_u

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return self.mass * np.ones(self.nu)

    ############
    # compliance
    ############
    def K_c(self):
        return np.diag([1 / self.stiffness])

    def W_c(self, t, q):
        return np.ones((self.nu, self.nla_c))

    def c(self, t, q, u):
        x, x_D = q
        g_F = x - x_D
        return np.array([g_F])
        # return np.array([g_F**3])

    def c_q(self, t, q, u):
        c_q = np.zeros((self.nc, self.nq))
        c_q[0, 0] = 1
        c_q[0, 1] = -1
        return c_q

        # x, x_D = q
        # g_F = x - x_D
        # factor = 3 * g_F**2
        # c_q = np.zeros((self.nc, self.nq))
        # c_q[0, 0] = factor
        # c_q[0, 1] = -factor
        # return c_q

class MaxwellElementForceElement:
    def __init__(self, mass, stiffness, damping, x0, g_d0, x_dot0):
        
        self.system = System()
        pm = PointMass(mass, q0=np.array([x0, 0, 0]), u0=np.array([x_dot0, 0, 0]))
        pm.name = 'point mass'
        self.system.add(pm)
        max = MaxwellElementFL(self.system.origin, pm, stiffness, damping,g_ref=0, q0=np.array([g_d0]))
        max.name = 'Maxwell-element'
        self.system.add(max)
        self.system.assemble()

    def get_system(self):
        return self.system



if __name__ == "__main__":
    mass = 1e-3
    stiffness = 1e1
    damping = 1
    x0 = 1.
    x_D0 = 0.
    x_dot0 = 0.
    q0 = np.array([x0, x_D0], dtype=float)
    u0 = np.array([x_dot0], dtype=float)

    # maxwell_element = MaxwellElement(mass, stiffness, damping, q0, u0)
    # maxwell_element = MaxwellElementCompliance(mass, stiffness, damping, q0, u0)
    maxwell_element = MaxwellElementForceElement(mass, stiffness, damping, x0, x_D0, x_dot0).get_system()

    system = System()
    system.add(maxwell_element)
    system.assemble()

    t1 = 10
    dt = 5e-3
    sol = EulerBackward(system, t1, dt).solve()

    t, q, u = sol.t, sol.q, sol.u
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(t, q[:, 0], "-b", label="x")
    ax[0].plot(t, q[:, -1], "--r", label="x_D")
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(t, u[:, 0], label="x_dot")
    ax[1].grid()
    ax[1].legend()
    plt.show()
