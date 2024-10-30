import numpy as np
import pytest
from itertools import product
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from cardillo import System
from cardillo.discrete import PointMass
from cardillo.interactions import TwoPointInteraction
from cardillo.force_laws import MaxwellElement as MaxwellElementFL
from cardillo.solver import ScipyIVP, ScipyDAE, BackwardEuler, Moreau, Rattle


class MaxwellElement:
    """Pointmass connected with Maxwell element to origin."""

    def __init__(self, mass, stiffness, damping, l0, q0, u0):
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping
        self.l0 = l0

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
        q_dot[1] = (self.stiffness / self.damping) * (x - x_D - self.l0)
        return q_dot

    def q_dot_q(self, t, q, u):
        q_dot_q = np.zeros((self.nq, self.nq))
        ratio = self.stiffness / self.damping
        q_dot_q[1, 0] = ratio
        q_dot_q[1, 1] = -ratio
        return q_dot_q

    def q_dot_u(self, t, q):
        q_dot_u = np.zeros((self.nq, self.nu))
        q_dot_u[0, 0] = 1
        return q_dot_u

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return self.mass * np.eye(self.nu)

    def h(self, t, q, u):
        x, x_D = q
        return -self.stiffness * (x - x_D - self.l0)


class MaxwellElementCompliance:
    """Pointmass connected with Maxwell element to origin."""

    def __init__(self, mass, stiffness, damping, l0, q0, u0, la_c0):
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping
        self.l0 = l0

        self.nq = 2
        self.nu = 1
        self.nla_c = 1
        self.q0 = q0
        self.u0 = u0
        self.la_c0 = la_c0
        assert self.nq == len(q0)
        assert self.nu == len(u0)
        assert self.nla_c == len(la_c0)

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        x, x_D = q
        q_dot = np.zeros(self.nq)
        q_dot[0] = u[0]
        q_dot[1] = (self.stiffness / self.damping) * (x - x_D - self.l0)
        return q_dot

    def q_dot_q(self, t, q, u):
        q_dot_q = np.zeros((self.nq, self.nq))
        ratio = self.stiffness / self.damping
        q_dot_q[1, 0] = ratio
        q_dot_q[1, 1] = -ratio
        return q_dot_q

    def q_dot_u(self, t, q):
        q_dot_u = np.zeros((self.nq, self.nu))
        q_dot_u[0, 0] = 1
        return q_dot_u

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return self.mass * np.eye(self.nu)

    ############
    # compliance
    ############
    def la_c(self, t, q, u):
        x, x_D = q
        return -self.stiffness * np.array([x - x_D - self.l0])

    def W_c(self, t, q):
        return np.ones((self.nu, self.nla_c))

    def c(self, t, q, u, la_c):
        x, x_D = q
        g_F = 1 / self.stiffness * la_c + x - x_D - self.l0
        return np.array([g_F])

    def c_q(self, t, q, u, la_c):
        c_q = np.zeros((self.nla_c, self.nq))
        c_q[0, 0] = 1
        c_q[0, 1] = -1
        return c_q

    def c_la_c(self):
        return np.diag([1 / self.stiffness])

    def Wla_c_q(self, t, q, la_c):
        np.zeros((self.nu, self.nq))


class MaxwellElementForceElement:
    def __init__(self, mass, stiffness, damping, l0, x0, l_d0, x_dot0):
        self.system = System()
        pm = PointMass(mass, q0=np.array([x0, 0, 0]), u0=np.array([x_dot0, 0, 0]))
        pm.name = "point mass"
        self.system.add(pm)
        tpi = TwoPointInteraction(self.system.origin, pm)
        max = MaxwellElementFL(tpi, stiffness, damping, l_ref=l0, q0=np.array([l_d0]))
        max.name = "Maxwell-element"
        self.system.add(tpi, max)
        self.system.assemble()

    def get_system(self):
        return self.system


solvers_and_kwargs = [
    (ScipyIVP, {}),
    (ScipyDAE, {}),
    (BackwardEuler, {}),
    (Moreau, {}),
    (Rattle, {}),
]

formulation = [
    "MaxwellElement",
    "MaxwellElementCompliance",
    "MaxwellElementForceElement",
]

test_parameters = product(solvers_and_kwargs, formulation)


@pytest.mark.parametrize("Solver_and_kwargs, formulation", test_parameters)
def test_maxwell(Solver_and_kwargs, formulation, show=False):
    Solver, solver_kwargs = Solver_and_kwargs
    mass = 1e-3
    stiffness = 1e1
    damping = 1
    l0 = 1
    x0 = 1.5
    x_D0 = 0.0
    x_dot0 = 0.0
    q0 = np.array([x0, x_D0], dtype=float)
    u0 = np.array([x_dot0], dtype=float)
    la_c0 = np.array([-5], dtype=float)

    match formulation:
        case "MaxwellElement":
            maxwell_element = MaxwellElement(mass, stiffness, damping, l0, q0, u0)
            system = System()
            system.add(maxwell_element)
            system.assemble()
        case "MaxwellElementCompliance":
            maxwell_element = MaxwellElementCompliance(
                mass, stiffness, damping, l0, q0, u0, la_c0
            )
            system = System()
            system.add(maxwell_element)
            system.assemble()
        case "MaxwellElementForceElement":
            system = MaxwellElementForceElement(
                mass, stiffness, damping, l0, x0, x_D0, x_dot0
            ).get_system()
        case _:
            raise NotImplementedError

    t1 = 1
    dt = 1e-3
    sol = Solver(system, t1, dt, *solver_kwargs).solve()
    t, q, u = sol.t, sol.q, sol.u

    # reference solution
    def eqm(t, z):
        x, x_d, u = z
        dx = u
        dx_d = (stiffness / damping) * (x - x_d - l0)
        du = -(stiffness / mass) * (x - x_d - l0)
        return np.array([dx, dx_d, du])

    sol_ref = solve_ivp(eqm, [t[0], t[-1]], np.array([x0, x_D0, x_dot0]), t_eval=t)
    t_ref = sol_ref.t
    z_ref = sol_ref.y

    if show:
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(t, q[:, 0], "-b", label="x")
        ax[0].plot(t_ref, z_ref[0], "--b", label="x_ref")
        ax[0].plot(t, q[:, -1], "-r", label="x_D")
        ax[0].plot(t_ref, z_ref[1], "--r", label="x_D_ref")
        ax[0].grid()
        ax[0].legend()
        ax[1].plot(t, u[:, 0], "-b", label="x_dot")
        ax[1].plot(t_ref, z_ref[2], "--b", label="x_dot_ref")
        ax[1].grid()

        ax[0].legend()
        ax[1].legend()
        plt.show()


if __name__ == "__main__":
    for p in test_parameters:
        test_maxwell(*p, show=True)
