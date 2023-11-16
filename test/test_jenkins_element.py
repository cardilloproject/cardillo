import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from cardillo import System
from cardillo.math import prox_sphere, prox_sphere_x, fsolve
from cardillo.discrete import PointMass
from cardillo.forces import MaxwellElement as MaxwellElementFL  # TODO: This is unused!
from cardillo.forces import Force
from cardillo.solver import ScipyIVP, Moreau, BackwardEuler


class JenkinsElement:
    """Pointmass connected with Jenkins element to origin."""

    def __init__(self, mass, stiffness, yieldstress, l0, q0, u0, la_c0):
        self.mass = mass
        self.stiffness = stiffness
        self.yieldstress = yieldstress
        self.l0 = l0
        self.s0 = self.yieldstress

        self.nq = 1
        self.nu = 2
        self.nla_c = 1
        self.q0 = q0
        self.u0 = u0
        self.la_c0 = la_c0
        assert self.nq == len(q0)
        assert self.nu == len(u0)

    #####################
    # auxiliary functions
    #####################
    def local_qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def local_uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)

    def J_P(self, t, q, frame_ID=None, K_r_SP=None):
        return np.eye(2, self.nu, dtype=q.dtype)

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=None):
        return np.zeros((2, self.nu, self.nq))

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.nq)
        q_dot[0] = u[0]
        return q_dot

    def q_ddot(self, t, q, u, u_dot):
        q_ddot = np.zeros(self.nq)
        q_ddot[0] = u_dot[0]
        return q_ddot

    def q_dot_q(self, t, q, u):
        q_dot_q = np.zeros((self.nq, self.nq))
        return q_dot_q

    def q_dot_u(self, t, q, u):
        q_dot_u = np.zeros((self.nq, self.nu))
        q_dot_u[0, 0] = 1
        return q_dot_u

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return np.diag([self.mass, 1 / self.stiffness])

    def h(self, t, q, u):
        u_PM, sigma = u
        return np.array((-sigma, -u_PM))

    ############
    # compliance
    ############
    def la_c(self, t, q, u):
        raise RuntimeError("This is not working yet.")
        f = lambda la_c: self.c(t, q, u, la_c)
        jac = lambda la_c: self.c_la_c(t, q, u, la_c)
        # jac = "2-point"

        # _, sigma = u
        # r = 1 / self.stiffness
        # # prox = lambda la_c: prox_sphere(sigma - r * la_c, self.yieldstress)
        # # from scipy.optimize import fixed_point
        # # la_c = fixed_point(prox, self.la_c0.copy(), xtol=1e-4, method="del2", maxiter=int(1e5))
        # # converged = True

        # la_c0 = self.la_c0.copy()
        # converged = False
        # tol = 1e-10
        # i = 0
        # prox_r = 0.99
        # while not converged:
        #     i += 1
        #     la_c = prox_sphere(sigma - prox_r * r * la_c0, self.yieldstress)
        #     error = np.max(np.abs(la_c - la_c0))
        #     converged = error < tol
        #     la_c0 = la_c.copy()

        # from scipy.optimize import least_squares
        # sol = least_squares(f, self.la_c0, jac=jac, method="lm", ftol=1e-10, xtol=1e-10, gtol=1e-10)
        # converged = sol.success
        # la_c = sol.x

        # from scipy.optimize import root_scalar
        # method = "secant"
        # method = "newton"
        # sol = root_scalar(f, x0=self.la_c0, method=method, fprime=jac)
        # converged = sol.converged
        # la_c = sol.root

        # la_c, converged, error, i, _ = fsolve(f, self.la_c0, jac=jac, atol=1e-4, max_iter=1000)

        assert converged, "JenkinsElement.la_c is not converged"
        self.la_c0 = la_c.copy()
        return la_c

    def W_c(self, t, q):
        W_c = np.zeros((self.nu, self.nla_c))
        W_c[1, 0] = 1
        return W_c

    def c(self, t, q, u, la_c):
        _, sigma = u
        r = 1 / self.stiffness
        return -sigma + prox_sphere(sigma - r * la_c, self.yieldstress)

    def c_u(self, t, q, u, la_c):
        c_u = np.zeros((self.nla_c, self.nu))
        _, sigma = u
        r = 1 / self.stiffness
        c_u[0, 1] = -1 + prox_sphere_x(sigma - r * la_c, self.yieldstress)[0]
        return c_u

    def c_la_c(self, t, q, u, la_c):
        c_la_c = np.zeros((self.nla_c, self.nla_c))
        _, sigma = u
        r = 1 / self.stiffness
        c_la_c[0] = -r * prox_sphere_x(sigma - r * la_c, self.yieldstress)[0]
        return c_la_c


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
    q0 = np.array([eps0], dtype=float)
    u0 = np.array([eps_dot0, sigma0], dtype=float)
    la_c0 = np.array([eps_p0], dtype=float)

    jenkins_element = JenkinsElement(mass, stiffness, yieldstress, l0, q0, u0, la_c0)

    def f(t):
        if t < 1.5:
            return np.array([10, 0])
        else:
            return np.array([-30, 0])

    system = System()
    system.add(jenkins_element)
    system.add(Force(f, jenkins_element))
    system.assemble()

    # system = MaxwellElementForceElement(
    #     mass, stiffness, damping, l0, x0, x_D0, x_dot0
    # ).get_system()

    t0 = 0
    t1 = 4
    dt = 5e-3

    # sol = ScipyIVP(system, t1, dt, method="RK45").solve()
    # sol = ScipyIVP(system, t1, dt, method="RK23").solve()
    # sol = ScipyIVP(system, t1, dt, method="BDF").solve()
    sol = BackwardEuler(system, t1, dt, debug=False).solve()

    # - ref. solution
    # def eqm(t, z):
    #     x, x_d, u = z
    #     dx = u
    #     dx_d = (stiffness / damping) * (x - x_d - l0)
    #     du = -(stiffness / mass) * (x - x_d - l0)
    #     return np.array([dx, dx_d, du])

    # sol_ref = solve_ivp(eqm, [0, t1], np.array([x0, x_D0, x_dot0]))
    # t_ref = sol_ref.t
    # z_ref = sol_ref.y

    t, q, u, eps_p_dot = sol.t, sol.q, sol.u, sol.la_c

    fig, ax = plt.subplots(1, 4)
    ax[0].plot(t, q[:, 0], "-b", label="$\epsilon$")
    ax[0].grid()
    ax[0].legend()
    plt.xlabel("$t$")

    ax[1].plot(t, -u[:, 1], label="$-\sigma$")
    ax[1].grid()
    ax[1].legend()
    plt.xlabel("$t$")

    ax[2].plot(t, u[:, 0], label="$\dot\epsilon$")
    if eps_p_dot is not None:
        ax[2].plot(t, eps_p_dot[:, 0], "-r", label="$\dot\epsilon_p$")
    ax[2].grid()
    ax[2].legend()
    plt.xlabel("$t$")

    # TODO: add axis labels
    ax[3].plot(q[:, 0], -u[:, 1])
    ax[3].grid()
    plt.xlabel("$\epsilon$")
    plt.ylabel("$-\sigma$")

    plt.show()
