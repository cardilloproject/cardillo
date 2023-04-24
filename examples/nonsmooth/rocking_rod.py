import numpy as np
import matplotlib.pyplot as plt

from cardillo.math import approx_fprime
from cardillo import System
from cardillo.solver import MoreauShifted, Rattle, NPIRK
from cardillo.solver._butcher_tableaus import RadauIIATableau


class RockingRod:
    def __init__(self, mass, a, l, eN, eF, mu, q0, u0):
        """Constructor for creating an instance of the rocking rod system.

        Arguments:
        mass -- mass of the rod
        a -- half the distance between the supports
        l -- half the length of the rod
        eN -- restitution coefficient in normal direction
        eF -- restitution coefficient in tangential direction
        mu -- friction coefficient
        q0 -- initial generalized coordinates
        u0 -- initial generalized velocities
        """
        self.m = mass
        self.a = a
        self.l = l
        self.thetaS = (
            1 / 3 * self.m * self.l**2
        )  # rotational inertia around center of mass
        self.g_ = 10  # gravitational acceleration

        # dimensions
        self.nq = 3
        self.nu = 3
        self.nla_N = 2
        self.nla_F = 2
        self.mu = mu * np.ones(2, dtype=float)
        self.e_N = eN * np.ones(2, dtype=float)
        self.e_F = eF * np.ones(2, dtype=float)
        self.NF_connectivity = [[0], [1]]
        self.q0 = q0
        self.u0 = u0
        self.la_N0 = np.zeros(self.nla_N, dtype=float)
        self.la_F0 = np.zeros(self.nla_F, dtype=float)

    #####################
    # Kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q, coo):
        coo.extend(np.eye(self.nu), (self.qDOF, self.uDOF))

    #####################
    # equations of motion
    #####################

    # mass matrix
    def M(self, t, q, coo):
        dense = np.diag([self.m, self.m, self.thetaS])
        coo.extend(dense, (self.uDOF, self.uDOF))

    # total force vector
    def h(self, t, q, u):
        return np.array([0, -self.m * self.g_, 0])

    #################
    # normal contacts
    #################
    def g_N(self, t, q):
        x, y, phi = q
        sp, cp = np.sin(phi), np.cos(phi)
        return np.array(
            [
                y * cp + (-self.a - x) * sp,
                y * cp + (self.a - x) * sp,
            ]
        )

    def g_N_q(self, t, q, coo):
        dense = approx_fprime(q, lambda q: self.g_N(t, q))
        coo.extend(dense, (self.la_NDOF, self.qDOF))

    def W_N_dense(self, t, q):
        x, y, phi = q
        sp, cp = np.sin(phi), np.cos(phi)
        W_N = np.zeros((self.nu, self.nla_N))
        W_N[0, 0] = -sp
        W_N[1, 0] = cp
        W_N[2, 0] = (-self.a - x) * cp - y * sp
        W_N[0, 1] = -sp
        W_N[1, 1] = cp
        W_N[2, 1] = (self.a - x) * cp - y * sp
        return W_N

    def W_N(self, t, q, coo):
        coo.extend(self.W_N_dense(t, q), (self.uDOF, self.la_NDOF))

    def zeta_N(self, t, q, u):
        x, y, phi = q
        x_dot, y_dot, phi_dot = u
        sp, cp = np.sin(phi), np.cos(phi)
        zeta_N1 = -2 * phi_dot * (x_dot * cp + y_dot * sp) - phi_dot**2 * (
            (-self.a - x) * sp + y_dot * cp
        )
        zeta_N2 = -2 * phi_dot * (x_dot * cp + y_dot * sp) - phi_dot**2 * (
            (self.a - x) * sp + y_dot * cp
        )
        return np.array([zeta_N1, zeta_N2])

    def g_N_dot(self, t, q, u):
        return self.W_N_dense(t, q).T @ u

    def g_N_ddot(self, t, q, u, u_dot):
        return self.W_N_dense(t, q).T @ u_dot + self.zeta_N(t, q, u)

    #################
    # friction
    #################
    def W_F_dense(self, t, q):
        x, y, phi = q
        sp, cp = np.sin(phi), np.cos(phi)
        W_F = np.zeros((self.nu, self.nla_F))
        W_F[0, 0] = cp
        W_F[1, 0] = sp
        W_F[0, 1] = cp
        W_F[1, 1] = sp
        return W_F

    def W_F(self, t, q, coo):
        coo.extend(self.W_F_dense(t, q), (self.uDOF, self.la_FDOF))

    def gamma_F(self, t, q, u):
        return self.W_F_dense(t, q).T @ u

    def gamma_F_q(self, t, q, u, coo):
        dense = approx_fprime(q, lambda q: self.gamma_F(t, q, u))
        coo.extend(dense, (self.la_FDOF, self.qDOF))

    def zeta_F(self, t, q, u):
        x, y, phi = q
        x_dot, y_dot, phi_dot = u
        sp, cp = np.sin(phi), np.cos(phi)
        return np.array(
            [
                -sp * phi_dot * x_dot + cp * phi_dot * y_dot,
                -sp * phi_dot * x_dot + cp * phi_dot * y_dot,
            ]
        )

    def gamma_F_dot(self, t, q, u, a):
        return self.W_F_dense(t, q).T @ a + self.zeta_F(t, q, u)


Solver1, label1, dt1, kwargs1 = (
    NPIRK,
    "NPIRK",
    5e-3,
    {"butcher_tableau": RadauIIATableau(2)},
)
# Solver1, label1, dt1, kwargs1 = Rattle, "Rattle", 5e-3, {}
# Solver1, label1, dt1, kwargs1 = MoreauShifted, "MoreauShifted", 5e-3, {}
Solver2, label2, dt2, kwargs2 = MoreauShifted, "MoreauShifted", 5e-3, {}


# This file implements the rocking rod system, see Section 5.2.2 in "Dynamik
# von Starrkörpersystemen mit Reibung und Stößen", Christoph Glocker, PhD
# thesis, TU Munich, 1995
if __name__ == "__main__":
    mass = 1
    l = 1
    eN, eF, mu = 0, 0, 1

    phi0 = 30 / 180 * np.pi

    rods = []

    # # case 1
    # a = 0.2
    # x0 = a * (np.cos(phi0) - 1)
    # y0 = a * np.sin(phi0)
    # q0 = np.array([x0, y0, phi0])
    # u0 = np.array([0, 0, 0])
    # rods.append(RockingRod(mass, a, l, eN, eF, mu, q0, u0))

    # case 2
    a = 0.3
    x0 = a * (np.cos(phi0) - 1)
    y0 = a * np.sin(phi0)
    q0 = np.array([x0, y0, phi0])
    u0 = np.array([0, 0, 0])
    rods.append(RockingRod(mass, a, l, eN, eF, mu, q0, u0))

    # # case 3
    # a = 0.6
    # x0 = a * (np.cos(phi0) - 1)
    # y0 = a * np.sin(phi0)
    # q0 = np.array([x0, y0, phi0])
    # u0 = np.array([0, 0, 0])
    # rods.append(RockingRod(mass, a, l, eN, eF, mu, q0, u0))

    t0 = 0  # initial simulation time
    t1 = 2  # end time
    rho_inf = 0.8
    newton_tol = 1.0e-6
    fix_point_tol = 1.0e-6

    # solve all cases with nonsmooth generalized-alpha scheme
    sol1 = []
    for rod in rods:
        system = System()
        system.add(rod)
        system.assemble()
        sol1.append(Solver1(system, t1, dt1, **kwargs1).solve())

    # solve all cases with Moreau's midpoint rule
    sol2 = []
    for rod in rods:
        system = System()
        system.add(rod)
        system.assemble()
        sol2.append(Solver2(system, t1, dt2, **kwargs2).solve())

    fig, ax = plt.subplots(1, 3)
    plt.title(r"Rocking rod")

    cases = len(rods)
    for case in range(cases):
        ax[case].set_xlabel(r"$\varphi$")
        ax[case].set_ylabel(r"$\dot{\varphi}$")
        ax[case].set_title(r"$a = 0.2$")
        ax[case].plot(sol1[case].q[:, 2], sol1[case].u[:, 2], "-k", label=label1)
        ax[case].plot(sol2[case].q[:, 2], sol2[case].u[:, 2], "--r", label=label2)
        ax[case].legend()

    ######
    fig, ax = plt.subplots(1, 3)

    for case in range(cases):
        ax[case].set_xlabel(r"$t$")
        ax[case].set_ylabel(r"$\dot{\varphi}$")
        ax[case].set_title(r"$a = 0.2$")
        ax[case].plot(sol1[case].t, sol1[case].u[:, 2], "-k", label=label1)
        ax[case].plot(sol2[case].t, sol2[case].u[:, 2], "--r", label=label2)
        ax[case].legend()

    plt.show()
