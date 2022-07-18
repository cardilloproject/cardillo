import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from cardillo.model import Model
from cardillo.solver import Moreau, MoreauGGL, SimplifiedGeneralizedAlphaFirstOrder
from cardillo.math.algebra import e1


class MathematicalPendulumCartesianContact:
    """Mathematical pendulum in Cartesian coordinates and with bilateral
    constraint, see Hairer1996 p. 464 - Example 2. Additionally,
    unilateral with the e_x^I-plane re implemented.

    References
    ----------
    Hairer1996: https://link.springer.com/book/10.1007/978-3-642-05221-7
    """

    def __init__(
        self,
        m,
        l,
        grav,
        q0=None,
        u0=None,
        la_g0=None,
        la_N0=None,
        la_F0=None,
        e_N=None,
        e_F=None,
        prox_r_N=1.0e-1,
        prox_r_T=1.0e-1,
    ):
        self.m = m
        self.l = l
        self.grav = grav

        self.nq = 2
        self.nu = 2
        self.nla_g = 1
        self.nla_N = 1
        self.nla_F = 0
        self.NF_connectivity = [[]]
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.la_N0 = np.zeros(self.nla_N) if la_N0 is None else la_N0
        self.la_F0 = np.zeros(self.nla_F) if la_F0 is None else la_F0
        self.e_N = np.zeros(self.nla_N) if e_N is None else np.array([e_N])
        self.e_F = np.zeros(self.nla_N) if e_F is None else np.array([e_F])
        self.prox_r_N = np.array([prox_r_N])
        self.prox_r_F = np.array([prox_r_T])
        self.mu = np.array([0.0])

    def q_dot(self, t, q, u):
        return u

    def q_dot_q(self, t, q, u, coo):
        pass

    def B(self, t, q, coo):
        coo.extend(np.eye(self.nq, self.nu), (self.qDOF, self.uDOF))

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def M_dense(self, t, q):
        return np.eye(self.nu, self.nu) * self.m

    def M(self, t, q, coo):
        coo.extend(self.M_dense(t, q), (self.uDOF, self.uDOF))

    def f_pot(self, t, q):
        return np.array([0, -self.m * self.grav])

    def f_pot_q(self, t, q, coo):
        pass

    def g(self, t, q):
        x, y = q
        return np.array([x * x + y * y - self.l * self.l])

    def g_dot(self, t, q, u):
        x, y = q
        u_x, u_y = u
        return np.array([2 * x * u_x + 2 * y * u_y])

    def g_dot_u(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def g_ddot(self, t, q, u, a):
        x, y = q
        u_x, u_y = u
        a_x, a_y = a
        return np.array([2 * (u_x * u_x + x * a_x) + 2 * (u_y * u_y + y * a_y)])

    def g_q_dense(self, t, q):
        x, y = q
        return np.array([2 * x, 2 * y])

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g_dense(self, t, q):
        return self.g_q_dense(t, q).T

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        coo.extend(np.eye(self.nu, self.nq) * 2 * la_g[0], (self.uDOF, self.qDOF))

    def G(self, t, q):
        W = self.W_g_dense(t, q)
        M = self.M_dense(t, q)
        # G1 = W.T @ np.linalg.inv(M) @ W
        G2 = W.T @ np.linalg.solve(M, W)
        # error = np.linalg.norm(G1 - G2)
        # print(f"error G: {error}")
        return G2

    def la_g(self, t, q, u):
        W = self.W_g_dense(t, q)
        M = self.M_dense(t, q)
        G = np.array([[W.T @ np.linalg.solve(M, W)]])
        zeta = self.g_ddot(t, q, u, np.zeros_like(u))
        h = self.f_pot(t, q)
        eta = zeta + W.T @ np.linalg.solve(M, h)
        return np.linalg.solve(G, -eta)

    def g_N(self, t, q):
        return e1[:2] @ q

    def g_N_q(self, t, q, coo):
        coo.extend(e1[:2], (self.la_NDOF, self.qDOF))

    def g_N_dot(self, t, q, u):
        return e1[:2] @ u

    def g_N_ddot(self, t, q, u, a):
        return e1[:2] @ a

    def W_N_dense(self, t, q):
        return e1[:2, np.newaxis]

    def W_N(self, t, q, coo):
        coo.extend(self.W_N_dense(t, q), (self.uDOF, self.la_NDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)


if __name__ == "__main__":
    # system parameters
    m = 1
    l = 1
    g = 15
    e_N = 0.5

    # initial state
    q0 = np.array([l, 0.0])
    u0 = np.array([0.0, 0.0])

    # system definition and assemble the model
    pendulum = MathematicalPendulumCartesianContact(m, l, g, q0, u0, e_N=e_N)
    model = Model()
    model.add(pendulum)
    model.assemble()

    # end time and numerical dissipation of generalized-alpha solver
    # t1 = 1
    t1 = 10
    dt = 5.0e-2
    # dt = 1.0e-2
    # dt = 5.0e-3

    # solve with GGL stabilized Moreau scheme
    theta = 0.5
    theta = 0.4
    # sol_Theta = MoreauGGL(model, t1, dt).solve()
    sol_Theta = SimplifiedGeneralizedAlphaFirstOrder(model, t1, dt).solve()
    t_Theta = sol_Theta.t
    q_Theta = sol_Theta.q
    u_Theta = sol_Theta.u
    P_g_Theta = sol_Theta.P_g
    P_N_Theta = sol_Theta.P_N

    # solve with classical Moreau scheme
    sol_Moreau = Moreau(model, t1, dt).solve()
    t_Moreau = sol_Moreau.t
    q_Moreau = sol_Moreau.q
    u_Moreau = sol_Moreau.u
    P_g_Moreau = sol_Moreau.P_g
    P_N_Moreau = sol_Moreau.P_N

    # visualize results
    fig, ax = plt.subplots(2, 2)

    # generalized coordinates
    ax[0, 0].plot(t_Moreau, q_Moreau[:, 0], "-xb", label="x - Moreau")
    ax[0, 0].plot(t_Moreau, q_Moreau[:, 1], "--ob", label="y - Moreau")
    ax[0, 0].plot(t_Theta, q_Theta[:, 0], "-xr", label="x - MoreauGGL")
    ax[0, 0].plot(t_Theta, q_Theta[:, 1], "--or", label="y - MoreauGGL")
    ax[0, 0].grid()
    ax[0, 0].legend()

    # generalized velocities
    ax[0, 1].plot(t_Moreau, u_Moreau[:, 0], "-xb", label="x_dot - Moreau")
    ax[0, 1].plot(t_Moreau, u_Moreau[:, 1], "--ob", label="y_dot - Moreau")
    ax[0, 1].plot(t_Theta, u_Theta[:, 0], "-xr", label="x_dot - MoreauGGL")
    ax[0, 1].plot(t_Theta, u_Theta[:, 1], "--or", label="y_dot - MoreauGGL")
    ax[0, 1].grid()
    ax[0, 1].legend()

    # bilateral constraints
    ax[1, 0].plot(t_Moreau, P_g_Moreau[:, 0], "-xb", label="P_g - Moreau")
    ax[1, 0].plot(t_Theta, P_g_Theta[:, 0], "-xr", label="P_g - MoreauGGL")
    ax[1, 0].grid()
    ax[1, 0].legend()

    # normal percussions
    ax[1, 1].plot(t_Moreau, P_N_Moreau[:, 0], "-xb", label="P_N - Moreau")
    ax[1, 1].plot(t_Theta, P_N_Theta[:, 0], "-xr", label="P_N - MoreauGGL")
    ax[1, 1].grid()
    ax[1, 1].legend()

    plt.show()
