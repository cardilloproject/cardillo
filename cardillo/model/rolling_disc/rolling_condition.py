import numpy as np

from cardillo.math.algebra import cross3, norm3, ax2skew
from cardillo.math.numerical_derivative import Numerical_derivative

class Rolling_condition_R_frame():
    def __init__(self, disc, la_gamma0=None):
        self.disc = disc

        self.nla_gamma = 3
        self.la_gamma0 = np.zeros(self.nla_gamma) if la_gamma0 is None else la_gamma0
        
    def assembler_callback(self):
        self.qDOF = self.disc.qDOF_P()
        self.uDOF = self.disc.uDOF_P()

    def r_SA(self, t, q):
        e_K_y = self.disc.A_IK(t, q)[:, 1]
        g_K_x = cross3( e_K_y, np.array([0, 0, 1]) )
        e_K_x = g_K_x / norm3(g_K_x)
        e_K_z = cross3(e_K_x, e_K_y)
        return -self.disc.r * e_K_z

    def A_RI(self, t, q):
        e_K_y = self.disc.A_IK(t, q)[:, 1]
        g_R_x = cross3( e_K_y, np.array([0, 0, 1]) )
        e_R_x = g_R_x / norm3(g_R_x)

        e_R_z = np.array([0, 0, 1])
        e_R_y = cross3( e_R_z, e_R_x )

        return np.vstack((e_R_x, e_R_y, e_R_z))

    def gamma(self, t, q, u):
        return self.A_RI(t, q) @ self.disc.v_P(t, q, u, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q))

    def gamma_q(self, t, q, u, coo):
        dense = Numerical_derivative(self.gamma, order=2)._x(t, q, u)
        coo.extend(dense, (self.la_gammaDOF, self.qDOF))

    def gamma_u_dense(self, t, q):
        return self.A_RI(t, q) @ self.disc.J_P(t, q, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q))

    def gamma_u(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q), (self.la_gammaDOF, self.uDOF))

    def W_gamma(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q).T, (self.uDOF, self.la_gammaDOF))

    def Wla_gamma_q(self, t, q, la_gamma, coo):
        dense = Numerical_derivative(lambda t, q: self.gamma_u_dense(t, q).T @ la_gamma)._x(t, q)
        coo.extend(dense, (self.uDOF, self.qDOF))

class Rolling_condition_I_frame():
    def __init__(self, disc, la_gamma0=None):
        self.disc = disc

        self.nla_gamma = 3
        self.la_gamma0 = np.zeros(self.nla_gamma) if la_gamma0 is None else la_gamma0
        
    def assembler_callback(self):
        self.qDOF = self.disc.qDOF_P()
        self.uDOF = self.disc.uDOF_P()

    def r_SA(self, t, q):
        e_K_y = self.disc.A_IK(t, q)[:, 1]
        g_K_x = cross3( e_K_y, np.array([0, 0, 1]) )
        e_K_x = g_K_x / norm3(g_K_x)
        e_K_z = cross3(e_K_x, e_K_y)
        return -self.disc.r * e_K_z

    def A_RI(self, t, q):
        e_K_y = self.disc.A_IK(t, q)[:, 1]
        g_R_x = cross3( e_K_y, np.array([0, 0, 1]) )
        e_R_x = g_R_x / norm3(g_R_x)

        e_R_z = np.array([0, 0, 1])
        e_R_y = cross3( e_R_z, e_R_x )

        return np.vstack((e_R_x, e_R_y, e_R_z))

    def gamma(self, t, q, u):
        return self.disc.v_P(t, q, u, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q))

    def gamma_q(self, t, q, u, coo):
        dense = Numerical_derivative(self.gamma, order=2)._x(t, q, u)
        coo.extend(dense, (self.la_gammaDOF, self.qDOF))

    def gamma_u_dense(self, t, q):
        return self.disc.J_P(t, q, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q))

    def gamma_u(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q), (self.la_gammaDOF, self.uDOF))

    def W_gamma(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q).T, (self.uDOF, self.la_gammaDOF))

    def Wla_gamma_q(self, t, q, la_gamma, coo):
        dense = Numerical_derivative(lambda t, q: self.gamma_u_dense(t, q).T @ la_gamma)._x(t, q)
        coo.extend(dense, (self.uDOF, self.qDOF))

class Rolling_condition_I_frame_g_gamma():
    def __init__(self, disc, la_g0=None, la_gamma0=None):
        self.disc = disc

        self.nla_g = 1
        self.nla_gamma = 2
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0
        self.la_gamma0 = np.zeros(self.nla_gamma) if la_gamma0 is None else la_gamma0
        
    def assembler_callback(self):
        self.qDOF = self.disc.qDOF_P()
        self.uDOF = self.disc.uDOF_P()

    def r_SA(self, t, q):
        e_K_y = self.disc.A_IK(t, q)[:, 1]
        g_K_x = cross3( e_K_y, np.array([0, 0, 1]) )
        e_K_x = g_K_x / norm3(g_K_x)
        e_K_z = cross3(e_K_x, e_K_y)
        return -self.disc.r * e_K_z

    def A_RI(self, t, q):
        e_K_y = self.disc.A_IK(t, q)[:, 1]
        g_R_x = cross3( e_K_y, np.array([0, 0, 1]) )
        e_R_x = g_R_x / norm3(g_R_x)

        e_R_z = np.array([0, 0, 1])
        e_R_y = cross3( e_R_z, e_R_x )

        return np.vstack((e_R_x, e_R_y, e_R_z))

    # bilateral constraints on position level
    def g(self, t, q):
        return np.array([ (self.disc.r_OP(t, q) + self.r_SA(t, q) )[2] ])

    def g_q_dense(self, t, q):
        return Numerical_derivative(self.g)._x(t, q)

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g_dense(self, t, q):
        return self.disc.J_P(t, q, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q) ).T[:, 2][:, None]

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        dense = Numerical_derivative(lambda t, q: self.W_g_dense(t, q) @ la_g)._x(t, q)
        coo.extend(dense, (self.uDOF, self.qDOF))

    # bilateral constraints on velocity level
    def gamma(self, t, q, u):
        return self.disc.v_P(t, q, u, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q))[:2]

    def gamma_q(self, t, q, u, coo):
        dense = Numerical_derivative(self.gamma, order=2)._x(t, q, u)[:2]
        coo.extend(dense, (self.la_gammaDOF, self.qDOF))

    def gamma_u_dense(self, t, q):
        return self.disc.J_P(t, q, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q))[:2]

    def gamma_u(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q), (self.la_gammaDOF, self.uDOF))

    def W_gamma(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q).T, (self.uDOF, self.la_gammaDOF))

    def Wla_gamma_q(self, t, q, la_gamma, coo):
        dense = Numerical_derivative(lambda t, q: self.gamma_u_dense(t, q).T @ la_gamma)._x(t, q)
        coo.extend(dense, (self.uDOF, self.qDOF))