import numpy as np

from cardillo.math.algebra import cross3, norm3, ax2skew
from cardillo.math.numerical_derivative import Numerical_derivative

class Rolling_condition():
    def __init__(self, disc, la_gamma0=None):
        self.subsystem = disc

        self.nla_gamma = 3
        self.la_gamma0 = np.zeros(self.nla_gamma) if la_gamma0 is None else la_gamma0
        
    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.qDOF_P()]
        self.uDOF = self.subsystem.qDOF[self.subsystem.uDOF_P()]

    def r_SA(self, t, q):
        e_K_y = self.subsystem.A_IK(t, q)[:, 1]
        g_K_x = cross3( e_K_y, np.array([0, 0, 1]) )
        e_K_x = g_K_x / norm3(g_K_x)
        e_K_z = cross3(e_K_x, e_K_y)
        return -self.subsystem.r * e_K_z

    def A_RI(self, t, q):
        e_K_y = self.subsystem.A_IK(t, q)[:, 1]
        g_R_x = cross3( e_K_y, np.array([0, 0, 1]) )
        e_R_x = g_R_x / norm3(g_R_x)

        e_R_z = np.array([0, 0, 1])
        e_R_y = cross3( e_R_z, e_R_x )

        return np.vstack((e_R_x, e_R_y, e_R_z))

    def gamma(self, t, q, u):
        return self.subsystem.v_P(t, q, u, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SA(t, q))

    def gamma_dot(self, t, q, u, u_dot):
        gamma_q = Numerical_derivative(self.gamma, order=2)._x(t, q, u)
        gamma_u = gamma_u = self.gamma_u_dense(t, q)
        
        return gamma_q @ self.subsystem.q_dot(t, q, u) + gamma_u @ u_dot

    def gamma_q(self, t, q, u, coo):
        dense = Numerical_derivative(self.gamma, order=2)._x(t, q, u)
        coo.extend(dense, (self.la_gammaDOF, self.qDOF))

    def gamma_u_dense(self, t, q):
        return self.subsystem.J_P(t, q, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SA(t, q))

    def gamma_u(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q), (self.la_gammaDOF, self.uDOF))

    def W_gamma(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q).T, (self.uDOF, self.la_gammaDOF))

    def Wla_gamma_q(self, t, q, la_gamma, coo):
        dense = Numerical_derivative(lambda t, q: self.gamma_u_dense(t, q).T @ la_gamma)._x(t, q)
        coo.extend(dense, (self.uDOF, self.qDOF))
