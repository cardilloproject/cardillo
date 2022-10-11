import numpy as np
from cardillo.math import norm


class Sphere2Sphere:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        radius1,
        radius2,
        e_N,
        # e_F,
        # mu,
        prox_r_N,
        # prox_r_F,
        frame_ID1=np.zeros(3, dtype=float),
        frame_ID2=np.zeros(3, dtype=float),
    ):
        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.radius1 = radius1
        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2
        self.radius2 = radius2

        self.e_N = np.array([e_N])
        self.prox_r_N = np.array([prox_r_N])

        # # self.e_F = np.array([e_F])
        # # self.mu = np.array([mu])
        # self.prox_r_F = np.array([prox_r_F])
        self.e_F = np.array([0])
        self.mu = np.array([0])
        self.prox_r_F = np.array([0])

        self.nla_N = 1
        # self.nla_F = 2
        # self.NF_connectivity = [[0], [0]]
        self.nla_F = 0
        self.NF_connectivity = [[]]

        self.r_OQ = lambda t: self.frame.r_OP(t)
        self.n = lambda t: self.frame.A_IK(t)[:, 2]
        # self.n = lambda t: np.array([1., 0., 0.])
        # self.n = lambda t: np.array([0., 1, 0.])
        # self.n = lambda t: np.array([0., 0., 1.])
        self.v_Q = lambda t: self.frame.v_P(t)
        self.a_Q = lambda t: self.frame.a_P(t)

        self.la_N0 = np.zeros(self.nla_N)
        self.la_F0 = np.zeros(self.nla_F)

        self.is_assembled = False

    def assembler_callback(self):
        qDOF1 = self.subsystem1.qDOF_P(self.frame_ID1)
        qDOF2 = self.subsystem2.qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate(
            [self.subsystem1.qDOF[qDOF1], self.subsystem2.qDOF[qDOF2]]
        )
        self.nq1 = nq1 = len(qDOF1)
        self.nq2 = len(qDOF2)
        self._nq = self.nq1 + self.nq2

        uDOF1 = self.subsystem1.uDOF_P(self.frame_ID1)
        uDOF2 = self.subsystem2.uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate(
            [self.subsystem1.uDOF[uDOF1], self.subsystem2.uDOF[uDOF2]]
        )
        self.nu1 = nu1 = len(uDOF1)
        self.nu2 = len(uDOF2)
        self._nu = self.nu1 + self.nu2

        # auxiliary functions for subsystem 1
        self.r_OS1 = lambda t, q: self.subsystem1.r_OP(t, q[:nq1], self.frame_ID1)
        self.r_OS1_q = lambda t, q: self.subsystem1.r_OP_q(t, q[:nq1], self.frame_ID1)
        self.v_S1 = lambda t, q, u: self.subsystem1.v_P(
            t, q[:nq1], u[:nu1], self.frame_ID1
        )
        self.v_S1_q = lambda t, q, u: self.subsystem1.v_P_q(
            t, q[:nq1], u[:nu1], self.frame_ID1
        )
        self.a_S1 = lambda t, q, u, u_dot: self.subsystem1.a_P(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1
        )
        self.a_S1_q = lambda t, q, u, u_dot: self.subsystem1.a_P_q(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1
        )
        self.a_S1_u = lambda t, q, u, u_dot: self.subsystem1.a_P_u(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1
        )
        self.J_S1 = lambda t, q: self.subsystem1.J_P(t, q[:nq1], self.frame_ID1)
        self.J_S1_q = lambda t, q: self.subsystem1.J_P_q(t, q[:nq1], self.frame_ID1)

        # auxiliary functions for subsystem 1
        self.r_OS2 = lambda t, q: self.subsystem2.r_OP(t, q[nq1:], self.frame_ID2)
        self.r_OS2_q = lambda t, q: self.subsystem2.r_OP_q(t, q[nq1:], self.frame_ID2)
        self.v_S2 = lambda t, q, u: self.subsystem2.v_P(
            t, q[nq1:], u[nu1:], self.frame_ID2
        )
        self.v_S2_q = lambda t, q, u: self.subsystem2.v_P_q(
            t, q[nq1:], u[nu1:], self.frame_ID2
        )
        self.a_S2 = lambda t, q, u, u_dot: self.subsystem2.a_P(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2
        )
        self.a_S2_q = lambda t, q, u, u_dot: self.subsystem2.a_P_q(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2
        )
        self.a_S2_u = lambda t, q, u, u_dot: self.subsystem2.a_P_u(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2
        )
        self.J_S2 = lambda t, q: self.subsystem2.J_P(t, q[nq1:], self.frame_ID2)
        self.J_S2_q = lambda t, q: self.subsystem2.J_P_q(t, q[nq1:], self.frame_ID2)

    def g_N(self, t, q):
        r_S1S2 = self.r_OS2(t, q) - self.r_OS1(t, q)
        return np.array([norm(r_S1S2) - self.radius1 - self.radius2])

    def g_N_q_dense(self, t, q):
        n = self.normal(t, q)
        r_OS1 = self.J_S1(t, q)
        r_OS2 = self.J_S2(t, q)
        return np.concatenate((-n @ r_OS1, n @ r_OS2))

    def g_N_q(self, t, q, coo):
        coo.extend(self.g_N_q_dense(t, q), (self.la_NDOF, self.qDOF))

    def normal(self, t, q):
        r_S1S2 = self.r_OS2(t, q) - self.r_OS1(t, q)
        return r_S1S2 / norm(r_S1S2)

    def g_N_dot(self, t, q, u):
        return np.array([self.normal(t, q) @ (self.v_S2(t, q, u) - self.v_S1(t, q, u))])

    # def g_N_dot_q_dense(self, t, q, u):
    #     return np.array([self.n(t) @ self.v_P_q(t, q, u)])

    # def g_N_dot_q(self, t, q, u, coo):
    #     coo.extend(self.g_N_dot_q_dense(t, q, u), (self.la_NDOF, self.qDOF))

    def g_N_dot_u_dense(self, t, q):
        n = self.normal(t, q)
        J_S1 = self.J_S1(t, q)
        J_S2 = self.J_S2(t, q)
        return np.concatenate((-n @ J_S1, n @ J_S2))

    def g_N_dot_u(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q), (self.la_NDOF, self.uDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    # def xi_N_q(self, t, q, u_pre, u_post, coo):
    #     g_N_q_pre = self.g_N_dot_q_dense(t, q, u_pre)
    #     g_N_q_post = self.g_N_dot_q_dense(t, q, u_post)
    #     dense = g_N_q_post + self.e_N * g_N_q_pre
    #     coo.extend(dense, (self.la_NDOF, self.qDOF))

    def W_N(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q).T, (self.uDOF, self.la_NDOF))

    def g_N_ddot(self, t, q, u, u_dot):
        return np.array(
            [
                self.normal(t, q)
                @ (self.a_S2(t, q, u, u_dot) - self.a_S1(t, q, u, u_dot))
            ]
        )

    # def g_N_ddot_q(self, t, q, u, u_dot, coo):
    #     dense = np.array([self.n(t) @ self.a_P_q(t, q, u, u_dot)])
    #     coo.extend(dense, (self.la_NDOF, self.qDOF))

    # def g_N_ddot_u(self, t, q, u, u_dot, coo):
    #     dense = np.array([self.n(t) @ self.a_P_u(t, q, u, u_dot)])
    #     coo.extend(dense, (self.la_NDOF, self.uDOF))

    # def Wla_N_q(self, t, q, la_N, coo):
    #     dense = la_N[0] * np.einsum("i,ijk->jk", self.n(t), self.J_P_q(t, q))
    #     # dense_num = np.einsum('i,ijk->jk', la_N, Numerical_derivative(self.g_N_dot_u_dense, order=2)._x(t, q))
    #     # error = np.linalg.norm(dense - dense_num)
    #     # print(f'error: {error}')
    #     coo.extend(dense, (self.uDOF, self.qDOF))
