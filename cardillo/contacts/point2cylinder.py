import numpy as np

from cardillo.math import approx_fprime


class Point2Cylinder:
    def __init__(
        self,
        frame,
        subsystem,
        radius,
        prox_r_N,
        e_N=None,
        frame_ID=np.zeros(3),
        K_r_SP=np.zeros(3),
        la_N0=None,
    ):

        self.frame = frame
        self.subsystem = subsystem
        self.radius = radius
        self.prox_r_N = np.array([prox_r_N])
        self.prox_r_T = np.array([prox_r_N])
        self.mu_T = np.array([0])

        self.nla_N = 1

        # TODO: Can we delete these for static examples?
        self.nla_T = 0
        self.NT_connectivity = [[]]

        self.e_N = np.zeros(self.nla_N) if e_N is None else np.array([e_N])
        self.e_T = np.zeros(self.nla_T)
        self.frame_ID = frame_ID

        self.r_OB = lambda t: self.frame.r_OP(t)
        self.d3 = lambda t: self.frame.A_IK(t)[:, 2]
        self.v_B = lambda t: self.frame.v_P(t)
        self.a_B = lambda t: self.frame.a_P(t)

        self.K_r_SP = K_r_SP

        self.la_N0 = np.zeros(self.nla_N) if la_N0 is None else la_N0
        self.la_T0 = np.zeros(self.nla_T)

        self.is_assembled = False

    def assembler_callback(self):
        qDOF = self.subsystem.local_qDOF_P(self.frame_ID)
        self.qDOF = self.subsystem.qDOF[qDOF]
        self.nq = len(self.qDOF)

        uDOF = self.subsystem.local_uDOF_P(self.frame_ID)
        self.uDOF = self.subsystem.uDOF[uDOF]
        self.nu = len(self.uDOF)

        self.r_OP = lambda t, q: self.subsystem.r_OP(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.r_OP_q = lambda t, q: self.subsystem.r_OP_q(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        # self.v_P = lambda t, q, u: self.subsystem.v_P(
        #     t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        # )
        # self.v_P_q = lambda t, q, u: self.subsystem.v_P_q(
        #     t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        # )
        self.J_P = lambda t, q: self.subsystem.J_P(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        # self.J_P_q = lambda t, q: self.subsystem.J_P_q(
        #     t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        # )
        # self.a_P = lambda t, q, u, a: self.subsystem.a_P(
        #     t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        # )
        # self.a_P_q = lambda t, q, u, a: self.subsystem.a_P_q(
        #     t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        # )
        # self.a_P_u = lambda t, q, u, a: self.subsystem.a_P_u(
        #     t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        # )

        # self.Omega = lambda t, q, u: self.subsystem.A_IK(
        #     t, q, frame_ID=self.frame_ID
        # ) @ self.subsystem.K_Omega(t, q, u, frame_ID=self.frame_ID)
        # self.Omega_q = lambda t, q, u: self.subsystem.A_IK(
        #     t, q, frame_ID=self.frame_ID
        # ) @ self.subsystem.K_Omega_q(t, q, u, frame_ID=self.frame_ID) + np.einsum(
        #     "ijk,j->ik",
        #     self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID),
        #     self.subsystem.K_Omega(t, q, u, frame_ID=self.frame_ID),
        # )
        # self.Omega_u = lambda t, q, u: self.subsystem.A_IK(
        #     t, q, frame_ID=self.frame_ID
        # ) @ self.subsystem.K_Omega_u(t, q, u, frame_ID=self.frame_ID)
        # self.J_R = lambda t, q: self.subsystem.A_IK(
        #     t, q, frame_ID=self.frame_ID
        # ) @ self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID)
        # self.J_R_q = lambda t, q: np.einsum(
        #     "ijl,jk->ikl",
        #     self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID),
        #     self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID),
        # ) + np.einsum(
        #     "ij,jkl->ikl",
        #     self.subsystem.A_IK(t, q, frame_ID=self.frame_ID),
        #     self.subsystem.K_J_R_q(t, q, frame_ID=self.frame_ID),
        # )
        # self.Psi = lambda t, q, u, a: self.subsystem.A_IK(
        #     t, q, frame_ID=self.frame_ID
        # ) @ self.subsystem.K_Psi(t, q, u, a, frame_ID=self.frame_ID)
        # self.Psi_q = lambda t, q, u, a: self.subsystem.A_IK(
        #     t, q, frame_ID=self.frame_ID
        # ) @ self.subsystem.K_Psi_q(t, q, u, a, frame_ID=self.frame_ID) + np.einsum(
        #     "ijk,j->ik",
        #     self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID),
        #     self.subsystem.K_Psi(t, q, u, a, frame_ID=self.frame_ID),
        # )
        # self.Psi_u = lambda t, q, u, a: self.subsystem.A_IK(
        #     t, q, frame_ID=self.frame_ID
        # ) @ self.subsystem.K_Psi_u(t, q, u, a, frame_ID=self.frame_ID)

    def g_N(self, t, q):
        r_OB = self.r_OB(t)
        r_OP = self.r_OP(t, q)
        r_BP = r_OP - r_OB
        d3 = self.d3(t)
        xi = r_BP @ d3
        r_OQ = r_OB + xi * d3
        r_QP = r_OP - r_OQ
        return np.array([r_QP @ r_QP - self.radius**2])

    def g_N_q_dense(self, t, q):
        r_OB = self.r_OB(t)
        r_OP = self.r_OP(t, q)
        r_BP = r_OP - r_OB
        d3 = self.d3(t)
        xi = r_BP @ d3
        r_OQ = r_OB + xi * d3
        r_QP = r_OP - r_OQ
        r_OP_q = self.r_OP_q(t, q)
        return np.array([2 * r_QP @ r_OP_q])

    def g_N_q(self, t, q, coo):
        coo.extend(self.g_N_q_dense(t, q), (self.la_NDOF, self.qDOF))

    # def g_N_dot(self, t, q, u):
    #     # TODO: n_dot(t)
    #     return np.array([self.d3(t) @ (self.v_P(t, q, u) - self.v_B(t))])

    # def g_N_dot_q_dense(self, t, q, u):
    #     return np.array([self.d3(t) @ self.v_P_q(t, q, u)])

    # def g_N_dot_q(self, t, q, u, coo):
    #     coo.extend(self.g_N_dot_q_dense(t, q, u), (self.la_NDOF, self.qDOF))

    # def g_N_dot_u_dense(self, t, q):
    #     # TODO: n_dot(t)
    #     return np.array([self.d3(t) @ self.J_P(t, q)])

    # def g_N_dot_u(self, t, q, coo):
    #     coo.extend(self.g_N_dot_u_dense(t, q), (self.la_NDOF, self.uDOF))

    # def xi_N(self, t, q, u_pre, u_post):
    #     return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    # def xi_N_q(self, t, q, u_pre, u_post, coo):
    #     g_N_q_pre = self.g_N_dot_q_dense(t, q, u_pre)
    #     g_N_q_post = self.g_N_dot_q_dense(t, q, u_post)
    #     dense = g_N_q_post + self.e_N * g_N_q_pre
    #     coo.extend(dense, (self.la_NDOF, self.qDOF))

    def W_N_dense(self, t, q):
        r_OB = self.r_OB(t)
        r_OP = self.r_OP(t, q)
        r_BP = r_OP - r_OB
        d3 = self.d3(t)
        xi = r_BP @ d3
        r_OQ = r_OB + xi * d3
        r_QP = r_OP - r_OQ
        J_P = self.J_P(t, q)
        # return np.array([J_P.T @ r_QP * 2])
        return np.array([2 * r_QP @ J_P]).T

    def W_N(self, t, q, coo):
        coo.extend(self.W_N_dense(t, q), (self.uDOF, self.la_NDOF))

    # def g_N_ddot(self, t, q, u, u_dot):
    #     return np.array([self.d3(t) @ (self.a_P(t, q, u, u_dot) - self.a_B(t))])

    # def g_N_ddot_q(self, t, q, u, u_dot, coo):
    #     dense = np.array([self.d3(t) @ self.a_P_q(t, q, u, u_dot)])
    #     coo.extend(dense, (self.la_NDOF, self.qDOF))

    # def g_N_ddot_u(self, t, q, u, u_dot, coo):
    #     dense = np.array([self.d3(t) @ self.a_P_u(t, q, u, u_dot)])
    #     coo.extend(dense, (self.la_NDOF, self.uDOF))

    def Wla_N_q(self, t, q, la_N, coo):
        f = lambda q: self.W_N_dense(t, q) @ la_N
        dense = approx_fprime(q, f)
        coo.extend(dense, (self.uDOF, self.qDOF))

    # def xi_N(self, t, q, u_pre, u_post):
    #     return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    # def xi_N_q(self, t, q, u_pre, u_post, coo):
    #     g_N_q_pre = self.g_N_dot_q_dense(t, q, u_pre)
    #     g_N_q_post = self.g_N_dot_q_dense(t, q, u_post)
    #     dense = g_N_q_post + self.e_N * g_N_q_pre
    #     coo.extend(dense, (self.la_NDOF, self.qDOF))
