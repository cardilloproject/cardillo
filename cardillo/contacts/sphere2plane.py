import numpy as np
from cardillo.math import approx_fprime
from cardillo.math.algebra import cross3, ax2skew
from scipy.sparse.linalg import spsolve
from cardillo.utility.coo import Coo


class Sphere2Plane:
    def __init__(
        self,
        frame,
        subsystem,
        r,
        mu,
        e_N=None,
        e_F=None,
        frame_ID=np.zeros(3),
        K_r_SP=np.zeros(3),
        la_N0=None,
        la_F0=None,
    ):
        self.frame = frame
        self.subsystem = subsystem
        self.r = r
        self.mu = np.array([mu])
        self.nla_N = 1

        if mu == 0:
            self.nla_F = 0
            self.NF_connectivity = [[]]
        else:
            self.nla_F = 2 * self.nla_N
            self.NF_connectivity = [[0, 1]]
            self.gamma_F = self.__gamma_F

        self.e_N = np.zeros(self.nla_N) if e_N is None else np.array([e_N])
        self.e_F = np.zeros(self.nla_N) if e_F is None else np.array([e_F])
        self.frame_ID = frame_ID

        self.r_OQ = lambda t: self.frame.r_OP(t)
        self.t1t2 = lambda t: self.frame.A_IK(t).T[:2]
        self.n = lambda t: self.frame.A_IK(t)[:, 2]
        self.v_Q = lambda t: self.frame.v_P(t)
        self.a_Q = lambda t: self.frame.a_P(t)

        self.K_r_SP = K_r_SP

        self.la_N0 = np.zeros(self.nla_N) if la_N0 is None else la_N0
        self.la_F0 = np.zeros(self.nla_F) if la_F0 is None else la_F0

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
        self.v_P = lambda t, q, u: self.subsystem.v_P(
            t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.v_P_q = lambda t, q, u: self.subsystem.v_P_q(
            t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.J_P = lambda t, q: self.subsystem.J_P(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.J_P_q = lambda t, q: self.subsystem.J_P_q(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.a_P = lambda t, q, u, a: self.subsystem.a_P(
            t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.a_P_q = lambda t, q, u, a: self.subsystem.a_P_q(
            t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.a_P_u = lambda t, q, u, a: self.subsystem.a_P_u(
            t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )

        self.Omega = lambda t, q, u: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Omega(t, q, u, frame_ID=self.frame_ID)
        self.Omega_q = lambda t, q, u: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Omega_q(t, q, u, frame_ID=self.frame_ID) + np.einsum(
            "ijk,j->ik",
            self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID),
            self.subsystem.K_Omega(t, q, u, frame_ID=self.frame_ID),
        )
        self.Omega_u = lambda t, q, u: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Omega_u(t, q, u, frame_ID=self.frame_ID)
        self.J_R = lambda t, q: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID)
        self.J_R_q = lambda t, q: np.einsum(
            "ijl,jk->ikl",
            self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID),
            self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem.A_IK(t, q, frame_ID=self.frame_ID),
            self.subsystem.K_J_R_q(t, q, frame_ID=self.frame_ID),
        )
        self.Psi = lambda t, q, u, a: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Psi(t, q, u, a, frame_ID=self.frame_ID)
        self.Psi_q = lambda t, q, u, a: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Psi_q(t, q, u, a, frame_ID=self.frame_ID) + np.einsum(
            "ijk,j->ik",
            self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID),
            self.subsystem.K_Psi(t, q, u, a, frame_ID=self.frame_ID),
        )
        self.Psi_u = lambda t, q, u, a: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Psi_u(t, q, u, a, frame_ID=self.frame_ID)

        self.is_assembled = True

    def g_N(self, t, q):
        return np.array([self.n(t) @ (self.r_OP(t, q) - self.r_OQ(t))]) - self.r

    def g_N_q_dense(self, t, q):
        return np.array([self.n(t) @ self.r_OP_q(t, q)])

    def g_N_q(self, t, q, coo):
        coo.extend(self.g_N_q_dense(t, q), (self.la_NDOF, self.qDOF))

    def g_N_dot(self, t, q, u):
        # TODO: n_dot(t)
        return np.array([self.n(t) @ (self.v_P(t, q, u) - self.v_Q(t))])

    def g_N_dot_q_dense(self, t, q, u):
        return np.array([self.n(t) @ self.v_P_q(t, q, u)])

    def g_N_dot_q(self, t, q, u, coo):
        coo.extend(self.g_N_dot_q_dense(t, q, u), (self.la_NDOF, self.qDOF))

    def g_N_dot_u_dense(self, t, q):
        # TODO: n_dot(t)
        return np.array([self.n(t) @ self.J_P(t, q)])

    def g_N_dot_u(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q), (self.la_NDOF, self.uDOF))

    def W_N(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q).T, (self.uDOF, self.la_NDOF))

    def g_N_ddot(self, t, q, u, u_dot):
        return np.array([self.n(t) @ (self.a_P(t, q, u, u_dot) - self.a_Q(t))])

    def g_N_ddot_q(self, t, q, u, u_dot, coo):
        dense = np.array([self.n(t) @ self.a_P_q(t, q, u, u_dot)])
        coo.extend(dense, (self.la_NDOF, self.qDOF))

    def g_N_ddot_u(self, t, q, u, u_dot, coo):
        dense = np.array([self.n(t) @ self.a_P_u(t, q, u, u_dot)])
        coo.extend(dense, (self.la_NDOF, self.uDOF))

    def Wla_N_q(self, t, q, la_N, coo):
        dense = la_N[0] * np.einsum("i,ijk->jk", self.n(t), self.J_P_q(t, q))
        coo.extend(dense, (self.uDOF, self.qDOF))

    def __gamma_F(self, t, q, u):
        v_C = self.v_P(t, q, u) + self.r * cross3(self.n(t), self.Omega(t, q, u))
        return self.t1t2(t) @ (v_C - self.v_Q(t))

    def gamma_F_q_dense(self, t, q, u):
        v_C_q = self.v_P_q(t, q, u) + self.r * ax2skew(self.n(t)) @ self.Omega_q(
            t, q, u
        )
        return self.t1t2(t) @ v_C_q

    def gamma_F_q(self, t, q, u, coo):
        coo.extend(self.gamma_F_q_dense(t, q, u), (self.la_FDOF, self.qDOF))

    def gamma_F_dot(self, t, q, u, u_dot):
        # #TODO: t1t2_dot(t) & n_dot(t)
        Omega = self.Omega(t, q, u)
        r_PC = -self.r * self.n(t)
        a_C = (
            self.a_P(t, q, u, u_dot)
            + cross3(self.Psi(t, q, u, u_dot), r_PC)
            + cross3(Omega, cross3(Omega, r_PC))
        )
        gamma_F_dot = self.t1t2(t) @ (a_C - self.a_Q(t))
        return gamma_F_dot

    def gamma_F_dot_q(self, t, q, u, u_dot, coo):
        # #TODO: t1t2_dot(t) & n_dot(t)
        gamma_F_dot_q_num = approx_fprime(
            q, lambda q: self.gamma_F_dot(t, q, u, u_dot), method="2-point"
        )
        # Omega = self.Omega(t, q, u)
        # r_PC = -self.r * self.n(t)
        # a_C = self.a_P(t, q, u, u_dot) + cross3(self.Psi(t, q, u, u_dot), r_PC) + cross3(Omega, cross3(Omega, r_PC))
        # gamma_F_dot = self.t1t2(t) @ (a_C - self.a_Q(t))
        coo.extend(gamma_F_dot_q_num, (self.la_FDOF, self.qDOF))

    def gamma_F_dot_u(self, t, q, u, u_dot, coo):
        # #TODO: t1t2_dot(t) & n_dot(t)
        gamma_F_dot_u_num = approx_fprime(
            u, lambda u: self.gamma_F_dot(t, q, u, u_dot), method="2-point"
        )
        # Omega = self.Omega(t, q, u)
        # r_PC = -self.r * self.n(t)
        # a_C = self.a_P(t, q, u, u_dot) + cross3(self.Psi(t, q, u, u_dot), r_PC) + cross3(Omega, cross3(Omega, r_PC))
        # gamma_F_dot = self.t1t2(t) @ (a_C - self.a_Q(t))
        coo.extend(gamma_F_dot_u_num, (self.la_FDOF, self.uDOF))

    def gamma_F_u_dense(self, t, q):
        J_C = self.J_P(t, q) + self.r * ax2skew(self.n(t)) @ self.J_R(t, q)
        return self.t1t2(t) @ J_C

    def gamma_F_u(self, t, q, coo):
        coo.extend(self.gamma_F_u_dense(t, q), (self.la_FDOF, self.uDOF))

    def W_F(self, t, q, coo):
        coo.extend(self.gamma_F_u_dense(t, q).T, (self.uDOF, self.la_FDOF))

    def Wla_F_q(self, t, q, la_F, coo):
        J_C_q = self.J_P_q(t, q) + self.r * np.einsum(
            "ij,jkl->ikl", ax2skew(self.n(t)), self.J_R_q(t, q)
        )
        dense = np.einsum("i,ij,jkl->kl", la_F, self.t1t2(t), J_C_q)
        coo.extend(dense, (self.uDOF, self.qDOF))

    def xi_F(self, t, q, u_pre, u_post):
        return self.gamma_F(t, q, u_post) + self.e_F * self.gamma_F(t, q, u_pre)

    # def xi_N_q(self, t, q, u_pre, u_post, coo):
    #     g_N_q_pre = self.g_N_dot_q_dense(t, q, u_pre)
    #     g_N_q_post = self.g_N_dot_q_dense(t, q, u_post)
    #     dense = g_N_q_post + self.e_N * g_N_q_pre
    #     coo.extend(dense, (self.la_NDOF, self.qDOF))

    # def xi_F_q(self, t, q, u_pre, u_post, coo):
    #     gamma_F_q_pre = self.gamma_F_q_dense(t, q, u_pre)
    #     gamma_F_q_post = self.gamma_F_q_dense(t, q, u_post)
    #     dense = gamma_F_q_post + self.e_F * gamma_F_q_pre
    #     coo.extend(dense, (self.la_FDOF, self.qDOF))
