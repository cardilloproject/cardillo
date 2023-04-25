import numpy as np
from cardillo.math.algebra import cross3, norm, e3, ax2skew
from cardillo.math.numerical_derivative import approx_fprime


class SphereInSphere:
    def __init__(
        self,
        frame,
        R,
        subsystem,
        r,
        mu,
        prox_r_N,
        prox_r_F,
        e_N=None,
        e_F=None,
        frame_ID=np.zeros(3),
        K_r_SP=np.zeros(3),
        la_N0=None,
        la_F0=None,
    ):
        self.frame = frame
        self.R = R
        self.subsystem = subsystem
        self.r = r
        self.mu = np.array([mu])
        self.prox_r_N = np.array([prox_r_N])
        self.prox_r_F = np.array([prox_r_F])

        self.nla_N = 1

        if mu == 0:
            self.nla_F = 0
            self.NF_connectivity = [[]]
        else:
            self.nla_F = 2 * self.nla_N
            self.NF_connectivity = [[0, 1]]
            self.gamma_F = self.__gamma_F

        self.e_N = np.zeros(self.nla_N) if e_N is None else np.array([e_N])
        self.e_F = np.zeros(self.nla_F) if e_F is None else np.array([e_F, e_F])
        self.frame_ID = frame_ID

        self.r_OQ = lambda t: self.frame.r_OP(t)
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

        self.n = lambda t, q: -(
            self.subsystem.r_OP(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
            - self.r_OQ(t)
        ) / norm(
            self.subsystem.r_OP(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP)
            - self.r_OQ(t)
        )

        self.r_OP = lambda t, q: self.subsystem.r_OP(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.v_P = lambda t, q, u: self.subsystem.v_P(
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

        self.Omega = lambda t, q, u: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Omega(t, q, u, frame_ID=self.frame_ID)
        self.J_R = lambda t, q: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID)
        # self.J_R_q = lambda t, q: np.einsum('ijl,jk->ikl', self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID), self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID)) + np.einsum('ij,jkl->ikl', self.subsystem.A_IK(t, q, frame_ID=self.frame_ID), self.subsystem.K_J_R_q(t, q, frame_ID=self.frame_ID))
        # self.Psi = lambda t, q, u, a: self.subsystem.A_IK(t, q, frame_ID=self.frame_ID) @ self.subsystem.K_Psi(t, q, u, a, frame_ID=self.frame_ID)

        self.is_assembled = True

    def g_N(self, t, q):
        return np.array([self.R - self.r - norm(self.r_OP(t, q) - self.r_OQ(t))])

    def g_N_q(self, t, q):
        return approx_fprime(q, lambda q: self.g_N(t, q))

    def g_N_dot(self, t, q, u):
        return np.array([self.n(t, q) @ (self.v_P(t, q, u) - self.v_Q(t))])

    def g_N_ddot(self, t, q, u, u_dot):
        g_N_dot_q = approx_fprime(q, lambda q: self.g_N_dot(t, q, u))
        g_N_dot_u = self.g_N_dot_u(t, q)
        return g_N_dot_q @ self.subsystem.q_dot(t, q, u) + g_N_dot_u @ u_dot

    def g_N_dot_u(self, t, q):
        return np.array([self.n(t, q) @ self.J_P(t, q)])

    def W_N(self, t, q):
        return self.g_N_dot_u(t, q).T

    def Wla_N_q(self, t, q, la_N):
        n_q = approx_fprime(q, lambda t, q: self.n(t, q))
        return la_N[0] * (
            np.einsum("i,ijk->jk", self.n(t, q), self.J_P_q(t, q))
            + np.einsum("ik,ij->jk", n_q, self.J_P(t, q))
        )

    def __gamma_F(self, t, q, u):
        t1t2 = np.zeros((2, 3), dtype=np.common_type(q, u))
        n = self.n(t, q)
        t1t2[0] = e3
        t1t2[1] = cross3(n, t1t2[0])
        v_C = self.v_P(t, q, u) + self.r * cross3(n, self.Omega(t, q, u))
        return t1t2 @ (v_C - self.v_Q(t))

    def gamma_F_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.__gamma_F(t, q, u))

    def gamma_F_dot(self, t, q, u, u_dot):
        # TODO: t1t2_dot(t)
        gamma_T_q = approx_fprime(q, lambda q: self.gamma_F(t, q, u))
        gamma_T_u = self.gamma_F_u(t, q)
        return gamma_T_q @ self.subsystem.q_dot(t, q, u) + gamma_T_u @ u_dot

    def gamma_F_u(self, t, q):
        t1t2 = np.zeros((2, 3))
        n = self.n(t, q)
        t1t2[0] = e3
        t1t2[1] = cross3(n, t1t2[0])
        J_C = self.J_P(t, q) + self.r * ax2skew(n) @ self.J_R(t, q)
        return t1t2 @ J_C

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def Wla_F_q(self, t, q, la_T):
        # J_C_q = self.J_P_q(t, q) + self.r * np.einsum('ij,jkl->ikl', ax2skew(self.n(t)), self.J_R_q(t, q))
        # dense = np.einsum('i,ij,jkl->kl', la_T, self.t1t2(t), J_C_q)
        # return dense
        dense_num = np.einsum(
            "i,ijk->jk",
            la_T,
            approx_fprime(q, lambda t, q, u: self.gamma_F_u(t, q, u)),
        )
        # error = np.linalg.norm(dense - dense_num)
        # print(f'error: {error}')
        return dense_num

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def xi_F(self, t, q, u_pre, u_post):
        return self.gamma_F(t, q, u_post) + self.e_F * self.gamma_F(t, q, u_pre)
