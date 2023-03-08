from cardillo.math import approx_fprime
import numpy as np


class SphericalJoint:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        r_OB0,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
        la_g0=None,
    ):
        self.nla_g = 3
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2
        self.r_OB0 = r_OB0

    def assembler_callback(self):
        qDOF1 = self.subsystem1.qDOF
        qDOF2 = self.subsystem2.qDOF
        local_qDOF1 = self.subsystem1.local_qDOF_P(self.frame_ID1)
        local_qDOF2 = self.subsystem2.local_qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate((qDOF1[local_qDOF1], qDOF2[local_qDOF2]))
        self.__nq1 = nq1 = len(local_qDOF1)
        self.__nq2 = len(local_qDOF2)
        self.__nq = self.__nq1 + self.__nq2

        uDOF1 = self.subsystem1.uDOF
        uDOF2 = self.subsystem2.uDOF
        local_uDOF1 = self.subsystem1.local_uDOF_P(self.frame_ID1)
        local_uDOF2 = self.subsystem2.local_uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate((uDOF1[local_uDOF1], uDOF2[local_uDOF2]))
        self.__nu1 = nu1 = len(local_uDOF1)
        self.__nu2 = len(local_uDOF2)
        self.__nu = self.__nu1 + self.__nu2

        r_OS1 = self.subsystem1.r_OP(
            self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.frame_ID1
        )
        if hasattr(self.subsystem1, "A_IK"):
            A_IK1 = self.subsystem1.A_IK(
                self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.frame_ID1
            )
            # subsystem-fixed vector from subsystem 1 frame origins to B
            K_r_S1B = A_IK1.T @ (self.r_OB0 - r_OS1)
        else:
            K_r_S1B = np.zeros(3)

        r_OS2 = self.subsystem2.r_OP(
            self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.frame_ID2
        )
        if hasattr(self.subsystem2, "A_IK"):
            A_IK2 = self.subsystem2.A_IK(
                self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.frame_ID2
            )
            # subsystem-fixed vector from subsystem 2 frame origins to B
            K_r_S2B = A_IK2.T @ (self.r_OB0 - r_OS2)
        else:
            K_r_S2B = np.zeros(3)

        # auxiliary functions for subsystem 1
        self.r_OB1 = lambda t, q: self.subsystem1.r_OP(
            t, q[:nq1], self.frame_ID1, K_r_S1B
        )
        self.r_OB1_q = lambda t, q: self.subsystem1.r_OP_q(
            t, q[:nq1], self.frame_ID1, K_r_S1B
        )
        self.v_B1 = lambda t, q, u: self.subsystem1.v_P(
            t, q[:nq1], u[:nu1], self.frame_ID1, K_r_S1B
        )
        self.v_B1_q = lambda t, q, u: self.subsystem1.v_P_q(
            t, q[:nq1], u[:nu1], self.frame_ID1, K_r_S1B
        )
        self.a_B1 = lambda t, q, u, u_dot: self.subsystem1.a_P(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1, K_r_S1B
        )
        self.a_B1_q = lambda t, q, u, u_dot: self.subsystem1.a_P_q(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1, K_r_S1B
        )
        self.a_B1_u = lambda t, q, u, u_dot: self.subsystem1.a_P_u(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1, K_r_S1B
        )
        self.J_B1 = lambda t, q: self.subsystem1.J_P(
            t, q[:nq1], self.frame_ID1, K_r_S1B
        )
        self.J_B1_q = lambda t, q: self.subsystem1.J_P_q(
            t, q[:nq1], self.frame_ID1, K_r_S1B
        )

        # auxiliary functions for subsystem 1
        self.r_OB2 = lambda t, q: self.subsystem2.r_OP(
            t, q[nq1:], self.frame_ID2, K_r_S2B
        )
        self.r_OB2_q = lambda t, q: self.subsystem2.r_OP_q(
            t, q[nq1:], self.frame_ID2, K_r_S2B
        )
        self.v_B2 = lambda t, q, u: self.subsystem2.v_P(
            t, q[nq1:], u[nu1:], self.frame_ID2, K_r_S2B
        )
        self.v_B2_q = lambda t, q, u: self.subsystem2.v_P_q(
            t, q[nq1:], u[nu1:], self.frame_ID2, K_r_S2B
        )
        self.a_B2 = lambda t, q, u, u_dot: self.subsystem2.a_P(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2, K_r_S2B
        )
        self.a_B2_q = lambda t, q, u, u_dot: self.subsystem2.a_P_q(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2, K_r_S2B
        )
        self.a_B2_u = lambda t, q, u, u_dot: self.subsystem2.a_P_u(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2, K_r_S2B
        )
        self.J_B2 = lambda t, q: self.subsystem2.J_P(
            t, q[nq1:], self.frame_ID2, K_r_S2B
        )
        self.J_B2_q = lambda t, q: self.subsystem2.J_P_q(
            t, q[nq1:], self.frame_ID2, K_r_S2B
        )

    def g(self, t, q):
        r_OP1 = self.r_OB1(t, q)
        r_OP2 = self.r_OB2(t, q)
        return r_OP2 - r_OP1

    def g_q_dense(self, t, q):
        r_OP1_q = self.r_OB1_q(t, q)
        r_OP2_q = self.r_OB2_q(t, q)
        return np.hstack([-r_OP1_q, r_OP2_q])

    def g_dot(self, t, q, u):
        v_P1 = self.v_B1(t, q, u)
        v_P2 = self.v_B2(t, q, u)
        return v_P2 - v_P1

    def g_dot_q(self, t, q, u, coo):
        nq1 = self.__nq1
        dense = np.zeros((self.nla_g, self.__nq))
        dense[:, :nq1] = -self.v_B1_q(t, q, u)
        dense[:, nq1:] = self.v_B2_q(t, q, u)
        coo.extend(dense, (self.la_gDOF, self.qDOF))

    def g_dot_u(self, t, q, coo):
        raise RuntimeError(
            "This is not tested yet. Run 'test/test_spherical_revolute_joint'."
        )
        coo.extend(self.W_g_dense(t, q).T, (self.la_gDOF, self.uDOF))

    def g_ddot(self, t, q, u, u_dot):
        a_P1 = self.a_B1(t, q, u, u_dot)
        a_P2 = self.a_B2(t, q, u, u_dot)
        return a_P2 - a_P1

    def g_ddot_q(self, t, q, u, u_dot, coo):
        nq1 = self.__nq1
        dense = np.zeros((self.nla_g, self.__nq))
        dense[:, :nq1] = -self.a_B1_q(t, q, u, u_dot)
        dense[:, nq1:] = self.a_B2_q(t, q, u, u_dot)
        coo.extend(dense, (self.la_gDOF, self.qDOF))

    def g_ddot_u(self, t, q, u, u_dot, coo):
        nu1 = self.__nu1
        dense = np.zeros((self.nla_g, self.__nu))
        dense[:, :nu1] = -self.a_B1_u(t, q, u, u_dot)
        dense[:, nu1:] = self.a_B2_u(t, q, u, u_dot)
        coo.extend(dense, (self.la_gDOF, self.uDOF))

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def g_q_T_mu_q(self, t, q, mu, coo):
        dense = approx_fprime(q, lambda q: self.g_q_dense(t, q).T @ mu)
        coo.extend(dense, (self.qDOF, self.qDOF))

    def W_g_dense(self, t, q):
        nu1 = self.__nu1
        J_P1 = self.J_B1(t, q)
        J_P2 = self.J_B2(t, q)
        W_g = np.zeros((self.__nu, self.nla_g), dtype=q.dtype)
        W_g[:nu1, :] = -J_P1.T
        W_g[nu1:, :] = J_P2.T
        return W_g

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        nq1 = self.__nq1
        nu1 = self.__nu1
        J_P1_q = self.J_B1_q(t, q)
        J_P2_q = self.J_B2_q(t, q)

        # dense blocks
        dense = np.zeros((self.__nu, self.__nq))
        dense[:nu1, :nq1] = np.einsum("i,ijk->jk", -la_g, J_P1_q)
        dense[nu1:, nq1:] = np.einsum("i,ijk->jk", la_g, J_P2_q)

        coo.extend(dense, (self.uDOF, self.qDOF))
