import numpy as np
from cardillo.math import approx_fprime
from cardillo.constraints._base import (
    concatenate_qDOF,
    concatenate_uDOF,
    auxiliary_functions,
)


class Spherical:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        r_OB0,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
    ):
        self.subsystem1 = subsystem1
        self.subsystem2 = subsystem2
        self.frame_ID1 = frame_ID1
        self.frame_ID2 = frame_ID2

        self.r_OB0 = r_OB0

        self.nla_g = 3

    def assembler_callback(self):
        local_qDOF1, local_qDOF2 = concatenate_qDOF(self)
        concatenate_uDOF(self)

        r_OP10 = self.subsystem1.r_OP(
            self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.frame_ID1
        )
        if hasattr(self.subsystem1, "A_IK"):
            A_IK10 = self.subsystem1.A_IK(
                self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.frame_ID1
            )
            # subsystem-fixed vector from subsystem 1 frame origins to B
            K1_r_P1B0 = A_IK10.T @ (self.r_OB0 - r_OP10)
        else:
            K1_r_P1B0 = np.zeros(3)

        r_OP20 = self.subsystem2.r_OP(
            self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.frame_ID2
        )
        if hasattr(self.subsystem2, "A_IK"):
            A_IK20 = self.subsystem2.A_IK(
                self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.frame_ID2
            )
            # subsystem-fixed vector from subsystem 2 frame origins to B
            K2_r_P2B0 = A_IK20.T @ (self.r_OB0 - r_OP20)
        else:
            K2_r_P2B0 = np.zeros(3)

        auxiliary_functions(self, K1_r_P1B0, K2_r_P2B0)

    def g(self, t, q):
        return self.r_OB2(t, q) - self.r_OB1(t, q)

    def g_q_dense(self, t, q):
        return np.hstack([-self.r_OB1_q1(t, q), self.r_OB2_q2(t, q)])

    def g_dot(self, t, q, u):
        return self.v_B2(t, q, u) - self.v_B1(t, q, u)

    def g_dot_q(self, t, q, u, coo):
        nq1 = self._nq1
        dense = np.zeros((self.nla_g, self._nq))
        dense[:, :nq1] = -self.v_B1_q1(t, q, u)
        dense[:, nq1:] = self.v_B2_q2(t, q, u)
        coo.extend(dense, (self.la_gDOF, self.qDOF))

    def g_dot_u(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q).T, (self.la_gDOF, self.uDOF))

    def g_ddot(self, t, q, u, u_dot):
        return self.a_B2(t, q, u, u_dot) - self.a_B1(t, q, u, u_dot)

    def g_ddot_q(self, t, q, u, u_dot, coo):
        nq1 = self._nq1
        dense = np.zeros((self.nla_g, self._nq))
        dense[:, :nq1] = -self.a_B1_q1(t, q, u, u_dot)
        dense[:, nq1:] = self.a_B2_q2(t, q, u, u_dot)
        coo.extend(dense, (self.la_gDOF, self.qDOF))

    def g_ddot_u(self, t, q, u, u_dot, coo):
        nu1 = self._nu1
        dense = np.zeros((self.nla_g, self._nu))
        dense[:, :nu1] = -self.a_B1_u1(t, q, u, u_dot)
        dense[:, nu1:] = self.a_B2_u2(t, q, u, u_dot)
        coo.extend(dense, (self.la_gDOF, self.uDOF))

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g_dense(self, t, q):
        nu1 = self._nu1
        W_g = np.zeros((self._nu, self.nla_g), dtype=q.dtype)
        W_g[:nu1, :] = -self.J_B1(t, q).T
        W_g[nu1:, :] = self.J_B2(t, q).T
        return W_g

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        nq1 = self._nq1
        nu1 = self._nu1

        dense = np.zeros((self._nu, self._nq))
        dense[:nu1, :nq1] = np.einsum("i,ijk->jk", -la_g, self.J_B1_q1(t, q))
        dense[nu1:, nq1:] = np.einsum("i,ijk->jk", la_g, self.J_B2_q2(t, q))

        coo.extend(dense, (self.uDOF, self.qDOF))

    # TODO analytical derivative
    def g_q_T_mu_q(self, t, q, mu, coo):
        dense = approx_fprime(q, lambda q: self.g_q_dense(t, q).T @ mu)
        coo.extend(dense, (self.qDOF, self.qDOF))
