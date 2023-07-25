import numpy as np
from cardillo.constraints._base import (
    concatenate_qDOF,
    concatenate_uDOF,
    auxiliary_functions,
)
from cardillo.math import approx_fprime
import warnings


class Rod:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
        K1_r_SP1=np.zeros(3),
        K2_r_SP2=np.zeros(3),
    ):
        self.nla_g = 1

        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.K1_r_SP1 = K1_r_SP1

        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2
        self.K2_r_SP2 = K2_r_SP2

    def assembler_callback(self):
        concatenate_qDOF(self)
        concatenate_uDOF(self)

        auxiliary_functions(self, self.K1_r_SP1, self.K2_r_SP2, np.eye(3), np.eye(3))

        r_OB10 = self.r_OB1(self.subsystem1.t0, self.subsystem1.q0)
        r_OB20 = self.r_OB2(self.subsystem2.t0, self.subsystem2.q0)
        self.dist = np.linalg.norm(r_OB20 - r_OB10)

        if self.dist < 1e-6:
            raise ValueError("Rod.distance is close to zero.")

    def g(self, t, q):
        r_B1B2 = self.r_OB2(t, q) - self.r_OB1(t, q)
        return r_B1B2 @ r_B1B2 - self.dist**2

    def g_q(self, t, q):
        r_B1B2 = self.r_OB2(t, q) - self.r_OB1(t, q)
        r_OB1_q = self.r_OB1_q1(t, q)
        r_OB2_q = self.r_OB2_q2(t, q)
        return np.array([2 * r_B1B2 @ np.hstack([-r_OB1_q, r_OB2_q])], dtype=q.dtype)

    def g_dot(self, t, q, u):
        r_B1B2 = self.r_OB2(t, q) - self.r_OB1(t, q)
        return 2 * r_B1B2 @ (self.v_B2(t, q, u) - self.v_B1(t, q, u))

    def g_dot_q(self, t, q, u):
        r_B1B2 = self.r_OB2(t, q) - self.r_OB1(t, q)
        v_B1B2 = self.v_B2(t, q, u) - self.v_B1(t, q, u)

        g_dot_q = np.zeros((self.nla_g, self._nq), dtype=np.common_type(q, u))
        g_dot_q[:, : self._nq1] = -2 * (
            r_B1B2 @ self.v_B1_q1(t, q, u) + v_B1B2 @ self.r_OB1_q1(t, q)
        )
        g_dot_q[:, self._nq1 :] = 2 * (
            r_B1B2 @ self.v_B2_q2(t, q, u) + v_B1B2 @ self.r_OB2_q2(t, q)
        )

        return g_dot_q

    def g_dot_u(self, t, q):
        return self.W_g(t, q).T

    def g_ddot(self, t, q, u, u_dot):
        r_B1B2 = self.r_OB2(t, q) - self.r_OB1(t, q)
        v_B1B2 = self.v_B2(t, q, u) - self.v_B1(t, q, u)
        a_B1B2 = self.a_B2(t, q, u, u_dot) - self.a_B1(t, q, u, u_dot)

        return 2 * v_B1B2 @ v_B1B2 + 2 * r_B1B2 @ a_B1B2

    def g_ddot_q(self, t, q, u, u_dot):
        r_B1B2 = self.r_OB2(t, q) - self.r_OB1(t, q)
        v_B1B2 = self.v_B2(t, q, u) - self.v_B1(t, q, u)
        a_B1B2 = self.a_B2(t, q, u, u_dot) - self.a_B1(t, q, u, u_dot)

        g_ddot_q = np.zeros((self.nla_g, self._nq), dtype=np.common_type(q, u, u_dot))
        g_ddot_q[:, : self._nq1] = -2 * (
            2 * v_B1B2 @ self.v_B1_q1(t, q, u)
            + a_B1B2 @ self.r_OB1_q1(t, q)
            + r_B1B2 @ self.a_B1_q1(t, q, u, u_dot)
        )
        g_ddot_q[:, self._nq1 :] = 2 * (
            2 * v_B1B2 @ self.v_B2_q2(t, q, u)
            + a_B1B2 @ self.r_OB2_q2(t, q)
            + r_B1B2 @ self.a_B2_q2(t, q, u, u_dot)
        )

        return g_ddot_q

    def g_ddot_u(self, t, q, u, u_dot):
        r_B1B2 = self.r_OB2(t, q) - self.r_OB1(t, q)
        v_B1B2 = self.v_B2(t, q, u) - self.v_B1(t, q, u)

        g_ddot_u = np.zeros((self.nla_g, self._nu), dtype=np.common_type(t, u, u_dot))
        g_ddot_u[:, : self._nu1] = -2 * (
            2 * v_B1B2 @ self.J_B1(t, q) + r_B1B2 @ self.a_B1_u1(t, q, u, u_dot)
        )
        g_ddot_u[:, self._nu1 :] = 2 * (
            2 * v_B1B2 @ self.J_B2(t, q) + r_B1B2 @ self.a_B2_u2(t, q, u, u_dot)
        )

        return g_ddot_u

    def g_q_T_mu_q(self, t, q, mu):
        warnings.warn("Rod.g_q_T_mu_q uses approx_fprime.")
        return approx_fprime(q, lambda q: self.g_q(t, q).T @ mu)

    def W_g(self, t, q):
        r_B1B2 = self.r_OB2(t, q) - self.r_OB1(t, q)
        J_B1 = self.J_B1(t, q)
        J_B2 = self.J_B2(t, q)
        return (
            2
            * np.array(
                [np.concatenate([-J_B1.T @ r_B1B2, J_B2.T @ r_B1B2])], dtype=q.dtype
            ).T
        )

    def Wla_g_q(self, t, q, la_g):
        nq1 = self._nq1
        nu1 = self._nu1
        r_B1B2 = self.r_OB2(t, q) - self.r_OB1(t, q)
        r_OB1_q1 = self.r_OB1_q1(t, q)
        r_OB2_q2 = self.r_OB2_q2(t, q)
        J_B1 = self.J_B1(t, q)
        J_B2 = self.J_B2(t, q)
        J_B1_q1 = self.J_B1_q1(t, q)
        J_B2_q2 = self.J_B2_q2(t, q)

        Wla_g_q = np.zeros((self._nu, self._nq))
        Wla_g_q[:nu1, :nq1] = (
            2 * la_g * (J_B1.T @ r_OB1_q1 - np.einsum("i,ijk->jk", r_B1B2, J_B1_q1))
        )
        Wla_g_q[:nu1, nq1:] = -2 * la_g * J_B1.T @ r_OB2_q2
        Wla_g_q[nu1:, :nq1] = -2 * la_g * J_B2.T @ r_OB1_q1
        Wla_g_q[nu1:, nq1:] = (
            2 * la_g * (J_B2.T @ r_OB2_q2 + np.einsum("i,ijk->jk", r_B1B2, J_B2_q2))
        )

        return Wla_g_q

    def export(self, sol_i, **kwargs):
        points = [
            self.r_OP1(sol_i.t, sol_i.q[self.qDOF]),
            self.r_OP2(sol_i.t, sol_i.q[self.qDOF]),
        ]
        cells = [("line", [[0, 1]])]

        return points, cells, None, None
