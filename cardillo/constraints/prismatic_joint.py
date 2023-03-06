import numpy as np
from cardillo.math.algebra import cross3, ax2skew

# TODO: Derive linear guidance from this joint by removing the third constraint.
class PrismaticJoint:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        r_OB,
        A_IB,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
        la_g0=None,
    ):
        # 1 position dof in exB direction
        self.nla_g = 5
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2
        self.r_OB = r_OB
        self.A_IB = A_IB

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

        A_IK1 = self.subsystem1.A_IK(
            self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.frame_ID1
        )
        A_IK2 = self.subsystem2.A_IK(
            self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.frame_ID2
        )
        A_K1B1 = A_IK1.T @ self.A_IB
        A_K2B2 = A_IK2.T @ self.A_IB

        r_OS1 = self.subsystem1.r_OP(
            self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.frame_ID1
        )
        r_OS2 = self.subsystem2.r_OP(
            self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.frame_ID2
        )
        K_r_SP1 = A_IK1.T @ (self.r_OB - r_OS1)
        K_r_SP2 = A_IK2.T @ (self.r_OB - r_OS2)

        self.r_OP1 = lambda t, q: self.subsystem1.r_OP(
            t, q[:nq1], self.frame_ID1, K_r_SP1
        )
        self.r_OP1_q = lambda t, q: self.subsystem1.r_OP_q(
            t, q[:nq1], self.frame_ID1, K_r_SP1
        )
        self.v_P1 = lambda t, q, u: self.subsystem1.v_P(
            t, q[:nq1], u[:nu1], self.frame_ID1, K_r_SP1
        )
        self.a_P1 = lambda t, q, u, u_dot: self.subsystem1.a_P(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1, K_r_SP1
        )
        self.J_P1 = lambda t, q: self.subsystem1.J_P(
            t, q[:nq1], self.frame_ID1, K_r_SP1
        )
        self.J_P1_q = lambda t, q: self.subsystem1.J_P_q(
            t, q[:nq1], self.frame_ID1, K_r_SP1
        )
        self.A_IB1 = (
            lambda t, q: self.subsystem1.A_IK(t, q[:nq1], self.frame_ID1) @ A_K1B1
        )
        self.A_IB1_q1 = lambda t, q: np.einsum(
            "ijl,jk->ikl", self.subsystem1.A_IK_q(t, q[:nq1], self.frame_ID1), A_K1B1
        )
        self.Omega1 = lambda t, q, u: self.subsystem1.A_IK(
            t, q[:nq1], self.frame_ID1
        ) @ self.subsystem1.K_Omega(t, q[:nq1], u[:nu1], self.frame_ID1)
        self.Psi1 = lambda t, q, u, u_dot: self.subsystem1.A_IK(
            t, q[:nq1], self.frame_ID1
        ) @ self.subsystem1.K_Psi(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1)
        self.K_J_R1 = lambda t, q: self.subsystem1.K_J_R(t, q[:nq1], self.frame_ID1)
        self.K_J_R1_q = lambda t, q: self.subsystem1.K_J_R_q(t, q[:nq1], self.frame_ID1)
        self.J_R1 = lambda t, q: self.subsystem1.A_IK(
            t, q[:nq1], self.frame_ID1
        ) @ self.subsystem1.K_J_R(t, q[:nq1], self.frame_ID1)
        self.J_R1_q = lambda t, q: np.einsum(
            "ijk,jl->ilk",
            self.subsystem1.A_IK_q(t, q[:nq1], self.frame_ID1),
            self.subsystem1.K_J_R(t, q[:nq1], self.frame_ID1),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem1.A_IK(t, q[:nq1], self.frame_ID1),
            self.subsystem1.K_J_R_q(t, q[:nq1], self.frame_ID1),
        )

        self.r_OP2 = lambda t, q: self.subsystem2.r_OP(
            t, q[nq1:], self.frame_ID2, K_r_SP2
        )
        self.r_OP2_q = lambda t, q: self.subsystem2.r_OP_q(
            t, q[nq1:], self.frame_ID2, K_r_SP2
        )
        self.v_P2 = lambda t, q, u: self.subsystem2.v_P(
            t, q[nq1:], u[nu1:], self.frame_ID2, K_r_SP2
        )
        self.a_P2 = lambda t, q, u, u_dot: self.subsystem2.a_P(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2, K_r_SP2
        )
        self.J_P2 = lambda t, q: self.subsystem2.J_P(
            t, q[nq1:], self.frame_ID2, K_r_SP2
        )
        self.J_P2_q = lambda t, q: self.subsystem2.J_P_q(
            t, q[nq1:], self.frame_ID2, K_r_SP2
        )
        self.A_IB2 = (
            lambda t, q: self.subsystem2.A_IK(t, q[nq1:], self.frame_ID2) @ A_K2B2
        )
        self.A_IB2_q2 = lambda t, q: np.einsum(
            "ijk,jl->ilk", self.subsystem2.A_IK_q(t, q[nq1:], self.frame_ID2), A_K2B2
        )
        self.Omega2 = lambda t, q, u: self.subsystem2.A_IK(
            t, q[nq1:], self.frame_ID2
        ) @ self.subsystem2.K_Omega(t, q[nq1:], u[nu1:], self.frame_ID2)
        self.Psi2 = lambda t, q, u, u_dot: self.subsystem2.A_IK(
            t, q[nq1:], self.frame_ID2
        ) @ self.subsystem2.K_Psi(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2)
        self.K_J_R2 = lambda t, q: self.subsystem2.K_J_R(t, q[nq1:], self.frame_ID2)
        self.K_J_R2_q = lambda t, q: self.subsystem2.K_J_R_q(t, q[nq1:], self.frame_ID2)
        self.J_R2 = lambda t, q: self.subsystem2.A_IK(
            t, q[nq1:], self.frame_ID2
        ) @ self.subsystem2.K_J_R(t, q[nq1:], self.frame_ID2)
        self.J_R2_q = lambda t, q: np.einsum(
            "ijk,jl->ilk",
            self.subsystem2.A_IK_q(t, q[nq1:], self.frame_ID2),
            self.subsystem2.K_J_R(t, q[nq1:], self.frame_ID2),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem2.A_IK(t, q[nq1:], self.frame_ID2),
            self.subsystem2.K_J_R_q(t, q[nq1:], self.frame_ID2),
        )

    def g(self, t, q):
        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        r_P1P2 = r_OP2 - r_OP1
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T
        return np.array([r_P1P2 @ ey1, r_P1P2 @ ez1, ex1 @ ey2, ey1 @ ez2, ez1 @ ex2])

    def g_q_dense(self, t, q):
        nq1 = self.__nq1
        g_q = np.zeros((self.nla_g, self.__nq), dtype=q.dtype)

        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        r_P1P2 = r_OP2 - r_OP1
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        A_IB1_q1 = self.A_IB1_q1(t, q)
        ex1_q = A_IB1_q1[:, 0]
        ey1_q = A_IB1_q1[:, 1]
        ez1_q = A_IB1_q1[:, 2]
        A_IB2_q2 = self.A_IB2_q2(t, q)
        ex2_q = A_IB2_q2[:, 0]
        ey2_q = A_IB2_q2[:, 1]
        ez2_q = A_IB2_q2[:, 2]

        # return np.array([r_P1P2 @ ey1, r_P1P2 @ ez1, ex1 @ ey2, ey1 @ ez2, ez1 @ ex2])
        g_q[0, :nq1] = -ey1 @ self.r_OP1_q(t, q) + r_P1P2 @ ey1_q
        g_q[0, nq1:] = ey1 @ self.r_OP2_q(t, q)
        g_q[1, :nq1] = -ez1 @ self.r_OP1_q(t, q) + r_P1P2 @ ez1_q
        g_q[1, nq1:] = ez1 @ self.r_OP2_q(t, q)
        g_q[2, :nq1] = ey2 @ ex1_q
        g_q[2, nq1:] = ex1 @ ey2_q
        g_q[3, :nq1] = ez2 @ ey1_q
        g_q[3, nq1:] = ey1 @ ez2_q
        g_q[4, :nq1] = ex2 @ ez1_q
        g_q[4, nq1:] = ez1 @ ex2_q
        return g_q

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g, dtype=np.common_type(q, u))

        r_P1P2 = self.r_OP2(t, q) - self.r_OP1(t, q)
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        v_P1P2 = self.v_P2(t, q, u) - self.v_P1(t, q, u)
        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)

        Omega21 = Omega1 - Omega2

        # return np.array([r_P1P2 @ ey1, r_P1P2 @ ez1, ex1 @ ey2, ey1 @ ez2, ez1 @ ex2])
        g_dot[0] = v_P1P2 @ ey1 + cross3(ey1, r_P1P2) @ Omega1
        g_dot[1] = v_P1P2 @ ez1 + cross3(ez1, r_P1P2) @ Omega1
        g_dot[2] = cross3(ex1, ey2) @ Omega21
        g_dot[3] = cross3(ey1, ez2) @ Omega21
        g_dot[4] = cross3(ez1, ex2) @ Omega21
        return g_dot

    def g_dot_q(self, t, q, u, coo):
        raise NotImplementedError

    def g_dot_u(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q).T, (self.la_gDOF, self.uDOF))

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g, dtype=np.common_type(q, u, u_dot))

        r_P1P2 = self.r_OP2(t, q) - self.r_OP1(t, q)
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        v_P1P2 = self.v_P2(t, q, u) - self.v_P1(t, q, u)
        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        Omega21 = Omega1 - Omega2

        a_P1P2 = self.a_P2(t, q, u, u_dot) - self.a_P1(t, q, u, u_dot)
        Psi1 = self.Psi1(t, q, u, u_dot)
        Psi2 = self.Psi2(t, q, u, u_dot)
        Psi21 = Psi1 - Psi2

        # g_dot[0] = v_P1P2 @ ey1 + cross3(ey1, r_P1P2) @ Omega1
        g_ddot[0] = (
            a_P1P2 @ ey1
            + v_P1P2 @ cross3(Omega1, ey1)
            + cross3(cross3(Omega1, ey1), r_P1P2) @ Omega1
            + cross3(ey1, v_P1P2) @ Omega1
            + cross3(ey1, r_P1P2) @ Psi1
        )

        # g_dot[1] = v_P1P2 @ ez1 + cross3(ez1, r_P1P2) @ Omega1
        g_ddot[1] = (
            a_P1P2 @ ez1
            + v_P1P2 @ cross3(Omega1, ez1)
            + cross3(cross3(Omega1, ez1), r_P1P2) @ Omega1
            + cross3(ez1, v_P1P2) @ Omega1
            + cross3(ez1, r_P1P2) @ Psi1
        )

        # g_dot[2] = cross3(ex1, ey2) @ Omega21
        g_ddot[2] = (
            cross3(cross3(Omega1, ex1), ey2) + cross3(ex1, cross3(Omega2, ey2))
        ) @ Omega21 + cross3(ex1, ey2) @ Psi21

        # g_dot[3] = cross3(ey1, ez2) @ Omega21
        g_ddot[3] = (
            cross3(cross3(Omega1, ey1), ez2) + cross3(ey1, cross3(Omega2, ez2))
        ) @ Omega21 + cross3(ey1, ez2) @ Psi21

        # g_dot[4] = cross3(ez1, ex2) @ Omega21
        g_ddot[4] = (
            cross3(cross3(Omega1, ez1), ex2) + cross3(ez1, cross3(Omega2, ex2))
        ) @ Omega21 + cross3(ez1, ex2) @ Psi21

        return g_ddot

    def g_ddot_q(self, t, q, u, u_dot, coo):
        raise NotImplementedError

    def g_ddot_u(self, t, q, u, u_dot, coo):
        raise NotImplementedError

    def W_g_dense(self, t, q):
        nu1 = self.__nu1
        W_g = np.zeros((self.__nu, self.nla_g), dtype=q.dtype)

        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        r_P1P2 = r_OP2 - r_OP1
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        J = np.hstack([J_R1, -J_R2])

        W_g[:nu1, 0] = -ey1 @ J_P1 + cross3(ey1, r_P1P2) @ J_R1
        W_g[nu1:, 0] = ey1 @ J_P2
        W_g[:nu1, 1] = -ez1 @ J_P1 + cross3(ez1, r_P1P2) @ J_R1
        W_g[nu1:, 1] = ez1 @ J_P2
        W_g[:, 2] = cross3(ex1, ey2) @ J
        W_g[:, 3] = cross3(ey1, ez2) @ J
        W_g[:, 4] = cross3(ez1, ex2) @ J

        return W_g

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        nu1 = self.__nu1
        nq1 = self.__nq1
        dense = np.zeros((self.__nu, self.__nq), dtype=np.common_type(q, la_g))

        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        r_P1P2 = r_OP2 - r_OP1
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)

        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        A_IB1_q1 = self.A_IB1_q1(t, q)
        ex1_q = A_IB1_q1[:, 0]
        ey1_q = A_IB1_q1[:, 1]
        ez1_q = A_IB1_q1[:, 2]

        A_IB2_q2 = self.A_IB2_q2(t, q)
        ex2_q = A_IB2_q2[:, 0]
        ey2_q = A_IB2_q2[:, 1]
        ez2_q = A_IB2_q2[:, 2]

        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        J_P1_q = self.J_P1_q(t, q)
        J_P2_q = self.J_P2_q(t, q)
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        J_R1_q = self.J_R1_q(t, q)
        J_R2_q = self.J_R2_q(t, q)

        # W_g[:nu1, 0] * la_g[0] = -ey1 @ J_P1 * la_g[0] + cross3(ey1, r_P1P2) @ J_R1 * la_g[0]
        # W_g[nu1:, 0] * la_g[0]=  ey1 @ J_P2 * la_g[0]
        n = cross3(ey1, r_P1P2)
        n_q1 = -ax2skew(r_P1P2) @ ey1_q - ax2skew(ey1) @ r_OP1_q
        n_q2 = ax2skew(ey1) @ r_OP2_q
        dense[:nu1, :nq1] += (
            np.einsum("i,ijk->jk", -la_g[0] * ey1, J_P1_q)
            + np.einsum("ij,ik->kj", -la_g[0] * ey1_q, J_P1)
            + np.einsum("i,ijk->jk", la_g[0] * n, J_R1_q)
            + np.einsum("ij,ik->kj", la_g[0] * n_q1, J_R1)
        )
        dense[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[0] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum("ij,ik->kj", la_g[0] * ey1_q, J_P2)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", la_g[0] * ey1, J_P2_q)

        # W_g[:nu1, 1] * la_g[1] = -ez1 @ J_P1 * la_g[1] + cross3(ez1, r_P1P2) @ J_R1 * la_g[1]
        # W_g[nu1:, 1] * la_g[1]=  ez1 @ J_P2 * la_g[1]
        n = cross3(ez1, r_P1P2)
        n_q1 = -ax2skew(r_P1P2) @ ez1_q - ax2skew(ez1) @ r_OP1_q
        n_q2 = ax2skew(ez1) @ r_OP2_q
        dense[:nu1, :nq1] += (
            np.einsum("i,ijk->jk", -la_g[1] * ez1, J_P1_q)
            + np.einsum("ij,ik->kj", -la_g[1] * ez1_q, J_P1)
            + np.einsum("i,ijk->jk", la_g[1] * n, J_R1_q)
            + np.einsum("ij,ik->kj", la_g[1] * n_q1, J_R1)
        )
        dense[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[1] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum("ij,ik->kj", la_g[1] * ez1_q, J_P2)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", la_g[1] * ez1, J_P2_q)
        # angular velocity

        # W_g[:nu1, 2] la_g[2]= cross3(ex1, ey2) @ J_R1 * la_g[2]
        # W_g[nu1:, 2] la_g[2]= - cross3(ex1, ey2) @ J_R2 * la_g[2]
        n = cross3(ex1, ey2)
        n_q1 = -ax2skew(ey2) @ ex1_q
        n_q2 = ax2skew(ex1) @ ey2_q
        dense[:nu1, :nq1] += np.einsum("i,ijk->jk", la_g[2] * n, J_R1_q) + np.einsum(
            "ij,ik->kj", la_g[2] * n_q1, J_R1
        )
        dense[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[2] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum("ij,ik->kj", -la_g[2] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", -la_g[2] * n, J_R2_q) + np.einsum(
            "ij,ik->kj", -la_g[2] * n_q2, J_R2
        )

        # W_g[:nu1, 3] la_g[3]= cross3(ey1, ez2) @ J_R1 * la_g[3]
        # W_g[nu1:, 3] la_g[3]= - cross3(ey1, ez2) @ J_R2 * la_g[3]
        n = cross3(ey1, ez2)
        n_q1 = -ax2skew(ez2) @ ey1_q
        n_q2 = ax2skew(ey1) @ ez2_q
        dense[:nu1, :nq1] += np.einsum("i,ijk->jk", la_g[3] * n, J_R1_q) + np.einsum(
            "ij,ik->kj", la_g[3] * n_q1, J_R1
        )
        dense[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[3] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum("ij,ik->kj", -la_g[3] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", -la_g[3] * n, J_R2_q) + np.einsum(
            "ij,ik->kj", -la_g[3] * n_q2, J_R2
        )

        # W_g[:nu1, 4] la_g[4]= cross3(ez1, ex2) @ J_R1 * la_g[4]
        # W_g[nu1:, 4] la_g[4]= - cross3(ez1, ex2) @ J_R2 * la_g[4]
        n = cross3(ez1, ex2)
        n_q1 = -ax2skew(ex2) @ ez1_q
        n_q2 = ax2skew(ez1) @ ex2_q
        dense[:nu1, :nq1] += np.einsum("i,ijk->jk", la_g[4] * n, J_R1_q) + np.einsum(
            "ij,ik->kj", la_g[4] * n_q1, J_R1
        )
        dense[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[4] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum("ij,ik->kj", -la_g[4] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", -la_g[4] * n, J_R2_q) + np.einsum(
            "ij,ik->kj", -la_g[4] * n_q2, J_R2
        )

        coo.extend(dense, (self.uDOF, self.qDOF))
