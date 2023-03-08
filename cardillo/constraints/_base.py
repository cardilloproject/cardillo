import numpy as np


class PositionOrientationBase:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        r_OB0,
        A_IB0,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
    ):
        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2
        self.r_OB0 = r_OB0
        self.A_IB0 = A_IB0

    def assembler_callback(self):
        qDOF1 = self.subsystem1.qDOF
        qDOF2 = self.subsystem2.qDOF
        local_qDOF1 = self.subsystem1.local_qDOF_P(self.frame_ID1)
        local_qDOF2 = self.subsystem2.local_qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate((qDOF1[local_qDOF1], qDOF2[local_qDOF2]))
        self._nq1 = nq1 = len(local_qDOF1)
        self._nq2 = len(local_qDOF2)
        self._nq = self._nq1 + self._nq2

        uDOF1 = self.subsystem1.uDOF
        uDOF2 = self.subsystem2.uDOF
        local_uDOF1 = self.subsystem1.local_uDOF_P(self.frame_ID1)
        local_uDOF2 = self.subsystem2.local_uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate((uDOF1[local_uDOF1], uDOF2[local_uDOF2]))
        self._nu1 = nu1 = len(local_uDOF1)
        self._nu2 = len(local_uDOF2)
        self._nu = self._nu1 + self._nu2

        A_IK10 = self.subsystem1.A_IK(
            self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.frame_ID1
        )
        A_IK20 = self.subsystem2.A_IK(
            self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.frame_ID2
        )

        A_K1B0 = A_IK10.T @ self.A_IB0
        A_K2B0 = A_IK20.T @ self.A_IB0

        r_OP10 = self.subsystem1.r_OP(
            self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.frame_ID1
        )
        r_OP20 = self.subsystem2.r_OP(
            self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.frame_ID2
        )

        K1_r_P1B0 = A_IK10.T @ (self.r_OB0 - r_OP10)
        K2_r_P2B0 = A_IK20.T @ (self.r_OB0 - r_OP20)

        # auxiliary functions for subsystem 1
        self.r_OB1 = lambda t, q: self.subsystem1.r_OP(
            t, q[:nq1], self.frame_ID1, K1_r_P1B0
        )
        self.r_OB1_q1 = lambda t, q: self.subsystem1.r_OP_q(
            t, q[:nq1], self.frame_ID1, K1_r_P1B0
        )
        self.v_B1 = lambda t, q, u: self.subsystem1.v_P(
            t, q[:nq1], u[:nu1], self.frame_ID1, K1_r_P1B0
        )
        self.v_B1_q1 = lambda t, q, u: self.subsystem1.v_P_q(
            t, q[:nq1], u[:nu1], self.frame_ID1, K1_r_P1B0
        )
        self.a_B1 = lambda t, q, u, u_dot: self.subsystem1.a_P(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1, K1_r_P1B0
        )
        self.a_B1_q1 = lambda t, q, u, u_dot: self.subsystem1.a_P_q(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1, K1_r_P1B0
        )
        self.a_B1_u1 = lambda t, q, u, u_dot: self.subsystem1.a_P_u(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1, K1_r_P1B0
        )
        self.J_B1 = lambda t, q: self.subsystem1.J_P(
            t, q[:nq1], self.frame_ID1, K1_r_P1B0
        )
        self.J_B1_q1 = lambda t, q: self.subsystem1.J_P_q(
            t, q[:nq1], self.frame_ID1, K1_r_P1B0
        )
        self.A_IB1 = (
            lambda t, q: self.subsystem1.A_IK(t, q[:nq1], self.frame_ID1) @ A_K1B0
        )
        self.A_IB1_q1 = lambda t, q: np.einsum(
            "ijl,jk->ikl", self.subsystem1.A_IK_q(t, q[:nq1], self.frame_ID1), A_K1B0
        )
        self.Omega1 = lambda t, q, u: self.subsystem1.A_IK(
            t, q[:nq1], self.frame_ID1
        ) @ self.subsystem1.K_Omega(t, q[:nq1], u[:nu1], self.frame_ID1)
        self.Omega1_q1 = lambda t, q, u: np.einsum(
            "ijk,j->ik",
            self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_Omega(t, q[:nq1], u[:nu1], self.frame_ID1),
        ) + self.subsystem1.A_IK(
            t, q[:nq1], frame_ID=self.frame_ID1
        ) @ self.subsystem1.K_Omega_q(
            t, q[:nq1], u[:nu1], self.frame_ID1
        )

        self.Psi1 = lambda t, q, u, u_dot: self.subsystem1.A_IK(
            t, q[:nq1], self.frame_ID1
        ) @ self.subsystem1.K_Psi(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1)
        self.Psi1_q1 = lambda t, q, u, u_dot: np.einsum(
            "ijk,j->ik",
            self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_Psi(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1),
        ) + self.subsystem1.A_IK(
            t, q[:nq1], frame_ID=self.frame_ID1
        ) @ self.subsystem1.K_Psi_q(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1
        )
        self.Psi1_u1 = lambda t, q, u, u_dot: self.subsystem1.A_IK(
            t, q[:nq1], frame_ID=self.frame_ID1
        ) @ self.subsystem1.K_Psi_u(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1)

        self.J_R1 = lambda t, q: self.subsystem1.A_IK(
            t, q[:nq1], self.frame_ID1
        ) @ self.subsystem1.K_J_R(t, q[:nq1], self.frame_ID1)
        self.J_R1_q1 = lambda t, q: np.einsum(
            "ijk,jl->ilk",
            self.subsystem1.A_IK_q(t, q[:nq1], self.frame_ID1),
            self.subsystem1.K_J_R(t, q[:nq1], self.frame_ID1),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem1.A_IK(t, q[:nq1], self.frame_ID1),
            self.subsystem1.K_J_R_q(t, q[:nq1], self.frame_ID1),
        )

        # auxiliary functions for subsystem 2
        self.r_OB2 = lambda t, q: self.subsystem2.r_OP(
            t, q[nq1:], self.frame_ID2, K2_r_P2B0
        )
        self.r_OB2_q2 = lambda t, q: self.subsystem2.r_OP_q(
            t, q[nq1:], self.frame_ID2, K2_r_P2B0
        )
        self.v_B2 = lambda t, q, u: self.subsystem2.v_P(
            t, q[nq1:], u[nu1:], self.frame_ID2, K2_r_P2B0
        )
        self.v_B2_q2 = lambda t, q, u: self.subsystem2.v_P_q(
            t, q[nq1:], u[nu1:], self.frame_ID2, K2_r_P2B0
        )
        self.a_B2 = lambda t, q, u, u_dot: self.subsystem2.a_P(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2, K2_r_P2B0
        )
        self.a_B2_q2 = lambda t, q, u, u_dot: self.subsystem2.a_P_q(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2, K2_r_P2B0
        )
        self.a_B2_u2 = lambda t, q, u, u_dot: self.subsystem2.a_P_u(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2, K2_r_P2B0
        )
        self.J_B2 = lambda t, q: self.subsystem2.J_P(
            t, q[nq1:], self.frame_ID2, K2_r_P2B0
        )
        self.J_B2_q2 = lambda t, q: self.subsystem2.J_P_q(
            t, q[nq1:], self.frame_ID2, K2_r_P2B0
        )
        self.A_IB2 = (
            lambda t, q: self.subsystem2.A_IK(t, q[nq1:], self.frame_ID2) @ A_K2B0
        )
        self.A_IB2_q2 = lambda t, q: np.einsum(
            "ijk,jl->ilk", self.subsystem2.A_IK_q(t, q[nq1:], self.frame_ID2), A_K2B0
        )
        self.Omega2 = lambda t, q, u: self.subsystem2.A_IK(
            t, q[nq1:], self.frame_ID2
        ) @ self.subsystem2.K_Omega(t, q[nq1:], u[nu1:], self.frame_ID2)
        self.Omega2_q2 = lambda t, q, u: np.einsum(
            "ijk,j->ik",
            self.subsystem2.A_IK_q(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_Omega(t, q[nq1:], u[nu1:], self.frame_ID2),
        ) + self.subsystem2.A_IK(
            t, q[nq1:], frame_ID=self.frame_ID2
        ) @ self.subsystem2.K_Omega_q(
            t, q[nq1:], u[nu1:], self.frame_ID2
        )

        self.Psi2 = lambda t, q, u, u_dot: self.subsystem2.A_IK(
            t, q[nq1:], self.frame_ID2
        ) @ self.subsystem2.K_Psi(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2)
        self.Psi2_q2 = lambda t, q, u, u_dot: np.einsum(
            "ijk,j->ik",
            self.subsystem2.A_IK_q(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_Psi(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2),
        ) + self.subsystem2.A_IK(
            t, q[nq1:], frame_ID=self.frame_ID2
        ) @ self.subsystem2.K_Psi_q(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2
        )
        self.Psi2_u2 = lambda t, q, u, u_dot: self.subsystem2.A_IK(
            t, q[nq1:], frame_ID=self.frame_ID2
        ) @ self.subsystem2.K_Psi_u(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2)

        self.J_R2 = lambda t, q: self.subsystem2.A_IK(
            t, q[nq1:], self.frame_ID2
        ) @ self.subsystem2.K_J_R(t, q[nq1:], self.frame_ID2)
        self.J_R2_q2 = lambda t, q: np.einsum(
            "ijk,jl->ilk",
            self.subsystem2.A_IK_q(t, q[nq1:], self.frame_ID2),
            self.subsystem2.K_J_R(t, q[nq1:], self.frame_ID2),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem2.A_IK(t, q[nq1:], self.frame_ID2),
            self.subsystem2.K_J_R_q(t, q[nq1:], self.frame_ID2),
        )
