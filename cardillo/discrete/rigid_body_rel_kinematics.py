import numpy as np

from cardillo.math.algebra import (
    cross3,
    ax2skew,
)
from cardillo.math.numerical_derivative import Numerical_derivative


class RigidBodyRelKinematics:
    def __init__(
        self,
        m,
        K_theta_S,
        joint,
        predecessor,
        frame_IDp=np.zeros(3),
        r_OS0=np.zeros(3),
        A_IK0=np.eye(3),
    ):
        self.m = m
        self.theta = K_theta_S
        self.r_OS0 = r_OS0
        self.A_IK0 = A_IK0

        self.predecessor = predecessor
        self.frame_IDp = frame_IDp

        self.joint = joint

        self.is_assembled = False

    def assembler_callback(self):
        if not self.joint.is_assembled:
            raise RuntimeError("Joint is not assembled; maybe not added to the model.")

        if not self.predecessor.is_assembled:
            raise RuntimeError(
                "Predecessor is not assembled; maybe not added to the model."
            )

        qDOFp = self.predecessor.local_qDOF_P(self.frame_IDp)
        self.qDOF = np.concatenate([self.predecessor.qDOF[qDOFp], self.joint.qDOF])
        self.nqp = nqp = len(qDOFp)
        self.q0 = np.concatenate([self.predecessor.q0[qDOFp], self.joint.q0])
        self.__nq = nqp + self.joint.nq

        uDOFp = self.predecessor.local_uDOF_P(self.frame_IDp)
        self.uDOF = np.concatenate([self.predecessor.uDOF[uDOFp], self.joint.uDOF])
        self.nup = nup = len(uDOFp)
        self.u0 = np.concatenate([self.predecessor.u0[uDOFp], self.joint.u0])
        self.__nu = nup + self.joint.nu

        A_IKp = self.predecessor.A_IK(
            self.predecessor.t0, self.predecessor.q0[qDOFp], self.frame_IDp
        )
        A_KpB1 = A_IKp.T @ self.joint.A_IB1
        A_KB2 = (
            self.A_IK0.T @ self.joint.A_IB1 @ self.joint.A_B1B2(self.t0, self.q0[nqp:])
        )
        self.A_B2K = A_KB2.T
        r_OSp = self.predecessor.r_OP(
            self.predecessor.t0, self.predecessor.q0[qDOFp], self.frame_IDp
        )
        K_r_SpB1 = A_IKp.T @ (self.joint.r_OB1 - r_OSp)
        self.K_r_SB2 = self.A_IK0.T @ (
            self.joint.r_OB1
            + self.joint.A_IB1 @ self.joint.B1_r_B1B2(self.t0, self.q0[nqp:])
            - self.r_OS0
        )

        self.r_OB1 = lambda t, q: self.predecessor.r_OP(
            t, q[:nqp], self.frame_IDp, K_r_SpB1
        )
        self.r_OB1_q1 = lambda t, q: self.predecessor.r_OP_q(
            t, q[:nqp], self.frame_IDp, K_r_SpB1
        )
        self.v_B1 = lambda t, q, u: self.predecessor.v_P(
            t, q[:nqp], u[:nup], self.frame_IDp, K_r_SpB1
        )
        self.v_B1_qp = lambda t, q, u: self.predecessor.v_P_q(
            t, q[:nqp], u[:nup], self.frame_IDp, K_r_SpB1
        )
        self.B1_Omegap = lambda t, q, u: A_KpB1.T @ self.predecessor.K_Omega(
            t, q[:nqp], u[:nup], self.frame_IDp
        )
        self.B1_Psip = lambda t, q, u, u_dot: A_KpB1.T @ self.predecessor.K_Psi(
            t, q[:nqp], u[:nup], u_dot[:nup], self.frame_IDp
        )
        self.a_B1 = lambda t, q, u, u_dot: self.predecessor.a_P(
            t, q[:nqp], u[:nup], u_dot[:nup], self.frame_IDp, K_r_SpB1
        )
        self.kappa_B1 = lambda t, q, u: self.predecessor.kappa_P(
            t, q[:nqp], u[:nup], self.frame_IDp, K_r_SpB1
        )
        self.kappa_B1_qp = lambda t, q, u: self.predecessor.kappa_P_q(
            t, q[:nqp], u[:nup], self.frame_IDp, K_r_SpB1
        )
        self.kappa_B1_up = lambda t, q, u: self.predecessor.kappa_P_u(
            t, q[:nqp], u[:nup], self.frame_IDp, K_r_SpB1
        )
        self.B1_kappa_Rp = lambda t, q, u: A_KpB1.T @ self.predecessor.K_kappa_R(
            t, q[:nqp], u[:nup], self.frame_IDp
        )
        self.B1_kappa_Rp_qp = lambda t, q, u: A_KpB1.T @ self.predecessor.K_kappa_R_q(
            t, q[:nqp], u[:nup], self.frame_IDp
        )
        self.B1_kappa_Rp_up = lambda t, q, u: A_KpB1.T @ self.predecessor.K_kappa_R_u(
            t, q[:nqp], u[:nup], self.frame_IDp
        )
        self.J_B1 = lambda t, q: self.predecessor.J_P(
            t, q[:nqp], self.frame_IDp, K_r_SpB1
        )
        self.J_B1_qp = lambda t, q: self.predecessor.J_P_q(
            t, q[:nqp], self.frame_IDp, K_r_SpB1
        )
        self.A_IB1 = (
            lambda t, q: self.predecessor.A_IK(t, q[:nqp], self.frame_IDp) @ A_KpB1
        )
        self.A_IB1_qp = lambda t, q: np.einsum(
            "ijl,jk->ikl", self.predecessor.A_IK_q(t, q[:nqp], self.frame_IDp), A_KpB1
        )
        self.B1_J_Rp = lambda t, q: A_KpB1.T @ self.predecessor.K_J_R(
            t, q[:nqp], self.frame_IDp
        )
        self.B1_J_Rp_qp = lambda t, q: np.einsum(
            "ij,jkl->ikl",
            A_KpB1.T,
            self.predecessor.K_J_R_q(t, q[:nqp], self.frame_IDp),
        )
        self.B1_Omegap_qp = lambda t, q, u: A_KpB1.T @ self.predecessor.K_Omega_q(
            t, q[:nqp], u[:nup]
        )

        self.A_B1B2 = lambda t, q: self.joint.A_B1B2(t, q[nqp:])
        self.A_B1B2_q2 = lambda t, q: self.joint.A_B1B2_q(t, q[nqp:])

        self.B1_r_B1B2 = lambda t, q: self.joint.B1_r_B1B2(t, q[nqp:])
        self.B1_r_B1B2_q2 = lambda t, q: self.joint.B1_r_B1B2_q(t, q[nqp:])
        self.B1_v_B1B2 = lambda t, q, u: self.joint.B1_v_B1B2(t, q[nqp:], u[nup:])
        self.B1_v_B1B2_q2 = lambda t, q, u: self.joint.B1_v_B1B2_q(t, q[nqp:], u[nup:])
        self.B1_J_B1B2 = lambda t, q: self.joint.B1_J_B1B2(t, q[nqp:])
        self.B1_J_B1B2_q2 = lambda t, q: self.joint.B1_J_B1B2_q(t, q[nqp:])
        self.B1_Omega_B1B2 = lambda t, q, u: self.joint.B1_Omega_B1B2(
            t, q[nqp:], u[nup:]
        )
        self.B1_Omega_B1B2_q2 = lambda t, q, u: self.joint.B1_Omega_B1B2_q(
            t, q[nqp:], u[nup:]
        )
        self.B1_a_B1B2 = lambda t, q, u, u_dot: self.joint.B1_a_B1B2(
            t, q[nqp:], u[nup:], u_dot[nup:]
        )
        self.B1_kappa_B1B2 = lambda t, q, u: self.joint.B1_kappa_B1B2(
            t, q[nqp:], u[nup:]
        )
        self.B1_kappa_B1B2_q2 = lambda t, q, u: self.joint.B1_kappa_B1B2_q(
            t, q[nqp:], u[nup:]
        )
        self.B1_Psi_B1B2 = lambda t, q, u, u_dot: self.joint.B1_Psi_B1B2(
            t, q[nqp:], u[nup:], u_dot[nup:]
        )
        self.B1_kappa_R_B1B2 = lambda t, q, u: self.joint.B1_kappa_R_B1B2(
            t, q[nqp:], u[nup:]
        )
        self.B1_kappa_R_B1B2_q2 = lambda t, q, u: self.joint.B1_kappa_R_B1B2_q(
            t, q[nqp:], u[nup:]
        )
        self.B1_kappa_R_B1B2_u2 = lambda t, q, u: self.joint.B1_kappa_R_B1B2_u(
            t, q[nqp:], u[nup:]
        )
        self.B1_J_R_B1B2 = lambda t, q: self.joint.B1_J_R_B1B2(t, q[nqp:])
        self.B1_J_R_B1B2_q2 = lambda t, q: self.joint.B1_J_R_B1B2_q(t, q[nqp:])

        self.is_assembled = True

    def M(self, t, q, coo):
        J_S = self.J_P(t, q)
        J_R = self.K_J_R(t, q)
        M = self.m * J_S.T @ J_S + J_R.T @ self.theta @ J_R
        coo.extend(M, (self.uDOF, self.uDOF))

    def Mu_q(self, t, q, u, coo):
        J_S = self.J_P(t, q)
        J_R = self.K_J_R(t, q)
        J_S_q = self.J_P_q(t, q)
        J_R_q = self.K_J_R_q(t, q)

        Mu_q = (
            np.einsum("ijl,ik,k->jl", J_S_q, J_S, self.m * u)
            + np.einsum("ij,ikl,k->jl", J_S, J_S_q, self.m * u)
            + np.einsum("ijl,ik,k->jl", J_R_q, self.theta @ J_R, u)
            + np.einsum("ij,jkl,k->il", J_R.T @ self.theta, J_R_q, u)
        )

        coo.extend(Mu_q, (self.uDOF, self.qDOF))

    def h(self, t, q, u):
        Omega = self.K_Omega(t, q, u)
        return -(
            self.m * self.J_P(t, q).T @ self.kappa_P(t, q, u)
            + self.K_J_R(t, q).T
            @ (self.theta @ self.K_kappa_R(t, q, u) + cross3(Omega, self.theta @ Omega))
        )

    def h_q(self, t, q, u, coo):
        Omega = self.K_Omega(t, q, u)
        Omega_q = self.K_Omega_q(t, q, u)
        tmp1 = self.theta @ self.K_kappa_R(t, q, u)
        tmp1_q = self.theta @ self.K_kappa_R_q(t, q, u)
        tmp2 = cross3(Omega, self.theta @ Omega)
        tmp2_q = (ax2skew(Omega) @ self.theta - ax2skew(self.theta @ Omega)) @ Omega_q

        f_gyr_q = -(
            np.einsum("jik,j->ik", self.J_P_q(t, q), self.m * self.kappa_P(t, q, u))
            + self.m * self.J_P(t, q).T @ self.kappa_P_q(t, q, u)
            + np.einsum("jik,j->ik", self.J_P_q(t, q), tmp1 + tmp2)
            + self.K_J_R(t, q).T @ (tmp1_q + tmp2_q)
        )

        coo.extend(f_gyr_q, (self.uDOF, self.qDOF))

    def h_u(self, t, q, u, coo):
        Omega = self.K_Omega(t, q, u)
        Omega_u = self.K_J_R(t, q)
        tmp1_u = self.theta @ self.K_kappa_R_u(t, q, u)
        tmp2_u = (ax2skew(Omega) @ self.theta - ax2skew(self.theta @ Omega)) @ Omega_u

        h_u = -(
            self.m * self.J_P(t, q).T @ self.kappa_P_u(t, q, u)
            + self.K_J_R(t, q).T @ (tmp1_u + tmp2_u)
        )
        coo.extend(h_u, (self.uDOF, self.uDOF))

        # f_gyr_u_num = Numerical_derivative(self.f_gyr, order=2)._y(t, q, u)
        # print(f'f_gyr_u error = {np.linalg.norm(f_gyr_u - f_gyr_u_num)}')
        # coo.extend(f_gyr_u_num, (self.uDOF, self.uDOF))

    #########################################
    # helper functions
    #########################################

    def local_qDOF_P(self, frame_ID=None):
        return np.arange(self.__nq)

    def local_uDOF_P(self, frame_ID=None):
        return np.arange(self.__nu)

    def A_IK(self, t, q, frame_ID=None):
        return self.A_IB1(t, q) @ self.A_B1B2(t, q) @ self.A_B2K

    def A_IK_q(self, t, q, frame_ID=None):
        A_IK_q = np.zeros((3, 3, self.__nq))
        A_IK_q[:, :, : self.nqp] = np.einsum(
            "ijk,jl,lm->imk", self.A_IB1_qp(t, q), self.A_B1B2(t, q), self.A_B2K
        )
        A_IK_q[:, :, self.nqp :] = np.einsum(
            "ij,jkl,km->iml", self.A_IB1(t, q), self.A_B1B2_q2(t, q), self.A_B2K
        )
        return A_IK_q

    def r_OP(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return (
            self.r_OB1(t, q)
            + self.A_IB1(t, q) @ self.B1_r_B1B2(t, q)
            + self.A_IK(t, q) @ (K_r_SP - self.K_r_SB2)
        )

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        r_OP_q = np.einsum("ijk,j->ik", self.A_IK_q(t, q), (K_r_SP - self.K_r_SB2))
        r_OP_q[:, : self.nqp] += self.r_OB1_q1(t, q) + np.einsum(
            "ijk,j->ik", self.A_IB1_qp(t, q), self.B1_r_B1B2(t, q)
        )
        r_OP_q[:, self.nqp :] += self.A_IB1(t, q) @ self.B1_r_B1B2_q2(t, q)
        return r_OP_q

    def r_OP_qq(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        raise NotImplementedError

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        # v_B1 + A_IB1 B1_v_B1B2 + A_IK ( K_Omega x (K_r_SP - self.K_r_SB2) )
        v_B2 = self.v_B1(t, q, u) + self.A_IB1(t, q) @ self.B1_v_B1B2(t, q, u)
        v_B2P = self.A_IK(t, q) @ cross3(self.K_Omega(t, q, u), K_r_SP - self.K_r_SB2)
        return v_B2 + v_B2P

    def v_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return Numerical_derivative(
            lambda t, q, u: self.v_P(t, q, u, frame_ID=frame_ID, K_r_SP=K_r_SP)
        )._x(t, q, u)

    def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        # J_P_num = np.zeros((3, self.__nu))
        # nu_P = self.v_P(t, q, np.zeros(self.__nu), K_r_SP=K_r_SP)
        # I = np.eye(self.__nu)
        # for i in range(self.__nu):
        #     J_P_num[:, i] = self.v_P(t, q, I[i], K_r_SP=K_r_SP) - nu_P
        # return J_P_num

        K_r_B2P = K_r_SP - self.K_r_SB2
        J_P = -self.A_IK(t, q) @ ax2skew(K_r_B2P) @ self.K_J_R(t, q)
        J_P[:, : self.nup] += self.J_B1(t, q)
        J_P[:, self.nup :] += self.A_IB1(t, q) @ self.B1_J_B1B2(t, q)
        return J_P

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        # return Numerical_derivative(lambda t, q: self.J_P(t, q, K_r_SP=K_r_SP))._x(t, q)

        K_r_B2P = K_r_SP - self.K_r_SB2
        J_P_q = np.einsum(
            "ij,jkl->ikl", -self.A_IK(t, q) @ ax2skew(K_r_B2P), self.K_J_R_q(t, q)
        )
        J_P_q -= np.einsum(
            "ijk,jl->ilk", self.A_IK_q(t, q), ax2skew(K_r_B2P) @ self.K_J_R(t, q)
        )
        J_P_q[:, : self.nup, : self.nqp] += self.J_B1_qp(t, q)
        J_P_q[:, self.nup :, : self.nqp] += np.einsum(
            "ijk,jl->ilk", self.A_IB1_qp(t, q), self.B1_J_B1B2(t, q)
        )
        J_P_q[:, self.nup :, self.nqp :] += np.einsum(
            "ij,jlk->ilk", self.A_IB1(t, q), self.B1_J_B1B2_q2(t, q)
        )

        return J_P_q

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        K_r_B2P = K_r_SP - self.K_r_SB2
        a_B2 = self.a_B1(t, q, u, u_dot) + self.A_IB1(t, q) @ self.B1_a_B1B2(
            t, q, u, u_dot
        )
        a_B2P = self.A_IK(t, q) @ (
            cross3(self.K_Psi(t, q, u, u_dot), K_r_B2P)
            + cross3(self.K_Omega(t, q, u), cross3(self.K_Omega(t, q, u), K_r_B2P))
        )
        return a_B2 + a_B2P

    def a_P_q(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return Numerical_derivative(
            lambda t, q, u: self.a_P(t, q, u, u_dot, frame_ID=frame_ID, K_r_SP=K_r_SP)
        )._x(t, q, u)

    def a_P_u(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return Numerical_derivative(
            lambda t, q, u: self.a_P(t, q, u, u_dot, frame_ID=frame_ID, K_r_SP=K_r_SP)
        )._y(t, q, u)

    def kappa_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        # return self.a_P(t, q, u, np.zeros(self.__nu), K_r_SP=K_r_SP)

        K_r_B2P = K_r_SP - self.K_r_SB2
        kappa_P = self.kappa_B1(t, q, u) + self.A_IB1(t, q) @ self.B1_kappa_B1B2(
            t, q, u
        )
        kappa_P += self.A_IK(t, q) @ (
            cross3(self.K_kappa_R(t, q, u), K_r_B2P)
            + cross3(self.K_Omega(t, q, u), cross3(self.K_Omega(t, q, u), K_r_B2P))
        )

        return kappa_P

    def kappa_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        K_r_B2P = K_r_SP - self.K_r_SB2
        K_Omega = self.K_Omega(t, q, u)
        K_Omega_q = self.K_Omega_q(t, q, u)

        tmp1 = cross3(self.K_kappa_R(t, q, u), K_r_B2P)
        tmp1_q = -ax2skew(K_r_B2P) @ self.K_kappa_R_q(t, q, u)
        tmp2 = cross3(K_Omega, cross3(K_Omega, K_r_B2P))
        tmp2_q = (
            -(ax2skew(cross3(K_Omega, K_r_B2P)) + ax2skew(K_Omega) @ ax2skew(K_r_B2P))
            @ K_Omega_q
        )

        kappa_P_q = np.einsum("ijk,j->ik", self.A_IK_q(t, q), tmp1 + tmp2) + self.A_IK(
            t, q
        ) @ (tmp1_q + tmp2_q)
        kappa_P_q[:, : self.nqp] += self.kappa_B1_qp(t, q, u) + np.einsum(
            "ijk,j->ik", self.A_IB1_qp(t, q), self.B1_kappa_B1B2(t, q, u)
        )
        kappa_P_q[:, self.nqp :] += self.A_IB1(t, q) @ self.B1_kappa_B1B2_q2(t, q, u)
        return kappa_P_q

    def kappa_P_u(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        K_r_B2P = K_r_SP - self.K_r_SB2
        K_Omega = self.K_Omega(t, q, u)
        K_Omega_u = self.K_J_R(t, q)

        tmp1_u = -ax2skew(K_r_B2P) @ self.K_kappa_R_u(t, q, u)
        tmp2_u = (
            -(ax2skew(cross3(K_Omega, K_r_B2P)) + ax2skew(K_Omega) @ ax2skew(K_r_B2P))
            @ K_Omega_u
        )

        kappa_P_u = self.A_IK(t, q) @ (tmp1_u + tmp2_u)
        kappa_P_u[:, : self.nup] += self.kappa_B1_up(t, q, u)
        kappa_P_u[:, self.nup :] += self.A_IB1(t, q) @ self.B1_kappa_R_B1B2_u2(t, q, u)
        return kappa_P_u

    def K_Omega(self, t, q, u, frame_ID=None):
        return (
            (self.B1_Omegap(t, q, u) + self.B1_Omega_B1B2(t, q, u))
            @ self.A_B1B2(t, q)
            @ self.A_B2K
        )

    def K_Omega_q(self, t, q, u, frame_ID=None):
        # return Numerical_derivative(self.K_Omega)._x(t, q, u)
        A_KB1 = (self.A_B1B2(t, q) @ self.A_B2K).T
        A_KB1_q2 = np.einsum("ijk,jl->lik", self.A_B1B2_q2(t, q), self.A_B2K)
        B1_Omega_pB2 = self.B1_Omegap(t, q, u) + self.B1_Omega_B1B2(t, q, u)

        K_Omega_q = np.zeros((3, self.__nq))
        K_Omega_q[:, : self.nqp] = A_KB1 @ self.B1_Omegap_qp(t, q, u)
        K_Omega_q[:, self.nqp :] = np.einsum(
            "ijk,j->ik", A_KB1_q2, B1_Omega_pB2
        ) + A_KB1 @ self.B1_Omega_B1B2_q2(t, q, u)

        return K_Omega_q

    def K_J_R(self, t, q, frame_ID=None):
        # J_R = np.zeros((3, self.__nu))
        # nu_R = self.K_Omega(t, q, np.zeros(self.__nu))
        # I = np.eye(self.__nu)
        # for i in range(self.__nu):
        #     J_R[:, i] = self.K_Omega(t, q, I[i]) - nu_R
        # return J_R

        J_R = np.zeros((3, self.__nu))
        A_KB1 = (self.A_B1B2(t, q) @ self.A_B2K).T
        J_R[:, : self.nup] = A_KB1 @ self.B1_J_Rp(t, q)
        J_R[:, self.nup :] = A_KB1 @ self.B1_J_R_B1B2(t, q)

        return J_R

    def K_J_R_q(self, t, q, frame_ID=None):
        # return Numerical_derivative(lambda t, q: self.K_J_R(t, q))._x(t, q)

        K_J_R_q = np.zeros((3, self.__nu, self.__nq))

        A_KB1 = (self.A_B1B2(t, q) @ self.A_B2K).T
        A_KB1_q2 = np.einsum("ijk,jl->lik", self.A_B1B2_q2(t, q), self.A_B2K)

        K_J_R_q[:, : self.nup, : self.nqp] = np.einsum(
            "ij,jkl->ikl", A_KB1, self.B1_J_Rp_qp(t, q)
        )
        K_J_R_q[:, : self.nup, self.nqp :] = np.einsum(
            "ijk,jl->ilk", A_KB1_q2, self.B1_J_Rp(t, q)
        )
        K_J_R_q[:, self.nup :, self.nqp :] = np.einsum(
            "ijk,jl->ilk", A_KB1_q2, self.B1_J_R_B1B2(t, q)
        ) + np.einsum("ij,jkl->ikl", A_KB1, self.B1_J_R_B1B2_q2(t, q))
        return K_J_R_q

    def K_Psi(self, t, q, u, u_dot, frame_ID=None):
        return (
            (self.B1_Psip(t, q, u, u_dot) + self.B1_Psi_B1B2(t, q, u, u_dot))
            @ self.A_B1B2(t, q)
            @ self.A_B2K
        )

    def K_kappa_R(self, t, q, u, frame_ID=None):
        # return self.K_Psi(t, q, u, np.zeros(self.__nu))
        return (
            (self.B1_kappa_Rp(t, q, u) + self.B1_kappa_R_B1B2(t, q, u))
            @ self.A_B1B2(t, q)
            @ self.A_B2K
        )

    def K_kappa_R_q(self, t, q, u, frame_ID=None):
        K_kappa_R_q = np.zeros((3, self.__nq))
        A_KB1 = (self.A_B1B2(t, q) @ self.A_B2K).T
        A_KB1_q2 = np.einsum("ijk,jl->lik", self.A_B1B2_q2(t, q), self.A_B2K)

        K_kappa_R_q[:, : self.nqp] = A_KB1 @ self.B1_kappa_Rp_qp(t, q, u)
        K_kappa_R_q[:, self.nqp :] = A_KB1 @ self.B1_kappa_R_B1B2_q2(
            t, q, u
        ) + np.einsum(
            "ijk,j->ik",
            A_KB1_q2,
            self.B1_kappa_Rp(t, q, u) + self.B1_kappa_R_B1B2(t, q, u),
        )

        return K_kappa_R_q

    def K_kappa_R_u(self, t, q, u, frame_ID=None):
        A_KB1 = (self.A_B1B2(t, q) @ self.A_B2K).T
        K_kappa_R_u = np.zeros((3, self.__nu))
        K_kappa_R_u[:, : self.nup] = A_KB1 @ self.B1_kappa_Rp_up(t, q, u)
        K_kappa_R_u[:, self.nup :] = A_KB1 @ self.B1_kappa_R_B1B2_u2(t, q, u)
        return K_kappa_R_u
