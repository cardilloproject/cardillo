from typing import Optional
import numpy.typing as npt
import numpy as np
from cardillo.math import (
    norm,
    cross3,
    ax2skew,
    quat2mat,
    quat2mat_p,
    quat2rot,
    quat2rot_p,
)


class RigidBodyQuaternion:
    """Rigid body parametrized by center of mass in inertial system and unit 
    quaternions for rotation.

    References
    ----------
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Schweizer2015: https://www.research-collection.ethz.ch/handle/20.500.11850/101867
    """

    def __init__(
        self,
        m: float,
        K_theta_S: npt.ArrayLike,
        q0: npt.ArrayLike,
        u0: Optional[npt.ArrayLike] = None,
    ):
        self.m = m
        self.K_Theta_S = np.asarray(K_theta_S)

        self.nq = 7
        self.nu = 6
        self.nla_S = 1
        self.q0 = np.array([0, 0, 0, 1, 0, 0, 0]) if q0 is None else np.asarray(q0)
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else np.asarray(u0)
        self.la_S0 = np.zeros(self.nla_S, dtype=float)

        self.M_ = np.zeros((self.nu, self.nu))
        self.M_[:3, :3] = m * np.eye(3, dtype=float)
        self.M_[3:, 3:] = self.K_Theta_S

    def g_S(self, t, q):
        P = q[3:]
        return np.array([P @ P - 1.0], dtype=float)

    def g_S_q(self, t, q, coo):
        P = q[3:]
        dense = np.zeros((1, 7), dtype=float)
        dense[0, 3:] = 2.0 * P
        coo.extend(dense, (self.la_SDOF, self.qDOF))

    def M(self, t, q, coo):
        coo.extend(self.M_, (self.uDOF, self.uDOF))

    def f_gyr(self, t, q, u):
        omega = u[3:]
        f = np.zeros(self.nu)
        f[3:] = cross3(omega, self.K_Theta_S @ omega)
        return f

    def f_gyr_u(self, t, q, u, coo):
        omega = u[3:]
        dense = np.zeros((self.nu, self.nu))
        dense[3:, 3:] = ax2skew(omega) @ self.K_Theta_S - ax2skew(
            self.K_Theta_S @ omega
        )
        coo.extend(dense, (self.uDOF, self.uDOF))

    def q_dot(self, t, q, u):
        p = q[3:]
        Q = quat2mat(p) / (2 * p @ p)

        q_dot = np.zeros(self.nq)
        q_dot[:3] = u[:3]
        q_dot[3:] = Q[:, 1:] @ u[3:]

        return q_dot

    def q_dot_q(self, t, q, u, coo):
        p = q[3:]
        p2 = p @ p
        Q_p = quat2mat_p(p) / (2 * p2) - np.einsum(
            "ij,k->ijk", quat2mat(p), p / (p2**2)
        )

        dense = np.zeros((self.nq, self.nq))
        dense[3:, 3:] = np.einsum("ijk,j->ik", Q_p[:, 1:, :], u[3:])
        coo.extend(dense, (self.qDOF, self.qDOF))

    def B(self, t, q, coo):
        p = q[3:]
        Q = quat2mat(p) / (2 * p @ p)

        B = np.zeros((self.nq, self.nu), dtype=float)
        B[:3, :3] = np.eye(3, dtype=float)
        B[3:, 3:] = Q[:, 1:]
        coo.extend(B, (self.qDOF, self.uDOF))

    def q_ddot(self, t, q, u, u_dot):
        p = q[3:]
        p2 = p @ p
        Q = quat2mat(p) / (2 * p2)
        p_dot = Q[:, 1:] @ u[3:]
        Q_p = quat2mat_p(p) / (2 * p2) - np.einsum(
            "ij,k->ijk", quat2mat(p), p / (p2**2)
        )

        q_ddot = np.zeros(self.nq, dtype=float)
        q_ddot[:3] = u_dot[:3]
        q_ddot[3:] = Q[:, 1:] @ u_dot[3:] + np.einsum(
            "ijk,k,j->i", Q_p[:, 1:, :], p_dot, u[3:]
        )

        return q_ddot

    def step_callback(self, t, q, u):
        q[3:] = q[3:] / norm(q[3:])
        return q, u

    def qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)

    def A_IK(self, t, q, frame_ID=None):
        return quat2rot(q[3:])

    def A_IK_q(self, t, q, frame_ID=None):
        A_IK_q = np.zeros((3, 3, self.nq), dtype=float)
        A_IK_q[:, :, 3:] = quat2rot_p(q[3:])
        return A_IK_q

    def r_OP(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return q[:3] + self.A_IK(t, q) @ K_r_SP

    def r_OP_t(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return np.zeros(3)

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        r_OP_q = np.zeros((3, self.nq))
        r_OP_q[:, :3] = np.eye(3)
        r_OP_q[:, :] += np.einsum("ijk,j->ik", self.A_IK_q(t, q), K_r_SP)
        return r_OP_q

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return u[:3] + self.A_IK(t, q) @ cross3(u[3:], K_r_SP)

    def v_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return np.einsum("ijk,j->ik", self.A_IK_q(t, q), cross3(u[3:], K_r_SP))

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return u_dot[:3] + self.A_IK(t, q) @ (
            cross3(u_dot[3:], K_r_SP) + cross3(u[3:], cross3(u[3:], K_r_SP))
        )

    def kappa_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return self.A_IK(t, q) @ (cross3(u[3:], cross3(u[3:], K_r_SP)))

    def kappa_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return np.einsum(
            "ijk,j->ik", self.A_IK_q(t, q), cross3(u[3:], cross3(u[3:], K_r_SP))
        )

    def kappa_P_u(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        kappa_P_u = np.zeros((3, self.nu))
        kappa_P_u[:, 3:] = -self.A_IK(t, q) @ (
            ax2skew(cross3(u[3:], K_r_SP)) + ax2skew(u[3:]) @ ax2skew(K_r_SP)
        )
        return kappa_P_u

    def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        J_P = np.zeros((3, self.nu))
        J_P[:, :3] = np.eye(3)
        J_P[:, 3:] = -self.A_IK(t, q) @ ax2skew(K_r_SP)
        return J_P

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        J_P_q = np.zeros((3, self.nu, self.nq))
        J_P_q[:, 3:, :] = np.einsum("ijk,jl->ilk", self.A_IK_q(t, q), -ax2skew(K_r_SP))
        return J_P_q

    def K_Omega(self, t, q, u, frame_ID=None):
        return u[3:]

    def K_Omega_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq), dtype=float)

    def K_Psi(self, t, q, u, u_dot, frame_ID=None):
        return u_dot[3:]

    def K_kappa_R(self, t, q, u, frame_ID=None):
        return np.zeros(3, dtype=float)

    def K_kappa_R_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq), dtype=float)

    def K_kappa_R_u(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nu), dtype=float)

    def K_J_R(self, t, q, frame_ID=None):
        J_R = np.zeros((3, self.nu), dtype=float)
        J_R[:, 3:] = np.eye(3)
        return J_R

    def K_J_R_q(self, t, q, frame_ID=None):
        return np.zeros((3, self.nu, self.nq), dtype=float)
