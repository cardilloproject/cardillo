from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np

from cardillo.math import cross3, ax2skew


class RigidBodyBase(ABC):
    def __init__(
        self,
        m: float,
        K_theta_S: npt.NDArray[np.float_],
        q0: npt.NDArray[np.float_],
        u0: npt.NDArray[np.float_],
    ) -> None:
        super().__init__()
        self.m = m
        self.K_theta_S = K_theta_S

        self.__M = np.zeros((self.nu, self.nu), dtype=float)
        self.__M[:3, :3] = self.m * np.eye(3, dtype=float)
        self.__M[3:, 3:] = self.K_theta_S

        assert q0.size == self.nq
        assert u0.size == self.nu

        self.u0 = u0
        self.q0 = q0

    def M(self, t, q, coo):
        coo.extend(self.__M, (self.uDOF, self.uDOF))

    def h(self, t, q, u):
        omega = u[3:]
        f = np.zeros(self.nu, dtype=np.common_type(q, u))
        f[3:] = -cross3(omega, self.K_theta_S @ omega)
        return f

    def h_u(self, t, q, u, coo):
        omega = u[3:]
        dense = np.zeros((self.nu, self.nu), dtype=np.common_type(q, u))
        dense[3:, 3:] = ax2skew(self.K_theta_S @ omega) - ax2skew(omega) @ self.K_theta_S
        coo.extend(dense, (self.uDOF, self.uDOF))

    def local_qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def local_uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)

    @abstractmethod
    def A_IK(self, t, q):
        ...

    @abstractmethod
    def A_IK_q(self, t, q):
        ...

    def r_OP(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return q[:3] + self.A_IK(t, q) @ K_r_SP

    def r_OP_t(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return np.zeros(3, dtype=q.dtype)

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        r_OP_q = np.zeros((3, self.nq), dtype=q.dtype)
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

    def a_P_q(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return np.einsum(
            "ijk,j->ik",
            self.A_IK_q(t, q),
            cross3(u_dot[3:], K_r_SP) + cross3(u[3:], cross3(u[3:], K_r_SP)),
        )

    def a_P_u(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return self.kappa_P_u(t, q, u, frame_ID=frame_ID, K_r_SP=K_r_SP)

    def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        J_P = np.zeros((3, self.nu), dtype=q.dtype)
        J_P[:, :3] = np.eye(3)
        J_P[:, 3:] = -self.A_IK(t, q) @ ax2skew(K_r_SP)
        return J_P

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        J_P_q = np.zeros((3, self.nu, self.nq), dtype=q.dtype)
        J_P_q[:, 3:, :] = np.einsum("ijk,jl->ilk", self.A_IK_q(t, q), -ax2skew(K_r_SP))
        return J_P_q

    def K_Omega(self, t, q, u, frame_ID=None):
        return u[3:]

    def K_Omega_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq), dtype=np.common_type(q, u))

    def K_Psi(self, t, q, u, u_dot, frame_ID=None):
        return u_dot[3:]

    def K_kappa_R(self, t, q, u, frame_ID=None):
        return np.zeros(3, dtype=np.common_type(q, u))

    def K_kappa_R_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq), dtype=np.common_type(q, u))

    def K_kappa_R_u(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nu), dtype=np.common_type(q, u))

    def K_J_R(self, t, q, frame_ID=None):
        J_R = np.zeros((3, self.nu), dtype=q.dtype)
        J_R[:, 3:] = np.eye(3)
        return J_R

    def K_J_R_q(self, t, q, frame_ID=None):
        return np.zeros((3, self.nu, self.nq), dtype=q.dtype)
