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

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    def M(self, t, q):
        return self.__M

    def h(self, t, q, u):
        omega = u[3:]
        f = np.zeros(self.nu, dtype=np.common_type(q, u))
        f[3:] = -cross3(omega, self.K_theta_S @ omega)
        return f

    def h_u(self, t, q, u):
        omega = u[3:]
        dense = np.zeros((self.nu, self.nu), dtype=np.common_type(q, u))
        dense[3:, 3:] = (
            ax2skew(self.K_theta_S @ omega) - ax2skew(omega) @ self.K_theta_S
        )
        return dense

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
        a_P_u = np.zeros((3, self.nu), dtype=float)
        a_P_u[:, 3:] = -self.A_IK(t, q) @ (
            ax2skew(cross3(u[3:], K_r_SP)) + ax2skew(u[3:]) @ ax2skew(K_r_SP)
        )
        return a_P_u

    def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        J_P = np.zeros((3, self.nu), dtype=q.dtype)
        J_P[:, :3] = np.eye(3)
        J_P[:, 3:] = -self.A_IK(t, q) @ ax2skew(K_r_SP)
        return J_P

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        J_P_q = np.zeros((3, self.nu, self.nq), dtype=q.dtype)
        J_P_q[:, 3:, :] = np.einsum("ijk,jl->ilk", self.A_IK_q(t, q), -ax2skew(K_r_SP))
        return J_P_q

    def kappa_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.A_IK(t, q) @ (cross3(u[3:], cross3(u[3:], K_r_SP)))

    def kappa_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.einsum(
            "ijk,j->ik", self.A_IK_q(t, q), cross3(u[3:], cross3(u[3:], K_r_SP))
        )

    def kappa_P_u(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        kappa_P_u = np.zeros((3, self.nu))
        kappa_P_u[:, 3:] = -self.A_IK(t, q) @ (
            ax2skew(cross3(u[3:], K_r_SP)) + ax2skew(u[3:]) @ ax2skew(K_r_SP)
        )
        return kappa_P_u

    def K_Omega(self, t, q, u, frame_ID=None):
        return u[3:]

    def K_Omega_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq), dtype=np.common_type(q, u))

    def K_Psi(self, t, q, u, u_dot, frame_ID=None):
        return u_dot[3:]

    def K_Psi_q(self, t, q, u, u_dot, frame_ID=None):
        return np.zeros((3, self.nq), dtype=np.common_type(q, u, u_dot))

    def K_Psi_u(self, t, q, u, u_dot, frame_ID=None):
        return np.zeros((3, self.nu), dtype=np.common_type(q, u, u_dot))

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

    def export(self, sol_i, **kwargs):
        points = [self.r_OP(sol_i.t, sol_i.q[self.qDOF])]
        vel = [self.v_P(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])]
        omega = [
            self.A_IK(sol_i.t, sol_i.q[self.qDOF])
            @ self.K_Omega(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])
        ]
        if sol_i.u_dot is not None:
            acc = [
                self.a_P(
                    sol_i.t,
                    sol_i.q[self.qDOF],
                    sol_i.u[self.uDOF],
                    sol_i.u_dot[self.uDOF],
                )
            ]
            psi = [
                self.A_IK(sol_i.t, sol_i.q[self.qDOF])
                @ self.K_Psi(
                    sol_i.t,
                    sol_i.q[self.qDOF],
                    sol_i.u[self.uDOF],
                    sol_i.u_dot[self.uDOF],
                )
            ]
        A_IK = np.vsplit(self.A_IK(sol_i.t, sol_i.q[self.qDOF]).T, 3)
        cells = [("vertex", [[0]])]
        if sol_i.u_dot is not None:
            cell_data = dict(
                v=[vel],
                Omega=[omega],
                a=[acc],
                psi=[psi],
                ex=[A_IK[0]],
                ey=[A_IK[1]],
                ez=[A_IK[2]],
            )
        else:
            cell_data = dict(
                v=[vel], Omega=[omega], ex=[A_IK[0]], ey=[A_IK[1]], ez=[A_IK[2]]
            )
        return points, cells, None, cell_data
