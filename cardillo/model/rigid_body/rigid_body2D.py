import numpy as np
from cardillo.math.algebra import cross3, ax2skew, A_IK_basic_z, dA_IK_basic_z


class Rigid_body2D:
    def __init__(self, m, theta, q0=None, u0=None):
        self.m = m
        self.theta = theta
        self.M_ = np.diag([self.m, self.m, self.theta])

        self.nq = 3
        self.nu = 3
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    def M(self, t, q, coo):
        coo.extend(self.M_, (self.uDOF, self.uDOF))

    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q, coo):
        coo.extend(np.eye(3), (self.qDOF, self.uDOF))

    def qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)

    def A_IK(self, t, q, frame_ID=None):
        return A_IK_basic_z(q[2])

    def A_IK_q(self, t, q, frame_ID=None):
        A_IK_q = np.zeros((3, 3, self.nq))
        A_IK_q[:, :, 2] = dA_IK_basic_z(q[2])
        return A_IK_q

    def __vec_xy(self, x):
        return np.array([x[0], x[1], 0])

    def __vec_z(self, x):
        return np.array([0, 0, x])

    def r_OP(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.__vec_xy(q[:2]) + self.A_IK(t, q) @ K_r_SP

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        r_OP_q = np.zeros((3, self.nq))
        r_OP_q[:2, :2] = np.eye(2)
        r_OP_q += np.einsum("ijk,j->ik", self.A_IK_q(t, q), K_r_SP)
        return r_OP_q

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.__vec_xy(u[:2]) + self.A_IK(t, q) @ cross3(
            self.__vec_z(u[2]), K_r_SP
        )

    def v_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.einsum(
            "ijk,j->ik", self.A_IK_q(t, q), cross3(self.__vec_z(u[2]), K_r_SP)
        )

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        omega = self.__vec_z(u[2])
        return self.__vec_xy(u_dot[:2]) + self.A_IK(t, q) @ (
            cross3(self.__vec_z(u_dot[2]), K_r_SP)
            + cross3(omega, cross3(omega, K_r_SP))
        )

    def a_P_q(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        omega = self.__vec_z(u[2])
        return np.einsum(
            "ijk,j->ik",
            self.A_IK_q(t, q),
            cross3(self.__vec_z(u_dot[2]), K_r_SP)
            + cross3(omega, cross3(omega, K_r_SP)),
        )

    def a_P_u(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.kappa_P_u(t, q, u, frame_ID=frame_ID, K_r_SP=K_r_SP)

    def kappa_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        omega = self.__vec_z(u[2])
        return self.A_IK(t, q) @ cross3(omega, cross3(omega, K_r_SP))

    def kappa_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        omega = self.__vec_z(u[2])
        return np.einsum(
            "ijk,j->ik", self.A_IK_q(t, q), cross3(omega, cross3(omega, K_r_SP))
        )

    def kappa_P_u(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        kappa_P_u = np.zeros((3, self.nu))
        omega = self.__vec_z(u[2])
        kappa_P_u[:, 2:] = (
            -self.A_IK(t, q)
            @ (ax2skew(cross3(omega, K_r_SP)) + ax2skew(omega) @ ax2skew(K_r_SP))[:, 2:]
        )
        return kappa_P_u

    def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        J_P = np.zeros((3, self.nu))
        J_P[:2, :2] = np.eye(2)
        J_P[:, 2:] = -(self.A_IK(t, q) @ ax2skew(K_r_SP))[:, 2:]
        return J_P

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        J_P_q = np.einsum("ijk,jl->ilk", self.A_IK_q(t, q), -ax2skew(K_r_SP))
        return J_P_q

    def K_Omega(self, t, q, u, frame_ID=None):
        return self.__vec_z(u[2])

    def K_Omega_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq))

    def K_Psi(self, t, q, u, u_dot, frame_ID=None):
        return self.__vec_z(u_dot[2])

    def K_kappa_R(self, t, q, u, frame_ID=None):
        return np.zeros(3)

    def K_kappa_R_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq))

    def K_kappa_R_u(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nu))

    def K_J_R(self, t, q, frame_ID=None):
        J_R = np.zeros((3, self.nu))
        J_R[2, 2] = 1
        return J_R

    def K_J_R_q(self, t, q, frame_ID=None):
        return np.zeros((3, self.nu, self.nq))
