import numpy as np
from cardillo.math import cross3, ax2skew, inv3D, A_IK_basic, approx_fprime


class RigidBodyEuler:
    def __init__(self, m, K_theta_S, axis="zxy", q0=None, u0=None):
        self.m = m
        self.theta = K_theta_S

        self.nq = 6
        self.nu = 6
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

        self.M_ = np.zeros((self.nu, self.nu))
        self.M_[:3, :3] = m * np.eye(3)
        self.M_[3:, 3:] = self.theta

        ex, ey, ez = np.eye(3)
        axis = axis.lower()
        self.e1 = eval(f"e{axis[0]}")
        self.e2 = eval(f"e{axis[1]}")
        self.e3 = eval(f"e{axis[2]}")

        self.A_I1 = eval(f"lambda q: A_IK_basic(q[3]).{axis[0]}()")
        self.A_12 = eval(f"lambda q: A_IK_basic(q[4]).{axis[1]}()")
        self.A_2K = eval(f"lambda q: A_IK_basic(q[5]).{axis[2]}()")

        self.dA_I1 = eval(f"lambda q: A_IK_basic(q[3]).d{axis[0]}()")
        self.dA_12 = eval(f"lambda q: A_IK_basic(q[4]).d{axis[1]}()")
        self.dA_2K = eval(f"lambda q: A_IK_basic(q[5]).d{axis[2]}()")

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    def M(self, t, q, coo):
        coo.extend(self.M_, (self.uDOF, self.uDOF))

    def f_gyr(self, t, q, u):
        omega = u[3:]
        f = np.zeros(self.nu)
        f[3:] = cross3(omega, self.theta @ omega)
        return f

    def f_gyr_u(self, t, q, u, coo):
        omega = u[3:]
        dense = np.zeros((self.nu, self.nu))
        dense[3:, 3:] = ax2skew(omega) @ self.theta - ax2skew(self.theta @ omega)
        coo.extend(dense, (self.uDOF, self.uDOF))

    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.nq)
        q_dot[:3] = u[:3]
        q_dot[3:] = self.Q(q) @ u[3:]

        return q_dot

    def Q(self, q):
        A_K2 = self.A_2K(q).T
        A_K1 = A_K2 @ self.A_12(q).T
        H = np.zeros((3, 3))
        H[:, 0] = A_K1 @ self.e1
        H[:, 1] = A_K2 @ self.e2
        H[:, 2] = self.e3
        return inv3D(H)

    def Q_q(self, q):
        A_K2 = self.A_2K(q).T
        A_K1 = A_K2 @ self.A_12(q).T

        # TODO:
        A_K1_q = 0
        A_K2_q = 0

        H = np.zeros((3, 3), dtype=float)
        H[:, 0] = A_K1 @ self.e1
        H[:, 1] = A_K2 @ self.e2
        H[:, 2] = self.e3

        H_q = np.zeros((3, 3, self.nq), dtype=float)
        H_q[:, 0] = np.einsum("ikj,k", A_K1_q, self.e1)
        H_q[:, 1] = np.einsum("ikj,k", A_K2_q, self.e1)

        Hinv = inv3D(H)

        return np.einsum("il,lmk,mj->ijk", -Hinv, H_q, Hinv)

    def q_ddot(self, t, q, u, u_dot):
        q_ddot = np.zeros(self.nq)
        q_ddot[:3] = u_dot[:3]
        q_ddot[3:] = self.Q(q) @ u_dot[3:]

        q_dot_q = approx_fprime(q, lambda q: self.q_dot(t, q, u))
        q_ddot += q_dot_q @ self.q_dot(t, q, u)
        return q_ddot

    def q_dot_q(self, t, q, u, coo):
        dense = approx_fprime(q, lambda q: self.q_dot(t, q, u))
        coo.extend(dense, (self.qDOF, self.qDOF))

    def B(self, t, q, coo):
        B = np.zeros((self.nq, self.nu))
        B[:3, :3] = np.eye(3)
        B[3:, 3:] = self.Q(q)
        coo.extend(B, (self.qDOF, self.uDOF))

    def qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)

    def A_IK(self, t, q, frame_ID=None):
        return self.A_I1(q) @ self.A_12(q) @ self.A_2K(q)

    def A_IK_q(self, t, q, frame_ID=None):
        A_IK_q = np.zeros((3, 3, self.nq))
        A_IK_q[:, :, 3] = self.dA_I1(q) @ self.A_12(q) @ self.A_2K(q)
        A_IK_q[:, :, 4] = self.A_I1(q) @ self.dA_12(q) @ self.A_2K(q)
        A_IK_q[:, :, 5] = self.A_I1(q) @ self.A_12(q) @ self.dA_2K(q)
        return A_IK_q

    def r_OP(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return q[:3] + self.A_IK(t, q) @ K_r_SP

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        r_OP_q = np.zeros((3, self.nq))
        r_OP_q[:, :3] = np.eye(3)
        r_OP_q[:, :] += np.einsum("ijk,j->ik", self.A_IK_q(t, q), K_r_SP)
        return r_OP_q

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return u[:3] + self.A_IK(t, q) @ cross3(u[3:], K_r_SP)

    def v_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.einsum("ijk,j->ik", self.A_IK_q(t, q), cross3(u[3:], K_r_SP))

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return u_dot[:3] + self.A_IK(t, q) @ (
            cross3(u_dot[3:], K_r_SP) + cross3(u[3:], cross3(u[3:], K_r_SP))
        )

    def a_P_q(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.einsum(
            "ijk,j->ik",
            self.A_IK_q(t, q),
            cross3(u_dot[3:], K_r_SP) + cross3(u[3:], cross3(u[3:], K_r_SP)),
        )

    def a_P_u(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.kappa_P_u(t, q, u, frame_ID=frame_ID, K_r_SP=K_r_SP)

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

    def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        J_P = np.zeros((3, self.nu))
        J_P[:, :3] = np.eye(3)
        J_P[:, 3:] = -self.A_IK(t, q) @ ax2skew(K_r_SP)
        return J_P

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        J_P_q = np.zeros((3, self.nu, self.nq))
        J_P_q[:, 3:, :] = np.einsum("ijk,jl->ilk", self.A_IK_q(t, q), -ax2skew(K_r_SP))
        return J_P_q

    def K_Omega(self, t, q, u, frame_ID=None):
        return u[3:]

    def K_Omega_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq))

    def K_Psi(self, t, q, u, u_dot, frame_ID=None):
        return u_dot[3:]

    def K_kappa_R(self, t, q, u, frame_ID=None):
        return np.zeros(3)

    def K_kappa_R_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq))

    def K_kappa_R_u(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nu))

    def K_J_R(self, t, q, frame_ID=None):
        J_R = np.zeros((3, self.nu))
        J_R[:, 3:] = np.eye(3)
        return J_R

    def K_J_R_q(self, t, q, frame_ID=None):
        return np.zeros((3, self.nu, self.nq))
