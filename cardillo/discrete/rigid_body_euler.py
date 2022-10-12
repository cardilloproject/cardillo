from typing import Optional
from unicodedata import ucd_3_2_0
import numpy as np
import numpy.typing as npt
from cardillo.discrete.rigid_body_base import RigidBodyBase
from cardillo.math import cross3, ax2skew, inv3D, A_IK_basic, approx_fprime


class RigidBodyEuler(RigidBodyBase):
    def __init__(
        self,
        m: float,
        K_theta_S: npt.NDArray,
        axis: str = "zxy",
        q0: Optional[npt.NDArray] = None,
        u0: Optional[npt.NDArray] = None,
    ):
        self.nq = 6
        self.nu = 6
        q0 = np.zeros(self.nq) if q0 is None else q0
        u0 = np.zeros(self.nu) if u0 is None else u0

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

        super().__init__(m, K_theta_S, q0, u0)

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

    def A_IK(self, t, q, frame_ID=None):
        return self.A_I1(q) @ self.A_12(q) @ self.A_2K(q)

    def A_IK_q(self, t, q, frame_ID=None):
        A_IK_q = np.zeros((3, 3, self.nq))
        A_IK_q[:, :, 3] = self.dA_I1(q) @ self.A_12(q) @ self.A_2K(q)
        A_IK_q[:, :, 4] = self.A_I1(q) @ self.dA_12(q) @ self.A_2K(q)
        A_IK_q[:, :, 5] = self.A_I1(q) @ self.A_12(q) @ self.dA_2K(q)
        return A_IK_q
