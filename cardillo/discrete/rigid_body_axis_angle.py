from typing import Optional
import numpy as np
import numpy.typing as npt
from cardillo.discrete.rigid_body_base import RigidBodyBase
from cardillo.math import norm
from cardillo.math import Exp_SO3, Exp_SO3_psi, T_SO3_inv, T_SO3_inv_psi, pi


class RigidBodyAxisAngle(RigidBodyBase):
    def __init__(
        self,
        m: float,
        K_theta_S: npt.NDArray,
        q0: Optional[npt.NDArray] = None,
        u0: Optional[npt.NDArray] = None,
    ):
        self.nq = 6
        self.nu = 6

        q0 = np.zeros(self.nq) if q0 is None else q0
        u0 = np.zeros(self.nu) if u0 is None else u0

        super().__init__(m, K_theta_S, q0, u0)

    def step_callback(self, t, q, u):
        psi = q[3:]
        angle = norm(psi)
        # Ibrahimbegovic1995 after (62)
        if angle > pi:
            print(f"complement rotation vector chosen")
            n = int((angle + pi) / (2 * pi))
            e = psi / angle
            psi_C = psi - 2 * n * pi * e
            q[3:] = psi_C
        return q, u

    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.nq, dtype=np.common_type(q, u))
        q_dot[:3] = u[:3]
        q_dot[3:] = self.B_omega(q) @ u[3:]

        return q_dot

    def B_omega(self, q):
        T_inv = T_SO3_inv(q[3:])
        return T_inv

    def q_dot_q_dense(self, t, q, u):
        dense = np.zeros((self.nq, self.nq), dtype=float)
        dense[3:, 3:] = np.einsum("ijk,j->ik", T_SO3_inv_psi(q[3:]), u[3:])
        return dense

    def q_dot_q(self, t, q, u, coo):
        coo.extend(self.q_dot_q_dense(t, q, u), (self.qDOF, self.qDOF))

    def q_ddot(self, t, q, u, u_dot):
        q_ddot = np.zeros(self.nq, dtype=np.common_type(q, u, u_dot))
        q_ddot[:3] = u_dot[:3]
        q_ddot[3:] = self.B_omega(q) @ u_dot[3:]

        q_ddot += self.q_dot_q_dense(t, q, u) @ self.q_dot(t, q, u)
        return q_ddot

    def B(self, t, q, coo):
        B = np.zeros((self.nq, self.nu), dtype=q.dtype)
        B[:3, :3] = np.eye(3)
        B[3:, 3:] = self.B_omega(q)
        coo.extend(B, (self.qDOF, self.uDOF))

    def A_IK(self, t, q, frame_ID=None):
        return Exp_SO3(q[3:])

    def A_IK_q(self, t, q, frame_ID=None):
        dense = np.zeros((3, 3, self.nq), dtype=q.dtype)
        dense[:, :, 3:] = Exp_SO3_psi(q[3:])
        return dense
