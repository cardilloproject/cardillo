from typing import Optional
import numpy as np
import numpy.typing as npt
from cardillo.discrete.rigid_body_base import RigidBodyBase
from cardillo.math import norm, approx_fprime
from cardillo.math import Exp_SO3, T_SO3_inv, pi


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
            # print(f"complement rotation vector is used")
            n = int((angle + pi) / (2 * pi))
            # n = 1
            if angle > 0:
                e = psi / angle
            else:
                e = psi.copy()
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
        # T = tangent_map(q[3:])
        # T_inv_num = np.linalg.inv(T)
        # # error = np.linalg.norm(T_inv - T_inv_num)
        # # print(f'error T_inv: {error}')
        # return T_inv_num

    # TODO: analytical derivative
    def q_dot_q_dense(self, t, q, u):
        return approx_fprime(q, lambda q: self.q_dot(t, q, u))

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

    # TODO: analytical derivative
    def A_IK_q(self, t, q, frame_ID=None):
        return approx_fprime(q, lambda q: self.A_IK(t, q))
