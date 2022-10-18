from typing import Optional
import numpy.typing as npt
import numpy as np
from cardillo.discrete.rigid_body_base import RigidBodyBase
from cardillo.math import (
    norm,
    cross3,
    ax2skew,
    quat2mat,
    quat2mat_p,
    quat2rot,
    quat2rot_p,
)


class RigidBodyQuaternion(RigidBodyBase):
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
        self.nq = 7
        self.nu = 6
        self.nla_S = 1

        q0 = np.array([0, 0, 0, 1, 0, 0, 0]) if q0 is None else np.asarray(q0)
        u0 = np.zeros(self.nu, dtype=float) if u0 is None else np.asarray(u0)
        self.la_S0 = np.zeros(self.nla_S, dtype=float)

        super().__init__(m, K_theta_S, q0, u0)

    def g_S(self, t, q):
        P = q[3:]
        return np.array([P @ P - 1.0], dtype=q.dtype)

    def g_S_q(self, t, q, coo):
        P = q[3:]
        dense = np.zeros((1, 7), dtype=q.dtype)
        dense[0, 3:] = 2.0 * P
        coo.extend(dense, (self.la_SDOF, self.qDOF))

    def q_dot(self, t, q, u):
        p = q[3:]
        Q = quat2mat(p) / (2 * p @ p)

        q_dot = np.zeros(self.nq, dtype=np.common_type(q, u))
        q_dot[:3] = u[:3]
        q_dot[3:] = Q[:, 1:] @ u[3:]

        return q_dot

    def q_dot_q(self, t, q, u, coo):
        p = q[3:]
        p2 = p @ p
        Q_p = quat2mat_p(p) / (2 * p2) - np.einsum(
            "ij,k->ijk", quat2mat(p), p / (p2**2)
        )

        dense = np.zeros((self.nq, self.nq), dtype=np.common_type(q, u))
        dense[3:, 3:] = np.einsum("ijk,j->ik", Q_p[:, 1:, :], u[3:])
        coo.extend(dense, (self.qDOF, self.qDOF))

    def B(self, t, q, coo):
        p = q[3:]
        Q = quat2mat(p) / (2 * p @ p)

        B = np.zeros((self.nq, self.nu), dtype=q.dtype)
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

        q_ddot = np.zeros(self.nq, dtype=np.common_type(q, u, u_dot))
        q_ddot[:3] = u_dot[:3]
        q_ddot[3:] = Q[:, 1:] @ u_dot[3:] + np.einsum(
            "ijk,k,j->i", Q_p[:, 1:, :], p_dot, u[3:]
        )

        return q_ddot

    def step_callback(self, t, q, u):
        q[3:] = q[3:] / norm(q[3:])
        return q, u

    def A_IK(self, t, q, frame_ID=None):
        return quat2rot(q[3:])

    def A_IK_q(self, t, q, frame_ID=None):
        A_IK_q = np.zeros((3, 3, self.nq), dtype=q.dtype)
        A_IK_q[:, :, 3:] = quat2rot_p(q[3:])
        return A_IK_q
