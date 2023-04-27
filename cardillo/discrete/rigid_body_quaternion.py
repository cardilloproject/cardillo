from typing import Optional
import numpy.typing as npt
import numpy as np
from cardillo.discrete._base import RigidBodyBase
from cardillo.math import (
    norm,
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
)


class RigidBodyQuaternion(RigidBodyBase):
    """Rigid body parametrized by center of mass in inertial base and unit 
    quaternions for rotation. The angular velocities expressed in the 
    body-fixed base are used as minimal velocities.
    
    Exponential function and kinematic differential equation are found in 
    Egeland2002 (6.199), (6.329) and (6.330). The implementation below 
    handles non-unit quaternions. After each successfull time step they are 
    projected to be of unit length. Alternatively, the constraint can be added 
    to the kinematic differential equations using g_S.

    References
    ----------
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Schweizer2015: https://www.research-collection.ethz.ch/handle/20.500.11850/101867 \\
    Egenland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
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

        q0 = (
            np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
            if q0 is None
            else np.asarray(q0)
        )
        u0 = np.zeros(self.nu, dtype=float) if u0 is None else np.asarray(u0)
        self.la_S0 = np.zeros(self.nla_S, dtype=float)

        super().__init__(m, K_theta_S, q0, u0)

    def g_S(self, t, q):
        P = q[3:]
        return np.array([P @ P - 1.0], dtype=q.dtype)

    def g_S_q(self, t, q):
        P = q[3:]
        g_S_q = np.zeros((1, 7), dtype=q.dtype)
        g_S_q[0, 3:] = 2.0 * P
        return g_S_q

    def g_S_q_T_mu_q(self, t, q, mu):
        g_S_q_T_mu_q = np.zeros((7, 7), dtype=q.dtype)
        g_S_q_T_mu_q[3:, 3:] = 2.0 * mu[0] * np.eye(4, 4, dtype=float)
        return g_S_q_T_mu_q

    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.nq, dtype=np.common_type(q, u))
        q_dot[:3] = u[:3]
        q_dot[3:] = T_SO3_inv_quat(q[3:]) @ u[3:]
        return q_dot

    def q_dot_q(self, t, q, u):
        q_dot_q = np.zeros((self.nq, self.nq), dtype=np.common_type(q, u))
        q_dot_q[3:, 3:] = np.einsum("ijk,j->ik", T_SO3_inv_quat_P(q[3:]), u[3:])
        return q_dot_q

    def B(self, t, q):
        B = np.zeros((self.nq, self.nu), dtype=q.dtype)
        B[:3, :3] = np.eye(3, dtype=q.dtype)
        B[3:, 3:] = T_SO3_inv_quat(q[3:])
        return B

    def q_ddot(self, t, q, u, u_dot):
        # raise RuntimeWarning("RigidBodyQuaternion.q_ddot is not tested yet!")
        p = q[3:]
        p2 = p @ p
        B = T_SO3_inv_quat(p) / (p @ p)
        p_dot = B @ u[3:]
        p_ddot = (
            B @ u_dot[3:]
            + np.einsum("ijk,k,j->i", T_SO3_inv_quat_P(q[3:]), p_dot, u[3:])
            + 2 * p_dot * (p @ p_dot) / p2
        )

        q_ddot = np.zeros(self.nq, dtype=np.common_type(q, u, u_dot))
        q_ddot[:3] = u_dot[:3]
        q_ddot[3:] = p_ddot
        return q_ddot

    def step_callback(self, t, q, u):
        q[3:] = q[3:] / norm(q[3:])
        return q, u

    def A_IK(self, t, q, frame_ID=None):
        return Exp_SO3_quat(q[3:])

    def A_IK_q(self, t, q, frame_ID=None):
        A_IK_q = np.zeros((3, 3, self.nq), dtype=q.dtype)
        A_IK_q[:, :, 3:] = Exp_SO3_quat_p(q[3:])
        return A_IK_q
