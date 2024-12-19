import numpy as np

from cardillo.math import (
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
)

class FloatingJoint:
    def __init__(self, r_OB1=np.zeros(3), A_IB1=np.eye(3), q0=None, u0=None):
        self.nq = 7
        self.nu = 6
        self.q0 = np.array([0, 0, 0, 1, 0, 0, 0]) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0
        self.r_OB1 = r_OB1
        self.A_IB1 = A_IB1

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    # kinematic equation
    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.nq)
        q_dot[:3] = u[:3]
        q_dot[3:] = self._H(t, q) @ u[3:]
        return q_dot

    def q_ddot(self, t, q, u, u_dot):
        raise RuntimeWarning("SphericalJoint.q_ddot is not tested yet!")
        q2 = q @ q
        B = self.B(t, q)
        q_dot = B @ u
        return (
            B @ u_dot
            + np.einsum("ijk,k,j->i", T_SO3_inv_quat_P(q), q_dot, u)
            + 2 * q_dot * (q @ q_dot) / q2
        )

    def _H(self, t, q):
        p = q[3:]
        return T_SO3_inv_quat(p) / (p @ p)
    
    def B(self, t, q):
        B = np.eye(self.nq, self.nu)
        B[3:, 3:] = self._H(t, q)
        return B

    def q_dot_q(self, t, q, u):
        p = q[3:]
        p2 = p @ p
        omega = u[3:]
        q_dot_q = np.zeros((self.nq, self.nq))
        q_dot_q[3:, 3:] = np.einsum("ijk,j->ik", T_SO3_inv_quat_P(p) / p2, omega) - np.outer(
            T_SO3_inv_quat(p) @ omega, 2 * p / (p2**2)
        )
        return q_dot_q

    # other functions
    def A_B1B2(self, t, q):
        return Exp_SO3_quat(q[3:])

    def A_B1B2_q(self, t, q):
        A_B1B2_q = np.zeros((3, 3, self.nq))
        A_B1B2_q[:, :, 3:] = Exp_SO3_quat_p(q[3:])
        return A_B1B2_q

    def B1_r_B1B2(self, t, q):
        return q[:3]

    def B1_r_B1B2_q(self, t, q):
        B1_r_B1B2_q = np.zeros((3, self.nq))
        B1_r_B1B2_q[:, :3] = np.eye(3)
        return B1_r_B1B2_q

    def B1_v_B1B2(self, t, q, u):
        return u[:3]

    def B1_v_B1B2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    def B1_J_B1B2(self, t, q):
        J = np.zeros((3, self.nu))
        J[:, :3] = np.eye(3)
        return J

    def B1_J_B1B2_q(self, t, q):
        return np.zeros((3, self.nu, self.nq))

    def B1_a_B1B2(self, t, q, u, u_dot):
        return u_dot[:3]

    def B1_kappa_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_kappa_B1B2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    def B1_kappa_B1B2_u(self, t, q, u):
        return np.zeros((3, self.nu))

    def B1_Omega_B1B2(self, t, q, u):
        return u[3:]

    def B1_Omega_B1B2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    def B1_J_R_B1B2(self, t, q):
        J = np.zeros((3, self.nu))
        J[:, 3:] = np.eye(3)
        return J

    def B1_J_R_B1B2_q(self, t, q):
        return np.zeros((3, self.nu, self.nq))

    def B1_Psi_B1B2(self, t, q, u, u_dot):
        return u_dot[3:]

    def B1_kappa_R_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_kappa_R_B1B2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    def B1_kappa_R_B1B2_u(self, t, q, u):
        return np.zeros((3, self.nu))
