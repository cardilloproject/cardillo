import numpy as np

from cardillo.math import (
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
)


class SphericalJoint:
    def __init__(self, r_OB1, A_IB1, q0=None, u0=None):
        self.nq = 4
        self.nu = 3
        self.q0 = np.array([1, 0, 0, 0]) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0
        self.r_OB1 = r_OB1
        self.A_IB1 = A_IB1

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    # kinematic equation
    def q_dot(self, t, q, u):
        return self.B(t, q) @ u

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

    def B(self, t, q):
        return T_SO3_inv_quat(q) / (q @ q)

    def q_dot_q(self, t, q, u):
        q2 = q @ q
        return np.einsum("ijk,j->ik", T_SO3_inv_quat_P() / q2, u) - np.outer(
            T_SO3_inv_quat(q) @ u, 2 * q / (q2**2)
        )

    # other functions
    def A_B1B2(self, t, q):
        return Exp_SO3_quat(q)

    def A_B1B2_q(self, t, q):
        return Exp_SO3_quat_p(q)

    def B1_r_B1B2(self, t, q):
        return np.zeros(3)

    def B1_r_B1B2_q(self, t, q):
        return np.zeros((3, self.nq))

    def B1_v_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_v_B1B2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    def B1_J_B1B2(self, t, q):
        return np.zeros((3, self.nu))

    def B1_J_B1B2_q(self, t, q):
        return np.zeros((3, self.nu, self.nq))

    def B1_a_B1B2(self, t, q, u, u_dot):
        return np.zeros(3)

    def B1_kappa_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_kappa_B1B2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    def B1_kappa_B1B2_u(self, t, q, u):
        return np.zeros((3, self.nu))

    def B1_Omega_B1B2(self, t, q, u):
        return u

    def B1_Omega_B1B2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    def B1_J_R_B1B2(self, t, q):
        return np.eye(3)

    def B1_J_R_B1B2_q(self, t, q):
        return np.zeros((3, self.nu, self.nq))

    def B1_Psi_B1B2(self, t, q, u, u_dot):
        return u_dot

    def B1_kappa_R_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_kappa_R_B1B2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    def B1_kappa_R_B1B2_u(self, t, q, u):
        return np.zeros((3, self.nu))
