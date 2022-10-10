import numpy as np

from cardillo.math.rotations import quat2mat, quat2mat_p, quat2rot, quat2rot_p


class Spherical_joint:
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
        return self.B_dense(t, q) @ u

    def q_ddot(self, t, q, u, u_dot):
        q2 = q @ q
        B_q = quat2mat_p(q) / (2 * q2) - np.einsum(
            "ij,k->ijk", quat2mat(q), q / (q2**2)
        )
        return self.B_dense(t, q) @ u_dot + np.einsum(
            "ijk,k,j->i", B_q[:, 1:], self.q_dot(t, q, u), u
        )

    def B_dense(self, t, q):
        Q = quat2mat(q) / (2 * q @ q)
        return Q[:, 1:]

    def B(self, t, q, coo):
        coo.extend(self.B_dense(t, q), (self.qDOF, self.uDOF))

    def q_dot_q(self, t, q, u, coo):
        q2 = q @ q
        B_q = quat2mat_p(q) / (2 * q2) - np.einsum(
            "ij,k->ijk", quat2mat(q), q / (q2**2)
        )
        coo.extend(np.einsum("ijk,j->ik", B_q[:, 1:], u), (self.qDOF, self.qDOF))

    # other functions
    def A_B1B2(self, t, q):
        return quat2rot(q)

    def A_B1B2_q(self, t, q):
        return quat2rot_p(q)

    def B1_r_B1B2(self, t, q):
        return np.zeros(3)

    def B1_r_B1B2_q(self, t, q):
        return np.zeros((3, self.nq))

    def B1_v_B1B2(self, t, q, u):
        return np.zeros(3)

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
