import numpy as np
from cardillo.math import norm, ax2skew, e1, e3


class RevoluteJoint:
    def __init__(self, r_OB1, A_IB1, B1_axis=e1, q0=None, u0=None):
        self.nq = 1
        self.nu = 1
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

        self.B1_axis = B1_axis / norm(B1_axis)
        self.B1_axis_tilde = ax2skew(self.B1_axis)
        self.B1_axis_tilde_squared = self.B1_axis_tilde @ self.B1_axis_tilde
        self.angle0 = self.q0[0]

        self.r_OB1 = r_OB1
        self.A_IB1 = A_IB1

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    # kinematic equation
    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q):
        return np.ones(self.nq)

    # other functions
    def A_B1B2(self, t, q):
        angle = q[0]
        return (
            np.eye(3)
            + np.sin(angle) * self.B1_axis_tilde
            + (1 - np.cos(angle)) * self.B1_axis_tilde_squared
        )

    def A_B1B2_q(self, t, q):
        angle = q[0]
        A_B1B2_q = np.zeros((3, 3, 1))
        A_B1B2_q[:, :, 0] = (
            np.cos(angle) * self.B1_axis_tilde
            + np.sin(angle) * self.B1_axis_tilde_squared
        )
        return A_B1B2_q

    def B1_r_B1B2(self, t, q):
        return np.zeros(3)

    def B1_r_B1B2_q(self, t, q):
        return np.zeros((3, self.nq))

    def B1_v_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_v_B1B2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    def B1_J_B1B2(self, t, q):
        return np.zeros((3, 1))

    def B1_J_B1B2_q(self, t, q):
        return np.zeros((3, 1, 1))

    def B1_a_B1B2(self, t, q, u, u_dot):
        return np.zeros(3)

    def B1_kappa_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_kappa_B1B2_q(self, t, q, u):
        return np.zeros((3, 1))

    def B1_kappa_B1B2_u(self, t, q, u):
        return np.zeros((3, 1))

    def B1_Omega_B1B2(self, t, q, u):
        return u[0] * self.B1_axis

    def B1_Omega_B1B2_q(self, t, q, u):
        return np.zeros((3, 1))

    def B1_J_R_B1B2(self, t, q):
        return self.B1_axis.reshape((3, 1))

    def B1_J_R_B1B2_q(self, t, q):
        return np.zeros((3, 1, 1))

    def B1_Psi_B1B2(self, t, q, u, u_dot):
        return u_dot[0] * self.B1_axis

    def B1_kappa_R_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_kappa_R_B1B2_q(self, t, q, u):
        return np.zeros((3, 1))

    def B1_kappa_R_B1B2_u(self, t, q, u):
        return np.zeros((3, 1))

    def l(self, t, q):
        return q[0]

    def l_dot(self, t, q, u):
        return u[0]

    def l_dot_q(self, t, q, u):
        return np.zeros(1)
    
    def l_dot_u(self, t, q, u):
        return np.ones(1)

    def l_q(self, t, q):
        return np.ones(1)

    def W_l(self, t, q):
        return np.eye(1)

    def W_l_q(self, t, q):
        return np.zeros((1, 1, 1))
