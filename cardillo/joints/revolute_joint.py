import numpy as np
from cardillo.math.rotations import A_IK_basic


class RevoluteJoint:
    def __init__(self, r_OB1, A_IB1, q0=None, u0=None):
        self.nq = 1
        self.nu = 1
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

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
        return A_IK_basic(q[0]).z()

    def A_B1B2_q(self, t, q):
        A_B1B2_q = np.zeros((3, 3, 1))
        A_B1B2_q[:, :, 0] = A_IK_basic(q[0]).dz()
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
        return np.array([0, 0, u[0]])

    def B1_Omega_B1B2_q(self, t, q, u):
        return np.zeros((3, 1))

    def B1_J_R_B1B2(self, t, q):
        return np.array([[0], [0], [1]])

    def B1_J_R_B1B2_q(self, t, q):
        return np.zeros((3, 1, 1))

    def B1_Psi_B1B2(self, t, q, u, u_dot):
        return np.array([0, 0, u_dot[0]])

    def B1_kappa_R_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_kappa_R_B1B2_q(self, t, q, u):
        return np.zeros((3, 1))

    def B1_kappa_R_B1B2_u(self, t, q, u):
        return np.zeros((3, 1))

    def angle(self, t, q):
        return q[0]

    def angle_dot(self, t, q, u):
        return u[0]

    def angle_dot_q(self, t, q, u):
        return np.zeros(1)

    def angle_q(self, t, q):
        return np.ones(1)

    def W_angle(self, t, q):
        return np.ones(1)

    def W_angle_q(self, t, q):
        return np.zeros((1, 1))