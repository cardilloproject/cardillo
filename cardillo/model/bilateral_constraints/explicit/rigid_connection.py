import numpy as np
from cardillo.math.algebra import A_IK_basic_z, dA_IK_basic_z

class Rigid_connection():
    def __init__(self, r_OB1, A_IB1):
        self.nq = 0
        self.nu = 0
        self.q0 = np.array([])
        self.u0 = np.array([])

        self.r_OB1 = r_OB1
        self.A_IB1  = A_IB1

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    def A_B1B2(self, t, q):
        return np.eye(3)

    def A_B1B2_q(self, t, q):
        return np.array([]).reshape((3, 3, 0))

    def B1_r_B1B2(self, t, q):
        return np.zeros(3)

    def B1_r_B1B2_q(self, t, q):
        return np.array([]).reshape((3, 0))

    def B1_v_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_J_B1B2(self, t, q):
        return np.array([]).reshape((3, 0))

    def B1_J_B1B2_q(self, t, q):
        return np.array([]).reshape((3, 0, 0))

    def B1_a_B1B2(self, t, q, u, u_dot):
        return np.zeros(3)

    def B1_kappa_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_kappa_B1B2_q(self, t, q, u):
        return np.array([]).reshape((3, 0))

    def B1_Omega_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_Omega_B1B2_q(self, t, q, u):
        return np.array([]).reshape((3, 0))

    def B1_J_R_B1B2(self, t, q):
        return np.array([]).reshape((3, 0))

    def B1_J_R_B1B2_q(self, t, q):
        return np.array([]).reshape((3, 0, 0))

    def B1_Psi_B1B2(self, t, q, u, u_dot):
        return np.zeros(3)

    def B1_kappa_R_B1B2(self, t, q, u):
        return np.zeros(3)

    def B1_kappa_R_B1B2_q(self, t, q, u):
        return np.array([]).reshape((3, 0))

    def B1_kappa_R_B1B2_u(self, t, q, u):
        return np.array([]).reshape((3, 0))