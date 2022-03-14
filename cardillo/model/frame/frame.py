import numpy as np
from cardillo.utility.check_time_derivatives import check_time_derivatives
from cardillo.math.algebra import skew2ax


class Frame:
    def __init__(
        self,
        r_OP=np.zeros(3),
        r_OP_t=None,
        r_OP_tt=None,
        A_IK=np.eye(3),
        A_IK_t=None,
        A_IK_tt=None,
    ):
        self.r_OP__, self.r_OP_t__, self.r_OP_tt__ = check_time_derivatives(
            r_OP, r_OP_t, r_OP_tt
        )
        self.A_IK__, self.A_IK_t__, self.A_IK_tt__ = check_time_derivatives(
            A_IK, A_IK_t, A_IK_tt
        )

        self.nq = 0
        self.nu = 0

        self.q0 = np.array([])
        self.u0 = np.array([])

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    #########################################
    # helper functions
    #########################################

    def qDOF_P(self, frame_ID=None):
        return np.array([], dtype=int)

    def uDOF_P(self, frame_ID=None):
        return np.array([], dtype=int)

    def q_dot(self, t, q, u):
        return np.array([])

    def q_ddot(self, t, q, u, u_dot):
        return np.array([])

    def A_IK(self, t, q=None, frame_ID=None):
        return self.A_IK__(t)

    def A_IK_q(self, t, q=None, frame_ID=None):
        return np.array([]).reshape((3, 3, 0))

    def r_OP(self, t, q=None, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.r_OP__(t) + self.A_IK__(t) @ K_r_SP

    def r_OP_q(self, t, q=None, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def r_OP_qq(self, t, q=None, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.array([]).reshape((3, 0, 0))

    def v_P(self, t, q=None, u=None, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.r_OP_t__(t) + self.A_IK_t__(t) @ K_r_SP

    def v_P_q(self, t, q=None, u=None, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def J_P(self, t, q=None, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.array([]).reshape((3, 0, 0))

    def a_P(self, t, q=None, u=None, u_dot=None, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.r_OP_tt__(t) + self.A_IK_tt__(t) @ K_r_SP

    def a_P_q(self, t, q=None, u=None, u_dot=None, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def a_P_u(self, t, q=None, u=None, u_dot=None, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def kappa_P(self, t, q=None, u=None, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.r_OP_tt__(t)

    def kappa_P_q(self, t, q=None, u=None, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def kappa_P_u(self, t, q=None, u=None, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def K_Omega(self, t, q=None, u=None, frame_ID=None):
        K_omega_IK = self.A_IK__(t).T @ self.A_IK_t__(t)
        return skew2ax(K_omega_IK)

    def K_Omega_q(self, t, q=None, u=None, frame_ID=None):
        return np.array([]).reshape((3, 0))

    def K_J_R(self, t, q, frame_ID=None):
        return np.array([]).reshape((3, 0))

    def K_J_R_q(self, t, q, frame_ID=None):
        return np.array([]).reshape((3, 0, 0))

    def K_Psi(self, t, q=None, u=None, u_dot=None, frame_ID=None):
        K_psi_IK = self.A_IK_t__(t).T @ self.A_IK_t__(t) + self.A_IK__(
            t
        ).T @ self.A_IK_tt__(t)
        return skew2ax(K_psi_IK)

    def K_kappa_R(self, t, q=None, u=None, frame_ID=None):
        return self.K_Psi(t)

    def K_kappa_R_q(self, t, q=None, u=None, frame_ID=None):
        return np.array([]).reshape((3, 0))

    def K_kappa_R_u(self, t, q=None, u=None, frame_ID=None):
        return np.array([]).reshape((3, 0))
