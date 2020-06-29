import numpy as np
from cardillo.utility.check_time_derivatives import check_time_derivatives

class Frame():
    def __init__(self, r_OP=np.zeros(3), r_OP_t=None, r_OP_tt=None, A_IK=np.eye(3), A_IK_t=None, A_IK_tt=None):
        self.r_OP__, self.r_OP_t__, self.r_OP_tt__ = check_time_derivatives(r_OP, r_OP_t, r_OP_tt)
        self.A_IK__, self.A_IK_t__, self.A_IK_tt__ = check_time_derivatives(A_IK, A_IK_t, A_IK_tt)

        self.nq = 0
        self.nu = 0

        self.q0 = np.array([])
        self.u0 = np.array([])

    def qDOF_P(self, point_ID=None):
        return self.qDOF

    def uDOF_P(self, point_ID=None):
        return self.uDOF

    def r_OP(self, t, q=None, point_ID=None):
        return self.r_OP__(t)

    def r_OP_q(self, t, q=None, point_ID=None):
        return np.array([]).reshape((3, 0))

    def A_IK(self, t, q=None, point_ID=None):
        return self.A_IK__(t)

    def A_IK_q(self, t, q=None, point_ID=None):
        return np.array([]).reshape((3, 3, 0))

    def J_P(self, t, q=None, point_ID=None):
        return np.array([]).reshape((3, 0))

    def J_P_q(self, t, q, point_ID=None):
        return np.array([]).reshape((3, 0, 0))