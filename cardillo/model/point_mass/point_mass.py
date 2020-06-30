import numpy as np

class Point_mass():
    def __init__(self, m, dim=3, q0=None, u0=None):
        self.m = m
        self.nq = dim
        self.nu = dim

        self.M_ = m * np.eye(dim)

        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

    def M(self, t, q, M_coo):
        M_coo.extend(self.M_, (self.uDOF, self.uDOF))

    def q_dot(self, t, q, u):
        return u

    def B_dense(self, t, q):
        return np.eye(self.nq)

    def B(self, t, q, B_coo):
        B_coo.extend(self.B_dense(t, q), (self.qDOF, self.uDOF))

    def qDOF_P(self, frame_ID=None):
        return self.qDOF

    def uDOF_P(self, frame_ID=None):
        return self.uDOF

    def r_OP(self, t, q, frame_ID=None, K_r_SP=None):
        r = np.zeros(3)
        r[:self.nq] = q
        return r

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=None):
        return np.eye(3, self.nq)

    def J_P(self, t, q, frame_ID=None, K_r_SP=None):
        return np.eye(3, self.nu)

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=None):
        return np.zeros((3, self.nu, self.nq))

