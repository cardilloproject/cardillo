import numpy as np


class Point_mass:
    def __init__(self, m, dim=3, q0=None, u0=None):
        self.m = m
        self.nq = dim
        self.nu = dim

        self.M_ = m * np.eye(dim)

        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

    def M(self, t, q, coo):
        coo.extend(self.M_, (self.uDOF, self.uDOF))

    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q, coo):
        coo.extend_diag(np.ones(self.nq), (self.qDOF, self.uDOF))

    def qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)

    def r_OP(self, t, q, frame_ID=None, K_r_SP=None):
        r = np.zeros(3)
        r[: self.nq] = q
        return r

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=None):
        return np.eye(3, self.nq)

    def J_P(self, t, q, frame_ID=None, K_r_SP=None):
        return np.eye(3, self.nu)

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=None):
        return np.zeros((3, self.nu, self.nq))

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=None):
        v_P = np.zeros(3)
        v_P[: self.nq] = u
        return v_P

    def v_P_q(self, t, q, u, frame_ID=None, K_r_SP=None):
        return np.zeros((3, self.nq))

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=None):
        a_P = np.zeros(3)
        a_P[: self.nq] = u_dot
        return a_P
