import numpy as np


class PointMass:
    def __init__(
        self,
        mass,
        q0=None,
        u0=None,
    ):
        self.nq = 3
        self.nu = 3

        self.q0 = np.zeros(self.nq) if q0 is None else np.asarray(q0)
        self.u0 = np.zeros(self.nu) if u0 is None else np.asarray(u0)
        assert self.q0.size == self.nq
        assert self.u0.size == self.nu

        self.mass = mass
        self.__M = mass * np.eye(3)

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q):
        return np.ones(self.nq)

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return self.__M

    #####################
    # auxiliary functions
    #####################
    def local_qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def local_uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)

    def r_OP(self, t, q, frame_ID=None, K_r_SP=None):
        r = np.zeros(3, dtype=q.dtype)
        r[: self.nq] = q
        return r

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=None):
        return np.eye(3, self.nq)

    def J_P(self, t, q, frame_ID=None, K_r_SP=None):
        return np.eye(3, self.nu, dtype=q.dtype)

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=None):
        return np.zeros((3, self.nu, self.nq))

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=None):
        v_P = np.zeros(3, dtype=np.common_type(q, u))
        v_P[: self.nq] = u
        return v_P

    def v_P_q(self, t, q, u, frame_ID=None, K_r_SP=None):
        return np.zeros((3, self.nq))

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=None):
        a_P = np.zeros(3, dtype=np.common_type(q, u, u_dot))
        a_P[: self.nq] = u_dot
        return a_P

    def a_P_q(self, t, q, u, u_dot, frame_ID=None, K_r_SP=None):
        return np.zeros((3, self.nq), dtype=np.common_type(q, u, u_dot))

    def a_P_u(self, t, q, u, u_dot, frame_ID=None, K_r_SP=None):
        return np.zeros((3, self.nu), dtype=np.common_type(q, u, u_dot))

    ########
    # export
    ########
    def export(self, sol_i, **kwargs):
        points = [self.r_OP(sol_i.t, sol_i.q[self.qDOF])]
        vel = [self.v_P(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])]
        cells = [("vertex", [[0]])]
        cell_data = dict(v=[[vel]])
        return points, cells, None, cell_data
