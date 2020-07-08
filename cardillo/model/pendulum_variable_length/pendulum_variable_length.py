import numpy as np

class Pendulum_variable_length():
    def __init__(self, m, l, l_t, F, r_pivot=np.zeros(3), q0=None, u0=None):
        self.m = m
        self.l = l
        self.l_t = l_t
        self.F = F
        self.r_pivot = r_pivot

        self.nq = 2
        self.nu = 1
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

    def M(self, t, q, coo):
        M = np.array([[self.m * self.l(t)**2]])
        coo.extend(M, (self.uDOF, self.uDOF))

    def f_gyr(self, t, q, u):
        return np.array([2 * self.m * self.l(t) * self.l_t(t) * u[0]])

    def f_gyr_u(self, t, q, u, coo):
        dense = np.array([[2 * self.m * self.l(t) * self.l_t(t)]])
        coo.extend(dense, (self.uDOF, self.uDOF))

    def f_pot(self, t, q):
        F = self.F(t)
        return np.array([F[0] * q[1] + F[1] * q[0]])

    def f_pot_q(self, t, q, coo):
        F = self.F(t)
        dense = np.array([[ F[1], F[0] ]])
        coo.extend(dense, (self.uDOF, self.qDOF))

    def q_dot(self, t, q, u):
        return np.array([q[1] * u[0], \
                         -q[0] * u[0]]) + self.l_t(t) / self.l(t) * q

    def q_dot_q(self, t, q, u, coo):
        dense = self.l_t(t) / self.l(t) * np.eye(2) \
                + np.array([[0, u[0]], \
                            [-u[0], 0]])
        coo.extend(dense, (self.qDOF, self.qDOF))        

    def B_dense(self, t, q):
        return np.array([[q[1]], \
                         [-q[0]]])

    def B(self, t, q, coo):
        coo.extend(self.B_dense(t, q), (self.qDOF, self.uDOF))

    def solver_step_callback(self, t, q, u):
        l_act = np.linalg.norm(q)
        q = self.l(t) / l_act * q
        return q, u

    def qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)

    def r_OP(self, t, q, frame_ID=(1,), K_r_SP=None):
        return self.r_pivot + frame_ID[0] * np.array([q[0], -q[1], 0])

    def r_OP_q(self, t, q, frame_ID=(1,), K_r_SP=None):
        return frame_ID[0] * np.array([[1 , 0 ], [0, -1], [0, 0]])

    def J_P(self, t, q, frame_ID=(1,), K_r_SP=None):
        return self.r_OP_q(t, q) @ self.B_dense(t, q)

    def J_P_q(self, t, q, frame_ID=(1,), K_r_SP=None):
        return frame_ID[0] * np.array( [ [ [0,1] ],[ [1, 0] ],[ [0,0] ] ])



