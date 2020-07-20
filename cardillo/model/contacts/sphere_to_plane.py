import numpy as np

class Sphere_to_plane():
    def __init__(self, frame, subsystem, r, prox_r_N, e_N=np.array([0]), frame_ID=np.zeros(3), K_r_SP=np.zeros(3), la_N0=None):

        self.nla_N = 1
        self.la_N0 = np.zeros(self.nla_N) if la_N0 is None else la_N0

        self.subsystem = subsystem
        self.frame_ID = frame_ID
        self.K_r_SP = K_r_SP
        self.frame = frame
        self.r = r
        self.e_N = e_N
        self.prox_r_N = prox_r_N
        
        self.is_assembled = False

    def assembler_callback(self):

        self.qDOF = self.subsystem.qDOF_P(self.frame_ID)
        self.nq = len(self.qDOF)

        self.uDOF = self.subsystem.uDOF_P(self.frame_ID)
        self.nu = len(self.uDOF)
        
        self.r_OP = lambda t, q: self.subsystem.r_OP(t, q, self.frame_ID, self.K_r_SP)
        # self.r_OP1_q = lambda t, q: self.subsystem1.r_OP_q(t, q[:nq1], self.frame_ID1, K_r_SP1)
        self.v_P = lambda t, q, u: self.subsystem.v_P(t, q, u, self.frame_ID, self.K_r_SP)
        # self.a_P1 = lambda t, q, u, u_dot: self.subsystem1.a_P(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1, K_r_SP1)
        self.J_P = lambda t, q: self.subsystem.J_P(t, q, self.frame_ID, self.K_r_SP)
        # self.J_P1_q = lambda t, q: self.subsystem1.J_P_q(t, q[:nq1], self.frame_ID1, K_r_SP1)

        self.r_OQ = lambda t: self.frame.r_OP(t)
        self.n = lambda t: self.frame.A_IK(t)[:, 2]
        # self.r_OP2_q = lambda t, q: self.subsystem2.r_OP_q(t, q[nq1:], self.frame_ID2, K_r_SP2)
        self.v_Q = lambda t: self.frame.v_P(t)
        # self.a_P2 = lambda t, q, u, u_dot: self.subsystem2.a_P(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2, K_r_SP2)
        # self.J_P2 = lambda t, q: self.subsystem2.J_P(t, q[nq1:], self.frame_ID2, K_r_SP2)
        # self.J_P2_q = lambda t, q: self.subsystem2.J_P_q(t, q[nq1:], self.frame_ID2, K_r_SP2)

        self.is_assembled = True

    
    def g_N(self, t, q):
        return self.n(t) @ (self.r_OP(t, q) - self.r_OQ(t)) - self.r

    def g_N_dot(self, t, q, u):
        return self.n(t) @ (self.v_P(t, q, u) - self.v_Q(t))
    
    def g_N_dot_u(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q), (self.la_NDOF, self.uDOF))

    def g_N_dot_u_dense(self, t, q):
        return np.array([self.n(t) @ self.J_P(t, q)])

    def W_N(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q).T, (self.uDOF, self.la_NDOF))