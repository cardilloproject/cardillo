import numpy as np
from cardillo.math.prox import prox_Rn0, prox_circle

class Sphere_to_plane():
    def __init__(self, frame, subsystem, r, mu, prox_r_N, prox_r_T, e_N=None, e_T=None, frame_ID=np.zeros(3), K_r_SP=np.zeros(3), la_N0=None, la_T0=None):
        
        self.frame = frame
        self.subsystem = subsystem
        self.r = r
        self.mu = mu
        self.prox_r_N = prox_r_N
        self.prox_r_T = prox_r_T

        self.nla_N = 1
        self.nla_T = 2 * self.nla_N
        self.NT_connectivity = np.array([ [0, 1] ])
        self.e_N = 0 if e_N is None else e_N
        self.e_T = 0 if e_T is None else e_T
        self.frame_ID = frame_ID

        self.r_OQ = lambda t: self.frame.r_OP(t)
        self.t1 = lambda t: self.frame.A_IK(t)[:, 0]
        self.t2 = lambda t: self.frame.A_IK(t)[:, 1]
        self.t1t2 = lambda t: self.frame.A_IK(t).T[:2]
        self.n = lambda t: self.frame.A_IK(t)[:, 2]
        self.v_Q = lambda t: self.frame.v_P(t)

        self.K_r_SP_ = K_r_SP 

        self.la_N0 = np.zeros(self.nla_N) if la_N0 is None else la_N0
        self.la_T0 = np.zeros(self.nla_T) if la_T0 is None else la_T0

        self.is_assembled = False

    def assembler_callback(self):

        self.qDOF = self.subsystem.qDOF_P(self.frame_ID)
        self.nq = len(self.qDOF)

        self.uDOF = self.subsystem.uDOF_P(self.frame_ID)
        self.nu = len(self.uDOF)

        self.K_r_SP = lambda t, q: self.K_r_SP_ - self.r * self.subsystem.A_IK(t, q, frame_ID=self.frame_ID).T @ self.n(t)
        
        self.r_OP = lambda t, q: self.subsystem.r_OP(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP(t, q))
        self.v_P = lambda t, q, u: self.subsystem.v_P(t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP(t, q))
        self.J_P = lambda t, q: self.subsystem.J_P(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP(t, q))

        self.is_assembled = True

    
    def g_N(self, t, q):
        return self.n(t) @ (self.r_OP(t, q) - self.r_OQ(t))

    def g_N_dot(self, t, q, u):
        return self.n(t) @ (self.v_P(t, q, u) - self.v_Q(t))
    
    def g_N_dot_u(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q), (self.la_NDOF, self.uDOF))

    def g_N_dot_u_dense(self, t, q):
        return np.array([self.n(t) @ self.J_P(t, q)])

    def W_N(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q).T, (self.uDOF, self.la_NDOF))

    def gamma_T(self, t, q, u):
        return self.t1t2(t) @ (self.v_P(t, q, u) - self.v_Q(t))

    def gamma_T_u_dense(self, t, q):
        return self.t1t2(t) @ self.J_P(t, q)

    def W_T(self, t, q, coo):
        coo.extend(self.gamma_T_u_dense(t, q).T, (self.uDOF, self.la_TDOF))

    def contact_force_fixpoint_update(self, t, q, u_pre, u_post, la_N, la_T):
        xi_N = self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)
        la_N1 = prox_Rn0(la_N - self.prox_r_N * xi_N) 
        xi_T = self.gamma_T(t, q, u_post) + self.e_T * self.gamma_T(t, q, u_pre)
        la_T1 = prox_circle(la_T - self.prox_r_T * xi_T, self.mu * la_N1)
        return la_N1, la_T1