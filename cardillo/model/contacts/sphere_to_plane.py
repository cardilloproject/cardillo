import numpy as np
from cardillo.math.prox import prox_Rn0, prox_circle
from cardillo.math.numerical_derivative import Numerical_derivative

class Sphere_to_plane():
    def __init__(self, frame, subsystem, r, mu, prox_r_N, prox_r_T, e_N=None, e_T=None, frame_ID=np.zeros(3), K_r_SP=np.zeros(3), la_N0=None, la_T0=None):
        
        self.frame = frame
        self.subsystem = subsystem
        self.r = r
        self.mu = np.array([mu])
        self.prox_r_N = np.array([prox_r_N])
        self.prox_r_T = np.array([prox_r_T])

        self.nla_N = 1

        if mu == 0:
            self.nla_T = 0
            self.NT_connectivity = [[]]
        else:
            self.nla_T =  2 * self.nla_N 
            self.NT_connectivity = [ [0, 1] ]
            self.gamma_T = self.__gamma_T
            
        self.e_N = np.zeros(self.nla_N) if e_N is None else np.array([e_N])
        self.e_T = np.zeros(self.nla_N) if e_T is None else np.array([e_T])
        self.frame_ID = frame_ID

        self.r_OQ = lambda t: self.frame.r_OP(t)
        self.t1t2 = lambda t: self.frame.A_IK(t).T[:2]
        self.n = lambda t: self.frame.A_IK(t)[:, 2]
        self.v_Q = lambda t: self.frame.v_P(t)
        self.a_Q = lambda t: self.frame.a_P(t)

        self.K_r_SP_ = K_r_SP 

        self.la_N0 = np.zeros(self.nla_N) if la_N0 is None else la_N0
        self.la_T0 = np.zeros(self.nla_T) if la_T0 is None else la_T0

        self.is_assembled = False

    def assembler_callback(self):
        qDOF = self.subsystem.qDOF_P(self.frame_ID)
        self.qDOF = self.subsystem.qDOF[qDOF]
        self.nq = len(self.qDOF)

        uDOF = self.subsystem.uDOF_P(self.frame_ID)
        self.uDOF = self.subsystem.uDOF[uDOF]
        self.nu = len(self.uDOF)

        if self.r == 0:
            self.K_r_SP = lambda t, q: self.K_r_SP_
        else:
            self.K_r_SP = lambda t, q: self.K_r_SP_ - self.r * self.subsystem.A_IK(t, q, frame_ID=self.frame_ID).T @ self.n(t)
        
        self.r_OP = lambda t, q: self.subsystem.r_OP(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP(t, q))
        self.v_P = lambda t, q, u: self.subsystem.v_P(t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP(t, q))
        self.J_P = lambda t, q: self.subsystem.J_P(t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP(t, q))
        self.a_P = lambda t, q, u, a: self.subsystem.a_P(t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP(t, q))

        self.is_assembled = True

    def g_N(self, t, q):
        return np.array([self.n(t) @ (self.r_OP(t, q) - self.r_OQ(t))])

    def g_N_dot(self, t, q, u):
        #TODO: n_dot(t)
        return np.array([self.n(t) @ (self.v_P(t, q, u) - self.v_Q(t))])

    def g_N_ddot(self, t, q, u, u_dot):
        # return np.array([self.n(t) @ (self.a_P(t, q, u, u_dot) - self.a_Q(t))])
        g_N_dot_q = Numerical_derivative(self.g_N_dot, order=2)._x(t, q, u)
        g_N_dot_u = self.g_N_dot_u_dense(t, q)
        return g_N_dot_q @ self.subsystem.q_dot(t, q, u) + g_N_dot_u @ u_dot
    
    def g_N_dot_u(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q), (self.la_NDOF, self.uDOF))

    def g_N_dot_u_dense(self, t, q):
        return np.array([self.n(t) @ self.J_P(t, q)])

    def W_N(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q).T, (self.uDOF, self.la_NDOF))

    def __gamma_T(self, t, q, u):
        return self.t1t2(t) @ (self.v_P(t, q, u) - self.v_Q(t))

    def gamma_T_dot(self, t, q, u, u_dot):
        #TODO: t1t2_dot(t)
        # return self.t1t2(t) @ (self.a_P(t, q, u, u_dot) - self.a_Q(t))
        gamma_T_q = Numerical_derivative(self.gamma_T, order=2)._x(t, q, u)
        gamma_T_u = self.gamma_T_u_dense(t, q)
        return gamma_T_q @ self.subsystem.q_dot(t, q, u) + gamma_T_u @ u_dot

    def gamma_T_u_dense(self, t, q):
        return self.t1t2(t) @ self.J_P(t, q)

    def W_T(self, t, q, coo):
        coo.extend(self.gamma_T_u_dense(t, q).T, (self.uDOF, self.la_TDOF))

    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def xi_T(self, t, q, u_pre, u_post):
        return self.gamma_T(t, q, u_post) + self.e_T * self.gamma_T(t, q, u_pre)