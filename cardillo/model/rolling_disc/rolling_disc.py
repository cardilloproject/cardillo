import numpy as np
from cardillo.math.numerical_derivative import Numerical_derivative
from cardillo.math.algebra import A_IK_basic_x, A_IK_basic_y, A_IK_basic_z

class Rolling_disc():
    def __init__(self, m, r, g, q0=None, u0=None):
        ''' Rolling disc of Ex. 5.11 (DMS)'''
        self.m = m
        self.r = r
        self.g = g
        self.A = m * r**2 / 4

        self.nq = 5
        self.nu = 3
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

    def M(self, t, q, coo):
        sb = np.sin(q[3])
        M = self.A * np.array([[5 * sb**2 + 1, 0, 6 * sb],\
                               [0            , 5, 0     ],\
                               [6 * sb       , 0, 6     ]])
        coo.extend(M, (self.uDOF, self.uDOF))

    def Mu_q(self, t, q, u, coo):
        sb = np.sin(q[3])
        cb = np.cos(q[3])
        dense = np.zeros((self.nu, self.nq))
        dense[0, 3] = 10 * sb * cb * u[0] + 6 * cb * u[2]
        dense[2, 3] =  6 * cb * u[0]
        coo.extend(self.A * dense, (self.uDOF, self.qDOF))

    def f_gyr(self, t, q, u):
        a_t = u[0]
        b_t = u[1]
        g_t = u[2]
        sb = np.sin(q[3])
        cb = np.cos(q[3])
        return self.A * np.array([10 * a_t * b_t * sb * cb + 2 * b_t * g_t * cb,\
                                  -5 * a_t**2 * sb * cb - 6 * a_t * g_t * cb,\
                                  10 * a_t * b_t * cb])

    def f_gyr_q(self, t, q, u, coo):
        dense = Numerical_derivative(self.f_gyr, order=2)._x(t, q, u)
        coo.extend(dense, (self.uDOF, self.qDOF))

    def f_gyr_u(self, t, q, u, coo):
        dense = Numerical_derivative(self.f_gyr, order=2)._y(t, q, u)
        coo.extend(dense, (self.uDOF, self.uDOF))

    def f_pot(self, t, q):
        sb = np.sin(q[3])
        return self.A * np.array([0, 4 * self.g / self.r * sb, 0])

    def f_pot_q(self, t, q, coo):
        cb = np.cos(q[3])
        dense = np.zeros((self.nu, self.nq))
        dense[1, 3] = 4 * self.g / self.r * cb
        coo.extend(dense, (self.uDOF, self.qDOF))

    def q_dot(self, t, q, u):
        return self.B_dense(t, q) @ u

    def q_dot_q(self, t, q, u, coo):
        dense = np.zeros((self.nq, self.nq))
        dense[0, 1] = u[0]
        dense[1, 0] = u[0]
        coo.extend(dense, (self.qDOF, self.qDOF))        

    def B_dense(self, t, q):
        x, y = q[:2]
        B = np.zeros((self.nq, self.nu))
        B[0] = np.array([y, 0, self.r])
        B[1, 0] = x
        B[2:, :] = np.eye(3)
        return B

    def B(self, t, q, coo):
        coo.extend(self.B_dense(t, q), (self.qDOF, self.uDOF))

    def r_OS(self, t, q):
        x, y, a, b, g = q
        A_IR = A_IK_basic_z(a)
        R_r_OS = np.array([x, y - self.r * np.sin(b), self.r * np.cos(b)])
        return A_IR @ R_r_OS

    def r_OA(self, t, q):
        x, y, a= q[:3]
        A_IR = A_IK_basic_z(a)
        R_r_OA = np.array([x, y, 0])
        return A_IR @ R_r_OA

    def A_IK(self, t, q):
        a, b, g = q[2:]
        A_IR = A_IK_basic_z(a)
        A_RB = A_IK_basic_x(b)
        A_BK = A_IK_basic_y(g)
        return A_IR @ A_RB @ A_BK

    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.r * np.vstack([np.sin(phi), np.zeros(n), np.cos(phi)])
        return np.repeat(self.r_OS(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP

    # def qDOF_P(self, frame_ID=None):
    #     return self.qDOF

    # def uDOF_P(self, frame_ID=None):
    #     return self.uDOF

    # def r_OP(self, t, q, frame_ID=(1,), K_r_SP=None):
    #     return self.r_pivot + frame_ID[0] * np.array([q[0], -q[1], 0])

    # def r_OP_q(self, t, q, frame_ID=(1,), K_r_SP=None):
    #     return frame_ID[0] * np.array([[1 , 0 ], [0, -1], [0, 0]])

    # def J_P(self, t, q, frame_ID=(1,), K_r_SP=None):
    #     return self.r_OP_q(t, q) @ self.B_dense(t, q)

    # def J_P_q(self, t, q, frame_ID=(1,), K_r_SP=None):
    #     return frame_ID[0] * np.array( [ [ [0,1] ],[ [1, 0] ],[ [0,0] ] ])



