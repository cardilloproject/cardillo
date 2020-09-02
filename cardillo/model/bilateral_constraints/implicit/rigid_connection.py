from cardillo.math.numerical_derivative import Numerical_derivative
import numpy as np
from cardillo.math.algebra import cross3, ax2skew

class Rigid_connection():
    def __init__(self, subsystem1, subsystem2, r_OB, frame_ID1=np.zeros(3), frame_ID2=np.zeros(3), la_g0=None):
        self.nla_g = 6
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2
        self.r_OB = r_OB

    def assembler_callback(self):
        qDOF1 = self.subsystem1.qDOF_P(self.frame_ID1)
        qDOF2 = self.subsystem2.qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate([self.subsystem1.qDOF[qDOF1], self.subsystem2.qDOF[qDOF2]])
        self.nq1 = nq1 = len(qDOF1)
        self.nq2 = len(qDOF2)
        self._nq = self.nq1 + self.nq2
        
        uDOF1 = self.subsystem1.uDOF_P(self.frame_ID1)
        uDOF2 = self.subsystem2.uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate([self.subsystem1.uDOF[uDOF1], self.subsystem2.uDOF[uDOF2]])
        self.nu1 = nu1 = len(uDOF1)
        self.nu2 = len(uDOF2)
        self._nu = self.nu1 + self.nu2
        
        A_IK1 = self.subsystem1.A_IK(self.subsystem1.t0, self.subsystem1.q0[qDOF1], frame_ID=self.frame_ID1)
        A_IK2 = self.subsystem2.A_IK(self.subsystem2.t0, self.subsystem2.q0[qDOF2], frame_ID=self.frame_ID2)
        A_K2B2 = A_IK2.T @ A_IK1

        r_OS1 = self.subsystem1.r_OP(self.subsystem1.t0, self.subsystem1.q0[qDOF1], frame_ID=self.frame_ID1) 
        r_OS2 = self.subsystem2.r_OP(self.subsystem2.t0, self.subsystem2.q0[qDOF2], frame_ID=self.frame_ID2)
        K_r_SP1 = A_IK1.T @ (self.r_OB - r_OS1)
        K_r_SP2 = A_IK2.T @ (self.r_OB - r_OS2)

        self.r_OP1 = lambda t, q: self.subsystem1.r_OP(t, q[:nq1], frame_ID=self.frame_ID1, K_r_SP=K_r_SP1)
        self.r_OP1_q = lambda t, q: self.subsystem1.r_OP_q(t, q[:nq1], frame_ID=self.frame_ID1, K_r_SP=K_r_SP1)
        self.v_P1 = lambda t, q, u: self.subsystem1.v_P(t, q[:nq1], u[:nu1], self.frame_ID1, K_r_SP1)
        self.a_P1 = lambda t, q, u, u_dot: self.subsystem1.a_P(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1, K_r_SP1)
        self.J_P1 = lambda t, q: self.subsystem1.J_P(t, q[:nq1], frame_ID=self.frame_ID1, K_r_SP=K_r_SP1)
        self.J_P1_q = lambda t, q: self.subsystem1.J_P_q(t, q[:nq1], frame_ID=self.frame_ID1, K_r_SP=K_r_SP1)
        self.A_IB1 = lambda t, q: self.subsystem1.A_IK(t, q[:nq1], frame_ID=self.frame_ID1)
        self.A_IB1_q = lambda t, q: self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1)
        self.Omega1 = lambda t, q, u: self.subsystem1.A_IK(t, q[:nq1], self.frame_ID1) @ self.subsystem1.K_Omega(t, q[:nq1], u[:nu1], self.frame_ID1)
        self.Psi1 = lambda t, q, u, u_dot: self.subsystem1.A_IK(t, q[:nq1], self.frame_ID1) @ self.subsystem1.K_Psi(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1)
        self.J_R1 = lambda t, q: self.subsystem1.A_IK(t, q[:nq1], frame_ID=self.frame_ID1) @ self.subsystem1.K_J_R(t, q[:nq1], frame_ID=self.frame_ID1)
        self.J_R1_q = lambda t, q: np.einsum('ijk,jl->ilk', self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1), self.subsystem1.K_J_R(t, q[:nq1], frame_ID=self.frame_ID1) ) + np.einsum('ij,jkl->ikl', self.subsystem1.A_IK(t, q[:nq1], frame_ID=self.frame_ID1), self.subsystem1.K_J_R_q(t, q[:nq1], frame_ID=self.frame_ID1) )

        self.r_OP2 = lambda t, q: self.subsystem2.r_OP(t, q[nq1:], frame_ID=self.frame_ID2, K_r_SP=K_r_SP2)
        self.r_OP2_q = lambda t, q: self.subsystem2.r_OP_q(t, q[nq1:], frame_ID=self.frame_ID2, K_r_SP=K_r_SP2)
        self.v_P2 = lambda t, q, u: self.subsystem2.v_P(t, q[nq1:], u[nu1:], self.frame_ID2, K_r_SP2)
        self.a_P2 = lambda t, q, u, u_dot: self.subsystem2.a_P(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2, K_r_SP2)
        self.J_P2 = lambda t, q: self.subsystem2.J_P(t, q[nq1:], frame_ID=self.frame_ID2, K_r_SP=K_r_SP2)
        self.J_P2_q = lambda t, q: self.subsystem2.J_P_q(t, q[nq1:], frame_ID=self.frame_ID2, K_r_SP=K_r_SP2)
        self.A_IB2 = lambda t, q:  self.subsystem2.A_IK(t, q[nq1:], frame_ID=self.frame_ID2) @ A_K2B2
        self.A_IB2_q = lambda t, q: np.einsum('ijk,jl->ilk', self.subsystem2.A_IK_q(t, q[nq1:], frame_ID=self.frame_ID2), A_K2B2 )
        self.Omega2 = lambda t, q, u:  self.subsystem2.A_IK(t, q[nq1:], self.frame_ID2) @ self.subsystem2.K_Omega(t, q[nq1:], u[nu1:], self.frame_ID2)
        self.Psi2 = lambda t, q, u, u_dot:  self.subsystem2.A_IK(t, q[nq1:], self.frame_ID2) @ self.subsystem2.K_Psi(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2)
        self.J_R2 = lambda t, q: self.subsystem2.A_IK(t, q[nq1:], frame_ID=self.frame_ID2) @ self.subsystem2.K_J_R(t, q[nq1:], frame_ID=self.frame_ID2)
        self.J_R2_q = lambda t, q: np.einsum('ijk,jl->ilk', self.subsystem2.A_IK_q(t, q[nq1:], frame_ID=self.frame_ID2), self.subsystem2.K_J_R(t, q[nq1:], frame_ID=self.frame_ID2) ) + np.einsum('ij,jkl->ikl', self.subsystem2.A_IK(t, q[nq1:], frame_ID=self.frame_ID2), self.subsystem2.K_J_R_q(t, q[nq1:], frame_ID=self.frame_ID2) )

    def g(self, t, q):
        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T
        return np.concatenate([r_OP2 - r_OP1, [ex1 @ ey2, ey1 @ ez2, ez1 @ ex2]])

    def g_q_dense(self, t, q):
        nq1 = self.nq1
        g_q = np.zeros((self.nla_g, self._nq))
        g_q[:3, :nq1] = - self.r_OP1_q(t, q) 
        g_q[:3, nq1:] = self.r_OP2_q(t, q)

        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        A_IB1_q = self.A_IB1_q(t, q)
        ex1_q = A_IB1_q[:, 0]
        ey1_q = A_IB1_q[:, 1]
        ez1_q = A_IB1_q[:, 2]
        A_IB2_q = self.A_IB2_q(t, q)
        ex2_q = A_IB2_q[:, 0]
        ey2_q = A_IB2_q[:, 1]
        ez2_q = A_IB2_q[:, 2]

        g_q[3, :nq1] = ey2 @ ex1_q
        g_q[3, nq1:] = ex1 @ ey2_q
        g_q[4, :nq1] = ez2 @ ey1_q
        g_q[4, nq1:] = ey1 @ ez2_q
        g_q[5, :nq1] = ex2 @ ez1_q
        g_q[5, nq1:] = ez1 @ ex2_q
        return g_q

        # g_q_num = Numerical_derivative(self.g, order=2)._x(t, q)
        # diff = g_q_num - g_q
        # diff_error = diff
        # error = np.linalg.norm(diff_error)
        # print(f'error g_q: {error}')
        # return g_q_num

    # TODO:
    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g)

        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        Omega21 = self.Omega1(t, q, u) - self.Omega2(t, q, u)

        # np.concatenate([r_OP2 - r_OP1, [ex1 @ ey2, ey1 @ ez2, ez1 @ ex2]])
        g_dot[:3] = self.v_P2(t, q, u) - self.v_P1(t, q, u)
        g_dot[3] = cross3(ex1, ey2) @ Omega21
        g_dot[4] = cross3(ey1, ez2) @ Omega21
        g_dot[5] = cross3(ez1, ex2) @ Omega21
        return g_dot

        # g_dot_num = Numerical_derivative(self.g, order=2)._t(t, q)
        # # g_dot_num += self.g_q_dense(t, q) @ u
        # g_dot_num += Numerical_derivative(self.g, order=2)._x(t, q) @ u
        # # diff = g_dot_num - g_dot
        # # diff_error = diff[3:]
        # # error = np.linalg.norm(diff_error)
        # # print(f'error g_dot: {error}')
        # return g_dot_num

    # TODO:
    def g_dot_q(self, t, q, u, coo):
        dense = Numerical_derivative(self.g_dot, order=2)._x(t, q, u)
        coo.extend(dense, (self.la_gDOF, self.qDOF))

    def g_dot_u(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q).T, (self.la_gDOF, self.uDOF))

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g)

        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        Omega21 = Omega1 - Omega2
        Psi21 = self.Psi1(t, q, u, u_dot) - self.Psi2(t, q, u, u_dot)

        #g_dot[:3] = self.v_P2(t, q, u) - self.v_P1(t, q, u)
        g_ddot[:3] = self.a_P2(t, q, u, u_dot) - self.a_P1(t, q, u, u_dot) 

        #g_dot[3] = cross3(ex1, ey2) @ Omega21
        g_ddot[3] =   cross3(cross3(Omega1, ex1), ey2) @ Omega21 \
                    + cross3(ex1, cross3(Omega2, ey2)) @ Omega21 \
                    + cross3(ex1, ey2) @ Psi21

        #g_dot[4] = cross3(ey1, ez2) @ Omega21
        g_ddot[4] =   cross3(cross3(Omega1, ey1), ez2) @ Omega21 \
                    + cross3(ey1, cross3(Omega2, ez2)) @ Omega21 \
                    + cross3(ey1, ez2) @ Psi21

        #g_dot[5] = cross3(ez1, ex2) @ Omega21
        g_ddot[5] =   cross3(cross3(Omega1, ez1), ex2) @ Omega21 \
                    + cross3(ez1, cross3(Omega2, ex2)) @ Omega21 \
                    + cross3(ez1, ex2) @ Psi21

        return g_ddot

        # g_ddot_num = Numerical_derivative(self.g_dot)._t(t, q, u)
        # g_ddot_num += Numerical_derivative(self.g_dot)._x(t, q, u) @ u
        # g_ddot_num += Numerical_derivative(self.g_dot)._y(t, q, u) @ u_dot
        # diff = g_ddot_num - g_ddot
        # error = np.linalg.norm(diff)
        # print(f'error g_ddot: {error}')
        # return g_ddot_num

    # TODO:
    def g_ddot_q(self, t, q, u, u_dot, coo):
        dense = Numerical_derivative(lambda t, q, u: self.g_ddot(t, q, u, u_dot), order=2)._x(t, q, u)
        coo.extend(dense, (self.la_gDOF, self.qDOF))

    # TODO:
    def g_ddot_u(self, t, q, u, u_dot, coo):
        dense = Numerical_derivative(lambda t, q, u: self.g_ddot(t, q, u, u_dot), order=2)._y(t, q, u)
        coo.extend(dense, (self.la_gDOF, self.uDOF))

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))
   
    def W_g_dense(self, t, q):
        nu1 = self.nu1
        W_g = np.zeros((self._nu, self.nla_g))

        # position 
        J_P1 = self.J_P1(t, q) 
        J_P2 = self.J_P2(t, q)
        W_g[:nu1, :3] = -J_P1.T
        W_g[nu1:, :3] = J_P2.T

        # orientations
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T
        J = np.hstack([self.J_R1(t, q), -self.J_R2(t, q)])

        W_g[:, 3] = cross3(ex1, ey2) @ J
        W_g[:, 4] = cross3(ey1, ez2) @ J
        W_g[:, 5] = cross3(ez1, ex2) @ J
        return W_g

        # W_g_num = Numerical_derivative(self.g_dot, order=2)._y(t, q, np.zeros(self._nu)).T
        # diff = W_g_num - W_g
        # diff_error = diff
        # error = np.linalg.norm(diff_error)
        # print(f'error W_g: {error}')
        # return W_g_num

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        nq1 = self.nq1
        nu1 = self.nu1
        dense = np.zeros((self._nu, self._nq))

        # position 
        J_P1_q = self.J_P1_q(t, q) 
        J_P2_q = self.J_P2_q(t, q)
        dense[:nu1, :nq1] += np.einsum('i,ijk->jk', -la_g[:3], J_P1_q)
        dense[nu1:, nq1:] += np.einsum('i,ijk->jk', la_g[:3], J_P2_q)

        # angular velocity
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T
        A_IB1_q = self.A_IB1_q(t, q)
        ex1_q = A_IB1_q[:, 0]
        ey1_q = A_IB1_q[:, 1]
        ez1_q = A_IB1_q[:, 2]
        A_IB2_q = self.A_IB2_q(t, q)
        ex2_q = A_IB2_q[:, 0]
        ey2_q = A_IB2_q[:, 1]
        ez2_q = A_IB2_q[:, 2]
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        J_R1_q = self.J_R1_q(t, q)
        J_R2_q = self.J_R2_q(t, q)

        # W_g[:nu1, 3] la_g[3]= cross3(ex1, ey2) @ J_R1 * la_g[3]
        # W_g[nu1:, 3] la_g[3]= - cross3(ex1, ey2) @ J_R2 * la_g[3]
        n = cross3(ex1, ey2)
        n_q1 = -ax2skew(ey2) @ ex1_q
        n_q2 = ax2skew(ex1) @ ey2_q
        dense[:nu1, :nq1] += np.einsum('i,ijk->jk', la_g[3] * n, J_R1_q) \
                                + np.einsum('ij,ik->kj', la_g[3] * n_q1, J_R1)
        dense[:nu1, nq1:] += np.einsum('ij,ik->kj', la_g[3] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum('ij,ik->kj', - la_g[3] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum('i,ijk->jk', - la_g[3] * n, J_R2_q) \
                                + np.einsum('ij,ik->kj', - la_g[3] * n_q2, J_R2)
        
        # W_g[:nu1, 4] la_g[4]= cross3(ey1, ez2) @ J_R1 * la_g[4]
        # W_g[nu1:, 4] la_g[4]= - cross3(ey1, ez2) @ J_R2 * la_g[4]
        n = cross3(ey1, ez2)
        n_q1 = -ax2skew(ez2) @ ey1_q
        n_q2 = ax2skew(ey1) @ ez2_q
        dense[:nu1, :nq1] += np.einsum('i,ijk->jk', la_g[4] * n, J_R1_q) \
                                + np.einsum('ij,ik->kj', la_g[4] * n_q1, J_R1)
        dense[:nu1, nq1:] += np.einsum('ij,ik->kj', la_g[4] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum('ij,ik->kj', - la_g[4] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum('i,ijk->jk', - la_g[4] * n, J_R2_q) \
                                + np.einsum('ij,ik->kj', - la_g[4] * n_q2, J_R2)
        
        # W_g[:nu1, 5] la_g[5]= cross3(ez1, ex2) @ J_R1 * la_g[5]
        # W_g[nu1:, 5] la_g[5]= - cross3(ez1, ex2) @ J_R2 * la_g[5]
        n = cross3(ez1, ex2)
        n_q1 = -ax2skew(ex2) @ ez1_q
        n_q2 = ax2skew(ez1) @ ex2_q
        dense[:nu1, :nq1] += np.einsum('i,ijk->jk', la_g[5] * n, J_R1_q) \
                                + np.einsum('ij,ik->kj', la_g[5] * n_q1, J_R1)
        dense[:nu1, nq1:] += np.einsum('ij,ik->kj', la_g[5] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum('ij,ik->kj', - la_g[5] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum('i,ijk->jk', - la_g[5] * n, J_R2_q) \
                                + np.einsum('ij,ik->kj', - la_g[5] * n_q2, J_R2)

        coo.extend( dense, (self.uDOF, self.qDOF))

        # W_g_q = Numerical_derivative(self.W_g_dense, order=2)._x(t, q)
        # dense_num = np.einsum('ijk,j->ik', W_g_q, la_g)
        # diff = dense_num - dense
        # diff_error = diff
        # error = np.linalg.norm(diff_error)
        # print(f'error Wla_g_q: {error}')
        # coo.extend(dense_num, (self.uDOF, self.qDOF))

class Rigid_connection2D(Rigid_connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nla_g = 3
        la_g0 = kwargs.get('la_g0')
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

    def g(self, t, q):
        r_OP1 = self.r_OP1(t, q)[:2]
        r_OP2 = self.r_OP2(t, q)[:2]
        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, _ = self.A_IB2(t, q).T
        return np.concatenate([r_OP2 - r_OP1, [ex1 @ ey2]]) 
        
    def g_q_dense(self, t, q):
        nq1 = self.nq1
        g_q = np.zeros((self.nla_g, self._nq))
        g_q[:2, :nq1] = - self.r_OP1_q(t, q)[:2]
        g_q[:2, nq1:] = self.r_OP2_q(t, q)[:2]

        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, _ = self.A_IB2(t, q).T

        A_IB1_q = self.A_IB1_q(t, q)
        ex1_q = A_IB1_q[:, 0]
        A_IB2_q = self.A_IB2_q(t, q)
        ey2_q = A_IB2_q[:, 1]

        g_q[2, :nq1] = ey2 @ ex1_q
        g_q[2, nq1:] = ex1 @ ey2_q
        return g_q

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g)

        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, _ = self.A_IB2(t, q).T

        # np.concatenate([r_OP2 - r_OP1, [ex1 @ ey2]])
        g_dot[:2] = self.v_P2(t, q, u)[:2] - self.v_P1(t, q, u)[:2]
        g_dot[2] = cross3(ex1, ey2) @ (self.Omega1(t, q, u) - self.Omega2(t, q, u))
        return g_dot

    def g_dot_q(self, t, q, u, coo):
        dense = Numerical_derivative(self.g_dot, order=2)._x(t, q, u)
        coo.extend(dense, (self.la_gDOF, self.qDOF))

    def g_dot_u(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q).T, (self.la_gDOF, self.uDOF))

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g)

        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, _ = self.A_IB2(t, q).T

        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        Omega21 = Omega1 - Omega2

        #g_dot[:2] = self.v_P2(t, q, u)[:2] - self.v_P1(t, q, u)[:2]
        g_ddot[:2] = self.a_P2(t, q, u, u_dot)[:2] - self.a_P1(t, q, u, u_dot)[:2]

        #g_dot[2] = cross3(ex1, ey2) @ Omega21
        g_ddot[2] =   cross3(cross3(Omega1, ex1), ey2) @ Omega21 \
                    + cross3(ex1, cross3(Omega2, ey2)) @ Omega21 \
                    + cross3(ex1, ey2) @ (self.Psi1(t, q, u, u_dot) - self.Psi2(t, q, u, u_dot))

        return g_ddot

    def g_ddot_q(self, t, q, u, u_dot, coo):
        dense = Numerical_derivative(lambda t, x: self.g_ddot(t, x, u, u_dot), order=2)._x(t, q)
        coo.extend(dense, (self.la_gDOF, self.qDOF))

    def g_ddot_u(self, t, q, u, u_dot, coo):
        dense = Numerical_derivative(lambda t, x: self.g_ddot(t, q, x, u_dot), order=2)._x(t, u)
        coo.extend(dense, (self.la_gDOF, self.uDOF))

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g_dense(self, t, q):
        nu1 = self.nu1
        W_g = np.zeros((self._nu, self.nla_g))

        # position 
        J_P1 = self.J_P1(t, q) 
        J_P2 = self.J_P2(t, q)
        W_g[:nu1, :2] = -J_P1[:2].T
        W_g[nu1:, :2] = J_P2[:2].T

        # angular velocity
        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, _ = self.A_IB2(t, q).T
        
        J = np.hstack([self.J_R1(t, q), -self.J_R2(t, q)])

        W_g[:, 2] = cross3(ex1, ey2) @ J
        return W_g

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        nq1 = self.nq1
        nu1 = self.nu1
        dense = np.zeros((self._nu, self._nq))

        # position 
        J_P1_q = self.J_P1_q(t, q) 
        J_P2_q = self.J_P2_q(t, q)
        dense[:nu1, :nq1] += np.einsum('i,ijk->jk', -la_g[:2], J_P1_q[:2])
        dense[nu1:, nq1:] += np.einsum('i,ijk->jk', la_g[:2], J_P2_q[:2])

        # angular velocity
        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, _ = self.A_IB2(t, q).T
        A_IB1_q = self.A_IB1_q(t, q)
        ex1_q = A_IB1_q[:, 0]
        A_IB2_q = self.A_IB2_q(t, q)
        ey2_q = A_IB2_q[:, 1]
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        J_R1_q = self.J_R1_q(t, q)
        J_R2_q = self.J_R2_q(t, q)

        n = cross3(ex1, ey2)
        n_q1 = -ax2skew(ey2) @ ex1_q
        n_q2 = ax2skew(ex1) @ ey2_q
        dense[:nu1, :nq1] += np.einsum('i,ijk->jk', la_g[2] * n, J_R1_q) \
                                + np.einsum('ij,ik->kj', la_g[2] * n_q1, J_R1)
        dense[:nu1, nq1:] += np.einsum('ij,ik->kj', la_g[2] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum('ij,ik->kj', - la_g[2] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum('i,ijk->jk', - la_g[2] * n, J_R2_q) \
                                + np.einsum('ij,ik->kj', - la_g[2] * n_q2, J_R2)

        coo.extend( dense, (self.uDOF, self.qDOF))