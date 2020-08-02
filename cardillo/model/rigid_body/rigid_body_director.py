import numpy as np
from cardillo.math.numerical_derivative import Numerical_derivative
from cardillo.math.algebra import ax2skew, cross3, norm3

# TODO: enable construction with standard inertia tensor
class Rigid_body_director():
    def __init__(self, V_rho, B_rho0, C_rho0, q0=None, u0=None, la0=None):
        self.nq = 12
        self.nu = 12
        self.nla_g = 6
        self.V_rho = V_rho
        self.B_rho0 = B_rho0
        self.C_rho0 = C_rho0

        if q0 is None:
            e1 = np.array([1, 0, 0])
            e2 = np.array([0, 1, 0])
            e3 = np.array([0, 0, 1])
            self.q0 = np.hstack((np.zeros(3), e1, e2, e3))
        else:
            self.q0 = q0

        self.u0 = np.zeros(self.nu) if u0 is None else u0
        self.la_g0 = np.zeros(self.nla_g) if la0 is None else la0

        self.M_ = np.zeros((self.nu, self.nu))
        self.M_[:3, :3] = np.eye(3) * self.V_rho
        self.M_[:3, 3:6] = np.eye(3) * self.B_rho0[0]
        self.M_[:3, 6:9] = np.eye(3) * self.B_rho0[1]
        self.M_[:3, 9:12] = np.eye(3) * self.B_rho0[2]
        
        self.M_[3:6, :3] = np.eye(3) * self.B_rho0[0]
        self.M_[3:6, 3:6] = np.eye(3) * self.C_rho0[0, 0]
        self.M_[3:6, 6:9] = np.eye(3) * self.C_rho0[0, 1]
        self.M_[3:6, 9:12] = np.eye(3) * self.C_rho0[0, 2]
        
        self.M_[6:9, :3] = np.eye(3) * self.B_rho0[1]
        self.M_[6:9, 3:6] = np.eye(3) * self.C_rho0[1, 0]
        self.M_[6:9, 6:9] = np.eye(3) * self.C_rho0[1, 1]
        self.M_[6:9, 9:12] = np.eye(3) * self.C_rho0[1, 2]

        self.M_[9:12, :3] = np.eye(3) * self.B_rho0[2]
        self.M_[9:12, 3:6] = np.eye(3) * self.C_rho0[2, 0]
        self.M_[9:12, 6:9] = np.eye(3) * self.C_rho0[2, 1]
        self.M_[9:12, 9:12] = np.eye(3) * self.C_rho0[2, 2]

    #########################################
    # equations of motion
    #########################################
    def M(self, t, q, coo):
        coo.extend(self.M_, (self.uDOF, self.uDOF))

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        return u

    def B(self, t, q, coo):
        coo.extend_diag(np.ones(self.nq), (self.qDOF, self.uDOF))

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    #########################################
    # bilateral constraints on position level
    #########################################
    def g(self, t, q):
        d1 = q[3:6]
        d2 = q[6:9]
        d3 = q[9:]

        gap = np.zeros(self.nla_g)
        gap[0] = d1 @ d1 - 1
        gap[1] = d2 @ d2 - 1
        gap[2] = d3 @ d3 - 1
        gap[3] = d1 @ d2
        gap[4] = d1 @ d3
        gap[5] = d2 @ d3

        return gap

    def g_dot(self, t, q, u):
        d1 = q[3:6]
        d2 = q[6:9]
        d3 = q[9:]
        d1_dot = u[3:6]
        d2_dot = u[6:9]
        d3_dot = u[9:]

        g_dot = np.zeros(self.nla_g)
        g_dot[0] = 2 * d1_dot @ d1
        g_dot[1] = 2 * d2_dot @ d2
        g_dot[2] = 2 * d3_dot @ d3
        g_dot[3] = d1_dot @ d2 + d1 @ d2_dot
        g_dot[4] = d1_dot @ d3 + d1 @ d3_dot
        g_dot[5] = d2_dot @ d3 + d2 @ d3_dot
        
        return g_dot

    def g_dot_u(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.uDOF))

    def g_ddot(self, t, q, u, u_dot):
        d1 = q[3:6]
        d2 = q[6:9]
        d3 = q[9:]
        d1_dot = u[3:6]
        d2_dot = u[6:9]
        d3_dot = u[9:]
        d1_ddot = u_dot[3:6]
        d2_ddot = u_dot[6:9]
        d3_ddot = u_dot[9:]

        g_ddot = np.zeros(self.nla_g)
        g_ddot[0] = 2 * (d1_ddot @ d1 + d1_dot @ d1_dot)
        g_ddot[1] = 2 * (d2_ddot @ d2 + d2_dot @ d2_dot)
        g_ddot[2] = 2 * (d3_ddot @ d3 + d3_dot @ d3_dot)
        g_ddot[3] = d1_ddot @ d2 + d1 @ d2_ddot + 2 * d1_dot @ d2_dot
        g_ddot[4] = d1_ddot @ d3 + d1 @ d3_ddot + 2 * d1_dot @ d3_dot
        g_ddot[5] = d2_ddot @ d3 + d2 @ d3_ddot + 2 * d2_dot @ d3_dot    
        return g_ddot

    def g_q_dense(self, t, q):

        d1 = q[3:6]
        d2 = q[6:9]
        d3 = q[9:]

        gap_q = np.zeros((self.nla_g, self.nq))
        gap_q[0, 3:6] = 2 * d1
        gap_q[1, 6:9] = 2 * d2
        gap_q[2, 9:12] = 2 * d3

        gap_q[3, 3:6] = d2
        gap_q[3, 6:9] = d1

        gap_q[4, 3:6] = d3
        gap_q[4, 9:12] = d1

        gap_q[5, 6:9] = d3
        gap_q[5, 9:12] = d2

        # gap_q_num = NumericalDerivativeNew(self.gap_dense, order=2).dR_dq(t, q)
        # diff = gap_q - gap_q_num
        # np.set_printoptions(precision=3)
        # error = np.linalg.norm(diff)
        # print(f'error num_tan - tan = {error}')
        # return gap_q_num

        return gap_q
    
    def g_qq_dense(self, t, q):

        gap_qq = np.zeros((self.nla_g, self.nq, self.nq))
        gap_qq[0, 3:6, 3:6] = 2 * np.eye(3)
        gap_qq[1, 6:9, 6:9] = 2 * np.eye(3)
        gap_qq[2, 9:12, 9:12] = 2 * np.eye(3)
        
        gap_qq[3, 3:6, 6:9] = np.eye(3)
        gap_qq[3, 6:9, 3:6] = np.eye(3)
        
        gap_qq[4, 3:6, 9:12] = np.eye(3)
        gap_qq[4, 9:12, 3:6] = np.eye(3)
        
        gap_qq[5, 6:9, 9:12] = np.eye(3)
        gap_qq[5, 9:12, 6:9] = np.eye(3)

        # gap_qq_num = NumericalDerivativeNew(self.gap_q_dense, order=2).dR_dq(t, q)
        # diff = gap_qq - gap_qq_num
        # np.set_printoptions(precision=3)
        # error = np.linalg.norm(diff)
        # print(f'error num_tan - tan = {error}')
        # return gap_qq_num

        return gap_qq

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))
   
    def W_g(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q).T, (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        dense = np.einsum('ijk,i->jk', self.g_qq_dense(t, q), la_g)
        coo.extend(dense, (self.uDOF, self.qDOF))

    #########################################
    # helper functions
    #########################################
    def qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)

    def A_IK(self, t, q, frame_ID=None):
        return np.vstack((q[3:6], q[6:9], q[9:12])).T

    def A_IK_q(self, t, q, frame_ID=None):
        A_IK_q = np.zeros((3, 3, self.nq))
        A_IK_q[:, 0, 3:6] = np.eye(3)
        A_IK_q[:, 1, 6:9] = np.eye(3)
        A_IK_q[:, 2, 9:12] = np.eye(3)
        return A_IK_q

    def r_OP(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return q[:3] + self.A_IK(t, q) @ K_r_SP

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        r_OP_q = np.zeros((3, self.nq))
        r_OP_q[:, :3] = np.eye(3)
        r_OP_q[:, :] += np.einsum('ijk,j->ik', self.A_IK_q(t, q), K_r_SP)
        return r_OP_q

        # r_OP_q_num = Numerical_derivative(lambda t, q: self.r_OP(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP))._x(t, q)
        # diff = r_OP_q_num - r_OP_q
        # error = np.max(np.abs(diff))
        # print(f'error in r_OP_q = {error}') 
        # return r_OP_q_num

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.r_OP_q(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP) @ u

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.r_OP_q(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP) @ u_dot

    def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.r_OP_q(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP)

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.zeros((3, self.nu, self.nq))

    # def K_J_R(self, t, q, frame_ID=None):
    #     J_R = np.zeros((3, self.nu))
    #     J_R[:, 3:] = np.eye(3)
    #     return J_R

    # def K_J_R_q(self, t, q, frame_ID=None):
    #     return np.zeros((3, self.nu, self.nq))

class Rigid_body_director_angular_velocities():
    def __init__(self, V_rho, B_rho0, C_rho0, q0=None, u0=None, la0=None):
        self.nq = 12
        self.nu = 6
        self.V_rho = V_rho
        self.B_rho0 = B_rho0
        self.C_rho0 = C_rho0

        if q0 is None:
            e1 = np.array([1, 0, 0])
            e2 = np.array([0, 1, 0])
            e3 = np.array([0, 0, 1])
            self.q0 = np.hstack((np.zeros(3), e1, e2, e3))
        else:
            self.q0 = q0

        self.u0 = np.zeros(self.nu) if u0 is None else u0

    #########################################
    # equations of motion
    #########################################
    def M(self, t, q, coo):
        d1 = q[3:6]
        d2 = q[6:9]
        d3 = q[9:]
        d1_tilde = ax2skew(d1)
        d2_tilde = ax2skew(d2)
        d3_tilde = ax2skew(d3)

        off_diag = -self.B_rho0[0] * d1_tilde - self.B_rho0[1] * d2_tilde - self.B_rho0[2] * d3_tilde

        M = np.zeros((self.nu, self.nu))
        M[:3, :3] = np.eye(3) * self.V_rho
        M[:3, 3:6] = off_diag
        M[3:6, :3] = -off_diag
        M[3:6, 3:6] = - self.C_rho0[0, 0] * d1_tilde.T @ d1_tilde - self.C_rho0[0, 1] * d1_tilde.T @ d2_tilde - self.C_rho0[0, 2] * d1_tilde.T @ d3_tilde
        M[3:6, 3:6] += -self.C_rho0[1, 0] * d2_tilde.T @ d1_tilde - self.C_rho0[1, 1] * d2_tilde.T @ d2_tilde - self.C_rho0[1, 2] * d2_tilde.T @ d3_tilde
        M[3:6, 3:6] += -self.C_rho0[2, 0] * d3_tilde.T @ d1_tilde - self.C_rho0[2, 1] * d3_tilde.T @ d2_tilde - self.C_rho0[2, 2] * d3_tilde.T @ d3_tilde

        # M[:3, 3:6] = -M[:3, 3:6]
        # M[3:6, :3] = -M[3:6, :3]
        M[3:6, 3:6] = -M[3:6, 3:6]

        coo.extend(M, (self.uDOF, self.uDOF))

    def f_gyr(self, t, q, u):
        d1 = q[3:6]
        d2 = q[6:9]
        d3 = q[9:]
        d1_tilde = ax2skew(d1)
        d2_tilde = ax2skew(d2)
        d3_tilde = ax2skew(d3)

        omega = u[3:]
        omega_tilde = ax2skew(omega)
        omega_tilde2 = omega_tilde @ omega_tilde

        f = np.zeros(self.nu)
        f[:3] = self.B_rho0[0] * omega_tilde2 @ d1 + self.B_rho0[1] * omega_tilde2 @ d2 + self.B_rho0[2] * omega_tilde2 @ d3
        f[3:] = -  self.C_rho0[0, 0] * d1_tilde.T @ omega_tilde2 @ d1 - self.C_rho0[0, 1] * d1_tilde.T @ omega_tilde2 @ d2 - self.C_rho0[0, 2] * d1_tilde.T @ omega_tilde2 @ d3
        f[3:] += - self.C_rho0[1, 0] * d2_tilde.T @ omega_tilde2 @ d1 - self.C_rho0[1, 1] * d2_tilde.T @ omega_tilde2 @ d2 - self.C_rho0[1, 2] * d2_tilde.T @ omega_tilde2 @ d3
        f[3:] += - self.C_rho0[2, 0] * d3_tilde.T @ omega_tilde2 @ d1 - self.C_rho0[2, 1] * d3_tilde.T @ omega_tilde2 @ d2 - self.C_rho0[2, 2] * d3_tilde.T @ omega_tilde2 @ d3
        # f[:3] = - f[:3]
        # f[3:] = - f[3:]
        return f

    # TODO:
    def f_gyr_u(self, t, q, u, coo):
        dense = Numerical_derivative(self.f_gyr)._y(t, q, u)
        coo.extend(dense, (self.uDOF, self.uDOF))

    #########################################
    # kinematic equation
    #########################################
    # TODO:
    def q_dot(self, t, q, u):
        return self.__B_dense(q) @ u

    def q_dot_q(self, t, q, u, coo):
        dense = Numerical_derivative(self.q_dot, order=2)._x(t, q, u)
        coo.extend(dense, (self.qDOF, self.qDOF))

    def __B_dense(self, q):
        d1 = q[3:6]
        d2 = q[6:9]
        d3 = q[9:]

        B = np.zeros((self.nq, self.nu))
        B[:3, :3] = np.eye(3)
        B[3:6, 3:] = -ax2skew(d1)
        B[6:9, 3:] = -ax2skew(d2)
        B[9:12, 3:] = -ax2skew(d3)

        return B

    def B(self, t, q, coo):
        coo.extend(self.__B_dense(q), (self.qDOF, self.uDOF))

    def q_ddot(self, t, q, u, u_dot):
        raise RuntimeError('not tested!')
        d1 = q[3:6]
        d2 = q[6:9]
        d3 = q[9:]

        omega = u[3:]
        omega_tilde = ax2skew(omega)
        omega_tilde2 = omega_tilde @ omega_tilde
        omega_dot = u_dot[3:]

        q_ddot = np.zeros(self.nq)
        q_ddot[:3] = u_dot[:3]
        q_ddot[3:6] = omega_tilde2 @ d1 - ax2skew(d1) @ omega_dot
        q_ddot[6:9] = omega_tilde2 @ d2 - ax2skew(d2) @ omega_dot
        q_ddot[9:12] = omega_tilde2 @ d3 - ax2skew(d3) @ omega_dot

        return q_ddot

    # def solver_step_callback(self, t, q, u):
    #     q[3:6] = q[3:6] / norm3(q[3:6])
    #     q[6:9] = q[6:9] / norm3(q[6:9])
    #     q[9:12] = q[9:12] / norm3(q[9:12])
    #     return q, u

    #########################################
    # helper functions
    #########################################
    def qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)

    def A_IK(self, t, q, frame_ID=None):
        return np.vstack((q[3:6], q[6:9], q[9:12])).T

    def A_IK_q(self, t, q, frame_ID=None):
        A_IK_q = np.zeros((3, 3, self.nq))
        A_IK_q[:, 0, 3:6] = np.eye(3)
        A_IK_q[:, 1, 6:9] = np.eye(3)
        A_IK_q[:, 2, 9:12] = np.eye(3)
        return A_IK_q

    def r_OP(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return q[:3] + self.A_IK(t, q) @ K_r_SP

    def r_OP_t(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.zeros(3)

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        r_OP_q = np.zeros((3, self.nq))
        r_OP_q[:, :3] = np.eye(3)
        r_OP_q[:, :] += np.einsum('ijk,j->ik', self.A_IK_q(t, q), K_r_SP)
        return r_OP_q

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return u[:3] + self.A_IK(t, q) @ cross3(u[3:], K_r_SP)

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return u_dot[:3] + self.A_IK(t, q) @ (cross3(u_dot[3:], K_r_SP) + cross3(u[3:], cross3(u[3:], K_r_SP)))
    
    def kappa_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.A_IK(t, q) @ (cross3(u[3:], cross3(u[3:], K_r_SP)))
    
    def kappa_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.einsum('ijk,j->ik', self.A_IK_q(t, q), cross3(u[3:], cross3(u[3:], K_r_SP)) )
    
    def kappa_P_u(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        kappa_P_u = np.zeros((3, self.nu))
        kappa_P_u[:, 3:] = -self.A_IK(t, q) @ (ax2skew(cross3(u[3:], K_r_SP)) + ax2skew(u[3:]) @ ax2skew(K_r_SP))
        return kappa_P_u

    def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        J_P = np.zeros((3, self.nu))
        J_P[:, :3] = np.eye(3)
        J_P[:, 3:] = - self.A_IK(t, q) @ ax2skew(K_r_SP)
        return J_P

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        J_P_q = np.zeros((3, self.nu, self.nq))
        J_P_q[:, 3:, :] = np.einsum('ijk,jl->ilk', self.A_IK_q(t, q), -ax2skew(K_r_SP))
        return J_P_q

    def K_Omega(self, t, q, u, frame_ID=None):
        return u[3:]

    def K_Omega_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq))

    def K_Psi(self, t, q, u, u_dot, frame_ID=None):
        return u_dot[3:]

    def K_kappa_R(self, t, q, u, frame_ID=None):
        return np.zeros(3)

    def K_kappa_R_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq))

    def K_kappa_R_u(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nu))

    def K_J_R(self, t, q, frame_ID=None):
        J_R = np.zeros((3, self.nu))
        J_R[:, 3:] = np.eye(3)
        return J_R

    def K_J_R_q(self, t, q, frame_ID=None):
        return np.zeros((3, self.nu, self.nq))