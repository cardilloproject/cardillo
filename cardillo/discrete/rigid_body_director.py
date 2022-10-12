import numpy as np
from cardillo.math.numerical_derivative import approx_fprime
from cardillo.math.algebra import ax2skew, ax2skew_a, cross3, skew2ax

# TODO: enable construction with standard inertia tensor
class RigidBodyDirector:
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

    def g_dot_q(self, t, q, u, coo):
        d1_dot = u[3:6]
        d2_dot = u[6:9]
        d3_dot = u[9:]

        g_dot_q = np.zeros((self.nla_g, self.nq))
        g_dot_q[0, 3:6] = 2 * d1_dot
        g_dot_q[1, 6:9] = 2 * d2_dot
        g_dot_q[2, 9:12] = 2 * d3_dot

        # g_dot[3] = d1_dot @ d2 + d1 @ d2_dot
        g_dot_q[3, 6:9] = d1_dot
        g_dot_q[3, 3:6] = d2_dot

        # g_dot[4] = d1_dot @ d3 + d1 @ d3_dot
        g_dot_q[4, 9:12] = d1_dot
        g_dot_q[4, 3:6] = d3_dot

        # g_dot[5] = d2_dot @ d3 + d2 @ d3_dot
        g_dot_q[5, 9:12] = d2_dot
        g_dot_q[5, 6:9] = d3_dot

        coo.extend(g_dot_q, (self.la_gDOF, self.qDOF))

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

    def g_ddot_q(self, t, q, u, u_dot, coo):
        d1_ddot = u_dot[3:6]
        d2_ddot = u_dot[6:9]
        d3_ddot = u_dot[9:]

        g_ddot_q = np.zeros((self.nla_g, self.nq))
        # g_ddot[0] = 2 * (d1_ddot @ d1 + d1_dot @ d1_dot)
        g_ddot_q[0, 3:6] = 2 * d1_ddot

        # g_ddot[1] = 2 * (d2_ddot @ d2 + d2_dot @ d2_dot)
        g_ddot_q[1, 6:9] = 2 * d2_ddot

        # g_ddot[2] = 2 * (d3_ddot @ d3 + d3_dot @ d3_dot)
        g_ddot_q[2, 9:12] = 2 * d3_ddot

        # g_ddot[3] = d1_ddot @ d2 + d1 @ d2_ddot + 2 * d1_dot @ d2_dot
        g_ddot_q[3, 6:9] = d1_ddot
        g_ddot_q[3, 3:6] = d2_ddot

        # g_ddot[4] = d1_ddot @ d3 + d1 @ d3_ddot + 2 * d1_dot @ d3_dot
        g_ddot_q[4, 9:12] = d1_ddot
        g_ddot_q[4, 3:6] = d3_ddot

        # g_ddot[5] = d2_ddot @ d3 + d2 @ d3_ddot + 2 * d2_dot @ d3_dot
        g_ddot_q[5, 9:12] = d2_ddot
        g_ddot_q[5, 6:9] = d3_ddot

        coo.extend(g_ddot_q, (self.la_gDOF, self.qDOF))

    def g_ddot_u(self, t, q, u, u_dot, coo):
        d1_dot = u[3:6]
        d2_dot = u[6:9]
        d3_dot = u[9:]

        g_ddot_u = np.zeros((self.nla_g, self.nq))
        # g_ddot[0] = 2 * (d1_ddot @ d1 + d1_dot @ d1_dot)
        g_ddot_u[0, 3:6] = 4 * d1_dot

        # # g_ddot[1] = 2 * (d2_ddot @ d2 + d2_dot @ d2_dot)
        g_ddot_u[1, 6:9] = 4 * d2_dot

        # # g_ddot[2] = 2 * (d3_ddot @ d3 + d3_dot @ d3_dot)
        g_ddot_u[2, 9:12] = 4 * d3_dot

        # # g_ddot[3] = d1_ddot @ d2 + d1 @ d2_ddot + 2 * d1_dot @ d2_dot
        g_ddot_u[3, 6:9] = 2 * d1_dot
        g_ddot_u[3, 3:6] = 2 * d2_dot

        # # g_ddot[4] = d1_ddot @ d3 + d1 @ d3_ddot + 2 * d1_dot @ d3_dot
        g_ddot_u[4, 9:12] = 2 * d1_dot
        g_ddot_u[4, 3:6] = 2 * d3_dot

        # # g_ddot[5] = d2_ddot @ d3 + d2 @ d3_ddot + 2 * d2_dot @ d3_dot
        g_ddot_u[5, 9:12] = 2 * d2_dot
        g_ddot_u[5, 6:9] = 2 * d3_dot

        coo.extend(g_ddot_u, (self.la_gDOF, self.uDOF))

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
        dense = np.einsum("ijk,i->jk", self.g_qq_dense(t, q), la_g)
        coo.extend(dense, (self.uDOF, self.qDOF))

    #########################################
    # helper functions
    #########################################
    def local_qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def local_uDOF_P(self, frame_ID=None):
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
        r_OP_q += np.einsum("ijk,j->ik", self.A_IK_q(t, q), K_r_SP)
        return r_OP_q

    def r_OP_qq(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.zeros((3, self.nq, self.nq))

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        # v_P1 = u[:3] + self.A_IK(t, q) @ cross3(self.K_Omega(t, q, u), K_r_SP)
        # v_P2 = u[:3] + self.A_IK(t, u) @ K_r_SP
        # print(v_P1 - v_P2)

        return u[:3] + self.A_IK(t, u) @ K_r_SP

    def v_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.zeros((3, self.nq))

    def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.r_OP_q(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP)

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.zeros((3, self.nu, self.nq))

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        # K_Omega = self.K_Omega(t, q, u, frame_ID=frame_ID)
        # K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID=frame_ID)
        # return u_dot[:3] + self.A_IK(t, q) @ (cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP))) # numerically unstable (A_IK is not orthogonal)

        return u_dot[:3] + self.A_IK(t, u_dot) @ K_r_SP

    def a_P_q(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.zeros((3, self.nq))

    def a_P_u(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.zeros((3, self.nu))

    def K_Omega(self, t, q, u, frame_ID=None):
        A_IK = self.A_IK(t, q)
        A_IK_dot = self.A_IK(t, u)
        K_Omega_tilde = A_IK.T @ A_IK_dot
        return skew2ax(K_Omega_tilde)

    def K_J_R(self, t, q, frame_ID=None):
        raise NotImplementedError("see director rigid body for implementation")
        A_IK = self.A_IK(t, q)
        K_J_R = np.zeros((3, self.nu))
        K_J_R[:, 3:6] = 0.5 * A_IK.T @ ax2skew(q[3:6])
        K_J_R[:, 6:9] = 0.5 * A_IK.T @ ax2skew(q[6:9])
        K_J_R[:, 9:12] = 0.5 * A_IK.T @ ax2skew(q[9:12])
        return K_J_R

    def K_J_R_q(self, t, q, frame_ID=None):
        raise NotImplementedError("see director rigid body for implementation")
        A_IK = self.A_IK(t, q)
        A_IK_q = self.A_IK_q(t, q)
        K_J_R_q = np.zeros((3, self.nu, self.nq))
        tmp = 0.5 * np.einsum("ij,jkl->ikl", A_IK.T, ax2skew_a())
        K_J_R_q[:, 3:6] = 0.5 * np.einsum("jil,jk->ikl", A_IK_q, ax2skew(q[3:6]))
        K_J_R_q[:, 3:6, 3:6] += tmp
        K_J_R_q[:, 6:9] = 0.5 * np.einsum("jil,jk->ikl", A_IK_q, ax2skew(q[6:9]))
        K_J_R_q[:, 6:9, 6:9] += tmp
        K_J_R_q[:, 9:12] = 0.5 * np.einsum("jil,jk->ikl", A_IK_q, ax2skew(q[9:12]))
        K_J_R_q[:, 9:12, 9:12] += tmp
        return K_J_R_q

    def K_Psi(self, t, q, u, u_dot, frame_ID=None):
        A_IK = self.A_IK(t, q)
        A_IK_dot = self.A_IK(t, u)
        A_IK_ddot = self.A_IK(t, u_dot)
        K_Psi_tilde = A_IK_dot.T @ A_IK_dot + A_IK.T @ A_IK_ddot
        return skew2ax(K_Psi_tilde)


class RigidBodyDirectorAngularVelocities:
    def __init__(self, m, K_Theta_S, q0=None, u0=None, nka_c0=None):
        self.m = m
        self.theta = K_Theta_S

        self.nq = 12
        self.nu = 6
        self.nla_S = 6

        self.M_ = np.zeros((self.nu, self.nu))
        self.M_[:3, :3] = m * np.eye(3)
        self.M_[3:, 3:] = self.theta

        if q0 is None:
            e1 = np.array([1, 0, 0])
            e2 = np.array([0, 1, 0])
            e3 = np.array([0, 0, 1])
            self.q0 = np.hstack((np.zeros(3), e1, e2, e3))
        else:
            self.q0 = q0

        self.u0 = np.zeros(self.nu) if u0 is None else u0
        self.ka_c0 = np.zeros(self.nka_c) if nka_c0 is None else nka_c0

    #########################################
    # equations of motion
    #########################################
    def M(self, t, q, coo):
        coo.extend(self.M_, (self.uDOF, self.uDOF))

    def h(self, t, q, u):
        omega = u[3:]
        h = np.zeros(self.nu)
        h[3:] = -cross3(omega, self.theta @ omega)
        return h

    def h_u(self, t, q, u, coo):
        omega = u[3:]
        dense = np.zeros((self.nu, self.nu))
        dense[3:, 3:] = ax2skew(self.theta @ omega) - ax2skew(omega) @ self.theta
        coo.extend(dense, (self.uDOF, self.uDOF))

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        return self.__B_dense(q) @ u

    def q_dot_q(self, t, q, u, coo):
        dense = approx_fprime(q, lambda q: self.q_dot(t, q, u))
        coo.extend(dense, (self.qDOF, self.qDOF))

    def __B_dense(self, q):
        d1 = q[3:6]
        d2 = q[6:9]
        d3 = q[9:]

        A_IK = self.A_IK(0, q)

        B = np.zeros((self.nq, self.nu))
        B[:3, :3] = np.eye(3)
        B[3:6, 3:] = -ax2skew(d1) @ A_IK
        B[6:9, 3:] = -ax2skew(d2) @ A_IK
        B[9:12, 3:] = -ax2skew(d3) @ A_IK

        return B

    def B(self, t, q, coo):
        coo.extend(self.__B_dense(q), (self.qDOF, self.uDOF))

    def q_ddot(self, t, q, u, u_dot):
        d1 = q[3:6]
        d2 = q[6:9]
        d3 = q[9:]
        # d1_tilde = ax2skew(d1)
        # d2_tilde = ax2skew(d2)
        # d3_tilde = ax2skew(d3)

        A_IK = self.A_IK(0, q)
        # A_IK_q = self.A_IK_q(0, q)

        K_Omega = u[3:]
        I_Omega = A_IK @ K_Omega
        I_Omega_tilde = ax2skew(I_Omega)
        I_Omega_tilde2 = I_Omega_tilde @ I_Omega_tilde
        # K_Omega_dot = u_dot[3:]

        # I_omega = A_IK @ K_Omega
        # tmp = (A_IK_q @ self.q_dot(t, q, u)) @ K_Omega

        q_ddot = self.__B_dense(q) @ u_dot
        q_ddot[3:6] += I_Omega_tilde2 @ d1
        q_ddot[6:9] += I_Omega_tilde2 @ d2
        q_ddot[9:12] += I_Omega_tilde2 @ d3
        # q_ddot[3:6] += cross3(I_Omega, cross3(I_Omega, d1))
        # q_ddot[6:9] += cross3(I_Omega, cross3(I_Omega, d2))
        # q_ddot[9:12] += cross3(I_Omega, cross3(I_Omega, d3))

        # q_ddot[3:6] = cross3(d1_tilde @ I_omega, I_omega) - d1_tilde @ (tmp + A_IK @ omega_dot)
        # q_ddot[6:9] = cross3(d2_tilde @ I_omega, I_omega) - d2_tilde @ (tmp + A_IK @ omega_dot)
        # q_ddot[9:12] = cross3(d3_tilde @ I_omega, I_omega) - d3_tilde @ (tmp + A_IK @ omega_dot)

        return q_ddot

    #########################################
    # stabilization conditions for the kinematic equation
    #########################################
    def g_S(self, t, q):
        d1 = q[3:6]
        d2 = q[6:9]
        d3 = q[9:]

        c = np.zeros(self.nka_c)
        c[0] = d1 @ d1 - 1
        c[1] = d2 @ d2 - 1
        c[2] = d3 @ d3 - 1
        c[3] = d1 @ d2
        c[4] = d1 @ d3
        c[5] = d2 @ d3

        return c

    def __g_S_q_dense(self, t, q):
        d1 = q[3:6]
        d2 = q[6:9]
        d3 = q[9:]

        c_q = np.zeros((self.nka_c, self.nq))
        c_q[0, 3:6] = 2 * d1
        c_q[1, 6:9] = 2 * d2
        c_q[2, 9:12] = 2 * d3

        c_q[3, 3:6] = d2
        c_q[3, 6:9] = d1

        c_q[4, 3:6] = d3
        c_q[4, 9:12] = d1

        c_q[5, 6:9] = d3
        c_q[5, 9:12] = d2
        return c_q

    def g_S_q(self, t, q, coo):
        coo.extend(self.__g_S_q_dense(t, q), (self.ka_cDOF, self.qDOF))

    #########################################
    # helper functions
    #########################################
    def local_qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def local_uDOF_P(self, frame_ID=None):
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
        r_OP_q[:, :] += np.einsum("ijk,j->ik", self.A_IK_q(t, q), K_r_SP)
        return r_OP_q

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return u[:3] + self.A_IK(t, q) @ cross3(u[3:], K_r_SP)

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return u_dot[:3] + self.A_IK(t, q) @ (
            cross3(u_dot[3:], K_r_SP) + cross3(u[3:], cross3(u[3:], K_r_SP))
        )

    def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        J_P = np.zeros((3, self.nu))
        J_P[:, :3] = np.eye(3)
        J_P[:, 3:] = -self.A_IK(t, q) @ ax2skew(K_r_SP)
        return J_P

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        J_P_q = np.zeros((3, self.nu, self.nq))
        J_P_q[:, 3:, :] = np.einsum("ijk,jl->ilk", self.A_IK_q(t, q), -ax2skew(K_r_SP))
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
