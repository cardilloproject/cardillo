import numpy as np
from cardillo.math.algebra import cross3, ax2skew, quat2mat, quat2mat_p, norm4, quat2rot, quat2rot_p

class Rigid_body_quaternion():
    def __init__(self, m, K_theta_S, q0=None, u0=None):
        self.m = m
        self.theta = K_theta_S

        self.nq = 7
        self.nu = 6

        self.M_ = np.zeros((self.nu, self.nu))
        self.M_[:3, :3] = m * np.eye(3)
        self.M_[3:, 3:] = self.theta

        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

    def M(self, t, q, M_coo):
        M_coo.extend(self.M_, (self.uDOF, self.uDOF))

    def f_gyr(self, t, q, u):
        omega = u[3:]
        f = np.zeros(self.nu)
        f[3:] = cross3(omega, self.theta @ omega)
        return f

    def f_gyr_u(self, t, q, u, coo):
        omega = u[3:]
        dense = np.zeros((self.nu, self.nu))
        dense[3:, 3:] = ax2skew(omega) @ self.theta - ax2skew(self.theta @ omega)
        coo.extend(dense, (self.uDOF, self.uDOF))

    def q_dot(self, t, q, u):
        p = q[3:]
        Q = quat2mat(p) / (2 * p @ p)
        
        q_dot = np.zeros(self.nq)
        q_dot[:3] = u[:3]
        q_dot[3:] = Q[:, 1:] @ u[3:]

        return q_dot

    def q_dot_q(self, t, q, u, coo):
        p = q[3:]
        p2 = p @ p
        Q_p = quat2mat_p(p) / (2 * p2) \
            - np.einsum('ij,k->ijk', quat2mat(p), p / (p2**2))
            
        dense = np.zeros((self.nq, self.nq))
        dense[:3, :3] = np.eye(3)
        dense[3:, 3:] = np.einsum('ijk,j->ik', Q_p[:, 1:, :], u[3:])
        coo.extend(dense, (self.qDOF, self.qDOF))

    def B_dense(self, t, q):
        p = q[3:]
        Q = quat2mat(p) / (2 * p @ p)
        
        B = np.zeros((self.nq, self.nu))
        B[:3, :3] = np.eye(3)
        B[3:, 3:] = Q[:, 1:]
        return B

    def B(self, t, q, coo):
        coo.extend(self.B_dense(t, q), (self.qDOF, self.uDOF))

    def solver_step_callback(self, t, q, u):
        q[3:] = q[3:] / norm4(q[3:])
        return q, u

    def qDOF_P(self, point_ID=None):
        return self.qDOF

    def uDOF_P(self, point_ID=None):
        return self.uDOF

    def A_IK(self, t, q, point_ID=None):
        return quat2rot(q[3:])

    def A_IK_q(self, t, q, point_ID=None):
        A_IK_q = np.zeros((3, 3, self.nq))
        A_IK_q[:, :, 3:] = quat2rot_p(q[3:])
        return A_IK_q

    def r_OP(self, t, q, point_ID=np.zeros(3)):
        return q[:3] + self.A_IK(t, q) @ point_ID

    def r_OP_q(self, t, q, point_ID=np.zeros(3)):
        r_OP_q = np.zeros((3, self.nq))
        r_OP_q[:, :3] = np.eye(3)
        r_OP_q[:, :] += np.einsum('ijk,j->ik', self.A_IK_q(t, q), point_ID)
        return r_OP_q

    def J_P(self, t, q, point_ID=np.zeros(3)):
        J_P = np.zeros((3, self.nu))
        J_P[:, :3] = np.eye(3)
        J_P[:, 3:] = -ax2skew(point_ID) @ self.A_IK(t, q)
        return J_P

    def J_P_q(self, t, q, point_ID=None):
        J_P_q = np.zeros((3, self.nu, self.nq))
        J_P_q[:, 3:, :] = np.einsum('ij,jkl->ikl', -ax2skew(point_ID), self.A_IK_q(t, q))
        return J_P_q



