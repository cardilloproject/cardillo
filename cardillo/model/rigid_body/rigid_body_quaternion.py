import numpy as np
from cardillo.math.algebra import cross3, ax2skew, quat2mat, quat2mat_p, norm4, quat2rot, quat2rot_p

class Rigid_body_quaternion():
    """Rigid body parametrized by center of mass in inertial system and unit quaternions for rotation.

    Parameters
    ----------
    m : float
        mass
    K_theta_S : ndarray
        inertia tensor in body-fixed frame
    q0 : ndarray, optional
        initial position coordinates
    u0 : ndarray, optional
        initial velocity coordinates

    References
    ----------
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Schweizer2015: https://www.research-collection.ethz.ch/handle/20.500.11850/101867
    """
    def __init__(self, m, K_theta_S, q0=None, u0=None):
        self.m = m
        self.theta = K_theta_S

        self.nq = 7
        self.nu = 6
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

        self.M_ = np.zeros((self.nu, self.nu))
        self.M_[:3, :3] = m * np.eye(3)
        self.M_[3:, 3:] = self.theta

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    def M(self, t, q, coo):
        coo.extend(self.M_, (self.uDOF, self.uDOF))

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
        dense[3:, 3:] = np.einsum('ijk,j->ik', Q_p[:, 1:, :], u[3:])
        coo.extend(dense, (self.qDOF, self.qDOF))

    def B(self, t, q, coo):
        p = q[3:]
        Q = quat2mat(p) / (2 * p @ p)
        
        B = np.zeros((self.nq, self.nu))
        B[:3, :3] = np.eye(3)
        B[3:, 3:] = Q[:, 1:]
        coo.extend(B, (self.qDOF, self.uDOF))

    def q_ddot(self, t, q, u, u_dot):
        p = q[3:]
        p2 = p @ p
        Q = quat2mat(p) / (2 * p2)
        p_dot = Q[:, 1:] @ u[3:]
        Q_p = quat2mat_p(p) / (2 * p2) \
            - np.einsum('ij,k->ijk', quat2mat(p), p / (p2**2))
        
        q_ddot = np.zeros(self.nq)
        q_ddot[:3] = u_dot[:3]
        q_ddot[3:] = Q[:, 1:] @ u_dot[3:] + np.einsum('ijk,k,j->i', Q_p[:, 1:, :], p_dot, u[3:])

        return q_ddot

    def solver_step_callback(self, t, q, u):
        q[3:] = q[3:] / norm4(q[3:])
        return q, u

    def qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)

    def A_IK(self, t, q, frame_ID=None):
        return quat2rot(q[3:])

    def A_IK_q(self, t, q, frame_ID=None):
        A_IK_q = np.zeros((3, 3, self.nq))
        A_IK_q[:, :, 3:] = quat2rot_p(q[3:])
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

    def K_J_R(self, t, q, frame_ID=None):
        J_R = np.zeros((3, self.nu))
        J_R[:, 3:] = np.eye(3)
        return J_R

    def K_J_R_q(self, t, q, frame_ID=None):
        return np.zeros((3, self.nu, self.nq))