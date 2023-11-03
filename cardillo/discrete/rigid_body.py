import numpy as np
from cardillo.math import (
    cross3,
    ax2skew,
    norm,
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
)


class RigidBody:
    """Rigid body parametrized by center of mass in inertial base and unit 
    quaternions for rotation. The angular velocities expressed in the 
    body-fixed base are used as minimal velocities.
    
    Exponential function and kinematic differential equation are found in 
    Egeland2002 (6.199), (6.329) and (6.330). The implementation below 
    handles non-unit quaternions. After each successfull time step they are 
    projected to be of unit length. Alternatively, the constraint can be added 
    to the kinematic differential equations using g_S.

    References
    ----------
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Schweizer2015: https://www.research-collection.ethz.ch/handle/20.500.11850/101867 \\
    Egenland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf

    Args
    ----------
    mass:       mass of rigid body
    K_Theta_S:  inertia of rigid body in body fixed K-frame
    q0:         generalized coordinates at t0
    u0:         generalized velocities at t0
    """

    def __init__(
        self,
        mass,
        K_Theta_S,
        q0=None,
        u0=None,
    ):
        self.nq = 7
        self.nu = 6
        self.nla_S = 1

        self.q0 = (
            np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
            if q0 is None
            else np.asarray(q0)
        )
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else np.asarray(u0)
        assert self.q0.size == self.nq
        assert self.u0.size == self.nu
        self.t = None
        self.q = np.empty(self.nq)
        # self.u = np.empty(self.nu)

        self.la_S0 = np.zeros(self.nla_S, dtype=float)
        assert self.la_S0.size == self.nla_S

        self.mass = mass
        self.K_Theta_S = K_Theta_S
        self.__M = np.zeros((self.nu, self.nu), dtype=float)
        self.__M[:3, :3] = self.mass * np.eye(3, dtype=float)
        self.__M[3:, 3:] = self.K_Theta_S

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.nq, dtype=np.common_type(q, u))
        q_dot[:3] = u[:3]
        q_dot[3:] = T_SO3_inv_quat(q[3:], normalize=False) @ u[3:]
        return q_dot

    def q_dot_q(self, t, q, u):
        q_dot_q = np.zeros((self.nq, self.nq), dtype=np.common_type(q, u))
        q_dot_q[3:, 3:] = np.einsum(
            "ijk,j->ik", T_SO3_inv_quat_P(q[3:], normalize=False), u[3:]
        )
        return q_dot_q

    def B(self, t, q):
        B = np.zeros((self.nq, self.nu), dtype=q.dtype)
        B[:3, :3] = np.eye(3, dtype=q.dtype)
        B[3:, 3:] = T_SO3_inv_quat(q[3:], normalize=False)
        return B

    def q_ddot(self, t, q, u, u_dot):
        raise NotImplementedError
        p = q[3:]
        p2 = p @ p
        B = T_SO3_inv_quat(p) / (p @ p)
        p_dot = B @ u[3:]
        p_ddot = (
            B @ u_dot[3:]
            + np.einsum("ijk,k,j->i", T_SO3_inv_quat_P(q[3:]), p_dot, u[3:])
            + 2 * p_dot * (p @ p_dot) / p2
        )

        q_ddot = np.zeros(self.nq, dtype=np.common_type(q, u, u_dot))
        q_ddot[:3] = u_dot[:3]
        q_ddot[3:] = p_ddot
        return q_ddot

    def step_callback(self, t, q, u):
        q[3:] = q[3:] / norm(q[3:])
        return q, u

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return self.__M

    def h(self, t, q, u):
        omega = u[3:]
        f = np.zeros(self.nu, dtype=np.common_type(q, u))
        f[3:] = -cross3(omega, self.K_Theta_S @ omega)
        return f

    def h_u(self, t, q, u):
        omega = u[3:]
        h_u = np.zeros((self.nu, self.nu), dtype=np.common_type(q, u))
        h_u[3:, 3:] = ax2skew(self.K_Theta_S @ omega) - ax2skew(omega) @ self.K_Theta_S
        return h_u

    #####################################################
    # stabilization conditions for the kinematic equation
    #####################################################
    def g_S(self, t, q):
        P = q[3:]
        return np.array([P @ P - 1.0], dtype=q.dtype)

    def g_S_q(self, t, q):
        P = q[3:]
        g_S_q = np.zeros((1, 7), dtype=q.dtype)
        g_S_q[0, 3:] = 2.0 * P
        return g_S_q

    def g_S_q_T_mu_q(self, t, q, mu):
        g_S_q_T_mu_q = np.zeros((7, 7), dtype=q.dtype)
        g_S_q_T_mu_q[3:, 3:] = 2.0 * mu[0] * np.eye(4, 4, dtype=float)
        return g_S_q_T_mu_q

    #####################
    # auxiliary functions
    #####################
    def local_qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def local_uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)
    
    def updated(self, t, q):
        ret = not (t == self.t and np.allclose(q, self.q))
        if ret:
            self.t = t
            self.q = q
        return ret

    def A_IK(self, t, q, frame_ID=None):
        if self.updated(t, q):
            self._A_IK = Exp_SO3_quat(q[3:])
        return self._A_IK

    def A_IK_q(self, t, q, frame_ID=None):
        if self.updated(t, q):
            A_IK_q = np.zeros((3, 3, self.nq), dtype=q.dtype)
            A_IK_q[:, :, 3:] = Exp_SO3_quat_p(q[3:])
        return A_IK_q

    def r_OP(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return q[:3] + self.A_IK(t, q) @ K_r_SP

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        r_OP_q = np.zeros((3, self.nq), dtype=q.dtype)
        r_OP_q[:, :3] = np.eye(3)
        r_OP_q[:, :] += np.einsum("ijk,j->ik", self.A_IK_q(t, q), K_r_SP)
        return r_OP_q

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return u[:3] + self.A_IK(t, q) @ cross3(u[3:], K_r_SP)

    def v_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return np.einsum("ijk,j->ik", self.A_IK_q(t, q), cross3(u[3:], K_r_SP))

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return u_dot[:3] + self.A_IK(t, q) @ (
            cross3(u_dot[3:], K_r_SP) + cross3(u[3:], cross3(u[3:], K_r_SP))
        )

    def a_P_q(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        return np.einsum(
            "ijk,j->ik",
            self.A_IK_q(t, q),
            cross3(u_dot[3:], K_r_SP) + cross3(u[3:], cross3(u[3:], K_r_SP)),
        )

    def a_P_u(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        a_P_u = np.zeros((3, self.nu), dtype=float)
        a_P_u[:, 3:] = -self.A_IK(t, q) @ (
            ax2skew(cross3(u[3:], K_r_SP)) + ax2skew(u[3:]) @ ax2skew(K_r_SP)
        )
        return a_P_u

    def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        J_P = np.zeros((3, self.nu), dtype=q.dtype)
        J_P[:, :3] = np.eye(3)
        J_P[:, 3:] = -self.A_IK(t, q) @ ax2skew(K_r_SP)
        return J_P

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3, dtype=float)):
        J_P_q = np.zeros((3, self.nu, self.nq), dtype=q.dtype)
        J_P_q[:, 3:, :] = np.einsum("ijk,jl->ilk", self.A_IK_q(t, q), -ax2skew(K_r_SP))
        return J_P_q

    def kappa_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.A_IK(t, q) @ (cross3(u[3:], cross3(u[3:], K_r_SP)))

    def kappa_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.einsum(
            "ijk,j->ik", self.A_IK_q(t, q), cross3(u[3:], cross3(u[3:], K_r_SP))
        )

    def kappa_P_u(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        kappa_P_u = np.zeros((3, self.nu))
        kappa_P_u[:, 3:] = -self.A_IK(t, q) @ (
            ax2skew(cross3(u[3:], K_r_SP)) + ax2skew(u[3:]) @ ax2skew(K_r_SP)
        )
        return kappa_P_u

    def K_Omega(self, t, q, u, frame_ID=None):
        return u[3:]

    def K_Omega_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq), dtype=np.common_type(q, u))

    def K_Psi(self, t, q, u, u_dot, frame_ID=None):
        return u_dot[3:]

    def K_Psi_q(self, t, q, u, u_dot, frame_ID=None):
        return np.zeros((3, self.nq), dtype=np.common_type(q, u, u_dot))

    def K_Psi_u(self, t, q, u, u_dot, frame_ID=None):
        return np.zeros((3, self.nu), dtype=np.common_type(q, u, u_dot))

    def K_kappa_R(self, t, q, u, frame_ID=None):
        return np.zeros(3, dtype=np.common_type(q, u))

    def K_kappa_R_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq), dtype=np.common_type(q, u))

    def K_kappa_R_u(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nu), dtype=np.common_type(q, u))

    def K_J_R(self, t, q, frame_ID=None):
        K_J_R = np.zeros((3, self.nu), dtype=q.dtype)
        K_J_R[:, 3:] = np.eye(3)
        return K_J_R

    def K_J_R_q(self, t, q, frame_ID=None):
        return np.zeros((3, self.nu, self.nq), dtype=q.dtype)

    ########
    # export
    ########
    def export(self, sol_i, **kwargs):
        points = [self.r_OP(sol_i.t, sol_i.q[self.qDOF])]
        vel = [self.v_P(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])]
        omega = [
            self.A_IK(sol_i.t, sol_i.q[self.qDOF])
            @ self.K_Omega(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])
        ]
        A_IK = np.vsplit(self.A_IK(sol_i.t, sol_i.q[self.qDOF]).T, 3)

        cells = [("vertex", [[0]])]
        cell_data = dict(
            v=[vel], Omega=[omega], ex=[A_IK[0]], ey=[A_IK[1]], ez=[A_IK[2]]
        )
        return points, cells, None, cell_data
