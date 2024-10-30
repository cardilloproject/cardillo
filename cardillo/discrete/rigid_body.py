from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
import numpy as np
from vtk import VTK_VERTEX

from cardillo.math import (
    cross3,
    ax2skew,
    norm,
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
    Spurrier,
)


class RigidBody:
    def __init__(self, mass, B_Theta_C, q0=None, u0=None, name="rigid_body"):
        """Rigid body parametrized by center of mass in inertial basis I_r_OP in
        R^3 and non-unit quaternions p in R^4 for rotation, i.e., the 
        generalized position coordinates are q = (I_r_OP, p) in R^7. The 
        generalized velocity coordinates u = (I_v_C, B_omega_IK) in R^6 are 
        composed of the velocity of the center of mass I_v_C in R^3 together 
        with the angular velocity represented in the body-fixed K-basis 
        B_omega_IK in R^3. 
        
        Exponential function and kinematic differential equation are found in 
        Egeland2002 (6.199), (6.329) and (6.330). The implementation below 
        handles non-unit quaternions. After each successfull time step they are 
        projected to be of unit length. Alternatively, the constraint can be added 
        to the kinematic differential equations using g_S.

        Parameters
        ----------
        mass: float
            Mass of rigid body
        B_Theta_C:  np.array(3,3)
            Inertia tensor represented w.r.t. body fixed K-system.
        q0 : np.array(7)
            Initial position coordinates at time t0.
        u0 : np.array(6)
            Initial velocity coordinates at time t0.
        name : str
            Name of rigid body.
        
        References
        ----------
        Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
        Schweizer2015: https://www.research-collection.ethz.ch/handle/20.500.11850/101867 \\
        Egenland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf

        """

        self.nq = 7
        self.nu = 6
        self.nla_S = 1

        self.q0 = (
            np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
            if q0 is None
            else np.asarray(q0)
        )
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else np.asarray(u0)
        self.la_S0 = np.zeros(self.nla_S, dtype=float)
        assert self.q0.size == self.nq
        assert self.u0.size == self.nu
        assert self.la_S0.size == self.nla_S

        self.mass = mass
        self.B_Theta_C = B_Theta_C
        self.__M = np.zeros((self.nu, self.nu), dtype=float)
        self.__M[:3, :3] = self.mass * np.eye(3, dtype=float)
        self.__M[3:, 3:] = self.B_Theta_C

        self.name = name

        self.A_IB_cache = LRUCache(maxsize=1)
        self.A_IB_q_cache = LRUCache(maxsize=1)
        self.r_OP_cache = LRUCache(maxsize=1)
        self.v_P_cache = LRUCache(maxsize=1)
        self.J_P_cache = LRUCache(maxsize=1)

    #####################
    # utility
    #####################
    @staticmethod
    def pose2q(r_OC, A_IB):
        return np.concatenate([r_OC, Spurrier(A_IB)])

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

    def q_dot_u(self, t, q):
        q_dot_u = np.zeros((self.nq, self.nu), dtype=q.dtype)
        q_dot_u[:3, :3] = np.eye(3, dtype=q.dtype)
        q_dot_u[3:, 3:] = T_SO3_inv_quat(q[3:], normalize=False)
        return q_dot_u

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
        f[3:] = -cross3(omega, self.B_Theta_C @ omega)
        return f

    def h_u(self, t, q, u):
        omega = u[3:]
        h_u = np.zeros((self.nu, self.nu), dtype=np.common_type(q, u))
        h_u[3:, 3:] = ax2skew(self.B_Theta_C @ omega) - ax2skew(omega) @ self.B_Theta_C
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
    def local_qDOF_P(self, xi=None):
        return np.arange(self.nq)

    def local_uDOF_P(self, xi=None):
        return np.arange(self.nu)

    @cachedmethod(
        lambda self: self.A_IB_cache,
        key=lambda self, t, q, xi=None: hashkey(t, *q),
    )
    def A_IB(self, t, q, xi=None):
        return Exp_SO3_quat(q[3:])

    @cachedmethod(
        lambda self: self.A_IB_q_cache,
        key=lambda self, t, q, xi=None: hashkey(t, *q),
    )
    def A_IB_q(self, t, q, xi=None):
        A_IB_q = np.zeros((3, 3, self.nq), dtype=q.dtype)
        A_IB_q[:, :, 3:] = Exp_SO3_quat_p(q[3:])
        return A_IB_q

    @cachedmethod(
        lambda self: self.r_OP_cache,
        key=lambda self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float): hashkey(
            t, *q, *B_r_CP
        ),
    )
    def r_OP(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return q[:3] + self.A_IB(t, q) @ B_r_CP

    def r_OP_q(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        r_OP_q = np.zeros((3, self.nq), dtype=q.dtype)
        r_OP_q[:, :3] = np.eye(3)
        r_OP_q[:, :] += np.einsum("ijk,j->ik", self.A_IB_q(t, q), B_r_CP)
        return r_OP_q

    @cachedmethod(
        lambda self: self.v_P_cache,
        key=lambda self, t, q, u, xi=None, B_r_CP=np.zeros(3, dtype=float): hashkey(
            t, *q, *u, *B_r_CP
        ),
    )
    def v_P(self, t, q, u, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return u[:3] + self.A_IB(t, q) @ cross3(u[3:], B_r_CP)

    def v_P_q(self, t, q, u, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return np.einsum("ijk,j->ik", self.A_IB_q(t, q), cross3(u[3:], B_r_CP))

    def a_P(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return u_dot[:3] + self.A_IB(t, q) @ (
            cross3(u_dot[3:], B_r_CP) + cross3(u[3:], cross3(u[3:], B_r_CP))
        )

    def a_P_q(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return np.einsum(
            "ijk,j->ik",
            self.A_IB_q(t, q),
            cross3(u_dot[3:], B_r_CP) + cross3(u[3:], cross3(u[3:], B_r_CP)),
        )

    def a_P_u(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        a_P_u = np.zeros((3, self.nu), dtype=float)
        a_P_u[:, 3:] = -self.A_IB(t, q) @ (
            ax2skew(cross3(u[3:], B_r_CP)) + ax2skew(u[3:]) @ ax2skew(B_r_CP)
        )
        return a_P_u

    @cachedmethod(
        lambda self: self.J_P_cache,
        key=lambda self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float): hashkey(
            t, *q, *B_r_CP
        ),
    )
    def J_P(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        J_P = np.zeros((3, self.nu), dtype=q.dtype)
        J_P[:, :3] = np.eye(3)
        J_P[:, 3:] = -self.A_IB(t, q) @ ax2skew(B_r_CP)
        return J_P

    def J_P_q(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        J_P_q = np.zeros((3, self.nu, self.nq), dtype=q.dtype)
        J_P_q[:, 3:, :] = np.einsum("ijk,jl->ilk", self.A_IB_q(t, q), -ax2skew(B_r_CP))
        return J_P_q

    def kappa_P(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return self.A_IB(t, q) @ (cross3(u[3:], cross3(u[3:], B_r_CP)))

    def kappa_P_q(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return np.einsum(
            "ijk,j->ik", self.A_IB_q(t, q), cross3(u[3:], cross3(u[3:], B_r_CP))
        )

    def kappa_P_u(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        kappa_P_u = np.zeros((3, self.nu))
        kappa_P_u[:, 3:] = -self.A_IB(t, q) @ (
            ax2skew(cross3(u[3:], B_r_CP)) + ax2skew(u[3:]) @ ax2skew(B_r_CP)
        )
        return kappa_P_u

    def B_Omega(self, t, q, u, xi=None):
        return u[3:]

    def B_Omega_q(self, t, q, u, xi=None):
        return np.zeros((3, self.nq), dtype=np.common_type(q, u))

    def B_Psi(self, t, q, u, u_dot, xi=None):
        return u_dot[3:]

    def B_Psi_q(self, t, q, u, u_dot, xi=None):
        return np.zeros((3, self.nq), dtype=np.common_type(q, u, u_dot))

    def B_Psi_u(self, t, q, u, u_dot, xi=None):
        return np.zeros((3, self.nu), dtype=np.common_type(q, u, u_dot))

    def B_kappa_R(self, t, q, u, xi=None):
        return np.zeros(3, dtype=np.common_type(q, u))

    def B_kappa_R_q(self, t, q, u, xi=None):
        return np.zeros((3, self.nq), dtype=np.common_type(q, u))

    def B_kappa_R_u(self, t, q, u, xi=None):
        return np.zeros((3, self.nu), dtype=np.common_type(q, u))

    def B_J_R(self, t, q, xi=None):
        B_J_R = np.zeros((3, self.nu), dtype=q.dtype)
        B_J_R[:, 3:] = np.eye(3)
        return B_J_R

    def B_J_R_q(self, t, q, xi=None):
        return np.zeros((3, self.nu, self.nq), dtype=q.dtype)

    ########
    # export
    ########
    def export(self, sol_i, **kwargs):
        points = [self.r_OP(sol_i.t, sol_i.q[self.qDOF])]
        vel = self.v_P(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])
        omega = self.A_IB(sol_i.t, sol_i.q[self.qDOF]) @ self.B_Omega(
            sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF]
        )

        ex, ey, ez = self.A_IB(sol_i.t, sol_i.q[self.qDOF]).T

        cells = [(VTK_VERTEX, [0])]
        cell_data = dict(v=[vel], Omega=[omega], ex=[ex], ey=[ey], ez=[ez])
        return points, cells, None, cell_data
