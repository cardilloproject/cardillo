import numpy as np
from vtk import VTK_VERTEX

from cardillo.math import (
    cross3,
    ax2skew,
)


eye3 = np.eye(3)
eye12 = np.eye(12)


class RigidBodyDirector:
    def __init__(self, mass, B_Theta_C, q0=None, u0=None, name="rigid_body_director"):
        """Rigid body parametrized by center of mass in inertial basis I_r_OP in
        R^3 and three orthonormal basis vectors I_e_i in R^3 for rotation, i.e., the 
        generalized position coordinates are q = (I_r_OP, I_e_x, I_e_y, I_e_z) in R^12. 
        The generalized velocity coordinates are given as u = q' in R^12.

        Orthonormality of the basis vectors is enforced by 6 bilateral constraints g.

        Parameters
        ----------
        mass: float
            Mass of rigid body
        B_Theta_C:  np.array(3,3)
            Inertia tensor represented w.r.t. body fixed K-system.
        q0 : np.array(12)
            Initial position coordinates at time t0.
        u0 : np.array(12)
            Initial velocity coordinates at time t0.
        name : str
            Name of rigid body.
        
        References
        ----------
        TODO!

        """

        self.nq = 12
        self.nu = 12
        self.nla_g = 6

        self.q0 = (
            np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
            if q0 is None
            else np.asarray(q0)
        )
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else np.asarray(u0)
        self.la_g0 = np.zeros(self.nla_g, dtype=float)
        assert self.q0.size == self.nq
        assert self.u0.size == self.nu
        assert self.la_g0.size == self.nla_g

        self.mass = mass
        self.B_Theta_C = B_Theta_C
        self.E0 = 0.5 * np.trace(B_Theta_C) * np.eye(3) - B_Theta_C
        self.__M = np.zeros((self.nu, self.nu), dtype=float)
        self.__M[:3, :3] = self.mass * np.eye(3)
        self.__M[3:, 3:] = np.kron(self.E0, eye3)

        self.name = name

    #####################
    # utility
    #####################
    @staticmethod
    def pose2q(r_OC, A_IB):
        return np.concatenate([r_OC, A_IB.reshape(-1, order="F")])
    
    @staticmethod
    def velocity2u(v_C, B_Omega, A_IB):
        A_IB_dot = A_IB @ ax2skew(B_Omega)
        return np.concatenate([v_C, A_IB_dot.reshape(-1, order="F")])

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return u

    def q_dot_u(self, t, q):
        return eye12

    def step_callback(self, t, q, u):
        # TODO: Maybe add orthogonalization via QR or Gram-Schmidt?
        return q, u

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return self.__M

    #####################################################
    # director constraints
    #####################################################
    def g(self, t, q):
        d1, d2, d3 = q[3:6], q[6:9], q[9:]
        return np.array([
            0.5 * (d1 @ d1 - 1.0),
            0.5 * (d2 @ d2 - 1.0),
            0.5 * (d3 @ d3 - 1.0),
            d1 @ d2,
            d2 @ d3,
            d3 @ d1,
        ], dtype=q.dtype)

    def g_q(self, t, q):
        d1, d2, d3 = q[3:6], q[6:9], q[9:]
        g_q = np.zeros((6, 12), dtype=q.dtype)
        g_q[0, 3:6] = d1
        g_q[1, 6:9] = d2
        g_q[2, 9:12] = d3
        g_q[3, 3:6] = d2
        g_q[3, 6:9] = d1
        g_q[4, 6:9] = d3
        g_q[4, 9:] = d2
        g_q[5, 3:6] = d3
        g_q[5, 9:] = d1
        return g_q
    
    def g_dot(self, t, q, u):
        d1, d2, d3 = q[3:6], q[6:9], q[9:]
        d1_dot, d2_dot, d3_dot = u[3:6], u[6:9], u[9:]
        return np.array([
            d1 @ d1_dot,
            d2 @ d2_dot,
            d3 @ d3_dot,
            d1 @ d2_dot + d2 @ d1_dot,
            d2 @ d3_dot + d3 @ d2_dot,
            d3 @ d1_dot + d1 @ d3_dot,
        ], dtype=np.common_type(q, u))
    
    def g_ddot(self, t, q, u, u_dot):
        d1, d2, d3 = q[3:6], q[6:9], q[9:]
        d1_dot, d2_dot, d3_dot = u[3:6], u[6:9], u[9:]
        d1_ddot, d2_ddot, d3_ddot = u_dot[3:6], u_dot[6:9], u_dot[9:]
        return np.array([
            d1 @ d1_ddot + d1_dot @ d1_dot,
            d2 @ d2_ddot + d2_dot @ d2_dot,
            d3 @ d3_ddot + d3_dot @ d3_dot,
            d1 @ d2_ddot + d1_dot @ d2_dot + d2 @ d1_ddot + d2_dot @ d1_dot,
            d2 @ d3_ddot + d3 @ d2_ddot + d2_dot @ d3_dot + d3_dot @ d2_dot,
            d3 @ d1_ddot + d1 @ d3_ddot + d3_dot @ d1_dot + d1_dot @ d3_dot,
        ], dtype=np.common_type(q, u))
    
    def g_dot_u(self, t, q):
        return self.g_q(t, q)
    
    def W_g(self, t, q):
        return self.g_q(t, q).T

    #####################
    # auxiliary functions
    #####################
    def local_qDOF_P(self, xi=None):
        return np.arange(self.nq)

    def local_uDOF_P(self, xi=None):
        return np.arange(self.nu)

    def A_IB(self, t, q, xi=None):
        return q[3:].reshape(3, 3, order="F")

    def A_IB_q(self, t, q, xi=None):
        return eye12[3:].reshape(3, 3, -1, order="F")
    
    def r_OP(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return q[:3] + self.A_IB(t, q) @ B_r_CP

    def r_OP_q(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        r_OP_q = np.zeros((3, self.nq), dtype=q.dtype)
        r_OP_q[:, :3] = np.eye(3)
        r_OP_q[:, :] += np.einsum("ijk,j->ik", self.A_IB_q(t, q), B_r_CP)
        return r_OP_q

    def v_P(self, t, q, u, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        # Note: Same formula as for r_OP but with u instead of q
        return self.r_OP(t, u, xi, B_r_CP)

    def v_P_q(self, t, q, u, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return np.zeros((3, self.nq), dtype=float)

    def a_P(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        # Note: Same formula as for r_OP but with u_dot instead of q
        return self.r_OP(t, u_dot, xi, B_r_CP)

    def a_P_q(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return np.zeros((3, self.nq), dtype=float)

    def a_P_u(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return np.zeros((3, self.nq), dtype=float)

    def J_P(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return self.r_OP_q(t, q, xi, B_r_CP)

    def J_P_q(self, t, q, xi=None, B_r_CP=np.zeros(3, dtype=float)):
        return np.zeros((3, self.nq, self.nq), dtype=float)

    def kappa_P(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return np.zeros(3, dtype=float)

    def kappa_P_q(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return np.zeros((3, self.nq), dtype=float)

    def kappa_P_u(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        return np.zeros((3, self.nu), dtype=float)

    def B_Omega(self, t, q, u, xi=None):
        d1, d2, d3 = q[3:6], q[6:9], q[9:]

        L = np.zeros((3, self.nq), dtype=np.common_type(q, u))
        L[0, 6:9] = -d3
        L[0, 9:] = d2
        L[1, 3:6] = d3
        L[1, 9:] = -d1
        L[2, 3:6] = -d2
        L[2, 6:9] = d1

        return -0.5 * (L @ u)

    def B_Omega_q(self, t, q, u, xi=None):
        raise NotImplementedError

    def B_Psi(self, t, q, u, u_dot, xi=None):
        raise NotImplementedError

    def B_Psi_q(self, t, q, u, u_dot, xi=None):
        raise NotImplementedError
        return np.zeros((3, self.nq), dtype=np.common_type(q, u, u_dot))

    def B_Psi_u(self, t, q, u, u_dot, xi=None):
        raise NotImplementedError
        return np.zeros((3, self.nu), dtype=np.common_type(q, u, u_dot))

    def B_kappa_R(self, t, q, u, xi=None):
        return np.zeros(3, dtype=float)

    def B_kappa_R_q(self, t, q, u, xi=None):
        return np.zeros((3, self.nq), dtype=float)

    def B_kappa_R_u(self, t, q, u, xi=None):
        return np.zeros((3, self.nu), dtype=float)

    def B_J_R(self, t, q, xi=None):
        d1, d2, d3 = q[3:6], q[6:9], q[9:]

        L = np.zeros((3, self.nq), dtype=q.dtype)
        L[0, 6:9] = -d3
        L[0, 9:] = d2
        L[1, 3:6] = d3
        L[1, 9:] = -d1
        L[2, 3:6] = -d2
        L[2, 6:9] = d1

        return -0.5 * L

    def B_J_R_q(self, t, q, xi=None):
        raise NotImplementedError
        # B_J_R_q = np.zeros((3, self.nq, self.nq), dtype=float)
        # B_J_R_q[0, 6:9, 9:[:, None]] -= N[node] * eye3 # -d3
        # B_J_R_q[0, 9:,  6:9[:, None]] += N[node] * eye3 # d2
        # B_J_R_q[1, 3:6, 9:[:, None]] += N[node] * eye3 # d3
        # B_J_R_q[1, 9:,  3:6[:, None]] -= N[node] * eye3 # -d1
        # B_J_R_q[2, 3:6, 6:9[:, None]] -= N[node] * eye3 # -d2
        # B_J_R_q[2, 6:9, 3:6[:, None]] += N[node] * eye3 # d1
        # return B_J_R_q

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
