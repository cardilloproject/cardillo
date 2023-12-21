import numpy as np
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey

from cardillo.math import norm, cross3, ax2skew
from cardillo.math.approx_fprime import approx_fprime
from cardillo.math.prox import Sphere


class Sphere2Sphere:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        radius1,
        radius2,
        mu,
        e_N=None,
        e_F=None,
        frame_ID1=np.zeros(3, dtype=float),
        frame_ID2=np.zeros(3, dtype=float),
    ):
        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.radius1 = radius1
        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2
        self.radius2 = radius2

        self.nla_N = 1
        self.e_N = np.zeros(self.nla_N) if e_N is None else e_N * np.ones(self.nla_N)

        if mu > 0:
            self.nla_F = 2 * self.nla_N
            self.gamma_F = self.__gamma_F
            self.e_F = (
                np.zeros(self.nla_F) if e_F is None else e_F * np.ones(self.nla_F)
            )

            # fmt: off
            self.friction_laws = [
                ([0], [0, 1], Sphere(mu)), # Coulomb
            ]
            # fmt: on

        self.normal_cache = LRUCache(maxsize=1)
        self.normal_and_tangents_cache = LRUCache(maxsize=1)

    def assembler_callback(self):
        qDOF1 = self.subsystem1.local_qDOF_P(self.frame_ID1)
        qDOF2 = self.subsystem2.local_qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate(
            [self.subsystem1.qDOF[qDOF1], self.subsystem2.qDOF[qDOF2]]
        )
        self.nq1 = nq1 = len(qDOF1)
        self.nq2 = len(qDOF2)
        self._nq = self.nq1 + self.nq2

        uDOF1 = self.subsystem1.local_uDOF_P(self.frame_ID1)
        uDOF2 = self.subsystem2.local_uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate(
            [self.subsystem1.uDOF[uDOF1], self.subsystem2.uDOF[uDOF2]]
        )
        self.nu1 = nu1 = len(uDOF1)
        self.nu2 = len(uDOF2)
        self._nu = self.nu1 + self.nu2

        #####################################
        # auxiliary functions for subsystem 1
        #####################################
        self.r_OS1 = lambda t, q: self.subsystem1.r_OP(t, q[:nq1], self.frame_ID1)
        self.r_OS1_q = lambda t, q: self.subsystem1.r_OP_q(t, q[:nq1], self.frame_ID1)
        self.v_S1 = lambda t, q, u: self.subsystem1.v_P(
            t, q[:nq1], u[:nu1], self.frame_ID1
        )
        self.v_S1_q = lambda t, q, u: self.subsystem1.v_P_q(
            t, q[:nq1], u[:nu1], self.frame_ID1
        )
        self.a_S1 = lambda t, q, u, u_dot: self.subsystem1.a_P(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1
        )
        self.a_S1_q = lambda t, q, u, u_dot: self.subsystem1.a_P_q(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1
        )
        self.a_S1_u = lambda t, q, u, u_dot: self.subsystem1.a_P_u(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1
        )
        self.J_S1 = lambda t, q: self.subsystem1.J_P(t, q[:nq1], self.frame_ID1)
        self.J_S1_q = lambda t, q: self.subsystem1.J_P_q(t, q[:nq1], self.frame_ID1)

        self.Omega1 = lambda t, q, u: self.subsystem1.A_IK(
            t, q[:nq1], frame_ID=self.frame_ID1
        ) @ self.subsystem1.K_Omega(t, q[:nq1], u[:nu1], frame_ID=self.frame_ID1)
        self.Omega1_q = lambda t, q, u: self.subsystem1.A_IK(
            t, q[:nq1], frame_ID=self.frame_ID1
        ) @ self.subsystem1.K_Omega_q(
            t, q[:nq1], u[:nu1], frame_ID=self.frame_ID1
        ) + np.einsum(
            "ijk,j->ik",
            self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_Omega(t, q[:nq1], u[:nu1], frame_ID=self.frame_ID1),
        )
        self.J1_R = lambda t, q: self.subsystem1.A_IK(
            t, q[:nq1], frame_ID=self.frame_ID1
        ) @ self.subsystem1.K_J_R(t, q[:nq1], frame_ID=self.frame_ID1)
        self.J1_R_q = lambda t, q: np.einsum(
            "ijl,jk->ikl",
            self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_J_R(t, q[:nq1], frame_ID=self.frame_ID1),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem1.A_IK(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_J_R_q(t, q[:nq1], frame_ID=self.frame_ID1),
        )
        self.Psi1 = lambda t, q, u, a: self.subsystem1.A_IK(
            t, q[:nq1], frame_ID=self.frame_ID1
        ) @ self.subsystem1.K_Psi(t, q[:nq1], u[:nu1], a[:nu1], frame_ID=self.frame_ID1)
        self.Psi1_q = lambda t, q, u, a: self.subsystem1.A_IK(
            t, q[:nq1], frame_ID=self.frame_ID1
        ) @ self.subsystem1.K_Psi_q(
            t, q[:nq1], u[:nu1], a[:nu1], frame_ID=self.frame_ID1
        ) + np.einsum(
            "ijk,j->ik",
            self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_Psi(
                t, q[:nq1], u[:nu1], a[:nu1], frame_ID=self.frame_ID1
            ),
        )
        self.Psi1_u = lambda t, q, u, a: self.subsystem1.A_IK(
            t, q[:nq1], frame_ID=self.frame_ID1
        ) @ self.subsystem1.K_Psi_u(
            t, q[:nq1], u[:nu1], a[:nu1], frame_ID=self.frame_ID1
        )

        #####################################
        # auxiliary functions for subsystem 1
        #####################################
        self.r_OS2 = lambda t, q: self.subsystem2.r_OP(t, q[nq1:], self.frame_ID2)
        self.r_OS2_q = lambda t, q: self.subsystem2.r_OP_q(t, q[nq1:], self.frame_ID2)
        self.v_S2 = lambda t, q, u: self.subsystem2.v_P(
            t, q[nq1:], u[nu1:], self.frame_ID2
        )
        self.v_S2_q = lambda t, q, u: self.subsystem2.v_P_q(
            t, q[nq1:], u[nu1:], self.frame_ID2
        )
        self.a_S2 = lambda t, q, u, u_dot: self.subsystem2.a_P(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2
        )
        self.a_S2_q = lambda t, q, u, u_dot: self.subsystem2.a_P_q(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2
        )
        self.a_S2_u = lambda t, q, u, u_dot: self.subsystem2.a_P_u(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2
        )
        self.J_S2 = lambda t, q: self.subsystem2.J_P(t, q[nq1:], self.frame_ID2)
        self.J_S2_q = lambda t, q: self.subsystem2.J_P_q(t, q[nq1:], self.frame_ID2)

        self.Omega2 = lambda t, q, u: self.subsystem2.A_IK(
            t, q[nq1:], frame_ID=self.frame_ID2
        ) @ self.subsystem2.K_Omega(t, q[nq1:], u[nu1:], frame_ID=self.frame_ID2)
        self.Omega2_q = lambda t, q, u: self.subsystem2.A_IK(
            t, q[nq1:], frame_ID=self.frame_ID2
        ) @ self.subsystem2.K_Omega_q(
            t, q[nq1:], u[nu1:], frame_ID=self.frame_ID2
        ) + np.einsum(
            "ijk,j->ik",
            self.subsystem2.A_IK_q(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_Omega(t, q[nq1:], u[nu1:], frame_ID=self.frame_ID2),
        )
        self.J2_R = lambda t, q: self.subsystem2.A_IK(
            t, q[nq1:], frame_ID=self.frame_ID2
        ) @ self.subsystem2.K_J_R(t, q[nq1:], frame_ID=self.frame_ID2)
        self.J2_R_q = lambda t, q: np.einsum(
            "ijl,jk->ikl",
            self.subsystem2.A_IK_q(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_J_R(t, q[nq1:], frame_ID=self.frame_ID2),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem2.A_IK(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_J_R_q(t, q[nq1:], frame_ID=self.frame_ID2),
        )
        self.Psi2 = lambda t, q, u, a: self.subsystem2.A_IK(
            t, q[nq1:], frame_ID=self.frame_ID2
        ) @ self.subsystem2.K_Psi(t, q[nq1:], u[nu1:], a[nu1:], frame_ID=self.frame_ID2)
        # self.Psi1_q = lambda t, q, u, a: self.subsystem1.A_IK(
        #     t, q[:nq1], frame_ID=self.frame_ID1
        # ) @ self.subsystem1.K_Psi_q(
        #     t, q[:nq1], u[:nu1], a[:nu1], frame_ID=self.frame_ID1
        # ) + np.einsum(
        #     "ijk,j->ik",
        #     self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1),
        #     self.subsystem1.K_Psi(
        #         t, q[:nq1], u[:nu1], a[:nu1], frame_ID=self.frame_ID1
        #     ),
        # )
        # self.Psi1_u = lambda t, q, u, a: self.subsystem1.A_IK(
        #     t, q[:nq1], frame_ID=self.frame_ID1
        # ) @ self.subsystem1.K_Psi_u(
        #     t, q[:nq1], u[:nu1], a[:nu1], frame_ID=self.frame_ID1
        # )

    @cachedmethod(
        lambda self: self.normal_cache,
        key=lambda self, t, q: hashkey(t, *q),
    )
    def normal(self, t, q):
        r_S1S2 = self.r_OS2(t, q) - self.r_OS1(t, q)
        return r_S1S2 / norm(r_S1S2)

    # parametrize with spherical coordinates theta and phi
    def __angles(self, n):
        # n[2] < 1
        return (np.arccos(n[2]), np.arctan2(n[1], n[0]))

    @cachedmethod(
        lambda self: self.normal_and_tangents_cache,
        key=lambda self, t, q: hashkey(t, *q),
    )
    def normal_and_tangents(self, t, q):
        n = self.normal(t, q)
        theta, phi = self.__angles(n)
        # # derivative of n in spherical coordinates with respect to theta
        # # t1 = np.array([np.cos(theta) * np.cos(phi),
        # #                np.cos(theta) * np.sin(phi),
        # #                -np.sin(theta)])
        # # # derivative of n in spherical coordinates with respect to phi
        # # t2 = np.array([-np.sin(theta) * np.sin(phi),
        # #                np.sin(theta) * np.cos(phi),
        # #                0])
        # # return ((t1, t2))
        # return np.array([[np.cos(theta) * np.cos(phi), -np.sin(phi)],
        #                  [np.cos(theta) * np.sin(phi),  np.cos(phi)],
        #                  [-np.sin(theta),               0]])

        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        t1 = np.array([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta])
        t2 = np.array([-sin_phi, cos_phi, 0])
        return n, t1, t2

    ################
    # normal contact
    ################
    def g_N(self, t, q):
        r_S1S2 = self.r_OS2(t, q) - self.r_OS1(t, q)
        return np.array([norm(r_S1S2) - self.radius1 - self.radius2])

    def g_N_q(self, t, q):
        n = self.normal(t, q)
        r_OS1_q = self.r_OS1_q(t, q)
        r_OS2_q = self.r_OS2_q(t, q)
        g_N_q = np.concatenate((-n @ r_OS1_q, n @ r_OS2_q)).reshape(
            (self.nla_N, self._nq)
        )
        return g_N_q

        # g_N_q_num = approx_fprime(q, lambda q: self.g_N(t, q))
        # diff = g_N_q - g_N_q_num
        # error = np.linalg.norm(diff)
        # print(f"error g_N_q: {error}")
        # return g_N_q_num.reshape((self.nla_N, self._nq))

    def g_N_dot(self, t, q, u):
        return np.array([self.normal(t, q) @ (self.v_S2(t, q, u) - self.v_S1(t, q, u))])

    def g_N_dot_q(self, t, q, u):
        raise NotImplementedError

    def g_N_dot_u(self, t, q):
        n = self.normal(t, q)
        J_S1 = self.J_S1(t, q)
        J_S2 = self.J_S2(t, q)
        return np.concatenate((-n @ J_S1, n @ J_S2)).reshape((self.nla_N, self._nu))

    def W_N(self, t, q):
        return self.g_N_dot_u(t, q).T

    def g_N_ddot(self, t, q, u, u_dot):
        return np.array(
            [
                self.normal(t, q)
                @ (self.a_S2(t, q, u, u_dot) - self.a_S1(t, q, u, u_dot))
            ]
        )

    def g_N_ddot_q(self, t, q, u, u_dot):
        raise NotImplementedError

    def g_N_ddot_u(self, t, q, u, u_dot):
        raise NotImplementedError

    def Wla_N_q(self, t, q, la_N):
        return approx_fprime(q, lambda q: la_N @ self.g_N_dot_u(t, q))

    ##########
    # friction
    ##########
    def __gamma_F(self, t, q, u):
        n, t1, t2 = self.normal_and_tangents(t, q)

        v_P1 = self.v_S1(t, q, u) + cross3(self.Omega1(t, q, u), self.radius1 * n)
        v_P2 = self.v_S2(t, q, u) + cross3(self.Omega2(t, q, u), -self.radius2 * n)
        v_P1P2 = v_P2 - v_P1

        return np.array(
            [
                t1 @ v_P1P2,
                t2 @ v_P1P2,
            ],
            dtype=np.common_type(q, u),
        )

    def gamma_F_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.gamma_F(t, q, u))

    def gamma_F_u(self, t, q):
        n, t1, t2 = self.normal_and_tangents(t, q)

        J_P1 = self.J_S1(t, q) - ax2skew(self.radius1 * n) @ self.J1_R(t, q)
        J_P2 = self.J_S2(t, q) - ax2skew(-self.radius2 * n) @ self.J2_R(t, q)

        gamma_F_u = np.zeros((self.nla_F, self._nu), dtype=q.dtype)
        gamma_F_u[0, : self.nu1] = -t1 @ J_P1
        gamma_F_u[0, self.nu1 :] = t1 @ J_P2
        gamma_F_u[1, : self.nu1] = -t2 @ J_P1
        gamma_F_u[1, self.nu1 :] = t2 @ J_P2
        return gamma_F_u

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def gamma_F_dot(self, t, q, u, u_dot):
        n, t1, t2 = self.normal_and_tangents(t, q)

        a_P1 = self.a_S1(t, q, u, u_dot) + cross3(
            self.Psi1(t, q, u, u_dot), self.radius1 * n
        )
        a_P2 = self.a_S2(t, q, u, u_dot) + cross3(
            self.Psi2(t, q, u, u_dot), -self.radius2 * n
        )
        a_P1P2 = a_P2 - a_P1

        return np.array(
            [
                t1 @ a_P1P2,
                t2 @ a_P1P2,
            ],
            dtype=np.common_type(q, u, u_dot),
        )

    def Wla_F_q(self, t, q, la_F):
        return approx_fprime(q, lambda q: la_F @ self.gamma_F_u(t, q))
