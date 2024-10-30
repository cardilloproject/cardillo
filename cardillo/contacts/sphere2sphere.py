import numpy as np
from cachetools import LRUCache, cachedmethod
from cachetools.keys import hashkey

from cardillo.math import ax2skew, ax2skew_a, cross3, norm, smallest_rotation
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
        xi1=None,
        xi2=None,
        name="sphere_to_sphere_contact",
    ):
        """Contact between two spheres modelled as unilateral constraint with set-valued Coulomb friction.

        Parameters
        ----------
        subsystem1 : object
            Sphere 1
        subsystem2 : object
            Sphere 2
        radius1 : float
            Radius of subsystem1
        radius2 : float
            Radius of subsystem2
        mu : float
            Frictional coefficient
        e_N : float
            Restitution coefficient for Newton-like impact law in normal direction.
        e_F : float
            Restitution coefficient for Newton-like impact law for friction.
        xi1 : TODO
        xi2 : TODO
        name : str
            Name of contribution.
        """
        self.subsystem1 = subsystem1
        self.xi1 = xi1
        self.radius1 = radius1
        self.subsystem2 = subsystem2
        self.xi2 = xi2
        self.radius2 = radius2
        self.mu = mu
        self.name = name

        self.nla_N = 1
        self.e_N = np.zeros(self.nla_N) if e_N is None else e_N * np.ones(self.nla_N)

        if mu > 0:
            self.nla_F = 2 * self.nla_N
            self.gamma_F = self.__gamma_F
            self.gamma_F_q = self.__gamma_F_q
            self.e_F = (
                np.zeros(self.nla_F) if e_F is None else e_F * np.ones(self.nla_F)
            )

            # fmt: off
            self.friction_laws = [
                ([0], [0, 1], Sphere(mu)), # Coulomb
            ]
            # fmt: on

        self.n_cache = LRUCache(maxsize=1)
        self.n_q1_q2_cache = LRUCache(maxsize=1)
        self.t1t2_cache = LRUCache(maxsize=1)
        self.t1t2_q1_q2_cache = LRUCache(maxsize=1)

    def assembler_callback(self):
        qDOF1 = self.subsystem1.local_qDOF_P(self.xi1)
        qDOF2 = self.subsystem2.local_qDOF_P(self.xi2)
        self.qDOF = np.concatenate(
            [self.subsystem1.qDOF[qDOF1], self.subsystem2.qDOF[qDOF2]]
        )
        self.nq1 = nq1 = len(qDOF1)
        self.nq2 = len(qDOF2)
        self._nq = self.nq1 + self.nq2

        uDOF1 = self.subsystem1.local_uDOF_P(self.xi1)
        uDOF2 = self.subsystem2.local_uDOF_P(self.xi2)
        self.uDOF = np.concatenate(
            [self.subsystem1.uDOF[uDOF1], self.subsystem2.uDOF[uDOF2]]
        )
        self.nu1 = nu1 = len(uDOF1)
        self.nu2 = len(uDOF2)
        self._nu = self.nu1 + self.nu2

        #####################################
        # auxiliary functions for subsystem 1
        #####################################
        self.r_OC1 = lambda t, q: self.subsystem1.r_OP(t, q[:nq1], self.xi1)
        self.r_OC1_q1 = lambda t, q: self.subsystem1.r_OP_q(t, q[:nq1], self.xi1)
        self.v_C1 = lambda t, q, u: self.subsystem1.v_P(t, q[:nq1], u[:nu1], self.xi1)
        self.v_C1_q1 = lambda t, q, u: self.subsystem1.v_P_q(
            t, q[:nq1], u[:nu1], self.xi1
        )
        self.a_C1 = lambda t, q, u, u_dot: self.subsystem1.a_P(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.xi1
        )
        self.a_C1_q1 = lambda t, q, u, u_dot: self.subsystem1.a_P_q(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.xi1
        )
        self.a_C1_u1 = lambda t, q, u, u_dot: self.subsystem1.a_P_u(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.xi1
        )
        self.J_C1 = lambda t, q: self.subsystem1.J_P(t, q[:nq1], self.xi1)
        self.J_C1_q1 = lambda t, q: self.subsystem1.J_P_q(t, q[:nq1], self.xi1)

        if hasattr(self.subsystem1, "A_IB"):
            self.A_IB1 = lambda t, q: self.subsystem1.A_IB(t, q[:nq1], xi=self.xi1)
            self.A_IB1_q1 = lambda t, q: self.subsystem1.A_IB_q(t, q[:nq1], xi=self.xi1)
            self.Omega1 = lambda t, q, u: self.subsystem1.A_IB(
                t, q[:nq1], xi=self.xi1
            ) @ self.subsystem1.B_Omega(t, q[:nq1], u[:nu1], xi=self.xi1)
            self.Omega1_q1 = lambda t, q, u: self.subsystem1.A_IB(
                t, q[:nq1], xi=self.xi1
            ) @ self.subsystem1.B_Omega_q(t, q[:nq1], u[:nu1], xi=self.xi1) + np.einsum(
                "ijk,j->ik",
                self.subsystem1.A_IB_q(t, q[:nq1], xi=self.xi1),
                self.subsystem1.B_Omega(t, q[:nq1], u[:nu1], xi=self.xi1),
            )
            self.J1_R = lambda t, q: self.subsystem1.A_IB(
                t, q[:nq1], xi=self.xi1
            ) @ self.subsystem1.B_J_R(t, q[:nq1], xi=self.xi1)
            self.J1_R_q1 = lambda t, q: np.einsum(
                "ijl,jk->ikl",
                self.subsystem1.A_IB_q(t, q[:nq1], xi=self.xi1),
                self.subsystem1.B_J_R(t, q[:nq1], xi=self.xi1),
            ) + np.einsum(
                "ij,jkl->ikl",
                self.subsystem1.A_IB(t, q[:nq1], xi=self.xi1),
                self.subsystem1.B_J_R_q(t, q[:nq1], xi=self.xi1),
            )
            self.Psi1 = lambda t, q, u, a: self.subsystem1.A_IB(
                t, q[:nq1], xi=self.xi1
            ) @ self.subsystem1.B_Psi(t, q[:nq1], u[:nu1], a[:nu1], xi=self.xi1)
            self.Psi1_q1 = lambda t, q, u, a: self.subsystem1.A_IB(
                t, q[:nq1], xi=self.xi1
            ) @ self.subsystem1.B_Psi_q(
                t, q[:nq1], u[:nu1], a[:nu1], xi=self.xi1
            ) + np.einsum(
                "ijk,j->ik",
                self.subsystem1.A_IB_q(t, q[:nq1], xi=self.xi1),
                self.subsystem1.B_Psi(t, q[:nq1], u[:nu1], a[:nu1], xi=self.xi1),
            )
            self.Psi1_u1 = lambda t, q, u, a: self.subsystem1.A_IB(
                t, q[:nq1], xi=self.xi1
            ) @ self.subsystem1.B_Psi_u(t, q[:nq1], u[:nu1], a[:nu1], xi=self.xi1)
        else:
            self.A_IB1 = lambda t, q: np.eye(3)
            self.Omega1 = lambda t, q, u: np.zeros(3)
            self.Omega1_q1 = lambda t, q, u: np.zeros((3, self.subsystem1.nq))
            self.J1_R = lambda t, q: np.zeros((self.subsystem1.nu, 3))
            self.J1_R_q1 = lambda t, q: np.zeros(
                (self.subsystem1.nu, 3, self.subsystem1.nq)
            )
            self.Psi1 = lambda t, q, u, u_dot: np.zeros(3)
            self.Psi1_q1 = lambda t, q, u, u_dot: np.zeros((3, self.subsystem1.nq))
            self.Psi1_u1 = lambda t, q, u, u_dot: np.zeros((3, self.subsystem1.nu))

        #####################################
        # auxiliary functions for subsystem 2
        #####################################
        self.r_OC2 = lambda t, q: self.subsystem2.r_OP(t, q[nq1:], self.xi2)
        self.r_OC2_q2 = lambda t, q: self.subsystem2.r_OP_q(t, q[nq1:], self.xi2)
        self.v_C2 = lambda t, q, u: self.subsystem2.v_P(t, q[nq1:], u[nu1:], self.xi2)
        self.v_C2_q2 = lambda t, q, u: self.subsystem2.v_P_q(
            t, q[nq1:], u[nu1:], self.xi2
        )
        self.a_C2 = lambda t, q, u, u_dot: self.subsystem2.a_P(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.xi2
        )
        self.a_C2_q2 = lambda t, q, u, u_dot: self.subsystem2.a_P_q(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.xi2
        )
        self.a_C2_u2 = lambda t, q, u, u_dot: self.subsystem2.a_P_u(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.xi2
        )
        self.J_C2 = lambda t, q: self.subsystem2.J_P(t, q[nq1:], self.xi2)
        self.J_C2_q2 = lambda t, q: self.subsystem2.J_P_q(t, q[nq1:], self.xi2)

        if hasattr(self.subsystem2, "A_IB"):
            self.A_IB2 = lambda t, q: self.subsystem2.A_IB(t, q[nq1:], xi=self.xi2)
            self.A_IB2_q2 = lambda t, q: self.subsystem2.A_IB_q(t, q[nq1:], xi=self.xi2)
            self.Omega2 = lambda t, q, u: self.subsystem2.A_IB(
                t, q[nq1:], xi=self.xi2
            ) @ self.subsystem2.B_Omega(t, q[nq1:], u[nu1:], xi=self.xi2)
            self.Omega2_q2 = lambda t, q, u: self.subsystem2.A_IB(
                t, q[nq1:], xi=self.xi2
            ) @ self.subsystem2.B_Omega_q(t, q[nq1:], u[nu1:], xi=self.xi2) + np.einsum(
                "ijk,j->ik",
                self.subsystem2.A_IB_q(t, q[nq1:], xi=self.xi2),
                self.subsystem2.B_Omega(t, q[nq1:], u[nu1:], xi=self.xi2),
            )
            self.J2_R = lambda t, q: self.subsystem2.A_IB(
                t, q[nq1:], xi=self.xi2
            ) @ self.subsystem2.B_J_R(t, q[nq1:], xi=self.xi2)
            self.J2_R_q2 = lambda t, q: np.einsum(
                "ijl,jk->ikl",
                self.subsystem2.A_IB_q(t, q[nq1:], xi=self.xi2),
                self.subsystem2.B_J_R(t, q[nq1:], xi=self.xi2),
            ) + np.einsum(
                "ij,jkl->ikl",
                self.subsystem2.A_IB(t, q[nq1:], xi=self.xi2),
                self.subsystem2.B_J_R_q(t, q[nq1:], xi=self.xi2),
            )
            self.Psi2 = lambda t, q, u, a: self.subsystem2.A_IB(
                t, q[nq1:], xi=self.xi2
            ) @ self.subsystem2.B_Psi(t, q[nq1:], u[nu1:], a[nu1:], xi=self.xi2)
            self.Psi2_q2 = lambda t, q, u, a: self.subsystem2.A_IB(
                t, q[nq1:], xi=self.xi
            ) @ self.subsystem2.B_Psi_q(
                t, q[nq1:], u[nu1:], a[nu1:], xi=self.xi
            ) + np.einsum(
                "ijk,j->ik",
                self.subsystem2.A_IB_q(t, q[nq1:], xi=self.xi),
                self.subsystem2.B_Psi(t, q[nq1:], u[nu1:], a[nu1:], xi=self.xi),
            )
            self.Psi_u = lambda t, q, u, a: self.subsystem2.A_IB(
                t, q[nq1:], xi=self.xi
            ) @ self.subsystem2.B_Psi_u(t, q[nq1:], u[nu1:], a[nu1:], xi=self.xi)
        else:
            self.A_IB2 = lambda t, q: np.eye(3)
            self.Omega2 = lambda t, q, u: np.zeros(3)
            self.Omega2_q2 = lambda t, q, u: np.zeros((3, self.subsystem2.nq))
            self.J2_R = lambda t, q: np.zeros((self.subsystem2.nu, 3))
            self.J2_R_q2 = lambda t, q: np.zeros(
                (self.subsystem2.nu, 3, self.subsystem2.nq)
            )
            self.Psi2 = lambda t, q, u, u_dot: np.zeros(3)
            self.Psi2_q2 = lambda t, q, u, u_dot: np.zeros((3, self.subsystem2.nq))
            self.Psi2_u2 = lambda t, q, u, u_dot: np.zeros((3, self.subsystem2.nu))

        #################################
        # compute reference contact basis
        #################################
        t0 = self.t0
        q0 = np.concatenate((self.subsystem1.q0, self.subsystem2.q0))
        n = self.n(t0, q0)
        vs = np.cross(n, np.eye(3), axisb=0)
        norm_vs = np.linalg.norm(vs, axis=-1)
        # use axis with best angle
        axis = np.argmax(norm_vs)
        t1 = vs[axis] / norm_vs[axis]
        w = cross3(n, t1)
        t2 = w / norm(w)
        self.reference_contact_basis = np.vstack((t1, t2, n)).T

    def step_callback(self, t, q, u):
        # update reference contact basis using smallest rotation, see Crisfield1997 (16.107)
        n = self.n(t, q)
        b1, b2, b3 = self.reference_contact_basis.T @ n
        factor = (self.reference_contact_basis[:, 2] + n) / (1 + b3)
        self.reference_contact_basis = np.vstack(
            (
                self.reference_contact_basis[:, 0] - b1 * factor,
                self.reference_contact_basis[:, 1] - b2 * factor,
                n,
            )
        ).T
        return q, u

    @cachedmethod(
        lambda self: self.n_cache,
        key=lambda self, t, q: hashkey(t, *q),
    )
    def n(self, t, q):
        r_C1C2 = self.r_OC2(t, q) - self.r_OC1(t, q)
        return r_C1C2 / norm(r_C1C2)

    @cachedmethod(
        lambda self: self.n_q1_q2_cache,
        key=lambda self, t, q: hashkey(t, *q),
    )
    def n_q1_q2(self, t, q):
        n = self.n(t, q)
        r_C1C2 = self.r_OC2(t, q) - self.r_OC1(t, q)
        tmp = (np.eye(3, dtype=q.dtype) - np.outer(n, n)) / norm(r_C1C2)
        n_q1 = -tmp @ self.r_OC1_q1(t, q)
        n_q2 = tmp @ self.r_OC2_q2(t, q)
        return n_q1, n_q2

    @cachedmethod(
        lambda self: self.t1t2_cache,
        key=lambda self, t, q: hashkey(t, *q),
    )
    def t1t2(self, t, q):
        n = self.n(t, q)
        # always use t2 of reference basis as second direction
        t2_ref = self.reference_contact_basis[:, 1]
        v = cross3(t2_ref, n)
        t1 = v / norm(v)
        w = cross3(n, t1)
        t2 = w / norm(w)
        return t1, t2

    @cachedmethod(
        lambda self: self.t1t2_q1_q2_cache,
        key=lambda self, t, q: hashkey(t, *q),
    )
    def t1t2_q1_q2(self, t, q):
        n = self.n(t, q)
        n_tilde = ax2skew(n)
        n_q1, n_q2 = self.n_q1_q2(t, q)
        t1, t2 = self.t1t2(t, q)
        t1_tilde = ax2skew(t1)
        t1_ref = self.reference_contact_basis[:, 0]
        t1_ref_tilde = ax2skew(t1_ref)
        l1, l2 = norm(cross3(n, t1_ref)), norm(cross3(n, t1))
        tmp1 = (np.eye(3) - np.outer(t1, t1)) / l1
        tmp2 = (np.eye(3) - np.outer(t2, t2)) / l2
        t1_q1 = -tmp1 @ t1_ref_tilde @ n_q1
        t1_q2 = -tmp1 @ t1_ref_tilde @ n_q2
        t2_q1 = tmp2 @ (n_tilde @ t1_q1 - t1_tilde @ n_q1)
        t2_q2 = tmp2 @ (n_tilde @ t1_q2 - t1_tilde @ n_q2)
        return t1_q1, t1_q2, t2_q1, t2_q2

    ################
    # normal contact
    ################
    def g_N(self, t, q):
        r_C1C2 = self.r_OC2(t, q) - self.r_OC1(t, q)
        return np.array([norm(r_C1C2) - self.radius1 - self.radius2])

    def g_N_q(self, t, q):
        n = self.n(t, q)
        r_OC1_q1 = self.r_OC1_q1(t, q)
        r_OC2_q2 = self.r_OC2_q2(t, q)
        g_N_q = np.concatenate((-n @ r_OC1_q1, n @ r_OC2_q2)).reshape(
            (self.nla_N, self._nq)
        )
        return g_N_q

    def g_N_dot(self, t, q, u):
        return np.array([self.n(t, q) @ (self.v_C2(t, q, u) - self.v_C1(t, q, u))])

    def g_N_dot_q(self, t, q, u):
        raise NotImplementedError

    def g_N_dot_u(self, t, q):
        n = self.n(t, q)
        J_C1 = self.J_C1(t, q)
        J_C2 = self.J_C2(t, q)
        return np.concatenate((-n @ J_C1, n @ J_C2)).reshape((self.nla_N, self._nu))

    def W_N(self, t, q):
        return self.g_N_dot_u(t, q).T

    def g_N_ddot(self, t, q, u, u_dot):
        return np.array(
            [self.n(t, q) @ (self.a_C2(t, q, u, u_dot) - self.a_C1(t, q, u, u_dot))]
        )

    def Wla_N_q(self, t, q, la_N):
        Wla_N_q = np.zeros((self._nu, self._nq), dtype=np.common_type(q, la_N))
        n = self.n(t, q)
        nq1, nq2 = self.n_q1_q2(t, q)
        J_C1 = self.J_C1(t, q)
        J_C2 = self.J_C2(t, q)
        J_C1_q1 = self.J_C1_q1(t, q)
        J_C2_q2 = self.J_C2_q2(t, q)

        Wla_N_q[: self.nu1, : self.nq1] = -np.einsum("ijk,i->jk", J_C1_q1, n) * la_N
        Wla_N_q[self.nu1 :, self.nq1 :] = np.einsum("ijk,i->jk", J_C2_q2, n) * la_N
        Wla_N_q[: self.nu1, : self.nq1] += -J_C1.T @ nq1 * la_N
        Wla_N_q[: self.nu1, self.nq1 :] += -J_C1.T @ nq2 * la_N
        Wla_N_q[self.nu1 :, : self.nq1] += J_C2.T @ nq1 * la_N
        Wla_N_q[self.nu1 :, self.nq1 :] += J_C2.T @ nq2 * la_N
        return Wla_N_q

    ##########
    # friction
    ##########
    def __gamma_F(self, t, q, u):
        n = self.n(t, q)
        t1, t2 = self.t1t2(t, q)

        v_P1 = self.v_C1(t, q, u) + cross3(self.Omega1(t, q, u), self.radius1 * n)
        v_P2 = self.v_C2(t, q, u) + cross3(self.Omega2(t, q, u), -self.radius2 * n)
        v_P1P2 = v_P2 - v_P1

        return np.array(
            [
                t1 @ v_P1P2,
                t2 @ v_P1P2,
            ],
            dtype=np.common_type(q, u),
        )

    def __gamma_F_q(self, t, q, u):
        n = self.n(t, q)
        t1, t2 = self.t1t2(t, q)
        n_q1, n_q2 = self.n_q1_q2(t, q)
        t1_q1, t1_q2, t2_q1, t2_q2 = self.t1t2_q1_q2(t, q)

        v_P1 = self.v_C1(t, q, u) + cross3(self.Omega1(t, q, u), self.radius1 * n)
        v_P2 = self.v_C2(t, q, u) + cross3(self.Omega2(t, q, u), -self.radius2 * n)
        v_P1P2 = v_P2 - v_P1

        r1 = self.radius1
        r2 = self.radius2
        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        v_P1_q1 = (
            self.v_C1_q1(t, q, u)
            + r1 * ax2skew(Omega1) @ n_q1
            - r1 * ax2skew(n) @ self.Omega1_q1(t, q, u)
        )
        v_P1_q2 = r1 * ax2skew(self.Omega1(t, q, u)) @ n_q2
        v_P2_q1 = -r2 * ax2skew(Omega2) @ n_q1
        v_P2_q2 = (
            self.v_C2_q2(t, q, u)
            - r2 * ax2skew(self.Omega2(t, q, u)) @ n_q2
            + r2 * ax2skew(n) @ self.Omega2_q2(t, q, u)
        )

        gamma_F_q = np.zeros((self.nla_F, self._nq))
        gamma_F_q[:, : self.nq1] = np.vstack(
            (
                v_P1P2 @ t1_q1 + t1 @ (v_P2_q1 - v_P1_q1),
                v_P1P2 @ t2_q1 + t2 @ (v_P2_q1 - v_P1_q1),
            )
        )
        gamma_F_q[:, self.nq1 :] += np.vstack(
            (
                v_P1P2 @ t1_q2 + t1 @ (v_P2_q2 - v_P1_q2),
                v_P1P2 @ t2_q2 + t2 @ (v_P2_q2 - v_P1_q2),
            )
        )
        return gamma_F_q

    def gamma_F_u(self, t, q):
        n = self.n(t, q)
        t1, t2 = self.t1t2(t, q)

        J_P1 = self.J_C1(t, q) - ax2skew(self.radius1 * n) @ self.J1_R(t, q)
        J_P2 = self.J_C2(t, q) - ax2skew(-self.radius2 * n) @ self.J2_R(t, q)

        gamma_F_u = np.zeros((self.nla_F, self._nu), dtype=q.dtype)
        gamma_F_u[0, : self.nu1] = -t1 @ J_P1
        gamma_F_u[0, self.nu1 :] = t1 @ J_P2
        gamma_F_u[1, : self.nu1] = -t2 @ J_P1
        gamma_F_u[1, self.nu1 :] = t2 @ J_P2
        return gamma_F_u

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def gamma_F_dot(self, t, q, u, u_dot):
        n = self.n(t, q)
        t1, t2 = self.t1t2(t, q)

        a_P1 = self.a_C1(t, q, u, u_dot) + cross3(
            self.Psi1(t, q, u, u_dot), self.radius1 * n
        )
        a_P2 = self.a_C2(t, q, u, u_dot) + cross3(
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
        n = self.n(t, q)
        t1, t2 = self.t1t2(t, q)
        n_q1, n_q2 = self.n_q1_q2(t, q)
        t1_q1, t1_q2, t2_q1, t2_q2 = self.t1t2_q1_q2(t, q)

        r1 = self.radius1
        r2 = self.radius2
        J1_R = self.J1_R(t, q)
        J2_R = self.J2_R(t, q)
        J_P1 = self.J_C1(t, q) - ax2skew(r1 * n) @ J1_R
        J_P2 = self.J_C2(t, q) - ax2skew(-r2 * n) @ J2_R
        J_P1_q1 = (
            self.J_C1_q1(t, q)
            - np.einsum("ij,jkl->ikl", ax2skew(r1 * n), self.J1_R_q1(t, q))
            - np.einsum("ijk,kl,jm->iml", ax2skew_a(), r1 * n_q1, J1_R)
        )
        J_P1_q2 = -np.einsum("ijk,kl,jm->iml", ax2skew_a(), r1 * n_q2, J1_R)
        J_P2_q1 = -np.einsum("ijk,kl,jm->iml", ax2skew_a(), -r2 * n_q1, J2_R)
        J_P2_q2 = (
            self.J_C2_q2(t, q)
            - np.einsum("ij,jkl->ikl", ax2skew(-r2 * n), self.J2_R_q2(t, q))
            - np.einsum("ijk,kl,jm->iml", ax2skew_a(), -r2 * n_q2, J2_R)
        )

        Wla_F_q = np.zeros((self._nu, self._nq))
        Wla_F_q[: self.nu1, : self.nq1] = -J_P1.T @ (
            la_F[0] * t1_q1 + la_F[1] * t2_q1
        ) - np.einsum("ijk, i->jk", J_P1_q1, la_F[0] * t1 + la_F[1] * t2)
        Wla_F_q[: self.nu1, self.nq1 :] = -J_P1.T @ (
            la_F[0] * t1_q2 + la_F[1] * t2_q2
        ) - np.einsum("ijk, i->jk", J_P1_q2, la_F[0] * t1 + la_F[1] * t2)
        Wla_F_q[self.nu1 :, : self.nq1] = J_P2.T @ (
            la_F[0] * t1_q1 + la_F[1] * t2_q1
        ) + np.einsum("ijk, i->jk", J_P2_q1, la_F[0] * t1 + la_F[1] * t2)
        Wla_F_q[self.nu1 :, self.nq1 :] = J_P2.T @ (
            la_F[0] * t1_q2 + la_F[1] * t2_q2
        ) + np.einsum("ijk, i->jk", J_P2_q2, la_F[0] * t1 + la_F[1] * t2)
        return Wla_F_q
