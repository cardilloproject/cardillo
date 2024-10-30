import warnings

import numpy as np

from cardillo.math.algebra import ax2skew, cross3
from cardillo.math.approx_fprime import approx_fprime


def concatenate_qDOF(object):
    qDOF1 = object.subsystem1.qDOF
    qDOF2 = object.subsystem2.qDOF
    local_qDOF1 = object.subsystem1.local_qDOF_P(object.xi1)
    local_qDOF2 = object.subsystem2.local_qDOF_P(object.xi2)

    object.qDOF = np.concatenate((qDOF1[local_qDOF1], qDOF2[local_qDOF2]))
    object._nq1 = nq1 = len(local_qDOF1)
    object._nq2 = len(local_qDOF2)
    object._nq = object._nq1 + object._nq2

    return local_qDOF1, local_qDOF2


def concatenate_uDOF(object):
    uDOF1 = object.subsystem1.uDOF
    uDOF2 = object.subsystem2.uDOF
    local_uDOF1 = object.subsystem1.local_uDOF_P(object.xi1)
    local_uDOF2 = object.subsystem2.local_uDOF_P(object.xi2)

    object.uDOF = np.concatenate((uDOF1[local_uDOF1], uDOF2[local_uDOF2]))
    object._nu1 = nu1 = len(local_uDOF1)
    object._nu2 = len(local_uDOF2)
    object._nu = object._nu1 + object._nu2

    return local_uDOF1, local_uDOF2


def auxiliary_functions(
    object,
    B1_r_P1B0,
    B2_r_P2B0,
    A_K1B0=None,
    A_K2B0=None,
):
    nq1 = object._nq1
    nu1 = object._nu1

    # auxiliary functions for subsystem 1
    object.r_OJ1 = lambda t, q: object.subsystem1.r_OP(
        t, q[:nq1], object.xi1, B1_r_P1B0
    )
    object.r_OJ1_q1 = lambda t, q: object.subsystem1.r_OP_q(
        t, q[:nq1], object.xi1, B1_r_P1B0
    )
    object.v_J1 = lambda t, q, u: object.subsystem1.v_P(
        t, q[:nq1], u[:nu1], object.xi1, B1_r_P1B0
    )
    object.v_J1_q1 = lambda t, q, u: object.subsystem1.v_P_q(
        t, q[:nq1], u[:nu1], object.xi1, B1_r_P1B0
    )
    object.a_J1 = lambda t, q, u, u_dot: object.subsystem1.a_P(
        t, q[:nq1], u[:nu1], u_dot[:nu1], object.xi1, B1_r_P1B0
    )
    object.a_J1_q1 = lambda t, q, u, u_dot: object.subsystem1.a_P_q(
        t, q[:nq1], u[:nu1], u_dot[:nu1], object.xi1, B1_r_P1B0
    )
    object.a_J1_u1 = lambda t, q, u, u_dot: object.subsystem1.a_P_u(
        t, q[:nq1], u[:nu1], u_dot[:nu1], object.xi1, B1_r_P1B0
    )
    object.J_J1 = lambda t, q: object.subsystem1.J_P(t, q[:nq1], object.xi1, B1_r_P1B0)
    object.J_J1_q1 = lambda t, q: object.subsystem1.J_P_q(
        t, q[:nq1], object.xi1, B1_r_P1B0
    )
    object.A_IJ1 = lambda t, q: object.subsystem1.A_IB(t, q[:nq1], object.xi1) @ A_K1B0
    object.A_IJ1_q1 = lambda t, q: np.einsum(
        "ijl,jk->ikl", object.subsystem1.A_IB_q(t, q[:nq1], object.xi1), A_K1B0
    )
    object.Omega1 = lambda t, q, u: object.subsystem1.A_IB(
        t, q[:nq1], object.xi1
    ) @ object.subsystem1.B_Omega(t, q[:nq1], u[:nu1], object.xi1)
    object.Omega1_q1 = lambda t, q, u: np.einsum(
        "ijk,j->ik",
        object.subsystem1.A_IB_q(t, q[:nq1], xi=object.xi1),
        object.subsystem1.B_Omega(t, q[:nq1], u[:nu1], object.xi1),
    ) + object.subsystem1.A_IB(t, q[:nq1], xi=object.xi1) @ object.subsystem1.B_Omega_q(
        t, q[:nq1], u[:nu1], object.xi1
    )

    object.Psi1 = lambda t, q, u, u_dot: object.subsystem1.A_IB(
        t, q[:nq1], object.xi1
    ) @ object.subsystem1.B_Psi(t, q[:nq1], u[:nu1], u_dot[:nu1], object.xi1)
    object.Psi1_q1 = lambda t, q, u, u_dot: np.einsum(
        "ijk,j->ik",
        object.subsystem1.A_IB_q(t, q[:nq1], xi=object.xi1),
        object.subsystem1.B_Psi(t, q[:nq1], u[:nu1], u_dot[:nu1], object.xi1),
    ) + object.subsystem1.A_IB(t, q[:nq1], xi=object.xi1) @ object.subsystem1.B_Psi_q(
        t, q[:nq1], u[:nu1], u_dot[:nu1], object.xi1
    )
    object.Psi1_u1 = lambda t, q, u, u_dot: object.subsystem1.A_IB(
        t, q[:nq1], xi=object.xi1
    ) @ object.subsystem1.B_Psi_u(t, q[:nq1], u[:nu1], u_dot[:nu1], object.xi1)

    object.J_R1 = lambda t, q: object.subsystem1.A_IB(
        t, q[:nq1], object.xi1
    ) @ object.subsystem1.B_J_R(t, q[:nq1], object.xi1)
    object.J_R1_q1 = lambda t, q: np.einsum(
        "ijk,jl->ilk",
        object.subsystem1.A_IB_q(t, q[:nq1], object.xi1),
        object.subsystem1.B_J_R(t, q[:nq1], object.xi1),
    ) + np.einsum(
        "ij,jkl->ikl",
        object.subsystem1.A_IB(t, q[:nq1], object.xi1),
        object.subsystem1.B_J_R_q(t, q[:nq1], object.xi1),
    )

    # auxiliary functions for subsystem 2
    object.r_OJ2 = lambda t, q: object.subsystem2.r_OP(
        t, q[nq1:], object.xi2, B2_r_P2B0
    )
    object.r_OJ2_q2 = lambda t, q: object.subsystem2.r_OP_q(
        t, q[nq1:], object.xi2, B2_r_P2B0
    )
    object.v_J2 = lambda t, q, u: object.subsystem2.v_P(
        t, q[nq1:], u[nu1:], object.xi2, B2_r_P2B0
    )
    object.v_J2_q2 = lambda t, q, u: object.subsystem2.v_P_q(
        t, q[nq1:], u[nu1:], object.xi2, B2_r_P2B0
    )
    object.a_J2 = lambda t, q, u, u_dot: object.subsystem2.a_P(
        t, q[nq1:], u[nu1:], u_dot[nu1:], object.xi2, B2_r_P2B0
    )
    object.a_J2_q2 = lambda t, q, u, u_dot: object.subsystem2.a_P_q(
        t, q[nq1:], u[nu1:], u_dot[nu1:], object.xi2, B2_r_P2B0
    )
    object.a_J2_u2 = lambda t, q, u, u_dot: object.subsystem2.a_P_u(
        t, q[nq1:], u[nu1:], u_dot[nu1:], object.xi2, B2_r_P2B0
    )
    object.J_J2 = lambda t, q: object.subsystem2.J_P(t, q[nq1:], object.xi2, B2_r_P2B0)
    object.J_J2_q2 = lambda t, q: object.subsystem2.J_P_q(
        t, q[nq1:], object.xi2, B2_r_P2B0
    )
    object.A_IJ2 = lambda t, q: object.subsystem2.A_IB(t, q[nq1:], object.xi2) @ A_K2B0
    object.A_IJ2_q2 = lambda t, q: np.einsum(
        "ijk,jl->ilk", object.subsystem2.A_IB_q(t, q[nq1:], object.xi2), A_K2B0
    )
    object.Omega2 = lambda t, q, u: object.subsystem2.A_IB(
        t, q[nq1:], object.xi2
    ) @ object.subsystem2.B_Omega(t, q[nq1:], u[nu1:], object.xi2)
    object.Omega2_q2 = lambda t, q, u: np.einsum(
        "ijk,j->ik",
        object.subsystem2.A_IB_q(t, q[nq1:], xi=object.xi2),
        object.subsystem2.B_Omega(t, q[nq1:], u[nu1:], object.xi2),
    ) + object.subsystem2.A_IB(t, q[nq1:], xi=object.xi2) @ object.subsystem2.B_Omega_q(
        t, q[nq1:], u[nu1:], object.xi2
    )

    object.Psi2 = lambda t, q, u, u_dot: object.subsystem2.A_IB(
        t, q[nq1:], object.xi2
    ) @ object.subsystem2.B_Psi(t, q[nq1:], u[nu1:], u_dot[nu1:], object.xi2)
    object.Psi2_q2 = lambda t, q, u, u_dot: np.einsum(
        "ijk,j->ik",
        object.subsystem2.A_IB_q(t, q[nq1:], xi=object.xi2),
        object.subsystem2.B_Psi(t, q[nq1:], u[nu1:], u_dot[nu1:], object.xi2),
    ) + object.subsystem2.A_IB(t, q[nq1:], xi=object.xi2) @ object.subsystem2.B_Psi_q(
        t, q[nq1:], u[nu1:], u_dot[nu1:], object.xi2
    )
    object.Psi2_u2 = lambda t, q, u, u_dot: object.subsystem2.A_IB(
        t, q[nq1:], xi=object.xi2
    ) @ object.subsystem2.B_Psi_u(t, q[nq1:], u[nu1:], u_dot[nu1:], object.xi2)

    object.J_R2 = lambda t, q: object.subsystem2.A_IB(
        t, q[nq1:], object.xi2
    ) @ object.subsystem2.B_J_R(t, q[nq1:], object.xi2)
    object.J_R2_q2 = lambda t, q: np.einsum(
        "ijk,jl->ilk",
        object.subsystem2.A_IB_q(t, q[nq1:], object.xi2),
        object.subsystem2.B_J_R(t, q[nq1:], object.xi2),
    ) + np.einsum(
        "ij,jkl->ikl",
        object.subsystem2.A_IB(t, q[nq1:], object.xi2),
        object.subsystem2.B_J_R_q(t, q[nq1:], object.xi2),
    )


class PositionOrientationBase:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        projection_pairs_rotation,
        r_OJ0=None,
        A_IJ0=None,
        xi1=None,
        xi2=None,
        **kwargs,
    ):
        self.subsystem1 = subsystem1
        self.subsystem2 = subsystem2
        self.xi1 = xi1
        self.xi2 = xi2
        self.r_OJ0 = r_OJ0
        self.A_IJ0 = A_IJ0

        # guard against flawed constrained_axes input
        self.nla_g_rot = len(projection_pairs_rotation)
        for pair in projection_pairs_rotation:
            assert len(np.unique(pair)) == 2
            for i in pair:
                assert i in [0, 1, 2]

        self.nla_g = 3 + self.nla_g_rot
        self.projection_pairs = projection_pairs_rotation

        self.constrain_orientation = self.nla_g_rot > 0

        if "name" in kwargs:
            self.name = kwargs.get("name")

    def assembler_callback(self):
        local_qDOF1, local_qDOF2 = concatenate_qDOF(self)
        concatenate_uDOF(self)

        r_OP10 = self.subsystem1.r_OP(
            self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.xi1
        )
        r_OP20 = self.subsystem2.r_OP(
            self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.xi2
        )

        # check for A_IB of subsystem 1
        if hasattr(self.subsystem1, "A_IB"):
            A_IB10 = self.subsystem1.A_IB(
                self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.xi1
            )

            if self.r_OJ0 is None:
                self.r_OJ0 = r_OP10

            if self.A_IJ0 is None:
                self.A_IJ0 = A_IB10

            B1_r_P1J0 = A_IB10.T @ (self.r_OJ0 - r_OP10)
            A_K1J0 = A_IB10.T @ self.A_IJ0
        else:
            B1_r_P1J0 = np.zeros(3)
            A_K1J0 = None  # unused
            assert self.nla_g_rot == 0  # Spherical case

        # check for A_IB of subsystem 2
        if hasattr(self.subsystem2, "A_IB"):
            A_IB20 = self.subsystem2.A_IB(
                self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.xi2
            )

            if self.r_OJ0 is None:
                self.r_OJ0 = r_OP20

            if self.A_IJ0 is None:
                self.A_IJ0 = A_IB20

            B2_r_P2J0 = A_IB20.T @ (self.r_OJ0 - r_OP20)
            A_K2J0 = A_IB20.T @ self.A_IJ0
        else:
            B2_r_P2J0 = np.zeros(3)
            A_K2J0 = None  # unused
            assert self.nla_g_rot == 0  # Spherical case

        auxiliary_functions(self, B1_r_P1J0, B2_r_P2J0, A_K1J0, A_K2J0)

    def g(self, t, q):
        g = np.zeros(self.nla_g, dtype=q.dtype)
        g[:3] = self.r_OJ2(t, q) - self.r_OJ1(t, q)

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)
            for i, (a, b) in enumerate(self.projection_pairs):
                g[3 + i] = A_IJ1[:, a] @ A_IJ2[:, b]

        return g

    def g_q(self, t, q):
        nq1 = self._nq1
        g_q = np.zeros((self.nla_g, self._nq), dtype=q.dtype)

        g_q[:3, :nq1] = -self.r_OJ1_q1(t, q)
        g_q[:3, nq1:] = self.r_OJ2_q2(t, q)

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)

            A_IJ1_q1 = self.A_IJ1_q1(t, q)
            A_IJ2_q2 = self.A_IJ2_q2(t, q)

            for i, (a, b) in enumerate(self.projection_pairs):
                g_q[3 + i, :nq1] = A_IJ2[:, b] @ A_IJ1_q1[:, a]
                g_q[3 + i, nq1:] = A_IJ1[:, a] @ A_IJ2_q2[:, b]

        return g_q

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g, dtype=np.common_type(q, u))
        g_dot[:3] = self.v_J2(t, q, u) - self.v_J1(t, q, u)

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)

            Omega21 = self.Omega1(t, q, u) - self.Omega2(t, q, u)

            for i, (a, b) in enumerate(self.projection_pairs):
                n = cross3(A_IJ1[:, a], A_IJ2[:, b])
                g_dot[3 + i] = n @ Omega21

        return g_dot

    def g_dot_q(self, t, q, u):
        nq1 = self._nq1
        g_dot_q = np.zeros((self.nla_g, self._nq), dtype=np.common_type(q, u))

        g_dot_q[:3, :nq1] = -self.v_J1_q1(t, q, u)
        g_dot_q[:3, nq1:] = self.v_J2_q2(t, q, u)

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)

            A_IJ1_q1 = self.A_IJ1_q1(t, q)
            A_IJ2_q2 = self.A_IJ2_q2(t, q)

            Omega21 = self.Omega1(t, q, u) - self.Omega2(t, q, u)
            Omega1_q1 = self.Omega1_q1(t, q, u)
            Omega2_q2 = self.Omega2_q2(t, q, u)

            for i, (a, b) in enumerate(self.projection_pairs):
                e_a, e_b = A_IJ1[:, a], A_IJ2[:, b]
                n = cross3(e_a, e_b)
                g_dot_q[3 + i, :nq1] = (
                    n @ Omega1_q1 - Omega21 @ ax2skew(e_b) @ A_IJ1_q1[:, a]
                )
                g_dot_q[3 + i, nq1:] = (
                    -n @ Omega2_q2 + Omega21 @ ax2skew(e_a) @ A_IJ2_q2[:, b]
                )

        return g_dot_q

    def g_dot_u(self, t, q):
        return self.W_g(t, q).T

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g, dtype=np.common_type(q, u, u_dot))
        g_ddot[:3] = self.a_J2(t, q, u, u_dot) - self.a_J1(t, q, u, u_dot)

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)

            Omega1 = self.Omega1(t, q, u)
            Omega2 = self.Omega2(t, q, u)
            Omega21 = Omega1 - Omega2
            Psi21 = self.Psi1(t, q, u, u_dot) - self.Psi2(t, q, u, u_dot)

            for i, (a, b) in enumerate(self.projection_pairs):
                e_a, e_b = A_IJ1[:, a], A_IJ2[:, b]
                n = cross3(e_a, e_b)
                g_ddot[3 + i] = (
                    cross3(cross3(Omega1, e_a), e_b) + cross3(e_a, cross3(Omega2, e_b))
                ) @ Omega21 + n @ Psi21

        return g_ddot

    def W_g(self, t, q):
        nu1 = self._nu1
        W_g = np.zeros((self._nu, self.nla_g), dtype=q.dtype)
        W_g[:nu1, :3] = -self.J_J1(t, q).T
        W_g[nu1:, :3] = self.J_J2(t, q).T

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)
            J = np.hstack([self.J_R1(t, q), -self.J_R2(t, q)])

            for i, (a, b) in enumerate(self.projection_pairs):
                n = cross3(A_IJ1[:, a], A_IJ2[:, b])
                W_g[:, 3 + i] = n @ J

        return W_g

    def Wla_g_q(self, t, q, la_g):
        nq1 = self._nq1
        nu1 = self._nu1
        Wla_g_q = np.zeros((self._nu, self._nq), dtype=np.common_type(q, la_g))

        Wla_g_q[:nu1, :nq1] += np.einsum("i,ijk->jk", -la_g[:3], self.J_J1_q1(t, q))
        Wla_g_q[nu1:, nq1:] += np.einsum("i,ijk->jk", la_g[:3], self.J_J2_q2(t, q))

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)

            A_IJ1_q1 = self.A_IJ1_q1(t, q)
            A_IJ2_q2 = self.A_IJ2_q2(t, q)

            J_R1 = self.J_R1(t, q)
            J_R2 = self.J_R2(t, q)
            J_R1_q1 = self.J_R1_q1(t, q)
            J_R2_q2 = self.J_R2_q2(t, q)

            for i, (a, b) in enumerate(self.projection_pairs):
                e_a, e_b = A_IJ1[:, a], A_IJ2[:, b]
                n = cross3(e_a, e_b)
                n_q1 = -ax2skew(e_b) @ A_IJ1_q1[:, a]
                n_q2 = ax2skew(e_a) @ A_IJ2_q2[:, b]
                Wla_g_q[:nu1, :nq1] += np.einsum(
                    "i,ijk->jk", la_g[3 + i] * n, J_R1_q1
                ) + np.einsum("ij,ik->kj", la_g[3 + i] * n_q1, J_R1)
                Wla_g_q[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[3 + i] * n_q2, J_R1)
                Wla_g_q[nu1:, :nq1] += np.einsum("ij,ik->kj", -la_g[3 + i] * n_q1, J_R2)
                Wla_g_q[nu1:, nq1:] += np.einsum(
                    "i,ijk->jk", -la_g[3 + i] * n, J_R2_q2
                ) + np.einsum("ij,ik->kj", -la_g[3 + i] * n_q2, J_R2)

        return Wla_g_q

    # TODO analytical derivative
    def g_q_T_mu_q(self, t, q, mu):
        warnings.warn("'PositionOrientationBase.g_q_T_mu_q' uses numerical derivative.")
        return approx_fprime(q, lambda q: self.g_q(t, q).T @ mu)


class ProjectedPositionOrientationBase:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        constrained_axes_translation,
        projection_pairs_rotation,
        r_OJ0=None,
        A_IJ0=None,
        xi1=None,
        xi2=None,
    ):
        self.subsystem1 = subsystem1
        self.subsystem2 = subsystem2
        self.xi1 = xi1
        self.xi2 = xi2
        self.r_OJ0 = r_OJ0
        self.A_IJ0 = A_IJ0

        self.nla_g_trans = len(constrained_axes_translation)

        # guard against flawed input
        self.nla_g_rot = len(projection_pairs_rotation)
        for pair in projection_pairs_rotation:
            assert len(np.unique(pair)) == 2
            for i in pair:
                assert i in [0, 1, 2]

        self.nla_g = self.nla_g_trans + self.nla_g_rot
        assert self.nla_g > 0

        self.constrained_axes_displacement = constrained_axes_translation
        self.projection_pairs_rotation = projection_pairs_rotation

        self.constrain_translation = self.nla_g_trans > 0
        self.constrain_orientation = self.nla_g_rot > 0

    def assembler_callback(self):
        local_qDOF1, local_qDOF2 = concatenate_qDOF(self)
        concatenate_uDOF(self)

        r_OP10 = self.subsystem1.r_OP(
            self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.xi1
        )
        r_OP20 = self.subsystem2.r_OP(
            self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.xi2
        )

        # check for A_IB of subsystem 1
        if hasattr(self.subsystem1, "A_IB"):
            A_IB10 = self.subsystem1.A_IB(
                self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.xi1
            )

            if self.r_OJ0 is None:
                self.r_OJ0 = r_OP10

            if self.A_IJ0 is None:
                self.A_IJ0 = A_IB10

            B1_r_P1J0 = A_IB10.T @ (self.r_OJ0 - r_OP10)
            A_K1J0 = A_IB10.T @ self.A_IJ0
        else:
            B1_r_P1J0 = np.zeros(3)
            A_K1J0 = None  # unused
            assert self.nla_g_rot == 0  # Spherical case

        # check for A_IB of subsystem 2
        if hasattr(self.subsystem2, "A_IB"):
            A_IB20 = self.subsystem2.A_IB(
                self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.xi2
            )

            if self.r_OJ0 is None:
                self.r_OJ0 = r_OP20

            if self.A_IJ0 is None:
                self.A_IJ0 = A_IB20

            B2_r_P2J0 = A_IB20.T @ (self.r_OJ0 - r_OP20)
            A_K2J0 = A_IB20.T @ self.A_IJ0
        else:
            B2_r_P2J0 = np.zeros(3)
            A_K2J0 = None  # unused
            assert self.nla_g_rot == 0  # Spherical case

        if self.r_OJ0 is None:
            self.r_OJ0 = r_OP10

        if self.A_IJ0 is None:
            self.A_IJ0 = A_IB10

        auxiliary_functions(self, B1_r_P1J0, B2_r_P2J0, A_K1J0, A_K2J0)

    def g(self, t, q):
        g = np.zeros(self.nla_g, dtype=q.dtype)

        A_IJ1 = self.A_IJ1(t, q)
        if self.constrain_translation:
            r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
            for i, ax in enumerate(self.constrained_axes_displacement):
                g[i] = r_J1J2 @ A_IJ1[:, ax]

        if self.constrain_orientation:
            A_IJ2 = self.A_IJ2(t, q)
            for i, (a, b) in enumerate(self.projection_pairs_rotation):
                g[self.nla_g_trans + i] = A_IJ1[:, a] @ A_IJ2[:, b]
        return g

    def g_q(self, t, q):
        nq1 = self._nq1
        g_q = np.zeros((self.nla_g, self._nq), dtype=q.dtype)

        A_IJ1 = self.A_IJ1(t, q)
        A_IJ1_q1 = self.A_IJ1_q1(t, q)

        if self.constrain_translation:
            r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
            r_OJ1_q1 = self.r_OJ1_q1(t, q)
            r_OJ2_q2 = self.r_OJ2_q2(t, q)
            for i, ax in enumerate(self.constrained_axes_displacement):
                g_q[i, :nq1] = -A_IJ1[:, ax] @ r_OJ1_q1 + r_J1J2 @ A_IJ1_q1[:, ax]
                g_q[i, nq1:] = A_IJ1[:, ax] @ r_OJ2_q2

        if self.constrain_orientation:
            A_IJ2 = self.A_IJ2(t, q)
            A_IJ2_q2 = self.A_IJ2_q2(t, q)

            for i, (a, b) in enumerate(self.projection_pairs_rotation):
                g_q[self.nla_g_trans + i, :nq1] = A_IJ2[:, b] @ A_IJ1_q1[:, a]
                g_q[self.nla_g_trans + i, nq1:] = A_IJ1[:, a] @ A_IJ2_q2[:, b]

        return g_q

        # g_q_num = approx_fprime(
        #     q, lambda q: self.g(t, q), method="cs", eps=1e-12
        # )
        # diff = g_q - g_q_num
        # error = np.linalg.norm(diff)
        # print(f"error g_q: {error}")
        # return g_q_num

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g, dtype=np.common_type(q, u))

        A_IJ1 = self.A_IJ1(t, q)
        Omega1 = self.Omega1(t, q, u)
        if self.constrain_translation:
            r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
            v_J1J2 = self.v_J2(t, q, u) - self.v_J1(t, q, u)
            for i, ax in enumerate(self.constrained_axes_displacement):
                g_dot[i] = A_IJ1[:, ax] @ v_J1J2 + cross3(A_IJ1[:, ax], r_J1J2) @ Omega1

        if self.constrain_orientation:
            A_IJ2 = self.A_IJ2(t, q)
            Omega21 = Omega1 - self.Omega2(t, q, u)

            for i, (a, b) in enumerate(self.projection_pairs_rotation):
                n = cross3(A_IJ1[:, a], A_IJ2[:, b])
                g_dot[self.nla_g_trans + i] = n @ Omega21

        return g_dot

    def g_dot_q(self, t, q, u):
        nq1 = self._nq1
        g_dot_q = np.zeros((self.nla_g, self._nq), dtype=np.common_type(q, u))

        A_IJ1 = self.A_IJ1(t, q)
        A_IJ1_q1 = self.A_IJ1_q1(t, q)
        Omega1 = self.Omega1(t, q, u)
        Omega1_q1 = self.Omega1_q1(t, q, u)
        if self.constrain_translation:
            r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
            v_J1J2 = self.v_J2(t, q, u) - self.v_J1(t, q, u)
            r_OJ1_q1 = self.r_OJ1_q1(t, q)
            r_OJ2_q2 = self.r_OJ2_q2(t, q)
            v_J1_q1 = self.v_J1_q1(t, q, u)
            v_J2_q2 = self.v_J2_q2(t, q, u)
            for i, ax in enumerate(self.constrained_axes_displacement):
                g_dot_q[i, :nq1] = (
                    -A_IJ1[:, ax] @ v_J1_q1
                    + cross3(A_IJ1[:, ax], r_J1J2) @ Omega1_q1
                    + (v_J1J2 + cross3(r_J1J2, Omega1)) @ A_IJ1_q1[:, ax]
                    - cross3(Omega1, A_IJ1[:, ax]) @ r_OJ1_q1
                )
                g_dot_q[i, nq1:] = (
                    A_IJ1[:, ax] @ v_J2_q2 + cross3(Omega1, A_IJ1[:, ax]) @ r_OJ2_q2
                )

        if self.constrain_orientation:
            A_IJ2 = self.A_IJ2(t, q)
            A_IJ2_q2 = self.A_IJ2_q2(t, q)

            Omega21 = Omega1 - self.Omega2(t, q, u)
            Omega2_q2 = self.Omega2_q2(t, q, u)

            for i, (a, b) in enumerate(self.projection_pairs_rotation):
                e_a, e_b = A_IJ1[:, a], A_IJ2[:, b]
                n = cross3(e_a, e_b)
                g_dot_q[self.nla_g_trans + i, :nq1] = (
                    n @ Omega1_q1 - Omega21 @ ax2skew(e_b) @ A_IJ1_q1[:, a]
                )
                g_dot_q[self.nla_g_trans + i, nq1:] = (
                    -n @ Omega2_q2 + Omega21 @ ax2skew(e_a) @ A_IJ2_q2[:, b]
                )

        return g_dot_q

        # g_dot_q_num = approx_fprime(
        #     q, lambda q: self.g_dot(t, q, u), method="cs", eps=1e-12
        # )
        # diff = g_dot_q - g_dot_q_num
        # error = np.linalg.norm(diff)
        # print(f"error g_dot_q: {error}")
        # return g_dot_q_num

    def g_dot_u(self, t, q):
        return self.W_g(t, q).T

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g, dtype=np.common_type(q, u, u_dot))

        A_IJ1 = self.A_IJ1(t, q)
        Omega1 = self.Omega1(t, q, u)
        Psi1 = self.Psi1(t, q, u, u_dot)
        if self.constrain_translation:
            r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
            v_J1J2 = self.v_J2(t, q, u) - self.v_J1(t, q, u)
            a_J1J2 = self.a_J2(t, q, u, u_dot) - self.a_J1(t, q, u, u_dot)
            for i, ax in enumerate(self.constrained_axes_displacement):
                e_dot = cross3(Omega1, A_IJ1[:, ax])
                g_ddot[i] = (
                    A_IJ1[:, ax] @ a_J1J2
                    + v_J1J2 @ e_dot
                    + cross3(A_IJ1[:, ax], r_J1J2) @ Psi1
                    + cross3(A_IJ1[:, ax], v_J1J2) @ Omega1
                    + cross3(e_dot, r_J1J2) @ Omega1
                )

        if self.constrain_orientation:
            A_IJ2 = self.A_IJ2(t, q)
            Omega2 = self.Omega2(t, q, u)
            Omega21 = Omega1 - Omega2
            Psi21 = Psi1 - self.Psi2(t, q, u, u_dot)

            for i, (a, b) in enumerate(self.projection_pairs_rotation):
                e_a, e_b = A_IJ1[:, a], A_IJ2[:, b]
                n = cross3(e_a, e_b)
                g_ddot[self.nla_g_trans + i] = (
                    cross3(cross3(Omega1, e_a), e_b) + cross3(e_a, cross3(Omega2, e_b))
                ) @ Omega21 + n @ Psi21

        return g_ddot

    def W_g(self, t, q):
        nu1 = self._nu1
        W_g = np.zeros((self._nu, self.nla_g), dtype=q.dtype)

        A_IJ1 = self.A_IJ1(t, q)
        J_R1 = self.J_R1(t, q)
        if self.constrain_translation:
            r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
            J_J1 = self.J_J1(t, q)
            J_J2 = self.J_J2(t, q)
            for i, ax in enumerate(self.constrained_axes_displacement):
                W_g[:nu1, i] = (
                    -A_IJ1[:, ax] @ J_J1 + cross3(A_IJ1[:, ax], r_J1J2) @ J_R1
                )
                W_g[nu1:, i] = A_IJ1[:, ax] @ J_J2

        if self.constrain_orientation:
            A_IJ2 = self.A_IJ2(t, q)
            J = np.hstack([J_R1, -self.J_R2(t, q)])

            for i, (a, b) in enumerate(self.projection_pairs_rotation):
                n = cross3(A_IJ1[:, a], A_IJ2[:, b])
                W_g[:, self.nla_g_trans + i] = n @ J

        return W_g

        # W_g_num = approx_fprime(
        #     np.zeros(self._nu), lambda u: self.g_dot(t, q, u), method="3-point", eps=1e-6
        # ).T
        # diff = W_g - W_g_num
        # error = np.linalg.norm(diff)
        # print(f"error W_g: {error}")
        # return W_g_num

    def Wla_g_q(self, t, q, la_g):
        nq1 = self._nq1
        nu1 = self._nu1
        Wla_g_q = np.zeros((self._nu, self._nq), dtype=np.common_type(q, la_g))

        A_IJ1 = self.A_IJ1(t, q)
        A_IJ1_q1 = self.A_IJ1_q1(t, q)
        J_R1 = self.J_R1(t, q)
        J_R1_q1 = self.J_R1_q1(t, q)
        if self.constrain_translation:
            r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
            r_OJ1_q1 = self.r_OJ1_q1(t, q)
            r_OJ2_q2 = self.r_OJ2_q2(t, q)
            J_J1 = self.J_J1(t, q)
            J_J2 = self.J_J2(t, q)
            J_J1_q1 = self.J_J1_q1(t, q)
            J_J2_q2 = self.J_J2_q2(t, q)
            for i, ax in enumerate(self.constrained_axes_displacement):
                Wla_g_q[:nu1, :nq1] += (
                    np.einsum("i,ijk->jk", -la_g[i] * A_IJ1[:, ax], J_J1_q1)
                    + np.einsum("ik,ij->jk", -la_g[i] * A_IJ1_q1[:, ax], J_J1)
                    + np.einsum(
                        "i,ijk->jk", la_g[i] * cross3(A_IJ1[:, ax], r_J1J2), J_R1_q1
                    )
                    + np.einsum(
                        "ik,ij->jk", -la_g[i] * ax2skew(A_IJ1[:, ax]) @ r_OJ1_q1, J_R1
                    )
                    + np.einsum(
                        "ik,ij->jk", -la_g[i] * ax2skew(r_J1J2) @ A_IJ1_q1[:, ax], J_R1
                    )
                )
                Wla_g_q[:nu1, nq1:] += np.einsum(
                    "ik,ij->jk", la_g[i] * ax2skew(A_IJ1[:, ax]) @ r_OJ2_q2, J_R1
                )

                Wla_g_q[nu1:, :nq1] += np.einsum(
                    "ij,ik->kj", la_g[i] * A_IJ1_q1[:, ax], J_J2
                )
                Wla_g_q[nu1:, nq1:] += np.einsum(
                    "i,ijk->jk", la_g[i] * A_IJ1[:, ax], J_J2_q2
                )

        # TODO: Compare with PositionOrientationBase
        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)

            A_IJ1_q1 = self.A_IJ1_q1(t, q)
            A_IJ2_q2 = self.A_IJ2_q2(t, q)

            J_R2 = self.J_R2(t, q)
            J_R2_q2 = self.J_R2_q2(t, q)

            for i, (a, b) in enumerate(self.projection_pairs_rotation):
                nla_g_trans = self.nla_g_trans
                e_a, e_b = A_IJ1[:, a], A_IJ2[:, b]
                n = cross3(e_a, e_b)
                n_q1 = -ax2skew(e_b) @ A_IJ1_q1[:, a]
                n_q2 = ax2skew(e_a) @ A_IJ2_q2[:, b]
                Wla_g_q[:nu1, :nq1] += np.einsum(
                    "i,ijk->jk", la_g[nla_g_trans + i] * n, J_R1_q1
                ) + np.einsum("ij,ik->kj", la_g[nla_g_trans + i] * n_q1, J_R1)
                Wla_g_q[:nu1, nq1:] += np.einsum(
                    "ij,ik->kj", la_g[nla_g_trans + i] * n_q2, J_R1
                )
                Wla_g_q[nu1:, :nq1] += np.einsum(
                    "ij,ik->kj", -la_g[nla_g_trans + i] * n_q1, J_R2
                )
                Wla_g_q[nu1:, nq1:] += np.einsum(
                    "i,ijk->jk", -la_g[nla_g_trans + i] * n, J_R2_q2
                ) + np.einsum("ij,ik->kj", -la_g[nla_g_trans + i] * n_q2, J_R2)

        return Wla_g_q

        # Wla_g_q_num = approx_fprime(
        #     # q, lambda q: self.W_g(t, q) @ la_g, method="3-point", eps=1e-6
        #     q, lambda q: self.W_g(t, q) @ la_g, method="cs", eps=1e-12
        # )
        # diff = Wla_g_q - Wla_g_q_num
        # error = np.linalg.norm(diff)
        # # if error > 1.0e-8:
        # print(f"error Wla_g_q: {error}")

        # return Wla_g_q_num

    def g_q_T_mu_q(self, t, q, mu):
        warnings.warn(
            "'ProjectedPositionOrientationBase.g_q_T_mu_q' uses numerical derivative."
        )
        return approx_fprime(q, lambda q: self.g_q(t, q).T @ mu)
