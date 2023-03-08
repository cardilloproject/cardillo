import numpy as np
from cardillo.math import approx_fprime
from cardillo.math.algebra import cross3, ax2skew
from cardillo.constraints._base import PositionOrientationBase


class RigidConnection(PositionOrientationBase):
    def __init__(
        self,
        subsystem1,
        subsystem2,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
    ):
        super().__init__(
            subsystem1,
            subsystem2,
            r_OB0=None,
            A_IB0=None,
            constrained_axes=[0, 1, 2],
            frame_ID1=frame_ID1,
            frame_ID2=frame_ID2,
        )


class RigidConnectionCable(RigidConnection):
    """Rigid connection for cable elements. The first director of subsystem1
    is constraint such that it is orthogonal to the second and third director
    of subsystem 2, see g(t, q).

    Note:
    -----
    subsystem1 has to be the cable element!
    """

    # def _init_(self, *args, **kwargs):
    def _init_(
        self,
        subsystem1,
        subsystem2,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
        la_g0=None,
    ):
        super()._init_(
            subsystem1,
            subsystem2,
            frame_ID1,
            frame_ID2,
            la_g0,
        )

        # overwrite some quantities of the base class
        self.nla_g = 5
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

    def g(self, t, q):
        r_OB1 = self.r_OB1(t, q)
        r_OB2 = self.r_OB2(t, q)
        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, ez2 = self.A_IB2(t, q).T
        return np.concatenate([r_OB2 - r_OB1, [ex1 @ ey2, ex1 @ ez2]])

    def g_q_dense(self, t, q):
        # return approx_fprime(q, lambda q: self.g(t, q))
        nq1 = self._nq1
        g_q = np.zeros((self.nla_g, self._nq))
        g_q[:3, :nq1] = -self.r_OB1_q1(t, q)
        g_q[:3, nq1:] = self.r_OB2_q2(t, q)

        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, ez2 = self.A_IB2(t, q).T

        A_IB1_q1 = self.A_IB1_q1(t, q)
        ex1_q = A_IB1_q1[:, 0]
        A_IB2_q2 = self.A_IB2_q2(t, q)
        ey2_q = A_IB1_q1[:, 1]
        ez2_q = A_IB1_q1[:, 2]

        # return np.concatenate([r_OB2 - r_OB1, [ex1 @ ey2, ex1 @ ez2]])
        g_q[3, :nq1] = ey2 @ ex1_q
        g_q[3, nq1:] = ex1 @ ey2_q
        g_q[4, :nq1] = ez2 @ ex1_q
        g_q[4, nq1:] = ex1 @ ez2_q
        return g_q

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g)

        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, ez2 = self.A_IB2(t, q).T

        Omega21 = self.Omega1(t, q, u) - self.Omega2(t, q, u)

        # return np.concatenate([r_OB2 - r_OB1, [ex1 @ ey2, ex1 @ ez2]])
        g_dot[:3] = self.v_B2(t, q, u) - self.v_B1(t, q, u)
        g_dot[3] = cross3(ex1, ey2) @ Omega21
        g_dot[4] = cross3(ex1, ez2) @ Omega21
        return g_dot

    # TODO: implement
    def g_dot_q(self, t, q, u, coo):
        dense = approx_fprime(q, lambda q: self.g_dot(t, q, u), method="3-point")
        coo.extend(dense, (self.la_gDOF, self.qDOF))

    def g_dot_u(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q).T, (self.la_gDOF, self.uDOF))

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g)

        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, ez2 = self.A_IB2(t, q).T

        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        Omega21 = Omega1 - Omega2
        Psi21 = self.Psi1(t, q, u, u_dot) - self.Psi2(t, q, u, u_dot)

        # g_dot[:3] = self.v_B2(t, q, u) - self.v_B1(t, q, u)
        g_ddot[:3] = self.a_B2(t, q, u, u_dot) - self.a_B1(t, q, u, u_dot)

        # g_dot[3] = cross3(ex1, ey2) @ Omega21
        g_ddot[3] = (
            cross3(cross3(Omega1, ex1), ey2) @ Omega21
            + cross3(ex1, cross3(Omega2, ey2)) @ Omega21
            + cross3(ex1, ey2) @ Psi21
        )

        # g_dot[4] = cross3(ex1, ez2) @ Omega21
        g_ddot[4] = (
            cross3(cross3(Omega1, ex1), ez2) @ Omega21
            + cross3(ex1, cross3(Omega2, ez2)) @ Omega21
            + cross3(ex1, ez2) @ Psi21
        )

        return g_ddot

    def g_ddot_q(self, t, q, u, u_dot, coo):
        dense = approx_fprime(
            q, lambda q: self.g_ddot(t, q, u, u_dot), method="3-point"
        )
        coo.extend(dense, (self.la_gDOF, self.qDOF))

    def g_ddot_u(self, t, q, u, u_dot, coo):
        dense = approx_fprime(
            u, lambda u: self.g_ddot(t, q, u, u_dot), method="3-point"
        )
        coo.extend(dense, (self.la_gDOF, self.uDOF))

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g_dense(self, t, q):
        return approx_fprime(np.zeros_like(q), lambda u: self.g_dot(t, q, u)).T
        nu1 = self._nu1
        W_g = np.zeros((self._nu, self.nla_g))

        # position
        J_B1 = self.J_B1(t, q)
        J_B2 = self.J_B2(t, q)
        W_g[:nu1, :3] = -J_B1.T
        W_g[nu1:, :3] = J_B2.T

        # orientations
        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, ez2 = self.A_IB2(t, q).T
        J = np.hstack([self.J_R1(t, q), -self.J_R2(t, q)])

        W_g[:, 3] = cross3(ex1, ey2) @ J
        W_g[:, 4] = cross3(ex1, ez2) @ J
        return W_g

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        nq1 = self._nq1
        nu1 = self._nu1
        dense = np.zeros((self._nu, self._nq))

        # position
        J_B1_q = self.J_B1_q1(t, q)
        J_B2_q = self.J_B2_q2(t, q)
        dense[:nu1, :nq1] += np.einsum("i,ijk->jk", -la_g[:3], J_B1_q)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", la_g[:3], J_B2_q)

        # angular velocity
        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, ez2 = self.A_IB2(t, q).T
        A_IB1_q1 = self.A_IB1_q1(t, q)
        ex1_q = A_IB1_q1[:, 0]
        A_IB2_q2 = self.A_IB2_q2(t, q)
        ey2_q = A_IB2_q2[:, 1]
        ez2_q = A_IB2_q2[:, 2]
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        J_R1_q = self.J_R1_q1(t, q)
        J_R2_q = self.J_R2_q2(t, q)

        # W_g[:nu1, 3] la_g[3] = cross3(ex1, ey2) @ J_R1 * la_g[3]
        # W_g[nu1:, 3] la_g[3] = - cross3(ex1, ey2) @ J_R2 * la_g[3]
        n = cross3(ex1, ey2)
        n_q1 = -ax2skew(ey2) @ ex1_q
        n_q2 = ax2skew(ex1) @ ey2_q
        dense[:nu1, :nq1] += np.einsum("i,ijk->jk", la_g[3] * n, J_R1_q) + np.einsum(
            "ij,ik->kj", la_g[3] * n_q1, J_R1
        )
        dense[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[3] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum("ij,ik->kj", -la_g[3] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", -la_g[3] * n, J_R2_q) + np.einsum(
            "ij,ik->kj", -la_g[3] * n_q2, J_R2
        )

        # W_g[:, 4] = cross3(ex1, ez2) @ J
        # W_g[:nu1, 4] la_g[4] = cross3(ex1, ez2) @ J_R1 * la_g[4]
        # W_g[nu1:, 4] la_g[4] = -cross3(ex1, ez2) @ J_R2 * la_g[4]
        # # W_g[:nu1, 4] la_g[4] = cross3(ey1, ez2) @ J_R1 * la_g[4]
        # # W_g[nu1:, 4] la_g[4] = - cross3(ey1, ez2) @ J_R2 * la_g[4]
        n = cross3(ex1, ez2)
        n_q1 = -ax2skew(ez2) @ ex1_q
        n_q2 = ax2skew(ex1) @ ez2_q
        dense[:nu1, :nq1] += np.einsum("i,ijk->jk", la_g[4] * n, J_R1_q) + np.einsum(
            "ij,ik->kj", la_g[4] * n_q1, J_R1
        )
        dense[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[4] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum("ij,ik->kj", -la_g[4] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", -la_g[4] * n, J_R2_q) + np.einsum(
            "ij,ik->kj", -la_g[4] * n_q2, J_R2
        )

        coo.extend(dense, (self.uDOF, self.qDOF))
