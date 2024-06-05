import numpy as np
from vtk import VTK_LINE

from cardillo.math import norm
from cardillo.definitions import IS_CLOSE_ATOL


class TwoPointInteraction:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        xi1=None,
        xi2=None,
        B_r_CP1=np.zeros(3, dtype=float),
        B_r_CP2=np.zeros(3, dtype=float),
        name="two_point_interaction",
    ):
        r"""Interface for scalar force interaction between two points. Provides distance between the points, its time derivatives and the generalized force direction of the scalar force acting along the connection line between the two points.

        Parameters
        ----------
        subsystem1 : object
            Object containing first point of interaction (P1)
        subsystem1 : object
            Object containing second point of interaction (P2)
        xi1 : #TODO
        xi2 : #TODO
        B_r_CP1 : np.ndarray (3,)
            Position vector of first point (P1) w.r.t. center of mass (C) in body-fixed K-basis of subsystem1.
        B_r_CP2 : np.ndarray (3,)
            Position vector of second point (P2) w.r.t. center of mass (C) in body-fixed K-basis of subsystem2.
        name : str
            Name of contribution.
        """
        self.subsystem1 = subsystem1
        self.xi1 = xi1
        self.B_r_CP1 = B_r_CP1

        self.subsystem2 = subsystem2
        self.xi2 = xi2
        self.B_r_CP2 = B_r_CP2

        self.name = name

    def assembler_callback(self):
        qDOF1 = self.subsystem1.qDOF
        qDOF2 = self.subsystem2.qDOF
        local_qDOF1 = self.subsystem1.local_qDOF_P(self.xi1)
        local_qDOF2 = self.subsystem2.local_qDOF_P(self.xi2)
        self.qDOF = np.concatenate((qDOF1[local_qDOF1], qDOF2[local_qDOF2]))
        self._nq1 = len(local_qDOF1)
        self._nq2 = len(local_qDOF2)
        self._nq = self._nq1 + self._nq2
        self.t0 = self.subsystem1.t0
        q01 = self.subsystem1.q0
        q02 = self.subsystem2.q0
        self.q0 = np.concatenate((q01[local_qDOF1], q02[local_qDOF2]))

        uDOF1 = self.subsystem1.uDOF
        uDOF2 = self.subsystem2.uDOF
        local_uDOF1 = self.subsystem1.local_uDOF_P(self.xi1)
        local_uDOF2 = self.subsystem2.local_uDOF_P(self.xi2)
        self.uDOF = np.concatenate((uDOF1[local_uDOF1], uDOF2[local_uDOF2]))
        self._nu1 = len(local_uDOF1)
        self._nu2 = len(local_uDOF2)
        self._nu = self._nu1 + self._nu2
        u01 = self.subsystem1.u0
        u02 = self.subsystem2.u0
        self.u0 = np.concatenate((u01[local_uDOF1], u02[local_uDOF2]))

        self.r_OP1 = lambda t, q: self.subsystem1.r_OP(
            t, q[: self._nq1], self.xi1, self.B_r_CP1
        )
        self.r_OP1_q = lambda t, q: self.subsystem1.r_OP_q(
            t, q[: self._nq1], self.xi1, self.B_r_CP1
        )
        self.J_P1 = lambda t, q: self.subsystem1.J_P(
            t, q[: self._nq1], self.xi1, self.B_r_CP1
        )
        self.J_P1_q = lambda t, q: self.subsystem1.J_P_q(
            t, q[: self._nq1], self.xi1, self.B_r_CP1
        )
        self.v_P1 = lambda t, q, u: self.subsystem1.v_P(
            t, q[: self._nq1], u[: self._nu1], self.xi1, self.B_r_CP1
        )
        self.v_P1_q = lambda t, q, u: self.subsystem1.v_P_q(
            t, q[: self._nq1], u[: self._nu1], self.xi1, self.B_r_CP1
        )

        self.r_OP2 = lambda t, q: self.subsystem2.r_OP(
            t, q[self._nq1 :], self.xi2, self.B_r_CP2
        )
        self.r_OP2_q = lambda t, q: self.subsystem2.r_OP_q(
            t, q[self._nq1 :], self.xi2, self.B_r_CP2
        )
        self.J_P2 = lambda t, q: self.subsystem2.J_P(
            t, q[self._nq1 :], self.xi2, self.B_r_CP2
        )
        self.J_P2_q = lambda t, q: self.subsystem2.J_P_q(
            t, q[self._nq1 :], self.xi2, self.B_r_CP2
        )
        self.v_P2 = lambda t, q, u: self.subsystem2.v_P(
            t, q[self._nq1 :], u[self._nu1 :], self.xi2, self.B_r_CP2
        )
        self.v_P2_q = lambda t, q, u: self.subsystem2.v_P_q(
            t, q[self._nq1 :], u[self._nu1 :], self.xi2, self.B_r_CP2
        )

        l0 = norm(self.r_OP2(self.t0, self.q0) - self.r_OP1(self.t0, self.q0))
        assert (
            l0 > IS_CLOSE_ATOL
        ), "Initial distance of two-point interaction is close to zero."

    # auxiliary functions
    def l(self, t, q):
        return norm(self.r_OP2(t, q) - self.r_OP1(t, q))

    def l_q(self, t, q):
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)

        n = self._n(t, q)
        return np.hstack((-n @ r_OP1_q, n @ r_OP2_q))

    def l_dot(self, t, q, u):
        return self._n(t, q) @ (self.v_P2(t, q, u) - self.v_P1(t, q, u))

    def l_dot_q(self, t, q, u):
        n_q1, n_q2 = self._n_q(t, q)
        n = self._n(t, q)
        v_P1 = self.v_P1(t, q, u)
        v_P2 = self.v_P2(t, q, u)
        v_P1P2 = v_P2 - v_P1

        nq1 = self._nq1
        gamma_q = np.zeros(self._nq)
        gamma_q[:nq1] = -n @ self.v_P1_q(t, q, u) + v_P1P2 @ n_q1
        gamma_q[nq1:] = n @ self.v_P2_q(t, q, u) + v_P1P2 @ n_q2
        return gamma_q

    def l_dot_u(self, t, q, u):
        n = self._n(t, q)

        nu1 = self._nu1
        l_dot_u = np.zeros(self._nu)
        l_dot_u[:nu1] = -n @ self.J_P1(t, q)
        l_dot_u[nu1:] = n @ self.J_P2(t, q)
        return l_dot_u

    def _n(self, t, q):
        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        return (r_OP2 - r_OP1) / norm(r_OP2 - r_OP1)

    def _n_q(self, t, q):
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)

        r_P1P2 = self.r_OP2(t, q) - self.r_OP1(t, q)
        g = norm(r_P1P2)
        n = r_P1P2 / g
        P = (np.eye(3) - np.outer(n, n)) / g
        n_q1 = -P @ r_OP1_q
        n_q2 = P @ r_OP2_q
        return n_q1, n_q2

    def W_l(self, t, q):
        n = self._n(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        return np.concatenate([-J_P1.T @ n, J_P2.T @ n])

    def W_l_q(self, t, q):
        nq1 = self._nq1
        nu1 = self._nu1
        n = self._n(t, q)
        n_q1, n_q2 = self._n_q(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        J_P1_q = self.J_P1_q(t, q)
        J_P2_q = self.J_P2_q(t, q)

        # dense blocks
        W_q = np.zeros((self._nu, self._nq))
        W_q[:nu1, :nq1] = -J_P1.T @ n_q1 + np.einsum("i,ijk->jk", -n, J_P1_q)
        W_q[:nu1, nq1:] = -J_P1.T @ n_q2
        W_q[nu1:, :nq1] = J_P2.T @ n_q1
        W_q[nu1:, nq1:] = J_P2.T @ n_q2 + np.einsum("i,ijk->jk", n, J_P2_q)
        return W_q

    def export(self, sol_i, **kwargs):
        points = [
            self.r_OP1(sol_i.t, sol_i.q[self.qDOF]),
            self.r_OP2(sol_i.t, sol_i.q[self.qDOF]),
        ]
        cells = [(VTK_LINE, [0, 1])]

        return points, cells, None, None
