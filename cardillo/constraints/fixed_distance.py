import numpy as np
from vtk import VTK_LINE, VTK_BEZIER_WEDGE
from cardillo.constraints._base import (
    concatenate_qDOF,
    concatenate_uDOF,
    auxiliary_functions,
)
from cardillo.math.algebra import norm, cross3
from cardillo.math.approx_fprime import approx_fprime
import warnings


class FixedDistance:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        xi1=None,
        xi2=None,
        B1_r_P1J1=np.zeros(3),
        B2_r_P2J2=np.zeros(3),
    ):
        self.nla_g = 1

        self.subsystem1 = subsystem1
        self.xi1 = xi1
        self.B1_r_P1J1 = B1_r_P1J1

        self.subsystem2 = subsystem2
        self.xi2 = xi2
        self.B2_r_P2J2 = B2_r_P2J2

    def assembler_callback(self):
        concatenate_qDOF(self)
        concatenate_uDOF(self)

        auxiliary_functions(self, self.B1_r_P1J1, self.B2_r_P2J2, np.eye(3), np.eye(3))

        q0 = np.hstack((self.subsystem1.q0, self.subsystem2.q0))
        r_OJ10 = self.r_OJ1(self.subsystem1.t0, q0)
        r_OJ20 = self.r_OJ2(self.subsystem2.t0, q0)
        self.dist = np.linalg.norm(r_OJ20 - r_OJ10)

        if self.dist < 1e-6:
            raise ValueError("FixedDistance.distance is close to zero.")

    def g(self, t, q):
        r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
        return r_J1J2 @ r_J1J2 - self.dist**2

    def g_q(self, t, q):
        r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
        r_OJ1_q = self.r_OJ1_q1(t, q)
        r_OJ2_q = self.r_OJ2_q2(t, q)
        return np.array([2 * r_J1J2 @ np.hstack([-r_OJ1_q, r_OJ2_q])], dtype=q.dtype)

    def g_dot(self, t, q, u):
        r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
        return 2 * r_J1J2 @ (self.v_J2(t, q, u) - self.v_J1(t, q, u))

    def g_dot_q(self, t, q, u):
        r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
        v_J1J2 = self.v_J2(t, q, u) - self.v_J1(t, q, u)

        g_dot_q = np.zeros((self.nla_g, self._nq), dtype=np.common_type(q, u))
        g_dot_q[:, : self._nq1] = -2 * (
            r_J1J2 @ self.v_J1_q1(t, q, u) + v_J1J2 @ self.r_OJ1_q1(t, q)
        )
        g_dot_q[:, self._nq1 :] = 2 * (
            r_J1J2 @ self.v_J2_q2(t, q, u) + v_J1J2 @ self.r_OJ2_q2(t, q)
        )

        return g_dot_q

    def g_dot_u(self, t, q):
        return self.W_g(t, q).T

    def g_ddot(self, t, q, u, u_dot):
        r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
        v_J1J2 = self.v_J2(t, q, u) - self.v_J1(t, q, u)
        a_J1J2 = self.a_J2(t, q, u, u_dot) - self.a_J1(t, q, u, u_dot)

        return 2 * v_J1J2 @ v_J1J2 + 2 * r_J1J2 @ a_J1J2

    def g_q_T_mu_q(self, t, q, mu):
        warnings.warn("FixedDistance.g_q_T_mu_q uses approx_fprime.")
        return approx_fprime(q, lambda q: self.g_q(t, q).T @ mu)

    def W_g(self, t, q):
        r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)
        return (
            2
            * np.array(
                [np.concatenate([-J_J1.T @ r_J1J2, J_J2.T @ r_J1J2])], dtype=q.dtype
            ).T
        )

    def Wla_g_q(self, t, q, la_g):
        nq1 = self._nq1
        nu1 = self._nu1
        r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
        r_OJ1_q1 = self.r_OJ1_q1(t, q)
        r_OJ2_q2 = self.r_OJ2_q2(t, q)
        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)
        J_J1_q1 = self.J_J1_q1(t, q)
        J_J2_q2 = self.J_J2_q2(t, q)

        Wla_g_q = np.zeros((self._nu, self._nq))
        Wla_g_q[:nu1, :nq1] = (
            2 * la_g * (J_J1.T @ r_OJ1_q1 - np.einsum("i,ijk->jk", r_J1J2, J_J1_q1))
        )
        Wla_g_q[:nu1, nq1:] = -2 * la_g * J_J1.T @ r_OJ2_q2
        Wla_g_q[nu1:, :nq1] = -2 * la_g * J_J2.T @ r_OJ1_q1
        Wla_g_q[nu1:, nq1:] = (
            2 * la_g * (J_J2.T @ r_OJ2_q2 + np.einsum("i,ijk->jk", r_J1J2, J_J2_q2))
        )

        return Wla_g_q

    def export(self, sol_i, base_export=True, **kwargs):
        points = [
            self.r_OJ1(sol_i.t, sol_i.q[self.qDOF]),
            self.r_OJ2(sol_i.t, sol_i.q[self.qDOF]),
        ]
        if base_export:
            cells = [(VTK_LINE, [0, 1])]
            point_data = None
            cell_data = None
        else:
            r = kwargs.pop("radius")
            a = np.cos(np.pi / 6)
            P_i = np.array(
                [
                    [0, a, 1 / 2],
                    [0, 0, 2],
                    [0, -a, 1 / 2],
                    [0, -2 * a, -1],
                    [0, 0, -1],
                    [0, 2 * a, -1],
                ]
            )  # control points
            w_i = [
                1,
                1 / 2,
                1,
                1 / 2,
                1,
                1 / 2,
                1,
                1 / 2,
                1,
                1 / 2,
                1,
                1 / 2,
            ]  # weights
            con = np.array(
                [
                    0,
                    2,
                    4,
                    6,
                    8,
                    10,
                    1,
                    3,
                    5,
                    7,
                    9,
                    11,
                ]
            )
            r_J1J2 = points[1] - points[0]
            n = r_J1J2 / norm(r_J1J2)
            tmp = cross3(n, np.array((1, 0, 0)))
            tmp = (
                tmp / norm(tmp)
                if norm(tmp) > 1e-6
                else cross3(n, np.array((0, 1, 0)))
                / norm(cross3(n, np.array((0, 1, 0))))
            )
            T2 = cross3(n, tmp)
            t2 = T2 / norm(T2)
            T1 = cross3(t2, n)
            t1 = T1 / norm(T1)
            A_IW = np.vstack((n, t1, t2)).T
            wedge_points = [p + r * (A_IW @ P_i.T).T for p in points]
            hod = [[2, 2, 1]]
            wedge_points = np.array(wedge_points).reshape((-1, 3))

            cells = [(VTK_BEZIER_WEDGE, con)]
            point_data = {"RationalWeights": np.array(w_i)[:, None]}
            cell_data = {"HigherOrderDegrees": hod}
            points = wedge_points
        return points, cells, point_data, cell_data
