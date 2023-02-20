import numpy as np
import numpy.typing as npt
from typing import Optional


class PointMass:
    def __init__(
        self,
        m: float,
        q0: npt.ArrayLike,
        u0: Optional[npt.ArrayLike] = None,
        dim: int = 3,
    ) -> None:
        self.m = m
        self.nq = dim
        self.nu = dim

        self.__M = m * np.eye(dim)

        self.q0 = np.asarray(q0)
        self.u0 = np.zeros(self.nu) if u0 is None else np.asarray(u0)

    def M(self, t, q, coo):
        coo.extend(self.__M, (self.uDOF, self.uDOF))

    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q, coo):
        coo.extend_diag(np.ones(self.nq), (self.qDOF, self.uDOF))

    def local_qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def local_uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)

    def r_OP(self, t, q, frame_ID=None, K_r_SP=None):
        r = np.zeros(3, dtype=q.dtype)
        r[: self.nq] = q
        return r

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=None):
        return np.eye(3, self.nq)

    def J_P(self, t, q, frame_ID=None, K_r_SP=None):
        return np.eye(3, self.nu, dtype=q.dtype)

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=None):
        return np.zeros((3, self.nu, self.nq))

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=None):
        v_P = np.zeros(3, dtype=np.common_type(q, u))
        v_P[: self.nq] = u
        return v_P

    def v_P_q(self, t, q, u, frame_ID=None, K_r_SP=None):
        return np.zeros((3, self.nq))

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=None):
        a_P = np.zeros(3, dtype=np.common_type(q, u, u_dot))
        a_P[: self.nq] = u_dot
        return a_P

    def a_P_q(self, t, q, u, u_dot, frame_ID=None, K_r_SP=None):
        return np.zeros((3, self.nq), dtype=np.common_type(q, u, u_dot))

    def a_P_u(self, t, q, u, u_dot, frame_ID=None, K_r_SP=None):
        return np.zeros((3, self.nu), dtype=np.common_type(q, u, u_dot))

    # export one point mass with vtk
    def export(self, sol_i, **kwargs):
        points = [self.r_OP(sol_i.t, sol_i.q[self.qDOF])]
        vel = [self.v_P(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])]
        if sol_i.u_dot is not None:
            acc = [
                self.a_P(
                    sol_i.t,
                    sol_i.q[self.qDOF],
                    sol_i.u[self.uDOF],
                    sol_i.u_dot[self.uDOF],
                )
            ]
        cells = [("vertex", [[0]])]
        if sol_i.u_dot is not None:
            cell_data = dict(v=[vel], a=[acc])
            point_data = dict(v=[vel], a=[acc])
        else:
            cell_data = dict(v=[vel])
            point_data = dict(v=[vel])
        return points, cells, point_data, cell_data
