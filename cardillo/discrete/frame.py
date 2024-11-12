import numpy as np
from vtk import VTK_VERTEX
from cardillo.math import skew2ax
from cardillo.utility.check_time_derivatives import check_time_derivatives


class Frame:
    def __init__(
        self,
        r_OP=np.zeros(3),
        r_OP_t=None,
        r_OP_tt=None,
        A_IB=np.eye(3),
        A_IB_t=None,
        A_IB_tt=None,
        name="frame",
        **kwargs,
    ):
        """Frame parameterized by time dependent position and orientation.

        Parameters
        ----------
        r_OP : np.array(3) (callable/non-callable)
            Frame position.
        r_OP_t : np.array(3) (callable/non-callable)
            Frame velocity.
        r_OP_tt : np.array(3) (callable/non-callable)
            Frame acceleration.
        A_IB : np.array(3, 3) (callable/non-callable)
            Frame orientation.
        A_IB_t : np.array(3, 3) (callable/non-callable)
            Time derivative of frame orientation.
        A_IB_tt : np.array(3, 3) (callable/non-callable)
            Second time derivative of frame orientation.
        name : str
            Name of frame.
        """
        self.r_OP__, self.r_OP_t__, self.r_OP_tt__ = check_time_derivatives(
            r_OP, r_OP_t, r_OP_tt
        )
        self.A_IB__, self.A_IB_t__, self.A_IB_tt__ = check_time_derivatives(
            A_IB, A_IB_t, A_IB_tt
        )

        self.nq = 0
        self.nu = 0
        self.q0 = np.array([])
        self.u0 = np.array([])

        self.name = name

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return np.array([])

    #####################
    # auxiliary functions
    #####################
    def local_qDOF_P(self, xi=None):
        return np.array([], dtype=int)

    def local_uDOF_P(self, xi=None):
        return np.array([], dtype=int)

    def A_IB(self, t, q=None, xi=None):
        return self.A_IB__(t)

    def A_IB_q(self, t, q=None, xi=None):
        return np.array([]).reshape((3, 3, 0))

    def r_OP(self, t, q=None, xi=None, B_r_CP=np.zeros(3)):
        return self.r_OP__(t) + self.A_IB__(t) @ B_r_CP

    def r_OP_q(self, t, q=None, xi=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def r_OP_qq(self, t, q=None, xi=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0, 0))

    def v_P(self, t, q=None, u=None, xi=None, B_r_CP=np.zeros(3)):
        return self.r_OP_t__(t) + self.A_IB_t__(t) @ B_r_CP

    def v_P_q(self, t, q=None, u=None, xi=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def J_P(self, t, q=None, xi=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def J_P_q(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0, 0))

    def a_P(self, t, q=None, u=None, u_dot=None, xi=None, B_r_CP=np.zeros(3)):
        return self.r_OP_tt__(t) + self.A_IB_tt__(t) @ B_r_CP

    def a_P_q(self, t, q=None, u=None, u_dot=None, xi=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def a_P_u(self, t, q=None, u=None, u_dot=None, xi=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def kappa_P(self, t, q=None, u=None, xi=None, B_r_CP=np.zeros(3)):
        return self.r_OP_tt__(t)

    def kappa_P_q(self, t, q=None, u=None, xi=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def kappa_P_u(self, t, q=None, u=None, xi=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def B_Omega(self, t, q=None, u=None, xi=None):
        B_omega_IB = self.A_IB__(t).T @ self.A_IB_t__(t)
        return skew2ax(B_omega_IB)

    def B_Omega_q(self, t, q=None, u=None, xi=None):
        return np.array([]).reshape((3, 0))

    def B_J_R(self, t, q, xi=None):
        return np.array([]).reshape((3, 0))

    def B_J_R_q(self, t, q=None, xi=None):
        return np.array([]).reshape((3, 0, 0))

    def B_Psi(self, t, q=None, u=None, u_dot=None, xi=None):
        B_psi_IB = self.A_IB_t__(t).T @ self.A_IB_t__(t) + self.A_IB__(
            t
        ).T @ self.A_IB_tt__(t)
        return skew2ax(B_psi_IB)

    def B_Psi_q(self, t, q=None, u=None, u_dot=None, xi=None):
        return np.array([]).reshape((3, 0))

    def B_Psi_u(self, t, q=None, u=None, u_dot=None, xi=None):
        return np.array([]).reshape((3, 0))

    def B_kappa_R(self, t, q=None, u=None, xi=None):
        return self.B_Psi(t)

    def B_kappa_R_q(self, t, q=None, u=None, xi=None):
        return np.array([]).reshape((3, 0))

    def B_kappa_R_u(self, t, q=None, u=None, xi=None):
        return np.array([]).reshape((3, 0))

    ########
    # export
    ########
    def export(self, sol_i, **kwargs):
        points = [self.r_OP(sol_i.t)]
        cells = [(VTK_VERTEX, [0])]
        ex, ey, ez = self.A_IB(sol_i.t).T
        cell_data = dict(
            v=[self.v_P(sol_i.t)],
            Omega=[self.A_IB(sol_i.t) @ self.B_Omega(sol_i.t)],
            ex=[ex],
            ey=[ey],
            ez=[ez],
        )
        return points, cells, None, cell_data
