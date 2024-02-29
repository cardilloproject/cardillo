import numpy as np

from cardillo.math import skew2ax
from cardillo.utility.check_time_derivatives import check_time_derivatives


class Frame:
    def __init__(
        self,
        r_OP=np.zeros(3),
        r_OP_t=None,
        r_OP_tt=None,
        A_IK=np.eye(3),
        A_IK_t=None,
        A_IK_tt=None,
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
        A_IK : np.array(3, 3) (callable/non-callable)
            Frame orientation.
        A_IK_t : np.array(3, 3) (callable/non-callable)
            Time derivative of frame orientation.
        A_IK_tt : np.array(3, 3) (callable/non-callable)
            Second time derivative of frame orientation.
        name : str
            Name of frame.
        """
        self.r_OP__, self.r_OP_t__, self.r_OP_tt__ = check_time_derivatives(
            r_OP, r_OP_t, r_OP_tt
        )
        self.A_IK__, self.A_IK_t__, self.A_IK_tt__ = check_time_derivatives(
            A_IK, A_IK_t, A_IK_tt
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

    def q_ddot(self, t, q, u, u_dot):
        return np.array([])

    #####################
    # auxiliary functions
    #####################
    def local_qDOF_P(self, frame_ID=None):
        return np.array([], dtype=int)

    def local_uDOF_P(self, frame_ID=None):
        return np.array([], dtype=int)

    def A_IK(self, t, q=None, frame_ID=None):
        return self.A_IK__(t)

    def A_IK_q(self, t, q=None, frame_ID=None):
        return np.array([]).reshape((3, 3, 0))

    def r_OP(self, t, q=None, frame_ID=None, B_r_CP=np.zeros(3)):
        return self.r_OP__(t) + self.A_IK__(t) @ B_r_CP

    def r_OP_q(self, t, q=None, frame_ID=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def r_OP_qq(self, t, q=None, frame_ID=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0, 0))

    def v_P(self, t, q=None, u=None, frame_ID=None, B_r_CP=np.zeros(3)):
        return self.r_OP_t__(t) + self.A_IK_t__(t) @ B_r_CP

    def v_P_q(self, t, q=None, u=None, frame_ID=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def J_P(self, t, q=None, frame_ID=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def J_P_q(self, t, q, frame_ID=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0, 0))

    def a_P(self, t, q=None, u=None, u_dot=None, frame_ID=None, B_r_CP=np.zeros(3)):
        return self.r_OP_tt__(t) + self.A_IK_tt__(t) @ B_r_CP

    def a_P_q(self, t, q=None, u=None, u_dot=None, frame_ID=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def a_P_u(self, t, q=None, u=None, u_dot=None, frame_ID=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def kappa_P(self, t, q=None, u=None, frame_ID=None, B_r_CP=np.zeros(3)):
        return self.r_OP_tt__(t)

    def kappa_P_q(self, t, q=None, u=None, frame_ID=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def kappa_P_u(self, t, q=None, u=None, frame_ID=None, B_r_CP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def K_Omega(self, t, q=None, u=None, frame_ID=None):
        K_omega_IK = self.A_IK__(t).T @ self.A_IK_t__(t)
        return skew2ax(K_omega_IK)

    def K_Omega_q(self, t, q=None, u=None, frame_ID=None):
        return np.array([]).reshape((3, 0))

    def K_J_R(self, t, q, frame_ID=None):
        return np.array([]).reshape((3, 0))

    def K_J_R_q(self, t, q=None, frame_ID=None):
        return np.array([]).reshape((3, 0, 0))

    def K_Psi(self, t, q=None, u=None, u_dot=None, frame_ID=None):
        K_psi_IK = self.A_IK_t__(t).T @ self.A_IK_t__(t) + self.A_IK__(
            t
        ).T @ self.A_IK_tt__(t)
        return skew2ax(K_psi_IK)

    def K_Psi_q(self, t, q=None, u=None, u_dot=None, frame_ID=None):
        return np.array([]).reshape((3, 0))

    def K_Psi_u(self, t, q=None, u=None, u_dot=None, frame_ID=None):
        return np.array([]).reshape((3, 0))

    def K_kappa_R(self, t, q=None, u=None, frame_ID=None):
        return self.K_Psi(t)

    def K_kappa_R_q(self, t, q=None, u=None, frame_ID=None):
        return np.array([]).reshape((3, 0))

    def K_kappa_R_u(self, t, q=None, u=None, frame_ID=None):
        return np.array([]).reshape((3, 0))

    ########
    # export
    ########
    def export(self, sol_i, **kwargs):
        points = [self.r_OP(sol_i.t)]
        cells = [("vertex", [[0]])]
        A_IK = np.vsplit(self.A_IK(sol_i.t).T, 3)
        cell_data = dict(
            v=[[self.v_P(sol_i.t)]],
            Omega=[[self.A_IK(sol_i.t) @ self.K_Omega(sol_i.t)]],
            ex=[A_IK[0]],
            ey=[A_IK[1]],
            ez=[A_IK[2]],
        )
        return points, cells, None, cell_data
