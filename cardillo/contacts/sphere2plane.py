import numpy as np
from cardillo.math import approx_fprime
from cardillo.math.algebra import cross3, ax2skew


class Sphere2Plane:
    def __init__(
        self,
        frame,
        subsystem,
        r,
        mu,
        e_N=None,
        e_F=None,
        frame_ID=np.zeros(3),
        K_r_SP=np.zeros(3),
        la_N0=None,
        la_F0=None,
    ):
        self.frame = frame
        self.subsystem = subsystem
        self.r = r
        self.mu = np.array([mu])
        self.nla_N = 1
        assert (
            self.nla_N == self.mu.size
        ), "Friction coefficient must have the same dimension as normal contact dim."

        if mu == 0:
            self.nla_F = 0
            self.NF_connectivity = [[]]
        else:
            self.nla_F = 2 * self.nla_N
            self.NF_connectivity = [[0, 1]]
            self.gamma_F = self.__gamma_F

        self.e_N = np.zeros(self.nla_N) if e_N is None else e_N * np.ones(self.nla_N)
        self.e_F = np.zeros(self.nla_F) if e_F is None else e_F * np.ones(self.nla_F)
        self.frame_ID = frame_ID

        self.r_OQ = self.frame.r_OP(0)
        self.t1t2 = self.frame.A_IK(0).T[:2]
        self.n = self.frame.A_IK(0)[:, 2]

        self.K_r_SP = K_r_SP

        self.la_N0 = np.zeros(self.nla_N) if la_N0 is None else la_N0
        self.la_F0 = np.zeros(self.nla_F) if la_F0 is None else la_F0

    def assembler_callback(self):
        qDOF = self.subsystem.local_qDOF_P(self.frame_ID)
        self.qDOF = self.subsystem.qDOF[qDOF]
        self.nq = len(self.qDOF)

        uDOF = self.subsystem.local_uDOF_P(self.frame_ID)
        self.uDOF = self.subsystem.uDOF[uDOF]
        self.nu = len(self.uDOF)

        self.r_OP = lambda t, q: self.subsystem.r_OP(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.r_OP_q = lambda t, q: self.subsystem.r_OP_q(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.v_P = lambda t, q, u: self.subsystem.v_P(
            t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.v_P_q = lambda t, q, u: self.subsystem.v_P_q(
            t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.J_P = lambda t, q: self.subsystem.J_P(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.J_P_q = lambda t, q: self.subsystem.J_P_q(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.a_P = lambda t, q, u, a: self.subsystem.a_P(
            t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.a_P_q = lambda t, q, u, a: self.subsystem.a_P_q(
            t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.a_P_u = lambda t, q, u, a: self.subsystem.a_P_u(
            t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )

        self.Omega = lambda t, q, u: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Omega(t, q, u, frame_ID=self.frame_ID)
        self.Omega_q = lambda t, q, u: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Omega_q(t, q, u, frame_ID=self.frame_ID) + np.einsum(
            "ijk,j->ik",
            self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID),
            self.subsystem.K_Omega(t, q, u, frame_ID=self.frame_ID),
        )
        self.J_R = lambda t, q: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID)
        self.J_R_q = lambda t, q: np.einsum(
            "ijl,jk->ikl",
            self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID),
            self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem.A_IK(t, q, frame_ID=self.frame_ID),
            self.subsystem.K_J_R_q(t, q, frame_ID=self.frame_ID),
        )
        self.Psi = lambda t, q, u, a: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Psi(t, q, u, a, frame_ID=self.frame_ID)
        self.Psi_q = lambda t, q, u, a: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Psi_q(t, q, u, a, frame_ID=self.frame_ID) + np.einsum(
            "ijk,j->ik",
            self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID),
            self.subsystem.K_Psi(t, q, u, a, frame_ID=self.frame_ID),
        )
        self.Psi_u = lambda t, q, u, a: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Psi_u(t, q, u, a, frame_ID=self.frame_ID)

    def g_N(self, t, q):
        return np.array([self.n @ (self.r_OP(t, q) - self.r_OQ)]) - self.r

    def g_N_q(self, t, q):
        return np.array([self.n @ self.r_OP_q(t, q)], dtype=q.dtype)

    def g_N_dot(self, t, q, u):
        return np.array([self.n @ self.v_P(t, q, u)], dtype=np.common_type(q, u))

    def g_N_dot_q(self, t, q, u):
        return np.array([self.n @ self.v_P_q(t, q, u)], dtype=np.common_type(q, u))

    def g_N_dot_u(self, t, q):
        return np.array([self.n @ self.J_P(t, q)], dtype=q.dtype)

    def W_N(self, t, q):
        return self.g_N_dot_u(t, q).T

    def g_N_ddot(self, t, q, u, u_dot):
        return np.array(
            [self.n @ self.a_P(t, q, u, u_dot)],
            dtype=np.common_type(q, u, u_dot),
        )

    def g_N_ddot_q(self, t, q, u, u_dot):
        return np.array(
            [self.n @ self.a_P_q(t, q, u, u_dot)], dtype=np.common_type(q, u, u_dot)
        )

    def g_N_ddot_u(self, t, q, u, u_dot):
        return np.array(
            [self.n @ self.a_P_u(t, q, u, u_dot)], dtype=np.common_type(q, u, u_dot)
        )

    def Wla_N_q(self, t, q, la_N):
        return la_N[0] * np.einsum("i,ijk->jk", self.n, self.J_P_q(t, q))

    def __gamma_F(self, t, q, u):
        v_C = self.v_P(t, q, u) + self.r * cross3(self.n, self.Omega(t, q, u))
        return self.t1t2 @ v_C

    def gamma_F_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.gamma_F(t, q, u))
        v_C_q = self.v_P_q(t, q, u) + self.r * ax2skew(self.n) @ self.Omega_q(t, q, u)
        return self.t1t2 @ v_C_q

    def gamma_F_dot(self, t, q, u, u_dot):
        r_PC = -self.r * self.n
        a_C = self.a_P(t, q, u, u_dot) + cross3(self.Psi(t, q, u, u_dot), r_PC)
        return self.t1t2 @ a_C

    def gamma_F_dot_q(self, t, q, u, u_dot):
        # return approx_fprime(q, lambda q: self.gamma_F_dot(t, q, u, u_dot))
        r_PC = -self.r * self.n
        a_C_q = self.a_P_q(t, q, u, u_dot) - ax2skew(r_PC) @ self.Psi_q(t, q, u, u_dot)
        return self.t1t2 @ a_C_q

    def gamma_F_dot_u(self, t, q, u, u_dot):
        # return approx_fprime(u, lambda u: self.gamma_F_dot(t, q, u, u_dot))
        r_PC = -self.r * self.n
        a_C_u = self.a_P_u(t, q, u, u_dot) - ax2skew(r_PC) @ self.Psi_u(t, q, u, u_dot)
        return self.t1t2 @ a_C_u

    def gamma_F_u(self, t, q):
        # return approx_fprime(np.zeros(self.nu), lambda u: self.gamma_F(t, q, u))
        J_C = self.J_P(t, q) + self.r * ax2skew(self.n) @ self.J_R(t, q)
        gamma_F_u = self.t1t2 @ J_C
        return gamma_F_u

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def Wla_F_q(self, t, q, la_F):
        return approx_fprime(q, lambda q: self.gamma_F_u(t, q).T @ la_F)
        # J_C_q = self.J_P_q(t, q) + self.r * np.einsum(
        #     "ij,jkl->ikl", ax2skew(self.n), self.J_R_q(t, q)
        # )
        # dense = np.einsum("i,ij,jkl->kl", la_F, self.t1t2, J_C_q)

    def export(self, sol_i, **kwargs):
        r_OP = self.r_OP(sol_i.t, sol_i.q[self.qDOF])
        n = self.n
        t1, t2 = self.t1t2
        g_N = self.g_N(sol_i.t, sol_i.q[self.qDOF])
        P_N = sol_i.P_N[self.la_NDOF]
        r_PC1 = -self.r * n
        r_QC2 = r_OP - self.r_OQ - n * (g_N + self.r)
        points = [r_OP + r_PC1, r_OP - n * (g_N + self.r)]
        cells = [("line", [[0, 1]])]
        A_IK1 = self.subsystem.A_IK(sol_i.t, sol_i.q[self.qDOF])
        A_IK2 = self.frame.A_IK(sol_i.t)
        point_data = dict(
            v_Ci=[
                self.subsystem.v_P(
                    sol_i.t,
                    sol_i.q[self.qDOF],
                    sol_i.u[self.uDOF],
                    self.frame_ID,
                    A_IK1 @ r_PC1,
                ),
                self.frame.v_P(sol_i.t, K_r_SP=A_IK2 @ r_QC2),
            ],
            Omega=[
                self.Omega(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF]),
                A_IK2 @ self.frame.K_Omega(sol_i.t),
            ],
            n=[-n, n],
            t1=[-t1, t1],
            t2=[-t2, t2],
            P_N=[P_N, P_N],
        )
        cell_data = dict(
            g_N=[[g_N]],
            g_N_dot=[[self.g_N_dot(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])]],
        )

        if hasattr(self, f"gamma_F"):
            cell_data["gamma_F"] = [
                [self.gamma_F(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])]
            ]
            P_F = sol_i.P_F[self.la_FDOF]
            point_data["P_F1"] = [P_F[0], P_F[0]]
            point_data["P_F2"] = [P_F[1], P_F[1]]

        return points, cells, point_data, cell_data
