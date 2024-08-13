import numpy as np
from vtk import VTK_LINE

from cardillo.math.algebra import ax2skew, cross3
from cardillo.math.approx_fprime import approx_fprime
from cardillo.math.prox import Sphere


# TODO: We have to add a function that computes the correct contact forces by
# application of A @ la_F. That should be done on system level and the solver
# calls this before the converged la_F's are stored.
class Sphere2Plane:
    def __init__(
        self,
        frame,
        subsystem,
        mu,
        r=0,
        xi=None,
        B_r_CP=np.zeros(3),
        e_N=None,
        e_F=None,
        anisotropy=np.ones(2),
        name="sphere_to_plane_contact",
    ):
        """Contact between a sphere and a plane modelled as unilateral constraint with set-valued Coulomb friction.

        Parameters
        ----------
        frame : cardillo.discrete.Frame
            Frame that defines the plane. e_z-axis of frame is plane's normal direction. Origin of frame is point on plane.
        subsystem : object
            Subsystem containing the point around which the spherical contact surface is defined.
        mu : float
            Frictional coefficient
        e_N : float
            Restitution coefficient for Newton-like impact law in normal direction.
        e_N : float
            Restitution coefficient for Newton-like impact law for friction.
        xi : TODO
        B_r_CP : np.ndarray (3,)
            Position of center of sphere (P) with respect to the center of mass/reference point TODO (C) of the subsystem in the body-fixed K-basis.
        r : float
            Radius of spherical contact surface. Possible values are in [0, inf].
        anisotropy : np.ndarray (2,)
            Scaling factors for stretching the friction force reservoir in e_x and e_y-direction of the 'frame'.
            anisotropy=(1,1) corresponds to a circular force reservoir, i.e., isotropic Coulomb friction.
        name : str
            Name of contribution.
        """
        self.frame = frame
        self.subsystem = subsystem
        self.r = r
        self.name = name

        self.nla_N = 1
        self.e_N = np.zeros(self.nla_N) if e_N is None else e_N * np.ones(self.nla_N)

        if mu > 0:
            self.A = np.diag(anisotropy)
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

        self.r_OQ = self.frame.r_OP(0)
        self.t1t2 = self.frame.A_IB(0).T[:2]
        self.n = self.frame.A_IB(0)[:, 2]

        self.xi = xi
        self.B_r_CP = B_r_CP

    def assembler_callback(self):
        qDOF = self.subsystem.local_qDOF_P(self.xi)
        self.qDOF = self.subsystem.qDOF[qDOF]
        self.nq = len(self.qDOF)

        uDOF = self.subsystem.local_uDOF_P(self.xi)
        self.uDOF = self.subsystem.uDOF[uDOF]
        self.nu = len(self.uDOF)

        self.r_OP = lambda t, q: self.subsystem.r_OP(
            t, q, xi=self.xi, B_r_CP=self.B_r_CP
        )
        self.r_OP_q = lambda t, q: self.subsystem.r_OP_q(
            t, q, xi=self.xi, B_r_CP=self.B_r_CP
        )
        self.v_P = lambda t, q, u: self.subsystem.v_P(
            t, q, u, xi=self.xi, B_r_CP=self.B_r_CP
        )
        self.v_P_q = lambda t, q, u: self.subsystem.v_P_q(
            t, q, u, xi=self.xi, B_r_CP=self.B_r_CP
        )
        self.J_P = lambda t, q: self.subsystem.J_P(t, q, xi=self.xi, B_r_CP=self.B_r_CP)
        self.J_P_q = lambda t, q: self.subsystem.J_P_q(
            t, q, xi=self.xi, B_r_CP=self.B_r_CP
        )
        self.a_P = lambda t, q, u, a: self.subsystem.a_P(
            t, q, u, a, xi=self.xi, B_r_CP=self.B_r_CP
        )
        self.a_P_q = lambda t, q, u, a: self.subsystem.a_P_q(
            t, q, u, a, xi=self.xi, B_r_CP=self.B_r_CP
        )
        self.a_P_u = lambda t, q, u, a: self.subsystem.a_P_u(
            t, q, u, a, xi=self.xi, B_r_CP=self.B_r_CP
        )

        if hasattr(self.subsystem, "A_IB"):
            self.A_IB = lambda t, q: self.subsystem.A_IB(t, q, xi=self.xi)
            self.Omega = lambda t, q, u: self.subsystem.A_IB(
                t, q, xi=self.xi
            ) @ self.subsystem.B_Omega(t, q, u, xi=self.xi)
            self.Omega_q = lambda t, q, u: self.subsystem.A_IB(
                t, q, xi=self.xi
            ) @ self.subsystem.B_Omega_q(t, q, u, xi=self.xi) + np.einsum(
                "ijk,j->ik",
                self.subsystem.A_IB_q(t, q, xi=self.xi),
                self.subsystem.B_Omega(t, q, u, xi=self.xi),
            )
            self.J_R = lambda t, q: self.subsystem.A_IB(
                t, q, xi=self.xi
            ) @ self.subsystem.B_J_R(t, q, xi=self.xi)
            self.J_R_q = lambda t, q: np.einsum(
                "ijl,jk->ikl",
                self.subsystem.A_IB_q(t, q, xi=self.xi),
                self.subsystem.B_J_R(t, q, xi=self.xi),
            ) + np.einsum(
                "ij,jkl->ikl",
                self.subsystem.A_IB(t, q, xi=self.xi),
                self.subsystem.B_J_R_q(t, q, xi=self.xi),
            )
            self.Psi = lambda t, q, u, a: self.subsystem.A_IB(
                t, q, xi=self.xi
            ) @ self.subsystem.B_Psi(t, q, u, a, xi=self.xi)
            self.Psi_q = lambda t, q, u, a: self.subsystem.A_IB(
                t, q, xi=self.xi
            ) @ self.subsystem.B_Psi_q(t, q, u, a, xi=self.xi) + np.einsum(
                "ijk,j->ik",
                self.subsystem.A_IB_q(t, q, xi=self.xi),
                self.subsystem.B_Psi(t, q, u, a, xi=self.xi),
            )
            self.Psi_u = lambda t, q, u, a: self.subsystem.A_IB(
                t, q, xi=self.xi
            ) @ self.subsystem.B_Psi_u(t, q, u, a, xi=self.xi)
        else:
            self.A_IB = lambda t, q: np.eye(3)
            self.Omega = lambda t, q, u: np.zeros(3)
            self.Omega_q = lambda t, q, u: np.zeros((3, self.subsystem.nq))
            self.J_R = lambda t, q: np.zeros((self.subsystem.nu, 3))
            self.J_R_q = lambda t, q: np.zeros(
                (self.subsystem.nu, 3, self.subsystem.nq)
            )
            self.Psi = lambda t, q, u, u_dot: np.zeros(3)
            self.Psi_q = lambda t, q, u, u_dot: np.zeros((3, self.subsystem.nq))
            self.Psi_u = lambda t, q, u, u_dot: np.zeros((3, self.subsystem.nu))

    ################
    # normal contact
    ################
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

    ##########
    # friction
    ##########
    def __gamma_F(self, t, q, u):
        v_C = self.v_P(t, q, u) + self.r * cross3(self.n, self.Omega(t, q, u))
        return self.A.T @ self.t1t2 @ v_C

    def __gamma_F_q(self, t, q, u):
        # return approx_fprime(q, lambda q: self.gamma_F(t, q, u))
        v_C_q = self.v_P_q(t, q, u) + self.r * ax2skew(self.n) @ self.Omega_q(t, q, u)
        return self.A.T @ self.t1t2 @ v_C_q

    def gamma_F_dot(self, t, q, u, u_dot):
        r_PC = -self.r * self.n
        a_C = self.a_P(t, q, u, u_dot) + cross3(self.Psi(t, q, u, u_dot), r_PC)
        return self.A.T @ self.t1t2 @ a_C

    def gamma_F_dot_q(self, t, q, u, u_dot):
        # return approx_fprime(q, lambda q: self.gamma_F_dot(t, q, u, u_dot))
        r_PC = -self.r * self.n
        a_C_q = self.a_P_q(t, q, u, u_dot) - ax2skew(r_PC) @ self.Psi_q(t, q, u, u_dot)
        return self.A.T @ self.t1t2 @ a_C_q

    def gamma_F_dot_u(self, t, q, u, u_dot):
        # return approx_fprime(u, lambda u: self.gamma_F_dot(t, q, u, u_dot))
        r_PC = -self.r * self.n
        a_C_u = self.a_P_u(t, q, u, u_dot) - ax2skew(r_PC) @ self.Psi_u(t, q, u, u_dot)
        return self.A.T @ self.t1t2 @ a_C_u

    def gamma_F_u(self, t, q):
        # return approx_fprime(np.zeros(self.nu), lambda u: self.gamma_F(t, q, u))
        J_C = self.J_P(t, q) + self.r * ax2skew(self.n) @ self.J_R(t, q)
        return self.A.T @ self.t1t2 @ J_C

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def Wla_F_q(self, t, q, la_F):
        J_C_q = self.J_P_q(t, q) + self.r * np.einsum(
            "ij,jkl->ikl", ax2skew(self.n), self.J_R_q(t, q)
        )
        Wla_F_q = np.einsum("i,ij,jkl->kl", la_F, self.A.T @ self.t1t2, J_C_q)
        return Wla_F_q
        # Wla_F_q_num = approx_fprime(q, lambda q: self.gamma_F_u(t, q).T @ la_F)
        # diff = Wla_F_q - Wla_F_q_num
        # error = np.linalg.norm(diff)
        # print(f"error Wla_F_q: {error}")
        # return Wla_F_q_num

    ############
    # vtk export
    ############
    def export(self, sol_i, **kwargs):
        r_OP = self.r_OP(sol_i.t, sol_i.q[self.qDOF])
        n = self.n
        t1, t2 = self.t1t2
        g_N = self.g_N(sol_i.t, sol_i.q[self.qDOF])
        P_N = sol_i.P_N[self.la_NDOF]
        r_PC1 = -self.r * n
        r_QC2 = r_OP - self.r_OQ - n * (g_N + self.r)
        points = [r_OP + r_PC1, r_OP - n * (g_N + self.r)]
        cells = [(VTK_LINE, [0, 1])]
        A_IB1 = self.A_IB(sol_i.t, sol_i.q[self.qDOF])
        A_IB2 = self.frame.A_IB(sol_i.t)
        point_data = dict(
            v_Ci=[
                self.subsystem.v_P(
                    sol_i.t,
                    sol_i.q[self.qDOF],
                    sol_i.u[self.uDOF],
                    self.xi,
                    A_IB1.T @ r_PC1,
                ),
                self.frame.v_P(sol_i.t, B_r_CP=A_IB2.T @ r_QC2),
            ],
            Omega=[
                self.Omega(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF]),
                A_IB2 @ self.frame.B_Omega(sol_i.t),
            ],
            n=[-n, n],
            t1=[-t1, t1],
            t2=[-t2, t2],
            P_N=[P_N, P_N],
        )
        cell_data = dict(
            g_N=[g_N],
            g_N_dot=[self.g_N_dot(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])],
        )

        if hasattr(self, f"gamma_F"):
            cell_data["gamma_F"] = [
                self.gamma_F(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])
            ]
            P_F = sol_i.P_F[self.la_FDOF]
            point_data["P_F"] = np.array([P_F, P_F])

        return points, cells, point_data, cell_data
