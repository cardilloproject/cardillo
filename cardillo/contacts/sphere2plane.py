import numpy as np
from cachetools import LRUCache, cachedmethod
from cachetools.keys import hashkey
from vtk import VTK_LINE
from warnings import warn

from cardillo.constraints._base import (
    concatenate_qDOF,
    concatenate_uDOF,
    auxiliary_functions,
)
from cardillo.math.algebra import cross3
from cardillo.math.prox import Sphere

zeros3 = np.zeros(3)
eye3 = np.eye(3)


# TODO: We have to add a function that computes the correct contact forces by
# application of A @ la_F. That should be done on system level and the solver
# calls this before the converged la_F's are stored.
class Sphere2Plane:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        mu,
        radius,
        B_r_CP1=zeros3,
        B_r_CP2=zeros3,
        A_B1P=eye3,
        e_N=None,
        e_F=None,
        xi1=None,
        xi2=None,
        anisotropy=np.ones(2),
        name="sphere_to_plane_contact",
    ):
        """Contact between a sphere and a plane modelled as unilateral constraint with set-valued Coulomb friction.

        Parameters
        ----------
        subsystem1 : object
            Subsystem that defines the plane.
            e_z-axis of P-basis is plane's normal direction: A_IP = A_IB1 @ A_B1P.
            P1 is point on plane: r_OP1 = r_OC1 + A_IB1 @ B_r_CP1
        subsystem2 : object
            Subsystem containing the point P2 around which the spherical contact surface is defined.
            r_OP2 = r_OC2 + A_IB2 @ B_r_CP2
        mu : float
            Frictional coefficient
        radius : float
            Radius of spherical contact surface. Possible values are in [0, inf].
        B_r_CP1: np.ndarray (3,)
            Vector from reference point of subsystem1 to point P1 in the plane in body-fixed coordinates of subsystem1.
        B_r_CP2: np.ndarray (3,)
            Vector from reference point of subsystem2 to point P2 in the plane in body-fixed coordinates of subsystem2.
        e_N : float
            Restitution coefficient for Newton-like impact law in normal direction.
        e_N : float
            Restitution coefficient for Newton-like impact law for friction.
        xi1 : TODO
        xi2 : TODO
        anisotropy : np.ndarray (2,)
            Scaling factors for stretching the friction force reservoir in e_x and e_y-direction of the plane.
            anisotropy=(1,1) corresponds to a circular force reservoir, i.e., isotropic Coulomb friction.
        name : str
            Name of contribution.
        """
        self.subsystem1 = subsystem1
        self.subsystem2 = subsystem2
        self.xi1 = xi1
        self.xi2 = xi2
        self.B_r_CP1 = B_r_CP1
        self.B_r_CP2 = B_r_CP2
        self.A_B1P = A_B1P

        self.radius = radius
        self.name = name

        self.nla_N = 1
        self.e_N = np.zeros(self.nla_N) if e_N is None else e_N * np.ones(self.nla_N)

        if mu > 0:
            self.A = np.diag(anisotropy)
            self.nla_F = 2 * self.nla_N
            self.gamma_F = lambda t, q, u: self.A @ self._gamma(t, q, u)[:2]
            self.gamma_F_q = lambda t, q, u: self.A @ self._gamma_q(t, q, u)[:2]
            self.gamma_F_u = lambda t, q: self.A @ self._gamma_u(t, q)[:2]
            self.gamma_F_dot = (
                lambda t, q, u, u_dot: self.A @ self._gamma_dot(t, q, u, u_dot)[:2]
            )

            self.e_F = (
                np.zeros(self.nla_F) if e_F is None else e_F * np.ones(self.nla_F)
            )

            # fmt: off
            self.friction_laws = [
                ([0], [0, 1], Sphere(mu)), # Coulomb
            ]
            # fmt: on

        self.gamma_cache = LRUCache(maxsize=1)
        self.gamma_q_cache = LRUCache(maxsize=1)
        self.gamma_u_cache = LRUCache(maxsize=1)
        self.gamma_dot_cache = LRUCache(maxsize=1)

    def assembler_callback(self):
        assert hasattr(self.subsystem1, "A_IB"), "subsystem1 must have A_IB"

        # check for A_IB of subsystem 2
        B_r_CP2 = self.B_r_CP2
        if not hasattr(self.subsystem2, "A_IB"):
            if B_r_CP2 @ B_r_CP2 > 0:
                warn(
                    "subsystem2 doesn't have A_IB, but B_r_CP2 was non-zero. Setting B_r_CP2 to zero."
                )
            B_r_CP2 = zeros3

        concatenate_qDOF(self)
        concatenate_uDOF(self)
        auxiliary_functions(self, self.B_r_CP1, B_r_CP2, self.A_B1P, None)

        # overwrite for subsystem2
        if not hasattr(self.subsystem2, "A_IB"):
            Omega2_q2 = np.zeros((3, self.subsystem2.nq))
            J_R2 = np.zeros((3, self.subsystem2.nu))
            J_R2_q2 = np.zeros((3, self.subsystem2.nu, self.subsystem2.nq))

            # auxiliary functions for subsystem 2
            self.Omega2 = lambda t, q, u: zeros3
            self.Omega2_q2 = lambda t, q, u: Omega2_q2
            self.Psi2 = lambda t, q, u, u_dot: zeros3
            self.J_R2 = lambda t, q: J_R2
            self.J_R2_q2 = lambda t, q: J_R2_q2

    # methods that share implementation
    @cachedmethod(
        lambda self: self.gamma_cache,
        key=lambda self, t, q, u: hashkey(t, *q, *u),
    )
    def _gamma(self, t, q, u):
        A_IJ1 = self.A_IJ1(t, q)
        n = A_IJ1[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n

        v_J1 = self.v_J1(t, q, u)
        v_J2 = self.v_J2(t, q, u)
        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        v_C1 = v_J1 + cross3(Omega1, r_J1C1)
        v_C2 = v_J2 + cross3(Omega2, r_J2C2)

        return A_IJ1.T @ (v_C2 - v_C1)

    @cachedmethod(
        lambda self: self.gamma_q_cache,
        key=lambda self, t, q, u: hashkey(t, *q, *u),
    )
    def _gamma_q(self, t, q, u):
        A_IJ1 = self.A_IJ1(t, q)
        n = A_IJ1[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n

        v_J1 = self.v_J1(t, q, u)
        v_J2 = self.v_J2(t, q, u)
        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        v_C1 = v_J1 + cross3(Omega1, r_J1C1)
        v_C2 = v_J2 + cross3(Omega2, r_J2C2)

        # derivatives
        A_IJ1_q1 = self.A_IJ1_q1(t, q)
        n_q1 = A_IJ1_q1[:, 2, :]
        r_OJ1_q1 = self.r_OJ1_q1(t, q)
        r_OJ2_q2 = self.r_OJ2_q2(t, q)
        r_J1C1_q1 = (
            -r_OJ1_q1
            - (n @ r_J1J2) * n_q1
            - np.outer(n, r_J1J2 @ n_q1)
            + np.outer(n, n @ r_OJ1_q1)
        )
        r_J1C1_q2 = r_OJ2_q2 - np.outer(n, n @ r_OJ2_q2)
        r_J2C2_q1 = -self.radius * n_q1

        v_J1_q1 = self.v_J1_q1(t, q, u)
        v_J2_q2 = self.v_J2_q2(t, q, u)
        Omega1_q1 = self.Omega1_q1(t, q, u)
        Omega2_q2 = self.Omega2_q2(t, q, u)
        v_C1_q1 = (
            v_J1_q1
            - np.cross(r_J1C1, Omega1_q1, axis=0)
            + np.cross(Omega1, r_J1C1_q1, axis=0)
        )
        v_C1_q2 = np.cross(Omega1, r_J1C1_q2, axis=0)
        v_C2_q1 = np.cross(Omega2, r_J2C2_q1, axis=0)
        v_C2_q2 = v_J2_q2 - np.cross(r_J2C2, Omega2_q2, axis=0)

        # compute
        nq1 = self._nq1
        gamma_q = np.zeros([3, self._nq], dtype=q.dtype)
        gamma_q[:, :nq1] = A_IJ1.T @ (v_C2_q1 - v_C1_q1) + np.einsum(
            "ijk,i->jk", A_IJ1_q1, v_C2 - v_C1
        )
        gamma_q[:, nq1:] = A_IJ1.T @ (v_C2_q2 - v_C1_q2)
        return gamma_q

    @cachedmethod(
        lambda self: self.gamma_u_cache,
        key=lambda self, t, q: hashkey(t, *q),
    )
    def _gamma_u(self, t, q):
        A_IJ1 = self.A_IJ1(t, q)
        n = A_IJ1[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n

        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        J_C1 = J_J1 - np.cross(r_J1C1, J_R1, axis=0)
        J_C2 = J_J2 - np.cross(r_J2C2, J_R2, axis=0)

        nu1 = self._nu1
        gamma_u = np.zeros([3, self._nu], dtype=q.dtype)
        gamma_u[:, :nu1] = -J_C1
        gamma_u[:, nu1:] = J_C2

        return A_IJ1.T @ gamma_u

    @cachedmethod(
        lambda self: self.gamma_dot_cache,
        key=lambda self, t, q, u, u_dot: hashkey(t, *q, *u, *u_dot),
    )
    def _gamma_dot(self, t, q, u, u_dot):
        A_IJ1 = self.A_IJ1(t, q)
        n = A_IJ1[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n

        v_J1 = self.v_J1(t, q, u)
        v_J2 = self.v_J2(t, q, u)
        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        v_C1 = v_J1 + cross3(Omega1, r_J1C1)
        v_C2 = v_J2 + cross3(Omega2, r_J2C2)

        # time derivatives
        n_dot = cross3(Omega1, n)
        v_J1J2 = v_J2 - v_J1
        r_J1C1_dot = (
            v_J1J2 - n_dot * (n @ r_J1J2) - n * (n_dot @ r_J1J2) - n * (n @ v_J1J2)
        )
        r_J2C2_dot = -self.radius * n_dot

        a_J1 = self.a_J1(t, q, u, u_dot)
        a_J2 = self.a_J2(t, q, u, u_dot)
        Psi1 = self.Psi1(t, q, u, u_dot)
        Psi2 = self.Psi2(t, q, u, u_dot)
        v_C1_dot = a_J1 + cross3(Psi1, r_J1C1) + cross3(Omega1, r_J1C1_dot)
        v_C2_dot = a_J2 + cross3(Psi2, r_J2C2) + cross3(Omega2, r_J2C2_dot)

        # compute
        gamma_dot = A_IJ1.T @ (v_C2_dot - v_C1_dot - cross3(Omega1, v_C2 - v_C1))
        return gamma_dot

    def _Wla_q(self, t, q, J1_F):
        # position and orientation
        A_IJ1 = self.A_IJ1(t, q)
        n = A_IJ1[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n

        # jacobians
        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)

        # generalized force
        # Wla = [-F @ J_J1 + m1 @ J_R1, F @ J_J2 + m2 @ J_R2]
        F = A_IJ1 @ J1_F
        m1 = -cross3(r_J1C1, F)
        m2 = cross3(r_J2C2, F)

        ###############
        # derivatives #
        ###############
        # positions and orientations
        A_IJ1_q1 = self.A_IJ1_q1(t, q)
        n_q1 = A_IJ1_q1[:, 2, :]
        r_OJ1_q1 = self.r_OJ1_q1(t, q)
        r_OJ2_q2 = self.r_OJ2_q2(t, q)
        r_J1C1_q1 = (
            -r_OJ1_q1
            - (n @ r_J1J2) * n_q1
            - np.outer(n, r_J1J2 @ n_q1)
            + np.outer(n, n @ r_OJ1_q1)
        )
        r_J1C1_q2 = r_OJ2_q2 - np.outer(n, n @ r_OJ2_q2)
        r_J2C2_q1 = -self.radius * n_q1

        # jacobians
        J_J1_q1 = self.J_J1_q1(t, q)
        J_J2_q2 = self.J_J2_q2(t, q)
        J_R1_q1 = self.J_R1_q1(t, q)
        J_R2_q2 = self.J_R2_q2(t, q)

        # forces
        F_q1 = np.einsum("ijk,j->ik", A_IJ1_q1, J1_F)
        m1_q1 = -np.cross(r_J1C1, F_q1, axis=0) - np.cross(r_J1C1_q1, F, axis=0)
        m1_q2 = -np.cross(r_J1C1_q2, F, axis=0)
        m2_q1 = np.cross(r_J2C2, F_q1, axis=0) + np.cross(r_J2C2_q1, F, axis=0)

        nu1 = self._nu1
        nq1 = self._nq1
        Wla_q = np.zeros([self._nu, self._nq], dtype=q.dtype)
        Wla_q[:nu1, :nq1] = (
            -np.einsum("i,ijk->jk", F, J_J1_q1)
            - J_J1.T @ F_q1
            + np.einsum("i,ijk->jk", m1, J_R1_q1)
            + J_R1.T @ m1_q1
        )
        Wla_q[:nu1, nq1:] = J_R1.T @ m1_q2
        Wla_q[nu1:, :nq1] = J_J2.T @ F_q1 + J_R2.T @ m2_q1
        Wla_q[nu1:, nq1:] = np.einsum("i,ijk->jk", F, J_J2_q2) + np.einsum(
            "i,ijk->jk", m2, J_R2_q2
        )

        return Wla_q

    ################
    # normal contact
    ################
    def g_N(self, t, q):
        n = self.A_IJ1(t, q)[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)

        return np.array([n @ (r_OJ2 - r_OJ1)]) - self.radius

    def g_N_q(self, t, q):
        n = self.A_IJ1(t, q)[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)

        n_q1 = self.A_IJ1_q1(t, q)[:, 2, :]
        r_OJ1_q1 = self.r_OJ1_q1(t, q)
        r_OJ2_q2 = self.r_OJ2_q2(t, q)

        nq1 = self._nq1
        g_N_q = np.zeros([self.nla_N, self._nq], dtype=q.dtype)
        g_N_q[:, :nq1] = (r_OJ2 - r_OJ1) @ n_q1 - n @ r_OJ1_q1
        g_N_q[:, nq1:] = n @ r_OJ2_q2

        return g_N_q

    def g_N_dot(self, t, q, u):
        return self._gamma(t, q, u)[2:]

    def g_N_dot_q(self, t, q, u):
        return self._gamma_q(t, q, u)[2:]

    def g_N_dot_u(self, t, q):
        return self._gamma_u(t, q)[2:]

    def W_N(self, t, q):
        return self.g_N_dot_u(t, q).T

    def g_N_ddot(self, t, q, u, u_dot):
        return self._gamma_dot(t, q, u, u_dot)[2:]

    def Wla_N_q(self, t, q, la_N):
        J1_F = np.zeros(3)
        J1_F[2:] = la_N
        return self._Wla_q(t, q, J1_F)

    ##########
    # friction
    ##########
    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def Wla_F_q(self, t, q, la_F):
        J1_F = np.zeros(3)
        J1_F[:2] = self.A @ la_F
        return self._Wla_q(t, q, J1_F)

    ############
    # vtk export
    ############
    def export(self, sol_i, **kwargs):
        # extract from solution
        t = sol_i.t
        q = sol_i.q[self.qDOF]
        u = sol_i.u[self.uDOF]
        P_N = sol_i.P_N[self.la_NDOF]

        # positions and orientation
        A_IJ1 = self.A_IJ1(t, q)
        t1, t2, n = A_IJ1.T
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n
        g_N = n @ r_J1J2 - self.radius

        # velocities
        v_J1 = self.v_J1(t, q, u)
        v_J2 = self.v_J2(t, q, u)
        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        v_C1 = v_J1 + cross3(Omega1, r_J1C1)
        v_C2 = v_J2 + cross3(Omega2, r_J2C2)
        _gamma = A_IJ1.T @ (v_C2 - v_C1)

        # vtk
        points = [r_OJ1 + r_J1C1, r_OJ2 + r_J2C2]
        cells = [(VTK_LINE, [0, 1])]
        point_data = dict(
            v_Ci=[v_C1, v_C2],
            Omega=[Omega1, Omega2],
            n=[n, -n],
            t1=[t1, -t1],
            t2=[t2, -t2],
            P_N=[P_N, P_N],
        )
        cell_data = dict(
            g_N=[[g_N]],
            g_N_dot=[[_gamma[2]]],
        )

        if hasattr(self, f"gamma_F"):
            P_F = sol_i.P_F[self.la_FDOF]
            cell_data["gamma_F"] = [_gamma[:2]]
            point_data["P_F"] = np.array([P_F, P_F])

        return points, cells, point_data, cell_data
