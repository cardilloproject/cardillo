from abc import ABC, abstractmethod
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
import numpy as np
from scipy.sparse.linalg import spsolve
import warnings

from cardillo.math.algebra import norm, cross3, ax2skew
from cardillo.math.approx_fprime import approx_fprime
from cardillo.math.rotations import Log_SO3_quat, T_SO3_inv_quat, T_SO3_inv_quat_P
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.utility.check_time_derivatives import check_time_derivatives

from ._base_export import RodExportBase
from ._cross_section import CrossSectionInertias
from .discretization.lagrange import LagrangeKnotVector
from .discretization.mesh1D import Mesh1D


class CosseratRod(RodExportBase, ABC):
    def __init__(
        self,
        cross_section,
        material_model,
        nelement,
        polynomial_degree,
        nquadrature,
        Q,
        q0=None,
        u0=None,
        nquadrature_dyn=None,
        cross_section_inertias=CrossSectionInertias(),
        **kwargs,
    ):
        """Base class for Petrov-Galerkin Cosserat rod formulations with
        quaternions for the nodal orientation parametrization.

        Parameters
        ----------
        cross_section : CrossSection
            Geometric cross-section properties: area, first and second moments
            of area.
        material_model: RodMaterialModel
            Constitutive law of Cosserat rod which relates the rod strain
            measures B_Gamma and B_Kappa with the contact forces B_n and couples
            B_m in the cross-section-fixed K-basis.
        nelement : int
            Number of rod elements.
        polynomial_degree : int
            Polynomial degree of ansatz and test functions.
        nquadrature : int
            Number of quadrature points.
        Q : np.ndarray (self.nq,)
            Generalized position coordinates of rod in a stress-free reference
            state. Q is a collection of nodal generalized position coordinates,
            which are given by the Cartesian coordinates of the nodal centerline
            point r_OP_i in R^3 together with non-unit quaternions p_i in R^4
            representing the nodal cross-section orientation.
        q0 : np.ndarray (self.nq,)
            Initial generalized position coordinates of rod at time t0.
        u0 : np.ndarray (self.nu,)
            Initial generalized velocity coordinates of rod at time t0.
            Generalized velocity coordinates u0 is a collection of the nodal
            generalized velocity coordinates, which are given by the nodal
            centerline velocity v_P_i in R^3 together with the cross-section
            angular velocity represented in the cross-section-fixed K-basis
            B_omega_IK.
        nquadrature_dyn : int
            Number of quadrature points to integrate dynamical and external
            virtual work functionals.
        cross_section_inertias : CrossSectionInertias
            Inertia properties of cross-sections: Cross-section mass density and
            Cross-section inertia tensor represented in the cross-section-fixed
            K-Basis.

        """
        # call base class for all export properties
        super().__init__(cross_section)

        # rod properties
        self.material_model = material_model
        self.cross_section_inertias = cross_section_inertias

        # distinguish between inertia quadrature and other parts
        self.nquadrature = nquadrature
        if nquadrature_dyn is None:
            self.nquadrature_dyn = nquadrature
        else:
            self.nquadrature_dyn = nquadrature_dyn
        # print(f"nquadrature_dyn: {nquadrature_dyn}")
        # print(f"nquadrature: {nquadrature}")
        self.nelement = nelement

        self._eval_cache = LRUCache(maxsize=nquadrature + 10)
        self._deval_cache = LRUCache(maxsize=nquadrature + 10)

        ##############################################################
        # discretization parameters centerline (r) and orientation (p)
        ##############################################################
        # TODO: combine mesh for position and orientation fields
        self.polynomial_degree_r = polynomial_degree
        self.polynomial_degree_p = polynomial_degree
        self.knot_vector_r = LagrangeKnotVector(self.polynomial_degree_r, nelement)
        self.knot_vector_p = LagrangeKnotVector(self.polynomial_degree_p, nelement)

        # build mesh objects
        self.mesh_r = Mesh1D(
            self.knot_vector_r,
            nquadrature,
            dim_q=3,
            derivative_order=1,
            basis="Lagrange",
            quadrature="Gauss",
        )
        self.mesh_r_dyn = Mesh1D(
            self.knot_vector_r,
            nquadrature_dyn,
            dim_q=3,
            derivative_order=1,
            basis="Lagrange",
            quadrature="Gauss",
        )
        self.mesh_p = Mesh1D(
            self.knot_vector_p,
            nquadrature,
            dim_q=4,
            derivative_order=1,
            basis="Lagrange",
            quadrature="Gauss",
            dim_u=3,
        )
        self.mesh_p_dyn = Mesh1D(
            self.knot_vector_p,
            nquadrature_dyn,
            dim_q=4,
            derivative_order=1,
            basis="Lagrange",
            quadrature="Gauss",
            dim_u=3,
        )

        # total number of nodes
        self.nnodes_r = self.mesh_r.nnodes
        self.nnodes_p = self.mesh_p.nnodes

        # number of nodes per element
        self.nnodes_element_r = self.mesh_r.nnodes_per_element
        self.nnodes_element_p = self.mesh_p.nnodes_per_element

        # total number of generalized position coordinates
        self.nq_r = self.mesh_r.nq
        self.nq_p = self.mesh_p.nq
        self.nq = self.nq_r + self.nq_p

        # total number of generalized velocity coordinates
        self.nu_r = self.mesh_r.nu
        self.nu_p = self.mesh_p.nu
        self.nu = self.nu_r + self.nu_p

        # number of generalized position coordinates per element
        self.nq_element_r = self.mesh_r.nq_per_element
        self.nq_element_p = self.mesh_p.nq_per_element
        self.nq_element = self.nq_element_r + self.nq_element_p

        # number of generalized velocity coordinates per element
        self.nu_element_r = self.mesh_r.nu_per_element
        self.nu_element_p = self.mesh_p.nu_per_element
        self.nu_element = self.nu_element_r + self.nu_element_p

        # global element connectivity
        # qe = q[elDOF_*[e]] "q^e = C_q,e q"
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_p = self.mesh_p.elDOF + self.nq_r
        # ue = u[elDOF_*_u[e]] "u^e = C_u,e u"
        # TODO: maybe rename to elDOF_u_r and elDOF_u_p
        self.elDOF_r_u = self.mesh_r.elDOF_u
        self.elDOF_p_u = self.mesh_p.elDOF_u + self.nu_r

        # global nodal connectivity
        self.nodalDOF_r = self.mesh_r.nodalDOF
        self.nodalDOF_p = self.mesh_p.nodalDOF + self.nq_r
        # TODO: maybe rename to nodalDOF_u_r and nodalDOF_u_r
        self.nodalDOF_r_u = self.mesh_r.nodalDOF_u
        self.nodalDOF_p_u = self.mesh_p.nodalDOF_u + self.nu_r

        # nodal connectivity on element level
        # r_OP_i^e = C_r,i^e * C_q,e q = C_r,i^e * q^e
        # r_OPi = qe[nodalDOF_element_r[i]]
        self.nodalDOF_element_r = self.mesh_r.nodalDOF_element
        self.nodalDOF_element_p = self.mesh_p.nodalDOF_element + self.nq_element_r
        # TODO: maybe rename it: switch r and u and p and u.
        # v_P_i^e = C_u,i^e * C_u,e q = C_u,i^e * u^e
        # v_Pi = ue[nodalDOF_element_r[i]]
        self.nodalDOF_element_r_u = self.mesh_r.nodalDOF_element_u
        self.nodalDOF_element_p_u = self.mesh_p.nodalDOF_element_u + self.nu_element_r

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nelement, self.nq_element), dtype=int)
        self.elDOF_u = np.zeros((nelement, self.nu_element), dtype=int)
        for el in range(nelement):
            self.elDOF[el, : self.nq_element_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_element_r :] = self.elDOF_p[el]
            self.elDOF_u[el, : self.nu_element_r] = self.elDOF_r_u[el]
            self.elDOF_u[el, self.nu_element_r :] = self.elDOF_p_u[el]

        # shape functions and their first derivatives
        self.N_r = self.mesh_r.N
        self.N_r_xi = self.mesh_r.N_xi
        self.N_p = self.mesh_p.N
        self.N_p_xi = self.mesh_p.N_xi

        self.N_r_dyn = self.mesh_r_dyn.N
        self.N_p_dyn = self.mesh_p_dyn.N

        # quadrature points
        # note: compliance equations use the same quadrature points
        self.qp = self.mesh_r.qp  # quadrature points
        self.qw = self.mesh_r.wp  # quadrature weights
        self.qp_dyn = self.mesh_r_dyn.qp  # quadrature points for dynamics
        self.qw_dyn = self.mesh_r_dyn.wp  # quadrature weights for dynamics

        # referential generalized position coordinates, initial generalized
        # position and velocity coordinates
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions_r = self.mesh_r.eval_basis
        self.basis_functions_p = self.mesh_p.eval_basis

        # unit quaternion constraints
        dim_g_S = 1
        self.nla_S = self.nnodes_p * dim_g_S
        self.nodalDOF_la_S = np.arange(self.nla_S).reshape(self.nnodes_p, dim_g_S)
        self.la_S0 = np.zeros(self.nla_S, dtype=float)

        self.set_reference_strains(self.Q)

    def set_reference_strains(self, Q):
        self.Q = Q.copy()

        # precompute values of the reference configuration in order to save
        # computation time J in Harsch2020b (5)
        self.J = np.zeros((self.nelement, self.nquadrature), dtype=float)
        self.J_dyn = np.zeros((self.nelement, self.nquadrature_dyn), dtype=float)
        # dilatation and shear strains of the reference configuration
        self.B_Gamma0 = np.zeros((self.nelement, self.nquadrature, 3), dtype=float)
        # curvature of the reference configuration
        self.B_Kappa0 = np.zeros((self.nelement, self.nquadrature, 3), dtype=float)

        for el in range(self.nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(self.nquadrature):
                # current quadrature point
                qpi = self.qp[el, i]

                # evaluate required quantities
                _, _, B_Gamma_bar, B_Kappa_bar = self._eval(
                    qe, qpi, N=self.N_r[el, i], N_xi=self.N_r_xi[el, i]
                )

                # length of reference tangential vector
                J = norm(B_Gamma_bar)

                # dilatation and shear strains
                B_Gamma = B_Gamma_bar / J

                # torsional and flexural strains
                B_Kappa = B_Kappa_bar / J

                # safe precomputed quantities for later
                self.J[el, i] = J
                self.B_Gamma0[el, i] = B_Gamma
                self.B_Kappa0[el, i] = B_Kappa

            for i in range(self.nquadrature_dyn):
                # current quadrature point
                qpi = self.qp_dyn[el, i]
                N, N_xi = self.basis_functions_r(qpi)
                # evaluate required quantities
                _, _, B_Gamma_bar, B_Kappa_bar = self._eval(qe, qpi, N, N_xi)

                # length of reference tangential vector
                self.J_dyn[el, i] = norm(B_Gamma_bar)

    @staticmethod
    def straight_configuration(
        nelement,
        L,
        polynomial_degree=1,
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
    ):
        """Compute generalized position coordinates for straight configuration."""
        nnodes = polynomial_degree * nelement + 1

        x0 = np.linspace(0, L, num=nnodes)
        y0 = np.zeros(nnodes)
        z0 = np.zeros(nnodes)
        r_OP = np.vstack((x0, y0, z0))
        for i in range(nnodes):
            r_OP[:, i] = r_OP0 + A_IB0 @ r_OP[:, i]

        # reshape generalized coordinates to nodal ordering
        q_r = r_OP.reshape(-1, order="C")

        # extract quaternion from orientation A_IB0
        p = Log_SO3_quat(A_IB0)
        q_p = np.repeat(p, nnodes)

        return np.concatenate([q_r, q_p])

    @staticmethod
    def deformed_configuration(
        nelement,
        r_OP,
        r_OP_xi,
        r_OP_xixi,
        xi1,
        polynomial_degree,
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
    ):
        """Compute generalized position coordinates for a pre-curved rod along curve r_OP."""
        nnodes_r = polynomial_degree * nelement + 1

        r_OP, r_OP_xi, r_OP_xixi = check_time_derivatives(r_OP, r_OP_xi, r_OP_xixi)

        xis = np.linspace(0, xi1, nnodes_r)

        # nodal positions and unit quaternions
        r0 = np.zeros((3, nnodes_r))
        p0 = np.zeros((4, nnodes_r))

        for i, xii in enumerate(xis):
            r0[:, i] = r_OP0 + A_IB0 @ r_OP(xii)
            A_K0K = np.zeros((3, 3))
            A_K0K[:, 0] = r_OP_xi(xii) / norm(r_OP_xi(xii))
            A_K0K[:, 1] = r_OP_xixi(xii) / norm(r_OP_xixi(xii))
            A_K0K[:, 2] = cross3(A_K0K[:, 0], A_K0K[:, 1])
            A_IB = A_IB0 @ A_K0K
            p0[:, i] = Log_SO3_quat(A_IB)

        # check for the right quaternion hemisphere
        for i in range(nnodes_r - 1):
            inner = p0[:, i] @ p0[:, i + 1]
            if inner < 0:
                p0[:, i + 1] *= -1

        # reshape generalized position coordinates to nodal ordering
        q_r = r0.reshape(-1, order="C")
        q_p = p0.reshape(-1, order="C")

        return np.concatenate([q_r, q_p])

    @staticmethod
    def straight_initial_configuration(
        nelement,
        L,
        polynomial_degree=1,
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
        v_P0=np.zeros(3, dtype=float),
        B_omega_IK0=np.zeros(3, dtype=float),
    ):
        """ "Compute initial generalized position and velocity coordinates for straight configuration"""
        nnodes = polynomial_degree * nelement + 1

        x0 = np.linspace(0, L, num=nnodes)
        y0 = np.zeros(nnodes)
        z0 = np.zeros(nnodes)

        r_OP = np.vstack((x0, y0, z0))
        for i in range(nnodes):
            r_OP[:, i] = r_OP0 + A_IB0 @ r_OP[:, i]

        # reshape generalized coordinates to nodal ordering
        q_r = r_OP.reshape(-1, order="C")

        # extract quaternion from orientation A_IB0
        p = Log_SO3_quat(A_IB0)
        q_p = np.repeat(p, nnodes)

        ################################
        # compute generalized velocities
        ################################
        # centerline velocities
        v_P = np.zeros_like(r_OP, dtype=float)
        for i in range(nnodes):
            v_P[:, i] = v_P0 + cross3(A_IB0 @ B_omega_IK0, (r_OP[:, i] - r_OP0))

        # reshape generalized velocity coordinates to nodal ordering
        u_r = v_P.reshape(-1, order="C")

        # all nodes share the same angular velocity
        u_p = np.repeat(B_omega_IK0, nnodes)

        q0 = np.concatenate([q_r, q_p])
        u0 = np.concatenate([u_r, u_p])

        return q0, u0

    def element_number(self, xi):
        """Compute element number from given xi."""
        return self.knot_vector_r.element_number(xi)[0]

    ############################
    # export of centerline nodes
    ############################
    def nodes(self, q):
        """Returns nodal position coordinates"""
        q_body = q[self.qDOF]
        return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF_r]).T

    ##################
    # abstract methods
    ##################
    @abstractmethod
    def _eval(self, qe, xi, N, N_xi):
        """Compute (r_OP, A_IB, B_Gamma_bar, B_Kappa_bar)."""
        ...

    @abstractmethod
    def _deval(self, qe, xi, N, N_xi):
        """Compute
        * r_OP
        * A_IB
        * B_Gamma_bar
        * B_Kappa_bar
        * r_OP_qe
        * A_IB_qe
        * B_Gamma_bar_qe
        * B_Kappa_bar_qe
        """
        ...

    @abstractmethod
    def A_IB(self, t, qe, xi): ...

    @abstractmethod
    def A_IB_q(self, t, qe, xi): ...

    def assembler_callback(self):
        self._M_coo()

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        # centerline part
        q_dot = np.zeros_like(q, dtype=np.common_type(q, u))

        # centerline time derivative from centerline velocities
        for node in range(self.nnodes_r):
            nodalDOF_r = self.nodalDOF_r[node]
            q_dot[nodalDOF_r] = u[nodalDOF_r]

        # quaternion time derivative from angular velocities
        for node in range(self.nnodes_p):
            nodalDOF_p = self.nodalDOF_p[node]
            nodalDOF_p_u = self.nodalDOF_p_u[node]
            p = q[nodalDOF_p]
            B_omega_IK = u[nodalDOF_p_u]
            q_dot[nodalDOF_p] = T_SO3_inv_quat(p, normalize=False) @ B_omega_IK

        return q_dot

    def q_dot_q(self, t, q, u):
        coo = CooMatrix((self.nq, self.nq))

        # orientation part
        for node in range(self.nnodes_p):
            nodalDOF_p = self.nodalDOF_p[node]
            nodalDOF_p_u = self.nodalDOF_p_u[node]
            p = q[nodalDOF_p]
            B_omega_IK = u[nodalDOF_p_u]

            coo[nodalDOF_p, nodalDOF_p] = np.einsum(
                "ijk,j->ik",
                T_SO3_inv_quat_P(p, normalize=False),
                B_omega_IK,
            )

        return coo

    def q_dot_u(self, t, q):
        coo = CooMatrix((self.nq, self.nu))

        # centerline part
        coo[range(self.nq_r), range(self.nu_r)] = np.eye(self.nq_r)

        # orientation part
        for node in range(self.nnodes_p):
            nodalDOF_p = self.nodalDOF_p[node]
            nodalDOF_p_u = self.nodalDOF_p_u[node]

            p = q[nodalDOF_p]
            p = p / norm(p)
            coo[nodalDOF_p, nodalDOF_p_u] = T_SO3_inv_quat(p, normalize=False)

        return coo

    def step_callback(self, t, q, u):
        """ "Quaternion normalization after each time step."""
        for node in range(self.nnodes_p):
            p = q[self.nodalDOF_p[node]]
            q[self.nodalDOF_p[node]] = p / norm(p)
        return q, u

    ############################
    # total energies and momenta
    ############################
    def E_pot(self, t, q):
        E_pot = 0.0
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            E_pot += self.E_pot_el(q[elDOF], el)
        return E_pot

    def E_pot_el(self, qe, el):
        E_pot_el = 0.0

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            B_Gamma0 = self.B_Gamma0[el, i]
            B_Kappa0 = self.B_Kappa0[el, i]

            # evaluate required quantities
            _, _, B_Gamma_bar, B_Kappa_bar = self._eval(
                qe, qpi, N=self.N_r[el, i], N_xi=self.N_r_xi[el, i]
            )

            # axial and shear strains
            B_Gamma = B_Gamma_bar / Ji

            # torsional and flexural strains
            B_Kappa = B_Kappa_bar / Ji

            # evaluate strain energy function
            E_pot_el += (
                self.material_model.potential(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
                * Ji
                * qwi
            )

        return E_pot_el

    def E_kin(self, t, q, u):
        E_kin = 0.0
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            qe = q[elDOF]
            ue = u[elDOF_u]

            E_kin += self.E_kin_el(qe, ue, el)

        return E_kin

    def E_kin_el(self, qe, ue, el):
        E_kin_el = 0.0

        for i in range(self.nquadrature_dyn):
            # extract reference state variables
            qwi = self.qw_dyn[el, i]
            Ji = self.J_dyn[el, i]

            v_P = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_r):
                v_P += self.N_r_dyn[el, i, node] * ue[self.nodalDOF_element_r_u[node]]

            B_omega_IK = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_p):
                B_omega_IK += (
                    self.N_p_dyn[el, i, node] * ue[self.nodalDOF_element_p_u[node]]
                )

            # delta_r A_rho0 r_ddot part
            E_kin_el += (
                0.5 * (v_P @ v_P) * self.cross_section_inertias.A_rho0 * Ji * qwi
            )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            E_kin_el += (
                0.5
                * (B_omega_IK @ self.cross_section_inertias.B_I_rho0 @ B_omega_IK)
                * Ji
                * qwi
            )

        return E_kin_el

    def linear_momentum(self, t, q, u):
        linear_momentum = np.zeros(3, dtype=float)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            qe = q[elDOF]
            ue = u[elDOF_u]

            linear_momentum += self.linear_momentum_el(qe, ue, el)

        return linear_momentum

    def linear_momentum_el(self, qe, ue, el):
        linear_momentum_el = np.zeros(3, dtype=float)

        for i in range(self.nquadrature_dyn):
            # extract reference state variables
            qwi = self.qw_dyn[el, i]
            Ji = self.J_dyn[el, i]

            v_P = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_r):
                v_P += self.N_r_dyn[el, i, node] * ue[self.nodalDOF_element_r[node]]

            linear_momentum_el += v_P * self.cross_section_inertias.A_rho0 * Ji * qwi

        return linear_momentum_el

    def angular_momentum(self, t, q, u):
        angular_momentum = np.zeros(3, dtype=float)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            qe = q[elDOF]
            ue = u[elDOF_u]

            angular_momentum += self.angular_momentum_el(t, qe, ue, el)

        return angular_momentum

    def angular_momentum_el(self, t, qe, ue, el):
        angular_momentum_el = np.zeros(3, dtype=float)

        for i in range(self.nquadrature_dyn):
            # extract reference state variables
            qpi = self.qp_dyn[el, i]
            qwi = self.qw_dyn[el, i]
            Ji = self.J_dyn[el, i]

            r_OP = np.zeros(3, dtype=float)
            v_P = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_r):
                r_OP += self.N_r_dyn[el, i, node] * qe[self.nodalDOF_element_r[node]]
                v_P += self.N_r_dyn[el, i, node] * ue[self.nodalDOF_element_r_u[node]]

            B_omega_IK = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_p):
                B_omega_IK += (
                    self.N_p_dyn[el, i, node] * ue[self.nodalDOF_element_p_u[node]]
                )

            A_IB = self.A_IB(t, qe, (qpi,))

            angular_momentum_el += (
                (
                    cross3(r_OP, v_P) * self.cross_section_inertias.A_rho0
                    + A_IB @ (self.cross_section_inertias.B_I_rho0 @ B_omega_IK)
                )
                * Ji
                * qwi
            )

        return angular_momentum_el

    #########################################
    # equations of motion
    #########################################
    def M(self, t, q):
        return self.__M

    def _M_coo(self):
        """ "Mass matrix is called in assembler callback."""
        self.__M = CooMatrix((self.nu, self.nu))
        for el in range(self.nelement):
            # extract element degrees of freedom
            elDOF_u = self.elDOF_u[el]

            # sparse assemble element mass matrix
            self.__M[elDOF_u, elDOF_u] = self.M_el(el)

    def M_el(self, el):
        M_el = np.zeros((self.nu_element, self.nu_element), dtype=float)

        for i in range(self.nquadrature_dyn):
            # extract reference state variables
            qwi = self.qw_dyn[el, i]
            Ji = self.J_dyn[el, i]

            # delta_r A_rho0 r_ddot part
            M_el_r_r = np.eye(3) * self.cross_section_inertias.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N_r_dyn[el, i, node_a] * self.N_r_dyn[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_p_p = self.cross_section_inertias.B_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_p):
                nodalDOF_a = self.nodalDOF_element_p_u[node_a]
                for node_b in range(self.nnodes_element_p):
                    nodalDOF_b = self.nodalDOF_element_p_u[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_p_p * (
                        self.N_p_dyn[el, i, node_a] * self.N_p_dyn[el, i, node_b]
                    )

        return M_el

    def h(self, t, q, u):
        h = np.zeros(self.nu, dtype=np.common_type(q, u))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            h[elDOF_u] += self.f_int_el(q[elDOF], el) - self.f_gyr_el(
                t, q[elDOF], u[elDOF_u], el
            )
        return h

    def h_q(self, t, q, u):
        coo = CooMatrix((self.nu, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            coo[elDOF_u, elDOF] = self.f_int_el_qe(q[elDOF], el)
        return coo

    def h_u(self, t, q, u):
        coo = CooMatrix((self.nu, self.nu))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            coo[elDOF_u, elDOF_u] = -self.f_gyr_el_ue(t, q[elDOF], u[elDOF_u], el)
        return coo

    def f_int_el(self, qe, el):
        f_int_el = np.zeros(self.nu_element, dtype=qe.dtype)

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            J = self.J[el, i]
            B_Gamma0 = self.B_Gamma0[el, i]
            B_Kappa0 = self.B_Kappa0[el, i]

            # evaluate required quantities
            _, A_IB, B_Gamma_bar, B_Kappa_bar = self._eval(
                qe, qpi, N=self.N_r[el, i], N_xi=self.N_r_xi[el, i]
            )

            # axial and shear strains
            B_Gamma = B_Gamma_bar / J

            # torsional and flexural strains
            B_Kappa = B_Kappa_bar / J

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            B_n = self.material_model.B_n(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
            B_m = self.material_model.B_m(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)

            ############################
            # virtual work contributions
            ############################
            for node in range(self.nnodes_element_r):
                f_int_el[self.nodalDOF_element_r_u[node]] -= (
                    self.N_r_xi[el, i, node] * A_IB @ B_n * qwi
                )

            for node in range(self.nnodes_element_p):
                f_int_el[self.nodalDOF_element_p_u[node]] -= (
                    self.N_p_xi[el, i, node] * B_m * qwi
                )

                f_int_el[self.nodalDOF_element_p_u[node]] += (
                    self.N_p[el, i, node]
                    * (cross3(B_Gamma_bar, B_n) + cross3(B_Kappa_bar, B_m))
                    * qwi
                )

        return f_int_el

    def f_int_el_qe(self, qe, el):
        f_int_el_qe = np.zeros((self.nu_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            B_Gamma0 = self.B_Gamma0[el, i]
            B_Kappa0 = self.B_Kappa0[el, i]

            # evaluate required quantities
            (
                r_OP,
                A_IB,
                B_Gamma_bar,
                B_Kappa_bar,
                r_OP_qe,
                A_IB_qe,
                B_Gamma_bar_qe,
                B_Kappa_bar_qe,
            ) = self._deval(qe, qpi, N=self.N_r[el, i], N_xi=self.N_r_xi[el, i])

            # axial and shear strains
            B_Gamma = B_Gamma_bar / Ji
            B_Gamma_qe = B_Gamma_bar_qe / Ji

            # torsional and flexural strains
            B_Kappa = B_Kappa_bar / Ji
            B_Kappa_qe = B_Kappa_bar_qe / Ji

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            B_n = self.material_model.B_n(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
            B_n_B_Gamma = self.material_model.B_n_B_Gamma(
                B_Gamma, B_Gamma0, B_Kappa, B_Kappa0
            )
            B_n_B_Kappa = self.material_model.B_n_B_Kappa(
                B_Gamma, B_Gamma0, B_Kappa, B_Kappa0
            )
            B_n_qe = B_n_B_Gamma @ B_Gamma_qe + B_n_B_Kappa @ B_Kappa_qe

            B_m = self.material_model.B_m(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
            B_m_B_Gamma = self.material_model.B_m_B_Gamma(
                B_Gamma, B_Gamma0, B_Kappa, B_Kappa0
            )
            B_m_B_Kappa = self.material_model.B_m_B_Kappa(
                B_Gamma, B_Gamma0, B_Kappa, B_Kappa0
            )
            B_m_qe = B_m_B_Gamma @ B_Gamma_qe + B_m_B_Kappa @ B_Kappa_qe

            ############################
            # virtual work contributions
            ############################
            for node in range(self.nnodes_element_r):
                f_int_el_qe[self.nodalDOF_element_r[node], :] -= (
                    self.N_r_xi[el, i, node]
                    * qwi
                    * (np.einsum("ikj,k->ij", A_IB_qe, B_n) + A_IB @ B_n_qe)
                )

            for node in range(self.nnodes_element_p):
                f_int_el_qe[self.nodalDOF_element_p_u[node], :] += (
                    self.N_p[el, i, node]
                    * qwi
                    * (ax2skew(B_Gamma_bar) @ B_n_qe - ax2skew(B_n) @ B_Gamma_bar_qe)
                )

                f_int_el_qe[self.nodalDOF_element_p_u[node], :] -= (
                    self.N_p_xi[el, i, node] * qwi * B_m_qe
                )

                f_int_el_qe[self.nodalDOF_element_p_u[node], :] += (
                    self.N_p[el, i, node]
                    * qwi
                    * (ax2skew(B_Kappa_bar) @ B_m_qe - ax2skew(B_m) @ B_Kappa_bar_qe)
                )

        return f_int_el_qe

    def f_gyr_el(self, t, qe, ue, el):
        f_gyr_el = np.zeros(self.nu_element, dtype=np.common_type(qe, ue))

        for i in range(self.nquadrature_dyn):
            # interpoalte angular velocity
            B_Omega = np.zeros(3, dtype=np.common_type(qe, ue))
            for node in range(self.nnodes_element_p):
                B_Omega += (
                    self.N_p_dyn[el, i, node] * ue[self.nodalDOF_element_p_u[node]]
                )

            # vector of gyroscopic forces
            f_gyr_el_p = (
                cross3(B_Omega, self.cross_section_inertias.B_I_rho0 @ B_Omega)
                * self.J_dyn[el, i]
                * self.qw_dyn[el, i]
            )

            # multiply vector of gyroscopic forces with nodal virtual rotations
            for node in range(self.nnodes_element_p):
                f_gyr_el[self.nodalDOF_element_p_u[node]] += (
                    self.N_p_dyn[el, i, node] * f_gyr_el_p
                )

        return f_gyr_el

    def f_gyr_el_ue(self, t, qe, ue, el):
        f_gyr_el_ue = np.zeros((self.nu_element, self.nu_element), dtype=float)

        for i in range(self.nquadrature_dyn):
            # interpoalte angular velocity
            B_Omega = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_p):
                B_Omega += (
                    self.N_p_dyn[el, i, node] * ue[self.nodalDOF_element_p_u[node]]
                )

            # derivative of vector of gyroscopic forces
            f_gyr_u_el_p = (
                (
                    (
                        ax2skew(B_Omega) @ self.cross_section_inertias.B_I_rho0
                        - ax2skew(self.cross_section_inertias.B_I_rho0 @ B_Omega)
                    )
                )
                * self.J_dyn[el, i]
                * self.qw_dyn[el, i]
            )

            # multiply derivative of gyroscopic force vector with nodal virtual rotations
            for node_a in range(self.nnodes_element_p):
                nodalDOF_a = self.nodalDOF_element_p_u[node_a]
                for node_b in range(self.nnodes_element_p):
                    nodalDOF_b = self.nodalDOF_element_p_u[node_b]
                    f_gyr_el_ue[nodalDOF_a[:, None], nodalDOF_b] += f_gyr_u_el_p * (
                        self.N_p_dyn[el, i, node_a] * self.N_p_dyn[el, i, node_b]
                    )

        return f_gyr_el_ue

    ###########################
    # unit-quaternion condition
    ###########################
    def g_S(self, t, q):
        g_S = np.zeros(self.nla_S, dtype=q.dtype)
        for node in range(self.nnodes_p):
            p = q[self.nodalDOF_p[node]]
            g_S[self.nodalDOF_la_S[node]] = np.array([p @ p - 1])
        return g_S

    def g_S_q(self, t, q):
        coo = CooMatrix((self.nla_S, self.nq))
        for node in range(self.nnodes_p):
            nodalDOF = self.nodalDOF_p[node]
            nodalDOF_S = self.nodalDOF_la_S[node]
            p = q[nodalDOF]
            coo[nodalDOF_S, nodalDOF] = 2 * p.reshape(1, -1)
        return coo

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def elDOF_P(self, xi):
        xi = xi
        el = self.element_number(xi)
        return self.elDOF[el]

    def elDOF_P_u(self, xi):
        xi = xi
        el = self.element_number(xi)
        return self.elDOF_u[el]

    def local_qDOF_P(self, xi):
        return self.elDOF_P(xi)

    def local_uDOF_P(self, xi):
        return self.elDOF_P_u(xi)

    ##########################
    # r_OP / A_IB contribution
    ##########################
    def r_OP(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        N, N_xi = self.basis_functions_r(xi)
        r_OC, A_IB, _, _ = self._eval(qe, xi, N, N_xi)
        return r_OC + A_IB @ B_r_CP

    # TODO: Think of a faster version than using _deval
    def r_OP_q(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        # evaluate required quantities
        N, N_xi = self.basis_functions_r(xi)
        (
            r_OC,
            A_IB,
            _,
            _,
            r_OC_q,
            A_IB_q,
            _,
            _,
        ) = self._deval(qe, xi, N, N_xi)

        return r_OC_q + np.einsum("ijk,j->ik", A_IB_q, B_r_CP)

    def v_P(self, t, qe, ue, xi, B_r_CP=np.zeros(3, dtype=float)):
        N, _ = self.basis_functions_r(xi)

        # interpolate A_IB and angular velocity in K-frame
        A_IB = self.A_IB(t, qe, xi)
        B_Omega = np.zeros(3, dtype=np.common_type(qe, ue))
        for node in range(self.nnodes_element_p):
            B_Omega += N[node] * ue[self.nodalDOF_element_p_u[node]]

        # centerline velocity
        v_C = np.zeros(3, dtype=np.common_type(qe, ue))
        for node in range(self.nnodes_element_r):
            v_C += N[node] * ue[self.nodalDOF_element_r[node]]

        return v_C + A_IB @ cross3(B_Omega, B_r_CP)

    def v_P_q(self, t, qe, ue, xi, B_r_CP=np.zeros(3, dtype=float)):
        # evaluate shape functions
        N, _ = self.basis_functions_r(xi)

        # interpolate derivative of A_IB and angular velocity in K-frame
        A_IB_q = self.A_IB_q(t, qe, xi)
        B_Omega = np.zeros(3, dtype=np.common_type(qe, ue))
        for node in range(self.nnodes_element_p):
            B_Omega += N[node] * ue[self.nodalDOF_element_p_u[node]]

        v_P_q = np.einsum(
            "ijk,j->ik",
            A_IB_q,
            cross3(B_Omega, B_r_CP),
        )
        return v_P_q

    def J_P(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        # evaluate required nodal shape functions
        N, _ = self.basis_functions_r(xi)

        # transformation matrix
        A_IB = self.A_IB(t, qe, xi)

        # skew symmetric matrix of B_r_CP
        B_r_CP_tilde = ax2skew(B_r_CP)

        # interpolate centerline and axis angle contributions
        J_P = np.zeros((3, self.nu_element), dtype=qe.dtype)
        for node in range(self.nnodes_element_r):
            J_P[:, self.nodalDOF_element_r[node]] += N[node] * np.eye(3)
        for node in range(self.nnodes_element_p):
            J_P[:, self.nodalDOF_element_p_u[node]] -= N[node] * A_IB @ B_r_CP_tilde

        return J_P

    def J_P_q(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        # evaluate required nodal shape functions
        N, _ = self.basis_functions_r(xi)

        B_r_CP_tilde = ax2skew(B_r_CP)
        A_IB_q = self.A_IB_q(t, qe, xi)
        prod = np.einsum("ijl,jk", A_IB_q, B_r_CP_tilde)

        # interpolate axis angle contributions since centerline contributon is
        # zero
        J_P_q = np.zeros((3, self.nu_element, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_p):
            nodalDOF_p_u = self.nodalDOF_element_p_u[node]
            J_P_q[:, nodalDOF_p_u] -= N[node] * prod

        return J_P_q

    def a_P(self, t, qe, ue, ue_dot, xi, B_r_CP=np.zeros(3, dtype=float)):
        N, _ = self.basis_functions_r(xi)

        # interpolate orientation
        A_IB = self.A_IB(t, qe, xi)

        # centerline acceleration
        a_C = np.zeros(3, dtype=np.common_type(qe, ue, ue_dot))
        for node in range(self.nnodes_element_r):
            a_C += N[node] * ue_dot[self.nodalDOF_element_r[node]]

        # angular velocity and acceleration in K-frame
        B_Omega = self.B_Omega(t, qe, ue, xi)
        B_Psi = self.B_Psi(t, qe, ue, ue_dot, xi)

        # rigid body formular
        return a_C + A_IB @ (
            cross3(B_Psi, B_r_CP) + cross3(B_Omega, cross3(B_Omega, B_r_CP))
        )

    def a_P_q(self, t, qe, ue, ue_dot, xi, B_r_CP=None):
        B_Omega = self.B_Omega(t, qe, ue, xi)
        B_Psi = self.B_Psi(t, qe, ue, ue_dot, xi)
        a_P_q = np.einsum(
            "ijk,j->ik",
            self.A_IB_q(t, qe, xi),
            cross3(B_Psi, B_r_CP) + cross3(B_Omega, cross3(B_Omega, B_r_CP)),
        )
        return a_P_q

    def a_P_u(self, t, qe, ue, ue_dot, xi, B_r_CP=None):
        B_Omega = self.B_Omega(t, qe, ue, xi)
        local = -self.A_IB(t, qe, xi) @ (
            ax2skew(cross3(B_Omega, B_r_CP)) + ax2skew(B_Omega) @ ax2skew(B_r_CP)
        )

        N, _ = self.basis_functions_r(xi)
        a_P_u = np.zeros((3, self.nu_element), dtype=float)
        for node in range(self.nnodes_element_r):
            a_P_u[:, self.nodalDOF_element_p_u[node]] += N[node] * local

        return a_P_u

    def B_Omega(self, t, qe, ue, xi):
        """Since we use Petrov-Galerkin method we only interpolate the nodal
        angular velocities in the K-frame.
        """
        N_p, _ = self.basis_functions_p(xi)
        B_Omega = np.zeros(3, dtype=np.common_type(qe, ue))
        for node in range(self.nnodes_element_p):
            B_Omega += N_p[node] * ue[self.nodalDOF_element_p_u[node]]
        return B_Omega

    def B_Omega_q(self, t, qe, ue, xi):
        return np.zeros((3, self.nq_element), dtype=float)

    def B_J_R(self, t, qe, xi):
        N_p, _ = self.basis_functions_p(xi)
        B_J_R = np.zeros((3, self.nu_element), dtype=float)
        for node in range(self.nnodes_element_p):
            B_J_R[:, self.nodalDOF_element_p_u[node]] += N_p[node] * np.eye(
                3, dtype=float
            )
        return B_J_R

    def B_J_R_q(self, t, qe, xi):
        return np.zeros((3, self.nu_element, self.nq_element), dtype=float)

    def B_Psi(self, t, qe, ue, ue_dot, xi):
        """Since we use Petrov-Galerkin method we only interpolate the nodal
        time derivative of the angular velocities in the K-frame.
        """
        N, _ = self.basis_functions_p(xi)
        B_Psi = np.zeros(3, dtype=np.common_type(qe, ue, ue_dot))
        for node in range(self.nnodes_element_p):
            B_Psi += N[node] * ue_dot[self.nodalDOF_element_p_u[node]]
        return B_Psi

    def B_Psi_q(self, t, qe, ue, ue_dot, xi):
        return np.zeros((3, self.nq_element), dtype=float)

    def B_Psi_u(self, t, qe, ue, ue_dot, xi):
        return np.zeros((3, self.nu_element), dtype=float)

    ##############################
    # stress and strain evaluation
    ##############################
    def eval_stresses(self, t, q, la_c, la_g, xi):
        el = self.element_number(xi)
        Qe = self.Q[self.elDOF[el]]
        qe = q[self.elDOF[el]]

        N, N_xi = self.basis_functions_r(xi)
        _, _, B_Gamma_bar0, B_Kappa_bar0 = self._eval(Qe, xi, N, N_xi)

        J = norm(B_Gamma_bar0)
        B_Gamma0 = B_Gamma_bar0 / J
        B_Kappa0 = B_Kappa_bar0 / J

        _, _, B_Gamma_bar, B_Kappa_bar = self._eval(qe, xi, N, N_xi)

        B_Gamma = B_Gamma_bar / J
        B_Kappa = B_Kappa_bar / J

        B_n = self.material_model.B_n(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)
        B_m = self.material_model.B_m(B_Gamma, B_Gamma0, B_Kappa, B_Kappa0)

        return B_n, B_m

    def eval_strains(self, t, q, la_c, la_g, xi):
        el = self.element_number(xi)
        Qe = self.Q[self.elDOF[el]]
        qe = q[self.elDOF[el]]

        N, N_xi = self.basis_functions_r(xi)
        _, _, B_Gamma_bar0, B_Kappa_bar0 = self._eval(Qe, xi, N, N_xi)

        J = norm(B_Gamma_bar0)
        B_Gamma0 = B_Gamma_bar0 / J
        B_Kappa0 = B_Kappa_bar0 / J

        _, _, B_Gamma_bar, B_Kappa_bar = self._eval(qe, xi, N, N_xi)

        B_Gamma = B_Gamma_bar / J
        B_Kappa = B_Kappa_bar / J

        return B_Gamma - B_Gamma0, B_Kappa - B_Kappa0


class CosseratRodMixed(CosseratRod):
    def __init__(
        self,
        cross_section,
        material_model,
        nelement,
        polynomial_degree,
        nquadrature,
        Q,
        q0=None,
        u0=None,
        nquadrature_dyn=None,
        cross_section_inertias=CrossSectionInertias(),
        idx_mixed=np.arange(6),
    ):
        """Base class for mixed Petrov-Galerkin Cosserat rod formulations with
        quaternions for the nodal orientation parametrization.

        Parameters
        ----------
        cross_section : CrossSection
            Geometric cross-section properties: area, first and second moments
            of area.
        material_model : RodMaterialModel
            Constitutive law of Cosserat rod which relates the rod strain
            measures B_Gamma and B_Kappa with the contact forces B_n and couples
            B_m in the cross-section-fixed K-basis.
        nelement : int
            Number of rod elements.
        polynomial_degree : int
            Polynomial degree of ansatz and test functions.
        nquadrature : int
            Number of quadrature points.
        Q : np.ndarray (self.nq,)
            Generalized position coordinates of rod in a stress-free reference
            state. Q is a collection of nodal generalized position coordinates,
            which are given by the Cartesian coordinates of the nodal centerline
            point r_OP_i in R^3 together with non-unit quaternions p_i in R^4
            representing the nodal cross-section orientation.
        q0 : np.ndarray (self.nq,)
            Initial generalized position coordinates of rod at time t0.
        u0 : np.ndarray (self.nu,)
            Initial generalized velocity coordinates of rod at time t0.
            Generalized velocity coordinates u0 is a collection of the nodal
            generalized velocity coordinates, which are given by the nodal
            centerline velocity v_P_i in R^3 together with the cross-section
            angular velocity represented in the cross-section-fixed K-basis
            B_omega_IK.
        nquadrature_dyn : int
            Number of quadrature points to integrate dynamical and external
            virtual work functionals.
        cross_section_inertias : CrossSectionInertias
            Inertia properties of cross-sections: Cross-section mass density and
            Cross-section inertia tensor represented in the cross-section-fixed
            K-Basis.
        idx_mixed : array_like
            Set of numbers between 0 and 5 to indicate which stress
            contributions obtain an independent field.
            0 : n_1 axial force.
            1 : n_2 shear force in e_y^K-direction.
            2 : n_3 shear force in e_z^K-direction.
            3 : m_1 torsion.
            4 : m_2 bending moment around e_y^K-direction.
            5 : m_3 bending moment around e_z^K-direction.

        """

        # call base class CosseratRod
        super().__init__(
            cross_section,
            material_model,
            nelement,
            polynomial_degree,
            nquadrature,
            Q,
            q0=q0,
            u0=u0,
            nquadrature_dyn=nquadrature_dyn,
            cross_section_inertias=cross_section_inertias,
        )

        #######################################################
        # discretization parameters internal forces and moments
        #######################################################
        self.idx_mixed = np.array(idx_mixed)
        self.idx_n = self.idx_mixed[(self.idx_mixed < 3)]
        self.nmixed_n = len(self.idx_n)
        # construct a sieve for the mixed force field la_c to span the internal
        # force, i.e., B_n = B_n_la_c_sieve @ la_c
        self.B_n_la_c_sieve = np.zeros((3, self.nmixed_n))
        for i, ni in enumerate(self.idx_n):
            self.B_n_la_c_sieve[ni, i] = 1

        self.idx_m = self.idx_mixed[(self.idx_mixed >= 3)] - 3
        self.nmixed_m = len(self.idx_m)
        # construct a sieve for the mixed force field la_c to span the internal
        # moment, i.e., B_m = B_m_la_c_sieve @ la_c
        self.B_m_la_c_sieve = np.zeros((3, self.nmixed_m))
        for i, mi in enumerate(self.idx_m):
            self.B_m_la_c_sieve[mi, i] = 1

        self.polynomial_degree_la_c = polynomial_degree - 1
        self.knot_vector_la_c = LagrangeKnotVector(
            self.polynomial_degree_la_c, nelement
        )

        # build mesh for internal force and moment field
        self.mesh_la_c = Mesh1D(
            self.knot_vector_la_c,
            nquadrature,
            dim_q=len(self.idx_mixed),
            derivative_order=0,
            basis="Lagrange_Disc",
            quadrature="Gauss",
            dim_u=len(self.idx_mixed),
        )

        # total number of nodes
        self.nnodes_la_c = self.mesh_la_c.nnodes

        # number of nodes per element
        self.nnodes_element_la_c = self.mesh_la_c.nnodes_per_element

        # total number of compliance coordinates
        self.nla_c = self.mesh_la_c.nq

        # number of compliance coordinates per element
        self.nla_c_element = self.mesh_la_c.nq_per_element

        # global element connectivity for copliance coordinates
        self.elDOF_la_c = self.mesh_la_c.elDOF

        # global nodal connectivity
        self.nodalDOF_la_c = self.mesh_la_c.nodalDOF + self.nq_r + self.nq_p

        # nodal connectivity on element level
        self.nodalDOF_element_la_c = self.mesh_la_c.nodalDOF_element

        # shape functions and their first derivatives
        self.N_la_c = self.mesh_la_c.N

        # evaluate shape functions at specific xi
        self.basis_functions_la_c = self.mesh_la_c.eval_basis

    def assembler_callback(self):
        super().assembler_callback()
        self._c_la_c_coo()

    ########################################
    # total complementary potential energies
    ########################################
    # TODO: If there is a demand, add it to system.
    def E_comp_pot(self, t, la_c):
        E_comp_pot = 0.0
        for el in range(self.nelement):
            elDOF_la_c = self.elDOF_la_c[el]
            E_comp_pot += self.E_pot_el(la_c[elDOF_la_c], el)
        return E_comp_pot

    def E_comp_pot_el(self, la_ce, el):
        E_comp_pot_el = 0.0

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            la_c = np.zeros(self.mesh_la_c.dim_q, dtype=la_ce.dtype)
            # interpolation of internal forces and moments
            for node in range(self.nnodes_element_la_c):
                la_c_node = la_ce[self.nodalDOF_element_la_c[node]]
                la_c += self.N_la_c[el, i, node] * la_c_node

            B_n = self.B_n_la_c_sieve @ la_c[: self.nmixed_n]
            B_m = self.B_m_la_c_sieve @ la_c[self.nmixed_n :]

            # evaluate complementary strain energy function
            E_comp_pot_el += (
                self.material_model.complementary_potential(B_n, B_m) * Ji * qwi
            )

        return E_comp_pot_el

    #########################################
    # equations of motion
    #########################################
    def h(self, t, q, u):
        # h required to overwrite function of base class.
        h = np.zeros(self.nu, dtype=np.common_type(q, u))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            h[elDOF_u] -= self.f_gyr_el(t, q[elDOF], u[elDOF_u], el)
        return h

    def h_q(self, t, q, u):
        # h_q required to overwrite function of base class.
        coo = CooMatrix((self.nu, self.nq))
        return coo

    # h_u is implmented in base class.
    ############
    # compliance
    ############
    def la_c(self, t, q, u):
        # TODO: implement affine part independently and invert matrix element wise
        return spsolve(self.c_la_c().tocsr(), self.c(t, q, u, np.zeros(self.nla_c)))

    def c(self, t, q, u, la_c):
        c = np.zeros(self.nla_c, dtype=np.common_type(q, u, la_c))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_la_c = self.elDOF_la_c[el]
            c[elDOF_la_c] = self.c_el(q[elDOF], la_c[elDOF_la_c], el)
        return c

    def c_el(self, qe, la_ce, el):
        # TODO: check for speed up by using constant c_la_c matrices
        c_el = np.zeros(self.nla_c_element, dtype=np.common_type(qe, la_ce))

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            B_Gamma0 = self.B_Gamma0[el, i]
            B_Kappa0 = self.B_Kappa0[el, i]

            # evaluate required quantities
            _, _, B_Gamma_bar, B_Kappa_bar = self._eval(
                qe, qpi, N=self.N_r[el, i], N_xi=self.N_r_xi[el, i]
            )

            la_c = np.zeros(self.mesh_la_c.dim_q, dtype=la_ce.dtype)
            # interpolation of internal forces and moments
            for node in range(self.nnodes_element_la_c):
                la_c_node = la_ce[self.nodalDOF_element_la_c[node]]
                la_c += self.N_la_c[el, i, node] * la_c_node

            B_n = self.B_n_la_c_sieve @ la_c[: self.nmixed_n]
            B_m = self.B_m_la_c_sieve @ la_c[self.nmixed_n :]

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            C_n_inv = self.material_model.C_n_inv
            C_m_inv = self.material_model.C_m_inv

            # TODO: Store B_Gamma_bar0 and B_Kappa_bar0
            c_qp = np.concatenate(
                (
                    (Ji * C_n_inv @ B_n - (B_Gamma_bar - Ji * B_Gamma0)),
                    (Ji * C_m_inv @ B_m - (B_Kappa_bar - Ji * B_Kappa0)),
                )
            )

            for node in range(self.nnodes_element_la_c):
                c_el[self.nodalDOF_element_la_c[node]] += (
                    self.N_la_c[el, i, node] * c_qp[self.idx_mixed] * qwi
                )

        return c_el

    def c_la_c(self):
        return self.__c_la_c

    def _c_la_c_coo(self):
        self.__c_la_c = CooMatrix((self.nla_c, self.nla_c))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_la_c = self.elDOF_la_c[el]
            self.__c_la_c[elDOF_la_c, elDOF_la_c] = self.c_la_c_el(el)

    def c_la_c_el(self, el):
        c_la_c_el = np.zeros((self.nla_c_element, self.nla_c_element))
        for i in range(self.nquadrature):
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            C_n_inv = self.material_model.C_n_inv
            C_m_inv = self.material_model.C_m_inv

            C_inv = np.block(
                [[C_n_inv, np.zeros_like(C_n_inv)], [np.zeros_like(C_m_inv), C_m_inv]]
            )

            C_inv_sliced = C_inv[self.idx_mixed[:, None], self.idx_mixed]

            for node_la_c1 in range(self.nnodes_element_la_c):
                nodalDOF_la_c1 = self.nodalDOF_element_la_c[node_la_c1]
                for node_la_c2 in range(self.nnodes_element_la_c):
                    nodalDOF_la_c2 = self.nodalDOF_element_la_c[node_la_c2]
                    c_la_c_el[nodalDOF_la_c1[:, None], nodalDOF_la_c2] += (
                        self.N_la_c[el, i, node_la_c1]
                        * C_inv_sliced
                        * self.N_la_c[el, i, node_la_c2]
                        * Ji
                        * qwi
                    )

        return c_la_c_el

    def c_q(self, t, q, u, la_c):
        coo = CooMatrix((self.nla_c, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_la_c = self.elDOF_la_c[el]
            coo[elDOF_la_c, elDOF] = self.c_el_qe(q[elDOF], la_c[elDOF_la_c], el)
        return coo

    def c_el_qe(self, qe, la_ce, el):
        c_el_qe = np.zeros(
            (self.nla_c_element, self.nq_element), dtype=np.common_type(qe, la_ce)
        )
        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]

            # evaluate required quantities
            (
                _,
                _,
                _,
                _,
                _,
                _,
                B_Gamma_bar_qe,
                B_Kappa_bar_qe,
            ) = self._deval(qe, qpi, N=self.N_r[el, i], N_xi=self.N_r_xi[el, i])

            delta_strains_qe = np.vstack((B_Gamma_bar_qe, B_Kappa_bar_qe))

            for node in range(self.nnodes_element_la_c):
                nodalDOF_la_c = self.nodalDOF_element_la_c[node]
                c_el_qe[nodalDOF_la_c, :] -= (
                    self.N_la_c[el, i, node] * delta_strains_qe[self.idx_mixed, :] * qwi
                )

        return c_el_qe

    def W_c(self, t, q):
        coo = CooMatrix((self.nu, self.nla_c))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            elDOF_la_c = self.elDOF_la_c[el]
            coo[elDOF_u, elDOF_la_c] = self.W_c_el(q[elDOF], el)
        return coo

    def W_c_el(self, qe, el):
        W_c_el = np.zeros((self.nu_element, self.nla_c_element), dtype=qe.dtype)

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]

            # evaluate required quantities
            _, A_IB, B_Gamma_bar, B_Kappa_bar = self._eval(
                qe, qpi, N=self.N_r[el, i], N_xi=self.N_r_xi[el, i]
            )

            ############################
            # virtual work contributions
            ############################
            for node_r in range(self.nnodes_element_r):
                nodalDOF_r = self.nodalDOF_element_r_u[node_r]
                N_r_xi = self.N_r_xi[el, i, node_r]
                for node_la_c in range(self.nnodes_element_la_c):
                    nodalDOF_la_c = self.nodalDOF_element_la_c[node_la_c]
                    W_c_el[nodalDOF_r[:, None], nodalDOF_la_c[: self.nmixed_n]] -= (
                        N_r_xi
                        * A_IB[:, self.idx_n]
                        * self.N_la_c[el, i, node_la_c]
                        * qwi
                    )

            for node_p in range(self.nnodes_element_p):
                nodalDOF_p = self.nodalDOF_element_p_u[node_p]
                N_p = self.N_p[el, i, node_p]
                N_p_xi = self.N_p_xi[el, i, node_p]

                for node_la_c in range(self.nnodes_element_la_c):
                    nodalDOF_la_c = self.nodalDOF_element_la_c[node_la_c]
                    W_c_el[nodalDOF_p[:, None], nodalDOF_la_c[self.nmixed_n :]] -= (
                        (N_p_xi * np.eye(3) - N_p * ax2skew(B_Kappa_bar))[:, self.idx_m]
                        * self.N_la_c[el, i, node_la_c]
                        * qwi
                    )

                    W_c_el[nodalDOF_p[:, None], nodalDOF_la_c[: self.nmixed_n]] += (
                        N_p
                        * ax2skew(B_Gamma_bar)[:, self.idx_n]
                        * self.N_la_c[el, i, node_la_c]
                        * qwi
                    )

        return W_c_el

    def Wla_c_q(self, t, q, la_c):
        coo = CooMatrix((self.nu, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            elDOF_la_c = self.elDOF_la_c[el]
            coo[elDOF_u, elDOF] = self.Wla_c_el_qe(q[elDOF], la_c[elDOF_la_c], el)
        return coo

    def Wla_c_el_qe(self, qe, la_ce, el):
        Wla_c_el_qe = np.zeros(
            (self.nu_element, self.nq_element), dtype=np.common_type(qe, la_ce)
        )

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]

            # evaluate required quantities
            (
                r_OP,
                A_IB,
                B_Gamma_bar,
                B_Kappa_bar,
                r_OP_qe,
                A_IB_qe,
                B_Gamma_bar_qe,
                B_Kappa_bar_qe,
            ) = self._deval(qe, qpi, N=self.N_r[el, i], N_xi=self.N_r_xi[el, i])

            # interpolation of the n and m
            la_c = np.zeros(self.mesh_la_c.dim_q, dtype=qe.dtype)

            for node in range(self.nnodes_element_la_c):
                la_c_node = la_ce[self.nodalDOF_element_la_c[node]]
                la_c += self.N_la_c[el, i, node] * la_c_node

            B_n = self.B_n_la_c_sieve @ la_c[: self.nmixed_n]
            B_m = self.B_m_la_c_sieve @ la_c[self.nmixed_n :]

            for node in range(self.nnodes_element_r):
                Wla_c_el_qe[self.nodalDOF_element_r_u[node], :] -= (
                    self.N_r_xi[el, i, node]
                    * qwi
                    * (np.einsum("ikj,k->ij", A_IB_qe, B_n))
                )

            for node in range(self.nnodes_element_p):
                Wla_c_el_qe[self.nodalDOF_element_p_u[node], :] += (
                    self.N_p[el, i, node]
                    * qwi
                    * (-ax2skew(B_n) @ B_Gamma_bar_qe - ax2skew(B_m) @ B_Kappa_bar_qe)
                )

        return Wla_c_el_qe

    ##############################
    # stress and strain evaluation
    ##############################
    def eval_stresses(self, t, q, la_c, la_g, xi):
        el = self.element_number(xi)
        la_ce = la_c[self.elDOF_la_c[el]]
        # TODO: lets see how to avoid the flatten
        N_la_ce = self.basis_functions_la_c(xi).flatten()
        la_cc = np.zeros(len(self.idx_mixed))

        for node in range(self.nnodes_element_la_c):
            la_c_node = la_ce[self.nodalDOF_element_la_c[node]]
            la_cc += N_la_ce[node] * la_c_node

        B_n = self.B_n_la_c_sieve @ la_cc[: self.nmixed_n]
        B_m = self.B_m_la_c_sieve @ la_cc[self.nmixed_n :]

        return B_n, B_m

    def eval_strains(self, t, q, la_c, la_g, xi):
        B_n, B_m = self.eval_stresses(t, q, la_c, la_g, xi)
        el = self.element_number(xi)
        Qe = self.Q[self.elDOF[el]]

        N, N_xi = self.basis_functions_r(xi)
        _, _, B_Gamma_bar0, B_Kappa_bar0 = self._eval(Qe, xi, N, N_xi)

        J = norm(B_Gamma_bar0)
        B_Gamma0 = B_Gamma_bar0 / J
        B_Kappa0 = B_Kappa_bar0 / J

        C_n_inv = self.material_model.C_n_inv
        C_m_inv = self.material_model.C_m_inv

        B_Gamma = C_n_inv @ B_n + B_Gamma0
        B_Kappa = C_m_inv @ B_m + B_Kappa0

        return B_Gamma - B_Gamma0, B_Kappa - B_Kappa0


def make_CosseratRodConstrained(mixed, constraints):
    if mixed == True:
        CosseratRodBase = CosseratRodMixed
        idx = np.arange(6)
        idx_constraints = np.array(constraints)
        idx_mixed = np.setdiff1d(idx, idx_constraints)
    else:
        CosseratRodBase = CosseratRod
        idx_mixed = None

    class CosseratRodConstrained(CosseratRodBase):
        def __init__(
            self,
            cross_section,
            material_model,
            nelement,
            polynomial_degree,
            nquadrature,
            Q,
            q0=None,
            u0=None,
            nquadrature_dyn=None,
            cross_section_inertias=CrossSectionInertias(),
        ):
            """Base class for Petrov-Galerkin Cosserat rod formulations with
            additional constraints with quaternions for the nodal orientation
            parametrization.

            Parameters
            ----------
            cross_section : CrossSection
                Geometric cross-section properties: area, first and second moments
                of area.
            material_model : RodMaterialModel
                Constitutive law of Cosserat rod which relates the rod strain
                measures B_Gamma and B_Kappa with the contact forces B_n and couples
                B_m in the cross-section-fixed K-basis.
            nelement : int
                Number of rod elements.
            polynomial_degree : int
                Polynomial degree of ansatz and test functions.
            nquadrature : int
                Number of quadrature points.
            Q : np.ndarray (self.nq,)
                Generalized position coordinates of rod in a stress-free reference
                state. Q is a collection of nodal generalized position coordinates,
                which are given by the Cartesian coordinates of the nodal centerline
                point r_OP_i in R^3 together with non-unit quaternions p_i in R^4
                representing the nodal cross-section orientation.
            q0 : np.ndarray (self.nq,)
                Initial generalized position coordinates of rod at time t0.
            u0 : np.ndarray (self.nu,)
                Initial generalized velocity coordinates of rod at time t0.
                Generalized velocity coordinates u0 is a collection of the nodal
                generalized velocity coordinates, which are given by the nodal
                centerline velocity v_P_i in R^3 together with the cross-section
                angular velocity represented in the cross-section-fixed K-basis
                B_omega_IK.
            nquadrature_dyn : int
                Number of quadrature points to integrate dynamical and external
                virtual work functionals.
            cross_section_inertias : CrossSectionInertias
                Inertia properties of cross-sections: Cross-section mass density and
                Cross-section inertia tensor represented in the cross-section-fixed
                K-Basis.
            """
            super().__init__(
                cross_section,
                material_model,
                nelement,
                polynomial_degree,
                nquadrature,
                Q,
                q0=q0,
                u0=u0,
                nquadrature_dyn=nquadrature_dyn,
                cross_section_inertias=cross_section_inertias,
                idx_mixed=idx_mixed,
            )

            ##################################################################
            # discretization parameters internal constraint forces and moments
            ##################################################################
            self.constraints = np.array(constraints)
            self.constraints_gamma = self.constraints[(self.constraints < 3)]
            self.nconstraints_gamma = len(self.constraints_gamma)
            # construct a sieve for the constraint force field la_g to span the internal
            # force, i.e., B_n = B_n_la_g_sieve @ la_g
            self.B_n_la_g_sieve = np.zeros((3, self.nconstraints_gamma))
            for i, gammai in enumerate(self.constraints_gamma):
                self.B_n_la_g_sieve[gammai, i] = 1

            self.constraints_kappa = self.constraints[(self.constraints >= 3)] - 3
            self.nconstraints_kappa = len(self.constraints_kappa)
            # construct a sieve for the constraint force field la_g to span the internal
            # moment, i.e., B_m = B_m_la_g_sieve @ la_g
            self.B_m_la_g_sieve = np.zeros((3, self.nconstraints_kappa))
            for i, kappai in enumerate(self.constraints_kappa):
                self.B_m_la_g_sieve[kappai, i] = 1

            self.polynomial_degree_la_g = polynomial_degree - 1
            self.knot_vector_la_g = LagrangeKnotVector(
                self.polynomial_degree_la_g, nelement
            )

            # build mesh for internal constraint force and moment field
            self.mesh_la_g = Mesh1D(
                self.knot_vector_la_g,
                nquadrature,
                dim_q=len(self.constraints),
                derivative_order=0,
                basis="Lagrange_Disc",
                quadrature="Gauss",
                dim_u=len(self.constraints),
            )

            # total number of nodes
            self.nnodes_la_g = self.mesh_la_g.nnodes

            # number of nodes per element
            self.nnodes_element_la_g = self.mesh_la_g.nnodes_per_element

            # total number of constraint coordinates
            self.nla_g = self.mesh_la_g.nq

            # number of compliance coordinates per element
            self.nla_g_element = self.mesh_la_g.nq_per_element

            # global element connectivity for copliance coordinates
            self.elDOF_la_g = self.mesh_la_g.elDOF

            # global nodal connectivity
            # TODO: Take care of self.nla_c for the mixed formulation. Not yet correct.
            # self.nodalDOF_la_g = self.mesh_la_g.nodalDOF + self.nq_r + self.nq_p

            # nodal connectivity on element level
            self.nodalDOF_element_la_g = self.mesh_la_g.nodalDOF_element

            # shape functions
            self.N_la_g = self.mesh_la_g.N

            # TODO: Dummy initial values for compliance
            self.la_g0 = np.zeros(self.nla_g, dtype=float)

            # evaluate shape functions at specific xi
            self.basis_functions_la_g = self.mesh_la_g.eval_basis

        #########################################
        # bilateral constraints on position level
        #########################################
        def g(self, t, q):
            g = np.zeros(self.nla_g, dtype=q.dtype)
            for el in range(self.nelement):
                elDOF = self.elDOF[el]
                elDOF_la_g = self.elDOF_la_g[el]
                g[elDOF_la_g] = self.g_el(q[elDOF], el)
            return g

        def g_el(self, qe, el):
            g_el = np.zeros(self.nla_g_element, dtype=qe.dtype)

            for i in range(self.nquadrature):
                # extract reference state variables
                qpi = self.qp[el, i]
                qwi = self.qw[el, i]
                Ji = self.J[el, i]
                B_Gamma0 = self.B_Gamma0[el, i]
                B_Kappa0 = self.B_Kappa0[el, i]

                # evaluate required quantities
                _, _, B_Gamma_bar, B_Kappa_bar = self._eval(
                    qe, qpi, N=self.N_r[el, i], N_xi=self.N_r_xi[el, i]
                )

                # TODO: Store B_Gamma_bar0 and B_Kappa_bar0
                delta_strains = np.concatenate(
                    [
                        (B_Gamma_bar - Ji * B_Gamma0) * qwi,
                        (B_Kappa_bar - Ji * B_Kappa0) * qwi,
                    ]
                )

                for node in range(self.nnodes_element_la_g):
                    g_el[self.nodalDOF_element_la_g[node]] -= (
                        self.N_la_g[el, i, node] * delta_strains[self.constraints]
                    )

            return g_el

        def g_q(self, t, q):
            coo = CooMatrix((self.nla_g, self.nq))
            for el in range(self.nelement):
                elDOF = self.elDOF[el]
                elDOF_la_g = self.elDOF_la_g[el]
                coo[elDOF_la_g, elDOF] = self.g_q_el(q[elDOF], el)
            return coo

        def g_q_el(self, qe, el):
            g_q_el = np.zeros((self.nla_g_element, self.nq_element), dtype=qe.dtype)
            for i in range(self.nquadrature):
                # extract reference state variables
                qpi = self.qp[el, i]
                qwi = self.qw[el, i]

                # evaluate required quantities
                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    B_Gamma_bar_qe,
                    B_Kappa_bar_qe,
                ) = self._deval(qe, qpi, N=self.N_r[el, i], N_xi=self.N_r_xi[el, i])

                delta_strains_qe = np.vstack((B_Gamma_bar_qe, B_Kappa_bar_qe))

                for node in range(self.nnodes_element_la_g):
                    nodalDOF_la_g = self.nodalDOF_element_la_g[node]
                    g_q_el[nodalDOF_la_g, :] -= (
                        self.N_la_g[el, i, node]
                        * delta_strains_qe[self.constraints, :]
                        * qwi
                    )

            return g_q_el

        def W_g(self, t, q):
            coo = CooMatrix((self.nu, self.nla_g))
            for el in range(self.nelement):
                elDOF = self.elDOF[el]
                elDOF_u = self.elDOF_u[el]
                elDOF_la_g = self.elDOF_la_g[el]
                coo[elDOF_u, elDOF_la_g] = self.W_g_el(q[elDOF], el)
            return coo

        def W_g_el(self, qe, el):
            W_g_el = np.zeros((self.nu_element, self.nla_g_element), dtype=qe.dtype)

            for i in range(self.nquadrature):
                # extract reference state variables
                qpi = self.qp[el, i]
                qwi = self.qw[el, i]

                # evaluate required quantities
                _, A_IB, B_Gamma_bar, B_Kappa_bar = self._eval(
                    qe, qpi, N=self.N_r[el, i], N_xi=self.N_r_xi[el, i]
                )

                ############################
                # virtual work contributions
                ############################
                for node_r in range(self.nnodes_element_r):
                    nodalDOF_r = self.nodalDOF_element_r_u[node_r]
                    N_r_xi = self.N_r_xi[el, i, node_r]
                    for node_la_g in range(self.nnodes_element_la_g):
                        nodalDOF_la_g = self.nodalDOF_element_la_g[node_la_g]
                        W_g_el[
                            nodalDOF_r[:, None],
                            nodalDOF_la_g[: self.nconstraints_gamma],
                        ] -= (
                            N_r_xi
                            * A_IB[:, self.constraints_gamma]
                            * self.N_la_g[el, i, node_la_g]
                            * qwi
                        )

                for node_p in range(self.nnodes_element_p):
                    nodalDOF_p = self.nodalDOF_element_p_u[node_p]
                    N_p = self.N_p[el, i, node_p]
                    N_p_xi = self.N_p_xi[el, i, node_p]

                    for node_la_g in range(self.nnodes_element_la_g):
                        nodalDOF_la_g = self.nodalDOF_element_la_g[node_la_g]
                        W_g_el[
                            nodalDOF_p[:, None],
                            nodalDOF_la_g[self.nconstraints_gamma :],
                        ] -= (
                            (N_p_xi * np.eye(3) - N_p * ax2skew(B_Kappa_bar))[
                                :, self.constraints_kappa
                            ]
                            * self.N_la_g[el, i, node_la_g]
                            * qwi
                        )

                        W_g_el[
                            nodalDOF_p[:, None],
                            nodalDOF_la_g[: self.nconstraints_gamma],
                        ] += (
                            N_p
                            * ax2skew(B_Gamma_bar)[:, self.constraints_gamma]
                            * self.N_la_g[el, i, node_la_g]
                            * qwi
                        )

            return W_g_el

        def Wla_g_q(self, t, q, la_g):
            coo = CooMatrix((self.nu, self.nq))
            for el in range(self.nelement):
                elDOF = self.elDOF[el]
                elDOF_u = self.elDOF_u[el]
                elDOF_la_g = self.elDOF_la_g[el]
                coo[elDOF_u, elDOF] = self.Wla_g_q_el(q[elDOF], la_g[elDOF_la_g], el)
            return coo

        def Wla_g_q_el(self, qe, la_ge, el):
            Wla_g_q_el = np.zeros(
                (self.nu_element, self.nq_element), dtype=np.common_type(qe, la_ge)
            )

            for i in range(self.nquadrature):
                # extract reference state variables
                qpi = self.qp[el, i]
                qwi = self.qw[el, i]
                Ji = self.J[el, i]

                # evaluate required quantities
                (
                    r_OP,
                    A_IB,
                    B_Gamma_bar,
                    B_Kappa_bar,
                    r_OP_qe,
                    A_IB_qe,
                    B_Gamma_bar_qe,
                    B_Kappa_bar_qe,
                ) = self._deval(qe, qpi, N=self.N_r[el, i], N_xi=self.N_r_xi[el, i])

                # interpolation of the n and m
                la_g = np.zeros(self.mesh_la_g.dim_q, dtype=qe.dtype)

                for node in range(self.nnodes_element_la_g):
                    la_g_node = la_ge[self.nodalDOF_element_la_g[node]]
                    la_g += self.N_la_g[el, i, node] * la_g_node

                B_n = self.B_n_la_g_sieve @ la_g[: self.nconstraints_gamma]
                B_m = self.B_m_la_g_sieve @ la_g[self.nconstraints_gamma :]

                for node in range(self.nnodes_element_r):
                    Wla_g_q_el[self.nodalDOF_element_r_u[node], :] -= (
                        self.N_r_xi[el, i, node]
                        * qwi
                        * (np.einsum("ikj,k->ij", A_IB_qe, B_n))
                    )

                for node in range(self.nnodes_element_p):
                    Wla_g_q_el[self.nodalDOF_element_p_u[node], :] += (
                        self.N_p[el, i, node]
                        * qwi
                        * (
                            -ax2skew(B_n) @ B_Gamma_bar_qe
                            - ax2skew(B_m) @ B_Kappa_bar_qe
                        )
                    )

            return Wla_g_q_el

        def g_dot(self, t, q, u):
            return self.W_g(t, q).toarray().T @ u

        def g_dot_u(self, t, q):
            W_g = self.W_g(t, q)
            coo = CooMatrix((self.nla_g, self.nu))
            coo.row = W_g.col
            coo.col = W_g.row
            coo.data = W_g.data
            return coo

        def g_dot_q(self, t, q, u):
            raise NotImplementedError

        def g_ddot(self, t, q, u, u_dot):
            # Check for already moving initial conditions
            g_ddot = np.zeros(self.nla_g, dtype=q.dtype)
            g_ddot += self.W_g(t, q).toarray().T @ u_dot
            W_g_T_q = approx_fprime(q, lambda q1: self.W_g(t, q1).toarray().T)
            q_dot = self.q_dot(t, q, u)
            g_ddot += np.einsum("ijk, k->ij", W_g_T_q, q_dot) @ u
            return g_ddot

        ##############################
        # stress and strain evaluation
        ##############################
        def eval_stresses(self, t, q, la_c, la_g, xi):
            B_n = np.zeros(3)
            B_m = np.zeros(3)
            B_n_impressed, B_m_impressed = super().eval_stresses(t, q, la_c, la_g, xi)
            B_n[self.constraints_gamma] += B_n_impressed[self.constraints_gamma]
            B_m[self.constraints_kappa] += B_m_impressed[self.constraints_kappa]

            el = self.element_number(xi)
            la_ge = la_g[self.elDOF_la_g[el]]
            # TODO: lets see how to avoid the flatten
            N_la_ge = self.basis_functions_la_g(xi).flatten()
            la_gg = np.zeros(self.nconstraints_gamma + self.nconstraints_kappa)

            for node in range(self.nnodes_element_la_g):
                la_g_node = la_ge[self.nodalDOF_element_la_g[node]]
                la_gg += N_la_ge[node] * la_g_node

            B_n_c = self.B_n_la_g_sieve @ la_gg[: self.nconstraints_gamma]
            B_m_c = self.B_m_la_g_sieve @ la_gg[self.nconstraints_gamma :]

            B_n += B_n_c
            B_m += B_m_c

            return B_n, B_m

    return CosseratRodConstrained
