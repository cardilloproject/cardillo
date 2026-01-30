from abc import ABC, abstractmethod
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
import numpy as np
from scipy.sparse import identity

from cardillo.math.algebra import norm, cross3, ax2skew
from cardillo.math.approx_fprime import approx_fprime
from cardillo.utility.coo_matrix import CooMatrix

from ._base_export import RodExportBase
from ._cross_section import CrossSectionInertias
from .discretization.lagrange import LagrangeKnotVector
from .discretization.mesh1D import Mesh1D

eye3 = np.eye(3, dtype=float)
eye12 = np.eye(12, dtype=float)


class CosseratRodDirectorBubnovGalerkin(RodExportBase, ABC):
    def __init__(
        self,
        cross_section,
        material_model,
        nelement,
        polynomial_degree,
        nquadrature,
        Q,
        *,
        q0=None,
        u0=None,
        nquadrature_dyn=None,
        cross_section_inertias=CrossSectionInertias(),
        name=None,
    ):
        """Base class for Bubnov-Galerkin Cosserat rod formulations with
        directors for the nodal orientation parametrization.

        Parameters
        ----------
        cross_section : CrossSection
            Geometric cross-section properties: area, first and second moments
            of area.
        material_model: RodMaterialModel
            Constitutive law of Cosserat rod which relates the rod strain
            measures B_Gamma and B_Kappa with the contact forces B_n and couples
            B_m in the cross-section-fixed B-basis.
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
            angular velocity represented in the cross-section-fixed B-basis
            B_omega_IB.
        nquadrature_dyn : int
            Number of quadrature points to integrate dynamical and external
            virtual work functionals.
        cross_section_inertias : CrossSectionInertias
            Inertia properties of cross-sections: Cross-section mass density and
            Cross-section inertia tensor represented in the cross-section-fixed
            B-Basis.
        name : str
            Name of contribution.
        """
        # call base class for all export properties
        super().__init__(cross_section)

        # rod properties
        self.material_model = material_model
        self.cross_section_inertias = cross_section_inertias
        self.name = "Cosserat_rod_director" if name is None else name

        # distinguish between inertia quadrature and other parts
        self.nquadrature = nquadrature
        if nquadrature_dyn is None:
            self.nquadrature_dyn = nquadrature
        else:
            self.nquadrature_dyn = nquadrature_dyn
        self.nelement = nelement

        # self._eval_cache = LRUCache(maxsize=nquadrature + 10)
        # self._deval_cache = LRUCache(maxsize=nquadrature + 10)

        ##############################################################
        # discretization parameters centerline (r) and orientation (p)
        ##############################################################
        # TODO: combine mesh for position and orientation fields
        self.polynomial_degree = polynomial_degree
        self.knot_vector = LagrangeKnotVector(self.polynomial_degree, nelement)

        # build mesh objects
        self.mesh = Mesh1D(
            self.knot_vector,
            nquadrature,
            dim_q=12,
            derivative_order=1,
            basis="Lagrange",
            quadrature="Gauss",
        )
        self.mesh_dyn = Mesh1D(
            self.knot_vector,
            nquadrature_dyn,
            dim_q=12,
            derivative_order=1,
            basis="Lagrange",
            quadrature="Gauss",
        )

        # total number of nodes
        self.nnodes = self.mesh.nnodes

        # number of nodes per element
        self.nnodes_element = self.mesh.nnodes_per_element

        # total number of generalized position coordinates
        self.nq = self.mesh.nq

        # total number of generalized velocity coordinates
        self.nu = self.nq

        # number of generalized position coordinates per element
        self.nq_element = self.mesh.nq_per_element

        # # number of generalized velocity coordinates per element
        # self.nu_element = self.mesh.nu_per_element

        # global element connectivity
        # qe = q[elDOF[e]] "q^e = C_q,e q"
        # ue = u[elDOF[e]] "u^e = C_u,e u"
        self.elDOF = self.mesh.elDOF

        # global nodal connectivity
        self.nodalDOF = self.mesh.nodalDOF

        # nodal connectivity on element level
        # r_OP_i^e = C_r,i^e * C_q,e q = C_r,i^e * q^e
        # r_OPi = qe[nodalDOF_element[i]]
        # v_P_i^e = C_u,i^e * C_u,e q = C_u,i^e * u^e
        # v_Pi = ue[nodalDOF_element[i]]
        self.nodalDOF_element = self.mesh.nodalDOF_element

        # shape functions and their first derivatives
        self.N = self.mesh.N
        self.N_xi = self.mesh.N_xi
        self.N_dyn = self.mesh_dyn.N

        # quadrature points
        # note: compliance equations use the same quadrature points
        self.qp = self.mesh.qp  # quadrature points
        self.qw = self.mesh.wp  # quadrature weights
        self.qp_dyn = self.mesh_dyn.qp  # quadrature points for dynamics
        self.qw_dyn = self.mesh_dyn.wp  # quadrature weights for dynamics

        # referential generalized position coordinates, initial generalized
        # position and velocity coordinates
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions = self.mesh.eval_basis

        # unit quaternion constraints
        dim_g = 6
        self.nla_g = self.nnodes * dim_g
        self.nodalDOF_la_g = np.arange(self.nla_g).reshape(self.nnodes, dim_g)
        # self.la_g0 = np.zeros(self.nla_g, dtype=float)

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
                    qe, N=self.N[el, i], N_xi=self.N_xi[el, i]
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
                N, N_xi = self.basis_functions(qpi, el)
                
                # evaluate required quantities
                _, _, B_Gamma_bar, B_Kappa_bar = self._eval(qe, N, N_xi)

                # length of reference tangential vector
                self.J_dyn[el, i] = norm(B_Gamma_bar)

    def element_number(self, xi):
        """Compute element number from given xi."""
        return self.knot_vector.element_number(xi)[0]

    def element_interval(self, el):
        return self.knot_vector.element_interval(el)

    ############################
    # export of centerline nodes
    ############################
    def nodes(self, q):
        """Returns nodal position coordinates"""
        q_body = q[self.qDOF]
        return np.array([q_body[nodalDOF][:3] for nodalDOF in self.nodalDOF]).T

    def nodalFrames(self, q, elementwise=False):
        """Returns nodal positions and nodal directors.
        If elementwise==True : returned arrays are each of shape [nnodes, 3]
        If elementwise==False : returned arrays are each of shape [nelements, nnodes_per_element, 3]
        """

        q_body = q[self.qDOF]
        if elementwise:
            r = np.zeros([self.nelement, self.nnodes_element, 3], dtype=float)
            ex = np.zeros([self.nelement, self.nnodes_element, 3], dtype=float)
            ey = np.zeros([self.nelement, self.nnodes_element, 3], dtype=float)
            ez = np.zeros([self.nelement, self.nnodes_element, 3], dtype=float)
            for el, elDOF in enumerate(self.elDOF):
                qe = q_body[elDOF]

                r[el] = [qe[nodalDOF_el][:3] for nodalDOF_el in self.nodalDOF_element]
                A_IB = np.array(
                    [
                        qe[nodalDOF_el][3:].reshape(3, 3, order="F")
                        for nodalDOF_el in self.nodalDOF_element
                    ]
                )

                ex[el] = A_IB[:, :, 0]
                ey[el] = A_IB[:, :, 1]
                ez[el] = A_IB[:, :, 2]

            return r, ex, ey, ez

        else:
            r = np.array([q_body[nodalDOF][:3] for nodalDOF in self.nodalDOF])
            A_IB = np.array(
                [
                    q_body[nodalDOF][3:].reshape(3, 3, order="F")
                    for nodalDOF in self.nodalDOF
                ]
            )
            return r, A_IB[:, :, 0], A_IB[:, :, 1], A_IB[:, :, 2]
        
    ##################
    # abstract methods
    ##################
    # @cachedmethod(
    #     lambda self: self._eval_cache,
    #     key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
    # )
    def _eval(self, qe, N, N_xi):
        # interpolation
        x = np.zeros(12, dtype=qe.dtype)
        x_xi = np.zeros(12, dtype=qe.dtype)
        for node in range(self.nnodes_element):
            q_node = qe[self.nodalDOF_element[node]]
            x += N[node] * q_node
            x_xi += N_xi[node] * q_node

        r_OC = x[:3]
        di = x[3:]
        A_IB = di.reshape(3, 3, order="F")

        r_OC_xi = x_xi[:3]
        di_xi = x_xi[3:]
        A_IB_xi = di_xi.reshape(3, 3, order="F")

        # axial and shear strains
        B_Gamma_bar = A_IB.T @ r_OC_xi

        # torsional and flexural strains
        d1, d2, d3 = A_IB.T
        d1_xi, d2_xi, d3_xi = A_IB_xi.T
        B_Kappa_bar = np.array(
            [
                0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                0.5 * (d2 @ d1_xi - d1 @ d2_xi),
            ]
        )

        return r_OC, A_IB, B_Gamma_bar, B_Kappa_bar

    # @cachedmethod(
    #     lambda self: self._deval_cache,
    #     key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
    # )
    def _deval(self, qe, N, N_xi):
        # interpolation
        x = np.zeros(12, dtype=qe.dtype)
        x_xi = np.zeros(12, dtype=qe.dtype)
        x_qe = np.zeros((12, self.nq_element), dtype=qe.dtype)
        x_xi_qe = np.zeros((12, self.nq_element), dtype=qe.dtype)
        for node in range(self.nnodes_element):
            nodalDOF = self.nodalDOF_element[node]
            q_node = qe[nodalDOF]

            x += N[node] * q_node
            x_xi += N_xi[node] * q_node

            x_qe[:, nodalDOF] += N[node] * eye12
            x_xi_qe[:, nodalDOF] += N_xi[node] * eye12

        r_OC = x[:3]
        di = x[3:]
        A_IB = di.reshape(3, 3, order="F")

        r_OC_qe = x_qe[:3]
        di_qe = x_qe[3:]
        A_IB_qe = di_qe.reshape(3, 3, -1, order="F")

        r_OC_xi = x_xi[:3]
        di_xi = x_xi[3:]
        A_IB_xi = di_xi.reshape(3, 3, order="F")

        r_OC_xi_qe = x_xi_qe[:3]
        di_xi_qe = x_xi_qe[3:]
        A_IB_xi_qe = di_xi_qe.reshape(3, 3, -1, order="F")

        # extract directors
        # TODO: This can be made easier.
        d1, d2, d3 = A_IB.T
        d1_xi, d2_xi, d3_xi = A_IB_xi.T
        d1_qe, d2_qe, d3_qe = A_IB_qe.transpose(1, 0, 2)
        d1_xi_qe, d2_xi_qe, d3_xi_qe = A_IB_xi_qe.transpose(1, 0, 2)

        # axial and shear strains
        B_Gamma_bar = A_IB.T @ r_OC_xi
        B_Gamma_bar_qe = np.einsum("k,kij", r_OC_xi, A_IB_qe) + A_IB.T @ r_OC_xi_qe

        # torsional and flexural strains
        B_Kappa_bar = np.array(
            [
                0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                0.5 * (d2 @ d1_xi - d1 @ d2_xi),
            ]
        )
        B_Kappa_bar_qe = np.array(
            [
                0.5
                * (d3 @ d2_xi_qe + d2_xi @ d3_qe - d2 @ d3_xi_qe - d3_xi @ d2_qe),
                0.5
                * (d1 @ d3_xi_qe + d3_xi @ d1_qe - d3 @ d1_xi_qe - d1_xi @ d3_qe),
                0.5
                * (d2 @ d1_xi_qe + d1_xi @ d2_qe - d1 @ d2_xi_qe - d2_xi @ d1_qe),
            ]
        )

        return (
            r_OC,
            A_IB,
            B_Gamma_bar,
            B_Kappa_bar,
            r_OC_qe,
            A_IB_qe,
            B_Gamma_bar_qe,
            B_Kappa_bar_qe,
        )

    def A_IB(self, t, qe, xi):
        # evaluate shape functions
        N, _ = self.basis_functions(xi)

        # interpolation
        x = np.zeros(12, dtype=qe.dtype)
        for node in range(self.nnodes_element):
            q_node = qe[self.nodalDOF_element[node]]
            x += N[node] * q_node
        
        di = x[3:]
        A_IB = di.reshape(3, 3, order="F")
        return A_IB

    def A_IB_q(self, t, qe, xi):
        # evaluate shape functions
        N, _ = self.basis_functions(xi)
        
        # interpolation
        x = np.zeros(12, dtype=qe.dtype)
        x_qe = np.zeros((12, self.nq_element), dtype=qe.dtype)
        for node in range(self.nnodes_element):
            nodalDOF = self.nodalDOF_element[node]
            x_qe[:, nodalDOF] += N[node] * eye12

        di_qe = x_qe[3:]
        A_IB_qe = di_qe.reshape(3, 3, -1, order="F")
        return A_IB_qe

    def assembler_callback(self):
        print(f"mass matrix is not implemented")
    #     self._M_coo()

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return u

    def q_dot_u(self, t, q):
        return identity(self.nq, format="coo")

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
        raise NotImplementedError
        E_kin_el = 0.0

        for i in range(self.nquadrature_dyn):
            # extract reference state variables
            qwi = self.qw_dyn[el, i]
            Ji = self.J_dyn[el, i]

            v_P = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element):
                v_P += self.N_r_dyn[el, i, node] * ue[self.nodalDOF_element_r_u[node]]

            B_omega_IB = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_p):
                B_omega_IB += (
                    self.N_p_dyn[el, i, node] * ue[self.nodalDOF_element[node]]
                )

            # delta_r A_rho0 r_ddot part
            E_kin_el += (
                0.5 * (v_P @ v_P) * self.cross_section_inertias.A_rho0 * Ji * qwi
            )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            E_kin_el += (
                0.5
                * (B_omega_IB @ self.cross_section_inertias.B_I_rho0 @ B_omega_IB)
                * Ji
                * qwi
            )

        return E_kin_el

    def linear_momentum(self, t, q, u):
        raise NotImplementedError
        linear_momentum = np.zeros(3, dtype=float)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            qe = q[elDOF]
            ue = u[elDOF_u]

            linear_momentum += self.linear_momentum_el(qe, ue, el)

        return linear_momentum

    def linear_momentum_el(self, qe, ue, el):
        raise NotImplementedError
        linear_momentum_el = np.zeros(3, dtype=float)

        for i in range(self.nquadrature_dyn):
            # extract reference state variables
            qwi = self.qw_dyn[el, i]
            Ji = self.J_dyn[el, i]

            v_P = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element):
                v_P += self.N_r_dyn[el, i, node] * ue[self.nodalDOF_element[node]]

            linear_momentum_el += v_P * self.cross_section_inertias.A_rho0 * Ji * qwi

        return linear_momentum_el

    def angular_momentum(self, t, q, u):
        raise NotImplementedError
        angular_momentum = np.zeros(3, dtype=float)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            qe = q[elDOF]
            ue = u[elDOF_u]

            angular_momentum += self.angular_momentum_el(t, qe, ue, el)

        return angular_momentum

    def angular_momentum_el(self, t, qe, ue, el):
        raise NotImplementedError
        angular_momentum_el = np.zeros(3, dtype=float)

        for i in range(self.nquadrature_dyn):
            # extract reference state variables
            qpi = self.qp_dyn[el, i]
            qwi = self.qw_dyn[el, i]
            Ji = self.J_dyn[el, i]

            r_OP = np.zeros(3, dtype=float)
            v_P = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element):
                r_OP += self.N_r_dyn[el, i, node] * qe[self.nodalDOF_element[node]]
                v_P += self.N_r_dyn[el, i, node] * ue[self.nodalDOF_element_r_u[node]]

            B_omega_IB = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_p):
                B_omega_IB += (
                    self.N_p_dyn[el, i, node] * ue[self.nodalDOF_element[node]]
                )

            A_IB = self.A_IB(t, qe, (qpi,))

            angular_momentum_el += (
                (
                    cross3(r_OP, v_P) * self.cross_section_inertias.A_rho0
                    + A_IB @ (self.cross_section_inertias.B_I_rho0 @ B_omega_IB)
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
        raise NotImplementedError
        """ "Mass matrix is called in assembler callback."""
        self.__M = CooMatrix((self.nu, self.nu))
        for el in range(self.nelement):
            # extract element degrees of freedom
            elDOF_u = self.elDOF_u[el]

            # sparse assemble element mass matrix
            self.__M[elDOF_u, elDOF_u] = self.M_el(el)

    def M_el(self, el):
        raise NotImplementedError
        M_el = np.zeros((self.nu_element, self.nu_element), dtype=float)

        for i in range(self.nquadrature_dyn):
            # extract reference state variables
            qwi = self.qw_dyn[el, i]
            Ji = self.J_dyn[el, i]

            # delta_r A_rho0 r_ddot part
            M_el_r_r = eye3 * self.cross_section_inertias.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N_r_dyn[el, i, node_a] * self.N_r_dyn[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_p_p = self.cross_section_inertias.B_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_p):
                nodalDOF_a = self.nodalDOF_element[node_a]
                for node_b in range(self.nnodes_element_p):
                    nodalDOF_b = self.nodalDOF_element[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_p_p * (
                        self.N_p_dyn[el, i, node_a] * self.N_p_dyn[el, i, node_b]
                    )

        return M_el

    ######################
    # director constraints
    ######################
    def g(self, t, q):
        g = np.zeros(self.nla_g, dtype=float)
        for node, nodalDOF in enumerate(self.nodalDOF):
            q_node = q[nodalDOF]
            d1, d2, d3 = q_node[3:6], q_node[6:9], q_node[9:]

            g[node * 6 + 0] = 0.5 * (d1 @ d1 - 1.0)
            g[node * 6 + 1] = 0.5 * (d2 @ d2 - 1.0)
            g[node * 6 + 2] = 0.5 * (d3 @ d3 - 1.0)
            g[node * 6 + 3] = d1 @ d2
            g[node * 6 + 4] = d2 @ d3
            g[node * 6 + 5] = d3 @ d1

        return g

    def g_q_node(self, q_node):
        d1, d2, d3 = q_node[3:6], q_node[6:9], q_node[9:]

        g_q_node = np.zeros((6, 12), dtype=float)
        g_q_node[0, 3:6] = d1
        g_q_node[1, 6:9] = d2
        g_q_node[2, 9:12] = d3
        g_q_node[3, 3:6] = d2
        g_q_node[3, 6:9] = d1
        g_q_node[4, 6:9] = d3
        g_q_node[4, 9:] = d2
        g_q_node[5, 3:6] = d3
        g_q_node[5, 9:] = d1
        return g_q_node

    def g_q(self, t, q):
        coo = CooMatrix((self.nla_g, self.nq))
        for node, nodalDOF in enumerate(self.nodalDOF):
            q_node = q[nodalDOF]
            coo[6 * node : 6 * (node + 1), nodalDOF] = self.g_q_node(q_node)
        return coo
    
    def W_g(self, t, q):
        coo = CooMatrix((self.nq, self.nla_g))
        for node, nodalDOF in enumerate(self.nodalDOF):
            q_node = q[nodalDOF]
            coo[nodalDOF, 6 * node : 6 * (node + 1)] = self.g_q_node(q_node).T
        return coo
    
    # def Wla_g_q(self, t, q, la_g):
    #     coo = CooMatrix((self.nq, self.nq))
    #     for node, nodalDOF in enumerate(self.nodalDOF):
    #         q_node = q[nodalDOF]
    #         la_g_node = la_g[6 * node : 6 * (node + 1)]
    #         coo[nodalDOF, nodalDOF] = approx_fprime(
    #             q_node, lambda q: self.g_q_node(q).T @ la_g_node,
    #         )
    #     return coo
    
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
        return self.elDOF[el]

    def local_qDOF_P(self, xi):
        return self.elDOF_P(xi)

    def local_uDOF_P(self, xi):
        return self.elDOF_P_u(xi)

    ##########################
    # r_OP / A_IB contribution
    ##########################
    def r_OP(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        N, N_xi = self.basis_functions(xi)
        r_OC, A_IB, _, _ = self._eval(qe, N, N_xi)
        return r_OC + A_IB @ B_r_CP

    # TODO: Think of a faster version than using _deval
    def r_OP_q(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        # evaluate required quantities
        N, N_xi = self.basis_functions(xi)
        (
            r_OC,
            A_IB,
            _,
            _,
            r_OC_q,
            A_IB_q,
            _,
            _,
        ) = self._deval(qe, N, N_xi)

        return r_OC_q + np.einsum("ijk,j->ik", A_IB_q, B_r_CP)

    def v_P(self, t, qe, ue, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.r_OP(t, ue, xi, B_r_CP)

    def v_P_q(self, t, qe, ue, xi, B_r_CP=np.zeros(3, dtype=float)):
        return np.zeros((3, self.nq_element), dtype=float)

    def J_P(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.r_OP_q(t, qe, xi, B_r_CP)

    def J_P_q(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        return np.zeros((3, self.nq_element, self.nq_element), dtype=float)

    def a_P(self, t, qe, ue, ue_dot, xi, B_r_CP=np.zeros(3, dtype=float)):
        raise NotImplementedError
        N, _ = self.basis_functions(xi)

        # interpolate orientation
        A_IB = self.A_IB(t, qe, xi)

        # centerline acceleration
        a_C = np.zeros(3, dtype=np.common_type(qe, ue, ue_dot))
        for node in range(self.nnodes_element):
            a_C += N[node] * ue_dot[self.nodalDOF_element[node]]

        # angular velocity and acceleration in B-frame
        B_Omega = self.B_Omega(t, qe, ue, xi)
        B_Psi = self.B_Psi(t, qe, ue, ue_dot, xi)

        # rigid body formular
        return a_C + A_IB @ (
            cross3(B_Psi, B_r_CP) + cross3(B_Omega, cross3(B_Omega, B_r_CP))
        )

    def a_P_q(self, t, qe, ue, ue_dot, xi, B_r_CP=None):
        raise NotImplementedError
        B_Omega = self.B_Omega(t, qe, ue, xi)
        B_Psi = self.B_Psi(t, qe, ue, ue_dot, xi)
        a_P_q = np.einsum(
            "ijk,j->ik",
            self.A_IB_q(t, qe, xi),
            cross3(B_Psi, B_r_CP) + cross3(B_Omega, cross3(B_Omega, B_r_CP)),
        )
        return a_P_q

    def a_P_u(self, t, qe, ue, ue_dot, xi, B_r_CP=None):
        raise NotImplementedError
        B_Omega = self.B_Omega(t, qe, ue, xi)
        local = -self.A_IB(t, qe, xi) @ (
            ax2skew(cross3(B_Omega, B_r_CP)) + ax2skew(B_Omega) @ ax2skew(B_r_CP)
        )

        N, _ = self.basis_functions(xi)
        a_P_u = np.zeros((3, self.nu_element), dtype=float)
        for node in range(self.nnodes_element):
            a_P_u[:, self.nodalDOF_element[node]] += N[node] * local

        return a_P_u

    def B_Omega(self, t, qe, ue, xi):
        N, _ = self.basis_functions(xi)
        x = np.zeros(12, dtype=qe.dtype)
        x_dot = np.zeros(12, dtype=ue.dtype)
        for node in range(self.nnodes_element):
            x += N[node] * qe[self.nodalDOF_element[node]]
            x_dot += N[node] * ue[self.nodalDOF_element[node]]

        d1, d2, d3 = x[3:6], x[6:9], x[9:]

        L = np.zeros((3, 12), dtype=np.common_type(qe, ue))
        L[0, 6:9] = -d3
        L[0, 9:] = d2
        L[1, 3:6] = d3
        L[1, 9:] = -d1
        L[2, 3:6] = -d2
        L[2, 6:9] = d1

        return -0.5 * (L @ x_dot)

    def B_Omega_q(self, t, qe, ue, xi):
        raise NotImplementedError
        N, _ = self.basis_functions(xi)
        x = np.zeros(12, dtype=qe.dtype)
        x_dot = np.zeros(12, dtype=ue.dtype)
        for node in range(self.nnodes_element):
            x += N[node] * qe[self.nodalDOF_element[node]]
            x_dot += N[node] * ue[self.nodalDOF_element[node]]

        d1, d2, d3 = x[3:6], x[6:9], x[9:]
        d1_dot, d2_dot, d3_dot = x_dot[3:6], x_dot[6:9], x_dot[9:]

        L = np.zeros((3, 12), dtype=np.common_type(qe, ue))
        L[0, 6:9] = -d3_dot
        L[0, 9:] = d2_dot
        L[1, 3:6] = d3_dot
        L[1, 9:] = -d1_dot
        L[2, 3:6] = -d2_dot
        L[2, 6:9] = d1_dot

        return 0.5 * L

    def B_J_R(self, t, qe, xi):
        N, _ = self.basis_functions(xi)
        x = np.zeros(12, dtype=qe.dtype)
        for node in range(self.nnodes_element):
            x += N[node] * qe[self.nodalDOF_element[node]]

        d1, d2, d3 = x[3:6], x[6:9], x[9:]

        L = np.zeros((3, self.nq_element), dtype=qe.dtype)
        for node in range(self.nnodes_element):
            nodalDOF = self.nodalDOF_element[node]
            L[0, nodalDOF[6:9]] = -d3 * N[node]
            L[0, nodalDOF[9:]] = d2 * N[node]
            L[1, nodalDOF[3:6]] = d3 * N[node]
            L[1, nodalDOF[9:]] = -d1 * N[node]
            L[2, nodalDOF[3:6]] = -d2 * N[node]
            L[2, nodalDOF[6:9]] = d1 * N[node]

        return -0.5 * L

    def B_J_R_q(self, t, qe, xi):
        N, _ = self.basis_functions(xi)
        B_J_R_q = np.zeros((3, self.nq_element, self.nq_element), dtype=float)
        for node in range(self.nnodes_element):
            nodalDOF = self.nodalDOF_element[node]
            B_J_R_q[0, nodalDOF[6:9], nodalDOF[9:][:, None]] -= N[node] * eye3 # -d3
            B_J_R_q[0, nodalDOF[9:], nodalDOF[6:9][:, None]] += N[node] * eye3 # d2
            B_J_R_q[1, nodalDOF[3:6], nodalDOF[9:][:, None]] += N[node] * eye3 # d3
            B_J_R_q[1, nodalDOF[9:], nodalDOF[3:6][:, None]] -= N[node] * eye3 # -d1
            B_J_R_q[2, nodalDOF[3:6], nodalDOF[6:9][:, None]] -= N[node] * eye3 # -d2
            B_J_R_q[2, nodalDOF[6:9], nodalDOF[3:6][:, None]] += N[node] * eye3 # d1

        return B_J_R_q

    def B_Psi(self, t, qe, ue, ue_dot, xi):
        raise NotImplementedError
        """Since we use Petrov-Galerkin method we only interpolate the nodal
        time derivative of the angular velocities in the B-frame.
        """
        N, _ = self.basis_functions_p(xi)
        B_Psi = np.zeros(3, dtype=np.common_type(qe, ue, ue_dot))
        for node in range(self.nnodes_element_p):
            B_Psi += N[node] * ue_dot[self.nodalDOF_element[node]]
        return B_Psi

    def B_Psi_q(self, t, qe, ue, ue_dot, xi):
        raise NotImplementedError
        return np.zeros((3, self.nq_element), dtype=float)

    def B_Psi_u(self, t, qe, ue, ue_dot, xi):
        raise NotImplementedError
        return np.zeros((3, self.nu_element), dtype=float)

def make_CosseratRodDirectorBubnovGalerkin(
    polynomial_degree=1,
    reduced_integration=True,
):
    class Derived(CosseratRodDirectorBubnovGalerkin):
        def __init__(
            self,
            cross_section,
            material_model,
            nelement,
            Q,
            *,
            q0=None,
            u0=None,
            cross_section_inertias=CrossSectionInertias(),
            name=None,
        ):
            nquadrature = polynomial_degree
            nquadrature_dyn = int(np.ceil((polynomial_degree + 1) ** 2 / 2))

            if not reduced_integration:
                import warnings

                warnings.warn("Full integration is used!")
                nquadrature = nquadrature_dyn

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
                name=name,
            )
        
        @staticmethod
        def straight_configuration(
            nelement,
            L,
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
            p = A_IB0.reshape(-1, order="F")
            q_p = np.repeat(p, nnodes)

            return np.concatenate([q_r, q_p])
        
    return Derived