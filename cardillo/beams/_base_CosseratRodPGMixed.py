import numpy as np
from abc import ABC, abstractmethod
import warnings

from cardillo.beams._base_export import RodExportBase

from cardillo.math import (
    norm,
    cross3,
    ax2skew,
    approx_fprime,
    Log_SO3_quat,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
)

from cardillo.utility.coo_matrix import CooMatrix
from cardillo.discretization.lagrange import LagrangeKnotVector
from cardillo.discretization.mesh1D import Mesh1D


class CosseratRodPGMixed(RodExportBase, ABC):
    def __init__(
        self,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree,
        nelement,
        nquadrature,
        nquadrature_dyn,
        Q,
        q0=None,
        u0=None,
        mixed=False,
    ):
        """Base class for Petrov-Galerkin Cosserat rod formulations that uses quaternions for the parametrization of the nodal orientations."""

        super().__init__(cross_section)

        assert mixed, "This implements the mixed formulation only!"

        # beam properties
        self.material_model = material_model
        self.A_rho0 = A_rho0
        self.K_S_rho0 = K_S_rho0
        self.K_I_rho0 = K_I_rho0

        # can we use a constant mass matrix
        if np.allclose(K_S_rho0, np.zeros_like(K_S_rho0)):
            self.constant_mass_matrix = True
        else:
            self.constant_mass_matrix = False

        # distinguish between inertia quadrature and other parts
        self.nquadrature_dyn = nquadrature_dyn
        self.nquadrature = nquadrature
        self.nelement = nelement
        print(f"nquadrature_dyn: {nquadrature_dyn}")
        print(f"nquadrature: {nquadrature}")

        ######################################################
        # discretization parameters centerline and orientation
        ######################################################
        self.polynomial_degree_r = polynomial_degree
        self.polynomial_degree_psi = polynomial_degree
        self.knot_vector_r = LagrangeKnotVector(self.polynomial_degree_r, nelement)
        self.knot_vector_psi = LagrangeKnotVector(self.polynomial_degree_psi, nelement)

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
        self.mesh_psi = Mesh1D(
            self.knot_vector_psi,
            nquadrature,
            dim_q=4,
            derivative_order=1,
            basis="Lagrange",
            quadrature="Gauss",
            dim_u=3,
        )
        self.mesh_psi_dyn = Mesh1D(
            self.knot_vector_psi,
            nquadrature_dyn,
            dim_q=4,
            derivative_order=1,
            basis="Lagrange",
            quadrature="Gauss",
            dim_u=3,
        )

        # total number of nodes
        self.nnodes_r = self.mesh_r.nnodes
        self.nnodes_psi = self.mesh_psi.nnodes

        # number of nodes per element
        self.nnodes_element_r = self.mesh_r.nnodes_per_element
        self.nnodes_element_psi = self.mesh_psi.nnodes_per_element

        # total number of generalized coordinates and velocities
        self.nq_r = self.mesh_r.nq
        self.nq_psi = self.mesh_psi.nq
        self.nq = self.nq_r + self.nq_psi

        self.nu_r = self.mesh_r.nu
        self.nu_psi = self.mesh_psi.nu
        self.nu = self.nu_r + self.nu_psi

        # number of generalized coordinates and velocities per element
        self.nq_element_r = self.mesh_r.nq_per_element
        self.nq_element_psi = self.mesh_psi.nq_per_element
        self.nq_element = self.nq_element_r + self.nq_element_psi

        self.nu_element_r = self.mesh_r.nu_per_element
        self.nu_element_psi = self.mesh_psi.nu_per_element
        self.nu_element = self.nu_element_r + self.nu_element_psi

        # global element connectivity
        # qe = q[elDOF[e]] "q^e = C_e,q q"
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_psi = self.mesh_psi.elDOF + self.nq_r
        self.elDOF_r_u = self.mesh_r.elDOF_u
        self.elDOF_psi_u = self.mesh_psi.elDOF_u + self.nu_r

        # global nodal connectivity
        self.nodalDOF_r = self.mesh_r.nodalDOF
        self.nodalDOF_psi = self.mesh_psi.nodalDOF + self.nq_r
        self.nodalDOF_r_u = self.mesh_r.nodalDOF_u
        self.nodalDOF_psi_u = self.mesh_psi.nodalDOF_u + self.nu_r

        # nodal connectivity on element level
        # r_OP_i^e = C_r,i^e * C_e,q q = C_r,i^e * q^e
        # r_OPi = qe[nodelDOF_element_r[i]]
        self.nodalDOF_element_r = self.mesh_r.nodalDOF_element
        self.nodalDOF_element_psi = self.mesh_psi.nodalDOF_element + self.nq_element_r
        self.nodalDOF_element_r_u = self.mesh_r.nodalDOF_element_u
        self.nodalDOF_element_psi_u = (
            self.mesh_psi.nodalDOF_element_u + self.nu_element_r
        )

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nelement, self.nq_element), dtype=int)
        self.elDOF_u = np.zeros((nelement, self.nu_element), dtype=int)
        for el in range(nelement):
            self.elDOF[el, : self.nq_element_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_element_r :] = self.elDOF_psi[el]
            self.elDOF_u[el, : self.nu_element_r] = self.elDOF_r_u[el]
            self.elDOF_u[el, self.nu_element_r :] = self.elDOF_psi_u[el]

        # shape functions and their first derivatives
        self.N_r = self.mesh_r.N
        self.N_r_xi = self.mesh_r.N_xi
        self.N_psi = self.mesh_psi.N
        self.N_psi_xi = self.mesh_psi.N_xi

        self.N_r_dyn = self.mesh_r_dyn.N
        self.N_psi_dyn = self.mesh_psi_dyn.N

        # quadrature points
        # note: compliance equations use the same quadrature points
        self.qp = self.mesh_r.qp  # quadrature points
        self.qw = self.mesh_r.wp  # quadrature weights
        self.qp_dyn = self.mesh_r_dyn.qp  # quadrature points for dynamics
        self.qw_dyn = self.mesh_r_dyn.wp  # quadrature weights for dynamics

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions_r = self.mesh_r.eval_basis
        self.basis_functions_psi = self.mesh_psi.eval_basis

        # quaternion constraints
        dim_g_S = 1
        self.nla_S = self.nnodes_psi * dim_g_S
        self.nodalDOF_la_S = np.arange(self.nla_S).reshape(self.nnodes_psi, dim_g_S)
        self.la_S0 = np.zeros(self.nla_S, dtype=float)

        #######################################################
        # discretization parameters internal forces and moments
        #######################################################
        self.polynomial_degree_n = polynomial_degree - 1
        self.polynomial_degree_m = polynomial_degree - 1
        self.knot_vector_n = LagrangeKnotVector(self.polynomial_degree_n, nelement)
        self.knot_vector_m = LagrangeKnotVector(self.polynomial_degree_m, nelement)

        # build mesh objects
        self.mesh_n = Mesh1D(
            self.knot_vector_n,
            nquadrature,
            dim_q=3,
            derivative_order=0,
            basis="Lagrange_Disc",
            quadrature="Gauss",
            dim_u=3,
        )
        self.mesh_m = Mesh1D(
            self.knot_vector_m,
            nquadrature,
            dim_q=3,
            derivative_order=0,
            basis="Lagrange_Disc",
            quadrature="Gauss",
            dim_u=3,
        )

        # total number of nodes
        self.nnodes_n = self.mesh_n.nnodes
        self.nnodes_m = self.mesh_m.nnodes

        # number of nodes per element
        self.nnodes_element_n = self.mesh_n.nnodes_per_element
        self.nnodes_element_m = self.mesh_m.nnodes_per_element

        # total number of compliance coordiantes
        self.nla_c_n = self.mesh_n.nq
        self.nla_c_m = self.mesh_m.nq
        self.nla_c = self.nla_c_n + self.nla_c_m

        # number of compliance coordinates per element
        self.nla_c_element_n = self.mesh_n.nq_per_element
        self.nla_c_element_m = self.mesh_m.nq_per_element
        self.nla_c_element = self.nla_c_element_n + self.nla_c_element_m

        # global element connectivity for copliance coordinates
        self.elDOF_n = self.mesh_n.elDOF
        self.elDOF_m = self.mesh_m.elDOF + self.nla_c_n

        # global nodal connectivity
        self.nodalDOF_n = self.mesh_n.nodalDOF + self.nq_r + self.nq_psi
        self.nodalDOF_m = self.mesh_m.nodalDOF + self.nq_r + self.nq_psi + self.nla_c_n

        # nodal connectivity on element level
        self.nodalDOF_element_n = self.mesh_n.nodalDOF_element
        self.nodalDOF_element_m = self.mesh_m.nodalDOF_element + self.nla_c_element_n

        # build global elDOF connectivity matrix
        self.elDOF_la_c = np.zeros((nelement, self.nla_c_element), dtype=int)
        for el in range(nelement):
            self.elDOF_la_c[el, : self.nla_c_element_n] = self.elDOF_n[el]
            self.elDOF_la_c[el, self.nla_c_element_n :] = self.elDOF_m[el]

        # shape functions and their first derivatives
        self.N_n = self.mesh_n.N
        self.N_m = self.mesh_m.N

        # TODO: Dummy initial values for compliance
        self.la_c0 = np.zeros(self.nla_c, dtype=float)

        # evaluate shape functions at specific xi
        self.basis_functions_n = self.mesh_n.eval_basis
        self.basis_functions_m = self.mesh_m.eval_basis

        self.set_reference_strains(self.Q)

    def set_reference_strains(self, Q):
        self.Q = Q.copy()

        # precompute values of the reference configuration in order to save computation time
        # J in Harsch2020b (5)
        self.J = np.zeros((self.nelement, self.nquadrature), dtype=float)
        self.J_dyn = np.zeros((self.nelement, self.nquadrature_dyn), dtype=float)
        # dilatation and shear strains of the reference configuration
        self.K_Gamma0 = np.zeros((self.nelement, self.nquadrature, 3), dtype=float)
        # curvature of the reference configuration
        self.K_Kappa0 = np.zeros((self.nelement, self.nquadrature, 3), dtype=float)

        for el in range(self.nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(self.nquadrature):
                # current quadrature point
                qpi = self.qp[el, i]

                # evaluate required quantities
                _, _, K_Gamma_bar, K_Kappa_bar = self._eval(qe, qpi)

                # length of reference tangential vector
                J = norm(K_Gamma_bar)

                # axial and shear strains
                K_Gamma = K_Gamma_bar / J

                # torsional and flexural strains
                K_Kappa = K_Kappa_bar / J

                # safe precomputed quantities for later
                self.J[el, i] = J
                self.K_Gamma0[el, i] = K_Gamma
                self.K_Kappa0[el, i] = K_Kappa

            for i in range(self.nquadrature_dyn):
                # current quadrature point
                qpi = self.qp_dyn[el, i]

                # evaluate required quantities
                # _, _, K_Gamma_bar, K_Kappa_bar = self._eval(qe, qpi)

                _, _, K_Gamma_bar, K_Kappa_bar = self._eval(qe, qpi)

                # length of reference tangential vector
                self.J_dyn[el, i] = norm(K_Gamma_bar)

    @staticmethod
    def straight_configuration(
        nelement,
        L,
        polynomial_degree=1,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
    ):
        nnodes_r = polynomial_degree * nelement + 1
        nnodes_psi = polynomial_degree * nelement + 1

        x0 = np.linspace(0, L, num=nnodes_r)
        y0 = np.zeros(nnodes_r)
        z0 = np.zeros(nnodes_r)

        r0 = np.vstack((x0, y0, z0))
        for i in range(nnodes_r):
            r0[:, i] = r_OP + A_IK @ r0[:, i]

        # reshape generalized coordinates to nodal ordering
        q_r = r0.reshape(-1, order="C")

        # we have to extract the rotation vector from the given rotation matrix
        # and set its value for each node
        psi = Log_SO3_quat(A_IK)
        q_psi = np.repeat(psi, nnodes_psi)

        return np.concatenate([q_r, q_psi])

    @staticmethod
    def deformed_configuration(
        nelement,
        curve,
        dcurve,
        ddcurve,
        angle,
        polynomial_degree=1,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
    ):
        nnodes_r = polynomial_degree * nelement + 1

        LL = np.linspace(0, angle, nnodes_r)

        # nodal positions
        r0 = np.zeros((3, nnodes_r))
        P0 = np.zeros((4, nnodes_r))

        for i in range(nnodes_r):
            r0[:, i] = r_OP + A_IK @ curve(LL[i])
            A_KC = np.zeros((3, 3))
            A_KC[:, 0] = dcurve(LL[i]) / norm(dcurve(LL[i]))
            A_KC[:, 1] = ddcurve(LL[i]) / norm(ddcurve(LL[i]))
            A_KC[:, 2] = cross3(A_KC[:, 0], A_KC[:, 1])
            A_IC = A_IK @ A_KC
            P0[:, i] = Log_SO3_quat(A_IC)

            # TODO: check for half space

        for i in range(nnodes_r - 1):
            inner = P0[:, i] @ P0[:, i + 1]
            print(f"i: {i}")
            if inner < 0:
                print("wrong hemisphere!")
                P0[:, i + 1] *= -1
            else:
                print(f"correct hemisphere")

        # reshape nodal positions for generalized coordinates tuple
        q_r = r0.reshape(-1, order="C")
        q_P = P0.reshape(-1, order="C")

        return np.concatenate([q_r, q_P])

    @staticmethod
    def straight_initial_configuration(
        nelement,
        L,
        polynomial_degree=1,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
        v_P=np.zeros(3, dtype=float),
        K_omega_IK=np.zeros(3, dtype=float),
    ):
        nnodes_r = polynomial_degree * nelement + 1
        nnodes_psi = polynomial_degree * nelement + 1

        x0 = np.linspace(0, L, num=nnodes_r)
        y0 = np.zeros(nnodes_r)
        z0 = np.zeros(nnodes_r)

        r_OC0 = np.vstack((x0, y0, z0))
        for i in range(nnodes_r):
            r_OC0[:, i] = r_OP + A_IK @ r_OC0[:, i]

        # reshape generalized coordinates to nodal ordering
        q_r = r_OC0.reshape(-1, order="C")

        psi = Log_SO3_quat(A_IK)
        q_psi = np.repeat(psi, nnodes_psi)

        ################################
        # compute generalized velocities
        ################################
        # centerline velocities
        v_C0 = np.zeros_like(r_OC0, dtype=float)
        for i in range(nnodes_r):
            v_C0[:, i] = v_P + cross3(A_IK @ K_omega_IK, (r_OC0[:, i] - r_OC0[:, 0]))

        # reshape generalized velocities to nodal ordering
        u_r = v_C0.reshape(-1, order="C")

        # all nodes share the same angular velocity
        u_psi = np.repeat(K_omega_IK, nnodes_psi)

        q0 = np.concatenate([q_r, q_psi])
        u0 = np.concatenate([u_r, u_psi])

        return q0, u0

    def element_number(self, xi):
        """Compute element number from given xi."""
        return self.knot_vector_r.element_number(xi)[0]

    ############################
    # export of centerline nodes
    ############################
    def nodes(self, q):
        q_body = q[self.qDOF]
        return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF_r]).T

    ##################
    # abstract methods
    ##################
    @abstractmethod
    def _eval(self, qe, xi, mixed=False):
        """Compute (r_OP, A_IK, K_Gamma_bar, K_Kappa_bar)."""
        ...

    # @abstractmethod
    # def _deval(self, qe, xi):
    #     """Compute
    #         * r_OP
    #         * A_IK
    #         * K_Gamma_bar
    #         * K_Kappa_bar
    #         * r_OP_qe
    #         * A_IK_qe
    #         * K_Gamma_bar_qe
    #         * K_Kappa_bar_qe
    #     """
    #     ...

    @abstractmethod
    def A_IK(self, t, q, frame_ID):
        ...

    @abstractmethod
    def A_IK_q(self, t, q, frame_ID):
        ...

    ################################################
    # constraint equations without constraint forces
    ################################################
    def g_S(self, t, q):
        g_S = np.zeros(self.nla_S, dtype=q.dtype)
        for node in range(self.nnodes_psi):
            psi = q[self.nodalDOF_psi[node]]
            g_S[self.nodalDOF_la_S[node]] = np.array([psi @ psi - 1])
        return g_S

    def g_S_q(self, t, q):
        coo = CooMatrix((self.nla_S, self.nq))
        for node in range(self.nnodes_psi):
            nodalDOF = self.nodalDOF_psi[node]
            nodalDOF_S = self.nodalDOF_la_S[node]
            psi = q[nodalDOF]
            coo[nodalDOF_S, nodalDOF] = 2 * psi.reshape(1, -1)
        return coo

    def g_S_q_T_mu_q(self, t, q, mu):
        coo = CooMatrix((self.nq, self.nq))
        for node in range(self.nnodes_psi):
            nodalDOF = self.nodalDOF_psi[node]
            nodalDOF_S = self.nodalDOF_la_S[node]
            coo[nodalDOF, nodalDOF] = 2 * mu * np.eye(4)
        return coo

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        # centerline part
        q_dot = np.zeros_like(q, dtype=np.common_type(q, u))

        # centerline part
        for node in range(self.nnodes_r):
            nodalDOF_r = self.nodalDOF_r[node]
            q_dot[nodalDOF_r] = u[nodalDOF_r]

        # axis angle vector part
        for node in range(self.nnodes_psi):
            nodalDOF_psi = self.nodalDOF_psi[node]
            nodalDOF_psi_u = self.nodalDOF_psi_u[node]
            psi = q[nodalDOF_psi]
            K_omega_IK = u[nodalDOF_psi_u]
            q_dot[nodalDOF_psi] = T_SO3_inv_quat(psi, normalize=False) @ K_omega_IK

        return q_dot

    def q_dot_u(self, t, q, u):
        coo = CooMatrix((self.nq, self.nu))

        # trivial kinematic equation for centerline
        coo[range(self.nq_r), range(self.nu_r)] = np.ones(self.nq_r)

        # axis angle vector part
        for node in range(self.nnodes_psi):
            nodalDOF_psi = self.nodalDOF_psi[node]
            nodalDOF_psi_u = self.nodalDOF_psi_u[node]

            psi = q[nodalDOF_psi]
            psi = psi / norm(psi)
            coo[nodalDOF_psi, nodalDOF_psi_u] = T_SO3_inv_quat(psi, normalize=False)

        return coo

    def q_dot_q(self, t, q, u):
        coo = CooMatrix((self.nq, self.nq))

        # axis angle vector part
        for node in range(self.nnodes_psi):
            nodalDOF_psi = self.nodalDOF_psi[node]
            nodalDOF_psi_u = self.nodalDOF_psi_u[node]
            psi = q[nodalDOF_psi]
            K_omega_IK = u[nodalDOF_psi_u]

            coo[nodalDOF_psi, nodalDOF_psi] = np.einsum(
                "ijk,j->ik",
                T_SO3_inv_quat_P(psi, normalize=False),
                K_omega_IK,
            )

        return coo

    def step_callback(self, t, q, u):
        for node in range(self.nnodes_psi):
            psi = q[self.nodalDOF_psi[node]]
            q[self.nodalDOF_psi[node]] = psi / norm(psi)
        return q, u

    ###############################
    # potential and internal forces
    ###############################
    # # TODO:
    # def E_pot(self, t, q):
    #     E_pot = 0.0
    #     for el in range(self.nelement):
    #         elDOF = self.elDOF[el]
    #         E_pot += self.E_pot_el(q[elDOF], el)
    #     return E_pot

    # # TODO:
    # def E_pot_el(self, qe, el):
    #     E_pot_el = 0.0

    #     for i in range(self.nquadrature):
    #         # extract reference state variables
    #         qpi = self.qp[el, i]
    #         qwi = self.qw[el, i]
    #         Ji = self.J[el, i]
    #         K_Gamma0 = self.K_Gamma0[el, i]
    #         K_Kappa0 = self.K_Kappa0[el, i]

    #         # evaluate required quantities
    #         _, _, K_Gamma_bar, K_Kappa_bar = self._eval(qe, qpi)

    #         # axial and shear strains
    #         K_Gamma = K_Gamma_bar / Ji

    #         # torsional and flexural strains
    #         K_Kappa = K_Kappa_bar / Ji

    #         # evaluate strain energy function
    #         E_pot_el += (
    #             self.material_model.potential(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
    #             * Ji
    #             * qwi
    #         )

    #     return E_pot_el

    def eval_stresses(self, t, q, xi, mixed=False):
        raise NotImplementedError
        el = self.element_number(xi)
        Qe = self.Q[self.elDOF[el]]
        qe = q[self.elDOF[el]]

        _, _, K_Gamma_bar0, K_Kappa_bar0 = self._eval(Qe, xi)

        J_0 = norm(K_Gamma_bar0)
        K_Gamma0 = K_Gamma_bar0 / J_0
        K_Kappa0 = K_Kappa_bar0 / J_0

        _, _, K_Gamma_bar, K_Kappa_bar = self._eval(qe, xi)

        J = norm(K_Gamma_bar)
        K_Gamma = K_Gamma_bar / J
        K_Kappa = K_Kappa_bar / J

        K_n_DB = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
        K_m_DB = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)

        if mixed:
            N_n = N_m = self.basis_functions_n(xi).flatten()
            K_n = np.zeros(3, dtype=qe.dtype)
            K_m = np.zeros(3, dtype=qe.dtype)

            for node in range(self.nnodes_element_n):
                n_node = qe[self.nodalDOF_element_n[node]]
                K_n += N_n[node] * n_node
                m_node = qe[self.nodalDOF_element_m[node]]
                K_m += N_m[node] * m_node

            return K_n, K_m, K_n_DB, K_m_DB
        else:
            return K_n_DB, K_m_DB

    def eval_strains(self, t, q, xi, mixed=False):
        raise NotImplementedError
        el = self.element_number(xi)
        Qe = self.Q[self.elDOF[el]]
        qe = q[self.elDOF[el]]

        _, _, K_Gamma_bar0, K_Kappa_bar0 = self._eval(Qe, xi)

        J_0 = norm(K_Gamma_bar0)
        K_Gamma0 = K_Gamma_bar0 / J_0
        K_Kappa0 = K_Kappa_bar0 / J_0

        _, _, K_Gamma_bar, K_Kappa_bar = self._eval(qe, xi)

        J = norm(K_Gamma_bar)
        K_Gamma_DB = K_Gamma_bar / J
        K_Kappa_DB = K_Kappa_bar / J

        if mixed:
            N_n = N_m = self.basis_functions_n(xi).flatten()
            K_n = np.zeros(3, dtype=qe.dtype)
            K_m = np.zeros(3, dtype=qe.dtype)

            for node in range(self.nnodes_element_n):
                n_node = qe[self.nodalDOF_element_n[node]]
                K_n += N_n[node] * n_node
                m_node = qe[self.nodalDOF_element_m[node]]
                K_m += N_m[node] * m_node

            K_Gamma = self.material_model.K_gam_comp(K_n, K_Gamma0)
            K_Kappa = self.material_model.K_kap_comp(K_m, K_Kappa0)

            return K_Gamma, K_Kappa, K_Gamma_DB, K_Kappa_DB
        else:
            return K_Gamma_DB, K_Kappa_DB

    if False:

        def f_pot_el(self, qe, el):
            f_pot_el = np.zeros(self.nu_element, dtype=qe.dtype)

            for i in range(self.nquadrature):
                # extract reference state variables
                qpi = self.qp[el, i]
                qwi = self.qw[el, i]
                J = self.J[el, i]
                K_Gamma0 = self.K_Gamma0[el, i]
                K_Kappa0 = self.K_Kappa0[el, i]

                # evaluate required quantities
                _, A_IK, K_Gamma_bar, K_Kappa_bar = self._eval(qe, qpi)

                # axial and shear strains
                K_Gamma = K_Gamma_bar / J

                # torsional and flexural strains
                K_Kappa = K_Kappa_bar / J

                if self.mixed:
                    # interpolation of the n and m
                    K_n = np.zeros(3, dtype=qe.dtype)
                    K_m = np.zeros(3, dtype=qe.dtype)
                    for node in range(self.nnodes_element_n):
                        n_node = qe[self.nodalDOF_element_n[node]]
                        K_n += self.N_n[el, i, node] * n_node

                    for node in range(self.nnodes_element_m):
                        m_node = qe[self.nodalDOF_element_m[node]]
                        K_m += self.N_m[el, i, node] * m_node

                    # compute contact forces and couples from partial derivatives of
                    # the strain energy function w.r.t. strain measures
                    K_gam_comp = self.material_model.K_gam_comp(K_n, K_Gamma0)
                    K_kap_comp = self.material_model.K_kap_comp(K_m, K_Kappa0)

                    ############################
                    # virtual work contributions of the internal stresses
                    ############################
                    for node in range(self.nnodes_element_n):
                        f_pot_el[self.nodalDOF_element_n_u[node]] -= (
                            self.N_n[el, i, node]
                            * (K_Gamma_bar - J * K_gam_comp)
                            * qwi
                            # for shear-rigid and inextensible rod
                            # self.N_n[el, i, node] * (K_Gamma_bar - J * np.array([1, 0, 0])) * qwi
                            # for shear-rigid
                            # self.N_n[el, i, node] * (K_Gamma_bar - J * np.array([K_gam_comp[0], 0, 0])) * qwi
                        )

                    for node in range(self.nnodes_element_m):
                        f_pot_el[self.nodalDOF_element_m_u[node]] -= (
                            self.N_m[el, i, node]
                            * (K_Kappa_bar - J * K_kap_comp)
                            * qwi
                            # For Ribbons
                            # self.N_m[el, i, node] * (K_Kappa_bar - J * np.array([K_kap_comp[0], K_kap_comp[1], 0])) * qwi
                        )

                else:
                    # compute contact forces and couples from partial derivatives of
                    # the strain energy function w.r.t. strain measures
                    K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
                    K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)

                ############################
                # virtual work contributions
                ############################
                for node in range(self.nnodes_element_r):
                    f_pot_el[self.nodalDOF_element_r_u[node]] -= (
                        self.N_r_xi[el, i, node] * A_IK @ K_n * qwi
                    )

                for node in range(self.nnodes_element_psi):
                    f_pot_el[self.nodalDOF_element_psi_u[node]] -= (
                        self.N_psi_xi[el, i, node] * K_m * qwi
                    )

                    f_pot_el[self.nodalDOF_element_psi_u[node]] += (
                        self.N_psi[el, i, node]
                        * (cross3(K_Gamma_bar, K_n) + cross3(K_Kappa_bar, K_m))
                        * qwi
                    )

            return f_pot_el

        def f_pot_el_q(self, qe, el):
            if not hasattr(self, "_deval"):
                warnings.warn(
                    "Class derived from TimoshenkoPetrovGalerkinBase does not implement _deval. We use a numerical Jacobian!"
                )
                return approx_fprime(
                    qe, lambda qe: self.f_pot_el(qe, el), method="3-point", eps=1e-6
                )
            else:
                f_pot_q_el = np.zeros((self.nu_element, self.nq_element), dtype=float)

                for i in range(self.nquadrature):
                    # extract reference state variables
                    qpi = self.qp[el, i]
                    qwi = self.qw[el, i]
                    J = self.J[el, i]
                    K_Gamma0 = self.K_Gamma0[el, i]
                    K_Kappa0 = self.K_Kappa0[el, i]

                    # evaluate required quantities
                    (
                        r_OP,
                        A_IK,
                        K_Gamma_bar,
                        K_Kappa_bar,
                        r_OP_qe,
                        A_IK_qe,
                        K_Gamma_bar_qe,
                        K_Kappa_bar_qe,
                    ) = self._deval(qe, qpi)

                    # axial and shear strains
                    K_Gamma = K_Gamma_bar / J
                    K_Gamma_qe = K_Gamma_bar_qe / J

                    # torsional and flexural strains
                    K_Kappa = K_Kappa_bar / J
                    K_Kappa_qe = K_Kappa_bar_qe / J

                    if self.mixed:
                        # interpolation of the n and m
                        K_n = np.zeros(3, dtype=qe.dtype)
                        K_n_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
                        K_m = np.zeros(3, dtype=qe.dtype)
                        K_m_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)

                        for node in range(self.nnodes_element_n):
                            nodalDOF_n = self.nodalDOF_element_n[node]
                            n_node = qe[self.nodalDOF_element_n[node]]
                            K_n += self.N_n[el, i, node] * n_node
                            K_n_qe[:, nodalDOF_n] += self.N_n[el, i, node] * np.eye(
                                3, dtype=float
                            )

                        for node in range(self.nnodes_element_m):
                            nodalDOF_m = self.nodalDOF_element_m[node]
                            m_node = qe[self.nodalDOF_element_m[node]]
                            K_m += self.N_m[el, i, node] * m_node
                            K_m_qe[:, nodalDOF_m] += self.N_m[el, i, node] * np.eye(
                                3, dtype=float
                            )

                        C_n_inv = self.material_model.C_n_inverse()
                        C_m_inv = self.material_model.C_m_inverse()

                        ############################
                        # third contribution
                        ############################
                        for node in range(self.nnodes_element_n):
                            f_pot_q_el[self.nodalDOF_element_n_u[node], :] -= (
                                self.N_n[el, i, node]
                                * (K_Gamma_bar_qe - J * C_n_inv @ K_n_qe)
                                * qwi
                            )

                        for node in range(self.nnodes_element_m):
                            f_pot_q_el[self.nodalDOF_element_m_u[node], :] -= (
                                self.N_m[el, i, node]
                                * (K_Kappa_bar_qe - J * C_m_inv @ K_m_qe)
                                * qwi
                            )

                    else:
                        # compute contact forces and couples from partial derivatives of
                        # the strain energy function w.r.t. strain measures
                        K_n = self.material_model.K_n(
                            K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
                        )
                        K_n_K_Gamma = self.material_model.K_n_K_Gamma(
                            K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
                        )
                        K_n_K_Kappa = self.material_model.K_n_K_Kappa(
                            K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
                        )
                        K_n_qe = K_n_K_Gamma @ K_Gamma_qe + K_n_K_Kappa @ K_Kappa_qe

                        K_m = self.material_model.K_m(
                            K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
                        )
                        K_m_K_Gamma = self.material_model.K_m_K_Gamma(
                            K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
                        )
                        K_m_K_Kappa = self.material_model.K_m_K_Kappa(
                            K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
                        )
                        K_m_qe = K_m_K_Gamma @ K_Gamma_qe + K_m_K_Kappa @ K_Kappa_qe

                    ############################
                    # virtual work contributions
                    ############################
                    for node in range(self.nnodes_element_r):
                        f_pot_q_el[self.nodalDOF_element_r[node], :] -= (
                            self.N_r_xi[el, i, node]
                            * qwi
                            * (np.einsum("ikj,k->ij", A_IK_qe, K_n) + A_IK @ K_n_qe)
                        )

                    for node in range(self.nnodes_element_psi):
                        f_pot_q_el[self.nodalDOF_element_psi_u[node], :] += (
                            self.N_psi[el, i, node]
                            * qwi
                            * (
                                ax2skew(K_Gamma_bar) @ K_n_qe
                                - ax2skew(K_n) @ K_Gamma_bar_qe
                            )
                        )

                        f_pot_q_el[self.nodalDOF_element_psi_u[node], :] -= (
                            self.N_psi_xi[el, i, node] * qwi * K_m_qe
                        )

                        f_pot_q_el[self.nodalDOF_element_psi_u[node], :] += (
                            self.N_psi[el, i, node]
                            * qwi
                            * (
                                ax2skew(K_Kappa_bar) @ K_m_qe
                                - ax2skew(K_m) @ K_Kappa_bar_qe
                            )
                        )

                return f_pot_q_el

                # f_pot_q_el_num = approx_fprime(
                #     qe, lambda qe: self.f_pot_el(qe, el), eps=1.0e-10, method="cs"
                # )
                # # f_pot_q_el_num = approx_fprime(
                # #     qe, lambda qe: self.f_pot_el(qe, el), eps=5.0e-6, method="2-point"
                # # )
                # diff = f_pot_q_el - f_pot_q_el_num
                # error = np.linalg.norm(diff)
                # print(f"error f_pot_q_el: {error}")
                # return f_pot_q_el_num

    def c(self, t, q, u, la_c):
        c = np.zeros(self.nla_c, dtype=np.common_type(q, u, la_c))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_la_c = self.elDOF_la_c[el]
            c[elDOF_la_c] = self.c_el(q[elDOF], la_c[elDOF_la_c], el)
        return c

    def c_el(self, qe, la_ce, el):
        c_el = np.zeros(self.nla_c_element, dtype=np.common_type(qe, la_ce))

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            J = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # evaluate required quantities
            _, _, K_Gamma_bar, K_Kappa_bar = self._eval(qe, qpi)

            # interpolation of internal forces and moments
            K_n = np.zeros(3, dtype=la_ce.dtype)
            K_m = np.zeros(3, dtype=la_ce.dtype)
            for node in range(self.nnodes_element_n):
                n_node = la_ce[self.nodalDOF_element_n[node]]
                K_n += self.N_n[el, i, node] * n_node

            for node in range(self.nnodes_element_m):
                m_node = la_ce[self.nodalDOF_element_m[node]]
                K_m += self.N_m[el, i, node] * m_node

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            C_n_inv = self.material_model.C_n_inv
            C_m_inv = self.material_model.C_m_inv

            # TODO: Store K_Gamma_bar0 and K_Kappa_bar0
            c_n = (J * C_n_inv @ K_n - (K_Gamma_bar - J * K_Gamma0)) * qwi
            c_m = (J * C_m_inv @ K_m - (K_Kappa_bar - J * K_Kappa0)) * qwi

            for node in range(self.nnodes_element_n):
                c_el[self.nodalDOF_element_n[node]] += self.N_n[el, i, node] * c_n

            for node in range(self.nnodes_element_m):
                c_el[self.nodalDOF_element_m[node]] += self.N_m[el, i, node] * c_m

        return c_el

    def c_la_c(self, t, q, u, la_c):
        coo = CooMatrix((self.nla_c, self.nla_c))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_la_c = self.elDOF_la_c[el]
            coo[elDOF_la_c, elDOF_la_c] = self.c_la_c_el(q[elDOF], la_c[elDOF_la_c], el)
        return coo

    def c_la_c_el(self, qe, la_ce, el):
        c_la_c_el = np.zeros((self.nla_c_element, self.nla_c_element), dtype=np.common_type(qe, la_ce))
        for i in range(self.nquadrature):
            qwi = self.qw[el, i]
            J = self.J[el, i]

            C_n_inv = self.material_model.C_n_inv
            C_m_inv = self.material_model.C_m_inv

            for node_n1 in range(self.nnodes_element_n):
                    nodalDOF_n1 = self.nodalDOF_element_n[node_n1]
                    for node_n2 in range(self.nnodes_element_n):
                        nodalDOF_n2 = self.nodalDOF_element_n[node_n2]
                        c_la_c_el[nodalDOF_n1[:, None], nodalDOF_n2] += (
                           self.N_n[el, i, node_n1] * C_n_inv * self.N_n[el, i, node_n2]* J * qwi
                        )

            for node_m1 in range(self.nnodes_element_m):
                    nodalDOF_m1 = self.nodalDOF_element_m[node_m1]
                    for node_m2 in range(self.nnodes_element_m):
                        nodalDOF_m2 = self.nodalDOF_element_m[node_m2]
                        c_la_c_el[nodalDOF_m1[:, None], nodalDOF_m2] += (
                           self.N_m[el, i, node_m1] * C_m_inv * self.N_m[el, i, node_n2]* J * qwi
                        )

        # c_la_c_el_num = approx_fprime(la_ce, lambda la_c: self.c_el(qe, la_c, el), method="cs")
        # diff = np.linalg.norm(c_la_c_el_num - c_la_c_el)
        # print(diff)

        return c_la_c_el

    def c_q(self, t, q, u, la_c):
        coo = CooMatrix((self.nla_c, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_la_c = self.elDOF_la_c[el]
            coo[elDOF_la_c, elDOF] = self.c_q_el(q[elDOF], la_c[elDOF_la_c], el)
        return coo

    def c_q_el(self, qe, la_ce, el):

        c_q_el = np.zeros((self.nla_c_element, self.nq_element), dtype=np.common_type(qe, la_ce))
        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            J = self.J[el, i]

            # evaluate required quantities
            _, _, _, _, _, _, K_Gamma_bar_qe, K_Kappa_bar_qe, = self._deval(qe, qpi)

            for node in range(self.nnodes_element_n):
                nodalDOF_n = self.nodalDOF_element_n[node]
                c_q_el[nodalDOF_n, :] -= (self.N_n[el, i, node] * K_Gamma_bar_qe * qwi)
            
            for node in range(self.nnodes_element_m):
                nodalDOF_m = self.nodalDOF_element_m[node]
                c_q_el[nodalDOF_m, :] -= (self.N_m[el, i, node] * K_Kappa_bar_qe * qwi)
        
        # c_q_el_num = approx_fprime(qe, lambda q: self.c_el(q, la_ce, el), method="cs")
        # diff = np.linalg.norm(c_q_el_num - c_q_el)
        # print(diff)

        return c_q_el

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
            _, A_IK, K_Gamma_bar, K_Kappa_bar = self._eval(qe, qpi)

            ############################
            # virtual work contributions
            ############################
            for node_r in range(self.nnodes_element_r):
                nodalDOF_r = self.nodalDOF_element_r_u[node_r]
                N_r_xi = self.N_r_xi[el, i, node_r]
                for node_n in range(self.nnodes_element_n):
                    nodalDOF_n = self.nodalDOF_element_n[node_n]
                    W_c_el[nodalDOF_r[:, None], nodalDOF_n] -= (
                        N_r_xi * A_IK * self.N_n[el, i, node_n] * qwi
                    )

            for node_psi in range(self.nnodes_element_psi):
                nodalDOF_psi = self.nodalDOF_element_psi_u[node_psi]
                N_psi = self.N_psi[el, i, node_psi]
                N_psi_xi = self.N_psi_xi[el, i, node_psi]

                for node_m in range(self.nnodes_element_m):
                    nodalDOF_m = self.nodalDOF_element_m[node_m]
                    W_c_el[nodalDOF_psi[:, None], nodalDOF_m] -= (
                        (N_psi_xi * np.eye(3) - N_psi * ax2skew(K_Kappa_bar))
                        * self.N_m[el, i, node_m]
                        * qwi
                    )

                for node_n in range(self.nnodes_element_n):
                    nodalDOF_n = self.nodalDOF_element_n[node_n]
                    W_c_el[nodalDOF_psi[:, None], nodalDOF_n] += (
                        N_psi * ax2skew(K_Gamma_bar) * self.N_n[el, i, node_n] * qwi
                    )

        return W_c_el

    def Wla_c_q(self, t, q, la_c):
        coo = CooMatrix((self.nu, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            elDOF_la_c = self.elDOF_la_c[el]
            coo[elDOF_u, elDOF] = self.Wla_c_q_el(q[elDOF], la_c[elDOF_la_c], el)
        return coo

    # TODO:
    def Wla_c_q_el(self, qe, la_ce, el):
        Wla_c_q_el = np.zeros((self.nu_element, self.nq_element), dtype=np.common_type(qe, la_ce))

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            J = self.J[el, i]

            # evaluate required quantities
            (
                r_OP,
                A_IK,
                K_Gamma_bar,
                K_Kappa_bar,
                r_OP_qe,
                A_IK_qe,
                K_Gamma_bar_qe,
                K_Kappa_bar_qe,
            ) = self._deval(qe, qpi)

            # interpolation of the n and m
            K_n = np.zeros(3, dtype=qe.dtype)
            K_m = np.zeros(3, dtype=qe.dtype)

            for node in range(self.nnodes_element_n):
                n_node = la_ce[self.nodalDOF_element_n[node]]
                K_n += self.N_n[el, i, node] * n_node

            for node in range(self.nnodes_element_m):
                m_node = la_ce[self.nodalDOF_element_m[node]]
                K_m += self.N_m[el, i, node] * m_node

            for node in range(self.nnodes_element_r):
                Wla_c_q_el[self.nodalDOF_element_r_u[node], :] -= (
                    self.N_r_xi[el, i, node] * qwi * (np.einsum("ikj,k->ij", A_IK_qe, K_n))
                )

            for node in range(self.nnodes_element_psi):
                Wla_c_q_el[self.nodalDOF_element_psi_u[node], :] += (
                    self.N_psi[el, i, node]
                    * qwi
                    * (
                        - ax2skew(K_n) @ K_Gamma_bar_qe
                        - ax2skew(K_m) @ K_Kappa_bar_qe
                    )
                )

        # Wla_c_q_el_num = approx_fprime(qe, lambda q: self.W_c_el(q, el) @ la_ce)
        # diff = np.linalg.norm(Wla_c_q_el_num - Wla_c_q_el)
        # print(diff)

        return Wla_c_q_el
    

    #########################################
    # equations of motion
    #########################################
    def assembler_callback(self):
        if self.constant_mass_matrix:
            self._M_coo()

    def M_el_constant(self, el):
        M_el = np.zeros((self.nu_element, self.nu_element), dtype=float)

        for i in range(self.nquadrature_dyn):
            # extract reference state variables
            qwi = self.qw_dyn[el, i]
            Ji = self.J_dyn[el, i]

            # delta_r A_rho0 r_ddot part
            M_el_r_r = np.eye(3) * self.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N_r_dyn[el, i, node_a] * self.N_r_dyn[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_psi_psi = self.K_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi_u[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi_u[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
                        self.N_psi_dyn[el, i, node_a] * self.N_psi_dyn[el, i, node_b]
                    )

        return M_el

    def M_el(self, qe, el):
        M_el = np.zeros((self.nu_element, self.nu_element), dtype=float)

        for i in range(self.nquadrature_dyn):
            # extract reference state variables
            qwi = self.qw_dyn[el, i]
            Ji = self.J_dyn[el, i]

            # delta_r A_rho0 r_ddot part
            M_el_r_r = np.eye(3) * self.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N_r_dyn[el, i, node_a] * self.N_r_dyn[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_psi_psi = self.K_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi_u[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi_u[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
                        self.N_psi_dyn[el, i, node_a] * self.N_psi_dyn[el, i, node_b]
                    )

            # For non symmetric cross sections there are also other parts
            # involved in the mass matrix. These parts are configuration
            # dependent and lead to configuration dependent mass matrix.
            _, A_IK, _, _ = self.eval(qe, self.qp_dyn[el, i])
            M_el_r_psi = A_IK @ self.K_S_rho0 * Ji * qwi
            M_el_psi_r = A_IK @ self.K_S_rho0 * Ji * qwi

            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi_u[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_psi * (
                        self.N_r_dyn[el, i, node_a] * self.N_psi_dyn[el, i, node_b]
                    )
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi_u[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_r * (
                        self.N_psi_dyn[el, i, node_a] * self.N_r_dyn[el, i, node_b]
                    )

        return M_el

    def _M_coo(self):
        self.__M = CooMatrix((self.nu, self.nu))
        for el in range(self.nelement):
            # extract element degrees of freedom
            elDOF_u = self.elDOF_u[el]

            # sparse assemble element mass matrix
            # self.__M[self.uDOF[elDOF_u], self.uDOF[elDOF_u]] = self.M_el_constant(
            #     el
            # )
            self.__M[elDOF_u, elDOF_u] = self.M_el_constant(el)

    def M(self, t, q):
        if self.constant_mass_matrix:
            return self.__M
        else:
            coo = CooMatrix((self.nu, self.nu))
            for el in range(self.nelement):
                # extract element degrees of freedom
                elDOF = self.elDOF[el]
                elDOF_u = self.elDOF_u[el]

                # sparse assemble element mass matrix
                coo[elDOF_u, elDOF_u] = self.M_el(q[elDOF], el)

            return coo

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

            K_omega_IK = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_psi):
                K_omega_IK += (
                    self.N_psi_dyn[el, i, node] * ue[self.nodalDOF_element_psi_u[node]]
                )

            # delta_r A_rho0 r_ddot part
            E_kin_el += 0.5 * (v_P @ v_P) * self.A_rho0 * Ji * qwi

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            E_kin_el += 0.5 * (K_omega_IK @ self.K_I_rho0 @ K_omega_IK) * Ji * qwi

        # E_kin_el = ue @ self.M_el_constant(el) @ ue

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

            linear_momentum_el += v_P * self.A_rho0 * Ji * qwi

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

            K_omega_IK = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_psi):
                K_omega_IK += (
                    self.N_psi_dyn[el, i, node] * ue[self.nodalDOF_element_psi_u[node]]
                )

            A_IK = self.A_IK(t, qe, (qpi,))

            angular_momentum_el += (
                (cross3(r_OP, v_P) * self.A_rho0 + A_IK @ (self.K_I_rho0 @ K_omega_IK))
                * Ji
                * qwi
            )

        return angular_momentum_el

    def f_gyr_el(self, t, qe, ue, el):
        if not self.constant_mass_matrix:
            raise NotImplementedError(
                "Gyroscopic forces are not implemented for nonzero K_S_rho0."
            )

        f_gyr_el = np.zeros(self.nu_element, dtype=np.common_type(qe, ue))

        for i in range(self.nquadrature_dyn):
            # interpoalte angular velocity
            K_Omega = np.zeros(3, dtype=np.common_type(qe, ue))
            for node in range(self.nnodes_element_psi):
                K_Omega += (
                    self.N_psi_dyn[el, i, node] * ue[self.nodalDOF_element_psi_u[node]]
                )

            # vector of gyroscopic forces
            f_gyr_el_psi = (
                cross3(K_Omega, self.K_I_rho0 @ K_Omega)
                * self.J_dyn[el, i]
                * self.qw_dyn[el, i]
            )

            # multiply vector of gyroscopic forces with nodal virtual rotations
            for node in range(self.nnodes_element_psi):
                f_gyr_el[self.nodalDOF_element_psi_u[node]] += (
                    self.N_psi_dyn[el, i, node] * f_gyr_el_psi
                )

        return f_gyr_el

    def f_gyr_u_el(self, t, qe, ue, el):
        f_gyr_u_el = np.zeros((self.nu_element, self.nu_element), dtype=float)

        for i in range(self.nquadrature_dyn):
            # interpoalte angular velocity
            K_Omega = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_psi):
                K_Omega += (
                    self.N_psi_dyn[el, i, node] * ue[self.nodalDOF_element_psi_u[node]]
                )

            # derivative of vector of gyroscopic forces
            f_gyr_u_el_psi = (
                ((ax2skew(K_Omega) @ self.K_I_rho0 - ax2skew(self.K_I_rho0 @ K_Omega)))
                * self.J_dyn[el, i]
                * self.qw_dyn[el, i]
            )

            # multiply derivative of gyroscopic force vector with nodal virtual rotations
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi_u[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi_u[node_b]
                    f_gyr_u_el[nodalDOF_a[:, None], nodalDOF_b] += f_gyr_u_el_psi * (
                        self.N_psi_dyn[el, i, node_a] * self.N_psi_dyn[el, i, node_b]
                    )

        return f_gyr_u_el

    def h(self, t, q, u):
        h = np.zeros(self.nu, dtype=np.common_type(q, u))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            h[elDOF_u] -= self.f_gyr_el(t, q[elDOF], u[elDOF_u], el)
        return h

    def h_u(self, t, q, u):
        coo = CooMatrix((self.nu, self.nu))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            coo[elDOF_u, elDOF_u] = -self.f_gyr_u_el(t, q[elDOF], u[elDOF_u], el)
        return coo

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def elDOF_P(self, frame_ID):
        xi = frame_ID[0]
        el = self.element_number(xi)
        return self.elDOF[el]

    def elDOF_P_u(self, frame_ID):
        xi = frame_ID[0]
        el = self.element_number(xi)
        return self.elDOF_u[el]

    def local_qDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def local_uDOF_P(self, frame_ID):
        return self.elDOF_P_u(frame_ID)

    #########################
    # r_OP/ A_IK contribution
    #########################
    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # TODO: correct this function _eval need to qe insteaf of q
        r_OP, A_IK, _, _ = self._eval(q, frame_ID[0])
        return r_OP + A_IK @ K_r_SP

    # TODO: Think of a faster version than using _deval
    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        if not hasattr(self, "_deval"):
            warnings.warn(
                "Class derived from TimoshenkoPetrovGalerkinBase does not implement _deval. We use a numerical Jacobian!"
            )
            return approx_fprime(
                q,
                lambda q: self.r_OP(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP),
                # method="cs",
                # eps=1.0e-12,
                method="3-point",
                eps=1e-6,
            )
        else:
            # evaluate required quantities
            (
                r_OP,
                A_IK,
                _,
                _,
                r_OP_q,
                A_IK_q,
                _,
                _,
            ) = self._deval(q, frame_ID[0])

        return r_OP_q + np.einsum("ijk,j->ik", A_IK_q, K_r_SP)

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        N_r, _ = self.basis_functions_r(frame_ID[0])
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate A_IK and angular velocity in K-frame
        A_IK = self.A_IK(t, q, frame_ID)
        K_Omega = np.zeros(3, dtype=np.common_type(q, u))
        for node in range(self.nnodes_element_psi):
            K_Omega += N_psi[node] * u[self.nodalDOF_element_psi_u[node]]

        # centerline velocity
        v = np.zeros(3, dtype=np.common_type(q, u))
        for node in range(self.nnodes_element_r):
            v += N_r[node] * u[self.nodalDOF_element_r[node]]

        return v + A_IK @ cross3(K_Omega, K_r_SP)

    def v_P_q(self, t, q, u, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # evaluate shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate derivative of A_IK and angular velocity in K-frame
        A_IK_q = self.A_IK_q(t, q, frame_ID)
        K_Omega = np.zeros(3, dtype=np.common_type(q, u))
        for node in range(self.nnodes_element_psi):
            K_Omega += N_psi[node] * u[self.nodalDOF_element_psi_u[node]]

        v_P_q = np.einsum(
            "ijk,j->ik",
            A_IK_q,
            cross3(K_Omega, K_r_SP),
        )
        return v_P_q

        # v_P_q_num = approx_fprime(
        #     q, lambda q: self.v_P(t, q, u, frame_ID, K_r_SP), method="3-point"
        # )
        # diff = v_P_q - v_P_q_num
        # error = np.linalg.norm(diff)
        # print(f"error v_P_q: {error}")
        # return v_P_q_num

    def J_P(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # evaluate required nodal shape functions
        N_r, _ = self.basis_functions_r(frame_ID[0])
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # transformation matrix
        A_IK = self.A_IK(t, q, frame_ID)

        # skew symmetric matrix of K_r_SP
        K_r_SP_tilde = ax2skew(K_r_SP)

        # interpolate centerline and axis angle contributions
        J_P = np.zeros((3, self.nu_element), dtype=q.dtype)
        for node in range(self.nnodes_element_r):
            J_P[:, self.nodalDOF_element_r[node]] += N_r[node] * np.eye(3)
        for node in range(self.nnodes_element_psi):
            J_P[:, self.nodalDOF_element_psi_u[node]] -= (
                N_psi[node] * A_IK @ K_r_SP_tilde
            )

        return J_P

        # J_P_num = approx_fprime(
        #     np.zeros(self.nq_element, dtype=float),
        #     lambda u: self.v_P(t, q, u, frame_ID, K_r_SP),
        # )
        # diff = J_P_num - J_P
        # error = np.linalg.norm(diff)
        # print(f"error J_P: {error}")
        # return J_P_num

    def J_P_q(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # evaluate required nodal shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        K_r_SP_tilde = ax2skew(K_r_SP)
        A_IK_q = self.A_IK_q(t, q, frame_ID)
        prod = np.einsum("ijl,jk", A_IK_q, K_r_SP_tilde)

        # interpolate axis angle contributions since centerline contributon is
        # zero
        J_P_q = np.zeros((3, self.nu_element, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_psi):
            nodalDOF_psi_u = self.nodalDOF_element_psi_u[node]
            J_P_q[:, nodalDOF_psi_u] -= N_psi[node] * prod

        return J_P_q

        # J_P_q_num = approx_fprime(
        #     q, lambda q: self.J_P(t, q, frame_ID, K_r_SP), method="3-point"
        # )
        # diff = J_P_q_num - J_P_q
        # error = np.linalg.norm(diff)
        # print(f"error J_P_q: {error}")
        # return J_P_q_num

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        N_r, _ = self.basis_functions_r(frame_ID[0])

        # interpolate orientation
        A_IK = self.A_IK(t, q, frame_ID)

        # centerline acceleration
        a = np.zeros(3, dtype=np.common_type(q, u, u_dot))
        for node in range(self.nnodes_element_r):
            a += N_r[node] * u_dot[self.nodalDOF_element_r[node]]

        # angular velocity and acceleration in K-frame
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID)

        # rigid body formular
        return a + A_IK @ (
            cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP))
        )

    def a_P_q(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID)
        a_P_q = np.einsum(
            "ijk,j->ik",
            self.A_IK_q(t, q, frame_ID),
            cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP)),
        )
        return a_P_q

        # a_P_q_num = approx_fprime(
        #     q,
        #     lambda q: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP),
        #     method="3-point",
        # )
        # diff = a_P_q_num - a_P_q
        # error = np.linalg.norm(diff)
        # print(f"error a_P_q: {error}")
        # return a_P_q_num

    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        local = -self.A_IK(t, q, frame_ID) @ (
            ax2skew(cross3(K_Omega, K_r_SP)) + ax2skew(K_Omega) @ ax2skew(K_r_SP)
        )

        N, _ = self.basis_functions_r(frame_ID[0])
        a_P_u = np.zeros((3, self.nu_element), dtype=float)
        for node in range(self.nnodes_element_r):
            a_P_u[:, self.nodalDOF_element_psi_u[node]] += N[node] * local

        return a_P_u

        # a_P_u_num = approx_fprime(
        #     u,
        #     lambda u: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP),
        #     method="3-point",
        # )
        # diff = a_P_u_num - a_P_u
        # error = np.linalg.norm(diff)
        # print(f"error a_P_u: {error}")
        # return a_P_u_num

    def K_Omega(self, t, q, u, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        angular velocities in the K-frame.
        """
        N_psi, _ = self.basis_functions_psi(frame_ID[0])
        K_Omega = np.zeros(3, dtype=np.common_type(q, u))
        for node in range(self.nnodes_element_psi):
            K_Omega += N_psi[node] * u[self.nodalDOF_element_psi_u[node]]
        return K_Omega

    def K_Omega_q(self, t, q, u, frame_ID):
        return np.zeros((3, self.nq_element), dtype=float)

    def K_J_R(self, t, q, frame_ID):
        N_psi, _ = self.basis_functions_psi(frame_ID[0])
        K_J_R = np.zeros((3, self.nu_element), dtype=float)
        for node in range(self.nnodes_element_psi):
            K_J_R[:, self.nodalDOF_element_psi_u[node]] += N_psi[node] * np.eye(
                3, dtype=float
            )
        return K_J_R

    def K_J_R_q(self, t, q, frame_ID):
        return np.zeros((3, self.nu_element, self.nq_element), dtype=float)

    def K_Psi(self, t, q, u, u_dot, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        time derivative of the angular velocities in the K-frame.
        """
        N_psi, _ = self.basis_functions_psi(frame_ID[0])
        K_Psi = np.zeros(3, dtype=np.common_type(q, u, u_dot))
        for node in range(self.nnodes_element_psi):
            K_Psi += N_psi[node] * u_dot[self.nodalDOF_element_psi_u[node]]
        return K_Psi

    def K_Psi_q(self, t, q, u, u_dot, frame_ID):
        return np.zeros((3, self.nq_element), dtype=float)

    def K_Psi_u(self, t, q, u, u_dot, frame_ID):
        return np.zeros((3, self.nu_element), dtype=float)

    ####################################################
    # body force
    ####################################################
    def distributed_force1D_pot_el(self, force, t, qe, el):
        Ve = 0.0

        # # TODO: We should use self.nquadrature_dyn!
        # for i in range(self.nquadrature):
        #     # extract reference state variables
        #     qpi = self.qp[el, i]
        #     qwi = self.qw[el, i]
        #     Ji = self.J[el, i]

        #     # interpolate centerline position
        #     r_C = np.zeros(3, dtype=float)
        #     for node in range(self.nnodes_element_r):
        #         r_C += self.N_r[el, i, node] * qe[self.nodalDOF_element_r[node]]

        #     # compute potential value at given quadrature point
        #     Ve -= (r_C @ force(t, qpi)) * Ji * qwi

        for i in range(self.nquadrature_dyn):
            # extract reference state variables
            qpi = self.qp_dyn[el, i]
            qwi = self.qw_dyn[el, i]
            Ji = self.J_dyn[el, i]

            # interpolate centerline position
            r_C = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_r):
                r_C += self.N_r_dyn[el, i, node] * qe[self.nodalDOF_element_r[node]]

            # compute potential value at given quadrature point
            Ve -= (r_C @ force(t, qpi)) * Ji * qwi

        return Ve

    def distributed_force1D_pot(self, t, q, force):
        V = 0
        for el in range(self.nelement):
            qe = q[self.elDOF[el]]
            V += self.distributed_force1D_pot_el(force, t, qe, el)
        return V

    # TODO: Decide which number of quadrature points should be used here?
    def distributed_force1D_el(self, force, t, el):
        fe = np.zeros(self.nu_element, dtype=float)

        # for i in range(self.nquadrature):
        #     # extract reference state variables
        #     qpi = self.qp[el, i]
        #     qwi = self.qw[el, i]
        #     Ji = self.J[el, i]

        #     # compute local force vector
        #     fe_r = force(t, qpi) * Ji * qwi

        #     # multiply local force vector with variation of centerline
        #     for node in range(self.nnodes_element_r):
        #         fe[self.nodalDOF_element_r[node]] += self.N_r[el, i, node] * fe_r

        for i in range(self.nquadrature_dyn):
            # extract reference state variables
            qpi = self.qp_dyn[el, i]
            qwi = self.qw_dyn[el, i]
            Ji = self.J_dyn[el, i]

            # compute local force vector
            fe_r = force(t, qpi) * Ji * qwi

            # multiply local force vector with variation of centerline
            for node in range(self.nnodes_element_r):
                fe[self.nodalDOF_element_r[node]] += self.N_r_dyn[el, i, node] * fe_r

        return fe

    def distributed_force1D(self, t, q, force):
        f = np.zeros(self.nu, dtype=float)
        for el in range(self.nelement):
            f[self.elDOF_u[el]] += self.distributed_force1D_el(force, t, el)
        return f

    def distributed_force1D_q(self, t, q, force):
        return None
