import numpy as np
from abc import ABC, abstractmethod
import warnings

from cardillo.math import (
    pi,
    e1,
    norm,
    cross3,
    ax2skew,
    Exp_SO3,
    Exp_SO3_psi,
    A_IK_basic,
    Log_SO3,
    smallest_rotation,
    T_SO3,
    T_SO3_psi,
    # T_SO3_inv,
    # T_SO3_inv_psi,
    # T_SO3_dot,
    approx_fprime,
    complex_atan2,
)

from cardillo.discretization.bezier import L2_projection_Bezier_curve
from cardillo.beams._cross_section import (
    CrossSection,
    CircularCrossSection,
    RectangularCrossSection,
)
from cardillo.beams._base import RodExportBase

from cardillo.utility.coo import Coo
from cardillo.discretization.hermite import HermiteNodeVector
from cardillo.discretization.lagrange import LagrangeKnotVector
from cardillo.discretization.mesh1D import Mesh1D


class Kirchhoff(RodExportBase):
    def __init__(
        self,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelement,
        nquadrature,
        nquadrature_dyn,
        Q,
        q0=None,
        u0=None,
    ):
        """Total Lagrangian Kirchhoff rod."""

        super().__init__(cross_section)

        # beam properties
        self.material_model = material_model  # material model
        self.A_rho0 = A_rho0
        self.K_S_rho0 = K_S_rho0
        self.K_I_rho0 = K_I_rho0

        # distinguish between inertia quadrature and other parts
        self.nquadrature_dyn = nquadrature_dyn
        self.nquadrature = nquadrature
        self.nelement = nelement
        print(f"nquadrature_dyn: {nquadrature_dyn}")
        print(f"nquadrature: {nquadrature}")

        self.knot_vector_r = HermiteNodeVector(3, nelement)
        self.knot_vector_la = LagrangeKnotVector(1, nelement)

        # build mesh objects
        self.mesh_r = Mesh1D(
            self.knot_vector_r,
            nquadrature,
            dim_q=3,
            derivative_order=2,
            basis="Hermite",
            quadrature="Gauss",
        )

        self.mesh_la = Mesh1D(
            self.knot_vector_la,
            nquadrature,
            dim_q=1,
            derivative_order=1,
            basis="Lagrange",
            quadrature="Gauss",
        )

        # total number of nodes
        self.nnodes_r = self.mesh_r.nnodes
        self.nnodes_la = self.mesh_la.nnodes

        # number of nodes per element
        self.nnodes_element_r = self.mesh_r.nnodes_per_element
        self.nnodes_element_la = self.mesh_la.nnodes_per_element

        # total number of generalized coordinates and velocities
        self.nq_r = self.mesh_r.nq
        self.nq_la = self.mesh_la.nq
        self.nq = self.nq_r + self.nq_la
        self.nu_r = self.mesh_r.nu
        self.nu_la = self.mesh_la.nu
        self.nu = self.nu_r + self.nu_la

        # number of generalized coordiantes and velocities per element
        self.nq_element_r = self.mesh_r.nq_per_element
        self.nq_element_la = self.mesh_la.nq_per_element
        self.nq_element = self.nq_element_r + self.nq_element_la
        self.nu_element_r = self.mesh_r.nu_per_element
        self.nu_element_la = self.mesh_la.nu_per_element
        self.nu_element = self.nu_element_r + self.nu_element_la

        # global element connectivity
        # qe = q[elDOF[e]] "q^e = C_e,q q"
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_la = self.mesh_la.elDOF + self.nq_r
        self.elDOF_r_u = self.mesh_r.elDOF_u
        self.elDOF_la_u = self.mesh_la.elDOF_u + self.nu_r

        # global nodal
        self.nodalDOF_r = self.mesh_r.nodalDOF
        self.nodalDOF_la = self.mesh_la.nodalDOF + self.nq_r
        self.nodalDOF_r_u = self.mesh_r.nodalDOF_u
        self.nodalDOF_la_u = self.mesh_la.nodalDOF_u + self.nu_r

        # nodal connectivity on element level
        # r_OP_i^e = C_r,i^e * C_e,q q = C_r,i^e * q^e
        # r_OPi = qe[nodelDOF_element_r[i]]
        self.nodalDOF_element_r = self.mesh_r.nodalDOF_element
        self.nodalDOF_element_la = self.mesh_la.nodalDOF_element + self.nq_element_r
        self.nodalDOF_element_r_u = self.mesh_r.nodalDOF_element_u
        self.nodalDOF_element_la_u = self.mesh_la.nodalDOF_element_u + self.nu_element_r

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nelement, self.nq_element), dtype=int)
        self.elDOF_u = np.zeros((nelement, self.nu_element), dtype=int)
        for el in range(nelement):
            self.elDOF[el, : self.nq_element_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_element_r :] = self.elDOF_la[el]
            self.elDOF_u[el, : self.nu_element_r] = self.elDOF_r_u[el]
            self.elDOF_u[el, self.nu_element_r :] = self.elDOF_la_u[el]

        # shape functions and their first derivatives
        self.N_r = self.mesh_r.N
        self.N_r_xi = self.mesh_r.N_xi
        self.N_la = self.mesh_la.N
        self.N_la_xi = self.mesh_la.N_xi

        # self.N_r_dyn = self.mesh_r_dyn.N
        # self.N_psi_dyn = self.mesh_psi_dyn.N
        self.N_r_dyn = self.N_r
        self.N_psi_dyn = self.N_la

        # quadrature points
        self.qp = self.mesh_r.qp  # quadrature points
        self.qw = self.mesh_r.wp  # quadrature weights
        # self.qp_dyn = self.mesh_r_dyn.qp  # quadrature points for dynamics
        # self.qw_dyn = self.mesh_r_dyn.wp  # quadrature weights for dynamics
        self.qp_dyn = self.qp
        self.qw_dyn = self.qw

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions_r = self.mesh_r.eval_basis
        self.basis_functions_la = self.mesh_la.eval_basis

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q  # reference configuration
        self.q0 = Q.copy() if q0 is None else q0  # initial configuration
        self.u0 = (
            np.zeros(self.nu, dtype=float) if u0 is None else u0
        )  # initial velocities

        # precompute values of the reference configuration in order to save computation time
        # J in Harsch2020b (5)
        self.J = np.zeros((nelement, nquadrature), dtype=float)
        self.J_dyn = np.zeros((nelement, nquadrature_dyn), dtype=float)
        # curvature of the reference configuration
        self.K_Kappa0 = np.zeros((nelement, nquadrature, 3), dtype=float)

        for el in range(nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(nquadrature):
                # current quadrature point
                qpi = self.qp[el, i]

                # evaluate required quantities
                _, _, r_OP_xi, K_Kappa_bar = self._eval(qe, qpi)

                # length of reference tangential vector
                J = norm(r_OP_xi)

                # torsional and flexural strains
                K_Kappa = K_Kappa_bar / J

                # safe precomputed quantities for later
                self.J[el, i] = J
                self.K_Kappa0[el, i] = K_Kappa

            for i in range(nquadrature_dyn):
                # current quadrature point
                qpi = self.qp_dyn[el, i]

                # evaluate required quantities
                _, _, r_OP_xi, K_Kappa_bar = self._eval(qe, qpi)

                # length of reference tangential vector
                self.J_dyn[el, i] = norm(r_OP_xi)

    @staticmethod
    def straight_configuration(
        nelement,
        L,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
    ):
        # number of nodes
        nnodes_r = nelement + 1
        nnodes_la = nelement + 1

        # compute rotation vector from given transformation matrix
        psi0 = Log_SO3(A_IK)

        xis = np.linspace(0, 1, num=nnodes_r)
        q_r = np.zeros((3, 2 * nnodes_r))
        t0 = A_IK @ (L * e1)
        for i, xi in enumerate(xis):
            ri = r_OP + xi * t0
            q_r[:, 2 * i] = ri
            # Note: We store the rotation vectors instead of the tangents!
            q_r[:, 2 * i + 1] = psi0

        # reshape generalized coordinates to nodal ordering
        q_r = q_r.reshape(-1, order="C")
        # q_r = q_r.reshape(-1, order="F")

        # initial stretch is zero
        q_la = norm(t0) * np.ones(nnodes_la, dtype=float)

        return np.concatenate([q_r, q_la])

    if False:

        @staticmethod
        def straight_initial_configuration(
            polynomial_degree_r,
            polynomial_degree_psi,
            basis_r,
            basis_psi,
            nelement,
            L,
            r_OP=np.zeros(3, dtype=float),
            A_IK=np.eye(3, dtype=float),
            v_P=np.zeros(3, dtype=float),
            K_omega_IK=np.zeros(3, dtype=float),
            rotation_parameterization=AxisAngleRotationParameterization(),
            # rotation_parameterization=QuaternionRotationParameterization(),
        ):
            raise NotImplementedError
            if basis_r == "Lagrange":
                nnodes_r = polynomial_degree_r * nelement + 1
            elif basis_r == "B-spline":
                nnodes_r = polynomial_degree_r + nelement
            elif basis_r == "Hermite":
                nnodes_r = nelement + 1
            else:
                raise RuntimeError(f'wrong basis_r: "{basis_r}" was chosen')

            if basis_psi == "Lagrange":
                nnodes_psi = polynomial_degree_psi * nelement + 1
            elif basis_psi == "B-spline":
                nnodes_psi = polynomial_degree_psi + nelement
            elif basis_psi == "Hermite":
                nnodes_psi = nelement + 1
            else:
                raise RuntimeError(f'wrong basis_psi: "{basis_psi}" was chosen')

            #################################
            # compute generalized coordinates
            #################################
            if basis_r == "B-spline" or basis_r == "Lagrange":
                x0 = np.linspace(0, L, num=nnodes_r)
                y0 = np.zeros(nnodes_r)
                z0 = np.zeros(nnodes_r)
                if basis_r == "B-spline":
                    # build Greville abscissae for B-spline basis
                    kv = BSplineKnotVector.uniform(polynomial_degree_r, nelement)
                    for i in range(nnodes_r):
                        x0[i] = np.sum(kv[i + 1 : i + polynomial_degree_r + 1])
                    x0 = x0 * L / polynomial_degree_r

                r_OC0 = np.vstack((x0, y0, z0))
                for i in range(nnodes_r):
                    r_OC0[:, i] = r_OP + A_IK @ r_OC0[:, i]

            elif basis_r == "Hermite":
                xis = np.linspace(0, 1, num=nnodes_r)
                r_OC0 = np.zeros((6, nnodes_r))
                t0 = A_IK @ (L * e1)
                for i, xi in enumerate(xis):
                    ri = r_OP + xi * t0
                    r_OC0[:3, i] = ri
                    r_OC0[3:, i] = t0

            # reshape generalized coordinates to nodal ordering
            q_r = r_OC0.reshape(-1, order="C")

            # we have to extract the rotation vector from the given rotation matrix
            # and set its value for each node
            if basis_psi == "Hermite":
                raise NotImplementedError
            # psi = Log_SO3(A_IK)
            psi = rotation_parameterization.Log_SO3(A_IK)
            q_psi = np.repeat(psi, nnodes_psi)

            ################################
            # compute generalized velocities
            ################################
            # centerline velocities
            v_C0 = np.zeros_like(r_OC0, dtype=float)
            for i in range(nnodes_r):
                v_C0[:, i] = v_P + cross3(
                    A_IK @ K_omega_IK, (r_OC0[:, i] - r_OC0[:, 0])
                )

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
        return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF_r[::2]]).T

    ##################
    # abstract methods
    ##################
    # # @abstractmethod
    # def _eval(self, qe, xi):
    #     """Compute (r_OP, A_IK, K_Gamma_bar, K_Kappa_bar)."""
    #     pass
    def _eval(self, qe, xi):
        # evaluate shape functions
        N_r, N_r_xi, N_r_xi_xi = self.basis_functions_r(xi)

        # extract nodalDOF's
        nodalDOFr0 = self.nodalDOF_element_r[0]
        nodalDOFr1 = self.nodalDOF_element_r[2]
        nodalDOFpsi0 = self.nodalDOF_element_r[1]
        nodalDOFpsi1 = self.nodalDOF_element_r[3]
        nodalDOFla0 = self.nodalDOF_element_la[0, 0]
        nodalDOFla1 = self.nodalDOF_element_la[1, 0]

        # extract nodal quantities
        r_OP0 = qe[nodalDOFr0]
        r_OP1 = qe[nodalDOFr1]
        psi0 = qe[nodalDOFpsi0]
        psi1 = qe[nodalDOFpsi1]
        la0 = qe[nodalDOFla0]
        la1 = qe[nodalDOFla1]

        # nodal transformations
        A_IK0 = Exp_SO3(psi0)
        A_IK1 = Exp_SO3(psi1)

        # nodal directors
        e_x_K0, e_y_K0, e_z_K0 = A_IK0.T
        e_x_K1, e_y_K1, e_z_K1 = A_IK1.T

        # compute nodal tangent vectors
        t0 = la0 * e_x_K0
        t1 = la1 * e_x_K1

        # interpolate position vector and its derivative
        r_OP = N_r[0] * r_OP0 + N_r[1] * t0 + N_r[2] * r_OP1 + N_r[3] * t1
        r_OP_xi = (
            N_r_xi[0] * r_OP0 + N_r_xi[1] * t0 + N_r_xi[2] * r_OP1 + N_r_xi[3] * t1
        )
        r_OP_xi_xi = (
            N_r_xi_xi[0] * r_OP0
            + N_r_xi_xi[1] * t0
            + N_r_xi_xi[2] * r_OP1
            + N_r_xi_xi[3] * t1
        )

        # compute norm of current tangent vector
        j = norm(r_OP_xi)

        # compute first base vector and its derivative
        e_x_J = r_OP_xi / j
        e_x_J_xi = (np.eye(3) - np.outer(e_x_J, e_x_J)) @ r_OP_xi_xi / j

        # rotate to K0-basis
        K0_e_x_J = A_IK0.T @ e_x_J
        # K0_e_x_K1 = A_IK0.T @ e_x_K1

        # smallest rotation from K0_e_x^K0 to K0_r_x_J
        A_K0J = smallest_rotation(e1, K0_e_x_J)
        # K0_e_x_J, K0_e_y_J, K0_e_z_J = A_K0J.T

        # # smallest rotation from K0_e_x^K0 to K0_r_x_J
        # A_K0K1 = smallest_rotation(e1, K0_e_x_K1)
        # K0_e_x_K1, K0_e_y_K1, K0_e_z_K1 = A_K0K1.T

        # # torsion error
        # # TODO: This yields wrong results.
        # sin = K0_e_y_K1 @ K0_e_z_J
        # cos = K0_e_y_K1 @ K0_e_y_J

        # auxiliary functions torsion error
        denom = 1 + e_x_K0 @ e_x_K1
        sin = (e_z_K0 @ e_y_K1) - (e_x_K0 @ e_y_K1) * (e_z_K0 @ e_x_K1) / denom
        cos = (e_y_K0 @ e_y_K1) - (e_x_K0 @ e_y_K1) * (e_y_K0 @ e_x_K1) / denom

        # denom_cross = cross3(e_x_K0, e_x_K1)
        # sin_cross = (
        #     cross3(e_z_K0, e_y_K1)
        #     - cross3(e_x_K0, e_y_K1) * (e_z_K0 @ e_x_K1) / denom
        #     - (e_x_K0 @ e_y_K1) * cross3(e_z_K0, e_x_K1) / denom
        #     + (e_x_K0 @ e_y_K1) * (e_z_K0 @ e_x_K1) / (denom**2) * denom_cross
        # )
        # cos_cross = (
        #     cross3(e_y_K0, e_y_K1)
        #     - cross3(e_x_K0, e_y_K1) * (e_y_K0 @ e_x_K1) / denom
        #     - (e_x_K0 @ e_y_K1) * cross3(e_y_K0, e_x_K1) / denom
        #     + (e_x_K0 @ e_y_K1) * (e_y_K0 @ e_x_K1) / (denom**2) * denom_cross
        # )

        # torsion error
        # alpha = np.arctan2(sin, cos)
        alpha = complex_atan2(sin, cos)

        # alpha_psi0 = (cos * sin_cross - sin * cos_cross) @ A_IK0 @ (T_SO3(psi0))
        # alpha_psi1 = (cos * sin_cross - sin * cos_cross) @ A_IK1 @ (-T_SO3(psi1))

        # alpha_q = np.zeros(self.nq_element, dtype=qe.dtype)
        # alpha_q[nodalDOFpsi0] = alpha_psi0
        # alpha_q[nodalDOFpsi1] = alpha_psi1

        # find element number containing xi
        el = self.element_number(xi)

        # get element interval
        xi0, xi1 = self.knot_vector_r.element_interval(el)

        # second linear Lagrange shape function
        N1_xi = 1.0 / (xi1 - xi0)
        N1 = (xi - xi0) * N1_xi

        # linear correction of torsion error
        A_JK = A_IK_basic(N1 * alpha).x()

        # composed transformation matrix
        A_IK = A_IK0 @ A_K0J @ A_JK

        # curvature
        denom = 1 + e_x_K0 @ e_x_J
        kappa_IK_bar_perp = cross3(e_x_J, e_x_J_xi)
        k_1 = -e_x_K0 @ kappa_IK_bar_perp / denom + alpha

        K_kappa_IK_bar = A_IK.T @ kappa_IK_bar_perp
        K_kappa_IK_bar[0] = k_1

        return r_OP, A_IK, r_OP_xi, K_kappa_IK_bar

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

    # @abstractmethod
    def A_IK(self, t, q, frame_ID):
        _, A_IK, _, _ = self._eval(q, frame_ID[0])
        return A_IK

    # @abstractmethod
    def A_IK_q(self, t, q, frame_ID):
        return approx_fprime(
            q,
            lambda q: self.A_IK(t, q, frame_ID),
            method="2-point",
            eps=1e-6,
        )

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        return u

    def B(self, t, q, coo):
        # trivial kinematic equation for centerline
        coo.extend_diag(np.ones(self.nq), (self.qDOF, self.uDOF))

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    # def step_callback(self, t, q, u):
    #     pass

    ###############################
    # potential and internal forces
    ###############################
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
            K_Kappa0 = self.K_Kappa0[el, i]

            # evaluate required quantities
            _, _, r_OP_xi, K_Kappa_bar = self._eval(qe, qpi)

            # stretch
            j = norm(r_OP_xi)
            lambda_ = j / Ji

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / Ji

            # evaluate strain energy function
            E_pot_el += (
                self.material_model.potential(lambda_, K_Kappa, K_Kappa0) * Ji * qwi
            )

        return E_pot_el

    def f_pot_el(self, qe, el):
        return approx_fprime(
            # qe, lambda qe: -self.E_pot_el(qe, el), method="3-point", eps=1e-6
            qe,
            lambda qe: -self.E_pot_el(qe, el),
            method="cs",
            eps=1e-12,
        )
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

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)

            ############################
            # virtual work contributions
            ############################
            for node in range(self.nnodes_element_r):
                f_pot_el[self.nodalDOF_element_r[node]] -= (
                    self.N_r_xi[el, i, node] * A_IK @ K_n * qwi
                )

            for node in range(self.nnodes_element_la):
                f_pot_el[self.nodalDOF_element_la_u[node]] -= (
                    self.N_la_xi[el, i, node] * K_m * qwi
                )

                f_pot_el[self.nodalDOF_element_la_u[node]] += (
                    self.N_la[el, i, node]
                    * (cross3(K_Gamma_bar, K_n) + cross3(K_Kappa_bar, K_m))
                    * qwi
                )

        return f_pot_el

    def f_pot_el_q(self, qe, el):
        return approx_fprime(
            qe,
            lambda qe: self.f_pot_el(qe, el),
            eps=1e-6,
            method="2-point",
        )

    #########################################
    # equations of motion
    #########################################
    # def assembler_callback(self):
    #     if self.constant_mass_matrix:
    #         self._M_coo()

    def M_el_constant(self, el):
        raise NotImplementedError
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
            for node_a in range(self.nnodes_element_la):
                nodalDOF_a = self.nodalDOF_element_la_u[node_a]
                for node_b in range(self.nnodes_element_la):
                    nodalDOF_b = self.nodalDOF_element_la_u[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
                        self.N_psi_dyn[el, i, node_a] * self.N_psi_dyn[el, i, node_b]
                    )

        return M_el

    def M_el(self, qe, el):
        raise NotImplementedError
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
            for node_a in range(self.nnodes_element_la):
                nodalDOF_a = self.nodalDOF_element_la_u[node_a]
                for node_b in range(self.nnodes_element_la):
                    nodalDOF_b = self.nodalDOF_element_la_u[node_b]
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
                for node_b in range(self.nnodes_element_la):
                    nodalDOF_b = self.nodalDOF_element_la_u[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_psi * (
                        self.N_r_dyn[el, i, node_a] * self.N_psi_dyn[el, i, node_b]
                    )
            for node_a in range(self.nnodes_element_la):
                nodalDOF_a = self.nodalDOF_element_la_u[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_r * (
                        self.N_psi_dyn[el, i, node_a] * self.N_r_dyn[el, i, node_b]
                    )

        return M_el

    def _M_coo(self):
        raise NotImplementedError
        self.__M = Coo((self.nu, self.nu))
        for el in range(self.nelement):
            # extract element degrees of freedom
            elDOF_u = self.elDOF_u[el]

            # sparse assemble element mass matrix
            self.__M.extend(
                self.M_el_constant(el), (self.uDOF[elDOF_u], self.uDOF[elDOF_u])
            )

    def M(self, t, q, coo):
        raise NotImplementedError
        if self.constant_mass_matrix:
            coo.extend_sparse(self.__M)
        else:
            for el in range(self.nelement):
                # extract element degrees of freedom
                elDOF = self.elDOF[el]
                elDOF_u = self.elDOF_u[el]

                # sparse assemble element mass matrix
                coo.extend(
                    self.M_el(q[elDOF], el),
                    (self.uDOF[elDOF_u], self.uDOF[elDOF_u]),
                )

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
            for node in range(self.nnodes_element_la):
                K_omega_IK += (
                    self.N_psi_dyn[el, i, node] * ue[self.nodalDOF_element_la_u[node]]
                )

            # delta_r A_rho0 r_ddot part
            E_kin_el += 0.5 * (v_P @ v_P) * self.A_rho0 * Ji * qwi

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            E_kin_el += 0.5 * (K_omega_IK @ self.K_I_rho0 @ K_omega_IK) * Ji * qwi

        # E_kin_el = ue @ self.M_el_constant(el) @ ue

        return E_kin_el

    def f_gyr_el(self, t, qe, ue, el):
        raise NotImplementedError
        if not self.constant_mass_matrix:
            raise NotImplementedError(
                "Gyroscopic forces are not implemented for nonzero K_S_rho0."
            )

        f_gyr_el = np.zeros(self.nu_element, dtype=np.common_type(qe, ue))

        for i in range(self.nquadrature_dyn):
            # interpoalte angular velocity
            K_Omega = np.zeros(3, dtype=np.common_type(qe, ue))
            for node in range(self.nnodes_element_la):
                K_Omega += (
                    self.N_psi_dyn[el, i, node] * ue[self.nodalDOF_element_la_u[node]]
                )

            # vector of gyroscopic forces
            f_gyr_el_psi = (
                cross3(K_Omega, self.K_I_rho0 @ K_Omega)
                * self.J_dyn[el, i]
                * self.qw_dyn[el, i]
            )

            # multiply vector of gyroscopic forces with nodal virtual rotations
            for node in range(self.nnodes_element_la):
                f_gyr_el[self.nodalDOF_element_la_u[node]] += (
                    self.N_psi_dyn[el, i, node] * f_gyr_el_psi
                )

        return f_gyr_el

    def f_gyr_u_el(self, t, qe, ue, el):
        raise NotImplementedError
        f_gyr_u_el = np.zeros((self.nu_element, self.nu_element), dtype=float)

        for i in range(self.nquadrature_dyn):
            # interpoalte angular velocity
            K_Omega = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_la):
                K_Omega += (
                    self.N_psi_dyn[el, i, node] * ue[self.nodalDOF_element_la_u[node]]
                )

            # derivative of vector of gyroscopic forces
            f_gyr_u_el_psi = (
                ((ax2skew(K_Omega) @ self.K_I_rho0 - ax2skew(self.K_I_rho0 @ K_Omega)))
                * self.J_dyn[el, i]
                * self.qw_dyn[el, i]
            )

            # multiply derivative of gyroscopic force vector with nodal virtual rotations
            for node_a in range(self.nnodes_element_la):
                nodalDOF_a = self.nodalDOF_element_la_u[node_a]
                for node_b in range(self.nnodes_element_la):
                    nodalDOF_b = self.nodalDOF_element_la_u[node_b]
                    f_gyr_u_el[nodalDOF_a[:, None], nodalDOF_b] += f_gyr_u_el_psi * (
                        self.N_psi_dyn[el, i, node_a] * self.N_psi_dyn[el, i, node_b]
                    )

        return f_gyr_u_el

    def h(self, t, q, u):
        h = np.zeros(self.nu, dtype=np.common_type(q, u))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            h[elDOF_u] += self.f_pot_el(q[elDOF], el)
            # - self.f_gyr_el(
            #     t, q[elDOF], u[elDOF_u], el
            # )
        return h

    def h_q(self, t, q, u, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            h_q_el = self.f_pot_el_q(q[elDOF], el)
            coo.extend(h_q_el, (self.uDOF[elDOF_u], self.qDOF[elDOF]))

    # def h_u(self, t, q, u, coo):
    #     for el in range(self.nelement):
    #         elDOF = self.elDOF[el]
    #         elDOF_u = self.elDOF_u[el]
    #         h_u_el = -self.f_gyr_u_el(t, q[elDOF], u[elDOF_u], el)
    #         coo.extend(h_u_el, (self.uDOF[elDOF_u], self.uDOF[elDOF_u]))

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
        r_OP, A_IK, _, _ = self._eval(q, frame_ID[0])
        return r_OP + A_IK @ K_r_SP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        return approx_fprime(
            q,
            lambda q: self.r_OP(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP),
            method="2-point",
            eps=1e-6,
        )

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        xi = frame_ID[0]
        assert xi == 0 or xi == 1.0

        v_P = np.zeros(3, dtype=q.dtype)
        if xi == 0.0:
            nodalDOFr0 = self.nodalDOF_element_r[0]
            nodalDOFpsi0 = self.nodalDOF_element_r[1]
            v_r0 = u[nodalDOFr0]
            psi_dot0 = u[nodalDOFpsi0]
            psi0 = q[nodalDOFpsi0]
            A_IK0 = Exp_SO3(psi0)
            v_P = v_r0 - A_IK0 @ cross3(K_r_SP, T_SO3(psi0) @ psi_dot0)
        elif xi == 1.0:
            nodalDOFr1 = self.nodalDOF_element_r[2]
            nodalDOFpsi1 = self.nodalDOF_element_r[3]
            v_r1 = u[nodalDOFr1]
            psi_dot1 = u[nodalDOFpsi1]
            psi1 = q[nodalDOFpsi1]
            A_IK1 = Exp_SO3(psi1)
            v_P = v_r1 - A_IK1 @ cross3(K_r_SP, T_SO3(psi1) @ psi_dot1)

        return v_P

    def v_P_q(self, t, q, u, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        raise NotImplementedError

    def J_P(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        xi = frame_ID[0]
        assert xi == 0 or xi == 1.0

        J_P = np.zeros((3, self.nu_element), dtype=q.dtype)
        if xi == 0.0:
            nodalDOFr0 = self.nodalDOF_element_r[0]
            nodalDOFpsi0 = self.nodalDOF_element_r[1]
            psi0 = q[nodalDOFpsi0]
            A_IK0 = Exp_SO3(psi0)

            J_P[:, nodalDOFr0] = np.eye(3)
            J_P[:, nodalDOFpsi0] = -A_IK0 @ (ax2skew(K_r_SP) @ T_SO3(psi0))
        elif xi == 1.0:
            nodalDOFr1 = self.nodalDOF_element_r[2]
            nodalDOFpsi1 = self.nodalDOF_element_r[3]
            psi1 = q[nodalDOFpsi1]
            A_IK1 = Exp_SO3(psi1)

            J_P[:, nodalDOFr1] = np.eye(3)
            J_P[:, nodalDOFpsi1] = -A_IK1 @ (ax2skew(K_r_SP) @ T_SO3(psi1))

        return J_P

        # J_P_num = approx_fprime(
        #     np.zeros(self.nu_element, dtype=float),
        #     lambda u: self.v_P(t, q, u, frame_ID, K_r_SP),
        #     method="2-point",
        #     eps=1e-3,
        # )
        # diff = J_P_num - J_P
        # error = np.linalg.norm(diff)
        # print(f"error J_P: {error}")
        # return J_P_num

    def J_P_q(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        xi = frame_ID[0]
        assert xi == 0 or xi == 1.0

        K_r_SP_skew = ax2skew(K_r_SP)
        J_P_q = np.zeros((3, self.nu_element, self.nq_element), dtype=q.dtype)
        if xi == 0.0:
            nodalDOFpsi0 = self.nodalDOF_element_r[1]
            psi0 = q[nodalDOFpsi0]
            J_P_q[:, nodalDOFpsi0[:, None], nodalDOFpsi0] = np.einsum(
                "ijl,jk->ikl", Exp_SO3_psi(psi0), K_r_SP_skew @ T_SO3(psi0)
            ) + np.einsum("ij,jkl->ikl", Exp_SO3(psi0), K_r_SP_skew @ T_SO3_psi(psi0))
        elif xi == 1.0:
            nodalDOFpsi1 = self.nodalDOF_element_r[3]
            psi1 = q[nodalDOFpsi1]
            J_P_q[:, nodalDOFpsi1[:, None], nodalDOFpsi1] = np.einsum(
                "ijl,jk->ikl", Exp_SO3_psi(psi1), K_r_SP_skew @ T_SO3(psi1)
            ) + np.einsum("ij,jkl->ikl", Exp_SO3(psi1), K_r_SP_skew @ T_SO3_psi(psi1))

        return J_P_q

        # J_P_q_num = approx_fprime(
        #     q,
        #     lambda q: self.J_P(t, q, frame_ID, K_r_SP),
        #     method="2-point",
        #     eps=1e-3,
        # )
        # diff = J_P_q_num - J_P_q
        # error = np.linalg.norm(diff)
        # print(f"error J_P_q: {error}")
        # return J_P_q_num

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        raise NotImplementedError
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

    # TODO: Analytical derivative.
    def a_P_q(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        raise NotImplementedError
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID)
        a_P_q = np.einsum(
            "ijk,j->ik",
            self.A_IK_q(t, q, frame_ID),
            cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP)),
        )
        # return a_P_q

        a_P_q_num = approx_fprime(
            q,
            lambda q: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP),
            method="3-point",
        )
        diff = a_P_q_num - a_P_q
        error = np.linalg.norm(diff)
        print(f"error a_P_q: {error}")
        return a_P_q_num

    # TODO: Analytical derivative.
    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        raise NotImplementedError
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        local = -self.A_IK(t, q, frame_ID) @ (
            ax2skew(cross3(K_Omega, K_r_SP)) + ax2skew(K_Omega) @ ax2skew(K_r_SP)
        )

        N, _ = self.basis_functions_r(frame_ID[0])
        a_P_u = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_r):
            a_P_u[:, self.nodalDOF_element_la[node]] += N[node] * local

        # return a_P_u

        a_P_u_num = approx_fprime(
            u,
            lambda u: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP),
            method="3-point",
        )
        diff = a_P_u_num - a_P_u
        error = np.linalg.norm(diff)
        print(f"error a_P_u: {error}")
        return a_P_u_num

    def K_Omega(self, t, q, u, frame_ID):
        xi = frame_ID[0]
        assert xi == 0 or xi == 1.0

        K_omega_IK = np.zeros((3, self.nu_element), dtype=q.dtype)
        if xi == 0.0:
            nodalDOFpsi0 = self.nodalDOF_element_r[1]
            psi_dot0 = u[nodalDOFpsi0]
            psi0 = q[nodalDOFpsi0]
            K_omega_IK = T_SO3(psi0) @ psi_dot0
        elif xi == 1.0:
            nodalDOFpsi1 = self.nodalDOF_element_r[3]
            psi_dot1 = u[nodalDOFpsi1]
            psi1 = q[nodalDOFpsi1]
            K_omega_IK = T_SO3(psi1) @ psi_dot1

        return K_omega_IK

    def K_Omega_q(self, t, q, u, frame_ID):
        raise NotImplementedError
        return np.zeros((3, self.nu_element), dtype=float)

    def K_J_R(self, t, q, frame_ID):
        xi = frame_ID[0]
        assert xi == 0 or xi == 1.0

        J_R = np.zeros((3, self.nu_element), dtype=q.dtype)
        if xi == 0.0:
            nodalDOFpsi0 = self.nodalDOF_element_r[1]
            psi0 = q[nodalDOFpsi0]
            J_R[:, nodalDOFpsi0] = T_SO3(psi0)
        elif xi == 1.0:
            nodalDOFpsi1 = self.nodalDOF_element_r[3]
            psi1 = q[nodalDOFpsi1]
            J_R[:, nodalDOFpsi1] = T_SO3(psi1)

        return J_R

        # J_R_num = approx_fprime(
        #     np.zeros(self.nu_element, dtype=float),
        #     lambda u: self.K_Omega(t, q, u, frame_ID),
        #     method="2-point",
        #     eps=1e-6,
        # )
        # diff = J_R_num - J_R
        # error = np.linalg.norm(diff)
        # print(f"error J_R: {error}")
        # return J_R_num

    def K_J_R_q(self, t, q, frame_ID):
        # return approx_fprime(
        #     q,
        #     lambda q: self.K_J_R(t, q, frame_ID),
        #     method="2-point",
        #     eps=1e-6,
        # )

        xi = frame_ID[0]
        assert xi == 0 or xi == 1.0

        J_R_q = np.zeros((3, self.nu_element, self.nq_element), dtype=q.dtype)
        if xi == 0.0:
            nodalDOFpsi0 = self.nodalDOF_element_r[1]
            psi0 = q[nodalDOFpsi0]
            J_R_q[:, nodalDOFpsi0[:, None], nodalDOFpsi0] = T_SO3_psi(psi0)
        elif xi == 1.0:
            nodalDOFpsi1 = self.nodalDOF_element_r[3]
            psi1 = q[nodalDOFpsi1]
            J_R_q[:, nodalDOFpsi1[:, None], nodalDOFpsi1] = T_SO3_psi(psi1)

        return J_R_q

        # J_R_q_num = approx_fprime(
        #     q,
        #     lambda q: self.K_J_R(t, q, frame_ID),
        #     method="2-point",
        #     eps=1e-6,
        # )
        # diff = J_R_q_num - J_R_q
        # error = np.linalg.norm(diff)
        # print(f"error J_R_q: {error}")
        # return J_R_q_num

    def K_Psi(self, t, q, u, u_dot, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        time derivative of the angular velocities in the K-frame.
        """
        raise NotImplementedError
        N_psi, _ = self.basis_functions_la(frame_ID[0])
        K_Psi = np.zeros(3, dtype=np.common_type(q, u, u_dot))
        for node in range(self.nnodes_element_la):
            K_Psi += N_psi[node] * u_dot[self.nodalDOF_element_la_u[node]]
        return K_Psi

    def K_Psi_q(self, t, q, u, u_dot, frame_ID):
        raise NotImplementedError
        return np.zeros((3, self.nq_element), dtype=float)

    def K_Psi_u(self, t, q, u, u_dot, frame_ID):
        raise NotImplementedError
        return np.zeros((3, self.nu_element), dtype=float)

    ####################################################
    # body force
    ####################################################
    def distributed_force1D_pot_el(self, force, t, qe, el):
        raise NotImplementedError
        Ve = 0.0

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # interpolate centerline position
            r_C = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_r):
                r_C += self.N_r[el, i, node] * qe[self.nodalDOF_element_r[node]]

            # compute potential value at given quadrature point
            Ve -= (r_C @ force(t, qpi)) * Ji * qwi

        # for i in range(self.nquadrature_dyn):
        #     # extract reference state variables
        #     qpi = self.qp_dyn[el, i]
        #     qwi = self.qw_dyn[el, i]
        #     Ji = self.J_dyn[el, i]

        #     # interpolate centerline position
        #     r_C = np.zeros(3, dtype=float)
        #     for node in range(self.nnodes_element_r):
        #         r_C += self.N_r_dyn[el, i, node] * qe[self.nodalDOF_element_r[node]]

        #     # compute potential value at given quadrature point
        #     Ve -= (r_C @ force(t, qpi)) * Ji * qwi

        return Ve

    def distributed_force1D_pot(self, t, q, force):
        raise NotImplementedError
        V = 0
        for el in range(self.nelement):
            qe = q[self.elDOF[el]]
            V += self.distributed_force1D_pot_el(force, t, qe, el)
        return V

    # TODO: Decide which number of quadrature points shoul dbe used here?
    def distributed_force1D_el(self, force, t, el):
        raise NotImplementedError
        fe = np.zeros(self.nu_element, dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # compute local force vector
            fe_r = force(t, qpi) * Ji * qwi

            # multiply local force vector with variation of centerline
            for node in range(self.nnodes_element_r):
                fe[self.nodalDOF_element_r[node]] += self.N_r[el, i, node] * fe_r

        # for i in range(self.nquadrature_dyn):
        #     # extract reference state variables
        #     qpi = self.qp_dyn[el, i]
        #     qwi = self.qw_dyn[el, i]
        #     Ji = self.J_dyn[el, i]

        #     # compute local force vector
        #     fe_r = force(t, qpi) * Ji * qwi

        #     # multiply local force vector with variation of centerline
        #     for node in range(self.nnodes_element_r):
        #         fe[self.nodalDOF_element_r[node]] += self.N_r_dyn[el, i, node] * fe_r

        return fe

    def distributed_force1D(self, t, q, force):
        raise NotImplementedError
        f = np.zeros(self.nu, dtype=float)
        for el in range(self.nelement):
            f[self.elDOF_u[el]] += self.distributed_force1D_el(force, t, el)
        return f

    def distributed_force1D_q(self, t, q, coo, force):
        raise NotImplementedError
        pass
