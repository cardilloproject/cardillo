import numpy as np
import meshio
import os

from cardillo.utility.coo import Coo
from cardillo.discretization.B_spline import KnotVector
from cardillo.discretization.lagrange import Node_vector
from cardillo.discretization.mesh1D import Mesh1D
from cardillo.math import norm, cross3, skew2ax, skew2ax_A, ax2skew, approx_fprime
from cardillo.math import (
    rodriguez,
    rodriguez_der,
    rodriguez_inv,
    tangent_map,
    inverse_tangent_map,
    tangent_map_s,
)


class DirectorAxisAngle:
    def __init__(
        self,
        material_model,
        A_rho0,
        I_rho0,
        polynomial_degree_r,
        polynomial_degree_psi,
        nquadrature,
        nelement,
        Q,
        q0=None,
        u0=None,
        basis="B-spline",
    ):

        # beam properties
        self.materialModel = material_model  # material model
        self.A_rho0 = A_rho0  # line density
        self.I_rho0 = I_rho0  # second moment

        # material model
        self.material_model = material_model

        # discretization parameters
        self.polynomial_degree_r = polynomial_degree_r  # polynomial degree centerline
        self.polynomial_degree_psi = (
            polynomial_degree_psi  # polynomial degree axis angle
        )
        self.nquadrature = nquadrature  # number of quadrature points
        self.nelement = nelement  # number of elements

        self.basis = basis
        if basis == "B-spline":
            self.knot_vector_r = KnotVector(polynomial_degree_r, nelement)
            self.knot_vector_psi = KnotVector(polynomial_degree_psi, nelement)
            self.nnode_r = nnode_r = nelement + polynomial_degree_r  # number of nodes
            self.nnode_psi = nnode_psi = (
                nelement + polynomial_degree_psi
            )  # number of nodes
        elif basis == "lagrange":
            self.knot_vector_r = Node_vector(polynomial_degree_r, nelement)
            self.knot_vector_psi = Node_vector(polynomial_degree_psi, nelement)
            self.nnode_r = nnode_r = (
                nelement * polynomial_degree_r + 1
            )  # number of nodes
            self.nnode_psi = nnode_psi = (
                nelement * polynomial_degree_psi + 1
            )  # number of nodes
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        # number of nodes per element
        self.nnodes_element_r = nnodes_element_r = polynomial_degree_r + 1
        self.nnodes_element_psi = nnodes_element_psi = polynomial_degree_psi + 1

        # number of degrees of freedom per node
        self.nq_node_r = nq_node_r = 3
        self.nq_node_psi = nq_node_psi = 3

        self.mesh_r = Mesh1D(
            self.knot_vector_r,
            nquadrature,
            derivative_order=1,
            basis=basis,
            dim=nq_node_r,
        )
        self.mesh_psi = Mesh1D(
            self.knot_vector_psi,
            nquadrature,
            derivative_order=1,
            basis=basis,
            dim=nq_node_psi,
        )

        self.nq_r = nq_r = nnode_r * nq_node_r
        self.nq_psi = nq_psi = nnode_psi * nq_node_psi
        self.nq = nq_r + nq_psi  # total number of generalized coordinates
        self.nu = self.nq
        self.nq_element = (
            nnodes_element_r * nq_node_r + nnodes_element_psi * nq_node_psi
        )  # total number of generalized coordinates per element
        self.nq_el_r = nnodes_element_r * nq_node_r
        self.nq_el_psi = nnodes_element_psi * nq_node_psi

        # global element connectivity
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_psi = self.mesh_psi.elDOF + nq_r

        # global nodal
        self.nodalDOF_r = self.mesh_r.nodalDOF
        self.nodalDOF_psi = self.mesh_psi.nodalDOF + nq_r

        # nodal connectivity on element level
        self.nodalDOF_element_r = self.mesh_r.nodalDOF_element_
        self.nodalDOF_element_psi = self.mesh_psi.nodalDOF_element_ + self.nq_el_r

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nelement, self.nq_element), dtype=int)
        for el in range(nelement):
            self.elDOF[el, : self.nq_el_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_el_r :] = self.elDOF_psi[el]

        # degrees of freedom on element level
        self.rDOF = np.arange(0, nq_node_r * nnodes_element_r)
        self.psiDOF = (
            np.arange(0, nq_node_psi * nnodes_element_psi)
            + nq_node_r * nnodes_element_r
        )

        # shape functions and their first derivatives
        self.N_r = self.mesh_r.N
        self.N_r_xi = self.mesh_r.N_xi
        self.N_psi = self.mesh_psi.N
        self.N_psi_xi = self.mesh_psi.N_xi

        # quadrature points
        self.qp = self.mesh_r.qp  # quadrature points
        self.qw = self.mesh_r.wp  # quadrature weights

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions_r = self.mesh_r.eval_basis
        self.basis_functions_psi = self.mesh_psi.eval_basis

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q  # reference configuration
        self.q0 = Q.copy() if q0 is None else q0  # initial configuration
        self.u0 = np.zeros(self.nu) if u0 is None else u0  # initial velocities

        # precompute values of the reference configuration in order to save computation time
        # J in Harsch2020b (5)
        self.J = np.zeros((nelement, nquadrature))
        # dilatation and shear strains of the reference configuration
        self.K_Gamma0 = np.zeros((nelement, nquadrature, 3))
        # curvature of the reference configuration
        self.K_Kappa0 = np.zeros((nelement, nquadrature, 3))

        # fix evaluation points of rotation vectors for each element since we
        # want to evaluate their derivatives but do not have them on fixed
        # conrtol nodes
        knotvector_psi_data = self.knot_vector_psi.element_data
        self.xi_el_psi = lambda el: np.linspace(
            knotvector_psi_data[el],
            knotvector_psi_data[el + 1],
            num=self.nnodes_element_psi,
        )

        for el in range(nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(nquadrature):
                # interpolate tangent vector
                r_xi = np.zeros(3)
                for node in range(self.nnodes_element_r):
                    r_xi += self.N_r_xi[el, i, node] * qe[self.nodalDOF_element_r[node]]

                # length of reference tangential vector
                Ji = norm(r_xi)

                # interpolate rotation and its derivative w.r.t. parameter space
                A_IK = np.zeros((3, 3))
                A_IK_xi = np.zeros((3, 3))
                for node in range(self.nnodes_element_psi):
                    psi_node = qe[self.nodalDOF_element_psi[node]]
                    A_IK_node = rodriguez(psi_node)
                    A_IK += self.N_psi[el, i, node] * A_IK_node
                    A_IK_xi += self.N_psi_xi[el, i, node] * A_IK_node

                # extract directors and their derivatives with respect to
                # parameter space
                d1, d2, d3 = A_IK.T
                d1_xi, d2_xi, d3_xi = A_IK_xi.T

                # compute derivatives with respect to arc length parameter
                r_s = r_xi / Ji
                d1_s = d1_xi / Ji
                d2_s = d2_xi / Ji
                d3_s = d3_xi / Ji

                # axial and shear strains
                K_Gamma = np.array([r_s @ d1, r_s @ d2, r_s @ d3])

                # torsional and flexural strains (formulation in skew coordinates,
                # see Eugster2014c)
                d = d1 @ cross3(d2, d3)
                dd = 0.5 / d
                K_Kappa = np.array(
                    [
                        dd * (d3 @ d2_s - d2 @ d3_s),
                        dd * (d1 @ d3_s - d3 @ d1_s),
                        dd * (d2 @ d1_s - d1 @ d2_s),
                    ]
                )

                # safe precomputed quantities for later
                self.J[el, i] = Ji
                self.K_Gamma0[el, i] = K_Gamma
                self.K_Kappa0[el, i] = K_Kappa

    @staticmethod
    def straight_configuration(
        polynomial_degree_r,
        polynomial_degree_psi,
        nEl,
        L,
        greville_abscissae=True,
        r_OP=np.zeros(3),
        A_IK=np.eye(3),
        basis="B-spline",
    ):
        if basis == "B-spline":
            nn_r = polynomial_degree_r + nEl
            nn_psi = polynomial_degree_psi + nEl
        elif basis == "Lagrange":
            nn_r = polynomial_degree_r * nEl + 1
            nn_psi = polynomial_degree_psi * nEl + 1
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        x0 = np.linspace(0, L, num=nn_r)
        y0 = np.zeros(nn_r)
        z0 = np.zeros(nn_r)
        if greville_abscissae and basis == "B-spline":
            kv = KnotVector.uniform(polynomial_degree_r, nEl)
            for i in range(nn_r):
                x0[i] = np.sum(kv[i + 1 : i + polynomial_degree_r + 1])
            x0 = x0 * L / polynomial_degree_r

        r0 = np.vstack((x0, y0, z0))
        for i in range(nn_r):
            r0[:, i] = r_OP + A_IK @ r0[:, i]

        # reshape generalized coordinates to nodal ordering
        q_r = r0.reshape(-1, order="F")

        # we have to extract the rotation vector from the given rotation matrix
        # and set its value for each node
        psi = rodriguez_inv(A_IK)
        q_psi = np.tile(psi, nn_psi)

        return np.concatenate([q_r, q_psi])

    def element_number(self, xi):
        # note the elements coincide for both meshes!
        return self.knot_vector_r.element_number(xi)[0]

    def assembler_callback(self):
        self.__M_coo()

    #########################################
    # equations of motion
    #########################################
    def M_el(self, el):
        M_el = np.zeros((self.nq_element, self.nq_element))

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # delta_r A_rho0 r_ddot part
            # TODO: Can this be simplified with a single nodal loop?
            M_el_r = np.eye(3) * self.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r * (
                        self.N_r[el, i, node_a] * self.N_r[el, i, node_b]
                    )
            # for node in range(self.nnodes_element_r):
            #     nodalDOF = self.nodalDOF_element_r[node]
            #     N_node = self.N_r[el, i, node]
            #     M_el[nodalDOF[:, None], nodalDOF] += M_el_r * (N_node * N_node)

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_psi = self.I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi * (
                        self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
                    )

        return M_el

    def __M_coo(self):
        self.__M = Coo((self.nu, self.nu))
        for el in range(self.nelement):
            # extract element degrees of freedom
            elDOF = self.elDOF[el]

            # sparse assemble element mass matrix
            self.__M.extend(self.M_el(el), (self.uDOF[elDOF], self.uDOF[elDOF]))

    def M(self, t, q, coo):
        coo.extend_sparse(self.__M)

    def f_gyr_el(self, t, qe, ue, el):
        f_gyr_el = np.zeros(self.nq_element)

        for i in range(self.nquadrature):
            # interpoalte angular velocity
            K_Omega = np.zeros(3)
            for node in range(self.nnodes_element_psi):
                K_Omega += self.N_psi[el, i, node] * ue[self.nodalDOF_element_psi[node]]

            # vector of gyroscopic forces
            f_gyr_el_psi = (
                cross3(K_Omega, self.I_rho0 @ K_Omega) * self.J[el, i] * self.qw[el, i]
            )

            # multiply vector of gyroscopic forces with nodal virtual rotations
            for node in range(self.nnodes_element_psi):
                f_gyr_el[self.nodalDOF_element_psi[node]] += (
                    self.N_psi[el, i, node] * f_gyr_el_psi
                )

        return f_gyr_el

    def f_gyr(self, t, q, u):
        f_gyr = np.zeros(self.nu)
        for el in range(self.nelement):
            f_gyr[self.elDOF[el]] += self.f_gyr_el(
                t, q[self.elDOF[el]], u[self.elDOF[el]], el
            )
        return f_gyr

    def f_gyr_u_el(self, t, qe, ue, el):
        f_gyr_u_el = np.zeros((self.nq_element, self.nq_element))

        for i in range(self.nquadrature):
            # interpoalte angular velocity
            K_Omega = np.zeros(3)
            for node in range(self.nnodes_element_psi):
                K_Omega += self.N_psi[el, i, node] * ue[self.nodalDOF_element_psi[node]]

            # derivative of vector of gyroscopic forces
            f_gyr_u_el_psi = (
                ((ax2skew(K_Omega) @ self.I_rho0 - ax2skew(self.I_rho0 @ K_Omega)))
                * self.J[el, i]
                * self.qw[el, i]
            )

            # multiply derivative of gyroscopic force vector with nodal virtual rotations
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    f_gyr_u_el[nodalDOF_a[:, None], nodalDOF_b] += f_gyr_u_el_psi * (
                        self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
                    )

        return f_gyr_u_el

    def f_gyr_u(self, t, q, u, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f_gyr_u_el = self.f_gyr_u_el(t, q[elDOF], u[elDOF], el)
            coo.extend(f_gyr_u_el, (self.uDOF[elDOF], self.uDOF[elDOF]))

    def E_pot(self, t, q):
        E_pot = 0
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            E_pot += self.E_pot_el(q[elDOF], el)
        return E_pot

    def E_pot_el(self, qe, el):
        E_pot_el = 0

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # interpolate tangent vector
            r_xi = np.zeros(3)
            for node in range(self.nnodes_element_r):
                r_xi += self.N_r_xi[el, i, node] * qe[self.nodalDOF_element_r[node]]

            # interpolate rotation and its derivative w.r.t. parameter space
            A_IK = np.zeros((3, 3))
            A_IK_xi = np.zeros((3, 3))
            for node in range(self.nnodes_element_psi):
                psi_node = qe[self.nodalDOF_element_psi[node]]
                A_IK_node = rodriguez(psi_node)
                A_IK += self.N_psi[el, i, node] * A_IK_node
                A_IK_xi += self.N_psi_xi[el, i, node] * A_IK_node

            # extract directors and their derivatives with respect to
            # parameter space
            d1, d2, d3 = A_IK.T
            d1_xi, d2_xi, d3_xi = A_IK_xi.T

            # compute derivatives with respect to arc length parameter
            r_s = r_xi / Ji
            d1_s = d1_xi / Ji
            d2_s = d2_xi / Ji
            d3_s = d3_xi / Ji

            # axial and shear strains
            K_Gamma = np.array([r_s @ d1, r_s @ d2, r_s @ d3])

            # torsional and flexural strains (formulation in skew coordinates,
            # see Eugster2014c)
            d = d1 @ cross3(d2, d3)
            dd = 0.5 / d
            K_Kappa = np.array(
                [
                    dd * (d3 @ d2_s - d2 @ d3_s),
                    dd * (d1 @ d3_s - d3 @ d1_s),
                    dd * (d2 @ d1_s - d1 @ d2_s),
                ]
            )

            # evaluate strain energy function
            E_pot_el += (
                self.material_model.potential(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
                * Ji
                * qwi
            )

        return E_pot_el

    def f_pot(self, t, q):
        f_pot = np.zeros(self.nu)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f_pot[elDOF] += self.f_pot_el(q[elDOF], el)
        return f_pot

    def f_pot_el(self, qe, el):
        f_pot_el = np.zeros(self.nq_element)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # interpolate tangent vector
            r_xi = np.zeros(3)
            for node in range(self.nnodes_element_r):
                r_xi += self.N_r_xi[el, i, node] * qe[self.nodalDOF_element_r[node]]

            # interpolate rotation and its derivative w.r.t. parameter space
            A_IK = np.zeros((3, 3))
            A_IK_xi = np.zeros((3, 3))
            for node in range(self.nnodes_element_psi):
                psi_node = qe[self.nodalDOF_element_psi[node]]
                A_IK_node = rodriguez(psi_node)
                A_IK += self.N_psi[el, i, node] * A_IK_node
                A_IK_xi += self.N_psi_xi[el, i, node] * A_IK_node

            # extract directors and their derivatives with respect to
            # parameter space
            d1, d2, d3 = A_IK.T
            d1_xi, d2_xi, d3_xi = A_IK_xi.T

            # compute derivatives with respect to arc length parameter
            r_s = r_xi / Ji
            d1_s = d1_xi / Ji
            d2_s = d2_xi / Ji
            d3_s = d3_xi / Ji

            # axial and shear strains
            K_Gamma = np.array([r_s @ d1, r_s @ d2, r_s @ d3])

            # torsional and flexural strains (formulation in skew coordinates,
            # see Eugster2014c)
            d = d1 @ cross3(d2, d3)
            dd = 0.5 / d
            K_Kappa = np.array(
                [
                    dd * (d3 @ d2_s - d2 @ d3_s),
                    dd * (d1 @ d3_s - d3 @ d1_s),
                    dd * (d2 @ d1_s - d1 @ d2_s),
                ]
            )

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)

            # - first delta Gamma part
            for node in range(self.nnodes_element_r):
                f_pot_el[self.nodalDOF_element_r[node]] -= (
                    self.N_r_xi[el, i, node] * A_IK @ K_n * qwi
                )

            # - second delta Gamma part
            for node in range(self.nnodes_element_psi):
                f_pot_el[self.nodalDOF_element_psi[node]] += (
                    self.N_psi[el, i, node] * cross3(A_IK.T @ r_xi, K_n) * qwi
                )

            # - delta kappa part
            for node in range(self.nnodes_element_psi):
                f_pot_el[self.nodalDOF_element_psi[node]] -= (
                    self.N_psi_xi[el, i, node] * K_m * qwi
                )
                f_pot_el[self.nodalDOF_element_psi[node]] += (
                    self.N_psi[el, i, node] * cross3(K_Kappa, K_m) * Ji * d * qwi
                )  # Euler term

        return f_pot_el

    def f_pot_q(self, t, q, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f_pot_q_el = self.f_pot_q_el(q[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(f_pot_q_el, (self.uDOF[elDOF], self.qDOF[elDOF]))

        # from scipy.sparse import csc_matrix
        # sparse = coo.tosparse(csc_matrix)
        # dense = sparse.toarray()
        # print(f"f_pot_q.shape: {dense.shape}")
        # print(f"f_pot_q.rank: {np.linalg.matrix_rank(dense)}")
        # print(f"")

    def f_pot_q_el(self, qe, el):
        return approx_fprime(qe, lambda qe: self.f_pot_el(qe, el), method="2-point")

        f_pot_q_el = np.zeros((self.nq_element, self.nq_element))

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # interpolate tangent vector
            r_xi = np.zeros(3)
            for node in range(self.nnodes_element_r):
                r_xi += self.N_r_xi[el, i, node] * qe[self.nodalDOF_element_r[node]]

            # interpolate rotation and its derivative w.r.t. parameter space
            A_IK = np.zeros((3, 3))
            A_IK_xi = np.zeros((3, 3))
            for node in range(self.nnodes_element_psi):
                psi_node = qe[self.nodalDOF_element_psi[node]]
                A_IK_node = rodriguez(psi_node)
                A_IK += self.N_psi[el, i, node] * A_IK_node
                A_IK_xi += self.N_psi_xi[el, i, node] * A_IK_node

            # extract directors and their derivatives with respect to
            # parameter space
            d1, d2, d3 = A_IK.T
            d1_xi, d2_xi, d3_xi = A_IK_xi.T

            # compute derivatives with respect to arc length parameter
            r_s = r_xi / Ji

            # axial and shear strains
            K_Gamma = A_IK.T @ r_s

            # torsional and flexural strains + derivatives
            # (formulation in skew coordinates, # see Eugster2014c)
            K_Kappa_bar = np.array(
                [
                    0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                    0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                    0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                ]
            )

            d = d1 @ cross3(d2, d3)
            dd = 1 / (d * Ji)
            K_Kappa = dd * K_Kappa_bar

            ################################
            # derivatives of strain measures
            ################################
            # first K_Gamma derivative w.r.t. qe_r
            K_Gamma_qe = np.zeros((3, self.nq_element))
            for node in range(self.nnodes_element_r):
                K_Gamma_qe[:, self.nodalDOF_element_r[node]] += (
                    A_IK.T * self.N_r_xi[el, i, node] / Ji
                )

            # Derivative of d = d1 @ cross3(d2, d3) w.r.t. qe_psi,
            # first part of K_Kappa derivative w.r.t. qe_psi and
            # second part of K_Gamma derivative w.r.t. qe_psi.
            d_qe = np.zeros(self.nq_element)
            K_Kappa_bar_qe = np.zeros((3, self.nq_element))

            tmp1 = d1 @ ax2skew(d2)
            tmp2 = -d1 @ ax2skew(d3)
            tmp3 = cross3(d2, d3)
            for node in range(self.nnodes_element_psi):
                nodalDOF = self.nodalDOF_element_psi[node]

                # nodal derivative of rodriguez formular
                A_IK_psi_node = rodriguez_der(qe[nodalDOF])

                # nodal shape functions
                N_psi = self.N_psi[el, i, node]
                N_psi_xi = self.N_psi_xi[el, i, node]

                # K_Gamma part
                K_Gamma_qe[:, nodalDOF] += N_psi * np.einsum(
                    "jik,j->ik", A_IK_psi_node, r_s
                )

                # d parts
                d_qe[nodalDOF] += N_psi * tmp1 @ A_IK_psi_node[:, 0]
                d_qe[nodalDOF] += N_psi * tmp2 @ A_IK_psi_node[:, 1]
                d_qe[nodalDOF] += N_psi * tmp3 @ A_IK_psi_node[:, 2]

                # kappa parts
                K_Kappa_bar_qe[0, nodalDOF] += 0.5 * (
                    N_psi_xi * d3 @ A_IK_psi_node[:, 1]
                    + N_psi * d2_xi @ A_IK_psi_node[:, 2]
                    - N_psi_xi * d2 @ A_IK_psi_node[:, 2]
                    - N_psi * d3_xi @ A_IK_psi_node[:, 1]
                )

                K_Kappa_bar_qe[1, nodalDOF] += 0.5 * (
                    N_psi_xi * d1 @ A_IK_psi_node[:, 2]
                    + N_psi * d3_xi @ A_IK_psi_node[:, 0]
                    - N_psi_xi * d3 @ A_IK_psi_node[:, 0]
                    - N_psi * d1_xi @ A_IK_psi_node[:, 2]
                )

                K_Kappa_bar_qe[2, nodalDOF] += 0.5 * (
                    N_psi_xi * d2 @ A_IK_psi_node[:, 0]
                    + N_psi * d1_xi @ A_IK_psi_node[:, 1]
                    - N_psi_xi * d1 @ A_IK_psi_node[:, 1]
                    - N_psi * d2_xi @ A_IK_psi_node[:, 0]
                )

            # derivative of K_Kappa
            K_Kappa_qe = dd * K_Kappa_bar_qe - np.outer(K_Kappa, d_qe) / d

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)

            # compute derivatives of contact forces and couples with respect
            # to their strain measures
            K_n_K_Gamma = self.material_model.K_n_K_Gamma(
                K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
            )
            K_n_K_Kappa = self.material_model.K_n_K_Kappa(
                K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
            )
            K_m_K_Gamma = self.material_model.K_m_K_Gamma(
                K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
            )
            K_m_K_Kappa = self.material_model.K_m_K_Kappa(
                K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
            )

            # chaing rule for derivative with respect to generalized coordinates
            K_n_qe = K_n_K_Gamma @ K_Gamma_qe + K_n_K_Kappa @ K_Kappa_qe
            K_m_qe = K_m_K_Gamma @ K_Gamma_qe + K_m_K_Kappa @ K_Kappa_qe

            # - first delta Gamma part
            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                N_r_xi = self.N_r_xi[el, i, node_a]

                f_pot_q_el[nodalDOF_a] -= N_r_xi * A_IK @ K_n_qe
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    f_pot_q_el[nodalDOF_a[:, None], nodalDOF_b] -= (
                        N_r_xi
                        * np.einsum("ijk,j->ik", rodriguez_der(qe[nodalDOF_b]), K_n)
                        * qwi
                    )

            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]

                N_psi = self.N_psi[el, i, node_a]
                N_psi_xi = self.N_psi_xi[el, i, node_a]

                # - second delta Gamma part (1)
                f_pot_q_el[nodalDOF_a] += N_psi * ax2skew(A_IK.T @ r_xi) @ K_n_qe * qwi

                # - delta kappa part and Euler term
                f_pot_q_el[nodalDOF_a] -= (
                    N_psi_xi
                    * (
                        (np.eye(3) - ax2skew(K_Kappa_bar)) @ K_m_qe
                        + ax2skew(K_m) @ K_Kappa_qe
                    )
                    * qwi
                )

                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]

                    # nodal derivative of rodriguez formular
                    A_IK_psi_node = rodriguez_der(qe[nodalDOF_b])

                    # - second delta Gamma part (2)
                    f_pot_q_el[nodalDOF_a[:, None], nodalDOF_b] -= (
                        N_psi
                        * ax2skew(K_n)
                        @ np.einsum("jik,j->ik", A_IK_psi_node, r_xi)
                        * qwi
                    )

        # return f_pot_q_el

        f_pot_q_el_num = approx_fprime(
            qe, lambda qe: self.f_pot_el(qe, el), method="3-point"
        )
        diff = f_pot_q_el_num - f_pot_q_el
        error = np.max(np.abs(diff))
        # error = np.max(np.abs(diff[self.rDOF[:, None], self.rDOF]))
        print(f"max error f_pot_q_el: {error}")
        print(f"")

        # print(f'diff[self.rDOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.rDOF]))}')
        # print(f'diff[self.rDOF[:, None], self.psiDOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.psiDOF]))}')
        # print(f'diff[self.psiDOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.psiDOF[:, None], self.rDOF]))}')
        # print(f'diff[self.psiDOF[:, None], self.psiDOF]: {np.max(np.abs(diff[self.psiDOF[:, None], self.psiDOF]))}')

        return f_pot_q_el_num

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        # centerline part
        q_dot = u

        # correct axis angle vector part
        for node in range(self.nnode_psi):
            nodalDOF_psi = self.nodalDOF_psi[node]

            psi = q[nodalDOF_psi]
            omega = u[nodalDOF_psi]
            psi_dot = inverse_tangent_map(psi) @ omega
            q_dot[nodalDOF_psi] = psi_dot

        return q_dot

    def B(self, t, q, coo):
        # trivial kinematic equation for centerline
        coo.extend_diag(
            np.ones(self.nq_r), (self.qDOF[: self.nq_r], self.uDOF[: self.nq_r])
        )

        # axis angle vector part
        for node in range(self.nnode_psi):
            nodalDOF_psi = self.nodalDOF_psi[node]

            psi = q[nodalDOF_psi]
            coo.extend(
                inverse_tangent_map(psi),
                (self.qDOF[nodalDOF_psi], self.uDOF[nodalDOF_psi]),
            )

    def q_ddot(self, t, q, u, u_dot):
        # centerline part
        q_ddot = u_dot

        # correct axis angle vector part
        for node in range(self.nnode_psi):
            nodalDOF_psi = self.nodalDOF_psi[node]

            psi = q[nodalDOF_psi]
            omega = u[nodalDOF_psi]
            omega_dot = u_dot[nodalDOF_psi]

            T_inv = inverse_tangent_map(psi)
            psi_dot = T_inv @ omega

            T_dot = tangent_map_s(psi, psi_dot)
            Tinv_dot = -T_inv @ T_dot @ T_inv
            psi_ddot = T_inv @ omega_dot + Tinv_dot @ omega

            q_ddot[nodalDOF_psi] = psi_ddot

        return q_ddot

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def elDOF_P(self, frame_ID):
        xi = frame_ID[0]
        el = self.element_number(xi)
        return self.elDOF[el]

    def qDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def uDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # compute centerline position
        N, _ = self.basis_functions_r(frame_ID[0])
        r_OC = np.zeros(3)
        for node in range(self.nnodes_element_r):
            r_OC += N[node] * q[self.nodalDOF_element_r[node]]

        # rigid body formular
        return r_OC + self.A_IK(t, q, frame_ID) @ K_r_SP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # compute centerline derivative
        N, _ = self.basis_functions_r(frame_ID[0])
        r_OP_q = np.zeros((3, self.nq_element))
        for node in range(self.nnodes_element_r):
            r_OP_q[:, self.nodalDOF_element_r[node]] += N[node] * np.eye(3)

        # derivative of rigid body formular
        r_OP_q += np.einsum("ijk,j->ik", self.A_IK_q(t, q, frame_ID), K_r_SP)
        return r_OP_q

        # r_OP_q_num = approx_fprime(q, lambda q: self.r_OP(t, q, frame_ID, K_r_SP), method="3-point")
        # diff = r_OP_q_num - r_OP_q
        # error = np.linalg.norm(diff)
        # print(f"error r_OP_q: {error}")
        # return r_OP_q_num

    def A_IK(self, t, q, frame_ID):
        # interpolate nodal rotation matrices
        N, _ = self.basis_functions_psi(frame_ID[0])
        A_IK = np.zeros((3, 3))
        for node in range(self.nnodes_element_psi):
            A_IK += N[node] * rodriguez(q[self.nodalDOF_element_psi[node]])
        return A_IK

    def A_IK_q(self, t, q, frame_ID):
        # interpolate nodal derivative of the rotation matrix
        N, _ = self.basis_functions_psi(frame_ID[0])
        A_IK_q = np.zeros((3, 3, self.nq_element))
        for node in range(self.nnodes_element_psi):
            nodalDOF_psi = self.nodalDOF_element_psi[node]
            A_IK_q[:, :, nodalDOF_psi] += N[node] * rodriguez_der(q[nodalDOF_psi])
        return A_IK_q

        # A_IK_q_num = approx_fprime(q, lambda q: self.A_IK(t, q, frame_ID), method="3-point")
        # diff = A_IK_q - A_IK_q_num
        # error = np.linalg.norm(diff)
        # print(f"error A_IK_q: {error}")
        # return A_IK_q_num

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3)):
        # compute centerline velocity
        N, _ = self.basis_functions_r(frame_ID[0])
        v_C = np.zeros(3)
        for node in range(self.nnodes_element_r):
            v_C += N[node] * u[self.nodalDOF_element_r[node]]

        # rigid body formular
        return v_C + self.A_IK(t, q, frame_ID) @ cross3(
            self.K_Omega(t, q, u, frame_ID), K_r_SP
        )

    def v_P_q(self, t, q, u, frame_ID, K_r_SP=np.zeros(3)):
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        return np.einsum(
            "ijk,j->ik", self.A_IK_q(t, q, frame_ID), cross3(K_Omega, K_r_SP)
        )

    def J_P(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # evaluate required nodal shape functions
        N_r, _ = self.basis_functions_r(frame_ID[0])
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # transformation matrix
        A_IK = self.A_IK(t, q, frame_ID)

        # skew symmetric matrix of K_r_SP
        K_r_SP_tilde = ax2skew(K_r_SP)

        # interpolate centerline and axis angle contributions
        J_P = np.zeros((3, self.nq_element))
        for node in range(self.nnodes_element_r):
            J_P[:, self.nodalDOF_element_r[node]] += N_r[node] * np.eye(3)
        for node in range(self.nnodes_element_psi):
            J_P[:, self.nodalDOF_element_psi[node]] -= N_psi[node] * A_IK @ K_r_SP_tilde

        return J_P

        # J_P_num = approx_fprime(
        #     np.zeros(self.nq_element), lambda u: self.v_P(t, q, u, frame_ID, K_r_SP)
        # )
        # diff = J_P_num - J_P
        # error = np.linalg.norm(diff)
        # print(f"error J_P: {error}")
        # return J_P_num

    def J_P_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # evaluate required nodal shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # skew symmetric matrix of K_r_SP
        K_r_SP_tilde = ax2skew(K_r_SP)

        # interpolate axis angle contributions since centerline contributon is
        # zero
        J_P_q = np.zeros((3, self.nq_element, self.nq_element))
        for node in range(self.nnodes_element_psi):
            nodalDOF = self.nodalDOF_element_psi[node]
            A_IK_q = rodriguez_der(q[nodalDOF])
            J_P_q[:, nodalDOF[:, None], nodalDOF] -= N_psi[node] * np.einsum(
                "ijl,jk", A_IK_q, K_r_SP_tilde
            )

        return J_P_q

        # J_P_q_num = approx_fprime(
        #     q, lambda q: self.J_P(t, q, frame_ID, K_r_SP), method="3-point"
        # )
        # diff = J_P_q_num - J_P_q
        # error = np.linalg.norm(diff)
        # print(f"error J_P_q: {error}")
        # return J_P_q_num

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=np.zeros(3)):
        # compute centerline acceleration
        N, _ = self.basis_functions_r(frame_ID[0])
        a_C = np.zeros(3)
        for node in range(self.nnodes_element_r):
            a_C += N[node] * u_dot[self.nodalDOF_element_r[node]]

        # rigid body formular
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID)
        return a_C + self.A_IK(t, q, frame_ID) @ (
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
        #     q, lambda q: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP), method="3-point"
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

        N, _ = self.basis_functions_psi(frame_ID[0])
        a_P_u = np.zeros((3, self.nq_element))
        for node in range(self.nnodes_element_r):
            a_P_u[:, self.nodalDOF_element_psi[node]] += N[node] * local

        return a_P_u

        # a_P_u_num = approx_fprime(
        #     u, lambda u: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP), method="3-point"
        # )
        # diff = a_P_u_num - a_P_u
        # error = np.linalg.norm(diff)
        # print(f"error a_P_u: {error}")
        # return a_P_u_num

    def K_Omega(self, t, q, u, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        angular velocities in the K-frame.
        """
        N, _ = self.basis_functions_psi(frame_ID[0])
        K_Omega = np.zeros(3)
        for node in range(self.nnodes_element_psi):
            K_Omega += N[node] * u[self.nodalDOF_element_psi[node]]
        return K_Omega

    def K_Omega_q(self, t, q, u, frame_ID):
        return np.zeros((3, self.nq_element))

    def K_J_R(self, t, q, frame_ID):
        N, _ = self.basis_functions_psi(frame_ID[0])
        K_J_R = np.zeros((3, self.nq_element))
        for node in range(self.nnodes_element_psi):
            K_J_R[:, self.nodalDOF_element_psi[node]] += N[node] * np.eye(3)
        return K_J_R

    def K_J_R_q(self, t, q, frame_ID):
        return np.zeros((3, self.nq_element, self.nq_element))

    def K_Psi(self, t, q, u, u_dot, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        time derivative of the angular velocities in the K-frame.
        """
        N, _ = self.basis_functions_psi(frame_ID[0])
        K_Psi = np.zeros(3)
        for node in range(self.nnodes_element_psi):
            K_Psi += N[node] * u_dot[self.nodalDOF_element_psi[node]]
        return K_Psi

    def K_Psi_q(self, t, q, u, u_dot, frame_ID):
        return np.zeros((3, self.nq_element))

    def K_Psi_u(self, t, q, u, u_dot, frame_ID):
        return np.zeros((3, self.nq_element))

    ####################################################
    # body force
    ####################################################
    def distributed_force1D_pot_el(self, force, t, qe, el):
        Ve = 0
        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # interpolate centerline position
            r_C = np.zeros(3)
            for node in range(self.nnodes_element_r):
                r_C += self.N_r[el, i, node] * qe[self.nodalDOF_element_r[node]]

            # compute potential value at given quadrature point
            Ve += (r_C @ force(t, qwi)) * Ji * qwi

        return Ve

    def distributed_force1D_pot(self, t, q, force):
        V = 0
        for el in range(self.nelement):
            qe = q[self.elDOF[el]]
            V += self.distributed_force1D_pot_el(force, t, qe, el)
        return V

    def distributed_force1D_el(self, force, t, el):
        fe = np.zeros(self.nq_element)
        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # compute local force vector
            fe_r = force(t, qwi) * Ji * qwi

            # multiply local force vector with variation of centerline
            for node in range(self.nnodes_element_r):
                fe[self.nodalDOF_element_r[node]] += self.N_r[el, i, node] * fe_r

        return fe

    def distributed_force1D(self, t, q, force):
        f = np.zeros(self.nq)
        for el in range(self.nelement):
            f[self.elDOF[el]] += self.distributed_force1D_el(force, t, el)
        return f

    def distributed_force1D_q(self, t, q, coo, force):
        pass

    ####################################################
    # visualization
    ####################################################
    def nodes(self, q):
        q_body = q[self.qDOF]
        return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF_r]).T

    def centerline(self, q, n=100):
        q_body = q[self.qDOF]
        r = []
        for xi in np.linspace(0, 1, n):
            frame_ID = (xi,)
            qp = q_body[self.qDOF_P(frame_ID)]
            r.append(self.r_OP(1, qp, frame_ID))
        return np.array(r).T

    def frames(self, q, n=10):
        q_body = q[self.qDOF]
        r = []
        d1 = []
        d2 = []
        d3 = []

        for xi in np.linspace(0, 1, n):
            frame_ID = (xi,)
            qp = q_body[self.qDOF_P(frame_ID)]
            r.append(self.r_OP(1, qp, frame_ID))

            d1i, d2i, d3i = self.A_IK(1, qp, frame_ID).T
            d1.extend([d1i])
            d2.extend([d2i])
            d3.extend([d3i])

        return np.array(r).T, np.array(d1).T, np.array(d2).T, np.array(d3).T

    def plot_centerline(self, ax, q, n=100, color="black"):
        ax.plot(*self.nodes(q), linestyle="dashed", marker="o", color=color)
        ax.plot(*self.centerline(q, n=n), linestyle="solid", color=color)

    def plot_frames(self, ax, q, n=10, length=1):
        r, d1, d2, d3 = self.frames(q, n=n)
        ax.quiver(*r, *d1, color="red", length=length)
        ax.quiver(*r, *d2, color="green", length=length)
        ax.quiver(*r, *d3, color="blue", length=length)

    ############
    # vtk export
    ############
    def post_processing_vtk_volume_circle(self, t, q, filename, R, binary=False):
        # This is mandatory, otherwise we cannot construct the 3D continuum without L2 projection!
        assert (
            self.polynomial_degree_r == self.polynomial_degree_psi
        ), "Not implemented for mixed polynomial degrees"

        # rearrange generalized coordinates from solver ordering to Piegl's Pw 3D array
        nn_xi = self.nelement + self.polynomial_degree_r
        nEl_eta = 1
        nEl_zeta = 4
        # see Cotrell2009 Section 2.4.2
        # TODO: Maybe eta and zeta have to be interchanged
        polynomial_degree_eta = 1
        polynomial_degree_zeta = 2
        nn_eta = nEl_eta + polynomial_degree_eta
        nn_zeta = nEl_zeta + polynomial_degree_zeta

        # # TODO: We do the hard coded case for rectangular cross section here, but this has to be extended to the circular cross section case too!
        # as_ = np.linspace(-a/2, a/2, num=nn_eta, endpoint=True)
        # bs_ = np.linspace(-b/2, b/2, num=nn_eta, endpoint=True)

        circle_points = (
            0.5
            * R
            * np.array(
                [
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [-1, 1, 0],
                    [-1, 0, 0],
                    [-1, -1, 0],
                    [0, -1, 0],
                    [1, -1, 0],
                    [1, 0, 0],
                ],
                dtype=float,
            )
        )

        Pw = np.zeros((nn_xi, nn_eta, nn_zeta, 3))
        for i in range(self.nnode_r):
            qr = q[self.nodalDOF_r[i]]
            q_di = q[self.nodalDOF_psi[i]]
            A_IK = q_di.reshape(3, 3, order="F")  # TODO: Check this!

            for k, point in enumerate(circle_points):
                # Note: eta index is always 0!
                Pw[i, 0, k] = qr + A_IK @ point

        if self.basis == "B-spline":
            knot_vector_eta = KnotVector(polynomial_degree_eta, nEl_eta)
            knot_vector_zeta = KnotVector(polynomial_degree_zeta, nEl_zeta)
        elif self.basis == "lagrange":
            knot_vector_eta = Node_vector(polynomial_degree_eta, nEl_eta)
            knot_vector_zeta = Node_vector(polynomial_degree_zeta, nEl_zeta)
        knot_vector_objs = [self.knot_vector_r, knot_vector_eta, knot_vector_zeta]
        degrees = (
            self.polynomial_degree_r,
            polynomial_degree_eta,
            polynomial_degree_zeta,
        )

        # Build Bezier patches from B-spline control points
        from cardillo.discretization.B_spline import decompose_B_spline_volume

        Qw = decompose_B_spline_volume(knot_vector_objs, Pw)

        nbezier_xi, nbezier_eta, nbezier_zeta, p1, q1, r1, dim = Qw.shape

        # build vtk mesh
        n_patches = nbezier_xi * nbezier_eta * nbezier_zeta
        patch_size = p1 * q1 * r1
        points = np.zeros((n_patches * patch_size, dim))
        cells = []
        HigherOrderDegrees = []
        RationalWeights = []
        vtk_cell_type = "VTK_BEZIER_HEXAHEDRON"
        from PyPanto.miscellaneous.indexing import flat3D, rearange_vtk3D

        for i in range(nbezier_xi):
            for j in range(nbezier_eta):
                for k in range(nbezier_zeta):
                    idx = flat3D(i, j, k, (nbezier_xi, nbezier_eta))
                    point_range = np.arange(idx * patch_size, (idx + 1) * patch_size)
                    points[point_range] = rearange_vtk3D(Qw[i, j, k])

                    cells.append((vtk_cell_type, point_range[None]))
                    HigherOrderDegrees.append(np.array(degrees, dtype=float)[None])
                    weight = np.sqrt(2) / 2
                    # tmp = np.array([np.sqrt(2) / 2, 1.0])
                    # RationalWeights.append(np.tile(tmp, 8)[None])
                    weights_vertices = weight * np.ones(8)
                    weights_edges = np.ones(4 * nn_xi)
                    weights_faces = np.ones(2)
                    weights_volume = np.ones(nn_xi - 2)
                    weights = np.concatenate(
                        (weights_edges, weights_vertices, weights_faces, weights_volume)
                    )
                    # weights = np.array([weight, weight, weight, weight,
                    #                     1.0,    1.0,    1.0,    1.0,
                    #                     0.0,    0.0,    0.0,    0.0 ])
                    weights = np.ones_like(point_range)
                    RationalWeights.append(weights[None])

        # RationalWeights = np.ones(len(points))
        RationalWeights = 2 * (np.random.rand(len(points)) + 1)

        # write vtk mesh using meshio
        meshio.write_points_cells(
            # filename.parent / (filename.stem + '.vtu'),
            filename,
            points,
            cells,
            point_data={
                "RationalWeights": RationalWeights,
            },
            cell_data={"HigherOrderDegrees": HigherOrderDegrees},
            binary=binary,
        )

    def post_processing_vtk_volume(self, t, q, filename, circular=True, binary=False):
        # This is mandatory, otherwise we cannot construct the 3D continuum without L2 projection!
        assert (
            self.polynomial_degree_r == self.polynomial_degree_psi
        ), "Not implemented for mixed polynomial degrees"

        # rearrange generalized coordinates from solver ordering to Piegl's Pw 3D array
        nn_xi = self.nelement + self.polynomial_degree_r
        nEl_eta = 1
        nEl_zeta = 1
        if circular:
            polynomial_degree_eta = 2
            polynomial_degree_zeta = 2
        else:
            polynomial_degree_eta = 1
            polynomial_degree_zeta = 1
        nn_eta = nEl_eta + polynomial_degree_eta
        nn_zeta = nEl_zeta + polynomial_degree_zeta

        # TODO: We do the hard coded case for rectangular cross section here, but this has to be extended to the circular cross section case too!
        if circular:
            r = 0.2
            a = b = r
        else:
            a = 0.2
            b = 0.1
        as_ = np.linspace(-a / 2, a / 2, num=nn_eta, endpoint=True)
        bs_ = np.linspace(-b / 2, b / 2, num=nn_eta, endpoint=True)

        Pw = np.zeros((nn_xi, nn_eta, nn_zeta, 3))
        for i in range(self.nnode_r):
            qr = q[self.nodalDOF_r[i]]
            q_di = q[self.nodalDOF_psi[i]]
            A_IK = q_di.reshape(3, 3, order="F")  # TODO: Check this!

            for j, aj in enumerate(as_):
                for k, bk in enumerate(bs_):
                    Pw[i, j, k] = qr + A_IK @ np.array([0, aj, bk])

        if self.basis == "B-spline":
            knot_vector_eta = KnotVector(polynomial_degree_eta, nEl_eta)
            knot_vector_zeta = KnotVector(polynomial_degree_zeta, nEl_zeta)
        elif self.basis == "lagrange":
            knot_vector_eta = Node_vector(polynomial_degree_eta, nEl_eta)
            knot_vector_zeta = Node_vector(polynomial_degree_zeta, nEl_zeta)
        knot_vector_objs = [self.knot_vector_r, knot_vector_eta, knot_vector_zeta]
        degrees = (
            self.polynomial_degree_r,
            polynomial_degree_eta,
            polynomial_degree_zeta,
        )

        # Build Bezier patches from B-spline control points
        from cardillo.discretization.B_spline import decompose_B_spline_volume

        Qw = decompose_B_spline_volume(knot_vector_objs, Pw)

        nbezier_xi, nbezier_eta, nbezier_zeta, p1, q1, r1, dim = Qw.shape

        # build vtk mesh
        n_patches = nbezier_xi * nbezier_eta * nbezier_zeta
        patch_size = p1 * q1 * r1
        points = np.zeros((n_patches * patch_size, dim))
        cells = []
        HigherOrderDegrees = []
        RationalWeights = []
        vtk_cell_type = "VTK_BEZIER_HEXAHEDRON"
        from PyPanto.miscellaneous.indexing import flat3D, rearange_vtk3D

        for i in range(nbezier_xi):
            for j in range(nbezier_eta):
                for k in range(nbezier_zeta):
                    idx = flat3D(i, j, k, (nbezier_xi, nbezier_eta))
                    point_range = np.arange(idx * patch_size, (idx + 1) * patch_size)
                    points[point_range] = rearange_vtk3D(Qw[i, j, k])

                    cells.append((vtk_cell_type, point_range[None]))
                    HigherOrderDegrees.append(np.array(degrees, dtype=float)[None])
                    weight = np.sqrt(2) / 2
                    # tmp = np.array([np.sqrt(2) / 2, 1.0])
                    # RationalWeights.append(np.tile(tmp, 8)[None])
                    weights_vertices = weight * np.ones(8)
                    weights_edges = np.ones(4 * nn_xi)
                    weights_faces = np.ones(2)
                    weights_volume = np.ones(nn_xi - 2)
                    weights = np.concatenate(
                        (weights_edges, weights_vertices, weights_faces, weights_volume)
                    )
                    # weights = np.array([weight, weight, weight, weight,
                    #                     1.0,    1.0,    1.0,    1.0,
                    #                     0.0,    0.0,    0.0,    0.0 ])
                    weights = np.ones_like(point_range)
                    RationalWeights.append(weights[None])

        # RationalWeights = np.ones(len(points))
        RationalWeights = 2 * (np.random.rand(len(points)) + 1)

        # write vtk mesh using meshio
        meshio.write_points_cells(
            # filename.parent / (filename.stem + '.vtu'),
            filename,
            points,
            cells,
            point_data={
                "RationalWeights": RationalWeights,
            },
            cell_data={"HigherOrderDegrees": HigherOrderDegrees},
            binary=binary,
        )

    def post_processing(self, t, q, filename, binary=True):
        # write paraview PVD file collecting time and all vtk files, see https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
        from xml.dom import minidom

        root = minidom.Document()

        vkt_file = root.createElement("VTKFile")
        vkt_file.setAttribute("type", "Collection")
        root.appendChild(vkt_file)

        collection = root.createElement("Collection")
        vkt_file.appendChild(collection)

        for i, (ti, qi) in enumerate(zip(t, q)):
            filei = filename + f"{i}.vtu"

            # write time step and file name in pvd file
            dataset = root.createElement("DataSet")
            dataset.setAttribute("timestep", f"{ti:0.6f}")
            dataset.setAttribute("file", filei)
            collection.appendChild(dataset)

            self.post_processing_single_configuration(ti, qi, filei, binary=binary)

        # write pvd file
        xml_str = root.toprettyxml(indent="\t")
        with open(filename + ".pvd", "w") as f:
            f.write(xml_str)

    def post_processing_single_configuration(self, t, q, filename, binary=True):
        # centerline and connectivity
        cells_r, points_r, HigherOrderDegrees_r = self.mesh_r.vtk_mesh(q[: self.nq_r])

        # if the centerline and the directors are interpolated with the same
        # polynomial degree we can use the values on the nodes and decompose the B-spline
        # into multiple Bezier patches, otherwise the directors have to be interpolated
        # onto the nodes of the centerline by a so-called L2-projection, see below
        same_shape_functions = False
        if self.polynomial_degree_r == self.polynomial_degree_psi:
            same_shape_functions = True

        if same_shape_functions:
            _, points_di, _ = self.mesh_psi.vtk_mesh(q[self.nq_r :])

            # fill dictionary storing point data with directors
            point_data = {
                "d1": points_di[:, 0:3],
                "d2": points_di[:, 3:6],
                "d3": points_di[:, 6:9],
            }

        else:
            point_data = {}

        # export existing values on quadrature points using L2 projection
        J0_vtk = self.mesh_r.field_to_vtk(
            self.J.reshape(self.nelement, self.nquadrature, 1)
        )
        point_data.update({"J0": J0_vtk})

        Gamma0_vtk = self.mesh_r.field_to_vtk(self.K_Gamma0)
        point_data.update({"Gamma0": Gamma0_vtk})

        Kappa0_vtk = self.mesh_r.field_to_vtk(self.K_Kappa0)
        point_data.update({"Kappa0": Kappa0_vtk})

        # evaluate fields at quadrature points that have to be projected onto the centerline mesh:
        # - strain measures Gamma & Kappa
        # - directors d1, d2, d3
        Gamma = np.zeros((self.mesh_r.nelement, self.mesh_r.nquadrature, 3))
        Kappa = np.zeros((self.mesh_r.nelement, self.mesh_r.nquadrature, 3))
        if not same_shape_functions:
            d1s = np.zeros((self.mesh_r.nelement, self.mesh_r.nquadrature, 3))
            d2s = np.zeros((self.mesh_r.nelement, self.mesh_r.nquadrature, 3))
            d3s = np.zeros((self.mesh_r.nelement, self.mesh_r.nquadrature, 3))
        for el in range(self.nelement):
            qe = q[self.elDOF[el]]

            # extract generalized coordinates for beam centerline and directors
            # in the current and reference configuration
            qe_r = qe[self.rDOF]
            qe_d1 = qe[self.psiDOF]
            qe_d2 = qe[self.d2DOF]
            qe_d3 = qe[self.d3DOF]

            for i in range(self.nquadrature):
                # build matrix of shape function derivatives
                NN_di_i = self.stack3psi(self.N_psi[el, i])
                NN_r_xii = self.stack3r(self.N_r_xi[el, i])
                NN_di_xii = self.stack3psi(self.N_psi_xi[el, i])

                # extract reference state variables
                J0i = self.J[el, i]
                K_Gamma0 = self.K_Gamma0[el, i]
                K_Kappa0 = self.K_Kappa0[el, i]

                # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
                # the parameter space \xi and thus need to be transformed later
                r_xi = NN_r_xii @ qe_r

                d1 = NN_di_i @ qe_d1
                d1_xi = NN_di_xii @ qe_d1

                d2 = NN_di_i @ qe_d2
                d2_xi = NN_di_xii @ qe_d2

                d3 = NN_di_i @ qe_d3
                d3_xi = NN_di_xii @ qe_d3

                # compute derivatives w.r.t. the arc lenght parameter s
                r_s = r_xi / J0i

                d1_s = d1_xi / J0i
                d2_s = d2_xi / J0i
                d3_s = d3_xi / J0i

                # build rotation matrices
                if not same_shape_functions:
                    d1s[el, i] = d1
                    d2s[el, i] = d2
                    d3s[el, i] = d3
                R = np.vstack((d1, d2, d3)).T

                # axial and shear strains
                Gamma[el, i] = R.T @ r_s

                # torsional and flexural strains
                Kappa[el, i] = np.array(
                    [
                        0.5 * (d3 @ d2_s - d2 @ d3_s),
                        0.5 * (d1 @ d3_s - d3 @ d1_s),
                        0.5 * (d2 @ d1_s - d1 @ d2_s),
                    ]
                )

        # L2 projection of strain measures
        Gamma_vtk = self.mesh_r.field_to_vtk(Gamma)
        point_data.update({"Gamma": Gamma_vtk})

        Kappa_vtk = self.mesh_r.field_to_vtk(Kappa)
        point_data.update({"Kappa": Kappa_vtk})

        # L2 projection of directors
        if not same_shape_functions:
            d1_vtk = self.mesh_r.field_to_vtk(d1s)
            point_data.update({"d1": d1_vtk})
            d2_vtk = self.mesh_r.field_to_vtk(d2s)
            point_data.update({"d2": d2_vtk})
            d3_vtk = self.mesh_r.field_to_vtk(d3s)
            point_data.update({"d3": d3_vtk})

        # fields depending on strain measures and other previously computed quantities
        point_data_fields = {
            "W": lambda Gamma, Gamma0, Kappa, Kappa0: np.array(
                [self.material_model.potential(Gamma, Gamma0, Kappa, Kappa0)]
            ),
            "n_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.n_i(
                Gamma, Gamma0, Kappa, Kappa0
            ),
            "m_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.m_i(
                Gamma, Gamma0, Kappa, Kappa0
            ),
        }

        for name, fun in point_data_fields.items():
            tmp = fun(Gamma_vtk[0], Gamma0_vtk[0], Kappa_vtk[0], Kappa0_vtk[0]).reshape(
                -1
            )
            field = np.zeros((len(Gamma_vtk), len(tmp)))
            for i, (K_Gamma, K_Gamma0, K_Kappa, K_Kappa0) in enumerate(
                zip(Gamma_vtk, Gamma0_vtk, Kappa_vtk, Kappa0_vtk)
            ):
                field[i] = fun(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0).reshape(-1)
            point_data.update({name: field})

        # write vtk mesh using meshio
        meshio.write_points_cells(
            os.path.splitext(os.path.basename(filename))[0] + ".vtu",
            points_r,  # only export centerline as geometry here!
            cells_r,
            point_data=point_data,
            cell_data={"HigherOrderDegrees": HigherOrderDegrees_r},
            binary=binary,
        )

    def post_processing_subsystem(self, t, q, u, binary=True):
        # centerline and connectivity
        cells_r, points_r, HigherOrderDegrees_r = self.mesh_r.vtk_mesh(q[: self.nq_r])

        # if the centerline and the directors are interpolated with the same
        # polynomial degree we can use the values on the nodes and decompose the B-spline
        # into multiple Bezier patches, otherwise the directors have to be interpolated
        # onto the nodes of the centerline by a so-called L2-projection, see below
        same_shape_functions = False
        if self.polynomial_degree_r == self.polynomial_degree_psi:
            same_shape_functions = True

        if same_shape_functions:
            _, points_di, _ = self.mesh_psi.vtk_mesh(q[self.nq_r :])

            # fill dictionary storing point data with directors
            point_data = {
                "u_r": points_r - points_r[0],
                "d1": points_di[:, 0:3],
                "d2": points_di[:, 3:6],
                "d3": points_di[:, 6:9],
            }

        else:
            point_data = {}

        # export existing values on quadrature points using L2 projection
        J0_vtk = self.mesh_r.field_to_vtk(
            self.J.reshape(self.nelement, self.nquadrature, 1)
        )
        point_data.update({"J0": J0_vtk})

        Gamma0_vtk = self.mesh_r.field_to_vtk(self.K_Gamma0)
        point_data.update({"Gamma0": Gamma0_vtk})

        Kappa0_vtk = self.mesh_r.field_to_vtk(self.K_Kappa0)
        point_data.update({"Kappa0": Kappa0_vtk})

        # evaluate fields at quadrature points that have to be projected onto the centerline mesh:
        # - strain measures Gamma & Kappa
        # - directors d1, d2, d3
        Gamma = np.zeros((self.mesh_r.nelement, self.mesh_r.nquadrature, 3))
        Kappa = np.zeros((self.mesh_r.nelement, self.mesh_r.nquadrature, 3))
        if not same_shape_functions:
            d1s = np.zeros((self.mesh_r.nelement, self.mesh_r.nquadrature, 3))
            d2s = np.zeros((self.mesh_r.nelement, self.mesh_r.nquadrature, 3))
            d3s = np.zeros((self.mesh_r.nelement, self.mesh_r.nquadrature, 3))
        for el in range(self.nelement):
            qe = q[self.elDOF[el]]

            # extract generalized coordinates for beam centerline and directors
            # in the current and reference configuration
            qe_r = qe[self.rDOF]
            qe_d1 = qe[self.psiDOF]
            qe_d2 = qe[self.d2DOF]
            qe_d3 = qe[self.d3DOF]

            for i in range(self.nquadrature):
                # build matrix of shape function derivatives
                NN_di_i = self.stack3psi(self.N_psi[el, i])
                NN_r_xii = self.stack3r(self.N_r_xi[el, i])
                NN_di_xii = self.stack3psi(self.N_psi_xi[el, i])

                # extract reference state variables
                J0i = self.J[el, i]
                K_Gamma0 = self.K_Gamma0[el, i]
                K_Kappa0 = self.K_Kappa0[el, i]

                # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
                # the parameter space \xi and thus need to be transformed later
                r_xi = NN_r_xii @ qe_r

                d1 = NN_di_i @ qe_d1
                d1_xi = NN_di_xii @ qe_d1

                d2 = NN_di_i @ qe_d2
                d2_xi = NN_di_xii @ qe_d2

                d3 = NN_di_i @ qe_d3
                d3_xi = NN_di_xii @ qe_d3

                # compute derivatives w.r.t. the arc lenght parameter s
                r_s = r_xi / J0i

                d1_s = d1_xi / J0i
                d2_s = d2_xi / J0i
                d3_s = d3_xi / J0i

                # build rotation matrices
                if not same_shape_functions:
                    d1s[el, i] = d1
                    d2s[el, i] = d2
                    d3s[el, i] = d3
                R = np.vstack((d1, d2, d3)).T

                # axial and shear strains
                Gamma[el, i] = R.T @ r_s

                # torsional and flexural strains
                Kappa[el, i] = np.array(
                    [
                        0.5 * (d3 @ d2_s - d2 @ d3_s),
                        0.5 * (d1 @ d3_s - d3 @ d1_s),
                        0.5 * (d2 @ d1_s - d1 @ d2_s),
                    ]
                )

        # L2 projection of strain measures
        Gamma_vtk = self.mesh_r.field_to_vtk(Gamma)
        point_data.update({"Gamma": Gamma_vtk})

        Kappa_vtk = self.mesh_r.field_to_vtk(Kappa)
        point_data.update({"Kappa": Kappa_vtk})

        # L2 projection of directors
        if not same_shape_functions:
            d1_vtk = self.mesh_r.field_to_vtk(d1s)
            point_data.update({"d1": d1_vtk})
            d2_vtk = self.mesh_r.field_to_vtk(d2s)
            point_data.update({"d2": d2_vtk})
            d3_vtk = self.mesh_r.field_to_vtk(d3s)
            point_data.update({"d3": d3_vtk})

        # fields depending on strain measures and other previously computed quantities
        point_data_fields = {
            "W": lambda Gamma, Gamma0, Kappa, Kappa0: np.array(
                [self.material_model.potential(Gamma, Gamma0, Kappa, Kappa0)]
            ),
            "n_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.n_i(
                Gamma, Gamma0, Kappa, Kappa0
            ),
            "m_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.m_i(
                Gamma, Gamma0, Kappa, Kappa0
            ),
        }

        for name, fun in point_data_fields.items():
            tmp = fun(Gamma_vtk[0], Gamma0_vtk[0], Kappa_vtk[0], Kappa0_vtk[0]).reshape(
                -1
            )
            field = np.zeros((len(Gamma_vtk), len(tmp)))
            for i, (K_Gamma, K_Gamma0, K_Kappa, K_Kappa0) in enumerate(
                zip(Gamma_vtk, Gamma0_vtk, Kappa_vtk, Kappa0_vtk)
            ):
                field[i] = fun(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0).reshape(-1)
            point_data.update({name: field})

        return points_r, point_data, cells_r, HigherOrderDegrees_r
