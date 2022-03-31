import numpy as np
import meshio
import os

from cardillo.utility.coo import Coo
from cardillo.discretization.B_spline import KnotVector
from cardillo.discretization.lagrange import Node_vector
from cardillo.discretization.mesh1D import Mesh1D
from cardillo.math import norm, cross3, skew2ax, skew2ax_A, ax2skew, approx_fprime
from PyRod.math.rotations import (
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
            nq_node=nq_node_r,
        )
        self.mesh_psi = Mesh1D(
            self.knot_vector_psi,
            nquadrature,
            derivative_order=1,
            basis=basis,
            nq_node=nq_node_psi,
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

        # global nodal connectivity
        self.nodalDOF_r = (
            np.arange(self.nq_node_r * self.nnode_r)
            .reshape(self.nq_node_r, self.nnode_r)
            .T
        )
        self.nodalDOF_psi = (
            np.arange(self.nq_node_psi * self.nnode_psi)
            .reshape(self.nq_node_psi, self.nnode_psi)
            .T
            + nq_r
        )

        # nodal connectivity on element level
        self.nodalDOF_element_r = np.arange(
            self.nq_node_r * self.nnodes_element_r
        ).reshape(self.nnodes_element_r, -1, order="F")
        self.nodalDOF_element_psi = np.arange(
            self.nq_node_psi * self.nnodes_element_psi
        ).reshape(self.nnodes_element_psi, -1, order="F") + self.nq_el_r

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
            # precompute quantities of the reference configuration
            qe = self.Q[self.elDOF[el]]
            qe_r = qe[self.rDOF]
            qe_psi = qe[self.psiDOF]

            # Extract nodal rotation vectors evaluate the rotation using
            # Rodriguez' formular and extract the nodal directors. Further,
            # they are rearranged such that they can be interpolated using
            # vector valued shape function stacks.
            qe_d1, qe_d2, qe_d3 = self.qe_psi2qe_di(qe_psi)

            # # since we cannot extract the derivatives of nodal rotation vectors
            # # we compute their values on fixed xis's
            # # psi = np.array([])
            # # for no in range(self.nnode_psi):
            # psis = np.zeros((self.nnode_psi, 3))
            # # psi_xis = np.zeros((self.nnode_psi, 3))
            # Rs = np.zeros((self.nnode_psi, 3, 3))
            # # R_xis = np.zeros((self.nnode_psi, 3, 3))
            # for i, xii in enumerate(self.xi_el_psi(el)):
            #     # TODO: Precompute these shape functions.
            #     NN_psi_i = self.stack3psi(self.basis_functions_psi(xii)[0])
            #     # NN_psi_xii = self.stack3psi(self.basis_functions_psi(xii)[1])
            #     psi = NN_psi_i @ qe_psi
            #     # psi_xi = NN_psi_xii @ qe_psi
            #     psis[i] = psi
            #     # psi_xis[i] = psi_xi
            #     # TODO: Do we require psi_s here? This requires the evaluation
            #     #       of J(xi) at the fixed xi values.

            #     # evaluate rotation and extract directors
            #     R = rodriguez(psi)
            #     d1, d2, d3 = R.T
            #     Rs[i] = R
            #     # # TODO: This is not Kapp_i since 1 / J is missing!
            #     # K_Kappa = tangent_map(psi) @ psi_xi
            #     # d1_xi = cross3(K_Kappa, d1)
            #     # d2_xi = cross3(K_Kappa, d2)
            #     # d3_xi = cross3(K_Kappa, d3)
            #     # R_xi = np.vstack((d1_xi, d2_xi, d3_xi)).T
            #     # R_xis[i] = R_xi

            #     # # TODO: Remove debugging prints.
            #     # print(f"psi: {psi}")
            #     # print(f"psi_xi: {psi_xi}")
            #     # print(f"K_Kappa * J: {K_Kappa}")
            #     # print(f"R:\n{R}")
            #     # print(f"R_xi:\n{R_xi}")
            #     # print(f"")

            for i in range(nquadrature):
                # build matrix of shape function derivatives
                NN_r_xii = self.stack3r(self.N_r_xi[el, i])
                NN_psi_i = self.stack3psi(self.N_psi[el, i])
                NN_psi_xii = self.stack3psi(self.N_psi_xi[el, i])

                # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
                # the parameter space \xi and thus need to be transformed later
                r_xi = NN_r_xii @ qe_r
                Ji = ji = norm(r_xi)
                self.J[el, i] = Ji

                # interpolate these directors using the existing shae functions
                # of the rotation vector
                d1 = NN_psi_i @ qe_d1
                d2 = NN_psi_i @ qe_d2
                d3 = NN_psi_i @ qe_d3
                d1_xi = NN_psi_xii @ qe_d1
                d2_xi = NN_psi_xii @ qe_d2
                d3_xi = NN_psi_xii @ qe_d3

                # compute derivatives w.r.t. the arc lenght parameter s
                r_s = r_xi / Ji
                d1_s = d1_xi / Ji
                d2_s = d2_xi / Ji
                d3_s = d3_xi / Ji

                # axial and shear strains
                self.K_Gamma0[el, i] = np.array([r_s @ d1, r_s @ d2, r_s @ d3])

                # torsional and flexural strains
                self.K_Kappa0[el, i] = np.array(
                    [
                        0.5 * (d3 @ d2_s - d2 @ d3_s),
                        0.5 * (d1 @ d3_s - d3 @ d1_s),
                        0.5 * (d2 @ d1_s - d1 @ d2_s),
                    ]
                )

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
        elif basis == "lagrange":
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

        r0 = np.vstack((x0, y0, z0)).T
        for i, r0i in enumerate(r0):
            x0[i], y0[i], z0[i] = r_OP + A_IK @ r0i

        # we have to extract the rotation vector from the given rotation matrix
        psi = rodriguez_inv(A_IK)
        psi0 = np.repeat(psi, nn_psi)
        # TODO: This is wrong. How do we get the Greville abszissae for the
        #       rotation vector?
        #       Since we have a straight beem all rotation vectors are the
        #       same. Thus, Greville abszissae should not be necessary!
        # if greville_abscissae and basis == "B-spline":
        #     kv = KnotVector.uniform(polynomial_degree_psi, nEl)
        #     for i in range(nn_r):
        #         psi0[i] = np.sum(kv[i + 1 : i + polynomial_degree_r + 1])
        #     psi0 = psi0 * L / polynomial_degree_psi

        return np.concatenate([x0, y0, z0, psi0])

    # TODO: I think we should replace this by a function that computes the
    # nodal rotation matrices. Later they are interpolated using the very
    # same procedure bu we use some more efficient BLAS level 3 functions
    # (matrx matrix multiplication) that should (at least in theory) be
    # faster than the corresponding level 2 equivalents (matrix vector
    # multiplications). The very same can be applied for their derivatives.
    # The curvatures is computed by ax2skew(A_IK.T @ A_IK_s).
    def nodal_rotations(self, qe_psi):
        # Extract nodal rotation vectors evaluate the rotation using
        # Rodriguez' formular and extract the nodal rotation.
        # In addition, they are rearranged such that they can be interpolated
        # using vector valued shape function stacks.
        psis = qe_psi.reshape(self.nnodes_element_psi, -1, order="F")
        Rs = np.array([rodriguez(psi) for psi in psis])
        return Rs

    def qe_psi2qe_di(self, qe_psi):
        # Extract nodal rotation vectors evaluate the rotation using
        # Rodriguez' formular and extract the nodal directors. Further,
        # they are rearranged such that they can be interpolated using
        # vector valued shape function stacks.
        psis = qe_psi.reshape(self.nnodes_element_psi, -1, order="F")
        Rs = np.array([rodriguez(psi) for psi in psis])
        qe_d1 = Rs[:, :, 0].reshape(-1, order="F")
        qe_d2 = Rs[:, :, 1].reshape(-1, order="F")
        qe_d3 = Rs[:, :, 2].reshape(-1, order="F")
        return qe_d1, qe_d2, qe_d3

    def qe_di_qe_psi(self, qe_psi):
        # Compute the derivative of the nonlinear mapping from nodal axial
        # vectors to directors via Rodriguez' formular with respect to nodal
        # axial vectors.
        # This is required for computing the derivative of the internal forces
        # with respect to the generalized coordinates.
        psis = qe_psi.reshape(self.nnodes_element_psi, -1, order="F")
        Rs_psi = np.array([rodriguez_der(psi) for psi in psis])
        # Rs = np.array([rodriguez(psi) for psi in psis])
        qe_d1 = Rs_psi[:, :, 0].reshape(-1, order="F")
        qe_d2 = Rs_psi[:, :, 1].reshape(-1, order="F")
        qe_d3 = Rs_psi[:, :, 2].reshape(-1, order="F")
        # return qe_d1, qe_d2, qe_d3

    def element_number(self, xi):
        # note the elements coincide for both meshes!
        return self.knot_vector_r.element_number(xi)[0]

    def stack3r(self, N):
        nn_el = self.nnodes_element_r
        NN = np.zeros((3, self.nq_el_r))
        NN[0, :nn_el] = N
        NN[1, nn_el : 2 * nn_el] = N
        NN[2, 2 * nn_el :] = N
        return NN

    def stack3psi(self, N):
        nn_el = self.nnodes_element_psi
        NN = np.zeros((3, self.nq_el_psi))
        NN[0, :nn_el] = N
        NN[1, nn_el : 2 * nn_el] = N
        NN[2, 2 * nn_el :] = N
        return NN

    def assembler_callback(self):
        self.__M_coo()

    #########################################
    # equations of motion
    #########################################
    def M_el(self, el):
        Me = np.zeros((self.nq_element, self.nq_element))

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # build matrix of shape functions
            NN_r_i = self.stack3r(self.N_r[el, i])
            NN_psi_i = self.stack3psi(self.N_psi[el, i])

            # delta_r A_rho0 r_ddot part
            Me[self.rDOF[:, None], self.rDOF] += (
                NN_r_i.T @ NN_r_i * self.A_rho0 * Ji * qwi
            )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            Me[self.psiDOF[:, None], self.psiDOF] += (
                NN_psi_i.T @ self.I_rho0 @ NN_psi_i * Ji * qwi
            )

        return Me

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
        fe = np.zeros(self.nq_element)

        # extract generalized velocities of axis angle vector
        ue_psi = ue[self.psiDOF]

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # build matrix of shape function derivatives
            NN_psi_i = self.stack3psi(self.N_psi[el, i])

            # angular velocity
            omega = NN_psi_i @ ue_psi
            fe[self.psiDOF] += (
                NN_psi_i.T @ cross3(omega, self.I_rho0 @ omega) * Ji * qwi
            )

        return fe

    def f_gyr(self, t, q, u):
        f = np.zeros(self.nu)
        for el in range(self.nelement):
            f[self.elDOF[el]] += self.f_gyr_el(
                t, q[self.elDOF[el]], u[self.elDOF[el]], el
            )
        return f

    def f_gyr_u_el(self, t, qe, ue, el):
        fe_ue = np.zeros((self.nq_element, self.nq_element))

        # extract generalized velocities of axis angle vector
        ue_psi = ue[self.psiDOF]

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # build matrix of shape function derivatives
            NN_psi_i = self.stack3psi(self.N_psi[el, i])

            # angular velocity
            omega = NN_psi_i @ ue_psi

            fe_ue[self.psiDOF[:, None], self.psiDOF] += (
                NN_psi_i.T @ (ax2skew(omega) @ self.I_rho0 - ax2skew(self.I_rho0 @ omega)) @ NN_psi_i * Ji * qwi
            )

        return fe_ue

    def f_gyr_u(self, t, q, u, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            Ke = self.f_gyr_u_el(t, q[elDOF], u[elDOF], el)
            coo.extend(Ke, (self.uDOF[elDOF], self.uDOF[elDOF]))

    def E_pot(self, t, q):
        E = 0
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            E += self.E_pot_el(q[elDOF], el)
        return E

    def E_pot_el(self, qe, el):
        Ee = 0

        # extract generalized coordinates for beam centerline and directors
        # in the current and reference configuration
        qe_r = qe[self.rDOF]
        qe_psi = qe[self.psiDOF]

        # Extract nodal rotation vectors evaluate the rotation using
        # Rodriguez' formular and extract the nodal directors. Further,
        # they are rearranged such that they can be interpolated using
        # vector valued shape function stacks.
        qe_d1, qe_d2, qe_d3 = self.qe_psi2qe_di(qe_psi)

        for i in range(self.nquadrature):
            # extract reference state variables
            Ji = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # build matrix of shape function derivatives
            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            NN_psi_i = self.stack3psi(self.N_psi[el, i])
            NN_psi_xii = self.stack3psi(self.N_psi_xi[el, i])

            # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
            # the parameter space \xi and thus need to be transformed later
            r_xi = NN_r_xii @ qe_r

            # interpolate these directors using the existing shae functions
            # of the rotation vector
            d1 = NN_psi_i @ qe_d1
            d2 = NN_psi_i @ qe_d2
            d3 = NN_psi_i @ qe_d3
            d1_xi = NN_psi_xii @ qe_d1
            d2_xi = NN_psi_xii @ qe_d2
            d3_xi = NN_psi_xii @ qe_d3

            # compute derivatives w.r.t. the arc lenght parameter s
            r_s = r_xi / Ji
            d1_s = d1_xi / Ji
            d2_s = d2_xi / Ji
            d3_s = d3_xi / Ji

            # axial and shear strains
            K_Gamma = np.array([r_s @ d1, r_s @ d2, r_s @ d3])

            # torsional and flexural strains
            K_Kappa = np.array(
                [
                    0.5 * (d3 @ d2_s - d2 @ d3_s),
                    0.5 * (d1 @ d3_s - d3 @ d1_s),
                    0.5 * (d2 @ d1_s - d1 @ d2_s),
                ]
            )

            # evaluate strain energy function
            Ee += (
                self.material_model.potential(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
                * Ji
                * self.qw[el, i]
            )

        return Ee

    def f_pot(self, t, q):
        f = np.zeros(self.nu)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            # f[elDOF] += self.f_pot_el(q[elDOF], el)
            f[elDOF] += self.f_pot_el_test(q[elDOF], el)
        # self.f_pot_el_test(q[elDOF], el)
        return f

    def f_pot_el(self, qe, el):
        fe = np.zeros(self.nq_element)

        # extract generalized coordinates for beam centerline and directors
        # in the current and reference configuration
        qe_r = qe[self.rDOF]
        qe_psi = qe[self.psiDOF]

        # Extract nodal rotation vectors evaluate the rotation using
        # Rodriguez' formular and extract the nodal directors. Further,
        # they are rearranged such that they can be interpolated using
        # vector valued shape function stacks.
        qe_d1, qe_d2, qe_d3 = self.qe_psi2qe_di(qe_psi)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # build matrix of shape function derivatives
            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            NN_psi_i = self.stack3psi(self.N_psi[el, i])
            NN_psi_xii = self.stack3psi(self.N_psi_xi[el, i])

            # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
            # the parameter space \xi and thus need to be transformed later
            r_xi = NN_r_xii @ qe_r

            # interpolate these directors using the existing shae functions
            # of the rotation vector
            d1 = NN_psi_i @ qe_d1
            d2 = NN_psi_i @ qe_d2
            d3 = NN_psi_i @ qe_d3
            d1_xi = NN_psi_xii @ qe_d1
            d2_xi = NN_psi_xii @ qe_d2
            d3_xi = NN_psi_xii @ qe_d3

            # compute derivatives w.r.t. the arc lenght parameter s
            r_s = r_xi / Ji
            d1_s = d1_xi / Ji
            d2_s = d2_xi / Ji
            d3_s = d3_xi / Ji

            # axial and shear strains
            K_Gamma = np.array([r_s @ d1, r_s @ d2, r_s @ d3])

            #################################################################
            # formulation of Harsch2020b
            #################################################################

            # torsional and flexural strains
            K_Kappa = np.array(
                [
                    0.5 * (d3 @ d2_s - d2 @ d3_s),
                    0.5 * (d1 @ d3_s - d3 @ d1_s),
                    0.5 * (d2 @ d1_s - d1 @ d2_s),
                ]
            )

            ###################################################
            # formulation in skew coordinates, see Eugster2014c
            ###################################################
            # TODO: Using this formulation a perfect circle is obtained.
            # This is not the case without the 1/d term. A huge number
            # of elements is required otherwise!
            # TODO: We have to check if this is the only change invoved?
            # I think the term involving the K_Kappa requires a factor d too!
            # TODO: Investiage why rotation of left beam end is not working
            # with this formulation.

            # # torsional and flexural strains
            # d = d1 @ cross3(d2, d3)
            # K_Kappa = np.array(
            #     [
            #         0.5 * (d3 @ d2_s - d2 @ d3_s) / d,
            #         0.5 * (d1 @ d3_s - d3 @ d1_s) / d,
            #         0.5 * (d2 @ d1_s - d1 @ d2_s) / d,
            #     ]
            # )

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)

            # stack rotation
            A_IK = np.vstack((d1, d2, d3)).T

            # first delta Gamma part
            fe[self.rDOF] += -NN_r_xii.T @ A_IK @ (K_n * qwi)

            #####################
            # K_delta_phi version
            #####################
            # - second delta Gamma part
            fe[self.psiDOF] += NN_psi_i.T @ cross3(A_IK.T @ r_xi, K_n) * qwi

            # - delta kappa part
            fe[self.psiDOF] += -NN_psi_xii.T @ K_m * qwi
            fe[self.psiDOF] += (
                NN_psi_i.T
                @ cross3(K_Kappa, K_m)
                * Ji
                * qwi
                # NN_psi_i.T @ cross3(K_Kappa, K_m) * Ji * d * qwi
            )  # Euler term

        return fe

    def f_pot_q(self, t, q, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            Ke = self.f_pot_q_el(q[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(Ke, (self.uDOF[elDOF], self.qDOF[elDOF]))

    # def f_pot_q_el(self, qe, el):
    #     return approx_fprime(qe, lambda qe: self.f_pot_el(qe, el), method="2-point")


    # Interpolation of nodal rotations
    def f_pot_el_test(self, qe, el):
        fe = np.zeros(self.nq_element)

        # extract generalized coordinates for beam centerline and directors
        # in the current and reference configuration
        qe_r = qe[self.rDOF]
        qe_psi = qe[self.psiDOF]

        # Extract nodal values for centerline and rotation vector. Further the
        # corresponding nodal rotation is evaluated using Rodriguez' formular.
        qe_r_nodes = qe_r.reshape(self.nnodes_element_psi, -1, order="F")
        qe_psi_nodes = qe_psi.reshape(self.nnodes_element_psi, -1, order="F")
        qe_R_nodes = np.array([rodriguez(psi) for psi in qe_psi_nodes])

        for i in range(self.nquadrature):
            # extract some already known variables
            N_r_xi = self.N_r_xi[el, i]
            N_psi = self.N_psi[el, i]
            N_psi_xi = self.N_psi_xi[el, i]
            qw = self.qw[el, i]
            J = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # interpolate tangent vector, transformation matrix and its
            # derivative with respect to parameter space
            r_xi = np.einsum("i,ij", N_r_xi, qe_r_nodes)
            A_IK = np.einsum("i,ijk", N_psi, qe_R_nodes)
            A_IK_xi = np.einsum("i,ijk", N_psi_xi, qe_R_nodes)

            # compute derivatives w.r.t. the arc lenght parameter s
            r_s = r_xi / J
            A_IK_s = A_IK_xi / J

            # axial and shear strains
            K_Gamma = A_IK.T @ r_s

            # torsional and flexural strains
            K_Kappa = skew2ax(A_IK.T @ A_IK_s)

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)

            ########################
            # first delta Gamma part
            ########################
            for a in range(self.nnodes_element_r):
                fe[self.nodalDOF_element_r[a]] -= N_r_xi[a] * A_IK @ (K_n * qw)

            #########################
            # second delta Gamma part
            #########################
            for a in range(self.nnodes_element_psi):
                fe[self.nodalDOF_element_psi[a]] += N_psi[a] * cross3(A_IK.T @ r_xi, K_n) * qw

            ##################
            # delta kappa part
            ##################
            for a in range(self.nnodes_element_psi):
                fe[self.nodalDOF_element_psi[a]] -= N_psi_xi[a] * K_m * qw

            ###############################
            # delta kappa part (Euler term)
            ###############################
            for a in range(self.nnodes_element_psi):
                fe[self.nodalDOF_element_psi[a]] += N_psi[a] * cross3(K_Kappa, K_m) * J * qw

        return fe

    # Interpolation of nodal rotations
    def f_pot_q_el(self, qe, el):
        # return approx_fprime(qe, lambda qe: self.f_pot_el(qe, el), method="2-point")

        Ke = np.zeros((self.nq_element, self.nq_element))

        # extract generalized coordinates for beam centerline and directors
        # in the current and reference configuration
        qe_r = qe[self.rDOF]
        qe_psi = qe[self.psiDOF]

        # Extract nodal values for centerline and rotation vector. Further the
        # corresponding nodal rotation is evaluated using Rodriguez' formular.
        qe_r_nodes = qe_r.reshape(self.nnodes_element_psi, -1, order="F")
        qe_psi_nodes = qe_psi.reshape(self.nnodes_element_psi, -1, order="F")
        qe_R_nodes = np.array([rodriguez(psi) for psi in qe_psi_nodes])
        qe_R_psi_nodes = np.array([rodriguez_der(psi) for psi in qe_psi_nodes])

        for i in range(self.nquadrature):
            # extract some already known variables
            N_r_xi = self.N_r_xi[el, i]
            N_psi = self.N_psi[el, i]
            N_psi_xi = self.N_psi_xi[el, i]
            qw = self.qw[el, i]
            J = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # interpolate tangent vector, transformation matrix and its
            # derivative with respect to parameter space
            r_xi = np.einsum("i,ij", N_r_xi, qe_r_nodes)
            A_IK = np.einsum("i,ijk", N_psi, qe_R_nodes)
            A_IK_xi = np.einsum("i,ijk", N_psi_xi, qe_R_nodes)

            # compute derivatives w.r.t. the arc lenght parameter s
            r_s = r_xi / J
            A_IK_s = A_IK_xi / J

            # # derivative of transformation matrix with respect to nodal 
            # # generalized coordiantes of the axis angle vector
            # A_IK_psi_node = np.zeros((3, 3, 3, self.nnodes_element_psi))
            # A_IK_s_psi_node = np.zeros((3, 3, 3, self.nnodes_element_psi))
            # for a in range(self.nnodes_element_psi):
            #     A_IK_psi_node[:, :, :, a] = N_psi[a] * qe_R_psi_nodes[a]
            #     A_IK_s_psi_node[:, :, :, a] = (N_psi_xi[a] / J) * qe_R_psi_nodes[a]

            # def A_IK_fun(qe_psi):
            #     qe_psi_nodes = qe_psi.reshape(self.nnodes_element_psi, -1, order="F")
            #     qe_R_nodes = np.array([rodriguez(psi) for psi in qe_psi_nodes])
            #     return np.einsum("i,ijk", N_psi, qe_R_nodes)

            # def A_IK_s_fun(qe_psi):
            #     qe_psi_nodes = qe_psi.reshape(self.nnodes_element_psi, -1, order="F")
            #     qe_R_nodes = np.array([rodriguez(psi) for psi in qe_psi_nodes])
            #     return np.einsum("i,ijk", N_psi_xi / J, qe_R_nodes)

            # # TODO: Replace these numerical derivatives!
            # A_IK_qe_psi = approx_fprime(qe_psi, A_IK_fun)
            # A_IK_s_qe_psi = approx_fprime(qe_psi, A_IK_s_fun)

            A_IK_qe_psi = np.array([N_psi[k] * rodriguez_der(psi) for k, psi in enumerate(qe_psi_nodes)]).transpose(1, 2, 3, 0).reshape(3, 3, -1, order="C")
            # diff = A_IK_qe_psi - A_IK_qe_psi2
            # error = np.linalg.norm(diff)
            # print(f"error A_IK_qe_psi: {error}")

            A_IK_s_qe_psi = np.array([(N_psi_xi[k] / J) * rodriguez_der(psi) for k, psi in enumerate(qe_psi_nodes)]).transpose(1, 2, 3, 0).reshape(3, 3, -1, order="C")
            # diff = A_IK_s_qe_psi - A_IK_s_qe_psi2
            # error = np.linalg.norm(diff)
            # print(f"error A_IK_s_qe_psi: {error}")

            # axial and shear strains
            K_Gamma = A_IK.T @ r_s
            # K_Gamma_qe_r = np.einsum(
            #     "ij,k", A_IK.T, N_r_xi / J
            # )
            K_Gamma_qe_r = np.einsum(
                "ij,k", A_IK.T, N_r_xi / J
            ).reshape(3, -1, order="C") # C ordering is correct here!
            # K_Gamma_qe_psi = np.einsum(
            #     "ijkm,i", A_IK_psi_node, r_s
            # )
            K_Gamma_qe_psi = np.einsum(
                "ijk,i", A_IK_qe_psi, r_s
            )

            # torsional and flexural strains
            K_Kappa = skew2ax(A_IK.T @ A_IK_s)
            # K_Kappa_qe_psi = np.einsum(
            #     "ijk,jklm",
            #     skew2ax_A(),
            #     np.einsum("ij,iklm", A_IK, A_IK_s_psi_node)
            #     + np.einsum("ijkm,il", A_IK_psi_node, A_IK_s),
            # )
            # K_Kappa_qe_psi = np.einsum(
            #     "ijk,jklm",
            #     skew2ax_A(),
            #     np.einsum("ij,iklm", A_IK, A_IK_s_psi_node)
            #     + np.einsum("ijlm,ik", A_IK_psi_node, A_IK_s),
            # )
            K_Kappa_qe_psi = np.einsum(
                "ijk,jkl",
                skew2ax_A(),
                np.einsum("ij,ikl", A_IK, A_IK_s_qe_psi)
                + np.einsum("ijl,ik", A_IK_qe_psi, A_IK_s),
            )

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
            K_n_qe_r = K_n_K_Gamma @ K_Gamma_qe_r
            K_n_qe_psi = K_n_K_Gamma @ K_Gamma_qe_psi + K_n_K_Kappa @ K_Kappa_qe_psi
            K_m_qe_r = K_m_K_Gamma @ K_Gamma_qe_r
            K_m_qe_psi = K_m_K_Gamma @ K_Gamma_qe_psi + K_m_K_Kappa @ K_Kappa_qe_psi

            ########################
            # first delta Gamma part
            ########################
            # for a in range(self.nnodes_element_r):
            #     fe[self.nodalDOF_element_r[a]] -= N_r_xi[a] * A_IK @ (K_n * qw)
            for a in range(self.nnodes_element_r):
                Ke[self.nodalDOF_element_r[a][:, None], self.rDOF] -= N_r_xi[a] * A_IK @ K_n_qe_r * qw
                Ke[self.nodalDOF_element_r[a][:, None], self.psiDOF] -= N_r_xi[a] * (
                    A_IK @ K_n_qe_psi + np.einsum("ijk,j", A_IK_qe_psi, K_n)
                ) * qw
                # for b in range(self.nnodes_element_r):
                #     Ke[self.nodalDOF_element_r[a][:, None], self.nodalDOF_element_r[b]] -= N_r_xi[a] * A_IK @ K_n_qe_r[:, :, b] * qw
                # for b in range(self.nnodes_element_psi):
                #     Ke[self.nodalDOF_element_r[a][:, None], self.nodalDOF_element_psi[b]] -= N_r_xi[a] * (
                #         A_IK @ K_n_qe_psi[:, :, b] + np.einsum("ijk,j", A_IK_psi_node[:, :, :, b], K_n)
                #     ) * qw

            #########################
            # second delta Gamma part
            #########################
            # for a in range(self.nnodes_element_psi):
            #     fe[self.nodalDOF_element_psi[a]] += N_psi[a] * cross3(A_IK.T @ r_xi, K_n) * qw
            for a in range(self.nnodes_element_psi):
                Ke[self.nodalDOF_element_psi[a][:, None], self.rDOF] += N_psi[a] * (
                    ax2skew(A_IK.T @ r_xi) @ K_n_qe_r
                    - ax2skew(K_n) @ np.einsum(
                        "ij,k", A_IK.T, N_r_xi
                    ).reshape(3, -1, order="C")
                ) * qw

                Ke[self.nodalDOF_element_psi[a][:, None], self.psiDOF] += N_psi[a] * (
                    ax2skew(A_IK.T @ r_xi) @ K_n_qe_psi
                    - ax2skew(K_n) @ np.einsum("ijk,i", A_IK_qe_psi, r_xi)
                ) * qw
                # for b in range(self.nnodes_element_r):
                #     Ke[self.nodalDOF_element_psi[a][:, None], self.nodalDOF_element_r[b]] += N_psi[a] * (
                #         ax2skew(A_IK.T @ r_xi) @ K_n_qe_r[:, :, b]
                #         - ax2skew(K_n) @ A_IK.T * N_r_xi[b]
                #     ) * qw
                # for b in range(self.nnodes_element_psi):
                #     Ke[self.nodalDOF_element_psi[a][:, None], self.nodalDOF_element_psi[b]] += N_psi[a] * (
                #         ax2skew(A_IK.T @ r_xi) @ K_n_qe_psi[:, :, b]
                #         - ax2skew(K_n) @ np.einsum("ijk,i", A_IK_psi_node[:, :, :, b], r_xi)
                #     ) * qw

            ##################
            # delta kappa part
            ##################
            # for a in range(self.nnodes_element_psi):
            #     fe[self.nodalDOF_element_psi[a]] -= N_psi_xi[a] * K_m * qw
            for a in range(self.nnodes_element_psi):
                Ke[self.nodalDOF_element_psi[a][:, None], self.rDOF] -= N_psi_xi[a] * K_m_qe_r * qw
                Ke[self.nodalDOF_element_psi[a][:, None], self.psiDOF] -= N_psi_xi[a] * K_m_qe_psi * qw
                # for b in range(self.nnodes_element_r):
                #     Ke[self.nodalDOF_element_psi[a][:, None], self.nodalDOF_element_r[b]] -= N_psi_xi[a] * K_m_qe_r[:, :, b] * qw
                # for b in range(self.nnodes_element_psi):
                #     Ke[self.nodalDOF_element_psi[a][:, None], self.nodalDOF_element_psi[b]] -= N_psi_xi[a] * K_m_qe_psi[:, :, b] * qw

            ###############################
            # delta kappa part (Euler term)
            ###############################
            # for a in range(self.nnodes_element_psi):
            #     fe[self.nodalDOF_element_psi[a]] += N_psi[a] * cross3(K_Kappa, K_m) * J * qw
            for a in range(self.nnodes_element_psi):
                Ke[self.nodalDOF_element_psi[a][:, None], self.rDOF] += N_psi[a] * (
                    ax2skew(K_Kappa) @ K_m_qe_r
                ) * J * qw
                Ke[self.nodalDOF_element_psi[a][:, None], self.psiDOF] += N_psi[a] * (
                    ax2skew(K_Kappa) @ K_m_qe_psi - ax2skew(K_m) @ K_Kappa_qe_psi
                ) * J * qw
                # for b in range(self.nnodes_element_r):
                #     Ke[self.nodalDOF_element_psi[a][:, None], self.nodalDOF_element_r[b]] += N_psi[a] * (
                #         ax2skew(K_Kappa) @ K_m_qe_r[:, :, b]
                #     ) * qw
                # for b in range(self.nnodes_element_psi):
                #     Ke[self.nodalDOF_element_psi[a][:, None], self.nodalDOF_element_psi[b]] += N_psi[a] * (
                #         ax2skew(K_Kappa) @ K_m_qe_psi[:, :, b] - ax2skew(K_m) @ K_Kappa_qe_psi[:, :, b]
                #     ) * qw

        return Ke

        Ke_num = approx_fprime(qe, lambda qe: self.f_pot_el(qe, el), method="3-point")
        # Ke_num = approx_fprime(qe, lambda qe: self.f_pot_el_test(qe, el), method="3-point")
        diff = Ke - Ke_num
        error = np.max(np.abs(diff))
        print(f'max error f_pot_q_el: {error}')

        # print(f'diff[self.rDOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.rDOF]))}')
        # print(f'diff[self.rDOF[:, None], self.psiDOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.psiDOF]))}')
        # print(f'diff[self.psiDOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.psiDOF[:, None], self.rDOF]))}')
        # print(f'diff[self.psiDOF[:, None], self.psiDOF]: {np.max(np.abs(diff[self.psiDOF[:, None], self.psiDOF]))}')

        return Ke_num

    #########################################
    # kinematic equation
    #########################################
    # TODO:
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
        # raise NotImplementedError
        # coo.extend_diag(np.ones(self.nq), (self.qDOF, self.uDOF))

        # trivial kinematic equation for centerline
        coo.extend_diag(np.ones(self.nq_r), (self.qDOF[:self.nq_r], self.uDOF[:self.nq_r]))

        # axis angle vector part
        for node in range(self.nnode_psi):
            nodalDOF_psi = self.nodalDOF_psi[node]

            psi = q[nodalDOF_psi]
            coo.extend(
                inverse_tangent_map(psi),
                (
                    self.qDOF[nodalDOF_psi],
                    self.uDOF[nodalDOF_psi]
                ),
                # (
                #     self.qDOF[elDOF[nodalDOF[:3]]],
                #     self.uDOF[elDOF[nodalDOF[:3]]],
                # ),
            )

    def q_ddot(self, t, q, u, u_dot):
        raise NotImplementedError
        return u_dot

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
        N, _ = self.basis_functions_r(frame_ID[0])
        NN = self.stack3r(N)
        return NN @ q[self.rDOF] + self.A_IK(t, q, frame_ID=frame_ID) @ K_r_SP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        N, _ = self.basis_functions_r(frame_ID[0])
        NN = self.stack3r(N)

        r_OP_q = np.zeros((3, self.nq_element))
        r_OP_q[:, self.rDOF] = NN
        r_OP_q += np.einsum("ijk,j->ik", self.A_IK_q(t, q, frame_ID=frame_ID), K_r_SP)
        return r_OP_q

    def A_IK(self, t, q, frame_ID):
        N, _ = self.basis_functions_psi(frame_ID[0])
        NN = self.stack3psi(N)

        # Compute nodal directors and interpoalte them using the given shape
        # functions.
        qe_d1, qe_d2, qe_d3 = self.qe_psi2qe_di(q[self.psiDOF])
        d1 = NN @ qe_d1
        d2 = NN @ qe_d2
        d3 = NN @ qe_d3
        return np.vstack((d1, d2, d3)).T

    # TODO:
    def A_IK_q(self, t, q, frame_ID):
        return approx_fprime(q, lambda q: self.A_IK(t, q, frame_ID))
        N, _ = self.basis_functions_psi(frame_ID[0])
        NN = self.stack3psi(N)

        A_IK_q = np.zeros((3, 3, self.nq_element))
        A_IK_q[:, 0, self.psiDOF] = NN
        A_IK_q[:, 1, self.d2DOF] = NN
        A_IK_q[:, 2, self.d3DOF] = NN
        return A_IK_q

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3)):
        N, _ = self.basis_functions_r(frame_ID[0])
        NN = self.stack3r(N)

        v_P = NN @ u[self.rDOF] + self.A_IK(t, q, frame_ID) @ cross3(
            self.K_Omega(t, q, u, frame_ID=frame_ID), K_r_SP
        )
        return v_P

    # TODO:
    def v_P_q(self, t, q, u, frame_ID, K_r_SP=np.zeros(3)):
        raise NotImplementedError("")

    # TODO:
    def J_P(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        return approx_fprime(
            np.zeros(self.nq_element), lambda u: self.v_P(t, q, u, frame_ID, K_r_SP)
        )

    # TODO:
    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return approx_fprime(q, lambda q: self.J_P(t, q, frame_ID, K_r_SP))

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=np.zeros(3)):
        N, _ = self.basis_functions_r(frame_ID[0])
        NN = self.stack3r(N)

        K_Omega = self.K_Omega(t, q, u, frame_ID=frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID=frame_ID)
        return NN @ u_dot[self.rDOF] + self.A_IK(t, q, frame_ID=frame_ID) @ (
            cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP))
        )

    # TODO:
    def a_P_q(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        raise NotImplementedError("")
        return np.zeros((3, self.nq_element))

    # TODO:
    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        raise NotImplementedError("")
        return np.zeros((3, self.nq_element))

    def K_Omega(self, t, q, u, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the angular
        velocity.
        """
        N, _ = self.basis_functions_psi(frame_ID[0])
        NN = self.stack3psi(N)
        return NN @ u[self.psiDOF]

    def K_J_R(self, t, q, frame_ID):
        # return approx_fprime(
        #     np.zeros(self.nq_element), lambda u: self.K_Omega(t, q, u, frame_ID)
        # )
        N, _ = self.basis_functions_psi(frame_ID[0])
        NN = self.stack3psi(N)
        K_J_R = np.zeros((3, self.nq_element))
        K_J_R[:, self.psiDOF] = NN
        return K_J_R

    def K_J_R_q(self, t, q, frame_ID):
        return np.zeros((3, self.nq_element, self.nq_element))

    def K_Psi(self, t, q, u, u_dot, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the time
        derivative of the angular velocity.
        """
        N, _ = self.basis_functions_psi(frame_ID[0])
        NN = self.stack3psi(N)
        return NN @ u_dot[self.psiDOF]

    ####################################################
    # body force
    ####################################################
    def distributed_force1D_pot_el(self, force, t, qe, el):
        Ve = 0
        for i in range(self.nquadrature):
            NNi = self.stack3r(self.N_r[el, i])
            r = NNi @ qe[self.rDOF]
            Ve += r @ force(t, self.qp[el, i]) * self.J[el, i] * self.qw[el, i]
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
            NNi = self.stack3r(self.N_r[el, i])
            fe[self.rDOF] += (
                NNi.T @ force(t, self.qp[el, i]) * self.J[el, i] * self.qw[el, i]
            )
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
        return q[self.qDOF][: 3 * self.nnode_r].reshape(3, -1)

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
        Gamma = np.zeros((self.mesh_r.nelement, self.mesh_r.n_quadrature_points, 3))
        Kappa = np.zeros((self.mesh_r.nelement, self.mesh_r.n_quadrature_points, 3))
        if not same_shape_functions:
            d1s = np.zeros((self.mesh_r.nelement, self.mesh_r.n_quadrature_points, 3))
            d2s = np.zeros((self.mesh_r.nelement, self.mesh_r.n_quadrature_points, 3))
            d3s = np.zeros((self.mesh_r.nelement, self.mesh_r.n_quadrature_points, 3))
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
        Gamma = np.zeros((self.mesh_r.nelement, self.mesh_r.n_quadrature_points, 3))
        Kappa = np.zeros((self.mesh_r.nelement, self.mesh_r.n_quadrature_points, 3))
        if not same_shape_functions:
            d1s = np.zeros((self.mesh_r.nelement, self.mesh_r.n_quadrature_points, 3))
            d2s = np.zeros((self.mesh_r.nelement, self.mesh_r.n_quadrature_points, 3))
            d3s = np.zeros((self.mesh_r.nelement, self.mesh_r.n_quadrature_points, 3))
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
