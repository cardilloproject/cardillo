import numpy as np

from cardillo.utility.coo import Coo
from cardillo.discretization.lagrange import LagrangeKnotVector
from cardillo.discretization.mesh1D import Mesh1D
from cardillo.math import (
    pi,
    norm,
    cross3,
    ax2skew,
    approx_fprime,
    Log_SO3,
    SE3,
    SE3inv,
    Exp_SO3,
    Exp_SE3,
    Log_SE3,
    Exp_SO3_psi,
    Exp_SE3_h,
    Log_SE3_H,
)
from cardillo.beams._base import RodExportBase
from cardillo.beams._base import TimoshenkoPetrovGalerkinBase


class TimoshenkoAxisAngleSE3_K_delta_r_P:
    # class TimoshenkoAxisAngleSE3:
    def __init__(
        self,
        polynomial_degree,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelement,
        Q,
        q0=None,
        u0=None,
    ):
        # beam properties
        self.materialModel = material_model  # material model
        self.A_rho0 = A_rho0  # line density
        self.K_S_rho0 = K_S_rho0  # first moment of area
        self.K_I_rho0 = K_I_rho0  # second moment of area

        if np.allclose(K_S_rho0, np.zeros_like(K_S_rho0)):
            self.constant_mass_matrix = True
        else:
            self.constant_mass_matrix = False

        # material model
        self.material_model = material_model

        if polynomial_degree == 1:
            self.eval = self.__eval_two_node
            self.d_eval = self.__d_eval_two_node
        else:
            self.eval = self.__eval_generic
            self.d_eval = self.__d_eval_generic

        # discretization parameters
        self.polynomial_degree = polynomial_degree
        self.nquadrature = nquadrature = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
        self.nelement = nelement  # number of elements

        self.knot_vector = LagrangeKnotVector(self.polynomial_degree, nelement)

        # build mesh object
        self.mesh = Mesh1D(
            self.knot_vector,
            nquadrature,
            derivative_order=1,
            basis="Lagrange",
            dim_q=3,
        )

        # total number of nodes
        self.nnode = self.mesh.nnodes

        # number of nodes per element
        self.nnodes_element = self.mesh.nnodes_per_element

        # total number of generalized coordinates and velocities
        self.nq_r = self.nu_r = self.mesh.nq
        self.nq_psi = self.nu_psi = self.mesh.nq
        self.nq = self.nu = self.nq_r + self.nq_psi

        # number of generalized coordiantes and velocities per element
        self.nq_element_r = self.nu_element_r = self.mesh.nq_per_element
        self.nq_element_psi = self.nu_element_psi = self.mesh.nq_per_element
        self.nq_element = self.nu_element = self.nq_element_r + self.nq_element_psi

        # global element connectivity
        self.elDOF_r = self.mesh.elDOF
        self.elDOF_psi = self.mesh.elDOF + self.nq_r

        # global nodal
        self.nodalDOF_r = self.mesh.nodalDOF
        self.nodalDOF_psi = self.mesh.nodalDOF + self.nq_r

        # nodal connectivity on element level
        self.nodalDOF_element_r = self.mesh.nodalDOF_element
        self.nodalDOF_element_psi = self.mesh.nodalDOF_element + self.nq_element_r

        # TODO: Check if this is valid!
        self.nodalDOF_element = np.zeros((self.nnodes_element, 6), dtype=int)
        for node in range(self.nnodes_element):
            self.nodalDOF_element[node, :3] = self.nodalDOF_element_r[node]
            self.nodalDOF_element[node, 3:] = self.nodalDOF_element_psi[node]

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nelement, self.nq_element), dtype=int)
        for el in range(nelement):
            self.elDOF[el, : self.nq_element_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_element_r :] = self.elDOF_psi[el]

        # shape functions and their first derivatives
        self.N = self.mesh.N
        self.N_xi = self.mesh.N_xi

        # quadrature points
        self.qp = self.mesh.qp  # quadrature points
        self.qw = self.mesh.wp  # quadrature weights

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions = self.mesh.eval_basis

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q  # reference configuration
        self.q0 = Q.copy() if q0 is None else q0  # initial configuration
        self.u0 = (
            np.zeros(self.nu, dtype=float) if u0 is None else u0
        )  # initial velocities

        # precompute values of the reference configuration in order to save computation time
        # J in Harsch2020b (5)
        self.J = np.zeros((nelement, nquadrature), dtype=float)
        # dilatation and shear strains of the reference configuration
        self.K_Gamma0 = np.zeros((nelement, nquadrature, 3), dtype=float)
        # curvature of the reference configuration
        self.K_Kappa0 = np.zeros((nelement, nquadrature, 3), dtype=float)

        for el in range(nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(nquadrature):
                # evaluate strain measures
                _, _, K_Gamma_bar, K_Kappa_bar = self.eval(qe, self.qp[el, i])

                # length of reference tangential vector
                Ji = norm(K_Gamma_bar)

                # axial and shear strains
                K_Gamma = K_Gamma_bar / Ji

                # torsional and flexural strains
                K_Kappa = K_Kappa_bar / Ji

                # safe precomputed quantities for later
                self.J[el, i] = Ji
                self.K_Gamma0[el, i] = K_Gamma
                self.K_Kappa0[el, i] = K_Kappa

    @staticmethod
    def straight_configuration(
        polynomial_degree,
        nelement,
        L,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
    ):
        nn_r = polynomial_degree * nelement + 1
        nn_psi = polynomial_degree * nelement + 1

        x0 = np.linspace(0, L, num=nn_r)
        y0 = np.zeros(nn_r, dtype=float)
        z0 = np.zeros(nn_r, dtype=float)

        r0 = np.vstack((x0, y0, z0))
        for i in range(nn_r):
            r0[:, i] = r_OP + A_IK @ r0[:, i]

        # reshape generalized coordinates to nodal ordering
        q_r = r0.reshape(-1, order="F")

        # we have to extract the rotation vector from the given rotation matrix
        # and set its value for each node
        psi = Log_SO3(A_IK)
        q_psi = np.tile(psi, nn_psi)

        return np.concatenate([q_r, q_psi])

    @staticmethod
    def initial_configuration(
        polynomial_degree,
        nelement,
        L,
        r_OP0=np.zeros(3, dtype=float),
        A_IK0=np.eye(3, dtype=float),
        v_P0=np.zeros(3, dtype=float),
        K_omega_IK0=np.zeros(3, dtype=float),
    ):
        # nn_r = nelement + 1
        # nn_psi = nelement + 1
        nn_r = polynomial_degree * nelement + 1
        nn_psi = polynomial_degree * nelement + 1

        x0 = np.linspace(0, L, num=nn_r)
        y0 = np.zeros(nn_r, dtype=float)
        z0 = np.zeros(nn_r, dtype=float)

        r_OC0 = np.vstack((x0, y0, z0))
        for i in range(nn_r):
            r_OC0[:, i] = r_OP0 + A_IK0 @ r_OC0[:, i]

        # reshape generalized coordinates to nodal ordering
        q_r = r_OC0.reshape(-1, order="F")

        # we have to extract the rotation vector from the given rotation matrix
        # and set its value for each node
        psi = Log_SO3(A_IK0)

        # centerline velocities
        v_C0 = np.zeros_like(r_OC0, dtype=float)
        for i in range(nn_r):
            v_C0[:, i] = v_P0 + cross3(A_IK0 @ K_omega_IK0, (r_OC0[:, i] - r_OC0[:, 0]))

        # reshape generalized coordinates to nodal ordering
        q_r = r_OC0.reshape(-1, order="F")
        u_r = v_C0.reshape(-1, order="F")
        q_psi = np.tile(psi, nn_psi)
        u_psi = np.tile(K_omega_IK0, nn_psi)

        return np.concatenate([q_r, q_psi]), np.concatenate([u_r, u_psi])

    @staticmethod
    def circular_segment_configuration(
        polynomial_degree,
        nelement,
        radius,
        max_angle,
        r_OP0=np.zeros(3, dtype=float),
        A_IK0=np.eye(3, dtype=float),
    ):
        nn = polynomial_degree * nelement + 1

        # rotation of reference frame
        e_x0, e_y0, e_z0 = A_IK0.T

        r_OPs = np.zeros((3, nn), dtype=float)
        psis = np.zeros((3, nn), dtype=float)
        for i in range(nn):
            xi = i / (nn - 1)
            phi_i = max_angle * xi
            sph = np.sin(phi_i)
            cph = np.cos(phi_i)

            # centerline position
            r_OP = r_OP0 + radius * cph * e_x0 + radius * sph * e_y0

            # compute orientation
            e_x = -sph * e_x0 + cph * e_y0
            e_y = -cph * e_x0 - sph * e_y0
            e_z = e_z0
            A_IK = np.vstack((e_x, e_y, e_z)).T

            # compute SE(3) object
            H_IK = SE3(A_IK, r_OP)
            h_IK = Log_SE3(H_IK)

            r_OPs[:, i] = h_IK[:3]
            psis[:, i] = h_IK[3:]

            # r_OPs[:, i] = r_OP
            # psis[:, i] = Log_SO3(A_IK)

        # reshape generalized coordinates to nodal ordering
        q_r = r_OPs.reshape(-1, order="F")
        q_psi = psis.reshape(-1, order="F")

        return np.concatenate([q_r, q_psi])

    def element_number(self, xi):
        return self.knot_vector.element_number(xi)[0]

    def __eval_two_node(self, qe, xi):
        # extract nodal screws
        nodalDOF0 = np.concatenate(
            (self.nodalDOF_element_r[0], self.nodalDOF_element_psi[0])
        )
        nodalDOF1 = np.concatenate(
            (self.nodalDOF_element_r[1], self.nodalDOF_element_psi[1])
        )
        h0 = qe[nodalDOF0]
        r_OP0 = h0[:3]
        psi0 = h0[3:]
        h1 = qe[nodalDOF1]
        r_OP1 = h1[:3]
        psi1 = h1[3:]

        # nodal transformations
        H_IK0 = SE3(Exp_SO3(psi0), r_OP0)
        H_IK1 = SE3(Exp_SO3(psi1), r_OP1)

        # inverse transformation of first node
        H_IK0_inv = SE3inv(H_IK0)

        # compute relative transformation
        H_K0K1 = H_IK0_inv @ H_IK1

        # compute relative screw
        h_K0K1 = Log_SE3(H_K0K1)

        # find element number containing xi
        el = self.element_number(xi)

        # get element interval
        xi0, xi1 = self.knot_vector.element_interval(el)

        # second linear Lagrange shape function
        N1_xi = 1.0 / (xi1 - xi0)
        N1 = (xi - xi0) * N1_xi

        N, N_xi = self.basis_functions(xi)
        N1 = N[1]
        N1_xi = N_xi[1]

        # relative interpolation of local se(3) objects
        h_local = N1 * h_K0K1
        h_local_xi = N1_xi * h_K0K1

        # composition of reference and local transformation
        H_local = Exp_SE3(h_local)
        H_IK = H_IK0 @ H_local

        # extract centerline and transformation matrix
        A_IK = H_IK[:3, :3]
        r_OP = H_IK[:3, 3]

        # extract strains
        K_Gamma_bar = h_local_xi[:3]
        K_Kappa_bar = h_local_xi[3:]

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def __d_eval_two_node(self, qe, xi):
        # extract nodal screws
        nodalDOF0 = np.concatenate(
            (self.nodalDOF_element_r[0], self.nodalDOF_element_psi[0])
        )
        nodalDOF1 = np.concatenate(
            (self.nodalDOF_element_r[1], self.nodalDOF_element_psi[1])
        )
        h0 = qe[nodalDOF0]
        r_OP0 = h0[:3]
        psi0 = h0[3:]
        h1 = qe[nodalDOF1]
        r_OP1 = h1[:3]
        psi1 = h1[3:]

        # nodal transformations
        A_IK0 = Exp_SO3(psi0)
        A_IK1 = Exp_SO3(psi1)
        H_IK0 = SE3(A_IK0, r_OP0)
        H_IK1 = SE3(A_IK1, r_OP1)
        A_IK0_psi0 = Exp_SO3_psi(psi0)
        A_IK1_psi1 = Exp_SO3_psi(psi1)

        H_IK0_h0 = np.zeros((4, 4, 6), dtype=float)
        H_IK0_h0[:3, :3, 3:] = A_IK0_psi0
        H_IK0_h0[:3, 3, :3] = np.eye(3, dtype=float)
        H_IK1_h1 = np.zeros((4, 4, 6), dtype=float)
        H_IK1_h1[:3, :3, 3:] = A_IK1_psi1
        H_IK1_h1[:3, 3, :3] = np.eye(3, dtype=float)

        # inverse transformation of first node
        H_IK0_inv = SE3inv(H_IK0)
        H_IK0_inv_h0 = np.zeros((4, 4, 6), dtype=float)
        H_IK0_inv_h0[:3, :3, 3:] = A_IK0_psi0.transpose(1, 0, 2)
        H_IK0_inv_h0[:3, 3, 3:] = -np.einsum("k,kij->ij", r_OP0, A_IK0_psi0)
        H_IK0_inv_h0[:3, 3, :3] = -A_IK0.T

        # compute relative transformation
        H_K0K1 = H_IK0_inv @ H_IK1
        H_K0K1_h0 = np.einsum("ilk,lj->ijk", H_IK0_inv_h0, H_IK1)
        H_K0K1_h1 = np.einsum("il,ljk->ijk", H_IK0_inv, H_IK1_h1)

        # compute relative screw
        h_K0K1 = Log_SE3(H_K0K1)
        h_K0K1_HK0K1 = Log_SE3_H(H_K0K1)
        h_K0K1_h0 = np.einsum("ikl,klj->ij", h_K0K1_HK0K1, H_K0K1_h0)
        h_K0K1_h1 = np.einsum("ikl,klj->ij", h_K0K1_HK0K1, H_K0K1_h1)

        # find element number containing xi
        el = self.element_number(xi)

        # get element interval
        xi0, xi1 = self.knot_vector.element_interval(el)

        # second linear Lagrange shape function
        N1_xi = 1.0 / (xi1 - xi0)
        N1 = (xi - xi0) * N1_xi

        # relative interpolation of local se(3) objects
        h_local = N1 * h_K0K1
        h_local_xi = N1_xi * h_K0K1
        h_local_h0 = N1 * h_K0K1_h0
        h_local_h1 = N1 * h_K0K1_h1
        h_local_xi_h0 = N1_xi * h_K0K1_h0
        h_local_xi_h1 = N1_xi * h_K0K1_h1

        # composition of reference and local transformation
        H_local = Exp_SE3(h_local)
        H_local_h = Exp_SE3_h(h_local)
        H_local_h0 = np.einsum("ijl,lk->ijk", H_local_h, h_local_h0)
        H_local_h1 = np.einsum("ijl,lk->ijk", H_local_h, h_local_h1)
        H_IK = H_IK0 @ H_local
        H_IK_h0 = np.einsum("ilk,lj", H_IK0_h0, H_local) + np.einsum(
            "il,ljk->ijk", H_IK0, H_local_h0
        )
        H_IK_h1 = np.einsum("il,ljk->ijk", H_IK0, H_local_h1)

        # extract centerline and transformation matrix
        A_IK = H_IK[:3, :3]
        r_OP = H_IK[:3, 3]
        A_IK_qe = np.zeros((3, 3, self.nq_element), dtype=float)
        A_IK_qe[:, :, nodalDOF0] = H_IK_h0[:3, :3]
        A_IK_qe[:, :, nodalDOF1] = H_IK_h1[:3, :3]
        r_OP_qe = np.zeros((3, self.nq_element), dtype=float)
        r_OP_qe[:, nodalDOF0] = H_IK_h0[:3, 3]
        r_OP_qe[:, nodalDOF1] = H_IK_h1[:3, 3]

        # extract strains
        K_Gamma_bar = h_local_xi[:3]
        K_Kappa_bar = h_local_xi[3:]
        K_Gamma_bar_qe = np.zeros((3, self.nq_element), dtype=float)
        K_Gamma_bar_qe[:, nodalDOF0] = h_local_xi_h0[:3]
        K_Gamma_bar_qe[:, nodalDOF1] = h_local_xi_h1[:3]
        K_Kappa_bar_qe = np.zeros((3, self.nq_element), dtype=float)
        K_Kappa_bar_qe[:, nodalDOF0] = h_local_xi_h0[3:]
        K_Kappa_bar_qe[:, nodalDOF1] = h_local_xi_h1[3:]

        return (
            r_OP,
            A_IK,
            K_Gamma_bar,
            K_Kappa_bar,
            r_OP_qe,
            A_IK_qe,
            K_Gamma_bar_qe,
            K_Kappa_bar_qe,
        )

    def __eval_generic(self, qe, xi):
        # extract nodal positions and rotation vectors of first node (reference)
        r_OP0 = qe[self.nodalDOF_element_r[0]]
        psi0 = qe[self.nodalDOF_element_psi[0]]

        # evaluate nodal rotation matrix
        A_IK0 = Exp_SO3(psi0)

        # evaluate inverse reference SE(3) object
        H_IR = SE3(A_IK0, r_OP0)
        H_IR_inv = SE3inv(H_IR)

        # evaluate shape functions
        N, N_xi = self.basis_functions(xi)

        # relative interpolation of local se(3) objects
        h_rel = np.zeros(6, dtype=qe.dtype)
        h_rel_xi = np.zeros(6, dtype=qe.dtype)

        for node in range(self.nnodes_element):
            # nodal centerline
            r_IK_node = qe[self.nodalDOF_element_r[node]]

            # nodal rotation
            A_IK_node = Exp_SO3(qe[self.nodalDOF_element_psi[node]])

            # nodal SE(3) object
            H_IK_node = SE3(A_IK_node, r_IK_node)

            # relative SE(3)/ se(3) objects
            H_RK = H_IR_inv @ H_IK_node
            h_RK = Log_SE3(H_RK)

            # add wheighted contribution of local se(3) object
            h_rel += N[node] * h_RK
            h_rel_xi += N_xi[node] * h_RK

        # composition of reference and local SE(3) objects
        H_IK = H_IR @ Exp_SE3(h_rel)

        # extract centerline and transformation matrix
        A_IK = H_IK[:3, :3]
        r_OP = H_IK[:3, 3]

        # strain measures
        strains = T_SE3(h_rel) @ h_rel_xi

        # extract strains
        K_Gamma_bar = strains[:3]
        K_Kappa_bar = strains[3:]

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def __d_eval_generic(self, qe, xi):
        raise NotImplementedError
        nodalDOF_element = lambda node: np.concatenate(
            (self.nodalDOF_element_r[node], self.nodalDOF_element_psi[node])
        )

        # compute nodal rotations
        H_IKs = np.array(
            [
                SE3(
                    Exp_SO3(qe[self.nodalDOF_element_psi[node]]),
                    qe[self.nodalDOF_element_r[node]],
                )
                for node in range(self.nnodes_element)
            ]
        )
        H_IK_hs = np.array(
            [
                Exp_SE3_h(qe[nodalDOF_element(node)])
                for node in range(self.nnodes_element)
            ]
        )

        # # compute inverse of derivative of SE(3) derivative of first node
        # # TODO: Check this using a numerical derivative!
        # H_IK0_inv_h = np.zeros((4, 4, 6), dtype=float)
        # H_IK0_inv_h[:3, :3] = H_IK_hs[0, :3, :3].transpose(1, 0, 2)
        # H_IK0_inv_h[:3, 3] = -np.einsum(
        #     "jik,j->ik",
        #     H_IK_hs[0, :3, :3],
        #     H_IKs[0, :3, 3]
        # ) - np.einsum(
        #     "ji,jk->ik",
        #     H_IKs[0, :3, :3],
        #     H_IK_hs[0, :3, 3]
        # )
        # diff = H_IK0_inv_h - Exp_SE3_inv_h(qe[nodalDOF_element(0)])
        # error = np.linalg.norm(diff)
        # print(f"error H_IK0_inv_h: {error}")
        H_IK0_inv_h = Exp_SE3_inv_h(qe[nodalDOF_element(0)])

        # compute relative rotations and the corresponding rotation vectors
        H_IK0_inv = SE3inv(H_IKs[0])
        H_K0Ki = np.array(
            [H_IK0_inv @ H_IKs[node] for node in range(self.nnodes_element)]
        )
        h_K0Ki = np.array(
            [Log_SE3(H_K0Ki[node]) for node in range(self.nnodes_element)]
        )

        # evaluate shape functions
        N, N_xi = self.basis_functions(xi)

        # relative interpolation of local rotation vector
        h_K0K = np.sum(
            [N[node] * h_K0Ki[node] for node in range(self.nnodes_element)], axis=0
        )
        h_K0K_xi = np.sum(
            [N_xi[node] * h_K0Ki[node] for node in range(self.nnodes_element)], axis=0
        )

        # evaluate rotation and its derivative at interpolated position
        H_K0K = Exp_SE3(h_K0K)
        H_K0K_h = Exp_SE3_h(h_K0K)

        H_IK_q = np.zeros((4, 4, self.nq_element), dtype=float)

        # first node contribution part I
        H_IK_q[:, :, nodalDOF_element(0)] = np.einsum("ilk,lj->ijk", H_IK_hs[0], H_K0K)

        Tmp1 = np.einsum("il,ljm->ijm", H_IKs[0], H_K0K_h)

        for node in range(self.nnodes_element):
            Tmp2 = np.einsum("ijm,mno->ijno", Tmp1, N[node] * Log_SE3_H(H_K0Ki[node]))

            H_IK_q[:, :, nodalDOF_element(0)] += np.einsum(
                "ijno,npk,po",
                Tmp2,
                H_IK0_inv_h,
                H_IKs[node],
            )
            H_IK_q[:, :, nodalDOF_element(node)] += np.einsum(
                "ijno,np,pok", Tmp2, SE3inv(H_IKs[0]), H_IK_hs[node]
            )

        # extract centerline and transformation matrix
        A_IK_q = H_IK_q[:3, :3]
        r_OP_q = H_IK_q[:3, 3]

        return r_OP_q, A_IK_q

    def assembler_callback(self):
        if self.constant_mass_matrix:
            self.__M_coo()

    #########################################
    # equations of motion
    #########################################
    def M_el_constant(self, el):
        M_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # delta_r A_rho0 r_ddot part
            M_el_r_r = np.eye(3) * self.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_psi_psi = self.K_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )

        return M_el

    def M_el(self, qe, el):
        M_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # delta_r A_rho0 r_ddot part
            M_el_r_r = np.eye(3) * self.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_psi_psi = self.K_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )

            # For non symmetric cross sections there are also other parts
            # involved in the mass matrix. These parts are configuration
            # dependent and lead to configuration dependent mass matrix.
            _, A_IK, _, _ = self.eval(qe, self.qp[el, i])
            M_el_r_psi = A_IK @ self.K_S_rho0 * Ji * qwi
            M_el_psi_r = A_IK @ self.K_S_rho0 * Ji * qwi

            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_psi * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_r * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )

        return M_el

    def __M_coo(self):
        self.__M = Coo((self.nu, self.nu))
        for el in range(self.nelement):
            # extract element degrees of freedom
            elDOF = self.elDOF[el]

            # sparse assemble element mass matrix
            self.__M.extend(
                self.M_el_constant(el), (self.uDOF[elDOF], self.uDOF[elDOF])
            )

    # TODO: Compute derivative of mass matrix for non constant mass matrix case!
    def M(self, t, q, coo):
        raise NotImplementedError
        if self.constant_mass_matrix:
            coo.extend_sparse(self.__M)
        else:
            for el in range(self.nelement):
                # extract element degrees of freedom
                elDOF = self.elDOF[el]

                # sparse assemble element mass matrix
                coo.extend(
                    self.M_el(q[elDOF], el), (self.uDOF[elDOF], self.uDOF[elDOF])
                )

    # TODO:
    def f_gyr_el(self, t, qe, ue, el):
        f_gyr_el = np.zeros(self.nq_element, dtype=float)

        # for i in range(self.nquadrature):
        #     # interpoalte angular velocity
        #     K_Omega = np.zeros(3, dtype=float)
        #     for node in range(self.nnodes_element):
        #         K_Omega += self.N[el, i, node] * ue[self.nodalDOF_element_psi[node]]

        #     # vector of gyroscopic forces
        #     f_gyr_el_psi = (
        #         cross3(K_Omega, self.K_I_rho0 @ K_Omega)
        #         * self.J[el, i]
        #         * self.qw[el, i]
        #     )

        #     # multiply vector of gyroscopic forces with nodal virtual rotations
        #     for node in range(self.nnodes_element):
        #         f_gyr_el[self.nodalDOF_element_psi[node]] += (
        #             self.N[el, i, node] * f_gyr_el_psi
        #         )

        return f_gyr_el

    def f_gyr(self, t, q, u):
        f_gyr = np.zeros(self.nu, dtype=float)
        for el in range(self.nelement):
            f_gyr[self.elDOF[el]] += self.f_gyr_el(
                t, q[self.elDOF[el]], u[self.elDOF[el]], el
            )
        return f_gyr

    def f_gyr_u_el(self, t, qe, ue, el):
        f_gyr_u_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # interpoalte angular velocity
            K_Omega = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element):
                K_Omega += self.N[el, i, node] * ue[self.nodalDOF_element_psi[node]]

            # derivative of vector of gyroscopic forces
            f_gyr_u_el_psi = (
                ((ax2skew(K_Omega) @ self.K_I_rho0 - ax2skew(self.K_I_rho0 @ K_Omega)))
                * self.J[el, i]
                * self.qw[el, i]
            )

            # multiply derivative of gyroscopic force vector with nodal virtual rotations
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    f_gyr_u_el[nodalDOF_a[:, None], nodalDOF_b] += f_gyr_u_el_psi * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
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
        raise NotImplementedError
        E_pot_el = 0

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # objective interpolation
            _, _, K_Gamma_bar, K_Kappa_bar = self.eval(qe, self.qp[el, i])

            # axial and shear strains
            K_Gamma = K_Gamma_bar / Ji

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / Ji

            # evaluate strain energy function
            E_pot_el += (
                self.material_model.potential(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
                * Ji
                * qwi
            )

        return E_pot_el

    def h(self, t, q, u):
        f_pot = np.zeros(self.nu, dtype=q.dtype)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f_pot[elDOF] += self.f_pot_el(q[elDOF], el)
        return f_pot

    def f_pot_el(self, qe, el):
        f_pot_el = np.zeros(self.nq_element, dtype=qe.dtype)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # objective interpolation
            _, A_IK, K_Gamma_bar, K_Kappa_bar = self.eval(qe, self.qp[el, i])

            # axial and shear strains
            K_Gamma = K_Gamma_bar / Ji

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / Ji

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)

            ############################
            # virtual work contributions
            ############################
            for node in range(self.nnodes_element):
                # - first delta Gamma part
                f_pot_el[self.nodalDOF_element_r[node]] -= (
                    self.N_xi[el, i, node] * K_n * qwi
                )

                # - third delta Gamma part
                f_pot_el[self.nodalDOF_element_r[node]] += (
                    self.N[el, i, node] * cross3(K_Kappa_bar, K_n) * qwi
                )

            for node in range(self.nnodes_element):
                # - second delta Gamma part
                f_pot_el[self.nodalDOF_element_psi[node]] += (
                    self.N[el, i, node] * cross3(K_Gamma_bar, K_n) * qwi
                )

                # - first delta kappa part
                f_pot_el[self.nodalDOF_element_psi[node]] -= (
                    self.N_xi[el, i, node] * K_m * qwi
                )

                # second delta kappa part
                f_pot_el[self.nodalDOF_element_psi[node]] += (
                    self.N[el, i, node] * cross3(K_Kappa_bar, K_m) * qwi
                )  # Euler term

        return f_pot_el

    def h_q(self, t, q, u, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f_pot_q_el = self.f_pot_q_el(q[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(f_pot_q_el, (self.uDOF[elDOF], self.qDOF[elDOF]))

    def f_pot_q_el(self, qe, el):
        # f_pot_q_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        # for i in range(self.nquadrature):
        #     # extract reference state variables
        #     qwi = self.qw[el, i]
        #     Ji = self.J[el, i]
        #     K_Gamma0 = self.K_Gamma0[el, i]
        #     K_Kappa0 = self.K_Kappa0[el, i]

        #     # objective interpolation
        #     (
        #         r_OP,
        #         A_IK,
        #         K_Gamma_bar,
        #         K_Kappa_bar,
        #         r_OP_qe,
        #         A_IK_qe,
        #         K_Gamma_bar_qe,
        #         K_Kappa_bar_qe,
        #     ) = self.d_eval(qe, self.qp[el, i])

        #     # axial and shear strains
        #     K_Gamma = K_Gamma_bar / Ji
        #     K_Gamma_qe = K_Gamma_bar_qe / Ji

        #     # torsional and flexural strains
        #     K_Kappa = K_Kappa_bar / Ji
        #     K_Kappa_qe = K_Kappa_bar_qe / Ji

        #     # compute contact forces and couples from partial derivatives of
        #     # the strain energy function w.r.t. strain measures
        #     K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
        #     K_n_K_Gamma = self.material_model.K_n_K_Gamma(
        #         K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
        #     )
        #     K_n_K_Kappa = self.material_model.K_n_K_Kappa(
        #         K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
        #     )
        #     K_n_qe = K_n_K_Gamma @ K_Gamma_qe + K_n_K_Kappa @ K_Kappa_qe

        #     K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
        #     K_m_K_Gamma = self.material_model.K_m_K_Gamma(
        #         K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
        #     )
        #     K_m_K_Kappa = self.material_model.K_m_K_Kappa(
        #         K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
        #     )
        #     K_m_qe = K_m_K_Gamma @ K_Gamma_qe + K_m_K_Kappa @ K_Kappa_qe

        #     ############################
        #     # virtual work contributions
        #     ############################
        #     # - first delta Gamma part
        #     for node in range(self.nnodes_element):
        #         f_pot_q_el[self.nodalDOF_element_r[node], :] -= (
        #             self.N_xi[el, i, node]
        #             * qwi
        #             * (np.einsum("ikj,k->ij", A_IK_qe, K_n) + A_IK @ K_n_qe)
        #         )

        #     for node in range(self.nnodes_element):
        #         # - second delta Gamma part
        #         f_pot_q_el[self.nodalDOF_element_psi[node], :] += (
        #             self.N[el, i, node]
        #             * qwi
        #             * (ax2skew(K_Gamma_bar) @ K_n_qe - ax2skew(K_n) @ K_Gamma_bar_qe)
        #         )

        #         # - first delta kappa part
        #         f_pot_q_el[self.nodalDOF_element_psi[node], :] -= (
        #             self.N_xi[el, i, node] * qwi * K_m_qe
        #         )

        #         # - second delta kappa part
        #         f_pot_q_el[self.nodalDOF_element_psi[node], :] += (
        #             self.N[el, i, node]
        #             * qwi
        #             * (ax2skew(K_Kappa_bar) @ K_m_qe - ax2skew(K_m) @ K_Kappa_bar_qe)
        #         )

        # return f_pot_q_el

        f_pot_q_el_num = approx_fprime(
            qe, lambda qe: self.f_pot_el(qe, el), eps=1.0e-10, method="cs"
        )
        # f_pot_q_el_num = approx_fprime(
        #     qe, lambda qe: self.f_pot_el(qe, el), eps=5.0e-6, method="2-point"
        # )
        # diff = f_pot_q_el - f_pot_q_el_num
        # error = np.linalg.norm(diff)
        # print(f"error f_pot_q_el: {error}")
        return f_pot_q_el_num

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        raise NotImplementedError
        # centerline part
        q_dot = u

        # correct axis angle vector part
        for node in range(self.nnode):
            nodalDOF_psi = self.nodalDOF_psi[node]
            psi = q[nodalDOF_psi]
            K_omega_IK = u[nodalDOF_psi]

            psi_dot = T_SO3_inv(psi) @ K_omega_IK
            q_dot[nodalDOF_psi] = psi_dot

        return q_dot

    def B(self, t, q, coo):
        # trivial kinematic equation for centerline
        coo.extend_diag(
            np.ones(self.nq_r), (self.qDOF[: self.nq_r], self.uDOF[: self.nq_r])
        )

        # axis angle vector part
        for node in range(self.nnode):
            nodalDOF_psi = self.nodalDOF_psi[node]

            psi = q[nodalDOF_psi]
            coo.extend(
                T_SO3_inv(psi),
                (self.qDOF[nodalDOF_psi], self.uDOF[nodalDOF_psi]),
            )

    def q_ddot(self, t, q, u, u_dot):
        raise RuntimeError("Not tested!")
        # centerline part
        q_ddot = u_dot

        # correct axis angle vector part
        for node in range(self.nnode):
            nodalDOF_psi = self.nodalDOF_psi[node]

            psi = q[nodalDOF_psi]
            omega = u[nodalDOF_psi]
            omega_dot = u_dot[nodalDOF_psi]

            T_inv = T_SO3_inv(psi)
            psi_dot = T_inv @ omega

            # TODO:
            T_dot = tangent_map_s(psi, psi_dot)
            Tinv_dot = -T_inv @ T_dot @ T_inv
            psi_ddot = T_inv @ omega_dot + Tinv_dot @ omega

            q_ddot[nodalDOF_psi] = psi_ddot

        return q_ddot

    # change between rotation vector and its complement in order to circumvent
    # singularities of the rotation vector
    @staticmethod
    def psi_C(psi):
        angle = norm(psi)
        if angle < pi:
            return psi
        else:
            # Ibrahimbegovic1995 after (62)
            psi_C = (1.0 - 2.0 * pi / angle) * psi
            return psi_C

    def step_callback(self, t, q, u):
        for node in range(self.nnode):
            psi = q[self.nodalDOF_psi[node]]
            q[self.nodalDOF_psi[node]] = TimoshenkoAxisAngleSE3.psi_C(psi)
        return q, u

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def elDOF_P(self, frame_ID):
        xi = frame_ID[0]
        el = self.element_number(xi)
        return self.elDOF[el]

    def local_qDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def local_uDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    ###################
    # r_OP contribution
    ###################
    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        r, A_IK, _, _ = self.__eval_two_node(q, frame_ID[0])
        return r + A_IK @ K_r_SP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        (
            r_OP,
            A_IK,
            K_Gamma_bar,
            K_Kappa_bar,
            r_q,
            A_IK_q,
            K_Gamma_bar_q,
            K_Kappa_bar_q,
        ) = self.d_eval(q, frame_ID[0])
        r_OP_q = r_q + np.einsum("ijk,j->ik", A_IK_q, K_r_SP)
        return r_OP_q

        # r_OP_q_num = approx_fprime(
        #     q, lambda q: self.r_OP(t, q, frame_ID, K_r_SP), eps=1.0e-10, method="cs"
        # )
        # # diff = r_OP_q - r_OP_q_num
        # # error = np.linalg.norm(diff)
        # # np.set_printoptions(3, suppress=True)
        # # if error > 1.0e-10:
        # #     print(f"r_OP_q:\n{r_OP_q}")
        # #     print(f"r_OP_q_num:\n{r_OP_q_num}")
        # #     print(f"error r_OP_q: {error}")
        # return r_OP_q_num

    def A_IK(self, t, q, frame_ID):
        _, A_IK, _, _ = self.eval(q, frame_ID[0])
        return A_IK

    def A_IK_q(self, t, q, frame_ID):
        (
            r_OP,
            A_IK,
            K_Gamma_bar,
            K_Kappa_bar,
            r_OP_q,
            A_IK_q,
            K_Gamma_bar_q,
            K_Kappa_bar_q,
        ) = self.d_eval(q, frame_ID[0])
        return A_IK_q

        # A_IK_q_num = approx_fprime(
        #     q, lambda q: self.A_IK(t, q, frame_ID), eps=1.0e-10, method="cs"
        # )
        # # diff = A_IK_q - A_IK_q_num
        # # error = np.linalg.norm(diff)
        # # # if error > 1.0e-10:
        # # print(f'error A_IK_q: {error}')
        # return A_IK_q_num

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3), dtype=float):
        N, _ = self.basis_functions(frame_ID[0])
        _, A_IK, _, _ = self.eval(q, frame_ID[0])

        # angular velocity in K-frame
        K_Omega = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element):
            K_Omega += N[node] * u[self.nodalDOF_element_psi[node]]

        # centerline velocity
        K_v_C = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element):
            K_v_C += N[node] * u[self.nodalDOF_element_r[node]]

        return A_IK @ (K_v_C + cross3(K_Omega, K_r_SP))

    # TODO:
    def v_P_q(self, t, q, u, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        raise RuntimeError(
            "Implement this derivative since it requires known parts only!"
        )
        v_P_q_num = approx_fprime(
            q, lambda q: self.v_P(t, q, u, frame_ID, K_r_SP), method="3-point"
        )
        return v_P_q_num

    def J_P(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # evaluate required nodal shape functions
        N, _ = self.basis_functions(frame_ID[0])

        # transformation matrix
        _, A_IK, _, _ = self.eval(q, frame_ID[0])

        # skew symmetric matrix of K_r_SP
        K_r_SP_tilde = ax2skew(K_r_SP)

        # interpolate centerline and axis angle contributions
        J_P = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element):
            J_P[:, self.nodalDOF_element_r[node]] += N[node] * A_IK
        for node in range(self.nnodes_element):
            J_P[:, self.nodalDOF_element_psi[node]] -= N[node] * A_IK @ K_r_SP_tilde

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
        # # evaluate required nodal shape functions
        # N, _ = self.basis_functions(frame_ID[0])

        # K_r_SP_tilde = ax2skew(K_r_SP)
        # A_IK_q = self.A_IK_q(t, q, frame_ID)
        # prod = np.einsum("ijl,jk", A_IK_q, K_r_SP_tilde)

        # # interpolate axis angle contributions since centerline contributon is
        # # zero
        # J_P_q = np.zeros((3, self.nq_element, self.nq_element), dtype=float)
        # for node in range(self.nnodes_element):
        #     nodalDOF_psi = self.nodalDOF_element_psi[node]
        #     J_P_q[:, nodalDOF_psi] -= N[node] * prod

        # return J_P_q

        J_P_q_num = approx_fprime(
            q, lambda q: self.J_P(t, q, frame_ID, K_r_SP), method="3-point"
        )
        # diff = J_P_q_num - J_P_q
        # error = np.linalg.norm(diff)
        # print(f"error J_P_q: {error}")
        return J_P_q_num

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        raise NotImplementedError
        N, _ = self.basis_functions(frame_ID[0])
        _, A_IK, _, _ = self.eval(q, frame_ID[0])

        # centerline acceleration
        a_C = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element):
            a_C += N[node] * u_dot[self.nodalDOF_element_r[node]]

        # angular velocity and acceleration in K-frame
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID)

        # rigid body formular
        return a_C + A_IK @ (
            cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP))
        )

    # TODO:
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
            q, lambda q: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP), method="3-point"
        )
        diff = a_P_q_num - a_P_q
        error = np.linalg.norm(diff)
        print(f"error a_P_q: {error}")
        return a_P_q_num

    # TODO:
    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        raise NotImplementedError
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        local = -self.A_IK(t, q, frame_ID) @ (
            ax2skew(cross3(K_Omega, K_r_SP)) + ax2skew(K_Omega) @ ax2skew(K_r_SP)
        )

        N, _ = self.basis_functions(frame_ID[0])
        a_P_u = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element):
            a_P_u[:, self.nodalDOF_element_psi[node]] += N[node] * local

        # return a_P_u

        a_P_u_num = approx_fprime(
            u, lambda u: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP), method="3-point"
        )
        diff = a_P_u_num - a_P_u
        error = np.linalg.norm(diff)
        print(f"error a_P_u: {error}")
        return a_P_u_num

    def K_Omega(self, t, q, u, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        angular velocities in the K-frame.
        """
        N, _ = self.basis_functions(frame_ID[0])
        K_Omega = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element):
            K_Omega += N[node] * u[self.nodalDOF_element_psi[node]]
        return K_Omega

    def K_Omega_q(self, t, q, u, frame_ID):
        return np.zeros((3, self.nq_element), dtype=float)

    def K_J_R(self, t, q, frame_ID):
        N, _ = self.basis_functions(frame_ID[0])
        K_J_R = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element):
            K_J_R[:, self.nodalDOF_element_psi[node]] += N[node] * np.eye(3)
        return K_J_R

        # K_J_R_num = approx_fprime(
        #     np.zeros(self.nu_element, dtype=float),
        #     lambda u: self.K_Omega(t, q, u, frame_ID),
        #     method="3-point",
        # )
        # diff = K_J_R - K_J_R_num
        # error = np.linalg.norm(diff)
        # print(f"error K_J_R: {error}")
        # return K_J_R_num

    def K_J_R_q(self, t, q, frame_ID):
        return np.zeros((3, self.nq_element, self.nq_element), dtype=float)

    def K_Psi(self, t, q, u, u_dot, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        time derivative of the angular velocities in the K-frame.
        """
        N, _ = self.basis_functions(frame_ID[0])
        K_Psi = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element):
            K_Psi += N[node] * u_dot[self.nodalDOF_element_psi[node]]
        return K_Psi

    def K_Psi_q(self, t, q, u, u_dot, frame_ID):
        return np.zeros((3, self.nq_element), dtype=float)

    def K_Psi_u(self, t, q, u, u_dot, frame_ID):
        return np.zeros((3, self.nq_element), dtype=float)

    ####################################################
    # body force
    ####################################################
    def distributed_force1D_pot_el(self, force, t, qe, el):
        raise NotImplementedError
        Ve = 0
        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # interpolate centerline position
            r_C = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element):
                r_C += self.N[el, i, node] * qe[self.nodalDOF_element_r[node]]

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
        fe = np.zeros(self.nq_element, dtype=float)
        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # compute local force vector
            fe_r = force(t, qwi) * Ji * qwi

            # multiply local force vector with variation of centerline
            for node in range(self.nnodes_element):
                fe[self.nodalDOF_element_r[node]] += self.N[el, i, node] * fe_r

        return fe

    def distributed_force1D(self, t, q, force):
        raise NotImplementedError
        f = np.zeros(self.nq, dtype=float)
        for el in range(self.nelement):
            f[self.elDOF[el]] += self.distributed_force1D_el(force, t, el)
        return f

    def distributed_force1D_q(self, t, q, coo, force):
        raise NotImplementedError
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
            qe = q_body[self.local_qDOF_P(frame_ID)]
            r.append(self.r_OP(1, qe, frame_ID))
        return np.array(r).T

    def frames(self, q, n=10):
        q_body = q[self.qDOF]
        r = []
        d1 = []
        d2 = []
        d3 = []

        for xi in np.linspace(0, 1, n):
            frame_ID = (xi,)
            qp = q_body[self.local_qDOF_P(frame_ID)]
            r.append(self.r_OP(1, qp, frame_ID))

            d1i, d2i, d3i = self.A_IK(1, qp, frame_ID).T
            d1.extend([d1i])
            d2.extend([d2i])
            d3.extend([d3i])

        return np.array(r).T, np.array(d1).T, np.array(d2).T, np.array(d3).T

    def cover(self, q, radius, n_xi=20, n_alpha=100):
        q_body = q[self.qDOF]
        points = []
        for xi in np.linspace(0, 1, num=n_xi):
            frame_ID = (xi,)
            elDOF = self.elDOF_P(frame_ID)
            qe = q_body[elDOF]

            # point on the centerline and tangent vector
            r = self.r_OC(0, qe, (xi,))

            # evaluate directors
            A_IK = self.A_IK(0, qe, frame_ID=(xi,))
            _, d2, d3 = A_IK.T

            # start with point on centerline
            points.append(r)

            # compute points on circular cross section
            x0 = None  # initial point is required twice
            for alpha in np.linspace(0, 2 * np.pi, num=n_alpha):
                x = r + radius * np.cos(alpha) * d2 + radius * np.sin(alpha) * d3
                points.append(x)
                if x0 is None:
                    x0 = x

            # end with first point on cross section
            points.append(x0)

            # end with point on centerline
            points.append(r)

        return np.array(points).T

    def plot_centerline(self, ax, q, n=100, color="black"):
        ax.plot(*self.nodes(q), linestyle="dashed", marker="o", color=color)
        ax.plot(*self.centerline(q, n=n), linestyle="solid", color=color)

    def plot_frames(self, ax, q, n=10, length=1):
        r, d1, d2, d3 = self.frames(q, n=n)
        ax.quiver(*r, *d1, color="red", length=length)
        ax.quiver(*r, *d2, color="green", length=length)
        ax.quiver(*r, *d3, color="blue", length=length)


class TimoshenkoAxisAngleSE3_I_delta_r_P(RodExportBase):
    def __init__(
        self,
        cross_section,
        polynomial_degree,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelement,
        Q,
        q0=None,
        u0=None,
    ):
        super().__init__(cross_section)

        # beam properties
        self.materialModel = material_model  # material model
        self.A_rho0 = A_rho0  # line density
        self.K_S_rho0 = K_S_rho0  # first moment of area
        self.K_I_rho0 = K_I_rho0  # second moment of area

        if np.allclose(K_S_rho0, np.zeros_like(K_S_rho0)):
            self.constant_mass_matrix = True
        else:
            self.constant_mass_matrix = False

        # material model
        self.material_model = material_model

        if polynomial_degree == 1:
            self.eval = self.__eval_two_node
            self.d_eval = self.__d_eval_two_node
        else:
            self.eval = self.__eval_generic
            self.d_eval = self.__d_eval_generic

        # discretization parameters
        self.polynomial_degree = polynomial_degree
        self.nquadrature = nquadrature = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
        self.nelement = nelement  # number of elements

        self.knot_vector = LagrangeKnotVector(self.polynomial_degree, nelement)

        # build mesh object
        self.mesh = Mesh1D(
            self.knot_vector,
            nquadrature,
            derivative_order=1,
            basis="Lagrange",
            dim_q=3,
        )

        # total number of nodes
        self.nnode = self.mesh.nnodes

        # number of nodes per element
        self.nnodes_element = self.mesh.nnodes_per_element

        # total number of generalized coordinates and velocities
        self.nq_r = self.nu_r = self.mesh.nq
        self.nq_psi = self.nu_psi = self.mesh.nq
        self.nq = self.nu = self.nq_r + self.nq_psi

        # number of generalized coordiantes and velocities per element
        self.nq_element_r = self.nu_element_r = self.mesh.nq_per_element
        self.nq_element_psi = self.nu_element_psi = self.mesh.nq_per_element
        self.nq_element = self.nu_element = self.nq_element_r + self.nq_element_psi

        # global element connectivity
        self.elDOF_r = self.mesh.elDOF
        self.elDOF_psi = self.mesh.elDOF + self.nq_r

        # global nodal
        self.nodalDOF_r = self.mesh.nodalDOF
        self.nodalDOF_psi = self.mesh.nodalDOF + self.nq_r

        # nodal connectivity on element level
        self.nodalDOF_element_r = self.mesh.nodalDOF_element
        self.nodalDOF_element_psi = self.mesh.nodalDOF_element + self.nq_element_r

        self.nodalDOF_element = np.zeros((self.nnodes_element, 6), dtype=int)
        for node in range(self.nnodes_element):
            self.nodalDOF_element[node, :3] = self.nodalDOF_element_r[node]
            self.nodalDOF_element[node, 3:] = self.nodalDOF_element_psi[node]

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nelement, self.nq_element), dtype=int)
        for el in range(nelement):
            self.elDOF[el, : self.nq_element_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_element_r :] = self.elDOF_psi[el]

        # shape functions and their first derivatives
        self.N = self.mesh.N
        self.N_xi = self.mesh.N_xi

        # quadrature points
        self.qp = self.mesh.qp  # quadrature points
        self.qw = self.mesh.wp  # quadrature weights

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions = self.mesh.eval_basis

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q  # reference configuration
        self.q0 = Q.copy() if q0 is None else q0  # initial configuration
        self.u0 = (
            np.zeros(self.nu, dtype=float) if u0 is None else u0
        )  # initial velocities

        # precompute values of the reference configuration in order to save computation time
        # J in Harsch2020b (5)
        self.J = np.zeros((nelement, nquadrature), dtype=float)
        # dilatation and shear strains of the reference configuration
        self.K_Gamma0 = np.zeros((nelement, nquadrature, 3), dtype=float)
        # curvature of the reference configuration
        self.K_Kappa0 = np.zeros((nelement, nquadrature, 3), dtype=float)

        for el in range(nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(nquadrature):
                # evaluate strain measures
                _, _, K_Gamma_bar, K_Kappa_bar = self.eval(qe, self.qp[el, i])

                # length of reference tangential vector
                Ji = norm(K_Gamma_bar)

                # axial and shear strains
                K_Gamma = K_Gamma_bar / Ji

                # torsional and flexural strains
                K_Kappa = K_Kappa_bar / Ji

                # safe precomputed quantities for later
                self.J[el, i] = Ji
                self.K_Gamma0[el, i] = K_Gamma
                self.K_Kappa0[el, i] = K_Kappa

    @staticmethod
    def straight_configuration(
        polynomial_degree,
        nelement,
        L,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
    ):
        nn_r = polynomial_degree * nelement + 1
        nn_psi = polynomial_degree * nelement + 1

        x0 = np.linspace(0, L, num=nn_r)
        y0 = np.zeros(nn_r, dtype=float)
        z0 = np.zeros(nn_r, dtype=float)

        r0 = np.vstack((x0, y0, z0))
        for i in range(nn_r):
            r0[:, i] = r_OP + A_IK @ r0[:, i]

        # reshape generalized coordinates to nodal ordering
        q_r = r0.reshape(-1, order="C")

        # we have to extract the rotation vector from the given rotation matrix
        # and set its value for each node
        psi = Log_SO3(A_IK)
        q_psi = np.repeat(psi, nn_psi)

        return np.concatenate([q_r, q_psi])

    @staticmethod
    def initial_configuration(
        polynomial_degree,
        nelement,
        L,
        r_OP0=np.zeros(3, dtype=float),
        A_IK0=np.eye(3, dtype=float),
        v_P0=np.zeros(3, dtype=float),
        K_omega_IK0=np.zeros(3, dtype=float),
    ):
        # nn_r = nelement + 1
        # nn_psi = nelement + 1
        nn_r = polynomial_degree * nelement + 1
        nn_psi = polynomial_degree * nelement + 1

        x0 = np.linspace(0, L, num=nn_r)
        y0 = np.zeros(nn_r, dtype=float)
        z0 = np.zeros(nn_r, dtype=float)

        r_OC0 = np.vstack((x0, y0, z0))
        for i in range(nn_r):
            r_OC0[:, i] = r_OP0 + A_IK0 @ r_OC0[:, i]

        # reshape generalized coordinates to nodal ordering
        q_r = r_OC0.reshape(-1, order="C")

        # we have to extract the rotation vector from the given rotation matrix
        # and set its value for each node
        psi = Log_SO3(A_IK0)

        # centerline velocities
        v_C0 = np.zeros_like(r_OC0, dtype=float)
        for i in range(nn_r):
            v_C0[:, i] = v_P0 + cross3(A_IK0 @ K_omega_IK0, (r_OC0[:, i] - r_OC0[:, 0]))

        # reshape generalized coordinates to nodal ordering
        q_r = r_OC0.reshape(-1, order="C")
        u_r = v_C0.reshape(-1, order="C")
        q_psi = np.repeat(psi, nn_psi)
        u_psi = np.repeat(K_omega_IK0, nn_psi)

        return np.concatenate([q_r, q_psi]), np.concatenate([u_r, u_psi])

    @staticmethod
    def circular_segment_configuration(
        polynomial_degree,
        nelement,
        radius,
        max_angle,
        r_OP0=np.zeros(3, dtype=float),
        A_IK0=np.eye(3, dtype=float),
    ):
        nn = polynomial_degree * nelement + 1

        # rotation of reference frame
        e_x0, e_y0, e_z0 = A_IK0.T

        r_OPs = np.zeros((3, nn), dtype=float)
        psis = np.zeros((3, nn), dtype=float)
        for i in range(nn):
            xi = i / (nn - 1)
            phi_i = max_angle * xi
            sph = np.sin(phi_i)
            cph = np.cos(phi_i)

            # centerline position
            r_OP = r_OP0 + radius * cph * e_x0 + radius * sph * e_y0

            # compute orientation
            e_x = -sph * e_x0 + cph * e_y0
            e_y = -cph * e_x0 - sph * e_y0
            e_z = e_z0
            A_IK = np.vstack((e_x, e_y, e_z)).T

            # compute SE(3) object
            H_IK = SE3(A_IK, r_OP)
            h_IK = Log_SE3(H_IK)

            r_OPs[:, i] = h_IK[:3]
            psis[:, i] = h_IK[3:]

            # r_OPs[:, i] = r_OP
            # psis[:, i] = Log_SO3(A_IK)

        # reshape generalized coordinates to nodal ordering
        q_r = r_OPs.reshape(-1, order="C")
        q_psi = psis.reshape(-1, order="C")

        return np.concatenate([q_r, q_psi])

    def element_number(self, xi):
        return self.knot_vector.element_number(xi)[0]

    def __eval_two_node(self, qe, xi):
        # extract nodal screws
        nodalDOF0 = np.concatenate(
            (self.nodalDOF_element_r[0], self.nodalDOF_element_psi[0])
        )
        nodalDOF1 = np.concatenate(
            (self.nodalDOF_element_r[1], self.nodalDOF_element_psi[1])
        )
        h0 = qe[nodalDOF0]
        r_OP0 = h0[:3]
        psi0 = h0[3:]
        h1 = qe[nodalDOF1]
        r_OP1 = h1[:3]
        psi1 = h1[3:]

        # nodal transformations
        H_IK0 = SE3(Exp_SO3(psi0), r_OP0)
        H_IK1 = SE3(Exp_SO3(psi1), r_OP1)

        # inverse transformation of first node
        H_IK0_inv = SE3inv(H_IK0)

        # compute relative transformation
        H_K0K1 = H_IK0_inv @ H_IK1

        # compute relative screw
        h_K0K1 = Log_SE3(H_K0K1)

        # find element number containing xi
        el = self.element_number(xi)

        # get element interval
        xi0, xi1 = self.knot_vector.element_interval(el)

        # second linear Lagrange shape function
        N1_xi = 1.0 / (xi1 - xi0)
        N1 = (xi - xi0) * N1_xi

        N, N_xi = self.basis_functions(xi)
        N1 = N[1]
        N1_xi = N_xi[1]

        # relative interpolation of local se(3) objects
        h_local = N1 * h_K0K1
        h_local_xi = N1_xi * h_K0K1

        # composition of reference and local transformation
        H_local = Exp_SE3(h_local)
        H_IK = H_IK0 @ H_local

        # extract centerline and transformation matrix
        A_IK = H_IK[:3, :3]
        r_OP = H_IK[:3, 3]

        # extract strains
        K_Gamma_bar = h_local_xi[:3]
        K_Kappa_bar = h_local_xi[3:]

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def __d_eval_two_node(self, qe, xi):
        # extract nodal screws
        nodalDOF0 = np.concatenate(
            (self.nodalDOF_element_r[0], self.nodalDOF_element_psi[0])
        )
        nodalDOF1 = np.concatenate(
            (self.nodalDOF_element_r[1], self.nodalDOF_element_psi[1])
        )
        h0 = qe[nodalDOF0]
        r_OP0 = h0[:3]
        psi0 = h0[3:]
        h1 = qe[nodalDOF1]
        r_OP1 = h1[:3]
        psi1 = h1[3:]

        # nodal transformations
        A_IK0 = Exp_SO3(psi0)
        A_IK1 = Exp_SO3(psi1)
        H_IK0 = SE3(A_IK0, r_OP0)
        H_IK1 = SE3(A_IK1, r_OP1)
        A_IK0_psi0 = Exp_SO3_psi(psi0)
        A_IK1_psi1 = Exp_SO3_psi(psi1)

        H_IK0_h0 = np.zeros((4, 4, 6), dtype=float)
        H_IK0_h0[:3, :3, 3:] = A_IK0_psi0
        H_IK0_h0[:3, 3, :3] = np.eye(3, dtype=float)
        H_IK1_h1 = np.zeros((4, 4, 6), dtype=float)
        H_IK1_h1[:3, :3, 3:] = A_IK1_psi1
        H_IK1_h1[:3, 3, :3] = np.eye(3, dtype=float)

        # inverse transformation of first node
        H_IK0_inv = SE3inv(H_IK0)
        H_IK0_inv_h0 = np.zeros((4, 4, 6), dtype=float)
        H_IK0_inv_h0[:3, :3, 3:] = A_IK0_psi0.transpose(1, 0, 2)
        H_IK0_inv_h0[:3, 3, 3:] = -np.einsum("k,kij->ij", r_OP0, A_IK0_psi0)
        H_IK0_inv_h0[:3, 3, :3] = -A_IK0.T

        # compute relative transformation
        H_K0K1 = H_IK0_inv @ H_IK1
        H_K0K1_h0 = np.einsum("ilk,lj->ijk", H_IK0_inv_h0, H_IK1)
        H_K0K1_h1 = np.einsum("il,ljk->ijk", H_IK0_inv, H_IK1_h1)

        # compute relative screw
        h_K0K1 = Log_SE3(H_K0K1)
        h_K0K1_HK0K1 = Log_SE3_H(H_K0K1)
        h_K0K1_h0 = np.einsum("ikl,klj->ij", h_K0K1_HK0K1, H_K0K1_h0)
        h_K0K1_h1 = np.einsum("ikl,klj->ij", h_K0K1_HK0K1, H_K0K1_h1)

        # find element number containing xi
        el = self.element_number(xi)

        # get element interval
        xi0, xi1 = self.knot_vector.element_interval(el)

        # second linear Lagrange shape function
        N1_xi = 1.0 / (xi1 - xi0)
        N1 = (xi - xi0) * N1_xi

        # relative interpolation of local se(3) objects
        h_local = N1 * h_K0K1
        h_local_xi = N1_xi * h_K0K1
        h_local_h0 = N1 * h_K0K1_h0
        h_local_h1 = N1 * h_K0K1_h1
        h_local_xi_h0 = N1_xi * h_K0K1_h0
        h_local_xi_h1 = N1_xi * h_K0K1_h1

        # composition of reference and local transformation
        H_local = Exp_SE3(h_local)
        H_local_h = Exp_SE3_h(h_local)
        H_local_h0 = np.einsum("ijl,lk->ijk", H_local_h, h_local_h0)
        H_local_h1 = np.einsum("ijl,lk->ijk", H_local_h, h_local_h1)
        H_IK = H_IK0 @ H_local
        H_IK_h0 = np.einsum("ilk,lj", H_IK0_h0, H_local) + np.einsum(
            "il,ljk->ijk", H_IK0, H_local_h0
        )
        H_IK_h1 = np.einsum("il,ljk->ijk", H_IK0, H_local_h1)

        # extract centerline and transformation matrix
        A_IK = H_IK[:3, :3]
        r_OP = H_IK[:3, 3]
        A_IK_qe = np.zeros((3, 3, self.nq_element), dtype=float)
        A_IK_qe[:, :, nodalDOF0] = H_IK_h0[:3, :3]
        A_IK_qe[:, :, nodalDOF1] = H_IK_h1[:3, :3]
        r_OP_qe = np.zeros((3, self.nq_element), dtype=float)
        r_OP_qe[:, nodalDOF0] = H_IK_h0[:3, 3]
        r_OP_qe[:, nodalDOF1] = H_IK_h1[:3, 3]

        # extract strains
        K_Gamma_bar = h_local_xi[:3]
        K_Kappa_bar = h_local_xi[3:]
        K_Gamma_bar_qe = np.zeros((3, self.nq_element), dtype=float)
        K_Gamma_bar_qe[:, nodalDOF0] = h_local_xi_h0[:3]
        K_Gamma_bar_qe[:, nodalDOF1] = h_local_xi_h1[:3]
        K_Kappa_bar_qe = np.zeros((3, self.nq_element), dtype=float)
        K_Kappa_bar_qe[:, nodalDOF0] = h_local_xi_h0[3:]
        K_Kappa_bar_qe[:, nodalDOF1] = h_local_xi_h1[3:]

        return (
            r_OP,
            A_IK,
            K_Gamma_bar,
            K_Kappa_bar,
            r_OP_qe,
            A_IK_qe,
            K_Gamma_bar_qe,
            K_Kappa_bar_qe,
        )

    def __eval_generic(self, qe, xi):
        # extract nodal positions and rotation vectors of first node (reference)
        r_OP0 = qe[self.nodalDOF_element_r[0]]
        psi0 = qe[self.nodalDOF_element_psi[0]]

        # evaluate nodal rotation matrix
        A_IK0 = Exp_SO3(psi0)

        # evaluate inverse reference SE(3) object
        H_IR = SE3(A_IK0, r_OP0)
        H_IR_inv = SE3inv(H_IR)

        # evaluate shape functions
        N, N_xi = self.basis_functions(xi)

        # relative interpolation of local se(3) objects
        h_rel = np.zeros(6, dtype=qe.dtype)
        h_rel_xi = np.zeros(6, dtype=qe.dtype)

        for node in range(self.nnodes_element):
            # nodal centerline
            r_IK_node = qe[self.nodalDOF_element_r[node]]

            # nodal rotation
            A_IK_node = Exp_SO3(qe[self.nodalDOF_element_psi[node]])

            # nodal SE(3) object
            H_IK_node = SE3(A_IK_node, r_IK_node)

            # relative SE(3)/ se(3) objects
            H_RK = H_IR_inv @ H_IK_node
            h_RK = Log_SE3(H_RK)

            # add wheighted contribution of local se(3) object
            h_rel += N[node] * h_RK
            h_rel_xi += N_xi[node] * h_RK

        # composition of reference and local SE(3) objects
        H_IK = H_IR @ Exp_SE3(h_rel)

        # extract centerline and transformation matrix
        A_IK = H_IK[:3, :3]
        r_OP = H_IK[:3, 3]

        # strain measures
        strains = T_SE3(h_rel) @ h_rel_xi

        # extract strains
        K_Gamma_bar = strains[:3]
        K_Kappa_bar = strains[3:]

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def __d_eval_generic(self, qe, xi):
        raise NotImplementedError
        nodalDOF_element = lambda node: np.concatenate(
            (self.nodalDOF_element_r[node], self.nodalDOF_element_psi[node])
        )

        # compute nodal rotations
        H_IKs = np.array(
            [
                SE3(
                    Exp_SO3(qe[self.nodalDOF_element_psi[node]]),
                    qe[self.nodalDOF_element_r[node]],
                )
                for node in range(self.nnodes_element)
            ]
        )
        H_IK_hs = np.array(
            [
                Exp_SE3_h(qe[nodalDOF_element(node)])
                for node in range(self.nnodes_element)
            ]
        )

        # # compute inverse of derivative of SE(3) derivative of first node
        # # TODO: Check this using a numerical derivative!
        # H_IK0_inv_h = np.zeros((4, 4, 6), dtype=float)
        # H_IK0_inv_h[:3, :3] = H_IK_hs[0, :3, :3].transpose(1, 0, 2)
        # H_IK0_inv_h[:3, 3] = -np.einsum(
        #     "jik,j->ik",
        #     H_IK_hs[0, :3, :3],
        #     H_IKs[0, :3, 3]
        # ) - np.einsum(
        #     "ji,jk->ik",
        #     H_IKs[0, :3, :3],
        #     H_IK_hs[0, :3, 3]
        # )
        # diff = H_IK0_inv_h - Exp_SE3_inv_h(qe[nodalDOF_element(0)])
        # error = np.linalg.norm(diff)
        # print(f"error H_IK0_inv_h: {error}")
        H_IK0_inv_h = Exp_SE3_inv_h(qe[nodalDOF_element(0)])

        # compute relative rotations and the corresponding rotation vectors
        H_IK0_inv = SE3inv(H_IKs[0])
        H_K0Ki = np.array(
            [H_IK0_inv @ H_IKs[node] for node in range(self.nnodes_element)]
        )
        h_K0Ki = np.array(
            [Log_SE3(H_K0Ki[node]) for node in range(self.nnodes_element)]
        )

        # evaluate shape functions
        N, N_xi = self.basis_functions(xi)

        # relative interpolation of local rotation vector
        h_K0K = np.sum(
            [N[node] * h_K0Ki[node] for node in range(self.nnodes_element)], axis=0
        )
        h_K0K_xi = np.sum(
            [N_xi[node] * h_K0Ki[node] for node in range(self.nnodes_element)], axis=0
        )

        # evaluate rotation and its derivative at interpolated position
        H_K0K = Exp_SE3(h_K0K)
        H_K0K_h = Exp_SE3_h(h_K0K)

        H_IK_q = np.zeros((4, 4, self.nq_element), dtype=float)

        # first node contribution part I
        H_IK_q[:, :, nodalDOF_element(0)] = np.einsum("ilk,lj->ijk", H_IK_hs[0], H_K0K)

        Tmp1 = np.einsum("il,ljm->ijm", H_IKs[0], H_K0K_h)

        for node in range(self.nnodes_element):
            Tmp2 = np.einsum("ijm,mno->ijno", Tmp1, N[node] * Log_SE3_H(H_K0Ki[node]))

            H_IK_q[:, :, nodalDOF_element(0)] += np.einsum(
                "ijno,npk,po",
                Tmp2,
                H_IK0_inv_h,
                H_IKs[node],
            )
            H_IK_q[:, :, nodalDOF_element(node)] += np.einsum(
                "ijno,np,pok", Tmp2, SE3inv(H_IKs[0]), H_IK_hs[node]
            )

        # extract centerline and transformation matrix
        A_IK_q = H_IK_q[:3, :3]
        r_OP_q = H_IK_q[:3, 3]

        return r_OP_q, A_IK_q

    def assembler_callback(self):
        if self.constant_mass_matrix:
            self.__M_coo()

    #########################################
    # equations of motion
    #########################################
    def M_el_constant(self, el):
        M_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # delta_r A_rho0 r_ddot part
            M_el_r_r = np.eye(3) * self.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_psi_psi = self.K_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )

        return M_el

    def M_el(self, qe, el):
        M_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # delta_r A_rho0 r_ddot part
            M_el_r_r = np.eye(3) * self.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_psi_psi = self.K_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )

            # For non symmetric cross sections there are also other parts
            # involved in the mass matrix. These parts are configuration
            # dependent and lead to configuration dependent mass matrix.
            _, A_IK, _, _ = self.eval(qe, self.qp[el, i])
            M_el_r_psi = A_IK @ self.K_S_rho0 * Ji * qwi
            M_el_psi_r = A_IK @ self.K_S_rho0 * Ji * qwi

            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_psi * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_r * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )

        return M_el

    def __M_coo(self):
        self.__M = Coo((self.nu, self.nu))
        for el in range(self.nelement):
            # extract element degrees of freedom
            elDOF = self.elDOF[el]

            # sparse assemble element mass matrix
            self.__M.extend(
                self.M_el_constant(el), (self.uDOF[elDOF], self.uDOF[elDOF])
            )

    # TODO: Compute derivative of mass matrix for non constant mass matrix case!
    def M(self, t, q, coo):
        if self.constant_mass_matrix:
            coo.extend_sparse(self.__M)
        else:
            for el in range(self.nelement):
                # extract element degrees of freedom
                elDOF = self.elDOF[el]

                # sparse assemble element mass matrix
                coo.extend(
                    self.M_el(q[elDOF], el), (self.uDOF[elDOF], self.uDOF[elDOF])
                )

    def h(self, t, q, u):
        h = np.zeros(self.nu, dtype=np.common_type(q, u))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            qe = q[elDOF]
            ue = u[elDOF]
            h[elDOF] += self.f_pot_el(qe, el) - self.f_gyr_el(t, qe, ue, el)
        return h

    def h_q(self, t, q, u, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            h_q_el = self.f_pot_q_el(q[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(h_q_el, (self.uDOF[elDOF], self.qDOF[elDOF]))

    def h_u(self, t, q, u, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            h_u_el = self.f_gyr_u_el(q[elDOF], u[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(h_u_el, (self.uDOF[elDOF], self.uDOF[elDOF]))

    def f_gyr_el(self, t, qe, ue, el):
        f_gyr_el = np.zeros(self.nq_element, dtype=float)

        for i in range(self.nquadrature):
            # interpoalte angular velocity
            K_Omega = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element):
                K_Omega += self.N[el, i, node] * ue[self.nodalDOF_element_psi[node]]

            # vector of gyroscopic forces
            f_gyr_el_psi = (
                cross3(K_Omega, self.K_I_rho0 @ K_Omega)
                * self.J[el, i]
                * self.qw[el, i]
            )

            # multiply vector of gyroscopic forces with nodal virtual rotations
            for node in range(self.nnodes_element):
                f_gyr_el[self.nodalDOF_element_psi[node]] += (
                    self.N[el, i, node] * f_gyr_el_psi
                )

        return f_gyr_el

    def f_gyr_u_el(self, t, qe, ue, el):
        f_gyr_u_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # interpoalte angular velocity
            K_Omega = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element):
                K_Omega += self.N[el, i, node] * ue[self.nodalDOF_element_psi[node]]

            # derivative of vector of gyroscopic forces
            f_gyr_u_el_psi = (
                ((ax2skew(K_Omega) @ self.K_I_rho0 - ax2skew(self.K_I_rho0 @ K_Omega)))
                * self.J[el, i]
                * self.qw[el, i]
            )

            # multiply derivative of gyroscopic force vector with nodal virtual rotations
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    f_gyr_u_el[nodalDOF_a[:, None], nodalDOF_b] += f_gyr_u_el_psi * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )

        return f_gyr_u_el

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

            # objective interpolation
            _, _, K_Gamma_bar, K_Kappa_bar = self.eval(qe, self.qp[el, i])

            # axial and shear strains
            K_Gamma = K_Gamma_bar / Ji

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / Ji

            # evaluate strain energy function
            E_pot_el += (
                self.material_model.potential(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
                * Ji
                * qwi
            )

        return E_pot_el

    def f_pot_el(self, qe, el):
        f_pot_el = np.zeros(self.nq_element, dtype=qe.dtype)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # objective interpolation
            _, A_IK, K_Gamma_bar, K_Kappa_bar = self.eval(qe, self.qp[el, i])

            # axial and shear strains
            K_Gamma = K_Gamma_bar / Ji

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / Ji

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)

            ############################
            # virtual work contributions
            ############################
            # - first delta Gamma part
            for node in range(self.nnodes_element):
                f_pot_el[self.nodalDOF_element_r[node]] -= (
                    self.N_xi[el, i, node] * A_IK @ K_n * qwi
                )

            for node in range(self.nnodes_element):
                # - second delta Gamma part
                f_pot_el[self.nodalDOF_element_psi[node]] += (
                    self.N[el, i, node] * cross3(K_Gamma_bar, K_n) * qwi
                )

                # - first delta kappa part
                f_pot_el[self.nodalDOF_element_psi[node]] -= (
                    self.N_xi[el, i, node] * K_m * qwi
                )

                # second delta kappa part
                f_pot_el[self.nodalDOF_element_psi[node]] += (
                    self.N[el, i, node] * cross3(K_Kappa_bar, K_m) * qwi
                )  # Euler term

        return f_pot_el

    def f_pot_q_el(self, qe, el):
        f_pot_q_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # objective interpolation
            (
                r_OP,
                A_IK,
                K_Gamma_bar,
                K_Kappa_bar,
                r_OP_qe,
                A_IK_qe,
                K_Gamma_bar_qe,
                K_Kappa_bar_qe,
            ) = self.d_eval(qe, self.qp[el, i])

            # axial and shear strains
            K_Gamma = K_Gamma_bar / Ji
            K_Gamma_qe = K_Gamma_bar_qe / Ji

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / Ji
            K_Kappa_qe = K_Kappa_bar_qe / Ji

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_n_K_Gamma = self.material_model.K_n_K_Gamma(
                K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
            )
            K_n_K_Kappa = self.material_model.K_n_K_Kappa(
                K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
            )
            K_n_qe = K_n_K_Gamma @ K_Gamma_qe + K_n_K_Kappa @ K_Kappa_qe

            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
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
            # - first delta Gamma part
            for node in range(self.nnodes_element):
                f_pot_q_el[self.nodalDOF_element_r[node], :] -= (
                    self.N_xi[el, i, node]
                    * qwi
                    * (np.einsum("ikj,k->ij", A_IK_qe, K_n) + A_IK @ K_n_qe)
                )

            for node in range(self.nnodes_element):
                # - second delta Gamma part
                f_pot_q_el[self.nodalDOF_element_psi[node], :] += (
                    self.N[el, i, node]
                    * qwi
                    * (ax2skew(K_Gamma_bar) @ K_n_qe - ax2skew(K_n) @ K_Gamma_bar_qe)
                )

                # - first delta kappa part
                f_pot_q_el[self.nodalDOF_element_psi[node], :] -= (
                    self.N_xi[el, i, node] * qwi * K_m_qe
                )

                # - second delta kappa part
                f_pot_q_el[self.nodalDOF_element_psi[node], :] += (
                    self.N[el, i, node]
                    * qwi
                    * (ax2skew(K_Kappa_bar) @ K_m_qe - ax2skew(K_m) @ K_Kappa_bar_qe)
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

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        # centerline part
        q_dot = u

        # correct axis angle vector part
        for node in range(self.nnode):
            nodalDOF_psi = self.nodalDOF_psi[node]
            psi = q[nodalDOF_psi]
            K_omega_IK = u[nodalDOF_psi]

            psi_dot = T_SO3_inv(psi) @ K_omega_IK
            q_dot[nodalDOF_psi] = psi_dot

        return q_dot

    def B(self, t, q, coo):
        # trivial kinematic equation for centerline
        coo.extend_diag(
            np.ones(self.nq_r), (self.qDOF[: self.nq_r], self.uDOF[: self.nq_r])
        )

        # axis angle vector part
        for node in range(self.nnode):
            nodalDOF_psi = self.nodalDOF_psi[node]

            psi = q[nodalDOF_psi]
            coo.extend(
                T_SO3_inv(psi),
                (self.qDOF[nodalDOF_psi], self.uDOF[nodalDOF_psi]),
            )

    def q_ddot(self, t, q, u, u_dot):
        raise RuntimeError("Not tested!")
        # centerline part
        q_ddot = u_dot

        # correct axis angle vector part
        for node in range(self.nnode):
            nodalDOF_psi = self.nodalDOF_psi[node]

            psi = q[nodalDOF_psi]
            omega = u[nodalDOF_psi]
            omega_dot = u_dot[nodalDOF_psi]

            T_inv = T_SO3_inv(psi)
            psi_dot = T_inv @ omega

            # TODO:
            T_dot = tangent_map_s(psi, psi_dot)
            Tinv_dot = -T_inv @ T_dot @ T_inv
            psi_ddot = T_inv @ omega_dot + Tinv_dot @ omega

            q_ddot[nodalDOF_psi] = psi_ddot

        return q_ddot

    # change between rotation vector and its complement in order to circumvent
    # singularities of the rotation vector
    @staticmethod
    def psi_C(psi):
        angle = norm(psi)
        if angle < pi:
            return psi
        else:
            # Ibrahimbegovic1995 after (62)
            psi_C = (1.0 - 2.0 * pi / angle) * psi
            return psi_C

    def step_callback(self, t, q, u):
        for node in range(self.nnode):
            psi = q[self.nodalDOF_psi[node]]
            q[self.nodalDOF_psi[node]] = TimoshenkoAxisAngleSE3.psi_C(psi)
        return q, u

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def elDOF_P(self, frame_ID):
        xi = frame_ID[0]
        el = self.element_number(xi)
        return self.elDOF[el]

    def local_qDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def local_uDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    ###################
    # r_OP contribution
    ###################
    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        r, A_IK, _, _ = self.__eval_two_node(q, frame_ID[0])
        return r + A_IK @ K_r_SP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # (
        #     r_OP,
        #     A_IK,
        #     K_Gamma_bar,
        #     K_Kappa_bar,
        #     r_q,
        #     A_IK_q,
        #     K_Gamma_bar_q,
        #     K_Kappa_bar_q,
        # ) = self.d_eval(q, frame_ID[0])
        # r_OP_q = r_q + np.einsum("ijk,j->ik", A_IK_q, K_r_SP)
        # return r_OP_q

        r_OP_q_num = approx_fprime(
            q, lambda q: self.r_OP(t, q, frame_ID, K_r_SP), eps=1.0e-10, method="cs"
        )
        # diff = r_OP_q - r_OP_q_num
        # error = np.linalg.norm(diff)
        # np.set_printoptions(3, suppress=True)
        # if error > 1.0e-10:
        #     print(f"r_OP_q:\n{r_OP_q}")
        #     print(f"r_OP_q_num:\n{r_OP_q_num}")
        #     print(f"error r_OP_q: {error}")
        return r_OP_q_num

    def A_IK(self, t, q, frame_ID):
        _, A_IK, _, _ = self.eval(q, frame_ID[0])
        return A_IK

    def A_IK_q(self, t, q, frame_ID):
        # (
        #     r_OP,
        #     A_IK,
        #     K_Gamma_bar,
        #     K_Kappa_bar,
        #     r_OP_q,
        #     A_IK_q,
        #     K_Gamma_bar_q,
        #     K_Kappa_bar_q,
        # ) = self.d_eval(q, frame_ID[0])
        # return A_IK_q

        A_IK_q_num = approx_fprime(
            q, lambda q: self.A_IK(t, q, frame_ID), eps=1.0e-10, method="cs"
        )
        # diff = A_IK_q - A_IK_q_num
        # error = np.linalg.norm(diff)
        # # if error > 1.0e-10:
        # print(f'error A_IK_q: {error}')
        return A_IK_q_num

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3), dtype=float):
        N, _ = self.basis_functions(frame_ID[0])
        _, A_IK, _, _ = self.eval(q, frame_ID[0])

        # angular velocity in K-frame
        K_Omega = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element):
            K_Omega += N[node] * u[self.nodalDOF_element_psi[node]]

        # centerline velocity
        v_C = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element):
            v_C += N[node] * u[self.nodalDOF_element_r[node]]

        return v_C + A_IK @ cross3(K_Omega, K_r_SP)

    # TODO:
    def v_P_q(self, t, q, u, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        raise RuntimeError(
            "Implement this derivative since it requires known parts only!"
        )
        v_P_q_num = approx_fprime(
            q, lambda q: self.v_P(t, q, u, frame_ID, K_r_SP), method="3-point"
        )
        return v_P_q_num

    def J_P(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # evaluate required nodal shape functions
        N, _ = self.basis_functions(frame_ID[0])

        # transformation matrix
        _, A_IK, _, _ = self.eval(q, frame_ID[0])

        # skew symmetric matrix of K_r_SP
        K_r_SP_tilde = ax2skew(K_r_SP)

        # interpolate centerline and axis angle contributions
        J_P = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element):
            J_P[:, self.nodalDOF_element_r[node]] += N[node] * np.eye(3, dtype=float)
        for node in range(self.nnodes_element):
            J_P[:, self.nodalDOF_element_psi[node]] -= N[node] * A_IK @ K_r_SP_tilde

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
        N, _ = self.basis_functions(frame_ID[0])

        K_r_SP_tilde = ax2skew(K_r_SP)
        A_IK_q = self.A_IK_q(t, q, frame_ID)
        prod = np.einsum("ijl,jk", A_IK_q, K_r_SP_tilde)

        # interpolate axis angle contributions since centerline contributon is
        # zero
        J_P_q = np.zeros((3, self.nq_element, self.nq_element), dtype=float)
        for node in range(self.nnodes_element):
            nodalDOF_psi = self.nodalDOF_element_psi[node]
            J_P_q[:, nodalDOF_psi] -= N[node] * prod

        return J_P_q

        # J_P_q_num = approx_fprime(
        #     q, lambda q: self.J_P(t, q, frame_ID, K_r_SP), method="3-point"
        # )
        # diff = J_P_q_num - J_P_q
        # error = np.linalg.norm(diff)
        # print(f"error J_P_q: {error}")
        # return J_P_q_num

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        N, _ = self.basis_functions(frame_ID[0])
        _, A_IK, _, _ = self.eval(q, frame_ID[0])

        # centerline acceleration
        a_C = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element):
            a_C += N[node] * u_dot[self.nodalDOF_element_r[node]]

        # angular velocity and acceleration in K-frame
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID)

        # rigid body formular
        return a_C + A_IK @ (
            cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP))
        )

    # TODO:
    def a_P_q(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID)
        a_P_q = np.einsum(
            "ijk,j->ik",
            self.A_IK_q(t, q, frame_ID),
            cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP)),
        )
        # return a_P_q

        a_P_q_num = approx_fprime(
            q, lambda q: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP), method="3-point"
        )
        diff = a_P_q_num - a_P_q
        error = np.linalg.norm(diff)
        print(f"error a_P_q: {error}")
        return a_P_q_num

    # TODO:
    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        local = -self.A_IK(t, q, frame_ID) @ (
            ax2skew(cross3(K_Omega, K_r_SP)) + ax2skew(K_Omega) @ ax2skew(K_r_SP)
        )

        N, _ = self.basis_functions(frame_ID[0])
        a_P_u = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element):
            a_P_u[:, self.nodalDOF_element_psi[node]] += N[node] * local

        # return a_P_u

        a_P_u_num = approx_fprime(
            u, lambda u: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP), method="3-point"
        )
        diff = a_P_u_num - a_P_u
        error = np.linalg.norm(diff)
        print(f"error a_P_u: {error}")
        return a_P_u_num

    def K_Omega(self, t, q, u, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        angular velocities in the K-frame.
        """
        N, _ = self.basis_functions(frame_ID[0])
        K_Omega = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element):
            K_Omega += N[node] * u[self.nodalDOF_element_psi[node]]
        return K_Omega

    def K_Omega_q(self, t, q, u, frame_ID):
        return np.zeros((3, self.nq_element), dtype=float)

    def K_J_R(self, t, q, frame_ID):
        N, _ = self.basis_functions(frame_ID[0])
        K_J_R = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element):
            K_J_R[:, self.nodalDOF_element_psi[node]] += N[node] * np.eye(3)
        return K_J_R

        # K_J_R_num = approx_fprime(
        #     np.zeros(self.nu_element, dtype=float),
        #     lambda u: self.K_Omega(t, q, u, frame_ID),
        #     method="3-point",
        # )
        # diff = K_J_R - K_J_R_num
        # error = np.linalg.norm(diff)
        # print(f"error K_J_R: {error}")
        # return K_J_R_num

    def K_J_R_q(self, t, q, frame_ID):
        return np.zeros((3, self.nq_element, self.nq_element), dtype=float)

    def K_Psi(self, t, q, u, u_dot, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        time derivative of the angular velocities in the K-frame.
        """
        N, _ = self.basis_functions(frame_ID[0])
        K_Psi = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element):
            K_Psi += N[node] * u_dot[self.nodalDOF_element_psi[node]]
        return K_Psi

    def K_Psi_q(self, t, q, u, u_dot, frame_ID):
        return np.zeros((3, self.nq_element), dtype=float)

    def K_Psi_u(self, t, q, u, u_dot, frame_ID):
        return np.zeros((3, self.nq_element), dtype=float)

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
            r_C = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element):
                r_C += self.N[el, i, node] * qe[self.nodalDOF_element_r[node]]

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
        fe = np.zeros(self.nq_element, dtype=float)
        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # compute local force vector
            fe_r = force(t, qwi) * Ji * qwi

            # multiply local force vector with variation of centerline
            for node in range(self.nnodes_element):
                fe[self.nodalDOF_element_r[node]] += self.N[el, i, node] * fe_r

        return fe

    def distributed_force1D(self, t, q, force):
        f = np.zeros(self.nq, dtype=float)
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
            qe = q_body[self.local_qDOF_P(frame_ID)]
            r.append(self.r_OP(1, qe, frame_ID))
        return np.array(r).T

    def cover(self, q, radius, n_xi=20, n_alpha=100):
        q_body = q[self.qDOF]
        points = []
        for xi in np.linspace(0, 1, num=n_xi):
            frame_ID = (xi,)
            elDOF = self.elDOF_P(frame_ID)
            qe = q_body[elDOF]

            # point on the centerline and tangent vector
            r = self.r_OC(0, qe, (xi,))

            # evaluate directors
            A_IK = self.A_IK(0, qe, frame_ID=(xi,))
            _, d2, d3 = A_IK.T

            # start with point on centerline
            points.append(r)

            # compute points on circular cross section
            x0 = None  # initial point is required twice
            for alpha in np.linspace(0, 2 * np.pi, num=n_alpha):
                x = r + radius * np.cos(alpha) * d2 + radius * np.sin(alpha) * d3
                points.append(x)
                if x0 is None:
                    x0 = x

            # end with first point on cross section
            points.append(x0)

            # end with point on centerline
            points.append(r)

        return np.array(points).T

    def plot_centerline(self, ax, q, n=100, color="black"):
        ax.plot(*self.nodes(q), linestyle="dashed", marker="o", color=color)
        ax.plot(*self.centerline(q, n=n), linestyle="solid", color=color)

    def plot_frames(self, ax, q, n=10, length=1):
        r, d1, d2, d3 = self.frames(q, n=n)
        ax.quiver(*r, *d1, color="red", length=length)
        ax.quiver(*r, *d2, color="green", length=length)
        ax.quiver(*r, *d3, color="blue", length=length)


# TimoshenkoAxisAngleSE3 = TimoshenkoAxisAngleSE3_I_delta_r_P
# TimoshenkoAxisAngleSE3 = TimoshenkoAxisAngleSE3_K_delta_r_P


class TimoshenkoAxisAngleSE3(TimoshenkoPetrovGalerkinBase):
    def __init__(
        self,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelement,
        Q,
        q0=None,
        u0=None,
    ):
        super().__init__(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            1,
            1,
            nelement,
            Q,
            q0=q0,
            u0=u0,
            basis_r="Lagrange",
            basis_psi="Lagrange",
        )

    @staticmethod
    def straight_configuration(
        nelement,
        L,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
    ):
        return TimoshenkoPetrovGalerkinBase.straight_configuration(
            1, 1, "Lagrange", "Lagrange", nelement, L, r_OP, A_IK
        )

    def _eval(self, qe, xi):
        # extract nodal screws
        nodalDOF0 = np.concatenate(
            (self.nodalDOF_element_r[0], self.nodalDOF_element_psi[0])
        )
        nodalDOF1 = np.concatenate(
            (self.nodalDOF_element_r[1], self.nodalDOF_element_psi[1])
        )
        h0 = qe[nodalDOF0]
        r_OP0 = h0[:3]
        psi0 = h0[3:]
        h1 = qe[nodalDOF1]
        r_OP1 = h1[:3]
        psi1 = h1[3:]

        # nodal transformations
        H_IK0 = SE3(Exp_SO3(psi0), r_OP0)
        H_IK1 = SE3(Exp_SO3(psi1), r_OP1)

        # inverse transformation of first node
        H_IK0_inv = SE3inv(H_IK0)

        # compute relative transformation
        H_K0K1 = H_IK0_inv @ H_IK1

        # compute relative screw
        h_K0K1 = Log_SE3(H_K0K1)

        # find element number containing xi
        el = self.element_number(xi)

        # get element interval
        xi0, xi1 = self.knot_vector_r.element_interval(el)

        # second linear Lagrange shape function
        N1_xi = 1.0 / (xi1 - xi0)
        N1 = (xi - xi0) * N1_xi

        # relative interpolation of local se(3) objects
        h_local = N1 * h_K0K1
        h_local_xi = N1_xi * h_K0K1

        # composition of reference and local transformation
        H_local = Exp_SE3(h_local)
        H_IK = H_IK0 @ H_local

        # extract centerline and transformation matrix
        A_IK = H_IK[:3, :3]
        r_OP = H_IK[:3, 3]

        # extract strains
        K_Gamma_bar = h_local_xi[:3]
        K_Kappa_bar = h_local_xi[3:]

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def _deval(self, qe, xi):
        # extract nodal screws
        nodalDOF0 = np.concatenate(
            (self.nodalDOF_element_r[0], self.nodalDOF_element_psi[0])
        )
        nodalDOF1 = np.concatenate(
            (self.nodalDOF_element_r[1], self.nodalDOF_element_psi[1])
        )
        h0 = qe[nodalDOF0]
        r_OP0 = h0[:3]
        psi0 = h0[3:]
        h1 = qe[nodalDOF1]
        r_OP1 = h1[:3]
        psi1 = h1[3:]

        # nodal transformations
        A_IK0 = Exp_SO3(psi0)
        A_IK1 = Exp_SO3(psi1)
        H_IK0 = SE3(A_IK0, r_OP0)
        H_IK1 = SE3(A_IK1, r_OP1)
        A_IK0_psi0 = Exp_SO3_psi(psi0)
        A_IK1_psi1 = Exp_SO3_psi(psi1)

        H_IK0_h0 = np.zeros((4, 4, 6), dtype=float)
        H_IK0_h0[:3, :3, 3:] = A_IK0_psi0
        H_IK0_h0[:3, 3, :3] = np.eye(3, dtype=float)
        H_IK1_h1 = np.zeros((4, 4, 6), dtype=float)
        H_IK1_h1[:3, :3, 3:] = A_IK1_psi1
        H_IK1_h1[:3, 3, :3] = np.eye(3, dtype=float)

        # inverse transformation of first node
        H_IK0_inv = SE3inv(H_IK0)
        H_IK0_inv_h0 = np.zeros((4, 4, 6), dtype=float)
        H_IK0_inv_h0[:3, :3, 3:] = A_IK0_psi0.transpose(1, 0, 2)
        H_IK0_inv_h0[:3, 3, 3:] = -np.einsum("k,kij->ij", r_OP0, A_IK0_psi0)
        H_IK0_inv_h0[:3, 3, :3] = -A_IK0.T

        # compute relative transformation
        H_K0K1 = H_IK0_inv @ H_IK1
        H_K0K1_h0 = np.einsum("ilk,lj->ijk", H_IK0_inv_h0, H_IK1)
        H_K0K1_h1 = np.einsum("il,ljk->ijk", H_IK0_inv, H_IK1_h1)

        # compute relative screw
        h_K0K1 = Log_SE3(H_K0K1)
        h_K0K1_HK0K1 = Log_SE3_H(H_K0K1)
        h_K0K1_h0 = np.einsum("ikl,klj->ij", h_K0K1_HK0K1, H_K0K1_h0)
        h_K0K1_h1 = np.einsum("ikl,klj->ij", h_K0K1_HK0K1, H_K0K1_h1)

        # find element number containing xi
        el = self.element_number(xi)

        # get element interval
        xi0, xi1 = self.knot_vector_r.element_interval(el)

        # second linear Lagrange shape function
        N1_xi = 1.0 / (xi1 - xi0)
        N1 = (xi - xi0) * N1_xi

        # relative interpolation of local se(3) objects
        h_local = N1 * h_K0K1
        h_local_xi = N1_xi * h_K0K1
        h_local_h0 = N1 * h_K0K1_h0
        h_local_h1 = N1 * h_K0K1_h1
        h_local_xi_h0 = N1_xi * h_K0K1_h0
        h_local_xi_h1 = N1_xi * h_K0K1_h1

        # composition of reference and local transformation
        H_local = Exp_SE3(h_local)
        H_local_h = Exp_SE3_h(h_local)
        H_local_h0 = np.einsum("ijl,lk->ijk", H_local_h, h_local_h0)
        H_local_h1 = np.einsum("ijl,lk->ijk", H_local_h, h_local_h1)
        H_IK = H_IK0 @ H_local
        H_IK_h0 = np.einsum("ilk,lj", H_IK0_h0, H_local) + np.einsum(
            "il,ljk->ijk", H_IK0, H_local_h0
        )
        H_IK_h1 = np.einsum("il,ljk->ijk", H_IK0, H_local_h1)

        # extract centerline and transformation matrix
        A_IK = H_IK[:3, :3]
        r_OP = H_IK[:3, 3]
        A_IK_qe = np.zeros((3, 3, self.nq_element), dtype=float)
        A_IK_qe[:, :, nodalDOF0] = H_IK_h0[:3, :3]
        A_IK_qe[:, :, nodalDOF1] = H_IK_h1[:3, :3]
        r_OP_qe = np.zeros((3, self.nq_element), dtype=float)
        r_OP_qe[:, nodalDOF0] = H_IK_h0[:3, 3]
        r_OP_qe[:, nodalDOF1] = H_IK_h1[:3, 3]

        # extract strains
        K_Gamma_bar = h_local_xi[:3]
        K_Kappa_bar = h_local_xi[3:]
        K_Gamma_bar_qe = np.zeros((3, self.nq_element), dtype=float)
        K_Gamma_bar_qe[:, nodalDOF0] = h_local_xi_h0[:3]
        K_Gamma_bar_qe[:, nodalDOF1] = h_local_xi_h1[:3]
        K_Kappa_bar_qe = np.zeros((3, self.nq_element), dtype=float)
        K_Kappa_bar_qe[:, nodalDOF0] = h_local_xi_h0[3:]
        K_Kappa_bar_qe[:, nodalDOF1] = h_local_xi_h1[3:]

        return (
            r_OP,
            A_IK,
            K_Gamma_bar,
            K_Kappa_bar,
            r_OP_qe,
            A_IK_qe,
            K_Gamma_bar_qe,
            K_Kappa_bar_qe,
        )

    def A_IK(self, t, q, frame_ID):
        return self._eval(q, frame_ID[0])[1]

    def A_IK_q(self, t, q, frame_ID):
        return self._deval(q, frame_ID[0])[5]
