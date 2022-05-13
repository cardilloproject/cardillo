import numpy as np
import meshio
import os
from math import sin, cos, sqrt
from cardillo.math.algebra import skew2ax

from cardillo.utility.coo import Coo
from cardillo.discretization.B_spline import KnotVector
from cardillo.discretization.lagrange import Node_vector
from cardillo.discretization.Hermite import HermiteNodeVector
from cardillo.discretization.mesh1D import Mesh1D
from cardillo.math import norm, cross3, ax2skew, approx_fprime
from cardillo.math import (
    rodriguez,
    rodriguez_der,
    rodriguez_inv,
    tangent_map,
    inverse_tangent_map,
    tangent_map_s,
    e1,
    Rotor,
    quat2mat,
    quat2mat_p,
    pi,
)


def SE3(A_IK, r_OP):
    H = np.zeros((4, 4), dtype=float)
    H[:3, :3] = A_IK
    H[:3, 3] = r_OP
    H[3, 3] = 1.0
    return H


def SE3inv(H):
    A_IK = H[:3, :3]
    r_OP = H[:3, 3]
    return SE3(A_IK.T, -A_IK.T @ r_OP)  # Sonneville2013 (12)


def se3exp(h):
    """See Murray1994 Example A.12 and Sonneville2014 (A.10)."""
    r_OP = h[:3]
    psi = h[3:]

    # H = np.eye(4, dtype=float)
    # psi2 = psi @ psi
    # if psi2 > 0:
    #     # exp SO(3)
    #     H[:3, :3] = rodriguez(psi)

    #     # tangent map
    #     abs_psi = sqrt(psi2)
    #     psi_tilde = ax2skew(psi)
    #     A = (
    #         np.eye(3, dtype=float)
    #         + (1.0 - cos(abs_psi)) / psi2 * psi_tilde
    #         + (abs_psi - sin(abs_psi)) / (abs_psi * psi2) * psi_tilde @ psi_tilde
    #     )
    #     # A = tangent_map(psi).T # Sonneville2013 (A.10)

    #     H[:3, 3] = A @ r_OP
    # else:
    #     H[:3, 3] = r_OP

    # Sonneville2013 (A.10)
    H = np.eye(4, dtype=float)
    H[:3, :3] = rodriguez(psi)
    H[:3, 3] = tangent_map(psi).T @ r_OP

    return H


def SE3log(H):
    """See Murray1994 Example A.14 and Sonneville2014 (A.15)."""
    A_IK = H[:3, :3]
    r_OP = H[:3, 3]

    # # log SO(3)
    # psi = rodriguez_inv(A_IK)

    # # inverse tangent map
    # psi2 = psi @ psi
    # A_inv = np.eye(3, dtype=float)
    # if psi2 > 0:
    #     abs_psi = sqrt(psi2)
    #     psi_tilde = ax2skew(psi)
    #     A_inv += (
    #         -0.5 * psi_tilde
    #         + (2.0 * sin(abs_psi) - abs_psi * (1.0 + cos(abs_psi)))
    #         / (2.0 * psi2 * sin(abs_psi))
    #         * psi_tilde
    #         @ psi_tilde
    #     )
    #     # A_inv = inverse_tangent_map(psi).T # Sonneville2013 (A.15)

    # h = np.concatenate((A_inv @ r_OP, psi))

    # Sonneville2013 (A.15)
    psi = rodriguez_inv(A_IK)
    h = np.concatenate((inverse_tangent_map(psi).T @ r_OP, psi))

    return h


def T_UOm_p(a, b):
    """Position part of the tangent map in se(3), see Sonnville2013 (A.12)."""
    a_tilde = ax2skew(a)

    b2 = b @ b
    if b2 > 0:
        abs_b = sqrt(b2)
        alpha = sin(abs_b) / abs_b
        beta = 2.0 * (1.0 - cos(abs_b)) / b2

        b_tilde = ax2skew(b)
        ab = a_tilde @ b_tilde + b_tilde @ a_tilde

        # Sonneville2014 (A.12)
        return (
            -0.5 * beta * a_tilde
            + (1.0 - alpha) * ab / b2
            + ((b @ a) / b2)
            * (
                (beta - alpha) * b_tilde
                + (0.5 * beta - 3.0 * (1.0 - alpha) / b2) * b_tilde @ b_tilde
            )
        )

        # # Park2005 (21)
        # return (
        #     0.5 * beta * a_tilde
        #     + (1. - alpha) * ab / b2
        #     + (b @ a)
        #     * (
        #         (alpha - beta) * b_tilde
        #         + (0.5 * beta - 3. * ((1. - alpha) / b2)) * b_tilde @ b_tilde
        #     )
        #     / b2
        # )
    else:
        return -0.5 * a_tilde  # Soneville2014
        # return 0.5 * a_tilde # Park2005


def se3tangent_map(h):
    """Tangent map in se(3), see Sonnville2013 (A.11)."""
    r = h[:3]
    psi = h[3:]

    T = np.zeros((6, 6), dtype=float)
    T[:3, :3] = T[3:, 3:] = tangent_map(psi)
    T[:3, 3:] = T_UOm_p(r, psi)
    return T


def T_UOm_m(a, b):
    """Position part of the inverse tangent map in se(3), see Sonnville2013 (A.14)."""
    a_tilde = ax2skew(a)

    b2 = b @ b
    if b2 > 0:
        abs_b = sqrt(b2)
        alpha = sin(abs_b) / abs_b
        beta = 2.0 * (1.0 - cos(abs_b)) / b2

        b_tilde = ax2skew(b)
        ab = a_tilde @ b_tilde + b_tilde @ a_tilde

        return (
            0.5 * a_tilde
            + (beta - alpha) / (beta * b2) * ab
            + (1.0 + alpha - 2.0 * beta)
            / (beta * b2 * b2)
            * (b @ a)
            * b_tilde
            @ b_tilde
        )
    else:
        return 0.5 * a_tilde


def se3inverse_tangent_map(h):
    """Inverse tangent map in se(3), see Sonnville2013 (A.13)."""
    r = h[:3]
    psi = h[3:]

    T = np.zeros((6, 6), dtype=float)
    T[:3, :3] = T[3:, 3:] = inverse_tangent_map(psi)
    T[:3, 3:] = T_UOm_m(r, psi)
    return T


class TimoshenkoAxisAngleSE3:
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
        # use_K_r=True,
        use_K_r=False,
    ):
        # use K_r instead of I_r
        self.use_K_r = use_K_r

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
        elif basis == "Lagrange":
            self.knot_vector_r = Node_vector(polynomial_degree_r, nelement)
            self.knot_vector_psi = Node_vector(polynomial_degree_psi, nelement)
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        # number of degrees of freedom per node
        self.nq_node_r = nq_node_r = 3
        self.nq_node_psi = nq_node_psi = 3

        # # Boolean array that detects if the complement rotaton vector has to
        # # be used on each node.
        # self.use_complement = np.zeros(self.nq_node_psi, dtype=bool)

        # build mesh objects
        self.mesh_r = Mesh1D(
            self.knot_vector_r,
            nquadrature,
            derivative_order=2,
            basis=basis,
            dim_q=nq_node_r,
        )
        self.mesh_psi = Mesh1D(
            self.knot_vector_psi,
            nquadrature,
            derivative_order=1,
            basis=basis,
            dim_q=nq_node_psi,
        )

        # total number of nodes
        self.nnode_r = self.mesh_r.nnodes
        self.nnode_psi = self.mesh_psi.nnodes

        # number of nodes per element
        self.nnodes_element_r = self.mesh_r.nnodes_per_element
        self.nnodes_element_psi = self.mesh_psi.nnodes_per_element

        # total number of generalized coordinates
        self.nq_r = self.mesh_r.nq
        self.nq_psi = self.mesh_psi.nq
        self.nq = self.nq_r + self.nq_psi  # total number of generalized coordinates
        self.nu = self.nq  # total number of generalized velocities

        # number of generalized coordiantes per element
        self.nq_element_r = self.mesh_r.nq_per_element
        self.nq_element_psi = self.mesh_psi.nq_per_element
        self.nq_element = self.nu_element = self.nq_element_r + self.nq_element_psi

        # global element connectivity
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_psi = self.mesh_psi.elDOF + self.nq_r

        # global nodal
        self.nodalDOF_r = self.mesh_r.nodalDOF
        self.nodalDOF_psi = self.mesh_psi.nodalDOF + self.nq_r

        # nodal connectivity on element level
        self.nodalDOF_element_r = self.mesh_r.nodalDOF_element
        self.nodalDOF_element_psi = self.mesh_psi.nodalDOF_element + self.nq_element_r

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nelement, self.nq_element), dtype=int)
        for el in range(nelement):
            self.elDOF[el, : self.nq_element_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_element_r :] = self.elDOF_psi[el]

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
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions_r = self.mesh_r.eval_basis
        self.basis_functions_psi = self.mesh_psi.eval_basis

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q  # reference configuration
        self.q0 = Q.copy() if q0 is None else q0  # initial configuration
        self.u0 = (
            np.zeros(self.nu, dtype=float) if u0 is None else u0
        )  # initial velocities

        # reference rotation for relative rotation proposed by Crisfield1999 (5.8)
        self.node_A = int(0.5 * (self.nnodes_element_psi + 1)) - 1
        self.node_B = int(0.5 * (self.nnodes_element_psi + 2)) - 1

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
                # TODO: Is this correct?
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
        polynomial_degree_r,
        polynomial_degree_psi,
        nelement,
        L,
        greville_abscissae=True,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
        basis="B-spline",
    ):
        if basis == "B-spline":
            nn_r = polynomial_degree_r + nelement
            nn_psi = polynomial_degree_psi + nelement
        elif basis == "Lagrange":
            nn_r = polynomial_degree_r * nelement + 1
            nn_psi = polynomial_degree_psi * nelement + 1
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        x0 = np.linspace(0, L, num=nn_r)
        y0 = np.zeros(nn_r, dtype=float)
        z0 = np.zeros(nn_r, dtype=float)
        if greville_abscissae and basis == "B-spline":
            kv = KnotVector.uniform(polynomial_degree_r, nelement)
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

    # def reference_rotation(self, qe: np.ndarray, case="left"):
    # def reference_rotation(self, qe: np.ndarray, case="right"):
    def reference_rotation(self, qe: np.ndarray, case="midway"):
        """Reference rotation for SE(3) object in analogy to the proposed
        formulation by Crisfield1999 (5.8).

        Three cases are implemented: 'midway', 'left'  and 'right'.
        """

        if case == "midway":
            # nodal centerline
            r_IA = qe[self.nodalDOF_element_r[self.node_A]]
            r_IB = qe[self.nodalDOF_element_r[self.node_B]]

            # nodal rotations
            A_IA = rodriguez(qe[self.nodalDOF_element_psi[self.node_A]])
            A_IB = rodriguez(qe[self.nodalDOF_element_psi[self.node_B]])

            # nodal SE(3) objects
            H_IA = SE3(A_IA, r_IA)
            H_IB = SE3(A_IB, r_IB)

            # midway SE(3) object
            return H_IA @ se3exp(0.5 * SE3log(SE3inv(H_IA) @ H_IB))
        elif case == "left":
            r_I0 = qe[self.nodalDOF_element_r[0]]
            A_I0 = rodriguez(qe[self.nodalDOF_element_psi[0]])
            return SE3(A_IA, r_I0)
        elif case == "right":
            r_I1 = qe[self.nodalDOF_element_r[-1]]
            A_I1 = rodriguez(qe[self.nodalDOF_element_psi[-1]])
            return SE3(A_I1, r_I1)
        else:
            raise RuntimeError("Unsupported case chosen.")

    def relative_interpolation(self, H_IR: np.ndarray, qe: np.ndarray, xi: float):
        """Interpolation function for relative rotation vectors proposed by
        Crisfield1999 (5.7) and (5.8)."""
        # evaluate shape functions
        # TODO: They have to coincide!
        N, N_xi, _ = self.basis_functions_r(xi)
        # N, _ = self.basis_functions_psi(xi)

        # relative interpolation of local se(3) objects
        h_rel = np.zeros(6, dtype=float)
        h_rel_xi = np.zeros(6, dtype=float)

        # evaluate inverse reference SE(3) object
        H_RI = SE3inv(H_IR)

        # TODO: We have to unify DOF's for r and psi again. They can't be
        # different for the SE(3) formulation!
        for node in range(self.nnodes_element_psi):
            # nodal centerline
            r_IK_node = qe[self.nodalDOF_element_r[node]]

            # nodal rotation
            A_IK_node = rodriguez(qe[self.nodalDOF_element_psi[node]])

            # nodal SE(3) object
            H_IK_node = SE3(A_IK_node, r_IK_node)

            # relative SE(3)/ se(3) objects
            H_RK = H_RI @ H_IK_node
            h_RK = SE3log(H_RK)

            # add wheighted contribution of local se(3) object
            h_rel += N[node] * h_RK
            h_rel_xi += N_xi[node] * h_RK

        return h_rel, h_rel_xi

    def eval(self, qe, xi):
        # reference SE(3) object
        H_IR = self.reference_rotation(qe)

        # relative interpolation of se(3) nodes
        h_rel, h_rel_xi = self.relative_interpolation(H_IR, qe, xi)

        # composition of reference rotation and relative one
        H_IK = H_IR @ se3exp(h_rel)

        ###################
        # objective strains
        ###################
        T = se3tangent_map(h_rel)
        strains = T @ h_rel_xi

        # ############################################
        # # alternative computation of strain measures
        # ############################################
        # H_RK = se3exp(h_rel)

        # R_r_xi = h_rel_xi[:3]
        # R_omega_xi = h_rel_xi[3:]
        # # H_RK_xi = np.zeros((4, 4,), dtype=float)
        # # H_RK_xi[:3, :3] = ax2skew(R_omega_xi)
        # # H_RK_xi[:3, 3] = R_r_xi

        # strains_tilde = SE3inv(H_RK) @ H_RK_xi
        # # strains_tilde = H_RK_xi @ SE3inv(H_RK)
        # # strains = np.concatenate((strains_tilde[:3, 3], skew2ax(strains_tilde[:3, :3])))
        # # # # strains2 = SE3log(strains_tilde)
        # strains2 = np.concatenate((strains_tilde[:3, 3], skew2ax(strains_tilde[:3, :3])))
        # strains = strains2
        # # strains = h_rel_xi

        # # diff = strains - strains2
        # # error = np.linalg.norm(diff)
        # # print(f"error strains: {error}")
        # # print(f"error strains: {diff}")

        # #################################################################
        # # This alternative formulation works for pure bending experiments
        # #################################################################
        # strains = h_rel_xi

        # extract centerline and transformation
        A_IK = H_IK[:3, :3]
        r_OP = H_IK[:3, 3]

        # extract strains
        K_Gamma_bar = strains[:3]  # this is K_r_xi
        K_Kappa_bar = strains[3:]

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def assembler_callback(self):
        self.__M_coo()

    #########################################
    # equations of motion
    #########################################
    def M_el(self, el):
        M_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

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
        f_gyr_el = np.zeros(self.nq_element, dtype=float)

        for i in range(self.nquadrature):
            # interpoalte angular velocity
            K_Omega = np.zeros(3, dtype=float)
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

    def f_pot(self, t, q):
        f_pot = np.zeros(self.nu, dtype=float)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f_pot[elDOF] += self.f_pot_el(q[elDOF], el)
        return f_pot

    def f_pot_el(self, qe, el):
        f_pot_el = np.zeros(self.nq_element, dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # objective interpolation
            r_OP, A_IK, K_Gamma_bar, K_Kappa_bar = self.eval(qe, self.qp[el, i])

            # centerline and tangent
            K_r_OP = A_IK.T @ r_OP
            K_r_xi = K_Gamma_bar

            # axial and shear strains
            K_Gamma = K_Gamma_bar / Ji

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / Ji

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)

            #######################
            # Original formulation!
            #######################
            if self.use_K_r:
                for node in range(self.nnodes_element_r):
                    f_pot_el[self.nodalDOF_element_r[node]] -= (
                        self.N_r_xi[el, i, node] * K_n * qwi
                    )
                    f_pot_el[self.nodalDOF_element_r[node]] += (
                        self.N_r[el, i, node] * cross3(K_Kappa_bar, K_n) * qwi
                    )  # Euler term

                # for node in range(self.nnodes_element_psi):
                #     f_pot_el[self.nodalDOF_element_psi[node]] += (
                #         self.N_psi[el, i, node] * cross3(K_Kappa_bar, K_n) * qwi
                #     )

                # old one
                # f_pot_el[self.nodalDOF_element_psi[node]] += (
                #     self.N_psi[el, i, node] * cross3(K_r_xi, K_n) * qwi
                # )
                # f_pot_el[self.nodalDOF_element_psi[node]] += (
                #     # self.N_psi[el, i, node] * cross3(K_r_xi, K_n) * qwi
                #     # - self.N_psi[el, i, node] * ax2skew(K_n) @ K_r_xi * qwi
                #     -self.N_psi[el, i, node]
                #     * ax2skew(K_n)
                #     @ cross3(K_Kappa_bar, K_r_OP)
                #     * qwi
                # )  # Euler term
            else:
                # - first delta Gamma part
                for node in range(self.nnodes_element_r):
                    f_pot_el[self.nodalDOF_element_r[node]] -= (
                        self.N_r_xi[el, i, node] * A_IK @ K_n * qwi
                    )

                # - second delta Gamma part
                for node in range(self.nnodes_element_psi):
                    f_pot_el[self.nodalDOF_element_psi[node]] += (
                        self.N_psi[el, i, node] * cross3(K_r_xi, K_n) * qwi
                    )

                    # f_pot_el[self.nodalDOF_element_psi[node]] -= (
                    #     # self.N_psi[el, i, node] * cross3(K_r_xi, K_n) * qwi
                    #     # - self.N_psi[el, i, node] * ax2skew(K_n) @ K_r_xi * qwi
                    #     -self.N_psi[el, i, node]
                    #     * ax2skew(K_n)
                    #     @ cross3(K_Kappa_bar, K_r_OP)
                    #     * qwi
                    # )  # Euler term

            # - delta kappa part
            for node in range(self.nnodes_element_psi):
                f_pot_el[self.nodalDOF_element_psi[node]] -= (
                    self.N_psi_xi[el, i, node] * K_m * qwi
                )
                f_pot_el[self.nodalDOF_element_psi[node]] += (
                    self.N_psi[el, i, node] * cross3(K_Kappa_bar, K_m) * qwi
                )  # Euler term

        return f_pot_el

    def f_pot_q(self, t, q, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f_pot_q_el = self.f_pot_q_el(q[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(f_pot_q_el, (self.uDOF[elDOF], self.qDOF[elDOF]))

    def f_pot_q_el(self, qe, el):
        return approx_fprime(qe, lambda qe: self.f_pot_el(qe, el), method="2-point")
        # return approx_fprime(qe, lambda qe: self.f_pot_el(qe, el), method="3-point")

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        raise NotImplementedError
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

    # change between rotation vector and its complement in order to circumvent
    # singularities of the rotation vector
    @staticmethod
    def psi_C(psi):
        angle = norm(psi)
        if angle < pi:
            return psi
        else:
            # Ibrahimbegovic1995 after (62)
            print(f"complement rotation vector is used")
            n = int((angle + pi) / (2 * pi))
            if angle > 0:
                e = psi / angle
            else:
                e = psi.copy()
            return psi - 2 * n * pi * e

    def step_callback(self, t, q, u):
        for node in range(self.nnode_psi):
            psi_node = q[self.nodalDOF_psi[node]]
            q[self.nodalDOF_psi[node]] = TimoshenkoAxisAngle.psi_C(psi_node)
        return q, u

    # # TODO: Do we have to count the number of complement rotations?
    # # I think so, since otherwise the next singularity occurs at 4pi, etc.
    # def step_callback(self, t, q, u):
    #     for node in range(self.nnode_psi):
    #         psi_node = q[self.nodalDOF_psi[node]]
    #         angle = norm(psi_node)
    #         if angle > pi:
    #             self.use_complement[node] = True
    #     return q, u

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

    # def r_OC(self, t, q, frame_ID):
    #     # compute centerline position
    #     N, _, _ = self.basis_functions_r(frame_ID[0])
    #     r_OC = np.zeros, dtype=float)
    #     for node in range(self.nnodes_element_r):
    #         r_OC += N[node] * q[self.nodalDOF_element_r[node]]
    #     return r_OC

    # def r_OC_q(self, t, q, frame_ID):
    #     # compute centerline position
    #     N, _, _ = self.basis_functions_r(frame_ID[0])
    #     r_OC_q = np.zeros((3, self.nq_element), dtype=float)
    #     for node in range(self.nnodes_element_r):
    #         r_OC_q[:, self.nodalDOF_element_r[node]] += N[node] * np.eye(3)
    #     return r_OC_q

    # def r_OC_xi(self, t, q, frame_ID):
    #     # compute centerline position
    #     _, N_xi, _ = self.basis_functions_r(frame_ID[0])
    #     r_OC_xi = np.zeros(3, dtype=float)
    #     for node in range(self.nnodes_element_r):
    #         r_OC_xi += N_xi[node] * q[self.nodalDOF_element_r[node]]
    #     return r_OC_xi

    # def r_OC_xixi(self, t, q, frame_ID):
    #     # compute centerline position
    #     _, _, N_xixi = self.basis_functions_r(frame_ID[0])
    #     r_OC_xixi = np.zeros(3, dtype=float)
    #     for node in range(self.nnodes_element_r):
    #         r_OC_xixi += N_xixi[node] * q[self.nodalDOF_element_r[node]]
    #     return r_OC_xixi

    # def J_C(self, t, q, frame_ID):
    #     # evaluate required nodal shape functions
    #     N, _, _ = self.basis_functions_r(frame_ID[0])

    #     # interpolate centerline and axis angle contributions
    #     J_C = np.zeros((3, self.nq_element), dtype=float)
    #     for node in range(self.nnodes_element_r):
    #         J_C[:, self.nodalDOF_element_r[node]] += N[node] * np.eye(3)

    #     return J_C

    # def J_C_q(self, t, q, frame_ID):
    #     return np.zeros((3, self.nq_element, self.nq_element), dtype=float)

    ###################
    # r_OP contribution
    ###################
    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3), dtype=float):
        r, A_IK, _, _ = self.eval(q, frame_ID[0])
        return r + A_IK @ K_r_SP

    # TODO:
    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3), dtype=float):
        r_OP_q_num = approx_fprime(
            q, lambda q: self.r_OP(t, q, frame_ID, K_r_SP), method="3-point"
        )
        return r_OP_q_num

    def A_IK(self, t, q, frame_ID):
        _, A_IK, _, _ = self.eval(q, frame_ID[0])
        return A_IK

    # TODO:
    def A_IK_q(self, t, q, frame_ID):
        A_IK_q_num = approx_fprime(
            q, lambda q: self.A_IK(t, q, frame_ID), method="3-point"
        )
        return A_IK_q_num

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3), dtype=float):
        N, _, _ = self.basis_functions_r(frame_ID[0])
        _, A_IK, _, _ = self.eval(q, frame_ID[0])

        # compute angular velocity in K-frame
        K_Omega = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_psi):
            K_Omega += N[node] * u[self.nodalDOF_element_psi[node]]

        # compute centerline velocity depending on chosen formulation
        if self.use_K_r:
            K_v_C = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_r):
                K_v_C += N[node] * u[self.nodalDOF_element_r[node]]
            return A_IK @ (K_v_C + cross3(K_Omega, K_r_SP))
        else:
            v_C = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_r):
                v_C += N[node] * u[self.nodalDOF_element_r[node]]
            return v_C + A_IK @ cross3(K_Omega, K_r_SP)

    # TODO:
    def v_P_q(self, t, q, u, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        v_P_q_num = approx_fprime(
            q, lambda q: self.v_P(t, q, u, frame_ID, K_r_SP), method="3-point"
        )
        return v_P_q_num

    def J_P(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # evaluate required nodal shape functions
        N, _, _ = self.basis_functions_r(frame_ID[0])

        # transformation matrix
        _, A_IK, _, _ = self.eval(q, frame_ID[0])

        # skew symmetric matrix of K_r_SP
        K_r_SP_tilde = ax2skew(K_r_SP)

        # interpolate centerline and axis angle contributions
        J_P = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_r):
            if self.use_K_r:
                J_P[:, self.nodalDOF_element_r[node]] += N[node] * A_IK
            else:
                J_P[:, self.nodalDOF_element_r[node]] += N[node] * np.eye(
                    3, dtype=float
                )
        for node in range(self.nnodes_element_psi):
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
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # skew symmetric matrix of K_r_SP
        K_r_SP_tilde = ax2skew(K_r_SP)

        # interpolate axis angle contributions since centerline contributon is
        # zero
        J_P_q = np.zeros((3, self.nq_element, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_psi):
            nodalDOF_r = self.nodalDOF_element_r[node]
            nodalDOF_psi = self.nodalDOF_element_psi[node]
            A_IK_q = rodriguez_der(q[nodalDOF_psi])

            # centerline part
            if self.use_K_r:
                J_P_q[:, nodalDOF_r[:, None], nodalDOF_psi] += N_psi[node] * A_IK_q

            # virtual rotation part
            J_P_q[:, nodalDOF_psi[:, None], nodalDOF_psi] -= N_psi[node] * np.einsum(
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

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        raise NotImplementedError
        # compute centerline acceleration
        N, _, _ = self.basis_functions_r(frame_ID[0])
        a_C = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_r):
            a_C += N[node] * u_dot[self.nodalDOF_element_r[node]]

        # rigid body formular
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID)
        return a_C + self.A_IK(t, q, frame_ID) @ (
            cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP))
        )

    def a_P_q(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        raise NotImplementedError
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
        raise NotImplementedError
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        local = -self.A_IK(t, q, frame_ID) @ (
            ax2skew(cross3(K_Omega, K_r_SP)) + ax2skew(K_Omega) @ ax2skew(K_r_SP)
        )

        N, _ = self.basis_functions_psi(frame_ID[0])
        a_P_u = np.zeros((3, self.nq_element), dtype=float)
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
        K_Omega = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_psi):
            K_Omega += N[node] * u[self.nodalDOF_element_psi[node]]
        return K_Omega

    def K_Omega_q(self, t, q, u, frame_ID):
        return np.zeros((3, self.nq_element), dtype=float)

    def K_J_R(self, t, q, frame_ID):
        N, _ = self.basis_functions_psi(frame_ID[0])
        K_J_R = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_psi):
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
        N, _ = self.basis_functions_psi(frame_ID[0])
        K_Psi = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_psi):
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
            for node in range(self.nnodes_element_r):
                r_C += self.N_r[el, i, node] * qe[self.nodalDOF_element_r[node]]

            # compute potential value at given quadrature point
            Ve += (r_C @ force(t, qwi)) * Ji * qwi

        return Ve

    def distributed_force1D_pot(self, t, q, force):
        raise NotImplementedError
        V = 0
        for el in range(self.nelement):
            qe = q[self.elDOF[el]]
            V += self.distributed_force1D_pot_el(force, t, qe, el)
        return V

    def distributed_force1D_el(self, force, t, el):
        raise NotImplementedError
        fe = np.zeros(self.nq_element, dtype=float)
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
                x = r + radius * cos(alpha) * d2 + radius * sin(alpha) * d3
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

        Pw = np.zeros((nn_xi, nn_eta, nn_zeta, 3), dtype=float)
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
        points = np.zeros((n_patches * patch_size, dim), dtype=float)
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

        Pw = np.zeros((nn_xi, nn_eta, nn_zeta, 3), dtype=float)
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
        points = np.zeros((n_patches * patch_size, dim), dtype=float)
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
        Gamma = np.zeros(
            (self.mesh_r.nelement, self.mesh_r.nquadrature, 3), dtype=float
        )
        Kappa = np.zeros(
            (self.mesh_r.nelement, self.mesh_r.nquadrature, 3), dtype=float
        )
        if not same_shape_functions:
            d1s = np.zeros(
                (self.mesh_r.nelement, self.mesh_r.nquadrature, 3), dtype=float
            )
            d2s = np.zeros(
                (self.mesh_r.nelement, self.mesh_r.nquadrature, 3), dtype=float
            )
            d3s = np.zeros(
                (self.mesh_r.nelement, self.mesh_r.nquadrature, 3), dtype=float
            )
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
            field = np.zeros((len(Gamma_vtk), len(tmp)), dtype=float)
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
        Gamma = np.zeros(
            (self.mesh_r.nelement, self.mesh_r.nquadrature, 3), dtype=float
        )
        Kappa = np.zeros(
            (self.mesh_r.nelement, self.mesh_r.nquadrature, 3), dtype=float
        )
        if not same_shape_functions:
            d1s = np.zeros(
                (self.mesh_r.nelement, self.mesh_r.nquadrature, 3), dtype=float
            )
            d2s = np.zeros(
                (self.mesh_r.nelement, self.mesh_r.nquadrature, 3), dtype=float
            )
            d3s = np.zeros(
                (self.mesh_r.nelement, self.mesh_r.nquadrature, 3), dtype=float
            )
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
            field = np.zeros((len(Gamma_vtk), len(tmp)), dtype=float)
            for i, (K_Gamma, K_Gamma0, K_Kappa, K_Kappa0) in enumerate(
                zip(Gamma_vtk, Gamma0_vtk, Kappa_vtk, Kappa0_vtk)
            ):
                field[i] = fun(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0).reshape(-1)
            point_data.update({name: field})

        return points_r, point_data, cells_r, HigherOrderDegrees_r


class TimoshenkoAxisAngle:
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
        elif basis == "Lagrange":
            self.knot_vector_r = Node_vector(polynomial_degree_r, nelement)
            self.knot_vector_psi = Node_vector(polynomial_degree_psi, nelement)
        elif basis == "Hermite":
            # Note: This implements a cubic Hermite spline for the centerline
            #       together with a linear Lagrange axis angle vector field
            #       for the superimposed rotation w.r.t. the smallest rotation.
            polynomial_degree_r = 3
            self.knot_vector_r = HermiteNodeVector(polynomial_degree_r, nelement)
            self.polynomial_degree_psi = polynomial_degree_psi = 1
            # TODO: Enbale Lagrange shape functions again if ready.
            # self.knot_vector_psi = Node_vector(polynomial_degree_psi, nelement)
            self.knot_vector_psi = KnotVector(polynomial_degree_psi, nelement)
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        # number of degrees of freedom per node
        self.nq_node_r = nq_node_r = 3
        self.nq_node_psi = nq_node_psi = 3

        self.mesh_r = Mesh1D(
            self.knot_vector_r,
            nquadrature,
            derivative_order=2,
            basis=basis,
            dim_q=nq_node_r,
        )
        # TODO: This is ugly!
        if basis == "Hermite":
            # TODO: Enable Lagrange again if ready.
            # basis = "Lagrange"
            basis = "B-spline"
        self.mesh_psi = Mesh1D(
            self.knot_vector_psi,
            nquadrature,
            derivative_order=1,
            basis=basis,
            dim_q=nq_node_psi,
        )

        # toal number of nodes
        self.nnode_r = self.mesh_r.nnodes
        self.nnode_psi = self.mesh_psi.nnodes

        # number of nodes per element
        self.nnodes_element_r = self.mesh_r.nnodes_per_element
        self.nnodes_element_psi = self.mesh_psi.nnodes_per_element

        # total number of generalized coordinates
        self.nq_r = self.mesh_r.nq
        self.nq_psi = self.mesh_psi.nq
        self.nq = self.nq_r + self.nq_psi  # total number of generalized coordinates
        self.nu = self.nq  # total number of generalized velocities

        # number of generalized coordiantes per element
        self.nq_element_r = self.mesh_r.nq_per_element
        self.nq_element_psi = self.mesh_psi.nq_per_element
        self.nq_element = self.nu_element = self.nq_element_r + self.nq_element_psi

        # global element connectivity
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_psi = self.mesh_psi.elDOF + self.nq_r

        # global nodal
        self.nodalDOF_r = self.mesh_r.nodalDOF
        self.nodalDOF_psi = self.mesh_psi.nodalDOF + self.nq_r

        # nodal connectivity on element level
        self.nodalDOF_element_r = self.mesh_r.nodalDOF_element
        self.nodalDOF_element_psi = self.mesh_psi.nodalDOF_element + self.nq_element_r

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nelement, self.nq_element), dtype=int)
        for el in range(nelement):
            self.elDOF[el, : self.nq_element_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_element_r :] = self.elDOF_psi[el]

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

        # reference rotation for relative rotation proposed by Crisfield1999 (5.8)
        self.node_A = int(0.5 * (self.nnodes_element_psi + 1)) - 1
        self.node_B = int(0.5 * (self.nnodes_element_psi + 2)) - 1

        # precompute values of the reference configuration in order to save computation time
        # J in Harsch2020b (5)
        self.J = np.zeros((nelement, nquadrature))
        # dilatation and shear strains of the reference configuration
        self.K_Gamma0 = np.zeros((nelement, nquadrature, 3))
        # curvature of the reference configuration
        self.K_Kappa0 = np.zeros((nelement, nquadrature, 3))

        for el in range(nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(nquadrature):
                # interpolate tangent vector
                r_xi = np.zeros(3)
                for node in range(self.nnodes_element_r):
                    r_xi += self.N_r_xi[el, i, node] * qe[self.nodalDOF_element_r[node]]

                # length of reference tangential vector
                Ji = norm(r_xi)

                # evaluate strain measures and other quantities depending on chosen formulation
                r_xi, A_IK, K_Kappa_bar = self.eval(qe, el, i)

                # axial and shear strains
                K_Gamma = A_IK.T @ (r_xi / Ji)

                # torsional and flexural strains
                K_Kappa = K_Kappa_bar / Ji

                # safe precomputed quantities for later
                self.J[el, i] = Ji
                self.K_Gamma0[el, i] = K_Gamma
                self.K_Kappa0[el, i] = K_Kappa

    @staticmethod
    def straight_configuration(
        polynomial_degree_r,
        polynomial_degree_psi,
        nelement,
        L,
        greville_abscissae=True,
        r_OP=np.zeros(3),
        A_IK=np.eye(3),
        basis="B-spline",
    ):
        if basis == "B-spline":
            nn_r = polynomial_degree_r + nelement
            nn_psi = polynomial_degree_psi + nelement
        elif basis == "Lagrange":
            nn_r = polynomial_degree_r * nelement + 1
            nn_psi = polynomial_degree_psi * nelement + 1
        elif basis == "Hermite":
            polynomial_degree_r = 3
            nn_r = nelement + 1
            polynomial_degree_psi = 1
            # TODO:
            nn_psi = polynomial_degree_psi + nelement  # B-spline
            # nn_psi = polynomial_degree_psi * nelement + 1 # Lagrange
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        if (basis == "B-spline") or (basis == "Lagrange"):
            x0 = np.linspace(0, L, num=nn_r)
            y0 = np.zeros(nn_r)
            z0 = np.zeros(nn_r)
            if greville_abscissae and basis == "B-spline":
                kv = KnotVector.uniform(polynomial_degree_r, nelement)
                for i in range(nn_r):
                    x0[i] = np.sum(kv[i + 1 : i + polynomial_degree_r + 1])
                x0 = x0 * L / polynomial_degree_r

            r0 = np.vstack((x0, y0, z0))
            for i in range(nn_r):
                r0[:, i] = r_OP + A_IK @ r0[:, i]

        elif basis == "Hermite":
            xis = np.linspace(0, 1, num=nn_r)
            r0 = np.zeros((6, nn_r))
            t0 = A_IK @ (L * e1)
            for i, xi in enumerate(xis):
                ri = r_OP + xi * t0
                r0[:3, i] = ri
                r0[3:, i] = t0

        # reshape generalized coordinates to nodal ordering
        q_r = r0.reshape(-1, order="F")

        # TODO: Relative interpolation case!
        # we have to extract the rotation vector from the given rotation matrix
        # and set its value for each node
        psi = rodriguez_inv(A_IK)
        q_psi = np.tile(psi, nn_psi)

        return np.concatenate([q_r, q_psi])

    def element_number(self, xi):
        # note the elements coincide for both meshes!
        return self.knot_vector_r.element_number(xi)[0]

    def reference_rotation(self, qe: np.ndarray):
        """Reference rotation proposed by Crisfield1999 (5.8)."""
        A_0I = rodriguez(qe[self.nodalDOF_element_psi[self.node_A]])
        A_0J = rodriguez(qe[self.nodalDOF_element_psi[self.node_B]])
        A_IJ = A_0I.T @ A_0J  # Crisfield1999 (5.8)
        phi_IJ = rodriguez_inv(A_IJ)
        return A_0I @ rodriguez(0.5 * phi_IJ)

    def relative_interpolation(
        self, A_IR: np.ndarray, qe: np.ndarray, el: int, qp: int
    ):
        """Interpolation function for relative rotation vectors proposed by
        Crisfield1999 (5.7) and (5.8)."""
        # relative interpolation of local rotation vectors
        psi_rel = np.zeros(3)
        psi_rel_xi = np.zeros(3)
        for node in range(self.nnodes_element_psi):
            # nodal axis angle vector
            psi_node = qe[self.nodalDOF_element_psi[node]]

            # nodal rotation
            A_IK_node = rodriguez(psi_node)

            # relative rotation of each node and corresponding
            # rotation vector
            A_RK_node = A_IR.T @ A_IK_node
            psi_RK_node = rodriguez_inv(A_RK_node)

            # add wheighted contribution of local rotation
            psi_rel += self.N_psi[el, qp, node] * psi_RK_node
            psi_rel_xi += self.N_psi_xi[el, qp, node] * psi_RK_node

        return psi_rel, psi_rel_xi

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

    def eval(self, qe, el, qp):
        # interpolate tangent vector
        r_xi = np.zeros(3)
        for node in range(self.nnodes_element_r):
            r_xi += self.N_r_xi[el, qp, node] * qe[self.nodalDOF_element_r[node]]

        # reference rotation, see. Crisfield1999 (5.8)
        A_IR = self.reference_rotation(qe)

        # relative interpolation of the rotation vector and it first derivative
        psi_rel, psi_rel_xi = self.relative_interpolation(A_IR, qe, el, qp)

        # objective rotation
        A_IK = A_IR @ rodriguez(psi_rel)

        # objective curvature
        T = tangent_map(psi_rel)
        K_Kappa_bar = T @ psi_rel_xi

        return r_xi, A_IK, K_Kappa_bar

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

            # evaluate strain measures and other quantities depending on chosen formulation
            r_xi, A_IK, K_Kappa_bar = self.eval(qe, el, i)

            # axial and shear strains
            K_Gamma = A_IK.T @ (r_xi / Ji)

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / Ji

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

            # evaluate strain measures and other quantities depending on chosen formulation
            r_xi, A_IK, K_Kappa_bar = self.eval(qe, el, i)

            # axial and shear strains
            K_Gamma = A_IK.T @ (r_xi / Ji)

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / Ji

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
                    self.N_psi[el, i, node] * cross3(K_Kappa_bar, K_m) * qwi
                )  # Euler term

        return f_pot_el

    def f_pot_q(self, t, q, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f_pot_q_el = self.f_pot_q_el(q[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(f_pot_q_el, (self.uDOF[elDOF], self.qDOF[elDOF]))

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

    # change between rotation vector nad its complement in order to circumvent
    # singularities of the rotation vector
    @staticmethod
    def psi_C(psi):
        angle = norm(psi)
        if angle < pi:
            return psi
        else:
            # Ibrahimbegovic1995 after (62)
            print(f"complement rotation vector is used")
            n = int((angle + pi) / (2 * pi))
            if angle > 0:
                e = psi / angle
            else:
                e = psi.copy()
            return psi - 2 * n * pi * e

    def step_callback(self, t, q, u):
        for node in range(self.nnode_psi):
            psi_node = q[self.nodalDOF_psi[node]]
            q[self.nodalDOF_psi[node]] = TimoshenkoAxisAngle.psi_C(psi_node)
        return q, u

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

    def r_OC(self, t, q, frame_ID):
        # compute centerline position
        N, _, _ = self.basis_functions_r(frame_ID[0])
        r_OC = np.zeros(3)
        for node in range(self.nnodes_element_r):
            r_OC += N[node] * q[self.nodalDOF_element_r[node]]
        return r_OC

    def r_OC_q(self, t, q, frame_ID):
        # compute centerline position
        N, _, _ = self.basis_functions_r(frame_ID[0])
        r_OC_q = np.zeros((3, self.nq_element))
        for node in range(self.nnodes_element_r):
            r_OC_q[:, self.nodalDOF_element_r[node]] += N[node] * np.eye(3)
        return r_OC_q

    def r_OC_xi(self, t, q, frame_ID):
        # compute centerline position
        _, N_xi, _ = self.basis_functions_r(frame_ID[0])
        r_OC_xi = np.zeros(3)
        for node in range(self.nnodes_element_r):
            r_OC_xi += N_xi[node] * q[self.nodalDOF_element_r[node]]
        return r_OC_xi

    def r_OC_xixi(self, t, q, frame_ID):
        # compute centerline position
        _, _, N_xixi = self.basis_functions_r(frame_ID[0])
        r_OC_xixi = np.zeros(3)
        for node in range(self.nnodes_element_r):
            r_OC_xixi += N_xixi[node] * q[self.nodalDOF_element_r[node]]
        return r_OC_xixi

    def J_C(self, t, q, frame_ID):
        # evaluate required nodal shape functions
        N, _, _ = self.basis_functions_r(frame_ID[0])

        # interpolate centerline and axis angle contributions
        J_C = np.zeros((3, self.nq_element))
        for node in range(self.nnodes_element_r):
            J_C[:, self.nodalDOF_element_r[node]] += N[node] * np.eye(3)

        return J_C

    def J_C_q(self, t, q, frame_ID):
        return np.zeros((3, self.nq_element, self.nq_element))

    ###################
    # r_OP contribution
    ###################
    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # compute centerline position
        N, _, _ = self.basis_functions_r(frame_ID[0])
        r_OC = np.zeros(3)
        for node in range(self.nnodes_element_r):
            r_OC += N[node] * q[self.nodalDOF_element_r[node]]

        # rigid body formular
        return r_OC + self.A_IK(t, q, frame_ID) @ K_r_SP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # compute centerline derivative
        N, _, _ = self.basis_functions_r(frame_ID[0])
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
        N, _ = self.basis_functions_psi(frame_ID[0])

        # reference rotation, cf. Crisfield1999 (5.8)
        # for convenience we use the most left nodal rotation
        # TODO: We can generalize this to a midway reference as done by Crisfield1999
        A_IR = rodriguez(q[self.nodalDOF_element_psi[0]])

        # relative interpolation of local rotation vectors
        psi_rel = np.zeros(3)
        for node in range(self.nnodes_element_psi):
            # nodal axis angle vector
            psi_node = q[self.nodalDOF_element_psi[node]]

            # nodal rotation
            A_IK_node = rodriguez(psi_node)

            # relative rotation of each node and corresponding
            # rotation vector
            A_RK_node = A_IR.T @ A_IK_node
            psi_RK_node = rodriguez_inv(A_RK_node)

            # add wheighted contribution of local rotation
            psi_rel += N[node] * psi_RK_node

        # objective rotation
        A_IK = A_IR @ rodriguez(psi_rel)

        return A_IK

    def A_IK_q(self, t, q, frame_ID):
        # # interpolate nodal derivative of the rotation matrix
        # N, _ = self.basis_functions_psi(frame_ID[0])
        # A_IK_q = np.zeros((3, 3, self.nq_element))
        # for node in range(self.nnodes_element_psi):
        #     nodalDOF_psi = self.nodalDOF_element_psi[node]
        #     A_IK_q[:, :, nodalDOF_psi] += N[node] * rodriguez_der(q[nodalDOF_psi])
        # return A_IK_q

        A_IK_q_num = approx_fprime(
            q, lambda q: self.A_IK(t, q, frame_ID), method="3-point"
        )
        # diff = A_IK_q - A_IK_q_num
        # error = np.linalg.norm(diff)
        # print(f"error A_IK_q: {error}")
        return A_IK_q_num

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3)):
        # compute centerline velocity
        N, _, _ = self.basis_functions_r(frame_ID[0])
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
        N_r, _, _ = self.basis_functions_r(frame_ID[0])
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
        N, _, _ = self.basis_functions_r(frame_ID[0])
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
        if self.basis == "Hermite":
            r = np.zeros((3, int(self.nnode_r / 2)))
            idx = 0
            for node, nodalDOF in enumerate(self.nodalDOF_r):
                if node % 2 == 0:
                    r[:, idx] = q_body[nodalDOF]
                    idx += 1
            return r
        else:
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
                x = r + radius * cos(alpha) * d2 + radius * sin(alpha) * d3
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


class TimoshenkoQuaternion:
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
        self.quadrature = "Gauss"  # default for all meshes
        if basis == "B-spline":
            self.knot_vector_r = KnotVector(polynomial_degree_r, nelement)
            self.knot_vector_psi = KnotVector(polynomial_degree_psi, nelement)
        elif basis == "Lagrange":
            self.knot_vector_r = Node_vector(polynomial_degree_r, nelement)
            self.knot_vector_psi = Node_vector(polynomial_degree_psi, nelement)
        elif basis == "Hermite":
            # Note: This implements a cubic Hermite spline for the centerline
            #       together with a linear Lagrange axis angle vector field
            #       for the superimposed rotation w.r.t. the smallest rotation.
            polynomial_degree_r = 3
            self.knot_vector_r = HermiteNodeVector(polynomial_degree_r, nelement)
            self.polynomial_degree_psi = polynomial_degree_psi = 1
            # TODO: Enbale Lagrange shape functions again if ready.
            # self.knot_vector_psi = Node_vector(polynomial_degree_psi, nelement)
            self.knot_vector_psi = KnotVector(polynomial_degree_psi, nelement)
            self.quadrature = "Lobatto"
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        # number of degrees of freedom per node
        self.nq_node_r = nq_node_r = 3
        self.nq_node_psi = nq_node_psi = 4
        self.nu_node_psi = nu_node_psi = 3
        # self.nla_g_node = nla_g_node = 1

        # centerline mesh
        self.mesh_r = Mesh1D(
            self.knot_vector_r,
            nquadrature,
            derivative_order=2,
            basis=basis,
            dim_q=nq_node_r,
            quadrature=self.quadrature,
        )

        # quaternion mesh (ugly modificatin for Hermite spline)
        # TODO: This is ugly!
        if basis == "Hermite":
            # TODO: Enable Lagrange again if ready.
            # basis = "Lagrange"
            basis = "B-spline"
        self.mesh_psi = Mesh1D(
            self.knot_vector_psi,
            nquadrature,
            derivative_order=1,
            basis=basis,
            dim_q=nq_node_psi,
            dim_u=nu_node_psi,
            quadrature=self.quadrature,
        )
        # self.mesh_la_g = Mesh1D(
        #     self.knot_vector_psi,
        #     nquadrature,
        #     derivative_order=1,
        #     basis=basis,
        #     dim_q=nla_g_node,
        # )

        # TODO: Move on here for bilateral constraints for quaternion nodes

        # toal number of nodes
        self.nnode_r = self.mesh_r.nnodes
        self.nnode_psi = self.mesh_psi.nnodes

        # number of nodes per element
        self.nnodes_element_r = self.mesh_r.nnodes_per_element
        self.nnodes_element_psi = self.mesh_psi.nnodes_per_element

        # total number of generalized coordinates
        self.nq_r = self.mesh_r.nq
        self.nq_psi = self.mesh_psi.nq
        self.nq = self.nq_r + self.nq_psi  # total number of generalized coordinates
        self.nu_r = self.mesh_r.nu
        self.nu_psi = self.mesh_psi.nu
        self.nu = self.nu_r + self.nu_psi  # total number of generalized velocities
        # self.nla_g = self.mesh_la_g.nq  # total number of Lagrange mutipliers
        self.nla_S = self.mesh_psi.nnodes  # total number of Lagrange mutipliers

        # number of generalized coordiantes per element
        self.nq_element_r = self.mesh_r.nq_per_element
        self.nq_element_psi = self.mesh_psi.nq_per_element
        self.nq_element = self.nq_element_r + self.nq_element_psi
        self.nu_element_r = self.mesh_r.nu_per_element
        self.nu_element_psi = self.mesh_psi.nu_per_element
        self.nu_element = self.nu_element_r + self.nu_element_psi
        # self.nla_g_element = self.mesh_la_g.nq_per_element

        # global element connectivity
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_psi = self.mesh_psi.elDOF + self.nq_r
        self.elDOF_u_r = self.mesh_r.elDOF_u
        self.elDOF_u_psi = self.mesh_psi.elDOF_u + self.nu_r

        # global nodal
        self.nodalDOF_r = self.mesh_r.nodalDOF
        self.nodalDOF_psi = self.mesh_psi.nodalDOF + self.nq_r
        self.nodalDOF_u_r = self.mesh_r.nodalDOF_u
        self.nodalDOF_u_psi = self.mesh_psi.nodalDOF_u + self.nu_r

        # nodal connectivity on element level
        self.nodalDOF_element_r = self.mesh_r.nodalDOF_element
        self.nodalDOF_element_psi = self.mesh_psi.nodalDOF_element + self.nq_element_r
        self.nodalDOF_element_u_r = self.mesh_r.nodalDOF_element_u
        self.nodalDOF_element_u_psi = (
            self.mesh_psi.nodalDOF_element_u + self.nu_element_r
        )

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nelement, self.nq_element), dtype=int)
        self.elDOF_u = np.zeros((nelement, self.nu_element), dtype=int)
        for el in range(nelement):
            self.elDOF[el, : self.nq_element_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_element_r :] = self.elDOF_psi[el]
            self.elDOF_u[el, : self.nu_element_r] = self.elDOF_u_r[el]
            self.elDOF_u[el, self.nu_element_r :] = self.elDOF_u_psi[el]

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
        self.la_S0 = np.zeros(self.nla_S)

        # evaluate shape functions at specific xi
        self.basis_functions_r = self.mesh_r.eval_basis
        self.basis_functions_psi = self.mesh_psi.eval_basis

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q  # reference configuration
        self.q0 = Q.copy() if q0 is None else q0  # initial configuration
        self.u0 = np.zeros(self.nu) if u0 is None else u0  # initial velocities

        # reference rotation for relative rotation proposed by Crisfield1999 (5.8)
        self.node_A = int(0.5 * (self.nnodes_element_psi + 1)) - 1
        self.node_B = int(0.5 * (self.nnodes_element_psi + 2)) - 1

        # precompute values of the reference configuration in order to save computation time
        # J in Harsch2020b (5)
        self.J = np.zeros((nelement, nquadrature))
        # dilatation and shear strains of the reference configuration
        self.K_Gamma0 = np.zeros((nelement, nquadrature, 3))
        # curvature of the reference configuration
        self.K_Kappa0 = np.zeros((nelement, nquadrature, 3))

        for el in range(nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(nquadrature):
                # interpolate tangent vector
                r_xi = np.zeros(3)
                for node in range(self.nnodes_element_r):
                    r_xi += self.N_r_xi[el, i, node] * qe[self.nodalDOF_element_r[node]]

                # length of reference tangential vector
                Ji = norm(r_xi)

                # evaluate strain measures and other quantities depending on chosen formulation
                r_xi, A_IK, K_Kappa_bar = self.eval(qe, el, i)

                # axial and shear strains
                K_Gamma = A_IK.T @ (r_xi / Ji)

                # torsional and flexural strains (formulation in skew coordinates,
                # see Eugster2014c)
                d1, d2, d3 = A_IK.T
                d = d1 @ cross3(d2, d3)
                dd = 1 / (d * Ji)
                K_Kappa = dd * K_Kappa_bar

                # safe precomputed quantities for later
                self.J[el, i] = Ji
                self.K_Gamma0[el, i] = K_Gamma
                self.K_Kappa0[el, i] = K_Kappa

    @staticmethod
    def straight_configuration(
        polynomial_degree_r,
        polynomial_degree_psi,
        nelement,
        L,
        greville_abscissae=True,
        r_OP=np.zeros(3),
        A_IK=np.eye(3),
        basis="B-spline",
    ):
        if basis == "B-spline":
            nn_r = polynomial_degree_r + nelement
            nn_psi = polynomial_degree_psi + nelement
        elif basis == "Lagrange":
            nn_r = polynomial_degree_r * nelement + 1
            nn_psi = polynomial_degree_psi * nelement + 1
        elif basis == "Hermite":
            polynomial_degree_r = 3
            nn_r = nelement + 1
            polynomial_degree_psi = 1
            # TODO:
            nn_psi = polynomial_degree_psi + nelement  # B-spline
            # nn_psi = polynomial_degree_psi * nelement + 1 # Lagrange
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        if (basis == "B-spline") or (basis == "Lagrange"):
            x0 = np.linspace(0, L, num=nn_r)
            y0 = np.zeros(nn_r)
            z0 = np.zeros(nn_r)
            if greville_abscissae and basis == "B-spline":
                kv = KnotVector.uniform(polynomial_degree_r, nelement)
                for i in range(nn_r):
                    x0[i] = np.sum(kv[i + 1 : i + polynomial_degree_r + 1])
                x0 = x0 * L / polynomial_degree_r

            r0 = np.vstack((x0, y0, z0))
            for i in range(nn_r):
                r0[:, i] = r_OP + A_IK @ r0[:, i]

        elif basis == "Hermite":
            xis = np.linspace(0, 1, num=nn_r)
            r0 = np.zeros((6, nn_r))
            t0 = A_IK @ (L * e1)
            for i, xi in enumerate(xis):
                ri = r_OP + xi * t0
                r0[:3, i] = ri
                r0[3:, i] = t0

        # reshape generalized coordinates to nodal ordering
        q_r = r0.reshape(-1, order="F")

        # we have to extract the rotation vector from the given rotation matrix
        # and set its value for each node
        # psi = rodriguez_inv(A_IK)
        p = Rotor.fromMatrix(A_IK)()
        q_psi = np.tile(p, nn_psi)

        return np.concatenate([q_r, q_psi])

    def element_number(self, xi):
        # note the elements coincide for both meshes!
        return self.knot_vector_r.element_number(xi)[0]

    def reference_rotation(self, qe: np.ndarray):
        """Reference rotation proposed by Crisfield1999 (5.8) applied on nodal
        quaternions."""
        A_0I = Rotor(qe[self.nodalDOF_element_psi[self.node_A]]).toMatrix()
        A_0J = Rotor(qe[self.nodalDOF_element_psi[self.node_B]]).toMatrix()
        A_IJ = A_0I.T @ A_0J  # Crisfield1999 (5.8)
        phi_IJ = rodriguez_inv(A_IJ)
        return A_0I @ rodriguez(0.5 * phi_IJ)

    def relative_interpolation(
        self, A_IR: np.ndarray, qe: np.ndarray, el: int, qp: int
    ):
        """Interpolation function for relative rotation vectors proposed by
        Crisfield1999 (5.7) and (5.8) applied on nodal quaternions."""
        # relative interpolation of local rotation vectors
        psi_rel = np.zeros(3)
        psi_rel_xi = np.zeros(3)
        for node in range(self.nnodes_element_psi):
            # nodal axis angle vector
            psi_node = qe[self.nodalDOF_element_psi[node]]

            # nodal rotation
            A_IK_node = Rotor(psi_node).toMatrix()

            # relative rotation of each node and corresponding
            # rotation vector
            A_RK_node = A_IR.T @ A_IK_node
            psi_RK_node = rodriguez_inv(A_RK_node)

            # add wheighted contribution of local rotation
            psi_rel += self.N_psi[el, qp, node] * psi_RK_node
            psi_rel_xi += self.N_psi_xi[el, qp, node] * psi_RK_node

        return psi_rel, psi_rel_xi

    def assembler_callback(self):
        self.__M_coo()

    #############################################
    # bilateral constraints for quaternion length
    #############################################
    def g_S(self, t, q):
        g_S = np.zeros(self.nla_S)
        for node in range(self.nnode_psi):
            psi_node = q[self.nodalDOF_psi[node]]
            g_S[node] = psi_node @ psi_node - 1.0
        return g_S

    def g_S_q(self, t, q, coo):
        for node in range(self.nnode_psi):
            nodalDOF = self.nodalDOF_psi[node]
            psi_node = q[nodalDOF]
            coo.extend(
                np.array([2 * psi_node]), (self.la_SDOF[node], self.qDOF[nodalDOF])
            )

    # normalization of the nodal quaternions length
    def step_callback(self, t, q, u):
        for node in range(self.nnode_psi):
            psi_node = q[self.nodalDOF_psi[node]]
            q[self.nodalDOF_psi[node]] = psi_node / norm(psi_node)
        return q, u

    # # TODO:
    # def W_S(self, t, q, coo):
    #     pass

    # def Wla_S_q(self, t, q, la_S, coo):
    #     pass

    #########################################
    # equations of motion
    #########################################
    def M_el(self, el):
        M_el = np.zeros((self.nu_element, self.nu_element))

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
                nodalDOF_a = self.nodalDOF_element_u_psi[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_u_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi * (
                        self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
                    )

        return M_el

    def __M_coo(self):
        self.__M = Coo((self.nu, self.nu))
        for el in range(self.nelement):
            # extract element degrees of freedom
            elDOF_u = self.elDOF_u[el]

            # sparse assemble element mass matrix
            self.__M.extend(self.M_el(el), (self.uDOF[elDOF_u], self.uDOF[elDOF_u]))

    def M(self, t, q, coo):
        coo.extend_sparse(self.__M)

    def f_gyr_el(self, t, qe, ue, el):
        f_gyr_el = np.zeros(self.nu_element)

        for i in range(self.nquadrature):
            # interpoalte angular velocity
            K_Omega = np.zeros(3)
            for node in range(self.nnodes_element_psi):
                K_Omega += (
                    self.N_psi[el, i, node] * ue[self.nodalDOF_element_u_psi[node]]
                )

            # vector of gyroscopic forces
            f_gyr_el_psi = (
                cross3(K_Omega, self.I_rho0 @ K_Omega) * self.J[el, i] * self.qw[el, i]
            )

            # multiply vector of gyroscopic forces with nodal virtual rotations
            for node in range(self.nnodes_element_psi):
                f_gyr_el[self.nodalDOF_element_u_psi[node]] += (
                    self.N_psi[el, i, node] * f_gyr_el_psi
                )

        return f_gyr_el

    def f_gyr(self, t, q, u):
        f_gyr = np.zeros(self.nu)
        for el in range(self.nelement):
            elDOF_q = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            f_gyr[elDOF_u] += self.f_gyr_el(t, q[elDOF_q], u[elDOF_u], el)
        return f_gyr

    def f_gyr_u_el(self, t, qe, ue, el):
        f_gyr_u_el = np.zeros((self.nu_element, self.nu_element))

        for i in range(self.nquadrature):
            # interpoalte angular velocity
            K_Omega = np.zeros(3)
            for node in range(self.nnodes_element_psi):
                K_Omega += (
                    self.N_psi[el, i, node] * ue[self.nodalDOF_element_u_psi[node]]
                )

            # derivative of vector of gyroscopic forces
            f_gyr_u_el_psi = (
                ((ax2skew(K_Omega) @ self.I_rho0 - ax2skew(self.I_rho0 @ K_Omega)))
                * self.J[el, i]
                * self.qw[el, i]
            )

            # multiply derivative of gyroscopic force vector with nodal virtual rotations
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_u_psi[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_u_psi[node_b]
                    f_gyr_u_el[nodalDOF_a[:, None], nodalDOF_b] += f_gyr_u_el_psi * (
                        self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
                    )

        return f_gyr_u_el

    def f_gyr_u(self, t, q, u, coo):
        for el in range(self.nelement):
            elDOF_q = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            f_gyr_u_el = self.f_gyr_u_el(t, q[elDOF_q], u[elDOF_u], el)
            coo.extend(f_gyr_u_el, (self.uDOF[elDOF_u], self.uDOF[elDOF_u]))

    def eval(self, qe, el, qp):
        # interpolate tangent vector
        r_xi = np.zeros(3)
        for node in range(self.nnodes_element_r):
            r_xi += self.N_r_xi[el, qp, node] * qe[self.nodalDOF_element_r[node]]

        # reference rotation, see. Crisfield1999 (5.8)
        A_IR = self.reference_rotation(qe)

        # relative interpolation of the rotation vector and it first derivative
        psi_rel, psi_rel_xi = self.relative_interpolation(A_IR, qe, el, qp)

        # objective rotation
        A_IK = A_IR @ rodriguez(psi_rel)

        # objective curvature
        T = tangent_map(psi_rel)
        K_Kappa_bar = T @ psi_rel_xi

        return r_xi, A_IK, K_Kappa_bar

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

            # evaluate strain measures and other quantities depending on chosen formulation
            r_xi, A_IK, K_Kappa_bar = self.eval(qe, el, i)

            # axial and shear strains
            K_Gamma = A_IK.T @ (r_xi / Ji)

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / Ji

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
            elDOF_q = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            f_pot[elDOF_u] += self.f_pot_el(q[elDOF_q], el)
        return f_pot

    def f_pot_el(self, qe, el):
        f_pot_el = np.zeros(self.nu_element)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # evaluate strain measures and other quantities depending on chosen formulation
            r_xi, A_IK, K_Kappa_bar = self.eval(qe, el, i)

            # axial and shear strains
            K_Gamma = A_IK.T @ (r_xi / Ji)

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / Ji

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
                f_pot_el[self.nodalDOF_element_u_psi[node]] += (
                    self.N_psi[el, i, node] * cross3(A_IK.T @ r_xi, K_n) * qwi
                )

            # - delta kappa part
            for node in range(self.nnodes_element_psi):
                f_pot_el[self.nodalDOF_element_u_psi[node]] -= (
                    self.N_psi_xi[el, i, node] * K_m * qwi
                )
                f_pot_el[self.nodalDOF_element_u_psi[node]] += (
                    self.N_psi[el, i, node] * cross3(K_Kappa_bar, K_m) * qwi
                )  # Euler term

        return f_pot_el

    def f_pot_q(self, t, q, coo):
        for el in range(self.nelement):
            elDOF_q = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            f_pot_q_el = self.f_pot_q_el(q[elDOF_q], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(f_pot_q_el, (self.uDOF[elDOF_u], self.qDOF[elDOF_q]))

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
        q_dot = np.zeros(self.nq)

        # centerline part
        q_dot[: self.nq_r] = u[: self.nu_r]
        # for node in range(self.nnode_r):
        #     q_dot[self.nodalDOF_r[node]] = u[self.nodalDOF_u_r[node]]

        # correct quaternion part
        for node in range(self.nnode_psi):
            nodalDOF_q_psi = self.nodalDOF_psi[node]
            nodalDOF_u_psi = self.nodalDOF_u_psi[node]

            psi = q[nodalDOF_q_psi]
            omega = u[nodalDOF_u_psi]
            Q = quat2mat(psi) / (2.0 * psi @ psi)
            psi_dot = Q[:, 1:] @ omega
            q_dot[nodalDOF_q_psi] = psi_dot

        return q_dot

    def B(self, t, q, coo):
        # trivial kinematic equation for centerline
        coo.extend_diag(
            np.ones(self.nq_r), (self.qDOF[: self.nq_r], self.uDOF[: self.nu_r])
        )

        # quaternion part
        for node in range(self.nnode_psi):
            nodalDOF_q_psi = self.nodalDOF_psi[node]
            nodalDOF_u_psi = self.nodalDOF_u_psi[node]

            psi = q[nodalDOF_q_psi]
            Q = quat2mat(psi) / (2.0 * psi @ psi)

            coo.extend(
                Q[:, 1:],
                (self.qDOF[nodalDOF_q_psi], self.uDOF[nodalDOF_u_psi]),
            )

    def q_ddot(self, t, q, u, u_dot):
        q_ddot = np.zeros(self.nq)

        # centerline part
        q_ddot[: self.nq_r] = u_dot[: self.nu_r]
        # for node in range(self.nnode_r):
        #     q_ddot[self.nodalDOF_r[node]] = u_dot[self.nodalDOF_u_r[node]]

        # correct quaternion part
        for node in range(self.nnode_psi):
            nodalDOF_q_psi = self.nodalDOF_psi[node]
            nodalDOF_u_psi = self.nodalDOF_u_psi[node]

            psi = q[nodalDOF_q_psi]
            psi2 = psi @ psi
            Q = quat2mat(psi) / (2 * psi2)
            Q_p = quat2mat_p(psi) / (2 * psi2) - np.einsum(
                "ij,k->ijk", quat2mat(psi), psi / (psi2**2)
            )

            omega = u[nodalDOF_u_psi]
            omega_dot = u_dot[nodalDOF_u_psi]
            psi_dot = Q[:, 1:] @ omega

            q_ddot[nodalDOF_q_psi] = Q[:, 1:] @ omega_dot + np.einsum(
                "ijk,k,j->i", Q_p[:, 1:, :], psi_dot, u[3:]
            )

        return q_ddot

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def qDOF_P(self, frame_ID):
        xi = frame_ID[0]
        el = self.element_number(xi)
        return self.elDOF[el]

    def uDOF_P(self, frame_ID):
        xi = frame_ID[0]
        el = self.element_number(xi)
        return self.elDOF_u[el]

    def r_OC(self, t, q, frame_ID):
        # compute centerline position
        N, _, _ = self.basis_functions_r(frame_ID[0])
        r_OC = np.zeros(3)
        for node in range(self.nnodes_element_r):
            r_OC += N[node] * q[self.nodalDOF_element_r[node]]
        return r_OC

    def r_OC_q(self, t, q, frame_ID):
        # compute centerline position
        N, _, _ = self.basis_functions_r(frame_ID[0])
        r_OC_q = np.zeros((3, self.nq_element))
        for node in range(self.nnodes_element_r):
            r_OC_q[:, self.nodalDOF_element_r[node]] += N[node] * np.eye(3)
        return r_OC_q

    def r_OC_xi(self, t, q, frame_ID):
        # compute centerline position
        _, N_xi, _ = self.basis_functions_r(frame_ID[0])
        r_OC_xi = np.zeros(3)
        for node in range(self.nnodes_element_r):
            r_OC_xi += N_xi[node] * q[self.nodalDOF_element_r[node]]
        return r_OC_xi

    def r_OC_xixi(self, t, q, frame_ID):
        # compute centerline position
        _, _, N_xixi = self.basis_functions_r(frame_ID[0])
        r_OC_xixi = np.zeros(3)
        for node in range(self.nnodes_element_r):
            r_OC_xixi += N_xixi[node] * q[self.nodalDOF_element_r[node]]
        return r_OC_xixi

    def J_C(self, t, q, frame_ID):
        # evaluate required nodal shape functions
        N, _, _ = self.basis_functions_r(frame_ID[0])

        # interpolate centerline and axis angle contributions
        J_C = np.zeros((3, self.nq_element))
        for node in range(self.nnodes_element_r):
            J_C[:, self.nodalDOF_element_r[node]] += N[node] * np.eye(3)

        return J_C

    def J_C_q(self, t, q, frame_ID):
        return np.zeros((3, self.nq_element, self.nq_element))

    ###################
    # r_OP contribution
    ###################
    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # compute centerline position
        N, _, _ = self.basis_functions_r(frame_ID[0])
        r_OC = np.zeros(3)
        for node in range(self.nnodes_element_r):
            r_OC += N[node] * q[self.nodalDOF_element_r[node]]

        # rigid body formular
        return r_OC + self.A_IK(t, q, frame_ID) @ K_r_SP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # compute centerline derivative
        N, _, _ = self.basis_functions_r(frame_ID[0])
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
        N, _ = self.basis_functions_psi(frame_ID[0])

        # reference rotation, see. Crisfield1999 (5.8)
        node_I = int(0.5 * (self.nnodes_element_psi + 1)) - 1
        node_J = int(0.5 * (self.nnodes_element_psi + 2)) - 1
        c = 0.5
        A_0I = Rotor(q[self.nodalDOF_element_psi[node_I]]).toMatrix()
        A_0J = Rotor(q[self.nodalDOF_element_psi[node_J]]).toMatrix()
        A_IJ = A_0I.T @ A_0J  # Crisfield1999 (5.8)
        phi_IJ = rodriguez_inv(A_IJ)
        # phi_IJ = Rotor.fromMatrix(A_IJ)
        A_IR = A_0I @ rodriguez(c * phi_IJ)
        # A_IR = A_0I @ (phi_IJ * c).toMatrix()

        # relative interpolation of local rotation vectors
        psi_rel = np.zeros(3)
        for node in range(self.nnodes_element_psi):
            # nodal axis angle vector
            psi_node = q[self.nodalDOF_element_psi[node]]

            # nodal rotation
            A_IK_node = Rotor(psi_node).toMatrix()

            # relative rotation of each node and corresponding
            # rotation vector
            A_RK_node = A_IR.T @ A_IK_node
            psi_RK_node = rodriguez_inv(A_RK_node)

            # add wheighted contribution of local rotation
            psi_rel += N[node] * psi_RK_node

        # objective rotation
        A_IK = A_IR @ rodriguez(psi_rel)

        return A_IK

    def A_IK_q(self, t, q, frame_ID):
        # # interpolate nodal derivative of the rotation matrix
        # N, _ = self.basis_functions_psi(frame_ID[0])
        # A_IK_q = np.zeros((3, 3, self.nq_element))
        # for node in range(self.nnodes_element_psi):
        #     nodalDOF_psi = self.nodalDOF_element_psi[node]
        #     A_IK_q[:, :, nodalDOF_psi] += N[node] * rodriguez_der(q[nodalDOF_psi])
        # return A_IK_q

        A_IK_q_num = approx_fprime(
            q, lambda q: self.A_IK(t, q, frame_ID), method="3-point"
        )
        # diff = A_IK_q - A_IK_q_num
        # error = np.linalg.norm(diff)
        # print(f"error A_IK_q: {error}")
        return A_IK_q_num

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3)):
        # compute centerline velocity
        N, _, _ = self.basis_functions_r(frame_ID[0])
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
        N_r, _, _ = self.basis_functions_r(frame_ID[0])
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # transformation matrix
        A_IK = self.A_IK(t, q, frame_ID)

        # skew symmetric matrix of K_r_SP
        K_r_SP_tilde = ax2skew(K_r_SP)

        # interpolate centerline and axis angle contributions
        J_P = np.zeros((3, self.nu_element))
        for node in range(self.nnodes_element_r):
            J_P[:, self.nodalDOF_element_r[node]] += N_r[node] * np.eye(3)
        for node in range(self.nnodes_element_psi):
            J_P[:, self.nodalDOF_element_u_psi[node]] -= (
                N_psi[node] * A_IK @ K_r_SP_tilde
            )

        return J_P

        # J_P_num = approx_fprime(
        #     np.zeros(self.nq_element), lambda u: self.v_P(t, q, u, frame_ID, K_r_SP)
        # )
        # diff = J_P_num - J_P
        # error = np.linalg.norm(diff)
        # print(f"error J_P: {error}")
        # return J_P_num

    # TODO:
    def J_P_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        return approx_fprime(q, lambda q: self.J_P(t, q, frame_ID, K_r_SP))

        # evaluate required nodal shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # skew symmetric matrix of K_r_SP
        K_r_SP_tilde = ax2skew(K_r_SP)

        # interpolate axis angle contributions since centerline contributon is
        # zero
        J_P_q = np.zeros((3, self.nu_element, self.nq_element))
        for node in range(self.nnodes_element_psi):
            nodalDOF_q = self.nodalDOF_element_psi[node]
            nodalDOF_u = self.nodalDOF_element_u_psi[node]
            A_IK_q = rodriguez_der(q[nodalDOF])
            J_P_q[:, nodalDOF_u[:, None], nodalDOF_q] -= N_psi[node] * np.einsum(
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
        N, _, _ = self.basis_functions_r(frame_ID[0])
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
        a_P_u = np.zeros((3, self.nu_element))
        for node in range(self.nnodes_element_r):
            a_P_u[:, self.nodalDOF_element_u_psi[node]] += N[node] * local

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
            K_Omega += N[node] * u[self.nodalDOF_element_u_psi[node]]
        return K_Omega

    def K_Omega_q(self, t, q, u, frame_ID):
        return np.zeros((3, self.nq_element))

    def K_J_R(self, t, q, frame_ID):
        N, _ = self.basis_functions_psi(frame_ID[0])
        K_J_R = np.zeros((3, self.nu_element))
        for node in range(self.nnodes_element_psi):
            K_J_R[:, self.nodalDOF_element_u_psi[node]] += N[node] * np.eye(3)
        return K_J_R

    def K_J_R_q(self, t, q, frame_ID):
        return np.zeros((3, self.nu_element, self.nq_element))

    def K_Psi(self, t, q, u, u_dot, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        time derivative of the angular velocities in the K-frame.
        """
        N, _ = self.basis_functions_psi(frame_ID[0])
        K_Psi = np.zeros(3)
        for node in range(self.nnodes_element_psi):
            K_Psi += N[node] * u_dot[self.nodalDOF_element_u_psi[node]]
        return K_Psi

    def K_Psi_q(self, t, q, u, u_dot, frame_ID):
        return np.zeros((3, self.nq_element))

    def K_Psi_u(self, t, q, u, u_dot, frame_ID):
        return np.zeros((3, self.nu_element))

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
        if self.basis == "Hermite":
            r = np.zeros((3, int(self.nnode_r / 2)))
            idx = 0
            for node, nodalDOF in enumerate(self.nodalDOF_r):
                if node % 2 == 0:
                    r[:, idx] = q_body[nodalDOF]
                    idx += 1
            return r
        else:
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
                x = r + radius * cos(alpha) * d2 + radius * sin(alpha) * d3
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
