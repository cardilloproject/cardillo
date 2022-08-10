import numpy as np
import meshio
import os
from math import sin, cos
from cardillo.math.algebra import sign, skew2ax
from cardillo.math.rotations import Spurrier, smallest_rotation

from cardillo.utility.coo import Coo
from cardillo.discretization.B_spline import BSplineKnotVector
from cardillo.discretization.lagrange import LagrangeKnotVector
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
    quat2rot,
    quatprod,
    quat2mat,
    quat2mat_p,
    trace3,
    LeviCivita,
)


# for smaller angles we use first order approximations of the equations since
# most of the SO(3) and SE(3) equations get singular for psi -> 0.
angle_singular = 1.0e-6


def Exp_SO3(psi: np.ndarray) -> np.ndarray:
    angle = norm(psi)
    if angle > angle_singular:
        # Park2005 (12)
        sa = np.sin(angle)
        ca = np.cos(angle)
        alpha = sa / angle
        beta2 = (1.0 - ca) / (angle * angle)
        psi_tilde = ax2skew(psi)
        return (
            np.eye(3, dtype=float) + alpha * psi_tilde + beta2 * psi_tilde @ psi_tilde
        )
    else:
        # first order approximation
        return np.eye(3, dtype=float) + ax2skew(psi)


def Exp_SO3_psi(psi: np.ndarray) -> np.ndarray:
    """Derivative of the axis-angle rotation found in Crisfield1999 above (4.1). 
    Derivations and final results are given in Gallego2015 (9).

    References
    ----------
    Crisfield1999: https://doi.org/10.1098/rspa.1999.0352 \\
    Gallego2015: https://doi.org/10.1007/s10851-014-0528-x
    """
    angle = norm(psi)

    # # Gallego2015 (9)
    # A_psi = np.zeros((3, 3, 3), dtype=float)
    # if isclose(angle, 0.0):
    #     # Derivative at the identity, see Gallego2015 Section 3.3
    #     for i in range(3):
    #         A_psi[:, :, i] = ax2skew(ei(i))
    # else:
    #     A = Exp_SO3(psi)
    #     eye_A = np.eye(3) - A
    #     psi_tilde = ax2skew(psi)
    #     angle2 = angle * angle
    #     for i in range(3):
    #         A_psi[:, :, i] = (
    #             (psi[i] * psi_tilde + ax2skew(cross3(psi, eye_A[:, i]))) @ A / angle2
    #         )

    A_psi = np.zeros((3, 3, 3), dtype=float)
    if angle > angle_singular:
        angle2 = angle * angle
        sa = np.sin(angle)
        ca = np.cos(angle)
        alpha = sa / angle
        alpha_psik = (ca - alpha) / angle2
        beta = 2.0 * (1.0 - ca) / angle2
        beta2_psik = (alpha - beta) / angle2

        psi_tilde = ax2skew(psi)
        psi_tilde2 = psi_tilde @ psi_tilde

        ############################
        # alpha * psi_tilde (part I)
        ############################
        A_psi[0, 2, 1] = A_psi[1, 0, 2] = A_psi[2, 1, 0] = alpha
        A_psi[0, 1, 2] = A_psi[1, 2, 0] = A_psi[2, 0, 1] = -alpha

        #############################
        # alpha * psi_tilde (part II)
        #############################
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    A_psi[i, j, k] += psi_tilde[i, j] * psi[k] * alpha_psik

        ###############################
        # beta2 * psi_tilde @ psi_tilde
        ###############################
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    A_psi[i, j, k] += psi_tilde2[i, j] * psi[k] * beta2_psik
                    for l in range(3):
                        A_psi[i, j, k] += (
                            0.5
                            * beta
                            * (
                                LeviCivita(k, l, i) * psi_tilde[l, j]
                                + psi_tilde[l, i] * LeviCivita(k, l, j)
                            )
                        )
    else:
        ###################
        # alpha * psi_tilde
        ###################
        A_psi[0, 2, 1] = A_psi[1, 0, 2] = A_psi[2, 1, 0] = 1.0
        A_psi[0, 1, 2] = A_psi[1, 2, 0] = A_psi[2, 0, 1] = -1.0

    return A_psi

    # A_psi_num = approx_fprime(psi, Exp_SO3, method="cs", eps=1.0e-10)
    # diff = A_psi - A_psi_num
    # error = np.linalg.norm(diff)
    # if error > 1.0e-10:
    #     print(f"error Exp_SO3_psi: {error}")
    # return A_psi_num


def Log_SO3(A: np.ndarray) -> np.ndarray:
    ca = 0.5 * (trace3(A) - 1.0)
    ca = np.clip(ca, -1, 1)  # clip to [-1, 1] for arccos!
    angle = np.arccos(ca)

    # fmt: off
    psi = 0.5 * np.array([
        A[2, 1] - A[1, 2],
        A[0, 2] - A[2, 0],
        A[1, 0] - A[0, 1]
    ], dtype=A.dtype)
    # fmt: on

    if angle > angle_singular:
        psi *= angle / np.sin(angle)
    return psi


def Log_SO3_A(A: np.ndarray) -> np.ndarray:
    """Derivative of the SO(3) Log map. See Blanco2010 (10.11)

    References:
    ===========
    Claraco2010: https://doi.org/10.48550/arXiv.2103.15980
    """
    ca = 0.5 * (trace3(A) - 1.0)
    ca = np.clip(ca, -1, 1)  # clip to [-1, 1] for arccos!
    angle = np.arccos(ca)

    psi_A = np.zeros((3, 3, 3), dtype=float)
    if angle > angle_singular:
        sa = np.sin(angle)
        b = 0.5 * angle / sa

        # fmt: off
        a = (angle * ca - sa) / (4.0 * sa**3) * np.array([
            A[2, 1] - A[1, 2],
            A[0, 2] - A[2, 0],
            A[1, 0] - A[0, 1]
        ], dtype=A.dtype)
        # fmt: on

        psi_A[0, 0, 0] = psi_A[0, 1, 1] = psi_A[0, 2, 2] = a[0]
        psi_A[1, 0, 0] = psi_A[1, 1, 1] = psi_A[1, 2, 2] = a[1]
        psi_A[2, 0, 0] = psi_A[2, 1, 1] = psi_A[2, 2, 2] = a[2]

        psi_A[0, 2, 1] = psi_A[1, 0, 2] = psi_A[2, 1, 0] = b
        psi_A[0, 1, 2] = psi_A[1, 2, 0] = psi_A[2, 0, 1] = -b
    else:
        psi_A[0, 2, 1] = psi_A[1, 0, 2] = psi_A[2, 1, 0] = 0.5
        psi_A[0, 1, 2] = psi_A[1, 2, 0] = psi_A[2, 0, 1] = -0.5

    return psi_A

    # psi_A_num = approx_fprime(A, Log_SO3, method="cs", eps=1.0e-10)
    # diff = psi_A - psi_A_num
    # error = np.linalg.norm(diff)
    # print(f"error Log_SO3_A: {error}")
    # return psi_A_num


def T_SO3(psi: np.ndarray) -> np.ndarray:
    angle = norm(psi)
    if angle > angle_singular:
        # Park2005 (19), actually its the transposed!
        sa = np.sin(angle)
        ca = np.cos(angle)
        psi_tilde = ax2skew(psi)
        alpha = sa / angle
        angle2 = angle * angle
        beta2 = (1.0 - ca) / angle2
        return (
            np.eye(3, dtype=float)
            - beta2 * psi_tilde
            + ((1.0 - alpha) / angle2) * psi_tilde @ psi_tilde
        )

        # # Barfoot2014 (98), actually its the transposed!
        # sa = np.sin(angle)
        # ca = np.cos(angle)
        # sinc = sa / angle
        # n = psi / angle
        # return (
        #     sinc * np.eye(3, dtype=float)
        #     - ((1.0 - ca) / angle) * ax2skew(n)
        #     + (1.0 - sinc) * np.outer(n, n)
        # )
    else:
        # first order approximation
        return np.eye(3, dtype=float) - 0.5 * ax2skew(psi)


def T_SO3_inv(psi: np.ndarray) -> np.ndarray:
    angle = norm(psi)
    psi_tilde = ax2skew(psi)
    if angle > angle_singular:
        # Park2005 (19), actually its the transposed!
        gamma = 0.5 * angle / (np.tan(0.5 * angle))
        return (
            np.eye(3, dtype=float)
            + 0.5 * psi_tilde
            + ((1.0 - gamma) / (angle * angle)) * psi_tilde @ psi_tilde
        )
    else:
        # first order approximation
        return np.eye(3, dtype=float) + 0.5 * psi_tilde


# use_quaternions = False
use_quaternions = True


class Kirchhoff:
    def __init__(
        self,
        material_model,
        A_rho0,
        I_rho0,
        nquadrature,
        nelement,
        Q,
        q0=None,
        u0=None,
    ):
        # beam properties
        self.materialModel = material_model  # material model
        self.A_rho0 = A_rho0  # line density
        self.I_rho0 = I_rho0  # second moment

        # material model
        self.material_model = material_model

        # discretization parameters
        self.nquadrature = nquadrature  # number of quadrature points
        self.nelement = nelement  # number of elements

        # Note: This implements a cubic Hermite spline for the centerline
        #       together with a linear Lagrange stretch vector field.
        polynomial_degree_r = 3
        self.knot_vector_r = HermiteNodeVector(polynomial_degree_r, nelement)
        polynomial_degree_psi = 1
        self.knot_vector_psi = LagrangeKnotVector(polynomial_degree_psi, nelement)
        self.quadrature = "Gauss"

        # number of degrees of freedom per node
        self.nq_node_r = nq_node_r = 3
        self.nq_node_psi = nq_node_psi = 1

        # centerline mesh (r, psi)
        self.mesh_r = Mesh1D(
            self.knot_vector_r,
            nquadrature,
            derivative_order=2,
            basis="Hermite",
            dim_q=nq_node_r,
            quadrature=self.quadrature,
        )

        # stretch mesh
        self.mesh_psi = Mesh1D(
            self.knot_vector_psi,
            nquadrature,
            derivative_order=1,
            basis="Lagrange",
            dim_q=nq_node_psi,
            quadrature=self.quadrature,
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
        self.nq = self.nu = (
            self.nq_r + self.nq_psi
        )  # total number of generalized coordinates

        # number of generalized coordiantes per element
        self.nq_element_r = self.mesh_r.nq_per_element
        self.nq_element_psi = self.mesh_psi.nq_per_element
        self.nq_element = self.nq_element_r + self.nq_element_psi

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
        self.N_r_xixi = self.mesh_r.N_xixi
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
        # curvature of the reference configuration
        self.K_Kappa0 = np.zeros((nelement, nquadrature, 3))

        for el in range(nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(nquadrature):
                qpi = self.qp[el, i]

                # evaluate strain measures and other quantities depending on chosen formulation
                # r, r_xi, r_xixi, A_IK, K_Kappa_bar = self.eval(qe, qpi)
                r, r_xi, A_IK, K_Gamma_bar, K_Kappa_bar = self.eval(qe, qpi)

                # length of reference tangential vector
                Ji = norm(r_xi)

                # torsional and flexural strains
                K_Kappa = K_Kappa_bar / Ji

                # safe precomputed quantities for later
                self.J[el, i] = Ji
                self.K_Kappa0[el, i] = K_Kappa

    @staticmethod
    def straight_configuration(
        nelement,
        L,
        r_OP=np.zeros(3),
        A_IK=np.eye(3),
    ):
        nn_r = nelement + 1
        nn_psi = nelement + 1

        # compute axis angle vector
        if use_quaternions:
            P0 = Spurrier(A_IK)
            P0 = np.sqrt(L) * P0
            p0 = P0[0]
            psi0 = P0[1:]
        else:
            psi0 = Log_SO3(A_IK)

        # centerline part
        xis = np.linspace(0, 1, num=nn_r)
        r0 = np.zeros((6, nn_r))
        t0 = A_IK @ (L * e1)
        for i, xi in enumerate(xis):
            ri = r_OP + xi * t0
            r0[:3, i] = ri
            # Note: This is the rotation vector corresponding to the tangent vector
            #       t0 = L * Exp_SO3(psi0) @ e1
            r0[3:, i] = psi0

        # reshape generalized coordinates to nodal ordering
        q_r = r0.reshape(-1, order="F")

        # initial length of the tangent vectors
        if use_quaternions:
            q_psi = p0 * np.ones(nn_psi)
        else:
            q_psi = L * np.ones(nn_psi)

        # assemble all reference generalized coordinates
        return np.concatenate([q_r, q_psi])

    def element_number(self, xi):
        # note the elements coincide for both meshes!
        return self.knot_vector_r.element_number(xi)[0]

    #########################################
    # equations of motion
    #########################################
    if False:

        def assembler_callback(self):
            self.__M_coo()

        def M_el(self, el):
            raise NotImplementedError
            M_el = np.zeros((self.nq_element, self.nq_element))
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

            print(f"f_gyr not implemented!")

            # for i in range(self.nquadrature):
            #     # interpoalte angular velocity
            #     K_Omega = np.zeros(3)
            #     for node in range(self.nnodes_element_psi):
            #         K_Omega += (
            #             self.N_psi[el, i, node] * ue[self.nodalDOF_element_psi[node]]
            #         )

            #     # vector of gyroscopic forces
            #     f_gyr_el_psi = (
            #         cross3(K_Omega, self.I_rho0 @ K_Omega) * self.J[el, i] * self.qw[el, i]
            #     )

            #     # multiply vector of gyroscopic forces with nodal virtual rotations
            #     for node in range(self.nnodes_element_psi):
            #         f_gyr_el[self.nodalDOF_element_psi[node]] += (
            #             self.N_psi[el, i, node] * f_gyr_el_psi
            #         )

            return f_gyr_el

        def f_gyr(self, t, q, u):
            f_gyr = np.zeros(self.nu)
            for el in range(self.nelement):
                elDOF = self.elDOF[el]
                elDOF = self.elDOF[el]
                f_gyr[elDOF] += self.f_gyr_el(t, q[elDOF], u[elDOF], el)
            return f_gyr

        def f_gyr_u_el(self, t, qe, ue, el):
            f_gyr_u_el = np.zeros((self.nq_element, self.nq_element))

            print(f"f_gyr_u not implemented!")

            # for i in range(self.nquadrature):
            #     # interpoalte angular velocity
            #     K_Omega = np.zeros(3)
            #     for node in range(self.nnodes_element_psi):
            #         K_Omega += (
            #             self.N_psi[el, i, node] * ue[self.nodalDOF_element_psi[node]]
            #         )

            #     # derivative of vector of gyroscopic forces
            #     f_gyr_u_el_psi = (
            #         ((ax2skew(K_Omega) @ self.I_rho0 - ax2skew(self.I_rho0 @ K_Omega)))
            #         * self.J[el, i]
            #         * self.qw[el, i]
            #     )

            #     # multiply derivative of gyroscopic force vector with nodal virtual rotations
            #     for node_a in range(self.nnodes_element_psi):
            #         nodalDOF_a = self.nodalDOF_element_psi[node_a]
            #         for node_b in range(self.nnodes_element_psi):
            #             nodalDOF_b = self.nodalDOF_element_psi[node_b]
            #             f_gyr_u_el[nodalDOF_a[:, None], nodalDOF_b] += f_gyr_u_el_psi * (
            #                 self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
            #             )

            return f_gyr_u_el

        def f_gyr_u(self, t, q, u, coo):
            for el in range(self.nelement):
                elDOF = self.elDOF[el]
                elDOF = self.elDOF[el]
                f_gyr_u_el = self.f_gyr_u_el(t, q[elDOF], u[elDOF], el)
                coo.extend(f_gyr_u_el, (self.uDOF[elDOF], self.uDOF[elDOF]))

    def eval(self, qe, qp):
        # evaluate shape functions
        N_r, N_r_xi, N_r_xixi = self.basis_functions_r(qp)
        N_psi, N_psi_xi = self.basis_functions_psi(qp)

        # extract nodal quantities
        r0 = qe[self.nodalDOF_element_r[0]]
        r1 = qe[self.nodalDOF_element_r[2]]
        psi0 = qe[self.nodalDOF_element_r[1]]
        psi1 = qe[self.nodalDOF_element_r[3]]

        if use_quaternions:
            # nodal real part of the quaternion
            p0 = qe[self.nodalDOF_element_psi[0]]
            p1 = qe[self.nodalDOF_element_psi[1]]

            # nodal quaternion
            P0 = np.array([*p0, *psi0])
            P1 = np.array([*p1, *psi1])

            # nodal tangent vector lengths
            j0 = P0 @ P0
            j1 = P1 @ P1

            # normalized quaternions
            Q0 = P0 / norm(P0)
            Q1 = P1 / norm(P1)

            # nodal rotation matrices
            A_IK0 = quat2rot(Q0)
            A_IK1 = quat2rot(Q1)
        else:
            # nodal tangent vector lengths
            j0 = qe[self.nodalDOF_element_psi[0]]
            j1 = qe[self.nodalDOF_element_psi[1]]

            # nodal rotation matrices
            A_IK0 = Exp_SO3(psi0)
            A_IK1 = Exp_SO3(psi1)

        # compute nodal tangent vectors
        ex0 = A_IK0 @ e1
        ex1 = A_IK1 @ e1
        t0 = j0 * ex0
        t1 = j1 * ex1

        # interpolate centerline and its derivatives using cubic Hermite spline
        r = N_r[0] * r0 + N_r[1] * t0 + N_r[2] * r1 + N_r[3] * t1
        r_xi = N_r_xi[0] * r0 + N_r_xi[1] * t0 + N_r_xi[2] * r1 + N_r_xi[3] * t1
        r_xixi = (
            N_r_xixi[0] * r0 + N_r_xixi[1] * t0 + N_r_xixi[2] * r1 + N_r_xixi[3] * t1
        )

        # ################################
        # # case 1: director interpolation
        # ################################
        # A_IK = N_psi[0] * A_IK0 + N_psi[1] * A_IK1
        # A_IK_xi = N_psi_xi[0] * A_IK0 + N_psi_xi[1] * A_IK1

        # K_Gamma_bar = A_IK.T @ r_xi
        # # TODO: Better curvature with volume correction
        # K_Kappa_bar = skew2ax(A_IK.T @ A_IK_xi)

        # ##################################
        # # case 2: SO(3) x R3 interpolation
        # ##################################
        # A_K0K1 = A_IK0.T @ A_IK1
        # psi_K0K0 = Log_SO3(A_K0K1)
        # psi = N_psi[1] * psi_K0K0
        # psi_xi = N_psi_xi[1] * psi_K0K0
        # A_IK = A_IK0 @ Exp_SO3(psi)
        # K_Kappa_bar = psi_xi
        # K_Gamma_bar = A_IK.T @ r_xi

        # #############################
        # # TODO: Why is this not working?
        # # case 3: SE(3) interpolation
        # #############################
        # from cardillo.beams.spatial.SE3 import Exp_SE3, Log_SE3, SE3, SE3inv
        # H_IK0 = SE3(A_IK0, r0)
        # H_IK1 = SE3(A_IK1, r1)
        # H_K0K1 = SE3inv(H_IK0) @ H_IK1
        # h_K0K1 = Log_SE3(H_K0K1)
        # h = N_psi[1] * h_K0K1
        # h_xi = N_psi_xi[1] * h_K0K1
        # H_IK = H_IK0 @ Exp_SE3(h)

        # K_Gamma_bar = h_xi[:3]
        # K_Kappa_bar = h_xi[3:]

        # A_IK = H_IK[:3, :3]
        # r = H_IK[:3, 3]
        # r_xi = A_IK @ K_Gamma_bar

        # #########################################
        # # case 4: smallest rotation interpolation
        # #########################################
        # def A_IK_fun(xi):
        #     # interpolate tangent vector using cubic Hermite spline
        #     _, N_r_xi, _ = self.basis_functions_r(xi)
        #     r_xi = N_r_xi[0] * r0 + N_r_xi[1] * t0 + N_r_xi[2] * r1 + N_r_xi[3] * t1

        #     # relative torsion from first to second node using smallest rotation
        #     A_K0K1 = smallest_rotation(t0, t1)

        #     # composed rotation to node 1
        #     A_IK1_bar = A_IK0 @ A_K0K1

        #     # extract difference in torsion angle of the two rotations
        #     d1, d2, d3 = A_IK1.T
        #     d1_bar, d2_bar, d3_bar = A_IK1_bar.T
        #     alpha1 = np.arctan2(d3 @ d2_bar, d2 @ d2_bar) # TODO: Not working with complex arguments!
        #     alpha1 = np.arctan((d3 @ d2_bar) / (d2 @ d2_bar))

        #     # psi = Log_SO3(A_IK1_bar.T @ A_IK1)
        #     # alpha1 = norm(psi)

        #     # interpolate relative torsion angle
        #     N_psi, _ = self.basis_functions_psi(xi)
        #     alpha = N_psi[1] * alpha1

        #     # current tangent vector
        #     t = r_xi / norm(r_xi)

        #     # relative smallest rotation w.r.t. first norde
        #     A_K0B = smallest_rotation(t0, t)

        #     # superimposed basic rotation with alpha around t
        #     A_BK = Exp_SO3(t * alpha)

        #     # composed rotation
        #     return A_IK0 @ A_K0B @ A_BK

        # A_IK = A_IK_fun(qp)
        # A_IK_xi = approx_fprime(qp, A_IK_fun)
        # K_Kappa_bar = skew2ax(A_IK.T @ A_IK_xi)

        # K_Gamma_bar = A_IK.T @ r_xi

        ##############################
        # without numerical derivative
        ##############################

        # relative torsion from first to second node using smallest rotation
        A_K0K1 = smallest_rotation(t0, t1)

        # composed rotation to node 1
        A_IK1_bar = A_IK0 @ A_K0K1

        # extract difference in torsion angle of the two rotations
        d1, d2, d3 = A_IK1.T
        d1_bar, d2_bar, d3_bar = A_IK1_bar.T
        cos_alpha1 = d2_bar @ d2
        sin_alpha1 = d3_bar @ d2

        def complex_atan2(y, x):
            """Atan2 implementation that can handle complex numbers, see https://de.wikipedia.org/wiki/Arctan2#Formel. It returns atan(y / x)."""
            if x > 0:
                return np.arctan(y / x)
            elif x < 0:
                if y > 0:
                    return np.arctan(y / x) + np.pi
                elif y < 0:
                    return np.arctan(y / x) - np.pi
                else:
                    return np.pi
            else:
                if y > 0:
                    return 0.5 * np.pi
                else:
                    return -0.5 * np.pi
            # elif x < 0 and y > 0:
            #     return np.arctan(y / x) + np.pi
            # elif x < 0 and y == 0:
            #     return np.pi
            # elif x < 0 and y < 0:
            #     return np.arctan(y / x) - np.pi
            # elif x == 0 and y > 0:
            #     return 0.5 * np.pi
            # elif x == 0 and y < 0:
            #     return -0.5 * np.pi

        alpha1 = complex_atan2(sin_alpha1, cos_alpha1)

        # #############################################
        # # angle extraction proposed by Meier2014 (54)
        # #############################################
        # # cos_alpha1 = min(-1.0, max(cos_alpha1, 1.0))
        # cos_alpha1 = np.clip(cos_alpha1, -1, 1)
        # alpha1 = np.sign(sin_alpha1) * np.arccos(cos_alpha1)

        # alpha1 = np.arctan2(d3_bar @ d2, d2_bar @ d2) # TODO: Not working with complex arguments!
        # # alpha1 = np.arctan((d3_bar @ d2) / (d2_bar @ d2))
        # # # alpha1 = np.arctan2(d3 @ d2_bar, d2 @ d2_bar) # TODO: Not working with complex arguments!
        # # alpha1 = np.arctan((d3 @ d2_bar) / (d2 @ d2_bar))

        # interpolate relative torsion angle
        alpha = N_psi[1] * alpha1
        alpha_xi = N_psi_xi[1] * alpha1

        # current tangent vector
        t = r_xi / norm(r_xi)

        # relative smallest rotation w.r.t. first norde
        A_K0B = smallest_rotation(t0, t)
        A_IB = A_IK0 @ A_K0B
        exB, eyB, ezB = A_IB.T

        # superimposed basic rotation with alpha around t
        A_BK = Exp_SO3(alpha * t)

        # composed rotation
        A_IK = A_IK0 @ A_K0B @ A_BK

        # # # # TODO: curvature!
        # # # K_Kappa_bar = A_IK.T @ cross3(r_xi, r_xixi) / (r_xi @ r_xi)
        # # # K_Kappa_bar[0] += alpha_xi
        # # # # # K_Kappa_bar = np.array([
        # # # # #     alpha_xi + ???,
        # # # # #     ???,
        # # # # #     ???
        # # # # # ])

        # # TODO: I think the curvature is still wrong!
        # j = norm(r_xi)
        # K_Kappa_bar = np.array(
        #     [
        #         # TODO: Do we have to add the torsion w.r.t. to the left node?
        #         # alpha_xi,
        #         # alpha_xi
        #         # + r_xixi
        #         # @ cross3(d1, ex0)
        #         # / (j * (1 + d1 @ ex0)),  # Mitterbach2020 (2.105)
        #         alpha_xi
        #         + r_xixi
        #         @ cross3(exB, ex0)
        #         / (j * (1.0 + exB @ ex0)),  # Mitterbach2020 (2.105): Curvature of the relative smallest rotation from t0 to current tangent vector t + torsion correction, cf. Meier2014 (59)
        #         (d3 @ r_xixi) / j,
        #         -(d2 @ r_xixi) / j,
        #     ]
        # )

        # cable version
        j = norm(r_xi)
        K_Kappa_bar = cross3(d1, r_xixi) / j
        K_Kappa_bar[0] += alpha_xi

        # diff = K_Kappa_bar - K_Kappa_bar2
        # error = norm(diff)
        # print(f"error K_Kappa_bar: {error}")

        K_Gamma_bar = A_IK.T @ r_xi

        return r, r_xi, A_IK, K_Gamma_bar, K_Kappa_bar

    def E_pot(self, t, q):
        E_pot = 0
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            E_pot += self.E_pot_el(q[elDOF], el)
        return E_pot

    def E_pot_el(self, qe, el):
        # raise NotImplementedError
        E_pot_el = 0

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = e1
            K_Kappa0 = self.K_Kappa0[el, i]

            # evaluate strain measures and other quantities depending on chosen formulation
            # r, r_xi, r_xixi, A_IK, K_Kappa_bar = self.eval(qe, qpi)
            r, r_xi, A_IK, K_Gamma_bar, K_Kappa_bar = self.eval(qe, qpi)

            # axial and shear strains
            K_Gamma = K_Gamma_bar / Ji
            # TODO: This works only for nodal smallest rotation!
            # K_Gamma[1] = 0
            # K_Gamma[2] = 0

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
            elDOF = self.elDOF[el]
            f_pot[elDOF] += self.f_pot_el(q[elDOF], el)
        return f_pot

    def f_pot_el(self, qe, el):
        return approx_fprime(
            qe, lambda qe: -self.E_pot_el(qe, el), eps=1.0e-10, method="cs"
        )
        # return approx_fprime(
        #     qe, lambda qe: -self.E_pot_el(qe, el), method="3-point"
        # )

        f_pot_el = np.zeros(self.nq_element)

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = e1
            K_Kappa0 = self.K_Kappa0[el, i]

            # evaluate strain measures and other quantities depending on chosen formulation
            r, r_xi, r_xixi, A_IK, K_Kappa_bar = self.eval(qe, qpi)
            r_xi2 = r_xi @ r_xi

            # axial and shear strains
            K_Gamma_bar = A_IK.T @ r_xi
            K_Gamma = K_Gamma_bar / Ji

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / Ji

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            # n = A_IK @ K_n
            # m = A_IK @ K_m

            # stretch = norm(K_Gamma)
            # self.material_model.n1(stretch, K_Kappa, K_Kappa0) * Ji * qwi

            ############################
            # virtual work contributions
            ############################
            for node in range(self.nnodes_element_r):
                f_pot_el[self.nodalDOF_element_r[node]] -= (
                    self.N_r_xi[el, i, node] * A_IK @ K_n * qwi
                )

            for node in range(self.nnodes_element_psi):
                # first delta phi parallel part
                f_pot_el[self.nodalDOF_element_psi[node]] -= (
                    self.N_psi_xi[el, i, node] * K_m[0] * qwi
                )

                # second delta phi parallel part
                tmp = cross3(K_Gamma_bar, K_n) + cross3(K_Kappa_bar, K_m)
                f_pot_el[self.nodalDOF_element_psi[node]] += (
                    self.N_psi[el, i, node] * tmp[0] * qwi
                )

                # first delta phi perp part
                f_pot_el[self.nodalDOF_element_r[node]] -= (
                    self.N_r_xi[el, i, node]
                    * cross3(
                        r_xixi / r_xi2 - r_xi * (r_xi @ r_xixi) / (r_xi2 * r_xi2), K_m
                    )
                    * qwi
                )
                f_pot_el[self.nodalDOF_element_r[node]] -= (
                    self.N_r_xixi[el, i, node] * cross3(r_xixi / r_xi2, K_m) * qwi
                )

                # second delta phi perp part
                f_pot_el[self.nodalDOF_element_r[node]] += (
                    self.N_r_xi[el, i, node] * cross3(r_xi / r_xi2, tmp) * qwi
                )

        return f_pot_el

    def f_pot_q(self, t, q, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
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
        raise NotImplementedError
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
        return self.elDOF[el]

    ###################
    # r_OP contribution
    ###################
    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # r_OC, _, _, A_IK, _ = self.eval(q, frame_ID[0])
        r_OC, _, A_IK, _, _ = self.eval(q, frame_ID[0])

        # rigid body formular
        return r_OC + A_IK @ K_r_SP

    # TODO:
    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # # TODO: The centerline part is not correct!
        # # compute centerline derivative
        # N, _, _ = self.basis_functions_r(frame_ID[0])
        # r_OP_q = np.zeros((3, self.nq_element), dtype=float)
        # for node in range(self.nnodes_element_r):
        #     raise NotImplementedError
        #     r_OP_q[:, self.nodalDOF_element_r[node]] += N[node] * np.eye(3)

        # # derivative of rigid body formular
        # r_OP_q += np.einsum("ijk,j->ik", self.A_IK_q(t, q, frame_ID), K_r_SP)
        # # return r_OP_q

        r_OP_q_num = approx_fprime(
            # q, lambda q: self.r_OP(t, q, frame_ID, K_r_SP), eps=1e-10, method="cs"
            q,
            lambda q: self.r_OP(t, q, frame_ID, K_r_SP),
            method="2-point",
        )
        # diff = r_OP_q_num - r_OP_q
        # error = np.linalg.norm(diff)
        # print(f"error r_OP_q: {error}")
        return r_OP_q_num

    def A_IK(self, t, q, frame_ID):
        # _, _, _, A_IK, _ = self.eval(q, frame_ID[0])
        _, _, A_IK, _, _ = self.eval(q, frame_ID[0])
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
            # q, lambda q: self.A_IK(t, q, frame_ID), eps=1e-10, method="cs"
            q,
            lambda q: self.A_IK(t, q, frame_ID),
            method="2-point",
        )
        # diff = A_IK_q - A_IK_q_num
        # error = np.linalg.norm(diff)
        # print(f"error A_IK_q: {error}")
        return A_IK_q_num

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3)):
        raise NotImplementedError
        # compute centerline velocity
        N, _, _ = self.basis_functions_r(frame_ID[0])
        v_C = np.zeros(3, dtype=float)
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

    # TODO:
    def J_P(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # J_P_num = approx_fprime(
        #     q,
        #     lambda q: self.r_OP(t, q, frame_ID, K_r_SP),
        #     eps=1e-10,
        #     method="cs",
        # )
        J_P_num = approx_fprime(
            q,
            lambda q: self.r_OP(t, q, frame_ID, K_r_SP),
            method="2-point",
        )
        # J_P_num = approx_fprime(
        #     np.zeros(self.nq_element), lambda u: self.v_P(t, q, u, frame_ID, K_r_SP)
        # )
        # diff = J_P_num - J_P
        # error = np.linalg.norm(diff)
        # print(f"error J_P: {error}")
        return J_P_num

    # TODO:
    def J_P_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        return approx_fprime(q, lambda q: self.J_P(t, q, frame_ID, K_r_SP))

        # evaluate required nodal shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # skew symmetric matrix of K_r_SP
        K_r_SP_tilde = ax2skew(K_r_SP)

        # interpolate axis angle contributions since centerline contributon is
        # zero
        J_P_q = np.zeros((3, self.nq_element, self.nq_element))
        for node in range(self.nnodes_element_psi):
            nodalDOF_q = self.nodalDOF_element_psi[node]
            nodalDOF_u = self.nodalDOF_element_psi[node]
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
        raise NotImplementedError
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

    # TODO:
    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        # K_Omega = self.K_Omega(t, q, u, frame_ID)
        # local = -self.A_IK(t, q, frame_ID) @ (
        #     ax2skew(cross3(K_Omega, K_r_SP)) + ax2skew(K_Omega) @ ax2skew(K_r_SP)
        # )

        # N, _ = self.basis_functions_psi(frame_ID[0])
        # a_P_u = np.zeros((3, self.nq_element))
        # for node in range(self.nnodes_element_r):
        #     a_P_u[:, self.nodalDOF_element_psi[node]] += N[node] * local

        # return a_P_u

        a_P_u_num = approx_fprime(
            u, lambda u: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP), method="3-point"
        )
        # diff = a_P_u_num - a_P_u
        # error = np.linalg.norm(diff)
        # print(f"error a_P_u: {error}")
        return a_P_u_num

    def K_Omega(self, t, q, u, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        angular velocities in the K-frame.
        """
        # evaluate shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # #########################
        # # Petrov-Galerkin version
        # #########################
        # # compute nodal angular velocities
        # K_Omega0 = u[self.nodalDOF_element_r[1]]
        # K_Omega1 = u[self.nodalDOF_element_r[3]]

        #########################
        # Bubnov-Galerkin version
        #########################

        # TODO: Nodal angular velcitiy of unit quaternions
        if use_quaternions:
            # nodal real parts of quaternions and their time derivative
            p00 = q[self.nodalDOF_element_psi[0]]
            p01 = q[self.nodalDOF_element_psi[1]]
            p0_dot0 = u[self.nodalDOF_element_psi[0]]
            p0_dot1 = u[self.nodalDOF_element_psi[1]]

            # nodal quaternion vector parts and their time derivatives
            p0 = q[self.nodalDOF_element_r[1]]
            p1 = q[self.nodalDOF_element_r[3]]
            p_dot0 = u[self.nodalDOF_element_r[1]]
            p_dot1 = u[self.nodalDOF_element_r[3]]

            # build nodal quaternions
            P0 = np.array([*p00, *p0])
            P_dot0 = np.array([*p0_dot0, *p_dot0])
            P1 = np.array([*p01, *p1])
            P_dot1 = np.array([*p0_dot1, *p_dot1])

            # compute normalized quaternion quantities
            Q0 = P0 / norm(P0)
            Q1 = P1 / norm(P1)
            Q0_inv = np.array([Q0[0], *-Q0[1:]])
            Q1_inv = np.array([Q1[0], *-Q1[1:]])
            Q_dot0 = (P_dot0 - Q0 * (Q0 @ P_dot0)) / norm(P0)
            Q_dot1 = (P_dot1 - Q1 * (Q1 @ P_dot1)) / norm(P1)

            # compute nodal angular velcoties in the body fixed frame,
            # see Egeland2002, (6.325)
            # Egenland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
            K_Omega0 = 2.0 * quatprod(Q0_inv, Q_dot0)[1:]
            K_Omega1 = 2.0 * quatprod(Q1_inv, Q_dot1)[1:]

        else:
            # extract nodal rotation vectors and their time derivatives
            psi0 = q[self.nodalDOF_element_r[1]]
            psi1 = q[self.nodalDOF_element_r[3]]
            psi_dot0 = u[self.nodalDOF_element_r[1]]
            psi_dot1 = u[self.nodalDOF_element_r[3]]

            # compute nodal angular velocities
            K_Omega0 = T_SO3(psi0) @ psi_dot0
            K_Omega1 = T_SO3(psi1) @ psi_dot1

        # interpolate angular velocities
        return N_psi[0] * K_Omega0 + N_psi[1] * K_Omega1

    # TODO:
    def K_Omega_q(self, t, q, u, frame_ID):
        return approx_fprime(
            # q, lambda q: self.K_Omega(t, q, u, frame_ID), eps=1e-10, method="cs"
            q,
            lambda q: self.K_Omega(t, q, u, frame_ID),
            method="2-point",
        )

    # TODO:
    def K_J_R(self, t, q, frame_ID):
        return approx_fprime(q, lambda u: self.K_Omega(t, q, u, frame_ID))
        # N, _ = self.basis_functions_psi(frame_ID[0])
        # K_J_R = np.zeros((3, self.nq_element))
        # for node in range(self.nnodes_element_psi):
        #     K_J_R[:, self.nodalDOF_element_psi[node]] += N[node] * np.eye(3)
        # return K_J_R

    # TODO:
    def K_J_R_q(self, t, q, frame_ID):
        return approx_fprime(q, lambda q: self.K_J_R(t, q, frame_ID))

    # TODO:
    def K_Psi(self, t, q, u, u_dot, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        time derivative of the angular velocities in the K-frame.
        """
        raise NotImplementedError
        N, _ = self.basis_functions_psi(frame_ID[0])
        K_Psi = np.zeros(3)
        for node in range(self.nnodes_element_psi):
            K_Psi += N[node] * u_dot[self.nodalDOF_element_psi[node]]
        return K_Psi

    def K_Psi_q(self, t, q, u, u_dot, frame_ID):
        raise NotImplementedError
        return np.zeros((3, self.nq_element))

    def K_Psi_u(self, t, q, u, u_dot, frame_ID):
        raise NotImplementedError
        return np.zeros((3, self.nq_element))

    ####################################################
    # body force
    ####################################################
    def distributed_force1D_pot_el(self, force, t, qe, el):
        raise NotADirectoryError
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
        raise NotADirectoryError
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
        r = np.zeros((3, int(self.nnode_r / 2)))
        idx = 0
        for node, nodalDOF in enumerate(self.nodalDOF_r):
            if node % 2 == 0:
                r[:, idx] = q_body[nodalDOF]
                idx += 1
        return r

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


class KirchhoffOld:
    def __init__(
        self,
        material_model,
        A_rho0,
        I_rho0,
        # polynomial_degree_r,
        # polynomial_degree_psi,
        nquadrature,
        nelement,
        Q,
        q0=None,
        u0=None,
    ):
        # beam properties
        self.materialModel = material_model  # material model
        self.A_rho0 = A_rho0  # line density
        self.I_rho0 = I_rho0  # second moment

        # material model
        self.material_model = material_model

        # discretization parameters
        self.nquadrature = nquadrature  # number of quadrature points
        self.nelement = nelement  # number of elements

        # Note: This implements a cubic Hermite spline for the centerline
        #       together with a linear Lagrange axis angle vector field
        #       for the superimposed rotation w.r.t. the smallest rotation.
        polynomial_degree_r = 3
        self.knot_vector_r = HermiteNodeVector(polynomial_degree_r, nelement)
        polynomial_degree_psi = 1
        # TODO: Enbale Lagrange shape functions again if ready.
        # self.knot_vector_psi = Node_vector(polynomial_degree_psi, nelement)
        polynomial_degree_psi = 1
        self.knot_vector_psi = BSplineKnotVector(polynomial_degree_psi, nelement)
        # self.quadrature = "Lobatto"
        self.quadrature = "Gauss"

        # number of degrees of freedom per node
        self.nq_node_r = nq_node_r = 3
        self.nq_node_psi = nq_node_psi = 1
        self.nu_node_psi = nu_node_psi = 1

        # centerline mesh
        self.mesh_r = Mesh1D(
            self.knot_vector_r,
            nquadrature,
            derivative_order=2,
            basis="Hermite",
            dim_q=nq_node_r,
            quadrature=self.quadrature,
        )

        # quaternion mesh (ugly modificatin for Hermite spline)
        # TODO: This is ugly!
        # TODO: Enable Lagrange again if ready.
        self.mesh_psi = Mesh1D(
            self.knot_vector_psi,
            nquadrature,
            derivative_order=1,
            basis="Lagrange",
            # basis="B-spline",
            dim_q=nq_node_psi,
            dim_u=nu_node_psi,
            quadrature=self.quadrature,
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
        self.nu_r = self.mesh_r.nu
        self.nu_psi = self.mesh_psi.nu
        self.nu = self.nu_r + self.nu_psi  # total number of generalized velocities

        # number of generalized coordiantes per element
        self.nq_element_r = self.mesh_r.nq_per_element
        self.nq_element_psi = self.mesh_psi.nq_per_element
        self.nq_element = self.nq_element_r + self.nq_element_psi
        self.nq_element_r = self.mesh_r.nu_per_element
        self.nq_element_psi = self.mesh_psi.nu_per_element
        self.nq_element = self.nq_element_r + self.nq_element_psi

        # global element connectivity
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_psi = self.mesh_psi.elDOF + self.nq_r
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_psi = self.mesh_psi.elDOF + self.nu_r

        # global nodal
        self.nodalDOF_r = self.mesh_r.nodalDOF
        self.nodalDOF_psi = self.mesh_psi.nodalDOF + self.nq_r
        self.nodalDOF_u_r = self.mesh_r.nodalDOF_u
        self.nodalDOF_u_psi = self.mesh_psi.nodalDOF_u + self.nu_r

        # nodal connectivity on element level
        self.nodalDOF_element_r = self.mesh_r.nodalDOF_element
        self.nodalDOF_element_psi = self.mesh_psi.nodalDOF_element + self.nq_element_r
        self.nodalDOF_element_r = self.mesh_r.nodalDOF_element_u
        self.nodalDOF_element_psi = self.mesh_psi.nodalDOF_element_u + self.nq_element_r

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nelement, self.nq_element), dtype=int)
        self.elDOF = np.zeros((nelement, self.nq_element), dtype=int)
        for el in range(nelement):
            self.elDOF[el, : self.nq_element_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_element_r :] = self.elDOF_psi[el]
            self.elDOF[el, : self.nq_element_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_element_r :] = self.elDOF_psi[el]

        # shape functions and their first derivatives
        self.N_r = self.mesh_r.N
        self.N_r_xi = self.mesh_r.N_xi
        self.N_r_xixi = self.mesh_r.N_xixi
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
        # curvature of the reference configuration
        self.K_Kappa0 = np.zeros((nelement, nquadrature, 3))

        for el in range(nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(nquadrature):
                qp = self.qp[el, i]

                # evaluate strain measures and other quantities depending on chosen formulation
                r_xi, r_xixi, A_IK, K_Kappa_bar = self.eval(qe, qp)

                # length of reference tangential vector
                Ji = norm(r_xi)

                # # stretch
                # stretch = 1. # Ji / Ji

                # torsional and flexural strains
                K_Kappa = K_Kappa_bar / Ji

                # safe precomputed quantities for later
                self.J[el, i] = Ji
                self.K_Kappa0[el, i] = K_Kappa

    @staticmethod
    def straight_configuration(
        nelement,
        L,
        r_OP=np.zeros(3),
        A_IK=np.eye(3),
    ):
        nn_r = nelement + 1
        polynomial_degree_psi = 1
        # TODO:
        nn_psi = polynomial_degree_psi + nelement  # B-spline
        # nn_psi = polynomial_degree_psi * nelement + 1 # Lagrange

        # centerline part
        xis = np.linspace(0, 1, num=nn_r)
        r0 = np.zeros((6, nn_r))
        t0 = A_IK @ (L * e1)
        for i, xi in enumerate(xis):
            ri = r_OP + xi * t0
            r0[:3, i] = ri
            r0[3:, i] = t0

        # reshape generalized coordinates to nodal ordering
        q_r = r0.reshape(-1, order="F")

        # extract first director and buld new rotation
        # note: This rotation does not coincide with the initial A_IK and has
        # to be corrected afterwards using the superimposed rotation with phi!
        d1, _, _ = A_IK.T
        A_IK0 = smallest_rotation(e1, d1)

        # extract axis angle vector between first and desired rotation
        psi = rodriguez_inv(A_IK0.T @ A_IK)

        # extract rotation angle
        norm_psi = norm(psi)

        # TODO: How to compute initial phi?
        q_phi = norm_psi * np.ones(nn_psi)

        # assemble all reference generalized coordinates
        return np.concatenate([q_r, q_phi])

    def element_number(self, xi):
        # note the elements coincide for both meshes!
        return self.knot_vector_r.element_number(xi)[0]

    def reference_rotation(self, qe: np.ndarray):
        """Reference rotation proposed by Crisfield1999 (5.8) applied on nodal
        tangent vectors using smallest rotation mapping superimposed by
        rodriguez formular with psi around normalized tangent vector."""
        raise RuntimeError("This should not be used!")
        # compute nodal smallest rotation
        t0 = qe[self.nodalDOF_element_r[1]]
        t1 = qe[self.nodalDOF_element_r[3]]
        A_IB0 = smallest_rotation(e1, t0)
        A_IB1 = smallest_rotation(e1, t1)

        # compute rodriguez formular with nodal rotation values around nodal
        # tangent vectors
        d10 = t0 / norm(t0)
        d11 = t1 / norm(t1)
        phi0 = qe[self.nodalDOF_element_psi[0]]
        phi1 = qe[self.nodalDOF_element_psi[1]]
        A_B0K0 = rodriguez(phi0 * d10)
        A_B1K1 = rodriguez(phi1 * d11)

        # composite rotation
        A_IK0 = A_IB0 @ A_B0K0
        A_IK1 = A_IB1 @ A_B1K1

        # retlative rotation and corresponding rotation vector
        A_01 = A_IK0.T @ A_IK1  # Crisfield1999 (5.8)
        phi_01 = rodriguez_inv(A_01)

        # midway reference rotation
        return A_01 @ rodriguez(0.5 * phi_01)

    def relative_interpolation(self, qe: np.ndarray, el: int, qp: int):
        """Interpolation function for relative rotation vectors proposed by
        Crisfield1999 (5.7) and (5.8) applied on nodal smallest rotation
        superimposed by rodriguez formular with psi around normalized tangent
        vector."""
        raise RuntimeWarning("This should not be used!")
        # compute nodal smallest rotation
        t0 = qe[self.nodalDOF_element_r[1]]
        t1 = qe[self.nodalDOF_element_r[3]]
        A_IB0 = smallest_rotation(e1, t0)
        A_IB1 = smallest_rotation(e1, t1)

        # compute rodriguez formular with nodal rotation values around nodal
        # tangent vectors
        d10 = t0 / norm(t0)
        d11 = t1 / norm(t1)
        phi0 = qe[self.nodalDOF_element_psi[0]]
        phi1 = qe[self.nodalDOF_element_psi[1]]
        A_B0K0 = rodriguez(phi0 * d10)
        A_B1K1 = rodriguez(phi1 * d11)

        # composite rotation
        A_IK0 = A_IB0 @ A_B0K0
        A_IK1 = A_IB1 @ A_B1K1

        # retlative rotation and corresponding rotation vector
        A_01 = A_IK0.T @ A_IK1  # Crisfield1999 (5.8)
        phi_01 = rodriguez_inv(A_01)

        # midway reference rotation
        A_IR = A_01 @ rodriguez(0.5 * phi_01)

        # relative rotation of each node and corresponding
        # rotation vector
        A_RK0 = A_IR.T @ A_IK0
        psi_RK0 = rodriguez_inv(A_RK0)
        A_RK1 = A_IR.T @ A_IK1
        psi_RK1 = rodriguez_inv(A_RK1)

        # add wheighted contribution of local rotation
        psi_rel = self.N_psi[el, qp, 0] * psi_RK0 + self.N_psi[el, qp, 1] * psi_RK1
        psi_rel_xi = (
            self.N_psi_xi[el, qp, 0] * psi_RK0 + self.N_psi_xi[el, qp, 1] * psi_RK1
        )

        return psi_rel, psi_rel_xi

    def assembler_callback(self):
        self.__M_coo()

    #########################################
    # equations of motion
    #########################################
    def M_el(self, el):
        M_el = np.zeros((self.nq_element, self.nq_element))

        print(f"Mass matrix not implemented!")

        # for i in range(self.nquadrature):
        #     # extract reference state variables
        #     qwi = self.qw[el, i]
        #     Ji = self.J[el, i]

        #     # delta_r A_rho0 r_ddot part
        #     # TODO: Can this be simplified with a single nodal loop?
        #     M_el_r = np.eye(3) * self.A_rho0 * Ji * qwi
        #     for node_a in range(self.nnodes_element_r):
        #         nodalDOF_a = self.nodalDOF_element_r[node_a]
        #         for node_b in range(self.nnodes_element_r):
        #             nodalDOF_b = self.nodalDOF_element_r[node_b]
        #             M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r * (
        #                 self.N_r[el, i, node_a] * self.N_r[el, i, node_b]
        #             )
        #     # for node in range(self.nnodes_element_r):
        #     #     nodalDOF = self.nodalDOF_element_r[node]
        #     #     N_node = self.N_r[el, i, node]
        #     #     M_el[nodalDOF[:, None], nodalDOF] += M_el_r * (N_node * N_node)

        #     # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
        #     M_el_psi = self.I_rho0 * Ji * qwi
        #     for node_a in range(self.nnodes_element_psi):
        #         nodalDOF_a = self.nodalDOF_element_psi[node_a]
        #         for node_b in range(self.nnodes_element_psi):
        #             nodalDOF_b = self.nodalDOF_element_psi[node_b]
        #             M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi * (
        #                 self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
        #             )

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

        print(f"f_gyr not implemented!")

        # for i in range(self.nquadrature):
        #     # interpoalte angular velocity
        #     K_Omega = np.zeros(3)
        #     for node in range(self.nnodes_element_psi):
        #         K_Omega += (
        #             self.N_psi[el, i, node] * ue[self.nodalDOF_element_psi[node]]
        #         )

        #     # vector of gyroscopic forces
        #     f_gyr_el_psi = (
        #         cross3(K_Omega, self.I_rho0 @ K_Omega) * self.J[el, i] * self.qw[el, i]
        #     )

        #     # multiply vector of gyroscopic forces with nodal virtual rotations
        #     for node in range(self.nnodes_element_psi):
        #         f_gyr_el[self.nodalDOF_element_psi[node]] += (
        #             self.N_psi[el, i, node] * f_gyr_el_psi
        #         )

        return f_gyr_el

    def f_gyr(self, t, q, u):
        f_gyr = np.zeros(self.nu)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF = self.elDOF[el]
            f_gyr[elDOF] += self.f_gyr_el(t, q[elDOF], u[elDOF], el)
        return f_gyr

    def f_gyr_u_el(self, t, qe, ue, el):
        f_gyr_u_el = np.zeros((self.nq_element, self.nq_element))

        print(f"f_gyr_u not implemented!")

        # for i in range(self.nquadrature):
        #     # interpoalte angular velocity
        #     K_Omega = np.zeros(3)
        #     for node in range(self.nnodes_element_psi):
        #         K_Omega += (
        #             self.N_psi[el, i, node] * ue[self.nodalDOF_element_psi[node]]
        #         )

        #     # derivative of vector of gyroscopic forces
        #     f_gyr_u_el_psi = (
        #         ((ax2skew(K_Omega) @ self.I_rho0 - ax2skew(self.I_rho0 @ K_Omega)))
        #         * self.J[el, i]
        #         * self.qw[el, i]
        #     )

        #     # multiply derivative of gyroscopic force vector with nodal virtual rotations
        #     for node_a in range(self.nnodes_element_psi):
        #         nodalDOF_a = self.nodalDOF_element_psi[node_a]
        #         for node_b in range(self.nnodes_element_psi):
        #             nodalDOF_b = self.nodalDOF_element_psi[node_b]
        #             f_gyr_u_el[nodalDOF_a[:, None], nodalDOF_b] += f_gyr_u_el_psi * (
        #                 self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
        #             )

        return f_gyr_u_el

    def f_gyr_u(self, t, q, u, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF = self.elDOF[el]
            f_gyr_u_el = self.f_gyr_u_el(t, q[elDOF], u[elDOF], el)
            coo.extend(f_gyr_u_el, (self.uDOF[elDOF], self.uDOF[elDOF]))

    def eval_old(self, qe, el, qp):
        # interpolate tangent vector
        r_xi = np.zeros(3)
        for node in range(self.nnodes_element_r):
            r_xi += self.N_r_xi[el, qp, node] * qe[self.nodalDOF_element_r[node]]

        # compute nodal smallest rotation
        t0 = qe[self.nodalDOF_element_r[1]]
        t1 = qe[self.nodalDOF_element_r[3]]
        A_IB0 = smallest_rotation(e1, t0)
        A_IB1 = smallest_rotation(e1, t1)

        # compute rodriguez formular with nodal rotation values around nodal
        # tangent vectors
        d10 = t0 / norm(t0)
        d11 = t1 / norm(t1)
        phi0 = qe[self.nodalDOF_element_psi[0]]
        phi1 = qe[self.nodalDOF_element_psi[1]]
        A_B0K0 = rodriguez(phi0 * d10)
        A_B1K1 = rodriguez(phi1 * d11)

        # composite rotation
        A_IK0 = A_IB0 @ A_B0K0
        A_IK1 = A_IB1 @ A_B1K1

        # retlative rotation and corresponding rotation vector
        A_01 = A_IK0.T @ A_IK1  # Crisfield1999 (5.8)
        phi_01 = rodriguez_inv(A_01)

        # midway reference rotation
        A_IR = A_01 @ rodriguez(0.5 * phi_01)

        # relative rotation of each node and corresponding
        # rotation vector
        A_RK0 = A_IR.T @ A_IK0
        psi_RK0 = rodriguez_inv(A_RK0)
        A_RK1 = A_IR.T @ A_IK1
        psi_RK1 = rodriguez_inv(A_RK1)

        # add wheighted contribution of local rotation
        psi_rel = self.N_psi[el, qp, 0] * psi_RK0 + self.N_psi[el, qp, 1] * psi_RK1
        psi_rel_xi = (
            self.N_psi_xi[el, qp, 0] * psi_RK0 + self.N_psi_xi[el, qp, 1] * psi_RK1
        )

        # objective rotation
        A_IK = A_IR @ rodriguez(psi_rel)

        # objective curvature
        T = tangent_map(psi_rel)
        K_Kappa_bar = T @ psi_rel_xi

        return r_xi, A_IK, K_Kappa_bar

    def eval(self, qe, qp):
        # evaluate shape functions
        _, N_r_xi, N_r_xixi = self.basis_functions_r(qp)
        N_psi, N_psi_xi = self.basis_functions_psi(qp)

        # interpolate tangent vector
        r_xi = np.zeros(3, dtype=qe.dtype)
        r_xixi = np.zeros(3, dtype=qe.dtype)
        for node in range(self.nnodes_element_r):
            r_xi += N_r_xi[node] * qe[self.nodalDOF_element_r[node]]
            r_xixi += N_r_xixi[node] * qe[self.nodalDOF_element_r[node]]

        # compute nodal smallest rotation
        t0 = qe[self.nodalDOF_element_r[1]]
        t1 = qe[self.nodalDOF_element_r[3]]
        A_IB0 = smallest_rotation(e1, t0)
        A_IB1 = smallest_rotation(e1, t1)

        # compute rodriguez formular with nodal rotation values around nodal
        # tangent vectors
        d10 = t0 / norm(t0)
        d11 = t1 / norm(t1)
        phi0 = qe[self.nodalDOF_element_psi[0]]
        phi1 = qe[self.nodalDOF_element_psi[1]]
        # TODO: This can be implemented more efficient!
        A_B0K0 = rodriguez(phi0 * d10)
        A_B1K1 = rodriguez(phi1 * d11)

        # composite rotation
        A_IK0 = A_IB0 @ A_B0K0
        A_IK1 = A_IB1 @ A_B1K1

        # interpolate transformation matrix an dits derivative
        A_IK = N_psi[0] * A_IK0 + N_psi[1] * A_IK1
        A_IK_xi = N_psi_xi[0] * A_IK0 + N_psi_xi[1] * A_IK1

        # compute curvature
        K_Kappa_bar = skew2ax(A_IK.T @ A_IK_xi)

        # # TODO: Better curvature with volume correction
        # # torsional and flexural strains
        # d1, d2, d3 = A_IK.T
        # d1_xi, d2_xi, d3_xi = A_IK_xi.T
        # half_d = d1 @ cross3(d2, d3)
        # K_Kappa_bar = (
        #     np.array(
        #         [
        #             0.5 * (d3 @ d2_xi - d2 @ d3_xi),
        #             0.5 * (d1 @ d3_xi - d3 @ d1_xi),
        #             0.5 * (d2 @ d1_xi - d1 @ d2_xi),
        #         ]
        #     )
        #     / half_d
        # )

        return r_xi, r_xixi, A_IK, K_Kappa_bar

    def E_pot(self, t, q):
        E_pot = 0
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            E_pot += self.E_pot_el(q[elDOF], el)
        return E_pot

    def E_pot_el(self, qe, el):
        # raise NotImplementedError
        E_pot_el = 0

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = e1
            K_Kappa0 = self.K_Kappa0[el, i]

            # evaluate strain measures and other quantities depending on chosen formulation
            r_xi, r_xixi, A_IK, K_Kappa_bar = self.eval(qe, qpi)

            # # stretch
            # ji = norm(r_xi)
            # stretch = ji / Ji

            # axial and shear strains
            K_Gamma_bar = A_IK.T @ r_xi
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
        f_pot = np.zeros(self.nu)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF = self.elDOF[el]
            f_pot[elDOF] += self.f_pot_el(q[elDOF], el)
        return f_pot

    def f_pot_el(self, qe, el):
        # return approx_fprime(qe, lambda qe: self.E_pot_el(qe, el), method="3-point")
        return approx_fprime(
            qe, lambda qe: self.E_pot_el(qe, el), eps=1.0e-10, method="cs"
        )
        f_pot_el = np.zeros(self.nq_element)

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            K_Gamma0 = e1
            K_Kappa0 = self.K_Kappa0[el, i]

            # evaluate strain measures and other quantities depending on chosen formulation
            r_xi, r_xixi, A_IK, K_Kappa_bar = self.eval(qe, qpi)
            r_xi2 = r_xi @ r_xi

            # axial and shear strains
            K_Gamma_bar = A_IK.T @ r_xi
            K_Gamma = K_Gamma_bar / Ji

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / Ji

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            # n = A_IK @ K_n
            # m = A_IK @ K_m

            # stretch = norm(K_Gamma)
            # self.material_model.n1(stretch, K_Kappa, K_Kappa0) * Ji * qwi

            ############################
            # virtual work contributions
            ############################
            for node in range(self.nnodes_element_r):
                f_pot_el[self.nodalDOF_element_r[node]] -= (
                    self.N_r_xi[el, i, node] * A_IK @ K_n * qwi
                )

            for node in range(self.nnodes_element_psi):
                # first delta phi parallel part
                f_pot_el[self.nodalDOF_element_psi[node]] -= (
                    self.N_psi_xi[el, i, node] * K_m[0] * qwi
                )

                # second delta phi parallel part
                tmp = cross3(K_Gamma_bar, K_n) + cross3(K_Kappa_bar, K_m)
                f_pot_el[self.nodalDOF_element_psi[node]] += (
                    self.N_psi[el, i, node] * tmp[0] * qwi
                )

                # first delta phi perp part
                f_pot_el[self.nodalDOF_element_r[node]] -= (
                    self.N_r_xi[el, i, node]
                    * cross3(
                        r_xixi / r_xi2 - r_xi * (r_xi @ r_xixi) / (r_xi2 * r_xi2), K_m
                    )
                    * qwi
                )
                f_pot_el[self.nodalDOF_element_r[node]] -= (
                    self.N_r_xixi[el, i, node] * cross3(r_xixi / r_xi2, K_m) * qwi
                )

                # second delta phi perp part
                f_pot_el[self.nodalDOF_element_r[node]] += (
                    self.N_r_xi[el, i, node] * cross3(r_xi / r_xi2, tmp) * qwi
                )

        return f_pot_el

    def f_pot_q(self, t, q, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
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
        raise NotImplementedError
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
        return self.elDOF[el]

    def r_OC(self, t, q, frame_ID):
        # compute centerline position
        N, _, _ = self.basis_functions_r(frame_ID[0])
        r_OC = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_r):
            r_OC += N[node] * q[self.nodalDOF_element_r[node]]
        return r_OC

    def r_OC_q(self, t, q, frame_ID):
        # compute centerline position
        N, _, _ = self.basis_functions_r(frame_ID[0])
        r_OC_q = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_r):
            r_OC_q[:, self.nodalDOF_element_r[node]] += N[node] * np.eye(3, dtype=float)
        return r_OC_q

    def r_OC_xi(self, t, q, frame_ID):
        # compute centerline position
        _, N_xi, _ = self.basis_functions_r(frame_ID[0])
        r_OC_xi = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_r):
            r_OC_xi += N_xi[node] * q[self.nodalDOF_element_r[node]]
        return r_OC_xi

    def r_OC_xixi(self, t, q, frame_ID):
        # compute centerline position
        _, _, N_xixi = self.basis_functions_r(frame_ID[0])
        r_OC_xixi = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_r):
            r_OC_xixi += N_xixi[node] * q[self.nodalDOF_element_r[node]]
        return r_OC_xixi

    def J_C(self, t, q, frame_ID):
        # evaluate required nodal shape functions
        N, _, _ = self.basis_functions_r(frame_ID[0])

        # interpolate centerline and axis angle contributions
        J_C = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_r):
            J_C[:, self.nodalDOF_element_r[node]] += N[node] * np.eye(3, dtype=float)

        return J_C

    def J_C_q(self, t, q, frame_ID):
        return np.zeros((3, self.nq_element, self.nq_element), dtype=float)

    ###################
    # r_OP contribution
    ###################
    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # compute centerline position
        N, _, _ = self.basis_functions_r(frame_ID[0])
        r_OC = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_r):
            r_OC += N[node] * q[self.nodalDOF_element_r[node]]

        # rigid body formular
        return r_OC + self.A_IK(t, q, frame_ID) @ K_r_SP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # compute centerline derivative
        N, _, _ = self.basis_functions_r(frame_ID[0])
        r_OP_q = np.zeros((3, self.nq_element), dtype=float)
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
        # N, _ = self.basis_functions_psi(frame_ID[0])

        # # compute nodal smallest rotation
        # t0 = q[self.nodalDOF_element_r[1]]
        # t1 = q[self.nodalDOF_element_r[3]]
        # A_IB0 = smallest_rotation(e1, t0)
        # A_IB1 = smallest_rotation(e1, t1)

        # # compute rodriguez formular with nodal rotation values around nodal
        # # tangent vectors
        # d10 = t0 / norm(t0)
        # d11 = t1 / norm(t1)
        # phi0 = q[self.nodalDOF_element_psi[0]]
        # phi1 = q[self.nodalDOF_element_psi[1]]
        # A_B0K0 = rodriguez(phi0 * d10)
        # A_B1K1 = rodriguez(phi1 * d11)

        # # composite rotation
        # A_IK0 = A_IB0 @ A_B0K0
        # A_IK1 = A_IB1 @ A_B1K1

        # # retlative rotation and corresponding rotation vector
        # A_01 = A_IK0.T @ A_IK1  # Crisfield1999 (5.8)
        # phi_01 = rodriguez_inv(A_01)

        # # midway reference rotation
        # A_IR = A_01 @ rodriguez(0.5 * phi_01)

        # # relative rotation of each node and corresponding
        # # rotation vector
        # A_RK0 = A_IR.T @ A_IK0
        # psi_RK0 = rodriguez_inv(A_RK0)
        # A_RK1 = A_IR.T @ A_IK1
        # psi_RK1 = rodriguez_inv(A_RK1)

        # # add wheighted contribution of local rotation
        # psi_rel = N[0] * psi_RK0 + N[1] * psi_RK1

        # # objective rotation
        # A_IK = A_IR @ rodriguez(psi_rel)

        _, _, A_IK, _ = self.eval(q, frame_ID[0])

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
        v_C = np.zeros(3, dtype=float)
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

    # TODO:
    def J_P(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # # evaluate required nodal shape functions
        # N_r, _, _ = self.basis_functions_r(frame_ID[0])
        # N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # # transformation matrix
        # A_IK = self.A_IK(t, q, frame_ID)

        # # skew symmetric matrix of K_r_SP
        # K_r_SP_tilde = ax2skew(K_r_SP)

        # # interpolate centerline and axis angle contributions
        # J_P = np.zeros((3, self.nq_element))
        # for node in range(self.nnodes_element_r):
        #     J_P[:, self.nodalDOF_element_r[node]] += N_r[node] * np.eye(3)
        # for node in range(self.nnodes_element_psi):
        #     J_P[:, self.nodalDOF_element_psi[node]] -= (
        #         N_psi[node] * A_IK @ K_r_SP_tilde
        #     )

        # return J_P

        J_P_num = approx_fprime(
            np.zeros(self.nq_element), lambda u: self.v_P(t, q, u, frame_ID, K_r_SP)
        )
        # diff = J_P_num - J_P
        # error = np.linalg.norm(diff)
        # print(f"error J_P: {error}")
        return J_P_num

    # TODO:
    def J_P_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        return approx_fprime(q, lambda q: self.J_P(t, q, frame_ID, K_r_SP))

        # evaluate required nodal shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # skew symmetric matrix of K_r_SP
        K_r_SP_tilde = ax2skew(K_r_SP)

        # interpolate axis angle contributions since centerline contributon is
        # zero
        J_P_q = np.zeros((3, self.nq_element, self.nq_element))
        for node in range(self.nnodes_element_psi):
            nodalDOF_q = self.nodalDOF_element_psi[node]
            nodalDOF_u = self.nodalDOF_element_psi[node]
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

    # TODO:
    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        # K_Omega = self.K_Omega(t, q, u, frame_ID)
        # local = -self.A_IK(t, q, frame_ID) @ (
        #     ax2skew(cross3(K_Omega, K_r_SP)) + ax2skew(K_Omega) @ ax2skew(K_r_SP)
        # )

        # N, _ = self.basis_functions_psi(frame_ID[0])
        # a_P_u = np.zeros((3, self.nq_element))
        # for node in range(self.nnodes_element_r):
        #     a_P_u[:, self.nodalDOF_element_psi[node]] += N[node] * local

        # return a_P_u

        a_P_u_num = approx_fprime(
            u, lambda u: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP), method="3-point"
        )
        # diff = a_P_u_num - a_P_u
        # error = np.linalg.norm(diff)
        # print(f"error a_P_u: {error}")
        return a_P_u_num

    def K_Omega(self, t, q, u, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        angular velocities in the K-frame.
        """
        _, N_r_xi, _ = self.basis_functions_r(frame_ID[0])
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate tangent vector and its derivative
        r_xi = np.zeros(3, dtype=float)
        r_xidot = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_r):
            r_xi += N_r_xi[node] * q[self.nodalDOF_element_r[node]]
            r_xidot += N_r_xi[node] * u[self.nodalDOF_element_r[node]]

        # interpolate superimposed angular velocity around d1
        psi_dot = 0.0
        for node in range(self.nnodes_element_psi):
            psi_dot += N_psi[node] * u[self.nodalDOF_element_psi[node]]

        # compute first director
        ji = norm(r_xi)
        d1 = r_xi / ji

        # first directors derivative
        d1_dot = (np.eye(3) - np.outer(d1, d1)) @ r_xidot / ji

        # angular velocity of centerline and angle part
        return cross3(d1, d1_dot) + d1 * psi_dot

        # return d1 * psi_dot + cross3(r_xi, r_xidot) / (r_xi @ r_xi)

    # TODO:
    def K_Omega_q(self, t, q, u, frame_ID):
        return approx_fprime(q, lambda q: self.K_Omega(t, q, u, frame_ID))

    # TODO:
    def K_J_R(self, t, q, frame_ID):
        return approx_fprime(
            np.zeros(self.nq_element), lambda u: self.K_Omega(t, q, u, frame_ID)
        )
        # N, _ = self.basis_functions_psi(frame_ID[0])
        # K_J_R = np.zeros((3, self.nq_element))
        # for node in range(self.nnodes_element_psi):
        #     K_J_R[:, self.nodalDOF_element_psi[node]] += N[node] * np.eye(3)
        # return K_J_R

    # TODO:
    def K_J_R_q(self, t, q, frame_ID):
        return approx_fprime(q, lambda q: self.K_J_R(t, q, frame_ID))

    # TODO:
    def K_Psi(self, t, q, u, u_dot, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        time derivative of the angular velocities in the K-frame.
        """
        raise NotImplementedError
        N, _ = self.basis_functions_psi(frame_ID[0])
        K_Psi = np.zeros(3)
        for node in range(self.nnodes_element_psi):
            K_Psi += N[node] * u_dot[self.nodalDOF_element_psi[node]]
        return K_Psi

    def K_Psi_q(self, t, q, u, u_dot, frame_ID):
        raise NotImplementedError
        return np.zeros((3, self.nq_element))

    def K_Psi_u(self, t, q, u, u_dot, frame_ID):
        raise NotImplementedError
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
    # def nodes(self, q):
    #     q_body = q[self.qDOF]
    #     if self.basis == "Hermite":
    #         r = np.zeros((3, int(self.nnode_r / 2)))
    #         idx = 0
    #         for node, nodalDOF in enumerate(self.nodalDOF_r):
    #             if node % 2 == 0:
    #                 r[:, idx] = q_body[nodalDOF]
    #                 idx += 1
    #         return r
    #     else:
    #         return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF_r]).T
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
