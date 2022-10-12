import numpy as np

from cardillo.utility.coo import Coo
from cardillo.discretization.lagrange import LagrangeKnotVector
from cardillo.discretization.b_spline import BSplineKnotVector
from cardillo.discretization.hermite import HermiteNodeVector
from cardillo.discretization.mesh1D import Mesh1D
from cardillo.math import (
    pi,
    norm,
    cross3,
    ax2skew,
    approx_fprime,
    tangent_map_s,
    trace3,
    LeviCivita3,
    e1,
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
                                LeviCivita3(k, l, i) * psi_tilde[l, j]
                                + psi_tilde[l, i] * LeviCivita3(k, l, j)
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


class DirectorAxisAngle:
    def __init__(
        self,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree_r,
        polynomial_degree_psi,
        nelement,
        Q,
        q0=None,
        u0=None,
        basis_r="B-spline",
        basis_psi="B-spline",
    ):
        # beam properties
        self.material_model = material_model  # material model
        self.A_rho0 = A_rho0
        self.K_S_rho0 = K_S_rho0
        self.K_I_rho0 = K_I_rho0

        # can we use a constant mass matrix
        if np.allclose(K_S_rho0, np.zeros_like(K_S_rho0)):
            self.constant_mass_matrix = True
        else:
            self.constant_mass_matrix = False

        # discretization parameters
        self.polynomial_degree_r = polynomial_degree_r
        self.polynomial_degree_psi = polynomial_degree_psi

        p = max(polynomial_degree_r, polynomial_degree_psi)
        self.nquadrature = nquadrature = int(np.ceil((p + 1) ** 2 / 2))
        # self.nquadrature = nquadrature = p
        self.nelement = nelement

        # chose basis functions
        self.basis_r = basis_r
        self.basis_psi = basis_psi

        if basis_r == "Lagrange":
            self.knot_vector_r = LagrangeKnotVector(polynomial_degree_r, nelement)
        elif basis_r == "B-spline":
            self.knot_vector_r = BSplineKnotVector(polynomial_degree_r, nelement)
        elif basis_r == "Hermite":
            assert (
                polynomial_degree_r == 3
            ), "only cubic Hermite splines are implemented!"
            self.knot_vector_r = HermiteNodeVector(polynomial_degree_r, nelement)
        else:
            raise RuntimeError(f'wrong basis_r: "{basis_r}" was chosen')

        if basis_psi == "Lagrange":
            self.knot_vector_psi = LagrangeKnotVector(polynomial_degree_psi, nelement)
        elif basis_psi == "B-spline":
            self.knot_vector_psi = BSplineKnotVector(polynomial_degree_psi, nelement)
        elif basis_psi == "Hermite":
            assert (
                polynomial_degree_psi == 3
            ), "only cubic Hermite splines are implemented!"
            self.knot_vector_psi = HermiteNodeVector(polynomial_degree_psi, nelement)
        else:
            raise RuntimeError(f'wrong basis_psi: "{basis_psi}" was chosen')

        # build mesh objects
        self.mesh_r = Mesh1D(
            self.knot_vector_r,
            nquadrature,
            dim_q=3,
            derivative_order=1,
            basis=basis_r,
            quadrature="Gauss",
        )

        self.mesh_psi = Mesh1D(
            self.knot_vector_psi,
            nquadrature,
            dim_q=3,
            derivative_order=1,
            basis=basis_psi,
            quadrature="Gauss",
        )

        # total number of nodes
        self.nnodes_r = self.mesh_r.nnodes
        self.nnodes_psi = self.mesh_psi.nnodes

        # number of nodes per element
        self.nnodes_element_r = self.mesh_r.nnodes_per_element
        self.nnodes_element_psi = self.mesh_psi.nnodes_per_element

        # total number of generalized coordinates and velocities
        self.nq_r = self.nu_r = self.mesh_r.nq
        self.nq_psi = self.nu_psi = self.mesh_psi.nq
        self.nq = self.nu = self.nq_r + self.nq_psi

        # number of generalized coordiantes and velocities per element
        self.nq_element_r = self.nu_element_r = self.mesh_r.nq_per_element
        self.nq_element_psi = self.nu_element_psi = self.mesh_psi.nq_per_element
        self.nq_element = self.nu_element = self.nq_element_r + self.nq_element_psi

        # global element connectivity
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_psi = self.mesh_psi.elDOF + self.nq_r
        # qe = q[elDOF[e]] "q^e = C_e,q q"

        # global nodal
        self.nodalDOF_r = self.mesh_r.nodalDOF
        self.nodalDOF_psi = self.mesh_psi.nodalDOF + self.nq_r

        # nodal connectivity on element level
        self.nodalDOF_element_r = self.mesh_r.nodalDOF_element
        self.nodalDOF_element_psi = self.mesh_psi.nodalDOF_element + self.nq_element_r
        # r_OP_i^e = C_r,i^e * C_e,q q = C_r,i^e * q^e
        # r_OPi = qe[nodelDOF_element_r[i]]

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
                # interpolate tangent vector
                r_OP_xi = np.zeros(3, dtype=float)
                for node in range(self.nnodes_element_r):
                    r_OP_xi += (
                        self.N_r_xi[el, i, node] * qe[self.nodalDOF_element_r[node]]
                    )

                # interpoalte transformation matrix and its derivative
                A_IK = np.zeros((3, 3), dtype=float)
                A_IK_xi = np.zeros((3, 3), dtype=float)
                for node in range(self.nnodes_element_psi):
                    A_IK_node = Exp_SO3(qe[self.nodalDOF_element_psi[node]])
                    A_IK += self.N_psi[el, i, node] * A_IK_node
                    A_IK_xi += self.N_psi_xi[el, i, node] * A_IK_node

                # length of reference tangential vector
                J = norm(r_OP_xi)

                # axial and shear strains
                K_Gamma_bar = A_IK.T @ r_OP_xi
                K_Gamma = K_Gamma_bar / J

                # torsional and flexural strains
                d1, d2, d3 = A_IK.T
                d1_xi, d2_xi, d3_xi = A_IK_xi.T
                half_d = d1 @ cross3(d2, d3)
                K_Kappa_bar = (
                    np.array(
                        [
                            0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                            0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                            0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                        ]
                    )
                    / half_d
                )
                K_Kappa = K_Kappa_bar / J

                # safe precomputed quantities for later
                self.J[el, i] = J
                self.K_Gamma0[el, i] = K_Gamma
                self.K_Kappa0[el, i] = K_Kappa

    @staticmethod
    def straight_configuration(
        polynomial_degree_r,
        polynomial_degree_psi,
        basis_r,
        basis_psi,
        nelement,
        L,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
    ):
        if basis_r == "Lagrange":
            nnodes_r = polynomial_degree_r * nelement + 1
        elif basis_r == "B-spline":
            nnodes_r = polynomial_degree_r + nelement
        elif basis_r == "Hermite":
            # nnodes_r = 2 * (nelement + 1)
            nnodes_r = nelement + 1
        else:
            raise RuntimeError(f'wrong basis_r: "{basis_r}" was chosen')

        if basis_psi == "Lagrange":
            nnodes_psi = polynomial_degree_psi * nelement + 1
        elif basis_psi == "B-spline":
            nnodes_psi = polynomial_degree_psi + nelement
        elif basis_psi == "Hermite":
            # nnodes_psi = 2 * (nelement + 1)
            nnodes_psi = nelement + 1
        else:
            raise RuntimeError(f'wrong basis_psi: "{basis_psi}" was chosen')

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

            r0 = np.vstack((x0, y0, z0))
            for i in range(nnodes_r):
                r0[:, i] = r_OP + A_IK @ r0[:, i]

        elif basis_r == "Hermite":
            xis = np.linspace(0, 1, num=nnodes_r)
            r0 = np.zeros((6, nnodes_r))
            t0 = A_IK @ (L * e1)
            for i, xi in enumerate(xis):
                ri = r_OP + xi * t0
                r0[:3, i] = ri
                r0[3:, i] = t0

        # reshape generalized coordinates to nodal ordering
        q_r = r0.reshape(-1, order="F")

        # we have to extract the rotation vector from the given rotation matrix
        # and set its value for each node
        if basis_psi == "Hermite":
            raise NotImplementedError
        psi = Log_SO3(A_IK)
        q_psi = np.tile(psi, nnodes_psi)

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
        raise NotImplementedError
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
    def fit_orientation(
        A_IKs,
        polynomial_degree,
        nelement,
    ):
        raise NotImplementedError
        # number of sample points
        n_samples = len(A_IKs)

        # linear spaced xi's for target curve points
        xis = np.linspace(0, 1, n_samples)
        node_vector = LagrangeKnotVector(polynomial_degree, nelement)

        # build mesh object
        mesh = Mesh1D(
            node_vector,
            1,
            dim_q=3,
            derivative_order=0,
            basis="Lagrange",
            quadrature="Gauss",
        )

        # build initial vector of orientations
        nq = mesh.nnodes * 3
        q0 = np.zeros(nq, dtype=float)

        xi_idx = np.array(
            [np.where(xis >= node_vector.data[i])[0][0] for i in range(mesh.nnodes)]
        )

        # warm start for optimization
        q0 = np.array([Log_SO3(A_IKs[idx]) for idx in xi_idx]).reshape(-1)

        def residual(q):
            nxis = len(xis)
            R = np.zeros(3 * nxis, dtype=q.dtype)
            for i, xii in enumerate(xis):
                # find element number and extract elemt degrees of freedom
                el = node_vector.element_number(xii)[0]
                elDOF = mesh.elDOF[el]
                qe = q[elDOF]

                # evaluate shape functions
                N = mesh.eval_basis(xii)

                # interpoalte rotations
                A_IK = np.zeros((3, 3), dtype=q.dtype)
                for node in range(mesh.nnodes_per_element):
                    A_IK += N[node] * Exp_SO3(qe[mesh.nodalDOF_element[node]])

                # compute relative rotation vector
                psi_rel = Log_SO3(A_IKs[i].T @ A_IK)

                # insert to residual
                R[3 * i : 3 * (i + 1)] = psi_rel

            return R

        def jacobian(q):
            nxis = len(xis)
            nq = len(q)
            J = np.zeros((3 * nxis, nq), dtype=float)

            for i, xii in enumerate(xis):
                # find element number and extract elemt degrees of freedom
                el = node_vector.element_number(xii)[0]
                elDOF = mesh.elDOF[el]
                qe = q[elDOF]
                nqe = len(qe)

                # evaluate shape functions
                N = mesh.eval_basis(xii)

                # interpoalte rotations
                A_IK = np.zeros((3, 3), dtype=float)
                A_IK_qe = np.zeros((3, 3, nqe), dtype=float)
                for node in range(mesh.nnodes_per_element):
                    nodalDOF = mesh.nodalDOF_element[node]
                    qe_node = qe[nodalDOF]
                    A_IK += N[node] * Exp_SO3(qe_node)
                    A_IK_qe[:, :, nodalDOF] += N[node] * Exp_SO3_psi(qe_node)

                # compute relative rotation vector
                psi_rel_A_rel = Log_SO3_A(A_IKs[i].T @ A_IK)
                psi_rel_qe = np.einsum(
                    "ikl,mk,mlj->ij", psi_rel_A_rel, A_IKs[i], A_IK_qe
                )

                # insert to residual
                J[3 * i : 3 * (i + 1), elDOF] = psi_rel_qe

                # def psi_rel(qe):
                #     # evaluate shape functions
                #     N = mesh.eval_basis(xii)

                #     # interpoalte rotations
                #     A_IK = np.zeros((3, 3), dtype=q.dtype)
                #     for node in range(mesh.nnodes_per_element):
                #         A_IK += N[node] * Exp_SO3(qe[mesh.nodalDOF_element[node]])

                #     # compute relative rotation vector
                #     return Log_SO3(A_IKs[i].T @ A_IK)

                # psi_rel_qe_num = approx_fprime(qe, psi_rel)
                # diff = psi_rel_qe - psi_rel_qe_num
                # error = np.linalg.norm(diff)
                # print(f"error psi_rel_qe: {error}")

                # # insert to residual
                # J[3 * i:3 * (i + 1), elDOF] = psi_rel_qe_num

            return J

        from scipy.optimize import minimize, least_squares

        res = least_squares(
            residual,
            q0,
            # jac="cs",
            jac=jacobian,
            verbose=2,
        )

        # def cost_function(q):
        #     R = residual(q)
        #     return 0.5 * np.sqrt(R @ R)

        # method = "SLSQP"
        # res = minimize(
        #     cost_function,
        #     q0,
        #     method=method,
        #     tol=1.0e-3,
        #     options={
        #         "maxiter": 1000,
        #         "disp": True,
        #     },
        # )

        return res.x

    def element_number(self, xi):
        return self.knot_vector_r.element_number(xi)[0]

    #########################################
    # equations of motion
    #########################################
    def assembler_callback(self):
        if self.constant_mass_matrix:
            self.__M_coo()

    def M_el_constant(self, el):
        M_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # delta_r A_rho0 r_ddot part
            M_el_r_r = np.eye(3) * self.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N_r[el, i, node_a] * self.N_r[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_psi_psi = self.K_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
                        self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
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
            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N_r[el, i, node_a] * self.N_r[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_psi_psi = self.K_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
                        self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
                    )

            # For non symmetric cross sections there are also other parts
            # involved in the mass matrix. These parts are configuration
            # dependent and lead to configuration dependent mass matrix.
            _, A_IK, _, _ = self.eval(qe, self.qp[el, i])
            M_el_r_psi = A_IK @ self.K_S_rho0 * Ji * qwi
            M_el_psi_r = A_IK @ self.K_S_rho0 * Ji * qwi

            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_psi * (
                        self.N_r[el, i, node_a] * self.N_psi[el, i, node_b]
                    )
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_r * (
                        self.N_psi[el, i, node_a] * self.N_r[el, i, node_b]
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

    def f_gyr_el(self, t, qe, ue, el):
        f_gyr_el = np.zeros(self.nq_element, dtype=float)

        for i in range(self.nquadrature):
            # interpoalte angular velocity
            K_Omega = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_psi):
                K_Omega += self.N_psi[el, i, node] * ue[self.nodalDOF_element_psi[node]]

            # vector of gyroscopic forces
            f_gyr_el_psi = (
                cross3(K_Omega, self.K_I_rho0 @ K_Omega)
                * self.J[el, i]
                * self.qw[el, i]
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
                ((ax2skew(K_Omega) @ self.K_I_rho0 - ax2skew(self.K_I_rho0 @ K_Omega)))
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
        E_pot_el = 0.0

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            J = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # interpolate tangent vector
            r_OP_xi = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_r):
                r_OP_xi += self.N_r_xi[el, i, node] * qe[self.nodalDOF_element_r[node]]

            # interpolate transformation matrix and its derivative
            A_IK = np.zeros((3, 3), dtype=float)
            A_IK_xi = np.zeros((3, 3), dtype=float)
            for node in range(self.nnodes_element_psi):
                A_IK_node = Exp_SO3(qe[self.nodalDOF_element_psi[node]])
                A_IK += self.N_psi[el, i, node] * A_IK_node
                A_IK_xi += self.N_psi_xi[el, i, node] * A_IK_node

            # axial and shear strains
            K_Gamma_bar = A_IK.T @ r_OP_xi
            K_Gamma = K_Gamma_bar / J

            # torsional and flexural strains
            d1, d2, d3 = A_IK.T
            d1_xi, d2_xi, d3_xi = A_IK_xi.T
            half_d = d1 @ cross3(d2, d3)
            K_Kappa_bar = (
                np.array(
                    [
                        0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                        0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                        0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                    ]
                )
                / half_d
            )
            K_Kappa = K_Kappa_bar / J

            # evaluate strain energy function
            E_pot_el += (
                self.material_model.potential(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
                * J
                * qwi
            )

        return E_pot_el

    def h(self, t, q, u):
        h = np.zeros(self.nu, dtype=q.dtype)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            h[elDOF] += self.h_el(q[elDOF], el)
        return h

    def h_el(self, qe, el):
        f_pot_el = np.zeros(self.nq_element, dtype=qe.dtype)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            J = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # interpolate tangent vector
            r_OP_xi = np.zeros(3, dtype=qe.dtype)
            for node in range(self.nnodes_element_r):
                r_OP_xi += self.N_r_xi[el, i, node] * qe[self.nodalDOF_element_r[node]]

            # interpolate transformation matrix and its derivative
            A_IK = np.zeros((3, 3), dtype=qe.dtype)
            A_IK_xi = np.zeros((3, 3), dtype=qe.dtype)
            for node in range(self.nnodes_element_psi):
                A_IK_node = Exp_SO3(qe[self.nodalDOF_element_psi[node]])
                A_IK += self.N_psi[el, i, node] * A_IK_node
                A_IK_xi += self.N_psi_xi[el, i, node] * A_IK_node

            # axial and shear strains
            K_Gamma_bar = A_IK.T @ r_OP_xi
            K_Gamma = K_Gamma_bar / J

            # torsional and flexural strains
            d1, d2, d3 = A_IK.T
            d1_xi, d2_xi, d3_xi = A_IK_xi.T
            half_d = d1 @ cross3(d2, d3)
            K_Kappa_bar = (
                np.array(
                    [
                        0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                        0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                        0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                    ]
                )
                / half_d
            )
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

            for node in range(self.nnodes_element_psi):
                f_pot_el[self.nodalDOF_element_psi[node]] -= (
                    self.N_psi_xi[el, i, node] * K_m * qwi
                )

                f_pot_el[self.nodalDOF_element_psi[node]] += (
                    self.N_psi[el, i, node]
                    * (cross3(K_Gamma_bar, K_n) + cross3(K_Kappa_bar, K_m))
                    * qwi
                )

        return f_pot_el

    def h_q(self, t, q, u, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            h_q_el = self.h_q_el(q[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(h_q_el, (self.uDOF[elDOF], self.qDOF[elDOF]))

    def h_q_el(self, qe, el):
        f_pot_q_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            J = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # interpolate tangent vector
            r_OP_xi = np.zeros(3, dtype=qe.dtype)
            r_OP_xi_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
            for node in range(self.nnodes_element_r):
                nodalDOF_r = self.nodalDOF_element_r[node]
                r_OP_xi += self.N_r_xi[el, i, node] * qe[nodalDOF_r]
                r_OP_xi_qe[:, nodalDOF_r] += self.N_r_xi[el, i, node] * np.eye(
                    3, dtype=float
                )

            # interpolate transformation matrix and its derivative
            A_IK = np.zeros((3, 3), dtype=qe.dtype)
            A_IK_xi = np.zeros((3, 3), dtype=qe.dtype)
            A_IK_qe = np.zeros((3, 3, self.nq_element), dtype=qe.dtype)
            A_IK_xi_qe = np.zeros((3, 3, self.nq_element), dtype=qe.dtype)
            for node in range(self.nnodes_element_psi):
                nodalDOF_psi = self.nodalDOF_element_psi[node]
                psi_node = qe[nodalDOF_psi]
                A_IK_node = Exp_SO3(psi_node)
                A_IK_q_node = Exp_SO3_psi(psi_node)

                A_IK += self.N_psi[el, i, node] * A_IK_node
                A_IK_xi += self.N_psi_xi[el, i, node] * A_IK_node
                A_IK_qe[:, :, nodalDOF_psi] += self.N_psi[el, i, node] * A_IK_q_node
                A_IK_xi_qe[:, :, nodalDOF_psi] += (
                    self.N_psi_xi[el, i, node] * A_IK_q_node
                )

            # extract directors
            d1, d2, d3 = A_IK.T
            d1_xi, d2_xi, d3_xi = A_IK_xi.T
            d1_qe, d2_qe, d3_qe = A_IK_qe.transpose(1, 0, 2)
            d1_xi_qe, d2_xi_qe, d3_xi_qe = A_IK_xi_qe.transpose(1, 0, 2)

            # axial and shear strains
            K_Gamma_bar = A_IK.T @ r_OP_xi
            K_Gamma = K_Gamma_bar / J
            K_Gamma_bar_qe = np.einsum("k,kij", r_OP_xi, A_IK_qe) + A_IK.T @ r_OP_xi_qe
            K_Gamma_qe = K_Gamma_bar_qe / J

            # torsional and flexural strains
            half_d = d1 @ cross3(d2, d3)
            half_d_qe = (
                cross3(d2, d3) @ d1_qe + cross3(d3, d1) @ d2_qe + cross3(d1, d2) @ d3_qe
            )
            K_Kappa_bar = (
                np.array(
                    [
                        0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                        0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                        0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                    ]
                )
                / half_d
            )
            K_Kappa_bar_qe = np.array(
                [
                    0.5
                    * (d3 @ d2_xi_qe + d2_xi @ d3_qe - d2 @ d3_xi_qe - d3_xi @ d2_qe),
                    0.5
                    * (d1 @ d3_xi_qe + d3_xi @ d1_qe - d3 @ d1_xi_qe - d1_xi @ d3_qe),
                    0.5
                    * (d2 @ d1_xi_qe + d1_xi @ d2_qe - d1 @ d2_xi_qe - d2_xi @ d1_qe),
                ]
            ) / half_d - np.outer(K_Kappa_bar / half_d, half_d_qe)
            K_Kappa = K_Kappa_bar / J
            K_Kappa_qe = K_Kappa_bar_qe / J

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
            for node in range(self.nnodes_element_r):
                f_pot_q_el[self.nodalDOF_element_r[node], :] -= (
                    self.N_r_xi[el, i, node]
                    * qwi
                    * (np.einsum("ikj,k->ij", A_IK_qe, K_n) + A_IK @ K_n_qe)
                )

            for node in range(self.nnodes_element_psi):
                f_pot_q_el[self.nodalDOF_element_psi[node], :] += (
                    self.N_psi[el, i, node]
                    * qwi
                    * (ax2skew(K_Gamma_bar) @ K_n_qe - ax2skew(K_n) @ K_Gamma_bar_qe)
                )

                f_pot_q_el[self.nodalDOF_element_psi[node], :] -= (
                    self.N_psi_xi[el, i, node] * qwi * K_m_qe
                )

                f_pot_q_el[self.nodalDOF_element_psi[node], :] += (
                    self.N_psi[el, i, node]
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
        for node in range(self.nnodes_psi):
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
        for node in range(self.nnodes_psi):
            nodalDOF_psi = self.nodalDOF_psi[node]

            psi = q[nodalDOF_psi]
            coo.extend(
                T_SO3_inv(psi),
                (self.qDOF[nodalDOF_psi], self.uDOF[nodalDOF_psi]),
            )

    def q_ddot(self, t, q, u, u_dot):
        # centerline part
        q_ddot = u_dot

        # correct axis angle vector part
        for node in range(self.nnodes_psi):
            nodalDOF_psi = self.nodalDOF_psi[node]

            psi = q[nodalDOF_psi]
            K_omega_IK = u[nodalDOF_psi]
            K_omega_IK_dot = u_dot[nodalDOF_psi]

            T_inv = T_SO3_inv(psi)
            psi_dot = T_inv @ K_omega_IK

            # TODO:
            T_dot = tangent_map_s(psi, psi_dot)
            Tinv_dot = -T_inv @ T_dot @ T_inv
            psi_ddot = T_inv @ K_omega_IK_dot + Tinv_dot @ K_omega_IK

            # psi_ddot = (
            #     T_inv @ K_omega_IK_dot
            #     + np.einsum("ijk,j,k",
            #         approx_fprime(psi, T_SO3_inv, eps=1.0e-10, method="cs"),
            #         K_omega_IK,
            #         psi_dot
            #     )
            # )

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
        for node in range(self.nnodes_psi):
            psi = q[self.nodalDOF_psi[node]]
            q[self.nodalDOF_psi[node]] = DirectorAxisAngle.psi_C(psi)
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
        # evaluate shape functions
        N_r, _ = self.basis_functions_r(frame_ID[0])
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate centerline
        r = np.zeros(3, dtype=q.dtype)
        for node in range(self.nnodes_element_r):
            r += N_r[node] * q[self.nodalDOF_element_r[node]]

        # interpolate orientation
        A_IK = np.zeros((3, 3), dtype=q.dtype)
        for node in range(self.nnodes_element_psi):
            A_IK += N_psi[node] * Exp_SO3(q[self.nodalDOF_element_psi[node]])

        return r + A_IK @ K_r_SP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # evaluate shape functions
        N_r, _ = self.basis_functions_r(frame_ID[0])
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate centerline position
        r_q = np.zeros((3, self.nq_element), dtype=q.dtype)
        for node in range(self.nnodes_element_r):
            nodalDOF_r = self.nodalDOF_element_r[node]
            r_q[:, nodalDOF_r] += N_r[node] * np.eye(3, dtype=float)

        # interpolate orientation
        A_IK = np.zeros((3, 3), dtype=q.dtype)
        A_IK_q = np.zeros((3, 3, self.nq_element), dtype=q.dtype)
        for node in range(self.nnodes_element_psi):
            nodalDOF_psi = self.nodalDOF_element_psi[node]
            psi_node = q[nodalDOF_psi]
            A_IK += N_psi[node] * Exp_SO3(psi_node)
            A_IK_q[:, :, nodalDOF_psi] += N_psi[node] * Exp_SO3_psi(psi_node)

        r_OP_q = r_q + np.einsum("k,kij", K_r_SP, A_IK_q)
        return r_OP_q

        # r_OP_q_num = approx_fprime(
        #     q, lambda q: self.r_OP(t, q, frame_ID, K_r_SP), eps=1.0e-10, method="cs"
        # )
        # diff = r_OP_q - r_OP_q_num
        # error = np.linalg.norm(diff)
        # print(f"error r_OP_q: {error}")
        # return r_OP_q_num

    def A_IK(self, t, q, frame_ID):
        # evaluate shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate orientation
        A_IK = np.zeros((3, 3), dtype=q.dtype)
        for node in range(self.nnodes_element_psi):
            A_IK += N_psi[node] * Exp_SO3(q[self.nodalDOF_element_psi[node]])

        return A_IK

    def A_IK_q(self, t, q, frame_ID):
        # evaluate shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate centerline position and orientation
        A_IK_q = np.zeros((3, 3, self.nq_element), dtype=q.dtype)
        for node in range(self.nnodes_element_psi):
            nodalDOF_psi = self.nodalDOF_element_psi[node]
            A_IK_q[:, :, nodalDOF_psi] += N_psi[node] * Exp_SO3_psi(q[nodalDOF_psi])

        return A_IK_q

        # A_IK_q_num = approx_fprime(
        #     q, lambda q: self.A_IK(t, q, frame_ID), eps=1.0e-10, method="cs"
        # )
        # diff = A_IK_q - A_IK_q_num
        # error = np.linalg.norm(diff)
        # print(f"error A_IK_q: {error}")
        # return A_IK_q_num

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        N_r, _ = self.basis_functions_r(frame_ID[0])
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate orientation
        A_IK = np.zeros((3, 3), dtype=q.dtype)
        for node in range(self.nnodes_element_psi):
            A_IK += N_psi[node] * Exp_SO3(q[self.nodalDOF_element_psi[node]])

        # angular velocity in K-frame
        K_Omega = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_psi):
            K_Omega += N_psi[node] * u[self.nodalDOF_element_psi[node]]

        # centerline velocity
        v = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_r):
            v += N_r[node] * u[self.nodalDOF_element_r[node]]

        return v + A_IK @ cross3(K_Omega, K_r_SP)

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
        N_r, _ = self.basis_functions_r(frame_ID[0])
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate orientation
        A_IK = np.zeros((3, 3), dtype=q.dtype)
        for node in range(self.nnodes_element_psi):
            A_IK += N_psi[node] * Exp_SO3(q[self.nodalDOF_element_psi[node]])

        # skew symmetric matrix of K_r_SP
        K_r_SP_tilde = ax2skew(K_r_SP)

        # interpolate centerline and axis angle contributions
        J_P = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_r):
            J_P[:, self.nodalDOF_element_r[node]] += N_r[node] * np.eye(
                3, dtype=q.dtype
            )
        for node in range(self.nnodes_element_psi):
            J_P[:, self.nodalDOF_element_psi[node]] -= N_psi[node] * A_IK @ K_r_SP_tilde

        return J_P

        # J_P_num = approx_fprime(
        #     np.zeros(self.nq_element, dtype=float),
        #     lambda u: self.v_P(t, q, u, frame_ID, K_r_SP),
        # )
        # diff = J_P_num - J_P
        # error = np.linalg.norm(diff)
        # print(f"error J_P: {error}")
        # return J_P_num

    # TODO:
    def J_P_q(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # evaluate required nodal shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        K_r_SP_tilde = ax2skew(K_r_SP)
        A_IK_q = self.A_IK_q(t, q, frame_ID)
        prod = np.einsum("ijl,jk", A_IK_q, K_r_SP_tilde)

        # interpolate axis angle contributions since centerline contributon is
        # zero
        J_P_q = np.zeros((3, self.nq_element, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_psi):
            nodalDOF_psi = self.nodalDOF_element_psi[node]
            J_P_q[:, nodalDOF_psi] -= N_psi[node] * prod

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
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate orientation
        A_IK = np.zeros((3, 3), dtype=q.dtype)
        for node in range(self.nnodes_element_psi):
            A_IK += N_psi[node] * Exp_SO3(q[self.nodalDOF_element_psi[node]])

        # centerline acceleration
        a = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_r):
            a += N_r[node] * u_dot[self.nodalDOF_element_r[node]]

        # angular velocity and acceleration in K-frame
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID)

        # rigid body formular
        return a + A_IK @ (
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

        N, _ = self.basis_functions_r(frame_ID[0])
        a_P_u = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_r):
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
        N_psi, _ = self.basis_functions_psi(frame_ID[0])
        K_Omega = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_psi):
            K_Omega += N_psi[node] * u[self.nodalDOF_element_psi[node]]
        return K_Omega

    def K_Omega_q(self, t, q, u, frame_ID):
        return np.zeros((3, self.nq_element), dtype=float)

    def K_J_R(self, t, q, frame_ID):
        N_psi, _ = self.basis_functions_psi(frame_ID[0])
        K_J_R = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_psi):
            K_J_R[:, self.nodalDOF_element_psi[node]] += N_psi[node] * np.eye(
                3, dtype=float
            )
        return K_J_R

    def K_J_R_q(self, t, q, frame_ID):
        return np.zeros((3, self.nq_element, self.nq_element), dtype=float)

    def K_Psi(self, t, q, u, u_dot, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        time derivative of the angular velocities in the K-frame.
        """
        N_psi, _ = self.basis_functions_psi(frame_ID[0])
        K_Psi = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_psi):
            K_Psi += N_psi[node] * u_dot[self.nodalDOF_element_psi[node]]
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
        if self.basis_r == "Hermite":
            r = np.zeros((3, int(self.nnodes_r / 2)), dtype=float)
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
