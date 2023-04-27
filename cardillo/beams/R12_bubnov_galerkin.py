import numpy as np
from abc import ABCMeta, abstractmethod
import meshio
import os

from cardillo.utility.coo_matrix import CooMatrix
from cardillo.discretization.b_spline import BSplineKnotVector
from cardillo.discretization.lagrange import LagrangeKnotVector
from cardillo.math.algebra import norm
from cardillo.math import approx_fprime
from cardillo.discretization.mesh1D import Mesh1D
from cardillo.beams._base_petrov_galerkin import RodExportBase


class I_R12_BubonvGalerkin_R12(RodExportBase, metaclass=ABCMeta):
    def __init__(
        self,
        cross_section,
        material_model,
        A_rho0,
        B_rho0,
        C_rho0,
        polynomial_degree_r,
        polynomial_degree_di,
        nquadrature,
        nelement,
        Q,
        q0=None,
        u0=None,
        basis="Lagrange",
    ):
        super().__init__(cross_section)

        # beam properties
        self.materialModel = material_model  # material model
        self.A_rho0 = A_rho0  # line density
        self.B_rho0 = B_rho0  # first moment
        self.C_rho0 = C_rho0  # second moment

        # material model
        self.material_model = material_model

        # discretization parameters
        self.polynomial_degree_r = polynomial_degree_r  # polynomial degree centerline
        self.polynomial_degree_di = polynomial_degree_di  # polynomial degree directors
        self.nquadrature = nquadrature  # number of quadrature points
        self.nelement = nelement  # number of elements

        self.basis = basis
        if basis == "B-spline":
            self.knot_vector_r = BSplineKnotVector(polynomial_degree_r, nelement)
            self.knot_vector_di = BSplineKnotVector(polynomial_degree_di, nelement)
        elif basis == "Lagrange":
            self.knot_vector_r = LagrangeKnotVector(polynomial_degree_r, nelement)
            self.knot_vector_di = LagrangeKnotVector(polynomial_degree_di, nelement)
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        # number of degrees of freedom per node
        self.nq_node_r = nq_node_r = 3
        self.nq_node_di = nq_node_di = 9

        # build mesh objects
        self.mesh_r = Mesh1D(
            self.knot_vector_r,
            nquadrature,
            derivative_order=1,
            basis=basis,
            dim_q=nq_node_r,
        )
        self.mesh_di = Mesh1D(
            self.knot_vector_di,
            nquadrature,
            derivative_order=1,
            basis=basis,
            dim_q=nq_node_di,
        )

        # total number of nodes
        self.nnode_r = self.mesh_r.nnodes
        self.nnode_di = self.mesh_di.nnodes

        # number of nodes per element
        self.nnodes_element_r = self.mesh_r.nnodes_per_element
        self.nnodes_element_di = self.mesh_di.nnodes_per_element

        # total number of generalized coordinates
        self.nq_r = self.mesh_r.nq
        self.nq_di = self.mesh_di.nq
        self.nq = self.nq_r + self.nq_di  # total number of generalized coordinates
        self.nu = self.nq  # total number of generalized velocities

        # number of generalized coordiantes per element
        self.nq_element_r = self.mesh_r.nq_per_element
        self.nq_element_di = self.mesh_di.nq_per_element
        self.nq_element = self.nu_element = self.nq_element_r + self.nq_element_di

        # global element connectivity
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_di = self.mesh_di.elDOF + self.nq_r

        # global nodal
        self.nodalDOF_r = self.mesh_r.nodalDOF
        self.nodalDOF_di = self.mesh_di.nodalDOF + self.nq_r

        # nodal connectivity on element level
        self.nodalDOF_element_r = self.mesh_r.nodalDOF_element
        self.nodalDOF_element_di = self.mesh_di.nodalDOF_element + self.nq_element_r

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nelement, self.nq_element), dtype=int)
        for el in range(nelement):
            self.elDOF[el, : self.nq_element_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_element_r :] = self.elDOF_di[el]

        # shape functions and their first derivatives
        self.N_r = self.mesh_r.N
        self.N_r_xi = self.mesh_r.N_xi
        self.N_di = self.mesh_di.N
        self.N_di_xi = self.mesh_di.N_xi

        # quadrature points
        self.qp = self.mesh_r.qp  # quadrature points
        self.qw = self.mesh_r.wp  # quadrature weights

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions_r = self.mesh_r.eval_basis
        self.basis_functions_di = self.mesh_di.eval_basis

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
                # interpoalte centerline
                r_xi = np.zeros(3, dtype=float)
                for node in range(self.nnodes_element_r):
                    r_xi += self.N_r_xi[el, i, node] * qe[self.nodalDOF_element_r[node]]

                # interpoalte transformation matrix
                A_IK = np.zeros((3, 3), dtype=float)
                A_IK_xi = np.zeros((3, 3), dtype=float)
                for node in range(self.nnodes_element_di):
                    A_IK_node = qe[self.nodalDOF_element_di[node]].reshape(3, 3)
                    A_IK += self.N_di[el, i, node] * A_IK_node
                    A_IK_xi += self.N_di_xi[el, i, node] * A_IK_node

                # extract directors
                d1, d2, d3 = A_IK.T
                d1_xi, d2_xi, d3_xi = A_IK_xi.T

                # length of the tangent vector
                J = norm(r_xi)
                self.J[el, i] = J

                # compute derivatives w.r.t. the arc lenght parameter s
                r_s = r_xi / J
                d1_s = d1_xi / J
                d2_s = d2_xi / J
                d3_s = d3_xi / J

                # axial and shear strains
                self.K_Gamma0[el, i] = A_IK.T @ r_s

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
        polynomial_degree_di,
        nelement,
        L,
        greville_abscissae=True,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
        basis="B-spline",
    ):
        if basis == "B-spline":
            nn_r = polynomial_degree_r + nelement
            nn_di = polynomial_degree_di + nelement
        elif basis == "Lagrange":
            nn_r = polynomial_degree_r * nelement + 1
            nn_di = polynomial_degree_di * nelement + 1
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        x0 = np.linspace(0, L, num=nn_r)
        y0 = np.zeros(nn_r, dtype=float)
        z0 = np.zeros(nn_r, dtype=float)
        if greville_abscissae and basis == "B-spline":
            kv = BSplineKnotVector.uniform(polynomial_degree_r, nelement)
            for i in range(nn_r):
                x0[i] = np.sum(kv[i + 1 : i + polynomial_degree_r + 1])
            x0 = x0 * L / polynomial_degree_r

        r0 = np.vstack((x0, y0, z0))
        for i in range(nn_r):
            r0[:, i] = r_OP + A_IK @ r0[:, i]

        # reshape generalized coordinates to nodal ordering
        q_r = r0.reshape(-1, order="C")

        # linearize transformation matrix
        A_IK_node = A_IK.reshape(-1)
        q_di = np.repeat(A_IK_node, nn_di)

        return np.concatenate([q_r, q_di])

    @staticmethod
    def fit_field(
        points,
        polynomial_degree,
        nelement,
    ):
        # number of sample points
        # n_samples = len(points)
        n_samples, dim = points.shape

        # linear spaced xi's for target curve points
        xis = np.linspace(0, 1, n_samples)
        knot_vector = BSplineKnotVector(polynomial_degree, nelement)

        # build mesh object
        mesh = Mesh1D(
            knot_vector,
            1,
            dim_q=dim,
            derivative_order=0,
            basis="B-spline",
            quadrature="Gauss",
        )

        # build initial vector of orientations
        nq = mesh.nnodes * dim
        q0 = np.zeros(nq, dtype=float)

        # find best matching indices
        xi_idx = np.array(
            [np.where(xis >= knot_vector.data[i])[0][0] for i in range(mesh.nnodes)]
        )

        # warm start for optimization
        q0 = np.array([points[idx] for idx in xi_idx]).reshape(-1)

        def residual(q):
            R = np.zeros(dim * n_samples, dtype=q.dtype)
            for i, xii in enumerate(xis):
                # find element number and extract elemt degrees of freedom
                el = knot_vector.element_number(xii)[0]
                elDOF = mesh.elDOF[el]
                qe = q[elDOF]

                # evaluate shape functions
                N = mesh.eval_basis(xii)

                # interpoalte rotations
                f = np.zeros(dim, dtype=q.dtype)
                for node in range(mesh.nnodes_per_element):
                    f += N[node] * qe[mesh.nodalDOF_element[node]]

                # insert to residual
                R[dim * i : dim * (i + 1)] = points[i] - f

            return R

        from scipy.optimize import least_squares, minimize

        res = least_squares(
            residual,
            q0,
            jac="cs",
            verbose=2,
            ftol=1.0e-8,
            gtol=1.0e-8,
        )

        # def cost_function(q):
        #     R = residual(q)
        #     return 0.5 * np.sqrt(R @ R)

        # method = "SLSQP"
        # res = minimize(
        #     cost_function,
        #     q0,
        #     method=method,
        #     tol=1.0e-8,
        #     options={
        #         "maxiter": 1000,
        #         "disp": True,
        #     },
        # )

        return res.x

    def element_number(self, xi):
        # note the elements coincide for both meshes!
        return self.knot_vector_r.element_number(xi)[0]

    def assembler_callback(self):
        self.__M_coo()

    #########################################
    # equations of motion
    #########################################
    def M_el(self, el):
        Me = np.zeros((self.nq_element, self.nq_element), dtype=float)
        return Me
        print(f"M not implemented")

        for i in range(self.nquadrature):
            # build matrix of shape function derivatives
            NN_r_i = self.stack3r(self.N_r[el, i])
            NN_di_i = self.stack3di(self.N_di[el, i])

            # extract reference state variables
            Ji = self.J[el, i]
            qwi = self.qw[el, i]
            factor_rr = NN_r_i.T @ NN_r_i * Ji * qwi
            factor_rdi = NN_r_i.T @ NN_di_i * Ji * qwi
            factor_dir = NN_di_i.T @ NN_r_i * Ji * qwi
            factor_didi = NN_di_i.T @ NN_di_i * Ji * qwi

            # delta r * ddot r
            Me[self.rDOF[:, None], self.rDOF] += self.A_rho0 * factor_rr
            # delta r * ddot d2
            Me[self.rDOF[:, None], self.d2DOF] += self.B_rho0[1] * factor_rdi
            # delta r * ddot d3
            Me[self.rDOF[:, None], self.d3DOF] += self.B_rho0[2] * factor_rdi

            # delta d2 * ddot r
            Me[self.d2DOF[:, None], self.rDOF] += self.B_rho0[1] * factor_dir
            Me[self.d2DOF[:, None], self.d2DOF] += self.C_rho0[1, 1] * factor_didi
            Me[self.d2DOF[:, None], self.d3DOF] += self.C_rho0[1, 2] * factor_didi

            # delta d3 * ddot r
            Me[self.d3DOF[:, None], self.rDOF] += self.B_rho0[2] * factor_dir
            Me[self.d3DOF[:, None], self.d2DOF] += self.C_rho0[2, 1] * factor_didi
            Me[self.d3DOF[:, None], self.d3DOF] += self.C_rho0[2, 2] * factor_didi

        return Me

    def __M_coo(self):
        self.__M = CooMatrix((self.nu, self.nu))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            self.__M[elDOF, elDOF] = self.M_el(el)

    def M(self, t, q):
        return self.__M

    def E_pot(self, t, q):
        E = 0
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            E += self.E_pot_el(q[elDOF], el)
        return E

    def E_pot_el(self, qe, el):
        Ee = 0
        for i in range(self.nquadrature):
            # extract reference state variables
            J = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # interpoalte centerline
            r_xi = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_r):
                r_xi += self.N_r_xi[el, i, node] * qe[self.nodalDOF_element_r[node]]

            # interpoalte directors
            d1d2d3 = np.zeros(9, dtype=qe.dtype)
            d1d2d3_xi = np.zeros(9, dtype=qe.dtype)
            for node in range(self.nnodes_element_di):
                nodalDOF = self.nodalDOF_element_di[node]
                qe_node = qe[nodalDOF]
                d1d2d3 += self.N_di[el, i, node] * qe_node
                d1d2d3_xi += self.N_di_xi[el, i, node] * qe_node

            d1, d2, d3 = np.split(d1d2d3, 3)
            d1_xi, d2_xi, d3_xi = np.split(d1d2d3_xi, 3)

            # axial and shear strains
            K_Gamma = np.array([r_xi @ d1, r_xi @ d2, r_xi @ d3]) / J

            # torsional and flexural strains
            K_Kappa = (
                np.array(
                    [
                        0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                        0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                        0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                    ]
                )
                / J
            )

            # evaluate strain energy function
            Ee += (
                self.material_model.potential(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
                * J
                * self.qw[el, i]
            )

        return Ee

    def h(self, t, q, u):
        f = np.zeros(self.nu)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f[elDOF] += self.f_pot_el(q[elDOF], el)
        return f

    def f_pot_el(self, qe, el):
        fe = np.zeros(self.nq_element, dtype=qe.dtype)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            J = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # interpoalte centerline
            r_xi = np.zeros(3, dtype=qe.dtype)
            for node in range(self.nnodes_element_r):
                r_xi += self.N_r_xi[el, i, node] * qe[self.nodalDOF_element_r[node]]

            # interpoalte directors
            d1d2d3 = np.zeros(9, dtype=qe.dtype)
            d1d2d3_xi = np.zeros(9, dtype=qe.dtype)
            for node in range(self.nnodes_element_di):
                nodalDOF = self.nodalDOF_element_di[node]
                qe_node = qe[nodalDOF]
                d1d2d3 += self.N_di[el, i, node] * qe_node
                d1d2d3_xi += self.N_di_xi[el, i, node] * qe_node

            d1, d2, d3 = np.split(d1d2d3, 3)
            d1_xi, d2_xi, d3_xi = np.split(d1d2d3_xi, 3)

            # axial and shear strains
            K_Gamma = np.array([r_xi @ d1, r_xi @ d2, r_xi @ d3]) / J

            # torsional and flexural strains
            K_Kappa = (
                np.array(
                    [
                        0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                        0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                        0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                    ]
                )
                / J
            )

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)

            # quadrature point contribution to element residual
            for node in range(self.nnodes_element_r):
                fe[self.nodalDOF_element_r[node]] -= (
                    self.N_r_xi[el, i, node]
                    * (K_n[0] * d1 + K_n[1] * d2 + K_n[2] * d3)
                    * qwi
                )

            di = [d1, d2, d3]
            di_xi = [d1_xi, d2_xi, d3_xi]
            for node in range(self.nnodes_element_di):
                N_di = self.N_di[el, i, node]
                N_di_xi = self.N_di_xi[el, i, node]

                def fe_di(i1):
                    i1, i2, i3 = np.roll([0, 1, 2], (i1 + 1, i1 + 2))
                    return (
                        N_di * K_n[i1] * r_xi
                        + 0.5 * N_di * (K_m[i2] * di_xi[i3] - K_m[i3] * di_xi[i2])
                        + 0.5 * N_di_xi * (K_m[i3] * di[i2] - K_m[i2] * di[i3])
                    )

                fe[self.nodalDOF_element_di[node]] -= (
                    np.concatenate(list(map(fe_di, [0, 1, 2]))) * qwi
                )

        return fe

        # fe_num = approx_fprime(
        #     qe,
        #     lambda qe: -self.E_pot_el(qe, el),
        #     method="3-point",
        # )
        # diff = fe - fe_num
        # error = np.linalg.norm(diff)
        # print(f"error: {error}")
        # return fe_num

    def h_q(self, t, q, u):
        coo = CooMatrix((self.nu, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            coo[elDOF, elDOF] = self.f_pot_q_el(q[elDOF], el)

        return coo

    def f_pot_q_el(self, qe, el):
        Ke = np.zeros((self.nq_element, self.nq_element))

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            J = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # interpoalte centerline
            r_xi = np.zeros(3, dtype=float)
            r_xi_q = np.zeros((3, self.nq_element), dtype=float)
            for node in range(self.nnodes_element_r):
                nodalDOF_element_r = self.nodalDOF_element_r[node]
                r_xi += self.N_r_xi[el, i, node] * qe[nodalDOF_element_r]
                r_xi_q[:, nodalDOF_element_r] += self.N_r_xi[el, i, node] * np.eye(
                    3, dtype=float
                )

            d1d2d3 = np.zeros(9, dtype=float)
            d1d2d3_xi = np.zeros(9, dtype=float)
            d1d2d3_q = np.zeros((9, self.nq_element), dtype=float)
            d1d2d3_xi_q = np.zeros((9, self.nq_element), dtype=float)
            for node in range(self.nnodes_element_di):
                nodalDOF = self.nodalDOF_element_di[node]
                qe_node = qe[nodalDOF]
                d1d2d3 += self.N_di[el, i, node] * qe_node
                d1d2d3_xi += self.N_di_xi[el, i, node] * qe_node
                d1d2d3_q[:, nodalDOF] += self.N_di[el, i, node] * np.eye(9, dtype=float)
                d1d2d3_xi_q[:, nodalDOF] += self.N_di_xi[el, i, node] * np.eye(
                    9, dtype=float
                )

            d1, d2, d3 = np.split(d1d2d3, 3)
            d1_xi, d2_xi, d3_xi = np.split(d1d2d3_xi, 3)
            d1_q, d2_q, d3_q = np.split(d1d2d3_q, 3)
            d1_xi_q, d2_xi_q, d3_xi_q = np.split(d1d2d3_xi_q, 3)

            # axial and shear strains
            K_Gamma = np.array([d1 @ r_xi, d2 @ r_xi, d3 @ r_xi]) / J

            # derivative of axial and shear strains
            K_Gamma_q = (
                np.array(
                    [
                        d1 @ r_xi_q,
                        d2 @ r_xi_q,
                        d3 @ r_xi_q,
                    ]
                )
                / J
                + np.array(
                    [
                        r_xi @ d1_q,
                        r_xi @ d2_q,
                        r_xi @ d3_q,
                    ]
                )
                / J
            )

            # torsional and flexural strains
            K_Kappa = (
                0.5
                * np.array(
                    [
                        d3 @ d2_xi - d2 @ d3_xi,
                        d1 @ d3_xi - d3 @ d1_xi,
                        d2 @ d1_xi - d1 @ d2_xi,
                    ]
                )
                / J
            )
            K_Kappa_q = (
                0.5
                * np.array(
                    [
                        d3 @ d2_xi_q + d2_xi @ d3_q - d2 @ d3_xi_q - d3_xi @ d2_q,
                        d1 @ d3_xi_q + d3_xi @ d1_q - d3 @ d1_xi_q - d1_xi @ d3_q,
                        d2 @ d1_xi_q + d1_xi @ d2_q - d1 @ d2_xi_q - d2_xi @ d1_q,
                    ]
                )
                / J
            )

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
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

            K_n_q = K_n_K_Gamma @ K_Gamma_q + K_n_K_Kappa @ K_Kappa_q
            K_m_q = K_m_K_Gamma @ K_Gamma_q + K_m_K_Kappa @ K_Kappa_q

            # quadrature point contribution to element residual
            for node in range(self.nnodes_element_r):
                # fe[self.nodalDOF_element_r[node]] -= (
                #     self.N_r_xi[el, i, node] * (
                #         K_n[0] * d1 + K_n[1] * d2 + K_n[2] * d3
                #     ) * qwi
                # )
                Ke[self.nodalDOF_element_r[node]] -= (
                    self.N_r_xi[el, i, node]
                    * (
                        K_n[0] * d1_q
                        + np.outer(d1, K_n_q[0])
                        + K_n[1] * d2_q
                        + np.outer(d2, K_n_q[1])
                        + K_n[2] * d3_q
                        + np.outer(d3, K_n_q[2])
                    )
                    * qwi
                )

            di = [d1, d2, d3]
            di_xi = [d1_xi, d2_xi, d3_xi]
            di_q = [d1_q, d2_q, d3_q]
            di_xi_q = [d1_xi_q, d2_xi_q, d3_xi_q]

            for node in range(self.nnodes_element_di):
                N_di = self.N_di[el, i, node]
                N_di_xi = self.N_di_xi[el, i, node]

                def fe_di_q(i1):
                    i1, i2, i3 = np.roll([0, 1, 2], (i1 + 1, i1 + 2))
                    return (
                        N_di * (K_n[i1] * r_xi_q + np.outer(r_xi, K_n_q[i1]))
                        + 0.5
                        * N_di
                        * (
                            K_m[i2] * di_xi_q[i3]
                            + np.outer(di_xi[i3], K_m_q[i2])
                            - K_m[i3] * di_xi_q[i2]
                            - np.outer(di_xi[i2], K_m_q[i3])
                        )
                        + 0.5
                        * N_di_xi
                        * (
                            K_m[i3] * di_q[i2]
                            + np.outer(di[i2], K_m_q[i3])
                            - K_m[i2] * di_q[i3]
                            - np.outer(di[i3], K_m_q[i2])
                        )
                    )

                # # fe[self.nodalDOF_element_di[node]] -= (
                # #     np.concatenate((fe_di(0), fe_di(1), fe_di(2)))
                # # ) * qwi
                # Ke[self.nodalDOF_element_di[node]] -= (
                #     np.concatenate((fe_di_q(0), fe_di_q(1), fe_di_q(2)))
                # ) * qwi

                Ke[self.nodalDOF_element_di[node]] -= (
                    np.concatenate(list(map(fe_di_q, [0, 1, 2]))) * qwi
                )

        return Ke

        # Ke_num = approx_fprime(
        #     qe, lambda qe: self.f_pot_el(qe, el), method="3-point", eps=1.0e-6
        # )
        # diff = Ke_num - Ke
        # error = np.linalg.norm(diff)
        # print(f"error: {error}")
        # return Ke_num

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        return u

    def B(self, t, q):
        return np.ones(self.nq)

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

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

    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        N_r, _ = self.basis_functions_r(frame_ID[0])
        r_OC = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_r):
            r_OC += N_r[node] * q[self.nodalDOF_element_r[node]]

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

    def A_IK(self, t, q, frame_ID):
        N_di, _ = self.basis_functions_di(frame_ID[0])
        d1d2d3 = np.zeros(9, dtype=q.dtype)
        for node in range(self.nnodes_element_di):
            d1d2d3 += N_di[node] * q[self.nodalDOF_element_di[node]]
        return d1d2d3.reshape(3, 3, order="F")

    def A_IK_q(self, t, q, frame_ID):
        N_di, _ = self.basis_functions_di(frame_ID[0])
        d1d2d3_q = np.zeros((9, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_di):
            d1d2d3_q[:, self.nodalDOF_element_di[node]] += N_di[node] * np.eye(
                9, dtype=float
            )

        A_IK_q = d1d2d3_q.reshape(3, 3, -1, order="F")
        return A_IK_q

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3)):
        # Note: This is correct but confusing!
        return self.r_OP(t, u, frame_ID, K_r_SP)

    def v_P_q(self, t, q, u, frame_ID, K_r_SP=np.zeros(3)):
        return np.zeros((3, self.nq_element))

    def J_P(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        return self.r_OP_q(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP)

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.zeros((3, self.nu_element, self.nq_element))

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=np.zeros(3)):
        # Note: This is correct but confusing!
        return self.r_OP(t, u_dot, frame_ID, K_r_SP)

    def a_P_q(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_el))

    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_el))

    def K_Omega(self, t, q, u, frame_ID):
        N_di, _ = self.basis_functions_di(frame_ID[0])

        d1d2d3 = np.zeros(9, dtype=q.dtype)
        d1d2d3_dot = np.zeros(9, dtype=q.dtype)
        for node in range(self.nnodes_element_di):
            d1d2d3 += N_di[node] * q[self.nodalDOF_element_di[node]]
            d1d2d3_dot += N_di[node] * u[self.nodalDOF_element_di[node]]
        d1, d2, d3 = np.split(d1d2d3, 3)
        d1_dot, d2_dot, d3_dot = np.split(d1d2d3_dot, 3)

        return 0.5 * np.array(
            [
                d3 @ d2_dot - d2 @ d3_dot,
                d1 @ d3_dot - d3 @ d1_dot,
                d2 @ d1_dot - d1 @ d2_dot,
            ]
        )

    def K_J_R(self, t, q, frame_ID):
        N_di, _ = self.basis_functions_di(frame_ID[0])

        d1d2d3 = np.zeros(9, dtype=q.dtype)
        d1d2d3_dot_u = np.zeros((9, self.nq_element), dtype=q.dtype)
        for node in range(self.nnodes_element_di):
            nodalDOF = self.nodalDOF_element_di[node]
            d1d2d3 += N_di[node] * q[nodalDOF]
            d1d2d3_dot_u[:, nodalDOF] += N_di[node] * np.eye(9)
        d1, d2, d3 = np.split(d1d2d3, 3)
        d1_dot_u, d2_dot_u, d3_dot_u = np.split(d1d2d3_dot_u, 3)

        K_J_R = 0.5 * np.array(
            [
                d3 @ d2_dot_u - d2 @ d3_dot_u,
                d1 @ d3_dot_u - d3 @ d1_dot_u,
                d2 @ d1_dot_u - d1 @ d2_dot_u,
            ]
        )
        return K_J_R

        # K_J_R_num = approx_fprime(
        #     np.zeros(self.nu_element),
        #     lambda u: self.K_Omega(t, q, u, frame_ID=frame_ID),
        #     method="3-point",
        # )
        # diff = K_J_R_num - K_J_R
        # diff_error = diff
        # error = np.linalg.norm(diff_error)
        # if error > 1.0e-10:
        #     print(f'error K_J_R: {error}')
        # return K_J_R_num

    # TODO: Implement derivative of K_J_R_q
    def K_J_R_q(self, t, q, frame_ID):
        # N, _ = self.basis_functions_di(frame_ID[0])
        # NN = self.stack3di(N)

        # A_IK_q = np.zeros((3, 3, self.nq_el))
        # A_IK_q[:, 0, self.d1DOF] = NN
        # A_IK_q[:, 1, self.d2DOF] = NN
        # A_IK_q[:, 2, self.d3DOF] = NN

        # K_Omega_tilde_Omega_tilde = skew2ax_A()
        # tmp = np.einsum("jil,jk->ikl", A_IK_q, NN)

        # K_J_R_q = np.zeros((3, self.nq_el, self.nq_el))
        # K_J_R_q[:, self.d1DOF] = np.einsum(
        #     "ij,jkl->ikl", K_Omega_tilde_Omega_tilde[0], tmp
        # )
        # K_J_R_q[:, self.d2DOF] = np.einsum(
        #     "ij,jkl->ikl", K_Omega_tilde_Omega_tilde[1], tmp
        # )
        # K_J_R_q[:, self.d3DOF] = np.einsum(
        #     "ij,jkl->ikl", K_Omega_tilde_Omega_tilde[2], tmp
        # )
        # return K_J_R_q

        K_J_R_q_num = approx_fprime(
            q, lambda q: self.K_J_R(t, q, frame_ID), method="2-point"
        )
        # error = np.max(np.abs(K_J_R_q_num - K_J_R_q))
        # print(f'error K_J_R_q: {error}')
        return K_J_R_q_num

    def K_Psi(self, t, q, u, u_dot, frame_ID):
        N_di, _ = self.basis_functions_di(frame_ID[0])

        d1d2d3 = np.zeros(9, dtype=q.dtype)
        d1d2d3_dot = np.zeros(9, dtype=q.dtype)
        d1d2d3_ddot = np.zeros(9, dtype=q.dtype)
        for node in range(self.nnodes_element_di):
            d1d2d3 += N_di[node] * q[self.nodalDOF_element_di[node]]
            d1d2d3_dot += N_di[node] * u[self.nodalDOF_element_di[node]]
            d1d2d3_dot += N_di[node] * u_dot[self.nodalDOF_element_di[node]]
        d1, d2, d3 = np.split(d1d2d3, 3)
        d1_dot, d2_dot, d3_dot = np.split(d1d2d3_dot, 3)
        d1_ddot, d2_ddot, d3_ddot = np.split(d1d2d3_ddot, 3)

        return 0.5 * np.array(
            [
                d3 @ d2_ddot - d2 @ d3_ddot + d3_dot @ d2_dot - d2_dot @ d3_dot,
                d1 @ d3_ddot - d3 @ d1_ddot + d1_dot @ d3_dot - d3_dot @ d1_dot,
                d2 @ d1_ddot - d1 @ d2_ddot + d2_dot @ d1_dot - d1_dot @ d2_dot,
            ]
        )

    ####################################################
    # body force
    ####################################################
    def distributed_force1D_el(self, force, t, el):
        # fe = np.zeros(self.nq_element)
        # for i in range(self.nquadrature):
        #     NNi = self.stack3r(self.N_r[el, i])
        #     fe[self.rDOF] += (
        #         NNi.T @ force(t, self.qp[el, i]) * self.J[el, i] * self.qw[el, i]
        #     )
        # return fe
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
        f = np.zeros(self.nq)
        for el in range(self.nelement):
            f[self.elDOF[el]] += self.distributed_force1D_el(force, t, el)
        return f

    def distributed_force1D_q(self, t, q, force):
        return None

    ##############################################################
    # abstract methods for bilateral constraints on position level
    ##############################################################
    @abstractmethod
    def g(self, t, q):
        pass

    @abstractmethod
    def g_dot(self, t, q):
        pass

    @abstractmethod
    def g_ddot(self, t, q):
        pass

    @abstractmethod
    def g_q(self, t, q):
        pass

    @abstractmethod
    def W_g(self, t, q):
        pass

    @abstractmethod
    def Wla_g_q(self, t, q, la_g):
        pass

    ####################################################
    # visualization
    ####################################################
    def nodes(self, q):
        q_body = q[self.qDOF]
        return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF_r]).T

    def centerline(self, q, num=100):
        q_body = q[self.qDOF]
        r = []
        for xi in np.linspace(0, 1, num):
            frame_ID = (xi,)
            qp = q_body[self.local_qDOF_P(frame_ID)]
            r.append(self.r_OP(1, qp, frame_ID))
        return np.array(r).T

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
            self.polynomial_degree_r == self.polynomial_degree_di
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
        for i in range(self.nn_r):
            qr = q[self.nodalDOF_r[i]]
            q_di = q[self.nodalDOF_di[i]]
            A_IK = q_di.reshape(3, 3, order="F")  # TODO: Check this!

            for k, point in enumerate(circle_points):
                # Note: eta index is always 0!
                Pw[i, 0, k] = qr + A_IK @ point

        if self.basis == "B-spline":
            knot_vector_eta = BSplineKnotVector(polynomial_degree_eta, nEl_eta)
            knot_vector_zeta = BSplineKnotVector(polynomial_degree_zeta, nEl_zeta)
        elif self.basis == "Lagrange":
            knot_vector_eta = LagrangeKnotVector(polynomial_degree_eta, nEl_eta)
            knot_vector_zeta = LagrangeKnotVector(polynomial_degree_zeta, nEl_zeta)
        knot_vector_objs = [self.knot_vector_r, knot_vector_eta, knot_vector_zeta]
        degrees = (
            self.polynomial_degree_r,
            polynomial_degree_eta,
            polynomial_degree_zeta,
        )

        # Build Bezier patches from B-spline control points
        from cardillo.discretization.b_spline import decompose_B_spline_volume

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
            self.polynomial_degree_r == self.polynomial_degree_di
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
        for i in range(self.nn_r):
            qr = q[self.nodalDOF_r[i]]
            q_di = q[self.nodalDOF_di[i]]
            A_IK = q_di.reshape(3, 3, order="F")  # TODO: Check this!

            for j, aj in enumerate(as_):
                for k, bk in enumerate(bs_):
                    Pw[i, j, k] = qr + A_IK @ np.array([0, aj, bk])

        if self.basis == "B-spline":
            knot_vector_eta = BSplineKnotVector(polynomial_degree_eta, nEl_eta)
            knot_vector_zeta = BSplineKnotVector(polynomial_degree_zeta, nEl_zeta)
        elif self.basis == "Lagrange":
            knot_vector_eta = LagrangeKnotVector(polynomial_degree_eta, nEl_eta)
            knot_vector_zeta = LagrangeKnotVector(polynomial_degree_zeta, nEl_zeta)
        knot_vector_objs = [self.knot_vector_r, knot_vector_eta, knot_vector_zeta]
        degrees = (
            self.polynomial_degree_r,
            polynomial_degree_eta,
            polynomial_degree_zeta,
        )

        # Build Bezier patches from B-spline control points
        from cardillo.discretization.b_spline import decompose_B_spline_volume

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
        if self.polynomial_degree_r == self.polynomial_degree_di:
            same_shape_functions = True

        if same_shape_functions:
            _, points_di, _ = self.mesh_di.vtk_mesh(q[self.nq_r :])

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

        Gamma0_vtk = self.mesh_r.field_to_vtk(self.Gamma0)
        point_data.update({"Gamma0": Gamma0_vtk})

        Kappa0_vtk = self.mesh_r.field_to_vtk(self.Kappa0)
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
            qe_d1 = qe[self.d1DOF]
            qe_d2 = qe[self.d2DOF]
            qe_d3 = qe[self.d3DOF]

            for i in range(self.nquadrature):
                # build matrix of shape function derivatives
                NN_di_i = self.stack3di(self.N_di[el, i])
                NN_r_xii = self.stack3r(self.N_r_xi[el, i])
                NN_di_xii = self.stack3di(self.N_di_xi[el, i])

                # extract reference state variables
                Ji = self.J[el, i]
                Gamma0_i = self.Gamma0[el, i]
                Kappa0_i = self.Kappa0[el, i]

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
                r_s = r_xi / Ji

                d1_s = d1_xi / Ji
                d2_s = d2_xi / Ji
                d3_s = d3_xi / Ji

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
            for i, (Gamma_i, Gamma0_i, Kappa_i, Kappa0_i) in enumerate(
                zip(Gamma_vtk, Gamma0_vtk, Kappa_vtk, Kappa0_vtk)
            ):
                field[i] = fun(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i).reshape(-1)
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
        if self.polynomial_degree_r == self.polynomial_degree_di:
            same_shape_functions = True

        if same_shape_functions:
            _, points_di, _ = self.mesh_di.vtk_mesh(q[self.nq_r :])

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

        Gamma0_vtk = self.mesh_r.field_to_vtk(self.Gamma0)
        point_data.update({"Gamma0": Gamma0_vtk})

        Kappa0_vtk = self.mesh_r.field_to_vtk(self.Kappa0)
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
            qe_d1 = qe[self.d1DOF]
            qe_d2 = qe[self.d2DOF]
            qe_d3 = qe[self.d3DOF]

            for i in range(self.nquadrature):
                # build matrix of shape function derivatives
                NN_di_i = self.stack3di(self.N_di[el, i])
                NN_r_xii = self.stack3r(self.N_r_xi[el, i])
                NN_di_xii = self.stack3di(self.N_di_xi[el, i])

                # extract reference state variables
                Ji = self.J[el, i]
                Gamma0_i = self.Gamma0[el, i]
                Kappa0_i = self.Kappa0[el, i]

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
                r_s = r_xi / Ji

                d1_s = d1_xi / Ji
                d2_s = d2_xi / Ji
                d3_s = d3_xi / Ji

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
            for i, (Gamma_i, Gamma0_i, Kappa_i, Kappa0_i) in enumerate(
                zip(Gamma_vtk, Gamma0_vtk, Kappa_vtk, Kappa0_vtk)
            ):
                field[i] = fun(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i).reshape(-1)
            point_data.update({name: field})

        return points_r, point_data, cells_r, HigherOrderDegrees_r


####################################################
# constraint beam using nodal constraints
####################################################
class I_R12_BubonvGalerkin_R12_Dirac(I_R12_BubonvGalerkin_R12):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nnode_la_g = 6
        self.nla_g = (
            self.nnode_la_g * self.nnode_di
        )  # we enforce constraints on every director node
        la_g0 = kwargs.get("la_g0")
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0
        self.nodalDOF_la_g = np.arange(self.nla_g).reshape(
            self.nnode_di, self.nnode_la_g, order="F"
        )
        self.projection_pairs = [(0, 1), (1, 2), (2, 0)]

    # constraints on a single node
    def __g(self, qn):
        A_IK = qn.reshape(3, 3, order="F")

        g = np.zeros(6, dtype=qn.dtype)
        for i in range(3):
            di = A_IK[:, i]
            g[i] = di @ di - 1
        for i, (a, b) in enumerate(self.projection_pairs):
            g[3 + i] = A_IK[:, a] @ A_IK[:, b]

        # d1, d2, d3 = np.split(qn, 3)
        # g = np.zeros(6, dtype=qn.dtype)
        # g[0] = d1 @ d1 - 1.0
        # g[1] = d2 @ d2 - 1.0
        # g[2] = d3 @ d3 - 1.0
        # g[3] = d1 @ d2
        # g[4] = d2 @ d3
        # g[5] = d3 @ d1

        return g

    def __g_dot(self, qn, un):
        # d1, d2, d3 = np.split(qn, 3)
        # d1_dot, d2_dot, d3_dot = np.split(un, 3)

        # g_dot = np.zeros(6, dtype=qn.dtype)
        # g_dot[0] = 2 * d1 @ d1_dot
        # g_dot[1] = 2 * d2 @ d2_dot
        # g_dot[2] = 2 * d3 @ d3_dot
        # g_dot[3] = d1 @ d2_dot + d2 @ d1_dot
        # g_dot[4] = d2 @ d3_dot + d3 @ d2_dot
        # g_dot[5] = d3 @ d1_dot + d1 @ d3_dot

        A_IK = qn.reshape(3, 3, order="F")
        A_IK_dot = un.reshape(3, 3, order="F")

        g_dot = np.zeros(6, dtype=qn.dtype)
        for i in range(3):
            g_dot[i] = 2 * A_IK[:, i] @ A_IK_dot[:, i]
        for i, (a, b) in enumerate(self.projection_pairs):
            g_dot[3 + i] = A_IK[:, a] @ A_IK_dot[:, b] + A_IK[:, b] @ A_IK_dot[:, a]

        return g_dot

    def __g_ddot(self, qn, un, un_dot):
        A_IK = qn.reshape(3, 3, order="F")
        A_IK_dot = un.reshape(3, 3, order="F")
        A_IK_ddot = un_dot.reshape(3, 3, order="F")

        g_ddot = np.zeros(6, dtype=qn.dtype)
        for i in range(3):
            g_ddot[i] = (
                2 * A_IK[:, i] @ A_IK_ddot[:, i] + 2 * A_IK_dot[:, i] @ A_IK_dot[:, i]
            )
        for i, (a, b) in enumerate(self.projection_pairs):
            g_ddot[3 + i] = (
                A_IK[:, a] @ A_IK_ddot[:, b]
                + A_IK_dot[:, a] @ A_IK_dot[:, b]
                + A_IK[:, b] @ A_IK_ddot[:, a]
                + A_IK_dot[:, b] @ A_IK_dot[:, a]
            )

        return g_ddot

    def __g_q(self, qn):
        # d1, d2, d3 = np.split(qn, 3)

        # g_q = np.zeros((6, 9), dtype=qn.dtype)
        # g_q[0, :3] = 2.0 * d1
        # g_q[1, 3:6] = 2.0 * d2
        # g_q[2, 6:] = 2.0 * d3
        # g_q[3, :3] = d2
        # g_q[3, 3:6] = d1
        # g_q[4, 3:6] = d3
        # g_q[4, 6:] = d2
        # g_q[5, :3] = d3
        # g_q[5, 6:] = d1
        # return g_q

        A_IK = qn.reshape(3, 3, order="F")

        g_q = np.zeros((6, 9), dtype=qn.dtype)
        for i in range(3):
            di = A_IK[:, i]
            g_q[i, 3 * i : 3 * (i + 1)] = 2 * di
        for i, (a, b) in enumerate(self.projection_pairs):
            g_q[3 + i, 3 * b : 3 * (b + 1)] = A_IK[:, a]
            g_q[3 + i, 3 * a : 3 * (a + 1)] = A_IK[:, b]

        return g_q

        # g_q_num = approx_fprime(qn, self.__g)
        # diff = g_q - g_q_num
        # error = np.linalg.norm(diff)
        # print(f"error g_q: {error}")
        # return g_q_num

    def __Wla_g_q(self, la_g):
        eye3 = np.eye(3, dtype=float)

        Wla_g_q = np.zeros((9, 9), dtype=float)

        # g_q[0, :3] = 2.0 * d1
        # g_q[3, :3] = d2
        # g_q[5, :3] = d3
        Wla_g_q[:3, :3] = 2.0 * la_g[0] * eye3
        Wla_g_q[:3, 3:6] = la_g[3] * eye3
        Wla_g_q[:3, 6:9] = la_g[5] * eye3

        # g_q[1, 3:6] = 2.0 * d2
        # g_q[3, 3:6] = d1
        # g_q[4, 3:6] = d3
        Wla_g_q[3:6, 3:6] = 2.0 * la_g[1] * eye3
        Wla_g_q[3:6, :3] = la_g[3] * eye3
        Wla_g_q[3:6, 6:9] = la_g[4] * eye3

        # g_q[2, 6:] = 2.0 * d3
        # g_q[4, 6:] = d2
        # g_q[5, 6:] = d1
        Wla_g_q[6:9, 6:9] = 2.0 * la_g[2] * eye3
        Wla_g_q[6:9, 3:6] = la_g[4] * eye3
        Wla_g_q[6:9, :3] = la_g[5] * eye3

        return Wla_g_q

        # Wla_g_q_num = approx_fprime(
        #     np.zeros(9),
        #     lambda q: la_g @ self.__g_q(q),
        # )
        # diff = Wla_g_q_num - Wla_g_q
        # error = np.linalg.norm(diff)
        # print(f"error Wla_g_q: {error}")
        # return Wla_g_q_num

    # global constraint functions
    def g(self, t, q):
        g = np.zeros(self.nla_g)
        for node in range(self.nnode_di):
            nodalDOF_la_g = self.nodalDOF_la_g[node]
            nodalDOF_di = self.nodalDOF_di[node]
            g[nodalDOF_la_g] = self.__g(q[nodalDOF_di])

        return g

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g)
        for node in range(self.nnode_di):
            nodalDOF_la_g = self.nodalDOF_la_g[node]
            nodalDOF_di = self.nodalDOF_di[node]
            g_dot[nodalDOF_la_g] = self.__g_dot(q[nodalDOF_di], u[nodalDOF_di])

        return g_dot

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g)
        for node in range(self.nnode_di):
            nodalDOF_la_g = self.nodalDOF_la_g[node]
            nodalDOF_di = self.nodalDOF_di[node]
            g_ddot[nodalDOF_la_g] = self.__g_ddot(
                q[nodalDOF_di], u[nodalDOF_di], u[nodalDOF_di]
            )

        return g_ddot

    def g_q(self, t, q):
        coo = CooMatrix((self.nla_g, self.nq))
        for node in range(self.nnode_di):
            nodalDOF_la_g = self.nodalDOF_la_g[node]
            nodalDOF_di = self.nodalDOF_di[node]
            coo[nodalDOF_la_g, nodalDOF_di] = self.__g_q(q[nodalDOF_di])

        return coo

    def W_g(self, t, q):
        coo = CooMatrix((self.nu, self.nla_g))
        for node in range(self.nnode_di):
            nodalDOF_la_g = self.nodalDOF_la_g[node]
            nodalDOF_di = self.nodalDOF_di[node]
            coo[nodalDOF_di, nodalDOF_la_g] = self.__g_q(q[nodalDOF_di]).T

        return coo

    def Wla_g_q(self, t, q, la_g):
        coo = CooMatrix((self.nu, self.nq))
        for node in range(self.nnode_di):
            nodalDOF_la_g = self.nodalDOF_la_g[node]
            nodalDOF_di = self.nodalDOF_di[node]
            coo[nodalDOF_di, nodalDOF_di] = self.__Wla_g_q(la_g[nodalDOF_la_g])

        return coo


####################################################
# constraint beam using integral constraints
####################################################
class I_R12_BubonvGalerkin_R12_Integral(I_R12_BubonvGalerkin_R12):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Refactor me!")
        super().__init__(*args, **kwargs)

        self.polynomial_degree_g = self.polynomial_degree_di
        self.nn_el_g = self.polynomial_degree_g + 1  # number of nodes per element

        if self.basis == "B-spline":
            self.knot_vector_g = BSplineKnotVector(
                self.polynomial_degree_g, self.nelement
            )
            self.nn_g = self.nelement + self.polynomial_degree_g  # number of nodes
        elif self.basis == "Lagrange":
            self.knot_vector_g = LagrangeKnotVector(
                self.polynomial_degree_g, self.nelement
            )
            self.nn_g = self.nelement * self.polynomial_degree_g + 1  # number of nodes
        else:
            raise NotImplementedError

        self.element_span_g = self.knot_vector_g.element_data
        self.mesh_g = Mesh1D(
            self.knot_vector_g,
            self.nquadrature,
            6,
            derivative_order=0,
            basis=self.basis,
        )

        # compute shape functions
        self.N_g = self.mesh_g.N

        # total number of nodes
        self.nnode_g = self.mesh_g.nnodes

        # number of nodes per element
        self.nnodes_element_g = self.mesh_g.nnodes_per_element

        # total number of generalized coordinates
        self.nla_g = self.mesh_g.nq

        # number of Lagrange multipliers per element
        self.nla_g_element = self.mesh_g.nq_per_element

        # global element connectivity
        self.elDOF_g = self.mesh_g.elDOF

        # global nodal
        self.nodalDOF_g = self.mesh_g.nodalDOF

        # nodal connectivity on element level
        self.nodalDOF_element_g = self.mesh_g.nodalDOF_element

        la_g0 = kwargs.get("la_g0")
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

    def __g_el(self, qe, el):
        g = np.zeros(self.nla_g_element, dtype=qe.dtype)

        for i in range(self.nquadrature):
            d1d2d3 = np.zeros(9, dtype=qe.dtype)
            for node in range(self.nnodes_element_di):
                d1d2d3 += self.N_di[el, i, node] * qe[self.nodalDOF_element_di[node]]
            d1, d2, d3 = np.split(d1d2d3, 3)

            for node in range(self.nnodes_element_g):
                factor = self.N_g[el, i, node] * self.J[el, i] * self.qw[el, i]

                g[self.nodalDOF_element_g[node, 0]] += (d1 @ d1 - 1.0) * factor
                g[self.nodalDOF_element_g[node, 1]] += (d1 @ d2) * factor
                g[self.nodalDOF_element_g[node, 2]] += (d1 @ d3) * factor
                g[self.nodalDOF_element_g[node, 3]] += (d2 @ d2 - 1.0) * factor
                g[self.nodalDOF_element_g[node, 4]] += (d2 @ d3) * factor
                g[self.nodalDOF_element_g[node, 5]] += (d3 @ d3 - 1.0) * factor

        return g

    def __g_q_el(self, qe, el):
        g_q = np.zeros((self.nla_g_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            d1d2d3 = np.zeros(9, dtype=qe.dtype)
            d1d2d3_q = np.zeros((9, self.nq_element), dtype=qe.dtype)
            for node in range(self.nnodes_element_di):
                d1d2d3 += self.N_di[el, i, node] * qe[self.nodalDOF_element_di[node]]
                d1d2d3_q[:, self.nodalDOF_element_di[node]] += self.N_di[
                    el, i, node
                ] * np.eye(9)

            d1, d2, d3 = np.split(d1d2d3, 3)
            d1_q, d2_q, d3_q = np.split(d1d2d3_q, 3)

            for node in range(self.nnodes_element_g):
                factor = self.N_g[el, i, node] * self.J[el, i] * self.qw[el, i]

                g_q[self.nodalDOF_element_g[node, 0]] += factor * 2 * d1 @ d1_q
                g_q[self.nodalDOF_element_g[node, 1]] += factor * (
                    d1 @ d2_q + d2 @ d1_q
                )
                g_q[self.nodalDOF_element_g[node, 2]] += factor * (
                    d1 @ d3_q + d3 @ d1_q
                )
                g_q[self.nodalDOF_element_g[node, 3]] += factor * 2 * d2 @ d2_q
                g_q[self.nodalDOF_element_g[node, 4]] += factor * (
                    d2 @ d3_q + d3 @ d2_q
                )
                g_q[self.nodalDOF_element_g[node, 5]] += factor * 2 * d3 @ d3_q

        return g_q

    def __Wla_g_q_el(self, qe, la_g, el):
        Wla_g_q = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            d1d2d3_q = np.zeros((9, self.nq_element), dtype=qe.dtype)
            for node in range(self.nnodes_element_di):
                d1d2d3_q[:, self.nodalDOF_element_di[node]] += self.N_di[
                    el, i, node
                ] * np.eye(9)

            d1_q = d1d2d3_q[:3]
            d2_q = d1d2d3_q[3:6]
            d3_q = d1d2d3_q[6:9]

            for node in range(self.nnodes_element_g):
                factor = self.N_g[el, i, node] * self.J[el, i] * self.qw[el, i]

                # g_q[self.nodalDOF_element_g[node, 0]] += factor * 2 * d1 @ d1_q
                Wla_g_q += (
                    factor * la_g[self.nodalDOF_element_g[node, 0]] * 2 * d1_q.T @ d1_q
                )

                # g_q[self.nodalDOF_element_g[node, 1]] += factor * (d1 @ d2_q + d2 @ d1_q)
                Wla_g_q += (
                    factor
                    * la_g[self.nodalDOF_element_g[node, 1]]
                    * (d2_q.T @ d1_q + d1_q.T @ d2_q)
                )

                # g_q[self.nodalDOF_element_g[node, 2]] += factor * (d1 @ d3_q + d3 @ d1_q)
                Wla_g_q += (
                    factor
                    * la_g[self.nodalDOF_element_g[node, 2]]
                    * (d3_q.T @ d1_q + d1_q.T @ d3_q)
                )

                # g_q[self.nodalDOF_element_g[node, 3]] += factor * 2 * d2 @ d2_q
                Wla_g_q += (
                    factor * la_g[self.nodalDOF_element_g[node, 3]] * 2 * d2_q.T @ d2_q
                )

                # g_q[self.nodalDOF_element_g[node, 4]] += factor * (d2 @ d3_q + d3 @ d2_q)
                Wla_g_q += (
                    factor
                    * la_g[self.nodalDOF_element_g[node, 4]]
                    * (d3_q.T @ d2_q + d2_q.T @ d3_q)
                )

                # g_q[self.nodalDOF_element_g[node, 5]] += factor * 2 * d3 @ d3_q
                Wla_g_q += (
                    factor * la_g[self.nodalDOF_element_g[node, 5]] * 2 * d3_q.T @ d3_q
                )

        return Wla_g_q

    def __g_dot_el(self, qe, ue, el):
        raise NotImplementedError
        g_dot = np.zeros(self.nla_g_element)

        N, N_g, J0, qw = self.N_di[el], self.N_g[el], self.J[el], self.qw[el]
        for Ni, N_gi, Ji, qwi in zip(N, N_g, J0, qw):
            NNi = self.stack3r(Ni)

            d1 = NNi @ qe[self.d1DOF]
            d2 = NNi @ qe[self.d2DOF]
            d3 = NNi @ qe[self.d3DOF]
            d1_dot = NNi @ ue[self.d1DOF]
            d2_dot = NNi @ ue[self.d2DOF]
            d3_dot = NNi @ ue[self.d3DOF]

            factor = N_gi * Ji * qwi

            g_dot[self.g11DOF] += 2 * d1 @ d1_dot * factor
            g_dot[self.g12DOF] += (d1 @ d2_dot + d2 @ d1_dot) * factor
            g_dot[self.g13DOF] += (d1 @ d3_dot + d3 @ d1_dot) * factor

            g_dot[self.g22DOF] += 2 * d2 @ d2_dot * factor
            g_dot[self.g23DOF] += (d2 @ d3_dot + d3 @ d2_dot) * factor

            g_dot[self.g33DOF] += 2 * d3 @ d3_dot * factor

        return g_dot

    def __g_dot_q_el(self, qe, ue, el):
        raise NotImplementedError
        g_dot_q = np.zeros((self.nla_g_element, self.nq_el))

        N, N_g, J0, qw = self.N[el], self.N_g[el], self.J[el], self.qw[el]
        for Ni, N_gi, Ji, qwi in zip(N, N_g, J0, qw):
            NNi = self.stack3r(Ni)

            d1_dot = NNi @ ue[self.d1DOF]
            d2_dot = NNi @ ue[self.d2DOF]
            d3_dot = NNi @ ue[self.d3DOF]

            factor = N_gi * Ji * qwi
            d1_dot_N = np.outer(factor, d1_dot @ NNi)
            d2_dot_N = np.outer(factor, d2_dot @ NNi)
            d3_dot_N = np.outer(factor, d3_dot @ NNi)

            # g_dot[self.g11DOF] += 2 * d1 @ d1_dot * factor
            g_dot_q[self.g11DOF[:, None], self.d1DOF] += 2 * d1_dot_N
            # g_dot[self.g12DOF] += (d1 @ d2_dot + d2 @ d1_dot) * factor
            g_dot_q[self.g12DOF[:, None], self.d1DOF] += d2_dot_N
            g_dot_q[self.g12DOF[:, None], self.d2DOF] += d1_dot_N
            # g_dot[self.g13DOF] += (d1 @ d3_dot + d3 @ d1_dot) * factor
            g_dot_q[self.g13DOF[:, None], self.d1DOF] += d3_dot_N
            g_dot_q[self.g13DOF[:, None], self.d3DOF] += d1_dot_N

            # g_dot[self.g22DOF] += 2 * d2 @ d2_dot * factor
            g_dot_q[self.g22DOF[:, None], self.d2DOF] += 2 * d2_dot_N
            # g_dot[self.g23DOF] += (d2 @ d3_dot + d3 @ d2_dot) * factor
            g_dot_q[self.g23DOF[:, None], self.d2DOF] += d3_dot_N
            g_dot_q[self.g23DOF[:, None], self.d3DOF] += d2_dot_N

            # g_dot[self.g33DOF] += 2 * d3 @ d3_dot * factor
            g_dot_q[self.g33DOF[:, None], self.d3DOF] += 2 * d3_dot_N

        return g_dot_q

    def __g_ddot_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError
        g_ddot = np.zeros(self.nla_g_element)

        for i in range(self.nquadrature):
            NNi = self.stack3di(self.N_di[el, i])

            d1 = NNi @ qe[self.d1DOF]
            d2 = NNi @ qe[self.d2DOF]
            d3 = NNi @ qe[self.d3DOF]
            d1_dot = NNi @ ue[self.d1DOF]
            d2_dot = NNi @ ue[self.d2DOF]
            d3_dot = NNi @ ue[self.d3DOF]
            d1_ddot = NNi @ ue_dot[self.d1DOF]
            d2_ddot = NNi @ ue_dot[self.d2DOF]
            d3_ddot = NNi @ ue_dot[self.d3DOF]

            factor = self.N_g[el, i] * self.J[el, i] * self.qw[el, i]

            g_ddot[self.g11DOF] += 2 * (d1 @ d1_ddot + d1_dot @ d1_dot) * factor
            g_ddot[self.g12DOF] += (
                d1 @ d2_ddot + 2 * d1_dot @ d2_dot + d2 @ d1_ddot
            ) * factor
            g_ddot[self.g13DOF] += (
                d1 @ d3_ddot + 2 * d1_dot @ d3_dot + d3 @ d1_ddot
            ) * factor

            g_ddot[self.g22DOF] += 2 * (d2 @ d2_ddot + d2_dot @ d2_dot) * factor
            g_ddot[self.g23DOF] += (
                d2 @ d3_ddot + 2 * d2_dot @ d3_dot + d3 @ d2_ddot
            ) * factor

            g_ddot[self.g33DOF] += 2 * (d3 @ d3_ddot + d3_dot @ d3_dot) * factor

        return g_ddot

    def __g_ddot_q_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError
        g_ddot_q = np.zeros((self.nla_g_element, self.nq_el))

        N, N_g, J0, qw = self.N[el], self.N_g[el], self.J[el], self.qw[el]
        for Ni, N_gi, Ji, qwi in zip(N, N_g, J0, qw):
            NNi = self.stack3r(Ni)

            d1_ddot = NNi @ ue_dot[self.d1DOF]
            d2_ddot = NNi @ ue_dot[self.d2DOF]
            d3_ddot = NNi @ ue_dot[self.d3DOF]

            factor = N_gi * Ji * qwi
            d1_ddot_N = np.outer(factor, d1_ddot @ NNi)
            d2_ddot_N = np.outer(factor, d2_ddot @ NNi)
            d3_ddot_N = np.outer(factor, d3_ddot @ NNi)

            # g_ddot[self.g11DOF] += 2 * (d1 @ d1_ddot + d1_dot @ d1_dot) * factor
            g_ddot_q[self.g11DOF[:, None], self.d1DOF] += 2 * d1_ddot_N

            # g_ddot[self.g12DOF] += (d1 @ d2_ddot + 2 * d1_dot @ d2_dot + d2 @ d1_ddot) * factor
            g_ddot_q[self.g12DOF[:, None], self.d1DOF] += d2_ddot_N
            g_ddot_q[self.g12DOF[:, None], self.d2DOF] += d1_ddot_N
            # g_ddot[self.g13DOF] += (d1 @ d3_ddot + 2 * d1_dot @ d3_dot + d3 @ d1_ddot) * factor
            g_ddot_q[self.g13DOF[:, None], self.d1DOF] += d3_ddot_N
            g_ddot_q[self.g13DOF[:, None], self.d3DOF] += d1_ddot_N

            # g_ddot[self.g22DOF] += 2 * (d2 @ d2_ddot + d2_dot @ d2_dot) * factor
            g_ddot_q[self.g22DOF[:, None], self.d2DOF] += 2 * d2_ddot_N
            # g_ddot[self.g23DOF] += (d2 @ d3_ddot + 2 * d2_dot @ d3_dot + d3 @ d2_ddot) * factor
            g_ddot_q[self.g23DOF[:, None], self.d2DOF] += d3_ddot_N
            g_ddot_q[self.g23DOF[:, None], self.d3DOF] += d2_ddot_N

            # g_ddot[self.g33DOF] += 2 * (d3 @ d3_ddot + d3_dot @ d3_dot) * factor
            g_ddot_q[self.g33DOF[:, None], self.d3DOF] += 2 * d3_ddot_N

        return g_ddot_q

    def __g_ddot_u_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError
        g_ddot_u = np.zeros((self.nla_g_element, self.nq_el))

        N, N_g, J0, qw = self.N[el], self.N_g[el], self.J[el], self.qw[el]
        for Ni, N_gi, Ji, qwi in zip(N, N_g, J0, qw):
            NNi = self.stack3r(Ni)

            d1_dot = NNi @ ue[self.d1DOF]
            d2_dot = NNi @ ue[self.d2DOF]
            d3_dot = NNi @ ue[self.d3DOF]

            factor = N_gi * Ji * qwi
            d1_dot_N = np.outer(factor, d1_dot @ NNi)
            d2_dot_N = np.outer(factor, d2_dot @ NNi)
            d3_dot_N = np.outer(factor, d3_dot @ NNi)

            # g_ddot[self.g11DOF] += 2 * (d1 @ d1_ddot + d1_dot @ d1_dot) * factor
            g_ddot_u[self.g11DOF[:, None], self.d1DOF] += 4 * d1_dot_N

            # g_ddot[self.g12DOF] += (d1 @ d2_ddot + 2 * d1_dot @ d2_dot + d2 @ d1_ddot) * factor
            g_ddot_u[self.g12DOF[:, None], self.d1DOF] += 2 * d2_dot_N
            g_ddot_u[self.g12DOF[:, None], self.d2DOF] += 2 * d1_dot_N
            # g_ddot[self.g13DOF] += (d1 @ d3_ddot + 2 * d1_dot @ d3_dot + d3 @ d1_ddot) * factor
            g_ddot_u[self.g13DOF[:, None], self.d1DOF] += 2 * d3_dot_N
            g_ddot_u[self.g13DOF[:, None], self.d3DOF] += 2 * d1_dot_N

            # g_ddot[self.g22DOF] += 2 * (d2 @ d2_ddot + d2_dot @ d2_dot) * factor
            g_ddot_u[self.g22DOF[:, None], self.d2DOF] += 4 * d2_dot_N
            # g_ddot[self.g23DOF] += (d2 @ d3_ddot + 2 * d2_dot @ d3_dot + d3 @ d2_ddot) * factor
            g_ddot_u[self.g23DOF[:, None], self.d2DOF] += 2 * d3_dot_N
            g_ddot_u[self.g23DOF[:, None], self.d3DOF] += 2 * d2_dot_N

            # g_ddot[self.g33DOF] += 2 * (d3 @ d3_ddot + d3_dot @ d3_dot) * factor
            g_ddot_u[self.g33DOF[:, None], self.d3DOF] += 4 * d3_dot_N

        return g_ddot_u

    # global constraint functions
    def g(self, t, q):
        g = np.zeros(self.nla_g)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g[elDOF_g] += self.__g_el(q[elDOF], el)
        return g

    def g_q(self, t, q):
        coo = CooMatrix((self.nla_g, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            coo[elDOF_g, elDOF] = self.__g_q_el(q[elDOF], el)

        return coo

    def W_g(self, t, q):
        coo = CooMatrix((self.nu, self.nla_g))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            coo[elDOF, elDOF_g] = self.__g_q_el(q[elDOF], el).T

        return coo

    def Wla_g_q(self, t, q, la_g):
        coo = CooMatrix((self.nu, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            coo[elDOF, elDOF] = self.__Wla_g_q_el(q[elDOF], la_g[elDOF_g], el)

        return coo

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_dot[elDOF_g] += self.__g_dot_el(q[elDOF], u[elDOF], el)
        return g_dot

    def g_dot_q(self, t, q, u):
        coo = CooMatrix((self.nla_g, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            coo[elDOF_g, elDOF] = self.__g_dot_q_el(q[elDOF], u[elDOF], el)

        return coo

    def g_dot_u(self, t, q):
        coo = CooMatrix((self.nla_g, self.nu))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            coo[elDOF_g, elDOF] = self.__g_q_el(q[elDOF], el)

        return coo

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_ddot[elDOF_g] += self.__g_ddot_el(q[elDOF], u[elDOF], u_dot[elDOF], el)
        return g_ddot

    def g_ddot_q(self, t, q, u, u_dot):
        coo = CooMatrix((self.nla_g, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            coo[elDOF_g, elDOF] = self.__g_ddot_q_el(
                q[elDOF], u[elDOF], u_dot[elDOF], el
            )

        return coo

    def g_ddot_u(self, t, q, u, u_dot):
        coo = CooMatrix((self.nla_g, self.nu))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            coo[elDOF_g, elDOF] = self.__g_ddot_u_el(
                q[elDOF], u[elDOF], u_dot[elDOF], el
            )

        return coo


# TODO: implement time derivatives of constraint functions
class EulerBernoulliDirectorIntegral(I_R12_BubonvGalerkin_R12):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Refactor me!")
        super().__init__(*args, **kwargs)

        self.polynomial_degree_g = (
            self.polynomial_degree_di
        )  # if polynomial_degree_g is None else polynomial_degree_g
        self.nn_el_g = self.polynomial_degree_g + 1  # number of nodes per element

        if self.basis == "B-spline":
            self.knot_vector_g = BSplineKnotVector(
                self.polynomial_degree_g, self.nelement
            )
            self.nn_g = self.nelement + self.polynomial_degree_g  # number of nodes
        elif self.basis == "Lagrange":
            self.knot_vector_g = LagrangeKnotVector(
                self.polynomial_degree_g, self.nelement
            )
            self.nn_g = self.nelement * self.polynomial_degree_g + 1  # number of nodes

        self.nq_n_g = 8  # number of degrees of freedom per node
        self.nla_g = self.nn_g * self.nq_n_g
        self.nla_g_element = self.nn_el_g * self.nq_n_g

        la_g0 = kwargs.get("la_g0")
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.element_span_g = self.knot_vector_g.element_data
        self.mesh_g = Mesh1D(
            self.knot_vector_g,
            self.nquadrature,
            derivative_order=0,
            dim_q=self.nq_n_g,
            basis=self.basis,
        )
        self.elDOF_g = self.mesh_g.elDOF

        # compute shape functions
        self.N_g = self.mesh_g.N

        # degrees of freedom on the element level (for the constraints)
        self.g11DOF = np.arange(self.nn_el_g)
        self.g12DOF = np.arange(self.nn_el_g, 2 * self.nn_el_g)
        self.g13DOF = np.arange(2 * self.nn_el_g, 3 * self.nn_el_g)
        self.g22DOF = np.arange(3 * self.nn_el_g, 4 * self.nn_el_g)
        self.g23DOF = np.arange(4 * self.nn_el_g, 5 * self.nn_el_g)
        self.g33DOF = np.arange(5 * self.nn_el_g, 6 * self.nn_el_g)
        self.g2DOF = np.arange(
            6 * self.nn_el_g, 7 * self.nn_el_g
        )  # unshearability in d2-direction
        self.g3DOF = np.arange(
            7 * self.nn_el_g, 8 * self.nn_el_g
        )  # unshearability in d3-direction

    def __g_el(self, qe, el):
        g = np.zeros(self.nla_g_element)

        for i in range(self.nquadrature):
            NN_dii = self.stack3di(self.N_di[el, i])
            d1 = NN_dii @ qe[self.d1DOF]
            d2 = NN_dii @ qe[self.d2DOF]
            d3 = NN_dii @ qe[self.d3DOF]

            r_xi = self.stack3r(self.N_r_xi[el, i]) @ qe[self.rDOF]

            factor1 = self.N_g[el, i] * self.J[el, i] * self.qw[el, i]
            factor2 = self.N_g[el, i] * self.qw[el, i]

            # director constraints
            g[self.g11DOF] += (d1 @ d1 - 1) * factor1
            g[self.g12DOF] += (d1 @ d2) * factor1
            g[self.g13DOF] += (d1 @ d3) * factor1
            g[self.g22DOF] += (d2 @ d2 - 1) * factor1
            g[self.g23DOF] += (d2 @ d3) * factor1
            g[self.g33DOF] += (d3 @ d3 - 1) * factor1

            # unshearability in d2-direction
            g[self.g2DOF] += d2 @ r_xi * factor2
            g[self.g3DOF] += d3 @ r_xi * factor2

        return g

    def __g_q_el(self, qe, el):
        g_q = np.zeros((self.nla_g_element, self.nq_el))

        for i in range(self.nquadrature):
            NN_dii = self.stack3di(self.N_di[el, i])
            d1 = NN_dii @ qe[self.d1DOF]
            d2 = NN_dii @ qe[self.d2DOF]
            d3 = NN_dii @ qe[self.d3DOF]

            d1_NNi = d1 @ NN_dii
            d2_NNi = d2 @ NN_dii
            d3_NNi = d3 @ NN_dii

            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            r_xi = NN_r_xii @ qe[self.rDOF]

            factor1 = self.N_g[el, i] * self.J[el, i] * self.qw[el, i]
            factor2 = self.N_g[el, i] * self.qw[el, i]

            ######################
            # director constraints
            ######################

            # 2 * delta d1 * d1
            g_q[self.g11DOF[:, None], self.d1DOF] += np.outer(factor1, 2 * d1_NNi)
            # delta d1 * d2
            g_q[self.g12DOF[:, None], self.d1DOF] += np.outer(factor1, d2_NNi)
            # delta d2 * d1
            g_q[self.g12DOF[:, None], self.d2DOF] += np.outer(factor1, d1_NNi)
            # delta d1 * d3
            g_q[self.g13DOF[:, None], self.d1DOF] += np.outer(factor1, d3_NNi)
            # delta d3 * d1
            g_q[self.g13DOF[:, None], self.d3DOF] += np.outer(factor1, d1_NNi)

            # 2 * delta d2 * d2
            g_q[self.g22DOF[:, None], self.d2DOF] += np.outer(factor1, 2 * d2_NNi)
            # delta d2 * d3
            g_q[self.g23DOF[:, None], self.d2DOF] += np.outer(factor1, d3_NNi)
            # delta d3 * d2
            g_q[self.g23DOF[:, None], self.d3DOF] += np.outer(factor1, d2_NNi)

            # 2 * delta d3 * d3
            g_q[self.g33DOF[:, None], self.d3DOF] += np.outer(factor1, 2 * d3_NNi)

            ################################
            # unshearability in d2-direction
            ################################
            g_q[self.g2DOF[:, None], self.rDOF] += np.outer(factor2, d2 @ NN_r_xii)
            g_q[self.g2DOF[:, None], self.d2DOF] += np.outer(factor2, r_xi @ NN_dii)
            g_q[self.g3DOF[:, None], self.rDOF] += np.outer(factor2, d3 @ NN_r_xii)
            g_q[self.g3DOF[:, None], self.d3DOF] += np.outer(factor2, r_xi @ NN_dii)

        return g_q

    def __g_qq_el(self, qe, el):
        g_qq = np.zeros((self.nla_g_element, self.nq_el, self.nq_el))

        for i in range(self.nquadrature):
            NN_dii = self.stack3di(self.N_di[el, i])
            NN_r_xii = self.stack3r(self.N_r_xi[el, i])

            factor1 = NN_dii.T @ NN_dii * self.J[el, i] * self.qw[el, i]
            factor2 = self.N_g[el, i] * self.qw[el, i]

            ######################
            # director constraints
            ######################
            for j, N_gij in enumerate(self.N_g[el, i]):
                N_gij_factor1 = N_gij * factor1

                # 2 * delta d1 * d1
                g_qq[self.g11DOF[j], self.d1DOF[:, None], self.d1DOF] += (
                    2 * N_gij_factor1
                )
                # delta d1 * d2
                g_qq[self.g12DOF[j], self.d1DOF[:, None], self.d2DOF] += N_gij_factor1
                # delta d2 * d1
                g_qq[self.g12DOF[j], self.d2DOF[:, None], self.d1DOF] += N_gij_factor1
                # delta d1 * d3
                g_qq[self.g13DOF[j], self.d1DOF[:, None], self.d3DOF] += N_gij_factor1
                # delta d3 * d1
                g_qq[self.g13DOF[j], self.d3DOF[:, None], self.d1DOF] += N_gij_factor1

                # 2 * delta d2 * d2
                g_qq[self.g22DOF[j], self.d2DOF[:, None], self.d2DOF] += (
                    2 * N_gij_factor1
                )
                # delta d2 * d3
                g_qq[self.g23DOF[j], self.d2DOF[:, None], self.d3DOF] += N_gij_factor1
                # delta d3 * d2
                g_qq[self.g23DOF[j], self.d3DOF[:, None], self.d2DOF] += N_gij_factor1

                # 2 * delta d3 * d3
                g_qq[self.g33DOF[j], self.d3DOF[:, None], self.d3DOF] += (
                    2 * N_gij_factor1
                )

            ################################
            # unshearability in d2-direction
            ################################
            arg1 = np.einsum("i,jl,jk->ikl", factor2, NN_dii, NN_r_xii)
            arg2 = np.einsum("i,jl,jk->ikl", factor2, NN_r_xii, NN_dii)
            g_qq[self.g2DOF[:, None, None], self.rDOF[:, None], self.d2DOF] += arg1
            g_qq[self.g2DOF[:, None, None], self.d2DOF[:, None], self.rDOF] += arg2
            g_qq[self.g3DOF[:, None, None], self.rDOF[:, None], self.d3DOF] += arg1
            g_qq[self.g3DOF[:, None, None], self.d3DOF[:, None], self.rDOF] += arg2

        return g_qq

    def __g_dot_el(self, qe, ue, el):
        raise NotImplementedError("...")

    def __g_dot_q_el(self, qe, ue, el):
        raise NotImplementedError("...")

    def __g_ddot_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError("...")

    def __g_ddot_q_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError("...")

    def __g_ddot_u_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError("...")

    # global constraint functions
    def g(self, t, q):
        g = np.zeros(self.nla_g)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g[elDOF_g] += self.__g_el(q[elDOF], el)
        return g

    def g_q(self, t, q):
        coo = CooMatrix((self.nla_g, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_q = self.__g_q_el(q[elDOF], el)
            coo.extend(g_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))

        return coo

    def W_g(self, t, q):
        coo = CooMatrix((self.nu, self.nla_g))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_q = self.__g_q_el(q[elDOF], el)
            coo.extend(g_q.T, (self.uDOF[elDOF], self.la_gDOF[elDOF_g]))

        return coo

    def Wla_g_q(self, t, q, la_g):
        coo = CooMatrix((self.nu, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_qq = self.__g_qq_el(q[elDOF], el)
            coo.extend(
                np.einsum("i,ijk->jk", la_g[elDOF_g], g_qq),
                (self.uDOF[elDOF], self.qDOF[elDOF]),
            )

        return coo

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_dot[elDOF_g] += self.__g_dot_el(q[elDOF], u[elDOF], el)
        return g_dot

    def g_dot_q(self, t, q, u):
        coo = CooMatrix((self.nla_g, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_dot_q = self.__g_dot_q_el(q[elDOF], u[elDOF], el)
            coo.extend(g_dot_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))

        return coo

    def g_dot_u(self, t, q):
        coo = CooMatrix((self.nla_g, self.nu))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_dot_u = self.__g_q_el(q[elDOF], el)
            coo.extend(g_dot_u, (self.la_gDOF[elDOF_g], self.uDOF[elDOF]))

        return coo

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_ddot[elDOF_g] += self.__g_ddot_el(q[elDOF], u[elDOF], u_dot[elDOF], el)
        return g_ddot

    def g_ddot_q(self, t, q, u, u_dot):
        coo = CooMatrix((self.nla_g, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_ddot_q = self.__g_ddot_q_el(q[elDOF], u[elDOF], u_dot[elDOF], el)
            coo.extend(g_ddot_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))

        return coo

    def g_ddot_u(self, t, q, u, u_dot):
        coo = CooMatrix((self.nla_g, self.nu))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_ddot_u = self.__g_ddot_u_el(q[elDOF], u[elDOF], u_dot[elDOF], el)
            coo.extend(g_ddot_u, (self.la_gDOF[elDOF_g], self.uDOF[elDOF]))

        return coo


# TODO: implement time derivatives of constraint functions
class InextensibleEulerBernoulliDirectorIntegral(I_R12_BubonvGalerkin_R12):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Refactor me!")
        super().__init__(*args, **kwargs)

        self.polynomial_degree_g = (
            self.polynomial_degree_di
        )  # if polynomial_degree_g is None else polynomial_degree_g
        self.nn_el_g = self.polynomial_degree_g + 1  # number of nodes per element

        if self.basis == "B-spline":
            self.knot_vector_g = BSplineKnotVector(
                self.polynomial_degree_g, self.nelement
            )
            self.nn_g = self.nelement + self.polynomial_degree_g  # number of nodes
        elif self.basis == "Lagrange":
            self.knot_vector_g = LagrangeKnotVector(
                self.polynomial_degree_g, self.nelement
            )
            self.nn_g = self.nelement * self.polynomial_degree_g + 1  # number of nodes

        self.nq_n_g = 9  # number of degrees of freedom per node
        self.nla_g = self.nn_g * self.nq_n_g
        self.nla_g_element = self.nn_el_g * self.nq_n_g

        la_g0 = kwargs.get("la_g0")
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.element_span_g = self.knot_vector_g.element_data
        self.mesh_g = Mesh1D(
            self.knot_vector_g,
            self.nquadrature,
            derivative_order=0,
            dim_q=self.nq_n_g,
            basis=self.basis,
        )
        self.elDOF_g = self.mesh_g.elDOF

        # compute shape functions
        self.N_g = self.mesh_g.N

        # degrees of freedom on the element level (for the constraints)
        self.g11DOF = np.arange(self.nn_el_g)
        self.g12DOF = np.arange(self.nn_el_g, 2 * self.nn_el_g)
        self.g13DOF = np.arange(2 * self.nn_el_g, 3 * self.nn_el_g)
        self.g22DOF = np.arange(3 * self.nn_el_g, 4 * self.nn_el_g)
        self.g23DOF = np.arange(4 * self.nn_el_g, 5 * self.nn_el_g)
        self.g33DOF = np.arange(5 * self.nn_el_g, 6 * self.nn_el_g)
        self.g2DOF = np.arange(
            6 * self.nn_el_g, 7 * self.nn_el_g
        )  # unshearability in d2-direction
        self.g3DOF = np.arange(
            7 * self.nn_el_g, 8 * self.nn_el_g
        )  # unshearability in d3-direction
        self.g1DOF = np.arange(8 * self.nn_el_g, 9 * self.nn_el_g)  # inextensibility

    def __g_el(self, qe, el):
        g = np.zeros(self.nla_g_element)

        for i in range(self.nquadrature):
            NN_dii = self.stack3di(self.N_di[el, i])
            d1 = NN_dii @ qe[self.d1DOF]
            d2 = NN_dii @ qe[self.d2DOF]
            d3 = NN_dii @ qe[self.d3DOF]

            r_xi = self.stack3r(self.N_r_xi[el, i]) @ qe[self.rDOF]

            factor1 = self.N_g[el, i] * self.J[el, i] * self.qw[el, i]
            factor2 = self.N_g[el, i] * self.qw[el, i]

            # director constraints
            g[self.g11DOF] += (d1 @ d1 - 1) * factor1
            g[self.g12DOF] += (d1 @ d2) * factor1
            g[self.g13DOF] += (d1 @ d3) * factor1
            g[self.g22DOF] += (d2 @ d2 - 1) * factor1
            g[self.g23DOF] += (d2 @ d3) * factor1
            g[self.g33DOF] += (d3 @ d3 - 1) * factor1

            # unshearability in d2-direction
            g[self.g2DOF] += d2 @ r_xi * factor2
            g[self.g3DOF] += d3 @ r_xi * factor2

            # inextensibility
            g[self.g1DOF] += (d1 @ r_xi - self.J[el, i]) * factor2
            # g[self.g1DOF] += (d1 @ r_xi / self.J[el, i] - 1) * factor1

            # r_s = r_xi / self.J[el, i]
            # g[self.g1DOF] += (r_s @ r_s - 1) * factor1

        return g

    def __g_q_el(self, qe, el):
        g_q = np.zeros((self.nla_g_element, self.nq_el))

        for i in range(self.nquadrature):
            NN_dii = self.stack3di(self.N_di[el, i])
            d1 = NN_dii @ qe[self.d1DOF]
            d2 = NN_dii @ qe[self.d2DOF]
            d3 = NN_dii @ qe[self.d3DOF]

            d1_NNi = d1 @ NN_dii
            d2_NNi = d2 @ NN_dii
            d3_NNi = d3 @ NN_dii

            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            r_xi = NN_r_xii @ qe[self.rDOF]

            factor1 = self.N_g[el, i] * self.J[el, i] * self.qw[el, i]
            factor2 = self.N_g[el, i] * self.qw[el, i]

            ######################
            # director constraints
            ######################

            # 2 * delta d1 * d1
            g_q[self.g11DOF[:, None], self.d1DOF] += np.outer(factor1, 2 * d1_NNi)
            # delta d1 * d2
            g_q[self.g12DOF[:, None], self.d1DOF] += np.outer(factor1, d2_NNi)
            # delta d2 * d1
            g_q[self.g12DOF[:, None], self.d2DOF] += np.outer(factor1, d1_NNi)
            # delta d1 * d3
            g_q[self.g13DOF[:, None], self.d1DOF] += np.outer(factor1, d3_NNi)
            # delta d3 * d1
            g_q[self.g13DOF[:, None], self.d3DOF] += np.outer(factor1, d1_NNi)

            # 2 * delta d2 * d2
            g_q[self.g22DOF[:, None], self.d2DOF] += np.outer(factor1, 2 * d2_NNi)
            # delta d2 * d3
            g_q[self.g23DOF[:, None], self.d2DOF] += np.outer(factor1, d3_NNi)
            # delta d3 * d2
            g_q[self.g23DOF[:, None], self.d3DOF] += np.outer(factor1, d2_NNi)

            # 2 * delta d3 * d3
            g_q[self.g33DOF[:, None], self.d3DOF] += np.outer(factor1, 2 * d3_NNi)

            ################################
            # unshearability in d2-direction
            ################################
            g_q[self.g2DOF[:, None], self.rDOF] += np.outer(factor2, d2 @ NN_r_xii)
            g_q[self.g2DOF[:, None], self.d2DOF] += np.outer(factor2, r_xi @ NN_dii)
            g_q[self.g3DOF[:, None], self.rDOF] += np.outer(factor2, d3 @ NN_r_xii)
            g_q[self.g3DOF[:, None], self.d3DOF] += np.outer(factor2, r_xi @ NN_dii)

            #################
            # inextensibility
            #################
            # g[self.g1DOF] += (d1 @ r_xi - self.J[el, i]) * factor2
            g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor2, d1 @ NN_r_xii)
            g_q[self.g1DOF[:, None], self.d1DOF] += np.outer(factor2, r_xi @ NN_dii)

            # # g[self.g1DOF] += (d1 @ r_xi / self.J[el, i] - 1) * factor1
            # g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor1, d1 @ NN_r_xii / self.J[el, i])
            # g_q[self.g1DOF[:, None], self.d1DOF] += np.outer(factor1, r_xi / self.J[el, i] @ NN_dii)

            # r_s = r_xi / self.J[el, i]
            # # g[self.g1DOF] += (r_s @ r_s - 1) * factor1
            # g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor1, 2 * r_s @ NN_r_xii / self.J[el, i])

        return g_q

    def __g_qq_el(self, qe, el):
        g_qq = np.zeros((self.nla_g_element, self.nq_el, self.nq_el))

        for i in range(self.nquadrature):
            NN_dii = self.stack3di(self.N_di[el, i])
            NN_r_xii = self.stack3r(self.N_r_xi[el, i])

            factor1 = NN_dii.T @ NN_dii * self.J[el, i] * self.qw[el, i]
            factor2 = self.N_g[el, i] * self.qw[el, i]

            ######################
            # director constraints
            ######################
            for j, N_gij in enumerate(self.N_g[el, i]):
                N_gij_factor1 = N_gij * factor1

                # 2 * delta d1 * d1
                g_qq[self.g11DOF[j], self.d1DOF[:, None], self.d1DOF] += (
                    2 * N_gij_factor1
                )
                # delta d1 * d2
                g_qq[self.g12DOF[j], self.d1DOF[:, None], self.d2DOF] += N_gij_factor1
                # delta d2 * d1
                g_qq[self.g12DOF[j], self.d2DOF[:, None], self.d1DOF] += N_gij_factor1
                # delta d1 * d3
                g_qq[self.g13DOF[j], self.d1DOF[:, None], self.d3DOF] += N_gij_factor1
                # delta d3 * d1
                g_qq[self.g13DOF[j], self.d3DOF[:, None], self.d1DOF] += N_gij_factor1

                # 2 * delta d2 * d2
                g_qq[self.g22DOF[j], self.d2DOF[:, None], self.d2DOF] += (
                    2 * N_gij_factor1
                )
                # delta d2 * d3
                g_qq[self.g23DOF[j], self.d2DOF[:, None], self.d3DOF] += N_gij_factor1
                # delta d3 * d2
                g_qq[self.g23DOF[j], self.d3DOF[:, None], self.d2DOF] += N_gij_factor1

                # 2 * delta d3 * d3
                g_qq[self.g33DOF[j], self.d3DOF[:, None], self.d3DOF] += (
                    2 * N_gij_factor1
                )

            ################################
            # unshearability in d2-direction
            ################################
            arg1 = np.einsum("i,jl,jk->ikl", factor2, NN_dii, NN_r_xii)
            arg2 = np.einsum("i,jl,jk->ikl", factor2, NN_r_xii, NN_dii)
            g_qq[self.g2DOF[:, None, None], self.rDOF[:, None], self.d2DOF] += arg1
            g_qq[self.g2DOF[:, None, None], self.d2DOF[:, None], self.rDOF] += arg2
            g_qq[self.g3DOF[:, None, None], self.rDOF[:, None], self.d3DOF] += arg1
            g_qq[self.g3DOF[:, None, None], self.d3DOF[:, None], self.rDOF] += arg2

            #################
            # inextensibility
            #################
            g_qq[self.g1DOF[:, None, None], self.rDOF[:, None], self.d1DOF] += arg1
            g_qq[self.g1DOF[:, None, None], self.d1DOF[:, None], self.rDOF] += arg2
            # g_qq[self.g1DOF[:, None, None], self.rDOF[:, None], self.rDOF] += np.einsum('i,jl,jk->ikl', 2 * factor2, NN_r_xii, NN_r_xii)

            # # g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor1, d1 @ NN_r_xii / self.J[el, i])
            # # g_q[self.g1DOF[:, None], self.d1DOF] += np.outer(factor1, r_xi / self.J[el, i] @ NN_dii)
            # g_qq[self.g1DOF[:, None, None], self.rDOF[:, None], self.d1DOF] += arg1
            # g_qq[self.g1DOF[:, None, None], self.d1DOF[:, None], self.rDOF] += arg2

            # # # r_s = r_xi / self.J[el, i]
            # # g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor1, 2 * r_s @ NN_r_xii / self.J[el, i])
            # # g_qq[self.g1DOF[:, None, None], self.rDOF[:, None], self.rDOF] += np.einsum('i,jl,jk->ikl', factor2 * self.J[el, i], NN_r_xii / self.J[el, i], NN_r_xii / self.J[el, i])
            # g_qq[self.g1DOF[:, None, None], self.rDOF[:, None], self.rDOF] += np.einsum('i,jl,jk->ikl', 2 * factor2, NN_r_xii, NN_r_xii / self.J[el, i])

        return g_qq

    def __g_dot_el(self, qe, ue, el):
        raise NotImplementedError("...")

    def __g_dot_q_el(self, qe, ue, el):
        raise NotImplementedError("...")

    def __g_ddot_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError("...")

    def __g_ddot_q_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError("...")

    def __g_ddot_u_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError("...")

    # global constraint functions
    def g(self, t, q):
        g = np.zeros(self.nla_g)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g[elDOF_g] += self.__g_el(q[elDOF], el)
        return g

    def g_q(self, t, q):
        coo = CooMatrix((self.nla_g, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_q = self.__g_q_el(q[elDOF], el)
            coo.extend(g_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))

        return coo

    def W_g(self, t, q):
        coo = CooMatrix((self.nq, self.nla_g))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_q = self.__g_q_el(q[elDOF], el)
            coo.extend(g_q.T, (self.uDOF[elDOF], self.la_gDOF[elDOF_g]))

        return coo

    def Wla_g_q(self, t, q, la_g):
        coo = CooMatrix((self.nu, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_qq = self.__g_qq_el(q[elDOF], el)
            coo.extend(
                np.einsum("i,ijk->jk", la_g[elDOF_g], g_qq),
                (self.uDOF[elDOF], self.qDOF[elDOF]),
            )

        return coo

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_dot[elDOF_g] += self.__g_dot_el(q[elDOF], u[elDOF], el)
        return g_dot

    def g_dot_q(self, t, q, u):
        coo = CooMatrix((self.nla_g, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_dot_q = self.__g_dot_q_el(q[elDOF], u[elDOF], el)
            coo.extend(g_dot_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))

        return coo

    def g_dot_u(self, t, q):
        coo = CooMatrix((self.nla_g, self.nu))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_dot_u = self.__g_q_el(q[elDOF], el)
            coo.extend(g_dot_u, (self.la_gDOF[elDOF_g], self.uDOF[elDOF]))

        return coo

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_ddot[elDOF_g] += self.__g_ddot_el(q[elDOF], u[elDOF], u_dot[elDOF], el)
        return g_ddot

    def g_ddot_q(self, t, q, u, u_dot):
        coo = CooMatrix((self.nla_g, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_ddot_q = self.__g_ddot_q_el(q[elDOF], u[elDOF], u_dot[elDOF], el)
            coo.extend(g_ddot_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))

        return coo

    def g_ddot_u(self, t, q, u, u_dot):
        coo = CooMatrix((self.nla_g, self.nu))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_ddot_u = self.__g_ddot_u_el(q[elDOF], u[elDOF], u_dot[elDOF], el)
            coo.extend(g_ddot_u, (self.la_gDOF[elDOF_g], self.uDOF[elDOF]))

        return coo
