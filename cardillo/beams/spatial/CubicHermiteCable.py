import numpy as np
import meshio
import os

from cardillo.utility.coo import Coo
from cardillo.discretization.Hermite import HermiteNodeVector
from cardillo.math import (
    e1,
    norm,
    cross3,
    skew2ax,
    smallest_rotation,
    approx_fprime,
    sign,
)
from cardillo.discretization.mesh1D import Mesh1D


class CubicHermiteCable:
    def __init__(
        self,
        material_model,
        A_rho0,  # TODO
        B_rho0,  # TODO
        C_rho0,  # TODO
        polynomial_degree,
        nquadrature,
        nelement,
        Q,
        q0=None,
        u0=None,
        # symmetric_formulation=False,
        symmetric_formulation=True,
    ):
        # assume symmetric cross section
        self.symmetric_formulation = symmetric_formulation

        # beam properties
        self.materialModel = material_model  # material model
        self.A_rho0 = A_rho0  # line density
        self.B_rho0 = B_rho0  # first moment
        self.C_rho0 = C_rho0  # second moment

        # material model
        self.material_model = material_model

        # discretization parameters
        self.polynomial_degree_r = polynomial_degree  # polynomial degree
        self.nquadrature = nquadrature  # number of quadrature points
        self.nelement = nelement  # number of elements

        self.knot_vector = HermiteNodeVector(polynomial_degree, nelement)
        self.nnodes = nnodes = nelement + 1  # number of nodes

        self.nnodes_element = nnodes_element = 2  # number of nodes per element

        # number of degrees of freedom per node of the centerline
        self.nq_node = nq_node = 6

        self.mesh = Mesh1D(
            self.knot_vector,
            nquadrature,
            nq_node,
            derivative_order=2,
            basis="Hermite",
        )

        self.nq = nnodes * nq_node  # total number of generalized coordinates
        self.nu = self.nq
        self.nq_el = (
            nnodes_element * nq_node
        )  # total number of generalized coordinates per element

        # global connectivity matrix
        self.elDOF = self.mesh.elDOF
        self.nodalDOF = (
            np.arange(self.nq_node * self.nnodes).reshape(self.nq_node, self.nnodes).T
        )

        # shape functions
        self.N_r = self.mesh.N  # shape functions
        self.N_r_xi = self.mesh.N_xi  # first derivative w.r.t. xi
        self.N_r_xixi = self.mesh.N_xixi  # second derivative w.r.t. xi

        # quadrature points
        self.qp = self.mesh.qp  # quadrature points
        self.qw = self.mesh.wp  # quadrature weights

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions = self.mesh.eval_basis

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q  # reference configuration
        self.q0 = Q.copy() if q0 is None else q0  # initial configuration
        self.u0 = np.zeros(self.nu) if u0 is None else u0  # initial velocities

        # precompute values of the reference configuration in order to save computation time
        self.J0 = np.zeros(
            (nelement, nquadrature)
        )  # stretch of the reference centerline, J in Harsch2020b (5)
        self.Kappa0 = np.zeros(
            (nelement, nquadrature, 3)
        )  # curvature of the reference configuration

        for el in range(nelement):
            # precompute quantities of the reference configuration
            qe = self.Q[self.elDOF[el]]

            for i in range(nquadrature):
                # build matrix of shape function derivatives
                NN_r_xii = self.stack_Hermite(self.N_r_xi[el, i])
                NN_r_xixii = self.stack_Hermite(self.N_r_xixi[el, i])

                # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
                # the parameter space \xi and thus need to be transformed later
                r_xi = NN_r_xii @ qe
                r_xixi = NN_r_xixii @ qe
                ji = norm(r_xi)
                self.J0[el, i] = ji

                # first director
                d1 = r_xi / ji

                # build rotation matrices
                R = smallest_rotation(e1, d1)
                d1, d2, d3 = R.T

                # print(f"d1: {d1}")
                # print(f"R:\n{R}")

                # raise NotImplementedError(
                #     "TODO: We have an error in the shape functions. A straight " +
                #     "rod yields negative tangent vector values."
                # )

                # torsional and flexural strains
                if self.symmetric_formulation:
                    # first directors derivative
                    d1_s = (np.eye(3) - np.outer(d1, d1)) @ r_xixi / (ji * ji)

                    self.Kappa0[el, i] = cross3(d1, d1_s)
                else:
                    self.Kappa0[el, i] = np.array(
                        [
                            0,  # no torsion for cable element
                            -(d3 @ r_xixi) / ji**2,
                            (d2 @ r_xixi) / ji**2,
                        ]
                    )

    @staticmethod
    def straight_configuration(
        nEl,
        L,
        r_OP=np.zeros(3),
        A_IK=np.eye(3),
    ):
        # linear spaced points
        nn = nEl + 1
        xis = np.linspace(0, 1, num=nn)

        q0 = np.zeros(6 * nn)
        for i, xi in enumerate(xis):
            r = r_OP + A_IK @ (xi * np.array([L, 0, 0]))
            t = L * A_IK @ e1
            q0[6 * i : 6 * (i + 1)] = np.concatenate([r, t])

        return q0

    def element_number(self, xi):
        # note the elements coincide for both meshes!
        return self.knot_vector.element_number(xi)[0]

    def stack_Hermite(self, N):
        NN = np.hstack(
            [
                N[0] * np.eye(3),
                N[1] * np.eye(3),
                N[2] * np.eye(3),
                N[3] * np.eye(3),
            ]
        )
        return NN

    def assembler_callback(self):
        self.__M_coo()

    #########################################
    # equations of motion
    #########################################
    # TODO!
    def M_el(self, el):
        Me = np.zeros((self.nq_el, self.nq_el))
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
            J0i = self.J0[el, i]
            Kappa0_i = self.Kappa0[el, i]

            # build matrix of shape function derivatives
            NN_r_xii = self.stack_Hermite(self.N_r_xi[el, i])
            NN_r_xixii = self.stack_Hermite(self.N_r_xixi[el, i])

            # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
            # the parameter space \xi and thus need to be transformed later
            r_xi = NN_r_xii @ qe
            r_xixi = NN_r_xixii @ qe
            ji = norm(r_xi)
            self.J0[el, i] = ji

            # first director
            d1 = r_xi / ji

            # build rotation matrices
            R = smallest_rotation(e1, d1)
            d1, d2, d3 = R.T

            # print(f"\nd1: {d1}")
            # print(f"R:\n{R}")

            # torsional and flexural strains
            if self.symmetric_formulation:
                # first directors derivative
                d1_s = (np.eye(3) - np.outer(d1, d1)) @ r_xixi / (ji * J0i)

                Kappa_i = cross3(d1, d1_s)
            else:
                Kappa_i = np.array(
                    [
                        0,  # no torsion for cable element
                        -(d3 @ r_xixi) / (ji * J0i),
                        (d2 @ r_xixi) / (ji * J0i),
                    ]
                )

            # axial stretch
            lambda_ = ji / J0i

            # evaluate strain energy function
            Ee += (
                self.material_model.potential(lambda_, Kappa_i, Kappa0_i)
                * J0i
                * self.qw[el, i]
            )

        return Ee

    def f_pot(self, t, q):
        f = np.zeros(self.nu)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f[elDOF] += self.f_pot_el(q[elDOF], el)
        return f

    def f_pot_el(self, qe, el):
        return -approx_fprime(qe, lambda qe: self.E_pot_el(qe, el), method="3-point")

    def f_pot_q(self, t, q, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            Ke = self.f_pot_q_el(q[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(Ke, (self.uDOF[elDOF], self.qDOF[elDOF]))

    def f_pot_q_el(self, qe, el):
        return approx_fprime(qe, lambda qe: self.f_pot_el(qe, el), method="2-point")

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        return u

    def B(self, t, q, coo):
        coo.extend_diag(np.ones(self.nq), (self.qDOF, self.uDOF))

    def q_ddot(self, t, q, u, u_dot):
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
        N, _, _ = self.basis_functions(frame_ID[0])
        NN = self.stack_Hermite(N)
        return NN @ q + self.A_IK(t, q, frame_ID=frame_ID) @ K_r_SP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        N, _, _ = self.basis_functions(frame_ID[0])
        NN = self.stack_Hermite(N)

        r_OP_q = NN + np.einsum(
            "ijk,j->ik", self.A_IK_q(t, q, frame_ID=frame_ID), K_r_SP
        )
        return r_OP_q

    def A_IK(self, t, q, frame_ID):
        _, N_xi, _ = self.basis_functions(frame_ID[0])
        NN_xi = self.stack_Hermite(N_xi)
        r_xi = NN_xi @ q

        # compute first director
        ji = norm(r_xi)
        d1 = r_xi / ji

        # build rotation matrices
        return smallest_rotation(e1, d1)

    # TODO
    def A_IK_q(self, t, q, frame_ID):
        return approx_fprime(q, lambda q: self.A_IK(t, q, frame_ID))

    # TODO
    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3)):
        N, _, _ = self.basis_functions(frame_ID[0])
        NN = self.stack_Hermite(N)

        v_P = NN @ u + self.A_IK(t, q, frame_ID) @ cross3(
            self.K_Omega(t, q, u, frame_ID=frame_ID), K_r_SP
        )
        return v_P

    # TODO
    def v_P_q(self, t, q, u, frame_ID, K_r_SP=np.zeros(3)):
        return approx_fprime(q, lambda q: self.v_P(t, q, u, frame_ID, K_r_SP))

    # TODO
    def J_P(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        return approx_fprime(
            np.zeros_like(q),
            lambda u: self.v_P(t, q, u, frame_ID, K_r_SP),
            method="3-point",
        )

    # TODO
    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return approx_fprime(
            q, lambda q: self.J_P(t, q, frame_ID, K_r_SP), method="2-point"
        )

    # TODO: optimized implementation for boundaries
    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=np.zeros(3)):
        N, _ = self.basis_functions(frame_ID[0])
        NN = self.stack_Hermite(N)

        K_Omega = self.K_Omega(t, q, u, frame_ID=frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID=frame_ID)
        a_P = NN @ u_dot + self.A_IK(t, q, frame_ID=frame_ID) @ (
            cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP))
        )
        return a_P

    def a_P_q(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return approx_fprime(q, lambda q: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP))

    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return approx_fprime(u, lambda u: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP))

    # TODO: The implementation cross3(d1, d1_dot) yields the angular velocity
    # in the I-frame?
    def K_Omega(self, t, q, u, frame_ID):
        _, N_xi, _ = self.basis_functions(frame_ID[0])
        NN_xi = self.stack_Hermite(N_xi)
        r_xi = NN_xi @ q
        r_xi_dot = NN_xi @ u

        # compute first director
        ji = norm(r_xi)
        d1 = r_xi / ji

        if self.symmetric_formulation:
            # first directors derivative
            d1_dot = (np.eye(3) - np.outer(d1, d1)) @ r_xi_dot / ji

            K_Omega = cross3(d1, d1_dot)
        else:
            # build rotation matrices
            A = smallest_rotation(e1, d1)
            d1, d2, d3 = A.T

            K_Omega = np.array(
                [
                    0,
                    -(d3 @ r_xi_dot) / ji,
                    (d2 @ r_xi_dot) / ji,
                ]
            )
        return K_Omega

    # TODO:
    def K_J_R(self, t, q, frame_ID):
        return approx_fprime(
            np.zeros_like(q),
            lambda u: self.K_Omega(t, q, u, frame_ID),
            method="3-point",
        )

    # TODO:
    def K_J_R_q(self, t, q, frame_ID):
        return approx_fprime(q, lambda q: self.K_J_R(t, q, frame_ID), method="2-point")

    # TODO:
    def K_Psi(self, t, q, u, u_dot, frame_ID):
        raise NotImplementedError("")
        N, _ = self.basis_functions_phi(frame_ID[0])
        NN = self.stack3di(N)

        d1 = NN @ q[self.phiDOF]
        d2 = NN @ q[self.d2DOF]
        d3 = NN @ q[self.d3DOF]
        A_IK = np.vstack((d1, d2, d3)).T

        d1_dot = NN @ u[self.phiDOF]
        d2_dot = NN @ u[self.d2DOF]
        d3_dot = NN @ u[self.d3DOF]
        A_IK_dot = np.vstack((d1_dot, d2_dot, d3_dot)).T

        d1_ddot = NN @ u_dot[self.phiDOF]
        d2_ddot = NN @ u_dot[self.d2DOF]
        d3_ddot = NN @ u_dot[self.d3DOF]
        A_IK_ddot = np.vstack((d1_ddot, d2_ddot, d3_ddot)).T

        K_Psi_tilde = A_IK_dot.T @ A_IK_dot + A_IK.T @ A_IK_ddot
        return skew2ax(K_Psi_tilde)

    ####################################################
    # body force
    ####################################################
    def distributed_force1D_el(self, force, t, el):
        fe = np.zeros(self.nq_el)
        for i in range(self.nquadrature):
            NNi = self.stack_Hermite(self.N_r[el, i])
            fe += NNi.T @ force(t, self.qp[el, i]) * self.J0[el, i] * self.qw[el, i]
        return fe

    def distributed_force1D(self, t, q, force):
        f = np.zeros(self.nq)
        for el in range(self.nelement):
            f[self.elDOF[el]] += self.distributed_force1D_el(force, t, el)
        return f

    # def body_force_q(self, t, q, coo, force):
    def distributed_force1D_q(self, t, q, coo, force):
        pass

    ####################################################
    # visualization
    ####################################################
    # TODO: Correct returned nodes!
    def nodes(self, q):
        # return q[self.qDOF][: 3 * self.nnodes].reshape(3, -1)
        return np.array(np.array_split(q[self.qDOF], self.nnodes))[:, :3].T

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

    # def plot_centerline(self, ax, q, n=100, color="black"):
    #     ax.plot(*self.nodes(q), linestyle="dashed", marker="o", color=color)
    #     ax.plot(*self.centerline(q, n=n), linestyle="solid", color=color)

    # def plot_frames(self, ax, q, n=10, length=1):
    #     r, d1, d2, d3 = self.frames(q, n=n)
    #     ax.quiver(*r, *d1, color="red", length=length)
    #     ax.quiver(*r, *d2, color="green", length=length)
    #     ax.quiver(*r, *d3, color="blue", length=length)

    # ############
    # # vtk export
    # ############
    # def post_processing_vtk_volume_circle(self, t, q, filename, R, binary=False):
    #     # This is mandatory, otherwise we cannot construct the 3D continuum without L2 projection!
    #     assert (
    #         self.polynomial_degree_r == self.polynomial_degree_phi
    #     ), "Not implemented for mixed polynomial degrees"

    #     # rearrange generalized coordinates from solver ordering to Piegl's Pw 3D array
    #     nn_xi = self.nEl + self.polynomial_degree_r
    #     nEl_eta = 1
    #     nEl_zeta = 4
    #     # see Cotrell2009 Section 2.4.2
    #     # TODO: Maybe eta and zeta have to be interchanged
    #     polynomial_degree_eta = 1
    #     polynomial_degree_zeta = 2
    #     nn_eta = nEl_eta + polynomial_degree_eta
    #     nn_zeta = nEl_zeta + polynomial_degree_zeta

    #     # # TODO: We do the hard coded case for rectangular cross section here, but this has to be extended to the circular cross section case too!
    #     # as_ = np.linspace(-a/2, a/2, num=nn_eta, endpoint=True)
    #     # bs_ = np.linspace(-b/2, b/2, num=nn_eta, endpoint=True)

    #     circle_points = (
    #         0.5
    #         * R
    #         * np.array(
    #             [
    #                 [1, 0, 0],
    #                 [1, 1, 0],
    #                 [0, 1, 0],
    #                 [-1, 1, 0],
    #                 [-1, 0, 0],
    #                 [-1, -1, 0],
    #                 [0, -1, 0],
    #                 [1, -1, 0],
    #                 [1, 0, 0],
    #             ],
    #             dtype=float,
    #         )
    #     )

    #     Pw = np.zeros((nn_xi, nn_eta, nn_zeta, 3))
    #     for i in range(self.nn_r):
    #         qr = q[self.nodalDOF_r[i]]
    #         q_di = q[self.nodalDOF_phi[i]]
    #         A_IK = q_di.reshape(3, 3, order="F")  # TODO: Check this!

    #         for k, point in enumerate(circle_points):
    #             # Note: eta index is always 0!
    #             Pw[i, 0, k] = qr + A_IK @ point

    #     if self.basis == "B-spline":
    #         knot_vector_eta = Knot_vector(polynomial_degree_eta, nEl_eta)
    #         knot_vector_zeta = Knot_vector(polynomial_degree_zeta, nEl_zeta)
    #     elif self.basis == "lagrange":
    #         knot_vector_eta = Node_vector(polynomial_degree_eta, nEl_eta)
    #         knot_vector_zeta = Node_vector(polynomial_degree_zeta, nEl_zeta)
    #     knot_vector_objs = [self.knot_vector_r, knot_vector_eta, knot_vector_zeta]
    #     degrees = (
    #         self.polynomial_degree_r,
    #         polynomial_degree_eta,
    #         polynomial_degree_zeta,
    #     )

    #     # Build Bezier patches from B-spline control points
    #     from cardillo.discretization.B_spline import decompose_B_spline_volume

    #     Qw = decompose_B_spline_volume(knot_vector_objs, Pw)

    #     nbezier_xi, nbezier_eta, nbezier_zeta, p1, q1, r1, dim = Qw.shape

    #     # build vtk mesh
    #     n_patches = nbezier_xi * nbezier_eta * nbezier_zeta
    #     patch_size = p1 * q1 * r1
    #     points = np.zeros((n_patches * patch_size, dim))
    #     cells = []
    #     HigherOrderDegrees = []
    #     RationalWeights = []
    #     vtk_cell_type = "VTK_BEZIER_HEXAHEDRON"
    #     from PyPanto.miscellaneous.indexing import flat3D, rearange_vtk3D

    #     for i in range(nbezier_xi):
    #         for j in range(nbezier_eta):
    #             for k in range(nbezier_zeta):
    #                 idx = flat3D(i, j, k, (nbezier_xi, nbezier_eta))
    #                 point_range = np.arange(idx * patch_size, (idx + 1) * patch_size)
    #                 points[point_range] = rearange_vtk3D(Qw[i, j, k])

    #                 cells.append((vtk_cell_type, point_range[None]))
    #                 HigherOrderDegrees.append(np.array(degrees, dtype=float)[None])
    #                 weight = np.sqrt(2) / 2
    #                 # tmp = np.array([np.sqrt(2) / 2, 1.0])
    #                 # RationalWeights.append(np.tile(tmp, 8)[None])
    #                 weights_vertices = weight * np.ones(8)
    #                 weights_edges = np.ones(4 * nn_xi)
    #                 weights_faces = np.ones(2)
    #                 weights_volume = np.ones(nn_xi - 2)
    #                 weights = np.concatenate(
    #                     (weights_edges, weights_vertices, weights_faces, weights_volume)
    #                 )
    #                 # weights = np.array([weight, weight, weight, weight,
    #                 #                     1.0,    1.0,    1.0,    1.0,
    #                 #                     0.0,    0.0,    0.0,    0.0 ])
    #                 weights = np.ones_like(point_range)
    #                 RationalWeights.append(weights[None])

    #     # RationalWeights = np.ones(len(points))
    #     RationalWeights = 2 * (np.random.rand(len(points)) + 1)

    #     # write vtk mesh using meshio
    #     meshio.write_points_cells(
    #         # filename.parent / (filename.stem + '.vtu'),
    #         filename,
    #         points,
    #         cells,
    #         point_data={
    #             "RationalWeights": RationalWeights,
    #         },
    #         cell_data={"HigherOrderDegrees": HigherOrderDegrees},
    #         binary=binary,
    #     )

    # def post_processing_vtk_volume(self, t, q, filename, circular=True, binary=False):
    #     # This is mandatory, otherwise we cannot construct the 3D continuum without L2 projection!
    #     assert (
    #         self.polynomial_degree_r == self.polynomial_degree_phi
    #     ), "Not implemented for mixed polynomial degrees"

    #     # rearrange generalized coordinates from solver ordering to Piegl's Pw 3D array
    #     nn_xi = self.nEl + self.polynomial_degree_r
    #     nEl_eta = 1
    #     nEl_zeta = 1
    #     if circular:
    #         polynomial_degree_eta = 2
    #         polynomial_degree_zeta = 2
    #     else:
    #         polynomial_degree_eta = 1
    #         polynomial_degree_zeta = 1
    #     nn_eta = nEl_eta + polynomial_degree_eta
    #     nn_zeta = nEl_zeta + polynomial_degree_zeta

    #     # TODO: We do the hard coded case for rectangular cross section here, but this has to be extended to the circular cross section case too!
    #     if circular:
    #         r = 0.2
    #         a = b = r
    #     else:
    #         a = 0.2
    #         b = 0.1
    #     as_ = np.linspace(-a / 2, a / 2, num=nn_eta, endpoint=True)
    #     bs_ = np.linspace(-b / 2, b / 2, num=nn_eta, endpoint=True)

    #     Pw = np.zeros((nn_xi, nn_eta, nn_zeta, 3))
    #     for i in range(self.nn_r):
    #         qr = q[self.nodalDOF_r[i]]
    #         q_di = q[self.nodalDOF_phi[i]]
    #         A_IK = q_di.reshape(3, 3, order="F")  # TODO: Check this!

    #         for j, aj in enumerate(as_):
    #             for k, bk in enumerate(bs_):
    #                 Pw[i, j, k] = qr + A_IK @ np.array([0, aj, bk])

    #     if self.basis == "B-spline":
    #         knot_vector_eta = Knot_vector(polynomial_degree_eta, nEl_eta)
    #         knot_vector_zeta = Knot_vector(polynomial_degree_zeta, nEl_zeta)
    #     elif self.basis == "lagrange":
    #         knot_vector_eta = Node_vector(polynomial_degree_eta, nEl_eta)
    #         knot_vector_zeta = Node_vector(polynomial_degree_zeta, nEl_zeta)
    #     knot_vector_objs = [self.knot_vector_r, knot_vector_eta, knot_vector_zeta]
    #     degrees = (
    #         self.polynomial_degree_r,
    #         polynomial_degree_eta,
    #         polynomial_degree_zeta,
    #     )

    #     # Build Bezier patches from B-spline control points
    #     from cardillo.discretization.B_spline import decompose_B_spline_volume

    #     Qw = decompose_B_spline_volume(knot_vector_objs, Pw)

    #     nbezier_xi, nbezier_eta, nbezier_zeta, p1, q1, r1, dim = Qw.shape

    #     # build vtk mesh
    #     n_patches = nbezier_xi * nbezier_eta * nbezier_zeta
    #     patch_size = p1 * q1 * r1
    #     points = np.zeros((n_patches * patch_size, dim))
    #     cells = []
    #     HigherOrderDegrees = []
    #     RationalWeights = []
    #     vtk_cell_type = "VTK_BEZIER_HEXAHEDRON"
    #     from PyPanto.miscellaneous.indexing import flat3D, rearange_vtk3D

    #     for i in range(nbezier_xi):
    #         for j in range(nbezier_eta):
    #             for k in range(nbezier_zeta):
    #                 idx = flat3D(i, j, k, (nbezier_xi, nbezier_eta))
    #                 point_range = np.arange(idx * patch_size, (idx + 1) * patch_size)
    #                 points[point_range] = rearange_vtk3D(Qw[i, j, k])

    #                 cells.append((vtk_cell_type, point_range[None]))
    #                 HigherOrderDegrees.append(np.array(degrees, dtype=float)[None])
    #                 weight = np.sqrt(2) / 2
    #                 # tmp = np.array([np.sqrt(2) / 2, 1.0])
    #                 # RationalWeights.append(np.tile(tmp, 8)[None])
    #                 weights_vertices = weight * np.ones(8)
    #                 weights_edges = np.ones(4 * nn_xi)
    #                 weights_faces = np.ones(2)
    #                 weights_volume = np.ones(nn_xi - 2)
    #                 weights = np.concatenate(
    #                     (weights_edges, weights_vertices, weights_faces, weights_volume)
    #                 )
    #                 # weights = np.array([weight, weight, weight, weight,
    #                 #                     1.0,    1.0,    1.0,    1.0,
    #                 #                     0.0,    0.0,    0.0,    0.0 ])
    #                 weights = np.ones_like(point_range)
    #                 RationalWeights.append(weights[None])

    #     # RationalWeights = np.ones(len(points))
    #     RationalWeights = 2 * (np.random.rand(len(points)) + 1)

    #     # write vtk mesh using meshio
    #     meshio.write_points_cells(
    #         # filename.parent / (filename.stem + '.vtu'),
    #         filename,
    #         points,
    #         cells,
    #         point_data={
    #             "RationalWeights": RationalWeights,
    #         },
    #         cell_data={"HigherOrderDegrees": HigherOrderDegrees},
    #         binary=binary,
    #     )

    # def post_processing(self, t, q, filename, binary=True):
    #     # write paraview PVD file collecting time and all vtk files, see https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
    #     from xml.dom import minidom

    #     root = minidom.Document()

    #     vkt_file = root.createElement("VTKFile")
    #     vkt_file.setAttribute("type", "Collection")
    #     root.appendChild(vkt_file)

    #     collection = root.createElement("Collection")
    #     vkt_file.appendChild(collection)

    #     for i, (ti, qi) in enumerate(zip(t, q)):
    #         filei = filename + f"{i}.vtu"

    #         # write time step and file name in pvd file
    #         dataset = root.createElement("DataSet")
    #         dataset.setAttribute("timestep", f"{ti:0.6f}")
    #         dataset.setAttribute("file", filei)
    #         collection.appendChild(dataset)

    #         self.post_processing_single_configuration(ti, qi, filei, binary=binary)

    #     # write pvd file
    #     xml_str = root.toprettyxml(indent="\t")
    #     with open(filename + ".pvd", "w") as f:
    #         f.write(xml_str)

    # def post_processing_single_configuration(self, t, q, filename, binary=True):
    #     # centerline and connectivity
    #     cells_r, points_r, HigherOrderDegrees_r = self.mesh_r.vtk_mesh(q[: self.nq_r])

    #     # if the centerline and the directors are interpolated with the same
    #     # polynomial degree we can use the values on the nodes and decompose the B-spline
    #     # into multiple Bezier patches, otherwise the directors have to be interpolated
    #     # onto the nodes of the centerline by a so-called L2-projection, see below
    #     same_shape_functions = False
    #     if self.polynomial_degree_r == self.polynomial_degree_phi:
    #         same_shape_functions = True

    #     if same_shape_functions:
    #         _, points_di, _ = self.mesh_phi.vtk_mesh(q[self.nq_r :])

    #         # fill dictionary storing point data with directors
    #         point_data = {
    #             "d1": points_di[:, 0:3],
    #             "d2": points_di[:, 3:6],
    #             "d3": points_di[:, 6:9],
    #         }

    #     else:
    #         point_data = {}

    #     # export existing values on quadrature points using L2 projection
    #     J0_vtk = self.mesh_r.field_to_vtk(self.J0.reshape(self.nEl, self.nQP, 1))
    #     point_data.update({"J0": J0_vtk})

    #     Gamma0_vtk = self.mesh_r.field_to_vtk(self.lambda0)
    #     point_data.update({"Gamma0": Gamma0_vtk})

    #     Kappa0_vtk = self.mesh_r.field_to_vtk(self.Kappa0)
    #     point_data.update({"Kappa0": Kappa0_vtk})

    #     # evaluate fields at quadrature points that have to be projected onto the centerline mesh:
    #     # - strain measures Gamma & Kappa
    #     # - directors d1, d2, d3
    #     Gamma = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #     Kappa = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #     if not same_shape_functions:
    #         d1s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #         d2s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #         d3s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #     for el in range(self.nEl):
    #         qe = q[self.elDOF[el]]

    #         # extract generalized coordinates for beam centerline and directors
    #         # in the current and reference configuration
    #         qe_r = qe
    #         qe_d1 = qe[self.phiDOF]
    #         qe_d2 = qe[self.d2DOF]
    #         qe_d3 = qe[self.d3DOF]

    #         for i in range(self.nQP):
    #             # build matrix of shape function derivatives
    #             NN_di_i = self.stack3di(self.N_phi[el, i])
    #             NN_r_xii = self.stack3r(self.N_r_xi[el, i])
    #             NN_di_xii = self.stack3di(self.N_phi_xi[el, i])

    #             # extract reference state variables
    #             J0i = self.J0[el, i]
    #             Gamma0_i = self.lambda0[el, i]
    #             Kappa0_i = self.Kappa0[el, i]

    #             # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
    #             # the parameter space \xi and thus need to be transformed later
    #             r_xi = NN_r_xii @ qe_r

    #             d1 = NN_di_i @ qe_d1
    #             d1_xi = NN_di_xii @ qe_d1

    #             d2 = NN_di_i @ qe_d2
    #             d2_xi = NN_di_xii @ qe_d2

    #             d3 = NN_di_i @ qe_d3
    #             d3_xi = NN_di_xii @ qe_d3

    #             # compute derivatives w.r.t. the arc lenght parameter s
    #             r_s = r_xi / J0i

    #             d1_s = d1_xi / J0i
    #             d2_s = d2_xi / J0i
    #             d3_s = d3_xi / J0i

    #             # build rotation matrices
    #             if not same_shape_functions:
    #                 d1s[el, i] = d1
    #                 d2s[el, i] = d2
    #                 d3s[el, i] = d3
    #             R = np.vstack((d1, d2, d3)).T

    #             # axial and shear strains
    #             Gamma[el, i] = R.T @ r_s

    #             # torsional and flexural strains
    #             Kappa[el, i] = np.array(
    #                 [
    #                     0.5 * (d3 @ d2_s - d2 @ d3_s),
    #                     0.5 * (d1 @ d3_s - d3 @ d1_s),
    #                     0.5 * (d2 @ d1_s - d1 @ d2_s),
    #                 ]
    #             )

    #     # L2 projection of strain measures
    #     Gamma_vtk = self.mesh_r.field_to_vtk(Gamma)
    #     point_data.update({"Gamma": Gamma_vtk})

    #     Kappa_vtk = self.mesh_r.field_to_vtk(Kappa)
    #     point_data.update({"Kappa": Kappa_vtk})

    #     # L2 projection of directors
    #     if not same_shape_functions:
    #         d1_vtk = self.mesh_r.field_to_vtk(d1s)
    #         point_data.update({"d1": d1_vtk})
    #         d2_vtk = self.mesh_r.field_to_vtk(d2s)
    #         point_data.update({"d2": d2_vtk})
    #         d3_vtk = self.mesh_r.field_to_vtk(d3s)
    #         point_data.update({"d3": d3_vtk})

    #     # fields depending on strain measures and other previously computed quantities
    #     point_data_fields = {
    #         "W": lambda Gamma, Gamma0, Kappa, Kappa0: np.array(
    #             [self.material_model.potential(Gamma, Gamma0, Kappa, Kappa0)]
    #         ),
    #         "n_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.n_i(
    #             Gamma, Gamma0, Kappa, Kappa0
    #         ),
    #         "m_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.m_i(
    #             Gamma, Gamma0, Kappa, Kappa0
    #         ),
    #     }

    #     for name, fun in point_data_fields.items():
    #         tmp = fun(Gamma_vtk[0], Gamma0_vtk[0], Kappa_vtk[0], Kappa0_vtk[0]).reshape(
    #             -1
    #         )
    #         field = np.zeros((len(Gamma_vtk), len(tmp)))
    #         for i, (Gamma_i, Gamma0_i, Kappa_i, Kappa0_i) in enumerate(
    #             zip(Gamma_vtk, Gamma0_vtk, Kappa_vtk, Kappa0_vtk)
    #         ):
    #             field[i] = fun(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i).reshape(-1)
    #         point_data.update({name: field})

    #     # write vtk mesh using meshio
    #     meshio.write_points_cells(
    #         os.path.splitext(os.path.basename(filename))[0] + ".vtu",
    #         points_r,  # only export centerline as geometry here!
    #         cells_r,
    #         point_data=point_data,
    #         cell_data={"HigherOrderDegrees": HigherOrderDegrees_r},
    #         binary=binary,
    #     )

    # def post_processing_subsystem(self, t, q, u, binary=True):
    #     # centerline and connectivity
    #     cells_r, points_r, HigherOrderDegrees_r = self.mesh_r.vtk_mesh(q[: self.nq_r])

    #     # if the centerline and the directors are interpolated with the same
    #     # polynomial degree we can use the values on the nodes and decompose the B-spline
    #     # into multiple Bezier patches, otherwise the directors have to be interpolated
    #     # onto the nodes of the centerline by a so-called L2-projection, see below
    #     same_shape_functions = False
    #     if self.polynomial_degree_r == self.polynomial_degree_phi:
    #         same_shape_functions = True

    #     if same_shape_functions:
    #         _, points_di, _ = self.mesh_phi.vtk_mesh(q[self.nq_r :])

    #         # fill dictionary storing point data with directors
    #         point_data = {
    #             "u_r": points_r - points_r[0],
    #             "d1": points_di[:, 0:3],
    #             "d2": points_di[:, 3:6],
    #             "d3": points_di[:, 6:9],
    #         }

    #     else:
    #         point_data = {}

    #     # export existing values on quadrature points using L2 projection
    #     J0_vtk = self.mesh_r.field_to_vtk(self.J0.reshape(self.nEl, self.nQP, 1))
    #     point_data.update({"J0": J0_vtk})

    #     Gamma0_vtk = self.mesh_r.field_to_vtk(self.lambda0)
    #     point_data.update({"Gamma0": Gamma0_vtk})

    #     Kappa0_vtk = self.mesh_r.field_to_vtk(self.Kappa0)
    #     point_data.update({"Kappa0": Kappa0_vtk})

    #     # evaluate fields at quadrature points that have to be projected onto the centerline mesh:
    #     # - strain measures Gamma & Kappa
    #     # - directors d1, d2, d3
    #     Gamma = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #     Kappa = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #     if not same_shape_functions:
    #         d1s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #         d2s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #         d3s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #     for el in range(self.nEl):
    #         qe = q[self.elDOF[el]]

    #         # extract generalized coordinates for beam centerline and directors
    #         # in the current and reference configuration
    #         qe_r = qe
    #         qe_d1 = qe[self.phiDOF]
    #         qe_d2 = qe[self.d2DOF]
    #         qe_d3 = qe[self.d3DOF]

    #         for i in range(self.nQP):
    #             # build matrix of shape function derivatives
    #             NN_di_i = self.stack3di(self.N_phi[el, i])
    #             NN_r_xii = self.stack3r(self.N_r_xi[el, i])
    #             NN_di_xii = self.stack3di(self.N_phi_xi[el, i])

    #             # extract reference state variables
    #             J0i = self.J0[el, i]
    #             Gamma0_i = self.lambda0[el, i]
    #             Kappa0_i = self.Kappa0[el, i]

    #             # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
    #             # the parameter space \xi and thus need to be transformed later
    #             r_xi = NN_r_xii @ qe_r

    #             d1 = NN_di_i @ qe_d1
    #             d1_xi = NN_di_xii @ qe_d1

    #             d2 = NN_di_i @ qe_d2
    #             d2_xi = NN_di_xii @ qe_d2

    #             d3 = NN_di_i @ qe_d3
    #             d3_xi = NN_di_xii @ qe_d3

    #             # compute derivatives w.r.t. the arc lenght parameter s
    #             r_s = r_xi / J0i

    #             d1_s = d1_xi / J0i
    #             d2_s = d2_xi / J0i
    #             d3_s = d3_xi / J0i

    #             # build rotation matrices
    #             if not same_shape_functions:
    #                 d1s[el, i] = d1
    #                 d2s[el, i] = d2
    #                 d3s[el, i] = d3
    #             R = np.vstack((d1, d2, d3)).T

    #             # axial and shear strains
    #             Gamma[el, i] = R.T @ r_s

    #             # torsional and flexural strains
    #             Kappa[el, i] = np.array(
    #                 [
    #                     0.5 * (d3 @ d2_s - d2 @ d3_s),
    #                     0.5 * (d1 @ d3_s - d3 @ d1_s),
    #                     0.5 * (d2 @ d1_s - d1 @ d2_s),
    #                 ]
    #             )

    #     # L2 projection of strain measures
    #     Gamma_vtk = self.mesh_r.field_to_vtk(Gamma)
    #     point_data.update({"Gamma": Gamma_vtk})

    #     Kappa_vtk = self.mesh_r.field_to_vtk(Kappa)
    #     point_data.update({"Kappa": Kappa_vtk})

    #     # L2 projection of directors
    #     if not same_shape_functions:
    #         d1_vtk = self.mesh_r.field_to_vtk(d1s)
    #         point_data.update({"d1": d1_vtk})
    #         d2_vtk = self.mesh_r.field_to_vtk(d2s)
    #         point_data.update({"d2": d2_vtk})
    #         d3_vtk = self.mesh_r.field_to_vtk(d3s)
    #         point_data.update({"d3": d3_vtk})

    #     # fields depending on strain measures and other previously computed quantities
    #     point_data_fields = {
    #         "W": lambda Gamma, Gamma0, Kappa, Kappa0: np.array(
    #             [self.material_model.potential(Gamma, Gamma0, Kappa, Kappa0)]
    #         ),
    #         "n_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.n_i(
    #             Gamma, Gamma0, Kappa, Kappa0
    #         ),
    #         "m_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.m_i(
    #             Gamma, Gamma0, Kappa, Kappa0
    #         ),
    #     }

    #     for name, fun in point_data_fields.items():
    #         tmp = fun(Gamma_vtk[0], Gamma0_vtk[0], Kappa_vtk[0], Kappa0_vtk[0]).reshape(
    #             -1
    #         )
    #         field = np.zeros((len(Gamma_vtk), len(tmp)))
    #         for i, (Gamma_i, Gamma0_i, Kappa_i, Kappa0_i) in enumerate(
    #             zip(Gamma_vtk, Gamma0_vtk, Kappa_vtk, Kappa0_vtk)
    #         ):
    #             field[i] = fun(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i).reshape(-1)
    #         point_data.update({name: field})

    #     return points_r, point_data, cells_r, HigherOrderDegrees_r
