import numpy as np

from cardillo.utility.coo_matrix import CooMatrix
from cardillo.discretization.b_spline import BSplineKnotVector
from cardillo.discretization.hermite import HermiteNodeVector
from cardillo.math import (
    e1,
    norm,
    cross3,
    approx_fprime,
)
from cardillo.discretization.mesh1D import Mesh1D


class QuadraticMaterial:
    def __init__(self, k_e, k_b):
        self.k_e = k_e  # axial stiffness
        self.k_b = k_b  # bending stiffness
        self.C_b = np.diag([k_b, k_b, k_b])

    def potential(self, t, xi, la, la0, kappa, kappa0):
        dK = kappa - kappa0
        return 0.5 * self.k_e * (la - la0) ** 2 + 0.5 * dK @ self.C_b @ dK

    def n(self, t, xi, la, la0, kappa, kappa0):
        return self.k_e * (la - la0)

    def m(self, t, xi, la, la0, kappa, kappa0):
        dK = kappa - kappa0
        return self.C_b @ dK

    def n_lambda(self, t, xi, la, la0, kappa, kappa0):
        return self.k_e

    def n_kappa(self, t, xi, la, la0, kappa, kappa0):
        return np.zeros(3, dtype=float)

    def m_lambda(self, t, xi, la, la0, kappa, kappa0):
        return np.zeros(3, dtype=float)

    def m_kappa(self, t, xi, la, la0, kappa, kappa0):
        return self.C_b


class QuadraticMaterialDegraded:
    def __init__(self, k_e, k_b):
        self.k_e = k_e  # axial stiffness
        self.k_b = k_b  # bending stiffness
        self.C_b = np.diag([k_b, k_b, k_b])

    def potential(self, t, xi, la, la0, kappa, kappa0):
        f = 1.0 - t
        dK = kappa - kappa0
        return 0.5 * self.k_e * (la - la0) ** 2 + f * 0.5 * dK @ self.C_b @ dK

    def n(self, t, xi, la, la0, kappa, kappa0):
        return self.k_e * (la - la0)

    def m(self, t, xi, la, la0, kappa, kappa0):
        f = 1.0 - t
        dK = kappa - kappa0
        return f * self.C_b @ dK

    def n_lambda(self, t, xi, la, la0, kappa, kappa0):
        return self.k_e

    def n_kappa(self, t, xi, la, la0, kappa, kappa0):
        return np.zeros(3, dtype=float)

    def m_lambda(self, t, xi, la, la0, kappa, kappa0):
        return np.zeros(3, dtype=float)

    def m_kappa(self, t, xi, la, la0, kappa, kappa0):
        f = 1.0 - t
        return f * self.C_b


class Cable:
    def __init__(
        self,
        material_model,
        A_rho0,
        polynomial_degree,
        nelement,
        Q,
        q0=None,
        u0=None,
        basis="B-spline",
    ):
        # beam properties
        self.material_model = material_model
        self.A_rho0 = A_rho0  # line density

        # discretization parameters
        self.polynomial_degree = polynomial_degree
        self.nquadrature = nquadrature = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
        self.nelement = nelement  # number of elements

        # chose basis
        self.basis = basis
        if basis == "B-spline":
            assert polynomial_degree >= 2, "use at least quadratic B-splines"
            self.knot_vector = BSplineKnotVector(polynomial_degree, nelement)
        elif basis == "Hermite":
            assert polynomial_degree == 3, "only cubic Hermite splines are implemented!"
            self.knot_vector = HermiteNodeVector(polynomial_degree, nelement)
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        # build mesh object
        self.mesh = Mesh1D(
            self.knot_vector,
            nquadrature,
            derivative_order=2,
            basis=basis,
            dim_q=3,
        )

        # total number of nodes
        self.nnode = self.mesh.nnodes

        # number of nodes per element
        self.nnodes_element = self.mesh.nnodes_per_element

        # total number of generalized coordinates and velocities
        self.nq = self.nu = self.mesh.nq

        # number of generalized coordiantes and velocities per element
        self.nq_element = self.nu_element = self.mesh.nq_per_element

        # global element connectivity
        self.elDOF = self.mesh.elDOF

        # global nodal
        self.nodalDOF = self.mesh.nodalDOF

        # nodal connectivity on element level
        self.nodalDOF_element = self.mesh.nodalDOF_element

        # shape functions and their first derivatives
        self.N = self.mesh.N
        self.N_xi = self.mesh.N_xi
        self.N_xixi = self.mesh.N_xixi

        # quadrature points
        self.qp = self.mesh.qp  # quadrature points
        self.qw = self.mesh.wp  # quadrature weights

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions = self.mesh.eval_basis

        # precompute values of the reference configuration in order to save computation time
        # J in Harsch2020b (5)
        self.J = np.zeros((nelement, nquadrature), dtype=float)
        self.kappa0 = np.zeros((nelement, nquadrature, 3), dtype=float)
        for el in range(nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(nquadrature):
                # interpolate tangent vector
                r_xi = np.zeros(3, dtype=float)
                r_xixi = np.zeros(3, dtype=float)
                for node in range(self.nnodes_element):
                    r_xi += self.N_xi[el, i, node] * qe[self.nodalDOF_element[node]]
                    r_xixi += self.N_xixi[el, i, node] * qe[self.nodalDOF_element[node]]

                # length of reference tangential vector
                self.J[el, i] = ji = norm(r_xi)

                # normalized tangent vector
                d1 = r_xi / ji

                # # first directors derivative
                # d1_s = (np.eye(3, dtype=float) - np.outer(d1, d1)) @ r_xixi / (ji * ji)

                # # curvature
                # self.kappa0[el, i] = cross3(d1, d1_s)
                self.kappa0[el, i] = cross3(d1, r_xixi) / (ji * ji)

    @staticmethod
    def straight_configuration(
        basis,
        polynomial_degree,
        nelement,
        L,
        r_OP0=np.zeros(3, dtype=float),
        A_IK0=np.eye(3, dtype=float),
    ):
        if basis == "B-spline":
            nn = polynomial_degree + nelement
        elif basis == "Hermite":
            assert polynomial_degree == 3, "only cubic Hermite splines are implemented!"
            nn = nelement + 1
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        if basis == "B-spline":
            x0 = np.linspace(0, L, num=nn)
            y0 = np.zeros(nn)
            z0 = np.zeros(nn)
            # build Greville abscissae for B-spline basis
            kv = BSplineKnotVector.uniform(polynomial_degree, nelement)
            for i in range(nn):
                x0[i] = np.sum(kv[i + 1 : i + polynomial_degree + 1])
            x0 = x0 * L / polynomial_degree

            q = np.vstack((x0, y0, z0))
            for i in range(nn):
                q[:, i] = r_OP0 + A_IK0 @ q[:, i]

        elif basis == "Hermite":
            xis = np.linspace(0, 1, num=nn)
            q = np.zeros((3, 2 * nn))
            t0 = A_IK0 @ (L * e1)
            for i, xi in enumerate(xis):
                ri = r_OP0 + xi * t0
                q[:, 2 * i] = ri
                q[:, 2 * i + 1] = t0

        # reshape generalized coordinates to nodal ordering
        q = q.reshape(-1, order="C")

        return q

    @staticmethod
    def circular_segment_configuration(
        basis,
        polynomial_degree,
        nelement,
        R,
        phi,
    ):
        if basis == "B-spline":
            print(f"circular_segment_configuration is not correct for B-spline basis!")
            nn = polynomial_degree + nelement
        # elif basis == "Hermite":
        #     assert polynomial_degree == 3, "only cubic Hermite splines are implemented!"
        #     nn = nelement + 1
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        r0 = np.zeros((3, nn), dtype=float)
        for i in range(nn):
            xi = i / (nn - 1)
            phi_i = phi * (2 * xi - 1)
            r0[0, i] = R * np.sin(phi_i)
            r0[1, i] = R * np.cos(phi_i) - R * np.cos(phi)

        # reshape generalized coordinates to nodal ordering
        q = r0.reshape(-1, order="C")
        return q

    def element_number(self, xi):
        # note the elements coincide for both meshes!
        return self.knot_vector.element_number(xi)[0]

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
            __M_el = np.eye(3) * self.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element):
                nodalDOF_a = self.nodalDOF_element[node_a]
                for node_b in range(self.nnodes_element):
                    nodalDOF_b = self.nodalDOF_element[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += __M_el * (
                        self.N[el, i, node_a] * self.N[el, i, node_b]
                    )

        return M_el

    def __M_coo(self):
        self.__M = CooMatrix((self.nu, self.nu))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            self.__M[elDOF, elDOF] = self.M_el(el)

    def M(self, t, q):
        return self.__M

    def E_pot(self, t, q):
        E = 0
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            E += self.E_pot_el(t, q[elDOF], el)
        return E

    def E_pot_el(self, t, qe, el):
        E = np.zeros(1, dtype=qe.dtype)[0]
        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            kappa0 = self.kappa0[el, i]

            # interpolate tangent vector
            r_xi = np.zeros(3, dtype=qe.dtype)
            r_xixi = np.zeros(3, dtype=qe.dtype)
            for node in range(self.nnodes_element):
                r_xi += self.N_xi[el, i, node] * qe[self.nodalDOF_element[node]]
                r_xixi += self.N_xixi[el, i, node] * qe[self.nodalDOF_element[node]]

            # length of the current tangent vector
            ji2 = r_xi @ r_xi
            ji = np.sqrt(ji2)

            # axial strain
            la = ji / Ji
            la0 = 1.0

            # curvature
            kappa_bar = cross3(r_xi, r_xixi) / ji2
            kappa = kappa_bar / Ji

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            E += (
                self.material_model.potential(t, qpi, la, la0, kappa, kappa0) * Ji * qwi
            )

        return E

    def h(self, t, q, u):
        f = np.zeros(self.nu, dtype=q.dtype)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f[elDOF] += self.f_pot_el(t, q[elDOF], el)
        return f

    def f_pot_el(self, t, qe, el):
        f_pot_el = np.zeros(self.nq_element, dtype=qe.dtype)

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]
            kappa0 = self.kappa0[el, i]

            # interpolate tangent vector
            r_xi = np.zeros(3, dtype=qe.dtype)
            r_xixi = np.zeros(3, dtype=qe.dtype)
            for node in range(self.nnodes_element):
                r_xi += self.N_xi[el, i, node] * qe[self.nodalDOF_element[node]]
                r_xixi += self.N_xixi[el, i, node] * qe[self.nodalDOF_element[node]]

            # length of the current tangent vector
            ji2 = r_xi @ r_xi
            ji = np.sqrt(ji2)

            # normalized tangent vector
            d1 = r_xi / ji

            # axial strain
            la = ji / Ji
            la0 = 1.0

            # curvature
            kappa_bar = cross3(r_xi, r_xixi) / ji2
            kappa = kappa_bar / Ji

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            n = self.material_model.n(t, qpi, la, la0, kappa, kappa0)
            m = self.material_model.m(t, qpi, la, la0, kappa, kappa0)

            # assemble internal forces
            for node in range(self.nnodes_element):
                # axial strain
                f_pot_el[self.nodalDOF_element[node]] -= (
                    self.N_xi[el, i, node] * d1 * n * qwi
                )

                # cruvature part 1
                f_pot_el[self.nodalDOF_element[node]] -= (
                    self.N_xi[el, i, node]
                    * (cross3(r_xixi, m) / ji2 - 2.0 * r_xi * (kappa_bar @ m) / ji2)
                    * qwi
                )

                # cruvature part 2
                f_pot_el[self.nodalDOF_element[node]] += (
                    self.N_xixi[el, i, node] * cross3(r_xi, m) / ji2 * qwi
                )

        return f_pot_el

        # f_pot_el_num = -approx_fprime(
        #     qe, lambda qe: self.E_pot_el(t, qe, el), method="3-point"
        # )
        # diff = f_pot_el - f_pot_el_num
        # error = np.linalg.norm(diff)
        # print(f"error f_pot_el: {error}")

        # return f_pot_el_num

    def h_q(self, t, q, u):
        coo = CooMatrix((self.nu, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            coo[elDOF, elDOF] = self.f_pot_q_el(t, q[elDOF], el)

        return coo

    def f_pot_q_el(self, t, qe, el):
        # f_pot_q_el = np.zeros((self.nu_element, self.nq_element), dtype=float)

        # for i in range(self.nquadrature):
        #     # extract reference state variables
        #     qwi = self.qw[el, i]
        #     Ji = self.J[el, i]

        #     # interpolate tangent vector
        #     r_xi = np.zeros(3, dtype=qe.dtype)
        #     r_xi_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
        #     for node_A in range(self.nnodes_element):
        #         r_xi += self.N_xi[el, i, node_A] * qe[self.nodalDOF_element[node_A]]
        #         r_xi_qe[:, self.nodalDOF_element[node_A]] += self.N_xi[
        #             el, i, node_A
        #         ] * np.eye(3, dtype=qe.dtype)

        #     # length of the current tangent vector
        #     ji = norm(r_xi)

        #     # axial strain
        #     la = ji / Ji
        #     la0 = 1.0

        #     # compute contact forces and couples from partial derivatives of
        #     # the strain energy function w.r.t. strain measures
        #     n = self.material_model.pot_g(la, la0)
        #     n_la = self.material_model.pot_gg(la, la0)

        #     # unit tangent vector
        #     e = r_xi / ji

        #     # assemble internal stiffness
        #     Ke = (
        #         (
        #             (n / ji) * np.eye(3, dtype=float)
        #             + (n_la / Ji - n / ji) * np.outer(e, e)
        #         )
        #         @ r_xi_qe
        #         * qwi
        #     )
        #     for node_A in range(self.nnodes_element):
        #         f_pot_q_el[self.nodalDOF_element[node_A]] -= (
        #             self.N_xi[el, i, node_A] * Ke
        #         )
        # return f_pot_q_el

        f_pot_q_el_num = approx_fprime(
            # qe, lambda qe: self.f_pot_el(t, qe, el), eps=1.0e-12, method="cs"
            qe,
            lambda qe: self.f_pot_el(t, qe, el),
            eps=1.0e-6,
            method="2-point",
        )
        # diff = f_pot_q_el - f_pot_q_el_num
        # error = np.linalg.norm(diff)
        # print(f"error f_pot_q_el: {error}")
        return f_pot_q_el_num

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

    ###################
    # r_OP contribution
    ###################
    def r_OP(self, t, q, frame_ID, K_r_SP=None):
        # evaluate shape functions
        N, _, _ = self.basis_functions(frame_ID[0])

        # interpolate tangent vector
        r_OP = np.zeros(3, dtype=q.dtype)
        for node in range(self.nnodes_element):
            r_OP += N[node] * q[self.nodalDOF_element[node]]
        return r_OP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=None):
        # evaluate shape functions
        N, _, _ = self.basis_functions(frame_ID[0])

        # interpolate tangent vector
        r_OP_q = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element):
            r_OP_q[:, self.nodalDOF_element[node]] += N[node] * np.eye(3, dtype=float)
        return r_OP_q

    def v_P(self, t, q, u, frame_ID, K_r_SP=None):
        # evaluate shape functions
        N, _, _ = self.basis_functions(frame_ID[0])

        # interpolate tangent vector
        v_P = np.zeros(3, dtype=np.common_type(q, u))
        for node in range(self.nnodes_element):
            v_P += N[node] * u[self.nodalDOF_element[node]]
        return v_P

    def v_P_q(self, t, q, u, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_element), dtype=float)

    def J_P(self, t, q, frame_ID, K_r_SP=None):
        # evaluate shape functions
        N, _, _ = self.basis_functions(frame_ID[0])

        # interpolate Jacobian
        J_P = np.zeros((3, self.nu_element), dtype=float)
        for node in range(self.nnodes_element):
            J_P[:, self.nodalDOF_element[node]] += N[node] * np.eye(3, dtype=float)
        return J_P

    def J_P_q(self, t, q, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nu_element, self.nq_element), dtype=float)

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        # evaluate shape functions
        N, _, _ = self.basis_functions(frame_ID[0])

        # interpolate tangent vector
        a_P = np.zeros(3, dtype=q.dtype)
        for node in range(self.nnodes_element):
            a_P += N[node] * u_dot[self.nodalDOF_element[node]]
        return a_P

    def a_P_q(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_element), dtype=float)

    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nu_element), dtype=float)

    def r_xi(self, t, q, frame_ID):
        # evaluate shape functions
        _, N_xi, _ = self.basis_functions(frame_ID[0])

        # interpolate tangent vector
        r_xi = np.zeros(3, dtype=q.dtype)
        for node in range(self.nnodes_element):
            r_xi += N_xi[node] * q[self.nodalDOF_element[node]]
        return r_xi

    ###########
    # rotations
    ###########
    # def A_IK(self, t, q, frame_ID):
    #     # evaluate shape functions
    #     _, N_xi, _ = self.basis_functions(frame_ID[0])

    #     # interpolate tangent vector
    #     r_xi = np.zeros(3, dtype=q.dtype)
    #     for node in range(self.nnodes_element):
    #         r_xi += N_xi[node] * q[self.nodalDOF_element[node]]

    #     A_IK = np.eye(3, dtype=q.dtype)
    #     A_IK[:, 0] = r_xi / norm(r_xi)
    #     return A_IK

    # def K_Omega(self, t, q, u, frame_ID):
    #     # evaluate shape functions
    #     _, N_xi, _ = self.basis_functions(frame_ID[0])

    #     # interpolate tangent vector
    #     r_xi = np.zeros(3, dtype=np.common_type(q, u))
    #     r_xi_dot = np.zeros(3, dtype=np.common_type(q, u))
    #     for node in range(self.nnodes_element):
    #         r_xi += N_xi[node] * q[self.nodalDOF_element[node]]
    #         r_xi_dot += N_xi[node] * u[self.nodalDOF_element[node]]

    #     j = norm(r_xi)
    #     d1 = r_xi / j
    #     d1_dot = (1 / j) * (np.eye(3) - np.outer(d1, d1)) @ r_xi_dot

    #     return np.array(
    #         [
    #             0.0,
    #             -e3 @ d1_dot,
    #             e2 @ d1_dot,
    #         ]
    #     )

    ####################################################
    # body force
    ####################################################
    def distributed_force1D_pot_el(self, force, t, qe, el):
        Ve = 0
        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # interpolate centerline position
            r = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element):
                r += self.N[el, i, node] * qe[self.nodalDOF_element[node]]

            # compute potential value at given quadrature point
            Ve += (r @ force(t, qpi)) * Ji * qwi

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
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # compute local force vector
            fe_r = force(t, qpi) * Ji * qwi

            # multiply local force vector with variation of centerline
            for node in range(self.nnodes_element):
                fe[self.nodalDOF_element[node]] += self.N[el, i, node] * fe_r

        return fe

    def distributed_force1D(self, t, q, force):
        f = np.zeros(self.nq, dtype=float)
        for el in range(self.nelement):
            f[self.elDOF[el]] += self.distributed_force1D_el(force, t, el)
        return f

    def distributed_force1D_q(self, t, q, force):
        pass

    ####################################################
    # visualization
    ####################################################
    def nodes(self, q):
        q_body = q[self.qDOF]
        if self.basis == "Hermite":
            return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF[::2]]).T
        else:
            return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF]).T

    def centerline(self, q, num=100):
        q_body = q[self.qDOF]
        r = []
        for xi in np.linspace(0, 1, num):
            frame_ID = (xi,)
            qe = q_body[self.local_qDOF_P(frame_ID)]
            r.append(self.r_OP(1, qe, frame_ID))
        return np.array(r).T

    def plot_centerline(self, ax, q, num=100, color="black"):
        ax.plot(*self.nodes(q), linestyle="dashed", marker="o", color=color)
        ax.plot(*self.centerline(q, num=num), linestyle="solid", color=color)

    def stretch(self, q, num=100):
        q_body = q[self.qDOF]
        la = []
        for xi in np.linspace(0, 1, num):
            frame_ID = (xi,)
            qe = q_body[self.local_qDOF_P(frame_ID)]
            Qe = self.Q[self.local_qDOF_P(frame_ID)]
            r_xi = self.r_xi(1, qe, frame_ID)
            r0_xi = self.r_xi(1, Qe, frame_ID)
            la.append(norm(r_xi) / norm(r0_xi))
        return np.array(la).T
