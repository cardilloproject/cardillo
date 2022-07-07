import numpy as np

from cardillo.utility.coo import Coo
from cardillo.discretization.lagrange import Node_vector
from cardillo.discretization.mesh1D import Mesh1D
from cardillo.math import (
    norm,
    approx_fprime,
)


class QuadraticPotential:
    def __init__(self, k):
        self.k = k

    def pot(self, g, g0):
        return 0.5 * self.k * (g - g0) ** 2

    def pot_g(self, g, g0):
        return self.k * (g - g0)

    def pot_gg(self, g, g0):
        return self.k


class Rope:
    def __init__(
        self,
        k_e,
        polynomial_degree,
        A_rho0,
        nelement,
        Q,
        q0=None,
        u0=None,
    ):
        # beam properties
        self.material_model = QuadraticPotential(k_e)  # material model
        self.A_rho0 = A_rho0  # line density

        # discretization parameters
        self.polynomial_degree = polynomial_degree
        self.nquadrature = nquadrature = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
        self.nelement = nelement  # number of elements

        # TODO: Generalize for arbitrary shape functions
        self.knot_vector = Node_vector(self.polynomial_degree, nelement)

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
        for el in range(nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(nquadrature):
                # interpolate tangent vector
                r_xi = np.zeros(3, dtype=float)
                for node in range(self.nnodes_element):
                    r_xi += self.N_xi[el, i, node] * qe[self.nodalDOF_element[node]]

                # length of reference tangential vector
                self.J[el, i] = norm(r_xi)

    @staticmethod
    def straight_configuration(
        polynomial_degree,
        nelement,
        L,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
    ):
        nn = polynomial_degree * nelement + 1

        x0 = np.linspace(0, L, num=nn)
        y0 = np.zeros(nn, dtype=float)
        z0 = np.zeros(nn, dtype=float)

        r0 = np.vstack((x0, y0, z0))
        for i in range(nn):
            r0[:, i] = r_OP + A_IK @ r0[:, i]

        # reshape generalized coordinates to nodal ordering
        q = r0.reshape(-1, order="F")

        return q

    def element_number(self, xi):
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
        self.__M = Coo((self.nu, self.nu))
        for el in range(self.nelement):
            # extract element degrees of freedom
            elDOF = self.elDOF[el]

            # sparse assemble element mass matrix
            self.__M.extend(self.M_el(el), (self.uDOF[elDOF], self.uDOF[elDOF]))

    def M(self, t, q, coo):
        coo.extend_sparse(self.__M)

    def E_pot(self, t, q):
        E_pot = 0
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            E_pot += self.E_pot_el(q[elDOF], el)
        return E_pot

    def E_pot_el(self, qe, el):
        raise NotImplementedError

    def f_pot(self, t, q):
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

            # interpolate tangent vector
            r_xi = np.zeros(3, dtype=qe.dtype)
            for node in range(self.nnodes_element):
                r_xi += self.N_xi[el, i, node] * qe[self.nodalDOF_element[node]]

            # length of the current tangent vector
            ji = norm(r_xi)

            # axial strain
            la = ji / Ji
            la0 = 1.0

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            n = self.material_model.pot_g(la, la0)

            # unit tangent vector
            e = r_xi / ji

            ############################
            # virtual work contributions
            ############################
            for node in range(self.nnodes_element):
                f_pot_el[self.nodalDOF_element[node]] -= (
                    self.N_xi[el, i, node] * e * n * qwi
                )

        return f_pot_el

    def f_pot_q(self, t, q, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f_pot_q_el = self.f_pot_q_el(q[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(f_pot_q_el, (self.uDOF[elDOF], self.qDOF[elDOF]))

    # TODO:
    def f_pot_q_el(self, qe, el):
        return approx_fprime(
            qe, lambda qe: self.f_pot_el(qe, el), eps=1.0e-10, method="cs"
        )

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

    ###################
    # r_OP contribution
    ###################
    def r_OP(self, t, q, frame_ID, K_r_SP=None):
        # evaluate shape functions
        N, _ = self.basis_functions(frame_ID[0])

        # interpolate tangent vector
        r_OP = np.zeros(3, dtype=q.dtype)
        for node in range(self.nnodes_element):
            r_OP += N[node] * q[self.nodalDOF_element[node]]
        return r_OP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=None):
        # evaluate shape functions
        N, _ = self.basis_functions(frame_ID[0])

        # interpolate tangent vector
        r_OP_q = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element):
            r_OP_q[:, self.nodalDOF_element[node]] += N[node] * np.eye(3, dtype=float)
        return r_OP_q

    def v_P(self, t, q, u, frame_ID, K_r_SP=None):
        # evaluate shape functions
        N, _ = self.basis_functions(frame_ID[0])

        # interpolate tangent vector
        v_P = np.zeros(3, dtype=q.dtype)
        for node in range(self.nnodes_element):
            v_P += N[node] * u[self.nodalDOF_element[node]]
        return v_P

    def v_P_q(self, t, q, u, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_element), dtype=float)

    def J_P(self, t, q, frame_ID, K_r_SP=None):
        # evaluate shape functions
        N, _ = self.basis_functions(frame_ID[0])

        # interpolate tangent vector
        J_P = np.zeros((3, self.nu_element), dtype=float)
        for node in range(self.nnodes_element):
            J_P[:, self.nodalDOF_element[node]] += N[node] * np.eye(3, dtype=float)
        return J_P

    def J_P_q(self, t, q, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nu_element, self.nq_element), dtype=float)

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        # evaluate shape functions
        N, _ = self.basis_functions(frame_ID[0])

        # interpolate tangent vector
        a_P = np.zeros(3, dtype=q.dtype)
        for node in range(self.nnodes_element):
            a_P += N[node] * u_dot[self.nodalDOF_element[node]]
        return a_P

    def a_P_q(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_element), dtype=float)

    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nu_element), dtype=float)

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
            r = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element):
                r += self.N[el, i, node] * qe[self.nodalDOF_element[node]]

            # compute potential value at given quadrature point
            Ve += (r @ force(t, qwi)) * Ji * qwi

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
                fe[self.nodalDOF_element[node]] += self.N[el, i, node] * fe_r

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
        return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF]).T

    def centerline(self, q, n=100):
        q_body = q[self.qDOF]
        r = []
        for xi in np.linspace(0, 1, n):
            frame_ID = (xi,)
            qe = q_body[self.qDOF_P(frame_ID)]
            r.append(self.r_OP(1, qe, frame_ID))
        return np.array(r).T

    def plot_centerline(self, ax, q, n=100, color="black"):
        ax.plot(*self.nodes(q), linestyle="dashed", marker="o", color=color)
        ax.plot(*self.centerline(q, n=n), linestyle="solid", color=color)
