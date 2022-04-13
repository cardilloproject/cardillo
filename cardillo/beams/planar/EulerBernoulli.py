import numpy as np
from math import sqrt
import meshio
import os

from cardillo.utility.coo import Coo
from cardillo.math.algebra import ax2skew, norm, cross3, e3
from cardillo.math import approx_fprime
from cardillo.discretization.B_spline import KnotVector
from cardillo.discretization.mesh1D import Mesh1D


class EulerBernoulli2D:
    """Planar Euler-Bernoulli beam using B-spline shape functions introduced in
    Eugster2020b (https://doi.org/10.1007/978-3-030-50460-1_9) and Harsch2020a
    (https://doi.org/10.1007/978-3-030-50460-1_10).
    """

    def __init__(
        self, A_rho0, material_model, polynomial_degree, nEl, nQP, Q, q0=None, u0=None
    ):
        # line density
        self.A_rho0 = A_rho0

        # material model
        self.material_model = material_model

        # discretization parameters
        self.polynomial_degree = polynomial_degree  # polynomial degree
        self.nQP = nQP  # number of quadrature points
        self.nEl = nEl  # number of elements

        nn = nEl + polynomial_degree  # number of nodes
        nn_el = polynomial_degree + 1  # number of nodes per element
        self.nq_n = nq_n = 2  # number of degrees of freedom per node (x, y)

        # knot vector object and element span
        self.knot_vector = KnotVector(polynomial_degree, nEl)
        self.element_span = self.knot_vector.element_data

        # B-spline mesh object
        self.mesh = Mesh1D(
            self.knot_vector,
            nQP,
            derivative_order=2,
            basis="B-spline",
            dim=self.nq_n,
        )

        # element connectivity matrix
        self.elDOF = self.mesh.elDOF

        # evaluate shape functions at specific xi
        self.basis_functions = self.mesh.eval_basis

        # number of generalized coordinates
        self.nq = self.nu = (
            nn * nq_n
        )  # total number of generalized coordinates and velocities
        self.nq_el = (
            nn_el * nq_n
        )  # number of generalized coordinates and velocities per element

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q  # reference configuration
        self.q0 = Q.copy() if q0 is None else q0  # initial configuration
        self.u0 = np.zeros(self.nu) if u0 is None else u0  # initial velocities

        # compute shape functions
        self.N = self.mesh.N
        self.N_xi = self.mesh.N_xi
        self.N_xixi = self.mesh.N_xixi
        self.qw = self.mesh.wp
        self.xi = self.mesh.qp

        # compute values of the reference configuration in advance to save computation time later
        self.J0 = np.zeros((nEl, nQP))
        self.kappa0 = np.zeros((nEl, nQP))

        for el in range(nEl):
            Qe = self.Q[self.elDOF[el]]
            for i in range(self.nQP):
                # build matrix of shape function and derivatives
                NN_xii = self.stack2(self.N_xi[el, i])
                NN_xixii = self.stack2(self.N_xixi[el, i])

                # tangent vector and its squared norm
                t0 = NN_xii @ Qe
                J02 = t0[0] * t0[0] + t0[1] * t0[1]
                self.J0[el, i] = J0 = sqrt(J02)

                # normal vector
                n0 = NN_xixii @ Qe

                # rotated tangent vector
                t0_perp = np.array([-t0[1], t0[0]])

                # reference curvature
                self.kappa0[el, i] = t0_perp @ n0 / (J0**3)

        # evaluate shape functions on the boundary in davance to save
        # computation time later
        N_bdry0, N_xi_bdry0, _ = self.mesh.NN_bdry0
        NN_bdry0 = self.stack2(N_bdry0)
        NN_xi_bdry0 = self.stack2(N_xi_bdry0)

        N_bdry1, N_xi_bdry1, _ = self.mesh.NN_bdry1
        NN_bdry1 = self.stack2(N_bdry1)
        NN_xi_bdry1 = self.stack2(N_xi_bdry1)

        self.N_bdry = np.array([NN_bdry0, NN_bdry1])
        self.dN_bdry = np.array([NN_xi_bdry0, NN_xi_bdry1])

    @staticmethod
    def straight_configuration(
        polynomial_degree,
        nEl,
        L,
        greville_abscissae=True,
        r_OP=np.zeros(3),
        A_IK=np.eye(3),
    ):

        # number of nodes
        nn = polynomial_degree + nEl

        X = np.linspace(0, L, num=nn)
        Y = np.zeros(nn)
        if greville_abscissae:
            kv = KnotVector.uniform(polynomial_degree, nEl)
            for i in range(nn):
                X[i] = np.sum(kv[i + 1 : i + polynomial_degree + 1])
            X = X * L / polynomial_degree

        r0 = np.vstack((X, Y)).T
        for i in range(nn):
            r0i = np.array([*r0[i], 0])
            X[i], Y[i], _ = r_OP + A_IK @ r0i

        # assemble all reference generalized coordinates
        return np.concatenate([X, Y])

    def assembler_callback(self):
        self.__M_coo()

    def stacked_basis_functions(self, frame_ID):
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
            dNN = self.dN_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[-1]
            dNN = self.dN_bdry[-1]
        else:
            N, dN, _ = self.basis_functions(xi)
            NN = self.stack2(N)
            dNN = self.stack2(dN)
        return NN, dNN

    def stack2(self, N):
        n2 = int(self.nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, :n2] = N
        NN[1, n2:] = N
        return NN

        # this is nice but way to slow in python
        # return np.kron(np.eye(2), N)

    def stack2_perp(self, N):
        n2 = int(self.nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, n2:] = -N
        NN[1, :n2] = N
        return NN

        # this is nice but way to slow in python
        # return np.kron(np.array([[0, -1], [1, 0]]), N)

    #########################################
    # equations of motion
    #########################################
    def M_el(self, N, J0, qw):
        Me = np.zeros((self.nq_el, self.nq_el))

        for Ni, J0i, qwi in zip(N, J0, qw):
            # build matrix of shape functions and derivatives
            NNi = self.stack2(Ni)

            # integrate elemente mass matrix
            # note: we only use the constant part of the mass matrix introduced in
            # Harsch2020a (50) and thus neglect the inertia of the cross-sectional rotations.
            # Also see Eugster2020b (104) and Elishakoff2015 for a discusson of the neglected term.
            Me += NNi.T @ NNi * self.A_rho0 * J0i * qwi

        return Me

    def __M_coo(self):
        self.__M = Coo((self.nu, self.nu))
        for el in range(self.nEl):
            # extract element degrees of freedom
            elDOF = self.elDOF[el]

            # compute element mass matrix
            Me = self.M_el(self.N[el], self.J0[el], self.qw[el])

            # sparse assemble element mass matrix
            self.__M.extend(Me, (self.uDOF[elDOF], self.uDOF[elDOF]))

    def M(self, t, q, coo):
        coo.extend_sparse(self.__M)

    def f_pot_el(self, qe, el):
        fe = np.zeros(self.nq_el)
        for i in range(self.nQP):
            # extract values of the reference configuration at the current quadrature point
            J0i = self.J0[el, i]
            kappa0i = self.kappa0[el, i]

            # build matrix of shape function derivatives
            NN_xii = self.stack2(self.N_xi[el, i])
            NN_xixii = self.stack2(self.N_xixi[el, i])

            # tangent and normal vector
            t = NN_xii @ qe
            n = NN_xixii @ qe

            # norm and squared norm of the tangent vector
            g2_ = t[0] * t[0] + t[1] * t[1]
            g_ = sqrt(g2_)

            # rotated tangent and normal vector
            t_perp = np.array([-t[1], t[0]])
            n_perp = np.array([-n[1], n[0]])

            # change of angle, see Eugster2020b (99) and Harsch2020a (18)
            theta_bar_xi = t_perp @ n / g2_

            # strain measures
            g = g_ / J0i
            kappa = theta_bar_xi / J0i

            # evaluate material model
            N = self.material_model.n(g, kappa, kappa0i)
            M = self.material_model.m(g, kappa, kappa0i)

            # quadrature contribution to element internal force vector
            # see Harsch2020a (51)
            R1 = NN_xii.T @ (t * N / g_ - M / g2_ * (2 * theta_bar_xi * t + n_perp))
            R2 = NN_xixii.T @ t_perp * M / g2_

            # integrate element force vector using specified quadrature rule
            fe -= (R1 + R2) * self.qw[el, i]

        return fe

    def f_pot(self, t, q):
        f = np.zeros(self.nu)
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            f[elDOF] += self.f_pot_el(q[elDOF], el)
        return f

    def f_pot_q_el(self, qe, el):
        fe_q = np.zeros((self.nq_el, self.nq_el))
        for i in range(self.nQP):
            # extract values of the reference configuration at the current quadrature point
            J0i = self.J0[el, i]
            kappa0i = self.kappa0[el, i]
            N_xii = self.N_xi[el, i]
            N_xixii = self.N_xixi[el, i]

            # build matrix of shape function derivatives
            NN_xii = self.stack2(N_xii)
            NN_xixii = self.stack2(N_xixii)

            NN_xii_perp = self.stack2_perp(N_xii)
            NN_xixii_perp = self.stack2_perp(N_xixii)

            # tangent and normal vector
            t = NN_xii @ qe
            n = NN_xixii @ qe

            # norm and squared norm of the tangent vector
            g2_ = t[0] * t[0] + t[1] * t[1]
            g_ = sqrt(g2_)

            # rotated tangent and normal vector
            t_perp = np.array([-t[1], t[0]])
            n_perp = np.array([-n[1], n[0]])

            # change of angle, see Eugster2020b (99) and Harsch2020a (18)
            theta_bar_xi = t_perp @ n / g2_

            # strain measures
            g = g_ / J0i
            kappa = theta_bar_xi / J0i

            # auxiliary functions, see Harsch2020a (53)
            g_bar_q = t @ NN_xii / g_
            theta_bar_xi_q = (
                n @ NN_xii_perp + t_perp @ NN_xixii
            ) / g2_ - 2 * theta_bar_xi / g_ * g_bar_q

            # derivative of strain measures
            g_q = g_bar_q / J0i
            kappa_q = theta_bar_xi_q / J0i

            # evaluate material model
            N = self.material_model.n(g, kappa, kappa0i)
            M = self.material_model.m(g, kappa, kappa0i)

            N_g = self.material_model.n_lambda(g, kappa, kappa0i)
            N_kappa = self.material_model.n_kappa(g, kappa, kappa0i)
            M_g = self.material_model.m_lambda(g, kappa, kappa0i)
            M_kappa = self.material_model.m_kappa(g, kappa, kappa0i)

            N_q = (
                N_g * g_q + N_kappa * kappa_q
            )  # we need the derivatives of g w.r.t. q not those of \overline{g}
            M_q = (
                M_g * g_q + M_kappa * kappa_q
            )  # we need the derivatives of g w.r.t. q not those of \overline{g}

            # auxiliary functions and their derivatives, see Harsch2020a (53)
            k1 = t * N / g_
            k2 = M / g2_
            k3 = 2 * theta_bar_xi * t
            k1_q = np.outer(-k1 / g_, g_bar_q) + (N * NN_xii + np.outer(t, N_q)) / g_
            k2_q = -2 * k2 / g_ * g_bar_q + M_q / g2_
            k3_q = 2 * (np.outer(t, theta_bar_xi_q) + theta_bar_xi * NN_xii)

            # quadrature contribution to element stiffness matrix integrated using specified
            # quadrature rule, see Harsch2020a (52)
            fe_q -= (
                NN_xii.T
                @ (k1_q - np.outer(k3 + n_perp, k2_q) - k2 * (k3_q + NN_xixii_perp))
                + NN_xixii.T @ (k2 * NN_xii_perp + np.outer(t_perp, k2_q))
            ) * self.qw[el, i]

        # fe_q_num = Numerical_derivative(lambda t, qe: self.f_pot_el(qe, el), order=2)._x(0, qe, eps=1.0e-6)
        # # return fe_q_num

        # diff = fe_q_num - fe_q
        # # np.set_printoptions(2)
        # # print(f'diff:\n{diff}')
        # error = np.linalg.norm(diff)
        # print(f'error in f_pot_q_el: {error:.4e}')
        # return fe_q_num

        return fe_q

    def f_pot_q(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            Ke = self.f_pot_q_el(q[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(Ke, (self.uDOF[elDOF], self.qDOF[elDOF]))

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
        if xi == 0:
            return self.elDOF[0]
        elif xi == 1:
            return self.elDOF[-1]
        else:
            el = np.where(xi >= self.element_span)[0][-1]
            return self.elDOF[el]

    def qDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def uDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def r_OP(self, t, q, frame_ID, K_r_SP=None):
        return self.r_OP_q(t, q, frame_ID) @ q

    def r_OP_q(self, t, q, frame_ID, K_r_SP=None):
        NN, _ = self.stacked_basis_functions(frame_ID)
        r_q = np.zeros((3, self.nq_el))
        r_q[:2] = NN
        return r_q

    def A_IK(self, t, q, frame_ID):
        _, dNN = self.stacked_basis_functions(frame_ID)
        t = np.zeros(3)
        t[:2] = dNN @ q
        d1 = t / norm(t)
        d2 = cross3(e3, d1)
        A_IK = np.eye(3)
        A_IK[:, 0] = d1
        A_IK[:, 1] = d2
        return A_IK

    def A_IK_q(self, t, q, frame_ID):
        _, dNN = self.stacked_basis_functions(frame_ID)
        t = np.zeros(3)
        t_ = dNN @ q
        t[:2] = t_
        g_ = norm(t)
        d1_q = np.zeros((3, self.nq_el))
        d1_q[:2] = dNN / g_ - np.outer(t_ / (g_**3), t_ @ dNN)
        d2_q = ax2skew(e3) @ d1_q

        A_IK_q = np.zeros((3, 3, self.nq_el))
        A_IK_q[:, 0] = d1_q
        A_IK_q[:, 1] = d2_q

        return A_IK_q

        # A_IK_q_num = Numerical_derivative(lambda t, q: self.A_IK(t, q, frame_ID=frame_ID), order=2)._x(t, q)
        # diff = A_IK_q - A_IK_q_num
        # error = np.max(np.abs(diff))
        # print(f'error A_IK_q: {error}')
        # return A_IK_q_num

    def v_P(self, t, q, u, frame_ID, K_r_SP=None):
        return self.r_OP(t, u, frame_ID=frame_ID)

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return self.r_OP(t, u_dot, frame_ID=frame_ID)

    def a_P_q(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_el))

    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_el))

    def v_P_q(self, t, q, u, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_el))

    def J_P(self, t, q, frame_ID, K_r_SP=None):
        return self.r_OP_q(t, None, frame_ID=frame_ID)

    def J_P_q(self, t, q, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_el, self.nq_el))

    def K_Omega(self, t, q, u, frame_ID):
        _, dNN = self.stacked_basis_functions(frame_ID)
        t = dNN @ q
        t_perp = np.array([-t[1], t[0]])
        g2_ = t[0] * t[0] + t[1] * t[1]
        t_dot = dNN @ u
        phi_dot = t_perp @ t_dot / g2_

        return np.array([0, 0, phi_dot])

    def K_Omega_q(self, t, q, u, frame_ID):
        return np.zeros((3, self.nq_el))

    def K_Psi(self, t, q, u, u_dot, frame_ID):
        _, dNN = self.stacked_basis_functions(frame_ID)
        t = dNN @ q
        t_perp = np.array([-t[1], t[0]])
        g2_ = t[0] * t[0] + t[1] * t[1]
        g4_ = g2_ * g2_
        t_dot = dNN @ u
        t_dot_perp = np.array([-t_dot[1], t_dot[0]])
        t_ddot = dNN @ u_dot
        phi_ddot = (t_perp @ t_ddot + t_dot @ t_dot_perp) / g2_ - (
            t_perp @ t_dot
        ) / g4_ * 2 * t @ t_dot

        return np.array([0, 0, phi_ddot])

    def K_J_R(self, t, q, frame_ID):
        _, dNN = self.stacked_basis_functions(frame_ID)
        t = dNN @ q
        t_perp = np.array([-t[1], t[0]])
        g2_ = t[0] * t[0] + t[1] * t[1]

        K_J_R = np.zeros((3, self.nq_el))
        K_J_R[2] = t_perp @ dNN / g2_
        return K_J_R

    def K_J_R_q(self, t, q, frame_ID):
        _, dN, _ = self.basis_functions(frame_ID[0])
        dNN = self.stack2(dN)
        dNN_perp = self.stack2_perp(dN)
        t = dNN @ q
        t_perp = dNN_perp @ q
        g2_ = t[0] * t[0] + t[1] * t[1]

        K_J_R_q = np.zeros((3, self.nq_el, self.nq_el))
        K_J_R_q[2] = np.einsum("ik,ij->jk", dNN_perp, dNN) / g2_ - (
            2 / g2_**2
        ) * np.outer(t_perp @ dNN, t @ dNN)
        return K_J_R_q

        # K_J_R_q_num = Numerical_derivative(lambda t, q: self.K_J_R(t, q, frame_ID=frame_ID), order=2)._x(t, q)
        # diff = K_J_R_q - K_J_R_q_num
        # error = np.max(np.abs(diff))
        # print(f'error K_J_R_q: {error}')
        # return K_J_R_q_num

    ####################################################
    # body force
    ####################################################
    def body_force_el(self, force, t, el):
        fe = np.zeros(self.nq_el)
        N, J0, xi, qw, = (
            self.N[el],
            self.J0[el],
            self.xi[el],
            self.qw[el],
        )

        for Ni, xii, J0i, qwi in zip(N, xi, J0, qw):
            NNi = self.stack2(Ni)
            r_q = np.zeros((3, self.nq_el))
            r_q[:2] = NNi
            fe += r_q.T @ force(xii, t) * J0i * qwi
        return fe

    def body_force(self, t, q, force):
        f = np.zeros(self.nq)
        for el in range(self.nEl):
            f[self.elDOF[el]] += self.body_force_el(force, t, el)
        return f

    def body_force_q(self, t, q, coo, force):
        pass

    ####################################################
    # visualization
    ####################################################
    def nodes(self, q):
        nodes2D = q.reshape(2, -1)
        nodes = np.zeros((3, nodes2D.shape[1]))
        nodes[:2, :] = nodes2D
        return nodes

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

    ############
    # vtk export
    ############
    def post_processing(self, t, q, filename, u=None, binary=True, dim=3):
        # write paraview PVD file collecting time and all vtk files, see https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
        from xml.dom import minidom

        root = minidom.Document()

        vkt_file = root.createElement("VTKFile")
        vkt_file.setAttribute("type", "Collection")
        root.appendChild(vkt_file)

        collection = root.createElement("Collection")
        vkt_file.appendChild(collection)

        if u is None:
            u = np.zeros_like(q)

        for i, (ti, qi, ui) in enumerate(zip(t, q, u)):
            filei = filename + f"{i}.vtu"

            # write time step and file name in pvd file
            dataset = root.createElement("DataSet")
            dataset.setAttribute("timestep", f"{ti:0.6f}")
            dataset.setAttribute("file", filei)
            collection.appendChild(dataset)

            geom_points = np.array([]).reshape(0, dim)
            cells = []
            HigherOrderDegrees = []
            point_data = {}

            (
                geom_points,
                point_data,
                cells,
                HigherOrderDegrees,
            ) = self.post_processing_subsystem(
                ti, qi[self.qDOF], ui[self.uDOF], binary=binary
            )

            # write vtk mesh using meshio
            meshio.write_points_cells(
                os.path.splitext(os.path.basename(filei))[0] + ".vtu",
                geom_points,  # only export centerline as geometry here!
                cells,
                point_data=point_data,
                cell_data={"HigherOrderDegrees": HigherOrderDegrees},
                binary=binary,
            )

        # write pvd file
        xml_str = root.toprettyxml(indent="\t")
        with open(filename + ".pvd", "w") as f:
            f.write(xml_str)

    def post_processing_subsystem(self, t, q, u, binary=True):
        # centerline and connectivity + director data
        cells, points, HigherOrderDegrees = self.mesh.vtk_mesh(q)

        geom_points = np.zeros((points.shape[0], 3))
        geom_points[:, :2] = points

        point_data = {}

        _, displacement, _ = self.mesh.vtk_mesh(q - self.q0)
        point_data = {"u": displacement}

        _, velocities, _ = self.mesh.vtk_mesh(u)
        point_data.update({"v": velocities})

        # export existing values on quadrature points using L2 projection
        J0_vtk = self.mesh.field_to_vtk(self.J0.reshape(self.nEl, self.nQP, 1))
        point_data.update({"J0": J0_vtk})

        kappa0_vtk = self.mesh.field_to_vtk(self.kappa0.reshape(self.nEl, self.nQP, 1))
        point_data.update({"kappa0": kappa0_vtk})

        # evaluate strain measures at quadrature points
        kappa = np.zeros((self.mesh.nelement, self.mesh.nquadrature))
        stretch = np.zeros((self.mesh.nelement, self.mesh.nquadrature))

        d1 = np.zeros((self.mesh.nelement, self.mesh.nquadrature, 3))
        d2 = np.zeros((self.mesh.nelement, self.mesh.nquadrature, 3))
        d3 = np.zeros((self.mesh.nelement, self.mesh.nquadrature, 3))

        for el in range(self.mesh.nelement):
            qe = q[self.elDOF[el]]
            N_xi, N_xixi, J0 = self.N_xi[el], self.N_xixi[el], self.J0[el]

            for i, (N_xii, N_xixii, J0i) in enumerate(zip(N_xi, N_xixi, J0)):
                # build matrix of shape function derivatives
                NN_xii = self.stack2(N_xii)
                NN_xixii = self.stack2(N_xixii)
                # tangential vectors
                t = NN_xii @ qe
                n = NN_xixii @ qe
                # rotated tangential and normal vectors
                t_perp = np.array([-t[1], t[0]])
                # directors
                d1[el, i] = np.array([*t / norm(t), 0])
                d2[el, i] = np.array([*t_perp / norm(t_perp), 0])
                d3[el, i] = np.array([0, 0, 1])
                # current stretch
                g2_ = t[0] * t[0] + t[1] * t[1]
                g_ = sqrt(g2_)
                # change of angle
                theta_bar_xi = t_perp @ n / g2_
                # strain measures
                stretch[el, i] = g_ / J0i
                kappa[el, i] = theta_bar_xi / J0i

        # L2 projection of strain measures
        kappa_vtk = self.mesh.field_to_vtk(kappa.reshape(self.nEl, self.nQP, 1))
        point_data.update({"kappa": kappa_vtk})

        stretch_vtk = self.mesh.field_to_vtk(stretch.reshape(self.nEl, self.nQP, 1))
        point_data.update({"stretch": stretch_vtk})

        # L2 projections of directors
        d1_vtk = self.mesh.field_to_vtk(d1)
        point_data.update({"d1": d1_vtk})
        d2_vtk = self.mesh.field_to_vtk(d2)
        point_data.update({"d2": d2_vtk})
        # TODO: d3 cannot be exported due to limited number of fileds that can be
        #       tessellated by paraview in the default setting
        # d3_vtk = self.mesh.field_to_vtk(d3)
        # point_data.update({"d3": d3_vtk})

        # fields depending on strain measures and other previously computed quantities
        point_data_fields = {
            "W": lambda stretch, kappa, kappa0: np.array(
                [self.material_model.potential(stretch, kappa, kappa0)]
            ),
            "n": lambda stretch, kappa, kappa0: self.material_model.n(
                stretch, kappa, kappa0
            ),
            "m": lambda stretch, kappa, kappa0: self.material_model.m(
                stretch, kappa, kappa0
            ),
        }

        for name, fun in point_data_fields.items():
            tmp = fun(stretch_vtk[0], kappa_vtk[0], kappa0_vtk[0]).reshape(-1)
            field = np.zeros((len(stretch_vtk), len(tmp)))
            for i, (stretch_i, kappa_i, kappa0_i) in enumerate(
                zip(stretch_vtk, kappa_vtk, kappa0_vtk)
            ):
                field[i] = fun(stretch_i, kappa_i, kappa0_i).reshape(-1)
            point_data.update({name: field})

        return geom_points, point_data, cells, HigherOrderDegrees


# TODO: adapt this implementation with new style from spatial beams
class Inextensible_Euler_bernoulli(EulerBernoulli2D):
    def __init__(self, *args, la_g0=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.polynomial_degree_g = self.polynomial_degree - 1
        self.nn_el_g = self.polynomial_degree_g + 1  # number of nodes per element
        self.nn_g = self.nEl + self.polynomial_degree_g  # number of nodes
        self.nq_n_g = 1  # number of degrees of freedom per node
        self.nla_g = self.nn_g * self.nq_n_g
        self.nla_g_el = self.nn_el_g * self.nq_n_g

        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.knot_vector_g = KnotVector(
            self.polynomial_degree_g, self.nEl
        )  # uniform open knot vector
        self.element_span_g = self.knot_vector_g.data[
            self.polynomial_degree_g : -self.polynomial_degree_g
        ]
        self.mesh_g = Mesh1D(
            self.knot_vector_g,
            self.nQP,
            derivative_order=0,
            dim=self.nq_n_g,
        )

        self.elDOF_g = self.mesh_g.elDOF

        # compute shape functions
        self.N_g = self.mesh_g.N

    def __g_el(self, qe, el):
        g = np.zeros(self.nla_g_el)

        N_xi, N_g, J0, qw = self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el]

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack2(N_xii)

            r_xi = NN_xii @ qe

            g += (r_xi @ r_xi / J0i - J0i) * N_gi * qwi

        return g

    def __g_q_el(self, qe, el):
        g_q = np.zeros((self.nla_g_el, self.nq_el))

        N_xi, N_g, J0, qw = self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el]

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack2(N_xii)

            r_xi = NN_xii @ qe

            g_q += np.outer(2 * N_gi * qwi / J0i, r_xi @ NN_xii)

        return g_q

        # g_q_num = Numerical_derivative(lambda t, q: self.__g_el(q, N, N_xi, N_g, J0, qw), order=2)._x(0, qe)
        # diff = g_q_num - g_q
        # error = np.linalg.norm(diff)
        # print(f'error g_q: {error}')
        # return g_q_num

    def __g_dot_el(self, qe, ue, el):
        g_dot = np.zeros(self.nla_g_el)

        N_xi, N_g, J0, qw = self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el]

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack2(N_xii)

            r_xi = NN_xii @ qe
            r_xi_dot = NN_xii @ ue

            g_dot += 2 * r_xi @ r_xi_dot / J0i * N_gi * qwi

        return g_dot

    def __g_dot_q_el(self, qe, ue, el):
        g_dot_q = np.zeros((self.nla_g_el, self.nq_el))

        N_xi, N_g, J0, qw = self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el]

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack2(N_xii)

            r_xi_dot = NN_xii @ ue

            g_dot_q += np.outer(N_gi, 2 * r_xi_dot @ NN_xii / J0i * qwi)

        return g_dot_q

    def __g_ddot_el(self, qe, ue, ue_dot, el):
        g_ddot = np.zeros(self.nla_g_el)

        N_xi, N_g, J0, qw = self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el]

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack2(N_xii)

            r_xi = NN_xii @ qe
            r_xi_dot = NN_xii @ ue
            r_xi_ddot = NN_xii @ ue_dot

            g_ddot += (r_xi @ r_xi_ddot + r_xi_dot @ r_xi_dot) * 2 / J0i * N_gi * qwi

        return g_ddot

    def __g_ddot_q_el(self, qe, ue, ue_dot, el):
        g_ddot_q = np.zeros((self.nla_g_el, self.nq_el))

        N_xi, N_g, J0, qw = self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el]

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack2(N_xii)

            r_xi_ddot = NN_xii @ ue_dot

            g_ddot_q += np.outer(N_gi, (r_xi_ddot @ NN_xii) * 2 / J0i * qwi)

        return g_ddot_q

    def __g_ddot_u_el(self, qe, ue, ue_dot, el):
        g_ddot_u = np.zeros((self.nla_g_el, self.nq_el))

        N_xi, N_g, J0, qw = self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el]

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack2(N_xii)

            r_xi_dot = NN_xii @ ue

            g_ddot_u += np.outer(N_gi, (r_xi_dot @ NN_xii) * 4 / J0i * qwi)

        return g_ddot_u

    def __g_qq_el(self, qe, el):
        g_qq = np.zeros((self.nla_g_el, self.nq_el, self.nq_el))

        N_xi, N_g, J0, qw = self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el]

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack2(N_xii)

            g_qq += np.einsum("i,jl,jk->ikl", 2 * N_gi * qwi / J0i, NN_xii, NN_xii)

        return g_qq

        # g_qq_num = Numerical_derivative(lambda t, q: self.__g_q_el(q, N, N_xi, N_g, J0, qw), order=2)._x(0, qe)
        # diff = g_qq_num - g_qq
        # error = np.linalg.norm(diff)
        # print(f'error g_qq: {error}')
        # return g_qq_num

    # global constraint functions
    def g(self, t, q):
        g = np.zeros(self.nla_g)
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g[elDOF_g] += self.__g_el(q[elDOF], el)
        return g

    def g_q(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_q = self.__g_q_el(q[elDOF], el)
            coo.extend(g_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))

    def W_g(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_q = self.__g_q_el(q[elDOF], el)
            coo.extend(g_q.T, (self.uDOF[elDOF], self.la_gDOF[elDOF_g]))

    def Wla_g_q(self, t, q, la_g, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_qq = self.__g_qq_el(q[elDOF], el)
            coo.extend(
                np.einsum("i,ijk->jk", la_g[elDOF_g], g_qq),
                (self.uDOF[elDOF], self.qDOF[elDOF]),
            )

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g)
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_dot[elDOF_g] += self.__g_dot_el(q[elDOF], u[elDOF], el)
        return g_dot

    def g_dot_q(self, t, q, u, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_dot_q = self.__g_dot_q_el(q[elDOF], u[elDOF], el)
            coo.extend(g_dot_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))

    def g_dot_u(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_dot_u = self.__g_q_el(q[elDOF], el)
            coo.extend(g_dot_u, (self.la_gDOF[elDOF_g], self.uDOF[elDOF]))

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g)
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_ddot[elDOF_g] += self.__g_ddot_el(q[elDOF], u[elDOF], u_dot[elDOF], el)
        return g_ddot

    def g_ddot_q(self, t, q, u, u_dot, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_ddot_q = self.__g_ddot_q_el(q[elDOF], u[elDOF], u_dot[elDOF], el)
            coo.extend(g_ddot_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))

    def g_ddot_u(self, t, q, u, u_dot, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_ddot_u = self.__g_ddot_u_el(q[elDOF], u[elDOF], u_dot[elDOF], el)
            coo.extend(g_ddot_u, (self.la_gDOF[elDOF_g], self.uDOF[elDOF]))
