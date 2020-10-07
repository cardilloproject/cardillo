import numpy as np
from abc import ABCMeta, abstractmethod
import meshio
import os

from cardillo.utility.coo import Coo
from cardillo.discretization import uniform_knot_vector, B_spline_basis1D, Lagrange_basis
from cardillo.discretization.B_spline import Knot_vector
from cardillo.math.algebra import norm3, cross3, skew2ax, skew2ax_A, ax2skew
from cardillo.math.numerical_derivative import Numerical_derivative
from cardillo.discretization.mesh1D import Mesh1D

class Timoshenko_beam_director(metaclass=ABCMeta):
    def __init__(self, material_model, A_rho0, B_rho0, C_rho0, polynomial_degree, nQP, nEl, Q, q0=None, u0=None, la_g0=None, basis='B-spline'):
        # beam properties
        self.materialModel = material_model # material model
        self.A_rho0 = A_rho0 # line density
        self.B_rho0 = B_rho0 # first moment
        self.C_rho0 = C_rho0 # second moment

        # material model
        self.material_model = material_model

        # discretization parameters
        self.polynomial_degree = polynomial_degree # polynomial degree
        self.nQP = nQP # number of quadrature points
        self.nEl = nEl # number of elements

        self.nn = nn = nEl + polynomial_degree # number of nodes
        self.knot_vector = Knot_vector(polynomial_degree, nEl)
        self.element_span = self.knot_vector.data[polynomial_degree:-polynomial_degree]

        self.nn_el = nn_el = polynomial_degree + 1 # number of nodes per element
        self.nq_n = nq_n = 12 # number of degrees of freedom per node (x, y, z) + 3 * d_i
        
        self.mesh_kinematics = Mesh1D(self.knot_vector, nQP, derivative_order=1, basis=basis, nq_n=self.nq_n)

        self.nq = nn * nq_n # total number of generalized coordinates
        self.nu = self.nq
        self.nq_el = nn_el * nq_n # total number of generalized coordinates per element
        self.nq_el_r = 3 * nn_el

        self.basis = basis

        self.elDOF = self.mesh_kinematics.elDOF
        self.nodalDOF = np.arange(self.nq_n * self.nn).reshape(self.nq_n, self.nn).T

        # degrees of freedom on element level
        self.rDOF = np.arange(3 * nn_el)
        self.d1DOF = np.arange(3 * nn_el, 6  * nn_el)
        self.d2DOF = np.arange(6 * nn_el, 9  * nn_el)
        self.d3DOF = np.arange(9 * nn_el, 12 * nn_el)
            
        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

        # compute shape functions
        self.N = self.mesh_kinematics.N
        self.N_xi = self.mesh_kinematics.N_xi
        self.qw = self.mesh_kinematics.wp
        self.xi = self.mesh_kinematics.qp
        self.J0 = np.zeros((nEl, nQP))
        self.Gamma0 = np.zeros((nEl, nQP, 3))
        self.Kappa0 = np.zeros((nEl, nQP, 3))
        for el in range(nEl):
            if self.basis == 'B-spline':
                self.basis_functions = self.__basis_functions_b_splines
            else:
                self.basis_functions = self.__basis_functions_lagrange

            # precompute quantities of the reference configuration
            Qe = self.Q[self.elDOF[el]]
            Qe_r0 = Qe[self.rDOF]
            Qe_D1 = Qe[self.d1DOF]
            Qe_D2 = Qe[self.d2DOF]
            Qe_D3 = Qe[self.d3DOF]

            # for i in range(nQP):
            for i, (Ni, N_xii) in enumerate(zip(self.N[el], self.N_xi[el])):
                # r0_xi = self.stack_shapefunctions(self.N_xi[el, i]) @ Qe
                # self.J0[el, i] = norm3(r0_xi)
                
                # build matrix of shape function derivatives
                NNi = self.stack_shapefunctions(Ni)
                NN_xii = self.stack_shapefunctions(N_xii)

                # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
                # the parameter space \xi and thus need to be transformed later
                r0_xi = NN_xii @ Qe_r0
                J0i = norm3(r0_xi)
                self.J0[el, i] = J0i

                D1 = NNi @ Qe_D1
                D1_xi = NN_xii @ Qe_D1

                D2 = NNi @ Qe_D2
                D2_xi = NN_xii @ Qe_D2

                D3 = NNi @ Qe_D3
                D3_xi = NN_xii @ Qe_D3
                
                # compute derivatives w.r.t. the arc lenght parameter s
                r0_s = r0_xi / J0i                
                D1_s = D1_xi / J0i
                D2_s = D2_xi / J0i
                D3_s = D3_xi / J0i

                # build rotation matrices
                R0 = np.vstack((D1, D2, D3)).T

                # axial and shear strains
                self.Gamma0[el, i] = R0.T @ r0_s

                # torsional and flexural strains
                self.Kappa0[el, i] = np.array([0.5 * (D3 @ D2_s - D2 @ D3_s), \
                                                 0.5 * (D1 @ D3_s - D3 @ D1_s), \
                                                 0.5 * (D2 @ D1_s - D1 @ D2_s)])

        # shape functions on the boundary
        N_bdry, dN_bdry = self.basis_functions(0)
        N_bdry_left = self.stack_shapefunctions(N_bdry)
        dN_bdry_left = self.stack_shapefunctions(dN_bdry)

        N_bdry, dN_bdry = self.basis_functions(1)
        N_bdry_right = self.stack_shapefunctions(N_bdry)
        dN_bdry_right = self.stack_shapefunctions(dN_bdry)

        self.N_bdry = np.array([N_bdry_left, N_bdry_right])
        self.dN_bdry = np.array([dN_bdry_left, dN_bdry_right])

    def __basis_functions_b_splines(self, xi):
        return B_spline_basis1D(self.polynomial_degree, 1, self.knot_vector.data, xi).T

    def __basis_functions_lagrange(self, xi):
        el = np.where(xi >= self.element_span)[0][-1]
        diff_xi = self.element_span[el + 1] - self.element_span[el]
        sum_xi = self.element_span[el + 1] + self.element_span[el]
        xi_tilde = (2 * xi - sum_xi) / diff_xi
        return Lagrange_basis(self.polynomial_degree, xi_tilde, derivative=True)

    def stack_shapefunctions(self, N):
        nn_el = self.nn_el
        NN = np.zeros((3, self.nq_el_r))
        NN[0, :nn_el] = N
        NN[1, nn_el:2*nn_el] = N
        NN[2, 2*nn_el:] = N
        return NN

    def assembler_callback(self):
        self.__M_coo()

    #########################################
    # equations of motion
    #########################################
    def M_el(self, N, J0, qw):
        Me = np.zeros((self.nq_el, self.nq_el))

        for Ni, J0i, qwi in zip(N, J0, qw):
            # build matrix of shape functions and derivatives
            NNi = self.stack_shapefunctions(Ni)
            factor = NNi.T @ NNi * J0i * qwi

            # delta r * ddot r
            Me[self.rDOF[:, None], self.rDOF] += self.A_rho0 * factor
            # delta r * ddot d2
            Me[self.rDOF[:, None], self.d2DOF] += self.B_rho0[1] * factor
            # delta r * ddot d3
            Me[self.rDOF[:, None], self.d3DOF] += self.B_rho0[2] * factor

            # delta d2 * ddot r
            Me[self.d2DOF[:, None], self.rDOF] += self.B_rho0[1] * factor
            Me[self.d2DOF[:, None], self.d2DOF] += self.C_rho0[1, 1] * factor
            Me[self.d2DOF[:, None], self.d3DOF] += self.C_rho0[1, 2] * factor

            # delta d3 * ddot r
            Me[self.d3DOF[:, None], self.rDOF] += self.B_rho0[2] * factor
            Me[self.d3DOF[:, None], self.d2DOF] += self.C_rho0[2, 1] * factor
            Me[self.d3DOF[:, None], self.d3DOF] += self.C_rho0[2, 2] * factor

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
        
    def E_pot(self, t, q):
        E = 0
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            E += self.E_pot_el(q[elDOF], self.Q[elDOF], self.N[el], self.N_xi[el], self.J0[el], self.qw[el])
        return E
    
    def E_pot_el(self, qe, Qe, N, N_xi, J0, qw):
        Ee = 0

        # extract generalized coordinates for beam centerline and directors 
        # in the current and reference configuration
        qe_r = qe[self.rDOF]
        qe_d1 = qe[self.d1DOF]
        qe_d2 = qe[self.d2DOF]
        qe_d3 = qe[self.d3DOF]
        
        Qe_r0 = Qe[self.rDOF]
        Qe_D1 = Qe[self.d1DOF]
        Qe_D2 = Qe[self.d2DOF]
        Qe_D3 = Qe[self.d3DOF]

        # integrate element force vector
        for Ni, N_xii, J0i, qwi in zip(N, N_xi, J0, qw):
            # build matrix of shape function derivatives
            NNi = self.stack_shapefunctions(Ni)
            NN_xii = self.stack_shapefunctions(N_xii)

            # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
            # the parameter space \xi and thus need to be transformed later
            dr_dxi = NN_xii @ qe_r
            dr0_dxi = NN_xii @ Qe_r0

            d1 = NNi @ qe_d1
            D1 = NNi @ Qe_D1
            dd1_dxi = NN_xii @ qe_d1
            dD1_dxi = NN_xii @ Qe_D1

            d2 = NNi @ qe_d2
            D2 = NNi @ Qe_D2
            dd2_dxi = NN_xii @ qe_d2
            dD2_dxi = NN_xii @ Qe_D2

            d3 = NNi @ qe_d3
            D3 = NNi @ Qe_D3
            dd3_dxi = NN_xii @ qe_d3
            dD3_dxi = NN_xii @ Qe_D3
            
            # compute derivatives w.r.t. the arc lenght parameter s
            dr_ds = dr_dxi / J0i
            dr0_ds = dr0_dxi / J0i

            dd1_ds = dd1_dxi / J0i
            dd2_ds = dd2_dxi / J0i
            dd3_ds = dd3_dxi / J0i
            
            dD1_ds = dD1_dxi / J0i
            dD2_ds = dD2_dxi / J0i
            dD3_ds = dD3_dxi / J0i

            # build rotation matrices
            R = np.vstack((d1, d2, d3)).T
            R0 = np.vstack((D1, D2, D3)).T
            
            # axial and shear strains
            Gamma_i = R.T @ dr_ds
            Gamma0_i = R0.T @ dr0_ds

            # torsional and flexural strains
            Kappa_i = np.array([0.5 * (d3 @ dd2_ds - d2 @ dd3_ds), \
                                0.5 * (d1 @ dd3_ds - d3 @ dd1_ds), \
                                0.5 * (d2 @ dd1_ds - d1 @ dd2_ds)])
            Kappa0_i = np.array([0.5 * (D3 @ dD2_ds - D2 @ dD3_ds), \
                                 0.5 * (D1 @ dD3_ds - D3 @ dD1_ds), \
                                 0.5 * (D2 @ dD1_ds - D1 @ dD2_ds)])
            
            # evaluate strain energy function
            Ee += self.material_model.potential(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i) * J0i * qwi

        return Ee

    def f_pot(self, t, q):
        f = np.zeros(self.nu)
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            f[elDOF] += self.f_pot_el(q[elDOF], el)
        return f
    
    def f_pot_el(self, qe, el):
        fe = np.zeros(self.nq_el)

        N, N_xi, Gamma0, Kappa0, J0, qw = self.N[el], self.N_xi[el], self.Gamma0[el], self.Kappa0[el], self.J0[el], self.qw[el]

        # extract generalized coordinates for beam centerline and directors 
        # in the current and reference configuration
        qe_r = qe[self.rDOF]
        qe_d1 = qe[self.d1DOF]
        qe_d2 = qe[self.d2DOF]
        qe_d3 = qe[self.d3DOF]

        # integrate element force vector
        for Ni, N_xii, Gamma0_i, Kappa0_i, J0i, qwi in zip(N, N_xi, Gamma0, Kappa0, J0, qw):
            # build matrix of shape function derivatives
            NNi = self.stack_shapefunctions(Ni)
            NN_xii = self.stack_shapefunctions(N_xii)

            # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
            # the parameter space \xi and thus need to be transformed later
            r_xi = NN_xii @ qe_r
            d1 = NNi @ qe_d1
            d1_xi = NN_xii @ qe_d1
            d2 = NNi @ qe_d2
            d2_xi = NN_xii @ qe_d2
            d3 = NNi @ qe_d3
            d3_xi = NN_xii @ qe_d3
            
            # compute derivatives w.r.t. the arc lenght parameter s
            r_s = r_xi / J0i
            d1_s = d1_xi / J0i
            d2_s = d2_xi / J0i
            d3_s = d3_xi / J0i

            # build rotation matrices
            R = np.vstack((d1, d2, d3)).T
            
            # axial and shear strains
            Gamma_i = R.T @ r_s

            # torsional and flexural strains
            Kappa_i = np.array([0.5 * (d3 @ d2_s - d2 @ d3_s), \
                                0.5 * (d1 @ d3_s - d3 @ d1_s), \
                                0.5 * (d2 @ d1_s - d1 @ d2_s)])
            
            # compute contact forces and couples from partial derivatives of the strain energy function w.r.t. strain measures
            n_i = self.material_model.n_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            # n = n_i[0] * d1 + n_i[1] * d2 + n_i[2] * d3
            n = R @ n_i
            m_i = self.material_model.m_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            # m = m_i[0] * d1 + m_i[1] * d2 + m_i[2] * d3
            # m = R @ m_i
            
            # new version
            # quadrature point contribution to element residual
            n1, n2, n3 = n_i
            m1, m2, m3 = m_i
            fe[self.rDOF] -= ( NN_xii.T @ n ) * qwi
            fe[self.d1DOF] -= ( NNi.T @ ( r_xi * n1 + (m2 / 2 * d3_xi - m3 / 2 * d2_xi) ) + NN_xii.T @ (m3 / 2 * d2 - m2 / 2 * d3) ) * qwi
            fe[self.d2DOF] -= ( NNi.T @ ( r_xi * n2 + (m3 / 2 * d1_xi - m1 / 2 * d3_xi) ) + NN_xii.T @ (m1 / 2 * d3 - m3 / 2 * d1) ) * qwi
            fe[self.d3DOF] -= ( NNi.T @ ( r_xi * n3 + (m1 / 2 * d2_xi - m2 / 2 * d1_xi) ) + NN_xii.T @ (m2 / 2 * d1 - m1 / 2 * d2) ) * qwi
            
            # old version:
            # # quadrature point contribution to element residual
            # fe[self.rDOF] -= ( NN_xii.T @ n ) * qwi
            # fe[self.d1DOF] -= ( NNi.T @ ( r_xi * n_i[0] + 0.5 * cross3(m, d1_xi) ) + 0.5 * NN_xii.T @ cross3(m, d1) ) * qwi
            # fe[self.d2DOF] -= ( NNi.T @ ( r_xi * n_i[1] + 0.5 * cross3(m, d2_xi) ) + 0.5 * NN_xii.T @ cross3(m, d2) ) * qwi
            # fe[self.d3DOF] -= ( NNi.T @ ( r_xi * n_i[2] + 0.5 * cross3(m, d3_xi) ) + 0.5 * NN_xii.T @ cross3(m, d3) ) * qwi

        return fe

    def f_pot_q(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            Ke = self.f_pot_q_el(q[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(Ke, (self.uDOF[elDOF], self.qDOF[elDOF]))

    def f_pot_q_el(self, qe, el):
        # return Numerical_derivative(lambda t, q: self.f_pot_el(q, el))._x(0, qe)
        Ke = np.zeros((self.nq_el, self.nq_el))

        N, N_xi, Gamma0, Kappa0, J0, qw = self.N[el], self.N_xi[el], self.Gamma0[el], self.Kappa0[el], self.J0[el], self.qw[el]

        # extract generalized coordinates for beam centerline and directors 
        # in the current and reference configuration
        qe_r = qe[self.rDOF]
        qe_d1 = qe[self.d1DOF]
        qe_d2 = qe[self.d2DOF]
        qe_d3 = qe[self.d3DOF]

        # integrate element force vector
        for Ni, N_xii, Gamma0_i, Kappa0_i, J0i, qwi in zip(N, N_xi, Gamma0, Kappa0, J0, qw):
            # build matrix of shape function derivatives
            NNi = self.stack_shapefunctions(Ni)
            NN_xii = self.stack_shapefunctions(N_xii)

            # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
            # the parameter space \xi and thus need to be transformed later
            r_xi = NN_xii @ qe_r
            d1 = NNi @ qe_d1
            d1_xi = NN_xii @ qe_d1
            d2 = NNi @ qe_d2
            d2_xi = NN_xii @ qe_d2
            d3 = NNi @ qe_d3
            d3_xi = NN_xii @ qe_d3
            
            # compute derivatives w.r.t. the arc lenght parameter s
            r_s = r_xi / J0i
            d1_s = d1_xi / J0i
            d2_s = d2_xi / J0i
            d3_s = d3_xi / J0i

            # build rotation matrices
            R = np.vstack((d1, d2, d3)).T
            
            # axial and shear strains
            Gamma_i = R.T @ r_s

            # derivative of axial and shear strains
            Gamma_j_qr = R.T @ NN_xii / J0i
            Gamma_1_qd1 = r_xi @ NNi / J0i
            Gamma_2_qd2 = r_xi @ NNi / J0i
            Gamma_3_qd3 = r_xi @ NNi / J0i

            # torsional and flexural strains
            Kappa_i = np.array([0.5 * (d3 @ d2_s - d2 @ d3_s), \
                                0.5 * (d1 @ d3_s - d3 @ d1_s), \
                                0.5 * (d2 @ d1_s - d1 @ d2_s)])

            # derivative of torsional and flexural strains
            kappa_1_qe_d1 = np.zeros(3 * self.nn_el)
            kappa_1_qe_d2 = 0.5 * (d3 @ NN_xii - d3_xi @ NNi) / J0i
            kappa_1_qe_d3 = 0.5 * (d2_xi @ NNi - d2 @ NN_xii) / J0i
                    
            kappa_2_qe_d1 = 0.5 * (d3_xi @ NNi - d3 @ NN_xii) / J0i
            kappa_2_qe_d2 = np.zeros(3 * self.nn_el)
            kappa_2_qe_d3 = 0.5 * (d1 @ NN_xii - d1_xi @ NNi) / J0i
            
            kappa_3_qe_d1 = 0.5 * (d2 @ NN_xii - d2_xi @ NNi) / J0i
            kappa_3_qe_d2 = 0.5 * (d1_xi @ NNi - d1 @ NN_xii) / J0i
            kappa_3_qe_d3 = np.zeros(3 * self.nn_el)

            kappa_j_qe_d1 = np.vstack((kappa_1_qe_d1, kappa_2_qe_d1, kappa_3_qe_d1))
            kappa_j_qe_d2 = np.vstack((kappa_1_qe_d2, kappa_2_qe_d2, kappa_3_qe_d2))
            kappa_j_qe_d3 = np.vstack((kappa_1_qe_d3, kappa_2_qe_d3, kappa_3_qe_d3))

            # compute contact forces and couples from partial derivatives of the strain energy function w.r.t. strain measures
            n1, n2, n3 = self.material_model.n_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            n_i_Gamma_i_j = self.material_model.n_i_Gamma_i_j(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            n_i_Kappa_i_j = self.material_model.n_i_Kappa_i_j(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            n1_qr, n2_qr, n3_qr = n_i_Gamma_i_j @ Gamma_j_qr

            n1_qd1, n2_qd1, n3_qd1 = np.outer(n_i_Gamma_i_j[0], Gamma_1_qd1) + n_i_Kappa_i_j @ kappa_j_qe_d1
            n1_qd2, n2_qd2, n3_qd2 = np.outer(n_i_Gamma_i_j[1], Gamma_2_qd2) + n_i_Kappa_i_j @ kappa_j_qe_d2
            n1_qd3, n2_qd3, n3_qd3 = np.outer(n_i_Gamma_i_j[2], Gamma_3_qd3) + n_i_Kappa_i_j @ kappa_j_qe_d3

            m1, m2, m3 = self.material_model.m_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            m_i_Gamma_i_j = self.material_model.m_i_Gamma_i_j(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            m_i_Kappa_i_j = self.material_model.m_i_Kappa_i_j(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)

            m1_qr, m2_qr, m3_qr = m_i_Gamma_i_j @ Gamma_j_qr
            m1_qd1, m2_qd1, m3_qd1 = np.outer(m_i_Gamma_i_j[0], Gamma_1_qd1) + m_i_Kappa_i_j @ kappa_j_qe_d1
            m1_qd2, m2_qd2, m3_qd2 = np.outer(m_i_Gamma_i_j[1], Gamma_2_qd2) + m_i_Kappa_i_j @ kappa_j_qe_d2
            m1_qd3, m2_qd3, m3_qd3 = np.outer(m_i_Gamma_i_j[2], Gamma_3_qd3) + m_i_Kappa_i_j @ kappa_j_qe_d3

            Ke[self.rDOF[:, None], self.rDOF] -= NN_xii.T @ (np.outer(d1, n1_qr) + np.outer(d2, n2_qr) + np.outer(d3, n3_qr)) * qwi
            Ke[self.rDOF[:, None], self.d1DOF] -= NN_xii.T @ (np.outer(d1, n1_qd1) + np.outer(d2, n2_qd1) + np.outer(d3, n3_qd1) + n1 * NNi) * qwi
            Ke[self.rDOF[:, None], self.d2DOF] -= NN_xii.T @ (np.outer(d1, n1_qd2) + np.outer(d2, n2_qd2) + np.outer(d3, n3_qd2) + n2 * NNi) * qwi
            Ke[self.rDOF[:, None], self.d3DOF] -= NN_xii.T @ (np.outer(d1, n1_qd3) + np.outer(d2, n2_qd3) + np.outer(d3, n3_qd3) + n3 * NNi) * qwi

            Ke[self.d1DOF[:, None], self.rDOF] -= NNi.T @ (np.outer(r_xi, n1_qr) + n1 * NN_xii + np.outer(0.5 * d3_xi, m2_qr) - np.outer(0.5 * d2_xi, m3_qr)) * qwi \
                                                  + NN_xii.T @ (np.outer(0.5 * d2, m3_qr) - np.outer(0.5 * d3, m2_qr)) * qwi
            Ke[self.d1DOF[:, None], self.d1DOF] -= NNi.T @ (np.outer(r_xi, n1_qd1) + np.outer(0.5 * d3_xi, m2_qd1) - np.outer(0.5 * d2_xi, m3_qd1)) * qwi \
                                                  + NN_xii.T @ (np.outer(0.5 * d2, m3_qd1) - np.outer(0.5 * d3, m2_qd1)) * qwi
            Ke[self.d1DOF[:, None], self.d2DOF] -= NNi.T @ (np.outer(r_xi, n1_qd2) + np.outer(0.5 * d3_xi, m2_qd2) - np.outer(0.5 * d2_xi, m3_qd2) - 0.5 * m3 * NN_xii) * qwi \
                                                  + NN_xii.T @ (np.outer(0.5 * d2, m3_qd2) + 0.5 * m3 * NNi - np.outer(0.5 * d3, m2_qd2)) * qwi
            Ke[self.d1DOF[:, None], self.d3DOF] -= NNi.T @ (np.outer(r_xi, n1_qd3) + np.outer(0.5 * d3_xi, m2_qd3) + 0.5 * m2 * NN_xii - np.outer(0.5 * d2_xi, m3_qd3)) * qwi \
                                                  + NN_xii.T @ (np.outer(0.5 * d2, m3_qd3) - np.outer(0.5 * d3, m2_qd3) - 0.5 * m2 * NNi) * qwi

            Ke[self.d2DOF[:, None], self.rDOF] -= NNi.T @ (np.outer(r_xi, n2_qr) + n2 * NN_xii + np.outer(0.5 * d1_xi, m3_qr) - np.outer(0.5 * d3_xi, m1_qr)) * qwi \
                                                  + NN_xii.T @ (np.outer(0.5 * d3, m1_qr) - np.outer(0.5 * d1, m3_qr)) * qwi
            Ke[self.d2DOF[:, None], self.d1DOF] -= NNi.T @ (np.outer(r_xi, n2_qd1) + np.outer(0.5 * d1_xi, m3_qd1) + 0.5 * m3 * NN_xii - np.outer(0.5 * d3_xi, m1_qd1)) * qwi \
                                                  + NN_xii.T @ (np.outer(0.5 * d3, m1_qd1) - np.outer(0.5 * d1, m3_qd1) - 0.5 * m3 * NNi) * qwi
            Ke[self.d2DOF[:, None], self.d2DOF] -= NNi.T @ (np.outer(r_xi, n2_qd2) + np.outer(0.5 * d1_xi, m3_qd2) - np.outer(0.5 * d3_xi, m1_qd2)) * qwi \
                                                  + NN_xii.T @ (np.outer(0.5 * d3, m1_qd2) - np.outer(0.5 * d1, m3_qd2)) * qwi
            Ke[self.d2DOF[:, None], self.d3DOF] -= NNi.T @ (np.outer(r_xi, n2_qd3) + np.outer(0.5 * d1_xi, m3_qd3) - np.outer(0.5 * d3_xi, m1_qd3) - 0.5 * m1 * NN_xii) * qwi \
                                                  + NN_xii.T @ (np.outer(0.5 * d3, m1_qd3) + 0.5 * m1 * NNi - np.outer(0.5 * d1, m3_qd3)) * qwi

            Ke[self.d3DOF[:, None], self.rDOF] -= NNi.T @ (np.outer(r_xi, n3_qr) + n3 * NN_xii + np.outer(0.5 * d2_xi, m1_qr) - np.outer(0.5 * d1_xi, m2_qr)) * qwi \
                                                  + NN_xii.T @ (np.outer(0.5 * d1, m2_qr) - np.outer(0.5 * d2, m1_qr)) * qwi
            Ke[self.d3DOF[:, None], self.d1DOF] -= NNi.T @ (np.outer(r_xi, n3_qd1) + np.outer(0.5 * d2_xi, m1_qd1) - np.outer(0.5 * d1_xi, m2_qd1) - 0.5 * m2 * NN_xii) * qwi \
                                                  + NN_xii.T @ (np.outer(0.5 * d1, m2_qd1) + 0.5 * m2 * NNi - np.outer(0.5 * d2, m1_qd1)) * qwi
            Ke[self.d3DOF[:, None], self.d2DOF] -= NNi.T @ (np.outer(r_xi, n3_qd2) + np.outer(0.5 * d2_xi, m1_qd2) + 0.5 * m1 * NN_xii - np.outer(0.5 * d1_xi, m2_qd2)) * qwi \
                                                  + NN_xii.T @ (np.outer(0.5 * d1, m2_qd2) - np.outer(0.5 * d2, m1_qd2) - 0.5 * m1 * NNi) * qwi
            Ke[self.d3DOF[:, None], self.d3DOF] -= NNi.T @ (np.outer(r_xi, n3_qd3) + np.outer(0.5 * d2_xi, m1_qd3) - np.outer(0.5 * d1_xi, m2_qd3)) * qwi \
                                                  + NN_xii.T @ (np.outer(0.5 * d1, m2_qd3) - np.outer(0.5 * d2, m1_qd3)) * qwi

        return Ke

        # Ke_num = Numerical_derivative(lambda t, qe: self.f_pot_el(qe, el), order=2)._x(0, qe)
        # diff = Ke_num - Ke
        # error = np.max(np.abs(diff))
        # print(f'max error f_pot_q_el: {error}')
        
        # # # print(f'diff[self.rDOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.rDOF]))}')
        # # # print(f'diff[self.rDOF[:, None], self.d1DOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.d1DOF]))}')
        # # # print(f'diff[self.rDOF[:, None], self.d2DOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.d2DOF]))}')
        # # # print(f'diff[self.rDOF[:, None], self.d3DOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.d3DOF]))}')
        
        # # # print(f'diff[self.d1DOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.d1DOF[:, None], self.rDOF]))}')
        # # # print(f'diff[self.d1DOF[:, None], self.d1DOF]: {np.max(np.abs(diff[self.d1DOF[:, None], self.d1DOF]))}')
        # # # print(f'diff[self.d1DOF[:, None], self.d2DOF]: {np.max(np.abs(diff[self.d1DOF[:, None], self.d2DOF]))}')
        # # # print(f'diff[self.d1DOF[:, None], self.d3DOF]: {np.max(np.abs(diff[self.d1DOF[:, None], self.d3DOF]))}')

        # # # print(f'diff[self.d2DOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.d2DOF[:, None], self.rDOF]))}')
        # # # print(f'diff[self.d2DOF[:, None], self.d1DOF]: {np.max(np.abs(diff[self.d2DOF[:, None], self.d1DOF]))}')
        # # # print(f'diff[self.d2DOF[:, None], self.d2DOF]: {np.max(np.abs(diff[self.d2DOF[:, None], self.d2DOF]))}')
        # # # print(f'diff[self.d2DOF[:, None], self.d3DOF]: {np.max(np.abs(diff[self.d2DOF[:, None], self.d3DOF]))}')

        # # # print(f'diff[self.d3DOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.d3DOF[:, None], self.rDOF]))}')
        # # # print(f'diff[self.d3DOF[:, None], self.d1DOF]: {np.max(np.abs(diff[self.d3DOF[:, None], self.d1DOF]))}')
        # # # print(f'diff[self.d3DOF[:, None], self.d2DOF]: {np.max(np.abs(diff[self.d3DOF[:, None], self.d2DOF]))}')
        # # # print(f'diff[self.d3DOF[:, None], self.d3DOF]: {np.max(np.abs(diff[self.d3DOF[:, None], self.d3DOF]))}')

        # return Ke_num

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

    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[-1]
        else:
            N, _ = self.basis_functions(frame_ID[0])
            NN = self.stack_shapefunctions(N)
        return NN @ q[self.rDOF] + self.A_IK(t, q, frame_ID=frame_ID) @ K_r_SP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[-1]
        else:
            N, _ = self.basis_functions(frame_ID[0])
            NN = self.stack_shapefunctions(N)

        r_OP_q = np.zeros((3, self.nq_el))
        r_OP_q[:, self.rDOF] = NN

        tmp = np.einsum('ijk,j->ik', self.A_IK_q(t, q, frame_ID=frame_ID), K_r_SP)
        r_OP_q[:, self.d1DOF] = tmp[:, self.d1DOF]
        r_OP_q[:, self.d2DOF] = tmp[:, self.d2DOF]
        r_OP_q[:, self.d3DOF] = tmp[:, self.d3DOF]
        return r_OP_q

        # r_OP_q_num = Numerical_derivative(lambda t, q: self.r_OP(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP))._x(t, q)
        # error = np.max(np.abs(r_OP_q_num - r_OP_q))
        # print(f'error in r_OP_q: {error}')
        # return r_OP_q_num

    def A_IK(self, t, q, frame_ID):
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[-1]
        else:
            N, _ = self.basis_functions(frame_ID[0])
            NN = self.stack_shapefunctions(N)

        d1 = NN @ q[self.d1DOF]
        d2 = NN @ q[self.d2DOF]
        d3 = NN @ q[self.d3DOF]
        return np.vstack((d1, d2, d3)).T

    def A_IK_q(self, t, q, frame_ID):
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[-1]
        else:
            N, _ = self.basis_functions(frame_ID[0])
            NN = self.stack_shapefunctions(N)

        A_IK_q = np.zeros((3, 3, self.nq_el))
        A_IK_q[:, 0, self.d1DOF] = NN
        A_IK_q[:, 1, self.d2DOF] = NN
        A_IK_q[:, 2, self.d3DOF] = NN
        return A_IK_q

        # A_IK_q_num =  Numerical_derivative(lambda t, q: self.A_IK(t, q, frame_ID=frame_ID))._x(t, q)
        # error = np.linalg.norm(A_IK_q - A_IK_q_num)
        # print(f'error in A_IK_q: {error}')
        # return A_IK_q_num

    def v_P(self, t, q, u, frame_ID, K_r_SP=None):
        # raise NotImplementedError('not tested!')
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[-1]
        else:
            N, _ = self.basis_functions(frame_ID[0])
            NN = self.stack_shapefunctions(N)
        
        # v_P1 = NN @ u[self.rDOF] + self.A_IK(t, q, frame_ID) @ cross3(self.K_Omega(t, q, u, frame_ID=frame_ID), K_r_SP)
        v_P2 = NN @ u[self.rDOF] + self.A_IK(t, u, frame_ID) @ K_r_SP
        # print(v_P1 - v_P2)
        return v_P2

    def v_P_q(self, t, q, u, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_el))

    def J_P(self, t, q, frame_ID, K_r_SP=None):
        return self.r_OP_q(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP)

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.zeros((3, self.nq_el, self.nq_el))

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        # raise NotImplementedError('not tested!')
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[-1]
        else:
            N, _ = self.basis_functions(frame_ID[0])
            NN = self.stack_shapefunctions(N)
        
        # TODO: simplifications?
        # K_Omega = self.K_Omega(t, q, u, frame_ID=frame_ID)
        # K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID=frame_ID)
        # a_P1 = NN @ u_dot[self.rDOF] + self.A_IK(t, q, frame_ID=frame_ID) @ (cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP)))
        a_P2 = NN @ u_dot[self.rDOF] + self.A_IK(t, u_dot, frame_ID=frame_ID) @ K_r_SP
        # print(a_P1 - a_P2)
        return a_P2

    def a_P_q(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_el))
    
    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_el))

    def K_Omega(self, t, q, u, frame_ID):
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[-1]
        else:
            N, _ = self.basis_functions(frame_ID[0])
            NN = self.stack_shapefunctions(N)

        d1 = NN @ q[self.d1DOF]
        d2 = NN @ q[self.d2DOF]
        d3 = NN @ q[self.d3DOF]
        A_IK = np.vstack((d1, d2, d3)).T

        d1_dot = NN @ u[self.d1DOF]
        d2_dot = NN @ u[self.d2DOF]
        d3_dot = NN @ u[self.d3DOF]
        A_IK_dot = np.vstack((d1_dot, d2_dot, d3_dot)).T

        K_Omega_tilde = A_IK.T @ A_IK_dot
        return skew2ax(K_Omega_tilde)

    def K_J_R(self, t, q, frame_ID):
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[-1]
        else:
            N, _ = self.basis_functions(frame_ID[0])
            NN = self.stack_shapefunctions(N)

        d1 = NN @ q[self.d1DOF]
        d2 = NN @ q[self.d2DOF]
        d3 = NN @ q[self.d3DOF]
        A_IK = np.vstack((d1, d2, d3)).T

        K_Omega_tilde_Omega_tilde = skew2ax_A()

        K_J_R = np.zeros((3, self.nq_el))
        K_J_R[:, self.d1DOF] = K_Omega_tilde_Omega_tilde[0] @ A_IK.T @ NN
        K_J_R[:, self.d2DOF] = K_Omega_tilde_Omega_tilde[1] @ A_IK.T @ NN
        K_J_R[:, self.d3DOF] = K_Omega_tilde_Omega_tilde[2] @ A_IK.T @ NN
        return K_J_R

        # K_J_R_num = Numerical_derivative(lambda t, q, u: self.K_Omega(t, q, u, frame_ID=frame_ID), order=2)._y(t, q, np.zeros_like(q))
        # diff = K_J_R_num - K_J_R
        # diff_error = diff
        # error = np.linalg.norm(diff_error)
        # print(f'error K_J_R: {error}')
        # return K_J_R_num

    def K_J_R_q(self, t, q, frame_ID):
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[-1]
        else:
            N, _ = self.basis_functions(frame_ID[0])
            NN = self.stack_shapefunctions(N)

        A_IK_q = np.zeros((3, 3, self.nq_el))
        A_IK_q[:, 0, self.d1DOF] = NN
        A_IK_q[:, 1, self.d2DOF] = NN
        A_IK_q[:, 2, self.d3DOF] = NN

        K_Omega_tilde_Omega_tilde = skew2ax_A()
        tmp = np.einsum('jil,jk->ikl', A_IK_q, NN)

        K_J_R_q = np.zeros((3, self.nq_el, self.nq_el))
        K_J_R_q[:, self.d1DOF] = np.einsum('ij,jkl->ikl', K_Omega_tilde_Omega_tilde[0], tmp)
        K_J_R_q[:, self.d2DOF] = np.einsum('ij,jkl->ikl', K_Omega_tilde_Omega_tilde[1], tmp)
        K_J_R_q[:, self.d3DOF] = np.einsum('ij,jkl->ikl', K_Omega_tilde_Omega_tilde[2], tmp)
        return K_J_R_q

        # K_J_R_q_num = Numerical_derivative(lambda t, q: self.K_J_R(t, q, frame_ID))._x(t, q)
        # error = np.max(np.abs(K_J_R_q_num - K_J_R_q))
        # print(f'error K_J_R_q: {error}')
        # return K_J_R_q_num

    def K_Psi(self, t, q, u, u_dot, frame_ID):
        # raise NotImplementedError('not tested!')
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[-1]
        else:
            N, _ = self.basis_functions(frame_ID[0])
            NN = self.stack_shapefunctions(N)

        d1 = NN @ q[self.d1DOF]
        d2 = NN @ q[self.d2DOF]
        d3 = NN @ q[self.d3DOF]
        A_IK = np.vstack((d1, d2, d3)).T

        d1_dot = NN @ u[self.d1DOF]
        d2_dot = NN @ u[self.d2DOF]
        d3_dot = NN @ u[self.d3DOF]
        A_IK_dot = np.vstack((d1_dot, d2_dot, d3_dot)).T

        d1_ddot = NN @ u_dot[self.d1DOF]
        d2_ddot = NN @ u_dot[self.d2DOF]
        d3_ddot = NN @ u_dot[self.d3DOF]
        A_IK_ddot = np.vstack((d1_ddot, d2_ddot, d3_ddot)).T

        K_Psi_tilde = A_IK_dot.T @ A_IK_dot + A_IK.T @ A_IK_ddot
        return skew2ax(K_Psi_tilde)

    ####################################################
    # body force
    ####################################################
    def body_force_el(self, force, t, N, xi, J0, qw):
        fe = np.zeros(self.nq_el)
        for Ni, xii, J0i, qwi in zip(N, xi, J0, qw):
            NNi = self.stack_shapefunctions(Ni)
            fe[self.rDOF] += NNi.T @ force(xii, t) * J0i * qwi
        return fe

    def body_force(self, t, q, force):
        f = np.zeros(self.nq)
        for el in range(self.nEl):
            f[self.elDOF[el]] += self.body_force_el(force, t, self.N[el], self.xi[el], self.J0[el], self.qw[el])
        return f

    def body_force_q(self, t, q, coo, force):
        pass

    ##############################################################
    # abstract methods for bilateral constraints on position level
    ##############################################################
    @abstractmethod
    def g(self, t, q):
        pass

    @abstractmethod
    def g_q(self, t, q, coo):
        pass

    @abstractmethod
    def W_g(self, t, q, coo):
        pass

    @abstractmethod
    def Wla_g_q(self, t, q, la_g, coo):
        pass

    ####################################################
    # visualization
    ####################################################
    def nodes(self, q):
        return q[self.qDOF][:3*self.nn].reshape(3, -1)

    def centerline(self, q, n=100):
        q_body = q[self.qDOF]
        r = []
        for xi in np.linspace(0, 1, n):
            frame_ID = (xi,)
            qp = q_body[self.qDOF_P(frame_ID)]
            r.append( self.r_OP(1, qp, frame_ID) )
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
            r.append( self.r_OP(1, qp, frame_ID) )

            d1i, d2i, d3i = self.A_IK(1, qp, frame_ID).T
            d1.extend([d1i])
            d2.extend([d2i])
            d3.extend([d3i])

        return np.array(r).T, np.array(d1).T, np.array(d2).T, np.array(d3).T

    def plot_centerline(self, ax, q, n=100, color='black'):
        ax.plot(*self.nodes(q), linestyle='dashed', marker='o', color=color)
        ax.plot(*self.centerline(q, n=n), linestyle='solid', color=color)

    def plot_frames(self, ax, q, n=10, length=1):
        r, d1, d2, d3 = self.frames(q, n=n)
        ax.quiver(*r, *d1, color='red', length=length)
        ax.quiver(*r, *d2, color='green', length=length)
        ax.quiver(*r, *d3, color='blue', length=length)

    ############
    # vtk export
    ############
    def post_processing(self, t, q, filename, binary=True):
        # write paraview PVD file collecting time and all vtk files, see https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
        from xml.dom import minidom
        
        root = minidom.Document()
        
        vkt_file = root.createElement('VTKFile')
        vkt_file.setAttribute('type', 'Collection')
        root.appendChild(vkt_file)
        
        collection = root.createElement('Collection')
        vkt_file.appendChild(collection)

        for i, (ti, qi) in enumerate(zip(t, q)):
            filei = filename + f'{i}.vtu'

            # write time step and file name in pvd file
            dataset = root.createElement('DataSet')
            dataset.setAttribute('timestep', f'{ti:0.6f}')
            dataset.setAttribute('file', filei)
            collection.appendChild(dataset)

            self.post_processing_single_configuration(ti, qi, filei, binary=binary)

        # write pvd file        
        xml_str = root.toprettyxml(indent ="\t")          
        with open(filename + '.pvd', "w") as f:
            f.write(xml_str)

    def post_processing_single_configuration(self, t, q, filename, binary=True):
        # centerline and connectivity + director data
        cells, points, HigherOrderDegrees = self.mesh_kinematics.vtk_mesh(q)

        # fill dictionary storing point data with directors
        point_data = {
            "d1": points[:, 3:6],
            "d2": points[:, 6:9],
            "d3": points[:, 9:12],
        }

        # export existing values on quadrature points using L2 projection
        J0_vtk = self.mesh_kinematics.field_to_vtk(self.J0.reshape(self.nEl, self.nQP, 1))
        point_data.update({"J0": J0_vtk})
        
        Gamma0_vtk = self.mesh_kinematics.field_to_vtk(self.Gamma0)
        point_data.update({"Gamma0": Gamma0_vtk})
        
        Kappa0_vtk = self.mesh_kinematics.field_to_vtk(self.Kappa0)
        point_data.update({"Kappa0": Kappa0_vtk})

        # evaluate strain measures at quadrature points
        Gamma = np.zeros((self.mesh_kinematics.nel, self.mesh_kinematics.nqp, 3))
        Kappa = np.zeros((self.mesh_kinematics.nel, self.mesh_kinematics.nqp, 3))
        for el in range(self.mesh_kinematics.nel):
            qe = q[self.elDOF[el]]
            N, N_xi, Gamma0, Kappa0, J0, qw = self.N[el], self.N_xi[el], self.Gamma0[el], self.Kappa0[el], self.J0[el], self.qw[el]

            # extract generalized coordinates for beam centerline and directors 
            # in the current and reference configuration
            qe_r = qe[self.rDOF]
            qe_d1 = qe[self.d1DOF]
            qe_d2 = qe[self.d2DOF]
            qe_d3 = qe[self.d3DOF]

            # integrate element force vector
            for i, (Ni, N_xii, Gamma0_i, Kappa0_i, J0i, qwi) in enumerate(zip(N, N_xi, Gamma0, Kappa0, J0, qw)):
                # build matrix of shape function derivatives
                NNi = self.stack_shapefunctions(Ni)
                NN_xii = self.stack_shapefunctions(N_xii)

                # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
                # the parameter space \xi and thus need to be transformed later
                r_xi = NN_xii @ qe_r
                d1 = NNi @ qe_d1
                d1_xi = NN_xii @ qe_d1
                d2 = NNi @ qe_d2
                d2_xi = NN_xii @ qe_d2
                d3 = NNi @ qe_d3
                d3_xi = NN_xii @ qe_d3
                
                # compute derivatives w.r.t. the arc lenght parameter s
                r_s = r_xi / J0i
                d1_s = d1_xi / J0i
                d2_s = d2_xi / J0i
                d3_s = d3_xi / J0i

                # build rotation matrices
                R = np.vstack((d1, d2, d3)).T
                
                # axial and shear strains
                Gamma[el, i] = R.T @ r_s

                # torsional and flexural strains
                Kappa[el, i] = np.array([0.5 * (d3 @ d2_s - d2 @ d3_s), \
                                    0.5 * (d1 @ d3_s - d3 @ d1_s), \
                                    0.5 * (d2 @ d1_s - d1 @ d2_s)])
        
        # L2 projection of strain measures
        Gamma_vtk = self.mesh_kinematics.field_to_vtk(Gamma)
        point_data.update({"Gamma": Gamma_vtk})
        
        Kappa_vtk = self.mesh_kinematics.field_to_vtk(Kappa)
        point_data.update({"Kappa": Kappa_vtk})

        # fields depending on strain measures and other previously computed quantities
        point_data_fields = {
            "W": lambda Gamma, Gamma0, Kappa, Kappa0: np.array([self.material_model.potential(Gamma, Gamma0, Kappa, Kappa0)]),
            "n_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.n_i(Gamma, Gamma0, Kappa, Kappa0),
            "m_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.m_i(Gamma, Gamma0, Kappa, Kappa0)
        }

        for name, fun in point_data_fields.items():
            tmp = fun(Gamma_vtk[0], Gamma0_vtk[0], Kappa_vtk[0], Kappa0_vtk[0]).reshape(-1)
            field = np.zeros((len(Gamma_vtk), len(tmp)))
            for i, (Gamma_i, Gamma0_i, Kappa_i, Kappa0_i) in enumerate(zip(Gamma_vtk, Gamma0_vtk, Kappa_vtk, Kappa0_vtk)):
                field[i] = fun(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i).reshape(-1)
            point_data.update({name: field})

        # write vtk mesh using meshio
        meshio.write_points_cells(
            os.path.splitext(os.path.basename(filename))[0] + '.vtu',
            points[:, :3], # only export centerline as geometry here!
            cells,
            point_data=point_data,
            cell_data={"HigherOrderDegrees": HigherOrderDegrees},
            binary=binary
        )

####################################################
# straight initial configuration
####################################################
def straight_configuration(polynomial_degree, nEl, L, greville_abscissae=True, r_OP=np.zeros(3), A_IK=np.eye(3)):
    nn = polynomial_degree + nEl
    
    X = np.linspace(0, L, num=nn)
    Y = np.zeros(nn)
    Z = np.zeros(nn)
    if greville_abscissae:
        kv = uniform_knot_vector(polynomial_degree, nEl)
        for i in range(nn):
            X[i] = np.sum(kv[i+1:i+polynomial_degree+1])
        X = X * L / polynomial_degree

    r0 = np.vstack((X, Y, Z)).T
    for i, r0i in enumerate(r0):
        X[i], Y[i], Z[i] = r_OP + A_IK @ r0i
    
    # compute reference directors
    D1, D2, D3 = A_IK.T
    D1 = np.repeat(D1, nn)
    D2 = np.repeat(D2, nn)
    D3 = np.repeat(D3, nn)
    
    # assemble all reference generalized coordinates
    return np.concatenate([X, Y, Z, D1, D2, D3])

####################################################
# constraint beam using nodal constraints
####################################################
class Timoshenko_director_dirac(Timoshenko_beam_director):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nla_g = 6 * self.nn
        la_g0 = kwargs.get('la_g0')
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        # nodal degrees of freedom
        self.rDOF_node = np.arange(3)
        self.d1DOF_node = np.arange(3, 6)
        self.d2DOF_node = np.arange(6, 9)
        self.d3DOF_node = np.arange(9, 12)

    # constraints on a single node
    def __g(self, qn):
        g = np.zeros(6)

        d1 = qn[self.d1DOF_node]
        d2 = qn[self.d2DOF_node]
        d3 = qn[self.d3DOF_node]

        g[0] = d1 @ d1 - 1
        g[1] = d2 @ d2 - 1
        g[2] = d3 @ d3 - 1
        g[3] = d1 @ d2
        g[4] = d1 @ d3
        g[5] = d2 @ d3

        return g

    def __g_q(self, qn):
        g_q = np.zeros((6, 12))

        d1 = qn[self.d1DOF_node]
        d2 = qn[self.d2DOF_node]
        d3 = qn[self.d3DOF_node]

        g_q[0, self.d1DOF_node] = 2 * d1
        g_q[1, self.d2DOF_node] = 2 * d2
        g_q[2, self.d3DOF_node] = 2 * d3
        g_q[3, self.d1DOF_node] = d2
        g_q[3, self.d2DOF_node] = d1
        g_q[4, self.d1DOF_node] = d3
        g_q[4, self.d3DOF_node] = d1
        g_q[5, self.d2DOF_node] = d3
        g_q[5, self.d3DOF_node] = d2

        return g_q
    
    def __g_qq(self):
        g_qq = np.zeros((6, 12, 12))

        eye3 = np.eye(3)

        g_qq[0, self.d1DOF_node[:, None], self.d1DOF_node] = 2 * eye3
        g_qq[1, self.d2DOF_node[:, None], self.d2DOF_node] = 2 * eye3
        g_qq[2, self.d3DOF_node[:, None], self.d3DOF_node] = 2 * eye3
        
        g_qq[3, self.d1DOF_node[:, None], self.d2DOF_node] = eye3
        g_qq[3, self.d2DOF_node[:, None], self.d1DOF_node] = eye3

        g_qq[4, self.d1DOF_node[:, None], self.d3DOF_node] = eye3
        g_qq[4, self.d3DOF_node[:, None], self.d3DOF_node] = eye3
        
        g_qq[5, self.d2DOF_node[:, None], self.d3DOF_node] = eye3
        g_qq[5, self.d3DOF_node[:, None], self.d2DOF_node] = eye3
        
        return g_qq

    # global constraint functions
    def g(self, t, q):
        g = np.zeros(self.nla_g)
        for i, DOF in enumerate(self.nodalDOF):
            idx = i * 6
            g[idx:idx+6] = self.__g(q[DOF])
        return g

    def g_q(self, t, q, coo):
        for i, DOF in enumerate(self.nodalDOF):
            idx = i * 6
            coo.extend(self.__g_q(q[DOF]), (self.la_gDOF[np.arange(idx, idx+6)], self.qDOF[DOF]))

    def W_g(self, t, q, coo):
        for i, DOF in enumerate(self.nodalDOF):
            idx = i * 6
            coo.extend(self.__g_q(q[DOF]).T, (self.uDOF[DOF], self.la_gDOF[np.arange(idx, idx+6)]))

    def Wla_g_q(self, t, q, la_g, coo):
        for i, DOF in enumerate(self.nodalDOF):
            idx = i * 6
            coo.extend(np.einsum('i,ijk->jk', la_g[idx:idx+6], self.__g_qq()), (self.uDOF[DOF], self.qDOF[DOF]))

####################################################
# constraint beam using integral constraints
####################################################
class Timoshenko_director_integral(Timoshenko_beam_director):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.polynomial_degree_g = self.polynomial_degree
        self.nn_el_g = self.polynomial_degree_g + 1 # number of nodes per element
        self.nn_g = self.nEl + self.polynomial_degree_g # number of nodes
        self.nq_n_g = 6 # number of degrees of freedom per node
        self.nla_g = self.nn_g * self.nq_n_g
        self.nla_g_el = self.nn_el_g * self.nq_n_g

        la_g0 = kwargs.get('la_g0')
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0
        self.knot_vector_g = Knot_vector(self.polynomial_degree_g, self.nEl)
        self.element_span_g = self.knot_vector_g.data[self.polynomial_degree_g:-self.polynomial_degree_g]
        self.mesh_g = Mesh1D(self.knot_vector_g, self.nQP, derivative_order=0, nq_n=self.nq_n_g)
        self.elDOF_g = self.mesh_g.elDOF

        # compute shape functions
        self.N_g = self.mesh_g.N
            
        # degrees of freedom on the element level (for the constraints)
        self.g11DOF = np.arange(self.nn_el_g)
        self.g12DOF = np.arange(self.nn_el_g,   2*self.nn_el_g)
        self.g13DOF = np.arange(2*self.nn_el_g, 3*self.nn_el_g)
        self.g22DOF = np.arange(3*self.nn_el_g, 4*self.nn_el_g)
        self.g23DOF = np.arange(4*self.nn_el_g, 5*self.nn_el_g)
        self.g33DOF = np.arange(5*self.nn_el_g, 6*self.nn_el_g)

    def __g_el(self, qe, el):
        g = np.zeros(self.nla_g_el)

        N, N_g , J0, qw = self.N[el], self.N_g[el], self.J0[el], self.qw[el]
        for Ni, N_gi, J0i, qwi in zip(N, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)

            d1 = NNi @ qe[self.d1DOF]
            d2 = NNi @ qe[self.d2DOF]
            d3 = NNi @ qe[self.d3DOF]

            factor = N_gi * J0i * qwi
         
            g[self.g11DOF] += (d1 @ d1 - 1) * factor
            g[self.g12DOF] += (d1 @ d2)     * factor
            g[self.g13DOF] += (d1 @ d3)     * factor
            g[self.g22DOF] += (d2 @ d2 - 1) * factor
            g[self.g23DOF] += (d2 @ d3)     * factor
            g[self.g33DOF] += (d3 @ d3 - 1) * factor

        return g

    def __g_q_el(self, qe, el):
        g_q = np.zeros((self.nla_g_el, self.nq_el))

        N, N_g , J0, qw = self.N[el], self.N_g[el], self.J0[el], self.qw[el]
        for Ni, N_gi, J0i, qwi in zip(N, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)

            d1 = NNi @ qe[self.d1DOF]
            d2 = NNi @ qe[self.d2DOF]
            d3 = NNi @ qe[self.d3DOF]

            factor = N_gi * J0i * qwi
            d1_NNi = d1 @ NNi
            d2_NNi = d2 @ NNi
            d3_NNi = d3 @ NNi

            # 2 * delta d1 * d1
            g_q[self.g11DOF[:, None], self.d1DOF] += np.outer(factor, 2 * d1_NNi)
            # delta d1 * d2
            g_q[self.g12DOF[:, None], self.d1DOF] += np.outer(factor, d2_NNi)
            # delta d2 * d1
            g_q[self.g12DOF[:, None], self.d2DOF] += np.outer(factor, d1_NNi)
            # delta d1 * d3
            g_q[self.g13DOF[:, None], self.d1DOF] += np.outer(factor, d3_NNi)
            # delta d3 * d1
            g_q[self.g13DOF[:, None], self.d3DOF] += np.outer(factor, d1_NNi)

            # 2 * delta d2 * d2
            g_q[self.g22DOF[:, None], self.d2DOF] += np.outer(factor, 2 * d2_NNi)
            # delta d2 * d3
            g_q[self.g23DOF[:, None], self.d2DOF] += np.outer(factor, d3_NNi)
            # delta d3 * d2
            g_q[self.g23DOF[:, None], self.d3DOF] += np.outer(factor, d2_NNi)

            # 2 * delta d3 * d3
            g_q[self.g33DOF[:, None], self.d3DOF] += np.outer(factor, 2 * d3_NNi)

        return g_q

        # g_q_num = Numerical_derivative(lambda t, q: self.__g_el(q, N, N_g, J0, qw), order=2)._x(0, qe)
        # diff = g_q_num - g_q
        # error = np.linalg.norm(diff)
        # print(f'error g_q: {error}')
        # return g_q_num

    def __g_qq_el(self, el):
        g_qq = np.zeros((self.nla_g_el, self.nq_el, self.nq_el))

        N, N_g , J0, qw = self.N[el], self.N_g[el], self.J0[el], self.qw[el]
        for Ni, N_gi, J0i, qwi in zip(N, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)
            factor = NNi.T @ NNi * J0i * qwi

            for j, N_gij in enumerate(N_gi):
                N_gij_factor = N_gij * factor

                # 2 * delta d1 * d1
                g_qq[self.g11DOF[j], self.d1DOF[:, None], self.d1DOF] += 2 * N_gij_factor
                # delta d1 * d2
                g_qq[self.g12DOF[j], self.d1DOF[:, None], self.d2DOF] += N_gij_factor
                # delta d2 * d1
                g_qq[self.g12DOF[j], self.d2DOF[:, None], self.d1DOF] += N_gij_factor
                # delta d1 * d3
                g_qq[self.g13DOF[j], self.d1DOF[:, None], self.d3DOF] += N_gij_factor
                # delta d3 * d1
                g_qq[self.g13DOF[j], self.d3DOF[:, None], self.d1DOF] += N_gij_factor

                # 2 * delta d2 * d2
                g_qq[self.g22DOF[j], self.d2DOF[:, None], self.d2DOF] += 2 * N_gij_factor
                # delta d2 * d3
                g_qq[self.g23DOF[j], self.d2DOF[:, None], self.d3DOF] += N_gij_factor
                # delta d3 * d2
                g_qq[self.g23DOF[j], self.d3DOF[:, None], self.d2DOF] += N_gij_factor

                # 2 * delta d3 * d3
                g_qq[self.g33DOF[j], self.d3DOF[:, None], self.d3DOF] += 2 * N_gij_factor

        return g_qq

        # g_qq_num = Numerical_derivative(lambda t, q: self.__g_q_el(q, N, N_g, J0, qw), order=2)._x(0, qe)
        # diff = g_qq_num - g_qq
        # error = np.linalg.norm(diff)
        # print(f'error g_qq: {error}')
        # return g_qq_num

    def __g_dot_el(self, qe, ue, el):
        g_dot = np.zeros(self.nla_g_el)

        N, N_g , J0, qw = self.N[el], self.N_g[el], self.J0[el], self.qw[el]
        for Ni, N_gi, J0i, qwi in zip(N, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)

            d1 = NNi @ qe[self.d1DOF]
            d2 = NNi @ qe[self.d2DOF]
            d3 = NNi @ qe[self.d3DOF]
            d1_dot = NNi @ ue[self.d1DOF]
            d2_dot = NNi @ ue[self.d2DOF]
            d3_dot = NNi @ ue[self.d3DOF]

            factor = N_gi * J0i * qwi
         
            g_dot[self.g11DOF] += 2 * d1 @ d1_dot * factor
            g_dot[self.g12DOF] += (d1 @ d2_dot + d2 @ d1_dot) * factor
            g_dot[self.g13DOF] += (d1 @ d3_dot + d3 @ d1_dot) * factor

            g_dot[self.g22DOF] += 2 * d2 @ d2_dot * factor
            g_dot[self.g23DOF] += (d2 @ d3_dot + d3 @ d2_dot) * factor

            g_dot[self.g33DOF] += 2 * d3 @ d3_dot * factor

        return g_dot
        
        # g_dot_num = Numerical_derivative(lambda t, q, u: self.__g_el(q, el))._t(0, qe, ue)
        # g_dot_num += Numerical_derivative(lambda t, q, u: self.__g_el(q, el))._x(0, qe, ue) @ ue
        # diff = g_dot_num - g_dot
        # error = np.linalg.norm(diff)
        # print(f'error g_dot: {error}')
        # return g_dot_num

    def __g_dot_q_el(self, qe, ue, el):
        g_dot_q = np.zeros((self.nla_g_el, self.nq_el))

        N, N_g , J0, qw = self.N[el], self.N_g[el], self.J0[el], self.qw[el]
        for Ni, N_gi, J0i, qwi in zip(N, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)

            d1_dot = NNi @ ue[self.d1DOF]
            d2_dot = NNi @ ue[self.d2DOF]
            d3_dot = NNi @ ue[self.d3DOF]

            factor = N_gi * J0i * qwi
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

        # g_dot_q_num = Numerical_derivative(lambda t, q, u: self.__g_dot_el(q, u, el), order=2)._x(0, qe, ue)
        # diff = g_dot_q - g_dot_q_num
        # error = np.linalg.norm(diff)
        # print(f'error g_dot_q: {error}')
        # return g_dot_q_num

    def __g_ddot_el(self, qe, ue, ue_dot, el):
        g_ddot = np.zeros(self.nla_g_el)

        N, N_g , J0, qw = self.N[el], self.N_g[el], self.J0[el], self.qw[el]
        for Ni, N_gi, J0i, qwi in zip(N, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)

            d1 = NNi @ qe[self.d1DOF]
            d2 = NNi @ qe[self.d2DOF]
            d3 = NNi @ qe[self.d3DOF]
            d1_dot = NNi @ ue[self.d1DOF]
            d2_dot = NNi @ ue[self.d2DOF]
            d3_dot = NNi @ ue[self.d3DOF]
            d1_ddot = NNi @ ue_dot[self.d1DOF]
            d2_ddot = NNi @ ue_dot[self.d2DOF]
            d3_ddot = NNi @ ue_dot[self.d3DOF]

            factor = N_gi * J0i * qwi
         
            g_ddot[self.g11DOF] += 2 * (d1 @ d1_ddot + d1_dot @ d1_dot) * factor
            g_ddot[self.g12DOF] += (d1 @ d2_ddot + 2 * d1_dot @ d2_dot + d2 @ d1_ddot) * factor
            g_ddot[self.g13DOF] += (d1 @ d3_ddot + 2 * d1_dot @ d3_dot + d3 @ d1_ddot) * factor

            g_ddot[self.g22DOF] += 2 * (d2 @ d2_ddot + d2_dot @ d2_dot) * factor
            g_ddot[self.g23DOF] += (d2 @ d3_ddot + 2 * d2_dot @ d3_dot + d3 @ d2_ddot) * factor

            g_ddot[self.g33DOF] += 2 * (d3 @ d3_ddot + d3_dot @ d3_dot) * factor

        return g_ddot

        # g_ddot_num = Numerical_derivative(lambda t, q, u: self.__g_dot_el(q, u, el))._t(0, qe, ue)
        # g_ddot_num += Numerical_derivative(lambda t, q, u: self.__g_dot_el(q, u, el))._x(0, qe, ue) @ ue
        # g_ddot_num += Numerical_derivative(lambda t, q, u: self.__g_dot_el(q, u, el))._y(0, qe, ue) @ ue_dot
        # diff = g_ddot_num - g_ddot
        # error = np.linalg.norm(diff)
        # print(f'error g_ddot: {error}')
        # return g_ddot_num

    def __g_ddot_q_el(self, qe, ue, ue_dot, el):
        g_ddot_q = np.zeros((self.nla_g_el, self.nq_el))

        N, N_g , J0, qw = self.N[el], self.N_g[el], self.J0[el], self.qw[el]
        for Ni, N_gi, J0i, qwi in zip(N, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)

            d1_ddot = NNi @ ue_dot[self.d1DOF]
            d2_ddot = NNi @ ue_dot[self.d2DOF]
            d3_ddot = NNi @ ue_dot[self.d3DOF]

            factor = N_gi * J0i * qwi
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

        # g_ddot_q_num = Numerical_derivative(lambda t, q, u: self.__g_ddot_el(q, u, ue_dot, el), order=2)._x(0, qe, ue)
        # diff = g_ddot_q - g_ddot_q_num
        # diff_error = diff #[self.g11DOF]
        # error = np.linalg.norm(diff_error)
        # print(f'error g_ddot_q: {error}')
        # return g_ddot_q_num

    def __g_ddot_u_el(self, qe, ue, ue_dot, el):
        g_ddot_u = np.zeros((self.nla_g_el, self.nq_el))

        N, N_g , J0, qw = self.N[el], self.N_g[el], self.J0[el], self.qw[el]
        for Ni, N_gi, J0i, qwi in zip(N, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)

            d1_dot = NNi @ ue[self.d1DOF]
            d2_dot = NNi @ ue[self.d2DOF]
            d3_dot = NNi @ ue[self.d3DOF]

            factor = N_gi * J0i * qwi
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

        # g_ddot_u_num = Numerical_derivative(lambda t, q, u: self.__g_ddot_el(q, u, ue_dot, el), order=2)._y(0, qe, ue)
        # diff = g_ddot_u - g_ddot_u_num
        # diff_error = diff
        # error = np.linalg.norm(diff_error)
        # print(f'error g_ddot_u: {error}')
        # return g_ddot_u_num

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
            g_qq = self.__g_qq_el(el)
            coo.extend(np.einsum('i,ijk->jk', la_g[elDOF_g], g_qq), (self.uDOF[elDOF], self.qDOF[elDOF]))

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

class Euler_Bernoulli_director_integral(Timoshenko_beam_director):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.polynomial_degree_g = self.polynomial_degree
        self.nn_el_g = self.polynomial_degree_g + 1 # number of nodes per element
        self.nn_g = self.nEl + self.polynomial_degree_g # number of nodes
        self.nq_n_g = 8 # number of degrees of freedom per node
        self.nla_g = self.nn_g * self.nq_n_g
        self.nla_g_el = self.nn_el_g * self.nq_n_g
        
        la_g0 = kwargs.get('la_g0')
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0
        self.knot_vector_g = Knot_vector(self.polynomial_degree_g, self.nEl)
        self.element_span_g = self.knot_vector_g.data[self.polynomial_degree_g:-self.polynomial_degree_g]
        self.mesh_g = Mesh1D(self.knot_vector_g, self.nQP, derivative_order=0, nq_n=self.nq_n_g)
        self.elDOF_g = self.mesh_g.elDOF

        # compute shape functions
        self.N_g = self.mesh_g.N
            
        # degrees of freedom on the element level (for the constraints)
        self.g11DOF = np.arange(self.nn_el_g)
        self.g12DOF = np.arange(self.nn_el_g,   2*self.nn_el_g)
        self.g13DOF = np.arange(2*self.nn_el_g, 3*self.nn_el_g)
        self.g22DOF = np.arange(3*self.nn_el_g, 4*self.nn_el_g)
        self.g23DOF = np.arange(4*self.nn_el_g, 5*self.nn_el_g)
        self.g33DOF = np.arange(5*self.nn_el_g, 6*self.nn_el_g)
        self.g2DOF = np.arange(6*self.nn_el_g, 7*self.nn_el_g) # unshearability in d2-direction
        self.g3DOF = np.arange(7*self.nn_el_g, 8*self.nn_el_g) # unshearability in d3-direction

    def __g_el(self, qe, N, N_xi, N_g, J0, qw):
        g = np.zeros(self.nla_g_el)

        for Ni, N_xii, N_gi, J0i, qwi in zip(N, N_xi, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)
            NN_xii = self.stack_shapefunctions(N_xii)

            d1 = NNi @ qe[self.d1DOF]
            d2 = NNi @ qe[self.d2DOF]
            d3 = NNi @ qe[self.d3DOF]

            r_xi = NN_xii @ qe[self.rDOF]

            factor1 = N_gi * J0i * qwi
            factor2 = N_gi * qwi
         
            # director constraints
            g[self.g11DOF] += (d1 @ d1 - 1) * factor1
            g[self.g12DOF] += (d1 @ d2)     * factor1
            g[self.g13DOF] += (d1 @ d3)     * factor1
            g[self.g22DOF] += (d2 @ d2 - 1) * factor1
            g[self.g23DOF] += (d2 @ d3)     * factor1
            g[self.g33DOF] += (d3 @ d3 - 1) * factor1

            # unshearability in d2-direction
            g[self.g2DOF] += d2 @ r_xi * factor2
            g[self.g3DOF] += d3 @ r_xi * factor2

        return g

    def __g_q_el(self, qe, N, N_xi, N_g, J0, qw):
        g_q = np.zeros((self.nla_g_el, self.nq_el))

        for Ni, N_xii, N_gi, J0i, qwi in zip(N, N_xi, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)
            NN_xii = self.stack_shapefunctions(N_xii)

            d1 = NNi @ qe[self.d1DOF]
            d2 = NNi @ qe[self.d2DOF]
            d3 = NNi @ qe[self.d3DOF]

            d1_NNi = d1 @ NNi
            d2_NNi = d2 @ NNi
            d3_NNi = d3 @ NNi

            r_xi = NN_xii @ qe[self.rDOF]

            factor1 = N_gi * J0i * qwi
            factor2 = N_gi * qwi
            
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
            g_q[self.g2DOF[:, None], self.rDOF] += np.outer(factor2, d2 @ NN_xii)
            g_q[self.g2DOF[:, None], self.d2DOF] += np.outer(factor2, r_xi @ NNi)
            g_q[self.g3DOF[:, None], self.rDOF] += np.outer(factor2, d3 @ NN_xii)
            g_q[self.g3DOF[:, None], self.d3DOF] += np.outer(factor2, r_xi @ NNi)

        return g_q

        # g_q_num = Numerical_derivative(lambda t, q: self.__g_el(q, N, N_xi, N_g, J0, qw), order=2)._x(0, qe)
        # diff = g_q_num - g_q
        # error = np.linalg.norm(diff)
        # print(f'error g_q: {error}')
        # return g_q_num

    def __g_qq_el(self, qe, N, N_xi, N_g, J0, qw):
        g_qq = np.zeros((self.nla_g_el, self.nq_el, self.nq_el))

        for Ni, N_xii, N_gi, J0i, qwi in zip(N, N_xi, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)
            NN_xii = self.stack_shapefunctions(N_xii)

            factor1 = NNi.T @ NNi * J0i * qwi
            factor2 = N_gi * qwi

            ######################
            # director constraints
            ######################
            for j, N_gij in enumerate(N_gi):
                N_gij_factor1 = N_gij * factor1

                # 2 * delta d1 * d1
                g_qq[self.g11DOF[j], self.d1DOF[:, None], self.d1DOF] += 2 * N_gij_factor1
                # delta d1 * d2
                g_qq[self.g12DOF[j], self.d1DOF[:, None], self.d2DOF] += N_gij_factor1
                # delta d2 * d1
                g_qq[self.g12DOF[j], self.d2DOF[:, None], self.d1DOF] += N_gij_factor1
                # delta d1 * d3
                g_qq[self.g13DOF[j], self.d1DOF[:, None], self.d3DOF] += N_gij_factor1
                # delta d3 * d1
                g_qq[self.g13DOF[j], self.d3DOF[:, None], self.d1DOF] += N_gij_factor1

                # 2 * delta d2 * d2
                g_qq[self.g22DOF[j], self.d2DOF[:, None], self.d2DOF] += 2 * N_gij_factor1
                # delta d2 * d3
                g_qq[self.g23DOF[j], self.d2DOF[:, None], self.d3DOF] += N_gij_factor1
                # delta d3 * d2
                g_qq[self.g23DOF[j], self.d3DOF[:, None], self.d2DOF] += N_gij_factor1

                # 2 * delta d3 * d3
                g_qq[self.g33DOF[j], self.d3DOF[:, None], self.d3DOF] += 2 * N_gij_factor1

            ################################
            # unshearability in d2-direction
            ################################
            arg1 = np.einsum('i,jl,jk->ikl', factor2, NNi, NN_xii)
            arg2 = np.einsum('i,jl,jk->ikl', factor2, NN_xii, NNi)
            g_qq[self.g2DOF[:, None, None], self.rDOF[:, None], self.d2DOF] += arg1
            g_qq[self.g2DOF[:, None, None], self.d2DOF[:, None], self.rDOF] += arg2
            g_qq[self.g3DOF[:, None, None], self.rDOF[:, None], self.d3DOF] += arg1
            g_qq[self.g3DOF[:, None, None], self.d3DOF[:, None], self.rDOF] += arg2

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
            g[elDOF_g] += self.__g_el(q[elDOF], self.N[el], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
        return g

    def g_q(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_q = self.__g_q_el(q[elDOF], self.N[el], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
            coo.extend(g_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))

    def W_g(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_q = self.__g_q_el(q[elDOF], self.N[el], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
            coo.extend(g_q.T, (self.uDOF[elDOF], self.la_gDOF[elDOF_g]))

    def Wla_g_q(self, t, q, la_g, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_qq = self.__g_qq_el(q[elDOF], self.N[el], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
            coo.extend(np.einsum('i,ijk->jk', la_g[elDOF_g], g_qq), (self.uDOF[elDOF], self.qDOF[elDOF]))

class Inextensible_Euler_Bernoulli_director_integral(Timoshenko_beam_director):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.polynomial_degree_g = self.polynomial_degree
        self.nn_el_g = self.polynomial_degree_g + 1 # number of nodes per element
        self.nn_g = self.nEl + self.polynomial_degree_g # number of nodes
        self.nq_n_g = 9 # number of degrees of freedom per node
        self.nla_g = self.nn_g * self.nq_n_g
        self.nla_g_el = self.nn_el_g * self.nq_n_g

        la_g0 = kwargs.get('la_g0')
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0
        self.knot_vector_g = Knot_vector(self.polynomial_degree_g, self.nEl)
        self.element_span_g = self.knot_vector_g.data[self.polynomial_degree_g:-self.polynomial_degree_g]
        self.mesh_g = Mesh1D(self.knot_vector_g, self.nQP, derivative_order=0, nq_n=self.nq_n_g)
        self.elDOF_g = self.mesh_g.elDOF

        # compute shape functions
        self.N_g = self.mesh_g.N
            
        # degrees of freedom on the element level (for the constraints)
        self.g11DOF = np.arange(self.nn_el_g)
        self.g12DOF = np.arange(self.nn_el_g,   2*self.nn_el_g)
        self.g13DOF = np.arange(2*self.nn_el_g, 3*self.nn_el_g)
        self.g22DOF = np.arange(3*self.nn_el_g, 4*self.nn_el_g)
        self.g23DOF = np.arange(4*self.nn_el_g, 5*self.nn_el_g)
        self.g33DOF = np.arange(5*self.nn_el_g, 6*self.nn_el_g)
        self.g2DOF = np.arange(6*self.nn_el_g, 7*self.nn_el_g) # unshearability in d2-direction
        self.g3DOF = np.arange(7*self.nn_el_g, 8*self.nn_el_g) # unshearability in d3-direction
        self.g1DOF = np.arange(8*self.nn_el_g, 9*self.nn_el_g) # inextensibility

    def __g_el(self, qe, N, N_xi, N_g, J0, qw):
        g = np.zeros(self.nla_g_el)

        for Ni, N_xii, N_gi, J0i, qwi in zip(N, N_xi, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)
            NN_xii = self.stack_shapefunctions(N_xii)

            d1 = NNi @ qe[self.d1DOF]
            d2 = NNi @ qe[self.d2DOF]
            d3 = NNi @ qe[self.d3DOF]

            r_xi = NN_xii @ qe[self.rDOF]

            factor1 = N_gi * J0i * qwi
            factor2 = N_gi * qwi
         
            # director constraints
            g[self.g11DOF] += (d1 @ d1 - 1) * factor1
            g[self.g12DOF] += (d1 @ d2)     * factor1
            g[self.g13DOF] += (d1 @ d3)     * factor1
            g[self.g22DOF] += (d2 @ d2 - 1) * factor1
            g[self.g23DOF] += (d2 @ d3)     * factor1
            g[self.g33DOF] += (d3 @ d3 - 1) * factor1

            # unshearability in d2-direction
            g[self.g2DOF] += d2 @ r_xi * factor2
            g[self.g3DOF] += d3 @ r_xi * factor2

            # inextensibility
            g[self.g1DOF] += (d1 @ r_xi - J0i) * factor2

            # g[self.g1DOF] += (d1 @ r_xi / J0i - 1) * factor1

            # r_s = r_xi / J0i
            # g[self.g1DOF] += (r_s @ r_s - 1) * factor1

        return g

    def __g_q_el(self, qe, N, N_xi, N_g, J0, qw):
        g_q = np.zeros((self.nla_g_el, self.nq_el))

        for Ni, N_xii, N_gi, J0i, qwi in zip(N, N_xi, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)
            NN_xii = self.stack_shapefunctions(N_xii)

            d1 = NNi @ qe[self.d1DOF]
            d2 = NNi @ qe[self.d2DOF]
            d3 = NNi @ qe[self.d3DOF]

            d1_NNi = d1 @ NNi
            d2_NNi = d2 @ NNi
            d3_NNi = d3 @ NNi

            r_xi = NN_xii @ qe[self.rDOF]

            factor1 = N_gi * J0i * qwi
            factor2 = N_gi * qwi
            
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
            g_q[self.g2DOF[:, None], self.rDOF] += np.outer(factor2, d2 @ NN_xii)
            g_q[self.g2DOF[:, None], self.d2DOF] += np.outer(factor2, r_xi @ NNi)
            g_q[self.g3DOF[:, None], self.rDOF] += np.outer(factor2, d3 @ NN_xii)
            g_q[self.g3DOF[:, None], self.d3DOF] += np.outer(factor2, r_xi @ NNi)

            #################
            # inextensibility
            #################
            # g[self.g1DOF] += (d1 @ r_xi - J0i) * factor2
            g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor2, d1 @ NN_xii)
            g_q[self.g1DOF[:, None], self.d1DOF] += np.outer(factor2, r_xi @ NNi)
            
            # # g[self.g1DOF] += (d1 @ r_xi / J0i - 1) * factor1
            # g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor1, d1 @ NN_xii / J0i)
            # g_q[self.g1DOF[:, None], self.d1DOF] += np.outer(factor1, r_xi / J0i @ NNi)
            
            # r_s = r_xi / J0i
            # # g[self.g1DOF] += (r_s @ r_s - 1) * factor1
            # g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor1, 2 * r_s @ NN_xii / J0i)

        return g_q

        # g_q_num = Numerical_derivative(lambda t, q: self.__g_el(q, N, N_xi, N_g, J0, qw), order=2)._x(0, qe)
        # # diff = g_q_num - g_q
        # # error = np.linalg.norm(diff)
        # # print(f'error g_q: {error}')
        # return g_q_num

    def __g_qq_el(self, qe, N, N_xi, N_g, J0, qw):
        g_qq = np.zeros((self.nla_g_el, self.nq_el, self.nq_el))

        for Ni, N_xii, N_gi, J0i, qwi in zip(N, N_xi, N_g, J0, qw):
            NNi = self.stack_shapefunctions(Ni)
            NN_xii = self.stack_shapefunctions(N_xii)

            factor1 = NNi.T @ NNi * J0i * qwi
            factor2 = N_gi * qwi

            ######################
            # director constraints
            ######################
            for j, N_gij in enumerate(N_gi):
                N_gij_factor1 = N_gij * factor1

                # 2 * delta d1 * d1
                g_qq[self.g11DOF[j], self.d1DOF[:, None], self.d1DOF] += 2 * N_gij_factor1
                # delta d1 * d2
                g_qq[self.g12DOF[j], self.d1DOF[:, None], self.d2DOF] += N_gij_factor1
                # delta d2 * d1
                g_qq[self.g12DOF[j], self.d2DOF[:, None], self.d1DOF] += N_gij_factor1
                # delta d1 * d3
                g_qq[self.g13DOF[j], self.d1DOF[:, None], self.d3DOF] += N_gij_factor1
                # delta d3 * d1
                g_qq[self.g13DOF[j], self.d3DOF[:, None], self.d1DOF] += N_gij_factor1

                # 2 * delta d2 * d2
                g_qq[self.g22DOF[j], self.d2DOF[:, None], self.d2DOF] += 2 * N_gij_factor1
                # delta d2 * d3
                g_qq[self.g23DOF[j], self.d2DOF[:, None], self.d3DOF] += N_gij_factor1
                # delta d3 * d2
                g_qq[self.g23DOF[j], self.d3DOF[:, None], self.d2DOF] += N_gij_factor1

                # 2 * delta d3 * d3
                g_qq[self.g33DOF[j], self.d3DOF[:, None], self.d3DOF] += 2 * N_gij_factor1

            ################################
            # unshearability in d2-direction
            ################################
            arg1 = np.einsum('i,jl,jk->ikl', factor2, NNi, NN_xii)
            arg2 = np.einsum('i,jl,jk->ikl', factor2, NN_xii, NNi)
            g_qq[self.g2DOF[:, None, None], self.rDOF[:, None], self.d2DOF] += arg1
            g_qq[self.g2DOF[:, None, None], self.d2DOF[:, None], self.rDOF] += arg2
            g_qq[self.g3DOF[:, None, None], self.rDOF[:, None], self.d3DOF] += arg1
            g_qq[self.g3DOF[:, None, None], self.d3DOF[:, None], self.rDOF] += arg2

            #################
            # inextensibility
            #################
            g_qq[self.g1DOF[:, None, None], self.rDOF[:, None], self.d1DOF] += arg1
            g_qq[self.g1DOF[:, None, None], self.d1DOF[:, None], self.rDOF] += arg2
            # g_qq[self.g1DOF[:, None, None], self.rDOF[:, None], self.rDOF] += np.einsum('i,jl,jk->ikl', 2 * factor2, NN_xii, NN_xii)
            
            # # g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor1, d1 @ NN_xii / J0i)
            # # g_q[self.g1DOF[:, None], self.d1DOF] += np.outer(factor1, r_xi / J0i @ NNi)
            # g_qq[self.g1DOF[:, None, None], self.rDOF[:, None], self.d1DOF] += arg1
            # g_qq[self.g1DOF[:, None, None], self.d1DOF[:, None], self.rDOF] += arg2
            
            # # # r_s = r_xi / J0i
            # # g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor1, 2 * r_s @ NN_xii / J0i)
            # # g_qq[self.g1DOF[:, None, None], self.rDOF[:, None], self.rDOF] += np.einsum('i,jl,jk->ikl', factor2 * J0i, NN_xii / J0i, NN_xii / J0i)
            # g_qq[self.g1DOF[:, None, None], self.rDOF[:, None], self.rDOF] += np.einsum('i,jl,jk->ikl', 2 * factor2, NN_xii, NN_xii / J0i)

        return g_qq

        # g_qq_num = Numerical_derivative(lambda t, q: self.__g_q_el(q, N, N_xi, N_g, J0, qw), order=2)._x(0, qe)
        # # diff = g_qq_num - g_qq
        # # error = np.linalg.norm(diff)
        # # print(f'error g_qq: {error}')
        # return g_qq_num

    # global constraint functions
    def g(self, t, q):
        g = np.zeros(self.nla_g)
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g[elDOF_g] += self.__g_el(q[elDOF], self.N[el], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
        return g

    def g_q(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_q = self.__g_q_el(q[elDOF], self.N[el], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
            coo.extend(g_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))

    def W_g(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_q = self.__g_q_el(q[elDOF], self.N[el], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
            coo.extend(g_q.T, (self.uDOF[elDOF], self.la_gDOF[elDOF_g]))

    def Wla_g_q(self, t, q, la_g, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_qq = self.__g_qq_el(q[elDOF], self.N[el], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
            coo.extend(np.einsum('i,ijk->jk', la_g[elDOF_g], g_qq), (self.uDOF[elDOF], self.qDOF[elDOF]))
