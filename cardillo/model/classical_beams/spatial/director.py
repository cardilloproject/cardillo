import numpy as np
from abc import ABCMeta, abstractmethod
import meshio
import os

from cardillo.utility.coo import Coo
from cardillo.discretization import uniform_knot_vector
from cardillo.discretization.B_spline import Knot_vector
from cardillo.discretization.lagrange import Node_vector
from cardillo.math.algebra import norm3, cross3, skew2ax, skew2ax_A
from cardillo.math.numerical_derivative import Numerical_derivative
from cardillo.discretization.mesh1D import Mesh1D

order = 2

class Timoshenko_beam_director(metaclass=ABCMeta):
    def __init__(self, material_model, A_rho0, B_rho0, C_rho0, 
                 polynomial_degree_r, polynomial_degree_di, nQP, nEl, 
                 Q, q0=None, u0=None, basis='B-spline'):

        # beam properties
        self.materialModel = material_model # material model
        self.A_rho0 = A_rho0 # line density
        self.B_rho0 = B_rho0 # first moment
        self.C_rho0 = C_rho0 # second moment

        # material model
        self.material_model = material_model

        # discretization parameters
        self.polynomial_degree_r = polynomial_degree_r # polynomial degree
        self.polynomial_degree_di = polynomial_degree_di # polynomial degree
        self.nQP = nQP # number of quadrature points
        self.nEl = nEl # number of elements

        self.basis = basis
        if basis == 'B-spline':
            self.knot_vector_r = Knot_vector(polynomial_degree_r, nEl)
            self.knot_vector_di = Knot_vector(polynomial_degree_di, nEl)
            self.nn_r = nn_r = nEl + polynomial_degree_r # number of nodes
            self.nn_di = nn_di = nEl + polynomial_degree_di # number of nodes
        elif basis == 'lagrange':
            self.knot_vector_r = Node_vector(polynomial_degree_r, nEl)
            self.knot_vector_di = Node_vector(polynomial_degree_di, nEl)
            self.nn_r = nn_r = nEl * polynomial_degree_r + 1 # number of nodes
            self.nn_di = nn_di = nEl * polynomial_degree_di + 1 # number of nodes
        else:
            raise RuntimeError(f'wrong basis: "{basis}" was chosen')

        self.nn_el_r = nn_el_r = polynomial_degree_r + 1 # number of nodes per element
        self.nn_el_di = nn_el_di = polynomial_degree_di + 1 # number of nodes per element

        # distinguish centerline and director meshes
        self.nq_n_r = nq_n_r = 3 # number of degrees of freedom per node of the centerline
        self.nq_n_di = nq_n_di = 3 * 3 # number of degrees of freedom per node of the director 1, 2 and 3

        self.mesh_r = Mesh1D(self.knot_vector_r, nQP, derivative_order=1, basis=basis, nq_n=nq_n_r)
        self.mesh_di = Mesh1D(self.knot_vector_di, nQP, derivative_order=1, basis=basis, nq_n=nq_n_di)

        self.nq_r = nq_r = nn_r * nq_n_r
        self.nq_di = nq_di = nn_di * nq_n_di
        self.nq = nq_r + nq_di # total number of generalized coordinates
        self.nu = self.nq
        self.nq_el = nn_el_r * nq_n_r + nn_el_di * nq_n_di # total number of generalized coordinates per element
        self.nq_el_r = nn_el_r * nq_n_r
        self.nq_el_di = nn_el_di * 3 # TODO: can we do this more verbose instead of hard coded 3?
                                     # this corresponds to the number of degrees of freedoms of a single
                                     # director per element -> nn_el_di * 3

        # connectivity matrices for both meshes
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_di = self.mesh_di.elDOF + nq_r # offset of first field (centerline r)
        self.nodalDOF_r = np.arange(self.nq_n_r * self.nn_r).reshape(self.nq_n_r, self.nn_r).T
        self.nodalDOF_di = np.arange(self.nq_n_di * self.nn_di).reshape(self.nq_n_di, self.nn_di).T + nq_r

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nEl, self.nq_el), dtype=int)
        for el in range(nEl):
            self.elDOF[el, :self.nq_el_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_el_r:] = self.elDOF_di[el]

        # degrees of freedom on element level
        self.rDOF  = np.arange(0,            3 * nn_el_r)
        self.d1DOF = np.arange(0,            3 * nn_el_di) + 3 * nn_el_r
        self.d2DOF = np.arange(3 * nn_el_di, 6 * nn_el_di) + 3 * nn_el_r
        self.d3DOF = np.arange(6 * nn_el_di, 9 * nn_el_di) + 3 * nn_el_r

        # shape functions
        self.N_r = self.mesh_r.N # shape functions
        self.N_r_xi = self.mesh_r.N_xi # first derivative w.r.t. xi
        self.N_di = self.mesh_di.N         
        self.N_di_xi = self.mesh_di.N_xi

        # quadrature points
        self.qp = self.mesh_r.qp # quadrature points
        self.qw = self.mesh_r.wp # quadrature weights
            
        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions_r = self.mesh_r.eval_basis
        self.basis_functions_di = self.mesh_di.eval_basis
            
        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q                                        # reference configuration
        self.q0 = Q.copy() if q0 is None else q0          # initial configuration
        self.u0 = np.zeros(self.nu) if u0 is None else u0 # initial velocities

        # precompute values of the reference configuration in order to save computation time
        self.J0 = np.zeros((nEl, nQP))        # J in Harsch2020b (5)
        self.Gamma0 = np.zeros((nEl, nQP, 3)) # dilatation and shear strains of the reference configuration
        self.Kappa0 = np.zeros((nEl, nQP, 3)) # curvature of the reference configuration

        for el in range(nEl):
            # precompute quantities of the reference configuration
            Qe = self.Q[self.elDOF[el]]
            Qe_r = Qe[self.rDOF]
            Qe_D1 = Qe[self.d1DOF]
            Qe_D2 = Qe[self.d2DOF]
            Qe_D3 = Qe[self.d3DOF]

            for i in range(nQP):
                # build matrix of shape function derivatives
                # NN_r_i = self.stack3r(self.N_r[el, i])
                NN_di_i = self.stack3di(self.N_di[el, i])
                NN_r_xii = self.stack3r(self.N_r_xi[el, i])
                NN_di_xii = self.stack3di(self.N_di_xi[el, i])

                # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
                # the parameter space \xi and thus need to be transformed later
                r0_xi = NN_r_xii @ Qe_r
                J0i = norm3(r0_xi)
                self.J0[el, i] = J0i

                D1 = NN_di_i @ Qe_D1
                D1_xi = NN_di_xii @ Qe_D1

                D2 = NN_di_i @ Qe_D2
                D2_xi = NN_di_xii @ Qe_D2

                D3 = NN_di_i @ Qe_D3
                D3_xi = NN_di_xii @ Qe_D3
                
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

        # TODO: we have to implement this for all required fields for better performance
        # # shape functions on the boundary
        # N_bdry, dN_bdry = self.basis_functions(0)
        # N_bdry_left = self.stack_shapefunctions(N_bdry)
        # dN_bdry_left = self.stack_shapefunctions(dN_bdry)

        # N_bdry, dN_bdry = self.basis_functions(1)
        # N_bdry_right = self.stack_shapefunctions(N_bdry)
        # dN_bdry_right = self.stack_shapefunctions(dN_bdry)

        # self.N_bdry = np.array([N_bdry_left, N_bdry_right])
        # self.dN_bdry = np.array([dN_bdry_left, dN_bdry_right])

    def element_number(self, xi):
        # note the elements coincide for both meshes!
        return self.knot_vector_r.element_number(xi)[0]

    def stack3r(self, N):
        nn_el = self.nn_el_r
        NN = np.zeros((3, self.nq_el_r))
        NN[0, :nn_el] = N
        NN[1, nn_el:2*nn_el] = N
        NN[2, 2*nn_el:] = N
        return NN

    def stack3di(self, N):
        nn_el = self.nn_el_di
        NN = np.zeros((3, self.nq_el_di))
        NN[0, :nn_el] = N
        NN[1, nn_el:2*nn_el] = N
        NN[2, 2*nn_el:] = N
        return NN

    def assembler_callback(self):
        self.__M_coo()

    #########################################
    # equations of motion
    #########################################
    def M_el(self, el):
        Me = np.zeros((self.nq_el, self.nq_el))

        for i in range(self.nQP):
            # build matrix of shape function derivatives
            NN_r_i = self.stack3r(self.N_r[el, i])
            NN_di_i = self.stack3di(self.N_di[el, i])

            # extract reference state variables
            J0i = self.J0[el, i]
            qwi = self.qw[el, i]
            factor_rr = NN_r_i.T @ NN_r_i * J0i * qwi
            factor_rdi = NN_r_i.T @ NN_di_i * J0i * qwi
            factor_dir = NN_di_i.T @ NN_r_i * J0i * qwi
            factor_didi = NN_di_i.T @ NN_di_i * J0i * qwi

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
        self.__M = Coo((self.nu, self.nu))
        for el in range(self.nEl):
            # extract element degrees of freedom
            elDOF = self.elDOF[el]
            
            # sparse assemble element mass matrix
            self.__M.extend(self.M_el(el), (self.uDOF[elDOF], self.uDOF[elDOF]))
            
    def M(self, t, q, coo):
        coo.extend_sparse(self.__M)
        
    def E_pot(self, t, q):
        E = 0
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            E += self.E_pot_el(q[elDOF], el)
        return E
    
    def E_pot_el(self, qe, el):
        Ee = 0

        # extract generalized coordinates for beam centerline and directors 
        # in the current and reference configuration
        qe_r = qe[self.rDOF]
        qe_d1 = qe[self.d1DOF]
        qe_d2 = qe[self.d2DOF]
        qe_d3 = qe[self.d3DOF]

        for i in range(self.nQP):
            # build matrix of shape function derivatives
            NN_di_i = self.stack3di(self.N_di[el, i])
            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            NN_di_xii = self.stack3di(self.N_di_xi[el, i])

            # extract reference state variables
            J0i = self.J0[el, i]
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
            
            # evaluate strain energy function
            Ee += self.material_model.potential(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i) * J0i * self.qw[el, i]

        return Ee

    def f_pot(self, t, q):
        f = np.zeros(self.nu)
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            f[elDOF] += self.f_pot_el(q[elDOF], el)
        return f
    
    def f_pot_el(self, qe, el):
        fe = np.zeros(self.nq_el)

        # extract generalized coordinates for beam centerline and directors 
        # in the current and reference configuration
        qe_r = qe[self.rDOF]
        qe_d1 = qe[self.d1DOF]
        qe_d2 = qe[self.d2DOF]
        qe_d3 = qe[self.d3DOF]

        for i in range(self.nQP):
            # build matrix of shape function derivatives
            NN_di_i = self.stack3di(self.N_di[el, i])
            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            NN_di_xii = self.stack3di(self.N_di_xi[el, i])

            # extract reference state variables
            J0i = self.J0[el, i]
            Gamma0_i = self.Gamma0[el, i]
            Kappa0_i = self.Kappa0[el, i]
            qwi = self.qw[el, i]

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
            R = np.vstack((d1, d2, d3)).T
            
            # axial and shear strains
            Gamma_i = R.T @ r_s

            #################################################################
            # formulation of Harsch2020b
            #################################################################

            # torsional and flexural strains
            Kappa_i = np.array([0.5 * (d3 @ d2_s - d2 @ d3_s), \
                                0.5 * (d1 @ d3_s - d3 @ d1_s), \
                                0.5 * (d2 @ d1_s - d1 @ d2_s)])

            # compute contact forces and couples from partial derivatives of the strain energy function w.r.t. strain measures
            n1, n2, n3 = self.material_model.n_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            m1, m2, m3 = self.material_model.m_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            
            # quadrature point contribution to element residual
            fe[self.rDOF] -= NN_r_xii.T @ ( n1 * d1 + n2 * d2 + n3 * d3 ) * qwi
            fe[self.d1DOF] -= ( NN_di_i.T @ ( r_xi * n1 + (m2 / 2 * d3_xi - m3 / 2 * d2_xi) ) + NN_di_xii.T @ (m3 / 2 * d2 - m2 / 2 * d3) ) * qwi
            fe[self.d2DOF] -= ( NN_di_i.T @ ( r_xi * n2 + (m3 / 2 * d1_xi - m1 / 2 * d3_xi) ) + NN_di_xii.T @ (m1 / 2 * d3 - m3 / 2 * d1) ) * qwi
            fe[self.d3DOF] -= ( NN_di_i.T @ ( r_xi * n3 + (m1 / 2 * d2_xi - m2 / 2 * d1_xi) ) + NN_di_xii.T @ (m2 / 2 * d1 - m1 / 2 * d2) ) * qwi

        #     #################################################################
        #     # alternative formulation assuming orthogonality of the directors
        #     #################################################################

        #     # torsional and flexural strains
        #     Kappa_i = np.array([d3 @ d2_s, \
        #                         d1 @ d3_s, \
        #                         d2 @ d1_s])
            
        #     # compute contact forces and couples from partial derivatives of the strain energy function w.r.t. strain measures
        #     n_i = self.material_model.n_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
        #     # n = n_i[0] * d1 + n_i[1] * d2 + n_i[2] * d3
        #     # n = R @ n_i
        #     m_i = self.material_model.m_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
        #     # m = m_i[0] * d1 + m_i[1] * d2 + m_i[2] * d3
        #     # m = R @ m_i
            
        #     # new version
        #     # quadrature point contribution to element residual
        #     n1, n2, n3 = n_i
        #     m1, m2, m3 = m_i
        #     fe[self.rDOF] -=  NN_r_xii.T @ ( n1 * d1 + n2 * d2 + n3 * d3 ) * qwi # delta r'
        #     fe[self.d1DOF] -= ( NN_di_i.T @ ( r_xi * n1 + m2 * d3_xi ) # delta d1 \
        #                         + NN_di_xii.T @ ( m3 * d2 ) ) * qwi    # delta d1'
        #     fe[self.d2DOF] -= ( NN_di_i.T @ ( r_xi * n2 + m3 * d1_xi ) # delta d2 \
        #                         + NN_di_xii.T @ ( m1 * d3 ) ) * qwi    # delta d2'
        #     fe[self.d3DOF] -= ( NN_di_i.T @ ( r_xi * n3 + m1 * d2_xi ) # delta d3 \
        #                         + NN_di_xii.T @ ( m2 * d1 ) ) * qwi    # delta d3'

        return fe

    def f_pot_q(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            Ke = self.f_pot_q_el(q[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(Ke, (self.uDOF[elDOF], self.qDOF[elDOF]))

    def f_pot_q_el(self, qe, el):
        # return Numerical_derivative(lambda t, qe: self.f_pot_el(qe, el), order=order)._x(0, qe)

        Ke = np.zeros((self.nq_el, self.nq_el))

        # extract generalized coordinates for beam centerline and directors 
        # in the current and reference configuration
        qe_r = qe[self.rDOF]
        qe_d1 = qe[self.d1DOF]
        qe_d2 = qe[self.d2DOF]
        qe_d3 = qe[self.d3DOF]

        for i in range(self.nQP):
            # build matrix of shape function derivatives
            # NN_r_i = self.stack3r(self.N_r[el, i])
            NN_di_i = self.stack3di(self.N_di[el, i])
            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            NN_di_xii = self.stack3di(self.N_di_xi[el, i])

            # extract reference state variables
            J0i = self.J0[el, i]
            Gamma0_i = self.Gamma0[el, i]
            Kappa0_i = self.Kappa0[el, i]
            qwi = self.qw[el, i]

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
            R = np.vstack((d1, d2, d3)).T
            
            # axial and shear strains
            Gamma_i = R.T @ r_s

            # derivative of axial and shear strains
            Gamma_j_qr = R.T @ NN_r_xii / J0i
            Gamma_1_qd1 = r_xi @ NN_di_i / J0i
            Gamma_2_qd2 = r_xi @ NN_di_i / J0i
            Gamma_3_qd3 = r_xi @ NN_di_i / J0i

            #################################################################
            # formulation of Harsch2020b
            #################################################################

            # torsional and flexural strains
            Kappa_i = np.array([0.5 * (d3 @ d2_s - d2 @ d3_s), \
                                0.5 * (d1 @ d3_s - d3 @ d1_s), \
                                0.5 * (d2 @ d1_s - d1 @ d2_s)])

            # derivative of torsional and flexural strains
            kappa_1_qe_d1 = np.zeros(3 * self.nn_el_di)
            kappa_1_qe_d2 = 0.5 * (d3 @ NN_di_xii - d3_xi @ NN_di_i) / J0i
            kappa_1_qe_d3 = 0.5 * (d2_xi @ NN_di_i - d2 @ NN_di_xii) / J0i
                    
            kappa_2_qe_d1 = 0.5 * (d3_xi @ NN_di_i - d3 @ NN_di_xii) / J0i
            kappa_2_qe_d2 = np.zeros(3 * self.nn_el_di)
            kappa_2_qe_d3 = 0.5 * (d1 @ NN_di_xii - d1_xi @ NN_di_i) / J0i
            
            kappa_3_qe_d1 = 0.5 * (d2 @ NN_di_xii - d2_xi @ NN_di_i) / J0i
            kappa_3_qe_d2 = 0.5 * (d1_xi @ NN_di_i - d1 @ NN_di_xii) / J0i
            kappa_3_qe_d3 = np.zeros(3 * self.nn_el_di)

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

            Ke[self.rDOF[:, None], self.rDOF] -= NN_r_xii.T @ (np.outer(d1, n1_qr) + np.outer(d2, n2_qr) + np.outer(d3, n3_qr)) * qwi
            Ke[self.rDOF[:, None], self.d1DOF] -= NN_r_xii.T @ (np.outer(d1, n1_qd1) + np.outer(d2, n2_qd1) + np.outer(d3, n3_qd1) + n1 * NN_di_i) * qwi
            Ke[self.rDOF[:, None], self.d2DOF] -= NN_r_xii.T @ (np.outer(d1, n1_qd2) + np.outer(d2, n2_qd2) + np.outer(d3, n3_qd2) + n2 * NN_di_i) * qwi
            Ke[self.rDOF[:, None], self.d3DOF] -= NN_r_xii.T @ (np.outer(d1, n1_qd3) + np.outer(d2, n2_qd3) + np.outer(d3, n3_qd3) + n3 * NN_di_i) * qwi

            Ke[self.d1DOF[:, None], self.rDOF] -= NN_di_i.T @ (np.outer(r_xi, n1_qr) + n1 * NN_r_xii + np.outer(0.5 * d3_xi, m2_qr) - np.outer(0.5 * d2_xi, m3_qr)) * qwi \
                                                  + NN_di_xii.T @ (np.outer(0.5 * d2, m3_qr) - np.outer(0.5 * d3, m2_qr)) * qwi
            Ke[self.d1DOF[:, None], self.d1DOF] -= NN_di_i.T @ (np.outer(r_xi, n1_qd1) + np.outer(0.5 * d3_xi, m2_qd1) - np.outer(0.5 * d2_xi, m3_qd1)) * qwi \
                                                  + NN_di_xii.T @ (np.outer(0.5 * d2, m3_qd1) - np.outer(0.5 * d3, m2_qd1)) * qwi
            Ke[self.d1DOF[:, None], self.d2DOF] -= NN_di_i.T @ (np.outer(r_xi, n1_qd2) + np.outer(0.5 * d3_xi, m2_qd2) - np.outer(0.5 * d2_xi, m3_qd2) - 0.5 * m3 * NN_di_xii) * qwi \
                                                  + NN_di_xii.T @ (np.outer(0.5 * d2, m3_qd2) + 0.5 * m3 * NN_di_i - np.outer(0.5 * d3, m2_qd2)) * qwi
            Ke[self.d1DOF[:, None], self.d3DOF] -= NN_di_i.T @ (np.outer(r_xi, n1_qd3) + np.outer(0.5 * d3_xi, m2_qd3) + 0.5 * m2 * NN_di_xii - np.outer(0.5 * d2_xi, m3_qd3)) * qwi \
                                                  + NN_di_xii.T @ (np.outer(0.5 * d2, m3_qd3) - np.outer(0.5 * d3, m2_qd3) - 0.5 * m2 * NN_di_i) * qwi

            Ke[self.d2DOF[:, None], self.rDOF] -= NN_di_i.T @ (np.outer(r_xi, n2_qr) + n2 * NN_r_xii + np.outer(0.5 * d1_xi, m3_qr) - np.outer(0.5 * d3_xi, m1_qr)) * qwi \
                                                  + NN_di_xii.T @ (np.outer(0.5 * d3, m1_qr) - np.outer(0.5 * d1, m3_qr)) * qwi
            Ke[self.d2DOF[:, None], self.d1DOF] -= NN_di_i.T @ (np.outer(r_xi, n2_qd1) + np.outer(0.5 * d1_xi, m3_qd1) + 0.5 * m3 * NN_di_xii - np.outer(0.5 * d3_xi, m1_qd1)) * qwi \
                                                  + NN_di_xii.T @ (np.outer(0.5 * d3, m1_qd1) - np.outer(0.5 * d1, m3_qd1) - 0.5 * m3 * NN_di_i) * qwi
            Ke[self.d2DOF[:, None], self.d2DOF] -= NN_di_i.T @ (np.outer(r_xi, n2_qd2) + np.outer(0.5 * d1_xi, m3_qd2) - np.outer(0.5 * d3_xi, m1_qd2)) * qwi \
                                                  + NN_di_xii.T @ (np.outer(0.5 * d3, m1_qd2) - np.outer(0.5 * d1, m3_qd2)) * qwi
            Ke[self.d2DOF[:, None], self.d3DOF] -= NN_di_i.T @ (np.outer(r_xi, n2_qd3) + np.outer(0.5 * d1_xi, m3_qd3) - np.outer(0.5 * d3_xi, m1_qd3) - 0.5 * m1 * NN_di_xii) * qwi \
                                                  + NN_di_xii.T @ (np.outer(0.5 * d3, m1_qd3) + 0.5 * m1 * NN_di_i - np.outer(0.5 * d1, m3_qd3)) * qwi

            Ke[self.d3DOF[:, None], self.rDOF] -= NN_di_i.T @ (np.outer(r_xi, n3_qr) + n3 * NN_r_xii + np.outer(0.5 * d2_xi, m1_qr) - np.outer(0.5 * d1_xi, m2_qr)) * qwi \
                                                  + NN_di_xii.T @ (np.outer(0.5 * d1, m2_qr) - np.outer(0.5 * d2, m1_qr)) * qwi
            Ke[self.d3DOF[:, None], self.d1DOF] -= NN_di_i.T @ (np.outer(r_xi, n3_qd1) + np.outer(0.5 * d2_xi, m1_qd1) - np.outer(0.5 * d1_xi, m2_qd1) - 0.5 * m2 * NN_di_xii) * qwi \
                                                  + NN_di_xii.T @ (np.outer(0.5 * d1, m2_qd1) + 0.5 * m2 * NN_di_i - np.outer(0.5 * d2, m1_qd1)) * qwi
            Ke[self.d3DOF[:, None], self.d2DOF] -= NN_di_i.T @ (np.outer(r_xi, n3_qd2) + np.outer(0.5 * d2_xi, m1_qd2) + 0.5 * m1 * NN_di_xii - np.outer(0.5 * d1_xi, m2_qd2)) * qwi \
                                                  + NN_di_xii.T @ (np.outer(0.5 * d1, m2_qd2) - np.outer(0.5 * d2, m1_qd2) - 0.5 * m1 * NN_di_i) * qwi
            Ke[self.d3DOF[:, None], self.d3DOF] -= NN_di_i.T @ (np.outer(r_xi, n3_qd3) + np.outer(0.5 * d2_xi, m1_qd3) - np.outer(0.5 * d1_xi, m2_qd3)) * qwi \
                                                  + NN_di_xii.T @ (np.outer(0.5 * d1, m2_qd3) - np.outer(0.5 * d2, m1_qd3)) * qwi

            # #################################################################
            # # alternative formulation assuming orthogonality of the directors
            # #################################################################

            # # torsional and flexural strains
            # Kappa_i = np.array([d3 @ d2_s, \
            #                     d1 @ d3_s, \
            #                     d2 @ d1_s])

            # # derivative of torsional and flexural strains
            # kappa_1_qe_d1 = np.zeros(3 * self.nn_el_di)
            # kappa_1_qe_d2 = d3 @ NN_di_xii / J0i
            # kappa_1_qe_d3 = d2_s @ NN_di_i
                    
            # kappa_2_qe_d1 = d3_s @ NN_di_i
            # kappa_2_qe_d2 = np.zeros(3 * self.nn_el_di)
            # kappa_2_qe_d3 = d1 @ NN_di_xii / J0i
            
            # kappa_3_qe_d1 = d2 @ NN_di_xii / J0i
            # kappa_3_qe_d2 = d1_s @ NN_di_i
            # kappa_3_qe_d3 = np.zeros(3 * self.nn_el_di)

            # kappa_j_qe_d1 = np.vstack((kappa_1_qe_d1, kappa_2_qe_d1, kappa_3_qe_d1))
            # kappa_j_qe_d2 = np.vstack((kappa_1_qe_d2, kappa_2_qe_d2, kappa_3_qe_d2))
            # kappa_j_qe_d3 = np.vstack((kappa_1_qe_d3, kappa_2_qe_d3, kappa_3_qe_d3))

            # # compute contact forces and couples from partial derivatives of the strain energy function w.r.t. strain measures
            # n1, n2, n3 = self.material_model.n_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            # n_i_Gamma_i_j = self.material_model.n_i_Gamma_i_j(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            # n_i_Kappa_i_j = self.material_model.n_i_Kappa_i_j(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            # n1_qr, n2_qr, n3_qr = n_i_Gamma_i_j @ Gamma_j_qr

            # n1_qd1, n2_qd1, n3_qd1 = np.outer(n_i_Gamma_i_j[0], Gamma_1_qd1) + n_i_Kappa_i_j @ kappa_j_qe_d1
            # n1_qd2, n2_qd2, n3_qd2 = np.outer(n_i_Gamma_i_j[1], Gamma_2_qd2) + n_i_Kappa_i_j @ kappa_j_qe_d2
            # n1_qd3, n2_qd3, n3_qd3 = np.outer(n_i_Gamma_i_j[2], Gamma_3_qd3) + n_i_Kappa_i_j @ kappa_j_qe_d3

            # m1, m2, m3 = self.material_model.m_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            # m_i_Gamma_i_j = self.material_model.m_i_Gamma_i_j(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            # m_i_Kappa_i_j = self.material_model.m_i_Kappa_i_j(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)

            # m1_qr, m2_qr, m3_qr = m_i_Gamma_i_j @ Gamma_j_qr
            # m1_qd1, m2_qd1, m3_qd1 = np.outer(m_i_Gamma_i_j[0], Gamma_1_qd1) + m_i_Kappa_i_j @ kappa_j_qe_d1
            # m1_qd2, m2_qd2, m3_qd2 = np.outer(m_i_Gamma_i_j[1], Gamma_2_qd2) + m_i_Kappa_i_j @ kappa_j_qe_d2
            # m1_qd3, m2_qd3, m3_qd3 = np.outer(m_i_Gamma_i_j[2], Gamma_3_qd3) + m_i_Kappa_i_j @ kappa_j_qe_d3
                        
            # # quadrature point contribution to element stiffness matrix

            # # fe[self.rDOF] -=  NN_r_xii.T @ ( n1 * d1 + n2 * d2 + n3 * d3 ) * qwi # delta r'
            # Ke[self.rDOF[:, None], self.rDOF] -= NN_r_xii.T @ (np.outer(d1, n1_qr) + np.outer(d2, n2_qr) + np.outer(d3, n3_qr)) * qwi
            # Ke[self.rDOF[:, None], self.d1DOF] -= NN_r_xii.T @ (np.outer(d1, n1_qd1) + np.outer(d2, n2_qd1) + np.outer(d3, n3_qd1) + n1 * NN_di_i) * qwi
            # Ke[self.rDOF[:, None], self.d2DOF] -= NN_r_xii.T @ (np.outer(d1, n1_qd2) + np.outer(d2, n2_qd2) + np.outer(d3, n3_qd2) + n2 * NN_di_i) * qwi
            # Ke[self.rDOF[:, None], self.d3DOF] -= NN_r_xii.T @ (np.outer(d1, n1_qd3) + np.outer(d2, n2_qd3) + np.outer(d3, n3_qd3) + n3 * NN_di_i) * qwi

            # # fe[self.d1DOF] -= ( NN_di_i.T @ ( r_xi * n1 + m2 * d3_xi ) # delta d1 \
            # #                     + NN_di_xii.T @ ( m3 * d2 ) ) * qwi    # delta d1'
            # Ke[self.d1DOF[:, None], self.rDOF] -= NN_di_i.T @ (np.outer(r_xi, n1_qr) + n1 * NN_r_xii + np.outer(d3_xi, m2_qr)) * qwi \
            #                                       + NN_di_xii.T @ np.outer(d2, m3_qr) * qwi
            # Ke[self.d1DOF[:, None], self.d1DOF] -= NN_di_i.T @ (np.outer(r_xi, n1_qd1) + np.outer(d3_xi, m2_qd1)) * qwi \
            #                                        + NN_di_xii.T @ np.outer(d2, m3_qd1) * qwi
            # Ke[self.d1DOF[:, None], self.d2DOF] -= NN_di_i.T @ (np.outer(r_xi, n1_qd2) + np.outer(d3_xi, m2_qd2)) * qwi \
            #                                        + NN_di_xii.T @ (np.outer(d2, m3_qd2) + m3 * NN_di_i) * qwi
            # Ke[self.d1DOF[:, None], self.d3DOF] -= NN_di_i.T @ (np.outer(r_xi, n1_qd3) + np.outer(d3_xi, m2_qd3) + m2 * NN_di_xii) * qwi \
            #                                        + NN_di_xii.T @ np.outer(d2, m3_qd3) * qwi

            # # fe[self.d2DOF] -= ( NN_di_i.T @ ( r_xi * n2 + m3 * d1_xi ) # delta d2 \
            # #                     + NN_di_xii.T @ ( m1 * d3 ) ) * qwi    # delta d2'
            # Ke[self.d2DOF[:, None], self.rDOF] -= NN_di_i.T @ (np.outer(r_xi, n2_qr) + n2 * NN_r_xii + np.outer(d1_xi, m3_qr)) * qwi \
            #                                       + NN_di_xii.T @ np.outer(d3, m1_qr) * qwi
            # Ke[self.d2DOF[:, None], self.d1DOF] -= NN_di_i.T @ (np.outer(r_xi, n2_qd1) + np.outer(d1_xi, m3_qd1) + m3 * NN_di_xii) * qwi \
            #                                        + NN_di_xii.T @ np.outer(d3, m1_qd1) * qwi
            # Ke[self.d2DOF[:, None], self.d2DOF] -= NN_di_i.T @ (np.outer(r_xi, n2_qd2) + np.outer(d1_xi, m3_qd2)) * qwi \
            #                                        + NN_di_xii.T @ np.outer(d3, m1_qd2) * qwi
            # Ke[self.d2DOF[:, None], self.d3DOF] -= NN_di_i.T @ (np.outer(r_xi, n2_qd3) + np.outer(d1_xi, m3_qd3)) * qwi \
            #                                        + NN_di_xii.T @ (np.outer(d3, m1_qd3) + m1 * NN_di_i) * qwi

            # # fe[self.d3DOF] -= ( NN_di_i.T @ ( r_xi * n3 + m1 * d2_xi ) # delta d3 \
            # #                     + NN_di_xii.T @ ( m2 * d1 ) ) * qwi    # delta d3'
            # Ke[self.d3DOF[:, None], self.rDOF] -= NN_di_i.T @ (np.outer(r_xi, n3_qr) + n3 * NN_r_xii + np.outer(d2_xi, m1_qr)) * qwi \
            #                                       + NN_di_xii.T @ np.outer(d1, m2_qr) * qwi
            # Ke[self.d3DOF[:, None], self.d1DOF] -= NN_di_i.T @ (np.outer(r_xi, n3_qd1) + np.outer(d2_xi, m1_qd1)) * qwi \
            #                                        + NN_di_xii.T @ (np.outer(d1, m2_qd1) + m2 * NN_di_i) * qwi
            # Ke[self.d3DOF[:, None], self.d2DOF] -= NN_di_i.T @ (np.outer(r_xi, n3_qd2) + np.outer(d2_xi, m1_qd2) + m1 * NN_di_xii) * qwi \
            #                                        + NN_di_xii.T @ np.outer(d1, m2_qd2) * qwi
            # Ke[self.d3DOF[:, None], self.d3DOF] -= NN_di_i.T @ (np.outer(r_xi, n3_qd3) + np.outer(d2_xi, m1_qd3)) * qwi \
            #                                        + NN_di_xii.T @ np.outer(d1, m2_qd3) * qwi

        return Ke

        # Ke_num = Numerical_derivative(lambda t, qe: self.f_pot_el(qe, el), order=2)._x(0, qe)
        # diff = Ke_num - Ke
        # error = np.max(np.abs(diff))
        # print(f'max error f_pot_q_el: {error}')
        
        # print(f'diff[self.rDOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.rDOF]))}')
        # print(f'diff[self.rDOF[:, None], self.d1DOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.d1DOF]))}')
        # print(f'diff[self.rDOF[:, None], self.d2DOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.d2DOF]))}')
        # print(f'diff[self.rDOF[:, None], self.d3DOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.d3DOF]))}')
        
        # print(f'diff[self.d1DOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.d1DOF[:, None], self.rDOF]))}')
        # print(f'diff[self.d1DOF[:, None], self.d1DOF]: {np.max(np.abs(diff[self.d1DOF[:, None], self.d1DOF]))}')
        # print(f'diff[self.d1DOF[:, None], self.d2DOF]: {np.max(np.abs(diff[self.d1DOF[:, None], self.d2DOF]))}')
        # print(f'diff[self.d1DOF[:, None], self.d3DOF]: {np.max(np.abs(diff[self.d1DOF[:, None], self.d3DOF]))}')

        # print(f'diff[self.d2DOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.d2DOF[:, None], self.rDOF]))}')
        # print(f'diff[self.d2DOF[:, None], self.d1DOF]: {np.max(np.abs(diff[self.d2DOF[:, None], self.d1DOF]))}')
        # print(f'diff[self.d2DOF[:, None], self.d2DOF]: {np.max(np.abs(diff[self.d2DOF[:, None], self.d2DOF]))}')
        # print(f'diff[self.d2DOF[:, None], self.d3DOF]: {np.max(np.abs(diff[self.d2DOF[:, None], self.d3DOF]))}')

        # print(f'diff[self.d3DOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.d3DOF[:, None], self.rDOF]))}')
        # print(f'diff[self.d3DOF[:, None], self.d1DOF]: {np.max(np.abs(diff[self.d3DOF[:, None], self.d1DOF]))}')
        # print(f'diff[self.d3DOF[:, None], self.d2DOF]: {np.max(np.abs(diff[self.d3DOF[:, None], self.d2DOF]))}')
        # print(f'diff[self.d3DOF[:, None], self.d3DOF]: {np.max(np.abs(diff[self.d3DOF[:, None], self.d3DOF]))}')

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
    # TODO: optimized implementation for boundaries
    def elDOF_P(self, frame_ID):
        xi = frame_ID[0]
        # if xi == 0:
        #     return self.elDOF[0]
        # elif xi == 1:
        #     return self.elDOF[-1]
        # else:
        #     el = np.where(xi >= self.element_span)[0][-1]
        #     return self.elDOF[el]
        # el = np.where(xi >= self.element_span)[0][-1]
        el = self.element_number(xi)
        return self.elDOF[el]

    def qDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def uDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    # TODO: optimized implementation for boundaries
    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # xi = frame_ID[0]
        # if xi == 0:
        #     NN = self.N_bdry[0]
        # elif xi == 1:
        #     NN = self.N_bdry[-1]
        # else:
        #     N, _ = self.basis_functions(frame_ID[0])
        #     NN = self.stack3r(N)
        N, _ = self.basis_functions_r(frame_ID[0])
        NN = self.stack3r(N)
        return NN @ q[self.rDOF] + self.A_IK(t, q, frame_ID=frame_ID) @ K_r_SP

    # TODO: optimized implementation for boundaries
    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        # xi = frame_ID[0]
        # if xi == 0:
        #     NN = self.N_bdry[0]
        # elif xi == 1:
        #     NN = self.N_bdry[-1]
        # else:
        #     N, _ = self.basis_functions(frame_ID[0])
        #     NN = self.stack3r(N)
        N, _ = self.basis_functions_r(frame_ID[0])
        NN = self.stack3r(N)

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

    # TODO: optimized implementation for boundaries
    def A_IK(self, t, q, frame_ID):
        # xi = frame_ID[0]
        # if xi == 0:
        #     NN = self.N_bdry[0]
        # elif xi == 1:
        #     NN = self.N_bdry[-1]
        # else:
        #     N, _ = self.basis_functions(frame_ID[0])
        #     NN = self.stack3r(N)
        N, _ = self.basis_functions_di(frame_ID[0])
        NN = self.stack3di(N)

        d1 = NN @ q[self.d1DOF]
        d2 = NN @ q[self.d2DOF]
        d3 = NN @ q[self.d3DOF]
        return np.vstack((d1, d2, d3)).T

    # TODO: optimized implementation for boundaries
    def A_IK_q(self, t, q, frame_ID):
        # xi = frame_ID[0]
        # if xi == 0:
        #     NN = self.N_bdry[0]
        # elif xi == 1:
        #     NN = self.N_bdry[-1]
        # else:
        #     N, _ = self.basis_functions(frame_ID[0])
        #     NN = self.stack3r(N)
        N, _ = self.basis_functions_di(frame_ID[0])
        NN = self.stack3di(N)

        A_IK_q = np.zeros((3, 3, self.nq_el))
        A_IK_q[:, 0, self.d1DOF] = NN
        A_IK_q[:, 1, self.d2DOF] = NN
        A_IK_q[:, 2, self.d3DOF] = NN
        return A_IK_q

        # A_IK_q_num =  Numerical_derivative(lambda t, q: self.A_IK(t, q, frame_ID=frame_ID))._x(t, q)
        # error = np.linalg.norm(A_IK_q - A_IK_q_num)
        # print(f'error in A_IK_q: {error}')
        # return A_IK_q_num

    # TODO: optimized implementation for boundaries
    def v_P(self, t, q, u, frame_ID, K_r_SP=None):
        # xi = frame_ID[0]
        # if xi == 0:
        #     NN = self.N_bdry[0]
        # elif xi == 1:
        #     NN = self.N_bdry[-1]
        # else:
        #     N, _ = self.basis_functions(frame_ID[0])
        #     NN = self.stack3r(N)
        N, _ = self.basis_functions_r(frame_ID[0])
        NN = self.stack3r(N)
        
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

    # TODO: optimized implementation for boundaries
    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        # xi = frame_ID[0]
        # if xi == 0:
        #     NN = self.N_bdry[0]
        # elif xi == 1:
        #     NN = self.N_bdry[-1]
        # else:
        #     N, _ = self.basis_functions(frame_ID[0])
        #     NN = self.stack3r(N)
        N, _ = self.basis_functions_r(frame_ID[0])
        NN = self.stack3r(N)
        
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

    # TODO: optimized implementation for boundaries
    def K_Omega(self, t, q, u, frame_ID):
        # xi = frame_ID[0]
        # if xi == 0:
        #     NN = self.N_bdry[0]
        # elif xi == 1:
        #     NN = self.N_bdry[-1]
        # else:
        #     N, _ = self.basis_functions(frame_ID[0])
        #     NN = self.stack3r(N)
        N, _ = self.basis_functions_di(frame_ID[0])
        NN = self.stack3di(N)

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

    # TODO: optimized implementation for boundaries
    def K_J_R(self, t, q, frame_ID):
        # xi = frame_ID[0]
        # if xi == 0:
        #     NN = self.N_bdry[0]
        # elif xi == 1:
        #     NN = self.N_bdry[-1]
        # else:
        #     N, _ = self.basis_functions(frame_ID[0])
        #     NN = self.stack3r(N)
        N, _ = self.basis_functions_di(frame_ID[0])
        NN = self.stack3di(N)

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

    # TODO: optimized implementation for boundaries
    def K_J_R_q(self, t, q, frame_ID):
        # xi = frame_ID[0]
        # if xi == 0:
        #     NN = self.N_bdry[0]
        # elif xi == 1:
        #     NN = self.N_bdry[-1]
        # else:
        #     N, _ = self.basis_functions(frame_ID[0])
        #     NN = self.stack3r(N)
        N, _ = self.basis_functions_di(frame_ID[0])
        NN = self.stack3di(N)

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

    # TODO: optimized implementation for boundaries
    def K_Psi(self, t, q, u, u_dot, frame_ID):
        # xi = frame_ID[0]
        # if xi == 0:
        #     NN = self.N_bdry[0]
        # elif xi == 1:
        #     NN = self.N_bdry[-1]
        # else:
        #     N, _ = self.basis_functions(frame_ID[0])
        #     NN = self.stack3r(N)
        N, _ = self.basis_functions_di(frame_ID[0])
        NN = self.stack3di(N)

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
    def body_force_el(self, force, t, el):
        fe = np.zeros(self.nq_el)
        for i in range(self.nQP):
            NNi = self.stack3r(self.N_r[el, i])
            fe[self.rDOF] += NNi.T @ force(self.qp[el, i], t) * self.J0[el, i] * self.qw[el, i]
        return fe

    def body_force(self, t, q, force):
        f = np.zeros(self.nq)
        for el in range(self.nEl):
            f[self.elDOF[el]] += self.body_force_el(force, t, el)
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
        return q[self.qDOF][:3*self.nn_r].reshape(3, -1)

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
        # centerline and connectivity
        cells_r, points_r, HigherOrderDegrees_r = self.mesh_r.vtk_mesh(q[:self.nq_r])

        # if the centerline and the directors are interpolated with the same
        # polynomial degree we can use the values on the nodes and decompose the B-spline
        # into multiple Bezier patches, otherwise the directors have to be interpolated
        # onto the nodes of the centerline by a so-called L2-projection, see below 
        same_shape_functions = False
        if self.polynomial_degree_r == self.polynomial_degree_di:
            same_shape_functions = True
            
        if same_shape_functions:
            _, points_di, _ = self.mesh_di.vtk_mesh(q[self.nq_r:])

            # fill dictionary storing point data with directors
            point_data = {
                "d1": points_di[:, 0:3],
                "d2": points_di[:, 3:6],
                "d3": points_di[:, 6:9],
            }

        else:
            point_data = {}

        # export existing values on quadrature points using L2 projection
        J0_vtk = self.mesh_r.field_to_vtk(self.J0.reshape(self.nEl, self.nQP, 1))
        point_data.update({"J0": J0_vtk})
        
        Gamma0_vtk = self.mesh_r.field_to_vtk(self.Gamma0)
        point_data.update({"Gamma0": Gamma0_vtk})
        
        Kappa0_vtk = self.mesh_r.field_to_vtk(self.Kappa0)
        point_data.update({"Kappa0": Kappa0_vtk})

        # evaluate fields at quadrature points that have to be projected onto the centerline mesh:
        # - strain measures Gamma & Kappa
        # - directors d1, d2, d3
        Gamma = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
        Kappa = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
        if not same_shape_functions:
            d1s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
            d2s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
            d3s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
        for el in range(self.nEl):
            qe = q[self.elDOF[el]]

            # extract generalized coordinates for beam centerline and directors 
            # in the current and reference configuration
            qe_r = qe[self.rDOF]
            qe_d1 = qe[self.d1DOF]
            qe_d2 = qe[self.d2DOF]
            qe_d3 = qe[self.d3DOF]

            for i in range(self.nQP):
                # build matrix of shape function derivatives
                NN_di_i = self.stack3di(self.N_di[el, i])
                NN_r_xii = self.stack3r(self.N_r_xi[el, i])
                NN_di_xii = self.stack3di(self.N_di_xi[el, i])

                # extract reference state variables
                J0i = self.J0[el, i]
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
                Kappa[el, i] = np.array([0.5 * (d3 @ d2_s - d2 @ d3_s), \
                                         0.5 * (d1 @ d3_s - d3 @ d1_s), \
                                         0.5 * (d2 @ d1_s - d1 @ d2_s)])
        
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
            points_r, # only export centerline as geometry here!
            cells_r,
            point_data=point_data,
            cell_data={"HigherOrderDegrees": HigherOrderDegrees_r},
            binary=binary
        )

####################################################
# straight initial configuration
####################################################
def straight_configuration(polynomial_degree_r, polynomial_degree_di, nEl, L, 
                           greville_abscissae=True, r_OP=np.zeros(3), A_IK=np.eye(3), 
                           basis='B-spline'):
    if basis == 'B-spline':
        nn_r = polynomial_degree_r + nEl
        nn_di = polynomial_degree_di + nEl
    elif basis == 'lagrange':
        nn_r = polynomial_degree_r * nEl + 1
        nn_di = polynomial_degree_di * nEl + 1
    else:
        raise RuntimeError(f'wrong basis: "{basis}" was chosen')
    
    X = np.linspace(0, L, num=nn_r)
    Y = np.zeros(nn_r)
    Z = np.zeros(nn_r)
    if greville_abscissae and basis == 'B-Splines':
        kv = uniform_knot_vector(polynomial_degree_r, nEl)
        for i in range(nn_r):
            X[i] = np.sum(kv[i+1:i+polynomial_degree_r+1])
        X = X * L / polynomial_degree_r

    r0 = np.vstack((X, Y, Z)).T
    for i, r0i in enumerate(r0):
        X[i], Y[i], Z[i] = r_OP + A_IK @ r0i
    
    # compute reference directors
    D1, D2, D3 = A_IK.T
    D1 = np.repeat(D1, nn_di)
    D2 = np.repeat(D2, nn_di)
    D3 = np.repeat(D3, nn_di)
    
    # assemble all reference generalized coordinates
    return np.concatenate([X, Y, Z, D1, D2, D3])

####################################################
# constraint beam using nodal constraints
####################################################
class Timoshenko_director_dirac(Timoshenko_beam_director):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nla_g = 6 * self.nn_di # we enforce constraints on every director node
        la_g0 = kwargs.get('la_g0')
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        # nodal degrees of freedom
        self.d1DOF_node = np.arange(0, 3)
        self.d2DOF_node = np.arange(3, 6)
        self.d3DOF_node = np.arange(6, 9)

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
        g_q = np.zeros((6, 9))

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
        g_qq = np.zeros((6, 9, 9))

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
        for i, DOF in enumerate(self.nodalDOF_di):
            idx = i * 6
            g[idx:idx+6] = self.__g(q[DOF])
        return g

    def g_q(self, t, q, coo):
        for i, DOF in enumerate(self.nodalDOF_di):
            idx = i * 6
            coo.extend(self.__g_q(q[DOF]), (self.la_gDOF[np.arange(idx, idx+6)], self.qDOF[DOF]))

    def W_g(self, t, q, coo):
        for i, DOF in enumerate(self.nodalDOF_di):
            idx = i * 6
            coo.extend(self.__g_q(q[DOF]).T, (self.uDOF[DOF], self.la_gDOF[np.arange(idx, idx+6)]))

    def Wla_g_q(self, t, q, la_g, coo):
        for i, DOF in enumerate(self.nodalDOF_di):
            idx = i * 6
            coo.extend(np.einsum('i,ijk->jk', la_g[idx:idx+6], self.__g_qq()), (self.uDOF[DOF], self.qDOF[DOF]))

####################################################
# constraint beam using integral constraints
####################################################
class Timoshenko_director_integral(Timoshenko_beam_director):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.polynomial_degree_g = self.polynomial_degree_di # if polynomial_degree_g is None else polynomial_degree_g
        self.nn_el_g = self.polynomial_degree_g + 1 # number of nodes per element

        if self.basis == 'B-spline':
            self.knot_vector_g = Knot_vector(self.polynomial_degree_g, self.nEl)
            self.nn_g = self.nEl + self.polynomial_degree_g # number of nodes
        elif self.basis == 'lagrange':
            self.knot_vector_g = Node_vector(self.polynomial_degree_g, self.nEl)
            self.nn_g = self.nEl * self.polynomial_degree_g + 1 # number of nodes

        self.nq_n_g = 6 # number of degrees of freedom per node
        self.nla_g = self.nn_g * self.nq_n_g
        self.nla_g_el = self.nn_el_g * self.nq_n_g

        la_g0 = kwargs.get('la_g0')
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.element_span_g = self.knot_vector_g.element_data
        self.mesh_g = Mesh1D(self.knot_vector_g, self.nQP, derivative_order=0, nq_n=self.nq_n_g, basis=self.basis)
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
        
        for i in range(self.nQP):
            NNi = self.stack3di(self.N_di[el, i])

            d1 = NNi @ qe[self.d1DOF]
            d2 = NNi @ qe[self.d2DOF]
            d3 = NNi @ qe[self.d3DOF]

            factor = self.N_g[el, i] * self.J0[el, i] * self.qw[el, i]
         
            g[self.g11DOF] += (d1 @ d1 - 1) * factor
            g[self.g12DOF] += (d1 @ d2)     * factor
            g[self.g13DOF] += (d1 @ d3)     * factor
            g[self.g22DOF] += (d2 @ d2 - 1) * factor
            g[self.g23DOF] += (d2 @ d3)     * factor
            g[self.g33DOF] += (d3 @ d3 - 1) * factor

        return g
        
    def __g_q_el(self, qe, el):
        # return Numerical_derivative(lambda t, q: self.__g_el(q, el), order=order)._x(0, qe)

        g_q = np.zeros((self.nla_g_el, self.nq_el))
        
        for i in range(self.nQP):
            NNi = self.stack3di(self.N_di[el, i])

            d1 = NNi @ qe[self.d1DOF]
            d2 = NNi @ qe[self.d2DOF]
            d3 = NNi @ qe[self.d3DOF]

            factor = self.N_g[el, i] * self.J0[el, i] * self.qw[el, i]
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

        # g_q_num = Numerical_derivative(lambda t, q: self.__g_el(q, el), order=2)._x(0, qe)
        # diff = g_q_num - g_q
        # error = np.linalg.norm(diff)
        # if error > 1.0e-8:
        #     print(f'error g_q: {error}')
        # return g_q_num

    def __g_qq_el(self, el):
        # return Numerical_derivative(lambda t, q: self.__g_q_el(q, el), order=order)._x(0, np.zeros(self.nq_el))
        
        g_qq = np.zeros((self.nla_g_el, self.nq_el, self.nq_el))

        for i in range(self.nQP):
            NNi = self.stack3di(self.N_di[el, i])
            factor = NNi.T @ NNi * self.J0[el, i] * self.qw[el, i]

            for j, N_gij in enumerate(self.N_g[el, i]):
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

        # g_qq_num = Numerical_derivative(lambda t, q: self.__g_q_el(q, el), order=2)._x(0, np.zeros(self.nq_el))
        # diff = g_qq_num - g_qq
        # error = np.linalg.norm(diff)
        # if error > 1.0e-8:
        #     print(f'error g_qq: {error}')
        # return g_qq_num

    def __g_dot_el(self, qe, ue, el):
        g_dot = np.zeros(self.nla_g_el)

        N, N_g , J0, qw = self.N[el], self.N_g[el], self.J0[el], self.qw[el]
        for Ni, N_gi, J0i, qwi in zip(N, N_g, J0, qw):
            NNi = self.stack3r(Ni)

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
            NNi = self.stack3r(Ni)

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

        for i in range(self.nQP):
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

            factor = self.N_g[el, i] * self.J0[el, i] * self.qw[el, i]
         
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
            NNi = self.stack3r(Ni)

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
            NNi = self.stack3r(Ni)

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

# TODO: implement time derivatives of constraint functions
class Euler_Bernoulli_director_integral(Timoshenko_beam_director):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.polynomial_degree_g = self.polynomial_degree_di # if polynomial_degree_g is None else polynomial_degree_g
        self.nn_el_g = self.polynomial_degree_g + 1 # number of nodes per element

        if self.basis == 'B-spline':
            self.knot_vector_g = Knot_vector(self.polynomial_degree_g, self.nEl)
            self.nn_g = self.nEl + self.polynomial_degree_g # number of nodes
        elif self.basis == 'lagrange':
            self.knot_vector_g = Node_vector(self.polynomial_degree_g, self.nEl)
            self.nn_g = self.nEl * self.polynomial_degree_g + 1 # number of nodes

        self.nq_n_g = 8 # number of degrees of freedom per node
        self.nla_g = self.nn_g * self.nq_n_g
        self.nla_g_el = self.nn_el_g * self.nq_n_g

        la_g0 = kwargs.get('la_g0')
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.element_span_g = self.knot_vector_g.element_data
        self.mesh_g = Mesh1D(self.knot_vector_g, self.nQP, derivative_order=0, nq_n=self.nq_n_g, basis=self.basis)
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
        self.g2DOF  = np.arange(6*self.nn_el_g, 7*self.nn_el_g) # unshearability in d2-direction
        self.g3DOF  = np.arange(7*self.nn_el_g, 8*self.nn_el_g) # unshearability in d3-direction

    def __g_el(self, qe, el):
        g = np.zeros(self.nla_g_el)
        
        for i in range(self.nQP):
            NN_dii = self.stack3di(self.N_di[el, i])
            d1 = NN_dii @ qe[self.d1DOF]
            d2 = NN_dii @ qe[self.d2DOF]
            d3 = NN_dii @ qe[self.d3DOF]

            r_xi = self.stack3r(self.N_r_xi[el, i]) @ qe[self.rDOF]

            factor1 = self.N_g[el, i] * self.J0[el, i] * self.qw[el, i]
            factor2 = self.N_g[el, i] * self.qw[el, i]
         
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

    def __g_q_el(self, qe, el):
        # return Numerical_derivative(lambda t, q: self.__g_el(q, el), order=order)._x(0, qe)

        g_q = np.zeros((self.nla_g_el, self.nq_el))
        
        for i in range(self.nQP):
            NN_dii = self.stack3di(self.N_di[el, i])
            d1 = NN_dii @ qe[self.d1DOF]
            d2 = NN_dii @ qe[self.d2DOF]
            d3 = NN_dii @ qe[self.d3DOF]

            d1_NNi = d1 @ NN_dii
            d2_NNi = d2 @ NN_dii
            d3_NNi = d3 @ NN_dii

            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            r_xi = NN_r_xii @ qe[self.rDOF]

            factor1 = self.N_g[el, i] * self.J0[el, i] * self.qw[el, i]
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

        # g_q_num = Numerical_derivative(lambda t, q: self.__g_el(q, el), order=2)._x(0, qe)
        # diff = g_q_num - g_q
        # error = np.linalg.norm(diff)
        # print(f'error g_q: {error}')
        # return g_q_num

    def __g_qq_el(self, qe, el):
        # return Numerical_derivative(lambda t, q: self.__g_q_el(q, el), order=order)._x(0, np.zeros(self.nq_el))

        g_qq = np.zeros((self.nla_g_el, self.nq_el, self.nq_el))
        
        for i in range(self.nQP):
            NN_dii = self.stack3di(self.N_di[el, i])
            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            
            factor1 = NN_dii.T @ NN_dii * self.J0[el, i] * self.qw[el, i]
            factor2 = self.N_g[el, i] * self.qw[el, i]

            ######################
            # director constraints
            ######################
            for j, N_gij in enumerate(self.N_g[el, i]):
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
            arg1 = np.einsum('i,jl,jk->ikl', factor2, NN_dii, NN_r_xii)
            arg2 = np.einsum('i,jl,jk->ikl', factor2, NN_r_xii, NN_dii)
            g_qq[self.g2DOF[:, None, None], self.rDOF[:, None], self.d2DOF] += arg1
            g_qq[self.g2DOF[:, None, None], self.d2DOF[:, None], self.rDOF] += arg2
            g_qq[self.g3DOF[:, None, None], self.rDOF[:, None], self.d3DOF] += arg1
            g_qq[self.g3DOF[:, None, None], self.d3DOF[:, None], self.rDOF] += arg2

        return g_qq

        # g_qq_num = Numerical_derivative(lambda t, q: self.__g_q_el(q, el), order=2)._x(0, np.zeros(self.nq_el))
        # diff = g_qq_num - g_qq
        # error = np.linalg.norm(diff)
        # print(f'error g_qq: {error}')
        # return g_qq_num

    def __g_dot_el(self, qe, ue, el):
        raise NotImplementedError('...')

    def __g_dot_q_el(self, qe, ue, el):
        raise NotImplementedError('...')

    def __g_ddot_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError('...')

    def __g_ddot_q_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError('...')

    def __g_ddot_u_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError('...')

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

# TODO: implement time derivatives of constraint functions
class Inextensible_Euler_Bernoulli_director_integral(Timoshenko_beam_director):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.polynomial_degree_g = self.polynomial_degree_di # if polynomial_degree_g is None else polynomial_degree_g
        self.nn_el_g = self.polynomial_degree_g + 1 # number of nodes per element

        if self.basis == 'B-spline':
            self.knot_vector_g = Knot_vector(self.polynomial_degree_g, self.nEl)
            self.nn_g = self.nEl + self.polynomial_degree_g # number of nodes
        elif self.basis == 'lagrange':
            self.knot_vector_g = Node_vector(self.polynomial_degree_g, self.nEl)
            self.nn_g = self.nEl * self.polynomial_degree_g + 1 # number of nodes

        self.nq_n_g = 9 # number of degrees of freedom per node
        self.nla_g = self.nn_g * self.nq_n_g
        self.nla_g_el = self.nn_el_g * self.nq_n_g

        la_g0 = kwargs.get('la_g0')
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.element_span_g = self.knot_vector_g.element_data
        self.mesh_g = Mesh1D(self.knot_vector_g, self.nQP, derivative_order=0, nq_n=self.nq_n_g, basis=self.basis)
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
        self.g2DOF  = np.arange(6*self.nn_el_g, 7*self.nn_el_g) # unshearability in d2-direction
        self.g3DOF  = np.arange(7*self.nn_el_g, 8*self.nn_el_g) # unshearability in d3-direction
        self.g1DOF  = np.arange(8*self.nn_el_g, 9*self.nn_el_g) # inextensibility

    def __g_el(self, qe, el):
        g = np.zeros(self.nla_g_el)
        
        for i in range(self.nQP):
            NN_dii = self.stack3di(self.N_di[el, i])
            d1 = NN_dii @ qe[self.d1DOF]
            d2 = NN_dii @ qe[self.d2DOF]
            d3 = NN_dii @ qe[self.d3DOF]

            r_xi = self.stack3r(self.N_r_xi[el, i]) @ qe[self.rDOF]

            factor1 = self.N_g[el, i] * self.J0[el, i] * self.qw[el, i]
            factor2 = self.N_g[el, i] * self.qw[el, i]
         
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
            g[self.g1DOF] += (d1 @ r_xi - self.J0[el, i]) * factor2
            # g[self.g1DOF] += (d1 @ r_xi / self.J0[el, i] - 1) * factor1

            # r_s = r_xi / self.J0[el, i]
            # g[self.g1DOF] += (r_s @ r_s - 1) * factor1

        return g

    def __g_q_el(self, qe, el):
        # return Numerical_derivative(lambda t, q: self.__g_el(q, el), order=order)._x(0, qe)

        g_q = np.zeros((self.nla_g_el, self.nq_el))
        
        for i in range(self.nQP):
            NN_dii = self.stack3di(self.N_di[el, i])
            d1 = NN_dii @ qe[self.d1DOF]
            d2 = NN_dii @ qe[self.d2DOF]
            d3 = NN_dii @ qe[self.d3DOF]

            d1_NNi = d1 @ NN_dii
            d2_NNi = d2 @ NN_dii
            d3_NNi = d3 @ NN_dii

            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            r_xi = NN_r_xii @ qe[self.rDOF]

            factor1 = self.N_g[el, i] * self.J0[el, i] * self.qw[el, i]
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
            # g[self.g1DOF] += (d1 @ r_xi - self.J0[el, i]) * factor2
            g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor2, d1 @ NN_r_xii)
            g_q[self.g1DOF[:, None], self.d1DOF] += np.outer(factor2, r_xi @ NN_dii)
            
            # # g[self.g1DOF] += (d1 @ r_xi / self.J0[el, i] - 1) * factor1
            # g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor1, d1 @ NN_r_xii / self.J0[el, i])
            # g_q[self.g1DOF[:, None], self.d1DOF] += np.outer(factor1, r_xi / self.J0[el, i] @ NN_dii)
            
            # r_s = r_xi / self.J0[el, i]
            # # g[self.g1DOF] += (r_s @ r_s - 1) * factor1
            # g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor1, 2 * r_s @ NN_r_xii / self.J0[el, i])

        return g_q

        # g_q_num = Numerical_derivative(lambda t, q: self.__g_el(q, el), order=order)._x(0, qe)
        # diff = g_q_num - g_q
        # error = np.linalg.norm(diff)
        # print(f'error g_q: {error}')
        # return g_q_num

    def __g_qq_el(self, qe, el):
        # return Numerical_derivative(lambda t, q: self.__g_q_el(q, el), order=order)._x(0, np.zeros(self.nq_el))

        g_qq = np.zeros((self.nla_g_el, self.nq_el, self.nq_el))
        
        for i in range(self.nQP):
            NN_dii = self.stack3di(self.N_di[el, i])
            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            
            factor1 = NN_dii.T @ NN_dii * self.J0[el, i] * self.qw[el, i]
            factor2 = self.N_g[el, i] * self.qw[el, i]

            ######################
            # director constraints
            ######################
            for j, N_gij in enumerate(self.N_g[el, i]):
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
            arg1 = np.einsum('i,jl,jk->ikl', factor2, NN_dii, NN_r_xii)
            arg2 = np.einsum('i,jl,jk->ikl', factor2, NN_r_xii, NN_dii)
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
            
            # # g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor1, d1 @ NN_r_xii / self.J0[el, i])
            # # g_q[self.g1DOF[:, None], self.d1DOF] += np.outer(factor1, r_xi / self.J0[el, i] @ NN_dii)
            # g_qq[self.g1DOF[:, None, None], self.rDOF[:, None], self.d1DOF] += arg1
            # g_qq[self.g1DOF[:, None, None], self.d1DOF[:, None], self.rDOF] += arg2
            
            # # # r_s = r_xi / self.J0[el, i]
            # # g_q[self.g1DOF[:, None], self.rDOF] += np.outer(factor1, 2 * r_s @ NN_r_xii / self.J0[el, i])
            # # g_qq[self.g1DOF[:, None, None], self.rDOF[:, None], self.rDOF] += np.einsum('i,jl,jk->ikl', factor2 * self.J0[el, i], NN_r_xii / self.J0[el, i], NN_r_xii / self.J0[el, i])
            # g_qq[self.g1DOF[:, None, None], self.rDOF[:, None], self.rDOF] += np.einsum('i,jl,jk->ikl', 2 * factor2, NN_r_xii, NN_r_xii / self.J0[el, i])

        return g_qq

        # g_qq_num = Numerical_derivative(lambda t, q: self.__g_q_el(q, el), order=order)._x(0, np.zeros(self.nq_el))
        # diff = g_qq_num - g_qq
        # error = np.linalg.norm(diff)
        # print(f'error g_qq: {error}')
        # return g_qq_num

    def __g_dot_el(self, qe, ue, el):
        raise NotImplementedError('...')

    def __g_dot_q_el(self, qe, ue, el):
        raise NotImplementedError('...')

    def __g_ddot_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError('...')

    def __g_ddot_q_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError('...')

    def __g_ddot_u_el(self, qe, ue, ue_dot, el):
        raise NotImplementedError('...')

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
