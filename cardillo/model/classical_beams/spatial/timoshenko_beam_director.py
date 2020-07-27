import numpy as np

from cardillo.utility.coo import Coo
from cardillo.discretization import gauss
from cardillo.discretization import uniform_knot_vector, B_spline_basis, Lagrange_basis
from cardillo.math.algebra import norm3, cross3, e1, e2, e3
from cardillo.math.numerical_derivative import Numerical_derivative

class Timoshenko_beam_director():
    def __init__(self, material_model, A_rho0, B_rho0, C_rho0, polynomial_degree, nQP, nEl, Q, q0=None, u0=None, la_g0=None, B_splines=True):
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

        nn = nEl + polynomial_degree # number of nodes
        self.knot_vector = knot_vector = uniform_knot_vector(polynomial_degree, nEl) # uniform open knot vector
        self.element_span = self.knot_vector[polynomial_degree:-polynomial_degree]

        self.nn_el = nn_el = polynomial_degree + 1 # number of nodes per element
        self.nq_n = nq_n = 12 # number of degrees of freedom per node (x, y, z) + 3 * d_i

        self.nq = nn * nq_n # total number of generalized coordinates
        self.nu = self.nq
        self.nq_el = nn_el * nq_n # total number of generalized coordinates per element
        self.nq_el_r = 3 * nn_el

        # compute allocation matrix
        row_offset = np.arange(nEl)
        elDOF_row = (np.zeros((nq_n * nn_el, nEl), dtype=int) + row_offset).T
        elDOF_tile = np.tile(np.arange(0, nn_el), nq_n)
        elDOF_repeat = np.repeat(np.arange(0, nq_n * nn, step=nn), nn_el)
        self.elDOF = elDOF_row + elDOF_tile + elDOF_repeat

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
        derivative_order = 1
        self.N  = np.empty((nEl, nQP, nn_el))
        self.N_xi = np.empty((nEl, nQP, nn_el))
        self.qw = np.zeros((nEl, nQP))
        self.xi = np.zeros((nEl, nQP))
        self.J0 = np.zeros((nEl, nQP))
        for el in range(nEl):
            if B_splines:
                self.basis_functions = self.__basis_functions_b_splines

                # evaluate Gauss points and weights on [xi^el, xi^{el+1}]
                qp, qw = gauss(nQP, self.element_span[el:el+2])

                # store quadrature points and weights
                self.qw[el] = qw
                self.xi[el] = qp

                # evaluate B-spline shape functions
                self.N[el], self.N_xi[el] = B_spline_basis(polynomial_degree, derivative_order, knot_vector, qp)
            else:
                self.basis_functions = self.__basis_functions_lagrange

                # evaluate Gauss points and weights on [-1, 1]
                qp, qw = gauss(nQP)

                # store quadrature points and weights
                self.qw[el] = qw
                diff_xi = self.element_span[el + 1] - self.element_span[el]
                sum_xi = self.element_span[el + 1] + self.element_span[el]
                self.xi[el] = diff_xi * qp  / 2 + sum_xi / 2
                
                self.N[el], self.N_xi[el] = Lagrange_basis(polynomial_degree, qp, derivative=True)

            # compute change of integral measures
            Qe = self.Q[self.elDOF[el]][self.rDOF]
            for i in range(nQP):
                r0_xi = np.kron(np.eye(3), self.N_xi[el, i]) @ Qe
                self.J0[el, i] = norm3(r0_xi)

        # shape functions on the boundary
        N_bdry, dN_bdry = self.basis_functions(0)
        N_bdry_left = self.stack_shapefunctions(N_bdry)
        dN_bdry_left = self.stack_shapefunctions(dN_bdry)

        N_bdry, dN_bdry = self.basis_functions(1.0 - 1.0e-9)
        N_bdry_right = self.stack_shapefunctions(N_bdry)
        dN_bdry_right = self.stack_shapefunctions(dN_bdry)

        self.N_bdry = np.array([N_bdry_left, N_bdry_right])
        self.dN_bdry = np.array([dN_bdry_left, dN_bdry_right])

    def __basis_functions_b_splines(self, xi):
        return B_spline_basis(self.polynomial_degree, 1, self.knot_vector, xi)

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

    #########################################
    # equations of motion
    #########################################
    def M_el(self, N, J0, qw):
        Me = np.zeros((self.nq_el, self.nq_el))

        for Ni, J0i, qwi in zip(N, J0, qw):
            # build matrix of shape functions and derivatives
            NNi = self.stack_shapefunctions(Ni)
            
            # integrate elemente mass matrix
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
            
    def M(self, t, q, M_coo):
        M_coo.extend(self.M_, (self.uDOF, self.uDOF))

    # #########################################
    # # kinematic equation
    # #########################################
    # def q_dot(self, t, q, u):
    #     return u

    # def q_dot_q(self, t, q, u, coo):
    #     coo.extend(np.zeros((self.nq, self.nq)), (self.qDOF, self.qDOF))

    # def B(self, t, q, coo):
    #     coo.extend(np.eye(self.nq, self.nu), (self.qDOF, self.uDOF))

    # #########################################
    # # bilateral constraints on position level
    # #########################################
    # def g(self, t, q):
    #     d1 = q[3:6]
    #     d2 = q[6:9]
    #     d3 = q[9:]

    #     gap = np.zeros(self.nla_g)
    #     gap[0] = d1 @ d1 - 1
    #     gap[1] = d2 @ d2 - 1
    #     gap[2] = d3 @ d3 - 1
    #     gap[3] = d1 @ d2
    #     gap[4] = d1 @ d3
    #     gap[5] = d2 @ d3

    #     return gap

    # def g_q_dense(self, t, q):

    #     d1 = q[3:6]
    #     d2 = q[6:9]
    #     d3 = q[9:]

    #     gap_q = np.zeros((self.nla_g, self.nq))
    #     gap_q[0, 3:6] = 2 * d1
    #     gap_q[1, 6:9] = 2 * d2
    #     gap_q[2, 9:12] = 2 * d3

    #     gap_q[3, 3:6] = d2
    #     gap_q[3, 6:9] = d1

    #     gap_q[4, 3:6] = d3
    #     gap_q[4, 9:12] = d1

    #     gap_q[5, 6:9] = d3
    #     gap_q[5, 9:12] = d2

    #     # gap_q_num = NumericalDerivativeNew(self.gap_dense, order=2).dR_dq(t, q)
    #     # diff = gap_q - gap_q_num
    #     # np.set_printoptions(precision=3)
    #     # error = np.linalg.norm(diff)
    #     # print(f'error num_tan - tan = {error}')
    #     # return gap_q_num

    #     return gap_q
    
    # def g_qq_dense(self, t, q):

    #     gap_qq = np.zeros((self.nla_g, self.nq, self.nq))
    #     gap_qq[0, 3:6, 3:6] = 2 * np.eye(3)
    #     gap_qq[1, 6:9, 6:9] = 2 * np.eye(3)
    #     gap_qq[2, 9:12, 9:12] = 2 * np.eye(3)
        
    #     gap_qq[3, 3:6, 6:9] = np.eye(3)
    #     gap_qq[3, 6:9, 3:6] = np.eye(3)
        
    #     gap_qq[4, 3:6, 9:12] = np.eye(3)
    #     gap_qq[4, 9:12, 3:6] = np.eye(3)
        
    #     gap_qq[5, 6:9, 9:12] = np.eye(3)
    #     gap_qq[5, 9:12, 6:9] = np.eye(3)

    #     # gap_qq_num = NumericalDerivativeNew(self.gap_q_dense, order=2).dR_dq(t, q)
    #     # diff = gap_qq - gap_qq_num
    #     # np.set_printoptions(precision=3)
    #     # error = np.linalg.norm(diff)
    #     # print(f'error num_tan - tan = {error}')
    #     # return gap_qq_num

    #     return gap_qq

    # def g_q(self, t, q, coo):
    #     coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))
   
    # def W_g(self, t, q, coo):
    #     coo.extend(self.g_q_dense(t, q).T, (self.uDOF, self.la_gDOF))

    # def Wla_g_q(self, t, q, la_g, coo):
    #     dense = np.einsum('ijk,i->jk', self.g_qq_dense(t, q), la_g)
    #     coo.extend(dense, (self.uDOF, self.qDOF))

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
        return self.r_OP_q(t, q, frame_ID) @ q[self.rDOF]

    def r_OP_q(self, t, q, frame_ID, K_r_SP=None):
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[1]
        else:
            N, _ = self.basis_functions(xi)
            NN = self.stack_shapefunctions(N)
        return NN

    def A_IK(self, t, q, frame_ID):
        N, _ = self.basis_functions(frame_ID[0])
        NN = self.stack_shapefunctions(N)
        d1 = NN @ q[self.d1DOF]
        d2 = NN @ q[self.d2DOF]
        d3 = NN @ q[self.d3DOF]
        return np.vstack((d1, d2, d3)).T

    # TODO:
    def A_IK_q(self, t, q, frame_ID):
        A_IK_q_num =  Numerical_derivative(lambda t, q: self.A_IK(t, q, frame_ID=frame_ID))._x(t, q)
        
        # N, _ = self.basis_functions(frame_ID[0])
        # NN = self.stack_shapefunctions(N)
        # A_IK_q = np.zeros((3, 3, self.nq_el))
        # A_IK_q[:, 0] = NN
        # A_IK_q[:, 1] = NN
        # A_IK_q[:, 2] = NN

        return A_IK_q_num

    # def v_P(self, t, q, u, frame_ID, K_r_SP=None):
    #     return self.r_OP(t, u, frame_ID=frame_ID)

    # def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
    #     return self.r_OP(t, u_dot, frame_ID=frame_ID)

    # def v_P_q(self, t, q, u, frame_ID, K_r_SP=None):
    #     return self.r_OP_q(t, None, frame_ID=frame_ID)

    # def J_P(self, t, q, frame_ID, K_r_SP=None):
    #     return self.r_OP_q(t, None, frame_ID=frame_ID)

    # def J_P_q(self, t, q, frame_ID, K_r_SP=None):
    #     return np.zeros((3, self.nq_el, self.nq_el))

    # def K_Omega(self, t, q, u, frame_ID):
    #     _, dNN, _ = self.__basis_functions(frame_ID)
    #     t = dNN @ q
    #     t_perp = np.array([-t[1], t[0]])
    #     g2_ = t[0] * t[0] + t[1] * t[1]
    #     t_dot = dNN @ u
    #     phi_dot = t_perp @ t_dot / g2_

    #     return np.array([0, 0, phi_dot])

    # def K_J_R(self, t, q, frame_ID):
    #     _, dNN = self.__basis_functions(frame_ID)
    #     t = dNN @ q
    #     t_perp = np.array([-t[1], t[0]])
    #     g2_ = t[0] * t[0] + t[1] * t[1]

    #     K_J_R = np.zeros((3, self.nq_el))
    #     K_J_R[2] = t_perp @ dNN / g2_
    #     return K_J_R

    # # TODO
    # def K_J_R_q(self, t, q, frame_ID):
    #     return Numerical_derivative(lambda t, q: self.K_J_R(t, q, frame_ID=frame_ID))._x(t, q)

    # # TODO!
    # ####################################################
    # # body force
    # ####################################################
    # def body_force_el(self, force, t, N, xi, J0, qw):
    #     fe = np.zeros(self.nq_el)
    #     for Ni, xii, J0i, qwi in zip(N, xi, J0, qw):
    #         NNi = self.stack_shapefunctions(Ni)
    #         r_q = np.zeros((3, self.nq_el))
    #         r_q[:2] = NNi
    #         fe += r_q.T @ force(xii, t) * J0i * qwi
    #     return fe

    # def body_force(self, t, q, force):
    #     f = np.zeros(self.nq)
    #     for el in range(self.nEl):
    #         f[self.elDOF[el]] += self.body_force_el(force, t, self.N[el], self.xi[el], self.J0[el], self.qw[el])
    #     return f

    # def body_force_q(self, t, q, coo, force):
    #     pass

    # ####################################################
    # # visualization
    # ####################################################
    # def centerline(self, q, n=100):
    #     q_body = q[self.qDOF]
    #     r = []
    #     for xi in np.linspace(0, 1 - 1.0e-9, n):
    #         frame_ID = (xi,)
    #         qp = q_body[self.qDOF_P(frame_ID)]
    #         r.append( self.r_OP(1, qp, frame_ID) )
    #     return np.array(r)

####################################################
# straight initial configuration
####################################################
def straight_configuration(polynomial_degree, nEl, L, greville_abscissae=True):
    nn = polynomial_degree + nEl
    
    X = np.linspace(0, L, num=nn)
    Y = np.zeros(nn)
    Z = np.zeros(nn)
    if greville_abscissae:
        kv = uniform_knot_vector(polynomial_degree, nEl)
        for i in range(nn):
            X[i] = np.sum(kv[i+1:i+polynomial_degree+1])
        X = X * L / polynomial_degree
    
    # compute reference directors
    D1 = np.repeat(e1, nn)
    D2 = np.repeat(e2, nn)
    D3 = np.repeat(e3, nn)
    
    # assemble all reference generalized coordinates
    return np.concatenate([X, Y, Z, D1, D2, D3])
