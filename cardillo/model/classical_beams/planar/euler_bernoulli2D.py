import numpy as np
from math import atan2, sqrt

from cardillo.utility.coo import Coo
from cardillo.discretization import gauss
from cardillo.discretization import uniform_knot_vector, B_spline_basis
from cardillo.math.algebra import norm2
from cardillo.math.numerical_derivative import Numerical_derivative

class Euler_bernoulli2D():
    """Planar Euler-Bernoulli beam using B-spline shape functions.
    """
    def __init__(self, A_rho0, material_model, polynomial_degree, nEl, nQP, Q=None, q0=None, u0=None):        
        # physical parameters
        self.A_rho0 = A_rho0

        # material model
        self.material_model = material_model

        # discretization parameters
        self.polynomial_degree = polynomial_degree # polynomial degree
        self.nQP = nQP # number of quadrature points
        self.nEl = nEl # number of elements

        nn = nEl + polynomial_degree # number of nodes
        self.knot_vector = knot_vector = uniform_knot_vector(polynomial_degree, nEl) # uniform open knot vector
        self.element_span = self.knot_vector[polynomial_degree:-polynomial_degree]

        nn_el = polynomial_degree + 1 # number of nodes per element
        nq_n = 2 # number of degrees of freedom per node (x, y)

        self.nq = nn * nq_n # total number of generalized coordinates
        self.nu = self.nq
        self.nq_el = nn_el * nq_n # total number of generalized coordinates per element

        # compute allocation matrix
        row_offset = np.arange(nEl)
        elDOF_row = (np.zeros((nq_n * nn_el, nEl), dtype=int) + row_offset).T
        elDOF_tile = np.tile(np.arange(0, nn_el), nq_n)
        elDOF_repeat = np.repeat(np.arange(0, nq_n * nn, step=nn), nn_el)
        self.elDOF = elDOF_row + elDOF_tile + elDOF_repeat
            
        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = q0
        self.u0 = u0

        # compute shape functions
        derivative_order = 2
        self.N  = np.empty((nEl, nQP, nn_el))
        self.N_xi = np.empty((nEl, nQP, nn_el))
        self.N_xixi = np.empty((nEl, nQP, nn_el))
        self.qw = np.zeros((nEl, nQP))
        self.xi = np.zeros((nEl, nQP))
        self.J0 = np.zeros((nEl, nQP))
        for el in range(nEl):

            # evaluate Gauss points and weights on [xi^el, xi^{el+1}]
            qp, qw = gauss(nQP, self.element_span[el:el+2])

            # store quadrature points and weights
            self.qw[el] = qw
            self.xi[el] = qp

            # evaluate B-spline shape functions
            N = B_spline_basis(polynomial_degree, derivative_order, knot_vector, qp)
            self.N[el] = N[:, 0]
            self.N_xi[el] = N[:, 1]
            self.N_xixi[el] = N[:, 2]

            # compute change of integral measures
            Qe = self.Q[self.elDOF[el]]
            for i in range(nQP):
                r0_xi = np.kron(np.eye(2), self.N_xi[el, i]) @ Qe
                self.J0[el, i] = norm2(r0_xi)

        # shape functions on the boundary
        N_bdry = np.zeros(nn_el)
        N_bdry[0] = 1
        N_bdry_left = np.kron(np.eye(2), N_bdry)

        N_bdry = np.zeros(nn_el)
        N_bdry[-1] = 1
        N_bdry_right = np.kron(np.eye(2), N_bdry)

        self.N_bdry = np.array([N_bdry_left, N_bdry_right])

    def assembler_callback(self):
        self.__M_coo()

    #########################################
    # equations of motion
    #########################################
    def M_el(self, N, J0, qw):
        Me = np.zeros((self.nq_el, self.nq_el))

        for Ni, J0i, qwi in zip(N, J0, qw):
            # build matrix of shape functions and derivatives
            NNi = np.kron(np.eye(2), Ni)
            
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

    def M(self, t, q, coo):
        coo.extend_sparse(self.__M)

    def f_pot_el(self, qe, Qe, N_xi, N_xixi, J0, qw):
        fe = np.zeros(self.nq_el)

        for N_xii, N_xixii, J0i, qwi in zip(N_xi, N_xixi, J0, qw):
            # build matrix of shape function derivatives
            NN_xii = np.kron(np.eye(2), N_xii)
            NN_xixii = np.kron(np.eye(2), N_xixii)

            # tangential vectors
            t  = NN_xii @ qe
            t0 = NN_xii @ Qe
            n  = NN_xixii @ qe
            n0 = NN_xixii @ Qe
        
            G2_ = t0[0] * t0[0] + t0[1] * t0[1]
            G_ = sqrt(G2_)
            g2_ = t[0] * t[0] + t[1] * t[1]
            g_ = sqrt(g2_)

            # rotated tangential and normal vectors
            t_perp = np.array([-t[1], t[0]])
            n_perp = np.array([-n[1], n[0]])
            t0_perp = np.array([-t0[1], t0[0]])
            # n0_perp = np.array([-n0[1], n0[0]])

            # change of angle
            dtheta = t_perp @ n / g2_
            dtheta0 = t0_perp @ n0 / G2_

            # strain measures
            # lambda_ = g / J0i
            g = g_ / G_
            # g0 = 1 # = G_ / G_
            # kappa = dtheta / J0i
            # kappa0 = dtheta0 / J0i
            kappa = dtheta / G_
            kappa0 = dtheta0 / G_
            
            # evaluate potential
            N = self.material_model.n(g, kappa, kappa0)
            M = self.material_model.m(g, kappa, kappa0)

            # quadrature contribution to element internal force vector
            R1 = NN_xii.T @ (t * N / g_ \
                             - M / g2_ * (2 * dtheta * t + n_perp)
            )

            R2 = NN_xixii.T @ t_perp * M / g2_

            fe -= (R1 + R2) * J0i * qwi

            # fe += NN_xii.T @ (t * N / g) * J0i * qwi

        # print('bending stiffness not active!!!')
        return fe
    
    def f_pot(self, t, q):
        f = np.zeros(self.nu)
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            f[elDOF] += self.f_pot_el(q[elDOF], self.Q[elDOF], self.N_xi[el], self.N_xixi[el], self.J0[el], self.qw[el])
        print(f'f: {f}')
        return f

    def f_pot_q_el(self, qe, Qe, N_xi, N_xixi, J0, qw):    
        fe_q_num = Numerical_derivative(lambda t, qe: self.f_pot_el(qe, Qe, N_xi, N_xixi, J0, qw), order=2)._x(0, qe, eps=1.0e-6)
        return fe_q_num

        # fe_q = np.zeros((self.nq_el, self.nq_el))

        # for N_xii, J0i, qwi in zip(N_xi, J0, qw):
        #     # build matrix of shape function derivatives
        #     NN_xii = np.kron(np.eye(self.dim), N_xii)

        #     # tangential vectors
        #     dr  = NN_xii @ qe 
        #     g = self.norm(dr)
            
        #     # Calculate the strain and stress
        #     lambda_ = g / J0i
        #     n = self.material_model.n(lambda_)
        #     dn = self.material_model.dn(lambda_)
        #     dstress = NN_xii / g * n + np.outer(dr, dr) @ NN_xii / g**2 * (dn / J0i - n / g)

        #     # Calcualte element stiffness matrix
        #     # fe_q -= (NN_xii.T / J0i) @ dstress * J0i * qwi
        #     fe_q -= NN_xii.T @ dstress * qwi

        # # fe_q_num = Numerical_derivative(lambda t, qe: self.f_pot_el(qe, N_xi, J0, qw), order=2)._x(0, qe, eps=1.0e-6)
        # # diff = fe_q_num - fe_q
        # # error = np.linalg.norm(diff)
        # # print(f'error in stiffness matrix: {error:.4e}')
        # # return fe_q_num

        # return fe_q

    def f_pot_q(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            Ke = self.f_pot_q_el(q[elDOF], self.Q[elDOF], self.N_xi[el], self.N_xixi[el], self.J0[el], self.qw[el])

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
            print('local_elDOF can only be computed at frame_ID = (0,) or (1,)')

    def qDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def uDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def r_OP(self, t, q, frame_ID, K_r_SP=None):
        return self.r_OP_q(t, q, frame_ID) @ q

    def r_OP_q(self, t, q, frame_ID, K_r_SP=None):
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[1]
        else:
            print('r_OP_q can only be computed at frame_ID = (0,) or (1,)')

        # interpolate position vector
        r_q = np.zeros((3, self.nq_el))
        r_q[:2] = NN
        return r_q

    def v_P(self, t, q, u, frame_ID, K_r_SP=None):
        return self.r_OP(t, u, frame_ID=frame_ID)

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return self.r_OP(t, u_dot, frame_ID=frame_ID)

    def v_P_q(self, t, q, u, frame_ID, K_r_SP=None):
        return self.r_OP_q(t, None, frame_ID=frame_ID)

    def J_P(self, t, q, frame_ID, K_r_SP=None):
        return self.r_OP_q(t, None, frame_ID=frame_ID)

    def J_P_q(self, t, q, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_el, self.nq_el))

    ####################################################
    # body force
    ####################################################
    # def body_force_pot_el(self, force, t, qe, N, xi, J0, qw):
    #     E_pot = 0
    #     for Ni, xii, J0i, qwi in zip(N, xi, J0, qw):
    #         NNi = np.kron(np.eye(2), Ni)
    #         r_q = np.zeros((3, self.nq_el))
    #         r_q[:2] = NNi
    #         E_pot -= (r_q @ qe) @ force(xii, t) * J0i * qwi
    #     return E_pot

    # def body_force_pot(self, t, q, force):
    #     E_pot = 0
    #     for el in range(self.nEl):
    #         E_pot += self.body_force_pot_el(force, t, q[self.elDOF[el]], self.N[el], self.xi[el], self.J0[el], self.qw[el])
    #     return E_pot

    def body_force_el(self, force, t, N, xi, J0, qw):
        fe = np.zeros(self.nq_el)

        for Ni, xii, J0i, qwi in zip(N, xi, J0, qw):
            NNi = np.kron(np.eye(2), Ni)
            r_q = np.zeros((3, self.nq_el))
            r_q[:2] = NNi
            fe += r_q.T @ force(xii, t) * J0i * qwi
        
        return fe

    def body_force(self, t, q, force):
        f = np.zeros(self.nq)

        for el in range(self.nEl):
            f[self.elDOF[el]] += self.body_force_el(force, t, self.N[el], self.xi[el], self.J0[el], self.qw[el])
        
        print(f'f_body: {f}')
        return f

    def body_force_q(self, t, q, coo, force):
        pass