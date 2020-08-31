import numpy as np
from math import atan2, sqrt

from cardillo.utility.coo import Coo
from cardillo.discretization import gauss
from cardillo.discretization import uniform_knot_vector, B_spline_basis1D
from cardillo.math.algebra import ax2skew, norm2, norm3, cross3, e3
from cardillo.math.numerical_derivative import Numerical_derivative

class Euler_bernoulli():
    """Planar Euler-Bernoulli beam using B-spline shape functions.
    """
    def __init__(self, A_rho0, material_model, polynomial_degree, nEl, nQP, Q, q0=None, u0=None):
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
        self.element_span = np.array(self.knot_vector[polynomial_degree:-polynomial_degree])

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
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

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
            self.N[el], self.N_xi[el], self.N_xixi[el] = B_spline_basis1D(polynomial_degree, derivative_order, knot_vector, qp).transpose(2, 0, 1)

            # compute change of integral measures
            Qe = self.Q[self.elDOF[el]]
            for i in range(nQP):
                r0_xi = self.stack_shapefunctions(self.N_xi[el, i]) @ Qe
                self.J0[el, i] = norm2(r0_xi)

        # shape functions on the boundary
        N_bdry, dN_bdry = B_spline_basis1D(self.polynomial_degree, 1, self.knot_vector, 0).T
        N_bdry_left = self.stack_shapefunctions(N_bdry)
        dN_bdry_left = self.stack_shapefunctions(dN_bdry)

        N_bdry, dN_bdry = B_spline_basis1D(self.polynomial_degree, 1, self.knot_vector, 1).T
        N_bdry_right = self.stack_shapefunctions(N_bdry)
        dN_bdry_right = self.stack_shapefunctions(dN_bdry)

        self.N_bdry = np.array([N_bdry_left, N_bdry_right])
        self.dN_bdry = np.array([dN_bdry_left, dN_bdry_right])

    def assembler_callback(self):
        self.__M_coo()

    def __basis_functions(self, frame_ID):
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
            dNN = self.dN_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[-1]
            dNN = self.dN_bdry[-1]
        else:
            N, dN = B_spline_basis1D(self.polynomial_degree, 1, self.knot_vector, xi).T
            NN = self.stack_shapefunctions(N)
            dNN = self.stack_shapefunctions(dN)
        return NN, dNN

    def stack_shapefunctions(self, N):
        # return np.kron(np.eye(2), N)
        n2 = int(self.nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, :n2] = N
        NN[1, n2:] = N
        return NN

    def stack_shapefunctions_perp(self, N):
        # return np.kron(np.array([[0, -1], [1, 0]]), N)
        n2 = int(self.nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, n2:] = -N
        NN[1, :n2] = N
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

    def M(self, t, q, coo):
        coo.extend_sparse(self.__M)

    def f_pot_el(self, qe, Qe, N_xi, N_xixi, J0, qw):
        fe = np.zeros(self.nq_el)

        for N_xii, N_xixii, qwi in zip(N_xi, N_xixi, qw):
            # build matrix of shape function derivatives
            NN_xii = self.stack_shapefunctions(N_xii)
            NN_xixii = self.stack_shapefunctions(N_xixii)

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
            theta_bar_xi = t_perp @ n / g2_
            theta0_bar_xi = t0_perp @ n0 / G2_

            # strain measures
            g = g_ / G_
            kappa = theta_bar_xi / G_
            kappa0 = theta0_bar_xi / G_
            
            # evaluate material model
            N = self.material_model.n(g, kappa, kappa0)
            M = self.material_model.m(g, kappa, kappa0)

            # quadrature contribution to element internal force vector
            R1 = NN_xii.T @ (t * N / g_ \
                             - M / g2_ * (2 * theta_bar_xi * t + n_perp)
            )

            R2 = NN_xixii.T @ t_perp * M / g2_

            fe -= (R1 + R2) * qwi

        return fe
    
    def f_pot(self, t, q):
        f = np.zeros(self.nu)
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            f[elDOF] += self.f_pot_el(q[elDOF], self.Q[elDOF], self.N_xi[el], self.N_xixi[el], self.J0[el], self.qw[el])
        return f

    def f_pot_q_el(self, qe, Qe, N_xi, N_xixi, J0, qw):        
        fe_q = np.zeros((self.nq_el, self.nq_el))

        for N_xii, N_xixii, qwi in zip(N_xi, N_xixi, qw):
            # build matrix of shape function derivatives
            NN_xii = self.stack_shapefunctions(N_xii)
            NN_xixii = self.stack_shapefunctions(N_xixii)

            NN_xii_perp = self.stack_shapefunctions_perp(N_xii)
            NN_xixii_perp = self.stack_shapefunctions_perp(N_xixii)

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

            # change of angle
            theta_bar_xi = t_perp @ n / g2_
            theta0_bar_xi = t0_perp @ n0 / G2_

            # strain measures
            g = g_ / G_
            kappa = theta_bar_xi / G_
            kappa0 = theta0_bar_xi / G_

            # auxiliary functions
            g_bar_q = t @ NN_xii / g_
            theta_bar_xi_q = (n @ NN_xii_perp + t_perp @ NN_xixii) / g2_ \
                            - 2 * theta_bar_xi / g_ * g_bar_q

            # derivative of strain measures
            g_q = g_bar_q / G_
            kappa_q = theta_bar_xi_q / G_
            
            # evaluate material model
            N = self.material_model.n(g, kappa, kappa0)
            M = self.material_model.m(g, kappa, kappa0)
        
            N_g = self.material_model.n_lambda(g, kappa, kappa0)
            N_kappa = self.material_model.n_kappa(g, kappa, kappa0)
            M_g = self.material_model.m_lambda(g, kappa, kappa0)
            M_kappa = self.material_model.m_kappa(g, kappa, kappa0)

            N_q = N_g * g_q + N_kappa * kappa_q # we need the derivatives of g w.r.t. q not \overline{g}
            M_q = M_g * g_q + M_kappa * kappa_q # we need the derivatives of g w.r.t. q not \overline{g}
            
            # auxiliary functions and their derivatives
            k1 = t * N / g_
            k2 = M / g2_
            k3 = 2 * theta_bar_xi * t
            k1_q = np.outer(-k1 / g_, g_bar_q) + (N * NN_xii + np.outer(t, N_q)) / g_
            k2_q = - 2 * k2 / g_ * g_bar_q + M_q / g2_
            k3_q = 2 * (np.outer(t, theta_bar_xi_q) + theta_bar_xi * NN_xii)
            
            # quadrature contribution to element stiffness matrix
            fe_q -= ( \
                NN_xii.T @ (k1_q - np.outer(k3 + n_perp, k2_q) - k2 * (k3_q + NN_xixii_perp) ) \
                + NN_xixii.T @ (k2 * NN_xii_perp + np.outer(t_perp, k2_q)) \
                    ) * qwi
        
        # fe_q_num = Numerical_derivative(lambda t, qe: self.f_pot_el(qe, Qe, N_xi, N_xixi, J0, qw), order=2)._x(0, qe, eps=1.0e-6)
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
            el = np.where(xi >= self.element_span)[0][-1]
            return self.elDOF[el]

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
            N = B_spline_basis1D(self.polynomial_degree, 0, self.knot_vector, xi)
            NN = self.stack_shapefunctions(N)

        # interpolate position vector
        r_q = np.zeros((3, self.nq_el))
        r_q[:2] = NN
        return r_q

    def A_IK(self, t, q, frame_ID):
        _, dNN = self.__basis_functions(frame_ID)
        t = np.zeros(3)
        t[:2] = dNN @ q
        d1 = t / norm3(t)
        d2 = cross3(e3, d1)
        A_IK = np.eye(3)
        A_IK[:, 0] = d1
        A_IK[:, 1] = d2
        return A_IK

    def A_IK_q(self, t, q, frame_ID):
        _, dNN = self.__basis_functions(frame_ID)
        t = np.zeros(3)
        t_ = dNN @ q
        t[:2] = t_
        g_ = norm3(t)
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
        _, dNN = self.__basis_functions(frame_ID)
        t = dNN @ q
        t_perp = np.array([-t[1], t[0]])
        g2_ = t[0] * t[0] + t[1] * t[1]
        t_dot = dNN @ u
        phi_dot = t_perp @ t_dot / g2_

        return np.array([0, 0, phi_dot])

    def K_Omega_q(self, t, q, u, frame_ID):
        return np.zeros((3, self.nq_el))

    def K_Psi(self, t, q, u, u_dot, frame_ID):
        _, dNN = self.__basis_functions(frame_ID)
        t = dNN @ q
        t_perp = np.array([-t[1], t[0]])
        g2_ = t[0] * t[0] + t[1] * t[1]
        g4_ = g2_ * g2_
        t_dot = dNN @ u
        t_dot_perp = np.array([-t_dot[1], t_dot[0]])
        t_ddot = dNN @ u_dot
        phi_ddot = (t_perp @ t_ddot + t_dot @ t_dot_perp) / g2_ - (t_perp @ t_dot) / g4_ * 2 * t @ t_dot

        return np.array([0, 0, phi_ddot])

    def K_J_R(self, t, q, frame_ID):
        _, dNN = self.__basis_functions(frame_ID)
        t = dNN @ q
        t_perp = np.array([-t[1], t[0]])
        g2_ = t[0] * t[0] + t[1] * t[1]

        K_J_R = np.zeros((3, self.nq_el))
        K_J_R[2] = t_perp @ dNN / g2_
        return K_J_R

    def K_J_R_q(self, t, q, frame_ID):
        _, dN = B_spline_basis1D(self.polynomial_degree, 1, self.knot_vector, frame_ID[0]).T
        dNN = self.stack_shapefunctions(dN)
        dNN_perp = self.stack_shapefunctions_perp(dN)
        t = dNN @ q
        t_perp = dNN_perp @ q
        g2_ = t[0] * t[0] + t[1] * t[1]

        K_J_R_q = np.zeros((3, self.nq_el, self.nq_el))
        K_J_R_q[2] = np.einsum('ik,ij->jk', dNN_perp, dNN) / g2_ - (2 / g2_**2) * np.outer(t_perp @ dNN, t @ dNN)
        return K_J_R_q
 
        # K_J_R_q_num = Numerical_derivative(lambda t, q: self.K_J_R(t, q, frame_ID=frame_ID), order=2)._x(t, q)
        # diff = K_J_R_q - K_J_R_q_num
        # error = np.max(np.abs(diff))
        # print(f'error K_J_R_q: {error}')
        # return K_J_R_q_num

    ####################################################
    # body force
    ####################################################
    def body_force_el(self, force, t, N, xi, J0, qw):
        fe = np.zeros(self.nq_el)
        for Ni, xii, J0i, qwi in zip(N, xi, J0, qw):
            NNi = self.stack_shapefunctions(Ni)
            r_q = np.zeros((3, self.nq_el))
            r_q[:2] = NNi
            fe += r_q.T @ force(xii, t) * J0i * qwi
        return fe

    def body_force(self, t, q, force):
        f = np.zeros(self.nq)
        for el in range(self.nEl):
            f[self.elDOF[el]] += self.body_force_el(force, t, self.N[el], self.xi[el], self.J0[el], self.qw[el])
        return f

    def body_force_q(self, t, q, coo, force):
        pass

    ####################################################
    # visualization
    ####################################################
    def centerline(self, q, n=100):
        q_body = q[self.qDOF]
        r = []
        for xi in np.linspace(0, 1, n):
            frame_ID = (xi,)
            qp = q_body[self.qDOF_P(frame_ID)]
            r.append( self.r_OP(1, qp, frame_ID) )
        return np.array(r)

class Inextensible_Euler_bernoulli(Euler_bernoulli):
    def __init__(self, *args, la_g0=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.polynomial_degree_g = self.polynomial_degree - 1
        self.nn_el_g = self.polynomial_degree_g + 1 # number of nodes per element
        self.nn_g = self.nEl + self.polynomial_degree_g # number of nodes
        self.nq_n_g = 1 # number of degrees of freedom per node
        self.nla_g = self.nn_g * self.nq_n_g
        self.nla_g_el = self.nn_el_g * self.nq_n_g

        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0
        self.knot_vector_g = uniform_knot_vector(self.polynomial_degree_g, self.nEl) # uniform open knot vector
        self.element_span_g = self.knot_vector_g[self.polynomial_degree_g:-self.polynomial_degree_g]

        row_offset = np.arange(self.nEl)
        elDOF_row = (np.zeros((self.nq_n_g * self.nn_el_g, self.nEl), dtype=int) + row_offset).T
        elDOF_tile = np.tile(np.arange(0, self.nn_el_g), self.nq_n_g)
        elDOF_repeat = np.repeat(np.arange(0, self.nq_n_g * self.nn_g, step=self.nn_g), self.nn_el_g)
        self.elDOF_g = elDOF_row + elDOF_tile + elDOF_repeat

        # compute shape functions
        self.N_g = np.empty((self.nEl, self.nQP, self.nn_el_g))
        for el in range(self.nEl):
            # evaluate Gauss points and weights on [xi^el, xi^{el+1}]
            qp, _ = gauss(self.nQP, self.element_span_g[el:el+2])

            # evaluate B-spline shape functions
            self.N_g[el] = B_spline_basis1D(self.polynomial_degree_g, 0, self.knot_vector_g, qp)

    def __g_el(self, qe, N_xi, N_g, J0, qw):
        g = np.zeros(self.nla_g_el)

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack_shapefunctions(N_xii)

            r_xi = NN_xii @ qe

            g += (r_xi @ r_xi / J0i - J0i) * N_gi * qwi

        return g

    def __g_q_el(self, qe, N_xi, N_g, J0, qw):
        g_q = np.zeros((self.nla_g_el, self.nq_el))

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack_shapefunctions(N_xii)

            r_xi = NN_xii @ qe
            
            g_q += np.outer(2 * N_gi * qwi / J0i, r_xi @ NN_xii)

        return g_q

        # g_q_num = Numerical_derivative(lambda t, q: self.__g_el(q, N, N_xi, N_g, J0, qw), order=2)._x(0, qe)
        # diff = g_q_num - g_q
        # error = np.linalg.norm(diff)
        # print(f'error g_q: {error}')
        # return g_q_num

    def __g_dot_el(self, qe, ue, N_xi, N_g, J0, qw):
        g_dot = np.zeros(self.nla_g_el)

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack_shapefunctions(N_xii)

            r_xi = NN_xii @ qe
            r_xi_dot = NN_xii @ ue

            g_dot += 2 * r_xi @ r_xi_dot / J0i * N_gi * qwi

        return g_dot

    def __g_dot_q_el(self, qe, ue, N_xi, N_g, J0, qw):
        g_dot_q = np.zeros((self.nla_g_el, self.nq_el))

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack_shapefunctions(N_xii)

            r_xi_dot = NN_xii @ ue

            g_dot_q += np.outer(N_gi, 2 * r_xi_dot @ NN_xii / J0i * qwi)

        return g_dot_q

    def __g_ddot_el(self, qe, ue, ue_dot, N_xi, N_g, J0, qw):
        g_ddot = np.zeros(self.nla_g_el)

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack_shapefunctions(N_xii)

            r_xi = NN_xii @ qe
            r_xi_dot = NN_xii @ ue
            r_xi_ddot = NN_xii @ ue_dot

            g_ddot += (r_xi @ r_xi_ddot + r_xi_dot @ r_xi_dot) * 2 / J0i * N_gi * qwi

        return g_ddot

    def __g_ddot_q_el(self, qe, ue, ue_dot, N_xi, N_g, J0, qw):
        g_ddot_q = np.zeros((self.nla_g_el, self.nq_el))

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack_shapefunctions(N_xii)

            r_xi_ddot = NN_xii @ ue_dot

            g_ddot_q += np.outer(N_gi, (r_xi_ddot @ NN_xii) * 2 / J0i * qwi)

        return g_ddot_q

    def __g_ddot_u_el(self, qe, ue, ue_dot, N_xi, N_g, J0, qw):
        g_ddot_u = np.zeros((self.nla_g_el, self.nq_el))

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack_shapefunctions(N_xii)

            r_xi_dot = NN_xii @ ue

            g_ddot_u += np.outer(N_gi, (r_xi_dot @ NN_xii) * 4 / J0i * qwi)

        return g_ddot_u

    

    def __g_qq_el(self, qe, N_xi, N_g, J0, qw):
        g_qq = np.zeros((self.nla_g_el, self.nq_el, self.nq_el))

        for N_xii, N_gi, J0i, qwi in zip(N_xi, N_g, J0, qw):
            NN_xii = self.stack_shapefunctions(N_xii)

            g_qq += np.einsum('i,jl,jk->ikl', 2 * N_gi * qwi / J0i, NN_xii, NN_xii)

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
            g[elDOF_g] += self.__g_el(q[elDOF], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
        return g

    def g_q(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_q = self.__g_q_el(q[elDOF], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
            coo.extend(g_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))

    def W_g(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_q = self.__g_q_el(q[elDOF], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
            coo.extend(g_q.T, (self.uDOF[elDOF], self.la_gDOF[elDOF_g]))

    def Wla_g_q(self, t, q, la_g, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_qq = self.__g_qq_el(q[elDOF], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
            coo.extend(np.einsum('i,ijk->jk', la_g[elDOF_g], g_qq), (self.uDOF[elDOF], self.qDOF[elDOF]))

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g)
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_dot[elDOF_g] += self.__g_dot_el(q[elDOF], u[elDOF], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
        return g_dot

    def g_dot_q(self, t, q, u, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_dot_q = self.__g_dot_q_el(q[elDOF], u[elDOF], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
            coo.extend(g_dot_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))  

    def g_dot_u(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_dot_u = self.__g_q_el(q[elDOF], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
            coo.extend(g_dot_u, (self.la_gDOF[elDOF_g], self.uDOF[elDOF]))  

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g)
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_ddot[elDOF_g] += self.__g_ddot_el(q[elDOF], u[elDOF], u_dot[elDOF], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
        return g_ddot

    def g_ddot_q(self, t, q, u, u_dot, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_ddot_q = self.__g_ddot_q_el(q[elDOF], u[elDOF], u_dot[elDOF], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
            coo.extend(g_ddot_q, (self.la_gDOF[elDOF_g], self.qDOF[elDOF]))  

    def g_ddot_u(self, t, q, u, u_dot, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            elDOF_g = self.elDOF_g[el]
            g_ddot_u = self.__g_ddot_u_el(q[elDOF], u[elDOF], u_dot[elDOF], self.N_xi[el], self.N_g[el], self.J0[el], self.qw[el])
            coo.extend(g_ddot_u, (self.la_gDOF[elDOF_g], self.uDOF[elDOF]))  