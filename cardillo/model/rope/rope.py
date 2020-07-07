import numpy as np

from cardillo.utility.coo import Coo
from cardillo.discretization import gauss
from cardillo.discretization import uniform_knot_vector, B_spline_basis
from cardillo.math.algebra import norm2, norm3
from cardillo.math.numerical_derivative import Numerical_derivative

class Rope(object):
    def __init__(self, A_rho0, material_model, polynomial_degree, nEl, nQP, Q=None, q0=None, u0=None, B_splines=True, dim=3):
        self.dim = dim
        if dim == 2:
            self.norm = norm2
        elif dim == 3:
            self.norm = norm3
        else:
            raise ValueError('dim has to be 2 or 3')
        
        # physical parameters
        self.A_rho0 = A_rho0

        # material model
        self.material_model = material_model

        # discretization parameters
        self.polynomial_degree = polynomial_degree # polynomial degree
        self.nQP = nQP # number of quadrature points
        self.nEl = nEl # number of elements

        if B_splines:
            nn = nEl + p # number of nodes
            self.knot_vector = knot_vector = uniform_knot_vector(p, nEl) # uniform open knot vector
            self.element_span = self.knot_vector[p:-p] # TODO!
        else:
            nn = nEl * p + 1 # number of nodes
            self.element_span = np.linspace(0, 1, nEl + 1)

        nn_el = p + 1 # number of nodes per element
        nq_n = dim # number of degrees of freedom per node

        self.nq = nn * nq_n # total number of generalized coordinates
        self.nu = self.nq
        self.nq_el = nn_el * nq_n # total number of generalized coordinates per element

        # compute allocation matrix
        elDOF_nEl = (np.zeros((nq_n * nn_el, nEl), dtype=int) + np.arange(nEl)).T
        elDOF_tile = np.tile(np.arange(0, nn_el), nq_n)
        elDOF_repeat = np.repeat(np.arange(0, nq_n * nn, step=nn), nn_el)
        self.elDOF = elDOF_nEl + elDOF_tile + elDOF_repeat

        # TODO: do we need nodal degrees of freedom?
        # tmp3 = (np.zeros((self.nNDOF, nNd), dtype=int) + np.arange(nNd)).T
        # tmp4 = np.tile(np.arange(0, nNDOF * nNd, step=nNd), nNd).reshape((nNd, nNDOF))
        # self.nodalDOF = tmp3 + tmp4
            
        # reference generalized coordinates, initial coordinates and initial velocities
        # TODO: Greville abscissae/ check 2D or 3D
        # X0 = np.linspace(0, L, nn)
        # Y0 = np.zeros_like(X0)
        # Z0 = np.zeros_like(X0)
        # self.Q = np.hstack((X0, Y0, Z0)) if Q is None else Q
        # self.q0 = np.hstack((X0, Y0, Z0)) if q0 is None else q0
        # self.u0 = np.zeros(self.nu) if u0 is None else u0
        self.Q = Q
        self.q0 = q0
        self.u0 = u0

        # compute shape functions
        derivative_order = 1
        self.N  = np.empty((nEl, nQP, nn_el))
        self.N_xi = np.empty((nEl, nQP, nn_el))
        self.qw = np.zeros((nEl, nQP))
        self.qp = np.zeros((nEl, nQP))
        self.J0 = np.zeros((nEl, nQP)) # TODO
        for el in range(nEl):
            delta_xi = self.element_span[el + 1] - self.element_span[el]
            if B_splines:
                # evaluate Gauss points and weights on [xi^el, xi^{el+1}]
                qp, qw = gauss(nQP, self.element_span[el:el+2])

                # store quadrature points and weights
                self.qp[el] = qp
                self.qw[el] = qw

                # evaluate B-spline shape functions
                N_dN = B_spline_basis(polynomial_degree, derivative_order, knot_vector, qp)
                # ordering: (number of evaluation points, derivative number, nonzero shape functions)
                self.N[el] = N_dN[:, 0]
                self.N_xi[el] = N_dN[:, 1]
                # self.N_s[el] = N_dN[:, 1] / G
            else:
                # evaluate Gauss points and weights on [-1, 1]
                qp, qw = gauss(nQP)

                # store quadrature weights
                self.qw[el] = qw

                raise NotImplementedError('not implemented')
                # N_dN = lagrange_basis(degree, 1, qp)
                # self.N[el] = N_dN[:, 0]
                # self.dN[el] = N_dN[:, 1]

            # TODO: doc me!
            Qe = self.Q[self.elDOF[el]]
            for i in range(nQP):
                r0_xi = np.kron(np.eye(self.dim), self.N_xi[el, i]) @ Qe
                self.J0[el, i] = self.norm(r0_xi)

        # shape functions on the boundary
        N_bdry = np.zeros(nn_el)
        N_bdry[0] = 1
        N_bdry_left = np.kron(np.eye(dim), N_bdry)

        N_bdry = np.zeros(nn_el)
        N_bdry[-1] = 1
        N_bdry_right = np.kron(np.eye(dim), N_bdry)

        self.N_bdry = np.array([N_bdry_left, N_bdry_right])

        # TODO: store constant mass matrix

    def M_el(self, N, J0, qw):
        Me = np.zeros((self.nq_el, self.nq_el))

        for Ni, J0i, qwi in zip(N, J0, qw):
            # build matrix of shape functions and derivatives
            NNi = np.kron(np.eye(self.dim), Ni)
            
            # integrate elemente mass matrix
            Me += NNi.T @ NNi * self.A_rho0 * J0i * qwi

        return Me

    # TODO: compute constant mass matrix within an assembler callback
    def M(self, t, q, coo):
        for el in range(self.nEl):
            # extract element degrees of freedom
            elDOF = self.elDOF[el, :]

            # compute element mass matrix
            Me = self.M_el(self.N[el], self.J0[el], self.qw[el])
            
            # sparse assemble element mass matrix
            coo.extend(Me, (self.uDOF[elDOF], self.uDOF[elDOF]))
            
    def f_pot_el(self, qe, N_xi, J0, qw):
        fe = np.zeros(self.nq_el)

        for N_xii, J0i, qwi in zip(N_xi, J0, qw):
            # build matrix of shape function derivatives
            NN_xii = np.kron(np.eye(self.dim), N_xii)

            # tangential vectors
            dr  = NN_xii @ qe 
            g = self.norm(dr)
            
            # Calculate the strain and stress
            lambda_ = g / J0i
            stress = self.material_model.n(lambda_) * dr / g

            # integrate element force vector
            # fe -= (NN_xii.T / J0i) @ stress * J0i * qwi
            fe -= NN_xii.T @ stress * qwi

        # print(f'fe: {fe.T}')
        return fe
    
    def f_pot(self, t, q):
        f = np.zeros(self.nu)

        for el in range(self.nEl):
            # extract element degrees of freedom
            elDOF = self.elDOF[el]

            # assemble internal element potential forces
            f[elDOF] += self.f_pot_el(q[elDOF], self.N_xi[el], self.J0[el], self.qw[el])
                    
        # print(f'f: {f.T}')
        return f

    def f_pot_q_el(self, qe, N_xi, J0, qw):
        fe_q_num = Numerical_derivative(lambda t, qe: self.f_pot_el(qe, N_xi, J0, qw), order=2)._x(0, qe, eps=1.0e-6)
        return fe_q_num
        
        # fe_q = np.zeros((self.nq_el, self.nq_el))

        # for dNi, qwi in zip(dN, qw):
        #     # build matrix of shape function derivatives
        #     dNNi = np.kron(np.eye(self.dim), dNi)
            
        #     # compute current and reference tangent vector w.r.t. [-1, 1]
        #     dr  = dNNi @ qe 
        #     dr0 = dNNi @ Qe
        #     g = norm2(dr)
        #     G = norm2(dr0)
            
        #     # Calculate the strain and stress
        #     strain = g / G
        #     n = self.material_model.n(strain)
        #     dn = self.material_model.dn(strain)
        #     dstress = dNNi / g * n + np.outer(dr, dr) @ dNNi / g**2 * (dn / G - n / g)

        #     # Calcualte element stiffness matrix
        #     fe_q -= dNNi.T @ dstress * qwi

        # # # np.set_printoptions(3)
        # # diff = fe_q_num - fe_q
        # # # # print(diff)
        # # # print(f'fe_q_num =\n{fe_q_num}')
        # # # print(f'fe_q =\n{fe_q}')
        # # error = np.linalg.norm(diff)
        # # print(f'error in stiffness matrix: {error:.4e}')
        # # return fe_q_num

        # return fe_q

    def f_pot_q(self, t, q, coo):
        for el in range(self.nEl):
            # extract element degrees of freedom
            elDOF = self.elDOF[el, :]

            # integrate element internal stiffness matrix
            Ke = self.f_pot_q_el(q[elDOF], self.N_xi[el], self.J0[el], self.qw[el])

            # sparse assemble element internal stiffness matrix
            coo.extend(Ke, (self.uDOF[elDOF], self.qDOF[elDOF]))

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
        elDOF = self.elDOF_P(frame_ID)
        return self.qDOF[elDOF]

    def uDOF_P(self, frame_ID):
        elDOF = self.elDOF_P(frame_ID)
        return self.uDOF[elDOF]

    def r_OP(self, t, q, frame_ID, K_r_SP=None):
        # xi = frame_ID[0]
        # if xi == 0:
        #     NN = self.N_bdry[0]
        # elif xi == 1:
        #     NN = self.N_bdry[1]
        # else:
        #     print('r_OP can only be computed at frame_ID = (0,) or (1,)')

        # # interpolate position vector
        # r = np.zeros(3)
        # r[:self.dim] = NN @ q
        # return r

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
        r_q[:self.dim] = NN
        return r_q

    def v_P(self, t, q, u, frame_ID, K_r_SP=None):
        return self.r_OP(t, u, frame_ID=frame_ID)

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return self.r_OP(t, u_dot, frame_ID=frame_ID)

    def v_P_q(self, t, q, u, frame_ID, K_r_SP=None):
        return self.r_OP_q(t, None, frame_ID=frame_ID)

    def J_P(self, t, q, frame_ID, K_r_SP=None):
        return self.r_OP_q(t, None, frame_ID, K_r_SP)

    def J_P_q(self, t, q, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_el, self.nq_el))

statics = True
# statics = False

if __name__ == "__main__":
    from cardillo.model.rope.hooke import Hooke
    from cardillo.model.frame import Frame
    from cardillo.model.bilateral_constraints import Spherical_joint
    from cardillo.model import Model
    from cardillo.solver import Euler_backward, Moreau, Moreau_sym, Generalized_alpha_1, Scipy_ivp ,Newton
    from cardillo.model.line_force.line_force import Line_force

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    
    # physical properties of the rope
    dim = 3
    assert dim == 3
    L = 50
    r = 3.0e-3
    A = np.pi * r**2
    EA = 4.0e8 * A
    material_model = Hooke(EA)
    A_rho0 = 10 * A

    # discretization properties
    B_splines = True
    p = 2
    nQP = int(np.ceil((p + 1)**2 / 2))
    print(f'nQP: {nQP}')
    nEl = 15

    # build reference configuration
    if B_splines:
        nNd = nEl + p
    else:
        nNd = nEl * p + 1
    X0 = np.linspace(0, L, nNd)
    Xi = uniform_knot_vector(p, nEl)
    for i in range(nNd):
        X0[i] = np.sum(Xi[i+1:i+p+1])
    X0 = X0 * L / p
    Y0 = np.zeros_like(X0)
    Z0 = np.zeros_like(X0)
    Q = np.hstack((X0, Y0, Z0))
    u0 = np.zeros_like(Q)

    # X0 = np.linspace(0, L, nNd)
    q0 = np.hstack((X0, Y0, Z0))

    # excitation of the initial configuration
    fac = 1.0e1
    q0[nNd+1:2*nNd-1] += np.random.rand(nNd - 2) * fac
    if dim == 3:
        q0[2*nNd+1:3*nNd-1] += np.random.rand(nNd - 2) * fac

    rope = Rope(A_rho0 * 1.0e-10, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0, B_splines=B_splines, dim=dim)

    # # left joint
    # r_OB1 = np.zeros(3)
    # frame_left = Frame(r_OP=r_OB1)
    # joint_left = Spherical_joint(frame_left, rope, r_OB1, frame_ID2=(0,))

    omega = 2 * np.pi
    A = 5
    r_OB1 = lambda t: np.array([0, 0, A * np.sin(omega * t)])
    r_OB1_t = lambda t: np.array([0, 0, A * omega * np.cos(omega * t)])
    r_OB1_tt = lambda t: np.array([0, 0, -A * omega**2 * np.sin(omega * t)])
    frame_left = Frame(r_OP=r_OB1, r_OP_t=r_OB1_t, r_OP_tt=r_OB1_tt)
    joint_left = Spherical_joint(frame_left, rope, r_OB1(0), frame_ID2=(0,))

    # right joint
    r_OB2 = np.array([1.2 * L, 0, 0])
    frame_right = Frame(r_OP=r_OB2)
    joint_right = Spherical_joint(rope, frame_right, r_OB2, frame_ID1=(1,))

    # gravity
    g = np.array([0, 0, - A_rho0 * L * 9.81]) * 1.0e3
    if statics:
        f_g = Line_force(lambda xi, t: t * g, rope)
    else:
        f_g = Line_force(lambda xi, t: g, rope)

    # assemble the model
    model = Model()
    model.add(rope)
    model.add(frame_left)
    model.add(joint_left)
    model.add(frame_right)
    model.add(joint_right)
    model.add(f_g)
    model.assemble()

    if statics:
        solver = Newton(model, n_load_stepts=5, max_iter=10)
        t, q, la = solver.solve()
    else:
        t0 = 0
        t1 = 10
        dt = 1e-1
        t_span = t0, t1
        # solver = Euler_backward(model, t_span=t_span, dt=dt, numerical_jacobian=False, debug=False)
        # t, q, u, la_g, la_gamma = solver.solve()
        # solver = Euler_backward(model, t_span=t_span, dt=dt, numerical_jacobian=True, debug=True)
        # t, q, u, la_g, la_gamma = solver.solve()
        # solver = Moreau_sym(model, t_span=t_span, dt=dt, numerical_jacobian=True, debug=False)
        # t, q, u, la_g, la_gamma = solver.solve()
        # solver = Moreau(model, t_span, dt)
        # t, q, u, la_g, la_gamma = solver.solve()
        # solver = Generalized_alpha_1(model, t1, dt, numerical_jacobian=True)
        # t, q, u, la_g, la_gamma = solver.solve()
        solver = Scipy_ivp(model, t1, dt)
        t, q, u = solver.solve()

    # animate configurations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    scale = L
    ax.set_xlim3d(left=0, right=L)
    ax.set_ylim3d(bottom=-L/2, top=L/2)
    ax.set_zlim3d(bottom=-L/2, top=L/2)

    # # prepare data for animation
    # frames = len(t)
    # target_frames = 100
    # frac = int(frames / target_frames)
    # animation_time = 1
    # interval = animation_time * 1000 / target_frames

    # frames = target_frames
    # t = t[::frac]
    # q = q[::frac]

    frames = len(t)
    interval = 100
    
    x0, y0, z0 = q0.reshape((3, -1))
    center_line0, = ax.plot(x0, y0, z0, '-ok')

    x1, y1, z1 = q[-1].reshape((3, -1))
    center_line, = ax.plot(x1, y1, z1, '-ob')

    def update(t, q, center_line):
        if dim ==2:
            x, y = q.reshape((2, -1))
            center_line.set_data(x, y)
            center_line.set_3d_properties(np.zeros_like(x))
        elif dim == 3:
            x, y, z = q.reshape((3, -1))
            center_line.set_data(x, y)
            center_line.set_3d_properties(z)

        return center_line,

    def animate(i):
        update(t[i], q[i], center_line)

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
    plt.show()