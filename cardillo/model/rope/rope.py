import numpy as np

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

        self.qp, qw = gauss(nQP) # quadrature points and quadrature weights for Gauss-Legendre quadrature rule

        # TODO: find good name for element intervals?
        self.xi_span = np.linspace(0, 1, nEl + 1)
        if B_splines:
            self.nNd = nNd = nEl + p # number of nodes
            self.knot_vector = knot_vector = uniform_knot_vector(p, nEl) # uniform open knot vector
        else:
            self.nNd = nNd = nEl * p + 1 # number of nodes

        # TODO: good name?
        self.nNd_el = nNd_el = p + 1 # number of nodes given in a single element
        self.nNdDOF = nNdDOF = dim # number of degrees of freedom of a single node

        self.nq = nNdDOF * nNd # total number of generalized coordinates
        self.nu = self.nq
        self.nq_el = nNdDOF * nNd_el # total number of generalized coordinates in a single element

        # compute allocation matrices
        tmp1 = np.tile(np.arange(0, nNd_el), nNdDOF)
        tmp2 = np.repeat(np.arange(0, nNdDOF * nNd, step=nNd), nNd_el)
        self.elDOF = (np.zeros((nNdDOF * nNd_el, nEl), dtype=int) + np.arange(nEl)).T + tmp1 + tmp2

        # TODO: do we need nodal degrees of freedom?
        # tmp3 = (np.zeros((self.nNDOF, nNd), dtype=int) + np.arange(nNd)).T
        # tmp4 = np.tile(np.arange(0, nNDOF * nNd, step=nNd), nNd).reshape((nNd, nNDOF))
        # self.nodalDOF = tmp3 + tmp4

        # compute shape functions
        derivative_order = 1
        self.N  = np.empty((nEl, nQP, nNd_el))
        self.dN = np.empty((nEl, nQP, nNd_el))
        self.qw = np.zeros((nEl, nQP))
        for el in range(nEl):
            delta_xi = self.xi_span[el + 1] - self.xi_span[el]
            if B_splines:
                # transform quadrature points for B-spline shape functions
                xi = self.xi_span[el] + delta_xi * (self.qp + 1) / 2
                # print(f'xi: {xi}')

                # transform quadrature weights for B-spline shape functions
                self.qw[el] = qw * delta_xi / 2
                # print(f'qw: {self.qw[el]}')

                # ordering: (number of evaluation points, derivative order, different nonzero shape functions)
                N_dN = B_spline_basis(polynomial_degree, 1, knot_vector, xi)
                self.N[el] = N_dN[:, 0]
                self.dN[el] = N_dN[:, 1]
            else:
                raise NotImplementedError('not implemented')
                xi = self.qp
                # self.shape_functions[el] = lagrange_basis(knot_vector, p, 1, xi)

        # shape functions on the boundary
        if B_splines:
            self.N_bdry = B_spline_basis(polynomial_degree, 0, knot_vector, np.array([0, 1.0 - 1.0e-9])).squeeze()
        else:
            raise NotImplementedError('not implemented')
            
        # reference generalized coordinates, initial coordinates and initial velocities
        X0 = np.linspace(0, L, nNd)
        Y0 = np.zeros_like(X0)
        Z0 = np.zeros_like(X0)
        self.Q = np.hstack((X0, Y0, Z0)) if Q is None else Q
        self.q0 = np.hstack((X0, Y0, Z0)) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0
    
    def M_el(self, qe, Qe, N, dN, qw):
        Me = np.zeros((self.nq_el, self.nq_el))

        for Ni, dNi, qwi in zip(N, dN, qw):
            # build matrix of shape functions and derivatives
            NN = np.kron(np.eye(self.dim), Ni)
            dNNi = np.kron(np.eye(self.dim), dNi)

            # reference tangential vector
            dr0 = dNNi @ Qe
            G = norm2(dr0)
            
            # integrate elemente mass matrix
            Me += NN.T @ NN * self.A_rho0 * G * qwi

        return Me

    def M_dense(self, t, q):
        M = np.zeros((self.nu, self.nu))

        for el in range(self.nEl):
            # extract element degrees of freedom
            elDOF = self.elDOF[el, :]

            # store element mass matrix at the correct indices in the global mass matrix
            M[elDOF[:, None], elDOF] += self.M_el(q[elDOF], self.Q[elDOF], self.N[el], self.dN[el], self.qw[el])
        
        return M

    # TODO: compute constant mass matrix within an assembler callback
    def M(self, t, q, coo):
        for el in range(self.nEl):
            # extract element degrees of freedom
            elDOF = self.elDOF[el, :]

            # compute element mass matrix
            Me = self.M_el(q[elDOF], self.Q[elDOF], self.N[el], self.dN[el], self.qw[el])
            
            # sparse assemble element mass matrix
            coo.extend(Me, (self.uDOF[elDOF], self.uDOF[elDOF]))
            
    def f_pot_el(self, qe, Qe, dN, qw):
        fe = np.zeros(self.nq_el)

        for dNi, qwi in zip(dN, qw):
            # build matrix of shape function derivatives
            dNNi = np.kron(np.eye(self.dim), dNi)

            # tangential vectors
            dr  = dNNi @ qe 
            dr0 = dNNi @ Qe
            g = norm2(dr)
            G = norm2(dr0)
            
            # Calculate the strain and stress
            strain = g / G
            stress = self.material_model.n(strain) * dr / g

            # integrate element force vector
            # fe -= (dNNi.T / G) @ stress * G * self.wp[i]
            fe -= dNNi.T @ stress * qwi

        return fe
    
    def f_pot(self, t, q):
        f =  np.zeros(self.nu)

        for el in range(self.nEl):
            # extract element degrees of freedom
            elDOF = self.elDOF[el, :]

            # assemble internal element potential forces
            f[elDOF] += self.f_pot_el(q[elDOF], self.Q[elDOF], self.dN[el], self.qw[el])
                    
        return f

    def f_pot_q_el(self, qe, Qe, dN, qw):
        # fe_q_num = Numerical_derivative(lambda t, qe: self.f_pot_el(qe, Qe, dN, qw), order=2)._x(0, qe, eps=1.0e-6)
        
        fe_q = np.zeros((self.nq_el, self.nq_el))

        for dNi, qwi in zip(dN, qw):
            # build matrix of shape function derivatives
            dNNi = np.kron(np.eye(self.dim), dNi)
            
            # compute current and reference tangent vector w.r.t. [-1, 1]
            dr  = dNNi @ qe 
            dr0 = dNNi @ Qe
            g = norm2(dr)
            G = norm2(dr0)
            
            # Calculate the strain and stress
            strain = g / G
            n = self.material_model.n(strain)
            dn = self.material_model.dn(strain)
            dstress = dNNi / g * n + np.outer(dr, dr) @ dNNi / g**2 * (dn / G - n / g)

            # Calcualte element stiffness matrix
            fe_q -= dNNi.T @ dstress * qwi

        # # np.set_printoptions(3)
        # diff = fe_q_num - fe_q
        # # # print(diff)
        # # print(f'fe_q_num =\n{fe_q_num}')
        # # print(f'fe_q =\n{fe_q}')
        # error = np.linalg.norm(diff)
        # print(f'error in stiffness matrix: {error:.4e}')
        # return fe_q_num

        return fe_q

    def f_pot_q(self, t, q, coo):
        for el in range(self.nEl):
            # extract element degrees of freedom
            elDOF = self.elDOF[el, :]

            # integrate element internal stiffness matrix
            Ke = self.f_pot_q_el(q[elDOF], self.Q[elDOF], self.dN[el], self.qw[el])

            # sparse assemble element internal stiffness matrix
            coo.extend(Ke, (self.uDOF[elDOF], self.qDOF[elDOF]))

    # def M(self, t, q, coo):
    #     for el in range(self.nEl):
    #         # extract element degrees of freedom
    #         elDOF = self.elDOF[el, :]

    #         # compute element mass matrix
    #         Me = self.M_el(q[elDOF], self.Q[elDOF], self.N[el], self.dN[el], self.qw[el])
            
    #         # sparse assemble element mass matrix
    #         coo.extend(Me, (self.uDOF[elDOF], self.uDOF[elDOF]))
    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def local_elDOF(self, frame_ID):
        xi = frame_ID[0]
        if xi == 0:
            return self.elDOF[0]
        elif xi == 1:
            return self.elDOF[-1]
        else:
            print('get_elDOF can only be computed at xi = 0 and 1')

    def qDOF_P(self, frame_ID=(0,)):
        elDOF = self.local_elDOF(frame_ID)
        return self.qDOF[elDOF]

    def uDOF_P(self, frame_ID=(0,)):
        elDOF = self.local_elDOF(frame_ID)
        return self.uDOF[elDOF]

    def r_OP(self, t, q, frame_ID=(0,), K_r_SP=None):
        xi = frame_ID[0]
        if xi == 0:
            # elDOF = self.elDOF[0]
            N = self.N_bdry[0]
        elif xi == 1:
            # elDOF = self.elDOF[-1]
            N = self.N_bdry[1]
        else:
            print('r_OP can only be computed at frame_ID = (0,) or (1,)')

        # interpolate position vector
        return np.kron(np.eye(self.dim), N) @ q

    def r_OP_q(self, t, q, frame_ID=(0,), K_r_SP=None):
        xi = frame_ID[0]
        if xi == 0:
            N = self.N_bdry[0]
        elif xi == 1:
            N = self.N_bdry[1]
        else:
            print('r_OP_q can only be computed at frame_ID = (0,) or (1,)')

        # interpolate position vector
        return np.kron(np.eye(self.dim), N)

    def J_P(self, t, q, frame_ID=(0,), K_r_SP=None):
        return np.eye(3, self.nq_el)

    def J_P_q(self, t, q, frame_ID=(0,), K_r_SP=None):
        return np.zeros((3, self.nq_el, self.nq_el))

    def v_P(self, t, q, u, frame_ID=(0,), K_r_SP=None):
        # return self.r_OP(t, u, frame_ID=frame_ID)

        xi = frame_ID[0]
        if xi == 0:
            N = self.N_bdry[0]
        elif xi == 1:
            N = self.N_bdry[1]
        else:
            print('r_OP can only be computed at frame_ID = (0,) or (1,)')

        # interpolate position vector
        return np.kron(np.eye(self.dim), N) @ u

    def v_P_q(self, t, q, u, frame_ID=(0,), K_r_SP=None):
        # return self.r_OP_q(t, u, frame_ID=frame_ID)

        xi = frame_ID[0]
        if xi == 0:
            N = self.N_bdry[0]
        elif xi == 1:
            N = self.N_bdry[1]
        else:
            print('r_OP can only be computed at frame_ID = (0,) or (1,)')

        # interpolate position vector
        return np.kron(np.eye(self.dim), N)

if __name__ == "__main__":
    from cardillo.model.rope.hooke import Hooke
    from cardillo.model.frame import Frame
    from cardillo.model.bilateral_constraints import Spherical_joint
    from cardillo.model import Model
    from cardillo.solver import Euler_backward, Moreau, Moreau_sym

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation

    # physical properties of the rope
    dim = 3
    L = 2 * np.pi
    EA = 1000
    material_model = Hooke(EA)
    A_rho0 = 1

    # discretization properties
    B_splines = True
    p = 2
    nQP = int(np.ceil((p + 1)**2 / 2))
    print(f'nQP: {nQP}')
    nEl = 10

    # build reference configuration
    if B_splines:
        nNd = nEl + p
    else:
        nNd = nEl * p + 1
    X0 = np.linspace(0, L, nNd)
    Y0 = np.zeros_like(X0)
    if dim == 2:
        Q = np.hstack((X0, Y0))
    elif dim == 3:
        Z0 = np.zeros_like(X0)
        Q = np.hstack((X0, Y0, Z0))
    u0 = np.zeros_like(Q)
    q0 = np.copy(Q) + np.random.rand(3 * nNd) * 1.0e-1

    rope = Rope(A_rho0, material_model, p, nEl, nQP, Q, q0, u0, B_splines=B_splines, dim=dim)

    # q = q0.copy() + np.random.rand(len(q0)) * 0.1

    # np.set_printoptions(precision=1)

    # M = rope.M_dense(0, q)
    # print(f'M:\n{M}')

    # f_pot = rope.f_pot(0, q)
    # print(f'f_pot:\n{f_pot}')

    # f_pot_q = rope.f_pot_q(0, q)
    # print(f'f_pot_q:\n{f_pot_q}')

    r_OB1 = np.zeros(3)
    frame_left = Frame(r_OP=r_OB1)
    joint_left = Spherical_joint(frame_left, rope, r_OB1, frame_ID1=(0,))
    # omega = 2 * np.pi
    # A = 1
    # r_OB1 = lambda t: np.array([0, A * np.sin(omega * t), 0])
    # r_OB1_t = lambda t: np.array([0, A * omega * np.cos(omega * t), 0])
    # r_OB1_tt = lambda t: np.array([0, -A * omega**2 * np.sin(omega * t), 0])
    # frame_left = Frame(r_OP=r_OB1, r_OP_t=r_OB1_t, r_OP_tt=r_OB1_tt)
    # joint_left = Spherical_joint(frame_left, rope, r_OB1(0), frame_ID1=(0,))

    r_OB2 = np.array([L, 0, 0])
    frame_right = Frame(r_OP=r_OB2)
    joint_right = Spherical_joint(frame_right, rope, r_OB2, frame_ID1=(1,))

    model = Model()
    model.add(rope)
    model.add(frame_left)
    model.add(joint_left)
    model.add(frame_right)
    model.add(joint_right)
    model.assemble()

    # model.q0 += np.random.rand(model.nq) * 1.0e-1

    t0 = 0
    t1 = 2
    dt = 1e-2
    t_span = t0, t1
    solver = Euler_backward(model, t_span=t_span, dt=dt, numerical_jacobian=False, debug=False)
    t, q, u, la_g, la_gamma = solver.solve()
    # solver = Euler_backward(model, t_span=t_span, dt=dt, numerical_jacobian=True, debug=True)
    # t, q, u, la_g, la_gamma = solver.solve()
    # solver = Moreau_sym(model, t_span=t_span, dt=dt, numerical_jacobian=True, debug=False)
    # t, q, u, la_g, la_gamma = solver.solve()
    # solver = Moreau(model, t_span, dt)
    # t, q, u, la_g, la_gamma = solver.solve()

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

    # prepare data for animation
    frames = len(t)
    target_frames = 100
    frac = int(frames / target_frames)
    animation_time = 1
    interval = animation_time * 1000 / target_frames

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    q0 = q[0]
    x0, y0, z0 = q0.reshape((3, -1))
    center_line, = ax.plot(x0, y0, z0, '-ok')
    # plt.show()
    # exit()

    def update(t, q, center_line):
        x, y, z = q.reshape((3, -1))
        center_line.set_data(x, y)
        center_line.set_3d_properties(z)

        return center_line,

    def animate(i):
        update(t[i], q[i], center_line)

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
    plt.show()