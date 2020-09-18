import numpy as np
from scipy.sparse.linalg import spsolve
# from cardillo_fem.discretization.lagrange import lagrange2D, lagrange1D, lagrange3D
#from cardillo.discretization.B_spline import uniform_knot_vector, B_spline_basis1D, B_spline_basis2D, B_spline_basis3D, q_to_Pw_3D, decompose_B_spline_volume, flat3D_vtk
from cardillo.discretization.lagrange import lagrange_basis3D, lagrange_basis2D
from cardillo.math.algebra import inverse2D, determinant2D, inverse3D, determinant3D, quat2rot, norm3, cross3
from cardillo.discretization.indexing import flat2D, split2D
#from cardillo.discretization.mesh2D_lagrange import Mesh2D_lagrange
from cardillo.discretization.gauss import gauss
from cardillo.utility.coo import Coo

def rectangle(shape, mesh, Greville=False, Fuzz=None):
    L, B = shape

    X = np.linspace(0, L, mesh.nn_xi)
    Y = np.linspace(0, B, mesh.nn_eta)
    if Greville:
        for i in range(len(X)):
            X[i] = np.sum(mesh.Xi.data[i+1:i+mesh.p+1])
        for i in range(len(Y)):
            Y[i] = np.sum(mesh.Eta.data[i+1:i+mesh.q+1])
        X = X * L / mesh.p
        Y = Y * B / mesh.q
        
    Xs = np.tile(X, mesh.nn_eta)
    Ys = np.repeat(Y, mesh.nn_xi)

    # manipulate reference coordinates with small random numbers
    # prevents singular stifness matrix
    if Fuzz is not None:
        Xs += np.random.rand(len(Xs)) * L * Fuzz
        Ys += np.random.rand(len(Ys)) * B * Fuzz

    # build generalized coordinates
    return np.concatenate((Xs, Ys))

def scatter_Qs(Q):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    max_val = np.max(np.abs(Q))
    ax.set_xlim(left=-0.1 * max_val, right=1.1 * max_val)
    ax.set_ylim(bottom=-0.1 * max_val, top=1.1 *max_val)
    ax.scatter(*Q.reshape(2, -1), marker='p')
    plt.show()

# Mesh for hexahedral lagrange elements on 2D domain
class Mesh2D_lagrange():
    def __init__(self, degrees, nqp_per_dim, nel_per_dim, derivative_order=1, nq_n=3):
        self.basis = 'lagrange'
        self.derivative_order = derivative_order

        # number of quadrature points
        self.nqp_per_dim = nqp_per_dim
        self.nqp_xi, self.nqp_eta = nqp_per_dim
        self.nqp = self.nqp_xi * self.nqp_eta
        
        self.nel_per_dim = nel_per_dim
        self.nel_xi, self.nel_eta = self.nel_per_dim
        self.nel = self.nel_xi * self.nel_eta
        self.element_shape = (self.nel_xi, self.nel_eta)

        # polynomial degree
        self.degrees = degrees
        self.degrees1 = np.array(self.degrees) + 1
        self.p, self.q = self.degrees

        # number of total nodes
        self.nn = (self.p * self.nel_xi + 1) * (self.p * self.nel_eta + 1)

        #nodes per row
        self.nn_xi = self.p * self.nel_xi + 1
        self.nn_eta = self.q * self.nel_eta + 1

        # number of nodes influencing each element
        self.nn_el = (self.p + 1) * (self.q + 1)

        # compute gauss poits and weights
        self.quadrature_points()

        # evaluate element shape functions at quadrature points
        self.shape_functions()
            
        # number of nodes per element per dimension
        self.nn_per_dim = (self.nn_xi, self.nn_eta)

        self.nq_n = nq_n # number of degrees of freedom per node
        self.nq_el = self.nn_el * nq_n # total number of generalized coordinates per element

        # construct selection matrix elDOF assigning to each element its DOFs of the displacement
        # q[elDOF[el]] is equivalent to q_e = C^e * q
        self.elDOF = np.zeros((self.nel, self.nq_n * self.nn_el), dtype=int)
        for el in range(self.nel):
            el_xi, el_eta = split2D(el, self.element_shape)
            for a in range(self.nn_el):
                a_xi, a_eta = split2D(a, self.degrees1)
                elDOF_x = el_xi + a_xi + (el_eta + a_eta) * (self.nn_xi)
                for d in range(self.nq_n):
                    self.elDOF[el, a + self.nn_el * d] = elDOF_x + self.nn * d

        # construct selection matrix ndDOF assining to each node its DOFs
        # qe[ndDOF[a]] is equivalent to q_e^a = C^a * q^e
        self.nodalDOF = np.zeros((self.nn_el, self.nq_n), dtype=int)
        for d in range(self.nq_n):
            self.nodalDOF[:, d] = np.arange(self.nn_el) + d * self.nn_el

        # stores evaluated shape functions for all 6 surfaces together with position degrees of freedom
        # self.surfaces()

    def quadrature_points(self):
        qp_xi, w_xi = gauss(self.nqp_xi)
        qp_eta, w_eta = gauss(self.nqp_eta)

        self.qp_xi = np.tile(qp_xi, (self.nel_xi,1))
        self.qp_eta = np.tile(qp_eta, (self.nel_eta,1))

        wp = np.zeros(self.nqp)
        for i in range(self.nqp):
            i_xi, i_eta = split2D(i, self.nqp_per_dim)
            wp[i] = w_xi[i_xi] * w_eta[i_eta]

        self.wp = np.tile(wp, (self.nel,1))

    def shape_functions(self):
        self.N = np.zeros((self.nel, self.nqp, self.nn_el))
        if self.derivative_order > 0:
            self.N_xi = np.zeros((self.nel, self.nqp, self.nn_el, 2))
            if self.derivative_order > 1:
                self.N_xixi = np.zeros((self.nel, self.nqp, self.nn_el, 2, 2))

        NN = lagrange_basis2D(self.degrees,(self.qp_xi[0], self.qp_eta[0]))
        self.N = np.vstack([[NN[:, :, 0]]] * self.nel)
        if self.derivative_order > 0:
            self.N_xi = np.vstack([[NN[:, :, range(1, 3)]]] * self.nel)
            if self.derivative_order > 1:
                self.N_xixi = np.vstack([[NN[:, :, range(4, 7)].reshape(self.nqp, self.nn_el, 3, 3)]] * self.nel)
    
    def edge_DOF(self):
        pass

    # TODO: handle derivatives
    def interpolate(self, knots, q, derivative_order=0):
        raise NotImplementedError('...')
        n = len(knots)
        x = np.zeros((n, 3))

        for i, (xi, eta, zeta) in enumerate(knots):
            el_xi = self.Xi.element_number(xi)[0]
            el_eta = self.Eta.element_number(eta)[0]
            el_zeta = self.Zeta.element_number(zeta)[0]
            el = flat2D(el_xi, el_eta, el_zeta, self.nel_per_dim)
            elDOF = self.elDOF[el]
            qe = q[elDOF]

            NN = lagrange_basis2D(self.degrees, (xi, eta), derivative=0)
            for a in range(self.nn_el):
                x[i] += NN[0, a, 0] * qe[self.nodalDOF[a]]
            
        return x

    def reference_mappings(self, Q):
        """Compute inverse gradient from the reference configuration to the parameter space and scale quadrature points by its determinant. See Bonet 1997 (7.6a,b)
        """
        if self.nq_n == 2:
            kappa0_xi_inv = np.zeros((self.nel, self.nqp, self.nq_n, self.nq_n))
            N_X = np.zeros((self.nel, self.nqp, self.nn_el, self.nq_n))
            w_J0 = np.zeros((self.nel, self.nqp))
            for el in range(self.nel):
                Qe = Q[self.elDOF[el]]

                for i in range(self.nqp):
                    N_xi = self.N_xi[el, i]

                    kappa0_xi = np.zeros((self.nq_n, 2))
                    for a in range(self.nn_el):
                        kappa0_xi += np.outer(Qe[self.nodalDOF[a]], N_xi[a]) # Bonet 1997 (7.6b)
                    
                    kappa0_xi_inv[el, i] = inverse2D(kappa0_xi)
                    w_J0[el, i] = determinant2D(kappa0_xi) * self.wp[el, i]

                    for a in range(self.nn_el):
                        N_X[el, i, a] = N_xi[a] @ kappa0_xi_inv[el, i] # Bonet 1997 (7.6a) modified
                        # N_X[el, i, a] = kappa0_xi_inv[el, i].T @ N_xi[a] # Bonet 1997 (7.6a)

            return kappa0_xi_inv, N_X, w_J0

        # surface Jacobian for embedded surfaces
        else:
            w_J0 = np.zeros((self.nel, self.nqp))
            for el in range(self.nel):
                Qe = Q[self.elDOF[el]]

                for i in range(self.nqp):
                    N_xi = self.N_xi[el, i]

                    kappa0_xi = np.zeros((self.nq_n, 2))
                    for a in range(self.nn_el):
                        kappa0_xi += np.outer(Qe[self.nodalDOF[a]], N_xi[a]) # Bonet 1997 (7.6b)

                    # Ciarlet2005 Theorem 2.3-1 (a) and Schulte2020 below (5)
                    w_J0[el, i] = norm3(cross3(kappa0_xi[:, 0], kappa0_xi[:, 1])) * self.wp[el, i]
            
            return w_J0

    # functions for vtk export
    def field_to_vtk(self, field):
        _, _, *shape = field.shape
        dim = np.prod(shape)

        # L2 projection on B-spline mesh
        self.ensure_L2_projection_A()
        b = self.rhs_L2_projection(field)
        q = np.zeros((self.nn, dim))
        for i, bi in enumerate(b.T):
            q[:, i] = spsolve(self.A, bi)

        # rearrange q's from solver to Piegl's 3D ordering
        Pw = q_to_Pw_3D(self.knot_vector_objs, q.reshape(-1, order='F'), dim=dim)

        # decompose B-spline mesh in Bezier patches      
        Qw = decompose_B_spline_volume(self.knot_vector_objs, Pw)
        nbezier_xi, nbezier_eta, nbezier_zeta, p1, q1, r1, dim = Qw.shape

        # rearrange Bezier mesh points for vtk ordering
        n_patches = nbezier_xi * nbezier_eta * nbezier_zeta
        patch_size = p1 * q1 * r1
        point_data = np.zeros((n_patches * patch_size, dim))
        for i in range(nbezier_xi):
            for j in range(nbezier_eta):
                for k in range(nbezier_zeta):
                    idx = flat3D(i, j, k, (nbezier_xi, nbezier_eta))
                    point_range = np.arange(idx * patch_size, (idx + 1) * patch_size)
                    point_data[point_range] = flat3D_vtk(Qw[i, j, k])

        return point_data
        
    def vtk_mesh(self, q):
        # rearrange q's from solver to Piegl's 3D ordering
        Pw = q_to_Pw_3D(self.knot_vector_objs, q, dim=self.nq_n)
        
        # decompose B-spline mesh in Bezier patches       
        Qw = decompose_B_spline_volume(self.knot_vector_objs, Pw)
        nbezier_xi, nbezier_eta, nbezier_zeta, p1, q1, r1, dim = Qw.shape
        
        # build vtk mesh
        n_patches = nbezier_xi * nbezier_eta * nbezier_zeta
        patch_size = p1 * q1 * r1
        points = np.zeros((n_patches * patch_size, dim))
        cells = []
        HigherOrderDegrees = []
        for i in range(nbezier_xi):
            for j in range(nbezier_eta):
                for k in range(nbezier_zeta):
                    idx = flat3D(i, j, k, (nbezier_xi, nbezier_eta))
                    point_range = np.arange(idx * patch_size, (idx + 1) * patch_size)
                    points[point_range] = flat3D_vtk(Qw[i, j, k])
                    
                    cells.append( ("VTK_BEZIER_HEXAHEDRON", point_range[None]) )
                    HigherOrderDegrees.append( np.array(self.degrees)[None] )

        return cells, points, HigherOrderDegrees


def test_lagrange_cube_vtk():
    degrees = (1, 1, 1)
    QP_shape = (3, 4, 2)
    element_shape = (1, 2, 3)

    mesh = Mesh3D(degrees, QP_shape, element_shape, derivative_order=1, basis='lagrange', nq_n=3)

    cube_shape = (3, 3, 3)
    Q = cube(cube_shape, mesh, Greville=False, Fuzz=0)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    max_val = np.max(np.abs(Q))
    ax.set_xlim3d(left=-max_val, right=max_val)
    ax.set_ylim3d(bottom=-max_val, top=max_val)
    ax.set_zlim3d(bottom=-max_val, top=max_val)
    ax.scatter(*Q.reshape(3, -1), marker='p')

    # ax.scatter(*Q[mesh.surface_qDOF[0].reshape(-1)].reshape(3,-1), marker='x', color='red')
    # ax.scatter(*Q[mesh.surface_qDOF[2].reshape(-1)].reshape(3,-1), marker='x', color='green')
    # ax.scatter(*Q[mesh.surface_qDOF[4].reshape(-1)].reshape(3,-1), marker='x', color='black')
    # ax.scatter(*Q[mesh.surface_qDOF[1].reshape(-1)].reshape(3,-1), marker='x', color='red')
    # ax.scatter(*Q[mesh.surface_qDOF[3].reshape(-1)].reshape(3,-1), marker='x', color='green')
    # ax.scatter(*Q[mesh.surface_qDOF[5].reshape(-1)].reshape(3,-1), marker='x', color='black')

    plt.show()

def test_surface_DOF():
    from cardillo.discretization.B_spline import Knot_vector
    # degrees = (1, 2, 3)
    # element_shape = (3, 2, 1)
    degrees = (3, 2, 1)
    QP_shape = (3, 4, 2)
    element_shape = (4, 5, 3)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)
    
    #from cardillo.discretization.mesh import Mesh, cube, scatter_Qs
    mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=2, basis='B-spline', nq_n=3)

    cube_shape = (3, 3, 3)
    Q = cube(cube_shape, mesh, Greville=True, Fuzz=0)
    #scatter_Qs(Q_cube)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    max_val = np.max(np.abs(Q))
    ax.set_xlim3d(left=-max_val, right=max_val)
    ax.set_ylim3d(bottom=-max_val, top=max_val)
    ax.set_zlim3d(bottom=-max_val, top=max_val)
#    ax.scatter(*Q.reshape(3, -1), marker='p')

    ax.scatter(*Q[mesh.surface_qDOF[0].reshape(-1)].reshape(3,-1), marker='x', color='red')
    ax.scatter(*Q[mesh.surface_qDOF[2].reshape(-1)].reshape(3,-1), marker='x', color='green')
    ax.scatter(*Q[mesh.surface_qDOF[4].reshape(-1)].reshape(3,-1), marker='x', color='black')
    ax.scatter(*Q[mesh.surface_qDOF[1].reshape(-1)].reshape(3,-1), marker='x', color='red')
    ax.scatter(*Q[mesh.surface_qDOF[3].reshape(-1)].reshape(3,-1), marker='x', color='green')
    ax.scatter(*Q[mesh.surface_qDOF[5].reshape(-1)].reshape(3,-1), marker='x', color='black')

    plt.show()

if __name__ == "__main__":
    #test_surface_DOF()
    test_lagrange_cube_vtk()