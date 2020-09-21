import numpy as np
from scipy.sparse.linalg import spsolve
# from cardillo_fem.discretization.lagrange import lagrange2D, lagrange1D, lagrange3D
from cardillo.discretization.B_spline import q_to_Pw_3D,  flat3D_vtk
from cardillo.discretization.lagrange import lagrange_basis3D, lagrange_basis2D
from cardillo.math.algebra import inverse2D, determinant2D, inverse3D, determinant3D, quat2rot
from cardillo.discretization.indexing import flat3D, split3D
from cardillo.discretization.mesh2D_lagrange import Mesh2D_lagrange
from cardillo.discretization.gauss import gauss
from cardillo.utility.coo import Coo

def cube(shape, mesh, Fuzz=None):
    L, B, H = shape

    X = np.linspace(0, L, mesh.nn_xi)
    Y = np.linspace(0, B, mesh.nn_eta)
    Z = np.linspace(0, H, mesh.nn_zeta)
        
    Xs = np.tile(X, mesh.nn_eta * mesh.nn_zeta)
    Ys = np.tile(np.repeat(Y, mesh.nn_xi), mesh.nn_zeta)
    Zs = np.repeat(Z, mesh.nn_eta * mesh.nn_xi)

    # manipulate reference coordinates with small random numbers
    # prevents singular stifness matrix
    if Fuzz is not None:
        Xs += np.random.rand(len(Xs)) * L * Fuzz
        Ys += np.random.rand(len(Ys)) * B * Fuzz
        Zs += np.random.rand(len(Ys)) * H * Fuzz

    # build generalized coordinates
    return np.concatenate((Xs, Ys, Zs))

def scatter_Qs(Q):
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
    plt.show()

# Mesh for hexahedral lagrange elements on 3D domain
class Mesh3D_lagrange():
    def __init__(self, degrees, nqp_per_dim, nel_per_dim, derivative_order=1, nq_n=3):
        self.basis = 'lagrange'
        self.derivative_order = derivative_order

        # number of quadrature points
        self.nqp_per_dim = nqp_per_dim
        self.nqp_xi, self.nqp_eta, self.nqp_zeta = nqp_per_dim
        self.nqp = self.nqp_xi * self.nqp_eta * self.nqp_zeta

        
        self.nel_per_dim = nel_per_dim
        self.nel_xi, self.nel_eta, self.nel_zeta = self.nel_per_dim
        self.nel = self.nel_xi * self.nel_eta * self.nel_zeta
        self.element_shape = (self.nel_xi, self.nel_eta, self.nel_zeta)

        # polynomial degree
        self.degrees = degrees
        self.degrees1 = np.array(self.degrees) + 1
        self.p, self.q, self.r = self.degrees

        # number of total nodes
        self.nn = (self.p * self.nel_xi + 1) * (self.p * self.nel_eta + 1) * (self.r * self.nel_zeta +1 )

        #nodes per row
        self.nn_xi = self.p * self.nel_xi + 1
        self.nn_eta = self.q * self.nel_eta + 1
        self.nn_zeta = self.r * self.nel_zeta + 1

        # number of nodes influencing each element
        self.nn_el = (self.p + 1) * (self.q + 1) * (self.r + 1)

        # compute gauss poits and weights
        self.quadrature_points()

        # evaluate element shape functions at quadrature points
        self.shape_functions()

        # number of nodes per element per dimension
        self.nn_per_dim = (self.nn_xi, self.nn_eta, self.nn_zeta)

        self.nq_n = nq_n # number of degrees of freedom per node
        self.nq_el = self.nn_el * nq_n # total number of generalized coordinates per element

        # construct selection matrix elDOF assigning to each element its DOFs of the displacement
        # q[elDOF[el]] is equivalent to q_e = C^e * q
        self.elDOF = np.zeros((self.nel, self.nq_n * self.nn_el), dtype=int)
        for el in range(self.nel):
            el_xi, el_eta, el_zeta = split3D(el, self.element_shape)
            for a in range(self.nn_el):
                a_xi, a_eta, a_zeta = split3D(a, self.degrees1)
                elDOF_x = el_xi * self.p + a_xi + (el_eta * self.q + a_eta) * (self.nn_xi) + (el_zeta * self.r + a_zeta) * (self.nn_xi) * (self.nn_eta)
                for d in range(self.nq_n):
                    self.elDOF[el, a + self.nn_el * d] = elDOF_x + self.nn * d

        # construct selection matrix ndDOF assining to each node its DOFs
        # qe[ndDOF[a]] is equivalent to q_e^a = C^a * q^e
        self.nodalDOF = np.zeros((self.nn_el, self.nq_n), dtype=int)
        for d in range(self.nq_n):
            self.nodalDOF[:, d] = np.arange(self.nn_el) + d * self.nn_el

        # stores evaluated shape functions for all 6 surfaces together with position degrees of freedom
        self.surfaces()

    def quadrature_points(self):
        qp_xi, w_xi = gauss(self.nqp_xi)
        qp_eta, w_eta = gauss(self.nqp_eta)
        qp_zeta, w_zeta = gauss(self.nqp_zeta)

        self.qp_xi = np.tile(qp_xi, (self.nel_xi,1))
        self.qp_eta = np.tile(qp_eta, (self.nel_eta,1))
        self.qp_zeta = np.tile(qp_zeta, (self.nel_zeta,1))

        wp = np.zeros(self.nqp)
        for i in range(self.nqp):
            i_xi, i_eta, i_zeta = split3D(i, self.nqp_per_dim)
            wp[i] = w_xi[i_xi] * w_eta[i_eta] * w_zeta[i_zeta]

        self.wp = np.tile(wp, (self.nel,1))

    def shape_functions(self):
        self.N = np.zeros((self.nel, self.nqp, self.nn_el))
        if self.derivative_order > 0:
            self.N_xi = np.zeros((self.nel, self.nqp, self.nn_el, 3))
            if self.derivative_order > 1:
                self.N_xixi = np.zeros((self.nel, self.nqp, self.nn_el, 3, 3))

        NN = lagrange_basis3D(self.degrees,(self.qp_xi[0], self.qp_eta[0], self.qp_zeta[0]))
        self.N = np.vstack([[NN[:, :, 0]]] * self.nel)
        if self.derivative_order > 0:
            self.N_xi = np.vstack([[NN[:, :, range(1, 4)]]] * self.nel)
            if self.derivative_order > 1:
                self.N_xixi = np.vstack([[NN[:, :, range(4, 13)].reshape(self.nqp, self.nn_el, 3, 3)]] * self.nel)

    def surfaces(self):
        def select_surface(**kwargs):
            nn_0 =  kwargs.get('nn_0', range(self.nn_xi))
            nn_1 =  kwargs.get('nn_1', range(self.nn_eta))
            nn_2 =  kwargs.get('nn_2', range(self.nn_zeta))

            surface = []
            for k in nn_2:
                for j in nn_1:
                    for i in nn_0:
                        surface.append(flat3D(i, j, k, self.nn_per_dim))

            DOF_x = np.array(surface)
            nn_surface = len(surface)
            DOF = np.zeros((self.nq_n, nn_surface), dtype=int)
            for i in range(self.nq_n):
                DOF[i] = DOF_x + i * self.nn
                
            return DOF

        self.surface_qDOF = (
            select_surface(nn_0=[0]),
            select_surface(nn_0=[self.nn_xi - 1]),
            select_surface(nn_1=[0]),
            select_surface(nn_1=[self.nn_eta - 1]),
            select_surface(nn_2=[0]), 
            select_surface(nn_2=[self.nn_zeta - 1]), 
        )

        surface01 = Mesh2D_lagrange(self.degrees[1:3], self.nqp_per_dim[1:3], nel_per_dim=self.nel_per_dim[1:3], derivative_order=self.derivative_order, nq_n=self.nq_n)

        surface23 = Mesh2D_lagrange(self.degrees[::2], self.nqp_per_dim[::2], nel_per_dim=self.nel_per_dim[::2], derivative_order=self.derivative_order, nq_n=self.nq_n)
        
        surface45 = Mesh2D_lagrange(self.degrees[:2], self.nqp_per_dim[:2], nel_per_dim=self.nel_per_dim[:2], derivative_order=self.derivative_order, nq_n=self.nq_n)

        self.surface_mesh = (surface01, surface01, surface23, surface23, surface45, surface45)
        for i in range(6):
            self.surface_mesh[i].idx = i

    def L2_projection_A(self, knots):
        A = Coo((self.nn, self.nn))
        for xi, eta, zeta in knots:
            el_xi = self.Xi.element_number(xi)[0]
            el_eta = self.Eta.element_number(eta)[0]
            el_zeta = self.Zeta.element_number(zeta)[0]
            el = flat3D(el_xi, el_eta, el_zeta, self.nel_per_dim)
            elDOF = self.elDOF[el, :self.nn_el]

            Ae = np.zeros((self.nn_el, self.nn_el))
            NN = lagrange_basis3D(self.degrees, 0, self.knot_vectors, (xi, eta, zeta))
            for i in range(self.nn_el):
                for j in range(self.nn_el):
                    Ae[i, j] = NN[0, i, 0] * NN[0, j, 0]
            A.extend(Ae, (elDOF, elDOF))
        return A

    def L2_projection_b(self, knots, Pw):
        b = np.zeros(self.nn)
        for (xi, eta, zeta), Pwi in zip(knots, Pw):
            el_xi = self.Xi.element_number(xi)[0]
            el_eta = self.Eta.element_number(eta)[0]
            el_zeta = self.Zeta.element_number(zeta)[0]
            el = flat3D(el_xi, el_eta, el_zeta, self.nel_per_dim)
            elDOF = self.elDOF[el, :self.nn_el]

            be = np.zeros((self.nn_el))
            NN = lagrange_basis3D(self.degrees, 0, self.knot_vectors, (xi, eta, zeta))
            for i in range(self.nn_el):
                be[i] = NN[0, i, 0] * Pwi
            b[elDOF] += be
        return b

    # TODO: handle derivatives
    def compute_F_nodes(self, Q, q, derivative_order=0):
        # calculate F values at nodes from reference configuration Q and state q
        F_nodes = np.zeros((self.nel, self.nn_el,  self.nq_n , self.nq_n))
        xi = np.linspace(-1,1,self.p+1)
        eta = np.linspace(-1,1,self.q+1)
        zeta = np.linspace(-1,1,self.r+1)
        xis = (xi, eta, zeta)
        NN = lagrange_basis3D(self.degrees, xis)
        for el in range(self.nel):
            q_el = q[self.elDOF[el]]
            Q_el = Q[self.elDOF[el]]
            for i in range(self.nn_el):
                N_xi = NN[:,:,range(1,4)]
                Fhat0 = np.zeros((self.nq_n, self.nq_n))
                for a in range(self.nn_el):
                    Fhat0 += np.eye(3) @ np.outer(Q_el[self.nodalDOF[a]], N_xi[a])
                Fhat0_inv[el, i] = inverse3D(Fhat0)

                for a in range(self.nn_el):
                    N_X[el,i,a] = N_xi[a] @ Fhat0_inv[el, i] 
                    F_nodes[el,i] += np.outer(q_el[self.nodalDOF[a]], N_X[el,i,a])

        return F_nodes

    def reference_mappings(self, Q):
        """Compute inverse gradient from the reference configuration to the parameter space and scale quadrature points by its determinant. See Bonet 1997 (7.6a,b)
        """
        kappa0_xi_inv = np.zeros((self.nel, self.nqp, self.nq_n, self.nq_n))
        N_X = np.zeros((self.nel, self.nqp, self.nn_el, self.nq_n))
        w_J0 = np.zeros((self.nel, self.nqp))
        for el in range(self.nel):
            Qe = Q[self.elDOF[el]]

            for i in range(self.nqp):
                N_xi = self.N_xi[el, i]

                kappa0_xi = np.zeros((self.nq_n, self.nq_n))
                for a in range(self.nn_el):
                    kappa0_xi += np.outer(Qe[self.nodalDOF[a]], N_xi[a]) # Bonet 1997 (7.6b)
                
                kappa0_xi_inv[el, i] = inverse3D(kappa0_xi)
                w_J0[el, i] = determinant3D(kappa0_xi) * self.wp[el, i]

                for a in range(self.nn_el):
                    N_X[el, i, a] = N_xi[a] @ kappa0_xi_inv[el, i] # Bonet 1997 (7.6a) modified
                    # N_X[el, i, a] = kappa0_xi_inv[el, i].T @ N_xi[a] # Bonet 1997 (7.6a)

        return kappa0_xi_inv, N_X, w_J0

    def ensure_L2_projection_A(self):
        if not hasattr(self, "A"):
            A = Coo((self.nn, self.nn))
            for el in range(self.nel):
                elDOF_el = self.elDOF[el, :self.nn_el]
                Ae = np.zeros((self.nn_el, self.nn_el))
                Nel = self.N[el]
                for a in range(self.nn_el):
                    for b in range(self.nn_el):
                        for i in range(self.nqp):
                            Ae[a, b] += Nel[i, a] * Nel[i, b]
                A.extend(Ae, (elDOF_el, elDOF_el))
            self.A = A.tocsc()

    def rhs_L2_projection(self, field):
        _, nqp, *shape = field.shape
        dim = np.prod(shape)

        b = np.zeros((self.nn, dim))
        for el in range(self.nel):
            elDOF_el = self.elDOF[el, :self.nn_el]
            be = np.zeros((self.nn_el, dim))
            Nel = self.N[el]
            for a in range(self.nn_el):
                for i in range(self.nqp):
                    be[a] += Nel[i, a] * field[el, i].ravel()
            b[elDOF_el] += be
        return b

    # functions for vtk export
    def field_to_vtk(self, field):
        _, _, *shape = field.shape
        dim = np.prod(shape)

        # L2 projection on lagrange mesh
        self.ensure_L2_projection_A()
        b = self.rhs_L2_projection(field)
        q = np.zeros((self.nn, dim))
        for i, bi in enumerate(b.T):
            q[:, i] = spsolve(self.A, bi)

        # rearrange q's from solver to Piegl's 3D ordering
        Qw = np.zeros((self.nel_xi, self.nel_eta, self.nel_zeta, self.p+1, self.q+1, self.r+1, dim))
        for el in range(self.nel):
            el_xi, el_eta, el_zeta = split3D(el, self.element_shape)
            for a in range(self.nn_el):
                a_xi, a_eta, a_zeta = split3D(a, self.degrees1)
                Qw[el_xi, el_eta, el_zeta, a_xi, a_eta, a_zeta] = q[self.elDOF[el][self.nodalDOF[a][0]]]

        # rearrange mesh points for vtk ordering
        point_data = np.zeros((self.nel * self.nn_el, dim))
        for i in range(self.nel_xi):
            for j in range(self.nel_eta):
                for k in range(self.nel_zeta):
                    idx = flat3D(i, j, k, (self.nel_xi, self.nel_eta))
                    point_range = np.arange(idx * self.nn_el, (idx + 1) * self.nn_el)
                    point_data[point_range] = flat3D_vtk(Qw[i, j, k])

        return point_data
        
    def vtk_mesh(self, q): 

        # rearrange q's from solver to Piegl's 3D ordering
        Qw = np.zeros((self.nel_xi, self.nel_eta, self.nel_zeta, self.p+1, self.q+1, self.r+1, self.nq_n))
        for el in range(self.nel):
            el_xi, el_eta, el_zeta = split3D(el, self.element_shape)
            for a in range(self.nn_el):
                a_xi, a_eta, a_zeta = split3D(a, self.degrees1)
                Qw[el_xi, el_eta, el_zeta, a_xi, a_eta, a_zeta] = q[self.elDOF[el][self.nodalDOF[a]]]
        
        # build vtk mesh
        points = np.zeros((self.nel * self.nn_el, self.nq_n))
        cells = []
        HigherOrderDegrees = []
        for i in range(self.nel_xi):
            for j in range(self.nel_eta):
                for k in range(self.nel_zeta):
                    idx = flat3D(i, j, k, (self.nel_xi, self.nel_eta))
                    point_range = np.arange(idx * self.nn_el, (idx + 1) * self.nn_el)
                    points[point_range] = flat3D_vtk(Qw[i, j, k])
                    cells.append( ("VTK_LAGRANGE_HEXAHEDRON", point_range[None]) )
                    HigherOrderDegrees.append( np.array(self.degrees)[None] )

        return cells, points, HigherOrderDegrees

def test_surface_DOF():
    degrees = (2, 2, 2)
    QP_shape = (3, 4, 2)
    element_shape = (3, 4, 5)

    mesh = Mesh3D_lagrange(degrees, QP_shape, element_shape, derivative_order=1, nq_n=3)

    cube_shape = (3, 3, 3)
    Q = cube(cube_shape, mesh, Fuzz=0)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
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

    ax.scatter(*Q[mesh.surface_qDOF[0].reshape(-1)].reshape(3,-1), marker='x', color='red')
    ax.scatter(*Q[mesh.surface_qDOF[2].reshape(-1)].reshape(3,-1), marker='x', color='green')
    ax.scatter(*Q[mesh.surface_qDOF[4].reshape(-1)].reshape(3,-1), marker='x', color='black')
    ax.scatter(*Q[mesh.surface_qDOF[1].reshape(-1)].reshape(3,-1), marker='x', color='red')
    ax.scatter(*Q[mesh.surface_qDOF[3].reshape(-1)].reshape(3,-1), marker='x', color='green')
    ax.scatter(*Q[mesh.surface_qDOF[5].reshape(-1)].reshape(3,-1), marker='x', color='black')

    plt.show()

if __name__ == "__main__":
    test_surface_DOF()