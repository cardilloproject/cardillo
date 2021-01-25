import numpy as np
from scipy.sparse.linalg import spsolve
from cardillo.discretization.B_spline import B_spline_basis1D, decompose_B_spline_curve, flat1D_vtk
from cardillo.discretization.lagrange import lagrange_basis1D
from cardillo.discretization.gauss import gauss
from cardillo.utility.coo import Coo

def line2D(L, mesh, Greville=False, Fuzz=None):
    X = np.linspace(0, L, mesh.nn)
    Y = np.zeros(mesh.nn)
    if Greville:
        for i in range(len(X)):
            X[i] = np.sum(mesh.Xi.data[i+1:i+mesh.p+1])
        X = X * L / mesh.p

    # manipulate reference coordinates with small random numbers
    # prevents singular stifness matrix
    if Fuzz is not None:
        X += np.random.rand(len(X)) * L * Fuzz
        Y += np.random.rand(len(Y)) * Fuzz

    # build generalized coordinates
    return np.concatenate((X, Y))

class Mesh1D():
    def __init__(self, knot_vector, nqp, derivative_order=1, basis='B-spline',
                 nq_n=2):
        self.basis = basis
        # number of elements
        self.nel = knot_vector.nel

        # knot vectors
        self.knot_vector = knot_vector
        self.Xi = self.knot_vector.data
        self.degree = self.knot_vector.degree
        self.degrees1 = self.degree + 1
        self.p = self.degree

        self.derivative_order = derivative_order
        
        # number of quadrature points
        self.nqp = nqp

        # number of nodes influencing each element
        self.nn_el = self.p + 1
        self.nq_n = nq_n  # number of degrees of freedom per node
        self.nq_el = self.nn_el * nq_n  # total number of generalized coordinates per element
        
        if basis == 'lagrange':
            self.nn = self.p * self.nel + 1

            # elDOF
            self.elDOF = np.zeros((self.nel, self.nq_n*self.nn_el), dtype=int)
            for el in range(self.nel):
                for a in range(self.nn_el):
                    elDOF_x = self.p * el + a
                    for d in range(self.nq_n):
                        self.elDOF[el, a+self.nn_el*d] = elDOF_x + self.nn * d

            self.vtk_cell_type = 'VTK_LAGRANGE_CURVE'

        elif basis == 'B-spline':
            # number of total nodes
            self.nn = self.p + self.nel

            # construct selection matrix elDOF assigning to each element its DOFs of the displacement
            # q[elDOF[el]] is equivalent to q_e = C^e * q
            self.elDOF = np.zeros((self.nel, self.nq_n * self.nn_el), dtype=int)
            for el in range(self.nel):
                for a in range(self.nn_el):
                    elDOF_x = el + a
                    for d in range(self.nq_n):
                        self.elDOF[el, a+self.nn_el*d] = elDOF_x + self.nn * d

            self.vtk_cell_type = 'VTK_BEZIER_CURVE'

        # self.nn_per_dim = (self.nn_xi, self.nn_eta)

        # construct selection matrix ndDOF assining to each node its DOFs
        # qe[ndDOF[a]] is equivalent to q_e^a = C^a * q^e
        self.nodalDOF = np.zeros((self.nn_el, self.nq_n), dtype=int)
        for d in range(self.nq_n):
            self.nodalDOF[:, d] = np.arange(self.nn_el) + d * self.nn_el

        # transform quadrature points on element intervals
        self.quadrature_points()

        # evaluate element shape functions at quadrature points
        self.shape_functions()

        # end_points degrees of freedom
        self.end_points_DOF = self.end_points()

    def basis1D(self, degree, derivative_order, knot_vector, knots):
        if self.basis == 'B-spline':
            NN_ = B_spline_basis1D(degree, derivative_order,
                                    knot_vector.data, knots, squeeze=False)
            # rearrange dimensions to match order in Mesh2D and lagrange
            return np.einsum('kij->ijk', NN_) 
        elif self.basis == 'lagrange':
            return lagrange_basis1D(degree, knots, derivative_order,
                                    knot_vector)

    def quadrature_points(self):
        self.qp = np.zeros((self.nel, self.nqp))
        self.wp = np.zeros((self.nel, self.nqp))
                
        for el in range(self.nel):            
            Xi_element_interval = self.knot_vector.element_interval(el)
            self.qp[el], self.wp[el] = gauss(self.nqp,
                                             interval=Xi_element_interval)

    def shape_functions(self):
        self.N = np.zeros((self.nel, self.nqp, self.nn_el))
        if self.derivative_order > 0:
            self.N_xi = np.zeros((self.nel, self.nqp, self.nn_el))
            if self.derivative_order > 1:
                self.N_xixi = np.zeros((self.nel, self.nqp, self.nn_el))
        
        for el in range(self.nel):
            NN = self.basis1D(self.degree, self.derivative_order,
                              self.knot_vector, self.qp[el])
            self.N[el] = NN[:, :, 0]
            if self.derivative_order > 0:
                self.N_xi[el] = NN[:, :, 1]
                if self.derivative_order > 1:
                    self.N_xixi[el] = NN[:, :, 2]

    def end_points(self):
        def select_end_points(**kwargs):
            nn_0 = kwargs.get('nn_0', range(self.nn))

            end_points = []
            for i in nn_0:
                end_points.append(i)

            DOF_x = np.array(end_points)
            nn_edge = len(end_points)
            DOF = np.zeros((self.nq_n, nn_edge), dtype=int)
            for i in range(self.nq_n):
                DOF[i] = DOF_x + i * self.nn
                
            return DOF

        self.point_qDOF = (
            select_end_points(nn_0=[0]),
            select_end_points(nn_0=[self.nn - 1])
        )
 
    def reference_mappings(self, Q):
        """Compute inverse gradient from the reference configuration to the parameter space and scale quadrature points by its determinant. See Bonet 1997 (7.6a,b)
        """
        if self.nq_n == 3:
            w_J0 = np.zeros((self.nel, self.nqp))
            for el in range(self.nel):
                Qe = Q[self.elDOF[el]]

                for i in range(self.nqp):
                    N_xi = self.N_xi[el, i]

                    kappa0_xi = np.zeros((self.nq_n, self.nq_n))
                    for a in range(self.nn_el):
                        kappa0_xi += Qe[self.nodalDOF[a]] * N_xi[a] # Bonet 1997 (7.6b)

                    w_J0[el, i] = np.linalg.norm(kappa0_xi)

            return w_J0        

    # functions for vtk export
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
                for i in range(nqp):
                    be[a] += Nel[i, a] * field[el, i].reshape(-1)
            b[elDOF_el] += be
        return b

    def field_to_vtk(self, field):
        _, _, *shape = field.shape
        dim = np.prod(shape)

        # L2 projection on B-spline mesh
        self.ensure_L2_projection_A()
        b = self.rhs_L2_projection(field)
        q = np.zeros((self.nn, dim))
        for i, bi in enumerate(b.T):
            q[:, i] = spsolve(self.A, bi)

        if self.basis == 'B-spline':
            # rearrange q's from solver to Piegl's 1D ordering
            Pw = q.reshape(-1, dim, order='F')

            # decompose B-spline mesh in Bezier patches
            Qw = decompose_B_spline_curve(self.knot_vector, Pw)

        elif self.basis == 'lagrange':
            # rearrange q's from solver to Piegl's 3D ordering
            Qw = np.zeros((self.nel_xi, self.p+1, dim))
            for el in range(self.nel):
                for a in range(self.nn_el):
                    Qw[el, a] = q[self.elDOF[el][self.nodalDOF[a]]]

        nbezier_xi, p1, dim = Qw.shape

        # rearrange Bezier mesh points for vtk ordering
        n_patches = nbezier_xi
        patch_size = p1
        point_data = np.zeros((n_patches * patch_size, dim))
        for i in range(nbezier_xi):
            point_range = np.arange(i * patch_size, (i + 1) * patch_size)
            point_data[point_range] = flat1D_vtk(Qw[i])

        return point_data

    def vtk_mesh(self, q):
        if self.basis == 'B-spline':
            # rearrange q's from solver to Piegl's 1D ordering
            Pw = q.reshape(-1, self.nq_n, order='F')

            # decompose B-spline mesh in Bezier patches
            Qw = decompose_B_spline_curve(self.knot_vector, Pw)

        elif self.basis == 'lagrange':
            # rearrange q's from solver to Piegl's 3D ordering
            Qw = np.zeros((self.nel_xi, self.p+1, self.nq_n))
            for el in range(self.nel):
                for a in range(self.nn_el):
                    Qw[el, a] = q[self.elDOF[el][self.nodalDOF[a]]]

        nbezier_xi, p1, dim = Qw.shape

        # build vtk mesh
        n_patches = nbezier_xi
        patch_size = p1
        points = np.zeros((n_patches * patch_size, dim))
        cells = []
        HigherOrderDegrees = []
        for i in range(nbezier_xi):
            point_range = np.arange(i * patch_size, (i + 1) * patch_size)
            points[point_range] = flat1D_vtk(Qw[i])

            cells.append((self.vtk_cell_type, point_range[None]))
            HigherOrderDegrees.append(np.array((self.degree, 0, 0))[None])

        return cells, points, HigherOrderDegrees

def test_point_qDOF():
    from cardillo.discretization.B_spline import Knot_vector
    polynomial_degree = 3
    quadrature_points = polynomial_degree + 1
    elements = 4

    Xi = Knot_vector(polynomial_degree, elements)
    
    mesh = Mesh1D(Xi, quadrature_points, derivative_order=1, basis='B-spline', nq_n=2)

    Q = line2D(5, mesh, Greville=True, Fuzz=0)

    import matplotlib.pyplot as plt
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    max_val = np.max(np.abs(Q))
    ax.set_xlim(left=-max_val, right=max_val)
    ax.set_ylim(bottom=-max_val, top=max_val)

    ax.scatter(*Q[mesh.point_qDOF[0].reshape(-1)].reshape(2,-1), marker='x', color='green')
    ax.scatter(*Q[mesh.point_qDOF[1].reshape(-1)].reshape(2,-1), marker='x', color='red')

    plt.show()

if __name__ == "__main__":
    test_point_qDOF()