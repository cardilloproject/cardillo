import numpy as np
from scipy.sparse.linalg import spsolve
from cardillo.discretization.B_spline import uniform_knot_vector, B_spline_basis1D, decompose_B_spline_curve, flat1D_vtk
# from cardillo.math.algebra import inverse2D, determinant2D, inverse3D, determinant3D, quat2rot, norm3, cross3
# from cardillo.discretization.indexing import flat2D, split2D
from cardillo.discretization.gauss import gauss
from cardillo.utility.coo import Coo

class Mesh1D():
    def __init__(self, knot_vector, nqp, derivative_order=1, basis='B-spline', nq_n=2):
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

        self.basis = basis # B-spline or Lagrange basis
        self.nq_n = nq_n # number of degrees of freedom per node
        self.nq_el = self.nn_el * nq_n # total number of generalized coordinates per element
        
        if basis == 'lagrange':
            raise NotImplementedError('...')

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
                        self.elDOF[el, a + self.nn_el * d] = elDOF_x + self.nn * d

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

    def quadrature_points(self):
        self.qp = np.zeros((self.nel, self.nqp))
        self.wp = np.zeros((self.nel, self.nqp))
                
        for el in range(self.nel):            
            Xi_element_interval = self.knot_vector.element_interval(el)
            self.qp[el], self.wp[el] = gauss(self.nqp, interval=Xi_element_interval)

    def shape_functions(self):
        self.N = np.zeros((self.nel, self.nqp, self.nn_el))
        if self.derivative_order > 0:
            self.N_xi = np.zeros((self.nel, self.nqp, self.nn_el))
            if self.derivative_order > 1:
                self.N_xixi = np.zeros((self.nel, self.nqp, self.nn_el))
        
        for el in range(self.nel):
            NN = B_spline_basis1D(self.degree, self.derivative_order, self.Xi, self.qp[el], squeeze=False)
            self.N[el] = NN[:, :, 0]
            if self.derivative_order > 0:
                self.N_xi[el] = NN[:, :, 1]
                if self.derivative_order > 1:
                    self.N_xixi[el] = NN[:, :, 2]

    def end_points(self):
        def select_end_points(**kwargs):
            nn_0 =  kwargs.get('nn_0', range(self.nn))

            end_points = []
            for i in nn_0:
                end_points.append(i)

            DOF_x = np.array(end_points)
            nn_edge = len(end_points)
            DOF = np.zeros((self.nq_n, nn_edge), dtype=int)
            for i in range(self.nq_n):
                DOF[i] = DOF_x + i * self.nn
                
            return DOF

        DOF_tup = (
            select_end_points(nn_0=[0]),
            select_end_points(nn_0=[self.nn - 1])
        )

        return DOF_tup

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

        # rearrange q's from solver to Piegl's 1D ordering
        Pw = q.reshape(-1, dim, order='F')

        # decompose B-spline mesh in Bezier patches      
        Qw = decompose_B_spline_curve(self.knot_vector, Pw)
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
        # rearrange q's from solver to Piegl's 1D ordering
        Pw = q.reshape(-1, self.nq_n, order='F')

        # decompose B-spline mesh in Bezier patches 
        Qw = decompose_B_spline_curve(self.knot_vector, Pw)
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
            
            cells.append( ("VTK_BEZIER_CURVE", point_range[None]) )
            HigherOrderDegrees.append( np.array((self.degree, 0, 0))[None] )

        return cells, points, HigherOrderDegrees

if __name__ == "__main__":
    pass