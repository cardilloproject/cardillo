import numpy as np
from scipy.sparse.linalg import spsolve
# from cardillo_fem.discretization.lagrange import lagrange2D, lagrange1D, lagrange3D
from cardillo.discretization.B_spline import flat1D_vtk
from cardillo.discretization.lagrange import lagrange_basis1D
#from cardillo.discretization.mesh2D_lagrange import Mesh2D_lagrange
from cardillo.discretization.gauss import gauss
from cardillo.utility.coo import Coo

# Mesh for lagrange elements on 1D domain
class Mesh1D_lagrange():
    def __init__(self, degree, nqp, nel, derivative_order=1, nq_n=3):
        self.basis = 'lagrange'
        self.derivative_order = derivative_order

        # number of quadrature points
        self.nqp = nqp
        self.nqp_xi = self.nqp
        
        self.nel = nel
        self.nel_xi = self.nel
        self.element_shape = (self.nel_xi, )

        # polynomial degree
        self.degree = degree
        self.degree1 = self.degree + 1
        self.p = self.degree

        # number of total nodes
        self.nn = self.p * self.nel_xi + 1

        #nodes per row
        self.nn_xi = self.p * self.nel_xi + 1

        # number of nodes influencing each element
        self.nn_el = self.p + 1

        # compute gauss poits and weights
        self.quadrature_points()

        # evaluate element shape functions at quadrature points
        self.shape_functions()
            
        # number of nodes per element per dimension
        self.nn_per_dim = (self.nn_xi, )

        self.nq_n = nq_n # number of degrees of freedom per node
        self.nq_el = self.nn_el * nq_n # total number of generalized coordinates per element

        self.end_points()

        # construct selection matrix elDOF assigning to each element its DOFs of the displacement
        # q[elDOF[el]] is equivalent to q_e = C^e * q
        self.elDOF = np.zeros((self.nel, self.nq_n * self.nn_el), dtype=int)
        for el in range(self.nel):
            for a in range(self.nn_el):
                elDOF_x = el * self.p + a
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

        self.qp = np.tile(qp_xi, (self.nel_xi,1))
        self.wp = np.tile(w_xi, (self.nel,1))

    def shape_functions(self):
        self.N = np.zeros((self.nel, self.nqp, self.nn_el))
        if self.derivative_order > 0:
            self.N_xi = np.zeros((self.nel, self.nqp, self.nn_el))
            if self.derivative_order > 1:
                self.N_xixi = np.zeros((self.nel, self.nqp, self.nn_el))

        NN = lagrange_basis1D(self.degree, self.qp[0])
        self.N = np.vstack([[NN[:, :, 0]]] * self.nel)
        if self.derivative_order > 0:
            self.N_xi = np.vstack([[NN[:, :, 1]]] * self.nel)
            if self.derivative_order > 1:
                self.N_xixi = np.vstack([[NN[:, :, 2].reshape(self.nqp, self.nn_el)]] * self.nel)
    

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

        self.end_points_DOF = (
            select_end_points(nn_0=[0]),
            select_end_points(nn_0=[self.nn - 1])
        )

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
        Qw = np.zeros((self.nel_xi, self.p+1, dim))
        for el in range(self.nel):
            for a in range(self.nn_el):
                Qw[el,  a] = q[self.elDOF[el][self.nodalDOF[a][0]]]

        # rearrange mesh points for vtk ordering
        point_data = np.zeros((self.nel * self.nn_el, dim))
        for i in range(self.nel_xi):
            for j in range(self.nel_eta):
                point_range = np.arange(i * self.nn_el, (i + 1) * self.nn_el)
                point_data[point_range] = flat1D_vtk(Qw[i])

        return point_data
        
    def vtk_mesh(self, q): 
        # rearrange q's from solver to Piegl's 1D ordering
        Qw = np.zeros((self.nel,  self.p+1, self.nq_n))
        for el in range(self.nel):
            for a in range(self.nn_el):
                Qw[el,  a] = q[self.elDOF[el][self.nodalDOF[a]]]
        
        # build vtk mesh
        points = np.zeros((self.nel * self.nn_el, self.nq_n))
        cells = []
        HigherOrderDegrees = []
        for i in range(self.nel):
            point_range = np.arange(i * self.nn_el, (i + 1) * self.nn_el)
            points[point_range] = flat1D_vtk(Qw[i])
            cells.append( ("VTK_LAGRANGE_CURVE", point_range[None]) )
            HigherOrderDegrees.append( np.array(self.degrees)[None] )

        return cells, points, HigherOrderDegrees

if __name__ == "__main__":
    pass