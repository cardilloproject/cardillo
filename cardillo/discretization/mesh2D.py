import numpy as np
from scipy.sparse.linalg import spsolve
from cardillo.discretization.B_spline import B_spline_basis2D, q_to_Pw_2D, decompose_B_spline_surface, flat2D_vtk
from cardillo.discretization.lagrange import lagrange_basis2D
from cardillo.math.algebra import inverse2D, determinant2D, norm3, cross3
from cardillo.discretization.indexing import flat2D, split2D
from cardillo.discretization.mesh1D import Mesh1D
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

# Mesh for quadrilateral elements on rectangular domain
class Mesh2D():
    def __init__(self, knot_vector_objs, nqp_per_dim, derivative_order=1, 
                basis='B-spline', nq_n=2, vxi=None, elDOF=None):
        self.basis = basis
        self.derivative_order = derivative_order
        # number of elements
        if vxi is None:
            self.nel_per_dim = (knot_vector_objs[0].nel, knot_vector_objs[1].nel)
        else:
            self.nel_per_dim = (1, 1)
        self.nel_xi, self.nel_eta = self.nel_per_dim
        self.nel = self.nel_xi * self.nel_eta
        self.element_shape = (self.nel_xi, self.nel_eta)

        # knot vectors
        self.knot_vector_objs = knot_vector_objs
        self.knot_vectors = [kv.data for kv in knot_vector_objs]
        self.Xi, self.Eta = knot_vector_objs
        self.degrees = self.Xi.degree, self.Eta.degree
        self.degrees1 = np.array(self.degrees) + 1
        self.p, self.q = self.degrees

        # number of quadrature points
        if vxi is None:
            self.nqp_per_dim = nqp_per_dim
        else:
            self.nqp_per_dim = (1, 1)
        self.nqp_xi, self.nqp_eta = self.nqp_per_dim
        self.nqp = self.nqp_xi * self.nqp_eta

        # number of nodes influencing each element
        self.nn_el = (self.p + 1) * (self.q + 1)
        self.nq_n = nq_n  # number of degrees of freedom per node
        self.nq_el = self.nn_el * nq_n  # total number of generalized coordinates per element

        if self.basis == 'lagrange':
            # number of total nodes
            self.nn = (self.p * self.nel_xi + 1) * (self.p * self.nel_eta + 1)

            # Nodes per row
            self.nn_xi = self.p * self.nel_xi + 1
            self.nn_eta = self.q * self.nel_eta + 1

            # elDOF
            self.elDOF = np.zeros((self.nel, self.nq_n*self.nn_el), dtype=int)
            for el in range(self.nel):
                el_xi, el_eta = split2D(el, self.element_shape)
                for a in range(self.nn_el):
                    a_xi, a_eta = split2D(a, self.degrees1)
                    elDOF_x = (self.p * el_xi + a_xi + (el_eta * self.q + a_eta)
                               * (self.nn_xi))
                    for d in range(self.nq_n):
                        self.elDOF[el, a+self.nn_el*d] = elDOF_x + self.nn * d

            self.vtk_cell_type = 'VTK_LAGRANGE_QUADRILATERAL'

        elif basis == 'B-spline':
            # number of total nodes
            self.nn = (self.p + self.nel_xi) * (self.q + self.nel_eta)

            # nodes per row
            self.nn_xi = self.p + self.nel_xi
            self.nn_eta = self.q + self.nel_eta

            if elDOF is None:
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
            else: 
                self.elDOF = elDOF

            self.vtk_cell_type = 'VTK_BEZIER_QUADRILATERAL'

        # construct selection matrix ndDOF assining to each node its DOFs
        # qe[ndDOF[a]] is equivalent to q_e^a = C^a * q^e
        self.nodalDOF = np.zeros((self.nn_el, self.nq_n), dtype=int)
        for d in range(self.nq_n):
            self.nodalDOF[:, d] = np.arange(self.nn_el) + d * self.nn_el

        # transform quadrature points on element intervals
        if vxi is None:
            self.quadrature_points()
        else:
            self.evaluation_points(vxi)

        # evaluate element shape functions at quadrature points
        self.shape_functions()

        # edge degrees of freedom
        if vxi is None:
            self.edges()

    def basis2D(self, degrees, derivative_order, knot_vectors, knots):
        if self.basis == 'B-spline':
            return B_spline_basis2D(degrees, derivative_order, [kv.data for kv in knot_vectors],
                                    knots)
        elif self.basis == 'lagrange':
            return lagrange_basis2D(degrees, knots, derivative_order,
                                    knot_vectors)

    def evaluation_points(self, vxi):
        self.qp_xi = np.zeros((self.nel_xi, self.nqp_xi))
        self.qp_eta = np.zeros((self.nel_eta, self.nqp_eta))
        self.wp = np.zeros((self.nel, self.nqp))

        self.qp_xi, self.qp_eta = vxi[:, None]
        self.wp[0, 0] = 1

    def quadrature_points(self):
        self.qp_xi = np.zeros((self.nel_xi, self.nqp_xi))
        self.qp_eta = np.zeros((self.nel_eta, self.nqp_eta))
        self.wp = np.zeros((self.nel, self.nqp))
        
        for el in range(self.nel):
            el_xi, el_eta = split2D(el, self.nel_per_dim)
            
            Xi_element_interval = self.Xi.element_interval(el_xi)
            Eta_element_interval = self.Eta.element_interval(el_eta)

            self.qp_xi[el_xi], w_xi = gauss(self.nqp_xi, interval=Xi_element_interval)
            self.qp_eta[el_eta], w_eta = gauss(self.nqp_eta, interval=Eta_element_interval)
            
            for i in range(self.nqp):
                i_xi, i_eta = split2D(i, self.nqp_per_dim)
                self.wp[el, i] = w_xi[i_xi] * w_eta[i_eta]

    def shape_functions(self):
        self.N = np.zeros((self.nel, self.nqp, self.nn_el))
        if self.derivative_order > 0:
            self.N_xi = np.zeros((self.nel, self.nqp, self.nn_el, 2))
            if self.derivative_order > 1:
                self.N_xixi = np.zeros((self.nel, self.nqp, self.nn_el, 2, 2))
        
        for el in range(self.nel):
            el_xi, el_eta = split2D(el, self.nel_per_dim)

            NN = self.basis2D(self.degrees, self.derivative_order,
                         self.knot_vector_objs, (self.qp_xi[el_xi],
                                                 self.qp_eta[el_eta]))
            self.N[el] = NN[:, :, 0]
            if self.derivative_order > 0:
                self.N_xi[el] = NN[:, :, range(1, 3)]
                if self.derivative_order > 1:
                    self.N_xixi[el] = NN[:, :, range(3, 7)].reshape(self.nqp,
                                                                    self.nn_el,
                                                                     2, 2)

    def edges(self):
        def select_edge(**kwargs):
            nn_0 = kwargs.get('nn_0', range(self.nn_xi))
            nn_1 = kwargs.get('nn_1', range(self.nn_eta))

            edge = []
            for j in nn_1:
                for i in nn_0:
                    edge.append(flat2D(i, j, (self.nn_xi, )))

            DOF_x = np.array(edge)
            nn_edge = len(edge)
            DOF = np.zeros((self.nq_n, nn_edge), dtype=int)
            for i in range(self.nq_n):
                DOF[i] = DOF_x + i * self.nn
                
            return DOF

        self.edge_qDOF = (
            select_edge(nn_0=[0]),
            select_edge(nn_0=[self.nn_xi - 1]),
            select_edge(nn_1=[0]),
            select_edge(nn_1=[self.nn_eta - 1])
        )

        line01 = Mesh1D(self.knot_vector_objs[1], self.nqp_per_dim[1],
                        derivative_order=self.derivative_order,
                        basis=self.basis, nq_n=self.nq_n)

        line23 = Mesh1D(self.knot_vector_objs[0], self.nqp_per_dim[0],
                        derivative_order=self.derivative_order,
                        basis=self.basis, nq_n=self.nq_n)

        self.edge_mesh = (line01, line01, line23, line23)

    # TODO: handle derivatives, check usage of function
    def interpolate(self, knots, q, derivative_order=0):
        n = len(knots)
        x = np.zeros((n, 3))

        for i, (xi, eta) in enumerate(knots):
            el_xi = self.Xi.element_number(xi)[0]
            el_eta = self.Eta.element_number(eta)[0]
            el = flat2D(el_xi, el_eta, self.nel_per_dim)
            elDOF = self.elDOF[el]
            qe = q[elDOF]

            NN = self.basis2D(self.degrees, derivative_order,
                              self.knot_vectors_objs, (xi, eta))
            for a in range(self.nn_el):
                x[i] += NN[0, a, 0] * qe[self.nodalDOF[a]]
            
        return x

    def L2_projection_A(self, knots):
        A = Coo((self.nn, self.nn))
        for xi, eta in knots:
            el_xi = self.Xi.element_number(xi)[0]
            el_eta = self.Eta.element_number(eta)[0]
            el = flat2D(el_xi, el_eta, self.nel_per_dim)
            elDOF = self.elDOF[el, :self.nn_el]

            Ae = np.zeros((self.nn_el, self.nn_el))
            NN = self.basis2D(self.degrees, 0, self.knot_vectors_objs, (xi, eta))
            for i in range(self.nn_el):
                for j in range(self.nn_el):
                    Ae[i, j] = NN[0, i, 0] * NN[0, j, 0]
            A.extend(Ae, (elDOF, elDOF))
        return A

    def L2_projection_b(self, knots, Pw):
        b = np.zeros(self.nn)
        for (xi, eta), Pwi in zip(knots, Pw):
            el_xi = self.Xi.element_number(xi)[0]
            el_eta = self.Eta.element_number(eta)[0]
            el = flat2D(el_xi, el_eta, self.nel_per_dim)
            elDOF = self.elDOF[el, :self.nn_el]

            be = np.zeros((self.nn_el))
            NN = self.basis2D(self.degrees, 0, self.knot_vectors_objs, (xi, eta))
            for i in range(self.nn_el):
                be[i] = NN[0, i, 0] * Pwi
            b[elDOF] += be
        return b

    def reference_mappings(self, Q):
        """Compute inverse gradient from the reference configuration to the parameter space (only possible for planar elements) and scale quadrature points by its determinant. See Bonet 1997 (7.6a,b)
        """
        # planar elements
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
                        kappa0_xi += np.outer(Qe[self.nodalDOF[a]], N_xi[a])  # Bonet 1997 (7.6b)
                    
                    kappa0_xi_inv[el, i] = inverse2D(kappa0_xi)
                    w_J0[el, i] = determinant2D(kappa0_xi) * self.wp[el, i]

                    for a in range(self.nn_el):
                        N_X[el, i, a] = N_xi[a] @ kappa0_xi_inv[el, i]  # Bonet 1997 (7.6a) modified
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
                    w_J0[el, i] = (norm3(cross3(kappa0_xi[:, 0], kappa0_xi[:, 1]))
                                   * self.wp[el, i])
            
            return w_J0
                        
    def N_XX(self, Q, kappa0_xi_inv):
        assert self.derivative_order == 2
        N_XX = np.zeros((self.nel, self.nqp, self.nn_el, self.nq_n, self.nq_n))
        for el in range(self.nel):
            Qe = Q[self.elDOF[el]]

            for i in range(self.nqp):
                N_xi = self.N_xi[el, i]
                N_xixi = self.N_xixi[el, i]
                kappa0_xi_inv_el_i = kappa0_xi_inv[el, i]
                
                kappa0_xixi = np.zeros((self.nq_n, 2, 2))
                for a in range(self.nn_el):
                    kappa0_xixi += np.einsum('i,jk->ijk', Qe[self.nodalDOF[a]], N_xixi[a])

                for a in range(self.nn_el):
                    N_XX[el, i, a] = kappa0_xi_inv_el_i.T @ (N_xixi[a] - np.einsum('i,ijk->jk', N_xi[a] @ kappa0_xi_inv_el_i, kappa0_xixi)) @ kappa0_xi_inv_el_i # Maurin2019 (28) modified
        return N_XX

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
            # rearrange q's from solver to Piegl's 2D ordering
            Pw = q_to_Pw_2D(self.knot_vector_objs, q.reshape(-1, order='F'), dim=dim)

            # decompose B-spline mesh in Bezier patches
            Qw = decompose_B_spline_surface(self.knot_vector_objs, Pw)
            

        elif self.basis == 'lagrange':
            # rearrange q's from solver to Piegl's 3D ordering
            Qw = np.zeros((self.nel_xi, self.nel_eta, self.p+1,
                           self.q+1, dim))
            for el in range(self.nel):
                el_xi, el_eta = split2D(el, self.element_shape)
                for a in range(self.nn_el):
                    a_xi, a_eta = split2D(a, self.degrees1)
                    Qw[el_xi, el_eta, a_xi, a_eta] = \
                        q[self.elDOF[el][self.nodalDOF[a]]]

        nbezier_xi, nbezier_eta, p1, q1, dim = Qw.shape

        # rearrange Bezier mesh points for vtk ordering
        n_patches = nbezier_xi * nbezier_eta
        patch_size = p1 * q1
        point_data = np.zeros((n_patches * patch_size, dim))
        for i in range(nbezier_xi):
            for j in range(nbezier_eta):
                idx = flat2D(i, j, (nbezier_xi,))
                point_range = np.arange(idx * patch_size, (idx + 1)
                                        * patch_size)
                point_data[point_range] = flat2D_vtk(Qw[i, j])

        return point_data
        
    def vtk_mesh(self, q):
        if self.basis == 'B-spline':
            # rearrange q's from solver to Piegl's 2D ordering
            Pw = q_to_Pw_2D(self.knot_vector_objs, q, dim=self.nq_n)
            
            # decompose B-spline mesh in Bezier patches       
            Qw = decompose_B_spline_surface(self.knot_vector_objs, Pw)
        
        elif self.basis == 'lagrange':
            # rearrange q's from solver to Piegl's 3D ordering
            Qw = np.zeros((self.nel_xi, self.nel_eta, self.p+1,
                           self.q+1, self.nq_n))
            for el in range(self.nel):
                el_xi, el_eta = split2D(el, self.element_shape)
                for a in range(self.nn_el):
                    a_xi, a_eta  = split2D(a, self.degrees1)
                    Qw[el_xi, el_eta, a_xi, a_eta] = \
                        q[self.elDOF[el][self.nodalDOF[a][0]]]

        nbezier_xi, nbezier_eta, p1, q1, dim = Qw.shape
        
        # build vtk mesh
        n_patches = nbezier_xi * nbezier_eta
        patch_size = p1 * q1
        points = np.zeros((n_patches * patch_size, dim))
        cells = []
        HigherOrderDegrees = []
        for i in range(nbezier_xi):
            for j in range(nbezier_eta):
                idx = flat2D(i, j, (nbezier_xi,))
                point_range = np.arange(idx * patch_size, (idx + 1)
                                        * patch_size)
                points[point_range] = flat2D_vtk(Qw[i, j])
                
                cells.append( (self.vtk_cell_type, point_range[None]) )
                HigherOrderDegrees.append( np.array((*self.degrees, 0))[None] )

        return cells, points, HigherOrderDegrees

def test_edge_qDOF():
    from cardillo.discretization.B_spline import Knot_vector
    # degrees = (1, 2, 3)
    # element_shape = (3, 2, 1)
    degrees = (3, 2)
    QP_shape = (3, 4)
    element_shape = (4, 5)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    knot_vectors = (Xi, Eta)
    
    #from cardillo.discretization.mesh import Mesh, cube, scatter_Qs
    mesh = Mesh2D(knot_vectors, QP_shape, derivative_order=1, basis='B-spline', nq_n=2)

    rectangle_shape = (5, 3)
    Q = rectangle(rectangle_shape, mesh, Greville=True, Fuzz=0)
    # scatter_Qs(Q)

    import matplotlib.pyplot as plt
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    max_val = np.max(np.abs(Q))
    ax.set_xlim(left=-max_val, right=max_val)
    ax.set_ylim(bottom=-max_val, top=max_val)

    ax.scatter(*Q[mesh.edge_qDOF[2].reshape(-1)].reshape(2,-1), marker='x', color='green')
    ax.scatter(*Q[mesh.edge_qDOF[0].reshape(-1)].reshape(2,-1), marker='x', color='red')
    ax.scatter(*Q[mesh.edge_qDOF[1].reshape(-1)].reshape(2,-1), marker='x', color='red')
    ax.scatter(*Q[mesh.edge_qDOF[3].reshape(-1)].reshape(2,-1), marker='x', color='green')

    plt.show()

if __name__ == "__main__":
    test_edge_qDOF()