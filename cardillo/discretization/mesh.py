import numpy as np
from scipy.sparse.linalg import spsolve
# from cardillo_fem.discretization.lagrange import lagrange2D, lagrange1D, lagrange3D
from cardillo.discretization.B_spline import uniform_knot_vector, B_spline_basis1D, B_spline_basis2D, B_spline_basis3D, q_to_Pw_3D, decompose_B_spline_volume, flat3D_vtk
from cardillo.math.algebra import inverse2D, determinant2D, inverse3D, determinant3D, quat2rot
from cardillo.discretization.indexing import flat3D, split3D
from cardillo.discretization.gauss import gauss
from cardillo.utility.coo import Coo

def cube(shape, mesh, Greville=False, Fuzz=None):
    L, B, H = shape

    X = np.linspace(0, L, mesh.nn_xi)
    Y = np.linspace(0, B, mesh.nn_eta)
    Z = np.linspace(0, H, mesh.nn_zeta)
    if Greville:
        for i in range(len(X)):
            X[i] = np.sum(mesh.Xi.data[i+1:i+mesh.p+1])
        for i in range(len(Y)):
            Y[i] = np.sum(mesh.Eta.data[i+1:i+mesh.q+1])
        for i in range(len(Z)):
            Z[i] = np.sum(mesh.Zeta.data[i+1:i+mesh.r+1])
        X = X * L / mesh.p
        Y = Y * B / mesh.q
        Z = Z * H / mesh.r
        
    Xs = np.tile(X, mesh.nn_eta * mesh.nn_zeta)
    Ys = np.tile(np.repeat(Y, mesh.nn_xi), mesh.nn_zeta)
    Zs = np.repeat(Z, mesh.nn_eta * mesh.nn_xi)

    # manipulate reference coordinates with small random numbers
    # prevents singular stifness matrix
    if Fuzz is not None:
        Xs += np.random.rand(len(Xs)) * L * Fuzz
        Ys += np.random.rand(len(Ys)) * H * Fuzz
        Zs += np.random.rand(len(Ys)) * B * Fuzz

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

# Mesh for quadrilateral elements on rectangular domain
class Mesh():
    def __init__(self, knot_vector_objs, nqp_per_dim, derivative_order=1, basis='B-spline', nq_n=3):
        # number of elements
        self.nel_per_dim = (knot_vector_objs[0].nel, knot_vector_objs[1].nel, knot_vector_objs[2].nel)
        self.nel_xi, self.nel_eta, self.nel_zeta = self.nel_per_dim
        self.nel = self.nel_xi * self.nel_eta * self.nel_zeta
        self.element_shape = (self.nel_xi, self.nel_eta, self.nel_zeta)

        # knot vectors
        self.knot_vector_objs = knot_vector_objs
        self.knot_vectors = [kv.data for kv in knot_vector_objs]
        self.Xi, self.Eta, self.Zeta = knot_vector_objs
        self.degrees = self.Xi.degree, self.Eta.degree, self.Zeta.degree
        self.degrees1 = np.array(self.degrees) + 1
        self.p, self.q, self.r  = self.degrees

        self.derivative_order = derivative_order
        
        # number of quadrature points
        self.nqp_per_dim = nqp_per_dim
        self.nqp_xi, self.nqp_eta, self.nqp_zeta = nqp_per_dim
        self.nqp = self.nqp_xi * self.nqp_eta * self.nqp_zeta

        # number of nodes influencing each element
        self.nn_el = (self.p + 1) * (self.q + 1) * (self.r + 1)

        self.basis = basis # B-spline or Lagrange basis
        self.nq_n = nq_n # number of degrees of freedom per node
        self.nq_el = self.nn_el * nq_n # total number of generalized coordinates per element
        
        if basis == 'lagrange':
            raise NotImplementedError('...')
        
            # number of total nodes
            self.ntot = (self.p * self.nEl_xi +1) * (self.p * self.nEl_eta +1) * (self.r * self.nEl_zeta +1)
            #Nodes per row
            self.nN_xi = self.p * self.nEl_xi + 1
            self.nN_eta = self.q * self.nEl_eta + 1
            self.nN_nu = self.r * self.nEl_zeta + 1
            #elDOF
            self.elDOF = np.zeros((self.nEl, self.nq_n*self.n), dtype=int)
            for el in range(self.nEl):
                el_xi, el_eta, el_nu = self.split_el(el)
                for a in range(self.n):
                    a_xi, a_eta, a_nu = self.split_a(a)
                    elDOF_x = self.p * el_xi + a_xi + (el_eta*self.q + a_eta) * (self.nN_xi) + (el_nu*self.r + a_nu) * (self.nN_xi) * (self.nN_eta) 
                    for d in range(self.nq_n):
                        self.elDOF[el,a+self.n*d] = elDOF_x + self.ntot *d

            self.compute_all_N_3D = self.compute_all_N_3D_lagrange
            #self.compute_all_N_2D = self.compute_all_N_2D_lagrange
            #self.compute_all_N_1D = self.compute_all_N_1D_lagrange

        elif basis == 'B-spline':
            # number of total nodes
            self.nn = (self.p + self.nel_xi) * (self.q + self.nel_eta) * (self.r + self.nel_zeta)

            # nodes per row
            self.nn_xi = self.p + self.nel_xi
            self.nn_eta = self.q + self.nel_eta
            self.nn_zeta = self.r + self.nel_zeta

            # construct selection matrix elDOF assigning to each element its DOFs of the displacement
            # q[elDOF[el]] is equivalent to q_e = C^e * q
            self.elDOF = np.zeros((self.nel, self.nq_n * self.nn_el), dtype=int)
            for el in range(self.nel):
                el_xi, el_eta, el_zeta = split3D(el, self.element_shape)
                for a in range(self.nn_el):
                    a_xi, a_eta, a_zeta = split3D(a, self.degrees1)
                    elDOF_x = el_xi + a_xi + (el_eta + a_eta) * (self.nn_xi) + (el_zeta + a_zeta) * (self.nn_xi) * (self.nn_eta)
                    for d in range(self.nq_n):
                        self.elDOF[el, a + self.nn_el * d] = elDOF_x + self.nn * d

        self.nn_per_dim = (self.nn_xi, self.nn_eta, self.nn_zeta)

        # construct selection matrix ndDOF assining to each node its DOFs
        # qe[ndDOF[a]] is equivalent to q_e^a = C^a * q^e
        self.nodalDOF = np.zeros((self.nn_el, self.nq_n), dtype=int)
        for d in range(self.nq_n):
            self.nodalDOF[:, d] = np.arange(self.nn_el) + d * self.nn_el

        # transform quadrature poitns on element intervals
        self.quadrature_points()

        # evaluate element shape functions at quadrature points
        self.shape_functions()

        # constraint matrices 3D
        self.surface_DOF = {}

    def select_surface_2(self, **kwargs):

        nn_0 =  kwargs.get('nn_0',range(self.nn_xi))
        nn_1 =  kwargs.get('nn_1',range(self.nn_eta))
        nn_2 =  kwargs.get('nn_2',range(self.nn_zeta))

        surface = []

        for k in nn_2:
            for j in nn_1:
                for i in nn_0:

                    surface.append(flat3D(i,j,k,self.nn_per_dim))

        return np.array(surface)

       # select_surface_2(nn_2 = [0])

#        for s in range(3):

            
        #     surface_0 = np.zeros((3,self.nn_xi*self.nn_eta))


        #    # for k in np.array([0,self.nn_zeta]):
        #     for i in range(self.nn_xi):
        #         for j in range(self.nn_eta):

        #             surface_0[0,flat2D(i,j,self.nn_xi)] = flat3D(i,j,0,self.nn_per_dim)
        #             surface_1[0,flat2D(i,j,self.nn_xi)] = flat3D(i,j,self.nn_zeta,self.nn_per_dim)

        #     surface_0[1:,n] =


        # def select_surface_DOF(pos_blocked):
        #     temp = [0, 1, 2].pop(pos_blocked)

        #     nn_0, nn_1 = self.nn_per_dim[temp]
        #     for i in range(nn_0):
        #         for j in range(nn_1):
        #             for k in range(2):

        #                 idx = [i,j].insert(pos_blocked, k*self.nn_per_dim[pos_blocked])

        #                 surface[0,flat2D(i,j,nn_xi)] = flat3D(*idx,self.nn_per_dim)

        #     return surface

            

        # xy plane
        self.i_bottom_x = np.arange(0, self.nn_xi * self.nn_eta)
        self.i_bottom_y = self.i_bottom_x + self.nn
        self.i_bottom_z = self.i_bottom_y + self.nn
        self.i_top_x = np.arange(self.nn-self.nn_xi * self.nn_eta,self.nn)
        self.i_top_y = self.i_top_x + self.nn
        self.i_top_z = self.i_top_y + self.nn
        
        # yz plane 
        self.i_back_x = np.arange(0,self.nn,self.nn_xi)
        self.i_back_y = self.i_back_x + self.nn
        self.i_back_z = self.i_back_y + self.nn
        self.i_front_x = np.arange(self.nn_xi-1,self.nn,self.nn_xi)
        self.i_front_y = self.i_front_x + self.nn
        self.i_front_z = self.i_front_y + self.nn

    def evaluation_points(self):
        self.qp_xi = np.zeros((self.nel_xi, self.nqp_xi))
        self.qp_eta = np.zeros((self.nel_eta, self.nqp_eta))
        self.qp_zeta = np.zeros((self.nel_zeta, self.nqp_zeta))
        self.wp = np.zeros((self.nel, self.nqp))
                
        for el in range(self.nel):
            el_xi, el_eta, el_zeta = split3D(el, self.nel_per_dim)
            
            Xi_element_interval = self.Xi.element_interval(el_xi)
            Eta_element_interval = self.Eta.element_interval(el_eta)
            Zeta_element_interval = self.Zeta.element_interval(el_zeta)

            self.qp_xi[el_xi], w_xi = gauss(self.nqp_xi, interval=Xi_element_interval)
            self.qp_eta[el_eta], w_eta = gauss(self.nqp_eta, interval=Eta_element_interval)
            self.qp_zeta[el_zeta], w_zeta = gauss(self.nqp_zeta, interval=Zeta_element_interval)
            
            for i in range(self.nqp):
                i_xi, i_eta, i_zeta = split3D(i, self.nqp_per_dim)
                self.wp[el, i] = w_xi[i_xi] * w_eta[i_eta] * w_zeta[i_zeta]

    def quadrature_points(self):
        self.qp_xi = np.zeros((self.nel_xi, self.nqp_xi))
        self.qp_eta = np.zeros((self.nel_eta, self.nqp_eta))
        self.qp_zeta = np.zeros((self.nel_zeta, self.nqp_zeta))
        self.wp = np.zeros((self.nel, self.nqp))
                
        for el in range(self.nel):
            el_xi, el_eta, el_zeta = split3D(el, self.nel_per_dim)
            
            Xi_element_interval = self.Xi.element_interval(el_xi)
            Eta_element_interval = self.Eta.element_interval(el_eta)
            Zeta_element_interval = self.Zeta.element_interval(el_zeta)

            self.qp_xi[el_xi], w_xi = gauss(self.nqp_xi, interval=Xi_element_interval)
            self.qp_eta[el_eta], w_eta = gauss(self.nqp_eta, interval=Eta_element_interval)
            self.qp_zeta[el_zeta], w_zeta = gauss(self.nqp_zeta, interval=Zeta_element_interval)
            
            for i in range(self.nqp):
                i_xi, i_eta, i_zeta = split3D(i, self.nqp_per_dim)
                self.wp[el, i] = w_xi[i_xi] * w_eta[i_eta] * w_zeta[i_zeta]

    def shape_functions(self):
        self.N = np.zeros((self.nel, self.nqp, self.nn_el))
        if self.derivative_order > 0:
            self.N_xi = np.zeros((self.nel, self.nqp, self.nn_el, 3))
            if self.derivative_order > 1:
                self.N_xixi = np.zeros((self.nel, self.nqp, self.nn_el, 3, 3))
        
        for el in range(self.nel):
            el_xi, el_eta, el_zeta = split3D(el, self.nel_per_dim)

            NN = B_spline_basis3D(self.degrees, self.derivative_order, self.knot_vectors, (self.qp_xi[el_xi], self.qp_eta[el_eta], self.qp_zeta[el_zeta]))
            self.N[el] = NN[:, :, 0]
            if self.derivative_order > 0:
                self.N_xi[el] = NN[:, :, range(1, 4)]
                if self.derivative_order > 1:
                    self.N_xixi[el] = NN[:, :, range(4, 13)].reshape(self.nqp, self.nn_el, 3, 3)

    # TODO: handle derivatives
    def interpolate(self, knots, q, derivative_order=0):
        n = len(knots)
        x = np.zeros((n, 3))

        for i, (xi, eta, zeta) in enumerate(knots):
            el_xi = self.Xi.element_number(xi)[0]
            el_eta = self.Eta.element_number(eta)[0]
            el_zeta = self.Zeta.element_number(zeta)[0]
            el = flat3D(el_xi, el_eta, el_zeta, self.nel_per_dim)
            elDOF = self.elDOF[el]
            qe = q[elDOF]

            NN = B_spline_basis3D(self.degrees, derivative_order, self.knot_vectors, (xi, eta, zeta))
            for a in range(self.nn_el):
                x[i] += NN[0, a, 0] * qe[self.nodalDOF[a]]
            
        return x

    def L2_projection_A(self, knots):
        A = Coo((self.nn, self.nn))
        for xi, eta, zeta in knots:
            el_xi = self.Xi.element_number(xi)[0]
            el_eta = self.Eta.element_number(eta)[0]
            el_zeta = self.Zeta.element_number(zeta)[0]
            el = flat3D(el_xi, el_eta, el_zeta, self.nel_per_dim)
            elDOF = self.elDOF[el, :self.nn_el]

            Ae = np.zeros((self.nn_el, self.nn_el))
            NN = B_spline_basis3D(self.degrees, 0, self.knot_vectors, (xi, eta, zeta))
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
            NN = B_spline_basis3D(self.degrees, 0, self.knot_vectors, (xi, eta, zeta))
            for i in range(self.nn_el):
                be[i] = NN[0, i, 0] * Pwi
            b[elDOF] += be
        return b

    # TODO:
    def compute_all_N_3D_lagrange(self, PointsOnEdge=False):
        r"""Computes the shape functions and their derivatives for all Gauss points of all elements. Also returns weighting of the Gauss points.

        Returns
        -------
        N : numpy.ndarray
            (n_QP)-by-((p+1)*(q+1)) array holding for each Gauss point on each element the shape function values.
        dN_dvxi : numpy.ndarray
            (n_QP)-by-((p+1)*(q+1))-by-(2) array holding for each Gauss point on each element the derivative of the shape function wrt xi and eta.

        """
        # compute Gauss points
        if PointsOnEdge:
            gp_xi, wp_xi = np.linspace(start = -1, stop = 1, num = self.nqp_xi), np.ones(self.nqp_xi)
            gp_eta, wp_eta = np.linspace(start = -1, stop = 1, num = self.nqp_eta), np.ones(self.nqp_eta)
            gp_nu, wp_nu = np.linspace(start = -1, stop = 1, num = self.nqp_zeta), np.ones(self.nqp_zeta)
        else:
            gp_xi, wp_xi = np.polynomial.legendre.leggauss(self.nqp_xi)
            gp_eta, wp_eta = np.polynomial.legendre.leggauss(self.nqp_eta)
            gp_nu, wp_nu = np.polynomial.legendre.leggauss(self.nqp_zeta)

        N = np.zeros((self.nqp, self.nn_el))
        dN = np.zeros((self.nqp, self.nn_el, 3))

        for gi in range(self.nqp):
            gix, gie, ginu = self.split_gp(gi)
            N[gi], dN[gi] = lagrange3D(self.p, self.q, self.r, gp_xi[gix], gp_eta[gie], gp_nu[ginu])

        wp = np.zeros((self.nqp))
        for i in range(self.nqp):
            i_xi, i_eta, i_nu = self.split_gp(i)
            wp[i] = wp_xi[i_xi] * wp_eta[i_eta] * wp_nu[i_nu]

        return np.vstack([[N]]*self.nel), np.vstack([[dN]]*self.nel), None ,  np.vstack([gp_xi]*self.nel), np.vstack([gp_eta]*self.nel), np.vstack([gp_nu]*self.nel), np.vstack([wp]*self.nel)

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

        # rearrange q's from solver to Piegl's 3D ordering
        Pw = q_to_Pw_3D(self.knot_vector_objs, q.reshape(-1, order='F'), dim=self.nq_n * self.nq_n)

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