import numpy as np
# from cardillo_fem.discretization.lagrange import lagrange2D, lagrange1D, lagrange3D
from cardillo.discretization.B_spline import uniform_knot_vector, B_spline_basis1D, B_spline_basis2D, B_spline_basis3D
from cardillo.math.algebra import inverse2D, determinant2D, inverse3D, determinant3D
from cardillo.discretization.indexing import split3D
from cardillo.discretization.gauss import gauss

# def __unpack_tuple(t):
#     if len(t) == 2:
#         return 

class Cube_mesh():
    pass

# Mesh for quadrilateral elements on rectangular domain
class Mesh():
    def __init__(self, knot_vectors, nqp_per_dim, derivative_order=1, basis='B-spline', nq_n=3):
        # number of elements
        self.nel_per_dim = (knot_vectors[0].nel, knot_vectors[1].nel, knot_vectors[2].nel)
        self.nel_xi, self.nel_eta, self.nel_zeta = self.nel_per_dim
        self.nel = self.nel_xi * self.nel_eta * self.nel_zeta

        # knot vectors
        self.knot_vectors = [kv.data for kv in knot_vectors]
        self.Xi, self.Eta, self.Zeta = knot_vectors
        self.degrees = self.Xi.degree, self.Eta.degree, self.Zeta.degree
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
                el_xi, el_eta, el_zeta = split3D(el, element_shape)
                for a in range(self.nn_el):
                    a_xi, a_eta, a_zeta = split3D(a, degrees)
                    elDOF_x = el_xi + a_xi + (el_eta + a_eta) * (self.nn_xi) + (el_zeta + a_zeta) * (self.nn_xi) * (self.nn_eta)
                    for d in range(self.nq_n):
                        self.elDOF[el, a + self.nn_el * d] = elDOF_x + self.nn * d

        # construct selection matrix ndDOF assining to each node its DOFs
        # qe[ndDOF[a]] is equivalent to q_e^a = C^a * q^e
        self.nodalDOF = np.zeros((self.nn_el, self.nq_n))
        for d in range(self.nq_n):
            self.nodalDOF[:, d] = np.arange(self.nn_el) + d * self.nn_el

        # transform quadrature poitns on element intervals
        self.quadrature_points()

        # evaluate element shape functions at quadrature points
        self.shape_functions()
        # self.N, self.dN_dvxi, self.d2N_dvxi2, self.xi_qp, self.eta_qp, self.zeta_qp, self.w_qp = self.compute_all_N_3D()
        
        #Compute 1 element shape functions at Gauss points for one element
        #Nxi_tmp, dNxi_tmp, xi_gp_tmp, self.wpxi = self.compute_all_N_1D(self.nel_xi, self.nQP_xi, self.p)
        #Neta_tmp, dNeta_tmp, eta_gp_tmp, self.wp_eta = self.compute_all_N_1D(self.nel_eta, self.nQP_eta, self.q)
    
        # constraint matrices 3D
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

        # nN_xi = self.p * self.nel_xi + 1
        # self.i_bottom_x = np.arange(nN_xi)
        # self.i_right_x = np.arange(nN_xi-1,self.ntot,nN_xi)
        # self.i_top_x = np.arange(self.ntot-nN_xi,self.ntot,1)
        # self.i_left_x = np.arange(0,self.ntot,nN_xi)
        # tmp = np.array([self.i_top_x, self.i_bottom_x, self.i_left_x, self.i_right_x])
        # self.i_top_x, self.i_bottom_x, self.i_left_x, self.i_right_x = tmp
        # self.i_top_y, self.i_bottom_y, self.i_left_y, self.i_right_y = tmp + self.ntot
        # self.bottom_corners_y = np.array([0, nN_xi-1]) + self.ntot

        # Selection matrices for 1D shape functions
        #self.ndDOF_xi = np.array([np.arange(self.p+1), np.arange(self.p+1) + self.p+1]).T
        #self.ndDOF_eta = np.array([np.arange(self.q+1), np.arange(self.q+1) + self.q+1]).T

    def quadrature_points(self):
        self.qp_xi = np.zeros((self.nel_xi, self.nqp_xi))
        self.qp_eta = np.zeros((self.nel_eta, self.nqp_eta))
        self.qp_zeta = np.zeros((self.nel_zeta, self.nqp_zeta))
        self.wp = np.zeros((self.nel, self.nqp))
                
        for el in range(self.nel):
            el_xi, el_eta, el_zeta = split3D(el, self.nel_per_dim)
            
            xi_element_interval = self.Xi.element_interval(el_xi)
            eta_element_interval = self.Eta.element_interval(el_eta)
            zeta_element_interval = self.Zeta.element_interval(el_zeta)

            self.qp_xi[el_xi], w_xi = gauss(self.nqp_xi, interval=xi_element_interval)
            self.qp_eta[el_eta], w_eta = gauss(self.nqp_eta, interval=eta_element_interval)
            self.qp_zeta[el_zeta], w_zeta = gauss(self.nqp_zeta, interval=zeta_element_interval)
            
            for i in range(self.nqp):
                i_xi, i_eta, i_zeta = split3D(i, self.nqp_per_dim)
                self.wp[el, i] = w_xi[i_xi] * w_eta[i_eta] * w_zeta[i_zeta]

    # TODO: profile me!
    def shape_functions(self):
        self.NN = []
        # self.N = np.zeros((self.nel, self.nqp, self.nn_el))
        # if self.derivative_order > 0:
        #     self.N_xi = np.zeros((self.nel, self.nqp, self.nn_el, 3))
        #     if self.derivative_order > 1:
        #         self.N_xixi = np.zeros((self.nel, self.nqp, self.nn_el, 3, 3))
        
        for el in range(self.nel):
            el_xi, el_eta, el_zeta = split3D(el, self.nel_per_dim)

            self.NN.append( B_spline_basis3D(self.degrees, self.derivative_order, self.knot_vectors, (self.qp_xi[el_xi], self.qp_eta[el_eta], self.qp_zeta[el_zeta])) )

            # NN = B_spline_basis3D(self.degrees, self.derivative_order, self.knot_vectors, (self.qp_xi[el_xi], self.qp_eta[el_eta], self.qp_zeta[el_zeta]))
            # self.N[el] = NN[0]
            # if self.derivative_order > 0:
            #     self.N_xi[el] = NN[1]
            #     if self.derivative_order > 1:
            #         self.N_xixi[el] = NN[2]

    def build_Q(self, L, H, B, Greville=False, Fuzz=True):
        r"""Constructs reference position vector

                    Parameters
        ----------    
        L, H : float
            Length and height of the rectangle.
        Fuzz : boolean
            Fuzz option adds a random, small displacement to the initial configuration, in order to avoid singularities in the material


        Returns
        -------
        Q : numpy.ndarray
            ()-by-() matrix with the X- and Y- coordinates of the nodes. 

        """

        if Greville and self.basis == 'splines':
            X = np.zeros(self.nn_xi)
            Y = np.zeros(self.nn_eta)
            Z = np.zeros(self.nn_zeta)
            for i in range(len(X)):
                X[i] = np.sum(self.Xi[i+1:i+self.p+1])
            for i in range(len(Y)):
                Y[i] = np.sum(self.Eta[i+1:i+self.q+1])
            for i in range(len(Z)):
                Z[i] = np.sum(self.Zeta[i+1:i+self.r+1])
            X = X * L / self.p
            Y = Y * H / self.q
            Z = Z * B / self.r
        
        else:
            # Number of nodes per in x and y
            X = np.linspace(0, L, self.nn_xi)
            Y = np.linspace(0, H, self.nn_eta)
            Z = np.linspace(0, B, self.nn_zeta)
            # row-wise 1D index
        Xs = np.tile(X, (self.nn_eta) * (self.nn_zeta))
        Ys = np.tile(np.repeat(Y, self.nn_xi), self.nn_zeta)
        Zs = np.repeat(Z,(self.nn_eta*self.nn_xi))
        # manipulate reference coordinates with small random numbers
        # prevents singular stifness matrix
        if Fuzz:
            Xs += np.random.rand(len(Xs))*L*0.01
            Ys += np.random.rand(len(Ys))*H*0.01
            Zs += np.random.rand(len(Ys))*B*0.01
        # build generalized coordinates
        Q = np.hstack((Xs, Ys, Zs))
        return Q

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

    def compute_all_N_1D_lagrange(self, nel_1D, nQP_1D, p_1D):
        r"""Computes the shape functions and their derivatives for all Gauss points of all elements. Also returns weighting of the Gauss points.

        Returns
        -------
        N : numpy.ndarray
            (n_QP)-by-((p+1)*(q+1)) array holding for each Gauss point on each element the shape function values.
        dN_dvxi : numpy.ndarray
            (n_QP)-by-((p+1)*(q+1))-by-(2) array holding for each Gauss point on each element the derivative of the shape function wrt xi and eta.

        """
        # compute Gauss points
        gp, wp = np.polynomial.legendre.leggauss(nQP_1D)

        N = np.zeros((nQP_1D, p_1D+1))
        dN = np.zeros((nQP_1D, p_1D+1))

        for gi in range(nQP_1D):
                N[gi], dN[gi] = lagrange1D(gp[gi],p_1D)

        return N, dN, gp, wp

    def compute_all_N_3D_splines(self, PointsOnEdge=False):
        r"""Computes the shape functions and their derivatives for all Gauss points of all elements. Also returns weighting of the Gauss points.

        Parameters
        -------
        PointsOnEdge: Bool
            Enables a linear distribution of "Gauss" points, including Points on the edges of the elements. This is not used for integration, but only for visualiztation.
        Returns
        -------
        N : numpy.ndarray
            (n_el)-by-(n_QP)-by-((p+1)*(q+1)) array holding for each Gauss point on each element the shape function values.
        dN_dvxi : numpy.ndarray
            (n_el)-by-(n_QP)-by-((p+1)*(q+1))-by-(2) array holding for each Gauss point on each element the derivative of the shape function wrt xi and eta.
        d2N_dvxi2 : numpy.ndarray
            (n_el)-by-(n_QP)-by-((p+1)*(q+1))-by-(2)-by-(2) array holding for each Gauss point on each element the second derivatives of the shape function wrt xi and eta.
        xi_gp, eta_gp : numpy.ndarray
            (nel_xi)-by-(nQP_xi) array holding the position of the gauss points in xi or eta direction (for the purpose of plotting).
        wp : numpy.ndarray
            (n_QP) array holding for the Gauss points on one element the respective Gauss point weighting.

        """
        N = np.zeros((self.nel, self.nqp, self.nn_el))
        dN_dvxi = np.zeros((self.nel, self.nqp, self.nn_el, 3))
        d2N_dvxi2 = np.zeros((self.nel, self.nqp, self.nn_el, 3, 3))
        xi_gp = np.zeros((self.nel_xi, self.nqp_xi))
        eta_gp = np.zeros((self.nel_eta, self.nqp_eta))
        nu_gp = np.zeros((self.nel_zeta, self.nqp_zeta))
        w_gp = np.zeros((self.nel, self.nqp))
        if not PointsOnEdge:
            #TODO: use generic gauss method
            gp_xi, wp_xi = np.polynomial.legendre.leggauss(self.nqp_xi)
            gp_eta, wp_eta = np.polynomial.legendre.leggauss(self.nqp_eta)
            gp_nu, wp_nu = np.polynomial.legendre.leggauss(self.nqp_zeta)
        else:
            gp_xi, wp_xi = np.linspace(start = -1, stop = 1, num = self.nqp_xi), np.ones(self.nqp_xi)
            gp_eta, wp_eta = np.linspace(start = -1, stop = 1, num = self.nqp_eta), np.ones(self.nqp_eta)
            gp_nu, wp_nu = np.linspace(start = -1, stop = 1, num = self.nqp_zeta), np.ones(self.nqp_zeta)

        for el in range(self.nel):
            el_xi, el_eta, el_nu = self.split_el(el)
            # TODO: ONLY WORKS FOR KNOT VECTORS WITHOUT REPEATED VALUES
            xi_interval = self.Xi[el_xi+self.p:el_xi+self.p+2]
            eta_interval = self.Eta[el_eta+self.q:el_eta+self.q+2]
            nu_interval = self.Zeta[el_nu+self.r:el_nu+self.r+2]

            # the xi and eta position of all gauss points located in the element
            xi_gp[el_xi] = xi_interval[0] + (xi_interval[1] - xi_interval[0]) * (gp_xi + 1) / 2
            eta_gp[el_eta] = eta_interval[0] + (eta_interval[1] - eta_interval[0]) * (gp_eta + 1) / 2
            nu_gp[el_nu] = nu_interval[0] + (nu_interval[1] - nu_interval[0]) * (gp_nu + 1) / 2

            for i in range(self.nqp):
                i_xi, i_eta, i_nu = self.split_gp(i)
                w_gp[el, i] = wp_xi[i_xi] / 2 * (xi_interval[1] - xi_interval[0])  *  wp_eta[i_eta] / 2 * (eta_interval[1] - eta_interval[0]) *  wp_nu[i_nu] / 2 * (nu_interval[1] - nu_interval[0])

            N[el], dN_dvxi[el], d2N_dvxi2[el] = bSplineBasis3D(self.Xi, self.Eta, self.Zeta, self.p, self.q, self.r, xi_gp[el_xi], eta_gp[el_eta], nu_gp[el_nu])

        #wp = np.zeros((self.nQP))
        #for i in range(self.nQP):
         #   i_xi, i_eta, i_nu = self.split_gp(i)
            #wp[i] = wp_xi[i_xi] * wp_eta[i_eta]
        return N, dN_dvxi, d2N_dvxi2, xi_gp, eta_gp, nu_gp, w_gp



    def compute_Fhat0_inv(self, D=np.eye(3)):
            r"""Computes the reference configuration of the body in the physical space and computes gradients in this configuration.

            Parameters
            ----------    
            D : numpy.ndarray
                Rotation matrix, describes rotation of basis (D1, D2) against basis (E1, E2). (see  "Dynamics of pantographic fabrics")
            
            Returns
            -------
            Fhat0_inv : numpy.ndarray
                (n_el)-by-(n_QP)-by-(2)-by-(2) array holding for each Gauss point the inverse of the gradient (wrp \xi \eta) of the mapping ? from element space to physical space. See Bonet 1997 eq. (7.6b) dvX/dvxi
            w_J0 : numpy.ndarray
                (n_el)-by-(n_QP) array holding the determinant of Fhat0_inv of each Gauss point, multiplied by the respective Gauss point weighting.
            nablaX_N : numpy.ndarray
                (n_el)-by-(n_QP)-by-((p+1)*(q+1))-by-(2) array holding for each Gauss point the gradients of the shape function wrt the physical coordinate \vX. See Bonet 1997 eq. (7.6)
                #TODO: why not write dN_dvX instead of nablaX_N?
            """
            Fhat0_inv = np.empty((self.nel, self.nqp, self.nq_n, self.nq_n))
            w_J0 = np.empty((self.nel, self.nqp))
            nablaX_N = np.empty((self.nel, self.nqp, self.nn_el, self.nq_n))
            for el in range(self.nel):
                Qe = self.Q[self.elDOF[el]]

                for i in range(self.nqp):
                    dN = self.dN_dvxi[el,i]

                    Fhat0 = np.zeros((self.nq_n, self.nq_n))
                    for a in range(self.nn_el):
                        Fhat0 += D @ np.outer(Qe[self.nodalDOF[a]], dN[a]) # returns 2x2 matrix
                    
                    Fhat0_inv[el, i] = inverse3D(Fhat0)
                    w_J0[el, i] = determinant3D(Fhat0) * self.wp[el,i]

                    for a in range(self.nn_el):
                            nablaX_N[el,i,a] = dN[a] @ Fhat0_inv[el, i]  

            return Fhat0_inv, w_J0, nablaX_N

    # split iterators into xi, eta and nu components
    def split_el(self,el):
        el_xi = el % self.nel_xi
        el_eta = (el // self.nel_xi) % self.nel_eta
        el_nu = el // (self.nel_xi * self.nel_eta)
        return el_xi, el_eta, el_nu

    def split_a(self,a):
        a_xi = a % (self.p+1)
        a_eta = (a // (self.p+1)) % (self.q+1)
        a_nu = a // ((self.p+1) * (self.q+1))
        return a_xi, a_eta, a_nu

    def split_gp(self,i):
        i_xi = i % self.nqp_xi
        i_eta = (i // self.nqp_xi) % self.nqp_eta
        i_nu = i // (self.nqp_xi * self.nqp_eta)
        return i_xi, i_eta, i_nu

    #     # split iterators into xi and eta components
    # def split_a(self, a):
    #     # splits the iterator running over all shape functions which are non-zero at a Gauss point
    #     a_xi = a % (self.p +1)
    #     a_eta = a // (self.p +1)
    #     return a_xi, a_eta

    # def split_el(self, el):
    #     # splits the iterator running over all elements
    #     el_xi = el % self.nel_xi
    #     el_eta = el // self.nel_xi
    #     return el_xi, el_eta

    # # def split_gp(self, i):
    # #     # splits the iterator running over all Gauss points of one element.
    # #     i_xi = i % self.nQP_xi
    # #     i_eta = i // self.nQP_xi
    # #     return i_xi, i_eta


def rearrange_points(obj, el, sort = True):
    """ Rearranges either a Point Array or a sorting array like elDOF in vtk ordering
        if sort is true, obj must be a mesh object.
        If sort is false, object must ba an array with dimensions:
        (mesh.n) -by- (p+1) -by- (q+1) -by- (r+1)
    """
    #Creates a selection matrix like elDOF for vtk ordering
    elDOF_vtk = []
    if sort:
        mesh = obj
        # reshape element elDOF for easier rearrangement
        elDOF_3d = np.reshape(mesh.elDOF[el][:mesh.n],(mesh.p+1,mesh.q+1,mesh.r+1)).T
    else:
        elDOF_3d = obj[el]
    # corners
    elDOF_vtk.extend([elDOF_3d[0,0,0], elDOF_3d[-1,0,0], elDOF_3d[-1,-1,0], elDOF_3d[0,-1,0], elDOF_3d[0,0,-1], elDOF_3d[-1,0,-1],elDOF_3d[-1,-1,-1],elDOF_3d[0,-1,-1]])
    # edges
    for iz in [0,-1]:
        elDOF_vtk.extend(elDOF_3d[1:-1,0,iz])
        elDOF_vtk.extend(elDOF_3d[-1,1:-1,iz])
        elDOF_vtk.extend(elDOF_3d[1:-1,-1,iz])
        elDOF_vtk.extend(elDOF_3d[0,1:-1,iz])
    for ix in [0,-1]:
        elDOF_vtk.extend(elDOF_3d[ix,0,1:-1])
    for ix in [0,-1]:
        elDOF_vtk.extend(elDOF_3d[ix,-1,1:-1])

    # faces 
    # yz
    for ix in [0,-1]:
        for iz in range(elDOF_3d.shape[2]-2):
            elDOF_vtk.extend(elDOF_3d[ix,1:-1,iz+1])
    # xz
    for iy in [0,-1]:
        for iz in range(elDOF_3d.shape[2]-2):
            elDOF_vtk.extend(elDOF_3d[1:-1,iy,iz+1])
    # xy
    for iz in [0,-1]:
        for iy in range(elDOF_3d.shape[1]-2):
            elDOF_vtk.extend(elDOF_3d[1:-1,iy+1,iz])
    # Volume
    for iz in range(elDOF_3d.shape[2]-2):
        for iy in range(elDOF_3d.shape[1]-2):
            for ix in range(elDOF_3d.shape[0]-2):
                elDOF_vtk.extend([elDOF_3d[ix+1,iy+1,iz+1]])

    return np.array(elDOF_vtk)

def export_mesh_vtk_meshio_lagrange(mesh,body, q, filename):
    from cardillo_fem.utility.meshio.vtu._vtu import write
    from cardillo_fem.utility.meshio import _mesh, _common

    elDOF_vtk = np.zeros_like(mesh.elDOF[:,:mesh.n])
    for el in range(mesh.nel):
        elDOF_vtk[el] = rearrange_points(mesh,el,sort = True)

    def compute_F(D=np.eye(3)):
                # F calculated element wise at q
                dN_nodes = np.zeros((mesh.n, mesh.n, mesh.nodal_DOF))
                for node in range(mesh.n):
                    a_xi, a_eta, a_nu = mesh.split_a(node)
                    xi = (2*a_xi-1) /(mesh.p+1)
                    eta = (2*a_eta-1) / (mesh.q+1)
                    nu = (2*a_nu-1) / (mesh.r+1)
                    _, dN_nodes[node] = lagrange3D(mesh.p, mesh.q, mesh.r, xi, eta, nu)
                Fhat0_inv = np.zeros((mesh.nel, mesh.n, mesh.nodal_DOF, mesh.nodal_DOF))
                F = np.zeros((mesh.nel , mesh.n, mesh.nodal_DOF, mesh.nodal_DOF))
                nablaX_N = np.zeros((mesh.nel, mesh.n, mesh.n, mesh.nodal_DOF))
                for el in range(mesh.nel):
                    qe = q[mesh.elDOF[el]]
                    Qe = mesh.Q[mesh.elDOF[el]]
                    for i in range(mesh.n):
                        dN = dN_nodes[i]
                        Fhat0 = np.zeros((mesh.nodal_DOF, mesh.nodal_DOF))
                        for a in range(mesh.n):
                            Fhat0 += D @ np.outer(Qe[mesh.ndDOF[a]], dN[a])
                        Fhat0_inv[el, i] = inverse3D(Fhat0)

                        for a in range(mesh.n):
                            nablaX_N[el,i,a] = dN[a] @ Fhat0_inv[el, i] 
                            F[el,i] += np.outer(qe[mesh.ndDOF[a]], nablaX_N[el,i,a])

                return F


    def nodal_extrapolation():
        # Extrapolation on the nodes of P, J, W
        # Get F on nodes of each Element
        F_nodes = compute_F()       
        # Flatten F and use nodal averaging
        F = np.zeros((mesh.ntot,mesh.nodal_DOF,mesh.nodal_DOF))
        _, counts = np.unique(mesh.elDOF,return_counts = True)
        counts = counts[:mesh.ntot]
        for el in range(mesh.nel):  
            F[mesh.elDOF[el][:mesh.n]] += F_nodes[el]
        F /= counts[:,np.newaxis,np.newaxis]
        J_all = []
        P_all = []
        W_all = []
        # J_all_p = []
        # F_all_p, P_all_p, W_all_p = body.eval_all_gp(q)
        # for F_i in F_all_p:
        #     J_all_p.append(np.array([determinant2D(F_i)]))
        for F_i in F:           
            J_all.append(np.array([determinant3D(F_i)]))
            P_all.append(body.mat.P(F_i))
            W_all.append(body.mat.W(F_i.T @ F_i))
        # Flatten P
        P_all_t = np.reshape(P_all, (mesh.ntot , mesh.nodal_DOF * mesh.nodal_DOF))
        F_all_t = np.reshape(F, (mesh.ntot , mesh.nodal_DOF * mesh.nodal_DOF))

        return F_all_t,np.array(J_all), P_all_t, np.array(W_all)
        

    F_all, J_all, P_all, W_all = nodal_extrapolation()

    # TODO: How picky is meshio with array types?
    # TODO: Is cell data needed? No
    points = q.reshape((3,mesh.ntot)).T
    cells_type = ["VTK_LAGRANGE_HEXAHEDRON"]*mesh.nel
    cells_array = np.reshape(elDOF_vtk,(mesh.nel, mesh.n))
    cells = list(zip(cells_type, [cells_array]))
    point_data = {"F": F_all, "J": J_all, "P": P_all, "W": W_all}

    

    mesh_meshio = _mesh.Mesh(
        points,
        cells,
        point_data=point_data,
        #cell_data=cell_data,
        #field_data=field_data,
        #point_sets,
        #cell_sets,
        #gmsh_periodic,
        #info,
    )

    write(filename, mesh_meshio, binary=False)

def point_plt_3d(mesh,q):
    qt = q.reshape((3,mesh.ntot))
    X,Y,Z = qt[0],qt[1],qt[2]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X,Y,Z,marker='p')
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')

    plt.grid()
    plt.show()


def export_mesh_vtk_meshio_splines(mesh_calc, body_calc,  q, filename):
    r""" Computes the Bezier patches of the NURBS surface, and exports the data as a .vtk file.

    """
    from cardillo_fem.discretization.bspline import decomposeVolumeToBezier, split_nb, approximate_surface
    from cardillo_fem.utility.algebra import determinant3D
    from cardillo_fem.utility.meshio import _mesh, _common
    from cardillo_fem.utility.meshio.vtu._vtu import write
    from cardillo_fem.bodies.first_gradient import FirstGradient
    #from cardillo_fem.bodies.first_gradient.first_gradient import internal_forces


    # create new mesh for export, with desired number of evaluation points
    mesh_exp = Mesh(basis='splines',p=mesh_calc.p, q=mesh_calc.q, r=mesh_calc.r, d=mesh_calc.d, 
                    nel_xi=mesh_calc.nel_xi, nel_eta=mesh_calc.nel_eta, nel_nu=mesh_calc.nel_nu, L=mesh_calc.L , H=mesh_calc.H, B=mesh_calc.B,
                    nQP_xi=5, nQP_eta=5, nQP_nu=5,
                    PointsOnEdge=True)
    body_exp = FirstGradient(body_calc.mat, mesh_exp)

    n = mesh_exp.nel_xi + mesh_exp.p
    m = mesh_exp.nel_eta + mesh_exp.q
    o = mesh_exp.nel_zeta + mesh_exp.r



    # evaluate individual points (for now: use gauss points)
    F_all_gp, P_all_gp, W_all_gp =  body_exp.eval_all_gp(q)  #TODO: this is first_gradient specific at the moment
    #J_all_gp = np.zeros(F_all_gp[:,0,0].shape)
    J_all = []
    W_all = []
    P_all = []
    F_all = []
    for i in range(len(F_all_gp)):
        J_all.append([determinant3D(F_all_gp[i])])
        W_all.append([W_all_gp[i]])
        P_all.append([P_all_gp[i,j,k] for j in range(mesh_calc.nodal_DOF) for k in range(mesh_calc.nodal_DOF)])
        F_all.append([F_all_gp[i,j,k] for j in range(mesh_calc.nodal_DOF) for k in range(mesh_calc.nodal_DOF)])


    # compute xi and eta position of all evaluated points (corresponds to Gauss points of mesh_exp)
    xi_vals = np.zeros((mesh_exp.nel_xi * mesh_exp.nqp_xi))
    eta_vals = np.zeros((mesh_exp.nel_eta * mesh_exp.nqp_eta))
    nu_vals = np.zeros((mesh_exp.nel_zeta * mesh_exp.nqp_zeta))
    for el_xi in range(mesh_exp.nel_xi):
        for i in range(mesh_exp.nqp_xi):
            xi_vals[i + el_xi *mesh_exp.nqp_xi] = mesh_exp.qp_xi[el_xi,i]
    for el_eta in range(mesh_exp.nel_eta):
        for i in range(mesh_exp.nqp_eta):
            eta_vals[i + el_eta *mesh_exp.nqp_eta] = mesh_exp.qp_eta[el_eta,i]
    for el_nu in range(mesh_exp.nel_zeta):
        for i in range(mesh_exp.nqp_zeta):
            nu_vals[i + el_nu *mesh_exp.nqp_zeta] = mesh_exp.qp_zeta[el_nu,i]

    nxi = mesh_exp.nqp_xi * mesh_exp.nel_xi
    neta = mesh_exp.nqp_eta * mesh_exp.nel_eta
    nnu = mesh_exp.nqp_zeta * mesh_exp.nel_zeta
 

    ### create bezier patches

    # rearrange points
    Pw = np.zeros((n, m, o, 3))
    for j in range(mesh_exp.nn):
        j_xi = j % n
        j_eta =  (j // n) % m   
        j_nu = j // (n*m)
        Pw[j_xi, j_eta, j_nu] = np.array([q[j], q[j+mesh_exp.nn], q[j+2*mesh_exp.nn]])


    Qw, nb_xi, nb_eta, nb_nu = decomposeVolumeToBezier(n-1, mesh_exp.p, mesh_exp.Xi, m-1, mesh_exp.q, mesh_exp.Eta, o-1, mesh_exp.r, mesh_exp.Zeta, Pw)

    # number of bezier patches
    nb = nb_xi * nb_eta * nb_nu

    offset = np.zeros(nb)
    cells = []
    HigherOrderDegree_patches = []
    np_xi = Qw.shape[1]
    np_eta = Qw.shape[2]
    np_nu = Qw.shape[3]
    conn = np.zeros((nb * np_xi * np_eta * np_nu))

    # number of points on corners, edges and faces per patch
    # np_corners = 8
    # np_edges = 4*(np_xi-2) + 4*(np_eta-2) + 4*(np_nu-2)
    # np_faces = 2 * np_xi*np_eta + 2 * np_eta*np_nu + 2*np_nu*np_xi - 3 * np_corners - 2*np_edges
    # np_volume = np_xi*np_eta*np_nu - np_corners - np_edges - np_faces
    # idx_edges = np_corners + np_edges
    # idx_faces = idx_edges + np_faces
    # idx_volume = idx_faces + np_volume

    # use same function for lagrange and bezier

    def build_point_dat(P_all_gp):
        r""" Fits control points at specified xi- and eta- values to input data P_all_gp via a least-squares algorithm.
        """
        dim = len(P_all_gp[0])
        P_pd_vec = approximate_surface(P_all_gp, nxi, neta, mesh_exp.p, mesh_exp.q, 
                                        uk_vl=(xi_vals,eta_vals), 
                                        kv_u=mesh_exp.Xi, kv_v=mesh_exp.Eta, 
                                        ctrlpts_size_u=n, ctrlpts_size_v=m, 
                                        interp=True)

        P_pd = np.zeros((n, m, dim))
        # note that indexing is different here than usual: sorted collumn-wise
        for j in range(mesh_exp.nn):
            j_xi = j // m
            j_eta =  j % m 
            P_pd[j_xi, j_eta] = P_pd_vec[j_eta  + j_xi *m]

        Q_pd, _, _ = decomposeSurfaceToBezier(n-1, mesh_exp.p, mesh_exp.Xi, m-1, mesh_exp.q, mesh_exp.Eta, P_pd)

        point_dat = np.zeros(( nb * np_xi * np_eta, dim))
        for i in range(nb):
            
            point_data_patch = rearrange_points(Q_pd, dim, i)
         
            point_range =  np.arange(i*(np_xi*np_eta), (i+1)*(np_xi*np_eta))  # point_range of patch
            point_dat[point_range ,:] = point_data_patch
        return point_dat


    # initialize the array containing all points of all patches
    points = np.zeros(( nb * np_xi * np_eta * np_nu, 3))
    # iterate over all bezier patches
    for i in range(nb):
        #i_xi, i_eta = split_nb(i, nb_xi, nb_eta)

        r""" create list of all points (ignoring possible redundancies).
            
        """
        # point_range of patch
        point_range =  np.arange(i*(np_xi*np_eta*np_nu), (i+1)*(np_xi*np_eta*np_nu)) 

        # define points
        points_patch = np.zeros((np_xi * np_eta * np_nu, 3))
        points_patch[:,:] = rearrange_points(Qw,i,sort=False)
        points[point_range ,:] = points_patch

        # define connectivity or vertices that belongs to each element
        conn[point_range] = point_range

        # Define offset of last vertex of each element
        offset[i] =  point_range[-1] +1

        # Define cell types
        #ctype[i] = VtkBezierQuadrilateral.tid
        cells.append( ("VTK_BEZIER_HEXAHEDRON", point_range[None,:]) )


    # Define HigherOrderDegree
    HigherOrderDegree_patches = np.stack((np.ones(nb)*(np_xi-1), np.ones(nb)*(np_eta-1), np.ones(nb)*(np_nu-1),  )).T     # third degree is 0 for planar case

    cell_data = { "HigherOrderDegrees": HigherOrderDegree_patches}

    # point_data = {
    #                 "J": build_point_dat(J_all),
    #                 "W": build_point_dat(W_all),
    #                 "P": build_point_dat(P_all),
    #                 "F": build_point_dat(F_all),
    #                 }


    # create meshio Mesh
    mesh_meshio = _mesh.Mesh(
        points,
        cells,
        #point_data=point_data,
        cell_data=cell_data,
        #field_data=field_data,
        #point_sets,
        #cell_sets,
        #gmsh_periodic,
        #info,
    )

    write(filename, mesh_meshio, binary=False)

if __name__ == "__main__":
    degrees = (2, 3, 1)
    QP_shape = (3, 4, 2)
    element_shape = (1, 2, 3)

    from cardillo.discretization.B_spline import Knot_vector

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    
    mesh = Mesh((Xi, Eta, Zeta), QP_shape, derivative_order=1, basis='B-spline', nq_n=3)