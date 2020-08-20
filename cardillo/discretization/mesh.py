import numpy as np
# from cardillo_fem.discretization.lagrange import lagrange2D, lagrange1D, lagrange3D
from cardillo.discretization.B_spline import uniform_knot_vector, B_spline_basis1D, B_spline_basis2D, B_spline_basis3D
from cardillo.math.algebra import inverse2D, determinant2D, inverse3D, determinant3D
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

def rearrange_points(obj, el, sort=True):
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
    # elDOF_3d = obj

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

def export_mesh_vtk_meshio_splines(mesh, body_calc, q, filename):
    r""" Computes the Bezier patches of the NURBS surface, and exports the data as a .vtk file.
    """
    # from cardillo_fem.discretization.bspline import decomposeVolumeToBezier, split_nb, approximate_surface
    from meshio import _mesh
    from meshio.vtu._vtu import write
    # from cardillo_fem.bodies.first_gradient import FirstGradient
    #from cardillo_fem.bodies.first_gradient.first_gradient import internal_forces

    # # create new mesh for export, with desired number of evaluation points
    # mesh_exp = Mesh(basis='splines', p=mesh_calc.p, q=mesh_calc.q, r=mesh_calc.r, d=mesh_calc.d, 
    #                 nel_xi=mesh_calc.nel_xi, nel_eta=mesh_calc.nel_eta, nel_nu=mesh_calc.nel_nu, L=mesh_calc.L , H=mesh_calc.H, B=mesh_calc.B,
    #                 nQP_xi=5, nQP_eta=5, nQP_nu=5,
    #                 PointsOnEdge=True)



    # body_exp = FirstGradient(body_calc.mat, mesh_exp)


    # # evaluate individual points (for now: use gauss points)
    # F_all_gp, P_all_gp, W_all_gp =  body_exp.eval_all_gp(q)  #TODO: this is first_gradient specific at the moment
    # #J_all_gp = np.zeros(F_all_gp[:,0,0].shape)
    # J_all = []
    # W_all = []
    # P_all = []
    # F_all = []
    # for i in range(len(F_all_gp)):
    #     J_all.append([determinant3D(F_all_gp[i])])
    #     W_all.append([W_all_gp[i]])
    #     P_all.append([P_all_gp[i,j,k] for j in range(mesh.nodal_DOF) for k in range(mesh.nodal_DOF)])
    #     F_all.append([F_all_gp[i,j,k] for j in range(mesh.nodal_DOF) for k in range(mesh.nodal_DOF)])


    # # compute xi and eta position of all evaluated points (corresponds to Gauss points of mesh_exp)
    # xi_vals = np.zeros((mesh_exp.nel_xi * mesh_exp.nqp_xi))
    # eta_vals = np.zeros((mesh_exp.nel_eta * mesh_exp.nqp_eta))
    # nu_vals = np.zeros((mesh_exp.nel_zeta * mesh_exp.nqp_zeta))
    # for el_xi in range(mesh_exp.nel_xi):
    #     for i in range(mesh_exp.nqp_xi):
    #         xi_vals[i + el_xi *mesh_exp.nqp_xi] = mesh_exp.qp_xi[el_xi,i]
    # for el_eta in range(mesh_exp.nel_eta):
    #     for i in range(mesh_exp.nqp_eta):
    #         eta_vals[i + el_eta *mesh_exp.nqp_eta] = mesh_exp.qp_eta[el_eta,i]
    # for el_nu in range(mesh_exp.nel_zeta):
    #     for i in range(mesh_exp.nqp_zeta):
    #         nu_vals[i + el_nu *mesh_exp.nqp_zeta] = mesh_exp.qp_zeta[el_nu,i]

    # nxi = mesh_exp.nqp_xi * mesh_exp.nel_xi
    # neta = mesh_exp.nqp_eta * mesh_exp.nel_eta
    # nnu = mesh_exp.nqp_zeta * mesh_exp.nel_zeta
 

    ### create bezier patches

    # rearrange points
    nn_xi = mesh.nn_xi
    nn_eta = mesh.nn_eta
    nn_zeta = mesh.nn_zeta
    points = np.zeros((nn_xi, nn_xi, nn_zeta, 3))
    for j in range(mesh.nn):
        j_xi, j_eta, j_zeta = split3D(j, mesh.nn_per_dim)
        points[j_xi, j_eta, j_zeta] = np.array([q[j], q[j+mesh.nn], q[j+2*mesh.nn]])

    Qw, nb_xi, nb_eta, nb_nu = decomposeVolumeToBezier(nn_xi-1, mesh_exp.p, mesh_exp.Xi, nn_eta-1, mesh_exp.q, mesh_exp.Eta, nn_zeta-1, mesh_exp.r, mesh_exp.Zeta, points)

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
                                        ctrlpts_size_u=nn_xi, ctrlpts_size_v=nn_eta, 
                                        interp=True)

        P_pd = np.zeros((nn_xi, nn_eta, dim))
        # note that indexing is different here than usual: sorted collumn-wise
        for j in range(mesh_exp.nn):
            j_xi = j // nn_eta
            j_eta =  j % nn_eta 
            P_pd[j_xi, j_eta] = P_pd_vec[j_eta  + j_xi *nn_eta]

        Q_pd, _, _ = decomposeSurfaceToBezier(nn_xi-1, mesh_exp.p, mesh_exp.Xi, nn_eta-1, mesh_exp.q, mesh_exp.Eta, P_pd)

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

def test_mesh():
    degrees = (2, 3, 1)
    QP_shape = (3, 4, 2)
    element_shape = (1, 2, 3)

    from cardillo.discretization.B_spline import Knot_vector

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    
    mesh = Mesh((Xi, Eta, Zeta), QP_shape, derivative_order=2, basis='B-spline', nq_n=3)

    cube_shape = (3, 3, 3)
    Q_cube = cube(cube_shape, mesh, Greville=False, Fuzz=0)
    # scatter_Qs(Q_cube)

    num = 5
    xis = np.linspace(0, 1, num=num)
    etas = np.linspace(0, 1, num=num)
    zetas = np.linspace(0, 1, num=num)
    
    xis2 = np.tile(xis, num * num)
    etas2 = np.tile(np.repeat(etas, num), num)
    zetas2 = np.repeat(zetas, num * num)
    knots = np.vstack((xis2, etas2, zetas2)).T
    x = mesh.interpolate(knots, Q_cube)
    
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    # ax.set_zlabel('z [m]')
    # max_val = np.max(np.abs(x))
    # ax.set_xlim3d(left=-max_val, right=max_val)
    # ax.set_ylim3d(bottom=-max_val, top=max_val)
    # ax.set_zlim3d(bottom=-max_val, top=max_val)
    # ax.scatter(*x.T, marker='p')
    # plt.show()

    kappa0_xi_inv, N_X, w_J0 = mesh.reference_mappings(Q_cube)

# if __name__ == "__main__":
#     export_mesh_vtk_meshio_splines(0, 0, 0, 0)