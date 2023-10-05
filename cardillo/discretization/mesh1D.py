import numpy as np
from scipy.sparse.linalg import spsolve
from cardillo.discretization.b_spline import (
    B_spline_basis1D,
    decompose_B_spline_curve,
    flat1D_vtk,
)
from cardillo.discretization.lagrange import lagrange_basis1D
from cardillo.discretization.hermite import cubic_Hermite_basis_1D
from cardillo.discretization.gauss import gauss, lobatto
from cardillo.utility.coo_matrix import CooMatrix


def line2D(L, mesh, Greville=False, Fuzz=None):
    X = np.linspace(0, L, mesh.nn)
    Y = np.zeros(mesh.nn)
    if Greville:
        for i in range(len(X)):
            X[i] = np.sum(mesh.Xi.data[i + 1 : i + mesh.p + 1])
        X = X * L / mesh.p

    # manipulate reference coordinates with small random numbers
    # prevents singular stifness matrix
    if Fuzz is not None:
        X += np.random.rand(len(X)) * L * Fuzz
        Y += np.random.rand(len(Y)) * Fuzz

    # build generalized coordinates
    return np.concatenate((X, Y))


class Mesh1D:
    def __init__(
        self,
        knot_vector,
        nquadrature,
        dim_q,
        derivative_order=1,
        basis="B-spline",
        quadrature="Gauss",
        dim_u=None,
    ):
        self.basis = basis
        self.nelement = knot_vector.nel
        self.knot_vector = knot_vector
        self.data = self.knot_vector.data
        self.degree = self.knot_vector.degree
        self.derivative_order = derivative_order
        self.nquadrature = nquadrature
        if quadrature == "Gauss":
            self.quadrature = gauss
        elif quadrature == "Lobatto":
            self.quadrature = lobatto
        else:
            raise NotImplementedError(
                f"Quadrature method '{quadrature}' is not implemented!"
            )

        # we might have different meshes for q and u, e.g. quaternions for
        # describing spatial rotations
        if dim_u is None:
            self.dim_u = dim_u = dim_q

        if basis in ["Lagrange", "B-spline", "Lagrange_Disc"]:
            self.nnodes_per_element = (
                self.degree + 1
            )  # number of nodes influencing each element
            self.dim_q = dim_q  # number of degrees of freedom per node
            self.nbasis_element = (
                self.nnodes_per_element
            )  # number of basis function per element
        elif basis == "Hermite":
            self.nnodes_per_element = 4
            self.dim_q = dim_q  # number of degrees of freedom per node
            self.nbasis_element = 4  # number of basis function per element
        else:
            raise NotImplementedError("")
        self.nq_per_element = (
            self.nnodes_per_element * dim_q
        )  # total number of generalized coordinates per element
        self.nu_per_element = (
            self.nnodes_per_element * dim_u
        )  # total number of generalized velocities per element

        # Boolean connectivity matrix for element polynomial_degrees of
        # freedom. This is used to extract the element degrees of freedom via
        # q[elDOF[el]] = q_e = C^e * q.
        self.elDOF = np.zeros((self.nelement, self.nq_per_element), dtype=int)
        self.elDOF_u = np.zeros((self.nelement, self.nu_per_element), dtype=int)

        if basis == "Lagrange":
            # total number of nodes
            self.nnodes = self.degree * self.nelement + 1

            elDOF_el = np.concatenate(
                [
                    np.arange(self.nnodes_per_element) + i * self.nnodes
                    for i in range(dim_q)
                ]
            )
            elDOF_el_u = np.concatenate(
                [
                    np.arange(self.nnodes_per_element) + i * self.nnodes
                    for i in range(dim_u)
                ]
            )

            for el in range(self.nelement):
                self.elDOF[el] = elDOF_el + el * self.degree 
                self.elDOF_u[el] = elDOF_el_u + el * self.degree

            self.vtk_cell_type = "VTK_LAGRANGE_CURVE"
        elif basis == "Lagrange_Disc":
            # total number of nodes
            self.nnodes = (self.degree + 1) * self.nelement

            elDOF_el = np.concatenate(
                [
                    np.arange(self.nnodes_per_element) + i * self.nnodes
                    for i in range(dim_q)
                ]
            )

            elDOF_el_u = np.concatenate(
                [
                    np.arange(self.nnodes_per_element) + i * self.nnodes
                    for i in range(dim_u)
                ]
            )

            for el in range(self.nelement):
                self.elDOF[el] = elDOF_el + el * (self.degree + 1) 
                self.elDOF_u[el] = elDOF_el_u + el * (self.degree + 1) 
            
            self.vtk_cell_type = "VTK_LAGRANGE_CURVE"
        elif basis == "Hermite":
            # raise NotImplementedError("Adapt according to new ordering of q!")

            # total number of nodes
            self.nnodes = 2 * (self.nelement + 1)

            elDOF_el = np.concatenate(
                [np.arange(4) + i * self.nnodes for i in range(dim_q)]
            )
            elDOF_el_u = np.concatenate(
                [np.arange(4) + i * self.nnodes for i in range(dim_u)]
            )

            for el in range(self.nelement):
                self.elDOF[el] = elDOF_el + 2 * el
                self.elDOF_u[el] = elDOF_el_u + 2 * el

            # TODO: Does VTK implement Hermite curves?
            # TODO: We should convert the Hermite spline into Bezier cells.
            self.vtk_cell_type = "VTK_LAGRANGE_CURVE"
        elif basis == "B-spline":
            # total number of nodes
            self.nnodes = self.degree + self.nelement

            elDOF_el = np.concatenate(
                [
                    np.arange(self.nnodes_per_element) + i * self.nnodes
                    for i in range(dim_q)
                ]
            )
            elDOF_el_u = np.concatenate(
                [
                    np.arange(self.nnodes_per_element) + i * self.nnodes
                    for i in range(dim_u)
                ]
            )

            for el in range(self.nelement):
                self.elDOF[el] = elDOF_el + el
                self.elDOF_u[el] = elDOF_el_u + el

            self.vtk_cell_type = "VTK_BEZIER_CURVE"

        # todal number of degrees of freedoms

        self.nq = self.nnodes * dim_q 
        self.nu = self.nnodes * dim_u

        # construct the Boolean selection matrix that choses the coordinates
        # of an individual node via q[nodalDOF[a]] = C^a * q
        self.nodalDOF = np.arange(self.nq).reshape(self.nnodes, dim_q, order="F")
        self.nodalDOF_u = np.arange(self.nu).reshape(self.nnodes, dim_u, order="F")


        # Boolean connectivity matrix for nodal polynomial_degrees of freedom
        # inside each element. This is only required if multiple fields are
        # discretized. It is used as qe[nodalDOF_element_[a]] = q_e^a = C^a * q_e
        self.nodalDOF_element = np.arange(self.nq_per_element).reshape(
            self.nnodes_per_element, dim_q, order="F"
        )
        self.nodalDOF_element_u = np.arange(self.nu_per_element).reshape(
            self.nnodes_per_element, dim_u, order="F"
        )

        # transform quadrature points on element intervals
        self.quadrature_points() # non cambia con Lagrange_Disc

        # evaluate element shape functions at quadrature points
        self.shape_functions() # non cambia con Lagrange_Disc

        # end_points degrees of freedom
        # self.end_points()

    def basis1D(self, xis):
        if self.basis == "B-spline":
            return B_spline_basis1D(
                self.degree,
                self.derivative_order,
                self.knot_vector.data,
                xis,
                squeeze=False,
            )
        elif self.basis == "Lagrange":
            return lagrange_basis1D(
                self.degree,
                xis,
                self.derivative_order,
                self.knot_vector,
                squeeze=False,
            )
        elif self.basis == "Lagrange_Disc":
            return lagrange_basis1D(
                self.degree,
                xis,
                self.derivative_order,
                self.knot_vector,
                squeeze=False,
            )
        elif self.basis == "Hermite":
            return cubic_Hermite_basis_1D(
                xis, self.knot_vector, self.derivative_order, squeeze=False
            )

    def eval_basis(self, xi):
        if self.basis == "B-spline":
            return B_spline_basis1D(
                self.degree, self.derivative_order, self.data, xi, squeeze=True
            )
        elif self.basis == "Lagrange":
            return lagrange_basis1D(
                self.degree, xi, self.derivative_order, self.knot_vector, squeeze=True
            )
        elif self.basis == "Lagrange_Disc":
            return lagrange_basis1D(
                self.degree, xi, self.derivative_order, self.knot_vector, squeeze=True
            )
        elif self.basis == "Hermite":
            return cubic_Hermite_basis_1D(xi, self.knot_vector, self.derivative_order)

    def quadrature_points(self):
        self.qp = np.zeros((self.nelement, self.nquadrature))
        self.wp = np.zeros((self.nelement, self.nquadrature))

        for el in range(self.nelement):
            Xi_element_interval = self.knot_vector.element_interval(el)
            self.qp[el], self.wp[el] = self.quadrature(
                self.nquadrature, interval=Xi_element_interval
            )

    def shape_functions(self):
        self.N = np.zeros((self.nelement, self.nquadrature, self.nbasis_element))
        if self.derivative_order > 0:
            self.N_xi = np.zeros((self.nelement, self.nquadrature, self.nbasis_element))
            if self.derivative_order > 1:
                self.N_xixi = np.zeros(
                    (self.nelement, self.nquadrature, self.nbasis_element)
                )

        for el in range(self.nelement):
            NN = self.basis1D(self.qp[el])
            # expression = "self.N"
            # for i in range(self.derivative_order):
            #     expression += ", self.N_" + (i + 1) * "xi"
            # eval(expression) = NN
            self.N[el] = NN[0]
            if self.derivative_order > 0:
                self.N_xi[el] = NN[1]
                if self.derivative_order > 1:
                    self.N_xixi[el] = NN[2]


    def end_points(self):
        
        if self.basis == "Lagrange_Disc":
            

            '''restituisce i DOF nei nodi, avrà dimensioni nnodes x 2*dim_q'''
            def select_end_points(**kwargs):
                nn_0 = kwargs.get("nn_0", range(self.nnodes))

                end_points = []
                for i in nn_0:
                    end_points.append(i)

                DOF_x_1 = np.array([end_points[0],end_points[self.nnodes-1]],dtype=int)  # DOF_x_1 calculate the index of the end nodes
                DOF_x_int = np.array(end_points[1:-1], dtype = int) # DOF_x_int calculate the index of the internal nodes

                if self.degree == 1:
                    DOF_x_2_int = DOF_x_int    # DOF_x_2_int calculate the index of the element external nodes 
                elif self.degree > 1:   
                    DOF_x_2_int = [DOF_x_int[i] for i in range(self.degree - 1, self.nnodes - 2, self.degree)]
                    DOF_x_int_int = [DOF_x_int[i] for i in range(0, self.nnodes - 2, self.degree)]
                    dim_int_int = np.shape(DOF_x_int_int)[0]
                    DOF_int_int = np.zeros((self.dim_q,dim_int_int), dtype=int) # DOF_x_int_int calculate the index of the element internal nodes 

                dim_2_int = np.shape(DOF_x_2_int)[0]



                DOF_12 = np.zeros((self.dim_q, 2), dtype=int)
                DOF_2_i = np.zeros((2 * self.dim_q, dim_2_int), dtype=int)
                DOF = np.zeros((2 * self.dim_q,self.nnodes), dtype=int)

                for j in range(2):
                    for i in range(self.dim_q):
                        DOF_12[i,j] = DOF_x_1[j] + i * (self.nnodes + self.nnodes_1) + j * (self.nelement - 1)
                        DOF[i, DOF_x_1[j]] = DOF_12[i,j]

                for j in range(dim_2_int):
                    for i in range(self.dim_q):
                        DOF_2_i[i,j] = DOF_x_2_int[j] + i * (self.nnodes + self.nnodes_1) + j * (self.nelement - (self.nelement - 1)) 
                        DOF_2_i[i + self.dim_q,j] = DOF_2_i[i,j] + 1
                        DOF[i,DOF_x_2_int[j]] = DOF_2_i[i,j]
                        DOF[i + self.dim_q,DOF_x_2_int[j]] = DOF_2_i[i + self.dim_q,j]

                if self.degree > 1: 
                    for j in range(dim_int_int):
                        for i in range(self.dim_q):
                            DOF_int_int[i,j] = DOF_x_int_int[j] + i * (self.nnodes + self.nnodes_1) + j * (self.nelement - (self.nelement - 1))
                            DOF[i,DOF_x_int_int[j]] = DOF_int_int[i,j]

                return DOF # DOF is an np.array where each column represents the degrees of freedom associated to a single node
                       # for example if we have 5 nnodes and 4 elements in
        else:
            def select_end_points(**kwargs):
                nn_0 = kwargs.get("nn_0", range(self.nnodes))

                end_points = []
                for i in nn_0:
                    end_points.append(i)

                DOF_x = np.array(end_points)
                nn_edge = len(end_points)
                DOF = np.zeros((self.dim_q, nn_edge), dtype=int)
                for i in range(self.dim_q):
                    DOF[i] = DOF_x + i * self.nnodes

                return DOF

    
        self.point_qDOF = (
            select_end_points(nn_0=[0]),
            select_end_points(nn_0=[self.nnodes - 1]),
        ) 

        # evaluate shape functions at the boundaries
        self.NN_bdry0 = self.eval_basis(0)
        self.NN_bdry1 = self.eval_basis(1)

    def reference_mappings(self, Q):
        """Compute inverse gradient from the reference configuration to the parameter space and scale quadrature points by its determinant. See Bonet 1997 (7.6a,b)"""
        assert self.dim_q == 3
        w_J0 = np.zeros((self.nelement, self.nquadrature))
        for el in range(self.nelement):
            Qe = Q[self.elDOF[el]]

            for i in range(self.nquadrature):
                N_xi = self.N_xi[el, i]

                kappa0_xi = np.zeros((self.dim_q, self.dim_q))
                for a in range(self.nnodes_per_element):
                    kappa0_xi += (
                        Qe[self.nodalDOF_element[a]] * N_xi[a]
                    )  # Bonet 1997 (7.6b)

                w_J0[el, i] = np.linalg.norm(kappa0_xi)

        return (w_J0,)
    
    # functions for vtk export
    def ensure_L2_projection_A(self):
        if not hasattr(self, "A"):
            A = CooMatrix((self.nnodes, self.nnodes))
            for el in range(self.nelement):
                elDOF_el = self.elDOF[el, : self.nnodes_per_element]
                Ae = np.zeros((self.nnodes_per_element, self.nnodes_per_element))
                Nel = self.N[el]
                for a in range(self.nnodes_per_element):
                    for b in range(self.nnodes_per_element):
                        for i in range(self.nquadrature):
                            Ae[a, b] += Nel[i, a] * Nel[i, b]
                A.extend(Ae, (elDOF_el, elDOF_el))
            self.A = A.tocsc()

    def rhs_L2_projection(self, field):
        _, nqp, *shape = field.shape
        dim = np.prod(shape)

        b = np.zeros((self.nnodes, dim))
        for el in range(self.nelement):
            elDOF_el = self.elDOF[el, : self.nnodes_per_element]
            be = np.zeros((self.nnodes_per_element, dim))
            Nel = self.N[el]
            for a in range(self.nnodes_per_element):
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
        q = np.zeros((self.nnodes, dim))
        for i, bi in enumerate(b.T):
            q[:, i] = spsolve(self.A, bi)

        if self.basis == "B-spline":
            # rearrange q's from solver to Piegl's 1D ordering
            Pw = q.reshape(-1, dim, order="F")

            # decompose B-spline mesh in Bezier patches
            Qw = decompose_B_spline_curve(self.knot_vector, Pw)

        elif self.basis in ["Lagrange", "Lagrange_Disc"]:
            # rearrange q's from solver to Piegl's 3D ordering
            Qw = np.zeros((self.nelement, self.degree + 1, dim))
            for el in range(self.nelement):
                for a in range(self.nnodes_per_element):
                    Qw[el, a] = q[self.elDOF[el][self.nodalDOF_element[a][0]]]

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
        if self.basis == "B-spline":
            # rearrange q's from solver to Piegl's 1D ordering
            Pw = q.reshape(-1, self.dim_q, order="F")

            # decompose B-spline mesh in Bezier patches
            Qw = decompose_B_spline_curve(self.knot_vector, Pw)

        elif self.basis in ["Lagrange", "Lagrange_Disc"]:
            # rearrange q's from solver to Piegl's 1D ordering
            Qw = np.zeros((self.nelement, self.degree + 1, self.dim_q))
            for el in range(self.nelement):
                for a in range(self.nnodes_per_element):
                    Qw[el, a] = q[self.elDOF[el][self.nodalDOF_element[a]]]


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


class MultifieldMesh1D:
    """This new mesh class should deal with multi field 1D meshes.

    It should be possible to do operations like

        for el in range(nel):
            qe = q[elDOF[el]]
            r_OP = np.zeros(3)
            psi = np.zeros(3)
            for i in (nnodes_per_element):
                r_OP += N[i] * qe[self.nodalDOF[el]["r_OP"]
                psi += N[i] * qe[self.nodalDOF[el]["psi"]

    that means we have a global elDOF for all fields but can extract the
    nodalDOF's with a given string or an index. This results in the global
    ordering as described in the beam paper and makes is convenient to create
    objects with multiple fields.
    """

    def __init__(
        self,
        knot_vectors,
        dim_qs,
        dim_us,
        derivative_orders,
        basis_functions,
        nquadrature,
        quadrature="Gauss",
    ):
        # self.nmeshes = len(knot_vectors)
        self.nelement = knot_vectors[0].nel
        for kv in knot_vectors:
            assert (
                self.nelement == kv.nel
            ), "fields have to share the same number of elements"

        # self.knot_vectors = knot_vectors
        # self.datas = [kv.data for kv in knot_vectors]
        # self.degrees = [kv.degrees for kv in knot_vectors]
        # self.derivative_orders = [kv.derivative_order for kv in knot_vectors]

        self.nquadrature = nquadrature
        if quadrature == "Gauss":
            self.quadrature = gauss
        elif quadrature == "Lobatto":
            self.quadrature = lobatto
        else:
            raise NotImplementedError(
                f"Quadrature method '{quadrature}' is not implemented!"
            )

        # build simple mesh for each field
        self.meshes = []
        for kv, dim_q, dim_u, derivative_order, basis_function in zip(
            knot_vectors, dim_qs, dim_us, derivative_orders, basis_functions
        ):
            self.meshes.append(
                Mesh1D(
                    kv,
                    nquadrature,
                    dim_q,
                    derivative_order,
                    basis_function,
                    quadrature,
                    dim_u,
                )
            )

        # compute total number of generalized coordinates and velocities
        self.nq = sum([mesh.nq for mesh in self.meshes])
        self.nu = sum([mesh.nu for mesh in self.meshes])

        # compute number of generalized coordinates and velocities per elements
        self.nq_per_element = sum([mesh.nq_per_element for mesh in self.meshes])
        self.nu_per_element = sum([mesh.nu_per_element for mesh in self.meshes])

        # concatenate global Boolean connectivity matrix elDOF
        self.elDOF_q = np.zeros((self.nelement, self.nq_per_element), dtype=int)
        self.elDOF_u = np.zeros((self.nelement, self.nu_per_element), dtype=int)
        offset_q = 0
        offset_u = 0
        for el in range(self.nelement):
            for mesh in self.meshes:
                self.elDOF_q[
                    el, offset_q : offset_q + mesh.nq_per_element
                ] = mesh.elDOF  # TODO: We shoud rename this to elDOF_q as well!
                self.elDOF_u[
                    el, offset_u : offset_u + mesh.nu_per_element
                ] = mesh.elDOF_u

                offset_q += mesh.nq_per_element
                offset_u += mesh.nu_per_element

        # concatenate nodal Boolean connectivity matrix nodalDOF
        self.nodalDOF_element_q = []
        self.nodalDOF_element_u = []
        offset_q = 0
        offset_u = 0
        for mesh in self.meshes:
            self.nodalDOF_element_q.append(
                mesh.nodalDOF_element + offset_q
            )  # TODO: This should be named nodalDOF_q as well!
            self.nodalDOF_element_u.append(mesh.nodalDOF_element_u + offset_u)

            offset_q += mesh.nq_per_element
            offset_u += mesh.nu_per_element

        exit()

        # we might have different meshes for q and u, e.g. quaternions for
        # describing spatial rotations
        self.dim_qs = dim_qs  # number of q's per node of each field
        self.dim_us = dim_us  # number of u's per node of each field
        # self.dim_q = sum(dim_qs) # total number of q's per node
        # self.dim_u = sum(dim_us) # total number of u's per node

        self.nnodes_per_elements = []  # number of nodes influencing each element
        self.nbasis_elements = []  # number of basis function per element
        for basis in basis_functions:
            if basis == "Lagrange" or basis == "B-spline":
                self.nnodes_per_elements.append(self.degree + 1)
                self.nbasis_element.append(self.nnodes_per_element)
            elif basis == "Hermite":
                self.nnodes_per_elements.append(4)
                self.nbasis_element.append(4)
            else:
                raise NotImplementedError("")

        # total number of generalized coordinates and generalzed velocities per element
        self.nq_per_element = sum(
            [
                nnodes * dim_q
                for (nnodes, dim_q) in zip(self.nnodes_per_elements, dim_qs)
            ]
        )
        self.nu_per_element = sum(
            [
                nnodes * dim_u
                for (nnodes, dim_u) in zip(self.nnodes_per_elements, dim_us)
            ]
        )
        # self.nnodes_per_element = sum(self.nnodes_per_elements)

        # Boolean connectivity matrix for element polynomial_degrees of
        # freedom. This is used to extract the element degrees of freedom via
        # q[elDOF[el]] = q_e = C^e * q.
        self.elDOF = np.zeros((self.nelement, self.nq_per_element), dtype=int)
        self.elDOF_u = np.zeros((self.nelement, self.nu_per_element), dtype=int)

        exit()

        if basis == "Lagrange":
            # TODO: We want to allow for different shape functions for each field!
            exit()

            # total number of nodes
            self.nnodes_per_field = [
                degree * self.nelement + 1 for degree in self.degrees
            ]

            offset_q = sum(
                [
                    (nnodes - 1) * dim_q
                    for (nnodes, dim_q) in zip(self.nnodes_per_elements, dim_qs)
                ]
            )
            offset_u = sum(
                [
                    (nnodes - 1) * dim_u
                    for (nnodes, dim_u) in zip(self.nnodes_per_elements, dim_us)
                ]
            )

            elDOF_el = np.arange(self.nq_per_element)
            elDOF_el_u = np.arange(self.nu_per_element)
            for el in range(self.nelement):
                self.elDOF[el] = elDOF_el + el * offset_q
                self.elDOF_u[el] = elDOF_el_u + el * offset_u

            self.vtk_cell_type = "VTK_LAGRANGE_CURVE"
        elif basis == "Hermite":
            raise NotImplementedError
            # raise NotImplementedError("Adapt according to new ordering of q!")

            # total number of nodes
            self.nnodes = 2 * (self.nelement + 1)

            # ordering for a single node (needs to be shifted for each elements)
            elDOF_node = np.arange(self.nbasis_element * self.dim_q)
            elDOF_node_u = np.arange(self.nbasis_element * self.dim_u)
            for el in range(self.nelement):
                self.elDOF[el] = elDOF_node + el * 2 * self.dim_q
                self.elDOF_u[el] = elDOF_node_u + el * 2 * self.dim_u

            # TODO: Does VTK implement Hermite curves?
            # TODO: We should convert the Hermite spline into Bezier cells.
            self.vtk_cell_type = "VTK_LAGRANGE_CURVE"
        elif basis == "B-spline":
            raise NotImplementedError
            # total number of nodes
            self.nnodes = self.degree + self.nelement

            elDOF_el = np.arange(self.nq_per_element)
            elDOF_el_u = np.arange(self.nu_per_element)
            for el in range(self.nelement):
                self.elDOF[el] = elDOF_el + el * dim_q
                self.elDOF_u[el] = elDOF_el_u + el * dim_u

            self.vtk_cell_type = "VTK_BEZIER_CURVE"

        # total number of degrees of freedoms
        self.nq = sum(
            [nnodes * dim_q for (nnodes, dim_q) in zip(self.nnodes_per_field, dim_qs)]
        )
        self.nq = self.nnodes * dim_q
        self.nu = self.nnodes * dim_u

        # TODO: Move on here!
        exit()

        # construct tthe Boolean selection amtrix that choses the coordinates
        # of an individual node via q[nodalDOF[a]] = C^a * q
        self.nodalDOF = np.arange(self.nq).reshape(self.nnodes, dim_q)
        self.nodalDOF_element_u = np.arange(self.nu).reshape(self.nnodes, dim_u)

        # Boolean connectivity matrix for nodal polynomial_degrees of freedom
        # inside each element. This is only required if multiple fields are
        # discretized. It is used as qe[nodalDOF_element_[a]] = q_e^a = C^a * q_e
        self.nodalDOF_element = np.arange(self.nq_per_element).reshape(
            self.nnodes_per_element, dim_q
        )
        self.nodalDOF_element_u = np.arange(self.nu_per_element).reshape(
            self.nnodes_per_element, dim_u
        )

        # transform quadrature points on element intervals
        self.quadrature_points()

        # evaluate element shape functions at quadrature points
        self.shape_functions()

        # end_points degrees of freedom
        self.end_points()

    def basis1D(self, xis):
        if self.basis == "B-spline":
            return B_spline_basis1D(
                self.degree,
                self.derivative_order,
                self.knot_vector.data,
                xis,
                squeeze=False,
            )
        elif self.basis == "Lagrange":
            # return lagrange_basis1D(
            #     self.degree, xis, self.derivative_order, self.knot_vector
            # )
            return lagrange_basis1D(
                self.degree, xis, self.derivative_order, self.knot_vector
            )
        elif self.basis == "Hermite":
            return cubic_Hermite_basis_1D(xis, self.knot_vector, self.derivative_order)

    def eval_basis(self, xi):
        if self.basis == "B-spline":
            return B_spline_basis1D(
                self.degree, self.derivative_order, self.data, xi, squeeze=True
            )
        elif self.basis == "Lagrange":
            return lagrange_basis1D(
                self.degree, xi, self.derivative_order, self.knot_vector, squeeze=True
            )
        elif self.basis == "Hermite":
            return cubic_Hermite_basis_1D(xi, self.knot_vector, self.derivative_order)

    def quadrature_points(self):
        self.qp = np.zeros((self.nelement, self.nquadrature))
        self.wp = np.zeros((self.nelement, self.nquadrature))

        for el in range(self.nelement):
            Xi_element_interval = self.knot_vector.element_interval(el)
            self.qp[el], self.wp[el] = self.quadrature(
                self.nquadrature, interval=Xi_element_interval
            )

    def shape_functions(self):
        self.N = np.zeros((self.nelement, self.nquadrature, self.nbasis_element))
        if self.derivative_order > 0:
            self.N_xi = np.zeros((self.nelement, self.nquadrature, self.nbasis_element))
            if self.derivative_order > 1:
                self.N_xixi = np.zeros(
                    (self.nelement, self.nquadrature, self.nbasis_element)
                )

        for el in range(self.nelement):
            NN = self.basis1D(self.qp[el])
            # expression = "self.N"
            # for i in range(self.derivative_order):
            #     expression += ", self.N_" + (i + 1) * "xi"
            # eval(expression) = NN
            self.N[el] = NN[0]
            if self.derivative_order > 0:
                self.N_xi[el] = NN[1]
                if self.derivative_order > 1:
                    self.N_xixi[el] = NN[2]

    def end_points(self):
        def select_end_points(**kwargs):
            nn_0 = kwargs.get("nn_0", range(self.nnodes))

            end_points = []
            for i in nn_0:
                end_points.append(i)

            DOF_x = np.array(end_points)
            nn_edge = len(end_points)
            DOF = np.zeros((self.dim_q, nn_edge), dtype=int)
            for i in range(self.dim_q):
                DOF[i] = DOF_x + i * self.nnodes

            return DOF

        self.point_qDOF = (
            select_end_points(nn_0=[0]),
            select_end_points(nn_0=[self.nnodes - 1]),
        )

        # evaluate shape functions at the boundaries
        self.NN_bdry0 = self.eval_basis(0)
        self.NN_bdry1 = self.eval_basis(1)

    def reference_mappings(self, Q):
        """Compute inverse gradient from the reference configuration to the parameter space and scale quadrature points by its determinant. See Bonet 1997 (7.6a,b)"""
        if self.dim_q == 3:
            w_J0 = np.zeros((self.nelement, self.nquadrature))
            for el in range(self.nelement):
                Qe = Q[self.elDOF[el]]

                for i in range(self.nquadrature):
                    N_xi = self.N_xi[el, i]

                    kappa0_xi = np.zeros((self.dim_q, self.dim_q))
                    for a in range(self.nnodes_per_element):
                        kappa0_xi += (
                            Qe[self.nodalDOF_element[a]] * N_xi[a]
                        )  # Bonet 1997 (7.6b)

                    w_J0[el, i] = np.linalg.norm(kappa0_xi)

            return w_J0

    # functions for vtk export
    def ensure_L2_projection_A(self):
        if not hasattr(self, "A"):
            A = CooMatrix((self.nnodes, self.nnodes))
            for el in range(self.nelement):
                elDOF_el = self.elDOF[el, : self.nnodes_per_element]
                Ae = np.zeros((self.nnodes_per_element, self.nnodes_per_element))
                Nel = self.N[el]
                for a in range(self.nnodes_per_element):
                    for b in range(self.nnodes_per_element):
                        for i in range(self.nquadrature):
                            Ae[a, b] += Nel[i, a] * Nel[i, b]
                A.extend(Ae, (elDOF_el, elDOF_el))
            self.A = A.tocsc()

    def rhs_L2_projection(self, field):
        _, nqp, *shape = field.shape
        dim = np.prod(shape)

        b = np.zeros((self.nnodes, dim))
        for el in range(self.nelement):
            elDOF_el = self.elDOF[el, : self.nnodes_per_element]
            be = np.zeros((self.nnodes_per_element, dim))
            Nel = self.N[el]
            for a in range(self.nnodes_per_element):
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
        q = np.zeros((self.nnodes, dim))
        for i, bi in enumerate(b.T):
            q[:, i] = spsolve(self.A, bi)

        if self.basis == "B-spline":
            # rearrange q's from solver to Piegl's 1D ordering
            Pw = q.reshape(-1, dim, order="F")

            # decompose B-spline mesh in Bezier patches
            Qw = decompose_B_spline_curve(self.knot_vector, Pw)

        elif self.basis == "Lagrange":
            # rearrange q's from solver to Piegl's 3D ordering
            Qw = np.zeros((self.nelement, self.degree + 1, dim))
            for el in range(self.nelement):
                for a in range(self.nnodes_per_element):
                    Qw[el, a] = q[self.elDOF[el][self.nodalDOF_element[a][0]]]

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
        if self.basis == "B-spline":
            # rearrange q's from solver to Piegl's 1D ordering
            Pw = q.reshape(-1, self.dim_q, order="F")

            # decompose B-spline mesh in Bezier patches
            Qw = decompose_B_spline_curve(self.knot_vector, Pw)

        elif self.basis == "Lagrange":
            # rearrange q's from solver to Piegl's 1D ordering
            Qw = np.zeros((self.nelement, self.degree + 1, self.dim_q))
            for el in range(self.nelement):
                for a in range(self.nnodes_per_element):
                    Qw[el, a] = q[self.elDOF[el][self.nodalDOF_element[a]]]

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
    from cardillo.discretization.b_spline import BSplineKnotVector

    polynomial_degree = 3
    quadrature_points = polynomial_degree + 1
    elements = 4

    Xi = BSplineKnotVector(polynomial_degree, elements)

    mesh = Mesh1D(Xi, quadrature_points, derivative_order=1, basis="B-spline", dim_q=2)

    Q = line2D(5, mesh, Greville=True, Fuzz=0)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    max_val = np.max(np.abs(Q))
    ax.set_xlim(left=-max_val, right=max_val)
    ax.set_ylim(bottom=-max_val, top=max_val)

    ax.scatter(
        *Q[mesh.point_qDOF[0].reshape(-1)].reshape(2, -1), marker="x", color="green"
    )
    ax.scatter(
        *Q[mesh.point_qDOF[1].reshape(-1)].reshape(2, -1), marker="x", color="red"
    )

    plt.show()


if __name__ == "__main__":
    test_point_qDOF()
