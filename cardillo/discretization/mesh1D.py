import numpy as np
from scipy.sparse.linalg import spsolve
from cardillo.discretization.B_spline import (
    B_spline_basis1D,
    decompose_B_spline_curve,
    flat1D_vtk,
)
from cardillo.discretization.lagrange import lagrange_basis1D
from cardillo.discretization.Hermite import cubic_Hermite_basis_1D
from cardillo.discretization.gauss import gauss
from cardillo.utility.coo import Coo


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
        n_quadrature_points,
        nq_node,
        derivative_order=1,
        basis="B-spline",
    ):
        self.basis = basis
        self.nelement = knot_vector.nel
        self.knot_vector = knot_vector
        self.data = self.knot_vector.data
        self.degree = self.knot_vector.degree
        self.derivative_order = derivative_order
        self.n_quadrature_points = n_quadrature_points

        # TODO: Lagrange should be capitalized!
        if basis == "lagrange" or basis == "B-spline":
            self.nnodes_element = (
                self.degree + 1
            )  # number of nodes influencing each element
            self.nq_node = nq_node  # number of degrees of freedom per node
            self.nbasis_element = (
                self.nnodes_element
            )  # number of basis function per element
        elif basis == "Hermite":
            self.nnodes_element = 2  # number of nodes influencing each element
            self.nq_node = nq_node  # number of degrees of freedom per node
            # this assumes to be the sum of values and derivatives
            self.nbasis_element = 4  # number of basis function per element
        else:
            raise NotImplementedError("")
        self.nq_element = (
            self.nnodes_element * nq_node
        )  # total number of generalized coordinates per element

        # TODO: Lagrange should be capitalized!
        if basis == "lagrange":
            # total number of nodes
            self.nnodes = self.degree * self.nelement + 1

            # elDOF
            self.elDOF = np.zeros((self.nelement, self.nq_element), dtype=int)
            for el in range(self.nelement):
                for no in range(self.nnodes_element):
                    elDOF_x = self.degree * el + no
                    for dof in range(self.nq_node):
                        self.elDOF[el, no + self.nnodes_element * dof] = (
                            elDOF_x + self.nnodes * dof
                        )

            self.vtk_cell_type = "VTK_LAGRANGE_CURVE"
        elif basis == "Hermite":
            # total number of nodes
            self.nnodes = self.nelement + 1

            # elDOF
            self.elDOF = np.zeros((self.nelement, self.nq_element), dtype=int)
            # ordering for a single node (needs to be shifted for each elements)
            elDOF_node = np.arange(2 * self.nq_node)
            for el in range(self.nelement):
                self.elDOF[el] = elDOF_node + el * self.nq_node

            # TODO: Does VTK implement Hermite curves?
            self.vtk_cell_type = "VTK_LAGRANGE_CURVE"
        elif basis == "B-spline":
            # total number of nodes
            self.nnodes = self.degree + self.nelement

            # construct selection matrix elDOF assigning to each element its DOFs of the displacement
            # q[elDOF[el]] is equivalent to q_e = C^e * q
            self.elDOF = np.zeros((self.nelement, self.nq_element), dtype=int)
            for el in range(self.nelement):
                for no in range(self.nnodes_element):
                    elDOF_x = el + no
                    for dof in range(self.nq_node):
                        self.elDOF[el, no + self.nnodes_element * dof] = (
                            elDOF_x + self.nnodes * dof
                        )

            self.vtk_cell_type = "VTK_BEZIER_CURVE"

        # construct selection matrix ndDOF assining to each node its DOFs
        # qe[ndDOF[a]] is equivalent to q_e^a = C^a * q^e
        self.nodalDOF = np.zeros((self.nnodes_element, self.nq_node), dtype=int)
        for dof in range(self.nq_node):
            self.nodalDOF[:, dof] = (
                np.arange(self.nnodes_element) + dof * self.nnodes_element
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
        elif self.basis == "lagrange":
            return lagrange_basis1D(
                self.degree, xis, self.derivative_order, self.knot_vector
            )
        elif self.basis == "Hermite":
            return cubic_Hermite_basis_1D(
                xis, self.knot_vector, 1, self.derivative_order
            )

    def eval_basis(self, xi):
        if self.basis == "B-spline":
            return B_spline_basis1D(
                self.degree, self.derivative_order, self.data, xi, squeeze=True
            )
        elif self.basis == "lagrange":
            return lagrange_basis1D(
                self.degree, xi, self.derivative_order, self.knot_vector, squeeze=True
            )
        elif self.basis == "Hermite":
            return cubic_Hermite_basis_1D(
                xi, self.knot_vector, 1, self.derivative_order
            )

    def quadrature_points(self):
        self.qp = np.zeros((self.nelement, self.n_quadrature_points))
        self.wp = np.zeros((self.nelement, self.n_quadrature_points))

        for el in range(self.nelement):
            Xi_element_interval = self.knot_vector.element_interval(el)
            self.qp[el], self.wp[el] = gauss(
                self.n_quadrature_points, interval=Xi_element_interval
            )

    def shape_functions(self):
        self.N = np.zeros(
            (self.nelement, self.n_quadrature_points, self.nbasis_element)
        )
        if self.derivative_order > 0:
            self.N_xi = np.zeros(
                (self.nelement, self.n_quadrature_points, self.nbasis_element)
            )
            if self.derivative_order > 1:
                self.N_xixi = np.zeros(
                    (self.nelement, self.n_quadrature_points, self.nbasis_element)
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
            DOF = np.zeros((self.nq_node, nn_edge), dtype=int)
            for i in range(self.nq_node):
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
        if self.nq_node == 3:
            w_J0 = np.zeros((self.nelement, self.n_quadrature_points))
            for el in range(self.nelement):
                Qe = Q[self.elDOF[el]]

                for i in range(self.n_quadrature_points):
                    N_xi = self.N_xi[el, i]

                    kappa0_xi = np.zeros((self.nq_node, self.nq_node))
                    for a in range(self.nnodes_element):
                        kappa0_xi += Qe[self.nodalDOF[a]] * N_xi[a]  # Bonet 1997 (7.6b)

                    w_J0[el, i] = np.linalg.norm(kappa0_xi)

            return w_J0

    # functions for vtk export
    def ensure_L2_projection_A(self):
        if not hasattr(self, "A"):
            A = Coo((self.nnodes, self.nnodes))
            for el in range(self.nelement):
                elDOF_el = self.elDOF[el, : self.nnodes_element]
                Ae = np.zeros((self.nnodes_element, self.nnodes_element))
                Nel = self.N[el]
                for a in range(self.nnodes_element):
                    for b in range(self.nnodes_element):
                        for i in range(self.n_quadrature_points):
                            Ae[a, b] += Nel[i, a] * Nel[i, b]
                A.extend(Ae, (elDOF_el, elDOF_el))
            self.A = A.tocsc()

    def rhs_L2_projection(self, field):
        _, nqp, *shape = field.shape
        dim = np.prod(shape)

        b = np.zeros((self.nnodes, dim))
        for el in range(self.nelement):
            elDOF_el = self.elDOF[el, : self.nnodes_element]
            be = np.zeros((self.nnodes_element, dim))
            Nel = self.N[el]
            for a in range(self.nnodes_element):
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

        elif self.basis == "lagrange":
            # rearrange q's from solver to Piegl's 3D ordering
            Qw = np.zeros((self.nelement, self.degree + 1, dim))
            for el in range(self.nelement):
                for a in range(self.nnodes_element):
                    Qw[el, a] = q[self.elDOF[el][self.nodalDOF[a][0]]]

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
            Pw = q.reshape(-1, self.nq_node, order="F")

            # decompose B-spline mesh in Bezier patches
            Qw = decompose_B_spline_curve(self.knot_vector, Pw)

        elif self.basis == "lagrange":
            # rearrange q's from solver to Piegl's 1D ordering
            Qw = np.zeros((self.nelement, self.degree + 1, self.nq_node))
            for el in range(self.nelement):
                for a in range(self.nnodes_element):
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
    from cardillo.discretization.B_spline import KnotVector

    polynomial_degree = 3
    quadrature_points = polynomial_degree + 1
    elements = 4

    Xi = KnotVector(polynomial_degree, elements)

    mesh = Mesh1D(
        Xi, quadrature_points, derivative_order=1, basis="B-spline", nq_node=2
    )

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
