import numpy as np
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
from .lagrange import LagrangeBasis
from .gauss import gauss, lobatto


class Mesh1D:
    def __init__(
        self,
        knot_vector,
        nquadrature,
        dim_q,
        derivative_order=1,
        basis="Lagrange",
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

        self.lagrangebasis = LagrangeBasis(self.degree)
        self._eval_basis_cache = LRUCache(self.nelement * self.degree + 1)

        # we might have different meshes for q and u, e.g. quaternions for
        # describing spatial rotations
        if dim_u is None:
            self.dim_u = dim_u = dim_q

        if basis in ["Lagrange", "Lagrange_Disc"]:
            self.nnodes_per_element = (
                self.degree + 1
            )  # number of nodes influencing each element
            self.dim_q = dim_q  # number of degrees of freedom per node
            self.nbasis_element = (
                self.nnodes_per_element
            )  # number of basis function per element
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

            self.vtk_cell_type = "VTB_LAGRANGE_CURVE"
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

            # TODO: check if VTK export works
            self.vtk_cell_type = "VTB_LAGRANGE_CURVE"

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
        self.quadrature_points()

        # evaluate element shape functions at quadrature points
        self.shape_functions()

    def basis1D(self, xis):
        if self.basis == "Lagrange":
            return self.lagrange_basis1D(
                xis,
                squeeze=False,
            )
        elif self.basis == "Lagrange_Disc":
            return self.lagrange_basis1D(
                xis,
                squeeze=False,
            )

    def lagrange_basis1D(self, xis, squeeze=True):
        xis = np.atleast_1d(xis)
        nxis = len(xis)
        N = np.zeros((self.derivative_order + 1, nxis, self.degree + 1))
        for i, xi in enumerate(xis):
            el = self.knot_vector.element_number(xi)[0]
            interval = self.knot_vector.element_interval(el)
            self.lagrangebasis.set_interval(interval)
            N[0, i] = self.lagrangebasis(xi)
            if self.derivative_order:
                for j in range(1, self.derivative_order + 1):
                    N[j, i] = self.lagrangebasis.deriv(xi, n=j)
        if squeeze:
            return N.squeeze()
        else:
            return N

    @cachedmethod(
        lambda self: self._eval_basis_cache,
        key=lambda self, xi: hashkey(xi),
    )
    def eval_basis(self, xi):
        if self.basis == "Lagrange":
            return self.lagrange_basis1D(xi, squeeze=True)
        elif self.basis == "Lagrange_Disc":
            return self.lagrange_basis1D(xi, squeeze=False)

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


if __name__ == "__main__":
    print("hello world")
