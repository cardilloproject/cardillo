import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import BPoly


class HermiteNodeVector:
    def __init__(self, degree, nel, data=None):
        assert degree % 2 == 1, "degree must be odd"
        self.degree = degree
        self.nel = nel
        if data is None:
            self.data = HermiteNodeVector.uniform(nel)
        else:
            raise NotImplementedError("")
            # TODO: What happens here?
            self.data = np.zeros(self.nel + 1)
            for el in range(nel):
                for p in range(self.degree):
                    self.data[el * self.degree + p] = (
                        data[el] + p * (data[el + 1] - data[el]) / self.degree
                    )
            self.data[-1] = data[-1]

        self.element_data = self.data
        self.verify_data()

    @staticmethod
    def uniform(nel, interval=[0, 1]):
        return np.linspace(interval[0], interval[1], num=nel + 1)

    def element_number(self, nodes):
        nodes = np.atleast_1d(nodes)
        lenxi = len(nodes)

        element = np.zeros(lenxi, dtype=int)
        for j in range(lenxi):
            element[j] = np.asarray(self.element_data <= nodes[j]).nonzero()[0][-1]
            if nodes[j] == self.data[-1]:
                element[j] -= 1
        return element

    def element_interval(self, el):
        return np.array(
            [
                self.element_data[el],
                self.element_data[el + 1],
            ]
        )

    def verify_data(self):
        assert len(self.element_data) == self.nel + 1


class CubicHermiteBasis:
    def __init__(self, dim=3, window=[0, 1]):
        self.dim = dim

        # basis function, see https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Representations
        self.h00 = Polynomial([1, 0, -3, 2], domain=[0, 1], window=window)
        self.h01 = Polynomial([0, 1, -2, 1], domain=[0, 1], window=window)
        self.h10 = Polynomial([0, 0, 3, -2], domain=[0, 1], window=window)
        self.h11 = Polynomial([0, 0, -1, 1], domain=[0, 1], window=window)

    def __call__(self, xis):
        xis = np.atleast_1d(xis)
        values = np.zeros((len(xis), self.dim, 4 * self.dim), dtype=float)
        for i, xii in enumerate(xis):
            values[i] = np.hstack(
                [
                    self.h00(xii) * np.eye(self.dim),
                    self.h01(xii) * np.eye(self.dim),
                    self.h10(xii) * np.eye(self.dim),
                    self.h11(xii) * np.eye(self.dim),
                ]
            )
        return values.squeeze()

    def deriv(self, xis, n=1):
        xis = np.atleast_1d(xis)
        values = np.zeros((len(xis), self.dim, 4 * self.dim), dtype=float)
        for i, xii in enumerate(xis):
            values[i] = np.hstack(
                [
                    self.h00.deriv(n)(xii) * np.eye(self.dim),
                    self.h01.deriv(n)(xii) * np.eye(self.dim),
                    self.h10.deriv(n)(xii) * np.eye(self.dim),
                    self.h11.deriv(n)(xii) * np.eye(self.dim),
                ]
            )
        return values.squeeze()


def cubic_Hermite_basis_1D(xis, node_vector, dim=1, derivative=1, squeeze=True):
    """Compute cubic Hermite basis functions for a given knot vector."""
    xis = np.atleast_1d(xis)
    nxis = len(xis)
    N = np.zeros((derivative + 1, nxis, dim, 4 * dim))
    for i, xi in enumerate(xis):
        el = node_vector.element_number(xi)[0]
        basis = CubicHermiteBasis(dim, window=node_vector.element_interval(el))
        N[0, i] = basis(xi)
        if derivative:
            for j in range(1, derivative + 1):
                N[j, i] = basis.deriv(xi, n=j)
    if squeeze:
        return N.squeeze()
    else:
        return N


def basic_usage():
    # define generalized coordinates of cubic hermite spline
    from cardillo.math import e1, e2, e3, norm, cross3

    # ######################
    # # pathologic line case
    # ######################
    # r0 = np.array([0, 0.5, 0.5])
    # t0 = e1
    # n0 = e2
    # b0 = cross3(t0, n0) / norm(cross3(t0, n0))
    # r1 = np.array([1, 0.5, 0.5])
    # t1 = e1
    # n1 = e2
    # # n1 = e3
    # b1 = cross3(t1, n1) / norm(cross3(t1, n1))

    ################
    # curved 3D case
    ################
    r0 = np.zeros(3)
    t0 = e1 * 1.5
    r1 = np.array([1, 1, 1]) / np.sqrt(3)
    t1 = e2 * 0.5

    ###############################
    # build generalized coordinates
    ###############################
    q = np.concatenate([r0, t0, r1, t1])

    #####################
    # build Hermite basis
    #####################
    hermite = CubicHermiteBasis()
    r_poly = lambda xi: hermite(xi) @ q
    r_xi_poly = lambda xi: hermite.deriv(xi, n=1) @ q
    r_xixi_poly = lambda xi: hermite.deriv(xi, n=2) @ q

    ###############
    # visualization
    ###############
    import matplotlib.pyplot as plt

    num = 10
    xis = np.linspace(0, 1, num=num)
    r = r_poly(xis)
    r_xi = r_xi_poly(xis)
    r_xixi = r_xixi_poly(xis)

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.plot(r[:, 0], r[:, 1], r[:, 2], "-k")
    for i in range(len(r)):
        ax.quiver3D(*r[i], *r_xi[i], color="r", length=0.1)
        ax.quiver3D(*r[i], *r_xixi[i], color="g", length=0.1)
    ax.quiver3D(*r0, *t0, color="b")
    ax.quiver3D(*r1, *t1, color="b")
    ax.grid()
    plt.show()


def knotvector_usage():
    nel = 2
    knot_vector = HermiteNodeVector(3, nel)
    num = 10
    xis = np.linspace(0, 1, num=num)
    N, N_xi = cubic_Hermite_basis_1D(xis, knot_vector, dim=1)


if __name__ == "__main__":
    # basic_usage()
    knotvector_usage()
