import numpy as np
from numpy.polynomial import Polynomial


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
    def __init__(self, interval=[0, 1]):
        # Cubic Hermite basis function on [0, 1], see
        # https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Representations.
        # Note: We have to perform a change of coordinates as described in
        # https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Interpolation_on_an_arbitrary_interval.
        # But be aware that numpy's Polynomial class changes only the domain,
        # but does not scale the tangents so we have to do this manually!
        interval_length = interval[1] - interval[0]
        self.h00 = Polynomial([1, 0, -3, 2], domain=interval, window=[0, 1])
        self.h01 = interval_length * Polynomial(
            [0, 1, -2, 1], domain=interval, window=[0, 1]
        )
        self.h10 = Polynomial([0, 0, 3, -2], domain=interval, window=[0, 1])
        self.h11 = interval_length * Polynomial(
            [0, 0, -1, 1], domain=interval, window=[0, 1]
        )

    def __call__(self, xis):
        xis = np.atleast_1d(xis)
        values = np.zeros((len(xis), 4), dtype=float)
        for i, xii in enumerate(xis):
            values[i, 0] = self.h00(xii)
            values[i, 1] = self.h01(xii)
            values[i, 2] = self.h10(xii)
            values[i, 3] = self.h11(xii)
        return values

    def deriv(self, xis, n=1):
        xis = np.atleast_1d(xis)
        values = np.zeros((len(xis), 4), dtype=float)
        for i, xii in enumerate(xis):
            values[i, 0] = self.h00.deriv(n)(xii)
            values[i, 1] = self.h01.deriv(n)(xii)
            values[i, 2] = self.h10.deriv(n)(xii)
            values[i, 3] = self.h11.deriv(n)(xii)
        return values


class QuinticHermiteBasis:
    def __init__(self, interval=[0, 1]):
        # Quintic Hermite basis function on [0, 1], see
        # https://www.rose-hulman.edu/~finn/CCLI/Notes/day09.pdf.
        # Note: We have to perform a change of coordinates as described in
        # https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Interpolation_on_an_arbitrary_interval.
        # But be aware that numpy's Polynomial class changes only the domain,
        # but does not scale the tangents so we have to do this manually!
        interval_length = interval[1] - interval[0]

        self.h00 = Polynomial([1, 0, 0, -10, 15, -6], domain=interval, window=[0, 1])
        self.h01 = interval_length * Polynomial(
            [0, 1, 0, -6, 8, -3], domain=interval, window=[0, 1]
        )
        self.h02 = interval_length**2 * Polynomial(
            [0, 0, 0.5, -1.5, 1.5, -0.5], domain=interval, window=[0, 1]
        )

        self.h10 = Polynomial([0, 0, 0, 10, -15, 6], domain=interval, window=[0, 1])
        self.h11 = interval_length * Polynomial(
            [0, 0, 0, -4, 7, -3], domain=interval, window=[0, 1]
        )
        self.h12 = interval_length**2 * Polynomial(
            [0, 0, 0, 0.5, -1, 0.5], domain=interval, window=[0, 1]
        )

    def __call__(self, xis):
        xis = np.atleast_1d(xis)
        values = np.zeros((len(xis), 6), dtype=float)
        for i, xii in enumerate(xis):
            values[i, 0] = self.h00(xii)
            values[i, 1] = self.h01(xii)
            values[i, 2] = self.h02(xii)
            values[i, 3] = self.h10(xii)
            values[i, 4] = self.h11(xii)
            values[i, 5] = self.h12(xii)
        return values

    def deriv(self, xis, n=1):
        xis = np.atleast_1d(xis)
        values = np.zeros((len(xis), 6), dtype=float)
        for i, xii in enumerate(xis):
            values[i, 0] = self.h00.deriv(n)(xii)
            values[i, 1] = self.h01.deriv(n)(xii)
            values[i, 2] = self.h02.deriv(n)(xii)
            values[i, 3] = self.h10.deriv(n)(xii)
            values[i, 4] = self.h11.deriv(n)(xii)
            values[i, 5] = self.h12.deriv(n)(xii)
        return values


def cubic_Hermite_basis_1D(xis, node_vector, derivative=1, squeeze=True):
    """Compute cubic Hermite basis functions for a given knot vector."""
    xis = np.atleast_1d(xis)
    nxis = len(xis)
    N = np.zeros((derivative + 1, nxis, 4))
    for i, xi in enumerate(xis):
        el = node_vector.element_number(xi)[0]
        basis = CubicHermiteBasis(interval=node_vector.element_interval(el))
        N[0, i] = basis(xi)
        if derivative:
            for j in range(1, derivative + 1):
                N[j, i] = basis.deriv(xi, n=j)
    if squeeze:
        N = N.squeeze()
    return N


def basic_usage_cubic():
    from cardillo.math import e1, e2

    # ######################
    # # pathologic line case
    # ######################
    # L = 2 * np.pi
    # r0 = np.array([0, 0.5, 0.5])
    # t0 = L * e1
    # r1 = np.array([L, 0.5, 0.5])
    # t1 = L * e1

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

    def r_poly(xi):
        q_nodes = q.reshape(4, -1, order="C")
        N = hermite(xi)
        r = np.zeros((len(xi), 3))
        for i in range(len(xi)):
            for j in range(4):
                r[i] += N[i, j] * q_nodes[j]
        return r

    def r_xi_poly(xi):
        q_nodes = q.reshape(4, -1, order="C")
        N_xi = hermite.deriv(xi, n=1)
        r_xi = np.zeros((len(xi), 3))
        for i in range(len(xi)):
            for j in range(4):
                r_xi[i] += N_xi[i, j] * q_nodes[j]
        return r_xi

    def r_xixi_poly(xi):
        q_nodes = q.reshape(4, -1, order="C")
        N_xixi = hermite.deriv(xi, n=2)
        r_xixi = np.zeros((len(xi), 3))
        for i in range(len(xi)):
            for j in range(4):
                r_xixi[i] += N_xixi[i, j] * q_nodes[j]
        return r_xixi

    ###############
    # visualization
    ###############
    import matplotlib.pyplot as plt

    num = 10
    xis = np.linspace(0, 1, num=num)
    r = r_poly(xis)
    r_xi = r_xi_poly(xis)
    r_xixi = r_xixi_poly(xis)

    print(f"r:\n{r}")
    print(f"r_xi:\n{r_xi}")

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
    from cardillo.beams import DirectorAxisAngle

    polynomial_degree = 3
    nelement = 2
    L = 2 * np.pi
    q0 = DirectorAxisAngle.straight_configuration(
        polynomial_degree, 1, nelement, L, basis="Hermite"
    )[: (nelement + 1) * 6]

    # build node vector for Hermite basis
    knot_vector = HermiteNodeVector(polynomial_degree, nelement)

    # evaluate basis functions
    num = 4
    xis = np.linspace(0, 1, num=num)
    N, N_xi = cubic_Hermite_basis_1D(xis, knot_vector)

    r = np.zeros((num, 3))
    r_xi = np.zeros((num, 3))
    q0_nodal = q0.reshape(-1, 3, order="C")
    # for el in range(nelement):
    for i, xi in enumerate(xis):
        el = knot_vector.element_number(xi)[0]
        print(f"xi: {xi}")
        print(f"el: {el}")
        for node in range(4):
            print(f"node: {node}")

            idx = 2 * el + node
            q_node = q0_nodal[idx]
            print(f"q_node: {q_node}")

            print(f"N: {N[i, node]}")
            print(f"N_xi: {N_xi[i, node]}")

            r[i] += N[i, node] * q_node
            r_xi[i] += N_xi[i, node] * q_node

    print(f"r:\n{r}")
    print(f"r_xi:\n{r_xi}")


def poly_test():
    # domain = [0, 1]
    # window = [1, 2]
    domain = [1, 2]
    window = [0, 1]
    p0 = Polynomial([1, -1], domain=domain, window=window)
    p1 = Polynomial([0, 1], domain=domain, window=window)
    num = 5
    xis = np.linspace(0, 1, num=num)
    print(f"xis: {xis}")
    print(f"p0(xi): {p0(xis)}")
    print(f"p1(xi): {p1(xis)}")


def basic_usage_quintic():
    from cardillo.math import e1, e2, e3

    ######################
    # pathologic line case
    ######################
    L = 2 * np.pi
    r0 = np.zeros(3)
    t0 = e1
    n0 = e2
    # n0 = e1
    # n0 = np.zeros(3)
    r1 = L * e1
    t1 = e1
    n1 = e2
    # n1 = e1
    # n1 = np.zeros(3)

    # ################
    # # curved 3D case
    # ################
    # r0 = np.zeros(3)
    # t0 = e1
    # n0 = e2
    # r1 = np.array([1, 1, 1]) / np.sqrt(3)
    # t1 = e2
    # n1 = -e1

    ###############################
    # build generalized coordinates
    ###############################
    q = np.concatenate([r0, t0, n0, r1, t1, n1])

    #####################
    # build Hermite basis
    #####################
    hermite = QuinticHermiteBasis()

    def r_poly(xi):
        N = hermite(xi)
        n = N.shape[1]
        q_nodes = q.reshape(n, -1, order="C")
        r = np.zeros((len(xi), 3))
        for i in range(len(xi)):
            for j in range(n):
                r[i] += N[i, j] * q_nodes[j]
        return r

    def r_xi_poly(xi):
        N_xi = hermite.deriv(xi, n=1)
        n = N_xi.shape[1]
        q_nodes = q.reshape(n, -1, order="C")
        r_xi = np.zeros((len(xi), 3))
        for i in range(len(xi)):
            for j in range(n):
                r_xi[i] += N_xi[i, j] * q_nodes[j]
        return r_xi

    def r_xixi_poly(xi):
        N_xixi = hermite.deriv(xi, n=2)
        n = N_xixi.shape[1]
        q_nodes = q.reshape(n, -1, order="C")
        r_xixi = np.zeros((len(xi), 3))
        for i in range(len(xi)):
            for j in range(n):
                r_xixi[i] += N_xixi[i, j] * q_nodes[j]
        return r_xixi

    ###############
    # visualization
    ###############
    import matplotlib.pyplot as plt

    num = 20
    xis = np.linspace(0, 1, num=num)
    r = r_poly(xis)
    r_xi = r_xi_poly(xis)
    r_xixi = r_xixi_poly(xis)
    d1 = r_xi / np.linalg.norm(r_xi, axis=1)[:, np.newaxis]
    d2 = r_xixi / np.linalg.norm(r_xixi, axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.plot(r[:, 0], r[:, 1], r[:, 2], "-k")
    for i in range(len(r)):
        # ax.quiver3D(*r[i], *r_xi[i], color="r", length=0.1)
        # ax.quiver3D(*r[i], *r_xixi[i], color="g", length=0.1)
        ax.quiver3D(*r[i], *d1[i], color="r", length=0.1)
        ax.quiver3D(*r[i], *d2[i], color="g", length=0.1)
    ax.quiver3D(*r0, *t0, color="b")
    ax.quiver3D(*r1, *t1, color="b")
    ax.grid()
    plt.show()


if __name__ == "__main__":
    # basic_usage_cubic()
    # knotvector_usage()
    # poly_test()
    basic_usage_quintic()
