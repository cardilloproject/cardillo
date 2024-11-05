import numpy as np
from numpy.polynomial import Polynomial
from scipy.sparse.linalg import spsolve


# data are only the corner nodes
class LagrangeKnotVector:
    def __init__(self, degree, nel, data=None):
        self.degree = degree
        self.nel = nel
        if data is None:
            self.data = LagrangeKnotVector.uniform(self.degree, self.nel)
        else:
            self.data = np.zeros(self.nel * self.degree + 1)
            for el in range(nel):
                for p in range(self.degree):
                    self.data[el * self.degree + p] = (
                        data[el] + p * (data[el + 1] - data[el]) / self.degree
                    )
            self.data[-1] = data[-1]

        self.element_data = self.data
        self.verify_data()

    @staticmethod
    def uniform(degree, nel, interval=[0, 1]):
        return np.linspace(interval[0], interval[1], nel + 1)

    def element_number(self, nodes):
        if not hasattr(nodes, "__len__"):
            nodes = [nodes]
        lenxi = len(nodes)

        assert np.all(np.isfinite(nodes))

        element = np.zeros(lenxi, dtype=int)
        for j in range(lenxi):
            element[j] = np.asarray(self.element_data <= nodes[j]).nonzero()[0][-1]
            if nodes[j] == self.data[-1]:
                element[j] -= 1
        # if lenxi == 1:
        #     return int(element)
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


class LagrangeBasis:
    """Lagrange basis function on [interval[0], interval[1]], see Wiki.

    References:
    -----------
    Wiki: https://en.wikipedia.org/wiki/Lagrange_polynomial#Definition"""

    def __init__(self, degree, interval=[0, 1]):
        self.degree = degree

        # compute equally spaced points on [0, 1]
        nus = np.linspace(0, 1, num=degree + 1)

        # recursively construct Lagrange shape functions on [0, 1] and map
        # them on interval
        self.li = np.ones(degree + 1, dtype=object)
        for i in range(degree + 1):
            self.li[i] = Polynomial([1.0], domain=[0, 1], window=[0, 1])
            for j in range(degree + 1):
                if i != j:
                    diff = nus[i] - nus[j]
                    self.li[i] *= Polynomial(
                        [-nus[j] / diff, 1.0 / diff], domain=[0, 1], window=[0, 1]
                    )
        self.set_interval(interval)

    def set_interval(self, interval):
        self.__interval = interval
        self.__interval_len = interval[1] - interval[0]

    def __normalize_xi(self, xi):
        if xi == self.__interval[0]:
            return 0
        elif xi == self.__interval[1]:
            return 1
        else:
            return (xi - self.__interval[0]) / self.__interval_len

    def __call__(self, xis):
        xis = np.atleast_1d(xis)
        values = np.zeros((len(xis), self.degree + 1), dtype=float)
        for i, xii in enumerate(xis):
            for j in range(self.degree + 1):
                values[i, j] = self.li[j](self.__normalize_xi(xii))
        return values

    def deriv(self, xis, n=1):
        xis = np.atleast_1d(xis)
        values = np.zeros((len(xis), self.degree + 1), dtype=float)
        for i, xii in enumerate(xis):
            for j in range(self.degree + 1):
                values[i, j] = (
                    self.li[j].deriv(n)(self.__normalize_xi(xii)) / self.__interval_len
                )
        return values


def lagrange_basis1D(degree, xis, derivative, knot_vector, squeeze=True):
    """Compute Lagrange basis functions for a given node vector."""
    xis = np.atleast_1d(xis)
    nxis = len(xis)
    N = np.zeros((derivative + 1, nxis, degree + 1))
    basis = LagrangeBasis(degree)
    for i, xi in enumerate(xis):
        el = knot_vector.element_number(xi)[0]
        interval = knot_vector.element_interval(el)
        basis.set_interval(interval)
        N[0, i] = basis(xi)
        if derivative:
            for j in range(1, derivative + 1):
                N[j, i] = basis.deriv(xi, n=j)
    if squeeze:
        N = N.squeeze()
    return N


def find_element_number(mesh, xis):
    # finds the element number for a xis vector from the 0 to 1 parameter space
    # also gives the parameter space value of xis
    el = np.zeros(len(xis), dtype=int)
    xis_l = np.zeros_like(xis)
    for i, xi in enumerate(xis):
        if int((xi // (1 / mesh.element_shape[i]))) >= mesh.element_shape[i]:
            el[i] = mesh.element_shape[i] - 1
            xis_l[i] = 1
        else:
            el[i] = int((xi // (1 / mesh.element_shape[i])))
            xis_l[i] = (xis[i] * mesh.element_shape[i] - el[i]) * 2 - 1
    return el, xis_l


def test_shape_functions_der():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    x = np.linspace(-1, 1, 10)
    y = x
    z = x
    NN = lagrange_basis1D(2, x, derivative=1)

    plt.plot(x, NN[:10, 0, 0])
    plt.plot(x, NN[:10, 1, 0])
    plt.plot(x, NN[:10, 2, 0])
    plt.plot(x, NN[:10, 0, 1])
    plt.plot(x, NN[:10, 1, 1])
    plt.plot(x, NN[:10, 2, 1])
    plt.show()


if __name__ == "__main__":
    import numpy as np

    # test_shape_functions_der()

    # test_fit_lagrange_curve()

    # test_fit_lagrange_volume()

    # degree = 2
    # # x = -1
    # # x = 0
    # x = 1

    # lx = __lagrange(x, degree)
    # print(f'l({x}): {lx}')

    # lx_x = __lagrange_x(x, degree)
    # print(f'l_x({x}): {lx_x}')

    # lx_x = __lagrange_x_r(x, degree)
    # print(f'l_x({x}): {lx_x}')

    # lx_xx = __lagrange_xx_r(x, degree)
    # print(f'l_xx({x}): {lx_xx}')

    # degree = 1
    # derivative_order = 1
    # x_array = np.array([-1, 0, 1])
    # N, dN = Lagrange_basis(degree, x_array)
    # print(f'N({x_array}):\n{N}')
    # print(f'dN({x_array}):\n{dN}')
