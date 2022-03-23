import numpy as np
from numpy.core.fromnumeric import squeeze
from scipy.sparse.linalg import spsolve
from cardillo.discretization.indexing import flat2D, flat3D, split2D, split3D
import meshio


# data are only the corner nodes
class CubicHermiteNodeVector:
    def __init__(self, degree, nel, data=None):
        self.degree = degree
        self.nel = nel
        if data is None:
            self.data = CubicHermiteNodeVector.uniform(degree, nel)
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
    def uniform(nel, interval=[0, 1]):
        """TODO: Is this the correct number of nodes?
        I think yes, since we have two node elements for cubic hermite
        shapefunctons."""
        return np.linspace(interval[0], interval[1], num=nel + 1)

    def element_number(self, nodes):
        nodes = np.atleast_1d(nodes)
        lenxi = len(nodes)

        element = np.zeros(lenxi, dtype=int)
        for j in range(lenxi):
            # TODO: Modulo degree should not be neccesary here!
            element[j] = np.asarray(self.element_data <= nodes[j]).nonzero()[0][
                -1
            ]  # // (self.degree)
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


def cubic_Hermite_basis_1D(
    xi, derivative=1, node_vector=None, interval=[-1, 1], squeeze=False
):
    xi = np.atleast_1d(xi)

    # return NN
    if squeeze:
        return Lagrange_basis(
            p, xi, derivative=derivative, knot_vector=node_vector, interval=interval
        ).squeeze()
    else:
        return Lagrange_basis(
            p, xi, derivative=derivative, knot_vector=node_vector, interval=interval
        )


def CubicHermiteBasis(x, derivative=1, knot_vector=None, interval=[-1, 1]):
    """Compute cubic hermite basis functions.

    Parameters
    ----------
    x : ndarray, 1D
        array containing the evaluation points of the polynomial
    derivative : int
        whether to compute the derivative of the shape function or not
    returns : ndarray or (ndarray, ndarray)
        2D array of shape (len(x), degree + 1) containing the k = degree + 1 shape functions evaluated at x and optional the array containing the corresponding first derivatives

    """
    if not hasattr(x, "__len__"):
        x = [x]
    nx = len(x)
    N = np.zeros((derivative + 1, nx, degree + 1))
    if knot_vector is not None:
        for i, xi in enumerate(x):
            el = knot_vector.element_number(xi)[0]
            N[0, i] = __lagrange(xi, degree, interval=knot_vector.element_interval(el))
        if derivative:
            for i, xi in enumerate(x):
                el = knot_vector.element_number(xi)[0]
                N[1, i] = __lagrange_x(
                    xi, degree, interval=knot_vector.element_interval(el)
                )
        return N
    else:
        for i, xi in enumerate(x):
            N[0, i] = __lagrange(xi, degree, interval=interval)
        if derivative:
            for i, xi in enumerate(x):
                N[1, i] = __lagrange_x(xi, degree, interval=interval)
        return N


def __lagrange(x, degree, skip=[], interval=[-1, 1]):
    """1D Lagrange shape functions, see https://en.wikipedia.org/wiki/Lagrange_polynomial#Definition.

    Parameter
    ---------
    x : float
        evaluation point
    degree : int
        polynomial degree
    returns : ndarray, 1D
        array containing the k = degree + 1 shape functions evaluated at x
    """
    k = degree + 1
    xi = np.linspace(interval[0], interval[1], num=k)
    l = np.ones(k)
    for j in range(k):
        for m in range(k):
            if m == j or m in skip:
                continue
            l[j] *= (x - xi[m]) / (xi[j] - xi[m])

    return l


def __lagrange_x(x, degree, interval=[-1, 1]):
    """First derivative of 1D Lagrange shape functions, see https://en.wikipedia.org/wiki/Lagrange_polynomial#Derivatives.

    Parameter
    ---------
    x : float
        evaluation point
    degree : int
        polynomial degree
    returns : ndarray, 1D
        array containing the first derivative of the k = degree + 1 shape functions evaluated at x
    """
    k = degree + 1
    xi = np.linspace(interval[0], interval[1], num=k)
    l_x = np.zeros(k)
    for j in range(k):
        for i in range(k):
            if i == j:
                continue
            prod = 1
            for m in range(k):
                if m == i or m == j:
                    continue
                prod *= (x - xi[m]) / (xi[j] - xi[m])
            l_x[j] += prod / (xi[j] - xi[i])

    return l_x


def __lagrange_x_r(x, degree, skip=[]):
    """Recursive formular for first derivative of Lagrange shape functions."""
    k = degree + 1
    xi = np.linspace(-1, 1, num=k)
    l_x = np.zeros(k)
    for j in range(k):
        for i in range(k):
            if i == j or i in skip:
                continue
            l = __lagrange(x, degree, skip=[i] + skip)
            l_x[j] += l[j] / (xi[j] - xi[i])

    return l_x


def __lagrange_xx_r(x, degree):
    """Recursive formular for second derivative of Lagrange shape functions."""
    k = degree + 1
    xi = np.linspace(-1, 1, num=k)
    l_xx = np.zeros(k)
    for j in range(k):
        for i in range(k):
            if i == j:
                continue
            l_x = __lagrange_x_r(x, degree, skip=[i])
            l_xx[j] += l_x[j] / (xi[j] - xi[i])

    return l_xx


class OrientedPoint:
    # def __init__(self, r, d, dim=3):
    #     self.r = r
    #     self.d = d
    def __init__(self, qDOF, dim=3):
        self.qDOF = qDOF
        self.dim = dim
        # self.rDOF = np.arange(dim)
        # self.dDOF = np.arange(3, 2 * dim)

    def __call__(self, a, b=None):
        # TODO: Descide for useful version!
        if b is None:
            return self.paired(a)
        else:
            return self.stacked(a, b)

    def stacked(self, r, d):
        """For given r and d, returns the concatenated generalized coordinates
        q = np.concatenate([r, d]).
        """
        return np.concatenate([r, d])

    def paired(self, q):
        """For given generalized coordinates q, returns the stacked
        array([r, d]).
        """
        return np.array([q[: self.dim], q[self.dim :]])
        # return np.array([q[self.rDOF], q[self.dDOF]])


class CubicHermiteNode:
    def __init__(self, qDOF, dim=3):
        self.qDOF = qDOF
        self.dim = dim

    def __call__(self, *args, q=None):
        if q is None:
            return self.stacked(*args)
        else:
            return self.paired(q)

    def stacked(self, r0, t0, r1, t1):
        """For given r0, t0 and r1, t1, returns the concatenated generalized coordinates
        q = np.concatenate([r0, t0, r1, t1]).
        """
        return np.concatenate([r0, t0, r1, t1])

    def paired(self, q):
        """For given generalized coordinates q, returns the stacked
        [r0, t0, r1, t1].
        """
        return q.reshape(4, self.dim)
        # dim = self.dim
        # return np.array([q[:dim], q[dim:2*dim]])
        # return np.array([q[self.rDOF], q[self.dDOF]])


class CubicHermiteBasis:
    """TODO: Write transformation for arbitrary interval."""

    def __init__(self, nel, dim=3, interval=[0, 1]):
        self.nel = nel
        self.dim = dim
        self.interval = interval
        self.n_nodes = nel + 1

    def __call__(self, xis, q):
        xis = np.atleast_1d(xis)
        value = np.zeros((len(xis), self.dim))
        r0, t0, r1, t1 = q.reshape(4, self.dim)  # TODO: Is this the correct ordering?
        for i, xii in enumerate(xis):
            xii2 = xii * xii
            xii3 = xii2 * xii
            ar0 = 2 * xii3 - 3 * xii2 + 1.0
            at0 = xii3 - 2 * xii2 + xii
            ar1 = -2 * xii3 + 3 * xii2
            at1 = xii3 - xii2

            value[i] = ar0 * r0 + at0 * t0 + ar1 * r1 + at1 * t1

        return value

    def shape_functions(self, xis):
        """Evaluate cubic hermite shape functions.

        Note:
        -----
        This function assumes that the generalized coordiantes of each nodes
        are ordered as:

        \tqe = (r0, t0 r1, t1),

        where r0, r1 denote the coordinates at the first and second nodes,
        respectively. And further t0, t1 denote the respective derivatives
        at the corresponding nodes.
        """
        xis = np.atleast_1d(xis)
        basis = np.zeros((len(xis), self.dim, 12), dtype=float)
        for i, xii in enumerate(xis):
            xii2 = xii * xii
            xii3 = xii2 * xii
            ar0 = 2 * xii3 - 3 * xii2 + 1.0
            at0 = xii3 - 2 * xii2 + xii
            ar1 = -2 * xii3 + 3 * xii2
            at1 = xii3 - xii2

            basis[i] = np.hstack(
                [
                    ar0 * np.eye(3),
                    at0 * np.eye(3),
                    ar1 * np.eye(3),
                    at1 * np.eye(3),
                ]
            )

        return basis


if __name__ == "__main__":
    # from scipy.interpolate import CubicHermiteSpline

    # define generalized coordinates of cubic hermite spline
    r0 = np.zeros(3)
    t0 = np.array([1, 0, 0])
    r1 = np.array([1, 1, 0]) / np.sqrt(2)
    t1 = np.array([0, 1, 0])

    q0 = np.concatenate([r0, t0])
    q1 = np.concatenate([r1, t1])
    q = np.concatenate([q0, q1])

    p0 = CubicHermiteNode(np.arange(12), dim=3)
    print(f"p0(q):\n{p0(q=q)}")
    print(f"p0(r0, t0, r1, t1):\n{p0(r0, t0, r1, t1)}")

    p0 = OrientedPoint(np.arange(6))
    p1 = OrientedPoint(np.arange(6, 12))

    print(f"p0.paired():\n{p0.paired(q0)}")
    print(f"p1.paired():\n{p1.paired(q1)}")
    print(f"p0(q0):\n{p0(q0)}")
    print(f"p1(q1):\n{p1(q1)}")

    print(f"p0.stacked(): {p0.stacked(r0, t0)}")
    print(f"p1.stacked(): {p1.stacked(r1, t1)}")
    print(f"p0(r0, t0): {p0(r0, t0)}")
    print(f"p1(r1, t1): {p1(r1, t1)}")

    nel = 1
    basis = CubicHermiteBasis(nel)
    xis = np.linspace(0, 1, num=100)
    N = basis.shape_functions(xis)
    r = np.array([Ni @ q for Ni in N])

    r = basis(xis, q)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(r[:, 0], r[:, 1])
    ax.quiver(*r0[:2], *t0[:2])
    ax.quiver(*r1[:2], *t1[:2])
    ax.set_aspect(1)
    ax.grid()
    plt.show()

    exit()

    # collect generalized coordinates
    rs = np.array([r0, r1])
    ts = np.array([t0, t1])

    def cubic_herimite_shapefunctions(xis, r0, t0, r1, t1, interval=[0, 1]):
        xis2 = xis * xis
        xis3 = xis2 * xis
        ar0 = 2 * xis3 - 3 * xis2 + 1.0
        at0 = xis3 - 2 * xis2 + xis
        ar1 = -2 * xis3 + 3 * xis2
        at1 = xis3 - xis2
        return (
            np.outer(ar0, r0)
            + np.outer(at0, t0)
            + np.outer(ar1, r1)
            + np.outer(at1, t1)
        )

    # discretize parameter space
    num = 100
    xis = np.linspace(0, 1, num=num)

    # evaluate spline
    r = cubic_herimite_shapefunctions(xis, r0, t0, r1, t1)
    # print(f"xis: {xis}")
    # print(f"r(xis):\n{r}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(r[:, 0], r[:, 1])
    ax.quiver(*r0[:2], *t0[:2])
    ax.quiver(*r1[:2], *t1[:2])
    # ax.set_xlim(0, 2.5)
    # ax.set_ylim(0, 2.5)
    ax.set_aspect(1)
    ax.grid()
    plt.show()
    # spline = CubicHermiteSpline(xis, rs, ts)
