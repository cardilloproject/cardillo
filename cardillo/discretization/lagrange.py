import numpy as np
from numpy.polynomial import Polynomial
from scipy.sparse.linalg import spsolve
from cardillo.discretization.indexing import flat2D, flat3D, split2D, split3D
import meshio


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
        return np.linspace(interval[0], interval[1], degree * nel + 1)

    def element_number(self, nodes):
        if not hasattr(nodes, "__len__"):
            nodes = [nodes]
        lenxi = len(nodes)

        element = np.zeros(lenxi, dtype=int)
        for j in range(lenxi):
            element[j] = (
                np.asarray(self.element_data <= nodes[j]).nonzero()[0][-1]
            ) // (self.degree)
            if nodes[j] == self.data[-1]:
                element[j] -= 1
        # if lenxi == 1:
        #     return int(element)
        return element

    def element_interval(self, el):
        return np.array(
            [
                self.element_data[el * self.degree],
                self.element_data[(el + 1) * self.degree],
            ]
        )

    def verify_data(self):
        assert len(self.element_data) == self.nel * self.degree + 1


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
            self.li[i] = Polynomial([1.0], domain=interval, window=[0, 1])
            for j in range(degree + 1):
                if i != j:
                    diff = nus[i] - nus[j]
                    self.li[i] *= Polynomial(
                        [-nus[j] / diff, 1.0 / diff], domain=interval, window=[0, 1]
                    )

    def __call__(self, xis):
        xis = np.atleast_1d(xis)
        values = np.zeros((len(xis), self.degree + 1), dtype=float)
        for i, xii in enumerate(xis):
            for j in range(self.degree + 1):
                values[i, j] = self.li[j](xii)
        return values

    def deriv(self, xis, n=1):
        xis = np.atleast_1d(xis)
        values = np.zeros((len(xis), self.degree + 1), dtype=float)
        for i, xii in enumerate(xis):
            for j in range(self.degree + 1):
                values[i, j] = self.li[j].deriv(n)(xii)
        return values


def lagrange_basis1D(degree, xis, derivative, node_vector, squeeze=True):
    """Compute Lagrange basis functions for a given node vector."""
    xis = np.atleast_1d(xis)
    nxis = len(xis)
    N = np.zeros((derivative + 1, nxis, degree + 1))
    for i, xi in enumerate(xis):
        el = node_vector.element_number(xi)[0]
        basis = LagrangeBasis(degree, interval=node_vector.element_interval(el))
        N[0, i] = basis(xi)
        if derivative:
            for j in range(1, derivative + 1):
                N[j, i] = basis.deriv(xi, n=j)
    if squeeze:
        N = N.squeeze()
    return N


def lagrange_basis1D_old(
    degree, xi, derivative=1, knot_vector=None, interval=[-1, 1], squeeze=False
):
    p = degree

    if not hasattr(xi, "__len__"):
        xi = np.array([xi])

    # k = len(xi)
    # n = sum([1 for d in range(derivative + 1)])
    # NN = np.zeros((n, k, p+1))
    # Nxi, N_xi = Lagrange_basis(p, xi, derivative=True, knot_vector=knot_vector, interval=interval)

    # NN[0] = Nxi
    # if derivative > 0:
    #     NN[1] = N_xi

    # return NN
    if squeeze:
        return Lagrange_basis(
            p, xi, derivative=derivative, knot_vector=knot_vector, interval=interval
        ).squeeze()
    else:
        return Lagrange_basis(
            p, xi, derivative=derivative, knot_vector=knot_vector, interval=interval
        )


def lagrange_basis2D(degrees, xis, derivative=1, knot_vectors=None, interval=[-1, 1]):
    p, q = degrees
    xi, eta = xis
    p1q1 = (p + 1) * (q + 1)

    if not hasattr(xi, "__len__"):
        xi = np.array([xi])
    if not hasattr(eta, "__len__"):
        eta = np.array([eta])

    k = len(xi)
    l = len(eta)
    kl = k * l

    n = sum([2**d for d in range(derivative + 1)])
    NN = np.zeros((n, kl, p1q1))
    # TODO: make seperate 1D Basis function with second derrivative
    Nxi = lagrange_basis1D(
        p, xi, derivative=derivative, knot_vector=knot_vectors[0], interval=interval
    )
    Neta = lagrange_basis1D(
        q, eta, derivative=derivative, knot_vector=knot_vectors[1], interval=interval
    )

    for i in range(kl):
        ik, il = split2D(i, (k,))

        for a in range(p1q1):
            a_xi, a_eta = split2D(a, (p + 1,))
            NN[0, i, a] = Nxi[0, ik, a_xi] * Neta[0, il, a_eta]

            if derivative > 0:
                NN[1, i, a] = Nxi[1, ik, a_xi] * Neta[0, il, a_eta]
                NN[2, i, a] = Nxi[0, ik, a_xi] * Neta[1, il, a_eta]
                if derivative > 1:
                    raise RuntimeWarning("This might not work")
                    NN[3, i, a] = Nxi[2, ik, a_xi] * Neta[0, il, a_eta]
                    NN[4, i, a] = Nxi[1, ik, a_xi] * Neta[1, il, a_eta]
                    NN[5, i, a] = NN[4, i, a]
                    NN[6, i, a] = Nxi[0, ik, a_xi] * Neta[2, il, a_eta]

    return NN


def lagrange_basis3D(degrees, xis, derivative=1, knot_vectors=None, interval=[-1, 1]):
    p, q, r = degrees
    xi, eta, zeta = xis
    p1q1r1 = (p + 1) * (q + 1) * (r + 1)

    if not hasattr(xi, "__len__"):
        xi = np.array([xi])
    if not hasattr(eta, "__len__"):
        eta = np.array([eta])
    if not hasattr(zeta, "__len__"):
        zeta = np.array([zeta])

    k = len(xi)
    l = len(eta)
    m = len(zeta)
    klm = k * l * m

    n = sum([3**d for d in range(derivative + 1)])
    NN = np.zeros((n, klm, p1q1r1))
    # TODO: make seperate 1D Basis function with second derrivative
    if knot_vectors:
        Nxi = lagrange_basis1D(
            p, xi, derivative=derivative, knot_vector=knot_vectors[0], interval=interval
        )
        Neta = lagrange_basis1D(
            q,
            eta,
            derivative=derivative,
            knot_vector=knot_vectors[1],
            interval=interval,
        )
        Nzeta = lagrange_basis1D(
            r,
            zeta,
            derivative=derivative,
            knot_vector=knot_vectors[2],
            interval=interval,
        )
    else:
        Nxi = lagrange_basis1D(p, xi, interval=interval)
        Neta = lagrange_basis1D(q, eta, interval=interval)
        Nzeta = lagrange_basis1D(r, zeta, interval=interval)

    for i in range(klm):
        ik, il, im = split3D(i, (k, l))

        for a in range(p1q1r1):
            a_xi, a_eta, a_zeta = split3D(a, (p + 1, q + 1))
            NN[0, i, a] = Nxi[0, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[0, im, a_zeta]

            if derivative > 0:
                NN[1, i, a] = (
                    Nxi[1, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[0, im, a_zeta]
                )
                NN[2, i, a] = (
                    Nxi[0, ik, a_xi] * Neta[1, il, a_eta] * Nzeta[0, im, a_zeta]
                )
                NN[3, i, a] = (
                    Nxi[0, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[1, im, a_zeta]
                )
                if derivative > 1:
                    raise RuntimeWarning("This might not work")
                    NN[4, i, a] = (
                        Nxi[2, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[0, im, a_zeta]
                    )
                    NN[5, i, a] = (
                        Nxi[1, ik, a_xi] * Neta[1, il, a_eta] * Nzeta[0, im, a_zeta]
                    )
                    NN[6, i, a] = (
                        Nxi[1, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[1, im, a_zeta]
                    )
                    NN[7, i, a] = NN[5, i, a]
                    NN[8, i, a] = (
                        Nxi[0, ik, a_xi] * Neta[2, il, a_eta] * Nzeta[0, im, a_zeta]
                    )
                    NN[9, i, a] = (
                        Nxi[0, ik, a_xi] * Neta[1, il, a_eta] * Nzeta[1, im, a_zeta]
                    )
                    NN[10, i, a] = NN[6, i, a]
                    NN[11, i, a] = NN[9, i, a]
                    NN[12, i, a] = (
                        Nxi[0, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[2, im, a_zeta]
                    )

    return NN


def Lagrange_basis(degree, x, derivative=1, knot_vector=None, interval=[-1, 1]):
    """Compute Lagrange shape function basis.

    Parameters
    ----------
    degree : int
        polynomial degree
    x : ndarray, 1D
        array containing the evaluation points of the polynomial
    derivative : bool
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


def lagrange_volume2vtk(mesh, Q, filename, binary=False):
    cells, points, HigherOrderDegrees = mesh.vtk_mesh(Q)

    meshio.write_points_cells(
        filename,
        points,
        cells,
        cell_data={"HigherOrderDegrees": HigherOrderDegrees},
        binary=binary,
    )


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


def fit_lagrange_volume(mesh, xis, Pw, qc, cDOF):
    A = mesh.L2_projection_A(xis)
    nq = A.shape[0]
    qDOF = np.arange(nq)
    fDOF = np.setdiff1d(qDOF, cDOF)

    Acsc = A.tocsc()
    Aff = Acsc[fDOF[:, None], fDOF]
    Afc = Acsc[fDOF[:, None], cDOF]

    qs = np.zeros((nq,) + Pw.shape[1:])
    for i, Pwi in enumerate(Pw.T):
        b = mesh.L2_projection_b(xis, Pwi)
        qs[cDOF, i] = qc[:, i]
        qs[fDOF, i] = spsolve(Aff, b[fDOF] - Afc @ qs[cDOF, i])

    return qs.T


def fit_lagrange_surface(mesh, xis, Pw, qc, cDOF):
    return fit_lagrange_volume(mesh, xis, Pw, qc, cDOF)


def fit_lagrange_curve(mesh, xi, Pw, qc, cDOF):
    xis = np.array([(xii,) for xii in xi])
    Pws = np.array([(xii, Pwi) for xii, Pwi in zip(xi, Pw)])
    return fit_lagrange_volume(mesh, xis, Pws, qc, cDOF)


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


# def test_fit_lagrange_volume():
#     # degrees = np.ones(3, dtype=int) * 3
#     degrees = (2, 2, 2)
#     QP_shape = (1, 1, 1)
#     element_shape = np.ones(3, dtype=int) * 5
#     element_shape = (3, 3, 3)

#     from cardillo.discretization.mesh3D_lagrange import Mesh3D_lagrange
#     mesh = Mesh3D_lagrange(degrees, QP_shape, element_shape, derivative_order=0, nq_n=3)

#     def shear(xi, eta, zeta, gamma=1.5, L=5, B=2, H=1):
#         x = xi * L + gamma * eta * B
#         y = eta * B
#         z = zeta * H
#         return x, y, z

#     def bending(xi, eta, zeta, phi0=np.pi, R=1, B=2, H=1):
#         phi = (1 - xi) * phi0
#         x = (R + B * eta) * np.cos(phi)
#         y = (R + B * eta) * np.sin(phi)
#         # x = (R + B * eta**2) * np.cos(phi)
#         # y = (R + B * eta**2) * np.sin(phi)
#         z = zeta * H
#         return x, y, z

#     def sherical_dome(xi, eta, zeta, phi0=np.pi, theta0=np.pi/2, R=1, H=1):
#         phi = (1 - xi) * phi0
#         theta = eta * theta0
#         r = R + zeta * H
#         x = r * np.cos(phi) * np.sin(theta)
#         y = r * np.sin(phi) * np.sin(theta)
#         z = r * np.cos(theta)
#         return x, y, z

#     def parabolic(xi, eta, zeta, L=1, B=1, H=1):
#         x = xi * L
#         y = eta * B + (xi - L/2)**2 * eta
#         z = zeta * H
#         return x, y, z

#     def twist(xi, eta, zeta, phi0=np.pi/2, R=1, d=1, B=1, H=1):
#         phi = xi * phi0
#         r = R + B * eta
#         x = r * np.cos(phi)
#         y = r * np.sin(phi)
#         z = zeta * H + eta**2 * zeta * d
#         return x, y, z

#     def cylinder(xi, eta, zeta, R=1, H=1):
#         xi_ = 2 * xi - 1
#         eta_ = 2 * eta - 1

#         if np.abs(xi_) > np.abs(eta_):
#             r = np.sqrt(1 + eta_**2)
#         else:
#             r = np.sqrt(1 + xi_**2)

#         x = R / r * xi_
#         y = R / r * eta_
#         z = zeta * H
#         return x, y, z

#     # nxi, neta, nzeta = 20, 5, 5
#     nxi, neta, nzeta = 10, 10, 10
#     # nxi, neta, nzeta = 20, 20, 20
#     xi = np.linspace(0, 1, num=nxi)
#     eta = np.linspace(0, 1, num=neta)
#     zeta = np.linspace(0, 1, num=nzeta)

#     B = 2
#     R = 3

#     n3 = nxi * neta * nzeta
#     xis = np.zeros((n3, 3))
#     Pw = np.zeros((n3, 3))
#     for i, xii in enumerate(xi):
#         for j, etai in enumerate(eta):
#             for k, zetai in enumerate(zeta):
#                 idx = flat3D(i, j, k, (nxi, neta, nzeta))
#                 xis[idx] = xii, etai, zetai
#                 Pw[idx] = shear(xii, etai, zetai)
#                 # Pw[idx] = bending(xii, etai, zetai, R=R, B=B)
#                 # Pw[idx] = sherical_dome(xii, etai, zetai, R=R, H=B)
#                 # Pw[idx] = parabolic(xii, etai, zetai)
#                 # Pw[idx] = twist(xii, etai, zetai)
#                 # Pw[idx] = cylinder(xii, etai, zetai)

#     cDOF = np.array([], dtype=int)
#     qc = np.array([], dtype=float).reshape((0, 3))
#     # cDOF = np.array([0], dtype=int)
#     # qc = np.array([-np.ones(3) * 0.1])
#     X, Y, Z = fit_lagrange_volume(mesh, xis, Pw, qc, cDOF)

#     lagrange_volume2vtk(mesh, np.concatenate((X, Y, Z)), 'fit_lagrange_volume.vtu')

#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(*Pw.T)
#     ax.scatter(X, Y, Z, color='red')
#     # RB = R + 0.5*B
#     # # RB = R
#     # ax.set_xlim(-RB, RB)
#     # ax.set_ylim(-RB, RB)
#     # ax.set_zlim(-RB, RB)
#     plt.show()

# def test_fit_lagrange_curve():

#     degree = 3
#     QP_shape = 2
#     element_shape = 2
#     from cardillo.discretization.mesh1D import Mesh1D
#     mesh = Mesh1D_lagrange(degree, QP_shape, element_shape, derivative_order=0, nq_n=1)

#     def polynom(xi, a):
#         return np.polynomial.polynomial.polyval(xi, a)

#     xi = np.linspace(0, 1, 100)
#     a = np.array([1.46, 6.8, -28.8, 49, -32.6, 5.8])
#     Pw = polynom(xi, a)

#     cDOF = np.array([], dtype=int)
#     qc = np.array([], dtype=float).reshape((0, 2))

#     X = fit_lagrange_curve(mesh, xi, Pw, qc, cDOF)

#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.plot(xi, Pw)
#     plt.scatter(*X,   color='red')

#     plt.show()


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
