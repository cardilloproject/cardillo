import numpy as np
from scipy.interpolate import BPoly


class HermiteNodeVector:
    def __init__(self, degree, nel, data=None):
        assert degree % 2 == 1, "degree must be odd"
        self.degree = degree
        self.nel = nel
        if data is None:
            self.data = HermiteNodeVector.uniform(degree, nel)
        else:
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


class HermiteBasis:
    def __init__(self, nel, degree=3, dim=3, interval=[0, 1]):
        assert degree == 3 or degree == 5, "degree has to be 3 or 5"
        self.degree = degree
        self.nel = nel
        self.dim = dim
        self.interval = interval
        self.n_nodes = nel + 1

        # self.__call__ = self.__call3 if degree == 3 else self.__call5
        self.basis = self.__basis3 if degree == 3 else self.__basis5

    @staticmethod 
    def __basis3_impl(xi, interval=[0, 1]):
        """Evaluate cubic hermite basis functions at xi on an arbitrary 
        interval, see Wiki1.

        The required coordinate transformation is described in Wiki2.
        
        References:
        -----------
        Wiki1: https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Representations \\
        Wiki2: https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Interpolation_on_an_arbitrary_interval
        """
        # compute coordinate transformation on arbitrary interval
        diff = (interval[1] - interval[0])
        t = (xi - interval[0]) / diff

        # compute basis functions on arbitrary interval
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h01 = (t3 - 2 * t2 + t) * diff
        h10 = -2 * t3 + 3 * t2
        h11 = (t3 - t2) * diff

        return np.array([h00, h01, h10, h11])

    def __basis3(self, xis):
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

        basis = np.zeros((len(xis), self.dim, 4 * self.dim), dtype=float)
        for i, xii in enumerate(xis):
            h00, h01, h10, h11 = self.__basis3_impl(xii, interval=self.interval)

            basis[i] = np.hstack(
                [
                    h00 * np.eye(3),
                    h01 * np.eye(3),
                    h10 * np.eye(3),
                    h11 * np.eye(3),
                ]
            )

        return basis.squeeze()

    def __basis5(self, xis):
        raise NotImplementedError("")

    def __call3(self, xis, q):
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

    def __call5(self, xis, q):
        raise NotImplementedError("")


if __name__ == "__main__":
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
    t0 = e1
    n0 = e2
    # r1 = np.array([1, 1, 0]) / np.sqrt(2)
    r1 = np.array([1, 1, 1]) / np.sqrt(3)
    t1 = e2
    # n1 = -e1
    n1 = -e3
    # n1 = e1

    #######################
    # TODO: Remove old code
    #######################
    # q0 = np.concatenate([r0, t0])
    # q1 = np.concatenate([r1, t1])
    # q = np.concatenate([q0, q1])

    # p0 = CubicHermiteNode(np.arange(12), dim=3)
    # print(f"p0(q):\n{p0(q=q)}")
    # print(f"p0(r0, t0, r1, t1):\n{p0(r0, t0, r1, t1)}")

    # p0 = OrientedPoint(np.arange(6))
    # p1 = OrientedPoint(np.arange(6, 12))

    # print(f"p0.paired():\n{p0.paired(q0)}")
    # print(f"p1.paired():\n{p1.paired(q1)}")
    # print(f"p0(q0):\n{p0(q0)}")
    # print(f"p1(q1):\n{p1(q1)}")

    # print(f"p0.stacked(): {p0.stacked(r0, t0)}")
    # print(f"p1.stacked(): {p1.stacked(r1, t1)}")
    # print(f"p0(r0, t0): {p0(r0, t0)}")
    # print(f"p1(r1, t1): {p1(r1, t1)}")

    # nel = 1
    # basis = CubicHermiteBasis(nel)
    # xis = np.linspace(0, 1, num=10)
    # N = basis.shape_functions(xis)
    # r = np.array([Ni @ q for Ni in N])

    # r = basis(xis, q)

    case = "cubic"
    # case = "quintic"
    # case = "septic"
    # case = "mixed"
    # case = "mixed2"
    # case = "test"
    from scipy.interpolate import BPoly

    xi = np.array([0, 1])  # interval
    if case == "cubic":
        hermite = HermiteBasis(1)
        r_poly = lambda xi: hermite.basis(xi)

        # N = hermite.basis(0)
        q = np.concatenate([r0, t0, r1, t1])
        # r = N @ q
        r_poly = lambda xi: hermite.basis(xi) @ q
        r = r_poly(0)

        # function values and their derivatives
        yi = np.zeros((2, 2, 3))
        yi[0, 0] = r0
        yi[0, 1] = t0
        yi[1, 0] = r1
        yi[1, 1] = t1

        # build spline objects
        r_poly = BPoly.from_derivatives(xi, yi, extrapolate=False)
        r_xi_poly = r_poly.derivative(1)
        r_xixi_poly = r_poly.derivative(2)
        r_xixixi_poly = r_poly.derivative(3)
    elif case == "quintic":
        # function values and their derivatives
        yi = np.zeros((2, 3, 3))
        yi[0, 0] = r0
        yi[0, 1] = t0
        yi[0, 2] = n0
        yi[1, 0] = r1
        yi[1, 1] = t1
        yi[1, 2] = n1

        # build spline objects
        r_poly = BPoly.from_derivatives(xi, yi, extrapolate=False)
        r_xi_poly = r_poly.derivative(1)
        r_xixi_poly = r_poly.derivative(2)
        r_xixixi_poly = r_poly.derivative(3)
    elif case == "septic":
        # function values and their derivatives
        yi = np.zeros((2, 4, 3))
        yi[0, 0] = r0
        yi[0, 1] = t0
        yi[0, 2] = n0
        yi[0, 3] = b0
        yi[1, 0] = r1
        yi[1, 1] = t1
        yi[1, 2] = n1
        yi[1, 3] = b1

        # build spline objects
        r_poly = BPoly.from_derivatives(xi, yi, extrapolate=False)
        r_xi_poly = r_poly.derivative(1)
        r_xixi_poly = r_poly.derivative(2)
        r_xixixi_poly = r_poly.derivative(3)
    elif case == "mixed":
        # function values and their derivatives for the centerline spline
        yri = np.zeros((2, 3, 3))
        yri[0, 0] = r0
        yri[0, 1] = t0
        yri[1, 0] = r1
        yri[1, 1] = t1

        # function values and their derivatives for the tangent spline
        # yti = np.zeros((2, 2, 3))
        # yti[0, 0] = t0
        # yti[0, 1] = n0
        # yti[1, 0] = t1
        # yti[1, 1] = n1
        yti = np.zeros((2, 1, 3))
        yti[0, 0] = n0
        yti[1, 0] = n1

        # build spline objects
        r_poly = BPoly.from_derivatives(xi, yri, extrapolate=False)
        r_xi_poly = r_poly.derivative(1)
        r_xixi_poly = r_poly.derivative(2)

        t_poly = BPoly.from_derivatives(xi, yti, extrapolate=False)
        t_xi_poly = t_poly.derivative(1)
        t_xixi_poly = t_poly.derivative(2)
    elif case == "mixed2":
        # function values and their derivatives for the centerline spline
        yri = np.zeros((2, 2, 3))
        yri[0, 0] = r0
        yri[0, 1] = t0
        yri[1, 0] = r1
        yri[1, 1] = t1
        # yri = np.zeros((2, 3, 3))
        # yri[0, 0] = r0
        # yri[0, 1] = t0
        # yri[0, 2] = n0
        # yri[1, 0] = r1
        # yri[1, 1] = t1
        # yri[1, 2] = n1

        # function values and their derivatives for the tangent spline
        # yti = np.zeros((2, 3, 3))
        # yti[0, 0] = t0
        # yti[0, 1] = n0
        # yti[0, 2] = b0
        # yti[1, 0] = t1
        # yti[1, 1] = n1
        # yti[1, 2] = b1
        yti = np.zeros((2, 2, 3))
        yti[0, 0] = t0
        yti[0, 1] = n0
        yti[1, 0] = t1
        yti[1, 1] = n1
        # yti = np.zeros((2, 1, 3))
        # yti[0, 0] = t0
        # yti[1, 0] = t1

        # function values and their derivatives for the normal spline
        # yni = np.zeros((2, 2, 3))
        # yni[0, 0] = n0
        # yni[0, 1] = b0
        # yni[1, 0] = n1
        # yni[1, 1] = b1
        yni = np.zeros((2, 1, 3))
        yni[0, 0] = n0
        yni[1, 0] = n1

        # build spline objects
        r_poly = BPoly.from_derivatives(xi, yri, extrapolate=False)
        r_xi_poly = r_poly.derivative(1)
        r_xixi_poly = r_poly.derivative(2)

        t_poly = BPoly.from_derivatives(xi, yti, extrapolate=False)
        t_xi_poly = t_poly.derivative(1)
        t_xixi_poly = t_poly.derivative(2)

        n_poly = BPoly.from_derivatives(xi, yni, extrapolate=False)
        n_xi_poly = n_poly.derivative(1)
        n_xixi_poly = n_poly.derivative(2)
    elif case == "test":
        # function values and their derivatives for the centerline spline
        yri = np.zeros((2, 2, 3))
        yri[0, 0] = r0
        yri[0, 1] = t0
        yri[1, 0] = r1
        yri[1, 1] = t1
        # yri = np.zeros((2, 3, 3))
        # yri[0, 0] = r0
        # yri[0, 1] = t0
        # yri[0, 2] = n0
        # yri[1, 0] = r1
        # yri[1, 1] = t1
        # yri[1, 2] = n1

        # function values and their derivatives for the tangent spline
        # yti = np.zeros((2, 3, 3))
        # yti[0, 0] = t0
        # yti[0, 1] = n0
        # yti[0, 2] = b0
        # yti[1, 0] = t1
        # yti[1, 1] = n1
        # yti[1, 2] = b1
        # yti = np.zeros((2, 2, 3))
        # yti[0, 0] = t0
        # yti[0, 1] = n0
        # yti[1, 0] = t1
        # yti[1, 1] = n1
        yti = np.zeros((2, 1, 3))
        yti[0, 0] = t0
        yti[1, 0] = t1

        # # function values and their derivatives for the normal spline
        # yni = np.zeros((2, 2, 3))
        # yni[0, 0] = n0
        # yni[0, 1] = b0
        # yni[1, 0] = n1
        # yni[1, 1] = b1
        yni = np.zeros((2, 1, 3))
        yni[0, 0] = n0
        yni[1, 0] = n1

        # build spline objects
        r_poly = BPoly.from_derivatives(xi, yri, extrapolate=False)
        r_xi_poly = r_poly.derivative(1)
        r_xixi_poly = r_poly.derivative(2)

        t_poly = BPoly.from_derivatives(xi, yti, extrapolate=False)
        t_xi_poly = t_poly.derivative(1)
        t_xixi_poly = t_poly.derivative(2)

        n_poly = BPoly.from_derivatives(xi, yni, extrapolate=False)
        n_xi_poly = n_poly.derivative(1)
        n_xixi_poly = n_poly.derivative(2)
    else:
        raise RuntimeError(
            "Wrong order chosen. Allowed orders are 'cubic' and 'quintic'."
        )

    # evaluation points
    xis = np.linspace(0, 1, num=20)

    # evaluate centerline basis
    r = np.array([r_poly(xi) for xi in xis])

    # evaluate tangent basis
    if case == "mixed":
        r_xi = np.array([r_xi_poly(xi) for xi in xis])

        t = np.array([t_poly(xi) for xi in xis])
        t_xi = np.array([t_xi_poly(xi) for xi in xis])
        t_xixi = np.array([t_xixi_poly(xi) for xi in xis])

        # first director from derivative of centerline spline
        d1 = np.array([r_xii / norm(r_xii) for r_xii in r_xi])
        # d1 = np.array([ti / norm(ti) for ti in r_xi])
        # second director from derivative of tangent spline
        d2 = np.array(
            [ni / norm(ni) for ni in t]
        )  # TODO: We abused the spline for the normal here!
        # d2 = np.array([t_xii / norm(t_xii) for t_xii in t_xi])
        # third director is computed via the cross product
        t = np.array(
            [cross3(d1i, d2i) / norm(cross3(d1i, d2i)) for (d1i, d2i) in zip(d1, d2)]
        )
    elif case == "mixed2":
        # t = np.array([t_poly(xi) for xi in xis])
        t = np.array([r_xi_poly(xi) for xi in xis])
        n = np.array([n_poly(xi) for xi in xis])
        # n = np.array([t_xi_poly(xi) for xi in xis])

        # first director from derivative of centerline spline
        d1 = np.array([ti / norm(ti) for ti in t])
        # second director from derivative of tangent spline
        d2 = np.array([ni / norm(ni) for ni in n])
        # third director is computed via the cross product
        t = np.array(
            [cross3(d1i, d2i) / norm(cross3(d1i, d2i)) for (d1i, d2i) in zip(d1, d2)]
        )
    elif case == "test":
        t = np.array([r_xi_poly(xi) for xi in xis])
        # n = np.array([r_xixi_poly(xi) for xi in xis])

        # t = np.array([t_poly(xi) for xi in xis])
        # n = np.array([t_xi_poly(xi) for xi in xis])
        # n = np.array([n_poly(xi) for xi in xis])

        b = np.array([cross3(r_xi_poly(xi), n_poly(xi)) for xi in xis])
        n = np.array([cross3(bi, ti) for (ti, bi) in zip(t, b)])

        # first director from derivative of centerline spline
        d1 = np.array([ti / norm(ti) for ti in t])
        # second director from derivative of tangent spline
        d2 = np.array([ni / norm(ni) for ni in n])
        # third director is computed via the cross product
        t = np.array(
            [cross3(d1i, d2i) / norm(cross3(d1i, d2i)) for (d1i, d2i) in zip(d1, d2)]
        )
    else:
        r_xi = np.array([r_xi_poly(xi) for xi in xis])
        r_xixi = np.array([r_xixi_poly(xi) for xi in xis])

        t = r_xi
        b = np.array([cross3(r_xi_poly(xi), r_xixi_poly(xi)) for xi in xis])
        n = np.array([cross3(bi, ti) for (ti, bi) in zip(t, b)])

        # first director from derivative of centerline spline
        d1 = np.array([ti / norm(ti) for ti in t])
        # second director from derivative of tangent spline
        d2 = np.array([ni / norm(ni) for ni in n])
        # third director is computed via the cross product
        t = np.array(
            [cross3(d1i, d2i) / norm(cross3(d1i, d2i)) for (d1i, d2i) in zip(d1, d2)]
        )

        # d2 = np.array([r_xixii / norm(r_xixii) for r_xixii in r_xixi])
        # d3 = np.array(
        #     [cross3(d1i, d2i) / norm(cross3(d1i, d2i)) for (d1i, d2i) in zip(d1, d2)]
        # )

    # d1_05 = r_xi_poly(0.5) / norm(r_xi_poly(0.5))
    # d2_05 = r_xixi_poly(0.5) / norm(r_xixi_poly(0.5))
    # d3_05 = cross3(d1_05, d2_05) / norm(cross3(d1_05, d2_05))
    # print(f"d1(0.5) @ d2(0.5): {d1_05 @ d2_05}")
    # print(f"d1(0.5) @ d3(0.5): {d1_05 @ d3_05}")
    # print(f"d2(0.5) @ d3(0.5): {d2_05 @ d3_05}")

    # check orthogonality
    for i in range(len(xis)):
        d1d2 = d1[i] @ d2[i]
        # d1d3 = d1[i] @ d3[i]
        # d2d3 = d2[i] @ d3[i]
        print(f"d1 @ d2: {d1d2}")
        # assert np.allclose(d1d2, 0), ""

    # visualize spline
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.plot(r[:, 0], r[:, 1], r[:, 2], "-k")
    for i in range(len(r)):
        # ax.quiver3D(*r[i], *r_xi[i], color="r", length=0.1)
        # ax.quiver3D(*r[i], *r_xixi[i], color="g", length=0.1)
        # # ax.quiver3D(*r[i], *r_xixixi[i], color="b", length=0.1)
        ax.quiver3D(*r[i], *d1[i], color="r", length=0.1)
        ax.quiver3D(*r[i], *d2[i], color="g", length=0.1)
        ax.quiver3D(*r[i], *t[i], color="b", length=0.1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.grid()
    plt.show()
