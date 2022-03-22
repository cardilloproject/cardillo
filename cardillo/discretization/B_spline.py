import numpy as np
import meshio
from scipy.sparse.linalg import spsolve
from cardillo.discretization.indexing import flat2D, flat3D, split2D, split3D


class KnotVector:
    @staticmethod
    def uniform(degree, nel, interval=[0, 1]):
        return np.concatenate(
            [
                np.ones(degree) * interval[0],
                np.linspace(interval[0], interval[1], nel + 1),
                np.ones(degree) * interval[1],
            ]
        )

    def __init__(self, degree, nel, data=None):
        self.degree = degree
        self.nel = nel
        if data is None:
            self.data = KnotVector.uniform(degree, nel)
        else:
            self.data = data

        self.element_data = np.unique(self.data)
        self.verify_data()

    def element_number(self, knots):
        if not hasattr(knots, "__len__"):
            knots = [knots]
        lenxi = len(knots)

        element = np.zeros(lenxi, dtype=int)
        for j in range(lenxi):
            element[j] = np.where(self.element_data <= knots[j])[0][-1]
            if knots[j] == self.data[-1]:
                element[j] -= 1
        return element

    def element_interval(self, el):
        return self.element_data[el : el + 2]

    def verify_data(self):
        assert len(self.element_data) == self.nel + 1


def find_knotspan(degree, knot_vector, knots):
    r"""Finds the span index in the `knot_vector` for each element in `knots`.

    Parameters
    ----------
    degree : int
        polynomial degree of the shape functions
    knot_vector : numpy.ndarray
        knot vector.
    knots : numpy.ndarray or float
        evaulation points or single point in parameter space

    Returns
    -------
    span : numpy.ndarray or int
        knot span corresponding to the values of `knots`

    References
    ----------
    Piegl1997 - ALGORITHM A2.1, p.68: http://read.pudn.com/downloads713/ebook/2859558/The%20NURBS%20Book%202nd.pdf
    """

    if not hasattr(knots, "__len__"):
        knots = [knots]
    lenxi = len(knots)

    span = np.zeros(lenxi, dtype=int)
    for j in range(lenxi):
        span[j] = np.where(knot_vector <= knots[j])[0][-1]
        if knots[j] == knot_vector[-1]:
            span[j] += -degree - 1
    return span


def __basis_functions_ders(degree, knot_vector, spans, knots, order, dtype=np.float64):
    basis_ders = np.zeros((order + 1, len(knots), degree + 1))
    for i, (span, knot) in enumerate(zip(spans, knots)):
        basis_ders[:, i] = __basis_function_ders(degree, knot_vector, span, knot, order)
    return basis_ders


def __basis_function_ders(degree, knot_vector, span, knot, order, dtype=np.float64):
    # initialize output
    ndu = np.zeros((degree + 1, degree + 1), dtype=dtype)
    a = np.zeros((2, degree + 1), dtype=dtype)
    N = np.zeros((order + 1, degree + 1), dtype=dtype)

    # lambda functions for left and right
    left = lambda j: knot - knot_vector[span - j + 1]
    right = lambda j: knot_vector[span + j] - knot

    # ALGORITHM A2.3 of Piegl1997
    ndu[0, 0] = 1

    for j in range(1, degree + 1):
        saved = 0
        for r in range(j):
            # lower triangle
            ndu[j, r] = right(r + 1) + left(j - r)
            temp = ndu[r, j - 1] / ndu[j, r]

            # upper triangle
            ndu[r, j] = saved + right(r + 1) * temp
            saved = temp * left(j - r)

        ndu[j, j] = saved

    # load the basis functions
    for j in range(0, degree + 1):
        N[0, j] = ndu[j, degree]

    # compute the derivatives (Eq. 2.9)
    for r in range(degree + 1):  # loop over function index
        s1 = 0
        s2 = 1
        a[0, 0] = 1

        # loop to compute k-th derivative
        for k in range(1, order + 1):
            dd = 0
            rk = r - k
            pk = degree - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                dd = a[s2, 0] * ndu[rk, pk]

            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = degree - r

            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                dd += a[s2, j] * ndu[rk + j, pk]

            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                dd += a[s2, k] * ndu[r, pk]

            N[k, r] = dd

            # alternate rows in array a
            j = s1
            s1 = s2
            s2 = j

    # multiply through by the correct factors
    r = degree
    for k in range(1, order + 1):
        for j in range(degree + 1):
            N[k, j] *= r
        r *= degree - k

    return N


def B_spline_basis1D(degree, derivative_order, knot_vector, knots, squeeze=True):
    r"""Calculates the values up to the `derivative_order`-th derivative of the B-Spline basis functions of polynomial degree `degree`
    for the given knot vector `knot_vector` evaluated at all k values in `knots`.

    Parameters
    ----------
    degree : int
        polynomial degree of the shape functions
    derivative_order : int
        numbe rof computed derivatives
    knot_vector : numpy.ndarray
        knot vector.
    knots : numpy.ndarray or float
        evaulation points or single point in parameter space

    Returns
    -------
    N : numpy.ndarray
        A (d+1)-by-(k)-by-(p+1) array holding the values of the B-Spline functions and its derivatives up to degree d, evaluated at the k positions specified in the input variable xi.

    References
    ----------
    Piegl1997 - ALGORITHM A2.1, p. 72-73: http://read.pudn.com/downloads713/ebook/2859558/The%20NURBS%20Book%202nd.pdf

    """
    if not hasattr(knots, "__len__"):
        knots = np.array([knots])
    spans = find_knotspan(degree, knot_vector, knots)
    if squeeze:
        return __basis_functions_ders(
            degree, knot_vector, spans, knots, derivative_order
        ).squeeze()
    else:
        return __basis_functions_ders(
            degree, knot_vector, spans, knots, derivative_order
        )


def B_spline_basis2D(degrees, derivative_order, knot_vectors, knots):
    Xi, Eta = knot_vectors
    xi, eta = knots
    p, q = degrees
    p1q1 = (p + 1) * (q + 1)

    if not hasattr(xi, "__len__"):
        xi = np.array([xi])
    if not hasattr(eta, "__len__"):
        eta = np.array([eta])
    k = len(xi)
    l = len(eta)
    kl = k * l

    # compute number of shape functions and derivatives
    # and store them consecutively
    n = sum([2**d for d in range(derivative_order + 1)])
    NN = np.zeros((n, kl, p1q1))

    Nxi = B_spline_basis1D(p, derivative_order, Xi, xi, squeeze=False)
    Neta = B_spline_basis1D(q, derivative_order, Eta, eta, squeeze=False)

    for i in range(kl):
        ik, il = split2D(i, (k,))

        for a in range(p1q1):
            a_xi, a_eta = split2D(a, (p + 1,))
            NN[0, i, a] = Nxi[0, ik, a_xi] * Neta[0, il, a_eta]

            if derivative_order > 0:
                NN[1, i, a] = Nxi[1, ik, a_xi] * Neta[0, il, a_eta]
                NN[2, i, a] = Nxi[0, ik, a_xi] * Neta[1, il, a_eta]

                if derivative_order > 1:
                    NN[3, i, a] = Nxi[2, ik, a_xi] * Neta[0, il, a_eta]
                    NN[4, i, a] = Nxi[1, ik, a_xi] * Neta[1, il, a_eta]
                    NN[5, i, a] = NN[4, i, a]
                    NN[6, i, a] = Nxi[0, ik, a_xi] * Neta[2, il, a_eta]

    return NN


def B_spline_basis3D(degrees, derivative_order, knot_vectors, knots):
    Xi, Eta, Zeta = knot_vectors
    xi, eta, zeta = knots
    p, q, r = degrees
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

    # compute number of shape functions and derivatives
    # and store them consecutively
    n = sum([3**d for d in range(derivative_order + 1)])
    NN = np.zeros((n, klm, p1q1r1))

    Nxi = B_spline_basis1D(p, derivative_order, Xi, xi, squeeze=False)
    Neta = B_spline_basis1D(q, derivative_order, Eta, eta, squeeze=False)
    Nzeta = B_spline_basis1D(r, derivative_order, Zeta, zeta, squeeze=False)

    for i in range(klm):
        ik, il, im = split3D(i, (k, l))

        for a in range(p1q1r1):
            a_xi, a_eta, a_zeta = split3D(a, (p + 1, q + 1))
            NN[0, i, a] = Nxi[0, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[0, im, a_zeta]

            if derivative_order > 0:
                NN[1, i, a] = (
                    Nxi[1, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[0, im, a_zeta]
                )
                NN[2, i, a] = (
                    Nxi[0, ik, a_xi] * Neta[1, il, a_eta] * Nzeta[0, im, a_zeta]
                )
                NN[3, i, a] = (
                    Nxi[0, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[1, im, a_zeta]
                )
                if derivative_order > 1:
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
                    NN[11, i, a] = NN[7, i, a]
                    NN[12, i, a] = (
                        Nxi[0, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[2, im, a_zeta]
                    )

    return NN


# fitting
def fit_B_spline_curve(points, degree, nEl, fixFirst=True, fixLast=True):
    r"""Fits a B-spline polynomial curve of degree p and with nEl elements to a 
    spatial curve defined by a set of points P.

    Parameters
    ----------
    points : numpy.ndarray
        array of points defining the n dimensional curve
    degree : int
        polynomial degree of the B-spline curve
    nEl : int
        number of elements
    fixFirst : bool
        Fix first node to first point in P.
    fixLast : bool
        Fix last node to last point in P.

    Returns
    -------
    Q : numpy.ndarray
        list of all nEl + p fitted n dimensional control points

    References
    ----------
    Piegl1997 - ALGORITHM A9.6, p. 417: http://read.pudn.com/downloads713/ebook/2859558/The%20NURBS%20Book%202nd.pdf
    """
    # number of target curve points and their dimension
    n_xi = points.shape[0]
    dim = points.shape[1]

    # linear spaced xi's for target curve points
    xi = np.linspace(0, 1, n_xi)
    Xi = KnotVector.uniform(degree, nEl)

    # B-spline related things
    # - knot vector
    # - shape functions
    # - connectivity matrices
    basis = B_spline_basis1D(degree, 0, Xi, xi)
    nNd = nEl + degree
    ndPerEl = degree + 1
    tmp1 = np.tile(np.arange(0, ndPerEl), dim)
    tmp2 = np.repeat(np.arange(0, dim * nNd, step=nNd), ndPerEl)
    elDOF = (np.zeros((dim * ndPerEl, nEl), dtype=int) + np.arange(nEl)).T + tmp1 + tmp2

    # find corresponding knot vector indices of the target curve xi's
    # and compute the element number
    ind = find_knotspan(degree, Xi, xi)
    elArray = ind - degree

    # memory allocation for the positions
    nq_DOF = dim * nNd
    M = np.zeros((nq_DOF, nq_DOF))
    f = np.zeros(nq_DOF)

    # assemble matrices
    for k, (P_k, el_k) in enumerate(zip(points, elArray)):
        N = np.kron(np.eye(dim), basis[k])
        M[elDOF[el_k][:, None], elDOF[el_k]] += N.T @ N
        f[elDOF[el_k]] += P_k.T @ N

    # compute rhs contributions of boundary terms and collect constraint degrees of freedom
    cDOF1 = []
    if fixFirst:
        cDOF1.extend([i * nNd for i in range(dim)])
        f -= points[0].T @ M[cDOF1]
    cDOF2 = []
    if fixLast:
        cDOF2.extend([(i + 1) * nNd - 1 for i in range(dim)])
        f -= points[-1].T @ M[cDOF2]

    # remove boundary equations from the system
    cDOF = cDOF1 + cDOF2
    qDOF = np.arange(nq_DOF)
    fDOF = np.setdiff1d(qDOF, cDOF)

    # solve least square problem with eliminated first and last node
    Q_r = np.zeros(nq_DOF)
    Q_r[fDOF] = np.linalg.solve(M[fDOF[:, None], fDOF], f[fDOF])

    # set first and last node to given values
    if fixFirst:
        Q_r[cDOF1] = points[0]
    if fixLast:
        Q_r[cDOF2] = points[-1]
    return Q_r.reshape(dim, -1).T


def fit_B_spline_volume(mesh, knots, Pw, qc, cDOF):
    A = mesh.L2_projection_A(knots)
    nq = A.shape[0]
    qDOF = np.arange(nq)
    fDOF = np.setdiff1d(qDOF, cDOF)

    Acsc = A.tocsc()
    Aff = Acsc[fDOF[:, None], fDOF]
    Afc = Acsc[fDOF[:, None], cDOF]

    qs = np.zeros((nq,) + Pw.shape[1:])
    for i, Pwi in enumerate(Pw.T):
        b = mesh.L2_projection_b(knots, Pwi)
        qs[cDOF, i] = qc[:, i]
        qs[fDOF, i] = spsolve(Aff, b[fDOF] - Afc @ qs[cDOF, i])

    return qs.T


# Bezier decomposition
def decompose_B_spline_curve(knot_vector, Pw):
    r"""Decomposes a NURBS curve into Bezier patches. See Piegl1997 algorithm A5.6 (p. 173).
    We use n = number of weighted control points instead of n = number of weighted control points + 1
    as done in Piegl1997 (p. 93).

    Parameters
    ----------
    knot_vector: Knot_vector
        knot vector object
    Pw : numpy.ndarray
        (n)x(dim) array containing (weighted) control points

    Returns
    -------
    Qw: numpy.ndarray
        (nbezier + 1)x(Xi.degree + 1) array containing the control points of the nbezier Bezier patches.
    """
    n, dim = Pw.shape
    p = knot_vector.degree
    Xi = knot_vector.data
    Qw = np.zeros((knot_vector.nel, p + 1, dim))

    mknot = n + p
    a = p
    b = p + 1
    nbezier = 0
    for i in range(p + 1):
        Qw[nbezier, i] = Pw[i]

    while b < mknot:
        i = b
        # compute multiplicity of knot
        while b < mknot and Xi[b + 1] == Xi[b]:
            b += 1
        mult = b - i + 1

        if mult < p:
            # numerator of alpha
            numer = Xi[b] - Xi[a]

            # compute and store alphas
            alphas = np.zeros(p - mult)
            for j in reversed(range(mult + 1, p + 1)):
                alphas[j - mult - 1] = numer / (Xi[a + j] - Xi[a])
            r = p - mult
            # insert knot r times
            for j in range(1, r + 1):
                save = r - j
                # this many new points
                s = mult + j
                for k in reversed(range(s, p + 1)):
                    alpha = alphas[k - s]
                    Qw[nbezier, k] = (
                        alpha * Qw[nbezier, k] + (1 - alpha) * Qw[nbezier, k - 1]
                    )
                if b < mknot:
                    # control point of next segment
                    Qw[nbezier + 1, save] = Qw[nbezier, p]
        nbezier += 1  # bezier segment completed

        if b < mknot:
            # initialize for next segment
            for i in range(p - mult, p + 1):
                Qw[nbezier, i] = Pw[b - p + i]
            a = b
            b += 1

    return Qw


def decompose_B_spline_surface(knot_vectors, Pw):
    r"""Decomposes a NURBS surface into Bezier patches. See Piegl1997 algorithm A5.6 (p. 173).
    We use n = number of weighted control points in xi direction instead of n = number of weighted
    control points in xi direction + 1 as done in Piegl1997 (p. 93). Same applies for m = number of
    control points in eta direction.

    Parameters
    ----------
    knot_vectors: (Knot_vector, Knot_vector)
        tuple containing both knot vector objects
    Pw : numpy.ndarray
        (n)x(m)x(dim) array containing (weighted) control points

    Returns
    -------
    Qw: numpy.ndarray
        (nbezier_xi + 1)x(mbezier_eta + 1)x(Xi.degree + 1)x(Eta.degree + 1)x(dim) array containing
        the control points of the nbezier_xi x mbezier_eta Bezier patches.
    """

    def decompose_1D(n, m, p, Xi, Pw, Qw):
        a = p
        b = p + 1
        nbezier = 0
        mknot = n + p
        for i in range(p + 1):
            for row in range(m):
                Qw[nbezier, i, row] = Pw[i, row]
        while b < mknot:
            i = b
            # compute multiplicity of knot
            while b < mknot and Xi[b + 1] == Xi[b]:
                b += 1
            mult = b - i + 1

            if mult < p:
                # numerator of alpha
                numer = Xi[b] - Xi[a]

                # compute and store alphas
                alphas = np.zeros(p - mult)
                for j in reversed(range(mult + 1, p + 1)):
                    alphas[j - mult - 1] = numer / (Xi[a + j] - Xi[a])
                r = p - mult
                # insert knot r times
                for j in range(1, r + 1):
                    save = r - j
                    # this many new points
                    s = mult + j
                    for k in reversed(range(s, p + 1)):
                        alpha = alphas[k - s]
                        for row in range(m):
                            Qw[nbezier, k, row] = (
                                alpha * Qw[nbezier, k, row]
                                + (1 - alpha) * Qw[nbezier, k - 1, row]
                            )
                    if b < mknot:
                        for row in range(m):
                            Qw[nbezier + 1, save, row] = Qw[nbezier, p, row]
            nbezier += 1
            if b < mknot:
                for i in range(p - mult, p + 1):
                    for row in range(m):
                        Qw[nbezier, i, row] = Pw[b - p + i, row]
                a = b
                b += 1

    n, m, dim = Pw.shape
    p = knot_vectors[0].degree
    Xi = knot_vectors[0].data
    q = knot_vectors[1].degree
    Eta = knot_vectors[1].data

    nbezier_xi = knot_vectors[0].nel
    mbezier_eta = knot_vectors[1].nel

    Qw_xi = np.zeros((nbezier_xi, p + 1, m, dim))
    decompose_1D(n, m, p, Xi, Pw, Qw_xi)

    # store (nbezier_xi+1)x(mbezier_eta+1) Bezier segments of shape (q+1)x(p+1) and dimension dim
    # each segment has to be transposed in the end!
    Qw = np.zeros((nbezier_xi, mbezier_eta, q + 1, p + 1, dim))
    # for i, Qw_xii in enumerate(Qw_xi):
    for i in range(nbezier_xi):
        # we rotate xi and eta direction, thus xi and eta direction has to be swapped
        decompose_1D(m, p + 1, q, Eta, Qw_xi[i].transpose(1, 0, 2), Qw[i])
    # return correct ordered Bezier patches
    return Qw.transpose(0, 1, 3, 2, 4)


def decompose_B_spline_volume(knot_vectors, Pw):
    r"""Decomposes a NURBS volume into Bezier patches. See Piegl1997 algorithm A5.6 (p. 173).
    We use n = number of weighted control points in xi direction instead of n = number of weighted
    control points in xi direction + 1 as done in Piegl1997 (p. 93). Same applies for m = number of
    control points in eta direction.

    Parameters
    ----------
    knot_vectors: (Knot_vector, Knot_vector)
        tuple containing both knot vector objects
    Pw : numpy.ndarray
        (n)x(m)x(dim) array containing (weighted) control points

    Returns
    -------
    Qw: numpy.ndarray
        (nbezier_xi + 1)x(mbezier_eta + 1)x(Xi.degree + 1)x(Eta.degree + 1)x(dim) array containing
        the control points of the nbezier_xi x mbezier_eta Bezier patches.
    """

    def decompose_1D(n, m, o, p, Xi, Pw, Qw):
        a = p
        b = p + 1
        nbezier = 0
        mknot = n + p
        for i in range(p + 1):
            for row in range(m):
                for col in range(o):
                    Qw[nbezier, i, row, col] = Pw[i, row, col]
        while b < mknot:
            i = b
            # compute multiplicity of knot
            while b < mknot and Xi[b + 1] == Xi[b]:
                b += 1
            mult = b - i + 1

            if mult < p:
                # numerator of alpha
                numer = Xi[b] - Xi[a]

                # compute and store alphas
                alphas = np.zeros(p - mult)
                for j in reversed(range(mult + 1, p + 1)):
                    alphas[j - mult - 1] = numer / (Xi[a + j] - Xi[a])
                rk = p - mult
                # insert knot rk times
                for j in range(1, rk + 1):
                    save = rk - j
                    # this many new points
                    s = mult + j
                    for k in reversed(range(s, p + 1)):
                        alpha = alphas[k - s]
                        for row in range(m):
                            for col in range(o):
                                Qw[nbezier, k, row, col] = (
                                    alpha * Qw[nbezier, k, row, col]
                                    + (1 - alpha) * Qw[nbezier, k - 1, row, col]
                                )
                    if b < mknot:
                        for row in range(m):
                            for col in range(o):
                                Qw[nbezier + 1, save, row, col] = Qw[
                                    nbezier, p, row, col
                                ]
            nbezier += 1
            if b < mknot:
                for i in range(p - mult, p + 1):
                    for row in range(m):
                        for col in range(o):
                            Qw[nbezier, i, row, col] = Pw[b - p + i, row, col]
                a = b
                b += 1

    n, m, o, dim = Pw.shape
    p = knot_vectors[0].degree
    Xi = knot_vectors[0].data
    q = knot_vectors[1].degree
    Eta = knot_vectors[1].data
    r = knot_vectors[2].degree
    Zeta = knot_vectors[2].data

    nbezier_xi = knot_vectors[0].nel
    nbezier_eta = knot_vectors[1].nel
    nbezier_zeta = knot_vectors[2].nel

    Qw_xi = np.zeros((nbezier_xi, p + 1, m, o, dim))
    decompose_1D(n, m, o, p, Xi, Pw, Qw_xi)

    Qw_eta = np.zeros((nbezier_xi, nbezier_eta, q + 1, o, p + 1, dim))
    Qw = np.zeros((nbezier_xi, nbezier_eta, nbezier_zeta, r + 1, p + 1, q + 1, dim))
    for i in range(nbezier_xi):
        decompose_1D(m, o, p + 1, q, Eta, Qw_xi[i].transpose(1, 2, 0, 3), Qw_eta[i])
        for j in range(nbezier_eta):
            decompose_1D(
                o, p + 1, q + 1, r, Zeta, Qw_eta[i, j].transpose(1, 2, 0, 3), Qw[i, j]
            )

    # return correct ordered Bezier patches
    return Qw.transpose(0, 1, 2, 4, 5, 3, 6)


# vtk export
def B_spline_curve2vtk(knot_vector, Pw, filename, binary=False):
    # create bezier patches
    Qw = decompose_B_spline_curve(knot_vector, Pw)
    nbezier, degree1, dim = Qw.shape

    # initialize the array containing all points of all patches
    points = np.zeros((nbezier * degree1, dim))

    # mask rearrange point ordering in a single Bezier patch
    mask = np.concatenate([[0], [degree1 - 1], np.arange(1, degree1 - 1)])

    # iterate over all bezier patches and fill cell data and connectivities
    cells = []
    HigherOrderDegree_patches = []
    for i in range(nbezier):
        # point_range of patch
        point_range = np.arange(i * degree1, (i + 1) * degree1).reshape(1, -1)

        # define points
        points[point_range] = Qw[i, mask]

        # build cells
        cells.append(("VTK_BEZIER_CURVE", point_range))

        # build cells polynomial degrees
        HigherOrderDegree_patches.append((np.ones(3, dtype=int) * (degree1 - 1))[None])

    meshio.write_points_cells(
        filename,
        points,
        cells,
        cell_data={"HigherOrderDegrees": HigherOrderDegree_patches},
        binary=binary,
    )


# TODO: rename
def flat1D_vtk(Qw):
    p1, _ = Qw.shape

    mask = np.concatenate([[0], [p1 - 1], np.arange(1, p1 - 1)])

    return np.array(Qw[mask])


def B_spline_surface2vtk(knot_vectors, Q, filename, binary=False):
    # rearrange q's from solver to Piegl's 3D ordering
    Pw = q_to_Pw_2D(knot_vectors, Q)

    p = knot_vectors[0].degree
    q = knot_vectors[1].degree
    degrees = (p, q, 0)

    Qw = decompose_B_spline_surface(knot_vectors, Pw)
    nbezier_xi, nbezier_eta, p1, q1, dim = Qw.shape

    # rearrange Qw's
    n_patches = nbezier_xi * nbezier_eta
    patch_size = p1 * q1
    points = np.zeros((n_patches * patch_size, dim))
    cells = []
    HigherOrderDegree_patches = []
    for i in range(nbezier_xi):
        for j in range(nbezier_eta):
            idx = flat2D(i, j, (nbezier_xi, nbezier_eta))
            point_range = np.arange(idx * patch_size, (idx + 1) * patch_size)
            points[point_range] = flat2D_vtk(Qw[i, j])

            cells.append(("VTK_BEZIER_QUADRILATERAL", point_range[None]))
            HigherOrderDegree_patches.append(np.array(degrees)[None])

    meshio.write_points_cells(
        filename,
        points,
        cells,
        cell_data={"HigherOrderDegrees": HigherOrderDegree_patches},
        binary=binary,
    )


def q_to_Pw_2D(knot_vectors, q, dim=3):
    Xi, Eta = knot_vectors
    nn_xi = Xi.nel + Xi.degree
    nn_eta = Eta.nel + Eta.degree
    nn = nn_xi * nn_eta

    # rearrange generalized coordinates from solver ordering to Piegl's Pw 3D array
    Pw = np.zeros((nn_xi, nn_eta, dim))
    for j in range(nn):
        j_xi, j_eta = split2D(j, (nn_xi, nn_eta))
        idx = j + np.arange(dim) * nn
        Pw[j_xi, j_eta] = q[idx]

    return Pw


# TODO: rename
def flat2D_vtk(Qw):
    """TODO: Rearranges either a Point Array or a sorting array like elDOF in vtk ordering
    if sort is true, obj must be a mesh object.
    If sort is false, object must ba an array with dimensions:
    (mesh.n) -by- (p+1) -by- (q+1) -by- (r+1)
    See https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
    """
    # creates a selection matrix like elDOF for vtk ordering
    p1, q1, dim = Qw.shape

    # corners
    points = []
    points.extend([Qw[0, 0], Qw[-1, 0], Qw[-1, -1], Qw[0, -1]])

    # edges
    points.extend(Qw[1:-1, 0])
    points.extend(Qw[-1, 1:-1])
    points.extend(Qw[1:-1, -1])
    points.extend(Qw[0, 1:-1])

    # xy face
    for iy in range(1, q1 - 1):
        points.extend(Qw[1:-1, iy])

    return np.array(points)


def B_spline_volume2vtk(knot_vectors, Q, filename, binary=False):
    # rearrange q's from solver to Piegl's 3D ordering
    Pw = q_to_Pw_3D(knot_vectors, Q)

    p = knot_vectors[0].degree
    q = knot_vectors[1].degree
    r = knot_vectors[2].degree
    degrees = (p, q, r)

    Qw = decompose_B_spline_volume(knot_vectors, Pw)
    nbezier_xi, nbezier_eta, nbezier_zeta, p1, q1, r1, dim = Qw.shape

    # rearrange Qw's
    n_patches = nbezier_xi * nbezier_eta * nbezier_zeta
    patch_size = p1 * q1 * r1
    points = np.zeros((n_patches * patch_size, dim))
    # connectivity = np.zeros((n_patches * patch_size))
    cells = []
    HigherOrderDegree_patches = []
    for i in range(nbezier_xi):
        for j in range(nbezier_eta):
            for k in range(nbezier_zeta):
                idx = flat3D(i, j, k, (nbezier_xi, nbezier_eta))
                point_range = np.arange(idx * patch_size, (idx + 1) * patch_size)
                points[point_range] = flat3D_vtk(Qw[i, j, k])

                cells.append(("VTK_BEZIER_HEXAHEDRON", point_range[None]))
                HigherOrderDegree_patches.append(np.array(degrees)[None])

    meshio.write_points_cells(
        filename,
        points,
        cells,
        cell_data={"HigherOrderDegrees": HigherOrderDegree_patches},
        binary=binary,
    )


def q_to_Pw_3D(knot_vectors, q, dim=3):
    Xi, Eta, Zeta = knot_vectors
    nn_xi = Xi.nel + Xi.degree
    nn_eta = Eta.nel + Eta.degree
    nn_zeta = Zeta.nel + Zeta.degree
    nn = nn_xi * nn_eta * nn_zeta

    # rearrange generalized coordinates from solver ordering to Piegl's Pw 3D array
    Pw = np.zeros((nn_xi, nn_eta, nn_zeta, dim))
    for j in range(nn):
        j_xi, j_eta, j_zeta = split3D(j, (nn_xi, nn_eta, nn_zeta))
        idx = j + np.arange(dim) * nn
        Pw[j_xi, j_eta, j_zeta] = q[idx]

    return Pw


# TODO: rename
def flat3D_vtk(Qw):
    """TODO: Rearranges either a Point Array or a sorting array like elDOF in vtk ordering
    if sort is true, obj must be a mesh object.
    If sort is false, object must ba an array with dimensions:
    (mesh.n) -by- (p+1) -by- (q+1) -by- (r+1)
    See https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
    """
    # creates a selection matrix like elDOF for vtk ordering
    p1, q1, r1, dim = Qw.shape

    # corners
    points = []
    points.extend(
        [
            Qw[0, 0, 0],
            Qw[-1, 0, 0],
            Qw[-1, -1, 0],
            Qw[0, -1, 0],
            Qw[0, 0, -1],
            Qw[-1, 0, -1],
            Qw[-1, -1, -1],
            Qw[0, -1, -1],
        ]
    )

    # edges
    for iz in [0, -1]:
        points.extend(Qw[1:-1, 0, iz])
        points.extend(Qw[-1, 1:-1, iz])
        points.extend(Qw[1:-1, -1, iz])
        points.extend(Qw[0, 1:-1, iz])
    for ix in [0, -1]:
        points.extend(Qw[ix, 0, 1:-1])
    for ix in [0, -1]:
        points.extend(Qw[ix, -1, 1:-1])

    # faces
    # yz
    for ix in [0, -1]:
        for iz in range(r1 - 2):
            points.extend(Qw[ix, 1:-1, iz + 1])
    # xz
    for iy in [0, -1]:
        for iz in range(r1 - 2):
            points.extend(Qw[1:-1, iy, iz + 1])
    # xy
    for iz in [0, -1]:
        for iy in range(q1 - 2):
            points.extend(Qw[1:-1, iy + 1, iz])
    # Volume
    for iz in range(r1 - 2):
        for iy in range(q1 - 2):
            for ix in range(p1 - 2):
                points.extend([Qw[ix + 1, iy + 1, iz + 1]])

    return np.array(points)


# tests
def test_Knot_vector():
    degree = 2
    nel = 3
    U = KnotVector(degree, nel)
    print(f"U.data: {U.data}")

    U = KnotVector(degree, nel, data=np.array([0, 0, 0, 0.25, 0.5, 0.5, 1, 1, 1]))
    print(f"U.data: {U.data}")
    print(f"U.element_data: {U.element_data}")

    print(f"el(0.3): {U.element_number(0.3)}")
    print(f"el(0.7): {U.element_number(0.7)}")
    print(f"el([0.3, 0.5, 0.7, 1.0]): {U.element_number([0.2, 0.5, 0.7, 1.0])}")


def test_Piegl_Fig5_18():
    degree = 3
    nel = 4
    data = np.array([0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4])
    knot_vector = KnotVector(degree, nel, data=data)

    q = np.array(
        [[0, 0], [0.1, 1], [1, 1], [1.5, -0.1], [2.2, -0.1], [2.5, 0.8], [2.0, 1.3]]
    )

    B_spline_curve2vtk(knot_vector, q, "Bezier_curve.vtu")

    num = 100
    xis = np.linspace(0, 4, num=num)
    Nxi = np.zeros((degree + nel, num))
    for i, xi in enumerate(xis):
        el = knot_vector.element_number(xi)[0]
        Nxi[el : el + degree + 1, i] = B_spline_basis1D(degree, 0, data, xi)
    c = Nxi.T @ q

    Qw = decompose_B_spline_curve(knot_vector, q)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1)
    for Ni in Nxi:
        ax[0].plot(xis, Ni)
    ax[1].plot(*q.T, "--ob")
    ax[1].plot(*c.T, "-k")
    ax[1].plot(*Qw.T, "--xr")

    plt.show()


def test_Piegl_Fig5_20():
    degrees = (2, 3)
    nels = (2, 2)
    data0 = np.array([0, 0, 0, 2 / 5, 1, 1, 1])
    data1 = np.array([0, 0, 0, 0, 3 / 5, 1, 1, 1, 1])
    knot_vector0 = KnotVector(degrees[0], nels[0], data=data0)
    knot_vector1 = KnotVector(degrees[1], nels[1], data=data1)

    # n = 200
    # xi_idx = 2
    # eta_idx = 1

    # xis = np.linspace(0, 1, num=n)
    # etas = np.linspace(0, 1, num=n)

    # import matplotlib.pyplot as plt
    # from matplotlib import gridspec

    # fig = plt.figure()
    # gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1])
    # ax0 = fig.add_subplot(gs[0, 0])
    # ax1 = fig.add_subplot(gs[1, 0])
    # ax2 = fig.add_subplot(gs[:, 1], projection='3d')

    # Nxi = np.zeros((degrees[0] + nels[0], n))
    # for i, xi in enumerate(xis):
    #     el = knot_vector0.element_number(xi)[0]
    #     Nxi[el:el+degrees[0] + 1, i] = B_spline_basis1D(degrees[0], 0, data0, xi)

    # ax0.plot(xis, Nxi[xi_idx])
    # # for Nxii in Nxi:
    # #     ax0.plot(xis, Nxii)

    # Neta = np.zeros((degrees[1] + nels[1], n))
    # for j, eta in enumerate(etas):
    #     el = knot_vector1.element_number(eta)[0]
    #     Neta[el:el+degrees[1] + 1, j] = B_spline_basis1D(degrees[1], 0, data1, eta)

    # ax1.plot(etas, Neta[eta_idx])
    # # for Netaj in Neta:
    # #     ax1.plot(etas, Netaj)

    # ax2.set_xlabel('x [m]')
    # ax2.set_ylabel('y [m]')
    # ax2.set_zlabel('z [m]')
    # ax2.set_xlim3d(left=0, right=1)
    # ax2.set_ylim3d(bottom=0, top=1)
    # ax2.set_zlim3d(bottom=0, top=1)

    # ax2.plot_surface(*np.meshgrid(xis, etas), np.outer(Nxi[xi_idx], Neta[eta_idx]))

    # plt.show()

    q0 = np.array([[0, 0, 2], [1, 0, 2], [1.5, 0, 1], [2.5, 0, 1]])
    q1 = q0.copy()
    q1[:, 1] += 0.5
    q2 = q1.copy()
    q2[:, 1] += 0.4
    q2[:, 2] -= 0.6
    q3 = q2.copy()
    q3[:, 1] += 0.6
    q3[:, 2] -= 0.4
    q4 = q3.copy()
    q4[:, 1] += 0.5

    X, Y, Z = np.vstack((q0, q1, q2, q3, q4)).T
    Q = np.concatenate((X, Y, Z))

    # rearrange points
    nn_xi = knot_vector0.nEl + knot_vector0.degree
    nn_eta = knot_vector1.nEl + knot_vector1.degree
    nn = nn_xi * nn_eta
    Pw = np.zeros((nn_xi, nn_eta, 3))
    for j in range(nn):
        j_xi, j_eta = split2D(j, (nn_xi, nn_eta))
        Pw[j_xi, j_eta] = np.array([Q[j], Q[j + nn], Q[j + 2 * nn]])

    B_spline_surface2vtk((knot_vector0, knot_vector1), Q, "Bezier_surface.vtu")
    exit()

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X, Y, Z)
    # plt.show()

    Qw = decompose_B_spline_surface((knot_vector0, knot_vector1), Pw)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax0 = fig.add_subplot(121, projection="3d")
    ax1 = fig.add_subplot(122, projection="3d")
    n, m = Qw.shape[:2]
    colors1 = ["red", "blue"]
    colors2 = [["red", "blue"], ["green", "black"]]
    # for i, c1 in enumerate(colors1):
    for i in range(n):
        # Qw_i = Qw[i]
        # Qw_i_1D = Qw_i.reshape(-1, 3)
        # ax0.scatter(*Qw_i_1D.T, color=c1)
        for j in range(m):
            Qw_ij = Qw[i, j]
            Qw_ij_1D = Qw_ij.reshape(-1, 3)
            ax0.scatter(*Qw_ij_1D.T, color=colors2[i][j])
            # ax0.scatter(*Qw_ij_1D.T, color='black')

    # ax1.scatter(*Qw[0, 0].reshape(-1, 3).T, color='black')
    ax1.scatter(X, Y, Z, color="red")
    plt.show()


def test_Piegl_Ex3_4():
    degrees = (3, 2)
    nEls = (4, 6)
    # U = np.array([0, 0, 0, 0, 1/4, 1/2, 3/4, 1, 1, 1, 1])
    U = uniform_knot_vector(degrees[0], nEls[0])
    V = np.array([0, 0, 0, 1 / 5, 2 / 5, 3 / 5, 3 / 5, 4 / 5, 1, 1, 1])

    n = 200
    xi_idx = 2
    eta_idx = 4

    xis = np.linspace(0, 1, num=n)
    etas = np.linspace(0, 1, num=n)

    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[:, 1], projection="3d")

    Nxi = np.zeros((degrees[0] + nEls[0], n))
    for i, xi in enumerate(xis):
        span = find_knotspan(degrees[0], U, xi)[0]
        el = span - degrees[0]
        Nxi[el : el + degrees[0] + 1, i] = B_spline_basis1D(degrees[0], 0, U, xi)

    ax0.plot(xis, Nxi[xi_idx])
    # for Nxii in Nxi:
    #     ax0.plot(xis, Nxii)

    Neta = np.zeros((degrees[1] + nEls[1], n))
    for j, eta in enumerate(etas):
        span = find_knotspan(degrees[1], V, eta)[0]
        el = span - degrees[1]
        Neta[el : el + degrees[1] + 1, j] = B_spline_basis1D(degrees[1], 0, V, eta)

    ax1.plot(etas, Neta[eta_idx])
    # for Netaj in Neta:
    #     ax1.plot(etas, Netaj)

    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_zlabel("z [m]")
    ax2.set_xlim3d(left=0, right=1)
    ax2.set_ylim3d(bottom=0, top=1)
    ax2.set_zlim3d(bottom=0, top=1)

    ax2.plot_surface(*np.meshgrid(xis, etas), np.outer(Nxi[xi_idx], Neta[eta_idx]))

    plt.show()


def test_mesh3D_vtk_export():
    # degrees = (1, 2, 3)
    # element_shape = (3, 2, 1)
    degrees = (2, 2, 2)
    QP_shape = (3, 4, 2)
    element_shape = (1, 1, 1)

    Xi = KnotVector(degrees[0], element_shape[0])
    Eta = KnotVector(degrees[1], element_shape[1])
    Zeta = KnotVector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)

    from cardillo.discretization.mesh3D import Mesh3D, cube, scatter_Qs

    mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=2, basis="B-spline", nq_n=3)

    cube_shape = (3, 3, 3)
    Q_cube = cube(cube_shape, mesh, Greville=True, Fuzz=1.0e-1)
    # scatter_Qs(Q_cube)

    # rearrange generalized coordinates
    B_spline_volume2vtk((Xi, Eta, Zeta), Q_cube, "Bezier_volume.vtu")


def test_fit_B_spline_volume():
    # degrees = np.ones(3, dtype=int) * 3
    degrees = (3, 3, 1)
    QP_shape = (1, 1, 1)
    element_shape = np.ones(3, dtype=int) * 5
    # element_shape = (10, 3, 3)

    Xi = KnotVector(degrees[0], element_shape[0])
    Eta = KnotVector(degrees[1], element_shape[1])
    Zeta = KnotVector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)

    from cardillo.discretization.mesh3D import Mesh3D

    mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=0, basis="B-spline", nq_n=3)

    def shear(xi, eta, zeta, gamma=1.5, L=5, B=2, H=1):
        x = xi * L + gamma * eta * B
        y = eta * B
        z = zeta * H
        return x, y, z

    def bending(xi, eta, zeta, phi0=np.pi, R=1, B=2, H=1):
        phi = (1 - xi) * phi0
        x = (R + B * eta) * np.cos(phi)
        y = (R + B * eta) * np.sin(phi)
        # x = (R + B * eta**2) * np.cos(phi)
        # y = (R + B * eta**2) * np.sin(phi)
        z = zeta * H
        return x, y, z

    def sherical_dome(xi, eta, zeta, phi0=np.pi, theta0=np.pi / 2, R=1, H=1):
        phi = (1 - xi) * phi0
        theta = eta * theta0
        r = R + zeta * H
        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(theta)
        return x, y, z

    def parabolic(xi, eta, zeta, L=1, B=1, H=1):
        x = xi * L
        y = eta * B + (xi - L / 2) ** 2 * eta
        z = zeta * H
        return x, y, z

    def twist(xi, eta, zeta, phi0=np.pi / 2, R=1, d=1, B=1, H=1):
        phi = xi * phi0
        r = R + B * eta
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = zeta * H + eta**2 * zeta * d
        return x, y, z

    def cylinder(xi, eta, zeta, R=1, H=1):
        xi_ = 2 * xi - 1
        eta_ = 2 * eta - 1

        if np.abs(xi_) > np.abs(eta_):
            r = np.sqrt(1 + eta_**2)
        else:
            r = np.sqrt(1 + xi_**2)

        x = R / r * xi_
        y = R / r * eta_
        z = zeta * H
        return x, y, z

    # nxi, neta, nzeta = 20, 5, 5
    nxi, neta, nzeta = 10, 10, 10
    # nxi, neta, nzeta = 20, 20, 20
    xi = np.linspace(0, 1, num=nxi)
    eta = np.linspace(0, 1, num=neta)
    zeta = np.linspace(0, 1, num=nzeta)

    B = 1
    R = 1

    n3 = nxi * neta * nzeta
    knots = np.zeros((n3, 3))
    Pw = np.zeros((n3, 3))
    for i, xii in enumerate(xi):
        for j, etai in enumerate(eta):
            for k, zetai in enumerate(zeta):
                idx = flat3D(i, j, k, (nxi, neta, nzeta))
                knots[idx] = xii, etai, zetai
                # Pw[idx] = shear(xii, etai, zetai)
                # Pw[idx] = bending(xii, etai, zetai, R=R, B=B)
                # Pw[idx] = sherical_dome(xii, etai, zetai, R=R, H=B)
                # Pw[idx] = parabolic(xii, etai, zetai)
                # Pw[idx] = twist(xii, etai, zetai)
                Pw[idx] = cylinder(xii, etai, zetai)

    cDOF = np.array([], dtype=int)
    qc = np.array([], dtype=float).reshape((0, 3))
    # cDOF = np.array([0], dtype=int)
    # qc = np.array([-np.ones(3) * 0.1])
    X, Y, Z = fit_B_spline_volume(mesh, knots, Pw, qc, cDOF)

    B_spline_volume2vtk(
        knot_vectors, np.concatenate((X, Y, Z)), "fit_B_spline_colume.vtu"
    )

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*Pw.T)
    ax.scatter(X, Y, Z, color="red")
    # RB = R + 0.5*B
    # # RB = R
    # ax.set_xlim(-RB, RB)
    # ax.set_ylim(-RB, RB)
    # ax.set_zlim(-RB, RB)
    plt.show()


if __name__ == "__main__":
    # test_Knot_vector()
    # test_Piegl_Ex3_4()
    # test_Piegl_Fig5_18()
    # test_Piegl_Fig5_20()
    # test_mesh3D_vtk_export()
    test_fit_B_spline_volume()
