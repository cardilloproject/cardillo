from geomdl.knotvector import generate
from geomdl.helpers import find_span_linear, find_span_binsearch, find_spans
from geomdl.helpers import basis_function_ders, basis_functions_ders
from geomdl import BSpline, helpers, linalg
from geomdl import fitting

from geomdl.visualization import VisMPL

import numpy as np
import matplotlib.pyplot as plt

def uniform_knot_vector(degree, nEl, interval=[0, 1]):
    return np.concatenate( [np.ones(degree) * interval[0], \
                            np.linspace(interval[0], interval[1], nEl+1), \
                            np.ones(degree) * interval[1] ] ) 

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

    if not hasattr(knots, '__len__'):
        knots = [knots]
    lenxi = len(knots)

    span = np.zeros(lenxi, dtype=int)
    for j in range(lenxi):
        span[j] = np.where(knot_vector <= knots[j])[0][-1]
        if knots[j] == 1:
            span[j] += -degree - 1
    return span

def B_spline_basis(degree, derivative_order, knot_vector, knots):
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
    if not hasattr(knots, '__len__'):
        knots = np.array([knots])
    spans = find_knotspan(degree, knot_vector, knots)
    return __basis_functions_ders(degree, knot_vector, spans, knots, derivative_order).squeeze()
    
def __basis_functions_ders(degree, knot_vector, spans, knots, order, dtype=np.float64):
    basis_ders = np.zeros((order + 1, len(knots), degree + 1))
    for i, (span, knot) in enumerate(zip(spans, knots)):
        basis_ders[:, i] = __basis_function_ders(degree, knot_vector, span, knot, order)
    return basis_ders

def __basis_function_ders(degree, knot_vector, span, knot, order, dtype=np.float64):
    p = degree
    d = order
    Xi = knot_vector
    xi = knot
    i = span
    
    # initialize output
    ndu = np.zeros((p + 1, p + 1), dtype=dtype)
    a = np.zeros((2, p + 1), dtype=dtype)
    N  = np.zeros((d + 1, p + 1), dtype=dtype)

    # lambda functions for left and right
    left = lambda j: xi - Xi[i - j + 1]
    right = lambda j: Xi[i + j] - xi
    
    # for every element of xi
    a.fill(0)
    ndu.fill(0)

    # ALGORITHM A2.3 of Piegl1997
    ndu[0, 0] = 1

    for j in range(1,p+1):
        saved = 0
        for r in range(j): 
            # lower triangle
            ndu[j, r] = right(r + 1) + left(j - r)
            temp = ndu[r, j - 1] / ndu[j, r]

            # upper triangle
            ndu[r, j] = saved + right(r + 1) * temp
            saved = temp * left(j - r)

        ndu[j,j] = saved

    # load the basis functions
    for j in range(0, p + 1):
        N[0, j] = ndu[j, p]

    # compute the derivatives (Eq. 2.9)
    for r in range(p + 1): # loop over function index
        s1 = 0
        s2 = 1
        a[0, 0] = 1

        # loop to compute k-th derivative
        for k in range(1, d + 1):
            dd = 0
            rk = r - k
            pk = p - k
            if (r >= k):
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                dd = a[s2, 0] * ndu[rk, pk]
            
            if (rk >= -1):
                j1 = 1
            else:
                j1 = -rk
            if (r - 1 <= pk):
                j2 = k - 1 
            else:
                j2 = p - r
            
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                dd += a[s2, j] * ndu[rk + j, pk]
            
            if (r <= pk):
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                dd += a[s2, k] * ndu[r, pk]

            N[k, r] = dd

            # alternate rows in array a
            j = s1
            s1 = s2
            s2 = j

    # multiply through by the correct factors
    r = p
    for k in range(1, d + 1):
        for j in range(p + 1):
            N[k, j] *= r
        r *= (p - k)

    return N
    
def fit_B_Spline(points, degree, nEl, fixFirst=True, fixLast=True):
    r"""Fits a B-spline polynomial of degree p and with nEl elements to a spatial curve defined by a set of points P.

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
    Xi = uniform_knot_vector(degree, nEl)

    # B-spline related things
    # - knot vector
    # - shape functions
    # - connectivity matrices
    basis = B_spline_basis(degree, 0, Xi, xi)
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