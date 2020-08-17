import numpy as np
from cardillo.discretization.indexing import split2D, split3D

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

def __basis_functions_ders(degree, knot_vector, spans, knots, order, dtype=np.float64):
    basis_ders = np.zeros((order + 1, len(knots), degree + 1))
    for i, (span, knot) in enumerate(zip(spans, knots)):
        basis_ders[:, i] = __basis_function_ders(degree, knot_vector, span, knot, order)
    return basis_ders

def __basis_function_ders(degree, knot_vector, span, knot, order, dtype=np.float64):    
    # initialize output
    ndu = np.zeros((degree + 1, degree + 1), dtype=dtype)
    a = np.zeros((2, degree + 1), dtype=dtype)
    N  = np.zeros((order + 1, degree + 1), dtype=dtype)

    # lambda functions for left and right
    left = lambda j: knot - knot_vector[span - j + 1]
    right = lambda j: knot_vector[span + j] - knot

    # ALGORITHM A2.3 of Piegl1997
    ndu[0, 0] = 1

    for j in range(1,degree+1):
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
    for j in range(0, degree + 1):
        N[0, j] = ndu[j, degree]

    # compute the derivatives (Eq. 2.9)
    for r in range(degree + 1): # loop over function index
        s1 = 0
        s2 = 1
        a[0, 0] = 1

        # loop to compute k-th derivative
        for k in range(1, order + 1):
            dd = 0
            rk = r - k
            pk = degree - k
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
                j2 = degree - r
            
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
    r = degree
    for k in range(1, order + 1):
        for j in range(degree + 1):
            N[k, j] *= r
        r *= (degree - k)

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
    if not hasattr(knots, '__len__'):
        knots = np.array([knots])
    spans = find_knotspan(degree, knot_vector, knots)
    if squeeze:
        return __basis_functions_ders(degree, knot_vector, spans, knots, derivative_order).squeeze()
    else:
        return __basis_functions_ders(degree, knot_vector, spans, knots, derivative_order)

def B_spline_basis2D(degrees, derivative_order, knot_vectors, knots):
    Xi, Eta = knot_vectors
    xi, eta = knots
    p, q = degrees
    p1q1 = (p + 1) * (q + 1)
    
    if not hasattr(xi, '__len__'):
        xi = np.array([xi])
    if not hasattr(eta, '__len__'):
        eta = np.array([eta])
    k = len(xi)
    l = len(eta)
    kl = k * l

    NN = []
    for d in range(derivative_order + 1):
        N = np.zeros((kl, p1q1, *(2,) * d))
        NN.append(N)

    Nxi = B_spline_basis1D(p, derivative_order, Xi, xi, squeeze=False)
    Neta = B_spline_basis1D(q, derivative_order, Eta, eta, squeeze=False)

    for i in range(kl):
        ik, il = split2D(i, (k,))

        for a in range(p1q1):
            a_xi, a_eta = split2D(a, (p + 1,))

            NN[0][i, a] = Nxi[0, ik, a_xi] * Neta[0, il, a_eta]

            if derivative_order > 0:
                NN[1][i, a, 0] = Nxi[1, ik, a_xi] * Neta[0, il, a_eta]
                NN[1][i, a, 1] = Nxi[0, ik, a_xi] * Neta[1, il, a_eta]
                if derivative_order > 1:
                    NN[2][i, a, 0, 0] = Nxi[2, ik, a_xi] * Neta[0, il, a_eta]
                    NN[2][i, a, 0, 1] = Nxi[1, ik, a_xi] * Neta[1, il, a_eta]
                    NN[2][i, a, 1, 1] = Nxi[0, ik, a_xi] * Neta[2, il, a_eta]
                    NN[2][i, a, 1, 0] = NN[2][i, a, 0, 1]

    return NN
    
def B_spline_basis3D(degrees, derivative_order, knot_vectors, knots):
    Xi, Eta, Zeta = knot_vectors
    xi, eta, zeta = knots
    p, q, r = degrees
    p1q1r1 = (p + 1) * (q + 1) * (r + 1)
    
    if not hasattr(xi, '__len__'):
        xi = np.array([xi])
    if not hasattr(eta, '__len__'):
        eta = np.array([eta])
    if not hasattr(zeta, '__len__'):
        zeta = np.array([zeta])
    k = len(xi)
    l = len(eta)
    m = len(zeta)
    klm = k * l * m

    NN = []
    for d in range(derivative_order + 1):
        N = np.zeros((klm, p1q1r1, *(3,) * d))
        NN.append(N)

    Nxi = B_spline_basis1D(p, derivative_order, Xi, xi, squeeze=False)
    Neta = B_spline_basis1D(q, derivative_order, Eta, eta, squeeze=False)
    Nzeta = B_spline_basis1D(r, derivative_order, Zeta, zeta, squeeze=False)

    for i in range(klm):
        ik, il, im = split3D(i, (k, l))

        for a in range(p1q1r1):
            a_xi, a_eta, a_zeta = split3D(a, (p + 1, q + 1))

            NN[0][i, a] = Nxi[0, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[0, im, a_zeta]

            if derivative_order > 0:
                NN[1][i, a, 0] = Nxi[1, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[0, im, a_zeta]
                NN[1][i, a, 1] = Nxi[0, ik ,a_xi] * Neta[1, il, a_eta] * Nzeta[0, im, a_zeta]
                NN[1][i, a, 2] = Nxi[0, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[1, im, a_zeta]
                if derivative_order > 1:
                    NN[2][i, a, 0, 0] = Nxi[2, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[0, im, a_zeta]
                    NN[2][i, a, 0, 1] = Nxi[1, ik, a_xi] * Neta[1, il, a_eta] * Nzeta[0, im, a_zeta]
                    NN[2][i, a, 0, 2] = Nxi[1, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[1, im, a_zeta]
                    NN[2][i, a, 1, 0] = NN[2][i, a, 0, 1]
                    NN[2][i, a, 1, 1] = Nxi[0, ik, a_xi] * Neta[2, il, a_eta] * Nzeta[0, im, a_zeta]
                    NN[2][i, a, 1, 2] = Nxi[0, ik, a_xi] * Neta[1, il, a_eta] * Nzeta[1, im, a_zeta]
                    NN[2][i, a, 2, 0] = NN[2][i, a, 0, 2]
                    NN[2][i, a, 2, 1] = NN[2][i, a, 1, 2]
                    NN[2][i, a, 2, 2] = Nxi[0, ik, a_xi] * Neta[0, il, a_eta] * Nzeta[2, im, a_zeta]

    return NN

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

def Piegl_Ex3_4():
    degrees = (3, 2)
    nEls = (4, 5)
    # U = np.array([0, 0, 0, 0, 1/4, 1/2, 3/4, 1, 1, 1, 1])
    U = uniform_knot_vector(degrees[0], nEls[0])
    V = np.array([0, 0, 0, 1/5, 2/5, 3/5, 3/5, 4/5, 1, 1, 1])

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
    ax2 = fig.add_subplot(gs[:, 1], projection='3d')

    Nxi = np.zeros((degrees[0] * nEls[0], n))
    for i, xi in enumerate(xis):
        span = find_knotspan(degrees[0], U, xi)[0]
        el = span - degrees[0]
        Nxi[el:el+degrees[0] + 1, i] = B_spline_basis1D(degrees[0], 0, U, xi)

    ax0.plot(xis, Nxi[xi_idx])
    # for Nxii in Nxi:
    #     ax0.plot(xis, Nxii)

    Neta = np.zeros((degrees[1] * nEls[1], n))
    for j, eta in enumerate(etas):
        span = find_knotspan(degrees[1], V, eta)[0]
        el = span - degrees[1]
        Neta[el:el+degrees[1] + 1, j] = B_spline_basis1D(degrees[1], 0, V, eta)

    ax1.plot(etas, Neta[eta_idx])
    # for Netaj in Neta:
    #     ax1.plot(etas, Netaj)
    
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_zlabel('z [m]')
    ax2.set_xlim3d(left=0, right=1)
    ax2.set_ylim3d(bottom=0, top=1)
    ax2.set_zlim3d(bottom=0, top=1)
    
    ax2.plot_surface(*np.meshgrid(xis, etas), np.outer(Nxi[xi_idx], Neta[eta_idx]))

    plt.show()

if __name__ == '__main__':
    # degrees = (2, 2)
    # nEls = (3, 3)
    # Xi = uniform_knot_vector(degrees[0], nEls[0])
    # Eta = uniform_knot_vector(degrees[1], nEls[1])
    # N, N_xi, N_xixi = B_spline_basis2D(degrees, 2, (Xi, Eta), (0.25, 0.35))
    
    # degrees = (2, 5, 1)
    # nEls = (1, 3, 2)
    # Xi = uniform_knot_vector(degrees[0], nEls[0])
    # Eta = uniform_knot_vector(degrees[1], nEls[1])
    # Zeta = uniform_knot_vector(degrees[2], nEls[2])
    # N, N_xi, N_xixi = B_spline_basis3D(degrees, 2, (Xi, Eta, Zeta), (0.25, 0.35, 0))

    # print(f'')

    Piegl_Ex3_4()