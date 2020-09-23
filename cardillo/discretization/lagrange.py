import numpy as np
from scipy.sparse.linalg import spsolve
from cardillo.discretization.indexing import flat2D, flat3D, split2D, split3D
import meshio

def lagrange_basis1D(degree, xi, derivative=1):
    p = degree

    if not hasattr(xi, '__len__'):
        xi = np.array([xi])

    k = len(xi)
    n = sum([1 for d in range(derivative + 1)])
    NN = np.zeros((k, p+1, n))
    Nxi, N_xi = Lagrange_basis(p, xi, derivative=True)

    NN[:, :, 0] = Nxi
    if derivative > 0:
        NN[:, :, 1] = N_xi
        if derivative > 1:
            NN[:, :, 2] = N_xixi

    return NN

def lagrange_basis2D(degrees, xis, derivative=1):
    p, q = degrees
    xi, eta = xis
    p1q1 = (p + 1) * (q + 1)

    if not hasattr(xi, '__len__'):
        xi = np.array([xi])
    if not hasattr(eta, '__len__'):
        eta = np.array([eta])

    k = len(xi)
    l = len(eta)
    kl = k * l
 
    n = sum([2**d for d in range(derivative + 1)])
    NN = np.zeros((kl, p1q1, n))
    #TODO: make seperate 1D Basis function with second derrivative
    Nxi = lagrange_basis1D(p, xi)
    Neta = lagrange_basis1D(q, eta)

    for i in range(kl):
        ik, il = split2D(i, (k, ))

        for a in range(p1q1):
            a_xi, a_eta = split2D(a, (p + 1, ))
            NN[i, a, 0] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 0]

            if derivative > 0:
                NN[i, a, 1] = Nxi[ik, a_xi, 1] * Neta[il, a_eta, 0] 
                NN[i, a, 2] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 1] 
                if derivative > 1:
                    raise NotImplementedError('...')
                    NN[i, a, 3] = Nxi[ik, a_xi, 2] * Neta[il, a_eta, 0]
                    NN[i, a, 4] = Nxi[ik, a_xi, 1] * Neta[il, a_eta, 1]
                    NN[i, a, 6] = NN[i, a, 4]
                    NN[i, a, 5] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 2] 

    return NN

def lagrange_basis3D(degrees, xis, derivative=1):
    p, q, r = degrees
    xi, eta, zeta = xis
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
 
    n = sum([3**d for d in range(derivative + 1)])
    NN = np.zeros((klm, p1q1r1, n))
    #TODO: make seperate 1D Basis function with second derrivative
    Nxi = lagrange_basis1D(p, xi)
    Neta = lagrange_basis1D(q, eta)
    Nzeta = lagrange_basis1D(r, zeta)

    for i in range(klm):
        ik, il, im = split3D(i, (k, l))

        for a in range(p1q1r1):
            a_xi, a_eta, a_zeta = split3D(a, (p + 1, q + 1))
            NN[i, a, 0] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 0] * Nzeta[im, a_zeta, 0]

            if derivative > 0:
                NN[i, a, 1] = Nxi[ik, a_xi, 1] * Neta[il, a_eta, 0] * Nzeta[im, a_zeta, 0]
                NN[i, a, 2] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 1] * Nzeta[im, a_zeta, 0]
                NN[i, a, 3] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 0] * Nzeta[im, a_zeta, 1]
                if derivative > 1:
                    raise NotImplementedError('...')
                    NN[i, a, 4] = Nxi[ik, a_xi, 2] * Neta[il, a_eta, 0] * Nzeta[im, a_zeta, 0]
                    NN[i, a, 5] = Nxi[ik, a_xi, 1] * Neta[il, a_eta, 1] * Nzeta[im, a_zeta, 0]
                    NN[i, a, 6] = Nxi[ik, a_xi, 1] * Neta[il, a_eta, 0] * Nzeta[im, a_zeta, 1]
                    NN[i, a, 7] = NN[i, a, 5]
                    NN[i, a, 8] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 2] * Nzeta[im, a_zeta, 0]
                    NN[i, a, 9] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 1] * Nzeta[im, a_zeta, 1]
                    NN[i, a, 10] = NN[i, a, 6]
                    NN[i, a, 11] = NN[i, a, 7]
                    NN[i, a, 12] = Nxi[ik, a_xi, 0] * Neta[il, a_eta, 0] * Nzeta[im, a_zeta, 2]

    return NN

def Lagrange_basis(degree, x, derivative=True):
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
    if not hasattr(x, '__len__'):
        x = [x]
    nx = len(x)
    N = np.zeros((nx, degree + 1))
    for i, xi in enumerate(x):
        N[i] = __lagrange(xi, degree)
    if derivative == True:
        dN = np.zeros((nx, degree + 1))
        for i, xi in enumerate(x):
            dN[i] = __lagrange_x(xi, degree)
        return N, dN
    else:
        return N

def __lagrange(x, degree, skip=[]):
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
    xi = np.linspace(-1, 1, num=k)
    l = np.ones(k)
    for j in range(k):
        for m in range(k):
            if m == j or m in skip:
                continue
            l[j] *= (x - xi[m]) / (xi[j] - xi[m])

    return l

def __lagrange_x(x, degree):
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
    xi = np.linspace(-1, 1, num=k)
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
    """Recursive formular for first derivative of Lagrange shape functions.
    """
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
    """Recursive formular for second derivative of Lagrange shape functions.
    """
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
        binary=binary
    )
def find_element_number(mesh, xis):
    # finds the element number for a xis vector from the 0 to 1 parameter space
    # also gives the parameter space value of xis
    el = np.zeros(len(xis))
    xis_l = np.zeros_like(xis)
    for i, xi in enumerate(xis):
        if int((xi // (1 / mesh.element_shape[i]))) >= mesh.element_shape[i]:
            el[i] = mesh.element_shape[i] - 1
            xis_l[i] = 1
        else:
            el[i] = int((xi // (1 / mesh.element_shape[i])))
            xis_l[i] = (xis[i]  * mesh.element_shape[i] - el[i]) * 2 - 1
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

def test_shape_functions_der():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    x = np.linspace (-1,1,10)
    y = x
    z = x
    NN = lagrange_basis1D(2, x, derivative=1)

    plt.plot(x,NN[:10,0,0])
    plt.plot(x,NN[:10,1,0])
    plt.plot(x,NN[:10,2,0])
    plt.plot(x,NN[:10,0,1])
    plt.plot(x,NN[:10,1,1])
    plt.plot(x,NN[:10,2,1])
    plt.show()

def test_fit_lagrange_volume():
    # degrees = np.ones(3, dtype=int) * 3
    degrees = (3, 3, 3)
    QP_shape = (1, 1, 1)
    element_shape = np.ones(3, dtype=int) * 5
    element_shape = (3, 3, 3)
    
    from cardillo.discretization.mesh3D_lagrange import Mesh3D_lagrange
    mesh = Mesh3D_lagrange(degrees, QP_shape, element_shape, derivative_order=0, nq_n=3)

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

    def sherical_dome(xi, eta, zeta, phi0=np.pi, theta0=np.pi/2, R=1, H=1):
        phi = (1 - xi) * phi0
        theta = eta * theta0
        r = R + zeta * H
        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(theta)
        return x, y, z

    def parabolic(xi, eta, zeta, L=1, B=1, H=1):
        x = xi * L
        y = eta * B + (xi - L/2)**2 * eta
        z = zeta * H
        return x, y, z

    def twist(xi, eta, zeta, phi0=np.pi/2, R=1, d=1, B=1, H=1):
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
    xis = np.zeros((n3, 3))
    Pw = np.zeros((n3, 3))
    for i, xii in enumerate(xi):
        for j, etai in enumerate(eta):
            for k, zetai in enumerate(zeta):
                idx = flat3D(i, j, k, (nxi, neta, nzeta))
                xis[idx] = xii, etai, zetai
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
    X, Y, Z = fit_lagrange_volume(mesh, xis, Pw, qc, cDOF)

    lagrange_volume2vtk(mesh, np.concatenate((X, Y, Z)), 'fit_lagrange_volume.vtu')
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*Pw.T)
    ax.scatter(X, Y, Z, color='red')
    # RB = R + 0.5*B
    # # RB = R
    # ax.set_xlim(-RB, RB)
    # ax.set_ylim(-RB, RB)
    # ax.set_zlim(-RB, RB)
    plt.show()


if __name__ == "__main__":
    import numpy as np



    #test_shape_functions_der()

    test_fit_lagrange_volume()

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