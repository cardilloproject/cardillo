import numpy as np

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
    if derivative:
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

if __name__ == "__main__":
    import numpy as np
    degree = 2
    # x = np.array([-1, 1])
    # x = -1
    # x = 0
    x = 1

    # lx = __lagrange(x, degree)
    # print(f'l({x}): {lx}')

    lx_x = __lagrange_x(x, degree)
    print(f'l_x({x}): {lx_x}')

    lx_x = __lagrange_x_r(x, degree)
    print(f'l_x({x}): {lx_x}')

    lx_xx = __lagrange_xx_r(x, degree)
    print(f'l_xx({x}): {lx_xx}')

    # degree = 1
    # derivative_order = 1
    # x_array = np.array([-1, 0, 1])
    # N, dN = Lagrange_basis(degree, x_array)
    # print(f'N({x_array}):\n{N}')
    # print(f'dN({x_array}):\n{dN}')