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

def __lagrange(x, degree):
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
            if m == j:
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

def lagrange_x2(x, degree=1):
    k = degree + 1
    xi = linspace(-1, 1, num=k)
    l = lagrange(x, degree=degree)
    l_x = np.zeros(k)
    for j in range(k):
        coeff_sum = 0
        for i in range(k):
            if i == j:
                continue
            coeff_sum += 1 / (xi[] - xi[i])
        l_x[j] = l[j] * coeff_sum
            
    return l_x

if __name__ == "__main__":
    import numpy as np
    degree = 1
    # x = np.array([-1, 1])
    # x = -1
    x = 0
    # x = 1

    lx = __lagrange(x, degree)
    print(f'l({x}): {lx}')

    lx_x = __lagrange_x(x, degree)
    print(f'l_x({x}): {lx_x}')

    degree = 1
    derivative_order = 1
    x_array = np.array([-1, 0, 1])
    N, dN = Lagrange_basis(degree, x_array)
    print(f'N({x_array}):\n{N}')
    print(f'dN({x_array}):\n{dN}')