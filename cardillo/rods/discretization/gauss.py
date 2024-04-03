import numpy as np
from scipy.special import roots_legendre


def gauss(n, interval=np.array([-1, 1])):
    """Calculate the abscissae and weights of the n Gauss-Legendre points on
    an interval [-1,1] and map them onto a given interval [a, b] using linear
    transformation rule.

    See `wiki <https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Variablentransformation_bei_der_Gauß-Quadratur>`_ for details.

    Parameters
    ----------
    n : int
        Number of desired quadrature points.
    interval : numpy.ndarray
        Interval [a, b] onto which the Quadrature points should be transformed.
        Default interval is [-1, 1].

    Returns
    -------
    res : tuple(numpy.ndarray, numpy.ndarray)
        Tuple storing array of quadrature points and their corresponding weights.
    """
    points, weights = roots_legendre(n)

    # transfrom gauss points on new interval,
    # see https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Variablentransformation_bei_der_Gauß-Quadratur
    points = (interval[1] - interval[0]) / 2 * points + (interval[1] + interval[0]) / 2
    weights = (interval[1] - interval[0]) / 2 * weights

    return points, weights


def lobatto(n, interval=np.array([-1, 1])):
    """Calculate the abscissae and weights of the n Gauss-Lobatto points on
    an interval [-1,1] and map them onto a given interval [a, b] using linear
    transformation rule.

    See `wiki <https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Gau%C3%9F-Lobatto-Integration>`_ for details.

    Parameters
    ----------
    n : int
        Number of desired quadrature points.
    interval : numpy.ndarray
        Interval [a, b] onto which the gauss points should be transformed.
        Default interval is [-1, 1].

    Returns
    -------
    res : tuple(numpy.ndarray, numpy.ndarray)
        Tuple storing array of quadrature points and their corresponding weights.
    """
    assert n >= 2, "Number of points must be at least 2."

    # get (n-1)th Legendre basis
    basis = np.polynomial.legendre.Legendre.basis(n - 1)

    # initialize points and add endpoints
    points = np.empty(n)
    points[0] = -1
    points[-1] = 1

    # compute the roots of the (n-1)th Legendre polynomial's derivative
    points[1:-1] = basis.deriv().roots()

    # compute the weights
    weights = np.empty(n)
    weights[0] = 2.0 / (n * (n - 1))
    weights[-1] = weights[0]
    weights[1:-1] = 2.0 / (n * (n - 1) * basis(points[1:-1]) ** 2)

    # transfrom gauss points on new interval,
    # see https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Variablentransformation_bei_der_Gauß-Quadratur
    points = (interval[1] - interval[0]) / 2 * points + (interval[1] + interval[0]) / 2
    weights = (interval[1] - interval[0]) / 2 * weights

    return points, weights
