import numpy as np
from math import sqrt


def gauss(nQP, interval=np.array([-1, 1])):
    """Calculate the abscissae and weights of the nQP Gauss-Legendre points on
    an interval [-1,1] and map them onto a given interval [a, b] using linear
    transformation rule.

    See `wiki <https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Variablentransformation_bei_der_Gauß-Quadratur>`_ for details.

    Parameters
    ----------
    nQP : int
        Number of desired quadrature points.
    interval : numpy.ndarray
        Interval [a, b] onto which the Quadrature points should be transformed.
        Default interval is [-1, 1].

    Returns
    -------
    res : tuple(numpy.ndarray, numpy.ndarray)
        Tuple storing array of quadrature points and their corresponding weights.
    """

    gp, wp = np.polynomial.legendre.leggauss(nQP)

    # transfrom gauss points on new interval,
    # see https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Variablentransformation_bei_der_Gauß-Quadratur
    gp = (interval[1] - interval[0]) / 2 * gp + (interval[1] + interval[0]) / 2
    wp = (interval[1] - interval[0]) / 2 * wp

    return gp, wp


def lobatto(nQP, interval=np.array([-1, 1])):
    """Calculate the abscissae and weights of the nQP Gauss-Lobatto points on
    an interval [-1,1] and map them onto a given interval [a, b] using linear
    transformation rule.

    See `wiki <https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Gau%C3%9F-Lobatto-Integration>`_ for details.

    Parameters
    ----------
    nQP : int
        Number of desired quadrature points.
    interval : numpy.ndarray
        Interval [a, b] onto which the gauss points should be transformed.
        Default interval is [-1, 1].

    Returns
    -------
    res : tuple(numpy.ndarray, numpy.ndarray)
        Tuple storing array of quadrature points and their corresponding weights.
    """

    assert nQP >= 2

    if nQP == 2:
        gp = np.array([-1, 1], dtype=float)
        wp = np.array([1, 1], dtype=float)
    elif nQP == 3:
        gp = np.array([-1, 0, 1], dtype=float)
        wp = np.array([1, 4, 1], dtype=float) / 3
    elif nQP == 4:
        sqrt15 = sqrt(1 / 5)
        gp = np.array([-1, -sqrt15, sqrt15, 1], dtype=float)
        wp = np.array([1, 5, 5, 1], dtype=float) / 6
    elif nQP == 5:
        sqrt37 = sqrt(3 / 7)
        gp = np.array([-1, -sqrt37, 0, sqrt37, 1], dtype=float)
        wp = np.array([1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10], dtype=float)
    else:
        raise NotImplementedError(f"lobatto is not implemented for nQP == {nQP}")

    # transfrom gauss points on new interval,
    # see https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Variablentransformation_bei_der_Gauß-Quadratur
    gp = (interval[1] - interval[0]) / 2 * gp + (interval[1] + interval[0]) / 2
    wp = (interval[1] - interval[0]) / 2 * wp

    return gp, wp
