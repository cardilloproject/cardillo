import numpy as np

def gauss(nGP, interval=np.array([-1, 1])):
    """Calculate the abscissae and weights of the nG Gauss-Legendre points on an interval [-1,1] and map them onto a given interval [a, b] using linear transformation rule.

    See `wiki <https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Variablentransformation_bei_der_Gauß-Quadratur>`_ for details.

    Parameters
    ----------
    nGP : int
        Number of desired gauss points.
    interval : numpy.ndarray
        Interval [a, b] onto which the gauss points should be transformed. Default interval is [-1, 1].

    Returns
    -------
    res : tuple(numpy.ndarray, numpy.ndarray)
        Tuple storing array of quadrature points and their corresponding weights.
    """

    (gp, wp) = np.polynomial.legendre.leggauss(nGP)

    # transfrom gauss points on new interval, 
    # see https://de.wikipedia.org/wiki/Gau%C3%9F-Quadratur#Variablentransformation_bei_der_Gauß-Quadratur
    gp = (interval[1] - interval[0] ) / 2 * gp + (interval[1] + interval[0] ) / 2
    wp = (interval[1] - interval[0] ) / 2 * wp

    return gp, wp