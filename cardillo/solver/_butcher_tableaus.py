import numpy as np


class ButcherTableau:
    def __init__(self, A, b, c):
        s1, s2 = A.shape
        assert s1 == s2
        assert s1 == len(b)
        assert len(b) == len(c)

        self.__s = s1
        self.__A = A.copy()
        self.__b = b.copy()
        self.__c = c.copy()

        print(f"A:\n{A}")
        print(f"b: {b}")
        print(f"c: {c}")

    @property
    def A(self):
        return self.__A

    @property
    def b(self):
        return self.__b

    @property
    def c(self):
        return self.__c

    @property
    def s(self):
        return self.__s


class DualSimplecticTableau(ButcherTableau):
    def __init__(
        self,
        butcher_tableau: ButcherTableau,
    ):
        """
        Dual simplectiv Butcher tableau for a given Butcher tableau, see Jay1996.

        References:
        -----------
        Jay1999: https://dl.acm.org/doi/10.1137/0733019
        """
        s = butcher_tableau.s
        A = butcher_tableau.A
        b = butcher_tableau.b
        c = butcher_tableau.c

        assert np.allclose(
            A[0], np.zeros(s)
        ), "Given Butcher tableau requires first explicit stage!"

        # same b vector, see Jay1996 Theorem 2.2
        b_hat = b.copy()

        # same c vector, see Jay1996 Theorem 2.2
        c_hat = c.copy()

        # fmt: off
        A_hat = np.zeros((s, s), dtype=float)
        for i in range(s):
            for j in range(s - 1):
                # Jay1996 Theorem 2.2
                A_hat[i, j] = b[j] * (1 - A[j, i] / b[i])
        # fmt: on

        assert np.allclose(
            A_hat[:, -1], np.zeros(s)
        ), "Dual simplectic Butcher tableau requires last explicit stage!"

        super().__init__(A_hat, b_hat, c_hat)


class GaussLegendre(ButcherTableau):
    def __init__(self, order=4):
        """
        Butcher tableau for Gauss-Legendre quadrature.
        """
        assert order in [4, 6]

        if order == 4:
            s3 = np.sqrt(3)
            # fmt: off
            A = np.array([
                [       1/4, 1/4 - s3/6],
                [1/4 + s3/6,        1/4],
            ], dtype=float)
            # fmt: on
            b = np.array([1 / 2, 1 / 2], dtype=float)
            c = np.array([1 / 2 - s3 / 6, 1 / 2 + s3 / 6], dtype=float)
        elif order == 6:
            raise NotImplementedError
            # fmt: off
            A = np.array([
                [   0,   0,     0],
                [5/24, 1/3, -1/24],
                [ 1/6, 2/3,   1/6],
            ], dtype=float)
            # fmt: on
            b = A[-1, :]
            c = np.array([0, 0.5, 1], dtype=float)
        else:
            raise NotImplementedError

        super().__init__(A, b, c)


class TRX2Tableau(ButcherTableau):
    def __init__(self):
        """
        Butcher tableau for TRX2 as DIRK, see Hosea1996

        References:
        -----------
        Hosea1996: https://doi.org/10.1016/0168-9274(95)00115-8
        """
        c = np.array([0, 1 / 2, 1], dtype=float)
        b = np.array([1 / 4, 1 / 2, 1 / 4], dtype=float)
        A = np.array(
            [
                [0, 0, 0],
                [1 / 4, 1 / 4, 0],
                [1 / 4, 1 / 2, 1 / 4],
            ],
            dtype=float,
        )

        super().__init__(A, b, c)


class TRBDF2Tableau(ButcherTableau):
    def __init__(self):
        """
        Butcher tableau for TR-BDF2 as DIRK, see Hosea1996.

        References:
        -----------
        Hosea1996: https://doi.org/10.1016/0168-9274(95)00115-8
        """
        gamma = 2.0 - np.sqrt(2.0)
        d = 0.5 * gamma
        w = 0.25 * np.sqrt(2)

        # fmt: off
        A = np.array([
            [0.0, 0.0, 0.0],
            [  d,   d, 0.0],
            [  w,   w,   d],
        ], dtype=float)
        # fmt: on
        b = np.array([w, w, d], dtype=float)
        # b_bar = np.array([1.0 - w, 3.0 * w + 1, d], dtype=float) / 3.0
        c = np.array([0.0, gamma, 1.0], dtype=float)

        super().__init__(A, b, c)


class RadauIA3Tableau(ButcherTableau):
    def __init__(self):
        """
        Butcher tableau for Radau IA third order.
        """
        c = np.array([0, 2 / 3], dtype=float)
        b = np.array([1 / 4, 3 / 4], dtype=float)
        # fmt: off
        A = np.array([
            [1 / 4, -1 / 4],
            [1 / 4,  5 / 12]], dtype=float)
        # fmt: on

        super().__init__(A, b, c)


class RadauIIATableau(ButcherTableau):
    def __init__(self, s=2):
        """
        Butcher tableau for s-stage Radau IIA method, see Hairer1999.

        The nodes c_i are computed from Hairer1999 (7) and the coefficients
        a_ij from Hairer1999 (11).

        References:
        -----------
        Hairer1999: https://doi.org/10.1016/S0377-0427(99)00134-X
        """

        # solve zeros of Radau polynomial, see Hairer1999 (7)
        from numpy.polynomial import Polynomial as P

        poly = P([0, 1]) ** (s - 1) * P([-1, 1]) ** s
        poly_der = poly.deriv(s - 1)
        c = poly_der.roots()

        # compute coefficients a_ij, see Hairer1999 (11)
        A = np.zeros((s, s), dtype=float)
        for i in range(s):
            Mi = np.zeros((s, s), dtype=float)
            ri = np.zeros(s, dtype=float)
            for q in range(s):
                Mi[q] = c**q
                ri[q] = c[i] ** (q + 1) / (q + 1)
            A[i] = np.linalg.solve(Mi, ri)

        super().__init__(A, A[-1, :], c)


class LobattoIIIATableau(ButcherTableau):
    def __init__(self, stages=2):
        """
        Butcher tableau for Lobatto IIIA of order 2, 4 and 6, see Hairer1996, Table Table 5.7.

        References:
        -----------
        Hairer1996: https://doi.org/10.1007/978-3-642-05221-7
        """
        assert stages in [2, 3, 4]

        if stages == 2:
            # fmt: off
            A = np.array([
                [0.0, 0.0],
                [0.5, 0.5],
            ], dtype=float)
            # fmt: on
            b = np.array([0.5, 0.5], dtype=float)
            c = np.array([0.0, 1.0], dtype=float)
        elif stages == 3:
            # fmt: off
            A = np.array([
                [   0,   0,     0],
                [5/24, 1/3, -1/24],
                [ 1/6, 2/3,   1/6],
            ], dtype=float)
            # fmt: on
            b = A[-1, :]
            c = np.array([0, 0.5, 1], dtype=float)
        else:
            s5 = np.sqrt(5)
            # fmt: off
            A = np.array([
                [   0,   0,     0, 0],
                [11 + s5, 25 - s5, 25 - 13 * s5, -1 + s5],
                [11 - s5, 25 + 13 * s5, 25 + s5, -1 - s5],
                [10, 50, 50, 10],
            ], dtype=float) / 120
            # fmt: on
            b = A[-1, :]
            c = np.array([0, (5 - s5) / 10, (5 + s5) / 10, 1], dtype=float)

        super().__init__(A, b, c)


def LobattoIIIBTableau(stages=2):
    return DualSimplecticTableau(LobattoIIIATableau(stages))


class LobattoIIICTableau(ButcherTableau):
    def __init__(self, order=2):
        """
        Butcher tableau for Lobatto IIIC of order 2 and 4, see Hairer1996, Table 5.11.

        References:
        -----------
        Hairer1996: https://doi.org/10.1007/978-3-642-05221-7
        """
        assert order in [2, 4]

        if order == 2:
            c = np.array([0.0, 1.0], dtype=float)
            # fmt: off
            A = np.array([
                [0.5, -0.5],
                [0.5,  0.5],
            ], dtype=float)
            # fmt: on
        else:
            c = np.array([0.0, 0.5, 1.0], dtype=float)
            # fmt: off
            A = np.array([
                [2.0, -4.0,  2.0],
                [2.0,  5.0, -1.0],
                [2.0,  8.0,  2.0],
            ], dtype=float) / 12.0
            # fmt: on

        b = A[-1, :]

        super().__init__(A, b, c)


class LobattoIIIDTableau(ButcherTableau):
    def __init__(self, order=2):
        """
        Lobatto IIID family (Nørsett and Wanner, 1981), see Wiki.

        References:
        -----------
        Wiki: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        """
        assert order in [2, 4]

        if order == 2:
            c = np.array([0.0, 1.0], dtype=float)
            b = np.array([0.5, 0.5], dtype=float)
            # fmt: off
            A = np.array([
                [ 0.5, 0.5],
                [-0.5, 0.5],
            ], dtype=float)
            # fmt: on
        else:
            c = np.array([0, 0.5, 1], dtype=float)
            b = np.array([1 / 6, 2 / 3, 1 / 6], dtype=float)
            # fmt: off
            A = np.array([
                [ 1 / 6,      0, -1 / 6],
                [1 / 12, 5 / 12,      0],
                [ 1 / 2,  1 / 3,  1 / 6],
            ], dtype=float)
            # fmt: on

        super().__init__(A, b, c)


class QinZhangTableau(ButcherTableau):
    def __init__(self):
        """
        Qin and Zhang's two-stage, 2nd order, symplectic Diagonally Implicit Runge–Kutta method, see Wiki.

        References:
        -----------
        Wiki: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        """
        c = np.array([0.25, 0.75], dtype=float)
        b = np.array([0.5, 0.5], dtype=float)
        # fmt: off
        A = np.array([
            [0.25,  0.0],
            [ 0.5, 0.25],
        ], dtype=float)
        # fmt: on

        super().__init__(A, b, c)


class AlexanderTableau(ButcherTableau):
    def __init__(self, order=2):
        """
        Butcher tableau for DIRK methods of Alexander, see Alexander1977 and Hosea1996.

        References:
        -----------
        Alexander1977: https://doi.org/10.1137/0714068 \\
        Hosea1996: https://doi.org/10.1016/0168-9274(95)00115-8
        """
        if order == 2:
            d = 1 - np.sqrt(2) / 2  # ensures c0 <= 1
            # d = 1 + np.sqrt(2) / 2
            c = np.array([d, 1], dtype=float)
            # fmt: off
            A = np.array([
                [      d, 0.0],
                [1.0 - d,   d],
            ], dtype=float)
            # fmt: on
        elif order == 3:
            # solve zeros of polynomial alpha^3 - 3 alpha + 3/2 alpha - 1/6
            from numpy.polynomial import Polynomial as P

            poly = P([-1 / 6, 3 / 2, -3, 1])
            alphas = poly.roots()
            print(f"alphas: {alphas}")
            alpha2 = alphas[0]  # all A's get positive

            tau = (1 + alpha2) / 2
            b1 = -(6 * alpha2**2 - 16 * alpha2 + 1) / 4
            b2 = (6 * alpha2**2 - 20 * alpha2 + 5) / 4
            c = np.array([alpha2, tau, 1], dtype=float)
            # fmt: off
            A = np.array([
                [      alpha2,     0,       0],
                [tau - alpha2, alpha2,      0],
                [          b1,     b2, alpha2],
            ], dtype=float)
            # fmt: on
        else:
            raise NotImplementedError

        b = A[-1]

        super().__init__(A, b, c)


class KraaijevangerSpijkerTableau(ButcherTableau):
    def __init__(self):
        """
        Kraaijevanger and Spijker's two-stage Diagonally Implicit Runge–Kutta method, see Wiki.

        References:
        -----------
        Wiki: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        """
        c = np.array([0.5, 1.5], dtype=float)
        b = np.array([-0.5, 1.5], dtype=float)
        # fmt: off
        A = np.array([
            [ 0.5, 0.0],
            [-0.5, 2.0],
        ], dtype=float)
        # fmt: on

        super().__init__(A, b, c)
