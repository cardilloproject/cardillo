import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, eye, diags, kron
from tqdm import tqdm
import numpy.typing as npt

from cardillo.math.prox import prox_R0_nm, prox_R0_np, prox_sphere
from cardillo.math import approx_fprime, fsolve, mngn
from cardillo.solver import Solution, consistent_initial_conditions


class ButcherTableau:
    def __init__(
        self,
        A: npt.NDArray[np.float_],
        b: npt.NDArray[np.float_],
        c: npt.NDArray[np.float_],
    ):
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
    def __init__(self, order=2):
        """
        Butcher tableau for Lobatto IIIA of order 2, 4 and 6, see Hairer1996, Table Table 5.7.

        References:
        -----------
        Hairer1996: https://doi.org/10.1007/978-3-642-05221-7
        """
        assert order in [2, 4, 6]

        if order == 2:
            # fmt: off
            A = np.array([
                [0.0, 0.0],
                [0.5, 0.5],
            ], dtype=float)
            # fmt: on
            b = np.array([0.5, 0.5], dtype=float)
            c = np.array([0.0, 1.0], dtype=float)
        elif order == 4:
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


class LobattoIIIBTableau(ButcherTableau):
    def __init__(self, order=2):
        """
        Butcher tableau for Lobatto IIIB of order 2 and 4, see Hairer1996, Table Table 5.9.

        References:
        -----------
        Hairer1996: https://doi.org/10.1007/978-3-642-05221-7
        """
        assert order in [2, 4]

        if order == 2:
            # fmt: off
            A = np.array([
                [0.5, 0.0],
                [0.5, 0.0],
            ], dtype=float)
            # fmt: on
            b = np.array([0.5, 0.5], dtype=float)
            c = np.array([0.0, 1.0], dtype=float)
        else:
            # fmt: off
            A = np.array([
                [1/6, -1/6, 0],
                [1/6,  1/3, 0],
                [1/6,  5/3, 0],
            ], dtype=float)
            # fmt: on
            b = np.array([1 / 6, 2 / 3, 1 / 6], dtype=float)
            c = np.array([0, 0.5, 1], dtype=float)

        super().__init__(A, b, c)


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


# TODO:
# - analytical Jacobian
# - uncouple projection stage
# - implement fixed point iteration as done in rattle
class NonsmoothPIRK:
    def __init__(
        self,
        system,
        t1,
        h,
        butcher_tableau,
        atol=1e-8,
        max_iter=50,
    ):
        """Nonsmooth projected implicit Runge-Kutta (N-PIRK) methods.
        See Ascher1991, Haierer1996 p. 502-503, Hairer2006 Section IV.4.

        References:
        -----------
        Ascher1991: https://www.jstor.org/stable/2157789 \\
        Hairer1996: https://doi.org/10.1007/978-3-642-05221-7 \\
        Hairer2006: https://doi.org/10.1007/3-540-30666-8
        """
        # unpack butcher tableau for positions
        self.butcher_tableau = butcher_tableau
        self.stages = butcher_tableau.s
        self.A = butcher_tableau.A
        self.b = butcher_tableau.b
        self.c = butcher_tableau.c

        # exclude explicit first stage
        assert not np.allclose(self.A[0], np.zeros(self.stages))

        # ensure stiffly accurate method
        assert np.allclose(self.A[-1], self.b)

        self.system = system

        # integration time
        t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.h = h
        self.t = np.arange(t0, self.t1 + self.h, self.h)

        self.atol = atol
        self.max_iter = max_iter

        self.nq = system.nq
        self.nu = system.nu
        self.nla_g = system.nla_g
        self.nla_gamma = system.nla_gamma
        self.nla_N = system.nla_N
        self.nla_F = system.nla_F

        # consistent initial conditions
        (
            self.tn,
            self.qn,
            self.un,
            self.q_dotn,
            self.u_dotn,
            self.la_gn,
            self.la_gamman,
        ) = consistent_initial_conditions(system)

        self.la_Nn = system.la_N0
        self.la_Fn = system.la_F0

        # TODO: How do we initialized contact forces/ percussions?
        self.yn = np.concatenate(
            (
                0 * np.tile(self.q_dotn, self.stages),
                0 * np.tile(self.u_dotn, self.stages),
                0 * np.tile(self.la_gn, self.stages),
                0 * np.tile(self.la_gamman, self.stages),
                0 * np.tile(self.la_Nn, self.stages),
                0 * np.tile(self.la_Fn, self.stages),
                # np.outer(self.q_dotn, self.b).reshape(-1, order="F"),
                # np.outer(self.u_dotn, self.b).reshape(-1, order="F"),
                # np.outer(self.la_gn, self.b).reshape(-1, order="F"),
                # np.outer(self.la_gamman, self.b).reshape(-1, order="F"),
                # np.outer(self.la_Nn, self.b).reshape(-1, order="F"),
                # np.outer(self.la_Fn, self.b).reshape(-1, order="F"),
                0 * self.un,
                0 * self.la_gn,
                0 * self.la_gamman,
                0 * self.la_Nn,
                0 * self.la_Fn,
            )
        )
        self.ny = len(self.yn)

        self.split_y = np.cumsum(
            np.array(
                [
                    self.stages * self.nq,
                    self.stages * self.nu,
                    self.stages * self.nla_g,
                    self.stages * self.nla_gamma,
                    self.stages * self.nla_N,
                    self.stages * self.nla_F,
                    self.nu,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_N,
                ],
                dtype=int,
            )
        )

        self.I_N = np.zeros((self.stages, self.nla_N), dtype=bool)
        self.A_N = np.zeros(self.nla_N, dtype=bool)
        self.B_N = np.zeros(self.nla_N, dtype=bool)

    def unpack_reshape(self, y):
        """Unpack and reshape vector of unknowns and return arrays of unkowns
        for each stage + projection quantities.
        """
        (
            dq,
            du,
            dP_g,
            dP_gamma,
            dP_N,
            dP_F,
            Delta_U,
            Delta_P_g,
            Delta_P_gamma,
            Delta_P_N,
            Delta_P_F,
        ) = np.array_split(y, self.split_y)

        dq = dq.reshape(-1, self.stages, order="F")
        du = du.reshape(-1, self.stages, order="F")
        dP_g = dP_g.reshape(-1, self.stages, order="F")
        dP_gamma = dP_gamma.reshape(-1, self.stages, order="F")
        dP_N = dP_N.reshape(-1, self.stages, order="F")
        dP_F = dP_F.reshape(-1, self.stages, order="F")

        return (
            dq,
            du,
            dP_g,
            dP_gamma,
            dP_N,
            dP_F,
            Delta_U,
            Delta_P_g,
            Delta_P_gamma,
            Delta_P_N,
            Delta_P_F,
        )

    def R(self, yn1, update_index=False):
        h = self.h
        tn = self.tn
        tn1 = tn + h

        # previous time step
        qn = self.qn
        un = self.un

        # current time step and intermediate stages
        (
            dq,
            du,
            dP_g,
            dP_gamma,
            dP_N,
            dP_F,
            Delta_U,
            Delta_P_g,
            Delta_P_gamma,
            Delta_P_N,
            Delta_P_F,
        ) = self.unpack_reshape(yn1)

        # quadrature for position and velocity
        # qn1 = qn + h * dq @ self.b
        # un1 = un + h * du @ self.b + Delta_U
        qn1 = qn + dq @ self.b
        un1 = un + du @ self.b + Delta_U

        # quadrature for percussions
        # P_gn1 = h * dP_g @ self.b + Delta_P_g
        # P_gamman1 = h * dP_gamma @ self.b + Delta_P_gamma
        # P_Nn1 = h * dP_N @ self.b + Delta_P_N
        # P_Fn1 = h * dP_F @ self.b + Delta_P_F
        P_gn1 = dP_g @ self.b + Delta_P_g
        P_gamman1 = dP_gamma @ self.b + Delta_P_gamma
        P_Nn1 = dP_N @ self.b + Delta_P_N
        P_Fn1 = dP_F @ self.b + Delta_P_F

        # initialize residual
        R = np.zeros(self.ny, dtype=yn1.dtype)
        # R = yn1.copy()  # TODO: Remove this!

        #####################
        # kinematic equations
        #####################
        for i in range(self.stages):
            ti = tn + self.c[i] * h
            # Qi = qn + h * dq @ self.A[i]
            # Ui = un + h * du @ self.A[i]
            Qi = qn + dq @ self.A[i]
            Ui = un + du @ self.A[i]
            # R[i * self.nq : (i + 1) * self.nq] = dq[:, i] - self.system.q_dot(
            #     ti, Qi, Ui
            # )
            R[i * self.nq : (i + 1) * self.nq] = dq[:, i] - h * self.system.q_dot(
                ti, Qi, Ui
            )

        ####################
        # eqations of motion
        ####################
        for i in range(self.stages):
            ti = tn + self.c[i] * h
            # Qi = qn + h * dq @ self.A[i]
            # Ui = un + h * du @ self.A[i]
            Qi = qn + dq @ self.A[i]
            Ui = un + du @ self.A[i]
            R[self.split_y[0] + i * self.nu : self.split_y[0] + (i + 1) * self.nu] = (
                self.system.M(ti, Qi, scipy_matrix=csr_matrix) @ du[:, i]
                # - self.system.h(ti, Qi, Ui)
                - h * self.system.h(ti, Qi, Ui)
                - self.system.W_g(ti, Qi) @ dP_g[:, i]
                - self.system.W_gamma(ti, Qi) @ dP_gamma[:, i]
                - self.system.W_N(ti, Qi) @ dP_N[:, i]
                - self.system.W_F(ti, Qi) @ dP_F[:, i]
            )

        #########################################
        # bilateral constraints on position level
        #########################################
        for i in range(self.stages):
            ti = tn + self.c[i] * h
            # Qi = qn + h * dq @ self.A[i]
            Qi = qn + dq @ self.A[i]
            R[
                self.split_y[1]
                + i * self.nla_g : self.split_y[1]
                + (i + 1) * self.nla_g
            ] = self.system.g(ti, Qi)

        #########################################
        # bilateral constraints on velocity level
        #########################################
        for i in range(self.stages):
            ti = tn + self.c[i] * h
            # Qi = qn + h * dq @ self.A[i]
            # Ui = un + h * du @ self.A[i]
            Qi = qn + dq @ self.A[i]
            Ui = un + du @ self.A[i]
            R[
                self.split_y[2]
                + i * self.nla_gamma : self.split_y[2]
                + (i + 1) * self.nla_gamma
            ] = self.system.gamma(ti, Qi, Ui)

        ###########
        # Signorini
        ###########
        prox_r_N = self.prox_r_N
        for i in range(self.stages):
            ti = tn + self.c[i] * h
            # Qi = qn + h * dq @ self.A[i]
            # Ui = un + h * du @ self.A[i]
            # P_Ni = h * dP_N @ self.A[i]
            Qi = qn + dq @ self.A[i]
            Ui = un + du @ self.A[i]
            P_Ni = dP_N @ self.A[i]

            g_Ni = self.system.g_N(ti, Qi)

            prox_arg = g_Ni - prox_r_N * P_Ni
            if update_index:
                self.I_N[i] = prox_arg <= 0.0

            R[
                self.split_y[3]
                + i * self.nla_N : self.split_y[3]
                + (i + 1) * self.nla_N
            ] = np.where(
                self.I_N[i],
                g_Ni,
                P_Ni,
            )

        ##################
        # Coulomb friction
        ##################
        prox_r_F = self.prox_r_F
        mu = self.system.mu
        for i in range(self.stages):
            ti = tn + self.c[i] * h
            # Qi = qn + h * dq @ self.A[i]
            # Ui = un + h * du @ self.A[i]
            # P_Ni = h * dP_N @ self.A[i]
            # P_Fi = h * dP_F @ self.A[i]
            Qi = qn + dq @ self.A[i]
            Ui = un + du @ self.A[i]
            P_Ni = dP_N @ self.A[i]
            P_Fi = dP_F @ self.A[i]

            gamma_Fi = self.system.gamma_F(ti, Qi, Ui)

            for i_N, i_F in enumerate(self.system.NF_connectivity):
                i_F = np.array(i_F)
                n_F = len(i_F)
                if n_F > 0:
                    arg_F = prox_r_F[i_F] * gamma_Fi[i_F] - P_Fi[i_F]
                    norm_arg_F = np.linalg.norm(arg_F)
                    radius = mu[i_N] * P_Ni[i_N]

                    # TODO: Investigate why strict smaller is important!
                    if norm_arg_F < radius:
                        R[self.split_y[4] + i * self.nla_F + i_F] = gamma_Fi[i_F]
                    else:
                        if norm_arg_F > 0:
                            R[self.split_y[4] + i * self.nla_F + i_F] = (
                                P_Fi[i_F] + radius * arg_F / norm_arg_F
                            )
                        else:
                            R[self.split_y[4] + i * self.nla_F + i_F] = (
                                P_Fi[i_F] + radius * arg_F
                            )

        #################
        # impact equation
        #################
        # # nonlinear projection with h vector
        # R[self.split_y[5] : self.split_y[6]] = (
        #     self.system.M(tn1, qn1) @ (un1 - un)
        #     - h * self.system.h(tn1, qn1, un1)
        #     - self.system.W_g(tn1, qn1) @ P_gn1
        #     - self.system.W_gamma(tn1, qn1) @ P_gamman1
        #     - self.system.W_N(tn1, qn1) @ P_Nn1
        #     - self.system.W_F(tn1, qn1) @ P_Fn1
        # )

        # simple linear projection
        R[self.split_y[5] : self.split_y[6]] = (
            self.system.M(tn1, qn1) @ Delta_U
            - self.system.W_g(tn1, qn1) @ Delta_P_g
            - self.system.W_gamma(tn1, qn1) @ Delta_P_gamma
            - self.system.W_N(tn1, qn1) @ Delta_P_N
            - self.system.W_F(tn1, qn1) @ Delta_P_F
        )

        #################################
        # impulsive bilateral constraints
        #################################
        R[self.split_y[6] : self.split_y[7]] = self.system.g_dot(tn1, qn1, un1)
        R[self.split_y[7] : self.split_y[8]] = self.system.gamma(tn1, qn1, un1)

        ##################################################
        # mixed Singorini on velocity level and impact law
        ##################################################
        xi_Nn1 = self.system.xi_N(tn1, qn1, un, un1)
        prox_arg = xi_Nn1 - prox_r_N * P_Nn1
        if update_index:
            self.A_N = np.any(self.I_N, axis=0)
            self.B_N = self.A_N * (prox_arg <= 0)

        R[self.split_y[8] : self.split_y[9]] = np.where(
            self.B_N,
            xi_Nn1,
            P_Nn1,
        )

        # R[self.split_y[8] : self.split_y[9]] = np.where(
        #     np.any(self.I, axis=0),
        #     xi_Nn1 - prox_R0_np(xi_Nn1 - prox_r_N * P_Nn1),
        #     # P_Nn1 + prox_R0_nm(prox_r_N * xi_Nn1 - P_Nn1),
        #     P_Nn1,
        # )

        ##################################################
        # mixed Coulomb friction and tangent impact law
        ##################################################
        xi_Fn1 = self.system.xi_F(tn1, qn1, un, un1)

        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)
            n_F = len(i_F)
            if n_F > 0:
                arg_F = prox_r_F[i_F] * xi_Fn1[i_F] - P_Fn1[i_F]
                norm_arg_F = np.linalg.norm(arg_F)
                radius = mu[i_N] * P_Nn1[i_N]

                # TODO: Investigate why strict smaller is important!
                if norm_arg_F < radius:
                    R[self.split_y[9] + i_F] = xi_Fn1[i_F]
                else:
                    if norm_arg_F > 0:
                        R[self.split_y[9] + i_F] = (
                            P_Fn1[i_F] + radius * arg_F / norm_arg_F
                        )
                    else:
                        R[self.split_y[9] + i_F] = P_Fn1[i_F] + radius * arg_F

        # save integrated solutions
        # TODO: This is inefficient; unpack yn1 if it is converged.
        self.tn1 = tn1
        self.qn1 = qn1.copy()
        self.un1 = un1.copy()
        self.P_gn1 = P_gn1.copy()
        self.P_gamman1 = P_gamman1.copy()
        self.P_Nn1 = P_Nn1.copy()
        self.P_Fn1 = P_Fn1.copy()

        return R

    def J(self, yn1, *args, **kwargs):
        h = self.h
        tn = self.tn
        tn1 = tn + h

        # previous time step
        qn = self.qn
        un = self.un

        # current time step and intermediate stages
        (
            dq,
            du,
            dP_g,
            dP_gamma,
            dP_N,
            dP_F,
            Delta_U,
            Delta_P_g,
            Delta_P_gamma,
            Delta_P_N,
            Delta_P_F,
        ) = self.unpack_reshape(yn1)

        # quadrature for position and velocity
        qn1 = qn + dq @ self.b
        un1 = un + du @ self.b + Delta_U

        # quadrature for percussions
        # # TODO: Are they required?
        # P_gn1 = dP_g @ self.b + Delta_P_g
        # P_gamman1 = dP_gamma @ self.b + Delta_P_gamma
        P_Nn1 = dP_N @ self.b + Delta_P_N
        P_Fn1 = dP_F @ self.b + Delta_P_F

        J = lil_matrix((self.ny, self.ny))

        #####################
        # kinematic equations
        #####################
        for i in range(self.stages):
            ti = tn + self.c[i] * h
            Qi = qn + dq @ self.A[i]
            Ui = un + du @ self.A[i]

            idxi = i * self.nq
            idxi1 = (i + 1) * self.nq

            J[idxi:idxi1, idxi:idxi1] = eye(self.nq)
            J[idxi:idxi1, : self.split_y[0]] += kron(
                -h * self.A[i], self.system.q_dot_q(ti, Qi, Ui)
            )

            J[idxi:idxi1, self.split_y[0] : self.split_y[1]] = kron(
                -h * self.A[i], self.system.B(ti, Qi)
            )

        ####################
        # eqations of motion
        ####################
        for i in range(self.stages):
            ti = tn + self.c[i] * h
            Qi = qn + dq @ self.A[i]
            Ui = un + du @ self.A[i]

            idxi = self.split_y[0] + i * self.nu
            idxi1 = self.split_y[0] + (i + 1) * self.nu

            J[idxi:idxi1, : self.split_y[0]] = kron(
                self.A[i],
                self.system.Mu_q(ti, Qi, du[:, i])
                - h * self.system.h_q(ti, Qi, un1)
                - self.system.Wla_g_q(ti, Qi, dP_g[:, i])
                - self.system.Wla_gamma_q(ti, Qi, dP_gamma[:, i])
                - self.system.Wla_N_q(ti, Qi, dP_N[:, i])
                - self.system.Wla_F_q(ti, Qi, dP_F[:, i]),
            )

            J[idxi:idxi1, idxi:idxi1] = self.system.M(ti, Qi)
            J[idxi:idxi1, self.split_y[0] : self.split_y[1]] += kron(
                -h * self.A[i], self.system.h_u(ti, Qi, Ui)
            )

            J[
                idxi:idxi1,
                self.split_y[1]
                + i * self.nla_g : self.split_y[1]
                + (i + 1) * self.nla_g,
            ] = -self.system.W_g(ti, Qi)
            J[
                idxi:idxi1,
                self.split_y[2]
                + i * self.nla_gamma : self.split_y[2]
                + (i + 1) * self.nla_gamma,
            ] = -self.system.W_gamma(ti, Qi)
            J[
                idxi:idxi1,
                self.split_y[3]
                + i * self.nla_N : self.split_y[3]
                + (i + 1) * self.nla_N,
            ] = -self.system.W_N(ti, Qi)
            J[
                idxi:idxi1,
                self.split_y[4]
                + i * self.nla_F : self.split_y[4]
                + (i + 1) * self.nla_F,
            ] = -self.system.W_F(ti, Qi)

        #########################################
        # bilateral constraints on position level
        #########################################
        for i in range(self.stages):
            ti = tn + self.c[i] * h
            Qi = qn + dq @ self.A[i]

            idxi = self.split_y[1] + i * self.nla_g
            idxi1 = self.split_y[1] + (i + 1) * self.nla_g

            J[idxi:idxi1, : self.split_y[0]] = kron(self.A[i], self.system.g_q(ti, Qi))

        #########################################
        # bilateral constraints on velocity level
        #########################################
        for i in range(self.stages):
            ti = tn + self.c[i] * h
            Qi = qn + dq @ self.A[i]
            Ui = un + du @ self.A[i]

            idxi = self.split_y[2] + i * self.nla_gamma
            idxi1 = self.split_y[2] + (i + 1) * self.nla_gamma

            J[idxi:idxi1, : self.split_y[0]] = kron(
                self.A[i], self.system.gamma_q(ti, Qi, Ui)
            )

            J[idxi:idxi1, self.split_y[0] : self.split_y[1]] = kron(
                self.A[i], self.system.W_gamma(ti, Qi).T
            )

        ###########
        # Signorini
        ###########
        for i in range(self.stages):
            ti = tn + self.c[i] * h
            Qi = qn + dq @ self.A[i]

            idxi = self.split_y[3] + i * self.nla_N
            idxi1 = self.split_y[3] + (i + 1) * self.nla_N

            J[idxi:idxi1, : self.split_y[0]] = kron(
                self.A[i],
                diags(np.asarray(self.I_N[i], dtype=float)) @ self.system.g_N_q(ti, Qi),
            )
            J[idxi:idxi1, self.split_y[3] : self.split_y[4]] = kron(
                self.A[i], diags(np.asarray(~self.I_N[i], dtype=float))
            )

        ##################
        # Coulomb friction
        ##################
        prox_r_F = self.prox_r_F
        mu = self.system.mu

        for i in range(self.stages):
            ti = tn + self.c[i] * h
            Qi = qn + dq @ self.A[i]
            Ui = un + du @ self.A[i]
            P_Ni = dP_N @ self.A[i]
            P_Fi = dP_F @ self.A[i]

            gamma_Fi = self.system.gamma_F(ti, Qi, Ui)
            gamma_Fi_q = self.system.gamma_F_q(ti, Qi, Ui, scipy_matrix=csr_matrix)
            gamma_Fi_u = self.system.W_F(ti, Qi, scipy_matrix=csc_matrix).T

            # local derivatives that have to be distributed with kron
            Rla_F_q = lil_matrix((self.nla_F, self.nq))
            Rla_F_u = lil_matrix((self.nla_F, self.nu))
            Rla_F_la_N = lil_matrix((self.nla_F, self.nla_N))
            Rla_F_la_F = lil_matrix((self.nla_F, self.nla_F))

            for i_N, i_F in enumerate(self.system.NF_connectivity):
                i_F = np.array(i_F)
                n_F = len(i_F)
                if n_F > 0:
                    arg_F = prox_r_F[i_F] * gamma_Fi[i_F] - P_Fi[i_F]
                    norm_arg_F = np.linalg.norm(arg_F)
                    radius = mu[i_N] * P_Ni[i_N]

                    # TODO: Investigate why strict smaller is important!
                    if norm_arg_F < radius:
                        # print(f"stick")
                        Rla_F_q[i_F] = gamma_Fi_q[i_F]
                        Rla_F_u[i_F] = gamma_Fi_u[i_F]
                    else:
                        if norm_arg_F > 0:
                            # print(f"slip norm_arg_F > 0")
                            slip_dir = arg_F / norm_arg_F
                            factor = (
                                np.eye(n_F) - np.outer(slip_dir, slip_dir)
                            ) / norm_arg_F
                            Rla_F_q[i_F] = (
                                radius * factor @ diags(prox_r_F[i_F]) @ gamma_Fi_q[i_F]
                            )
                            Rla_F_u[i_F] = (
                                radius * factor @ diags(prox_r_F[i_F]) @ gamma_Fi_u[i_F]
                            )
                            Rla_F_la_N[i_F[:, None], i_N] = mu[i_N] * slip_dir
                            Rla_F_la_F[i_F[:, None], i_F] = (
                                np.eye(n_F) - radius * factor
                            )
                        else:
                            # print(f"slip norm_arg_F = 0")
                            Rla_F_q[i_F] = (
                                radius * diags(prox_r_F[i_F]) @ gamma_Fi_q[i_F]
                            )
                            Rla_F_u[i_F] = (
                                radius * diags(prox_r_F[i_F]) @ gamma_Fi_u[i_F]
                            )
                            Rla_F_la_N[i_F[:, None], i_N] = mu[i_N] * arg_F
                            Rla_F_la_F[i_F[:, None], i_F] = (1 - radius) * eye(n_F)

            idxi = self.split_y[4] + i * self.nla_F
            idxi1 = self.split_y[4] + (i + 1) * self.nla_F

            J[idxi:idxi1, : self.split_y[0]] = kron(self.A[i], Rla_F_q)
            J[idxi:idxi1, self.split_y[0] : self.split_y[1]] = kron(self.A[i], Rla_F_u)
            J[idxi:idxi1, self.split_y[3] : self.split_y[4]] = kron(
                self.A[i], Rla_F_la_N
            )
            J[idxi:idxi1, self.split_y[4] : self.split_y[5]] = kron(
                self.A[i], Rla_F_la_F
            )

        #################
        # impact equation
        #################
        J[self.split_y[5] : self.split_y[6], : self.split_y[0]] = kron(
            self.b,
            self.system.Mu_q(tn1, qn1, Delta_U)
            - self.system.Wla_g_q(tn1, qn1, Delta_P_g)
            - self.system.Wla_gamma_q(tn1, qn1, Delta_P_gamma)
            - self.system.Wla_N_q(tn1, qn1, Delta_P_N)
            - self.system.Wla_F_q(tn1, qn1, Delta_P_F),
        )

        J[
            self.split_y[5] : self.split_y[6], self.split_y[5] : self.split_y[6]
        ] = self.system.M(tn1, qn1)

        J[
            self.split_y[5] : self.split_y[6], self.split_y[6] : self.split_y[7]
        ] = -self.system.W_g(tn1, qn1)
        J[
            self.split_y[5] : self.split_y[6], self.split_y[7] : self.split_y[8]
        ] = -self.system.W_gamma(tn1, qn1)
        J[
            self.split_y[5] : self.split_y[6], self.split_y[8] : self.split_y[9]
        ] = -self.system.W_N(tn1, qn1)
        J[self.split_y[5] : self.split_y[6], self.split_y[9] :] = -self.system.W_F(
            tn1, qn1
        )

        #################################
        # impulsive bilateral constraints
        #################################
        J[self.split_y[6] : self.split_y[7], : self.split_y[0]] = kron(
            self.b, self.system.g_dot_q(tn1, qn1, un1)
        )

        # TODO: If we rearange the vector of unknowns we can build and
        # extended Butcher tableau for the velocities
        # u = (U0, ..., Us, Delta U) and use a single kron call.
        g_dot_u = self.system.W_g(tn1, qn1).T
        J[
            self.split_y[6] : self.split_y[7], self.split_y[5] : self.split_y[6]
        ] = g_dot_u
        J[self.split_y[6] : self.split_y[7], self.split_y[0] : self.split_y[1]] = kron(
            self.b, g_dot_u
        )

        J[self.split_y[7] : self.split_y[8], : self.split_y[0]] = kron(
            self.b, self.system.gamma_q(tn1, qn1, un1)
        )

        # TODO: If we rearange the vector of unknowns we can build and
        # extended Butcher tableau for the velocities
        # u = (U0, ..., Us, Delta U) and use a single kron call.
        gamma_u = self.system.W_gamma(tn1, qn1).T
        J[
            self.split_y[7] : self.split_y[8], self.split_y[5] : self.split_y[6]
        ] = gamma_u
        J[self.split_y[7] : self.split_y[8], self.split_y[0] : self.split_y[1]] = kron(
            self.b, gamma_u
        )

        ##################################################
        # mixed Singorini on velocity level and impact law
        ##################################################
        B_N = diags(np.asarray(self.B_N, dtype=float))
        B_N_bar = diags(np.asarray(~self.B_N, dtype=float))

        J[self.split_y[8] : self.split_y[9], : self.split_y[0]] = kron(
            self.b, B_N @ self.system.xi_N_q(tn1, qn1, un, un1)
        )

        W_N = self.system.W_N(tn1, qn1)
        J[self.split_y[8] : self.split_y[9], self.split_y[0] : self.split_y[1]] = kron(
            self.b, B_N @ W_N.T
        )
        J[self.split_y[8] : self.split_y[9], self.split_y[5] : self.split_y[6]] = (
            B_N @ W_N.T
        )

        J[self.split_y[8] : self.split_y[9], self.split_y[3] : self.split_y[4]] = kron(
            self.b, B_N_bar
        )

        J[
            self.split_y[8] : self.split_y[9], self.split_y[8] : self.split_y[9]
        ] = B_N_bar

        ##################################################
        # mixed Coulomb friction and tangent impact law
        ##################################################
        xi_Fn1 = self.system.xi_F(tn1, qn1, un, un1)
        xi_Fn1_q = self.system.xi_F_q(tn1, qn1, un, un1, scipy_matrix=csr_matrix)
        xi_Fn1_u = self.system.W_F(tn1, qn1, scipy_matrix=csc_matrix).T

        # local derivatives that have to be distributed with kron
        Rla_F_q = lil_matrix((self.nla_F, self.nq))
        Rla_F_u = lil_matrix((self.nla_F, self.nu))
        Rla_F_la_N = lil_matrix((self.nla_F, self.nla_N))
        Rla_F_la_F = lil_matrix((self.nla_F, self.nla_F))

        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)
            n_F = len(i_F)
            if n_F > 0:
                arg_F = prox_r_F[i_F] * xi_Fn1[i_F] - P_Fn1[i_F]
                norm_arg_F = np.linalg.norm(arg_F)
                radius = mu[i_N] * P_Nn1[i_N]

                # TODO: Investigate why strict smaller is important!
                if norm_arg_F < radius:
                    # print(f"stick")
                    Rla_F_q[i_F] = xi_Fn1_q[i_F]
                    Rla_F_u[i_F] = xi_Fn1_u[i_F]
                else:
                    if norm_arg_F > 0:
                        # print(f"slip norm_arg_F > 0")
                        slip_dir = arg_F / norm_arg_F
                        factor = (
                            np.eye(n_F) - np.outer(slip_dir, slip_dir)
                        ) / norm_arg_F
                        Rla_F_q[i_F] = (
                            radius * factor @ diags(prox_r_F[i_F]) @ xi_Fn1_q[i_F]
                        )
                        Rla_F_u[i_F] = (
                            radius * factor @ diags(prox_r_F[i_F]) @ xi_Fn1_u[i_F]
                        )
                        Rla_F_la_N[i_F[:, None], i_N] = mu[i_N] * slip_dir
                        Rla_F_la_F[i_F[:, None], i_F] = np.eye(n_F) - radius * factor
                    else:
                        # print(f"slip norm_arg_F = 0")
                        Rla_F_q[i_F] = radius * diags(prox_r_F[i_F]) @ xi_Fn1_q[i_F]
                        Rla_F_u[i_F] = radius * diags(prox_r_F[i_F]) @ xi_Fn1_u[i_F]
                        Rla_F_la_N[i_F[:, None], i_N] = mu[i_N] * arg_F
                        Rla_F_la_F[i_F[:, None], i_F] = (1 - radius) * eye(n_F)

            J[self.split_y[9] :, : self.split_y[0]] = kron(self.b, Rla_F_q)

            J[self.split_y[9] :, self.split_y[0] : self.split_y[1]] = kron(
                self.b, Rla_F_u
            )
            J[self.split_y[9] :, self.split_y[5] : self.split_y[6]] = Rla_F_u

            J[self.split_y[9] :, self.split_y[3] : self.split_y[4]] = kron(
                self.b, Rla_F_la_N
            )
            J[self.split_y[9] :, self.split_y[8] : self.split_y[9]] = Rla_F_la_N

            J[self.split_y[9] :, self.split_y[4] : self.split_y[5]] = kron(
                self.b, Rla_F_la_F
            )
            J[self.split_y[9] :, self.split_y[9] :] = Rla_F_la_F

        return J.tocsc()

        J_num = approx_fprime(yn1, self.R)
        diff = J - J_num
        # diff = diff[: self.split_y[0]]
        # diff = diff[self.split_y[0] : self.split_y[1]]
        # diff = diff[self.split_y[1] : self.split_y[2]]
        # diff = diff[self.split_y[2] : self.split_y[3]]
        # diff = diff[self.split_y[3] : self.split_y[4]]

        # diff = diff[self.split_y[4] : self.split_y[5]]  # TODO:
        # diff = diff[self.split_y[4] : self.split_y[5], : self.split_y[0]]
        # diff = diff[self.split_y[4] : self.split_y[5], self.split_y[0] : self.split_y[1]]
        # diff = diff[self.split_y[4] : self.split_y[5], self.split_y[3] : self.split_y[4]]
        # diff = diff[self.split_y[4] : self.split_y[5], self.split_y[4] : self.split_y[5]]

        # diff = diff[self.split_y[5] : self.split_y[6]]
        # diff = diff[self.split_y[6] : self.split_y[7]]
        # diff = diff[self.split_y[7] : self.split_y[8]]
        # diff = diff[self.split_y[8] : self.split_y[9]]

        # diff = diff[self.split_y[9] :] # TODO:
        # diff = diff[self.split_y[9] :, : self.split_y[0]]
        diff = diff[self.split_y[9] :, self.split_y[0] : self.split_y[1]]
        # diff = diff[self.split_y[9] :, self.split_y[3] : self.split_y[4]]
        # # diff = diff[self.split_y[9] :, self.split_y[4] : self.split_y[5]]
        # # diff = diff[self.split_y[9] :, self.split_y[5] : self.split_y[6]]
        # # diff = diff[self.split_y[9] :, self.split_y[8] : self.split_y[9]]

        error = np.linalg.norm(diff)
        if error > 1.0e-10:
            print(f"error J: {error}")
        return csc_matrix(J_num)

    def R_fixed_point(self, yn1):
        h = self.h
        tn = self.tn
        tn1 = tn + h

        # previous time step
        qn = self.qn
        un = self.un

        # current time step and intermediate stages
        (
            dq,
            du,
            dP_g,
            dP_gamma,
            dP_N,
            dP_F,
            Delta_U,
            Delta_P_g,
            Delta_P_gamma,
            Delta_P_N,
            Delta_P_F,
        ) = self.unpack_reshape(yn1)

        # quadrature for position and velocity
        qn1 = qn + dq @ self.b
        un1 = un + du @ self.b + Delta_U

        # quadrature for percussions
        P_gn1 = dP_g @ self.b + Delta_P_g
        P_gamman1 = dP_gamma @ self.b + Delta_P_gamma
        P_Nn1 = dP_N @ self.b + Delta_P_N
        P_Fn1 = dP_F @ self.b + Delta_P_F

        # initialize residual
        R = np.zeros(self.ny, dtype=yn1.dtype)
        R = yn1.copy()  # TODO: Remove this!

        # build iteration matrix and rhs
        s = self.stages
        n = s * (self.nq + self.nu + self.nla_g + self.nla_gamma)
        # n = s * (self.nq + self.nu + self.nla_gamma)
        # n = s * (self.nq + self.nu)
        A = lil_matrix((n, n), dtype=float)
        b = np.zeros(n, dtype=float)
        nq, nu, nla_g, nla_gamma = self.nq, self.nu, self.nla_g, self.nla_gamma
        for i in range(s):
            ti = tn + self.c[i] * h
            Qi = qn + dq @ self.A[i]
            Ui = un + du @ self.A[i]

            ####################
            # kinematic equation
            ####################
            A[i * nq : (i + 1) * nq, i * nq : (i + 1) * nq] = np.eye(nq, dtype=float)
            b[i * nq : (i + 1) * nq] = h * self.system.q_dot(ti, Qi, Ui)

            ####################
            # eqations of motion
            ####################
            A[
                s * nq + i * nu : s * nq + (i + 1) * nu,
                s * nq + i * nu : s * nq + (i + 1) * nu,
            ] = self.system.M(ti, Qi)

            W_gi = self.system.W_g(ti, Qi)
            A[
                s * nq + i * nu : s * nq + (i + 1) * nu,
                s * (nq + nu) + i * nla_g : s * (nq + nu) + (i + 1) * nla_g,
            ] = -W_gi

            W_gammai = self.system.W_gamma(ti, Qi)
            A[
                s * nq + i * nu : s * nq + (i + 1) * nu,
                # s * (nq + nu) + i * nla_gamma : s * (nq + nu) + (i + 1) * nla_gamma,
                s * (nq + nu + nla_g)
                + i * nla_gamma : s * (nq + nu + nla_g)
                + (i + 1) * nla_gamma,
            ] = -W_gammai

            b[s * nq + i * nu : s * nq + (i + 1) * nu] = h * self.system.h(
                ti, Qi, Ui
            )  # + self.system.W_g(ti, Qi) @ dP_g[:, i]

            #######################
            # bilateral constraints
            #######################
            # A[
            #     s * (nq + nu) + i * nla_g : s * (nq + nu) + (i + 1) * nla_g,
            #     s * nq + i * nu : s * nq + (i + 1) * nu,
            # ] = -W_gi.T
            A[
                s * (nq + nu) + i * nla_g : s * (nq + nu) + (i + 1) * nla_g,
                i * nq : (i + 1) * nq,
            ] = (
                self.system.g_q(tn, qn) * self.A[i, i]
            )
            b[
                s * (nq + nu) + i * nla_g : s * (nq + nu) + (i + 1) * nla_g
            ] = -self.system.g(tn, qn)
            # for j in range(s):
            #     if i != j:
            #         b[s * (nq + nu) + i * nla_g : s * (nq + nu) + (i + 1) * nla_g] -= self.system.g_q(tn, qn) @ dq[:, j] * self.A[i, j]

            A[
                # s * (nq + nu) + i * nla_gamma : s * (nq + nu) + (i + 1) * nla_gamma,
                s * (nq + nu + nla_g)
                + i * nla_gamma : s * (nq + nu + nla_g)
                + (i + 1) * nla_gamma,
                s * nq + i * nu : s * nq + (i + 1) * nu,
            ] = -W_gammai.T
            b[
                # s * (nq + nu) + i * nla_gamma : s * (nq + nu) + (i + 1) * nla_gamma
                s * (nq + nu + nla_g)
                + i * nla_gamma : s * (nq + nu + nla_g)
                + (i + 1) * nla_gamma,
            ] = self.system.gamma(tn, qn, np.zeros_like(Ui))

        R[:n] = spsolve(A, b)

        # dq_du_dP_gama = spsolve(A, b)
        # R[: s * (nq + nu)] = dq_du_dP_gama[: s * (nq + nu)]
        # R[s * (nq + nu + nla_g) : s * (nq + nu + nla_g + nla_gamma)] = dq_du_dP_gama[
        #     s * (nq + nu) :
        # ]

        # for i in range(s):
        #     r = 1e0
        #     # r = self.system.prox_r_g(ti, Qi)
        #     R[s * (nq + nu) + i * nla_g : s * (nq + nu) + (i + 1) * nla_g] = dP_g[
        #         :, i
        #     ] - r * self.system.g(ti, Qi)

        # np.set_printoptions(3, suppress=True)
        # print(f"A:\n{A.toarray()}")
        # print(f"b: {b}")
        # exit()

        # #########################################
        # # bilateral constraints on velocity level
        # #########################################
        # off += self.stages * self.nla_g

        # for i in range(self.stages):
        #     ti = tn + self.c[i] * h
        #     Qi = dq[:, i]
        #     Ui = du[:, i]
        #     R[
        #         off + i * self.nla_gamma : off + (i + 1) * self.nla_gamma
        #     ] = self.system.gamma(ti, Qi, Ui)

        # ###########
        # # Signorini
        # ###########
        # off += self.stages * self.nla_gamma

        # ########################
        # # pure inelastic impacts
        # ########################
        # for i in range(self.stages):
        #     ti = tn + self.c[i] * h
        #     Qi = dq[:, i]
        #     P_Ni = dP_N @ self.A[i]
        #     # P_Ni = dP_N[:, i]

        #     prox_r_N = self.system.prox_r_N(ti, Qi)
        #     g_Ni = self.system.g_N(ti, Qi)
        #     # prox_arg = g_Ni - prox_r_N * P_Ni

        #     # self.I[i] = prox_arg <= 0.0
        #     self.I[i] = g_Ni <= 0.0

        #     # R[off + i * self.nla_N : off + (i + 1) * self.nla_N] = np.where(
        #     #     self.I[i],
        #     #     prox_R0_np(P_Ni - prox_r_N * xi_Nn1),
        #     #     P_Ni,
        #     # )
        #     R[off + i * self.nla_N : off + (i + 1) * self.nla_N] = prox_R0_np(
        #         P_Ni - prox_r_N * g_Ni
        #     )

        # ##################
        # # Coulomb friction
        # ##################
        # off += self.stages * self.nla_N

        # # for i in range(self.stages):
        # #     ti = tn + self.c[i] * h
        # #     Qi = Q[:, i]
        # #     Ui = U[:, i]
        # #     P_Ni = dP_N @ self.A[i]
        # #     P_Fi = dP_F @ self.A[i]

        # #     prox_r_F = self.system.prox_r_F(ti, Qi)
        # #     gamma_Fi = self.system.gamma_F(ti, Qi, Ui)

        # #     for i_N, i_F in enumerate(self.system.NF_connectivity):
        # #         i_F = np.array(i_F)
        # #         if len(i_F) > 0:
        # #             R[off + i * self.nla_F + i_F] = P_Fi[i_F] + prox_sphere(
        # #                 prox_r_F[i_N] * gamma_Fi[i_F] - P_Fi[i_F],
        # #                 self.system.mu[i_N] * P_Ni[i_N],
        # #             )

        # #################
        # # impact equation
        # #################
        # off += self.stages * self.nla_F

        # # # nonlinear projection with h vector
        # # # TODO: What is wrong here?
        # # R[off : off + self.nu] = (
        # #     self.system.M(tn1, qn1) @ (un1 - un)
        # #     - h * self.system.h(tn1, qn1, un1)
        # #     - self.system.W_g(tn1, qn1) @ P_gn1
        # #     - self.system.W_gamma(tn1, qn1) @ P_gamman1
        # #     - self.system.W_N(tn1, qn1) @ P_Nn1
        # #     - self.system.W_F(tn1, qn1) @ P_Fn1
        # # )

        # # simple linear projection
        # R[off : off + self.nu] = spsolve(
        #     self.system.M(tn1, qn1, scipy_matrix=csc_matrix),
        #     self.system.W_g(tn1, qn1) @ Delta_P_g
        #     + self.system.W_gamma(tn1, qn1) @ Delta_P_gamma
        #     + self.system.W_N(tn1, qn1) @ Delta_P_N
        #     + self.system.W_F(tn1, qn1) @ Delta_P_F,
        # )

        # #################################
        # # impulsive bilateral constraints
        # #################################
        # off += self.nu
        # R[off : off + self.nla_g] = self.system.g_dot(tn1, qn1, un1)

        # off += self.nla_g
        # R[off : off + self.nla_gamma] = self.system.gamma(tn1, qn1, un1)

        # ##################################################
        # # mixed Singorini on velocity level and impact law
        # ##################################################
        # off += self.nla_gamma

        # xi_Nn1 = self.system.xi_N(tn1, qn1, un, un1)
        # # R[off : off + self.nla_N] = np.where(
        # #     np.any(self.I, axis=0),
        # #     P_Nn1 - prox_r_N * xi_Nn1,
        # #     P_Nn1,
        # #     # Delta_P_N - prox_r_N * xi_Nn1,
        # #     # Delta_P_N,
        # # )
        # for k in range(self.nla_N):
        #     if np.any(self.I, axis=0)[k]:
        #         R[off + k] = P_Nn1[k] - prox_r_N[k] * xi_Nn1[k]
        #         # R[off + k] = Delta_P_N[k] - prox_r_N[k] * xi_Nn1[k]
        #     else:
        #         Delta_P_N[k] = 0
        #         R[off + k] = 0

        # ##################################################
        # # mixed Coulomb friction and tangent impact law
        # ##################################################
        # off += self.nla_N

        # # prox_r_F = self.system.prox_r_F(tn1, qn1)
        # # xi_Fn1 = self.system.xi_F(tn1, qn1, un, un1)

        # # for i_N, i_F in enumerate(self.system.NF_connectivity):
        # #     i_F = np.array(i_F)
        # #     if len(i_F) > 0:
        # #         R[off + i_F] = P_Fn1[i_F] + prox_sphere(
        # #             prox_r_F[i_N] * xi_Fn1[i_F] - P_Fn1[i_F],
        # #             self.system.mu[i_N] * P_Nn1[i_N],
        # #         )

        # save integrated solutions
        # TODO: This is inefficient; unpack yn1 if it is converged.
        self.tn1 = tn1
        self.qn1 = qn1.copy()
        self.un1 = un1.copy()
        self.P_gn1 = P_gn1.copy()
        self.P_gamman1 = P_gamman1.copy()
        self.P_Nn1 = P_Nn1.copy()
        self.P_Fn1 = P_Fn1.copy()

        return R

    def solve(self):
        # lists storing output variables
        sol_q = [self.qn]
        sol_u = [self.un]
        sol_P_g = [self.h * self.la_gn]
        sol_P_gamma = [self.h * self.la_gamman]
        sol_P_N = [self.h * self.la_Nn]
        sol_P_F = [self.h * self.la_Fn]
        iterations = []

        pbar = tqdm(self.t[:-1])
        for _ in pbar:
            # only compute optimized proxparameters once per time step
            # self.prox_r_N = self.system.prox_r_N(self.tn, self.qn)
            # self.prox_r_F = self.system.prox_r_F(self.tn, self.qn)
            # print(f"self.prox_r_N: {self.prox_r_N}")
            # print(f"self.prox_r_F: {self.prox_r_F}")
            self.prox_r_N = np.ones(self.nla_N) * 1.0
            self.prox_r_F = np.ones(self.nla_F) * 0.285
            # self.prox_r_N = np.ones(self.nla_N) * 0.1
            # self.prox_r_F = np.ones(self.nla_F) * 0.1
            # self.prox_r_N = np.ones(self.nla_N) * 0.01
            # self.prox_r_F = np.ones(self.nla_F) * 0.01

            # # no percussions as initial guess
            # self.yn[self.split_y[1] : self.split_y[2]] = 0
            # self.yn[self.split_y[2] : self.split_y[3]] = 0
            # self.yn[self.split_y[3] : self.split_y[4]] = 0
            # self.yn[self.split_y[4] : self.split_y[5]] = 0

            # self.yn[self.split_y[6] : self.split_y[7]] = 0
            # self.yn[self.split_y[7] : self.split_y[8]] = 0
            # self.yn[self.split_y[8] : ] = 0

            yn1, converged, error, i, _ = fsolve(
                self.R,
                self.yn,
                jac=self.J,
                # jac="2-point",
                # jac="3-point",  # TODO: keep this, otherwise sinuglairites arise if g_N0=0!
                eps=1.0e-6,
                atol=self.atol,
                fun_args=(True,),
                jac_args=(False,),
            )

            tn1 = self.tn1

            pbar.set_description(f"t: {tn1:0.2e}; step: {i}; error: {error:.3e}")
            if not converged:
                # raise RuntimeError(
                print(
                    f"step is not converged after {i} iterations with error: {error:.5e}"
                )

            self.qn1, self.un1 = self.system.step_callback(tn1, self.qn1, self.un1)

            sol_q.append(self.qn1)
            sol_u.append(self.un1)
            sol_P_g.append(self.P_gn1)
            sol_P_gamma.append(self.P_gamman1)
            sol_P_N.append(self.P_Nn1)
            sol_P_F.append(self.P_Fn1)
            iterations.append(i)

            # update local variables for accepted time step
            self.qn = self.qn1.copy()
            self.un = self.un1.copy()

            # warmstart for next iteration
            self.tn = tn1
            self.yn = yn1.copy()

        print("-----------------")
        print(
            f"Iterations per time step: max = {max(iterations)}, avg={sum(iterations) / float(len(iterations))}"
        )
        print("-----------------")

        return Solution(
            t=np.array(self.t),
            q=np.array(sol_q),
            u=np.array(sol_u),
            P_g=np.array(sol_P_g),
            P_gamma=np.array(sol_P_gamma),
            P_N=np.array(sol_P_N),
            P_F=np.array(sol_P_F),
        )
