from __future__ import annotations
import numpy as np
from math import sqrt
from cardillo.math import (
    ax2skew,
    rodriguez,
    rodriguez_inv,
    tangent_map,
    inverse_tangent_map,
)
from cardillo.math.algebra import skew2ax

#TODO: are these classes used? how do they relate to rotations.py?

class so3:
    def __init__(self, omega=None) -> None:
        if omega is None:
            self.__omega = np.zeros(3, dtype=float)
        else:
            assert omega.shape == (3,), "omega has to be an array of shape (3,)"
            self.__omega = np.asarray(omega, dtype=float)

    def __call__(self) -> np.ndarray:
        return self.__omega

    def __str__(self) -> str:
        return f"{self()}"

    def __invert__(self) -> np.ndarray:
        """Tilde operation"""
        return ax2skew(self())

    def exp(self) -> np.ndarray:
        """Exponential function. This is computed using Rodriguez formular."""
        return rodriguez(self())

    def T(self) -> np.ndarray:
        return tangent_map(self())

    def T_inv(self) -> np.ndarray:
        return inverse_tangent_map(self())


class SO3:
    def __init__(self, R=None) -> None:
        if R is None:
            self.__R = np.eye(3, dtype=float)
        else:
            assert R.shape == (3, 3), "R has to be an array of shape (3, 3)"
            self.__R = np.asarray(R, dtype=float)

    def __call__(self) -> np.ndarray:
        return self.__R

    def __str__(self) -> str:
        return f"{self()}"

    def __invert__(self) -> SO3:
        return SO3(self().T)

    @property
    def T(self) -> np.ndarray:
        return self().T

    def log(self):
        return rodriguez_inv(self())


class se3:
    @staticmethod
    def fromso3R3(h_U: np.ndarray, h_Om: np.ndarray) -> se3:
        assert h_U.shape == (3,), "h_U has to be an array of shape (3,)"
        assert h_Om.shape == (3,), "h_Om has to be an array of shape (3,)"
        return se3(h_U, h_Om)

    @staticmethod
    def fromR6(h: np.ndarray) -> se3:
        assert h.shape == (6,), "h has to be an array of shape (6,)"
        return se3(h[:3], h[3:])

    @staticmethod
    def fromse3(other: se3) -> se3:
        return se3(other.h_U, other.h_Om)

    def __init__(self, h_U=None, h_Om=None) -> None:
        if h_U is None:
            self.__h_U = np.zeros(3, dtype=float)
        else:
            assert h_U.shape == (3,), "h_U has to be an array of shape (3,)"
            self.__h_U = np.asarray(h_U, dtype=float)

        if h_Om is None:
            self.__h_Om = np.zeros(3, dtype=float)
        else:
            assert h_Om.shape == (3,), "h_Om has to be an array of shape (3,)"
            self.__h_Om = np.asarray(h_Om, dtype=float)

    @property
    def h_U(self) -> np.ndarray:
        return self.__h_U

    @h_U.setter
    def h_U(self, value: np.ndarray):
        assert value.shape == (3,), "value has to be an array of shape (3,)"
        self.__h_U = value

    @property
    def h_Om(self) -> np.ndarray:
        return self.__h_Om

    @h_Om.setter
    def h_Om(self, value: np.ndarray):
        assert value.shape == (3,), "value has to be an array of shape (3,)"
        self.__h_Om = value

    def __call__(self) -> np.ndarray:
        return np.concatenate((self.h_U, self.h_Om))

    def __str__(self) -> str:
        return f"h_U: {self.h_U}; h_OM: {self.h_Om}"

    def __invert__(self) -> np.ndarray:
        tilde = np.zeros((4, 4), dtype=float)
        tilde[:3, :3] = ax2skew(self.h_Om)
        tilde[:3, 3] = self.h_U
        return tilde

    def wedge(self) -> np.ndarray:
        wedge = np.zeros((6, 6), dtype=float)
        wedge[:3, :3] = wedge[3:, 3:] = ax2skew(self.h_Om)
        wedge[:3, 3:] = ax2skew(self.h_U)
        return wedge

    def check(self) -> np.ndarray:
        check = np.zeros((6, 6), dtype=float)
        check[3:, :3] = check[:3, 3:] = ax2skew(self.h_Om)
        check[3:, 3:] = ax2skew(self.h_U)
        return check

    def exp(self):
        out = np.zeros((4, 4), dtype=float)
        omega = so3(self.h_Om)
        out[:3, :3] = omega.exp()
        out[:3, 3] = omega.T() @ self.h_U
        out[3, 3] = 1.0
        return out

    @staticmethod
    def T_UOm(a, b):
        a_tilde = ax2skew(a)

        b2 = b @ b
        if b2 > 0:
            abs_b = sqrt(b2)
            alpha = np.sin(abs_b) / abs_b
            beta = 2.0 * (1.0 - np.cos(abs_b)) / b2

            b_tilde = ax2skew(b)
            ab = a_tilde @ b_tilde + b_tilde @ a_tilde

            return (
                -0.5 * beta * a_tilde
                + (1.0 - alpha) * ab / b2
                + (b @ a)
                * (
                    (beta - alpha) * b_tilde
                    + (0.5 * beta - 3.0 * ((1.0 - alpha) / b2) * b_tilde @ b_tilde)
                )
                / b2
            )
        else:
            return -0.5 * a_tilde

    def T(self):
        out = np.zeros((6, 6), dtype=float)
        out[:3, :3] = out[3:, 3:] = so3(self.h_Om).T()
        out[:3, 3:] = self.T_UOm(self.h_U, self.h_Om)
        return out


class SE3:
    @staticmethod
    def fromSE3(other: SE3) -> SE3:
        return SE3(other.R, other.r)

    @staticmethod
    def fromH(H: np.ndarray) -> SE3:
        assert H.shape == (4, 4), "H has to be an array of shape (3, 3)"
        return SE3(H[:3, :3], H[:3, 3])

    def __init__(self, R=None, r=None) -> None:
        if R is None:
            self.__R = np.eye(3, dtype=float)
        else:
            assert R.shape == (3, 3), "R has to be an array of shape (3, 3)"
            self.__R = np.asarray(R, dtype=float)

        if r is None:
            self.__r = np.zeros(3, dtype=float)
        else:
            assert r.shape == (3,), "r has to be an array of shape (3,)"
            self.__r = np.asarray(r, dtype=float)

    @property
    def R(self) -> np.ndarray:
        return self.__R

    @R.setter
    def R(self, value: np.ndarray):
        assert value.shape == (3, 3), "value has to be an array of shape (3, 3)"
        self.__R = value

    @property
    def r(self) -> np.ndarray:
        return self.__r

    @r.setter
    def r(self, value: np.ndarray):
        assert value.shape == (3,), "value has to be an array of shape (3,)"
        self.__r = value

    def __call__(self):
        H = np.zeros((4, 4), dtype=float)
        H[:3, :3] = self.R
        H[:3, 3] = self.r
        H[3, 3] = 1.0
        return H

    def __str__(self) -> str:
        return f"{self()}"

    def __invert__(self) -> SE3:
        return SE3(self.R.T, -self.R.T @ self.r)

    def __matmul__(self, other) -> SE3:
        """Inner product."""
        return SE3.fromH(self() @ other())

    def log(self) -> np.ndarray:
        h_Om = SO3(self.R).log()
        h_U = so3(h_Om).T_inv().T @ self.r
        return np.concatenate((h_U, h_Om))


if __name__ == "__main__":
    # ###########
    # # se3 tests
    # ###########
    # h_U = np.array([-1, 0, 1])
    # h_Om = np.array([1, 2, 3])

    # # h = se3(h_U, h_Om)
    # h = se3.fromso3R3(h_U, h_Om)
    # h = se3.fromR6(np.concatenate((h_U, h_Om)))
    # h = se3.fromse3(h)

    # h_tilde = h.tilde()
    # h_wedge = h.wedge()
    # h_check = h.check()
    # print(f"h: {h}")
    # print(f"h_tilde:\n{h_tilde}")
    # print(f"h_wedge:\n{h_wedge}")
    # print(f"h_check:\n{h_check}")

    ###########
    # SO3 tests
    ###########
    from cardillo.math import A_IB_basic

    R = A_IB_basic(np.pi / 4).x()
    r = np.array([1, 2, 3])
    H = SE3(R, r)
    # H = SE3()
    H_inv = ~H

    print(f"H:\n{H}")
    print(f"H_inv:\n{H_inv}")
