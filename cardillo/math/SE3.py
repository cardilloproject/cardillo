from __future__ import annotations
import numpy as np
from cardillo.math import ax2skew


class se3:
    @staticmethod
    def fromso3R3(h_U: np.ndarray, h_Om: np.ndarray) -> se3:
        assert h_U.shape == (3,), "h_U has to be an array of length 3"
        assert h_Om.shape == (3,), "h_Om has to be an array of length 3"
        return se3(np.concatenate((h_U, h_Om)))

    @staticmethod
    def fromR6(h: np.ndarray) -> se3:
        assert h.shape == (6,), "h has to be an array of length 6"
        return se3(h)

    @staticmethod
    def fromse3(h: se3) -> se3:
        return se3.fromso3R3(h.h_U, h.h_Om)

    def __init__(self, h=None) -> None:
        if h is None:
            self.__h = np.zeros(6, dtype=float)
        else:
            # ensure float ndarray data
            assert h.shape == (6,), "h has to be an array of length 6"
            self.__h = np.asarray(h, dtype=float)

    @property
    def h_U(self) -> np.ndarray:
        return self.__h[:3]

    @h_U.setter
    def h_U(self, value: np.ndarray):
        assert value.shape == (3,), "value has to be an array of length 3"
        self.__h[:3] = value

    @property
    def h_Om(self) -> np.ndarray:
        return self.__h[3:]

    @h_Om.setter
    def h_Om(self, value: np.ndarray):
        assert value.shape == (3,), "value has to be an array of length 3"
        self.__h[3:] = value

    def __call__(self) -> np.ndarray:
        return self.__h

    def __str__(self) -> str:
        return f"h_U: {self.h_U}, h_OM: {self.h_Om}"

    def __repr__(self) -> str:
        return f'se3("{self.__str__}")'

    def tilde(self) -> np.ndarray:
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


if __name__ == "__main__":
    h_U = np.array([-1, 0, 1])
    h_Om = np.array([1, 2, 3])

    # h = se3(h_U, h_Om)
    h = se3.fromso3R3(h_U, h_Om)
    h = se3.fromR6(np.concatenate((h_U, h_Om)))
    h = se3.fromse3(h)

    h_tilde = h.tilde()
    h_wedge = h.wedge()
    h_check = h.check()
    print(f"h: {h}")
    print(f"h_tilde:\n{h_tilde}")
    print(f"h_wedge:\n{h_wedge}")
    print(f"h_check:\n{h_check}")
