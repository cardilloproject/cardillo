import numpy as np
from math import copysign

e1 = np.array([1, 0, 0], dtype=float)
e2 = np.array([0, 1, 0], dtype=float)
e3 = np.array([0, 0, 1], dtype=float)


def atan2(y, x):
    """Atan2 implementation that can handle complex numbers,
    see https://en.wikipedia.org/wiki/Atan2#Definition. It returns
    atan(y / x).
    """
    if x > 0:
        return np.arctan(y / x)
    elif x < 0:
        if y >= 0:
            return np.arctan(y / x) + np.pi
        elif y < 0:
            return np.arctan(y / x) - np.pi
    else:
        # x == 0
        if y > 0:
            return 0.5 * np.pi
        elif y < 0:
            return -0.5 * np.pi
        else:
            # x == 0 and y == 0
            # this is undefined but we set it to 0
            return 0


def ei(i: int) -> np.ndarray:
    """Retuns the i-th Cartesian basis vector.
    With i=0: e1, i=1: e2, i=2: e3, i=3: e1, etc."""
    return np.roll(e1, i)


def sign(x: float) -> float:
    """Sign of x."""
    return copysign(1.0, x)


def norm(a: np.ndarray) -> float:
    """Euclidean norm of an array of arbitrary length."""
    return np.sqrt(a @ a)


def LeviCivita3(i: int, j: int, k: int) -> int:
    """Levi-Civita symbol, see https://en.wikipedia.org/wiki/Levi-Civita_symbol"""
    return (i - j) * (j - k) * (k - i) // 2


def ax2skew(a: np.ndarray) -> np.ndarray:
    """Computes the skew symmetric matrix from a 3D vector."""
    # unpacking python floats is much faster than indexing numpy scalars
    a0, a1, a2 = a.tolist()
    A = np.zeros((3, 3), dtype=a.dtype)
    A[0, 1] = -a2
    A[0, 2] = a1
    A[1, 0] = a2
    A[1, 2] = -a0
    A[2, 0] = -a1
    A[2, 1] = a0
    return A


def ax2skew_squared(a: np.ndarray) -> np.ndarray:
    """Computes the product of a skew-symmetric matrix with itself from a given axial vector."""
    assert a.size == 3
    a1, a2, a3 = a
    # fmt: off
    return np.array([
        [-a2**2 - a3**2,              a1 * a2,              a1 * a3],
        [             a2 * a1, -a1**2 - a3**2,              a2 * a3],
        [             a3 * a1,              a3 * a2, -a1**2 - a2**2],
    ], dtype=a.dtype)
    # fmt: on


def skew2ax(A: np.ndarray) -> np.ndarray:
    """Computes the axial vector from a skew symmetric 3x3 matrix."""
    assert A.shape == (3, 3)
    # fmt: off
    return 0.5 * np.array([A[2, 1] - A[1, 2], 
                           A[0, 2] - A[2, 0], 
                           A[1, 0] - A[0, 1]], dtype=A.dtype)
    # fmt: on


_ax2skew_a = np.zeros((3, 3, 3), dtype=float)
_ax2skew_a[1, 2, 0] = -1
_ax2skew_a[2, 1, 0] = 1
_ax2skew_a[0, 2, 1] = 1
_ax2skew_a[2, 0, 1] = -1
_ax2skew_a[0, 1, 2] = -1
_ax2skew_a[1, 0, 2] = 1

_skew2ax_A = np.zeros((3, 3, 3), dtype=float)
_skew2ax_A[0, 2, 1] = 0.5
_skew2ax_A[0, 1, 2] = -0.5
_skew2ax_A[1, 0, 2] = 0.5
_skew2ax_A[1, 2, 0] = -0.5
_skew2ax_A[2, 1, 0] = 0.5
_skew2ax_A[2, 0, 1] = -0.5


def ax2skew_a():
    """
    Partial derivative of the `ax2skew` function with respect to its argument.

    Note:
    -----
    This is a constant 3x3x3 ndarray."""
    return _ax2skew_a.copy()


def skew2ax_A() -> np.ndarray:
    """
    Partial derivative of the `skew2ax` function with respect to its argument.

    Note:
    -----
    This is a constant 3x3x3 ndarray."""
    return _skew2ax_A.copy()


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector product of two 3D vectors."""
    a0, a1, a2 = a.tolist()
    b0, b1, b2 = b.tolist()
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def is_positive_definite(A) -> bool:
    A = np.asarray(A)
    rows, cols = A.shape
    assert rows == cols
    for i in range(rows):
        det = np.linalg.det(A[:i, :i])
        if det > 0:
            continue
        else:
            return False
    return True
