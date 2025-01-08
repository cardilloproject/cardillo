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
    assert a.size == 3
    # fmt: off
    return np.array([[0,    -a[2], a[1] ],
                     [a[2],  0,    -a[0]],
                     [-a[1], a[0], 0    ]], dtype=a.dtype)
    # fmt: on


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


def ax2skew_a():
    """
    Partial derivative of the `ax2skew` function with respect to its argument.

    Note:
    -----
    This is a constant 3x3x3 ndarray."""
    A = np.zeros((3, 3, 3), dtype=float)
    A[1, 2, 0] = -1
    A[2, 1, 0] = 1
    A[0, 2, 1] = 1
    A[2, 0, 1] = -1
    A[0, 1, 2] = -1
    A[1, 0, 2] = 1
    return A


def skew2ax_A() -> np.ndarray:
    """
    Partial derivative of the `skew2ax` function with respect to its argument.

    Note:
    -----
    This is a constant 3x3x3 ndarray."""
    A = np.zeros((3, 3, 3), dtype=float)
    A[0, 2, 1] = 0.5
    A[0, 1, 2] = -0.5

    A[1, 0, 2] = 0.5
    A[1, 2, 0] = -0.5

    A[2, 1, 0] = 0.5
    A[2, 0, 1] = -0.5
    return A


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector product of two 3D vectors."""
    assert a.size == 3
    assert b.size == 3
    # fmt: off
    return np.array([a[1] * b[2] - a[2] * b[1], \
                     a[2] * b[0] - a[0] * b[2], \
                     a[0] * b[1] - a[1] * b[0] ])
    # fmt: on


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
