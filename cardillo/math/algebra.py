import numpy as np
from math import sqrt, sin, cos, tan, asin, acos, atan, pi, copysign

e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])


def ei(i):
    return np.roll(e1, i)


def sign(x):
    return copysign(1.0, x)


def norm(a: np.ndarray) -> float:
    """Euclidean norm of an array of arbitrary length."""
    return sqrt(a @ a)


def ax2skew(a: np.ndarray) -> np.ndarray:
    """Computes the skew symmetric matrix from a 3D vector."""
    assert a.size == 3
    # fmt: off
    return np.array([[0,    -a[2], a[1] ],
                     [a[2],  0,    -a[0]],
                     [-a[1], a[0], 0    ]], dtype=a.dtype)
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


###############################################################################
# TODO: move on here!
###############################################################################
def trace(J):
    ndim = len(J)
    if ndim == 1:
        return J
    elif ndim == 2:
        return J[0, 0] + J[1, 1]
    elif ndim == 3:
        return J[0, 0] + J[1, 1] + J[2, 2]
    else:
        return np.trace(J)


def det(J):
    ndim = len(J)
    if ndim == 1:
        return J
    elif ndim == 2:
        return det2D(J)
    elif ndim == 3:
        return det3D(J)
    else:
        return np.linalg.det(J)


def det2D(J):
    return J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]


def det3D(J):
    a, b, c = J[0]
    d, e, f = J[1]
    g, h, i = J[2]

    return a * (e * i - h * f) - b * (d * i - g * f) + c * (d * h - g * e)


def inv(J):
    ndim = len(J)
    if ndim == 1:
        return 1 / J
    elif ndim == 2:
        return inv2D(J)
    elif ndim == 3:
        return inv3D(J)
    elif ndim == 4:
        return inv4D(J)
    else:
        return np.linalg.inv(J)


def inv2D(J):
    # see https://de.wikipedia.org/wiki/Inverse_Matrix
    j = det2D(J)
    # fmt: off
    Jinv = 1 / j * np.array([[ J[1, 1], -J[0, 1]], 
                             [-J[1, 0],  J[0, 0]]])
    # fmt: on
    return Jinv


def inv3D(J):
    # see https://de.wikipedia.org/wiki/Inverse_Matrix
    j = det3D(J)

    a, b, c = J[0]
    d, e, f = J[1]
    g, h, i = J[2]

    A = e * i - f * h
    B = c * h - b * i
    C = b * f - c * e
    D = f * g - d * i
    E = a * i - c * g
    F = c * d - a * f
    G = d * h - e * g
    H = b * g - a * h
    I = a * e - b * d
    # fmt: off
    Jinv = 1 / j * np.array([[A, B, C], \
                             [D, E, F], \
                             [G, H, I]])
    # fmt: on
    return Jinv


def inv4D(J):
    # see https://stackoverflow.com/a/60374938/7280763
    # and https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/LU/InverseImpl.h#L198-252

    J2323 = J[2, 2] * J[3, 3] - J[2, 3] * J[3, 2]
    J1323 = J[2, 1] * J[3, 3] - J[2, 3] * J[3, 1]
    J1223 = J[2, 1] * J[3, 2] - J[2, 2] * J[3, 1]
    J0323 = J[2, 0] * J[3, 3] - J[2, 3] * J[3, 0]
    J0223 = J[2, 0] * J[3, 2] - J[2, 2] * J[3, 0]
    J0123 = J[2, 0] * J[3, 1] - J[2, 1] * J[3, 0]
    J2313 = J[1, 2] * J[3, 3] - J[1, 3] * J[3, 2]
    J1313 = J[1, 1] * J[3, 3] - J[1, 3] * J[3, 1]
    J1213 = J[1, 1] * J[3, 2] - J[1, 2] * J[3, 1]
    J2312 = J[1, 2] * J[2, 3] - J[1, 3] * J[2, 2]
    J1312 = J[1, 1] * J[2, 3] - J[1, 3] * J[2, 1]
    J1212 = J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]
    J0313 = J[1, 0] * J[3, 3] - J[1, 3] * J[3, 0]
    J0213 = J[1, 0] * J[3, 2] - J[1, 2] * J[3, 0]
    J0312 = J[1, 0] * J[2, 3] - J[1, 3] * J[2, 0]
    J0212 = J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0]
    J0113 = J[1, 0] * J[3, 1] - J[1, 1] * J[3, 0]
    J0112 = J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]

    det = (
        J[0, 0] * (J[1, 1] * J2323 - J[1, 2] * J1323 + J[1, 3] * J1223)
        - J[0, 1] * (J[1, 0] * J2323 - J[1, 2] * J0323 + J[1, 3] * J0223)
        + J[0, 2] * (J[1, 0] * J1323 - J[1, 1] * J0323 + J[1, 3] * J0123)
        - J[0, 3] * (J[1, 0] * J1223 - J[1, 1] * J0223 + J[1, 2] * J0123)
    )
    det = 1 / det

    J_inv = np.zeros((4, 4))
    J_inv[0, 0] = det * (J[1, 1] * J2323 - J[1, 2] * J1323 + J[1, 3] * J1223)
    J_inv[0, 1] = det * -(J[0, 1] * J2323 - J[0, 2] * J1323 + J[0, 3] * J1223)
    J_inv[0, 2] = det * (J[0, 1] * J2313 - J[0, 2] * J1313 + J[0, 3] * J1213)
    J_inv[0, 3] = det * -(J[0, 1] * J2312 - J[0, 2] * J1312 + J[0, 3] * J1212)
    J_inv[1, 0] = det * -(J[1, 0] * J2323 - J[1, 2] * J0323 + J[1, 3] * J0223)
    J_inv[1, 1] = det * (J[0, 0] * J2323 - J[0, 2] * J0323 + J[0, 3] * J0223)
    J_inv[1, 2] = det * -(J[0, 0] * J2313 - J[0, 2] * J0313 + J[0, 3] * J0213)
    J_inv[1, 3] = det * (J[0, 0] * J2312 - J[0, 2] * J0312 + J[0, 3] * J0212)
    J_inv[2, 0] = det * (J[1, 0] * J1323 - J[1, 1] * J0323 + J[1, 3] * J0123)
    J_inv[2, 1] = det * -(J[0, 0] * J1323 - J[0, 1] * J0323 + J[0, 3] * J0123)
    J_inv[2, 2] = det * (J[0, 0] * J1313 - J[0, 1] * J0313 + J[0, 3] * J0113)
    J_inv[2, 3] = det * -(J[0, 0] * J1312 - J[0, 1] * J0312 + J[0, 3] * J0112)
    J_inv[3, 0] = det * -(J[1, 0] * J1223 - J[1, 1] * J0223 + J[1, 2] * J0123)
    J_inv[3, 1] = det * (J[0, 0] * J1223 - J[0, 1] * J0223 + J[0, 2] * J0123)
    J_inv[3, 2] = det * -(J[0, 0] * J1213 - J[0, 1] * J0213 + J[0, 2] * J0113)
    J_inv[3, 3] = det * (J[0, 0] * J1212 - J[0, 1] * J0212 + J[0, 2] * J0112)
    return J_inv


#############################################################
# invariants and their derivatives
# see https://en.wikipedia.org/wiki/Tensor_derivative_(continuum_mechanics)#Derivatives_of_the_invariants_of_a_second-order_tensor
#############################################################
def I1(A):
    """First matrix invariant (trace)."""
    return trace(A)


def dI1(A):
    """Gradient of first matrix invariant (trace)."""
    a, _ = A.shape
    return np.identity(a)


def I2(A):
    """Second matrix invariant."""
    I1_ = I1(A)
    I21 = I1(A.T @ A)
    return 0.5 * (I1_ * I1_ + I21)


def dI2(A):
    """Gradient of second matrix invariant."""
    a, _ = A.shape
    return I1(A) * np.identity(a) - A.T


def I3(A):
    """Third matrix invariant (determinant)."""
    return det(A)


def dI3(A):
    """Gradient of first matrix invariant (determinant)."""
    return I3(A) * inv(A).T


# def eig(C):
#     """Eigenvalues of 3x3 matrix C, see Basar2001"""
#     I1_ = I1(C)
#     I2_ = I2(C)
#     I3_ = I3(C)

#     tmp1 = I1_**2 - 3 * I2_
#     Theta = acos((2 * I1_**3 - 9 * I1_ * I2_ + 27 * I3_) / (2 * tmp1**(3/2)))

#     la1 = (I1_ + 2 * sqrt(tmp1) * cos((Theta + 2 * pi) / 3)) / 3
#     la2 = (I1_ + 2 * sqrt(tmp1) * cos((Theta + 4 * pi) / 3)) / 3
#     la3 = (I1_ + 2 * sqrt(tmp1) * cos((Theta + 6 * pi) / 3)) / 3

#     return la1, la2, la3


def eig(C):
    """eigenvalues of 3x3 matrix, see https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices"""
    p1 = C[0, 1] ** 2 + C[0, 2] ** 2 + C[1, 2] ** 2
    if p1 == 0:
        # C is diagonal
        eig1 = C[0, 0]
        eig2 = C[1, 1]
        eig3 = C[2, 2]
    else:
        q = (C[0, 0] + C[1, 1] + C[2, 2]) / 3  # trace(C)
        p2 = (C[0, 0] - q) ** 2 + (C[1, 1] - q) ** 2 + (C[2, 2] - q) ** 2 + 2 * p1
        p = sqrt(p2 / 6)
        B = (1 / p) * (C - q * np.eye(3))
        r = det3D(B) / 2

        # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        # but computation error can leave it slightly outside this range.
        if r <= -1:
            phi = pi / 3
        elif r >= 1:
            phi = 0
        else:
            phi = acos(r) / 3

        # the eigenvalues satisfy eig3 <= eig2 <= eig1
        eig1 = q + 2 * p * cos(phi)
        eig3 = q + 2 * p * cos(phi + (2 * pi / 3))
        eig2 = 3 * q - eig1 - eig3  # since trace(A) = eig1 + eig2 + eig3

    return eig3, eig2, eig1


def eigh_Basar(C):
    la1, la2, la3 = eig(C)

    I1_ = I1(C)
    I3_ = I3(C)

    Cinv = inv3D(C)

    C1 = la1 * (
        (C - (I1_ - la1) * np.eye(3) + I3_ * Cinv / la1)
        / (2 * la1**2 - I1_ * la1 + I3_ / la1)
    )
    C2 = la2 * (
        (C - (I1_ - la2) * np.eye(3) + I3_ * Cinv / la2)
        / (2 * la2**2 - I1_ * la2 + I3_ / la2)
    )
    C3 = la3 * (
        (C - (I1_ - la3) * np.eye(3) + I3_ * Cinv / la3)
        / (2 * la3**2 - I1_ * la3 + I3_ / la3)
    )

    return la1, la2, la3, C1, C2, C3


def eigh_numpy(C):
    la, u = np.linalg.eigh(C)
    C1 = np.outer(u[:, 0], u[:, 0])
    C2 = np.outer(u[:, 1], u[:, 1])
    C3 = np.outer(u[:, 2], u[:, 2])
    return la[0], la[1], la[2], C1, C2, C3
