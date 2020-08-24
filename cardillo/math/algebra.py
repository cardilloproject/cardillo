import numpy as np
from math import sqrt, sin, cos, acos, pi
from cardillo.math.numerical_derivative import Numerical_derivative

e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

def norm2(a):
    return sqrt(a[0] * a[0] + a[1] * a[1])

def norm3(a):
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

def norm4(a):
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3])

def skew2ax(A):
    # build skew symmetric matrix first and then compute axial vector
    B = 0.5 * (A - A.T)
    return np.array([B[2, 1], B[0, 2], B[1, 0]])

# see https://pythonpath.wordpress.com/2012/09/04/skew-with-numpy-operations/
def ax2skew(a):
    return np.array([[0,    -a[2], a[1] ],
                     [a[2],  0,    -a[0]],
                     [-a[1], a[0], 0    ]])

def ax2skew_a():
    A = np.zeros((3, 3, 3))
    A[1, 2, 0] = -1
    A[2, 1, 0] =  1
    A[0, 2, 1] =  1
    A[2, 0, 1] = -1
    A[0, 1, 2] = -1
    A[1, 0, 2] =  1
    return A

def ax2skew2(a):
    skv = np.roll(np.roll(np.diag(a.flatten()), 1, 1), -1, 0)
    return skv - skv.T

def cross3(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], \
                     a[2] * b[0] - a[0] * b[2], \
                     a[0] * b[1] - a[1] * b[0] ])

# TODO: think of a test for this function
def cross_nm(a, b):
    na0 = a.shape[0]
    na1 = a.shape[1]
    nb0 = b.shape[0]
    nb1 = b.shape[1]
    A = np.zeros((na0, na1, nb1))
    for i in range(na0):
        for n in range(na1):
            for m in range(nb1):
                for j in range(na0):
                    for k in range(nb0):
                        A[i, n, m] += LeviCivita3(i, j, k) * a[k, n] * b[j, m]
    return A

def LeviCivita3(i, j, k):
    return (i - j) * (j - k) * (k - i) / 2

def cross3LeviCivita(a, b):
    c = np.zeros(3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c[i] += LeviCivita3(i, j, k) * a[j] * b[k]
    return c

def skew2axLeviCevita(A):
    # build skew symmetric matrix first and then compute axial vector
    B = 0.5 * (A - A.T)
    a = np.zeros(3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a[i] += LeviCivita3(i, j, k) * B[k, j]
    return 0.5 * a

def ax2skewLeviCevita(a):
    A = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                A[k, j] += LeviCivita3(i, j, k) * a[i]
    return A

def quat2mat(p):
    p0, p1, p2, p3 = p
    return np.array([[p0, -p1, -p2, -p3], \
                     [p1,  p0, -p3,  p2], \
                     [p2,  p3,  p0, -p1], \
                     [p3, -p2,  p1,  p0]])

def quat2mat_p(p):
    A_p = np.zeros((4, 4, 4))
    A_p[:, :, 0] = np.eye(4)
    A_p[0, 1:, 1:] = -np.eye(3)
    A_p[1:, 0, 1:] =  np.eye(3)
    A_p[1:, 1:, 1:] = ax2skew_a()

    return A_p

def quat2rot(p):
    p_ = p / norm4(p)
    v_p_tilde = ax2skew(p_[1:])
    return np.eye(3) + 2 * (v_p_tilde @ v_p_tilde + p_[0] * v_p_tilde)

def quat2rot_p(p):
    norm_p = norm4(p)
    q = p / norm_p
    v_q_tilde = ax2skew(q[1:])
    v_q_tilde_v_q = ax2skew_a()
    q_p = np.eye(4) / norm_p - np.outer(p, p) / (norm_p**3)
    
    A_q = np.zeros((3, 3, 4))
    A_q[:, :, 0] = 2 * v_q_tilde
    A_q[:, :, 1:] += np.einsum('ijk,jl->ilk', v_q_tilde_v_q, 2 * v_q_tilde)
    A_q[:, :, 1:] += np.einsum('ij,jkl->ikl', 2 * v_q_tilde, v_q_tilde_v_q)
    A_q[:, :, 1:] += 2 * (q[0] * v_q_tilde_v_q)
    
    return np.einsum('ijk,kl->ijl', A_q, q_p)

def axis_angle2quat(axis, angle):
    n = axis / norm3(axis)
    return np.concatenate([ [np.cos(angle/2)], np.sin(angle/2)*n])

def A_IK_basic_x(phi):
    sp = sin(phi)
    cp = cos(phi)
    return np.array([[1,  0,   0],\
                     [0, cp, -sp],\
                     [0, sp,  cp]])

def dA_IK_basic_x(phi):
    sp = sin(phi)
    cp = cos(phi)
    return np.array([[0,  0,   0],\
                     [0, -sp, -cp],\
                     [0, cp,  -sp]])

def A_IK_basic_y(phi):
    sp = sin(phi)
    cp = cos(phi)
    return np.array([[ cp,  0,  sp],\
                     [  0,  1,   0],\
                     [-sp,  0,  cp]])

def dA_IK_basic_y(phi):
    sp = sin(phi)
    cp = cos(phi)
    return np.array([[ -sp,  0,  cp],\
                     [  0,  0,   0],\
                     [-cp,  0,  -sp]])

def A_IK_basic_z(phi):
    sp = sin(phi)
    cp = cos(phi)
    return np.array([[ cp, -sp, 0],\
                     [ sp,  cp, 0],\
                     [  0,   0, 1]])

def dA_IK_basic_z(phi):
    sp = sin(phi)
    cp = cos(phi)
    return np.array([[ -sp, -cp, 0],\
                     [ cp,  -sp, 0],\
                     [  0,   0, 0]])

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

def determinant(J):
    ndim = len(J)
    if ndim == 1:
        return J
    elif ndim == 2:
        return determinant2D(J)
    elif ndim == 3:
        return determinant3D(J)
    else:
        return np.linalg.det(J)

def determinant2D(J):
    return J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

def determinant3D(J):
    a, b, c = J[0]
    d, e, f = J[1]
    g, h, i = J[2]

    return a * (e * i - h * f) - b * ( d * i - g * f) + c * (d * h - g * e)

def inverse(J):
    ndim = len(J)
    if ndim == 1:
        return 1 / J
    elif ndim == 2:
        return inverse2D(J)
    elif ndim == 3:
        return inverse3D(J)
    else:
        return np.linalg.inv(J)

def inverse2D(J):
    # see https://de.wikipedia.org/wiki/Inverse_Matrix
    j = determinant2D(J)
    Jinv = 1 / j * np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]])
    return Jinv

def inverse3D(J):
    # see https://de.wikipedia.org/wiki/Inverse_Matrix
    j = determinant3D(J)

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
    Jinv = 1 / j * np.array([[A, B, C], \
                             [D, E, F], \
                             [G, H, I]])
    return Jinv

def test_determinant():
    n = np.array([1, 2, 3, 4])
    for i in n:
        A = np.random.rand(i, i)
        detA = determinant(A)
        assert np.allclose(detA, np.linalg.det(A))

def test_inverse():
    n = np.array([1, 2, 3, 4])
    for i in n:
        A = np.random.rand(i, i)
        Ainv = inverse(A)
        I = np.identity(i)
        assert np.allclose(I, A @ Ainv)

#############################################################
# invariants and their derivatives
# see https://en.wikipedia.org/wiki/Tensor_derivative_(continuum_mechanics)#Derivatives_of_the_invariants_of_a_second-order_tensor
#############################################################

def I1(A):
    r"""First matrix invariant (trace).
    """
    return trace(A)

def dI1(A):
    r"""Gradient of first matrix invariant (trace).
    """
    a, _ = A.shape
    return np.identity(a)

def I2(A):
    r"""Second matrix invariant.
    """
    I1_ = I1(A)
    I21 = I1(A.T @ A)
    return 0.5 * (I1_ * I1_ + I21)

def dI2(A):
    r"""Gradient of second matrix invariant.
    """
    a, _ = A.shape
    return I1(A) * np.identity(a) - A.T

def I3(A):
    r"""Third matrix invariant (determinant).
    """
    return determinant(A)

def dI3(A):
    r"""Gradient of first matrix invariant (determinant).
    """
    return I3(A) * inverse(A).T

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
    p1 = C[0, 1]**2 + C[0, 2]**2 + C[1, 2]**2
    if p1 == 0:
        # C is diagonal
        eig1 = C[0, 0]
        eig2 = C[1, 1]
        eig3 = C[2, 2]
    else:
        q = (C[0, 0] + C[1, 1] + C[2, 2]) / 3 # trace(C)
        p2 = (C[0, 0] - q)**2 \
             + (C[1, 1] - q)**2 \
             + (C[2, 2] - q)**2 \
             + 2 * p1
        p = sqrt(p2 / 6)
        B = (1 / p) * (C - q * np.eye(3))
        r = determinant3D(B) / 2

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
        eig2 = 3 * q - eig1 - eig3     # since trace(A) = eig1 + eig2 + eig3

    return eig3, eig2, eig1

def eigh_Basar(C):
    la1, la2, la3 = eig(C)

    I1_ = I1(C)
    I3_ = I3(C)

    Cinv = inverse3D(C)

    C1 = la1 * ((C - (I1_ - la1) * np.eye(3) + I3_ * Cinv / la1) / (2 * la1**2 - I1_ * la1 + I3_ / la1))
    C2 = la2 * ((C - (I1_ - la2) * np.eye(3) + I3_ * Cinv / la2) / (2 * la2**2 - I1_ * la2 + I3_ / la2))
    C3 = la3 * ((C - (I1_ - la3) * np.eye(3) + I3_ * Cinv / la3) / (2 * la3**2 - I1_ * la3 + I3_ / la3))

    return la1, la2, la3, C1, C2, C3

def eigh_numpy(C):
    la, u = np.linalg.eigh(C)
    C1 = np.outer(u[:, 0], u[:, 0])
    C2 = np.outer(u[:, 1], u[:, 1])
    C3 = np.outer(u[:, 2], u[:, 2])
    return la[0], la[1], la[2], C1, C2, C3

def test_eigh():
    import time

    det = -1
    while det < 0:
        F = np.random.rand(3, 3)
        det = determinant(F)
    C = F.T @ F

    print(C)

    n = 100

    start_time = time.time()
    for _ in range(n):
        while det < 0:
            F = np.random.rand(3, 3)
            det = determinant(F)
        C = F.T @ F
        la1, la2, la3, C1, C2, C3 = eigh_numpy(C)
    elapsed_time_numpy = time.time() - start_time

    start_time = time.time()
    for _ in range(n):
        while det < 0:
            F = np.random.rand(3, 3)
            det = determinant(F)
        C = F.T @ F
        la1_, la2_, la3_, C1_, C2_, C3_ = eigh_Basar(C)
    elapsed_time_Basar = time.time() - start_time
    
    assert np.allclose(la1, la1_)
    assert np.allclose(la2, la2_)
    assert np.allclose(la3, la3_)
    assert np.allclose(C1, C1_)
    assert np.allclose(C2, C2_)
    assert np.allclose(C3, C3_)

    print(f'elapsed time numpy: {elapsed_time_numpy}')
    print(f'elapsed time Basar: {elapsed_time_Basar}')

def test_quat2mat():
    quat2rot_num = Numerical_derivative(lambda t,x: quat2rot(x), order=2)
    p = np.random.rand(4)
    p = p/np.linalg.norm(p)
    diff = quat2rot_p(p) - quat2rot_num._x(0,p)
    print(np.linalg.norm(diff))

    quat2mat_num = Numerical_derivative(lambda t,x: quat2mat(x))
    p = np.random.rand(4)
    p = p/np.linalg.norm(p)
    diff2 = quat2mat_p(p) - quat2mat_num._x(0,p)
    print(np.linalg.norm(diff2))


if __name__ == "__main__":
    # test_quat2mat()
    test_eigh()