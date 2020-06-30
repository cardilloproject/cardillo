import numpy as np
from math import sqrt
from cardillo.math.numerical_derivative import Numerical_derivative

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
    A = np.zeros((3,3,3))
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

if __name__ == "__main__":
    # tests
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


