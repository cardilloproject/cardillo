from __future__ import annotations
import numpy as np
from math import sin, cos, tan, sqrt, atan2, acos
from cardillo.math import norm, cross3, skew2ax, ax2skew, ax2skew_a, ei


class A_IK_basic:
    """Basic rotations in Euclidean space."""

    def __init__(self, phi: float):
        self.phi = phi
        self.sp = sin(phi)
        self.cp = cos(phi)

    def x(self) -> np.ndarray:
        """Rotation around x-axis."""
        # fmt: off
        return np.array([[1,       0,        0],\
                         [0, self.cp, -self.sp],\
                         [0, self.sp,  self.cp]])
        # fmt: on

    def dx(self) -> np.ndarray:
        """Derivative of Rotation around x-axis."""
        # fmt: off
        return np.array([[0,        0,        0],\
                         [0, -self.sp, -self.cp],\
                         [0,  self.cp, -self.sp]])
        # fmt: on

    def y(self) -> np.ndarray:
        """Rotation around y-axis."""
        # fmt: off
        return np.array([[ self.cp, 0, self.sp],\
                         [       0, 1,       0],\
                         [-self.sp, 0, self.cp]])
        # fmt: on

    def dy(self) -> np.ndarray:
        """Derivative of Rotation around y-axis."""
        # fmt: off
        return np.array([[-self.sp, 0,  self.cp],\
                         [       0, 0,        0],\
                         [-self.cp, 0, -self.sp]])
        # fmt: on

    def z(self) -> np.ndarray:
        """Rotation around z-axis."""
        # fmt: off
        return np.array([[self.cp, -self.sp, 0],\
                         [self.sp,  self.cp, 0],\
                         [      0,        0, 1]])
        # fmt: on

    def dz(self) -> np.ndarray:
        """Derivative of Rotation around z-axis."""
        # fmt: off
        return np.array([[-self.sp,  -self.cp,  0],\
                         [  self.cp,  -self.sp, 0],\
                         [        0,         0, 0]])
        # fmt: on


def rodriguez(psi: np.ndarray) -> np.ndarray:
    """Axis-angle rotation, see Crisfield1999 above (4.1).

    References
    ----------
    Crisfield1999: https://doi.org/10.1098/rspa.1999.0352
    """
    angle = norm(psi)
    if angle > 0:
        sa = np.sin(angle)
        ca = np.cos(angle)
        psi_tilde = ax2skew(psi)
        return (
            np.eye(3)
            + (sa / angle) * psi_tilde
            + ((1.0 - ca) / (angle**2)) * psi_tilde @ psi_tilde
        )
    else:
        return np.eye(3)


def rodriguez_der(psi: np.ndarray) -> np.ndarray:
    """Derivative of the axis-angle rotation, see Crisfield1999 above (4.1). 
    Derivations and final results are given in 

    References
    ----------
    Crisfield1999: https://doi.org/10.1098/rspa.1999.0352 \\
    Gallego2015: https://doi.org/10.1007/s10851-014-0528-x
    """
    angle = norm(psi)

    # Gallego2015 (9)
    R_psi = np.zeros((3, 3, 3))
    if angle > 0:
        R = rodriguez(psi)
        eye_R = np.eye(3) - R
        psi_tilde = ax2skew(psi)
        angle2 = angle * angle
        for i in range(3):
            R_psi[:, :, i] = (
                (psi[i] * psi_tilde + ax2skew(cross3(psi, eye_R[:, i]))) @ R / angle2
            )
    else:
        # Derivative at the identity, see Gallego2015 Section 3.3
        for i in range(3):
            R_psi[:, :, i] = ax2skew(ei(i))

    return R_psi


def tangent_map(psi: np.ndarray) -> np.ndarray:
    """Tangent map, see Crisfield1999 (4.2). Different forms are found in 
    Cardona1988 (38), Ibrahimbegovic1995 (14b) and Barfoot2014 (98). Actually 
    in Ibrahimbegovic1995 and Barfoot2014 T^T is shown!

    References
    ----------
    Crisfield1999: https://doi.org/10.1098/rspa.1999.0352 \\
    Cardona1988: https://doi.org/10.1002/nme.1620261105 \\
    Ibrahimbegovic1995: https://doi.org/10.1002/nme.1620382107 \\
    Barfoot2014: https://doi.org/10.1109/TRO.2014.2298059
    """
    angle = norm(psi)
    if angle > 0:
        sa = sin(angle)
        ca = cos(angle)
        T = (
            (sa / angle) * np.eye(3)
            - ((1.0 - ca) / angle**2) * ax2skew(psi)
            + ((1.0 - sa / angle) / angle**2) * np.outer(psi, psi)
        )
        return T
    else:
        return np.eye(3)


def inverse_tangent_map(psi: np.ndarray) -> np.ndarray:
    """Inverse tangent map, see Cardona1988 (45), Ibrahimbegovic1995 (28) and Barfoot2014 (99).
    Actually in Ibrahimbegovic1995 and Barfoot2014 T^{-T} is shown!

    References
    ----------
    Cardona1988: https://doi.org/10.1002/nme.1620261105 \\
    Ibrahimbegovic1995: https://doi.org/10.1002/nme.1620382107 \\
    Barfoot2014: https://doi.org/10.1109/TRO.2014.2298059
    """
    angle = norm(psi)
    if angle > 0:
        ta = tan(angle / 2)
        T_inv = (
            (angle / (2 * ta)) * np.eye(3)
            + 0.5 * ax2skew(psi)
            + (1 - (angle / 2) / ta) * np.outer(psi / angle, psi / angle)
        )
        return T_inv
    else:
        return np.eye(3)  # Cardona1988 after (46)


def tangent_map_s(psi: np.ndarray, psi_s: np.ndarray) -> np.ndarray:
    """Derivative of tangent map w.r.t. arc length coordinate s, see
    Ibrahimbegović1995 (71). Actually in Ibrahimbegovic1995 (28) T_s^{T}
    is shown!

    References
    ----------
    Ibrahimbegovic1995: https://doi.org/10.1002/nme.1620382107
    """
    angle = norm(psi)
    if angle > 0:
        sa = sin(angle)
        ca = cos(angle)
        c1 = (angle * ca - sa) / angle**3
        c2 = (angle * sa + 2 * ca - 2) / angle**4
        c3 = (3 * sa - 2 * angle - angle * ca) / angle**5
        c4 = (1 - ca) / angle**2
        c5 = (angle - sa) / angle**3

        return (
            c1 * np.outer(psi_s, psi)
            + c2 * np.outer(cross3(psi, psi_s), psi)
            + c3 * (psi @ psi_s) * np.outer(psi, psi)
            - c4 * ax2skew(psi_s)
            + c5 * (psi @ psi_s) * np.eye(3)
            + c5 * np.outer(psi, psi_s)
        ).T  #  transpose of Ibrahimbegović1995 (71)
    else:
        return np.zeros((3, 3))  # Cardona1988 after (46)


def Spurrier(R: np.ndarray) -> np.ndarray:
    """
    Spurrier's algorithm to extract the unit quaternion from a given rotation
    matrix, see Spurrier19978, Simo1986 Table 12 and Crisfield1997 Section 16.10.

    References
    ----------
    Spurrier19978: https://arc.aiaa.org/doi/10.2514/3.57311 \\
    Simo1986: https://doi.org/10.1016/0045-7825(86)90079-4 \\
    Crisfield1997: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    A = np.array([trace, R[0, 0], R[1, 1], R[2, 2]])
    idx = A.argmax()
    a = A[idx]

    if idx > 0:
        i = idx - 1
        # make i, j, k a cyclic permutation of 0, 1, 2
        i, j, k = np.roll(np.arange(3), -i)

        q = np.zeros(4)
        q[i + 1] = sqrt(0.5 * a + 0.25 * (1 - trace))
        q[0] = 0.25 * (R[k, j] - R[j, k]) / q[i + 1]
        q[j + 1] = 0.25 * (R[j, i] + R[i, j]) / q[i + 1]
        q[k + 1] = 0.25 * (R[k, i] + R[i, k]) / q[i + 1]
    else:
        q0 = 0.5 * sqrt(1 + trace)
        q1 = 0.25 * (R[2, 1] - R[1, 2]) / q0
        q2 = 0.25 * (R[0, 2] - R[2, 0]) / q0
        q3 = 0.25 * (R[1, 0] - R[0, 1]) / q0
        q = np.array([q0, q1, q2, q3])

    return q


def quat2axis_angle(Q: np.ndarray) -> np.ndarray:
    """Extract the rotation vector psi for a given quaterion Q = [q0, q] in
    accordance with Wiki2021.

    References
    ----------
    Wiki2021: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Recovering_the_axis-angle_representation
    """
    q0, vq = Q[0], Q[1:]
    q = norm(vq)
    if q > 0:
        axis = vq / q
        angle = 2 * atan2(q, q0)
        return angle * axis
    else:
        return np.zeros(3)


def rodriguez_inv(R: np.ndarray) -> np.ndarray:
    # trace = R[0, 0] + R[1, 1] + R[2, 2]
    # psi = acos(0.5 * (trace - 1.))
    # if psi > 0:
    #     return skew2ax(0.5 * psi / sin(psi) * (R - R.T))
    # else:
    #     return np.zeros(3, dtype=float)

    return quat2axis_angle(Spurrier(R))

    # # alternative formulation using scipy's Rotation module
    # from scipy.spatial.transform import Rotation
    # return Rotation.from_matrix(R).as_rotvec()


def smallest_rotation(a0: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Rotation matrix that rotates an unit vector a0 / ||a0|| to another unit vector
    a / ||a||, see Crisfield1996 16.13 and (16.104). This rotation is sometimes
    referred to 'smallest rotation'. Can we use the SVD proposed by eigen3?

    References
    ----------
    Crisfield1996: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf
    eigen3: https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Quaternion.h#L633-669
    """
    a0_bar = a0 / norm(a0)
    a_bar = a / norm(a)

    # ########################
    # # Crisfield1996 (16.104)
    # ########################
    # e_tilde = ax2skew(e)
    # return np.eye(3) + e_tilde + (e_tilde @ e_tilde) / (1 + a0_bar @ a_bar)

    ########################
    # Crisfield1996 (16.105)
    ########################
    cos_psi = a0_bar @ a_bar
    denom = 1 + cos_psi
    if denom > 0:
        e = cross3(a0_bar, a_bar)
        return cos_psi * np.eye(3) + ax2skew(e) + np.outer(e, e) / denom
    else:
        print("svd case")
        M = np.vstack((a0_bar, a_bar))
        U, S, Vh = np.linalg.svd(M)
        axis = Vh[2]
        psi = np.arccos(cos_psi)
        return rodriguez(psi * axis)


class Rotor:
    """Rotor from geometric algebra, see https://marctenbosch.com/quaternions/#h_0."""

    @staticmethod
    def fromMatrix(A: np.ndarray):
        return Rotor(Spurrier(A))

    @staticmethod
    def fromRotor(R: Rotor):
        return Rotor(R.data().copy())

    @staticmethod
    def fromVector(v: np.ndarray):
        return Rotor(np.array([0, v]))

    @staticmethod
    def from2Vectors(u: np.ndarray, v: np.ndarray):
        return Rotor(np.array([1.0 + u @ v, *cross3(u, v)]))

    @staticmethod
    def fromComponents(a: float, b: np.ndarray):
        return Rotor(np.array([a, *b]))

    def __init__(self, data=None) -> None:
        if data is None:
            self.__data = np.array([1, 0, 0, 0], dtype=float)
        else:
            # ensure float ndarray data
            self.__data = np.asarray(data, dtype=float)

            # normalize rotor
            l = norm(self.__data)
            if l > 0:
                self.__data /= l

    def __call__(self):
        return self.__data

    @property
    def a(self) -> float:
        return self.__data[0]

    @a.setter
    def a(self, value: float):
        self.__data[0] = value

    @property
    def b(self) -> np.ndarray:
        return self.__data[1:]

    @b.setter
    def b(self, value: np.ndarray):
        self.__data[1:] = value

    def __str__(self):
        return f"{self.a} + " + f"{self.b}"

    def __repr__(self):
        return f'Rotor("{self.__str__}")'

    def __mul__(self, scalar: float) -> Rotor:
        """Scalar product."""
        return Rotor(self() * scalar)

    def __matmul__(self, other) -> Rotor:
        """Inner product."""
        return self() @ other()

    def __xor__(self, other) -> np.ndarray:
        """Outer product."""
        return cross3(self.b, other.b)

    def __mod__(self, other) -> Rotor:
        """Geometric product."""
        res = Rotor()
        res.a = self * other
        res.b = self ^ other
        return res

    def __invert__(self) -> Rotor:
        """Reverse rotor."""
        return Rotor.fromComponents(self.a, -self.b)

    def rotate(self, r) -> Rotor:
        """Rotate a vector or another rotor."""
        if isinstance(r, Rotor):
            return self @ r @ ~self
        else:
            return self @ Rotor.fromVector(r) @ ~self

    # TODO: Is this correct?
    def toMatrix(self) -> np.ndarray:
        """Compute rotation matrix defined by rotor."""
        a = self.a
        b_tilde = ax2skew(self.b)
        return np.eye(3) + 2 * (b_tilde @ b_tilde + a * b_tilde)


##########################################
# TODO: Refactor these and add references!
##########################################
def quat2mat(p):
    p0, p1, p2, p3 = p
    # fmt: off
    return np.array([
        [p0, -p1, -p2, -p3], 
        [p1,  p0, -p3,  p2], 
        [p2,  p3,  p0, -p1], 
        [p3, -p2,  p1,  p0]]
    )
    # fmt: on


def quat2mat_p(p):
    A_p = np.zeros((4, 4, 4))
    A_p[:, :, 0] = np.eye(4)
    A_p[0, 1:, 1:] = -np.eye(3)
    A_p[1:, 0, 1:] = np.eye(3)
    A_p[1:, 1:, 1:] = ax2skew_a()

    return A_p


def quat2rot(p):
    p_ = p / norm(p)
    v_p_tilde = ax2skew(p_[1:])
    return np.eye(3) + 2 * (v_p_tilde @ v_p_tilde + p_[0] * v_p_tilde)


def quat2rot_p(p):
    norm_p = norm(p)
    q = p / norm_p
    v_q_tilde = ax2skew(q[1:])
    v_q_tilde_v_q = ax2skew_a()
    q_p = np.eye(4) / norm_p - np.outer(p, p) / (norm_p**3)

    A_q = np.zeros((3, 3, 4))
    A_q[:, :, 0] = 2 * v_q_tilde
    A_q[:, :, 1:] += np.einsum("ijk,jl->ilk", v_q_tilde_v_q, 2 * v_q_tilde)
    A_q[:, :, 1:] += np.einsum("ij,jkl->ikl", 2 * v_q_tilde, v_q_tilde_v_q)
    A_q[:, :, 1:] += 2 * (q[0] * v_q_tilde_v_q)

    return np.einsum("ijk,kl->ijl", A_q, q_p)


def axis_angle2quat(axis, angle):
    n = axis / norm(axis)
    return np.concatenate([[cos(angle / 2)], sin(angle / 2) * n])


def test_smallest_rotation():
    from cardillo.math import e1, e2, e3, pi, sin, cos

    a0 = e1
    a_fun = lambda t: cos(t) * e1 + sin(t) * e2

    # random rotation axis
    psi = np.random.rand(3)
    psi = psi / norm(psi)
    print(f"psi: {psi}")

    def a_fun(t):
        R = rodriguez(t * pi * psi)
        return R @ e1

    num = 100
    ts = np.linspace(0, 1.0 * pi, num=num)

    as_ = np.array([a_fun(t) for t in ts])

    Rs = np.array([smallest_rotation(a0, a) for a in as_]).transpose(1, 2, 0)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="3d"))
    ax.quiver(*np.zeros((3, num)), *Rs[:, 0], color="red")
    ax.quiver(*np.zeros((3, num)), *Rs[:, 1], color="green")
    ax.quiver(*np.zeros((3, num)), *Rs[:, 2], color="blue")
    plt.show()


def test_rotor():
    from cardillo.math import e1, e2, e3

    r1 = Rotor.from2Vectors(e1, e2)
    # r1 = Rotor.from2Vectors(e1, e3)
    r2 = Rotor()
    # r2 = Rotor(u=e1, v=e1)
    print(f"r1: {r1}")
    print(f"r2: {r2}")

    print(f"r1 * r2: {r1 * r2}")
    print(f"r1 ^ r2: {r1 ^ r2}")
    print(f"r1 @ r2: {r1 @ r2}")
    print(f"r1.rotate(r2): {r1.rotate(r2)}")
    print(f"r1.toMatrix():\n{r1.toMatrix()}")
    print(f"r2.toMatrix():\n{r2.toMatrix()}")


if __name__ == "__main__":
    test_smallest_rotation()
    # test_rotor()
