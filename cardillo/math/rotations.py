from __future__ import annotations
import numpy as np
from math import tan, sqrt, atan2
from cardillo.math import norm, cross3, ax2skew, trace3, ax2skew_a, ei, LeviCivita3

# for small angles we use first order approximations of the equations since
# most of the SO(3) and SE(3) equations get singular for psi -> 0.
angle_singular = 1.0e-6


def Exp_SO3(psi: np.ndarray) -> np.ndarray:
    """SO(3) exponential function, see Crisfield1999 above (4.1) and 
    Park2005 (12).

    References
    ----------
    Crisfield1999: https://doi.org/10.1098/rspa.1999.0352 \\
    Park2005: https://doi.org/10.1109/TRO.2005.852253
    """
    angle = norm(psi)
    if angle > angle_singular:
        # Park2005 (12)
        sa = np.sin(angle)
        ca = np.cos(angle)
        alpha = sa / angle
        beta2 = (1.0 - ca) / (angle * angle)
        psi_tilde = ax2skew(psi)
        return (
            np.eye(3, dtype=float) + alpha * psi_tilde + beta2 * psi_tilde @ psi_tilde
        )
    else:
        # first order approximation
        return np.eye(3, dtype=float) + ax2skew(psi)


def Exp_SO3_psi(psi: np.ndarray) -> np.ndarray:
    """Derivative of the axis-angle rotation found in Crisfield1999 above (4.1). 
    Derivations and final results are given in Gallego2015 (9).

    References
    ----------
    Crisfield1999: https://doi.org/10.1098/rspa.1999.0352 \\
    Gallego2015: https://doi.org/10.1007/s10851-014-0528-x
    """
    angle = norm(psi)

    # # Gallego2015 (9)
    # A_psi = np.zeros((3, 3, 3), dtype=float)
    # if isclose(angle, 0.0):
    #     # Derivative at the identity, see Gallego2015 Section 3.3
    #     for i in range(3):
    #         A_psi[:, :, i] = ax2skew(ei(i))
    # else:
    #     A = Exp_SO3(psi)
    #     eye_A = np.eye(3) - A
    #     psi_tilde = ax2skew(psi)
    #     angle2 = angle * angle
    #     for i in range(3):
    #         A_psi[:, :, i] = (
    #             (psi[i] * psi_tilde + ax2skew(cross3(psi, eye_A[:, i]))) @ A / angle2
    #         )

    A_psi = np.zeros((3, 3, 3), dtype=psi.dtype)
    if angle > angle_singular:
        angle2 = angle * angle
        sa = np.sin(angle)
        ca = np.cos(angle)
        alpha = sa / angle
        alpha_psik = (ca - alpha) / angle2
        beta = 2.0 * (1.0 - ca) / angle2
        beta2_psik = (alpha - beta) / angle2

        psi_tilde = ax2skew(psi)
        psi_tilde2 = psi_tilde @ psi_tilde

        ############################
        # alpha * psi_tilde (part I)
        ############################
        A_psi[0, 2, 1] = A_psi[1, 0, 2] = A_psi[2, 1, 0] = alpha
        A_psi[0, 1, 2] = A_psi[1, 2, 0] = A_psi[2, 0, 1] = -alpha

        #############################
        # alpha * psi_tilde (part II)
        #############################
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    A_psi[i, j, k] += psi_tilde[i, j] * psi[k] * alpha_psik

        ###############################
        # beta2 * psi_tilde @ psi_tilde
        ###############################
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    A_psi[i, j, k] += psi_tilde2[i, j] * psi[k] * beta2_psik
                    for l in range(3):
                        A_psi[i, j, k] += (
                            0.5
                            * beta
                            * (
                                LeviCivita3(k, l, i) * psi_tilde[l, j]
                                + psi_tilde[l, i] * LeviCivita3(k, l, j)
                            )
                        )
    else:
        ###################
        # alpha * psi_tilde
        ###################
        A_psi[0, 2, 1] = A_psi[1, 0, 2] = A_psi[2, 1, 0] = 1.0
        A_psi[0, 1, 2] = A_psi[1, 2, 0] = A_psi[2, 0, 1] = -1.0

    return A_psi

    # A_psi_num = approx_fprime(psi, Exp_SO3, method="cs", eps=1.0e-10)
    # diff = A_psi - A_psi_num
    # error = np.linalg.norm(diff)
    # if error > 1.0e-10:
    #     print(f"error Exp_SO3_psi: {error}")
    # return A_psi_num


def Log_SO3(A: np.ndarray) -> np.ndarray:
    ca = 0.5 * (trace3(A) - 1.0)
    ca = np.clip(ca, -1, 1)  # clip to [-1, 1] for arccos!
    angle = np.arccos(ca)

    # fmt: off
    psi = 0.5 * np.array([
        A[2, 1] - A[1, 2],
        A[0, 2] - A[2, 0],
        A[1, 0] - A[0, 1]
    ], dtype=A.dtype)
    # fmt: on

    if angle > angle_singular:
        psi *= angle / np.sin(angle)
    return psi


def Log_SO3_A(A: np.ndarray) -> np.ndarray:
    """Derivative of the SO(3) Log map. See Blanco2010 (10.11)

    References:
    ===========
    Claraco2010: https://doi.org/10.48550/arXiv.2103.15980
    """
    ca = 0.5 * (trace3(A) - 1.0)
    ca = np.clip(ca, -1, 1)  # clip to [-1, 1] for arccos!
    angle = np.arccos(ca)

    psi_A = np.zeros((3, 3, 3), dtype=A.dtype)
    if angle > angle_singular:
        sa = np.sin(angle)
        b = 0.5 * angle / sa

        # fmt: off
        a = (angle * ca - sa) / (4.0 * sa**3) * np.array([
            A[2, 1] - A[1, 2],
            A[0, 2] - A[2, 0],
            A[1, 0] - A[0, 1]
        ], dtype=A.dtype)
        # fmt: on

        psi_A[0, 0, 0] = psi_A[0, 1, 1] = psi_A[0, 2, 2] = a[0]
        psi_A[1, 0, 0] = psi_A[1, 1, 1] = psi_A[1, 2, 2] = a[1]
        psi_A[2, 0, 0] = psi_A[2, 1, 1] = psi_A[2, 2, 2] = a[2]

        psi_A[0, 2, 1] = psi_A[1, 0, 2] = psi_A[2, 1, 0] = b
        psi_A[0, 1, 2] = psi_A[1, 2, 0] = psi_A[2, 0, 1] = -b
    else:
        psi_A[0, 2, 1] = psi_A[1, 0, 2] = psi_A[2, 1, 0] = 0.5
        psi_A[0, 1, 2] = psi_A[1, 2, 0] = psi_A[2, 0, 1] = -0.5

    return psi_A

    # psi_A_num = approx_fprime(A, Log_SO3, method="cs", eps=1.0e-10)
    # diff = psi_A - psi_A_num
    # error = np.linalg.norm(diff)
    # print(f"error Log_SO3_A: {error}")
    # return psi_A_num


def T_SO3(psi: np.ndarray) -> np.ndarray:
    angle = norm(psi)
    if angle > angle_singular:
        # Park2005 (19), actually its the transposed!
        sa = np.sin(angle)
        ca = np.cos(angle)
        psi_tilde = ax2skew(psi)
        alpha = sa / angle
        angle2 = angle * angle
        beta2 = (1.0 - ca) / angle2
        return (
            np.eye(3, dtype=float)
            - beta2 * psi_tilde
            + ((1.0 - alpha) / angle2) * psi_tilde @ psi_tilde
        )

        # # Barfoot2014 (98), actually its the transposed!
        # sa = np.sin(angle)
        # ca = np.cos(angle)
        # sinc = sa / angle
        # n = psi / angle
        # return (
        #     sinc * np.eye(3, dtype=float)
        #     - ((1.0 - ca) / angle) * ax2skew(n)
        #     + (1.0 - sinc) * np.outer(n, n)
        # )
    else:
        # first order approximation
        return np.eye(3, dtype=float) - 0.5 * ax2skew(psi)


def T_SO3_psi(psi: np.ndarray) -> np.ndarray:
    T_SO3_psi = np.zeros((3, 3, 3), dtype=float)

    angle = norm(psi)
    if angle > angle_singular:
        sa = np.sin(angle)
        ca = np.cos(angle)
        alpha = sa / angle
        angle2 = angle * angle
        angle4 = angle2 * angle2
        beta2 = (1.0 - ca) / angle2
        beta2_psik = (2.0 * beta2 - alpha) / angle2
        c = (1.0 - alpha) / angle2
        c_psik = (3.0 * alpha - 2.0 - ca) / angle4

        psi_tilde = ax2skew(psi)
        psi_tilde2 = psi_tilde @ psi_tilde

        ####################
        # -beta2 * psi_tilde
        ####################
        T_SO3_psi[0, 1, 2] = T_SO3_psi[1, 2, 0] = T_SO3_psi[2, 0, 1] = beta2
        T_SO3_psi[0, 2, 1] = T_SO3_psi[1, 0, 2] = T_SO3_psi[2, 1, 0] = -beta2
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_SO3_psi[i, j, k] += psi_tilde[i, j] * psi[k] * beta2_psik

        ##################################################
        # ((1.0 - alpha) / angle2) * psi_tilde @ psi_tilde
        ##################################################
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_SO3_psi[i, j, k] += psi_tilde2[i, j] * psi[k] * c_psik
                    for l in range(3):
                        T_SO3_psi[i, j, k] += c * (
                            LeviCivita3(k, l, i) * psi_tilde[l, j]
                            + psi_tilde[l, i] * LeviCivita3(k, l, j)
                        )
    else:
        ####################
        # -beta2 * psi_tilde
        ####################
        T_SO3_psi[0, 1, 2] = T_SO3_psi[1, 2, 0] = T_SO3_psi[2, 0, 1] = 0.5
        T_SO3_psi[0, 2, 1] = T_SO3_psi[1, 0, 2] = T_SO3_psi[2, 1, 0] = -0.5

    return T_SO3_psi

    # T_SO3_psi_num = approx_fprime(psi, T_SO3, method="cs", eps=1.0e-10)
    # diff = T_SO3_psi - T_SO3_psi_num
    # error = np.linalg.norm(diff)
    # if error > 1.0e-8:
    #     print(f"error T_SO3_psi: {error}")
    # return T_SO3_psi_num


def T_SO3_dot(psi: np.ndarray, psi_dot: np.ndarray) -> np.ndarray:
    """Derivative of tangent map w.r.t. scalar argument of rotationv ector, see
    Ibrahimbegović1995 (71). Actually in Ibrahimbegovic1995 (28) T_s^{T}
    is shown!

    References
    ----------
    Ibrahimbegovic1995: https://doi.org/10.1002/nme.1620382107
    """
    angle = norm(psi)
    if angle > 0:
        sa = np.sin(angle)
        ca = np.cos(angle)
        c1 = (angle * ca - sa) / angle**3
        c2 = (angle * sa + 2 * ca - 2) / angle**4
        c3 = (3 * sa - 2 * angle - angle * ca) / angle**5
        c4 = (1 - ca) / angle**2
        c5 = (angle - sa) / angle**3

        return (
            c1 * np.outer(psi_dot, psi)
            + c2 * np.outer(cross3(psi, psi_dot), psi)
            + c3 * (psi @ psi_dot) * np.outer(psi, psi)
            - c4 * ax2skew(psi_dot)
            + c5 * (psi @ psi_dot) * np.eye(3)
            + c5 * np.outer(psi, psi_dot)
        ).T  #  transpose of Ibrahimbegović1995 (71)
    else:
        return np.zeros((3, 3), dtype=float)  # Cardona1988 after (46)


def T_SO3_inv(psi: np.ndarray) -> np.ndarray:
    angle = norm(psi)
    psi_tilde = ax2skew(psi)
    if angle > angle_singular:
        # Park2005 (19), actually its the transposed!
        gamma = 0.5 * angle / (np.tan(0.5 * angle))
        return (
            np.eye(3, dtype=float)
            + 0.5 * psi_tilde
            + ((1.0 - gamma) / (angle * angle)) * psi_tilde @ psi_tilde
        )

        # # Barfoot2014 (98), actually its the transposed!
        # angle2 = 0.5 * angle
        # cot = 1.0 / tan(angle2)
        # n = psi / angle
        # return (
        #     angle2 * cot * np.eye(3, dtype=float)
        #     + angle2 * ax2skew(n)
        #     + (1.0 - angle2 * cot) * np.outer(n, n)
        # )
    else:
        # first order approximation
        return np.eye(3, dtype=float) + 0.5 * psi_tilde


def T_SO3_inv_psi(psi: np.ndarray) -> np.ndarray:
    T_SO3_inv_psi = np.zeros((3, 3, 3), dtype=psi.dtype)

    #################
    # 0.5 * psi_tilde
    #################
    T_SO3_inv_psi[0, 1, 2] = T_SO3_inv_psi[1, 2, 0] = T_SO3_inv_psi[2, 0, 1] = -0.5
    T_SO3_inv_psi[0, 2, 1] = T_SO3_inv_psi[1, 0, 2] = T_SO3_inv_psi[2, 1, 0] = 0.5

    angle = norm(psi)
    if angle > angle_singular:
        psi_tilde = ax2skew(psi)
        psi_tilde2 = psi_tilde @ psi_tilde
        cot = 1.0 / np.tan(0.5 * angle)
        gamma = 0.5 * angle * cot
        angle2 = angle * angle
        c = (1.0 - gamma) / angle2
        # c_psi_k = (
        #     -2.0 * c / angle2
        #     - cot / (2.0 * angle2 * angle)
        #     + 1.0 / (4.0 * angle2 * np.sin(0.5 * angle) ** 2)
        # )
        c_psi_k = (
            1.0 / (4.0 * np.sin(0.5 * angle) ** 2) - cot / (2.0 * angle) - 2.0 * c
        ) / angle2

        ###########################################################
        # ((1.0 - gamma) / (angle * angle)) * psi_tilde @ psi_tilde
        ###########################################################
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_SO3_inv_psi[i, j, k] += psi_tilde2[i, j] * psi[k] * c_psi_k
                    for l in range(3):
                        T_SO3_inv_psi[i, j, k] += c * (
                            LeviCivita3(i, k, l) * psi_tilde[l, j]
                            + psi_tilde[i, l] * LeviCivita3(l, k, j)
                        )

    return T_SO3_inv_psi

    # T_SO3_inv_psi_num = approx_fprime(psi, T_SO3_inv, eps=1.0e-10, method="cs")
    # diff = T_SO3_inv_psi - T_SO3_inv_psi_num
    # error = np.linalg.norm(diff)
    # if error > 1.0e-10:
    #     print(f"error T_SO3_inv_psi: {error}")
    # return T_SO3_inv_psi_num


def SE3(A_IK: np.ndarray, r_OP: np.ndarray) -> np.ndarray:
    H = np.zeros((4, 4), dtype=np.common_type(A_IK, r_OP))
    H[:3, :3] = A_IK
    H[:3, 3] = r_OP
    H[3, 3] = 1.0
    return H


def SE3inv(H: np.ndarray) -> np.ndarray:
    A_IK = H[:3, :3]
    r_OP = H[:3, 3]
    return SE3(A_IK.T, -A_IK.T @ r_OP)


def Exp_SE3(h: np.ndarray) -> np.ndarray:
    r = h[:3]
    psi = h[3:]

    H = np.zeros((4, 4), dtype=h.dtype)
    H[:3, :3] = Exp_SO3(psi)
    H[:3, 3] = T_SO3(psi).T @ r
    H[3, 3] = 1.0

    return H


def Exp_SE3_h(h: np.ndarray) -> np.ndarray:
    r = h[:3]
    psi = h[3:]

    H_h = np.zeros((4, 4, 6), dtype=h.dtype)
    H_h[:3, :3, 3:] = Exp_SO3_psi(psi)
    H_h[:3, 3, 3:] = np.einsum("k,kij->ij", r, T_SO3_psi(psi))
    H_h[:3, 3, :3] = T_SO3(psi).T
    return H_h

    # H_h_num =  approx_fprime(h, Exp_SE3, method="cs", eps=1.0e-10)
    # diff = H_h - H_h_num
    # error = np.linalg.norm(diff)
    # if error > 1.0e-10:
    #     print(f"error Exp_SE3_h: {error}")
    # return H_h_num


def Log_SE3(H: np.ndarray) -> np.ndarray:
    A = H[:3, :3]
    r = H[:3, 3]
    psi = Log_SO3(A)
    h = np.concatenate((T_SO3_inv(psi).T @ r, psi))
    return h


def Log_SE3_H(H: np.ndarray) -> np.ndarray:
    A = H[:3, :3]
    r = H[:3, 3]
    psi = Log_SO3(A)
    psi_A = Log_SO3_A(A)
    h_H = np.zeros((6, 4, 4), dtype=H.dtype)
    h_H[:3, :3, :3] = np.einsum("l,lim,mjk", r, T_SO3_inv_psi(psi), psi_A)
    h_H[:3, :3, 3] = T_SO3_inv(psi).T
    h_H[3:, :3, :3] = psi_A
    return h_H

    # h_H_num = approx_fprime(H, Log_SE3, method="cs", eps=1.0e-10)
    # diff = h_H - h_H_num
    # error = np.linalg.norm(diff)
    # if error > 1.0e-10:
    #     print(f"error Log_SE3_H: {error}")
    # return h_H_num


def U(a, b):
    a_tilde = ax2skew(a)

    b2 = b @ b
    if b2 > 0:
        abs_b = np.sqrt(b2)
        alpha = np.sin(abs_b) / abs_b
        beta = 2.0 * (1.0 - np.cos(abs_b)) / b2

        b_tilde = ax2skew(b)

        # Sonneville2014 (A.12); how is this related to Park2005 (20) and (21)?
        return (
            -0.5 * beta * a_tilde
            + (1.0 - alpha) * (a_tilde @ b_tilde + b_tilde @ a_tilde) / b2
            + ((b @ a) / b2)
            * (
                (beta - alpha) * b_tilde
                + (0.5 * beta - 3.0 * (1.0 - alpha) / b2) * b_tilde @ b_tilde
            )
        )
    else:
        return -0.5 * a_tilde  # Soneville2014


def T_SE3(h: np.ndarray) -> np.ndarray:
    r = h[:3]
    psi = h[3:]

    T = np.zeros((6, 6), dtype=h.dtype)
    T[:3, :3] = T[3:, 3:] = T_SO3(psi)
    T[:3, 3:] = U(r, psi)
    return T


class A_IK_basic:
    """Basic rotations in Euclidean space."""

    def __init__(self, phi: float):
        self.phi = phi
        self.sp = np.sin(phi)
        self.cp = np.cos(phi)

    def x(self) -> np.ndarray:
        """Rotation around x-axis."""
        # fmt: off
        return np.array([[1,       0,        0],
                         [0, self.cp, -self.sp],
                         [0, self.sp,  self.cp]])
        # fmt: on

    def dx(self) -> np.ndarray:
        """Derivative of Rotation around x-axis."""
        # fmt: off
        return np.array([[0,        0,        0],
                         [0, -self.sp, -self.cp],
                         [0,  self.cp, -self.sp]])
        # fmt: on

    def y(self) -> np.ndarray:
        """Rotation around y-axis."""
        # fmt: off
        return np.array([[ self.cp, 0, self.sp],
                         [       0, 1,       0],
                         [-self.sp, 0, self.cp]])
        # fmt: on

    def dy(self) -> np.ndarray:
        """Derivative of Rotation around y-axis."""
        # fmt: off
        return np.array([[-self.sp, 0,  self.cp],
                         [       0, 0,        0],
                         [-self.cp, 0, -self.sp]])
        # fmt: on

    def z(self) -> np.ndarray:
        """Rotation around z-axis."""
        # fmt: off
        return np.array([[self.cp, -self.sp, 0],
                         [self.sp,  self.cp, 0],
                         [      0,        0, 1]])
        # fmt: on

    def dz(self) -> np.ndarray:
        """Derivative of Rotation around z-axis."""
        # fmt: off
        return np.array([[-self.sp,  -self.cp, 0],
                         [  self.cp, -self.sp, 0],
                         [        0,        0, 0]])
        # fmt: on


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


# TODO: Remove these function since they are already replaced by the ones above.
if False:

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
                    (psi[i] * psi_tilde + ax2skew(cross3(psi, eye_R[:, i])))
                    @ R
                    / angle2
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
            sa = np.sin(angle)
            ca = np.cos(angle)
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
            sa = np.sin(angle)
            ca = np.cos(angle)
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

    def rodriguez_inv(R: np.ndarray) -> np.ndarray:
        # trace = R[0, 0] + R[1, 1] + R[2, 2]
        # psi = acos(0.5 * (trace - 1.))
        # if psi > 0:
        #     return skew2ax(0.5 * psi / np.sin(psi) * (R - R.T))
        # else:
        #     return np.zeros(3, dtype=float)

        return quat2axis_angle(Spurrier(R))

        # # alternative formulation using scipy's Rotation module
        # from scipy.spatial.transform import Rotation
        # return Rotation.from_matrix(R).as_rotvec()


def smallest_rotation(
    a0: np.ndarray, a: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Rotation matrix that rotates an unit vector a0 / ||a0|| to another unit vector
    a / ||a||, see Crisfield1996 16.13 and (16.104). This rotation is sometimes
    referred to 'smallest rotation'.

    References
    ----------
    Crisfield1996: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf
    """
    if normalize:
        a0 = a0 / norm(a0)
        a = a / norm(a)

    # ########################
    # # Crisfield1996 (16.104)
    # ########################
    # e_tilde = ax2skew(e)
    # return np.eye(3) + e_tilde + (e_tilde @ e_tilde) / (1 + a0_bar @ a_bar)

    ########################
    # Crisfield1996 (16.105)
    ########################
    cos_psi = a0 @ a
    denom = 1.0 + cos_psi
    e = cross3(a0, a)
    # return cos_psi * np.eye(3, dtype=e.dtype) + ax2skew(e) + np.outer(e, e) / denom
    e_tilde = ax2skew(e)
    return np.eye(3, dtype=e.dtype) + e_tilde + e_tilde @ e_tilde / denom


##########################################
# TODO: Refactor these and add references!
##########################################
def quatprod(P, Q):
    """Quaternion product, see Egeland2002 (6.190).

    References:
    -----------
    Egenland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
    """
    p0 = P[0]
    p = P[1:]
    q0 = Q[0]
    q = Q[1:]

    z0 = p0 * q0 - p @ q
    z = p0 * q + q0 * p + cross3(p, q)
    return np.array([z0, *z])


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
    """Rotation matrix defined by (non unit) quaternion, see Egeland2002 (6.199).

    References:
    -----------
    Egenland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
    """
    p_ = p / norm(p)
    v_p_tilde = ax2skew(p_[1:])
    return np.eye(3, dtype=p.dtype) + 2.0 * (v_p_tilde @ v_p_tilde + p_[0] * v_p_tilde)


def quat2rot_p(p):
    norm_p = norm(p)
    q = p / norm_p
    v_q_tilde = ax2skew(q[1:])
    v_q_tilde_v_q = ax2skew_a()
    q_p = np.eye(4) / norm_p - np.outer(p, p) / (norm_p**3)

    A_q = np.zeros((3, 3, 4), dtype=p.dtype)
    A_q[:, :, 0] = 2 * v_q_tilde
    A_q[:, :, 1:] += np.einsum("ijk,jl->ilk", v_q_tilde_v_q, 2 * v_q_tilde)
    A_q[:, :, 1:] += np.einsum("ij,jkl->ikl", 2 * v_q_tilde, v_q_tilde_v_q)
    A_q[:, :, 1:] += 2 * (q[0] * v_q_tilde_v_q)

    return np.einsum("ijk,kl->ijl", A_q, q_p)


def axis_angle2quat(axis, angle):
    n = axis / norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * n])
