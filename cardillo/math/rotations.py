import numpy as np
from cardillo.math import norm, cross3, ax2skew, ax2skew_a, LeviCivita3, ax2skew_squared

# for small angles we use first order approximations of the equations since
# most of the SO(3) and SE(3) equations get singular for psi -> 0.
# angle_singular = 1.0e-6
angle_singular = 0.0

eye3 = np.eye(3, dtype=float)


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
        return eye3 + alpha * ax2skew(psi) + beta2 * ax2skew_squared(psi)
    else:
        # first order approximation
        return eye3 + ax2skew(psi)


def Exp_SO3_psi(psi: np.ndarray) -> np.ndarray:
    """Derivative of the axis-angle rotation found in Crisfield1999 above (4.1). 
    Derivations and final results are given in Gallego2015 (9).

    References
    ----------
    Crisfield1999: https://doi.org/10.1098/rspa.1999.0352 \\
    Gallego2015: https://doi.org/10.1007/s10851-014-0528-x
    """
    angle = norm(psi)

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
        psi_tilde2 = ax2skew_squared(psi)

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


def Log_SO3(A: np.ndarray) -> np.ndarray:
    ca = 0.5 * (np.trace(A) - 1.0)
    ca = np.clip(ca, -1, 1)  # clip to [-1, 1] for arccos!
    angle = np.arccos(ca)

    # fmt: off
    psi = 0.5 * np.array([
        A[2, 1] - A[1, 2],
        A[0, 2] - A[2, 0],
        A[1, 0] - A[0, 1]
    ], dtype=A.dtype)
    # fmt: on

    if angle > angle_singular and angle < np.pi:
        psi *= angle / np.sqrt(1.0 - ca * ca)
    return psi


def Log_SO3_A(A: np.ndarray) -> np.ndarray:
    """Derivative of the SO(3) Log map. See Blanco-Claraco2010 (10.11)

    References:
    ===========
    Blanco-Claraco2010: https://doi.org/10.48550/arXiv.2103.15980
    """
    ca = 0.5 * (np.trace(A) - 1.0)
    ca = np.clip(ca, -1, 1)  # clip to [-1, 1] for arccos!
    angle = np.arccos(ca)

    psi_A = np.zeros((3, 3, 3), dtype=A.dtype)
    if angle > angle_singular and angle < np.pi:
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


def T_SO3(psi: np.ndarray) -> np.ndarray:
    angle2 = psi @ psi
    angle = np.sqrt(angle2)
    if angle > angle_singular:
        # Park2005 (19), actually its the transposed!
        sa = np.sin(angle)
        ca = np.cos(angle)
        psi_tilde = ax2skew(psi)
        alpha = sa / angle
        beta2 = (1.0 - ca) / angle2
        return (
            eye3 - beta2 * psi_tilde + ((1.0 - alpha) / angle2) * ax2skew_squared(psi)
        )
    else:
        # first order approximation
        return eye3 - 0.5 * ax2skew(psi)


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
        psi_tilde2 = ax2skew_squared(psi)

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


def T_SO3_dot(psi: np.ndarray, psi_dot: np.ndarray) -> np.ndarray:
    """Derivative of tangent map w.r.t. scalar argument of rotation vector, see
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
            + c5 * (psi @ psi_dot) * eye3
            + c5 * np.outer(psi, psi_dot)
        ).T  #  transpose of Ibrahimbegović1995 (71)
    else:
        return np.zeros((3, 3), dtype=float)  # Cardona1988 after (46)


def T_SO3_inv(psi: np.ndarray) -> np.ndarray:
    angle2 = psi @ psi
    angle = np.sqrt(angle2)
    psi_tilde = ax2skew(psi)
    if angle > angle_singular:
        # Park2005 (19), actually its the transposed!
        gamma = 0.5 * angle / (np.tan(0.5 * angle))
        return eye3 + 0.5 * psi_tilde + ((1.0 - gamma) / angle2) * ax2skew_squared(psi)
    else:
        # first order approximation
        return eye3 + 0.5 * psi_tilde


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
        psi_tilde2 = ax2skew_squared(psi)
        cot = 1.0 / np.tan(0.5 * angle)
        gamma = 0.5 * angle * cot
        angle2 = angle * angle
        c = (1.0 - gamma) / angle2
        gamma_psi_k = gamma / angle2 * (1.0 - gamma) - 1.0 / 4.0
        c_psi_k = (-2.0 * c - gamma_psi_k) / angle2

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


def SE3(A_IB: np.ndarray, r_OP: np.ndarray) -> np.ndarray:
    H = np.zeros((4, 4), dtype=np.common_type(A_IB, r_OP))
    H[:3, :3] = A_IB
    H[:3, 3] = r_OP
    H[3, 3] = 1.0
    return H


def SE3inv(H: np.ndarray) -> np.ndarray:
    A_IB = H[:3, :3]
    r_OP = H[:3, 3]
    return SE3(A_IB.T, -A_IB.T @ r_OP)


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
    H_h[:3, 3, 3:] = np.einsum("l,lik->ik", r, T_SO3_psi(psi))
    H_h[:3, 3, :3] = T_SO3(psi).T
    return H_h


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


class A_IB_basic:
    """Basic rotations in Euclidean space."""

    def __init__(self, phi: float):
        self.phi = phi
        self.sp = np.sin(phi)
        self.cp = np.cos(phi)

    @property
    def x(self) -> np.ndarray:
        """Rotation around x-axis."""
        # fmt: off
        return np.array([[1,       0,        0],
                         [0, self.cp, -self.sp],
                         [0, self.sp,  self.cp]])
        # fmt: on

    @property
    def dx(self) -> np.ndarray:
        """Derivative of Rotation around x-axis."""
        # fmt: off
        return np.array([[0,        0,        0],
                         [0, -self.sp, -self.cp],
                         [0,  self.cp, -self.sp]])
        # fmt: on

    @property
    def y(self) -> np.ndarray:
        """Rotation around y-axis."""
        # fmt: off
        return np.array([[ self.cp, 0, self.sp],
                         [       0, 1,       0],
                         [-self.sp, 0, self.cp]])
        # fmt: on

    @property
    def dy(self) -> np.ndarray:
        """Derivative of Rotation around y-axis."""
        # fmt: off
        return np.array([[-self.sp, 0,  self.cp],
                         [       0, 0,        0],
                         [-self.cp, 0, -self.sp]])
        # fmt: on

    @property
    def z(self) -> np.ndarray:
        """Rotation around z-axis."""
        # fmt: off
        return np.array([[self.cp, -self.sp, 0],
                         [self.sp,  self.cp, 0],
                         [      0,        0, 1]])
        # fmt: on

    @property
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
    decision = np.zeros(4, dtype=float)
    decision[:3] = np.diag(R)
    decision[3] = np.trace(R)
    i = np.argmax(decision)

    quat = np.zeros(4, dtype=float)
    if i != 3:
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[i + 1] = np.sqrt(0.5 * R[i, i] + 0.25 * (1 - decision[3]))
        quat[0] = (R[k, j] - R[j, k]) / (4 * quat[i + 1])
        quat[j + 1] = (R[j, i] + R[i, j]) / (4 * quat[i + 1])
        quat[k + 1] = (R[k, i] + R[i, k]) / (4 * quat[i + 1])

    else:
        quat[0] = 0.5 * np.sqrt(1 + decision[3])
        quat[1] = (R[2, 1] - R[1, 2]) / (4 * quat[0])
        quat[2] = (R[0, 2] - R[2, 0]) / (4 * quat[0])
        quat[3] = (R[1, 0] - R[0, 1]) / (4 * quat[0])

    return quat


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
        angle = 2 * np.arctan2(q, q0)
        return angle * axis
    else:
        return np.zeros(3)


def smallest_rotation(
    J_a: np.ndarray, J_b: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Compute the transformation matrix A_JK in accordance with 
    {}_J a = A_JK {}_J b. This is sometimes referred to 'smallest rotation',
    see Crisield1996 Section 16.13. Both vectors are normalized if 
    normalize=True is used.

    This tranformation has a singularity for {}_J b = -{}_J a. This can be 
    overcome using a singular value decomposition that determines the rotation 
    axis, see Eigen3.

    References
    ----------
    Crisfield1996: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf \\
    Eigen3: https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Quaternion.h#L633-669
    """
    if normalize:
        J_a = J_a / norm(J_a)
        J_b = J_b / norm(J_b)

    cos_psi = J_a @ J_b
    denom = 1.0 + cos_psi

    # TODO: What is a good singular value her?
    # if denom > 0:
    if denom > 1e-6:
        e = cross3(J_a, J_b)
        return cos_psi * eye3 + ax2skew(e) + np.outer(e, e) / denom
    else:
        M = np.vstack((J_a, J_b))
        _, _, Vh = np.linalg.svd(M)
        axis = Vh[2]
        cos_psi = np.clip(cos_psi, -1, 1)
        psi = np.arccos(cos_psi)
        return Exp_SO3(psi * axis)


def Exp_SO3_quat(P, normalize=True):
    """Exponential mapping defined by (unit) quaternion, see 
    Egeland2002 (6.199) and Nuetzi2016 (3.31).

    References:
    -----------
    Egenland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165
    """
    p0, p = np.array_split(P, [1])
    p_tilde = ax2skew(p)
    if normalize:
        P2 = P @ P
        return eye3 + (2 / P2) * (p0 * p_tilde + p_tilde @ p_tilde)
    else:
        return eye3 + 2 * (p0 * p_tilde + p_tilde @ p_tilde)


def Exp_SO3_quat_p(P, normalize=True):
    """Derivative of Exp_SO3_quat with respect to P."""
    p0, p = np.array_split(P, [1])
    p_tilde = ax2skew(p)
    p_tilde_p = ax2skew_a()

    if normalize:
        P2 = P @ P
        A_P = np.einsum(
            "ij,k->ijk", p0 * p_tilde + p_tilde @ p_tilde, -(4 / (P2 * P2)) * P
        )
        s2 = 2 / P2
        A_P[:, :, 0] += s2 * p_tilde
        A_P[:, :, 1:] += (
            s2 * p0 * p_tilde_p
            + np.einsum("ijl,jk->ikl", p_tilde_p, s2 * p_tilde)
            + np.einsum("ij,jkl->ikl", s2 * p_tilde, p_tilde_p)
        )
    else:
        A_P = np.zeros((3, 3, 4), dtype=P.dtype)
        A_P[:, :, 0] = 2 * p_tilde
        A_P[:, :, 1:] = (
            2 * p0 * p_tilde_p
            + np.einsum("ijl,jk->ikl", p_tilde_p, 2 * p_tilde)
            + np.einsum("ij,jkl->ikl", 2 * p_tilde, p_tilde_p)
        )

    return A_P

    # from cardillo.math import approx_fprime
    # A_P_num = approx_fprime(P, Exp_SO3_quat, method="3-point", eps=1e-6)
    # diff = A_P - A_P_num
    # error = np.linalg.norm(diff)
    # # if error > 1e-7:
    # print(f"error Exp_SO3_quat_psi: {error}")
    # return A_P_num


Log_SO3_quat = Spurrier
# def Log_SO3_quat(A):
#     # from scipy.spatial.transform import Rotation
#     # return Rotation.from_matrix(A).as_quat()
#     psi = Log_SO3(A)
#     angle = norm(psi)
#     if angle > 0:
#         axis = psi / angle
#     else:
#         axis = np.array([1, 0, 0])
#     return axis_angle2quat(axis, angle)

# def Log_SO3_quat(A):
#     """Unit quaternion from rotation matrix, see scipy and Markley2012

#     References:
#     -----------
#     Markley2012: https://doi.org/10.2514/1.31730 \\
#     scipy: https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx#L848-L995
#     """
#     decision = np.zeros(4, dtype=float)
#     decision[:3] = np.diag(A)
#     decision[3] = np.trace(A)
#     choice = np.argmax(decision)

#     quat = np.zeros(4, dtype=float)
#     if choice != 3:
#         i = choice
#         j = (i + 1) % 3
#         k = (j + 1) % 3

#         quat[i] = 1 + 2 * A[i, i] - decision[3]
#         quat[j] = A[j, i] - A[i, j]
#         quat[k] = A[k, i] - A[i, k]
#         quat[3] = A[k, j] - A[j, k]
#     #     quat[i] = 0.5 * np.sqrt(1 + 2 * A[i, i] - decision[3])
#     #     quat[j] = 0.5 * (A[j, i] - A[i, j]) / (4 * quat[i])
#     #     quat[k] = 0.5 * (A[k, i] - A[i, k]) / (4 * quat[i])
#     #     quat[3] = 0.5 * (A[k, j] - A[j, k]) / (4 * quat[i])
#     else:
#         quat[0] = A[2, 1] - A[1, 2]
#         quat[1] = A[0, 2] - A[2, 0]
#         quat[2] = A[1, 0] - A[0, 1]
#         quat[3] = 1 + decision[3]
#         # quat[3] = 0.5 * np.sqrt(1 + decision[3])
#         # quat[0] = (A[2, 1] - A[1, 2]) / (4 * quat[3])
#         # quat[1] = (A[0, 2] - A[2, 0]) / (4 * quat[3])
#         # quat[2] = (A[1, 0] - A[0, 1]) / (4 * quat[3])

#     return quat / norm(quat)


def T_SO3_quat(P, normalize=True):
    """Tangent map for unit quaternion. See Egeland2002 (6.327).

    References:
    -----------
    Egenland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
    """
    p0, p = np.array_split(P, [1])
    if normalize:
        return (2 / (P @ P)) * np.hstack((-p[:, None], p0 * eye3 - ax2skew(p)))
    else:
        return 2 * np.hstack((-p[:, None], p0 * eye3 - ax2skew(p)))


def T_SO3_inv_quat(P, normalize=True):
    """Inverse tangent map for unit quaternion. See Egeland2002 (6.329) and
    (6.330) as well as Nuetzi2016 (3.11), (3.12) and (4.19).

    References:
    -----------
    Egenland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165
    """
    p0, p = np.array_split(P, [1])
    if normalize:
        return (0.5 / (P @ P)) * np.vstack((-p.T, p0 * eye3 + ax2skew(p)))
    else:
        return 0.5 * np.vstack((-p.T, p0 * eye3 + ax2skew(p)))


def T_SO3_quat_P(P, normalize=True):
    if normalize:
        p0, p = np.array_split(P, [1])
        P2 = P @ P
        T_P = np.einsum(
            "ij,k->ijk",
            np.hstack((-p[:, None], p0 * eye3 - ax2skew(p))),
            -4 * P / (P2 * P2),
        )
        P22 = 2 / P2
        T_P[:, 0, 1:] -= P22 * eye3
        T_P[:, 1:, 0] += P22 * eye3
        T_P[:, 1:, 1:] -= P22 * ax2skew_a()
    else:
        T_P = np.zeros((3, 4, 4), dtype=float)
        T_P[:, 0, 1:] -= 2 * eye3
        T_P[:, 1:, 0] += 2 * eye3
        T_P[:, 1:, 1:] -= 2 * ax2skew_a()

    return T_P

    # from cardillo.math import approx_fprime

    # T_P_num = approx_fprime(P, T_SO3_quat, method="3-point", eps=1e-6)
    # diff = T_P - T_P_num
    # error = np.linalg.norm(diff)
    # print(f"error T_P: {error}")
    # return T_P_num


def T_SO3_inv_quat_P(P, normalize=True):
    if normalize:
        p0, p = np.array_split(P, [1])
        s = P @ P
        T_inv_P = np.einsum(
            "ij,k->ijk",
            np.vstack((-p.T, p0 * eye3 + ax2skew(p))),
            -P / (s * s),
        )
        s2 = 0.5 / s
        T_inv_P[0, :, 1:] -= s2 * eye3
        T_inv_P[1:, :, 0] += s2 * eye3
        T_inv_P[1:, :, 1:] += s2 * ax2skew_a()
    else:
        T_inv_P = np.zeros((4, 3, 4), dtype=float)
        T_inv_P[0, :, 1:] = -0.5 * eye3
        T_inv_P[1:, :, 0] = 0.5 * eye3
        T_inv_P[1:, :, 1:] = 0.5 * ax2skew_a()
    return T_inv_P

    # from cardillo.math import approx_fprime
    # T_inv_P_num = approx_fprime(P, T_SO3_inv_quat, method="3-point", eps=1e-6)
    # diff = T_inv_P - T_inv_P_num
    # error = np.linalg.norm(diff)
    # print(f"error T_inv_P: {error}")
    # return T_inv_P_num


def quatprod(P, Q):
    """Quaternion product, see Egeland2002 (6.190).

    References:
    -----------
    Egenland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
    """
    p0, p = np.array_split(P, [1])
    q0, q = np.array_split(Q, [1])
    z0 = p0 * q0 - p @ q
    z = p0 * q + q0 * p + cross3(p, q)
    return np.array([z0, *z])


def axis_angle2quat(axis, angle):
    n = axis / norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * n])
