import numpy as np
from abc import ABC, abstractmethod

from cardillo.math import (
    pi,
    norm,
    Exp_SO3,
    Log_SO3,
    Exp_SO3_psi,
    T_SO3_inv,
    T_SO3_inv_psi,
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    Log_SO3_quat,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
)


# TODO: Implement q_dot, B, q_dot_q, q_ddot for angular velocity in inertial frame
class RotationParameterizationBase(ABC):
    @staticmethod
    @abstractmethod
    def dim():
        ...

    @staticmethod
    @abstractmethod
    def Exp_SO3(psi):
        ...

    @staticmethod
    @abstractmethod
    def Exp_SO3_psi(psi):
        ...

    @staticmethod
    @abstractmethod
    def Log_SO3(A_IK):
        ...

    @staticmethod
    @abstractmethod
    def q_dot(psi, K_omega_IK):
        ...

    @staticmethod
    @abstractmethod
    def q_dot_q(psi, K_omega_IK):
        ...

    @staticmethod
    @abstractmethod
    def B(psi):
        ...

    @staticmethod
    @abstractmethod
    def q_ddot(psi, K_omega_IK, K_omega_IK_dot):
        ...

    @staticmethod
    @abstractmethod
    def step_callback(psi):
        ...


class AxisAngleRotationParameterization(RotationParameterizationBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def dim():
        return 3

    @staticmethod
    def Exp_SO3(psi):
        return Exp_SO3(psi)

    @staticmethod
    def Exp_SO3_psi(psi):
        return Exp_SO3_psi(psi)

    @staticmethod
    def Log_SO3(A_IK):
        return Log_SO3(A_IK)

    @staticmethod
    def q_dot(psi, K_omega_IK):
        return T_SO3_inv(psi) @ K_omega_IK

    @staticmethod
    def q_dot_q(psi, K_omega_IK):
        return np.einsum(
            "ijk,j->ik",
            T_SO3_inv_psi(psi),
            K_omega_IK,
        )

    @staticmethod
    def B(psi):
        return T_SO3_inv(psi)

    @staticmethod
    def q_ddot(psi, K_omega_IK, K_omega_IK_dot):
        raise NotImplementedError

    @staticmethod
    def step_callback(psi):
        """Change between rotation vector and its complement in order to
        circumvent singularities of the rotation vector."""
        # return psi
        angle = norm(psi)
        if angle < pi:
            return psi
        else:
            # Ibrahimbegovic1995 after (62)
            # n = int((angle + np.pi) / (2.0 * np.pi))
            # psi_C = (1.0 - 2.0 * n * pi / angle) * psi
            psi_C = (1.0 - 2.0 * pi / angle) * psi
            print(f"complement rotation vector chosen")
            print(f"psi: {psi}")
            print(f"psi_C: {psi_C}")
            return psi_C


class QuaternionRotationParameterization(RotationParameterizationBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def dim():
        return 4

    @staticmethod
    def dim_g_S():
        return 1

    @staticmethod
    def Exp_SO3(psi):
        return Exp_SO3_quat(psi)

    @staticmethod
    def Exp_SO3_psi(psi):
        return Exp_SO3_quat_p(psi)

    @staticmethod
    def Log_SO3(A_IK):
        return Log_SO3_quat(A_IK)

    @staticmethod
    def q_dot(psi, K_omega_IK):
        return T_SO3_inv_quat(psi) @ K_omega_IK

    @staticmethod
    def q_dot_q(psi, K_omega_IK):
        return np.einsum(
            "ijk,j->ik",
            T_SO3_inv_quat_P(psi),
            K_omega_IK,
        )

    @staticmethod
    def B(psi):
        psi = psi / norm(psi)
        return T_SO3_inv_quat(psi)

    @staticmethod
    def q_ddot(psi, K_omega_IK, K_omega_IK_dot):
        raise NotImplementedError

    @staticmethod
    def step_callback(psi):
        return psi / norm(psi)

    @staticmethod
    def g_S(psi):
        return np.array([psi @ psi - 1])

    @staticmethod
    def g_S_q(psi):
        return 2 * psi.reshape(1, -1)

    @staticmethod
    def g_S_q_T_mu_q(psi, mu):
        return 2 * mu * np.eye(4)


class R9RotationParameterization(RotationParameterizationBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def dim():
        return 9

    @staticmethod
    def dim_g_S():
        return 6

    @staticmethod
    def Exp_SO3(psi):
        return psi.reshape(3, 3, order="F")

        # # orthonormalization, see https://de.wikipedia.org/wiki/Gram-Schmidtsches_Orthogonalisierungsverfahren#Algorithmus_des_Orthonormalisierungsverfahrens
        # w1, w2, w3 = psi.reshape(3, 3, order="F").T
        # ex = w1 / norm(w1)
        # v2 = w2 - (w2 @ ex) * ex
        # ey = v2 / norm(v2)
        # v3 = w3 - (w3 @ ex) * ex - (w3 @ ey) * ey
        # ez = v3 / norm(v3)

        # return np.vstack((ex, ey, ez)).T

    @staticmethod
    def Exp_SO3_psi(psi):
        A_IK_psi = np.eye(9, 9, dtype=float).reshape(3, 3, -1, order="F")
        return A_IK_psi

        # from cardillo.math import approx_fprime
        # return approx_fprime(psi, R9RotationParameterization.Exp_SO3, method="3-point", eps=1e-6)

    @staticmethod
    def Log_SO3(A_IK):
        return A_IK.reshape(-1, order="F")

    @staticmethod
    def q_dot(psi, K_omega_IK):
        raise NotImplementedError
        psi = psi / norm(psi)
        return T_SO3_inv_quat(psi) @ K_omega_IK

    @staticmethod
    def q_dot_q(psi, K_omega_IK):
        raise NotImplementedError
        return np.einsum(
            "ijk,j->ik",
            T_SO3_inv_quat_P(psi),
            K_omega_IK,
        )

    @staticmethod
    def B(psi):
        raise NotImplementedError
        psi = psi / norm(psi)
        return T_SO3_inv_quat(psi)

    @staticmethod
    def q_ddot(psi, K_omega_IK, K_omega_IK_dot):
        raise NotImplementedError

    @staticmethod
    def step_callback(psi):
        return psi

    @staticmethod
    def g_S(psi):
        A_IK = R9RotationParameterization.Exp_SO3(psi)
        ex, ey, ez = A_IK.T
        g_S = np.zeros(6, dtype=psi.dtype)
        g_S[0] = ex @ ex - 1
        g_S[1] = ey @ ey - 1
        g_S[2] = ez @ ez - 1
        g_S[3] = ex @ ey
        g_S[4] = ey @ ez
        g_S[5] = ez @ ex
        return g_S

    @staticmethod
    def g_S_q(psi):
        A_IK = R9RotationParameterization.Exp_SO3(psi)
        ex, ey, ez = A_IK.T

        g_S_q = np.zeros((6, 9), dtype=psi.dtype)
        g_S_q[0, :3] = 2 * ex
        g_S_q[1, 3:6] = 2 * ey
        g_S_q[2, 6:] = 2 * ez

        g_S_q[3, :3] = ey
        g_S_q[3, 3:6] = ex

        g_S_q[4, 3:6] = ez
        g_S_q[4, 6:] = ey

        g_S_q[5, :3] = ez
        g_S_q[5, 6:] = ex

        return g_S_q

    @staticmethod
    def g_S_q_T_mu_q(psi, mu):
        raise NotImplementedError
