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


# TODO: Implement q_dot, B, q_dot_q, q_ddot for angular velocity in inertial frame
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


# TODO: Implement q_dot, B, q_dot_q, q_ddot for angular velocity in inertial frame
class QuaternionRotationParameterization(RotationParameterizationBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def dim():
        return 4

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
        psi = psi / norm(psi)
        return T_SO3_inv_quat(psi)

    @staticmethod
    def q_ddot(psi, K_omega_IK, K_omega_IK_dot):
        raise NotImplementedError

    @staticmethod
    def step_callback(psi):
        return psi / norm(psi)
