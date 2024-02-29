from abc import ABC, abstractmethod
import numpy as np

# from cardillo.math import norm, approx_fprime
from cardillo.math.algebra import norm


class RodMaterialModel(ABC):
    """Abstract class for rod material models"""

    @abstractmethod
    def B_n(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        """Contact force in cross-section-fixed K-basis."""
        ...

    @abstractmethod
    def B_m(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        """Contact moment in cross-section-fixed K-basis."""
        ...

    @abstractmethod
    def B_n_B_Gamma(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        """Derivative of contact force w.r.t. strain B_Gamma."""
        ...

    @abstractmethod
    def B_n_B_Kappa(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        """Derivative of contact force w.r.t. strain B_Kappa."""
        ...

    @abstractmethod
    def B_m_B_Gamma(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        """Derivative of contact couple w.r.t. strain B_Gamma."""
        ...

    @abstractmethod
    def B_m_B_Kappa(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        """Derivative of contact couple w.r.t. strain B_Kappa."""
        ...


class Simo1986(RodMaterialModel):
    def __init__(self, Ei, Fi):
        """
        Material model for shear deformable rod with quadratic strain energy
        function found in Simo1986 (2.8), (2.9) and (2.10).

        Parameters
        ----------
        Ei : np.ndarray (3,)
            E0: dilatational stiffness, i.e., rigidity with resepct to volumetric change.
            E1: shear stiffness in e_y^K-direction.
            E2: shear stiffness in e_z^K-direction.
        Fi : np.ndarray (3,)
            F0: torsional stiffness
            F1: flexural stiffness around e_y^K-direction.
            F2: flexural stiffness around e_z^K-direction.

        References
        ----------
        Simo1986 : https://doi.org/10.1016/0045-7825(86)90079-4
        """

        self.Ei = Ei
        self.Fi = Fi

        self.C_n = np.diag(self.Ei)
        self.C_m = np.diag(self.Fi)

        self.C_n_inv = np.linalg.inv(self.C_n)
        self.C_m_inv = np.linalg.inv(self.C_m)

    def potential(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        dG = B_Gamma - B_Gamma0
        dK = B_Kappa - B_Kappa0
        return 0.5 * dG @ self.C_n @ dG + 0.5 * dK @ self.C_m @ dK

    def complementary_potential(self, B_n, B_m):
        return 0.5 * B_n @ self.C_n_inv @ B_n + 0.5 * B_m @ self.C_m_inv @ B_m

    def B_n(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        dG = B_Gamma - B_Gamma0
        return self.C_n @ dG

    def B_m(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        dK = B_Kappa - B_Kappa0
        return self.C_m @ dK

    def B_n_B_Gamma(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return self.C_n

    def B_n_B_Kappa(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return np.zeros((3, 3), dtype=float)

    def B_m_B_Gamma(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return np.zeros((3, 3), dtype=float)

    def B_m_B_Kappa(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return self.C_m


class Harsch2021(RodMaterialModel):
    def __init__(self, Ei, Fi):
        """
        Material model for shear deformable beam with nonlinear axial deformation.

        Parameters
        ----------
        Ei : np.ndarray(3)
            E0: extensional stiffness, i.e., stiffness in r_OP'-direction.
            E1: shear stiffness in e_y^K-direction.
            E2: shear stiffness in e_z^K-direction.
        Fi : np.ndarray(3)
            F0: torsional stiffness
            F1: flexural stiffness around e_y^K-direction.
            F2: flexural stiffness around e_z^K-direction.

        References
        ----------
        Harsch2021: https://doi.org/10.1177/10812865211000790
        """
        self.Ei = Ei
        self.Fi = Fi

        self.C_n = np.diag([0, self.Ei[1], self.Ei[2]])
        self.C_m = np.diag(self.Fi)

    def potential(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        dG = B_Gamma - B_Gamma0
        lambda_ = norm(B_Gamma)
        lambda0_ = norm(B_Gamma0)
        dK = B_Kappa - B_Kappa0
        return (
            0.5 * dG @ self.C_n @ dG
            + 0.5 * self.Ei[0] * (lambda_ - lambda0_) ** 2
            + 0.5 * dK @ self.C_m @ dK
        )

    def B_n(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        dG = B_Gamma - B_Gamma0
        lambda_ = norm(B_Gamma)
        lambda0_ = norm(B_Gamma0)
        return self.C_n @ dG + self.Ei[0] * (1 - lambda0_ / lambda_) * B_Gamma

    def B_m(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        dK = B_Kappa - B_Kappa0
        return self.C_m @ dK

    def B_n_B_Gamma(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        lambda_ = norm(B_Gamma)
        lambda0_ = norm(B_Gamma0)
        return self.C_n + self.Ei[0] * (
            (1 - lambda0_ / lambda_) * np.eye(3)
            + np.outer(B_Gamma, B_Gamma) / lambda_**3
        )

    def B_n_B_Kappa(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return np.zeros((3, 3), dtype=float)

    def B_m_B_Gamma(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return np.zeros((3, 3), dtype=float)

    def B_m_B_Kappa(self, B_Gamma, B_Gamma0, B_Kappa, B_Kappa0):
        return self.C_m
