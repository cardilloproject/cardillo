import numpy as np
from cardillo.math import norm, approx_fprime


class ShearStiffQuadratic:
    """
    Material model for shear stiff beam with quadratic strain energy
    function found in Simo1986 (2.8), (2.9) and (2.10).

    References
    ----------
    Simo1986: https://doi.org/10.1016/0045-7825(86)90079-4
    """

    def __init__(self, E1, Fi):
        self.E1 = E1  # axial stiffness E1
        self.Fi = Fi  # torsional stiffness F1 and both flexural stiffnesses F2
        # and F3
        self.C_m = np.diag(self.Fi)

    def potential(self, lambda_, K_Kappa, K_Kappa0):
        dK = K_Kappa - K_Kappa0
        return 0.5 * self.E1 * (lambda_ - 1.0) ** 2 + 0.5 * dK @ self.C_m @ dK

    def n(self, lambda_, K_Kappa, K_Kappa0):
        return self.E1 * (lambda_ - 1)

    def K_m(self, lambda_, K_Kappa, K_Kappa0):
        dK = K_Kappa - K_Kappa0
        return self.C_m @ dK

    def n_lambda(self, lambda_, K_Kappa, K_Kappa0):
        return self.E1

    def n_K_Kappa(self, lambda_, K_Kappa, K_Kappa0):
        return np.zeros(3)

    def K_m_lambda(self, lambda_, K_Kappa, K_Kappa0):
        return np.zeros(3)

    def K_m_K_Kappa(self, lambda_, K_Kappa, K_Kappa0):
        return self.C_m


class Harsch2021:
    """
    Material model for shear deformable beam with nonlinear axial deformation.

    References
    ----------
    Harsch2021: https://doi.org/10.1177/10812865211000790
    """

    def __init__(self, Ei, Fi):
        self.Ei = Ei  # axial stiffness E1 and both shear stiffnesses E2 and E3
        self.Fi = Fi  # torsional stiffness F1 and both flexural stiffnesses F2
        # and F3

        self.C_n = np.diag([0, self.Ei[1], self.Ei[2]])
        self.C_m = np.diag(self.Fi)

    def potential(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        dG = K_Gamma - K_Gamma0
        lambda_ = norm(K_Gamma)
        lambda0_ = norm(K_Gamma0)
        dK = K_Kappa - K_Kappa0
        return (
            0.5 * dG @ self.C_n @ dG
            + 0.5 * self.Ei[0] * (lambda_ - lambda0_) ** 2
            + 0.5 * dK @ self.C_m @ dK
        )

    def K_n(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        dG = K_Gamma - K_Gamma0
        lambda_ = norm(K_Gamma)
        lambda0_ = norm(K_Gamma0)
        return self.C_n @ dG + self.Ei[0] * (1 - lambda0_ / lambda_) * K_Gamma

    def K_m(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        dK = K_Kappa - K_Kappa0
        return self.C_m @ dK

    def K_n_K_Gamma(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        lambda_ = norm(K_Gamma)
        lambda0_ = norm(K_Gamma0)
        return self.C_n + self.Ei[0] * (
            (1 - lambda0_ / lambda_) * np.eye(3)
            + np.outer(K_Gamma, K_Gamma) / lambda_**3
        )

    def K_n_K_Kappa(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        return np.zeros((3, 3), dtype=float)

    def K_m_K_Gamma(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        return np.zeros((3, 3), dtype=float)

    def K_m_K_Kappa(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        return self.C_m


class Simo1986:
    """
    Material model for shear deformable beam with quadratic strain energy
    function found in Simo1986 (2.8), (2.9) and (2.10).

    References
    ----------
    Simo1986: https://doi.org/10.1016/0045-7825(86)90079-4
    """

    def __init__(self, Ei, Fi):
        self.Ei = Ei  # axial stiffness E1 and both shear stiffnesses E2 and E3
        self.Fi = Fi  # torsional stiffness F1 and both flexural stiffnesses F2
        # and F3

        self.C_n = np.diag(self.Ei)
        self.C_m = np.diag(self.Fi)

        self.C_n_inv = np.linalg.inv(self.C_n)
        self.C_m_inv = np.linalg.inv(self.C_m)

    def C_n_inverse(self):
        return self.C_n_inv

    def C_m_inverse(self):
        return self.C_m_inv

    def potential(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        dG = K_Gamma - K_Gamma0
        dK = K_Kappa - K_Kappa0
        return 0.5 * dG @ self.C_n @ dG + 0.5 * dK @ self.C_m @ dK

    def complementary_potential(self, K_n_S, K_m_S):
        return 0.5 * K_n_S @ self.C_n_inv @ K_n_S + 0.5 * K_m_S @ self.C_m_inv @ K_m_S

    def K_gam_comp(self, K_n_S, K_Gamma0):
        # return self.C_n_inv @ K_n_S + np.array([1, 0, 0])
        return self.C_n_inv @ K_n_S + K_Gamma0

    def K_kap_comp(self, K_m_S, K_Kappa0):
        return self.C_m_inv @ K_m_S + K_Kappa0

    def K_n(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        dG = K_Gamma - K_Gamma0
        return self.C_n @ dG

    def K_m(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        dK = K_Kappa - K_Kappa0
        return self.C_m @ dK

    def K_n_K_Gamma(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        return self.C_n

    def K_n_K_Kappa(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        return np.zeros((3, 3), dtype=float)

    def K_m_K_Gamma(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        return np.zeros((3, 3), dtype=float)

    def K_m_K_Kappa(self, K_Gamma, K_Gamma0, K_Kappa, K_Kappa0):
        return self.C_m


class Ogden:
    """
    Nonlinear Ogden material law for the normal force which takes 
    into account stiffening behavior of the material during compression
    and extension, see SoroDatabase2021 and Wikipedia2021. It can be derived
    from Ogden1997 (4.3.66) by assuming only one pair of (mu, alpha) and the 
    case of an incompressible material under uniaxial extension with 
    axisymmetric contraction Ogden1997 (2.2.47).

    References
    ----------
    Ogden1997: https://store.doverpublications.com/0486696480.html \\
    SoroDatabase2021: https://soro-materials-database.herokuapp.com/ \\
    Wikipedia2021: https://en.wikipedia.org/wiki/Ogden_(hyperelastic_model)#Uniaxial_tension
    """

    def __init__(self, E, G, A, I2, I3, I23, alpha):
        # make Young's modulus a callable of the material point xi
        if not callable(E):
            self.E = lambda xi: E
        else:
            self.E = E

        # make shear modulus a callable of the material point xi
        if not callable(G):
            self.G = lambda xi: G
        else:
            self.G = G

        # make area (inclusive pressure chambers) a callable of xi the
        # material point
        if not callable(A):
            self.A = lambda xi: A
        else:
            self.A = A

        # make second moment of area w.r.t. d2-axis (inclusive pressure
        # chambers) a callable of the material point xi
        if not callable(I2):
            self.I2 = lambda xi: I2
        else:
            self.I2 = I2

        # make second moment of area w.r.t. d3-axis (inclusive pressure
        # chambers) a callable of the material point xi
        if not callable(I3):
            self.I3 = lambda xi: I3
        else:
            self.I3 = I3

        # make biaxial second moment of area (inclusive pressure # chambers)
        # a callable of the material point xi
        if not callable(I23):
            self.I23 = lambda xi: I23
        else:
            self.I23 = I23

        # make Ogden exponent a callable of the material point xi
        if not callable(alpha):
            self.alpha = lambda xi: alpha
        else:
            self.alpha = alpha

        # store material point and time in order to save evaluations
        self.xi = None
        self.t = None
        self.U = 0
        self.Ei = np.zeros(3)
        self.Fi = np.zeros(3)
        self.n_m = np.zeros(3)
        self.m_m = np.zeros(3)

    def update_state(self, xi, t, K_Gamma, K_Gamma0, Kappa_i, Kappa0_i):
        old_state = (xi == self.xi) and (t == self.t)
        if not old_state:
            # update already computed state
            self.xi = xi
            self.t = t

            # strain measures
            nga = norm(K_Gamma)
            nga0 = norm(K_Gamma0)
            lambda_ = nga / nga0
            lambda_K_Gamma = K_Gamma / (nga0 * nga)
            K_Gamma_Gamma_j = np.outer(lambda_K_Gamma, lambda_K_Gamma)
            lambda_K_Gamma_Gamma_j = (np.eye(3) - K_Gamma_Gamma_j / nga**2) / (
                nga * nga0
            )
            dga = K_Gamma - K_Gamma0
            dka = Kappa_i - Kappa0_i

            # evaluate functions at material point xi
            E = self.E(xi)
            G = self.G(xi)
            A = self.A(xi)
            I2 = self.I2(xi)
            I3 = self.I3(xi)
            I23 = self.I23(xi)
            alpha = self.alpha(xi)

            # polar moment of area
            J = I2 + I3

            self.Ei = np.diag([0, G * A, G * A])
            self.Fi = np.array(
                [[G * J, 0, 0], [0, E * I2, E * I23], [0, E * I23, E * I3]]
            )

            # potential energy
            mu = (2.0 / 3.0) * E * A / alpha
            self.U = (
                mu / alpha * (lambda_**alpha + 2.0 / (lambda_ ** (0.5 * alpha)) - 3)
                + 0.5 * dga @ self.Ei @ dga
                + 0.5 * dka @ self.Fi @ dka
            )

            # material contact forces
            W1_lambda = mu * (
                lambda_ ** (alpha - 1) - 1 / (lambda_ ** (0.5 * alpha + 1))
            )
            W1_lambda2 = mu * (
                (alpha - 1) * lambda_ ** (alpha - 2)
                + (0.5 * alpha + 1) / (lambda_ ** (0.5 * alpha + 2))
            )

            self.n_m = W1_lambda * lambda_K_Gamma + self.Ei @ dga
            self.n_m_Gamma_j = (
                W1_lambda2 * K_Gamma_Gamma_j
                + W1_lambda * lambda_K_Gamma_Gamma_j
                + self.Ei
            )

            # material contact couples
            self.m_m = self.Fi @ dka
            self.m_m_Kappa_j = self.Fi

    def potential(self, xi, t, K_Gamma, K_Gamma0, Kappa_i, Kappa0_i):
        self.update_state(xi, t, K_Gamma, K_Gamma0, Kappa_i, Kappa0_i)
        return self.U

    def K_n(self, xi, t, K_Gamma, K_Gamma0, Kappa_i, Kappa0_i):
        self.update_state(xi, t, K_Gamma, K_Gamma0, Kappa_i, Kappa0_i)
        return self.n_m

    def K_m(self, xi, t, K_Gamma, K_Gamma0, Kappa_i, Kappa0_i):
        self.update_state(xi, t, K_Gamma, K_Gamma0, Kappa_i, Kappa0_i)
        return self.m_m

    def K_n_K_Gamma(self, xi, t, K_Gamma, K_Gamma0, Kappa_i, Kappa0_i):
        self.update_state(xi, t, K_Gamma, K_Gamma0, Kappa_i, Kappa0_i)
        return self.n_m_Gamma_j

    def K_n_K_Kappa(self, xi, t, K_Gamma, K_Gamma0, Kappa_i, Kappa0_i):
        return np.zeros((3, 3))

    def K_m_K_Gamma(self, xi, t, K_Gamma, K_Gamma0, Kappa_i, Kappa0_i):
        return np.zeros((3, 3))

    def K_m_K_Kappa(self, xi, t, K_Gamma, K_Gamma0, Kappa_i, Kappa0_i):
        self.update_state(xi, t, K_Gamma, K_Gamma0, Kappa_i, Kappa0_i)
        return self.m_m_Kappa_j
