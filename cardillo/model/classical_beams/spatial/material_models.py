import numpy as np
from cardillo.math import e1, e2, e3, norm


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

    def potential(self, Gamma_i, Gamma0_i, K_i, K0_i):
        dG = Gamma_i - Gamma0_i
        dK = K_i - K0_i
        return 0.5 * dG @ self.C_n @ dG + 0.5 * dK @ self.C_m @ dK

    def n_i(self, Gamma_i, Gamma0_i, K_i, K0_i):
        dG = Gamma_i - Gamma0_i
        return self.C_n @ dG

    def m_i(self, Gamma_i, Gamma0_i, K_i, K0_i):
        dK = K_i - K0_i
        return self.C_m @ dK

    def n_i_Gamma_j(self, Gamma_i, Gamma0_i, K_i, K0_i):
        return self.C_n

    def n_i_K_j(self, Gamma_i, Gamma0_i, K_i, K0_i):
        return np.zeros((3, 3))

    def m_i_Gamma_j(self, Gamma_i, Gamma0_i, K_i, K0_i):
        return np.zeros((3, 3))

    def m_i_K_j(self, Gamma_i, Gamma0_i, K_i, K0_i):
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

    def update_state(self, xi, t, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        old_state = (xi == self.xi) and (t == self.t)
        if not old_state:
            # update already computed state
            self.xi = xi
            self.t = t

            # strain measures
            nga = norm(Gamma_i)
            nga0 = norm(Gamma0_i)
            lambda_ = nga / nga0
            lambda_Gamma_i = Gamma_i / (nga0 * nga)
            Gamma_i_Gamma_j = np.outer(lambda_Gamma_i, lambda_Gamma_i)
            lambda_Gamma_i_Gamma_j = (np.eye(3) - Gamma_i_Gamma_j / nga**2) / (
                nga * nga0
            )
            dga = Gamma_i - Gamma0_i
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

            self.n_m = W1_lambda * lambda_Gamma_i + self.Ei @ dga
            self.n_m_Gamma_j = (
                W1_lambda2 * Gamma_i_Gamma_j
                + W1_lambda * lambda_Gamma_i_Gamma_j
                + self.Ei
            )

            # material contact couples
            self.m_m = self.Fi @ dka
            self.m_m_Kappa_j = self.Fi

    def potential(self, xi, t, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        self.update_state(xi, t, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
        return self.U

    def n_i(self, xi, t, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        self.update_state(xi, t, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
        return self.n_m

    def m_i(self, xi, t, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        self.update_state(xi, t, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
        return self.m_m

    def n_i_Gamma_i_j(self, xi, t, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        self.update_state(xi, t, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
        return self.n_m_Gamma_j

    def n_i_Kappa_i_j(self, xi, t, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        return np.zeros((3, 3))

    def m_i_Gamma_i_j(self, xi, t, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        return np.zeros((3, 3))

    def m_i_Kappa_i_j(self, xi, t, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        self.update_state(xi, t, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
        return self.m_m_Kappa_j
