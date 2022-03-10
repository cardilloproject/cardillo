import numpy as np
from cardillo.math.algebra import norm3


class Hooke(object):
    r"""Simple Hooke material model which accounts for compression effects in simple shear experiments.
    
    The potential function is given by

    .. math::

        \psi(\vGa, \vka) = \frac{1}{2} E_1 \left(\frac{\|\vGa\|}{\|\vGa_0\|} - 1\right)^2 + \frac{1}{2} E_\alpha \left(\Gamma_\alpha - \Gamma_\alpha^0\right)^2 + \frac{1}{2} F_i \left(\Kappa_i_i - \Kappa_i_i^0\right)^2

    The contact forces $n_i$ and couples $m_i$ are computes as

    .. math::

        n_i(\vGa, \vka) &= E_1 \left(\frac{1}{\|\vGa_0\|} - \frac{1}{\|\vGa\|}\right) \frac{\Gamma_i}{\|\vGa_0\|} + \sum_{\alpha=2}^{3} \delta_{i\alpha} E_\alpha \left(\Gamma_\alpha - \Gamma_\alpha^0\right) \; , \\
        m_i(\vGa, \vka) &= F_i \left(\Kappa_i_i - \Kappa_i_i^0\right) \; ,

    together with their stiffnesses

    .. math::

        \pd{n_i}{\Gamma_j}(\vGa, \vka) &= \frac{E_1}{\|\vGa_0\|} \left[ \left(\frac{1}{\|\vGa_0\|} - \frac{1}{\|\vGa\|} \right) \delta_{ij} + \frac{\Gamma_i \Gamma_j}{\|\vGa\|^3}\right] + \sum_{\alpha=2}^{3} \delta_{i\alpha} E_\alpha \delta_{\alpha j} \; , \\
        \pd{n_i}{\Kappa_i_j} &= \mathbf{0} \; , \quad \pd{m_i}{\Gamma_j} = \mathbf{0} \; , \quad \pd{m_i}{\Kappa_i_j} = F_i \delta_{ij} \ve_i \otimes \ve_j \; ,

    Parameters
    ----------
    Ei : :class:`numpy.ndarray` with shape=(3,)
        Axial stiffness $E_1$ and shear stiffnesses $E_2, E_3$. 
    Fi : :class:`numpy.ndarray` with shape=(3,)
        Torsional stiffness $F_1$ and flexural stiffnesses $F_2, F_3$.  
    """

    def __init__(self, Ei, Fi):
        self.Ei = Ei  # axial stiffness E1 and shear stiffnesses E2 and E3
        self.Fi = Fi  # torsional stiffness F1 and both flexural stiffnesses F2 and F3

    def potential(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the strain energy density `\psi(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        potential : float
            Returns the scalar potential for the given material law.

        """
        nga = norm3(Gamma_i)
        nga0 = norm3(Gamma0_i)
        dla = nga / nga0 - 1.0
        dga1 = Gamma_i[1] - Gamma0_i[1]
        dga2 = Gamma_i[2] - Gamma0_i[2]
        dka = Kappa_i - Kappa0_i

        return (
            0.5 * self.Ei[0] * dla * dla
            + 0.5 * self.Ei[1] * dga1 * dga1
            + 0.5 * self.Ei[2] * dga2 * dga2
            + 0.5 * self.Fi @ (dka * dka)
        )

    def n_i(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the contact forces `n_i(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        n : :class:`numpy.ndarray` with shape=(3,)
            Returns the vector valued contact forces for the given material law.

        """
        nga = norm3(Gamma_i)
        nga0 = norm3(Gamma0_i)
        dga1 = Gamma_i[1] - Gamma0_i[1]
        dga2 = Gamma_i[2] - Gamma0_i[2]

        return (
            self.Ei[0] * (1 / nga0 - 1 / nga) * Gamma_i / nga0
            + self.Ei[1] * dga1 * np.array([0, 1, 0])
            + self.Ei[2] * dga2 * np.array([0, 0, 1])
        )

    def m_i(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the contact couples `m_i(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        m : :class:`numpy.ndarray` with shape=(3,)
            Returns the vector valued contact couples for the given material law.

        """
        dka = Kappa_i - Kappa0_i

        return self.Fi * dka

    def n_i_Gamma_i_j(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the stiffness of the contact forces `\pd{n_i}{\Gamma_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dn_dGamma_i : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact forces for the given material law.

        """
        nga = norm3(Gamma_i)
        nga0 = norm3(Gamma0_i)

        dn_dGamma_i1 = (
            self.Ei[0] / (nga * nga * nga * nga0) * np.outer(Gamma_i, Gamma_i)
        )

        dn_dGamma_i2 = self.Ei[0] * (1 / nga0 - 1 / nga) / nga0 * np.diag(np.ones(3))

        dn_dGamma_i3 = np.diag(self.Ei)
        dn_dGamma_i3[0, 0] = 0

        return dn_dGamma_i1 + dn_dGamma_i2 + dn_dGamma_i3

    def n_i_Kappa_i_j(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the stiffness of the contact forces `\pd{n_i}{\Kappa_i_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dn_dKappa_i : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact forces for the given material law.

        """
        return np.zeros((3, 3))

    def m_i_Gamma_i_j(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the stiffness of the contact couples `\pd{m_i}{\Gamma_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dm_dGamma_i : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact couples for the given material law.

        """
        return np.zeros((3, 3))

    def m_i_Kappa_i_j(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the stiffness of the contact couples `\pd{m_i}{\Kappa_i_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dm_dKappa_i : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact couples for the given material law.

        """
        return np.diag(self.Fi)


class Hooke_quadratic(object):
    r"""Simple Hooke material model with quadratic energy function as found in Simo1985, (4.13) and (4.14).

    Parameters
    ----------
    Ei : :class:`numpy.ndarray` with shape=(3,)
        Axial stiffness $E_1$ and shear stiffnesses $E_2, E_3$.
    Fi : :class:`numpy.ndarray` with shape=(3,)
        Torsional stiffness $F_1$ and flexural stiffnesses $F_2, F_3$.
    """

    def __init__(self, Ei, Fi):
        self.Ei = Ei  # axial stiffness E1 and shear stiffnesses E2 and E3
        self.Fi = Fi  # torsional stiffness F1 and both flexural stiffnesses F2 and F3

    def potential(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the strain energy density `\psi(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        potential : float
            Returns the scalar potential for the given material law.

        """
        dga = Gamma_i - Gamma0_i
        dka = Kappa_i - Kappa0_i
        return 0.5 * self.Ei @ (dga * dga) + 0.5 * self.Fi @ (dka * dka)

    def n_i(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the contact forces `n_i(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        n : :class:`numpy.ndarray` with shape=(3,)
            Returns the vector valued contact forces for the given material law.

        """
        dga = Gamma_i - Gamma0_i
        return self.Ei * dga

    def m_i(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the contact couples `m_i(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        m : :class:`numpy.ndarray` with shape=(3,)
            Returns the vector valued contact couples for the given material law.

        """
        dka = Kappa_i - Kappa0_i

        return self.Fi * dka

    def n_i_Gamma_i_j(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the stiffness of the contact forces `\pd{n_i}{\Gamma_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dn_dGamma_i : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact forces for the given material law.

        """
        return np.diag(self.Ei)

    def n_i_Kappa_i_j(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the stiffness of the contact forces `\pd{n_i}{\Kappa_i_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dn_dKappa_i : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact forces for the given material law.

        """
        return np.zeros((3, 3))

    def m_i_Gamma_i_j(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the stiffness of the contact couples `\pd{m_i}{\Gamma_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dm_dGamma_i : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact couples for the given material law.

        """
        return np.zeros((3, 3))

    def m_i_Kappa_i_j(self, Gamma_i, Gamma0_i, Kappa_i, Kappa0_i):
        """Compute the stiffness of the contact couples `\pd{m_i}{\Kappa_i_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        Gamma_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        Gamma0_i : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        Kappa_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        Kappa0_i : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dm_dKappa_i : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact couples for the given material law.

        """
        return np.diag(self.Fi)
