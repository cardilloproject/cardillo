import numpy as np
from cardillo.math.algebra import norm3

class Hooke(object):
    r"""Simple Hooke material model which accounts for compression effects in simple shear experiments.
    
    The potential function is given by

    .. math::

        \psi(\vGa, \vka) = \frac{1}{2} E_1 \left(\frac{\|\vGa\|}{\|\vGa_0\|} - 1\right)^2 + \frac{1}{2} E_\alpha \left(\Gamma_\alpha - \Gamma_\alpha^0\right)^2 + \frac{1}{2} F_i \left(\kappa_i - \kappa_i^0\right)^2

    The contact forces $n_i$ and couples $m_i$ are computes as

    .. math::

        n_i(\vGa, \vka) &= E_1 \left(\frac{1}{\|\vGa_0\|} - \frac{1}{\|\vGa\|}\right) \frac{\Gamma_i}{\|\vGa_0\|} + \sum_{\alpha=2}^{3} \delta_{i\alpha} E_\alpha \left(\Gamma_\alpha - \Gamma_\alpha^0\right) \; , \\
        m_i(\vGa, \vka) &= F_i \left(\kappa_i - \kappa_i^0\right) \; ,

    together with their stiffnesses

    .. math::

        \pd{n_i}{\Gamma_j}(\vGa, \vka) &= \frac{E_1}{\|\vGa_0\|} \left[ \left(\frac{1}{\|\vGa_0\|} - \frac{1}{\|\vGa\|} \right) \delta_{ij} + \frac{\Gamma_i \Gamma_j}{\|\vGa\|^3}\right] + \sum_{\alpha=2}^{3} \delta_{i\alpha} E_\alpha \delta_{\alpha j} \; , \\
        \pd{n_i}{\kappa_j} &= \mathbf{0} \; , \quad \pd{m_i}{\Gamma_j} = \mathbf{0} \; , \quad \pd{m_i}{\kappa_j} = F_i \delta_{ij} \ve_i \otimes \ve_j \; ,

    Parameters
    ----------
    Ei : :class:`numpy.ndarray` with shape=(3,)
        Axial stiffness $E_1$ and shear stiffnesses $E_2, E_3$. 
    Fi : :class:`numpy.ndarray` with shape=(3,)
        Torsional stiffness $F_1$ and flexural stiffnesses $F_2, F_3$.  
    """

    def __init__(self, Ei, Fi):
        self.Ei = Ei # axial stiffness E1 and shear stiffnesses E2 and E3
        self.Fi = Fi # torsional stiffness F1 and both flexural stiffnesses F2 and F3
        
    def potential(self, gamma, gamma0, kappa, kappa0):
        """Compute the strain energy density `\psi(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        potential : float
            Returns the scalar potential for the given material law.

        """
        nga = norm3(gamma)
        nga0 = norm3(gamma0)
        dla = nga / nga0 - 1.0
        dga1 = gamma[1] - gamma0[1]
        dga2 = gamma[2] - gamma0[2]
        dka = kappa - kappa0

        return 0.5 * self.Ei[0] * dla * dla \
                + 0.5 * self.Ei[1] * dga1 * dga1 \
                + 0.5 * self.Ei[2] * dga2 * dga2 \
                + 0.5 * self.Fi @ (dka * dka)

    def n(self, gamma, gamma0, kappa, kappa0):
        """Compute the contact forces `n_i(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        n : :class:`numpy.ndarray` with shape=(3,)
            Returns the vector valued contact forces for the given material law.

        """
        nga = norm3(gamma)
        nga0 = norm3(gamma0)
        dga1 = gamma[1] - gamma0[1]
        dga2 = gamma[2] - gamma0[2]

        return self.Ei[0] * (1 / nga0 - 1 / nga) * gamma / nga0 \
                + self.Ei[1] * dga1 * np.array([0, 1, 0]) \
                + self.Ei[2] * dga2 * np.array([0, 0, 1])

    def m(self, gamma, gamma0, kappa, kappa0):
        """Compute the contact couples `m_i(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        m : :class:`numpy.ndarray` with shape=(3,)
            Returns the vector valued contact couples for the given material law.

        """
        dka = kappa - kappa0

        return self.Fi * dka

    def dn_dgamma(self, gamma, gamma0, kappa, kappa0):
        """Compute the stiffness of the contact forces `\pd{n_i}{\Gamma_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dn_dgamma : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact forces for the given material law.

        """
        nga = norm3(gamma)
        nga0 = norm3(gamma0)
        
        dn_dgamma1 = self.Ei[0] / (nga * nga * nga * nga0) * np.outer(gamma, gamma)
        
        dn_dgamma2 = self.Ei[0] * (1 / nga0 - 1 / nga) / nga0 * np.diag(np.ones(3))

        dn_dgamma3 = np.diag(self.Ei)
        dn_dgamma3[0, 0] = 0

        return dn_dgamma1 + dn_dgamma2 + dn_dgamma3

    def dn_dkappa(self, gamma, gamma0, kappa, kappa0):
        """Compute the stiffness of the contact forces `\pd{n_i}{\kappa_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dn_dkappa : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact forces for the given material law.

        """
        return np.zeros((3, 3))

    def dm_dgamma(self, gamma, gamma0, kappa, kappa0):
        """Compute the stiffness of the contact couples `\pd{m_i}{\Gamma_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dm_dgamma : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact couples for the given material law.

        """
        return np.zeros((3, 3))

    def dm_dkappa(self, gamma, gamma0, kappa, kappa0):
        """Compute the stiffness of the contact couples `\pd{m_i}{\kappa_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dm_dkappa : :class:`numpy.ndarray` with shape=(3,3)
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
        self.Ei = Ei # axial stiffness E1 and shear stiffnesses E2 and E3
        self.Fi = Fi # torsional stiffness F1 and both flexural stiffnesses F2 and F3
        
    def potential(self, gamma, gamma0, kappa, kappa0):
        """Compute the strain energy density `\psi(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        potential : float
            Returns the scalar potential for the given material law.

        """
        dga = gamma - gamma0
        dka = kappa - kappa0
        return 0.5 * self.Ei @ (dga * dga) + 0.5 * self.Fi @ (dka * dka)

    def n(self, gamma, gamma0, kappa, kappa0):
        """Compute the contact forces `n_i(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        n : :class:`numpy.ndarray` with shape=(3,)
            Returns the vector valued contact forces for the given material law.

        """
        dga = gamma - gamma0
        return self.Ei * dga

    def m(self, gamma, gamma0, kappa, kappa0):
        """Compute the contact couples `m_i(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        m : :class:`numpy.ndarray` with shape=(3,)
            Returns the vector valued contact couples for the given material law.

        """
        dka = kappa - kappa0

        return self.Fi * dka

    def dn_dgamma(self, gamma, gamma0, kappa, kappa0):
        """Compute the stiffness of the contact forces `\pd{n_i}{\Gamma_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dn_dgamma : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact forces for the given material law.

        """
        return np.diag(self.Ei)

    def dn_dkappa(self, gamma, gamma0, kappa, kappa0):
        """Compute the stiffness of the contact forces `\pd{n_i}{\kappa_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dn_dkappa : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact forces for the given material law.

        """
        return np.zeros((3, 3))

    def dm_dgamma(self, gamma, gamma0, kappa, kappa0):
        """Compute the stiffness of the contact couples `\pd{m_i}{\Gamma_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dm_dgamma : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact couples for the given material law.

        """
        return np.zeros((3, 3))

    def dm_dkappa(self, gamma, gamma0, kappa, kappa0):
        """Compute the stiffness of the contact couples `\pd{m_i}{\kappa_j}(\\vGa, \\vka)`.

        Parameters
        ----------
        gamma : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the current configuration
        gamma0 : :class:`numpy.ndarray` with shape=(3,)
            Axial and shear strains in the reference configuration
        kappa : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the current configuration
        kappa0 : :class:`numpy.ndarray` with shape=(3,)
            Torsional and flexural strains in the reference configuration

        Returns
        -------
        dm_dkappa : :class:`numpy.ndarray` with shape=(3,3)
            Returns the stiffness of the contact couples for the given material law.

        """
        return np.diag(self.Fi)