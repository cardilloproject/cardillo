from abc import ABC, abstractmethod
import numpy as np
from math import sqrt, log
from cardillo.math.algebra import determinant2D, determinant3D

class Material_model_ev(ABC):
    """Abstract base class for Ogden type material models.
    """

    def __init__(self, dim=3):
        self.dim = dim
        if dim == 2:
            self.J = lambda C: sqrt(determinant2D(C))
        elif dim == 3:
            self.J = lambda C: sqrt(determinant3D(C))
        else:
            raise ValueError('dim has to be 2 or 3')

    def EV(self, A):
        r"""Calculate eigenvalues and eigenvectors of A
        """
        # TODO
        # return np.linalg.eigh(A)
        return np.linalg.eig(A)

    #############################################################
    # stresses
    #############################################################
        
    def S(self, C):
        r"""Second Piola Kirchhoff stress.
        """
        la2, u = self.EV(C)
        S = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            S += self.W_w(C, la2, i) * np.outer(u[:, i], u[:, i])
        return 2 * S
        
    def P(self, F):
        r"""First Piola Kirchhoff stress.
        """
        return F @ self.S(F.T @ F)
        
    #############################################################
    # abstract methods for energy functions and their derivatives
    # every derived material model has to implement these methods
    #############################################################
    @abstractmethod
    def W(self, C):
        return
               
    @abstractmethod
    def W_w(self, C, la2, i):
        return

class Ogden1997_compressible():
    """Ogden 1997 p. 222, (4.4.1)
    """
    def __init__(self, mu1, mu2):
        self.mu1 = mu1
        self.mu2 = mu2

    def W(self, F):
        la2, _ = np.linalg.eigh(F.T @ F)
        
        I1 = sum(la2)
        I3 = np.prod(la2)
        J = sqrt(I3)
        lnJ = log(J)

        return self.mu1 / 2 * (I1 - 3) \
               - self.mu1 * lnJ \
               + self.mu2 * (J - 1)**2

    def S(self, F):
        la2, u = np.linalg.eigh(F.T @ F)
        J = sqrt(np.prod(la2))

        S = np.zeros_like(F)
        for A in range(len(la2)):
            SA = self.mu1 * (1 - 1 / la2[A]) \
                 + self.mu2 * J * (J - 1) / la2[A]
            S += SA * np.outer(u[:, A], u[:, A])
        return S

    def P(self, F):
        return F @ self.S(F)
        
class Ogden1997_incompressible():
    """Ogden 1997 p. 293, (7.2.20)
    """
    def __init__(self, mu, alpha):
        self.mu = mu
        self.alpha = alpha

    def W(self, F):        
        raise NotImplementedError('...')

    def P(self, F):
        raise NotImplementedError('...')

def test_Ogden1997_compressible():
    mu1 = 0.3
    mu2 = 0.5
    mat = Ogden1997_compressible(mu1, mu2)

    # F = np.random.rand(3, 3)
    F = np.eye(3)
    # F = np.diag(np.array([1, 1, 0.95]))
    C = F.T @ F

    W = mat.W(C)
    print(f'W: {W}')
    
    P = mat.P(F)
    print(f'P:\n{P}')

if __name__ == "__main__":
    test_Ogden1997_compressible()