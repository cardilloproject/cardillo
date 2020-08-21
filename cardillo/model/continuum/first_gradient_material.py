from abc import ABC, abstractmethod
import numpy as np
from math import sqrt
from cardillo.math.algebra import determinant2D, determinant3D

class Material_model_ev(ABC):
    """Abstract base class for Ogden type material models.
    """

    def __init__(self, ndim=3):
        self.ndim = ndim
        if ndim == 2:
            self.J = lambda A: determinant2D(A)
        elif ndim == 3:
            self.J = lambda A: determinant3D(A)
        else:
            raise ValueError('ndim has to be 2 or 3')

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
        w, v = self.EV(C)
        J = self.J(C)
        Sk = np.zeros((self.ndim, self.ndim))
        for n in range(self.ndim):
            Sk += self.dW_dw(C, w, n, J) * np.outer(v[:,n], v[:,n])
        return 2 * Sk
        
    def P(self, F):
        r"""First Piola Kirchhoff stress.
        """
        C = F.T @ F
        # TODO: proof if F @ S is the single contraction or if we have to use F.T @ S!
        return F @ self.S(C)
        
    #############################################################
    # abstract methods for energy functions and their derivatives
    # every derived material model has to implement these methods
    #############################################################
    @abstractmethod
    def W(self, C):
        return
               
    @abstractmethod
    def dW_dw(self, C, w, n, J):
        return
        
# # TODO: Needs implementation of pressure
# # Bonet1997 p. 176 6.6.1 / Ogden1985 p.221
# class Ogden_comp(Material_model_ev):
#     def __init__(self, al, mu, d, ndim):
#         Material_model_ev.__init__(self, ndim)
#         self.al = al
#         self.mu = mu
#         self.d = d
#         self.ndim = ndim

#     def W(self,C):
#         J = self.J(C)
#         w, _ = self.EV(C)
#         Wk = 0
#         for k in range(len(self.mu)):
#             Wk += self.mu[k] / self.al[k] * (sum(wn**(self.al[k]/2) for wn in w) - self.ndim) + 1 / self.d[k] * (J-1)**(2 * (k+1))
#         return Wk

#     def dW_dw(self,C,w,n,J):
#         Wk = 0
#         for k in range(len(self.mu)):
#             Wk += self.mu[k] / self.al[k] * (self.al[k] / 2 * w[n]**(self.al[k] / 2 - 1)) + J * (k+1) / (self.d[k] * w[n]) * (J-1)**(2 * k + 1) 
#         return Wk

# #bonet 1997 p. 176 6.6.1 / Ogden 1985 p.221
# class Ogden_incomp(Material_model_ev):
#     def __init__(self,al,mu,d,ndim):
#         Material_model_ev.__init__(self, ndim)
#         self.al = al
#         self.mu = mu
#         self.d = d
#         self.ndim = ndim
#     # Plane stress assumption
#     def W(self,C):
#         J = self.J(C)
#         w, _ = self.EV(C)
#         Wk = 0
#         for k in range(len(self.mu)):
#             Wk += self.mu[k] / self.al[k] * (sum(wn**(self.al[k]/2) for wn in w) + 1 / J**(self.al[k]) - 3)
#         return Wk

#     def dW_dw(self,C,w,n,J):
#         Wk = 0
#         for k in range(len(self.mu)):
#             Wk += self.mu[k] / self.al[k] * (self.al[k] / 2 * w[n]**(self.al[k] / 2 - 1) - self.al[k] / (2 * w[n]) * 1 / J**(self.al[k]))
#         return Wk  

# class Ogden_incomp_plane_strain(Material_model_ev):
#     def __init__(self,al,mu,d,ndim):
#         Material_model_ev.__init__(self, ndim)
#         self.al = al
#         self.mu = mu
#         self.d = d
#         self.ndim = ndim
#     # Plane strain assumption
#     # Simo 1985 p. 284
#     def W(self,C):
#         w, _ = self.EV(C)
#         Wk = 0
#         for k in range(len(self.mu)):
#             Wk += self.mu[k] / self.al[k] * (sum(wn**(self.al[k]/2) for wn in w) - 2)
#         return Wk

#     def dW_dw(self,C,w,n,J):
#         #dW_iso_dw
#         Wk = 0
#         for k in range(len(self.mu)):
#             Wk += self.mu[k] / 2 * J**(-self.al[k]/3) * ( w[n]**(self.al[k] / 2) - 1/3 * sum(wn**(self.al[k]/2) for wn in w) - 1/3)
#         return Wk  

# 3D Ogden type material model
class Ogden(Material_model_ev):
    def __init__(self, al, mu, d):
        Material_model_ev.__init__(self)
        self.al = al
        self.mu = mu
        self.d = d

    def W(self, C):
        J = self.J(C)
        w, _ = self.EV(C)
        Wk = 0
        for k in range(len(self.mu)):
            Wk += self.mu[k] / self.al[k] * (sum(wn**(self.al[k] / 2) for wn in w) - 3 - 1 / self.d[k] * np.log(J)) #+ 1 / self.d[k] * (J-1)**(2 * (k+1))
        return Wk

    def dW_dw(self, C, w, n, J):
        dW_dw = 0
        for k in range(len(self.mu)):
            dW_dw += self.mu[k] * (0.5 * w[n]**(self.al[k] / 2 - 1) - 1 / self.d[k] / w[n]) #+ J * (k+1) / (self.d[k] * w[n]) * (J-1)**(2 * k + 1) 
        return dW_dw