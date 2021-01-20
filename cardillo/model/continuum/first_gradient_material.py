# from abc import ABC, abstractmethod
import numpy as np
from math import sqrt, log, isclose
from cardillo.math.algebra import determinant2D, determinant3D
from cardillo.math.numerical_derivative import Numerical_derivative

# class Material_model_ev(ABC):
#     """Abstract base class for Ogden type material models.
#     """

#     def __init__(self, dim=3):
#         self.dim = dim
#         if dim == 2:
#             self.J = lambda C: sqrt(determinant2D(C))
#         elif dim == 3:
#             self.J = lambda C: sqrt(determinant3D(C))
#         else:
#             raise ValueError('dim has to be 2 or 3')

#     def EV(self, A):
#         r"""Calculate eigenvalues and eigenvectors of A
#         """
#         # TODO
#         # return np.linalg.eigh(A)
#         return np.linalg.eig(A)

#     #############################################################
#     # stresses
#     #############################################################
        
#     def S(self, C):
#         r"""Second Piola Kirchhoff stress.
#         """
#         la2, u = self.EV(C)
#         S = np.zeros((self.dim, self.dim))
#         for i in range(self.dim):
#             S += self.W_w(C, la2, i) * np.outer(u[:, i], u[:, i])
#         return 2 * S
        
#     def P(self, F):
#         r"""First Piola Kirchhoff stress.
#         """
#         return F @ self.S(F.T @ F)
        
#     #############################################################
#     # abstract methods for energy functions and their derivatives
#     # every derived material model has to implement these methods
#     #############################################################
#     @abstractmethod
#     def W(self, C):
#         return
               
#     @abstractmethod
#     def W_w(self, C, la2, i):
#         return

class Ogden1997_compressible():
    """Ogden 1997 p. 222, (4.4.1)
    """
    def __init__(self, mu1, mu2, dim=3):
        self.mu1 = mu1
        self.mu2 = mu2
        self.dim = dim

    def W(self, F):
        La, _ = np.linalg.eigh(F.T @ F)
        
        I1 = sum(La)
        I3 = np.prod(La)
        J = sqrt(I3)
        lnJ = log(J)

        return self.mu1 / 2 * (I1 - 3) \
               - self.mu1 * lnJ \
               + self.mu2 * (J - 1)**2

    def S(self, F):
        La, u = np.linalg.eigh(F.T @ F)
        J = sqrt(np.prod(La))

        S = np.zeros_like(F)
        for i in range(len(La)):
            Si = self.mu1 * (1 - 1 / La[i]) \
                 + self.mu2 * J * (J - 1) / La[i]
            S += Si * np.outer(u[:, i], u[:, i])
        return S

    def S_F(self, F):
        La, u = np.linalg.eigh(F.T @ F)
        J = sqrt(np.prod(La))

        Si = np.zeros(self.dim)
        Si_Laj = np.zeros((self.dim, self.dim))
        for i in range(len(La)):
            Si[i] = self.mu1 * (1 - 1 / La[i]) \
                    + self.mu2 * J * (J - 1) / La[i]

            Si_Laj[i, i] += (self.mu1 - self.mu2 * J * (J - 1)) / La[i]**2
            for j in range(len(La)):
                Si_Laj[i, j] += self.mu2 * J * (J - 0.5) / (La[j] * La[i])

        S_C = np.zeros((self.dim, self.dim, self.dim, self.dim))
        for i in range(len(La)):
            for j in range(len(La)):
                S_C += Si_Laj[i, j] * np.einsum('i,j,k,l->ijkl', u[:, i], u[:, i], u[:, j], u[:, j])
                if i != j:
                    # for La[j] -> La[i] we use L'Hôpital's rule, see Connolly2019 - Isotropic hyperelasticity in principal stretches: explicit elasticity tensors and numerical implementation
                    if isclose(La[i], La[j]):
                        S_C += 0.5 * (Si_Laj[j, j] - Si_Laj[i, j]) * (np.einsum('i,j,k,l->ijkl', u[:, i], u[:, j], u[:, i], u[:, j]) + np.einsum('i,j,k,l->ijkl', u[:, i], u[:, j], u[:, j], u[:, i]))
                    else:
                        S_C += 0.5 * ((Si[j] - Si[i]) / (La[j] - La[i])) * (np.einsum('i,j,k,l->ijkl', u[:, i], u[:, j], u[:, i], u[:, j]) + np.einsum('i,j,k,l->ijkl', u[:, i], u[:, j], u[:, j], u[:, i]))

        eye_dim = np.eye(self.dim)
        C_F = np.einsum('kj,il->ijkl', F, eye_dim) + np.einsum('ki,jl->ijkl', F, eye_dim)
        S_F = np.einsum('ijkl,klmn->ijmn', S_C, C_F)

        # S_F_num = Numerical_derivative(self.S, order=2)._X(F)
        # error = np.linalg.norm(S_F - S_F_num)
        # # print(f'error: {error}')
        # if error > 1.0e-5:
        #     print(f'error: {error}')
        # return S_F_num

        return S_F

    def P(self, F):
        return F @ self.S(F)
          

class Ogden1997_complete_2D_incompressible():
    """Ogden 1997 p. 222, (4.4.1)
        Plane stress assumption
    """ 

    def __init__(self, mu, al, dim=2):
        self.mu = mu
        self.al = al
        self.dim = dim

    def W(self,F):
        La, u = np.linalg.eigh(F.T @ F)
        J = sqrt(np.prod(La))
        W = 0
        for k in range(len(self.mu)):
            # W += self.mu[k] / self.al[k] * (sum(wn**(self.al[k]/2) for wn in La) + 1 / J**(self.al[k]) - 3)
            W += self.mu[k] / self.al[k] * (sum(wn**(self.al[k]/2) for wn in La) + 1 / J**(self.al[k]) - 3)
        return W

    def S(self, F):
        La, u = np.linalg.eigh(F.T @ F)
        J = sqrt(np.prod(La))

        S = np.zeros_like(F)
        for i in range(len(La)):
            Si = 0
            for k in range(len(self.mu)):
                # Si += self.mu[k] / 2 * ( La[i]**(self.al[k] / 2 - 1) -  1 / ( La[i]) * 1 / J**(self.al[k]))
                Si += 2 * self.mu[k] / self.al[k] * (self.al[k] / 2 * La[i]**(self.al[k] / 2 - 1) - self.al[k] / (2 * La[i]) * 1 / J**(self.al[k]))

            S += Si * np.outer(u[:, i], u[:, i])
        return S

    def S_F(self, F):
        La, u = np.linalg.eigh(F.T @ F)
        J = sqrt(np.prod(La))

        Si = np.zeros(self.dim)
        Si_Laj = np.zeros((self.dim, self.dim))
        for i in range(len(La)):
            for k in range(len(self.mu)):
                # Si[i] += self.mu[k] / 2 * ( La[i]**(self.al[k] / 2 - 1) - 1 / ( La[i]) * 1 / J**(self.al[k]))
                Si[i] += 2 * self.mu[k] / self.al[k] * (self.al[k] / 2 * La[i]**(self.al[k] / 2 - 1) - self.al[k] / (2 * La[i]) * 1 / J**(self.al[k]))

                Si_Laj[i, i] += 2 * self.mu[k] / 2  * ( (self.al[k] / 2 - 1) * La[i]**(self.al[k] / 2 - 2) +  (self.al[k] / 2 + 1) * 1 / ( La[i]**2) * 1 / J**(self.al[k]))
            for j in range(len(La)):
                if i != j:
                    for k in range(len(self.mu)):
                        Si_Laj[i, j] += 2 * self.mu[k] / 2 * self.al[k] / 2 * (1 / J**(self.al[k] + 2))   

        S_C = np.zeros((self.dim, self.dim, self.dim, self.dim))
        for i in range(len(La)):
            for j in range(len(La)):
                S_C += Si_Laj[i, j] * np.einsum('i,j,k,l->ijkl', u[:, i], u[:, i], u[:, j], u[:, j])
                if i != j:
                    # for La[j] -> La[i] we use L'Hôpital's rule, see Connolly2019 - Isotropic hyperelasticity in principal stretches: explicit elasticity tensors and numerical implementation
                    if isclose(La[i], La[j]):
                        S_C += 0.5 * (Si_Laj[j, j] - Si_Laj[i, j]) * (np.einsum('i,j,k,l->ijkl', u[:, i], u[:, j], u[:, i], u[:, j]) + np.einsum('i,j,k,l->ijkl', u[:, i], u[:, j], u[:, j], u[:, i]))
                    else:
                        S_C += 0.5 * ((Si[j] - Si[i]) / (La[j] - La[i])) * (np.einsum('i,j,k,l->ijkl', u[:, i], u[:, j], u[:, i], u[:, j]) + np.einsum('i,j,k,l->ijkl', u[:, i], u[:, j], u[:, j], u[:, i]))

        eye_dim = np.eye(self.dim)
        C_F = np.einsum('kj,il->ijkl', F, eye_dim) + np.einsum('ki,jl->ijkl', F, eye_dim)
        S_F = np.einsum('ijkl,klmn->ijmn', S_C, C_F)

        # S_F_num = Numerical_derivative(self.S, order=2)._X(F)
        # error = np.linalg.norm(S_F - S_F_num)
        # # print(f'error: {error}')
        # if error > 1.0e-5:
        #     print(f'error: {error}')
        # return S_F_num

        return S_F

    def P(self, F):
        return F @ self.S(F)

    def test_derivatives(self):
        from cardillo.math.numerical_derivative import Numerical_derivative
        def W(La):
            J = sqrt(np.prod(La))
            W = 0
            for k in range(len(self.mu)):
                # W += self.mu[k] / self.al[k] * (sum(wn**(self.al[k]/2) for wn in La) + 1 / J**(self.al[k]) - 3)
                W += self.mu[k] / self.al[k] * (sum(wn**(self.al[k]/2) for wn in La) + 1 / J**(self.al[k]) - 3)
            return W

        def W_La(La):
            J = sqrt(np.prod(La))
            W_La = np.zeros(len(La))
            for i in range(len(La)):
               
                for k in range(len(self.mu)):
                    W_La[i] += self.mu[k] / self.al[k] * (self.al[k] / 2 * La[i]**(self.al[k] / 2 - 1) - self.al[k] / (2 * La[i]) * 1 / J**(self.al[k]))
            return W_La

        def W_La_La(La):
            J = sqrt(np.prod(La))
            W_La_La = np.zeros((len(La), len(La)))
            for i in range(len(La)):

                for k in range(len(self.mu)):
                    W_La_La[i,i] += self.mu[k] / 2  * ( (self.al[k] / 2 - 1) * La[i]**(self.al[k] / 2 - 2) +  (self.al[k] / 2 + 1) * 1 / ( La[i]**2) * 1 / J**(self.al[k]))
                    

                for j in range(len(La)):
                    if i != j:
                        for k in range(len(self.mu)):
                            W_La_La[i,j] += self.mu[k] / 2 * self.al[k] / 2 * (1 / J**(self.al[k] + 2))          

            return W_La_La  
                  
        La_0 = np.array([2.,3.])
        W_La_num = Numerical_derivative(lambda La: W(La))._X(La_0)
        W_La_an = W_La(La_0)

        W_La_La_an = W_La_La(La_0)
        W_La_La_num = Numerical_derivative(lambda La: W_La(La))._X(La_0)
        print(".")

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

def test_Ogden1997_complete_2D_incompressible():
    al = [1.3, 5, -2]    # values: see Ogden 1997 p. 498
    #mu = [6.3, 0.012, -0.1]
    mu_star = [1.491, 0.003, -0.0237]
    mu = [mu_star[i] *4.225 for i in range(len(al))]
    mat = Ogden1997_complete_2D_incompressible(mu, al)
    mat.test_derivatives()

if __name__ == "__main__":
    # test_Ogden1997_compressible()
    test_Ogden1997_complete_2D_incompressible()