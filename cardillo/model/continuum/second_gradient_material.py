# from abc import ABC, abstractmethod
import numpy as np
from math import sqrt, log, isclose
from cardillo.math.algebra import determinant2D, determinant3D, inverse3D, norm, norm3, cross3
from cardillo.math.numerical_derivative import Numerical_derivative

class Pantobox_beam_network():
    """Based on Giorgio 2017 extended to 4 fibers
    """
    def __init__(self, Ke, Ks, Kg, Kn, Kt, Kc, dim=3):
        # Beam stiffnesses
        self.Ke = Ke
        self.Ks = Ks
        self.Kg = Kg
        self.Kn = Kn
        self.Kt = Kt
        self.Kc = Kc
        # Fiber directions in reference configuration TODO: make it choosable
        self.D1 = np.array([1, 1, 0]) / np.sqrt(2)
        self.D2 = np.array([-1, 1, 0]) / np.sqrt(2)
        self.D3 = np.array([0, 1, 1]) / np.sqrt(2)
        self.D4 = np.array([0, -1, 1]) / np.sqrt(2)
        self.D = np.array([self.D1, self.D2, self.D3, self.D4])

    def W(self, F, G):
        # strain energy density
        # Layer a
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1  # same as G@D1@D1 speed is similar
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        tau1 = na @ (g1 - singa * c1) / cosga
        tau2 = na @ (g2 - singa * c2) / cosga

        # Layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        singb = e3 @ e4
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4
        g3 = np.einsum('ijk,j,k->i', G, self.D4, self.D3) / rho4
        g4 = np.einsum('ijk,j,k->i', G, self.D3, self.D4) / rho3

        tau3 = nb @ (g3 - singb * c3) / cosgb
        tau4 = nb @ (g4 - singb * c4) / cosgb

        singc = na @ nb

        # complete energy
        W = .5 * self.Ke * ((rho1-1)**2 + (rho2-1)**2 + (rho3-1)**2 + (rho4-1)**2) \
            + .5 * self.Kg * ((na@c1)**2 + (na@c2)**2 + (nb@c3)**2 + (nb@c4)**2) \
            + .5 * self.Kn * ((m1@c1)**2 + (m2@c2)**2 + (m3@c3)**2 + (m4@c4)**2) \
            + .5 * self.Kt * (tau1**2 + tau2**2 + tau3**2 + tau4**2) \
            + .5 * self.Ks * (np.arcsin(singa)**2 + np.arcsin(singb)**2) \
            + .5 * self.Kc * (np.arcsin(singc)**2)

        return W

    def P(self, F, G):
        """Piola Lagrange stress tensor
        """
        P_num = Numerical_derivative(lambda F: self.W(F, G), order=2)._X(F)
        return P_num

    def bbP(self, F, G):
        """Piola Lagrange double stress
        """
        bbP_num = Numerical_derivative(lambda G: self.W(F, G), order=2)._X(G)
        return bbP_num

    def P_F(self, F, G):
        P_F_num = Numerical_derivative(lambda F: self.P(F, G), order=2)._X(F)
        return P_F_num

    def P_G(self, F, G):
        P_G_num = Numerical_derivative(lambda G: self.P(F, G), order=2)._X(G)
        return P_G_num

    def bbP_F(self, F, G):
        bbP_F_num = Numerical_derivative(lambda F: self.bbP(F, G), order=2)._X(F)
        return bbP_F_num

    def bbP_G(self, F, G):
        bbP_G_num = Numerical_derivative(lambda G: self.bbP(F, G), order=2)._X(G) 
        return bbP_G_num       
# %%
# import numpy as np
# import time
# A=np.ones((30,30,30))
# B=np.ones((30,30))
# C=np.random.rand(30)
# print(C)
# t1 = time.time()
# print(np.einsum('ijk,j,k->i',A,C,C))
# t2 = time.time()
# print(A@C@C)
# t3=time.time()
# print(t2-t1)
# print(t3-t2)
