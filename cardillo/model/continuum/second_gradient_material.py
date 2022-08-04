# from abc import ABC, abstractmethod
import numpy as np
from math import sqrt, log, isclose
from cardillo.math.algebra import determinant2D, determinant3D, inverse3D, norm, norm3, cross3, LeviCivita3
from cardillo.math.numerical_derivative import Numerical_derivative

num_order = 2
G0 = np.zeros((3,3,3))

class Pantosheet_beam_network():
    """Giorgio et al 2017 Surface embedded in 3D space
    """

    def __init__(self, Ke, Ks, Kg, Kn, Kt, dim=2, numerical_derivative=True):
        # Beam stiffnesses
        self.Ke = Ke
        self.Ks = Ks
        self.Kg = Kg
        self.Kn = Kn
        self.Kt = Kt
        # Fiber directions in reference configuration TODO: make it choosable
        self.D1 = np.array([1, 1, 0]) / np.sqrt(2)
        self.D2 = np.array([-1, 1, 0]) / np.sqrt(2)
        self.D = np.array([self.D1, self.D2])
        self.I2 = np.eye(2)
       # self.lc3 = np.array([[[LeviCivita3(i, j, k) for k in range(3)]
       #                     for j in range(3)] for i in range(3)])
        if numerical_derivative:
            self.P_F = self.P_F_num
            self.P_G = self.P_G_num
            self.bbP_F = self.bbP_F_num
            self.bbP_G = self.bbP_G_num

    def P(self, F, G):
        """Piola-Lagrange stress tensor
        """
        # P_num = Numerical_derivative(lambda F: self.W(F, G), order=num_order)._X(F)
        # Parameters
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
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        kg1 = na @ c1
        kg2 = na @ c2
        kn1 = - m1 @ c1
        kn2 = - m2 @ c2
        tau1 = - na @ (g1 - singa * c1) / cosga
        tau2 = - na @ (g2 - singa * c2) / cosga

        # Elongation
        Pe = self.Ke * ((rho1-1) * np.outer(e1, self.D1) +
                        (rho2-1) * np.outer(e2, self.D2))

        # Shear
        PsaD1 = m1 / rho1
        PsaD2 = -m2 / rho2

        Psa = self.Ks * \
            np.arcsin(singa) * (np.outer(PsaD1, self.D1) +
                                np.outer(PsaD2, self.D2))

        # PcD1 = (singc * tanga * m1 + cross3(e1,
        #         cross3(cross3(e2, nb), e1)) / cosga) / rho1
        # PcD2 = (-singc * tanga * m2 - cross3(e2,
        #         cross3(cross3(e1, nb), e2)) / cosga) / rho2

        # # Pc = self.Kc * np.arcsin(singc) / cosgc * \
        #     (np.outer(PcD1, self.D1) + np.outer(PcD2, self.D2))

        # Normal curvature
        Pn1D1 = ((kn1 * tanga + e1 @ c1) * m1
                 + tanga * cross3(e1, cross3(c1, e1)) - kn1 * e1) / rho1
        Pn1D2 = ((-kn1 * singa - cosga * e1 @ c1) * m2
                 - cross3(e2, cross3(c1, e2))) / (rho2 * cosga)

        Pn2D1 = ((kn2 * singa - cosga * e2 @ c2) * m1
                 + cross3(e1, cross3(c2, e1))) / (cosga * rho1)
        Pn2D2 = ((- kn2 * tanga + e2 @ c2) * m2
                 - tanga * cross3(e2, cross3(c2, e2)) - kn2 * e2) / rho2

        Pn = self.Kn * ((np.outer(kn1*Pn1D1 + kn2*Pn2D1, self.D1)
                        + np.outer(kn1*Pn1D2 + kn2*Pn2D2, self.D2)))

        # Geodesic curvature
        Pg1D1 = (kg1 * singa * m1 + cross3(e1, cross3(cross3(e2, c1), e1))
                 - cosga * kg1 * e1) / (rho1 * cosga)
        Pg1D2 = (- kg1 * singa * m2 - cross3(e2,
                                             cross3(cross3(e1, c1), e2))) / (rho2 * cosga)

        Pg2D1 = (kg2 * singa * m1 + cross3(e1,
                                           cross3(cross3(e2, c2), e1))) / (rho1 * cosga)
        Pg2D2 = (- kg2 * singa * m2 - cross3(e2, cross3(cross3(e1, c2), e2))
                 - cosga * kg2 * e2) / (rho2 * cosga)

        Pg = self.Kg * ((np.outer(kg1*Pg1D1 + kg2*Pg2D1, self.D1)
                         + np.outer(kg1*Pg1D2 + kg2*Pg2D2, self.D2)))

        # Torsion
        Pt1D1 = ((2 * tau1 * singa + kg1 * cosga) * m1 - kg1 * singa * e1
                 - cross3(e1, cross3(cross3(e2, g1 - singa*c1), e1)) / cosga) / (rho1 * cosga)
        Pt1D2 = ((-2 * tau1 * singa - kg1 * cosga) * m2 + (g1@na) * e2
                 + cross3(e2, cross3(cross3(e1, g1-singa*c1), e2)) / cosga) / (rho2 * cosga)

        Pt2D1 = ((2 * tau2 * singa + kg2 * cosga) * m1 + (g2@na) * e1
                 - cross3(e1, cross3(cross3(e2, g2 - singa*c2), e1)) / cosga) / (rho1 * cosga)
        Pt2D2 = ((-2 * tau2 * singa - kg2 * cosga) * m2 - kg2 * singa * e2
                 + cross3(e2, cross3(cross3(e1, g2-singa*c2), e2)) / cosga) / (rho2 * cosga)

        Pt = self.Kt * (np.outer(tau1*Pt1D1 + tau2*Pt2D1, self.D1)
                        + np.outer(tau1*Pt1D2 + tau2*Pt2D2, self.D2))

        return Pe + Psa + Pn + Pg + Pt

    def bbP(self, F, G):
        """Piola-Lagrange double stress tensor
        """
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        kg1 = na @ c1
        kg2 = na @ c2
        kn1 = - m1 @ c1
        kn2 = - m2 @ c2
        tau1 = - na @ (g1 - singa * c1) / cosga
        tau2 = - na @ (g2 - singa * c2) / cosga

        # bbP_num = Numerical_derivative(lambda G: self.W(F, G), order=num_order)._X(G)
        # normal

        # geodesic
        bbPg1 = self.Kg * kg1 / rho1 * \
            np.einsum('i,j,k->ijk', na, self.D1, self.D1)
        bbPg2 = self.Kg * kg2 / rho2 * \
            np.einsum('i,j,k->ijk', na, self.D2, self.D2)

        bbPg = bbPg1 + bbPg2

        # normal
        bbPn1 = - self.Kn * kn1 / rho1 * \
            np.einsum('i,j,k->ijk', m1, self.D1, self.D1)
        bbPn2 = - self.Kn * kn2 / rho2 * \
            np.einsum('i,j,k->ijk', m2, self.D2, self.D2)

        bbPn = bbPn1 + bbPn2

        # torsion
        bbPt1 = self.Kt * tau1 * (tanga * np.einsum('i,j,k->ijk', na, self.D1, self.D1) / rho1
                                  - np.einsum('i,j,k->ijk', na, self.D2, self.D1) / (rho2 * cosga))
        bbPt2 = self.Kt * tau2 * (tanga * np.einsum('i,j,k->ijk', na, self.D2, self.D2) / rho2
                                  - np.einsum('i,j,k->ijk', na, self.D1, self.D2) / (rho1 * cosga))

        bbPt = bbPt1 + bbPt2

        return bbPn + bbPg + bbPt

    # numerical derivatives
    # def P_num(self, F, G, W=W):
    #     P_num = Numerical_derivative(lambda F: W(F, G), order=2)._X(F)
    #     return P_num

    # def bbP_num(self, F, G, W=W):
    #     bbP_num = Numerical_derivative(lambda G: W(F, G), order=2)._X(G)
    #     return bbP_num

    def P_F_num(self, F, G):
        P_F_num = Numerical_derivative(lambda F: self.P(F, G), order=2)._X(F)
        return P_F_num

    def P_G_num(self, F, G):
        P_G_num = Numerical_derivative(lambda G: self.P(F, G), order=2)._X(G)
        return P_G_num

    def bbP_F_num(self, F, G):
        bbP_F_num = Numerical_derivative(
            lambda F: self.bbP(F, G), order=num_order)._X(F)
        return bbP_F_num

    def bbP_G_num(self, F, G):
        bbP_G_num = Numerical_derivative(
            lambda G: self.bbP(F, G), order=num_order)._X(G)
        return bbP_G_num

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
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / \
            rho1  # same as G@D1@D1 speed is similar
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        tau1 = - na @ (g1 - singa * c1) / cosga
        tau2 = - na @ (g2 - singa * c2) / cosga

        # complete energy
        W = .5 * self.Ke * ((rho1-1)**2 + (rho2-1)**2) \
            + .5 * self.Kg * ((na@c1)**2 + (na@c2)**2) \
            + .5 * self.Kn * ((-m1@c1)**2 + (-m2@c2)**2) \
            + .5 * self.Kt * (tau1**2 + tau2**2) \
            + .5 * self.Ks * (np.arcsin(singa)**2)

        return W

    def We(self, F, G=G0):

        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm(d1)
        rho2 = norm(d2)

        We = .5 * self.Ke * ((rho1-1)**2 + (rho2-1)**2)
        return We

    def Ws(self, F, G=G0):
        # layer a
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        singa = e1 @ e2

        Wsa = .5 * self.Ks * np.arcsin(singa)**2
        return Wsa


class Pantobox_beam_network():
    """Based on Giorgio et al 2017 extended to 4 fibers
    """

    def __init__(self, Ke, Ks, Kg, Kn, Kt, Kc, dim=3, numerical_derivative=False, first_grad=False):
        # numerical derivatives
        if numerical_derivative:
            self.P = self.P_num
            self.P_F = self.P_F_num
            self.P_G = self.P_G_num
            self.bbP = self.bbP_num
            self.bbP_F = self.bbP_F_num
            self.bbP_G = self.bbP_G_num

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
        self.I3 = np.eye(3)
        self.lc3 = np.array([[[LeviCivita3(i, j, k) for k in range(3)]
                            for j in range(3)] for i in range(3)])

    def P(self, F, G=G0):
        """Piola-Lagrange stress tensor
        """
        # P_num = Numerical_derivative(lambda F: self.W(F, G), order=num_order)._X(F)
        # Parameters
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
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        kg1 = na @ c1
        kg2 = na @ c2
        kn1 = - m1 @ c1
        kn2 = - m2 @ c2
        tau1 = - na @ (g1 - singa * c1) / cosga
        tau2 = - na @ (g2 - singa * c2) / cosga

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
        tangb = singb / cosgb
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4
        g3 = np.einsum('ijk,j,k->i', G, self.D4, self.D3) / rho4
        g4 = np.einsum('ijk,j,k->i', G, self.D3, self.D4) / rho3

        kg3 = nb @ c3
        kg4 = nb @ c4
        kn3 = - m3 @ c3
        kn4 = - m4 @ c4
        tau3 = - nb @ (g3 - singb * c3) / cosgb
        tau4 = - nb @ (g4 - singb * c4) / cosgb

        singc = na @ nb
        cosgc = norm3(cross3(na, nb))

        # Elongation
        Pe = self.Ke * ((rho1-1) * np.outer(e1, self.D1) + (rho2-1) * np.outer(e2, self.D2)
                        + (rho3-1) * np.outer(e3, self.D3) + (rho4-1) * np.outer(e4, self.D4))

        # Shear
        PsaD1 = m1 / rho1
        PsaD2 = -m2 / rho2
        PsbD3 = m3 / rho3
        PsbD4 = -m4 / rho4
        Psa = self.Ks * \
            np.arcsin(singa) * (np.outer(PsaD1, self.D1) +
                                np.outer(PsaD2, self.D2))
        Psb = self.Ks * \
            np.arcsin(singb) * (np.outer(PsbD3, self.D3) +
                                np.outer(PsbD4, self.D4))

        PcD1 = (singc * tanga * m1 + cross3(e1,
                                            cross3(cross3(e2, nb), e1)) / cosga) / rho1
        PcD2 = (-singc * tanga * m2 - cross3(e2,
                                             cross3(cross3(e1, nb), e2)) / cosga) / rho2
        PcD3 = (singc * tangb * m3 + cross3(e3,
                                            cross3(cross3(e4, na), e3)) / cosgb) / rho3
        PcD4 = (-singc * tangb * m4 - cross3(e4,
                                             cross3(cross3(e3, na), e4)) / cosgb) / rho4
        Pc = self.Kc * np.arcsin(singc) / cosgc * (np.outer(PcD1, self.D1) + np.outer(PcD2, self.D2)
                                                   + np.outer(PcD3, self.D3) + np.outer(PcD4, self.D4))

        # Normal curvature
        Pn1D1 = ((kn1 * tanga + e1 @ c1) * m1
                 + tanga * cross3(e1, cross3(c1, e1)) - kn1 * e1) / rho1
        Pn1D2 = ((-kn1 * singa - cosga * e1 @ c1) * m2
                 - cross3(e2, cross3(c1, e2))) / (rho2 * cosga)

        Pn2D1 = ((kn2 * singa - cosga * e2 @ c2) * m1
                 + cross3(e1, cross3(c2, e1))) / (cosga * rho1)
        Pn2D2 = ((- kn2 * tanga + e2 @ c2) * m2
                 - tanga * cross3(e2, cross3(c2, e2)) - kn2 * e2) / rho2

        Pn3D3 = ((kn3 * tangb + e3 @ c3) * m3
                 + tangb * cross3(e3, cross3(c3, e3)) - kn3 * e3) / rho3
        Pn3D4 = ((-kn3 * singb - cosgb * e3 @ c3) * m4
                 - cross3(e4, cross3(c3, e4))) / (rho4 * cosgb)

        Pn4D3 = ((kn4 * singb - cosgb * e4 @ c4) * m3
                 + cross3(e3, cross3(c4, e3))) / (cosgb * rho3)
        Pn4D4 = ((- kn4 * tangb + e4 @ c4) * m4
                 - tangb * cross3(e4, cross3(c4, e4)) - kn4 * e4) / rho4

        Pn = self.Kn * ((np.outer(kn1*Pn1D1 + kn2*Pn2D1, self.D1)
                         + np.outer(kn1*Pn1D2 + kn2*Pn2D2, self.D2))
                        + np.outer(kn3*Pn3D3 + kn4*Pn4D3, self.D3) \
            + np.outer(kn3*Pn3D4 + kn4*Pn4D4, self.D4))

        # Geodesic curvature
        Pg1D1 = (kg1 * singa * m1 + cross3(e1, cross3(cross3(e2, c1), e1))
                 - cosga * kg1 * e1) / (rho1 * cosga)
        Pg1D2 = (- kg1 * singa * m2 - cross3(e2,
                                             cross3(cross3(e1, c1), e2))) / (rho2 * cosga)

        Pg2D1 = (kg2 * singa * m1 + cross3(e1,
                                           cross3(cross3(e2, c2), e1))) / (rho1 * cosga)
        Pg2D2 = (- kg2 * singa * m2 - cross3(e2, cross3(cross3(e1, c2), e2))
                 - cosga * kg2 * e2) / (rho2 * cosga)

        Pg3D3 = (kg3 * singb * m3 + cross3(e3, cross3(cross3(e4, c3), e3))
                 - cosgb * kg3 * e3) / (rho3 * cosgb)
        Pg3D4 = (- kg3 * singb * m4 - cross3(e4,
                                             cross3(cross3(e3, c3), e4))) / (rho4 * cosgb)

        Pg4D3 = (kg4 * singb * m3 + cross3(e3,
                                           cross3(cross3(e4, c4), e3))) / (rho3 * cosgb)
        Pg4D4 = (- kg4 * singb * m4 - cross3(e4, cross3(cross3(e3, c4), e4))
                 - cosgb * kg4 * e4) / (rho4 * cosgb)

        Pg = self.Kg * ((np.outer(kg1*Pg1D1 + kg2*Pg2D1, self.D1)
                         + np.outer(kg1*Pg1D2 + kg2*Pg2D2, self.D2))
                        + np.outer(kg3*Pg3D3 + kg4*Pg4D3, self.D3) \
            + np.outer(kg3*Pg3D4 + kg4*Pg4D4, self.D4))

        # Torsion
        Pt1D1 = ((2 * tau1 * singa + kg1 * cosga) * m1 - kg1 * singa * e1
                 - cross3(e1, cross3(cross3(e2, g1 - singa*c1), e1)) / cosga) / (rho1 * cosga)
        Pt1D2 = ((-2 * tau1 * singa - kg1 * cosga) * m2 + (g1@na) * e2
                 + cross3(e2, cross3(cross3(e1, g1-singa*c1), e2)) / cosga) / (rho2 * cosga)

        Pt2D1 = ((2 * tau2 * singa + kg2 * cosga) * m1 + (g2@na) * e1
                 - cross3(e1, cross3(cross3(e2, g2 - singa*c2), e1)) / cosga) / (rho1 * cosga)
        Pt2D2 = ((-2 * tau2 * singa - kg2 * cosga) * m2 - kg2 * singa * e2
                 + cross3(e2, cross3(cross3(e1, g2-singa*c2), e2)) / cosga) / (rho2 * cosga)

        Pt3D3 = ((2 * tau3 * singb + kg3 * cosgb) * m3 - kg3 * singb * e3
                 - cross3(e3, cross3(cross3(e4, g3 - singb*c3), e3)) / cosgb) / (rho3 * cosgb)
        Pt3D4 = ((-2 * tau3 * singb - kg3 * cosgb) * m4 + (g3@nb) * e4
                 + cross3(e4, cross3(cross3(e3, g3-singb*c3), e4)) / cosgb) / (rho4 * cosgb)

        Pt4D3 = ((2 * tau4 * singb + kg4 * cosgb) * m3 + (g4@nb) * e3
                 - cross3(e3, cross3(cross3(e4, g4 - singb*c4), e3)) / cosgb) / (rho3 * cosgb)
        Pt4D4 = ((-2 * tau4 * singb - kg4 * cosgb) * m4 - kg4 * singb * e4
                 + cross3(e4, cross3(cross3(e3, g4-singb*c4), e4)) / cosgb) / (rho4 * cosgb)

        Pt = self.Kt * (np.outer(tau1*Pt1D1 + tau2*Pt2D1, self.D1)
                        + np.outer(tau1*Pt1D2 + tau2*Pt2D2, self.D2)
                        + np.outer(tau3*Pt3D3 + tau4*Pt4D3, self.D3)
                        + np.outer(tau3*Pt3D4 + tau4*Pt4D4, self.D4))

        return Pe + Psa + Psb + Pc + Pn + Pg + Pt

    def P_F(self, F, G=G0):
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
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        tau1 = - na @ (g1 - singa * c1) / cosga
        tau2 = - na @ (g2 - singa * c2) / cosga

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
        tangb = singb / cosgb
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4
        g3 = np.einsum('ijk,j,k->i', G, self.D4, self.D3) / rho4
        g4 = np.einsum('ijk,j,k->i', G, self.D3, self.D4) / rho3

        tau3 = - nb @ (g3 - singb * c3) / cosgb
        tau4 = - nb @ (g4 - singb * c4) / cosgb

        singc = na @ nb
        cosgc = norm3(cross3(na, nb))
        tangc = singc / cosgc

        rho1_F = np.outer(e1, self.D1)
        rho2_F = np.outer(e2, self.D2)
        rho3_F = np.outer(e3, self.D3)
        rho4_F = np.outer(e4, self.D4)

        e1_F = np.einsum('ij,k->ijk', self.I3, self.D1) / rho1 \
            - np.einsum('i,j,k->ijk', e1,  e1, self.D1) / rho1
        e2_F = np.einsum('ij,k->ijk', self.I3, self.D2) / rho2 \
            - np.einsum('i,j,k->ijk', e2,  e2, self.D2) / rho2
        e3_F = np.einsum('ij,k->ijk', self.I3, self.D3) / rho3 \
            - np.einsum('i,j,k->ijk', e3,  e3, self.D3) / rho3
        e4_F = np.einsum('ij,k->ijk', self.I3, self.D4) / rho4 \
            - np.einsum('i,j,k->ijk', e4,  e4, self.D4) / rho4

        # e_F = e1_F, e2_F, e3_F, e4_F

        rho1_F_F = np.einsum('ijk,l->iljk', e1_F, self.D1)
        rho2_F_F = np.einsum('ijk,l->iljk', e2_F, self.D2)
        rho3_F_F = np.einsum('ijk,l->iljk', e3_F, self.D3)
        rho4_F_F = np.einsum('ijk,l->iljk', e4_F, self.D4)

        e1_F_F = - (np.einsum('ijk,lm->ijklm', e1_F, rho1_F)
                    + np.einsum('ijk,lm->ilmjk', e1_F,
                                rho1_F)
                    + np.einsum('i,jklm->ijklm', e1, rho1_F_F)
                    ) / rho1
        e2_F_F = - (np.einsum('ijk,lm->ijklm', e2_F, rho2_F)
                    + np.einsum('ijk,lm->ilmjk', e2_F,
                                rho2_F)
                    + np.einsum('i,jklm->ijklm', e2, rho2_F_F)
                    ) / rho2
        e3_F_F = - (np.einsum('ijk,lm->ijklm', e3_F, rho3_F)
                    + np.einsum('ijk,lm->ilmjk', e3_F,
                                rho3_F)
                    + np.einsum('i,jklm->ijklm', e3, rho3_F_F)
                    ) / rho3
        e4_F_F = - (np.einsum('ijk,lm->ijklm', e4_F, rho4_F)
                    + np.einsum('ijk,lm->ilmjk', e4_F,
                                rho4_F)
                    + np.einsum('i,jklm->ijklm', e4, rho4_F_F)
                    ) / rho4

        ga_F = np.outer(m1, self.D1) / rho1 - \
            np.outer(m2, self.D2) / rho2

        gb_F = np.outer(m3, self.D3) / rho3 - \
            np.outer(m4, self.D4) / rho4

        m1_F = np.einsum('i,jk->ijk', tanga * m1 - e1, ga_F) \
            + e2_F / cosga - tanga * e1_F
        m2_F = np.einsum('i,jk->ijk', tanga * m2 + e2, ga_F) \
            - e1_F / cosga + tanga * e2_F
        m3_F = np.einsum('i,jk->ijk', tangb * m3 - e3, gb_F) \
            + e4_F / cosgb - tangb * e3_F
        m4_F = np.einsum('i,jk->ijk', tangb * m4 + e4, gb_F) \
            - e3_F / cosgb + tangb * e4_F

        ga_F_F = -np.einsum('i,j,kl->ijkl', m1, self.D1, rho1_F) / rho1**2 \
            + np.einsum('ikl,j->ijkl', m1_F, self.D1) / rho1 \
            + np.einsum('i,j,kl->ijkl', m2, self.D2, rho2_F) / rho2**2 \
            - np.einsum('ikl,j->ijkl', m2_F, self.D2) / rho2

        gb_F_F = -np.einsum('i,j,kl->ijkl', m3, self.D3, rho3_F) / rho3**2 \
            + np.einsum('ikl,j->ijkl', m3_F, self.D3) / rho3 \
            + np.einsum('i,j,kl->ijkl', m4, self.D4, rho4_F) / rho4**2 \
            - np.einsum('ikl,j->ijkl', m4_F, self.D4) / rho4

        m1_F_F = np.einsum('ilm,jk->ijklm', np.einsum('i,lm->ilm', m1 / cosga**2, ga_F) + tanga * m1_F - e1_F, ga_F) \
            + np.einsum('i,jklm', tanga * m1 - e1, ga_F_F) \
            + np.einsum('ijk,lm->ijklm', (tanga * e2_F - e1_F / cosga) / cosga, ga_F) \
            + e2_F_F / cosga - tanga * e1_F_F
        m2_F_F = np.einsum('ilm,jk->ijklm', np.einsum('i,lm->ilm', m2 / cosga**2, ga_F) + tanga * m2_F + e2_F, ga_F) \
            + np.einsum('i,jklm', tanga * m2 + e2, ga_F_F) \
            - np.einsum('ijk,lm->ijklm', (tanga * e1_F - e2_F / cosga) / cosga, ga_F) \
            - e1_F_F / cosga + tanga * e2_F_F
        m3_F_F = np.einsum('ilm,jk->ijklm', np.einsum('i,lm->ilm', m3 / cosgb**2, gb_F) + tangb * m3_F - e3_F, gb_F) \
            + np.einsum('i,jklm', tangb * m3 - e3, gb_F_F) \
            + np.einsum('ijk,lm->ijklm', (tangb * e4_F - e3_F / cosgb) / cosgb, gb_F) \
            + e4_F_F / cosgb - tangb * e3_F_F
        m4_F_F = np.einsum('ilm,jk->ijklm', np.einsum('i,lm->ilm', m4 / cosgb**2, gb_F) + tangb * m4_F + e4_F, gb_F) \
            + np.einsum('i,jklm', tangb * m4 + e4, gb_F_F) \
            - np.einsum('ijk,lm->ijklm', (tangb * e3_F - e4_F / cosgb) / cosgb, gb_F) \
            - e3_F_F / cosgb + tangb * e4_F_F

        na_F = np.einsum('i,jA->ijA', tanga * na, ga_F) \
            + np.cross(e1_F, e2 / cosga, axis=0)  \
            - np.cross(e2_F, e1 / cosga, axis=0)
        nb_F = np.einsum('i,jA->ijA', tangb * nb, gb_F) \
            + np.cross(e3_F, e4 / cosgb, axis=0)  \
            - np.cross(e4_F, e3 / cosgb, axis=0)

        na_F_F = np.einsum('i,jk,lm', na / cosga**2, ga_F, ga_F)  \
            + np.einsum('i,jklm->ijklm', tanga * na, ga_F_F) \
            + np.einsum('ilm,jk->ijklm', na_F, ga_F * tanga) \
            + np.einsum('ijk,lm->ijklm', np.cross(e1_F, e2 / cosga, axis=0)
                        - np.cross(e2_F, e1 / cosga, axis=0), tanga * ga_F) \
            + np.cross(e1_F_F, e2 / cosga, axis=0) \
            + np.einsum('opi,ojk,plm->ijklm', self.lc3, e1_F, e2_F / cosga) \
            - np.cross(e2_F_F, e1 / cosga, axis=0) \
            - np.einsum('opi,ojk,plm->ijklm', self.lc3,
                        e2_F, e1_F / cosga)
        nb_F_F = np.einsum('i,jk,lm', nb / cosgb**2, gb_F, gb_F)  \
            + np.einsum('i,jklm->ijklm', tangb * nb, gb_F_F) \
            + np.einsum('ilm,jk->ijklm', nb_F, gb_F * tangb) \
            + np.einsum('ijk,lm->ijklm', np.cross(e3_F, e4 / cosgb, axis=0)
                        - np.cross(e4_F, e3 / cosgb, axis=0), tangb * gb_F) \
            + np.cross(e3_F_F, e4 / cosgb, axis=0) \
            + np.einsum('opi,ojk,plm->ijklm', self.lc3, e3_F, e4_F / cosgb) \
            - np.cross(e4_F_F, e3 / cosgb, axis=0) \
            - np.einsum('opi,ojk,plm->ijklm', self.lc3,
                        e4_F, e3_F / cosgb)

        c1_F = - np.einsum('i,j,A->ijA', c1, e1, self.D1) / rho1
        c2_F = - np.einsum('i,j,A->ijA', c2, e2, self.D2) / rho2
        c3_F = - np.einsum('i,j,A->ijA', c3, e3, self.D3) / rho3
        c4_F = - np.einsum('i,j,A->ijA', c4, e4, self.D4) / rho4

        g1_F = - np.einsum('i,j,A->ijA', g1, e2, self.D2) / rho2
        g2_F = - np.einsum('i,j,A->ijA', g2, e1, self.D1) / rho1
        g3_F = - np.einsum('i,j,A->ijA', g3, e4, self.D4) / rho4
        g4_F = - np.einsum('i,j,A->ijA', g4, e3, self.D3) / rho3

        c1_F_F = np.einsum('i,j,k,lm->ijklm', c1, e1 / rho1**2, self.D1, rho1_F) \
            - np.einsum('ilm, j, k->ijklm', c1_F, e1, self.D1 / rho1) \
            - np.einsum('i,jlm,k->ijklm', c1, e1_F, self.D1 / rho1)
        c2_F_F = np.einsum('i,j,k,lm->ijklm', c2, e2 / rho2**2, self.D2, rho2_F) \
            - np.einsum('ilm, j, k->ijklm', c2_F, e2, self.D2 / rho2) \
            - np.einsum('i,jlm,k->ijklm', c2, e2_F, self.D2 / rho2)
        c3_F_F = np.einsum('i,j,k,lm->ijklm', c3, e3 / rho3**2, self.D3, rho3_F) \
            - np.einsum('ilm, j, k->ijklm', c3_F, e3, self.D3 / rho3) \
            - np.einsum('i,jlm,k->ijklm', c3, e3_F, self.D3 / rho3)
        c4_F_F = np.einsum('i,j,k,lm->ijklm', c4, e4 / rho4**2, self.D4, rho4_F) \
            - np.einsum('ilm, j, k->ijklm', c4_F, e4, self.D4 / rho4) \
            - np.einsum('i,jlm,k->ijklm', c4, e4_F, self.D4 / rho4)

        g1_F_F = np.einsum('i,j,k,lm->ijklm', g1, e2 / rho2**2, self.D2, rho2_F) \
            - np.einsum('ilm, j, k->ijklm', g1_F, e2, self.D2 / rho2) \
            - np.einsum('i,jlm,k->ijklm', g1, e2_F, self.D2 / rho2)
        g2_F_F = np.einsum('i,j,k,lm->ijklm', g2, e1 / rho1**2, self.D1, rho1_F) \
            - np.einsum('ilm, j, k->ijklm', g2_F, e1, self.D1 / rho1) \
            - np.einsum('i,jlm,k->ijklm', g2, e1_F, self.D1 / rho1)
        g3_F_F = np.einsum('i,j,k,lm->ijklm', g3, e4 / rho4**2, self.D4, rho4_F) \
            - np.einsum('ilm, j, k->ijklm', g3_F, e4, self.D4 / rho4) \
            - np.einsum('i,jlm,k->ijklm', g3, e4_F, self.D4 / rho4)
        g4_F_F = np.einsum('i,j,k,lm->ijklm', g4, e3 / rho3**2, self.D3, rho3_F) \
            - np.einsum('ilm, j, k->ijklm', g4_F, e3, self.D3 / rho3) \
            - np.einsum('i,jlm,k->ijklm', g4, e3_F, self.D3 / rho3)

        # Elongation
        Pe1_F = np.einsum('i,j,k,l->ijkl', self.Ke * e1, self.D1, e1, self.D1) \
            + np.einsum('i,jkl->jikl', self.Ke * self.D1 * (rho1-1), e1_F)
        Pe2_F = np.einsum('i,j,k,l->ijkl', self.Ke * e2, self.D2, e2, self.D2) \
            + np.einsum('i,jkl->jikl', self.Ke * self.D2 * (rho2-1), e2_F)
        Pe3_F = np.einsum('i,j,k,l->ijkl', self.Ke * e3, self.D3, e3, self.D3) \
            + np.einsum('i,jkl->jikl', self.Ke * self.D3 * (rho3-1), e3_F)
        Pe4_F = np.einsum('i,j,k,l->ijkl', self.Ke * e4, self.D4, e4, self.D4) \
            + np.einsum('i,jkl->jikl', self.Ke * self.D4 * (rho4-1), e4_F)

        Pe_F = Pe1_F + Pe2_F + Pe3_F + Pe4_F

        # shear
        Psa_F = self.Ks * (np.arcsin(singa) * ga_F_F + \
            np.einsum('jA,kB->jAkB',
                      ga_F, ga_F))
        Psb_F = self.Ks * (np.arcsin(singb) * gb_F_F + \
            np.einsum('jA,kB->jAkB',
                      gb_F, gb_F))

        gc_F = np.einsum('i,ijk->jk', na / cosgc, nb_F) + \
            np.einsum('i,ijk->jk', nb / cosgc, na_F)
        gc_F_F = np.einsum('i,ijAkB->jAkB', nb / cosgc, na_F_F) \
            + np.einsum('i,ijAkB->jAkB', na / cosgc, nb_F_F) \
            + np.einsum('ijA,ikB->jAkB', na_F / cosgc, nb_F) \
            + np.einsum('ijA,ikB->jAkB', nb_F / cosgc, na_F) \
            + np.einsum('i,ijA,kB->jAkB', na * tangc / cosgc, nb_F, gc_F) \
            + np.einsum('i,ijA,kB->jAkB', nb * tangc / cosgc, na_F, gc_F) \

        Pc_F = self.Kc * (np.arcsin(singc) * gc_F_F + \
            np.einsum('jA,kB->jAkB',  gc_F, gc_F))

        # normal bending
        Pn1_F = self.Kn * (np.einsum('jk,lm->jklm',
                                     (np.einsum('ijk,i->jk', m1_F, c1) +
                                      np.einsum('ijk,i->jk', c1_F, m1)),
                                     (np.einsum('ijk,i->jk', m1_F, c1) + np.einsum('ijk,i->jk', c1_F, m1)))
                           + (m1 @ c1) * (np.einsum('ijklm,i->jklm', m1_F_F, c1)
                                          + np.einsum('ijk,ilm->jklm', m1_F, c1_F) + np.einsum('ilm,ijk->jklm', m1_F, c1_F)
                                          + np.einsum('i,ijklm->jklm', m1, c1_F_F)))
        Pn2_F = self.Kn * (np.einsum('jk,lm->jklm',
                                     (np.einsum('ijk,i->jk', m2_F, c2) +
                                      np.einsum('ijk,i->jk', c2_F, m2)),
                                     (np.einsum('ijk,i->jk', m2_F, c2) + np.einsum('ijk,i->jk', c2_F, m2)))
                           + (m2 @ c2) * (np.einsum('ijklm,i->jklm', m2_F_F, c2)
                                          + np.einsum('ijk,ilm->jklm', m2_F, c2_F) + np.einsum('ilm,ijk->jklm', m2_F, c2_F)
                                          + np.einsum('i,ijklm->jklm', m2, c2_F_F)))
        Pn3_F = self.Kn * (np.einsum('jk,lm->jklm',
                                     (np.einsum('ijk,i->jk', m3_F, c3) +
                                      np.einsum('ijk,i->jk', c3_F, m3)),
                                     (np.einsum('ijk,i->jk', m3_F, c3) + np.einsum('ijk,i->jk', c3_F, m3)))
                           + (m3 @ c3) * (np.einsum('ijklm,i->jklm', m3_F_F, c3)
                                          + np.einsum('ijk,ilm->jklm', m3_F, c3_F) + np.einsum('ilm,ijk->jklm', m3_F, c3_F)
                                          + np.einsum('i,ijklm->jklm', m3, c3_F_F)))
        Pn4_F = self.Kn * (np.einsum('jk,lm->jklm',
                                     (np.einsum('ijk,i->jk', m4_F, c4) +
                                      np.einsum('ijk,i->jk', c4_F, m4)),
                                     (np.einsum('ijk,i->jk', m4_F, c4) + np.einsum('ijk,i->jk', c4_F, m4)))
                           + (m4 @ c4) * (np.einsum('ijklm,i->jklm', m4_F_F, c4)
                                          + np.einsum('ijk,ilm->jklm', m4_F, c4_F) + np.einsum('ilm,ijk->jklm', m4_F, c4_F)
                                          + np.einsum('i,ijklm->jklm', m4, c4_F_F)))

        Pn_F = Pn1_F + Pn2_F + Pn3_F + Pn4_F

        Pg1_F = self.Kg * (np.einsum('jk,lm->jklm',
                                     (np.einsum('ijk,i->jk', na_F, c1) +
                                      np.einsum('ijk,i->jk', c1_F, na)),
                                     (np.einsum('ijk,i->jk', na_F, c1) + np.einsum('ijk,i->jk', c1_F, na)))
                           + (na @ c1) * (np.einsum('ijklm,i->jklm', na_F_F, c1)
                                          + np.einsum('ijk,ilm->jklm', na_F, c1_F) + np.einsum('ilm,ijk->jklm', na_F, c1_F)
                                          + np.einsum('i,ijklm->jklm', na, c1_F_F)))
        Pg2_F = self.Kg * (np.einsum('jk,lm->jklm',
                                     (np.einsum('ijk,i->jk', na_F, c2) +
                                      np.einsum('ijk,i->jk', c2_F, na)),
                                     (np.einsum('ijk,i->jk', na_F, c2) + np.einsum('ijk,i->jk', c2_F, na)))
                           + (na @ c2) * (np.einsum('ijklm,i->jklm', na_F_F, c2)
                                          + np.einsum('ijk,ilm->jklm', na_F, c2_F) + np.einsum('ilm,ijk->jklm', na_F, c2_F)
                                          + np.einsum('i,ijklm->jklm', na, c2_F_F)))
        Pg3_F = self.Kg * (np.einsum('jk,lm->jklm',
                                     (np.einsum('ijk,i->jk', nb_F, c3) +
                                      np.einsum('ijk,i->jk', c3_F, nb)),
                                     (np.einsum('ijk,i->jk', nb_F, c3) + np.einsum('ijk,i->jk', c3_F, nb)))
                           + (nb @ c3) * (np.einsum('ijklm,i->jklm', nb_F_F, c3)
                                          + np.einsum('ijk,ilm->jklm', nb_F, c3_F) + np.einsum('ilm,ijk->jklm', nb_F, c3_F)
                                          + np.einsum('i,ijklm->jklm', nb, c3_F_F)))
        Pg4_F = self.Kg * (np.einsum('jk,lm->jklm',
                                     (np.einsum('ijk,i->jk', nb_F, c4) +
                                      np.einsum('ijk,i->jk', c4_F, nb)),
                                     (np.einsum('ijk,i->jk', nb_F, c4) + np.einsum('ijk,i->jk', c4_F, nb)))
                           + (nb @ c4) * (np.einsum('ijklm,i->jklm', nb_F_F, c4)
                                          + np.einsum('ijk,ilm->jklm', nb_F, c4_F) + np.einsum('ilm,ijk->jklm', nb_F, c4_F)
                                          + np.einsum('i,ijklm->jklm', nb, c4_F_F)))

        Pg_F = Pg1_F + Pg2_F + Pg3_F + Pg4_F

        tau1_F = ga_F * (tau1 * tanga + na @ c1) + np.einsum('ijk,i->jk', - na_F, g1 / cosga - tanga * c1)\
            + np.einsum('i,ijk->jk', - na, g1_F / cosga - tanga * c1_F)
        tau2_F = ga_F * (tau2 * tanga + na @ c2) + np.einsum('ijk,i->jk', - na_F, (g2 - singa * c2) / cosga)\
            + np.einsum('i,ijk->jk', - na / cosga, g2_F) + \
            np.einsum('i,ijk->jk', tanga * na, c2_F)
        tau3_F = gb_F * (tau3 * tangb + nb @ c3) + np.einsum('ijk,i->jk', - nb_F, (g3 - singb * c3) / cosgb)\
            + np.einsum('i,ijk->jk', - nb / cosgb, g3_F) + \
            np.einsum('i,ijk->jk', tangb * nb, c3_F)
        tau4_F = gb_F * (tau4 * tangb + nb @ c4) + np.einsum('ijk,i->jk', - nb_F, (g4 - singb * c4) / cosgb)\
            + np.einsum('i,ijk->jk', - nb / cosgb, g4_F) + \
            np.einsum('i,ijk->jk', tangb * nb, c4_F)

        tau1_F_F = ga_F_F * (tau1 * tanga + na @ c1) \
            + np.einsum('jA,kB->jAkB', ga_F, tau1_F * tanga + (tau1 / cosga**2) * ga_F
                        + np.einsum('ikB,i->kB', na_F, c1) + np.einsum('ikB,i->kB', c1_F, na)) \
            + np.einsum('ijAkB,i->jAkB', - na_F_F, g1 / cosga - tanga * c1) \
            + np.einsum('ijA,i,kB->jAkB', - na_F, (g1 * (tanga / cosga) - c1 / cosga**2), ga_F) \
            + np.einsum('ijA,ikB->jAkB', - na_F, g1_F / cosga - tanga * c1_F) \
            + np.einsum('ikB,ijA->jAkB', - na_F, g1_F / cosga - tanga * c1_F) \
            + np.einsum('i,ijA,kB->jAkB', - na, g1_F * (tanga / cosga) - c1_F / cosga**2, ga_F) \
            + np.einsum('i,ijAkB->jAkB', - na, g1_F_F / cosga - tanga * c1_F_F)
        tau2_F_F = ga_F_F * (tau2 * tanga + na @ c2) \
            + np.einsum('jA,kB->jAkB', ga_F, tau2_F * tanga + (tau2 / cosga**2) * ga_F
                        + np.einsum('ikB,i->kB', na_F, c2) + np.einsum('ikB,i->kB', c2_F, na)) \
            + np.einsum('ijAkB,i->jAkB', - na_F_F, g2 / cosga - tanga * c2) \
            + np.einsum('ijA,i,kB->jAkB', - na_F, (g2 * (tanga / cosga) - c2 / cosga**2), ga_F) \
            + np.einsum('ijA,ikB->jAkB', - na_F, g2_F / cosga - tanga * c2_F) \
            + np.einsum('ikB,ijA->jAkB', - na_F, g2_F / cosga - tanga * c2_F) \
            + np.einsum('i,ijA,kB->jAkB', - na, g2_F * (tanga / cosga) - c2_F / cosga**2, ga_F) \
            + np.einsum('i,ijAkB->jAkB', - na, g2_F_F / cosga - tanga * c2_F_F)
        tau3_F_F = gb_F_F * (tau3 * tangb + nb @ c3) \
            + np.einsum('jA,kB->jAkB', gb_F, tau3_F * tangb + (tau3 / cosgb**2) * gb_F
                        + np.einsum('ikB,i->kB', nb_F, c3) + np.einsum('ikB,i->kB', c3_F, nb)) \
            + np.einsum('ijAkB,i->jAkB', - nb_F_F, g3 / cosgb - tangb * c3) \
            + np.einsum('ijA,i,kB->jAkB', - nb_F, (g3 * (tangb / cosgb) - c3 / cosgb**2), gb_F) \
            + np.einsum('ijA,ikB->jAkB', - nb_F, g3_F / cosgb - tangb * c3_F) \
            + np.einsum('ikB,ijA->jAkB', - nb_F, g3_F / cosgb - tangb * c3_F) \
            + np.einsum('i,ijA,kB->jAkB', - nb, g3_F * (tangb / cosgb) - c3_F / cosgb**2, gb_F) \
            + np.einsum('i,ijAkB->jAkB', - nb, g3_F_F / cosgb - tangb * c3_F_F)
        tau4_F_F = gb_F_F * (tau4 * tangb + nb @ c4) \
            + np.einsum('jA,kB->jAkB', gb_F, tau4_F * tangb + (tau4 / cosgb**2) * gb_F
                        + np.einsum('ikB,i->kB', nb_F, c4) + np.einsum('ikB,i->kB', c4_F, nb)) \
            + np.einsum('ijAkB,i->jAkB', - nb_F_F, g4 / cosgb - tangb * c4) \
            + np.einsum('ijA,i,kB->jAkB', - nb_F, (g4 * (tangb / cosgb) - c4 / cosgb**2), gb_F) \
            + np.einsum('ijA,ikB->jAkB', - nb_F, g4_F / cosgb - tangb * c4_F) \
            + np.einsum('ikB,ijA->jAkB', - nb_F, g4_F / cosgb - tangb * c4_F) \
            + np.einsum('i,ijA,kB->jAkB', - nb, g4_F * (tangb / cosgb) - c4_F / cosgb**2, gb_F) \
            + np.einsum('i,ijAkB->jAkB', - nb, g4_F_F / cosgb - tangb * c4_F_F)

        Pt1_F = np.einsum('iA,kB->iAkB', self.Kt * tau1_F,
                          tau1_F) + self.Kt * tau1 * tau1_F_F
        Pt2_F = np.einsum('iA,kB->iAkB', self.Kt * tau2_F,
                          tau2_F) + self.Kt * tau2 * tau2_F_F
        Pt3_F = np.einsum('iA,kB->iAkB', self.Kt * tau3_F,
                          tau3_F) + self.Kt * tau3 * tau3_F_F
        Pt4_F = np.einsum('iA,kB->iAkB', self.Kt * tau4_F,
                          tau4_F) + self.Kt * tau4 * tau4_F_F

        Pt_F = Pt1_F + Pt2_F + Pt3_F + Pt4_F
        # num = Numerical_derivative(\
        #     lambda F: self.na_F(F, G), order=2)._X(F)
        # print(np.linalg.norm(num - na_F_F))
        # print('a')

        # [0, 0, 0]num2 = Numererical_derivative(lambda F :Numerical_derivative(\
        #     lambda F: self.na_F(F, G), order=2)._X(F), order=2)._X(F)
        # print(np.linalg.norm(num2 - tau1_F_F))
        # print('a')

        return Pe_F + Psa_F + Psb_F + Pc_F + Pg_F + Pn_F + Pt_F

    def P_G(self, F, G=G0):

        # layer a
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        # layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        singb = e3 @ e4
        tangb = singb / cosgb
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4
        g3 = np.einsum('ijk,j,k->i', G, self.D4, self.D3) / rho4
        g4 = np.einsum('ijk,j,k->i', G, self.D3, self.D4) / rho3

        singc = na @ nb
        cosgc = norm(cross3(na, nb))

        rho1_F = np.outer(e1, self.D1)
        rho2_F = np.outer(e2, self.D2)
        rho3_F = np.outer(e3, self.D3)
        rho4_F = np.outer(e4, self.D4)

        e1_F = np.einsum('ij,k->ijk', self.I3, self.D1) / rho1 \
            - np.einsum('i,j,k->ijk', e1,  e1, self.D1) / rho1
        e2_F = np.einsum('ij,k->ijk', self.I3, self.D2) / rho2 \
            - np.einsum('i,j,k->ijk', e2,  e2, self.D2) / rho2
        e3_F = np.einsum('ij,k->ijk', self.I3, self.D3) / rho3 \
            - np.einsum('i,j,k->ijk', e3,  e3, self.D3) / rho3
        e4_F = np.einsum('ij,k->ijk', self.I3, self.D4) / rho4 \
            - np.einsum('i,j,k->ijk', e4,  e4, self.D4) / rho4

        e_F = e1_F, e2_F, e3_F, e4_F

        ga_F = np.outer(m1, self.D1) / rho1 - \
            np.outer(m2, self.D2) / rho2

        gb_F = np.outer(m3, self.D3) / rho3 - \
            np.outer(m4, self.D4) / rho4

        m1_F = np.einsum('i,jk->ijk', tanga * m1 - e1, ga_F) \
            + e2_F / cosga - tanga * e1_F
        m2_F = np.einsum('i,jk->ijk', tanga * m2 + e2, ga_F) \
            - e1_F / cosga + tanga * e2_F
        m3_F = np.einsum('i,jk->ijk', tangb * m3 - e3, gb_F) \
            + e4_F / cosgb - tangb * e3_F
        m4_F = np.einsum('i,jk->ijk', tangb * m4 + e4, gb_F) \
            - e3_F / cosgb + tangb * e4_F

        na_F = np.einsum('i,jk->ijk', tanga * na, ga_F) \
            + np.cross(e1_F, e2 / cosga, axis=0)  \
            - np.cross(e2_F, e1 / cosga, axis=0)
        nb_F = np.einsum('i,jk->ijk', tangb * nb, gb_F) \
            + np.cross(e3_F, e4 / cosgb, axis=0)  \
            - np.cross(e4_F, e3 / cosgb, axis=0)

        c1_F = - np.einsum('i,j,k', c1, e1, self.D1) / rho1
        c2_F = - np.einsum('i,j,k', c2, e2, self.D2) / rho2
        c3_F = - np.einsum('i,j,k', c3, e3, self.D3) / rho3
        c4_F = - np.einsum('i,j,k', c4, e4, self.D4) / rho4

        c1_G = np.einsum('ij,A,B->ijAB', self.I3, self.D1, self.D1 / rho1)
        c2_G = np.einsum('ij,A,B->ijAB', self.I3, self.D2, self.D2 / rho2)
        c3_G = np.einsum('ij,A,B->ijAB', self.I3, self.D3, self.D3 / rho3)
        c4_G = np.einsum('ij,A,B->ijAB', self.I3, self.D4, self.D4 / rho4)

        c1_F_G = - np.einsum('ik,j,A,B,C->ijAkBC', self.I3,
                             e1, self.D1, self.D1, self.D1 / rho1**2)
        c2_F_G = - np.einsum('ik,j,A,B,C->ijAkBC', self.I3,
                             e2, self.D2, self.D2, self.D2 / rho2**2)
        c3_F_G = - np.einsum('ik,j,A,B,C->ijAkBC', self.I3,
                             e3, self.D3, self.D3, self.D3 / rho3**2)
        c4_F_G = - np.einsum('ik,j,A,B,C->ijAkBC', self.I3,
                             e4, self.D4, self.D4, self.D4 / rho4**2)

        g1_F = - np.einsum('i,j,k', g1, e2, self.D2) / rho2
        g2_F = - np.einsum('i,j,k', g2, e1, self.D1) / rho1
        g3_F = - np.einsum('i,j,k', g3, e4, self.D4) / rho4
        g4_F = - np.einsum('i,j,k', g4, e3, self.D3) / rho3

        g1_G = np.einsum('ij,A,B->ijAB', self.I3, self.D2, self.D1 / rho2)
        g2_G = np.einsum('ij,A,B->ijAB', self.I3, self.D1, self.D2 / rho1)
        g3_G = np.einsum('ij,A,B->ijAB', self.I3, self.D4, self.D3 / rho4)
        g4_G = np.einsum('ij,A,B->ijAB', self.I3, self.D3, self.D4 / rho3)

        g1_F_G = - np.einsum('ik,j,A,B,C->ijAkBC', self.I3,
                             e2, self.D2, self.D2, self.D1 / rho2**2)
        g2_F_G = - np.einsum('ik,j,A,B,C->ijAkBC', self.I3,
                             e1, self.D1, self.D1, self.D2 / rho1**2)
        g3_F_G = - np.einsum('ik,j,A,B,C->ijAkBC', self.I3,
                             e4, self.D4, self.D4, self.D3 / rho4**2)
        g4_F_G = - np.einsum('ik,j,A,B,C->ijAkBC', self.I3,
                             e3, self.D3, self.D3, self.D4 / rho3**2)

        kn1_F = np.einsum('ijA,i->jA', m1_F, c1) + \
            np.einsum('ijA,i->jA', c1_F, m1)
        kn2_F = np.einsum('ijA,i->jA', m2_F, c2) + \
            np.einsum('ijA,i->jA', c2_F, m2)
        kn3_F = np.einsum('ijA,i->jA', m3_F, c3) + \
            np.einsum('ijA,i->jA', c3_F, m3)
        kn4_F = np.einsum('ijA,i->jA', m4_F, c4) + \
            np.einsum('ijA,i->jA', c4_F, m4)

        kn1_F_G = np.einsum('ijA,ikBC->jAkBC', m1_F, c1_G) + \
            np.einsum('ijAkBC,i->jAkBC', c1_F_G, m1)
        kn2_F_G = np.einsum('ijA,ikBC->jAkBC', m2_F, c2_G) + \
            np.einsum('ijAkBC,i->jAkBC', c2_F_G, m2)
        kn3_F_G = np.einsum('ijA,ikBC->jAkBC', m3_F, c3_G) + \
            np.einsum('ijAkBC,i->jAkBC', c3_F_G, m3)
        kn4_F_G = np.einsum('ijA,ikBC->jAkBC', m4_F, c4_G) + \
            np.einsum('ijAkBC,i->jAkBC', c4_F_G, m4)

        Pn1_G = np.einsum('i,ikBC,jA->jAkBC', self.Kn * m1, c1_G, kn1_F) \
            + self.Kn * m1 @ c1 * kn1_F_G
        Pn2_G = np.einsum('i,ikBC,jA->jAkBC', self.Kn * m2, c2_G, kn2_F) \
            + self.Kn * m2 @ c2 * kn2_F_G
        Pn3_G = np.einsum('i,ikBC,jA->jAkBC', self.Kn * m3, c3_G, kn3_F) \
            + self.Kn * m3 @ c3 * kn3_F_G
        Pn4_G = np.einsum('i,ikBC,jA->jAkBC', self.Kn * m4, c4_G, kn4_F) \
            + self.Kn * m4 @ c4 * kn4_F_G

        Pn_G = Pn1_G + Pn2_G + Pn3_G + Pn4_G

        kg1_F = np.einsum('ijA,i->jA', na_F, c1) + \
            np.einsum('ijA,i->jA', c1_F, na)
        kg2_F = np.einsum('ijA,i->jA', na_F, c2) + \
            np.einsum('ijA,i->jA', c2_F, na)
        kg3_F = np.einsum('ijA,i->jA', nb_F, c3) + \
            np.einsum('ijA,i->jA', c3_F, nb)
        kg4_F = np.einsum('ijA,i->jA', nb_F, c4) + \
            np.einsum('ijA,i->jA', c4_F, nb)

        kg1_F_G = np.einsum('ijA,ikBC->jAkBC', na_F, c1_G) + \
            np.einsum('ijAkBC,i->jAkBC', c1_F_G, na)
        kg2_F_G = np.einsum('ijA,ikBC->jAkBC', na_F, c2_G) + \
            np.einsum('ijAkBC,i->jAkBC', c2_F_G, na)
        kg3_F_G = np.einsum('ijA,ikBC->jAkBC', nb_F, c3_G) + \
            np.einsum('ijAkBC,i->jAkBC', c3_F_G, nb)
        kg4_F_G = np.einsum('ijA,ikBC->jAkBC', nb_F, c4_G) + \
            np.einsum('ijAkBC,i->jAkBC', c4_F_G, nb)

        Pg1_G = np.einsum('i,ikBC,jA->jAkBC', self.Kg * na, c1_G, kg1_F) \
            + self.Kg * na @ c1 * kg1_F_G
        Pg2_G = np.einsum('i,ikBC,jA->jAkBC', self.Kg * na, c2_G, kg2_F) \
            + self.Kg * na @ c2 * kg2_F_G
        Pg3_G = np.einsum('i,ikBC,jA->jAkBC', self.Kg * nb, c3_G, kg3_F) \
            + self.Kg * nb @ c3 * kg3_F_G
        Pg4_G = np.einsum('i,ikBC,jA->jAkBC', self.Kg * nb, c4_G, kg4_F) \
            + self.Kg * nb @ c4 * kg4_F_G

        Pg_G = Pg1_G + Pg2_G + Pg3_G + Pg4_G

        tau1 = - na @ (g1 - singa * c1) / cosga
        tau2 = - na @ (g2 - singa * c2) / cosga
        tau3 = - nb @ (g3 - singb * c3) / cosgb
        tau4 = - nb @ (g4 - singb * c4) / cosgb

        tau1_F = ga_F * (tau1 * tanga + na @ c1) + np.einsum('ijA,i->jA', - na_F, (g1 - singa * c1) / cosga)\
            + np.einsum('i,ijA->jA', - na / cosga, g1_F) + \
            np.einsum('i,ijA->jA', tanga * na, c1_F)
        tau2_F = ga_F * (tau2 * tanga + na @ c2) + np.einsum('ijk,i->jk', - na_F, (g2 - singa * c2) / cosga)\
            + np.einsum('i,ijk->jk', - na / cosga, g2_F) + \
            np.einsum('i,ijk->jk', tanga * na, c2_F)
        tau3_F = gb_F * (tau3 * tangb + nb @ c3) + np.einsum('ijk,i->jk', - nb_F, (g3 - singb * c3) / cosgb)\
            + np.einsum('i,ijk->jk', - nb / cosgb, g3_F) + \
            np.einsum('i,ijk->jk', tangb * nb, c3_F)
        tau4_F = gb_F * (tau4 * tangb + nb @ c4) + np.einsum('ijk,i->jk', - nb_F, (g4 - singb * c4) / cosgb)\
            + np.einsum('i,ijk->jk', - nb / cosgb, g4_F) + \
            np.einsum('i,ijk->jk', tangb * nb, c4_F)

        tau1_G = np.einsum('ijAB,i->jAB', g1_G / cosga - tanga * c1_G, - na)
        tau2_G = np.einsum('ijAB,i->jAB', g2_G / cosga - tanga * c2_G, - na)
        tau3_G = np.einsum('ijAB,i->jAB', g3_G / cosgb - tangb * c3_G, - nb)
        tau4_G = np.einsum('ijAB,i->jAB', g4_G / cosgb - tangb * c4_G, - nb)

        tau1_F_G = np.einsum('jA,kBC->jAkBC', ga_F, tau1_G * tanga + np.einsum('i,ijAB->jAB', na, c1_G)) \
            + np.einsum('ijA,ikBC->jAkBC', - na_F, g1_G / cosga - tanga * c1_G) \
            + np.einsum('i,ijAkBC->jAkBC', - na / cosga, g1_F_G) \
            + np.einsum('i,ijAkBC->jAkBC', tanga * na, c1_F_G)
        tau2_F_G = np.einsum('jA,kBC->jAkBC', ga_F, tau2_G * tanga + np.einsum('i,ijAB->jAB', na, c2_G)) \
            + np.einsum('ijA,ikBC->jAkBC', - na_F, g2_G / cosga - tanga * c2_G) \
            + np.einsum('i,ijAkBC->jAkBC', - na / cosga, g2_F_G) \
            + np.einsum('i,ijAkBC->jAkBC', tanga * na, c2_F_G)
        tau3_F_G = np.einsum('jA,kBC->jAkBC', gb_F, tau3_G * tangb + np.einsum('i,ijAB->jAB', nb, c3_G)) \
            + np.einsum('ijA,ikBC->jAkBC', - nb_F, g3_G / cosgb - tangb * c3_G) \
            + np.einsum('i,ijAkBC->jAkBC', - nb / cosgb, g3_F_G) \
            + np.einsum('i,ijAkBC->jAkBC', tangb * nb, c3_F_G)
        tau4_F_G = np.einsum('jA,kBC->jAkBC', gb_F, tau4_G * tangb + np.einsum('i,ijAB->jAB', nb, c4_G)) \
            + np.einsum('ijA,ikBC->jAkBC', - nb_F, g4_G / cosgb - tangb * c4_G) \
            + np.einsum('i,ijAkBC->jAkBC', - nb / cosgb, g4_F_G) \
            + np.einsum('i,ijAkBC->jAkBC', tangb * nb, c4_F_G)

        Pt1_G = self.Kt * tau1 * tau1_F_G + \
            np.einsum('jA,kBC->jAkBC', self.Kt * tau1_F, tau1_G)
        Pt2_G = self.Kt * tau2 * tau2_F_G + \
            np.einsum('jA,kBC->jAkBC', self.Kt * tau2_F, tau2_G)
        Pt3_G = self.Kt * tau3 * tau3_F_G + \
            np.einsum('jA,kBC->jAkBC', self.Kt * tau3_F, tau3_G)
        Pt4_G = self.Kt * tau4 * tau4_F_G + \
            np.einsum('jA,kBC->jAkBC', self.Kt * tau4_F, tau4_G)

        Pt_G = Pt1_G + Pt2_G + Pt3_G + Pt4_G

        # num = Numerical_derivative(\
        # lambda G: self.g(F, G)[0], order=2)._X(G)
        # print(np.linalg.norm(num - g1_G))
        # print('a')

        # num = Numerical_derivative(\
        # lambda F: - na @ (self.g(F, G)[0] - singa * self.c(F, G)[0]) / cosga, order=2)._X(F)
        # print(np.linalg.norm(num - tau1_F))
        # print('a')

        return Pn_G + Pg_G + Pt_G

    def bbP(self, F, G):
        """Piola-Lagrange double stress tensor
        """
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        kg1 = na @ c1
        kg2 = na @ c2
        kn1 = - m1 @ c1
        kn2 = - m2 @ c2
        tau1 = - na @ (g1 - singa * c1) / cosga
        tau2 = - na @ (g2 - singa * c2) / cosga

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
        tangb = singb / cosgb
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4
        g3 = np.einsum('ijk,j,k->i', G, self.D4, self.D3) / rho4
        g4 = np.einsum('ijk,j,k->i', G, self.D3, self.D4) / rho3

        kg3 = nb @ c3
        kg4 = nb @ c4
        kn3 = - m3 @ c3
        kn4 = - m4 @ c4
        tau3 = - nb @ (g3 - singb * c3) / cosgb
        tau4 = - nb @ (g4 - singb * c4) / cosgb

        # bbP_num = Numerical_derivative(lambda G: self.W(F, G), order=num_order)._X(G)
        # normal

        # geodesic
        bbPg1 = self.Kg * kg1 / rho1 * \
            np.einsum('i,j,k->ijk', na, self.D1, self.D1)
        bbPg2 = self.Kg * kg2 / rho2 * \
            np.einsum('i,j,k->ijk', na, self.D2, self.D2)
        bbPg3 = self.Kg * kg3 / rho3 * \
            np.einsum('i,j,k->ijk', nb, self.D3, self.D3)
        bbPg4 = self.Kg * kg4 / rho4 * \
            np.einsum('i,j,k->ijk', nb, self.D4, self.D4)

        bbPg = bbPg1 + bbPg2 + bbPg3 + bbPg4

        # normal
        bbPn1 = - self.Kn * kn1 / rho1 * \
            np.einsum('i,j,k->ijk', m1, self.D1, self.D1)
        bbPn2 = - self.Kn * kn2 / rho2 * \
            np.einsum('i,j,k->ijk', m2, self.D2, self.D2)
        bbPn3 = - self.Kn * kn3 / rho3 * \
            np.einsum('i,j,k->ijk', m3, self.D3, self.D3)
        bbPn4 = - self.Kn * kn4 / rho4 * \
            np.einsum('i,j,k->ijk', m4, self.D4, self.D4)

        bbPn = bbPn1 + bbPn2 + bbPn3 + bbPn4

        # torsion
        bbPt1 = self.Kt * tau1 * (tanga * np.einsum('i,j,k->ijk', na, self.D1, self.D1) / rho1
                                  - np.einsum('i,j,k->ijk', na, self.D2, self.D1) / (rho2 * cosga))
        bbPt2 = self.Kt * tau2 * (tanga * np.einsum('i,j,k->ijk', na, self.D2, self.D2) / rho2
                                  - np.einsum('i,j,k->ijk', na, self.D1, self.D2) / (rho1 * cosga))
        bbPt3 = self.Kt * tau3 * (tangb * np.einsum('i,j,k->ijk', nb, self.D3, self.D3) / rho3
                                  - np.einsum('i,j,k->ijk', nb, self.D4, self.D3) / (rho4 * cosgb))
        bbPt4 = self.Kt * tau4 * (tangb * np.einsum('i,j,k->ijk', nb, self.D4, self.D4) / rho4
                                  - np.einsum('i,j,k->ijk', nb, self.D3, self.D4) / (rho3 * cosgb))

        bbPt = bbPt1 + bbPt2 + bbPt3 + bbPt4

        return bbPn + bbPg + bbPt

    def bbP_F(self, F, G):
        # layer a
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        # layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        singb = e3 @ e4
        tangb = singb / cosgb
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4
        g3 = np.einsum('ijk,j,k->i', G, self.D4, self.D3) / rho4
        g4 = np.einsum('ijk,j,k->i', G, self.D3, self.D4) / rho3

        rho1_F = np.outer(e1, self.D1)
        rho2_F = np.outer(e2, self.D2)
        rho3_F = np.outer(e3, self.D3)
        rho4_F = np.outer(e4, self.D4)

        e1_F = np.einsum('ij,k->ijk', self.I3, self.D1) / rho1  \
            - np.einsum('i,j,k->ijk', e1,  e1, self.D1) / rho1
        e2_F = np.einsum('ij,k->ijk', self.I3, self.D2) / rho2  \
            - np.einsum('i,j,k->ijk', e2,  e2, self.D2) / rho2
        e3_F = np.einsum('ij,k->ijk', self.I3, self.D3) / rho3  \
            - np.einsum('i,j,k->ijk', e3,  e3, self.D3) / rho3
        e4_F = np.einsum('ij,k->ijk', self.I3, self.D4) / rho4  \
            - np.einsum('i,j,k->ijk', e4,  e4, self.D4) / rho4

        e_F = e1_F, e2_F, e3_F, e4_F

        ga_F = np.outer(m1, self.D1 / rho1) - \
            np.outer(m2, self.D2 / rho2)

        gb_F = np.outer(m3, self.D3) / rho3 - \
            np.outer(m4, self.D4) / rho4

        m1_F = np.einsum('i,jk->ijk', tanga * m1 - e1, ga_F) \
            + e2_F / cosga - tanga * e1_F
        m2_F = np.einsum('i,jk->ijk', tanga * m2 + e2, ga_F) \
            - e1_F / cosga + tanga * e2_F
        m3_F = np.einsum('i,jk->ijk', tangb * m3 - e3, gb_F) \
            + e4_F / cosgb - tangb * e3_F
        m4_F = np.einsum('i,jk->ijk', tangb * m4 + e4, gb_F) \
            - e3_F / cosgb + tangb * e4_F

        na_F = np.einsum('i,jk->ijk', tanga * na, ga_F) \
            + np.cross(e1_F, e2 / cosga, axis=0)  \
            - np.cross(e2_F, e1 / cosga, axis=0)
        nb_F = np.einsum('i,jk->ijk', tangb * nb, gb_F) \
            + np.cross(e3_F, e4 / cosgb, axis=0)  \
            - np.cross(e4_F, e3 / cosgb, axis=0)

        c1_F = - np.einsum('i,j,A->ijA', c1, e1, self.D1) / rho1
        c2_F = - np.einsum('i,j,A->ijA', c2, e2, self.D2) / rho2
        c3_F = - np.einsum('i,j,A->ijA', c3, e3, self.D3) / rho3
        c4_F = - np.einsum('i,j,A->ijA', c4, e4, self.D4) / rho4
        g1_F = - np.einsum('i,j,A->ijA', g1, e2, self.D2) / rho2
        g2_F = - np.einsum('i,j,A->ijA', g2, e1, self.D1) / rho1
        g3_F = - np.einsum('i,j,A->ijA', g3, e4, self.D4) / rho4
        g4_F = - np.einsum('i,j,A->ijA', g4, e3, self.D3) / rho3

        kg1_F = np.einsum('ijA,i->jA', na_F, c1) + \
            np.einsum('ijA,i->jA', c1_F, na)
        kg2_F = np.einsum('ijA,i->jA', na_F, c2) + \
            np.einsum('ijA,i->jA', c2_F, na)
        kg3_F = np.einsum('ijA,i->jA', nb_F, c3) + \
            np.einsum('ijA,i->jA', c3_F, nb)
        kg4_F = np.einsum('ijA,i->jA', nb_F, c4) + \
            np.einsum('ijA,i->jA', c4_F, nb)

        bbPg1_F = np.einsum('kC,j,A,B->jABkC', kg1_F, na, self.D1, self.D1 * self.Kg / rho1) \
            + np.einsum('ikC,A,B->iABkC', na_F, self.D1, self.D1 * (na @ c1) * self.Kg / rho1) \
            - np.einsum('kC,j,A,B->jABkC', rho1_F, na, self.D1,
                        self.D1 * (na @ c1) * self.Kg / rho1**2)
        bbPg2_F = np.einsum('kC,j,A,B->jABkC', kg2_F, na, self.D2, self.D2 * self.Kg / rho2) \
            + np.einsum('ikC,A,B->iABkC', na_F, self.D2, self.D2 * (na @ c2) * self.Kg / rho2) \
            - np.einsum('kC,j,A,B->jABkC', rho2_F, na, self.D2,
                        self.D2 * (na @ c2) * self.Kg / rho2**2)
        bbPg3_F = np.einsum('kC,j,A,B->jABkC', kg3_F, nb, self.D3, self.D3 * self.Kg / rho3) \
            + np.einsum('ikC,A,B->iABkC', nb_F, self.D3, self.D3 * (nb @ c3) * self.Kg / rho3) \
            - np.einsum('kC,j,A,B->jABkC', rho3_F, nb, self.D3,
                        self.D3 * (nb @ c3) * self.Kg / rho3**2)
        bbPg4_F = np.einsum('kC,j,A,B->jABkC', kg4_F, nb, self.D4, self.D4 * self.Kg / rho4) \
            + np.einsum('ikC,A,B->iABkC', nb_F, self.D4, self.D4 * (nb @ c4) * self.Kg / rho4) \
            - np.einsum('kC,j,A,B->jABkC', rho4_F, nb, self.D4,
                        self.D4 * (nb @ c4) * self.Kg / rho4**2)

        bbPg_F = bbPg1_F + bbPg2_F + bbPg3_F + bbPg4_F

        tau1 = - na @ (g1 - singa * c1) / cosga
        tau2 = - na @ (g2 - singa * c2) / cosga
        tau3 = - nb @ (g3 - singb * c3) / cosgb
        tau4 = - nb @ (g4 - singb * c4) / cosgb

        tau1_F = ga_F * (tau1 * tanga + na @ c1) + np.einsum('ijA,i->jA', - na_F, (g1 - singa * c1) / cosga)\
            + np.einsum('i,ijA->jA', - na / cosga, g1_F) + \
            np.einsum('i,ijA->jA', tanga * na, c1_F)
        tau2_F = ga_F * (tau2 * tanga + na @ c2) + np.einsum('ijk,i->jk', - na_F, (g2 - singa * c2) / cosga)\
            + np.einsum('i,ijk->jk', - na / cosga, g2_F) + \
            np.einsum('i,ijk->jk', tanga * na, c2_F)
        tau3_F = gb_F * (tau3 * tangb + nb @ c3) + np.einsum('ijk,i->jk', - nb_F, (g3 - singb * c3) / cosgb)\
            + np.einsum('i,ijk->jk', - nb / cosgb, g3_F) + \
            np.einsum('i,ijk->jk', tangb * nb, c3_F)
        tau4_F = gb_F * (tau4 * tangb + nb @ c4) + np.einsum('ijk,i->jk', - nb_F, (g4 - singb * c4) / cosgb)\
            + np.einsum('i,ijk->jk', - nb / cosgb, g4_F) + \
            np.einsum('i,ijk->jk', tangb * nb, c4_F)

        bbPt1_F = np.einsum('i,A,B,kC->iABkC', na, self.D1, self.D1 * (self.Kt / rho1),
                            tau1_F * tanga + ga_F * (tau1 / cosga**2) - rho1_F * (tau1 * tanga / rho1)) \
            + np.einsum('ikC,A,B->iABkC', na_F, self.D1, self.D1 * (self.Kt * tau1 * tanga / rho1)) \
            - np.einsum('i,A,B,kC->iABkC', na, self.D2, self.D1 * (self.Kt / (rho2 * cosga)),
                        tau1_F + ga_F * tau1 * tanga - rho2_F * (tau1 / rho2)) \
            - np.einsum('ikC,A,B->iABkC', na_F, self.D2,
                        self.D1 * (self.Kt * tau1 / (rho2 * cosga)))
        bbPt2_F = np.einsum('i,A,B,kC->iABkC', na, self.D2, self.D2 * (self.Kt / rho2),
                            tau2_F * tanga + ga_F * (tau2 / cosga**2) - rho2_F * (tau2 * tanga / rho2)) \
            + np.einsum('ikC,A,B->iABkC', na_F, self.D2, self.D2 * (self.Kt * tau2 * tanga / rho2)) \
            - np.einsum('i,A,B,kC->iABkC', na, self.D1, self.D2 * (self.Kt / (rho1 * cosga)),
                        tau2_F + ga_F * tau2 * tanga - rho1_F * (tau2 / rho1)) \
            - np.einsum('ikC,A,B->iABkC', na_F, self.D1,
                        self.D2 * (self.Kt * tau2 / (rho1 * cosga)))
        bbPt3_F = np.einsum('i,A,B,kC->iABkC', nb, self.D3, self.D3 * (self.Kt / rho3),
                            tau3_F * tangb + gb_F * (tau3 / cosgb**2) - rho3_F * (tau3 * tangb / rho3)) \
            + np.einsum('ikC,A,B->iABkC', nb_F, self.D3, self.D3 * (self.Kt * tau3 * tangb / rho3)) \
            - np.einsum('i,A,B,kC->iABkC', nb, self.D4, self.D3 * (self.Kt / (rho4 * cosgb)),
                        tau3_F + gb_F * tau3 * tangb - rho4_F * (tau3 / rho4)) \
            - np.einsum('ikC,A,B->iABkC', nb_F, self.D4,
                        self.D3 * (self.Kt * tau3 / (rho4 * cosgb)))
        bbPt4_F = np.einsum('i,A,B,kC->iABkC', nb, self.D4, self.D4 * (self.Kt / rho4),
                            tau4_F * tangb + gb_F * (tau4 / cosgb**2) - rho4_F * (tau4 * tangb / rho4)) \
            + np.einsum('ikC,A,B->iABkC', nb_F, self.D4, self.D4 * (self.Kt * tau4 * tangb / rho4)) \
            - np.einsum('i,A,B,kC->iABkC', nb, self.D3, self.D4 * (self.Kt / (rho3 * cosgb)),
                        tau4_F + gb_F * tau4 * tangb - rho3_F * (tau4 / rho3)) \
            - np.einsum('ikC,A,B->iABkC', nb_F, self.D3,
                        self.D4 * (self.Kt * tau4 / (rho3 * cosgb)))

        bbPt_F = bbPt1_F + bbPt2_F + bbPt3_F + bbPt4_F

        kn1_F = np.einsum('ijA,i->jA', m1_F, c1) + \
            np.einsum('ijA,i->jA', c1_F, m1)
        kn2_F = np.einsum('ijA,i->jA', m2_F, c2) + \
            np.einsum('ijA,i->jA', c2_F, m2)
        kn3_F = np.einsum('ijA,i->jA', m3_F, c3) + \
            np.einsum('ijA,i->jA', c3_F, m3)
        kn4_F = np.einsum('ijA,i->jA', m4_F, c4) + \
            np.einsum('ijA,i->jA', c4_F, m4)

        bbPn1_F = np.einsum('kC,j,A,B->jABkC', kn1_F, m1, self.D1, self.D1 * self.Kn / rho1) \
            + np.einsum('ikC,A,B->iABkC', m1_F, self.D1, self.D1 * (m1 @ c1) * self.Kn / rho1) \
            - np.einsum('kC,j,A,B->jABkC', rho1_F, m1, self.D1,
                        self.D1 * (m1 @ c1) * self.Kn / rho1**2)
        bbPn2_F = np.einsum('kC,j,A,B->jABkC', kn2_F, m2, self.D2, self.D2 * self.Kn / rho2) \
            + np.einsum('ikC,A,B->iABkC', m2_F, self.D2, self.D2 * (m2 @ c2) * self.Kn / rho2) \
            - np.einsum('kC,j,A,B->jABkC', rho2_F, m2, self.D2,
                        self.D2 * (m2 @ c2) * self.Kn / rho2**2)
        bbPn3_F = np.einsum('kC,j,A,B->jABkC', kn3_F, m3, self.D3, self.D3 * self.Kn / rho3) \
            + np.einsum('ikC,A,B->iABkC', m3_F, self.D3, self.D3 * (m3 @ c3) * self.Kn / rho3) \
            - np.einsum('kC,j,A,B->jABkC', rho3_F, m3, self.D3,
                        self.D3 * (m3 @ c3) * self.Kn / rho3**2)
        bbPn4_F = np.einsum('kC,j,A,B->jABkC', kn4_F, m4, self.D4, self.D4 * self.Kn / rho4) \
            + np.einsum('ikC,A,B->iABkC', m4_F, self.D4, self.D4 * (m4 @ c4) * self.Kn / rho4) \
            - np.einsum('kC,j,A,B->jABkC', rho4_F, m4, self.D4,
                        self.D4 * (m4 @ c4) * self.Kn / rho4**2)

        bbPn_F = bbPn1_F + bbPn2_F + bbPn3_F + bbPn4_F

        return bbPn_F + bbPg_F + bbPt_F

    def bbP_G(self, F, G):
        # layer a
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        # layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        singb = e3 @ e4
        tangb = singb / cosgb
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4
        g3 = np.einsum('ijk,j,k->i', G, self.D4, self.D3) / rho4
        g4 = np.einsum('ijk,j,k->i', G, self.D3, self.D4) / rho3

        c1_G = np.einsum('ij,A,B->ijAB', self.I3, self.D1, self.D1 / rho1)
        c2_G = np.einsum('ij,A,B->ijAB', self.I3, self.D2, self.D2 / rho2)
        c3_G = np.einsum('ij,A,B->ijAB', self.I3, self.D3, self.D3 / rho3)
        c4_G = np.einsum('ij,A,B->ijAB', self.I3, self.D4, self.D4 / rho4)

        g1_G = np.einsum('ij,A,B->ijAB', self.I3, self.D2, self.D1 / rho2)
        g2_G = np.einsum('ij,A,B->ijAB', self.I3, self.D1, self.D2 / rho1)
        g3_G = np.einsum('ij,A,B->ijAB', self.I3, self.D4, self.D3 / rho4)
        g4_G = np.einsum('ij,A,B->ijAB', self.I3, self.D3, self.D4 / rho3)

        # normal bending
        bbPn1_G = np.einsum('i,ikCD,j,A,B->jABkCD', m1, c1_G,
                            m1, self.D1, self.D1 * self.Kn / rho1)
        bbPn2_G = np.einsum('i,ikCD,j,A,B->jABkCD', m2, c2_G,
                            m2, self.D2, self.D2 * self.Kn / rho2)
        bbPn3_G = np.einsum('i,ikCD,j,A,B->jABkCD', m3, c3_G,
                            m3, self.D3, self.D3 * self.Kn / rho3)
        bbPn4_G = np.einsum('i,ikCD,j,A,B->jABkCD', m4, c4_G,
                            m4, self.D4, self.D4 * self.Kn / rho4)

        bbPn_G = bbPn1_G + bbPn2_G + bbPn3_G + bbPn4_G

        # geodesic bending
        bbPg1_G = np.einsum('i,ikCD,j,A,B->jABkCD', na, c1_G,
                            na, self.D1, self.D1 * self.Kg / rho1)
        bbPg2_G = np.einsum('i,ikCD,j,A,B->jABkCD', na, c2_G,
                            na, self.D2, self.D2 * self.Kg / rho2)
        bbPg3_G = np.einsum('i,ikCD,j,A,B->jABkCD', nb, c3_G,
                            nb, self.D3, self.D3 * self.Kg / rho3)
        bbPg4_G = np.einsum('i,ikCD,j,A,B->jABkCD', nb, c4_G,
                            nb, self.D4, self.D4 * self.Kg / rho4)

        bbPg_G = bbPg1_G + bbPg2_G + bbPg3_G + bbPg4_G

        # torsion
        bbPt1_G = np.einsum('i,ikCD,j,A,B->jABkCD', - na * tanga / cosga, g1_G - singa * c1_G, na, self.D1, self.D1 * self.Kt / rho1) \
            + np.einsum('i,ikCD,j,A,B->jABkCD', na / cosga**2, g1_G -
                        singa * c1_G, na, self.D2, self.D1 * self.Kt / rho2)
        bbPt2_G = np.einsum('i,ikCD,j,A,B->jABkCD', - na * tanga / cosga, g2_G - singa * c2_G, na, self.D2, self.D2 * self.Kt / rho2) \
            + np.einsum('i,ikCD,j,A,B->jABkCD', na / cosga**2, g2_G -
                        singa * c2_G, na, self.D1, self.D2 * self.Kt / rho1)
        bbPt3_G = np.einsum('i,ikCD,j,A,B->jABkCD', - nb * tangb / cosgb, g3_G - singb * c3_G, nb, self.D3, self.D3 * self.Kt / rho3) \
            + np.einsum('i,ikCD,j,A,B->jABkCD', nb / cosgb**2, g3_G -
                        singb * c3_G, nb, self.D4, self.D3 * self.Kt / rho4)
        bbPt4_G = np.einsum('i,ikCD,j,A,B->jABkCD', - nb * tangb / cosgb, g4_G - singb * c4_G, nb, self.D4, self.D4 * self.Kt / rho4) \
            + np.einsum('i,ikCD,j,A,B->jABkCD', nb / cosgb**2, g4_G -
                        singb * c4_G, nb, self.D3, self.D4 * self.Kt / rho3)

        bbPt_G = bbPt1_G + bbPt2_G + bbPt3_G + bbPt4_G

        return bbPn_G + bbPg_G + bbPt_G

    # derivatives of variables. Maybe for export
    def rho_F(self, F, G, al):
        d1 = F @ self.D1
        d2 = F @ self.D2
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e3 = d3 / rho3
        e4 = d4 / rho4
        e = [e1, e2, e3, e4]

        rho_F = np.outer(e[al], self.D[al])

        return rho_F

    def rho_F_F(self, F, G, al):
        rho_F_F = np.einsum('ijk,l->iljk', self.e_F(F, G, al), self.D[al])
        return rho_F_F

    def e1(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2

        return e1

    def e_F(self, F, G):
        """ derivative of tangent vectors w.r.t. to F
        """
        # layer a
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2

        # layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4

        e1_F = np.einsum('ij,k->ijk', self.I3, self.D1) / rho1 \
            - np.einsum('i,j,k->ijk', e1,  e1, self.D1) / rho1
        e2_F = np.einsum('ij,k->ijk', self.I3, self.D2) / rho2 \
            - np.einsum('i,j,k->ijk', e2,  e2, self.D2) / rho2
        e3_F = np.einsum('ij,k->ijk', self.I3, self.D3) / rho3 \
            - np.einsum('i,j,k->ijk', e3,  e3, self.D3) / rho3
        e4_F = np.einsum('ij,k->ijk', self.I3, self.D4) / rho4 \
            - np.einsum('i,j,k->ijk', e4,  e4, self.D4) / rho4

        return e1_F, e2_F, e3_F, e4_F

    def e_F_F(self, F, G, al):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e = [e1, e2]
        rho = [rho1, rho2]

        e_F_F = - (np.einsum('ijk,lm->ijklm', self.e_F(F, G, al), self.rho_F(F, G, al))
                   + np.einsum('ijk,lm->ilmjk', self.e_F(F, G, al),
                               self.rho_F(F, G, al))
                   + np.einsum('i,jklm->ijklm', e[al], self.rho_F_F(F, G, al))
                   ) / rho[al]

        return e_F_F

    def ga(self, F, G):

        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2

        return np.arcsin(e1@e2)

    def ga_F(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)

        ga_F = np.outer(m1, self.D1) / rho1 - \
            np.outer(m2, self.D2) / rho2

        return ga_F

    def gb_F(self, F, G):
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        singb = e3 @ e4
        tangb = singb / cosgb
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)

        gb_F = np.outer(m3, self.D3) / rho3 - \
            np.outer(m4, self.D4) / rho4

        return gb_F

    def ga_F_F(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)

        ga_F_F = -np.einsum('i,j,kl->ijkl', m1, self.D1, self.rho_F(F, G, 0)) / rho1**2 \
            + np.einsum('ikl,j->ijkl', self.m_F(F, G, 0), self.D1) / rho1 \
            + np.einsum('i,j,kl->ijkl', m2, self.D2, self.rho_F(F, G, 1)) / rho2**2 \
            - np.einsum('ikl,j->ijkl', self.m_F(F, G, 1), self.D2) / rho2

        return ga_F_F

    def m(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)

        # layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        singb = e3 @ e4
        tangb = singb / cosgb
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)

        return m1, m2, m3, m4

    def m_F(self, F, G):
        # layer a
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)

        ga_F = self.ga_F(F, G)
        gb_F = self.gb_F(F, G)
        e_F = self.e_F(F, G)

        # layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        singb = e3 @ e4
        tangb = singb / cosgb
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)

        e1_F, e2_F, e3_F, e4_F = e_F

        m1_F = np.einsum('i,jk->ijk', tanga * m1 - e1, ga_F) \
            + e2_F / cosga - tanga * e1_F

        m2_F = np.einsum('i,jk->ijk', tanga * m2 + e2, ga_F) \
            - e1_F / cosga + tanga * e2_F

        m3_F = np.einsum('i,jk->ijk', tangb * m3 - e3, gb_F) \
            + e4_F / cosgb - tangb * e3_F

        m4_F = np.einsum('i,jk->ijk', tangb * m4 + e4, gb_F) \
            - e3_F / cosgb + tangb * e4_F

        return m1_F, m2_F, m3_F, m4_F

    def m_F_F(self, F, G, al):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        m = [m1, m2]
        e = [e1, e2]

        m1_F_F = np.einsum('ilm,jk->ijklm', np.einsum('i,lm->ilm', m[0] / cosga**2, self.ga_F(
            F, G)) + tanga * self.m_F(F, G, 0) - self.e_F(F, G, 0), self.ga_F(F, G)) \
            + np.einsum('i,jklm', tanga * m[0] - e[0], self.ga_F_F(F, G)) \
            + np.einsum('ijk,lm->ijklm', (tanga * self.e_F(F, G, 1) - self.e_F(F, G, 0) / cosga) / cosga, self.ga_F(F, G)) \
            + self.e_F_F(F, G, 1) / cosga - tanga * self.e_F_F(F, G, 0)

        return m1_F_F

    def na(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga

        return na

    def na_F(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga

        ga_F = self.ga_F(F, G)
        e1_F, e2_F, _, _ = self.e_F(F, G)

        na_F = np.einsum('i,jk->ijk', tanga * na, ga_F) \
            + np.cross(e1_F, e2 / cosga, axis=0)  \
            - np.cross(e2_F, e1 / cosga, axis=0)
        return na_F

    def nb_F(self, F, G):
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        singb = e3 @ e4
        tangb = singb / cosgb
        nb = e3xe4 / cosgb
        gb_F = self.gb_F(F, G)
        _, _, e3_F, e4_F = self.e_F(F, G)

        nb_F = np.einsum('i,jk->ijk', tangb * nb, gb_F) \
            + np.cross(e3_F, e4 / cosgb, axis=0)  \
            - np.cross(e4_F, e3 / cosgb, axis=0)
        return nb_F

    def na_F_F(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga

        na_F_F = np.einsum('i,jk,lm', na / cosga**2, self.ga_F(F, G), self.ga_F(F, G))  \
            + np.einsum('i,jklm->ijklm', tanga * na, self.ga_F_F(F, G)) \
            + np.einsum('ilm,jk->ijklm', self.na_F(F, G), self.ga_F(F, G) * tanga) \
            + np.einsum('ijk,lm->ijklm', np.cross(self.e_F(F, G, 0), e2 / cosga, axis=0)
                        - np.cross(self.e_F(F, G, 1), e1 / cosga, axis=0), tanga * self.ga_F(F, G)) \
            + np.cross(self.e_F_F(F, G, 0), e2 / cosga, axis=0) \
            + np.einsum('opi,ojk,plm->ijklm', self.lc3, self.e_F(F, G, 0), self.e_F(F, G, 1) / cosga) \
            - np.cross(self.e_F_F(F, G, 1), e1 / cosga, axis=0) \
            - np.einsum('opi,ojk,plm->ijklm', self.lc3,
                        self.e_F(F, G, 1), self.e_F(F, G, 0) / cosga)

        return na_F_F

    def c_F(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2

        # Layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4

        c1_F = - np.einsum('i,j,k', c1, e1, self.D1) / rho1
        c2_F = - np.einsum('i,j,k', c2, e2, self.D2) / rho2
        c3_F = - np.einsum('i,j,k', c3, e3, self.D3) / rho3
        c4_F = - np.einsum('i,j,k', c4, e4, self.D4) / rho4

        return c1_F, c2_F, c3_F, c4_F

    def c_F_F(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2

        # Layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4

        c1_F_F = np.einsum('i,j,k,lm', c1, e1 / rho1**2, self.D1, rho1_F) \
            - np.einsum('ilm, j, k->ijklm', c1_F, e1, self.D1 / rho1) \
            - np, einsum('i,jlm,k->ijklm', c1, e1_F, self.D1)

        return c1_F_F

    def c(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2

        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4

        return c1, c2, c3, c4

    def g(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        g3 = np.einsum('ijk,j,k->i', G, self.D4, self.D3) / rho4
        g4 = np.einsum('ijk,j,k->i', G, self.D3, self.D4) / rho3

        return g1, g2, g3, g4

    def g_F(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        g3 = np.einsum('ijk,j,k->i', G, self.D4, self.D3) / rho4
        g4 = np.einsum('ijk,j,k->i', G, self.D3, self.D4) / rho3

        g1_F = - np.einsum('i,j,k', g1, e2, self.D2) / rho2
        g2_F = - np.einsum('i,j,k', g2, e1, self.D1) / rho1
        g3_F = - np.einsum('i,j,k', g3, e4, self.D4) / rho4
        g4_F = - np.einsum('i,j,k', g4, e3, self.D3) / rho3

        return g1_F, g2_F, g3_F, g4_F

    def v_F_num(self, F, G, v=e_F):
        v_F_num = Numerical_derivative(lambda F: v(F, G), order=2)._X(F)
        return v_F_num

    def P_alt(self, F, G):
        ''' Piola-Lagrange stress tensor by direct derivatives w.r.t. F of the kinematic descriptors
        '''
        # layer a
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        # layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        singb = e3 @ e4
        tangb = singb / cosgb
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4
        g3 = np.einsum('ijk,j,k->i', G, self.D4, self.D3) / rho4
        g4 = np.einsum('ijk,j,k->i', G, self.D3, self.D4) / rho3

        singc = na @ nb
        cosgc = norm(cross3(na, nb))

        rho1_F = self.rho_F(F, G, 0)
        rho2_F = self.rho_F(F, G, 1)
        rho3_F = self.rho_F(F, G, 2)
        rho4_F = self.rho_F(F, G, 3)

        e_F = self.e_F(F, G)
        e1_F, e2_F, e3_F, e4_F = e_F

        ga_F = self.ga_F(F, G)
        gb_F = self.gb_F(F, G)

        Pe1 = self.Ke * rho1_F * (rho1-1)
        Pe2 = self.Ke * rho2_F * (rho2-1)
        Pe3 = self.Ke * rho3_F * (rho3-1)
        Pe4 = self.Ke * rho4_F * (rho4-1)

        Pe = Pe1 + Pe2 + Pe3 + Pe4

        Pga = self.Ks * ga_F * np.arcsin(singa)
        Pgb = self.Ks * gb_F * np.arcsin(singb)

        Ps = Pga + Pgb

        m1_F, m2_F, m3_F, m4_F = self.m_F(F, G)

        c_F = self.c_F(F, G)
        c1_F, c2_F, c3_F, c4_F = c_F

        Pn1 = self.Kn * m1 @ c1 * \
            (np.einsum('ijk,i->jk', m1_F, c1) + np.einsum('ijk,i->jk', c1_F, m1))
        Pn2 = self.Kn * m2 @ c2 * \
            (np.einsum('ijk,i->jk', m2_F, c2) + np.einsum('ijk,i->jk', c2_F, m2))
        Pn3 = self.Kn * m3 @ c3 * \
            (np.einsum('ijk,i->jk', m3_F, c3) + np.einsum('ijk,i->jk', c3_F, m3))
        Pn4 = self.Kn * m4 @ c4 * \
            (np.einsum('ijk,i->jk', m4_F, c4) + np.einsum('ijk,i->jk', c4_F, m4))

        Pn = Pn1 + Pn2 + Pn3 + Pn4

        na_F = self.na_F(F, G, ga_F, e_F)
        nb_F = self.nb_F(F, G, gb_F, e_F)

        Pg1 = self.Kg * na @ c1 * \
            (np.einsum('ijk,i->jk', na_F, c1) + np.einsum('ijk,i->jk', c1_F, na))
        Pg2 = self.Kg * na @ c2 * \
            (np.einsum('ijk,i->jk', na_F, c2) + np.einsum('ijk,i->jk', c2_F, na))
        Pg3 = self.Kg * nb @ c3 * \
            (np.einsum('ijk,i->jk', nb_F, c3) + np.einsum('ijk,i->jk', c3_F, nb))
        Pg4 = self.Kg * nb @ c4 * \
            (np.einsum('ijk,i->jk', nb_F, c4) + np.einsum('ijk,i->jk', c4_F, nb))

        Pg = Pg1 + Pg2 + Pg3 + Pg4

        tau1 = - na @ (g1 - singa * c1) / cosga
        tau2 = - na @ (g2 - singa * c2) / cosga
        tau3 = - nb @ (g3 - singb * c3) / cosgb
        tau4 = - nb @ (g4 - singb * c4) / cosgb

        g1_F, g2_F, g3_F, g4_F = self.g_F(F, G)

        Pt1 = self.Kt * tau1 * (ga_F * (tau1 * tanga + na @ c1) + np.einsum('ijk,i->jk', - na_F, (g1 - singa * c1) / cosga)
                                + np.einsum('i,ijk->jk', - na / cosga, g1_F) + np.einsum('i,ijk->jk', tanga * na, c1_F))
        Pt2 = self.Kt * tau2 * (ga_F * (tau2 * tanga + na @ c2) + np.einsum('ijk,i->jk', - na_F, (g2 - singa * c2) / cosga)
                                + np.einsum('i,ijk->jk', - na / cosga, g2_F) + np.einsum('i,ijk->jk', tanga * na, c2_F))
        Pt3 = self.Kt * tau3 * (gb_F * (tau3 * tangb + nb @ c3) + np.einsum('ijk,i->jk', - nb_F, (g3 - singb * c3) / cosgb)
                                + np.einsum('i,ijk->jk', - nb / cosgb, g3_F) + np.einsum('i,ijk->jk', tangb * nb, c3_F))
        Pt4 = self.Kt * tau4 * (gb_F * (tau4 * tangb + nb @ c4) + np.einsum('ijk,i->jk', - nb_F, (g4 - singb * c4) / cosgb)
                                + np.einsum('i,ijk->jk', - nb / cosgb, g4_F) + np.einsum('i,ijk->jk', tangb * nb, c4_F))

        Pt = Pt1 + Pt2 + Pt3 + Pt4

        # shear c
        Pc = self.Kc * np.arcsin(singc) / cosgc * (np.einsum('i,ijk->jk',
                                                             na, nb_F) + np.einsum('i,ijk->jk', nb, na_F))

        return Pe + Ps + Pn + Pg + Pt + Pc

    # Fields for export
    def W(self, F, G=G0):
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
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / \
            rho1  # same as G@D1@D1 speed is similar
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        tau1 = - na @ (g1 - singa * c1) / cosga
        tau2 = - na @ (g2 - singa * c2) / cosga

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

        tau3 = - nb @ (g3 - singb * c3) / cosgb
        tau4 = - nb @ (g4 - singb * c4) / cosgb

        singc = na @ nb

        # complete energy
        W = .5 * self.Ke * ((rho1-1)**2 + (rho2-1)**2 + (rho3-1)**2 + (rho4-1)**2) \
            + .5 * self.Kg * ((na@c1)**2 + (na@c2)**2 + (nb@c3)**2 + (nb@c4)**2) \
            + .5 * self.Kn * ((-m1@c1)**2 + (-m2@c2)**2 + (-m3@c3)**2 + (-m4@c4)**2) \
            + .5 * self.Kt * (tau1**2 + tau2**2 + tau3**2 + tau4**2) \
            + .5 * self.Ks * (np.arcsin(singa)**2 + np.arcsin(singb)**2) \
            + .5 * self.Kc * (np.arcsin(singc)**2)

        return W

    def We(self, F, G=G0):

        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm(d1)
        rho2 = norm(d2)
        # e1 = d1 / rho1
        # e2 = d2 / rho2
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm(d3)
        rho4 = norm(d4)
        # e3 = d3 / rho3
        # e4 = d4 / rho4
        We = .5 * self.Ke * ((rho1-1)**2 + (rho2-1)**2 +
                             (rho3-1)**2 + (rho4-1)**2)
        return We

    def Pe(self, F, G=G0):

        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4

        # Pe1 = self.Ke * (rho1-1) * np.outer(e1, self.D1)

        Pe = self.Ke * ((rho1-1) * np.outer(e1, self.D1) + (rho2-1) * np.outer(e2, self.D2)
                        + (rho3-1) * np.outer(e3, self.D3) + (rho4-1) * np.outer(e4, self.D4))

        return Pe

    def Wn(self, F, G=G0):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / \
            rho1  # same as G@D1@D1 speed is similar
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2

        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4

        Wn1 = .5 * self.Kn * (-m1 @ c1)**2
        Wn2 = .5 * self.Kn * (-m2 @ c2)**2
        Wn3 = .5 * self.Kn * (-m3 @ c3)**2
        Wn4 = .5 * self.Kn * (-m4 @ c4)**2

        return Wn1 + Wn2 + Wn3 + Wn4

    def Pn(self, F, G=G0):

        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa / cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2

        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        singb = e3 @ e4
        cosgb = norm3(e3xe4)
        tangb = singb / cosgb
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4

        kn1 = -m1 @ c1
        kn2 = -m2 @ c2
        kn3 = -m3 @ c3
        kn4 = -m4 @ c4

        # Pn1 = self.Kn * kn1 * np.outer((kn1 * tanga + e1 @ c1) * m1 \
        #      + tanga * cross3(e1, cross3(c1, e1)) - kn1 * e1, self.D1)  / rho1 \
        #      + self.Kn * kn1 * np.outer((-kn1 * singa  - cosga * e1 @ c1) * m2 \
        #      - cross3(e2, cross3(c1, e2)), self.D2) / (rho2 * cosga)

        Pn1D1 = ((kn1 * tanga + e1 @ c1) * m1
                 + tanga * cross3(e1, cross3(c1, e1)) - kn1 * e1) / rho1
        Pn1D2 = ((-kn1 * singa - cosga * e1 @ c1) * m2
                 - cross3(e2, cross3(c1, e2))) / (rho2 * cosga)

        Pn2D1 = ((kn2 * singa - cosga * e2 @ c2) * m1
                 + cross3(e1, cross3(c2, e1))) / (cosga * rho1)
        Pn2D2 = ((- kn2 * tanga + e2 @ c2) * m2
                 - tanga * cross3(e2, cross3(c2, e2)) - kn2 * e2) / rho2

        Pn3D3 = ((kn3 * tangb + e3 @ c3) * m3
                 + tangb * cross3(e3, cross3(c3, e3)) - kn3 * e3) / rho3
        Pn3D4 = ((-kn3 * singb - cosgb * e3 @ c3) * m4
                 - cross3(e4, cross3(c3, e4))) / (rho4 * cosgb)

        Pn4D3 = ((kn4 * singb - cosgb * e4 @ c4) * m3
                 + cross3(e3, cross3(c4, e3))) / (cosgb * rho3)
        Pn4D4 = ((- kn4 * tangb + e4 @ c4) * m4
                 - tangb * cross3(e4, cross3(c4, e4)) - kn4 * e4) / rho4

        # Pn1 = self.Kn * (np.outer(kn1*Pn1D1, self.D1)
        #                 + np.outer(kn1*Pn1D2, self.D2))

        # Pn2 = self.Kn * (np.outer(kn2*Pn2D1, self.D1)
        #                  + np.outer( kn2*Pn2D2, self.D2))

        Pn = self.Kn * ((np.outer(kn1*Pn1D1 + kn2*Pn2D1, self.D1)
                         + np.outer(kn1*Pn1D2 + kn2*Pn2D2, self.D2))
                        + np.outer(kn3*Pn3D3 + kn4*Pn4D3, self.D3) \
            + np.outer(kn3*Pn3D4 + kn4*Pn4D4, self.D4))

        # num = Numerical_derivative(\
        # lambda G: self.c_F(F, G), order=2)._X(G)
        # print(np.linalg.norm(num - c_F_G))
        # print('a')

        return Pn

    def Wg(self, F, G=G0):
        # geodesic curvature energy density
        # layer a
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
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / \
            rho1  # same as G@D1@D1 speed is similar
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2

        # layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        nb = e3xe4 / cosgb
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4

        kg1 = na @ c1
        kg2 = na @ c2
        kg3 = nb @ c3
        kg4 = nb @ c4

        Wg1 = .5 * self.Kg * kg1**2
        Wg2 = .5 * self.Kg * kg2**2
        Wg3 = .5 * self.Kg * kg3**2
        Wg4 = .5 * self.Kg * kg4**2

        return Wg1 + Wg2 + Wg3 + Wg4

    def Pg(self, F, G=G0):
        # geodesic curvature part of First Piola-Lagrange stress tensor

        # layer a
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
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / \
            rho1  # same as G@D1@D1 speed is similar
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2

        # layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        singb = e3 @ e4
        cosgb = norm3(e3xe4)
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4

        kg1 = na @ c1
        kg2 = na @ c2
        kg3 = nb @ c3
        kg4 = nb @ c4

        Pg1D1 = (kg1 * singa * m1 + cross3(e1, cross3(cross3(e2, c1), e1))
                 - cosga * kg1 * e1) / (rho1 * cosga)
        Pg1D2 = (- kg1 * singa * m2 - cross3(e2,
                                             cross3(cross3(e1, c1), e2))) / (rho2 * cosga)

        Pg2D1 = (kg2 * singa * m1 + cross3(e1,
                                           cross3(cross3(e2, c2), e1))) / (rho1 * cosga)
        Pg2D2 = (- kg2 * singa * m2 - cross3(e2, cross3(cross3(e1, c2), e2))
                 - cosga * kg2 * e2) / (rho2 * cosga)

        Pg3D3 = (kg3 * singb * m3 + cross3(e3, cross3(cross3(e4, c3), e3))
                 - cosgb * kg3 * e3) / (rho3 * cosgb)
        Pg3D4 = (- kg3 * singb * m4 - cross3(e4,
                                             cross3(cross3(e3, c3), e4))) / (rho4 * cosgb)

        Pg4D3 = (kg4 * singb * m3 + cross3(e3,
                                           cross3(cross3(e4, c4), e3))) / (rho3 * cosgb)
        Pg4D4 = (- kg4 * singb * m4 - cross3(e4, cross3(cross3(e3, c4), e4))
                 - cosgb * kg4 * e4) / (rho4 * cosgb)

        Pg2 = self.Kg * kg2 * \
            (np.outer(Pg2D1, self.D1) + np.outer(Pg2D2, self.D2))

        Pg1 = self.Kg * kg1 * \
            (np.outer(Pg1D1, self.D1) + np.outer(Pg1D2, self.D2))

        Pg3 = self.Kg * kg3 * \
            (np.outer(Pg3D3, self.D3) + np.outer(Pg3D4, self.D4))

        Pg4 = self.Kg * kg4 * \
            (np.outer(Pg4D3, self.D3) + np.outer(Pg4D4, self.D4))

        return Pg1 + Pg2 + Pg3 + Pg4

    def Wt(self, F, G=G0):
        # torisonal energy density
        # layer a
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
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / \
            rho1  # same as G@D1@D1 speed is similar
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        # layer b
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
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4
        g3 = np.einsum('ijk,j,k->i', G, self.D4, self.D3) / rho4
        g4 = np.einsum('ijk,j,k->i', G, self.D3, self.D4) / rho3

        tau1 = - na @ (g1 - singa * c1) / cosga
        tau2 = - na @ (g2 - singa * c2) / cosga
        tau3 = - nb @ (g3 - singb * c3) / cosgb
        tau4 = - nb @ (g4 - singb * c4) / cosgb

        Wt1 = .5 * self.Kt * tau1**2
        Wt2 = .5 * self.Kt * tau2**2
        Wt3 = .5 * self.Kt * tau3**2
        Wt4 = .5 * self.Kt * tau4**2

        return Wt1 + Wt2 + Wt3 + Wt4

    def Pt(self, F, G=G0):
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
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / \
            rho1  # same as G@D1@D1 speed is similar
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        kg1 = na@c1
        kg2 = na@c2

        # layer b
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

        kg3 = nb@c3
        kg4 = nb@c4

        tau1 = - na @ (g1 - singa * c1) / cosga
        tau2 = - na @ (g2 - singa * c2) / cosga
        tau3 = - nb @ (g3 - singb * c3) / cosgb
        tau4 = - nb @ (g4 - singb * c4) / cosgb

        Pt1D1 = ((2 * tau1 * singa + kg1 * cosga) * m1 - kg1 * singa * e1
                 - cross3(e1, cross3(cross3(e2, g1 - singa*c1), e1)) / cosga) / (rho1 * cosga)
        Pt1D2 = ((-2 * tau1 * singa - kg1 * cosga) * m2 + (g1@na) * e2
                 + cross3(e2, cross3(cross3(e1, g1-singa*c1), e2)) / cosga) / (rho2 * cosga)

        Pt2D1 = ((2 * tau2 * singa + kg2 * cosga) * m1 + (g2@na) * e1
                 - cross3(e1, cross3(cross3(e2, g2 - singa*c2), e1)) / cosga) / (rho1 * cosga)
        Pt2D2 = ((-2 * tau2 * singa - kg2 * cosga) * m2 - kg2 * singa * e2
                 + cross3(e2, cross3(cross3(e1, g2-singa*c2), e2)) / cosga) / (rho2 * cosga)

        Pt3D3 = ((2 * tau3 * singb + kg3 * cosgb) * m3 - kg3 * singb * e3
                 - cross3(e3, cross3(cross3(e4, g3 - singb*c3), e3)) / cosgb) / (rho3 * cosgb)
        Pt3D4 = ((-2 * tau3 * singb - kg3 * cosgb) * m4 + (g3@nb) * e4
                 + cross3(e4, cross3(cross3(e3, g3-singb*c3), e4)) / cosgb) / (rho4 * cosgb)

        Pt4D3 = ((2 * tau4 * singb + kg4 * cosgb) * m3 + (g4@nb) * e3
                 - cross3(e3, cross3(cross3(e4, g4 - singb*c4), e3)) / cosgb) / (rho3 * cosgb)
        Pt4D4 = ((-2 * tau4 * singb - kg4 * cosgb) * m4 - kg4 * singb * e4
                 + cross3(e4, cross3(cross3(e3, g4-singb*c4), e4)) / cosgb) / (rho4 * cosgb)

        Pt1 = self.Kt * tau1 * \
            (np.outer(Pt1D1, self.D1) + np.outer(Pt1D2, self.D2))
        Pt2 = self.Kt * tau2 * \
            (np.outer(Pt2D1, self.D1) + np.outer(Pt2D2, self.D2))
        Pt3 = self.Kt * tau3 * \
            (np.outer(Pt3D3, self.D3) + np.outer(Pt3D4, self.D4))
        Pt4 = self.Kt * tau4 * \
            (np.outer(Pt4D3, self.D3) + np.outer(Pt4D4, self.D4))

        return Pt1 + Pt2 + Pt3 + Pt4

    def Ws(self, F, G=G0):
        # layer a
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2

        # layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        singb = e3 @ e4

        Wsa = .5 * self.Ks * np.arcsin(singa)**2
        Wsb = .5 * self.Ks * np.arcsin(singb)**2
        return Wsa + Wsb

    def Ps(self, F, G=G0):
        # layer a
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

        # layer b
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

        PsaD1 = m1 / rho1
        PsaD2 = -m2 / rho2

        PsbD3 = m3 / rho3
        PsbD4 = -m4 / rho4

        Psa = self.Ks * \
            np.arcsin(singa) * (np.outer(PsaD1, self.D1) +
                                np.outer(PsaD2, self.D2))
        Psb = self.Ks * \
            np.arcsin(singb) * (np.outer(PsbD3, self.D3) +
                                np.outer(PsbD4, self.D4))

        return Psa + Psb

    def Wc(self, F, G=G0):
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

        Wc = .5 * self.Kc * np.arcsin(na@nb)**2

        return Wc

    def Pc(self, F, G=G0):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        singa = e1 @ e2
        tanga = singa/cosga
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        singb = e3 @ e4
        tangb = singb / cosgb
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)

        singc = na @ nb
        cosgc = norm(cross3(na, nb))

        PcD1 = (singc * tanga * m1 + cross3(e1,
                                            cross3(cross3(e2, nb), e1)) / cosga) / rho1
        PcD2 = (-singc * tanga * m2 - cross3(e2,
                                             cross3(cross3(e1, nb), e2)) / cosga) / rho2
        PcD3 = (singc * tangb * m3 + cross3(e3,
                                            cross3(cross3(e4, na), e3)) / cosgb) / rho3
        PcD4 = (-singc * tangb * m4 - cross3(e4,
                                             cross3(cross3(e3, na), e4)) / cosgb) / rho4

        Pc = self.Kc * np.arcsin(singc) / cosgc * (np.outer(PcD1, self.D1) + np.outer(PcD2, self.D2)
                                                   + np.outer(PcD3, self.D3) + np.outer(PcD4, self.D4))

        return Pc

    def rho(self, F, G=G0):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm(d1)
        rho2 = norm(d2)
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm(d3)
        rho4 = norm(d4)

        return np.array([rho1, rho2, rho3, rho4])

    def ga(self, F, G=G0):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        singa = e1 @ e2

        # Layer b
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        singb = e3 @ e4

        return np.array([np.arcsin(singa), np.arcsin(singb)])

    def tau(self, F, G=G0):
        # layer a
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
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / \
            rho1  # same as G@D1@D1 speed is similar
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2
        g1 = np.einsum('ijk,j,k->i', G, self.D2, self.D1) / rho2
        g2 = np.einsum('ijk,j,k->i', G, self.D1, self.D2) / rho1

        # layer b
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
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4
        g3 = np.einsum('ijk,j,k->i', G, self.D4, self.D3) / rho4
        g4 = np.einsum('ijk,j,k->i', G, self.D3, self.D4) / rho3

        tau1 = - na @ (g1 - singa * c1) / cosga
        tau2 = - na @ (g2 - singa * c2) / cosga
        tau3 = - nb @ (g3 - singb * c3) / cosgb
        tau4 = - nb @ (g4 - singb * c4) / cosgb

        return np.array([tau1, tau2, tau3, tau4])   

    def kappa_n(self, F, G=G0):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        na = e1xe2 / cosga
        m1 = cross3(na, e1)
        m2 = cross3(na, e2)
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / \
            rho1  # same as G@D1@D1 speed is similar
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2

        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        nb = e3xe4 / cosgb
        m3 = cross3(nb, e3)
        m4 = cross3(nb, e4)
        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4

        kn1 = -m1 @ c1
        kn2 = -m2 @ c2
        kn3 = -m3 @ c3
        kn4 = -m4 @ c4

        return np.array([kn1, kn2, kn3, kn4])    

    def kappa_g(self, F, G=G0):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e1xe2 = cross3(e1, e2)
        cosga = norm3(e1xe2)
        na = e1xe2 / cosga

        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / \
            rho1  # same as G@D1@D1 speed is similar
        c2 = np.einsum('ijk,j,k->i', G, self.D2, self.D2) / rho2

        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        e3 = d3 / rho3
        e4 = d4 / rho4
        e3xe4 = cross3(e3, e4)
        cosgb = norm3(e3xe4)
        nb = e3xe4 / cosgb

        c3 = np.einsum('ijk,j,k->i', G, self.D3, self.D3) / rho3
        c4 = np.einsum('ijk,j,k->i', G, self.D4, self.D4) / rho4

        kg1 = na @ c1
        kg2 = na @ c2
        kg3 = nb @ c3
        kg4 = nb @ c4

        return np.array([kg1, kg2, kg3, kg4])  

    # numerical derivatives
    def P_num(self, F, G, W=W):
        P_num = Numerical_derivative(lambda F: W(F, G), order=2)._X(F)
        return P_num

    def bbP_num(self, F, G, W=W):
        bbP_num = Numerical_derivative(lambda G: self.W(F, G), order=2)._X(G)
        return bbP_num

    def P_F_num(self, F, G, P=P):
        P_F_num = Numerical_derivative(lambda F: P(F, G), order=2)._X(F)
        return P_F_num

    def P_G_num(self, F, G, P=P):
        P_G_num = Numerical_derivative(lambda G: P(F, G), order=2)._X(G)
        return P_G_num

    def bbP_F_num(self, F, G):
        bbP_F_num = Numerical_derivative(
            lambda F: self.bbP(F, G), order=num_order)._X(F)
        return bbP_F_num

    def bbP_G_num(self, F, G):
        bbP_G_num = Numerical_derivative(
            lambda G: self.bbP(F, G), order=num_order)._X(G)
        return bbP_G_num

    def scalar_F_num(self, F, G, var):
        return Numerical_derivative(lambda F: var(F, G), order=num_order)._X(F)

def verify_derivatives():
    K = np.random.rand(6)*50
    KE, KS, KG, KN, KT, KC = K
    mat = Pantobox_beam_network(KE*0, KS*0, KG*0, KN*0, KT, KC*0, numerical_derivative=False)
    F = np.random.rand(3, 3)
    G = np.random.rand(3, 3, 3)
    # mat.P_F_alt(F, G)
    # mat.var(F, G)
    # Pe = mat.Pe(F, G)
    # Pe_num = mat.Pe_num(F, G)
    # P1 = mat.P_F(F, G)
    # P1_num = mat.P_F_num(F, G, P=mat.P)
    P1 = mat.bbP_G(F, G)
    P1_num = mat.bbP_G_num(F, G)
    error = np.linalg.norm(P1 - P1_num)
    # print("%s\n\n%s" % (P1[0, 0], P1_num[0, 0]))
    # print(P1)
    # print(P1_num)
    print("%s" % error)


def speed_test():
    import time
    mat = Pantobox_beam_network(1, 1, 1, 1, 1, 1)
    F = np.random.rand(3, 3)
    G = np.random.rand(3, 3, 3)
    # t0 = time.time()
    # # P1 = mat.P_G_num(F, G, P=mat.P)
    # P1 = mat.P_F(F, G)
    # t1 = time.time()
    # P2 = mat.P_F_num(F, G)
    # t2 = time.time()
    # print(t1-t0, t2-t1)
    # t0 = time.time()
    # for i in range(100000):
    #     #np.outer(np.array([1,2, 3]), np.array([4, 5, 6]) ) / 10
    #     np.einsum('i,j,k->ijk',np.array([4.1, 5.2, 6.3]),np.array([4.4, 5.5, 6.6]),np.array([4.7, 5.8, 6.9])) / 10.7
    # t1 = time.time()
    # for i in range(100000):
    #     # np.outer(np.array([1,2, 3]), np.array([4, 5, 6]) / 10)
    #     np.einsum('i,j,k->ijk',np.array([4.1, 5.2, 6.3]),np.array([4.4, 5.5, 6.6]),np.array([4.7, 5.8, 6.9])/ 10.7)
    # t2 = time.time()

    # print(t1-t0, t2-t1)


if __name__ == "__main__":
    verify_derivatives()
    # speed_test()
