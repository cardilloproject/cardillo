# from abc import ABC, abstractmethod
import numpy as np
from math import sqrt, log, isclose
from cardillo.math.algebra import determinant2D, determinant3D, inverse3D, norm, norm3, cross3
from cardillo.math.numerical_derivative import Numerical_derivative

num_order = 2

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
        self.I3 = np.eye(3)

    def rho_F(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2

        rho1_F = np.outer(e1, self.D1)

        return rho1_F

    def rho_F_F(self, F, G):
        rho_F_F = np.einsum('ijk,l->iljk', self.e_F(F, G, 0), self.D1)
        return rho_F_F

    def e(self, F, G, al):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2

        return e1

    def e_F(self, F, G, al):
        """ derivative of tangent vectors w.r.t. to F
        """
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2
        e_F = np.einsum('ij,k->ijk', self.I3, self.D[al]) / rho1 \
            - np.einsum('i,j,k->ijk', e1,  e1, self.D[al]) / rho1

        return e_F

    def e_F_F(self, F, G, al=0):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2

        e_F_F = - np.einsum('ijk,lm->ijklm', self.e_F(F, G, 0), self.rho_F(F, G)) / rho1 \
                - np.einsum('ijk,lm->ilmjk', self.e_F(F, G, 0), self.rho_F(F, G)) / rho1 \
                - np.einsum('i,jklm->ijklm', e1, self.rho_F_F(F, G)) / rho1

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
                
        ga_F = np.outer(self.m[0], self.D1) / self.rho[0] - np.outer(self.m[1], self.D2) / rho[1]

        return ga_F

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

        return m1

    def m_F(self, F, G):

        m_F = np.einsum('i,jk->ijk', tanga * m1 - e1, self.ga_F(F, G)) \
                + self.e_F(F, G, 1) / cosga - tanga * self.e_F(F, G, 0)

        return m_F

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

        na_F = np.einsum('i,jk->ijk', tanga * na, self.ga_F(F, G)) \
                + np.cross(self.e_F(F, G, 0), e2, axisa=0, axisc=0) / cosga \
                 - np.cross(self.e_F(F, G, 1), e1, axisa=0, axisc=0) / cosga
        return na_F

    def c_F(self, F, G):
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

        c_F = - np.einsum('i,j,k', c1, e1, self.D1) / rho1

        return c_F

    def c(self, F, G):
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

        return c1

    def g_F(sefl, F, G, al=0):
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

        g_F = - np.einsum('i,j,k', g1, e2, self.D2) / rho2

        return g_F


    def v_F_num(self, F, G, v=e_F):
        v_F_num = Numerical_derivative(lambda F: v(F, G, 0), order=2)._X(F)
        return v_F_num

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
        cosgc = norm(cross3(na, nb))

        # Elongation
        Pe = self.Ke * ( (rho1-1) * np.outer(e1, self.D1) + (rho2-1) * np.outer(e2, self.D2) \
                    + (rho3-1) * np.outer(e3, self.D3) + (rho4-1) * np.outer(e4, self.D4))

        # Shear
        PsaD1 = m1 / rho1
        PsaD2 = -m2 / rho2
        PsbD3 = m3 / rho3
        PsbD4 = -m4 / rho4
        Psa = self.Ks * np.arcsin(singa) * (np.outer(PsaD1, self.D1) + np.outer(PsaD2, self.D2))
        Psb = self.Ks * np.arcsin(singb) * (np.outer(PsbD3, self.D3) + np.outer(PsbD4, self.D4))
        
        PcD1 = (singc * tanga * m1 + cross3(e1, cross3(cross3(e2, nb),e1)) / cosga) / rho1
        PcD2 = (-singc * tanga * m2 - cross3(e2, cross3(cross3(e1, nb),e2)) / cosga) / rho2
        PcD3 = (singc * tangb * m3 + cross3(e3, cross3(cross3(e4, na), e3)) / cosgb) / rho3
        PcD4 = (-singc * tangb * m4 - cross3(e4, cross3(cross3(e3, na), e4)) / cosgb) / rho4
        Pc = self.Kc * np.arcsin(singc) / cosgc * (np.outer(PcD1, self.D1) + np.outer(PcD2, self.D2) \
            + np.outer(PcD3, self.D3) + np.outer(PcD4, self.D4))

        # Normal curvature
        Pn1D1 = ((kn1 * tanga + e1 @ c1) * m1 \
             + tanga * cross3(e1, cross3(c1, e1)) - kn1 * e1) / rho1
        Pn1D2 = ((-kn1 * singa  - cosga * e1 @ c1) * m2 \
             - cross3(e2, cross3(c1, e2))) / (rho2 * cosga)

        Pn2D1 = ( (kn2 * singa  - cosga * e2 @ c2) * m1 \
             + cross3(e1, cross3(c2, e1)) ) / (cosga * rho1)  
        Pn2D2 = (( - kn2 * tanga + e2 @ c2) * m2 \
                - tanga * cross3(e2, cross3(c2, e2)) - kn2 * e2) / rho2  

        Pn3D3 = ((kn3 * tangb + e3 @ c3) * m3 \
             + tangb * cross3(e3, cross3(c3, e3)) - kn3 * e3) / rho3
        Pn3D4 = ((-kn3 * singb  - cosgb * e3 @ c3) * m4 \
             - cross3(e4, cross3(c3, e4))) / (rho4 * cosgb)

        Pn4D3 = ( (kn4 * singb  - cosgb * e4 @ c4) * m3 \
             + cross3(e3, cross3(c4, e3)) ) / (cosgb * rho3)  
        Pn4D4 = (( - kn4 * tangb + e4 @ c4) * m4 \
                - tangb * cross3(e4, cross3(c4, e4)) - kn4 * e4) / rho4  

        Pn = self.Kn * ((np.outer(kn1*Pn1D1 + kn2*Pn2D1, self.D1) \
                        + np.outer(kn1*Pn1D2 + kn2*Pn2D2, self.D2)) \
                            +np.outer(kn3*Pn3D3 + kn4*Pn4D3, self.D3)) \
                                +np.outer(kn3*Pn3D4 + kn4*Pn4D4, self.D4)

        # Geodesic curvature
        Pg1D1 = (kg1 * singa * m1 + cross3(e1, cross3(cross3(e2, c1), e1)) \
                    - cosga * kg1 * e1) / (rho1 * cosga)
        Pg1D2 = ( - kg1 * singa * m2 - cross3(e2, cross3(cross3(e1, c1), e2))) / (rho2 * cosga)

        Pg2D1 = (kg2 * singa * m1 + cross3(e1, cross3(cross3(e2, c2), e1))) / (rho1 * cosga)
        Pg2D2 = ( - kg2 * singa * m2 - cross3(e2, cross3(cross3(e1, c2), e2)) \
                  - cosga * kg2 * e2 )/ (rho2 * cosga)

        Pg3D3 = (kg3 * singb * m3 + cross3(e3, cross3(cross3(e4, c3), e3)) \
                    - cosgb * kg3 * e3) / (rho3 * cosgb)
        Pg3D4 = ( - kg3 * singb * m4 - cross3(e4, cross3(cross3(e3, c3), e4))) / (rho4 * cosgb)

        Pg4D3 = (kg4 * singb * m3 + cross3(e3, cross3(cross3(e4, c4), e3))) / (rho3 * cosgb)
        Pg4D4 = ( - kg4 * singb * m4 - cross3(e4, cross3(cross3(e3, c4), e4)) \
                  - cosgb * kg4 * e4 )/ (rho4 * cosgb)


        Pg = self.Kg * ((np.outer(kg1*Pg1D1 + kg2*Pg2D1, self.D1) \
                        + np.outer(kg1*Pg1D2 + kg2*Pg2D2, self.D2)) \
                            +np.outer(kg3*Pg3D3 + kg4*Pg4D3, self.D3)) \
                                +np.outer(kg3*Pg3D4 + kg4*Pg4D4, self.D4)

        # Torsion
        Pt1D1 = ((2 * tau1 * singa + kg1 * cosga) * m1 - kg1 * singa * e1 \
                    - cross3(e1, cross3(cross3(e2, g1 - singa*c1), e1)) / cosga) / (rho1 *cosga)
        Pt1D2 = ((-2 * tau1 * singa - kg1 * cosga) * m2 + (g1@na) * e2 \
                 + cross3(e2, cross3(cross3(e1, g1-singa*c1), e2)) / cosga) / (rho2 * cosga)

        Pt2D1 = ((2 * tau2 * singa + kg2 * cosga) * m1 + (g2@na) * e1\
                    - cross3(e1, cross3(cross3(e2, g2 - singa*c2), e1)) / cosga) / (rho1 *cosga)
        Pt2D2 = ((-2 * tau2 * singa - kg2 * cosga) * m2 - kg2 * singa * e2 \
                 + cross3(e2, cross3(cross3(e1, g2-singa*c2), e2)) / cosga) / (rho2 * cosga)

        Pt3D3 = ((2 * tau3 * singb + kg3 * cosgb) * m3 - kg3 * singb * e3 \
                    - cross3(e3, cross3(cross3(e4, g3 - singb*c3), e3)) / cosgb) / (rho3 *cosgb)
        Pt3D4 = ((-2 * tau3 * singb - kg3 * cosgb) * m4 + (g3@nb) * e4 \
                 + cross3(e4, cross3(cross3(e3, g3-singb*c3), e4)) / cosgb) / (rho4 * cosgb)

        Pt4D3 = ((2 * tau4 * singb + kg4 * cosgb) * m3 + (g4@nb) * e3\
                    - cross3(e3, cross3(cross3(e4, g4 - singb*c4), e3)) / cosgb) / (rho3 *cosgb)
        Pt4D4 = ((-2 * tau4 * singb - kg4 * cosgb) * m4 - kg4 * singb * e4 \
                 + cross3(e4, cross3(cross3(e3, g4-singb*c4), e4)) / cosgb) / (rho4 * cosgb)


        Pt = self.Kt * ( np.outer(tau1*Pt1D1 + tau2*Pt2D1, self.D1) \
                        + np.outer(tau1*Pt1D2 + tau2*Pt2D2, self.D2) \
                            + np.outer(tau3*Pt3D3 + tau4*Pt4D3, self.D3) \
                                + np.outer(tau3*Pt3D4 + tau4*Pt4D4, self.D4) )

        return Pe + Psa + Psb + Pc + Pn + Pg + Pt

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
        bbPg1 = self.Kg * kg1 / rho1 * np.einsum('i,j,k->ijk',na, self.D1, self.D1)
        bbPg2 = self.Kg * kg2 / rho2 * np.einsum('i,j,k->ijk',na,self.D2, self.D2)
        bbPg3 = self.Kg * kg3 / rho3 * np.einsum('i,j,k->ijk',nb,self.D3, self.D3)
        bbPg4 = self.Kg * kg4 / rho4 * np.einsum('i,j,k->ijk',nb,self.D4, self.D4)

        bbPg = bbPg1 + bbPg2 + bbPg3 + bbPg4 

        # normal
        bbPn1 = - self.Kn * kn1 / rho1 * np.einsum('i,j,k->ijk',m1, self.D1, self.D1)
        bbPn2 = -self.Kn * kn2 / rho2 * np.einsum('i,j,k->ijk',m2,self.D2, self.D2)
        bbPn3 = -self.Kn * kn3 / rho3 * np.einsum('i,j,k->ijk',m3,self.D3, self.D3)
        bbPn4 = -self.Kn * kn4 / rho4 * np.einsum('i,j,k->ijk',m4,self.D4, self.D4)

        bbPn = bbPn1 + bbPn2 + bbPn3 + bbPn4

        # torsion
        bbPt1 = self.Kt * tau1 * ( tanga * np.einsum('i,j,k->ijk', na, self.D1, self.D1) / rho1 \
                                -  np.einsum('i,j,k->ijk', na, self.D2, self.D1) / (rho2 * cosga) )
        bbPt2 = self.Kt * tau2 * ( tanga * np.einsum('i,j,k->ijk', na, self.D2, self.D2) / rho2 \
                                -  np.einsum('i,j,k->ijk', na, self.D1, self.D2) / (rho1 * cosga) )
        bbPt3 = self.Kt * tau3 * ( tangb * np.einsum('i,j,k->ijk', nb, self.D3, self.D3) / rho3 \
                                -  np.einsum('i,j,k->ijk', nb, self.D4, self.D3) / (rho4 * cosgb) )
        bbPt4 = self.Kt * tau4 * ( tangb * np.einsum('i,j,k->ijk', nb, self.D4, self.D4) / rho4 \
                                -  np.einsum('i,j,k->ijk', nb, self.D3, self.D4) / (rho3 * cosgb) )

        bbPt = bbPt1 + bbPt2 + bbPt3 + bbPt4

        return bbPg + bbPn + bbPt

    def P_F(self, F, G):
        P_F_num = Numerical_derivative(lambda F: self.P(F, G), order=num_order)._X(F)
        return P_F_num

    def P_G(self, F, G):
        P_G_num = Numerical_derivative(lambda G: self.P(F, G), order=num_order)._X(G)
        return P_G_num

    def bbP_F(self, F, G):
        bbP_F_num = Numerical_derivative(lambda F: self.bbP(F, G), order=num_order)._X(F)
        return bbP_F_num

    def bbP_G(self, F, G):
        bbP_G_num = Numerical_derivative(lambda G: self.bbP(F, G), order=num_order)._X(G) 
        # bbP_G_num = np.zeros((3,3,3,3,3,3))
        return bbP_G_num       

    # separated deformation fields for export
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

    def We(self, F, G):

        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        # e1 = d1 / rho1
        # e2 = d2 / rho2
        d3 = F @ self.D3
        d4 = F @ self.D4
        rho3 = norm3(d3)
        rho4 = norm3(d4)
        # e3 = d3 / rho3
        # e4 = d4 / rho4
        We = .5 * self.Ke * ((rho1-1)**2 + (rho2-1)**2 + (rho3-1)**2 + (rho4-1)**2)
        return np.array(We)

    def Pe(self, F, G):

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

        Pe1 = self.Ke * (rho1-1) * np.outer(e1, self.D1)

        Pe = self.Ke * ( (rho1-1) * np.outer(e1, self.D1) + (rho2-1) * np.outer(e2, self.D2) \
                    + (rho3-1) * np.outer(e3, self.D3) + (rho4-1) * np.outer(e4, self.D4))

        return Pe1

    def Wn(self, F, G):
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
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1  # same as G@D1@D1 speed is similar
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

        return Wn2 + Wn1 + Wn3 + Wn4

    def Pn(self, F, G):

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

        Pn1D1 = ((kn1 * tanga + e1 @ c1) * m1 \
             + tanga * cross3(e1, cross3(c1, e1)) - kn1 * e1) / rho1
        Pn1D2 = ((-kn1 * singa  - cosga * e1 @ c1) * m2 \
             - cross3(e2, cross3(c1, e2))) / (rho2 * cosga)

        Pn2D1 = ( (kn2 * singa  - cosga * e2 @ c2) * m1 \
             + cross3(e1, cross3(c2, e1)) ) / (cosga * rho1)  
        Pn2D2 = (( - kn2 * tanga + e2 @ c2) * m2 \
                - tanga * cross3(e2, cross3(c2, e2)) - kn2 * e2) / rho2  

        Pn3D3 = ((kn3 * tangb + e3 @ c3) * m3 \
             + tangb * cross3(e3, cross3(c3, e3)) - kn3 * e3) / rho3
        Pn3D4 = ((-kn3 * singb  - cosgb * e3 @ c3) * m4 \
             - cross3(e4, cross3(c3, e4))) / (rho4 * cosgb)

        Pn4D3 = ( (kn4 * singb  - cosgb * e4 @ c4) * m3 \
             + cross3(e3, cross3(c4, e3)) ) / (cosgb * rho3)  
        Pn4D4 = (( - kn4 * tangb + e4 @ c4) * m4 \
                - tangb * cross3(e4, cross3(c4, e4)) - kn4 * e4) / rho4  

        Pn = self.Kn * ((np.outer(kn1*Pn1D1 + kn2*Pn2D1, self.D1) \
                        + np.outer(kn1*Pn1D2 + kn2*Pn2D2, self.D2)) \
                            +np.outer(kn3*Pn3D3 + kn4*Pn4D3, self.D3)) \
                                +np.outer(kn3*Pn3D4 + kn4*Pn4D4, self.D4)

        return Pn

    def Wg(self, F, G):
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
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1  # same as G@D1@D1 speed is similar
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

    def Pg(self, F, G):
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
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1  # same as G@D1@D1 speed is similar
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

        Pg1D1 = (kg1 * singa * m1 + cross3(e1, cross3(cross3(e2, c1), e1)) \
                    - cosga * kg1 * e1) / (rho1 * cosga)
        Pg1D2 = ( - kg1 * singa * m2 - cross3(e2, cross3(cross3(e1, c1), e2))) / (rho2 * cosga)

        Pg2D1 = (kg2 * singa * m1 + cross3(e1, cross3(cross3(e2, c2), e1))) / (rho1 * cosga)
        Pg2D2 = ( - kg2 * singa * m2 - cross3(e2, cross3(cross3(e1, c2), e2)) \
                  - cosga * kg2 * e2 )/ (rho2 * cosga)

        Pg3D3 = (kg3 * singb * m3 + cross3(e3, cross3(cross3(e4, c3), e3)) \
                    - cosgb * kg3 * e3) / (rho3 * cosgb)
        Pg3D4 = ( - kg3 * singb * m4 - cross3(e4, cross3(cross3(e3, c3), e4))) / (rho4 * cosgb)

        Pg4D3 = (kg4 * singb * m3 + cross3(e3, cross3(cross3(e4, c4), e3))) / (rho3 * cosgb)
        Pg4D4 = ( - kg4 * singb * m4 - cross3(e4, cross3(cross3(e3, c4), e4)) \
                  - cosgb * kg4 * e4 )/ (rho4 * cosgb)


        Pg2 = self.Kg * kg2 * (np.outer(Pg2D1, self.D1) + np.outer(Pg2D2, self.D2))

        Pg1 = self.Kg * kg1 * (np.outer(Pg1D1, self.D1) + np.outer(Pg1D2, self.D2))

        Pg3 = self.Kg * kg3 * (np.outer(Pg3D3, self.D3) + np.outer(Pg3D4, self.D4))

        Pg4 = self.Kg * kg4 * (np.outer(Pg4D3, self.D3) + np.outer(Pg4D4, self.D4))

        return Pg1 + Pg2 + Pg3 + Pg4

    def Wt(self, F, G):
        # torisonl energy density
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
        c1 = np.einsum('ijk,j,k->i', G, self.D1, self.D1) / rho1  # same as G@D1@D1 speed is similar
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

    def Pt(self, F, G):
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

        Pt1D1 = ((2 * tau1 * singa + kg1 * cosga) * m1 - kg1 * singa * e1 \
                    - cross3(e1, cross3(cross3(e2, g1 - singa*c1), e1)) / cosga) / (rho1 *cosga)
        Pt1D2 = ((-2 * tau1 * singa - kg1 * cosga) * m2 + (g1@na) * e2 \
                 + cross3(e2, cross3(cross3(e1, g1-singa*c1), e2)) / cosga) / (rho2 * cosga)

        Pt2D1 = ((2 * tau2 * singa + kg2 * cosga) * m1 + (g2@na) * e1\
                    - cross3(e1, cross3(cross3(e2, g2 - singa*c2), e1)) / cosga) / (rho1 *cosga)
        Pt2D2 = ((-2 * tau2 * singa - kg2 * cosga) * m2 - kg2 * singa * e2 \
                 + cross3(e2, cross3(cross3(e1, g2-singa*c2), e2)) / cosga) / (rho2 * cosga)

        Pt3D3 = ((2 * tau3 * singb + kg3 * cosgb) * m3 - kg3 * singb * e3 \
                    - cross3(e3, cross3(cross3(e4, g3 - singb*c3), e3)) / cosgb) / (rho3 *cosgb)
        Pt3D4 = ((-2 * tau3 * singb - kg3 * cosgb) * m4 + (g3@nb) * e4 \
                 + cross3(e4, cross3(cross3(e3, g3-singb*c3), e4)) / cosgb) / (rho4 * cosgb)

        Pt4D3 = ((2 * tau4 * singb + kg4 * cosgb) * m3 + (g4@nb) * e3\
                    - cross3(e3, cross3(cross3(e4, g4 - singb*c4), e3)) / cosgb) / (rho3 *cosgb)
        Pt4D4 = ((-2 * tau4 * singb - kg4 * cosgb) * m4 - kg4 * singb * e4 \
                 + cross3(e4, cross3(cross3(e3, g4-singb*c4), e4)) / cosgb) / (rho4 * cosgb)


        Pt1 = self.Kt * tau1 * (np.outer(Pt1D1, self.D1) + np.outer(Pt1D2, self.D2))
        Pt2 = self.Kt * tau2 * (np.outer(Pt2D1, self.D1) + np.outer(Pt2D2, self.D2))
        Pt3 = self.Kt * tau3 * (np.outer(Pt3D3, self.D3) + np.outer(Pt3D4, self.D4))
        Pt4 = self.Kt * tau4 * (np.outer(Pt4D3, self.D3) + np.outer(Pt4D4, self.D4))

        return Pt1 + Pt2 + Pt3 + Pt4

    def Ws(self, F, G):
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

    def Ps(self, F, G):
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

        Psa = self.Ks * np.arcsin(singa) * (np.outer(PsaD1, self.D1) + np.outer(PsaD2, self.D2))
        Psb = self.Ks * np.arcsin(singb) * (np.outer(PsbD3, self.D3) + np.outer(PsbD4, self.D4))

        return Psa + Psb

    def Wc(self, F, G):
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

    def Pc(self, F, G):
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

        PcD1 = (singc * tanga * m1 + cross3(e1, cross3(cross3(e2, nb),e1)) / cosga) / rho1
        PcD2 = (-singc * tanga * m2 - cross3(e2, cross3(cross3(e1, nb),e2)) / cosga) / rho2
        PcD3 = (singc * tangb * m3 + cross3(e3, cross3(cross3(e4, na), e3)) / cosgb) / rho3
        PcD4 = (-singc * tangb * m4 - cross3(e4, cross3(cross3(e3, na), e4)) / cosgb) / rho4

        Pc = self.Kc * np.arcsin(singc) / cosgc * (np.outer(PcD1, self.D1) + np.outer(PcD2, self.D2) \
            + np.outer(PcD3, self.D3) + np.outer(PcD4, self.D4))

        return Pc

    def P_num(self, F, G, W=W):
        P_num = Numerical_derivative(lambda F: W(F, G), order=2)._X(F)
        return P_num

    def bbP_num(self, F, G, W=W):
        bbP_num = Numerical_derivative(lambda G: W(F, G), order=2)._X(G)
        return bbP_num

    # second variation
    def Pe_F(self, F, G):
        d1 = F @ self.D1
        d2 = F @ self.D2
        rho1 = norm3(d1)
        rho2 = norm3(d2)
        e1 = d1 / rho1
        e2 = d2 / rho2

        Pe1_F = self.Ke * 1 / rho1 * np.einsum('i,j,k,l->ijkl', e1, self.D1 ,e1 ,self.D1) \
                + self.Ke * (rho1-1) / rho1 * np.einsum('i,jk,l->jikl', self.D1, np.eye(3), self.D1)

        Pe1_F = self.Ke *  np.einsum('i,j,k,l->ijkl', e1, self.D1 ,e1 ,self.D1) \
                + self.Ke * np.einsum('i,jkl->jikl', self.D1, e_F(F, G))

        return Pe1_F

    def P_F_num(self, F, G, P=P):
        P_F_num = Numerical_derivative(lambda F: P(F, G), order=2)._X(F)
        return P_F_num

def verify_derivatives():
    mat = Pantobox_beam_network(1,1,1,1,1,1)
    F = np.random.rand(3,3)
    G = np.random.rand(3,3,3)
    # Pe = mat.Pe(F, G)
    # Pe_num = mat.Pe_num(F, G)
    P1 = mat.e_F_F(F, G)
    P1_num = mat.v_F_num(F, G, v=mat.e_F)    
    error = np.linalg.norm(P1 - P1_num)
    print("%s\n\n%s" %(P1,P1_num))
    # print(P1 - P1_num)
    print("%s" % error)
if __name__ == "__main__":
    verify_derivatives()




