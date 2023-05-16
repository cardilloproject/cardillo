import numpy as np
from math import asin

from cardillo.math.algebra import norm
from cardillo.model.continuum import Pantographic_sheet, strain_measures
from cardillo.utility.coo_matrix import CooMatrix


class Pantographic_lattice(Pantographic_sheet):
    def W(self, rho, rho_s, Gamma, theta_s):
        W = 0
        for i in range(2):
            W += 0.5 * self.K_rho * (rho[i] - 1) ** 2
            W += 0.5 * self.K_Theta_s * theta_s[i, i] ** 2
        return W + 0.5 * self.K_Gamma * np.abs(Gamma) ** self.gamma

    # first derivatives
    def W_rho(self, rho, rho_s, Gamma, theta_s):
        return self.K_rho * (rho - 1)

    def W_Gamma(self, rho, rho_s, Gamma, theta_s):
        return (
            0.5
            * self.gamma
            * self.K_Gamma
            * np.sign(Gamma)
            * np.abs(Gamma) ** (self.gamma - 1)
        )

    def W_theta_s(self, rho, rho_s, Gamma, theta_s):
        return self.K_Theta_s * theta_s * np.eye(2)  # only s-s derivatives

    # second derivatives
    def W_rho_rho(self, rho, rho_s, Gamma, theta_s):
        return self.K_rho * np.eye(2)

    def W_Gamma_Gamma(self, rho, rho_s, Gamma, theta_s):
        return (
            0.5
            * self.gamma
            * self.K_Gamma
            * (self.gamma - 1)
            * np.abs(Gamma) ** (self.gamma - 2)
        )

    def W_theta_s_theta_s(self, rho, rho_s, Gamma, theta_s):
        W_theta_s_theta_s = np.zeros((2, 2, 2, 2))
        for i in range(2):
            W_theta_s_theta_s[i, i, i, i] = self.K_Theta_s
        return W_theta_s_theta_s

    def __init__(
        self,
        density,
        material_param,
        mesh,
        Z,
        z0=None,
        v0=None,
        cDOF=[],
        b=None,
        fiber_angle=np.pi / 4,
    ):
        super().__init__(
            density, mesh, Z, z0=z0, v0=v0, cDOF=cDOF, b=b, fiber_angle=fiber_angle
        )

        self.K_rho = material_param[0]  # [Nm^-1]
        self.K_Gamma = material_param[1]  # [Nm^-1]
        self.K_Theta_s = material_param[2]  # [Nm]
        self.gamma = material_param[3]
        self.assembler_callback_W(self.W)

    def f_pot_el(self, ze, el):
        f = np.zeros(self.nq_el)

        for i in range(self.nqp):
            N_Theta = self.N_Theta[el, i]
            N_ThetaTheta = self.N_ThetaTheta[el, i]
            w_J0 = self.w_J0[el, i]

            # first deformation gradient
            F = np.zeros((self.dim, self.dim))
            for a in range(self.nn_el):
                F += np.outer(ze[self.nodalDOF[a]], N_Theta[a])  # Bonet 1997 (7.5)

            # second deformation gradient
            G = np.zeros((self.dim, self.dim, self.dim))
            for a in range(self.nn_el):
                G += np.einsum(
                    "i,jk->ijk", ze[self.nodalDOF[a]], N_ThetaTheta[a]
                )  # TODO: reference to Evan's thesis

            # strain measures of pantographic sheet
            d1 = F[:, 0]
            d2 = F[:, 1]
            rho1 = norm(d1)
            rho2 = norm(d2)
            rho = np.array([rho1, rho2])

            e1 = d1 / rho1
            e2 = d2 / rho2

            d1_1 = G[:, 0, 0]
            d1_2 = G[:, 0, 1]
            d2_1 = G[:, 1, 0]
            d2_2 = G[:, 1, 1]
            rho1_1 = d1_1 @ e1
            rho1_2 = d1_2 @ e1
            rho2_1 = d2_1 @ e2
            rho2_2 = d2_2 @ e2
            rho_s = np.array([[rho1_1, rho1_2], [rho2_1, rho2_2]])

            d1_perp = np.array([-d1[1], d1[0]])
            d2_perp = np.array([-d2[1], d2[0]])
            d1_1_perp = np.array([d1_1[1], -d1_1[0]])
            d2_2_perp = np.array([d2_2[1], -d2_2[0]])
            theta1_1 = d1_1 @ d1_perp / rho1**2
            theta1_2 = d1_2 @ d1_perp / rho1**2
            theta2_1 = d2_1 @ d2_perp / rho2**2
            theta2_2 = d2_2 @ d2_perp / rho2**2
            theta_s = np.array([[theta1_1, theta1_2], [theta2_1, theta2_2]])

            Gamma = asin(e2 @ e1)

            # evaluate material model (-> derivatives w.r.t. strain measures)
            W_rho = self.W_rho(rho, rho_s, Gamma, theta_s)
            W_Gamma = self.W_Gamma(rho, rho_s, Gamma, theta_s)
            W_theta_s = self.W_theta_s(rho, rho_s, Gamma, theta_s)

            # internal forces
            for a in range(self.nn_el):
                # delta rho_s
                rho1_q = N_Theta[a, 0] * e1
                rho2_q = N_Theta[a, 1] * e2

                # delta Gamma
                Gamma_q = (
                    1
                    / (1 - (e2 @ e1) ** 2) ** 0.5
                    * (
                        (d1 * N_Theta[a, 1] + d2 * N_Theta[a, 0]) / (rho1 * rho2)
                        - (d1 @ d2)
                        / (rho1 * rho2) ** 2
                        * (rho2 * rho1_q + rho1 * rho2_q)
                    )
                )

                theta1_1_q = (
                    d1_1_perp * N_Theta[a, 0] + d1_perp * N_ThetaTheta[a, 0, 0]
                ) / rho1**2 - (d1_1 @ d1_perp) / rho1**3 * 2 * rho1_q
                theta2_2_q = (
                    d2_2_perp * N_Theta[a, 1] + d2_perp * N_ThetaTheta[a, 1, 1]
                ) / rho2**2 - (d2_2 @ d2_perp) / rho2**3 * 2 * rho2_q

                f[self.nodalDOF[a]] -= (
                    W_rho[0] * rho1_q
                    + W_rho[1] * rho2_q
                    + W_Gamma * Gamma_q
                    + W_theta_s[0, 0] * theta1_1_q
                    + W_theta_s[1, 1] * theta2_2_q
                ) * w_J0

        return f

    def h(self, t, q, u):
        z = self.z(t, q)
        f_pot = np.zeros(self.nz)
        for el in range(self.nel):
            f_pot[self.elDOF[el]] += self.f_pot_el(z[self.elDOF[el]], el)
        return f_pot[self.fDOF]

    def f_pot_q_el(self, ze, el):
        Ke = np.zeros((self.nq_el, self.nq_el))

        for i in range(self.nqp):
            N_Theta = self.N_Theta[el, i]
            N_ThetaTheta = self.N_ThetaTheta[el, i]
            w_J0 = self.w_J0[el, i]

            # first deformation gradient
            F = np.zeros((self.dim, self.dim))
            for a in range(self.nn_el):
                F += np.outer(ze[self.nodalDOF[a]], N_Theta[a])  # Bonet 1997 (7.5)

            # second deformation gradient
            G = np.zeros((self.dim, self.dim, self.dim))
            for a in range(self.nn_el):
                G += np.einsum(
                    "i,jk->ijk", ze[self.nodalDOF[a]], N_ThetaTheta[a]
                )  # TODO: reference to Evan's thesis

            # strain measures of pantographic sheet
            d1 = F[:, 0]
            d2 = F[:, 1]
            rho1 = norm(d1)
            rho2 = norm(d2)
            rho = np.array([rho1, rho2])

            e1 = d1 / rho1
            e2 = d2 / rho2

            d1_1 = G[:, 0, 0]
            d1_2 = G[:, 0, 1]
            d2_1 = G[:, 1, 0]
            d2_2 = G[:, 1, 1]
            rho1_1 = d1_1 @ e1
            rho1_2 = d1_2 @ e1
            rho2_1 = d2_1 @ e2
            rho2_2 = d2_2 @ e2
            rho_s = np.array([[rho1_1, rho1_2], [rho2_1, rho2_2]])

            d1_perp = np.array([-d1[1], d1[0]])
            d2_perp = np.array([-d2[1], d2[0]])
            d1_1_perp = np.array([d1_1[1], -d1_1[0]])
            d2_2_perp = np.array([d2_2[1], -d2_2[0]])
            theta1_1 = d1_1 @ d1_perp / rho1**2
            theta1_2 = d1_2 @ d1_perp / rho1**2
            theta2_1 = d2_1 @ d2_perp / rho2**2
            theta2_2 = d2_2 @ d2_perp / rho2**2
            theta_s = np.array([[theta1_1, theta1_2], [theta2_1, theta2_2]])

            Gamma = asin(e2 @ e1)

            # precompute matrices
            G_rho1 = (np.eye(2) - np.outer(e1, e1)) / rho1
            G_rho2 = (np.eye(2) - np.outer(e2, e2)) / rho2

            tmp = (1 - (e1 @ e2) ** 2) ** 0.5
            G_Gamma11 = (
                e1 @ e2 * G_rho1 @ np.outer(e2, e2) @ G_rho1 / (1 - (e1 @ e2) ** 2)
                - (
                    np.outer(e1, e2)
                    + np.outer(e2, e1)
                    + (np.eye(2) - 3 * np.outer(e1, e1)) * (e1 @ e2)
                )
                / rho1**2
            ) / tmp
            G_Gamma12 = (
                e1 @ e2 * G_rho1 @ np.outer(e2, e1) @ G_rho2 / (1 - (e1 @ e2) ** 2)
                + G_rho1 @ G_rho2
            ) / tmp
            G_Gamma21 = (
                e1 @ e2 * G_rho2 @ np.outer(e1, e2) @ G_rho1 / (1 - (e1 @ e2) ** 2)
                + G_rho2 @ G_rho1
            ) / tmp
            G_Gamma22 = (
                e1 @ e2 * G_rho2 @ np.outer(e1, e1) @ G_rho2 / (1 - (e1 @ e2) ** 2)
                - (
                    np.outer(e1, e2)
                    + np.outer(e2, e1)
                    + (np.eye(2) - 3 * np.outer(e2, e2)) * (e1 @ e2)
                )
                / rho2**2
            ) / tmp
            G_theta11_1 = (
                -2
                * np.array([[0, -1], [1, 0]])
                @ (
                    np.outer(d1, d1_1)
                    + np.outer(d1_1, d1)
                    + (np.eye(2) - 4 * np.outer(e1, e1)) * (d1_1 @ d1)
                )
                / rho1**4
            )
            G_theta22_2 = (
                -2
                * np.array([[0, -1], [1, 0]])
                @ (
                    np.outer(d2, d2_2)
                    + np.outer(d2_2, d2)
                    + (np.eye(2) - 4 * np.outer(e2, e2)) * (d2_2 @ d2)
                )
                / rho2**4
            )
            G_theta11 = (
                np.array([[0, -1], [1, 0]]) @ (np.eye(2) - np.outer(e1, e1)) / rho1**2
            )
            G_theta22 = (
                np.array([[0, -1], [1, 0]]) @ (np.eye(2) - np.outer(e2, e2)) / rho2**2
            )

            # evaluate material model (-> derivatives w.r.t. strain measures)
            W_rho = self.W_rho(rho, rho_s, Gamma, theta_s)
            W_Gamma = self.W_Gamma(rho, rho_s, Gamma, theta_s)
            W_theta_s = self.W_theta_s(rho, rho_s, Gamma, theta_s)
            W_rho_rho = self.W_rho_rho(rho, rho_s, Gamma, theta_s)
            W_Gamma_Gamma = self.W_Gamma_Gamma(rho, rho_s, Gamma, theta_s)
            W_theta_s_theta_s = self.W_theta_s_theta_s(rho, rho_s, Gamma, theta_s)

            rho1_q = np.zeros((self.nq_el))
            rho2_q = np.zeros((self.nq_el))
            rho1_1_q = np.zeros((self.nq_el))
            rho2_2_q = np.zeros((self.nq_el))
            Gamma_q = np.zeros((self.nq_el))
            theta1_1_q = np.zeros((self.nq_el))
            theta2_2_q = np.zeros((self.nq_el))

            for a in range(self.nn_el):
                ndDOFa = self.mesh.nodalDOF[a]

                # delta rho_s
                rho1_q[ndDOFa] = N_Theta[a, 0] * e1
                rho2_q[ndDOFa] = N_Theta[a, 1] * e2

                # delta rho_s_s
                rho1_1_q[ndDOFa] = (
                    N_ThetaTheta[a, 0, 0] * e1 + N_Theta[a, 0] * G_rho1 @ d1_1
                )
                rho2_2_q[ndDOFa] = (
                    N_ThetaTheta[a, 1, 1] * e2 + N_Theta[a, 1] * G_rho2 @ d2_2
                )

                # delta Gamma
                Gamma_q[ndDOFa] = (
                    1
                    / (1 - (e2 @ e1) ** 2) ** 0.5
                    * (
                        (d1 * N_Theta[a, 1] + d2 * N_Theta[a, 0]) / (rho1 * rho2)
                        - (d1 @ d2)
                        / (rho1 * rho2) ** 2
                        * (rho2 * rho1_q[ndDOFa] + rho1 * rho2_q[ndDOFa])
                    )
                )

                # delta theta_s_s
                theta1_1_q[ndDOFa] = (
                    d1_1_perp * N_Theta[a, 0] + d1_perp * N_ThetaTheta[a, 0, 0]
                ) / rho1**2 - (d1_1 @ d1_perp) / rho1**3 * 2 * rho1_q[ndDOFa]
                theta2_2_q[ndDOFa] = (
                    d2_2_perp * N_Theta[a, 1] + d2_perp * N_ThetaTheta[a, 1, 1]
                ) / rho2**2 - (d2_2 @ d2_perp) / rho2**3 * 2 * rho2_q[ndDOFa]

            for a in range(self.nn_el):
                ndDOFa = self.mesh.nodalDOF[a]
                for b in range(self.nn_el):
                    ndDOFb = self.mesh.nodalDOF[b]

                    rho1_qq = N_Theta[a, 0] * G_rho1 * N_Theta[b, 0]
                    rho2_qq = N_Theta[a, 1] * G_rho2 * N_Theta[b, 1]

                    Gamma_qq = (
                        N_Theta[a, 0] * G_Gamma11 * N_Theta[b, 0]
                        + N_Theta[a, 0] * G_Gamma12 * N_Theta[b, 1]
                        + N_Theta[a, 1] * G_Gamma21 * N_Theta[b, 0]
                        + N_Theta[a, 1] * G_Gamma22 * N_Theta[b, 1]
                    )

                    theta1_1_qq = (
                        N_Theta[a, 0] * G_theta11_1 * N_Theta[b, 0]
                        + N_ThetaTheta[a, 0, 0] * G_theta11 * N_Theta[b, 0]
                        + N_Theta[a, 0] * G_theta11 * N_ThetaTheta[b, 0, 0]
                    )
                    theta2_2_qq = (
                        N_Theta[a, 1] * G_theta22_2 * N_Theta[b, 1]
                        + N_ThetaTheta[a, 1, 1] * G_theta22 * N_Theta[b, 1]
                        + N_Theta[a, 1] * G_theta22 * N_ThetaTheta[b, 1, 1]
                    )

                    Ke[np.ix_(ndDOFa, ndDOFb)] -= (
                        W_rho[0] * rho1_qq
                        + W_rho[1] * rho2_qq
                        + W_Gamma * Gamma_qq
                        + W_theta_s[0, 0] * theta1_1_qq
                        + W_theta_s[1, 1] * theta2_2_qq
                        + W_rho_rho[0, 0] * np.outer(rho1_q[ndDOFa], rho1_q[ndDOFb])
                        + W_rho_rho[1, 1] * np.outer(rho2_q[ndDOFa], rho2_q[ndDOFb])
                        + W_theta_s_theta_s[0, 0, 0, 0]
                        * np.outer(theta1_1_q[ndDOFa], theta1_1_q[ndDOFb])
                        + W_theta_s_theta_s[1, 1, 1, 1]
                        * np.outer(theta2_2_q[ndDOFa], theta2_2_q[ndDOFb])
                        + W_Gamma_Gamma * np.outer(Gamma_q[ndDOFa], Gamma_q[ndDOFb])
                    ) * w_J0

        return Ke

    def h_q(self, t, q, u):
        z = self.z(t, q)
        coo = CooMatrix((self.nu, self.nq))
        for el in range(self.nel):
            Ke = self.f_pot_q_el(z[self.elDOF[el]], el)

            # sparse assemble element internal stiffness matrix
            elfDOF = self.elfDOF[el]
            eluDOF = self.eluDOF[el]
            elqDOF = self.elqDOF[el]
            coo[eluDOF, elqDOF] = Ke[elfDOF[:, None], elfDOF]

        return coo
