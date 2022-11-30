import numpy as np
from cardillo.math import (
    cross3,
    Exp_SO3,
    Exp_SO3_psi,
)
from cardillo.beams._base import TimoshenkoPetrovGalerkinBase


class DirectorAxisAngle(TimoshenkoPetrovGalerkinBase):
    def __init__(
        self,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree_r,
        polynomial_degree_psi,
        nelement,
        Q,
        q0=None,
        u0=None,
        basis_r="Lagrange",
        basis_psi="Lagrange",
    ):
        super().__init__(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            polynomial_degree_r,
            polynomial_degree_psi,
            nelement,
            Q,
            q0=q0,
            u0=u0,
            basis_r=basis_r,
            basis_psi=basis_psi,
        )

    def _eval(self, qe, xi):
        # evaluate shape functions
        N_r, N_r_xi = self.basis_functions_r(xi)
        N_psi, N_psi_xi = self.basis_functions_psi(xi)

        # interpolate position and tangent vector
        r_OP = np.zeros(3, dtype=qe.dtype)
        r_OP_xi = np.zeros(3, dtype=qe.dtype)
        for node in range(self.nnodes_element_r):
            r_OP_node = qe[self.nodalDOF_element_r[node]]
            r_OP += N_r[node] * r_OP_node
            r_OP_xi += N_r_xi[node] * r_OP_node

        # interpolate transformation matrix and its derivative
        A_IK = np.zeros((3, 3), dtype=qe.dtype)
        A_IK_xi = np.zeros((3, 3), dtype=qe.dtype)
        for node in range(self.nnodes_element_psi):
            A_IK_node = Exp_SO3(qe[self.nodalDOF_element_psi[node]])
            A_IK += N_psi[node] * A_IK_node
            A_IK_xi += N_psi_xi[node] * A_IK_node

        # axial and shear strains
        K_Gamma_bar = A_IK.T @ r_OP_xi

        # torsional and flexural strains
        d1, d2, d3 = A_IK.T
        d1_xi, d2_xi, d3_xi = A_IK_xi.T
        half_d = d1 @ cross3(d2, d3)
        K_Kappa_bar = (
            np.array(
                [
                    0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                    0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                    0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                ]
            )
            / half_d
        )

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def _deval(self, qe, xi):
        # evaluate shape functions
        N_r, N_r_xi = self.basis_functions_r(xi)
        N_psi, N_psi_xi = self.basis_functions_psi(xi)

        # interpolate position and tangent vector + their derivatives
        r_OP = np.zeros(3, dtype=qe.dtype)
        r_OP_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
        r_OP_xi = np.zeros(3, dtype=qe.dtype)
        r_OP_xi_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
        for node in range(self.nnodes_element_r):
            nodalDOF_r = self.nodalDOF_element_r[node]
            r_OP_node = qe[nodalDOF_r]

            r_OP += N_r[node] * r_OP_node
            r_OP_qe[:, nodalDOF_r] += N_r[node] * np.eye(3, dtype=float)

            r_OP_xi += N_r_xi[node] * r_OP_node
            r_OP_xi_qe[:, nodalDOF_r] += N_r_xi[node] * np.eye(3, dtype=float)

        # interpolate transformation matrix and its derivative + their derivatives
        A_IK = np.zeros((3, 3), dtype=qe.dtype)
        A_IK_xi = np.zeros((3, 3), dtype=qe.dtype)
        A_IK_qe = np.zeros((3, 3, self.nq_element), dtype=qe.dtype)
        A_IK_xi_qe = np.zeros((3, 3, self.nq_element), dtype=qe.dtype)
        for node in range(self.nnodes_element_psi):
            nodalDOF_psi = self.nodalDOF_element_psi[node]
            psi_node = qe[nodalDOF_psi]
            A_IK_node = Exp_SO3(psi_node)
            A_IK_q_node = Exp_SO3_psi(psi_node)

            A_IK += N_psi[node] * A_IK_node
            A_IK_qe[:, :, nodalDOF_psi] += N_psi[node] * A_IK_q_node

            A_IK_xi += N_psi_xi[node] * A_IK_node
            A_IK_xi_qe[:, :, nodalDOF_psi] += N_psi_xi[node] * A_IK_q_node

        # extract directors
        d1, d2, d3 = A_IK.T
        d1_xi, d2_xi, d3_xi = A_IK_xi.T
        d1_qe, d2_qe, d3_qe = A_IK_qe.transpose(1, 0, 2)
        d1_xi_qe, d2_xi_qe, d3_xi_qe = A_IK_xi_qe.transpose(1, 0, 2)

        # axial and shear strains
        K_Gamma_bar = A_IK.T @ r_OP_xi
        K_Gamma_bar_qe = np.einsum("k,kij", r_OP_xi, A_IK_qe) + A_IK.T @ r_OP_xi_qe

        # torsional and flexural strains
        half_d = d1 @ cross3(d2, d3)
        half_d_qe = (
            cross3(d2, d3) @ d1_qe + cross3(d3, d1) @ d2_qe + cross3(d1, d2) @ d3_qe
        )
        K_Kappa_bar = (
            np.array(
                [
                    0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                    0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                    0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                ]
            )
            / half_d
        )
        K_Kappa_bar_qe = np.array(
            [
                0.5 * (d3 @ d2_xi_qe + d2_xi @ d3_qe - d2 @ d3_xi_qe - d3_xi @ d2_qe),
                0.5 * (d1 @ d3_xi_qe + d3_xi @ d1_qe - d3 @ d1_xi_qe - d1_xi @ d3_qe),
                0.5 * (d2 @ d1_xi_qe + d1_xi @ d2_qe - d1 @ d2_xi_qe - d2_xi @ d1_qe),
            ]
        ) / half_d - np.outer(K_Kappa_bar / half_d, half_d_qe)

        return (
            r_OP,
            A_IK,
            K_Gamma_bar,
            K_Kappa_bar,
            r_OP_qe,
            A_IK_qe,
            K_Gamma_bar_qe,
            K_Kappa_bar_qe,
        )

    def A_IK(self, t, q, frame_ID):
        # evaluate shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate orientation
        A_IK = np.zeros((3, 3), dtype=q.dtype)
        for node in range(self.nnodes_element_psi):
            A_IK += N_psi[node] * Exp_SO3(q[self.nodalDOF_element_psi[node]])

        return A_IK

    def A_IK_q(self, t, q, frame_ID):
        # evaluate shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate centerline position and orientation
        A_IK_q = np.zeros((3, 3, self.nq_element), dtype=q.dtype)
        for node in range(self.nnodes_element_psi):
            nodalDOF_psi = self.nodalDOF_element_psi[node]
            A_IK_q[:, :, nodalDOF_psi] += N_psi[node] * Exp_SO3_psi(q[nodalDOF_psi])

        return A_IK_q
