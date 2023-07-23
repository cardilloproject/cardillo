import numpy as np

from cardillo.math import ax2skew, Exp_SO3_quat, approx_fprime
from cardillo.beams._base_petrov_galerkin import (
    K_TimoshenkoPetrovGalerkinBaseQuaternion,
)


class K_PetrovGalerkinQuaternionInterpolation(K_TimoshenkoPetrovGalerkinBaseQuaternion):
    def __init__(
        self,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree,
        nelement,
        Q,
        q0=None,
        u0=None,
        basis="Lagrange",
    ):
        p = polynomial_degree
        nquadrature = p
        nquadrature_dyn = int(np.ceil((p + 1) ** 2 / 2))

        super().__init__(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            polynomial_degree,
            polynomial_degree,
            nelement,
            nquadrature,
            nquadrature_dyn,
            Q,
            q0=q0,
            u0=u0,
            basis_r=basis,
            basis_psi=basis,
        )

    @staticmethod
    def straight_configuration(
        polynomial_degree,
        basis,
        nelement,
        L,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
    ):
        return K_TimoshenkoPetrovGalerkinBaseQuaternion.straight_configuration(
            polynomial_degree, polynomial_degree, basis, basis, nelement, L, r_OP, A_IK
        )

    @staticmethod
    def straight_initial_configuration(
        polynomial_degree,
        basis,
        nelement,
        L,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
        v_P=np.zeros(3, dtype=float),
        K_omega_IK=np.zeros(3, dtype=float),
    ):
        return K_TimoshenkoPetrovGalerkinBaseQuaternion.straight_initial_configuration(
            polynomial_degree,
            polynomial_degree,
            basis,
            basis,
            nelement,
            L,
            r_OP,
            A_IK,
            v_P,
            K_omega_IK,
        )

    def _eval(self, qe, xi):
        # evaluate shape functions
        N_r, N_r_xi = self.basis_functions_r(xi)

        # interpolate
        r_OP = np.zeros(3, dtype=qe.dtype)
        r_OP_xi = np.zeros(3, dtype=qe.dtype)
        Q = np.zeros(4, dtype=float)
        Q_xi = np.zeros(4, dtype=float)
        for node in range(self.nnodes_element_r):
            r_OP_node = qe[self.nodalDOF_element_r[node]]
            r_OP += N_r[node] * r_OP_node
            r_OP_xi += N_r_xi[node] * r_OP_node

            Q_node = qe[self.nodalDOF_element_psi[node]]
            Q += N_r[node] * Q_node
            Q_xi += N_r_xi[node] * Q_node

        # transformation matrix
        A_IK = Exp_SO3_quat(Q, normalize=True)

        # # derivative of transformation matrix w.r.t. xi
        # Q2 = Q @ Q
        # q0 = Q[0]
        # q = Q[1:]
        # q0_xi = Q_xi[0]
        # q_xi = Q_xi[1:]
        # q_tilde = ax2skew(q)
        # q_xi_tilde = ax2skew(q_xi)
        # A_IK_xi = (np.eye(3) - A_IK) * 2 * (Q @ Q_xi) / Q2 + (
        #     q_xi_tilde @ q_tilde
        #     + q_tilde @ q_xi_tilde
        #     + q0_xi * q_tilde
        #     + q0 * q_xi_tilde
        # ) * 2 / Q2

        # axial and shear strains
        K_Gamma_bar = A_IK.T @ r_OP_xi

        # # torsional and flexural strains
        # # TODO: Directly compute curvature from Q and Q_xi.
        # d1, d2, d3 = A_IK.T
        # d1_xi, d2_xi, d3_xi = A_IK_xi.T
        # K_Kappa_bar = np.array(
        #     [
        #         0.5 * (d3 @ d2_xi - d2 @ d3_xi),
        #         0.5 * (d1 @ d3_xi - d3 @ d1_xi),
        #         0.5 * (d2 @ d1_xi - d1 @ d2_xi),
        #     ]
        # )

        # curvature, Rucker2018 (17)
        q0 = Q[0]
        q = Q[1:]
        K_Kappa_bar = (
            2 * np.hstack((-q[:, None], q0 * np.eye(3) - ax2skew(q))) @ Q_xi / (Q @ Q)
        )

        # diff = K_Kappa_bar - K_Kappa_bar2
        # print(f"kappa error: {np.linalg.norm(diff)}")

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def A_IK(self, t, q, frame_ID):
        # evaluate shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate orientation
        A_IK = np.zeros((3, 3), dtype=q.dtype)
        for node in range(self.nnodes_element_psi):
            A_IK += N_psi[node] * self.RotationBase.Exp_SO3(
                q[self.nodalDOF_element_psi[node]]
            )

        return A_IK

    def A_IK_q(self, t, q, frame_ID):
        return approx_fprime(q, lambda q: self.A_IK(t, q, frame_ID))
