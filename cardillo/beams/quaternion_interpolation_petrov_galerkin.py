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
        q0, q = np.array_split(Q, [1])
        q_tilde = ax2skew(q)
        Q2 = Q @ Q
        A_IK = np.eye(3, dtype=Q.dtype) + (2 / Q2) * (q0 * q_tilde + q_tilde @ q_tilde)

        # axial and shear strains
        K_Gamma_bar = A_IK.T @ r_OP_xi

        # curvature, Rucker2018 (17)
        K_Kappa_bar = (
            (2 / Q2) * np.hstack((-q[:, None], q0 * np.eye(3) - ax2skew(q))) @ Q_xi
        )

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
