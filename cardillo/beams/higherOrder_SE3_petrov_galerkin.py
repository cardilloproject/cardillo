import numpy as np
from cardillo.math import (
    SE3,
    SE3inv,
    Exp_SE3,
    Log_SE3,
    T_SE3,
    approx_fprime,
)
from cardillo.beams._base_petrov_galerkin import (
    K_TimoshenkoPetrovGalerkinBaseQuaternion,
)


class HigherOrder_K_SE3_PetrovGalerkin_Quaternion(
    K_TimoshenkoPetrovGalerkinBaseQuaternion
):
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
        symmetric_reference=True,
    ):
        p = polynomial_degree
        # TODO: Use full quadrature everywhere for locking investigations.
        nquadrature = p
        nquadrature_dyn = int(np.ceil((p + 1) ** 2 / 2))

        # warnings.warn("'HigherOrder_K_SE3_PetrovGalerkin_Quaternion: Full integration is used!")
        # nquadrature = nquadrature_dyn

        # reference nodes for relative twist, cf. Crisfield1999 (5.8)
        self.symmetric_reference = symmetric_reference
        if symmetric_reference:
            self.node_A = int(0.5 * (polynomial_degree + 2)) - 1
            self.node_B = int(0.5 * (polynomial_degree + 3)) - 1

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
            basis_r="Lagrange",
            basis_psi="Lagrange",
        )

    @staticmethod
    def straight_configuration(
        polynomial_degree,
        nelement,
        L,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
    ):
        return K_TimoshenkoPetrovGalerkinBaseQuaternion.straight_configuration(
            polynomial_degree,
            polynomial_degree,
            "Lagrange",
            "Lagrange",
            nelement,
            L,
            r_OP,
            A_IK,
        )

    @staticmethod
    def straight_initial_configuration(
        polynomial_degree,
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
            "Lagrange",
            "Lagrange",
            nelement,
            L,
            r_OP,
            A_IK,
            v_P,
            K_omega_IK,
        )

    def _reference_Euclidean_transformation(self, qe):
        """Reference Euclidean transformation, cf. Crisfield1999 (5.8)."""
        if self.symmetric_reference:
            r_OP_A = qe[self.nodalDOF_element_r[self.node_A]]
            P_A = qe[self.nodalDOF_element_psi[self.node_A]]
            A_IK_A = self.RotationBase.Exp_SO3(P_A)
            H_IK_A = SE3(A_IK_A, r_OP_A)

            r_OP_B = qe[self.nodalDOF_element_r[self.node_A]]
            P_B = qe[self.nodalDOF_element_psi[self.node_A]]
            A_IK_B = self.RotationBase.Exp_SO3(P_B)
            H_IK_B = SE3(A_IK_B, r_OP_B)

            return H_IK_A @ Exp_SE3(0.5 * Log_SE3(SE3inv(H_IK_A) @ H_IK_B))
        else:
            r_OP0 = qe[self.nodalDOF_element_r[self.node_A]]
            P0 = qe[self.nodalDOF_element_psi[self.node_A]]
            A_IK0 = self.RotationBase.Exp_SO3(P0)
            return SE3(A_IK0, r_OP0)

    def _eval(self, qe, xi):
        # evaluate shape functions
        N, N_xi = self.basis_functions_r(xi)

        # compute reference Euclidean transformation matrix
        H_IK_ref = self._reference_Euclidean_transformation(qe)
        H_IK_ref_inv = SE3inv(H_IK_ref)

        # interpolate relative twists
        theta_MK = np.zeros(6, dtype=qe.dtype)
        theta_MK_xi = np.zeros(6, dtype=qe.dtype)
        for i in range(self.nnodes_element_r):
            r_OPi = qe[self.nodalDOF_element_r[i]]
            Pi = qe[self.nodalDOF_element_psi[i]]
            A_IKi = self.RotationBase.Exp_SO3(Pi)
            H_IKi = SE3(A_IKi, r_OPi)
            theta_KMKi = Log_SE3(H_IK_ref_inv @ H_IKi)

            theta_MK += N[i] * theta_KMKi
            theta_MK_xi += N_xi[i] * theta_KMKi

        # Euclidean transformation matrix
        H_IK = H_IK_ref @ Exp_SE3(theta_MK)

        # centerline position and transformation matrix
        r_OP = H_IK[:-1, -1]
        A_IK = H_IK[:3, :3]

        # strain measures
        K_eps_bar = T_SE3(theta_MK) @ theta_MK_xi
        K_Gamma_bar, K_Kappa_bar = np.array_split(K_eps_bar, 2)

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def A_IK(self, t, q, frame_ID):
        return self._eval(q, frame_ID[0])[1]

    def A_IK_q(self, t, q, frame_ID):
        return approx_fprime(q, lambda q: self.A_IK(t, q, frame_ID))
