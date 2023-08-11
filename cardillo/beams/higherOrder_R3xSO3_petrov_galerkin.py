import numpy as np
from cardillo.math import (
    Exp_SO3,
    Log_SO3,
    T_SO3,
    approx_fprime,
)
from cardillo.beams._base_petrov_galerkin import (
    K_TimoshenkoPetrovGalerkinBaseQuaternion,
)


class HigherOrder_K_R3xSO3_PetrovGalerkin_Quaternion(
    K_TimoshenkoPetrovGalerkinBaseQuaternion
):
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
        symmetric_reference=True,
        reduced_integration=True,
    ):
        polynomial_degree = max(polynomial_degree_r, polynomial_degree_psi)
        self.polynomial_degree = polynomial_degree
        nquadrature = polynomial_degree
        nquadrature_dyn = int(np.ceil((polynomial_degree + 1) ** 2 / 2))

        if not reduced_integration:
            import warnings

            warnings.warn(
                "'HigherOrder_K_R3xSO3_PetrovGalerkin_Quaternion': Full integration is used!"
            )
            nquadrature = nquadrature_dyn

        # reference nodes for relative rotation vector, see Crisfield1999 (5.8)
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
            polynomial_degree_r,
            polynomial_degree_psi,
            nelement,
            nquadrature,
            nquadrature_dyn,
            Q,
            q0=q0,
            u0=u0,
            basis_r=basis_r,
            basis_psi=basis_psi,
        )

    def _reference_transformation(self, qe):
        """Reference transformation, see Crisfield1999 (5.8)."""
        if self.symmetric_reference:
            P_A = qe[self.nodalDOF_element_psi[self.node_A]]
            A_IK_A = self.RotationBase.Exp_SO3(P_A)

            P_B = qe[self.nodalDOF_element_psi[self.node_A]]
            A_IK_B = self.RotationBase.Exp_SO3(P_B)

            return A_IK_A @ Exp_SO3(0.5 * Log_SO3(A_IK_A.T @ A_IK_B))
        else:
            P0 = qe[self.nodalDOF_element_psi[0]]
            A_IK0 = self.RotationBase.Exp_SO3(P0)
            return A_IK0

    def _eval(self, qe, xi):
        # evaluate shape functions
        N, N_xi = self.basis_functions_r(xi)

        # compute reference Euclidean transformation matrix
        A_IK_ref = self._reference_transformation(qe)

        # pth-order Lagrange interpolation of positions
        r_OP = np.zeros(3, dtype=qe.dtype)
        r_OP_xi = np.zeros(3, dtype=qe.dtype)
        for i in range(self.nnodes_element_r):
            r_OPi = qe[self.nodalDOF_element_r[i]]
            r_OP += N[i] * r_OPi
            r_OP_xi += N_xi[i] * r_OPi

        # interpolate relative rotation vectors
        psi_MK = np.zeros(3, dtype=qe.dtype)
        psi_MK_xi = np.zeros(3, dtype=qe.dtype)
        for i in range(self.nnodes_element_r):
            Pi = qe[self.nodalDOF_element_psi[i]]
            A_IKi = self.RotationBase.Exp_SO3(Pi)
            psi_KMKi = Log_SO3(A_IK_ref.T @ A_IKi)

            psi_MK += N[i] * psi_KMKi
            psi_MK_xi += N_xi[i] * psi_KMKi

        # transformation matrix
        A_IK = A_IK_ref @ Exp_SO3(psi_MK)

        # strain measures
        K_Gamma_bar = A_IK.T @ r_OP_xi
        if self.polynomial_degree > 1:
            K_Kappa_bar = T_SO3(psi_MK) @ psi_MK_xi
        else:
            K_Kappa_bar = psi_MK_xi

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def A_IK(self, t, q, frame_ID):
        return self._eval(q, frame_ID[0])[1]

    def A_IK_q(self, t, q, frame_ID):
        return approx_fprime(q, lambda q: self.A_IK(t, q, frame_ID))
