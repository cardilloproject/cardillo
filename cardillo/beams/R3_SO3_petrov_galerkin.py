import numpy as np

from cardillo.math import Exp_SO3, Log_SO3, T_SO3, approx_fprime
from cardillo.beams._base_petrov_galerkin import (
    I_TimoshenkoPetrovGalerkinBaseAxisAngle,
    K_TimoshenkoPetrovGalerkinBaseAxisAngle,
    I_TimoshenkoPetrovGalerkinBaseQuaternion,
    K_TimoshenkoPetrovGalerkinBaseQuaternion,
    I_TimoshenkoPetrovGalerkinBaseR9,
    K_TimoshenkoPetrovGalerkinBaseR9,
)


# TODO: implement _deval and A_IK_q
def __make_R3_SO3(Base):
    class Derived(Base):
        def __init__(
            self,
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            polynomial_degree_r,
            nelement,
            Q,
            q0=None,
            u0=None,
            basis_r="Lagrange",
        ):
            p = polynomial_degree_r
            nquadrature = p
            nquadrature_dyn = int(np.ceil((p + 1) ** 2 / 2))

            super().__init__(
                cross_section,
                material_model,
                A_rho0,
                K_S_rho0,
                K_I_rho0,
                polynomial_degree_r,
                1,
                nelement,
                nquadrature,
                nquadrature_dyn,
                Q,
                q0=q0,
                u0=u0,
                basis_r=basis_r,
                basis_psi="Lagrange",
            )

        @staticmethod
        def straight_configuration(
            polynomial_degree_r,
            basis_r,
            nelement,
            L,
            r_OP=np.zeros(3, dtype=float),
            A_IK=np.eye(3, dtype=float),
        ):
            return Base.straight_configuration(
                polynomial_degree_r, 1, basis_r, "Lagrange", nelement, L, r_OP, A_IK
            )

        @staticmethod
        def straight_initial_configuration(
            polynomial_degree_r,
            basis_r,
            nelement,
            L,
            r_OP=np.zeros(3, dtype=float),
            A_IK=np.eye(3, dtype=float),
            v_P=np.zeros(3, dtype=float),
            K_omega_IK=np.zeros(3, dtype=float),
        ):
            return Base.straight_initial_configuration(
                polynomial_degree_r,
                1,
                basis_r,
                "Lagrange",
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
            N_psi, N_psi_xi = self.basis_functions_psi(xi)

            # interpolate centerline and tangent vector
            r_OP = np.zeros(3, dtype=qe.dtype)
            r_OP_xi = np.zeros(3, dtype=qe.dtype)
            for node in range(self.nnodes_element_r):
                r_OP_node = qe[self.nodalDOF_element_r[node]]
                r_OP += N_r[node] * r_OP_node
                r_OP_xi += N_r_xi[node] * r_OP_node

            # nodal rotation parametrization
            psi0, psi1 = qe[self.nodalDOF_element_psi]

            # nodal transformation matrices
            A_IK0 = self.RotationBase.Exp_SO3(psi0)
            A_IK1 = self.RotationBase.Exp_SO3(psi1)

            # relative interpolation of the rotation vector and it first derivative
            psi01 = Log_SO3(A_IK0.T @ A_IK1)

            # objective rotation
            A_IK = A_IK0 @ Exp_SO3(N_psi[1] * psi01)

            # axial and shear strains
            K_Gamma_bar = A_IK.T @ r_OP_xi

            # objective curvature
            # note: for two-node element T_SO3 @ psi_xi = psi_xi
            # T = T_SO3(N_psi[1] * psi01)
            # K_Kappa_bar = T @ (N_psi_xi[1] * psi01)
            K_Kappa_bar = N_psi_xi[1] * psi01

            return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

        def A_IK(self, t, q, frame_ID):
            # evaluate shape functions
            N_psi, _ = self.basis_functions_psi(frame_ID[0])

            # nodal rotation parametrization
            psi0, psi1 = q[self.nodalDOF_element_psi]

            # nodal transformation matrices
            A_IK0 = self.RotationBase.Exp_SO3(psi0)
            A_IK1 = self.RotationBase.Exp_SO3(psi1)

            # relative interpolation of the rotation vector and it first derivative
            psi01 = Log_SO3(A_IK0.T @ A_IK1)

            # objective rotation
            A_IK = A_IK0 @ Exp_SO3(N_psi[1] * psi01)

            return A_IK

        def A_IK_q(self, t, q, frame_ID):
            from cardillo.math import approx_fprime

            A_IK_q_num = approx_fprime(
                q, lambda q: self.A_IK(t, q, frame_ID), method="3-point", eps=1e-3
            )
            return A_IK_q_num

    return Derived


I_R3_SO3_PetrovGalerkin_AxisAngle = __make_R3_SO3(
    I_TimoshenkoPetrovGalerkinBaseAxisAngle
)
K_R3_SO3_PetrovGalerkin_AxisAngle = __make_R3_SO3(
    K_TimoshenkoPetrovGalerkinBaseAxisAngle
)
I_R3_SO3_PetrovGalerkin_Quaternion = __make_R3_SO3(
    I_TimoshenkoPetrovGalerkinBaseQuaternion
)
K_R3_SO3_PetrovGalerkin_Quaternion = __make_R3_SO3(
    K_TimoshenkoPetrovGalerkinBaseQuaternion
)
I_R3_SO3_PetrovGalerkin_R9 = __make_R3_SO3(I_TimoshenkoPetrovGalerkinBaseR9)
K_R3_SO3_PetrovGalerkin_R9 = __make_R3_SO3(K_TimoshenkoPetrovGalerkinBaseR9)


if False:
    # TODO: Use two node rotation element with left node as reference according
    # to SE(3) implementation.
    class Crisfield1999(K_TimoshenkoPetrovGalerkinBaseAxisAngle):
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
            p = max(polynomial_degree_r, polynomial_degree_psi)
            nquadrature = p
            nquadrature_dyn = int(np.ceil((p + 1) ** 2 / 2))

            # import warnings
            # warnings.warn("'Crisfield1999: Full integration is used!")
            # nquadrature = nquadrature_dyn

            # reference rotation for relative rotation proposed by Crisfield1999 (5.8)
            nnodes_element_psi = polynomial_degree_psi + 1
            self.node_A = int(0.5 * (nnodes_element_psi + 1)) - 1
            self.node_B = int(0.5 * (nnodes_element_psi + 2)) - 1

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

        def _reference_rotation(self, qe: np.ndarray, symmetric=False):
            """Reference rotation proposed by Crisfield1999 (5.8)."""
            if symmetric:
                A_0I = Exp_SO3(qe[self.nodalDOF_element_psi[self.node_A]])
                A_0J = Exp_SO3(qe[self.nodalDOF_element_psi[self.node_B]])
                A_IJ = A_0I.T @ A_0J  # Crisfield1999 (5.8)
                phi_IJ = Log_SO3(A_IJ)
                return A_0I @ Exp_SO3(0.5 * phi_IJ)
            else:
                return Exp_SO3(qe[self.nodalDOF_element_psi[0]])

        def _relative_interpolation(
            self,
            A_IR: np.ndarray,
            qe: np.ndarray,
            N_psi: np.ndarray,
            N_psi_xi: np.ndarray,
        ):
            """Interpolation function for relative rotation vectors proposed by
            Crisfield1999 (5.7) and (5.8)."""
            # relative interpolation of local rotation vectors
            psi_rel = np.zeros(3, qe.dtype)
            psi_rel_xi = np.zeros(3, qe.dtype)
            for node in range(self.nnodes_element_psi):
                # nodal axis angle vector
                psi_node = qe[self.nodalDOF_element_psi[node]]

                # nodal rotation
                A_IK_node = Exp_SO3(psi_node)

                # relative rotation of each node and corresponding
                # rotation vector
                A_RK_node = A_IR.T @ A_IK_node
                psi_RK_node = Log_SO3(A_RK_node)

                # add wheighted contribution of local rotation
                psi_rel += N_psi[node] * psi_RK_node
                psi_rel_xi += N_psi_xi[node] * psi_RK_node

            return psi_rel, psi_rel_xi

        def _eval(self, qe, xi):
            # evaluate shape functions
            N_r, N_r_xi = self.basis_functions_r(xi)
            N_psi, N_psi_xi = self.basis_functions_psi(xi)

            # interpolate tangent vector
            r_OP = np.zeros(3, dtype=qe.dtype)
            r_OP_xi = np.zeros(3, dtype=qe.dtype)
            for node in range(self.nnodes_element_r):
                r_OP_node = qe[self.nodalDOF_element_r[node]]
                r_OP += N_r[node] * r_OP_node
                r_OP_xi += N_r_xi[node] * r_OP_node

            # reference rotation, see. Crisfield1999 (5.8)
            A_IR = self._reference_rotation(qe)

            # relative interpolation of the rotation vector and it first derivative
            psi_rel, psi_rel_xi = self._relative_interpolation(
                A_IR, qe, N_psi, N_psi_xi
            )

            # objective rotation
            A_IK = A_IR @ Exp_SO3(psi_rel)

            # axial and shear strains
            K_Gamma_bar = A_IK.T @ r_OP_xi

            # objective curvature
            T = T_SO3(psi_rel)
            K_Kappa_bar = T @ psi_rel_xi

            return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

        def A_IK(self, t, q, frame_ID):
            # evaluate shape functions
            N_psi, N_psi_xi = self.basis_functions_psi(frame_ID[0])

            # reference rotation, see. Crisfield1999 (5.8)
            A_IR = self._reference_rotation(q)

            # relative interpolation of the rotation vector and it first derivative
            psi_rel, _ = self._relative_interpolation(A_IR, q, N_psi, N_psi_xi)

            # objective rotation
            return A_IR @ Exp_SO3(psi_rel)

        def A_IK_q(self, t, q, frame_ID):
            return approx_fprime(
                q,
                lambda q: self.A_IK(t, q, frame_ID),
                method="3-point",
                eps=1e-6,
            )

            # evaluate shape functions
            N_psi, _ = self.basis_functions_psi(frame_ID[0])

            # interpolate centerline position and orientation
            A_IK_q = np.zeros((3, 3, self.nq_element), dtype=q.dtype)
            for node in range(self.nnodes_element_psi):
                nodalDOF_psi = self.nodalDOF_element_psi[node]
                A_IK_q[:, :, nodalDOF_psi] += N_psi[node] * Exp_SO3_psi(q[nodalDOF_psi])

            return A_IK_q
