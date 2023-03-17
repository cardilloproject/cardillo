import numpy as np
import warnings
from cardillo.math import (
    Exp_SO3,
    T_SO3,
    Exp_SO3_quat,
    T_SO3_quat,
    norm,
    pi,
)
from cardillo.beams._base_petrov_galerkin import (
    K_TimoshenkoPetrovGalerkinBaseAxisAngle,
    K_TimoshenkoPetrovGalerkinBaseQuaternion,
)


# TODO: Do we want to keep this?
class K_TimoshenkoLerp(K_TimoshenkoPetrovGalerkinBaseQuaternion):
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
        warnings.warn("K_TimoshenkoLerp is not objective!")

        p = max(polynomial_degree_r, polynomial_degree_psi)
        nquadrature = p
        nquadrature_dyn = int(np.ceil((p + 1) ** 2 / 2))

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

        # interpolate rotation vector
        psi = np.zeros(4, dtype=qe.dtype)
        psi_xi = np.zeros(4, dtype=qe.dtype)
        for node in range(self.nnodes_element_psi):
            psi_node = qe[self.nodalDOF_element_psi[node]]
            psi += N_psi[node] * psi_node
            psi_xi += N_psi_xi[node] * psi_node

        # # TODO: normalize quaternion
        # psi2 = psi @ psi
        # psi = psi / np.sqrt(psi2)

        # transformation matrix from quaternion
        A_IK = Exp_SO3_quat(psi)

        # axial and shear strains
        K_Gamma_bar = A_IK.T @ r_OP_xi

        # curvature
        K_Kappa_bar = T_SO3_quat(psi) @ psi_xi

        # # Egeland2002 (6.325), this is exactly T_SO3_quat(psi) @ psi_xi
        # from cardillo.math import quatprod, quat_conjugate
        # K_Kappa_bar2 = 2 * quatprod(quat_conjugate(psi), psi_xi)[1:]

        # # Egeland2002 (6.327), this is exactly T_SO3_quat(psi) @ psi_xi
        # from cardillo.math import cross3
        # K_Kappa_bar3 = 2 * (psi[0] * psi_xi[1:] - psi_xi[0] * psi[1:] - cross3(psi[1:], psi_xi[1:]))

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def A_IK(self, t, q, frame_ID):
        return self._eval(q, frame_ID[0])[1]
        # # evaluate shape functions
        # N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # # interpolate rotation vector
        # psi = np.zeros(3, dtype=q.dtype)
        # for node in range(self.nnodes_element_psi):
        #     psi += N_psi[node] * q[self.nodalDOF_element_psi[node]]

        # return Exp_SO3(psi)

    def A_IK_q(self, t, q, frame_ID):
        # # evaluate shape functions
        # N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # # interpolate rotation vector
        # psi = np.zeros(3, dtype=q.dtype)
        # for node in range(self.nnodes_element_psi):
        #     psi += N_psi[node] * q[self.nodalDOF_element_psi[node]]

        # A_IK_psi = Exp_SO3_psi(psi)
        # A_IK_q = np.zeros((3, 3, self.nq_element), dtype=float)
        # for node in range(self.nnodes_element_psi):
        #     A_IK_q[:, :, self.nodalDOF_element_psi[node]] = N_psi[node] * A_IK_psi

        from cardillo.math import approx_fprime

        A_IK_q_num = approx_fprime(
            q,
            lambda q: self.A_IK(t, q, frame_ID),
            method="2-point",
            eps=1e-6,
        )
        # diff = A_IK_q - A_IK_q_num
        # error = np.linalg.norm(diff)
        # print(f"error A_IK_q: {error}")

        return A_IK_q_num


class K_Cardona(K_TimoshenkoPetrovGalerkinBaseAxisAngle):
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
        warnings.warn("K_Cardona is not objective!")

        p = max(polynomial_degree_r, polynomial_degree_psi)
        nquadrature = p
        nquadrature_dyn = int(np.ceil((p + 1) ** 2 / 2))

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

        # compute rotation vectors and complement rotation vectors
        use_complement = False
        n = 0
        # angle = 0.0
        for node in range(self.nnodes_element_psi):
            psi_node = qe[self.nodalDOF_element_psi[node]]
            angle_node = norm(psi_node)
            if angle_node > pi:
                use_complement = True
                # n_node = int((angle_node + np.pi) / (2.0 * np.pi))
                # n = max(n, n_node)
                n = 1
                # angle = max(angle, angle_node)

        # interpolate rotation vector
        psi = np.zeros(3, dtype=qe.dtype)
        # psi_C = np.zeros(3, dtype=qe.dtype)
        psi_xi = np.zeros(3, dtype=qe.dtype)
        # psi_C_xi = np.zeros(3, dtype=qe.dtype)
        for node in range(self.nnodes_element_psi):
            psi_node = qe[self.nodalDOF_element_psi[node]]
            # if use_complement:
            #     angle = norm(psi_node)
            #     # n = int((angle + np.pi) / (2.0 * np.pi))
            #     # angle_C = (angle - 2.0 * n * pi)
            #     psi_node_C = (1.0 - 2.0 * n * pi / angle) * psi_node.copy()
            #     # psi_node = psi_node_C
            #     # psi_node = (1.0 - 2.0 * pi / angle) * psi_node
            #     # print(f"complement used")
            #     # print(f"- n: {n}")
            #     # print(f"- angle: {angle}")
            #     # print(f"- angle_C: {angle_C}")
            #     # print(f"- psi: {psi_node}")
            # else:
            #     psi_node_C = psi_node.copy()

            psi += N_psi[node] * psi_node
            psi_xi += N_psi_xi[node] * psi_node
            # psi_C += N_psi[node] * psi_node_C
            # psi_C_xi += N_psi_xi[node] * psi_node_C

        # angle = norm(psi)
        # if use_complement:
        #     # psi_C = (1.0 - 2.0 * n * pi / angle) * psi.copy()
        #     psi_C = (psi - 2.0 * n * pi * psi / angle)
        #     e = psi / angle
        #     A = (
        #         (1 - 2 * n * np.pi / angle) * np.eye(3, dtype=qe.dtype)
        #         + 2 * n * np.pi / angle * np.outer(e, e)
        #     )
        # else:
        #     psi_C = psi.copy()
        #     A = np.eye(3)

        # psi_C_xi = A @ psi_xi

        # diff = T_SO3(psi) @ psi_xi - T_SO3(psi_C) @ psi_C_xi
        # error = np.linalg.norm(diff)
        # print(f"error K_kappa: {error}")

        # diff = Exp_SO3(psi).T @ r_OP_xi - Exp_SO3(psi_C).T @ r_OP_xi
        # error = np.linalg.norm(diff)
        # print(f"error K_Gamma: {error}")

        # # transformation matrix
        # # print(f"complement used: {use_complement}")
        # # print(f"- angle: {norm(psi)}")
        # # print(f"- psi: {psi}")
        # A_IK = Exp_SO3(psi_C)
        A_IK = Exp_SO3(psi)

        # axial and shear strains
        K_Gamma_bar = A_IK.T @ r_OP_xi

        # curvature
        K_Kappa_bar = T_SO3(psi) @ psi_xi

        # # curvature
        # # T = T_SO3(psi_C)
        # # n = psi / angle
        # # A = (
        # #     (1 - 2 * n * np.pi / angle) * np.eye(3, dtype=qe.dtype)
        # #     + 2 * n * np.pi * np.outer(n, n)
        # # )
        # # K_Kappa_bar = T @ A @ psi_xi
        # K_Kappa_bar = T_SO3(psi_C) @ psi_C_xi

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def A_IK(self, t, q, frame_ID):
        return self._eval(q, frame_ID[0])[1]
        # # evaluate shape functions
        # N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # # interpolate rotation vector
        # psi = np.zeros(3, dtype=q.dtype)
        # for node in range(self.nnodes_element_psi):
        #     psi += N_psi[node] * q[self.nodalDOF_element_psi[node]]

        # return Exp_SO3(psi)

    def A_IK_q(self, t, q, frame_ID):
        # # evaluate shape functions
        # N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # # interpolate rotation vector
        # psi = np.zeros(3, dtype=q.dtype)
        # for node in range(self.nnodes_element_psi):
        #     psi += N_psi[node] * q[self.nodalDOF_element_psi[node]]

        # A_IK_psi = Exp_SO3_psi(psi)
        # A_IK_q = np.zeros((3, 3, self.nq_element), dtype=float)
        # for node in range(self.nnodes_element_psi):
        #     A_IK_q[:, :, self.nodalDOF_element_psi[node]] = N_psi[node] * A_IK_psi

        from cardillo.math import approx_fprime

        A_IK_q_num = approx_fprime(
            q,
            lambda q: self.A_IK(t, q, frame_ID),
            method="2-point",
            eps=1e-6,
        )
        # diff = A_IK_q - A_IK_q_num
        # error = np.linalg.norm(diff)
        # print(f"error A_IK_q: {error}")

        return A_IK_q_num
