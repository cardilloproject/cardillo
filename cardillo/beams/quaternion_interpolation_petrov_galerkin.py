import numpy as np

from cardillo.math import (
    ax2skew,
    ax2skew_a,
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    T_SO3_quat,
    T_SO3_quat_P,
    approx_fprime,
)
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
        N, N_xi = self.basis_functions_r(xi)

        # quats = np.array(
        #     [
        #         qe[self.nodalDOF_element_psi[node]].copy()
        #         for node in range(self.nnodes_element_r)
        #     ]
        # )
        # signs = np.ones(len(quats))
        # Q0 = quats[0]
        # for i in range(1, len(quats)):
        #     Qi = quats[i]
        #     # inner = Q0 @ Qi
        #     # if inner < 0:
        #     if np.linalg.norm(Qi - Q0) > 1:
        #         quats[i] *= -1
        #         signs[i] = -1

        # interpolate
        r_OP = np.zeros(3, dtype=qe.dtype)
        r_OP_xi = np.zeros(3, dtype=qe.dtype)
        Q = np.zeros(4, dtype=float)
        Q_xi = np.zeros(4, dtype=float)
        for node in range(self.nnodes_element_r):
            r_OP_node = qe[self.nodalDOF_element_r[node]]
            r_OP += N[node] * r_OP_node
            r_OP_xi += N_xi[node] * r_OP_node

            Q_node = qe[self.nodalDOF_element_psi[node]]
            # Q_node = quats[i]
            Q += N[node] * Q_node
            Q_xi += N_xi[node] * Q_node
            # Q += signs[node] * N_r[node] * Q_node
            # Q_xi += N_r_xi[node] * Q_node

        # transformation matrix
        q0, q = np.array_split(Q, [1])
        q_tilde = ax2skew(q)
        Q2 = Q @ Q
        A_IK = np.eye(3, dtype=Q.dtype) + (2 / Q2) * (q0 * q_tilde + q_tilde @ q_tilde)

        # axial and shear strains
        K_Gamma_bar = A_IK.T @ r_OP_xi

        # curvature, Rucker2018 (17)
        # K_Kappa_bar = (
        #     (2 / Q2) * np.hstack((-q[:, None], q0 * np.eye(3) - ax2skew(q))) @ Q_xi
        # )
        K_Kappa_bar = T_SO3_quat(Q) @ Q_xi

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def _deval(self, qe, xi):
        # evaluate shape functions
        N_r, N_r_xi = self.basis_functions_r(xi)

        # interpolate
        r_OP = np.zeros(3, dtype=qe.dtype)
        r_OP_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
        r_OP_xi = np.zeros(3, dtype=qe.dtype)
        r_OP_xi_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)

        Q = np.zeros(4, dtype=float)
        Q_qe = np.zeros((4, self.nq_element), dtype=qe.dtype)
        Q_xi = np.zeros(4, dtype=float)
        Q_xi_qe = np.zeros((4, self.nq_element), dtype=qe.dtype)

        for node in range(self.nnodes_element_r):
            nodalDOF_r = self.nodalDOF_element_r[node]
            r_OP_node = qe[nodalDOF_r]

            r_OP += N_r[node] * r_OP_node
            r_OP_qe[:, nodalDOF_r] += N_r[node] * np.eye(3, dtype=float)

            r_OP_xi += N_r_xi[node] * r_OP_node
            r_OP_xi_qe[:, nodalDOF_r] += N_r_xi[node] * np.eye(3, dtype=float)

            nodalDOF_psi = self.nodalDOF_element_psi[node]
            Q_node = qe[nodalDOF_psi]

            Q += N_r[node] * Q_node
            Q_qe[:, nodalDOF_psi] += N_r[node] * np.eye(4, dtype=float)

            Q_xi += N_r_xi[node] * Q_node
            Q_xi_qe[:, nodalDOF_psi] += N_r_xi[node] * np.eye(4, dtype=float)

        # transformation matrix
        A_IK = Exp_SO3_quat(Q, normalize=True)
        # q0, q = np.array_split(Q, [1])
        # q_tilde = ax2skew(q)
        # Q2 = Q @ Q
        # A_IK = np.eye(3, dtype=Q.dtype) + (2 / Q2) * (q0 * q_tilde + q_tilde @ q_tilde)

        # derivative w.r.t. generalized coordinates
        A_IK_qe = Exp_SO3_quat_p(Q, normalize=True) @ Q_qe
        # q_tilde_q = ax2skew_a()
        # A_IK_Q = np.einsum(
        #     "ij,k->ijk", q0 * q_tilde + q_tilde @ q_tilde, -(4 / (Q2 * Q2)) * Q
        # )
        # Q22 = 2 / Q2
        # A_IK_Q[:, :, 0] += Q22 * q_tilde
        # A_IK_Q[:, :, 1:] += (
        #     Q22 * q0 * q_tilde_q
        #     + np.einsum("ijl,jk->ikl", q_tilde_q, Q22 * q_tilde)
        #     + np.einsum("ij,jkl->ikl", Q22 * q_tilde, q_tilde_q)
        # )
        # A_IK_qe = A_IK_Q @ Q_qe

        # axial and shear strains
        K_Gamma_bar = A_IK.T @ r_OP_xi
        K_Gamma_bar_qe = np.einsum("k,kij", r_OP_xi, A_IK_qe) + A_IK.T @ r_OP_xi_qe

        # curvature, Rucker2018 (17)
        T = T_SO3_quat(Q, normalize=True)
        K_Kappa_bar = T @ Q_xi
        # K_Kappa_bar = (
        #     (2 / Q2) * np.hstack((-q[:, None], q0 * np.eye(3) - ax2skew(q))) @ Q_xi
        # )

        # K_Kappa_bar_qe = approx_fprime(qe, lambda qe: self._eval(qe, xi)[3])
        K_Kappa_bar_qe = (
            np.einsum(
                "ijk,j->ik",
                T_SO3_quat_P(Q, normalize=True),
                Q_xi,
            )
            @ Q_qe
            + T @ Q_xi_qe
        )

        # ###################################
        # # compare with numerical derivative
        # ###################################
        # r_OP_qe_num = approx_fprime(qe, lambda qe: self._eval(qe, xi)[0])
        # # diff = r_OP_qe - r_OP_qe_num
        # # error = np.linalg.norm(diff)
        # # print(f"error r_OP_qe: {error}")

        # A_IK_qe_num = approx_fprime(qe, lambda qe: self._eval(qe, xi)[1])
        # # diff = A_IK_qe - A_IK_qe_num
        # # error = np.linalg.norm(diff)
        # # print(f"error A_IK_qe: {error}")

        # K_Gamma_bar_qe_num = approx_fprime(qe, lambda qe: self._eval(qe, xi)[2])
        # # diff = K_Gamma_bar_qe - K_Gamma_bar_qe_num
        # # error = np.linalg.norm(diff)
        # # print(f"error K_Gamma_bar_qe: {error}")

        # K_Kappa_bar_qe_num = approx_fprime(qe, lambda qe: self._eval(qe, xi)[3])
        # # diff = K_Kappa_bar_qe - K_Kappa_bar_qe_num
        # # error = np.linalg.norm(diff)
        # # print(f"error K_Kappa_bar_qe: {error}")

        # r_OP, A_IK, K_Gamma_bar, K_Kappa_bar = self._eval(qe, xi)
        # r_OP_qe = r_OP_qe_num
        # A_IK_qe = A_IK_qe_num
        # K_Gamma_bar_qe = K_Gamma_bar_qe_num
        # K_Kappa_bar_qe = K_Kappa_bar_qe_num

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
        return self._eval(q, frame_ID[0])[1]
        # # evaluate shape functions
        # N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # # interpolate orientation
        # A_IK = np.zeros((3, 3), dtype=q.dtype)
        # for node in range(self.nnodes_element_psi):
        #     A_IK += N_psi[node] * self.RotationBase.Exp_SO3(
        #         q[self.nodalDOF_element_psi[node]]
        #     )

        # return A_IK

    def A_IK_q(self, t, q, frame_ID):
        # return approx_fprime(q, lambda q: self.A_IK(t, q, frame_ID))
        return self._deval(q, frame_ID[0])[5]
