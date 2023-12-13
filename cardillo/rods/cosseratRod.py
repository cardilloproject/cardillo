import numpy as np
from cardillo.math import (
    ax2skew,
    SE3,
    SE3inv,
    Exp_SE3,
    Log_SE3,
    Exp_SE3_h,
    Log_SE3_H,
    T_SO3_quat,
    T_SO3_quat_P,
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    Log_SO3,
    Exp_SO3,
)

from cardillo.rods._base_CosseratRod import (
    CosseratRod,
    CosseratRodMixed,
    make_CosseratRodConstrained,
)
from ._cross_section import CrossSectionInertias


def make_CosseratRod_SE3(mixed=False, constraints=None):
    if mixed == True:
        if constraints == None:
            CosseratRodBase = CosseratRodMixed
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )
    else:
        if constraints == None:
            CosseratRodBase = CosseratRod
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )

    class CosseratRod_SE3(CosseratRodBase):
        def __init__(
            self,
            cross_section,
            material_model,
            nelement,
            Q,
            q0=None,
            u0=None,
            reduced_integration=True,
            cross_section_inertias=CrossSectionInertias(),
            **kwargs,
        ):
            super().__init__(
                cross_section,
                material_model,
                nelement,
                polynomial_degree=1,
                nquadrature=1 if reduced_integration else 2,
                Q=Q,
                q0=q0,
                u0=u0,
                nquadrature_dyn=2,
                cross_section_inertias=cross_section_inertias,
            )

        @staticmethod
        def straight_configuration(
            nelement,
            L,
            r_OP=np.zeros(3, dtype=float),
            A_IK=np.eye(3, dtype=float),
            **kwargs,
        ):
            return CosseratRod.straight_configuration(nelement, L, 1, r_OP, A_IK)

        @staticmethod
        def deformed_configuration(
            nelement,
            curve,
            dcurve,
            ddcurve,
            xi1,
            polynomial_degree=1,
            r_OP=np.zeros(3, dtype=float),
            A_IK=np.eye(3, dtype=float),
        ):
            return CosseratRod.deformed_configuration(
                nelement,
                curve,
                dcurve,
                ddcurve,
                xi1,
                polynomial_degree=1,
                r_OP=r_OP,
                A_IK=A_IK,
            )

        @staticmethod
        def straight_initial_configuration(
            nelement,
            L,
            r_OP=np.zeros(3, dtype=float),
            A_IK=np.eye(3, dtype=float),
            v_P=np.zeros(3, dtype=float),
            K_omega_IK=np.zeros(3, dtype=float),
            **kwargs,
        ):
            return CosseratRod.straight_initial_configuration(
                nelement,
                L,
                polynomial_degree=1,
                r_OP=r_OP,
                A_IK=A_IK,
                v_P=v_P,
                K_omega_IK=K_omega_IK,
            )

        def _eval(self, qe, xi):
            # nodal unknowns
            r_OP0, r_OP1 = qe[self.nodalDOF_element_r]
            psi0, psi1 = qe[self.nodalDOF_element_psi]

            # nodal transformations
            A_IK0 = Exp_SO3_quat(psi0)
            A_IK1 = Exp_SO3_quat(psi1)
            H_IK0 = SE3(A_IK0, r_OP0)
            H_IK1 = SE3(A_IK1, r_OP1)

            # inverse transformation of first node
            H_IK0_inv = SE3inv(H_IK0)

            # compute relative transformation
            H_K0K1 = H_IK0_inv @ H_IK1

            # compute relative screw
            h_K0K1 = Log_SE3(H_K0K1)

            # find element number containing xi
            el = self.element_number(xi)

            # get element interval
            xi0, xi1 = self.knot_vector_r.element_interval(el)

            # second linear Lagrange shape function
            N1_xi = 1.0 / (xi1 - xi0)
            N1 = (xi - xi0) * N1_xi

            # relative interpolation of local se(3) objects
            h_local = N1 * h_K0K1
            h_local_xi = N1_xi * h_K0K1

            # composition of reference and local transformation
            H_local = Exp_SE3(h_local)
            H_IK = H_IK0 @ H_local

            # extract centerline and transformation matrix
            A_IK = H_IK[:3, :3]
            r_OP = H_IK[:3, 3]

            # extract strains
            K_Gamma_bar = h_local_xi[:3]
            K_Kappa_bar = h_local_xi[3:]

            return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

        def _deval(self, qe, xi):
            # extract nodal screws
            nodalDOF0 = np.concatenate(
                (self.nodalDOF_element_r[0], self.nodalDOF_element_psi[0])
            )
            nodalDOF1 = np.concatenate(
                (self.nodalDOF_element_r[1], self.nodalDOF_element_psi[1])
            )
            h0 = qe[nodalDOF0]
            r_OP0 = h0[:3]
            psi0 = h0[3:]
            h1 = qe[nodalDOF1]
            r_OP1 = h1[:3]
            psi1 = h1[3:]

            # nodal transformations
            A_IK0 = Exp_SO3_quat(psi0)
            A_IK1 = Exp_SO3_quat(psi1)
            H_IK0 = SE3(A_IK0, r_OP0)
            H_IK1 = SE3(A_IK1, r_OP1)
            A_IK0_psi0 = Exp_SO3_quat_p(psi0)
            A_IK1_psi1 = Exp_SO3_quat_p(psi1)

            H_IK0_h0 = np.zeros((4, 4, 7), dtype=float)
            H_IK0_h0[:3, :3, 3:] = A_IK0_psi0
            H_IK0_h0[:3, 3, :3] = np.eye(3, dtype=float)
            H_IK1_h1 = np.zeros((4, 4, 7), dtype=float)
            H_IK1_h1[:3, :3, 3:] = A_IK1_psi1
            H_IK1_h1[:3, 3, :3] = np.eye(3, dtype=float)

            # inverse transformation of first node
            H_IK0_inv = SE3inv(H_IK0)
            H_IK0_inv_h0 = np.zeros((4, 4, 7), dtype=float)
            H_IK0_inv_h0[:3, :3, 3:] = A_IK0_psi0.transpose(1, 0, 2)
            H_IK0_inv_h0[:3, 3, 3:] = -np.einsum("k,kij->ij", r_OP0, A_IK0_psi0)
            H_IK0_inv_h0[:3, 3, :3] = -A_IK0.T

            # compute relative transformation
            H_K0K1 = H_IK0_inv @ H_IK1
            H_K0K1_h0 = np.einsum("ilk,lj->ijk", H_IK0_inv_h0, H_IK1)
            H_K0K1_h1 = np.einsum("il,ljk->ijk", H_IK0_inv, H_IK1_h1)

            # compute relative screw
            h_K0K1 = Log_SE3(H_K0K1)
            h_K0K1_HK0K1 = Log_SE3_H(H_K0K1)
            h_K0K1_h0 = np.einsum("ikl,klj->ij", h_K0K1_HK0K1, H_K0K1_h0)
            h_K0K1_h1 = np.einsum("ikl,klj->ij", h_K0K1_HK0K1, H_K0K1_h1)

            # find element number containing xi
            el = self.element_number(xi)

            # get element interval
            xi0, xi1 = self.knot_vector_r.element_interval(el)

            # second linear Lagrange shape function
            N1_xi = 1.0 / (xi1 - xi0)
            N1 = (xi - xi0) * N1_xi

            # relative interpolation of local se(3) objects
            h_local = N1 * h_K0K1
            h_local_xi = N1_xi * h_K0K1
            h_local_h0 = N1 * h_K0K1_h0
            h_local_h1 = N1 * h_K0K1_h1
            h_local_xi_h0 = N1_xi * h_K0K1_h0
            h_local_xi_h1 = N1_xi * h_K0K1_h1

            # composition of reference and local transformation
            H_local = Exp_SE3(h_local)
            H_local_h = Exp_SE3_h(h_local)
            H_local_h0 = np.einsum("ijl,lk->ijk", H_local_h, h_local_h0)
            H_local_h1 = np.einsum("ijl,lk->ijk", H_local_h, h_local_h1)
            H_IK = H_IK0 @ H_local
            H_IK_h0 = np.einsum("ilk,lj", H_IK0_h0, H_local) + np.einsum(
                "il,ljk->ijk", H_IK0, H_local_h0
            )
            H_IK_h1 = np.einsum("il,ljk->ijk", H_IK0, H_local_h1)

            # extract centerline and transformation matrix
            A_IK = H_IK[:3, :3]
            r_OP = H_IK[:3, 3]
            A_IK_qe = np.zeros((3, 3, self.nq_element), dtype=float)
            A_IK_qe[:, :, nodalDOF0] = H_IK_h0[:3, :3]
            A_IK_qe[:, :, nodalDOF1] = H_IK_h1[:3, :3]
            r_OP_qe = np.zeros((3, self.nq_element), dtype=float)
            r_OP_qe[:, nodalDOF0] = H_IK_h0[:3, 3]
            r_OP_qe[:, nodalDOF1] = H_IK_h1[:3, 3]

            # extract strains
            K_Gamma_bar = h_local_xi[:3]
            K_Kappa_bar = h_local_xi[3:]
            K_Gamma_bar_qe = np.zeros((3, self.nq_element), dtype=float)
            K_Gamma_bar_qe[:, nodalDOF0] = h_local_xi_h0[:3]
            K_Gamma_bar_qe[:, nodalDOF1] = h_local_xi_h1[:3]
            K_Kappa_bar_qe = np.zeros((3, self.nq_element), dtype=float)
            K_Kappa_bar_qe[:, nodalDOF0] = h_local_xi_h0[3:]
            K_Kappa_bar_qe[:, nodalDOF1] = h_local_xi_h1[3:]

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

        def A_IK_q(self, t, q, frame_ID):
            return self._deval(q, frame_ID[0])[5]

    return CosseratRod_SE3


def make_CosseratRod_R3SO3(mixed=False, constraints=None):
    if mixed == True:
        if constraints == None:
            CosseratRodBase = CosseratRodMixed
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )
    else:
        if constraints == None:
            CosseratRodBase = CosseratRod
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )

    class CosseratRod_R3SO3(CosseratRodBase):
        def __init__(
            self,
            cross_section,
            material_model,
            nelement,
            Q,
            q0=None,
            u0=None,
            reduced_integration=True,
            cross_section_inertias=CrossSectionInertias(),
            **kwargs,
        ):
            super().__init__(
                cross_section,
                material_model,
                nelement,
                polynomial_degree=1,
                nquadrature=1 if reduced_integration else 2,
                Q=Q,
                q0=q0,
                u0=u0,
                nquadrature_dyn=2,
                cross_section_inertias=cross_section_inertias,
            )

        @staticmethod
        def straight_configuration(
            nelement,
            L,
            r_OP=np.zeros(3, dtype=float),
            A_IK=np.eye(3, dtype=float),
            **kwargs,
        ):
            return CosseratRod.straight_configuration(
                nelement, L, polynomial_degree=1, r_OP=r_OP, A_IK=A_IK
            )

        @staticmethod
        def straight_initial_configuration(
            nelement,
            L,
            r_OP=np.zeros(3, dtype=float),
            A_IK=np.eye(3, dtype=float),
            v_P=np.zeros(3, dtype=float),
            K_omega_IK=np.zeros(3, dtype=float),
            **kwargs,
        ):
            return CosseratRod.straight_initial_configuration(
                nelement,
                L,
                polynomial_degree=1,
                r_OP=r_OP,
                A_IK=A_IK,
                v_P=v_P,
                K_omega_IK=K_omega_IK,
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
            A_IK0 = Exp_SO3_quat(psi0)
            A_IK1 = Exp_SO3_quat(psi1)

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
            A_IK0 = Exp_SO3_quat(psi0)
            A_IK1 = Exp_SO3_quat(psi1)

            # relative interpolation of the rotation vector and it first derivative
            psi01 = Log_SO3(A_IK0.T @ A_IK1)

            # objective rotation
            A_IK = A_IK0 @ Exp_SO3(N_psi[1] * psi01)

            return A_IK

        def A_IK_q(self, t, q, frame_ID):
            from cardillo.math.approx_fprime import approx_fprime

            A_IK_q_num = approx_fprime(
                q, lambda q: self.A_IK(t, q, frame_ID), method="3-point", eps=1e-3
            )
            return A_IK_q_num

    return CosseratRod_R3SO3


def make_CosseratRod_Quat(mixed=False, constraints=None):
    if mixed == True:
        if constraints == None:
            CosseratRodBase = CosseratRodMixed
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )
    else:
        if constraints == None:
            CosseratRodBase = CosseratRod
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )

    class CosseratRod_Quat(CosseratRodBase):
        def __init__(
            self,
            cross_section,
            material_model,
            nelement,
            Q,
            q0=None,
            u0=None,
            polynomial_degree=2,
            reduced_integration=True,
            cross_section_inertias=CrossSectionInertias(),
        ):
            nquadrature = polynomial_degree
            nquadrature_dyn = int(np.ceil((polynomial_degree + 1) ** 2 / 2))

            if not reduced_integration:
                import warnings

                warnings.warn("Quaternion interpolation: Full integration is used!")
                nquadrature = nquadrature_dyn

            super().__init__(
                cross_section,
                material_model,
                nelement,
                polynomial_degree=polynomial_degree,
                nquadrature=nquadrature,
                Q=Q,
                q0=q0,
                u0=u0,
                nquadrature_dyn=nquadrature_dyn,
                cross_section_inertias=cross_section_inertias,
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
            return CosseratRod.straight_initial_configuration(
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
            A_IK = np.eye(3, dtype=Q.dtype) + (2 / Q2) * (
                q0 * q_tilde + q_tilde @ q_tilde
            )

            # axial and shear strains
            K_Gamma_bar = A_IK.T @ r_OP_xi

            # curvature, Rucker2018 (17)
            # K_Kappa_bar = (
            #     (2 / Q2) * np.hstack((-q[:, None], q0 * np.eye(3) - ax2skew(q))) @ Q_xi
            # )
            K_Kappa_bar = T_SO3_quat(Q, normalize=True) @ Q_xi

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
            #     A_IK += N_psi[node] * self.Exp_SO3_quat(
            #         q[self.nodalDOF_element_psi[node]]
            #     )

            # return A_IK

        def A_IK_q(self, t, q, frame_ID):
            # return approx_fprime(q, lambda q: self.A_IK(t, q, frame_ID))
            return self._deval(q, frame_ID[0])[5]

    return CosseratRod_Quat


def make_CosseratRod_R12(mixed=False, constraints=None):
    if mixed == True:
        if constraints == None:
            CosseratRodBase = CosseratRodMixed
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )
    else:
        if constraints == None:
            CosseratRodBase = CosseratRod
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )

    class CosseratRod_R12(CosseratRodBase):
        def __init__(
            self,
            cross_section,
            material_model,
            nelement,
            Q,
            q0=None,
            u0=None,
            polynomial_degree=2,
            reduced_integration=True,
            cross_section_inertias=CrossSectionInertias(),
        ):
            nquadrature = polynomial_degree
            nquadrature_dyn = int(np.ceil((polynomial_degree + 1) ** 2 / 2))

            if not reduced_integration:
                import warnings

                warnings.warn("'R12_PetrovGalerkin': Full integration is used!")
                nquadrature = nquadrature_dyn

            super().__init__(
                cross_section,
                material_model,
                nelement,
                polynomial_degree=polynomial_degree,
                nquadrature=nquadrature,
                Q=Q,
                q0=q0,
                u0=u0,
                nquadrature_dyn=nquadrature_dyn,
                cross_section_inertias=cross_section_inertias,
            )

        # returns interpolated positions, orientations and strains at xi in [0,1]
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
                A_IK_node = Exp_SO3_quat(qe[self.nodalDOF_element_psi[node]])
                A_IK += N_psi[node] * A_IK_node
                A_IK_xi += N_psi_xi[node] * A_IK_node

            # axial and shear strains
            K_Gamma_bar = A_IK.T @ r_OP_xi

            # torsional and flexural strains
            d1, d2, d3 = A_IK.T
            d1_xi, d2_xi, d3_xi = A_IK_xi.T
            K_Kappa_bar = np.array(
                [
                    0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                    0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                    0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                ]
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
                A_IK_node = Exp_SO3_quat(psi_node)
                A_IK_q_node = Exp_SO3_quat_p(psi_node)

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
            K_Kappa_bar = np.array(
                [
                    0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                    0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                    0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                ]
            )
            K_Kappa_bar_qe = np.array(
                [
                    0.5
                    * (d3 @ d2_xi_qe + d2_xi @ d3_qe - d2 @ d3_xi_qe - d3_xi @ d2_qe),
                    0.5
                    * (d1 @ d3_xi_qe + d3_xi @ d1_qe - d3 @ d1_xi_qe - d1_xi @ d3_qe),
                    0.5
                    * (d2 @ d1_xi_qe + d1_xi @ d2_qe - d1 @ d2_xi_qe - d2_xi @ d1_qe),
                ]
            )

            # from cardillo.math import approx_fprime

            # r_OP_qe_num = approx_fprime(
            #     qe,
            #     lambda qe: self._eval(qe, xi)[0],
            # )
            # diff = r_OP_qe - r_OP_qe_num
            # error = np.linalg.norm(diff)
            # print(f"error r_OP_qe: {error}")

            # A_IK_qe_num = approx_fprime(
            #     qe,
            #     lambda qe: self._eval(qe, xi)[1],
            # )
            # diff = A_IK_qe - A_IK_qe_num
            # error = np.linalg.norm(diff)
            # print(f"error A_IK_qe: {error}")

            # K_Kappa_bar_qe_num = approx_fprime(
            #     qe,
            #     lambda qe: self._eval(qe, xi)[3],
            # )
            # diff = K_Kappa_bar_qe - K_Kappa_bar_qe_num
            # error = np.linalg.norm(diff)
            # print(f"error K_Kappa_bar_qe: {error}")

            # K_Gamma_bar_qe_num = approx_fprime(
            #     qe,
            #     lambda qe: self._eval(qe, xi)[2],
            # )
            # diff = K_Gamma_bar_qe - K_Gamma_bar_qe_num
            # error = np.linalg.norm(diff)
            # print(f"error K_Gamma_bar_qe: {error}")

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
                A_IK += N_psi[node] * Exp_SO3_quat(q[self.nodalDOF_element_psi[node]])

            return A_IK

        def A_IK_q(self, t, q, frame_ID):
            # evaluate shape functions
            N_psi, _ = self.basis_functions_psi(frame_ID[0])

            # interpolate centerline position and orientation
            A_IK_q = np.zeros((3, 3, self.nq_element), dtype=q.dtype)
            for node in range(self.nnodes_element_psi):
                nodalDOF_psi = self.nodalDOF_element_psi[node]
                A_IK_q[:, :, nodalDOF_psi] += N_psi[node] * Exp_SO3_quat_p(
                    q[nodalDOF_psi]
                )

            return A_IK_q

    return CosseratRod_R12
