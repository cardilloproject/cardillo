import numpy as np
from cardillo.math import (
    SE3,
    SE3inv,
    Exp_SE3,
    Log_SE3,
    Exp_SE3_h,
    Log_SE3_H,
)

from cardillo.beams._base_CosseratRodPG import CosseratRodPG

class CosseratRodPG_SE3(CosseratRodPG):
    def __init__(
        self,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelement,
        Q,
        q0=None,
        u0=None,
        reduced_integration=True,
    ):
        super().__init__(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            1,
            1,
            nelement,
            1 if reduced_integration else 2,
            2,
            Q,
            q0=q0,
            u0=u0,
            basis_r="Lagrange",
            basis_psi="Lagrange",
        )

    @staticmethod
    def straight_configuration(
        nelement,
        L,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
    ):
        return CosseratRodPG.straight_configuration(
            1, 1, "Lagrange", "Lagrange", nelement, L, r_OP, A_IK
        )

    @staticmethod
    def straight_initial_configuration(
        nelement,
        L,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
        v_P=np.zeros(3, dtype=float),
        K_omega_IK=np.zeros(3, dtype=float),
    ):
        return CosseratRodPG.straight_initial_configuration(
            1,
            1,
            "Lagrange",
            "Lagrange",
            nelement,
            L,
            r_OP,
            A_IK,
            v_P,
            K_omega_IK,
        )

    def _eval(self, qe, xi):
        # nodal unknowns
        r_OP0, r_OP1 = qe[self.nodalDOF_element_r]
        psi0, psi1 = qe[self.nodalDOF_element_psi]

        # nodal transformations
        A_IK0 = self.RotationBase.Exp_SO3(psi0)
        A_IK1 = self.RotationBase.Exp_SO3(psi1)
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
        A_IK0 = self.RotationBase.Exp_SO3(psi0)
        A_IK1 = self.RotationBase.Exp_SO3(psi1)
        H_IK0 = SE3(A_IK0, r_OP0)
        H_IK1 = SE3(A_IK1, r_OP1)
        A_IK0_psi0 = self.RotationBase.Exp_SO3_psi(psi0)
        A_IK1_psi1 = self.RotationBase.Exp_SO3_psi(psi1)

        n_psi = self.RotationBase.dim()
        H_IK0_h0 = np.zeros((4, 4, 3 + n_psi), dtype=float)
        H_IK0_h0[:3, :3, 3:] = A_IK0_psi0
        H_IK0_h0[:3, 3, :3] = np.eye(3, dtype=float)
        H_IK1_h1 = np.zeros((4, 4, 3 + n_psi), dtype=float)
        H_IK1_h1[:3, :3, 3:] = A_IK1_psi1
        H_IK1_h1[:3, 3, :3] = np.eye(3, dtype=float)

        # inverse transformation of first node
        H_IK0_inv = SE3inv(H_IK0)
        H_IK0_inv_h0 = np.zeros((4, 4, 3 + n_psi), dtype=float)
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
