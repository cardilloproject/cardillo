import numpy as np
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey


from cardillo.math import (
    norm,
    cross3,
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
    Log_SO3_quat,
    A_IB_basic,
)
from cardillo.utility.check_time_derivatives import check_time_derivatives


from ._base import (
    CosseratRodDisplacementBased,
    CosseratRodMixed,
    make_CosseratRodConstrained,
)
from ._cross_section import CrossSectionInertias


def make_CosseratRod(
    *,
    interpolation="Quaternion",
    mixed=True,
    constraints=None,
    polynomial_degree=None,
    reduced_integration=True,
):
    """Rod factory that returns Petrov-Galerkin Cosserat rod classes.

    Parameters
    ----------
    interpolation : str
        Chose interpolation functions
        Quaternion: nodal positions and non-quaternions are interpolated by
        Lagrangian polynomials.
        SE3 : nodal positions and orientations are interpolated respecting the
        SE(3) structure, resulting in a constant strain element.
        R12 : nodal positions anr orientations are interpolated by Lagrangian
        polynomials.
    mixed : bool
        Boolean whether internal forces and couples obtain an independent field.
        Works only in combination with quadratic strain energy functions.
    constraints : array_like
        Set of numbers between 0 and 5 to indicate which strain is constrained.
        0 : Gamma_1 dilatation
        1 : Gamma_2 shear in e_y^K-direction.
        2 : Gamma_3 shear in e_z^K-direction.
        3 : kappa_1 twist.
        4 : kappa_2 flexure around e_y^K-direction.
        5 : kappa_3 flexure around e_z^K-direction.
    polynomial_degree : int
        Polynomial degree (p) of the interpolation of the centerline, orientation, virtual displacement and virtual rotation. If mixed or constrained: The polynomial degree for B_n and B_m is (p-1).
        If not specified, the following values are taken in dependency of the interpolation:
        Quaternion : p=2
        SE3 : p=1 (only possible value)
        R12 : p=2
    reduced_integration : bool
        number of integration points (m) for the spatial integration of the internal virtual work.
        True : m = p
        False : m = ceil((p+1)**2 / 2)

    Returns:
    --------
    out : CosseratRod class
        Returns CosseratRod class.

    """
    if constraints is not None:
        if not (
            (np.array(constraints) >= 0).all() & (np.array(constraints) <= 5).all()
        ):
            raise ValueError("constraint values must between 0 and 5")

    if interpolation == "Quaternion":
        Basis = make_CosseratRod_Quat(mixed=mixed, constraints=constraints)
        polynomial_degree = 2 if polynomial_degree is None else polynomial_degree
    elif interpolation == "SE3":
        Basis = make_CosseratRod_SE3(mixed=mixed, constraints=constraints)
        polynomial_degree = 1 if polynomial_degree is None else polynomial_degree
        assert (
            polynomial_degree == 1
        ), f"SE3 interpolation works only with polynomial_degree=1"
    elif interpolation == "R12":
        Basis = make_CosseratRod_R12(mixed=mixed, constraints=constraints)
        polynomial_degree = 2 if polynomial_degree is None else polynomial_degree
    else:
        raise NotImplementedError(
            "This kind of interpolation function has not been implemented."
        )

    class CosseratRod(Basis):
        def __init__(
            self,
            cross_section,
            material_model,
            nelement,
            *,
            Q,
            q0=None,
            u0=None,
            cross_section_inertias=CrossSectionInertias(),
            name="Cosserat_rod",
        ):
            """Petrov-Galerkin Cosserat rod formulations with
            quaternions for the nodal orientation parametrization.

            Parameters
            ----------
            cross_section : CrossSection
                Geometric cross-section properties: area, first and second moments
                of area.
            material_model: RodMaterialModel
                Constitutive law of Cosserat rod which relates the rod strain
                measures B_Gamma and B_Kappa with the contact forces B_n and couples
                B_m in the cross-section-fixed B-basis.
            nelement : int
                Number of rod elements.
            Q : np.ndarray (self.nq,)
                Generalized position coordinates of rod in a stress-free reference
                state. Q is a collection of nodal generalized position coordinates,
                which are given by the Cartesian coordinates of the nodal centerline
                point r_OP_i in R^3 together with non-unit quaternions p_i in R^4
                representing the nodal cross-section orientation.
            q0 : np.ndarray (self.nq,)
                Initial generalized position coordinates of rod at time t0.
            u0 : np.ndarray (self.nu,)
                Initial generalized velocity coordinates of rod at time t0.
                Generalized velocity coordinates u0 is a collection of the nodal
                generalized velocity coordinates, which are given by the nodal
                centerline velocity v_P_i in R^3 together with the cross-section
                angular velocity represented in the cross-section-fixed B-basis
                B_omega_IB.
            cross_section_inertias : CrossSectionInertias
                Inertia properties of cross-sections: Cross-section mass density and
                Cross-section inertia tensor represented in the cross-section-fixed
                B-Basis.
            name : str
                Name of contribution.
            """
            nquadrature = polynomial_degree
            nquadrature_dyn = int(np.ceil((polynomial_degree + 1) ** 2 / 2))

            if not reduced_integration:
                import warnings

                warnings.warn("Full integration is used!")
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
                name=name,
            )

        @staticmethod
        def straight_configuration(
            nelement,
            L,
            r_OP0=np.zeros(3, dtype=float),
            A_IB0=np.eye(3, dtype=float),
        ):
            """Compute generalized position coordinates for straight configuration."""
            nnodes = polynomial_degree * nelement + 1

            x0 = np.linspace(0, L, num=nnodes)
            y0 = np.zeros(nnodes)
            z0 = np.zeros(nnodes)
            r_OP = np.vstack((x0, y0, z0))
            for i in range(nnodes):
                r_OP[:, i] = r_OP0 + A_IB0 @ r_OP[:, i]

            # reshape generalized coordinates to nodal ordering
            q_r = r_OP.reshape(-1, order="C")

            # extract quaternion from orientation A_IB0
            p = Log_SO3_quat(A_IB0)
            q_p = np.repeat(p, nnodes)

            return np.concatenate([q_r, q_p])

        @staticmethod
        def serret_frenet_configuration(
            nelement,
            r_OP,
            r_OP_xi,
            r_OP_xixi,
            xi1,
            alpha=0.0,
            r_OP0=np.zeros(3, dtype=float),
            A_IB0=np.eye(3, dtype=float),
        ):
            """Compute generalized position coordinates for a pre-curved rod along curve r_OP. The cross-section orientations are based on the Serret-Frenet equations and afterwards rotated by alpha."""
            nnodes_r = polynomial_degree * nelement + 1

            r_OP, r_OP_xi, r_OP_xixi = check_time_derivatives(r_OP, r_OP_xi, r_OP_xixi)
            alpha, _, _ = check_time_derivatives(alpha, None, None)

            xis = np.linspace(0, xi1, nnodes_r)

            # nodal positions and unit quaternions
            r0 = np.zeros((3, nnodes_r))
            p0 = np.zeros((4, nnodes_r))

            for i, xii in enumerate(xis):
                r0[:, i] = r_OP0 + A_IB0 @ r_OP(xii)
                r_xi = r_OP_xi(xii)
                r_xixi = r_OP_xixi(xii)
                ex = r_xi / norm(r_xi)
                ey = r_xixi - ex * (ex @ r_xixi)
                ey = ey / norm(ey)
                A_B0B = np.vstack([ex, ey, cross3(ex, ey)]).T
                A_IB = A_IB0 @ A_B0B @ A_IB_basic(alpha(xii)).x
                p0[:, i] = Log_SO3_quat(A_IB)

            # check for the right quaternion hemisphere
            for i in range(nnodes_r - 1):
                inner = p0[:, i] @ p0[:, i + 1]
                if inner < 0:
                    p0[:, i + 1] *= -1

            # reshape generalized position coordinates to nodal ordering
            q_r = r0.reshape(-1, order="C")
            q_p = p0.reshape(-1, order="C")

            return np.concatenate([q_r, q_p])

        @staticmethod
        def pose_configuration(
            nelement,
            r_OP,
            A_IB,
            xi1=1.0,
            r_OP0=np.zeros(3, dtype=float),
            A_IB0=np.eye(3, dtype=float),
        ):
            """Compute generalized position coordinates for a pre-curved rod with centerline curve r_OP and orientation of A_IB."""
            nnodes_r = polynomial_degree * nelement + 1

            assert callable(r_OP), "r_OP must be callable!"
            assert callable(A_IB), "A_IB must be callable!"

            xis = np.linspace(0, xi1, nnodes_r)

            # nodal positions and unit quaternions
            r0 = np.zeros((3, nnodes_r))
            p0 = np.zeros((4, nnodes_r))

            for i, xii in enumerate(xis):
                r0[:, i] = r_OP0 + A_IB0 @ r_OP(xii)
                A_IBi = A_IB0 @ A_IB(xii)
                p0[:, i] = Log_SO3_quat(A_IBi)

            # check for the right quaternion hemisphere
            for i in range(nnodes_r - 1):
                inner = p0[:, i] @ p0[:, i + 1]
                if inner < 0:
                    p0[:, i + 1] *= -1

            # reshape generalized position coordinates to nodal ordering
            q_r = r0.reshape(-1, order="C")
            q_p = p0.reshape(-1, order="C")

            return np.concatenate([q_r, q_p])

        @staticmethod
        def straight_initial_configuration(
            nelement,
            L,
            r_OP0=np.zeros(3, dtype=float),
            A_IB0=np.eye(3, dtype=float),
            v_P0=np.zeros(3, dtype=float),
            B_omega_IB0=np.zeros(3, dtype=float),
        ):
            """ "Compute initial generalized position and velocity coordinates for straight configuration"""
            nnodes = polynomial_degree * nelement + 1

            x0 = np.linspace(0, L, num=nnodes)
            y0 = np.zeros(nnodes)
            z0 = np.zeros(nnodes)

            r_OP = np.vstack((x0, y0, z0))
            for i in range(nnodes):
                r_OP[:, i] = r_OP0 + A_IB0 @ r_OP[:, i]

            # reshape generalized coordinates to nodal ordering
            q_r = r_OP.reshape(-1, order="C")

            # extract quaternion from orientation A_IB0
            p = Log_SO3_quat(A_IB0)
            q_p = np.repeat(p, nnodes)

            ################################
            # compute generalized velocities
            ################################
            # centerline velocities
            v_P = np.zeros_like(r_OP, dtype=float)
            for i in range(nnodes):
                v_P[:, i] = v_P0 + cross3(A_IB0 @ B_omega_IB0, (r_OP[:, i] - r_OP0))

            # reshape generalized velocity coordinates to nodal ordering
            u_r = v_P.reshape(-1, order="C")

            # all nodes share the same angular velocity
            u_p = np.repeat(B_omega_IB0, nnodes)

            q0 = np.concatenate([q_r, q_p])
            u0 = np.concatenate([u_r, u_p])

            return q0, u0

    return CosseratRod


def make_CosseratRod_Quat(mixed=True, constraints=None):
    if mixed == True:
        if constraints == None:
            CosseratRodBase = CosseratRodMixed
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )
    else:
        if constraints == None:
            CosseratRodBase = CosseratRodDisplacementBased
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )

    class CosseratRod_Quat(CosseratRodBase):
        @cachedmethod(
            lambda self: self._eval_cache,
            key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
        )
        def _eval(self, qe, xi, N, N_xi):
            # evaluate shape functions
            # N, N_xi = self.basis_functions_r(xi)

            # interpolate
            r_OP = np.zeros(3, dtype=qe.dtype)
            r_OP_xi = np.zeros(3, dtype=qe.dtype)
            p = np.zeros(4, dtype=float)
            p_xi = np.zeros(4, dtype=float)
            for node in range(self.nnodes_element_r):
                r_OP_node = qe[self.nodalDOF_element_r[node]]
                r_OP += N[node] * r_OP_node
                r_OP_xi += N_xi[node] * r_OP_node

                p_node = qe[self.nodalDOF_element_p[node]]
                p += N[node] * p_node
                p_xi += N_xi[node] * p_node

            # transformation matrix
            A_IB = Exp_SO3_quat(p, normalize=True)

            # dilatation and shear strains
            B_Gamma_bar = A_IB.T @ r_OP_xi

            # curvature, Rucker2018 (17)
            B_Kappa_bar = T_SO3_quat(p, normalize=True) @ p_xi

            return r_OP, A_IB, B_Gamma_bar, B_Kappa_bar

        @cachedmethod(
            lambda self: self._deval_cache,
            key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
        )
        def _deval(self, qe, xi, N, N_xi):
            # evaluate shape functions
            # N, N_xi = self.basis_functions_r(xi)

            # interpolate
            r_OP = np.zeros(3, dtype=qe.dtype)
            r_OP_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
            r_OP_xi = np.zeros(3, dtype=qe.dtype)
            r_OP_xi_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)

            p = np.zeros(4, dtype=float)
            p_qe = np.zeros((4, self.nq_element), dtype=qe.dtype)
            p_xi = np.zeros(4, dtype=float)
            p_xi_qe = np.zeros((4, self.nq_element), dtype=qe.dtype)

            for node in range(self.nnodes_element_r):
                nodalDOF_r = self.nodalDOF_element_r[node]
                r_OP_node = qe[nodalDOF_r]

                r_OP += N[node] * r_OP_node
                r_OP_qe[:, nodalDOF_r] += N[node] * np.eye(3, dtype=float)

                r_OP_xi += N_xi[node] * r_OP_node
                r_OP_xi_qe[:, nodalDOF_r] += N_xi[node] * np.eye(3, dtype=float)

                nodalDOF_p = self.nodalDOF_element_p[node]
                p_node = qe[nodalDOF_p]

                p += N[node] * p_node
                p_qe[:, nodalDOF_p] += N[node] * np.eye(4, dtype=float)

                p_xi += N_xi[node] * p_node
                p_xi_qe[:, nodalDOF_p] += N_xi[node] * np.eye(4, dtype=float)

            # transformation matrix
            A_IB = Exp_SO3_quat(p, normalize=True)

            # derivative w.r.t. generalized coordinates
            A_IB_qe = Exp_SO3_quat_p(p, normalize=True) @ p_qe

            # axial and shear strains
            B_Gamma_bar = A_IB.T @ r_OP_xi
            B_Gamma_bar_qe = np.einsum("k,kij", r_OP_xi, A_IB_qe) + A_IB.T @ r_OP_xi_qe

            # curvature, Rucker2018 (17)
            T = T_SO3_quat(p, normalize=True)
            B_Kappa_bar = T @ p_xi

            # B_Kappa_bar_qe = approx_fprime(qe, lambda qe: self._eval(qe, xi)[3])
            B_Kappa_bar_qe = (
                np.einsum(
                    "ijk,j->ik",
                    T_SO3_quat_P(p, normalize=True),
                    p_xi,
                )
                @ p_qe
                + T @ p_xi_qe
            )

            return (
                r_OP,
                A_IB,
                B_Gamma_bar,
                B_Kappa_bar,
                r_OP_qe,
                A_IB_qe,
                B_Gamma_bar_qe,
                B_Kappa_bar_qe,
            )

        def A_IB(self, t, qe, xi):
            N, N_xi = self.basis_functions_r(xi)
            return self._eval(qe, xi, N, N_xi)[1]
            # # evaluate shape functions
            # N_p, _ = self.basis_functions_p(xi)

            # # interpolate orientation
            # A_IB = np.zeros((3, 3), dtype=q.dtype)
            # for node in range(self.nnodes_element_p):
            #     A_IB += N_p[node] * self.Exp_SO3_quat(
            #         q[self.nodalDOF_element_p[node]]
            #     )

            # return A_IB

        def A_IB_q(self, t, qe, xi):
            # return approx_fprime(q, lambda q: self.A_IB(t, q, xi))
            N, N_xi = self.basis_functions_r(xi)
            return self._deval(qe, xi, N, N_xi)[5]

    return CosseratRod_Quat


def make_CosseratRod_SE3(mixed=True, constraints=None):
    if mixed == True:
        if constraints == None:
            CosseratRodBase = CosseratRodMixed
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )
    else:
        if constraints == None:
            CosseratRodBase = CosseratRodDisplacementBased
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )

    class CosseratRod_SE3(CosseratRodBase):
        @cachedmethod(
            lambda self: self._eval_cache,
            key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
        )
        def _eval(self, qe, xi, N, N_xi):
            # nodal unknowns
            r_OP0, r_OP1 = qe[self.nodalDOF_element_r]
            p0, p1 = qe[self.nodalDOF_element_p]

            # nodal transformations
            A_IB0 = Exp_SO3_quat(p0)
            A_IB1 = Exp_SO3_quat(p1)
            H_IK0 = SE3(A_IB0, r_OP0)
            H_IK1 = SE3(A_IB1, r_OP1)

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
            A_IB = H_IK[:3, :3]
            r_OP = H_IK[:3, 3]

            # extract strains
            B_Gamma_bar = h_local_xi[:3]
            B_Kappa_bar = h_local_xi[3:]

            return r_OP, A_IB, B_Gamma_bar, B_Kappa_bar

        @cachedmethod(
            lambda self: self._deval_cache,
            key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
        )
        def _deval(self, qe, xi, N, N_xi):
            # extract nodal screws
            nodalDOF0 = np.concatenate(
                (self.nodalDOF_element_r[0], self.nodalDOF_element_p[0])
            )
            nodalDOF1 = np.concatenate(
                (self.nodalDOF_element_r[1], self.nodalDOF_element_p[1])
            )
            h0 = qe[nodalDOF0]
            r_OP0 = h0[:3]
            p0 = h0[3:]
            h1 = qe[nodalDOF1]
            r_OP1 = h1[:3]
            p1 = h1[3:]

            # nodal transformations
            A_IB0 = Exp_SO3_quat(p0)
            A_IB1 = Exp_SO3_quat(p1)
            H_IK0 = SE3(A_IB0, r_OP0)
            H_IK1 = SE3(A_IB1, r_OP1)
            A_IB0_p0 = Exp_SO3_quat_p(p0)
            A_IB1_p1 = Exp_SO3_quat_p(p1)

            H_IK0_h0 = np.zeros((4, 4, 7), dtype=float)
            H_IK0_h0[:3, :3, 3:] = A_IB0_p0
            H_IK0_h0[:3, 3, :3] = np.eye(3, dtype=float)
            H_IB1_h1 = np.zeros((4, 4, 7), dtype=float)
            H_IB1_h1[:3, :3, 3:] = A_IB1_p1
            H_IB1_h1[:3, 3, :3] = np.eye(3, dtype=float)

            # inverse transformation of first node
            H_IK0_inv = SE3inv(H_IK0)
            H_IK0_inv_h0 = np.zeros((4, 4, 7), dtype=float)
            H_IK0_inv_h0[:3, :3, 3:] = A_IB0_p0.transpose(1, 0, 2)
            H_IK0_inv_h0[:3, 3, 3:] = -np.einsum("k,kij->ij", r_OP0, A_IB0_p0)
            H_IK0_inv_h0[:3, 3, :3] = -A_IB0.T

            # compute relative transformation
            H_K0K1 = H_IK0_inv @ H_IK1
            H_K0B1_h0 = np.einsum("ilk,lj->ijk", H_IK0_inv_h0, H_IK1)
            H_K0B1_h1 = np.einsum("il,ljk->ijk", H_IK0_inv, H_IB1_h1)

            # compute relative screw
            h_K0K1 = Log_SE3(H_K0K1)
            h_K0B1_HK0K1 = Log_SE3_H(H_K0K1)
            h_K0B1_h0 = np.einsum("ikl,klj->ij", h_K0B1_HK0K1, H_K0B1_h0)
            h_K0B1_h1 = np.einsum("ikl,klj->ij", h_K0B1_HK0K1, H_K0B1_h1)

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
            h_local_h0 = N1 * h_K0B1_h0
            h_local_h1 = N1 * h_K0B1_h1
            h_local_xi_h0 = N1_xi * h_K0B1_h0
            h_local_xi_h1 = N1_xi * h_K0B1_h1

            # composition of reference and local transformation
            H_local = Exp_SE3(h_local)
            H_local_h = Exp_SE3_h(h_local)
            H_local_h0 = np.einsum("ijl,lk->ijk", H_local_h, h_local_h0)
            H_local_h1 = np.einsum("ijl,lk->ijk", H_local_h, h_local_h1)
            H_IK = H_IK0 @ H_local
            H_IB_h0 = np.einsum("ilk,lj", H_IK0_h0, H_local) + np.einsum(
                "il,ljk->ijk", H_IK0, H_local_h0
            )
            H_IB_h1 = np.einsum("il,ljk->ijk", H_IK0, H_local_h1)

            # extract centerline and transformation matrix
            A_IB = H_IK[:3, :3]
            r_OP = H_IK[:3, 3]
            A_IB_qe = np.zeros((3, 3, self.nq_element), dtype=float)
            A_IB_qe[:, :, nodalDOF0] = H_IB_h0[:3, :3]
            A_IB_qe[:, :, nodalDOF1] = H_IB_h1[:3, :3]
            r_OP_qe = np.zeros((3, self.nq_element), dtype=float)
            r_OP_qe[:, nodalDOF0] = H_IB_h0[:3, 3]
            r_OP_qe[:, nodalDOF1] = H_IB_h1[:3, 3]

            # extract strains
            B_Gamma_bar = h_local_xi[:3]
            B_Kappa_bar = h_local_xi[3:]
            B_Gamma_bar_qe = np.zeros((3, self.nq_element), dtype=float)
            B_Gamma_bar_qe[:, nodalDOF0] = h_local_xi_h0[:3]
            B_Gamma_bar_qe[:, nodalDOF1] = h_local_xi_h1[:3]
            B_Kappa_bar_qe = np.zeros((3, self.nq_element), dtype=float)
            B_Kappa_bar_qe[:, nodalDOF0] = h_local_xi_h0[3:]
            B_Kappa_bar_qe[:, nodalDOF1] = h_local_xi_h1[3:]

            return (
                r_OP,
                A_IB,
                B_Gamma_bar,
                B_Kappa_bar,
                r_OP_qe,
                A_IB_qe,
                B_Gamma_bar_qe,
                B_Kappa_bar_qe,
            )

        def A_IB(self, t, qe, xi):
            N, N_xi = None, None
            return self._eval(qe, xi, N, N_xi)[1]

        def A_IB_q(self, t, qe, xi):
            N, N_xi = None, None
            return self._deval(qe, xi, N, N_xi)[5]

    return CosseratRod_SE3


def make_CosseratRod_R12(mixed=True, constraints=None):
    if mixed == True:
        if constraints == None:
            CosseratRodBase = CosseratRodMixed
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )
    else:
        if constraints == None:
            CosseratRodBase = CosseratRodDisplacementBased
        else:
            CosseratRodBase = make_CosseratRodConstrained(
                mixed=mixed, constraints=constraints
            )

    class CosseratRod_R12(CosseratRodBase):
        # returns interpolated positions, orientations and strains at xi in [0,1]
        @cachedmethod(
            lambda self: self._eval_cache,
            key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
        )
        def _eval(self, qe, xi, N, N_xi):
            # evaluate shape functions
            # N, N_xi = self.basis_functions_r(xi)

            # interpolate position and tangent vector
            r_OP = np.zeros(3, dtype=qe.dtype)
            r_OP_xi = np.zeros(3, dtype=qe.dtype)
            for node in range(self.nnodes_element_r):
                r_OP_node = qe[self.nodalDOF_element_r[node]]
                r_OP += N[node] * r_OP_node
                r_OP_xi += N_xi[node] * r_OP_node

            # interpolate transformation matrix and its derivative
            A_IB = np.zeros((3, 3), dtype=qe.dtype)
            A_IB_xi = np.zeros((3, 3), dtype=qe.dtype)
            for node in range(self.nnodes_element_p):
                A_IB_node = Exp_SO3_quat(qe[self.nodalDOF_element_p[node]])
                A_IB += N[node] * A_IB_node
                A_IB_xi += N_xi[node] * A_IB_node

            # axial and shear strains
            B_Gamma_bar = A_IB.T @ r_OP_xi

            # torsional and flexural strains
            d1, d2, d3 = A_IB.T
            d1_xi, d2_xi, d3_xi = A_IB_xi.T
            B_Kappa_bar = np.array(
                [
                    0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                    0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                    0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                ]
            )

            return r_OP, A_IB, B_Gamma_bar, B_Kappa_bar

        @cachedmethod(
            lambda self: self._deval_cache,
            key=lambda self, qe, xi, N, N_xi: hashkey(*qe, xi),
        )
        def _deval(self, qe, xi, N, N_xi):
            # evaluate shape functions
            # N, N_xi = self.basis_functions_r(xi)

            # interpolate position and tangent vector + their derivatives
            r_OP = np.zeros(3, dtype=qe.dtype)
            r_OP_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
            r_OP_xi = np.zeros(3, dtype=qe.dtype)
            r_OP_xi_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
            for node in range(self.nnodes_element_r):
                nodalDOF_r = self.nodalDOF_element_r[node]
                r_OP_node = qe[nodalDOF_r]

                r_OP += N[node] * r_OP_node
                r_OP_qe[:, nodalDOF_r] += N[node] * np.eye(3, dtype=float)

                r_OP_xi += N_xi[node] * r_OP_node
                r_OP_xi_qe[:, nodalDOF_r] += N_xi[node] * np.eye(3, dtype=float)

            # interpolate transformation matrix and its derivative + their derivatives
            A_IB = np.zeros((3, 3), dtype=qe.dtype)
            A_IB_xi = np.zeros((3, 3), dtype=qe.dtype)
            A_IB_qe = np.zeros((3, 3, self.nq_element), dtype=qe.dtype)
            A_IB_xi_qe = np.zeros((3, 3, self.nq_element), dtype=qe.dtype)
            for node in range(self.nnodes_element_p):
                nodalDOF_p = self.nodalDOF_element_p[node]
                p_node = qe[nodalDOF_p]
                A_IB_node = Exp_SO3_quat(p_node)
                A_IB_q_node = Exp_SO3_quat_p(p_node)

                A_IB += N[node] * A_IB_node
                A_IB_qe[:, :, nodalDOF_p] += N[node] * A_IB_q_node

                A_IB_xi += N_xi[node] * A_IB_node
                A_IB_xi_qe[:, :, nodalDOF_p] += N_xi[node] * A_IB_q_node

            # extract directors
            d1, d2, d3 = A_IB.T
            d1_xi, d2_xi, d3_xi = A_IB_xi.T
            d1_qe, d2_qe, d3_qe = A_IB_qe.transpose(1, 0, 2)
            d1_xi_qe, d2_xi_qe, d3_xi_qe = A_IB_xi_qe.transpose(1, 0, 2)

            # axial and shear strains
            B_Gamma_bar = A_IB.T @ r_OP_xi
            B_Gamma_bar_qe = np.einsum("k,kij", r_OP_xi, A_IB_qe) + A_IB.T @ r_OP_xi_qe

            # torsional and flexural strains
            B_Kappa_bar = np.array(
                [
                    0.5 * (d3 @ d2_xi - d2 @ d3_xi),
                    0.5 * (d1 @ d3_xi - d3 @ d1_xi),
                    0.5 * (d2 @ d1_xi - d1 @ d2_xi),
                ]
            )
            B_Kappa_bar_qe = np.array(
                [
                    0.5
                    * (d3 @ d2_xi_qe + d2_xi @ d3_qe - d2 @ d3_xi_qe - d3_xi @ d2_qe),
                    0.5
                    * (d1 @ d3_xi_qe + d3_xi @ d1_qe - d3 @ d1_xi_qe - d1_xi @ d3_qe),
                    0.5
                    * (d2 @ d1_xi_qe + d1_xi @ d2_qe - d1 @ d2_xi_qe - d2_xi @ d1_qe),
                ]
            )

            return (
                r_OP,
                A_IB,
                B_Gamma_bar,
                B_Kappa_bar,
                r_OP_qe,
                A_IB_qe,
                B_Gamma_bar_qe,
                B_Kappa_bar_qe,
            )

        def A_IB(self, t, qe, xi):
            # evaluate shape functions
            N_p, _ = self.basis_functions_p(xi)

            # interpolate orientation
            A_IB = np.zeros((3, 3), dtype=qe.dtype)
            for node in range(self.nnodes_element_p):
                A_IB += N_p[node] * Exp_SO3_quat(qe[self.nodalDOF_element_p[node]])

            return A_IB

        def A_IB_q(self, t, qe, xi):
            # evaluate shape functions
            N_p, _ = self.basis_functions_p(xi)

            # interpolate centerline position and orientation
            A_IB_q = np.zeros((3, 3, self.nq_element), dtype=qe.dtype)
            for node in range(self.nnodes_element_p):
                nodalDOF_p = self.nodalDOF_element_p[node]
                A_IB_q[:, :, nodalDOF_p] += N_p[node] * Exp_SO3_quat_p(qe[nodalDOF_p])

            return A_IB_q

    return CosseratRod_R12
