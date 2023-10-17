import numpy as np
from scipy.optimize import least_squares
from cardillo.math import SE3, SE3inv, Log_SE3, Log_SE3_H, Log_SO3_quat, Exp_SO3, norm
from cardillo.beams._base_parametrization import QuaternionRotationParameterization


# TODO: Add clamping for first and last node
def fit_configuration(
    rod,
    r_OPs,
    A_IKs,
    use_cord_length=False,
    nodal_cDOF=[0, -1],
    verbose=1,
):
    # number of sample points
    n_samples = len(r_OPs)

    # compute homogeneous transformations for target points
    Hs = np.array([SE3(A_IKs[i], r_OPs[i]) for i in range(n_samples)])

    # cord-length of relative twists
    if use_cord_length:
        relative_twists = np.array(
            [Log_SE3(SE3inv(Hs[i - 1]) @ Hs[i]) for i in range(1, n_samples)]
        )
        segment_lengths = np.linalg.norm(relative_twists, axis=1)
        cord_lengths = np.cumsum(segment_lengths)
        xis = np.zeros(n_samples, dtype=float)
        xis[1:] = cord_lengths / cord_lengths[-1]
    else:
        # linear spaced evaluation points for the rod
        xis = np.linspace(0, 1, n_samples)

    # initial vector of generalized coordinates
    # Z0 = np.zeros(rod.nq, dtype=float)
    Z0 = np.zeros(rod.nq_r + rod.nq_psi, dtype=float)

    ###########
    # warmstart
    ###########
    # initialized nodal unknowns with closest data w.r.t. xi values
    xi_idx_r = np.array(
        [
            np.where(xis >= rod.mesh_r.knot_vector.data[i])[0][0]
            for i in range(rod.mesh_r.nnodes)
        ]
    )
    xi_idx_psi = np.array(
        [
            np.where(xis >= rod.mesh_psi.knot_vector.data[i])[0][0]
            for i in range(rod.mesh_psi.nnodes)
        ]
    )

    for node, xi_idx in enumerate(xi_idx_r):
        Z0[rod.nodalDOF_r[node]] = r_OPs[xi_idx]
    for node, xi_idx in enumerate(xi_idx_psi):
        # Z0[rod.nodalDOF_psi[node]] = rod.RotationBase.Log_SO3(A_IKs[xi_idx])
        Z0[rod.nodalDOF_psi[node]] = QuaternionRotationParameterization.Log_SO3(A_IKs[xi_idx])

    # # TODO: Ensure that all quaternions are on the same hemisphere
    # # quats = np.array([
    # #     Z0[rod.nodalDOF_psi[node]] for node in range(rod.nnodes_psi)
    # # ])
    # # P0 = Z0[rod.nodalDOF_psi[0]]
    # # for i in range(1, rod.nnodes_psi):
    # #     Pi = Z0[rod.nodalDOF_psi[i]]
    # #     inner = P0 @ Pi
    # #     print(f"i: {i}")
    # #     if inner < 0:
    # #         print("wrong hemisphere!")
    # #         Z0[rod.nodalDOF_psi[i]] *= -1
    # #     else:
    # #         print(f"correct hemisphere")

    # if rod.RotationBase is QuaternionRotationParameterization:
    for i in range(rod.nnodes_psi - 1):
        Pi = Z0[rod.nodalDOF_psi[i]]
        Pi1 = Z0[rod.nodalDOF_psi[i + 1]]
        inner = Pi @ Pi1
        print(f"i: {i}")
        if inner < 0:
            print("wrong hemisphere!")
            Z0[rod.nodalDOF_psi[i + 1]] *= -1
        else:
            print(f"correct hemisphere")

    # constrained nodes
    # zDOF = np.arange(rod.nq)
    zDOF = np.arange(rod.nq_r + rod.nq_psi)
    cDOF_r = rod.nodalDOF_r[nodal_cDOF]
    cDOF_psi = rod.nodalDOF_psi[nodal_cDOF]
    cDOF = np.concatenate((cDOF_r.flatten(), cDOF_psi.flatten()))
    fDOF = np.setdiff1d(zDOF, cDOF)

    # generate redundant coordinates
    Z0_boundary_r = Z0[cDOF_r]
    Z0_boundary_psi = Z0[cDOF_psi]

    def make_redundant_coordinates(q):
        # z = np.zeros(rod.nq, dtype=q.dtype)
        z = np.zeros(rod.nq_r + rod.nq_psi, dtype=q.dtype)
        z[fDOF] = q
        z[cDOF_r] = Z0_boundary_r
        z[cDOF_psi] = Z0_boundary_psi
        return z

    def residual(q):
        z = make_redundant_coordinates(q)

        R = np.zeros(6 * n_samples, dtype=z.dtype)
        for i, xi in enumerate(xis):
            # find element number and extract elemt degrees of freedom
            el = rod.element_number(xi)
            # elDOF = rod.elDOF[el]
            elDOF = np.concatenate([rod.elDOF_r[el], rod.elDOF_psi[el]])
            ze = z[elDOF]

            # interpolate position and orientation
            r_OP, A_IK, _, _ = rod._eval(ze, xi)

            # compute homogeneous transformation
            H = SE3(A_IK, r_OP)

            # use relative twist as error measure
            R[6 * i : 6 * (i + 1)] = Log_SE3(SE3inv(Hs[i]) @ H)

        return R

    def jac(q):
        z = make_redundant_coordinates(q)

        J = np.zeros((6 * n_samples, len(z)), dtype=z.dtype)
        for i, xi in enumerate(xis):
            # find element number and extract elemt degrees of freedom
            el = rod.element_number(xi)
            # elDOF = rod.elDOF[el]
            elDOF = np.concatenate([rod.elDOF_r[el], rod.elDOF_psi[el]])

            ze = z[elDOF]

            # interpolate position and orientation
            (
                r_OP,
                A_IK,
                _,
                _,
                r_OP_ze,
                A_IK_ze,
                _,
                _,
            ) = rod._deval(ze, xi)

            # compute homogeneous transformation and derivative
            H = SE3(A_IK, r_OP)
            H_qe = np.zeros((4, 4, len(ze)))
            nodalDOF_element = np.concatenate([rod.nodalDOF_element_r.T.flatten(), rod.nodalDOF_element_psi.T.flatten()])
            H_qe[:3, :3, :] = A_IK_ze[:, :, nodalDOF_element]
            H_qe[:3, 3, :] = r_OP_ze[:, nodalDOF_element]

            # insert to jacobian
            J[6 * i : 6 * (i + 1), elDOF] = np.einsum(
                # "ijk,jkl->il",
                # Log_SE3_H(SE3inv(Hs[i]) @ H),
                # np.einsum("mn,nop->mop", SE3inv(Hs[i]), H_qe)
                #
                "ijk,jn,nkl->il",
                Log_SE3_H(SE3inv(Hs[i]) @ H),
                SE3inv(Hs[i]),
                H_qe,
            )

        # jacobian w.r.t. to minimal coordinates
        J = J[:, fDOF]
        return J

        # from cardillo.math import approx_fprime
        # J_num = approx_fprime(q, residual, method="cs")
        # diff = J - J_num
        # error = np.linalg.norm(diff)
        # print(f"error J: {error}")
        # return J_num

    res = least_squares(
        residual,
        Z0[fDOF],
        jac=jac,
        method="trf",
        verbose=verbose,
    )
    Q0 = res.x
    Z0 = np.zeros(rod.nq)
    Z0[:rod.nq_r + rod.nq_psi] = make_redundant_coordinates(Q0)

    rod.set_initial_strains(Z0)
    return Z0


def fit_configuration_quaternion(
    rod,
    r_OPs,
    A_IKs,
    nodal_cDOF=[0, -1],
    verbose=1,
):
    # number of sample points
    n_samples = len(r_OPs)

    # compute unit quaternions from transformation matrices
    Ps = np.array([Log_SO3_quat(A_IK) for A_IK in A_IKs])

    # ensure that all quaternions are on the same hemisphere
    # P_ref = Log_SO3_quat(Exp_SO3(np.random.rand(3))) # TODO: is this a good choice?
    P_ref = Ps[0]
    for i in range(n_samples):
        if P_ref @ Ps[i] < 0:
            Ps[i] *= -1

    # linear spaced evaluation points for the rod
    xis = np.linspace(0, 1, n_samples)

    # initial vector of generalized coordinates
    Z0 = np.zeros(rod.nq, dtype=float)

    ###########
    # warmstart
    ###########
    # initialized nodal unknowns with closest data w.r.t. xi values
    xi_idx_r = np.array(
        [
            np.where(xis >= rod.mesh_r.knot_vector.data[i])[0][0]
            for i in range(rod.mesh_r.nnodes)
        ]
    )
    xi_idx_psi = np.array(
        [
            np.where(xis >= rod.mesh_psi.knot_vector.data[i])[0][0]
            for i in range(rod.mesh_psi.nnodes)
        ]
    )

    for node, xi_idx in enumerate(xi_idx_r):
        Z0[rod.nodalDOF_r[node]] = r_OPs[xi_idx].copy()
    for node, xi_idx in enumerate(xi_idx_psi):
        Z0[rod.nodalDOF_psi[node]] = Ps[xi_idx].copy()

    # constrained nodes
    zDOF = np.arange(rod.nq)
    cDOF_r = rod.nodalDOF_r[nodal_cDOF]
    cDOF_psi = rod.nodalDOF_psi[nodal_cDOF]
    cDOF = np.concatenate((cDOF_r.flatten(), cDOF_psi.flatten()))
    fDOF = np.setdiff1d(zDOF, cDOF)

    # generate redundant coordinates
    Z0_boundary_r = Z0[cDOF_r]
    Z0_boundary_psi = Z0[cDOF_psi]

    def make_redundant_coordinates(q):
        z = np.zeros(rod.nq, dtype=q.dtype)
        z[fDOF] = q
        z[cDOF_r] = Z0_boundary_r
        z[cDOF_psi] = Z0_boundary_psi
        return z

    def residual(q):
        z = make_redundant_coordinates(q)

        R = np.zeros(7 * n_samples, dtype=z.dtype)
        # R = np.zeros(8 * n_samples, dtype=z.dtype)
        for i, xi in enumerate(xis):
            # find element number and extract elemt degrees of freedom
            el = rod.element_number(xi)
            elDOF = rod.elDOF[el]
            ze = z[elDOF]

            # evaluate shape functions
            N, _ = rod.basis_functions_r(xi)

            # interpolate position and quaternion
            r_OP = np.zeros(3)
            P = np.zeros(4)
            for j in range(rod.nnodes_element_r):
                r_OP += N[j] * ze[rod.nodalDOF_element_r[j]]
                P += N[j] * ze[rod.nodalDOF_element_psi[j]]

            # P /= norm(P)

            # difference in position and quaternion
            R[7 * i : 7 * i + 3] = r_OP - r_OPs[i]
            R[7 * i + 3 : 7 * (i + 1)] = P - Ps[i]
            # R[8 * i : 8 * i + 3] = r_OP - r_OPs[i]
            # R[8 * i + 3 : 8 * i + 7] = P - Ps[i]
            # R[8 * i + 7 : 8 * (i + 1)] = Ps[i] @ Ps[i] - 1

        return R

    # TODO
    def jac(q):
        pass

    res = least_squares(
        residual,
        Z0[fDOF],
        jac="3-point",
        # jac=jac,
        method="trf",
        # method="lm",
        verbose=verbose,
    )
    Q0 = res.x
    Z0 = make_redundant_coordinates(Q0)

    rod.set_initial_strains(Z0)
    return Z0
