import numpy as np

from scipy.optimize import minimize
from scipy.optimize import least_squares
from cardillo.math import SE3, SE3inv, Log_SE3


# TODO: Add clamping for first and last node
def fit_configuration(
    rod,
    r_OPs,
    A_IKs,
    use_cord_length=True,
    # nodal_cDOF=[0, -1],
):
    # # constrained nodes
    # zDOF = np.arange(rod.nnodes)
    # cDOF = np.asarray(cDOF, dtype=int)
    # cDOF = zDOF[cDOF]
    # fDOF = np.setdiff1d(zDOF, cDOF)

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
    Q0 = np.zeros(rod.nq, dtype=float)

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
        Q0[rod.nodalDOF_r[node]] = r_OPs[xi_idx]
    for node, xi_idx in enumerate(xi_idx_psi):
        Q0[rod.nodalDOF_psi[node]] = rod.RotationBase.Log_SO3(A_IKs[xi_idx])

    # TODO: We should always use the axis-angle rotation parametrization for
    # this fitting since it converges very good. Sadly, this breaks with the
    # current way the rod hierarchie is implemented.
    def residual(q):
        R = np.zeros(6 * n_samples, dtype=q.dtype)
        for i, xi in enumerate(xis):
            # find element number and extract elemt degrees of freedom
            el = rod.element_number(xi)
            elDOF = rod.elDOF[el]
            qe = q[elDOF]

            # interpolate position and orientation
            r_OP, A_IK, _, _ = rod._eval(qe, xi)

            # compute homogeneous transformation
            H = SE3(A_IK, r_OP)

            # use relative twist as error measure
            R[6 * i : 6 * (i + 1)] = Log_SE3(SE3inv(Hs[i]) @ H)

        return R

    def jac(q):
        J = np.zeros((6 * n_samples, len(q)), dtype=q.dtype)
        for i, xi in enumerate(xis):
            # # find element number and extract elemt degrees of freedom
            # el = node_vector.element_number(xii)[0]
            # elDOF = mesh.elDOF[el]
            # qe = q[elDOF]
            # nqe = len(qe)

            # # evaluate shape functions
            # N = mesh.eval_basis(xii)

            # # interpoalte rotations
            # A_IK = np.zeros((3, 3), dtype=float)
            # A_IK_qe = np.zeros((3, 3, nqe), dtype=float)
            # for node in range(mesh.nnodes_per_element):
            #     nodalDOF = mesh.nodalDOF_element[node]
            #     qe_node = qe[nodalDOF]
            #     A_IK += N[node] * Exp_SO3(qe_node)
            #     A_IK_qe[:, :, nodalDOF] += N[node] * Exp_SO3_psi(qe_node)

            # find element number and extract elemt degrees of freedom
            el = rod.element_number(xi)
            elDOF = rod.elDOF[el]
            qe = q[elDOF]

            # interpolate position and orientation
            # r_OP, A_IK, _, _ = rod._eval(qe, xi)
            (
                r_OP,
                A_IK,
                K_Gamma_bar,
                K_Kappa_bar,
                r_OP_qe,
                A_IK_qe,
                K_Gamma_bar_qe,
                K_Kappa_bar_qe,
            ) = rod._deval(qe, xi)

            H_qe = SE

            # compute relative rotation vector
            psi_rel_A_rel = Log_SO3_A(r_OPs[i].T @ A_IK)
            psi_rel_qe = np.einsum("ikl,mk,mlj->ij", psi_rel_A_rel, r_OPs[i], A_IK_qe)

            # insert to residual
            J[3 * i : 3 * (i + 1), elDOF] = psi_rel_qe

            # def psi_rel(qe):
            #     # evaluate shape functions
            #     N = mesh.eval_basis(xii)

            #     # interpoalte rotations
            #     A_IK = np.zeros((3, 3), dtype=q.dtype)
            #     for node in range(mesh.nnodes_per_element):
            #         A_IK += N[node] * Exp_SO3(qe[mesh.nodalDOF_element[node]])

            #     # compute relative rotation vector
            #     return Log_SO3(A_IKs[i].T @ A_IK)

            # psi_rel_qe_num = approx_fprime(qe, psi_rel)
            # diff = psi_rel_qe - psi_rel_qe_num
            # error = np.linalg.norm(diff)
            # print(f"error psi_rel_qe: {error}")

            # # insert to residual
            # J[3 * i:3 * (i + 1), elDOF] = psi_rel_qe_num

        # return J

        from cardillo.math import approx_fprime

        J_num = approx_fprime(q, residual)
        diff = J - J_num
        error = np.linalg.norm(diff)
        print(f"error J: {error}")
        return J_num

    res = least_squares(
        residual,
        Q0,
        # jac=jac,
        method="trf",
        verbose=2,
    )
    Q0 = res.x
    print(f"res: {res}")

    # def fun(q):
    #     r = residual(q)
    #     return 0.5 * (r @ r)

    # # method = "SLSQP"
    # method = "trust-constr"
    # sol = minimize(fun, Q0, method=method, options={"verbose": 2})
    # print(f"sol: {sol}")
    # Q0 = sol.x

    rod.set_initial_strains(Q0)

    return Q0
