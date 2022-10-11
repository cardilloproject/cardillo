import numpy as np
from cardillo.math import e1, e2, e3, cross3, ax2skew, norm, approx_fprime


class RollingCondition:
    def __init__(self, disc, la_gamma0=None):
        self.subsystem = disc

        self.nla_gamma = 3
        self.la_gamma0 = np.zeros(self.nla_gamma) if la_gamma0 is None else la_gamma0

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.qDOF_P()]
        self.uDOF = self.subsystem.qDOF[self.subsystem.uDOF_P()]

    def r_SC(self, t, q):
        e_K_y = self.subsystem.A_IK(t, q)[:, 1]
        g_K_x = cross3(e_K_y, np.array([0, 0, 1]))
        e_K_x = g_K_x / norm(g_K_x)
        e_K_z = cross3(e_K_x, e_K_y)
        return -self.subsystem.r * e_K_z

    def A_RI(self, t, q):
        e_K_y = self.subsystem.A_IK(t, q)[:, 1]
        g_R_x = cross3(e_K_y, np.array([0, 0, 1]))
        e_R_x = g_R_x / norm(g_R_x)

        e_R_z = np.array([0, 0, 1])
        e_R_y = cross3(e_R_z, e_R_x)

        return np.vstack((e_R_x, e_R_y, e_R_z))

    def gamma(self, t, q, u):
        return self.subsystem.v_P(
            t, q, u, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SC(t, q)
        )

    def gamma_dot(self, t, q, u, u_dot):
        gamma_q = approx_fprime(q, lambda q: self.gamma(t, q, u), method="2-point")
        gamma_u = gamma_u = self.gamma_u_dense(t, q)

        return gamma_q @ self.subsystem.q_dot(t, q, u) + gamma_u @ u_dot

    def gamma_q(self, t, q, u, coo):
        dense = approx_fprime(q, lambda q: self.gamma(t, q, u), method="2-point")
        coo.extend(dense, (self.la_gammaDOF, self.qDOF))

    def gamma_u_dense(self, t, q):
        return self.subsystem.J_P(
            t, q, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SC(t, q)
        )

        # gamma_u_dense = self.subsystem.J_P(t, q, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SA(t, q))
        # gamma_u_dense_num = Numerical_derivative(self.gamma)._y(t, q, np.zeros(self.subsystem.nu))
        # error = np.max(np.abs(gamma_u_dense_num - gamma_u_dense))
        # print(f'error gamma_u_dense: {error}')
        # return gamma_u_dense_num

    def gamma_u(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q), (self.la_gammaDOF, self.uDOF))

    def W_gamma(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q).T, (self.uDOF, self.la_gammaDOF))

    def Wla_gamma_q(self, t, q, la_gamma, coo):
        dense = approx_fprime(
            q, lambda q: self.gamma_u_dense(t, q).T @ la_gamma, method="2-point"
        )
        coo.extend(dense, (self.uDOF, self.qDOF))


class RollingCondition_I_Frame:
    def __init__(self, disc, la_gamma0=None):
        self.disc = disc

        self.nla_gamma = 3
        self.la_gamma0 = np.zeros(self.nla_gamma) if la_gamma0 is None else la_gamma0

    def assembler_callback(self):
        self.qDOF = self.disc.qDOF_P()
        self.uDOF = self.disc.uDOF_P()

    def r_SA(self, t, q):
        e_K_y = self.disc.A_IK(t, q)[:, 1]
        g_K_x = cross3(e_K_y, np.array([0, 0, 1]))
        e_K_x = g_K_x / norm(g_K_x)
        e_K_z = cross3(e_K_x, e_K_y)
        return -self.disc.r * e_K_z

    def A_RI(self, t, q):
        e_K_y = self.disc.A_IK(t, q)[:, 1]
        g_R_x = cross3(e_K_y, np.array([0, 0, 1]))
        e_R_x = g_R_x / norm(g_R_x)

        e_R_z = np.array([0, 0, 1])
        e_R_y = cross3(e_R_z, e_R_x)

        return np.vstack((e_R_x, e_R_y, e_R_z))

    def gamma(self, t, q, u):
        return self.disc.v_P(t, q, u, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q))

    def gamma_q(self, t, q, u, coo):
        dense = approx_fprime(q, lambda q: self.gamma(t, q, u), method="2-point")
        coo.extend(dense, (self.la_gammaDOF, self.qDOF))

    def gamma_u_dense(self, t, q):
        return self.disc.J_P(t, q, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q))

    def gamma_u(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q), (self.la_gammaDOF, self.uDOF))

    def W_gamma(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q).T, (self.uDOF, self.la_gammaDOF))

    def Wla_gamma_q(self, t, q, la_gamma, coo):
        dense = approx_fprime(
            q, lambda q: self.gamma_u_dense(t, q).T @ la_gamma, method="2-point"
        )
        coo.extend(dense, (self.uDOF, self.qDOF))


class RollingCondition_g_I_Frame_gamma:
    """Rolling condition for rigid disc mixed on position and velocity level."""

    def __init__(self, subsystem, la_g0=None, la_gamma0=None):
        self.subsystem = subsystem

        self.nla_g = 1
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0
        self.nla_gamma = 2
        self.la_gamma0 = np.zeros(self.nla_gamma) if la_gamma0 is None else la_gamma0

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF_P()
        self.uDOF = self.subsystem.uDOF_P()

    def r_SC(self, t, q):
        # evaluate body fixed frame
        e_K_x, e_K_y, e_K_z = self.subsystem.A_IK(t, q).T

        # compute e_x axis of grinding-(G)-frame, see LeSaux2005 (2.11)
        g_G_x = cross3(e_K_y, e3)
        e_G_x = g_G_x / norm(g_G_x)

        # compute e_z axis of G-frame, see LeSaux2005 (2.12)
        e_G_z = cross3(e_G_x, e_K_y)

        # contact point is - radius * e_z axis of grinding frame, see LeSaux2005 (2.13)
        return -self.subsystem.r * e_G_z

    #################
    # non penetration
    #################
    def g(self, t, q):
        # see LeSaux2005 (2.15a)
        r_OS = self.subsystem.r_OP(t, q)
        r_OC = r_OS + self.r_SC(t, q)
        return r_OC @ e3

    def g_dot(self, t, q, u):
        # v_C = self.subsystem.v_P(
        #     t, q, u, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SC(t, q)
        # )
        # return v_C @ e3

        g_q = self.g_q_dense(t, q)
        return g_q @ self.subsystem.q_dot(t, q, u)

    def g_ddot(self, t, q, u, u_dot):
        # a_C = self.subsystem.a_P(
        #     t, q, u, u_dot, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SC(t, q)
        # )
        # return a_C @ e3

        g_dot_q = approx_fprime(q, lambda q: self.g_dot(t, q, u), method="2-point")
        g_dot_u = self.W_g_dense(t, q).T

        return g_dot_q @ self.subsystem.q_dot(t, q, u) + g_dot_u @ u_dot

    # TODO:
    def g_q_dense(self, t, q):
        return approx_fprime(q, lambda q: self.g(t, q), method="2-point").reshape(
            self.nla_g, self.subsystem.nq
        )

    # TODO:
    def g_qq_dense(self, t, q):
        return approx_fprime(
            q, lambda q: self.g_q_dense(t, q), method="3-point"
        ).reshape(self.nla_g, self.subsystem.nq, self.subsystem.nq)

    def g_q_T_mu_g(self, t, q, mu_g, coo):
        dense = np.einsum("ijk,i", self.g_qq_dense(t, q), mu_g)
        coo.extend(dense, (self.qDOF, self.qDOF))

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    # TODO:
    def g_dot_q(self, t, q, u, coo):
        coo.extend(
            approx_fprime(q, lambda q: self.g_dot(t, q, u), method="2-point").reshape(
                self.nla_g, self.subsystem.nq
            ),
            (self.la_gDOF, self.qDOF),
        )

    # TODO:
    def g_ddot_q(self, t, q, u, u_dot, coo):
        coo.extend(
            approx_fprime(
                q, lambda q: self.g_ddot(t, q, u, u_dot), method="2-point"
            ).reshape(self.nla_g, self.subsystem.nq),
            (self.la_gDOF, self.qDOF),
        )

    def W_g_dense(self, t, q):
        J_C = self.subsystem.J_P(
            t, q, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SC(t, q)
        )
        return (e3 @ J_C).reshape(-1, self.nla_g)

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        # J_C_q = self.subsystem.J_P_q(
        #     t, q, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SC(t, q)
        # )
        # dense = np.einsum("k,kij", e3 * la_g[0], J_C_q)
        # # TODO: A_IK.T term is missing!

        dense_num = approx_fprime(
            q, lambda q: self.W_g_dense(t, q) @ la_g, method="2-point"
        )
        # diff = dense - dense_num
        # error = np.linalg.norm(diff)
        # print(f"error Wla_g_q: {error}")
        coo.extend(dense_num, (self.uDOF, self.qDOF))

    ########################
    # no in plane velocities
    ########################
    # if False:
    if True:

        def gamma(self, t, q, u):
            v_C = self.subsystem.v_P(
                t, q, u, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SC(t, q)
            )
            return np.array([v_C @ e1, v_C @ e2])

        def gamma_dot(self, t, q, u, u_dot):
            gamma_q = approx_fprime(q, lambda q: self.gamma(t, q, u), method="2-point")
            gamma_u = gamma_u = self.gamma_u_dense(t, q)

            return gamma_q @ self.subsystem.q_dot(t, q, u) + gamma_u @ u_dot

            # a_C = self.subsystem.a_P(
            #     t, q, u, u_dot, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SC(t, q)
            # )
            # return np.array([a_C @ e1, a_C @ e2])

        # TODO:
        def gamma_q(self, t, q, u, coo):
            # K_Omega = self.subsystem.K_Omega(t, q)
            # dense = (
            #     self.subsystem.v_P_q(
            #         t, q, u, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SC(t, q)
            #     )[:2]
            #     + np.einsum("", ax2skew(K_Omega), ...)
            # )

            dense_num = approx_fprime(
                q, lambda q: self.gamma(t, q, u), method="3-point"
            )
            # diff = dense - dense_num
            # error = np.linalg.norm(diff)
            # print(f"error gamma_q: {error}")
            coo.extend(dense_num, (self.la_gammaDOF, self.qDOF))

        # TODO:
        def gamma_dot_q(self, t, q, u, u_dot, coo):
            raise NotImplementedError("")
            coo.extend(
                approx_fprime(
                    q, lambda q: self.gamma_dot(t, q, u, u_dot), method="2-point"
                ).reshape(self.nla_gamma, self.subsystem.nq),
                (self.la_gammaDOF, self.qDOF),
            )

        def gamma_u_dense(self, t, q):
            return self.subsystem.J_P(
                t, q, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SC(t, q)
            )[:2]

        def gamma_u(self, t, q, coo):
            coo.extend(self.gamma_u_dense(t, q), (self.la_gammaDOF, self.uDOF))

        def W_gamma(self, t, q, coo):
            coo.extend(self.gamma_u_dense(t, q).T, (self.uDOF, self.la_gammaDOF))

        # TODO:
        def Wla_gamma_q(self, t, q, la_gamma, coo):
            dense = approx_fprime(
                q, lambda q: self.gamma_u_dense(t, q).T @ la_gamma, method="2-point"
            )
            coo.extend(dense, (self.uDOF, self.qDOF))
