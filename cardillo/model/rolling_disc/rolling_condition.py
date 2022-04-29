import numpy as np
from cardillo.math import cross3, norm, approx_fprime, Numerical_derivative


class Rolling_condition:
    def __init__(self, disc, la_gamma0=None):
        self.subsystem = disc

        self.nla_gamma = 3
        self.la_gamma0 = np.zeros(self.nla_gamma) if la_gamma0 is None else la_gamma0

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.qDOF_P()]
        self.uDOF = self.subsystem.qDOF[self.subsystem.uDOF_P()]

    def r_SA(self, t, q):
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
            t, q, u, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SA(t, q)
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
            t, q, K_r_SP=self.subsystem.A_IK(t, q).T @ self.r_SA(t, q)
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


class Rolling_condition_I_frame:
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


class Rolling_condition_I_frame_g_gamma:
    """Rolling condition for rigid disc mixed on position and velocity level."""

    def __init__(self, disc, la_g0=None, la_gamma0=None):
        self.disc = disc

        self.nla_g = 1
        self.nla_gamma = 2
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0
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

    # bilateral constraints on position level
    # r_OS + (r_SA)_z
    def g(self, t, q):
        return np.array([(self.disc.r_OP(t, q) + self.r_SA(t, q))[2]])

    # TODO: Compue time derivative!
    def g_dot(self, t, q, u):
        return self.disc.v_P(t, q, u, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q))[
            2
        ]

    # TODO: Compute time derivative g_ddot!
    def g_ddot(self, t, q, u, u_dot):
        # f = lambda t, q, u: self.g_dot(t, q, u)
        # return Numerical_derivative(f)._dot(t, q, u, u_dot)
        return self.disc.a_P(
            t, q, u, u_dot, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q)
        )[2]

    def g_q_dense(self, t, q):
        # return Numerical_derivative(self.g)._x(t, q)
        return approx_fprime(q, lambda q: self.g(t, q), method="2-point").reshape(
            self.nla_g, self.disc.nq
        )

    def g_qq_dense(self, t, q):
        return approx_fprime(
            q, lambda q: self.g_q_dense(t, q), method="3-point"
        ).reshape(self.nla_g, self.disc.nq, self.disc.nq)

    def g_q_T_mu_g(self, t, q, mu_g, coo):
        dense = np.einsum("ijk,i", self.g_qq_dense(t, q), mu_g)
        coo.extend(dense, (self.qDOF, self.qDOF))

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def g_dot_q(self, t, q, u, coo):
        coo.extend(
            approx_fprime(q, lambda q: self.g_dot(t, q, u), method="2-point").reshape(
                self.nla_g, self.disc.nq
            ),
            (self.la_gDOF, self.qDOF),
        )

    def g_ddot_q(self, t, q, u, u_dot, coo):
        coo.extend(
            approx_fprime(
                q, lambda q: self.g_ddot(t, q, u, u_dot), method="2-point"
            ).reshape(self.nla_g, self.disc.nq),
            (self.la_gDOF, self.qDOF),
        )

    def W_g_dense(self, t, q):
        return self.disc.J_P(t, q, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q)).T[
            :, 2
        ][:, None]

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        dense = approx_fprime(
            q, lambda q: self.W_g_dense(t, q) @ la_g, method="3-point"
        )
        coo.extend(dense, (self.uDOF, self.qDOF))

    # bilateral constraints on velocity level
    def gamma(self, t, q, u):
        return self.disc.v_P(t, q, u, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q))[
            :2
        ]

    # TODO: Compute time derivative gamma_dot!
    def gamma_dot(self, t, q, u, u_dot):
        f = lambda t, q, u: self.gamma(t, q, u)
        return Numerical_derivative(f)._dot(t, q, u, u_dot)

    def gamma_q(self, t, q, u, coo):
        dense = Numerical_derivative(self.gamma, order=2)._x(t, q, u)[:2]
        coo.extend(dense, (self.la_gammaDOF, self.qDOF))

    # TODO:
    def gamma_dot_q(self, t, q, u, u_dot, coo):
        raise NotImplementedError("")
        coo.extend(
            approx_fprime(
                q, lambda q: self.gamma_dot(t, q, u, u_dot), method="2-point"
            ).reshape(self.nla_gamma, self.disc.nq),
            (self.la_gammaDOF, self.qDOF),
        )

    def gamma_u_dense(self, t, q):
        return self.disc.J_P(t, q, K_r_SP=self.disc.A_IK(t, q).T @ self.r_SA(t, q))[:2]

    def gamma_u(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q), (self.la_gammaDOF, self.uDOF))

    def W_gamma(self, t, q, coo):
        coo.extend(self.gamma_u_dense(t, q).T, (self.uDOF, self.la_gammaDOF))

    def Wla_gamma_q(self, t, q, la_gamma, coo):
        dense = Numerical_derivative(
            lambda t, q: self.gamma_u_dense(t, q).T @ la_gamma
        )._x(t, q)
        coo.extend(dense, (self.uDOF, self.qDOF))
