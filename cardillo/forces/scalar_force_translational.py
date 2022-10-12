import numpy as np
from cardillo.math.algebra import norm


class ScalarForceTranslational:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        force_law_spring=None,
        force_law_damper=None,
        frame_ID1=np.zeros(3, dtype=float),
        frame_ID2=np.zeros(3, dtype=float),
        K_r_SP1=np.zeros(3, dtype=float),
        K_r_SP2=np.zeros(3, dtype=float),
    ):
        self.force_law_spring = force_law_spring
        self.force_law_damper = force_law_damper

        assert (self.force_law_spring is not None) or (
            self.force_law_damper is not None
        )

        if self.force_law_spring is not None:
            self.E_pot = lambda t, q: self.force_law_spring.E_pot(t, self.__g(t, q))
            if self.force_law_damper is not None:
                self.h = lambda t, q, u: self.__f_pot(t, q) + self.__f_npot(t, q, u)
                self.h_q = lambda t, q, u, coo: self.__f_pot_q(
                    t, q, coo
                ) + self.__f_npot_q(t, q, u, coo)
                self.h_u = lambda t, q, u, coo: self.__f_npot_u(t, q, u, coo)
            else:
                self.h = lambda t, q, u: self.__f_pot(t, q)
                self.h_q = lambda t, q, u, coo: self.__f_pot_q(t, q, coo)
        else:
            self.h = lambda t, q, u: self.__f_npot(t, q, u)
            self.h_q = lambda t, q, u, coo: self.__f_npot_q(t, q, u, coo)
            self.h_u = lambda t, q, u, coo: self.__f_npot_u(t, q, u, coo)

        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.K_r_SP1 = K_r_SP1

        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2
        self.K_r_SP2 = K_r_SP2

    def assembler_callback(self):
        qDOF1 = self.subsystem1.qDOF
        qDOF2 = self.subsystem2.qDOF
        local_qDOF1 = self.subsystem1.local_qDOF_P(self.frame_ID1)
        local_qDOF2 = self.subsystem2.local_qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate((qDOF1[local_qDOF1], qDOF2[local_qDOF2]))
        self.__nq1 = nq1 = len(local_qDOF1)
        self.__nq2 = len(local_qDOF2)
        self.__nq = self.__nq1 + self.__nq2

        uDOF1 = self.subsystem1.uDOF
        uDOF2 = self.subsystem2.uDOF
        local_uDOF1 = self.subsystem1.local_uDOF_P(self.frame_ID1)
        local_uDOF2 = self.subsystem2.local_uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate((uDOF1[local_uDOF1], uDOF2[local_uDOF2]))
        self.__nu1 = nu1 = len(local_uDOF1)
        self.__nu2 = len(local_uDOF2)
        self.__nu = self.__nu1 + self.__nu2

        if (self.force_law_spring is not None) and (
            self.force_law_spring.g_ref is None
        ):
            r_OP10 = self.subsystem1.r_OP(
                self.subsystem1.t0,
                self.subsystem1.q0[local_qDOF1],
                self.frame_ID1,
                self.K_r_SP1,
            )
            r_OP20 = self.subsystem2.r_OP(
                self.subsystem2.t0,
                self.subsystem2.q0[local_qDOF2],
                self.frame_ID2,
                self.K_r_SP2,
            )
            self.force_law_spring.g_ref = norm(r_OP20 - r_OP10)
            if self.force_law_spring.g_ref < 1e-6:
                raise ValueError(
                    "Computed g_ref from given subsystems is close to zero. Generalized force direction cannot be computed."
                )

        self.r_OP1 = lambda t, q: self.subsystem1.r_OP(
            t, q[: self.__nq1], self.frame_ID1, self.K_r_SP1
        )
        self.r_OP1_q = lambda t, q: self.subsystem1.r_OP_q(
            t, q[: self.__nq1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1 = lambda t, q: self.subsystem1.J_P(
            t, q[: self.__nq1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1_q = lambda t, q: self.subsystem1.J_P_q(
            t, q[: self.__nq1], self.frame_ID1, self.K_r_SP1
        )
        self.v_P1 = lambda t, q, u: self.subsystem1.v_P(
            t, q[: self.__nq1], u[: self.__nu1], self.frame_ID1, self.K_r_SP1
        )
        self.v_P1_q = lambda t, q, u: self.subsystem1.v_P_q(
            t, q[: self.__nq1], u[: self.__nu1], self.frame_ID1, self.K_r_SP1
        )

        self.r_OP2 = lambda t, q: self.subsystem2.r_OP(
            t, q[self.__nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.r_OP2_q = lambda t, q: self.subsystem2.r_OP_q(
            t, q[self.__nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2 = lambda t, q: self.subsystem2.J_P(
            t, q[self.__nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2_q = lambda t, q: self.subsystem2.J_P_q(
            t, q[self.__nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.v_P2 = lambda t, q, u: self.subsystem2.v_P(
            t, q[self.__nq1 :], u[self.__nu1 :], self.frame_ID2, self.K_r_SP2
        )
        self.v_P2_q = lambda t, q, u: self.subsystem2.v_P_q(
            t, q[self.__nq1 :], u[self.__nu1 :], self.frame_ID2, self.K_r_SP2
        )

        self.__n = self.__n
        self.__n_q = self.__n_q

    # private functions

    def __g(self, t, q):
        return norm(self.r_OP2(t, q) - self.r_OP1(t, q))

    def __g_q(self, t, q):
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)

        n = self.__n(t, q)
        return np.hstack((-n @ r_OP1_q, n @ r_OP2_q))

    def __g_dot(self, t, q, u):
        return self.__n(t, q) @ (self.v_P2(t, q, u) - self.v_P1(t, q, u))

    def __g_dot_q(self, t, q, u):
        n_q1, n_q2 = self.__n_q(t, q)
        n = self.__n(t, q)
        v_P1 = self.v_P1(t, q, u)
        v_P2 = self.v_P2(t, q, u)

        nq1 = self.__nq1
        gamma_q = np.zeros(self.__nq)
        gamma_q[:nq1] = -n @ self.v_P1_q(t, q, u) + (v_P2 - v_P1) @ n_q1
        gamma_q[nq1:] = n @ self.v_P2_q(t, q, u) + (v_P2 - v_P1) @ n_q2
        return gamma_q

    def __n(self, t, q):
        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        return (r_OP2 - r_OP1) / norm(r_OP2 - r_OP1)

    def __n_q(self, t, q):
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)

        r_P1P2 = self.r_OP2(t, q) - self.r_OP1(t, q)
        g = norm(r_P1P2)
        tmp = np.outer(r_P1P2, r_P1P2) / (g**3)
        n_q1 = -r_OP1_q / g + tmp @ r_OP1_q
        n_q2 = r_OP2_q / g - tmp @ r_OP2_q

        return n_q1, n_q2

    def __W(self, t, q):
        n = self.__n(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        return np.concatenate([-J_P1.T @ n, J_P2.T @ n])

    def __W_q(self, t, q):
        nq1 = self.__nq1
        nu1 = self.__nu1
        n = self.__n(t, q)
        n_q1, n_q2 = self.__n_q(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        J_P1_q = self.J_P1_q(t, q)
        J_P2_q = self.J_P2_q(t, q)

        # dense blocks
        dense = np.zeros((self.__nu, self.__nq))
        dense[:nu1, :nq1] = -J_P1.T @ n_q1 + np.einsum("i,ijk->jk", -n, J_P1_q)
        dense[:nu1, nq1:] = -J_P1.T @ n_q2
        dense[nu1:, :nq1] = J_P2.T @ n_q1
        dense[nu1:, nq1:] = J_P2.T @ n_q2 + np.einsum("i,ijk->jk", n, J_P2_q)

        return dense

    def __f_pot(self, t, q):
        g = self.__g(t, q)
        return -self.__W(t, q) * self.force_law_spring.la(t, g)

    def __f_pot_q(self, t, q, coo):
        g = self.__g(t, q)
        dense = -self.force_law_spring.la(t, g) * self.__W_q(
            t, q
        ) - self.force_law_spring.la_g(t, g) * np.outer(
            self.__W(t, q), self.__g_q(t, q)
        )
        coo.extend(dense, (self.uDOF, self.qDOF))

    def __f_npot(self, t, q, u):
        g_dot = self.__g_dot(t, q, u)
        return -self.__W(t, q) * self.force_law_damper.la(t, g_dot)

    def __f_npot_q(self, t, q, u, coo):
        g_dot = self.__g_dot(t, q, u)
        dense = -self.force_law_damper.la(t, g_dot) * self.__W_q(t, q)
        coo.extend(dense, (self.uDOF, self.qDOF))

    def __f_npot_u(self, t, q, u, coo):
        gamma = self.__g_dot(t, q, u)
        dense = -self.force_law_damper.la_gamma(t, gamma) * np.outer(
            self.__W(t, q), self.__W(t, q)
        )
        coo.extend(dense, (self.uDOF, self.uDOF))
