import numpy as np
from cardillo.math.algebra import norm


class Translational_f_pot:
    def __init__(
        self,
        force_law,
        subsystem1,
        subsystem2,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
        K_r_SP1=np.zeros(3),
        K_r_SP2=np.zeros(3),
        n=None,
    ):
        self.force_law = force_law

        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.K_r_SP1 = K_r_SP1

        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2
        self.K_r_SP2 = K_r_SP2

        self.n = n

    def assembler_callback(self):
        self.qDOF1 = self.subsystem1.qDOF[self.subsystem1.qDOF_P(self.frame_ID1)]
        self.qDOF2 = self.subsystem2.qDOF[self.subsystem2.qDOF_P(self.frame_ID2)]
        self.qDOF = np.concatenate([self.qDOF1, self.qDOF2])
        self.nq1 = len(self.qDOF1)
        self.nq2 = len(self.qDOF2)
        self.nq_ = self.nq1 + self.nq2

        if self.force_law.g0 is None:
            r_OP10 = self.subsystem1.r_OP(
                self.subsystem1.t0,
                self.subsystem1.q0[self.qDOF1],
                self.frame_ID1,
                self.K_r_SP1,
            )
            r_OP20 = self.subsystem2.r_OP(
                self.subsystem2.t0,
                self.subsystem2.q0[self.qDOF2],
                self.frame_ID2,
                self.K_r_SP2,
            )
            self.force_law.g0 = np.linalg.norm(r_OP20 - r_OP10)
            if self.force_law.g0 < 1e-6:
                raise ValueError("computed g0 from given subsystems is close to zero.")

        self.uDOF1 = self.subsystem1.uDOF[self.subsystem1.uDOF_P(self.frame_ID1)]
        self.uDOF2 = self.subsystem2.uDOF[self.subsystem2.uDOF_P(self.frame_ID2)]
        self.uDOF = np.concatenate([self.uDOF1, self.uDOF2])
        self.nu1 = len(self.uDOF1)
        self.nu2 = len(self.uDOF2)
        self.nu = self.nu1 + self.nu2

        self.r_OP1 = lambda t, q: self.subsystem1.r_OP(
            t, q[: self.nq1], self.frame_ID1, self.K_r_SP1
        )
        self.r_OP1_q = lambda t, q: self.subsystem1.r_OP_q(
            t, q[: self.nq1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1 = lambda t, q: self.subsystem1.J_P(
            t, q[: self.nq1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1_q = lambda t, q: self.subsystem1.J_P_q(
            t, q[: self.nq1], self.frame_ID1, self.K_r_SP1
        )

        self.r_OP2 = lambda t, q: self.subsystem2.r_OP(
            t, q[self.nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.r_OP2_q = lambda t, q: self.subsystem2.r_OP_q(
            t, q[self.nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2 = lambda t, q: self.subsystem2.J_P(
            t, q[self.nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2_q = lambda t, q: self.subsystem2.J_P_q(
            t, q[self.nq1 :], self.frame_ID2, self.K_r_SP2
        )

        if self.n is not None:
            if not callable(self.n):
                self.n = lambda t, n=self.n: n
            self.__n = lambda t, q: self.n(t)

            self.__n_q = lambda t, q: (np.zeros((3, self.nq1)), np.zeros((3, self.nq2)))
            self.__g = lambda t, q: self.n(t) @ (self.r_OP2(t, q) - self.r_OP1(t, q))
        else:
            self.__n = self.__n_PQ
            self.__n_q = self.__n_PQ_q
            self.__g = lambda t, q: norm(self.r_OP2(t, q) - self.r_OP1(t, q))

    def __n_PQ(self, t, q):
        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        return (r_OP2 - r_OP1) / norm(r_OP2 - r_OP1)

    def __n_PQ_q(self, t, q):
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)

        r_P1P2 = self.r_OP2(t, q) - self.r_OP1(t, q)
        g = norm(r_P1P2)
        tmp = np.outer(r_P1P2, r_P1P2) / (g**3)
        n_q1 = -r_OP1_q / g + tmp @ r_OP1_q
        n_q2 = r_OP2_q / g - tmp @ r_OP2_q

        return n_q1, n_q2

    def __g_q(self, t, q):
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)

        n = self.__n(t, q)
        return np.hstack((-n @ r_OP1_q, n @ r_OP2_q))

    def __W(self, t, q):
        n = self.__n(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        return np.concatenate([-J_P1.T @ n, J_P2.T @ n])

    def __W_q(self, t, q):
        nq1 = self.nq1
        nu1 = self.nu1
        n = self.__n(t, q)
        n_q1, n_q2 = self.__n_q(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        J_P1_q = self.J_P1_q(t, q)
        J_P2_q = self.J_P2_q(t, q)

        # dense blocks
        dense = np.zeros((self.nu, self.nq))
        dense[:nu1, :nq1] = -J_P1.T @ n_q1 + np.einsum("i,ijk->jk", -n, J_P1_q)
        dense[:nu1, nq1:] = -J_P1.T @ n_q2
        dense[nu1:, :nq1] = J_P2.T @ n_q1
        dense[nu1:, nq1:] = J_P2.T @ n_q2 + np.einsum("i,ijk->jk", n, J_P2_q)

        return dense

    # public functions
    def pot(self, t, q):
        return self.force_law.pot(t, self.__g(t, q))

    def f_pot(self, t, q):
        g = self.__g(t, q)
        return -self.__W(t, q) * self.force_law.pot_g(t, g)

    def f_pot_q(self, t, q, coo):
        g = self.__g(t, q)
        dense = -self.__W_q(t, q) * self.force_law.pot_g(t, g) - self.force_law.pot_gg(
            t, g
        ) * np.outer(self.__W(t, q), self.__g_q(t, q))
        coo.extend(dense, (self.uDOF, self.qDOF))
