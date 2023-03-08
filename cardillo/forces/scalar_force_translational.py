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
            self.E_pot = lambda t, q: self.force_law_spring.E_pot(t, self._g(t, q))
            if self.force_law_damper is not None:
                self._h = lambda t, q, u: self.__f_spring(t, q) + self.__f_damper(
                    t, q, u
                )
                self._h_q = lambda t, q, u, coo: coo.extend(
                    self.__f_spring_q(t, q) + self.__f_damper_q(t, q, u),
                    (self.uDOF, self.qDOF),
                )
                self.h_u = lambda t, q, u, coo: self.__f_damper_u(t, q, u, coo)
            else:
                self._h = lambda t, q, u: self.__f_spring(t, q)
                self._h_q = lambda t, q, u, coo: coo.extend(
                    self.__f_spring_q(t, q), (self.uDOF, self.qDOF)
                )
        else:
            self._h = lambda t, q, u: self.__f_damper(t, q, u)
            self._h_q = lambda t, q, u, coo: coo.extend(
                self.__f_damper_q(t, q, u), (self.uDOF, self.qDOF)
            )
            self.h_u = lambda t, q, u, coo: self.__f_damper_u(t, q, u, coo)

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
        self._nq1 = len(local_qDOF1)
        self._nq2 = len(local_qDOF2)
        self._nq = self._nq1 + self._nq2

        uDOF1 = self.subsystem1.uDOF
        uDOF2 = self.subsystem2.uDOF
        local_uDOF1 = self.subsystem1.local_uDOF_P(self.frame_ID1)
        local_uDOF2 = self.subsystem2.local_uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate((uDOF1[local_uDOF1], uDOF2[local_uDOF2]))
        self._nu1 = len(local_uDOF1)
        self._nu2 = len(local_uDOF2)
        self._nu = self._nu1 + self._nu2

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
            t, q[: self._nq1], self.frame_ID1, self.K_r_SP1
        )
        self.r_OP1_q = lambda t, q: self.subsystem1.r_OP_q(
            t, q[: self._nq1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1 = lambda t, q: self.subsystem1.J_P(
            t, q[: self._nq1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1_q = lambda t, q: self.subsystem1.J_P_q(
            t, q[: self._nq1], self.frame_ID1, self.K_r_SP1
        )
        self.v_P1 = lambda t, q, u: self.subsystem1.v_P(
            t, q[: self._nq1], u[: self._nu1], self.frame_ID1, self.K_r_SP1
        )
        self.v_P1_q = lambda t, q, u: self.subsystem1.v_P_q(
            t, q[: self._nq1], u[: self._nu1], self.frame_ID1, self.K_r_SP1
        )

        self.r_OP2 = lambda t, q: self.subsystem2.r_OP(
            t, q[self._nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.r_OP2_q = lambda t, q: self.subsystem2.r_OP_q(
            t, q[self._nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2 = lambda t, q: self.subsystem2.J_P(
            t, q[self._nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2_q = lambda t, q: self.subsystem2.J_P_q(
            t, q[self._nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.v_P2 = lambda t, q, u: self.subsystem2.v_P(
            t, q[self._nq1 :], u[self._nu1 :], self.frame_ID2, self.K_r_SP2
        )
        self.v_P2_q = lambda t, q, u: self.subsystem2.v_P_q(
            t, q[self._nq1 :], u[self._nu1 :], self.frame_ID2, self.K_r_SP2
        )

        self._n = self._n
        self._n_q = self._n_q

    # auxiliary functions
    def _g(self, t, q):
        return norm(self.r_OP2(t, q) - self.r_OP1(t, q))

    def _g_q(self, t, q):
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)

        n = self._n(t, q)
        return np.hstack((-n @ r_OP1_q, n @ r_OP2_q))

    def _g_dot(self, t, q, u):
        return self._n(t, q) @ (self.v_P2(t, q, u) - self.v_P1(t, q, u))

    def _g_dot_q(self, t, q, u):
        n_q1, n_q2 = self._n_q(t, q)
        n = self._n(t, q)
        v_P1 = self.v_P1(t, q, u)
        v_P2 = self.v_P2(t, q, u)

        nq1 = self._nq1
        gamma_q = np.zeros(self._nq)
        gamma_q[:nq1] = -n @ self.v_P1_q(t, q, u) + (v_P2 - v_P1) @ n_q1
        gamma_q[nq1:] = n @ self.v_P2_q(t, q, u) + (v_P2 - v_P1) @ n_q2
        return gamma_q

    def _n(self, t, q):
        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        return (r_OP2 - r_OP1) / norm(r_OP2 - r_OP1)

    def _n_q(self, t, q):
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)

        r_P1P2 = self.r_OP2(t, q) - self.r_OP1(t, q)
        g = norm(r_P1P2)
        tmp = np.outer(r_P1P2, r_P1P2) / (g**3)
        n_q1 = -r_OP1_q / g + tmp @ r_OP1_q
        n_q2 = r_OP2_q / g - tmp @ r_OP2_q

        return n_q1, n_q2

    def _W(self, t, q):
        n = self._n(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        return np.concatenate([-J_P1.T @ n, J_P2.T @ n])

    def _W_q(self, t, q):
        nq1 = self._nq1
        nu1 = self._nu1
        n = self._n(t, q)
        n_q1, n_q2 = self._n_q(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        J_P1_q = self.J_P1_q(t, q)
        J_P2_q = self.J_P2_q(t, q)

        # dense blocks
        dense = np.zeros((self._nu, self._nq))
        dense[:nu1, :nq1] = -J_P1.T @ n_q1 + np.einsum("i,ijk->jk", -n, J_P1_q)
        dense[:nu1, nq1:] = -J_P1.T @ n_q2
        dense[nu1:, :nq1] = J_P2.T @ n_q1
        dense[nu1:, nq1:] = J_P2.T @ n_q2 + np.einsum("i,ijk->jk", n, J_P2_q)

        return dense

    # private functions
    def __f_spring(self, t, q):
        g = self._g(t, q)
        return -self._W(t, q) * self.force_law_spring.la(t, g)

    def __f_spring_q(self, t, q):
        g = self._g(t, q)
        dense = -self.force_law_spring.la(t, g) * self._W_q(
            t, q
        ) - self.force_law_spring.la_g(t, g) * np.outer(self._W(t, q), self._g_q(t, q))
        return dense

    def __f_damper(self, t, q, u):
        g_dot = self._g_dot(t, q, u)
        return -self._W(t, q) * self.force_law_damper.la(t, g_dot)

    def __f_damper_q(self, t, q, u):
        g_dot = self._g_dot(t, q, u)
        dense = -self.force_law_damper.la(t, g_dot) * self._W_q(t, q)
        return dense
        # coo.extend(dense, (self.uDOF, self.qDOF))

    def __f_damper_u(self, t, q, u, coo):
        gamma = self._g_dot(t, q, u)
        dense = -self.force_law_damper.la_gamma(t, gamma) * np.outer(
            self._W(t, q), self._W(t, q)
        )
        coo.extend(dense, (self.uDOF, self.uDOF))

    # public functions
    def h(self, t, q, u):
        return self._h(t, q, u)
    
    def h_q(self, t, q, u, coo):
        self._h_q(t, q, u, coo)
    
    # E_pot and h_u defined in init if necessary

    def export(self, sol_i, **kwargs):
        points = [
            self.r_OP1(sol_i.t, sol_i.q[self.qDOF]),
            self.r_OP2(sol_i.t, sol_i.q[self.qDOF]),
        ]
        cells = [("line", [[0, 1]])]
        h = self._h(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])
        la = self._W(sol_i.t, sol_i.q[self.qDOF]).T @ h
        n = self._n(sol_i.t, sol_i.q[self.qDOF])
        point_data = dict(la=[la, la], n=[n, -n])
        # cell_data = dict(h=[h])
        cell_data = dict(
            n=[[n]],
            g=[[self._g(sol_i.t, sol_i.q[self.qDOF])]],
            g_dot=[[self._g_dot(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])]],
        )
        if hasattr(self, "E_pot"):
            E_pot = [self.E_pot(sol_i.t, sol_i.q[self.qDOF])]
            cell_data["E_pot"] = [E_pot]

        return points, cells, point_data, cell_data
