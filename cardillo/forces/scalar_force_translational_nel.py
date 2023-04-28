import numpy as np
import numpy.typing as npt

from cardillo.math import norm


class nElScalarForceTranslational:
    def __init__(
        self,
        subsystem_list: list,
        connectivity: list[tuple[int, int]],
        force_law_spring=None,
        force_law_damper=None,
        frame_ID_list: list[npt.ArrayLike] = None,
        K_r_SP_list: list[npt.ArrayLike] = None,
    ) -> None:
        self.force_law_spring = force_law_spring
        self.force_law_damper = force_law_damper

        assert (self.force_law_spring is not None) or (
            self.force_law_damper is not None
        )

        if self.force_law_spring is not None:
            self.E_pot = lambda t, q: self.force_law_spring.E_pot(t, self._g(t, q))
            if self.force_law_damper is not None:
                self._h = lambda t, q, u: self._f_spring(t, q) + self._f_damper(t, q, u)
                self._h_q = lambda t, q, u: self._f_spring_q(t, q) + self._f_damper_q(
                    t, q, u
                )
                self.h_u = lambda t, q, u: self._f_damper_u(t, q, u)
            else:
                self._h = lambda t, q, u: self._f_spring(t, q)
                self._h_q = lambda t, q, u: self._f_spring_q(t, q)
        else:
            self._h = lambda t, q, u: self._f_damper(t, q, u)
            self._h_q = lambda t, q, u: self._f_damper_q(t, q, u)
            self.h_u = lambda t, q, u: self._f_damper_u(t, q, u)

        self.subsystems = subsystem_list
        self.n_subsystems = len(subsystem_list)
        self.frame_IDs = (
            frame_ID_list
            if frame_ID_list is not None
            else self.n_subsystems * [np.zeros(3)]
        )
        self.Ki_r_SPis = (
            K_r_SP_list
            if K_r_SP_list is not None
            else self.n_subsystems * [np.zeros(3)]
        )

        self.connectivity = connectivity

    def assembler_callback(self):
        self._nq: list[int] = []
        self._nu: list[int] = []

        self.qDOF = np.array([], dtype=int)
        self.uDOF = np.array([], dtype=int)

        for sys, frame_ID in zip(self.subsystems, self.frame_IDs):
            self._nq.append(len(self.qDOF))
            local_qDOFi = sys.local_qDOF_P(frame_ID)
            self.qDOF = np.concatenate([self.qDOF, sys.qDOF[local_qDOFi]])

            self._nu.append(len(self.uDOF))
            local_uDOFi = sys.local_uDOF_P(frame_ID)
            self.uDOF = np.concatenate([self.uDOF, sys.uDOF[local_uDOFi]])
        self._nq.append(len(self.qDOF))
        self._nu.append(len(self.uDOF))

        if self.force_law_spring is not None and self.force_law_spring.g_ref is None:
            g_ref = 0
            for i, j in self.connectivity:
                sysj = self.subsystems[j]
                sysi = self.subsystems[i]
                frame_IDj = self.frame_IDs[j]
                frame_IDi = self.frame_IDs[i]
                g_ref += norm(
                    sysj.r_OP(
                        sysj.t0,
                        sysj.q0[sysj.local_qDOF_P(frame_IDj)],
                        frame_IDj,
                        self.Ki_r_SPis[j],
                    )
                    - sysi.r_OP(
                        sysi.t0,
                        sysi.q0[sysi.local_qDOF_P(frame_IDi)],
                        frame_IDi,
                        self.Ki_r_SPis[i],
                    )
                )
            self.force_law_spring.g_ref = g_ref
            if self.force_law_spring.g_ref < 1e-6:
                raise ValueError(
                    "Computed g_ref from given subsystems is close to zero. Generalized force direction cannot be computed."
                )

        self.nq_fun = lambda k: range(*self._nq[k : k + 2])
        self.nu_fun = lambda k: range(*self._nu[k : k + 2])

        # auxiliary functions
        self.r_OPk = lambda t, q, k: self.subsystems[k].r_OP(
            t, q[self.nq_fun(k)], self.frame_IDs[k], self.Ki_r_SPis[k]
        )
        self.r_OPk_qk = lambda t, q, k: self.subsystems[k].r_OP_q(
            t, q[self.nq_fun(k)], self.frame_IDs[k], self.Ki_r_SPis[k]
        )
        self.v_Pk = lambda t, q, u, k: self.subsystems[k].v_P(
            t,
            q[self.nq_fun(k)],
            u[self.nu_fun(k)],
            self.frame_IDs[k],
            self.Ki_r_SPis[k],
        )
        self.v_Pk_qk = lambda t, q, u, k: self.subsystems[k].v_P_q(
            t,
            q[self.nq_fun(k)],
            u[self.nu_fun(k)],
            self.frame_IDs[k],
            self.Ki_r_SPis[k],
        )
        self.J_Pk = lambda t, q, k: self.subsystems[k].J_P(
            t, q[self.nq_fun(k)], self.frame_IDs[k], self.Ki_r_SPis[k]
        )
        self.J_Pk_qk = lambda t, q, k: self.subsystems[k].J_P_q(
            t, q[self.nq_fun(k)], self.frame_IDs[k], self.Ki_r_SPis[k]
        )
        self.r_PiPj = lambda t, q, i, j: self.r_OPk(t, q, j) - self.r_OPk(t, q, i)

    # auxiliary functions
    def _nij(self, t, q, i, j):
        r_PiPj = self.r_PiPj(t, q, i, j)
        return r_PiPj / norm(r_PiPj)

    def _nij_qij(self, t, q, i, j):
        r_PiPj = self.r_PiPj(t, q, i, j)
        gij = norm(r_PiPj)
        tmp = np.outer(r_PiPj, r_PiPj) / (gij**3)
        r_OPi_qi = self.r_OPk_qk(t, q, i)
        r_OPj_qj = self.r_OPk_qk(t, q, j)
        n_qi = -r_OPi_qi / gij + tmp @ r_OPi_qi
        n_qj = r_OPj_qj / gij - tmp @ r_OPj_qj
        return n_qi, n_qj

    def _g(self, t, q):
        g = 0
        for i, j in self.connectivity:
            g += norm(self.r_OPk(t, q, j) - self.r_OPk(t, q, i))
        return g

    def _g_q(self, t, q):
        g_q = np.zeros((self._nq[-1]), dtype=q.dtype)
        for i, j in self.connectivity:
            nij = self._nij(t, q, i, j)
            g_q[self.nq_fun(i)] += -nij @ self.r_OPk_qk(t, q, i)
            g_q[self.nq_fun(j)] += nij @ self.r_OPk_qk(t, q, j)
        return g_q

    def _gamma(self, t, q, u):
        gamma = 0
        for i, j in self.connectivity:
            gamma += self._nij(t, q, i, j) @ (
                self.v_Pk(t, q, u, j) - self.v_Pk(t, q, u, i)
            )
        return gamma

    def _gamma_q(self, t, q, u):
        gamma_q = np.zeros((self._nq[-1]), dtype=np.common_type(q, u))
        for i, j in self.connectivity:
            nij_qi, nij_qj = self._nij_qij(t, q, i, j)
            nij = self._nij(t, q, i, j)
            vi, vj = self.v_Pk(t, q, u, i), self.v_Pk(t, q, u, j)
            gamma_q[self.nq_fun(i)] += (vj - vi) @ nij_qi - nij @ self.v_Pk_qk(
                t, q, u, i
            )
            gamma_q[self.nq_fun(j)] += (vj - vi) @ nij_qj - nij @ self.v_Pk_qk(
                t, q, u, j
            )
        return gamma_q

    def _W(self, t, q):
        W = np.zeros((self._nu[-1]), dtype=q.dtype)
        for i, j in self.connectivity:
            nij = self._nij(t, q, i, j)
            W[self.nu_fun(i)] += -self.J_Pk(t, q, i).T @ nij
            W[self.nu_fun(j)] += self.J_Pk(t, q, j).T @ nij
        return W

    def _W_q(self, t, q):
        W_q = np.zeros((self._nu[-1], self._nq[-1]), dtype=q.dtype)
        for i, j in self.connectivity:
            nui, nui1, nuj, nuj1 = map(lambda k: self._nu[k], [i, i + 1, j, j + 1])
            nqi, nqi1, nqj, nqj1 = map(lambda k: self._nq[k], [i, i + 1, j, j + 1])
            nij = self._nij(t, q, i, j)
            nij_qi, nij_qj = self._nij_qij(t, q, i, j)
            J_Pi = self.J_Pk(t, q, i)
            J_Pj = self.J_Pk(t, q, j)
            W_q[nui:nui1, nqi:nqi1] += (
                np.einsum("i,ijk->jk", -nij, self.J_Pk_qk(t, q, i)) - J_Pi.T @ nij_qi
            )
            W_q[nuj:nuj1, nqj:nqj1] += (
                np.einsum("i,ijk->jk", nij, self.J_Pk_qk(t, q, j)) + J_Pj.T @ nij_qj
            )
            W_q[nui:nui1, nqj:nqj1] += -J_Pi.T @ nij_qj
            W_q[nuj:nuj1, nqi:nqi1] += J_Pj.T @ nij_qi
        return W_q

    def _f_spring(self, t, q):
        return -self._W(t, q) * self.force_law_spring.la(t, self._g(t, q))

    def _f_spring_q(self, t, q):
        f_spring_q = -self._W_q(t, q) * self.force_law_spring.la(
            t, self._g(t, q)
        ) - self.force_law_spring.la_g(t, self._g(t, q)) * np.outer(
            self._W(t, q), self._g_q(t, q)
        )
        # f_q_num = approx_fprime(q, lambda q: self._f_spring(t, q), method = "cs")
        # diff = np.linalg.norm(f_q_num - f_spring_q)
        # print(f"diff: {diff}")
        return f_spring_q

    def _f_damper(self, t, q, u):
        return -self._W(t, q) * self.force_law_damper.la(t, self._gamma(t, q, u))

    def _f_damper_q(self, t, q, u):
        gamma = self._gamma(t, q, u)
        f_damper_q = -self._W_q(t, q) * self.force_law_damper.la(
            t, gamma
        ) - self.force_law_damper.la_gamma(t, gamma) * np.outer(
            self._W(t, q), self._gamma_q(t, q, u)
        )
        return f_damper_q

    def _f_damper_u(self, t, q, u):
        W = self._W(t, q)
        return -self.force_law_damper.la_gamma(t, self._gamma(t, q, u)) * np.outer(W, W)

    def h(self, t, q, u):
        return self._h(t, q, u)

    def h_q(self, t, q, u):
        return self._h_q(t, q, u)

    # E_pot and h_u defined in init if necessary

    def export(self, sol_i, **kwargs):
        points = []
        for i, _ in enumerate(self.subsystems):
            points.append(self.r_OPk(sol_i.t, sol_i.q[self.qDOF], i))

        cells = [("line", self.connectivity)]
        h = self.h(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])
        la = -self._W(sol_i.t, sol_i.q[self.qDOF]).T @ h
        # n = self._n(sol_i.t, sol_i.q[self.qDOF])
        point_data = dict(la=len(self.subsystems) * [la])  # , n=[n, -n])
        g = self._g(sol_i.t, sol_i.q[self.qDOF])
        cell_data = dict(
            # n=[[n]],
            delta_g=[
                [
                    g - self.force_law_spring.g_ref
                    if self.force_law_spring is not None
                    else g
                ]
            ],
            la=[[la]]
            # g_dot=[[self._g_dot(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])]],
        )
        # if hasattr(self, "E_pot"):
        #     E_pot = self.E_pot(sol_i.t, sol_i.q[self.qDOF])
        #     cell_data["E_pot"] = [[E_pot, E_pot, E_pot]]

        return points, cells, point_data, None
