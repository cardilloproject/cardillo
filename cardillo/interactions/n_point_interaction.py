import numpy as np
from cardillo.math import norm
from vtk import VTK_LINE


class nPointInteraction:
    def __init__(
        self,
        subsystem_list,
        connectivity,
        xi_list=None,
        B_r_CP_list=None,
    ) -> None:
        raise NotImplementedError("This class is not tested yet.")
        self.subsystems = subsystem_list
        self.n_subsystems = len(subsystem_list)
        self.xis = xi_list if xi_list is not None else self.n_subsystems * [np.zeros(3)]
        self.Bi_r_CPis = (
            B_r_CP_list
            if B_r_CP_list is not None
            else self.n_subsystems * [np.zeros(3)]
        )

        self.connectivity = connectivity

    def assembler_callback(self):
        self._nq: list[int] = []
        self._nu: list[int] = []

        self.qDOF = np.array([], dtype=int)
        self.uDOF = np.array([], dtype=int)

        for sys, xi in zip(self.subsystems, self.xis):
            self._nq.append(len(self.qDOF))
            local_qDOFi = sys.local_qDOF_P(xi)
            self.qDOF = np.concatenate([self.qDOF, sys.qDOF[local_qDOFi]])

            self._nu.append(len(self.uDOF))
            local_uDOFi = sys.local_uDOF_P(xi)
            self.uDOF = np.concatenate([self.uDOF, sys.uDOF[local_uDOFi]])
        self._nq.append(len(self.qDOF))
        self._nu.append(len(self.uDOF))

        self.nq_fun = lambda k: range(*self._nq[k : k + 2])
        self.nu_fun = lambda k: range(*self._nu[k : k + 2])

        # auxiliary functions
        self.r_OPk = lambda t, q, k: self.subsystems[k].r_OP(
            t, q[self.nq_fun(k)], self.xis[k], self.Bi_r_CPis[k]
        )
        self.r_OPk_qk = lambda t, q, k: self.subsystems[k].r_OP_q(
            t, q[self.nq_fun(k)], self.xis[k], self.Bi_r_CPis[k]
        )
        self.v_Pk = lambda t, q, u, k: self.subsystems[k].v_P(
            t,
            q[self.nq_fun(k)],
            u[self.nu_fun(k)],
            self.xis[k],
            self.Bi_r_CPis[k],
        )
        self.v_Pk_qk = lambda t, q, u, k: self.subsystems[k].v_P_q(
            t,
            q[self.nq_fun(k)],
            u[self.nu_fun(k)],
            self.xis[k],
            self.Bi_r_CPis[k],
        )
        self.J_Pk = lambda t, q, k: self.subsystems[k].J_P(
            t, q[self.nq_fun(k)], self.xis[k], self.Bi_r_CPis[k]
        )
        self.J_Pk_qk = lambda t, q, k: self.subsystems[k].J_P_q(
            t, q[self.nq_fun(k)], self.xis[k], self.Bi_r_CPis[k]
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

    def l(self, t, q):
        g = 0
        for i, j in self.connectivity:
            g += norm(self.r_OPk(t, q, j) - self.r_OPk(t, q, i))
        return g

    def l_q(self, t, q):
        g_q = np.zeros((self._nq[-1]), dtype=q.dtype)
        for i, j in self.connectivity:
            nij = self._nij(t, q, i, j)
            g_q[self.nq_fun(i)] += -nij @ self.r_OPk_qk(t, q, i)
            g_q[self.nq_fun(j)] += nij @ self.r_OPk_qk(t, q, j)
        return g_q

    def l_dot(self, t, q, u):
        gamma = 0
        for i, j in self.connectivity:
            gamma += self._nij(t, q, i, j) @ (
                self.v_Pk(t, q, u, j) - self.v_Pk(t, q, u, i)
            )
        return gamma

    def l_dot_q(self, t, q, u):
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

    def W_l(self, t, q):
        W = np.zeros((self._nu[-1]), dtype=q.dtype)
        for i, j in self.connectivity:
            nij = self._nij(t, q, i, j)
            W[self.nu_fun(i)] += -self.J_Pk(t, q, i).T @ nij
            W[self.nu_fun(j)] += self.J_Pk(t, q, j).T @ nij
        return W

    def W_l_q(self, t, q):
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

    def export(self, sol_i, **kwargs):
        points = []
        for i, _ in enumerate(self.subsystems):
            points.append(self.r_OPk(sol_i.t, sol_i.q[self.qDOF], i))

        cells = [(VTK_LINE, con) for con in self.connectivity]
        g = self.l(sol_i.t, sol_i.q[self.qDOF])
        gamma = self.l_dot(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])
        la = self._la(sol_i.t, g, gamma)
        point_data = dict(la=len(self.subsystems) * [[la]])  # , n=[n, -n])
        cell_data = dict(
            delta_g=[
                len(self.connectivity)
                * [
                    [
                        (
                            g - self.force_law_spring.g_ref
                            if self.force_law_spring is not None
                            else g
                        )
                    ]
                ]
            ],
            gamma=len(self.connectivity) * [[gamma]],
            la=len(self.connectivity) * [[la]],
        )
        # if hasattr(self, "E_pot"):
        #     E_pot = self.E_pot(sol_i.t, sol_i.q[self.qDOF])
        #     cell_data["E_pot"] = [[E_pot, E_pot, E_pot]]

        return points, cells, point_data, cell_data
