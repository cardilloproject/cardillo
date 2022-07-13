import numpy as np
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import splu
from tqdm import tqdm

from cardillo.solver import Solution
from cardillo.math import prox_R0_np, prox_sphere


class Moreau:
    def __init__(
        self,
        model,
        t1,
        dt,
        fix_point_tol=1e-8,
        fix_point_max_iter=1000,
        error_function=lambda x: np.max(np.abs(x)),
    ):
        self.model = model

        # integration time
        t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.fix_point_error_function = error_function
        self.fix_point_tol = fix_point_tol
        self.fix_point_max_iter = fix_point_max_iter

        self.nq = self.model.nq
        self.nu = self.model.nu
        self.nla_g = self.model.nla_g
        self.nla_gamma = self.model.nla_gamma
        self.nla_N = self.model.nla_N
        self.nla_F = self.model.nla_F
        self.nR_smooth = self.nu + self.nla_g + self.nla_gamma
        self.nR = self.nR_smooth + self.nla_N + self.nla_F

        self.tk = t0
        self.qk = model.q0
        self.uk = model.u0
        self.P_Nk = model.la_N0 * dt
        self.P_Fk = model.la_F0 * dt

        # connectivity matrix of normal force directions and friction force directions
        self.NF_connectivity = self.model.NF_connectivity

        if hasattr(model, "step_callback"):
            self.step_callback = model.step_callback
        else:
            self.step_callback = self.__step_callback

        # initial velocites
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        # solve for consistent initial accelerations and Lagrange mutlipliers
        M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        h0 = self.model.h(self.tk, self.qk, self.uk)
        W_g0 = self.model.W_g(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_gamma0 = self.model.W_gamma(self.tk, self.qk, scipy_matrix=csr_matrix)
        zeta_g0 = self.model.zeta_g(t0, self.qk, self.uk)
        zeta_gamma0 = self.model.zeta_gamma(t0, self.qk, self.uk)
        A = bmat(
            [[M0, -W_g0, -W_gamma0], [W_g0.T, None, None], [W_gamma0.T, None, None]],
            format="csc",
        )
        b = np.concatenate([h0, -zeta_g0, -zeta_gamma0])
        u_dot_la_g_la_gamma = splu(A).solve(b)
        self.u_dotk = u_dot_la_g_la_gamma[: self.nu]
        self.P_gk = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
        self.P_gammak = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

        # check if initial conditions satisfy constraints on position, velocity
        # and acceleration level
        g0 = model.g(self.tk, self.qk)
        g_dot0 = model.g_dot(self.tk, self.qk, self.uk)
        g_ddot0 = model.g_ddot(self.tk, self.qk, self.uk, self.u_dotk)
        gamma0 = model.gamma(self.tk, self.qk, self.uk)
        gamma_dot0 = model.gamma_dot(self.tk, self.qk, self.uk, self.u_dotk)

        assert np.allclose(
            g0, np.zeros(self.nla_g)
        ), "Initial conditions do not fulfill g0!"
        assert np.allclose(
            g_dot0, np.zeros(self.nla_g)
        ), "Initial conditions do not fulfill g_dot0!"
        assert np.allclose(
            g_ddot0, np.zeros(self.nla_g)
        ), "Initial conditions do not fulfill g_ddot0!"
        assert np.allclose(
            gamma0, np.zeros(self.nla_gamma)
        ), "Initial conditions do not fulfill gamma0!"
        assert np.allclose(
            gamma_dot0, np.zeros(self.nla_gamma)
        ), "Initial conditions do not fulfill gamma_dot0!"

    def __step_callback(self, q, u):
        return q, u

    def step(self):
        # general quantities
        dt = self.dt
        uk = self.uk
        tk1 = self.tk + dt

        # position update
        qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, uk)

        # get quantities from model
        M = self.model.M(tk1, qk1)
        h = self.model.h(tk1, qk1, uk)
        W_g = self.model.W_g(tk1, qk1)
        W_gamma = self.model.W_gamma(tk1, qk1)
        chi_g = self.model.g_dot(tk1, qk1, np.zeros_like(uk))
        chi_gamma = self.model.gamma(tk1, qk1, np.zeros_like(uk))
        W_N = self.model.W_N(
            tk1, qk1, scipy_matrix=csr_matrix
        )  # csr for column slicing
        W_F = self.model.W_F(
            tk1, qk1, scipy_matrix=csr_matrix
        )  # csr for column slicing
        prox_r_N = self.model.prox_r_N
        prox_r_F = self.model.prox_r_F
        mu = self.model.mu

        # identify active normal and tangential contacts
        g_N = self.model.g_N(tk1, qk1)
        I_N = g_N <= 0
        if np.any(I_N):
            I_F = np.array(
                [
                    c
                    for i, I_N_i in enumerate(I_N)
                    for c in self.model.NF_connectivity[i]
                    if I_N_i
                ],
                dtype=int,
            )
        else:
            I_F = np.array([], dtype=int)

        # solve for new velocities and bilateral constraint forces
        # M (uk1 - uk) - dt h + W_g P_g + W_gamma P_gamma + W_gN P_N + W_gT P_T = 0
        # W_g.T @ uk1 + chi_g = 0
        # W_gamma.T @ uk1 + chi_gamma = 0
        # fmt: off
        A = bmat([[        M, -W_g, -W_gamma], \
                  [    W_g.T, None,     None], \
                  [W_gamma.T, None,     None]], format="csc")
        # fmt: on

        # perform LU decomposition only once since matrix A is constant in
        # each time step saves alot work in the fixed point iteration
        lu = splu(A)

        # initial right hand side
        b = np.concatenate(
            (
                M @ uk
                + dt * h
                + W_N[:, I_N] @ self.P_Nk[I_N]
                + W_F[:, I_F] @ self.P_Fk[I_F],
                -chi_g,
                -chi_gamma,
            )
        )

        # solve for initial velocities and Lagrange multipliers of bilateral constraints for the fixed point iteration
        # x = lu_solve((lu, piv), b)
        x = lu.solve(b)
        uk1 = x[: self.nu]
        P_gk1 = x[self.nu : self.nu + self.nla_g]
        P_gammak1 = x[self.nu + self.nla_g :]

        P_Nk1 = np.zeros(self.nla_N)
        P_Fk1 = np.zeros(self.nla_F)

        converged = True
        error = 0
        j = 0
        uk_fixed_point = uk
        # if any contact is active
        if np.any(I_N):
            converged = False
            P_Nk1_i = self.P_Nk.copy()
            P_Nk1_i1 = self.P_Nk.copy()
            P_Fk1_i = self.P_Fk.copy()
            P_Fk1_i1 = self.P_Fk.copy()
            # fixed-point iterations
            for j in range(self.fix_point_max_iter):

                # fixed-point update normal direction
                P_Nk1_i1[I_N] = prox_R0_np(
                    P_Nk1_i[I_N]
                    - prox_r_N[I_N] * self.model.xi_N(tk1, qk1, uk, uk1)[I_N]
                )

                # fixed-point update friction
                xi_F = self.model.xi_F(tk1, qk1, uk, uk1)
                for i_N, i_F in enumerate(self.NF_connectivity):
                    if I_N[i_N] and len(i_F):
                        P_Fk1_i1[i_F] = prox_sphere(
                            P_Fk1_i[i_F] - prox_r_F[i_N] * xi_F[i_F],
                            mu[i_N] * P_Nk1_i1[i_N],
                        )

                # update rhs
                b = np.concatenate(
                    (
                        M @ uk
                        + dt * h
                        + W_N[:, I_N] @ P_Nk1_i1[I_N]
                        + W_F[:, I_F] @ P_Fk1_i1[I_F],
                        -chi_g,
                        -chi_gamma,
                    )
                )

                # solve for new velocities and Lagrange multipliers of bilateral constraints
                x = lu.solve(b)
                uk1 = x[: self.nu]
                P_gk1 = x[self.nu : self.nu + self.nla_g]
                P_gammak1 = x[self.nu + self.nla_g :]

                # check for convergence
                error = self.fix_point_error_function(uk1 - uk_fixed_point)
                uk_fixed_point = uk1
                converged = error < self.fix_point_tol  # TODO: rtol, atol?
                if converged:
                    P_Nk1[I_N] = P_Nk1_i1[I_N]
                    P_Fk1[I_F] = P_Fk1_i1[I_F]
                    break
                P_Nk1_i = P_Nk1_i1.copy()
                P_Fk1_i = P_Fk1_i1.copy()

        return (converged, j, error), tk1, qk1, uk1, P_gk1, P_gammak1, P_Nk1, P_Fk1

    def solve(self):
        # lists storing output variables
        q = [self.qk]
        u = [self.uk]
        P_g = [self.P_gk]
        P_gamma = [self.P_gammak]
        P_N = [self.P_Nk]
        P_F = [self.P_Fk]

        pbar = tqdm(self.t[:-1])
        for _ in pbar:
            (
                (converged, j, error),
                tk1,
                qk1,
                uk1,
                P_gk1,
                P_gammak1,
                P_Nk1,
                P_Fk1,
            ) = self.step()
            pbar.set_description(
                f"t: {tk1:0.2e}; fixed-point iterations: {j+1}; error: {error:.3e}"
            )
            if not converged:
                raise RuntimeError(
                    f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                )

            qk1, uk1 = self.step_callback(tk1, qk1, uk1)

            q.append(qk1)
            u.append(uk1)
            P_g.append(P_gk1)
            P_gamma.append(P_gammak1)
            P_N.append(P_Nk1)
            P_F.append(P_Fk1)

            # update local variables for accepted time step
            (
                self.tk,
                self.qk,
                self.uk,
                self.P_gk,
                self.P_gammak,
                self.P_Nk,
                self.P_Fk,
            ) = (tk1, qk1, uk1, P_gk1, P_gammak1, P_Nk1, P_Fk1)

        return Solution(
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
        )
