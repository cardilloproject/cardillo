import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, bmat
from scipy.sparse.linalg import splu
from tqdm import tqdm

from cardillo.solver import Solution, compute_I_F
from cardillo.math import prox_R0_np, prox_sphere


class MoreauShifted:
    def __init__(
        self,
        system,
        t1,
        dt,
        fix_point_tol=1e-8,
        fix_point_max_iter=100,
        error_function=lambda x: np.max(np.abs(x)),
    ):
        self.system = system

        # integration time
        t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.fix_point_error_function = error_function
        self.fix_point_tol = fix_point_tol
        self.fix_point_max_iter = fix_point_max_iter

        self.nq = self.system.nq
        self.nu = self.system.nu
        self.nla_g = self.system.nla_g
        self.nla_gamma = self.system.nla_gamma
        self.nla_N = self.system.nla_N
        self.nla_F = self.system.nla_F
        self.nR_smooth = self.nu + self.nla_g + self.nla_gamma
        self.nR = self.nR_smooth + self.nla_N + self.nla_F

        # initial conditions
        self.tk = system.t0
        self.qk = system.q0
        self.uk = system.u0
        self.q_dotk = system.q_dot0
        self.u_dotk = system.u_dot0
        la_g0 = system.la_g0
        la_gamma0 = system.la_gamma0
        la_N0 = system.la_N0
        la_F0 = system.la_F0

        # consistent initial percussion
        self.P_gk = la_g0 * dt
        self.P_gammak = la_gamma0 * dt
        self.P_Nk = la_N0 * dt
        self.P_Fk = la_F0 * dt

        self.split_x = np.array([self.nu, self.nu + self.nla_g])

        # connectivity matrix of normal force directions and friction force directions
        self.NF_connectivity = self.system.NF_connectivity

    def step(self):
        # general quantities
        dt = self.dt
        uk = self.uk
        tk1 = self.tk + dt

        # explicit position update with projection
        qk1 = self.qk + dt * self.system.q_dot(self.tk, self.qk, uk)
        qk1, uk = self.system.step_callback(tk1, qk1, uk)
        qk1, uk = self.system.pre_iteration_update(tk1, qk1, uk)

        # get quantities from model
        M = self.system.M(tk1, qk1)
        h = self.system.h(tk1, qk1, uk)
        W_g = self.system.W_g(tk1, qk1)
        W_gamma = self.system.W_gamma(tk1, qk1)
        chi_g = self.system.g_dot(tk1, qk1, np.zeros_like(uk))
        chi_gamma = self.system.gamma(tk1, qk1, np.zeros_like(uk))
        # note: we use csc_matrix for efficient column slicing later,
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html#scipy.sparse.csc_array
        W_N = self.system.W_N(tk1, qk1, scipy_matrix=csc_matrix)
        W_F = self.system.W_F(tk1, qk1, scipy_matrix=csc_matrix)

        # compute new estimates for prox parameters and get friction coefficient
        prox_r_N = self.system.prox_r_N(tk1, qk1)
        prox_r_F = self.system.prox_r_F(tk1, qk1)
        mu = self.system.mu

        # identify active contacts
        I_N = self.system.g_N(tk1, qk1) <= 0

        # identify active tangent contacts based on active normal contacts and
        # NF-connectivity lists
        if np.any(I_N):
            I_F = np.array(
                [
                    c
                    for i, I_N_i in enumerate(I_N)
                    for c in self.system.NF_connectivity[i]
                    if I_N_i
                ],
                dtype=int,
            )
        else:
            I_F = np.array([], dtype=int)

        # solve for new velocities and bilateral constraint percussions
        # M (uk1 - uk) - dt h - W_g P_g - W_gamma P_gamma - W_gN P_N - W_gT P_T = 0
        # -(W_g.T @ uk1 + chi_g) = 0
        # -(W_gamma.T @ uk1 + chi_gamma) = 0
        # fmt: off
        A = bmat([[         M, -W_g, -W_gamma], \
                  [    -W_g.T, None,     None], \
                  [-W_gamma.T, None,     None]], format="csc")
        # fmt: on

        # perform LU decomposition only once since matrix A is constant in
        # each time step saves alot work in the fixed point iteration
        lu = splu(A)

        # initial right hand side
        rhs = M @ uk + dt * h
        b = np.concatenate(
            (
                rhs,
                chi_g,
                chi_gamma,
            )
        )
        # if there is no contact, this is wrong! TODO
        # b = np.concatenate(
        #     (
        #         rhs + W_N[:, I_N] @ self.P_Nk[I_N] + W_F[:, I_F] @ self.P_Fk[I_F],
        #         chi_g,
        #         chi_gamma,
        #     )
        # )

        # solve for initial velocities and percussions of the bilateral
        # constraints for the fixed point iteration
        x = lu.solve(b)
        uk1, P_gk1, P_gammak1 = np.array_split(x, self.split_x)

        P_Nk1 = np.zeros(self.nla_N, dtype=float)
        P_Fk1 = np.zeros(self.nla_F, dtype=float)

        converged = True
        error = 0
        j = 0
        uk_fixed_point = uk.copy()
        # only enter fixed-point loop if any contact is active
        if np.any(I_N):
            converged = False
            P_Nk1_i1 = self.P_Nk.copy()
            P_Fk1_i1 = self.P_Fk.copy()
            for j in range(self.fix_point_max_iter):
                # fixed-point update normal direction
                P_Nk1_i1[I_N] = prox_R0_np(
                    P_Nk1_i1[I_N]
                    - prox_r_N[I_N] * self.system.xi_N(tk1, qk1, uk, uk1)[I_N]
                )

                # fixed-point update friction
                xi_F = self.system.xi_F(tk1, qk1, uk, uk1)
                for i_N, i_F in enumerate(self.NF_connectivity):
                    if I_N[i_N] and len(i_F):
                        P_Fk1_i1[i_F] = prox_sphere(
                            P_Fk1_i1[i_F] - prox_r_F[i_N] * xi_F[i_F],
                            mu[i_N] * P_Nk1_i1[i_N],
                        )

                # update rhs
                b = np.concatenate(
                    (
                        rhs + W_N[:, I_N] @ P_Nk1_i1[I_N] + W_F[:, I_F] @ P_Fk1_i1[I_F],
                        chi_g,
                        chi_gamma,
                    )
                )

                # solve for new velocities and Lagrange multipliers of bilateral constraints
                x = lu.solve(b)
                uk1, P_gk1, P_gammak1 = np.array_split(x, self.split_x)

                # check for convergence
                error = self.fix_point_error_function(uk1 - uk_fixed_point)
                uk_fixed_point = uk1
                converged = error < self.fix_point_tol
                if converged:
                    P_Nk1[I_N] = P_Nk1_i1[I_N]
                    P_Fk1[I_F] = P_Fk1_i1[I_F]
                    break

        return (converged, j, error), tk1, qk1, uk1, P_gk1, P_gammak1, P_Nk1, P_Fk1

    def solve(self):
        # lists storing output variables
        q = [self.qk]
        u = [self.uk]
        P_g = [self.P_gk]
        P_gamma = [self.P_gammak]
        P_N = [self.P_Nk]
        P_F = [self.P_Fk]

        nfrac = 100
        pbar = tqdm(self.t[1:], leave=True, mininterval=0.5, miniters=nfrac)
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
                # raise RuntimeError(
                #     f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                # )
                print(
                    f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                )

            qk1, uk1 = self.system.step_callback(tk1, qk1, uk1)

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
            la_g=np.array(P_g) / self.dt,
            la_gamma=np.array(P_gamma) / self.dt,
            la_N=np.array(P_N) / self.dt,
            la_F=np.array(P_F) / self.dt,
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
        )


# TODO: Use this implementation or remove it.
class MoreauShiftedNew:
    def __init__(
        self,
        system,
        t1,
        dt,
        atol=1e-8,
        max_iter=100,
        error_function=lambda x: np.max(np.abs(x)),
        continue_with_unconverged=True,
    ):
        self.system = system

        # integration time
        t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.fix_point_error_function = error_function
        self.atol = atol
        self.max_iter = max_iter
        self.continue_with_unconverged = continue_with_unconverged

        self.nq = self.system.nq
        self.nu = self.system.nu
        self.nla_g = self.system.nla_g
        self.nla_gamma = self.system.nla_gamma
        self.nla_N = self.system.nla_N
        self.nla_F = self.system.nla_F
        self.nR_smooth = self.nu + self.nla_g + self.nla_gamma
        self.nR = self.nR_smooth + self.nla_N + self.nla_F

        # connectivity matrix of normal force directions and friction force directions
        self.NF_connectivity = np.array(self.system.NF_connectivity)

        # initial conditions
        self.tn = system.t0
        self.qn = system.q0
        self.un = system.u0
        self.q_dotn = system.q_dot0
        self.u_dotn = system.u_dot0
        la_g0 = system.la_g0
        la_gamma0 = system.la_gamma0
        la_N0 = system.la_N0
        la_F0 = system.la_F0

        # consistent initial percussion
        self.P_gn = la_g0 * dt
        self.P_gamman = la_gamma0 * dt
        self.P_Nn = la_N0 * dt
        self.P_Fn = la_F0 * dt

        self.split_x = np.array(
            [
                self.nu,
                self.nu + self.nla_g,
            ],
            dtype=int,
        )
        self.x = np.concatenate(
            (
                self.un,
                self.P_gn,
                self.P_gamman,
            )
        )
        self.x0 = self.x.copy()

    def p(self, z, lu_A, b, W_N, W_F, I_N, I_F, tn1, qn1, un, prox_r_N, prox_r_F, mu):
        P_N = z[: self.nla_N]
        P_F = z[self.nla_N :]

        un1, _, _ = np.array_split(self.x, self.split_x)

        # fixed-point update normal direction
        P_N[I_N] = prox_R0_np(
            P_N[I_N] - prox_r_N[I_N] * self.system.xi_N(tn1, qn1, un, un1)[I_N]
        )

        # fixed-point update friction (Gauss-Seidel)
        xi_F = self.system.xi_F(tn1, qn1, un, un1)
        P_F[I_F] = np.array(
            list(
                map(
                    prox_sphere,
                    P_F[I_F] - prox_r_F[I_F] * xi_F[I_F],
                    mu[I_F][:, 0] * P_N[I_N],
                )
            )
        )
        # for i_N, i_F in enumerate(self.NF_connectivity):
        #     if I_N[i_N] and len(i_F):
        #         P_F[i_F] = prox_sphere(
        #             P_F[i_F] - prox_r_F[i_N] * xi_F[i_F],
        #             mu[i_N] * P_N[i_N],
        #         )
        I_F = I_F.flatten()
        # update rhs
        bb = b.copy()
        bb[: self.nu] += W_N[:, I_N] @ P_N[I_N] + W_F[:, I_F] @ P_F[I_F]

        # solve for new velocities and Lagrange multipliers of bilateral constraints
        self.x0 = self.x.copy()
        self.x = lu_A.solve(bb)

        return np.concatenate([P_N, P_F])

    def step(self):
        # general quantities
        dt = self.dt
        un = self.un
        tn1 = self.tn + dt

        # explicit position update with projection
        qn1 = self.qn + dt * self.system.q_dot(self.tn, self.qn, un)
        qn1, un = self.system.step_callback(tn1, qn1, un)
        qn1, un = self.system.pre_iteration_update(tn1, qn1, un)

        # get quantities from model
        M = self.system.M(tn1, qn1)
        h = self.system.h(tn1, qn1, un)
        W_g = self.system.W_g(tn1, qn1)
        W_gamma = self.system.W_gamma(tn1, qn1)
        chi_g = self.system.g_dot(tn1, qn1, np.zeros_like(un))
        chi_gamma = self.system.gamma(tn1, qn1, np.zeros_like(un))
        # note: we use csc_matrix for efficient column slicing later,
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html#scipy.sparse.csc_array
        W_N = self.system.W_N(tn1, qn1, scipy_matrix=csc_matrix)
        W_F = self.system.W_F(tn1, qn1, scipy_matrix=csc_matrix)

        # identify active contacts
        I_N = self.system.g_N(tn1, qn1) <= 0

        # identify active tangent contacts based on active normal contacts and
        # NF-connectivity lists
        I_F = self.NF_connectivity[I_N]
        # I_F = compute_I_F(I_N, self.system.NF_connectivity)

        # compute new estimates for prox parameters and get friction coefficient
        prox_r_N = self.system.prox_r_N(tn1, qn1)
        prox_r_F = self.system.prox_r_F(tn1, qn1)
        mu = self.system.mu

        # Build matrix A for computation of new velocities and bilateral constraint percussions
        # M (uk1 - uk) - dt h - W_g P_g - W_gamma P_gamma - W_gN P_N - W_gT P_T = 0
        # -(W_g.T @ uk1 + chi_g) = 0
        # -(W_gamma.T @ uk1 + chi_gamma) = 0
        # fmt: off
        A = bmat([[         M, -W_g, -W_gamma], \
                  [    -W_g.T, None,     None], \
                  [-W_gamma.T, None,     None]], format="csc")
        # fmt: on

        # perform LU decomposition only once since matrix A is constant in
        # each time step saves alot work in the fixed point iteration
        lu_A = splu(A)

        # initial right hand side without contact forces
        b = np.concatenate(
            (
                M @ un + dt * h,
                chi_g,
                chi_gamma,
            )
        )

        # solve for initial velocities and percussions of the bilateral
        # constraints for the fixed point iteration
        self.x = lu_A.solve(b)

        P_Nn1 = np.zeros(self.nla_N, dtype=float)
        P_Fn1 = np.zeros(self.nla_F, dtype=float)

        converged = True
        error = 0
        j = 0
        # only enter fixed-point loop if any contact is active
        if np.any(I_N):
            z0 = z = np.concatenate([self.P_Nn, self.P_Fn])
            for j in range(self.max_iter):
                z = self.p(
                    z0,
                    lu_A,
                    b,
                    W_N,
                    W_F,
                    I_N,
                    I_F,
                    tn1,
                    qn1,
                    un,
                    prox_r_N,
                    prox_r_F,
                    mu,
                )

                # check for convergence of percussions
                # error = self.fix_point_error_function(z - z0)

                # check for convergence of velocities
                error = self.fix_point_error_function(
                    self.x[: self.nu] - self.x0[: self.nu]
                )

                converged = error < self.atol
                if converged:
                    P_Nn1[I_N] = z[: self.nla_N][I_N]
                    P_Fn1[I_F] = z[self.nla_N :][I_F]
                    break
                z0 = z

        un1, P_gn1, P_gamman1 = np.array_split(self.x, self.split_x)

        return (converged, j, error), tn1, qn1, un1, P_gn1, P_gamman1, P_Nn1, P_Fn1

    def solve(self):
        # lists storing output variables
        q = [self.qn]
        u = [self.un]
        P_g = [self.P_gn]
        P_gamma = [self.P_gamman]
        P_N = [self.P_Nn]
        P_F = [self.P_Fn]

        nfrac = 100
        pbar = tqdm(self.t[1:], leave=True, mininterval=0.5, miniters=nfrac)
        for _ in pbar:
            (
                (converged, j, error),
                tn1,
                qn1,
                un1,
                P_gn1,
                P_gamman1,
                P_Nn1,
                P_Fn1,
            ) = self.step()
            pbar.set_description(
                f"t: {tn1:0.2e}; fixed-point iterations: {j+1}; error: {error:.3e}"
            )
            if not converged:
                if self.continue_with_unconverged:
                    print(
                        f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                    )
                else:
                    raise RuntimeError(
                        f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                    )

            qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            q.append(qn1)
            u.append(un1)
            P_g.append(P_gn1)
            P_gamma.append(P_gamman1)
            P_N.append(P_Nn1)
            P_F.append(P_Fn1)

            # update local variables for accepted time step
            (
                self.tn,
                self.qn,
                self.un,
                self.P_gn,
                self.P_gamman,
                self.P_Nn,
                self.P_Fn,
            ) = (tn1, qn1, un1, P_gn1, P_gamman1, P_Nn1, P_Fn1)

        return Solution(
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
            la_g=np.array(P_g) / self.dt,
            la_gamma=np.array(P_gamma) / self.dt,
            la_N=np.array(P_N) / self.dt,
            la_F=np.array(P_F) / self.dt,
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
        )


class MoreauClassical:
    def __init__(
        self,
        system,
        t1,
        dt,
        atol=1e-8,
        max_iter=100,
        error_function=lambda x: np.max(np.abs(x)),
        continue_with_unconverged=True,
    ):
        self.system = system

        # integration time
        t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.fix_point_error_function = error_function
        self.atol = atol
        self.max_iter = max_iter
        self.continue_with_unconverged = continue_with_unconverged

        self.nq = self.system.nq
        self.nu = self.system.nu
        self.nla_g = self.system.nla_g
        self.nla_gamma = self.system.nla_gamma
        self.nla_N = self.system.nla_N
        self.nla_F = self.system.nla_F
        self.nR_smooth = self.nu + self.nla_g + self.nla_gamma
        self.nR = self.nR_smooth + self.nla_N + self.nla_F

        # connectivity matrix of normal force directions and friction force directions
        self.NF_connectivity = np.array(self.system.NF_connectivity)

        # initial conditions
        self.tn = system.t0
        self.qn = system.q0
        self.un = system.u0
        self.q_dotn = system.q_dot0
        self.u_dotn = system.u_dot0
        la_g0 = system.la_g0
        la_gamma0 = system.la_gamma0
        la_N0 = system.la_N0
        la_F0 = system.la_F0

        # consistent initial percussion
        self.P_gn = la_g0 * dt
        self.P_gamman = la_gamma0 * dt
        self.P_Nn = la_N0 * dt
        self.P_Fn = la_F0 * dt

        self.split_x = np.array(
            [
                self.nu,
                self.nu + self.nla_g,
            ],
            dtype=int,
        )
        self.x = np.concatenate(
            (
                self.un,
                self.P_gn,
                self.P_gamman,
            )
        )
        self.x0 = self.x.copy()

    def p(self, z, lu_A, b, W_N, W_F, I_N, I_F, tn1, qn1, un, prox_r_N, prox_r_F, mu):
        P_N = z[: self.nla_N]
        P_F = z[self.nla_N :]

        un1, _, _ = np.array_split(self.x, self.split_x)

        # fixed-point update normal direction
        P_N[I_N] = prox_R0_np(
            P_N[I_N] - prox_r_N[I_N] * self.system.xi_N(tn1, qn1, un, un1)[I_N]
        )

        # fixed-point update friction (Gauss-Seidel)
        xi_F = self.system.xi_F(tn1, qn1, un, un1)
        P_F[I_F] = np.array(
            list(
                map(
                    prox_sphere,
                    P_F[I_F] - prox_r_F[I_F] * xi_F[I_F],
                    mu[I_F][:, 0] * P_N[I_N],
                )
            )
        )
        # for i_N, i_F in enumerate(self.NF_connectivity):
        #     if I_N[i_N] and len(i_F):
        #         P_F[i_F] = prox_sphere(
        #             P_F[i_F] - prox_r_F[i_N] * xi_F[i_F],
        #             mu[i_N] * P_N[i_N],
        #         )
        I_F = I_F.flatten()

        # update rhs
        bb = b.copy()
        bb[: self.nu] += W_N[:, I_N] @ P_N[I_N] + W_F[:, I_F] @ P_F[I_F]

        # solve for new velocities and Lagrange multipliers of bilateral constraints
        self.x0 = self.x.copy()
        self.x = lu_A.solve(bb)

        return np.concatenate([P_N, P_F])

    def step(self):
        # general quantities
        dt = self.dt
        un = self.un
        tn1 = self.tn + dt
        tn12 = self.tn + 0.5 * dt

        # explicit position update (midpoint) with projection
        qn12 = self.qn + 0.5 * dt * self.system.q_dot(self.tn, self.qn, un)
        qn1, un = self.system.step_callback(tn1, qn1, un)
        qn1, un = self.system.pre_iteration_update(tn1, qn1, un)

        # get quantities from model
        M = self.system.M(tn12, qn12)
        h = self.system.h(tn12, qn12, un)
        W_g = self.system.W_g(tn12, qn12)
        W_gamma = self.system.W_gamma(tn12, qn12)
        chi_g = self.system.g_dot(tn12, qn12, np.zeros_like(un))
        chi_gamma = self.system.gamma(tn12, qn12, np.zeros_like(un))
        # note: we use csc_matrix for efficient column slicing later,
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html#scipy.sparse.csc_array
        W_N = self.system.W_N(tn12, qn12, scipy_matrix=csc_matrix)
        W_F = self.system.W_F(tn12, qn12, scipy_matrix=csc_matrix)

        # identify active contacts
        I_N = self.system.g_N(tn12, qn12) <= 0

        # identify active tangent contacts based on active normal contacts and
        # NF-connectivity lists
        # I_F = compute_I_F(I_N, self.system.NF_connectivity)
        I_F = self.NF_connectivity[I_N]

        # compute new estimates for prox parameters and get friction coefficient
        prox_r_N = self.system.prox_r_N(tn12, qn12)
        prox_r_F = self.system.prox_r_F(tn12, qn12)
        mu = self.system.mu

        # Build matrix A for computation of new velocities and bilateral constraint percussions
        # M (uk1 - uk) - dt h - W_g P_g - W_gamma P_gamma - W_gN P_N - W_gT P_T = 0
        # -(W_g.T @ uk1 + chi_g) = 0
        # -(W_gamma.T @ uk1 + chi_gamma) = 0
        # fmt: off
        A = bmat([[         M, -W_g, -W_gamma], \
                  [    -W_g.T, None,     None], \
                  [-W_gamma.T, None,     None]], format="csc")
        # fmt: on

        # perform LU decomposition only once since matrix A is constant in
        # each time step saves alot work in the fixed point iteration
        lu_A = splu(A)

        # initial right hand side without contact forces
        b = np.concatenate(
            (
                M @ un + dt * h,
                chi_g,
                chi_gamma,
            )
        )

        # solve for initial velocities and percussions of the bilateral
        # constraints for the fixed point iteration
        self.x = lu_A.solve(b)

        P_Nn1 = np.zeros(self.nla_N, dtype=float)
        P_Fn1 = np.zeros(self.nla_F, dtype=float)

        converged = True
        error = 0
        j = 0
        # only enter fixed-point loop if any contact is active
        if np.any(I_N):
            z0 = z = np.concatenate([self.P_Nn, self.P_Fn])
            for j in range(self.max_iter):
                z = self.p(
                    z0,
                    lu_A,
                    b,
                    W_N,
                    W_F,
                    I_N,
                    I_F,
                    tn1,
                    qn12,
                    un,
                    prox_r_N,
                    prox_r_F,
                    mu,
                )

                # check for convergence of percussions
                # error = self.fix_point_error_function(z - z0)

                # check for convergence of velocities
                error = self.fix_point_error_function(
                    self.x[: self.nu] - self.x0[: self.nu]
                )

                converged = error < self.atol
                if converged:
                    P_Nn1[I_N] = z[: self.nla_N][I_N]
                    P_Fn1[I_F] = z[self.nla_N :][I_F]
                    break
                z0 = z

        un1, P_gn1, P_gamman1 = np.array_split(self.x, self.split_x)

        # second half step
        qn1 = qn12 + 0.5 * dt * self.system.q_dot(tn12, qn12, un1)

        return (converged, j, error), tn1, qn1, un1, P_gn1, P_gamman1, P_Nn1, P_Fn1

    def solve(self):
        # lists storing output variables
        q = [self.qn]
        u = [self.un]
        P_g = [self.P_gn]
        P_gamma = [self.P_gamman]
        P_N = [self.P_Nn]
        P_F = [self.P_Fn]

        nfrac = 100
        pbar = tqdm(self.t[1:], leave=True, mininterval=0.5, miniters=nfrac)
        for _ in pbar:
            (
                (converged, j, error),
                tn1,
                qn1,
                un1,
                P_gn1,
                P_gamman1,
                P_Nn1,
                P_Fn1,
            ) = self.step()
            pbar.set_description(
                f"t: {tn1:0.2e}; fixed-point iterations: {j+1}; error: {error:.3e}"
            )
            if not converged:
                if self.continue_with_unconverged:
                    print(
                        f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                    )
                else:
                    raise RuntimeError(
                        f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                    )

            qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            q.append(qn1)
            u.append(un1)
            P_g.append(P_gn1)
            P_gamma.append(P_gamman1)
            P_N.append(P_Nn1)
            P_F.append(P_Fn1)

            # update local variables for accepted time step
            (
                self.tn,
                self.qn,
                self.un,
                self.P_gn,
                self.P_gamman,
                self.P_Nn,
                self.P_Fn,
            ) = (tn1, qn1, un1, P_gn1, P_gamman1, P_Nn1, P_Fn1)

        return Solution(
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
            la_g=np.array(P_g) / self.dt,
            la_gamma=np.array(P_gamma) / self.dt,
            la_N=np.array(P_N) / self.dt,
            la_F=np.array(P_F) / self.dt,
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
        )
