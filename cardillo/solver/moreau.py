import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, bmat, eye, diags
from scipy.sparse.linalg import splu, spsolve, lsqr
from tqdm import tqdm

from cardillo.solver import Solution
from cardillo.math import prox_R0_np, prox_sphere, approx_fprime, norm, fsolve

# see https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/Dot.h#L126
EPS_GAMMA_F = 0
# EPS_GAMMA_F = 1.0e-10


class Moreau:
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

        self.tk = t0
        self.qk = system.q0
        self.uk = system.u0
        la_N0 = system.la_N0
        la_F0 = system.la_F0
        self.P_Nk = la_N0 * dt
        self.P_Fk = la_F0 * dt

        # connectivity matrix of normal force directions and friction force directions
        self.NF_connectivity = self.system.NF_connectivity

        # initial velocites
        self.q_dotk = self.system.q_dot(self.tk, self.qk, self.uk)

        # solve for consistent initial accelerations and Lagrange mutlipliers
        M0 = self.system.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        h0 = self.system.h(self.tk, self.qk, self.uk)
        W_g0 = self.system.W_g(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_gamma0 = self.system.W_gamma(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_N0 = self.system.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_F0 = self.system.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)

        zeta_g0 = self.system.zeta_g(t0, self.qk, self.uk)
        zeta_gamma0 = self.system.zeta_gamma(t0, self.qk, self.uk)
        # fmt: off
        A = bmat(
            [[       M0, -W_g0, -W_gamma0],
             [    W_g0.T, None,      None],
             [W_gamma0.T, None,      None]],
            format="csc",
        )
        # fmt: on
        b = np.concatenate([h0 + W_N0 @ la_N0 + W_F0 @ la_F0, -zeta_g0, -zeta_gamma0])
        u_dot_la_g_la_gamma = splu(A).solve(b)
        self.u_dotk = u_dot_la_g_la_gamma[: self.nu]
        self.P_gk = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g] * dt
        self.P_gammak = u_dot_la_g_la_gamma[self.nu + self.nla_g :] * dt

        # check if initial conditions satisfy constraints on position, velocity
        # and acceleration level
        g0 = system.g(self.tk, self.qk)
        g_dot0 = system.g_dot(self.tk, self.qk, self.uk)
        g_ddot0 = system.g_ddot(self.tk, self.qk, self.uk, self.u_dotk)
        gamma0 = system.gamma(self.tk, self.qk, self.uk)
        gamma_dot0 = system.gamma_dot(self.tk, self.qk, self.uk, self.u_dotk)

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

    def step(self):
        # general quantities
        dt = self.dt
        uk = self.uk
        tk1 = self.tk + dt

        # explicit position update
        qk1 = self.qk + dt * self.system.q_dot(self.tk, self.qk, uk)

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

        # # TODO: Discuss this tests for damping with stiffness matrix.
        # beta = 5.0e-6
        # K = -beta * self.system.h_q(tk1, qk1, uk) @ self.system.B(tk1, qk1)
        # A = bmat([[         M + K, -W_g, -W_gamma], \
        #           [    -W_g.T, None,     None], \
        #           [-W_gamma.T, None,     None]], format="csc")

        # perform LU decomposition only once since matrix A is constant in
        # each time step saves alot work in the fixed point iteration
        lu = splu(A)

        # initial right hand side
        rhs = M @ uk + dt * h
        b = np.concatenate(
            (
                rhs + W_N[:, I_N] @ self.P_Nk[I_N] + W_F[:, I_F] @ self.P_Fk[I_F],
                chi_g,
                chi_gamma,
            )
        )

        # solve for initial velocities and percussions of the bilateral
        # constraints for the fixed point iteration
        x = lu.solve(b)
        uk1 = x[: self.nu]
        P_gk1 = x[self.nu : self.nu + self.nla_g]
        P_gammak1 = x[self.nu + self.nla_g :]

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
                uk1 = x[: self.nu]
                P_gk1 = x[self.nu : self.nu + self.nla_g]
                P_gammak1 = x[self.nu + self.nla_g :]

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


# TODO:
# - Is it necessary we have to do a safe normalization?
# - alternative solution strategy (fixed-point iterations)
#   / is the lsqr approach working?
# - variable step-size with Richardson iteration, see Acary presentation/ Hairer
class NonsmoothBackwardEulerDecoupled:
    def __init__(
        self,
        model,
        t1,
        dt,
        tol=1e-6,
        max_iter=10,
        error_function=lambda x: np.max(np.abs(x)),
        solve_monolythic=True,
        with_elastic_impacts=True,
        # with_elastic_impacts=False,
    ):
        self.model = model
        self.solve_monolythic = solve_monolythic
        self.with_elastic_impacts = with_elastic_impacts

        #######################################################################
        # integration time
        #######################################################################
        self.t0 = t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        #######################################################################
        # newton settings
        #######################################################################
        self.tol = tol
        self.max_iter = max_iter
        self.error_function = error_function

        #######################################################################
        # dimensions
        #######################################################################
        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = model.nla_g
        self.nla_gamma = model.nla_gamma
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F
        self.nx = (
            self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_N + self.nla_F
        )
        self.ny = self.nu + self.nla_g + self.nla_gamma + self.nla_N + self.nla_F

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = t0 = model.t0
        self.qk = q0 = model.q0
        self.uk = u0 = model.u0
        self.uk_free = model.u0
        self.Uk = np.zeros(self.nu)
        self.La_gk = np.zeros(self.nla_g)
        self.La_gammak = np.zeros(self.nla_gamma)
        self.La_Nk = np.zeros(self.nla_N)
        self.la_Nk = model.la_N0
        self.La_Fk = np.zeros(self.nla_F)
        self.la_Fk = model.la_F0

        # initial velocites
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        # solve for consistent initial accelerations and Lagrange mutlipliers
        M0 = self.model.M(t0, q0, scipy_matrix=csr_matrix)
        h0 = self.model.h(t0, q0, u0)
        W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_F0 = self.model.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_g0 = self.model.W_g(t0, q0, scipy_matrix=csr_matrix)
        W_gamma0 = self.model.W_gamma(t0, q0, scipy_matrix=csr_matrix)
        zeta_g0 = self.model.zeta_g(t0, q0, u0)
        zeta_gamma0 = self.model.zeta_gamma(t0, q0, u0)
        # fmt: off
        A = bmat(
            [
                [        M0, -W_g0, -W_gamma0],
                [    W_g0.T,  None,      None],
                [W_gamma0.T,  None,      None],
            ],
            format="csc",
        )
        b = np.concatenate([
                h0 + W_N0 @ model.la_N0 + W_F0 @ model.la_F0, 
                -zeta_g0, 
                -zeta_gamma0
        ])
        # fmt: on

        u_dot_la_g_la_gamma = spsolve(A, b)
        self.u_dotk = u_dot_la_g_la_gamma[: self.nu]
        self.la_gk = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
        self.la_gammak = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

        #######################################################################
        # starting values for generalized state vector, its derivatives and
        # auxiliary velocities
        #######################################################################
        self.xk = np.concatenate(
            (
                self.q_dotk,
                self.u_dotk,
                self.la_gk,
                self.la_gammak,
                self.la_Nk,
                self.la_Fk,
            )
        )
        self.yk = np.concatenate(
            (self.Uk, self.La_gk, self.La_gammak, self.La_Nk, self.La_Fk)
        )
        self.sk = np.concatenate(
            (
                self.q_dotk,
                self.u_dotk,
                self.Uk,
                self.la_Nk,
                self.La_Nk,
                self.la_Fk,
                self.La_Fk,
            )
        )

        # #########################################
        # # solve for consistent initial conditions
        # #########################################
        # from scipy.optimize import fsolve

        # res = fsolve(self.Rx, self.xk.copy(), full_output=1)
        # self.xk = res[0].copy()

        # (
        #     self.q_dotk,
        #     self.u_dotk,
        #     self.la_gk,
        #     self.la_gammak,
        #     self.la_Nk,
        #     self.la_Fk,
        # ) = self.unpack_x(self.xk)
        # print(f"la_g0: {self.la_gk}")

        # initialize index sets
        self.I_Nk1 = np.zeros(self.nla_N, dtype=bool)
        self.B_Nk1 = np.zeros(self.nla_N, dtype=bool)

        # boolean that detects the first time step
        self.fist_step = True

    def unpack_x(self, xk1):
        q_dotk1 = xk1[: self.nq]
        u_dotk1 = xk1[self.nq : self.nq + self.nu]
        la_gk1 = xk1[self.nq + self.nu : self.nq + self.nu + self.nla_g]
        la_gammak1 = xk1[
            self.nq
            + self.nu
            + self.nla_g : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
        ]
        la_Nk1 = xk1[
            self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
            + self.nla_N
        ]
        la_Fk1 = xk1[self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_N :]

        return q_dotk1, u_dotk1, la_gk1, la_gammak1, la_Nk1, la_Fk1

    def unpack_y(self, yk1):
        Uk1 = yk1[: self.nu]
        La_gk1 = yk1[self.nu : self.nu + self.nla_g]
        La_gammak1 = yk1[self.nu + self.nla_g : self.nu + self.nla_g + self.nla_gamma]
        La_Nk1 = yk1[
            self.nu
            + self.nla_g
            + self.nla_gamma : self.nu
            + self.nla_g
            + self.nla_gamma
            + self.nla_N
        ]
        La_Fk1 = yk1[self.nu + self.nla_g + self.nla_gamma + self.nla_N :]

        return Uk1, La_gk1, La_gammak1, La_Nk1, La_Fk1

    def update_x(self, xk1):
        q_dotk1 = xk1[: self.nq]
        u_dotk1 = xk1[self.nq : self.nq + self.nu]

        # backward Euler
        tk1 = self.tk + self.dt
        uk1_free = self.uk + self.dt * u_dotk1
        qk1 = self.qk + self.dt * q_dotk1

        # tk1 = self.tk + self.dt
        # if self.fist_step:
        #     # if self.fist_step or np.any(self.I_Nk1):
        #     # BDF1 (backward Euler)
        #     uk1_free = self.uk + self.dt * u_dotk1
        #     qk1 = self.qk + self.dt * q_dotk1
        # else:
        #     # BDF2
        #     uk1_free = (4.0 * self.uk_free - self.uk_1_free + 2.0 * self.dt * u_dotk1) / 3.0
        #     # uk1_free = (4.0 * self.uk - self.uk_1 + 2.0 * self.dt * u_dotk1) / 3.0 #- self.Uk * 4 / 3 - self.Uk_1 / 3
        #     qk1 = (4.0 * self.qk - self.qk_1 + 2.0 * self.dt * q_dotk1) / 3.0

        #     # # half step forward Euler
        #     # # self.uk_free = self.uk_1_free + self.dt * self.u_dotk
        #     # # self.qk = self.qk_1 + self.dt * self.q_dotk
        #     # # self.uk_free = self.uk_1_free + self.dt * u_dotk1
        #     # # self.uk_free = self.uk_1 + self.dt * u_dotk1
        #     # # self.qk = self.qk_1 + self.dt * q_dotk1
        #     # self.uk_free = self.uk_1 + self.dt * u_dotk1
        #     # self.qk = self.qk_1 + self.dt * q_dotk1

        #     # # # half explicit trapezoidal rule
        #     # # # self.uk_free = self.uk_1_free + 0.5 * self.dt * (self.u_dotk + u_dotk1)
        #     # # # self.qk = self.qk_1 + 0.5 * self.dt * (self.q_dotk + q_dotk1)
        #     # # self.uk_free = self.uk_1 + 0.5 * self.dt * (self.u_dotk + u_dotk1)
        #     # # self.qk = self.qk_1 + 0.5 * self.dt * (self.q_dotk + q_dotk1)
        #     # # # uk1_free = self.uk_free
        #     # # # qk1 = self.qk

        #     # # haf step BDF2
        #     # # uk1_free = (4.0 * self.uk_free - self.uk_1_free + 2.0 * self.dt * u_dotk1) / 3.0
        #     # uk1_free = (4.0 * self.uk_free - self.uk_1 + 2.0 * self.dt * u_dotk1) / 3.0
        #     # qk1 = (4.0 * self.qk - self.qk_1 + 2.0 * self.dt * q_dotk1) / 3.0

        # ##############
        # # Newmark beta, see https://de.wikipedia.org/wiki/Newmark-beta-Verfahren
        # ##############
        # gamma = 0.5
        # beta = 1/6
        # tk1 = self.tk + self.dt
        # uk1_free = self.uk + gamma * self.dt * (self.u_dotk + u_dotk1)
        # # qk1 = self.qk + self.dt * q_dotk1
        # # qk1 = self.qk + self.dt * self.u_dotk + self.dt**2 * ((0.5 - beta) * self.u_dotk + beta * u_dotk1)
        # # qk1 = self.qk + self.dt * (self.uk - self.Uk) + self.dt**2 * ((0.5 - beta) * self.u_dotk + beta * u_dotk1)
        # qk1 = self.qk + self.dt * self.uk + self.dt**2 * ((0.5 - beta) * self.u_dotk + beta * u_dotk1)

        # ################
        # # trapezoid rule,
        # # TODO: Works only with unilateral constraints on acceleration level!
        # ################
        # tk1 = self.tk + self.dt
        # uk1_free = self.uk + 0.5 * self.dt * (self.u_dotk + u_dotk1) #- self.Uk
        # qk1 = self.qk + 0.5 * self.dt * (self.q_dotk + q_dotk1)

        return (
            tk1,
            qk1,
            uk1_free,
        )

    def Rx(self, xk1, update_index=False):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        mu = self.model.mu

        q_dotk1, u_dotk1, la_gk1, la_gammak1, la_Nk1, la_Fk1 = self.unpack_x(xk1)
        tk1, qk1, uk1_free = self.update_x(xk1)

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)
        hk1 = self.model.h(tk1, qk1, uk1_free)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        gk1 = self.model.g(tk1, qk1)
        gammak1 = self.model.gamma(tk1, qk1, uk1_free)
        g_Nk1 = self.model.g_N(tk1, qk1)
        # g_N_dotk1 = self.model.g_N_dot(tk1, qk1, uk1_free)
        # g_N_ddotk1 = self.model.g_N_ddot(tk1, qk1, uk1_free, u_dotk1)
        gamma_Fk1 = self.model.gamma_F(tk1, qk1, uk1_free)
        # xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1_free)
        # xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1_free)

        # if not self.fist_step:
        #     la_Nk1 = 0.5 * (la_Nk1 + self.la_Nk)

        ###################
        # evaluate residual
        ###################
        Rx = np.zeros(self.nx, dtype=xk1.dtype)

        ####################
        # kinematic equation
        ####################
        Rx[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1_free)

        ####################
        # euations of motion
        ####################
        Rx[nq : nq + nu] = (
            Mk1 @ u_dotk1
            - hk1
            - W_gk1 @ la_gk1
            - W_gammak1 @ la_gammak1
            - W_Nk1 @ la_Nk1
            - W_Fk1 @ la_Fk1
        )

        #######################
        # bilateral constraints
        #######################
        Rx[nq + nu : nq + nu + nla_g] = gk1
        Rx[nq + nu + nla_g : nq + nu + nla_g + nla_gamma] = gammak1

        ################
        # normal contact
        ################
        prox_r_N = self.model.prox_r_N(tk1, qk1)
        prox_arg = g_Nk1 - prox_r_N * la_Nk1
        if update_index:
            # self.I_Nk1 = g_Nk1 <= 0.0
            self.I_Nk1 = prox_arg <= 0.0

            # self.B_Nk1 = self.I_Nk1 * (g_N_dotk1 <= 0)

        # Rx[
        #     nq + nu + nla_g + nla_gamma : nq + nu + nla_g + nla_gamma + nla_N
        # ] = g_Nk1 - prox_R0_np(prox_arg)
        Rx[
            nq + nu + nla_g + nla_gamma : nq + nu + nla_g + nla_gamma + nla_N
        ] = np.where(self.I_Nk1, g_Nk1, la_Nk1)
        # Rx[
        #     nq + nu + nla_g + nla_gamma : nq + nu + nla_g + nla_gamma + nla_N
        # ] = np.minimum(g_Nk1, la_Nk1)

        # Rx[
        #     nq + nu + nla_g + nla_gamma : nq + nu + nla_g + nla_gamma + nla_N
        # ] = np.where(
        #     self.I_Nk1,
        #     # la_Nk1 + prox_R0_np(prox_r_N * g_N_dotk1 - la_Nk1),
        #     g_N_dotk1 - prox_R0_np(g_N_dotk1 - prox_r_N * la_Nk1),
        #     la_Nk1
        # )

        # Rx[
        #     nq + nu + nla_g + nla_gamma : nq + nu + nla_g + nla_gamma + nla_N
        # ] = np.where(
        #     self.B_Nk1,
        #     # la_Nk1 + prox_R0_np(prox_r_N * g_N_ddotk1 - la_Nk1),
        #     g_N_ddotk1 - prox_R0_np(g_N_ddotk1 - prox_r_N * la_Nk1),
        #     la_Nk1
        # )

        ##########
        # friction
        ##########
        # Rx[nq + nu + nla_g + nla_gamma + nla_N :] = la_Fk1

        prox_r_F = self.model.prox_r_F(tk1, qk1)
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                # Rx[nq + nu + nla_g + nla_gamma + nla_N + i_F] = (
                #     la_Fk1[i_F]
                #     + prox_sphere(
                #         prox_r_F[i_N] * gamma_Fk1[i_F] - la_Fk1[i_F],
                #         mu[i_N] * la_Nk1[i_N],
                #     )
                # )

                la_Fk1_local = la_Fk1[i_F]
                gamma_Fk1_local = gamma_Fk1[i_F]
                la_Nk1_local = la_Nk1[i_N]
                prox_arg_friction = prox_r_F[i_F] * gamma_Fk1_local - la_Fk1_local
                radius = mu[i_N] * la_Nk1_local
                norm_prox_arg_friction = norm(prox_arg_friction)

                if norm_prox_arg_friction <= radius:
                    Rx[nq + nu + nla_g + nla_gamma + nla_N + i_F] = (
                        prox_r_F[i_F] * gamma_Fk1_local
                    )
                else:
                    Rx[nq + nu + nla_g + nla_gamma + nla_N + i_F] = (
                        la_Fk1_local
                        + radius * prox_arg_friction / norm_prox_arg_friction
                    )

            # if len(i_F) > 0:
            #     la_Fk1_local = la_Fk1[i_F]
            #     if self.I_Nk1[i_N]:
            #         gamma_Fk1_local = gamma_Fk1[i_F]
            #         la_Nk1_local = la_Nk1[i_N]
            #         prox_arg_friction = prox_r_F[i_F] * gamma_Fk1_local - la_Fk1_local
            #         radius = mu[i_N] * la_Nk1_local
            #         norm_prox_arg_friction = norm(prox_arg_friction)
            #         if norm_prox_arg_friction <= radius:
            #             Rx[nq + nu + nla_g + nla_gamma + nla_N + i_F] = (
            #                 # TODO: remove prox_r_F here and change derivative accordingly
            #                 prox_r_F[i_F]
            #                 * gamma_Fk1_local
            #             )
            #         else:
            #             Rx[nq + nu + nla_g + nla_gamma + nla_N + i_F] = (
            #                 la_Fk1_local
            #                 + radius * prox_arg_friction / norm_prox_arg_friction
            #             )
            #             # gamma_Fk1_local_norm = norm(gamma_Fk1_local)
            #             # # Rx[nq + nu + nla_g + nla_gamma + nla_N + i_F] = (
            #             # #     la_Fk1_free_local
            #             # #     + radius * gamma_Fk1_local / gamma_Fk1_local_norm
            #             # # )
            #             # if gamma_Fk1_local_norm > EPS_GAMMA_F:
            #             #     Rx[nq + nu + nla_g + nla_gamma + nla_N + i_F] = (
            #             #         la_Fk1_free_local
            #             #         + radius * gamma_Fk1_local / gamma_Fk1_local_norm
            #             #     )
            #             # else:
            #             #     Rx[nq + nu + nla_g + nla_gamma + nla_N + i_F] = (
            #             #         la_Fk1_free_local + radius * gamma_Fk1_local
            #             #     )
            #     else:
            #         Rx[nq + nu + nla_g + nla_gamma + nla_N + i_F] = la_Fk1_local

        # update quantities of new time step for projection step
        self.tk1 = tk1
        self.qk1 = qk1
        self.uk1_free = uk1_free
        self.la_gk1 = la_gk1
        self.la_gammak1 = la_gammak1
        self.la_Nk1 = la_Nk1
        self.la_Fk1 = la_Fk1

        return Rx

    def Jx(self, xk1):
        # return csr_matrix(approx_fprime(xk1, self.Rx, method="2-point"))
        # return csr_matrix(approx_fprime(xk1, self.Rx, method="3-point"))
        # return csr_matrix(approx_fprime(xk1, self.Rx, method="cs", eps=1.0e-12))

        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        mu = self.model.mu

        q_dotk1, u_dotk1, la_gk1, la_gammak1, la_Nk1, la_Fk1 = self.unpack_x(xk1)
        tk1, qk1, uk1_free = self.update_x(xk1)

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        # note: csc.T gives csr for efficient row slicing
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csc_matrix)
        gamma_Fk1 = self.model.gamma_F(tk1, qk1, uk1_free)

        # chain rules for backward Euler update
        qk1_q_dotk1 = self.dt
        uk1_free_u_dotk1 = self.dt

        ####################
        # kinematic equation
        ####################
        J_q_dotk1_q_dotk1 = (
            eye(nq, nq) - self.model.q_dot_q(tk1, qk1, uk1_free) * qk1_q_dotk1
        )
        J_q_dotk1_u_dotk1 = -self.model.B(tk1, qk1) * uk1_free_u_dotk1

        ####################
        # euations of motion
        ####################
        J_u_dotk1_q_dotk1 = (
            self.model.Mu_q(tk1, qk1, uk1_free)
            - self.model.h_q(tk1, qk1, uk1_free)
            - self.model.Wla_g_q(tk1, qk1, la_gk1)
            - self.model.Wla_gamma_q(tk1, qk1, la_gammak1)
            - self.model.Wla_N_q(tk1, qk1, la_Nk1)
            - self.model.Wla_F_q(tk1, qk1, la_Fk1)
        ) * qk1_q_dotk1
        J_u_dotk1_u_dotk1 = Mk1 - self.model.h_u(tk1, qk1, uk1_free) * uk1_free_u_dotk1

        #######################
        # bilateral constraints
        #######################
        J_gk1_q_dotk1 = self.model.g_q(tk1, qk1) * qk1_q_dotk1
        J_gammak1_q_dotk1 = self.model.gamma_q(tk1, qk1, uk1_free) * qk1_q_dotk1
        J_gammak1_u_dotk1 = W_gammak1.T * uk1_free_u_dotk1

        ################
        # normal contact
        ################
        # note: csr_matrix is best for row slicing, see
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
        g_Nk1_qk1 = self.model.g_N_q(tk1, qk1, scipy_matrix=csr_matrix)

        J_la_Nk1_qk1 = lil_matrix((self.nla_N, self.nq))
        J_la_Nk1_la_Nk1 = lil_matrix((self.nla_N, self.nla_N))
        for i in range(self.nla_N):
            if self.I_Nk1[i]:
                J_la_Nk1_qk1[i] = g_Nk1_qk1[i]
            else:
                J_la_Nk1_la_Nk1[i, i] = 1.0

        J_la_Nk1_q_dotk1 = J_la_Nk1_qk1 * qk1_q_dotk1

        ##########
        # friction
        ##########
        prox_r_F = self.model.prox_r_F(tk1, qk1)
        gamma_Fk1_qk1 = self.model.gamma_F_q(
            tk1, qk1, uk1_free, scipy_matrix=csr_matrix
        )
        gamma_Fk1_uk1 = W_Fk1.T

        J_la_Fk1_qk1 = lil_matrix((self.nla_F, self.nq))
        J_la_Fk1_uk1_free = lil_matrix((self.nla_F, self.nu))
        J_la_Fk1_la_Nk1 = lil_matrix((self.nla_F, self.nla_N))
        J_la_Fk1_la_Fk1 = lil_matrix((self.nla_F, self.nla_F))
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)
            n_F = len(i_F)
            if n_F > 0:

                la_Fk1_local = la_Fk1[i_F]
                gamma_Fk1_local = gamma_Fk1[i_F]
                la_Nk1_local = la_Nk1[i_N]
                prox_arg_friction = prox_r_F[i_F] * gamma_Fk1_local - la_Fk1_local
                radius = mu[i_N] * la_Nk1_local
                norm_prox_arg_friction = norm(prox_arg_friction)

                if norm_prox_arg_friction <= radius:
                    c_F_gamma_F = diags(prox_r_F[i_F])
                else:
                    slip_dir = prox_arg_friction / norm_prox_arg_friction
                    s = radius / norm_prox_arg_friction
                    c_F_gamma_F = (
                        s
                        * diags(prox_r_F[i_F])
                        @ (np.eye(n_F, dtype=float) - np.outer(slip_dir, slip_dir))
                    )

                    J_la_Fk1_la_Nk1[i_F, i_N] = mu[i_N] * slip_dir

                    dense = (1.0 - s) * np.eye(n_F, dtype=float) + s * np.outer(
                        slip_dir, slip_dir
                    )
                    for j, j_F in enumerate(i_F):
                        for k, k_F in enumerate(i_F):
                            J_la_Fk1_la_Fk1[j_F, k_F] = dense[j, k]

                # same chain rule for different c_F_gamma_Fs
                J_la_Fk1_qk1[i_F] = c_F_gamma_F @ gamma_Fk1_qk1[i_F]
                J_la_Fk1_uk1_free[i_F] = c_F_gamma_F @ gamma_Fk1_uk1[i_F]
                # tmp_q = c_F_gamma_F @ gamma_Fk1_qk1[i_F]
                # tmp_u = c_F_gamma_F @ gamma_Fk1_uk1[i_F]
                # for k_F, j_F in enumerate(i_F):
                #     J_la_Fk1_qk1[j_F] = prox_r_F[j_F] * tmp_q[k_F]
                #     J_la_Fk1_uk1_free[j_F] = prox_r_F[j_F] * tmp_u[k_F]

                # # old implementation
                # la_Fk1_local = la_Fk1[i_F]
                # if self.I_Nk1[i_N]:
                #     gamma_Fk1_local = gamma_Fk1[i_F]
                #     la_Nk1_local = la_Nk1[i_N]
                #     prox_arg_friction = prox_r_F[i_F] * gamma_Fk1_local - la_Fk1_local
                #     radius = mu[i_N] * la_Nk1_local

                #     eyeF = np.eye(n_F, dtype=float)
                #     norm_prox_arg_friction = norm(prox_arg_friction)
                #     if norm_prox_arg_friction <= radius:
                #         c_F_gamma_F = eyeF
                #     else:
                #         slip_dir = prox_arg_friction / norm_prox_arg_friction
                #         c_F_gamma_F = (radius / norm_prox_arg_friction) * (
                #             eyeF - np.outer(slip_dir, slip_dir)
                #         )

                #         J_la_Fk1_la_Nk1[i_F, i_N] = mu[i_N] * slip_dir

                #         for j, j_F in enumerate(i_F):
                #             for k, k_F in enumerate(i_F):
                #                 if j == k:
                #                     J_la_Fk1_la_Fk1[j_F, k_F] = 1.0 - c_F_gamma_F[j, k]
                #                 else:
                #                     J_la_Fk1_la_Fk1[j_F, k_F] = c_F_gamma_F[j, k]

                #         # for j_F in i_F:
                #         #     J_la_Fk1_la_Fk1[j_F, j_F] = 1.0

                #         # gamma_Fk1_local_norm = norm(gamma_Fk1_local)

                #         # slip_dir = gamma_Fk1_local / gamma_Fk1_local_norm
                #         # c_F_gamma_F = (radius / gamma_Fk1_local_norm) * (
                #         #     eyeF - np.outer(slip_dir, slip_dir)
                #         # )

                #         # J_la_Fk1_la_Nk1[i_F, i_N] = mu[i_N] * slip_dir

                #         # if gamma_Fk1_local_norm > EPS_GAMMA_F:
                #         #     slip_dir = gamma_Fk1_local / gamma_Fk1_local_norm
                #         #     c_F_gamma_F = (radius / gamma_Fk1_local_norm) * (
                #         #         eyeF - np.outer(slip_dir, slip_dir)
                #         #     )

                #         #     J_la_Fk1_la_Nk1[i_F, i_N] = mu[i_N] * slip_dir
                #         # else:
                #         #     c_F_gamma_F = radius * eyeF

                #         #     J_la_Fk1_la_Nk1[i_F, i_N] = mu[i_N] * gamma_Fk1_local

                #     # same chain rule for different c_F_gamma_Fs
                #     # J_la_Fk1_qk1[i_F] = c_F_gamma_F @ gamma_Fk1_qk1[i_F]
                #     # J_la_Fk1_uk1_free[i_F] = c_F_gamma_F @ gamma_Fk1_uk1[i_F]
                #     tmp_q = c_F_gamma_F @ gamma_Fk1_qk1[i_F]
                #     tmp_u = c_F_gamma_F @ gamma_Fk1_uk1[i_F]
                #     for k_F, j_F in enumerate(i_F):
                #         J_la_Fk1_qk1[j_F] = prox_r_F[j_F] * tmp_q[k_F]
                #         J_la_Fk1_uk1_free[j_F] = prox_r_F[j_F] * tmp_u[k_F]
                # else:
                #     for j_F in i_F:
                #         J_la_Fk1_la_Fk1[j_F, j_F] = 1.0

        J_la_Fk1_q_dotk1 = J_la_Fk1_qk1 * qk1_q_dotk1
        J_la_Fk1_q_uotk1 = J_la_Fk1_uk1_free * uk1_free_u_dotk1

        # fmt: off
        Jx = bmat(
            [
                [J_q_dotk1_q_dotk1, J_q_dotk1_u_dotk1,   None,       None,            None,            None],
                [J_u_dotk1_q_dotk1, J_u_dotk1_u_dotk1, -W_gk1, -W_gammak1,          -W_Nk1,          -W_Fk1],
                [    J_gk1_q_dotk1,              None,   None,       None,            None,            None],
                [J_gammak1_q_dotk1, J_gammak1_u_dotk1,   None,       None,            None,            None],
                [ J_la_Nk1_q_dotk1,              None,   None,       None, J_la_Nk1_la_Nk1,            None],
                [ J_la_Fk1_q_dotk1,  J_la_Fk1_q_uotk1,   None,       None, J_la_Fk1_la_Nk1, J_la_Fk1_la_Fk1]
            ],
            format="csr",
        )
        # fmt: on

        return Jx

        Jx_num = csr_matrix(approx_fprime(xk1, self.Rx, method="2-point"))
        # Jx_num = csr_matrix(approx_fprime(xk1, self.Rx, method="3-point"))

        # Jx[nq + nu + nla_g + nla_gamma + nla_N:, :nq] = Jx_num[nq + nu + nla_g + nla_gamma + nla_N:, :nq]
        # Jx[nq + nu + nla_g + nla_gamma + nla_N:, nq : nq + nu] = Jx_num[nq + nu + nla_g + nla_gamma + nla_N:, nq : nq + nu]
        # Jx[nq + nu + nla_g + nla_gamma + nla_N:, nq + nu + nla_g + nla_gamma : nq + nu + nla_g + nla_gamma + nla_N] = Jx_num[nq + nu + nla_g + nla_gamma + nla_N:, nq + nu + nla_g + nla_gamma : nq + nu + nla_g + nla_gamma + nla_N]

        # return Jx

        nq, nu, nla_g, nla_gamma, nla_N, nla_F = (
            self.nq,
            self.nu,
            self.nla_g,
            self.nla_gamma,
            self.nla_N,
            self.nla_F,
        )
        # diff = (Jx - Jx_num).toarray()
        # diff = (Jx - Jx_num).toarray()[:nq] # TODO: Find error!
        # diff = (Jx - Jx_num).toarray()[nq : nq + nu]
        # diff = (Jx - Jx_num).toarray()[nq + nu : nq + nu + nla_g]
        # diff = (Jx - Jx_num).toarray()[nq + nu + nla_g : nq + nu + nla_g + nla_gamma]
        # diff = (Jx - Jx_num).toarray()[nq + nu + nla_g + nla_gamma: nq + nu + nla_g + nla_gamma + nla_N]
        diff = (Jx - Jx_num).toarray()[nq + nu + nla_g + nla_gamma + nla_N :]
        # diff = (Jx - Jx_num).toarray()[nq + nu + nla_g + nla_gamma + nla_N:, :nq] # TODO: find error
        # diff = (Jx - Jx_num).toarray()[nq + nu + nla_g + nla_gamma + nla_N:, nq : nq + nu] # TODO: find error
        # diff = (Jx - Jx_num).toarray()[
        #     nq + nu + nla_g + nla_gamma + nla_N :,
        #     nq + nu + nla_g + nla_gamma : nq + nu + nla_g + nla_gamma + nla_N,
        # ]  # TODO: find error
        # diff = (Jx - Jx_num).toarray()[nq + nu + nla_g + nla_gamma + nla_N:, nq + nu + nla_g + nla_gamma + nla_N:]
        error = np.linalg.norm(diff)
        if error > 1.0e-6:
            print(f"error Jx: {error}")
        return Jx_num

    def unpack_x_smooth(self, xk1):
        q_dotk1 = xk1[: self.nq]
        u_dotk1 = xk1[self.nq : self.nq + self.nu]
        la_gk1 = xk1[self.nq + self.nu : self.nq + self.nu + self.nla_g]
        la_gammak1 = xk1[
            self.nq
            + self.nu
            + self.nla_g : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
        ]

        return q_dotk1, u_dotk1, la_gk1, la_gammak1

    def update_x_smooth(self, xk1):
        q_dotk1, u_dotk1, la_gk1, la_gammak1 = self.unpack_x_smooth(xk1)
        tk1 = self.tk + self.dt
        uk1 = self.uk + self.dt * u_dotk1
        qk1 = self.qk + self.dt * q_dotk1

        return tk1, qk1, uk1

    def Rx_smooth(self, xk1, la_Nk1, la_Fk1):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        q_dotk1, u_dotk1, la_gk1, la_gammak1 = self.unpack_x_smooth(xk1)
        tk1, qk1, uk1 = self.update_x_smooth(xk1)

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)
        hk1 = self.model.h(tk1, qk1, uk1)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        gk1 = self.model.g(tk1, qk1)
        gammak1 = self.model.gamma(tk1, qk1, uk1)

        ###################
        # evaluate residual
        ###################
        Rx = np.zeros(nq + nu + nla_g + nla_gamma, dtype=xk1.dtype)

        ####################
        # kinematic equation
        ####################
        Rx[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1)

        ####################
        # euations of motion
        ####################
        Rx[nq : nq + nu] = (
            Mk1 @ u_dotk1
            - hk1
            - W_gk1 @ la_gk1
            - W_gammak1 @ la_gammak1
            - W_Nk1 @ la_Nk1
            - W_Fk1 @ la_Fk1
        )

        #######################
        # bilateral constraints
        #######################
        Rx[nq + nu : nq + nu + nla_g] = gk1
        Rx[nq + nu + nla_g : nq + nu + nla_g + nla_gamma] = gammak1

        return Rx

    def Jx_smooth(self, xk1, la_Nk1, la_Fk1):
        # return csr_matrix(approx_fprime(xk1, self.Rx, method="2-point"))
        # # return csr_matrix(approx_fprime(xk1, self.Rx, method="3-point"))
        # # return csr_matrix(approx_fprime(xk1, self.Rx, method="cs", eps=1.0e-10))
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        q_dotk1, u_dotk1, la_gk1, la_gammak1 = self.unpack_x_smooth(xk1)
        tk1, qk1, uk1 = self.update_x_smooth(xk1)

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)

        # chain rules for backward Euler update
        qk1_q_dotk1 = self.dt
        uk1_u_dotk1 = self.dt

        ####################
        # kinematic equation
        ####################
        # Rx[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1)
        J_q_dotk1_q_dotk1 = (
            eye(nq, nq) - self.model.q_dot_q(tk1, qk1, uk1) * qk1_q_dotk1
        )
        J_q_dotk1_u_dotk1 = -self.model.B(tk1, qk1) * uk1_u_dotk1

        ####################
        # euations of motion
        ####################
        # Rx[nq : nq + nu] = (
        #     Mk1 @ u_dotk1
        #     - hk1
        #     - W_gk1 @ la_gk1
        #     - W_gammak1 @ la_gammak1
        #     - W_Nk1 @ la_Nk1
        #     - W_Fk1 @ la_Fk1
        # )
        J_u_dotk1_q_dotk1 = (
            self.model.Mu_q(tk1, qk1, uk1)
            - self.model.h_q(tk1, qk1, uk1)
            - self.model.Wla_g_q(tk1, qk1, la_gk1)
            - self.model.Wla_gamma_q(tk1, qk1, la_gammak1)
            - self.model.Wla_N_q(tk1, qk1, la_Nk1)
            - self.model.Wla_F_q(tk1, qk1, la_Fk1)
        ) * qk1_q_dotk1
        J_u_dotk1_u_dotk1 = Mk1 - self.model.h_u(tk1, qk1, uk1) * uk1_free_u_dotk1

        #######################
        # bilateral constraints
        #######################
        # Rx[nq + nu : nq + nu + nla_g] = gk1
        # Rx[nq + nu + nla_g : nq + nu + nla_g + nla_gamma] = gammak1
        J_gk1_q_dotk1 = self.model.g_q(tk1, qk1) * qk1_q_dotk1
        J_gammak1_q_dotk1 = self.model.gamma_q(tk1, qk1, uk1) * qk1_q_dotk1
        J_gammak1_u_dotk1 = W_gammak1.T * uk1_u_dotk1

        # fmt: off
        Jx = bmat(
            [
                [J_q_dotk1_q_dotk1, J_q_dotk1_u_dotk1,  None,        None],
                [J_u_dotk1_q_dotk1, J_u_dotk1_u_dotk1, -W_gk1, -W_gammak1],
                [    J_gk1_q_dotk1,              None,   None,       None],
                [J_gammak1_q_dotk1, J_gammak1_u_dotk1,   None,       None],
            ],
            format="csr",
        )
        # fmt: on

        # Jx_num = csr_matrix(approx_fprime(xk1, self.Rx, method="2-point"))
        Jx_num = csr_matrix(approx_fprime(xk1, self.Rx, method="3-point"))

        diff = (Jx - Jx_num).toarray()
        # # diff = (Jx - Jx_num).toarray()[:self.nq]
        # # diff = (Jx - Jx_num).toarray()[self.nq : self.nq + self.nu, :self.nq]
        # # diff = (Jx - Jx_num).toarray()[self.nq : self.nq + self.nu]
        # # diff = (Jx - Jx_num).toarray()[self.nq + self.nu : self.nq + self.nu + self.nla_g]
        # # diff = (Jx - Jx_num).toarray()[self.nq + self.nu + self.nla_g : self.nq + self.nu + self.nla_g + self.nla_gamma]
        # # diff = (Jx - Jx_num).toarray()[self.nq + self.nu + self.nla_g + self.nla_gamma: self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_N, :self.nq]
        error = np.linalg.norm(diff)
        if error > 1.0e-6:
            print(f"error Jx: {error}")
        return Jx_num
        # return Jx

    def Ry(self, yk1, update_index=False):
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        mu = self.model.mu

        # quantities of old time step
        tk1 = self.tk1
        qk1 = self.qk1
        uk1_free = self.uk1_free

        # unpack xk1
        Uk1, La_gk1, La_gammak1, La_Nk1, La_Fk1 = self.unpack_y(yk1)

        # update velocities
        uk1 = uk1_free + Uk1

        # update percussions
        P_gk1 = self.dt * self.la_gk1 + La_gk1
        P_gammak1 = self.dt * self.la_gammak1 + La_gammak1
        P_Nk1 = self.dt * self.la_Nk1 + La_Nk1
        P_Fk1 = self.dt * self.la_Fk1 + La_Fk1

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_dot = self.model.g_dot(tk1, qk1, uk1)
        gamma = self.model.gamma(tk1, qk1, uk1)
        if self.fist_step:
            xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
            xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)
        else:
            xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk_1, uk1)
            xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk_1, uk1)

        ###################
        # evaluate residual
        ###################
        Ry = np.zeros(self.nu + self.nla_g + self.nla_gamma + self.nla_N + self.nla_F)

        #################
        # impact equation
        #################
        # print(f"det(Mk1): {np.linalg.det(Mk1.toarray())}")
        # Gk1 = W_Nk1.T @ spsolve(Mk1, W_Nk1)
        # print(f"det(Gk1): {np.linalg.det(Gk1.toarray())}")
        Ry[:nu] = (
            Mk1 @ Uk1
            - W_gk1 @ La_gk1
            - W_gammak1 @ La_gammak1
            - W_Nk1 @ La_Nk1
            - W_Fk1 @ La_Fk1
            # - W_gk1 @ P_gk1
            # - W_gammak1 @ P_gammak1
            # - W_Nk1 @ P_Nk1
            # - W_Fk1 @ P_Fk1
        )

        # impulsive bilateral constraints
        # TODO: Are they correct?
        Ry[nu : nu + nla_g] = g_dot
        Ry[nu + nla_g : nu + nla_g + nla_gamma] = gamma

        ###################
        # normal impact law
        ###################
        prox_r_N = self.model.prox_r_N(tk1, qk1)
        Ry[nu + nla_g + nla_gamma : nu + nla_g + nla_gamma + nla_N] = np.where(
            self.I_Nk1,
            xi_Nk1 - prox_R0_np(xi_Nk1 - prox_r_N * La_Nk1),
            La_Nk1,
            # xi_Nk1 - prox_R0_np(xi_Nk1 - prox_r_N * P_Nk1),
            # P_Nk1,
        )

        ####################
        # tangent impact law
        ####################
        prox_r_F = self.model.prox_r_F(tk1, qk1)
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                Ry[nu + nla_g + nla_gamma + nla_N + i_F] = np.where(
                    self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                    -La_Fk1[i_F]
                    - prox_sphere(
                        -La_Fk1[i_F] + prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * La_Nk1[i_N],
                    ),
                    La_Fk1[i_F],
                    # -P_Fk1[i_F]
                    # - prox_sphere(
                    #     -P_Fk1[i_F] + prox_r_F[i_N] * xi_Fk1[i_F],
                    #     mu[i_N] * P_Nk1[i_N],
                    # ),
                    # P_Fk1[i_F],
                )

        return Ry

    def Jy(self, yk1):
        return csr_matrix(approx_fprime(yk1, self.Ry, method="2-point"))
        # return csr_matrix(approx_fprime(yk1, self.Ry, method="3-point"))
        # return csr_matrix(approx_fprime(yk1, self.Ry, method="cs"))

    def unpack_s(self, sk1):
        nq = self.nq
        nu = self.nu
        nla_N = self.nla_N
        nla_F = self.nla_F

        q_dotk1 = sk1[:nq]
        u_dotk1 = sk1[nq : nq + nu]
        Uk1 = sk1[nq + nu : nq + 2 * nu]
        la_Nk1 = sk1[nq + 2 * nu : nq + 2 * nu + nla_N]
        La_Nk1 = sk1[nq + 2 * nu + nla_N : nq + 2 * nu + 2 * nla_N]
        la_Fk1 = sk1[nq + 2 * nu + 2 * nla_N : nq + 2 * nu + 2 * nla_N + nla_F]
        La_Fk1 = sk1[nq + 2 * nu + 2 * nla_N + nla_F :]

        return q_dotk1, u_dotk1, Uk1, la_Nk1, La_Nk1, la_Fk1, La_Fk1

    def update_s(self, sk1):
        q_dotk1, u_dotk1, Uk1, la_Nk1, La_Nk1, la_Fk1, La_Fk1 = self.unpack_s(sk1)

        ################
        # backward Euler
        ################
        tk1 = self.tk + self.dt
        uk1_free = self.uk + self.dt * u_dotk1
        uk1 = uk1_free + Uk1
        qk1 = self.qk + self.dt * q_dotk1
        P_Nk1 = La_Nk1 + self.dt * la_Nk1
        P_Fk1 = La_Fk1 + self.dt * la_Fk1

        # ################
        # # trapezoid rule
        # ################
        # tk1 = self.tk + self.dt
        # uk1_free = self.uk + 0.5 * self.dt * (self.u_dotk + u_dotk1)
        # uk1 = uk1_free + Uk1
        # # uk1 = uk1_free + 0.5 * self.dt * (self.Uk + Uk1)
        # # uk1 = uk1_free + 0.5 * (self.Uk + Uk1)
        # qk1 = self.qk + 0.5 * self.dt * (self.q_dotk + q_dotk1)
        # # P_Nk1 = La_Nk1 + self.dt * la_Nk1
        # # P_Fk1 = La_Fk1 + self.dt * la_Fk1
        # P_Nk1 = La_Nk1 + 0.5 * self.dt * (self.la_Nk + la_Nk1)
        # P_Fk1 = La_Fk1 + 0.5 * self.dt * (self.la_Fk + la_Fk1)

        return tk1, qk1, uk1, uk1_free, P_Nk1, P_Fk1

    def Rs(self, sk1, update_index=False, use_percussions=False):
        nq = self.nq
        nu = self.nu
        nla_N = self.nla_N
        nla_F = self.nla_F
        mu = self.model.mu

        q_dotk1, u_dotk1, Uk1, la_Nk1, La_Nk1, la_Fk1, La_Fk1 = self.unpack_s(sk1)
        tk1, qk1, uk1, uk1_free, P_Nk1, P_Fk1 = self.update_s(sk1)

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)
        # TODO: Do we want to use uk1 or uk1_free in h-vector?
        hk1 = self.model.h(tk1, qk1, uk1_free)
        # hk1 = self.model.h(tk1, qk1, uk1)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_Nk1 = self.model.g_N(tk1, qk1)
        # g_N_dotk1_free = self.model.g_N_dot(tk1, qk1, uk1_free)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        # xi_Fk1_free = self.model.xi_F(tk1, qk1, self.uk, uk1_free)
        gamma_Fk1_free = self.model.gamma_F(tk1, qk1, uk1_free)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        ###################
        # evaluate residual
        ###################
        R = np.zeros(self.nq + 2 * self.nu + 2 * self.nla_N + 2 * self.nla_F)

        ####################
        # kinematic equation
        ####################
        R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1_free)

        ####################
        # euations of motion
        ####################
        R[nq : nq + nu] = Mk1 @ u_dotk1 - hk1 - W_Nk1 @ la_Nk1 - W_Fk1 @ la_Fk1

        #################
        # impact equation
        #################
        if use_percussions:
            R[nq + nu : nq + 2 * nu] = Mk1 @ Uk1 - W_Nk1 @ P_Nk1 - W_Fk1 @ P_Fk1
        else:
            R[nq + nu : nq + 2 * nu] = Mk1 @ Uk1 - W_Nk1 @ La_Nk1 - W_Fk1 @ La_Fk1

        ################
        # normal contact
        ################
        prox_r_N = self.model.prox_r_N(tk1, qk1)
        prox_arg = g_Nk1 - prox_r_N * la_Nk1
        if update_index:
            self.I_Nk1 = prox_arg <= 0.0
            # self.I_Nk1 = g_Nk1 <= 0.0
        # R[nq + 2 * nu : nq + 2 * nu + nla_N] = g_Nk1 - prox_R0_np(prox_arg)
        R[nq + 2 * nu : nq + 2 * nu + nla_N] = np.where(
            self.I_Nk1,
            g_Nk1,
            la_Nk1,
        )

        ##########
        # friction
        ##########
        prox_r_F = self.model.prox_r_F(tk1, qk1)
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                R[nq + 2 * nu + 2 * nla_N + i_F] = la_Fk1[i_F] + prox_sphere(
                    prox_r_F[i_N] * gamma_Fk1_free[i_F] - la_Fk1[i_F],
                    mu[i_N] * la_Nk1[i_N],
                )

        if self.with_elastic_impacts:

            ################
            # normal impacts
            ################
            if use_percussions:
                R[nq + 2 * nu + nla_N : nq + 2 * nu + 2 * nla_N] = np.where(
                    self.I_Nk1,
                    xi_Nk1 - prox_R0_np(xi_Nk1 - prox_r_N * P_Nk1),
                    P_Nk1,
                )
            else:
                R[nq + 2 * nu + nla_N : nq + 2 * nu + 2 * nla_N] = np.where(
                    self.I_Nk1,
                    xi_Nk1 - prox_R0_np(xi_Nk1 - prox_r_N * La_Nk1),
                    La_Nk1,
                )

            #################
            # tangent impacts
            #################
            for i_N, i_F in enumerate(self.model.NF_connectivity):
                i_F = np.array(i_F)
                if len(i_F) > 0:
                    if use_percussions:
                        R[nq + 2 * nu + 2 * nla_N + nla_F + i_F] = np.where(
                            self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                            P_Fk1[i_F]
                            + prox_sphere(
                                prox_r_F[i_N] * xi_Fk1[i_F] - P_Fk1[i_F],
                                mu[i_N] * P_Nk1[i_N],
                            ),
                            P_Fk1[i_F],
                        )
                    else:
                        R[nq + 2 * nu + 2 * nla_N + nla_F + i_F] = np.where(
                            self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                            La_Fk1[i_F]
                            + prox_sphere(
                                prox_r_F[i_N] * xi_Fk1[i_F] - La_Fk1[i_F],
                                mu[i_N] * La_Nk1[i_N],
                            ),
                            La_Fk1[i_F],
                        )
        else:
            R[nq + 2 * nu + nla_N : nq + 2 * nu + 2 * nla_N] = La_Nk1
            R[
                nq + 2 * nu + 2 * nla_N + nla_F : nq + 2 * nu + 2 * nla_N + 2 * nla_F
            ] = La_Fk1

        return R

    def Js(self, sk1):
        # return csc_matrix(approx_fprime(sk1, self.Rs, method="2-point"))
        return csc_matrix(approx_fprime(sk1, self.Rs, method="3-point"))
        # return csc_matrix(approx_fprime(sk1, self.Rs, method="cs"))

    def step(self, xk1, f, G):
        # initial residual and error
        R = f(xk1, update_index=True)
        error = self.error_function(R)
        converged = error < self.tol

        # print(f"initial error: {error}")
        j = 0
        if not converged:
            while j < self.max_iter:
                # jacobian
                J = G(xk1)

                # Newton update
                j += 1

                # dx = spsolve(J, R, use_umfpack=True)
                dx = spsolve(J, R, use_umfpack=False)

                # dx = lsqr(J, R, atol=1.0e-12, btol=1.0e-12)[0]

                # # no underflow errors
                # dx = np.linalg.lstsq(J.toarray(), R, rcond=None)[0]

                # # Can we get this sparse?
                # # using QR decomposition, see https://de.wikipedia.org/wiki/QR-Zerlegung#L%C3%B6sung_regul%C3%A4rer_oder_%C3%BCberbestimmter_Gleichungssysteme
                # b = R.copy()
                # Q, R = np.linalg.qr(J.toarray())
                # z = Q.T @ b
                # dx = np.linalg.solve(R, z)  # solving R*x = Q^T*b

                # # solve normal equation (should be independent of the conditioning
                # # number!)
                # dx = spsolve(J.T @ J, J.T @ R)

                xk1 -= dx

                R = f(xk1, update_index=True)
                error = self.error_function(R)
                converged = error < self.tol
                if converged:
                    break

            if not converged:
                # raise RuntimeError("internal Newton-Raphson not converged")
                print(f"not converged!")

        return converged, j, error, xk1

    def step_fixed_point(self):
        xk1 = np.concatenate((self.q_dotk, self.u_dotk, self.la_gk, self.la_gammak))

        # initial guess for nonsmooth Lagrange multipliers
        la_Nk1_i1 = self.la_Nk.copy()
        la_Fk1_i1 = self.la_Fk.copy()

        # initial guess for positions and velocities
        qk1_i = self.qk.copy()
        uk1_i = self.uk.copy()

        # compute smooth step
        xk1, converged_x, error_x, n_iter_x, R_x = fsolve(
            self.Rx_smooth, xk1, fun_args=(la_Nk1_i1, la_Fk1_i1)
        )
        tk1, qk1_i1, uk1_i1 = self.update_x_smooth(xk1)

        # fixed-point iteration
        i = 0
        diff = np.concatenate((qk1_i1 - qk1_i, uk1_i1 - uk1_i))
        error = self.error_function(diff)
        converged_x = error < self.tol
        # while (self.error_function(qk1_i1 - qk1_i) > self.tol) or (self.error_function(uk1_i1 - uk1_i) > self.tol):
        while not converged_x:
            i += 1

            # evaluate kinematic quantities of current iteration
            prox_r_N = self.model.prox_r_N(tk1, qk1_i1)
            prox_r_F = self.model.prox_r_F(tk1, qk1_i1)
            # r = 1.0e-0
            # prox_r_N = np.ones_like(prox_r_N) * r
            # prox_r_F = np.ones_like(prox_r_F) * r
            g_Nk1 = self.model.g_N(tk1, qk1_i1)
            gamma_Fk1 = self.model.gamma_F(tk1, qk1_i1, uk1_i1)

            # detect contacts
            self.I_Nk1 = g_Nk1 <= 0

            # if self.I_Nk1[0]:
            #     print(f"contact")

            # # fixed-point update normal direction
            # P_Nk1_i1[I_N] = prox_R0_np(
            #     P_Nk1_i[I_N]
            #     - prox_r_N[I_N] * self.system.xi_N(tk1, qk1, uk, uk1)[I_N]
            # )

            # # fixed-point update friction
            # xi_F = self.system.xi_F(tk1, qk1, uk, uk1)
            # for i_N, i_F in enumerate(self.NF_connectivity):
            #     if I_N[i_N] and len(i_F):
            #         P_Fk1_i1[i_F] = prox_sphere(
            #             P_Fk1_i[i_F] - prox_r_F[i_N] * xi_F[i_F],
            #             mu[i_N] * P_Nk1_i1[i_N],
            #         )

            # fixed-point update for normal contacts
            la_Nk1_i1 = np.where(
                self.I_Nk1,
                # -prox_R0_np(prox_r_N * g_Nk1 - la_Nk1_i1),
                prox_R0_np(la_Nk1_i1 - prox_r_N * g_Nk1),
                np.zeros_like(la_Nk1_i1),
            )

            # fixed-point update for friction
            mu = self.model.mu
            for i_N, i_F in enumerate(self.model.NF_connectivity):
                if len(i_F) > 0:
                    la_Fk1_i1[i_F] = np.where(
                        self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                        # -prox_sphere(prox_r_N * gamma_Fk1[i_F] - la_Fk1_i1[i_F], mu[i_N] * la_Nk1_i1[i_N]),
                        prox_sphere(
                            la_Fk1_i1[i_F] - prox_r_F * gamma_Fk1[i_F],
                            mu[i_N] * la_Nk1_i1[i_N],
                        ),
                        np.zeros(len(i_F), dtype=float),
                    )

            # update "previous" position and velocity estimates
            qk1_i = qk1_i1.copy()
            uk1_i = uk1_i1.copy()

            # recompute smooth step
            xk1, converged_x, error_x, n_iter_x, R_x = fsolve(
                self.Rx_smooth, xk1, fun_args=(la_Nk1_i1, la_Fk1_i1)
            )
            tk1, qk1_i1, uk1_i1 = self.update_x_smooth(xk1)

            diff = np.concatenate((qk1_i1 - qk1_i, uk1_i1 - uk1_i))
            error = self.error_function(diff)
            converged_x = error < self.tol

            print(f"- i: {i}")
            print(f"  error: {error}/ {self.tol}")

        # converged quantities of this time step
        qk1 = qk1_i1
        uk1 = uk1_i1
        la_Nk1 = la_Nk1_i1
        la_Fk1 = la_Fk1_i1

        q_dotk1, u_dotk1, la_gk1, la_gammak1 = self.unpack_x_smooth(xk1)

        Uk1 = np.zeros_like(uk1)
        La_gk1 = np.zeros_like(la_gk1)
        La_gammak1 = np.zeros_like(la_gammak1)
        La_Nk1 = np.zeros_like(la_Nk1)
        La_Fk1 = np.zeros_like(la_Fk1)

        return (
            (converged_x, i, error),
            tk1,
            q_dotk1,
            u_dotk1,
            qk1,
            uk1,
            Uk1,
            la_gk1,
            La_gk1,
            la_gammak1,
            La_gammak1,
            la_Nk1,
            La_Nk1,
            la_Fk1,
            La_Fk1,
        )

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        q_dot = [self.q_dotk]
        a = [self.u_dotk]
        U = [self.Uk]
        la_g = [self.la_gk]
        La_g = [self.La_gk]
        P_g = [self.dt * self.la_gk + self.La_gk]
        la_gamma = [self.la_gammak]
        La_gamma = [self.La_gammak]
        P_gamma = [self.dt * self.la_gammak + self.La_gammak]
        la_N = [self.la_Nk]
        La_N = [self.La_Nk]
        P_N = [self.dt * self.la_Nk + self.La_Nk]
        la_F = [self.la_Fk]
        La_F = [self.La_Fk]
        P_F = [self.dt * self.la_Fk + self.La_Fk]

        # prox_r_N = []
        # prox_r_F = []
        # error = []
        niter = [0]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            xk1 = self.xk.copy()
            yk1 = self.yk.copy()
            sk1 = self.sk.copy()

            # (converged_x, n_iter_x, error_x), tk1, q_dotk1, u_dotk1, qk1, uk1, Uk1, la_gk1, La_gk1, la_gammak1, La_gammak1, la_Nk1, La_Nk1, la_Fk1, La_Fk1 = self.step_fixed_point()

            # self.tk = tk1
            # self.qk = qk1.copy()
            # self.uk = uk1.copy()
            # self.la_gk = la_gk1.copy()
            # self.la_gammak = la_gammak1.copy()

            if self.solve_monolythic:
                converged, n_iter, error, sk1 = self.step(sk1, self.Rs, self.Js)
                niter.append(n_iter)
            else:
                converged_x, n_iter_x, error_x, xk1 = self.step(xk1, self.Rx, self.Jx)
                if self.with_elastic_impacts:
                    converged_y, n_iter_y, error_y, yk1 = self.step(
                        yk1, self.Ry, self.Jy
                    )

                if self.with_elastic_impacts:
                    niter.append(n_iter_x + n_iter_y)
                else:
                    niter.append(n_iter_x)

            # update progress bar and check convergence
            if self.solve_monolythic:
                pbar.set_description(
                    f"t: {tk1:0.2e}s < {self.t1:0.2e}s; ||R||: {error:0.2e} ({n_iter}/{self.max_iter})"
                )
            else:
                if self.with_elastic_impacts:
                    pbar.set_description(
                        f"t: {tk1:0.2e}s < {self.t1:0.2e}s; ||R_x||: {error_x:0.2e} ({n_iter_x}/{self.max_iter}); ||R_y||: {error_y:0.2e} ({n_iter_y}/{self.max_iter})"
                    )
                else:
                    pbar.set_description(
                        f"t: {tk1:0.2e}s < {self.t1:0.2e}s; ||R_x||: {error_x:0.2e} ({n_iter_x}/{self.max_iter})"
                    )
                converged = (
                    (converged_x and converged_y)
                    if self.with_elastic_impacts
                    else converged_x
                )
            if not converged:
                if self.solve_monolythic:
                    print(
                        f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                    )
                else:
                    print(
                        f"internal Newton-Raphson method not converged after {n_iter_x} x-steps with error: {error_x:.5e}"
                    )
                    if self.with_elastic_impacts:
                        print(
                            f"internal Newton-Raphson method not converged after {n_iter_y} y-steps with error: {error_y:.5e}"
                        )

                # write solution
                return Solution(
                    t=np.array(t),
                    q=np.array(q),
                    u=np.array(u),
                    q_dot=np.array(q_dot),
                    a=np.array(a),
                    U=np.array(U),
                    la_g=np.array(la_g),
                    La_g=np.array(La_g),
                    P_g=np.array(P_g),
                    la_gamma=np.array(la_gamma),
                    La_gamma=np.array(La_gamma),
                    P_gamma=np.array(P_gamma),
                    la_N=np.array(la_N),
                    La_N=np.array(La_N),
                    P_N=np.array(P_N),
                    la_F=np.array(la_F),
                    La_F=np.array(La_F),
                    P_F=np.array(P_F),
                    niter=np.array(niter),
                )

            if self.solve_monolythic:
                q_dotk1, u_dotk1, Uk1, la_Nk1, La_Nk1, la_Fk1, La_Fk1 = self.unpack_s(
                    sk1
                )
                tk1, qk1, uk1, uk1_free, P_Nk1, P_Fk1 = self.update_s(sk1)
            else:
                q_dotk1, u_dotk1, la_gk1, la_gammak1, la_Nk1, la_Fk1 = self.unpack_x(
                    xk1
                )
                tk1, qk1, uk1_free = self.update_x(xk1)

                Uk1, La_gk1, La_gammak1, La_Nk1, La_Fk1 = self.unpack_y(yk1)
                uk1 = uk1_free + Uk1

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            q_dot.append(q_dotk1)
            a.append(u_dotk1)
            U.append(Uk1)
            # la_g.append(la_gk1)
            # La_g.append(La_gk1)
            # P_g.append(self.dt * la_gk1 + La_gk1)
            # la_gamma.append(la_gammak1)
            # La_gamma.append(La_gammak1)
            # P_gamma.append(self.dt * la_gammak1 + La_gammak1)
            la_N.append(la_Nk1)
            La_N.append(La_Nk1)
            P_N.append(self.dt * la_Nk1 + La_Nk1)
            la_F.append(la_Fk1)
            La_F.append(La_Fk1)
            P_F.append(self.dt * la_Fk1 + La_Fk1)

            # update local variables for accepted time step
            self.tk = tk1

            # required for BDF2
            self.qk_1 = self.qk.copy()
            self.qk = qk1.copy()
            self.uk_1_free = self.uk_free.copy()
            self.uk_free = uk1_free.copy()
            self.uk_1 = self.uk.copy()
            self.uk = uk1.copy()
            self.fist_step = False

            self.q_dotk = q_dotk1.copy()
            self.u_dotk = u_dotk1.copy()
            self.Uk_1 = self.Uk.copy()
            self.Uk = Uk1.copy()
            # self.la_gk = la_gk1.copy()
            # self.la_gammak = la_gammak1.copy()
            self.la_Nk_1 = self.la_Nk.copy()
            self.la_Nk = la_Nk1.copy()
            self.la_Fk = la_Fk1.copy()

            self.xk = xk1.copy()
            self.yk = yk1.copy()
            self.sk = sk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            q_dot=np.array(q_dot),
            a=np.array(a),
            U=np.array(U),
            la_g=np.array(la_g),
            La_g=np.array(La_g),
            P_g=np.array(P_g),
            la_gamma=np.array(la_gamma),
            La_gamma=np.array(La_gamma),
            P_gamma=np.array(P_gamma),
            La_N=np.array(La_N),
            la_N=np.array(la_N),
            P_N=np.array(P_N),
            la_F=np.array(la_F),
            La_F=np.array(La_F),
            P_F=np.array(P_F),
            niter=np.array(niter),
        )
