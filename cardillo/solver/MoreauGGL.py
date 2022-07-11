import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, bmat, eye
from tqdm import tqdm

from cardillo.solver import Solution
from cardillo.math import norm, prox_R0_nm, prox_R0_np, prox_sphere, approx_fprime

# use_midpoint = True
use_midpoint = False


class MoreauGGL:
    """Moreau's midpoint rule with GGL stabilization for unilateral contacts, 
    see Schoeder2013 and Schindler2015 section 15.2.

    References
    ----------
    Schoeder2013: https://doi.org/10.1007/s11044-013-9370-y \\
    Schindler2015: https://mediatum.ub.tum.de/download/1422349/1422349.pdf
    """

    def __init__(
        self,
        model,
        t1,
        dt,
        tol=1e-10,
        max_iter=40,
        error_function=lambda x: np.max(np.abs(x)),
        numerical_jacobian=True,
    ):
        if numerical_jacobian == False:
            raise NotImplementedError("Analytical Jacobian is not implemented yet!")

        self.model = model

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
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F
        self.nx_s = self.nq + self.nu
        self.nx = self.nx_s + 2 * self.nla_N + self.nla_F

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.P_Nk = dt * model.la_N0
        self.mu_Nk = np.zeros_like(self.P_Nk)
        self.P_Fk = dt * model.la_F0

        # initial velocites
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        h0 = self.model.h(self.tk, self.qk, self.uk)
        W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_F0 = self.model.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)
        self.u_dotk = spsolve(M0, h0 + W_N0 @ self.P_Nk + W_F0 @ self.P_Fk)

        # # check if initial conditions satisfy constraints on position, velocity
        # # and acceleration level
        # g0 = model.g(self.tk, self.qk)
        # g_dot0 = model.g_dot(self.tk, self.qk, self.uk)
        # g_ddot0 = model.g_ddot(self.tk, self.qk, self.uk, self.u_dotk)
        # gamma0 = model.gamma(self.tk, self.qk, self.uk)
        # gamma_dot0 = model.gamma_dot(self.tk, self.qk, self.uk, self.u_dotk)

        # assert np.allclose(g0, np.zeros(self.nla_g)), "Initial conditions do not fulfill g0!"
        # assert np.allclose(g_dot0, np.zeros(self.nla_g)), "Initial conditions do not fulfill g_dot0!"
        # assert np.allclose(g_ddot0, np.zeros(self.nla_g)), "Initial conditions do not fulfill g_ddot0!"
        # assert np.allclose(gamma0, np.zeros(self.nla_gamma)), "Initial conditions do not fulfill gamma0!"
        # assert np.allclose(gamma_dot0, np.zeros(self.nla_gamma)), "Initial conditions do not fulfill gamma_dot0!"

        #######################################################################
        # starting values for generalized state vector, its derivatives and
        # auxiliary velocities
        #######################################################################
        self.xk = np.concatenate((self.qk, self.uk, self.P_Nk, self.mu_Nk, self.P_Fk))
        # self.xk = np.concatenate((self.q_dotk, self.u_dotk, self.P_Nk, self.mu_Nk, self.P_Fk))

        # initialize index sets
        self.A_N = np.zeros(self.nla_N, dtype=bool)
        self.B_N = np.zeros(self.nla_N, dtype=bool)
        # self.Ck1 = np.zeros(self.nla_N, dtype=bool)
        self.D_st = np.zeros(self.nla_N, dtype=bool)
        # self.Ek1_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nq = self.nq
        nu = self.nu
        nx_s = self.nx_s
        nla_N = self.nla_N
        nla_F = self.nla_F

        q = x[:nq]
        u = x[nq : nq + nu]
        P_N = x[nx_s : nx_s + nla_N]
        mu_N = x[nx_s + nla_N : nx_s + 2 * nla_N]
        P_F = x[nx_s + 2 * nla_N : nx_s + 2 * nla_N + nla_F]

        return q, u, P_N, mu_N, P_F

    # def update(self, xk1):
    #     q_dotk, u_dotk, P_Nk, mu_Nk, P_Fk = self.unpack(self.xk)
    #     q_dotk1, u_dotk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

    #     dt = self.dt
    #     qk1 = self.qk + dt * q_dotk1
    #     uk1 = self.uk + dt * u_dotk1
    #     return qk1, uk1

    def R_gen(self, tk1, xk1):
        yield self.R(tk1, xk1, update_index_set=True)
        yield csr_matrix(
            approx_fprime(
                xk1, lambda x: self.R(tk1, x, update_index_set=False), method="2-point"
            )
        )

    def R(self, tk1, xk1, update_index_set=False, primal_form=False):
        nq = self.nq
        nu = self.nu
        nx_s = self.nx_s
        nla_N = self.nla_N
        nla_F = self.nla_F
        dt = self.dt
        mu = self.model.mu

        # extract all variables from xk and xk1
        qk, uk, P_Nk, mu_Nk, P_Fk = self.unpack(self.xk)
        qk1, uk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

        # compute integrated mu as done in gen alpha
        # mu_hat_Nk1 = mu_Nk1 # TODO: This is not working!
        mu_hat_Nk1 = mu_Nk1 + self.dt * P_Nk1  # TODO: This is the key ingredient!

        # update kinematic quantities (trivial for Euler backward!)
        q_dotk1 = (qk1 - qk) / dt
        u_dotk1 = (uk1 - uk) / dt

        # # TODO: Investigate theta method
        # theta = 0.5
        # q_dotk1 = (theta * qk + (1.0 - theta) * qk1 - qk) / dt
        # u_dotk1 = (theta * uk + (1.0 - theta) * uk1 - uk) / dt

        # mu_Nk1 = theta * mu_Nk + (1.0 - theta) * mu_Nk1
        # mu_hat_Nk = mu_Nk + self.dt * P_Nk # TODO: This is the key ingredient!
        # mu_hat_Nk1 = theta * mu_hat_Nk + (1.0 - theta) * mu_hat_Nk1
        # P_Nk1 = theta * P_Nk + (1.0 - theta) * P_Nk1
        # P_Fk1 = theta * P_Fk + (1.0 - theta) * P_Fk1

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_N_qk1 = self.model.g_N_q(tk1, qk1, scipy_matrix=csr_matrix)
        gamma_F_qk1 = self.model.gamma_F_q(tk1, qk1, uk1, scipy_matrix=csr_matrix)
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, uk, uk1)

        ###################
        # update index sets
        ###################
        primal_form = True
        # primal_form = False
        if primal_form:
            prox_N_arg_position = g_Nk1 - self.model.prox_r_N * mu_hat_Nk1
            prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        else:
            prox_N_arg_position = -mu_hat_Nk1 + self.model.prox_r_N * g_Nk1
            prox_N_arg_velocity = -P_Nk1 + self.model.prox_r_N * xi_Nk1
        if update_index_set:
            # normal contact sets
            self.A_N = prox_N_arg_position <= 0
            self.B_N = self.A_N * (prox_N_arg_velocity <= 0)

            # frictional contact sets
            for i_N, i_F in enumerate(self.model.NF_connectivity):
                i_F = np.array(i_F)
                if len(i_F) > 0:
                    # eqn. (139):
                    self.D_st[i_N] = self.A_N[i_N] and (
                        norm(self.model.prox_r_F[i_N] * xi_Fk1[i_F] - P_Fk1[i_F])
                        <= mu[i_N] * P_Nk1[i_N]
                    )

        A_N = self.A_N
        _A_N = ~A_N
        A_N_ind = np.where(A_N)[0]
        _A_N_ind = np.where(_A_N)[0]

        B_N = self.B_N
        _B_N = ~B_N
        B_N_ind = np.where(B_N)[0]
        _B_N_ind = np.where(_B_N)[0]

        ###################
        # evaluate residual
        ###################
        R = np.zeros(self.nx)

        #################################
        # kinematic differential equation
        #################################
        # R[:nq] = (
        #     qk1
        #     - qk
        #     - dt * self.model.q_dot(tk1, qk1, uk1)
        #     - g_N_qk1.T @ mu_Nk1
        #     - gamma_F_qk1.T @ (dt * P_Fk) # TODO: Not necessary but consistent
        # )
        R[:nq] = (
            q_dotk1
            - self.model.q_dot(tk1, qk1, uk1)
            - g_N_qk1.T @ mu_Nk1 / dt
            - gamma_F_qk1.T @ P_Fk1  # TODO: Not necessary but consistent
        )

        #####################
        # equations of motion
        #####################
        # R[nq : nq + nu] = (
        #     Mk1 @ (uk1 - uk) - dt * self.model.h(tk1, qk1, uk1) - W_Nk1 @ P_Nk1 - W_Fk1 @ P_Fk1
        # )
        R[nq : nq + nu] = (
            Mk1 @ u_dotk1
            - self.model.h(tk1, qk1, uk1)
            - W_Nk1 @ (P_Nk1 / dt)
            - W_Fk1 @ (P_Fk1 / dt)
        )

        #################################################
        # Mixed Signorini on velcity level and impact law
        #################################################
        # if primal_form:
        #     # TODO: Why is this the prox on the positive real numbers?
        #     R[nq + nu + A_N_ind] = xi_Nk1 - prox_R0_np(prox_N_arg_velocity)
        # else:
        #     R[nq + nu + A_N_ind] = -P_Nk1 - prox_R0_nm(prox_N_arg_velocity)
        # R[nq + nu + _A_N_ind] = P_Nk1[_A_N]

        R[nx_s + B_N_ind] = xi_Nk1[B_N]
        R[nx_s + _B_N_ind] = P_Nk1[_B_N]

        ########################
        # position stabilization
        ########################
        # if primal_form:
        #     # TODO: Why is this the prox on the positive real numbers?
        #     R[nq + nu + nla_N :] = g_Nk1 - prox_R0_np(prox_N_arg_position)
        # else:
        #     R[nq + nu + nla_N :] = -mu_hat_Nk1 - prox_R0_nm(prox_N_arg_position)

        R[nx_s + nla_N + A_N_ind] = g_Nk1[A_N]
        R[nx_s + nla_N + _A_N_ind] = mu_hat_Nk1[_A_N]

        ##########
        # friction
        ##########
        D_st = self.D_st

        # # TODO: No friction case can be implemented like this:
        # R[nx_s + 2 * nla_N :] = P_Fk1

        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                if A_N[i_N]:

                    # if primal_form:
                    #     raise NotImplementedError
                    #     R[nx_s + 2 * nla_N + i_F] = xi_Fk1[i_F] - prox_sphere(xi_Fk1[i_F] - self.model.prox_r_F[i_N] * P_Fk1[i_F], mu[i_N] * P_Nk1[i_N])
                    # else:
                    #     raise NotImplementedError
                    #     R[nx_s + 2 * nla_N + i_F] = -P_Fk1[i_F] - prox_sphere(-P_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F], mu[i_N] * P_Nk1[i_N])

                    if D_st[i_N]:
                        # eqn. (138a)
                        R[nx_s + 2 * nla_N + i_F] = xi_Fk1[i_F]
                    else:
                        # eqn. (138b)
                        norm_xi_Fi1 = norm(xi_Fk1[i_F])
                        xi_Fk1_normalized = xi_Fk1.copy()
                        if norm_xi_Fi1 > 0:
                            xi_Fk1_normalized /= norm_xi_Fi1
                        R[nx_s + 2 * nla_N + i_F] = (
                            P_Fk1[i_F] + mu[i_N] * P_Nk1[i_N] * xi_Fk1_normalized[i_F]
                        )
                else:
                    # eqn. (138c)
                    R[nx_s + 2 * nla_N + i_F] = P_Fk1[i_F]

        return R

    def step(self, tk1, xk1):
        # initial residual and error
        R_gen = self.R_gen(tk1, xk1)
        R = next(R_gen)
        error = self.error_function(R)
        converged = error < self.tol
        j = 0
        if not converged:
            while j < self.max_iter:
                # jacobian
                R_x = next(R_gen)

                # Newton update
                j += 1
                dx = spsolve(R_x, R, use_umfpack=True)
                xk1 -= dx
                R_gen = self.R_gen(tk1, xk1)
                R = next(R_gen)

                error = self.error_function(R)
                converged = error < self.tol
                if converged:
                    break

        return converged, j, error, xk1

    # TODO: Step fixed point!
    # def step_fixed_point(self, tk1, xk1):
    #     def R_s(tk1, yk1):
    #         nq = self.nq
    #         nu = self.nu

    #         qk1 = yk1[:nq]
    #         uk1 = yk1[nq:nq + nu]

    #         # evaluate repeatedly used quantities
    #         Mk1 = self.model.M(tk1, qk1)
    #         W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
    #         g_Nk1 = self.model.g_N(tk1, qk1)
    #         # g_Nk1 = self.model.g_N(tk1, 0.5 * (qk + qk1))
    #         # g_Nk1 = self.model.g_N(tk1, qk + 0.5 * self.dt * self.model.q_dot(self.tk, qk1, uk1))
    #         xi_Nk1 = self.model.xi_N(tk1, qk1, uk, uk1)

    #         R = np.zeros(nq + nu)

    #         # kinematic differential equation
    #         R[:nq] = (
    #             qk1
    #             - qk
    #             - self.dt * self.model.q_dot(tk1, qk1, uk1)
    #             + self.model.B(tk1, qk1) @ W_Nk1 @ mu_Nk1
    #         )

    #         # equations of motion
    #         R[nq : nq + nu] = (
    #             Mk1 @ (uk1 - uk) - self.dt * self.model.h(tk1, qk1, uk1) - W_Nk1 @ P_Nk1
    #         )

    #     nq = self.nq
    #     nu = self.nu
    #     nla_N = self.nla_N

    #     # extract all variables from xk and xk1
    #     qk, uk, P_Nk, mu_Nk = self.unpack(self.xk)
    #     qk1, uk1, P_Nk1, mu_Nk1 = self.unpack(xk1)

    #     # initial residual and error
    #     R_gen = self.R_gen(tk1, xk1)
    #     R = next(R_gen)
    #     R_s = R[:nq + nu]

    #     # identify active contacts
    #     g_Nk1 = self.model.q_N(tk1, qk1)

    #     error = self.error_function(R)
    #     converged = error < self.tol
    #     j = 0
    #     if not converged:
    #         while j < self.max_iter:
    #             # jacobian
    #             R_x = next(R_gen)

    #             # Newton update
    #             j += 1
    #             dx = spsolve(R_x, R, use_umfpack=True)
    #             xk1 -= dx
    #             R_gen = self.R_gen(tk1, xk1)
    #             R = next(R_gen)

    #             # if tk1 > 1.19:
    #             # # if tk1 > 1.21:
    #             #     print(f"xk: {self.xk}")
    #             #     print(f"xk1: {xk1}")
    #             #     print(f"R: {R}")
    #             #     print(f"I_N: {self.I_N}")
    #             #     print(f"")

    #             error = self.error_function(R)
    #             converged = error < self.tol
    #             if converged:
    #                 break

    #     return converged, j, error, xk1

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        # mu_g = [self.mu_gk]
        # P_g = [self.P_gk]
        # P_gamma = [self.P_gammak]
        P_N = [self.P_Nk]
        mu_N = [self.mu_Nk]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            xk1 = self.xk.copy()  # This copy is mandatory since we modify xk1
            # in the step function

            # # perform Euler forward
            # tk = self.tk
            # dt = self.dt
            # nq = self.nq
            # nu = self.nu
            # nla_N = self.nla_N
            # qk = self.xk[:nq]
            # uk = self.xk[nq:nq + nu]
            # xk1[:nq] = qk + dt * self.model.q_dot(tk, qk, uk)
            # Mk = self.model.M(tk, qk)
            # hk = self.model.h(tk, qk, uk)
            # f_N = self.model.W_N(tk, qk) @ xk1[nq + nu : nq + nu + nla_N]
            # xk1[nq:nq + nu] = self.xk[:nq] + dt * spsolve(Mk, hk + f_N)

            converged, n_iter, error, xk1 = self.step(tk1, xk1)
            qk1, uk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )
            if not converged:
                raise RuntimeError(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            # mu_g.append(mu_gk1)
            # P_g.append(P_gk1)
            # P_gamma.append(P_gammak1)
            mu_N.append(mu_Nk1)
            P_N.append(P_Nk1)

            # update local variables for accepted time step
            self.tk = tk1
            self.xk = xk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            P_N=np.array(P_N),
            mu_N=np.array(mu_N),
        )
        # mu_g=np.array(mu_g), P_g=np.array(P_g),
        # P_gamma=np.array(P_gamma),
