import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, bmat, eye
from tqdm import tqdm

from cardillo.solver import Solution
from cardillo.math import prox_R0_nm, prox_R0_np, prox_sphere, approx_fprime

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
        self.nx = self.nq + self.nu + 2 * self.nla_N

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.P_Nk = dt * model.la_N0
        self.mu_Nk = np.zeros_like(self.P_Nk)
        # TODO: We have to solve for initial contact forces as well and
        # identify activ eindex sets?

        # initial velocites
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)
        M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        h0 = self.model.h(self.tk, self.qk, self.uk)
        W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        self.u_dotk = spsolve(M0, h0 + W_N0 @ self.P_Nk)

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
        # TODO: Add stabelized bilateral constraints
        self.xk = np.concatenate((self.qk, self.uk, self.P_Nk, self.mu_Nk))

    def unpack(self, x):
        nq = self.nq
        nu = self.nu
        nla_N = self.nla_N

        q = x[:nq]
        u = x[nq : nq + nu]
        P_N = x[nq + nu : nq + nu + nla_N]
        mu_N = x[nq + nu + nla_N :]

        return q, u, P_N, mu_N

    def R_gen(self, tk1, xk1):
        yield self.R(tk1, xk1, update_index_set=True)
        yield csr_matrix(
            approx_fprime(
                xk1, lambda x: self.R(tk1, x, update_index_set=False), method="2-point"
            )
        )

    def R(self, tk1, xk1, update_index_set=False, primal_form=True):
        nq = self.nq
        nu = self.nu
        nla_N = self.nla_N

        # extract all variables from xk and xk1
        qk, uk, P_Nk, mu_Nk = self.unpack(self.xk)
        qk1, uk1, P_Nk1, mu_Nk1 = self.unpack(xk1)

        # compute integrated mu as done in gen alpha
        mu_hat_Nk1 = mu_Nk1 + self.dt * P_Nk1
        # mu_hat_Nk1 = mu_Nk1 + P_Nk1

        if use_midpoint:
            q_M = 0.5 * (qk1 + qk)
            u_M = 0.5 * (uk1 + uk)

            M_M = self.model.M(tk1, q_M)
            W_N_M = self.model.W_N(tk1, q_M, scipy_matrix=csr_matrix)
            g_N_M = self.model.g_N(tk1, q_M)
            # g_Nk = self.model.g_N(tk1, qk)
            # g_Nk1 = self.model.g_N(tk1, qk1)
            # xi_N_M = self.model.xi_N(tk1, q_M, uk, uk1)
            g_Nk1 = self.model.g_N(tk1, qk1)
            xi_Nk1 = self.model.xi_N(tk1, qk1, uk, uk1)

            # update index sets
            # TODO: Which index set?
            if update_index_set:
                self.I_N = (
                    # mu_Nk1 - self.model.prox_r_N * g_Nk >= 0
                    # mu_Nk1 - self.model.prox_r_N * g_Nk1 >= 0
                    mu_Nk1 - self.model.prox_r_N * g_N_M
                    >= 0
                )

            # active contact set
            I_N = self.I_N
            _I_N = ~I_N
            I_N_ind = np.where(I_N)[0]
            _I_N_ind = np.where(_I_N)[0]

            ###################
            # evaluate residual
            ###################
            R = np.zeros(self.nx)

            # kinematic differential equation
            R[:nq] = (
                qk1
                - qk
                # TODO: Schindler uses weighted sum of kinematic equation?
                - self.dt * self.model.q_dot(tk1, q_M, u_M)
                + self.model.B(tk1, q_M) @ W_N_M @ mu_Nk1
            )

            # equations of motion
            R[nq : nq + nu] = (
                M_M @ (uk1 - uk) - self.dt * self.model.h(tk1, q_M, u_M) - W_N_M @ P_Nk1
            )

            # Mixed Signorini on velcity level and impact law
            R[nq + nu + I_N_ind] = P_Nk1[I_N] - prox_R0_np(
                P_Nk1[I_N] - self.model.prox_r_N[I_N] * xi_Nk1[I_N]
            )
            R[nq + nu + _I_N_ind] = P_Nk1[_I_N]

            # position stabilization
            R[nq + nu + nla_N :] = mu_Nk1 - prox_R0_np(
                mu_Nk1 - self.model.prox_r_N * g_Nk1
            )

        else:
            # evaluate repeatedly used quantities
            Mk1 = self.model.M(tk1, qk1)
            W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
            g_N_qk1 = self.model.g_N_q(tk1, qk1, scipy_matrix=csr_matrix)
            g_Nk1 = self.model.g_N(tk1, qk1)
            xi_Nk1 = self.model.xi_N(tk1, qk1, uk, uk1)

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
                self.A_N = prox_N_arg_position <= 0
                self.B_N = self.A_N * (prox_N_arg_velocity <= 0)

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
            R[:nq] = (
                qk1
                - qk
                - self.dt * self.model.q_dot(tk1, qk1, uk1)
                + g_N_qk1.T @ mu_Nk1
            )

            #####################
            # equations of motion
            #####################
            R[nq : nq + nu] = (
                Mk1 @ (uk1 - uk) - self.dt * self.model.h(tk1, qk1, uk1) - W_Nk1 @ P_Nk1
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

            R[nq + nu + B_N_ind] = xi_Nk1[B_N]
            R[nq + nu + _B_N_ind] = P_Nk1[_B_N]

            ########################
            # position stabilization
            ########################
            # if primal_form:
            #     # TODO: Why is this the prox on the positive real numbers?
            #     R[nq + nu + nla_N :] = g_Nk1 - prox_R0_np(prox_N_arg_position)
            # else:
            #     R[nq + nu + nla_N :] = -mu_hat_Nk1 - prox_R0_nm(prox_N_arg_position)

            R[nq + nu + nla_N + A_N_ind] = g_Nk1[A_N]
            R[nq + nu + nla_N + _A_N_ind] = mu_hat_Nk1[_A_N]

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

                # if tk1 > 1.19:
                # # if tk1 > 1.21:
                #     print(f"xk: {self.xk}")
                #     print(f"xk1: {xk1}")
                #     print(f"R: {R}")
                #     print(f"I_N: {self.I_N}")
                #     print(f"")

                error = self.error_function(R)
                converged = error < self.tol
                if converged:
                    break

        return converged, j, error, xk1

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
            qk1, uk1, P_Nk1, mu_Nk1 = self.unpack(xk1)

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
