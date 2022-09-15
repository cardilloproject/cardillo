import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, bmat, eye
from tqdm import tqdm

from cardillo.solver import Solution
from cardillo.math import norm, prox_R0_nm, prox_R0_np, prox_sphere, approx_fprime

use_position_formulation = True
# use_position_formulation = False


# TODO: Not working yet!
class NonsmoothEulerBackwardsGGL:
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
        self.nla_g = model.nla_g
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F
        self.nx_s = self.nq + 2 * self.nu + 2 * self.nla_g
        self.nx = self.nx_s + 2 * self.nla_N + self.nla_F

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.u_sk = model.u0
        self.Uk = np.zeros_like(self.uk)
        self.P_gk = dt * model.la_g0
        self.mu_gk = np.zeros_like(self.P_gk)
        self.P_Nk = dt * model.la_N0
        self.mu_Nk = np.zeros_like(self.P_Nk)
        self.P_Fk = dt * model.la_F0

        # initial velocites
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        h0 = self.model.h(self.tk, self.qk, self.uk)
        W_g0 = self.model.W_g(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_F0 = self.model.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)
        self.u_dot_sk = spsolve(
            M0, h0 + W_g0 @ model.la_g0 + W_N0 @ model.la_N0 + W_F0 @ model.la_F0
        )

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
        self.xk = np.concatenate(
            (
                self.q_dotk,
                self.u_dot_sk,
                self.Uk,
                self.P_gk,
                self.mu_gk,
                self.P_Nk,
                self.mu_Nk,
                self.P_Fk,
            )
        )

        # initialize index sets
        self.A_N = np.zeros(self.nla_N, dtype=bool)
        self.B_N = np.zeros(self.nla_N, dtype=bool)
        self.D_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        nla_F = self.nla_F

        q = x[:nq]
        u = x[nq : nq + nu]
        U = x[nq + nu : nq + 2 * nu]
        P_g = x[nq + 2 * nu : nq + 2 * nu + nla_g]
        mu_g = x[nq + 2 * nu + nla_g : nq + 2 * nu + 2 * nla_g]
        P_N = x[nx_s : nx_s + nla_N]
        mu_N = x[nx_s + nla_N : nx_s + 2 * nla_N]
        P_F = x[nx_s + 2 * nla_N : nx_s + 2 * nla_N + nla_F]

        return q, u, U, P_g, mu_g, P_N, mu_N, P_F

    def update(self, xk1):
        dt = self.dt
        q_dotk1, u_dotk1, Uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

        # implicit Euler
        qk1 = self.qk + dt * q_dotk1
        u_sk1 = self.uk + dt * u_dotk1

        uk1 = u_sk1 + Uk1

        return qk1, u_sk1, uk1

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
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        dt = self.dt
        mu = self.model.mu

        # extract all variables from xk1
        q_dotk1, u_dot_sk1, Uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

        # update generalzed coordiantes
        qk1, u_sk1, uk1 = self.update(xk1)

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)

        gk1 = self.model.g(tk1, qk1)
        g_dotk1 = self.model.g_dot(tk1, qk1, uk1)  # TODO: Only smooth part?
        g_qk1 = self.model.g_q(tk1, qk1, scipy_matrix=csr_matrix)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)

        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        g_N_qk1 = self.model.g_N_q(tk1, qk1, scipy_matrix=csr_matrix)
        g_Nk1 = self.model.g_N(tk1, qk1)

        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        gamma_F_qk1 = self.model.gamma_F_q(tk1, qk1, uk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        ###################
        # update index sets
        ###################
        primal_form = True
        # primal_form = False
        if primal_form:
            prox_N_arg_position = g_Nk1 - self.model.prox_r_N * mu_Nk1
            prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        else:
            prox_N_arg_position = -mu_Nk1 + self.model.prox_r_N * g_Nk1
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
        # R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, u_sk1) - g_N_qk1.T @ mu_Nk1 - g_qk1.T @ mu_gk1
        R[:nq] = (
            q_dotk1
            # # TODO: Impulsive uk1 is not working
            # - self.model.q_dot(tk1, qk1, uk1)
            # TODO: Smooth u_sk1 is working
            - self.model.q_dot(tk1, qk1, u_sk1)
            - g_qk1.T @ mu_gk1
            - g_N_qk1.T @ mu_Nk1
            # - 0.5 * dt * gamma_F_qk1.T @ P_Fk1
        )

        #####################
        # equations of motion
        #####################
        # R[nq : nq + nu] = Mk1 @ u_dot_sk1 - self.model.h(tk1, qk1, uk1)
        R[nq : nq + nu] = Mk1 @ u_dot_sk1 - self.model.h(tk1, qk1, u_sk1)

        #################
        # impact equation
        #################
        R[nq + nu : nq + 2 * nu] = (
            Mk1 @ Uk1 - W_gk1 @ P_gk1 - W_Nk1 @ P_Nk1 - W_Fk1 @ P_Fk1
        )

        #######################################################
        # bilateral constraints on position and velocitiy level
        #######################################################
        R[nq + 2 * nu : nq + 2 * nu + nla_g] = g_dotk1
        R[nq + 2 * nu + nla_g : nq + 2 * nu + 2 * nla_g] = gk1

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
        #     R[nq + nu + nla_N :] = -mu_Nk1 - prox_R0_nm(prox_N_arg_position)

        R[nx_s + nla_N + A_N_ind] = g_Nk1[A_N]
        R[nx_s + nla_N + _A_N_ind] = mu_Nk1[_A_N]

        ##########
        # friction
        ##########

        # TODO: No friction case can be implemented like this:
        R[nx_s + 2 * nla_N :] = P_Fk1

        # D_st = self.D_st

        # for i_N, i_F in enumerate(self.model.NF_connectivity):
        #     i_F = np.array(i_F)
        #     if len(i_F) > 0:
        #         if A_N[i_N]:

        #             # if primal_form:
        #             #     raise NotImplementedError
        #             #     R[nx_s + 2 * nla_N + i_F] = xi_Fk1[i_F] - prox_sphere(xi_Fk1[i_F] - self.model.prox_r_F[i_N] * P_Fk1[i_F], mu[i_N] * P_Nk1[i_N])
        #             # else:
        #             #     raise NotImplementedError
        #             #     R[nx_s + 2 * nla_N + i_F] = -P_Fk1[i_F] - prox_sphere(-P_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F], mu[i_N] * P_Nk1[i_N])

        #             if D_st[i_N]:
        #                 # eqn. (138a)
        #                 R[nx_s + 2 * nla_N + i_F] = xi_Fk1[i_F]
        #             else:
        #                 # eqn. (138b)
        #                 norm_xi_Fi1 = norm(xi_Fk1[i_F])
        #                 xi_Fk1_normalized = xi_Fk1.copy()
        #                 if norm_xi_Fi1 > 0:
        #                     xi_Fk1_normalized /= norm_xi_Fi1
        #                 R[nx_s + 2 * nla_N + i_F] = (
        #                     P_Fk1[i_F] + mu[i_N] * P_Nk1[i_N] * xi_Fk1_normalized[i_F]
        #                 )
        #         else:
        #             # eqn. (138c)
        #             R[nx_s + 2 * nla_N + i_F] = P_Fk1[i_F]

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

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        U = [self.Uk]
        P_g = [self.P_gk]
        mu_g = [self.mu_gk]
        # P_gamma = [self.P_gammak]
        P_N = [self.P_Nk]
        mu_N = [self.mu_Nk]
        P_F = [self.P_Fk]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            xk1 = self.xk.copy()  # This copy is mandatory since we modify xk1
            # in the step function

            converged, n_iter, error, xk1 = self.step(tk1, xk1)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )
            if not converged:
                # raise RuntimeError(
                #     f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                # )
                print(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )
                # write solution
                return Solution(
                    t=np.array(t),
                    q=np.array(q),
                    u=np.array(u),
                    U=np.array(U),
                    P_g=np.array(P_g),
                    mu_g=np.array(mu_g),
                    P_N=np.array(P_N),
                    mu_N=np.array(mu_N),
                    P_F=np.array(P_F),
                )

            # extract all variables from xk and xk1
            q_dotk1, u_dot_sk1, Uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(
                xk1
            )

            # update generalzed coordiantes
            qk1, u_sk1, uk1 = self.update(xk1)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # update converged and updated quantities of previous time step
            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.q_dotk = q_dotk1.copy()
            self.u_dot_sk = u_dot_sk1.copy()
            self.u_sk = u_sk1.copy()

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            U.append(Uk1)
            P_g.append(P_gk1)
            mu_g.append(mu_gk1)
            P_N.append(P_Nk1)
            mu_N.append(mu_Nk1)
            P_F.append(P_Fk1)

            # update local variables for accepted time step
            self.tk = tk1
            self.xk = xk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            U=np.array(U),
            P_g=np.array(P_g),
            mu_g=np.array(mu_g),
            P_N=np.array(P_N),
            mu_N=np.array(mu_N),
            P_F=np.array(P_F),
        )


class NonsmoothThetaGGL:
    """Moreau's midpoint rule with GGL stabilization for unilateral contacts, 
    see Schoeder2013 and Schindler2015 section 15.2.

    TODO: Bilateral constraints on velocity level!

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
        theta=0.5,
        tol=1e-10,
        max_iter=40,
        error_function=lambda x: np.max(np.abs(x)),
        numerical_jacobian=True,
    ):
        if numerical_jacobian == False:
            raise NotImplementedError("Analytical Jacobian is not implemented yet!")

        self.model = model
        # 0:   Euler forward
        # 0.5: trapezoidla rule
        # 1:   Euler backward
        self.theta = theta

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
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F
        self.nx_s = self.nq + 2 * self.nu + 2 * self.nla_g
        self.nx = self.nx_s + 2 * self.nla_N + self.nla_F

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.u_sk = model.u0
        self.Uk = np.zeros_like(self.uk)
        self.P_gk = dt * model.la_g0
        self.mu_gk = np.zeros_like(self.P_gk)
        self.P_Nk = dt * model.la_N0
        self.mu_Nk = np.zeros_like(self.P_Nk)
        self.P_Fk = dt * model.la_F0

        # initial velocites
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        h0 = self.model.h(self.tk, self.qk, self.uk)
        W_g0 = self.model.W_g(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_F0 = self.model.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)
        self.u_dot_sk = spsolve(
            M0, h0 + W_g0 @ model.la_g0 + W_N0 @ model.la_N0 + W_F0 @ model.la_F0
        )

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
        self.xk = np.concatenate(
            (
                self.q_dotk,
                self.u_dot_sk,
                self.Uk,
                self.P_gk,
                self.mu_gk,
                self.P_Nk,
                self.mu_Nk,
                self.P_Fk,
            )
        )

        # initialize index sets
        self.A_N = np.zeros(self.nla_N, dtype=bool)
        self.B_N = np.zeros(self.nla_N, dtype=bool)
        self.D_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        nla_F = self.nla_F

        q = x[:nq]
        u = x[nq : nq + nu]
        U = x[nq + nu : nq + 2 * nu]
        P_g = x[nq + 2 * nu : nq + 2 * nu + nla_g]
        mu_g = x[nq + 2 * nu + nla_g : nq + 2 * nu + 2 * nla_g]
        P_N = x[nx_s : nx_s + nla_N]
        mu_N = x[nx_s + nla_N : nx_s + 2 * nla_N]
        P_F = x[nx_s + 2 * nla_N : nx_s + 2 * nla_N + nla_F]

        return q, u, U, P_g, mu_g, P_N, mu_N, P_F

    def update(self, xk1):
        dt = self.dt
        q_dotk1, u_dot_sk1, Uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

        # Theta method
        qk1 = (
            self.qk + dt * (1.0 - self.theta) * self.q_dotk + dt * self.theta * q_dotk1
        )
        u_sk1 = (
            self.uk
            + dt * (1.0 - self.theta) * self.u_dot_sk
            + dt * self.theta * u_dot_sk1
        )

        uk1 = u_sk1 + Uk1

        return qk1, u_sk1, uk1

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
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        dt = self.dt
        mu = self.model.mu

        # extract all variables from xk1
        q_dotk1, u_dot_sk1, Uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

        # update generalzed coordiantes
        qk1, u_sk1, uk1 = self.update(xk1)

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)

        gk1 = self.model.g(tk1, qk1)
        g_dotk1 = self.model.g_dot(tk1, qk1, uk1)
        g_qk1 = self.model.g_q(tk1, qk1, scipy_matrix=csr_matrix)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)

        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        g_N_qk1 = self.model.g_N_q(tk1, qk1, scipy_matrix=csr_matrix)
        g_Nk1 = self.model.g_N(tk1, qk1)

        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        gamma_F_qk1 = self.model.gamma_F_q(tk1, qk1, uk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        ###################
        # update index sets
        ###################
        primal_form = True
        # primal_form = False
        if primal_form:
            prox_N_arg_position = g_Nk1 - self.model.prox_r_N * mu_Nk1
            prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        else:
            prox_N_arg_position = -mu_Nk1 + self.model.prox_r_N * g_Nk1
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
        # R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, u_sk1) - g_N_qk1.T @ mu_Nk1 - g_qk1.T @ mu_gk1
        R[:nq] = (
            q_dotk1
            # # TODO: Implicit kinematic equations is not working
            # - (1.0 - self.theta) * self.model.q_dot(self.tk, self.qk, self.uk)
            # - self.theta * self.model.q_dot(tk1, qk1, uk1)
            # # TODO: Explicit kinematic equations is working but with strange physics
            # - (1.0 - self.theta) * self.model.q_dot(self.tk, self.qk, self.u_sk)
            # - self.theta * self.model.q_dot(tk1, qk1, u_sk1)
            - self.model.q_dot(tk1, qk1, u_sk1)
            # - self.model.q_dot(tk1, qk1, uk1)
            - g_qk1.T @ mu_gk1
            - g_N_qk1.T @ mu_Nk1
            # - 0.5 * dt * gamma_F_qk1.T @ P_Fk1 # TODO: Necessary?
        )

        #####################
        # equations of motion
        #####################
        R[nq : nq + nu] = (
            Mk1 @ u_dot_sk1
            # - (1.0 - self.theta) * self.model.h(self.tk, self.qk, self.uk)
            # - self.theta * self.model.h(tk1, qk1, uk1)
            - self.model.h(tk1, qk1, uk1)
        )

        #################
        # impact equation
        #################
        R[nq + nu : nq + 2 * nu] = (
            Mk1 @ Uk1 - W_gk1 @ P_gk1 - W_Nk1 @ P_Nk1 - W_Fk1 @ P_Fk1
        )

        #######################################################
        # bilateral constraints on position and velocitiy level
        #######################################################
        R[nq + 2 * nu : nq + 2 * nu + nla_g] = g_dotk1
        R[nq + 2 * nu + nla_g : nq + 2 * nu + 2 * nla_g] = gk1

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
        #     R[nq + nu + nla_N :] = -mu_Nk1 - prox_R0_nm(prox_N_arg_position)

        R[nx_s + nla_N + A_N_ind] = g_Nk1[A_N]
        R[nx_s + nla_N + _A_N_ind] = mu_Nk1[_A_N]

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

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        U = [self.Uk]
        P_g = [self.P_gk]
        mu_g = [self.mu_gk]
        # P_gamma = [self.P_gammak]
        P_N = [self.P_Nk]
        mu_N = [self.mu_Nk]
        P_F = [self.P_Fk]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            xk1 = self.xk.copy()  # This copy is mandatory since we modify xk1
            # in the step function

            converged, n_iter, error, xk1 = self.step(tk1, xk1)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )
            if not converged:
                # raise RuntimeError(
                #     f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                # )
                print(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )
                # write solution
                return Solution(
                    t=np.array(t),
                    q=np.array(q),
                    u=np.array(u),
                    U=np.array(U),
                    P_g=np.array(P_g),
                    mu_g=np.array(mu_g),
                    P_N=np.array(P_N),
                    mu_N=np.array(mu_N),
                    P_F=np.array(P_F),
                )

            # extract all variables from xk and xk1
            q_dotk1, u_dot_sk1, Uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(
                xk1
            )

            # update generalzed coordiantes
            qk1, u_sk1, uk1 = self.update(xk1)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # update converged and updated quantities of previous time step
            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.q_dotk = q_dotk1.copy()
            self.u_dot_sk = u_dot_sk1.copy()
            self.u_sk = u_sk1.copy()

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            U.append(Uk1)
            P_g.append(P_gk1)
            mu_g.append(mu_gk1)
            P_N.append(P_Nk1)
            mu_N.append(mu_Nk1)
            P_F.append(P_Fk1)

            # update local variables for accepted time step
            self.tk = tk1
            self.xk = xk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            U=np.array(U),
            P_g=np.array(P_g),
            mu_g=np.array(mu_g),
            P_N=np.array(P_N),
            mu_N=np.array(mu_N),
            P_F=np.array(P_F),
        )


class MoreauGGLSOlveFor_u_dot_s:
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
        self.nla_g = model.nla_g
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F
        self.nx_s = self.nq + 2 * self.nu + self.nla_g
        self.nx = self.nx_s + 2 * self.nla_N + self.nla_F

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.u_sk = model.u0
        self.Uk = np.zeros_like(self.uk)
        self.la_gk = model.la_g0
        self.P_Nk = dt * model.la_N0
        self.mu_Nk = np.zeros_like(self.P_Nk)
        self.P_Fk = dt * model.la_F0

        # initial velocites
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        h0 = self.model.h(self.tk, self.qk, self.uk)
        W_g0 = self.model.W_g(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_F0 = self.model.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)
        self.u_dot_sk = spsolve(
            M0, h0 + W_g0 @ model.la_g0 + W_N0 @ model.la_N0 + W_F0 @ model.la_F0
        )

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
        self.xk = np.concatenate(
            (
                self.q_dotk,
                self.u_dot_sk,
                self.Uk,
                self.la_gk,
                self.P_Nk,
                self.mu_Nk,
                self.P_Fk,
            )
        )

        # initialize index sets
        self.A_N = np.zeros(self.nla_N, dtype=bool)
        self.B_N = np.zeros(self.nla_N, dtype=bool)
        self.D_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        nla_F = self.nla_F

        q = x[:nq]
        u = x[nq : nq + nu]
        U = x[nq + nu : nq + 2 * nu]
        la_g = x[nq + 2 * nu : nq + 2 * nu + nla_g]
        P_N = x[nx_s : nx_s + nla_N]
        mu_N = x[nx_s + nla_N : nx_s + 2 * nla_N]
        P_F = x[nx_s + 2 * nla_N : nx_s + 2 * nla_N + nla_F]

        return q, u, U, la_g, P_N, mu_N, P_F

    def update(self, xk1):
        dt = self.dt
        q_dotk1, u_dot_sk1, Uk1, la_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

        # implicit Euler
        qk1 = self.qk + dt * q_dotk1
        u_sk1 = self.uk + dt * u_dot_sk1

        # # Theta method
        # qk1 = self.qk + dt * (1.0 - theta) * self.q_dotk + dt * theta * q_dotk1
        # u_sk1 = self.uk + dt * (1.0 - theta) * self.u_dot_sk + dt * theta * u_dot_sk1

        uk1 = u_sk1 + Uk1

        return qk1, u_sk1, uk1

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
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        dt = self.dt
        mu = self.model.mu

        # extract all variables from xk and xk1
        q_dotk1, u_dot_sk1, Uk1, la_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

        # update generalzed coordiantes
        qk1, u_sk1, uk1 = self.update(xk1)

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        gk1 = self.model.g(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_N_qk1 = self.model.g_N_q(tk1, qk1, scipy_matrix=csr_matrix)
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        ###################
        # update index sets
        ###################
        primal_form = True
        # primal_form = False
        if primal_form:
            prox_N_arg_position = g_Nk1 - self.model.prox_r_N * mu_Nk1
            prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        else:
            prox_N_arg_position = -mu_Nk1 + self.model.prox_r_N * g_Nk1
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
        R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, u_sk1) - g_N_qk1.T @ mu_Nk1
        # R[:nq] = (
        #     q_dotk1
        #     - (1.0 - theta) * self.model.q_dot(tk1, qk1, u_sk1)
        #     - theta * self.model.q_dot(self.tk, self.qk, self.u_sk)
        #     - g_N_qk1.T @ mu_Nk1
        # )

        #####################
        # equations of motion
        #####################
        R[nq : nq + nu] = Mk1 @ u_dot_sk1 - self.model.h(tk1, qk1, uk1) - W_gk1 @ la_gk1
        # R[nq : nq + nu] = (
        #     Mk1 @ u_dot_sk1
        #     - (1.0 - theta) * self.model.h(self.tk, self.qk, self.uk)
        #     - theta * self.model.h(tk1, qk1, uk1)
        #     - (1.0 - theta) * self.model.W_g(self.tk, self.qk) @ self.la_gk
        #     - theta * W_gk1 @ la_gk1
        # )

        #################
        # impact equation
        #################
        R[nq + nu : nq + 2 * nu] = Mk1 @ Uk1 - W_Nk1 @ P_Nk1 - W_Fk1 @ P_Fk1

        #######################################################
        # bilateral constraints on position and velocitiy level
        #######################################################
        R[nq + 2 * nu : nq + 2 * nu + nla_g] = gk1

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
        #     R[nq + nu + nla_N :] = -mu_Nk1 - prox_R0_nm(prox_N_arg_position)

        R[nx_s + nla_N + A_N_ind] = g_Nk1[A_N]
        R[nx_s + nla_N + _A_N_ind] = mu_Nk1[_A_N]

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

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        # U = [self.Uk]
        la_g = [self.la_gk]
        # P_gamma = [self.P_gammak]
        P_N = [self.P_Nk]
        mu_N = [self.mu_Nk]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            xk1 = self.xk.copy()  # This copy is mandatory since we modify xk1
            # in the step function

            converged, n_iter, error, xk1 = self.step(tk1, xk1)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )
            if not converged:
                raise RuntimeError(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )

            # extract all variables from xk and xk1
            q_dotk1, u_dot_sk1, Uk1, la_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

            # update generalzed coordiantes
            # qk1 = self.qk + self.dt * q_dotk1
            # u_sk1 = self.uk + self.dt * u_s_dotk1
            # uk1 = u_sk1 + Uk1
            qk1, u_sk1, uk1 = self.update(xk1)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # update converged and updated quantities of previous time step
            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.q_dotk = q_dotk1.copy()
            self.u_dot_sk = u_dot_sk1.copy()
            self.u_sk = u_sk1.copy()
            self.la_gk = la_gk1.copy()

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            la_g.append(la_gk1)
            P_N.append(P_Nk1)
            mu_N.append(mu_Nk1)

            # update local variables for accepted time step
            self.tk = tk1
            self.xk = xk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            P_g=np.array(la_g),
            P_N=np.array(P_N),
            mu_N=np.array(mu_N),
        )


class MoreauGGLInvertM:
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
        self.nla_g = model.nla_g
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F
        self.nx_s = self.nq + self.nu + self.nla_g
        self.nx = self.nx_s + 2 * self.nla_N + self.nla_F

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.la_gk = model.la_g0
        self.P_Nk = dt * model.la_N0
        self.mu_Nk = np.zeros_like(self.P_Nk)
        self.P_Fk = dt * model.la_F0

        # initial velocites
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        h0 = self.model.h(self.tk, self.qk, self.uk)
        W_g0 = self.model.W_g(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_F0 = self.model.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)
        self.u_dotk = spsolve(
            M0, h0 + W_g0 @ model.la_g0 + W_N0 @ model.la_N0 + W_F0 @ model.la_F0
        )

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
        if use_position_formulation:
            self.xk = np.concatenate(
                (
                    self.qk,
                    self.uk,
                    self.la_gk,
                    self.P_Nk,
                    self.mu_Nk,
                    self.P_Fk,
                )
            )
        else:
            self.xk = np.concatenate(
                (
                    self.q_dotk,
                    self.u_dotk,
                    self.la_gk,
                    self.P_Nk,
                    self.mu_Nk,
                    self.P_Fk,
                )
            )

        # initialize index sets
        self.A_N = np.zeros(self.nla_N, dtype=bool)
        self.B_N = np.zeros(self.nla_N, dtype=bool)
        # self.Ck1 = np.zeros(self.nla_N, dtype=bool)
        self.D_st = np.zeros(self.nla_N, dtype=bool)
        # self.Ek1_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        nla_F = self.nla_F

        q = x[:nq]
        u = x[nq : nq + nu]
        la_g = x[nq + nu : nq + nu + nla_g]
        P_N = x[nx_s : nx_s + nla_N]
        mu_N = x[nx_s + nla_N : nx_s + 2 * nla_N]
        P_F = x[nx_s + 2 * nla_N : nx_s + 2 * nla_N + nla_F]

        return q, u, la_g, P_N, mu_N, P_F

    def update_position_formulation(self, xk1):
        dt = self.dt
        qk1, uk1, la_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
        q_dotk1 = (qk1 - self.qk) / dt
        u_dotk1 = (uk1 - self.uk) / dt
        return q_dotk1, u_dotk1

    def update_velocity_formulation(self, xk1):
        raise NotImplementedError
        dt = self.dt

        q_dotk1, u_dotk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
        qk1 = self.qk + dt * q_dotk1
        uk1 = self.uk + dt * u_dotk1

        # # Newmark method
        # # TODO: Higher order interpolation schemes are not working when we are
        # # mixing Signorini's law and the impact equation. The explanation might
        # # be that we also interpoalte the velcity jump which detroys the impact
        # # law! Thus, we have to distinguish between small and capital lambdas,
        # # i.e., percistent and impulsive contat forces.
        # # beta = 0.5
        # beta = 0.9
        # # beta = 1.0
        # q_dot_bar = (1.0 - beta) * self.q_dotk + beta * q_dotk1
        # u_dot_bar = (1.0 - beta) * self.u_dotk + beta * u_dotk1
        # # P_N_bar = (1.0 - beta) * self.P_Nk + beta * P_Nk1
        # # mu_N_bar = (1.0 - beta) * self.mu_Nk + beta * mu_Nk1
        # # P_F_bar = (1.0 - beta) * self.P_Fk + beta * P_Fk1
        # qk1 = self.qk + dt * q_dot_bar
        # uk1 = self.uk + dt * u_dot_bar

        return qk1, uk1
        # return qk1, uk1, P_N_bar, mu_N_bar, P_F_bar

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
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        dt = self.dt
        mu = self.model.mu

        # extract all variables from xk and xk1
        if use_position_formulation:
            qk1, uk1, la_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
            q_dotk1, u_dotk1 = self.update_position_formulation(xk1)
        else:
            q_dotk1, u_dotk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
            qk1, uk1 = self.update_velocity_formulation(xk1)
            # qk1, uk1, P_N_bar, mu_N_bar, P_F_bar = self.update_velocity_formulation(xk1)
            # P_Nk1, mu_Nk1, P_Fk1 = P_N_bar, mu_N_bar, P_F_bar

        # compute integrated mu as done in gen alpha
        mu_hat_Nk1 = mu_Nk1  # TODO: This is not working!
        # mu_hat_Nk1 = mu_Nk1 + self.dt * P_Nk1  # TODO: This is the key ingredient!

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        gk1 = self.model.g(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_N_qk1 = self.model.g_N_q(tk1, qk1, scipy_matrix=csr_matrix)
        # gamma_F_qk1 = self.model.gamma_F_q(tk1, qk1, uk1, scipy_matrix=csr_matrix)
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

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
        # TODO: I think it is better to solve for u_dot_sk1 using a separate
        # unknown and having the velocity jump U as a further unknown like we
        # do in the generalized alpha method.
        u_dot_s_k1 = spsolve(Mk1.tocsr(), self.model.h(tk1, qk1, uk1) + W_gk1 @ la_gk1)
        u_sk1 = self.uk + dt * u_dot_s_k1
        R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, u_sk1) - g_N_qk1.T @ mu_Nk1

        #####################
        # equations of motion
        #####################
        R[nq : nq + nu] = (
            Mk1 @ u_dotk1
            - self.model.h(tk1, qk1, uk1)
            - W_gk1 @ la_gk1
            - W_Nk1 @ (P_Nk1 / dt)
            - W_Fk1 @ (P_Fk1 / dt)
        )

        #######################################################
        # bilateral constraints on position and velocitiy level
        #######################################################
        R[nq + nu : nq + nu + nla_g] = gk1

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

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        la_g = [self.la_gk]
        # P_gamma = [self.P_gammak]
        P_N = [self.P_Nk]
        mu_N = [self.mu_Nk]
        P_F = [self.P_Fk]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            xk1 = self.xk.copy()  # This copy is mandatory since we modify xk1
            # in the step function

            converged, n_iter, error, xk1 = self.step(tk1, xk1)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )
            if not converged:
                raise RuntimeError(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )

            # extract all variables from xk and xk1
            if use_position_formulation:
                qk1, uk1, la_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
                q_dotk1, u_dotk1 = self.update_position_formulation(xk1)
            else:
                q_dotk1, u_dotk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
                qk1, uk1 = self.update_velocity_formulation(xk1)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # update converged and updated quantities of previous time step
            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.q_dotk = q_dotk1.copy()
            self.u_dotk = u_dotk1.copy()

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            la_g.append(la_gk1)
            P_N.append(P_Nk1)
            mu_N.append(mu_Nk1)
            P_F.append(P_Fk1)

            # update local variables for accepted time step
            self.tk = tk1
            self.xk = xk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            P_g=np.array(la_g),
            P_N=np.array(P_N),
            mu_N=np.array(mu_N),
            P_F=np.array(P_F),
        )


class MoreauGGLWorkingSolution:
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
        self.nla_g = model.nla_g
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F
        self.nx_s = self.nq + self.nu + 2 * self.nla_g
        self.nx = self.nx_s + 2 * self.nla_N + self.nla_F

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.P_gk = dt * model.la_g0
        self.mu_gk = np.zeros_like(self.P_gk)
        self.P_Nk = dt * model.la_N0
        self.mu_Nk = np.zeros_like(self.P_Nk)
        self.P_Fk = dt * model.la_F0

        # initial velocites
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        h0 = self.model.h(self.tk, self.qk, self.uk)
        W_g0 = self.model.W_g(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_F0 = self.model.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)
        self.u_dotk = spsolve(
            M0, h0 + W_g0 @ self.P_gk + W_N0 @ self.P_Nk + W_F0 @ self.P_Fk
        )

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
        if use_position_formulation:
            self.xk = np.concatenate(
                (
                    self.qk,
                    self.uk,
                    self.P_gk,
                    self.mu_gk,
                    self.P_Nk,
                    self.mu_Nk,
                    self.P_Fk,
                )
            )
        else:
            self.xk = np.concatenate(
                (
                    self.q_dotk,
                    self.u_dotk,
                    self.P_gk,
                    self.mu_gk,
                    self.P_Nk,
                    self.mu_Nk,
                    self.P_Fk,
                )
            )

        # initialize index sets
        self.A_N = np.zeros(self.nla_N, dtype=bool)
        self.B_N = np.zeros(self.nla_N, dtype=bool)
        # self.Ck1 = np.zeros(self.nla_N, dtype=bool)
        self.D_st = np.zeros(self.nla_N, dtype=bool)
        # self.Ek1_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        nla_F = self.nla_F

        q = x[:nq]
        u = x[nq : nq + nu]
        P_g = x[nq + nu : nq + nu + nla_g]
        mu_g = x[nq + nu + nla_g : nq + nu + 2 * nla_g]
        P_N = x[nx_s : nx_s + nla_N]
        mu_N = x[nx_s + nla_N : nx_s + 2 * nla_N]
        P_F = x[nx_s + 2 * nla_N : nx_s + 2 * nla_N + nla_F]

        return q, u, P_g, mu_g, P_N, mu_N, P_F

    def update_position_formulation(self, xk1):
        dt = self.dt
        qk1, uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
        q_dotk1 = (qk1 - self.qk) / dt
        u_dotk1 = (uk1 - self.uk) / dt
        return q_dotk1, u_dotk1

    def update_velocity_formulation(self, xk1):
        raise NotImplementedError
        dt = self.dt

        q_dotk1, u_dotk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
        qk1 = self.qk + dt * q_dotk1
        uk1 = self.uk + dt * u_dotk1

        # # Newmark method
        # # TODO: Higher order interpolation schemes are not working when we are
        # # mixing Signorini's law and the impact equation. The explanation might
        # # be that we also interpoalte the velcity jump which detroys the impact
        # # law! Thus, we have to distinguish between small and capital lambdas,
        # # i.e., percistent and impulsive contat forces.
        # # beta = 0.5
        # beta = 0.9
        # # beta = 1.0
        # q_dot_bar = (1.0 - beta) * self.q_dotk + beta * q_dotk1
        # u_dot_bar = (1.0 - beta) * self.u_dotk + beta * u_dotk1
        # # P_N_bar = (1.0 - beta) * self.P_Nk + beta * P_Nk1
        # # mu_N_bar = (1.0 - beta) * self.mu_Nk + beta * mu_Nk1
        # # P_F_bar = (1.0 - beta) * self.P_Fk + beta * P_Fk1
        # qk1 = self.qk + dt * q_dot_bar
        # uk1 = self.uk + dt * u_dot_bar

        return qk1, uk1
        # return qk1, uk1, P_N_bar, mu_N_bar, P_F_bar

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
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        dt = self.dt
        mu = self.model.mu

        # extract all variables from xk and xk1
        if use_position_formulation:
            qk1, uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
            q_dotk1, u_dotk1 = self.update_position_formulation(xk1)
        else:
            q_dotk1, u_dotk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
            qk1, uk1 = self.update_velocity_formulation(xk1)
            # qk1, uk1, P_N_bar, mu_N_bar, P_F_bar = self.update_velocity_formulation(xk1)
            # P_Nk1, mu_Nk1, P_Fk1 = P_N_bar, mu_N_bar, P_F_bar

        # compute integrated mu as done in gen alpha
        mu_hat_Nk1 = mu_Nk1  # TODO: This is not working!
        # mu_hat_Nk1 = mu_Nk1 + self.dt * P_Nk1  # TODO: This is the key ingredient!

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        gk1 = self.model.g(tk1, qk1)
        g_dotk1 = self.model.g_dot(tk1, qk1, uk1)  # TODO: Smooth velocity only?
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        g_qk1 = self.model.g_q(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_N_qk1 = self.model.g_N_q(tk1, qk1, scipy_matrix=csr_matrix)
        # gamma_F_qk1 = self.model.gamma_F_q(tk1, qk1, uk1, scipy_matrix=csr_matrix)
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

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
        u_dot_s_k1 = spsolve(Mk1.tocsr(), self.model.h(tk1, qk1, uk1))
        u_sk1 = self.uk + dt * u_dot_s_k1
        # u_sk1 = self.uk # TODO: This is not sufficient!
        R[:nq] = (
            q_dotk1
            - self.model.q_dot(tk1, qk1, u_sk1)
            - g_qk1.T @ mu_gk1
            - g_N_qk1.T @ mu_Nk1
            # - g_N_qk1.T @ mu_Nk1 / dt
            # - gamma_F_qk1.T @ P_Fk1  # TODO: Not necessary but consistent
        )

        #####################
        # equations of motion
        #####################
        R[nq : nq + nu] = (
            Mk1 @ u_dotk1
            - self.model.h(tk1, qk1, uk1)
            - W_gk1 @ (P_gk1 / dt)
            - W_Nk1 @ (P_Nk1 / dt)
            - W_Fk1 @ (P_Fk1 / dt)
        )

        #######################################################
        # bilateral constraints on position and velocitiy level
        #######################################################
        R[nq + nu : nq + nu + nla_g] = g_dotk1
        R[nq + nu + nla_g : nq + nu + 2 * nla_g] = gk1

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

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        P_g = [self.P_gk]
        mu_g = [self.mu_gk]
        # P_gamma = [self.P_gammak]
        P_N = [self.P_Nk]
        mu_N = [self.mu_Nk]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            xk1 = self.xk.copy()  # This copy is mandatory since we modify xk1
            # in the step function

            converged, n_iter, error, xk1 = self.step(tk1, xk1)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )
            if not converged:
                raise RuntimeError(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )

            # extract all variables from xk and xk1
            if use_position_formulation:
                qk1, uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
                q_dotk1, u_dotk1 = self.update_position_formulation(xk1)
            else:
                q_dotk1, u_dotk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
                qk1, uk1 = self.update_velocity_formulation(xk1)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # update converged and updated quantities of previous time step
            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.q_dotk = q_dotk1.copy()
            self.u_dotk = u_dotk1.copy()

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            P_g.append(P_gk1)
            mu_g.append(mu_gk1)
            P_N.append(P_Nk1)
            mu_N.append(mu_Nk1)

            # update local variables for accepted time step
            self.tk = tk1
            self.xk = xk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            P_g=np.array(P_g),
            mu_g=np.array(mu_g),
            P_N=np.array(P_N),
            mu_N=np.array(mu_N),
        )


class MoreauGGLUNilateralOnly:
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
        if use_position_formulation:
            self.xk = np.concatenate(
                (self.qk, self.uk, self.P_Nk, self.mu_Nk, self.P_Fk)
            )
        else:
            self.xk = np.concatenate(
                (self.q_dotk, self.u_dotk, self.P_Nk, self.mu_Nk, self.P_Fk)
            )

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

    def update_position_formulation(self, xk1):
        dt = self.dt
        qk1, uk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
        q_dotk1 = (qk1 - self.qk) / dt
        u_dotk1 = (uk1 - self.uk) / dt
        return q_dotk1, u_dotk1

    def update_velocity_formulation(self, xk1):
        dt = self.dt

        q_dotk1, u_dotk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
        qk1 = self.qk + dt * q_dotk1
        uk1 = self.uk + dt * u_dotk1

        # # Newmark method
        # # TODO: Higher order interpolation schemes are not working when we are
        # # mixing Signorini's law and the impact equation. The explanation might
        # # be that we also interpoalte the velcity jump which detroys the impact
        # # law! Thus, we have to distinguish between small and capital lambdas,
        # # i.e., percistent and impulsive contat forces.
        # # beta = 0.5
        # beta = 0.9
        # # beta = 1.0
        # q_dot_bar = (1.0 - beta) * self.q_dotk + beta * q_dotk1
        # u_dot_bar = (1.0 - beta) * self.u_dotk + beta * u_dotk1
        # # P_N_bar = (1.0 - beta) * self.P_Nk + beta * P_Nk1
        # # mu_N_bar = (1.0 - beta) * self.mu_Nk + beta * mu_Nk1
        # # P_F_bar = (1.0 - beta) * self.P_Fk + beta * P_Fk1
        # qk1 = self.qk + dt * q_dot_bar
        # uk1 = self.uk + dt * u_dot_bar

        return qk1, uk1
        # return qk1, uk1, P_N_bar, mu_N_bar, P_F_bar

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
        dt = self.dt
        mu = self.model.mu

        # extract all variables from xk and xk1
        if use_position_formulation:
            qk1, uk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
            q_dotk1, u_dotk1 = self.update_position_formulation(xk1)
        else:
            q_dotk1, u_dotk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
            qk1, uk1 = self.update_velocity_formulation(xk1)
            # qk1, uk1, P_N_bar, mu_N_bar, P_F_bar = self.update_velocity_formulation(xk1)
            # P_Nk1, mu_Nk1, P_Fk1 = P_N_bar, mu_N_bar, P_F_bar

        # compute integrated mu as done in gen alpha
        mu_hat_Nk1 = mu_Nk1  # TODO: This is not working!
        # mu_hat_Nk1 = mu_Nk1 + self.dt * P_Nk1  # TODO: This is the key ingredient!

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_N_qk1 = self.model.g_N_q(tk1, qk1, scipy_matrix=csr_matrix)
        gamma_F_qk1 = self.model.gamma_F_q(tk1, qk1, uk1, scipy_matrix=csr_matrix)
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

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
        # R[:nq] = (
        #     q_dotk1
        #     - self.model.q_dot(tk1, qk1, uk1)
        #     - g_N_qk1.T @ mu_Nk1 / dt
        #     - gamma_F_qk1.T @ P_Fk1  # TODO: Not necessary but consistent
        # )

        u_dot_s_k1 = spsolve(Mk1.tocsr(), self.model.h(tk1, qk1, uk1))
        u_sk1 = self.uk + dt * u_dot_s_k1
        # u_sk1 = self.uk # TODO: This is not sufficient!
        R[:nq] = (
            q_dotk1
            - self.model.q_dot(tk1, qk1, u_sk1)
            - g_N_qk1.T @ mu_Nk1
            # - g_N_qk1.T @ mu_Nk1 / dt
            # - gamma_F_qk1.T @ P_Fk1  # TODO: Not necessary but consistent
        )
        # TODO: Since bilateral constraints on position level should not be
        # influenced by the velocity jumps we might be able to add them on
        # position level only!

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

            converged, n_iter, error, xk1 = self.step(tk1, xk1)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )
            if not converged:
                raise RuntimeError(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )

            # extract all variables from xk and xk1
            if use_position_formulation:
                qk1, uk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
                q_dotk1, u_dotk1 = self.update_position_formulation(xk1)
            else:
                q_dotk1, u_dotk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)
                qk1, uk1 = self.update_velocity_formulation(xk1)
                # qk1, uk1, P_N_bar, mu_N_bar, P_F_bar = self.update_velocity_formulation(xk1)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # update converged and updated quantities of previous time step
            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.q_dotk = q_dotk1.copy()
            self.u_dotk = u_dotk1.copy()
            # self.P_Nk = P_Nk1.copy()
            # self.mu_Nk = mu_Nk1.copy()
            # self.P_Fk = P_Fk1.copy()

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
