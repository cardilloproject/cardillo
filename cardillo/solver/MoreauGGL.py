import numpy as np

# np.seterr(all="raise")
# import warnings

# warnings.filterwarnings("error")

from scipy.sparse.linalg import spsolve, lsqr, LinearOperator
from scipy.sparse import csr_matrix, bmat, eye
from tqdm import tqdm

from cardillo.solver import Solution
from cardillo.math import norm, prox_R0_nm, prox_R0_np, prox_sphere, approx_fprime

use_position_formulation = True
# use_position_formulation = False


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
        # TODO: This seems not to be important!
        # kappa_Nk1 = mu_Nk1 + dt * P_Nk1
        kappa_Nk1 = mu_Nk1

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
            prox_N_arg_position = g_Nk1 - self.model.prox_r_N * kappa_Nk1
            prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        else:
            prox_N_arg_position = -kappa_Nk1 + self.model.prox_r_N * g_Nk1
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
        R[nq : nq + nu] = Mk1 @ u_dot_sk1 - self.model.h(tk1, qk1, uk1)
        # R[nq : nq + nu] = Mk1 @ u_dot_sk1 - self.model.h(tk1, qk1, u_sk1)

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
        #     R[nq + nu + nla_N :] = -kappa_Nk1 - prox_R0_nm(prox_N_arg_position)

        R[nx_s + nla_N + A_N_ind] = g_Nk1[A_N]
        R[nx_s + nla_N + _A_N_ind] = kappa_Nk1[_A_N]

        ##########
        # friction
        ##########

        # # TODO: No friction case can be implemented like this:
        # R[nx_s + 2 * nla_N :] = P_Fk1

        D_st = self.D_st

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


# TODO:
# - Implement fixed-point iteration case for redundant contacts.
# - Investigate/ discuss with Remco if least squares solution is more
#   meaningfull since it choses "the best solution" in some norm. What
#   is its physical interpretation?
# - Implement Newmark method with explicit kinematic equation.
# - Investigate preconditioning by making all residual values of the same
#   order in h. Otherwise we get problems when solving the least squares
#   solution later du to the twice as bad vondition number.
class NonsmoothEulerBackwardsGGL_V2:
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
    ):
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
        self.u_dotk = spsolve(
            M0, h0 + W_g0 @ model.la_g0 + W_N0 @ model.la_N0 + W_F0 @ model.la_F0
        )

        # auxiliary quantities for generalized alpha method
        self.u_sk = self.uk.copy()
        self.vk = self.q_dotk.copy()
        self.ak = self.u_dotk.copy()

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
                self.u_dotk,
                self.Uk,
                self.P_gk,
                self.mu_gk,
                self.P_Nk,
                self.mu_Nk,
                self.P_Fk,
            )
        )

        # initialize index sets
        self.Ak1 = np.zeros(self.nla_N, dtype=bool)
        self.Bk1 = np.zeros(self.nla_N, dtype=bool)
        self.Dk1_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        nla_F = self.nla_F

        q_dot = x[:nq]
        u_dot = x[nq : nq + nu]
        U = x[nq + nu : nq + 2 * nu]
        P_g = x[nq + 2 * nu : nq + 2 * nu + nla_g]
        mu_g = x[nq + 2 * nu + nla_g : nq + 2 * nu + 2 * nla_g]
        P_N = x[nx_s : nx_s + nla_N]
        mu_N = x[nx_s + nla_N : nx_s + 2 * nla_N]
        P_F = x[nx_s + 2 * nla_N : nx_s + 2 * nla_N + nla_F]

        return q_dot, u_dot, U, P_g, mu_g, P_N, mu_N, P_F

    def update(self, xk1, store=False):
        dt = self.dt
        q_dotk1, u_dotk1, Uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

        ################
        # implicit Euler
        ################
        qk1 = self.qk + dt * q_dotk1
        uk1 = self.uk + dt * u_dotk1 + Uk1

        # # #################
        # # trapezoidal rule
        # # #################
        # qk1 = self.qk + 0.5 * dt * (self.q_dotk + q_dotk1)
        # u_sk1 = self.uk + 0.5 * dt * (self.u_dotk + u_dotk1)

        # ##############
        # # theta method
        # ##############
        # theta = 0.5
        # qk1 = self.qk + dt * (1.0 - theta) * self.q_dotk + dt * theta * q_dotk1
        # uk1 = self.uk + dt * (1.0 - theta) * self.u_dotk + dt * theta * u_dotk1 + Uk1

        # #################################################################
        # # generalized alpha method for first order differential equations
        # #################################################################

        # # constants
        # # rho_inf = 1.0
        # rho_inf = 0.8
        # # rho_inf = 0.1
        # # rho_inf = 0.0
        # alpha_m = (3.0 * rho_inf - 1.0) / (2.0 * (rho_inf + 1.0))
        # alpha_f = rho_inf / (rho_inf + 1.0)
        # gamma = 0.5 + alpha_f - alpha_m

        # # update auxiliary quantities
        # vk1 = (
        #     alpha_f * self.q_dotk + (1.0 - alpha_f) * q_dotk1 - alpha_m * self.vk
        # ) / (1.0 - alpha_m)
        # ak1 = (
        #     alpha_f * self.u_dotk + (1.0 - alpha_f) * u_dotk1 - alpha_m * self.ak
        # ) / (1.0 - alpha_m)

        # qk1 = self.qk + dt * ((1.0 - gamma) * self.vk + gamma * vk1)
        # uk1 = self.uk + dt * ((1.0 - gamma) * self.ak + gamma * ak1) + Uk1
        # if store:
        #     self.qk = qk1.copy()
        #     self.uk = uk1.copy()
        #     self.q_dotk = q_dotk1.copy()
        #     self.u_dotk = u_dotk1.copy()
        #     self.vk = vk1.copy()
        #     self.ak = ak1.copy()

        return qk1, uk1

    def R(self, tk1, xk1, update_index_set=False, primal_form=True):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        dt = self.dt
        mu = self.model.mu

        # extract all variables from xk1
        q_dotk1, u_dotk1, Uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

        # update generalzed coordiantes
        qk1, uk1 = self.update(xk1)

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
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        ###################
        # evaluate residual
        ###################
        R = np.zeros(self.nx)

        ###################################
        # kinematic equation
        ###################################
        R[:nq] = (
            q_dotk1
            - self.model.q_dot(tk1, qk1, uk1 - Uk1)
            # - self.model.q_dot(tk1, qk1, self.uk)
            # - self.model.q_dot(tk1, qk1, uk1)
            - g_qk1.T @ mu_gk1
            - g_N_qk1.T @ mu_Nk1
        ) * dt

        #####################
        # equations of motion
        #####################
        R[nq : nq + nu] = (Mk1 @ u_dotk1 - self.model.h(tk1, qk1, uk1)) * dt

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

        ###################
        # update index sets
        ###################
        if primal_form:
            prox_N_arg_position = g_Nk1 - self.model.prox_r_N * mu_Nk1
            prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        else:
            prox_N_arg_position = -mu_Nk1 + self.model.prox_r_N * g_Nk1
            prox_N_arg_velocity = -P_Nk1 + self.model.prox_r_N * xi_Nk1

        if update_index_set:
            # implicit index set
            self.Ak1 = prox_N_arg_position <= 0
            # self.Ak1 = g_Nk1 <= 0 # TODO: Is this better?

            # # secplicit forecas like Moreau
            # # TODO: This yields chattering for 1D bouncing ball
            # q_M = self.qk + 0.5 * dt * self.model.q_dot(self.tk, self.qk, self.uk)
            # g_N_M = self.model.g_N(self.tk, q_M)
            # self.Ak1 = g_N_M <= 0

        #################################################
        # Mixed Signorini on velcity level and impact law
        #################################################
        if primal_form:
            R[nx_s : nx_s + nla_N] = np.where(
                self.Ak1, xi_Nk1 - prox_R0_np(prox_N_arg_velocity), P_Nk1
            )
        else:
            R[nx_s : nx_s + nla_N] = np.where(
                self.Ak1, -P_Nk1 - prox_R0_nm(prox_N_arg_velocity), P_Nk1
            )

        ########################
        # position stabilization
        ########################
        if primal_form:
            R[nx_s + nla_N : nx_s + 2 * nla_N] = g_Nk1 - prox_R0_np(prox_N_arg_position)

        else:
            R[nx_s + nla_N : nx_s + 2 * nla_N] = -mu_Nk1 - prox_R0_nm(
                prox_N_arg_position
            )

        ##########
        # friction
        ##########
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                # TODO: Is there a primal/ dual form?
                R[nx_s + 2 * nla_N + i_F] = np.where(
                    self.Ak1[i_N] * np.ones(len(i_F), dtype=bool),
                    -P_Fk1[i_F]
                    - prox_sphere(
                        -P_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * P_Nk1[i_N],
                    ),
                    P_Fk1[i_F],
                )

        return R

    def step(self, tk1, xk1):
        # initial residual and error
        R = self.R(tk1, xk1, update_index_set=True)
        error = self.error_function(R)
        converged = error < self.tol
        j = 0
        if not converged:
            while j < self.max_iter:
                # jacobian
                J = csr_matrix(
                    approx_fprime(
                        xk1,
                        lambda x: self.R(tk1, x, update_index_set=False),
                        method="2-point",
                    )
                )

                # Newton update
                j += 1

                # from scipy.linalg import det
                # det_ = det(J.toarray())
                # print(f"det: {det_}")

                dx = spsolve(J, R, use_umfpack=True)

                # LinearOperator()
                # def Ax(x):
                #     """Returns A*x"""
                #     return J @ x

                # def Atb(b):
                #     """Returns A^T*b"""
                #     return J.T @ b

                # # def mv(v):
                # #     return np.array([2*v[0], 3*v[1]])
                # # A = LinearOperator((2,2), matvec=mv)

                # # A = LinearOperator(shape=(self.nx, self.nx), matvec=Ax, rmatvec=Atb)
                # A = LinearOperator(shape=(self.nx, self.nx), matvec=Ax, rmatvec=Atb)
                # dx = lsqr(A, R)[0]

                # # guard against rank deficiency
                # # TODO: Why we get underflow errors of the sparse solvers?
                # # dx = lsqr(J, R, atol=1.0e-12, btol=1.0e-12, show=True)[0]
                # dx = lsqr(J, R, atol=1.0e-12, btol=1.0e-12, show=False)[0]
                # # dx = lsqr(J, R, show=True)[0]
                # # dx = lsqr(J, R)[0]
                # # from scipy.sparse.linalg import lsmr
                # # dx = lsmr(J, R)[0]

                # # no underflow errors
                # dx = np.linalg.lstsq(J.toarray(), R, rcond=None)[0]

                # # TODO: Can we get this sparse?
                # # using QR decomposition, see https://de.wikipedia.org/wiki/QR-Zerlegung#L%C3%B6sung_regul%C3%A4rer_oder_%C3%BCberbestimmter_Gleichungssysteme
                # b = R.copy()
                # Q, R = np.linalg.qr(J.toarray())
                # z = Q.T @ b
                # dx = np.linalg.solve(R, z)  # solving R*x = Q^T*b

                # # solve normal equation (should be independent of the conditioning
                # # number!)
                # dx = spsolve(J.T @ J, J.T @ R)

                # try:
                #     # dx = spsolve(J, R, use_umfpack=True)
                #     dx = spsolve(J, R, use_umfpack=False)
                # except:
                #     print(f"lsqr case")
                #     # dx = lsqr(J, R)[0]
                #     dx = lsqr(J, R, atol=1.0e-10, btol=1.0e-10)[0]
                # # except np.linalg.LinAlgError as err:
                # #     if 'Singular matrix' in str(err):
                # #         print(f"lsqr case")
                # #         # TODO: Is it beneficial to initialize with the difference of the last step?
                # #         # dx = lsqr(J, R, x0=xk1 - xk)[0]
                # #         # dx = lsqr(J, R)[0]
                # #         dx = lsqr(J, R, atol=1.0e-8, btol=1.0e-8)[0]
                # #     else:
                # #         raise RuntimeError("Unexpected problem occurred when inverting the Jacobian.")

                xk1 -= dx
                R = self.R(tk1, xk1, update_index_set=True)

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
            (
                q_dotk1,
                u_dot_sk1,
                Uk1,
                P_gk1,
                mu_gk1,
                P_Nk1,
                mu_Nk1,
                P_Fk1,
            ) = self.unpack(xk1)

            # update generalzed coordiantes
            qk1, uk1 = self.update(xk1, store=True)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # update converged and updated quantities of previous time step
            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.q_dotk = q_dotk1.copy()
            self.u_dotk = u_dot_sk1.copy()
            self.Uk = Uk1.copy()
            self.P_Nk = P_Nk1.copy()
            self.P_gk = P_gk1.copy()

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


class NonsmoothNewmarkGGL:
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
    ):
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
        self.nx_s = 2 * self.nu + 2 * self.nla_g
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
        self.u_dotk = spsolve(
            M0, h0 + W_g0 @ model.la_g0 + W_N0 @ model.la_N0 + W_F0 @ model.la_F0
        )

        #######################################################################
        # starting values for generalized state vector, its derivatives and
        # auxiliary velocities
        #######################################################################
        self.xk = np.concatenate(
            (
                self.u_dotk,
                self.Uk,
                self.P_gk,
                self.mu_gk,
                self.P_Nk,
                self.mu_Nk,
                self.P_Fk,
            )
        )

        # initialize index sets
        self.Ak1 = np.zeros(self.nla_N, dtype=bool)
        self.Bk1 = np.zeros(self.nla_N, dtype=bool)
        self.Dk1_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nu = self.nu
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        nla_F = self.nla_F

        u_dot = x[:nu]
        U = x[nu : 2 * nu]
        P_g = x[2 * nu : 2 * nu + nla_g]
        mu_g = x[2 * nu + nla_g : 2 * nu + 2 * nla_g]
        P_N = x[nx_s : nx_s + nla_N]
        mu_N = x[nx_s + nla_N : nx_s + 2 * nla_N]
        P_F = x[nx_s + 2 * nla_N : nx_s + 2 * nla_N + nla_F]

        return u_dot, U, P_g, mu_g, P_N, mu_N, P_F

    def update(self, xk1):
        dt = self.dt
        u_dotk1, Uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

        #################################################################
        # Newmark, see
        # https://de.wikipedia.org/wiki/Newmark-beta-Verfahren#Herleitung
        #################################################################
        # gamma = 0.5
        # beta = 1.0 / 6.0
        # u_sk1 = self.uk + dt * (1.0 - gamma) * self.u_dotk + dt * gamma * u_dotk1
        # Delta_u_k1 = self.uk + dt * (0.5 - beta) * self.u_dotk + dt * beta * u_dotk1

        # mu_Nk1 = dt * (1.0 - gamma) * self.mu_Nk + dt * gamma * mu_Nk1
        # mu_gk1 = dt * (1.0 - gamma) * self.mu_gk + dt * gamma * mu_gk1

        # gamma = 1, beta = 0.5
        u_sk1 = self.uk + dt * u_dotk1
        Delta_u_k1 = self.uk + 0.5 * dt * u_dotk1  # + 0.5 * dt * Uk1

        # qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, Delta_u_k1) # + 0.5 * dt * Uk1
        # qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, Delta_u_k1 - Uk1)
        qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, Delta_u_k1)
        uk1 = u_sk1 + Uk1

        return qk1, u_sk1, uk1, mu_gk1, mu_Nk1

    def R(self, tk1, xk1, update_index_set=False):
        nu = self.nu
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        dt = self.dt
        mu = self.model.mu

        # extract all variables from xk1
        u_dotk1, Uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

        # update generalzed coordiantes
        qk1, u_sk1, uk1, mu_gk1, mu_Nk1 = self.update(xk1)

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
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        La_gk1 = P_gk1.copy()
        La_Nk1 = P_Nk1.copy()
        P_gk1 = La_gk1 + dt * mu_gk1.copy()
        P_Nk1 = La_Nk1 + dt * mu_Nk1.copy()

        ###################
        # evaluate residual
        ###################
        R = np.zeros(self.nx)

        #####################
        # equations of motion
        #####################
        R[:nu] = (
            Mk1 @ u_dotk1
            - self.model.h(tk1, qk1, uk1)
            - W_gk1 @ mu_gk1
            - W_Nk1 @ mu_Nk1
        )

        #################
        # impact equation
        #################
        R[nu : 2 * nu] = (
            # Mk1 @ Uk1 - W_gk1 @ La_gk1 - W_Nk1 @ La_Nk1 - W_Fk1 @ P_Fk1
            Mk1 @ Uk1
            - W_gk1 @ P_gk1
            - W_Nk1 @ P_Nk1
            - W_Fk1 @ P_Fk1
        )

        #######################################################
        # bilateral constraints on position and velocitiy level
        #######################################################
        R[2 * nu : 2 * nu + nla_g] = g_dotk1
        # R[2 * nu + nla_g : 2 * nu + 2 * nla_g] = gk1
        R[2 * nu + nla_g : 2 * nu + 2 * nla_g] = mu_gk1

        ###################
        # update index sets
        ###################
        prox_N_arg_position = g_Nk1 - self.model.prox_r_N * mu_Nk1
        prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        # prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * La_Nk1

        if update_index_set:
            # self.Ak1 = g_Nk1 <= 0
            self.Ak1 = prox_N_arg_position <= 0
            # q_M = self.qk + 0.5 * dt * self.model.q_dot(self.tk, self.qk, self.uk)
            # g_N_M = self.model.g_N(self.tk, q_M)
            # self.Ak1 = g_N_M <= 0

        #################################################
        # Mixed Signorini on velcity level and impact law
        #################################################
        R[nx_s : nx_s + nla_N] = np.where(
            self.Ak1,
            xi_Nk1 - prox_R0_np(prox_N_arg_velocity),
            P_Nk1
            # self.Ak1, xi_Nk1 - prox_R0_np(prox_N_arg_velocity), La_Nk1
        )

        ########################
        # position stabilization
        ########################
        R[nx_s + nla_N : nx_s + 2 * nla_N] = g_Nk1 - prox_R0_np(prox_N_arg_position)
        # R[nx_s + nla_N : nx_s + 2 * nla_N] = mu_Nk1

        ##########
        # friction
        ##########
        # no friction!
        R[nx_s + 2 * nla_N :] = P_Fk1

        # for i_N, i_F in enumerate(self.model.NF_connectivity):
        #     i_F = np.array(i_F)

        #     if len(i_F) > 0:
        #         # TODO: Is there a primal/ dual form?
        #         R[nx_s + 2 * nla_N + i_F] = np.where(
        #             self.Ak1[i_N] * np.ones(len(i_F), dtype=bool),
        #             -P_Fk1[i_F]
        #             - prox_sphere(
        #                 -P_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
        #                 mu[i_N] * P_Nk1[i_N],
        #             ),
        #             P_Fk1[i_F],
        #         )

        return R

    def step(self, tk1, xk1):
        # initial residual and error
        R = self.R(tk1, xk1, update_index_set=True)
        error = self.error_function(R)
        converged = error < self.tol
        j = 0
        if not converged:
            while j < self.max_iter:
                # jacobian
                J = csr_matrix(
                    approx_fprime(
                        xk1,
                        lambda x: self.R(tk1, x, update_index_set=False),
                        method="2-point",
                    )
                )

                # Newton update
                j += 1

                # from scipy.linalg import det
                # det_ = det(J.toarray())
                # print(f"det: {det_}")

                # guard against rank deficiency
                dx = spsolve(J, R, use_umfpack=True)
                # # TODO: Why we get underflow errors of the sparse solvers?
                # # dx = lsqr(J, R, atol=1.0e-12, btol=1.0e-12, show=True)[0]
                # # dx = lsqr(J, R, show=True)[0]
                # dx = lsqr(J, R)[0]
                # # from scipy.sparse.linalg import lsmr
                # # dx = lsmr(J, R)[0]

                # # no underflow errors
                # dx = np.linalg.lstsq(J.toarray(), R, rcond=None)[0]

                # # TODO: Can we get this sparse?
                # # using QR decomposition
                # b = R.copy()
                # Q, R = np.linalg.qr(J.toarray())
                # Qb = np.dot(Q.T, b)
                # dx = np.linalg.solve(R, Qb) # solving R*x = Q^T*b

                # # solve normal equation (should be independent of the conditioning
                # # number!)
                # dx = spsolve(J.T @ J, J.T @ R)

                # try:
                #     # dx = spsolve(J, R, use_umfpack=True)
                #     dx = spsolve(J, R, use_umfpack=False)
                # except:
                #     print(f"lsqr case")
                #     # dx = lsqr(J, R)[0]
                #     dx = lsqr(J, R, atol=1.0e-10, btol=1.0e-10)[0]
                # # except np.linalg.LinAlgError as err:
                # #     if 'Singular matrix' in str(err):
                # #         print(f"lsqr case")
                # #         # TODO: Is it beneficial to initialize with the difference of the last step?
                # #         # dx = lsqr(J, R, x0=xk1 - xk)[0]
                # #         # dx = lsqr(J, R)[0]
                # #         dx = lsqr(J, R, atol=1.0e-8, btol=1.0e-8)[0]
                # #     else:
                # #         raise RuntimeError("Unexpected problem occurred when inverting the Jacobian.")

                xk1 -= dx
                R = self.R(tk1, xk1, update_index_set=True)

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
            (
                u_dot_sk1,
                Uk1,
                P_gk1,
                mu_gk1,
                P_Nk1,
                mu_Nk1,
                P_Fk1,
            ) = self.unpack(xk1)

            # update generalzed coordiantes
            qk1, u_sk1, uk1, mu_gk1, mu_Nk1 = self.update(xk1)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # update converged and updated quantities of previous time step
            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.u_dotk = u_dot_sk1.copy()
            self.u_sk = u_sk1.copy()
            self.mu_Nk = mu_Nk1.copy()
            self.mu_gk = mu_gk1.copy()
            self.P_Nk = P_Nk1.copy()
            self.P_gk = P_gk1.copy()

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


class NonsmoothNewmarkGGLDoNotKow:
    def __init__(
        self,
        model,
        t1,
        dt,
        tol=1e-10,
        max_iter=40,
        error_function=lambda x: np.max(np.abs(x)),
    ):
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
        self.nx_s = 2 * self.nu + 2 * self.nla_g
        self.nx = self.nx_s + 2 * self.nla_N + self.nla_F

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
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
        self.xk = np.concatenate(
            (
                self.u_dotk,
                self.Uk,
                self.P_gk,
                self.mu_gk,
                self.P_Nk,
                self.mu_Nk,
                self.P_Fk,
            )
        )

        # initialize index sets
        self.Ak1 = np.zeros(self.nla_N, dtype=bool)
        self.Bk1 = np.zeros(self.nla_N, dtype=bool)
        self.Dk1_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nu = self.nu
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        nla_F = self.nla_F

        u_dot = x[:nu]
        U = x[nu : 2 * nu]
        P_g = x[2 * nu : 2 * nu + nla_g]
        mu_g = x[2 * nu + nla_g : 2 * nu + 2 * nla_g]
        P_N = x[nx_s : nx_s + nla_N]
        mu_N = x[nx_s + nla_N : nx_s + 2 * nla_N]
        P_F = x[nx_s + 2 * nla_N : nx_s + 2 * nla_N + nla_F]

        return u_dot, U, P_g, mu_g, P_N, mu_N, P_F

    def update(self, xk1):
        dt = self.dt
        u_dotk1, Uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

        #################################################################
        # Newmark, see
        # https://de.wikipedia.org/wiki/Newmark-beta-Verfahren#Herleitung
        #################################################################
        gamma = 0.5
        beta = 1.0 / 6.0
        u_sk1 = self.uk + dt * (1.0 - gamma) * self.u_dotk + dt * gamma * u_dotk1
        Delta_u_k1 = self.uk + dt * (0.5 - beta) * self.u_dotk + dt * beta * u_dotk1

        # u_sk1 = self.uk + dt * u_dotk1
        # Delta_u_k1 = self.uk + 0.5 * dt * u_dotk1

        qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, Delta_u_k1)
        uk1 = u_sk1 + Uk1

        return qk1, u_sk1, uk1

    def R(self, tk1, xk1, update_index_set=False, primal_form=True):
        nu = self.nu
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        dt = self.dt
        mu = self.model.mu

        # extract all variables from xk1
        u_dotk1, Uk1, P_gk1, mu_gk1, P_Nk1, mu_Nk1, P_Fk1 = self.unpack(xk1)

        la_gk1 = mu_gk1.copy()
        la_Nk1 = mu_Nk1.copy()
        La_gk1 = P_gk1.copy()
        La_Nk1 = P_Nk1.copy()
        P_gk1 = dt * la_gk1 + La_gk1
        P_Nk1 = dt * la_Nk1 + La_Nk1

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
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        ###################
        # evaluate residual
        ###################
        R = np.zeros(self.nx)

        #####################
        # equations of motion
        #####################
        R[:nu] = (
            Mk1 @ u_dotk1
            - self.model.h(tk1, qk1, uk1)
            - W_gk1 @ la_gk1
            - W_Nk1 @ la_Nk1
        )

        #################
        # impact equation
        #################
        R[nu : 2 * nu] = Mk1 @ Uk1 - W_gk1 @ La_gk1 - W_Nk1 @ La_Nk1 - W_Fk1 @ P_Fk1

        #######################################################
        # bilateral constraints on position and velocitiy level
        #######################################################
        R[2 * nu : 2 * nu + nla_g] = g_dotk1
        R[2 * nu + nla_g : 2 * nu + 2 * nla_g] = gk1

        ###################
        # update index sets
        ###################
        if primal_form:
            prox_N_arg_position = g_Nk1 - self.model.prox_r_N * la_Nk1
            prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        else:
            prox_N_arg_position = -la_Nk1 + self.model.prox_r_N * g_Nk1
            prox_N_arg_velocity = -P_Nk1 + self.model.prox_r_N * xi_Nk1

        if update_index_set:
            self.Ak1 = prox_N_arg_position <= 0

        #################################################
        # Mixed Signorini on velcity level and impact law
        #################################################
        if primal_form:
            R[nx_s : nx_s + nla_N] = np.where(
                self.Ak1, xi_Nk1 - prox_R0_np(prox_N_arg_velocity), P_Nk1
            )
        else:
            R[nx_s : nx_s + nla_N] = np.where(
                self.Ak1, -P_Nk1 - prox_R0_nm(prox_N_arg_velocity), P_Nk1
            )

        ########################
        # position stabilization
        ########################
        if primal_form:
            R[nx_s + nla_N : nx_s + 2 * nla_N] = g_Nk1 - prox_R0_np(prox_N_arg_position)

        else:
            R[nx_s + nla_N : nx_s + 2 * nla_N] = -la_Nk1 - prox_R0_nm(
                prox_N_arg_position
            )

        ##########
        # friction
        ##########
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                # TODO: Is there a primal/ dual form?
                R[nx_s + 2 * nla_N + i_F] = np.where(
                    self.Ak1[i_N] * np.ones(len(i_F), dtype=bool),
                    -P_Fk1[i_F]
                    - prox_sphere(
                        -P_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * P_Nk1[i_N],
                    ),
                    P_Fk1[i_F],
                )

        return R

    def step(self, tk1, xk1):
        # initial residual and error
        R = self.R(tk1, xk1, update_index_set=True)
        error = self.error_function(R)
        converged = error < self.tol
        j = 0
        if not converged:
            while j < self.max_iter:
                # jacobian
                J = csr_matrix(
                    approx_fprime(
                        xk1,
                        lambda x: self.R(tk1, x, update_index_set=False),
                        method="2-point",
                    )
                )

                # Newton update
                j += 1

                # from scipy.linalg import det
                # det_ = det(J.toarray())
                # print(f"det: {det_}")

                # guard against rank deficiency
                dx = spsolve(J, R, use_umfpack=True)
                # TODO: Why we get underflow errors of the sparse solvers?
                # dx = lsqr(J, R, atol=1.0e-12, btol=1.0e-12, show=True)[0]
                # dx = lsqr(J, R, show=True)[0]
                # dx = lsqr(J, R)[0]
                # from scipy.sparse.linalg import lsmr
                # dx = lsmr(J, R)[0]

                # # no underflow errors
                # dx = np.linalg.lstsq(J.toarray(), R, rcond=None)[0]

                # # using QR decomposition
                # b = R.copy()
                # Q, R = np.linalg.qr(J.toarray())
                # Qb = np.dot(Q.T, b)
                # dx = np.linalg.solve(R, Qb) # solving R*x = Q^T*b

                # # solve normal equation (should be independent of the conditioning
                # # number!)
                # # A = J.T @ J
                # # b = J.T @ R
                # # dx = spsolve(A, b)
                # dx = spsolve(J.T @ J, J.T @ R)

                # try:
                #     # dx = spsolve(J, R, use_umfpack=True)
                #     dx = spsolve(J, R, use_umfpack=False)
                # except:
                #     print(f"lsqr case")
                #     # dx = lsqr(J, R)[0]
                #     dx = lsqr(J, R, atol=1.0e-10, btol=1.0e-10)[0]
                # # except np.linalg.LinAlgError as err:
                # #     if 'Singular matrix' in str(err):
                # #         print(f"lsqr case")
                # #         # TODO: Is it beneficial to initialize with the difference of the last step?
                # #         # dx = lsqr(J, R, x0=xk1 - xk)[0]
                # #         # dx = lsqr(J, R)[0]
                # #         dx = lsqr(J, R, atol=1.0e-8, btol=1.0e-8)[0]
                # #     else:
                # #         raise RuntimeError("Unexpected problem occurred when inverting the Jacobian.")

                xk1 -= dx
                R = self.R(tk1, xk1, update_index_set=True)

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
            (
                u_dot_sk1,
                Uk1,
                P_gk1,
                mu_gk1,
                P_Nk1,
                mu_Nk1,
                P_Fk1,
            ) = self.unpack(xk1)

            # update generalzed coordiantes
            qk1, u_sk1, uk1 = self.update(xk1)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # update converged and updated quantities of previous time step
            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.u_dotk = u_dot_sk1.copy()
            self.u_sk = u_sk1.copy()
            self.P_Nk = P_Nk1.copy()
            self.P_gk = P_gk1.copy()

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


class NonsmoothEulerBackwardsGGL_V3:
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
        self.nx_s = 2 * self.nq + 2 * self.nu + 2 * self.nla_g
        self.nx = self.nx_s + 2 * self.nla_N + self.nla_F

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.u_sk = model.u0
        self.Uk = np.zeros(self.nq)
        self.Qk = np.zeros(self.nu)
        self.La_gk = np.zeros(self.nla_g)
        self.la_gk = model.la_g0
        self.La_Nk = np.zeros(self.nla_N)
        self.la_Nk = model.la_N0
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

        #######################################################################
        # starting values for generalized state vector, its derivatives and
        # auxiliary velocities
        #######################################################################
        self.xk = np.concatenate(
            (
                self.q_dotk,
                self.Qk,
                self.u_dot_sk,
                self.Uk,
                self.La_gk,
                self.la_gk,
                self.La_Nk,
                self.la_Nk,
                self.P_Fk,
            )
        )

        # initialize index sets
        self.Ak1 = np.zeros(self.nla_N, dtype=bool)
        self.Bk1 = np.zeros(self.nla_N, dtype=bool)
        self.Dk1_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nx_s = self.nx_s
        nla_N = self.nla_N
        nla_F = self.nla_F

        q = x[:nq]
        Q = x[nq : 2 * nq]
        u = x[2 * nq : 2 * nq + nu]
        U = x[2 * nq + nu : 2 * nq + 2 * nu]
        La_g = x[2 * nq + 2 * nu : 2 * nq + 2 * nu + nla_g]
        la_g = x[2 * nq + 2 * nu + nla_g : 2 * nq + 2 * nu + 2 * nla_g]
        La_N = x[nx_s : nx_s + nla_N]
        la_N = x[nx_s + nla_N : nx_s + 2 * nla_N]
        P_F = x[nx_s + 2 * nla_N : nx_s + 2 * nla_N + nla_F]

        return q, Q, u, U, La_g, la_g, La_N, la_N, P_F

    def update(self, xk1):
        dt = self.dt
        q_dotk1, Qk1, u_dotk1, Uk1, La_gk1, la_gk1, La_Nk1, la_Nk1, P_Fk1 = self.unpack(
            xk1
        )

        # implicit Euler
        q_sk1 = self.qk + dt * q_dotk1
        u_sk1 = self.uk + dt * u_dotk1

        P_gk1 = La_gk1 + dt * la_gk1
        P_Nk1 = La_Nk1 + dt * la_Nk1

        # # TODO: What is wrong here?
        # # midpoint rule
        # q_sk1 = self.qk + 0.5 * dt * (self.q_dotk + q_dotk1)
        # u_sk1 = self.uk + 0.5 * dt * (self.u_dot_sk + u_dotk1)

        # P_gk1 = La_gk1 + 0.5 * dt * (self.la_gk + la_gk1)
        # P_Nk1 = La_Nk1 + 0.5 * dt * (self.la_Nk + la_Nk1)

        # ##############
        # # theta method
        # ##############
        # # theta = 0.75
        # theta = 0.5
        # # theta = 1.0

        # q_sk1 = self.qk + dt * (1.0 - theta) * self.q_dotk + dt * theta * q_dotk1
        # u_sk1 = self.uk + dt * (1.0 - theta) * self.u_dot_sk + dt * theta * u_dotk1

        # P_gk1 = La_gk1 + dt * (1.0 - theta) * self.la_gk + dt * theta * la_gk1
        # P_Nk1 = La_Nk1 + dt * (1.0 - theta) * self.la_Nk + dt * theta * la_Nk1

        qk1 = q_sk1 + Qk1  # + 0.5 * dt * Uk1
        uk1 = u_sk1 + Uk1

        return q_sk1, qk1, u_sk1, uk1, P_gk1, P_Nk1

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
        (
            q_dotk1,
            Qk1,
            u_dot_sk1,
            Uk1,
            La_gk1,
            la_gk1,
            La_Nk1,
            la_Nk1,
            P_Fk1,
        ) = self.unpack(xk1)

        # update generalzed coordiantes
        q_sk1, qk1, u_sk1, uk1, P_gk1, P_Nk1 = self.update(xk1)

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
        # evaluate residual
        ###################
        R = np.zeros(self.nx)

        ####################
        # kinematic equation
        ####################
        R[:nq] = (
            q_dotk1
            - self.model.q_dot(tk1, qk1, uk1 - Uk1)
            # - self.model.q_dot(tk1, qk1, uk1)
        )
        R[nq : 2 * nq] = Qk1

        #####################
        # equations of motion
        #####################
        R[2 * nq : 2 * nq + nu] = (
            Mk1 @ u_dot_sk1
            - self.model.h(tk1, qk1, uk1)
            - W_gk1 @ la_gk1
            - W_Nk1 @ la_Nk1
            # - W_Fk1 @ P_Fk1 * dt # TODO: Is this a good idea?
        )

        #################
        # impact equation
        #################
        R[2 * nq + nu : 2 * nq + 2 * nu] = (
            Mk1 @ Uk1
            - W_gk1 @ La_gk1
            - W_Nk1 @ La_Nk1
            # - W_gk1 @ P_gk1
            # - W_Nk1 @ P_Nk1
            - W_Fk1 @ P_Fk1
        )

        #######################################################
        # bilateral constraints on position and velocitiy level
        #######################################################
        R[2 * nq + 2 * nu : 2 * nq + 2 * nu + nla_g] = g_dotk1
        R[2 * nq + 2 * nu + nla_g : 2 * nq + 2 * nu + 2 * nla_g] = gk1

        ###################
        # update index sets
        ###################
        prox_N_arg_position = g_Nk1 - self.model.prox_r_N * la_Nk1
        prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * La_Nk1
        # prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1

        if update_index_set:
            self.Ak1 = prox_N_arg_position <= 0
            # self.Ak1 = g_Nk1 <= 0

            # q_M = self.qk + 0.5 * dt * self.model.q_dot(self.tk, self.qk, self.uk)
            # g_N_M = self.model.g_N(self.tk, q_M)
            # self.Ak1 = g_N_M <= 0

        #################################################
        # Mixed Signorini on velcity level and impact law
        #################################################
        R[nx_s : nx_s + nla_N] = np.where(
            self.Ak1,
            xi_Nk1 - prox_R0_np(prox_N_arg_velocity),
            La_Nk1,
            # P_Nk1,
        )

        ########################
        # position stabilization
        ########################
        R[nx_s + nla_N : nx_s + 2 * nla_N] = g_Nk1 - prox_R0_np(prox_N_arg_position)

        ##########
        # friction
        ##########
        # # no friction
        # R[nx_s + 2 * nla_N :] = P_Fk1

        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                # TODO: Is there a primal/ dual form?
                R[nx_s + 2 * nla_N + i_F] = np.where(
                    self.Ak1[i_N] * np.ones(len(i_F), dtype=bool),
                    -P_Fk1[i_F]
                    - prox_sphere(
                        -P_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * P_Nk1[i_N],
                    ),
                    P_Fk1[i_F],
                )

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

                # dx = lsqr(R_x, R)[0]

                # # no underflow errors
                # dx = np.linalg.lstsq(R_x.toarray(), R, rcond=None)[0]

                # # using QR decomposition
                # b = R.copy()
                # Q, R = np.linalg.qr(R_x.toarray())
                # Qb = np.dot(Q.T, b)
                # dx = np.linalg.solve(R, Qb) # solving R*x = Q^T*b

                # # solve normal equation (should be independent of the conditioning
                # # number!)
                # dx = spsolve(R_x.T @ R_x, R_x.T @ R)

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
        a = [np.zeros(self.nu)]
        U = [self.Uk]
        la_g = [self.la_gk]
        La_g = [self.La_gk]
        P_g = [self.dt * self.la_gk + self.La_gk]
        # P_gamma = [self.P_gammak]
        la_N = [self.la_Nk]
        La_N = [self.La_Nk]
        P_N = [self.dt * self.la_Nk + self.La_Nk]
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
                    a=np.array(a),
                    U=np.array(U),
                    la_g=np.array(la_g),
                    La_g=np.array(La_g),
                    P_g=np.array(P_g),
                    la_N=np.array(la_N),
                    La_N=np.array(La_N),
                    P_N=np.array(P_N),
                    la_F=np.zeros((len(t), self.nla_F)),
                    La_F=np.zeros((len(t), self.nla_F)),
                    P_F=np.array(P_F),
                )

            # extract all variables from xk and xk1
            (
                q_dotk1,
                Qk1,
                u_dot_sk1,
                Uk1,
                La_gk1,
                la_gk1,
                La_Nk1,
                la_Nk1,
                P_Fk1,
            ) = self.unpack(xk1)

            # update generalzed coordiantes
            q_sk1, qk1, u_sk1, uk1, P_gk1, P_Nk1 = self.update(xk1)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # update converged and updated quantities of previous time step
            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.q_dotk = q_dotk1.copy()
            self.u_dot_sk = u_dot_sk1.copy()
            self.u_sk = u_sk1.copy()
            self.la_gk = la_gk1.copy()
            self.la_Nk = la_Nk1.copy()

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            a.append((uk1 - self.uk) / self.dt)
            U.append(Uk1)
            la_g.append(la_gk1)
            La_g.append(La_gk1)
            P_g.append(P_gk1)
            la_N.append(la_Nk1)
            La_N.append(La_Nk1)
            P_N.append(P_Nk1)
            P_F.append(P_Fk1)

            # update local variables for accepted time step
            self.tk = tk1
            self.xk = xk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            a=np.array(a),
            U=np.array(U),
            la_g=np.array(la_g),
            La_g=np.array(La_g),
            P_g=np.array(P_g),
            La_N=np.array(La_N),
            la_N=np.array(la_N),
            P_N=np.array(P_N),
            la_F=np.zeros((len(t), self.nla_F)),
            La_F=np.zeros((len(t), self.nla_F)),
            P_F=np.array(P_F),
        )


class RemcoOriginal:
    def __init__(
        self,
        model,
        t1,
        dt,
        tol=1e-8,
        max_iter=40,
        error_function=lambda x: np.max(np.abs(x)),
    ):
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
        self.nx = self.nu + self.nla_N
        # self.ny = self.nq + self.nu + self.nla_N
        self.ny = self.nu + self.nla_N

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.Uk = np.zeros(self.nq)
        # self.La_gk = np.zeros(self.nla_g)
        # self.la_gk = model.la_g0
        self.La_Nk = np.zeros(self.nla_N)
        self.la_Nk = model.la_N0
        # self.P_Fk = dt * model.la_F0

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
        self.xk = np.concatenate((self.Uk, self.La_Nk))
        # self.yk = np.concatenate((self.q_dotk, self.u_dotk, self.la_Nk))
        self.yk = np.concatenate((self.u_dotk, self.la_Nk))

        # initialize index sets
        self.Ak1 = np.zeros(self.nla_N, dtype=bool)
        self.Bk1 = np.zeros(self.nla_N, dtype=bool)
        self.Dk1_st = np.zeros(self.nla_N, dtype=bool)

    def unpack_x(self, xk1):
        nu = self.nu
        nla_N = self.nla_N

        Uk1 = xk1[:nu]
        La_Nk_plus = xk1[nu : nu + nla_N]

        return Uk1, La_Nk_plus

    def unpack_y(self, yk1):
        # nq = self.nq
        nu = self.nu
        nla_N = self.nla_N

        # # unpack yk1
        # q_dotk1 = yk1[:nq]
        # u_dotk1 = yk1[nq : nq + nu]
        # la_Nk = yk1[nq + nu : nq + nu + nla_N]

        # return q_dotk1, u_dotk1, la_Nk

        # unpack yk1
        u_dotk1 = yk1[:nu]
        la_Nk = yk1[nu : nu + nla_N]

        return u_dotk1, la_Nk

    def update_x(self, xk1):
        Uk1, La_Nk_plus = self.unpack_x(xk1)

        # compute intermediate velocity
        uk_plus = self.uk + Uk1

        return uk_plus

    def update_y(self, yk1):
        dt = self.dt
        tk = self.tk
        qk = self.qk

        # q_dotk1, u_dotk1, la_Nk1 = self.unpack_y(yk1)
        u_dotk1, la_Nk1 = self.unpack_y(yk1)

        # backward Euler
        tk1 = tk + dt
        # uk1 = self.uk_plus + dt * u_dotk1
        uk1 = self.uk + dt * u_dotk1 + self.Uk1
        # qk1 = qk + dt * self.uk + 0.5 * dt**2 * u_dotk1 #+ dt * self.Uk1
        qk1 = qk + dt * (uk1 - self.Uk1)

        return tk1, qk1, uk1

    def Rx(self, xk1):
        nu = self.nu
        nla_N = self.nla_N

        # quantities of old time step
        tk = self.tk
        qk = self.qk
        uk = self.uk

        # unpack xk1
        Uk1, La_Nk_plus = self.unpack_x(xk1)
        self.Uk1 = Uk1.copy()

        # compute intermediate velocity
        uk_plus = self.update_x(xk1)
        self.uk_plus = uk_plus

        # evaluate repeatedly used quantities
        Mk = self.model.M(tk, qk)
        W_Nk = self.model.W_N(tk, qk, scipy_matrix=csr_matrix)
        g_Nk = self.model.g_N(tk, qk)
        xi_Nk_plus = self.model.xi_N(tk, qk, uk, uk_plus)

        # W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        # gamma_F_qk1 = self.model.gamma_F_q(tk1, qk1, uk1)
        # xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        # compute set of active contacts
        I_N = g_Nk <= 0

        ###################
        # evaluate residual
        ###################
        Rx = np.zeros(self.nx)

        #################
        # impact equation
        #################
        Rx[:nu] = Mk @ Uk1 - W_Nk @ La_Nk_plus  # - W_Fk1 @ P_Fk1

        ############
        # impact law
        ############
        Rx[nu : nu + nla_N] = np.where(
            I_N,
            xi_Nk_plus - prox_R0_np(xi_Nk_plus - self.model.prox_r_N * La_Nk_plus),
            La_Nk_plus,
        )

        return Rx

    def Ry(self, yk1):
        nq = self.nq
        nu = self.nu
        nla_N = self.nla_N

        # q_dotk1, u_dotk1, la_Nk1 = self.unpack_y(yk1)
        u_dotk1, la_Nk1 = self.unpack_y(yk1)
        tk1, qk1, uk1 = self.update_y(yk1)

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        hk1 = self.model.h(tk1, qk1, uk1)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        g_Nk1 = self.model.g_N(tk1, qk1)

        ###################
        # evaluate residual
        ###################
        Ry = np.zeros(self.ny)

        # ####################
        # # kinematic equation
        # ####################
        # # Ry[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1)
        # Ry[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1 - self.Uk1)

        ####################
        # euations of motion
        ####################
        # Ry[nq : nq + nu] = Mk1 @ u_dotk1 - hk1 - W_Nk1 @ la_Nk1
        Ry[:nu] = Mk1 @ u_dotk1 - hk1 - W_Nk1 @ la_Nk1

        ################
        # normal contact
        ################
        # Ry[nq + nu : nq + nu + nla_N] = g_Nk1 - prox_R0_np(
        #     g_Nk1 - self.model.prox_r_N * la_Nk1
        # )
        Ry[nu : nu + nla_N] = g_Nk1 - prox_R0_np(g_Nk1 - self.model.prox_r_N * la_Nk1)

        return Ry

    def step(self, xk1, f):
        # initial residual and error
        R = f(xk1)
        error = self.error_function(R)
        converged = error < self.tol

        # print(f"initial error: {error}")
        j = 0
        if not converged:
            while j < self.max_iter:
                # jacobian
                # J = csr_matrix(approx_fprime(xk1, f, method="2-point"))
                J = csr_matrix(approx_fprime(xk1, f, method="3-point"))

                # Newton update
                j += 1
                dx = spsolve(J, R, use_umfpack=True)
                # dx = lsqr(J, R)[0]
                xk1 -= dx
                R = f(xk1)

                error = self.error_function(R)
                # print(f" - j: {j}; error: {error}")
                converged = error < self.tol
                if converged:
                    break

            if not converged:
                # raise RuntimeError("internal Newton-Raphson not converged")
                print(f"not converged!")

        return converged, j, error, xk1

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        a = [np.zeros(self.nu)]
        U = [self.Uk]
        # la_g = [self.la_gk]
        # La_g = [self.La_gk]
        # P_g = [self.dt * self.la_gk + self.La_gk]
        # P_gamma = [self.P_gammak]
        la_N = [self.la_Nk]
        La_N = [self.La_Nk]
        P_N = [self.dt * self.la_Nk + self.La_Nk]
        # P_F = [self.P_Fk]

        # pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        # for _ in pbar:
        for _ in np.arange(self.t0, self.t1, self.dt):
            # perform a sovler step
            tk1 = self.tk + self.dt
            xk1 = self.xk.copy()
            yk1 = self.yk.copy()

            converged_x, n_iter_x, error_x, xk1 = self.step(xk1, self.Rx)
            converged_y, n_iter_y, error_y, yk1 = self.step(yk1, self.Ry)

            # update progress bar and check convergence
            # pbar.set_description(
            #     f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter_x}/{self.max_iter} iterations; error: {error_x:0.2e}"
            # )
            print(f"t: {tk1:0.2e}s < {self.t1:0.2e}s")
            print(
                f"  Newton_x: {n_iter_x}/{self.max_iter} iterations; error: {error_x:0.2e}"
            )
            print(
                f"  Newton_y: {n_iter_y}/{self.max_iter} iterations; error: {error_y:0.2e}"
            )
            if not (converged_x and converged_y):
                print(
                    f"internal Newton-Raphson method not converged after {n_iter_x} x-steps with error: {error_x:.5e}"
                )
                print(
                    f"internal Newton-Raphson method not converged after {n_iter_y} y-steps with error: {error_y:.5e}"
                )
                # write solution
                return Solution(
                    t=np.array(t),
                    q=np.array(q),
                    u=np.array(u),
                    a=np.array(a),
                    U=np.array(U),
                    # la_g=np.array(la_g),
                    # La_g=np.array(La_g),
                    # P_g=np.array(P_g),
                    la_N=np.array(la_N),
                    La_N=np.array(La_N),
                    P_N=np.array(P_N),
                    # la_F=np.zeros((len(t), self.nla_F)),
                    # La_F=np.zeros((len(t), self.nla_F)),
                    # P_F=np.array(P_F),
                )

            Uk1, La_Nk_plus = self.unpack_x(xk1)
            # q_dotk1, u_dotk1, la_Nk1 = self.unpack_y(yk1)
            u_dotk1, la_Nk1 = self.unpack_y(yk1)

            uk_plus = self.update_x(xk1)
            tk1, qk1, uk1 = self.update_y(yk1)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # update converged and updated quantities of previous time step
            self.qk = qk1.copy()
            self.uk = uk1.copy()
            # self.q_dotk = q_dotk1.copy()
            self.u_dotk = u_dotk1.copy()

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            a.append((uk1 - self.uk) / self.dt)
            U.append(Uk1)
            # la_g.append(la_gk1)
            # La_g.append(La_gk1)
            # P_g.append(self.dt * la_gk1 + La_gk1)
            la_N.append(la_Nk1)
            La_N.append(La_Nk_plus)
            P_N.append(self.dt * la_Nk1 + La_Nk_plus)
            # P_F.append(P_Fk1)

            # update local variables for accepted time step
            self.tk = tk1
            self.xk = xk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            a=np.array(a),
            U=np.array(U),
            # la_g=np.array(la_g),
            # La_g=np.array(La_g),
            # P_g=np.array(P_g),
            La_N=np.array(La_N),
            la_N=np.array(la_N),
            P_N=np.array(P_N),
            # la_F=np.zeros((len(t), self.nla_F)),
            # La_F=np.zeros((len(t), self.nla_F)),
            # P_F=np.array(P_F),
        )


# TODO:
# - add bilateral constraints
# - solve Ry as LCP or using fixed point iteration an in Moreau in oder to
#   overcome redundant contacts!
# - use half explicit Runge-Kutta method for smooth Rx solution
class Remco:
    def __init__(
        self,
        model,
        t1,
        dt,
        tol=1e-8,
        max_iter=40,
        error_function=lambda x: np.max(np.abs(x)),
    ):
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
        self.nla_gamma = model.nla_gamma
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F
        self.nx = self.nq + self.nu + self.nla_N + self.nla_F
        self.ny = self.nu + self.nla_N + self.nla_F

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.Uk = np.zeros(self.nq)
        self.La_Nk = np.zeros(self.nla_N)
        self.la_Nk = model.la_N0
        self.La_Fk = np.zeros(self.nla_F)
        self.la_Fk = model.la_F0

        # initial velocites
        self.q_dotk = self.model.q_dot(self.tk, self.qk, self.uk)

        M0 = self.model.M(self.tk, self.qk, scipy_matrix=csr_matrix)
        h0 = self.model.h(self.tk, self.qk, self.uk)
        W_N0 = self.model.W_N(self.tk, self.qk, scipy_matrix=csr_matrix)
        W_F0 = self.model.W_F(self.tk, self.qk, scipy_matrix=csr_matrix)
        self.u_dotk = spsolve(M0, h0 + W_N0 @ model.la_N0 + W_F0 @ model.la_F0)

        #######################################################################
        # starting values for generalized state vector, its derivatives and
        # auxiliary velocities
        #######################################################################
        self.xk = np.concatenate((self.q_dotk, self.u_dotk, self.la_Nk, self.la_Fk))
        self.yk = np.concatenate((self.Uk, self.La_Nk, self.La_Fk))

        # initialize index sets
        self.Ak1 = np.zeros(self.nla_N, dtype=bool)

    def f(self, tk, xk, zk1):
        # unpack state
        qk = xk[: self.nq]
        uk = xk[self.nq :]

        # unpack constraint forces
        la_gk1 = zk1[: self.nla_g]
        la_gammak1 = zk1[self.nla_g : self.nla_g + self.nla_gamma]
        la_Nk1 = zk1[
            self.nla_g + self.nla_gamma : self.nla_g + self.nla_gamma + self.nla_N
        ]
        la_Fk1 = zk1[self.nla_g + self.nla_gamma + self.nla_N :]

        # evaluate quantities of previous time step
        Mk = self.model.M(tk, qk, scipy_matrix=csr_matrix)
        hk = self.model.h(tk, qk, uk)
        W_gk = self.model.W_g(tk, qk, scipy_matrix=csr_matrix)
        W_gammak = self.model.W_gamma(tk, qk, scipy_matrix=csr_matrix)
        W_Nk = self.model.W_N(tk, qk, scipy_matrix=csr_matrix)
        W_Fk = self.model.W_F(tk, qk, scipy_matrix=csr_matrix)

        return np.concatenate(
            (
                self.model.q_dot(tk, qk, uk),
                spsolve(
                    Mk,
                    hk
                    + W_gk @ la_gk1
                    + W_gammak @ la_gammak1
                    + W_Nk @ la_Nk1
                    + W_Fk @ la_Fk1,
                ),
            )
        )

    def c_x(self, zk1, update_index_set=True):
        # unpack percussions
        la_gk1 = zk1[: self.nla_g]
        la_gammak1 = zk1[self.nla_g : self.nla_g + self.nla_gamma]
        la_Nk1 = zk1[
            self.nla_g + self.nla_gamma : self.nla_g + self.nla_gamma + self.nla_N
        ]
        la_Fk1 = zk1[self.nla_g + self.nla_gamma + self.nla_N :]

        #####################
        # explicit Euler step
        #####################
        tk1 = self.tk + self.dt
        xk = np.concatenate((self.qk, self.uk))
        xk1 = xk + self.dt * self.f(self.tk, xk, zk1)
        qk1 = xk1[: self.nq]
        uk1_free = xk1[self.nq :]

        # constraint equations
        gk1 = self.model.g(tk1, qk1)
        gammak1 = self.model.gamma(tk1, qk1, uk1_free)
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1_free)

        prox_arg = g_Nk1 - self.model.prox_r_N * la_Nk1
        c_Nk1 = g_Nk1 - prox_R0_np(prox_arg)

        if update_index_set:
            # self.Ak1 = g_Nk1 <= 0
            self.Ak1 = prox_arg <= 0

        ##########
        # friction
        ##########
        mu = self.model.mu
        c_Fk1 = la_Fk1.copy()  # this is the else case => P_Fk1 = 0
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                # TODO: Is there a primal/ dual form?
                if self.Ak1[i_N]:
                    c_Fk1[i_F] = -la_Fk1[i_F] - prox_sphere(
                        -la_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * la_Nk1[i_N],
                    )

        ck1 = np.concatenate((gk1, gammak1, c_Nk1, c_Fk1))

        return ck1, tk1, qk1, uk1_free, la_gk1, la_gammak1, la_Nk1, la_Fk1

    def unpack_x(self, xk1):
        nq = self.nq
        nu = self.nu
        nla_N = self.nla_N

        # unpack yk1
        q_dotk1 = xk1[:nq]
        u_dotk1 = xk1[nq : nq + nu]
        la_Nk = xk1[nq + nu : nq + nu + nla_N]
        la_Fk = xk1[nq + nu + nla_N :]

        return q_dotk1, u_dotk1, la_Nk, la_Fk

    def unpack_y(self, yk1):
        nu = self.nu
        nla_N = self.nla_N

        Uk1 = yk1[:nu]
        La_Nk_plus = yk1[nu : nu + nla_N]
        La_Fk_plus = yk1[nu + nla_N :]

        return Uk1, La_Nk_plus, La_Fk_plus

    def update_x(self, xk1):
        q_dotk1, u_dotk1, la_Nk1, la_Fk1 = self.unpack_x(xk1)

        # backward Euler
        tk1 = self.tk + self.dt
        uk1_free = self.uk + self.dt * u_dotk1
        qk1 = self.qk + self.dt * q_dotk1

        return tk1, qk1, uk1_free

    def Rx(self, xk1):
        nq = self.nq
        nu = self.nu
        nla_N = self.nla_N
        mu = self.model.mu

        q_dotk1, u_dotk1, la_Nk1, la_Fk1 = self.unpack_x(xk1)
        tk1, qk1, uk1_free = self.update_x(xk1)

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        hk1 = self.model.h(tk1, qk1, uk1_free)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1_free)

        ###################
        # evaluate residual
        ###################
        Rx = np.zeros(self.nx)

        ####################
        # kinematic equation
        ####################
        Rx[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1_free)

        ####################
        # euations of motion
        ####################
        Rx[nq : nq + nu] = Mk1 @ u_dotk1 - hk1 - W_Nk1 @ la_Nk1 - W_Fk1 @ la_Fk1

        ################
        # normal contact
        ################
        prox_arg = g_Nk1 - self.model.prox_r_N * la_Nk1
        self.I_N = prox_arg <= 0.0
        # self.I_N = g_Nk1 <= 0.0
        Rx[nq + nu : nq + nu + nla_N] = g_Nk1 - prox_R0_np(prox_arg)

        ##########
        # friction
        ##########
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                Rx[nq + nu + nla_N + i_F] = np.where(
                    self.I_N[i_N] * np.ones(len(i_F), dtype=bool),
                    -la_Fk1[i_F]
                    - prox_sphere(
                        -la_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * la_Nk1[i_N],
                    ),
                    la_Fk1[i_F],
                )

        # update quantities of new time step
        self.tk1 = tk1
        self.qk1 = qk1
        self.uk1_free = uk1_free

        return Rx

    def Ry(self, yk1):
        nu = self.nu
        nla_N = self.nla_N
        mu = self.model.mu

        # quantities of old time step
        tk1 = self.tk1
        qk1 = self.qk1
        uk1_free = self.uk1_free

        # unpack xk1
        Uk1, La_Nk1, La_Fk1 = self.unpack_y(yk1)

        # update velocities
        uk1 = uk1_free + Uk1

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        xi_Nk1_plus = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1_plus = self.model.xi_F(tk1, qk1, self.uk, uk1)

        ###################
        # evaluate residual
        ###################
        Ry = np.zeros(self.ny)

        #################
        # impact equation
        #################
        Ry[:nu] = Mk1 @ Uk1 - W_Nk1 @ La_Nk1 - W_Fk1 @ La_Fk1

        ###################
        # normal impact law
        ###################
        Ry[nu : nu + nla_N] = np.where(
            self.I_N,
            xi_Nk1_plus - prox_R0_np(xi_Nk1_plus - self.model.prox_r_N * La_Nk1),
            La_Nk1,
        )

        ####################
        # tangent impact law
        ####################
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                Ry[nu + nla_N + i_F] = np.where(
                    self.I_N[i_N] * np.ones(len(i_F), dtype=bool),
                    -La_Fk1[i_F]
                    - prox_sphere(
                        -La_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1_plus[i_F],
                        mu[i_N] * La_Nk1[i_N],
                    ),
                    La_Fk1[i_F],
                )

        return Ry

    def step(self, xk1, f):
        # initial residual and error
        R = f(xk1)
        error = self.error_function(R)
        converged = error < self.tol

        # print(f"initial error: {error}")
        j = 0
        if not converged:
            while j < self.max_iter:
                # jacobian
                # J = csr_matrix(approx_fprime(xk1, f, method="2-point"))
                J = csr_matrix(approx_fprime(xk1, f, method="3-point"))

                # Newton update
                j += 1

                dx = spsolve(J, R, use_umfpack=True)

                # dx = lsqr(J, R, atol=1.0e-12, btol=1.0e-12)[0]

                # # no underflow errors
                # dx = np.linalg.lstsq(J.toarray(), R, rcond=None)[0]

                # # TODO: Can we get this sparse?
                # # using QR decomposition, see https://de.wikipedia.org/wiki/QR-Zerlegung#L%C3%B6sung_regul%C3%A4rer_oder_%C3%BCberbestimmter_Gleichungssysteme
                # b = R.copy()
                # Q, R = np.linalg.qr(J.toarray())
                # z = Q.T @ b
                # dx = np.linalg.solve(R, z)  # solving R*x = Q^T*b

                # # solve normal equation (should be independent of the conditioning
                # # number!)
                # dx = spsolve(J.T @ J, J.T @ R)

                xk1 -= dx

                R = f(xk1)
                error = self.error_function(R)
                converged = error < self.tol
                if converged:
                    break

            if not converged:
                # raise RuntimeError("internal Newton-Raphson not converged")
                print(f"not converged!")

        return converged, j, error, xk1

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        q_dot = [self.q_dotk]
        a = [self.u_dotk]
        U = [self.Uk]
        la_N = [self.la_Nk]
        La_N = [self.La_Nk]
        P_N = [self.dt * self.la_Nk + self.La_Nk]
        la_F = [self.la_Fk]
        La_F = [self.La_Fk]
        P_F = [self.dt * self.la_Fk + self.La_Fk]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            xk1 = self.xk.copy()
            yk1 = self.yk.copy()

            # converged_x, n_iter_x, error_x, xk1 = self.step(xk1, lambda x: self.c_x(x)[0])

            converged_x, n_iter_x, error_x, xk1 = self.step(xk1, self.Rx)

            converged_y, n_iter_y, error_y, yk1 = self.step(yk1, self.Ry)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; ||R_x||: {error_y:0.2e} ({n_iter_x}/{self.max_iter}); ||R_y||: {error_x:0.2e} ({n_iter_y}/{self.max_iter})"
            )
            if not (converged_x and converged_y):
                print(
                    f"internal Newton-Raphson method not converged after {n_iter_x} x-steps with error: {error_x:.5e}"
                )
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
                    la_N=np.array(la_N),
                    La_N=np.array(La_N),
                    P_N=np.array(P_N),
                    la_F=np.array(la_F),
                    La_F=np.array(La_F),
                    P_F=np.array(P_F),
                )

            q_dotk1, u_dotk1, la_Nk1, la_Fk1 = self.unpack_x(xk1)
            Uk1, La_Nk_plus, La_Fk_plus = self.unpack_y(yk1)

            tk1, qk1, uk1_free = self.update_x(xk1)
            uk1 = uk1_free + Uk1

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
            q_dot.append(q_dotk1)
            a.append(u_dotk1)
            U.append(Uk1)
            la_N.append(la_Nk1)
            La_N.append(La_Nk_plus)
            P_N.append(self.dt * la_Nk1 + La_Nk_plus)
            la_F.append(la_Fk1)
            La_F.append(La_Fk_plus)
            P_F.append(self.dt * la_Fk1 + La_Fk_plus)

            # update local variables for accepted time step
            self.tk = tk1
            self.xk = xk1.copy()
            self.yk = yk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            q_dot=np.array(q_dot),
            a=np.array(a),
            U=np.array(U),
            # la_g=np.array(la_g),
            # La_g=np.array(La_g),
            # P_g=np.array(P_g),
            La_N=np.array(La_N),
            la_N=np.array(la_N),
            P_N=np.array(P_N),
            la_F=np.array(la_F),
            La_F=np.array(La_F),
            P_F=np.array(P_F),
        )


# TODO:
# - analytical Jacobian
# - alternative solution strategy (fixed-point iterations)
#   is the lsqr aproach working?
# - variable step-size with Richardson iteration, see Acary presentation/ Hairer
class NonsmoothDecoupled:
    def __init__(
        self,
        model,
        t1,
        dt,
        tol=1e-6,
        max_iter=10,
        error_function=lambda x: np.max(np.abs(x)),
    ):
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

        #########################################
        # solve for consistent initial conditions
        #########################################
        from scipy.optimize import fsolve

        res = fsolve(self.Rx, self.xk.copy(), full_output=1)
        self.xk = res[0].copy()

        (
            self.q_dotk,
            self.u_dotk,
            self.la_gk,
            self.la_gammak,
            self.la_Nk,
            self.la_Fk,
        ) = self.unpack_x(self.xk)
        print(f"la_g0: {self.la_gk}")

        # initialize index sets
        self.I_Nk1 = np.zeros(self.nla_N, dtype=bool)

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
        q_dotk1, u_dotk1, la_gk1, la_gammak1, la_Nk1, la_Fk1 = self.unpack_x(xk1)

        tk1 = self.tk + self.dt

        la_gk1_free = la_gk1.copy()
        la_gammak1_free = la_gammak1.copy()
        la_Nk1_free = la_Nk1.copy()
        la_Fk1_free = la_Fk1.copy()

        ################
        # backward Euler
        ################
        uk1_free = self.uk + self.dt * u_dotk1
        qk1 = self.qk + self.dt * q_dotk1
        # qk1 = self.qk + self.dt * self.uk + 0.5 * self.dt**2 * u_dotk1

        # ##############
        # # Newmark beta, see https://de.wikipedia.org/wiki/Newmark-beta-Verfahren
        # ##############
        # gamma = 0.5
        # beta = 1/6
        # uk1_free = self.uk + gamma * self.dt * (self.u_dotk + u_dotk1)
        # # qk1 = self.qk + self.dt * q_dotk1
        # # qk1 = self.qk + self.dt * self.u_dotk + self.dt**2 * ((0.5 - beta) * self.u_dotk + beta * u_dotk1)
        # # qk1 = self.qk + self.dt * (self.uk - self.Uk) + self.dt**2 * ((0.5 - beta) * self.u_dotk + beta * u_dotk1)
        # qk1 = self.qk + self.dt * self.uk + self.dt**2 * ((0.5 - beta) * self.u_dotk + beta * u_dotk1)

        # ################
        # # trapezoid rule,
        # # TODO: Works only with unilateral constraints on acceleration level!
        # ################
        # uk1_free = self.uk + 0.5 * self.dt * (self.u_dotk + u_dotk1)
        # qk1 = self.qk + 0.5 * self.dt * (self.q_dotk + q_dotk1)

        return (
            tk1,
            qk1,
            uk1_free,
            la_gk1_free,
            la_gammak1_free,
            la_Nk1_free,
            la_Fk1_free,
        )

    def Rx(self, xk1, update_index=False):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        mu = self.model.mu

        q_dotk1, u_dotk1, la_gk1, la_gammak1, la_Nk1, la_Fk1 = self.unpack_x(xk1)
        (
            tk1,
            qk1,
            uk1_free,
            la_gk1_free,
            la_gammak1_free,
            la_Nk1_free,
            la_Fk1_free,
        ) = self.update_x(xk1)

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
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1_free)

        ###################
        # evaluate residual
        ###################
        Rx = np.zeros(self.nx, dtype=float)

        ####################
        # kinematic equation
        ####################
        Rx[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1_free)
        # Rx[:nq] = q_dotk1

        ####################
        # euations of motion
        ####################
        Rx[nq : nq + nu] = (
            Mk1 @ u_dotk1
            - hk1
            - W_gk1 @ la_gk1_free
            - W_gammak1 @ la_gammak1_free
            - W_Nk1 @ la_Nk1_free
            - W_Fk1 @ la_Fk1_free
        )

        #######################
        # bilateral constraints
        #######################
        Rx[nq + nu : nq + nu + nla_g] = gk1
        Rx[nq + nu + nla_g : nq + nu + nla_g + nla_gamma] = gammak1

        ################
        # normal contact
        ################
        prox_arg = g_Nk1 - self.model.prox_r_N * la_Nk1_free
        if update_index:
            self.I_Nk1 = prox_arg <= 0.0
            # self.I_Nk1 = g_Nk1 <= 0.0
            # self.B_Nk1 = self.I_Nk1 * (g_N_dotk1 <= 0.0)

        Rx[
            nq + nu + nla_g + nla_gamma : nq + nu + nla_g + nla_gamma + nla_N
        ] = g_Nk1 - prox_R0_np(prox_arg)
        # Rx[
        #     nq + nu + nla_g + nla_gamma : nq + nu + nla_g + nla_gamma + nla_N
        # ] = np.where(
        #     self.I_Nk1,
        #     g_N_dotk1 - prox_R0_np(g_N_dotk1 - self.model.prox_r_N * la_Nk1_free),
        #     la_Nk1_free
        # )
        # Rx[
        #     nq + nu + nla_g + nla_gamma : nq + nu + nla_g + nla_gamma + nla_N
        # ] = np.where(
        #     self.B_Nk1,
        #     g_N_ddotk1 - prox_R0_np(g_N_ddotk1 - self.model.prox_r_N * la_Nk1_free),
        #     la_Nk1_free
        # )

        # a = 1.0e-1
        # b = 1.0e-1
        # g_bar = g_N_ddotk1 + a * g_N_dotk1 + b * g_Nk1
        # Rx[
        #     nq + nu + nla_g + nla_gamma : nq + nu + nla_g + nla_gamma + nla_N
        # ] = np.where(
        #     self.B_Nk1,
        #     g_bar - prox_R0_np(g_bar - self.model.prox_r_N * la_Nk1_free),
        #     la_Nk1_free
        # )

        ##########
        # friction
        ##########
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                Rx[nq + nu + nla_g + nla_gamma + nla_N + i_F] = np.where(
                    self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                    -la_Fk1_free[i_F]
                    - prox_sphere(
                        -la_Fk1_free[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * la_Nk1_free[i_N],
                    ),
                    la_Fk1_free[i_F],
                )

        # update quantities of new time step
        self.tk1 = tk1
        self.qk1 = qk1
        self.uk1_free = uk1_free

        return Rx

    def Jx(self, xk1):
        # nq = self.nq
        # nu = self.nu
        # nla_g = self.nla_g
        # nla_gamma = self.nla_gamma
        # nla_N = self.nla_N
        # mu = self.model.mu

        # q_dotk1, u_dotk1, la_gk1, la_gammak1, la_Nk1, la_Fk1 = self.unpack_x(xk1)
        # (
        #     tk1,
        #     qk1,
        #     uk1_free,
        #     la_gk1_free,
        #     la_gammak1_free,
        #     la_Nk1_free,
        #     la_Fk1_free,
        # ) = self.update_x(xk1)

        # # evaluate repeatedly used quantities
        # Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)
        # hk1 = self.model.h(tk1, qk1, uk1_free)
        # W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        # W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)
        # W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        # W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        # gk1 = self.model.g(tk1, qk1)
        # gammak1 = self.model.gamma(tk1, qk1, uk1_free)
        # g_Nk1 = self.model.g_N(tk1, qk1)
        # xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1_free)

        # # chain rules for backward Euler update
        # qk1_q_dotk1 = self.dt
        # uk1_free_u_dotk1 = self.dt

        # ####################
        # # kinematic equation
        # ####################
        # # Rx[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1_free)
        # J_q_dotk1_q_dotk1 = (
        #     eye(nq, nq) - self.model.q_dot_q(tk1, qk1, uk1_free) * qk1_q_dotk1
        # )
        # J_q_dotk1_u_dotk1 = -self.model.B(tk1, qk1) * uk1_free_u_dotk1

        # ####################
        # # euations of motion
        # ####################
        # # Rx[nq : nq + nu] = (
        # #     Mk1 @ u_dotk1
        # #     - hk1
        # #     - W_gk1 @ la_gk1_free
        # #     - W_gammak1 @ la_gammak1_free
        # #     - W_Nk1 @ la_Nk1_free
        # #     - W_Fk1 @ la_Fk1_free
        # # )
        # J_u_dotk1_q_dotk1 = (
        #     self.model.Mu_q(tk1, qk1, uk1_free)
        #     - self.model.h_q(tk1, qk1, uk1_free)
        #     - self.model.Wla_g_q(tk1, qk1, la_gk1)
        #     - self.model.Wla_gamma_q(tk1, qk1, la_gammak1)
        #     - self.model.Wla_N_q(tk1, qk1, la_Nk1)
        #     - self.model.Wla_F_q(tk1, qk1, la_Fk1)
        # ) * qk1_q_dotk1
        # J_u_dotk1_u_dotk1 = Mk1 - self.model.h_u(tk1, qk1, uk1_free) * uk1_free_u_dotk1

        # #######################
        # # bilateral constraints
        # #######################
        # # Rx[nq + nu : nq + nu + nla_g] = gk1
        # # Rx[nq + nu + nla_g : nq + nu + nla_g + nla_gamma] = gammak1
        # J_gk1_q_dotk1 = self.model.g_q(tk1, qk1) * qk1_q_dotk1
        # J_gammak1_q_dotk1 = self.model.gamma_q(tk1, qk1, uk1_free) * qk1_q_dotk1
        # J_gammak1_u_dotk1 = W_gammak1.T * uk1_free_u_dotk1

        # ################
        # # normal contact
        # ################
        # # prox_arg = g_Nk1 - self.model.prox_r_N * la_Nk1_free
        # # if update_index:
        # #     self.I_Nk1 = prox_arg <= 0.0
        # # Rx[
        # #     nq
        # #     + nu
        # #     + nla_g
        # #     + nla_gamma : nq
        # #     + nu
        # #     + nla_g
        # #     + nla_gamma
        # #     + nla_N
        # # ] = g_Nk1 - prox_R0_np(prox_arg)

        # raise RuntimeError("Move on here!")

        # Jx = bmat(
        #     [
        #         [J_q_dotk1_q_dotk1, J_q_dotk1_u_dotk1, None, None, None, None],
        #         [J_u_dotk1_q_dotk1, J_u_dotk1_u_dotk1, W_gk1, W_gammak1, W_Nk1, W_Fk1],
        #         [J_gk1_q_dotk1, None, None, None, None, None],
        #         [J_gammak1_q_dotk1, J_gammak1_u_dotk1, None, None, None, None],
        #     ],
        #     format="csr",
        # )

        Jx_num = csr_matrix(approx_fprime(xk1, self.Rx, method="2-point"))
        return Jx_num

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

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_dot = self.model.g_dot(tk1, qk1, uk1)
        gamma = self.model.gamma(tk1, qk1, uk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

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
        )

        # impulsive bilateral constraints
        # TODO: Are they correct?
        Ry[nu : nu + nla_g] = g_dot
        Ry[nu + nla_g : nu + nla_g + nla_gamma] = gamma

        ###################
        # normal impact law
        ###################
        Ry[nu + nla_g + nla_gamma : nu + nla_g + nla_gamma + nla_N] = np.where(
            self.I_Nk1,
            xi_Nk1 - prox_R0_np(xi_Nk1 - self.model.prox_r_N * La_Nk1),
            La_Nk1,
        )

        ####################
        # tangent impact law
        ####################
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                Ry[nu + nla_g + nla_gamma + nla_N + i_F] = np.where(
                    self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                    -La_Fk1[i_F]
                    - prox_sphere(
                        -La_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * La_Nk1[i_N],
                    ),
                    La_Fk1[i_F],
                )

        return Ry

    def Jy(self, yk1):
        return csr_matrix(approx_fprime(yk1, self.Ry, method="2-point"))

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
        # def Rs(self, sk1, update_index=False, use_percussions=True):
        nq = self.nq
        nu = self.nu
        nla_N = self.nla_N
        nla_F = self.nla_F
        mu = self.model.mu

        q_dotk1, u_dotk1, Uk1, la_Nk1, La_Nk1, la_Fk1, La_Fk1 = self.unpack_s(sk1)
        tk1, qk1, uk1, uk1_free, P_Nk1, P_Fk1 = self.update_s(sk1)

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)
        hk1 = self.model.h(tk1, qk1, uk1_free)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_Nk1 = self.model.g_N(tk1, qk1)
        g_N_dotk1_free = self.model.g_N_dot(tk1, qk1, uk1_free)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1_free = self.model.xi_F(tk1, qk1, self.uk, uk1_free)
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
        prox_arg = g_Nk1 - self.model.prox_r_N * la_Nk1
        if update_index:
            self.I_Nk1 = prox_arg <= 0.0
            # self.I_Nk1 = g_Nk1 <= 0.0
        R[nq + 2 * nu : nq + 2 * nu + nla_N] = g_Nk1 - prox_R0_np(prox_arg)
        # R[nq + 2 * nu : nq + 2 * nu + nla_N] = np.where(
        #     self.I_Nk1,
        #     g_N_dotk1_free - prox_R0_np(g_N_dotk1_free - self.model.prox_r_N * la_Nk1),
        #     la_Nk1,
        # )
        if use_percussions:
            R[nq + 2 * nu + nla_N : nq + 2 * nu + 2 * nla_N] = np.where(
                self.I_Nk1,
                xi_Nk1 - prox_R0_np(xi_Nk1 - self.model.prox_r_N * P_Nk1),
                P_Nk1,
            )
        else:
            R[nq + 2 * nu + nla_N : nq + 2 * nu + 2 * nla_N] = np.where(
                self.I_Nk1,
                xi_Nk1 - prox_R0_np(xi_Nk1 - self.model.prox_r_N * La_Nk1),
                La_Nk1,
            )

        ##########
        # friction
        ##########
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                R[nq + 2 * nu + 2 * nla_N + i_F] = np.where(
                    self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                    -la_Fk1[i_F]
                    - prox_sphere(
                        -la_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1_free[i_F],
                        mu[i_N] * la_Nk1[i_N],
                    ),
                    la_Fk1[i_F],
                )
                if use_percussions:
                    R[nq + 2 * nu + 2 * nla_N + nla_F + i_F] = np.where(
                        self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                        -P_Fk1[i_F]
                        - prox_sphere(
                            -P_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                            mu[i_N] * P_Nk1[i_N],
                        ),
                        P_Fk1[i_F],
                    )
                else:
                    R[nq + 2 * nu + 2 * nla_N + nla_F + i_F] = np.where(
                        self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                        -La_Fk1[i_F]
                        - prox_sphere(
                            -La_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                            mu[i_N] * La_Nk1[i_N],
                        ),
                        La_Fk1[i_F],
                    )

        return R

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
                # J = csr_matrix(approx_fprime(xk1, f, method="2-point"))
                # # J = csr_matrix(approx_fprime(xk1, f, method="3-point"))
                J = G(xk1)

                # Newton update
                j += 1

                dx = spsolve(J, R, use_umfpack=True)

                # dx = lsqr(J, R, atol=1.0e-12, btol=1.0e-12)[0]

                # # no underflow errors
                # dx = np.linalg.lstsq(J.toarray(), R, rcond=None)[0]

                # # TODO: Can we get this sparse?
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

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            xk1 = self.xk.copy()
            yk1 = self.yk.copy()
            sk1 = self.sk.copy()

            converged_x, n_iter_x, error_x, xk1 = self.step(xk1, self.Rx, self.Jx)
            converged_y, n_iter_y, error_y, yk1 = self.step(yk1, self.Ry, self.Jy)
            # converged, n_iter, error, sk1 = self.step(sk1, self.Rs)

            # update progress bar and check convergence
            # pbar.set_description(
            #     f"t: {tk1:0.2e}s < {self.t1:0.2e}s; ||R_x||: {error_x:0.2e} ({n_iter_x}/{self.max_iter});"
            # )
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; ||R_x||: {error_x:0.2e} ({n_iter_x}/{self.max_iter}); ||R_y||: {error_y:0.2e} ({n_iter_y}/{self.max_iter})"
            )
            # pbar.set_description(
            #     f"t: {tk1:0.2e}s < {self.t1:0.2e}s; ||R||: {error:0.2e} ({n_iter}/{self.max_iter})"
            # )
            if not (converged_x and converged_y):
                # if not converged:
                print(
                    f"internal Newton-Raphson method not converged after {n_iter_x} x-steps with error: {error_x:.5e}"
                )
                print(
                    f"internal Newton-Raphson method not converged after {n_iter_y} y-steps with error: {error_y:.5e}"
                )
                # print(
                #     f"internal Newton-Raphson method not converged after {n_iter} x-steps with error: {error:.5e}"
                # )

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
                )

            q_dotk1, u_dotk1, la_gk1, la_gammak1, la_Nk1, la_Fk1 = self.unpack_x(xk1)
            (
                tk1,
                qk1,
                uk1_free,
                la_gk1_free,
                la_gammak1_free,
                la_Nk1_free,
                la_Fk1_free,
            ) = self.update_x(xk1)

            Uk1, La_gk1, La_gammak1, La_Nk1, La_Fk1 = self.unpack_y(yk1)
            uk1 = uk1_free + Uk1
            # uk1 = uk1_free

            # q_dotk1, u_dotk1, Uk1, la_Nk1, La_Nk1, la_Fk1, La_Fk1 = self.unpack_s(sk1)
            # tk1, qk1, uk1, uk1_free, P_Nk1, P_Fk1 = self.update_s(sk1)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            q_dot.append(q_dotk1)
            a.append(u_dotk1)
            U.append(Uk1)
            la_g.append(la_gk1)
            La_g.append(La_gk1)
            P_g.append(self.dt * la_gk1 + La_gk1)
            la_gamma.append(la_gammak1)
            La_gamma.append(La_gammak1)
            P_gamma.append(self.dt * la_gammak1 + La_gammak1)
            la_N.append(la_Nk1)
            La_N.append(La_Nk1)
            P_N.append(self.dt * la_Nk1 + La_Nk1)
            la_F.append(la_Fk1)
            La_F.append(La_Fk1)
            P_F.append(self.dt * la_Fk1 + La_Fk1)

            # update local variables for accepted time step
            self.tk = tk1

            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.q_dotk = q_dotk1.copy()
            self.u_dotk = u_dotk1.copy()
            self.Uk = Uk1.copy()
            # self.la_gk = la_gk1.copy()
            # self.la_gammak = la_gammak1.copy()
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
        )


class NonsmoothDecoupledGGLRx:
    def __init__(
        self,
        model,
        t1,
        dt,
        tol=1e-8,
        max_iter=40,
        error_function=lambda x: np.max(np.abs(x)),
    ):
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
        self.nla_gamma = model.nla_gamma
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F
        self.nx = (
            # self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_N + self.nla_F
            self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
            + 2 * self.nla_N
            + self.nla_F
        )
        self.ny = self.nu + self.nla_g + self.nla_gamma + self.nla_N + self.nla_F

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = t0 = model.t0
        self.qk = q0 = model.q0
        self.uk = u0 = model.u0
        self.Uk = np.zeros(self.nq)
        self.La_gk = np.zeros(self.nla_g)
        self.La_gammak = np.zeros(self.nla_gamma)
        self.La_Nk = np.zeros(self.nla_N)
        self.mu_Nk = np.zeros(self.nla_N)
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
                self.mu_Nk,
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

        # initialize index sets
        self.I_Nk1 = np.zeros(self.nla_N, dtype=bool)

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
        mu_Nk1 = xk1[
            self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
            + self.nla_N : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
            + 2
            + self.nla_N
        ]
        la_Fk1 = xk1[self.nq + self.nu + self.nla_g + self.nla_gamma + 2 * self.nla_N :]

        return q_dotk1, u_dotk1, la_gk1, la_gammak1, la_Nk1, mu_Nk1, la_Fk1

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
        q_dotk1, u_dotk1, la_gk1, la_gammak1, la_Nk1, mu_Nk1, la_Fk1 = self.unpack_x(
            xk1
        )

        ################
        # backward Euler
        ################
        tk1 = self.tk + self.dt
        uk1_free = self.uk + self.dt * u_dotk1
        qk1 = self.qk + self.dt * q_dotk1
        la_gk1_free = la_gk1.copy()
        la_gammak1_free = la_gammak1.copy()
        la_Nk1_free = la_Nk1.copy()
        mu_Nk1_free = mu_Nk1.copy()
        la_Fk1_free = la_Fk1.copy()

        # ################
        # # trapezoid rule
        # ################
        # tk1 = self.tk + self.dt
        # uk1_free = self.uk + 0.5 * self.dt * (self.u_dotk + u_dotk1)
        # qk1 = self.qk + 0.5 * self.dt * (self.q_dotk + q_dotk1)
        # # la_gk1_free = 0.5 * self.dt * (self.la_gk + la_gk1)
        # # la_gammak1_free = 0.5 * self.dt * (self.la_gammak + la_gammak1)
        # # la_Nk1_free = 0.5 * self.dt * (self.la_Nk + la_Nk1)
        # # la_Fk1_free = 0.5 * self.dt * (self.la_Fk + la_Fk1)

        return (
            tk1,
            qk1,
            uk1_free,
            la_gk1_free,
            la_gammak1_free,
            la_Nk1_free,
            mu_Nk1_free,
            la_Fk1_free,
        )

    def Rx(self, xk1, update_index=False):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        mu = self.model.mu

        q_dotk1, u_dotk1, la_gk1, la_gammak1, la_Nk1, mu_Nk1, la_Fk1 = self.unpack_x(
            xk1
        )
        (
            tk1,
            qk1,
            uk1_free,
            la_gk1_free,
            la_gammak1_free,
            la_Nk1_free,
            mu_Nk1_free,
            la_Fk1_free,
        ) = self.update_x(xk1)

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
        g_N_dotk1_free = self.model.g_N_dot(tk1, qk1, uk1_free)
        g_N_qk1 = self.model.g_N_q(tk1, qk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1_free)

        ###################
        # evaluate residual
        ###################
        Rx = np.zeros(self.nx)

        ####################
        # kinematic equation
        ####################
        Rx[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1_free) - g_N_qk1.T @ mu_Nk1

        ####################
        # euations of motion
        ####################
        Rx[nq : nq + nu] = (
            Mk1 @ u_dotk1
            - hk1
            - W_gk1 @ la_gk1_free
            - W_gammak1 @ la_gammak1_free
            - W_Nk1 @ la_Nk1_free
            - W_Fk1 @ la_Fk1_free
        )

        #######################
        # bilateral constraints
        #######################
        Rx[nq + nu : nq + nu + nla_g] = gk1
        Rx[nq + nu + nla_g : nq + nu + nla_g + nla_gamma] = gammak1

        ################
        # normal contact
        ################
        prox_arg = g_Nk1 - self.model.prox_r_N * mu_Nk1_free
        if update_index:
            self.I_Nk1 = prox_arg <= 0.0
            # self.I_Nk1 = g_Nk1 <= 0.0

        Rx[
            nq + nu + nla_g + nla_gamma : nq + nu + nla_g + nla_gamma + nla_N
        ] = np.where(
            self.I_Nk1,
            g_N_dotk1_free
            - prox_R0_np(g_N_dotk1_free - self.model.prox_r_N * la_Nk1_free),
            la_Nk1_free,
        )
        Rx[
            nq
            + nu
            + nla_g
            + nla_gamma
            + nla_N : nq
            + nu
            + nla_g
            + nla_gamma
            + 2 * nla_N
        ] = g_Nk1 - prox_R0_np(prox_arg)

        ##########
        # friction
        ##########
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                Rx[nq + nu + nla_g + nla_gamma + 2 * nla_N + i_F] = np.where(
                    self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                    -la_Fk1_free[i_F]
                    - prox_sphere(
                        -la_Fk1_free[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * la_Nk1_free[i_N],
                    ),
                    la_Fk1_free[i_F],
                )

        # update quantities of new time step
        self.tk1 = tk1
        self.qk1 = qk1
        self.uk1_free = uk1_free

        return Rx

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

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_dot = self.model.g_dot(tk1, qk1, uk1)
        gamma = self.model.gamma(tk1, qk1, uk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        ###################
        # evaluate residual
        ###################
        Ry = np.zeros(self.nu + self.nla_g + self.nla_gamma + self.nla_N + self.nla_F)

        #################
        # impact equation
        #################
        Ry[:nu] = (
            Mk1 @ Uk1
            - W_gk1 @ La_gk1
            - W_gammak1 @ La_gammak1
            - W_Nk1 @ La_Nk1
            - W_Fk1 @ La_Fk1
        )

        # bilateral constraints
        # TODO: Are they correct?
        Ry[nu : nu + nla_g] = g_dot
        Ry[nu + nla_g : nu + nla_g + nla_gamma] = gamma

        ###################
        # normal impact law
        ###################
        Ry[nu + nla_g + nla_gamma : nu + nla_g + nla_gamma + nla_N] = np.where(
            self.I_Nk1,
            xi_Nk1 - prox_R0_np(xi_Nk1 - self.model.prox_r_N * La_Nk1),
            La_Nk1,
        )

        ####################
        # tangent impact law
        ####################
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                Ry[nu + nla_g + nla_gamma + nla_N + i_F] = np.where(
                    self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                    -La_Fk1[i_F]
                    - prox_sphere(
                        -La_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * La_Nk1[i_N],
                    ),
                    La_Fk1[i_F],
                )

        return Ry

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
        # def Rs(self, sk1, update_index=False, use_percussions=True):
        nq = self.nq
        nu = self.nu
        nla_N = self.nla_N
        nla_F = self.nla_F
        mu = self.model.mu

        q_dotk1, u_dotk1, Uk1, la_Nk1, La_Nk1, la_Fk1, La_Fk1 = self.unpack_s(sk1)
        tk1, qk1, uk1, uk1_free, P_Nk1, P_Fk1 = self.update_s(sk1)

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)
        hk1 = self.model.h(tk1, qk1, uk1_free)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_Nk1 = self.model.g_N(tk1, qk1)
        g_N_dotk1_free = self.model.g_N_dot(tk1, qk1, uk1_free)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1_free = self.model.xi_F(tk1, qk1, self.uk, uk1_free)
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
        prox_arg = g_Nk1 - self.model.prox_r_N * la_Nk1
        if update_index:
            self.I_Nk1 = prox_arg <= 0.0
            # self.I_Nk1 = g_Nk1 <= 0.0
        R[nq + 2 * nu : nq + 2 * nu + nla_N] = g_Nk1 - prox_R0_np(prox_arg)
        # R[nq + 2 * nu : nq + 2 * nu + nla_N] = np.where(
        #     self.I_Nk1,
        #     g_N_dotk1_free - prox_R0_np(g_N_dotk1_free - self.model.prox_r_N * la_Nk1),
        #     la_Nk1,
        # )
        if use_percussions:
            R[nq + 2 * nu + nla_N : nq + 2 * nu + 2 * nla_N] = np.where(
                self.I_Nk1,
                xi_Nk1 - prox_R0_np(xi_Nk1 - self.model.prox_r_N * P_Nk1),
                P_Nk1,
            )
        else:
            R[nq + 2 * nu + nla_N : nq + 2 * nu + 2 * nla_N] = np.where(
                self.I_Nk1,
                xi_Nk1 - prox_R0_np(xi_Nk1 - self.model.prox_r_N * La_Nk1),
                La_Nk1,
            )

        ##########
        # friction
        ##########
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                R[nq + 2 * nu + 2 * nla_N + i_F] = np.where(
                    self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                    -la_Fk1[i_F]
                    - prox_sphere(
                        -la_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1_free[i_F],
                        mu[i_N] * la_Nk1[i_N],
                    ),
                    la_Fk1[i_F],
                )
                if use_percussions:
                    R[nq + 2 * nu + 2 * nla_N + nla_F + i_F] = np.where(
                        self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                        -P_Fk1[i_F]
                        - prox_sphere(
                            -P_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                            mu[i_N] * P_Nk1[i_N],
                        ),
                        P_Fk1[i_F],
                    )
                else:
                    R[nq + 2 * nu + 2 * nla_N + nla_F + i_F] = np.where(
                        self.I_Nk1[i_N] * np.ones(len(i_F), dtype=bool),
                        -La_Fk1[i_F]
                        - prox_sphere(
                            -La_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                            mu[i_N] * La_Nk1[i_N],
                        ),
                        La_Fk1[i_F],
                    )

        return R

    def step(self, xk1, f):
        # initial residual and error
        R = f(xk1, update_index=True)
        error = self.error_function(R)
        converged = error < self.tol

        # print(f"initial error: {error}")
        j = 0
        if not converged:
            while j < self.max_iter:
                # jacobian
                # J = csr_matrix(approx_fprime(xk1, f, method="2-point"))
                J = csr_matrix(approx_fprime(xk1, f, method="3-point"))

                # Newton update
                j += 1

                dx = spsolve(J, R, use_umfpack=True)

                # dx = lsqr(J, R, atol=1.0e-12, btol=1.0e-12)[0]

                # # no underflow errors
                # dx = np.linalg.lstsq(J.toarray(), R, rcond=None)[0]

                # # TODO: Can we get this sparse?
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

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            xk1 = self.xk.copy()
            yk1 = self.yk.copy()
            sk1 = self.sk.copy()

            converged_x, n_iter_x, error_x, xk1 = self.step(xk1, self.Rx)
            converged_y, n_iter_y, error_y, yk1 = self.step(yk1, self.Ry)
            # converged, n_iter, error, sk1 = self.step(sk1, self.Rs)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; ||R_x||: {error_y:0.2e} ({n_iter_x}/{self.max_iter}); ||R_y||: {error_x:0.2e} ({n_iter_y}/{self.max_iter})"
            )
            # pbar.set_description(
            #     f"t: {tk1:0.2e}s < {self.t1:0.2e}s; ||R||: {error:0.2e} ({n_iter}/{self.max_iter})"
            # )
            if not (converged_x and converged_y):
                # if not converged:
                print(
                    f"internal Newton-Raphson method not converged after {n_iter_x} x-steps with error: {error_x:.5e}"
                )
                print(
                    f"internal Newton-Raphson method not converged after {n_iter_y} y-steps with error: {error_y:.5e}"
                )
                # print(
                #     f"internal Newton-Raphson method not converged after {n_iter} x-steps with error: {error:.5e}"
                # )

                # write solution
                return Solution(
                    t=np.array(t),
                    q=np.array(q),
                    u=np.array(u),
                    q_dot=np.array(q_dot),
                    a=np.array(a),
                    U=np.array(U),
                    la_N=np.array(la_N),
                    La_N=np.array(La_N),
                    P_N=np.array(P_N),
                    la_F=np.array(la_F),
                    La_F=np.array(La_F),
                    P_F=np.array(P_F),
                )

            (
                q_dotk1,
                u_dotk1,
                la_gk1,
                la_gammak1,
                la_Nk1,
                mu_Nk1,
                la_Fk1,
            ) = self.unpack_x(xk1)
            (
                tk1,
                qk1,
                uk1_free,
                la_gk1_free,
                la_gammak1_free,
                la_Nk1_free,
                mu_Nk1_free,
                la_Fk1_free,
            ) = self.update_x(xk1)

            Uk1, La_gk1, La_gammak1, La_Nk1, La_Fk1 = self.unpack_y(yk1)
            uk1 = uk1_free + Uk1

            # q_dotk1, u_dotk1, Uk1, la_Nk1, La_Nk1, la_Fk1, La_Fk1 = self.unpack_s(sk1)
            # tk1, qk1, uk1, uk1_free, P_Nk1, P_Fk1 = self.update_s(sk1)

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
            # P_N.append(P_Nk1)
            la_F.append(la_Fk1)
            La_F.append(La_Fk1)
            P_F.append(self.dt * la_Fk1 + La_Fk1)
            # P_F.append(P_Fk1)

            # update local variables for accepted time step
            self.tk = tk1

            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.q_dotk = q_dotk1.copy()
            self.u_dotk = u_dotk1.copy()
            self.Uk = Uk1.copy()
            # self.la_gk = la_gk1.copy()
            # self.la_gammak = la_gammak1.copy()
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
            # la_g=np.array(la_g),
            # La_g=np.array(La_g),
            # P_g=np.array(P_g),
            # la_gamma=np.array(la_gamma),
            # La_gamma=np.array(La_gamma),
            # P_gamma=np.array(P_gamma),
            La_N=np.array(La_N),
            la_N=np.array(la_N),
            P_N=np.array(P_N),
            la_F=np.array(la_F),
            La_F=np.array(La_F),
            P_F=np.array(P_F),
        )


class DecoupledNonsmoothHalfExplicitRungeKutta:
    def __init__(
        self,
        model,
        t1,
        dt,
        tol=1e-8,
        max_iter=40,
        error_function=lambda x: np.max(np.abs(x)),
    ):
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
        self.nla_gamma = model.nla_gamma
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F
        # self.nx = self.nq + self.nu + self.nla_N + self.nla_F
        # self.ny = self.nu + self.nla_N + self.nla_F

        #######################################################################
        # consistent initial conditions
        #######################################################################
        self.tk = t0 = model.t0
        self.qk = q0 = model.q0
        self.uk = u0 = model.u0
        self.Uk = np.zeros(self.nq)
        self.La_gk = np.zeros(self.nla_g)
        self.La_gammak = np.zeros(self.nla_gamma)
        self.La_Nk = np.zeros(self.nla_N)
        self.mu_Nk = np.zeros(self.nla_N)
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
        # self.xk = np.concatenate((self.q_dotk, self.u_dotk, self.la_gk, self.la_gammak, self.la_Nk, self.la_Fk))
        self.xk = np.concatenate((self.la_gk, self.la_gammak, self.la_Nk, self.la_Fk))
        self.yk = np.concatenate(
            (self.Uk, self.La_gk, self.La_gammak, self.La_Nk, self.La_Fk)
        )

        # initialize index sets
        self.Ak1 = np.zeros(self.nla_N, dtype=bool)

    def f(self, tk, xk, zk1):
        # unpack state
        qk = xk[: self.nq]
        uk = xk[self.nq :]

        # unpack constraint forces
        la_gk1 = zk1[: self.nla_g]
        la_gammak1 = zk1[self.nla_g : self.nla_g + self.nla_gamma]
        la_Nk1 = zk1[
            self.nla_g + self.nla_gamma : self.nla_g + self.nla_gamma + self.nla_N
        ]
        mu_Nk1 = zk1[
            self.nla_g
            + self.nla_gamma
            + self.nla_N : self.nla_g
            + self.nla_gamma
            + 2 * self.nla_N
        ]
        la_Fk1 = zk1[self.nla_g + self.nla_gamma + 2 * self.nla_N :]

        # evaluate quantities of previous time step
        Mk = self.model.M(tk, qk, scipy_matrix=csr_matrix)
        hk = self.model.h(tk, qk, uk)
        W_gk = self.model.W_g(tk, qk, scipy_matrix=csr_matrix)
        W_gammak = self.model.W_gamma(tk, qk, scipy_matrix=csr_matrix)
        g_N_qk = self.model.g_N_q(tk, qk, scipy_matrix=csr_matrix)
        W_Nk = self.model.W_N(tk, qk, scipy_matrix=csr_matrix)
        W_Fk = self.model.W_F(tk, qk, scipy_matrix=csr_matrix)

        u_free = uk + self.dt * spsolve(Mk, hk)

        return np.concatenate(
            (
                # self.model.q_dot(tk, qk, uk) + g_N_qk.T @ mu_Nk1,
                self.model.q_dot(tk, qk, u_free) + g_N_qk.T @ mu_Nk1,
                spsolve(
                    Mk,
                    hk
                    + W_gk @ la_gk1
                    + W_gammak @ la_gammak1
                    + W_Nk @ la_Nk1
                    + W_Fk @ la_Fk1,
                ),
            )
        )

    def c(self, zk1, update_index_set=True):
        # unpack percussions
        la_gk1 = zk1[: self.nla_g]
        la_gammak1 = zk1[self.nla_g : self.nla_g + self.nla_gamma]
        la_Nk1 = zk1[
            self.nla_g + self.nla_gamma : self.nla_g + self.nla_gamma + self.nla_N
        ]
        mu_Nk1 = zk1[
            self.nla_g
            + self.nla_gamma
            + self.nla_N : self.nla_g
            + self.nla_gamma
            + 2 * self.nla_N
        ]
        la_Fk1 = zk1[self.nla_g + self.nla_gamma + 2 * self.nla_N :]

        #####################
        # explicit Euler step
        #####################
        tk1 = self.tk + self.dt
        xk = np.concatenate((self.qk, self.uk))
        xk1 = xk + self.dt * self.f(self.tk, xk, zk1)
        qk1 = xk1[: self.nq]
        uk1_free = xk1[self.nq :]

        # ###############
        # # forward Euler
        # ###############
        # c = [0.0]

        # A = [[]]

        # b = [1.0]

        # # ################
        # # # midpoint rule,
        # # # see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Second-order_methods_with_two_stages
        # # ################
        # # c = [
        # #     0,
        # #     0.5,
        # # ]

        # # A = [
        # #     [],
        # #     [0.5],
        # # ]

        # # b = [
        # #     0.0,
        # #     1.0
        # # ]

        # ##########################
        # # classical Runge-Kutta 4,
        # # see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Examples
        # ##########################
        # ################
        # c = [
        #     0,
        #     0.5,
        #     0.5,
        #     1.0,
        # ]

        # A = [
        #     [],
        #     [0.5],
        #     [0.0, 0.5],
        #     [0.0, 0.0, 1.0],
        # ]

        # b = [
        #     1 / 6,
        #     1 / 3,
        #     1 / 3,
        #     1 / 6,
        # ]

        # dt = self.dt
        # tk = self.tk
        # yk = np.concatenate((self.qk, self.uk))

        # k = []
        # for i, (ci, ai) in enumerate(zip(c, A)):
        #     ti = tk + ci * dt
        #     Yi = yk + dt * sum([aij * kj for aij, kj in zip(ai, k)])
        #     if i < len(c) - 1:
        #         k.append(self.f(ti, Yi, np.zeros_like(zk1)))
        #     else:
        #         k.append(self.f(ti, Yi, zk1 / b[-1]))

        # yk1 = yk + dt * sum([bj * kj for bj, kj in zip(b, k)])

        # tk1 = self.tk + self.dt
        # qk1 = yk1[: self.nq]
        # uk1_free = yk1[self.nq :]

        # constraint equations
        gk1 = self.model.g(tk1, qk1)
        gammak1 = self.model.gamma(tk1, qk1, uk1_free)
        g_Nk1 = self.model.g_N(tk1, qk1)
        # g_dot_Nk1 = self.model.g_N_dot(tk1, qk1, uk1_free)
        g_dot_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1_free)  # TODO!!!
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1_free)

        # prox_arg = g_Nk1 - self.model.prox_r_N * la_Nk1
        # c_Nk1 = g_Nk1 - prox_R0_np(prox_arg)
        # c_Nk1 = (g_Nk1 - prox_R0_np(prox_arg)) / self.model.prox_r_N
        # c_Nk1 = la_Nk1

        prox_arg = g_Nk1 - self.model.prox_r_N * mu_Nk1
        c_Nk1_stab = g_Nk1 - prox_R0_np(prox_arg)

        if update_index_set:
            # self.Ak1 = g_Nk1 <= 0
            self.Ak1 = prox_arg <= 0

        c_Nk1 = np.where(
            self.Ak1,
            g_dot_Nk1 - prox_R0_np(g_dot_Nk1 - self.model.prox_r_N * la_Nk1),
            la_Nk1,
        )

        ##########
        # friction
        ##########
        mu = self.model.mu
        c_Fk1 = la_Fk1.copy()  # this is the else case => P_Fk1 = 0
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                # TODO: Is there a primal/ dual form?
                if self.Ak1[i_N]:
                    c_Fk1[i_F] = -la_Fk1[i_F] - prox_sphere(
                        -la_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * la_Nk1[i_N],
                    )

        # ck1 = np.concatenate((gk1, gammak1, c_Nk1, c_Fk1))
        ck1 = np.concatenate((gk1, gammak1, c_Nk1, c_Nk1_stab, c_Fk1))

        return ck1, tk1, qk1, uk1_free, la_gk1, la_gammak1, la_Nk1, mu_Nk1, la_Fk1

    def c_x(self, zk1):
        nx = self.nla_g + self.nla_gamma + 2 * self.nla_N + self.nla_F
        dense = approx_fprime(
            zk1, lambda z: self.c(z, update_index_set=False)[0]
        ).reshape((nx, nx))
        return csr_matrix(dense)
        # return csr_matrix(
        #     approx_fprime(zk1, lambda z: self.c(z, update_index_set=False)[0])
        #     # approx_fprime(
        #     #     zk1, lambda z: self.c(z, update_index_set=False)[0], method="3-point"
        #     # )
        # )

    # def unpack_x(self, xk1):
    #     # nq = self.nq
    #     # nu = self.nu
    #     # nla_N = self.nla_N
    #     # nla_N = self.nla_N
    #     # nla_N = self.nla_N

    #     # # unpack yk1
    #     # q_dotk1 = xk1[:nq]
    #     # u_dotk1 = xk1[nq : nq + nu]
    #     # la_Nk = xk1[nq + nu : nq + nu + nla_N]
    #     # la_Fk = xk1[nq + nu + nla_N :]

    #     # return q_dotk1, u_dotk1, la_Nk, la_Fk

    #     # unpack constraint forces
    #     la_gk1 = xk1[: self.nla_g]
    #     la_gammak1 = xk1[self.nla_g : self.nla_g + self.nla_gamma]
    #     la_Nk1 = xk1[
    #         self.nla_g + self.nla_gamma : self.nla_g + self.nla_gamma + self.nla_N
    #     ]
    #     la_Fk1 = xk1[self.nla_g + self.nla_gamma + self.nla_N :]

    def unpack_y(self, yk1):
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N

        Uk1 = yk1[:nu]
        La_gk1 = yk1[nu : nu + nla_g]
        La_gammak1 = yk1[nu + nla_g : nu + nla_g + nla_gamma]
        La_Nk1 = yk1[nu + nla_g + nla_gamma : nu + nla_g + nla_gamma + nla_N]
        La_Fk1 = yk1[nu + nla_g + nla_gamma + nla_N :]

        return Uk1, La_gk1, La_gammak1, La_Nk1, La_Fk1

    # def update_x(self, xk1):
    #     q_dotk1, u_dotk1, la_Nk1, la_Fk1 = self.unpack_x(xk1)

    #     # backward Euler
    #     tk1 = self.tk + self.dt
    #     uk1_free = self.uk + self.dt * u_dotk1
    #     qk1 = self.qk + self.dt * q_dotk1

    #     return tk1, qk1, uk1_free

    def Rx(self, xk1):
        nq = self.nq
        nu = self.nu
        nla_N = self.nla_N
        mu = self.model.mu

        q_dotk1, u_dotk1, la_Nk1, la_Fk1 = self.unpack_x(xk1)
        tk1, qk1, uk1_free = self.update_x(xk1)

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        hk1 = self.model.h(tk1, qk1, uk1_free)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1_free)

        ###################
        # evaluate residual
        ###################
        Rx = np.zeros(self.nx)

        ####################
        # kinematic equation
        ####################
        Rx[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1_free)

        ####################
        # euations of motion
        ####################
        Rx[nq : nq + nu] = Mk1 @ u_dotk1 - hk1 - W_Nk1 @ la_Nk1 - W_Fk1 @ la_Fk1

        ################
        # normal contact
        ################
        prox_arg = g_Nk1 - self.model.prox_r_N * la_Nk1
        self.I_N = prox_arg <= 0.0
        # self.I_N = g_Nk1 <= 0.0
        Rx[nq + nu : nq + nu + nla_N] = g_Nk1 - prox_R0_np(prox_arg)

        ##########
        # friction
        ##########
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                Rx[nq + nu + nla_N + i_F] = np.where(
                    self.I_N[i_N] * np.ones(len(i_F), dtype=bool),
                    -la_Fk1[i_F]
                    - prox_sphere(
                        -la_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1[i_F],
                        mu[i_N] * la_Nk1[i_N],
                    ),
                    la_Fk1[i_F],
                )

        # update quantities of new time step
        self.tk1 = tk1
        self.qk1 = qk1
        self.uk1_free = uk1_free

        return Rx

    def Ry(self, yk1):
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

        # evaluate repeatedly used quantities
        Mk1 = self.model.M(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        W_Fk1 = self.model.W_F(tk1, qk1, scipy_matrix=csr_matrix)
        g_dot = self.model.g_dot(tk1, qk1, uk1)
        gamma = self.model.gamma(tk1, qk1, uk1)
        xi_Nk1_plus = self.model.xi_N(tk1, qk1, self.uk, uk1)
        xi_Fk1_plus = self.model.xi_F(tk1, qk1, self.uk, uk1)

        ###################
        # evaluate residual
        ###################
        Ry = np.zeros(self.nu + self.nla_g + self.nla_gamma + self.nla_N + self.nla_F)

        #################
        # impact equation
        #################
        Ry[:nu] = (
            Mk1 @ Uk1
            - W_gk1 @ La_gk1
            - W_gammak1 @ La_gammak1
            - W_Nk1 @ La_Nk1
            - W_Fk1 @ La_Fk1
        )

        # bilateral constraints
        # TODO: Are they correct?
        Ry[nu : nu + nla_g] = g_dot
        Ry[nu + nla_g : nu + nla_g + nla_gamma] = gamma

        ###################
        # normal impact law
        ###################
        # Ry[nu + nla_g + nla_gamma : nu + nla_g + nla_gamma + nla_N] = La_Nk1
        Ry[nu + nla_g + nla_gamma : nu + nla_g + nla_gamma + nla_N] = np.where(
            self.Ak1,
            xi_Nk1_plus - prox_R0_np(xi_Nk1_plus - self.model.prox_r_N * La_Nk1),
            La_Nk1,
        )

        ####################
        # tangent impact law
        ####################
        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                Ry[nu + nla_g + nla_gamma + nla_N + i_F] = np.where(
                    self.Ak1[i_N] * np.ones(len(i_F), dtype=bool),
                    -La_Fk1[i_F]
                    - prox_sphere(
                        -La_Fk1[i_F] + self.model.prox_r_F[i_N] * xi_Fk1_plus[i_F],
                        mu[i_N] * La_Nk1[i_N],
                    ),
                    La_Fk1[i_F],
                )

        return Ry

    def step(self):
        j = 0
        converged = False
        # zk1 = np.concatenate((self.la_gk, self.la_gammak, self.la_Nk, self.la_Fk))
        zk1 = np.concatenate(
            (self.la_gk, self.la_gammak, self.la_Nk, self.mu_Nk, self.la_Fk)
        )
        ck1, tk1, qk1, uk1, la_gk1, la_gammak1, la_Nk1, mu_Nk1, la_Fk1 = self.c(zk1)

        error = self.error_function(ck1)
        converged = error < self.tol
        while (not converged) and (j < self.max_iter):
            j += 1

            # compute Jacobian and make a Newton step
            c_yk1 = self.c_x(zk1)

            zk1 -= spsolve(c_yk1, ck1)

            # from scipy.sparse.linalg import lsqr
            # zk1 -= lsqr(c_yk1, ck1)[0]

            # zk1 -= np.linalg.lstsq(c_yk1.toarray(), ck1, rcond=None)[0]

            # zk1 -= spsolve(c_yk1.T @ c_yk1, c_yk1.T @ ck1)

            # # fixed-point iteration
            # # self.atol = 1.0e-4
            # # r = 7.0e-1
            # r = 4.0e-1
            # # r = 3.0e-1
            # zk1 -= r * ck1

            # check error for new Lagrange multipliers
            ck1, tk1, qk1, uk1, la_gk1, la_gammak1, la_Nk1, mu_Nk1, la_Fk1 = self.c(zk1)
            error = self.error_function(ck1)
            converged = error < self.tol

        if not converged:
            # raise RuntimeError("Internal Newton-Raphson scheme is not converged!")
            print(f"Internal Newton-Raphson scheme is not converged with error {error}")

        return (
            (converged, j, error),
            tk1,
            qk1,
            uk1,
            la_gk1,
            la_gammak1,
            la_Nk1,
            mu_Nk1,
            la_Fk1,
        )

    def step_y(self, xk1, f):
        # initial residual and error
        R = f(xk1)
        error = self.error_function(R)
        converged = error < self.tol

        # print(f"initial error: {error}")
        j = 0
        if not converged:
            while j < self.max_iter:
                # jacobian
                # J = csr_matrix(approx_fprime(xk1, f, method="2-point"))
                J = csr_matrix(approx_fprime(xk1, f, method="3-point"))

                # Newton update
                j += 1

                dx = spsolve(J, R, use_umfpack=True)

                # dx = lsqr(J, R, atol=1.0e-12, btol=1.0e-12)[0]

                # # no underflow errors
                # dx = np.linalg.lstsq(J.toarray(), R, rcond=None)[0]

                # # TODO: Can we get this sparse?
                # # using QR decomposition, see https://de.wikipedia.org/wiki/QR-Zerlegung#L%C3%B6sung_regul%C3%A4rer_oder_%C3%BCberbestimmter_Gleichungssysteme
                # b = R.copy()
                # Q, R = np.linalg.qr(J.toarray())
                # z = Q.T @ b
                # dx = np.linalg.solve(R, z)  # solving R*x = Q^T*b

                # # solve normal equation (should be independent of the conditioning
                # # number!)
                # dx = spsolve(J.T @ J, J.T @ R)

                xk1 -= dx

                R = f(xk1)
                error = self.error_function(R)
                converged = error < self.tol
                if converged:
                    break

            if not converged:
                # raise RuntimeError("internal Newton-Raphson not converged")
                print(f"not converged!")

        return converged, j, error, xk1

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

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            xk1 = self.xk.copy()
            yk1 = self.yk.copy()

            (
                (converged_x, n_iter_x, error_x),
                tk1,
                qk1,
                uk1_free,
                la_gk1,
                la_gammak1,
                la_Nk1,
                mu_Nk1,
                la_Fk1,
            ) = self.step()
            # converged_x, n_iter_x, error_x, xk1 = self.step_x()
            # converged_x, n_iter_x, error_x, xk1 = self.step(xk1, lambda x: self.c(x)[0])

            # converged_x, n_iter_x, error_x, xk1 = self.step(xk1, self.Rx)

            # update quantities of new time step
            self.tk1 = tk1
            self.qk1 = qk1
            self.uk1_free = uk1_free

            converged_y, n_iter_y, error_y, yk1 = self.step_y(yk1, self.Ry)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; ||R_x||: {error_y:0.2e} ({n_iter_x}/{self.max_iter}); ||R_y||: {error_x:0.2e} ({n_iter_y}/{self.max_iter})"
            )
            if not (converged_x and converged_y):
                print(
                    f"internal Newton-Raphson method not converged after {n_iter_x} x-steps with error: {error_x:.5e}"
                )
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
                    la_N=np.array(la_N),
                    La_N=np.array(La_N),
                    P_N=np.array(P_N),
                    la_F=np.array(la_F),
                    La_F=np.array(La_F),
                    P_F=np.array(P_F),
                )

            Uk1, La_gk1, La_gammak1, La_Nk1, La_Fk1 = self.unpack_y(yk1)

            uk1 = uk1_free + Uk1

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            # a.append((uk1 - self.uk) / self.dt)
            U.append(Uk1)
            la_g.append(la_gk1)
            La_g.append(La_gk1)
            P_g.append(self.dt * la_gk1 + La_gk1)
            la_gamma.append(la_gammak1)
            La_gamma.append(La_gammak1)
            P_gamma.append(self.dt * la_gammak1 + La_gammak1)
            la_N.append(la_Nk1)
            La_N.append(La_Nk1)
            P_N.append(self.dt * la_Nk1 + La_Nk1)
            la_F.append(la_Fk1)
            La_F.append(La_Fk1)
            P_F.append(self.dt * la_Fk1 + La_Fk1)

            # update local variables for accepted time step
            self.tk = tk1

            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.la_gk = la_gk1.copy()
            self.la_gammak = la_gammak1.copy()
            self.la_Nk = la_Nk1.copy()
            self.mu_Nk = mu_Nk1.copy()
            self.la_Fk = la_Fk1.copy()

            self.xk = xk1.copy()
            self.yk = yk1.copy()

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
