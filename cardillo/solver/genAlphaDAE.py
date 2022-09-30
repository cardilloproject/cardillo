import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, bmat, eye, block_diag
from tqdm import tqdm

from cardillo.math import approx_fprime
from cardillo.solver import Solution

# GGL2 = True
GGL2 = False


class GenAlphaFirstOrderGGL2_V1:
    """Generalized alpha solver for first order ODE's with two stages of GGL stabilization."""

    def __init__(
        self,
        model,
        t1,
        dt,
        rho_inf=1,
        tol=1e-10,
        max_iter=40,
        error_function=lambda x: np.max(np.abs(x)),
        # numerical_jacobian=False,
        numerical_jacobian=True,
        # unknowns="positions",
        unknowns="velocities",
        # unknowns="auxiliary",
    ):

        self.model = model
        self.unknowns = unknowns

        #######################################################################
        # integration time
        #######################################################################
        self.t0 = t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        #######################################################################
        # gen alpha parameter
        #######################################################################
        self.rho_inf = rho_inf
        self.alpha_m = (3.0 * rho_inf - 1.0) / (2.0 * (rho_inf + 1.0))
        self.alpha_f = rho_inf / (rho_inf + 1.0)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m

        self.gamma_prime = dt * self.gamma
        self.alpha_prime = (1.0 - self.alpha_m) / (1.0 - self.alpha_f)
        self.eta = self.alpha_prime / self.gamma_prime

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
        self.nx = self.ny = self.nq + self.nu  # dimension of the state space
        self.ns = self.nx + self.nla_g + self.nla_gamma  # vector of unknowns
        self.ns += 2 * self.nla_g + self.nla_gamma + self.nu  # GGL2 parts

        if numerical_jacobian:
            self.__R_gen = self.__R_gen_num
        else:
            self.__R_gen = self.__R_gen_analytic

        def initial_values(t0, q0, u0):
            # initial velocites
            q_dot0 = self.model.q_dot(t0, q0, u0)

            # solve for consistent initial accelerations and Lagrange mutlipliers
            M0 = self.model.M(t0, q0, scipy_matrix=csr_matrix)
            h0 = self.model.h(t0, q0, u0)
            W_g0 = self.model.W_g(t0, q0, scipy_matrix=csr_matrix)
            W_gamma0 = self.model.W_gamma(t0, q0, scipy_matrix=csr_matrix)
            zeta_g0 = self.model.zeta_g(t0, q0, u0)
            zeta_gamma0 = self.model.zeta_gamma(t0, q0, u0)
            A = bmat(
                [
                    [M0, -W_g0, -W_gamma0],
                    [W_g0.T, None, None],
                    [W_gamma0.T, None, None],
                ],
                format="csc",
            )
            b = np.concatenate([h0, -zeta_g0, -zeta_gamma0])
            u_dot_la_g_la_gamma = spsolve(A, b)
            u_dot0 = u_dot_la_g_la_gamma[: self.nu]
            la_g0 = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
            la_gamma0 = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

            x0 = np.concatenate((q0, u0))
            x_dot0 = np.concatenate((q_dot0, u_dot0))
            y0 = x_dot0.copy()  # TODO: Use perturbed values foun din Arnold2015
            if self.unknowns == "positions":
                a0 = u_dot0.copy()
                mu_g0 = np.zeros(self.nla_g)
                kappa_g0 = np.zeros(self.nla_g)
                kappa_gamma0 = np.zeros(self.nla_gamma)
                s0 = self.pack(x0, la_g0, la_gamma0, mu_g0, kappa_g0, kappa_gamma0, a0)
            elif self.unknowns == "velocities":
                a0 = u_dot0.copy()
                mu_g0 = np.zeros(self.nla_g)
                kappa_g0 = np.zeros(self.nla_g)
                kappa_gamma0 = np.zeros(self.nla_gamma)
                s0 = self.pack(
                    x_dot0, la_g0, la_gamma0, mu_g0, kappa_g0, kappa_gamma0, a0
                )
            elif self.unknowns == "auxiliary":
                a0 = u_dot0.copy()
                mu_g0 = np.zeros(self.nla_g)
                kappa_g0 = np.zeros(self.nla_g)
                kappa_gamma0 = np.zeros(self.nla_gamma)
                s0 = self.pack(y0, la_g0, la_gamma0, mu_g0, kappa_g0, kappa_gamma0, a0)
            else:
                raise RuntimeError("Wrong set of unknowns chosen!")

            self.tk = t0
            self.qk = q0
            self.uk = u0
            self.q_dotk = q_dot0
            self.u_dotk = u_dot0
            self.la_gk = la_g0
            self.la_gammak = la_gamma0
            self.mu_gk = mu_g0
            self.kappa_gk = kappa_g0
            self.kappa_gammak = kappa_gamma0
            self.ak = a0
            self.xk = x0
            self.x_dotk = x_dot0
            self.yk = y0
            self.sk = s0

        # compute consistent initial conditions
        initial_values(t0, model.q0, model.u0)

        # check if initial conditions satisfy constraints on position, velocity
        # and acceleration level
        g0 = model.g(self.tk, self.qk)
        g_dot0 = model.g_dot(self.tk, self.qk, self.uk)
        g_ddot0 = model.g_ddot(self.tk, self.qk, self.uk, self.u_dotk)
        gamma0 = model.gamma(self.tk, self.qk, self.uk)
        gamma_dot0 = model.gamma_dot(self.tk, self.qk, self.uk, self.u_dotk)

        # TODO: These tolerances should be used defined. Maybe all these
        #       initial computations and checks should be moved to a
        #       SolverOptions object ore something similar?
        rtol = 1.0e-5
        atol = 1.0e-5

        assert np.allclose(
            g0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g0!"
        assert np.allclose(
            g_dot0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g_dot0!"
        assert np.allclose(
            g_ddot0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g_ddot0!"
        assert np.allclose(
            gamma0, np.zeros(self.nla_gamma)
        ), "Initial conditions do not fulfill gamma0!"
        assert np.allclose(
            gamma_dot0, np.zeros(self.nla_gamma), rtol, atol
        ), "Initial conditions do not fulfill gamma_dot0!"

    def update(self, sk1, store=False):
        """Update dependent variables."""
        nq = self.nq
        nu = self.nu
        ny = self.ny

        # constants
        dt = self.dt
        gamma = self.gamma
        alpha_f = self.alpha_f
        alpha_m = self.alpha_m

        if self.unknowns == "positions":
            xk1 = sk1[:ny]
            yk1 = (xk1 - self.xk) / (dt * gamma) - ((1.0 - gamma) / gamma) * self.yk
            x_dotk1 = (
                alpha_m * self.yk + (1.0 - alpha_m) * yk1 - alpha_f * self.x_dotk
            ) / (1.0 - alpha_f)
        elif self.unknowns == "velocities":
            x_dotk1 = sk1[:ny]
            yk1 = (
                alpha_f * self.x_dotk + (1.0 - alpha_f) * x_dotk1 - alpha_m * self.yk
            ) / (1.0 - alpha_m)
            xk1 = self.xk + dt * ((1.0 - gamma) * self.yk + gamma * yk1)
        elif self.unknowns == "auxiliary":
            yk1 = sk1[:ny]
            xk1 = self.xk + dt * ((1.0 - gamma) * self.yk + gamma * yk1)
            x_dotk1 = (
                alpha_m * self.yk + (1.0 - alpha_m) * yk1 - alpha_f * self.x_dotk
            ) / (1.0 - alpha_f)

        if store:
            self.sk = sk1.copy()
            self.xk = xk1.copy()
            self.x_dotk = x_dotk1.copy()
            self.yk = yk1.copy()

        # extract generaliezd coordinates and velocities
        qk1 = xk1[:nq]
        uk1 = xk1[nq : nq + nu]
        q_dotk1 = x_dotk1[:nq]
        u_dotk1 = x_dotk1[nq : nq + nu]
        return qk1, uk1, q_dotk1, u_dotk1

    def pack(self, *args):
        return np.concatenate([*args])

    def unpack(self, s):
        nq = self.nq
        nu = self.nu
        nx = self.nx
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        q = s[:nq]
        u = s[nq:nx]
        la_g = s[nx : nq + nu + nla_g]
        la_gamma = s[nx + nla_g : nx + nla_g + nla_gamma]
        mu_g = s[nx + nla_g + nla_gamma : nx + 2 * nla_g + nla_gamma]
        kappa_g = s[nx + 2 * nla_g + nla_gamma : nx + 3 * nla_g + nla_gamma]
        kappa_gamma = s[nx + 3 * nla_g + nla_gamma : nx + 3 * nla_g + 2 * nla_gamma]
        a = s[nx + 3 * nla_g + 2 * nla_gamma :]
        return q, u, la_g, la_gamma, mu_g, kappa_g, kappa_gamma, a

    def __R_gen_num(self, tk1, sk1):
        yield self.__R(tk1, sk1)
        yield csr_matrix(self.__J_num(tk1, sk1))

    def __R_gen_analytic(self, tk1, sk1):
        nq = self.nq
        nx = self.nx
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        # extract Lagrange multipliers and acceleration
        (
            _,
            _,
            la_gk1,
            la_gammak1,
            mu_gk1,
            kappa_gk1,
            kappa_gammak1,
            ak1,
        ) = self.unpack(sk1)

        # update dependent variables
        qk1, uk1, q_dotk1, u_dotk1 = self.update(sk1, store=False)

        # evaluate repeated used quantities
        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)

        ###################
        # evaluate residual
        ###################
        R = np.zeros(self.ns)

        # kinematic differential equation
        g_qk1 = self.model.g_q(tk1, qk1)
        R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1) - g_qk1.T @ mu_gk1

        # equations of motion
        R[nq:nx] = (
            # Mk1 @ ak1
            Mk1 @ u_dotk1  # This works, but why?
            # Mk1 @ (u_dotk1 + ak1)
            - self.model.h(tk1, qk1, uk1)
            - W_gk1 @ la_gk1
            - W_gammak1 @ la_gammak1
        )

        # bilateral constraints
        R[nx : nx + nla_g] = self.model.g_ddot(tk1, qk1, uk1, ak1)
        R[nx + nla_g : nx + nla_g + nla_gamma] = self.model.gamma_dot(
            tk1, qk1, uk1, ak1
        )
        # R[nx : nx + nla_g] = self.model.g_ddot(tk1, qk1, uk1, u_dotk1 + ak1)
        # R[nx + nla_g : nx + nla_g + nla_gamma] = self.model.gamma_dot(
        #     tk1, qk1, uk1, u_dotk1 + ak1
        # )

        R[nx + nla_g + nla_gamma : nx + 2 * nla_g + nla_gamma] = self.model.g_dot(
            tk1, qk1, uk1
        )
        R[nx + 2 * nla_g + nla_gamma : nx + 3 * nla_g + nla_gamma] = self.model.g(
            tk1, qk1
        )

        R[
            nx + 3 * nla_g + nla_gamma : nx + 3 * nla_g + 2 * nla_gamma
        ] = self.model.gamma(tk1, qk1, uk1)

        R[nx + 3 * nla_g + 2 * nla_gamma :] = (
            ak1
            - u_dotk1
            - W_gk1 @ kappa_gk1
            - W_gammak1 @ kappa_gammak1
            # ak1 - W_gk1 @ kappa_gk1 - W_gammak1 @ kappa_gammak1
        )

        yield R

        raise NotImplementedError

        ###################
        # evaluate jacobian
        ###################
        eye_nq = eye(self.nq)
        A = self.model.q_dot_q(tk1, qk1, uk1)
        Bk1 = self.model.B(tk1, qk1, scipy_matrix=csr_matrix)
        K = (
            self.model.Mu_q(tk1, qk1, u_dotk1)
            - self.model.h_q(tk1, qk1, uk1)
            - self.model.Wla_g_q(tk1, qk1, la_gk1)
            - self.model.Wla_gamma_q(tk1, qk1, la_gammak1)
        )
        h_uk1 = self.model.h_u(tk1, qk1, uk1)
        g_qk1 = self.model.g_q(tk1, qk1)
        gamma_qk1 = self.model.gamma_q(tk1, qk1, uk1)

        # sparse assemble global tangent matrix
        if self.GGL:
            g_dot_qk1 = self.model.g_dot_q(tk1, qk1, uk1)
            C = A + self.model.g_q_T_mu_g(tk1, qk1, mu_gk1)
            if self.unknowns == "positions":
                eta = self.eta
                # fmt: off
                J = bmat(
                    [
                        [eta * eye_nq - C,              -Bk1,   None,       None, -g_qk1.T],
                        [               K, eta * Mk1 - h_uk1, -W_gk1, -W_gammak1,     None],
                        [       g_dot_qk1,           W_gk1.T,   None,       None,     None],
                        [       gamma_qk1,       W_gammak1.T,   None,       None,     None],
                        [           g_qk1,              None,   None,       None,     None],
                    ],
                    format="csr",
                )
                # fmt: on
            elif self.unknowns == "velocities":
                etap = 1.0 / self.eta
                # fmt: off
                J = bmat(
                    [
                        [eye_nq - etap * C,        -etap * Bk1,   None,       None, -g_qk1.T],
                        [         etap * K, Mk1 - etap * h_uk1, -W_gk1, -W_gammak1,     None],
                        [ etap * g_dot_qk1,     etap * W_gk1.T,   None,       None,     None],
                        [ etap * gamma_qk1, etap * W_gammak1.T,   None,       None,     None],
                        [     etap * g_qk1,               None,   None,       None,     None],
                    ],
                    format="csr",
                )
                # fmt: on
            elif self.unknowns == "auxiliary":
                gap = self.gamma_prime
                alp = self.alpha_prime
                # fmt: off
                J = bmat(
                    [
                        [alp * eye_nq - gap * C,              -gap * Bk1,   None,       None, -g_qk1.T],
                        [               gap * K, alp * Mk1 - gap * h_uk1, -W_gk1, -W_gammak1,     None],
                        [       gap * g_dot_qk1,           gap * W_gk1.T,   None,       None,     None],
                        [       gap * gamma_qk1,       gap * W_gammak1.T,   None,       None,     None],
                        [           gap * g_qk1,                    None,   None,       None,     None],
                    ],
                    format="csr",
                )
                # fmt: on
        else:
            if self.unknowns == "positions":
                eta = self.eta
                if self.DAE_index == 3:
                    # fmt: off
                    J = bmat(
                        [
                            [eta * eye_nq - A,              -Bk1,   None,       None],
                            [               K, eta * Mk1 - h_uk1, -W_gk1, -W_gammak1],
                            [           g_qk1,              None,   None,       None],
                            [       gamma_qk1,       W_gammak1.T,   None,       None],
                        ],
                        format="csr",
                    )
                    # fmt: on
                elif self.DAE_index == 2:
                    raise NotImplementedError
                elif self.DAE_index == 1:
                    raise NotImplementedError
            elif self.unknowns == "velocities":
                etap = 1.0 / self.eta
                if self.DAE_index == 3:
                    # fmt: off
                    J = bmat(
                        [
                            [eye_nq - etap * A,        -etap * Bk1,   None,       None],
                            [         etap * K, Mk1 - etap * h_uk1, -W_gk1, -W_gammak1],
                            [     etap * g_qk1,               None,   None,       None],
                            [ etap * gamma_qk1, etap * W_gammak1.T,   None,       None],
                        ],
                        format="csr",
                    )
                    # fmt: on
                else:
                    raise NotImplementedError
            elif self.unknowns == "auxiliary":
                gap = self.gamma_prime
                alp = self.alpha_prime
                if self.DAE_index == 3:
                    # fmt: off
                    J = bmat(
                        [
                            [alp * eye_nq - gap * A,              -gap * Bk1,   None,       None],
                            [               gap * K, alp * Mk1 - gap * h_uk1, -W_gk1, -W_gammak1],
                            [           gap * g_qk1,                    None,   None,       None],
                            [       gap * gamma_qk1,       gap * W_gammak1.T,   None,       None],
                        ],
                        format="csr",
                    )
                    # fmt: on
                else:
                    raise NotImplementedError

        # # TODO: Keep this for debugging!
        # J_num = self.__J_num(tk1, sk1)
        # diff = (J - J_num).toarray()
        # error = np.linalg.norm(diff)
        # print(f"error J: {error}")

        yield J

    def __R(self, tk1, sk1):
        return next(self.__R_gen_analytic(tk1, sk1))

    def __J_num(self, tk1, sk1):
        return csr_matrix(
            approx_fprime(sk1, lambda x: self.__R(tk1, x), method="2-point")
        )

    def step(self, tk1, sk1):
        # initial residual and error
        R_gen = self.__R_gen(tk1, sk1)
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
                ds = spsolve(R_x, R, use_umfpack=True)
                sk1 -= ds
                R_gen = self.__R_gen(tk1, sk1)
                R = next(R_gen)

                error = self.error_function(R)
                converged = error < self.tol
                if converged:
                    break

        return converged, j, error, sk1

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        q_dot = [self.q_dotk]
        u_dot = [self.u_dotk]
        la_g = [self.la_gk]
        la_gamma = [self.la_gammak]
        mu_g = [self.mu_gk]
        kappa_g = [self.kappa_gk]
        kappa_gamma = [self.kappa_gammak]
        a = [self.ak]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            sk1 = self.sk.copy()  # This copy is mandatory since we modify sk1
            # in the step function
            converged, n_iter, error, sk1 = self.step(tk1, sk1)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )
            if not converged:
                raise RuntimeError(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )

            # update dependent variables
            qk1, uk1, q_dotk1, u_dotk1 = self.update(sk1, store=True)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # extract Lagrange multipliers
            (
                _,
                _,
                la_gk1,
                la_gammak1,
                mu_gk1,
                kappa_gk1,
                kappa_gammak1,
                ak1,
            ) = self.unpack(sk1)

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            q_dot.append(q_dotk1)
            u_dot.append(u_dotk1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)
            mu_g.append(mu_gk1)
            kappa_g.append(kappa_gk1)
            kappa_gamma.append(kappa_gammak1)
            a.append(ak1)

            # update local variables for accepted time step
            self.tk = tk1
            self.sk = sk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            q_dot=np.array(q_dot),
            u_dot=np.array(u_dot),
            la_g=np.array(la_g),
            la_gamma=np.array(la_gamma),
            mu_g=np.array(mu_g),
            kappa_g=np.array(kappa_g),
            kappa_gamma=np.array(kappa_gamma),
            a=np.array(a),
        )


class GenAlphaFirstOrderGGL2_V2:
    """Generalized alpha solver for first order ODE's with two stages of GGL stabilization."""

    def __init__(
        self,
        model,
        t1,
        dt,
        rho_inf=1,
        tol=1e-10,
        max_iter=40,
        error_function=lambda x: np.max(np.abs(x)),
        numerical_jacobian=True,
        # unknowns="positions",
        # unknowns="velocities",
        unknowns="auxiliary",
    ):

        self.model = model
        self.unknowns = unknowns

        #######################################################################
        # integration time
        #######################################################################
        self.t0 = t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        #######################################################################
        # gen alpha parameter
        #######################################################################
        self.rho_inf = rho_inf
        self.alpha_m = (3.0 * rho_inf - 1.0) / (2.0 * (rho_inf + 1.0))
        self.alpha_f = rho_inf / (rho_inf + 1.0)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m

        self.gamma_prime = dt * self.gamma
        self.alpha_prime = (1.0 - self.alpha_m) / (1.0 - self.alpha_f)
        self.eta = self.alpha_prime / self.gamma_prime

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
        self.nx = self.ny = self.nq + self.nu  # dimension of the state space
        self.ns = self.nx + self.nla_g + self.nla_gamma  # vector of unknowns
        self.ns += 2 * self.nla_g + self.nla_gamma + self.nx  # GGL2 parts

        if numerical_jacobian:
            self.__R_gen = self.__R_gen_num
        else:
            self.__R_gen = self.__R_gen_analytic

        def initial_values(t0, q0, u0):
            # initial velocites
            q_dot0 = self.model.q_dot(t0, q0, u0)

            # solve for consistent initial accelerations and Lagrange mutlipliers
            M0 = self.model.M(t0, q0, scipy_matrix=csr_matrix)
            h0 = self.model.h(t0, q0, u0)
            W_g0 = self.model.W_g(t0, q0, scipy_matrix=csr_matrix)
            W_gamma0 = self.model.W_gamma(t0, q0, scipy_matrix=csr_matrix)
            zeta_g0 = self.model.zeta_g(t0, q0, u0)
            zeta_gamma0 = self.model.zeta_gamma(t0, q0, u0)
            A = bmat(
                [
                    [M0, -W_g0, -W_gamma0],
                    [W_g0.T, None, None],
                    [W_gamma0.T, None, None],
                ],
                format="csc",
            )
            b = np.concatenate([h0, -zeta_g0, -zeta_gamma0])
            u_dot_la_g_la_gamma = spsolve(A, b)
            u_dot0 = u_dot_la_g_la_gamma[: self.nu]
            la_g0 = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
            la_gamma0 = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

            x0 = np.concatenate((q0, u0))
            x_dot0 = np.concatenate((q_dot0, u_dot0))
            y0 = x_dot0.copy()  # TODO: Use perturbed values foun din Arnold2015
            # v0 = q_dot0.copy() # stabilization for q_dot
            # a0 = u_dot0.copy() # stabilization for u_dot
            v0 = np.zeros(self.nq)  # stabilization for q_dot
            a0 = np.zeros(self.nu)  # stabilization for u_dot
            if self.unknowns == "positions":
                mu_g0 = np.zeros(self.nla_g)
                kappa_g0 = np.zeros(self.nla_g)
                kappa_gamma0 = np.zeros(self.nla_gamma)
                s0 = self.pack(
                    x0, la_g0, la_gamma0, mu_g0, kappa_g0, kappa_gamma0, v0, a0
                )
            elif self.unknowns == "velocities":
                mu_g0 = np.zeros(self.nla_g)
                kappa_g0 = np.zeros(self.nla_g)
                kappa_gamma0 = np.zeros(self.nla_gamma)
                s0 = self.pack(
                    x_dot0, la_g0, la_gamma0, mu_g0, kappa_g0, kappa_gamma0, v0, a0
                )
            elif self.unknowns == "auxiliary":
                mu_g0 = np.zeros(self.nla_g)
                kappa_g0 = np.zeros(self.nla_g)
                kappa_gamma0 = np.zeros(self.nla_gamma)
                s0 = self.pack(
                    y0, la_g0, la_gamma0, mu_g0, kappa_g0, kappa_gamma0, v0, a0
                )
            else:
                raise RuntimeError("Wrong set of unknowns chosen!")

            self.tk = t0
            self.qk = q0
            self.uk = u0
            self.q_dotk = q_dot0
            self.u_dotk = u_dot0
            self.la_gk = la_g0
            self.la_gammak = la_gamma0
            self.mu_gk = mu_g0
            self.kappa_gk = kappa_g0
            self.kappa_gammak = kappa_gamma0
            self.vk = v0
            self.ak = a0
            self.xk = x0
            self.x_dotk = x_dot0
            self.yk = y0
            self.sk = s0

        # compute consistent initial conditions
        initial_values(t0, model.q0, model.u0)

        # check if initial conditions satisfy constraints on position, velocity
        # and acceleration level
        g0 = model.g(self.tk, self.qk)
        g_dot0 = model.g_dot(self.tk, self.qk, self.uk)
        g_ddot0 = model.g_ddot(self.tk, self.qk, self.uk, self.u_dotk)
        gamma0 = model.gamma(self.tk, self.qk, self.uk)
        gamma_dot0 = model.gamma_dot(self.tk, self.qk, self.uk, self.u_dotk)

        # TODO: These tolerances should be used defined. Maybe all these
        #       initial computations and checks should be moved to a
        #       SolverOptions object ore something similar?
        rtol = 1.0e-5
        atol = 1.0e-5

        assert np.allclose(
            g0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g0!"
        assert np.allclose(
            g_dot0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g_dot0!"
        assert np.allclose(
            g_ddot0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g_ddot0!"
        assert np.allclose(
            gamma0, np.zeros(self.nla_gamma)
        ), "Initial conditions do not fulfill gamma0!"
        assert np.allclose(
            gamma_dot0, np.zeros(self.nla_gamma), rtol, atol
        ), "Initial conditions do not fulfill gamma_dot0!"

    def update(self, sk1, store=False):
        """Update dependent variables."""
        nq = self.nq
        nu = self.nu
        nx = self.nx
        ny = self.ny

        # constants
        dt = self.dt
        gamma = self.gamma
        alpha_f = self.alpha_f
        alpha_m = self.alpha_m

        Xk1 = sk1[-nx:]  # stabilization part
        if self.unknowns == "positions":
            raise NotImplementedError
            xk1 = sk1[:ny]
            yk1 = (xk1 - self.xk) / (dt * gamma) - ((1.0 - gamma) / gamma) * self.yk
            # yk1 = (xk1 + Xk1 - self.xk) / (dt * gamma) - ((1.0 - gamma) / gamma) * self.yk
            x_dotk1 = (
                alpha_m * self.yk + (1.0 - alpha_m) * yk1 - alpha_f * self.x_dotk
            ) / (1.0 - alpha_f) + Xk1
        elif self.unknowns == "velocities":
            raise NotImplementedError
            x_dotk1 = sk1[:ny]
            # yk1 = (
            #     alpha_f * self.x_dotk + (1.0 - alpha_f) * x_dotk1 - alpha_m * self.yk
            # ) / (1.0 - alpha_m)
            yk1 = (
                alpha_f * self.x_dotk
                + (1.0 - alpha_f) * (x_dotk1 + Xk1)
                - alpha_m * self.yk
            ) / (1.0 - alpha_m)
            xk1 = self.xk + dt * ((1.0 - gamma) * self.yk + gamma * yk1)  # + Xk1
        elif self.unknowns == "auxiliary":
            yk1 = sk1[:ny]
            xk1 = self.xk + dt * ((1.0 - gamma) * self.yk + gamma * yk1) + Xk1
            x_dotk1 = (
                alpha_m * self.yk + (1.0 - alpha_m) * yk1 - alpha_f * self.x_dotk
            ) / (1.0 - alpha_f)

        if store:
            self.sk = sk1.copy()
            self.xk = xk1.copy()
            self.x_dotk = x_dotk1.copy()
            self.yk = yk1.copy()

        # extract generaliezd coordinates and velocities
        qk1 = xk1[:nq]
        uk1 = xk1[nq : nq + nu]
        q_dotk1 = x_dotk1[:nq]
        u_dotk1 = x_dotk1[nq : nq + nu]
        return qk1, uk1, q_dotk1, u_dotk1

    def pack(self, *args):
        return np.concatenate([*args])

    def unpack(self, s):
        nq = self.nq
        nu = self.nu
        nx = self.nx
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        q = s[:nq]
        u = s[nq:nx]
        la_g = s[nx : nq + nu + nla_g]
        la_gamma = s[nx + nla_g : nx + nla_g + nla_gamma]
        mu_g = s[nx + nla_g + nla_gamma : nx + 2 * nla_g + nla_gamma]
        kappa_g = s[nx + 2 * nla_g + nla_gamma : nx + 3 * nla_g + nla_gamma]
        kappa_gamma = s[nx + 3 * nla_g + nla_gamma : nx + 3 * nla_g + 2 * nla_gamma]
        v = s[nx + 3 * nla_g + 2 * nla_gamma : nx + 3 * nla_g + 2 * nla_gamma + nq]
        a = s[nx + 3 * nla_g + 2 * nla_gamma + nq :]
        return q, u, la_g, la_gamma, mu_g, kappa_g, kappa_gamma, v, a

    def __R_gen_num(self, tk1, sk1):
        yield self.__R(tk1, sk1)
        yield csr_matrix(self.__J_num(tk1, sk1))

    def __R_gen_analytic(self, tk1, sk1):
        nq = self.nq
        nx = self.nx
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        # extract Lagrange multipliers and acceleration
        (
            _,
            _,
            la_gk1,
            la_gammak1,
            mu_gk1,
            kappa_gk1,
            kappa_gammak1,
            vk1,
            ak1,
        ) = self.unpack(sk1)

        # update dependent variables
        qk1, uk1, q_dotk1, u_dotk1 = self.update(sk1, store=False)

        # evaluate repeated used quantities
        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)

        ###################
        # evaluate residual
        ###################
        R = np.zeros(self.ns)

        # kinematic differential equation
        g_qk1 = self.model.g_q(tk1, qk1)
        # R[:nq] = q_dotk1 + vk1 - self.model.q_dot(tk1, qk1, uk1) #- g_qk1.T @ mu_gk1
        R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1)

        # equations of motion
        R[nq:nx] = (
            # Mk1 @ ak1
            Mk1 @ u_dotk1  # This works, but why?
            # Mk1 @ (u_dotk1 + ak1)
            - self.model.h(tk1, qk1, uk1)
            - W_gk1 @ la_gk1
            - W_gammak1 @ la_gammak1
        )

        # bilateral constraints
        R[nx : nx + nla_g] = self.model.g_ddot(tk1, qk1, uk1, u_dotk1)
        R[nx + nla_g : nx + nla_g + nla_gamma] = self.model.gamma_dot(
            tk1, qk1, uk1, u_dotk1
        )
        # R[nx : nx + nla_g] = self.model.g_ddot(tk1, qk1, uk1, ak1)
        # R[nx + nla_g : nx + nla_g + nla_gamma] = self.model.gamma_dot(
        #     tk1, qk1, uk1, ak1
        # )

        R[nx + nla_g + nla_gamma : nx + 2 * nla_g + nla_gamma] = self.model.g_dot(
            tk1, qk1, uk1
        )
        R[nx + 2 * nla_g + nla_gamma : nx + 3 * nla_g + nla_gamma] = self.model.g(
            tk1, qk1
        )

        R[
            nx + 3 * nla_g + nla_gamma : nx + 3 * nla_g + 2 * nla_gamma
        ] = self.model.gamma(tk1, qk1, uk1)

        # R[nx + 3 * nla_g + 2 * nla_gamma :] = (
        #     ak1
        #     - u_dotk1
        #     - W_gk1 @ kappa_gk1
        #     - W_gammak1 @ kappa_gammak1
        #     # ak1 - W_gk1 @ kappa_gk1 - W_gammak1 @ kappa_gammak1
        # )

        R[nx + 3 * nla_g + 2 * nla_gamma :] = np.concatenate(
            [
                vk1 - g_qk1.T @ mu_gk1,
                Mk1 @ ak1 - W_gk1 @ kappa_gk1 - W_gammak1 @ kappa_gammak1
                # q_dotk1 + vk1 - g_qk1.T @ mu_gk1,
                # Mk1 @ (u_dotk1 + ak1) - W_gk1 @ kappa_gk1 - W_gammak1 @ kappa_gammak1
            ]
        )

        yield R

        raise NotImplementedError

        ###################
        # evaluate jacobian
        ###################
        eye_nq = eye(self.nq)
        A = self.model.q_dot_q(tk1, qk1, uk1)
        Bk1 = self.model.B(tk1, qk1, scipy_matrix=csr_matrix)
        K = (
            self.model.Mu_q(tk1, qk1, u_dotk1)
            - self.model.h_q(tk1, qk1, uk1)
            - self.model.Wla_g_q(tk1, qk1, la_gk1)
            - self.model.Wla_gamma_q(tk1, qk1, la_gammak1)
        )
        h_uk1 = self.model.h_u(tk1, qk1, uk1)
        g_qk1 = self.model.g_q(tk1, qk1)
        gamma_qk1 = self.model.gamma_q(tk1, qk1, uk1)

        # sparse assemble global tangent matrix
        if self.GGL:
            g_dot_qk1 = self.model.g_dot_q(tk1, qk1, uk1)
            C = A + self.model.g_q_T_mu_g(tk1, qk1, mu_gk1)
            if self.unknowns == "positions":
                eta = self.eta
                # fmt: off
                J = bmat(
                    [
                        [eta * eye_nq - C,              -Bk1,   None,       None, -g_qk1.T],
                        [               K, eta * Mk1 - h_uk1, -W_gk1, -W_gammak1,     None],
                        [       g_dot_qk1,           W_gk1.T,   None,       None,     None],
                        [       gamma_qk1,       W_gammak1.T,   None,       None,     None],
                        [           g_qk1,              None,   None,       None,     None],
                    ],
                    format="csr",
                )
                # fmt: on
            elif self.unknowns == "velocities":
                etap = 1.0 / self.eta
                # fmt: off
                J = bmat(
                    [
                        [eye_nq - etap * C,        -etap * Bk1,   None,       None, -g_qk1.T],
                        [         etap * K, Mk1 - etap * h_uk1, -W_gk1, -W_gammak1,     None],
                        [ etap * g_dot_qk1,     etap * W_gk1.T,   None,       None,     None],
                        [ etap * gamma_qk1, etap * W_gammak1.T,   None,       None,     None],
                        [     etap * g_qk1,               None,   None,       None,     None],
                    ],
                    format="csr",
                )
                # fmt: on
            elif self.unknowns == "auxiliary":
                gap = self.gamma_prime
                alp = self.alpha_prime
                # fmt: off
                J = bmat(
                    [
                        [alp * eye_nq - gap * C,              -gap * Bk1,   None,       None, -g_qk1.T],
                        [               gap * K, alp * Mk1 - gap * h_uk1, -W_gk1, -W_gammak1,     None],
                        [       gap * g_dot_qk1,           gap * W_gk1.T,   None,       None,     None],
                        [       gap * gamma_qk1,       gap * W_gammak1.T,   None,       None,     None],
                        [           gap * g_qk1,                    None,   None,       None,     None],
                    ],
                    format="csr",
                )
                # fmt: on
        else:
            if self.unknowns == "positions":
                eta = self.eta
                if self.DAE_index == 3:
                    # fmt: off
                    J = bmat(
                        [
                            [eta * eye_nq - A,              -Bk1,   None,       None],
                            [               K, eta * Mk1 - h_uk1, -W_gk1, -W_gammak1],
                            [           g_qk1,              None,   None,       None],
                            [       gamma_qk1,       W_gammak1.T,   None,       None],
                        ],
                        format="csr",
                    )
                    # fmt: on
                elif self.DAE_index == 2:
                    raise NotImplementedError
                elif self.DAE_index == 1:
                    raise NotImplementedError
            elif self.unknowns == "velocities":
                etap = 1.0 / self.eta
                if self.DAE_index == 3:
                    # fmt: off
                    J = bmat(
                        [
                            [eye_nq - etap * A,        -etap * Bk1,   None,       None],
                            [         etap * K, Mk1 - etap * h_uk1, -W_gk1, -W_gammak1],
                            [     etap * g_qk1,               None,   None,       None],
                            [ etap * gamma_qk1, etap * W_gammak1.T,   None,       None],
                        ],
                        format="csr",
                    )
                    # fmt: on
                else:
                    raise NotImplementedError
            elif self.unknowns == "auxiliary":
                gap = self.gamma_prime
                alp = self.alpha_prime
                if self.DAE_index == 3:
                    # fmt: off
                    J = bmat(
                        [
                            [alp * eye_nq - gap * A,              -gap * Bk1,   None,       None],
                            [               gap * K, alp * Mk1 - gap * h_uk1, -W_gk1, -W_gammak1],
                            [           gap * g_qk1,                    None,   None,       None],
                            [       gap * gamma_qk1,       gap * W_gammak1.T,   None,       None],
                        ],
                        format="csr",
                    )
                    # fmt: on
                else:
                    raise NotImplementedError

        # # TODO: Keep this for debugging!
        # J_num = self.__J_num(tk1, sk1)
        # diff = (J - J_num).toarray()
        # error = np.linalg.norm(diff)
        # print(f"error J: {error}")

        yield J

    def __R(self, tk1, sk1):
        return next(self.__R_gen_analytic(tk1, sk1))

    def __J_num(self, tk1, sk1):
        return csr_matrix(
            approx_fprime(sk1, lambda x: self.__R(tk1, x), method="2-point")
        )

    def step(self, tk1, sk1):
        # initial residual and error
        R_gen = self.__R_gen(tk1, sk1)
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
                ds = spsolve(R_x, R, use_umfpack=True)
                sk1 -= ds
                R_gen = self.__R_gen(tk1, sk1)
                R = next(R_gen)

                error = self.error_function(R)
                converged = error < self.tol
                if converged:
                    break

        return converged, j, error, sk1

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        q_dot = [self.q_dotk]
        u_dot = [self.u_dotk]
        la_g = [self.la_gk]
        la_gamma = [self.la_gammak]
        mu_g = [self.mu_gk]
        kappa_g = [self.kappa_gk]
        kappa_gamma = [self.kappa_gammak]
        v = [self.vk]
        a = [self.ak]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            sk1 = self.sk.copy()  # This copy is mandatory since we modify sk1
            # in the step function
            converged, n_iter, error, sk1 = self.step(tk1, sk1)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )
            if not converged:
                raise RuntimeError(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )

            # update dependent variables
            qk1, uk1, q_dotk1, u_dotk1 = self.update(sk1, store=True)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # extract Lagrange multipliers
            (
                _,
                _,
                la_gk1,
                la_gammak1,
                mu_gk1,
                kappa_gk1,
                kappa_gammak1,
                vk1,
                ak1,
            ) = self.unpack(sk1)

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            q_dot.append(q_dotk1)
            u_dot.append(u_dotk1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)
            mu_g.append(mu_gk1)
            kappa_g.append(kappa_gk1)
            kappa_gamma.append(kappa_gammak1)
            v.append(vk1)
            a.append(ak1)

            # update local variables for accepted time step
            self.tk = tk1
            self.sk = sk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            q_dot=np.array(q_dot),
            u_dot=np.array(u_dot),
            la_g=np.array(la_g),
            la_gamma=np.array(la_gamma),
            mu_g=np.array(mu_g),
            kappa_g=np.array(kappa_g),
            kappa_gamma=np.array(kappa_gamma),
            v=np.array(v),
            a=np.array(a),
        )


class GenAlphaFirstOrderGGL2_V3:
    """Generalized alpha solver for first order ODE's with two stages of GGL stabilization."""

    def __init__(
        self,
        model,
        t1,
        dt,
        rho_inf=1,
        tol=1e-10,
        max_iter=40,
        error_function=lambda x: np.max(np.abs(x)),
        numerical_jacobian=True,
        # unknowns="positions",
        unknowns="velocities",
        # unknowns="auxiliary",
    ):

        self.model = model
        self.unknowns = unknowns
        assert unknowns in [
            "positions",
            "velocities",
            "auxiliary",
        ], f'wrong set of unknowns "{unknowns}" chosen!'

        #######################################################################
        # integration time
        #######################################################################
        self.t0 = t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        #######################################################################
        # gen alpha parameter
        #######################################################################
        self.rho_inf = rho_inf
        self.alpha_m = (3.0 * rho_inf - 1.0) / (2.0 * (rho_inf + 1.0))
        self.alpha_f = rho_inf / (rho_inf + 1.0)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m

        self.gamma_prime = dt * self.gamma
        self.alpha_prime = (1.0 - self.alpha_m) / (1.0 - self.alpha_f)
        self.eta = self.alpha_prime / self.gamma_prime

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
        self.nx = self.nq + self.nu  # dimension of the state space
        self.ns = self.nx + self.nla_g + self.nla_gamma  # vector of unknowns
        self.ns += 2 * self.nla_g + self.nla_gamma + self.nu  # GGL2 parts

        if numerical_jacobian:
            self.__R_gen = self.__R_gen_num
        else:
            self.__R_gen = self.__R_gen_analytic

        def initial_values(t0, q0, u0):
            # initial velocites
            q_dot0 = self.model.q_dot(t0, q0, u0)

            # solve for consistent initial accelerations and Lagrange mutlipliers
            M0 = self.model.M(t0, q0, scipy_matrix=csr_matrix)
            h0 = self.model.h(t0, q0, u0)
            W_g0 = self.model.W_g(t0, q0, scipy_matrix=csr_matrix)
            W_gamma0 = self.model.W_gamma(t0, q0, scipy_matrix=csr_matrix)
            zeta_g0 = self.model.zeta_g(t0, q0, u0)
            zeta_gamma0 = self.model.zeta_gamma(t0, q0, u0)
            A = bmat(
                [
                    [M0, -W_g0, -W_gamma0],
                    [W_g0.T, None, None],
                    [W_gamma0.T, None, None],
                ],
                format="csc",
            )
            b = np.concatenate([h0, -zeta_g0, -zeta_gamma0])
            u_dot_la_g_la_gamma = spsolve(A, b)
            u_dot0 = u_dot_la_g_la_gamma[: self.nu]
            la_g0 = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
            la_gamma0 = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

            x0 = np.concatenate((q0, u0))
            x_dot0 = np.concatenate((q_dot0, u_dot0))
            y0 = x_dot0.copy()  # TODO: Use perturbed values foun din Arnold2015
            # a0 = u_dot0.copy()  # stabilized accelerations
            a0 = np.zeros_like(u_dot0)  # u_S
            mu_g0 = np.zeros(self.nla_g)
            kappa_g0 = np.zeros(self.nla_g)
            kappa_gamma0 = np.zeros(self.nla_gamma)
            if self.unknowns == "positions":
                s0 = self.pack(x0, la_g0, la_gamma0, mu_g0, kappa_g0, kappa_gamma0, a0)
            elif self.unknowns == "velocities":
                s0 = self.pack(
                    x_dot0, la_g0, la_gamma0, mu_g0, kappa_g0, kappa_gamma0, a0
                )
            elif self.unknowns == "auxiliary":
                s0 = self.pack(y0, la_g0, la_gamma0, mu_g0, kappa_g0, kappa_gamma0, a0)

            self.tk = t0
            self.qk = q0
            self.uk = u0
            self.q_dotk = q_dot0
            self.u_dotk = u_dot0
            self.la_gk = la_g0
            self.la_gammak = la_gamma0
            self.mu_gk = mu_g0
            self.kappa_gk = kappa_g0
            self.kappa_gammak = kappa_gamma0
            self.ak = a0
            self.xk = x0
            self.x_dotk = x_dot0
            self.yk = y0
            self.sk = s0

        # compute consistent initial conditions
        initial_values(t0, model.q0, model.u0)

        # check if initial conditions satisfy constraints on position, velocity
        # and acceleration level
        g0 = model.g(self.tk, self.qk)
        g_dot0 = model.g_dot(self.tk, self.qk, self.uk)
        g_ddot0 = model.g_ddot(self.tk, self.qk, self.uk, self.u_dotk)
        gamma0 = model.gamma(self.tk, self.qk, self.uk)
        gamma_dot0 = model.gamma_dot(self.tk, self.qk, self.uk, self.u_dotk)

        # TODO: These tolerances should be used defined. Maybe all these
        #       initial computations and checks should be moved to a
        #       SolverOptions object ore something similar?
        rtol = 1.0e-5
        atol = 1.0e-5

        assert np.allclose(
            g0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g0!"
        assert np.allclose(
            g_dot0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g_dot0!"
        assert np.allclose(
            g_ddot0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g_ddot0!"
        assert np.allclose(
            gamma0, np.zeros(self.nla_gamma)
        ), "Initial conditions do not fulfill gamma0!"
        assert np.allclose(
            gamma_dot0, np.zeros(self.nla_gamma), rtol, atol
        ), "Initial conditions do not fulfill gamma_dot0!"

    def update(self, sk1, store=False):
        """Update dependent variables."""
        nq = self.nq
        nu = self.nu
        nx = self.nx

        # constants
        dt = self.dt
        gamma = self.gamma
        alpha_f = self.alpha_f
        alpha_m = self.alpha_m

        if self.unknowns == "positions":
            xk1 = sk1[:nx]
            yk1 = (xk1 - self.xk) / (dt * gamma) - ((1.0 - gamma) / gamma) * self.yk
            x_dotk1 = (
                alpha_m * self.yk + (1.0 - alpha_m) * yk1 - alpha_f * self.x_dotk
            ) / (1.0 - alpha_f)
        elif self.unknowns == "velocities":
            x_dotk1 = sk1[:nx]
            yk1 = (
                alpha_f * self.x_dotk + (1.0 - alpha_f) * x_dotk1 - alpha_m * self.yk
            ) / (1.0 - alpha_m)
            xk1 = self.xk + dt * ((1.0 - gamma) * self.yk + gamma * yk1)
        elif self.unknowns == "auxiliary":
            yk1 = sk1[:nx]
            xk1 = self.xk + dt * ((1.0 - gamma) * self.yk + gamma * yk1)
            x_dotk1 = (
                alpha_m * self.yk + (1.0 - alpha_m) * yk1 - alpha_f * self.x_dotk
            ) / (1.0 - alpha_f)

        if store:
            self.sk = sk1.copy()
            self.xk = xk1.copy()
            self.x_dotk = x_dotk1.copy()
            self.yk = yk1.copy()

        # extract generaliezd coordinates and velocities
        qk1 = xk1[:nq]
        uk1 = xk1[nq : nq + nu]
        q_dotk1 = x_dotk1[:nq]
        u_dotk1 = x_dotk1[nq : nq + nu]
        return qk1, uk1, q_dotk1, u_dotk1

    def pack(self, *args):
        return np.concatenate([*args])

    def unpack(self, s):
        nq = self.nq
        nu = self.nu
        nx = self.nx
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        q = s[:nq]
        u = s[nq:nx]
        la_g = s[nx : nq + nu + nla_g]
        la_gamma = s[nx + nla_g : nx + nla_g + nla_gamma]
        mu_g = s[nx + nla_g + nla_gamma : nx + 2 * nla_g + nla_gamma]
        kappa_g = s[nx + 2 * nla_g + nla_gamma : nx + 3 * nla_g + nla_gamma]
        kappa_gamma = s[nx + 3 * nla_g + nla_gamma : nx + 3 * nla_g + 2 * nla_gamma]
        a = s[nx + 3 * nla_g + 2 * nla_gamma :]
        return q, u, la_g, la_gamma, mu_g, kappa_g, kappa_gamma, a

    def __R_gen_num(self, tk1, sk1):
        yield self.__R(tk1, sk1)
        yield csr_matrix(self.__J_num(tk1, sk1))

    def __R_gen_analytic(self, tk1, sk1):
        nq = self.nq
        nx = self.nx
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        # extract Lagrange multipliers and acceleration
        (
            _,
            _,
            la_gk1,
            la_gammak1,
            mu_gk1,
            kappa_gk1,
            kappa_gammak1,
            ak1,
        ) = self.unpack(sk1)

        # Note: ak1 takes the role of u_s here!
        u_sk1 = ak1

        # update dependent variables
        qk1, uk1, q_dotk1, u_dotk1 = self.update(sk1, store=False)

        # evaluate repeated used quantities
        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        g_qk1 = self.model.g_q(tk1, qk1)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)

        ###################
        # evaluate residual
        ###################
        R = np.zeros(self.ns)

        ####################################
        # version with absolute acceleration
        ####################################

        # kinematic differential equation
        # R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1) - g_qk1.T @ mu_gk1
        R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1 + u_sk1) - g_qk1.T @ mu_gk1

        # equations of motion
        # R[nq:nx] = (
        #     Mk1 @ ak1
        #     - self.model.h(tk1, qk1, uk1)
        #     - W_gk1 @ la_gk1
        #     - W_gammak1 @ la_gammak1
        # )
        R[nq:nx] = (
            Mk1 @ u_dotk1
            - self.model.h(tk1, qk1, uk1 + u_sk1)
            - W_gk1 @ la_gk1
            - W_gammak1 @ la_gammak1
        )

        # bilateral constraints on acceleration level
        # (corresponds to constraint forces la_gk1, la_gammak1)
        # R[nx : nx + nla_g] = self.model.g_ddot(tk1, qk1, uk1, ak1)
        # R[nx + nla_g : nx + nla_g + nla_gamma] = self.model.gamma_dot(
        #     tk1, qk1, uk1, ak1
        # )
        R[nx : nx + nla_g] = self.model.g_ddot(tk1, qk1, uk1 + u_sk1, u_dotk1)
        R[nx + nla_g : nx + nla_g + nla_gamma] = self.model.gamma_dot(
            tk1, qk1, uk1 + u_sk1, u_dotk1
        )

        # bilateral constraints on position level
        # (correspsonds to position correction mu_g)
        R[nx + nla_g + nla_gamma : nx + 2 * nla_g + nla_gamma] = self.model.g(tk1, qk1)

        # bilateral constraints on velocity level
        # (correspsonds to velocity correction kappa_gk1)
        # R[nx + 2 * nla_g + nla_gamma : nx + 3 * nla_g + nla_gamma] = self.model.g_dot(
        #     tk1, qk1, uk1
        # )
        R[nx + 2 * nla_g + nla_gamma : nx + 3 * nla_g + nla_gamma] = self.model.g_dot(
            tk1, qk1, uk1 + u_sk1
        )

        # bilateral constraints on velocity level
        # (gamma; corresponds to velocity correction kappa_gammak1)
        # R[
        #     nx + 3 * nla_g + nla_gamma : nx + 3 * nla_g + 2 * nla_gamma
        # ] = self.model.gamma(tk1, qk1, uk1)
        R[
            nx + 3 * nla_g + nla_gamma : nx + 3 * nla_g + 2 * nla_gamma
        ] = self.model.gamma(tk1, qk1, uk1 + u_sk1)

        # stabilization on velocity level
        # R[nx + 3 * nla_g + 2 * nla_gamma :] = (
        #     Mk1 @ (ak1 - u_dotk1) - W_gk1 @ kappa_gk1 - W_gammak1 @ kappa_gammak1
        # )
        # Note: ak1 takes the role of u_s here!
        R[nx + 3 * nla_g + 2 * nla_gamma :] = (
            Mk1 @ ak1 - W_gk1 @ kappa_gk1 - W_gammak1 @ kappa_gammak1
        )

        # ####################################
        # # version with relative acceleration
        # ####################################

        # # kinematic differential equation
        # g_qk1 = self.model.g_q(tk1, qk1)
        # R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1) - g_qk1.T @ mu_gk1

        # # equations of motion
        # R[nq:nx] = (
        #     Mk1 @ (u_dotk1 + ak1)
        #     - self.model.h(tk1, qk1, uk1)
        #     - W_gk1 @ la_gk1
        #     - W_gammak1 @ la_gammak1
        # )

        # # bilateral constraints on acceleration level
        # # (corresponds to constraint forces la_gk1, la_gammak1)
        # R[nx : nx + nla_g] = self.model.g_ddot(tk1, qk1, uk1, u_dotk1 + ak1)
        # R[nx + nla_g : nx + nla_g + nla_gamma] = self.model.gamma_dot(
        #     tk1, qk1, uk1, u_dotk1 + ak1
        # )

        # # bilateral constraints on position level
        # # (correspsonds to position correction mu_g)
        # R[nx + nla_g + nla_gamma : nx + 2 * nla_g + nla_gamma] = self.model.g(tk1, qk1)

        # # bilateral constraints on velocity level
        # # (correspsonds to velocity correction kappa_gk1)
        # R[nx + 2 * nla_g + nla_gamma : nx + 3 * nla_g + nla_gamma] = self.model.g_dot(
        #     tk1, qk1, uk1
        # )

        # # bilateral constraints on velocity level
        # # (gamma; correspsonds to velocity correction kappa_gammak1)
        # R[
        #     nx + 3 * nla_g + nla_gamma : nx + 3 * nla_g + 2 * nla_gamma
        # ] = self.model.gamma(tk1, qk1, uk1)

        # # stabilization on velocity level
        # R[nx + 3 * nla_g + 2 * nla_gamma :] = (
        #     Mk1 @ ak1 - W_gk1 @ kappa_gk1 - W_gammak1 @ kappa_gammak1
        # )

        yield R

        raise NotImplementedError

    def __R(self, tk1, sk1):
        return next(self.__R_gen_analytic(tk1, sk1))

    def __J_num(self, tk1, sk1):
        return csr_matrix(
            approx_fprime(sk1, lambda x: self.__R(tk1, x), method="2-point")
        )

    def step(self, tk1, sk1):
        # initial residual and error
        R_gen = self.__R_gen(tk1, sk1)
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
                ds = spsolve(R_x, R, use_umfpack=True)
                sk1 -= ds
                R_gen = self.__R_gen(tk1, sk1)
                R = next(R_gen)

                error = self.error_function(R)
                converged = error < self.tol
                if converged:
                    break

        return converged, j, error, sk1

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        # u = [self.uk]
        u = [self.uk + self.ak]
        q_dot = [self.q_dotk]
        u_dot = [self.u_dotk]
        la_g = [self.la_gk]
        la_gamma = [self.la_gammak]
        mu_g = [self.mu_gk]
        kappa_g = [self.kappa_gk]
        kappa_gamma = [self.kappa_gammak]
        a = [self.ak]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            sk1 = self.sk.copy()  # This copy is mandatory since we modify sk1
            # in the step function
            converged, n_iter, error, sk1 = self.step(tk1, sk1)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )
            if not converged:
                raise RuntimeError(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )

            # update dependent variables
            qk1, uk1, q_dotk1, u_dotk1 = self.update(sk1, store=True)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # extract Lagrange multipliers
            (
                _,
                _,
                la_gk1,
                la_gammak1,
                mu_gk1,
                kappa_gk1,
                kappa_gammak1,
                ak1,
            ) = self.unpack(sk1)

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            # u.append(uk1)
            u.append(uk1 + ak1)
            q_dot.append(q_dotk1)
            u_dot.append(u_dotk1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)
            mu_g.append(mu_gk1)
            kappa_g.append(kappa_gk1)
            kappa_gamma.append(kappa_gammak1)
            a.append(ak1)

            # update local variables for accepted time step
            self.tk = tk1
            self.sk = sk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            q_dot=np.array(q_dot),
            u_dot=np.array(u_dot),
            la_g=np.array(la_g),
            la_gamma=np.array(la_gamma),
            mu_g=np.array(mu_g),
            kappa_g=np.array(kappa_g),
            kappa_gamma=np.array(kappa_gamma),
            a=np.array(a),
        )


##########################################################################
# Subsequently follow the new implementations for the generalized alpha
# scheme for constrained mechanical systems in first and second order form
##########################################################################


class GeneralizedAlphaFirstOrderGGLGiuseppe:
    def __init__(
        self,
        model,
        t1,
        dt,
        rho_inf=1,
        tol=1e-10,
        max_iter=40,
        error_function=lambda x: np.max(np.abs(x)),
        # numerical_jacobian=False,
        numerical_jacobian=True,
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
        # gen alpha parameter
        #######################################################################
        self.rho_inf = rho_inf
        self.alpha_m = (3.0 * rho_inf - 1.0) / (2.0 * (rho_inf + 1.0))
        self.alpha_f = rho_inf / (rho_inf + 1.0)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m

        # self.gamma_prime = dt * self.gamma
        # self.alpha_prime = (1.0 - self.alpha_m) / (1.0 - self.alpha_f)
        # self.eta = self.alpha_prime / self.gamma_prime

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
        self.nx = self.ny = self.nq + self.nu  # dimension of the state space
        self.ns = (
            2 * self.nx + 3 * self.nla_g + 2 * self.nla_gamma
        )  # vector of unknowns

        if numerical_jacobian:
            self.__R_gen = self.__R_gen_num
        else:
            self.__R_gen = self.__R_gen_analytic

        #################
        # preconditioning
        #################
        def consistent_initial_values(t0, q0, u0):
            """compute physically consistent initial values"""

            # initial velocites
            q_dot0 = self.model.q_dot(t0, q0, u0)

            # solve for consistent initial accelerations and Lagrange mutlipliers
            M0 = self.model.M(t0, q0, scipy_matrix=csr_matrix)
            h0 = self.model.h(t0, q0, u0)
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
            # fmt: on
            b = np.concatenate([h0, -zeta_g0, -zeta_gamma0])
            u_dot_la_g_la_gamma = spsolve(A, b)

            u_dot0 = u_dot_la_g_la_gamma[: self.nu]

            la_g0 = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
            la_gamma0 = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

            mu_g0 = np.zeros(self.nla_g)
            kappa_g0 = np.zeros(self.nla_g)
            kappa_gamma0 = np.zeros(self.nla_gamma)

            v0 = q_dot0.copy()
            a0 = u_dot0.copy()

            # TODO: I think we have to solve for initial conditions such that g, g_dot, g_ddot, gamma and gamma_dot are fulfilled. This requries the solution of a nonlinear system corresponding to the residual of the twice stabilized GGL scheme. This might remvoe the oszillations from the beginning!

            return (
                q_dot0,
                u_dot0,
                v0,
                a0,
                la_g0,
                mu_g0,
                kappa_g0,
                la_gamma0,
                kappa_gamma0,
            )

        # compute consistent initial conditions
        q0 = model.q0
        u0 = model.u0
        (
            q_dot0,
            u_dot0,
            v0,
            a0,
            la_g0,
            mu_g0,
            kappa_g0,
            la_gamma0,
            kappa_gamma0,
        ) = consistent_initial_values(t0, q0, u0)

        self.tk = t0
        self.qk = q0
        self.uk = u0
        self.q_dotk = q_dot0
        self.u_dotk = u_dot0
        self.la_gk = la_g0
        self.mu_gk = mu_g0
        self.kappa_gk = kappa_g0
        self.la_gammak = la_gamma0
        self.kappa_gammak = kappa_gamma0
        self.xk = np.concatenate((q0, u0))
        self.x_dotk = np.concatenate((q_dot0, u_dot0))
        self.yk = self.xk.copy()  # TODO: Use better values!
        self.sk = np.concatenate(
            (q_dot0, u_dot0, v0, a0, la_g0, mu_g0, kappa_g0, la_gamma0, kappa_gamma0)
        )

        # TODO: Where do we have to satisfy the initial conditions? This has to coincide with the constraint equation evaluation of the scheme!
        # check if initial conditions satisfy constraints on position, velocity
        # and acceleration level
        g0 = model.g(t0, q0)
        g_dot0 = model.g_dot(t0, q0, u0)  # TODO: Use g_q @ v + g_t
        g_ddot0 = model.g_ddot(t0, q0, u0, a0)
        gamma0 = model.gamma(t0, q0, u0)  # TODO: Use gamma_q @ v + ???
        gamma_dot0 = model.gamma_dot(t0, q0, u0, a0)

        # TODO: These tolerances should be used defined. Maybe all these
        #       initial computations and checks should be moved to a
        #       SolverOptions object ore something similar?
        rtol = 1.0e-10
        atol = 1.0e-10

        assert np.allclose(
            g0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g0!"
        assert np.allclose(
            g_dot0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g_dot0!"
        assert np.allclose(
            g_ddot0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g_ddot0!"
        assert np.allclose(
            gamma0, np.zeros(self.nla_gamma)
        ), "Initial conditions do not fulfill gamma0!"
        assert np.allclose(
            gamma_dot0, np.zeros(self.nla_gamma), rtol, atol
        ), "Initial conditions do not fulfill gamma_dot0!"

    def update(self, sk1, store=False):
        """Update dependent variables."""
        nq = self.nq
        nu = self.nu
        ny = self.ny

        # constants
        dt = self.dt
        gamma = self.gamma
        alpha_f = self.alpha_f
        alpha_m = self.alpha_m

        x_dotk1 = sk1[:ny]
        yk1 = (
            alpha_f * self.x_dotk + (1.0 - alpha_f) * x_dotk1 - alpha_m * self.yk
        ) / (1.0 - alpha_m)
        xk1 = self.xk + dt * ((1.0 - gamma) * self.yk + gamma * yk1)

        if store:
            self.sk = sk1.copy()
            self.xk = xk1.copy()
            self.x_dotk = x_dotk1.copy()
            self.yk = yk1.copy()

        # extract generaliezd coordinates and velocities
        qk1 = xk1[:nq]
        uk1 = xk1[nq : nq + nu]
        q_dotk1 = x_dotk1[:nq]
        u_dotk1 = x_dotk1[nq : nq + nu]
        return qk1, uk1, q_dotk1, u_dotk1

    def pack(self, *args):
        return np.concatenate([*args])

    def unpack(self, s):
        nq = self.nq
        nu = self.nu
        nx = self.nx
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        q_dot = s[:nq]
        u_dot = s[nq:nx]
        v = s[nx : nx + nq]
        a = s[nx + nq : 2 * nx]
        la_g = s[2 * nx : 2 * nx + nla_g]
        mu_g = s[2 * nx + nla_g : 2 * nx + 2 * nla_g]
        kappa_g = s[2 * nx + 2 * nla_g : 2 * nx + 3 * nla_g]
        la_gamma = s[2 * nx + 3 * nla_g : 2 * nx + 3 * nla_g + nla_gamma]
        kappa_gamma = s[
            2 * nx + 3 * nla_g + nla_gamma : 2 * nx + 3 * nla_g + 2 * nla_gamma
        ]

        return q_dot, u_dot, v, a, la_g, mu_g, kappa_g, la_gamma, kappa_gamma

    def __R_gen_num(self, tk1, sk1):
        yield self.__R(tk1, sk1)
        yield csr_matrix(self.__J_num(tk1, sk1))

    def __R_gen_analytic(self, tk1, sk1):
        nq = self.nq
        nx = self.nx
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        # extract Lagrange multiplier
        (
            _,
            _,
            vk1,
            ak1,
            la_gk1,
            mu_gk1,
            kappa_gk1,
            la_gammak1,
            kappa_gammak1,
        ) = self.unpack(sk1)

        # update dependent variables
        qk1, uk1, q_dotk1, u_dotk1 = self.update(sk1, store=False)

        # evaluate repeated used quantities
        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)

        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)

        g_qk1 = self.model.g_q(tk1, qk1)
        gamma_qk1 = self.model.gamma_q(tk1, qk1, vk1)
        # gamma_qk1 = self.model.gamma_q(tk1, qk1, uk1)

        ###################
        # evaluate residual
        ###################
        R = np.zeros(self.ns)

        # kinematic differential equation
        R[:nq] = vk1 - self.model.q_dot(tk1, qk1, uk1)

        # equations of motion
        R[nq:nx] = (
            Mk1 @ ak1
            - self.model.h(tk1, qk1, uk1)
            - W_gk1 @ la_gk1
            - W_gammak1 @ la_gammak1
        )

        # stabilized derivatives
        R[nx : nx + nq] = vk1 - q_dotk1 - g_qk1.T @ mu_gk1
        R[nx + nq : 2 * nx] = (
            ak1 - u_dotk1 - W_gk1 @ kappa_gk1 - W_gammak1 @ kappa_gammak1
        )

        # bilateral constraints
        R[2 * nx : 2 * nx + nla_g] = self.model.g(tk1, qk1)
        R[2 * nx + nla_g : 2 * nx + 2 * nla_g] = g_qk1 @ vk1
        # + self.model.g_dot(
        #     tk1, qk1, np.zeros(self.nu)
        # )
        R[2 * nx + 2 * nla_g : 2 * nx + 3 * nla_g] = self.model.g_ddot(
            tk1, qk1, uk1, ak1
        )

        # R[
        #     2 * nx + 3 * nla_g : 2 * nx + 3 * nla_g + nla_gamma
        # ] = gamma_qk1 @ vk1
        # + self.model.gamma(
        #     tk1, qk1, np.zeros(self.nu)
        # )  # TODO: affine part
        # R[
        #     2 * nx + 3 * nla_g : 2 * nx + 3 * nla_g + nla_gamma
        # ] = W_gammak1.T @ uk1 #+ self.model.gamma(tk1, qk1, np.zeros(self.nu))
        R[2 * nx + 3 * nla_g : 2 * nx + 3 * nla_g + nla_gamma] = self.model.gamma(
            tk1, qk1, uk1
        )
        R[
            2 * nx + 3 * nla_g + nla_gamma : 2 * nx + 3 * nla_g + 2 * nla_gamma
        ] = self.model.gamma_dot(tk1, qk1, uk1, ak1)
        # R[2 * nx + 3 * nla_g + nla_gamma: 2 * nx + 3 * nla_g + 2 * nla_gamma] = W_gammak1.T @ ak1 #+ gamma_qk1 @ vk1 + self.model.gamma_dot(tk1, qk1, np.zeros(self.nu), np.zeros(self.nu))

        # R[2 * nx + 3 * nla_g: 2 * nx + 3 * nla_g + nla_gamma] = self.model.gamma(tk1, qk1, uk1)
        # R[2 * nx + 3 * nla_g + nla_gamma: 2 * nx + 3 * nla_g + 2 * nla_gamma] = self.model.gamma_dot(tk1, qk1, uk1, ak1)

        yield R

        raise NotImplementedError

        ###################
        # evaluate jacobian
        ###################
        eye_nq = eye(self.nq)
        A = self.model.q_dot_q(tk1, qk1, uk1)
        Bk1 = self.model.B(tk1, qk1, scipy_matrix=csr_matrix)
        K = (
            self.model.Mu_q(tk1, qk1, u_dotk1)
            - self.model.h_q(tk1, qk1, uk1)
            - self.model.Wla_g_q(tk1, qk1, la_gk1)
            - self.model.Wla_gamma_q(tk1, qk1, la_gammak1)
        )
        h_uk1 = self.model.h_u(tk1, qk1, uk1)
        g_qk1 = self.model.g_q(tk1, qk1)
        gamma_qk1 = self.model.gamma_q(tk1, qk1, uk1)

        # sparse assemble global tangent matrix
        if self.GGL:
            g_dot_qk1 = self.model.g_dot_q(tk1, qk1, uk1)
            C = A + self.model.g_q_T_mu_g(tk1, qk1, mu_gk1)
            if self.unknowns == "positions":
                eta = self.eta
                # fmt: off
                J = bmat(
                    [
                        [eta * eye_nq - C,              -Bk1,   None,       None, -g_qk1.T],
                        [               K, eta * Mk1 - h_uk1, -W_gk1, -W_gammak1,     None],
                        [       g_dot_qk1,           W_gk1.T,   None,       None,     None],
                        [       gamma_qk1,       W_gammak1.T,   None,       None,     None],
                        [           g_qk1,              None,   None,       None,     None],
                    ],
                    format="csr",
                )
                # fmt: on
            elif self.unknowns == "velocities":
                etap = 1.0 / self.eta
                # fmt: off
                J = bmat(
                    [
                        [eye_nq - etap * C,        -etap * Bk1,   None,       None, -g_qk1.T],
                        [         etap * K, Mk1 - etap * h_uk1, -W_gk1, -W_gammak1,     None],
                        [ etap * g_dot_qk1,     etap * W_gk1.T,   None,       None,     None],
                        [ etap * gamma_qk1, etap * W_gammak1.T,   None,       None,     None],
                        [     etap * g_qk1,               None,   None,       None,     None],
                    ],
                    format="csr",
                )
                # fmt: on
            elif self.unknowns == "auxiliary":
                gap = self.gamma_prime
                alp = self.alpha_prime
                # fmt: off
                J = bmat(
                    [
                        [alp * eye_nq - gap * C,              -gap * Bk1,   None,       None, -g_qk1.T],
                        [               gap * K, alp * Mk1 - gap * h_uk1, -W_gk1, -W_gammak1,     None],
                        [       gap * g_dot_qk1,           gap * W_gk1.T,   None,       None,     None],
                        [       gap * gamma_qk1,       gap * W_gammak1.T,   None,       None,     None],
                        [           gap * g_qk1,                    None,   None,       None,     None],
                    ],
                    format="csr",
                )
                # fmt: on
        else:
            if self.unknowns == "positions":
                eta = self.eta
                if self.DAE_index == 3:
                    # fmt: off
                    J = bmat(
                        [
                            [eta * eye_nq - A,              -Bk1,   None,       None],
                            [               K, eta * Mk1 - h_uk1, -W_gk1, -W_gammak1],
                            [           g_qk1,              None,   None,       None],
                            [       gamma_qk1,       W_gammak1.T,   None,       None],
                        ],
                        format="csr",
                    )
                    # fmt: on
                elif self.DAE_index == 2:
                    raise NotImplementedError
                elif self.DAE_index == 1:
                    raise NotImplementedError
            elif self.unknowns == "velocities":
                etap = 1.0 / self.eta
                if self.DAE_index == 3:
                    # fmt: off
                    J = bmat(
                        [
                            [eye_nq - etap * A,        -etap * Bk1,   None,       None],
                            [         etap * K, Mk1 - etap * h_uk1, -W_gk1, -W_gammak1],
                            [     etap * g_qk1,               None,   None,       None],
                            [ etap * gamma_qk1, etap * W_gammak1.T,   None,       None],
                        ],
                        format="csr",
                    )
                    # fmt: on
                else:
                    raise NotImplementedError
            elif self.unknowns == "auxiliary":
                gap = self.gamma_prime
                alp = self.alpha_prime
                if self.DAE_index == 3:
                    # fmt: off
                    J = bmat(
                        [
                            [alp * eye_nq - gap * A,              -gap * Bk1,   None,       None],
                            [               gap * K, alp * Mk1 - gap * h_uk1, -W_gk1, -W_gammak1],
                            [           gap * g_qk1,                    None,   None,       None],
                            [       gap * gamma_qk1,       gap * W_gammak1.T,   None,       None],
                        ],
                        format="csr",
                    )
                    # fmt: on
                else:
                    raise NotImplementedError

        # # TODO: Keep this for debugging!
        # J_num = self.__J_num(tk1, sk1)
        # diff = (J - J_num).toarray()
        # error = np.linalg.norm(diff)
        # print(f"error J: {error}")

        yield J

    def __R(self, tk1, sk1):
        return next(self.__R_gen_analytic(tk1, sk1))

    def __J_num(self, tk1, sk1):
        return csr_matrix(
            approx_fprime(sk1, lambda x: self.__R(tk1, x), method="2-point")
        )

    def step(self, tk1, sk1):
        # initial residual and error
        R_gen = self.__R_gen(tk1, sk1)
        R = next(R_gen)
        error = self.error_function(R)
        converged = error < self.tol
        j = 0
        if not converged:
            while j < self.max_iter:
                # jacobian
                J = next(R_gen)

                # Newton update
                j += 1
                ds = spsolve(J, R, use_umfpack=True)
                sk1 -= ds

                R_gen = self.__R_gen(tk1, sk1)
                R = next(R_gen)

                error = self.error_function(R)
                converged = error < self.tol
                if converged:
                    break

        return converged, j, error, sk1

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        q_dot = [self.q_dotk]
        u_dot = [self.u_dotk]
        la_g = [self.la_gk]
        la_gamma = [self.la_gammak]

        # TODO:
        # v =
        # a =
        # mu_g = [self.mu_gk]
        # kappa_g = [self.kappa_gk]
        # kappa_gamma = [self.kappa_gammak]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            sk1 = self.sk.copy()  # This copy is mandatory since we modify sk1
            # in the step function
            converged, n_iter, error, sk1 = self.step(tk1, sk1)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )
            if not converged:
                raise RuntimeError(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )

            # extract Lagrange multipliers
            (
                _,
                _,
                vk1,
                ak1,
                la_gk1,
                mu_gk1,
                kappa_gk1,
                la_gammak1,
                kappa_gammak1,
            ) = self.unpack(sk1)

            # update dependent variables
            qk1, uk1, q_dotk1, u_dotk1 = self.update(sk1, store=True)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            # q_dot.append(q_dotk1)
            # u_dot.append(u_dotk1)
            q_dot.append(vk1)
            u_dot.append(ak1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)
            # if self.GGL == 1:
            #     mu_g.append(mu_gk1)

            # update local variables for accepted time step
            self.tk = tk1
            self.sk = sk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            q_dot=np.array(q_dot),
            u_dot=np.array(u_dot),
            la_g=np.array(la_g),
            la_gamma=np.array(la_gamma),
            # mu_g=np.array(mu_g),
        )


class GeneralizedAlphaFirstOrder:
    """Generalized alpha solver for first order ODE's.
    
    To-Do:
    -----
    * Think about preconditioning according to Arnold2008 and Bottasso2008?

    References
    ----------
    Arnold2008: https://doi.org/10.1007/s11044-007-9084-0 \\
    Bottasso2008 https://doi.org/10.1007/s11044-007-9051-9
    """

    def __init__(
        self,
        model,
        t1,
        dt,
        rho_inf=1,
        tol=1e-10,
        max_iter=40,
        error_function=lambda x: np.max(np.abs(x)),
        # numerical_jacobian=False,
        numerical_jacobian=True,
        DAE_index=3,
        # preconditioning=True,
        preconditioning=False,
        # unknowns="positions",
        unknowns="velocities",
        # unknowns="auxiliary",
        # GGL=False,
        # # GGL=True,
        GGL=0,
    ):

        self.model = model
        assert DAE_index >= 1 and DAE_index <= 3, "DAE_index hast to be in [1, 3]!"
        self.DAE_index = DAE_index
        self.unknowns = unknowns
        assert unknowns in [
            "positions",
            "velocities",
            "auxiliary",
        ], f'wrong set of unknowns "{unknowns}" chosen!'
        self.GGL = GGL
        assert GGL in [0, 1, 2], f"GGL has to be in [0, 1, 2]!"

        #######################################################################
        # integration time
        #######################################################################
        self.t0 = t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        #######################################################################
        # gen alpha parameter
        #######################################################################
        self.rho_inf = rho_inf
        self.alpha_m = (3.0 * rho_inf - 1.0) / (2.0 * (rho_inf + 1.0))
        self.alpha_f = rho_inf / (rho_inf + 1.0)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m

        self.gamma_prime = dt * self.gamma
        self.alpha_prime = (1.0 - self.alpha_m) / (1.0 - self.alpha_f)
        self.eta = self.alpha_prime / self.gamma_prime

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
        self.nx = self.ny = self.nq + self.nu  # dimension of the state space
        self.ns = self.nx + self.nla_g + self.nla_gamma  # vector of unknowns
        if self.GGL == 1:
            self.ns += self.nla_g
        if self.GGL == 2:
            self.ns += 2 * self.nla_g
            self.ns += self.nla_gamma
            self.ns += self.nu

        if numerical_jacobian:
            self.__R_gen = self.__R_gen_num
        else:
            self.__R_gen = self.__R_gen_analytic

        #################
        # preconditioning
        #################
        self.preconditioning = preconditioning
        if preconditioning:
            # raise RuntimeError("This is not working as expected!")

            # TODO: Scaling of second equation by h and solving for h u_dot
            # comes from the fact that we know that u_dot are accelerations,
            # so for having consisten units this equations has to be scaled.
            # This might not be correct in the sense of a preconditioner.
            # fmt: off
            if unknowns == "positions":
                eta = self.eta
                self.D_L = block_diag([
                    eye(self.nq) * eta,
                    eye(self.nu) * eta,
                    eye(self.nla_g),
                    eye(self.nla_gamma),
                ], format="csr")
                self.D_R = block_diag([
                    eye(self.nq),
                    eye(self.nu),
                    eye(self.nla_g) / eta,
                    eye(self.nla_gamma) / eta,
                ], format="csr")
                # self.D_L = bmat([[eye(self.nq) * eta,               None,            None,                 None],
                #                  [              None, eye(self.nu) * eta,            None,                 None],
                #                  [              None,               None, eye(self.nla_g),                 None],
                #                  [              None,               None,            None, eye(self.nla_gamma)]])
                # self.D_R = bmat([[eye(self.nq),         None,                  None,                     None],
                #                  [        None, eye(self.nu),                  None,                     None],
                #                  [        None,         None,  eye(self.nla_g) / dt,                     None],
                #                  [        None,         None,                  None, eye(self.nla_gamma) / dt]])
            else:
                raise NotImplementedError
            # fmt: on

        def initial_values(t0, q0, u0):
            # initial velocites
            q_dot0 = self.model.q_dot(t0, q0, u0)

            # solve for consistent initial accelerations and Lagrange mutlipliers
            M0 = self.model.M(t0, q0, scipy_matrix=csr_matrix)
            h0 = self.model.h(t0, q0, u0)
            W_g0 = self.model.W_g(t0, q0, scipy_matrix=csr_matrix)
            W_gamma0 = self.model.W_gamma(t0, q0, scipy_matrix=csr_matrix)
            zeta_g0 = self.model.zeta_g(t0, q0, u0)
            zeta_gamma0 = self.model.zeta_gamma(t0, q0, u0)
            A = bmat(
                [
                    [M0, -W_g0, -W_gamma0],
                    [W_g0.T, None, None],
                    [W_gamma0.T, None, None],
                ],
                format="csc",
            )
            b = np.concatenate([h0, -zeta_g0, -zeta_gamma0])
            u_dot_la_g_la_gamma = spsolve(A, b)
            u_dot0 = u_dot_la_g_la_gamma[: self.nu]
            la_g0 = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
            la_gamma0 = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

            x0 = np.concatenate((q0, u0))
            x_dot0 = np.concatenate((q_dot0, u_dot0))
            y0 = x_dot0.copy()  # TODO: Use perturbed values found in Arnold2015
            if self.unknowns == "positions":
                raise NotImplementedError
                if self.GGL:
                    mu_g0 = np.zeros(self.nla_g)
                    s0 = self.pack(x0, la_g0, la_gamma0, mu_g0)
                else:
                    s0 = self.pack(x0, la_g0, la_gamma0)
            elif self.unknowns == "velocities":
                if self.GGL == 1:
                    mu_g0 = np.zeros(self.nla_g)
                    s0 = self.pack(x_dot0, la_g0, la_gamma0, mu_g0)
                elif self.GGL == 2:
                    mu_g0 = np.zeros(self.nla_g, dtype=float)
                    kappa_g0 = np.zeros(self.nla_g, dtype=float)
                    kappa_gamma0 = np.zeros(self.nla_gamma, dtype=float)
                    U0 = np.zeros(self.nu, dtype=float)
                    s0 = self.pack(
                        x_dot0, la_g0, la_gamma0, mu_g0, kappa_g0, kappa_gamma0, U0
                    )
                else:
                    s0 = self.pack(x_dot0, la_g0, la_gamma0)
            elif self.unknowns == "auxiliary":
                raise NotImplementedError
                if self.GGL:
                    mu_g0 = np.zeros(self.nla_g)
                    s0 = self.pack(y0, la_g0, la_gamma0, mu_g0)
                else:
                    s0 = self.pack(y0, la_g0, la_gamma0)
            else:
                raise RuntimeError("Wrong set of unknowns chosen!")

            self.tk = t0
            self.qk = q0
            self.uk = u0
            self.q_dotk = q_dot0
            self.u_dotk = u_dot0
            self.la_gk = la_g0
            self.la_gammak = la_gamma0
            if self.GGL == 1:
                self.mu_gk = mu_g0
            elif self.GGL == 2:
                # self.ak = a0
                self.mu_gk = mu_g0
                self.kappa_gk = kappa_g0
                self.kappa_gk = kappa_gamma0

            self.xk = x0
            self.x_dotk = x_dot0
            self.yk = y0
            self.sk = s0

        def consistent_initial_values(t0, q0, u0):
            """compute physically consistent initial values"""

            # initial velocites
            q_dot0 = self.model.q_dot(t0, q0, u0)

            # solve for consistent initial accelerations and Lagrange mutlipliers
            M0 = self.model.M(t0, q0, scipy_matrix=csr_matrix)
            h0 = self.model.h(t0, q0, u0)
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
            # fmt: on
            b = np.concatenate([h0, -zeta_g0, -zeta_gamma0])
            u_dot_la_g_la_gamma = spsolve(A, b)
            u_dot0 = u_dot_la_g_la_gamma[: self.nu]
            la_g0 = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
            la_gamma0 = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

            return q_dot0, u_dot0, la_g0, la_gamma0

        def initial_values_Martin(t0, q0, u0):
            ##############################################
            # compute physically consistent initial values
            ##############################################
            q_dot0, u_dot0, la_g0, la_gamma0 = consistent_initial_values(t0, q0, u0)
            x0 = np.concatenate((q0, u0))
            x_dot0 = np.concatenate((q_dot0, u_dot0))

            ##################################
            # compute perturbed initial values
            ##################################
            s = 1.0e-1  # Arnold2016 p. 150 last paragraph
            Delta_alpha = self.alpha_m - self.alpha_f  # Arnold2016 (41)

            t0_plus = t0 + s * dt
            u0_plus = u0 + s * dt * u_dot0
            # q0_plus = q0 + s * dt * q_dot0
            q0_plus = q0 + s * dt * self.model.q_dot(t0, q0, u0)
            # q0_plus = q0 + s * dt * self.model.q_dot(t0, q0, u0_plus)
            # q0_plus = q0 + s * dt * self.model.q_dot(t0, q0, u0) + 0.5 * s**2 * dt**2 * self.model.q_ddot(t0, q0, u0, u_dot0)
            # q0_plus = q0 + s * dt * self.model.q_dot(
            #     t0, q0, u0_plus + s**2 * dt**2 * u_dot0 / 2
            # )
            (
                q_dot0_plus,
                u_dot0_plus,
                la_g0_plus,
                la_gamma0_plus,
            ) = consistent_initial_values(t0_plus, q0_plus, u0_plus)
            q_dot0_plus, u_dot0_plus, _, _ = consistent_initial_values(
                t0_plus, q0, u0_plus
            )
            y0_plus = np.concatenate((q_dot0_plus, u_dot0_plus))

            t0_minus = t0 - s * dt
            u0_minus = u0 - s * dt * u_dot0
            # q0_minus = q0 - s * dt * q_dot0
            q0_minus = q0 - s * dt * self.model.q_dot(t0, q0, u0)
            # q0_minus = q0 - s * dt * self.model.q_dot(t0, q0, u0_minus)
            # q0_minus = q0 - s * dt * self.model.q_dot(t0, q0, u0) - 0.5 * s**2 * dt**2 * self.model.q_ddot(t0, q0, u0, u_dot0)
            # q0_minus = q0 + self.model.q_dot(t0, q0, u0_minus - s**2 * dt**2 * u_dot0 / 2)
            (
                q_dot0_minus,
                u_dot0_minus,
                la_g0_minus,
                la_gamma0_minus,
            ) = consistent_initial_values(t0_minus, q0_minus, u0_minus)
            y0_minus = np.concatenate((q_dot0_minus, u_dot0_minus))

            y0 = x_dot0 + Delta_alpha * dt * (y0_plus - y0_minus) / (2.0 * s * dt)

            if self.unknowns == "positions":
                raise NotImplementedError
                if self.GGL:
                    mu_g0 = np.zeros(self.nla_g)
                    s0 = self.pack(x0, la_g0, la_gamma0, mu_g0)
                else:
                    s0 = self.pack(x0, la_g0, la_gamma0)
            elif self.unknowns == "velocities":
                if self.GGL == 1:
                    mu_g0 = np.zeros(self.nla_g)
                    s0 = self.pack(x_dot0, la_g0, la_gamma0, mu_g0)
                elif self.GGL == 2:
                    mu_g0 = np.zeros(self.nla_g)
                    kappa_g0 = np.zeros(self.nla_g)
                    kappa_gamma0 = np.zeros(self.nla_gamma)
                    U0 = np.zeros(self.nu)
                    s0 = self.pack(
                        x_dot0, la_g0, la_gamma0, mu_g0, kappa_g0, kappa_gamma0, U0
                    )
                else:
                    s0 = self.pack(x_dot0, la_g0, la_gamma0)
            elif self.unknowns == "auxiliary":
                raise NotImplementedError
                if self.GGL:
                    mu_g0 = np.zeros(self.nla_g)
                    s0 = self.pack(y0, la_g0, la_gamma0, mu_g0)
                else:
                    s0 = self.pack(y0, la_g0, la_gamma0)
            else:
                raise RuntimeError("Wrong set of unknowns chosen!")

            self.tk = t0
            self.qk = q0
            self.uk = u0
            self.q_dotk = q_dot0
            self.u_dotk = u_dot0
            self.la_gk = la_g0
            self.la_gammak = la_gamma0
            if self.GGL:
                self.mu_gk = mu_g0
            self.xk = x0
            self.x_dotk = x_dot0
            self.yk = y0
            self.sk = s0

        # compute consistent initial conditions
        # initial_values(t0, model.q0, model.u0)
        initial_values_Martin(t0, model.q0, model.u0)

        # check if initial conditions satisfy constraints on position, velocity
        # and acceleration level
        g0 = model.g(self.tk, self.qk)
        g_dot0 = model.g_dot(self.tk, self.qk, self.uk)
        g_ddot0 = model.g_ddot(self.tk, self.qk, self.uk, self.u_dotk)
        gamma0 = model.gamma(self.tk, self.qk, self.uk)
        gamma_dot0 = model.gamma_dot(self.tk, self.qk, self.uk, self.u_dotk)

        # TODO: These tolerances should be used defined. Maybe all these
        #       initial computations and checks should be moved to a
        #       SolverOptions object ore something similar?
        rtol = 1.0e-10
        atol = 1.0e-10

        assert np.allclose(
            g0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g0!"
        assert np.allclose(
            g_dot0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g_dot0!"
        assert np.allclose(
            g_ddot0, np.zeros(self.nla_g), rtol, atol
        ), "Initial conditions do not fulfill g_ddot0!"
        assert np.allclose(
            gamma0, np.zeros(self.nla_gamma)
        ), "Initial conditions do not fulfill gamma0!"
        assert np.allclose(
            gamma_dot0, np.zeros(self.nla_gamma), rtol, atol
        ), "Initial conditions do not fulfill gamma_dot0!"

    def update(self, sk1, store=False):
        """Update dependent variables."""
        nq = self.nq
        nu = self.nu
        ny = self.ny

        # constants
        dt = self.dt
        gamma = self.gamma
        alpha_f = self.alpha_f
        alpha_m = self.alpha_m

        if self.unknowns == "positions":
            xk1 = sk1[:ny]
            yk1 = (xk1 - self.xk) / (dt * gamma) - ((1.0 - gamma) / gamma) * self.yk
            x_dotk1 = (
                alpha_m * self.yk + (1.0 - alpha_m) * yk1 - alpha_f * self.x_dotk
            ) / (1.0 - alpha_f)
        elif self.unknowns == "velocities":
            x_dotk1 = sk1[:ny]
            yk1 = (
                alpha_f * self.x_dotk + (1.0 - alpha_f) * x_dotk1 - alpha_m * self.yk
            ) / (1.0 - alpha_m)
            xk1 = self.xk + dt * ((1.0 - gamma) * self.yk + gamma * yk1)
        elif self.unknowns == "auxiliary":
            yk1 = sk1[:ny]
            xk1 = self.xk + dt * ((1.0 - gamma) * self.yk + gamma * yk1)
            x_dotk1 = (
                alpha_m * self.yk + (1.0 - alpha_m) * yk1 - alpha_f * self.x_dotk
            ) / (1.0 - alpha_f)

        if store:
            self.sk = sk1.copy()
            self.xk = xk1.copy()
            self.x_dotk = x_dotk1.copy()
            self.yk = yk1.copy()

        # extract generaliezd coordinates and velocities
        qk1 = xk1[:nq]
        uk1 = xk1[nq : nq + nu]
        q_dotk1 = x_dotk1[:nq]
        u_dotk1 = x_dotk1[nq : nq + nu]
        return qk1, uk1, q_dotk1, u_dotk1

    def pack(self, *args):
        return np.concatenate([*args])

    def unpack(self, s):
        nq = self.nq
        nu = self.nu
        nx = self.nx
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        if self.GGL == 1:
            q = s[:nq]
            u = s[nq:nx]
            la_g = s[nx : nq + nu + nla_g]
            la_gamma = s[nx + nla_g : nx + nla_g + nla_gamma]
            mu_g = s[nx + nla_g + nla_gamma :]
            return q, u, la_g, la_gamma, mu_g
        elif self.GGL == 2:
            q = s[:nq]
            u = s[nq:nx]
            la_g = s[nx : nx + nla_g]
            la_gamma = s[nx + nla_g : nx + nla_g + nla_gamma]
            mu_g = s[nx + nla_g + nla_gamma : nx + 2 * nla_g + nla_gamma]
            kappa_g = s[nx + 2 * nla_g + nla_gamma : nx + 3 * nla_g + nla_gamma]
            kappa_gamma = s[nx + 3 * nla_g + nla_gamma : nx + 3 * nla_g + 2 * nla_gamma]
            U = s[nx + 3 * nla_g + 2 * nla_gamma :]
            return q, u, la_g, la_gamma, mu_g, kappa_g, kappa_gamma, U
        else:
            q = s[:nq]
            u = s[nq:nx]
            la_g = s[nx : nx + nla_g]
            la_gamma = s[nx + nla_g :]
            return q, u, la_g, la_gamma

    def __R_gen_num(self, tk1, sk1):
        yield self.__R(tk1, sk1)
        yield csr_matrix(self.__J_num(tk1, sk1))

    def __R_gen_analytic(self, tk1, sk1):
        nq = self.nq
        nu = self.nu
        nx = self.nx
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        # extract Lagrange multiplier
        if self.GGL == 1:
            _, _, la_gk1, la_gammak1, mu_gk1 = self.unpack(sk1)
        elif self.GGL == 2:
            (
                _,
                _,
                la_gk1,
                la_gammak1,
                mu_gk1,
                kappa_gk1,
                kappa_gammak1,
                Uk1,
            ) = self.unpack(sk1)
        else:
            _, _, la_gk1, la_gammak1 = self.unpack(sk1)

        # update dependent variables
        qk1, uk1, q_dotk1, u_dotk1 = self.update(sk1, store=False)

        # evaluate repeated used quantities
        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)

        if self.GGL == 2:
            uk1 += Uk1

        ###################
        # evaluate residual
        ###################
        R = np.zeros(self.ns)

        # kinematic differential equation
        # TODO: Use Bk1
        R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1)
        if self.GGL >= 1:
            g_qk1 = self.model.g_q(tk1, qk1)
            R[:nq] -= g_qk1.T @ mu_gk1

        # equations of motion
        R[nq:nx] = (
            Mk1 @ u_dotk1
            - self.model.h(tk1, qk1, uk1)
            - W_gk1 @ la_gk1
            - W_gammak1 @ la_gammak1
        )

        # bilateral constraints
        if self.GGL == 0:
            if self.DAE_index == 3:
                R[nx : nx + nla_g] = self.model.g(tk1, qk1)
                R[nx + nla_g :] = self.model.gamma(tk1, qk1, uk1)
            elif self.DAE_index == 2:
                R[nx : nx + nla_g] = self.model.g_dot(tk1, qk1, uk1)
                R[nx + nla_g :] = self.model.gamma(tk1, qk1, uk1)
            elif self.DAE_index == 1:
                R[nx : nx + nla_g] = self.model.g_ddot(tk1, qk1, uk1, u_dotk1)
                R[nx + nla_g :] = self.model.gamma_dot(tk1, qk1, uk1, u_dotk1)
        if self.GGL == 1:
            R[nx : nx + nla_g] = self.model.g_dot(tk1, qk1, uk1)
            R[nx + nla_g : nx + nla_g + nla_gamma] = self.model.gamma(tk1, qk1, uk1)
            R[nx + nla_g + nla_gamma : nx + 2 * nla_g + nla_gamma] = self.model.g(
                tk1, qk1
            )
        if self.GGL == 2:
            R[nx : nx + nla_g] = self.model.g(tk1, qk1)
            R[nx + nla_g : nx + nla_g + nla_gamma] = self.model.gamma(tk1, qk1, uk1)
            R[nx + nla_g + nla_gamma : nx + 2 * nla_g + nla_gamma] = self.model.g_dot(
                tk1, qk1, uk1
            )
            R[
                nx + 2 * nla_g + nla_gamma : nx + 3 * nla_g + nla_gamma
            ] = self.model.g_ddot(tk1, qk1, uk1, u_dotk1)
            R[
                nx + 3 * nla_g + nla_gamma : nx + 3 * nla_g + 2 * nla_gamma
            ] = self.model.gamma_dot(tk1, qk1, uk1, u_dotk1)
            R[nx + 3 * nla_g + 2 * nla_gamma : nx + 3 * nla_g + 2 * nla_gamma + nu] = (
                Uk1 - W_gk1 @ kappa_gk1 + W_gammak1 @ kappa_gammak1
            )

        yield R

        raise NotImplementedError

        ###################
        # evaluate jacobian
        ###################
        eye_nq = eye(self.nq)
        A = self.model.q_dot_q(tk1, qk1, uk1)
        Bk1 = self.model.B(tk1, qk1, scipy_matrix=csr_matrix)
        K = (
            self.model.Mu_q(tk1, qk1, u_dotk1)
            - self.model.h_q(tk1, qk1, uk1)
            - self.model.Wla_g_q(tk1, qk1, la_gk1)
            - self.model.Wla_gamma_q(tk1, qk1, la_gammak1)
        )
        h_uk1 = self.model.h_u(tk1, qk1, uk1)
        g_qk1 = self.model.g_q(tk1, qk1)
        gamma_qk1 = self.model.gamma_q(tk1, qk1, uk1)

        # sparse assemble global tangent matrix
        if self.GGL:
            g_dot_qk1 = self.model.g_dot_q(tk1, qk1, uk1)
            C = A + self.model.g_q_T_mu_g(tk1, qk1, mu_gk1)
            if self.unknowns == "positions":
                eta = self.eta
                # fmt: off
                J = bmat(
                    [
                        [eta * eye_nq - C,              -Bk1,   None,       None, -g_qk1.T],
                        [               K, eta * Mk1 - h_uk1, -W_gk1, -W_gammak1,     None],
                        [       g_dot_qk1,           W_gk1.T,   None,       None,     None],
                        [       gamma_qk1,       W_gammak1.T,   None,       None,     None],
                        [           g_qk1,              None,   None,       None,     None],
                    ],
                    format="csr",
                )
                # fmt: on
            elif self.unknowns == "velocities":
                etap = 1.0 / self.eta
                # fmt: off
                J = bmat(
                    [
                        [eye_nq - etap * C,        -etap * Bk1,   None,       None, -g_qk1.T],
                        [         etap * K, Mk1 - etap * h_uk1, -W_gk1, -W_gammak1,     None],
                        [ etap * g_dot_qk1,     etap * W_gk1.T,   None,       None,     None],
                        [ etap * gamma_qk1, etap * W_gammak1.T,   None,       None,     None],
                        [     etap * g_qk1,               None,   None,       None,     None],
                    ],
                    format="csr",
                )
                # fmt: on
            elif self.unknowns == "auxiliary":
                gap = self.gamma_prime
                alp = self.alpha_prime
                # fmt: off
                J = bmat(
                    [
                        [alp * eye_nq - gap * C,              -gap * Bk1,   None,       None, -g_qk1.T],
                        [               gap * K, alp * Mk1 - gap * h_uk1, -W_gk1, -W_gammak1,     None],
                        [       gap * g_dot_qk1,           gap * W_gk1.T,   None,       None,     None],
                        [       gap * gamma_qk1,       gap * W_gammak1.T,   None,       None,     None],
                        [           gap * g_qk1,                    None,   None,       None,     None],
                    ],
                    format="csr",
                )
                # fmt: on
        else:
            if self.unknowns == "positions":
                eta = self.eta
                if self.DAE_index == 3:
                    # fmt: off
                    J = bmat(
                        [
                            [eta * eye_nq - A,              -Bk1,   None,       None],
                            [               K, eta * Mk1 - h_uk1, -W_gk1, -W_gammak1],
                            [           g_qk1,              None,   None,       None],
                            [       gamma_qk1,       W_gammak1.T,   None,       None],
                        ],
                        format="csr",
                    )
                    # fmt: on
                elif self.DAE_index == 2:
                    raise NotImplementedError
                elif self.DAE_index == 1:
                    raise NotImplementedError
            elif self.unknowns == "velocities":
                etap = 1.0 / self.eta
                if self.DAE_index == 3:
                    # fmt: off
                    J = bmat(
                        [
                            [eye_nq - etap * A,        -etap * Bk1,   None,       None],
                            [         etap * K, Mk1 - etap * h_uk1, -W_gk1, -W_gammak1],
                            [     etap * g_qk1,               None,   None,       None],
                            [ etap * gamma_qk1, etap * W_gammak1.T,   None,       None],
                        ],
                        format="csr",
                    )
                    # fmt: on
                else:
                    raise NotImplementedError
            elif self.unknowns == "auxiliary":
                gap = self.gamma_prime
                alp = self.alpha_prime
                if self.DAE_index == 3:
                    # fmt: off
                    J = bmat(
                        [
                            [alp * eye_nq - gap * A,              -gap * Bk1,   None,       None],
                            [               gap * K, alp * Mk1 - gap * h_uk1, -W_gk1, -W_gammak1],
                            [           gap * g_qk1,                    None,   None,       None],
                            [       gap * gamma_qk1,       gap * W_gammak1.T,   None,       None],
                        ],
                        format="csr",
                    )
                    # fmt: on
                else:
                    raise NotImplementedError

        # # TODO: Keep this for debugging!
        # J_num = self.__J_num(tk1, sk1)
        # diff = (J - J_num).toarray()
        # error = np.linalg.norm(diff)
        # print(f"error J: {error}")

        yield J

    def __R(self, tk1, sk1):
        return next(self.__R_gen_analytic(tk1, sk1))

    def __J_num(self, tk1, sk1):
        return csr_matrix(
            approx_fprime(sk1, lambda x: self.__R(tk1, x), method="2-point")
        )

    def step(self, tk1, sk1):
        # initial residual and error
        R_gen = self.__R_gen(tk1, sk1)
        R = next(R_gen)
        error = self.error_function(R)
        converged = error < self.tol
        j = 0
        if not converged:
            while j < self.max_iter:
                # jacobian
                J = next(R_gen)

                # Newton update
                j += 1
                if self.preconditioning:
                    # raise NotImplementedError("Not correct yet!")
                    # TODO: Is this efficient? Blas level 3 and blas level 2
                    #       operation shouldn't be that bad for sparse
                    #       matrices.

                    # left and right preconditioner
                    ds = spsolve(
                        self.D_L @ J @ self.D_R, self.D_L @ R, use_umfpack=True
                    )
                    sk1 -= self.D_R @ ds

                    # # right preconditioner
                    # ds = spsolve(J @ self.D_R, R, use_umfpack=True)
                    # sk1 -= self.D_R @ ds

                    # # left preconditioner
                    # ds = spsolve(self.D_L @ J, self.D_L @ R, use_umfpack=True)
                    # sk1 -= ds

                    # # no preconditioner
                    # ds = spsolve(J, R, use_umfpack=True)
                    # sk1 -= ds
                else:
                    ds = spsolve(J, R, use_umfpack=True)
                    sk1 -= ds

                R_gen = self.__R_gen(tk1, sk1)
                R = next(R_gen)

                error = self.error_function(R)
                converged = error < self.tol
                if converged:
                    break

        return converged, j, error, sk1

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        q_dot = [self.q_dotk]
        u_dot = [self.u_dotk]
        la_g = [self.la_gk]
        la_gamma = [self.la_gammak]
        if self.GGL:
            mu_g = [self.mu_gk]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            sk1 = self.sk.copy()  # This copy is mandatory since we modify sk1
            # in the step function
            converged, n_iter, error, sk1 = self.step(tk1, sk1)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )
            if not converged:
                raise RuntimeError(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )

            # update dependent variables
            qk1, uk1, q_dotk1, u_dotk1 = self.update(sk1, store=True)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # extract Lagrange multipliers
            if self.GGL == 0:
                _, _, la_gk1, la_gammak1 = self.unpack(sk1)
            elif self.GGL == 1:
                _, _, la_gk1, la_gammak1, mu_gk1 = self.unpack(sk1)
            else:
                (
                    _,
                    _,
                    la_gk1,
                    la_gammak1,
                    mu_gk1,
                    ak1,
                    kappa_gk1,
                    kappa_gammak1,
                ) = self.unpack(sk1)

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            q_dot.append(q_dotk1)
            u_dot.append(u_dotk1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)
            if self.GGL == 1:
                mu_g.append(mu_gk1)

            # update local variables for accepted time step
            self.tk = tk1
            self.sk = sk1.copy()

        # write solution
        if self.GGL:
            return Solution(
                t=np.array(t),
                q=np.array(q),
                u=np.array(u),
                q_dot=np.array(q_dot),
                u_dot=np.array(u_dot),
                la_g=np.array(la_g),
                la_gamma=np.array(la_gamma),
                mu_g=np.array(mu_g),
            )
        else:
            return Solution(
                t=np.array(t),
                q=np.array(q),
                u=np.array(u),
                q_dot=np.array(q_dot),
                u_dot=np.array(u_dot),
                la_g=np.array(la_g),
                la_gamma=np.array(la_gamma),
            )


class GeneralizedAlphaSecondOrder:
    def __init__(
        self,
        model,
        t1,
        dt,
        rho_inf=1.0,
        tol=1e-6,
        max_iter=30,
        error_function=lambda x: np.max(np.abs(x)),
        numerical_jacobian=False,
        # numerical_jacobian=True,
        # preconditioning=True,
        preconditioning=False,
        GGL=False,
    ):

        self.model = model
        self.GGL = GGL

        # initial time, final time, time step
        self.t0 = t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        # eqn. (72): parameters
        self.rho_inf = rho_inf
        self.alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
        self.alpha_f = rho_inf / (rho_inf + 1.0)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m
        self.beta = 0.25 * (0.5 + self.gamma) ** 2

        # newton settings
        self.tol = tol
        self.max_iter = max_iter
        self.error_function = error_function

        # dimensions (nq = number of coordinates q, etc.)
        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = model.nla_g
        self.nla_gamma = model.nla_gamma

        # dimensions of residual
        self.nR = self.nu + self.nla_g + self.nla_gamma
        if self.GGL:
            self.nR += self.nla_g

        # # initial velocites
        # self.q_dotk = self.model.q_dot(t0, model.q0, model.u0)

        # # solve for consistent initial accelerations and Lagrange mutlipliers
        # M0 = self.model.M(t0, model.q0, scipy_matrix=csr_matrix)
        # h0 = self.model.h(t0, model.q0, model.u0)
        # W_g0 = self.model.W_g(t0, model.q0, scipy_matrix=csr_matrix)
        # W_gamma0 = self.model.W_gamma(t0, model.q0, scipy_matrix=csr_matrix)
        # zeta_g0 = self.model.zeta_g(t0, model.q0, model.u0)
        # zeta_gamma0 = self.model.zeta_gamma(t0, model.q0, model.u0)
        # A = bmat(
        #     [
        #         [M0, -W_g0, -W_gamma0],
        #         [W_g0.T, None, None],
        #         [W_gamma0.T, None, None],
        #     ],
        #     format="csc",
        # )
        # b = np.concatenate([h0, -zeta_g0, -zeta_gamma0])
        # u_dot_la_g_la_gamma = spsolve(A, b)
        # self.u_dotk = u_dot_la_g_la_gamma[: self.nu]
        # self.la_gk = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
        # self.la_gammak = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

        # # initialize auxilary variables
        # self.ak = self.u_dotk.copy()

        def consistent_initial_values(t0, q0, u0):
            """compute physically consistent initial values"""

            # initial velocites
            q_dot0 = self.model.q_dot(t0, q0, u0)

            # solve for consistent initial accelerations and Lagrange mutlipliers
            M0 = self.model.M(t0, q0, scipy_matrix=csr_matrix)
            h0 = self.model.h(t0, q0, u0)
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
            # fmt: on
            b = np.concatenate([h0, -zeta_g0, -zeta_gamma0])
            u_dot_la_g_la_gamma = spsolve(A, b)
            u_dot0 = u_dot_la_g_la_gamma[: self.nu]
            la_g0 = u_dot_la_g_la_gamma[self.nu : self.nu + self.nla_g]
            la_gamma0 = u_dot_la_g_la_gamma[self.nu + self.nla_g :]

            a0 = u_dot0.copy()

            return q0, u0, q_dot0, u_dot0, la_g0, la_gamma0, a0

        def initial_values_Martin(t0, q0, u0):
            ##############################################
            # compute physically consistent initial values
            ##############################################
            q0, u0, q_dot0, u_dot0, la_g0, la_gamma0, a0 = consistent_initial_values(
                t0, q0, u0
            )

            ##################################
            # compute perturbed initial values
            ##################################
            s = 1.0e-3  # Arnold2016 p. 150 last paragraph
            Delta_alpha = self.alpha_m - self.alpha_f  # Arnold2016 (41)

            t0_plus = t0 + s * dt
            u0_plus = u0 + s * dt * u_dot0
            # q0_plus = q0 + s * dt * q_dot0
            # from scipy.optimize import fsolve
            # def q_plus_fun(q0_plus):
            #     return q0_plus - q0 - s * dt * self.model.q_dot(t0, q0_plus, u0)
            # q0_plus = fsolve(q_plus_fun, q0)
            # q0_plus = q0 + s * dt * self.model.q_dot(t0, q0, u0)
            # q0_plus = q0 + s * dt * self.model.q_dot(t0, q0, u0_plus)
            # q0_plus = q0 + s * dt * self.model.q_dot(t0, q0, u0) + 0.5 * s**2 * dt**2 * self.model.q_ddot(t0, q0, u0, u_dot0)
            q0_plus = q0 + s * dt * self.model.q_dot(t0, q0, u0_plus)
            (
                _,
                _,
                _,
                u_dot0_plus,
                _,
                _,
                _,
            ) = consistent_initial_values(t0_plus, q0_plus, u0_plus)

            t0_minus = t0 - s * dt
            u0_minus = u0 - s * dt * u_dot0
            # q0_minus = q0 - s * dt * q_dot0
            # q0_minus = q0 - s * dt * self.model.q_dot(t0, q0, u0)
            # def q_minus_fun(q0_minus):
            #     return q0_minus - q0 + s * dt * self.model.q_dot(t0, q0_minus, u0)
            # q0_minus = fsolve(q_minus_fun, q0)
            # q0_minus = q0 - s * dt * self.model.q_dot(t0, q0, u0_minus)
            # q0_minus = q0 - s * dt * self.model.q_dot(t0, q0, u0) - 0.5 * s**2 * dt**2 * self.model.q_ddot(t0, q0, u0, u_dot0)
            q0_minus = q0 + s * dt * self.model.q_dot(t0, q0, u0_minus)
            (
                _,
                _,
                _,
                u_dot0_minus,
                _,
                _,
                _,
            ) = consistent_initial_values(t0_minus, q0_minus, u0_minus)

            a0 = u_dot0 + Delta_alpha * dt * (u_dot0_plus - u_dot0_minus) / (
                2.0 * s * dt
            )

            # def fun(a0):
            #     u_dot0_plus = consistent_initial_values(
            #         t0 - s * dt,
            #         q0 - s * dt * q_dot0, u0_minus)[3]
            #     return a0 - u_dot0 - Delta_alpha * dt * (u_dot0_plus - u_dot0_minus) / (2.0 * s * dt)

            return q0, u0, q_dot0, u_dot0, la_g0, la_gamma0, a0

        # compute consistent initial conditions
        # q0, u0, q_dot0, u_dot0, la_g0, la_gamma0, a0 = consistent_initial_values(t0, model.q0, model.u0)
        q0, u0, q_dot0, u_dot0, la_g0, la_gamma0, a0 = initial_values_Martin(
            t0, model.q0, model.u0
        )

        # set initial conditions
        self.tk = t0
        self.qk = q0
        self.uk = u0
        self.q_dotk = q_dot0
        self.u_dotk = u_dot0
        self.ak = a0
        self.la_gk = la_g0
        self.la_gammak = la_gamma0
        if self.GGL:
            self.mu_gk = np.zeros_like(la_g0)

        # initial state
        if self.GGL:
            self.xk = np.concatenate(
                (self.u_dotk, self.la_gk, self.la_gammak, self.mu_gk)
            )
        else:
            self.xk = np.concatenate((self.u_dotk, self.la_gk, self.la_gammak))

        if numerical_jacobian:
            self.__R_gen = self.__R_gen_num
        else:
            self.__R_gen = self.__R_gen_analytic

        # check if initial conditions satisfy constraints on position, velocity
        # and acceleration level
        g0 = model.g(self.tk, self.qk)
        g_dot0 = model.g_dot(self.tk, self.qk, self.uk)
        g_ddot0 = model.g_ddot(self.tk, self.qk, self.uk, self.u_dotk)
        gamma0 = model.gamma(self.tk, self.qk, self.uk)
        gamma_dot0 = model.gamma_dot(self.tk, self.qk, self.uk, self.u_dotk)

        # TODO:
        # # TODO: These tolerances should be used defined. Maybe all these
        # #       initial computations and checks should be moved to a
        # #       SolverOptions object ore something similar?
        # rtol = 1.0e-5
        # atol = 1.0e-5

        # assert np.allclose(
        #     g0, np.zeros(self.nla_g), rtol, atol
        # ), "Initial conditions do not fulfill g0!"
        # assert np.allclose(
        #     g_dot0, np.zeros(self.nla_g), rtol, atol
        # ), "Initial conditions do not fulfill g_dot0!"
        # assert np.allclose(
        #     g_ddot0, np.zeros(self.nla_g), rtol, atol
        # ), "Initial conditions do not fulfill g_ddot0!"
        # assert np.allclose(
        #     gamma0, np.zeros(self.nla_gamma)
        # ), "Initial conditions do not fulfill gamma0!"
        # assert np.allclose(
        #     gamma_dot0, np.zeros(self.nla_gamma), rtol, atol
        # ), "Initial conditions do not fulfill gamma_dot0!"

        self.preconditioning = preconditioning
        if preconditioning:
            eta = self.beta * dt * dt
            self.D_L = block_diag(
                [
                    eye(self.nu) * eta,
                    eye(self.nla_g),
                    eye(self.nla_gamma),
                ],
                format="csr",
            )
            self.D_R = block_diag(
                [
                    eye(self.nu),
                    eye(self.nla_g) / eta,
                    eye(self.nla_gamma) / eta,
                ],
                format="csr",
            )

    def update(self, xk1, store=False):
        """Update dependent variables."""
        nu = self.nu
        nla_g = self.nla_g

        # constants
        dt = self.dt
        dt2 = dt * dt
        gamma = self.gamma
        alpha_f = self.alpha_f
        alpha_m = self.alpha_m
        beta = self.beta

        # extract accelerations
        u_dotk1 = xk1[:nu]

        # eqn. (71): compute auxiliary acceleration variables
        ak1 = (
            alpha_f * self.u_dotk + (1.0 - alpha_f) * u_dotk1 - alpha_m * self.ak
        ) / (1.0 - alpha_m)

        # eqn. (73): velocity update formula
        uk1 = self.uk + dt * ((1.0 - gamma) * self.ak + gamma * ak1)

        # # eqn. (125): generalized position update formula
        # u_dot_beta = (1.0 - 2.0 * beta) * self.u_dot_bark + 2.0 * beta * u_dot_bark1
        # qk1 = (
        #     self.qk
        #     + dt * self.model.q_dot(self.tk, self.qk, self.uk)
        #     + 0.5 * dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, u_dot_beta)
        # )

        # kinematic update proposed by Arnold2017, (56a) and (56b)
        Delta_uk1 = self.uk + dt * ((0.5 - beta) * self.ak + beta * ak1)
        qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, Delta_uk1)

        if self.GGL:
            mu_gk1 = xk1[-nla_g:]
            qk1 += self.model.g_q(self.tk, self.qk).T @ mu_gk1

            # def fun(q):
            #     return q - self.qk - dt * self.model.q_dot(self.tk, q, Delta_uk1) - self.model.g_q(self.tk, q).T @ mu_gk1
            # from scipy.optimize import fsolve
            # qk1 = fsolve(fun, qk1)

        if store:
            self.u_dotk = u_dotk1
            self.uk = uk1
            self.qk = qk1
            self.ak = ak1

        return qk1, uk1

    def pack(self, *args):
        return np.concatenate([*args])

    def unpack(self, x):
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        a = x[:nu]
        la_g = x[nu : nu + nla_g]
        la_gamma = x[nu + nla_g : nu + nla_g + nla_gamma]
        if self.GGL:
            mu_g = x[nu + nla_g + nla_gamma :]
            return a, la_g, la_gamma, mu_g
        else:
            return a, la_g, la_gamma

    def __R_gen_num(self, tk1, sk1):
        yield self.__R(tk1, sk1)
        yield csr_matrix(self.__J_num(tk1, sk1))

    def __R_gen_analytic(self, tk1, xk1):
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma

        # extract vector of nknowns
        if self.GGL:
            u_dotk1, la_gk1, la_gammak1, mu_gk1 = self.unpack(xk1)
        else:
            u_dotk1, la_gk1, la_gammak1 = self.unpack(xk1)

        # update dependent variables
        qk1, uk1 = self.update(xk1, store=False)

        # evaluate repeated used quantities
        Mk1 = self.model.M(tk1, qk1, scipy_matrix=csr_matrix)
        W_gk1 = self.model.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.model.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)

        ###################
        # evaluate residual
        ###################
        R = np.zeros(self.nR)

        # equations of motion
        R[:nu] = (
            Mk1 @ u_dotk1
            - self.model.h(tk1, qk1, uk1)
            - W_gk1 @ la_gk1
            - W_gammak1 @ la_gammak1
        )

        # bilateral constraints
        R[nu : nu + nla_g] = self.model.g(tk1, qk1)
        R[nu + nla_g : nu + nla_g + nla_gamma] = self.model.gamma(tk1, qk1, uk1)
        if self.GGL:
            R[nu + nla_g + nla_gamma :] = self.model.g_dot(tk1, qk1, uk1)

        yield R

        ###################
        # evaluate jacobian
        ###################

        # chain rules
        ak1_u_dotk1 = (1.0 - self.alpha_f) / (1.0 - self.alpha_m)
        uk1_ak1 = self.dt * self.gamma
        uk1_u_dotk1 = uk1_ak1 * ak1_u_dotk1
        qk1_ak1 = self.dt**2 * self.beta * self.model.B(self.tk, self.qk)
        qk1_u_dotk1 = qk1_ak1 * ak1_u_dotk1
        qk1_muk1 = self.model.g_q(self.tk, self.qk).T

        K = (
            self.model.Mu_q(tk1, qk1, u_dotk1)
            - self.model.h_q(tk1, qk1, uk1)
            - self.model.Wla_g_q(tk1, qk1, la_gk1)
            - self.model.Wla_gamma_q(tk1, qk1, la_gammak1)
        )
        h_uk1 = self.model.h_u(tk1, qk1, uk1)

        g_qk1 = self.model.g_q(tk1, qk1)
        gamma_qk1 = self.model.gamma_q(tk1, qk1, uk1)

        g_u_dotk1 = g_qk1 @ qk1_u_dotk1
        gamma_u_dotk1 = gamma_qk1 @ qk1_u_dotk1 + W_gammak1.T * uk1_u_dotk1

        if self.GGL:
            g_muk1 = g_qk1 @ qk1_muk1

            gamma_muk1 = gamma_qk1 @ qk1_muk1

            g_dot_qk1 = self.model.g_dot_q(tk1, qk1, uk1)
            g_dot_u_dotk1 = g_dot_qk1 @ qk1_u_dotk1 + W_gk1.T * uk1_u_dotk1
            g_dot_muk1 = g_dot_qk1 @ qk1_muk1

        # sparse assemble global tangent matrix
        if self.GGL:
            # fmt: off
            J = bmat(
                [
                    [Mk1 + K @ qk1_u_dotk1 - h_uk1 * uk1_u_dotk1, -W_gk1, -W_gammak1, K @ qk1_muk1],
                    [                                  g_u_dotk1,   None,       None,       g_muk1],
                    [                              gamma_u_dotk1,   None,       None,   gamma_muk1],
                    [                              g_dot_u_dotk1,   None,       None,   g_dot_muk1],
                ],
                format="csr",
            )
            # fmt: on
        else:
            # fmt: off
            J = bmat(
                [
                    [Mk1 + K @ qk1_u_dotk1 - h_uk1 * uk1_u_dotk1, -W_gk1, -W_gammak1],
                    [                                  g_u_dotk1,   None,       None],
                    [                              gamma_u_dotk1,   None,       None],
                ],
                format="csr",
            )
            # fmt: on

        # # TODO: Keep this for debugging!
        # J_num = self.__J_num(tk1, xk1)
        # diff = (J - J_num).toarray()
        # error = np.linalg.norm(diff)
        # print(f"error J: {error}")

        yield J

    def __R(self, tk1, xk1):
        return next(self.__R_gen_analytic(tk1, xk1))

    def __J_num(self, tk1, xk1):
        return csr_matrix(
            approx_fprime(
                # xk1, lambda x: self.__R(tk1, x), method="2-point", eps=1.0e-4
                xk1,
                lambda x: self.__R(tk1, x),
                method="2-point",
                eps=1.0e-5,
            )
            # approx_fprime(xk1, lambda x: self.__R(tk1, x), method="2-point")
            # approx_fprime(xk1, lambda x: self.__R(tk1, x), method="3-point")
        )

    def step(self, tk1, xk1):
        # initial residual and error
        R_gen = self.__R_gen(tk1, xk1)
        R = next(R_gen)
        error = self.error_function(R)
        converged = error < self.tol
        j = 0
        if not converged:
            while j < self.max_iter:
                # jacobian
                J = next(R_gen)

                # Newton update
                j += 1
                if self.preconditioning:
                    # left and right preconditioner
                    dx = spsolve(
                        self.D_L @ J @ self.D_R, self.D_L @ R, use_umfpack=True
                    )
                    xk1 -= self.D_R @ dx

                    # # right preconditioner
                    # ds = spsolve(J @ self.D_R, R, use_umfpack=True)
                    # sk1 -= self.D_R @ ds

                    # # left preconditioner
                    # ds = spsolve(self.D_L @ J, self.D_L @ R, use_umfpack=True)
                    # sk1 -= ds
                else:
                    dx = spsolve(J, R, use_umfpack=True)
                    xk1 -= dx

                R_gen = self.__R_gen(tk1, xk1)
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
        u_dot = [self.u_dotk]
        la_g = [self.la_gk]
        la_gamma = [self.la_gammak]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # perform a sovler step
            tk1 = self.tk + self.dt
            xk1 = self.xk.copy()  # This copy is mandatory since we modify sk1
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

            # update dependent variables
            if self.GGL:
                ak1, la_gk1, la_gammak1, mu_gk1 = self.unpack(xk1)
            else:
                ak1, la_gk1, la_gammak1 = self.unpack(xk1)
            qk1, uk1 = self.update(xk1, store=True)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            u_dot.append(ak1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)

            # update local variables for accepted time step
            self.tk = tk1
            self.xk = xk1.copy()

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            u_dot=np.array(u_dot),
            la_g=np.array(la_g),
            la_gamma=np.array(la_gamma),
        )
