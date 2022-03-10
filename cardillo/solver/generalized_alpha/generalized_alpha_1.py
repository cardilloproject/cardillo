import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, csr_matrix, identity, bmat
from tqdm import tqdm

from cardillo.math import Numerical_derivative
from cardillo.solver import Solution


class Generalized_alpha_1:
    def __init__(
        self,
        model,
        t1,
        dt=None,
        t_eval=None,
        variable_dt=True,
        rho_inf=1,
        beta=None,
        gamma=None,
        alpha_m=None,
        alpha_f=None,
        atol=1e-3,
        rtol=1e-3,
        newton_tol=1e-6,
        newton_max_iter=10,
        newton_error_function=lambda x: np.max(np.abs(x)),
        numerical_jacobian=False,
        debug=False,
    ):

        self.model = model

        # integration time
        self.t0 = t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.t_eval = t_eval

        # parameter
        self.rho_inf = rho_inf
        if None in [beta, gamma, alpha_m, alpha_f]:
            self.alpha_m = (2 * rho_inf - 1) / (1 + rho_inf)
            self.alpha_f = rho_inf / (1 + rho_inf)
            self.gamma = 0.5 + self.alpha_f - self.alpha_m
            self.beta = 0.25 * ((self.gamma + 0.5) ** 2)
        else:
            self.gamma = gamma
            self.beta = beta
            self.alpha_m = alpha_m
            self.alpha_f = alpha_f
        self.alpha_ratio = (1 - self.alpha_f) / (1 - self.alpha_m)

        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.newton_error_function = newton_error_function

        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = model.nla_g
        self.nla_gamma = model.nla_gamma
        self.nR = self.nu + self.nla_g + self.nla_gamma

        self.aDOF = np.arange(self.nu)
        self.la_gDOF = self.nu + np.arange(self.nla_g)
        self.la_gammaDOF = self.nu + self.nla_g + np.arange(self.nla_gamma)

        self.Mk1 = model.M(t0, model.q0)
        self.W_gk1 = self.model.W_g(t0, model.q0)
        self.W_gammak1 = self.model.W_gamma(t0, model.q0)

        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.la_gk = model.la_g0
        self.la_gammak = model.la_gamma0
        self.ak = spsolve(
            self.Mk1.tocsr(),
            self.model.h(t0, model.q0, model.u0)
            + self.W_gk1 @ model.la_g0
            + self.W_gammak1 @ model.la_gamma0,
        )
        self.a_bark = self.ak.copy()

        # time step selection parameters
        self.variable_dt = variable_dt
        self.atol = atol
        self.rtol = rtol
        self.MAX_REJECTED_ITER = 10  # maximal number of rejected steps in a row
        self.SAFETY = 0.8  # safety for scaling factor
        self.MIN_FACTOR = 0.2  # minimal scaling factor
        self.MAX_FACTOR = 2  # maximal scaling factor

        # initial step size
        if dt is None:
            self.dt = self.select_initial_step()
        else:
            self.dt = dt

        # pbar
        self.pbar_frac = (t1 - t0) / 100
        self.pbar = tqdm(total=100, leave=True)
        self.pbar_i = 0

        self.numerical_jacobian = numerical_jacobian
        if numerical_jacobian:
            self.__R_x = self.__R_x_num
        else:
            self.__R_x = self.__R_x_analytic
        self.debug = debug
        if debug:
            self.__R_x = self.__R_x_debug

    def scale(self, q0, q1):
        """scaled tolerance, see :cite:`Hairer1993` eqn. 4.10"""
        return self.atol + self.rtol * np.maximum(np.abs(q0), np.abs(q1))

    def estimate_error_norm(self, e, scale):
        """estimated error norm, see :cite:`Hairer1993` eqn. 4.11"""
        return np.linalg.norm(e / scale) / np.sqrt(len(e))

    def select_initial_step(self):
        """selects initial step size, see :cite:`Hairer1993` p. 169"""
        t0 = self.tk
        q0 = self.qk
        u0 = self.uk
        a0 = self.ak
        q_dot0 = self.model.q_dot(t0, q0, u0)
        q_ddot0 = self.model.q_ddot(t0, q0, u0, a0)
        sc = self.scale(q0, q0)
        d0 = self.estimate_error_norm(q0, sc)
        d1 = self.estimate_error_norm(q_dot0, sc)
        d2 = self.estimate_error_norm(q_ddot0, sc)
        h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
        h1 = np.cbrt(0.01 / np.max([d1, d2]))
        return np.min([100 * h0, h1])

    def __R(self, tk1, ak1, la_gk1, la_gammak1):
        dt = self.dt
        dt2 = self.dt**2
        # update dependent variables
        a_bark1 = (
            self.alpha_f * self.ak
            + (1 - self.alpha_f) * ak1
            - self.alpha_m * self.a_bark
        ) / (1 - self.alpha_m)
        self.uk1 = self.uk + dt * (
            (1 - self.gamma) * self.a_bark + self.gamma * a_bark1
        )
        a_bar_beta = (0.5 - self.beta) * self.a_bark + self.beta * a_bark1
        self.qk1 = (
            self.qk
            + dt * self.model.q_dot(self.tk, self.qk, self.uk)
            + dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, a_bar_beta)
        )

        self.Mk1 = self.model.M(tk1, self.qk1)
        self.W_gk1 = self.model.W_g(tk1, self.qk1)
        self.W_gammak1 = self.model.W_gamma(tk1, self.qk1)

        # evaluate residual R(ak1, la_gk1, la_gammak1)
        R = np.zeros(self.nR)
        R[self.aDOF] = self.Mk1 @ ak1 - (
            self.model.h(tk1, self.qk1, self.uk1)
            + self.W_gk1 @ la_gk1
            + self.W_gammak1 @ la_gammak1
        )
        R[self.la_gDOF] = self.model.g(tk1, self.qk1)
        R[self.la_gammaDOF] = self.model.gamma(tk1, self.qk1, self.uk1)
        return R

    def __R_wrapper(self, t, x):
        ak1 = x[self.aDOF]
        la_gk1 = x[self.la_gDOF]
        la_gammak1 = x[self.la_gammaDOF]

        return self.__R(t, ak1, la_gk1, la_gammak1)

    def __R_x_num(self, tk1, ak1, la_gk1, la_gammak1):
        x = np.zeros(self.nR)
        x[self.aDOF] = ak1
        x[self.la_gDOF] = la_gk1
        x[self.la_gammaDOF] = la_gammak1

        R_x_num = Numerical_derivative(self.__R_wrapper, order=2)._x(tk1, x)

        return csr_matrix(R_x_num)

    def __R_x_analytic(self, tk1, ak1, la_gk1, la_gammak1):
        # equations of motion
        Ra_q = self.model.Mu_q(tk1, self.qk1, ak1) - (
            self.model.h_q(tk1, self.qk1, self.uk1)
            + self.model.Wla_g_q(tk1, self.qk1, la_gk1)
            + self.model.Wla_gamma_q(tk1, self.qk1, la_gammak1)
        )
        Ra_u = -self.model.h_u(tk1, self.qk1, self.uk1)

        Ra_a = self.Mk1 + Ra_q @ self.q_a + Ra_u * self.u_a
        Ra_la_g = -self.W_gk1
        Ra_la_gamma = -self.W_gammak1

        # constraint equations
        Rla_g_a = self.model.g_q(tk1, self.qk1) @ self.q_a
        Rla_gamma_a = (
            self.model.gamma_q(tk1, self.qk1, self.uk1) @ self.q_a
            + self.model.gamma_u(tk1, self.qk1) * self.u_a
        )

        return bmat(
            [
                [Ra_a, Ra_la_g, Ra_la_gamma],
                [Rla_g_a, None, None],
                [Rla_gamma_a, None, None],
            ]
        ).tocsc()

    def __R_x_debug(self, tk1, ak1, la_gk1, la_gammak1):
        R_x_num = self.__R_x_num(tk1, ak1, la_gk1, la_gammak1)
        R_x_analytic = self.__R_x_analytic(tk1, ak1, la_gk1, la_gammak1)
        diff = R_x_num - R_x_analytic.toarray()

        print(f"total error jacobian: {np.linalg.norm(diff)/ self.nR:.5e}")

        if self.numerical_jacobian:
            return R_x_num
        else:
            return R_x_analytic

    def step(self):
        dt = self.dt
        tk1 = self.tk + dt
        ak1 = self.ak.copy()
        la_gk1 = self.la_gk.copy()
        la_gammak1 = self.la_gammak.copy()

        # initial guess for Newton-Raphson solver
        xk1 = np.zeros(self.nR)
        xk1[self.aDOF] = ak1
        xk1[self.la_gDOF] = la_gk1
        xk1[self.la_gammaDOF] = la_gammak1

        # initial residual and error
        R = self.__R(tk1, ak1, la_gk1, la_gammak1)
        error = self.newton_error_function(R)
        converged = error < self.newton_tol
        j = 0
        if not converged:
            while j < self.newton_max_iter:
                # jacobian
                R_x = self.__R_x(tk1, ak1, la_gk1, la_gammak1)

                # Newton update
                j += 1
                dx = spsolve(R_x, R)
                xk1 -= dx
                ak1 = xk1[self.aDOF]
                la_gk1 = xk1[self.la_gDOF]
                la_gammak1 = xk1[self.la_gammaDOF]

                R = self.__R(tk1, ak1, la_gk1, la_gammak1)
                error = self.newton_error_function(R)
                converged = error < self.newton_tol
                if converged:
                    break

        return (converged, j, error), tk1, ak1, la_gk1, la_gammak1

    def solve(self):
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        la_g = [self.la_gk]
        la_gamma = [self.la_gammak]

        n_rejected_iter = 0
        step_accepted = True
        max_factor = self.MAX_FACTOR
        t_eval = self.t_eval
        t_eval_i = 0
        while self.tk < self.t1:
            if int(self.tk // self.pbar_frac) == self.pbar_i:
                self.pbar.update(1)
                self.pbar.set_description(f"t: {self.tk:0.2e}s < {self.t1:0.2e}s")
                self.pbar_i += 1
            dt = self.dt
            dt2 = self.dt**2
            self.q_a = (
                dt2 * self.beta * self.alpha_ratio * self.model.B(self.tk, self.qk)
            )
            self.u_a = dt * self.gamma * self.alpha_ratio

            (converged, n_iter, error), tk1, ak1, la_gk1, la_gammak1 = self.step()
            # pbar.set_description(f't: {tk1:0.2e}; Newton: {n_iter}/{self.newton_max_iter} iterations; error: {error:0.2e}')

            if self.variable_dt:
                # implicit euler step as comparative solution
                q1_comp = self.qk1 + self.model.q_dot(tk1, self.qk1, self.uk1) * dt

                # step size selection, see hairer1993 (eqn. 4.10 - 4.11)
                sc = self.scale(self.qk, self.qk1)
                e = self.qk1 - q1_comp
                ERR = self.estimate_error_norm(e, sc)
                # SAFETY parameter depending on newton iterations https://github.com/scipy/scipy/blob/maintenance/1.4.x/scipy/integrate/_ivp/bdf.py#L373
                # SAFETY = self.SAFETY * (2 * self.newton_max_iter + 1) / (2 * self.newton_max_iter + newton_iter)
                SAFETY = self.SAFETY
                fac = SAFETY * np.sqrt(1 / ERR)
                # TODO: enable obtionally choice for factor range that is rejected
                # fac = 1 if fac >= 1 and fac <= 1.15 else fac
                self.dt *= min(max_factor, max(self.MIN_FACTOR, fac))

                if ERR <= 1 and converged:
                    step_accepted = True
                    n_rejected_iter = 0
                    max_factor = self.MAX_FACTOR
                else:
                    step_accepted = False
                    n_rejected_iter += 1
                    max_factor = 1
                if n_rejected_iter > self.MAX_REJECTED_ITER:
                    raise RuntimeError(
                        "Maximum number of rejected steps per time step reached."
                    )
            else:
                if not converged:
                    raise RuntimeError(
                        f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                    )
            if step_accepted:
                # print(f'n_rejected = {n_rejected_iter}')
                # print(f'dt = {self.dt}')
                qk1, uk1 = self.model.step_callback(tk1, self.qk1, self.uk1)
                # inspired by https://github.com/scipy/scipy/blob/maintenance/1.4.x/scipy/integrate/_ivp/ivp.py#L616
                if t_eval is not None:
                    t_eval_i_new = np.searchsorted(t_eval, tk1, side="right")
                    t_eval_span = t_eval[t_eval_i:t_eval_i_new]
                    if t_eval_span.size > 0:
                        # get dense output
                        xis = (t_eval_span - self.tk) / (tk1 - self.tk)
                        for i, xi in enumerate(xis):
                            # linear interpolation
                            t.append(t_eval_span[i])
                            q.append(xi * qk1 + (1 - xi) * self.qk)
                            u.append(xi * uk1 + (1 - xi) * self.uk)
                            la_g.append(xi * la_gk1 + (1 - xi) * self.la_gk)
                            la_gamma.append(xi * la_gammak1 + (1 - xi) * self.la_gammak)
                        t_eval_i = t_eval_i_new
                else:
                    t.append(tk1)
                    q.append(qk1)
                    u.append(uk1)
                    la_g.append(la_gk1)
                    la_gamma.append(la_gammak1)

                # update local variables for accepted time step
                self.tk = tk1
                self.qk = qk1
                self.uk = uk1
                self.a_bark = (
                    self.alpha_f * self.ak
                    + (1 - self.alpha_f) * ak1
                    - self.alpha_m * self.a_bark
                ) / (1 - self.alpha_m)
                self.ak = ak1
                self.la_gk = la_gk1
                self.la_gammak = la_gammak1

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            la_g=np.array(la_g),
            la_gamma=np.array(la_gamma),
        )
