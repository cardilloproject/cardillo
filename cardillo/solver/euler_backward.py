import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, eye, bmat
from tqdm import tqdm

from cardillo.math import prox_sphere, fsolve
from cardillo.solver import Solution, consistent_initial_conditions


class EulerBackward:
    def __init__(
        self,
        system,
        t1,
        dt,
        atol=1e-6,
        max_iter=10,
        error_function=lambda x: np.max(np.abs(x)),
        method="index 2 GGL",
    ):
        self.system = system
        assert method in ["index 1", "index 2", "index 3", "index 2 GGL"]
        self.method = method
        self.atol = atol
        self.max_iter = max_iter
        self.error_function = error_function

        #######################################################################
        # integration time
        #######################################################################
        t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t_eval = np.arange(t0, self.t1 + self.dt, self.dt)

        #######################################################################
        # dimensions
        #######################################################################
        self.nq = self.system.nq
        self.nu = self.system.nu
        self.nla_g = self.system.nla_g
        self.nla_gamma = self.system.nla_gamma
        self.nla_S = self.system.nla_S
        self.ny = self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_S
        if method == "index 2 GGL":
            self.ny += self.nla_g

        #######################################################################
        # consistent initial conditions
        #######################################################################
        (
            t0,
            self.qn,
            self.un,
            q_dot0,
            u_dot0,
            la_g0,
            la_gamma0,
        ) = consistent_initial_conditions(system)

        self.y = np.zeros(self.ny, dtype=float)
        self.y[: self.nq] = q_dot0
        self.y[self.nq : self.nq + self.nu] = u_dot0
        self.y[self.nq + self.nu : self.nq + self.nu + self.nla_g] = la_g0
        self.y[
            self.nq
            + self.nu
            + self.nla_g : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
        ] = la_gamma0

    def _unpack(self, y):
        q_dot = y[: self.nq]
        u_dot = y[self.nq : self.nq + self.nu]
        la_g = y[self.nq + self.nu : self.nq + self.nu + self.nla_g]
        la_gamma = y[
            self.nq
            + self.nu
            + self.nla_g : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
        ]
        mu_S = y[
            self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma : self.nq
            + self.nu
            + self.nla_g
            + self.nla_gamma
            + self.nla_S
        ]
        mu_g = y[self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_S :]
        return q_dot, u_dot, la_g, la_gamma, mu_S, mu_g

    def _update(self, y):
        q_dot = y[: self.nq]
        u_dot = y[self.nq : self.nq + self.nu]
        q = self.qn + self.dt * q_dot
        u = self.un + self.dt * u_dot
        return q, u

    def _R(self, y):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_S = self.nla_S
        nqu = nq + nu

        t = self.t
        q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = self._unpack(y)
        q, u = self._update(y)

        self.M = self.system.M(t, q, scipy_matrix=csr_matrix)
        self.W_g = self.system.W_g(t, q, scipy_matrix=csr_matrix)
        self.W_gamma = self.system.W_gamma(t, q, scipy_matrix=csr_matrix)
        R = np.zeros(self.ny, dtype=y.dtype)

        self.g_S_q = self.system.g_S_q(t, q, scipy_matrix=csc_matrix)
        R[:nq] = q_dot - self.system.q_dot(t, q, u) - self.g_S_q.T @ mu_S
        if self.method == "index 2 GGL":
            self.g_q = self.system.g_q(t, q, scipy_matrix=csc_matrix)
            R[:nq] -= self.g_q.T @ mu_g

        R[nq:nqu] = self.M @ u_dot - (
            self.system.h(t, q, u) + self.W_g @ la_g + self.W_gamma @ la_gamma
        )

        if self.method == "index 1":
            R[nqu + nla_g : nqu + nla_g + nla_gamma] = self.system.gamma_dot(
                t, q, u, u_dot
            )
        else:
            R[nqu + nla_g : nqu + nla_g + nla_gamma] = self.system.gamma(t, q, u)

        if self.method == "index 2 GGL":
            R[nqu : nqu + nla_g] = self.system.g_dot(t, q, u)
            R[nqu + nla_g + nla_gamma + nla_S :] = self.system.g(t, q)
        elif self.method == "index 3":
            R[nqu : nqu + nla_g] = self.system.g(t, q)
        elif self.method == "index 2":
            R[nqu : nqu + nla_g] = self.system.g_dot(t, q, u)
        elif self.method == "index 1":
            R[nqu : nqu + nla_g] = self.system.g_ddot(t, q, u, u_dot)

        R[nqu + nla_g + nla_gamma : nqu + nla_g + nla_gamma + nla_S] = self.system.g_S(
            t, q
        )

        return R

    def _J(self, y):
        t = self.t
        dt = self.dt
        q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = self._unpack(y)
        q, u = self._update(y)

        A = (
            eye(self.nq, format="coo")
            - dt * self.system.q_dot_q(t, q, u)
            - dt * self.system.g_S_q_T_mu_q(t, q, mu_S)
        )
        B = self.system.B(t, q)
        C = (
            self.system.Mu_q(t, q, u_dot)
            - self.system.h_q(t, q, u)
            - self.system.Wla_g_q(t, q, la_g)
            - self.system.Wla_gamma_q(t, q, la_gamma)
        )
        D = self.M - dt * self.system.h_u(t, q, u)

        gamma_q = self.system.gamma_q(t, q, u)
        g_S_q = self.g_S_q

        # fmt: off
        if self.method == "index 2 GGL":
            g_q = self.g_q
            g_dot_q = self.system.g_dot_q(t, q, u)
            A -= dt * self.system.g_q_T_mu_q(t, q, mu_g)
            J = bmat([
                [           A,             -dt * B,      None,          None, -g_S_q.T, -g_q.T],
                [      dt * C,                   D, -self.W_g, -self.W_gamma,     None,   None],
                [dt * g_dot_q,     dt * self.W_g.T,      None,          None,     None,   None],
                [dt * gamma_q, dt * self.W_gamma.T,      None,          None,     None,   None],
                [  dt * g_S_q,                None,      None,          None,     None,   None],
                [    dt * g_q,                None,      None,          None,     None,   None],
            ], format="csc")
        elif self.method == "index 3":
            g_q = self.system.g_q(t, q)
            J = bmat([
                [           A,             -dt * B,      None,          None, -g_S_q.T],
                [      dt * C,                   D, -self.W_g, -self.W_gamma,     None],
                [    dt * g_q,                None,      None,          None,     None],
                [dt * gamma_q, dt * self.W_gamma.T,      None,          None,     None],
                [  dt * g_S_q,                None,      None,          None,     None],
            ], format="csc")
        elif self.method == "index 2":
            g_dot_q = self.system.g_dot_q(t, q, u)
            J = bmat([
                [           A,             -dt * B,      None,          None, -g_S_q.T],
                [      dt * C,                   D, -self.W_g, -self.W_gamma,     None],
                [dt * g_dot_q,     dt * self.W_g.T,      None,          None,     None],
                [dt * gamma_q, dt * self.W_gamma.T,      None,          None,     None],
                [  dt * g_S_q,                None,      None,          None,     None],
            ], format="csc")
        elif self.method == "index 1":
            g_ddot_q = self.system.g_ddot_q(t, q, u, u_dot)
            g_ddot_u = self.system.g_ddot_u(t, q, u, u_dot)
            gamma_dot_q = self.system.gamma_dot_q(t, q, u, u_dot)
            gamma_dot_u = self.system.gamma_dot_u(t, q, u, u_dot)
            J = bmat([
                [               A,                           -dt * B,      None,          None, -g_S_q.T],
                [          dt * C,                                 D, -self.W_g, -self.W_gamma,     None],
                [   dt * g_ddot_q,        self.W_g.T + dt * g_ddot_u,      None,          None,     None],
                [dt * gamma_dot_q, self.W_gamma.T + dt * gamma_dot_u,      None,          None,     None],
                [      dt * g_S_q,                              None,      None,          None,     None],
            ], format="csc")
        else:
            raise NotImplementedError
        # fmt: on

        return J

        # J_num = csc_matrix(approx_fprime(y, self._R, method="2-point"))
        # J_num = csc_matrix(approx_fprime(y, self._R, method="3-point"))
        J_num = csc_matrix(approx_fprime(y, self._R, method="cs", eps=1.0e-12))
        diff = (J - J_num).toarray()
        # diff = diff[:self.nq]
        # diff = diff[self.nq : ]
        error = np.linalg.norm(diff)
        print(f"error Jacobian: {error}")
        return J_num

    def solve(self):
        q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = self._unpack(self.y)

        # lists storing output variables
        q_list = [self.qn]
        u_list = [self.un]
        q_dot_list = [q_dot]
        u_dot_list = [u_dot]
        la_g_list = [la_g]
        la_gamma_list = [la_gamma]
        mu_S_list = [mu_S]
        mu_g_list = [mu_g]

        pbar = tqdm(self.t_eval[:-1])
        for t in pbar:
            self.t = t
            sol = fsolve(self._R, self.y, jac=self._J)
            self.y = sol[0]
            converged = sol[1]
            error = sol[2]
            n_iter = sol[3]
            assert converged

            pbar.set_description(
                f"t: {t:0.2e}; {n_iter}/{self.max_iter} iterations; error: {error:0.2e}"
            )

            q, u = self._update(self.y)
            self.qn, self.un = self.system.step_callback(t, q, u)
            q_dot, u_dot, la_g, la_gamma, mu_S, mu_g = self._unpack(self.y)

            q_list.append(self.qn)
            u_list.append(self.un)
            q_dot_list.append(q_dot)
            u_dot_list.append(u_dot)
            la_g_list.append(la_g)
            la_gamma_list.append(la_gamma)
            mu_S_list.append(mu_S)
            mu_g_list.append(mu_g)

        # write solution
        return Solution(
            t=self.t_eval,
            q=np.array(q_list),
            u=np.array(u_list),
            q_dot=np.array(q_dot_list),
            u_dot=np.array(u_dot_list),
            la_g=np.array(la_g_list),
            la_gamma=np.array(la_gamma_list),
            mu_s=np.array(mu_S_list),
            mu_g=np.array(mu_g_list),
        )


class NonsmoothBackwardEuler:
    def __init__(
        self,
        system,
        t1,
        dt,
        tol=1e-6,
        max_iter=10,
    ):
        self.system = system

        #######################################################################
        # integration time
        #######################################################################
        self.t0 = t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        #######################################################################
        # newton settings
        #######################################################################
        self.tol = tol
        self.max_iter = max_iter

        #######################################################################
        # dimensions
        #######################################################################
        self.nq = system.nq
        self.nu = system.nu
        self.nla_g = system.nla_g
        self.nla_gamma = system.nla_gamma
        self.nla_N = system.nla_N
        self.nla_F = system.nla_F
        self.ny = (
            self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_N + self.nla_F
        )
        self.split = np.cumsum(
            np.array(
                [
                    self.nq,
                    self.nu,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_N,
                ],
                dtype=int,
            )
        )

        #######################################################################
        # consistent initial conditions
        #######################################################################
        (
            self.tn,
            self.qn,
            self.un,
            self.q_dotn,
            self.u_dotn,
            self.la_gn,
            self.la_gamman,
        ) = consistent_initial_conditions(system)

        self.la_Nn = system.la_N0
        self.la_Fn = system.la_F0

        #######################################################################
        # initial values
        #######################################################################
        self.yn = np.concatenate(
            (
                self.q_dotn,
                self.u_dotn,
                self.la_gn,
                self.la_gamman,
                self.la_Nn,
                self.la_Fn,
            )
        )

        # initialize index sets
        self.I_N = np.zeros(self.nla_N, dtype=bool)

    def R(self, yn1, update_index=False):
        tn, dt, qn, un = self.tn, self.dt, self.qn, self.un

        q_dotn1, u_dotn1, la_gn1, la_gamman1, la_Nn1, la_Fn1 = np.array_split(
            yn1, self.split
        )
        tn1 = tn + dt
        qn1 = qn + dt * q_dotn1
        un1 = un + dt * u_dotn1

        ###################
        # evaluate residual
        ###################
        R = np.zeros(self.ny, dtype=yn1.dtype)

        ####################
        # kinematic equation
        ####################
        R[: self.split[0]] = q_dotn1 - self.system.q_dot(tn1, qn1, un1)

        ####################
        # euations of motion
        ####################
        R[self.split[0] : self.split[1]] = (
            self.system.M(tn1, qn1, scipy_matrix=csr_matrix) @ u_dotn1
            - self.system.h(tn1, qn1, un1)
            - self.system.W_g(tn1, qn1, scipy_matrix=csr_matrix) @ la_gn1
            - self.system.W_gamma(tn1, qn1, scipy_matrix=csr_matrix) @ la_gamman1
            - self.system.W_N(tn1, qn1, scipy_matrix=csr_matrix) @ la_Nn1
            - self.system.W_F(tn1, qn1, scipy_matrix=csr_matrix) @ la_Fn1
        )

        #######################
        # bilateral constraints
        #######################
        R[self.split[1] : self.split[2]] = self.system.g(tn1, qn1)
        R[self.split[2] : self.split[3]] = self.system.gamma(tn1, qn1, un1)

        ###########
        # Signorini
        ###########
        g_Nn1 = self.system.g_N(tn1, qn1)
        prox_arg = g_Nn1 - self.prox_r_N * la_Nn1
        if update_index:
            self.I_N = prox_arg <= 0.0

        R[self.split[3] : self.split[4]] = np.where(self.I_N, g_Nn1, la_Nn1)

        ##########
        # friction
        ##########
        gamma_Fn1 = self.system.gamma_F(tn1, qn1, un1)
        mu = self.system.mu
        prox_r_F = self.prox_r_F
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            i_F = np.array(i_F)

            if len(i_F) > 0:
                # Note: This is the simplest formulation for friction but we
                # subsequently decompose the function it into both cases of
                # the prox_sphere function for easy derivation
                R[self.split[4] + i_F] = la_Fn1[i_F] + prox_sphere(
                    prox_r_F[i_N] * gamma_Fn1[i_F] - la_Fn1[i_F],
                    mu[i_N] * la_Nn1[i_N],
                )

        return R

    def J_contact(self, xk1, update_index=False):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        mu = self.system.mu

        q_dotk1, u_dotk1, la_gk1, la_gammak1, la_Nk1, la_Fk1 = self.unpack_x(xk1)
        tk1, qk1, uk1_free = self.update_x(xk1)

        # evaluate repeatedly used quantities
        Mk1 = self.system.M(tk1, qk1, scipy_matrix=csr_matrix)
        W_gk1 = self.system.W_g(tk1, qk1, scipy_matrix=csr_matrix)
        W_gammak1 = self.system.W_gamma(tk1, qk1, scipy_matrix=csr_matrix)
        W_Nk1 = self.system.W_N(tk1, qk1, scipy_matrix=csr_matrix)
        # note: csc.T gives csr for efficient row slicing
        W_Fk1 = self.system.W_F(tk1, qk1, scipy_matrix=csc_matrix)
        gamma_Fk1 = self.system.gamma_F(tk1, qk1, uk1_free)

        # chain rules for backward Euler update
        qk1_q_dotk1 = self.dt
        uk1_free_u_dotk1 = self.dt

        ####################
        # kinematic equation
        ####################
        J_q_dotk1_q_dotk1 = (
            eye(nq, nq) - self.system.q_dot_q(tk1, qk1, uk1_free) * qk1_q_dotk1
        )
        J_q_dotk1_u_dotk1 = -self.system.B(tk1, qk1) * uk1_free_u_dotk1

        ####################
        # euations of motion
        ####################
        J_u_dotk1_q_dotk1 = (
            self.system.Mu_q(tk1, qk1, uk1_free)
            - self.system.h_q(tk1, qk1, uk1_free)
            - self.system.Wla_g_q(tk1, qk1, la_gk1)
            - self.system.Wla_gamma_q(tk1, qk1, la_gammak1)
            - self.system.Wla_N_q(tk1, qk1, la_Nk1)
            - self.system.Wla_F_q(tk1, qk1, la_Fk1)
        ) * qk1_q_dotk1
        J_u_dotk1_u_dotk1 = Mk1 - self.system.h_u(tk1, qk1, uk1_free) * uk1_free_u_dotk1

        #######################
        # bilateral constraints
        #######################
        J_gk1_q_dotk1 = self.system.g_q(tk1, qk1) * qk1_q_dotk1
        J_gammak1_q_dotk1 = self.system.gamma_q(tk1, qk1, uk1_free) * qk1_q_dotk1
        J_gammak1_u_dotk1 = W_gammak1.T * uk1_free_u_dotk1

        ################
        # normal contact
        ################
        # note: csr_matrix is best for row slicing, see
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
        g_Nk1_qk1 = self.system.g_N_q(tk1, qk1, scipy_matrix=csr_matrix)

        J_la_Nk1_qk1 = lil_matrix((self.nla_N, self.nq))
        J_la_Nk1_la_Nk1 = lil_matrix((self.nla_N, self.nla_N))
        for i in range(self.nla_N):
            if self.I_N[i]:
                J_la_Nk1_qk1[i] = g_Nk1_qk1[i]
            else:
                J_la_Nk1_la_Nk1[i, i] = 1.0

        J_la_Nk1_q_dotk1 = J_la_Nk1_qk1 * qk1_q_dotk1

        ##########
        # friction
        ##########
        prox_r_F = self.system.prox_r_F(tk1, qk1)
        gamma_Fk1_qk1 = self.system.gamma_F_q(
            tk1, qk1, uk1_free, scipy_matrix=csr_matrix
        )
        gamma_Fk1_uk1 = W_Fk1.T

        J_la_Fk1_qk1 = lil_matrix((self.nla_F, self.nq))
        J_la_Fk1_uk1_free = lil_matrix((self.nla_F, self.nu))
        J_la_Fk1_la_Nk1 = lil_matrix((self.nla_F, self.nla_N))
        J_la_Fk1_la_Fk1 = lil_matrix((self.nla_F, self.nla_F))
        for i_N, i_F in enumerate(self.system.NF_connectivity):
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

        # Note: Keep this for checking used derivative if no convergence is obtained.
        Jx_num = csr_matrix(approx_fprime(xk1, self.R, method="2-point"))

        nq, nu, nla_g, nla_gamma, nla_N, nla_F = (
            self.nq,
            self.nu,
            self.nla_g,
            self.nla_gamma,
            self.nla_N,
            self.nla_F,
        )
        diff = (Jx - Jx_num).toarray()
        error = np.linalg.norm(diff)
        if error > 1.0e-6:
            print(f"error Jx: {error}")
        return Jx_num

    def step(self, xn1, f, G):
        # only compute optimized proxparameters once per time step
        self.prox_r_N = self.system.prox_r_N(self.tn, self.qn)
        self.prox_r_F = self.system.prox_r_F(self.tn, self.qn)

        # initial residual and error
        R = f(xn1, update_index=True)
        error = self.error_function(R)
        converged = error < self.tol

        # print(f"initial error: {error}")
        j = 0
        if not converged:
            while j < self.max_iter:
                # jacobian
                J = G(xn1)

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

                xn1 -= dx

                R = f(xn1, update_index=True)
                error = self.error_function(R)
                converged = error < self.tol
                if converged:
                    break

            if not converged:
                # raise RuntimeError("internal Newton-Raphson not converged")
                print(f"not converged!")

        return converged, j, error, xn1

    def solve(self):
        # lists storing output variables
        t = [self.tn]
        q = [self.qn]
        u = [self.un]
        q_dot = [self.q_dotn]
        u_dot = [self.u_dotn]
        P_g = [self.dt * self.la_gn]
        P_gamma = [self.dt * self.la_gamman]
        P_N = [self.dt * self.la_Nn]
        P_F = [self.dt * self.la_Fn]

        # prox_r_N = []
        # prox_r_F = []
        # error = []
        niter = [0]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # only compute optimized proxparameters once per time step
            self.prox_r_N = self.system.prox_r_N(self.tn, self.qn)
            self.prox_r_F = self.system.prox_r_F(self.tn, self.qn)

            # perform a sovler step
            tn1 = self.tn + self.dt

            yn1, converged, error, n_iter, _ = fsolve(
                self.R,
                self.yn,
                jac="2-point",
                eps=1e-6,
                fun_args=(True,),
                jac_args=(False,),
            )
            niter.append(n_iter)

            # update progress bar and check convergence
            pbar.set_description(
                f"t: {tn1:0.2e}s < {self.t1:0.2e}s; ||R||: {error:0.2e} ({n_iter}/{self.max_iter})"
            )
            if not converged:
                print(
                    f"internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}"
                )

                break

            q_dotn1, u_dotn1, la_gn1, la_gamman1, la_Nn1, la_Fn1 = np.array_split(
                yn1, self.split
            )
            qn1 = self.qn + self.dt * q_dotn1
            un1 = self.un + self.dt * u_dotn1

            # modify converged quantities
            qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            # store soltuion fields
            t.append(tn1)
            q.append(qn1)
            u.append(un1)
            q_dot.append(q_dotn1)
            u_dot.append(u_dotn1)
            P_g.append(self.dt * la_gn1)
            P_gamma.append(self.dt * la_gamman1)
            P_N.append(self.dt * la_Nn1)
            P_F.append(self.dt * la_Fn1)

            # update local variables for accepted time step
            self.yn = yn1.copy()
            self.tn = tn1
            self.qn = qn1
            self.un = un1

        # write solution
        return Solution(
            t=np.array(t),
            q=np.array(q),
            u=np.array(u),
            q_dot=np.array(q_dot),
            u_dot=np.array(u_dot),
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
            niter=np.array(niter),
        )
