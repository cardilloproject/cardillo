import numpy as np
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from cardillo.solver import Solution
from cardillo.math import norm, prox_R0_np, prox_sphere, approx_fprime


class SimplifiedGeneralizedAlpha:
    """Simplified generalized-alpha solver for mechanical systems with frictional contact."""

    def __init__(
        self,
        model,
        t1,
        dt,
        atol=1e-6,
        max_iter=100,
    ):
        self.model = model

        # initial time, final time, time step
        self.t0 = t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        # newton settings
        self.atol = atol
        self.max_iter = max_iter

        # dimensions (nq = number of coordinates q, etc.)
        self.nq = model.nq
        self.nu = model.nu
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F

        # eqn. (127): dimensions of residual
        self.nR_s = 3 * self.nu
        self.nR_c = 3 * self.nla_N + 2 * self.nla_F
        self.nR = self.nR_s + self.nR_c

        # set initial conditions
        self.tk = t0
        self.qk = model.q0
        self.uk = model.u0
        self.kappa_Nk = np.zeros_like(model.la_N0)
        self.la_Nk = model.la_N0
        self.La_Nk = np.zeros_like(model.la_N0)
        self.la_Fk = model.la_F0
        self.La_Fk = np.zeros_like(model.la_F0)
        # self.Qk = np.zeros(self.nu)
        self.Qk = model.q0
        self.Uk = np.zeros(self.nu)

        # solve for initial accelerations
        self.ak = spsolve(
            model.M(t0, model.q0, scipy_matrix=csr_matrix),
            self.model.h(t0, model.q0, model.u0)
            + self.model.W_N(t0, model.q0) @ self.model.la_N0
            + self.model.W_F(t0, model.q0) @ self.model.la_F0,
        )

        self.xk = np.concatenate(
            (
                self.ak,
                self.Uk,
                self.Qk,
                self.kappa_Nk,
                self.La_Nk,
                self.la_Nk,
                self.La_Fk,
                self.la_Fk,
            )
        )

        # initialize index sets
        self.Ak1 = np.zeros(self.nla_N, dtype=bool)
        self.Bk1 = np.zeros(self.nla_N, dtype=bool)
        self.Ck1 = np.zeros(self.nla_N, dtype=bool)
        self.Dk1_st = np.zeros(self.nla_N, dtype=bool)
        self.Ek1_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nu = self.nu
        nla_N = self.nla_N
        nla_F = self.nla_F
        nR_s = self.nR_s

        a = x[:nu]
        U = x[nu : 2 * nu]
        Q = x[2 * nu : 3 * nu]
        kappa_N = x[nR_s : nR_s + nla_N]
        La_N = x[nR_s + nla_N : nR_s + 2 * nla_N]
        la_N = x[nR_s + 2 * nla_N : nR_s + 3 * nla_N]
        La_F = x[nR_s + 3 * nla_N : nR_s + 3 * nla_N + nla_F]
        la_F = x[nR_s + 3 * nla_N + nla_F : nR_s + 3 * nla_N + 2 * nla_F]

        return a, U, Q, kappa_N, La_N, la_N, La_F, la_F

    def update(self, xk1):
        dt = self.dt
        dt2 = dt * dt

        ak1, Uk1, Qk1, kappa_Nk1, La_Nk1, la_Nk1, La_Fk1, la_Fk1 = self.unpack(xk1)
        q_dotk1 = Qk1

        uk1 = self.uk + dt * ak1 + Uk1
        # qk1 = self.qk + dt * self.uk + 0.5 * dt2 * ak1 + Qk1 + dt * Uk1
        # qk1 = self.qk + dt * self.uk + 0.5 * dt2 * ak1 + dt * Uk1
        # qk1 = self.qk + dt * self.uk + dt * Uk1
        qk1 = self.qk + dt * q_dotk1  # + 0.5 * dt2 * ak1
        # TODO: A possible solution would look like this:
        # qk1 = self.qk + dt * q_dotk1 + Qk1
        # thus we have the unknowns q_dotk1 and Qk1 similar to ak1 and Uk1!
        P_Nk1 = La_Nk1 + dt * la_Nk1
        kappa_hat_Nk1 = kappa_Nk1 + dt * La_Nk1 + 0.5 * dt2 * la_Nk1
        P_Fk1 = La_Fk1 + dt * la_Fk1

        return qk1, uk1, P_Nk1, kappa_hat_Nk1, P_Fk1

    def residual(self, tk1, xk1, update_index_set=False):
        mu = self.model.mu
        dt = self.dt

        ###########################
        # unpack vector of unknowns
        ###########################
        ak1, Uk1, Qk1, kappak1, La_Nk1, la_Nk1, La_Fk1, la_Fk1 = self.unpack(xk1)

        #############################
        # compute dependent variables
        #############################
        qk1, uk1, P_Nk1, kappa_hat_Nk1, P_Fk1 = self.update(xk1)

        # abuse Qk1 as qk1!
        q_dotk1 = Qk1

        ##############################
        # evaluate required quantities
        ##############################
        # mass matrix
        Mk1 = self.model.M(tk1, qk1)

        # vector of smooth generalized forces
        hk1 = self.model.h(tk1, qk1, uk1)

        # generalized force directions
        W_Nk1 = self.model.W_N(tk1, qk1)
        W_Fk1 = self.model.W_F(tk1, qk1)
        g_N_qk1 = self.model.g_N_q(tk1, qk1)
        gamma_F_qk1 = self.model.gamma_F_q(tk1, qk1, uk1)

        # kinematic quantities of normal contacts
        g_Nk1 = self.model.g_N(tk1, qk1)
        g_N_ddotk1 = self.model.g_N_ddot(tk1, qk1, uk1, ak1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)

        # kinematic quantities of frictional contacts
        gamma_Fk1 = self.model.gamma_F(tk1, qk1, uk1)
        gamma_F_dotk1 = self.model.gamma_F_dot(tk1, qk1, uk1, ak1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)

        #########################
        # compute residual vector
        #########################
        nu = self.nu
        nla_N = self.nla_N
        nR_s = self.nR_s
        R = np.empty(self.nR, dtype=xk1.dtype)

        # equations of motion
        R[:nu] = Mk1 @ ak1 - hk1 - W_Nk1 @ la_Nk1 - W_Fk1 @ la_Fk1

        # impact equation
        R[nu : 2 * nu] = Mk1 @ Uk1 - W_Nk1 @ La_Nk1 - W_Fk1 @ La_Fk1

        # position correction
        # R[2 * nu : 3 * nu] = Mk1 @ Qk1 - W_Nk1 @ kappak1 - 0.5 * dt * W_Fk1 @ La_Fk1
        # R[2 * nu : 3 * nu] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1) - g_N_qk1.T @ kappak1 - 0.5 * dt * gamma_F_qk1.T @ La_Fk1
        R[2 * nu : 3 * nu] = (
            q_dotk1
            - self.model.q_dot(tk1, qk1, self.uk + dt * ak1)
            - g_N_qk1.T @ kappak1
            - 0.5 * dt * gamma_F_qk1.T @ La_Fk1
        )

        prox_N_arg_position = g_Nk1 - self.model.prox_r_N * kappa_hat_Nk1
        prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        prox_N_arg_acceleration = g_N_ddotk1 - self.model.prox_r_N * la_Nk1
        if update_index_set:
            self.Ak1 = prox_N_arg_position <= 0
            self.Bk1 = self.Ak1 * (prox_N_arg_velocity <= 0)
            self.Ck1 = self.Bk1 * (prox_N_arg_acceleration <= 0)

            for i_N, i_F in enumerate(self.model.NF_connectivity):
                i_F = np.array(i_F)
                if len(i_F) > 0:
                    # eqn. (139):
                    self.Dk1_st[i_N] = self.Ak1[i_N] and (
                        norm(self.model.prox_r_F[i_N] * xi_Fk1[i_F] - P_Fk1[i_F])
                        <= mu[i_N] * P_Nk1[i_N]
                    )
                    # eqn. (141):
                    self.Ek1_st[i_N] = self.Dk1_st[i_N] and (
                        norm(
                            self.model.prox_r_F[i_N] * gamma_F_dotk1[i_F] - la_Fk1[i_F]
                        )
                        <= mu[i_N] * la_Nk1[i_N]
                    )

        ###################################
        # complementarity on position level
        ###################################
        Ak1 = self.Ak1
        Ak1_ind = np.where(Ak1)[0]
        _Ak1_ind = np.where(~Ak1)[0]
        R[nR_s + Ak1_ind] = g_Nk1[Ak1]
        R[nR_s + _Ak1_ind] = kappa_hat_Nk1[~Ak1]
        # R[nR_s : nR_s + nla_N] = g_Nk1 - prox_Rn0(prox_arg_position)

        ###################################
        # complementarity on velocity level
        ###################################
        Bk1 = self.Bk1
        Bk1_ind = np.where(Bk1)[0]
        _Bk1_ind = np.where(~Bk1)[0]
        R[nR_s + nla_N + Bk1_ind] = xi_Nk1[Bk1]
        R[nR_s + nla_N + _Bk1_ind] = P_Nk1[~Bk1]

        # R[nR_s + nla_N : nR_s + 2 * nla_N] = np.select(
        #     self.Ak1, xi_Nk1 - prox_Rn0(prox_arg_velocity), Pk1
        # )

        # R[nR_s + nla_N + Ak1_ind] = (xi_Nk1 - prox_Rn0(prox_arg_velocity))[Ak1]
        # R[nR_s + nla_N + _Ak1_ind] = Pk1[~Ak1]

        #######################################
        # complementarity on acceleration level
        #######################################
        Ck1 = self.Ck1
        Ck1_ind = np.where(Ck1)[0]
        _Ck1_ind = np.where(~Ck1)[0]
        R[nR_s + 2 * nla_N + Ck1_ind] = g_N_ddotk1[Ck1]
        R[nR_s + 2 * nla_N + _Ck1_ind] = la_Nk1[~Ck1]

        # R[nR_s + 2 * nla_N : nR_s + 3 * nla_N] = np.select(
        #     self.Bk1, g_N_ddotk1 - prox_Rn0(prox_arg_acceleration), lak1
        # )

        # R[nR_s + 2 * nla_N + Bk1_ind] = (g_N_ddotk1 - prox_Rn0(prox_arg_acceleration))[Bk1]
        # R[nR_s + 2 * nla_N + _Bk1_ind] = lak1[~Bk1]

        ##########
        # friction
        ##########
        Dk1_st = self.Dk1_st
        Ek1_st = self.Ek1_st
        nla_F = self.nla_F

        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                if Ak1[i_N]:
                    if Dk1_st[i_N]:
                        # eqn. (138a)
                        R[nR_s + 3 * nla_N + i_F] = xi_Fk1[i_F]

                        if Ek1_st[i_N]:
                            # eqn. (142a)
                            R[nR_s + 3 * nla_N + nla_F + i_F] = gamma_F_dotk1[i_F]
                        else:
                            # eqn. (142b)
                            norm_gamma_Fdoti1 = norm(gamma_F_dotk1[i_F])
                            gamma_F_dotk1_normalized = gamma_F_dotk1.copy()
                            if norm_gamma_Fdoti1 > 0:
                                gamma_F_dotk1_normalized /= norm_gamma_Fdoti1
                            R[nR_s + 3 * nla_N + nla_F + i_F] = (
                                la_Fk1[i_F]
                                + mu[i_N] * la_Nk1[i_N] * gamma_F_dotk1_normalized[i_F]
                            )
                    else:
                        # eqn. (138b)
                        norm_xi_Fi1 = norm(xi_Fk1[i_F])
                        xi_Fk1_normalized = xi_Fk1.copy()
                        if norm_xi_Fi1 > 0:
                            xi_Fk1_normalized /= norm_xi_Fi1
                        R[nR_s + 3 * nla_N + i_F] = (
                            P_Fk1[i_F] + mu[i_N] * P_Nk1[i_N] * xi_Fk1_normalized[i_F]
                        )

                        # eqn. (142c)
                        norm_gamma_Fi1 = norm(gamma_Fk1[i_F])
                        gamma_Fk1_normalized = gamma_Fk1.copy()
                        if norm_gamma_Fi1 > 0:
                            gamma_Fk1_normalized /= norm_gamma_Fi1
                        R[nR_s + 3 * nla_N + nla_F + i_F] = (
                            la_Fk1[i_F]
                            + mu[i_N] * la_Nk1[i_N] * gamma_Fk1_normalized[i_F]
                        )
                else:
                    # eqn. (138c)
                    R[nR_s + 3 * nla_N + i_F] = P_Fk1[i_F]
                    # eqn. (142d)
                    R[nR_s + 3 * nla_N + nla_F + i_F] = la_Fk1[i_F]

        return R

    def Jacobian(self, tk1, xk1):
        return csr_matrix(
            approx_fprime(xk1, lambda x: self.residual(tk1, x), method="2-point")
        )
        # return approx_fprime(xk1, lambda x: self.residual(tk1, x), method="cs")

    def solve(self):
        q = []
        u = []
        a = []
        Q = []
        U = []

        la_N = []
        La_N = []
        P_N = []
        kappa_N = []
        kappa_hat_N = []

        la_F = []
        La_F = []
        P_F = []

        def write_solution(xk1):
            ak1, Uk1, Qk1, kappa_Nk1, La_Nk1, la_Nk1, La_Fk1, la_Fk1 = self.unpack(xk1)
            qk1, uk1, P_Nk1, kappa_hat_Nk1, P_Fk1 = self.update(xk1)

            q_dotk1 = Qk1

            self.qk = qk1.copy()
            self.uk = uk1.copy()

            q.append(qk1.copy())
            u.append(uk1.copy())
            a.append(ak1.copy())
            Q.append(Qk1.copy())
            U.append(Uk1.copy())

            la_N.append(la_Nk1.copy())
            La_N.append(La_Nk1.copy())
            P_N.append(P_Nk1.copy())
            kappa_N.append(kappa_Nk1.copy())
            kappa_hat_N.append(kappa_hat_Nk1.copy())

            la_F.append(la_Fk1.copy())
            La_F.append(La_Fk1.copy())
            P_F.append(P_Fk1.copy())

        t = np.arange(self.t0, self.t1, self.dt)
        pbar = tqdm(t)

        xk1 = self.xk.copy()
        # for k, tk1 in enumerate(t):
        k = 0
        for tk1 in pbar:
            k += 1
            # print(f"k: {k}; tk1: {tk1:2.3f}")

            # initial residual and error; update active contact set during each
            # redidual computation
            R = self.residual(tk1, xk1, update_index_set=True)
            error = np.max(np.abs(R))

            j = 0
            if error < self.atol:
                write_solution(xk1)
            else:
                # Newton-Raphson loop
                for _ in range(self.max_iter):
                    j += 1

                    # compute Jacobian matrix with same index set
                    J = self.Jacobian(tk1, xk1)

                    # compute updated state
                    xk1 -= spsolve(J, R)

                    # new residual and error; update active contact set during
                    # each redidual computation
                    R = self.residual(tk1, xk1, update_index_set=True)
                    error = np.max(np.abs(R))

                    if error < self.atol:
                        pbar.set_description(
                            f"t: {tk1:0.2e}s < {self.t1:0.2e}s; {j}/{self.max_iter} iterations; error: {error:0.2e}"
                        )
                        write_solution(xk1)
                        break
            if j >= self.max_iter - 1:
                print(
                    f"Newton-Raphson not converged after {j+1} steps with error {error:2.4f}"
                )
                n = len(q)
                return Solution(
                    t=t[:n],
                    q=np.array(q),
                    u=np.array(u),
                    a=np.array(a),
                    Q=np.array(Q),
                    U=np.array(U),
                    la_N=np.array(la_N),
                    La_N=np.array(La_N),
                    P_N=np.array(P_N),
                    kappa_N=np.array(kappa_N),
                    kappa_hat_N=np.array(kappa_hat_N),
                    la_F=np.array(la_F),
                    La_F=np.array(La_F),
                    P_F=np.array(P_F),
                )
            else:
                # print(
                #     f"Newton-Raphson converged after {j+1} steps with error {error:2.4f}"
                # )
                pbar.set_description(
                    f"t: {tk1:0.2e}s < {self.t1:0.2e}s; {j}/{self.max_iter} iterations; error: {error:0.2e}"
                )

        n = len(q)
        return Solution(
            t=t[:n],
            q=np.array(q),
            u=np.array(u),
            a=np.array(a),
            Q=np.array(Q),
            U=np.array(U),
            la_N=np.array(la_N),
            La_N=np.array(La_N),
            P_N=np.array(P_N),
            kappa_N=np.array(kappa_N),
            kappa_hat_N=np.array(kappa_hat_N),
            la_F=np.array(la_F),
            La_F=np.array(La_F),
            P_F=np.array(P_F),
        )


# TODO: Add Newmark method (https://de.wikipedia.org/wiki/Newmark-beta-Verfahren#Herleitung).
# This gives us a ways to perform the double integrals (similar to gen alpha
# but without these ugly bar quantites)
gamma = 1.0 / 2.0
beta = 1.0 / 6.0
# gamma = 1.0
# beta = 0.5


class NonsmoothNewmarkFirstOrderTest:
    """Simplified generalized-alpha solver for mechanical systems in first
    order form with unilateral frictional contact."""

    def __init__(
        self,
        model,
        t1,
        dt,
        atol=1e-6,
        max_iter=20,
    ):
        self.model = model

        # initial time, final time, time step
        self.t0 = t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        # newton settings
        self.atol = atol
        self.max_iter = max_iter

        # dimensions (nq = number of coordinates q, etc.)
        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = model.nla_g
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F

        # eqn. (127): dimensions of residual
        self.nR_s = self.nq + 2 * self.nu + 2 * self.nla_g
        self.nR_c = 2 * self.nla_N + 2 * self.nla_F
        self.nR = self.nR_s + self.nR_c

        # set initial conditions
        self.tk = t0

        self.qk = model.q0
        self.uk = model.u0
        self.Uk = np.zeros(self.nu)

        self.la_gk = model.la_g0
        self.La_gk = np.zeros_like(model.la_g0)

        self.la_Nk = model.la_N0
        self.La_Nk = np.zeros_like(model.la_N0)

        self.la_Fk = model.la_F0
        self.La_Fk = np.zeros_like(model.la_F0)

        # compute initial velocity of generalized coordinates
        self.q_dotk = self.model.q_dot(t0, model.q0, model.u0)

        # solve for initial accelerations
        self.ak = spsolve(
            model.M(t0, model.q0, scipy_matrix=csr_matrix),
            self.model.h(t0, model.q0, model.u0)
            + self.model.W_g(t0, model.q0) @ self.model.la_g0
            + self.model.W_N(t0, model.q0) @ self.model.la_N0
            + self.model.W_F(t0, model.q0) @ self.model.la_F0,
        )

        self.xk = np.concatenate(
            (
                self.q_dotk,
                self.ak,
                self.Uk,
                self.La_gk,
                self.la_gk,
                self.La_Nk,
                self.la_Nk,
                self.La_Fk,
                self.la_Fk,
            )
        )

        # initialize index sets
        self.Ak1 = np.zeros(self.nla_N, dtype=bool)
        self.Bk1 = np.zeros(self.nla_N, dtype=bool)
        self.Ck1 = np.zeros(self.nla_N, dtype=bool)
        self.Dk1_st = np.zeros(self.nla_N, dtype=bool)
        self.Ek1_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_N = self.nla_N
        nla_F = self.nla_F
        nR_s = self.nR_s

        q_dot = x[:nq]

        a = x[nq : nq + nu]
        U = x[nq + nu : nq + 2 * nu]

        La_g = x[nq + 2 * nu : nq + 2 * nu + nla_g]
        la_g = x[nq + 2 * nu + nla_g : nq + 2 * nu + 2 * nla_g]

        La_N = x[nR_s : nR_s + nla_N]
        la_N = x[nR_s + nla_N : nR_s + 2 * nla_N]

        La_F = x[nR_s + 2 * nla_N : nR_s + 2 * nla_N + nla_F]
        la_F = x[nR_s + 2 * nla_N + nla_F : nR_s + 2 * nla_N + 2 * nla_F]

        return q_dot, a, U, La_g, la_g, La_N, la_N, La_F, la_F

    def update(self, xk1):
        dt = self.dt
        dt2 = dt * dt

        (
            q_dotk1,
            ak1,
            Uk1,
            La_gk1,
            la_gk1,
            La_Nk1,
            la_Nk1,
            La_Fk1,
            la_Fk1,
        ) = self.unpack(xk1)

        # update generalized coordinates and generalized velocities
        qk1 = self.qk + dt * q_dotk1
        uk1 = self.uk + dt * ak1 + Uk1
        # qk1 = self.qk + (1.0 - gamma) * dt * self.q_dotk + gamma * dt * q_dotk1
        # uk1 = self.uk + (1.0 - gamma) * dt * self.ak + gamma * dt * ak1 + Uk1

        # integrated bilateral constraint contributions
        P_gk1 = La_gk1 + dt * la_gk1
        # P_gk1 = La_gk1 + (1.0 - gamma) * dt * self.la_gk + gamma * dt * la_gk1
        # # kappa_hat_gk1 = (
        # #     kappa_gk1
        # #     + (1.0 - gamma) * dt * self.La_gk
        # #     + gamma * dt * La_gk1
        # #     + 0.5 * dt2 * la_gk1  # TODO: How do we want to approximate that integral?
        # # )
        # kappa_hat_gk1 = (
        #     kappa_gk1
        #     + (1.0 - gamma) * dt * self.La_gk
        #     + gamma * dt * La_gk1
        #     + dt2 * (0.5 - beta) * self.la_gk
        #     + dt2 * beta * la_gk1
        # )

        # integrated contact contributions
        P_Nk1 = La_Nk1 + dt * la_Nk1
        # kappa_hat_Nk1 = kappa_Nk1 + dt * La_Nk1 + 0.5 * dt2 * la_Nk1
        P_Fk1 = La_Fk1 + dt * la_Fk1
        # P_Nk1 = La_Nk1 + (1.0 - gamma) * dt * self.la_Nk + gamma * dt * la_Nk1
        # kappa_hat_Nk1 = (
        #     kappa_Nk1
        #     + (1.0 - gamma) * dt * self.La_Nk
        #     + gamma * dt * La_Nk1
        #     + dt2 * (0.5 - beta) * self.la_Nk
        #     + dt2 * beta * la_Nk1
        # )
        # P_Fk1 = La_Fk1 + (1.0 - gamma) * dt * self.la_Fk + gamma * dt * la_Fk1

        return qk1, uk1, P_gk1, P_Nk1, P_Fk1

    def residual(self, tk1, xk1, update_index_set=False):
        mu = self.model.mu
        dt = self.dt

        ###########################
        # unpack vector of unknowns
        ###########################
        (
            q_dotk1,
            ak1,
            Uk1,
            La_gk1,
            la_gk1,
            La_Nk1,
            la_Nk1,
            La_Fk1,
            la_Fk1,
        ) = self.unpack(xk1)

        #############################
        # compute dependent variables
        #############################
        qk1, uk1, P_gk1, P_Nk1, P_Fk1 = self.update(xk1)

        ##############################
        # evaluate required quantities
        ##############################
        # mass matrix
        Mk1 = self.model.M(tk1, qk1)

        # vector of smooth generalized forces
        hk1 = self.model.h(tk1, qk1, uk1)

        # generalized force directions
        W_gk1 = self.model.W_g(tk1, qk1)
        W_Nk1 = self.model.W_N(tk1, qk1)
        W_Fk1 = self.model.W_F(tk1, qk1)
        g_qk1 = self.model.g_q(tk1, qk1)
        g_N_qk1 = self.model.g_N_q(tk1, qk1)
        gamma_F_qk1 = self.model.gamma_F_q(tk1, qk1, uk1)

        # kinematic quantities of bilateral constraints
        gk1 = self.model.g(tk1, qk1)
        g_dotk1 = self.model.g_dot(tk1, qk1, uk1)
        # g_ddotk1 = self.model.g_ddot(tk1, qk1, uk1, ak1)

        # kinematic quantities of normal contacts
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        g_N_ddotk1 = self.model.g_N_ddot(tk1, qk1, uk1, ak1)

        # kinematic quantities of frictional contacts
        gamma_Fk1 = self.model.gamma_F(tk1, qk1, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)
        gamma_F_dotk1 = self.model.gamma_F_dot(tk1, qk1, uk1, ak1)

        #########################
        # compute residual vector
        #########################
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_N = self.nla_N
        nR_s = self.nR_s
        R = np.empty(self.nR, dtype=xk1.dtype)

        # TODO: Can we merge the smooth kinematic equation and the position
        # correction as done for the implicit Euler sheme? By that we would
        # have the same number of unknowns as the previous method but we can
        # deal with arbitrary kinematic equations implicitely.

        ####################
        # kinematic equation
        ####################
        # TODO: Why do we have to use the smooth velocity here?
        # This is similar to the case of an implicit Moreau scheme, where the
        # contact detecten is not allowed to be implicitely depending on qk1?
        # R[:nq] = (
        #     q_dotk1
        #     -self.model.q_dot(tk1, qk1, uk1)
        #     # -g_qk1.T @ la_gk1
        #     # -g_N_qk1.T @ la_Nk1
        #     # -gamma_F_qk1.T @ la_Fk1
        # )
        # This is the solution: Only use the smooth part of the velocity!
        # TODO: Why is this required if we do not use the integrated quantity
        # P_Nk1 = ... + 0.5 * dt2 * la_Nk1 in the update of P_Nk1
        # # R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, self.uk + dt * ak1)
        R[:nq] = q_dotk1 - self.model.q_dot(
            tk1, qk1, self.uk + (1.0 - gamma) * dt * self.ak + gamma * dt * ak1
        )

        ####################
        # equation of motion
        ####################
        R[nq : nq + nu] = (
            Mk1 @ ak1 - hk1 - W_gk1 @ la_gk1 - W_Nk1 @ la_Nk1 - W_Fk1 @ la_Fk1
        )

        #################
        # impact equation
        #################
        R[nq + nu : nq + 2 * nu] = (
            Mk1 @ Uk1 - W_gk1 @ La_gk1 - W_Nk1 @ La_Nk1 - W_Fk1 @ La_Fk1
        )

        ################################################
        # bilteral constraints on all kinematical levels
        ################################################
        R[nq + 2 * nu : nq + 2 * nu + nla_g] = gk1
        R[nq + 2 * nu + nla_g : nq + 2 * nu + 2 * nla_g] = g_dotk1

        #######################
        # unilateral index sets
        #######################
        # prox_N_arg_position = g_Nk1 - self.model.prox_r_N * kappa_hat_Nk1
        prox_N_arg_position = g_Nk1 - self.model.prox_r_N * la_Nk1
        # prox_N_arg_position = g_N_ddotk1 - self.model.prox_r_N * la_Nk1
        prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        if update_index_set:
            self.Ak1 = prox_N_arg_position <= 0
            self.Bk1 = self.Ak1 * (prox_N_arg_velocity <= 0)

            for i_N, i_F in enumerate(self.model.NF_connectivity):
                i_F = np.array(i_F)
                if len(i_F) > 0:
                    # eqn. (139):
                    self.Dk1_st[i_N] = self.Ak1[i_N] and (
                        norm(self.model.prox_r_F[i_N] * xi_Fk1[i_F] - P_Fk1[i_F])
                        <= mu[i_N] * P_Nk1[i_N]
                    )
                    # eqn. (141):
                    self.Ek1_st[i_N] = self.Dk1_st[i_N] and (
                        norm(
                            self.model.prox_r_F[i_N] * gamma_F_dotk1[i_F] - la_Fk1[i_F]
                        )
                        <= mu[i_N] * la_Nk1[i_N]
                    )

        ###################################
        # complementarity on position level
        ###################################
        Ak1 = self.Ak1
        Ak1_ind = np.where(Ak1)[0]
        _Ak1_ind = np.where(~Ak1)[0]
        R[nR_s + Ak1_ind] = g_Nk1[Ak1]
        # R[nR_s + _Ak1_ind] = kappa_hat_Nk1[~Ak1]
        R[nR_s + _Ak1_ind] = la_Nk1[~Ak1]
        # R[nR_s : nR_s + nla_N] = g_Nk1 - prox_R0_np(prox_N_arg_position)

        ###################################
        # complementarity on velocity level
        ###################################
        Bk1 = self.Bk1
        Bk1_ind = np.where(Bk1)[0]
        _Bk1_ind = np.where(~Bk1)[0]
        R[nR_s + nla_N + Bk1_ind] = xi_Nk1[Bk1]
        R[nR_s + nla_N + _Bk1_ind] = P_Nk1[~Bk1]

        # R[nR_s + nla_N : nR_s + 2 * nla_N] = np.select(
        #     self.Ak1, xi_Nk1 - prox_Rn0(prox_N_arg_velocity), P_Nk1
        # )

        # R[nR_s + nla_N + Ak1_ind] = (xi_Nk1 - prox_Rn0(prox_N_arg_velocity))[Ak1]
        # R[nR_s + nla_N + _Ak1_ind] = Pk1[~Ak1]

        ##########
        # friction
        ##########
        Dk1_st = self.Dk1_st
        Ek1_st = self.Ek1_st
        nla_F = self.nla_F

        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                if Ak1[i_N]:
                    if Dk1_st[i_N]:
                        # eqn. (138a)
                        R[nR_s + 2 * nla_N + i_F] = xi_Fk1[i_F]

                        if Ek1_st[i_N]:
                            # eqn. (142a)
                            R[nR_s + 2 * nla_N + nla_F + i_F] = gamma_F_dotk1[i_F]
                        else:
                            # eqn. (142b)
                            norm_gamma_Fdoti1 = norm(gamma_F_dotk1[i_F])
                            gamma_F_dotk1_normalized = gamma_F_dotk1.copy()
                            if norm_gamma_Fdoti1 > 0:
                                gamma_F_dotk1_normalized /= norm_gamma_Fdoti1
                            R[nR_s + 2 * nla_N + nla_F + i_F] = (
                                la_Fk1[i_F]
                                + mu[i_N] * la_Nk1[i_N] * gamma_F_dotk1_normalized[i_F]
                            )
                    else:
                        # eqn. (138b)
                        norm_xi_Fi1 = norm(xi_Fk1[i_F])
                        xi_Fk1_normalized = xi_Fk1.copy()
                        if norm_xi_Fi1 > 0:
                            xi_Fk1_normalized /= norm_xi_Fi1
                        R[nR_s + 2 * nla_N + i_F] = (
                            P_Fk1[i_F] + mu[i_N] * P_Nk1[i_N] * xi_Fk1_normalized[i_F]
                        )

                        # eqn. (142c)
                        norm_gamma_Fi1 = norm(gamma_Fk1[i_F])
                        gamma_Fk1_normalized = gamma_Fk1.copy()
                        if norm_gamma_Fi1 > 0:
                            gamma_Fk1_normalized /= norm_gamma_Fi1
                        R[nR_s + 2 * nla_N + nla_F + i_F] = (
                            la_Fk1[i_F]
                            + mu[i_N] * la_Nk1[i_N] * gamma_Fk1_normalized[i_F]
                        )
                else:
                    # eqn. (138c)
                    R[nR_s + 2 * nla_N + i_F] = P_Fk1[i_F]
                    # eqn. (142d)
                    R[nR_s + 2 * nla_N + nla_F + i_F] = la_Fk1[i_F]

        return R

    def Jacobian(self, tk1, xk1):
        return csr_matrix(
            approx_fprime(xk1, lambda x: self.residual(tk1, x), method="2-point")
        )

    def solve(self):
        q = []
        q_dot = []
        u = []
        a = []
        U = []

        la_g = []
        La_g = []
        P_g = []

        la_N = []
        La_N = []
        P_N = []

        la_F = []
        La_F = []
        P_F = []

        def write_solution(xk1):
            (
                q_dotk1,
                ak1,
                Uk1,
                La_gk1,
                la_gk1,
                La_Nk1,
                la_Nk1,
                La_Fk1,
                la_Fk1,
            ) = self.unpack(xk1)
            qk1, uk1, P_gk1, P_Nk1, P_Fk1 = self.update(xk1)

            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.q_dotk = q_dotk1.copy()
            self.ak = ak1.copy()

            self.la_gk = la_gk1.copy()
            self.la_Nk = la_Nk1.copy()
            self.la_Fk = la_Fk1.copy()

            self.La_gk = La_gk1.copy()
            self.La_Nk = La_Nk1.copy()
            self.La_Fk = La_Fk1.copy()

            q.append(qk1.copy())
            q_dot.append(q_dotk1.copy())
            u.append(uk1.copy())
            a.append(ak1.copy())
            U.append(Uk1.copy())

            la_g.append(la_gk1.copy())
            La_g.append(La_gk1.copy())
            P_g.append(P_gk1.copy())

            la_N.append(la_Nk1.copy())
            La_N.append(La_Nk1.copy())
            P_N.append(P_Nk1.copy())

            la_F.append(la_Fk1.copy())
            La_F.append(La_Fk1.copy())
            P_F.append(P_Fk1.copy())

        xk1 = self.xk.copy()
        write_solution(xk1)

        t = np.arange(self.t0, self.t1 + self.dt, self.dt)
        pbar = tqdm(t[:-1])
        # for tk1 in pbar:
        for k, tk1 in enumerate(pbar):

            # initial residual and error; update active contact set during each
            # redidual computation
            R = self.residual(tk1, xk1, update_index_set=True)
            error = np.max(np.abs(R))

            j = 0
            if error < self.atol:
                write_solution(xk1)
            else:
                # Newton-Raphson loop
                for _ in range(self.max_iter):
                    j += 1

                    # compute Jacobian matrix with same index set
                    J = self.Jacobian(tk1, xk1)

                    # compute updated state
                    xk1 -= spsolve(J, R)

                    # new residual and error; update active contact set during
                    # each redidual computation
                    R = self.residual(tk1, xk1, update_index_set=True)
                    error = np.max(np.abs(R))

                    if error < self.atol:
                        pbar.set_description(
                            f"t: {tk1:0.2e}s < {self.t1:0.2e}s; {j}/{self.max_iter} iterations; error: {error:0.2e}"
                        )
                        write_solution(xk1)
                        break
            if j >= self.max_iter - 1:
                print(
                    f"Newton-Raphson not converged after {j+1} steps with error {error:2.4f}"
                )
                n = len(q)
                return Solution(
                    t=t[:n],
                    q=np.array(q),
                    q_dot=np.array(q_dot),
                    u=np.array(u),
                    a=np.array(a),
                    U=np.array(U),
                    la_g=np.array(la_g),
                    La_g=np.array(La_g),
                    P_g=np.array(P_g),
                    la_N=np.array(la_N),
                    La_N=np.array(La_N),
                    P_N=np.array(P_N),
                    la_F=np.array(la_F),
                    La_F=np.array(La_F),
                    P_F=np.array(P_F),
                )
            else:
                pbar.set_description(
                    f"t: {tk1:0.2e}s < {self.t1:0.2e}s; {j}/{self.max_iter} iterations; error: {error:0.2e}"
                )

        n = len(q)
        return Solution(
            t=t[:n],
            q=np.array(q),
            q_dot=np.array(q_dot),
            u=np.array(u),
            a=np.array(a),
            U=np.array(U),
            la_g=np.array(la_g),
            La_g=np.array(La_g),
            P_g=np.array(P_g),
            la_N=np.array(la_N),
            La_N=np.array(La_N),
            P_N=np.array(P_N),
            la_F=np.array(la_F),
            La_F=np.array(La_F),
            P_F=np.array(P_F),
        )


class NonsmoothTheta:
    """Nonsmooth theta method for mechanical systems in first order form with
    unilateral frictional contact.

    TODO:
    -----
    - check for consistent initial conditions
    - Jacobian matrix
    - check friction
    """

    def __init__(
        self,
        model,
        t1,
        dt,
        atol=1e-6,
        max_iter=20,
    ):
        self.model = model

        # initial time, final time, time step
        self.t0 = t0 = model.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        # newton settings
        self.atol = atol
        self.max_iter = max_iter

        # dimensions (nq = number of coordinates q, etc.)
        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = model.nla_g
        self.nla_N = model.nla_N
        self.nla_F = model.nla_F

        # eqn. (127): dimensions of residual
        self.nR_s = 2 * self.nq + 2 * self.nu + 3 * self.nla_g
        self.nR_c = 3 * self.nla_N + 2 * self.nla_F
        self.nR = self.nR_s + self.nR_c

        # set initial conditions
        self.tk = t0

        self.qk = model.q0
        self.uk = model.u0
        self.Qk = np.zeros(self.nq)
        self.Uk = np.zeros(self.nu)

        self.kappa_gk = np.zeros_like(model.la_g0)
        self.la_gk = model.la_g0
        self.La_gk = np.zeros_like(model.la_g0)

        self.kappa_Nk = np.zeros_like(model.la_N0)
        self.la_Nk = model.la_N0
        self.La_Nk = np.zeros_like(model.la_N0)

        self.la_Fk = model.la_F0
        self.La_Fk = np.zeros_like(model.la_F0)

        # compute initial velocity of generalized coordinates
        self.q_dotk = self.model.q_dot(t0, model.q0, model.u0)

        # solve for initial accelerations
        self.u_dotk = spsolve(
            model.M(t0, model.q0, scipy_matrix=csr_matrix),
            self.model.h(t0, model.q0, model.u0)
            + self.model.W_g(t0, model.q0) @ self.model.la_g0
            + self.model.W_N(t0, model.q0) @ self.model.la_N0
            + self.model.W_F(t0, model.q0) @ self.model.la_F0,
        )

        self.xk = np.concatenate(
            (
                self.q_dotk,
                self.Qk,
                self.u_dotk,
                self.Uk,
                self.kappa_gk,
                self.La_gk,
                self.la_gk,
                self.kappa_Nk,
                self.La_Nk,
                self.la_Nk,
                self.La_Fk,
                self.la_Fk,
            )
        )

        # initialize index sets
        self.Ak1 = np.zeros(self.nla_N, dtype=bool)
        self.Bk1 = np.zeros(self.nla_N, dtype=bool)
        self.Ck1 = np.zeros(self.nla_N, dtype=bool)
        self.Dk1_st = np.zeros(self.nla_N, dtype=bool)
        self.Ek1_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_N = self.nla_N
        nla_F = self.nla_F
        nR_s = self.nR_s

        q_dot = x[:nq]
        Q = x[nq : 2 * nq]

        u_dot = x[2 * nq : 2 * nq + nu]
        U = x[2 * nq + nu : 2 * nq + 2 * nu]

        kappa_g = x[2 * nq + 2 * nu : 2 * nq + 2 * nu + nla_g]
        Gamma_g = x[2 * nq + 2 * nu + nla_g : 2 * nq + 2 * nu + 2 * nla_g]
        lambda_g = x[2 * nq + 2 * nu + 2 * nla_g : 2 * nq + 2 * nu + 3 * nla_g]

        kappa_N = x[nR_s : nR_s + nla_N]
        Gamma_N = x[nR_s + nla_N : nR_s + 2 * nla_N]
        lambda_N = x[nR_s + 2 * nla_N : nR_s + 3 * nla_N]

        Lambda_F = x[nR_s + 3 * nla_N : nR_s + 3 * nla_N + nla_F]
        lambda_F = x[nR_s + 3 * nla_N + nla_F : nR_s + 3 * nla_N + 2 * nla_F]

        return (
            q_dot,
            Q,
            u_dot,
            U,
            kappa_g,
            Gamma_g,
            lambda_g,
            kappa_N,
            Gamma_N,
            lambda_N,
            Lambda_F,
            lambda_F,
        )

    def update(self, xk1):
        dt = self.dt
        dt2 = dt * dt

        (
            q_dotk1,
            Qk1,
            u_dotk1,
            Uk1,
            kappa_gk1,
            Gamma_gk1,
            lambda_gk1,
            kappa_Nk1,
            Gamma_Nk1,
            lambda_Nk1,
            Gamma_Fk1,
            lambda_Fk1,
        ) = self.unpack(xk1)

        # update generalized coordinates and generalized velocities
        qk1 = self.qk + (1.0 - gamma) * dt * self.q_dotk + gamma * dt * q_dotk1 + Qk1
        uk1 = self.uk + (1.0 - gamma) * dt * self.u_dotk + gamma * dt * u_dotk1 + Uk1

        # integrated contact contributions

        # - simple version
        P_Nk1 = Gamma_Nk1 + dt * lambda_Nk1
        # TODO: last part (with 0.5 * dt2) is necessary for accumulation points!
        S_Nk1 = kappa_Nk1 + dt * Gamma_Nk1 + 0.5 * dt2 * lambda_Nk1
        P_Fk1 = Gamma_Fk1 + dt * lambda_Fk1

        # # - newmark type version
        # #   TODO: This is the cause for negative values of La_N after impacts!
        # P_Nk1 = Gamma_Nk1 + (1.0 - gamma) * dt * self.la_Nk + gamma * dt * lambda_Nk1
        # S_Nk1 = (
        #     kappa_Nk1
        #     + (1.0 - gamma) * dt * self.La_Nk
        #     + gamma * dt * Gamma_Nk1
        #     # TODO: This partis necessary for accumulation points!
        #     + dt2 * (0.5 - beta) * self.la_Nk
        #     + dt2 * beta * lambda_Nk1
        # )
        # P_Fk1 = Gamma_Fk1 + (1.0 - gamma) * dt * self.la_Fk + gamma * dt * lambda_Fk1

        return qk1, uk1, P_Nk1, S_Nk1, P_Fk1

    def residual(self, tk1, xk1, update_index_set=False):
        mu = self.model.mu
        dt = self.dt

        ###########################
        # unpack vector of unknowns
        ###########################
        (
            q_dotk1,
            Qk1,
            u_dotk1,
            Uk1,
            kappa_gk1,
            Gamma_gk1,
            lambda_gk1,
            kappa_Nk1,
            Gamma_Nk1,
            lambda_Nk1,
            Gamma_Fk1,
            lambda_Fk1,
        ) = self.unpack(xk1)

        #############################
        # compute dependent variables
        #############################
        qk1, uk1, P_Nk1, S_Nk1, P_Fk1 = self.update(xk1)

        ##############################
        # evaluate required quantities
        ##############################
        # mass matrix
        Mk1 = self.model.M(tk1, qk1)

        # vector of smooth generalized forces
        hk1 = self.model.h(tk1, qk1, uk1)

        # generalized force directions
        W_gk1 = self.model.W_g(tk1, qk1)
        W_Nk1 = self.model.W_N(tk1, qk1)
        W_Fk1 = self.model.W_F(tk1, qk1)
        g_qk1 = self.model.g_q(tk1, qk1)
        g_N_qk1 = self.model.g_N_q(tk1, qk1)
        gamma_F_qk1 = self.model.gamma_F_q(tk1, qk1, uk1)

        # kinematic quantities of bilateral constraints
        gk1 = self.model.g(tk1, qk1)
        g_dotk1 = self.model.g_dot(tk1, qk1, uk1)
        g_ddotk1 = self.model.g_ddot(tk1, qk1, uk1, u_dotk1)

        # kinematic quantities of normal contacts
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        g_N_ddotk1 = self.model.g_N_ddot(tk1, qk1, uk1, u_dotk1)

        # kinematic quantities of frictional contacts
        gamma_Fk1 = self.model.gamma_F(tk1, qk1, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)
        gamma_F_dotk1 = self.model.gamma_F_dot(tk1, qk1, uk1, u_dotk1)

        #########################
        # compute residual vector
        #########################
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_N = self.nla_N
        nR_s = self.nR_s
        R = np.empty(self.nR, dtype=xk1.dtype)

        # TODO: Can we merge the smooth kinematic equation and the position
        # correction as done for the implicit Euler sheme? By that we would
        # have the same number of unknowns as the previous method but we can
        # deal with arbitrary kinematic equations implicitely.

        ####################
        # kinematic equation
        ####################
        R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1)

        #####################
        # position correction
        #####################
        R[nq : 2 * nq] = (
            Qk1
            - g_qk1.T @ kappa_gk1
            - g_N_qk1.T @ kappa_Nk1
            # - 0.5 * dt * gamma_F_qk1.T @ La_Fk1 # TODO: Do we need the coupling with La_Fk1?
            # - (1.0 - gamma) * dt * self.model.gamma_F_q(self.tk, self.qk, self.uk).T @ self.La_Fk
            # - gamma * dt * gamma_F_qk1.T @ La_Fk1
        )

        # # Remove Qk1 solution by solving for 0 and add position correction
        # # directly to kinematic equation
        # R[nq : 2 * nq] = Qk1
        # R[:nq] = (
        #     q_dotk1
        #     - self.model.q_dot(tk1, qk1, uk1)
        #     - g_qk1.T @ kappa_gk1 / dt
        #     - g_N_qk1.T @ kappa_Nk1 / dt
        #     # - gamma_F_qk1.T @ La_Fk1 # TODO: Do we need the coupling with La_Fk1?
        # )

        ####################
        # equation of motion
        ####################
        R[2 * nq : 2 * nq + nu] = (
            Mk1 @ u_dotk1
            - hk1
            - W_gk1 @ lambda_gk1
            - W_Nk1 @ lambda_Nk1
            - W_Fk1 @ lambda_Fk1
        )

        #################
        # impact equation
        #################
        R[2 * nq + nu : 2 * nq + 2 * nu] = (
            Mk1 @ Uk1 - W_gk1 @ Gamma_gk1 - W_Nk1 @ Gamma_Nk1 - W_Fk1 @ Gamma_Fk1
        )

        ################################################
        # bilteral constraints on all kinematical levels
        ################################################
        R[2 * nq + 2 * nu : 2 * nq + 2 * nu + nla_g] = gk1
        R[2 * nq + 2 * nu + nla_g : 2 * nq + 2 * nu + 2 * nla_g] = g_dotk1
        R[2 * nq + 2 * nu + 2 * nla_g : 2 * nq + 2 * nu + 3 * nla_g] = g_ddotk1

        #######################
        # unilateral index sets
        #######################
        prox_N_arg_position = g_Nk1 - self.model.prox_r_N * S_Nk1
        prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        prox_N_arg_acceleration = g_N_ddotk1 - self.model.prox_r_N * lambda_Nk1
        if update_index_set:
            self.Ak1 = prox_N_arg_position <= 0
            self.Bk1 = self.Ak1 * (prox_N_arg_velocity <= 0)
            self.Ck1 = self.Bk1 * (prox_N_arg_acceleration <= 0)

            for i_N, i_F in enumerate(self.model.NF_connectivity):
                i_F = np.array(i_F)
                if len(i_F) > 0:
                    # eqn. (139):
                    self.Dk1_st[i_N] = self.Ak1[i_N] and (
                        norm(self.model.prox_r_F[i_N] * xi_Fk1[i_F] - P_Fk1[i_F])
                        <= mu[i_N] * P_Nk1[i_N]
                    )
                    # eqn. (141):
                    self.Ek1_st[i_N] = self.Dk1_st[i_N] and (
                        norm(
                            self.model.prox_r_F[i_N] * gamma_F_dotk1[i_F]
                            - lambda_Fk1[i_F]
                        )
                        <= mu[i_N] * lambda_Nk1[i_N]
                    )

        ###################################
        # complementarity on position level
        ###################################
        Ak1 = self.Ak1
        # Ak1_ind = np.where(Ak1)[0]
        # _Ak1_ind = np.where(~Ak1)[0]
        # R[nR_s + Ak1_ind] = g_Nk1[Ak1]
        # R[nR_s + _Ak1_ind] = S_Nk1[~Ak1]
        R[nR_s : nR_s + nla_N] = g_Nk1 - prox_R0_np(prox_N_arg_position)

        ###################################
        # complementarity on velocity level
        ###################################
        Bk1 = self.Bk1
        # Bk1_ind = np.where(Bk1)[0]
        # _Bk1_ind = np.where(~Bk1)[0]
        # R[nR_s + nla_N + Bk1_ind] = xi_Nk1[Bk1]
        # R[nR_s + nla_N + _Bk1_ind] = P_Nk1[~Bk1]

        R[nR_s + nla_N : nR_s + 2 * nla_N] = np.select(
            self.Ak1, xi_Nk1 - prox_R0_np(prox_N_arg_velocity), P_Nk1
        )

        # R[nR_s + nla_N + Ak1_ind] = (xi_Nk1 - prox_Rn0(prox_N_arg_velocity))[Ak1]
        # R[nR_s + nla_N + _Ak1_ind] = Pk1[~Ak1]

        #######################################
        # complementarity on acceleration level
        #######################################
        Ck1 = self.Ck1
        # Ck1_ind = np.where(Ck1)[0]
        # _Ck1_ind = np.where(~Ck1)[0]
        # R[nR_s + 2 * nla_N + Ck1_ind] = g_N_ddotk1[Ck1]
        # R[nR_s + 2 * nla_N + _Ck1_ind] = lambda_Nk1[~Ck1]

        R[nR_s + 2 * nla_N : nR_s + 3 * nla_N] = np.select(
            self.Bk1, g_N_ddotk1 - prox_R0_np(prox_N_arg_acceleration), lambda_Nk1
        )

        # R[nR_s + 2 * nla_N + Bk1_ind] = (g_N_ddotk1 - prox_Rn0(prox_arg_acceleration))[Bk1]
        # R[nR_s + 2 * nla_N + _Bk1_ind] = lak1[~Bk1]

        ##########
        # friction
        ##########
        Dk1_st = self.Dk1_st
        Ek1_st = self.Ek1_st
        nla_F = self.nla_F

        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                if Ak1[i_N]:
                    if Dk1_st[i_N]:
                        # eqn. (138a)
                        R[nR_s + 3 * nla_N + i_F] = xi_Fk1[i_F]

                        if Ek1_st[i_N]:
                            # eqn. (142a)
                            R[nR_s + 3 * nla_N + nla_F + i_F] = gamma_F_dotk1[i_F]
                        else:
                            # eqn. (142b)
                            norm_gamma_Fdoti1 = norm(gamma_F_dotk1[i_F])
                            gamma_F_dotk1_normalized = gamma_F_dotk1.copy()
                            if norm_gamma_Fdoti1 > 0:
                                gamma_F_dotk1_normalized /= norm_gamma_Fdoti1
                            R[nR_s + 3 * nla_N + nla_F + i_F] = (
                                lambda_Fk1[i_F]
                                + mu[i_N]
                                * lambda_Nk1[i_N]
                                * gamma_F_dotk1_normalized[i_F]
                            )
                    else:
                        # eqn. (138b)
                        norm_xi_Fi1 = norm(xi_Fk1[i_F])
                        xi_Fk1_normalized = xi_Fk1.copy()
                        if norm_xi_Fi1 > 0:
                            xi_Fk1_normalized /= norm_xi_Fi1
                        R[nR_s + 3 * nla_N + i_F] = (
                            P_Fk1[i_F] + mu[i_N] * P_Nk1[i_N] * xi_Fk1_normalized[i_F]
                        )

                        # eqn. (142c)
                        norm_gamma_Fi1 = norm(gamma_Fk1[i_F])
                        gamma_Fk1_normalized = gamma_Fk1.copy()
                        if norm_gamma_Fi1 > 0:
                            gamma_Fk1_normalized /= norm_gamma_Fi1
                        R[nR_s + 3 * nla_N + nla_F + i_F] = (
                            lambda_Fk1[i_F]
                            + mu[i_N] * lambda_Nk1[i_N] * gamma_Fk1_normalized[i_F]
                        )
                else:
                    # eqn. (138c)
                    R[nR_s + 3 * nla_N + i_F] = P_Fk1[i_F]
                    # eqn. (142d)
                    R[nR_s + 3 * nla_N + nla_F + i_F] = lambda_Fk1[i_F]

        return R

    def Jacobian(self, tk1, xk1):
        return csr_matrix(
            approx_fprime(xk1, lambda x: self.residual(tk1, x), method="2-point")
        )

    def solve(self):
        q = []
        q_dot = []
        u = []
        a = []
        Q = []
        U = []

        la_g = []
        La_g = []

        la_N = []
        La_N = []
        P_N = []
        kappa_N = []
        kappa_hat_N = []

        la_F = []
        La_F = []
        P_F = []

        def write_solution(xk1):
            (
                q_dotk1,
                Qk1,
                ak1,
                Uk1,
                kappa_gk1,
                La_gk1,
                la_gk1,
                kappa_Nk1,
                La_Nk1,
                la_Nk1,
                La_Fk1,
                la_Fk1,
            ) = self.unpack(xk1)
            qk1, uk1, P_Nk1, kappa_hat_Nk1, P_Fk1 = self.update(xk1)

            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.q_dotk = q_dotk1.copy()
            self.u_dotk = ak1.copy()

            self.la_gk = la_gk1.copy()
            self.la_Nk = la_Nk1.copy()
            self.la_Fk = la_Fk1.copy()

            self.La_gk = La_gk1.copy()
            self.La_Nk = La_Nk1.copy()
            self.La_Fk = La_Fk1.copy()

            q.append(qk1.copy())
            q_dot.append(q_dotk1.copy())
            u.append(uk1.copy())
            a.append(ak1.copy())
            Q.append(Qk1.copy())
            U.append(Uk1.copy())

            la_g.append(la_gk1.copy())
            La_g.append(La_gk1.copy())
            # kappa_g.append(kappa_gk1.copy())
            # kappa_hat_g.append(kappa_hat_gk1.copy())

            la_N.append(la_Nk1.copy())
            La_N.append(La_Nk1.copy())
            P_N.append(P_Nk1.copy())
            kappa_N.append(kappa_Nk1.copy())
            kappa_hat_N.append(kappa_hat_Nk1.copy())

            la_F.append(la_Fk1.copy())
            La_F.append(La_Fk1.copy())
            P_F.append(P_Fk1.copy())

        xk1 = self.xk.copy()
        write_solution(xk1)

        t = np.arange(self.t0, self.t1 + self.dt, self.dt)
        pbar = tqdm(t[:-1])
        for k, tk1 in enumerate(pbar):
            # # TODO: Use Euler forward as predictor?
            # qk1 = self.qk + self.dt * self.model.q_dot(tk1, self.qk, self.uk)
            # # uk1 = self.uk + self.dt * spsolve(...)
            # xk1[:self.nq] = qk1

            # initial residual and error; update active contact set during each
            # redidual computation
            R = self.residual(tk1, xk1, update_index_set=True)
            error = np.max(np.abs(R))

            j = 0
            if error < self.atol:
                write_solution(xk1)
            else:
                # Newton-Raphson loop
                for _ in range(self.max_iter):
                    j += 1

                    # compute Jacobian matrix with same index set
                    J = self.Jacobian(tk1, xk1)

                    # compute updated state
                    xk1 -= spsolve(J, R)

                    # new residual and error; update active contact set during
                    # each redidual computation
                    R = self.residual(tk1, xk1, update_index_set=True)
                    error = np.max(np.abs(R))

                    if error < self.atol:
                        pbar.set_description(
                            f"t: {tk1:0.2e}s < {self.t1:0.2e}s; {j}/{self.max_iter} iterations; error: {error:0.2e}"
                        )
                        write_solution(xk1)
                        break
            if j >= self.max_iter - 1:
                print(
                    f"Newton-Raphson not converged after {j+1} steps with error {error:2.4f}"
                )
                n = len(q)
                return Solution(
                    t=t[:n],
                    q=np.array(q),
                    q_dot=np.array(q_dot),
                    u=np.array(u),
                    a=np.array(a),
                    Q=np.array(Q),
                    U=np.array(U),
                    la_g=np.array(la_g),
                    La_g=np.array(La_g),
                    la_N=np.array(la_N),
                    La_N=np.array(La_N),
                    P_N=np.array(P_N),
                    kappa_N=np.array(kappa_N),
                    kappa_hat_N=np.array(kappa_hat_N),
                    la_F=np.array(la_F),
                    La_F=np.array(La_F),
                    P_F=np.array(P_F),
                )
            else:
                pbar.set_description(
                    f"t: {tk1:0.2e}s < {self.t1:0.2e}s; {j}/{self.max_iter} iterations; error: {error:0.2e}"
                )

        n = len(q)
        return Solution(
            t=t[:n],
            q=np.array(q),
            q_dot=np.array(q_dot),
            u=np.array(u),
            a=np.array(a),
            Q=np.array(Q),
            U=np.array(U),
            la_g=np.array(la_g),
            La_g=np.array(La_g),
            la_N=np.array(la_N),
            La_N=np.array(La_N),
            P_N=np.array(P_N),
            kappa_N=np.array(kappa_N),
            kappa_hat_N=np.array(kappa_hat_N),
            la_F=np.array(la_F),
            La_F=np.array(La_F),
            P_F=np.array(P_F),
        )


class NonsmoothGenAlphaFirstOrder:
    """Nonsmooth generalized-alpha method for mechanical systems in first
    order form with unilateral frictional contact.

    TODO:
    -----
    - bilateral constraints on velocity level
    - check friction
    - check for consistent initial conditions
    - Jacobian matrix
    """

    def __init__(
        self,
        model,
        t1,
        dt,
        rho_inf=1,
        atol=1e-10,
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
        # gen alpha parameter
        #######################################################################
        self.rho_inf = rho_inf
        self.alpha_m = (3.0 * rho_inf - 1.0) / (2.0 * (rho_inf + 1.0))
        self.alpha_f = rho_inf / (rho_inf + 1.0)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m
        self.beta = 0.25 * (0.5 + self.gamma) ** 2

        #######################################################################
        # newton settings
        #######################################################################
        self.atol = atol
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

        # dimensions of residual
        self.nR_s = 2 * self.nq + 2 * self.nu + 3 * self.nla_g
        self.nR_c = 3 * self.nla_N + 2 * self.nla_F
        self.nR = self.nR_s + self.nR_c

        ####################
        # initial conditions
        ####################
        q0 = model.q0
        u0 = model.u0

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

        # set initial conditions
        self.tk = t0

        self.qk = model.q0
        self.uk = model.u0
        self.Qk = np.zeros(self.nq)
        self.Uk = np.zeros(self.nu)

        self.kappa_gk = np.zeros(self.nla_g)
        self.la_gk = la_g0
        self.La_gk = np.zeros(self.nla_g)

        self.la_gammak = la_gamma0
        self.La_gammak = np.zeros(self.nla_gamma)

        self.kappa_Nk = np.zeros(self.nla_N)
        self.la_Nk = model.la_N0
        self.la_Nbark = self.la_Nk.copy()
        self.La_Nk = np.zeros(self.nla_N)

        self.la_Fk = model.la_F0
        self.la_Fbark = model.la_F0.copy()
        self.La_Fk = np.zeros(self.nla_F)

        # compute initial velocity of generalized coordinates
        self.q_dotk = self.model.q_dot(t0, q0, u0)

        # solve for initial accelerations
        self.u_dotk = u_dot0

        self.xk = np.concatenate(
            (
                self.q_dotk,
                self.Qk,
                self.u_dotk,
                self.Uk,
                self.kappa_gk,
                self.La_gk,
                self.la_gk,
                self.kappa_Nk,
                self.La_Nk,
                self.la_Nk,
                self.La_Fk,
                self.la_Fk,
            )
        )

        # auxiliary derivatives related to q_dot and u_dot
        self.vk = self.q_dotk.copy()
        self.ak = self.u_dotk.copy()

        # check if initial conditions satisfy constraints on position, velocity
        # and acceleration level
        # TODO: unilateral contacts!
        g0 = model.g(self.tk, self.qk)
        g_dot0 = model.g_dot(self.tk, self.qk, self.uk)
        g_ddot0 = model.g_ddot(self.tk, self.qk, self.uk, self.u_dotk)
        gamma0 = model.gamma(self.tk, self.qk, self.uk)
        gamma_dot0 = model.gamma_dot(self.tk, self.qk, self.uk, self.u_dotk)
        # g_N0 = model.g_N(self.tk, self.qk)
        # g_N_dot0 = model.g_N_dot(self.tk, self.qk, self.uk)
        # g_N_ddot0 = model.g_N_ddot(self.tk, self.qk, self.uk, self.u_dotk)

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

        # initialize index sets
        self.Ak1 = np.zeros(self.nla_N, dtype=bool)
        self.Bk1 = np.zeros(self.nla_N, dtype=bool)
        self.Ck1 = np.zeros(self.nla_N, dtype=bool)
        self.Dk1_st = np.zeros(self.nla_N, dtype=bool)
        self.Ek1_st = np.zeros(self.nla_N, dtype=bool)

    def pack(self, *args):
        return np.concatenate([*args])

    def unpack(self, x):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_N = self.nla_N
        nla_F = self.nla_F
        nR_s = self.nR_s

        q_dot = x[:nq]
        Q = x[nq : 2 * nq]

        u_dot = x[2 * nq : 2 * nq + nu]
        U = x[2 * nq + nu : 2 * nq + 2 * nu]

        kappa_g = x[2 * nq + 2 * nu : 2 * nq + 2 * nu + nla_g]
        Gamma_g = x[2 * nq + 2 * nu + nla_g : 2 * nq + 2 * nu + 2 * nla_g]
        lambda_g = x[2 * nq + 2 * nu + 2 * nla_g : 2 * nq + 2 * nu + 3 * nla_g]

        kappa_N = x[nR_s : nR_s + nla_N]
        Gamma_N = x[nR_s + nla_N : nR_s + 2 * nla_N]
        lambda_N = x[nR_s + 2 * nla_N : nR_s + 3 * nla_N]

        Lambda_F = x[nR_s + 3 * nla_N : nR_s + 3 * nla_N + nla_F]
        lambda_F = x[nR_s + 3 * nla_N + nla_F : nR_s + 3 * nla_N + 2 * nla_F]

        return (
            q_dot,
            Q,
            u_dot,
            U,
            kappa_g,
            Gamma_g,
            lambda_g,
            kappa_N,
            Gamma_N,
            lambda_N,
            Lambda_F,
            lambda_F,
        )

    def update(self, xk1, store=False):
        """Update dependent variables."""

        # constants
        dt = self.dt
        dt2 = dt * dt
        gamma = self.gamma
        alpha_f = self.alpha_f
        alpha_m = self.alpha_m

        (
            q_dotk1,
            Qk1,
            u_dotk1,
            Uk1,
            kappa_gk1,
            Gamma_gk1,
            lambda_gk1,
            kappa_Nk1,
            Gamma_Nk1,
            lambda_Nk1,
            Gamma_Fk1,
            lambda_Fk1,
        ) = self.unpack(xk1)

        # compute auxiliary derivatives from q_dotk1, u_dotk1, lambda_N and lambda_F
        vk1 = (
            alpha_f * self.q_dotk + (1.0 - alpha_f) * q_dotk1 - alpha_m * self.vk
        ) / (1.0 - alpha_m)
        ak1 = (
            alpha_f * self.u_dotk + (1.0 - alpha_f) * u_dotk1 - alpha_m * self.ak
        ) / (1.0 - alpha_m)
        lambda_Nbark1 = (
            alpha_f * self.la_Nk
            + (1.0 - alpha_f) * lambda_Nk1
            - alpha_m * self.la_Nbark
        ) / (1.0 - alpha_m)
        lambda_Fbark1 = (
            alpha_f * self.la_Fk
            + (1.0 - alpha_f) * lambda_Fk1
            - alpha_m * self.la_Fbark
        ) / (1.0 - alpha_m)

        # Newmark update using v and a + stabilization with Q and U
        qk1 = self.qk + dt * ((1.0 - gamma) * self.vk + gamma * vk1) + Qk1
        uk1 = self.uk + dt * ((1.0 - gamma) * self.ak + gamma * ak1) + Uk1

        ##################################
        # integrated contact contributions
        ##################################

        # - simple version
        P_Nk1 = Gamma_Nk1 + dt * lambda_Nk1
        # TODO: last part (with 0.5 * dt2) is necessary for accumulation points!
        S_Nk1 = kappa_Nk1 + dt * Gamma_Nk1 + 0.5 * dt2 * lambda_Nk1
        P_Fk1 = Gamma_Fk1 + dt * lambda_Fk1

        # # - Generalized-alpha version
        # P_Nk1 = (
        #     Gamma_Nk1 + (1.0 - gamma) * dt * self.la_Nbark + gamma * dt * lambda_Nbark1
        # )
        # S_Nk1 = (
        #     kappa_Nk1
        #     + dt2 * (0.5 - self.beta) * self.la_Nbark
        #     + dt2 * self.beta * lambda_Nbark1
        # )
        # P_Fk1 = (
        #     Gamma_Fk1 + (1.0 - gamma) * dt * self.la_Fbark + gamma * dt * lambda_Fbark1
        # )

        if store:
            self.q_dotk = q_dotk1.copy()
            self.u_dotk = u_dotk1.copy()
            self.vk = vk1.copy()
            self.ak = ak1.copy()
            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.la_Nk = lambda_Nk1.copy()
            self.La_Nk = Gamma_Nk1.copy()
            self.la_Fk = lambda_Fk1.copy()
            self.la_Nbark = lambda_Nbark1.copy()
            self.la_Fbark = lambda_Fbark1.copy()

        return qk1, uk1, P_Nk1, S_Nk1, P_Fk1

    def residual(self, tk1, xk1, update_index_set=False):
        mu = self.model.mu
        dt = self.dt

        ###########################
        # unpack vector of unknowns
        ###########################
        (
            q_dotk1,
            Qk1,
            u_dotk1,
            Uk1,
            kappa_gk1,
            Gamma_gk1,
            lambda_gk1,
            kappa_Nk1,
            Gamma_Nk1,
            lambda_Nk1,
            Gamma_Fk1,
            lambda_Fk1,
        ) = self.unpack(xk1)

        #############################
        # compute dependent variables
        #############################
        qk1, uk1, P_Nk1, S_Nk1, P_Fk1 = self.update(xk1, store=False)

        ##############################
        # evaluate required quantities
        ##############################
        # mass matrix
        Mk1 = self.model.M(tk1, qk1)

        # vector of smooth generalized forces
        hk1 = self.model.h(tk1, qk1, uk1)

        # generalized force directions
        W_gk1 = self.model.W_g(tk1, qk1)
        W_Nk1 = self.model.W_N(tk1, qk1)
        W_Fk1 = self.model.W_F(tk1, qk1)
        g_qk1 = self.model.g_q(tk1, qk1)
        g_N_qk1 = self.model.g_N_q(tk1, qk1)
        gamma_F_qk1 = self.model.gamma_F_q(tk1, qk1, uk1)

        # kinematic quantities of bilateral constraints
        gk1 = self.model.g(tk1, qk1)
        g_dotk1 = self.model.g_dot(tk1, qk1, uk1)
        g_ddotk1 = self.model.g_ddot(tk1, qk1, uk1, u_dotk1)

        # kinematic quantities of normal contacts
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        g_N_ddotk1 = self.model.g_N_ddot(tk1, qk1, uk1, u_dotk1)

        # kinematic quantities of frictional contacts
        gamma_Fk1 = self.model.gamma_F(tk1, qk1, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)
        gamma_F_dotk1 = self.model.gamma_F_dot(tk1, qk1, uk1, u_dotk1)

        #########################
        # compute residual vector
        #########################
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_N = self.nla_N
        nR_s = self.nR_s
        R = np.empty(self.nR, dtype=xk1.dtype)

        ####################
        # kinematic equation
        ####################
        R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, uk1)

        #####################
        # position correction
        #####################
        R[nq : 2 * nq] = (
            Qk1
            - g_qk1.T @ kappa_gk1
            - g_N_qk1.T @ kappa_Nk1
            # - 0.5 * dt * gamma_F_qk1.T @ Gamma_Fk1 # TODO: Do we need the coupling with La_Fk1?
            # - (1.0 - gamma) * dt * self.model.gamma_F_q(self.tk, self.qk, self.uk).T @ self.La_Fk
            # - gamma * dt * gamma_F_qk1.T @ La_Fk1
        )

        # # Remove Qk1 solution by solving for 0 and add position correction
        # # directly to kinematic equation
        # R[nq : 2 * nq] = Qk1
        # # R[:nq] = (
        # #     q_dotk1
        # #     - self.model.q_dot(tk1, qk1, uk1)
        # #     - g_qk1.T @ kappa_gk1 / dt
        # #     - g_N_qk1.T @ kappa_Nk1 / dt
        # #     # - gamma_F_qk1.T @ La_Fk1 # TODO: Do we need the coupling with La_Fk1?
        # # )
        # R[:nq] = (
        #     dt * q_dotk1
        #     - dt * self.model.q_dot(tk1, qk1, uk1)
        #     # - dt * self.model.q_dot(tk1, qk1, uk1 - Uk1)
        #     - g_qk1.T @ kappa_gk1
        #     - g_N_qk1.T @ kappa_Nk1
        # )

        ####################
        # equation of motion
        ####################
        R[2 * nq : 2 * nq + nu] = (
            Mk1 @ u_dotk1
            - hk1
            - W_gk1 @ lambda_gk1
            - W_Nk1 @ lambda_Nk1
            - W_Fk1 @ lambda_Fk1
        )

        #################
        # impact equation
        #################
        R[2 * nq + nu : 2 * nq + 2 * nu] = (
            Mk1 @ Uk1 - W_gk1 @ Gamma_gk1 - W_Nk1 @ Gamma_Nk1 - W_Fk1 @ Gamma_Fk1
        )

        ################################################
        # bilteral constraints on all kinematical levels
        ################################################
        R[2 * nq + 2 * nu : 2 * nq + 2 * nu + nla_g] = gk1
        R[2 * nq + 2 * nu + nla_g : 2 * nq + 2 * nu + 2 * nla_g] = g_dotk1
        R[2 * nq + 2 * nu + 2 * nla_g : 2 * nq + 2 * nu + 3 * nla_g] = g_ddotk1

        #######################
        # unilateral index sets
        #######################
        prox_N_arg_position = g_Nk1 - self.model.prox_r_N * S_Nk1
        prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        prox_N_arg_acceleration = g_N_ddotk1 - self.model.prox_r_N * lambda_Nk1
        if update_index_set:
            self.Ak1 = prox_N_arg_position <= 0
            self.Bk1 = self.Ak1 * (prox_N_arg_velocity <= 0)
            self.Ck1 = self.Bk1 * (prox_N_arg_acceleration <= 0)

            for i_N, i_F in enumerate(self.model.NF_connectivity):
                i_F = np.array(i_F)
                if len(i_F) > 0:
                    # eqn. (139):
                    self.Dk1_st[i_N] = self.Ak1[i_N] and (
                        norm(self.model.prox_r_F[i_N] * xi_Fk1[i_F] - P_Fk1[i_F])
                        <= mu[i_N] * P_Nk1[i_N]
                    )
                    # eqn. (141):
                    self.Ek1_st[i_N] = self.Dk1_st[i_N] and (
                        norm(
                            self.model.prox_r_F[i_N] * gamma_F_dotk1[i_F]
                            - lambda_Fk1[i_F]
                        )
                        <= mu[i_N] * lambda_Nk1[i_N]
                    )

        ###################################
        # complementarity on position level
        ###################################
        Ak1 = self.Ak1
        Ak1_ind = np.where(Ak1)[0]
        _Ak1_ind = np.where(~Ak1)[0]
        R[nR_s + Ak1_ind] = g_Nk1[Ak1]
        R[nR_s + _Ak1_ind] = S_Nk1[~Ak1]

        # R[nR_s : nR_s + nla_N] = g_Nk1 - prox_R0_np(prox_N_arg_position)

        ###################################
        # complementarity on velocity level
        ###################################
        Bk1 = self.Bk1
        Bk1_ind = np.where(Bk1)[0]
        _Bk1_ind = np.where(~Bk1)[0]
        R[nR_s + nla_N + Bk1_ind] = xi_Nk1[Bk1]
        R[nR_s + nla_N + _Bk1_ind] = P_Nk1[~Bk1]

        # R[nR_s + nla_N : nR_s + 2 * nla_N] = np.select(
        #     self.Ak1, xi_Nk1 - prox_R0_np(prox_N_arg_velocity), P_Nk1
        # )

        # R[nR_s + nla_N + Ak1_ind] = (xi_Nk1 - prox_Rn0(prox_N_arg_velocity))[Ak1]
        # R[nR_s + nla_N + _Ak1_ind] = Pk1[~Ak1]

        #######################################
        # complementarity on acceleration level
        #######################################
        Ck1 = self.Ck1
        Ck1_ind = np.where(Ck1)[0]
        _Ck1_ind = np.where(~Ck1)[0]
        R[nR_s + 2 * nla_N + Ck1_ind] = g_N_ddotk1[Ck1]
        R[nR_s + 2 * nla_N + _Ck1_ind] = lambda_Nk1[~Ck1]

        # R[nR_s + 2 * nla_N : nR_s + 3 * nla_N] = np.select(
        #     self.Bk1, g_N_ddotk1 - prox_R0_np(prox_N_arg_acceleration), lambda_Nk1
        # )

        # R[nR_s + 2 * nla_N + Bk1_ind] = (g_N_ddotk1 - prox_Rn0(prox_arg_acceleration))[Bk1]
        # R[nR_s + 2 * nla_N + _Bk1_ind] = lak1[~Bk1]

        ##########
        # friction
        ##########
        Dk1_st = self.Dk1_st
        Ek1_st = self.Ek1_st
        nla_F = self.nla_F

        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                if Ak1[i_N]:
                    if Dk1_st[i_N]:
                        # eqn. (138a)
                        R[nR_s + 3 * nla_N + i_F] = xi_Fk1[i_F]

                        if Ek1_st[i_N]:
                            # eqn. (142a)
                            R[nR_s + 3 * nla_N + nla_F + i_F] = gamma_F_dotk1[i_F]
                        else:
                            # eqn. (142b)
                            norm_gamma_Fdoti1 = norm(gamma_F_dotk1[i_F])
                            gamma_F_dotk1_normalized = gamma_F_dotk1.copy()
                            if norm_gamma_Fdoti1 > 0:
                                gamma_F_dotk1_normalized /= norm_gamma_Fdoti1
                            R[nR_s + 3 * nla_N + nla_F + i_F] = (
                                lambda_Fk1[i_F]
                                + mu[i_N]
                                * lambda_Nk1[i_N]
                                * gamma_F_dotk1_normalized[i_F]
                            )
                    else:
                        # eqn. (138b)
                        norm_xi_Fi1 = norm(xi_Fk1[i_F])
                        xi_Fk1_normalized = xi_Fk1.copy()
                        if norm_xi_Fi1 > 0:
                            xi_Fk1_normalized /= norm_xi_Fi1
                        R[nR_s + 3 * nla_N + i_F] = (
                            P_Fk1[i_F] + mu[i_N] * P_Nk1[i_N] * xi_Fk1_normalized[i_F]
                        )

                        # eqn. (142c)
                        norm_gamma_Fi1 = norm(gamma_Fk1[i_F])
                        gamma_Fk1_normalized = gamma_Fk1.copy()
                        if norm_gamma_Fi1 > 0:
                            gamma_Fk1_normalized /= norm_gamma_Fi1
                        R[nR_s + 3 * nla_N + nla_F + i_F] = (
                            lambda_Fk1[i_F]
                            + mu[i_N] * lambda_Nk1[i_N] * gamma_Fk1_normalized[i_F]
                        )
                else:
                    # eqn. (138c)
                    R[nR_s + 3 * nla_N + i_F] = P_Fk1[i_F]
                    # eqn. (142d)
                    R[nR_s + 3 * nla_N + nla_F + i_F] = lambda_Fk1[i_F]

        return R

    def Jacobian(self, tk1, xk1):
        return csr_matrix(
            # approx_fprime(xk1, lambda x: self.residual(tk1, x), method="2-point")
            approx_fprime(
                xk1, lambda x: self.residual(tk1, x), method="2-point", eps=1.0e-7
            )
        )

    def solve(self):
        q = []
        q_dot = []
        u = []
        a = []
        Q = []
        U = []

        kappa_g = []
        La_g = []
        la_g = []

        kappa_N = []
        La_N = []
        la_N = []
        P_N = []
        kappa_hat_N = []

        P_F = []
        La_F = []
        la_F = []

        def write_solution(tk1, xk1):
            (
                q_dotk1,
                Qk1,
                ak1,
                Uk1,
                kappa_gk1,
                La_gk1,
                la_gk1,
                kappa_Nk1,
                La_Nk1,
                la_Nk1,
                La_Fk1,
                la_Fk1,
            ) = self.unpack(xk1)

            qk1, uk1, P_Nk1, kappa_hat_Nk1, P_Fk1 = self.update(xk1, store=True)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            # self.qk = qk1.copy()
            # self.uk = uk1.copy()
            # self.q_dotk = q_dotk1.copy()
            # self.u_dotk = ak1.copy()

            # self.la_gk = la_gk1.copy()
            # self.la_Nk = la_Nk1.copy()
            # self.la_Fk = la_Fk1.copy()

            # self.La_gk = La_gk1.copy()
            # self.La_Nk = La_Nk1.copy()
            # self.La_Fk = La_Fk1.copy()

            q.append(qk1.copy())
            q_dot.append(q_dotk1.copy())
            u.append(uk1.copy())
            a.append(ak1.copy())
            Q.append(Qk1.copy())
            U.append(Uk1.copy())

            la_g.append(la_gk1.copy())
            La_g.append(La_gk1.copy())
            kappa_g.append(kappa_gk1.copy())
            # kappa_hat_g.append(kappa_hat_gk1.copy())

            la_N.append(la_Nk1.copy())
            La_N.append(La_Nk1.copy())
            P_N.append(P_Nk1.copy())
            kappa_N.append(kappa_Nk1.copy())
            kappa_hat_N.append(kappa_hat_Nk1.copy())

            la_F.append(la_Fk1.copy())
            La_F.append(La_Fk1.copy())
            P_F.append(P_Fk1.copy())

        xk1 = self.xk.copy()
        write_solution(self.t0, xk1)

        t = np.arange(self.t0, self.t1 + self.dt, self.dt)
        pbar = tqdm(t[:-1])
        for k, tk1 in enumerate(pbar):
            # # TODO: Use Euler forward as predictor?
            # qk1 = self.qk + self.dt * self.model.q_dot(tk1, self.qk, self.uk)
            # # uk1 = self.uk + self.dt * spsolve(...)
            # xk1[:self.nq] = qk1

            # initial residual and error; update active contact set during each
            # redidual computation
            R = self.residual(tk1, xk1, update_index_set=True)
            error = np.max(np.abs(R))

            j = 0
            if error < self.atol:
                write_solution(tk1, xk1)
            else:
                # Newton-Raphson loop
                for _ in range(self.max_iter):
                    j += 1

                    # compute Jacobian matrix with same index set
                    J = self.Jacobian(tk1, xk1)

                    # compute updated state
                    xk1 -= spsolve(J, R)

                    # new residual and error; update active contact set during
                    # each redidual computation
                    R = self.residual(tk1, xk1, update_index_set=True)
                    error = np.max(np.abs(R))

                    if error < self.atol:
                        pbar.set_description(
                            f"t: {tk1:0.2e}s < {self.t1:0.2e}s; {j}/{self.max_iter} iterations; error: {error:0.2e}"
                        )
                        write_solution(tk1, xk1)
                        break
            if j >= self.max_iter - 1:
                print(
                    f"Newton-Raphson not converged after {j+1} steps with error {error:2.4f}"
                )
                n = len(q)
                return Solution(
                    t=t[:n],
                    q=np.array(q),
                    q_dot=np.array(q_dot),
                    u=np.array(u),
                    a=np.array(a),
                    Q=np.array(Q),
                    U=np.array(U),
                    la_g=np.array(la_g),
                    La_g=np.array(La_g),
                    la_N=np.array(la_N),
                    La_N=np.array(La_N),
                    P_N=np.array(P_N),
                    kappa_N=np.array(kappa_N),
                    kappa_hat_N=np.array(kappa_hat_N),
                    la_F=np.array(la_F),
                    La_F=np.array(La_F),
                    P_F=np.array(P_F),
                )
            else:
                pbar.set_description(
                    f"t: {tk1:0.2e}s < {self.t1:0.2e}s; {j}/{self.max_iter} iterations; error: {error:0.2e}"
                )

        n = len(q)
        return Solution(
            t=t[:n],
            q=np.array(q),
            q_dot=np.array(q_dot),
            u=np.array(u),
            a=np.array(a),
            Q=np.array(Q),
            U=np.array(U),
            la_g=np.array(la_g),
            La_g=np.array(La_g),
            la_N=np.array(la_N),
            La_N=np.array(La_N),
            P_N=np.array(P_N),
            kappa_N=np.array(kappa_N),
            kappa_hat_N=np.array(kappa_hat_N),
            la_F=np.array(la_F),
            La_F=np.array(La_F),
            P_F=np.array(P_F),
        )


class NonsmoothNewmark:
    """Nonsmooth Newmark method for mechanical systems in with unilateral
    frictional contact, see
    https://de.wikipedia.org/wiki/Newmark-beta-Verfahren#Herleitung.

    TODO:
    -----
    - bilateral constraints on velocity level
    - check friction
    - check for consistent initial conditions
    - Jacobian matrix
    """

    def __init__(
        self,
        model,
        t1,
        dt,
        atol=1e-10,
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
        # gen alpha parameter
        #######################################################################
        self.gamma = 1.0 / 2.0
        self.beta = 1.0 / 6.0

        #######################################################################
        # newton settings
        #######################################################################
        self.atol = atol
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

        # dimensions of residual
        self.nR_s = self.nq + 2 * self.nu + 3 * self.nla_g
        self.nR_c = 3 * self.nla_N + 2 * self.nla_F
        self.nR = self.nR_s + self.nR_c

        ####################
        # initial conditions
        ####################
        q0 = model.q0
        u0 = model.u0

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

        # set initial conditions
        self.tk = t0

        self.qk = model.q0
        self.uk = model.u0
        self.Qk = np.zeros(self.nq)
        self.Uk = np.zeros(self.nu)

        self.kappa_gk = np.zeros(self.nla_g)
        self.la_gk = la_g0
        self.La_gk = np.zeros(self.nla_g)

        self.la_gammak = la_gamma0
        self.La_gammak = np.zeros(self.nla_gamma)

        self.kappa_Nk = np.zeros(self.nla_N)
        self.la_Nk = model.la_N0
        # self.la_Nbark = self.la_Nk.copy()
        self.La_Nk = np.zeros(self.nla_N)

        self.la_Fk = model.la_F0
        # self.la_Fbark = model.la_F0.copy()
        self.La_Fk = np.zeros(self.nla_F)

        # compute initial velocity of generalized coordinates
        self.q_dotk = self.model.q_dot(t0, q0, u0)

        # solve for initial accelerations
        self.u_dotk = u_dot0

        self.xk = np.concatenate(
            (
                self.u_dotk,
                self.Uk,
                self.Qk,
                self.kappa_gk,
                self.La_gk,
                self.la_gk,
                self.kappa_Nk,
                self.La_Nk,
                self.la_Nk,
                self.La_Fk,
                self.la_Fk,
            )
        )

        # # auxiliary derivatives related to q_dot and u_dot
        # self.vk = self.q_dotk.copy()
        # self.ak = self.u_dotk.copy()

        # check if initial conditions satisfy constraints on position, velocity
        # and acceleration level
        # TODO: unilateral contacts!
        g0 = model.g(self.tk, self.qk)
        g_dot0 = model.g_dot(self.tk, self.qk, self.uk)
        g_ddot0 = model.g_ddot(self.tk, self.qk, self.uk, self.u_dotk)
        gamma0 = model.gamma(self.tk, self.qk, self.uk)
        gamma_dot0 = model.gamma_dot(self.tk, self.qk, self.uk, self.u_dotk)
        # g_N0 = model.g_N(self.tk, self.qk)
        # g_N_dot0 = model.g_N_dot(self.tk, self.qk, self.uk)
        # g_N_ddot0 = model.g_N_ddot(self.tk, self.qk, self.uk, self.u_dotk)

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

        # initialize index sets
        self.Ak1 = np.zeros(self.nla_N, dtype=bool)
        self.Bk1 = np.zeros(self.nla_N, dtype=bool)
        self.Ck1 = np.zeros(self.nla_N, dtype=bool)
        self.Dk1_st = np.zeros(self.nla_N, dtype=bool)
        self.Ek1_st = np.zeros(self.nla_N, dtype=bool)

    def unpack(self, x):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_N = self.nla_N
        nla_F = self.nla_F
        nR_s = self.nR_s

        u_dot = x[:nu]
        U = x[nu : 2 * nu]
        Q = x[2 * nu : 2 * nu + nq]

        kappa_g = x[2 * nu + nq : 2 * nu + nq + nla_g]
        Gamma_g = x[2 * nu + nq + nla_g : 2 * nu + nq + 2 * nla_g]
        lambda_g = x[2 * nu + nq + 2 * nla_g : 2 * nu + nq + 3 * nla_g]

        kappa_N = x[nR_s : nR_s + nla_N]
        Gamma_N = x[nR_s + nla_N : nR_s + 2 * nla_N]
        lambda_N = x[nR_s + 2 * nla_N : nR_s + 3 * nla_N]

        Lambda_F = x[nR_s + 3 * nla_N : nR_s + 3 * nla_N + nla_F]
        lambda_F = x[nR_s + 3 * nla_N + nla_F : nR_s + 3 * nla_N + 2 * nla_F]

        return (
            u_dot,
            U,
            Q,
            kappa_g,
            Gamma_g,
            lambda_g,
            kappa_N,
            Gamma_N,
            lambda_N,
            Lambda_F,
            lambda_F,
        )

    def update(self, xk1, store=False):
        """Update dependent variables."""

        # constants
        dt = self.dt
        dt2 = dt * dt
        gamma = self.gamma
        beta = self.beta

        (
            u_dotk1,
            Uk1,
            Qk1,
            kappa_gk1,
            Gamma_gk1,
            lambda_gk1,
            kappa_Nk1,
            Gamma_Nk1,
            lambda_Nk1,
            Gamma_Fk1,
            lambda_Fk1,
        ) = self.unpack(xk1)

        # # compute auxiliary derivatives from q_dotk1, u_dotk1, lambda_N and lambda_F
        # vk1 = (
        #     alpha_f * self.q_dotk + (1.0 - alpha_f) * q_dotk1 - alpha_m * self.vk
        # ) / (1.0 - alpha_m)
        # ak1 = (
        #     alpha_f * self.u_dotk + (1.0 - alpha_f) * u_dotk1 - alpha_m * self.ak
        # ) / (1.0 - alpha_m)
        # lambda_Nbark1 = (
        #     alpha_f * self.la_Nk
        #     + (1.0 - alpha_f) * lambda_Nk1
        #     - alpha_m * self.la_Nbark
        # ) / (1.0 - alpha_m)
        # lambda_Fbark1 = (
        #     alpha_f * self.la_Fk
        #     + (1.0 - alpha_f) * lambda_Fk1
        #     - alpha_m * self.la_Fbark
        # ) / (1.0 - alpha_m)

        # Newmark update for velocities + stabilization
        uk1 = self.uk + dt * ((1.0 - gamma) * self.u_dotk + gamma * u_dotk1) + Uk1

        # Newmark update for generalized coordinates + stabilization
        Delta_uk1 = self.uk + self.dt * ((0.5 - beta) * self.u_dotk + beta * u_dotk1)
        qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, Delta_uk1) + Qk1

        ##################################
        # integrated contact contributions
        ##################################

        # # - simple version
        # P_Nk1 = Gamma_Nk1 + dt * lambda_Nk1
        # # TODO: last part (with 0.5 * dt2) is necessary for accumulation points!
        # S_Nk1 = kappa_Nk1 + dt * Gamma_Nk1 + 0.5 * dt2 * lambda_Nk1
        # P_Fk1 = Gamma_Fk1 + dt * lambda_Fk1

        # - Newmark version
        P_Nk1 = Gamma_Nk1 + (1.0 - gamma) * dt * self.la_Nk + gamma * dt * lambda_Nk1
        S_Nk1 = (
            kappa_Nk1
            + (1.0 - gamma) * dt * self.La_Nk
            + gamma * dt * Gamma_Nk1
            + dt2 * (0.5 - self.beta) * self.la_Nk
            + dt2 * self.beta * lambda_Nk1
        )
        P_Fk1 = Gamma_Fk1 + (1.0 - gamma) * dt * self.la_Fk + gamma * dt * lambda_Fk1

        if store:
            self.tk += dt
            self.qk = qk1.copy()
            self.uk = uk1.copy()
            self.u_dotk = u_dotk1.copy()
            self.la_Nk = lambda_Nk1.copy()
            self.la_Fk = lambda_Fk1.copy()
            self.La_Nk = Gamma_Nk1.copy()
            self.La_Fk = Gamma_Fk1.copy()

        return qk1, uk1, P_Nk1, S_Nk1, P_Fk1

    def residual(self, tk1, xk1, update_index_set=False):
        mu = self.model.mu
        dt = self.dt

        ###########################
        # unpack vector of unknowns
        ###########################
        (
            u_dotk1,
            Uk1,
            Qk1,
            kappa_gk1,
            Gamma_gk1,
            lambda_gk1,
            kappa_Nk1,
            Gamma_Nk1,
            lambda_Nk1,
            Gamma_Fk1,
            lambda_Fk1,
        ) = self.unpack(xk1)

        #############################
        # compute dependent variables
        #############################
        qk1, uk1, P_Nk1, S_Nk1, P_Fk1 = self.update(xk1, store=False)

        ##############################
        # evaluate required quantities
        ##############################
        # mass matrix
        Mk1 = self.model.M(tk1, qk1)

        # vector of smooth generalized forces
        hk1 = self.model.h(tk1, qk1, uk1)

        # generalized force directions
        W_gk1 = self.model.W_g(tk1, qk1)
        W_Nk1 = self.model.W_N(tk1, qk1)
        W_Fk1 = self.model.W_F(tk1, qk1)
        g_qk1 = self.model.g_q(tk1, qk1)
        g_N_qk1 = self.model.g_N_q(tk1, qk1)
        gamma_F_qk1 = self.model.gamma_F_q(tk1, qk1, uk1)

        # kinematic quantities of bilateral constraints
        gk1 = self.model.g(tk1, qk1)
        g_dotk1 = self.model.g_dot(tk1, qk1, uk1)
        g_ddotk1 = self.model.g_ddot(tk1, qk1, uk1, u_dotk1)

        # kinematic quantities of normal contacts
        g_Nk1 = self.model.g_N(tk1, qk1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        g_N_ddotk1 = self.model.g_N_ddot(tk1, qk1, uk1, u_dotk1)

        # kinematic quantities of frictional contacts
        gamma_Fk1 = self.model.gamma_F(tk1, qk1, uk1)
        xi_Fk1 = self.model.xi_F(tk1, qk1, self.uk, uk1)
        gamma_F_dotk1 = self.model.gamma_F_dot(tk1, qk1, uk1, u_dotk1)

        #########################
        # compute residual vector
        #########################
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_N = self.nla_N
        nR_s = self.nR_s
        R = np.empty(self.nR, dtype=xk1.dtype)

        ####################
        # equation of motion
        ####################
        R[:nu] = (
            Mk1 @ u_dotk1
            - hk1
            - W_gk1 @ lambda_gk1
            - W_Nk1 @ lambda_Nk1
            - W_Fk1 @ lambda_Fk1
        )

        #################
        # impact equation
        #################
        R[nu : 2 * nu] = (
            Mk1 @ Uk1 - W_gk1 @ Gamma_gk1 - W_Nk1 @ Gamma_Nk1 - W_Fk1 @ Gamma_Fk1
        )

        #####################
        # position correction
        #####################
        R[2 * nu : 2 * nu + nq] = (
            Qk1
            - g_qk1.T @ kappa_gk1
            - g_N_qk1.T @ kappa_Nk1
            # # TODO: Do we need the coupling with La_Fk1?
            # # - 0.5 * dt * gamma_F_qk1.T @ Gamma_Fk1
            # - (1.0 - gamma) * dt * self.model.gamma_F_q(self.tk, self.qk, self.uk).T @ self.La_Fk
            # - gamma * dt * gamma_F_qk1.T @ Gamma_Fk1
        )

        ################################################
        # bilteral constraints on all kinematical levels
        ################################################
        R[2 * nu + nq : 2 * nu + nq + nla_g] = gk1
        R[2 * nu + nq + nla_g : 2 * nu + nq + 2 * nla_g] = g_dotk1
        R[2 * nu + nq + 2 * nla_g : 2 * nu + nq + 3 * nla_g] = g_ddotk1

        #######################
        # unilateral index sets
        #######################
        prox_N_arg_position = g_Nk1 - self.model.prox_r_N * S_Nk1
        prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        prox_N_arg_acceleration = g_N_ddotk1 - self.model.prox_r_N * lambda_Nk1
        if update_index_set:
            self.Ak1 = prox_N_arg_position <= 0
            self.Bk1 = self.Ak1 * (prox_N_arg_velocity <= 0)
            self.Ck1 = self.Bk1 * (prox_N_arg_acceleration <= 0)

            for i_N, i_F in enumerate(self.model.NF_connectivity):
                i_F = np.array(i_F)
                if len(i_F) > 0:
                    # eqn. (139):
                    self.Dk1_st[i_N] = self.Ak1[i_N] and (
                        norm(self.model.prox_r_F[i_N] * xi_Fk1[i_F] - P_Fk1[i_F])
                        <= mu[i_N] * P_Nk1[i_N]
                    )
                    # eqn. (141):
                    self.Ek1_st[i_N] = self.Dk1_st[i_N] and (
                        norm(
                            self.model.prox_r_F[i_N] * gamma_F_dotk1[i_F]
                            - lambda_Fk1[i_F]
                        )
                        <= mu[i_N] * lambda_Nk1[i_N]
                    )

        ###################################
        # complementarity on position level
        ###################################
        Ak1 = self.Ak1
        Ak1_ind = np.where(Ak1)[0]
        _Ak1_ind = np.where(~Ak1)[0]
        R[nR_s + Ak1_ind] = g_Nk1[Ak1]
        R[nR_s + _Ak1_ind] = S_Nk1[~Ak1]

        # R[nR_s : nR_s + nla_N] = g_Nk1 - prox_R0_np(prox_N_arg_position)

        ###################################
        # complementarity on velocity level
        ###################################
        Bk1 = self.Bk1
        Bk1_ind = np.where(Bk1)[0]
        _Bk1_ind = np.where(~Bk1)[0]
        R[nR_s + nla_N + Bk1_ind] = xi_Nk1[Bk1]
        R[nR_s + nla_N + _Bk1_ind] = P_Nk1[~Bk1]

        # R[nR_s + nla_N : nR_s + 2 * nla_N] = np.select(
        #     self.Ak1, xi_Nk1 - prox_R0_np(prox_N_arg_velocity), P_Nk1
        # )

        # R[nR_s + nla_N + Ak1_ind] = (xi_Nk1 - prox_Rn0(prox_N_arg_velocity))[Ak1]
        # R[nR_s + nla_N + _Ak1_ind] = Pk1[~Ak1]

        #######################################
        # complementarity on acceleration level
        #######################################
        Ck1 = self.Ck1
        Ck1_ind = np.where(Ck1)[0]
        _Ck1_ind = np.where(~Ck1)[0]
        R[nR_s + 2 * nla_N + Ck1_ind] = g_N_ddotk1[Ck1]
        R[nR_s + 2 * nla_N + _Ck1_ind] = lambda_Nk1[~Ck1]

        # R[nR_s + 2 * nla_N : nR_s + 3 * nla_N] = np.select(
        #     self.Bk1, g_N_ddotk1 - prox_R0_np(prox_N_arg_acceleration), lambda_Nk1
        # )

        # R[nR_s + 2 * nla_N + Bk1_ind] = (g_N_ddotk1 - prox_Rn0(prox_arg_acceleration))[Bk1]
        # R[nR_s + 2 * nla_N + _Bk1_ind] = lak1[~Bk1]

        ##########
        # friction
        ##########
        Dk1_st = self.Dk1_st
        Ek1_st = self.Ek1_st
        nla_F = self.nla_F

        for i_N, i_F in enumerate(self.model.NF_connectivity):
            i_F = np.array(i_F)
            if len(i_F) > 0:
                if Ak1[i_N]:
                    if Dk1_st[i_N]:
                        # eqn. (138a)
                        R[nR_s + 3 * nla_N + i_F] = xi_Fk1[i_F]

                        if Ek1_st[i_N]:
                            # eqn. (142a)
                            R[nR_s + 3 * nla_N + nla_F + i_F] = gamma_F_dotk1[i_F]
                        else:
                            # eqn. (142b)
                            norm_gamma_Fdoti1 = norm(gamma_F_dotk1[i_F])
                            gamma_F_dotk1_normalized = gamma_F_dotk1.copy()
                            if norm_gamma_Fdoti1 > 0:
                                gamma_F_dotk1_normalized /= norm_gamma_Fdoti1
                            R[nR_s + 3 * nla_N + nla_F + i_F] = (
                                lambda_Fk1[i_F]
                                + mu[i_N]
                                * lambda_Nk1[i_N]
                                * gamma_F_dotk1_normalized[i_F]
                            )
                    else:
                        # eqn. (138b)
                        norm_xi_Fi1 = norm(xi_Fk1[i_F])
                        xi_Fk1_normalized = xi_Fk1.copy()
                        if norm_xi_Fi1 > 0:
                            xi_Fk1_normalized /= norm_xi_Fi1
                        R[nR_s + 3 * nla_N + i_F] = (
                            P_Fk1[i_F] + mu[i_N] * P_Nk1[i_N] * xi_Fk1_normalized[i_F]
                        )

                        # eqn. (142c)
                        norm_gamma_Fi1 = norm(gamma_Fk1[i_F])
                        gamma_Fk1_normalized = gamma_Fk1.copy()
                        if norm_gamma_Fi1 > 0:
                            gamma_Fk1_normalized /= norm_gamma_Fi1
                        R[nR_s + 3 * nla_N + nla_F + i_F] = (
                            lambda_Fk1[i_F]
                            + mu[i_N] * lambda_Nk1[i_N] * gamma_Fk1_normalized[i_F]
                        )
                else:
                    # eqn. (138c)
                    R[nR_s + 3 * nla_N + i_F] = P_Fk1[i_F]
                    # eqn. (142d)
                    R[nR_s + 3 * nla_N + nla_F + i_F] = lambda_Fk1[i_F]

        return R

    def Jacobian(self, tk1, xk1):
        return csr_matrix(
            # approx_fprime(xk1, lambda x: self.residual(tk1, x), method="2-point")
            # approx_fprime(xk1, lambda x: self.residual(tk1, x), method="3-point")
            approx_fprime(
                xk1, lambda x: self.residual(tk1, x), method="2-point", eps=1.0e-7
            )
        )

    def solve(self):
        q = []
        u = []
        u_dot = []
        Q = []
        U = []

        kappa_g = []
        La_g = []
        la_g = []

        kappa_N = []
        La_N = []
        la_N = []
        P_N = []
        kappa_hat_N = []

        P_F = []
        La_F = []
        la_F = []

        def write_solution(tk1, xk1):
            (
                ak1,
                Uk1,
                Qk1,
                kappa_gk1,
                La_gk1,
                la_gk1,
                kappa_Nk1,
                La_Nk1,
                la_Nk1,
                La_Fk1,
                la_Fk1,
            ) = self.unpack(xk1)

            qk1, uk1, P_Nk1, kappa_hat_Nk1, P_Fk1 = self.update(xk1, store=True)

            # modify converged quantities
            qk1, uk1 = self.model.step_callback(tk1, qk1, uk1)

            q.append(qk1.copy())
            u.append(uk1.copy())
            u_dot.append(ak1.copy())
            Q.append(Qk1.copy())
            U.append(Uk1.copy())

            la_g.append(la_gk1.copy())
            La_g.append(La_gk1.copy())
            kappa_g.append(kappa_gk1.copy())
            # kappa_hat_g.append(kappa_hat_gk1.copy())

            la_N.append(la_Nk1.copy())
            La_N.append(La_Nk1.copy())
            P_N.append(P_Nk1.copy())
            kappa_N.append(kappa_Nk1.copy())
            kappa_hat_N.append(kappa_hat_Nk1.copy())

            la_F.append(la_Fk1.copy())
            La_F.append(La_Fk1.copy())
            P_F.append(P_Fk1.copy())

        xk1 = self.xk.copy()
        write_solution(self.t0, xk1)

        t = np.arange(self.t0, self.t1 + self.dt, self.dt)
        pbar = tqdm(t[:-1])
        for k, tk1 in enumerate(pbar):
            # # TODO: Use Euler forward as predictor?
            # qk1 = self.qk + self.dt * self.model.q_dot(tk1, self.qk, self.uk)
            # # uk1 = self.uk + self.dt * spsolve(...)
            # xk1[:self.nq] = qk1

            # initial residual and error; update active contact set during each
            # redidual computation
            R = self.residual(tk1, xk1, update_index_set=True)
            error = np.max(np.abs(R))

            j = 0
            if error < self.atol:
                write_solution(tk1, xk1)
            else:
                # Newton-Raphson loop
                for _ in range(self.max_iter):
                    j += 1

                    # compute Jacobian matrix with same index set
                    J = self.Jacobian(tk1, xk1)

                    # compute updated state
                    xk1 -= spsolve(J, R)

                    # new residual and error; update active contact set during
                    # each redidual computation
                    R = self.residual(tk1, xk1, update_index_set=True)
                    error = np.max(np.abs(R))

                    if error < self.atol:
                        pbar.set_description(
                            f"t: {tk1:0.2e}s < {self.t1:0.2e}s; {j}/{self.max_iter} iterations; error: {error:0.2e}"
                        )
                        write_solution(tk1, xk1)
                        break
            if j >= self.max_iter - 1:
                print(
                    f"Newton-Raphson not converged after {j+1} steps with error {error:2.4f}"
                )
                n = len(q)
                return Solution(
                    t=t[:n],
                    q=np.array(q),
                    u=np.array(u),
                    a=np.array(u_dot),
                    Q=np.array(Q),
                    U=np.array(U),
                    la_g=np.array(la_g),
                    La_g=np.array(La_g),
                    la_N=np.array(la_N),
                    La_N=np.array(La_N),
                    P_N=np.array(P_N),
                    kappa_N=np.array(kappa_N),
                    kappa_hat_N=np.array(kappa_hat_N),
                    la_F=np.array(la_F),
                    La_F=np.array(La_F),
                    P_F=np.array(P_F),
                )
            else:
                pbar.set_description(
                    f"t: {tk1:0.2e}s < {self.t1:0.2e}s; {j}/{self.max_iter} iterations; error: {error:0.2e}"
                )

        n = len(q)
        return Solution(
            t=t[:n],
            q=np.array(q),
            u=np.array(u),
            a=np.array(u_dot),
            Q=np.array(Q),
            U=np.array(U),
            la_g=np.array(la_g),
            La_g=np.array(La_g),
            la_N=np.array(la_N),
            La_N=np.array(La_N),
            P_N=np.array(P_N),
            kappa_N=np.array(kappa_N),
            kappa_hat_N=np.array(kappa_hat_N),
            la_F=np.array(la_F),
            La_F=np.array(La_F),
            P_F=np.array(P_F),
        )
