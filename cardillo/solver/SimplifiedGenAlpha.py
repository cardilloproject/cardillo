import numpy as np
from scipy.sparse import csr_matrix
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


class NonsmoothNewmarkFirstOrder:
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
                self.Qk,
                self.ak,
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

        a = x[2 * nq : 2 * nq + nu]
        U = x[2 * nq + nu : 2 * nq + 2 * nu]

        kappa_g = x[2 * nq + 2 * nu : 2 * nq + 2 * nu + nla_g]
        La_g = x[2 * nq + 2 * nu + nla_g : 2 * nq + 2 * nu + 2 * nla_g]
        la_g = x[2 * nq + 2 * nu + 2 * nla_g : 2 * nq + 2 * nu + 3 * nla_g]

        kappa_N = x[nR_s : nR_s + nla_N]
        La_N = x[nR_s + nla_N : nR_s + 2 * nla_N]
        la_N = x[nR_s + 2 * nla_N : nR_s + 3 * nla_N]

        La_F = x[nR_s + 3 * nla_N : nR_s + 3 * nla_N + nla_F]
        la_F = x[nR_s + 3 * nla_N + nla_F : nR_s + 3 * nla_N + 2 * nla_F]

        return q_dot, Q, a, U, kappa_g, La_g, la_g, kappa_N, La_N, la_N, La_F, la_F

    def update(self, xk1):
        dt = self.dt
        dt2 = dt * dt

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

        # update generalized coordinates and generalized velocities
        # qk1 = self.qk + dt * q_dotk1 + Qk1
        # uk1 = self.uk + dt * ak1 + Uk1
        qk1 = self.qk + (1.0 - gamma) * dt * self.q_dotk + gamma * dt * q_dotk1 + Qk1
        uk1 = self.uk + (1.0 - gamma) * dt * self.ak + gamma * dt * ak1 + Uk1
        # qk1_tmp = self.qk + (1.0 - gamma) * dt * self.q_dotk + gamma * dt * q_dotk1 + Qk1
        # qk1 = qk1_tmp + 0.5 * dt2 * self.model.q_ddot(self.tk, qk1_tmp, uk1, ak1)

        # integrated contact contributions
        # P_Nk1 = La_Nk1 + dt * la_Nk1
        # kappa_hat_Nk1 = kappa_Nk1 + dt * La_Nk1 + 0.5 * dt2 * la_Nk1
        # P_Fk1 = La_Fk1 + dt * la_Fk1
        P_Nk1 = La_Nk1 + (1.0 - gamma) * dt * self.la_Nk + gamma * dt * la_Nk1
        kappa_hat_Nk1 = (
            kappa_Nk1
            + (1.0 - gamma) * dt * self.La_Nk
            + gamma * dt * La_Nk1
            + dt2 * (0.5 - beta) * self.la_Nk
            + dt2 * beta * la_Nk1
        )

        P_Fk1 = La_Fk1 + (1.0 - gamma) * dt * self.la_Fk + gamma * dt * la_Fk1

        return qk1, uk1, P_Nk1, kappa_hat_Nk1, P_Fk1

    def residual(self, tk1, xk1, update_index_set=False):
        mu = self.model.mu
        dt = self.dt

        ###########################
        # unpack vector of unknowns
        ###########################
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

        #############################
        # compute dependent variables
        #############################
        qk1, uk1, P_Nk1, kappa_hat_Nk1, P_Fk1 = self.update(xk1)

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
        g_ddotk1 = self.model.g_ddot(tk1, qk1, uk1, ak1)

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
        # R[:nq] = q_dotk1 - self.model.q_dot(
        #     tk1, qk1, uk1
        # )  # TODO: Why is this working now?
        R[:nq] = q_dotk1 - self.model.q_dot(
            tk1, qk1, uk1 - Uk1
        )  # - 0.5 * dt * self.model.q_ddot(tk1, qk1, uk1 - Uk1, ak1)
        # This is the solution: Only use the smooth part of the velocity!
        # TODO: Why is this required if we do not use the integrated quantity
        # P_Nk1 = ... + 0.5 * dt2 * la_Nk1 in the update of P_Nk1
        # # R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, self.uk + dt * ak1)
        # R[:nq] = q_dotk1 - self.model.q_dot(tk1, qk1, self.uk + (1.0 - gamma) * dt * self.ak + gamma * dt * ak1)

        #####################
        # position correction
        #####################
        R[nq : 2 * nq] = (
            Qk1
            - g_qk1.T @ kappa_gk1
            - g_N_qk1.T @ kappa_Nk1
            - 0.5 * dt * gamma_F_qk1.T @ La_Fk1
            # - g_qk1.T @ kappa_hat_gk1
            # - g_N_qk1.T @ kappa_hat_Nk1
            # - (1.0 - gamma) * dt * self.model.gamma_F_q(self.tk, self.qk, self.uk).T @ self.La_Fk
            # - gamma * dt * gamma_F_qk1.T @ La_Fk1
            # - 0.5 * dt * gamma_F_qk1.T @ La_Fk1 # TODO: Do we need the coupling with La_Fk1?
        )

        ####################
        # equation of motion
        ####################
        R[2 * nq : 2 * nq + nu] = (
            Mk1 @ ak1 - hk1 - W_gk1 @ la_gk1 - W_Nk1 @ la_Nk1 - W_Fk1 @ la_Fk1
        )

        #################
        # impact equation
        #################
        R[2 * nq + nu : 2 * nq + 2 * nu] = (
            Mk1 @ Uk1 - W_gk1 @ La_gk1 - W_Nk1 @ La_Nk1 - W_Fk1 @ La_Fk1
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
        prox_N_arg_position = g_Nk1 - self.model.prox_r_N * kappa_hat_Nk1
        prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * P_Nk1
        prox_N_arg_acceleration = g_N_ddotk1 - self.model.prox_r_N * la_Nk1
        # TODO: Why is this a bad choice?
        # prox_N_arg_position = g_Nk1 - self.model.prox_r_N * kappa_Nk1
        # prox_N_arg_velocity = xi_Nk1 - self.model.prox_r_N * La_Nk1
        # prox_N_arg_acceleration = g_N_ddotk1 - self.model.prox_r_N * la_Nk1
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

        #######################################
        # complementarity on acceleration level
        #######################################
        Ck1 = self.Ck1
        Ck1_ind = np.where(Ck1)[0]
        _Ck1_ind = np.where(~Ck1)[0]
        R[nR_s + 2 * nla_N + Ck1_ind] = g_N_ddotk1[Ck1]
        R[nR_s + 2 * nla_N + _Ck1_ind] = la_Nk1[~Ck1]

        # R[nR_s + 2 * nla_N : nR_s + 3 * nla_N] = np.select(
        #     self.Bk1, g_N_ddotk1 - prox_Rn0(prox_N_arg_acceleration), la_Nk1
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
