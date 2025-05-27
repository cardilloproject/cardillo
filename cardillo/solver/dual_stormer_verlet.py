import warnings
from tqdm import tqdm
import numpy as np
from scipy.sparse import bmat, block_diag, diags_array
from scipy.sparse.linalg import splu, LinearOperator, minres

from cardillo.math.prox import NegativeOrthant
from cardillo.solver import Solution, SolverOptions, SolverSummary


def fixed_point_iteration(f, q0, atol=1e-6, rtol=1e-6, max_iter=100):
    q = q0.copy()
    scale = atol + np.abs(q0) * rtol
    for k in range(max_iter):
        q_new = f(q)
        error = np.linalg.norm((q_new - q) / scale) / len(scale) ** 0.5
        if error < 1:
            return q_new, k + 1, error
        q = q_new
    raise ValueError(
        f"Fixed-point iteration did not converge after {k + 1} iterations with error: {error}"
    )


class DualStormerVerlet:
    def __init__(
        self,
        system,
        t1,
        dt,
        options=SolverOptions(),
        debug=False,
        linear_solver="LU",
        # linear_solver="MINRES",
        # linear_solver="MINRES (matrix free)",
        constant_mass_matrix=True,
    ):
        self.debug = debug
        self.system = system

        assert linear_solver in ["LU", "MINRES", "MINRES (matrix free)"]
        self.linear_solver = linear_solver

        # simplified Newton iterations
        options.reuse_lu_decomposition = False
        self.options = options

        #######################################################################
        # integration time
        #######################################################################
        self.t0 = t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt

        #######################################################################
        # dimensions
        #######################################################################
        self.nq = system.nq
        self.nu = system.nu
        self.nla_g = system.nla_g
        self.nla_gamma = system.nla_gamma
        self.nla_c = system.nla_c
        self.nla = self.nla_g + self.nla_gamma + self.nla_c
        self.nla_N = system.nla_N
        self.nla_F = system.nla_F

        self.split_la = np.cumsum(
            np.array(
                [
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_c,
                ],
                dtype=int,
            )
        )[:-1]

        #######################################################################
        # initial conditions
        #######################################################################
        self.tn = system.t0
        self.qn = system.q0
        self.un = system.u0
        self.la_gn = system.la_g0
        self.la_gamman = system.la_gamma0
        self.la_cn = system.la_c0
        self.la_Nn = system.la_N0
        self.la_Fn = system.la_F0
        self.Pi_Nn = dt * system.la_N0
        self.Pi_Fn = dt * system.la_F0
        self.Pin = dt * np.concatenate([system.la_g0, system.la_gamma0, system.la_c0])

        self.solver_summary = SolverSummary("DualStörmerVerlet")

        self.C = self.c_la(format="csr", dt=dt)

        # evaluate constant mass matrix only once
        # TODO: We should check if system._System__M_contr[system.I_M] is empty here.
        self.constant_mass_matrix = constant_mass_matrix
        if constant_mass_matrix:
            warnings.warn("DualStörmerVerlet: constant_mass_matrix=True.")
            self.M = system.M(self.tn, self.qn, format="csc")

    def c(self, t, q, u, la, dt=1.0):
        """Combine all constraint forces in order to simplify the solver."""
        la_g, la_gamma, la_c = np.array_split(la, self.split_la)

        c = np.zeros(self.nla)
        c[: self.split_la[0]] = self.system.g(t, q) * 2 / dt
        c[self.split_la[0] : self.split_la[1]] = self.system.gamma(t, q, u)
        c[self.split_la[1] :] = self.system.c(t, q, u, la_c) * 2 / dt

        return c

    def c_q(self, t, q, u, la, format="coo", dt=1.0):
        la_g, la_gamma, la_c = np.array_split(la, self.split_la)

        g_q = self.system.g_q(t, q) * 2 / dt
        gamma_q = self.system.gamma_q(t, q, u)
        c_q = self.system.c_q(t, q, u, la_c) * 2 / dt

        return bmat([[g_q], [gamma_q], [c_q]], format=format)

    def c_u(self, t, q, u, la, format="coo", dt=1.0):
        la_g, la_gamma, la_c = np.array_split(la, self.split_la)

        g_u = np.zeros((self.nla_g, self.nu))
        gamma_u = self.system.gamma_u(t, q)
        c_u = self.system.c_u(t, q, u, la_c) * 2 / dt

        return bmat([[g_u], [gamma_u], [c_u]], format=format)

    def c_la(self, format="coo", dt=1.0):
        g_la_g = np.zeros((self.nla_g, self.nla_g))
        gamma_la_gamma = np.zeros((self.nla_gamma, self.nla_gamma))
        c_la_c = self.system.c_la_c() * 2 / dt

        return block_diag([g_la_g, gamma_la_gamma, c_la_c], format=format)

    def W(self, t, q, format="coo"):
        W_g = self.system.W_g(t, q)
        W_gamma = self.system.W_gamma(t, q)
        W_c = self.system.W_c(t, q)

        return bmat(
            [
                [W_g, W_gamma, W_c],
            ],
            format=format,
        )

    def _step(self):
        nu = self.nu
        nx = self.nu + self.nla

        dt = self.dt
        tn = self.tn
        qn = self.qn
        un = self.un

        ######################################################################
        # 1. implicit mid-point step solved with simple fixed-point iterations
        ######################################################################
        tm = tn + 0.5 * dt
        qm = qn + 0.5 * dt * self.system.q_dot(tn, qn, un)
        qm, niter, error = fixed_point_iteration(
            lambda qm: qn + 0.5 * dt * self.system.q_dot(tm, qm, un),
            qm,
        )
        if self.debug:
            print(f"Fixed-point:")
            print(f"i: {niter}; error: {error}")

        #########################################
        # evaluate all quantities at the midpoint
        #########################################
        if self.constant_mass_matrix:
            M = self.M
        else:
            M = self.system.M(tm, qm, format="csr")

        q_dot_u = self.system.q_dot_u(tm, qm, format="csr")
        beta = self.system.q_dot(tm, qm, np.zeros_like(self.un))
        W_N = self.system.W_N(tm, qm, format="csr")
        W_F = self.system.W_F(tm, qm, format="csr")
        W = self.W(tm, qm, format="csr")
        C = self.C
        # c_q = self.c_q(tm, qm, un, self.Pin / dt, format="csc", dt=dt)
        # c_u = self.c_u(tm, qm, un, self.Pin / dt, format="csc", dt=dt)

        ############################
        # compute iteration matrices
        ############################
        if self.linear_solver in ["LU", "MINRES"]:
            # fmt: off
            A = bmat([
                [  M,       W],
                [W.T, -C / dt],
                # [0.5 * dt * c_q @ q_dot_u + c_u, -C / dt],
            ], format="csr" if self.linear_solver=="MINRES" else "csc")
            # fmt: on
        else:  # self.linear_solver == "MINRES (matrix free)"

            def mv(p):
                p1, p2 = p[:nu], p[nu:]
                return np.concatenate(
                    [
                        M @ p1 + W @ p2,
                        W.T @ p1 - C @ p2 / dt,
                    ]
                )

            A = LinearOperator((nx, nx), matvec=mv, dtype=float)

        if self.linear_solver in ["MINRES", "MINRES (matrix free)"]:
            # sparse minimal residual (MINRES) with Jacobi preconditioner
            A_inv = type("MINRES", (), {})()

            # diagonal mass matrix preconditioner
            AA_inv = np.ones(nx)
            AA_inv[: self.nu] = 1 / M.diagonal()
            preconditioner = LinearOperator(A.shape, lambda x: AA_inv * x)

            def solve(rhs):
                x, info = minres(A, rhs, M=preconditioner)
                if info != 0:
                    print(f"minres not converged")
                return x

            A_inv.solve = solve

        else:
            # LU solver approach (for reference only)
            A_inv = type("LU", (), {})()
            LU = splu(A)
            self.solver_summary.add_lu(1)

            def solve(rhs):
                return LU.solve(rhs)

            A_inv.solve = solve

        # - only compute optimized prox-parameters once per time step
        # - generalized force directions for constraint forces are only
        #   required if we have contacts
        if self.nla_N + self.nla_F > 0:
            W_N = self.system.W_N(tm, qm, format="csr")
            W_F = self.system.W_F(tm, qm, format="csr")
            alpha = self.options.prox_scaling
            M_diag_inv = diags_array(1 / M.diagonal())
            self.prox_r_N = alpha / (W_N.T @ M_diag_inv @ W_N).diagonal()
            self.prox_r_F = alpha / (W_F.T @ M_diag_inv @ W_F).diagonal()

            # evaluate active contacts
            g_Nm = self.system.g_N(tm, qm)
            I_N = g_Nm <= 0

            # TODO: Introduce slicing for active contacts as in Moreau

        ###################
        # newton iterations
        ###################
        if self.debug:
            print(f"Newton:")

        # compute initial positions velocities and percussions
        tn1 = self.tn + dt
        un1 = un.copy()
        qn1 = qm + 0.5 * dt * (q_dot_u @ un1 + beta)
        # Note: Warmstart seems to be only important for compliance part.
        Pin1 = self.Pin
        # warm start often reduces the number of iterations
        Pi_Nn1 = self.Pi_Nn
        Pi_Fn1 = self.Pi_Fn

        def prox(Pi_Nn1, Pi_Fn1):
            # normal contact
            # TODO: Decompose evaluation of xi_N = e_N * gamma_Nn + gamma_Nn1
            xi_N = self.system.xi_N(tm, tm, qm, qm, un, un1)
            Pi_Nn1 = np.where(
                I_N,
                -NegativeOrthant.prox(self.prox_r_N * xi_N - Pi_Nn1),
                np.zeros_like(Pi_Nn1),
            )

            # friction
            # TODO: Decompose evaluation of xi_F = e_F * gamma_Fn + gamma_Fn1
            xi_F = self.system.xi_F(tm, tm, qm, qm, un, un1)
            for contr in self.system.get_contribution_list("gamma_F"):
                la_FDOF = contr.la_FDOF
                gamma_F_contr = xi_F[la_FDOF]
                Pi_Nn1_contr = Pi_Fn1[la_FDOF]
                prox_r_F_contr = self.prox_r_F[la_FDOF]
                for i_N, i_F, force_recervoir in contr.friction_laws:
                    if len(i_N) > 0:
                        dP_Nn1i = Pi_Nn1[contr.la_NDOF[i_N]]
                    else:
                        dP_Nn1i = self.dt

                    Pi_Fn1[la_FDOF[i_F]] = -force_recervoir.prox(
                        prox_r_F_contr[i_F] * gamma_F_contr[i_F] - Pi_Nn1_contr[i_F],
                        dP_Nn1i,
                    )

            return Pi_Nn1, Pi_Fn1

        # evaluate residuals
        R1 = M @ (un1 - un) - dt * self.system.h(tm, qm, 0.5 * (un + un1)) - W @ Pin1
        if self.nla_N + self.nla_F > 0:
            Pi_Nn1, Pi_Fn1 = prox(Pi_Nn1, Pi_Fn1)
            R1 -= W_N @ Pi_Nn1 + W_F @ Pi_Fn1
        R2 = self.c(tn1, qn1, un1, Pin1 / dt, dt=dt)
        R = np.concatenate((R1, R2))

        # newton scaling
        scale1 = self.options.newton_atol + np.abs(R1) * self.options.newton_rtol
        scale2 = self.options.newton_atol + np.abs(R2) * self.options.newton_rtol
        scale = np.concatenate((scale1, scale2))

        # error of initial guess
        error = np.linalg.norm(R / scale) / scale.size**0.5
        converged = error < 1
        i = -1
        if self.debug:
            print(f"i: {i}; error: {error}; converged: {converged}")

        # Newton loop
        if not converged:
            for i in range(self.options.newton_max_iter):
                # Newton updates
                dx = A_inv.solve(-R)
                du, dPi = dx[:nu], dx[nu:]

                # update dependent variables
                un1 += du
                Pin1 -= dPi
                qn1 = qm + 0.5 * dt * (q_dot_u @ un1 + beta)

                # evaluate residuals
                R1 = (
                    M @ (un1 - un)
                    - dt * self.system.h(tm, qm, 0.5 * (un + un1))
                    - W @ Pin1
                )
                if self.nla_N + self.nla_F > 0:
                    Pi_Nn1, Pi_Fn1 = prox(Pi_Nn1, Pi_Fn1)
                    R1 -= W_N @ Pi_Nn1 + W_F @ Pi_Fn1
                R2 = self.c(tn1, qn1, un1, Pin1 / dt, dt=dt)
                R = np.concatenate((R1, R2))

                # error and convergence check
                error = np.linalg.norm(R / scale) / scale.size**0.5
                converged = error < 1
                if self.debug:
                    print(f"i: {i}; error: {error}; converged: {converged}")
                if converged:
                    break

            if not converged:
                warnings.warn(
                    f"Newton method is not converged after {i} iterations with error {error:.2e}"
                )

        self.solver_summary.add_newton(i + 1, np.max(np.abs(R)))

        # modify converged quantities
        qn1, un1 = self.system.step_callback(tn1, qn1, un1)

        # scale and unpack percussion
        Pi_gn1, Pi_gamman1, Pi_cn1 = np.array_split(Pin1, self.split_la)

        # store solution fields
        self.sol_t.append(tn1)
        self.sol_q.append(qn1)
        self.sol_u.append(un1)
        self.sol_la_c.append(Pi_cn1 / self.dt)
        self.sol_P_g.append(Pi_gn1)
        self.sol_P_gamma.append(Pi_gamman1)
        self.sol_P_N.append(Pi_Nn1)
        self.sol_P_F.append(Pi_Fn1)

        # update local variables for accepted time step
        self.tn = tn1
        self.qn = qn1.copy()
        self.un = un1.copy()
        self.Pin = Pin1.copy()
        self.Pi_Nn = Pi_Nn1.copy()
        self.Pi_Fn = Pi_Fn1.copy()

    def solve(self):
        # lists storing output variables
        self.sol_t = [self.tn]
        self.sol_q = [self.qn]
        self.sol_u = [self.un]
        self.sol_la_c = [self.la_cn]
        self.sol_P_g = [self.dt * self.la_gn]
        self.sol_P_gamma = [self.dt * self.la_gamman]
        self.sol_P_N = [self.dt * self.la_Nn]
        self.sol_P_F = [self.dt * self.la_Fn]

        self.pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in self.pbar:
            self._step()

        self.solver_summary.print()

        return Solution(
            system=self.system,
            t=np.array(self.sol_t),
            q=np.array(self.sol_q),
            u=np.array(self.sol_u),
            la_c=np.array(self.sol_la_c),
            P_g=np.array(self.sol_P_g),
            P_gamma=np.array(self.sol_P_gamma),
            P_N=np.array(self.sol_P_N),
            P_F=np.array(self.sol_P_F),
            solver_summary=self.solver_summary,
        )
