import warnings
from tqdm import tqdm
import numpy as np
from scipy.sparse import bmat, block_diag, diags_array
from scipy.sparse.linalg import splu, LinearOperator, minres

from cardillo.math.prox import NegativeOrthant
from cardillo.solver import Solution, SolverOptions, SolverSummary


def fixed_point_iteration(fun, x0, atol=1e-6, rtol=1e-6, max_iter=100, verbose=False):
    x = x0.copy()
    for k in range(max_iter):
        x_new = fun(x.copy())
        scale = atol + np.maximum(np.abs(x), np.abs(x_new)) * rtol
        scale = np.array([1])
        error = np.linalg.norm((x_new.copy() - x.copy()) / scale) / len(scale) ** 0.5
        if verbose:
            print(f"iter {k:3d} | error: {error:.2e}")
            pass
        if error < 1:
            return x_new, k + 1, error
        x = x_new.copy()
    raise ValueError(
        f"Fixed-point iteration did not converge after {k + 1} iterations with error: {error}"
    )


def fixed_point_iteration_with_momentum(
    fun,
    x0,
    atol=1e-6,
    rtol=1e-6,
    max_iter=50,
    verbose=False,
):
    """
    Nesterov acceleration for solving x = fun(x).

    Parameters:
        fun        : callable, function implementing the fixed-point map:
                     x, *args = fun(x)
        x0         : np.ndarray, initial guess
        atol, rtol : float, stopping tolerances for relative residual norm
        max_iter   : int, maximum iterations
        verbose    : bool, if True prints progress

    Returns:
        x          : final iterate
        niter      : number of required iterations
        error      : final relative residual error
    """
    xk = x0.copy()
    yk = xk.copy()
    n = len(x0)

    # parameters Nesterov acceleration
    thk = 1.0
    error_old = np.inf
    converged = False
    for k in range(0, max_iter):
        # next iterate
        xk1 = fun(yk.copy())
        gk = xk1 - yk.copy()

        # error
        scale = atol + np.maximum(np.abs(yk), np.abs(xk1)) * rtol
        error = np.linalg.norm(gk / scale) / np.sqrt(n)

        if verbose:
            print(f"iter {k:3d} | error: {error:.2e}")
            pass

        rate = error / error_old

        if verbose:
            print(f"iter {k:3d} | error: {error:.2e} | rate:: {rate:.2e}")
            pass

        if error < 1:
            converged = True
            break

        error_old = error

        # Nesterov acceleration
        # original defintions, see https://hengshuaiyao.github.io/papers/nesterov83.pdf
        thk1 = 0.5 * (1 + np.sqrt(4 * thk**2 + 1))
        betak1 = (thk - 1) / thk1

        # momentum
        yk1 = xk1 + betak1 * (xk1 - xk)

        # reset strategy, see eq. (12) and (13) in
        # https://link.springer.com/article/10.1007/s10208-013-9150-3
        if (
            np.dot(yk - xk1, xk1 - xk) > 0
            or not np.isfinite(betak1)
            or betak1 < 0
            or betak1 > 1
        ):
            if verbose:
                print(
                    f"restart triggered with: np.dot(yk - xk1, xk1 - xk) = {np.dot(yk - xk1, xk1 - xk)} > 0"
                )
            yk1 = xk1.copy()
            thk1 = 1.0

        # update previous values
        xk = xk1.copy()
        yk = yk1.copy()
        thk = thk1

    if not converged:
        print(f"k: {k + 1}")
        print(f"error: {error}")
        raise RuntimeError(
            f"Nesterov acceleration is not converged after {k} iterations with error {error}"
        )

    return xk, k + 1, error


class DualStormerVerlet:
    def __init__(
        self,
        system,
        t1,
        dt,
        options=SolverOptions(),
        debug=False,
        # linear_solver="LU",
        # linear_solver="MINRES",
        linear_solver="MINRES (matrix free)",
        constant_mass_matrix=True,
        accelerated=True,
    ):
        self.debug = debug
        self.system = system
        self.accelerated = accelerated

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
        self.solver_summary = SolverSummary("DualStörmerVerlet")

        self.C = self.c_la(format="csr")

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

    def c_la(self, format="coo"):
        g_la_g = np.zeros((self.nla_g, self.nla_g))
        gamma_la_gamma = np.zeros((self.nla_gamma, self.nla_gamma))
        c_la_c = self.system.c_la_c()

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
        qm, niter, error = fixed_point_iteration(
            lambda qm: qn + 0.5 * dt * self.system.q_dot(tm, qm, un),
            qn.copy(),
            atol=self.options.fixed_point_atol,
            rtol=self.options.fixed_point_rtol,
            max_iter=self.options.fixed_point_max_iter,
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

        ############################
        # compute iteration matrices
        ############################
        if self.linear_solver in ["LU", "MINRES"]:
            # fmt: off
            A = bmat([
                [  M,              W],
                [W.T, -C * 2 / dt**2],
            ], format="csr" if self.linear_solver=="MINRES" else "csc")
            # fmt: on
        else:  # self.linear_solver == "MINRES (matrix free)"

            def mv(p):
                p1, p2 = p[:nu], p[nu:]
                return np.concatenate(
                    [
                        M @ p1 + W @ p2,
                        W.T @ p1 - C @ p2 * 2 / dt**2,
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

        # the fixed-point equation
        h0 = self.system.h(tm, qm, un)

        def fun(z0):
            # unpack state
            x0, y0 = z0[:nx], z0[nx:]
            un1, Pin1 = x0[:nu], x0[nu:]
            Pi_Nn1, Pi_Fn1 = y0[: self.nla_N], y0[self.nla_N :]

            # grid-point update
            qn1 = qm + 0.5 * dt * (q_dot_u @ un1 + beta)

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
                            prox_r_F_contr[i_F] * gamma_F_contr[i_F]
                            - Pi_Nn1_contr[i_F],
                            dP_Nn1i,
                        )

                return Pi_Nn1, Pi_Fn1

            # evaluate residuals
            R1 = (
                M @ (un1 - un) - 0.5 * dt * (h0 + self.system.h(tm, qm, un1)) - W @ Pin1
            )
            if self.nla_N + self.nla_F > 0:
                Pi_Nn1, Pi_Fn1 = prox(Pi_Nn1, Pi_Fn1)
                R1 -= W_N @ Pi_Nn1 + W_F @ Pi_Fn1
            R2 = self.c(tn1, qn1, un1, Pin1 / dt, dt=dt)
            R = np.concatenate((R1, R2))

            # Newton updates
            dx = A_inv.solve(-R)
            du, dPi = dx[:nu], dx[nu:]

            # update dependent variables (confusing signs are intended!)
            un1 += du
            Pin1 -= dPi

            return np.concatenate((un1, Pin1, Pi_Nn1, Pi_Fn1))

        z0 = np.concatenate((un1, Pin1, Pi_Nn1, Pi_Fn1))
        rtol = self.options.fixed_point_rtol
        atol = self.options.fixed_point_atol * np.concatenate(
            [
                np.ones_like(un1),
                np.ones(self.nla_g) * dt / 2,
                np.ones(self.nla_gamma),
                np.ones(self.nla_c) * dt / 2,
                np.ones_like(Pi_Nn1),
                np.ones_like(Pi_Fn1),
            ]
        )
        if self.accelerated:
            z1, niter, error = fixed_point_iteration_with_momentum(
                fun,
                z0,
                atol=atol,
                rtol=rtol,
                max_iter=self.options.fixed_point_max_iter,
            )
        else:
            z1, niter, error = fixed_point_iteration(
                fun,
                z0,
                atol=atol,
                rtol=rtol,
                max_iter=self.options.fixed_point_max_iter,
            )

        self.solver_summary.add_newton(niter, error)

        if self.debug:
            print(f"i: {niter}; error: {error}")

        # unpack state
        x1, y1 = z1[:nx], z1[nx:]
        un1, Pin1 = x1[:nu], x1[nu:]
        Pi_Nn1, Pi_Fn1 = y1[: self.nla_N], y1[self.nla_N :]

        # grid-point update
        qn1 = qm + 0.5 * dt * (q_dot_u @ un1 + beta)

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

        self.Pi_Nn = self.dt * self.la_Nn
        self.Pi_Fn = self.dt * self.la_Fn
        self.Pin = self.dt * np.concatenate([self.la_gn, self.la_gamman, self.la_cn])

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
