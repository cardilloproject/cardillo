import warnings
from tqdm import tqdm
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import bmat, block_diag, csc_array
from scipy.sparse.linalg import splu, LinearOperator, cg, inv

from cardillo.math.approx_fprime import approx_fprime
from cardillo.math.fsolve import fsolve
from cardillo.math.prox import NegativeOrthant, estimate_prox_parameter
from cardillo.solver import Solution, SolverOptions, SolverSummary
from cardillo.math import cr, nonlinear_cr


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


class DualStörmerVerlet:
    def __init__(
        self,
        system,
        t1,
        dt,
        options=SolverOptions(),
        debug=True,
        linear_solver="CG",
        constant_mass_matrix=True,
    ):
        self.debug = debug
        self.system = system

        assert linear_solver in ["CG", "CR", "LU", "Cholesky"]
        self.linear_solver = linear_solver

        # simplified Newton iterations
        options.reuse_lu_decomposition = False
        self.options = options

        options.numerical_jacobian_method = "2-point"
        if options.numerical_jacobian_method:
            self.J_z = lambda z: csc_array(
                approx_fprime(
                    z,
                    lambda z: self.R_z(z),
                    method=options.numerical_jacobian_method,
                    eps=options.numerical_jacobian_eps,
                )
            )
        else:
            self.J_z = self._J_z

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
        self.split_z = np.cumsum(
            np.array(
                [
                    self.nq,
                    self.nla_N,
                    self.nu,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_c,
                    self.nla_N,
                    self.nla_F,
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
        self.q_dotn = system.q_dot0
        self.u_dotn = system.u_dot0
        self.la_gn = system.la_g0
        self.la_gamman = system.la_gamma0
        self.la_cn = system.la_c0
        self.la_Nn = system.la_N0
        self.la_Fn = system.la_F0
        self.Pi_Nn = dt * system.la_N0
        self.Pi_Fn = dt * system.la_F0

        #######################################################################
        # initial values
        #######################################################################
        self.zn = self.dt * np.concatenate(
            (
                self.q_dotn,
                0 * self.la_Nn,
                self.u_dotn,
                self.la_gn,
                self.la_gamman,
                self.la_cn,
                self.la_Nn,
                self.la_Fn,
            )
        )

        self.solver_summary = SolverSummary("DualStörmerVerlet")

        # invert constant mass matrix only once
        self.constant_mass_matrix = constant_mass_matrix
        if constant_mass_matrix:
            warnings.warn("DualStörmerVerlet: constant_mass_matrix=True.")
            self.M = system.M(self.tn, self.qn, format="csc")
            # TODO: We need to implement M_inv on subsystem level.
            self.M_inv = csc_array(inv(self.M).reshape((self.nu, self.nu)))
            self.solver_summary.add_lu(1)

    def R_z(self, zn1):
        (
            dqn1,
            dmu_Nn1,
            dun1,
            dP_gn1,
            dP_gamman1,
            dP_cn1,
            dP_Nn1,
            dP_Fn1,
        ) = np.array_split(zn1, self.split_z)

        # initialize residual
        R = np.empty_like(zn1)

        #############
        # integration
        #############
        theta = self.theta
        dt = self.dt
        tn = self.tn
        qn = self.qn
        un = self.un

        # - first stage
        tm = tn + (1 - theta) * dt
        qm = qn + (1 - theta) * dt * self.system.q_dot(tn, qn, un)
        # qm = qn + (1 - theta) * dqn1
        # solve for qm with fixed-point iterations
        qm, niter, error = fixed_point_iteration(
            lambda qm: qn + (1 - theta) * dt * self.system.q_dot(tm, qm, un),
            # qn, # naive initial guess
            qn
            + (1 - theta)
            * dt
            * self.system.q_dot(tn, qn, un),  # improved initial guess
        )
        # print(f"niter fixed-point: {niter}")

        # - second (final) stage
        tn1 = tn + dt
        # qn1 = qn + dqn1
        un1 = un + dun1
        qn1 = qm + theta * dt * self.system.q_dot(tm, qm, un1)
        # qn1 = qm + dqn1

        ####################
        # kinematic equation
        ####################
        # R[: self.split_z[0]] = dqn1 - dt * self.system.q_dot(tm, qm, un)
        # R[: self.split_z[0]] = dqn1 - theta * dt * self.system.q_dot(tn1, qn1, un1)
        R[: self.split_z[0]] = dqn1

        ##########################
        # unilateral stabilization
        ##########################
        if self.nla_N + self.nla_F > 0:
            g_N = self.system.g_N(tm, qm)
            # # I_N = np.where(g_N <= 0)[0]
            # # I_N = self.prox_r_N / dt * g_N - dmu_Nn1 <= 0
            # g_N_q = self.system.g_N_q(tm, qm)
            # R[: self.split_z[0]] -= g_N_q.T @ dmu_Nn1
            # R[self.split_z[0] : self.split_z[1]] = dmu_Nn1 + NegativeOrthant.prox(self.prox_r_N / dt * g_N - dmu_Nn1)
            R[self.split_z[0] : self.split_z[1]] = dmu_Nn1

        #####################
        # equations of motion
        #####################
        R[self.split_z[1] : self.split_z[2]] = (
            self.system.M(tm, qm) @ dun1
            # - 0.5 * self.dt * (self.system.h(tm, qm, un) + self.system.h(tm, qm, un1))
            # note: this variant is way better since only a single evaluation
            #       of h is required!
            - self.dt * self.system.h(tm, qm, 0.5 * (un + un1))
            - self.system.W_g(tm, qm, format="csr") @ dP_gn1
            - self.system.W_gamma(tm, qm, format="csr") @ dP_gamman1
            - self.system.W_c(tm, qm, format="csr") @ dP_cn1
            - self.system.W_N(tm, qm, format="csr") @ dP_Nn1
            - self.system.W_F(tm, qm, format="csr") @ dP_Fn1
        )

        #######################
        # bilateral constraints
        #######################
        R[self.split_z[2] : self.split_z[3]] = self.system.g(tn1, qn1)
        R[self.split_z[3] : self.split_z[4]] = self.system.gamma(tn1, qn1, un1)

        ############
        # compliance
        ############
        R[self.split_z[4] : self.split_z[5]] = self.system.c(
            tn1, qn1, un1, dP_cn1 / self.dt
        )

        ################
        # normal contact
        ################
        if self.nla_N + self.nla_F > 0:
            xi_N = self.system.xi_N(tm, tm, qm, qm, un, un1)
            # dP_Nn1 = np.where(
            #     g_N <= 0,
            #     -NegativeOrthant.prox(self.prox_r_N * xi_N - dP_Nn1),
            #     np.zeros_like(dP_Nn1),
            # )
            R[self.split_z[5] : self.split_z[6]] = np.where(
                g_N <= 0,
                # I_N,
                # dmu_Nn1 > 0,
                dP_Nn1 + NegativeOrthant.prox(self.prox_r_N * xi_N - dP_Nn1),
                dP_Nn1,
                # dP_Nn1 + dmu_Nn1 + NegativeOrthant.prox(self.prox_r_N * xi_N - dP_Nn1 - dmu_Nn1),
                # dP_Nn1 + dmu_Nn1,
            )

        ##########
        # friction
        ##########
        if self.nla_N + self.nla_F > 0:
            xi_F = self.system.xi_F(tn, tn1, qn, qn1, un, un1)
            for contr in self.system.get_contribution_list("gamma_F"):
                la_FDOF = contr.la_FDOF
                gamma_F_contr = xi_F[la_FDOF]
                dP_Fn1_contr = dP_Fn1[la_FDOF]
                prox_r_F_contr = self.prox_r_F[la_FDOF]
                for i_N, i_F, force_recervoir in contr.friction_laws:
                    if len(i_N) > 0:
                        dP_Nn1i = dP_Nn1[contr.la_NDOF[i_N]]
                    else:
                        dP_Nn1i = self.dt

                    R[self.split_z[6] + la_FDOF[i_F]] = dP_Fn1_contr[
                        i_F
                    ] + force_recervoir.prox(
                        prox_r_F_contr[i_F] * gamma_F_contr[i_F] - dP_Fn1_contr[i_F],
                        dP_Nn1i,
                    )

        self.qn1 = qn1.copy()
        self.un1 = un1.copy()

        return R

    def _make_solution(self):
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

    def _step_naive(self, theta=0.5):
        self.theta = theta
        warnings.warn("This is very experimental and should only be used for testing!")

        M = self.system.M(self.tn, self.qn, format="csc")
        W_N = self.system.W_N(self.tn, self.qn)
        W_F = self.system.W_F(self.tn, self.qn)

        # only compute optimized prox-parameters once per time step
        if self.nla_N + self.nla_F > 0:
            self.prox_r_N, self.prox_r_F = np.array_split(
                estimate_prox_parameter(
                    self.options.prox_scaling, bmat([[W_N, W_F]]), M
                ),
                [self.nla_N],
            )

        ########################
        # fixed-point iterations
        ########################
        # solve nonlinear system
        sol = fsolve(
            lambda z: self.R_z(z),
            self.zn,
            jac="2-point",
            options=self.options,
        )
        zn1 = sol.x
        self.solver_summary.add_lu(sol.njev)
        self.solver_summary.add_newton(sol.nit, sol.error)

        if not sol.success:
            if self.options.continue_with_unconverged:
                warnings.warn("Newton is not converged but integration is continued")
            else:
                warnings.warn(
                    f"Newton is not converged. Returning solution up to t={self.tn}."
                )
                return self._make_solution()

        # update progress bar
        tn1 = self.tn + self.dt
        self.pbar.set_description(
            f"t: {tn1:0.2e}s < {self.t1:0.2e}s; |dz|_rel: {sol.error:0.2e}; newton: {sol.nit}/{self.options.newton_max_iter}"
        )

        # modify converged quantities
        qn1, un1 = self.system.step_callback(tn1, self.qn1, self.un1)

        # unpack other solution fields
        (
            dqn1,
            dmu_Nn1,
            dun1,
            dP_gn1,
            dP_gamman1,
            dP_cn1,
            dP_Nn1,
            dP_Fn1,
        ) = np.array_split(zn1, self.split_z)

        # store solution fields
        self.sol_t.append(tn1)
        self.sol_q.append(qn1)
        self.sol_u.append(un1)
        self.sol_la_c.append(dP_cn1 / self.dt)
        self.sol_P_g.append(dP_gn1)
        self.sol_P_gamma.append(dP_gamman1)
        self.sol_P_N.append(dP_Nn1)
        self.sol_P_F.append(dP_Fn1)

        # update local variables for accepted time step
        self.zn = zn1.copy()
        self.tn = tn1
        self.qn = qn1.copy()
        self.un = un1.copy()

    def c(self, t, q, u, la, dt=1.0):
        """Combine all constraint forces in order to simplify the solver."""
        la_g, la_gamma, la_c = np.array_split(la, self.split_la)

        c = np.zeros(self.nla)
        c[: self.split_la[0]] = self.system.g(t, q) * 2 / dt
        c[self.split_la[0] : self.split_la[1]] = self.system.gamma(t, q, u)
        c[self.split_la[1] :] = self.system.c(t, q, u, la_c) * 2 / dt
        # c[self.split_la[1] :] = self.system.c(t, q, u, la_c)

        return c

    def c_q(self, t, q, u, la, format="coo", dt=1.0):
        la_g, la_gamma, la_c = np.array_split(la, self.split_la)

        g_q = self.system.g_q(t, q) * 2 / dt
        gamma_q = self.system.gamma_q(t, q, u)
        c_q = self.system.c_q(t, q, u, la_c) * 2 / dt
        # c_q = self.system.c_q(t, q, u, la_c)

        return bmat([[g_q], [gamma_q], [c_q]], format=format)

    def c_u(self, t, q, u, la, format="coo", dt=1.0):
        la_g, la_gamma, la_c = np.array_split(la, self.split_la)

        g_u = np.zeros((self.nla_g, self.nu))
        gamma_u = self.system.gamma_u(t, q)
        c_u = self.system.c_u(t, q, u, la_c) * 2 / dt
        # c_u = self.system.c_u(t, q, u, la_c)

        return bmat([[g_u], [gamma_u], [c_u]], format=format)

    def c_la(self, format="coo", dt=1.0):
        g_la_g = np.zeros((self.nla_g, self.nla_g))
        gamma_la_gamma = np.zeros((self.nla_gamma, self.nla_gamma))
        c_la_c = self.system.c_la_c() * 2 / dt
        # c_la_c = self.system.c_la_c()

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

    # TODO: Incorporate Uzawa iterations as described in
    # https://www.diva-portal.org/smash/get/diva2:1638655/FULLTEXT01.pdf
    def _step_schur(self):
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
            M_inv = self.M_inv
        else:
            M = self.system.M(tm, qm, format="csc")
            # TODO: We need to implement M_inv on subsystem level.
            M_inv = csc_array(inv(M).reshape((self.nu, self.nu)))
            self.solver_summary.add_lu(1)

        q_dot_u = self.system.q_dot_u(tm, qm)
        beta = self.system.q_dot(tm, qm, np.zeros_like(self.un))
        W = self.W(tm, qm)
        c_q = self.c_q(tm, qm, un, self.la_cn, format="csc")
        c_u = self.c_u(tm, qm, un, self.la_cn, format="csc")
        C = self.c_la(format="csc")

        ############################
        # compute iteration matrices
        ############################
        A = 0.5 * dt * c_q @ q_dot_u + c_u
        M_inv_W = M_inv @ W
        D = C / dt + A @ M_inv_W

        match self.linear_solver:
            case "LU":
                # sparse LU-decomposition
                D_inv = splu(D)
                self.solver_summary.add_lu(1)

            case "Cholesky":
                # dense Cholesky-decomposition
                D_fac = cho_factor(D.toarray())
                self.solver_summary.add_lu(1)
                D_inv = type(
                    "Cholesky", (), {"solve": lambda self, rhs: cho_solve(D_fac, rhs)}
                )()

            case "CR":
                # sparse conjugate gradient (CG) with Jacobi preconditioner
                D_inv = type("CR", (), {})()
                DD_inv = 1 / D.diagonal()
                preconditioner = LinearOperator(D.shape, lambda x: DD_inv * x)

                def solve(rhs):
                    x, iterations, r, converged = cr(D, rhs, M=preconditioner)
                    return x

                D_inv.solve = solve

            case _:
                # sparse conjugate gradient (CG) with Jacobi preconditioner
                D_inv = type("CG", (), {})()
                DD_inv = 1 / D.diagonal()
                preconditioner = LinearOperator(D.shape, lambda x: DD_inv * x)

                def solve(rhs):
                    x, info = cg(D, rhs, M=preconditioner)
                    if info > 0:
                        raise RuntimeError(
                            f"Iterative solver is not converged with 'info': {info}"
                        )
                    return x

                D_inv.solve = solve

        # - only compute optimized prox-parameters once per time step
        # - generalized force directions for constraint forces are only
        #   required if we have contacts
        if self.nla_N + self.nla_F > 0:
            W_N = self.system.W_N(tm, qm, format="csr")
            W_F = self.system.W_F(tm, qm, format="csr")
            M_inv_W_N = M_inv @ W_N
            M_inv_W_F = M_inv @ W_F
            alpha = self.options.prox_scaling
            self.prox_r_N = alpha / (W_N.T @ M_inv_W_N).diagonal()
            self.prox_r_F = alpha / (W_F.T @ M_inv_W_F).diagonal()

            # evaluate active contacts
            g_Nm = self.system.g_N(tm, qm)
            I_N = g_Nm <= 0

            # # TODO: Do these evaluations only once as in Moreau and introduce
            # # slicing for active contacts
            # chi_N = self.system.g_N_dot(tn12, qn12, np.zeros_like(un))[self.I_N]
            # chi_F = self.system.gamma_F(tn12, qn12, np.zeros_like(un))[self.I_F]
            # self.xi_N0 = e_N * (self.W_N.T @ un) + (1 + e_N) * chi_N
            # self.xi_F0 = e_F * (self.W_F.T @ un) + (1 + e_F) * chi_F

        ###################
        # newton iterations
        ###################
        if self.debug:
            print(f"Newton:")
        # compute initial positions velocities and percussions
        tn1 = self.tn + dt
        un1 = un.copy()
        qn1 = qm + 0.5 * dt * (q_dot_u @ un1 + beta)
        # note: warmstart seems to be not important at all
        # Pin1 = dt * np.concatenate([self.la_gn.copy(), self.la_gamman.copy(), self.la_cn.copy()])
        Pin1 = np.zeros(self.nla)
        # warm start often reduces the number of iterations
        Pi_Nn1 = self.Pi_Nn
        Pi_Fn1 = self.Pi_Fn

        def prox(Pi_Nn1, Pi_Fn1):
            # normal contact
            # # TODO: This is the desired evaluation
            # TODO: Decompose evaluation of xi_N = e_N * gamma_Nn + gamma_Nn1
            xi_N = self.system.xi_N(tm, tm, qm, qm, un, un1)
            # TODO: This leads to second-order convergence for case 1 in the
            # point mass on slope example
            # TODO: This also introduces a strange chattering in the same example!
            # xi_N = self.system.xi_N(tn, tn1, qn, qn1, un, un1)
            Pi_Nn1 = np.where(
                I_N,
                -NegativeOrthant.prox(self.prox_r_N * xi_N - Pi_Nn1),
                np.zeros_like(Pi_Nn1),
            )

            # friction
            # TODO: Decompose evaluation of xi_F = e_F * gamma_Fn + gamma_Fn1
            # xi_F = self.system.xi_F(tn, tn1, qn, qn1, un, un1)
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
        # R1 = M @ (un1 - un) - dt * self.system.h(tm, qm, 0.5 * (un + un1)) - W @ Pin1
        M_inv_R1 = (
            (un1 - un)
            - dt * M_inv @ self.system.h(tm, qm, 0.5 * (un + un1))
            - M_inv_W @ Pin1
        )
        if self.nla_N + self.nla_F > 0:
            Pi_Nn1, Pi_Fn1 = prox(Pi_Nn1, Pi_Fn1)
            # R1 -= W_N @ Pi_Nn1 + W_F @ Pi_Fn1
            M_inv_R1 -= M_inv_W_N @ Pi_Nn1 + M_inv_W_F @ Pi_Fn1
        R2 = self.c(tn1, qn1, un1, Pin1 / dt)
        # R = np.concatenate((R1, R2))
        R = np.concatenate((M_inv_R1, R2))

        # newton scaling
        # scale1 = self.options.newton_atol + np.abs(R1) * self.options.newton_rtol
        scale1 = self.options.newton_atol + np.abs(M_inv_R1) * self.options.newton_rtol
        scale2 = self.options.newton_atol + np.abs(R2) * self.options.newton_rtol
        scale = np.concatenate((scale1, scale2))

        # error of initial guess
        error = np.linalg.norm(R / scale) / scale.size**0.5
        converged = error < 1
        if self.debug:
            print(f"i: {-1}; error: {error}; converged: {converged}")

        # Newton loop
        i = 0
        if not converged:
            for i in range(self.options.newton_max_iter):
                # Newton updates
                # M_inv_R1 = M_inv.solve(R1)
                Delta_Pin1 = D_inv.solve(A @ M_inv_R1 - R2)
                Delta_Un1 = M_inv_W @ Delta_Pin1 - M_inv_R1

                # update dependent variables
                un1 += Delta_Un1
                Pin1 += Delta_Pin1
                qn1 = qm + 0.5 * dt * (q_dot_u @ un1 + beta)

                # evaluate residuals
                # R1 = (
                #     M @ (un1 - un)
                #     - dt * self.system.h(tm, qm, 0.5 * (un + un1))
                #     - W @ Pin1
                # )
                M_inv_R1 = (
                    (un1 - un)
                    - dt * M_inv @ self.system.h(tm, qm, 0.5 * (un + un1))
                    - M_inv_W @ Pin1
                )
                if self.nla_N + self.nla_F > 0:
                    Pi_Nn1, Pi_Fn1 = prox(Pi_Nn1, Pi_Fn1)
                    # R1 -= W_N @ Pi_Nn1 + W_F @ Pi_Fn1
                    M_inv_R1 -= M_inv_W_N @ Pi_Nn1 + M_inv_W_F @ Pi_Fn1
                R2 = self.c(tn1, qn1, un1, Pin1 / dt)
                # R = np.concatenate((R1, R2))
                R = np.concatenate((M_inv_R1, R2))

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

        # unpack percussion
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
        self.la_gn = Pi_gn1.copy() / dt
        self.la_gamman = Pi_gamman1.copy() / dt
        self.la_cn = Pi_cn1.copy() / dt
        self.Pi_Nn = Pi_Nn1.copy()
        self.Pi_Fn = Pi_Fn1.copy()

    def _step_schur_projected_cg(self):
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
            M_inv = self.M_inv
        else:
            M = self.system.M(tm, qm, format="csc")
            # TODO: We need to implement M_inv on subsystem level.
            M_inv = csc_array(inv(M).reshape((self.nu, self.nu)))
            self.solver_summary.add_lu(1)

        q_dot_u = self.system.q_dot_u(tm, qm)
        beta = self.system.q_dot(tm, qm, np.zeros_like(self.un))
        W = self.W(tm, qm)
        c_q = self.c_q(tm, qm, un, self.la_cn, format="csc")
        c_u = self.c_u(tm, qm, un, self.la_cn, format="csc")
        C = self.c_la(format="csc")

        ############################
        # compute iteration matrices
        ############################
        A = 0.5 * dt * c_q @ q_dot_u + c_u
        M_inv_W = M_inv @ W
        D = C / dt + A @ M_inv_W

        # match self.linear_solver:

        #     case _:
        #         # sparse conjugate gradient (CG) with Jacobi preconditioner
        #         D_inv = type("CG", (), {})()
        #         DD_inv = 1 / D.diagonal()
        #         preconditioner = LinearOperator(D.shape, lambda x: DD_inv * x)

        #         def solve(rhs):
        #             x, info = cg(D, rhs, M=preconditioner)
        #             if info > 0:
        #                 raise RuntimeError(
        #                     f"Iterative solver is not converged with 'info': {info}"
        #                 )
        #             return x

        #         D_inv.solve = solve

        # - only compute optimized prox-parameters once per time step
        # - generalized force directions for constraint forces are only
        #   required if we have contacts
        if self.nla_N + self.nla_F > 0:
            W_N = self.system.W_N(tm, qm, format="csr")
            W_F = self.system.W_F(tm, qm, format="csr")
            M_inv_W_N = M_inv @ W_N
            M_inv_W_F = M_inv @ W_F
            alpha = self.options.prox_scaling
            self.prox_r_N = alpha / (W_N.T @ M_inv_W_N).diagonal()
            self.prox_r_F = alpha / (W_F.T @ M_inv_W_F).diagonal()

            # evaluate active contacts
            g_Nm = self.system.g_N(tm, qm)
            I_N = g_Nm <= 0

            # # TODO: Do these evaluations only once as in Moreau and introduce
            # # slicing for active contacts
            # chi_N = self.system.g_N_dot(tn12, qn12, np.zeros_like(un))[self.I_N]
            # chi_F = self.system.gamma_F(tn12, qn12, np.zeros_like(un))[self.I_F]
            # self.xi_N0 = e_N * (self.W_N.T @ un) + (1 + e_N) * chi_N
            # self.xi_F0 = e_F * (self.W_F.T @ un) + (1 + e_F) * chi_F

        ###################
        # newton iterations
        ###################
        if self.debug:
            print(f"Newton:")
        # compute initial positions velocities and percussions
        tn1 = self.tn + dt
        un1 = un.copy()
        qn1 = qm + 0.5 * dt * (q_dot_u @ un1 + beta)
        # note: warmstart seems to be not important at all
        # Pin1 = dt * np.concatenate([self.la_gn.copy(), self.la_gamman.copy(), self.la_cn.copy()])
        Pin1 = np.zeros(self.nla)
        # warm start often reduces the number of iterations
        Pi_Nn1 = self.Pi_Nn
        Pi_Fn1 = self.Pi_Fn

        def prox(Pi_Nn1, Pi_Fn1):
            # normal contact
            # # TODO: This is the desired evaluation
            # TODO: Decompose evaluation of xi_N = e_N * gamma_Nn + gamma_Nn1
            xi_N = self.system.xi_N(tm, tm, qm, qm, un, un1)
            # TODO: This leads to second-order convergence for case 1 in the
            # point mass on slope example
            # TODO: This also introduces a strange chattering in the same example!
            # xi_N = self.system.xi_N(tn, tn1, qn, qn1, un, un1)
            Pi_Nn1 = np.where(
                I_N,
                -NegativeOrthant.prox(self.prox_r_N * xi_N - Pi_Nn1),
                np.zeros_like(Pi_Nn1),
            )

            # friction
            # TODO: Decompose evaluation of xi_F = e_F * gamma_Fn + gamma_Fn1
            # xi_F = self.system.xi_F(tn, tn1, qn, qn1, un, un1)
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
        # R1 = M @ (un1 - un) - dt * self.system.h(tm, qm, 0.5 * (un + un1)) - W @ Pin1
        M_inv_R1 = (
            (un1 - un)
            - dt * M_inv @ self.system.h(tm, qm, 0.5 * (un + un1))
            - M_inv_W @ Pin1
        )
        if self.nla_N + self.nla_F > 0:
            Pi_Nn1, Pi_Fn1 = prox(Pi_Nn1, Pi_Fn1)
            # R1 -= W_N @ Pi_Nn1 + W_F @ Pi_Fn1
            M_inv_R1 -= M_inv_W_N @ Pi_Nn1 + M_inv_W_F @ Pi_Fn1
        R2 = self.c(tn1, qn1, un1, Pin1 / dt)
        # R = np.concatenate((R1, R2))
        R = np.concatenate((M_inv_R1, R2))

        # newton scaling
        # scale1 = self.options.newton_atol + np.abs(R1) * self.options.newton_rtol
        scale1 = self.options.newton_atol + np.abs(M_inv_R1) * self.options.newton_rtol
        scale2 = self.options.newton_atol + np.abs(R2) * self.options.newton_rtol
        scale = np.concatenate((scale1, scale2))

        # error of initial guess
        error = np.linalg.norm(R / scale) / scale.size**0.5
        converged = error < 1
        if self.debug:
            print(f"i: {-1}; error: {error}; converged: {converged}")

        # Newton loop
        i = 0
        if not converged:
            for i in range(self.options.newton_max_iter):
                # conjugate gradient step for
                # D Pin1 = A @ (M_inv_R1 - M_inv_W_N @ Pi_Nn1 - M_inv_W_F @ Pi_Fn1) - R2 =: b
                raise RuntimeError("Move on here!")

                # Newton updates
                Delta_Pin1 = D_inv.solve(A @ M_inv_R1 - R2)
                Delta_Un1 = M_inv_W @ Delta_Pin1 - M_inv_R1

                # update dependent variables
                un1 += Delta_Un1
                Pin1 += Delta_Pin1
                qn1 = qm + 0.5 * dt * (q_dot_u @ un1 + beta)

                # evaluate residuals
                # R1 = (
                #     M @ (un1 - un)
                #     - dt * self.system.h(tm, qm, 0.5 * (un + un1))
                #     - W @ Pin1
                # )
                M_inv_R1 = (
                    (un1 - un)
                    - dt * M_inv @ self.system.h(tm, qm, 0.5 * (un + un1))
                    - M_inv_W @ Pin1
                )
                if self.nla_N + self.nla_F > 0:
                    Pi_Nn1, Pi_Fn1 = prox(Pi_Nn1, Pi_Fn1)
                    # R1 -= W_N @ Pi_Nn1 + W_F @ Pi_Fn1
                    M_inv_R1 -= M_inv_W_N @ Pi_Nn1 + M_inv_W_F @ Pi_Fn1
                R2 = self.c(tn1, qn1, un1, Pin1 / dt)
                # R = np.concatenate((R1, R2))
                R = np.concatenate((M_inv_R1, R2))

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

        # unpack percussion
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
        self.la_gn = Pi_gn1.copy() / dt
        self.la_gamman = Pi_gamman1.copy() / dt
        self.la_cn = Pi_cn1.copy() / dt
        self.Pi_Nn = Pi_Nn1.copy()
        self.Pi_Fn = Pi_Fn1.copy()

    def _step_cr(self):
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
        M = self.system.M(tm, qm, format="csc")
        q_dot_u = self.system.q_dot_u(tm, qm)
        beta = self.system.q_dot(tm, qm, np.zeros_like(self.un))
        W_N = self.system.W_N(tm, qm, format="csr")
        W_F = self.system.W_F(tm, qm, format="csr")
        W = self.W(tm, qm)
        c_q = self.c_q(tm, qm, un, self.la_cn, format="csc", dt=dt)
        c_u = self.c_u(tm, qm, un, self.la_cn, format="csc", dt=dt)
        C = self.c_la(format="csc", dt=dt)
        # c_q = self.c_q(tm, qm, un, self.la_cn, format="csc")
        # c_u = self.c_u(tm, qm, un, self.la_cn, format="csc")
        # C = self.c_la(format="csc")
        M_inv = csc_array(inv(M).reshape((self.nu, self.nu)))

        ############################
        # compute iteration matrices
        ############################
        # fmt: off
        A = bmat([
            # [M,   -W],
            [M,   W],
            # [W.T, C / dt],
            # [0.5 * dt * c_q @ q_dot_u + c_u, C / dt],
            # [0.5 * dt * c_q @ q_dot_u + c_u, -C / dt],
            [W.T, -C / dt],
        ], format="csr")
        # fmt: on

        # TODO: Why is W.T != 0.5 * dt * c_q @ q_dot_u? This leads to a
        # non-symmetric iteration matrix.

        # np.set_printoptions(3, suppress=True, linewidth=1000)
        # print(f"A:\n{A.toarray()}")
        # print(f"W.T:\n{W.T.toarray()}")
        # print(f"0.5 * dt * (c_q @ q_dot_u):\n{0.5 * dt * (c_q @ q_dot_u).toarray()}")
        # pass

        # # def mv(p):
        # #     p1, p2 = p[:nu], p[nu:]
        # #     return np.concatenate([
        # #         M @ p1 + W @ p2,
        # #         # (W.T @ p1).reshape(len(p2)),
        # #         W.T @ p1 + C @ p2,
        # #     ])

        # # A = LinearOperator((nx, nx), matvec=mv, dtype=float)

        # # sparse conjugate residual (CR) with Jacobi preconditioner
        # A_inv = type("CR", (), {})()

        # # diagonal mass matrix preconditioner
        # # # AA_inv = 1 / np.maximum(1, A.diagonal())
        # # AA_inv = np.ones(nx)
        # # # TODO: Can M.diagonal() be used to estimate the proximal point
        # # # parameters too? This neglects the coupling in M but might be
        # # # unnecessary.
        # # AA_inv[:self.nu] = 1 / M.diagonal()
        # # preconditioner = LinearOperator(A.shape, lambda x: AA_inv * x)

        # # Mass matrix preconditioner
        # def mv(p):
        #     p1, p2 = p[:nu], p[nu:]
        #     return np.concatenate([
        #         M_inv @ p1,
        #         p2,
        #     ])
        # # preconditioner = LinearOperator(A.shape, mv)
        # preconditioner = None

        # def solve(rhs):
        #     x, iterations, r, converged = cr(A, rhs, M=preconditioner)
        #     # x, iterations, r, converged = cr(A.T @ A, A.T @ rhs, M=preconditioner)
        #     # TODO: Add iterations to the solver_summary
        #     if not converged:
        #         print(f"cr not converged")
        #     return x

        # A_inv.solve = solve

        A_inv = type("LU", (), {})()
        LU = splu(A)

        def solve(rhs):
            return LU.solve(rhs)

        A_inv.solve = solve

        # - only compute optimized prox-parameters once per time step
        # - generalized force directions for constraint forces are only
        #   required if we have contacts
        if self.nla_N + self.nla_F > 0:
            self.solver_summary.add_lu(1)
            W_N = self.system.W_N(tm, qm, format="csr")
            W_F = self.system.W_F(tm, qm, format="csr")
            M_inv_W_N = M_inv @ W_N
            M_inv_W_F = M_inv @ W_F
            alpha = self.options.prox_scaling
            self.prox_r_N = alpha / (W_N.T @ M_inv_W_N).diagonal()
            self.prox_r_F = alpha / (W_F.T @ M_inv_W_F).diagonal()

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
        # note: warmstart seems to be not important at all
        # Pin1 = dt * np.concatenate([self.la_gn.copy(), self.la_gamman.copy(), self.la_cn.copy()])
        Pin1 = np.zeros(self.nla)
        # warm start often reduces the number of iterations
        Pi_Nn1 = self.Pi_Nn
        Pi_Fn1 = self.Pi_Fn

        def prox(Pi_Nn1, Pi_Fn1):
            # normal contact
            # # TODO: This is the desired evaluation
            # TODO: Decompose evaluation of xi_N = e_N * gamma_Nn + gamma_Nn1
            xi_N = self.system.xi_N(tm, tm, qm, qm, un, un1)
            # TODO: This leads to second-order convergence for case 1 in the
            # point mass on slope example
            # TODO: This also introduces a strange chattering in the same example!
            # xi_N = self.system.xi_N(tn, tn1, qn, qn1, un, un1)
            Pi_Nn1 = np.where(
                I_N,
                -NegativeOrthant.prox(self.prox_r_N * xi_N - Pi_Nn1),
                np.zeros_like(Pi_Nn1),
            )

            # friction
            # TODO: Decompose evaluation of xi_F = e_F * gamma_Fn + gamma_Fn1
            # xi_F = self.system.xi_F(tn, tn1, qn, qn1, un, un1)
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
        R1 = M @ (un1 - un) - dt * self.system.h(tm, qm, 0.5 * (un + un1)) + W @ Pin1
        if self.nla_N + self.nla_F > 0:
            Pi_Nn1, Pi_Fn1 = prox(Pi_Nn1, Pi_Fn1)
            R1 -= W_N @ Pi_Nn1 + W_F @ Pi_Fn1
        # R2 = self.c(tn1, qn1, un1, Pin1 / dt, dt=dt)
        R2 = self.c(tn1, qn1, un1, -Pin1 / dt, dt=dt)
        # R2 = self.c(tn1, qn1, un1, Pin1 / dt)
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
                Pin1 += dPi
                qn1 = qm + 0.5 * dt * (q_dot_u @ un1 + beta)

                # evaluate residuals
                R1 = (
                    M @ (un1 - un)
                    - dt * self.system.h(tm, qm, 0.5 * (un + un1))
                    + W @ Pin1
                )
                if self.nla_N + self.nla_F > 0:
                    Pi_Nn1, Pi_Fn1 = prox(Pi_Nn1, Pi_Fn1)
                    R1 -= W_N @ Pi_Nn1 + W_F @ Pi_Fn1
                # R2 = self.c(tn1, qn1, un1, Pin1 / dt, dt=dt)
                R2 = self.c(tn1, qn1, un1, -Pin1 / dt, dt=dt)
                # R2 = self.c(tn1, qn1, un1, Pin1 / dt)
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
        Pin1 *= -1
        Pi_gn1, Pi_gamman1, Pi_cn1 = np.array_split(Pin1, self.split_la)

        # def fun(x):
        #     du, Pin1 = x[: self.nu], x[self.nu :]

        #     tn1 = tn + dt
        #     un1 = un + du
        #     qn1 = qm + 0.5 * dt * self.system.q_dot(tm, qm, un1)

        #     return np.concatenate(
        #         [
        #             # M @ du - self.system.h(tm, qm, 0.5 * (un + un1)) - W @ Pin1,
        #             # self.c(tn1, qn1, un1, Pin1 / dt),
        #             # du - M_inv @ (dt * self.system.h(tm, qm, 0.5 * (un + un1)) - W @ Pin1),
        #             M @ du - (dt * self.system.h(tm, qm, 0.5 * (un + un1)) - W @ Pin1),
        #             self.c(tn1, qn1, un1, Pin1 / dt, dt),
        #         ]
        #     )

        # def jac(x):
        #     return approx_fprime(x, fun)

        # def gen_A(x, eps=1e-1):
        #     def mv(p):
        #         return (fun(x + eps * p) - fun(x)) / eps
        #         # return (fun(x) - fun(x + eps * p)) / eps
        #         # return approx_fprime(x, fun) @ p
        #         # return (fun(x + eps * p) - fun(x - eps * p)) / (2 * eps)

        #     # A = LinearOperator((nx, nx), matvec=mv, dtype=x.dtype)
        #     # A._matvec = mv
        #     # return A
        #     return LinearOperator((nx, nx), matvec=mv, dtype=x.dtype)

        # ##################
        # # Newton-CR method
        # ##################
        # if self.debug:
        #     print(f"Newton:")
        # nx = self.nu + self.nla
        # x = np.zeros(nx)
        # f = fun(x)
        # scale = self.options.newton_atol + np.abs(f) * self.options.newton_rtol

        # # TODO:
        # # - add contacts
        # # - try to incorporate prox iteration into cr step
        # for i in range(self.options.newton_max_iter):
        #     # # TODO: Why this is correct but cr fails with linear operator?
        #     # J = jac(x)
        #     # A = gen_A(x)
        #     # p = np.random.rand(len(x))
        #     # Jp = J @ p
        #     # Ap = A.matvec(p)
        #     # diff = Jp - Ap
        #     # error_matvec = np.linalg.norm(diff)
        #     # print(f"error_matvec: {error_matvec}")

        #     # A = J

        #     # def mv(p, fun=fun, eps=1e-6):
        #     #     return (fun(x + eps * p) - fun(x)) / eps
        #     # A = LinearOperator((nx, nx), matvec=mv, dtype=x.dtype)

        #     # # TODO: Add diagonal preconditioner with mass matrix
        #     # DD_inv = 1 / D.diagonal()
        #     # preconditioner = LinearOperator(D.shape, lambda x: DD_inv * x)

        #     A = jac(x)
        #     # A = gen_A(x)
        #     # maxiter = len(x) * 1000
        #     maxiter = None
        #     dx, iterations, r, converged = cr(A, f, rtol=1e-5, atol=0, maxiter=maxiter)
        #     if not converged:
        #         print(f"cr not converged")
        #     x -= dx
        #     f = fun(x)
        #     error = np.linalg.norm(f / scale) / scale.size**0.5
        #     converged = error < 1
        #     if self.debug:
        #         print(f"i: {i}; error: {error}; converged: {converged}")
        #     if converged:
        #         break

        # self.solver_summary.add_newton(i + 1, error)

        # # update dependent variables
        # du, Pin1 = x[: self.nu], x[self.nu :]
        # # Pin1 *= -dt
        # Pin1 *= -1
        # tn1 = tn + dt
        # un1 = un + du
        # qn1 = qm + 0.5 * dt * self.system.q_dot(tm, qm, un1)

        # # modify converged quantities
        # tn1 = tn + dt
        # qn1, un1 = self.system.step_callback(tn1, qn1, un1)

        # # unpack percussion
        # Pi_gn1, Pi_gamman1, Pi_cn1 = np.array_split(Pin1, self.split_la)

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
        self.la_gn = Pi_gn1.copy() / dt
        self.la_gamman = Pi_gamman1.copy() / dt
        self.la_cn = Pi_cn1.copy() / dt
        self.Pi_Nn = Pi_Nn1.copy()
        self.Pi_Fn = Pi_Fn1.copy()

    def _step_nonlinear_cr(self):
        raise RuntimeError("This is not working!")
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
        M = self.system.M(tm, qm, format="csc")
        W_N = self.system.W_N(tm, qm, format="csr")
        W_F = self.system.W_F(tm, qm, format="csr")
        W = self.W(tm, qm)

        # - only compute optimized prox-parameters once per time step
        # - generalized force directions for constraint forces are only
        #   required if we have contacts
        if self.nla_N + self.nla_F > 0:
            M_inv = csc_array(inv(M).reshape((self.nu, self.nu)))
            self.solver_summary.add_lu(1)
            W_N = self.system.W_N(tm, qm, format="csr")
            W_F = self.system.W_F(tm, qm, format="csr")
            M_inv_W_N = M_inv @ W_N
            M_inv_W_F = M_inv @ W_F
            alpha = self.options.prox_scaling
            self.prox_r_N = alpha / (W_N.T @ M_inv_W_N).diagonal()
            self.prox_r_F = alpha / (W_F.T @ M_inv_W_F).diagonal()

            # evaluate active contacts
            g_Nm = self.system.g_N(tm, qm)
            I_N = g_Nm <= 0

            # # TODO: Do these evaluations only once as in Moreau and introduce
            # # slicing for active contacts
            # chi_N = self.system.g_N_dot(tn12, qn12, np.zeros_like(un))[self.I_N]
            # chi_F = self.system.gamma_F(tn12, qn12, np.zeros_like(un))[self.I_F]
            # self.xi_N0 = e_N * (self.W_N.T @ un) + (1 + e_N) * chi_N
            # self.xi_F0 = e_F * (self.W_F.T @ un) + (1 + e_F) * chi_F

        def fun(x):
            du, Pin1 = x[: self.nu], x[self.nu :]

            tn1 = tn + dt
            un1 = un + du
            qn1 = qm + 0.5 * dt * self.system.q_dot(tm, qm, un1)

            return np.concatenate(
                [
                    M @ du - dt * self.system.h(tm, qm, 0.5 * (un + un1)) - W @ Pin1,
                    self.c(tn1, qn1, un1, Pin1 / dt),
                ]
            )

        def jac(x):
            return approx_fprime(x, fun)

        ##################
        # Newton-CR method
        ##################
        # x = np.concatenate([un, np.zeros(self.nla)])
        x0 = np.concatenate([np.zeros(self.nu), np.zeros(self.nla)])

        x, iterations, error = nonlinear_cr(fun, x0)
        converged = True

        self.solver_summary.add_newton(iterations, error)

        # update dependent variables
        du, Pin1 = x[: self.nu], x[self.nu :]
        tn1 = tn + dt
        un1 = un + du
        qn1 = qm + 0.5 * dt * self.system.q_dot(tm, qm, un1)

        # modify converged quantities
        qn1, un1 = self.system.step_callback(tn1, qn1, un1)

        # unpack percussion
        Pi_gn1, Pi_gamman1, Pi_cn1 = np.array_split(Pin1, self.split_la)

        # store solution fields
        self.sol_t.append(tn1)
        self.sol_q.append(qn1)
        self.sol_u.append(un1)
        self.sol_la_c.append(Pi_cn1 / self.dt)
        self.sol_P_g.append(Pi_gn1)
        self.sol_P_gamma.append(Pi_gamman1)
        # self.sol_P_N.append(Pi_Nn1)
        # self.sol_P_F.append(Pi_Fn1)

        # update local variables for accepted time step
        self.tn = tn1
        self.qn = qn1.copy()
        self.un = un1.copy()
        self.la_gn = Pi_gn1.copy() / dt
        self.la_gamman = Pi_gamman1.copy() / dt
        self.la_cn = Pi_cn1.copy() / dt
        # self.Pi_Nn = Pi_Nn1.copy()
        # self.Pi_Fn = Pi_Fn1.copy()

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
            # self._step_naive()
            # self._step_schur()
            self._step_cr()
            # self._step_schur_projected_cg()
            # self._step_nonlinear_cr()

        self.solver_summary.print()

        # write solution
        return self._make_solution()
