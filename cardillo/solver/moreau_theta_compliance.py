import warnings

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import bmat, block_diag, coo_array, csc_array, eye
from scipy.sparse.linalg import splu
from tqdm import tqdm

from cardillo.math.approx_fprime import approx_fprime
from cardillo.math.fsolve import fsolve
from cardillo.math.prox import NegativeOrthant, estimate_prox_parameter
from cardillo.solver import Solution, SolverOptions, SolverSummary


class MoreauThetaCompliance:
    def __init__(
        self,
        system,
        t1,
        dt,
        # velocity_level_contact=False,
        velocity_level_contact=True,
        theta=0.5,
        options=SolverOptions(),
    ):
        self.theta = theta
        assert 0 <= theta <= 1
        self.velocity_level_contact = velocity_level_contact
        if not velocity_level_contact:
            # otherwise an arbitrary impact law is realized
            assert np.isclose(theta, 1.0)

        self.system = system

        # simplified Newton iterations
        options.reuse_lu_decomposition = True
        self.options = options

        if options.numerical_jacobian_method:
            self.J_x = lambda x, y: csc_array(
                approx_fprime(
                    x,
                    lambda x: self.R_x(x, y),
                    method=options.numerical_jacobian_method,
                    eps=options.numerical_jacobian_eps,
                )
            )
        else:
            self.J_x = self._J_x

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

        # self.nx = self.nu + self.nla_g + self.nla_gamma + self.nla_c
        # self.ny = self.nla_N + self.nla_F
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
        self.split_x = np.cumsum(
            np.array(
                [
                    self.nu,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_c,
                ],
                dtype=int,
            )
        )[:-1]
        self.split_y = np.cumsum(
            np.array(
                [
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

        #######################################################################
        # initial values
        #######################################################################
        self.xn = self.dt * np.concatenate(
            (
                self.u_dotn,
                self.la_gn,
                self.la_gamman,
                self.la_cn,
            )
        )
        self.yn = self.dt * np.concatenate(
            (
                self.la_Nn,
                self.la_Fn,
            )
        )

        # # initial mass matrix and force directions for prox-parameter estimation
        # self.M = system.M(self.tn, self.qn)
        # self.W_N = system.W_N(self.tn, self.qn)
        # self.W_F = system.W_F(self.tn, self.qn)

    def R_x(self, xn1, yn1):
        (
            dun1,
            dP_gn1,
            dP_gamman1,
            dP_cn1,
        ) = np.array_split(xn1, self.split_x)
        (
            dP_Nn1,
            dP_Fn1,
        ) = np.array_split(yn1, self.split_y)

        # integration
        tn1 = self.tn + self.dt
        un1 = self.un_theta + dun1
        qn1 = self.qn_theta + self.dt * self.theta * self.system.q_dot(
            self.tn_theta, self.qn_theta, un1
        )

        # initialize residual
        R_x = np.empty_like(xn1)

        #####################
        # equations of motion
        #####################
        R_x[: self.split_x[0]] = (
            self.M @ dun1
            - self.W_g @ dP_gn1
            - self.W_gamma @ dP_gamman1
            - self.W_c @ dP_cn1
            - self.W_N @ dP_Nn1
            - self.W_F @ dP_Fn1
        )

        #######################
        # bilateral constraints
        #######################
        R_x[self.split_x[0] : self.split_x[1]] = self.system.g(tn1, qn1)
        R_x[self.split_x[1] : self.split_x[2]] = self.system.gamma(tn1, qn1, un1)

        ############
        # compliance
        ############
        R_x[self.split_x[2] :] = self.system.c(tn1, qn1, un1, dP_cn1 / self.dt)

        return R_x

    def _J_x(self, xn1, yn1):
        (
            dun1,
            dP_gn1,
            dP_gamman1,
            dP_cn1,
        ) = np.array_split(xn1, self.split_x)
        (
            dP_Nn1,
            dP_Fn1,
        ) = np.array_split(yn1, self.split_y)

        # integration
        dt_th = self.dt * self.theta
        tn1 = self.tn + self.dt
        un1 = self.un_theta + dun1
        qn1 = self.qn_theta + self.dt * self.theta * self.system.q_dot(
            self.tn_theta, self.qn_theta, un1
        )
        q_dot_u = self.system.q_dot_u(self.tn_theta, self.qn_theta)

        #######################
        # bilateral constraints
        #######################
        g_q = self.system.g_q(tn1, qn1)
        gamma_q = self.system.gamma_q(tn1, qn1, un1)
        gamma_u = self.system.gamma_u(tn1, qn1)

        ############
        # compliance
        ############
        c_q = self.system.c_q(tn1, qn1, un1, dP_cn1 / self.dt)
        c_u = self.system.c_u(tn1, qn1, un1, dP_cn1 / self.dt)
        c_la_c = self.system.c_la_c() / self.dt

        # fmt: off
        J = bmat(
            [
                [                             self.M, -self.W_g, -self.W_gamma,        -self.W_c],
                [              g_q @ q_dot_u * dt_th,      None,          None,             None],
                [gamma_q @ q_dot_u * dt_th + gamma_u,      None,          None,             None],
                [        c_q @ q_dot_u * dt_th + c_u,      None,          None, c_la_c / self.dt],
            ],
            format="csc",
        )
        # fmt: on
        return J

        J_num = approx_fprime(xn1, lambda x: self.R_x(x, yn1))
        diff = J_num - J
        error = np.linalg.norm(diff)
        print(f"error J: {error}")
        return J_num

    def prox(self, x1, y0):
        (
            dun1,
            _,
            _,
            _,
        ) = np.array_split(x1, self.split_x)
        (
            dP_Nn1,
            dP_Fn1,
        ) = np.array_split(y0, self.split_y)

        tn, dt, qn, un = self.tn, self.dt, self.qn, self.un
        tn1 = tn + dt
        un1 = self.un_theta + dun1
        qn1 = self.qn_theta + self.dt * self.theta * self.system.q_dot(
            self.tn_theta, self.qn_theta, un1
        )

        prox_r_N = self.prox_r_N
        prox_r_F = self.prox_r_F

        y1 = np.zeros_like(y0)

        ##############################
        # fixed-point update Signorini
        ##############################
        if self.velocity_level_contact:
            # explicit gap evaluation
            g_N = self.system.g_N(self.tn_theta, self.qn_theta)
            xi_N = self.system.xi_N(tn, tn1, qn, qn1, un, un1)
            dP_Nn1 = np.where(
                g_N <= 0,
                -NegativeOrthant.prox(prox_r_N * xi_N - dP_Nn1),
                np.zeros_like(dP_Nn1),
            )
        else:
            g_N = self.system.g_N(tn1, qn1)
            dP_Nn1 = -NegativeOrthant.prox((prox_r_N / self.dt) * g_N - dP_Nn1)
        y1[: self.split_y[0]] = dP_Nn1

        #############################
        # fixed-point update friction
        #############################
        if self.velocity_level_contact:
            gamma_F = self.system.xi_F(tn, tn1, qn, qn1, un, un1)
        else:
            gamma_F = self.system.gamma_F(tn1, qn1, un1)
        for contr in self.system.get_contribution_list("gamma_F"):
            la_FDOF = contr.la_FDOF
            gamma_F_contr = gamma_F[la_FDOF]
            dP_Fn1_contr = dP_Fn1[la_FDOF]
            prox_r_F_contr = prox_r_F[la_FDOF]
            for i_N, i_F, force_recervoir in contr.friction_laws:
                if len(i_N) > 0:
                    dP_Nn1i = dP_Nn1[contr.la_NDOF[i_N]]
                else:
                    dP_Nn1i = self.dt

                y1[self.split_y[0] + la_FDOF[i_F]] = -force_recervoir.prox(
                    prox_r_F_contr[i_F] * gamma_F_contr[i_F] - dP_Fn1_contr[i_F],
                    dP_Nn1i,
                )

        return y1

    def _solve_nonlinear_system(self, x0, y, lu):
        if self.options.reuse_lu_decomposition:
            sol = fsolve(
                lambda x, y, *args: self.R_x(x, y, *args),
                x0,
                jac=lu,
                fun_args=(y,),
                options=self.options,
            )
        else:
            sol = fsolve(
                lambda x, y, *args: self.R_x(x, y, *args),
                x0,
                jac=lambda x, y, *args: self.J_x(x, y, *args),
                fun_args=(y,),
                jac_args=(y,),
                options=self.options,
            )

        return sol

    def _make_solution(self):
        return Solution(
            system=self.system,
            t=np.array(self.sol_t),
            q=np.array(self.sol_q),
            u=np.array(self.sol_u),
            u_dot=np.array(self.sol_u_dot),
            la_c=np.array(self.sol_la_c),
            P_g=np.array(self.sol_P_g),
            P_gamma=np.array(self.sol_P_gamma),
            P_N=np.array(self.sol_P_N),
            P_F=np.array(self.sol_P_F),
            solver_summary=self.solver_summary,
        )

    def _step_naive(self):
        # theta step
        dt = self.dt
        self.tn_theta = self.tn + dt * (1 - self.theta)
        self.qn_theta = self.qn + dt * (1 - self.theta) * self.system.q_dot(
            self.tn, self.qn, self.un
        )

        self.M = self.system.M(self.tn_theta, self.qn_theta, format="csc")
        # TODO: What to do with control forces?
        h_Wla_tau = self.system.h(
            self.tn_theta, self.qn_theta, self.un
        ) + self.system.W_tau(
            self.tn_theta, self.qn_theta, format="csr"
        ) @ self.system.la_tau(
            self.tn_theta, self.qn_theta, self.un
        )
        M_inv = splu(self.M)  # TODO: Use this for prox parameter estimation
        self.un_theta = self.un + dt * M_inv.solve(h_Wla_tau)

        # evaluate quantities that are kept fixed during the simplified Newton iterations
        self.W_g = self.system.W_g(self.tn_theta, self.qn_theta)
        self.W_gamma = self.system.W_gamma(self.tn_theta, self.qn_theta)
        self.W_N = self.system.W_N(self.tn_theta, self.qn_theta)
        self.W_F = self.system.W_F(self.tn_theta, self.qn_theta)
        self.W_c = self.system.W_c(self.tn_theta, self.qn_theta)
        # self.c_q = self.system.c_q(self.tn_theta, self.qn_theta, self.la_cn)
        # self.c_la_c = self.system.c_la_c(self.tn_theta, self.qn_theta, self.la_cn)

        # only compute optimized prox-parameters once per time step
        if self.nla_N + self.nla_F > 0:
            self.prox_r_N, self.prox_r_F = np.array_split(
                estimate_prox_parameter(
                    self.options.prox_scaling, bmat([[self.W_N, self.W_F]]), M_inv
                ),
                [self.nla_N],
            )

        ########################
        # fixed-point iterations
        ########################
        # store old values
        x0 = self.xn.copy()
        y0 = self.yn.copy()

        # Jacobian and lu-decompositon
        if self.options.reuse_lu_decomposition:
            J_x = self.J_x(x0, y0)
            lu = splu(J_x)
            self.solver_summary.add_lu(1)

        # solve nonlinear system
        sol = self._solve_nonlinear_system(x0, y0, lu)
        xn1 = sol.x
        self.solver_summary.add_lu(sol.njev)

        if not sol.success:
            if self.options.continue_with_unconverged:
                warnings.warn("Newton is not converged but integration is continued")
            else:
                warnings.warn(
                    f"Newton is not converged. Returning solution up to t={self.tn}."
                )
                return self._make_solution()

        # fixed-point loop
        converged = False
        if self.nla_N + self.nla_F > 0:
            for i_fixed_point in range(self.options.fixed_point_max_iter):
                # find proximal point
                yn1 = self.prox(xn1, y0)

                # error measure, see Hairer1993, Section II.4
                diff = yn1 - y0
                sc = (
                    self.options.fixed_point_atol
                    + np.maximum(np.abs(yn1), np.abs(y0))
                    * self.options.fixed_point_rtol
                )
                error = np.linalg.norm(diff / sc) / sc.size**0.5
                converged = error < 1.0

                if converged:
                    break
                else:
                    # update values
                    x0 = xn1.copy()
                    y0 = yn1.copy()

                    # solve nonlinear system
                    sol = self._solve_nonlinear_system(x0, y0, lu)
                    xn1 = sol.x
                    self.solver_summary.add_lu(sol.njev)

                    if not sol.success:
                        if self.options.continue_with_unconverged:
                            warnings.warn(
                                "Newton is not converged but integration is continued"
                            )
                        else:
                            RuntimeWarning(
                                "Newton is not converged. Returning solution up to t={self.tn}."
                            )
                            return self._make_solution()

        else:
            converged = True
            i_fixed_point = 0
            error = 0

        if not converged:
            if self.options.continue_with_unconverged:
                warnings.warn(
                    "fixed-point iteration is not converged but integration is continued"
                )
            else:
                warnings.warn(
                    f"fixed-point iteration is not converged. Returning solution up to t={self.tn}."
                )
                return self._make_solution()

        self.solver_summary.add_fixed_point(i_fixed_point, error)
        self.solver_summary.add_newton(sol.nit, sol.error)

        # update progress bar
        tn1 = self.tn + self.dt
        self.pbar.set_description(
            f"t: {tn1:0.2e}s < {self.t1:0.2e}s; |y1 - y0|_rel: {error:0.2e}; fixed-point: {i_fixed_point}/{self.options.fixed_point_max_iter}; |dx|_rel: {sol.error:0.2e}; newton: {sol.nit}/{self.options.newton_max_iter}"
        )

        # compute state
        (
            dun1,
            dP_gn1,
            dP_gamman1,
            dP_cn1,
        ) = np.array_split(xn1, self.split_x)
        (
            dP_Nn1,
            dP_Fn1,
        ) = np.array_split(y0, self.split_y)
        un1 = self.un_theta + dun1
        qn1 = self.qn_theta + dt * self.theta * self.system.q_dot(
            self.tn_theta, self.qn_theta, un1
        )

        # modify converged quantities
        qn1, un1 = self.system.step_callback(tn1, qn1, un1)

        # store solution fields
        self.sol_t.append(tn1)
        self.sol_q.append(qn1)
        self.sol_u.append(un1)
        self.sol_u_dot.append(dun1 / self.dt)
        self.sol_la_c.append(dP_cn1 / self.dt)
        self.sol_P_g.append(dP_gn1)
        self.sol_P_gamma.append(dP_gamman1)
        self.sol_P_N.append(dP_Nn1)
        self.sol_P_F.append(dP_Fn1)

        # update local variables for accepted time step
        self.xn = xn1.copy()
        self.yn = y0.copy()
        self.tn = tn1
        self.qn = qn1
        self.un = un1

    def c(self, t, q, u, la):
        """Combine all constraint forces in order to simplify the solver."""
        la_g, la_gamma, la_c = np.array_split(la, self.split_la)

        c = np.zeros(self.nla)
        c[: self.split_la[0]] = self.system.g(t, q)
        c[self.split_la[0] : self.split_la[1]] = self.system.gamma(t, q, u)
        c[self.split_la[1] :] = self.system.c(t, q, u, la_c)

        return c

    def c_q(self, t, q, u, la, format="coo"):
        la_g, la_gamma, la_c = np.array_split(la, self.split_la)

        g_q = self.system.g_q(t, q)
        gamma_q = self.system.gamma_q(t, q, u)
        c_q = self.system.c_q(t, q, u, la_c)

        return bmat([[g_q], [gamma_q], [c_q]], format=format)

    def c_u(self, t, q, u, la, format="coo"):
        la_g, la_gamma, la_c = np.array_split(la, self.split_la)

        g_u = np.zeros((self.nla_g, self.nu))
        gamma_u = self.system.gamma_u(t, q)
        c_u = self.system.c_u(t, q, u, la_c)

        return bmat([[g_u], [gamma_u], [c_u]], format=format)

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

    def _step_schur(self):
        # theta step (part 1)
        dt = self.dt
        theta = self.theta
        tn_theta = self.tn + dt * (1 - theta)
        qn_theta = self.qn + dt * (1 - theta) * self.system.q_dot(
            self.tn, self.qn, self.un
        )

        M = self.system.M(tn_theta, qn_theta, format="csc")
        M_inv = splu(M)
        self.solver_summary.add_lu(1)

        # explicit h-vector and control forces
        h_Wla_tau = self.system.h(tn_theta, qn_theta, self.un)
        h_Wla_tau += self.system.W_tau(
            tn_theta, qn_theta, format="csr"
        ) @ self.system.la_tau(tn_theta, qn_theta, self.un)

        # theta step (part 2): free velocity (without constraints and contacts)
        un_theta = self.un + dt * M_inv.solve(h_Wla_tau)

        # evaluate all other quantities that are kept fixed during the
        # simplified Newton iterations
        q_dot_u = self.system.q_dot_u(tn_theta, qn_theta)
        beta = self.system.q_dot(tn_theta, qn_theta, np.zeros_like(self.un))
        W = self.W(tn_theta, qn_theta)
        c_q = self.c_q(tn_theta, qn_theta, un_theta, self.la_cn)
        c_u = self.c_u(tn_theta, qn_theta, un_theta, self.la_cn)
        c_la = self.c_la()
        self.solver_summary.add_lu(1)

        # compute iteration matrices
        A = dt * theta * c_q @ q_dot_u + c_u
        # TODO: This seems to be very bas since W might be very sparse!
        M_inv_W = M_inv.solve(W.toarray())
        D = A @ M_inv_W + c_la / dt

        # for a moderate number of constraints D is small and dense,
        # hence we use a dense factorization
        D_fac = lu_factor(D)
        self.solver_summary.add_lu(1)
        # def D_inv(rhs):
        #     return lu_solve(D_fac, rhs)
        # class D_inv:
        #     def solve(rhs):
        #         return lu_solve(D_fac, rhs)
        D_inv = type("LU", (), {"solve": lambda self, rhs: lu_solve(D_fac, rhs)})()

        # - only compute optimized prox-parameters once per time step
        # - generalized force directions for constraint forces are only
        #   required if we have contacts
        if self.nla_N + self.nla_F > 0:
            W_N = self.system.W_N(tn_theta, qn_theta)
            W_F = self.system.W_F(tn_theta, qn_theta)
            self.prox_r_N, self.prox_r_F = np.array_split(
                estimate_prox_parameter(
                    self.options.prox_scaling, bmat([[W_N, W_F]]), M_inv
                ),
                [self.nla_N],
            )

        # TODO: Add outer fixed point loop for contacts

        ###################
        # newton iterations
        ###################
        if self.nla > 0:
            # compute initial positions velocities and percussions
            tn1 = self.tn + dt
            un1 = un_theta
            qn1 = qn_theta + dt * theta * (q_dot_u @ un1 + beta)
            Pin1 = dt * np.concatenate([self.la_gn, self.la_gamman, self.la_cn])
            # Pin1 *= 0

            # evaluate residuals
            R1 = M @ (un1 - un_theta) - W @ Pin1
            R2 = self.c(tn1, qn1, un1, Pin1 / dt)
            R = np.concatenate((R1, R2))

            # newton scaling
            scale1 = self.options.newton_atol + np.abs(R1) * self.options.newton_rtol
            scale2 = self.options.newton_atol + np.abs(R2) * self.options.newton_rtol
            scale = np.concatenate((scale1, scale2))

            # error of initial guess
            error = np.linalg.norm(R / scale) / scale.size**0.5
            converged = error < 1
            # print(f"i: {-1}; error: {error}; converged: {converged}")

            # Newton loop
            if not converged:
                for i in range(self.options.newton_max_iter):
                    # Newton updates
                    Delta_Pin1 = D_inv.solve(A @ R1 - R2)
                    Delta_Un1 = M_inv_W @ Delta_Pin1 - R1

                    # update dependent variables
                    un1 += Delta_Un1
                    Pin1 += Delta_Pin1
                    qn1 = qn_theta + dt * theta * (q_dot_u @ un1 + beta)

                    # evaluate residuals
                    R1 = M @ (un1 - un_theta) - W @ Pin1
                    R2 = self.c(tn1, qn1, un1, Pin1 / dt)
                    R = np.concatenate((R1, R2))

                    # error and convergence check
                    error = np.linalg.norm(R / scale) / scale.size**0.5
                    converged = error < 1
                    # print(f"i: {i}; error: {error}; converged: {converged}")
                    if converged:
                        break

                if not converged:
                    warnings.warn(
                        f"Newton method is not converged after {i} iterations with error {error:.2e}"
                    )

                nit = i + 1

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
        # self.sol_P_N.append(dP_Nn1)
        # self.sol_P_F.append(dP_Fn1)

        # update local variables for accepted time step
        self.tn = tn1
        self.qn = qn1
        self.un = un1
        self.la_gn = Pi_gn1 / dt
        self.la_gamman = Pi_gamman1 / dt
        self.la_cn = Pi_cn1 / dt

    def solve(self):
        self.solver_summary = SolverSummary("MoreauThetaCompliance")

        # lists storing output variables
        self.sol_t = [self.tn]
        self.sol_q = [self.qn]
        self.sol_u = [self.un]
        self.sol_u_dot = [self.u_dotn]
        self.sol_la_c = [self.la_cn]
        self.sol_P_g = [self.dt * self.la_gn]
        self.sol_P_gamma = [self.dt * self.la_gamman]
        self.sol_P_N = [self.dt * self.la_Nn]
        self.sol_P_F = [self.dt * self.la_Fn]

        self.pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in self.pbar:
            # self._step_naive()
            self._step_schur()

        self.solver_summary.print()

        # write solution
        return self._make_solution()
