import numpy as np
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse import bmat, eye, diags, csc_array
from tqdm import tqdm

from cardillo.math.prox import prox_R0_nm, prox_R0_np, prox_sphere
from cardillo.math import fsolve, approx_fprime
from cardillo.solver import Solution, SolverOptions


class Rattle:
    def __init__(
        self,
        system,
        t1,
        dt,
        options=SolverOptions(),
    ):
        """
        Nonsmooth extension of RATTLE.

        A nice interpretation of the left and right limes are found in Hante2019.

        References:
        -----------
        Hante2019: https://doi.org/10.1016/j.cam.2019.112492
        """
        self.system = system
        self.options = options

        if options.numerical_jacobian_method:
            self.J_x1 = lambda x, y: csc_array(
                approx_fprime(
                    x,
                    lambda x: self.R_x1(x, y),
                    method=options.numerical_jacobian_method,
                    eps=options.numerical_jacobian_eps,
                )
            )
        else:
            self.J_x1 = self._J_x1

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
        self.nla_c = system.nla_c
        self.nla_g = system.nla_g
        self.nla_gamma = system.nla_gamma
        self.nla_N = system.nla_N
        self.nla_F = system.nla_F
        self.nla_S = self.system.nla_S

        self.nx1 = (
            self.nq + self.nu + self.nla_g + self.nla_gamma + self.nla_c + self.nla_S
        )
        self.nx2 = self.nu + self.nla_g + self.nla_gamma + self.nla_c

        self.ny = self.nla_N + self.nla_F

        self.split_x1 = np.cumsum(
            np.array(
                [
                    self.nq,
                    self.nu,
                    self.nla_c,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_S,
                ],
                dtype=int,
            )
        )[:-1]

        self.split_x2 = np.cumsum(
            np.array(
                [
                    self.nu,
                    self.nla_c,
                    self.nla_g,
                    self.nla_gamma,
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
        self.la_cn = system.la_c0
        self.P_gn = dt * system.la_g0
        self.P_gamman = dt * system.la_gamma0
        self.P_Nn = dt * system.la_N0
        self.P_Fn = dt * system.la_F0
        self.mu_Sn = np.zeros(self.nla_S)

        #######################################################################
        # initial values
        #######################################################################
        self.x1n = np.concatenate(
            (
                self.qn,
                self.un,
                self.la_cn,
                self.P_gn,
                self.P_gamman,
                self.mu_Sn,
            )
        )
        self.x2n = np.concatenate(
            (
                self.un,
                self.la_cn,
                self.P_gn,
                self.P_gamman,
            )
        )
        self.y1n = np.concatenate(
            (
                self.P_Nn,
                self.P_Fn,
            )
        )
        self.y2n = np.zeros_like(self.y1n)

        ###################################################
        # compute constant quantities for current time step
        ###################################################
        self.Mn1 = system.M(self.tn, self.qn, format="csr")
        self.W_gn1 = system.W_g(self.tn, self.qn, format="csr")
        self.W_gamman1 = system.W_gamma(self.tn, self.qn, format="csr")
        self.W_Nn1 = system.W_N(self.tn, self.qn, format="csr")
        self.W_Fn1 = system.W_F(self.tn, self.qn, format="csr")
        self.W_cn1 = system.W_c(self.tn, self.qn, format="csr")

    def R_x1(self, xn1, yn1):
        tn = self.tn
        dt = self.dt
        tn1 = tn + dt
        qn = self.x1n[: self.nq]
        un = self.x2n[: self.nu]

        (qn1, un12, la_c1, P_g1, P_gamma1, mu_S1) = np.array_split(xn1, self.split_x1)

        P_N1, P_F1 = np.array_split(yn1, self.split_y)

        R = np.zeros_like(xn1)

        ####################
        # kinematic equation
        ####################
        g_S_q = self.system.g_S_q(tn1, qn1, format="csc")
        R[: self.split_y[0]] = (
            qn1
            - qn
            - 0.5
            * dt
            * (self.system.q_dot(tn, qn, un12) + self.system.q_dot(tn1, qn1, un12))
            - g_S_q.T @ mu_S1
        )

        ########################
        # euations of motion (1)
        ########################
        R[self.split_y[0] : self.split_y[1]] = self.system.M(tn, qn, format="csr") @ (
            un12 - un
        ) - 0.5 * dt * (
            self.system.h(tn, qn, un12)
            + dt * self.system.W_c(tn, qn) @ la_c1
            + self.system.W_g(tn, qn) @ P_g1
            + self.system.W_gamma(tn, qn) @ P_gamma1
            + self.system.W_N(tn, qn) @ P_N1
            + self.system.W_F(tn, qn) @ P_F1
        )

        # ########################
        # # euations of motion (2)
        # ########################
        # R[self.split_y[1] : self.split_y[2]] = self.system.M(
        #     tn1, qn1, scipy_matrix=csr_matrix
        # ) @ (un1 - un12) - 0.5 * dt * (
        #     self.system.h(tn1, qn1, un12)
        #     + self.system.W_g(tn1, qn1) @ R_g2
        #     + self.system.W_gamma(tn1, qn1) @ R_gamma2
        #     + self.system.W_N(tn1, qn1) @ R_N2
        #     + self.system.W_F(tn1, qn1) @ R_F2
        # )

        ############
        # compliance
        ############
        R[self.split_y[1] : self.split_y[2]] = self.system.c(tn1, qn1, un12, la_c1)

        #######################
        # bilateral constraints
        #######################
        R[self.split_y[2] : self.split_y[3]] = self.system.g(tn1, qn1)
        R[self.split_y[3] : self.split_y[4]] = self.system.gamma(tn1, qn1, un12)

        ##########################
        # quaternion stabilization
        ##########################
        R[self.split_x[4] :] = self.system.g_S(tn1, qn1)

        return R

    def _J_x(self, xn1, yn1):
        raise NotImplementedError

    def prox1(self, xn1, yn1):
        tn = self.tn
        dt = self.dt
        tn1 = tn + dt
        (
            qn1,
            un12,
            _,
            _,
            _,
            _,
        ) = np.array_split(xn1, self.split_x1)

        P_N1, P_F1 = np.array_split(yn1, self.split_y)

        mu = self.system.mu
        prox_r_N = self.prox_r_N
        prox_r_F = self.prox_r_F

        yn1p = np.zeros_like(yn1) # initialize projected forces

        ##############################
        # fixed-point update Signorini
        ##############################
        g_N = self.system.g_N(tn1, qn1)
        prox_arg = (prox_r_N / self.dt) * g_N - P_N1
        yn1p[: self.split_y[0]] = -prox_R0_nm(prox_arg)

        #############################
        # fixed-point update friction
        #############################
        gamma_F = self.system.gamma_F(tn1, qn1, un12)
        for i_N, i_F in enumerate(self.system.NF_connectivity):
            if len(i_F):
                yn1p[self.split_y[0] + np.array(i_F)] = -prox_sphere(
                    prox_r_F[i_N] * gamma_F[i_F] - P_F1[i_F],
                    mu[i_N] * P_N1[i_N],
                )

        return yn1p

    def solve(self):
        solver_summary = So
        # lists storing output variables
        q = [self.qn]
        u = [self.un]
        P_g = [self.la_gn]
        P_gamma = [self.la_gamman]
        P_N = [self.dt * self.la_Nn]
        P_F = [self.dt * self.la_Fn]

        self.R_g2 = self.la_gn
        self.R_gamma2 = self.la_gamman

        pbar = tqdm(self.t[:-1])
        # pbar = tqdm(self.t)
        for n in pbar:
            # # only compute optimized proxparameters once per time step
            # self.prox_r_N = self.system.prox_r_N(self.tn, self.qn)
            # self.prox_r_F = self.system.prox_r_F(self.tn, self.qn)
            # print(f"prox_r_N: {self.prox_r_N}")
            # print(f"prox_r_F: {self.prox_r_F}")

            # TODO: We need a possibility to set prox parameters on system level!

            # ########################
            # # rotating bouncing ball
            # ########################
            # self.prox_r_N = np.ones(self.nla_N) * 0.5
            # self.prox_r_F = np.ones(self.nla_F) * 0.5

            ##########
            # tippetop
            ##########
            # self.prox_r_N = np.ones(self.nla_N) * 0.001
            # self.prox_r_F = np.ones(self.nla_F) * 0.001
            self.prox_r_N = np.ones(self.nla_N) * 1
            self.prox_r_F = np.ones(self.nla_F) * 1

            # ##############
            # # slider crank
            # ##############
            # self.prox_r_N = np.ones(self.nla_N) * 1
            # self.prox_r_F = np.ones(self.nla_F) * 1

            tn1 = self.tn + self.dt

            if self.method == "Newton_decoupled":
                y1, converged1, error1, i1, _ = fsolve(
                    self.R1,
                    self.y1n,
                    jac=self.J1,
                    atol=self.atol,
                    max_iter=self.max_iter,
                    fun_args=(True,),
                    jac_args=(False,),
                )

                (
                    self.qn1,
                    self.un12,
                    self.R_g1,
                    self.R_gamma1,
                    self.R_N1,
                    self.R_F1,
                ) = np.array_split(y1, self.split_y1)

                # compute constant quantities for next stage
                self.Mn1 = self.system.M(tn1, self.qn1, scipy_matrix=csr_matrix)
                self.W_gn1 = self.system.W_g(tn1, self.qn1, scipy_matrix=csr_matrix)
                self.W_gamman1 = self.system.W_gamma(
                    tn1, self.qn1, scipy_matrix=csr_matrix
                )
                self.W_Nn1 = self.system.W_N(tn1, self.qn1, scipy_matrix=csr_matrix)
                self.W_Fn1 = self.system.W_F(tn1, self.qn1, scipy_matrix=csr_matrix)

                y2, converged2, error2, i2, _ = fsolve(
                    self.R2,
                    self.y2n,
                    jac=self.J2,
                    atol=self.atol,
                    max_iter=self.max_iter,
                    fun_args=(True,),
                    jac_args=(False,),
                )

                un1, R_g2, R_gamma2, R_N2, R_F2 = np.array_split(y2, self.split_y2)

                converged = converged1 and converged2
                error = error1 + error2
                i = i1 + i2

                P_gn1 = self.dt * 0.5 * (self.R_g1 + R_g2)
                P_gamman1 = self.dt * 0.5 * (self.R_gamma1 + R_gamma2)
                P_Nn1 = self.dt * 0.5 * (self.R_N1 + R_N2)
                P_Fn1 = self.dt * 0.5 * (self.R_F1 + R_F2)

                qn1, un1 = self.system.step_callback(tn1, self.qn1, un1)

            elif self.method == "Newton_full":
                y, converged, error, i, _ = fsolve(
                    self.R,
                    self.yn,
                    # jac="2-point",
                    jac="3-point",  # TODO: keep this, otherwise sinuglairites arise
                    eps=1.0e-6,
                    atol=self.atol,
                    max_iter=self.max_iter,
                    fun_args=(True,),
                    jac_args=(False,),
                )

                (
                    qn1,
                    un12,
                    un1,
                    R_g1,
                    R_g2,
                    R_gamma1,
                    R_gamma2,
                    R_N1,
                    R_N2,
                    R_F1,
                    R_F2,
                ) = np.array_split(y, self.split_y)

                P_gn1 = self.dt * 0.5 * (R_g1 + R_g2)
                P_gamman1 = self.dt * 0.5 * (R_gamma1 + R_gamma2)
                P_Nn1 = self.dt * 0.5 * (R_N1 + R_N2)
                P_Fn1 = self.dt * 0.5 * (R_F1 + R_F2)

                # P_gn1 = R_g2
                # P_gamman1 = R_gamma2
                # P_Nn1 = R_N2
                # P_Fn1 = R_F2

                # # P_gn1 = 0.5 * (R_g1 + R_g2)
                # # P_gamman1 = 0.5 * (R_gamma1 + R_gamma2)
                # # P_gn1 = self.dt * 0.5 * (R_g1 + R_g2)
                # # P_gamman1 = self.dt * 0.5 * (R_gamma1 + R_gamma2)
                # # P_gn1 = R_g1
                # # P_gamman1 = R_gamma1
                # P_gn1 = R_g2
                # P_gamman1 = R_gamma2
                # # P_gn1 = 0.5 * (R_g1 + self.R_g2)
                # # P_gamman1 = 0.5 * (R_gamma1 + self.R_gamma2)

                self.R_g2 = R_g2.copy()
                self.R_gamma2 = R_gamma2.copy()

                qn1, un1 = self.system.step_callback(tn1, qn1, un1)

                i1, i2 = i, "-"
                error1, error2 = error, np.nan

            elif self.method == "fixed_point":
                z10 = self.z1n.copy()
                for i1 in range(self.fix_point_max_iter):
                    z1 = self.p1(z10)

                    # convergence percussions
                    # error1 = self.fix_point_error_function(z1 - z10)

                    # convergence positions and velocities
                    error1 = self.fix_point_error_function(
                        self.x1[: self.nq + self.nu] - self.x10[: self.nq + self.nu]
                    )

                    converged1 = error1 < self.fix_point_tol
                    if converged1:
                        break
                    z10 = z1

                qn1, un12, R_g1, R_gamma1 = np.array_split(self.x1, self.split_x1)

                #################
                # second stage
                #################

                # get quantities from system
                Mn = self.system.M(tn1, qn1)
                h = self.system.h(tn1, qn1, un12)
                W_g = self.system.W_g(tn1, qn1)
                W_gamma = self.system.W_gamma(tn1, qn1)
                chi_g = self.system.g_dot(tn1, qn1, np.zeros_like(un12))
                chi_gamma = self.system.gamma(tn1, qn1, np.zeros_like(un12))
                # note: we use csc_matrix for efficient column slicing later,
                # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html#scipy.sparse.csc_array
                W_N = self.system.W_N(tn1, qn1, scipy_matrix=csc_matrix)
                W_F = self.system.W_F(tn1, qn1, scipy_matrix=csc_matrix)

                A = bmat(
                    [
                        [Mn, -0.5 * W_g, -0.5 * W_gamma],
                        [-W_g.T, None, None],
                        [-W_gamma.T, None, None],
                    ],
                    format="csc",
                )

                lu_A = splu(A)

                # initial right hand side
                b = np.concatenate(
                    (
                        Mn @ un12 + 0.5 * self.dt * h,
                        chi_g,
                        chi_gamma,
                    )
                )

                z20 = self.z2n.copy()
                for i2 in range(self.fix_point_max_iter):
                    # z2 = self.p2(z1, z20)
                    z2 = self.p2(z1, z20, lu_A, b, W_N, W_F)

                    # convergence percussions
                    # error2 = self.fix_point_error_function(z2 - z20)

                    # convergence positions and velocities
                    error2 = self.fix_point_error_function(
                        self.x2[: self.nu] - self.x20[: self.nu]
                    )

                    converged2 = error2 < self.fix_point_tol
                    if converged2:
                        break
                    z20 = z2

                converged = converged1 and converged2
                error = error1 + error2
                i = i1 + i2
                print(f"i: {i}, i1: {i1}, i2: {i2}")

                (
                    un1,
                    R_g2,
                    R_gamma2,
                ) = np.array_split(self.x2, self.split_x2)

                P_Nn1 = z2[: self.nla_N]
                P_Fn1 = z2[self.nla_N :]

                P_gn1 = 0.5 * (R_g1 + R_g2)
                P_gamman1 = 0.5 * (R_gamma1 + R_gamma2)

                qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            elif self.method == "fixed_point_nonlinear_full":
                #######################
                # fixed-point iteration
                #######################
                z0 = self.zn.copy()
                for i2 in range(self.fix_point_max_iter):
                    z = self.p(z0)

                    # convergence percussions
                    # error2 = self.fix_point_error_function(z - z0)

                    # convergence positions and both velocities
                    error2 = self.fix_point_error_function(
                        self.x[: self.split_x[2]] - self.x0[: self.split_x[2]]
                    )

                    converged2 = error2 < self.fix_point_tol
                    if converged2:
                        break
                    z0 = z

                qn1, un12, un1, R_g1, R_g2, R_gamma1, R_gamma2 = np.array_split(
                    self.x, self.split_x
                )
                _, P_Nn1, _, P_Fn1 = np.array_split(z, self.split_z)

                P_gn1 = 0.5 * (R_g1 + R_g2)
                P_gamman1 = 0.5 * (R_gamma1 + R_gamma2)

                i1 = self.i1
                error1 = self.error1
                converged = self.converged1 and converged2
                i = i1

                ##########################
                # constrained optimization
                ##########################
                from scipy.optimize import minimize, NonlinearConstraint

                def fun(y):
                    x = y[: self.nx]
                    z = y[self.nx :]
                    F = self.F(x, *np.array_split(z, self.split_z))
                    return F @ F

                def fun_c(y):
                    x = y[: self.nx]
                    z = y[self.nx :]
                    return self.c(x, z)

                lb = ub = np.zeros(2 * self.nla_N + 2 * self.nla_F)
                constraints = NonlinearConstraint(fun_c, lb, ub)

                y0 = self.yn
                sol = minimize(fun, y0, method="SLSQP", constraints=[constraints])
                converged = sol.success
                y = sol.x
                x = y[: self.nx]
                z = y[self.nx :]
                i1 = sol.nit
                i2 = np.nan
                error1 = sol.fun
                error2 = np.nan
            else:
                raise NotImplementedError

            pbar.set_description(
                f"t: {tn1:0.2e}; iter: {i1},{i2}; error: {error1:.3e}, {error2:.3e}"
            )
            if not converged:
                if self.continue_with_unconverged:
                    print(
                        f"step {n:.0f} is not converged after i1: {i1}, i2: {i2} iterations with error:  err1 = {error1:.3e}, err2 = {error2:.3e}"
                    )
                else:
                    raise RuntimeError(
                        f"step {n:.0f} is not converged after i1: {i1}, i2: {i2} iterations with error:  err1 = {error1:.3e}, err2 = {error2:.3e}"
                    )

            # from cardillo.solver import constraint_forces
            # u_dotn1, P_gn1, P_gamman1 = constraint_forces(self.system, tn1, qn1, un1)

            # t.append(tn1)
            q.append(qn1)
            u.append(un1)
            P_g.append(P_gn1)
            P_gamma.append(P_gamman1)
            P_N.append(P_Nn1)
            P_F.append(P_Fn1)
            # q.append(qn1.copy())
            # u.append(un1.copy())
            # P_g.append(P_gn1.copy())
            # P_gamma.append(P_gamman1.copy())
            # P_N.append(P_Nn1.copy())
            # P_F.append(P_Fn1.copy())

            # update local variables for accepted time step
            self.tn = tn1
            if self.method == "Newton_decoupled":
                self.y1n = y1.copy()
                self.y2n = y2.copy()
                self.qn = qn1.copy()
                self.un = un1.copy()
            elif self.method == "Newton_full":
                self.yn = y.copy()
            elif self.method == "fixed_point":
                self.z1n = z1.copy()
                self.z2n = z2.copy()
                self.qn = qn1.copy()
                self.un = un1.copy()
            elif self.method == "fixed_point_nonlinear_full":
                self.zn = z.copy()
                self.qn = qn1.copy()
                self.un = un1.copy()
            else:
                raise NotImplementedError

        return Solution(
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
        )
