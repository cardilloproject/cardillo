from cardillo.math import approx_fprime, fsolve
from cardillo.solver.solution import Solution

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_array, csc_array, coo_array, bmat, lil_array
from tqdm import tqdm


class Newton:
    """Force and displacement controlled Newton-Raphson method. This solver
    is used to find a static solution for a mechanical system. Forces and
    bilateral constraint functions are incremented in each load step if they
    depend on the time t in [0, 1]. Thus, a force controlled Newton-Raphson method
    is obtained by constructing a time constant constraint function function.
    On the other hand a displacement controlled Newton-Raphson method is
    obtained by passing constant forces and time dependent constraint functions.
    """

    def __init__(
        self,
        system,
        n_load_steps=1,
        atol=1e-8,
        max_iter=50,
        jac=None,
        eps=1.0e-6,
        verbose=True,
        error_function=lambda x: np.max(np.absolute(x)),
    ):
        self.system = system
        self.eps = eps
        self.error_function = error_function
        self.verbose = verbose

        # compute Jacobian matrix using finite differences
        if jac in ["2-point", "3-point", "cs"]:
            self.jac = lambda x, *args: csc_array(
                approx_fprime(x, lambda y: self.fun(y, *args), method=jac, eps=eps)
            )
        else:
            self.jac = self.__jac

        self.atol = atol
        self.max_iter = max_iter
        self.load_steps = np.linspace(0, 1, n_load_steps + 1)
        self.nt = len(self.load_steps)

        self.len_t = len(str(self.nt))
        self.len_maxIter = len(str(self.max_iter))

        # other dimensions
        self.nq = system.nq
        self.nu = system.nu
        self.nla_N = system.nla_N

        self.split_f = np.cumsum(
            np.array(
                [system.nu, system.nla_g, system.nla_c, system.nla_S],
                dtype=int,
            )
        )
        self.split_x = np.cumsum(
            np.array(
                [system.nq, system.nla_g, system.nla_c],
                dtype=int,
            )
        )

        # initial conditions
        x0 = np.concatenate((system.q0, system.la_g0, system.la_c0, system.la_N0))
        nx = len(x0)
        self.u0 = np.zeros(system.nu)  # zero velocities as system is static

        # memory allocation
        self.x = np.zeros((self.nt, nx), dtype=float)
        self.x[0] = x0

    def fun(self, x, t):
        # unpack unknowns
        q, la_g, la_c, la_N = np.array_split(x, self.split_x)

        # evaluate quantites that are required for computing the residual and
        # the jacobian
        # csr is used for efficient matrix vector multiplication, see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
        self.W_g = self.system.W_g(t, q, scipy_matrix=csr_array)
        self.W_c = self.system.W_c(t, q, scipy_matrix=csr_array)
        self.W_N = self.system.W_N(t, q, scipy_matrix=csr_array)
        self.g_N = self.system.g_N(t, q)

        # static equilibrium
        F = np.zeros_like(x)
        F[: self.split_f[0]] = (
            self.system.h(t, q, self.u0)
            + self.W_g @ la_g
            + self.W_c @ la_c
            + self.W_N @ la_N
        )
        F[self.split_f[0] : self.split_f[1]] = self.system.g(t, q)
        F[self.split_f[1] : self.split_f[2]] = self.system.c(t, q, self.u0, la_c)
        F[self.split_f[2] : self.split_f[3]] = self.system.g_S(t, q)
        F[self.split_f[3] :] = np.minimum(la_N, self.g_N)
        return F

    def __jac(self, x, t):
        # unpack unknowns
        q, la_g, la_c, la_N = np.array_split(x, self.split_x)

        # evaluate additionally required quantites for computing the jacobian
        # coo is used for efficient bmat
        K = (
            self.system.h_q(t, q, self.u0, scipy_matrix=coo_array)
            + self.system.Wla_g_q(t, q, la_g, scipy_matrix=coo_array)
            + self.system.Wla_c_q(t, q, la_c, scipy_matrix=coo_array)
            + self.system.Wla_N_q(t, q, la_N, scipy_matrix=coo_array)
        )
        g_q = self.system.g_q(t, q, scipy_matrix=coo_array)
        g_S_q = self.system.g_S_q(t, q, scipy_matrix=coo_array)
        c_q = self.system.c_q(t, q, self.u0, la_c, scipy_matrix=coo_array)
        c_la_c = self.system.c_la_c(t, q, self.u0, la_c, scipy_matrix=coo_array)

        # note: csr_matrix is best for row slicing, see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
        g_N_q = self.system.g_N_q(t, q, scipy_matrix=coo_array)

        Rla_N_q = lil_array((self.nla_N, self.nq), dtype=float)
        Rla_N_la_N = lil_array((self.nla_N, self.nla_N), dtype=float)
        for i in range(self.nla_N):
            if la_N[i] < self.g_N[i]:
                Rla_N_la_N[i, i] = 1.0
            else:
                Rla_N_q[i] = g_N_q[i]

        # fmt: off
        return bmat([[      K, self.W_g, self.W_c,   self.W_N], 
                     [    g_q,     None,     None,       None],
                     [    c_q,     None,   c_la_c,       None],
                     [  g_S_q,     None,     None,       None],
                     [Rla_N_q,     None,     None, Rla_N_la_N]], format="csc")
        # fmt: on

    def __pbar_text(self, force_iter, newton_iter, error):
        return (
            f" force iter {force_iter+1:>{self.len_t}d}/{self.nt};"
            f" Newton steps {newton_iter+1:>{self.len_maxIter}d}/{self.max_iter};"
            f" error {error:.4e}/{self.atol:.2e}"
        )

    def solve(self):
        pbar = range(0, self.nt)
        if self.verbose:
            pbar = tqdm(pbar, leave=True)
        for i in pbar:
            self.x[i], converged, error, k, _ = fsolve(
                self.fun,
                self.x[i],
                jac=self.jac,
                fun_args=(self.load_steps[i],),
                jac_args=(self.load_steps[i],),
                error_function=self.error_function,
                atol=self.atol,
                max_iter=self.max_iter,
            )
            pbar.set_description(self.__pbar_text(i, k, error))

            if not converged:
                # return solution up to this iteration
                if self.verbose:
                    pbar.close()
                print(
                    f"Newton-Raphson method not converged, returning solution "
                    f"up to iteration {i+1:>{self.len_t}d}/{self.nt}"
                )
                return Solution(
                    t=self.load_steps,
                    q=self.x[: i + 1, : self.nq],
                    la_g=self.x[: i + 1, self.nq :],
                    la_N=self.x[: i + 1, self.nq + self.nla_g :],
                )

            # solver step callback
            self.x[i, : self.split_x[0]], _ = self.system.step_callback(
                self.load_steps[i], self.x[i, : self.split_x[0]], self.u0
            )

            # warm start for next step; store solution as new initial guess
            if i < self.nt - 1:
                self.x[i + 1] = self.x[i]

        # return solution object
        if self.verbose:
            pbar.close()
        return Solution(
            self.system,
            t=self.load_steps,
            q=self.x[: i + 1, : self.split_x[0]],
            u=np.zeros((len(self.load_steps), self.nu)),
            la_g=self.x[: i + 1, self.split_x[0] : self.split_x[1]],
            la_c=self.x[: i + 1, self.split_x[1] : self.split_x[2]],
            la_N=self.x[: i + 1, self.split_x[2] :],
        )


# TODO: Understand predictor of Feng mentioned in Neto1999.
# TODO: automatic increment cutting: Crisfield1991 section 9.5.1
# TODO: read Crisfield1996 section 21 Branch switching and further advanced solution procedures
# TODO: implement line searcher technique mentioned in Crisfield1991 and Crisfield1996
# TODO: implement dense output
class Riks:
    """Linear arc-length solver close to Riks method as dervied in Crisfield1991 
    section 9.3.2 p.273. A variable arc-length is chosen as shown by 
    Crisfield1981 or Crisfield 1983. For the first predictor a tangent predictor 
    is used. For all other predictors a simple secant predictor is used. This 
    enables the solver to 'run forward' instead of 'doubling back on its track'.

    References
    ----------
    - Wempner1971: https://doi.org/10.1016/0020-7683(71)90038-2 \\
    - Riks1972: https://doi.org/10.1115/1.3422829 \\
    - Riks1979: https://doi.org/10.1016/0020-7683(79)90081-7 \\
    - Crsfield1981: https://doi.org/10.1016/0045-7949(81)90108-5 \\
    - Crisfield1991: http://freeit.free.fr/Finite%20Element/Crisfield%20M.A.%20Vol.1.%20Non-Linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Essentials%20(Wiley,19.pdf \\
    - Crisfield1996: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf \\
    - Neto1999: https://doi.org/10.1016/S0045-7825(99)00042-0
    """

    def a(self, x):
        """The most simple arc length equation restricting the change of all
        generalized coordinates w.r.t. the last converged Newton step."""
        nq = self.nq
        dq = x[:nq] - self.xk[:nq]
        return dq @ dq

    def a_x(self, x):
        nq = self.nq
        dq = x[:nq] - self.xk[:nq]
        return 2 * dq, np.zeros(self.nla_g), 0

    def R(self, x):
        return next(self.gen_analytic(x))

    def gen_numeric(self, x):
        yield self.R(x)
        yield approx_fprime(x, self.R, method="2-point")

    def gen_analytic(self, x):
        # extract generalized coordinates and Lagrange multipliers
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_S = self.nla_S

        q = x[:nq]
        la_g = x[nq : nq + nla_g]
        la_arc = x[-1]

        # evaluate all functions with t = la_arc -> model does not change!
        # - this requires the external force that should be scaled to be of the form
        #   F_ext(t, q) = t * F(q)
        # - the constraints for displacement control have to be of the form
        #   g(t, q) = t * g(q)
        t = la_arc
        h = self.model.h(t, q, self.u0)
        g = self.model.g(t, q)
        W_g = self.model.W_g(t, q)

        # build residual
        R = np.zeros(self.nx)
        R[:nu] = h + W_g @ la_g
        R[nu : nu + nla_g] = g
        R[nu + nla_g : nu + nla_g + nla_S] = self.model.g_S(t, q)
        R[-1] = self.a(x) - self.ds**2

        yield R

        # evaluate all functions with t = la_arc -> model does not change!
        # - this requires the external force that should be scaled to be of the form
        #   F_ext(t, q) = t * F(q)
        # - the constraints for displacement control have to be of the form
        #   g(t, q) = t * g(q)
        h_q = self.model.h_q(t, q, self.u0)
        Wla_g_q = self.model.Wla_g_q(t, q, la_g)
        Rq_q = h_q + Wla_g_q
        g_q = self.model.g_q(t, q)
        g_S_q = self.model.g_S_q(t, q)

        # TODO: this is a hack and can't be fixed without specifying the scaled equations
        # but it is only two addition evaluations of the h vector and the constraints
        eps = 1.0e-6
        f_arc = (self.model.h(t + eps, q, self.u0) - h) / eps
        g_arc = (self.model.g(t + eps, q) - g) / eps

        # derivative of the arc length equation
        a_q, a_la, a_la_arc = self.a_x(x)

        yield bmat(
            [
                [Rq_q, W_g, f_arc[:, None]],
                [g_q, None, g_arc[:, None]],
                [g_S_q, None, None],
                [a_q, a_la, a_la_arc],
            ],
            format="csr",
        )

    def __init__(
        self,
        system,
        atol=1e-8,
        max_iter=50,
        iter_goal=4,
        la_arc0=1.0e-3,
        la_arc_span=np.array([0, 1], dtype=float),
        numerical_jacobian=False,
        scale_exponent=0.5,
        debug=0,
    ):
        self.atol = atol
        self.max_iter = max_iter
        self.model = system
        self.iter_goal = iter_goal
        self.la_arc0 = la_arc0
        self.la_arc_span = la_arc_span

        self.newton_error_function = lambda R: np.absolute(R).max()

        if numerical_jacobian:
            self.gen = self.gen_numeric
        else:
            self.gen = self.gen_analytic

        # parameter for the step size scaling
        self.MIN_FACTOR = 0.25  # minimal scaling factor
        self.MAX_FACTOR = 1.5  # maximal scaling factor

        # dimensions
        self.nq = self.model.nq
        self.nu = self.model.nu
        self.nla_g = self.model.nla_g
        self.nla_S = self.model.nla_S
        self.nx = self.nq + self.nla_g + 1

        # initial
        self.q0 = self.model.q0
        self.la_g0 = self.model.la_g0
        self.la_arc0 = la_arc0
        self.u0 = np.zeros(system.nu)  # statics

        # initial values for generalized coordinates, lagrange multipliers and force scaling
        self.x0 = self.xk = np.concatenate(
            (self.q0, self.la_g0, np.array([self.la_arc0]))
        )

        ####################################################################################################
        # Solve linearized system for fixed external force using Newtons method.
        # From this solution we can extract the initial ds using the arc length equation.
        # All other ds values will be modified according to the number of used Newton steps,
        # see https://scicomp.stackexchange.com/questions/28137/initialize-arc-length-control-in-riks-method
        ####################################################################################################
        print(f"solve linear system using the initial arc length parameter")
        self.ds = 0  # initial ds has to be zero!
        xk1 = self.x0.copy()  # this copy is essential!
        gen = self.gen(xk1)
        R = next(gen)[:-1]
        error = self.newton_error_function(R)
        newton_iter = 0
        print(f"   * iter = {newton_iter}, error = {error:2.4e}")
        while error > self.atol:
            newton_iter += 1
            R_x = next(gen)[:-1, :-1]
            xk1[:-1] -= spsolve(R_x, R)
            gen = self.gen(xk1)
            R = next(gen)[:-1]
            error = self.newton_error_function(R)
            print(f"   * iter = {newton_iter}, error = {error:2.4e}")

        # compute initial ds from arc-length equation
        self.ds = self.a(xk1) ** 0.5
        if self.ds <= 0:
            raise RuntimeError("ds <= 0")
        print(
            f" => Newton converged in {newton_iter} iterations with error {error:2.4e}; initial ds: {self.ds:2.4e}"
        )

        # chose scaling exponent, see https://scicomp.stackexchange.com/questions/28137/initialize-arc-length-control-in-riks-method
        self.scale_ds = False
        if scale_exponent is not None:
            self.scale_ds = True
            self.scale_exponent = scale_exponent

        # debug information or not
        self.debug = debug

    def solve(self):
        # count number of force increments to get first increment with tangential predictor
        i = 0

        # extract number of generalized coordinates and number of Lagrange multipliers
        nq = self.nq
        nla = self.nla_g

        # initialize current generalized coordinates, Lagrange multipliers and force scaling
        la_arc = [self.la_arc0]
        q = [self.q0]
        la_g = [self.la_g0]
        xk1 = np.concatenate((self.q0, self.la_g0, np.array([self.la_arc0])))

        # compute initial residual and error
        gen = self.gen(xk1)
        R = next(gen)
        error = self.newton_error_function(R)

        # loop over ranges of force scaling
        while xk1[-1] > self.la_arc_span[0] and xk1[-1] < self.la_arc_span[1]:
            i += 1
            # use secant predictor for all other force increments than the first one
            if i > 1:
                # secand predictor for all but the first newton iteration
                dx = self.xk - self.x0
                xk1 += dx

                # ###################################
                # # prediction of Feng (see Neto1998)
                # ##################################
                # # secant predictor
                # Dx = self.xk - self.x0

                # # tangent predictor
                # # gen = self.gen(xk1)
                # # R = next(gen)
                # R_x = next(gen)
                # dx = spsolve(R_x, R)

                # inner = dx @ Dx
                # sign_inner = sign(inner)

                # # update with correspinding sign
                # xk1[:-1] += sign_inner * Dx[:-1]
                # # xk1[:-1] -= sign_inner * dx[:-1]

            else:
                # TODO:
                # find out why it is essential to solve for the generalized coordinates
                # and Lagrange-multipliers but not for the external force scaling
                R_x = next(gen)
                dx = spsolve(R_x[:-1, :-1], R[:-1])
                xk1[:-1] -= dx

            # compute initial residual and error
            gen = self.gen(xk1)
            R = next(gen)
            error = self.newton_error_function(R)

            newton_iter = 0
            if self.debug > 0:
                print(f"   * iter = {newton_iter}, error = {error:2.4e}")
            while (error > self.atol) and (newton_iter <= self.max_iter):
                # solve linear system of equations
                R_x = next(gen)
                dx = spsolve(R_x, R)
                xk1 -= dx
                newton_iter += 1

                # check for convergence
                gen = self.gen(xk1)
                R = next(gen)
                error = self.newton_error_function(R)
                if self.debug > 0:
                    print(f"   * iter = {newton_iter}, error = {error:2.4e}")

            if newton_iter >= self.max_iter:
                print(
                    f" - internal Newton not converged for lambda = {xk1[-1]:2.4e} with error {error:2.2e}."
                )
                print(f"error = {error}")
                break
            else:
                print(
                    f" - internal Newton converged for lambda = {xk1[-1]:2.4e} with error {error:2.2e} in {newton_iter} steps."
                )

            # scale ds such that iter goal is satisfied
            # disable scaling if we have halfed the ds parameter before or after the first iteration which requires lots of iterations
            # see Crisfield1991, section 9.5 (9.40) or (9.41) for the square root scaling
            if self.scale_ds and newton_iter != 0 and i > 1:
                fac = (self.iter_goal / newton_iter) ** self.scale_exponent
                if self.debug > 1:
                    ds_old = self.ds
                self.ds *= min(self.MAX_FACTOR, max(self.MIN_FACTOR, fac))
                if self.debug > 1:
                    print(f"   * ds: {ds_old} => {self.ds}")

            # store last converged newton step
            self.x0 = self.xk.copy()

            # store new converged newton step
            self.xk = xk1.copy()

            # append solutions to lists
            # these copies are essential!
            q.append(xk1[:nq].copy())
            la_g.append(xk1[nq : nq + nla].copy())
            la_arc.append(xk1[-1].copy())

        # return solution object
        return Solution(t=np.asarray(la_arc), q=np.asarray(q), la_g=np.asarray(la_g))
