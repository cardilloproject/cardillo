from cardillo.math import approx_fprime
from cardillo.solver.Solution import Solution

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, bmat, lil_matrix
from tqdm import tqdm


def scaled_tolerance(yk, yk1, atol, rtol):
    """Scaled tolerance defined in Hairer1993 p. 167 (4.10).

    References
    ----------
    Hairer1993: https://doi.org/10.1007/978-3-540-78862-1
    """
    return atol + rtol * np.maximum(np.abs(yk), np.abs(yk1))


def is_converged(Rk1, yk, yk1, atol, rtol):
    """Check if error measure defined in Hairer1993 p. 167 (4.10) is satisfied.

    References
    ----------
    Hairer1993: https://doi.org/10.1007/978-3-540-78862-1
    """
    # compute scaled tolerances
    sc = scaled_tolerance(yk, yk1, atol, rtol)

    # check element wise convergence
    abs_Rk1 = np.abs(Rk1)
    error = abs_Rk1 - sc
    converged = np.all(error <= 0)

    # find maximum error and the corresponding tolerance
    idx_abs_Rk1 = np.argmax(error)
    max_error = abs_Rk1[idx_abs_Rk1]
    max_sc = sc[idx_abs_Rk1]

    return converged, max_error, max_sc


class Newton:
    """Force and displacement controlled Newton-Raphson method. This solver
    is used to find a static solution for a mechanical system. External forces
    and bilateral constraint functions are incremented in each load step if
    they depend on the time t. Thus, a force controlled Netwon-Raphson method
    is obtained by constructing a time constant constraint function function.
    On the other hand a displacement controlled Newton-Raphson method is
    obtained by passing a constant external forces and time dependent
    constraint functions.

    Dirichlet constraints can globally be handled by passing an integer array
    `cDOF` of the constraint degrees of freedoms together with a time dependent
    function `b` that defines their values at time instant `t`.
    """

    def __init__(
        self,
        model,
        cDOF_q=np.array([], dtype=int),
        cDOF_u=np.array([], dtype=int),
        cDOF_S=np.array([], dtype=int),
        b=lambda t: np.array([], dtype=float),
        n_load_steps=1,
        load_steps=None,
        atol=1e-8,
        rtol=1e-6,
        max_iter=50,
        prox_r_N=1.0e-2,
        numerical_jacobian=False,
        verbose=True,
        # newton_error_function=lambda x: np.max(np.abs(x)),
        numdiff_method="2-point",
        numdiff_eps=1.0e-6,
    ):
        self.model = model

        # handle constraint degrees of freedoms
        z0_q = model.q0.copy()
        z0_u = model.u0.copy()
        self.nz_q = len(z0_q)
        self.nz_u = len(z0_u)
        self.nz_S = model.nla_S
        self.cDOF_q = cDOF_q
        self.cDOF_u = cDOF_u
        self.cDOF_S = cDOF_S
        self.zDOF_q = np.arange(self.nz_q)
        self.zDOF_u = np.arange(self.nz_u)
        self.zDOF_S = np.arange(self.nz_S)
        self.fDOF_q = np.setdiff1d(self.zDOF_q, cDOF_q)
        self.fDOF_u = np.setdiff1d(self.zDOF_u, cDOF_u)
        self.fDOF_S = np.setdiff1d(self.zDOF_S, cDOF_S)
        q0 = z0_q[self.fDOF_q]
        u0 = z0_u[self.fDOF_u]

        # print(f"zDOF: {self.zDOF}")
        # print(f"cDOF: {cDOF}")
        # print(f"fDOF: {self.fDOF}")

        if callable(b):
            self.b = b
        else:
            self.b = lambda t: b

        self.max_iter = max_iter
        self.prox_r_N = prox_r_N

        if load_steps is None:
            self.load_steps = np.linspace(0, 1, n_load_steps)
        else:
            self.load_steps = np.array(load_steps)

        # other dimensions
        self.nt = len(self.load_steps)
        nc_q = len(cDOF_q)
        nc_u = len(cDOF_u)
        nc_S = len(cDOF_S)
        self.nq = self.nz_q - nc_q
        self.nu = self.nz_u - nc_u
        self.nla_g = self.model.nla_g
        # self.nla_S = self.model.nla_S
        self.nla_S = self.nz_S - nc_S
        self.nla_N = self.model.nla_N
        self.nx = self.nq + self.nla_g + self.nla_N
        self.nf = self.nu + self.nla_g + self.nla_S + self.nla_N

        # build atol, rtol vectors if scalars are given
        self.atol = np.atleast_1d(atol)
        self.rtol = np.atleast_1d(rtol)
        if len(self.atol) == 1:
            self.atol = np.ones(self.nf, dtype=float) * atol
        if len(self.rtol) == 1:
            self.rtol = np.ones(self.nf, dtype=float) * rtol
        assert len(self.atol) == self.nf
        assert len(self.rtol) == self.nf

        # memory allocation
        self.x = np.zeros((self.nt, self.nx))

        # initial conditions
        self.x[0] = np.concatenate((q0, self.model.la_g0, self.model.la_N0))
        self.u = np.zeros(self.nz_u)  # zero velocities as system is static

        self.numdiff_method = numdiff_method
        self.numdiff_eps = numdiff_eps
        self.verbose = verbose

        if numerical_jacobian:
            self.__eval__ = self.__eval__num
        else:
            self.__eval__ = self.__eval__analytic

    def z(self, t, q):
        z = np.zeros(self.nz_q)
        z[self.fDOF_q] = q
        z[self.cDOF_q] = self.b(t)
        return z

    def __eval__analytic(self, t, x):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_S = self.nla_S

        q = x[:nq]
        la_g = x[nq : nq + nla_g]
        la_N = x[nq + nla_g :]

        # compute redundant coordinates
        z = self.z(t, q)

        # evaluate quantites that are required for computing the residual and
        # the jacobian
        W_g = self.model.W_g(t, z, scipy_matrix=csr_matrix)[self.fDOF_u]
        W_N = self.model.W_N(t, z, scipy_matrix=csr_matrix)[self.fDOF_u]
        g_N = self.model.g_N(t, z)

        R = np.zeros(self.nf)
        R[:nu] = self.model.h(t, z, self.u)[self.fDOF_u] + W_g @ la_g + W_N @ la_N
        R[nu : nu + nla_g] = self.model.g(t, z)
        R[nu + nla_g : nu + nla_g + nla_S] = self.model.g_S(t, z)[self.fDOF_S]
        R[nu + nla_g + nla_S :] = np.minimum(la_N, g_N)

        yield R

        # evaluate additionally required quantites for computing the jacobian
        K = (
            self.model.h_q(t, z, self.u, scipy_matrix=csr_matrix)[
                self.fDOF_u[:, None], self.fDOF_q
            ]
            + self.model.Wla_g_q(t, z, la_g, scipy_matrix=csr_matrix)[
                self.fDOF_u[:, None], self.fDOF_q
            ]
        )
        g_q = self.model.g_q(t, z, scipy_matrix=csr_matrix)[:, self.fDOF_q]
        g_S_q = self.model.g_S_q(t, z, scipy_matrix=csr_matrix)[
            self.fDOF_S[:, None], self.fDOF_q
        ]
        # note: csr_matrix is best for row slicing, see
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
        g_N_q = self.model.g_N_q(t, z, scipy_matrix=csr_matrix)[:, self.fDOF_q]

        Rla_N_q = lil_matrix((self.nla_N, self.nq))
        Rla_N_la_N = lil_matrix((self.nla_N, self.nla_N))
        for i in range(self.nla_N):
            if la_N[i] < g_N[i]:
                Rla_N_la_N[i, i] = 1.0
            else:
                Rla_N_q[i] = g_N_q[i]

        # fmt: off
        # TODO: What is more efficient: Using csr or csc format?
        yield bmat([[  K,     W_g,         W_N], 
                    [g_q,     None,       None],
                    [g_S_q,   None,       None],
                    [Rla_N_q, None, Rla_N_la_N]], format="csr")
        # fmt: on

    def __residual(self, t, x):
        return next(self.__eval__(t, x))

    def __jacobian(self, t, x, scipy_matrix=csr_matrix):
        return scipy_matrix(
            approx_fprime(
                x,
                lambda x: self.__residual(t, x),
                eps=self.numdiff_eps,
                method=self.numdiff_method,
            )
        )

    def __eval__num(self, t, x):
        yield next(self.__eval__analytic(t, x))
        yield self.__jacobian(t, x)

    def solve(self):
        # compute numbe rof digits for status update
        len_t = len(str(self.nt))
        len_maxIter = len(str(self.max_iter))

        if self.verbose:
            pbar = tqdm(range(0, self.nt), leave=True)
        else:
            pbar = range(0, self.nt)
        for i in pbar:

            # ti = self.load_steps[i]
            # qi = self.x[i, : self.nq]
            # zi = self.z(ti, qi)
            # ui = self.u
            # zi, ui = self.model.pre_iteration_update(ti, zi, ui)
            # # qi = zi[self.fDOF_q]
            # # self.x[i, : self.nq] = qi

            # compute initial residual
            generator = self.__eval__(self.load_steps[i], self.x[i])
            R = next(generator)

            error = np.linalg.norm(R)
            converged = error < self.atol[0]
            # if i > 0:
            #     converged, error, sc = is_converged(
            #         R, self.x[i - 1], self.x[i], self.atol, self.rtol
            #     )
            # else:
            #     converged, error, sc = is_converged(
            #         R, self.x[i], self.x[i], self.atol, self.rtol
            #     )

            # reset counter and print inital status
            k = 0
            if self.verbose:
                pbar.set_description(
                    f" force iter {i+1:>{len_t}d}/{self.nt};"
                    f" Newton steps {k:>{len_maxIter}d}/{self.max_iter};"
                    f" error {error:.4e}/{self.atol[0]:.2e}"
                    # f" error {error:.4e}/{sc:.2e}"
                    # f" error {error:.4e}/{self.atol:.2e}"
                )

            # perform netwon step if necessary
            if not converged:
                while k <= self.max_iter:
                    # compute jacobian
                    dR = next(generator)

                    # solve linear system of equations
                    update = spsolve(dR, R)

                    # perform update
                    self.x[i] -= update

                    # compute new residual
                    # self.model.pre_iteration_update(self.load_steps[i], self.x[i, :self.nq], self.u)
                    generator = self.__eval__(self.load_steps[i], self.x[i])
                    R = next(generator)

                    error = np.linalg.norm(R)
                    converged = error < self.atol[0]
                    # if i > 0:
                    #     converged, error, sc = is_converged(
                    #         R, self.x[i - 1], self.x[i], self.atol, self.rtol
                    #     )
                    # else:
                    #     converged, error, sc = is_converged(
                    #         R, self.x[i], self.x[i], self.atol, self.rtol
                    #     )
                    # print(f"R: {R}")
                    # print(f"error: {error}")
                    # print(f"")

                    # update counter and print status
                    k += 1
                    if self.verbose:
                        pbar.set_description(
                            f" force iter {i+1:>{len_t}d}/{self.nt};"
                            f" Newton steps {k:>{len_maxIter}d}/{self.max_iter};"
                            f" error {error:.4e}/{self.atol[0]:.2e}"
                            # f" error {error:.4e}/{sc:.2e}"
                            # f" error {error:.4e}/{self.atol:.2e}"
                        )

                    # check convergence
                    if converged:
                        break

            if k > self.max_iter:
                # return solution up to this iteration
                if self.verbose:
                    pbar.close()
                print(
                    f"Newton-Raphson method not converged, returning solution "
                    f"up to iteration {i+1:>{len_t}d}/{self.nt}"
                )
                z = np.array(
                    [
                        self.z(self.load_steps[j], self.x[j, : self.nq])
                        for j in range(i + 1)
                    ]
                )
                return Solution(
                    t=self.load_steps,
                    q=z,
                    la_g=self.x[: i + 1, self.nq :],
                    la_N=self.x[: i + 1, self.nq + self.nla_g :],
                )

            # step callback and warm start for next step
            if i < self.nt - 1:
                # solver step callback
                ti = self.load_steps[i]
                qi = self.x[i, : self.nq]
                zi = self.z(ti, qi)
                ui = self.u

                zi, ui = self.model.step_callback(ti, zi, ui)
                self.x[i, : self.nq] = zi[self.fDOF_q]

                # store solution as new initial guess
                self.x[i + 1] = self.x[i]

        # return solution object
        if self.verbose:
            pbar.close()
        z = np.array(
            [self.z(self.load_steps[j], self.x[j, : self.nq]) for j in range(i + 1)]
        )
        return Solution(
            t=self.load_steps,
            q=z,
            la_g=self.x[: i + 1, self.nq :],
            la_N=self.x[: i + 1, self.nq + self.nla_g :],
        )
