from cardillo.math import approx_fprime
from cardillo.solver.Solution import Solution

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, bmat, lil_matrix
from tqdm import tqdm


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
        cDOF=np.array([], dtype=int),
        b=lambda t: np.array([]),
        n_load_steps=1,
        load_steps=None,
        atol=1e-8,
        max_iter=50,
        prox_r_N=1.0e-2,
        numerical_jacobian=False,
        verbose=True,
        newton_error_function=lambda x: np.max(np.abs(x)),
        numdiff_method="2-point",
        numdiff_eps=1.0e-6,
    ):
        self.model = model

        # handle constraint degrees of freedoms
        z0 = model.q0.copy()
        self.nz = len(z0)
        nc = len(cDOF)
        self.nq = self.nz - nc
        self.cDOF = cDOF
        self.zDOF = np.arange(self.nz)
        self.fDOF = np.setdiff1d(self.zDOF, cDOF)
        q0 = z0[self.fDOF]

        # print(f"zDOF: {self.zDOF}")
        # print(f"cDOF: {cDOF}")
        # print(f"fDOF: {self.fDOF}")

        if callable(b):
            self.b = b
        else:
            self.b = lambda t: b

        self.max_iter = max_iter
        self.atol = atol
        self.prox_r_N = prox_r_N

        if load_steps is None:
            self.load_steps = np.linspace(0, 1, n_load_steps)
        else:
            self.load_steps = np.array(load_steps)

        # dimensions
        self.nt = len(self.load_steps)
        # self.nu = self.model.nu
        # TODO: How do we define the cDOF/ fDOF for the u's?
        self.nu = self.model.nu
        self.nla_g = self.model.nla_g
        self.nla_N = self.model.nla_N
        self.nx = self.nq + self.nla_g + self.nla_N

        # memory allocation
        self.x = np.zeros((self.nt, self.nx))

        # initial conditions
        self.x[0] = np.concatenate((q0, self.model.la_g0, self.model.la_N0))
        self.u = np.zeros(self.nu)  # zero velocities as system is static

        self.numdiff_method = numdiff_method
        self.numdiff_eps = numdiff_eps

        self.verbose = verbose
        self.newton_error_function = newton_error_function

        if numerical_jacobian:
            self.__eval__ = self.__eval__num
        else:
            self.__eval__ = self.__eval__analytic

    def z(self, t, q):
        z = np.zeros(self.nz)
        z[self.fDOF] = q
        z[self.cDOF] = self.b(t)
        return z

    def __eval__analytic(self, t, x):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        q = x[:nq]
        la_g = x[nq : nq + nla_g]
        la_N = x[nq + nla_g :]

        # compute redundant coordinates
        z = self.z(t, q)

        # evaluate quantites that are required for computing the residual and
        # the jacobian
        W_g = self.model.W_g(t, z, scipy_matrix=csr_matrix)[self.fDOF]
        W_N = self.model.W_N(t, z, scipy_matrix=csr_matrix)[self.fDOF]
        g_N = self.model.g_N(t, z)

        R = np.zeros(self.nx)
        R[:nu] = self.model.h(t, z, self.u)[self.fDOF] + W_g @ la_g + W_N @ la_N
        R[nu : nu + nla_g] = self.model.g(t, z)
        R[nq + nla_g :] = np.minimum(la_N, g_N)

        yield R

        # evaluate additionally required quantites for computing the jacobian
        K = (
            self.model.h_q(t, z, self.u, scipy_matrix=csr_matrix)[
                self.fDOF[:, None], self.fDOF
            ]
            + self.model.Wla_g_q(t, z, la_g, scipy_matrix=csr_matrix)[
                self.fDOF[:, None], self.fDOF
            ]
        )
        g_q = self.model.g_q(t, z, scipy_matrix=csr_matrix)[:, self.fDOF]
        # note: csr_matrix is best for row slicing, see
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
        g_N_q = self.model.g_N_q(t, z, scipy_matrix=csr_matrix)[:, self.fDOF]

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

            # compute initial residual
            # self.model.pre_iteration_update(self.load_steps[i],
            #                                 self.x[i, :self.nq],
            #                                 self.u)
            generator = self.__eval__(self.load_steps[i], self.x[i])
            R = next(generator)
            # R = self.residual(self.load_steps[i], self.x[i])
            error = self.newton_error_function(R)
            converged = error < self.atol

            # reset counter and print inital status
            k = 0
            if self.verbose:
                pbar.set_description(
                    f" force iter {i+1:>{len_t}d}/{self.nt};"
                    f" Newton steps {k:>{len_maxIter}d}/{self.max_iter};"
                    f" error {error:.4e}/{self.atol:.2e}"
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
                    error = self.newton_error_function(R)
                    converged = error < self.atol

                    # update counter and print status
                    k += 1
                    if self.verbose:
                        pbar.set_description(
                            f" force iter {i+1:>{len_t}d}/{self.nt};"
                            f" Newton steps {k:>{len_maxIter}d}/{self.max_iter};"
                            f" error {error:.4e}/{self.atol:.2e}"
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

            # store solution as new initial guess
            if i < self.nt - 1:
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