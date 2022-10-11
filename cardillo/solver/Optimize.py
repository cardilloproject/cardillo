from cardillo.math import approx_fprime
from cardillo.solver import Solution

import numpy as np
from tqdm import tqdm

from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS, SR1


class Optimize:
    """Force and displacement optimization problem that minimizes the potential energy of the system."""

    def __init__(
        self,
        model,
        cDOF=np.array([], dtype=int),
        b=lambda t: np.array([]),
        n_load_steps=1,
        load_steps=None,
        tol=1e-8,
        max_iter=50,
        numerical_jacobian=False,
        initial_guess=None,
        verbose=True,
        newton_error_function=lambda x: np.max(np.abs(x)),
        numdiff_method="2-point",
        numdiff_eps=1.0e-6,
    ):

        raise RuntimeError("Refactor and test me!")

        # handle constraint degrees of freedoms
        z0 = model.q0.copy()
        v0 = model.u0
        self.nz = len(z0)
        nc = len(cDOF)
        self.nq = self.nz - nc
        self.cDOF = cDOF
        self.zDOF = np.arange(self.nz)
        self.fDOF = np.setdiff1d(self.zDOF, cDOF)
        q0 = z0[self.fDOF]

        print(f"zDOF: {self.zDOF}")
        print(f"cDOF: {cDOF}")
        print(f"fDOF: {self.fDOF}")

        if callable(b):
            self.b = b
        else:
            self.b = lambda t: b

        self.max_iter = max_iter
        self.tol = tol
        self.model = model

        if load_steps is None:
            self.load_steps = np.linspace(0, 1, n_load_steps)
        else:
            self.load_steps = np.array(load_steps)

        # dimensions
        self.nt = len(self.load_steps)
        self.nu = self.model.nu
        self.nla = self.model.nla_g
        self.nx = self.nq + self.nla

        # memory allocation
        self.x = np.zeros((self.nt, self.nx))

        # initial conditions
        self.x[0] = np.concatenate((q0, self.model.la_g0))
        self.u = np.zeros(self.nu)  # zero velocities as system is static

        # if numerical_jacobian:
        #     self._jacobian = self.jacobian_num
        # else:
        #     self._jacobian = self.jacobian_an
        self.numdiff_method = numdiff_method
        self.numdiff_eps = numdiff_eps

        self.initial_guess = initial_guess
        self.verbose = verbose
        self.newton_error_function = newton_error_function

    def z(self, t, q):
        z = np.zeros(self.nz)
        z[self.fDOF] = q
        z[self.cDOF] = self.b(t)
        return z

    def solve(self):
        # compute numbe rof digits for status update
        len_t = len(str(self.nt))
        len_maxIter = len(str(self.max_iter))

        if self.verbose:
            pbar = tqdm(range(0, self.nt), leave=True)
        else:
            pbar = range(0, self.nt)
        for i in pbar:
            if self.verbose:
                pbar.update(1)

            # reset counter and print inital status
            k = 0
            if self.verbose:
                pbar.set_description(
                    f" force iter {i+1:>{len_t}d}/{self.nt};"
                    f" Newton steps {k:>{len_maxIter}d}/{self.max_iter};"
                    # f" error {error:.4e}/{self.tol:.2e}"
                )

            # current iteration
            ti = self.load_steps[i]
            qi = self.x[i, : self.nq]
            # la_gi = self.x[i, self.nq:]

            def g(q):
                z = self.z(ti, q)
                return self.model.g(ti, z)

            def g_q(q):
                z = self.z(ti, q)
                return self.model.g_q(ti, z)

            def E_pot(q):
                z = self.z(ti, q)
                return self.model.E_pot(ti, z)

            def E_pot_q(q):
                z = self.z(ti, q)
                return -self.model.f_pot(ti, z)[self.fDOF]

            # define constraint function
            if self.model.nla_g > 0:
                nonlinear_constraint = [NonlinearConstraint(g, 0, 0, g_q)]
            else:
                nonlinear_constraint = []
            res = minimize(
                E_pot,
                qi,
                # method="trust-constr",
                method="SLSQP",
                jac=E_pot_q,
                # jac="2-point",
                # hess=lambda q: -model.f_pot_q(t0, q),
                # hess=BFGS,
                # hess=SR1,
                constraints=nonlinear_constraint,
                options={"verbose": 1, "max_iter": self.max_iter},
            )

            self.x[i, : self.nq] = res.x

            # store solution as new initial guess
            if i < self.nt - 1:
                if self.initial_guess is None:
                    self.x[i + 1] = self.x[i]
                else:
                    self.x[i + 1] = self.initial_guess(
                        self.load_steps[i], self.load_steps[i + 1], self.x[i]
                    )

        # return solution object
        if self.verbose:
            pbar.close()
        z = np.array(
            [self.z(self.load_steps[j], self.x[j, : self.nq]) for j in range(i + 1)]
        )
        return Solution(t=self.load_steps, q=z, la_g=self.x[:, self.nq :])
