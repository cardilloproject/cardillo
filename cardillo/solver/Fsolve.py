from cardillo.solver.Solution import Solution

import numpy as np
from scipy.sparse import csr_matrix, bmat, lil_matrix
from tqdm import tqdm
from scipy.optimize import fsolve, root, least_squares


class Fsolve:
    def __init__(
        self,
        model,
        n_load_steps=1,
        load_steps=None,
        atol=1e-8,
        rtol=1e-6,
        prox_r_N=1.0e-2,
        numerical_jacobian=False,
        numdiff_method="2-point",
        numdiff_eps=1.0e-6,
    ):
        self.model = model

        q0 = model.q0.copy()
        u0 = model.u0.copy()

        self.prox_r_N = prox_r_N

        if load_steps is None:
            self.load_steps = np.linspace(0, 1, n_load_steps)
        else:
            self.load_steps = np.array(load_steps)

        # other dimensions
        self.nt = len(self.load_steps)
        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = self.model.nla_g
        self.nla_S = self.model.nla_S
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
        self.u = np.zeros(self.nu)  # zero velocities as system is static

        self.numdiff_method = numdiff_method
        self.numdiff_eps = numdiff_eps

    def residual(self, x, t):
        nq = self.nq
        nu = self.nu
        nla_g = self.nla_g
        nla_S = self.nla_S

        q = x[:nq]
        la_g = x[nq : nq + nla_g]
        la_N = x[nq + nla_g :]

        # evaluate quantites that are required for computing the residual and
        # the jacobian
        W_g = self.model.W_g(t, q, scipy_matrix=csr_matrix)
        W_N = self.model.W_N(t, q, scipy_matrix=csr_matrix)
        g_N = self.model.g_N(t, q)

        R = np.zeros(self.nf)
        R[:nu] = self.model.h(t, q, self.u) + W_g @ la_g + W_N @ la_N
        R[nu : nu + nla_g] = self.model.g(t, q)
        R[nu + nla_g : nu + nla_g + nla_S] = self.model.g_S(t, q)
        R[nu + nla_g + nla_S :] = np.minimum(la_N, g_N)

        print(f"error: {np.linalg.norm(R)}")

        return R

    def jacobian(self, x, t):
        nq = self.nq
        nla_g = self.nla_g

        q = x[:nq]
        la_g = x[nq : nq + nla_g]
        la_N = x[nq + nla_g :]

        # evaluate quantites that are required for computing the residual and
        # the jacobian
        W_g = self.model.W_g(t, q, scipy_matrix=csr_matrix)
        W_N = self.model.W_N(t, q, scipy_matrix=csr_matrix)
        g_N = self.model.g_N(t, q)

        # evaluate additionally required quantites for computing the jacobian
        K = self.model.h_q(t, q, self.u, scipy_matrix=csr_matrix) + self.model.Wla_g_q(
            t, q, la_g, scipy_matrix=csr_matrix
        )
        g_q = self.model.g_q(t, q, scipy_matrix=csr_matrix)
        g_S_q = self.model.g_S_q(t, q, scipy_matrix=csr_matrix)
        g_N_q = self.model.g_N_q(t, q, scipy_matrix=csr_matrix)

        Rla_N_q = lil_matrix((self.nla_N, self.nq))
        Rla_N_la_N = lil_matrix((self.nla_N, self.nla_N))
        for i in range(self.nla_N):
            if la_N[i] < g_N[i]:
                Rla_N_la_N[i, i] = 1.0
            else:
                Rla_N_q[i] = g_N_q[i]

        # fmt: off
        return bmat([
            [      K,  W_g,        W_N], 
            [    g_q, None,       None],
            [  g_S_q, None,       None],
            [Rla_N_q, None, Rla_N_la_N]
        ], format="csr").toarray()
        # fmt: on

    def solve(self):
        # compute number of digits for status update
        len_t = len(str(self.nt))

        pbar = tqdm(range(0, self.nt), leave=True)
        for i in pbar:

            # use fsolve here
            ti = self.load_steps[i]
            # self.x[i], info, converged, mesg = fsolve(self.residual, self.x[i], args=(ti,), full_output=1, xtol=1.0e-14)
            # self.x[i], info, converged, mesg = fsolve(self.residual, self.x[i], fprime=self.jacobian, args=(ti,), full_output=1, xtol=1.0e-20, col_deriv=False)

            # sol = root(self.residual, self.x[i], args=(ti,))
            # sol = root(
            #     self.residual,
            #     self.x[i],
            #     args=(ti,),
            #     method="hybr",
            #     # method="lm",
            #     # method="broyden1",
            #     # method="broyden2",
            #     # method="anderson",
            #     # method="Krylov",
            #     # method="diagbroyden",
            #     # method="broyden2",
            #     # method="excitingmixing",
            #     jac=self.jacobian
            # )

            sol = least_squares(
                self.residual,
                self.x[i],
                args=(ti,),
                jac=self.jacobian,
                # method="trf",
                method="dogbox",
                # method="lm",
            )

            self.x[i] = sol.x
            converged = sol.success
            # nfev = sol.nfev
            if converged:
                pbar.set_description(
                    f" force iter {i+1:>{len_t}d}/{self.nt};"
                    # f" nfeval {nfev};"
                )
            else:
                pbar.close()
                print(
                    f"Newton-Raphson method not converged, returning solution "
                    f"up to iteration {i+1:>{len_t}d}/{self.nt}"
                )
                return Solution(
                    t=self.load_steps,
                    q=self.x[: i + 1, : self.nq],
                    la_g=self.x[: i + 1, self.nq :],
                    la_N=self.x[: i + 1, self.nq + self.nla_g :],
                )

            # step callback and warm start for next step
            if i < self.nt - 1:
                # # solver step callback
                # # self.model.step_callback(self.load_steps[i], )
                # ti = self.load_steps[i]
                # qi = self.x[i, : self.nq]
                # ui = self.u

                # qi, ui = self.model.step_callback(ti, qi, ui)
                # self.x[i, : self.nq] = qi

                # store solution as new initial guess
                self.x[i + 1] = self.x[i]

        # return solution object
        pbar.close()
        return Solution(
            t=self.load_steps,
            q=self.x[: i + 1, : self.nq],
            la_g=self.x[: i + 1, self.nq : self.nq + self.nla_g :],
            la_N=self.x[: i + 1, self.nq + self.nla_g :],
        )
