import numpy as np
import matplotlib.pyplot as plt


class Solution:
    """Class to store solver outputs."""

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


def approx_fprime(x0, f, eps=1.0e-6, method="2-point"):
    """Approximate derivatives using finite differences."""
    x0 = np.atleast_1d(x0)
    f0 = np.atleast_1d(f(x0))

    # reshape possible mutidimensional arguments to 1D arrays and wrap f
    # accordingly
    x_shape = x0.shape
    xx = x0.reshape(-1)
    ff = lambda x: f(x.reshape(x_shape))
    m = len(xx)

    f_shape = f0.shape
    grad = np.empty(f_shape + (m,))

    h = np.diag(eps * np.ones_like(x0))
    for i in range(m):
        if method == "2-point":
            x = x0 + h[i]
            dx = x[i] - x0[i]  # Recompute dx as exactly representable number.
            df = ff(x) - f0
        elif method == "3-point":
            x1 = x0 + h[i]
            x2 = x0 - h[i]
            dx = x2[i] - x1[i]  # Recompute dx as exactly representable number.
            df = ff(x2) - ff(x1)
        elif method == "cs":
            f1 = ff(x0 + h[i] * 1.0j)
            df = f1.imag
            dx = h[i]
        else:
            raise RuntimeError('method "{method}" is not implemented!')

        grad[..., i] = df / dx

    return np.squeeze(grad.reshape(f_shape + x_shape))


def prox_R0np(x):
    """Proximal point to the set R_0^{n+}."""
    return np.maximum(x, 0)


class BouncingBall:
    """Class defining the bouncing ball system."""

    def __init__(self, m, gravity, radius, e_N, prox_r, q0, u0):
        self.m = m
        self.gravity = gravity
        self.e_N = e_N
        self.radius = radius
        self.prox_r = prox_r

        self.q0 = q0
        self.u0 = u0

    def q_dot(self, t, q, u):
        return u

    def M(self, t, q):
        return np.array([[self.m]], dtype=float)

    def h(self, t, q, u):
        return np.array([-self.m * self.gravity], dtype=float)

    def g_N(self, t, q):
        return q - self.radius

    def g_N_dot(self, t, q, u):
        return u

    def g_N_ddot(self, t, q, u, a):
        return a

    def xi_N(self, t, q, u_pre, u_post):
        return u_post + self.e_N * u_pre

    def W_N(self, t, q):
        # return np.ones(1, dtype=float)
        return np.array([[1.0]], dtype=float)


class GenAlpha:
    """Generalized alpha solver for one dimensional bouncing ball example."""

    def __init__(self, t0, t1, dt, model, max_iter=20, atol=1.0e-8):
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.dt2 = dt**2
        self.model = model
        self.max_iter = max_iter
        self.atol = atol

        self.qk = model.q0
        self.uk = model.u0

        # solve for initial accelerations
        Mk = model.M(self.t0, self.qk)
        hk = model.h(self.t0, self.qk, self.uk)
        ak = np.linalg.solve(Mk, hk)

        # set all other quantities to zero since we assume to start with no contact!
        Uk = np.zeros(1, dtype=float)
        Qk = np.zeros(1, dtype=float)
        lak = np.zeros(1, dtype=float)
        Lak = np.zeros(1, dtype=float)
        kappak = np.zeros(1, dtype=float)

        # initial vector of unknowns
        self.xk = np.concatenate(
            [
                ak,
                Uk,
                Qk,
                kappak,
                Lak,
                lak,
            ]
        )

        # initialize inactive contact sets since we assume to start with no contact!
        self.Ak1 = np.zeros(1, dtype=bool)
        self.Bk1 = np.zeros(1, dtype=bool)
        self.Ck1 = np.zeros(1, dtype=bool)

    def unpack(self, xk1):
        """Unpack vector of unknowns."""
        ak1, Uk1, Qk1, kappak1, Lak1, lak1 = xk1
        return (
            np.array([ak1]),
            np.array([Uk1]),
            np.array([Qk1]),
            np.array([kappak1]),
            np.array([Lak1]),
            np.array([lak1]),
        )

    def update(self, xk1):
        """Compute dependent variables."""
        ak1, Uk1, Qk1, kappak1, Lak1, lak1 = self.unpack(xk1)
        uk1 = self.uk + self.dt * ak1 + Uk1
        # qk1 = self.qk + self.dt * self.uk + 0.5 * self.dt2 * ak1 + Qk1
        qk1 = (
            self.qk + self.dt * self.uk + 0.5 * self.dt2 * ak1 + Qk1 + self.dt * Uk1
        )  # TODO: Is this more straight forward?
        Pk1 = Lak1 + self.dt * lak1
        kappa_hatk1 = kappak1 + 0.5 * self.dt2 * lak1
        # kappa_hatk1 = kappak1 + self.dt * Lak1 + 0.5 * self.dt2 * lak1 # TODO: Giuseppe?
        return qk1, uk1, Pk1, kappa_hatk1

    def residual(self, tk1, xk1, update_index_set=False):
        ###########################
        # unpack vector of unknowns
        ###########################
        ak1, Uk1, Qk1, kappak1, Lak1, lak1 = self.unpack(xk1)

        #############################
        # compute dependent variables
        #############################
        qk1, uk1, Pk1, kappa_hatk1 = self.update(xk1)

        ##############################
        # evaluate required quantities
        ##############################
        Mk1 = self.model.M(tk1, qk1)
        hk1 = self.model.h(tk1, qk1, uk1)
        g_Nk1 = self.model.g_N(tk1, qk1)
        g_N_ddotk1 = self.model.g_N_ddot(tk1, qk1, uk1, ak1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        W_Nk1 = self.model.W_N(tk1, qk1)

        #########################
        # compute residual vector
        #########################
        R = np.empty(6, dtype=float)

        # equations of motion
        R[0] = Mk1 @ ak1 - hk1 - W_Nk1 @ lak1

        # impact equation
        R[1] = Mk1 @ Uk1 - W_Nk1 @ Lak1
        # R[1] = Mk1 @ Uk1 - W_Nk1 @ Pk1

        # position correction
        # R[2] = Mk1 @ Qk1 - W_Nk1 @ kappak1
        R[2] = (
            Mk1 @ (Qk1 + self.dt * Uk1) - W_Nk1 @ kappak1
        )  # TODO: Only use this together with modified update above!
        # R[2] = Mk1 @ (Qk1 + self.dt * Uk1) - W_Nk1 @ kappa_hatk1

        # prox_arg_position = g_Nk1 - self.model.prox_r * kappa_hatk1
        # prox_arg_position = g_Nk1 - self.model.prox_r * kappak1
        prox_arg_position = g_Nk1 - self.model.prox_r * (
            kappak1 + self.dt * Lak1 + 0.5 * self.dt2 * lak1
        )
        prox_arg_velocity = xi_Nk1 - self.model.prox_r * Pk1
        prox_arg_acceleration = g_N_ddotk1 - self.model.prox_r * lak1
        if update_index_set:
            self.Ak1 = prox_arg_position <= 0
            self.Bk1 = self.Ak1 * (prox_arg_velocity <= 0)
            self.Ck1 = self.Bk1 * (prox_arg_acceleration <= 0)

        # complementarity on position level
        # R[3] = g_Nk1 - prox_R0np(prox_arg_position)
        if self.Ak1:
            R[3] = g_Nk1
        else:
            # R[3] = kappa_hatk1
            R[3] = kappak1
            # R[3] = kappak1 + self.dt * Lak1 + 0.5 * self.dt2 * lak1

        # complementarity on velocity level
        # if self.Ak1:
        #     R[4] = xi_Nk1 - prox_R0p(prox_arg_velocity)
        # else:
        #     R[4] = Pk1
        if self.Bk1:
            R[4] = xi_Nk1
        else:
            # R[4] = Pk1
            # R[4] = Lak1 + self.dt * lak1
            R[4] = Lak1  # + self.dt * lak1

        # complementarity on acceleration level
        # if self.Bk1:
        #     R[5] = g_N_ddotk1 - prox_R0p(g_N_ddotk1 - self.model.prox_r * lak1)
        # else:
        #     R[5] = lak1
        if self.Ck1:
            R[5] = g_N_ddotk1
        else:
            R[5] = lak1

        return R

    def Jacobian(self, tk1, xk1):
        return approx_fprime(xk1, lambda x: self.residual(tk1, x), method="2-point")

    def solve(self):
        a = []
        Q = []
        U = []
        la = []
        P = []
        kappa = []
        kappa_hat = []
        u = []
        q = []
        La = []

        t = np.arange(self.t0, self.t1 + self.dt, self.dt)
        xk1 = self.xk.copy()

        for k, tk1 in enumerate(t):
            print(f"k: {k}; tk1: {tk1:2.3f}")

            # initial residual and error; update active contact set during each
            # redidual computation
            R = self.residual(tk1, xk1, update_index_set=True)
            error = np.max(np.abs(R))

            i = 0
            if error < self.atol:
                ak1, Uk1, Qk1, kappak1, Lak1, lak1 = self.unpack(xk1)
                qk1, uk1, Pk1, kappa_hatk1 = self.update(xk1)
                self.qk = qk1.copy()
                self.uk = uk1.copy()
                a.append(ak1)
                U.append(Uk1)
                Q.append(Qk1)
                la.append(lak1)
                P.append(Pk1)
                kappa.append(kappak1)
                kappa_hat.append(kappa_hatk1)
                u.append(uk1)
                q.append(qk1)
                La.append(Lak1)
            else:
                # Newton-Raphson loop
                for i in range(self.max_iter):
                    # compute Jacobian matrix with same index set
                    J = self.Jacobian(tk1, xk1)

                    # compute updated state
                    xk1 -= np.linalg.solve(J, R)

                    # new residual and error; update active contact set during
                    # each redidual computation
                    R = self.residual(tk1, xk1, update_index_set=True)
                    error = np.max(np.abs(R))

                    if error < self.atol:
                        ak1, Uk1, Qk1, kappak1, Lak1, lak1 = self.unpack(xk1)
                        qk1, uk1, Pk1, kappa_hatk1 = self.update(xk1)
                        self.qk = qk1.copy()
                        self.uk = uk1.copy()
                        a.append(ak1)
                        U.append(Uk1)
                        Q.append(Qk1)
                        la.append(lak1)
                        P.append(Pk1)
                        kappa.append(kappak1)
                        kappa_hat.append(kappa_hatk1)
                        u.append(uk1)
                        q.append(qk1)
                        La.append(Lak1)
                        break
            if i >= self.max_iter - 1:
                print(
                    f"Newton-Raphson not converged after {i+1} steps with error {error:2.4f}"
                )
                n = len(q)
                return Solution(
                    t=t[:n],
                    a=np.array(a),
                    U=np.array(U),
                    Q=np.array(Q),
                    la=np.array(la),
                    P=np.array(P),
                    kappa=np.array(kappa),
                    kappa_hat=np.array(kappa_hat),
                    u=np.array(u),
                    q=np.array(q),
                    La=np.array(La),
                )
            else:
                print(
                    f"Newton-Raphson converged after {i+1} steps with error {error:2.4f}"
                )

        n = len(q)
        return Solution(
            t=t[:n],
            a=np.array(a),
            U=np.array(U),
            Q=np.array(Q),
            la=np.array(la),
            P=np.array(P),
            kappa=np.array(kappa),
            kappa_hat=np.array(kappa_hat),
            u=np.array(u),
            q=np.array(q),
            La=np.array(La),
        )


class GenAlpha2:
    """Generalized alpha solver for one dimensional bouncing ball example."""

    def __init__(self, t0, t1, dt, model, max_iter=20, atol=1.0e-8):
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.dt2 = dt**2
        self.model = model
        self.max_iter = max_iter
        self.atol = atol

        self.qk = model.q0
        self.uk = model.u0

        # solve for initial accelerations
        Mk = model.M(self.t0, self.qk)
        hk = model.h(self.t0, self.qk, self.uk)
        ak = np.linalg.solve(Mk, hk)

        # set all other quantities to zero since we assume to start with no contact!
        Uk = np.zeros(1, dtype=float)
        Qk = np.zeros(1, dtype=float)
        lak = np.zeros(1, dtype=float)
        Lak = np.zeros(1, dtype=float)
        kappak = np.zeros(1, dtype=float)

        # initial vector of unknowns
        self.xk = np.concatenate(
            [
                ak,
                Uk,
                Qk,
                kappak,
                Lak,
                lak,
            ]
        )

        # initialize inactive contact sets since we assume to start with no contact!
        self.Ak1 = np.zeros(1, dtype=bool)
        self.Bk1 = np.zeros(1, dtype=bool)
        self.Ck1 = np.zeros(1, dtype=bool)

    def unpack(self, xk1):
        """Unpack vector of unknowns."""
        ak1, Uk1, Qk1, kappak1, Lak1, lak1 = xk1
        return (
            np.atleast_1d(ak1),
            np.atleast_1d(Uk1),
            np.atleast_1d(Qk1),
            np.atleast_1d(kappak1),
            np.atleast_1d(Lak1),
            np.atleast_1d(lak1),
        )

    def update(self, xk1):
        """Compute dependent variables."""
        ak1, Uk1, Qk1, kappak1, Lak1, lak1 = self.unpack(xk1)
        uk1 = self.uk + self.dt * ak1 + Uk1
        qk1 = self.qk + self.dt * self.uk + 0.5 * self.dt2 * ak1 + Qk1 + self.dt * Uk1
        # qk1 = self.qk + self.dt * self.uk + 0.5 * self.dt2 * ak1 + Qk1
        Pk1 = Lak1 + self.dt * lak1
        kappa_hatk1 = kappak1 + self.dt * Lak1 + 0.5 * self.dt2 * lak1
        return qk1, uk1, Pk1, kappa_hatk1

    def residual(self, tk1, xk1, update_index_set=False):
        ###########################
        # unpack vector of unknowns
        ###########################
        ak1, Uk1, Qk1, kappak1, Lak1, lak1 = self.unpack(xk1)

        #############################
        # compute dependent variables
        #############################
        qk1, uk1, Pk1, kappa_hatk1 = self.update(xk1)

        ##############################
        # evaluate required quantities
        ##############################
        Mk1 = self.model.M(tk1, qk1)
        hk1 = self.model.h(tk1, qk1, uk1)
        g_Nk1 = self.model.g_N(tk1, qk1)
        g_N_ddotk1 = self.model.g_N_ddot(tk1, qk1, uk1, ak1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        W_Nk1 = self.model.W_N(tk1, qk1)

        #########################
        # compute residual vector
        #########################
        R = np.empty(6, dtype=float)

        # equations of motion
        R[0] = Mk1 @ ak1 - hk1 - W_Nk1 @ lak1

        # impact equation
        R[1] = Mk1 @ Uk1 - W_Nk1 @ Lak1

        # position correction
        # R[2] = Mk1 @ Qk1 - W_Nk1 @ kappak1
        R[2] = Mk1 @ (Qk1 + dt * Uk1) - W_Nk1 @ kappak1

        prox_arg_position = g_Nk1 - self.model.prox_r * kappa_hatk1
        prox_arg_velocity = xi_Nk1 - self.model.prox_r * Pk1
        prox_arg_acceleration = g_N_ddotk1 - self.model.prox_r * lak1
        # prox_arg_position = g_Nk1 - self.model.prox_r * kappak1
        # prox_arg_velocity = xi_Nk1 - self.model.prox_r * Lak1
        # prox_arg_acceleration = g_N_ddotk1 - self.model.prox_r * lak1
        if update_index_set:
            self.Ak1 = prox_arg_position <= 0
            self.Bk1 = self.Ak1 * (prox_arg_velocity <= 0)
            self.Ck1 = self.Bk1 * (prox_arg_acceleration <= 0)

        # complementarity on position level
        R[3] = g_Nk1 - prox_R0np(prox_arg_position)
        # if self.Ak1:
        #     R[3] = g_Nk1
        # else:
        #     R[3] = kappak1
        #     # R[3] = kappa_hatk1

        # complementarity on velocity level
        if self.Ak1:
            R[4] = xi_Nk1 - prox_R0np(prox_arg_velocity)
        else:
            R[4] = Pk1
            # R[4] = Lak1
        # if self.Bk1:
        #     R[4] = xi_Nk1
        # else:
        #     R[4] = Lak1

        # complementarity on acceleration level
        if self.Bk1:
            R[5] = g_N_ddotk1 - prox_R0np(prox_arg_acceleration)
        else:
            R[5] = lak1
        # if self.Ck1:
        #     R[5] = g_N_ddotk1
        # else:
        #     R[5] = lak1

        return R

    def Jacobian(self, tk1, xk1):
        return approx_fprime(xk1, lambda x: self.residual(tk1, x), method="2-point")

    def solve(self):
        a = []
        Q = []
        U = []
        la = []
        P = []
        kappa = []
        kappa_hat = []
        u = []
        q = []
        La = []

        t = np.arange(self.t0, self.t1 + self.dt, self.dt)
        xk1 = self.xk.copy()

        for k, tk1 in enumerate(t):
            print(f"k: {k}; tk1: {tk1:2.3f}")

            # initial residual and error; update active contact set during each
            # redidual computation
            R = self.residual(tk1, xk1, update_index_set=True)
            error = np.max(np.abs(R))

            i = 0
            if error < self.atol:
                ak1, Uk1, Qk1, kappak1, Lak1, lak1 = self.unpack(xk1)
                qk1, uk1, Pk1, kappa_hatk1 = self.update(xk1)
                self.qk = qk1.copy()
                self.uk = uk1.copy()
                a.append(ak1)
                U.append(Uk1)
                Q.append(Qk1)
                la.append(lak1)
                P.append(Pk1)
                kappa.append(kappak1)
                kappa_hat.append(kappa_hatk1)
                u.append(uk1)
                q.append(qk1)
                La.append(Lak1)
            else:
                # Newton-Raphson loop
                for i in range(self.max_iter):
                    # compute Jacobian matrix with same index set
                    J = self.Jacobian(tk1, xk1)

                    # compute updated state
                    xk1 -= np.linalg.solve(J, R)

                    # new residual and error; update active contact set during
                    # each redidual computation
                    R = self.residual(tk1, xk1, update_index_set=True)
                    error = np.max(np.abs(R))

                    if error < self.atol:
                        ak1, Uk1, Qk1, kappak1, Lak1, lak1 = self.unpack(xk1)
                        qk1, uk1, Pk1, kappa_hatk1 = self.update(xk1)
                        self.qk = qk1.copy()
                        self.uk = uk1.copy()
                        a.append(ak1)
                        U.append(Uk1)
                        Q.append(Qk1)
                        la.append(lak1)
                        P.append(Pk1)
                        kappa.append(kappak1)
                        kappa_hat.append(kappa_hatk1)
                        u.append(uk1)
                        q.append(qk1)
                        La.append(Lak1)
                        break
            if i >= self.max_iter - 1:
                print(
                    f"Newton-Raphson not converged after {i+1} steps with error {error:2.4f}"
                )
                n = len(q)
                return Solution(
                    t=t[:n],
                    a=np.array(a),
                    U=np.array(U),
                    Q=np.array(Q),
                    la=np.array(la),
                    P=np.array(P),
                    kappa=np.array(kappa),
                    kappa_hat=np.array(kappa_hat),
                    u=np.array(u),
                    q=np.array(q),
                    La=np.array(La),
                )
            else:
                print(
                    f"Newton-Raphson converged after {i+1} steps with error {error:2.4f}"
                )

        n = len(q)
        return Solution(
            t=t[:n],
            a=np.array(a),
            U=np.array(U),
            Q=np.array(Q),
            la=np.array(la),
            P=np.array(P),
            kappa=np.array(kappa),
            kappa_hat=np.array(kappa_hat),
            u=np.array(u),
            q=np.array(q),
            La=np.array(La),
        )


class GenAlpha3:
    """Generalized alpha solver for one dimensional bouncing ball example."""

    def __init__(self, t0, t1, dt, model, max_iter=20, atol=1.0e-8):
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.dt2 = dt**2
        self.model = model
        self.max_iter = max_iter
        self.atol = atol

        self.qk = model.q0
        self.uk = model.u0

        # solve for initial q_dots and u_dots
        vk = model.q_dot(self.t0, self.qk, self.uk)
        Mk = model.M(self.t0, self.qk)
        hk = model.h(self.t0, self.qk, self.uk)
        ak = np.linalg.solve(Mk, hk)

        # set all other quantities to zero since we assume to start with no contact!
        Uk = np.zeros(1, dtype=float)
        lak = np.zeros(1, dtype=float)
        Lak = np.zeros(1, dtype=float)
        kappak = np.zeros(1, dtype=float)

        # initial vector of unknowns
        self.xk = np.concatenate(
            [
                vk,
                ak,
                Uk,
                kappak,
                Lak,
                lak,
            ]
        )

        # initialize inactive contact sets since we assume to start with no contact!
        self.Ak1 = np.zeros(1, dtype=bool)
        self.Bk1 = np.zeros(1, dtype=bool)
        self.Ck1 = np.zeros(1, dtype=bool)

    def unpack(self, xk1):
        """Unpack vector of unknowns."""
        vk1, ak1, Uk1, kappak1, Lak1, lak1 = xk1
        return (
            np.array([vk1]),
            np.array([ak1]),
            np.array([Uk1]),
            np.array([kappak1]),
            np.array([Lak1]),
            np.array([lak1]),
        )

    def update(self, xk1):
        """Compute dependent variables."""
        vk1, ak1, Uk1, kappak1, Lak1, lak1 = self.unpack(xk1)
        uk1 = self.uk + self.dt * ak1 + Uk1
        qk1 = self.qk + self.dt * vk1
        Pk1 = Lak1 + self.dt * lak1
        kappa_hatk1 = kappak1 + 0.5 * self.dt2 * lak1
        return qk1, uk1, Pk1, kappa_hatk1

    def residual(self, tk1, xk1, update_index_set=False):
        ###########################
        # unpack vector of unknowns
        ###########################
        vk1, ak1, Uk1, kappak1, Lak1, lak1 = self.unpack(xk1)

        #############################
        # compute dependent variables
        #############################
        qk1, uk1, Pk1, kappa_hatk1 = self.update(xk1)

        ##############################
        # evaluate required quantities
        ##############################
        q_dotk1 = self.model.q_dot(tk1, qk1, uk1)
        Mk1 = self.model.M(tk1, qk1)
        hk1 = self.model.h(tk1, qk1, uk1)
        g_Nk1 = self.model.g_N(tk1, qk1)
        g_N_ddotk1 = self.model.g_N_ddot(tk1, qk1, uk1, ak1)
        xi_Nk1 = self.model.xi_N(tk1, qk1, self.uk, uk1)
        W_Nk1 = self.model.W_N(tk1, qk1)

        #########################
        # compute residual vector
        #########################
        R = np.empty(6, dtype=float)

        # kinematic equation
        # TODO: This should be g_q.T instead of W_Nk1!
        R[0] = vk1 - q_dotk1 - W_Nk1.T @ kappak1
        # equations of motion
        R[1] = Mk1 @ ak1 - hk1 - W_Nk1 @ lak1

        # impact equation
        R[2] = Mk1 @ Uk1 - W_Nk1 @ Lak1

        prox_arg_position = g_Nk1 - self.model.prox_r * kappa_hatk1
        # prox_arg_position = g_Nk1 - self.model.prox_r * kappak1
        prox_arg_velocity = xi_Nk1 - self.model.prox_r * Pk1
        prox_arg_acceleration = g_N_ddotk1 - self.model.prox_r * lak1
        if update_index_set:
            self.Ak1 = prox_arg_position <= 0
            self.Bk1 = self.Ak1 * (prox_arg_velocity <= 0)
            self.Ck1 = self.Bk1 * (prox_arg_acceleration <= 0)

        # complementarity on position level
        # R[3] = g_Nk1 - prox_R0p(prox_arg_position)
        if self.Ak1:
            R[3] = g_Nk1
        else:
            R[3] = kappa_hatk1
            # R[3] = kappak1

        # complementarity on velocity level
        # if self.Ak1:
        #     R[4] = xi_Nk1 - prox_R0p(prox_arg_velocity)
        # else:
        #     R[4] = Pk1
        if self.Bk1:
            R[4] = xi_Nk1
        else:
            R[4] = Pk1

        # complementarity on acceleration level
        # if self.Bk1:
        #     R[5] = g_N_ddotk1 - prox_R0p(g_N_ddotk1 - self.model.prox_r * lak1)
        # else:
        #     R[5] = lak1
        if self.Ck1:
            R[5] = g_N_ddotk1
        else:
            R[5] = lak1

        return R

    def Jacobian(self, tk1, xk1):
        return approx_fprime(xk1, lambda x: self.residual(tk1, x), method="2-point")

    def solve(self):
        v = []
        a = []
        U = []
        la = []
        P = []
        kappa = []
        kappa_hat = []
        u = []
        q = []
        La = []

        t = np.arange(self.t0, self.t1 + self.dt, self.dt)
        xk1 = self.xk.copy()

        for k, tk1 in enumerate(t):
            print(f"k: {k}; tk1: {tk1:2.3f}")

            # initial residual and error; update active contact set during each
            # redidual computation
            R = self.residual(tk1, xk1, update_index_set=True)
            error = np.max(np.abs(R))

            i = 0
            if error < self.atol:
                vk1, ak1, Uk1, kappak1, Lak1, lak1 = self.unpack(xk1)
                qk1, uk1, Pk1, kappa_hatk1 = self.update(xk1)
                self.qk = qk1.copy()
                self.uk = uk1.copy()
                v.append(vk1)
                a.append(ak1)
                U.append(Uk1)
                la.append(lak1)
                P.append(Pk1)
                kappa.append(kappak1)
                kappa_hat.append(kappa_hatk1)
                u.append(uk1)
                q.append(qk1)
                La.append(Lak1)
            else:
                # Newton-Raphson loop
                for i in range(self.max_iter):
                    # compute Jacobian matrix with same index set
                    J = self.Jacobian(tk1, xk1)

                    # compute updated state
                    xk1 -= np.linalg.solve(J, R)

                    # new residual and error; update active contact set during
                    # each redidual computation
                    R = self.residual(tk1, xk1, update_index_set=True)
                    error = np.max(np.abs(R))

                    if error < self.atol:
                        vk1, ak1, Uk1, kappak1, Lak1, lak1 = self.unpack(xk1)
                        qk1, uk1, Pk1, kappa_hatk1 = self.update(xk1)
                        self.qk = qk1.copy()
                        self.uk = uk1.copy()
                        v.append(vk1)
                        a.append(ak1)
                        U.append(Uk1)
                        la.append(lak1)
                        P.append(Pk1)
                        kappa.append(kappak1)
                        kappa_hat.append(kappa_hatk1)
                        u.append(uk1)
                        q.append(qk1)
                        La.append(Lak1)
                        break
            if i >= self.max_iter - 1:
                print(
                    f"Newton-Raphson not converged after {i+1} steps with error {error:2.4f}"
                )
                n = len(q)
                return Solution(
                    t=t[:n],
                    v=np.array(v),
                    a=np.array(a),
                    U=np.array(U),
                    la=np.array(la),
                    P=np.array(P),
                    kappa=np.array(kappa),
                    kappa_hat=np.array(kappa_hat),
                    u=np.array(u),
                    q=np.array(q),
                    La=np.array(La),
                )
            else:
                print(
                    f"Newton-Raphson converged after {i+1} steps with error {error:2.4f}"
                )

        n = len(q)
        return Solution(
            t=t[:n],
            v=np.array(v),
            a=np.array(a),
            U=np.array(U),
            la=np.array(la),
            P=np.array(P),
            kappa=np.array(kappa),
            kappa_hat=np.array(kappa_hat),
            u=np.array(u),
            q=np.array(q),
            La=np.array(La),
        )


if __name__ == "__main__":

    # create bouncing ball object
    m = 1
    gravity = 10
    radius = 0.1
    e_N = 0.5
    prox_r = 0.5
    q0 = np.array([1], dtype=float)
    u0 = np.array([0], dtype=float)
    ball = BouncingBall(m, gravity, radius, e_N, prox_r, q0, u0)

    # create solver and solve the nonsmooth problem
    t0 = 0
    t1 = 1.5
    dt = 1.0e-2
    # sol = GenAlpha(t0, t1, dt, ball).solve()
    sol = GenAlpha2(t0, t1, dt, ball).solve()
    # sol = GenAlpha3(t0, t1, dt, ball).solve()

    fig, ax = plt.subplots(3, 3)

    ax[0, 0].set_title("q")
    ax[0, 0].plot(sol.t, sol.q)
    ax[0, 0].plot([t0, t1], [0.0, 0.0], "--k")
    ax[0, 0].plot([t0, t1], [1.0, 1.0], "--k")

    ax[1, 0].set_title("u")
    ax[1, 0].plot(sol.t, sol.u)
    ax[1, 0].plot([t0, t1], [0.0, 0.0], "--k")

    ax[2, 0].set_title("a")
    ax[2, 0].plot(sol.t, sol.a)
    ax[2, 0].plot([t0, t1], [0.0, 0.0], "--k")

    ax[0, 1].set_title("Q")
    ax[0, 1].plot(sol.t, sol.Q)
    ax[0, 1].plot([t0, t1], [0.0, 0.0], "--k")

    ax[1, 1].set_title("U")
    ax[1, 1].plot(sol.t, sol.U)
    ax[1, 1].plot([t0, t1], [0.0, 0.0], "--k")

    ax[2, 1].set_title("kappa")
    ax[2, 1].plot(sol.t, sol.kappa)
    ax[2, 1].plot([t0, t1], [0.0, 0.0], "--k")

    ax[0, 2].set_title("la")
    ax[0, 2].plot(sol.t, sol.la)
    ax[0, 2].plot([t0, t1], [0.0, 0.0], "--k")

    ax[1, 2].set_title("La")
    ax[1, 2].plot(sol.t, sol.La)
    ax[1, 2].plot([t0, t1], [0.0, 0.0], "--k")

    ax[2, 2].set_title("P")
    ax[2, 2].plot(sol.t, sol.P)
    ax[2, 2].plot([t0, t1], [0.0, 0.0], "--k")

    plt.tight_layout()
    plt.show()

    print(f"")
