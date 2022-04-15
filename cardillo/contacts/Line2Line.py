from math import sin, cos, sqrt
import numpy as np
from cardillo.math import approx_fprime, norm, A_IK_basic, cross3, e1, e2, e3
from scipy.optimize import fsolve
from scipy.optimize import minimize


class ContactPotential:
    def __init__(self, k):
        self.k = k

    @staticmethod
    def Macaulay(x):
        """Macaulay brackets, see https://en.wikipedia.org/wiki/Macaulay_brackets."""
        return min(0, x)

    def potential(self, g):
        return 0.5 * self.k * ContactPotential.Macaulay(g) ** 2

    def potential_g(self, g):
        return self.k * ContactPotential.Macaulay(g)

    def potential_gg(self, g):
        return self.k


class Line2Line:
    def __init__(self, eps, R1, R2, subsystem1, subsystem2, eta_c=0.5):

        # constact stiffness
        self.eps = eps

        # pulley radii
        self.R1 = R1
        self.R2 = R2

        # information of subsystem 1
        self.subsystem1 = subsystem1

        # information of subsystem 2
        self.subsystem2 = subsystem2

        # initial and current values for the closest point
        self.eta_c = eta_c
        self.alpha_c = 0
        self.beta_c = 0

        # contact potential
        self.contact_potential = ContactPotential(eps)

        # number of contact points
        self.n_contact_points = 2 * subsystem1.nelement * subsystem1.nquadrature

    def assembler_callback(self):
        qDOF_subsystems = np.concatenate([self.subsystem1.qDOF, self.subsystem2.qDOF])

        self.nq1 = len(self.subsystem1.qDOF)
        self.nq2 = len(self.subsystem2.qDOF)
        self.__nq = self.nq1 + self.nq2

        uDOF_subsystems = np.concatenate([self.subsystem1.uDOF, self.subsystem2.uDOF])

        self.nu1 = len(self.subsystem1.uDOF)
        self.nu2 = len(self.subsystem2.uDOF)
        self.__nu = self.nu1 + self.nu2

        self.qDOF = qDOF_subsystems
        self.uDOF = uDOF_subsystems

    def r1(self, t, q, xi):
        frame_ID = (xi,)
        qDOF_P = self.subsystem1.qDOF_P(frame_ID)
        q_subsystem1 = q[: self.nq1][qDOF_P]
        return self.subsystem1.r_OC(t, q_subsystem1, frame_ID)

    def r1_q(self, t, q, xi):
        frame_ID = (xi,)
        qDOF_P = self.subsystem1.qDOF_P(frame_ID)
        q_subsystem1 = q[: self.nq1][qDOF_P]
        return self.subsystem1.r_OC_q(t, q_subsystem1, frame_ID)

    def r1_xi(self, t, q, xi):
        frame_ID = (xi,)
        qDOF_P = self.subsystem1.qDOF_P(frame_ID)
        q_subsystem1 = q[: self.nq1][qDOF_P]
        return self.subsystem1.r_OC_xi(t, q_subsystem1, frame_ID)

    def r1_xixi(self, t, q, xi):
        frame_ID = (xi,)
        qDOF_P = self.subsystem1.qDOF_P(frame_ID)
        q_subsystem1 = q[: self.nq1][qDOF_P]
        return self.subsystem1.r_OC_xixi(t, q_subsystem1, frame_ID)

    def J1(self, t, q, xi):
        frame_ID = (xi,)
        qDOF_P = self.subsystem1.qDOF_P(frame_ID)
        q_subsystem1 = q[: self.nq1][qDOF_P]
        return self.subsystem1.J_C(t, q_subsystem1, frame_ID)

    def J1_q(self, t, q, xi):
        frame_ID = (xi,)
        qDOF_P = self.subsystem1.qDOF_P(frame_ID)
        q_subsystem1 = q[: self.nq1][qDOF_P]
        return self.subsystem1.J_C_q(t, q_subsystem1, frame_ID)

    def r2(self, t, q, eta):
        frame_ID = (eta,)
        qDOF_P = self.subsystem2.qDOF_P(frame_ID)
        q_subsystem2 = q[self.nq1 :][qDOF_P]
        return self.subsystem2.r_OC(t, q_subsystem2, frame_ID)

    def r2_q(self, t, q, eta):
        frame_ID = (eta,)
        qDOF_P = self.subsystem2.qDOF_P(frame_ID)
        q_subsystem2 = q[self.nq1 :][qDOF_P]
        return self.subsystem2.r_OC_q(t, q_subsystem2, frame_ID)

    def r2_eta(self, t, q, eta):
        frame_ID = (eta,)
        qDOF_P = self.subsystem2.qDOF_P(frame_ID)
        q_subsystem2 = q[self.nq1 :][qDOF_P]
        return self.subsystem2.r_OC_xi(t, q_subsystem2, frame_ID)

    def r2_etaeta(self, t, q, eta):
        frame_ID = (eta,)
        qDOF_P = self.subsystem2.qDOF_P(frame_ID)
        q_subsystem2 = q[self.nq1 :][qDOF_P]
        return self.subsystem2.r_OC_xixi(t, q_subsystem2, frame_ID)

    def J2(self, t, q, eta):
        frame_ID = (eta,)
        qDOF_P = self.subsystem2.qDOF_P(frame_ID)
        q_subsystem2 = q[self.nq1 :][qDOF_P]
        return self.subsystem2.J_C(t, q_subsystem2, frame_ID)

    def J2_q(self, t, q, eta):
        frame_ID = (eta,)
        qDOF_P = self.subsystem2.qDOF_P(frame_ID)
        q_subsystem2 = q[self.nq1 :][qDOF_P]
        return self.subsystem2.J_C_q(t, q_subsystem2, frame_ID)

    def distance(self, t, q, xi, eta):
        # position and tangant vector of subsystem 1
        r1 = self.r1(t, q, xi)

        # position and tangant vector of subsystem 2
        r2 = self.r2(t, q, eta)

        # distance of both centerlines and normal vector, Meier2017 (8), (12)
        delta_r = r1 - r2
        d = norm(delta_r)

        return d

    def distance_eta(self, t, q, xi, eta):
        # position of subsystem 1
        r1 = self.r1(t, q, xi)

        # position and tangant vector of subsystem 2
        r2 = self.r2(t, q, eta)
        r2_eta = self.r2_eta(t, q, eta)

        # distance of both centerlines and normal vector, Meier2017 (8), (12)
        delta_r = r1 - r2
        d = norm(delta_r)

        # gradient of distance function
        p2 = -0.5 / d * delta_r @ r2_eta

        return np.array([p2])

    def distance2(self, t, q, xi, eta):
        # position and tangant vector of subsystem 1
        r1 = self.r1(t, q, xi)

        # position and tangant vector of subsystem 2
        r2 = self.r2(t, q, eta)

        # squared distance of both centerlines, Meier2017 (8)
        delta_r = r1 - r2
        d2 = delta_r @ delta_r

        return d2

    def distance2_eta(self, t, q, xi, eta):
        # position and tangant vector of subsystem 1
        r1 = self.r1(t, q, xi)

        # position and tangant vector of subsystem 2
        r2 = self.r2(t, q, eta)
        r2_eta = self.r2_eta(t, q, eta)

        # distance of both centerlines and normal vector, Meier2017 (8), (12)
        delta_r = r1 - r2

        # Orthogonality constraints; close to Meier2017 (9) but it is the
        # correct gradient of the squared distance function
        p2 = -delta_r @ r2_eta

        return np.array([p2])

    def distance2_etaeta(self, t, q, xi, eta):
        # position and tangant vector of subsystem 1
        r1 = self.r1(t, q, xi)

        # position and tangant vector of subsystem 2
        r2 = self.r2(t, q, eta)
        r2_eta = self.r2_eta(t, q, eta)
        r2_etaeta = self.r2_etaeta(t, q, eta)

        # distance of both centerlines and normal vector, Meier2017 (8), (12)
        delta_r = r1 - r2

        # second gradient of squared distance function
        distance2_etaeta = np.array([[-delta_r @ r2_etaeta + r2_eta @ r2_eta]])
        # return distance2_etaeta

        distance2_etaeta_num = approx_fprime(
            eta, lambda eta: self.distance2_eta(t, q, xi, eta)
        )

        diff = distance2_etaeta - distance2_etaeta_num
        error = np.linalg.norm(diff)
        print(f"diff:\n{diff}")
        print(f"error distance2_etaeta_num: {error}")
        return distance2_etaeta_num

    def closest_points_minimize_d(self, t, q, xi):
        # optimization options
        # optimization_method = 'trust-constr'
        optimization_method = "SLSQP"

        fun = lambda eta: self.distance(t, q, xi, eta)

        # jac = '2-point'
        # # jac = '3-point'
        # jac = lambda p: self.grad_distance(t, q, *p)
        jac = lambda eta: self.distance_eta(t, q, xi, eta)
        # TODO:
        # hess = lambda eta: self.distance_etaeta(t, q, xi, eta)

        maxiter = 100
        tol = 1.0e-6

        bounds = [(0.0, 1.0) for _ in range(2)]

        if optimization_method == "trust-constr":
            options = {
                "verbose": 2,
                # 'disp': True,
                "disp": False,
                "maxiter": maxiter,
                "gtol": tol,
                "xtol": tol,
                "barrier_tol": tol,
            }
        elif optimization_method == "SLSQP":
            hess = None
            options = {
                # 'disp': True,
                "disp": False,
                "iprint": 2,
                "maxiter": maxiter,
                "ftol": tol,
            }
        else:
            raise RuntimeError(
                f'Unsupported optimization_method: "{optimization_method}" chosen!'
            )

        # start optimization
        try:
            p0 = np.array([self.xi_c, self.eta_c])
            res = minimize(
                fun,
                p0,
                method=optimization_method,
                jac=jac,
                hess=hess,
                #    constraints=nonlinear_constraints,
                options=options,
                bounds=bounds,
                tol=tol,
            )
            self.xi_c, self.eta_c = res.x  # update new initial guess
        except:
            print(f"problem occoured during optimization...")

        return res.x

    def closest_points_minimize_d2(self, t, q, xi):
        # optimization options
        # optimization_method = 'trust-constr'
        optimization_method = "SLSQP"

        fun = lambda p: self.distance2(t, q, xi, p)
        jac = lambda p: self.distance2_eta(t, q, xi, p)

        maxiter = 100
        tol = 1.0e-6
        bounds = [(0.0, 1.0)]

        if optimization_method == "trust-constr":
            options = {
                "verbose": 2,
                # 'disp': True,
                "disp": False,
                "maxiter": maxiter,
                "gtol": tol,
                "xtol": tol,
                "barrier_tol": tol,
            }
        elif optimization_method == "SLSQP":
            hess = None
            options = {
                # 'disp': True,
                "disp": False,
                "iprint": 2,
                "maxiter": maxiter,
                "ftol": tol,
            }
        else:
            raise RuntimeError(
                f'Unsupported optimization_method: "{optimization_method}" chosen!'
            )

        # start optimization
        try:
            p0 = np.array([self.eta_c])
            res = minimize(
                fun,
                p0,
                method=optimization_method,
                jac=jac,
                hess=hess,
                options=options,
                bounds=bounds,
                tol=tol,
            )
            self.eta_c = res.x[0]  # update new initial guess
        except:
            print(f"problem occoured during optimization...")

        return res.x

    def closest_points_prox_gradient_d2(
        self, t, q, xi, max_iter=500, tol=1.0e-6, prox_r=1.0e-3, debug=False
    ):
        """Optimize gradient of squared distance using a semi smooth Newton
        method with the inequality constraint 0 <= eta <= 1.

        The subsequent extended function is minimized without inequality
        constraints for eta:

             g_xi(eta) := min_{eta in R} d_xi(eta) + psi_[0, 1](eta)
             with d_xi = (r1(xi) - r2(eta))^2 - R1 - R2

        This can equivalently be written as:
             eta - prox_[0, 1](eta - r grad(g_xi(eta))) = 0
        """
        # def prox_R_ab(x, a=0, b=1):
        #     return max(a, min(b, x))

        # # mid function defined in Chapter 9.4, on p. 871 of Facchinei2003b
        # def mid(x, a=0, b=1):
        #     if x < a:
        #         return a
        #     elif x > b:
        #         return b
        #     else:
        #         return x

        # def phi(r, s, a=0, b=1):
        #     return r - mid(r - s, a, b)

        # def phi_smooth(r, s, a=0, b=1):
        #     """
        #     Smooth merit function for box constraints, see (9.4.7) in
        #     Facchinei2003b on p. 871.

        #     References
        #     ----------
        #     Facchinei2003b: https://link.springer.com/book/10.1007%2Fb97544
        #     """
        #     if s >= 0:
        #         if r <= a:
        #             return r - a
        #         elif r >= b:
        #             return (
        #                 r
        #                 - a
        #                 - sqrt((r - a) ** 2 + s**2)
        #                 + sqrt((r - b) ** 2 + s**2)
        #             )
        #         else:
        #             return r - a + s - sqrt((r - a) ** 2 + s**2)
        #     else:
        #         if r <= a:
        #             return (
        #                 r
        #                 - b
        #                 - sqrt((r - a) ** 2 + s**2)
        #                 + sqrt((r - b) ** 2 + s**2)
        #             )
        #         elif r >= b:
        #             return r - b
        #         else:
        #             return r - b + s + sqrt((r - b) ** 2 + s**2)

        def grad_fun(p):
            # extract unknown value
            eta = p[0]

            # gradient
            grad_g = self.distance2_eta(t, q, xi, eta)

            # # extended gradient reformulated using prox function
            # p = eta - prox_R_ab(eta - prox_r * grad_g[0])

            # distinguish all three cases of the prox equation above
            prox_arg = eta - prox_r * grad_g[0]
            if prox_arg <= 0:
                p = eta
            elif prox_arg >= 1:
                p = eta - 1.0
            else:
                p = grad_g[0]

            # # merit function defined in Chapter 9.4, on p. 871 of
            # # Facchinei2003b
            # r = eta
            # s = grad_g[0]
            # # p = phi(r, s)
            # p = phi(r, 1.0e-2 * s)
            # # p = phi_smooth(r, s)
            # # p = phi_smooth(r, 1.0e-2 * s)

            # return vector of merit function
            return np.array([p])

        # TODO:
        def hessian_fun(p):
            return np.array([[approx_fprime(p, lambda p: grad_fun(p))]])

        # initial error
        p = np.array([self.eta_c])
        grad = grad_fun(p)
        error = norm(grad)
        if error < tol:
            return p
        else:
            for i in range(max_iter):
                # perform Newton step
                hess = hessian_fun(p)
                try:
                    delta_p = -np.linalg.solve(hess, grad)
                except np.linalg.LinAlgError as err:
                    print(f"np.linalg.LinAlgError: {err}; we return p as it is")
                    return p

                # standard update
                p += delta_p

                # # Projection onto unit interval,
                # # see http://web.mit.edu/people/dimitrib/ProjectedNewton.pdf.
                # # This guards against evaluations outside, but might destory convergence.
                # p[0] = min(1., max(0., p[0]))

                # check convergence
                grad = grad_fun(p)
                error = norm(grad)

                if debug:
                    print(f" - i: {i}; p: {p}; error: {error:1.3e}")

                if error < tol:
                    self.eta_c = p[0]  # update new initial guess
                    if debug:
                        print(f"   => converged with p: {p}; error: {error:1.3e}")
                    return p

        if debug:
            print(f"   => not converged with p: {p}; error: {error:1.3e}")
            return p
        else:
            # raise RuntimeError('Internal Newton method is not converged.')
            print(f"   => not converged with p: {p}; error: {error:1.3e}")
            return p

    def distance2_boundary(self, t, q, xi, p):
        # extract vector p
        eta, alpha, beta = p

        # position and tangant vector of subsystem 1
        # smallest rotation in order to get an orthogonal frame
        r1 = self.r1(t, q, xi)
        r1_xi = self.r1_xi(t, q, xi)
        R1 = rodriguez_B(e1, r1_xi)
        # x1_b = r1 + self.R1 * (cos(alpha) * R1 @ e2 + sin(alpha) * R1 @ e3)
        x1_b = r1 + self.R1(xi, alpha) * (cos(alpha) * R1 @ e2 + sin(alpha) * R1 @ e3)

        # position and tangant vector of subsystem 2
        # smallest rotation in order to get an orthogonal frame
        r2 = self.r2(t, q, eta)
        r2_eta = self.r2_eta(t, q, eta)
        R2 = rodriguez_B(e1, r2_eta)
        # x2_b = r2 + self.R2 * (cos(beta) * R2 @ e2 + sin(beta) * R2 @ e3)
        x2_b = r2 + self.R2(eta, beta) * (cos(beta) * R2 @ e2 + sin(beta) * R2 @ e3)

        # squared distance of both centerlines, Meier2017 (8)
        delta_x_b = x1_b - x2_b
        d2 = delta_x_b @ delta_x_b

        # return d2
        return np.atleast_1d(d2)

        # # position and tangant vector of subsystem 1
        # # smallest rotation in order to get an orthogonal frame
        # def x1_b_fun(p):
        #     xi, alpha = p
        #     r1 = self.r1(t, q, xi)
        #     r1_xi = self.r1_xi(t, q, xi)
        #     R1 = rodriguez_B(e1, r1_xi)
        #     return r1 + self.R1(xi, alpha) * (cos(alpha) * R1 @ e2 + sin(alpha) * R1 @ e3)

        # p1 = np.array([xi, alpha])
        # x1_b = x1_b_fun(p1)

        # # position and tangant vector of subsystem 2
        # # smallest rotation in order to get an orthogonal frame
        # def x2_b_fun(p):
        #     eta, beta = p
        #     r2 = self.r2(t, q, eta)
        #     r2_eta = self.r2_eta(t, q, eta)
        #     R2 = rodriguez_B(e1, r2_eta)
        #     return r2 + self.R2(eta, beta) * (cos(beta) * R2 @ e2 + sin(beta) * R2 @ e3)

        # p2 = np.array([eta, beta])
        # x2_b = x2_b_fun(p2)

        # # distance of both centerlines and normal vector, Meier2016c (8), (12)
        # delta_x_b = x1_b - x2_b
        # d = norm(delta_x_b)

        # return np.atleast_1d(d)
        # # return np.atleast_1d(d * d)

    def distance2_boundary_p(self, t, q, xi, p):
        return approx_fprime(p, lambda p: self.distance2_boundary(t, q, xi, p))

    def closest_points_minimize_d2_boundary(self, t, q, xi):
        # optimization options
        optimization_method = "SLSQP"

        fun = lambda p: self.distance2_boundary(t, q, xi, p)
        jac = lambda p: self.distance2_boundary_p(t, q, xi, p)
        hess = None

        maxiter = 100
        tol = 1.0e-6
        bounds = [(0.0, 1.0), (-np.inf, +np.inf), (-np.inf, +np.inf)]
        # bounds = [(0.0, 1.0), (0, np.pi), (0, np.pi)]

        options = {
            # 'disp': True,
            "disp": False,
            "iprint": 2,
            "maxiter": maxiter,
            "ftol": tol,
        }

        # start optimization
        # try:
        p0 = np.array([self.eta_c, self.alpha_c, self.beta_c])
        res = minimize(
            fun,
            p0,
            method=optimization_method,
            jac=jac,
            hess=hess,
            options=options,
            bounds=bounds,
            tol=tol,
        )
        self.eta_c = res.x[0]  # update new initial guess
        self.alpha_c = res.x[1]
        self.beta_c = res.x[2]
        # except:
        #     print(f'problem occoured during optimization...')

        # print(f'xi_c: {res.x[0]}; alpha_c: {res.x[1]}; beta_c: {res.x[2]}')
        return res.x

    # def closest_points(self, t, q, xi, method='minimize d'):
    def closest_points(self, t, q, xi, method="minimize d2"):
        # def closest_points(self, t, q, xi, method="gradient d2 prox"):
        # def closest_points(self, t, q, xi, method='minimize d2 boundary'):
        if method == "minimize d":
            return self.closest_points_minimize_d(t, q, xi)
        elif method == "minimize d2":
            return self.closest_points_minimize_d2(t, q, xi)
        elif method == "gradient d2 prox":
            return self.closest_points_prox_gradient_d2(t, q, xi)
        elif method == "minimize d2 boundary":
            return self.closest_points_minimize_d2_boundary(t, q, xi)
        else:
            raise RuntimeError('Chosen method "' + method + '" is not implemented!')

    def f_pot(self, t, q):
        f = np.zeros(self.__nu)
        for el in range(self.subsystem1.nelement):
            for i in range(self.subsystem1.nquadrature):
                # material point of subsystem 1
                xi = self.subsystem1.qp[el, i]

                #####################################################
                # closest point on circular Bernoulli beam boundaries
                #####################################################

                # find closest point on second line, Meier2016c (35), (36)
                eta_c = self.closest_points(t, q, xi)[0]

                # position and tangant vector of subsystem 1
                r1 = self.r1(t, q, xi)
                J1 = self.J1(t, q, xi)
                uDOF1 = self.subsystem1.uDOF_P((xi,))

                # position and tangant vector of subsystem 2
                r2 = self.r2(t, q, eta_c)
                J2 = self.J2(t, q, eta_c)
                uDOF2 = self.subsystem1.uDOF_P((eta_c,))

                # distance of both centerlines and normal vector, Meier2016c (8), (12)
                delta_r = r1 - r2
                d = norm(delta_r)
                n = delta_r / d

                # gap function, Meier2016c (37)
                g = d - self.R1 - self.R2
                # g = d - self.R1(xi, 0) - self.R2(eta_c, 0)

                ###################################
                # generalized forces of constraints
                ###################################

                # constraint forces and potential, Meier2016c (38), (39), (40)
                # E_c = self.contact_potential.potential(g)
                f_c = self.contact_potential.potential_g(g) * n

                # virtual work contribution from the current qzadrature point
                # - subsystem 1 contribution
                f[: self.nu1][uDOF1] -= J1.T @ f_c * self.subsystem1.qw[el, i]
                # - subsystem 2 contribution
                f[self.nu1 :][uDOF2] += J2.T @ f_c * self.subsystem1.qw[el, i]

        return f

    def f_pot_q_dense(self, t, q):
        f_q = np.zeros((self.__nu, self.__nq))
        for el in range(self.subsystem1.nelement):
            for i in range(self.subsystem1.nquadrature):
                # material point of subsystem 1
                xi = self.subsystem1.qp[el, i]

                #####################################################
                # closest point on circular Bernoulli beam boundaries
                #####################################################

                # find closest point on second line, Meier2016c (35), (36)
                eta_c = self.closest_points(t, q, xi)[0]

                # position and tangant vector of subsystem 1
                r1 = self.r1(t, q, xi)
                r1_q = self.r1_q(t, q, xi)
                J1 = self.J1(t, q, xi)
                J1_q = self.J1(t, q, xi)
                elDOF1 = self.subsystem1.elDOF_P((xi,))

                # position and tangant vector of subsystem 2
                r2 = self.r2(t, q, eta_c)
                r2_q = self.r2_q(t, q, eta_c)
                J2 = self.J2(t, q, eta_c)
                elDOF2 = self.subsystem1.elDOF_P((eta_c,))

                # distance of both centerlines and normal vector, Meier2016c (8), (12)
                delta_r = r1 - r2
                d = norm(delta_r)
                n = delta_r / d

                # gap function, Meier2016c (37)
                g = d - self.R1 - self.R2

                # p2 = -0.5 / d * delta_r @ r2_eta
                g_q1 = (delta_r / d) @ r1_q
                g_q2 = -(delta_r / d) @ r2_q

                ###################################
                # generalized forces of constraints
                ###################################

                # constraint forces and potential, Meier2016c (38), (39), (40)
                # E_c = self.contact_potential.potential(g)
                f_c = self.contact_potential.potential_g(g) * n
                f_c_q = self.contact_potential.potential_g(
                    g
                ) * n_q + self.contact_potential.potential_gg(g) * np.outer(n, g_q)

                # virtual work contribution from the current qzadrature point
                uDOF1 = np.arange(0, self.nu1)
                qDOF1 = np.arange(0, self.nq1)
                uDOF2 = np.arange(self.nu1, self.__nu)
                qDOF2 = np.arange(self.nq1, self.__nq)

                # - subsystem 1 contribution
                uDOF1_P = uDOF1[elDOF1]
                qDOF1_P = qDOF1[elDOF1]
                f_q[uDOF1_P[:, None], qDOF1_P] -= (
                    J1.T @ f_c_q + np.einsum("jik,j", J1_q, f_c)
                ) * self.subsystem1.qw[el, i]

                # - subsystem 2 contribution
                uDOF2_P = uDOF2[elDOF2]
                qDOF2_P = qDOF2[elDOF2]
                f_q[uDOF2_P[:, None], qDOF2_P] += (
                    J2.T @ f_c_q + np.einsum("jik,j", J2_q, f_c)
                ) * self.subsystem1.qw[el, i]

        return f_q

    # TODO: Derive analytical derivative. This requires the derivatives of
    #       xi_c, eta_c w.r.t. changes of q's. See Meier2017, Appendix C
    #       for a derivation.
    def f_pot_q(self, t, q, coo):
        dense_num = approx_fprime(q, lambda q: self.f_pot(t, q))
        coo.extend(dense_num, (self.uDOF, self.qDOF))

    # ###############
    # # visualization
    # ###############
    # # TODO: new visualization for all quadrature points!
    # def contact_points(self, t, q):
    #     points = np.zeros((self.n_contact_points, 3, 2))
    #     active_contacts = np.zeros(self.n_contact_points, dtype=bool)

    #     q_subsystems = q[self.qDOF]

    #     subsystem1 = self.subsystem1
    #     for el in range(subsystem1.nEl):
    #         for i in range(subsystem1.nQP):
    #             # material point on subsystem 1 (slave)
    #             xi = subsystem1.qp[el, i]

    #             # # closest point on subsystem 2 (master)
    #             # eta = self.closest_points(t, q_subsystems, xi)

    #             # # position vector of subsystem 1 (slave)
    #             # r1 = self.r1(t, q, xi)

    #             # # position vector of subsystem 2 (master)
    #             # r2 = self.r2(t, q, eta)

    #             # # gap function, Meier2016c (37)
    #             # g = norm(r1 - r2) - self.R1(xi, 0) - self.R2(eta, 0)

    #             ########################
    #             # complex closest points
    #             ########################
    #             eta, alpha, beta = self.closest_points_minimize_d2_boundary(
    #                 t, q_subsystems, xi
    #             )

    #             # position and tangant vector of subsystem 1
    #             # smallest rotation in order to get an orthogonal frame
    #             r1 = self.r1(t, q, xi)
    #             r1_xi = self.r1_xi(t, q, xi)
    #             R1 = rodriguez_B(e1, r1_xi)
    #             x1_b = r1 + self.R1(xi, alpha) * (
    #                 cos(alpha) * R1 @ e2 + sin(alpha) * R1 @ e3
    #             )
    #             # x1_b = r1 + self.R1 * (cos(alpha) * R1 @ e2 + sin(alpha) * R1 @ e3)

    #             # position and tangant vector of subsystem 2
    #             # smallest rotation in order to get an orthogonal frame
    #             r2 = self.r2(t, q, eta)
    #             r2_eta = self.r2_eta(t, q, eta)
    #             R2 = rodriguez_B(e1, r2_eta)
    #             # x2_b = r2 + self.R2 * (cos(beta) * R2 @ e2 + sin(beta) * R2 @ e3)
    #             x2_b = r2 + self.R2(eta, beta) * (
    #                 cos(beta) * R2 @ e2 + sin(beta) * R2 @ e3
    #             )

    #             # gap function
    #             g = norm(x1_b - x2_b) - (self.R1(xi, alpha) + self.R2(eta, beta)) * 0.1

    #             ########################
    #             ########################

    #             # linear index
    #             idx = el * subsystem1.nQP + i

    #             # store active or inactive contact
    #             active_contacts[idx] = g <= 0

    #             # # # store all points
    #             # points[idx, 0] = np.array([r1[0], r2[0]])
    #             # points[idx, 1] = np.array([r1[1], r2[1]])
    #             # points[idx, 2] = np.array([r1[2], r2[2]])

    #             points[idx, 0] = np.array([x1_b[0], x2_b[0]])
    #             points[idx, 1] = np.array([x1_b[1], x2_b[1]])
    #             points[idx, 2] = np.array([x1_b[2], x2_b[2]])

    #     return points, active_contacts
