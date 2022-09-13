from math import pi, sin, cos
import numpy as np
from cardillo.math.algebra import (
    norm,
    ax2skew,
    cross3,
    e1,
    e2,
    e3,
)
from cardillo.math.numerical_derivative import Numerical_derivative
from scipy.optimize import fsolve
from scipy.optimize import minimize

# TODO: Use xi and eta as new generalized coordinates and make this a g_contr
#       subsystem that enforces both to be such that they satisfy the
#       orthogonality conditions.
class Point2Point:
    def __init__(
        self, eps, R1, R2, subsystem1, subsystem2, q0=None, u0=None, xi_c=0.5, eta_c=0.5
    ):

        # constact stiffness
        self.eps = eps

        # pulley radii
        self.R1 = R1
        self.R2 = R2

        # information of subsystem 1
        self.subsystem1 = subsystem1

        # information of subsystem 2
        self.subsystem2 = subsystem2

        # initial and current values for the closest points
        self.xi_c = xi_c
        self.eta_c = eta_c

        # # q_contr part
        # self.nu = self.nq = 2 # xi_c and eta_c
        # self.q0 = np.array([xi_c, eta_c]) if q0 is None else q0
        # self.u0 = np.zeros(self.nu) if u0 is None else u0

        # decide how often the jacobian will be reused
        # 1: means always recompute the jacobian which is expensive
        self.n_jacobian_update = 1
        self.jacobian_updates = 0
        self.f_npot_q_num = None

    def assembler_callback(self):
        # # generalized coordinates of the two material parameters xi_c and eta_c
        # qDOF_xi_eta_multipliers = self.qDOF.copy()
        # uDOF_xi_eta_multipliers = self.uDOF.copy()

        # generalized coordinates of the two subsystems
        # TODO: we cannot restrict this to the qDOF's of s single point
        #       identified by a frame_ID
        # qDOF1 = self.subsystem1.qDOF_P(self.frame_ID1)
        # qDOF2 = self.subsystem2.qDOF_P(self.frame_ID2)
        # qDOF_subsystems = np.concatenate([self.subsystem1.qDOF[qDOF1],
        #                                   self.subsystem2.qDOF[qDOF2]])
        qDOF_subsystems = np.concatenate([self.subsystem1.qDOF, self.subsystem2.qDOF])

        self.nq1 = nq1 = len(self.subsystem1.qDOF)
        self.nq2 = len(self.subsystem2.qDOF)
        self.__nq = self.nq1 + self.nq2

        uDOF_subsystems = np.concatenate([self.subsystem1.uDOF, self.subsystem2.uDOF])

        self.nu1 = nu1 = len(self.subsystem1.uDOF)
        self.nu2 = len(self.subsystem2.uDOF)
        self.__nu = self.nu1 + self.nu2

        # combine newly introduced generalized coordinates with the ones of
        # the two subsystems
        # self.qDOF = np.concatenate((qDOF_xi_eta_multipliers, qDOF_subsystems))
        # self.uDOF = np.concatenate((uDOF_xi_eta_multipliers, uDOF_subsystems))
        self.qDOF = qDOF_subsystems
        self.uDOF = uDOF_subsystems

        # local indices for generalized coordinates of the angles and
        # subsystems
        # self.xi_eta_DOF = np.arange(0, 2)
        # self.subsystemsDOF = np.arange(0, self.__nq) + self.nq

        # build functions that compute the position of the pulley anchors and
        # their orientation
        # TODO: The derivatives here are with respect to the qe's of a point identified by the
        #       frame_ID. This is not valid since the frame_ID changes during iterations.
        # def r1(self, t, q):
        #     q_subsystem = q[:nq1]
        #     xi_c = q[self.xi_eta_DOF[0]]
        #     self.subsystem1.r_OP(t, q_subsystem, frame_ID=(xi_c, ))

        # # self.r1 = lambda t, q, frame_ID: self.subsystem1.r_OP(t, q[:nq1], frame_ID=frame_ID)
        # self.r1_q = lambda t, q, frame_ID: self.subsystem1.r_OP_q(t, q[:nq1], frame_ID=frame_ID)
        # self.J_P1 = lambda t, q, frame_ID: self.subsystem1.J_P(t, q[:nq1], frame_ID=frame_ID)

        # self.r2 = lambda t, q, frame_ID: self.subsystem2.r_OP(t, q[nq1:], frame_ID=frame_ID)
        # self.r2_q = lambda t, q, frame_ID: self.subsystem2.r_OP_q(t, q[nq1:], frame_ID=frame_ID)
        # self.J_P2 = lambda t, q, frame_ID: self.subsystem2.J_P(t, q[nq1:], frame_ID=frame_ID)

        # TODO: We should compute the initial closest points for the given q's from both subsystems

        # # compute initial tendon length
        # q_init = np.concatenate((self.q0, self.subsystem1.q0, self.subsystem2.q0))

        # from scipy.optimize import fsolve
        # def g_p_func(p):
        #     q = q_init
        #     q[self.xi_eta_DOF] = p
        #     g_p, bar_w_p, w_p, l_PQ, bar_w_l, w_l = self.all_in_one(0, q)
        #     return g_p

        # def g_p_p_func(p):
        #     q = q_init
        #     q[self.xi_eta_DOF] = p
        #     g_p, bar_w_p, w_p, l_PQ, bar_w_l, w_l = self.all_in_one(0, q)
        #     return bar_w_p.T

        # # p_init = q_init[self.anglesDOF]
        # # p_init += np.random.rand(4) * 0.1
        # # # sol = fsolve(g_p_func, p_init.copy(), full_output=True)
        # # p0, infodict, _, message = fsolve(g_p_func, p_init, full_output=True, fprime=g_p_p_func)
        # # q0 = q_init.copy()
        # # q0[self.anglesDOF] = p0

        # q0 = q_init.copy()
        # self.l_QP_0 = self.l_QP(0, q0)
        # # # correct tendon length such that the initial force is
        # # # approximately 10N
        # # # self.l_QP_0 -= 0.7 * 1.0e-3
        # # # self.l_QP_0 -= 1 * 1.0e-3
        # # print(f'solve initial angles for SPP tendon:')
        # # print(f' - p0: {p0}')
        # # print(f' - p_init: {p_init}')
        # # print(f' - g_p(p0): {g_p_func(p0)}')
        # # print(f' - g_p(p_init): {g_p_func(p_init)}')
        # # print(f' - l_QP_0: {self.l_QP_0}')
        # # print(f' - l_QP_init: {self.l_QP(0, q_init)}')
        # # print(f' - message: {message}')
        # # print(f'')

    def r1(self, t, q, xi):
        q_subsystem1 = q[: self.nq1]
        return self.subsystem1.r(t, q_subsystem1, xi)

    def r1_q(self, t, q, xi):
        q_subsystem1 = q[: self.nq1]
        return self.subsystem1.r_q(t, q_subsystem1, xi)

    def r1_xi(self, t, q, xi):
        q_subsystem1 = q[: self.nq1]
        return self.subsystem1.r_xi(t, q_subsystem1, xi)

    def r1_xixi(self, t, q, xi):
        q_subsystem1 = q[: self.nq1]
        return self.subsystem1.r_xixi(t, q_subsystem1, xi)

    def r2(self, t, q, eta):
        q_subsystem2 = q[self.nq1 :]
        return self.subsystem2.r(t, q_subsystem2, eta)

    def r2_q(self, t, q, eta):
        q_subsystem2 = q[self.nq1 :]
        return self.subsystem2.r_q(t, q_subsystem2, eta)

    def r2_eta(self, t, q, eta):
        q_subsystem2 = q[self.nq1 :]
        return self.subsystem2.r_xi(t, q_subsystem2, eta)

    def r2_etaeta(self, t, q, eta):
        q_subsystem2 = q[self.nq1 :]
        return self.subsystem2.r_xixi(t, q_subsystem2, eta)

    # TODO: This is a problem! We have to extrapolate an arbitrary point
    #       on the centerline if xi, eta exceed the unit interval,
    #       otherwise the closest point cannot be computed!
    def extrapolate_centerline1(self, t, q, xi, xi0):
        r1_xi = self.r1_xi(t, q, xi0)
        tangent = r1_xi / norm(r1_xi)
        r1 = tangent * (xi - xi0)
        r1_q = self.r1_q(t, q, xi0)
        return r1, r1_xi, r1_q

    def extrapolate_centerline2(self, t, q, eta, eta0):
        r2_eta = self.r2_eta(t, q, eta0)
        tangent = r2_eta / norm(r2_eta)
        r2 = tangent * (eta - eta0)
        r2_q = self.r2_q(t, q, eta0)
        return r2, r2_eta, r2_q

    def distance(self, t, q, p):
        xi, eta = p

        # position and tangant vector of subsystem 1
        r1 = self.r1(t, q, xi)

        # position and tangant vector of subsystem 2
        r2 = self.r2(t, q, eta)

        # distance of both centerlines and normal vector, Meier2017 (8), (12)
        delta_r = r1 - r2
        d = norm(delta_r)
        return d

    def grad_distance(self, t, q, p):
        xi, eta = p

        # position and tangant vector of subsystem 1
        r1 = self.r1(t, q, xi)
        r1_xi = self.r1_xi(t, q, xi)

        # position and tangant vector of subsystem 2
        r2 = self.r2(t, q, eta)
        r2_eta = self.r2_eta(t, q, eta)

        # distance of both centerlines and normal vector, Meier2017 (8), (12)
        delta_r = r1 - r2
        d = norm(delta_r)

        # gradient of distance function
        p1 = 0.5 / d * delta_r @ r1_xi
        p2 = -0.5 / d * delta_r @ r2_eta

        return np.array([p1, p2])

    def distance2(self, t, q, p):
        xi, eta = p

        # position and tangant vector of subsystem 1
        r1 = self.r1(t, q, xi)
        r1_xi = self.r1_xi(t, q, xi)

        # position and tangant vector of subsystem 2
        r2 = self.r2(t, q, eta)
        r2_eta = self.r2_eta(t, q, eta)

        # squared distance of both centerlines, Meier2017 (8)
        delta_r = r1 - r2
        d2 = delta_r @ delta_r
        return d2

    def grad_distance2(self, t, q, p):
        # TODO: we have to make these function calls more robust w.r.t. xi,
        #       eta being outside the unit interval! Otherwise the
        #       orthogonality cannot be computed.
        xi, eta = p

        # # position and tangant vector of subsystem 1
        # if xi < 0:
        #     r1, r1_xi, r1_q = self.extrapolate_centerline1(t, q, xi, 0.0)
        # elif xi > 1:
        #     r1, r1_xi, r1_q = self.extrapolate_centerline1(t, q, xi, 1.0)
        # else:
        #     r1 = self.r1(t, q, xi)
        #     r1_xi = self.r1_xi(t, q, xi)

        # # position and tangant vector of subsystem 2
        # if eta < 0:
        #     r2, r2_eta, r2_q = self.extrapolate_centerline2(t, q, eta, 0.0)
        # elif eta > 1:
        #     r2, r2_eta, r2_q = self.extrapolate_centerline2(t, q, eta, 1.0)
        # else:
        #     r2 = self.r2(t, q, eta)
        #     r2_eta = self.r2_eta(t, q, eta)

        # xi = min(1., max(0., xi))
        # eta = min(1., max(0., eta))

        # position and tangant vector of subsystem 1
        r1 = self.r1(t, q, xi)
        r1_xi = self.r1_xi(t, q, xi)

        # position and tangant vector of subsystem 2
        r2 = self.r2(t, q, eta)
        r2_eta = self.r2_eta(t, q, eta)

        # distance of both centerlines and normal vector, Meier2017 (8), (12)
        delta_r = r1 - r2

        # Orthogonality constraints; close to Meier2017 (9) but it is the
        # correct gradient of the squared distance function
        p1 = r1_xi @ delta_r
        p2 = -r2_eta @ delta_r

        return np.array([p1, p2])

    def grad2_distance2(self, t, q, p):
        xi, eta = p

        # position and tangant vector of subsystem 1
        r1 = self.r1(t, q, xi)
        r1_xi = self.r1_xi(t, q, xi)
        r1_xixi = self.r1_xixi(t, q, xi)

        # position and tangant vector of subsystem 2
        r2 = self.r2(t, q, eta)
        r2_eta = self.r2_eta(t, q, eta)
        r2_etaeta = self.r2_etaeta(t, q, eta)

        # distance of both centerlines and normal vector, Meier2017 (8), (12)
        delta_r = r1 - r2

        # second gradient of squared distance function
        grad2_d2 = np.zeros((2, 2))
        grad2_d2[0, 0] = delta_r @ r1_xixi + r1_xi @ r1_xi
        grad2_d2[0, 1] = grad2_d2[1, 0] = -r2_eta @ r1_xi
        grad2_d2[1, 1] = -delta_r @ r2_etaeta + r2_eta @ r2_eta

        return grad2_d2

        # fun = lambda t, p: self.grad_distance2(t, q, p)
        # grad2_d2_num = Numerical_derivative(fun, order=2)._x(t, p)

        # diff = grad2_d2 - grad2_d2_num
        # error = np.linalg.norm(diff)
        # print(f'diff:\n{diff}')
        # print(f'error grad2_distance2: {error}')
        # return grad2_d2_num

    # TODO is this an old verion of the funtion below?
    # def closest_points_gradient_d2(self, t, q):
    #     p0 = np.array([self.xi_c, self.eta_c])
    #     f = lambda p: self.grad_distance2(t, q, p)
    #     f_p = lambda p: self.grad2_distance2(t, q, p)
    #     sol = fsolve(f, p0, fprime=f_p, full_output=True)
    #     self.xi_c, self.eta_c = sol[0]  # update new initial guess

    #     if sol[2] != 1:
    #         print(f"Closest point not found! Message: " + sol[3])

    #     return sol[0]

    # def closest_points_gradient_d2(self, t, q, max_iter=100, tol=1.0e-6, debug=False, damped=False):
    def closest_points_gradient_d2(
        self, t, q, max_iter=100, tol=1.0e-6, debug=False, damped=True
    ):
        if debug:
            print(f"closest point projection called:")
        # initial guess
        p = np.array([self.xi_c, self.eta_c])

        # initial error
        grad = self.grad_distance2(t, q, p)
        error = norm(grad)
        if error < tol:
            return p
        else:
            for i in range(max_iter):
                # perform Newton step
                hess = self.grad2_distance2(t, q, p)
                delta_p = -np.linalg.solve(hess, grad)

                if damped:
                    # # delta_p = np.array([0.5, 0.0001]) # dummy update that gets positive
                    # # delta_p = -np.array([0.5, 0.0001]) # dummy update that gets negative
                    # safety = 0.95
                    # damping = 1

                    # # print(f'compute new damping:')
                    # for j, pj in enumerate(p):
                    #     if delta_p[j] > 1.0e-5:
                    #         # xi, eta >= 0
                    #         arg1 = -safety * pj / delta_p[j]
                    #         # print(f'arg1: {arg1}')
                    #         damping = min(damping, arg1)

                    #         # xi, eta <= 1
                    #         arg2 = safety * (1 - pj) / delta_p[j]
                    #         # print(f'arg2: {arg2}')
                    #         damping = max(damping, arg2)

                    #     # print(f' - damping: {damping}')

                    # print(f'p before update: {p}')
                    # p += damping * delta_p
                    # print(f'p after update: {p}')

                    # standard update
                    p += delta_p

                    # projection onto unit interval,
                    # see http://web.mit.edu/people/dimitrib/ProjectedNewton.pdf
                    for j in range(2):
                        p[j] = min(1.0, max(0.0, p[j]))

                else:
                    # standard update
                    p += delta_p

                # check convergence
                grad = self.grad_distance2(t, q, p)
                error = norm(grad)

                if debug:
                    print(f" - i: {i}; p: {p}; error: {error}")

                if error < tol:
                    if debug:
                        print(f"   => converged with p: {p}; error: {error}")
                    return p

        if debug:
            print(f"   => not converged with p: {p}; error: {error}")
            # print('Internal Newton method is not converged.')
            return p
        else:
            raise RuntimeError("Internal Newton method is not converged.")

    # def closest_points_prox_gradient_d2(self, t, q, max_iter=100, tol=1.0e-6,
    # #                                     prox_r=0.1, debug=False):
    # def closest_points_prox_gradient_d2(self, t, q, max_iter=500, tol=1.0e-4,
    #                                     prox_r=0.1, debug=True):
    #     """Optimize gradient of squared distance using a semi smooth Newton
    #        method with the inequality constraints 0 <= xi <= 1 and
    #        0 <= eta <= 1. Is https://stanford.edu/group/SOL/multiscale/papers/14siopt-proxNewton.pdf
    #        a good reference? At least there is some for semi-smooth Newton methods."""
    #     if debug:
    #         print(f'closest point projection called:')
    #     # initial guess Lagrange multipliers
    #     # mu_xi0 = mu_xi1 = 0
    #     # mu_eta0 = mu_eta1 = 0
    #     # p = np.array([self.xi_c, self.eta_c, mu_xi0, mu_xi1, mu_eta0, mu_eta1])

    #     p = np.array([self.xi_c, self.eta_c, 0, 0])

    #     def prox_R0plus(x):
    #         return max(0, x)

    #     def prox_Ball(x, radius, center):
    #         arg = x - center
    #         norm_arg = norm(arg)
    #         if norm_arg <= radius:
    #             return x
    #         else:
    #             if norm_arg > 0:
    #                 return radius * arg / norm_arg
    #             else:
    #                 return arg

    #     def grad_fun(p):

    #         ##################
    #         # ball inequalitiy
    #         ##################
    #         # extract unknowns
    #         xi, eta, mu0, mu1 = p
    #         xieta = p[:2]
    #         mu = p[2:]

    #         # "constraint function" ~ relative contact velocities
    #         g = p[:2]
    #         g0 = p[0]
    #         g1 = p[1]

    #         # gradient w.r.t. xi, eta
    #         grad_g = np.eye(2)
    #         grad_g0 = np.array([1, 0])
    #         grad_g1 = np.array([0, 1])

    #         # inequality complementarity
    #         prox = mu - prox_Ball(mu - prox_r * g, 0.5, np.array([0.5, 0.5]))

    #         # residual equations
    #         grad_xi_eta = self.grad_distance2(t, q, xieta) + grad_g @ mu

    #         return np.array([*grad_xi_eta, *prox])

    #         ###################
    #         # four inequalities
    #         ###################
    #         # # extract unknowns
    #         # xi, eta, mu0, mu1, mu2, mu3 = p

    #         # # inequalities
    #         # g0 = -xi     # gi <= 0 => -xi <= 0
    #         # g1 = xi - 1  # gi <= 0 => xi - 1 <= 0
    #         # g2 = -eta    # gi <= 0 => -eta <= 0
    #         # g3 = eta - 1 # gi <= 0 => eta - 1 <= 0

    #         # # gradients w.r.t. xi, eta
    #         # grad_g0 = np.array([-1,  0])
    #         # grad_g1 = np.array([ 1,  0])
    #         # grad_g2 = np.array([ 0, -1])
    #         # grad_g3 = np.array([ 0,  1])

    #         # # inequality complementarities for xi:
    #         # #  -- g0 <= 0, mu0 >= 0, g0 * mu0 = 0
    #         # prox0 = mu0 - prox_R0plus(mu0 + prox_r * g0)
    #         # #  -- g1 <= 0, mu1 >= 0, g1 * mu1 = 0
    #         # prox1 = mu1 - prox_R0plus(mu1 + prox_r * g1)

    #         # # inequality complementarities for eta:
    #         # #  -- g2 <= 0, mu2 >= 0, g2 * mu2 = 0
    #         # prox2 = mu2 - prox_R0plus(mu2 + prox_r * g2)
    #         # #  -- g3 <= 0, mu3 >= 0, g3 * mu3 = 0
    #         # prox3 = mu3 - prox_R0plus(mu3 + prox_r * g3)

    #         # grad_xi_eta = self.grad_distance2(t, q, p[:2]) + grad_g0 * mu0 \
    #         #               + grad_g1 * mu1 + grad_g2 * mu2 + grad_g3 * mu3

    #         # return np.array([*grad_xi_eta,
    #         #                  prox0,
    #         #                  prox1,
    #         #                  prox2,
    #         #                  prox3])

    #     def hessian_fun(p):
    #         return Numerical_derivative(lambda t, p: grad_fun(p))._x(0, p)

    #     # initial error
    #     grad = grad_fun(p)
    #     error = norm(grad)
    #     if error < tol:
    #         return p
    #     else:
    #         for i in range(max_iter):
    #             # perform Newton step
    #             hess = hessian_fun(p)
    #             try:
    #                 delta_p = -np.linalg.solve(hess, grad)
    #             except np.linalg.LinAlgError as err:
    #                 print(f'np.linalg.LinAlgError: {err}; we return p as it is')
    #                 return p
    #                 # if 'Singular matrix' in str(err):
    #                 #     # your error handling block
    #                 # else:
    #                 #     raise

    #             # standard update
    #             p += delta_p

    #             # # Projection onto unit interval,
    #             # # see http://web.mit.edu/people/dimitrib/ProjectedNewton.pdf.
    #             # # This guards against evaluations outside, but might destory convergence.
    #             # eps = 1.0e-5
    #             # for j in range(2):
    #             #     # p[j] = min(1., max(0., p[j]))
    #             #     p[j] = min(1. - eps, max(eps, p[j]))

    #             # check convergence
    #             grad = grad_fun(p)
    #             error = norm(grad)

    #             if debug:
    #                 print(f' - i: {i}; p: {p}; error: {error}')

    #             if error < tol:
    #                 if debug:
    #                     print(f'   => converged with p: {p}; error: {error}')
    #                 return p

    #     if debug:
    #         print(f'   => not converged with p: {p}; error: {error}')
    #         return p
    #     else:
    #         raise RuntimeError('Internal Newton method is not converged.')

    # def closest_points_prox_gradient_d2(self, t, q, max_iter=500, tol=1.0e-6,
    #                                     prox_r=0.1, debug=True):
    def closest_points_prox_gradient_d2(
        self, t, q, max_iter=500, tol=1.0e-6, prox_r=0.1, debug=False
    ):
        """Optimize gradient of squared distance using a semi smooth Newton
        method with the inequality constraints 0 <= xi <= 1 and
        0 <= eta <= 1. Is https://stanford.edu/group/SOL/multiscale/papers/14siopt-proxNewton.pdf
        a good reference? At least there is some for semi-smooth Newton methods."""
        if debug:
            print(f"closest point projection called:")

        p = np.array([self.xi_c, self.eta_c])

        def indicator(x, a=0, b=1):
            if x < a:
                return np.inf
            elif x > b:
                return np.inf
            else:
                return 0

        def prox_R_ab(x, a=0, b=1):
            return max(a, min(b, x))

        def grad_fun(p):
            # extract unknowns
            xi, eta = p

            # gradient
            grad_g = self.grad_distance2(t, q, p)

            p0 = xi - prox_R_ab(xi - prox_r * grad_g[0])
            p1 = eta - prox_R_ab(eta - prox_r * grad_g[1])

            return np.array([p0, p1])

        def hessian_fun(p):
            return Numerical_derivative(lambda t, p: grad_fun(p))._x(0, p)

        # initial error
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
                    # if 'Singular matrix' in str(err):
                    #     # your error handling block
                    # else:
                    #     raise

                # standard update
                p += delta_p

                # # Projection onto unit interval,
                # # see http://web.mit.edu/people/dimitrib/ProjectedNewton.pdf.
                # # This guards against evaluations outside, but might destory convergence.
                # eps = 1.0e-5
                # for j in range(2):
                #     # p[j] = min(1., max(0., p[j]))
                #     p[j] = min(1. - eps, max(eps, p[j]))

                # check convergence
                grad = grad_fun(p)
                error = norm(grad)

                if debug:
                    print(f" - i: {i}; p: {p}; error: {error:1.3e}")

                if error < tol:
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

    def closest_points_minimize_d(self, t, q):
        # optimization options
        # optimization_method = 'trust-constr'
        optimization_method = "SLSQP"

        fun = lambda p: self.distance(t, q, p)

        # jac = '2-point'
        # # jac = '3-point'
        jac = lambda p: self.grad_distance(t, q, p)

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

    def closest_points_minimize_d2(self, t, q):
        # optimization options
        # optimization_method = 'trust-constr'
        optimization_method = "SLSQP"

        fun = lambda p: self.distance2(t, q, p)
        jac = lambda p: self.grad_distance2(t, q, p)

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

    # def closest_points(self, t, q, method='gradient'):
    # def closest_points(self, t, q, method='gradient prox'):
    def closest_points(self, t, q, method="minimize d"):
        # def closest_points(self, t, q, method='minimize d2'):
        if method == "gradient":
            return self.closest_points_gradient_d2(t, q)
        elif method == "gradient prox":
            return self.closest_points_prox_gradient_d2(t, q)[:2]
        elif method == "minimize d":
            return self.closest_points_minimize_d(t, q)
        elif method == "minimize d2":
            return self.closest_points_minimize_d2(t, q)
        else:
            raise RuntimeError('Chosen method "' + method + '" is not implemented!')

    def all_in_one(self, t, q):
        xi_c, eta_c = self.closest_points(t, q)

        # position and tangant vector of subsystem 1
        r1 = self.r1(t, q, xi_c)
        r1_q = self.r1_q(t, q, xi_c)

        # position and tangant vector of subsystem 2
        r2 = self.r2(t, q, eta_c)
        r2_q = self.r2_q(t, q, eta_c)

        # # position and tangant vector of subsystem 1
        # if xi_c < 0:
        #     r1, _, r1_q = self.extrapolate_centerline1(t, q, xi_c, 0.0)
        # elif xi_c > 1:
        #     r1, _, r1_q = self.extrapolate_centerline1(t, q, xi_c, 1.0)
        # else:
        #     r1 = self.r1(t, q, xi_c)
        #     r1_q = self.r1_q(t, q, xi_c)

        # # position and tangant vector of subsystem 2
        # if eta_c < 0:
        #     r2, r2_eta, r2_q = self.extrapolate_centerline2(t, q, eta_c, 0.0)
        # elif eta_c > 1:
        #     r2, r2_eta, r2_q = self.extrapolate_centerline2(t, q, eta_c, 1.0)
        # else:
        #     r2 = self.r2(t, q, eta_c)
        #     r2_q = self.r2_q(t, q, eta_c)

        # distance of both centerlines and normal vector, Meier2017 (8), (12)
        delta_r = r1 - r2
        d = norm(delta_r)
        n = delta_r / d

        # gap function, Meier2017 (10)
        g = d - self.R1 - self.R2
        # print(f'gap: {g}')

        def Macaulay(x):
            """Macaulay brackets, see https://en.wikipedia.org/wiki/Macaulay_brackets."""
            # return -min(0, x)
            # return max(0, -x)

            if x < 0:
                return x
            else:
                return 0

            # return x

        # constraint forces and potential, Meier2017 (11), (12)
        macauly_g = Macaulay(g)
        E_pot = 0.5 * self.eps * macauly_g**2
        f_pot = self.eps * macauly_g * n

        # TODO: Implement derivative of potential force and their variations!

        return E_pot, f_pot, r1_q, r2_q

    def f_npot(self, t, q, u):
        E_pot, f_pot, r1_q, r2_q = self.all_in_one(t, q)

        f = np.zeros(self.__nu)
        f[: self.nu1] = -r1_q.T @ f_pot
        f[self.nu1 :] = r2_q.T @ f_pot
        # f = np.zeros(self.nu + self.__nu)
        # f[self.xi_eta_DOF] = constr
        # f[self.subsystemsDOF[:self.nu1]] = -r1_q.T @ f_pot
        # f[self.subsystemsDOF[self.nu1:]] =  r2_q.T @ f_pot
        return f

    def f_npot_q(self, t, q, u, coo):
        # TODO: Derive analytical derivative. This requires the derivatives of
        #       xi_c, eta_c w.r.t. changes of q's. See Meier2017, Appendix C
        #       for a derivation.

        # reuse old jacobian if desired, otherwise recompute the correct one
        self.jacobian_updates += 1
        if (self.jacobian_updates >= self.n_jacobian_update) or (
            self.f_npot_q_num is None
        ):
            self.f_npot_q_num = Numerical_derivative(self.f_npot)._x(t, q, u)
            self.jacobian_updates = 0
        coo.extend(self.f_npot_q_num, (self.uDOF, self.qDOF))

    ###############
    # visualization
    ###############
    def contact_points(self, t, q):
        q_subsystems = q[self.qDOF]

        xi_c, eta_c = self.closest_points(t, q_subsystems)

        r1 = self.r1(t, q_subsystems, xi_c)
        r2 = self.r2(t, q_subsystems, eta_c)

        # return r1, r2
        return np.array([[r1[0], r2[0]], [r1[1], r2[1]], [r1[2], r2[2]]])
