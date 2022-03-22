import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from cardillo.model.frame.frame import Frame
from cardillo.model.rigid_body import RigidBodyEuler
from cardillo.model.bilateral_constraints.implicit import SphericalJoint
from cardillo.math.algebra import cross3, ax2skew
from cardillo.math import approx_fprime
from cardillo.model import Model
from cardillo.solver import (
    ScipyIVP,
    Moreau,
    EulerBackward,
    GenAlphaFirstOrderVelocity,
    GenAlphaFirstOrderVelocityGGL,
)


class HeavyTop2(RigidBodyEuler):
    def __init__(self, A, B, grav, q0=None, u0=None):
        self.grav = grav
        self.r_OQ = r_OQ

        # initialize rigid body
        self.K_Theta_S = np.diag([A, A, B])
        RigidBodyEuler.__init__(self, m, self.K_Theta_S, axis="zxz", q0=q0, u0=u0)

        # gravity
        self.f_g = np.array([0, 0, -self.m * self.grav])

    def f_pot(self, t, q):
        return self.f_g @ self.J_P(t, q)

    def f_pot_q(self, t, q, coo):
        dense = np.einsum("i,ijk->jk", self.f_g, self.J_P_q(t, q))
        coo.extend(dense, (self.uDOF, self.qDOF))


class HeavyTop:
    def __init__(self, m, l, A, B, grav, r_OQ, q0=None, u0=None, la_g0=None):
        self.m = m
        self.l = l
        self.A = A
        self.B_ = B
        self.grav = grav
        self.r_OQ = r_OQ

        self.K_Theta_S = np.diag([A, A, B])
        self.f_g = np.array([0, 0, -m * grav])

        self.nq = 6
        self.nu = 6
        self.nla_g = 3
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

    def assembler_callback(self):
        t0 = self.t0
        # compute K_r_SQ for initial configuration
        self.K_r_SQ = self.A_IK(t0, q0).T @ (self.r_OQ - self.r_OP(t0, q0))

    def qDOF_P(self, frame_ID=None):
        return np.arange(self.nq)

    def uDOF_P(self, frame_ID=None):
        return np.arange(self.nu)

    @staticmethod
    def A_IK(t, q, frame_ID=None):
        alpha, beta, gamma = q[3:]
        sa, ca = sin(alpha), cos(alpha)
        sb, cb = sin(beta), cos(beta)
        sg, cg = sin(gamma), cos(gamma)
        # fmt: off
        return np.array([
            [ca * cg - sa * cb * sg, - ca * sg - sa * cb * cg, sa * sb],
            [sa * cg + ca * cb * sg, -sa * sg + ca * cb * cg, -ca * sb],
            [sb * sg, sb * cg, cb]
        ])
        # fmt: on

    # TODO!
    def A_IK_q(self, t, q, frame_ID=None):
        return approx_fprime(q, lambda q: self.A_IK(t, q, frame_ID), method="3-point")

    # def A_IK_q(self, t, q, frame_ID=None):
    #     A_IK_q = np.zeros((3, 3, self.nq))
    #     A_IK_q[:, :, 3] = self.dA_I1(q) @ self.A_12(q) @ self.A_2K(q)
    #     A_IK_q[:, :, 4] = self.A_I1(q) @ self.dA_12(q) @ self.A_2K(q)
    #     A_IK_q[:, :, 5] = self.A_I1(q) @ self.A_12(q) @ self.dA_2K(q)
    #     return A_IK_q

    def r_OP(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return q[:3] + self.A_IK(t, q) @ K_r_SP

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        r_OP_q = np.zeros((3, self.nq))
        r_OP_q[:, :3] = np.eye(3)
        r_OP_q[:, :] += np.einsum("ijk,j->ik", self.A_IK_q(t, q), K_r_SP)
        return r_OP_q

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return u[:3] + self.A_IK(t, q) @ cross3(u[3:], K_r_SP)

    def v_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.einsum("ijk,j->ik", self.A_IK_q(t, q), cross3(u[3:], K_r_SP))

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return u_dot[:3] + self.A_IK(t, q) @ (
            cross3(u_dot[3:], K_r_SP) + cross3(u[3:], cross3(u[3:], K_r_SP))
        )

    def a_P_q(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.einsum(
            "ijk,j->ik",
            self.A_IK_q(t, q),
            cross3(u_dot[3:], K_r_SP) + cross3(u[3:], cross3(u[3:], K_r_SP)),
        )

    def a_P_u(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.kappa_P_u(t, q, u, frame_ID=frame_ID, K_r_SP=K_r_SP)

    def kappa_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.A_IK(t, q) @ (cross3(u[3:], cross3(u[3:], K_r_SP)))

    def kappa_P_q(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return np.einsum(
            "ijk,j->ik", self.A_IK_q(t, q), cross3(u[3:], cross3(u[3:], K_r_SP))
        )

    def kappa_P_u(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        kappa_P_u = np.zeros((3, self.nu))
        kappa_P_u[:, 3:] = -self.A_IK(t, q) @ (
            ax2skew(cross3(u[3:], K_r_SP)) + ax2skew(u[3:]) @ ax2skew(K_r_SP)
        )
        return kappa_P_u

    def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        J_P = np.zeros((3, self.nu))
        J_P[:, :3] = np.eye(3)
        J_P[:, 3:] = -self.A_IK(t, q) @ ax2skew(K_r_SP)
        return J_P

    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        J_P_q = np.zeros((3, self.nu, self.nq))
        J_P_q[:, 3:, :] = np.einsum("ijk,jl->ilk", self.A_IK_q(t, q), -ax2skew(K_r_SP))
        return J_P_q

    def K_Omega(self, t, q, u, frame_ID=None):
        return u[3:]

    def K_Omega_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq))

    def K_Psi(self, t, q, u, u_dot, frame_ID=None):
        return u_dot[3:]

    def K_kappa_R(self, t, q, u, frame_ID=None):
        return np.zeros(3)

    def K_kappa_R_q(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nq))

    def K_kappa_R_u(self, t, q, u, frame_ID=None):
        return np.zeros((3, self.nu))

    def K_J_R(self, t, q, frame_ID=None):
        J_R = np.zeros((3, self.nu))
        J_R[:, 3:] = np.eye(3)
        return J_R

    def K_J_R_q(self, t, q, frame_ID=None):
        return np.zeros((3, self.nu, self.nq))

    #################################
    # kinematic differential equation
    #################################
    def Q(self, t, q_angles):
        _, beta, gamma = q_angles
        sb, cb = sin(beta), cos(beta)
        sg, cg = sin(gamma), cos(gamma)
        # fmt: off
        return np.array([
            [sg, cg, 0],
            [sb * cg, -sb * sg, 0],
            [-cb * sg, -cb * cg, sb]
        ]) / sb
        # fmt: on

    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.nq)
        q_dot[:3] = u[:3]
        q_dot[3:] = self.Q(t, q[3:]) @ u[3:]
        return q_dot

    # TODO!
    def q_ddot(self, t, q, u, u_dot):
        q_ddot = np.zeros(self.nq)
        q_ddot[:3] = u_dot[:3]
        q_ddot[3:] = self.Q(t, q[3:]) @ u_dot[3:]
        q_dot_q = approx_fprime(q, lambda q: self.q_dot(t, q, u), method="3-point")
        q_ddot += q_dot_q @ self.q_dot(t, q, u)
        return q_ddot

    # TODO!
    def q_dot_q(self, t, q, u, coo):
        dense = approx_fprime(q, lambda q: self.q_dot(t, q, u), method="3-point")
        coo.extend(dense, (self.qDOF, self.qDOF))

    def B(self, t, q, coo):
        B = np.zeros((self.nq, self.nu))
        B[:3, :3] = np.eye(3)
        B[3:, 3:] = self.Q(t, q[3:])
        coo.extend(B, (self.qDOF, self.uDOF))

    ##################
    # potential forces
    ##################
    def f_pot(self, t, q):
        return self.f_g @ self.J_P(t, q)

    def f_pot_q(self, t, q, coo):
        dense = np.einsum("i,ijk->jk", self.f_g, self.J_P_q(t, q))
        coo.extend(dense, (self.uDOF, self.qDOF))

    ################
    # inertia forces
    ################
    def M_dense(self, t, q):
        return np.diag([self.m, self.m, self.m, self.A, self.A, self.B_])

    def M(self, t, q, coo):
        coo.extend(self.M_dense(t, q), (self.uDOF, self.uDOF))

    def f_gyr(self, t, q, u):
        K_Omega = u[3:]
        f = np.zeros(self.nu)
        f[3:] = cross3(K_Omega, self.K_Theta_S @ K_Omega)
        return f

    def f_gyr_u(self, t, q, u, coo):
        K_Omega = u[3:]
        dense = np.zeros((self.nu, self.nu))
        dense[3:, 3:] = ax2skew(K_Omega) @ self.K_Theta_S - ax2skew(
            self.K_Theta_S @ K_Omega
        )
        coo.extend(dense, (self.uDOF, self.uDOF))

    #######################
    # bilateral constraints
    #######################
    def g(self, t, q):
        return self.r_OP(t, q, K_r_SP=self.K_r_SQ) - self.r_OQ

    def g_dot(self, t, q, u):
        return self.v_P(t, q, u, K_r_SP=self.K_r_SQ)

    def g_dot_u(self, t, q, coo):
        coo.extend(self.J_P(t, q, K_r_SP=self.K_r_SQ), (self.la_gDOF, self.qDOF))

    def g_ddot(self, t, q, u, u_dot):
        return self.a_P(t, q, u, u_dot, K_r_SP=self.K_r_SQ)

    def g_q_dense(self, t, q):
        return self.r_OP_q(t, q, K_r_SP=self.K_r_SQ)

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g_dense(self, t, q):
        return self.J_P(t, q, K_r_SP=self.K_r_SQ).T

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        dense = np.einsum("ijk,i->jk", self.J_P_q(t, q, K_r_SP=self.K_r_SQ), la_g)
        coo.extend(dense, (self.uDOF, self.qDOF))

    ############################################
    # Identify Lagrange multipliers a posteriori
    ############################################
    def la_g(self, t, q, u):
        W = self.W_g_dense(t, q)
        M = self.M_dense(t, q)
        G = W.T @ np.linalg.solve(M, W)
        # G = W.T @ np.linalg.inv(M) @ W
        zeta = self.g_ddot(t, q, u, np.zeros_like(u))
        h = self.f_pot(t, q) - self.f_gyr(t, q, u)
        mu = zeta + W.T @ np.linalg.solve(M, h)
        return np.linalg.solve(G, -mu)

    #########################
    # minimal ode formulation
    #########################
    def __call__(self, t, x):
        q = x[:3]
        u = x[3:]

        # mass matrix
        A_ = self.A + self.m * self.l**2
        M_inv = np.diag([1 / A_, 1 / A_, 1 / self.B_])

        # h vector
        _, beta, gamma = q
        sb, cb = sin(beta), cos(beta)
        sg, cg = sin(gamma), cos(gamma)
        omega_x, omega_y, omega_z = u
        m, l, g, A, B = self.m, self.l, self.grav, self.A, self.B_
        Theta1 = A
        Theta2 = A
        Theta3 = B
        h = np.array(
            [
                (m * l**2 + Theta2 - Theta3) * omega_y * omega_z,
                (-m * l**2 + Theta3 - Theta1) * omega_x * omega_z,
                (Theta1 - Theta2) * omega_x * omega_y,
            ]
        ) + m * g * l * np.array([sb * cg, -sb * sg, 0])
        return np.concatenate([self.Q(t, q) @ u, M_inv @ h])


if __name__ == "__main__":
    ###################
    # system parameters
    ###################
    m = 0.1
    l = 0.2
    grav = 9.81
    r = 0.1
    A = 1 / 2 * m * r**2
    B = 1 / 4 * m * r**2
    alpha0 = 0
    beta0 = pi / 2
    gamma0 = 0
    omega_x0 = 0
    omega_y0 = 0
    omega_z0 = 2 * pi * 50

    # #####################
    # # Geradin2000, p 103.
    # #####################
    # m = 5
    # A = 0.8
    # B = 1.8
    # l = 1.3
    # grav = 9.81
    # alpha0 = 0
    # beta0 = pi / 9
    # gamma0 = 0
    # omega_x0 = 0
    # omega_y0 = 0
    # omega_z0 = 2 * pi * 50

    # #####################
    # # Arnold2015, p 174.
    # #####################
    # m = 15
    # A = 0.234375
    # B = 0.46875
    # l = 1.0
    # grav = 9.81
    # alpha0 = 0
    # beta0 = pi / 2
    # gamma0 = 0
    # omega_x0 = 0
    # omega_x0 = 4.61538
    # omega_y0 = 0
    # # omega_z0 = 150
    # omega_z0 = 300

    #############################
    # initial position and angles
    #############################
    phi0 = np.array([alpha0, beta0, gamma0])

    r_OQ = np.zeros(3)
    K_r_OS0 = np.array([0, 0, l])
    A_IK0 = HeavyTop.A_IK(0, [0, 0, 0, alpha0, beta0, gamma0])
    r_OS0 = A_IK0 @ K_r_OS0

    q0 = np.concatenate((r_OS0, phi0))

    ####################
    # initial velocities
    ####################
    K_Omega0 = np.array([omega_x0, omega_y0, omega_z0])
    v_S0 = A_IK0 @ cross3(K_Omega0, K_r_OS0)
    u0 = np.concatenate((v_S0, K_Omega0))

    ###########################
    # initial constraint forces
    # TODO: g_ddot is not zero for this initial state!
    ###########################

    # 1. hand written version
    top1 = HeavyTop(m, l, A, B, grav, r_OQ, q0, u0)
    model = Model()
    model.add(top1)
    model.assemble()

    # 2. reuse existing RigidBodyEuler and SphericalJoint
    top2 = HeavyTop2(A, B, grav, q0, u0)
    frame = Frame()
    spherical_joint = SphericalJoint(frame, top2, np.zeros(3))
    model = Model()
    model.add(top2)
    model.add(frame)
    model.add(spherical_joint)
    model.assemble()

    # end time and numerical dissipation of generalized-alpha solver
    # t1 = 5
    # t1 = 5 / 4
    # t1 = 5 / 8
    # t1 = 1
    t1 = 0.1
    # t1 = 0.01
    # rho_inf = 0.0
    # rho_inf = 0.3
    rho_inf = 0.9
    # rho_inf = 1.0

    # log spaced time steps
    num = 1
    # dts = np.logspace(-1, -num, num=num, endpoint=True)
    # dts = np.array([2.5e-3])
    dts = np.array([1e-4])
    dts_1 = dts
    dts_2 = dts**2
    print(f"dts: {dts}")

    # TODO: Compare error with theta method
    q_errors = np.inf * np.ones((4, 6, len(dts)), dtype=float)
    u_errors = np.inf * np.ones((4, 6, len(dts)), dtype=float)
    la_g_errors = np.inf * np.ones((4, 6, len(dts)), dtype=float)

    for i, dt in enumerate(dts):
        print(f"i: {i}, dt: {dt:1.1e}")

        # # solve with theta-method
        # # sol_ThetaNewton = ThetaNewton(model, t1, dt).solve()
        # sol_ThetaNewton = MoreauTheta(model, t1, dt).solve()
        # t_ThetaNewton = sol_ThetaNewton.t
        # q_ThetaNewton = sol_ThetaNewton.q
        # u_ThetaNewton = sol_ThetaNewton.u
        # la_g_ThetaNewton = sol_ThetaNewton.la_g

        # # solve with first order generalized-alpha method
        # sol_GenAlphaFirstOrder = GenAlphaFirstOrderVelocity(model, t1, dt, rho_inf=rho_inf).solve()
        # t_GenAlphaFirstOrder = sol_GenAlphaFirstOrder.t
        # q_GenAlphaFirstOrder = sol_GenAlphaFirstOrder.q
        # u_GenAlphaFirstOrder = sol_GenAlphaFirstOrder.u
        # la_g_GenAlphaFirstOrder = sol_GenAlphaFirstOrder.la_g

        # # solve with second order generalized-alpha method velocity implementation
        # sol_GenAlphaSecondOrder = GenAlphaDAEAcc(model, t1, dt, rho_inf=rho_inf).solve()
        # t_GenAlphaSecondOrder = sol_GenAlphaSecondOrder.t
        # q_GenAlphaSecondOrder = sol_GenAlphaSecondOrder.q
        # u_GenAlphaSecondOrder = sol_GenAlphaSecondOrder.u
        # la_g_GenAlphaSecondOrder = sol_GenAlphaSecondOrder.la_g

        # yet implemented solvers
        # sol_GenAlphaFirstOrderGGl = ScipyIVP(model, t1, dt).solve()
        # sol_GenAlphaFirstOrderGGl = Moreau(model, t1, dt).solve()
        # sol_GenAlphaFirstOrderGGl = EulerBackward(model, t1, dt).solve()
        sol_GenAlphaFirstOrderGGl = GenAlphaFirstOrderVelocity(model, t1, dt).solve()

        # solve with generalzed-alpha method positin implementation
        # sol_GenAlphaFirstOrderGGl = GenAlphaFirstOrderVelocityGGL(model, t1, dt, rho_inf=rho_inf).solve()
        # sol_GenAlphaFirstOrderGGl = GenAlphaFirstOrderVelocity(model, t1, dt, rho_inf=rho_inf).solve()
        # sol_GenAlphaFirstOrderGGl = GenAlphaFirstOrderPosition(model, t1, dt, rho_inf=rho_inf).solve()
        # sol_GenAlphaFirstOrderGGl = GenAlphaDAEAcc(model, t1, dt, rho_inf=rho_inf).solve()
        # sol_GenAlphaFirstOrderGGl = Scipy_ivp(model, t1, dt).solve()
        # sol_GenAlphaFirstOrderGGl = ThetaNewton(model, t1, dt).solve()
        # sol_GenAlphaFirstOrderGGl = Euler_backward(model, t1, dt).solve()
        t_GenAlphaFirstOrderGGl = sol_GenAlphaFirstOrderGGl.t
        q_GenAlphaFirstOrderGGl = sol_GenAlphaFirstOrderGGl.q
        u_GenAlphaFirstOrderGGl = sol_GenAlphaFirstOrderGGl.u
        la_g_GenAlphaFirstOrderGGl = sol_GenAlphaFirstOrderGGl.la_g

        # compute true solution using Runge-Kutta 4(5) method of the ODE formulation
        t_span = np.array([0, t1])
        y0 = np.concatenate([phi0, K_Omega0])
        t_eval = np.linspace(0, t1, num=len(t_GenAlphaFirstOrderGGl))
        method = "RK45"
        # method = "DOP853"
        # method = "BDF"
        sol_RK45 = solve_ivp(
            top1, t_span, y0, method=method, t_eval=t_eval, atol=1.0e-12, rtol=1.0e-12
        )
        t_RK45 = sol_RK45.t
        y_RK45 = sol_RK45.y
        # TODO: Compute Lagrange multipliers
        # q_RK45 = cartesian_coordinates(phi_RK45).T
        # u_RK45 = cartesian_coordinates_dot(phi_RK45, omega_RK45).T
        # la_g_RK4 = np.array([la_g_analytic(t, phi, omega) for (t, phi, omega) in zip(t_RK45, phi_RK45, omega_RK45)])

    #     # compute errors
    #     # x_y_errors[0, i] = np.linalg.norm(q_GenAlphaFirstOrder - q_RK45)
    #     # x_y_errors[1, i] = np.linalg.norm(q_GenAlphaSecondOrder - q_RK45)
    #     q_errors[2, i] = np.linalg.norm(q_GenAlphaFirstOrderGGl - q_RK45)
    #     # x_y_errors[3, i] = np.linalg.norm(q_ThetaNewton - q_RK45)
    #     # x_dot_y_dot_errors[0, i] = np.linalg.norm(u_GenAlphaFirstOrder - u_RK45)
    #     # x_dot_y_dot_errors[1, i] = np.linalg.norm(u_GenAlphaSecondOrder - u_RK45)
    #     u_errors[2, i] = np.linalg.norm(u_GenAlphaFirstOrderGGl - u_RK45)
    #     # x_dot_y_dot_errors[3, i] = np.linalg.norm(u_ThetaNewton - u_RK45)
    #     # la_g_errors[0, i] = np.linalg.norm(la_g_GenAlphaFirstOrder - la_g_RK4)
    #     # la_g_errors[1, i] = np.linalg.norm(la_g_GenAlphaSecondOrder - la_g_RK4)
    #     la_g_errors[2, i] = np.linalg.norm(la_g_GenAlphaFirstOrderGGl - la_g_RK4)
    #     # la_g_errors[3, i] = np.linalg.norm(la_g_ThetaNewton - la_g_RK4)

    # # names = ["GenAlphaFirstOrder", "GenAlphaSecondOrder", "GenAlphaFirstOrderGGl", "Theta"]
    # # ts = [t_GenAlphaFirstOrder, t_GenAlphaSecondOrder, t_GenAlphaFirstOrderGGl, t_ThetaNewton]
    # # qs = [q_GenAlphaFirstOrder, q_GenAlphaSecondOrder, q_GenAlphaFirstOrderGGl, q_ThetaNewton]
    # # us = [u_GenAlphaFirstOrder, u_GenAlphaSecondOrder, u_GenAlphaFirstOrderGGl, u_ThetaNewton]
    # # la_gs = [la_g_GenAlphaFirstOrder, la_g_GenAlphaSecondOrder, la_g_GenAlphaFirstOrderGGl, la_g_ThetaNewton]
    # names = ["GenAlphaFirstOrderGGl"]
    # ts = [t_GenAlphaFirstOrderGGl]
    # qs = [q_GenAlphaFirstOrderGGl]
    # us = [u_GenAlphaFirstOrderGGl]
    # la_gs = [la_g_GenAlphaFirstOrderGGl]
    # for i, name in enumerate(names):
    #     filename = "sol_" + name + "_cartesian_pendulum_.txt"
    #     # export_data = np.hstack((t_GenAlphaFirstOrderGGl[:, np.newaxis], q_GenAlphaFirstOrderGGl, u_GenAlphaFirstOrderGGl, la_g_GenAlphaFirstOrderGGl))
    #     export_data = np.hstack((ts[i][:, np.newaxis], qs[i], us[i], la_gs[i]))
    #     header = "t, x, y, x_dot, y_dot, la_g"
    #     np.savetxt(filename, export_data, delimiter=", ", header=header, comments="")

    #     filename = "error_" + name + "_cartesian_pendulum_.txt"
    #     header = "dt, dt2, error_xy, error_xy_dot, error_la_g"
    #     export_data = np.vstack((dts, dts_2, q_errors[i], u_errors[i], la_g_errors[i])).T
    #     np.savetxt(filename, export_data, delimiter=", ", header=header, comments="")

    # visualize results
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # postprocessing of RK45 solution
    def postporcessing(t, y):
        r_OS = []
        v_S = []
        la_g = []
        for i, (ti, yi) in enumerate(zip(t, y)):
            angles = yi[:3]
            omegas = yi[3:]
            r_OSi = top1.r_OP(
                t, np.array([0, 0, 0, *angles]), K_r_SP=np.array([0, 0, l])
            )
            r_OS.append(r_OSi)

            A_IK = HeavyTop.A_IK(ti, [0, 0, 0, *angles])
            v_Si = A_IK @ cross3(omegas, np.array([0, 0, l]))
            # v_Si = top.v_P(ti, [0, 0, 0, *angles], [0, 0, 0, *omegas], K_r_SP=np.array([0, 0, l]))
            v_S.append(v_Si)

            qi = np.concatenate((r_OSi, angles))
            ui = np.concatenate((v_Si, omegas))
            la_g.append(top1.la_g(ti, qi, ui))
        return np.array(r_OS), np.array(v_S), np.array(la_g)

    r_OS_RK45, v_S_RK45, la_g_RK45 = postporcessing(t_RK45, y_RK45.T)

    # center of mass
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        q_GenAlphaFirstOrderGGl[:, 0],
        "-r",
        label="x - GenAlphaFirstOrderGGl",
    )
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        q_GenAlphaFirstOrderGGl[:, 1],
        "-g",
        label="y - GenAlphaFirstOrderGGl",
    )
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        q_GenAlphaFirstOrderGGl[:, 2],
        "-b",
        label="z - GenAlphaFirstOrderGGl",
    )
    ax.plot(t_RK45, r_OS_RK45[:, 0], "--r", label="x - RK45")
    ax.plot(t_RK45, r_OS_RK45[:, 1], "--g", label="y - RK45")
    ax.plot(t_RK45, r_OS_RK45[:, 2], "--b", label="z - RK45")
    ax.grid()
    ax.legend()

    # alpha, beta, gamma
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        q_GenAlphaFirstOrderGGl[:, 3],
        "-r",
        label="alpha - GenAlphaFirstOrderGGl",
    )
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        q_GenAlphaFirstOrderGGl[:, 4],
        "-g",
        label="beta - GenAlphaFirstOrderGGl",
    )
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        q_GenAlphaFirstOrderGGl[:, 5],
        "-b",
        label="gamm - GenAlphaFirstOrderGGl",
    )
    ax.plot(t_RK45, y_RK45[0, :], "--r", label="alpha - RK45")
    ax.plot(t_RK45, y_RK45[1, :], "--g", label="beta - RK45")
    ax.plot(t_RK45, y_RK45[2, :], "--b", label="gamm - RK45")
    ax.grid()
    ax.legend()

    # x-y-z trajectory
    ax = fig.add_subplot(2, 3, 3, projection="3d")
    ax.plot3D(
        q_GenAlphaFirstOrderGGl[:, 0],
        q_GenAlphaFirstOrderGGl[:, 1],
        q_GenAlphaFirstOrderGGl[:, 2],
        "-b",
        label="x-y-z trajectory - GenAlphaFirstOrderGGl",
    )
    ax.plot3D(
        r_OS_RK45[:, 0],
        r_OS_RK45[:, 1],
        r_OS_RK45[:, 2],
        "--r",
        label="x-y-z trajectory - RK45",
    )
    ax.grid()
    ax.legend()

    # x_dot, y_dot, z_dot
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        u_GenAlphaFirstOrderGGl[:, 0],
        "-r",
        label="x_dot - GenAlphaFirstOrderGGl",
    )
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        u_GenAlphaFirstOrderGGl[:, 1],
        "-g",
        label="y_dot - GenAlphaFirstOrderGGl",
    )
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        u_GenAlphaFirstOrderGGl[:, 2],
        "-b",
        label="z_dot - GenAlphaFirstOrderGGl",
    )
    ax.plot(t_RK45, v_S_RK45[:, 0], "--r", label="x_dot - RK45")
    ax.plot(t_RK45, v_S_RK45[:, 1], "--g", label="y_dot - RK45")
    ax.plot(t_RK45, v_S_RK45[:, 2], "--b", label="z_dot - RK45")
    ax.grid()
    ax.legend()

    # omega_x, omega_y, omega_z
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        u_GenAlphaFirstOrderGGl[:, 3],
        "-r",
        label="omega_x - GenAlphaFirstOrderGGl",
    )
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        u_GenAlphaFirstOrderGGl[:, 4],
        "-g",
        label="omega_y - GenAlphaFirstOrderGGl",
    )
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        u_GenAlphaFirstOrderGGl[:, 5],
        "-b",
        label="omega_z - GenAlphaFirstOrderGGl",
    )
    ax.plot(t_RK45, y_RK45[3, :], "--r", label="omega_x - RK45")
    ax.plot(t_RK45, y_RK45[4, :], "--g", label="omega_y - RK45")
    ax.plot(t_RK45, y_RK45[5, :], "--b", label="omega_z - RK45")
    ax.grid()
    ax.legend()

    # la_g
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        la_g_GenAlphaFirstOrderGGl[:, 0],
        "-r",
        label="la_g0 - GenAlphaFirstOrderGGl",
    )
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        la_g_GenAlphaFirstOrderGGl[:, 1],
        "-g",
        label="la_g1 - GenAlphaFirstOrderGGl",
    )
    ax.plot(
        t_GenAlphaFirstOrderGGl,
        la_g_GenAlphaFirstOrderGGl[:, 2],
        "-b",
        label="la_g2 - GenAlphaFirstOrderGGl",
    )
    ax.plot(t_RK45, la_g_RK45[:, 0], "--r", label="la_g0 - RK45")
    ax.plot(t_RK45, la_g_RK45[:, 1], "--g", label="la_g1 - RK45")
    ax.plot(t_RK45, la_g_RK45[:, 2], "--b", label="la_g2 - RK45")
    ax.grid()
    ax.legend()

    plt.show()
    exit()

    # generalized velocities
    # ax[0, 1].plot(t_GenAlphaFirstOrder, u_GenAlphaFirstOrder[:, 0], 'xb', label="x_dot - GenAlphaFirstOrder")
    # ax[0, 1].plot(t_GenAlphaFirstOrder, u_GenAlphaFirstOrder[:, 1], 'ob', label="y_dot - GenAlphaFirstOrder")
    # ax[0, 1].plot(t_GenAlphaSecondOrder, u_GenAlphaSecondOrder[:, 0], 'xr', label="x_dot - GenAlphaSecondOrder")
    # ax[0, 1].plot(t_GenAlphaSecondOrder, u_GenAlphaSecondOrder[:, 1], 'or', label="y_dot - GenAlphaSecondOrder")
    ax[0, 1].plot(
        t_GenAlphaFirstOrderGGl,
        u_GenAlphaFirstOrderGGl[:, 0],
        "xg",
        label="x_dot - GenAlphaFirstOrderGGl",
    )
    ax[0, 1].plot(
        t_GenAlphaFirstOrderGGl,
        u_GenAlphaFirstOrderGGl[:, 1],
        "og",
        label="y_dot - GenAlphaFirstOrderGGl",
    )
    # ax[0, 1].plot(t_ThetaNewton, u_ThetaNewton[:, 0], 'xm', label="x_dot - Theta")
    # ax[0, 1].plot(t_ThetaNewton, u_ThetaNewton[:, 1], 'om', label="y_dot - Theta")
    ax[0, 1].plot(t_RK45, u_RK45[:, 0], "-k", label="x_dot - RK45")
    ax[0, 1].plot(t_RK45, u_RK45[:, 1], "--k", label="y_dot - RK45")
    ax[0, 1].grid()
    ax[0, 1].legend()

    # Lagrange multipliers
    # ax[0, 2].plot(t_GenAlphaFirstOrder, la_g_GenAlphaFirstOrder[:, 0], 'ob', label="la_g - GenAlphaFirstOrder")
    # ax[0, 2].plot(t_GenAlphaSecondOrder, la_g_GenAlphaSecondOrder[:, 0], 'xr', label="la_g - GenAlphaSecondOrder")
    ax[0, 2].plot(
        t_GenAlphaFirstOrderGGl,
        la_g_GenAlphaFirstOrderGGl[:, 0],
        "sg",
        label="la_g - GenAlphaFirstOrderGGl",
    )
    # ax[0, 2].plot(t_ThetaNewton, la_g_ThetaNewton[:, 0], 'sm', label="la_g - Theta")
    ax[0, 2].plot(t_RK45, la_g_RK4, "-k", label="la_g - RK45")
    ax[0, 2].grid()
    ax[0, 2].legend()

    # (x,y) - errors
    # ax[1, 0].loglog(dts, x_y_errors[0], '-b', label="(x,y) - error - GenAlphaFirstOrder")
    # ax[1, 0].loglog(dts, x_y_errors[1], '--r', label="(x,y) - error - GenAlphaSecondOrder")
    ax[1, 0].loglog(
        dts, q_errors[2], "-.g", label="(x,y) - error - GenAlphaFirstOrderGGl"
    )
    # ax[1, 0].loglog(dts, x_y_errors[3], '--m', label="(x,y) - error - Theta")
    ax[1, 0].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 0].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 0].grid()
    ax[1, 0].legend()

    # (x_dot,y_dot) - errors
    # ax[1, 1].loglog(dts, x_dot_y_dot_errors[0], '-b', label="(x_dot,y_dot) - error - GenAlphaFirstOrder")
    # ax[1, 1].loglog(dts, x_dot_y_dot_errors[1], '--r', label="(x_dot,y_dot) - error - GenAlphaSecondOrder")
    ax[1, 1].loglog(
        dts, u_errors[2], "-.g", label="(x_dot,y_dot) - error - GenAlphaFirstOrderGGl"
    )
    # ax[1, 1].loglog(dts, x_dot_y_dot_errors[3], '--m', label="(x_dot,y_dot) - error - Theta")
    ax[1, 1].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 1].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 1].grid()
    ax[1, 1].legend()

    # TODO: la_g - errors
    # ax[1, 2].loglog(dts, la_g_errors[0], '-b', label="la_g - error - GenAlphaFirstOrder")
    # ax[1, 2].loglog(dts, la_g_errors[1], '--r', label="la_g - error - GenAlphaSecondOrder")
    ax[1, 2].loglog(
        dts, la_g_errors[2], "-.g", label="la_g - error - GenAlphaFirstOrderGGl"
    )
    # ax[1, 2].loglog(dts, la_g_errors[3], '-.m', label="la_g - error - Theta")
    ax[1, 2].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 2].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 2].grid()
    ax[1, 2].legend()

    plt.show()
