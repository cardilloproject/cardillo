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
from cardillo.solver import GenAlphaFirstOrder, GenAlphaFirstOrderGGL2_V3


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
        zeta = self.g_ddot(t, q, u, np.zeros_like(u))
        h = self.f_pot(t, q) - self.f_gyr(t, q, u)
        mu = zeta + W.T @ np.linalg.solve(M, h)
        return np.linalg.solve(G, -mu)

    def postprocessing(self, t, y):
        angles = y[:3]
        omegas = y[3:]
        q_ = np.array([0, 0, 0, *angles])
        K_r_SP = np.array([0, 0, self.l])

        r_OS = self.r_OP(t, q_, K_r_SP=K_r_SP)
        v_S = self.A_IK(t, q_) @ cross3(omegas, K_r_SP)

        q = np.array([*r_OS, *angles])
        u = np.array([*v_S, *omegas])
        la_g = top1.la_g(t, q, u)

        return q, u, la_g

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
        sb, _ = sin(beta), cos(beta)
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
    # ###################
    # # system parameters
    # ###################
    # m = 0.1
    # l = 0.2
    # grav = 9.81
    # r = 0.1
    # A = 1 / 2 * m * r**2
    # B = 1 / 4 * m * r**2
    # alpha0 = 0
    # beta0 = pi / 2
    # gamma0 = 0
    # omega_x0 = 0
    # omega_y0 = 0
    # omega_z0 = 2 * pi * 50

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

    ########################################
    # Arnold2015, p. 174/ Arnold2015b, p. 13
    ########################################
    m = 15
    A = 0.234375
    B = 0.46875
    l = 1.0
    grav = 9.81
    alpha0 = 0
    beta0 = pi / 2
    gamma0 = 0

    omega_x0 = 0
    # omega_y0 = 0 # Arnodl2015 p. 174
    omega_y0 = -4.61538  # Arnold2015b p. 13
    omega_z0 = 150
    # omega_z0 *= ??? #  this yields a precession movement

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

    # 1. hand written version
    top1 = HeavyTop(m, l, A, B, grav, r_OQ, q0, u0)
    model = Model()
    model.add(top1)
    model.assemble()

    # # 2. reuse existing RigidBodyEuler and SphericalJoint
    # frame = Frame()
    # top2 = HeavyTop2(A, B, grav, q0, u0)
    # spherical_joint = SphericalJoint(frame, top2, np.zeros(3))
    # model = Model()
    # model.add(top2)
    # model.add(frame)
    # model.add(spherical_joint)
    # model.assemble()

    # end time and numerical dissipation of generalized-alpha solver
    # t1 = 5
    # t1 = 5 / 4
    # t1 = 5 / 8
    # t1 = 1
    # t1 = 0.25
    # t1 = 1
    t1 = 0.1  # this is used for investigating the transient bahavior
    rho_inf = 0.9

    # log spaced time steps
    # num = 3
    # dts = np.logspace(-1, -num, num=num, endpoint=True)
    # dts = np.logspace(-2, -num, num=num - 1, endpoint=True)
    dts = np.array([1.0e-2])
    # dts = np.array([1e-2, 1.0e-3])
    # dts = np.array([1e-2, 1e-3, 1e-4])
    dts_1 = dts
    dts_2 = dts**2
    print(f"dts: {dts}")

    # errors for all 6 possible solvers
    q_errors = np.inf * np.ones((6, len(dts)), dtype=float)
    u_errors = np.inf * np.ones((6, len(dts)), dtype=float)
    la_g_errors = np.inf * np.ones((6, len(dts)), dtype=float)

    # compute reference solution as described in Arnold2015 Section 3.3
    print(f"compute reference solution:")
    # dt_ref = 2.5e-5 # see Arnold2015 p. 174/ Arnodl2015b p. 14
    # dt_ref = 1.0e-4
    dt_ref = 1.0e-3
    # reference = GenAlphaFirstOrder(model, t1, dt_ref, rho_inf=rho_inf, unknowns="auxiliary", GGL=True).solve()
    reference = GenAlphaFirstOrderGGL2_V3(
        model, t1, dt_ref, rho_inf=rho_inf, unknowns="auxiliary"
    ).solve()
    t_ref = reference.t
    q_ref = reference.q
    u_ref = reference.u
    la_g_ref = reference.la_g

    plot_state = True
    # plot_state = False
    if plot_state:
        ###################
        # visualize results
        ###################
        fig = plt.figure(figsize=plt.figaspect(0.5))

        # center of mass
        ax = fig.add_subplot(2, 3, 1)
        ax.plot(t_ref, q_ref[:, 0], "-r", label="x")
        ax.plot(t_ref, q_ref[:, 1], "-g", label="y")
        ax.plot(t_ref, q_ref[:, 2], "-b", label="z")
        ax.grid()
        ax.legend()

        # alpha, beta, gamma
        ax = fig.add_subplot(2, 3, 2)
        ax.plot(t_ref, q_ref[:, 3], "-r", label="alpha")
        ax.plot(t_ref, q_ref[:, 4], "-g", label="beta")
        ax.plot(t_ref, q_ref[:, 5], "-b", label="gamm")
        ax.grid()
        ax.legend()

        # x-y-z trajectory
        ax = fig.add_subplot(2, 3, 3, projection="3d")
        ax.plot3D(
            q_ref[:, 0],
            q_ref[:, 1],
            q_ref[:, 2],
            "-r",
            label="x-y-z trajectory",
        )
        ax.grid()
        ax.legend()

        # x_dot, y_dot, z_dot
        ax = fig.add_subplot(2, 3, 4)
        ax.plot(t_ref, u_ref[:, 0], "-r", label="x_dot")
        ax.plot(t_ref, u_ref[:, 1], "-g", label="y_dot")
        ax.plot(t_ref, u_ref[:, 2], "-b", label="z_dot")
        ax.grid()
        ax.legend()

        # omega_x, omega_y, omega_z
        ax = fig.add_subplot(2, 3, 5)
        ax.plot(t_ref, u_ref[:, 3], "-r", label="omega_x")
        ax.plot(t_ref, u_ref[:, 4], "-g", label="omega_y")
        ax.plot(t_ref, u_ref[:, 5], "-b", label="omega_z")
        ax.grid()
        ax.legend()

        # la_g
        ax = fig.add_subplot(2, 3, 6)
        ax.plot(t_ref, la_g_ref[:, 0], "-r", label="la_g0")
        ax.plot(t_ref, la_g_ref[:, 1], "-g", label="la_g1")
        ax.plot(t_ref, la_g_ref[:, 2], "-b", label="la_g2")
        ax.grid()
        ax.legend()

        plt.show()

    exit()

    def errors(sol):
        t = sol.t
        q = sol.q
        u = sol.u
        la_g = sol.la_g

        # compute difference between computed solution and reference solution
        # for identical time instants
        idx = np.where(np.abs(t[:, None] - t_ref) < 1.0e-8)[1]

        # differences
        diff_q = q - q_ref[idx]
        diff_u = u - u_ref[idx]
        diff_la_g = la_g - la_g_ref[idx]

        # relative error
        q_error = np.linalg.norm(diff_q) / np.linalg.norm(q)
        u_error = np.linalg.norm(diff_u) / np.linalg.norm(u)
        la_g_error = np.linalg.norm(diff_la_g) / np.linalg.norm(la_g)

        return q_error, u_error, la_g_error

    for i, dt in enumerate(dts):
        print(f"i: {i}, dt: {dt:1.1e}")

        # # position formulation
        # sol = GenAlphaFirstOrder(
        #     model, t1, dt, rho_inf=rho_inf, unknowns="positions"
        # ).solve()
        # q_errors[0, i], u_errors[0, i], la_g_errors[0, i] = errors(sol)

        # # velocity formulation
        # sol = GenAlphaFirstOrder(
        #     model, t1, dt, rho_inf=rho_inf, unknowns="velocities"
        # ).solve()
        # q_errors[1, i], u_errors[1, i], la_g_errors[1, i] = errors(sol)

        # # auxiliary formulation
        # sol = GenAlphaFirstOrder(
        #     model, t1, dt, rho_inf=rho_inf, unknowns="auxiliary"
        # ).solve()
        # q_errors[2, i], u_errors[2, i], la_g_errors[2, i] = errors(sol)

        # # GGL formulation - positions
        # sol = GenAlphaFirstOrder(
        #     model, t1, dt, rho_inf=rho_inf, unknowns="positions", GGL=True
        # ).solve()
        # q_errors[3, i], u_errors[3, i], la_g_errors[3, i] = errors(sol)

        # # GGL formulation - velocityies
        # sol = GenAlphaFirstOrder(
        #     model, t1, dt, rho_inf=rho_inf, unknowns="velocities", GGL=True
        # ).solve()
        # q_errors[4, i], u_errors[4, i], la_g_errors[4, i] = errors(sol)

        # # GGL formulation - auxiliary
        # sol = GenAlphaFirstOrder(
        #     model, t1, dt, rho_inf=rho_inf, unknowns="auxiliary", GGL=True
        # ).solve()
        # q_errors[5, i], u_errors[5, i], la_g_errors[5, i] = errors(sol)

    ##################
    # visualize errors
    ##################
    fig, ax = plt.subplots(2, 3)

    # errors position formulation
    ax[0, 0].loglog(dts, q_errors[0], "-.r", label="q")
    ax[0, 0].loglog(dts, u_errors[0], "-.g", label="u")
    ax[0, 0].loglog(dts, la_g_errors[0], "-.b", label="la_g")
    ax[0, 0].loglog(dts, dts_1, "-k", label="dt")
    ax[0, 0].loglog(dts, dts_2, "--k", label="dt^2")
    ax[0, 0].set_title("position formulation")
    ax[0, 0].grid()
    ax[0, 0].legend()

    # errors velocity formulation
    ax[0, 1].loglog(dts, q_errors[1], "-.r", label="q")
    ax[0, 1].loglog(dts, u_errors[1], "-.g", label="u")
    ax[0, 1].loglog(dts, la_g_errors[1], "-.b", label="la_g")
    ax[0, 1].loglog(dts, dts_1, "-k", label="dt")
    ax[0, 1].loglog(dts, dts_2, "--k", label="dt^2")
    ax[0, 1].set_title("velocity formulation")
    ax[0, 1].grid()
    ax[0, 1].legend()

    # errors auxiliary formulation
    ax[0, 2].loglog(dts, q_errors[2], "-.r", label="q")
    ax[0, 2].loglog(dts, u_errors[2], "-.g", label="u")
    ax[0, 2].loglog(dts, la_g_errors[2], "-.b", label="la_g")
    ax[0, 2].loglog(dts, dts_1, "-k", label="dt")
    ax[0, 2].loglog(dts, dts_2, "--k", label="dt^2")
    ax[0, 2].set_title("auxiliary formulation")
    ax[0, 2].grid()
    ax[0, 2].legend()

    # errors position formulation
    ax[1, 0].loglog(dts, q_errors[3], "-.r", label="q")
    ax[1, 0].loglog(dts, u_errors[3], "-.g", label="u")
    ax[1, 0].loglog(dts, la_g_errors[3], "-.b", label="la_g")
    ax[1, 0].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 0].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 0].set_title("position formulation GGL")
    ax[1, 0].grid()
    ax[1, 0].legend()

    # errors velocity formulation
    ax[1, 1].loglog(dts, q_errors[4], "-.r", label="q")
    ax[1, 1].loglog(dts, u_errors[4], "-.g", label="u")
    ax[1, 1].loglog(dts, la_g_errors[4], "-.b", label="la_g")
    ax[1, 1].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 1].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 1].set_title("velocity formulation GGL")
    ax[1, 1].grid()
    ax[1, 1].legend()

    # errors auxiliary formulation
    ax[1, 2].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 2].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 2].loglog(dts, q_errors[5], "-.r", label="q")
    ax[1, 2].loglog(dts, u_errors[5], "-.g", label="u")
    ax[1, 2].loglog(dts, la_g_errors[5], "-.b", label="la_g")
    ax[1, 2].set_title("auxiliary formulation GGL")
    ax[1, 2].grid()
    ax[1, 2].legend()

    plt.show()
