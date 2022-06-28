import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cardillo.math.rotations import A_IK_basic, rodriguez_inv, Spurrier

from cardillo.model.frame.frame import Frame
from cardillo.model.rigid_body import (
    RigidBodyEuler,
    RigidBodyAxisAngle,
    RigidBodyQuaternion,
)
from cardillo.model.bilateral_constraints.implicit import SphericalJoint
from cardillo.math.algebra import cross3, ax2skew
from cardillo.math import approx_fprime
from cardillo.model import Model
from cardillo.solver import (
    GenAlphaFirstOrder,
    GenAlphaFirstOrderGGL2_V3,
    HalfExplicitEulerFixedPoint,
)

# case = "Euler"
# case = "Euler_self"
# case = "AxisAngle"
case = "Quaternion"


class HeavyTopQuaternion(RigidBodyQuaternion):
    def __init__(self, A, B, grav, q0=None, u0=None):
        self.grav = grav
        self.r_OQ = r_OQ

        # initialize rigid body
        self.K_Theta_S = np.diag([A, A, B])
        RigidBodyQuaternion.__init__(self, m, self.K_Theta_S, q0=q0, u0=u0)

        # gravity
        self.f_g = np.array([0, 0, -self.m * self.grav])

    def f_pot(self, t, q):
        return self.f_g @ self.J_P(t, q)

    def f_pot_q(self, t, q, coo):
        dense = np.einsum("i,ijk->jk", self.f_g, self.J_P_q(t, q))
        coo.extend(dense, (self.uDOF, self.qDOF))


class HeavyTopAxisAngle(RigidBodyAxisAngle):
    def __init__(self, A, B, grav, q0=None, u0=None):
        self.grav = grav
        self.r_OQ = r_OQ

        # initialize rigid body
        self.K_Theta_S = np.diag([A, A, B])
        RigidBodyAxisAngle.__init__(self, m, self.K_Theta_S, q0=q0, u0=u0)

        # gravity
        self.f_g = np.array([0, 0, -self.m * self.grav])

    def f_pot(self, t, q):
        return self.f_g @ self.J_P(t, q)

    def f_pot_q(self, t, q, coo):
        dense = np.einsum("i,ijk->jk", self.f_g, self.J_P_q(t, q))
        coo.extend(dense, (self.uDOF, self.qDOF))


class HeavyTopEuler(RigidBodyEuler):
    def __init__(self, A, B, grav, q0=None, u0=None):
        self.grav = grav
        self.r_OQ = r_OQ

        # initialize rigid body
        self.K_Theta_S = np.diag([A, A, B])
        # RigidBodyEuler.__init__(self, m, self.K_Theta_S, axis="zxz", q0=q0, u0=u0)
        RigidBodyEuler.__init__(self, m, self.K_Theta_S, axis="zyx", q0=q0, u0=u0)

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

    def g_dot_q(self, t, q, u, coo):
        coo.extend(self.v_P_q(t, q, u, K_r_SP=self.K_r_SQ), (self.la_gDOF, self.qDOF))

    def g_dot_u(self, t, q, coo):
        coo.extend(self.J_P(t, q, K_r_SP=self.K_r_SQ), (self.la_gDOF, self.uDOF))

    def g_ddot(self, t, q, u, u_dot):
        return self.a_P(t, q, u, u_dot, K_r_SP=self.K_r_SQ)

    def g_q_dense(self, t, q):
        return self.r_OP_q(t, q, K_r_SP=self.K_r_SQ)

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    # TODO:
    def g_q_T_mu_g(self, t, q, mu_g, coo):
        dense = approx_fprime(
            q, lambda q: mu_g @ self.g_q_dense(t, q), method="2-point"
        )
        coo.extend(dense, (self.uDOF, self.qDOF))

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
        la_g = top.la_g(t, q, u)

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
m = 15.0
A = 0.234375
B = 0.46875
l = 1.0
grav = 9.81
alpha0 = pi
beta0 = pi / 2
gamma0 = 0

omega_x0 = 0
# omega_y0 = 0 # Arnodl2015 p. 174
omega_y0 = -4.61538  # Arnold2015b p. 13
omega_z0 = 150

#############################
# initial position and angles
#############################
phi0 = np.array([alpha0, beta0, gamma0])

r_OQ = np.zeros(3)
K_r_OS0 = np.array([0, 0, l])
A_IK0 = HeavyTop.A_IK(0, [0, 0, 0, alpha0, beta0, gamma0])
r_OS0 = A_IK0 @ K_r_OS0

if case == "Euler" or case == "Euler_self":
    q0 = np.concatenate((r_OS0, phi0))
elif case == "AxisAngle":
    q0 = np.concatenate((r_OS0, rodriguez_inv(A_IK0)))
elif case == "Quaternion":
    q0 = np.concatenate((r_OS0, Spurrier(A_IK0)))
else:
    raise NotImplementedError

####################
# initial velocities
####################
K_Omega0 = np.array([omega_x0, omega_y0, omega_z0])
v_S0 = A_IK0 @ cross3(K_Omega0, K_r_OS0)
u0 = np.concatenate((v_S0, K_Omega0))

# 1. hand written version
if case == "Euler_self":
    top = HeavyTop(m, l, A, B, grav, r_OQ, q0, u0)
    model = Model()
    model.add(top)
    model.assemble()
else:
    # 2. reuse existing RigidBodyEuler and SphericalJoint
    frame = Frame()
    if case == "Euler":
        top = HeavyTopEuler(A, B, grav, q0, u0)
    elif case == "AxisAngle":
        top = HeavyTopAxisAngle(A, B, grav, q0, u0)
    elif case == "Quaternion":
        top = HeavyTopQuaternion(A, B, grav, q0, u0)

    spherical_joint = SphericalJoint(frame, top, np.zeros(3))
    model = Model()
    model.add(top)
    model.add(frame)
    model.add(spherical_joint)
    model.assemble()


def show_animation(top, t, q, scale=1, show=False):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=-scale, top=scale)

    slowmotion = 1
    fps = 200
    t0 = t[0]
    t1 = t[-1]
    animation_time = slowmotion * t1
    target_frames = int(fps * animation_time)
    frac = max(1, int(len(t) / target_frames))
    if frac == 1:
        target_frames = len(t)
    interval = 1000 / fps

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    def create(t, q):
        x_S, y_S, z_S = top.r_OP(t, q)

        A_IK = top.A_IK(t, q)
        d1, d2, d3 = A_IK.T

        (COM,) = ax.plot([0.0, x_S], [0.0, y_S], [0.0, z_S], "-ok")
        (d1_,) = ax.plot(
            [x_S, x_S + d1[0]], [y_S, y_S + d1[1]], [z_S, z_S + d1[2]], "-r"
        )
        (d2_,) = ax.plot(
            [x_S, x_S + d2[0]], [y_S, y_S + d2[1]], [z_S, z_S + d2[2]], "-g"
        )
        (d3_,) = ax.plot(
            [x_S, x_S + d3[0]], [y_S, y_S + d3[1]], [z_S, z_S + d3[2]], "-b"
        )

        return COM, d1_, d2_, d3_

    COM, d1_, d2_, d3_ = create(t0, q[0])

    def update(t, q, COM, d1_, d2_, d3_):
        x_S, y_S, z_S = top.r_OP(t, q)

        A_IK = top.A_IK(t, q)
        d1, d2, d3 = A_IK.T

        COM.set_data(np.array([0.0, x_S]), np.array([0.0, y_S]))
        COM.set_3d_properties(np.array([0.0, z_S]))

        d1_.set_data(np.array([x_S, x_S + d1[0]]), np.array([y_S, y_S + d1[1]]))
        d1_.set_3d_properties(np.array([z_S, z_S + d1[2]]))

        d2_.set_data(np.array([x_S, x_S + d2[0]]), np.array([y_S, y_S + d2[1]]))
        d2_.set_3d_properties(np.array([z_S, z_S + d2[2]]))

        d3_.set_data(np.array([x_S, x_S + d3[0]]), np.array([y_S, y_S + d3[1]]))
        d3_.set_3d_properties(np.array([z_S, z_S + d3[2]]))

        return COM, d1_, d2_, d3_

    def animate(i):
        update(t[i], q[i], COM, d1_, d2_, d3_)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    if show:
        plt.show()

    return anim


def state():
    rho_inf = 0.9
    tol = 1.0e-8
    t1 = 1
    # t1 = 0.1
    dt = 1.0e-3

    # sol = GenAlphaFirstOrder(
    #     model,
    #     t1,
    #     dt,
    #     rho_inf=rho_inf,
    #     tol=tol,
    #     unknowns="velocities",
    #     GGL=False,
    #     numerical_jacobian=False,
    # ).solve()

    sol = HalfExplicitEulerFixedPoint(model, t1, dt, atol=tol).solve()

    t = sol.t
    q = sol.q
    u = sol.u
    la_g = sol.la_g

    def export_q(sol, name):
        header = "t, x, y, z, al, be, ga"
        export_data = np.vstack([sol.t, *sol.q.T]).T
        np.savetxt(
            name,
            export_data,
            delimiter=", ",
            header=header,
            comments="",
        )

    export_q(sol, "trajectory.txt")

    ###################
    # visualize results
    ###################
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # center of mass
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(t, q[:, 0], "-r", label="x")
    ax.plot(t, q[:, 1], "-g", label="y")
    ax.plot(t, q[:, 2], "-b", label="z")
    ax.grid()
    ax.legend()

    # alpha, beta, gamma
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(t, q[:, 3], "-r", label="alpha")
    ax.plot(t, q[:, 4], "-g", label="beta")
    ax.plot(t, q[:, 5], "-b", label="gamm")
    ax.grid()
    ax.legend()

    # x-y-z trajectory
    ax = fig.add_subplot(2, 3, 3, projection="3d")
    ax.plot3D(
        q[:, 0],
        q[:, 1],
        q[:, 2],
        "-r",
        label="x-y-z trajectory",
    )
    ax.grid()
    ax.legend()

    # x_dot, y_dot, z_dot
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(t, u[:, 0], "-r", label="x_dot")
    ax.plot(t, u[:, 1], "-g", label="y_dot")
    ax.plot(t, u[:, 2], "-b", label="z_dot")
    ax.grid()
    ax.legend()

    # omega_x, omega_y, omega_z
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(t, u[:, 3], "-r", label="omega_x")
    ax.plot(t, u[:, 4], "-g", label="omega_y")
    ax.plot(t, u[:, 5], "-b", label="omega_z")
    ax.grid()
    ax.legend()

    # la_g
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(t, la_g[:, 0], "-r", label="la_g0")
    ax.plot(t, la_g[:, 1], "-g", label="la_g1")
    ax.plot(t, la_g[:, 2], "-b", label="la_g2")
    ax.grid()
    ax.legend()

    anim = show_animation(top, t, q)

    plt.show()


def transient():
    t1 = 0.1
    tol = 1.0e-8
    h = 1.0e-3

    def export_la_g(sol, name):
        header = "t, la_g0, la_g1, la_g2"
        t = sol.t
        la_g = sol.la_g
        export_data = np.vstack([t, *la_g.T]).T
        np.savetxt(
            name,
            export_data,
            delimiter=", ",
            header=header,
            comments="",
        )

    # solve index 3 problem with rho_inf = 0.9
    sol_9 = GenAlphaFirstOrder(
        model, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities", GGL=False
    ).solve()
    export_la_g(sol_9, "la_g_9.txt")

    # solve index 3 problem with rho_inf = 0.6
    sol_6 = GenAlphaFirstOrder(
        model, t1, h, rho_inf=0.6, tol=tol, unknowns="velocities", GGL=False
    ).solve()
    export_la_g(sol_6, "la_g_6.txt")

    # solve GGL with rho_inf = 0.9
    sol_9_GGL = GenAlphaFirstOrder(
        model, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities", GGL=True
    ).solve()
    export_la_g(sol_9_GGL, "la_g_9_GGL.txt")

    # solve GGL with rho_inf = 0.6
    sol_6_GGL = GenAlphaFirstOrder(
        model, t1, h, rho_inf=0.6, tol=tol, unknowns="velocities", GGL=True
    ).solve()
    export_la_g(sol_6_GGL, "la_g_6_GGL.txt")

    # # solve GGL with rho_inf = 0.9
    # sol_9_GGL2 = GenAlphaFirstOrderGGL2_V3(
    #     model, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities"
    # ).solve()
    # export_la_g(sol_9_GGL2, "la_g_9_GGL2.txt")

    # # solve GGL with rho_inf = 0.6
    # sol_6_GGL2 = GenAlphaFirstOrderGGL2_V3(
    #     model, t1, h, rho_inf=0.6, tol=tol, unknowns="velocities"
    # ).solve()
    # export_la_g(sol_6_GGL2, "la_g_6_GGL2.txt")

    ###################
    # visualize results
    ###################
    fig = plt.figure(figsize=plt.figaspect(1))

    ax = fig.add_subplot(1, 3, 1)
    ax.plot(sol_6.t, sol_6.la_g[:, 0], "-r", label="la_g0_6")
    ax.plot(sol_6.t, sol_6.la_g[:, 1], "-g", label="la_g1_6")
    ax.plot(sol_6.t, sol_6.la_g[:, 2], "-b", label="la_g2_6")
    ax.plot(sol_9.t, sol_9.la_g[:, 0], "--r", label="la_g0_9")
    ax.plot(sol_9.t, sol_9.la_g[:, 1], "--g", label="la_g1_9")
    ax.plot(sol_9.t, sol_9.la_g[:, 2], "--b", label="la_g2_9")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 0], "-r", label="la_g0_6_GGL")
    ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 1], "-g", label="la_g1_6_GGL")
    ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 2], "-b", label="la_g2_6_GGL")
    ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 0], "--r", label="la_g0_9_GGL")
    ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 1], "--g", label="la_g1_9_GGL")
    ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 2], "--b", label="la_g2_9_GGL")
    ax.grid()
    ax.legend()

    plt.show()
    exit()

    # index 3
    ax = fig.add_subplot(3, 3, 1)
    ax.plot(sol_6.t, sol_6.la_g[:, 0], "-k", label="la_g0_6")
    ax.plot(sol_9.t, sol_9.la_g[:, 0], "--k", label="la_g0_9")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 4)
    ax.plot(sol_6.t, sol_6.la_g[:, 1], "-k", label="la_g1_6")
    ax.plot(sol_9.t, sol_9.la_g[:, 1], "--k", label="la_g1_9")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 7)
    ax.plot(sol_6.t, sol_6.la_g[:, 2], "-k", label="la_g2_6")
    ax.plot(sol_9.t, sol_9.la_g[:, 2], "--k", label="la_g2_9")
    ax.grid()
    ax.legend()

    # index 2
    ax = fig.add_subplot(3, 3, 2)
    ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 0], "-k", label="la_g0_6_GGL")
    ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 0], "--k", label="la_g0_9_GGL")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 5)
    ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 1], "-k", label="la_g1_6_GGL")
    ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 1], "--k", label="la_g1_9_GGL")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 8)
    ax.plot(sol_6_GGL.t, sol_6_GGL.la_g[:, 2], "-k", label="la_g2_6_GGL")
    ax.plot(sol_9_GGL.t, sol_9_GGL.la_g[:, 2], "--k", label="la_g2_9_GGL")
    ax.grid()
    ax.legend()

    # index 1
    ax = fig.add_subplot(3, 3, 3)
    ax.plot(sol_6_GGL2.t, sol_6_GGL2.la_g[:, 0], "-k", label="la_g0_6_GGL2")
    ax.plot(sol_9_GGL2.t, sol_9_GGL2.la_g[:, 0], "--k", label="la_g0_9_GGL2")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 6)
    ax.plot(sol_6_GGL2.t, sol_6_GGL2.la_g[:, 1], "-k", label="la_g1_6_GGL2")
    ax.plot(sol_9_GGL2.t, sol_9_GGL2.la_g[:, 1], "--k", label="la_g1_9_GGL2")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 9)
    ax.plot(sol_6_GGL2.t, sol_6_GGL2.la_g[:, 2], "-k", label="la_g2_6_GGL2")
    ax.plot(sol_9_GGL2.t, sol_9_GGL2.la_g[:, 2], "--k", label="la_g2_9_GGL2")
    ax.grid()
    ax.legend()

    plt.show()


def gaps():
    t1 = 0.1
    tol = 1.0e-8
    h = 1.0e-3

    def export_gaps(sol, name):
        header = "t, g, g_dot, g_ddot"
        t = sol.t
        q = sol.q
        u = sol.u
        # try:
        #     u_dot = sol.a  # GGL2 solver
        # except:
        #     u_dot = sol.u_dot  # other solvers
        u_dot = sol.u_dot  # other solvers

        g = np.array([np.linalg.norm(model.g(ti, qi)) for ti, qi in zip(t, q)])

        g_dot = np.array(
            [np.linalg.norm(model.g_dot(ti, qi, ui)) for ti, qi, ui in zip(t, q, u)]
        )

        g_ddot = np.array(
            [
                np.linalg.norm(model.g_ddot(ti, qi, ui, u_doti))
                for ti, qi, ui, u_doti in zip(t, q, u, u_dot)
            ]
        )

        export_data = np.vstack([t, g, g_dot, g_ddot]).T
        np.savetxt(
            name,
            export_data,
            delimiter=", ",
            header=header,
            comments="",
        )

        return g, g_dot, g_ddot

    # solve index 3 problem with rho_inf = 0.9
    sol_9 = GenAlphaFirstOrder(
        model, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities", GGL=False
    ).solve()
    g_9, g_dot_9, g_ddot_9 = export_gaps(sol_9, "g_9.txt")

    # solve GGL with rho_inf = 0.9
    sol_9_GGL = GenAlphaFirstOrder(
        model, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities", GGL=True
    ).solve()
    g_9_GGL, g_dot_9_GGL, g_ddot_9_GGL = export_gaps(sol_9_GGL, "g_9_GGL.txt")

    # solve GGL2 with rho_inf = 0.9
    sol_9_GGL2 = GenAlphaFirstOrderGGL2_V3(
        model, t1, h, rho_inf=0.9, tol=tol, unknowns="velocities"
    ).solve()
    g_9_GGL2, g_dot_9_GGL2, g_ddot_9_GGL2 = export_gaps(sol_9_GGL2, "g_9_GGL2.txt")

    ###################
    # visualize results
    ###################
    fig = plt.figure(figsize=plt.figaspect(1))

    # index 3
    ax = fig.add_subplot(3, 3, 1)
    ax.plot(sol_9.t, g_9, "-k", label="||g_9||")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 4)
    ax.plot(sol_9.t, g_dot_9, "-k", label="||g_dot_9||")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 7)
    ax.plot(sol_9.t, g_ddot_9, "-k", label="||g_ddot_9||")
    ax.grid()
    ax.legend()

    # index 2
    ax = fig.add_subplot(3, 3, 2)
    ax.plot(sol_9_GGL.t, g_9_GGL, "-k", label="||g_9_GGL||")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 5)
    ax.plot(sol_9_GGL.t, g_dot_9_GGL, "-k", label="||g_dot_9_GGL||")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 8)
    ax.plot(sol_9_GGL.t, g_ddot_9_GGL, "-k", label="||g_ddot_9_GGL||")
    ax.grid()
    ax.legend()

    # index 1
    ax = fig.add_subplot(3, 3, 3)
    ax.plot(sol_9_GGL2.t, g_9_GGL2, "-k", label="||g_9_GGL2||")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 6)
    ax.plot(sol_9_GGL2.t, g_dot_9_GGL2, "-k", label="||g_dot_9_GGL2||")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(3, 3, 9)
    ax.plot(sol_9_GGL2.t, g_ddot_9_GGL2, "-k", label="||g_ddot_9_GGL2||")
    ax.grid()
    ax.legend()

    plt.show()


def convergence():
    rho_inf = 0.9
    tol_ref = 1.0e-8
    tol = 1.0e-8

    # compute step sizes with powers of 2
    dt_ref = 2.5e-5  # Arnold2015b
    dts = (2.0 ** np.arange(8, 1, -1)) * dt_ref  # [6.4e-3, 3.2e-3, ..., 2e-4, 1e-4]
    # dts = (2.0 ** np.arange(8, 5, -1)) * dt_ref  # [6.4e-3, 3.2e-3, 1.6e-3]

    # end time (note this has to be > 0.5, otherwise long term error throws ans error)
    t1 = (2.0**15) * dt_ref  # this yields 0.8192 for dt_ref = 2.5e-5
    # t1 = (2.0**16) * dt_ref # this yields 1.6384 for dt_ref = 2.5e-5

    # # TODO: Only for debugging!
    # dt_ref = 5e-4
    # dts = np.array([1.0e-3])
    # t1 = (2.0**11) * dt_ref

    dts_1 = dts
    dts_2 = dts**2
    print(f"dts: {dts}")

    # errors for all 6 possible solvers
    q_errors_transient = np.inf * np.ones((6, len(dts)), dtype=float)
    u_errors_transient = np.inf * np.ones((6, len(dts)), dtype=float)
    la_g_errors_transient = np.inf * np.ones((6, len(dts)), dtype=float)
    q_errors_longterm = np.inf * np.ones((6, len(dts)), dtype=float)
    u_errors_longterm = np.inf * np.ones((6, len(dts)), dtype=float)
    la_g_errors_longterm = np.inf * np.ones((6, len(dts)), dtype=float)

    # compute reference solution as described in Arnold2015 Section 3.3
    print(f"compute reference solution:")
    reference = GenAlphaFirstOrder(
        model, t1, dt_ref, rho_inf=rho_inf, tol=tol_ref, unknowns="velocities", GGL=True
    ).solve()
    t_ref = reference.t
    q_ref = reference.q
    u_ref = reference.u
    la_g_ref = reference.la_g

    # plot_state = True
    plot_state = False
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

    def errors(sol):
        t = sol.t
        q = sol.q
        u = sol.u
        la_g = sol.la_g

        # distinguish between transient and long term time steps
        t_idx_transient = np.where(t <= 0.1)[0]
        t_idx_longterm = np.where(t >= 0.5)[0]

        # compute difference between computed solution and reference solution
        # for identical time instants
        # idx = np.where(np.abs(t[:, None] - t_ref) < 1.0e-8)[1]
        t_ref_idx_transient = np.where(
            np.abs(t[t_idx_transient, None] - t_ref) < 1.0e-8
        )[1]
        t_ref_idx_longterm = np.where(np.abs(t[t_idx_longterm, None] - t_ref) < 1.0e-8)[
            1
        ]

        # differences
        # diff_q = q - q_ref[idx]
        # diff_u = u - u_ref[idx]
        # diff_la_g = la_g - la_g_ref[idx]

        q_transient = q[t_idx_transient]
        u_transient = u[t_idx_transient]
        la_g_transient = la_g[t_idx_transient]
        diff_transient_q = q_transient - q_ref[t_ref_idx_transient]
        diff_transient_u = u_transient - u_ref[t_ref_idx_transient]
        diff_transient_la_g = la_g_transient - la_g_ref[t_ref_idx_transient]

        q_longterm = q[t_idx_longterm]
        u_longterm = u[t_idx_longterm]
        la_g_longterm = la_g[t_idx_longterm]
        diff_longterm_q = q_longterm - q_ref[t_ref_idx_longterm]
        diff_longterm_u = u_longterm - u_ref[t_ref_idx_longterm]
        diff_longterm_la_g = la_g_longterm - la_g_ref[t_ref_idx_longterm]

        # max relative error
        q_error_transient = np.max(
            np.linalg.norm(diff_transient_q, axis=1)
            / np.linalg.norm(q_transient, axis=1)
        )
        u_error_transient = np.max(
            np.linalg.norm(diff_transient_u, axis=1)
            / np.linalg.norm(u_transient, axis=1)
        )
        la_g_error_transient = np.max(
            np.linalg.norm(diff_transient_la_g, axis=1)
            / np.linalg.norm(la_g_transient, axis=1)
        )
        q_error_longterm = np.max(
            np.linalg.norm(diff_longterm_q, axis=1) / np.linalg.norm(q_longterm, axis=1)
        )
        u_error_longterm = np.max(
            np.linalg.norm(diff_longterm_u, axis=1) / np.linalg.norm(u_longterm, axis=1)
        )
        la_g_error_longterm = np.max(
            np.linalg.norm(diff_longterm_la_g, axis=1)
            / np.linalg.norm(la_g_longterm, axis=1)
        )

        # return q_error, u_error, la_g_error
        return (
            q_error_transient,
            u_error_transient,
            la_g_error_transient,
            q_error_longterm,
            u_error_longterm,
            la_g_error_longterm,
        )

    for i, dt in enumerate(dts):
        print(f"i: {i}, dt: {dt:1.1e}")

        # # position formulation
        # sol = GenAlphaFirstOrder(
        #     model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="positions", GGL=False
        # ).solve()
        # (
        #     q_errors_transient[0, i],
        #     u_errors_transient[0, i],
        #     la_g_errors_transient[0, i],
        #     q_errors_longterm[0, i],
        #     u_errors_longterm[0, i],
        #     la_g_errors_longterm[0, i],
        # ) = errors(sol)

        # velocity formulation
        sol = GenAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="velocities", GGL=False
        ).solve()
        (
            q_errors_transient[1, i],
            u_errors_transient[1, i],
            la_g_errors_transient[1, i],
            q_errors_longterm[1, i],
            u_errors_longterm[1, i],
            la_g_errors_longterm[1, i],
        ) = errors(sol)

        # # auxiliary formulation
        # sol = GenAlphaFirstOrder(
        #     model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="auxiliary", GGL=False
        # ).solve()
        # (
        #     q_errors_transient[2, i],
        #     u_errors_transient[2, i],
        #     la_g_errors_transient[2, i],
        #     q_errors_longterm[2, i],
        #     u_errors_longterm[2, i],
        #     la_g_errors_longterm[2, i],
        # ) = errors(sol)

        # # GGL formulation - positions
        # sol = GenAlphaFirstOrder(
        #     model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="positions", GGL=True
        # ).solve()
        # (
        #     q_errors_transient[3, i],
        #     u_errors_transient[3, i],
        #     la_g_errors_transient[3, i],
        #     q_errors_longterm[3, i],
        #     u_errors_longterm[3, i],
        #     la_g_errors_longterm[3, i],
        # ) = errors(sol)

        # GGL formulation - velocityies
        sol = GenAlphaFirstOrder(
            model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="velocities", GGL=True
        ).solve()
        (
            q_errors_transient[4, i],
            u_errors_transient[4, i],
            la_g_errors_transient[4, i],
            q_errors_longterm[4, i],
            u_errors_longterm[4, i],
            la_g_errors_longterm[4, i],
        ) = errors(sol)

        # # GGL formulation - auxiliary
        # sol = GenAlphaFirstOrder(
        #     model, t1, dt, rho_inf=rho_inf, tol=tol, unknowns="auxiliary", GGL=True
        # ).solve()
        # (
        #     q_errors_transient[5, i],
        #     u_errors_transient[5, i],
        #     la_g_errors_transient[5, i],
        #     q_errors_longterm[5, i],
        #     u_errors_longterm[5, i],
        #     la_g_errors_longterm[5, i],
        # ) = errors(sol)

        # #############################
        # # export errors and dt, dt**2
        # #############################
        # if i == 0:
        #     header = "dt, dt2, pos, vel, aux, pos_GGL, vel_GGL, aux_GGL"
        #     export_data = np.array([[dt, dt**2, *q_errors_transient[:, i]]])
        #     np.savetxt(
        #         "transient_error_heavy_top_q.txt",
        #         export_data,
        #         delimiter=", ",
        #         header=header,
        #         comments="",
        #     )
        # else:
        #     with open("transient_error_heavy_top_q.txt", "ab") as f:
        #         # f.write(b"\n")
        #         export_data = np.array([[dt, dt**2, *q_errors_transient[:, i]]])
        #         np.savetxt(f, export_data, delimiter=", ", comments="")

    #############################
    # export errors and dt, dt**2
    #############################
    header = "dt, dt2, pos, vel, aux, pos_GGL, vel_GGL, aux_GGL"

    # transient errors
    export_data = np.vstack((dts, dts_2, *q_errors_transient)).T
    np.savetxt(
        "transient_error_heavy_top_q.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    export_data = np.vstack((dts, dts_2, *u_errors_transient)).T
    np.savetxt(
        "transient_error_heavy_top_u.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    export_data = np.vstack((dts, dts_2, *la_g_errors_transient)).T
    np.savetxt(
        "transient_error_heavy_top_la_g.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    # longterm errors
    export_data = np.vstack((dts, dts_2, *q_errors_longterm)).T
    np.savetxt(
        "longterm_error_heavy_top_q.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    export_data = np.vstack((dts, dts_2, *u_errors_longterm)).T
    np.savetxt(
        "longterm_error_heavy_top_u.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    export_data = np.vstack((dts, dts_2, *la_g_errors_longterm)).T
    np.savetxt(
        "longterm_error_heavy_top_la_g.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    ##################
    # visualize errors
    ##################
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].set_title("velocity formulation (transient)")
    ax[0, 0].loglog(dts, dts_1, "-k", label="dt")
    ax[0, 0].loglog(dts, dts_2, "--k", label="dt^2")
    ax[0, 0].loglog(dts, q_errors_transient[1], "-.ro", label="q")
    ax[0, 0].loglog(dts, u_errors_transient[1], "-.go", label="u")
    ax[0, 0].loglog(dts, la_g_errors_transient[1], "-.bo", label="la_g")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].set_title("velocity formulation GGL (transient)")
    ax[1, 0].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 0].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 0].loglog(dts, q_errors_transient[4], "-.ro", label="q")
    ax[1, 0].loglog(dts, u_errors_transient[4], "-.go", label="u")
    ax[1, 0].loglog(dts, la_g_errors_transient[4], "-.bo", label="la_g")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[0, 1].set_title("velocity formulation (long term)")
    ax[0, 1].loglog(dts, dts_1, "-k", label="dt")
    ax[0, 1].loglog(dts, dts_2, "--k", label="dt^2")
    ax[0, 1].loglog(dts, q_errors_longterm[1], "-.ro", label="q")
    ax[0, 1].loglog(dts, u_errors_longterm[1], "-.go", label="u")
    ax[0, 1].loglog(dts, la_g_errors_longterm[1], "-.bo", label="la_g")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].set_title("velocity formulation GGL (long term)")
    ax[1, 1].loglog(dts, dts_1, "-k", label="dt")
    ax[1, 1].loglog(dts, dts_2, "--k", label="dt^2")
    ax[1, 1].loglog(dts, q_errors_longterm[4], "-.ro", label="q")
    ax[1, 1].loglog(dts, u_errors_longterm[4], "-.go", label="u")
    ax[1, 1].loglog(dts, la_g_errors_longterm[4], "-.bo", label="la_g")
    ax[1, 1].grid()
    ax[1, 1].legend()

    plt.show()

    # #################
    # # transient errors
    # #################
    # fig, ax = plt.subplots(2, 3)
    # fig.suptitle("Transient phase", fontsize=16)

    # # errors position formulation
    # ax[0, 0].loglog(dts, dts_1, "-k", label="dt")
    # ax[0, 0].loglog(dts, dts_2, "--k", label="dt^2")
    # ax[0, 0].loglog(dts, q_errors_transient[0], "-.ro", label="q")
    # ax[0, 0].loglog(dts, u_errors_transient[0], "-.go", label="u")
    # ax[0, 0].loglog(dts, la_g_errors_transient[0], "-.bo", label="la_g")
    # ax[0, 0].set_title("position formulation")
    # ax[0, 0].grid()
    # ax[0, 0].legend()

    # # errors velocity formulation
    # ax[0, 1].loglog(dts, dts_1, "-k", label="dt")
    # ax[0, 1].loglog(dts, dts_2, "--k", label="dt^2")
    # ax[0, 1].loglog(dts, q_errors_transient[1], "-.ro", label="q")
    # ax[0, 1].loglog(dts, u_errors_transient[1], "-.go", label="u")
    # ax[0, 1].loglog(dts, la_g_errors_transient[1], "-.bo", label="la_g")
    # ax[0, 1].set_title("velocity formulation")
    # ax[0, 1].grid()
    # ax[0, 1].legend()

    # # errors auxiliary formulation
    # ax[0, 2].loglog(dts, dts_1, "-k", label="dt")
    # ax[0, 2].loglog(dts, dts_2, "--k", label="dt^2")
    # ax[0, 2].loglog(dts, q_errors_transient[2], "-.ro", label="q")
    # ax[0, 2].loglog(dts, u_errors_transient[2], "-.go", label="u")
    # ax[0, 2].loglog(dts, la_g_errors_transient[2], "-.bo", label="la_g")
    # ax[0, 2].set_title("auxiliary formulation")
    # ax[0, 2].grid()
    # ax[0, 2].legend()

    # # errors position formulation
    # ax[1, 0].loglog(dts, dts_1, "-k", label="dt")
    # ax[1, 0].loglog(dts, dts_2, "--k", label="dt^2")
    # ax[1, 0].loglog(dts, q_errors_transient[3], "-.ro", label="q")
    # ax[1, 0].loglog(dts, u_errors_transient[3], "-.go", label="u")
    # ax[1, 0].loglog(dts, la_g_errors_transient[3], "-.bo", label="la_g")
    # ax[1, 0].set_title("position formulation GGL")
    # ax[1, 0].grid()
    # ax[1, 0].legend()

    # # errors velocity formulation
    # ax[1, 1].loglog(dts, dts_1, "-k", label="dt")
    # ax[1, 1].loglog(dts, dts_2, "--k", label="dt^2")
    # ax[1, 1].loglog(dts, q_errors_transient[4], "-.ro", label="q")
    # ax[1, 1].loglog(dts, u_errors_transient[4], "-.go", label="u")
    # ax[1, 1].loglog(dts, la_g_errors_transient[4], "-.bo", label="la_g")
    # ax[1, 1].set_title("velocity formulation GGL")
    # ax[1, 1].grid()
    # ax[1, 1].legend()

    # # errors auxiliary formulation
    # ax[1, 2].loglog(dts, dts_1, "-k", label="dt")
    # ax[1, 2].loglog(dts, dts_2, "--k", label="dt^2")
    # ax[1, 2].loglog(dts, q_errors_transient[5], "-.ro", label="q")
    # ax[1, 2].loglog(dts, u_errors_transient[5], "-.go", label="u")
    # ax[1, 2].loglog(dts, la_g_errors_transient[5], "-.bo", label="la_g")
    # ax[1, 2].set_title("auxiliary formulation GGL")
    # ax[1, 2].grid()
    # ax[1, 2].legend()

    # #################
    # # longterm errors
    # #################
    # fig, ax = plt.subplots(2, 3)
    # fig.suptitle("Long term phase", fontsize=16)

    # # errors position formulation
    # ax[0, 0].loglog(dts, dts_1, "-k", label="dt")
    # ax[0, 0].loglog(dts, dts_2, "--k", label="dt^2")
    # ax[0, 0].loglog(dts, q_errors_longterm[0], "-.ro", label="q")
    # ax[0, 0].loglog(dts, u_errors_longterm[0], "-.go", label="u")
    # ax[0, 0].loglog(dts, la_g_errors_longterm[0], "-.bo", label="la_g")
    # ax[0, 0].set_title("position formulation")
    # ax[0, 0].grid()
    # ax[0, 0].legend()

    # # errors velocity formulation
    # ax[0, 1].loglog(dts, dts_1, "-k", label="dt")
    # ax[0, 1].loglog(dts, dts_2, "--k", label="dt^2")
    # ax[0, 1].loglog(dts, q_errors_longterm[1], "-.ro", label="q")
    # ax[0, 1].loglog(dts, u_errors_longterm[1], "-.go", label="u")
    # ax[0, 1].loglog(dts, la_g_errors_longterm[1], "-.bo", label="la_g")
    # ax[0, 1].set_title("velocity formulation")
    # ax[0, 1].grid()
    # ax[0, 1].legend()

    # # errors auxiliary formulation
    # ax[0, 2].loglog(dts, dts_1, "-k", label="dt")
    # ax[0, 2].loglog(dts, dts_2, "--k", label="dt^2")
    # ax[0, 2].loglog(dts, q_errors_longterm[2], "-.ro", label="q")
    # ax[0, 2].loglog(dts, u_errors_longterm[2], "-.go", label="u")
    # ax[0, 2].loglog(dts, la_g_errors_longterm[2], "-.bo", label="la_g")
    # ax[0, 2].set_title("auxiliary formulation")
    # ax[0, 2].grid()
    # ax[0, 2].legend()

    # # errors position formulation
    # ax[1, 0].loglog(dts, dts_1, "-k", label="dt")
    # ax[1, 0].loglog(dts, dts_2, "--k", label="dt^2")
    # ax[1, 0].loglog(dts, q_errors_longterm[3], "-.ro", label="q")
    # ax[1, 0].loglog(dts, u_errors_longterm[3], "-.go", label="u")
    # ax[1, 0].loglog(dts, la_g_errors_longterm[3], "-.bo", label="la_g")
    # ax[1, 0].set_title("position formulation GGL")
    # ax[1, 0].grid()
    # ax[1, 0].legend()

    # # errors velocity formulation
    # ax[1, 1].loglog(dts, dts_1, "-k", label="dt")
    # ax[1, 1].loglog(dts, dts_2, "--k", label="dt^2")
    # ax[1, 1].loglog(dts, q_errors_longterm[4], "-.ro", label="q")
    # ax[1, 1].loglog(dts, u_errors_longterm[4], "-.go", label="u")
    # ax[1, 1].loglog(dts, la_g_errors_longterm[4], "-.bo", label="la_g")
    # ax[1, 1].set_title("velocity formulation GGL")
    # ax[1, 1].grid()
    # ax[1, 1].legend()

    # # errors auxiliary formulation
    # ax[1, 2].loglog(dts, dts_1, "-k", label="dt")
    # ax[1, 2].loglog(dts, dts_2, "--k", label="dt^2")
    # ax[1, 2].loglog(dts, q_errors_longterm[5], "-.ro", label="q")
    # ax[1, 2].loglog(dts, u_errors_longterm[5], "-.go", label="u")
    # ax[1, 2].loglog(dts, la_g_errors_longterm[5], "-.bo", label="la_g")
    # ax[1, 2].set_title("auxiliary formulation GGL")
    # ax[1, 2].grid()
    # ax[1, 2].legend()

    # plt.show()


if __name__ == "__main__":
    state()
    # transient()
    # gaps()
    # convergence()
