import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pathlib import Path

from cardillo import System
from cardillo.discrete import RigidBodyQuaternion, RigidBodyEuler
from cardillo.math import axis_angle2quat, cross3, ax2skew, approx_fprime
from cardillo.forces import Force
from cardillo.solver import MoreauShifted, Rattle, MoreauClassical


class Sphere2PlaneCoulombContensouMoeller:
    def __init__(
        self,
        frame,
        subsystem,
        R_sphere,
        R_drilling,
        mu,
        e_N=None,
        e_F=None,
        frame_ID=np.zeros(3),
        K_r_SP=np.zeros(3),
        la_N0=None,
        la_F0=None,
    ):
        self.frame = frame
        self.subsystem = subsystem
        self.R_sphere = R_sphere
        self.R_drilling = R_drilling

        self.R_bar = (3 * np.pi / 16) * R_drilling
        self.A = np.diag([1, 1, self.R_bar])

        self.mu = np.array([mu])
        self.nla_N = 1

        if mu == 0:
            self.nla_F = 0
            self.NF_connectivity = [[]]
        else:
            self.nla_F = 3 * self.nla_N
            self.NF_connectivity = [[0, 1, 2]]
            self.gamma_F = self.__gamma_F

        self.e_N = np.zeros(self.nla_N) if e_N is None else e_N * np.ones(self.nla_N)
        self.e_F = np.zeros(self.nla_F) if e_F is None else e_F * np.ones(self.nla_F)
        self.frame_ID = frame_ID

        self.r_OQ = lambda t: self.frame.r_OP(t)
        self.t1t2 = lambda t: self.frame.A_IK(t)[:, :2].T
        self.n = lambda t: self.frame.A_IK(t)[:, 2]
        self.v_Q = lambda t: self.frame.v_P(t)
        self.a_Q = lambda t: self.frame.a_P(t)

        self.K_r_SP = K_r_SP

        self.la_N0 = np.zeros(self.nla_N) if la_N0 is None else la_N0
        self.la_F0 = np.zeros(self.nla_F) if la_F0 is None else la_F0

    def assembler_callback(self):
        qDOF = self.subsystem.local_qDOF_P(self.frame_ID)
        self.qDOF = self.subsystem.qDOF[qDOF]
        self.nq = len(self.qDOF)

        uDOF = self.subsystem.local_uDOF_P(self.frame_ID)
        self.uDOF = self.subsystem.uDOF[uDOF]
        self.nu = len(self.uDOF)

        self.r_OP = lambda t, q: self.subsystem.r_OP(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.r_OP_q = lambda t, q: self.subsystem.r_OP_q(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.v_P = lambda t, q, u: self.subsystem.v_P(
            t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.v_P_q = lambda t, q, u: self.subsystem.v_P_q(
            t, q, u, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.J_P = lambda t, q: self.subsystem.J_P(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.J_P_q = lambda t, q: self.subsystem.J_P_q(
            t, q, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.a_P = lambda t, q, u, a: self.subsystem.a_P(
            t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.a_P_q = lambda t, q, u, a: self.subsystem.a_P_q(
            t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )
        self.a_P_u = lambda t, q, u, a: self.subsystem.a_P_u(
            t, q, u, a, frame_ID=self.frame_ID, K_r_SP=self.K_r_SP
        )

        self.Omega = lambda t, q, u: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Omega(t, q, u, frame_ID=self.frame_ID)
        self.Omega_q = lambda t, q, u: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Omega_q(t, q, u, frame_ID=self.frame_ID) + np.einsum(
            "ijk,j->ik",
            self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID),
            self.subsystem.K_Omega(t, q, u, frame_ID=self.frame_ID),
        )
        self.Omega_u = lambda t, q, u: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Omega_u(t, q, u, frame_ID=self.frame_ID)
        self.J_R = lambda t, q: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID)
        self.J_R_q = lambda t, q: np.einsum(
            "ijl,jk->ikl",
            self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID),
            self.subsystem.K_J_R(t, q, frame_ID=self.frame_ID),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem.A_IK(t, q, frame_ID=self.frame_ID),
            self.subsystem.K_J_R_q(t, q, frame_ID=self.frame_ID),
        )
        self.Psi = lambda t, q, u, a: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Psi(t, q, u, a, frame_ID=self.frame_ID)
        self.Psi_q = lambda t, q, u, a: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Psi_q(t, q, u, a, frame_ID=self.frame_ID) + np.einsum(
            "ijk,j->ik",
            self.subsystem.A_IK_q(t, q, frame_ID=self.frame_ID),
            self.subsystem.K_Psi(t, q, u, a, frame_ID=self.frame_ID),
        )
        self.Psi_u = lambda t, q, u, a: self.subsystem.A_IK(
            t, q, frame_ID=self.frame_ID
        ) @ self.subsystem.K_Psi_u(t, q, u, a, frame_ID=self.frame_ID)

    #################
    # normal contacts
    #################
    def g_N(self, t, q):
        return np.array([self.n(t) @ (self.r_OP(t, q) - self.r_OQ(t))]) - self.R_sphere

    def g_N_q_dense(self, t, q):
        return np.array([self.n(t) @ self.r_OP_q(t, q)], dtype=q.dtype)

    def g_N_q(self, t, q, coo):
        coo.extend(self.g_N_q_dense(t, q), (self.la_NDOF, self.qDOF))

    def g_N_dot(self, t, q, u):
        # TODO: n_dot(t)
        return np.array(
            [self.n(t) @ (self.v_P(t, q, u) - self.v_Q(t))], dtype=np.common_type(q, u)
        )

    def g_N_dot_q_dense(self, t, q, u):
        return np.array([self.n(t) @ self.v_P_q(t, q, u)], dtype=np.common_type(q, u))

    def g_N_dot_q(self, t, q, u, coo):
        coo.extend(self.g_N_dot_q_dense(t, q, u), (self.la_NDOF, self.qDOF))

    def g_N_dot_u_dense(self, t, q):
        # TODO: n_dot(t)
        return np.array([self.n(t) @ self.J_P(t, q)], dtype=q.dtype)

    def g_N_dot_u(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q), (self.la_NDOF, self.uDOF))

    def W_N(self, t, q, coo):
        coo.extend(self.g_N_dot_u_dense(t, q).T, (self.uDOF, self.la_NDOF))

    def g_N_ddot(self, t, q, u, u_dot):
        return np.array(
            [self.n(t) @ (self.a_P(t, q, u, u_dot) - self.a_Q(t))],
            dtype=np.common_type(q, u, u_dot),
        )

    def g_N_ddot_q(self, t, q, u, u_dot, coo):
        dense = np.array(
            [self.n(t) @ self.a_P_q(t, q, u, u_dot)], dtype=np.common_type(q, u, u_dot)
        )
        coo.extend(dense, (self.la_NDOF, self.qDOF))

    def g_N_ddot_u(self, t, q, u, u_dot, coo):
        dense = np.array(
            [self.n(t) @ self.a_P_u(t, q, u, u_dot)], dtype=np.common_type(q, u, u_dot)
        )
        coo.extend(dense, (self.la_NDOF, self.uDOF))

    def Wla_N_q(self, t, q, la_N, coo):
        dense = la_N[0] * np.einsum("i,ijk->jk", self.n(t), self.J_P_q(t, q))
        coo.extend(dense, (self.uDOF, self.qDOF))

    ########################################
    # tangent contacts and drilling frcition
    ########################################
    def __gamma_F(self, t, q, u):
        v_C = self.v_P(t, q, u) + self.R_sphere * cross3(self.n(t), self.Omega(t, q, u))
        gamma_F = self.t1t2(t) @ (v_C - self.v_Q(t))
        omega = self.n(t) @ self.Omega(t, q, u)
        return self.A.T @ np.array([*gamma_F, omega])

    def gamma_F_q_dense(self, t, q, u):
        return approx_fprime(
            q, lambda q: self.__gamma_F(t, q, u), method="2-point", eps=1e-6
        )

    def gamma_F_q(self, t, q, u, coo):
        coo.extend(self.gamma_F_q_dense(t, q, u), (self.la_FDOF, self.qDOF))

    def gamma_F_dot(self, t, q, u, u_dot):
        raise NotImplementedError
        # #TODO: t1t2_dot(t) & n_dot(t)
        Omega = self.Omega(t, q, u)
        r_PC = -self.R_sphere * self.n(t)
        a_C = (
            self.a_P(t, q, u, u_dot)
            + cross3(self.Psi(t, q, u, u_dot), r_PC)
            + cross3(Omega, cross3(Omega, r_PC))
        )
        gamma_F_dot = self.t1t2(t) @ (a_C - self.a_Q(t))
        return gamma_F_dot

    def gamma_F_dot_q(self, t, q, u, u_dot, coo):
        # #TODO: t1t2_dot(t) & n_dot(t)
        gamma_F_dot_q_num = approx_fprime(
            q, lambda q: self.gamma_F_dot(t, q, u, u_dot), method="2-point"
        )
        # Omega = self.Omega(t, q, u)
        # r_PC = -self.r * self.n(t)
        # a_C = self.a_P(t, q, u, u_dot) + cross3(self.Psi(t, q, u, u_dot), r_PC) + cross3(Omega, cross3(Omega, r_PC))
        # gamma_F_dot = self.t1t2(t) @ (a_C - self.a_Q(t))
        coo.extend(gamma_F_dot_q_num, (self.la_FDOF, self.qDOF))

    def gamma_F_dot_u(self, t, q, u, u_dot, coo):
        # #TODO: t1t2_dot(t) & n_dot(t)
        gamma_F_dot_u_num = approx_fprime(
            u, lambda u: self.gamma_F_dot(t, q, u, u_dot), method="2-point"
        )
        # Omega = self.Omega(t, q, u)
        # r_PC = -self.r * self.n(t)
        # a_C = self.a_P(t, q, u, u_dot) + cross3(self.Psi(t, q, u, u_dot), r_PC) + cross3(Omega, cross3(Omega, r_PC))
        # gamma_F_dot = self.t1t2(t) @ (a_C - self.a_Q(t))
        coo.extend(gamma_F_dot_u_num, (self.la_FDOF, self.uDOF))

    def gamma_F_u_dense(self, t, q):
        J_C = self.J_P(t, q) + self.R_sphere * ax2skew(self.n(t)) @ self.J_R(t, q)
        J_R = self.J_R(t, q)
        return self.A.T @ np.concatenate(
            (self.t1t2(t) @ J_C, (self.n(t) @ J_R)[None, :])
        )

    def gamma_F_u(self, t, q, coo):
        coo.extend(self.gamma_F_u_dense(t, q), (self.la_FDOF, self.uDOF))

    def W_F(self, t, q, coo):
        coo.extend(self.gamma_F_u_dense(t, q).T, (self.uDOF, self.la_FDOF))

    def Wla_F_q(self, t, q, la_F, coo):
        raise NotImplementedError
        # J_C_q = self.J_P_q(t, q) + self.R_sphere * np.einsum(
        #     "ij,jkl->ikl", ax2skew(self.n(t)), self.J_R_q(t, q)
        # )
        # dense = np.einsum("i,ij,jkl->kl", la_F, self.t1t2(t), J_C_q)
        # dense
        # coo.extend(dense, (self.uDOF, self.qDOF))


def make_system(RigidBodyBase):
    assert RigidBodyBase in [RigidBodyQuaternion, RigidBodyEuler]

    system = System()

    # Dynamics:
    m = 6e-3  # kg
    I1 = 8e-7  # kg m2 # = I_2 # Leine2013
    I3 = 7e-7  # kg m2
    K_theta_S = np.diag([I1, I1, I3])
    g = 9.81  # kg m / s2

    # Geometry:
    a1 = 3e-3  # m
    a2 = 1.6e-2  # m
    R1 = 1.5e-2  # m
    R2 = 5e-3  # m
    K_r_SC1 = np.array([0, 0, a1])
    K_r_SC2 = np.array([0, 0, a2])

    mu = 0.3  # = mu1 = mu2
    e_N = 0  # = eN1 = eN2
    e_F = 0
    R = 5e-4  # m
    prox_r = 0.001

    # Initial conditions
    # all zeros exept:
    # Leine2003
    # z0 = 1.2015e-2  # m
    z0 = 0
    theta0 = 0.1  # rad
    psi_dot0 = 180  # rad / s

    # initial conditions:
    if RigidBodyBase is RigidBodyQuaternion:
        q0 = np.zeros(7, dtype=float)
        q0[2] = z0
        q0[3:] = axis_angle2quat(np.array([1, 0, 0]), theta0)
    elif RigidBodyBase is RigidBodyEuler:
        axis = "zxz"
        q0 = np.zeros(6, dtype=float)
        q0[2] = z0
        q0[4] = theta0

    u0 = np.zeros(6, dtype=float)
    u0[5] = psi_dot0

    if RigidBodyBase is RigidBodyQuaternion:
        top = RigidBodyQuaternion(m, K_theta_S, q0=q0, u0=u0)
    elif RigidBodyBase is RigidBodyEuler:
        top = RigidBodyEuler(m, K_theta_S, axis=axis, q0=q0, u0=u0)

    contact1 = Sphere2PlaneCoulombContensouMoeller(
        system.origin, top, R1, R, mu, e_N, e_F, K_r_SP=K_r_SC1
    )
    contact2 = Sphere2PlaneCoulombContensouMoeller(
        system.origin, top, R2, R, mu, e_N, e_F, K_r_SP=K_r_SC2
    )

    gravity = Force(np.array([0, 0, -m * g]), top)

    system.add(top, contact1, contact2, gravity)
    system.assemble()

    return system, top, contact1, contact2


def run(export=True):
    """Example 10.6 of Capobianco2021.

    References:
    -----------
    Capobianco2021: https://doi.org/10.1002/nme.6801 \\
    Moeller2009: https://www.inm.uni-stuttgart.de/institut/mitarbeiter/leine/papers/proceedings/Moeller_x_Leine_x_Glocker_-_An_efficient_approximation_of_set-valued_force_laws_of_normal_cone_type_ESMC2009.pdf \\
    Leine2003: https://doi.org/10.1016/S0997-7538(03)00025-1
    """

    system, top, contact1, contact2 = make_system(RigidBodyQuaternion)

    t0 = 0
    # t_final = 8
    # t_final = 2
    t_final = 0.1
    # dt1 = 1e-3
    # dt1 = 1e-4
    # dt2 = 1e-4
    dt1 = 1e-2
    dt2 = 1e-2

    sol1, label1 = Rattle(system, t_final, dt1, atol=1e-8).solve(), "Rattle"
    # sol1, label1 = (
    #     MoreauShiftedNew(system, t_final, dt2, atol=1e-6).solve(),
    #     "Moreau_new",
    # )
    sol2, label2 = (
        MoreauClassical(system, t_final, dt2, atol=1e-6).solve(),
        "Moreau",
    )

    t1 = sol1.t
    q1 = sol1.q
    u1 = sol1.u
    P_N1 = sol1.P_N
    P_F1 = sol1.P_F

    t2 = sol2.t
    q2 = sol2.q
    u2 = sol2.u
    P_N2 = sol2.P_N
    P_F2 = sol2.P_F

    fig, ax = plt.subplots(2, 3)

    ax[0, 0].set_title("x(t)")
    ax[0, 0].plot(t1, q1[:, 0], "-k", label=label1)
    ax[0, 0].plot(t2, q2[:, 0], "--r", label=label2)
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].set_title("y(t)")
    ax[0, 1].plot(t1, q1[:, 1], "-k", label=label1)
    ax[0, 1].plot(t2, q2[:, 1], "--r", label=label2)
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[0, 2].set_title("z(t)")
    ax[0, 2].plot(t1, q1[:, 2], "-k", label=label1)
    ax[0, 2].plot(t2, q2[:, 2], "--r", label=label2)
    ax[0, 2].grid()
    ax[0, 2].legend()

    ax[1, 0].set_title("u_x(t)")
    ax[1, 0].plot(t1, u1[:, 0], "-k", label=label1)
    ax[1, 0].plot(t2, u2[:, 0], "--r", label=label2)
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[1, 1].set_title("u_y(t)")
    ax[1, 1].plot(t1, u1[:, 1], "-k", label=label1)
    ax[1, 1].plot(t2, u2[:, 1], "--r", label=label2)
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[1, 2].set_title("u_z(t)")
    ax[1, 2].plot(t1, u1[:, 2], "-k", label=label1)
    ax[1, 2].plot(t2, u2[:, 2], "--r", label=label2)
    ax[1, 2].grid()
    ax[1, 2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(2, 3)

    nt1 = len(t1)
    angles1 = np.zeros((nt1, 3), dtype=float)
    for i in range(len(t1)):
        A_IK = top.A_IK(t1[i], q1[i])
        angles1[i] = Rotation.from_matrix(A_IK).as_euler("zxz")

    nt2 = len(t2)
    angles2 = np.zeros((nt2, 3), dtype=float)
    for i in range(len(t2)):
        A_IK = top.A_IK(t2[i], q2[i])
        angles2[i] = Rotation.from_matrix(A_IK).as_euler("zxz")

    ax[0, 0].set_title("psi(t)")
    ax[0, 0].plot(t1, angles1[:, 0], "-k", label=label1)
    ax[0, 0].plot(t2, angles2[:, 0], "--r", label=label2)
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].set_title("theta(t)")
    ax[0, 1].plot(t1, angles1[:, 1], "-k", label=label1)
    ax[0, 1].plot(t2, angles2[:, 1], "--r", label=label2)
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[0, 2].set_title("phi(t)")
    ax[0, 2].plot(t1, angles1[:, 2], "-k", label=label1)
    ax[0, 2].plot(t2, angles2[:, 2], "--r", label=label2)
    ax[0, 2].legend()
    ax[0, 2].grid()

    ax[1, 0].set_title("omega1(t)")
    ax[1, 0].plot(t1, u1[:, 3], "-k", label=label1)
    ax[1, 0].plot(t2, u2[:, 3], "--r", label=label2)
    ax[1, 0].legend()
    ax[1, 0].grid()

    ax[1, 1].set_title("omega2(t)")
    ax[1, 1].plot(t1, u1[:, 4], "-k", label=label1)
    ax[1, 1].plot(t2, u2[:, 4], "--r", label=label2)
    ax[1, 1].grid()
    ax[1, 1].legend()

    ax[1, 2].set_title("omega3(t)")
    ax[1, 2].plot(t1, u1[:, 5], "-k", label=label1)
    ax[1, 2].plot(t2, u2[:, 5], "--r", label=label2)
    ax[1, 2].grid()
    ax[1, 2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(4)

    ax[0].set_title("P_N(t)")
    ax[0].plot(t1, P_N1[:, 0], "-k", label=label1)
    ax[0].plot(t2, P_N2[:, 0], "--r", label=label2)
    ax[0].grid()
    ax[0].legend()

    ax[1].set_title("P_Fx(t)")
    ax[1].plot(t1, P_F1[:, 0], "-k", label=label1)
    ax[1].plot(t2, P_F2[:, 0], "--r", label=label2)
    ax[1].grid()
    ax[1].legend()

    ax[2].set_title("P_Fy(t)")
    ax[2].plot(t1, P_F1[:, 1], "-k", label=label1)
    ax[2].plot(t2, P_F2[:, 1], "--r", label=label2)
    ax[2].grid()
    ax[2].legend()

    ax[3].set_title("P_drill(t)")
    ax[3].plot(t1, P_F1[:, 2], "-k", label=label1)
    ax[3].plot(t2, P_F2[:, 2], "--r", label=label2)
    ax[3].grid()
    ax[3].legend()

    plt.tight_layout()

    if export:
        path = Path(__file__)

        np.savetxt(
            path.parent / "state1.dat",
            np.hstack((sol1.t[:, None], q1[:, :3], angles1)),
            delimiter=", ",
            header="t, x, y, z, psi, theta, phi",
            comments="",
        )

        np.savetxt(
            path.parent / "state2.dat",
            np.hstack((sol2.t[:, None], q2[:, :3], angles2)),
            delimiter=", ",
            header="t, x, y, z, psi, theta, phi",
            comments="",
        )

    plt.show()


def convergence(export=True):
    system, top, contact1, contact2 = make_system(RigidBodyQuaternion)

    tol_ref = 1.0e-8
    tol = 1.0e-8

    # compute step sizes with powers of 2
    dt_ref = 2.5e-5  # Arnold2015b
    dts = (2.0 ** np.arange(7, 1, -1)) * dt_ref  # [3.2e-3, ..., 2e-4, 1e-4]
    # dts = (2.0 ** np.arange(7, 4, -1)) * dt_ref  # [3.2e-3, 1.6e-3, 8e-4]

    # end time (note this has to be > 0.5, otherwise long term error throws ans error)
    # t1 = (2.0**9) * dt_ref  # this yields 0.256 for dt_ref = 5e-4
    # t1 = (2.0**10) * dt_ref  # this yields 0.512 for dt_ref = 5e-4
    # t1 = (2.0**11) * dt_ref  # this yields 0.2048 for dt_ref = 1e-4
    # t1 = (2.0**13) * dt_ref  # this yields 0.8192 for dt_ref = 1e-4
    t1 = (2.0**15) * dt_ref  # this yields 0.8192 for dt_ref = 2.5e-5
    # # t1 = (2.0**16) * dt_ref # this yields 1.6384 for dt_ref = 2.5e-5

    # # TODO: Only for debugging!
    # dt_ref = 2.5e-3
    # dts = np.array([5.0e-3, 1e-2])
    # # t1 = (2.0**8) * dt_ref
    # t1 = (2.0**6) * dt_ref

    # compute step sizes with powers of 2
    dt_ref = 1e-4
    # t1 = (2.0**12) * dt_ref # 0.4096 s
    # t1 = (2.0**10) * dt_ref # 0.1024 s
    t1 = (2.0**9) * dt_ref  # 0.0512 s
    dts = (2.0 ** np.arange(6, 4, -1)) * dt_ref

    print(f"t1: {t1}")
    print(f"dts: {dts}")
    # exit()

    # errors for possible solvers
    q_errors_transient = np.inf * np.ones((2, len(dts)), dtype=float)
    u_errors_transient = np.inf * np.ones((2, len(dts)), dtype=float)
    P_N_errors_transient = np.inf * np.ones((2, len(dts)), dtype=float)
    P_F_errors_transient = np.inf * np.ones((2, len(dts)), dtype=float)
    q_errors_longterm = np.inf * np.ones((2, len(dts)), dtype=float)
    u_errors_longterm = np.inf * np.ones((2, len(dts)), dtype=float)
    P_N_errors_longterm = np.inf * np.ones((2, len(dts)), dtype=float)
    P_F_errors_longterm = np.inf * np.ones((2, len(dts)), dtype=float)

    ###################################################################
    # compute reference solution as described in Arnold2015 Section 3.3
    ###################################################################
    # print(f"compute reference solution with classical Moreau:")
    # reference = MoreauClassical(
    #     system,
    #     t1,
    #     dt_ref,
    #     atol=tol_ref
    # ).solve()
    print(f"compute reference solution with rattle:")
    reference = Rattle(system, t1, dt_ref, atol=tol_ref).solve()
    print(f"done")

    plot_state = True
    if plot_state:
        t_ref = reference.t
        q_ref = reference.q
        u_ref = reference.u
        P_N_ref = reference.P_N
        P_F_ref = reference.P_F

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
        nt_ref = len(t_ref)
        angles_ref = np.zeros((nt_ref, 3), dtype=float)
        for i in range(nt_ref):
            A_IK = top.A_IK(t_ref[i], q_ref[i])
            angles_ref[i] = Rotation.from_matrix(A_IK).as_euler("zxz")

        ax = fig.add_subplot(2, 3, 2)
        ax.plot(t_ref, angles_ref[:, 0], "-r", label="alpha")
        ax.plot(t_ref, angles_ref[:, 1], "-g", label="beta")
        ax.plot(t_ref, angles_ref[:, 2], "-b", label="gamma")
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

        # percussions
        ax = fig.add_subplot(2, 3, 6)
        ax.plot(t_ref, P_N_ref[:, 0], "-r", label="P_N")
        ax.plot(t_ref, P_F_ref[:, 0], "-g", label="P_F0")
        ax.plot(t_ref, P_F_ref[:, 1], "-b", label="P_F1")
        ax.grid()
        ax.legend()

        plt.show()

    exit()

    # TODO: Adapt bounds
    # def errors(sol, sol_ref, t_transient=0.01, t_longterm=0.05):
    def errors(sol, sol_ref, t_transient=0.5 * t1, t_longterm=0.5 * t1):
        t = sol.t
        q = sol.q
        u = sol.u
        P_N = sol.P_N
        P_F = sol.P_F

        t_ref = sol_ref.t
        q_ref = sol_ref.q
        u_ref = sol_ref.u
        P_N_ref = sol_ref.P_N
        P_F_ref = sol_ref.P_F

        # distinguish between transient and long term time steps
        t_idx_transient = np.where(t <= t_transient)[0]
        t_idx_longterm = np.where(t >= t_longterm)[0]

        # compute difference between computed solution and reference solution
        # for identical time instants
        t_ref_idx_transient = np.where(
            np.abs(t[t_idx_transient, None] - t_ref) < 1.0e-8
        )[1]
        t_ref_idx_longterm = np.where(np.abs(t[t_idx_longterm, None] - t_ref) < 1.0e-8)[
            1
        ]

        # differences
        q_transient = q[t_idx_transient]
        u_transient = u[t_idx_transient]
        P_N_transient = P_N[t_idx_transient]
        P_F_transient = P_F[t_idx_transient]
        diff_transient_q = q_transient - q_ref[t_ref_idx_transient]
        diff_transient_u = u_transient - u_ref[t_ref_idx_transient]
        diff_transient_P_N = P_N_transient - P_N_ref[t_ref_idx_transient]
        diff_transient_P_F = P_F_transient - P_F_ref[t_ref_idx_transient]

        q_longterm = q[t_idx_longterm]
        u_longterm = u[t_idx_longterm]
        P_N_longterm = P_N[t_idx_longterm]
        P_F_longterm = P_F[t_idx_longterm]
        diff_longterm_q = q_longterm - q_ref[t_ref_idx_longterm]
        diff_longterm_u = u_longterm - u_ref[t_ref_idx_longterm]
        diff_longterm_P_N = P_N_longterm - P_N_ref[t_ref_idx_longterm]
        diff_longterm_P_F = P_F_longterm - P_F_ref[t_ref_idx_longterm]

        # max relative error
        q_error_transient = np.max(
            np.linalg.norm(diff_transient_q, axis=1)
            / np.linalg.norm(q_transient, axis=1)
        )
        u_error_transient = np.max(
            np.linalg.norm(diff_transient_u, axis=1)
            / np.linalg.norm(u_transient, axis=1)
        )
        P_N_error_transient = np.max(
            np.linalg.norm(diff_transient_P_N, axis=1)
            / np.linalg.norm(P_N_transient, axis=1)
        )
        P_F_error_transient = np.max(
            np.linalg.norm(diff_transient_P_F, axis=1)
            / np.linalg.norm(P_F_transient, axis=1)
        )

        q_error_longterm = np.max(
            np.linalg.norm(diff_longterm_q, axis=1) / np.linalg.norm(q_longterm, axis=1)
        )
        u_error_longterm = np.max(
            np.linalg.norm(diff_longterm_u, axis=1) / np.linalg.norm(u_longterm, axis=1)
        )
        P_N_error_longterm = np.max(
            np.linalg.norm(diff_longterm_P_N, axis=1)
            / np.linalg.norm(P_N_longterm, axis=1)
        )
        P_F_error_longterm = np.max(
            np.linalg.norm(diff_longterm_P_F, axis=1)
            / np.linalg.norm(P_F_longterm, axis=1)
        )

        return (
            q_error_transient,
            u_error_transient,
            P_N_error_transient,
            P_F_error_transient,
            q_error_longterm,
            u_error_longterm,
            P_N_error_longterm,
            P_F_error_longterm,
        )

    for i, dt in enumerate(dts):
        print(f"i: {i}, dt: {dt:1.1e}")

        sol = MoreauClassical(system, t1, dt, atol=tol).solve()
        (
            q_errors_transient[0, i],
            u_errors_transient[0, i],
            P_N_errors_transient[0, i],
            P_F_errors_transient[0, i],
            q_errors_longterm[0, i],
            u_errors_longterm[0, i],
            P_N_errors_longterm[0, i],
            P_F_errors_longterm[0, i],
        ) = errors(sol, reference)

        sol = Rattle(system, t1, dt, atol=tol).solve()
        (
            q_errors_transient[1, i],
            u_errors_transient[1, i],
            P_N_errors_transient[1, i],
            P_F_errors_transient[1, i],
            q_errors_longterm[1, i],
            u_errors_longterm[1, i],
            P_N_errors_longterm[1, i],
            P_F_errors_longterm[1, i],
        ) = errors(sol, reference)

    # #############################
    # # export errors and dt, dt**2
    # #############################
    # header = "dt, dt2, 2nd, 1st, 1st_GGL"

    # # transient errors
    # export_data = np.vstack((dts, dts**2, *q_errors_transient)).T
    # np.savetxt(
    #     "transient_error_heavy_top_q.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    # export_data = np.vstack((dts, dts**2, *u_errors_transient)).T
    # np.savetxt(
    #     "transient_error_heavy_top_u.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    # export_data = np.vstack((dts, dts**2, *P_N_errors_transient)).T
    # np.savetxt(
    #     "transient_error_heavy_top_la_g.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    # # longterm errors
    # export_data = np.vstack((dts, dts**2, *q_errors_longterm)).T
    # np.savetxt(
    #     "longterm_error_heavy_top_q.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    # export_data = np.vstack((dts, dts**2, *u_errors_longterm)).T
    # np.savetxt(
    #     "longterm_error_heavy_top_u.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    # export_data = np.vstack((dts, dts**2, *P_N_errors_longterm)).T
    # np.savetxt(
    #     "longterm_error_heavy_top_la_g.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    ##################
    # visualize errors
    ##################
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].set_title("transient: Moreau")
    ax[0, 0].loglog(dts, dts, "-k", label="dt")
    ax[0, 0].loglog(dts, dts**2, "--k", label="dt^2")
    ax[0, 0].loglog(dts, q_errors_transient[0], "-.ro", label="q")
    ax[0, 0].loglog(dts, u_errors_transient[0], "-.go", label="u")
    ax[0, 0].loglog(dts, P_N_errors_transient[0], "-.bo", label="P_N")
    ax[0, 0].loglog(dts, P_F_errors_transient[0], "-.mo", label="P_F")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[1, 0].set_title("transient: Rattle")
    ax[1, 0].loglog(dts, dts, "-k", label="dt")
    ax[1, 0].loglog(dts, dts**2, "--k", label="dt^2")
    ax[1, 0].loglog(dts, q_errors_transient[1], "-.ro", label="q")
    ax[1, 0].loglog(dts, u_errors_transient[1], "-.go", label="u")
    ax[1, 0].loglog(dts, P_N_errors_transient[1], "-.bo", label="P_N")
    ax[1, 0].loglog(dts, P_F_errors_transient[1], "-.mo", label="P_F")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[0, 1].set_title("long term: Moreau")
    ax[0, 1].loglog(dts, dts, "-k", label="dt")
    ax[0, 1].loglog(dts, dts**2, "--k", label="dt^2")
    ax[0, 1].loglog(dts, q_errors_longterm[0], "-.ro", label="q")
    ax[0, 1].loglog(dts, u_errors_longterm[0], "-.go", label="u")
    ax[0, 1].loglog(dts, P_N_errors_longterm[0], "-.bo", label="P_N")
    ax[0, 1].loglog(dts, P_F_errors_longterm[0], "-.mo", label="P_F")
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 1].set_title("long term: Rattle")
    ax[1, 1].loglog(dts, dts, "-k", label="dt")
    ax[1, 1].loglog(dts, dts**2, "--k", label="dt^2")
    ax[1, 1].loglog(dts, q_errors_longterm[1], "-.ro", label="q")
    ax[1, 1].loglog(dts, u_errors_longterm[1], "-.go", label="u")
    ax[1, 1].loglog(dts, P_N_errors_longterm[1], "-.bo", label="P_N")
    ax[1, 1].loglog(dts, P_F_errors_longterm[1], "-.mo", label="P_F")
    ax[1, 1].grid()
    ax[1, 1].legend()

    plt.show()


if __name__ == "__main__":
    # run()
    convergence()
