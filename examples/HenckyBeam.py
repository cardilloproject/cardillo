import numpy as np
from math import atan2

from cardillo.math import norm, e2
from cardillo.model import System
from cardillo.model.frame import Frame
from cardillo.model.point_mass import PointMass
from cardillo.model.bilateral_constraints.implicit import SphericalJoint2D
from cardillo.forces import Force
from cardillo.solver import Newton, ScipyIVP

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class QuadraticPotential:
    def __init__(self, k):
        self.k = k

    def potential(self, g, g0):
        return self.k * (g - g0) ** 2

    def potential_g(self, g, g0):
        return self.k * (g - g0)

    def potential_gg(self, g, g0):
        return self.k


class TranslationalSpring:
    def __init__(
        self,
        material_model,
        subsystem1,
        subsystem2,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
        K_r_SP1=np.zeros(3),
        K_r_SP2=np.zeros(3),
    ):

        self.material_model = material_model

        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.K_r_SP1 = K_r_SP1

        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2
        self.K_r_SP2 = K_r_SP2

    def assembler_callback(self):
        r_OP10 = self.subsystem1.r_OP(
            self.subsystem1.t0, self.subsystem1.q0, self.frame_ID1, self.K_r_SP1
        )
        r_OP20 = self.subsystem2.r_OP(
            self.subsystem2.t0, self.subsystem2.q0, self.frame_ID2, self.K_r_SP2
        )
        self.g0 = norm(r_OP20 - r_OP10)

        self.qDOF1 = self.subsystem1.qDOF[self.subsystem1.local_qDOF_P(self.frame_ID1)]
        self.qDOF2 = self.subsystem2.qDOF[self.subsystem2.local_qDOF_P(self.frame_ID2)]
        self.qDOF = np.concatenate([self.qDOF1, self.qDOF2])
        self.nq1 = len(self.qDOF1)
        self.nq2 = len(self.qDOF2)
        self.nq = self.nq1 + self.nq2

        self.uDOF1 = self.subsystem1.uDOF[self.subsystem1.local_uDOF_P(self.frame_ID1)]
        self.uDOF2 = self.subsystem2.uDOF[self.subsystem2.local_uDOF_P(self.frame_ID2)]
        self.uDOF = np.concatenate([self.uDOF1, self.uDOF2])
        self.nu1 = len(self.uDOF1)
        self.nu2 = len(self.uDOF2)
        self.nu = self.nu1 + self.nu2

        self.r_OP1 = lambda t, q: self.subsystem1.r_OP(
            t, q[: self.nq1], self.frame_ID1, self.K_r_SP1
        )
        self.r_OP1_q = lambda t, q: self.subsystem1.r_OP_q(
            t, q[: self.nq1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1 = lambda t, q: self.subsystem1.J_P(
            t, q[: self.nq1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1_q = lambda t, q: self.subsystem1.J_P_q(
            t, q[: self.nq1], self.frame_ID1, self.K_r_SP1
        )

        self.r_OP2 = lambda t, q: self.subsystem2.r_OP(
            t, q[self.nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.r_OP2_q = lambda t, q: self.subsystem2.r_OP_q(
            t, q[self.nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2 = lambda t, q: self.subsystem2.J_P(
            t, q[self.nq1 :], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2_q = lambda t, q: self.subsystem2.J_P_q(
            t, q[self.nq1 :], self.frame_ID2, self.K_r_SP2
        )

    def __g(self, t, q):
        return norm(self.r_OP2(t, q) - self.r_OP1(t, q))

    def __g_q(self, t, q):
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)

        n = self.__n(t, q)
        return np.hstack((-n @ r_OP1_q, n @ r_OP2_q))

    def __W(self, t, q):
        n = self.__n(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        return np.concatenate([-J_P1.T @ n, J_P2.T @ n])

    def __W_q(self, t, q):
        nq1 = self.nq1
        nu1 = self.nu1
        n = self.__n(t, q)
        n_q1, n_q2 = self.__n_q(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        J_P1_q = self.J_P1_q(t, q)
        J_P2_q = self.J_P2_q(t, q)

        # dense blocks
        dense = np.zeros((self.nu, self.nq))
        dense[:nu1, :nq1] = -J_P1.T @ n_q1 + np.einsum("i,ijk->jk", -n, J_P1_q)
        dense[:nu1, nq1:] = -J_P1.T @ n_q2
        dense[nu1:, :nq1] = J_P2.T @ n_q1
        dense[nu1:, nq1:] = J_P2.T @ n_q2 + np.einsum("i,ijk->jk", n, J_P2_q)

        return dense

    def __n(self, t, q):
        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        return (r_OP2 - r_OP1) / norm(r_OP2 - r_OP1)

    def __n_q(self, t, q):
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)

        r_P1P2 = self.r_OP2(t, q) - self.r_OP1(t, q)
        g = norm(r_P1P2)
        tmp = np.outer(r_P1P2, r_P1P2) / (g**3)
        n_q1 = -r_OP1_q / g + tmp @ r_OP1_q
        n_q2 = r_OP2_q / g - tmp @ r_OP2_q

        return n_q1, n_q2

    # public functions
    def pot(self, t, q):
        g = self.__g(t, q)
        return self.material_model.potential(g, self.g0)

    def h(self, t, q, u):
        g = self.__g(t, q)
        potential_g = self.material_model.potential_g(g, self.g0)
        return -self.__W(t, q) * potential_g
        # return -potential_g * self.__g_q(t, q)

    def h_q(self, t, q, u, coo):
        g = self.__g(t, q)
        potential_g = self.material_model.potential_g(g, self.g0)
        potential_gg = self.material_model.potential_gg(g, self.g0)
        dense = -self.__W_q(t, q) * potential_g - potential_gg * np.outer(
            self.__W(t, q), self.__g_q(t, q)
        )
        coo.extend(dense, (self.uDOF, self.qDOF))


class TorsionalSpring3Point2D:
    def __init__(
        self,
        material_model,
        subsystem1,
        subsystem2,
        subsystem3,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
        frame_ID3=np.zeros(3),
        K_r_SP1=np.zeros(3),
        K_r_SP2=np.zeros(3),
        K_r_SP3=np.zeros(3),
        method="relative",
    ):

        self.material_model = material_model

        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.K_r_SP1 = K_r_SP1

        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2
        self.K_r_SP2 = K_r_SP2

        self.subsystem3 = subsystem3
        self.frame_ID3 = frame_ID3
        self.K_r_SP3 = K_r_SP3

        if method == "absolute":
            self.relative_angle = self.__relative_angle_absolute
        else:
            self.relative_angle = self.__relative_angle_relative

    def __relative_angle_absolute(self, r_OP1, r_OP2, r_OP3):
        # discrete tangent vectors
        r_P1P2 = r_OP2 - r_OP1
        r_P2P3 = r_OP3 - r_OP2

        # relative rotation angle from difference of both absolute ones
        phi1 = atan2(r_P1P2[1], r_P1P2[0])
        phi2 = atan2(r_P2P3[1], r_P2P3[0])
        return phi2 - phi1

    def __relative_angle_relative(self, r_OP1, r_OP2, r_OP3):
        # discrete tangent vectors
        r_P1P2 = (r_OP2 - r_OP1)[:2]
        r_P2P3 = (r_OP3 - r_OP2)[:2]

        # direct computation of relative rotation angle
        # A = np.array([[0, -1],
        #               [1, 0]])
        # r_P1P2_perp = A @ r_P1P2
        r_P1P2_perp = np.array([-r_P1P2[1], r_P1P2[0]])
        return atan2(r_P2P3 @ r_P1P2_perp, r_P2P3 @ r_P1P2)

    def assembler_callback(self):
        # initial relative angle of springs
        r_OP10 = self.subsystem1.r_OP(
            self.subsystem1.t0, self.subsystem1.q0, self.frame_ID1, self.K_r_SP1
        )
        r_OP20 = self.subsystem2.r_OP(
            self.subsystem2.t0, self.subsystem2.q0, self.frame_ID2, self.K_r_SP2
        )
        r_OP30 = self.subsystem3.r_OP(
            self.subsystem3.t0, self.subsystem3.q0, self.frame_ID3, self.K_r_SP3
        )
        self.g0 = self.relative_angle(r_OP10, r_OP20, r_OP30)

        self.qDOF1 = self.subsystem1.qDOF[self.subsystem1.local_qDOF_P(self.frame_ID1)]
        self.qDOF2 = self.subsystem2.qDOF[self.subsystem2.local_qDOF_P(self.frame_ID2)]
        self.qDOF3 = self.subsystem3.qDOF[self.subsystem3.local_qDOF_P(self.frame_ID3)]
        self.qDOF = np.concatenate([self.qDOF1, self.qDOF2, self.qDOF3])
        self.nq1 = len(self.qDOF1)
        self.nq2 = len(self.qDOF2)
        self.nq3 = len(self.qDOF3)
        self.nq = self.nq1 + self.nq2 + self.nq3

        self.uDOF1 = self.subsystem1.uDOF[self.subsystem1.local_uDOF_P(self.frame_ID1)]
        self.uDOF2 = self.subsystem2.uDOF[self.subsystem2.local_uDOF_P(self.frame_ID2)]
        self.uDOF3 = self.subsystem3.uDOF[self.subsystem3.local_uDOF_P(self.frame_ID3)]
        self.uDOF = np.concatenate([self.uDOF1, self.uDOF2, self.uDOF3])
        self.nu1 = len(self.uDOF1)
        self.nu2 = len(self.uDOF2)
        self.nu3 = len(self.uDOF3)
        self.nu = self.nu1 + self.nu2 + self.nu3

        nq1 = self.nq1
        nq2 = self.nq2
        nq12 = nq1 + nq2

        self.r_OP1 = lambda t, q: self.subsystem1.r_OP(
            t, q[:nq1], self.frame_ID1, self.K_r_SP1
        )
        self.r_OP1_q = lambda t, q: self.subsystem1.r_OP_q(
            t, q[:nq1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1 = lambda t, q: self.subsystem1.J_P(
            t, q[:nq1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1_q = lambda t, q: self.subsystem1.J_P_q(
            t, q[:nq1], self.frame_ID1, self.K_r_SP1
        )

        self.r_OP2 = lambda t, q: self.subsystem2.r_OP(
            t, q[nq1:nq12], self.frame_ID2, self.K_r_SP2
        )
        self.r_OP2_q = lambda t, q: self.subsystem2.r_OP_q(
            t, q[nq1:nq12], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2 = lambda t, q: self.subsystem2.J_P(
            t, q[nq1:nq12], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2_q = lambda t, q: self.subsystem2.J_P_q(
            t, q[nq1:nq12], self.frame_ID2, self.K_r_SP2
        )

        self.r_OP3 = lambda t, q: self.subsystem3.r_OP(
            t, q[nq12:], self.frame_ID3, self.K_r_SP3
        )
        self.r_OP3_q = lambda t, q: self.subsystem3.r_OP_q(
            t, q[nq12:], self.frame_ID3, self.K_r_SP3
        )
        self.J_P3 = lambda t, q: self.subsystem3.J_P(
            t, q[nq12:], self.frame_ID3, self.K_r_SP3
        )
        self.J_P3_q = lambda t, q: self.subsystem3.J_P_q(
            t, q[nq12:], self.frame_ID3, self.K_r_SP3
        )

    def __g(self, t, q):
        return self.relative_angle(self.r_OP1(t, q), self.r_OP2(t, q), self.r_OP3(t, q))

    def __g_q(self, t, q):
        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        r_OP3 = self.r_OP3(t, q)

        # discrete tangent vectors
        r_P1P2 = (r_OP2 - r_OP1)[:2]
        r_P2P3 = (r_OP3 - r_OP2)[:2]

        # squared lengths
        l12 = r_P1P2 @ r_P1P2
        l22 = r_P2P3 @ r_P2P3

        # derivative of rotation angle from both absolute ones
        phi1_q = np.array([r_P1P2[1], -r_P1P2[0], -r_P1P2[1], r_P1P2[0], 0, 0]) / l12
        phi2_q = np.array([0, 0, r_P2P3[1], -r_P2P3[0], -r_P2P3[1], r_P2P3[0]]) / l22

        Delta_phi_q = phi2_q - phi1_q
        return Delta_phi_q

    def __g_qq(self, t, q):
        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        r_OP3 = self.r_OP3(t, q)

        # discrete tangent vectors
        r_P1P2 = r_OP2 - r_OP1
        r_P2P3 = r_OP3 - r_OP2

        # squared lengths
        l12 = r_P1P2 @ r_P1P2
        l22 = r_P2P3 @ r_P2P3

        tmp = np.array([[0, -1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0]])

        tmp1 = np.zeros((6, 6))
        tmp1[:4, :4] = tmp / l12
        tmp2 = np.zeros((6, 6))
        tmp2[2:, 2:] = tmp / l22

        l12_q = np.array([-r_P1P2[0], -r_P1P2[1], r_P1P2[0], r_P1P2[1], 0, 0])
        l22_q = np.array([0, 0, -r_P2P3[0], -r_P2P3[1], r_P2P3[0], r_P2P3[1]])

        phi1_q = np.array([r_P1P2[1], -r_P1P2[0], -r_P1P2[1], r_P1P2[0], 0, 0]) / l12
        phi2_q = np.array([0, 0, r_P2P3[1], -r_P2P3[0], -r_P2P3[1], r_P2P3[0]]) / l22
        w = phi2_q - phi1_q
        phi1_qq = tmp1 - np.outer(w, l12_q) / (l12 * l12)
        phi2_qq = tmp2 - np.outer(w, l22_q) / (l22 * l22)

        return phi2_qq - phi1_qq

    # public functions
    def pot(self, t, q):
        g = self.__g(t, q)
        return self.material_model.potential(g, self.g0)

    def h(self, t, q, u):
        g = self.__g(t, q)
        potential_g = self.material_model.potential_g(g, self.g0)
        f_pot = -potential_g * self.__g_q(t, q)
        return f_pot

    def h_q(self, t, q, u, coo):
        g = self.__g(t, q)
        potential_g = self.material_model.potential_g(g, self.g0)
        potential_gg = self.material_model.potential_gg(g, self.g0)
        g_q = self.__g_q(t, q)
        g_qq = self.__g_qq(t, q)

        dense = -potential_g * g_qq - potential_gg * np.outer(g_q, g_q)
        coo.extend(dense, (self.uDOF, self.qDOF))


class HenckyBeam:
    def __init__(self, M, Ke, Kt, L, nEl):
        # compute microscopic axial stiffness from global one
        self.ke = (Ke / L) * nEl

        # compute microscopic torsional stiffness from global one
        self.kt = (Kt / L) * (nEl - 1)

        # compute microscopic mass from global one
        self.m = (M / L) * (nEl + 1)

        self.L = L  # total length
        self.nEl = nEl  # number of elements
        self.nn = nEl + 1  # number of nodes

        # material model for translational and torsional springs
        self.translational_potential = QuadraticPotential(self.ke)
        self.torsional_potential = QuadraticPotential(self.kt)

        self.point_masses = []
        self.frames = []
        self.constraints = []
        self.translational_springs = []
        self.torsional_springs = []
        self.gravity = []

        # buil dpoint masses, translational and torsional springs
        self.__build_point_massese()
        self.__build_constraints()
        self.__build_translational_springs()
        self.__build_torsional_springs()

    def nodes(self, q):
        return q.reshape(-1, 2).T

    def __build_point_massese(self):
        Xi = np.linspace(0, self.L, num=self.nn)
        for i in range(self.nn):
            q0 = Xi[i] * np.array([1, 0])
            self.point_masses.append(PointMass(self.m, dim=2, q0=q0))

    def __build_constraints(self):
        for i in range(2):
            PM = self.point_masses[i]
            r_OP0 = PM.r_OP(0, PM.q0)
            frame = Frame(r_OP=r_OP0)
            self.frames.append(frame)
            self.constraints.append(SphericalJoint2D(frame, PM, r_OP0))

    def __build_translational_springs(self):
        for i in range(self.nn - 1):
            point_mass0 = self.point_masses[i]
            point_mass1 = self.point_masses[i + 1]
            self.translational_springs.append(
                TranslationalSpring(
                    self.translational_potential, point_mass0, point_mass1
                )
            )

    def __build_torsional_springs(self):
        for i in range(self.nn - 2):
            point_mass0 = self.point_masses[i]
            point_mass1 = self.point_masses[i + 1]
            point_mass2 = self.point_masses[i + 2]
            self.translational_springs.append(
                TorsionalSpring3Point2D(
                    self.torsional_potential, point_mass0, point_mass1, point_mass2
                )
            )


# statics = True
statics = False

if __name__ == "__main__":
    Ke = 20
    Kt = 5
    M = 1.0e-2
    g = 9.81
    L = 2 * np.pi
    nEl = 10

    henckyBeam = HenckyBeam(M, Ke, Kt, L, nEl)

    model = System()

    for pm in henckyBeam.point_masses:
        model.add(pm)
    for frame in henckyBeam.frames:
        model.add(frame)
    for constraint in henckyBeam.constraints:
        model.add(constraint)
    for ts in henckyBeam.translational_springs:
        model.add(ts)
    for ts in henckyBeam.torsional_springs:
        model.add(ts)

    # add gravity
    for pm in henckyBeam.point_masses:
        if statics:
            fg = lambda t: -t * henckyBeam.m * g * e2
        else:
            fg = lambda t: -henckyBeam.m * g * e2
        model.add(Force(fg, pm))

    model.assemble()

    if statics:
        n_load_steps = 10
        atol = 1.0e-8
        max_iter = 50
        sol = Newton(
            model, n_load_steps=n_load_steps, atol=atol, max_iter=max_iter
        ).solve()
    else:
        t1 = 5
        dt = 1.0e-2
        method = "RK45"
        rtol = 1.0e-6
        atol = 1.0e-6
        sol = ScipyIVP(model, t1, dt, method, rtol, atol).solve()

    t = sol.t
    q = sol.q

    # animate solution
    fig, ax = plt.subplots()
    ax.set_xlim(-0.3 * L, 1.1 * L)
    ax.set_ylim(-1.1 * L, 0.3 * L)
    (initial,) = ax.plot(*henckyBeam.nodes(q[0]), "-ob")
    (current,) = ax.plot(*henckyBeam.nodes(q[0]), "--xr")

    def animate(i):
        current.set_data(*henckyBeam.nodes(q[i]))

    frames = len(q) - 1
    animation_time = 2  # in seconds
    interval = animation_time * 1000 / frames
    anim = FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)

    plt.show()

    # r_OP0 = q[0].reshape(-1, 2)
    # r_OP1 = q[-1].reshape(-1, 2)

    # fig, ax = plt.subplots()
    # ax.plot(*r_OP0.T, "-ob")
    # ax.plot(*r_OP1.T, "-xr")
    # plt.show()
