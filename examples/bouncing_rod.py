import numpy as np
from math import pi

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math import e1, e2, e3

from cardillo import System
from cardillo.beams import (
    CircularCrossSection,
    Simo1986,
)

# from cardillo.beams import QuadraticMaterial
from cardillo.beams import (
    # Cable,
    DirectorAxisAngle,
    # animate_rope,
    animate_beam,
)
from cardillo.forces import DistributedForce1DBeam

from cardillo.discrete import Frame
from cardillo.contacts import Sphere2Plane

from cardillo.solver import (
    Moreau,
    NonsmoothGeneralizedAlpha,
    NonsmoothBackwardEulerDecoupled,
)


if __name__ == "__main__":
    animate = True
    # animate = False

    # discretization properties
    nelements = 3
    polynomial_degree_r = 2
    basis_r = "B-spline"
    polynomial_degree_psi = 2
    basis_psi = "B-spline"

    # # nelements = 2
    # nelements = 3
    # # polynomial_degree_r = 2
    # # # basis_r = "Hermite"
    # # # basis_r = "B-spline"
    # # basis_r = "Lagrange"
    # polynomial_degree_r = 1
    # basis_r = "Lagrange"
    # # polynomial_degree_r = 2
    # # basis_r = "Lagrange"
    # # polynomial_degree_psi = 1
    # # basis_psi = "Lagrange"
    # polynomial_degree_psi = 1
    # # basis_psi = "B-spline"
    # basis_psi = "Lagrange"

    # cross section and quadratic beam material
    L = np.pi
    line_density = 1.0e-2
    radius_rod = 1.0e-1
    # E = 1.0e2
    # G = 0.5e2
    E = 1.0e2
    G = 0.5e2
    E *= 2
    G *= 2
    cross_section = CircularCrossSection(line_density, radius_rod)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

    # starting point and corresponding orientation
    # r_OP0 = np.zeros(3, dtype=float)
    z0 = 0.25
    r_OP0 = np.array([-L / 2, 0, z0], dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    Q = DirectorAxisAngle.straight_configuration(
        polynomial_degree_r,
        polynomial_degree_psi,
        basis_r,
        basis_psi,
        nelements,
        L,
        r_OP0,
        A_IK0,
    )

    u0 = np.zeros_like(Q)
    u_z0 = -2
    n = int(len(Q) / 6)
    u0[2 * n : 3 * n] = u_z0

    rod = DirectorAxisAngle(
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree_r,
        polynomial_degree_psi,
        nelements,
        Q,
        q0=Q,
        u0=u0,
        basis_r=basis_r,
        basis_psi=basis_psi,
    )

    # gravity
    g = 9.81
    __fg = -A_rho0 * g * e3

    def fg(t, xi):
        return __fg

    gravity = DistributedForce1DBeam(fg, rod)

    mu = 0.1
    # mu = 0.0
    e_N = 0

    frame_left = Frame()
    plane_left = Sphere2Plane(
        frame_left,
        rod,
        radius_rod,
        mu,
        e_N=e_N,
        frame_ID=(0,),
    )

    frame_right = Frame()
    plane_right = Sphere2Plane(
        frame_right,
        rod,
        radius_rod,
        mu,
        e_N=e_N,
        frame_ID=(1,),
    )

    model = System()
    model.add(rod)
    # model.add(gravity)
    model.add(plane_left)
    model.add(plane_right)
    model.assemble()

    t0 = 0
    # t1 = 0.1
    t1 = 0.5
    # t1 = 1.0
    # t1 = 2

    # dt = 5e-2
    dt = 1e-2
    # dt = 5e-3
    # dt = 2.5e-3
    # dt = 1e-3
    # dt = 5e-4 # Moreau
    # dt = 1e-4

    # solver = NonsmoothGeneralizedAlpha(model, t1, dt, newton_max_iter=10)
    solver = NonsmoothBackwardEulerDecoupled(model, t1, dt, tol=1.0e-6)
    # solver = Moreau(model, t1, dt, fix_point_max_iter=100)
    sol = solver.solve()

    t = sol.t
    q = sol.q
    u = sol.u
    scale = L
    fig, ax, anim = animate_beam(t, q, [rod], scale, show=False)

    X = np.linspace(-L, L, num=2)
    Y = np.linspace(-L, L, num=2)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.5)

    ########################
    # extract contact forces
    ########################
    la_N = sol.la_N
    la_F = sol.la_F

    la_N_left = la_N[:, plane_left.la_NDOF]
    la_F_left = la_F[:, plane_left.la_FDOF]
    la_N_right = la_N[:, plane_right.la_NDOF]
    la_F_right = la_F[:, plane_right.la_FDOF]

    ##############
    # compute gaps
    ##############
    g_N_left = np.array(
        [plane_left.g_N(ti, qi[plane_left.qDOF]) for (ti, qi) in zip(t, q)]
    )
    g_N_right = np.array(
        [plane_right.g_N(ti, qi[plane_right.qDOF]) for (ti, qi) in zip(t, q)]
    )

    gamma_F_left = np.array(
        [
            plane_left.gamma_F(ti, qi[plane_left.qDOF], ui[plane_left.uDOF])
            for (ti, qi, ui) in zip(t, q, u)
        ]
    )
    gamma_F_right = np.array(
        [
            plane_right.gamma_F(ti, qi[plane_right.qDOF], ui[plane_right.uDOF])
            for (ti, qi, ui) in zip(t, q, u)
        ]
    )

    ################
    # visualize gaps
    ################
    fig = plt.figure()

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.plot(t, g_N_left, label="g_N_left")
    ax0.grid()
    ax0.legend()

    ax1 = fig.add_subplot(2, 2, 2)
    ax1.plot(t, g_N_right, label="g_N_right")
    ax1.grid()
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(t, gamma_F_left[:, 0], "-r", label="gamma_F_left0")
    ax2.plot(t, gamma_F_left[:, 1], "--b", label="gamma_F_left1")
    ax2.grid()
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(t, gamma_F_right[:, 0], "-r", label="gamma_F_right0")
    ax3.plot(t, gamma_F_right[:, 1], "--b", label="gamma_F_right1")
    ax3.grid()
    ax3.legend()

    ##########################
    # visualize contact forces
    ##########################
    fig = plt.figure()

    ax0 = fig.add_subplot(2, 2, 1)
    # ax0.tick_params('x', labelbottom=False)
    ax0.plot(t, la_N_left[:, 0], label="la_N0 - left")
    ax0.grid()
    ax0.legend()

    ax1 = fig.add_subplot(2, 2, 3, sharex=ax0)
    ax1.plot(t, la_F_left[:, 0], "-r", label="la_F0 - left")
    ax1.plot(t, la_F_left[:, 1], "--b", label="la_F1 - left")
    ax1.grid()
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    # ax2.tick_params('x', labelbottom=False)
    ax2.plot(t, la_N_right[:, 0], label="la_N0 - right")
    ax2.grid()
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(t, la_F_right[:, 0], "-r", label="la_F0 - right")
    ax3.plot(t, la_F_right[:, 1], "--b", label="la_F1 - right")
    ax3.grid()
    ax3.legend()

    plt.tight_layout()

    plt.show()

    # fig, ax = plt.subplots(2, 2)

    # ax[0, 0].plot(t, la_N[:, 0], label="la_N0 - left")
    # ax[0, 0].grid()
    # ax[0, 0].legend()

    # ax[1, 0].plot(t, la_F[:, 0], "-r", label="la_F0 - left")
    # ax[1, 0].plot(t, la_F[:, 1], "--b", label="la_F1 - left")
    # ax[1, 0].grid()
    # ax[1, 0].legend()

    # ax[0, 1].plot(t, la_N[:, 1], label="la_N0 - right")
    # ax[0, 1].grid()
    # ax[0, 1].legend()

    # ax[1, 1].plot(t, la_F[:, 2], "-r", label="la_F0 - right")
    # ax[1, 1].plot(t, la_F[:, 3], "--b", label="la_F1 - right")
    # ax[1, 1].grid()
    # ax[1, 1].legend()

    # plt.show()

    # fig, ax = plt.subplots(1, 2)

    # l1 = ax[0].plot(t, la_N[:, 0], "-r", label="la_N0 - left")
    # ax0 = ax[0].twinx()
    # l2 = ax0.plot(t, la_F[:, 0], "--g", label="la_F0 - left")
    # l3 = ax0.plot(t, la_F[:, 1], "-.b", label="la_F1 - left")
    # lns = l1 + l2 + l3
    # labs = [l.get_label() for l in lns]
    # ax[0].legend(lns, labs, loc=0)
    # ax[0].grid()

    # ax[1].plot(t, la_N[:, 1], "-r", label="la_N0 - right")
    # ax[1].plot(t, la_F[:, 2], "--g", label="la_F0 - right")
    # ax[1].plot(t, la_F[:, 3], "-.b", label="la_F1 - right")
    # ax[1].grid()
    # ax[1].legend()

    # plt.show()
