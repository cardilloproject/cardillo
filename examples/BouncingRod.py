import numpy as np
from math import pi

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math import e1, e2, e3

from cardillo.model import System
from cardillo.beams.spatial import (
    CircularCrossSection,
    Simo1986,
)
from cardillo.beams.spatial.Cable import QuadraticMaterial
from cardillo.beams import (
    DirectorAxisAngle,
    animate_rope,
)
from cardillo.forces import DistributedForce1DBeam

from cardillo.model.frame import Frame
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane

from cardillo.solver import (
    Moreau,
    NonsmoothGeneralizedAlpha,
    NonsmoothDecoupled,
)


if __name__ == "__main__":
    animate = True
    # animate = False

    # discretization properties
    nelements = 1
    # nelements = 3
    polynomial_degree_r = 2
    # basis_r = "Hermite"
    # basis_r = "B-spline"
    basis_r = "Lagrange"
    # polynomial_degree_r = 1
    # basis_r = "Lagrange"
    # polynomial_degree_r = 2
    # basis_r = "Lagrange"
    # polynomial_degree_psi = 1
    # basis_psi = "Lagrange"
    polynomial_degree_psi = 1
    # basis_psi = "B-spline"
    basis_psi = "Lagrange"

    # starting point and corresponding orientation
    # r_OP0 = np.zeros(3, dtype=float)
    z0 = 0.25
    r_OP0 = np.array([0, 0, z0], dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    # cross section and quadratic beam material
    L = np.pi
    line_density = 1.0e-1
    radius_rod = 1.0e-1
    E = 1.0e2
    G = 0.5e2
    cross_section = CircularCrossSection(line_density, radius_rod)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

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
        basis_r=basis_r,
        basis_psi=basis_psi,
    )

    # gravity
    g = 9.81
    __fg = -A_rho0 * g * e3
    fg = lambda t, xi: __fg
    gravity = DistributedForce1DBeam(fg, rod)

    mu = 0.1
    # r_N = 0.2
    r_N = 0.1
    # r_N = 5.0e-3
    e_N = 0

    frame_left = Frame()
    plane_left = Sphere2Plane(
        frame_left,
        rod,
        radius_rod,
        mu,
        prox_r_N=r_N,
        prox_r_F=r_N,
        e_N=e_N,
        frame_ID=(0,),
    )

    frame_right = Frame()
    plane_right = Sphere2Plane(
        frame_right,
        rod,
        radius_rod,
        mu,
        prox_r_N=r_N,
        prox_r_F=r_N,
        e_N=e_N,
        frame_ID=(1,),
    )

    model = System()
    model.add(rod)
    model.add(gravity)
    model.add(plane_left)
    model.add(plane_right)
    model.assemble()

    t0 = 0
    t1 = 1

    # dt = 5e-2
    # dt = 1e-2
    # dt = 5e-3
    dt = 1e-3
    # dt = 5e-4
    # dt = 1e-4

    # solver = NonsmoothGeneralizedAlpha(model, t1, dt, newton_max_iter=10)
    solver = NonsmoothDecoupled(model, t1, dt)
    # solver = Moreau(model, t1, dt, fix_point_max_iter=100)
    sol_other = solver.solve()

    t = sol_other.t
    q = sol_other.q
    scale = L
    animate_rope(t, q, [rod], scale)

    exit()
    # t = sol_other.t
    # q = sol_other.q
    # t_other = sol_other.t
    # q_other = sol_other.q
    # u_other = sol_other.u
    # P_N_other = sol_other.P_N
    # P_F_other = sol_other.P_F
    # if type(solver) in [
    #     NonsmoothThetaGGL,
    #     NonsmoothEulerBackwardsGGL,
    #     NonsmoothEulerBackwardsGGL_V2,
    #     NonsmoothHalfExplicitEuler,
    #     NonsmoothHalfExplicitEulerGGL,
    # ]:
    #     a_other = np.zeros_like(u_other)
    #     a_other[1:] = (u_other[1:] - u_other[:-1]) / dt
    #     la_N_other = np.zeros_like(P_N_other)
    #     la_F_other = np.zeros_like(P_F_other)
    #     La_N_other = np.zeros_like(P_N_other)
    #     La_F_other = np.zeros_like(P_F_other)
    #     # mu_g_other = sol_other.mu_g
    #     # mu_N_other = sol_other.mu_N
    #     mu_g_other = np.zeros(model.nla_g)
    #     mu_N_other = np.zeros_like(P_N_other)
    # else:
    #     a_other = sol_other.a
    #     la_N_other = sol_other.la_N
    #     # mu_N_other = sol_other.la_N
    #     la_F_other = sol_other.la_F
    #     La_N_other = sol_other.La_N
    #     La_F_other = sol_other.La_F

    solver_fp = Moreau(model, t1, dt)
    sol_fp = solver_fp.solve()
    t_fp = sol_fp.t
    q_fp = sol_fp.q
    u_fp = sol_fp.u
    a_fp = np.zeros_like(u_fp)
    a_fp[1:] = (u_fp[1:] - u_fp[:-1]) / dt
    P_N_fp = sol_fp.P_N
    P_F_fp = sol_fp.P_F

    fig, ax = plt.subplots(2, 1)
    ax[0].set_title("x(t)")
    ax[0].plot(t_fp, q_fp[:, 0], "-r", label="Moreau")
    ax[0].plot(t_other, q_other[:, 0], "--b", label="Other")
    ax[0].legend()

    ax[1].set_title("u_x(t)")
    ax[1].plot(t_fp, u_fp[:, 0], "-r", label="Moreau")
    ax[1].plot(t_other, u_other[:, 0], "--b", label="Other")
    ax[1].legend()

    # ax[2].set_title("a_x(t)")
    # ax[2].plot(t_fp, a_fp[:, 0], "-r", label="Moreau")
    # ax[2].plot(t_other, a_other[:, 0], "--b", label="Other")
    # ax[2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(2, 1)
    ax[0].set_title("y(t)")
    ax[0].plot([t_fp[0], t_fp[-1]], [r, r], "-k", label="ground")
    ax[0].plot(t_fp, q_fp[:, 1], "-r", label="y Moreau")
    ax[0].plot(t_other, q_other[:, 1], "--b", label="y Other")
    ax[0].legend()

    ax[1].set_title("u_y(t)")
    ax[1].plot(t_fp, u_fp[:, 1], "-r", label="Moreau")
    ax[1].plot(t_other, u_other[:, 1], "--b", label="Other")
    ax[1].legend()

    # ax[2].set_title("a_y(t)")
    # ax[2].plot(t_fp, a_fp[:, 1], "-r", label="Moreau")
    # ax[2].plot(t_other, a_other[:, 1], "--b", label="Other")
    # ax[2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(2, 1)
    ax[0].set_title("phi(t)")
    ax[0].plot(t_fp, q_fp[:, 3], "-r", label="Moreau")
    ax[0].plot(t_other, q_other[:, 3], "--b", label="Other")
    ax[0].legend()

    ax[1].set_title("u_phi(t)")
    ax[1].plot(t_fp, u_fp[:, -1], "-r", label="Moreau")
    ax[1].plot(t_other, u_other[:, -1], "--b", label="Other")
    ax[1].legend()

    # ax[2].set_title("a_phi(t)")
    # ax[2].plot(t_fp, a_fp[:, -1], "-r", label="Moreau")
    # ax[2].plot(t_other, a_other[:, -1], "--b", label="Other")
    # ax[2].legend()

    plt.tight_layout()

    fig, ax = plt.subplots(3, 1)

    ax[0].set_title("P_N(t)")
    ax[0].plot(t_fp, P_N_fp[:, 0], "-r", label="Moreau")
    ax[0].plot(t_other, dt * la_N_other[:, 0], "--b", label="dt * Other_la_N")
    ax[0].plot(t_other, La_N_other[:, 0], "--g", label="Other_La_N")
    ax[0].plot(t_other, P_N_other[:, 0], "--k", label="Other_P_N")
    # ax[0].plot(t_other, mu_N_other[:, 0], "--g", label="Other_mu_N")
    ax[0].legend()

    ax[1].set_title("P_Fx(t)")
    ax[1].plot(t_fp, P_F_fp[:, 0], "-r", label="Moreau")
    # ax[1].plot(t_other, la_F_other[:, 0], "--b", label="Other_la_F")
    # ax[1].plot(t_other, La_F_other[:, 0], "--g", label="Other_La_F")
    ax[1].plot(t_other, P_F_other[:, 0], "--k", label="Other_P_N")
    ax[1].legend()

    ax[2].set_title("P_Fy(t)")
    ax[2].plot(t_fp, P_F_fp[:, 1], "-r", label="Moreau")
    # ax[2].plot(t_other, la_F_other[:, 1], "--b", label="Other_la_F")
    # ax[2].plot(t_other, La_F_other[:, 1], "--g", label="Other_La_F")
    ax[2].plot(t_other, P_F_other[:, 1], "--k", label="Other_P_N")
    ax[2].legend()

    plt.tight_layout()

    plt.show()

    if animate:

        t = t_other
        q = q_other

        # animate configurations
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.set_xlim(-2 * y0, 2 * y0)
        ax.set_ylim(-2 * y0, 2 * y0)

        # prepare data for animation
        frames = len(t)
        target_frames = min(len(t), 200)
        frac = int(frames / target_frames)
        animation_time = 5
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t[::frac]
        q = q[::frac]

        # ax.plot([-2 * y0, 2 * y0], (y0-0.1)*np.array([1, 1]), '-k')

        # horizontal plane
        ax.plot([-2 * y0, 2 * y0], [0, 0], "-k")

        # # inclined planes
        # ax.plot([0, -y0 * np.cos(alpha)], [0, y0 * np.sin(alpha)], '-k')
        # ax.plot([0, y0 * np.cos(beta)], [0, - y0 * np.sin(beta)], '-k')

        def create(t, q):
            x_S, y_S, _ = RB.r_OP(t, q)

            A_IK = RB.A_IK(t, q)
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
            # d3 = A_IK[:, 2] * r

            (COM,) = ax.plot([x_S], [y_S], "ok")
            (bdry,) = ax.plot([], [], "-k")
            (d1_,) = ax.plot([x_S, x_S + d1[0]], [y_S, y_S + d1[1]], "-r")
            (d2_,) = ax.plot([x_S, x_S + d2[0]], [y_S, y_S + d2[1]], "-g")
            return COM, bdry, d1_, d2_

        COM, bdry, d1_, d2_ = create(0, q[0])

        def update(t, q, COM, bdry, d1_, d2_):

            x_S, y_S, _ = RB.r_OP(t, q)

            x_bdry, y_bdry, _ = RB.boundary(t, q)

            A_IK = RB.A_IK(t, q)
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
            # d3 = A_IK[:, 2] * r

            COM.set_data([x_S], [y_S])
            bdry.set_data(x_bdry, y_bdry)

            d1_.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
            d2_.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])

            return COM, bdry, d1_, d2_

        def animate(i):
            update(t[i], q[i], COM, bdry, d1_, d2_)

        anim = animation.FuncAnimation(
            fig, animate, frames=frames, interval=interval, blit=False
        )

        plt.show()
