from cardillo.model.classical_beams.spatial import Hooke_quadratic, Hooke
from cardillo.model.classical_beams.spatial import TimoshenkoDirectorIntegral
from cardillo.discretization.b_spline import fit_B_spline_curve
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import (
    Rigid_connection,
    Linear_guidance_x,
)
from cardillo.model import System
from cardillo.solver import Newton, Riks
from cardillo.model.force import Force
from cardillo.model.moment import K_Moment, K_Moment_scaled
from cardillo.math.algebra import e1, e2, e3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.spatial.transform import Rotation


def arc3D(t, R, plane="xz"):
    orientation = "xyz"
    perm = []
    for i in plane:
        perm.append(orientation.find(i))

    nt = len(t)
    P = np.zeros((nt, 3))
    dP = np.zeros((nt, 3))
    ddP = np.zeros((nt, 3))

    pi2 = 2 * np.pi
    rc_phi = R * np.cos(pi2 * t)
    rs_phi = R * np.sin(pi2 * t)

    P[:, perm] = np.array([-rc_phi, rs_phi]).T
    dP[:, perm] = np.array([pi2 * rs_phi, pi2 * rc_phi]).T
    ddP[:, perm] = np.array([pi2**2 * rc_phi, -(pi2**2) * rs_phi]).T

    return P, dP, ddP


Beam = TimoshenkoDirectorIntegral

if __name__ == "__main__":
    ################################
    # Romero2004, example 5.4
    # Smolenski1998, deployable ring
    ################################

    # rectangular cross-section, see https://de.wikipedia.org/wiki/Fl%C3%A4chentr%C3%A4gheitsmoment#Beispiele
    E = 2.1e7
    nu = 0.3
    G = E / (2 * (1 + nu))
    h = 1
    b = 1 / 3
    A = h * b
    I2 = b * h**3 / 12
    I3 = h * b**3 / 12
    print(f"E / G: {E / G} (should be 2.6 due to Goto1992)")
    print(f"h / b: {h / b} (should be 3 due to Goto1992)")

    # note this low torsional stiffness is required for this example!!!
    f = lambda n: np.tanh((2 * n - 1) * np.pi * h / (2 * b)) / (2 * n - 1) ** 2
    N = int(1.0e5)
    I1_ = (
        b**3
        * h
        / 3
        * (1 - 192 * b / (np.pi**5 * h) * np.array([f(n) for n in range(1, N)]).sum())
    )  # Goto1992 (19-d)
    I1 = 9.753e-3  # Romero2004
    print(f"I1: {I1}")
    print(f"I2: {I2}")
    print(f"I3: {I3}")
    print(f"I1_: {I1_}")

    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * I1, E * I2, E * I3])

    # material_model = Hooke(Ei, Fi)
    material_model = Hooke_quadratic(Ei, Fi)

    A_rho0 = 0.0
    B_rho0 = np.zeros(3)
    C_rho0 = np.zeros((3, 3))

    # beam fitting
    R = 20  # Goto1992 & Smolenski1998
    nxi = 500
    print(f"R / h: {R / h} (should be 20 due to Goto1992)")

    #################
    # external moment
    #################
    M = Fi[1] / R  # Smolenski1998 Fig.17 and Goto1992 Fig. 8 + (19-c)
    # M = 6.0e4               # Smolenski1998 Fig.17
    # M = 1.0e4               # Newton
    print(f"M: {M}")
    moment = lambda t: M * e2 * t

    ###########################
    # discretization properties
    ###########################
    # # Smolenski1998 used 48 elements -> nEl = 48 / 2 = 24
    # p = 3
    # nQP = p + 1
    # print(f'nQP: {nQP}')
    # nEl = 24 # p = 3 and nEl = 24 gets this example working!

    p = 3
    nQP = p + 1
    print(f"nQP: {nQP}")
    nEl = 10  # p = 3 and nEl = 10 gets this example working!

    #######################
    # fit first half circle
    #######################
    xi1 = np.linspace(0, 0.5, nxi)
    P1, dP1, ddP1 = arc3D(xi1, R, plane="xy")
    P1[:, 0] += R
    d1 = (dP1.T / np.linalg.norm(dP1, axis=-1)).T
    d2 = (ddP1.T / np.linalg.norm(ddP1, axis=-1)).T
    d3 = np.cross(d1, d2)
    qr0_1 = fit_B_spline_curve(P1, p, nEl)
    qd1_1 = fit_B_spline_curve(d1, p, nEl)
    qd2_1 = fit_B_spline_curve(d2, p, nEl)
    qd3_1 = fit_B_spline_curve(d3, p, nEl)
    Q1 = np.concatenate(
        (
            qr0_1.T.reshape(-1),
            qd1_1.T.reshape(-1),
            qd2_1.T.reshape(-1),
            qd3_1.T.reshape(-1),
        )
    )

    ########################
    # fit second half circle
    ########################
    xi1 = np.linspace(0.5, 1, nxi)
    P1, dP1, ddP1 = arc3D(xi1, R, plane="xy")
    P1[:, 0] += R
    d1 = (dP1.T / np.linalg.norm(dP1, axis=-1)).T
    d2 = (ddP1.T / np.linalg.norm(ddP1, axis=-1)).T
    d3 = np.cross(d1, d2)
    qr0_2 = fit_B_spline_curve(P1, p, nEl)
    qd1_2 = fit_B_spline_curve(d1, p, nEl)
    qd2_2 = fit_B_spline_curve(d2, p, nEl)
    qd3_2 = fit_B_spline_curve(d3, p, nEl)
    Q2 = np.concatenate(
        (
            qr0_2.T.reshape(-1),
            qd1_2.T.reshape(-1),
            qd2_2.T.reshape(-1),
            qd3_2.T.reshape(-1),
        )
    )

    # ###############
    # # debug fitting
    # ###############
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    # ax.set_zlabel('z [m]')
    # scale = 1.2 * R
    # ax.set_xlim3d(left=-scale, right=scale)
    # ax.set_ylim3d(bottom=-scale, top=scale)
    # ax.set_zlim3d(bottom=-scale, top=scale)

    # ax.plot(*qr0_1.T, '--ob')
    # ax.quiver(*qr0_1.T, *qd1_1.T, color='red', length=1, label='D1_1')
    # ax.quiver(*qr0_1.T, *qd2_1.T, color='green', length=1, label='D2_1')
    # ax.quiver(*qr0_1.T, *qd3_1.T, color='blue', length=1, label='D3_1')

    # ax.plot(*qr0_2.T, '--or')
    # ax.quiver(*qr0_2.T, *qd1_2.T, color='red', length=1, label='D1_1')
    # ax.quiver(*qr0_2.T, *qd2_2.T, color='green', length=1, label='D2_1')
    # ax.quiver(*qr0_2.T, *qd3_2.T, color='blue', length=1, label='D3_1')

    # plt.show()
    # exit()

    ########################################
    # junction at the origin and circle apex
    ########################################
    r_OB1 = np.zeros(3)  # origin
    r_OB2 = np.array([R, 0, 0])  # apex

    # build the model
    model = System()
    beam1 = Beam(material_model, A_rho0, B_rho0, C_rho0, p, p, nQP, nEl, Q=Q1)
    model.add(beam1)
    beam2 = Beam(material_model, A_rho0, B_rho0, C_rho0, p, p, nQP, nEl, Q=Q2)
    model.add(beam2)

    # clamp both beams at the origin
    frame1 = Frame(r_OP=r_OB1)
    model.add(frame1)
    model.add(Rigid_connection(frame1, beam1, r_OB1, frame_ID2=(0,)))
    model.add(Rigid_connection(frame1, beam2, r_OB1, frame_ID2=(1,)))

    # TODO: there is a bug here!
    model.add(Rigid_connection(beam1, beam2, r_OB2, frame_ID1=(1,), frame_ID2=(0,)))
    # model.add(Rigid_connection(beam2, beam1, r_OB2, frame_ID1=(0,), frame_ID2=(1,)))
    # model.add(Rigid_connection(beam2, beam1, r_OB1, frame_ID1=(0,), frame_ID2=(1,)))

    # linear guidance at the circle apex
    frame2 = Frame(r_OP=r_OB2)
    model.add(frame2)
    A_IB = np.eye(3)
    model.add(Linear_guidance_x(frame2, beam1, r_OB=r_OB2, A_IB=A_IB, frame_ID2=(1,)))

    # # model.add(Force(force, beam1, frame_ID=(1,)))
    model.add(K_Moment(moment, beam1, frame_ID=(1,)))
    # model.add(K_Moment_scaled(moment, beam1, frame_ID=(1,)))

    model.assemble()

    ###################
    # solver parameters
    ###################
    # n_load_steps = 20
    # max_iter = 20
    # tol = 1.0e-6
    # sol = Newton(model, n_load_steps=n_load_steps, max_iter=max_iter, tol=tol).solve()

    la_arc0 = 1.0e-3
    # la_arc_span = [0, 1.0e-1]
    la_arc_span = [-0.05, 1]
    iter_goal = 4
    tol = 1.0e-5
    sol = Riks(
        model,
        la_arc0=la_arc0,
        tol=tol,
        la_arc_span=la_arc_span,
        iter_goal=iter_goal,
        debug=0,
    ).solve()

    # visualization
    t = sol.t
    q = sol.q

    # ######################################
    # # plot initial and final configuration
    # ######################################
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scale = 1.2 * R
    # ax.set_xlim3d(left=-0.2 * scale, right=1.8 * scale)
    # ax.set_ylim3d(bottom=-scale, top=scale)
    # ax.set_zlim3d(bottom=-scale, top=scale)
    # beam1.plot_centerline(ax, q[0], color='black')
    # beam2.plot_centerline(ax, q[0], color='black')
    # beam1.plot_centerline(ax, q[-1], color='green')
    # beam2.plot_centerline(ax, q[-1], color='blue')
    # plt.show()
    # exit()

    # ########################
    # # compute tip deflection
    # ########################
    # r0 = sols[0].q[0][beam.qDOF].reshape(12, -1)[:3, -1]
    # dr = []
    # for i, qi in enumerate(sols[0].q):
    #     dr.append( beam.centerline(qi)[:, -1] - r0)
    #     # dr.append( qi[beam.qDOF].reshape(12, -1)[:3, -1] - r0)
    # dr = np.array(dr).T

    # ##############################################
    # # visualize initial and deformed configuration
    # ##############################################
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scale = 1.2 * R
    # ax.set_xlim3d(left=-0.2 * scale, right=1.8 * scale)
    # ax.set_ylim3d(bottom=-scale, top=scale)
    # ax.set_zlim3d(bottom=-scale, top=scale)
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    # ax.set_zlabel('z [m]')

    # # initial configuration
    # n_points = 100
    # ax.plot(*beam1.centerline(q[0], n=n_points), '-k')
    # ax.plot(*beam2.centerline(q[0], n=n_points), '-k')
    # ax.plot(*beam1.nodes(q[0]), '--ok')
    # ax.plot(*beam2.nodes(q[0]), '--ok')

    # # center_line1, = ax.plot([], [], '-g')
    # # center_line2, = ax.plot([], [], '-b')
    # # nodes1, = ax.plot([], [], '--og')
    # # nodes2, = ax.plot([], [], '--ob')
    # ax.plot(*beam1.centerline(q[-5], n=n_points), '-g')
    # ax.plot(*beam2.centerline(q[-5], n=n_points), '-b')
    # ax.plot(*beam1.nodes(q[-5]), '--og')
    # ax.plot(*beam2.nodes(q[-5]), '--ob')

    # plt.show()

    ###########
    # animation
    ###########
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    scale = 1.2 * R
    ax1.set_xlim3d(left=-0.2 * scale, right=1.8 * scale)
    ax1.set_ylim3d(bottom=-scale, top=scale)
    ax1.set_zlim3d(bottom=-scale, top=scale)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_zlabel("z [m]")

    slowmotion = 3
    fps = 50
    animation_time = slowmotion * 1
    target_frames = int(fps * animation_time)
    frac = max(1, int(len(t) / target_frames))
    if frac == 1:
        target_frames = len(t)
    interval = 1000 / fps

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    ###############################
    # compute tip deflection
    # of the middle point in beam 1
    ###############################
    M = M * t
    r0 = beam1.centerline(q[0], n=3)[:, 1]
    dr = np.zeros((len(t), 3))
    r = np.zeros((len(t), 3))
    for i, qi in enumerate(q):
        r[i] = beam1.centerline(qi, n=3)[:, 1]
        dr[i] = r[i] - r0

    #######################################
    # compute rotation angle ad circle apex
    #######################################
    alpha = np.zeros_like(t)
    beta = np.zeros_like(t)
    gamma = np.zeros_like(t)
    for i, qi in enumerate(q):
        _, d1, d2, d3 = beam1.frames(qi, n=2)
        Ri = np.vstack((d1[:, -1], d2[:, -1], d3[:, -1])).T
        alpha[i], beta[i], gamma[i] = Rotation.from_matrix(Ri).as_euler(
            "zyx", degrees=True
        )
    # correct angle
    gamma = -(gamma - 180)

    # initial configuration
    n_points = 100
    ax1.plot(*beam1.centerline(q[0], n=n_points), "-k")
    ax1.plot(*beam2.centerline(q[0], n=n_points), "-k")
    ax1.plot(*beam1.nodes(q[0]), "--ok")
    ax1.plot(*beam2.nodes(q[0]), "--ok")

    (center_line1,) = ax1.plot([], [], "-g")
    (center_line2,) = ax1.plot([], [], "-b")
    (nodes1,) = ax1.plot([], [], "--og")
    (nodes2,) = ax1.plot([], [], "--ob")

    # tip deflection
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(dr[:, 0], M, "-k", label="u_x")
    ax2.plot(dr[:, 1], M, "-.k", label="u_y")
    ax2.plot(dr[:, 2], M, "--k", label="u_z")
    ax2.grid()
    ax2.legend()

    # ax3 = ax2.twinx()
    ax3 = fig.add_subplot(1, 3, 3)
    # ax3.tick_params(axis='y', labelcolor='blue')
    # ax3.plot(alpha, M, '-b', label='alpha')
    # ax3.plot(beta, M, '-.b', label='beta')
    ax3.plot(gamma, M, "--b", label="gamma")
    (nodes_gamma,) = ax3.plot([], [], "or")
    ax3.grid()
    ax3.legend()

    # animate tip deflection
    (nodes_xyz,) = ax1.plot([], [], [], "or")
    (nodes_u_x,) = ax2.plot([], [], "or")
    (nodes_u_y,) = ax2.plot([], [], "or")
    (nodes_u_z,) = ax2.plot([], [], "or")

    def animate(i):
        qi = q[i].copy()

        x, y, z = beam1.centerline(qi, n=n_points)
        center_line1.set_data(x, y)
        center_line1.set_3d_properties(z)

        x, y, z = beam1.nodes(qi)
        nodes1.set_data(x, y)
        nodes1.set_3d_properties(z)

        x, y, z = beam2.centerline(qi, n=n_points)
        center_line2.set_data(x, y)
        center_line2.set_3d_properties(z)

        x, y, z = beam2.nodes(qi)
        nodes2.set_data(x, y)
        nodes2.set_3d_properties(z)

        # node where deflection is measured
        nodes_xyz.set_data(r[i, 0], r[i, 1])
        nodes_xyz.set_3d_properties(r[i, 2])

        nodes_u_x.set_data(dr[i, 0], M[i])
        nodes_u_y.set_data(dr[i, 1], M[i])
        nodes_u_z.set_data(dr[i, 2], M[i])

        # rotation angle
        nodes_gamma.set_data(gamma[i], M[i])

        return (
            center_line1,
            nodes1,
            center_line2,
            nodes2,
            nodes_xyz,
            nodes_u_x,
            nodes_u_y,
            nodes_u_z,
            nodes_gamma,
        )

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )
    plt.show()
