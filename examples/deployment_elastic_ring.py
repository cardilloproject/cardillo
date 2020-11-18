from cardillo.model.classical_beams.spatial import Hooke_quadratic, Hooke
from cardillo.model.classical_beams.spatial import Timoshenko_director_integral
from cardillo.discretization.B_spline import fit_B_Spline
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import Rigid_connection
from cardillo.model import Model
from cardillo.solver import Newton
from cardillo.model.force import Force
from cardillo.model.moment import K_Moment
from cardillo.math.algebra import e1, e2, e3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def arc3D(t, R, plane='xz'):
    orientation = 'xyz'
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
    ddP[:, perm] = np.array([pi2**2 * rc_phi, -pi2**2 * rs_phi]).T

    return P, dP, ddP

Beam = Timoshenko_director_integral

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

    # TODO: what parameter to use for I1?
    # I1 = b**3 * h / 3 * (1 - 192 * h / (np.pi**5 * h)
    # I1 = 
    # # I1 = I2 + I3
    I1 = 9.753e-3
    print(f'I1: {I1}')
    print(f'I2: {I2}')
    print(f'I3: {I3}')

    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * I1, E * I2, E * I3])

    # material_model = Hooke(Ei, Fi)
    material_model = Hooke_quadratic(Ei, Fi)
    
    A_rho0 = 0.0
    B_rho0 = np.zeros(3)
    C_rho0 = np.zeros((3, 3))

    # beam fitting
    Ro = 20
    Ri = 12
    R = (Ro + Ri) / 2
    nxi = 500

    #################
    # external forces
    #################
    P = 1.0e2
    print(f'P: {P}')
    force = lambda t: -P * e1 * t

    #################
    # external moment
    #################
    M = 1.0e4
    print(f'M: {M}')
    moment = lambda t: M * e2 * t

    ###########################
    # discretization properties
    ###########################
    p = 3
    nQP = p + 1
    print(f'nQP: {nQP}')
    nEl = 16

    #######################
    # fit first half circle
    #######################
    xi1 = np.linspace(0, 0.5, nxi)
    P1, dP1, ddP1 = arc3D(xi1, R, plane='xy')
    P1[:, 0] += R
    d1 = (dP1.T / np.linalg.norm(dP1, axis=-1)).T
    d2 = (ddP1.T / np.linalg.norm(ddP1, axis=-1)).T
    d3 = np.cross(d1, d2)
    qr0_1 = fit_B_Spline(P1, p, nEl)
    qd1_1 = fit_B_Spline(d1, p, nEl)
    qd2_1 = fit_B_Spline(d2, p, nEl)
    qd3_1 = fit_B_Spline(d3, p, nEl)
    Q1 = np.concatenate((qr0_1.T.reshape(-1), qd1_1.T.reshape(-1), qd2_1.T.reshape(-1), qd3_1.T.reshape(-1)))

    ########################
    # fit second half circle
    ########################
    xi1 = np.linspace(0.5, 1, nxi)
    P1, dP1, ddP1 = arc3D(xi1, R, plane='xy')
    P1[:, 0] += R
    d1 = (dP1.T / np.linalg.norm(dP1, axis=-1)).T
    d2 = (ddP1.T / np.linalg.norm(ddP1, axis=-1)).T
    d3 = np.cross(d1, d2)
    qr0_2 = fit_B_Spline(P1, p, nEl)
    qd1_2 = fit_B_Spline(d1, p, nEl)
    qd2_2 = fit_B_Spline(d2, p, nEl)
    qd3_2 = fit_B_Spline(d3, p, nEl)
    Q2 = np.concatenate((qr0_2.T.reshape(-1), qd1_2.T.reshape(-1), qd2_2.T.reshape(-1), qd3_2.T.reshape(-1)))

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
    r_OB1= np.zeros(3) # origin
    r_OB2= np.array([R, 0, 0]) # apex
    
    ###################
    # solver parameters
    ###################
    # n_load_steps = 60 # Bathe1979
    n_load_steps = 10
    max_iter = 10
    tol = 1.0e-6

    model = Model()
    beam1 = Beam(material_model, A_rho0, B_rho0, C_rho0, p, nQP, nEl, Q=Q1)
    model.add(beam1)
    beam2 = Beam(material_model, A_rho0, B_rho0, C_rho0, p, nQP, nEl, Q=Q2)
    model.add(beam2)

    frame = Frame(r_OP=r_OB1)
    model.add(frame)
    model.add(Rigid_connection(frame, beam1, r_OB1, frame_ID2=(0,)))
    model.add(Rigid_connection(beam1, beam2, r_OB2, frame_ID1=(1,), frame_ID2=(0,)))
    model.add(Rigid_connection(beam2, beam1, r_OB2, frame_ID1=(1,), frame_ID2=(0,)))

    # model.add(Force(force, beam1, frame_ID=(1,)))
    model.add(K_Moment(moment, beam1, frame_ID=(1,)))

    model.assemble()

    sol = Newton(model, n_load_steps=n_load_steps, max_iter=max_iter, tol=tol).solve()
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

    # #####################################
    # # TODO: export solution for each beam
    # #####################################
    # fig = plt.figure(figsize=(10, 6))
    # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # ax2 = fig.add_subplot(1, 2, 2)

    # # Pt = P * sols[0].t
    # # ax2.plot(-dr[0] / R, Pt, '-k', label='-u_x/R')
    # # ax2.plot(-dr[1] / R, Pt, '--k', label='-u_y/R')
    # # ax2.plot(dr[2] / R, Pt, '-.k', label='u_z/R')
    # kt = P * R**2 / (E * I2) * sols[0].t
    # ax2.plot(kt, -dr[0] / R, '-k', label='-u_x/R')
    # ax2.plot(kt, -dr[1] / R, '--k', label='-u_y/R')
    # ax2.plot(kt, dr[2] / R, '-.k', label='u_z/R')
    # ax2.grid(True)
    # ax2.legend()
    
    # ax1.set_xlabel('x [m]')
    # ax1.set_ylabel('y [m]')
    # ax1.set_zlabel('z [m]')
    # scale = 1.2 * R
    # ax1.set_xlim3d(left=-scale, right=scale)
    # ax1.set_ylim3d(bottom=-scale, top=scale)
    # ax1.set_zlim3d(bottom=-scale, top=scale)

    ###########
    # animation
    ###########
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scale = 1.2 * R
    ax.set_xlim3d(left=-0.2 * scale, right=1.8 * scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=-scale, top=scale)

    # prepare data for animation
    frames = q.shape[0]
    target_frames = min(frames, 200)
    frac = int(frames / target_frames)
    animation_time = 1
    interval = animation_time * 1000 / target_frames

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    # initial configuration
    n_points = 100
    ax.plot(*beam1.centerline(q[0], n=n_points), '-k')
    ax.plot(*beam2.centerline(q[0], n=n_points), '-k')
    ax.plot(*beam1.nodes(q[0]), '--ok')
    ax.plot(*beam2.nodes(q[0]), '--ok')

    center_line1, = ax.plot([], [], '-g')
    center_line2, = ax.plot([], [], '-b')
    nodes1, = ax.plot([], [], '--og')
    nodes2, = ax.plot([], [], '--ob')

    def animate(i):
        x, y, z = beam1.centerline(q[i], n=n_points)
        center_line1.set_data(x, y)
        center_line1.set_3d_properties(z)

        x, y, z = beam1.nodes(q[i])
        nodes1.set_data(x, y)
        nodes1.set_3d_properties(z)

        x, y, z = beam2.centerline(q[i], n=n_points)
        center_line2.set_data(x, y)
        center_line2.set_3d_properties(z)

        x, y, z = beam2.nodes(q[i])
        nodes2.set_data(x, y)
        nodes2.set_3d_properties(z)

        return center_line1, nodes1, center_line2, nodes2

    # animate(1)

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
    plt.show()