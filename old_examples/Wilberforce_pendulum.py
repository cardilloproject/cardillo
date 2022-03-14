from cardillo.model.classical_beams.spatial import Hooke_quadratic
from cardillo.model.classical_beams.spatial import TimoshenkoDirectorIntegral
from cardillo.discretization.B_spline import fit_B_Spline
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import Rigid_connection
from cardillo.model import Model
from cardillo.solver import Newton
from cardillo.solver import Generalized_alpha_4_singular_index3
from cardillo.model.force import Force
from cardillo.model.line_force.line_force import Line_force
from cardillo.math.algebra import e3, ax2skew
from cardillo.model.rigid_body import Rigid_body_euler
from cardillo.solver.solution import load_solution

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

import os

path = os.path.dirname(os.path.abspath(__file__))

c = 9.0e-3  # eccentricity


def cylinder(R, h):
    rho = 7850  # [kg / m^3]; steel
    R = 18e-3  # radius
    h = 50e-3  # height
    V = np.pi * R**2 * h  # volume
    m = V * rho  # mass

    I11 = I22 = (1 / 4) * m * R**2 + (1 / 12) * m * h**2
    I33 = (1 / 2) * m * R**2
    Theta_S = np.diag([I11, I22, I33])  # intertia tensor
    return m, Theta_S


def discs(a, R):
    rho = 7850  # [kg / m^3]; steel
    b = 5.925e-3  # height
    r = a / 2
    r2 = r**2
    V = np.pi * b * r2  # volume
    m = V * rho  # mass

    I11 = 0.5 * m * r2
    I22 = I33 = (1 / 4) * m * r2 + (1 / 12) * m * b**2

    Theta_S13 = np.diag([I11, I22, I33])  # intertia tensor
    Theta_S24 = np.diag(
        [I22, I11, I33]
    )  # intertia tensor ( rotated with pi/2 around d3)

    r13 = np.array([R + c, 0, 0])
    r24 = np.array([0, R + c, 0])
    r13_tilde = ax2skew(r13)
    r24_tilde = ax2skew(r24)
    Theta_Steiner = m * (
        2 * r13_tilde.T @ r13_tilde + 2 * r24_tilde.T @ r24_tilde
    )  # Steiner part

    Theta_S = 2 * Theta_S13 + 2 * Theta_S24 + Theta_Steiner

    return 4 * m, Theta_S


def screws(R):
    rho = 7850  # [kg / m^3]; steel
    d = 5e-3  # diameter of the skrew
    r = d / 2
    r2 = r**2
    l = 33e-3  # length of the skrew
    V = np.pi * r2 * l  # volume
    m = V * rho  # mass

    I11 = 0.5 * m * r2
    I22 = I33 = (1 / 4) * m * r2 + (1 / 12) * m * l**2

    Theta_S13 = np.diag([I11, I22, I33])  # intertia tensor
    Theta_S24 = np.diag(
        [I22, I11, I33]
    )  # intertia tensor ( rotated with pi/2 around d3)

    r_13 = np.array([R + l / 2, 0, 0])
    r_24 = np.array([0, R + l / 2, 0])
    r_13_tilde = ax2skew(r_13)
    r_24_tilde = ax2skew(r_24)

    Theta_Steiner = m * (
        2 * r_13_tilde.T @ r_13_tilde + 2 * r_24_tilde.T @ r_24_tilde
    )  # Steiner part
    Theta_S = 2 * Theta_S13 + 2 * Theta_S24 + Theta_Steiner

    return 4 * m, Theta_S


def Wilberforce_bob(R, h, debug=True):
    ##########
    # cylinder
    ##########
    m1, Theta1_S = cylinder(R, h)
    if debug:
        print(f"mass cylinder = {m1};")
        # print(f'inertia cylinder =\n{Theta1}')

    ###########
    # 4 screw's
    ###########
    m2, Theta2_S = screws(R)
    if debug:
        print(f"mass 4 screw's = {m2}")
        # print(f"inertia 4 screw's  =\n{Theta2}")

    # compute total mass of cylinder and screw's
    # this can be measured
    if debug:
        print(
            f"mass cylinder & 4 screw's = {m1 + m2}; measured masses = {0.412}; error = {np.abs(m1 + m2 - 0.412) / 0.412}"
        )
        # print(f"inertia cylinder & 4 screw's  =\n{Theta1 + Theta2}")

    ##########
    # 4 disc's
    ##########
    a = 19e-3  # outer radius
    d = 5e-3  # inner radius
    m3_outer, Theta3_outer_S = discs(a, R)
    m3_inner, Theta3_inner_S = discs(d, R)
    m3 = m3_outer - m3_inner
    Theta3_S = Theta3_outer_S - Theta3_inner_S
    if debug:
        print(
            f"mass 4 disc's = {m3}; measured mass = {0.049}; error = {np.abs(m3 - 0.049) / 0.049}"
        )
        # print(f"inertia 4 disc's =\n{Theta3}")

    m = m1 + m2 + m3
    Theta = Theta1_S + Theta2_S + Theta3_S

    if debug:
        print(f"total mass = {m}")
        print(f"total inertia =\n{Theta}")

    return m, Theta


def helix3D(t, r, c, plane="xyz"):
    """Helix function in xyz plane, see https://mathworld.wolfram.com/Helix.html"""
    assert len(plane) == 3
    orientation = "xyz"
    perm = []
    for i in plane:
        perm.append(orientation.find(i))

    nt = len(t)
    P = np.zeros((nt, 3))
    dP = np.zeros((nt, 3))
    ddP = np.zeros((nt, 3))

    pi2 = 2 * np.pi
    rc_phi = r * np.cos(pi2 * t)
    rs_phi = r * np.sin(pi2 * t)

    P[:, perm] = np.array([rc_phi, rs_phi, c * t]).T
    dP[:, perm] = np.array([-pi2 * rs_phi, pi2 * rc_phi, c * np.ones_like(t)]).T
    ddP[:, perm] = np.array(
        [-(pi2**2) * rc_phi, -(pi2**2) * rs_phi, np.zeros_like(t)]
    ).T

    return P, dP, ddP


# statics = True
statics = False

save = True
# save = False

profile = True
# profile = False

import os

path = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ################################################################################################
    # beam parameters, taken from https://faraday.physics.utoronto.ca/PHY182S/WilberforceRefBerg.pdf
    ################################################################################################
    # see https://de.wikipedia.org/wiki/Fl%C3%A4chentr%C3%A4gheitsmoment#Beispiele
    d = 1e-3  # 1mm cross sectional diameter
    r = wire_radius = d / 2  # cross sectional radius
    A = np.pi * r**2
    I2 = I3 = (np.pi / 4) * r**4
    Ip = I2 + I3

    # Federstahl nach EN 10270-1
    rho = 7850  # [kg / m^3]; steel
    E = 206e9  # Pa
    G = 81.5e9  # Pa

    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([E * Ip, E * I3, E * I2])
    material_model = Hooke_quadratic(Ei, Fi)

    A_rho0 = rho * A
    B_rho0 = np.zeros(3)  # symmetric cross section!
    C_rho0 = rho * np.diag(np.array([0, I3, I2]))

    ##########
    # gravity
    #########
    g = 9.81

    ###########################
    # discretization properties
    ###########################
    p = 3
    nQP = int(np.ceil((p**2 + 1) / 2)) + 1  # dynamics
    print(f"nQP: {nQP}")
    # nEl = 4 # 1 turn
    # nEl = 16 # 2 turns
    # nEl = 32 # 5 turns
    nEl = 64  # 10 turns
    # nEl = 128 # 20 turns

    #############################
    # fit reference configuration
    #############################
    coil_diameter = 32.0e-3  # 32mm
    coil_radius = coil_diameter / 2
    pitch_unloaded = 1.0e-3  # 1mm
    # turns = 0.5
    # turns = 2
    # turns = 5
    turns = 10
    # turns = 20
    nxi = 10000

    # evaluate helixa at discrete points
    xi = np.linspace(0, turns, nxi)
    P, dP, ddP = helix3D(xi, coil_radius, pitch_unloaded)

    # compute directors using Serret-Frenet frame
    d1 = (dP.T / np.linalg.norm(dP, axis=-1)).T
    d2 = (ddP.T / np.linalg.norm(ddP, axis=-1)).T
    d3 = np.cross(d1, d2)

    # fit reference configuration
    qr0 = fit_B_Spline(P, p, nEl)
    qd1 = fit_B_Spline(d1, p, nEl)
    qd2 = fit_B_Spline(d2, p, nEl)
    qd3 = fit_B_Spline(d3, p, nEl)
    Q = np.concatenate(
        (qr0.T.reshape(-1), qd1.T.reshape(-1), qd2.T.reshape(-1), qd3.T.reshape(-1))
    )

    ##############################
    # junction at upper spring end
    ##############################
    r_OB1 = P[-1]
    A_IK1 = np.vstack((d1[-1], d2[-1], d3[-1])).T
    frame = Frame(r_OP=r_OB1, A_IK=A_IK1)

    ############
    # rigid body
    ############
    R = 18e-3  # radius of the main cylinder
    h = 50e-3  # height of the main cylinder
    m, Theta = Wilberforce_bob(R, h, debug=True)
    q0 = np.zeros(6)
    q0[2] = -h / 2 - wire_radius  # center of mass is shifted!
    u0 = np.zeros(6)
    bob = Rigid_body_euler(m, Theta, q0=q0, u0=u0)

    ###################
    # solver parameters
    ###################
    n_load_steps = 10
    max_iter = 30
    tol = 1.0e-6

    # t1 = 10
    t1 = 5
    dt = 1e-3  # full beam dynamics generalized alpha

    beam = TimoshenkoDirectorIntegral(
        material_model, A_rho0, B_rho0, C_rho0, p, p, nQP, nEl, q0=Q, Q=Q
    )

    model = Model()

    model.add(beam)
    model.add(frame)
    model.add(Rigid_connection(frame, beam, r_OB1, frame_ID2=(1,)))

    model.add(bob)
    r_OB = np.zeros(3)
    model.add(Rigid_connection(bob, beam, r_OB, frame_ID2=(0,)))

    if statics:
        model.add(Line_force(lambda xi, t: -t * g * A_rho0 * e3, beam))
        model.add(Force(lambda t: -t * g * m * e3, bob))
    else:
        model.add(Line_force(lambda xi, t: -g * A_rho0 * e3, beam))
        model.add(Force(lambda t: -g * m * e3, bob))

    model.assemble()

    if statics:
        solver = Newton(model, n_load_steps=n_load_steps, max_iter=max_iter, tol=tol)
    else:
        # build algebraic degrees of freedom indices for multiple beams
        tmp = int(beam.nu / 4)
        uDOF_algebraic = beam.uDOF[tmp : 2 * tmp]  # include whole beam dynamics
        # uDOF_algebraic = beam.uDOF[tmp:4*tmp] # exclude centerline beam dynamics
        # uDOF_algebraic = beam.uDOF # beam as static force element
        solver = Generalized_alpha_4_singular_index3(
            model,
            t1,
            dt,
            rho_inf=0.5,
            uDOF_algebraic=uDOF_algebraic,
            newton_tol=1.0e-6,
            numerical_jacobian=False,
        )

    export_path = f"Wilberforce_pendulum_p{p}_nEL{nEl}_turns{turns}_t1{t1}_dt{dt}_c{c}"

    if save:
        sol = solver.solve()
        sol.save(export_path)
    else:
        sol = load_solution(export_path)

    ##################
    # export less data
    ##################
    frac = 20
    t = sol.t[::frac]
    q = sol.q[::frac]

    ##################################
    # export bob deflection and angles
    ##################################
    r = q[:, bob.qDOF[:3]]
    angles = q[:, bob.qDOF[3:]] * 180 / np.pi
    data = np.hstack((r, angles))

    columns = ["x", "y", "z", "alpha", "beta", "gamma"]
    frame = pd.DataFrame(data=data, index=t, columns=columns)
    frame = frame.rename_axis(index="t")
    frame.to_csv(os.path.join(path, f"../bob_position_angles.csv"))

    ########################
    # compute tip deflection
    ########################
    r0 = sol.q[0][beam.qDOF].reshape(12, -1)[:3, -1]
    dr = []
    for i, qi in enumerate(sol.q):
        dr.append(beam.centerline(qi)[:, -1] - r0)
    dr = np.array(dr).T

    ###################################
    # export data for Blender animation
    ###################################
    nt = len(t)
    n_points = 1000
    r = np.zeros((nt, 3, n_points))
    d1 = np.zeros((nt, 3, n_points))
    d2 = np.zeros((nt, 3, n_points))
    d3 = np.zeros((nt, 3, n_points))
    bob_r = np.zeros((nt, 3))
    bob_R = np.zeros((nt, 3, 3))
    for i, (ti, qi) in enumerate(zip(t, q)):
        r[i], d1[i], d2[i], d3[i] = beam.frames(qi, n_points)

        bob_r[i] = bob.r_OP(ti, qi[bob.qDOF])
        bob_R[i] = bob.A_IK(ti, qi[bob.qDOF])

    data = {
        "t": t,
        "dt": dt,
        "r": r.transpose(0, 2, 1),
        "R": np.array((d1, d2, d3)).transpose(1, 3, 0, 2),
        "radius": wire_radius,
        "bob_r": bob_r,
        "bob_R": bob_R,
    }
    export_path = os.path.join(path, "../Wilberforce_pendulum.npy")
    np.save(export_path, data, allow_pickle=True)

    #########
    # animate
    #########
    # fig = plt.figure()
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    ax3 = ax2.twinx()

    # visualize bob's displacements
    q_bob = q[:, bob.qDOF]
    ax2.plot(t, q_bob[:, 0], "-b", label="x")
    ax2.plot(t, q_bob[:, 1], "--b", label="y")
    ax2.plot(t, q_bob[:, 2], "-.b", label="z")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.grid(True)
    ax2.legend()

    # visualize rotation angles
    ax3.plot(t, q_bob[:, 3] * 180 / np.pi, "-r", label="phi_z")
    ax3.plot(t, q_bob[:, 4] * 180 / np.pi, "--r", label="phi_x")
    ax3.plot(t, q_bob[:, 5] * 180 / np.pi, "-.r", label="phi_y")
    ax3.tick_params(axis="y", labelcolor="red")
    ax3.grid(True)
    ax3.legend()

    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_zlabel("z [m]")
    scale = 2 * coil_radius
    length_directors = 1.0e-2
    ax1.set_xlim3d(left=-scale, right=scale)
    ax1.set_ylim3d(bottom=-scale, top=scale)
    ax1.set_zlim3d(bottom=-2 * scale, top=0)

    # prepare data for animation
    slowmotion = 10
    fps = 10
    animation_time = slowmotion * t1
    target_frames = int(fps * animation_time)
    frac = max(1, int(len(t) / target_frames))
    if frac == 1:
        target_frames = len(t)
    interval = 1000 / fps

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    (center_line,) = ax1.plot([], [], "-k")
    (nodes,) = ax1.plot([], [], "--ob")

    (bob_com,) = ax1.plot([], [], [], "ok")
    (d1_,) = ax1.plot([], [], [], "-r")
    (d2_,) = ax1.plot([], [], [], "-g")
    (d3_,) = ax1.plot([], [], [], "-b")

    (y_,) = ax2.plot([], [], "ob")
    (phi_z_,) = ax3.plot([], [], "or")

    def animate(i):
        # animate beam centerline
        x, y, z = beam.centerline(q[i], n=400)
        center_line.set_data(x, y)
        center_line.set_3d_properties(z)

        # animate beam nodes
        x, y, z = q[i][beam.qDOF].reshape(12, -1)[:3]
        nodes.set_data(x, y)
        nodes.set_3d_properties(z)

        # animate rigid body
        x_S, y_S, z_S = bob.r_OP(t[i], q[i][bob.qDOF])
        bob_com.set_data(np.array([x_S]), np.array([y_S]))
        bob_com.set_3d_properties(np.array([z_S]))

        d1, d2, d3 = bob.A_IK(t[i], q[i][bob.qDOF]).T * length_directors

        d1_.set_data(np.array([x_S, x_S + d1[0]]), np.array([y_S, y_S + d1[1]]))
        d1_.set_3d_properties(np.array([z_S, z_S + d1[2]]))

        d2_.set_data(np.array([x_S, x_S + d2[0]]), np.array([y_S, y_S + d2[1]]))
        d2_.set_3d_properties(np.array([z_S, z_S + d2[2]]))

        d3_.set_data(np.array([x_S, x_S + d3[0]]), np.array([y_S, y_S + d3[1]]))
        d3_.set_3d_properties(np.array([z_S, z_S + d3[2]]))

        # animate y and phi_z
        y_.set_data(np.array([t[i]]), np.array([z_S]))
        phi_z_.set_data(np.array([t[i]]), np.array([q[i, bob.qDOF[3]] * 180 / np.pi]))

        return center_line, nodes, bob_com, d1_, d2_, d3_, y_, phi_z_

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )
    plt.show()
