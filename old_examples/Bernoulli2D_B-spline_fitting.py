from cardillo.model.classical_beams.planar import Hooke, EulerBernoulli
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import (
    Spherical_joint2D,
    Rigid_connection2D,
    Spherical_joint,
    Rigid_connection,
)
from cardillo.model import System
from cardillo.solver import (
    Euler_backward,
    Moreau,
    Moreau_sym,
    Generalized_alpha_1,
    Scipy_ivp,
    Newton,
)
from cardillo.model.line_force.line_force import Line_force
from cardillo.model.force import Force
from cardillo.discretization import uniform_knot_vector
from cardillo.discretization.b_spline import fit_B_spline_curve
from cardillo.model.rigid_body import Rigid_body_quaternion
from cardillo.model.force import Force
from cardillo.utility.post_processing_vtk import post_processing

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

statics = True
# statics = False


def B_spline_fitting():
    # physical properties of the rope
    rho = 7850
    L = 50
    r = 5.0e-3
    A = np.pi * r**2
    I = A / 4 * r**2
    E = 210e9 * 1.0e-3
    EA = E * A
    EI = E * I
    material_model = Hooke(EA, EI)
    A_rho0 = A * rho * 10

    # discretization properties
    p = 2
    assert p >= 2
    nQP = int(np.ceil((p + 1) ** 2 / 2))
    print(f"nQP: {nQP}")
    nEl = 6

    # fit reference configuration
    nxi = 100
    R = 1.25
    phi_max = np.pi / 2

    phi1 = np.linspace(0, phi_max, num=nxi)
    points1 = R * np.array([-np.cos(phi1), np.sin(phi1)]).T
    # ctrlpts1, knot_vector1 = approximate_curve(points1.tolist(), p, nEl=nEl)
    ctrlpts1 = fit_B_spline_curve(points1, p, nEl)

    phi2 = np.linspace(phi_max, 2 * phi_max, num=nxi)
    points2 = R * np.array([-np.cos(phi2), np.sin(phi2)]).T
    # ctrlpts2, knot_vector2 = approximate_curve(points2.tolist(), p, nEl=nEl)
    ctrlpts2 = fit_B_spline_curve(points2, p, nEl)

    # plt.plot(*points1.T, '-k', label='target curve')
    # plt.plot(*ctrlpts1.T, '--ob', label='B-spline control polygon')
    # plt.plot(*points2.T, '-g', label='target curve')
    # plt.plot(*ctrlpts2.T, '--or', label='B-spline control polygon')
    # plt.axis('equal')
    # plt.legend()
    # plt.show()

    # exit()

    Q1 = ctrlpts1.T.reshape(-1)
    beam1 = EulerBernoulli(
        A_rho0, material_model, p, nEl, nQP, Q=Q1, q0=Q1, u0=np.zeros_like(Q1)
    )

    Q2 = ctrlpts2.T.reshape(-1)
    beam2 = EulerBernoulli(
        A_rho0, material_model, p, nEl, nQP, Q=Q2, q0=Q2, u0=np.zeros_like(Q2)
    )

    # joints
    r_OP0 = np.array([*ctrlpts1[0], 0])
    r_OP01 = np.array([*ctrlpts1[-1], 0])
    r_OP1 = np.array([*ctrlpts2[-1], 0])
    frame0 = Frame(r_OP=r_OP0)
    frame1 = Frame(r_OP=r_OP1)
    # joint0 = Rigid_connection2D(frame0, beam1, r_OP0, frame_ID2=(0,))
    joint0 = Spherical_joint2D(frame0, beam1, r_OP0, frame_ID2=(0,))
    joint01 = Rigid_connection2D(beam1, beam2, r_OP01, frame_ID1=(1,), frame_ID2=(0,))
    joint1 = Spherical_joint2D(beam2, frame1, r_OP1, frame_ID1=(1,))

    if statics:
        F = 8.0e0
        # F = 2.0e0
        f_point = Force(lambda t: t * F * np.array([0, -1, 0]), beam1, frame_ID=(1,))
    else:
        F = 8.0e1
        f_point = Force(lambda t: t * F * np.array([0, -1, 0]), beam1, frame_ID=(1,))

    # assemble the model
    model = System()
    model.add(beam1)
    model.add(beam2)

    model.add(frame0)
    model.add(joint0)

    model.add(joint01)

    model.add(frame1)
    model.add(joint1)

    # model.add(f_g_beam)
    model.add(f_point)
    model.assemble()

    # plt.plot(*points.T, '-k', label='target curve')
    # plt.plot(*ctrlpts.T, '--ob', label='B-spline control polygon')
    # plt.plot(*(beam.centerline(q0, n=100).T[:2]), '--r', label='B-spline cruve')
    # plt.axis('equal')
    # plt.legend()
    # plt.show()

    if statics:
        solver = Newton(model, n_load_steps=50, max_iter=10, numerical_jacobian=False)
    else:
        t1 = 1
        dt = 1.0e-2
        solver = Euler_backward(model, t1, dt)
        # solver = Generalized_alpha_1(model, t1, dt, rtol=0, atol=1.0e-1, rho_inf=0.25)
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='RK45')
    sol = solver.solve()
    t = sol.t
    q = sol.q

    # animate configurations
    fig, ax = plt.subplots()
    # ax.axis('equal')
    scale = 2.5 * R
    ax.axis("equal")
    ax.set_xlim([-scale, scale])
    ax.set_ylim([-scale, scale])

    # prepare data for animation
    frames = q.shape[0]
    target_frames = min(frames, 200)
    frac = int(frames / target_frames)
    animation_time = 1
    interval = animation_time * 1000 / target_frames

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    (center_line0,) = ax.plot([], [], "-k")
    (nodes0,) = ax.plot([], [], "--ob")
    (center_line1,) = ax.plot([], [], "-g")
    (nodes1,) = ax.plot([], [], "--or")

    def animate(i):
        x, y, z = beam1.centerline(q[i], n=50).T
        center_line0.set_data(x, y)

        x, y, z = beam2.centerline(q[i], n=50).T
        center_line1.set_data(x, y)

        x, y = q[i][beam1.qDOF].reshape(2, -1)
        nodes0.set_data(x, y)

        x, y = q[i][beam2.qDOF].reshape(2, -1)
        nodes1.set_data(x, y)

        return (
            center_line0,
            center_line1,
            nodes0,
            nodes1,
        )

    animate(1)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )
    plt.show()

    # vtk export
    post_processing([beam1, beam2], t, q, "Arch", binary=True)


def top():
    # solver parameter
    t0 = 0
    t1 = 3
    dt = 5e-2

    # physical properties of the rope
    rho = 7850
    L = 50
    r = 5.0e-3
    A = np.pi * r**2
    I = A / 4 * r**2
    E = 210e9 * 1.0e-3
    EA = E * A
    EI = E * I
    material_model = Hooke(EA, EI)
    A_rho0 = A * rho

    # amplitude = 0.8 * L / (t1 - t0)
    amplitude = 0
    e = lambda t: amplitude * t
    e_t = lambda t: amplitude
    e_tt = lambda t: 0

    r_OB1 = lambda t: np.array([e(t), 0, 0])
    r_OB1_t = lambda t: np.array([e_t(t), 0, 0])
    r_OB1_tt = lambda t: np.array([e_tt(t), 0, 0])
    frame_left = Frame(r_OP=r_OB1, r_OP_t=r_OB1_t, r_OP_tt=r_OB1_tt)

    # omega = 2 * 2 * np.pi / (t1 - t0)
    # amplitude = 5
    # e = lambda t: amplitude * np.sin(omega * t)
    # e_t = lambda t: amplitude * omega * np.cos(omega * t)
    # e_tt = lambda t: -amplitude * omega**2 * np.sin(omega * t)

    # r_OB1 = lambda t: np.array([0, e(t), 0])
    # r_OB1_t = lambda t: np.array([0, e_t(t), 0])
    # r_OB1_tt = lambda t: np.array([0, e_tt(t), 0])
    # frame_left = Frame(r_OP=r_OB1, r_OP_t=r_OB1_t, r_OP_tt=r_OB1_tt)

    # discretization properties
    p = 2
    assert p >= 2
    nQP = int(np.ceil((p + 1) ** 2 / 2))
    print(f"nQP: {nQP}")
    nEl = 5

    # build reference configuration
    nNd = nEl + p
    X0 = np.linspace(0, L, nNd)
    Xi = uniform_knot_vector(p, nEl)
    for i in range(nNd):
        X0[i] = np.sum(Xi[i + 1 : i + p + 1])
    X0 = X0 * L / p
    Y0 = np.zeros_like(X0)
    Q = np.hstack((X0, Y0))

    u0 = np.zeros_like(Q)
    u0[0] = r_OB1_t(0)[0]
    u0[nNd] = r_OB1_t(0)[1]

    q0 = np.hstack((X0, Y0))

    beam = EulerBernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)

    # left joint
    # joint_left = Spherical_joint2D(frame_left, beam, r_OB1(0), frame_ID2=(0,))
    joint_left = Rigid_connection2D(frame_left, beam, r_OB1(0), frame_ID2=(0,))

    # right joint
    r_OB2 = np.array([L, 0, 0])
    frame_right = Frame(r_OP=r_OB2)
    # joint_right = Spherical_joint2D(beam, frame_right, r_OB2, frame_ID1=(1,))
    joint_right = Rigid_connection2D(beam, frame_right, r_OB2, frame_ID1=(1,))

    # gravity beam
    __g = np.array([0, -A_rho0 * 9.81, 0])
    if statics:
        f_g_beam = Line_force(lambda xi, t: t * __g, beam)
    else:
        f_g_beam = Line_force(lambda xi, t: __g, beam)

    # rigid body
    m = 5
    K_theta_S = m * np.eye(3)
    p0 = np.array([1, 0, 0, 0])
    q0 = np.concatenate((r_OB2, p0))
    u0 = np.array([0, 0, 0, 1, 0, 0])
    RB = Rigid_body_quaternion(m, K_theta_S, q0=q0, u0=u0)

    # gravity rigid body
    f_g_RB = Force(lambda t: np.array([0, -m * 9.81, 0]), RB)

    # joint = Rigid_connection(beam, RB, r_OB2, frame_ID1=(1,))
    joint = Spherical_joint(beam, RB, r_OB2, frame_ID1=(1,))

    # assemble the model
    model = System()
    model.add(beam)
    model.add(frame_left)
    model.add(joint_left)
    # model.add(frame_right)
    # model.add(joint_right)
    # model.add(f_g_beam)
    model.add(RB)
    model.add(f_g_RB)
    model.add(joint)
    model.assemble()

    if statics:
        solver = Newton(model, n_load_steps=10, max_iter=10, numerical_jacobian=True)
        # solver = Newton(model, n_load_steps=50, max_iter=10, numerical_jacobian=True)
        sol = solver.solve()
        t = sol.t
        q = sol.q
        # print(f'pot(t0, q0): {model.E_pot(t[0], q[0])}')
        # print(f'pot(t1, q1): {model.E_pot(t[-1], q[-1])}')
        # exit()
    else:
        # solver = Euler_backward(model, t1, dt, numerical_jacobian=False, debug=False)
        # solver = Euler_backward(model, t1, dt, numerical_jacobian=False, debug=True)
        # solver = Moreau(model, t1, dt)
        # solver = Moreau_sym(model, t1, dt)
        solver = Generalized_alpha_1(model, t1, dt, rtol=0, atol=1.0e-1, rho_inf=0.25)
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-1, method='RK23')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='RK45')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='DOP853')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='Radau')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='BDF')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='LSODA')

        # import cProfile, pstats
        # pr = cProfile.Profile()
        # pr.enable()
        sol = solver.solve()
        # pr.disable()

        # sortby = 'cumulative'
        # ps = pstats.Stats(pr).sort_stats(sortby)
        # ps.print_stats(0.05) # print only first 10% of the list
        # exit()

        t = sol.t
        q = sol.q
        # t, q, u, la_g, la_gamma = sol.unpack()

    # plt.plot(t, q[:, int(1.5 * nNd)])
    # plt.show()
    # exit()

    # animate configurations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    scale = L
    ax.set_xlim3d(left=0, right=L)
    ax.set_ylim3d(bottom=-L / 2, top=L / 2)
    ax.set_zlim3d(bottom=-L / 2, top=L / 2)

    # prepare data for animation
    if statics:
        frames = len(t)
        interval = 100
    else:
        frames = len(t)
        target_frames = min(len(t), 100)
        frac = int(frames / target_frames)
        animation_time = 1
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t[::frac]
        q = q[::frac]

    x0, y0, z0 = beam.centerline(q[0]).T
    (center_line0,) = ax.plot(x0, y0, z0, "-k")

    x1, y1, z1 = beam.centerline(q[-1]).T
    (center_line,) = ax.plot(x1, y1, z1, "-b")

    x_S, y_S, z_S = RB.r_OP(t, q[0][RB.qDOF])
    A_IK = RB.A_IK(t, q[0][RB.qDOF])
    d1 = A_IK[:, 0] * L / 4
    d2 = A_IK[:, 1] * L / 4
    d3 = A_IK[:, 2] * L / 4

    (COM,) = ax.plot([x_S], [y_S], [z_S], "ok")
    (d1_,) = ax.plot([x_S, x_S + d1[0]], [y_S, y_S + d1[1]], [z_S, z_S + d1[2]], "-r")
    (d2_,) = ax.plot([x_S, x_S + d2[0]], [y_S, y_S + d2[1]], [z_S, z_S + d2[2]], "-g")
    (d3_,) = ax.plot([x_S, x_S + d3[0]], [y_S, y_S + d3[1]], [z_S, z_S + d3[2]], "-b")

    def update(t, q, center_line):
        x, y, z = beam.centerline(q).T
        center_line.set_data(x, y)
        center_line.set_3d_properties(z)

        x_S, y_S, z_S = RB.r_OP(t, q[RB.qDOF])
        A_IK = RB.A_IK(t, q[RB.qDOF])
        d1 = A_IK[:, 0] * L / 4
        d2 = A_IK[:, 1] * L / 4
        d3 = A_IK[:, 2] * L / 4

        COM.set_data(np.array([x_S]), np.array([y_S]))
        COM.set_3d_properties(np.array([z_S]))

        d1_.set_data(np.array([x_S, x_S + d1[0]]), np.array([y_S, y_S + d1[1]]))
        d1_.set_3d_properties(np.array([z_S, z_S + d1[2]]))

        d2_.set_data(np.array([x_S, x_S + d2[0]]), np.array([y_S, y_S + d2[1]]))
        d2_.set_3d_properties(np.array([z_S, z_S + d2[2]]))

        d3_.set_data(np.array([x_S, x_S + d3[0]]), np.array([y_S, y_S + d3[1]]))
        d3_.set_3d_properties(np.array([z_S, z_S + d3[2]]))

        return center_line, COM, d1_, d2_, d3_

    def animate(i):
        update(t[i], q[i], center_line)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )
    plt.show()


if __name__ == "__main__":
    # top()
    B_spline_fitting()
