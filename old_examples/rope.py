from scipy.optimize.minpack import fixed_point
from cardillo.model.rope import Hooke, Rope, Inextensible_Rope
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import (
    Spherical_joint,
    Spherical_joint2D,
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
from cardillo.discretization import uniform_knot_vector
from cardillo.math.algebra import A_IK_basic_z

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import numpy as np

# special cases of the smooth step function,
# see https://en.wikipedia.org/wiki/Smoothstep#Generalization_to_higher-order_equations
def smoothstep3(x, x_min=0, x_max=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return -20 * x**7 + 70 * x**6 - 84 * x**5 + 35 * x**4


statics = True
# statics = False

if __name__ == "__main__":
    # dynamic solver
    t0 = 0
    t1 = 1
    dt = 5e-3

    # physical properties of the rope
    dim = 3
    assert dim == 3
    L = 2 * np.pi
    EA = 1.0e3
    material_model = Hooke(EA)
    A_rho0 = 1e0

    # discretization properties
    B_splines = True
    # B_splines = False
    p = 2
    nQP = int(np.ceil((p + 1) ** 2 / 2))
    # nQP = 3
    print(f"nQP: {nQP}")
    nEl = 10

    # build reference configuration
    if B_splines:
        nNd = nEl + p
    else:
        nNd = nEl * p + 1
    X0 = np.linspace(0, L, nNd)
    Y0 = np.zeros(nNd)
    Z0 = np.zeros(nNd)
    Xi = uniform_knot_vector(p, nEl)
    if B_splines:
        for i in range(nNd):
            X0[i] = np.sum(Xi[i + 1 : i + p + 1])
        X0 = X0 * L / p

    angle = np.pi / 4
    for i in range(0, len(X0)):
        X0[i], Y0[i], _ = A_IK_basic_z(angle) @ np.array([X0[i], Y0[i], 0])

    if dim == 2:
        Q = np.hstack((X0, Y0))
    else:
        Q = np.hstack((X0, Y0, Z0))

    c1 = np.zeros(3)
    # c1 = np.array([0, 1, 0]) / t1
    r_OB1 = lambda t: t**2 * c1 + np.array([X0[0], Y0[0], Z0[0]])
    r_OB1_t = lambda t: 2 * t * c1
    r_OB1_tt = lambda t: 2 * c1
    # r_OB1 = lambda t: t * c1
    # r_OB1_t = lambda t: c1
    # r_OB1_tt = lambda t: np.zeros(3)
    # r_OB1 = lambda t: np.zeros(3)
    # r_OB1_t = lambda t: np.zeros(3)
    # r_OB1_tt = lambda t: np.zeros(3)
    frame0 = Frame(r_OP=r_OB1, r_OP_t=r_OB1_t, r_OP_tt=r_OB1_tt)

    # # c2 = np.array([0, 0.5*L, 0])
    # c2 = np.array([0, -np.sqrt(2) / 2 * L, 0]) / (t1 / 3)
    # # c2 = np.zeros(3)
    # r_OB2 = lambda t: t**2 * c2 + np.array([X0[-1], Y0[-1], Z0[-1]]) if t < t1 / 3 else np.zeros(3)
    # r_OB2_t = lambda t: 2 * t * c2 if t < t1 / 3 else np.zeros(3)
    # r_OB2_tt = lambda t: 2 * c2 if t < t1 / 3 else np.zeros(3)
    # frame1 = Frame(r_OP=r_OB2, r_OP_t=r_OB2_t, r_OP_tt=r_OB2_tt)
    r_OB2 = (
        lambda t: np.sqrt(2) / 2 * L * np.array([1, 1, 0])
    )  # + smoothstep3(t, x_min=0, x_max=t1/3) * np.array([0, -np.sqrt(2) / 2 * L, 0])
    # r_OB2 = lambda t: np.sqrt(2) / 2 * L * np.array([1, 1, 0]) + smoothstep3(t, x_min=0, x_max=t1/3) * np.array([-np.sqrt(2) / 4 * L, 0, 0])
    frame1 = Frame(r_OP=r_OB2)

    u0 = np.zeros_like(Q)
    u0[0] = r_OB1_t(0)[0]
    u0[nNd] = r_OB1_t(0)[1]
    if dim == 3:
        u0[2 * nNd] = r_OB1_t(0)[2]

    # u0[nNd-1] = r_OB2_t(0)[0]
    # u0[2*nNd-1] = r_OB2_t(0)[1]
    # if dim == 3:
    #     u0[3*nNd-1] = r_OB2_t(0)[2]

    if dim == 2:
        q0 = np.hstack((X0, Y0))
        # q0 = np.hstack((Y0, -X0))
    else:
        q0 = np.hstack((X0, Y0, Z0))
        # q0 = np.hstack((Y0, -X0, Z0))

    # alpha = 1.0e-2
    # beta = 1.0e-2
    alpha = 0
    beta = 0

    rope_ = Rope(
        A_rho0,
        material_model,
        p,
        nEl,
        nQP,
        alpha=alpha,
        beta=beta,
        Q=Q,
        q0=q0,
        u0=u0,
        B_splines=B_splines,
        dim=dim,
    )
    ropes = [rope_]

    # inextensible_rope = Inextensible_Rope(A_rho0, material_model, p, nEl, nQP, alpha=alpha, beta=beta, Q=Q, q0=q0, u0=u0, B_splines=B_splines, dim=dim)
    # ropes = [inextensible_rope]
    # ropes = [rope_, inextensible_rope]
    # ropes = [inextensible_rope, rope_]

    sols = []
    for rope in ropes:
        # left joint
        if dim == 2:
            # la_g0 = np.random.rand(3)*1.0e-6
            la_g0 = np.ones(2) * 1.0e-6
            # la_g0 = np.zeros(3)
            joint_left = Spherical_joint2D(
                frame0, rope, r_OB1(0), frame_ID2=(0,), la_g0=la_g0
            )
        else:
            # la_g0 = np.random.rand(3)*1.0e-6
            la_g0 = np.ones(3) * 1.0e-6
            # la_g0 = np.zeros(3)
            joint_left = Spherical_joint(
                frame0, rope, r_OB1(0), frame_ID2=(0,), la_g0=la_g0
            )

        # right joint
        if dim == 2:
            joint_right = Spherical_joint2D(
                rope, frame1, r_OB2(0), frame_ID2=(1,), la_g0=la_g0
            )
        else:
            # la_g0 = np.random.rand(3)*1.0e-6
            la_g0 = np.ones(3) * 1.0e-6
            # la_g0 = np.zeros(3)
            joint_right = Spherical_joint(
                rope, frame1, r_OB2(0), frame_ID1=(1,), la_g0=la_g0
            )

        # gravity
        fg = np.array([0, -A_rho0 * 9.81, 0]) * 1.0e0
        if statics:
            # # f_g = Line_force(lambda xi, t: max(0, min(t, 0.5)) * 2 * fg, rope)
            # f_g = Line_force(lambda xi, t: (t - 0.5) * 2 * fg if t >= 0.5 else np.zeros(3), rope)
            f_g = Line_force(lambda xi, t: t * fg, rope)
        else:
            # f_g = Line_force(lambda xi, t: t / t1 * g, rope)
            f_g = Line_force(lambda xi, t: fg, rope)

        # assemble the model
        model = System()
        model.add(rope)
        model.add(frame0)
        model.add(joint_left)
        model.add(frame1)
        model.add(joint_right)
        model.add(f_g)
        model.assemble()

        if statics:
            solver = Newton(
                model, n_load_steps=10, max_iter=30, numerical_jacobian=True, tol=1.0e-6
            )
        else:
            # solver = Euler_backward(model, t1, dt, numerical_jacobian=False, debug=False, newton_tol=1.0e-6)
            # solver = Moreau(model, t1, dt, fix_point_tol=1.0e-6)
            # solver = Moreau_sym(model, t1, dt)
            # solver = Generalized_alpha_1(model, t1, dt=dt, variable_dt=True, t_eval=np.linspace(t0, t1, 100), rho_inf=0.75)
            # solver = Scipy_ivp(model, t1, dt, atol=1.e-1, method='RK23')
            solver = Scipy_ivp(model, t1, dt, atol=1.0e-1, method="RK45")
            # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='DOP853')
            # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='Radau')
            # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='BDF')
            # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='LSODA')
        sols.append(solver.solve())

    fig, ax = plt.subplots()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim([-1.5 * L, 1.5 * L])
    ax.set_ylim([-1.5 * L, 1.5 * L])
    ax.grid(linestyle="-", linewidth="0.5")
    if statics:
        # raise RuntimeError('TODO!')
        x, y, _ = rope_.centerline(sols[0].q[-1]).T
        ax.plot(x, y, "-k")
        ax.plot(*sols[0].q[-1].reshape(3, -1)[:2], "ok")

        # x, y, _ = inextensible_rope.centerline(sols[1].q[-1]).T
        # ax.plot(x, y, '--r')
        # ax.plot(*sols[1].q[-1].reshape(3, -1)[:2], 'xr')

        plt.show()
    else:
        # prepare data for animation
        t = sols[0].t
        frames = len(t)
        target_frames = min(len(t), 100)
        frac = int(frames / target_frames)
        animation_time = 10
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t[::frac]
        q0 = sols[0].q[::frac]
        # q1 = sols[1].q[::frac]
        q1 = q0

        (center_line0,) = ax.plot([], [], "-k")
        (nodes0,) = ax.plot([], [], "ok")
        (center_line1,) = ax.plot([], [], "--r")
        (nodes1,) = ax.plot([], [], "xr")

        def animate(i):
            x, y, _ = rope_.centerline(q0[i], n=100).T
            center_line0.set_data(x, y)
            if dim == 2:
                nodes0.set_data(*q0[i].reshape(2, -1))
            else:
                nodes0.set_data(*q0[i].reshape(3, -1)[:2])

            # x, y, _ = inextensible_rope.centerline(q1[i], n=50).T
            # center_line1.set_data(x, y)
            # if dim == 2:
            #     nodes1.set_data(*q1[i].reshape(2, -1))
            # else:
            #     nodes1.set_data(*q1[i].reshape(3, -1)[:2])

            return center_line0, center_line1, nodes0, nodes1

        anim = animation.FuncAnimation(
            fig, animate, frames=frames, interval=interval, blit=False
        )
        plt.show()

    # # # animate configurations
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')

    # # ax.set_xlabel('x [m]')
    # # ax.set_ylabel('y [m]')
    # # ax.set_zlabel('z [m]')
    # # scale = L
    # # ax.set_xlim3d(left=0, right=L)
    # # ax.set_ylim3d(bottom=-L/2, top=L/2)
    # # ax.set_zlim3d(bottom=-L/2, top=L/2)

    # # # prepare data for animation
    # # if statics:
    # #     frames = len(t)
    # #     interval = 100
    # # else:
    # #     frames = len(t)
    # #     target_frames = 100
    # #     frac = int(frames / target_frames)
    # #     animation_time = 1
    # #     interval = animation_time * 1000 / target_frames

    # #     frames = target_frames
    # #     t = t[::frac]
    # #     q = q[::frac]

    # # x0, y0, z0 = q0.reshape((3, -1))
    # # center_line0, = ax.plot(x0, y0, z0, '-ok')

    # # x1, y1, z1 = q[-1].reshape((3, -1))
    # # center_line, = ax.plot(x1, y1, z1, '-ob')

    # # def update(t, q, center_line):
    # #     if dim ==2:
    # #         x, y = q.reshape((2, -1))
    # #         center_line.set_data(x, y)
    # #         center_line.set_3d_properties(np.zeros_like(x))
    # #     elif dim == 3:
    # #         x, y, z = q.reshape((3, -1))
    # #         center_line.set_data(x, y)
    # #         center_line.set_3d_properties(z)

    # #     return center_line,

    # # def animate(i):
    # #     update(t[i], q[i], center_line)

    # # anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
    # # plt.show()
