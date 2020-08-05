from scipy.optimize.minpack import fixed_point
from cardillo.model.rope import Hooke, Rope, Inextensible_Rope
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import Spherical_joint, Spherical_joint2D
from cardillo.model import Model
from cardillo.solver import Euler_backward, Moreau, Moreau_sym, Generalized_alpha_1, Scipy_ivp, Newton
from cardillo.model.line_force.line_force import Line_force
from cardillo.discretization import uniform_knot_vector

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import numpy as np

# statics = True
statics = False

if __name__ == "__main__":
    # dynamic solver
    t0 = 0
    t1 = 20
    dt = 5e-2
    
    # physical properties of the rope
    dim = 3
    assert dim == 3
    L = 2 * np.pi
    EA = 1.0e3
    material_model = Hooke(EA)
    A_rho0 = 1e-1

    # left joint
    # r_OB1 = lambda t: np.array([0, L, 0])
    r_OB1 = lambda t: np.zeros(3)
    r_OB1_t = lambda t: np.zeros(3)
    r_OB1_tt = lambda t: np.zeros(3)
    frame0 = Frame(r_OP=r_OB1)

    omega = 2 * np.pi
    a = 5
    # r_OB2 = lambda t: np.array([0, -L, 0]) + min(t, 0.5) * np.array([L/2, 0, 0]) + max(2*t-1, 0) * np.array([0, L/2, 0])
    # r_OB2_t = lambda t: np.array([L/2, L, 0]) if t <=1 else np.zeros(3)
    # r_OB2_tt = lambda t: np.zeros(3)
    # frame1 = Frame(r_OP=r_OB2, r_OP_t=r_OB2_t, r_OP_tt=r_OB2_tt)
    r_OB2 = lambda t: np.array([L, 0, 0]) - min(1, max(3/t1*t-1, 0)) * np.array([0.5*L, 0, 0])
    frame1 = Frame(r_OP=r_OB2)

    # discretization properties
    B_splines = True
    # B_splines = False
    p = 3
    nQP = int(np.ceil((p + 1)**2 / 2))
    # nQP = 2
    print(f'nQP: {nQP}')
    nEl = 5

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
            X0[i] = np.sum(Xi[i+1:i+p+1])
        X0 = X0 * L / p

    Q = np.hstack((X0, Y0, Z0))
    u0 = np.zeros_like(Q)
    u0[0] = r_OB1_t(0)[0]
    u0[nNd] = r_OB1_t(0)[1]
    u0[2 * nNd] = r_OB1_t(0)[2]

    q0 = np.hstack((X0, Y0, Z0))

    # rope_ = Rope(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0, B_splines=B_splines, dim=dim)
    inextensible_rope = Inextensible_Rope(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0, B_splines=B_splines, dim=dim)
    # ropes = [rope_, inextensible_rope]
    # ropes = [rope_]
    ropes = [inextensible_rope]

    sols = []
    for rope in ropes:
        # left joint
        la_g0 = np.ones(3)*1.0e-6
        # la_g0 = np.zeros(3)
        joint_left = Spherical_joint(frame0, rope, r_OB1(0), frame_ID2=(0,), la_g0=la_g0)

        # right joint
        joint_right = Spherical_joint(rope, frame1, r_OB2(0), frame_ID1=(1,), la_g0=la_g0)

        # gravity
        g = np.array([0, - A_rho0 * L * 9.81, 0]) * 1.0e-1
        if statics:
            # # f_g = Line_force(lambda xi, t: max(0, min(t, 0.5)) * 2 * g, rope)
            # f_g = Line_force(lambda xi, t: (t - 0.5) * 2 * g if t >= 0.5 else np.zeros(3), rope)
            f_g = Line_force(lambda xi, t: t * g, rope)
        else:
            f_g = Line_force(lambda xi, t: g, rope)

        # assemble the model
        model = Model()
        model.add(rope)
        model.add(frame0)
        model.add(joint_left)
        model.add(frame1)
        model.add(joint_right)
        model.add(f_g)
        model.assemble()

        if statics:
            solver = Newton(model, n_load_steps=20, max_iter=30, numerical_jacobian=True, tol=1.0e-6)
        else:
            solver = Euler_backward(model, t1, dt, numerical_jacobian=False, debug=False, newton_tol=1.0e-6)
            # solver = Moreau(model, t1, dt, fix_point_tol=1.0e-6)
            # solver = Moreau_sym(model, t1, dt)
            # solver = Generalized_alpha_1(model, t1, dt=dt, variable_dt=True, t_eval=np.linspace(t0, t1, 100), rho_inf=0.75)
            # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='RK23')
            # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='RK45')
            # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='DOP853')
            # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='Radau')
            # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='BDF')
            # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='LSODA')
        sols.append( solver.solve() )

    fig, ax = plt.subplots()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_xlim([-1.5*L, 1.5*L])
    ax.set_ylim([-1.5*L, 1.5*L])
    ax.grid(linestyle='-', linewidth='0.5')
    if statics:
        x, y, _ = rope_.centerline(sols[0].q[-1]).T
        ax.plot(x, y, '-k', linewidth=4)
        ax.plot(*sols[0].q[-1].reshape(3, -1)[:2], '--ob')

        # x, y, _ = inextensible_rope.centerline(sols[1].q[-1]).T
        # ax.plot(x, y, '-r', linewidth=2)

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
        
        center_line0, = ax.plot([], [], '-k')
        # center_line1, = ax.plot([], [], '--b')

        def animate(i):
            x, y, _ = rope_.centerline(q0[i], n=100).T
            center_line0.set_data(x, y)

            # x, y, _ = beam.centerline(q1[i], n=50).T
            # center_line1.set_data(x, y)

            return center_line0, #center_line1

        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
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