# from cardillo.model.rope import Hooke, Rope
# from cardillo.model.frame import Frame
# from cardillo.model.bilateral_constraints import Spherical_joint
# from cardillo.model import Model
# from cardillo.solver import Euler_backward, Moreau, Moreau_sym, Generalized_alpha_1, Scipy_ivp, Newton
# from cardillo.model.line_force.line_force import Line_force
# from cardillo.discretization import uniform_knot_vector

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.animation as animation

# import numpy as np

# # statics = True
# statics = False

# if __name__ == "__main__":
#     # # left joint
#     # r_OB1 = lambda t: np.zeros(3)
#     # r_OB1_t = lambda t: np.zeros(3)
#     # r_OB1_tt = lambda t: np.zeros(3)
#     # frame_left = Frame(r_OP=r_OB1)

#     omega = 2 * np.pi
#     A = 20
#     r_OB1 = lambda t: np.array([0, 0, A * np.sin(omega * t)])
#     r_OB1_t = lambda t: np.array([0, 0, A * omega * np.cos(omega * t)])
#     r_OB1_tt = lambda t: np.array([0, 0, -A * omega**2 * np.sin(omega * t)])
#     frame_left = Frame(r_OP=r_OB1, r_OP_t=r_OB1_t, r_OP_tt=r_OB1_tt)
    
#     # physical properties of the rope
#     dim = 3
#     assert dim == 3
#     L = 50
#     r = 3.0e-3
#     A = np.pi * r**2
#     EA = 4.0e8 * A * 1.0e-2
#     material_model = Hooke(EA)
#     A_rho0 = 10 * A

#     # discretization properties
#     B_splines = True
#     # B_splines = False
#     p = 2
#     nQP = int(np.ceil((p + 1)**2 / 2))
#     print(f'nQP: {nQP}')
#     nEl = 10

#     # build reference configuration
#     if B_splines:
#         nNd = nEl + p
#     else:
#         nNd = nEl * p + 1
#     X0 = np.linspace(0, L, nNd)
#     Xi = uniform_knot_vector(p, nEl)
#     if B_splines:
#         for i in range(nNd):
#             X0[i] = np.sum(Xi[i+1:i+p+1])
#         X0 = X0 * L / p
#     Y0 = np.zeros_like(X0)
#     Z0 = np.zeros_like(X0)
#     Q = np.hstack((X0, Y0, Z0))
#     u0 = np.zeros_like(Q)
#     u0[0] = r_OB1_t(0)[0]
#     u0[nNd] = r_OB1_t(0)[1]
#     u0[2 * nNd] = r_OB1_t(0)[2]

#     q0 = np.hstack((X0, Y0, Z0)) * 1.2

#     rope = Rope(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0, B_splines=B_splines, dim=dim)

#     joint_left = Spherical_joint(frame_left, rope, r_OB1(0), frame_ID2=(0,))

#     # right joint
#     r_OB2 = np.array([L, 0, 0]) * 1.2
#     frame_right = Frame(r_OP=r_OB2)
#     joint_right = Spherical_joint(rope, frame_right, r_OB2, frame_ID1=(1,))

#     # gravity
#     g = np.array([0, 0, - A_rho0 * L * 9.81])
#     if statics:
#         f_g = Line_force(lambda xi, t: t * g * 10, rope)
#     else:
#         f_g = Line_force(lambda xi, t: g, rope)

#     # assemble the model
#     model = Model()
#     model.add(rope)
#     model.add(frame_left)
#     model.add(joint_left)
#     model.add(frame_right)
#     model.add(joint_right)
#     model.add(f_g)
#     model.assemble()

#     if statics:
#         solver = Newton(model, n_load_stepts=10, max_iter=10, numerical_jacobian=False)
#         sol = solver.solve()
#         t = sol.t
#         q = sol.q
#         print(f'pot(t0, q0): {model.E_pot(t[0], q[0])}')
#         print(f'pot(t1, q1): {model.E_pot(t[-1], q[-1])}')
#         print(f'r_OP(0.5): {rope.r_OP(t[-1], q[-1][rope.elDOF_P((0.5,))], (0.5,))}')
#     else:
#         t0 = 0
#         t1 = 1
#         dt = 5e-3
#         solver = Euler_backward(model, t1, dt, numerical_jacobian=False, debug=False)
#         # solver = Moreau(model, t1, dt)
#         # solver = Moreau_sym(model, t1, dt)
#         # solver = Generalized_alpha_1(model, t1, dt, rho_inf=0.75)
#         # solver = Scipy_ivp(model, t1, dt, rtol=1.0e-2, atol=1.e-2, method='RK23')
#         # solver = Scipy_ivp(model, t1, dt, rtol=1.0e-2, atol=1.e-2, method='RK45')
#         # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='DOP853')
#         # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='Radau')
#         # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='BDF')
#         # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='LSODA')

#         sol = solver.solve()

#         # from cardillo.solver import save_solution
#         # save_solution(sol, f'test')

#         # from cardillo.solver import load_solution
#         # sol = load_solution(f'test')

#         t = sol.t
#         q = sol.q
#         # t, q, u, la_g, la_gamma = sol.unpack()

#     # animate configurations
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     ax.set_xlabel('x [m]')
#     ax.set_ylabel('y [m]')
#     ax.set_zlabel('z [m]')
#     scale = L
#     ax.set_xlim3d(left=0, right=L)
#     ax.set_ylim3d(bottom=-L/2, top=L/2)
#     ax.set_zlim3d(bottom=-L/2, top=L/2)

#     # prepare data for animation
#     if statics:
#         frames = len(t)
#         interval = 100
#     else:
#         frames = len(t)
#         target_frames = 100
#         frac = int(frames / target_frames)
#         animation_time = 1
#         interval = animation_time * 1000 / target_frames

#         frames = target_frames
#         t = t[::frac]
#         q = q[::frac]
    
#     x0, y0, z0 = q0.reshape((3, -1))
#     center_line0, = ax.plot(x0, y0, z0, '-ok')

#     x1, y1, z1 = q[-1].reshape((3, -1))
#     center_line, = ax.plot(x1, y1, z1, '-ob')

#     def update(t, q, center_line):
#         if dim ==2:
#             x, y = q.reshape((2, -1))
#             center_line.set_data(x, y)
#             center_line.set_3d_properties(np.zeros_like(x))
#         elif dim == 3:
#             x, y, z = q.reshape((3, -1))
#             center_line.set_data(x, y)
#             center_line.set_3d_properties(z)

#         return center_line,

#     def animate(i):
#         update(t[i], q[i], center_line)

#     anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
#     plt.show()

from cardillo.model.rope import Hooke, Rope
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints import Spherical_joint
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
    # # left joint
    # r_OB1 = np.zeros(3)
    # frame_left = Frame(r_OP=r_OB1)
    # joint_left = Spherical_joint(frame_left, rope, r_OB1, frame_ID2=(0,))

    omega = 2 * np.pi
    A = 5
    r_OB1 = lambda t: np.array([0, 0, A * np.sin(omega * t)])
    r_OB1_t = lambda t: np.array([0, 0, A * omega * np.cos(omega * t)])
    r_OB1_tt = lambda t: np.array([0, 0, -A * omega**2 * np.sin(omega * t)])
    frame_left = Frame(r_OP=r_OB1, r_OP_t=r_OB1_t, r_OP_tt=r_OB1_tt)
    
    # physical properties of the rope
    dim = 3
    assert dim == 3
    L = 50
    r = 3.0e-3
    A = np.pi * r**2
    EA = 4.0e8 * A * 1.0e-2
    material_model = Hooke(EA)
    A_rho0 = 10 * A

    # discretization properties
    B_splines = True
    # B_splines = False
    p = 2
    nQP = int(np.ceil((p + 1)**2 / 2))
    print(f'nQP: {nQP}')
    nEl = 10

    # build reference configuration
    if B_splines:
        nNd = nEl + p
    else:
        nNd = nEl * p + 1
    X0 = np.linspace(0, L, nNd)
    Xi = uniform_knot_vector(p, nEl)
    if B_splines:
        for i in range(nNd):
            X0[i] = np.sum(Xi[i+1:i+p+1])
        X0 = X0 * L / p
    Y0 = np.zeros_like(X0)
    Z0 = np.zeros_like(X0)
    Q = np.hstack((X0, Y0, Z0))
    u0 = np.zeros_like(Q)
    u0[0] = r_OB1_t(0)[0]
    u0[nNd] = r_OB1_t(0)[1]
    u0[2 * nNd] = r_OB1_t(0)[2]

    q0 = np.hstack((X0, Y0, Z0)) * 1.2

    rope = Rope(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0, B_splines=B_splines, dim=dim)

    joint_left = Spherical_joint(frame_left, rope, r_OB1(0), frame_ID2=(0,))

    # right joint
    r_OB2 = np.array([L, 0, 0]) * 1.2
    frame_right = Frame(r_OP=r_OB2)
    joint_right = Spherical_joint(rope, frame_right, r_OB2, frame_ID1=(1,))

    # gravity
    g = np.array([0, 0, - A_rho0 * L * 9.81])
    if statics:
        f_g = Line_force(lambda xi, t: t * g * 10, rope)
    else:
        f_g = Line_force(lambda xi, t: g, rope)

    # assemble the model
    model = Model()
    model.add(rope)
    model.add(frame_left)
    model.add(joint_left)
    model.add(frame_right)
    model.add(joint_right)
    model.add(f_g)
    model.assemble()

    if statics:
        solver = Newton(model, n_load_stepts=10, max_iter=10, numerical_jacobian=False)
        sol = solver.solve()
        t = sol.t
        q = sol.q
        print(f'pot(t0, q0): {model.E_pot(t[0], q[0])}')
        print(f'pot(t1, q1): {model.E_pot(t[-1], q[-1])}')
        print(f'r_OP(0.5): {rope.r_OP(t[-1], q[-1][rope.elDOF_P((0.5,))], (0.5,))}')
    else:
        t0 = 0
        t1 = 1
        dt = 5e-3
        # solver = Euler_backward(model, t1, dt, numerical_jacobian=False, debug=False)
        # solver = Moreau(model, t1, dt)
        # solver = Moreau_sym(model, t1, dt)
        # solver = Generalized_alpha_1(model, t1, dt, rho_inf=0.75)
        # solver = Scipy_ivp(model, t1, dt, rtol=1.0e-2, atol=1.e-2, method='RK23')
        solver = Scipy_ivp(model, t1, dt, rtol=1.0e-2, atol=1.e-2, method='RK45')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='DOP853')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='Radau')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='BDF')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='LSODA')

        sol = solver.solve()

        # from cardillo.solver import save_solution
        # save_solution(sol, f'test')

        # from cardillo.solver import load_solution
        # sol = load_solution(f'test')

        t = sol.t
        q = sol.q
        # t, q, u, la_g, la_gamma = sol.unpack()

    # animate configurations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    scale = L
    ax.set_xlim3d(left=0, right=L)
    ax.set_ylim3d(bottom=-L/2, top=L/2)
    ax.set_zlim3d(bottom=-L/2, top=L/2)

    # prepare data for animation
    if statics:
        frames = len(t)
        interval = 100
    else:
        frames = len(t)
        target_frames = 100
        frac = int(frames / target_frames)
        animation_time = 1
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t[::frac]
        q = q[::frac]
    
    x0, y0, z0 = q0.reshape((3, -1))
    center_line0, = ax.plot(x0, y0, z0, '-ok')

    x1, y1, z1 = q[-1].reshape((3, -1))
    center_line, = ax.plot(x1, y1, z1, '-ob')

    def update(t, q, center_line):
        if dim ==2:
            x, y = q.reshape((2, -1))
            center_line.set_data(x, y)
            center_line.set_3d_properties(np.zeros_like(x))
        elif dim == 3:
            x, y, z = q.reshape((3, -1))
            center_line.set_data(x, y)
            center_line.set_3d_properties(z)

        return center_line,

    def animate(i):
        update(t[i], q[i], center_line)

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
    plt.show()