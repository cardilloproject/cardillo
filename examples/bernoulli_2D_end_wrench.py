from cardillo.model.classical_beams import Hooke, Euler_bernoulli2D
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import Rigid_connection2D
from cardillo.model import Model
from cardillo.solver import Euler_backward, Moreau, Moreau_sym, Generalized_alpha_1, Scipy_ivp, Newton
from cardillo.model.line_force.line_force import Line_force
from cardillo.discretization import uniform_knot_vector
from cardillo.model.force import Force, K_Force
from cardillo.model.moment import K_Moment

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import numpy as np

statics = True
# statics = False
animate = False

if __name__ == "__main__":
    # solver parameter
    t0 = 0
    t1 = 10
    dt = 5e-2

    # physical properties of the rope
    rho = 7850
    L = 2 * np.pi
    r = 5.0e-3
    A = np.pi * r**2
    I = A / 4 * r**2
    E = 210e9 * 1.0e-3
    EA = E * A
    EI = E * I
    material_model = Hooke(EA, EI)
    A_rho0 = A * rho

    r_OB1 = np.zeros(3)
    frame_left = Frame(r_OP=r_OB1)

    # discretization properties
    p = 3
    assert p >= 2
    nQP = int(np.ceil((p + 1)**2 / 2))
    # nQP = 2
    print(f'nQP: {nQP}')
    nEl = 20

    # build reference configuration
    nNd = nEl + p
    X0 = np.linspace(0, L, nNd)
    Xi = uniform_knot_vector(p, nEl)
    for i in range(nNd):
        X0[i] = np.sum(Xi[i+1:i+p+1])
    X0 = X0 * L / p
    Y0 = np.zeros_like(X0)

    Q = np.hstack((X0, Y0))
    q0 = np.hstack((X0, Y0))
    u0 = np.zeros_like(Q)
    beam = Euler_bernoulli2D(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)

    # left joint
    joint_left = Rigid_connection2D(frame_left, beam, r_OB1, frame_ID2=(0,))

    # gravity beam
    __g = np.array([0, - A_rho0 * 9.81, 0])
    if statics:
        f_g_beam = Line_force(lambda xi, t: t * __g, beam)
    else:
        f_g_beam = Line_force(lambda xi, t: __g, beam)

    # wrench at right end
    F = lambda t: t * np.array([0, 1e-2, 0])
    force = K_Force(F, beam, frame_ID=(1,))

    M_z = lambda t: t * 2 * np.pi * EI / L
    M = lambda t: np.array([0, 0, M_z(t)])
    moment = K_Moment(M, beam, (1,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame_left)
    model.add(joint_left)
    # model.add(f_g_beam)
    # model.add(force)
    model.add(moment)
    model.assemble()

    if statics:
        solver = Newton(model, n_load_stepts=5, max_iter=10, numerical_jacobian=False)
        # solver = Newton(model, n_load_stepts=50, max_iter=10, numerical_jacobian=True)
        sol = solver.solve()
        t = sol.t
        q = sol.q
        # print(f'pot(t0, q0): {model.E_pot(t[0], q[0])}')
        # print(f'pot(t1, q1): {model.E_pot(t[-1], q[-1])}')
        # exit()
        x, y, z = beam.centerline(q[-1]).T
        # x = q[-1][:nNd]
        # y = q[-1][nNd:]
        plt.plot(x, y)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')

        plt.show()
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

        sol = solver.solve()

        # from cardillo.solver import save_solution
        # save_solution(sol, f'test')

        # from cardillo.solver import load_solution
        # sol = load_solution(f'test')

        t = sol.t
        q = sol.q
       

        # t, q, u, la_g, la_gamma = sol.unpack()


    # plt.plot(t, q[:, int(1.5 * nNd)])
    # plt.show()
    # exit()

    if animate:
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
            target_frames = min(len(t), 100)
            frac = int(frames / target_frames)
            animation_time = 1
            interval = animation_time * 1000 / target_frames

            frames = target_frames
            t = t[::frac]
            q = q[::frac]
        
        x0, y0, z0 = beam.centerline(q[0]).T
        center_line0, = ax.plot(x0, y0, z0, '-k')

        x1, y1, z1 = beam.centerline(q[-1]).T
        center_line, = ax.plot(x1, y1, z1, '-b')

        def update(t, q, center_line):
            x, y, z = beam.centerline(q).T
            center_line.set_data(x, y)
            center_line.set_3d_properties(z)


            return center_line

        def animate(i):
            update(t[i], q[i], center_line)

        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
        plt.show()