from cardillo.model.classical_beams.planar import Hooke, Euler_bernoulli, Inextensible_Euler_bernoulli
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import Rigid_connection2D, Spherical_joint2D
from cardillo.model import Model
from cardillo.solver import Newton, Euler_backward, Generalized_alpha_1
from cardillo.discretization import uniform_knot_vector
from cardillo.model.force import Force
from cardillo.model.line_force import Line_force

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

# statics = True
statics = False

def inextensible_rope():
    t1 = 10
    dt = 5e-1

    L = 2 * np.pi
    EA = 5
    EI = 2 * 1.0e-1
    material_model = Hooke(EA, EI)
    A_rho0 = 0.1

    r_OB1 = np.zeros(3)
    frame_left = Frame(r_OP=r_OB1)
    r_OB2 = np.array([L, 0, 0])
    frame_right = Frame(r_OP=r_OB2)

    # discretization properties
    p = 3
    assert p >= 2
    nQP = int(np.ceil((p + 1)**2 / 2))
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
    # q0 = np.hstack((Y0, -X0))
    u0 = np.zeros_like(Q)

    bernoulli = Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)
    inextensible_bernoulli = Inextensible_Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)
    # beams = [bernoulli, inextensible_bernoulli]
    # beams = [inextensible_bernoulli, bernoulli]
    beams = [bernoulli]

    sols = []
    for beam in beams:
        model = Model()
        model.add(beam)
        model.add(frame_left)
        model.add(Rigid_connection2D(frame_left, beam, r_OB1, frame_ID2=(0,)))
        model.add(frame_right)
        model.add(Rigid_connection2D(frame_right, beam, r_OB2, frame_ID2=(1,)))

        __g = np.array([0, - A_rho0 * 9.81, 0])
        if statics:
            f_g_beam = Line_force(lambda xi, t: t * __g, beam)
        else:
            f_g_beam = Line_force(lambda xi, t: __g, beam)
        model.add(f_g_beam)

        model.assemble()

        if statics:
            solver = Newton(model, n_load_steps=20, max_iter=20, tol=1.0e-6, numerical_jacobian=False)
        else:
            solver = Euler_backward(model, t1, dt, newton_max_iter=50, numerical_jacobian=False)
            # solver = Generalized_alpha_1(model, t1, dt, variable_dt=False, rho_inf=0.5)
        sols.append( solver.solve() )

    if statics:
        fig, ax = plt.subplots()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_xlim([-0.2, L])
        ax.set_ylim([-L, 0.2])
        ax.grid(linestyle='-', linewidth='0.5')

        x, y, z = bernoulli.centerline(sols[0].q[-1]).T
        ax.plot(x, y, '--k')

        x, y, z = inextensible_bernoulli.centerline(sols[1].q[-1]).T
        ax.plot(x, y, '-b')

        ax.plot(*sols[0].q[-1].reshape(2, -1), '--ok')
        ax.plot(*sols[1].q[-1].reshape(2, -1), '--ob')

        plt.show()
    else:
        # animate configurations
        fig, ax = plt.subplots()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_xlim([-0.1*L, 1.1*L])
        ax.set_ylim([-1.1*L, 0.1*L])
        ax.grid(linestyle='-', linewidth='0.5')

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
        q1 = sols[1].q[::frac]
        
        center_line0, = ax.plot([], [], '-k')
        center_line1, = ax.plot([], [], '--b')

        def animate(i):
            x, y, _ = beam.centerline(q0[i], n=50).T
            center_line0.set_data(x, y)

            x, y, _ = beam.centerline(q1[i], n=50).T
            center_line1.set_data(x, y)

            return center_line0, center_line1

        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
        plt.show()

def cantilever():
    t1 = 200
    dt = 5e-1

    L = 2 * np.pi
    EA = 5
    EI = 2
    material_model = Hooke(EA, EI)
    A_rho0 = 0.1
    alpha2 = 10

    r_OB1 = np.zeros(3)
    frame_left = Frame(r_OP=r_OB1)

    # discretization properties
    p = 3
    assert p >= 2
    nQP = int(np.ceil((p + 1)**2 / 2))
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
    # q0 = np.hstack((Y0, -X0))
    u0 = np.zeros_like(Q)

    bernoulli = Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)
    inextensible_bernoulli = Inextensible_Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)
    # beams = [bernoulli, inextensible_bernoulli]
    beams = [inextensible_bernoulli, bernoulli]

    sols = []
    for beam in beams:
        model = Model()
        model.add(beam)
        model.add(frame_left)
        model.add(Rigid_connection2D(frame_left, beam, r_OB1, frame_ID2=(0,)))
        # model.add(Spherical_joint2D(frame_left, beam, r_OB1, frame_ID2=(0,), la_g0=np.ones(2)*1.0e-6))

        if statics:
            F = lambda t: t * np.array([0, -EI * alpha2 / L**2, 0])
        else:
            F = lambda t: min(t, 10) / 10 * np.array([0, -EI * alpha2 / L**2, 0])
                
        model.add(Force(F, beam, frame_ID=(1,)))
        model.assemble()

        if statics:
            solver = Newton(model, n_load_stepts=20, max_iter=20, tol=1.0e-6, numerical_jacobian=False)
        else:
            solver = Euler_backward(model, t1, dt, newton_max_iter=50, numerical_jacobian=False)
            # solver = Generalized_alpha_1(model, t1, dt, variable_dt=False, rho_inf=0.5)
        sols.append( solver.solve() )

    if statics:
        fig, ax = plt.subplots()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_xlim([-0.2, L])
        ax.set_ylim([-L, 0.2])
        ax.grid(linestyle='-', linewidth='0.5')

        x, y, z = bernoulli.centerline(sols[0].q[-1]).T
        ax.plot(x, y, '--k')

        x, y, z = inextensible_bernoulli.centerline(sols[1].q[-1]).T
        ax.plot(x, y, '-b')

        ax.plot(*sols[0].q[-1].reshape(2, -1), '--ok')
        ax.plot(*sols[1].q[-1].reshape(2, -1), '--ob')

        plt.show()
    else:
        # animate configurations
        fig, ax = plt.subplots()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_xlim([-0.1*L, 1.1*L])
        ax.set_ylim([-1.1*L, 0.1*L])
        ax.grid(linestyle='-', linewidth='0.5')

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
        q1 = sols[1].q[::frac]
        
        center_line0, = ax.plot([], [], '-k')
        center_line1, = ax.plot([], [], '--b')

        def animate(i):
            x, y, _ = beam.centerline(q0[i], n=50).T
            center_line0.set_data(x, y)

            x, y, _ = beam.centerline(q1[i], n=50).T
            center_line1.set_data(x, y)

            return center_line0, center_line1

        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
        plt.show()

if __name__ == "__main__":
    inextensible_rope()