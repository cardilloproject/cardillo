from scipy.integrate._ivp.ivp import solve_ivp
from cardillo.model.classical_beams.planar import Hooke, Euler_bernoulli, Inextensible_Euler_bernoulli
from cardillo.model.rope import Rope, Inextensible_Rope
from cardillo.model.rope import Hooke as Hooke_rope
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import Rigid_connection2D, Spherical_joint2D, Linear_guidance_x_2D, Linear_guidance_xyz_2D
from cardillo.model.scalar_force_interactions.force_laws import Linear_spring
from cardillo.model.scalar_force_interactions.translational_f_pot import Translational_f_pot

from cardillo.model import Model
from cardillo.solver import Newton, Euler_backward, Generalized_alpha_1, Scipy_ivp
from cardillo.discretization import uniform_knot_vector
from cardillo.model.force import Force
from cardillo.model.line_force import Line_force

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

statics = True
# statics = False

def inextensible_rope():
    t1 = 5
    dt = 2e-1

    # beam parameters
    L = 5
    EA = 5 * 1.0e1
    EI = 2 * 1.0e-1
    material_model = Hooke(EA, EI)
    A_rho0 = 0.1

    # linear spring parameters
    k = 1e2
    l0 = 0.5 * L

    r_OB1 = np.zeros(3)
    frame_left = Frame(r_OP=r_OB1)
    r_OB2 = np.array([L+l0, 0, 0])
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
    u0 = np.zeros_like(Q)

    bernoulli = Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)
    inextensible_bernoulli = Inextensible_Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)

    beams = [bernoulli, inextensible_bernoulli]

    sols = []
    for beam in beams:
        model = Model()
        model.add(beam)
        model.add(frame_left)
        # model.add(Rigid_connection2D(frame_left, beam, r_OB1, frame_ID2=(0,)))
        model.add(Spherical_joint2D(frame_left, beam, r_OB1, frame_ID2=(0,)))
        
        model.add(frame_right)
        # model.add(Spherical_joint2D(frame_right, beam, r_OB2, frame_ID2=(1,)))
        # model.add(Linear_guidance_xyz_2D(frame_right, beam, r_OB2, frame_right.A_IK(0), frame_ID2=(1,)))
        # model.add(Linear_guidance_x_2D(beam, frame_right, r_OB2, frame_right.A_IK(0), frame_ID1=(1,)))
        model.add(Linear_guidance_xyz_2D(beam, frame_right, r_OB2, frame_right.A_IK(0), frame_ID1=(1,)))

        model.add(Translational_f_pot(Linear_spring(k, l0), beam, frame_right, frame_ID1=(1,)))

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
            # solver = Scipy_ivp(model, t1, dt, atol=1e-4)
            # solver = Generalized_alpha_1(model, t1, dt, variable_dt=False, rho_inf=0.5)
        sols.append( solver.solve() )

    if statics:
        material_model = Hooke(EA, EI * 1e-5)
        inextensible_bernoulli_precurved = Inextensible_Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=sols[1].q[-1], u0=u0)
        # material_model_rope = Hooke_rope(EA)
        # rope_precurved = Inextensible_Rope(A_rho0, material_model_rope, p, nEl, nQP, Q=Q, q0=sols[1].q[-1], u0=u0, dim=2, la_g0 = sols[1].la_g[-1, :-3])

        beam = inextensible_bernoulli_precurved
        # beam = rope_precurved
        # r_OB2 = lambda t: np.array([sols[1].q[-1, nNd-1], 0, 0])
        r_OB2 = lambda t: np.array([sols[1].q[-1, nNd-1] - t * L / 2, -t * L / 2, 0])
        frame_right = Frame(r_OP=r_OB2)

        model2 = Model()
        model2.add(beam)
        model2.add(frame_left)
        # model.add(Rigid_connection2D(frame_left, beam, r_OB1, frame_ID2=(0,)))
        model2.add(Spherical_joint2D(frame_left, beam, r_OB1, frame_ID2=(0,)))

        model2.add(frame_right)
        model2.add(Spherical_joint2D(frame_right, beam, r_OB2(0), frame_ID2=(1,)))
        # model2.add(Spherical_joint2D(frame_right, beam, r_OB2(0), frame_ID2=(1,), la_g0=-sols[1].la_g[-1, -3:-1]))

        __g = np.array([0, - A_rho0 * 9.81, EI * 1e-6])
        f_g_beam = Line_force(lambda xi, t: __g, beam)

        model2.add(f_g_beam)
        
        model2.assemble()

        solver = Newton(model2, n_load_steps=50, max_iter=50, tol=1.0e-6, numerical_jacobian=False)

        sols.append( solver.solve() )

        material_model = Hooke(EA, EI * 0)
        inextensible_bernoulli_precurved = Inextensible_Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=sols[2].q[-1], u0=u0)
        # material_model_rope = Hooke_rope(EA)
        # rope_precurved = Inextensible_Rope(A_rho0, material_model_rope, p, nEl, nQP, Q=Q, q0=sols[1].q[-1], u0=u0, dim=2, la_g0 = sols[1].la_g[-1, :-3])

        beam = inextensible_bernoulli_precurved
        # beam = rope_precurved
        # r_OB2 = lambda t: np.array([sols[1].q[-1, nNd-1], 0, 0])
        r_OB2 = lambda t: np.array([sols[1].q[-1, nNd-1] - t * L / 2, -t * L / 2, 0])
        frame_right = Frame(r_OP=r_OB2(1))

        model3 = Model()
        model3.add(beam)
        model3.add(frame_left)
        # model.add(Rigid_connection2D(frame_left, beam, r_OB1, frame_ID2=(0,)))
        model3.add(Spherical_joint2D(frame_left, beam, r_OB1, frame_ID2=(0,)))

        model3.add(frame_right)
        model3.add(Spherical_joint2D(frame_right, beam, r_OB2(0), frame_ID2=(1,)))
        # model2.add(Spherical_joint2D(frame_right, beam, r_OB2(0), frame_ID2=(1,), la_g0=-sols[1].la_g[-1, -3:-1]))

        __g = np.array([0, - A_rho0 * 9.81, 0])
        f_g_beam = Line_force(lambda xi, t: __g, beam)

        model3.add(f_g_beam)
        
        model3.assemble()

        solver = Newton(model3, n_load_steps=50, max_iter=50, tol=1.0e-6, numerical_jacobian=False)

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
        ax.plot(*sols[2].q[-1].reshape(2, -1), '--or')
        ax.plot(*sols[3].q[-1].reshape(2, -1), '--xg')

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
            x, y, _ = beam.centerline(q0[i], n=500).T
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

if __name__ == "__main__":
    inextensible_rope()
    # cantilever()