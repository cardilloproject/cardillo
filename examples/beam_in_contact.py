from cardillo.model.classical_beams.planar import Hooke, Euler_bernoulli2D
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import Spherical_joint2D, Spherical_joint
from cardillo.model import Model
from cardillo.solver import Generalized_alpha_1, Generalized_alpha_2
from cardillo.model.line_force.line_force import Line_force
from cardillo.discretization import uniform_knot_vector
from cardillo.model.point_mass import Point_mass
from cardillo.model.force import Force
from cardillo.math.algebra import A_IK_basic_z
from cardillo.model.contacts import Sphere_to_plane, Sphere_to_plane2D

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import numpy as np

# statics = True
statics = False

if __name__ == "__main__":
    # solver parameter
    t0 = 0
    t1 = 2
    dt = 1e-3

    # physical properties of the rope
    rho = 7850
    L = 10
    r = 1e-2
    A = np.pi * r**2
    I = A / 4 * r**2
    E = 210e8
    EA = E * A * 1e-3
    EI = E * I * 50
    material_model = Hooke(EA, EI)
    A_rho0 = A * rho

    amplitude = 0
    e = lambda t: amplitude * t - L/2
    e_t = lambda t: amplitude
    e_tt = lambda t: 0

    r_OB1 = lambda t: np.array([e(t), 0.1 * L, 0])
    r_OB1_t = lambda t: np.array([e_t(t), 0, 0])
    r_OB1_tt = lambda t: np.array([e_tt(t), 0, 0])
    frame_left = Frame(r_OP=r_OB1, r_OP_t=r_OB1_t, r_OP_tt=r_OB1_tt)

    # discretization properties
    p = 3
    assert p >= 2
    nQP = int(np.ceil((p + 1)**2 / 2))
    print(f'nQP: {nQP}')
    nEl = 5

    # build reference configuration
    nNd = nEl + p
    X0 = np.linspace(0, L, nNd)
    Xi = uniform_knot_vector(p, nEl)
    for i in range(nNd):
        X0[i] = np.sum(Xi[i+1:i+p+1])
    X0 = X0 * L / p
    Y0 = np.zeros_like(X0)
    q0 = np.hstack((X0 + r_OB1(0)[0], Y0 + r_OB1(0)[1]))
    # Y0 = X0 * L / p
    # X0 = np.zeros_like(Y0)
    # q0 = np.hstack((X0-L, Y0 - 0.9 * L))
    Q = np.hstack((X0, Y0))

    # ux0 = np.ones_like(Y0) * 3
    # uy0 = np.zeros_like(Y0)
    # u0 = np.hstack((ux0, uy0))

    u0 = np.zeros_like(Q)
    # u0[0] = r_OB1_t(0)[0]
    # u0[nNd] = r_OB1_t(0)[1]

    beam = Euler_bernoulli2D(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)

    # left joint
    joint_left = Spherical_joint2D(frame_left, beam, r_OB1(0), frame_ID2=(0,))

    # right joint
    r_OB2 = np.array([L, 0, 0])
    frame_right = Frame(r_OP=r_OB2)
    joint_right = Spherical_joint2D(beam, frame_right, r_OB2, frame_ID1=(1,))

    # gravity beam
    __g = np.array([0, - A_rho0 * 9.81, 0])
    f_g_beam = Line_force(lambda xi, t: __g, beam)

    # rigid body
    m = 5
    PM = Point_mass(m, q0=r_OB2)

    # gravity rigid body
    f_g_RB = Force(lambda t: np.array([0, - m * 9.81, 0]), PM)

    # joint = Rigid_connection(beam, RB, r_OB2, frame_ID1=(1,))
    joint = Spherical_joint(beam, PM, r_OB2, frame_ID1=(1,))
    
    alpha = np.pi / 4 * 0
    e1, e2, e3 = A_IK_basic_z(alpha)
    r_OP_frame = np.array([0, 0, 0])
    # frame = Frame(r_OP=r_OP_frame, A_IK=np.vstack( (e3, e1, e2) ).T )
    frame = Frame(r_OP=r_OP_frame, A_IK=A_IK_basic_z(alpha))
    mu = 0.1
    r_N = 0.1
    e_N = 0
    # plane = Sphere_to_plane(frame, PM, 0, mu, prox_r_N=r_N, prox_r_T=r_N, e_N=e_N)
    # plane = Sphere_to_plane2D(frame, PM, 0, mu, prox_r_N=r_N, prox_r_T=r_N, e_N=e_N)
    contact0 = Sphere_to_plane2D(frame, beam, 0, mu, prox_r_N=r_N, prox_r_T=r_N, e_N=e_N, frame_ID=(0,))
    contact1 = Sphere_to_plane2D(frame, beam, 0, mu, prox_r_N=r_N, prox_r_T=r_N, e_N=e_N, frame_ID=(1,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame_left)
    model.add(joint_left)
    model.add(f_g_beam)
    # model.add(contact0)
    model.add(contact1)
    model.assemble()
    
    solver = Generalized_alpha_2(model, t1, dt, rho_inf=0.5, newton_tol=1.0e-6)
    sol = solver.solve()

    t = sol.t
    q = sol.q

    # animate configurations
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    fig, ax = plt.subplots()
    ax.axis('equal')
    
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    # ax.set_zlabel('z [m]')
    # scale = L
    # ax.set_xlim3d(left=0, right=L)
    # ax.set_ylim3d(bottom=-L/2, top=L/2)
    # ax.set_zlim3d(bottom=-L/2, top=L/2)

    # prepare data for animation
    frames = len(t)
    target_frames = min(len(t), 100)
    frac = int(frames / target_frames)
    animation_time = 1
    interval = animation_time * 1000 / target_frames

    frames = target_frames
    t = t[::frac]
    q = q[::frac]
    
    ax.plot(np.array([- L, L]), np.ones(2) * r_OP_frame[1], '-k')

    x1, y1, z1 = beam.centerline(q[-1]).T
    center_line, = ax.plot(x1, y1, '-b')
    # x1, y1 = q[-1].reshape(2, -1)
    # nodes, = ax.plot(x1, y1, 'ro')

    # x_S, y_S, z_S = PM.r_OP(t, q[0][PM.qDOF])
    # A_IK = RB.A_IK(t, q[0][RB.qDOF])
    # d1 = A_IK[:, 0] * L / 4
    # d2 = A_IK[:, 1] * L / 4
    # d3 = A_IK[:, 2] * L / 4

    # COM, = ax.plot([x_S], [y_S], [z_S], 'ok')
    # d1_, = ax.plot([x_S, x_S + d1[0]], [y_S, y_S + d1[1]], [z_S, z_S + d1[2]], '-r')
    # d2_, = ax.plot([x_S, x_S + d2[0]], [y_S, y_S + d2[1]], [z_S, z_S + d2[2]], '-g')
    # d3_, = ax.plot([x_S, x_S + d3[0]], [y_S, y_S + d3[1]], [z_S, z_S + d3[2]], '-b')

    def update(t, q):
        x, y, z = beam.centerline(q).T
        center_line.set_data(x, y)
        # x, y = q.reshape(2, -1)
        # nodes.set_data(x, y)
        # center_line.set_3d_properties(z)

        # x_S, y_S, z_S = PM.r_OP(t, q[PM.qDOF])
        # A_IK = PM.A_IK(t, q[PM.qDOF])
        # d1 = A_IK[:, 0] * L / 4
        # d2 = A_IK[:, 1] * L / 4
        # d3 = A_IK[:, 2] * L / 4

        # COM.set_data(np.array([x_S]), np.array([y_S]))
        # COM.set_3d_properties(np.array([z_S]))

        # d1_.set_data(np.array([x_S, x_S + d1[0]]), np.array([y_S, y_S + d1[1]]))
        # d1_.set_3d_properties(np.array([z_S, z_S + d1[2]]))

        # d2_.set_data(np.array([x_S, x_S + d2[0]]), np.array([y_S, y_S + d2[1]]))
        # d2_.set_3d_properties(np.array([z_S, z_S + d2[2]]))

        # d3_.set_data(np.array([x_S, x_S + d3[0]]), np.array([y_S, y_S + d3[1]]))
        # d3_.set_3d_properties(np.array([z_S, z_S + d3[2]]))

        return center_line, #nodes, #COM, #d1_, d2_, d3_

    def animate(i):
        update(t[i], q[i])

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
    plt.show()