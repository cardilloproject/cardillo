from cardillo.model.classical_beams.spatial import Hooke_quadratic
from cardillo.model.classical_beams.spatial import Timoshenko_director_dirac
from cardillo.model.classical_beams.spatial import Timoshenko_director_integral, Euler_Bernoulli_director_integral, Inextensible_Euler_Bernoulli_director_integral
from cardillo.model.classical_beams.spatial.director import straight_configuration
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import Rigid_connection
from cardillo.model import Model
from cardillo.solver import Newton
from cardillo.model.line_force.line_force import Line_force
from cardillo.discretization import uniform_knot_vector
from cardillo.model.moment import K_Moment
from cardillo.model.force import Force

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import numpy as np

if __name__ == "__main__":
    # L = 1000
    # rho = 100
    # r = L / rho
    # A = r**2
    # E = 1.0
    # G = 0.5
    # Ei = np.array([E, G, G]) * A
    # I_12 = r**4 / 12
    # I_T = r**4 / 6
    # Fi = np.array([G * I_T, E * I_12, E * I_12])
    # A_rho0 = 0.0
    # B_rho0 = np.zeros(3)
    # C_rho0 = np.zeros((3, 3))

    # physical properties of the beam
    A_rho0 = 1
    B_rho0 = np.ones(3)
    C_rho0 = np.eye(3)

    L = 2 * np.pi
    Ei = np.array([10e-1, 3e-1, 1e-1])
    Fi = np.array([5, 1, 1])

    material_model = Hooke_quadratic(Ei, Fi)

    # junction at the origin
    r_OB1 = np.zeros(3)
    frame_left = Frame(r_OP=r_OB1)

    # discretization properties
    p = 3
    # nQP = int(np.ceil((p + 1)**2 / 2))
    nQP = p + 1
    print(f'nQP: {nQP}')
    nEl = 10

    # build reference configuration
    Q = straight_configuration(p, nEl, L)
    q0 = Q.copy()
    nn = nEl + p
    # q0[nn + 1] += 1.0e-3
    # q0[2 * nn + 1] += 1.0e-3
    # q0[] + np.random.rand(len(Q)) * 1.0e-4
    # la_g0 = np.ones(6 * nn) * 1.0e-3
    # beam = Timoshenko_director_dirac(material_model, A_rho0, B_rho0, C_rho0, p, nQP, nEl, Q=Q, q0=q0)
    # la_g0 = np.ones(9 * nn) * 1.0e-3
    # beam = Timoshenko_director_integral(material_model, A_rho0, B_rho0, C_rho0, p, nQP, nEl, Q=Q)
    # beam = Euler_Bernoulli_director_integral(material_model, A_rho0, B_rho0, C_rho0, p, nQP, nEl, Q=Q)
    beam = Inextensible_Euler_Bernoulli_director_integral(material_model, A_rho0, B_rho0, C_rho0, p, nQP, nEl, Q=Q, q0=q0)
    # exit()

    # left joint
    joint_left = Rigid_connection(frame_left, beam, r_OB1, frame_ID2=(0,))

    # gravity beam
    __g = np.array([0, 0, - A_rho0 * 9.81 * 1.0e-3])
    f_g_beam = Line_force(lambda xi, t: t * __g, beam)

    # wrench at right end
    # M = lambda t: -np.array([1, 0, 0]) * t * 2 * np.pi * Fi[0] / L * 0.25
    M = lambda t: -np.array([0, 1, 0]) * t * 2 * np.pi * Fi[1] / L * 0.25
    # M = lambda t: -np.array([0, 0, 1]) * t * 2 * np.pi * Fi[2] / L * 0.5
    # M = lambda t: -np.array([1, 1, 0]) * t * 2 * np.pi * Fi[1] / L * 0.5
    moment = K_Moment(M, beam, (1,))

    # force at right end
    F = lambda t: np.array([0, 0, -1]) * t * 1.0e-1
    force = Force(F, beam, frame_ID=(1,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame_left)
    model.add(joint_left)
    model.add(f_g_beam)
    model.add(force)
    model.add(moment)
    model.assemble()

    solver = Newton(model, n_load_steps=10, max_iter=20, tol=1.0e-8, numerical_jacobian=False)
    # solver = Newton(model, n_load_steps=10, max_iter=10, numerical_jacobian=True)
    sol = solver.solve()
    t = sol.t
    q = sol.q

    # vtk export
    beam.post_processing(t, q, 'director_beam')

    exit()

    ###############
    # visualization
    ###############
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    scale = 1.2 * L
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=-scale, top=scale)

    beam.plot_centerline(ax, q[0], color='black')
    # beam.plot_frames(ax, q[0], n=4, length=0.5)
    beam.plot_centerline(ax, q[-1], color='blue')
    beam.plot_frames(ax, q[-1], n=10, length=1)

    plt.show()


    # ###############
    # # visualization
    # ###############
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    # ax.set_zlabel('z [m]')
    # scale = 1.2 * L
    # ax.set_xlim3d(left=-scale, right=scale)
    # ax.set_ylim3d(bottom=-scale, top=scale)
    # ax.set_zlim3d(bottom=-scale, top=scale)

    # beam.plot_centerline(ax, model.q0)
    # beam.plot_frames(ax, model.q0, n=4, length=0.5)

    # plt.show()
    # exit()