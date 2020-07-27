from cardillo.model.classical_beams.spatial import Hooke_quadratic, Timoshenko_beam_director
from cardillo.model.classical_beams.spatial.timoshenko_beam_director import straight_configuration
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import Rigid_connection
from cardillo.model import Model
from cardillo.solver import Newton
from cardillo.model.line_force.line_force import Line_force
from cardillo.discretization import uniform_knot_vector
from cardillo.model.moment import K_Moment

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import numpy as np

if __name__ == "__main__":
    # physical properties of the beam
    A_rho0 = 0
    B_rho0 = np.zeros(3)
    C_rho0 = np.zeros((3, 3))

    L = 2 * np.pi
    Ei = np.array([5, 1, 1])
    Fi = np.array([5, 1, 1])
    material_model = Hooke_quadratic(Ei, Fi)

    # junction at the origin
    r_OB1 = np.zeros(3)
    frame_left = Frame(r_OP=r_OB1)

    # discretization properties
    p = 3
    nQP = int(np.ceil((p + 1)**2 / 2))
    print(f'nQP: {nQP}')
    nEl = 10

    # build reference configuration
    Q = straight_configuration(p, nEl, L)
    beam = Timoshenko_beam_director(material_model, A_rho0, B_rho0, C_rho0, p, nEl, nQP, Q=Q)

    # left joint
    joint_left = Rigid_connection(frame_left, beam, r_OB1, frame_ID2=(0,))

    # # gravity beam
    # __g = np.array([0, - A_rho0 * 9.81 * 8.0e-4, 0])
    # f_g_beam = Line_force(lambda xi, t: t * __g, beam)

    # wrench at right end
    M = lambda t: np.array([0, 1, 0]) * t * 2 * np.pi * Ei[1] / L
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

    exit()

    # solver = Newton(model, n_load_stepts=10, max_iter=20, tol=1.0e-6, numerical_jacobian=False)
    # # solver = Newton(model, n_load_stepts=50, max_iter=10, numerical_jacobian=True)
    # sol = solver.solve()
    # t = sol.t
    # q = sol.q

    # x, y, z = beam.centerline(q[-1]).T
    # plt.plot(x, y, '-k')
    # plt.plot(*q[-1].reshape(2, -1), '--ob')
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.axis('equal')

    # plt.show()