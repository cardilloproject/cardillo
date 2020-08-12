from cardillo.model.classical_beams.planar import Hooke, Euler_bernoulli, Inextensible_Euler_bernoulli
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

# statics = True
statics = False
animate = False

if __name__ == "__main__":
    # physical properties of the rope
    L = 2 * np.pi
    EA = 5
    EI = 2
    material_model = Hooke(EA, EI)
    A_rho0 = 0

    r_OB1 = np.zeros(3)
    frame_left = Frame(r_OP=r_OB1)

    # discretization properties
    p = 3
    assert p >= 2
    nQP = int(np.ceil((p + 1)**2 / 2))
    print(f'nQP: {nQP}')
    nEl = 10

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
    beam = Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)
    # beam = Inextensible_Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)

    # left joint
    joint_left = Rigid_connection2D(frame_left, beam, r_OB1, frame_ID2=(0,))

    # wrench at right end
    F = lambda t: t * np.array([0, -EI / L**2, 0])
    force = Force(F, beam, frame_ID=(1,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame_left)
    model.add(joint_left)
    model.add(force)
    model.assemble()

    solver = Newton(model, n_load_steps=10, max_iter=20, tol=1.0e-6, numerical_jacobian=False)
    # solver = Newton(model, n_load_steps=50, max_iter=10, numerical_jacobian=True)
    sol = solver.solve()
    t = sol.t
    q = sol.q

    x, y, z = beam.centerline(q[-1]).T
    plt.plot(x, y, '-k')
    plt.plot(*q[-1].reshape(2, -1), '--ob')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')

    plt.show()