from cardillo.math import A_IK_basic
from cardillo.model.frame import Frame
from cardillo.model.classical_beams.planar import Hooke, EulerBernoulli, straight_configuration
from cardillo.model.bilateral_constraints.implicit import Rigid_connection2D
from cardillo.model.force import Force
from cardillo.model.moment import K_Moment
from cardillo.model import Model
from cardillo.solver import Newton

from cardillo.model.scalar_force_interactions.force_laws import Linear_spring
from cardillo.model.scalar_force_interactions.translational_f_pot import Translational_f_pot

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import pi

# statics = True
statics = False
animate = True

if __name__ == "__main__":
    # physical properties of the planar beam model
    # TODO: get correct material data and geometrical properties
    L = 2 * np.pi
    EA = 5
    EI = 2
    material_model = Hooke(EA, EI)
    A_rho0 = 0

    # discretization properties
    p = 3
    assert p >= 2
    # nQP = int(np.ceil((p + 1)**2 / 2))
    nQP = p + 1
    print(f'nQP: {nQP}')
    nEl = 5
    
    # position and orientation of the rigid connection
    r_OP = np.zeros(3) # origin at (0, 0, 0)
    A_IK = A_IK_basic(pi / 2).z() # rotate beam and rigid connection by 90° clock-wise

    # build reference configuration
    Q = straight_configuration(p, nEl, L, r_OP=r_OP, A_IK=A_IK)

    # build beam model
    beam = EulerBernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q)

    # rigid connection at the bottom end of the beam (xi = 0)
    frame = Frame(r_OP=r_OP, A_IK=A_IK)
    joint = Rigid_connection2D(frame, beam, r_OP, frame_ID2=(0,))

    # force at top end of the beam (xi = 1)
    # F = lambda t: t * np.array([EI * 0.1, -EI / L**2, 0])
    F = lambda t: t * np.array([EI * 0.25, 0, 0])
    force = Force(F, beam, frame_ID=(1,))

    # moment at the top end of the beam (xi = 1)
    M = lambda t: t * np.array([0, 0, EI / L * pi * 4 / 4])
    moment = K_Moment(M, beam, frame_ID=(1,))

    # linear force element modeling the DE's or a tendon
    # TODO: these force elements are not implemented yet; we use a spring instead for now
    k = 1.0e3
    linear_spring = Linear_spring(k)
    spring_element = Translational_f_pot(linear_spring, frame, beam, frame_ID2=(1,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame)
    model.add(joint)
    # model.add(force)
    model.add(moment)
    # model.add(spring_element)
    model.assemble()

    # solver options and solve the static system
    n_load_steps = 10
    max_iter = 20
    atol = 1.0e-8
    solver = Newton(model, n_load_steps=n_load_steps, max_iter=max_iter, atol=atol)
    sol = solver.solve()
    t = sol.t
    q = sol.q

    # # vtk export
    # beam.post_processing(sol.t, sol.q, 'Bernoulli2D-cantilever')

    ##############################
    # visualize static deformation
    ##############################
    fig, ax = plt.subplots()

    # draw ground
    l = 1
    h = 0.5
    ax.plot([-l, l], [0, 0], '-k', lw=1)
    p = patches.Rectangle((-l, -h), 2 * l, h, linewidth=0, fill=None, hatch='///')
    ax.add_patch(p)

    # plot the deformed beam centerline and the control polygon of the B-splines
    ax.plot(*beam.nodes(q[-1]), '--ob')
    x, y, z = beam.centerline(q[-1]).T
    ax.plot(x, y, '-k')
    
    # set labels and scale axis equal
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.axis('equal')

    plt.show()