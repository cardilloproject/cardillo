from cardillo.math import A_IK_basic
from cardillo.model.frame import Frame
from cardillo.beams import (
    EulerBernoulli2D,
    animate_beam,
)
from cardillo.beams.planar import Hooke
from cardillo.model.bilateral_constraints.implicit import RigidConnection2D
from cardillo.forces import Force, K_Moment
from cardillo.model import System
from cardillo.solver import Newton

from cardillo.model.scalar_force_interactions.force_laws import Linear_spring
from cardillo.model.scalar_force_interactions.translational_f_pot import (
    Translational_f_pot,
)

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
    print(f"nQP: {nQP}")
    nEl = 5

    # position and orientation of the rigid connection
    r_OP = np.zeros(3)  # origin at (0, 0, 0)
    A_IK = A_IK_basic(pi / 2).z()  # rotate beam and rigid connection by 90° clock-wise

    # build reference configuration
    # Q = straight_configuration_EulerBernoulli2D(p, nEl, L, r_OP=r_OP, A_IK=A_IK)
    Q = EulerBernoulli2D.straight_configuration(p, nEl, L, r_OP=r_OP, A_IK=A_IK)

    # build beam model
    beam = EulerBernoulli2D(A_rho0, material_model, p, nEl, nQP, Q=Q)

    # rigid connection at the bottom end of the beam (xi = 0)
    frame = Frame(r_OP=r_OP, A_IK=A_IK)
    joint = RigidConnection2D(frame, beam, r_OP, frame_ID2=(0,))

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
    model = System()
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
    animate_beam(t, q, beam, L, show=True)
