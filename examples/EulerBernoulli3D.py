from cardillo.beams.spatial.material_models import Simo1986
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import RigidConnection
from cardillo.beams import (
    animate_beam,
    EulerBernoulli,
)
from cardillo.forces import Force, K_Moment, DistributedForce1D
from cardillo.model import Model
from cardillo.solver import Newton
from cardillo.contacts import Point2Plane

import numpy as np

if __name__ == "__main__":
    # physical properties of the beam
    A_rho0 = 1
    B_rho0 = np.zeros(3)
    C_rho0 = np.eye(3)

    L = 2 * np.pi
    Ei = np.ones(3) * 1.0e0
    Fi = np.ones(3) * 1.0e-1

    # note: we abuse this material model for an Euler-Bernoulli beam by 
    # setting Gamma = [lambda, 0, 0]!
    # TODO: Change this later.
    material_model = Simo1986(Ei, Fi)

    # junction at the origin
    r_OB1 = np.zeros(3)
    frame_left = Frame(r_OP=r_OB1)

    # discretization properties
    p = 3
    p_r = p
    p_phi = p
    # nQP = int(np.ceil((p + 1)**2 / 2))
    nQP = p + 1
    print(f"nQP: {nQP}")
    nEl = 10

    basis = "B-spline"
    # basis = 'lagrange'

    # build reference configuration
    Q = EulerBernoulli.straight_configuration(
        p_r, p_phi, nEl, L
    )
    q0 = Q.copy()

    beam = EulerBernoulli(
        material_model,
        A_rho0,
        B_rho0,
        C_rho0,
        p_r,
        p_phi,
        nQP,
        nEl,
        Q=Q,
        q0=q0,
    )

    # left joint
    joint_frame_beam = RigidConnection(frame_left, beam, r_OB1, frame_ID2=(0,))

    # gravity beam
    __g = np.array([0, 0, -A_rho0 * 9.81 * 5.0e-3])
    f_g_beam = DistributedForce1D(lambda t, xi: t * __g, beam)

    # moment at right end
    M = lambda t: -np.array([1, 0, 1]) * t * 2 * np.pi * Fi[1] / L * 0.5
    moment = K_Moment(M, beam, (1,))

    # force at right end
    F = lambda t: np.array([0, 0, -1]) * t * 1.0e-2
    force = Force(F, beam, frame_ID=(1,))

    # add point to plane contact
    r_OP_contact = np.array([L, 0, -0.0 * L])
    frame_contact = Frame(r_OP=r_OP_contact)
    prox_r_N = 1.0e-3
    e_N = 0
    contact = Point2Plane(frame_contact, beam, prox_r_N, e_N, frame_ID=(1.0,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame_left)
    model.add(joint_frame_beam)
    model.add(f_g_beam)
    model.add(frame_contact)
    model.add(contact)
    # model.add(moment)
    # model.add(force)
    model.assemble()

    solver = Newton(
        model,
        n_load_steps=10,
        max_iter=20,
        atol=1.0e-8,
        numerical_jacobian=False,
        prox_r_N=1.0e-3,
    )

    sol = solver.solve()
    t = sol.t
    q = sol.q

    ###########
    # animation
    ###########
    animate_beam(t, q, beam, L, show=True)
