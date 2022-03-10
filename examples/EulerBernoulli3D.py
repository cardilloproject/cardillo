from cardillo.beams.spatial.material_models import ShearStiffQuadratic
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import RigidConnection, SphericalJoint
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
    E1 = 1.0e0
    Fi = np.ones(3) * 1.0e-1

    material_model = ShearStiffQuadratic(E1, Fi)

    # junctions
    r_OB1 = np.zeros(3)
    r_OB2 = np.array([L, 0, 0])
    frame1 = Frame(r_OP=r_OB1)
    frame2 = Frame(r_OP=r_OB2)

    # discretization properties
    p = 2
    p_r = p
    p_phi = p
    # nQP = int(np.ceil((p + 1)**2 / 2))
    nQP = p + 1
    print(f"nQP: {nQP}")
    nEl = 2

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

    # ############################################
    # # dummy values for debugging internal forces
    # ############################################
    # # assemble the model
    # model = Model()
    # model.add(beam)
    # model.assemble()

    # t = 0
    # q = np.random.rand(model.nq)

    # E_pot = model.E_pot(t, q)
    # print(f"E_pot: {E_pot}")
    # f_pot = model.f_pot(t, q)
    # print(f"f_pot:\n{f_pot}")
    # # f_pot_q = model.f_pot_q(t, q)
    # # print(f"f_pot_q:\n{f_pot_q}")
    # exit()

    # left and right joint
    # joint1 = RigidConnection(frame1, beam, r_OB1, frame_ID2=(0,))
    # joint2 = RigidConnection(frame2, beam, r_OB2, frame_ID2=(1,))
    joint1 = SphericalJoint(frame1, beam, r_OB1, frame_ID2=(0,))
    joint2 = SphericalJoint(frame2, beam, r_OB2, frame_ID2=(1,))

    # gravity beam
    __g = np.array([0, 0, -A_rho0 * 9.81 * 5.0e-3])
    f_g_beam = DistributedForce1D(lambda t, xi: t * __g, beam)

    # # moment at right end
    # M = lambda t: -np.array([1, 0, 1]) * t * 2 * np.pi * Fi[1] / L * 0.5
    # moment = K_Moment(M, beam, (1,))

    # # force at right end
    # F = lambda t: np.array([0, 0, -1]) * t * 1.0e-2
    # force = Force(F, beam, frame_ID=(1,))

    # # add point to plane contact
    # r_OP_contact = np.array([L, 0, -0.0 * L])
    # frame_contact = Frame(r_OP=r_OP_contact)
    # prox_r_N = 1.0e-3
    # e_N = 0
    # contact = Point2Plane(frame_contact, beam, prox_r_N, e_N, frame_ID=(1.0,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    model.add(frame2)
    model.add(joint2)
    model.add(f_g_beam)
    # model.add(frame_contact)
    # model.add(contact)
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
