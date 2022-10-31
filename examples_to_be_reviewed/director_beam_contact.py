from cardillo.beams.spatial.material_models import Simo1986
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import RigidConnection
from cardillo.beams import (
    animate_beam,
    TimoshenkoDirectorIntegral,
)
from cardillo.forces import Force, K_Moment, DistributedForce1DBeam
from cardillo.model import System
from cardillo.solver import Newton
from cardillo.contacts import Point2Plane

import numpy as np
from pathlib import Path


def junction_left_plane_contact_right():
    # physical properties of the beam
    A_rho0 = 1
    B_rho0 = np.zeros(3)
    C_rho0 = np.eye(3)

    L = 2 * np.pi
    Ei = np.ones(3) * 1.0e0
    Fi = np.ones(3) * 1.0e-1

    material_model = Simo1986(Ei, Fi)

    # junction at the origin
    r_OB1 = np.zeros(3)
    frame_left = Frame(r_OP=r_OB1)

    # discretization properties
    p = 3
    p_r = p
    p_di = p  # - 1
    # nQP = int(np.ceil((p + 1)**2 / 2))
    nQP = p + 1
    print(f"nQP: {nQP}")
    nEl = 10

    basis = "B-spline"
    # basis = 'lagrange'

    # build reference configuration
    Q = TimoshenkoDirectorIntegral.straight_configuration(
        p_r, p_di, nEl, L, basis=basis
    )
    q0 = Q.copy()

    beam = TimoshenkoDirectorIntegral(
        material_model,
        A_rho0,
        B_rho0,
        C_rho0,
        p_r,
        p_di,
        nQP,
        nEl,
        Q=Q,
        q0=q0,
        basis=basis,
    )

    # left joint
    joint_frame_beam = RigidConnection(frame_left, beam, r_OB1, frame_ID2=(0,))

    # gravity beam
    __g = np.array([0, 0, -A_rho0 * 9.81 * 5.0e-3])
    f_g_beam = DistributedForce1DBeam(lambda t, xi: t * __g, beam)

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
    model = System()
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

    ############
    # vtk export
    ############

    # # vtk export
    # # beam.post_processing(t, q[:, beam.qDOF], f"director_beam_{basis}")

    # path = Path(__file__)
    # path = path.parent / path.stem
    # path.mkdir(parents=True, exist_ok=True)  # create directory
    # # post_processing(
    # #     [beam],
    # #     sol.t,
    # #     sol.q,
    # #     str(path / ("beam")),
    # # )

    # for i, (ti, qi) in enumerate(zip(t, q)):
    #     # beam.post_processing_vtk_volume(
    #     beam.post_processing_vtk_volume_circle(
    #         ti, qi[beam.qDOF], str(path / (f"beam_{i}.vtu")), R=1
    #     )


# def junction_left_cylindrical_contact_right():
#     # physical properties of the beam
#     A_rho0 = 1
#     B_rho0 = np.zeros(3)
#     C_rho0 = np.eye(3)

#     L = 2 * np.pi * 1.0e-1
#     Ei = np.ones(3) * 1.0e1
#     Fi = np.ones(3) * 1.0e-1

#     material_model = Simo1986(Ei, Fi)

#     # discretization properties
#     p = 3
#     p_r = p
#     p_di = p  # - 1
#     # nQP = int(np.ceil((p + 1)**2 / 2))
#     nQP = p + 1
#     print(f"nQP: {nQP}")
#     nEl = 5

#     basis = "B-spline"
#     # basis = 'lagrange'

#     # build reference configuration
#     Q = straight_configuration(p_r, p_di, nEl, L, basis=basis)
#     q0 = Q.copy()

#     beam = Timoshenko_director_integral(
#         material_model,
#         A_rho0,
#         B_rho0,
#         C_rho0,
#         p_r,
#         p_di,
#         nQP,
#         nEl,
#         Q=Q,
#         q0=q0,
#         basis=basis,
#     )

#     # # gravity beam
#     # __g = np.array([0, 0, -A_rho0 * 9.81 * 1.0e-3])
#     # f_g_beam = LineForce(lambda xi, t: t * __g, beam)

#     # left cylindrical contact
#     radius = 0.25 * L
#     # r_OP_contact = np.array([L / 3, 0, -1.0 * radius])
#     r_OP_contact = np.array([0, 0, -1.0 * radius])
#     A_IK = A_IK_basic(-np.pi / 2).x()
#     frame_contact_left = Frame(r_OP=r_OP_contact, A_IK=A_IK)
#     prox_r_N = 1.0e-3
#     # contact_left = Point2Cylinder(frame_contact_left, beam, radius, prox_r_N, frame_ID=(1. / 3.,))
#     contact_left = Point2Cylinder(
#         frame_contact_left, beam, radius, prox_r_N, frame_ID=(0,)
#     )

#     # left cylindrical contact
#     # r_OP_contact = np.array([L * 2 / 3, 0, -1.0 * radius])
#     r_OP_contact = np.array([L, 0, -1.0 * radius])
#     A_IK = A_IK_basic(-np.pi / 2).x()
#     frame_contact_right = Frame(r_OP=r_OP_contact, A_IK=A_IK)
#     prox_r_N = 1.0e-3
#     # contact_right = Point2Cylinder(frame_contact_right, beam, radius, prox_r_N, frame_ID=(2. / 3.,))
#     contact_right = Point2Cylinder(
#         frame_contact_right, beam, radius, prox_r_N, frame_ID=(1,)
#     )

#     # central joint
#     def r_OB1(t):
#         return np.array([0.5 * L, 0, 0]) + t * np.array([0, 0, -1.4 * radius])

#     frame_center = Frame(r_OP=r_OB1)
#     joint_frame_beam = RigidConnection(frame_center, beam, r_OB1, frame_ID2=(0.5,))

#     # assemble the model
#     model = Model()
#     model.add(beam)
#     model.add(frame_center)
#     model.add(joint_frame_beam)
#     model.add(frame_contact_left)
#     model.add(contact_left)
#     model.add(frame_contact_right)
#     model.add(contact_right)
#     model.assemble()

#     solver = Newton(
#         model,
#         n_load_steps=20,
#         max_iter=20,
#         atol=1.0e-8,
#         numerical_jacobian=False,
#         prox_r_N=1.0e-3,
#     )

#     sol = solver.solve()

#     # vtk export
#     path = Path(__file__)
#     path = path.parent / path.stem
#     path.mkdir(parents=True, exist_ok=True)  # create directory
#     post_processing(
#         [beam],
#         sol.t,
#         sol.q,
#         str(path / ("beam")),
#     )


if __name__ == "__main__":
    junction_left_plane_contact_right()
    # junction_left_cylindrical_contact_right()
