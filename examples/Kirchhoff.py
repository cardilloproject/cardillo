from cardillo.beams.spatial.material_models import ShearStiffQuadratic
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import (
    RigidConnection,
    SphericalJoint,
    RigidConnectionCable,
)
from cardillo.beams import (
    animate_beam,
    Cable,
    Kirchhoff,
)
from cardillo.forces import Force, K_Moment, DistributedForce1D
from cardillo.model import Model
from cardillo.solver import Newton
from cardillo.contacts import Point2Plane
from cardillo.math import e1, e2, e3, sin, pi, smoothstep2, A_IK_basic

import numpy as np
import matplotlib.pyplot as plt

# case = "Cable"
case = "Kirchhoff"

def tests():
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
    p = 3
    p_r = p
    p_phi = p
    # nQP = int(np.ceil((p + 1)**2 / 2))
    nQP = p + 1
    print(f"nQP: {nQP}")
    nEl = 5

    # build reference configuration
    if case == "Cable":
        Q = Cable.straight_configuration(p_r, nEl, L)
    elif case == "Kirchhoff":
        Q = Kirchhoff.straight_configuration(p_r, p_phi, nEl, L)
    else:
        raise NotImplementedError("")
    q0 = Q.copy()

    if case == "Cable":
        beam = Cable(
            material_model,
            A_rho0,
            B_rho0,
            C_rho0,
            p_r,
            nQP,
            nEl,
            Q=Q,
            q0=q0,
        )
    elif case == "Kirchhoff":
        beam = Kirchhoff(
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
    else:
        raise NotImplementedError("")

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
    if case == "Cable":
        joint1 = RigidConnectionCable(frame1, beam, r_OB1, frame_ID2=(0,))
        joint2 = RigidConnectionCable(frame2, beam, r_OB2, frame_ID2=(1,))
    elif case == "Kirchhoff":
        joint1 = RigidConnection(frame1, beam, r_OB1, frame_ID2=(0,))
        joint2 = RigidConnection(frame2, beam, r_OB2, frame_ID2=(1,))
    else:
        raise NotImplementedError("")
    # joint1 = SphericalJoint(frame1, beam, r_OB1, frame_ID2=(0,))
    # joint2 = SphericalJoint(frame2, beam, r_OB2, frame_ID2=(1,))

    # gravity beam
    __g = np.array([0, 0, -A_rho0 * 9.81 * 5.0e-3])
    f_g_beam = DistributedForce1D(lambda t, xi: t * __g, beam)

    # moment at right end
    M = lambda t: -np.array([1, 0, 1]) * t * 2 * np.pi * Fi[1] / L * 0.5
    # M = lambda t: -np.array([0, 1, 1]) * t * 2 * np.pi * Fi[1] / L * 0.5
    # M = lambda t: e1 * t * 2 * np.pi * Fi[0] / L * 1.0
    M = lambda t: e2 * t * 2 * np.pi * Fi[1] / L * 0.5
    # M = lambda t: e3 * t * 2 * np.pi * Fi[2] / L * 0.45
    moment = K_Moment(M, beam, (1,))

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
    # model.add(frame2)
    # model.add(joint2)
    # model.add(f_g_beam)
    # model.add(frame_contact)
    # model.add(contact)
    model.add(moment)
    # model.add(force)
    model.assemble()

    solver = Newton(
        model,
        n_load_steps=5,
        max_iter=30,
        atol=1.0e-8,
        numerical_jacobian=False,
        # numerical_jacobian=True,
        prox_r_N=1.0e-3,
    )

    sol = solver.solve()
    t = sol.t
    q = sol.q

    ###########
    # animation
    ###########
    animate_beam(t, q, beam, L, show=True)


def objectivity():
    # physical properties of the beam
    A_rho0 = 1
    B_rho0 = np.zeros(3)
    C_rho0 = np.eye(3)

    L = 2 * np.pi
    E1 = 1.0e0
    Fi = np.ones(3) * 1.0e-1

    material_model = ShearStiffQuadratic(E1, Fi)

    # number of full rotations after deformation
    n_circles = 2
    frac_deformation = 1 / (n_circles + 1)
    frac_rotation = 1 - frac_deformation
    print(f"n_circles: {n_circles}")
    print(f"frac_deformation: {frac_deformation}")
    print(f"frac_rotation:     {frac_rotation}")

    # junctions
    r_OB0 = np.zeros(3)
    phi = lambda t: n_circles * 2 * pi * smoothstep2(t, frac_deformation, 1.0)
    phi2 = lambda t: pi / 4 * sin(2 * pi * t)
    # A_IK0 = lambda t: A_IK_basic(phi(t)).x()
    A_IK0 = lambda t: A_IK_basic(phi2(t)).z() @ A_IK_basic(phi2(t)).y() @ A_IK_basic(phi(t)).x()
    frame1 = Frame(r_OP=r_OB0, A_IK=A_IK0)
    
    # discretization properties
    p = 3
    p_r = p
    p_phi = p + 1
    # p_phi = p + 1 # seems to cure the non-objectivity (for p = 2)
    # p_phi = p + 2 # this truely fixes the objectivity problems (for p = 2)
    # p_phi = p + 3 # seems to cure the non-objectivity (for p = 3)
    # p_phi = p + 4 # this truely fixes the objectivity problems (for p = 3)
    # nQP = int(np.ceil((p + 1)**2 / 2))
    # nQP = max(p_r, p_phi) + 1
    nQP = p + 1
    print(f"nQP: {nQP}")
    nEl = 1

    # build reference configuration
    Q = Kirchhoff.straight_configuration(p_r, p_phi, nEl, L)
    q0 = Q.copy()

    # build beam model
    beam = Kirchhoff(
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

    # left and right joint
    joint1 = RigidConnection(frame1, beam, r_OB0, frame_ID2=(0,))

    # moment at right end that yields quater circle in t in [0, 0.5] and then 
    # remains constant
    M = lambda t: np.pi / 2 * smoothstep2(t, 0.0, frac_deformation) * e2 * Fi[1] / L
    moment = K_Moment(M, beam, (1,))
    
    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    model.add(moment)
    model.assemble()

    n_steps_per_rotation = 30

    solver = Newton(
        model,
        n_load_steps=n_steps_per_rotation * (n_circles + 1),
        # n_load_steps=2,
        max_iter=30,
        atol=1.0e-12,
        numerical_jacobian=False,
        prox_r_N=1.0e-3,
    )

    sol = solver.solve()
    t = sol.t
    q = sol.q

    ##################################
    # TODO: Visualize potential energy
    ##################################
    E_pot = np.array([model.E_pot(ti, qi) for (ti, qi) in zip(t, q)])
    phis = phi(t)

    def alpha(t, q, frame_ID):
        # local degrees of freedom of the beam
        qBeam = q[beam.qDOF]

        # identify element degrees of freedom
        el = beam.element_number(frame_ID[0])
        elDOF = beam.elDOF[el]
        qe = qBeam[elDOF]

        # evaluate basis functions and angle
        N, _ = beam.basis_functions_phi(frame_ID[0])

        # interpolate angle
        return N @ qe[beam.phiDOF]

    alpha0s = np.array([alpha(ti, qi, (0,)) for (ti, qi) in zip(t, q)])
    alpha05s = np.array([alpha(ti, qi, (0.5,)) for (ti, qi) in zip(t, q)])
    alpha1s = np.array([alpha(ti, qi, (1,)) for (ti, qi) in zip(t, q)])

    fig, ax = plt.subplots(1, 3)

    ax[0].plot(t, phis, label="phi")
    ax[0].plot(t, alpha0s, label="alpha(xi=0)")
    ax[0].plot(t, alpha05s, label="alpha(xi=0.5)")
    ax[0].plot(t, alpha1s, label="alpha(xi=1)")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("angles")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, E_pot)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("E_pot")
    ax[1].grid()

    idx = np.where(t > frac_deformation)[0]
    ax[2].plot(t[idx], E_pot[idx])
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("E_pot")
    ax[2].grid()

    plt.show()
    exit()

    ###########
    # animation
    ###########
    animate_beam(t, q, beam, L, show=True)


if __name__ == "__main__":
    # tests()
    objectivity()