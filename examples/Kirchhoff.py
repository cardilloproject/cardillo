from cardillo.beams.spatial.material_models import ShearStiffQuadratic, Simo1986
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import (
    RigidConnection,
    SphericalJoint,
    RigidConnectionCable,
)
from cardillo.beams import (
    animate_beam,
    Cable,
    CubicHermiteCable,
    Kirchhoff,
    DirectorAxisAngle,
)
from cardillo.forces import Force, K_Moment, DistributedForce1D
from cardillo.model import Model
from cardillo.solver import Newton
from cardillo.contacts import Point2Plane
from cardillo.math import e1, e2, e3, sin, pi, smoothstep2, A_IK_basic

import numpy as np
import matplotlib.pyplot as plt

# # case = "Cable"
# # case = "CubicHermiteCable"
# # case = "Kirchhoff"
# case = "DirectorAxisAngle"

# TODO: Make this an ABC class!
class BeamCrossSection:
    pass


class CircularBeamCrossSection(BeamCrossSection):
    def __init__(self, radius, line_density):
        self.radius = radius

        # density per line element
        self.line_density = line_density  # TODO: Move to base class

        # area of the cross section
        self.area = pi * radius**2

        # second moments of area, see
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        self.I2 = self.I3 = 0.25 * pi * radius**4
        self.I1 = self.I2 + self.I3

        # inertia properties used in director beam formulations
        self.A_rho0 = self.line_density * self.area
        self.B_rho0 = np.zeros(3)  # symmetric cross section
        self.C_rho0 = self.line_density * np.diag(np.array([0, self.I3, self.I2]))

        # TODO: Inertia properties for standard beam formulations.
        # Note: These should coincide with the definition used for rigid bodies

        # TODO
        super().__init__()


class RectangularBeamCrossSection(BeamCrossSection):
    def __init__(self):
        raise NotImplementedError("")
        super().__init__()


def quadratic_beam_material(E, G, cross_section, Beam):
    A = cross_section.area
    I1 = cross_section.I1
    I2 = cross_section.I2
    I3 = cross_section.I3
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([E * I1, E * I3, E * I2])

    from cardillo.beams.spatial.material_models import ShearStiffQuadratic, Simo1986

    if Beam == Cable or Beam == CubicHermiteCable or Beam == Kirchhoff:
        return ShearStiffQuadratic(Ei[0], Fi)
    elif Beam == DirectorAxisAngle:
        return Simo1986(Ei, Fi)
    else:
        raise NotImplementedError("")


def beam_factory(
    nelements,
    polynomial_degree,
    nquadrature_points,
    shape_functions,
    cross_section,
    material_model,
    Beam,
    L,
    r_OP=np.zeros(3),
    A_IK=np.eye(3),
):
    ###############################
    # build reference configuration
    ###############################
    if Beam == Cable:
        Q = Cable.straight_configuration(
            polynomial_degree, nelements, L, r_OP=r_OP, A_IK=A_IK
        )
    elif Beam == CubicHermiteCable:
        Q = CubicHermiteCable.straight_configuration(
            nelements, L, r_OP=r_OP, A_IK=A_IK
        )
    elif Beam == Kirchhoff:
        p_r = polynomial_degree
        p_phi = polynomial_degree
        Q = Kirchhoff.straight_configuration(
            p_r, p_phi, nelements, L, r_OP=r_OP, A_IK=A_IK
        )
    elif Beam == DirectorAxisAngle:
        p_r = polynomial_degree
        p_psi = p_r - 1
        Q = DirectorAxisAngle.straight_configuration(
            p_r, p_psi, nelements, L, r_OP=r_OP, A_IK=A_IK
        )
    else:
        raise NotImplementedError("")

    # Initial configuration coincides with reference configuration.
    # Note: This might be adapted.
    q0 = Q.copy()

    # extract cross section properties.
    # TODO: Maybe we should pass this to the beam model itself?
    A_rho0 = cross_section.A_rho0
    B_rho0 = cross_section.B_rho0
    C_rho0 = cross_section.C_rho0

    ##################
    # build beam model
    ##################
    if Beam == Cable:
        beam = Cable(
            material_model,
            A_rho0,
            B_rho0,
            C_rho0,
            p_r,
            nquadrature_points,
            nelements,
            Q=Q,
            q0=q0,
        )
    elif Beam == CubicHermiteCable:
        beam = CubicHermiteCable(
            material_model,
            A_rho0,
            B_rho0,
            C_rho0,
            p_r,
            nquadrature_points,
            nelements,
            Q=Q,
            q0=q0,
        )
    elif Beam == Kirchhoff:
        beam = Kirchhoff(
            material_model,
            A_rho0,
            B_rho0,
            C_rho0,
            p_r,
            p_phi,
            nquadrature_points,
            nelements,
            Q=Q,
            q0=q0,
        )
    elif Beam == DirectorAxisAngle:
        # TODO: Inertia is wrong!
        beam = DirectorAxisAngle(
            material_model,
            A_rho0,
            B_rho0,
            C_rho0,
            p_r,
            p_psi,
            nquadrature_points,
            nelements,
            Q=Q,
            q0=q0,
        )
    else:
        raise NotImplementedError("")

    return beam


def tests():
    # used beam model
    # Beam = Cable
    # Beam = CubicHermiteCable
    # Beam = Kirchhoff
    Beam = DirectorAxisAngle

    # number of elements
    nelements = 10

    # used polynomial degree
    polynomial_degree = 2

    # number of quadrature points
    # nquadrature_points = int(np.ceil((polynomial_degree + 1)**2 / 2))
    nquadrature_points = polynomial_degree + 1

    # used shape functions for discretization
    shape_functions = "B-spline"

    # used cross section
    radius = 0.1
    line_density = 1
    cross_section = CircularBeamCrossSection(radius, line_density)

    # Young's and shear modulus
    E = 1
    nu = 0.5
    G = E / (2.0 * (1.0 + nu))

    # build quadratic material model
    material_model = quadratic_beam_material(E, G, cross_section, Beam)

    # starting point and orientation of initial point, initial length
    r_OP = np.zeros(3)
    A_IK = np.eye(3)
    L = 2 * pi

    # build beam model
    beam = beam_factory(
        nelements,
        polynomial_degree,
        nquadrature_points,
        shape_functions,
        cross_section,
        material_model,
        Beam,
        L,
        r_OP=r_OP,
        A_IK=A_IK,
    )

    exit()

    # physical properties of the beam
    A_rho0 = 1
    B_rho0 = np.zeros(3)
    C_rho0 = np.eye(3)

    L = 2 * np.pi
    E1 = 1.0e0
    Ei = np.ones(3)
    Fi = np.ones(3)

    material_model_Kirchhoff = ShearStiffQuadratic(E1, Fi)
    material_model_Timoshenko = Simo1986(Ei, Fi)

    # junctions
    r_OB1 = np.zeros(3)
    r_OB2 = np.array([L, 0, 0])
    frame1 = Frame(r_OP=r_OB1)
    frame2 = Frame(r_OP=r_OB2)

    # discretization properties
    polynomial_degree = 2
    p_r = polynomial_degree
    p_phi = (
        polynomial_degree - 1
    )  # TODO: p_phi = p_r - 1 seems to be a very good choice!
    # nQP = int(np.ceil((p + 1)**2 / 2))
    nquadrature_points = polynomial_degree + 1
    print(f"nQP: {nquadrature_points}")
    nelements = 10

    # build reference configuration
    if case == "Cable":
        Q = Cable.straight_configuration(p_r, nelements, L)
    elif case == "CubicHermiteCable":
        Q = CubicHermiteCable.straight_configuration(nelements, L)
    elif case == "Kirchhoff":
        Q = Kirchhoff.straight_configuration(p_r, p_phi, nelements, L)
    elif case == "DirectorAxisAngle":
        Q = DirectorAxisAngle.straight_configuration(p_r, p_phi, nelements, L)
    else:
        raise NotImplementedError("")
    q0 = Q.copy()

    if case == "Cable":
        beam = Cable(
            material_model_Kirchhoff,
            A_rho0,
            B_rho0,
            C_rho0,
            p_r,
            nquadrature_points,
            nelements,
            Q=Q,
            q0=q0,
        )
    elif case == "CubicHermiteCable":
        beam = CubicHermiteCable(
            material_model_Kirchhoff,
            A_rho0,
            B_rho0,
            C_rho0,
            p_r,
            nquadrature_points,
            nelements,
            Q=Q,
            q0=q0,
        )
    elif case == "Kirchhoff":
        beam = Kirchhoff(
            material_model_Kirchhoff,
            A_rho0,
            B_rho0,
            C_rho0,
            p_r,
            p_phi,
            nquadrature_points,
            nelements,
            Q=Q,
            q0=q0,
        )
    elif case == "DirectorAxisAngle":
        beam = DirectorAxisAngle(
            material_model_Timoshenko,
            A_rho0,
            B_rho0,
            C_rho0,
            p_r,
            p_phi,
            nquadrature_points,
            nelements,
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
    # # q = model.q0
    # q = np.random.rand(model.nq)

    # E_pot = model.E_pot(t, q)
    # print(f"E_pot: {E_pot}")
    # f_pot = model.f_pot(t, q)
    # print(f"f_pot:\n{f_pot}")
    # # f_pot_q = model.f_pot_q(t, q)
    # # print(f"f_pot_q:\n{f_pot_q}")
    # exit()

    # left and right joint
    if case == "Cable" or case == "CubicHermiteCable":
        joint1 = RigidConnectionCable(frame1, beam, r_OB1, frame_ID2=(0,))
        joint2 = RigidConnectionCable(frame2, beam, r_OB2, frame_ID2=(1,))
    elif case == "Kirchhoff" or case == "DirectorAxisAngle":
        joint1 = RigidConnection(frame1, beam, r_OB1, frame_ID2=(0,))
        joint2 = RigidConnection(frame2, beam, r_OB2, frame_ID2=(1,))
    else:
        raise NotImplementedError("")
    # joint1 = SphericalJoint(frame1, beam, r_OB1, frame_ID2=(0,))
    # joint2 = SphericalJoint(frame2, beam, r_OB2, frame_ID2=(1,))

    # gravity beam
    __g = np.array([0, 0, -A_rho0 * 9.81 * 1.0e-3])
    f_g_beam = DistributedForce1D(lambda t, xi: t * __g, beam)

    # moment at right end
    M = lambda t: -np.array([1, 0, 1]) * t * 2 * np.pi * Fi[1] / L * 1.0
    # M = lambda t: e1 * t * 2 * np.pi * Fi[0] / L * 1.0
    # M = lambda t: e2 * t * 2 * np.pi * Fi[1] / L * 1.0
    # M = lambda t: e3 * t * 2 * np.pi * Fi[2] / L * 1.0
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
        n_load_steps=10,
        max_iter=30,
        atol=1.0e-8,
        numerical_jacobian=False,
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
    n_circles = 1
    frac_deformation = 1 / (n_circles + 1)
    frac_rotation = 1 - frac_deformation
    print(f"n_circles: {n_circles}")
    print(f"frac_deformation: {frac_deformation}")
    print(f"frac_rotation:     {frac_rotation}")

    # junctions
    r_OB0 = np.zeros(3)
    # phi = lambda t: n_circles * 2 * pi * smoothstep2(t, frac_deformation, 1.0)
    # phi2 = lambda t: pi / 4 * sin(2 * pi * t)
    # # A_IK0 = lambda t: A_IK_basic(phi(t)).x()
    # A_IK0 = (
    #     lambda t: A_IK_basic(phi2(t)).z()
    #     @ A_IK_basic(phi2(t)).y()
    #     @ A_IK_basic(phi(t)).x()
    # )
    A_IK0 = lambda t: np.eye(3)
    frame1 = Frame(r_OP=r_OB0, A_IK=A_IK0)

    # discretization properties
    p = 2
    p_r = p
    p_phi = p
    # p_phi = p + 1
    # p_phi = p + 2
    # p_phi = p + 1 # seems to cure the non-objectivity (for p = 2)
    # p_phi = p + 2 # this truely fixes the objectivity problems (for p = 2)
    # p_phi = p + 3 # seems to cure the non-objectivity (for p = 3)
    # p_phi = p + 4 # this truely fixes the objectivity problems (for p = 3)
    # nQP = int(np.ceil((p + 1)**2 / 2))
    # nQP = max(p_r, p_phi) + 1
    nQP = p + 1  # reduced integration cuures nonobjectivity for p=2
    # objective pairs:
    # - p=2, p_r=p, p_phi=p+1, nQP=p+1 # reduced integration cures nonobjectivity
    # - p=3, ???
    print(f"nQP: {nQP}")
    nEl = 5

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

    # moment at right end that yields quater circle for t in [0, frac_deforation] and then
    # remains constant
    # M = lambda t: 2 * np.pi * smoothstep2(t, 0.0, frac_deformation) * e1 * Fi[0] / L
    M = lambda t: 2 * np.pi * smoothstep2(t, 0.0, frac_deformation) * e2 * Fi[1] / L
    # momen at right end that yields a quater helix for t in [0, frac_deforation] and then
    # remains constant
    # M = (
    #     lambda t: np.pi
    #     / 2
    #     * smoothstep2(t, 0.0, frac_deformation)
    #     * np.array([1, 0, 1])
    #     * Fi[1]
    #     / L
    # )
    moment = K_Moment(M, beam, (1,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    model.add(moment)
    model.assemble()

    # n_steps_per_rotation = 40
    n_steps_per_rotation = 30

    solver = Newton(
        model,
        n_load_steps=n_steps_per_rotation * (n_circles + 1),
        max_iter=30,
        atol=1.0e-8,
    )

    sol = solver.solve()
    t = sol.t
    q = sol.q

    ##################################
    # TODO: Visualize potential energy
    ##################################
    E_pot = np.array([model.E_pot(ti, qi) for (ti, qi) in zip(t, q)])
    # phis = phi(t)

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

    # ax[0].plot(t, phis, label="phi")
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

    # visualize final centerline projected in all three planes
    r_OPs = beam.centerline(q[-1])
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(r_OPs[0, :], r_OPs[1, :], label="x-y")
    ax[1].plot(r_OPs[1, :], r_OPs[2, :], label="y-z")
    ax[2].plot(r_OPs[2, :], r_OPs[0, :], label="z-x")
    ax[0].grid()
    ax[0].legend()
    ax[0].set_aspect(1)
    ax[1].grid()
    ax[1].legend()
    ax[1].set_aspect(1)
    ax[2].grid()
    ax[2].legend()
    ax[2].set_aspect(1)

    # plt.show()
    # exit()

    ###########
    # animation
    ###########
    animate_beam(t, q, beam, L, show=True)


if __name__ == "__main__":
    tests()
    # objectivity()
