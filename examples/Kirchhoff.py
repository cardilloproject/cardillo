from cardillo.beams.spatial import CircularCrossSection, ShearStiffQuadratic, Simo1986
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import (
    RigidConnection,
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
from cardillo.solver import (
    Newton,
    ScipyIVP,
    GenAlphaFirstOrderVelocityGGL,
    GenAlphaFirstOrderVelocity,
    GenAlphaDAEAcc,
    Moreau,
)
from cardillo.math import e1, e2, e3, sin, pi, smoothstep2, A_IK_basic

import numpy as np
import matplotlib.pyplot as plt


def quadratic_beam_material(E, G, cross_section, Beam):
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])

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
        Q = CubicHermiteCable.straight_configuration(nelements, L, r_OP=r_OP, A_IK=A_IK)
    elif Beam == Kirchhoff:
        p_r = polynomial_degree
        p_phi = polynomial_degree
        Q = Kirchhoff.straight_configuration(
            p_r, p_phi, nelements, L, r_OP=r_OP, A_IK=A_IK
        )
    elif Beam == DirectorAxisAngle:
        p_r = polynomial_degree
        p_psi = p_r
        Q = DirectorAxisAngle.straight_configuration(
            p_r, p_psi, nelements, L, r_OP=r_OP, A_IK=A_IK, basis=shape_functions
        )
    else:
        raise NotImplementedError("")

    # Initial configuration coincides with reference configuration.
    # Note: This might be adapted.
    q0 = Q.copy()

    # extract cross section properties
    # TODO: Maybe we should pass this to the beam model itself?
    area = cross_section.area
    line_density = cross_section.line_density
    first_moment = cross_section.first_moment
    second_moment = cross_section.second_moment

    # for constant line densities the required quantities are related to the
    # zeroth, first and second moment of area, see Harsch2021 footnote 2.
    # TODO: I think we made an error there since in the Wilberforce pendulum
    # example we used
    # C_rho0 = line_density * np.diag([0, I3, I2]).
    # I think it should be C_rho0^ab = rho0 * I_ba?
    # TODO: Compute again the relation between Binet inertia tensor and
    # standard one.
    A_rho0 = line_density * area
    B_rho0 = line_density * first_moment
    C_rho0 = line_density * second_moment
    # TODO: I think this is Binet's inertia tensor!
    # TODO: See Mäkinen2006, (24) on page 1022 for a clarification of the
    # classical inertia tensor
    C_rho0 = np.zeros((3, 3))
    for a in range(1, 3):
        for b in range(1, 3):
            C_rho0[a, b] = line_density * second_moment[b, a]

    # This is the standard second moment of area weighted by a constant line
    # density
    I_rho0 = line_density * second_moment

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
        beam = DirectorAxisAngle(
            material_model,
            A_rho0,
            I_rho0,
            p_r,
            p_psi,
            nquadrature_points,
            nelements,
            Q=Q,
            q0=q0,
            basis=shape_functions,
        )
    else:
        raise NotImplementedError("")

    return beam


# def run(statics=True):
def run(statics=False):
    # used beam model
    # Beam = Cable
    # Beam = CubicHermiteCable
    # Beam = Kirchhoff
    Beam = DirectorAxisAngle

    # number of elements
    # nelements = 1
    nelements = 5
    # nelements = 10
    # nelements = 20
    # nelements = 30

    # used polynomial degree
    # polynomial_degree = 1
    polynomial_degree = 2
    # polynomial_degree = 3

    # number of quadrature points
    # nquadrature_points = int(np.ceil((polynomial_degree + 1)**2 / 2))
    nquadrature_points = polynomial_degree + 1

    # used shape functions for discretization
    shape_functions = "B-spline"

    # used cross section
    radius = 1.0e-0
    # radius = 1.0e-1
    # radius = 1.0e-3 # this yields no deformation due to locking!
    line_density = 1
    cross_section = CircularCrossSection(line_density, radius)

    # Young's and shear modulus
    E = 1.0e3
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

    # number of full rotations after deformation
    # TODO: Allow zero circles!
    n_circles = 1
    frac_deformation = 1 / (n_circles + 1)
    frac_rotation = 1 - frac_deformation
    print(f"n_circles: {n_circles}")
    print(f"frac_deformation: {frac_deformation}")
    print(f"frac_rotation:     {frac_rotation}")

    # junctions
    r_OB0 = np.zeros(3)
    # r_OB0 = np.array([-1, 0.25, 3.14])
    if statics:
        phi = lambda t: n_circles * 2 * pi * smoothstep2(t, frac_deformation, 1.0) * 0.5
        # phi2 = lambda t: pi / 4 * sin(2 * pi * smoothstep2(t, frac_deformation, 1.0))
        # A_IK0 = lambda t: A_IK_basic(phi(t)).x()
        # TODO: Get this strange rotation working with a full circle
        # A_IK0 = lambda t: A_IK_basic(phi(t)).z()
        A_IK0 = (
            lambda t: A_IK_basic(0.5 * phi(t)).z()
            @ A_IK_basic(0.5 * phi(t)).y()
            @ A_IK_basic(phi(t)).x()
        )
        # A_IK0 = lambda t: np.eye(3)
    else:
        # phi = lambda t: smoothstep2(t, 0, 0.1) * sin(0.3 * pi * t) * pi / 4
        phi = lambda t: smoothstep2(t, 0, 0.1) * sin(0.6 * pi * t) * pi / 4
        # A_IK0 = lambda t: A_IK_basic(phi(t)).z()
        A_IK0 = (
            lambda t: A_IK_basic(0.5 * phi(t)).z()
            @ A_IK_basic(0.5 * phi(t)).y()
            @ A_IK_basic(phi(t)).x()
        )
        # A_IK0 = lambda t: np.eye(3)

    frame1 = Frame(r_OP=r_OB0, A_IK=A_IK0)

    # left and right joint
    if Beam == Cable or Beam == CubicHermiteCable:
        joint1 = RigidConnectionCable(frame1, beam, r_OB0, frame_ID2=(0,))
    elif Beam == Kirchhoff or Beam == DirectorAxisAngle:
        joint1 = RigidConnection(frame1, beam, r_OB0, frame_ID2=(0,))
    else:
        raise NotImplementedError("")

    # gravity beam
    g = np.array([0, 0, -cross_section.area * cross_section.line_density * 9.81])
    f_g_beam = DistributedForce1D(lambda t, xi: g, beam)

    # moment at right end
    Fi = material_model.Fi
    # M = lambda t: np.array([1, 1, 0]) * smoothstep2(t, 0.0, frac_deformation) * 2 * np.pi * Fi[1] / L
    # M = lambda t: e1 * smoothstep2(t, 0.0, frac_deformation) * 2 * np.pi * Fi[0] / L * 1.0
    # M = lambda t: e2 * smoothstep2(t, 0.0, frac_deformation) * 2 * np.pi * Fi[1] / L * 0.75
    M = (
        lambda t: e3
        * smoothstep2(t, 0.0, frac_deformation)
        * 2
        * np.pi
        * Fi[2]
        / L
        * 0.5
        # * 0.75
    )
    moment = K_Moment(M, beam, (1,))

    # force at right end
    F = lambda t: np.array([0, 0, -1]) * t * 1.0e-2
    force = Force(F, beam, frame_ID=(1,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    # model.add(force)
    if statics:
        model.add(moment)
    # else:
    #     model.add(f_g_beam)
    model.assemble()

    if statics:
        solver = Newton(
            model,
            n_load_steps=50,
            # n_load_steps=200,
            max_iter=30,
            atol=1.0e-8,
            numerical_jacobian=False,
        )
    else:
        # t1 = 5.0
        t1 = 10.0
        dt = 5.0e-2
        # dt = 2.5e-2
        method = "RK45"
        rtol = 1.0e-6
        atol = 1.0e-6
        rho_inf = 0.5

        # solver = ScipyIVP(model, t1, dt, method=method, rtol=rtol, atol=atol)
        # solver = GenAlphaFirstOrderVelocityGGL(model, t1, dt, rho_inf=rho_inf, tol=atol, numerical_jacobian=True)
        # solver = GenAlphaFirstOrderVelocity(
        #     model, t1, dt, rho_inf=rho_inf, tol=atol, numerical_jacobian=False
        # )
        # solver = GenAlphaDAEAcc(model, t1, dt, rho_inf=rho_inf, newton_tol=atol)
        dt = 1.0e-2
        solver = Moreau(model, t1, dt)

    sol = solver.solve()
    t = sol.t
    q = sol.q

    ############################
    # Visualize potential energy
    ############################
    E_pot = np.array([model.E_pot(ti, qi) for (ti, qi) in zip(t, q)])

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(t, E_pot)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("E_pot")
    ax[0].grid()

    idx = np.where(t > frac_deformation)[0]
    ax[1].plot(t[idx], E_pot[idx])
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("E_pot")
    ax[1].grid()

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

    ###########
    # animation
    ###########
    animate_beam(t, q, beam, L, show=True)


def run_contact():
    # used beam model
    Beam = DirectorAxisAngle

    # number of elements
    nelements = 5

    # used polynomial degree
    polynomial_degree = 2

    # number of quadrature points
    # nquadrature_points = int(np.ceil((polynomial_degree + 1)**2 / 2))
    nquadrature_points = polynomial_degree + 1

    # used shape functions for discretization
    shape_functions = "B-spline"

    # used cross section
    radius = 1.0
    # radius = 1.0e-1
    # radius = 1.0e-3 # this yields no deformation due to locking!
    line_density = 1
    cross_section = CircularCrossSection(line_density, radius)

    # Young's and shear modulus
    E = 1.0e3
    nu = 0.5
    G = E / (2.0 * (1.0 + nu))

    # build quadratic material model
    material_model = quadratic_beam_material(E, G, cross_section, Beam)

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3)
    A_IK0 = np.eye(3)
    r_OP1 = np.array([0, 0, 3 * radius])
    A_IK1 = A_IK_basic(pi / 3).z()
    L = 2 * pi

    # build beam model
    beam0 = beam_factory(
        nelements,
        polynomial_degree,
        nquadrature_points,
        shape_functions,
        cross_section,
        material_model,
        Beam,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
    )

    beam1 = beam_factory(
        nelements,
        polynomial_degree,
        nquadrature_points,
        shape_functions,
        cross_section,
        material_model,
        Beam,
        L,
        r_OP=r_OP1,
        A_IK=A_IK1,
    )

    # frames for the junctions
    r_OB0 = np.zeros(3)
    A_IK0 = np.eye(3)
    r_OB1 = np.array([L, 0, 0])
    A_IK1 = np.eye(3)
    frame0 = Frame(r_OP=r_OB0, A_IK=A_IK0)
    frame1 = Frame(r_OP=r_OB1, A_IK=A_IK1)

    # left and right joint
    joint0 = RigidConnection(frame0, beam0, r_OB0, frame_ID2=(0,))
    joint1 = RigidConnection(frame1, beam0, r_OB1, frame_ID2=(1,))

    # gravity beam
    g = np.array([0, 0, -cross_section.area * cross_section.line_density * 9.81])
    f_g_beam0 = DistributedForce1D(lambda t, xi: g, beam0)
    f_g_beam1 = DistributedForce1D(lambda t, xi: g, beam1)

    # assemble the model
    model = Model()
    model.add(beam0)
    model.add(beam1)
    model.add(f_g_beam0)
    model.add(f_g_beam1)
    model.add(frame0)
    model.add(joint0)
    model.add(frame1)
    model.add(joint1)
    model.assemble()

    # dynamic solver
    t1 = 1.0
    dt = 1.0e-2
    solver = Moreau(model, t1, dt)

    sol = solver.solve()
    t = sol.t
    q = sol.q

    ############################
    # Visualize potential energy
    ############################
    E_pot = np.array([model.E_pot(ti, qi) for (ti, qi) in zip(t, q)])

    fig, ax = plt.subplots()

    ax.plot(t, E_pot)
    ax.set_xlabel("t")
    ax.set_ylabel("E_pot")
    ax.grid()

    # visualize final centerline projected in all three planes
    r_OPs = beam0.centerline(q[-1])
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

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam0, beam1], L, show=True)


if __name__ == "__main__":
    # run()
    run_contact()
