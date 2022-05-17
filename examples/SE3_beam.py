from cardillo.math import e1, e2, e3, sqrt, sin, cos, pi, smoothstep2, A_IK_basic
from cardillo.beams.spatial import (
    UserDefinedCrossSection,
    CircularCrossSection,
    ShearStiffQuadratic,
    Simo1986,
)
from cardillo.math.SE3 import SE3, se3
from cardillo.math.algebra import ax2skew, skew2ax, norm
from cardillo.math.rotations import (
    inverse_tangent_map,
    rodriguez,
    rodriguez_inv,
    tangent_map,
)
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import (
    RigidConnection,
    SphericalJoint,
)
from cardillo.beams import (
    animate_beam,
    TimoshenkoAxisAngle,
    TimoshenkoAxisAngleSE3,
    TimoshenkoDirectorDirac,
    TimoshenkoQuarternionSE3,
)
from cardillo.forces import Force, K_Moment, Moment, DistributedForce1D
from cardillo.model import Model
from cardillo.solver import (
    Newton,
    ScipyIVP,
    GenAlphaFirstOrder,
    Moreau,
)

import numpy as np
import matplotlib.pyplot as plt


def quadratic_beam_material(E, G, cross_section, Beam):
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    return Simo1986(Ei, Fi)


def beam_factory(
    nelements,
    polynomial_degree,
    nquadrature_points,
    shape_functions,
    cross_section,
    material_model,
    Beam,
    L,
    r_OP0=np.zeros(3, dtype=float),
    A_IK0=np.eye(3, dtype=float),
    v_P0=np.zeros(3, dtype=float),
    K_omega_IK0=np.zeros(3, dtype=float),
):
    ###############################
    # build reference configuration
    ###############################
    if Beam == TimoshenkoAxisAngle:
        p_r = polynomial_degree
        p_psi = max(1, p_r - 1)
        Q = TimoshenkoAxisAngle.straight_configuration(
            p_r, p_psi, nelements, L, r_OP=r_OP0, A_IK=A_IK0, basis=shape_functions
        )
    elif Beam == TimoshenkoAxisAngleSE3:
        p_r = polynomial_degree
        p_psi = polynomial_degree
        # Q = TimoshenkoAxisAngleSE3.straight_configuration(
        #     p_r, p_psi, nelements, L, r_OP=r_OP0, A_IK=A_IK0, basis=shape_functions
        # )
        # TODO: Implement computation of initial velocities!
        Q, u0 = TimoshenkoAxisAngleSE3.initial_configuration(
            p_r,
            p_psi,
            nelements,
            L,
            r_OP0=r_OP0,
            A_IK0=A_IK0,
            v_P0=v_P0,
            K_omega_IK0=K_omega_IK0,
            basis=shape_functions,
        )
    elif Beam == TimoshenkoDirectorDirac:
        p_r = polynomial_degree
        p_psi = max(1, polynomial_degree - 1)
        Q = TimoshenkoDirectorDirac.straight_configuration(
            p_r, p_psi, nelements, L, r_OP=r_OP0, A_IK=A_IK0, basis=shape_functions
        )
    elif Beam == TimoshenkoQuarternionSE3:
        p_r = polynomial_degree
        p_psi = polynomial_degree
        Q = TimoshenkoQuarternionSE3.straight_configuration(
            p_r, p_psi, nelements, L, r_OP=r_OP0, A_IK=A_IK0, basis=shape_functions
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
    if Beam == TimoshenkoAxisAngle:
        beam = TimoshenkoAxisAngle(
            material_model,
            A_rho0,
            I_rho0,
            p_r,
            p_psi,
            nquadrature_points,
            nelements,
            Q=Q,
            q0=q0,
            u0=u0,
            basis=shape_functions,
        )
    elif Beam == TimoshenkoAxisAngleSE3:
        beam = TimoshenkoAxisAngleSE3(
            material_model,
            A_rho0,
            I_rho0,
            p_r,
            p_psi,
            nquadrature_points,
            nelements,
            Q=Q,
            q0=q0,
            u0=u0,
            basis=shape_functions,
        )
    elif Beam == TimoshenkoDirectorDirac:
        beam = TimoshenkoDirectorDirac(
            material_model,
            A_rho0,
            B_rho0,
            I_rho0,
            p_r,
            p_psi,
            nquadrature_points,
            nelements,
            Q=Q,
            q0=q0,
            u0=u0,
            basis=shape_functions,
        )
    elif Beam == TimoshenkoQuarternionSE3:
        beam = TimoshenkoQuarternionSE3(
            material_model,
            A_rho0,
            I_rho0,
            p_r,
            p_psi,
            nquadrature_points,
            nelements,
            Q=Q,
            q0=q0,
            u0=u0,
            basis=shape_functions,
        )
    else:
        raise NotImplementedError("")

    return beam


def run(statics):
    # Beam = TimoshenkoAxisAngle
    # Beam = TimoshenkoAxisAngleSE3
    Beam = TimoshenkoQuarternionSE3

    # number of elements
    # nelements = 1
    nelements = 2
    # nelements = 4
    # nelements = 8
    # nelements = 16
    # nelements = 32
    # nelements = 64

    # used polynomial degree
    polynomial_degree = 1
    # polynomial_degree = 2
    # polynomial_degree = 3
    # polynomial_degree = 5
    # polynomial_degree = 6

    # number of quadrature points
    # TODO: We have to distinguish between integration of the mass matrix,
    #       gyroscopic forces and potential forces!
    nquadrature_points = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
    # nquadrature_points = polynomial_degree + 2
    # nquadrature_points = polynomial_degree + 1 # this seems not to be sufficent for p > 1

    # used shape functions for discretization
    shape_functions = "B-spline"
    # shape_functions = "Lagrange"

    # used cross section
    # slenderness = 1
    slenderness = 1.0e1
    # slenderness = 1.0e2
    # slenderness = 1.0e3
    # slenderness = 1.0e4
    radius = 1
    # radius = 1.0e-0
    # radius = 1.0e-1
    # radius = 5.0e-2
    # radius = 1.0e-3 # this yields no deformation due to locking!
    line_density = 1
    cross_section = CircularCrossSection(line_density, radius)

    # Young's and shear modulus
    # E = 1.0e0
    E = 1.0e3
    nu = 0.5
    G = E / (2.0 * (1.0 + nu))

    # build quadratic material model
    material_model = quadratic_beam_material(E, G, cross_section, Beam)
    # print(f"Ei: {material_model.Ei}")
    print(f"Fi: {material_model.Fi}")

    # starting point and orientation of initial point, initial length
    r_OP = np.zeros(3)
    A_IK = np.eye(3)
    L = radius * slenderness

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
        r_OP0=r_OP,
        A_IK0=A_IK,
    )

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
        phi = (
            lambda t: n_circles * 2 * pi * smoothstep2(t, frac_deformation, 1.0)
        )  # * 0.5
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
        phi = lambda t: smoothstep2(t, 0, 0.1) * sin(0.3 * pi * t) * pi / 4
        # phi = lambda t: smoothstep2(t, 0, 0.1) * sin(0.6 * pi * t) * pi / 4
        # A_IK0 = lambda t: A_IK_basic(phi(t)).z()
        A_IK0 = (
            lambda t: A_IK_basic(0.5 * phi(t)).z()
            @ A_IK_basic(0.5 * phi(t)).y()
            @ A_IK_basic(phi(t)).x()
        )
        # A_IK0 = lambda t: np.eye(3)

    frame1 = Frame(r_OP=r_OB0, A_IK=A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, beam, r_OB0, frame_ID2=(0,))

    # gravity beam
    g = np.array(
        [0, 0, -cross_section.area * cross_section.line_density * 9.81 * 1.0e-1]
    )
    if statics:
        f_g_beam = DistributedForce1D(lambda t, xi: t * g, beam)
    else:
        f_g_beam = DistributedForce1D(lambda t, xi: g, beam)

    # moment at right end
    Fi = material_model.Fi
    # M = lambda t: np.array([1, 1, 0]) * smoothstep2(t, 0.0, frac_deformation) * 2 * np.pi * Fi[1] / L
    # M = lambda t: e1 * smoothstep2(t, 0.0, frac_deformation) * 2 * np.pi * Fi[0] / L * 1.0
    # M = lambda t: e2 * smoothstep2(t, 0.0, frac_deformation) * 2 * np.pi * Fi[1] / L * 0.75
    M = (
        lambda t: (e3 * Fi[2])
        # lambda t: (e1 * Fi[0] + e3 * Fi[2])
        * smoothstep2(t, 0.0, frac_deformation)
        * 2
        * np.pi
        / L
        # * 0.1
        * 0.25
        # * 0.5
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
        # model.add(f_g_beam)
    else:
        model.add(f_g_beam)
    model.assemble()

    # t = np.array([0])
    # q = np.array([model.q0])
    # animate_beam(t, q, [beam], scale=L)
    # exit()

    if statics:
        solver = Newton(
            model,
            # n_load_steps=10,
            # n_load_steps=50,
            n_load_steps=100,
            # n_load_steps=500,
            max_iter=30,
            # atol=1.0e-4,
            atol=1.0e-6,
            # atol=1.0e-8,
            # atol=1.0e-10,
            numerical_jacobian=False,
        )
    else:
        # t1 = 1.0
        t1 = 10.0
        dt = 5.0e-2
        # dt = 2.5e-2
        method = "RK45"
        rtol = 1.0e-6
        atol = 1.0e-6
        rho_inf = 0.5

        # solver = ScipyIVP(
        #     model, t1, dt, method=method, rtol=rtol, atol=atol
        # )  # this is no good idea for complement rotation vectors!
        solver = GenAlphaFirstOrder(model, t1, dt, rho_inf=rho_inf, tol=atol)
        # solver = GenAlphaDAEAcc(model, t1, dt, rho_inf=rho_inf, newton_tol=atol)
        # dt = 5.0e-3
        # solver = Moreau(model, t1, dt)

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    if Beam == TimoshenkoAxisAngle or Beam == TimoshenkoAxisAngleSE3:
        ##################################
        # visualize nodal rotation vectors
        ##################################
        fig, ax = plt.subplots()

        for i, nodalDOF_psi in enumerate(beam.nodalDOF_psi):
            psi = q[:, beam.qDOF[nodalDOF_psi]]
            ax.plot(t, np.linalg.norm(psi, axis=1), label=f"||psi{i}||")

        ax.set_xlabel("t")
        ax.set_ylabel("nodal rotation vectors")
        ax.grid()
        ax.legend()

    if Beam == TimoshenkoAxisAngleSE3:
        ################################
        # visualize norm strain measures
        ################################
        fig, ax = plt.subplots(1, 2)

        nxi = 1000
        xis = np.linspace(0, 1, num=nxi)

        K_Gamma = np.zeros((3, nxi))
        K_Kappa = np.zeros((3, nxi))
        for i in range(nxi):
            frame_ID = (xis[i],)
            elDOF = beam.qDOF_P(frame_ID)
            qe = q[-1, beam.qDOF][elDOF]
            _, _, K_Gamma[:, i], K_Kappa[:, i] = beam.eval(qe, xis[i])
        ax[0].plot(xis, K_Gamma[0], "-r", label="K_Gamma0")
        ax[0].plot(xis, K_Gamma[1], "-g", label="K_Gamma1")
        ax[0].plot(xis, K_Gamma[2], "-b", label="K_Gamma2")
        ax[0].grid()
        ax[0].legend()
        ax[1].plot(xis, K_Kappa[0], "-r", label="K_Kappa0")
        ax[1].plot(xis, K_Kappa[1], "-g", label="K_Kappa1")
        ax[1].plot(xis, K_Kappa[2], "-b", label="K_Kappa2")
        ax[1].grid()
        ax[1].legend()

    # ########################################################
    # # visualize norm of tangent vector and quadrature points
    # ########################################################
    # fig, ax = plt.subplots()

    # nxi = 1000
    # xis = np.linspace(0, 1, num=nxi)

    # abs_r_xi = np.zeros(nxi)
    # abs_r0_xi = np.zeros(nxi)
    # for i in range(nxi):
    #     frame_ID = (xis[i],)
    #     elDOF = beam.qDOF_P(frame_ID)
    #     qe = q[-1, beam.qDOF][elDOF]
    #     abs_r_xi[i] = np.linalg.norm(beam.r_OC_xi(t[-1], qe, frame_ID))
    #     q0e = q[0, beam.qDOF][elDOF]
    #     abs_r0_xi[i] = np.linalg.norm(beam.r_OC_xi(t[0], q0e, frame_ID))
    # ax.plot(xis, abs_r_xi, "-r", label="||r_xi||")
    # ax.plot(xis, abs_r0_xi, "--b", label="||r0_xi||")
    # ax.set_xlabel("xi")
    # ax.set_ylabel("||r_xi||")
    # ax.grid()
    # ax.legend()

    # # compute quadrature points
    # for el in range(beam.nelement):
    #     elDOF = beam.elDOF[el]
    #     q0e = q[0, beam.qDOF][elDOF]
    #     for i in range(beam.nquadrature):
    #         xi = beam.qp[el, i]
    #         abs_r0_xi = np.linalg.norm(beam.r_OC_xi(t[0], q0e, (xi,)))
    #         ax.plot(xi, abs_r0_xi, "xr")

    # plt.show()
    # exit()

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
    animate_beam(t, q, [beam], L, show=True)


def locking():
    # Beam = TimoshenkoAxisAngle
    Beam = TimoshenkoAxisAngleSE3

    # number of elements
    # nelements = 1
    # nelements = 2
    nelements = 3
    # nelements = 4
    # nelements = 8
    # nelements = 16
    # nelements = 32
    # nelements = 64

    # used polynomial degree
    polynomial_degree = 1
    # polynomial_degree = 2
    # polynomial_degree = 3
    # polynomial_degree = 5
    # polynomial_degree = 6

    # number of quadrature points
    # TODO: We have to distinguish between integration of the mass matrix,
    #       gyroscopic forces and potential forces!
    # nquadrature_points = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
    # nquadrature_points = polynomial_degree + 2
    # nquadrature_points = polynomial_degree + 1 # this seems not to be sufficent for p > 1
    nquadrature_points = (
        polynomial_degree  # this works for p = 1 and homogeneous deformations!
    )

    # used shape functions for discretization
    shape_functions = "B-spline"
    # shape_functions = "Lagrange"

    # beam length
    # L = 1.0e3 # Meier2015
    L = 5
    # L = 10

    # used cross section
    # slenderness = 1
    slenderness = 1.0e1
    # slenderness = 1.0e2
    # slenderness = 1.0e3
    # slenderness = 1.0e4
    radius = 1
    # radius = 1.0e-0
    # radius = 1.0e-1
    # radius = 5.0e-2
    # radius = 1.0e-3 # this yields no deformation due to locking!
    line_density = 1

    radius = L / slenderness  # Meier2015

    cross_section = CircularCrossSection(line_density, radius)

    # Young's and shear modulus
    E = 1.0  # Meier2015
    G = 0.5  # Meier2015
    # # E = 1.0e1
    # nu = 0.5
    # G = E / (2.0 * (1.0 + nu))

    # build quadratic material model
    material_model = quadratic_beam_material(E, G, cross_section, Beam)
    print(f"Ei: {material_model.Ei}")
    print(f"Fi: {material_model.Fi}")

    # starting point and orientation of initial point, initial length
    r_OP = np.zeros(3)
    A_IK = np.eye(3)
    # L = radius * slenderness

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
        r_OP0=r_OP,
        A_IK0=A_IK,
    )

    # junctions
    r_OB0 = np.zeros(3)
    A_IK0 = lambda t: np.eye(3)
    frame1 = Frame(r_OP=r_OB0, A_IK=A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, beam, r_OB0, frame_ID2=(0,))

    # moment at right end
    Fi = material_model.Fi
    M = (
        # lambda t: (e1 * Fi[0])
        # lambda t: (e2 * Fi[1])
        # lambda t: (e3 * Fi[2])
        lambda t: (e1 * Fi[0] + e3 * Fi[2])
        # lambda t: (e2 * Fi[1] + e3 * Fi[2])
        * smoothstep2(t, 0.0, 1)
        * 2
        * np.pi
        / L
        # * 0.25
        # * 0.5
    )
    moment = K_Moment(M, beam, (1,))

    # external force at the right end
    # force = Force(lambda t: 2.5e-3 * t * (e1 + e2 + e3), beam, frame_ID=(1,), K_r_SP=5 * e2)
    force = Force(lambda t: -1.0e-3 * t * e3, beam, frame_ID=(1,), K_r_SP=5 * e2)
    # # force = Force(lambda t: -1.0e-3 * t * e3, beam, frame_ID=(0.5,), K_r_SP=e2)
    # force = Force(lambda t: -4.0e-3 * t * e2, beam, frame_ID=(1,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    model.add(moment)
    # model.add(force)
    model.assemble()

    solver = Newton(
        model,
        # n_load_steps=10,
        n_load_steps=50,
        # n_load_steps=100,
        # n_load_steps=500,
        max_iter=30,
        # atol=1.0e-4,
        atol=1.0e-6,
        # atol=1.0e-8,
        # atol=1.0e-10,
        numerical_jacobian=False,
    )
    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # visualize nodal rotation vectors
    fig, ax = plt.subplots()

    for i, nodalDOF_psi in enumerate(beam.nodalDOF_psi):
        psi = q[:, beam.qDOF[nodalDOF_psi]]
        ax.plot(t, np.linalg.norm(psi, axis=1), label=f"||psi{i}||")

    ax.set_xlabel("t")
    ax.set_ylabel("nodal rotation vectors")
    ax.grid()
    ax.legend()

    ################################
    # visualize norm strain measures
    ################################
    fig, ax = plt.subplots(1, 2)

    nxi = 1000
    xis = np.linspace(0, 1, num=nxi)

    K_Gamma = np.zeros((3, nxi))
    K_Kappa = np.zeros((3, nxi))
    for i in range(nxi):
        frame_ID = (xis[i],)
        elDOF = beam.qDOF_P(frame_ID)
        qe = q[-1, beam.qDOF][elDOF]
        _, _, K_Gamma[:, i], K_Kappa[:, i] = beam.eval(qe, xis[i])
    ax[0].plot(xis, K_Gamma[0], "-r", label="K_Gamma0")
    ax[0].plot(xis, K_Gamma[1], "-g", label="K_Gamma1")
    ax[0].plot(xis, K_Gamma[2], "-b", label="K_Gamma2")
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(xis, K_Kappa[0], "-r", label="K_Kappa0")
    ax[1].plot(xis, K_Kappa[1], "-g", label="K_Kappa1")
    ax[1].plot(xis, K_Kappa[2], "-b", label="K_Kappa2")
    ax[1].grid()
    ax[1].legend()

    # ########################################################
    # # visualize norm of tangent vector and quadrature points
    # ########################################################
    # fig, ax = plt.subplots()

    # nxi = 1000
    # xis = np.linspace(0, 1, num=nxi)

    # abs_r_xi = np.zeros(nxi)
    # abs_r0_xi = np.zeros(nxi)
    # for i in range(nxi):
    #     frame_ID = (xis[i],)
    #     elDOF = beam.qDOF_P(frame_ID)
    #     qe = q[-1, beam.qDOF][elDOF]
    #     abs_r_xi[i] = np.linalg.norm(beam.r_OC_xi(t[-1], qe, frame_ID))
    #     q0e = q[0, beam.qDOF][elDOF]
    #     abs_r0_xi[i] = np.linalg.norm(beam.r_OC_xi(t[0], q0e, frame_ID))
    # ax.plot(xis, abs_r_xi, "-r", label="||r_xi||")
    # ax.plot(xis, abs_r0_xi, "--b", label="||r0_xi||")
    # ax.set_xlabel("xi")
    # ax.set_ylabel("||r_xi||")
    # ax.grid()
    # ax.legend()

    # # compute quadrature points
    # for el in range(beam.nelement):
    #     elDOF = beam.elDOF[el]
    #     q0e = q[0, beam.qDOF][elDOF]
    #     for i in range(beam.nquadrature):
    #         xi = beam.qp[el, i]
    #         abs_r0_xi = np.linalg.norm(beam.r_OC_xi(t[0], q0e, (xi,)))
    #         ax.plot(xi, abs_r0_xi, "xr")

    # plt.show()
    # exit()

    ############################
    # Visualize potential energy
    ############################
    E_pot = np.array([model.E_pot(ti, qi) for (ti, qi) in zip(t, q)])

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(t, E_pot)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("E_pot")
    ax[0].grid()

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
    animate_beam(t, q, [beam], L, show=True)


def SE3_interpolation():
    from cardillo.beams.spatial.Timoshenko import (
        SE3,
        SE3inv,
        SE3log,
        se3exp,
        se3tangent_map,
    )

    # def reference_rotation(r_IA, r_IB, A_IA, A_IB, case="left"):
    # def reference_rotation(r_IA, r_IB, A_IA, A_IB, case="right"):
    def reference_rotation(r_IA, r_IB, A_IA, A_IB, case="midway"):
        """Reference rotation for SE(3) object in analogy to the proposed
        formulation by Crisfield1999 (5.8).

        Three cases are implemented: 'midway', 'left'  and 'right'.
        """

        if case == "midway":
            # nodal SE(3) objects
            H_IA = SE3(A_IA, r_IA)
            H_IB = SE3(A_IB, r_IB)

            # midway SE(3) object
            return H_IA @ se3exp(0.5 * SE3log(SE3inv(H_IA) @ H_IB))
        elif case == "left":
            return SE3(A_IA, r_IA)
        elif case == "right":
            return SE3(A_IB, r_IB)
        else:
            raise RuntimeError("Unsupported case chosen.")

    def interp1(r_OA, r_OB, psi_A, psi_B, xi):
        # evaluate rodriguez formular for both nodes
        A_IA = rodriguez(psi_A)
        A_IB = rodriguez(psi_B)

        # nodal SE(3) objects
        H_IA = SE3(A_IA, r_OA)
        H_IB = SE3(A_IB, r_OB)

        # reference SE(3) object
        H_IR = reference_rotation(r_OA, r_OB, A_IA, A_IB)

        # evaluate inverse reference SE(3) object
        H_RI = SE3inv(H_IR)

        H_nodes = [H_IA, H_IB]
        h_rel = np.zeros(6, dtype=float)
        h_rel_xi = np.zeros(6, dtype=float)
        for node in range(2):
            # current SE(3) object
            H_IK = H_nodes[node]

            # relative SE(3)/ se(3) objects
            H_RK = H_RI @ H_IK
            h_RK = SE3log(H_RK)

            # relative interpolation of se(3) using linear shape functions
            if node == 0:
                h_rel += (1.0 - xi) * h_RK
                h_rel_xi += -h_RK
            else:
                h_rel += xi * h_RK
                h_rel_xi += h_RK

        # composition of reference rotation and relative one
        H_IK = H_IR @ se3exp(h_rel)

        # alternative computation of strain measures
        H_RK = se3exp(h_rel)

        R_r_xi = h_rel_xi[:3]
        R_omega_xi = h_rel_xi[3:]
        H_RK_xi = SE3(ax2skew(R_omega_xi), R_r_xi)

        strains_tilde = SE3inv(H_RK) @ H_RK_xi
        strains2 = SE3log(strains_tilde)

        # objective strains
        T = se3tangent_map(h_rel)
        strains = T @ h_rel_xi

        # extract centerline and transformation
        A_IK = H_IK[:3, :3]
        r_OP = H_IK[:3, 3]

        # extract strains
        K_Gamma_bar = strains[:3]  # this is K_r_xi
        K_Kappa_bar = strains[3:]

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    # orgin and zero rotatin vector
    r_OA = np.zeros(3, dtype=float)
    psi_A = np.zeros(3, dtype=float)

    # quater circle
    r_OB = np.sqrt(2.0) / 2.0 * np.array([1, 1, 0], dtype=float)
    psi_B = np.pi / 2.0 * np.array([0, 0, 1], dtype=float)

    # # half circle
    # r_OB = np.array([0, 1, 0], dtype=float)
    # psi_B = np.pi * np.array([0, 0, 1], dtype=float)

    num = 20
    xis = np.linspace(0, 1, num=num)

    r_OP = np.zeros((3, num), dtype=float)
    A_IK = np.zeros((num, 3, 3), dtype=float)
    K_Gamma_bar = np.zeros((3, num), dtype=float)
    K_Kappa_bar = np.zeros((3, num), dtype=float)
    for i, xi in enumerate(xis):
        r_OP[:, i], A_IK[i], K_Gamma_bar[:, i], K_Kappa_bar[:, i] = interp1(
            r_OA, r_OB, psi_A, psi_B, xi
        )

    d1 = np.array([A_IKi[:, 0] for A_IKi in A_IK]).T
    d2 = np.array([A_IKi[:, 1] for A_IKi in A_IK]).T
    # d3 = np.array([A_IKi[:, 2] for A_IKi in A_IK]).T

    fig, ax = plt.subplots(1, 3)

    # centerline and directors
    ax[0].plot(r_OP[0], r_OP[1], "-k")
    ax[0].quiver(*r_OP[:2], *d1[:2], color="red")
    ax[0].quiver(*r_OP[:2], *d2[:2], color="green")
    ax[0].axis("equal")
    ax[0].grid(True)

    # axial and shear strains
    # ax[1].quiver(*r_OP[:2], *K_Gamma_bar[:2], color="red")
    ax[1].plot(xis, K_Gamma_bar[0], "-r", label="K_Gamma_bar0")
    ax[1].plot(xis, K_Gamma_bar[1], "--g", label="K_Gamma_bar1")
    ax[1].plot(xis, K_Gamma_bar[2], "-.b", label="K_Gamma_bar2")
    ax[1].axis("equal")
    ax[1].grid(True)
    ax[1].legend()

    # atorsion and curvatures
    # ax[1].quiver(*r_OP[:2], *K_Kappa_bar[:2], color="red")
    ax[2].plot(xis, K_Kappa_bar[0], "-r", label="K_Kappa_bar0")
    ax[2].plot(xis, K_Kappa_bar[1], "--g", label="K_Kappa_bar1")
    ax[2].plot(xis, K_Kappa_bar[2], "-.b", label="K_Kappa_bar2")
    ax[2].axis("equal")
    ax[2].grid(True)
    ax[2].legend()

    r = sqrt(2.0) / 2.0
    print(f"2 * pi * r: {2 * pi * r}")

    print(f"2 * pi * r / 4: {2 * pi * r / 4.}")
    print(f"K_Gamma_bar0: {K_Gamma_bar[0]}")

    print(f"curvature = 1 / r: {1. / r}")
    print(f"K_Kappa_bar2: {K_Kappa_bar[2]}")

    plt.show()


def HelixIbrahimbegovic1997(export=True):
    """Beam bent to a helical form - Section 5.2 of Ibrahimbegovic1997.

    References
    ==========
    Ibrahimbegovic1997: https://doi.org/10.1016/S0045-7825(97)00059-5
    """
    # Beam = TimoshenkoAxisAngle
    # Beam = TimoshenkoAxisAngleSE3
    # Beam = TimoshenkoDirectorDirac
    Beam = TimoshenkoQuarternionSE3

    # fraction of 10 full rotations and the out of plane force
    # a corresponding fraction of 100 elements is chosen
    # fraction = 0.05  # 1 full rotations
    # fraction = 0.1  # 1 full rotations
    # fraction = 0.20  # 2 full rotations
    # fraction = 0.4  # 4 full rotations
    # fraction = 0.5  # 5 full rotations
    fraction = 1  # 10 full rotations

    # number of elements
    nelements_max = 30
    # nelements_max = 50
    # nelements_max = 100 # Ibrahimbegovic1997
    nelements = max(3, int(fraction * nelements_max))
    # nelements = 25
    print(f"nelemen: {nelements}")

    # used polynomial degree
    polynomial_degree = 1
    # polynomial_degree = 2
    # polynomial_degree = 3
    # polynomial_degree = 5
    # polynomial_degree = 6

    # number of quadrature points
    # TODO: We have to distinguish between integration of the mass matrix,
    #       gyroscopic forces and potential forces!
    nquadrature_points = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
    # nquadrature_points = polynomial_degree + 2
    # nquadrature_points = (
    #     polynomial_degree + 1
    # )  # this seems not to be sufficent for p > 1
    # nquadrature_points = polynomial_degree # this works for p = 1 and homogeneous deformations!

    # used shape functions for discretization
    shape_functions = "B-spline"
    # shape_functions = "Lagrange"

    # beam parameters found in Section 5.1 Ibrahimbegovic1997
    L = 10
    EA = GA = 1.0e4
    GJ = EI = 1.0e2

    # build quadratic material model
    Ei = np.array([EA, GA, GA], dtype=float)
    Fi = np.array([GJ, EI, EI], dtype=float)
    material_model = Simo1986(Ei, Fi)

    # Note: This is never used in statics!
    line_density = 1.0
    radius = 1.0
    cross_section = CircularCrossSection(line_density, radius)

    # starting point and orientation of initial point, initial length
    r_OP = np.zeros(3, dtype=float)
    A_IK = np.eye(3, dtype=float)

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
        r_OP0=r_OP,
        A_IK0=A_IK,
    )

    # junctions
    r_OB0 = np.zeros(3)
    A_IK0 = lambda t: np.eye(3)
    frame1 = Frame(r_OP=r_OB0, A_IK=A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, beam, r_OB0, frame_ID2=(0,))

    # moment at right end
    Fi = material_model.Fi
    M = lambda t: (e3 * 10 * 2 * np.pi * Fi[2] / L * t * fraction)  # 10 full rotations
    # moment = K_Moment(M, beam, (1,))
    moment = Moment(M, beam, (1,))

    # external force at the right end
    F_max = 50  # Ibrahimbegovic1997
    F = lambda t: F_max * e3 * t * fraction
    force = Force(F, beam, frame_ID=(1,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    model.add(moment)
    model.add(force)
    model.assemble()

    # n_load_steps = int(20 * 10 * fraction)
    # n_load_steps = int(100 * 10 * fraction)
    n_load_steps = int(200 * 10 * fraction)  # works for fraction = 0.5

    solver = Newton(
        model,
        n_load_steps=n_load_steps,
        max_iter=30,
        atol=1.0e-8,
        numerical_jacobian=False,
    )
    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    #########################
    # export tip displacement
    #########################
    if export:
        header = "t, x, y, z"
        r_OC_L = np.array(
            [beam.r_OP(ti, qi[beam.qDOF], frame_ID=(1,)) for (ti, qi) in zip(t, q)]
        )
        export_data = np.vstack([t, *r_OC_L.T]).T
        np.savetxt(
            "results/tip_displacement_helix.txt",
            export_data,
            delimiter=", ",
            header=header,
            comments="",
        )

    ###################
    # export centerline
    ###################
    if export:
        nframes = 17
        idxs = np.linspace(0, nt - 1, num=nframes, dtype=int)
        for i, idx in enumerate(idxs):
            ti = t[idx]
            qi = q[idx]
            r_OPs = beam.centerline(qi)

            header = "x, y, z"
            np.savetxt(
                f"results/centerline{i}.txt",
                r_OPs.T,
                delimiter=", ",
                header=header,
                comments="",
            )

    ########################
    # export strain measures
    ########################
    if export:
        nxi = 200
        xis = np.linspace(0.0, 1.0, num=nxi, dtype=float)

        K_Gamma = np.zeros((3, nxi), dtype=float)
        K_Kappa = np.zeros((3, nxi), dtype=float)
        for i, xi in enumerate(xis):
            elDOF = beam.qDOF_P((xi,))

            _, _, K_Gamma_bar, K_Kappa_bar = beam.eval(q[0, beam.qDOF[elDOF]], xi)
            J = norm(K_Gamma_bar)

            _, _, K_Gamma_bar, K_Kappa_bar = beam.eval(q[-1, beam.qDOF[elDOF]], xi)
            K_Gamma[:, i] = K_Gamma_bar / J
            K_Kappa[:, i] = K_Kappa_bar / J

        header = "xi, K_Gamma0, K_Gamma1, K_Gamma2"
        export_data = np.c_[xis, K_Gamma.T]
        np.savetxt(
            f"results/K_Gamma.txt",
            export_data,
            delimiter=", ",
            header=header,
            comments="",
        )

        header = "xi, K_Kappa0, K_Kappa1, K_Kappa2"
        export_data = np.c_[xis, K_Kappa.T]
        np.savetxt(
            f"results/K_Kappa.txt",
            export_data,
            delimiter=", ",
            header=header,
            comments="",
        )

    if Beam == TimoshenkoAxisAngle or Beam == TimoshenkoAxisAngleSE3:
        ##################################
        # visualize nodal rotation vectors
        ##################################
        fig, ax = plt.subplots()

        for i, nodalDOF_psi in enumerate(beam.nodalDOF_psi):
            psi = q[:, beam.qDOF[nodalDOF_psi]]
            ax.plot(t, np.linalg.norm(psi, axis=1), label=f"||psi{i}||")

        ax.set_xlabel("t")
        ax.set_ylabel("nodal rotation vectors")
        ax.grid()
        ax.legend()

        ################################
        # visualize norm strain measures
        ################################
        fig, ax = plt.subplots(1, 2)

        nxi = 1000
        xis = np.linspace(0, 1, num=nxi)

        K_Gamma = np.zeros((3, nxi))
        K_Kappa = np.zeros((3, nxi))
        for i in range(nxi):
            frame_ID = (xis[i],)
            elDOF = beam.qDOF_P(frame_ID)
            qe = q[-1, beam.qDOF][elDOF]
            _, _, K_Gamma[:, i], K_Kappa[:, i] = beam.eval(qe, xis[i])
        ax[0].plot(xis, K_Gamma[0], "-r", label="K_Gamma0")
        ax[0].plot(xis, K_Gamma[1], "-g", label="K_Gamma1")
        ax[0].plot(xis, K_Gamma[2], "-b", label="K_Gamma2")
        ax[0].grid()
        ax[0].legend()
        ax[1].plot(xis, K_Kappa[0], "-r", label="K_Kappa0")
        ax[1].plot(xis, K_Kappa[1], "-g", label="K_Kappa1")
        ax[1].plot(xis, K_Kappa[2], "-b", label="K_Kappa2")
        ax[1].grid()
        ax[1].legend()

    ############################
    # Visualize tip displacement
    ############################
    elDOF = beam.qDOF[beam.elDOF[-1]]
    r_OP = np.array([beam.r_OP(ti, qi[elDOF], (1,)) for (ti, qi) in zip(t, q)])

    fig, ax = plt.subplots()

    ax.plot(t, r_OP[:, 0], "-k", label="x")
    ax.plot(t, r_OP[:, 1], "--k", label="y")
    ax.plot(t, r_OP[:, 2], "-.k", label="z")
    ax.set_xlabel("t")
    ax.set_ylabel("tip displacement")
    ax.grid()
    ax.legend()

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam], L, show=True)


def HeavyTopMaekinen2006():
    # Beam = TimoshenkoAxisAngle
    Beam = TimoshenkoAxisAngleSE3

    # number of elements
    nelements = 3

    # used polynomial degree
    polynomial_degree = 1

    # number of quadrature points
    # TODO: We have to distinguish between integration of the mass matrix,
    #       gyroscopic forces and potential forces!
    nquadrature_points = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
    # nquadrature_points = polynomial_degree + 2
    # nquadrature_points = polynomial_degree + 1 # this seems not to be sufficent for p > 1

    # used shape functions for discretization
    shape_functions = "B-spline"

    # beam parameters found in Section 4.3. Fast symmetrical top - Maekinen2006
    L = 1
    EA = GA = 1.0e6
    GJ = EI = 1.0e3

    # build quadratic material model
    Ei = np.array([EA, GA, GA], dtype=float)
    Fi = np.array([GJ, EI, EI], dtype=float)
    material_model = Simo1986(Ei, Fi)

    # cross section and inertia
    K_Theta = np.diag([1.0, 0.5, 0.5])
    A_rho0 = 13.5
    # r = sqrt(2 / 13.5)
    cross_section = UserDefinedCrossSection(A_rho0, A_rho0, np.zeros(3), K_Theta)

    # gravity
    # TODO
    mg = 40

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = rodriguez(0.3 * e1) @ rodriguez(-pi / 2 * e2)
    # A_IK0 = np.eye(3, dtype=float)
    K_omega_IK0 = 50 * e1

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
        r_OP0=r_OP0,
        A_IK0=A_IK0,
        v_P0=np.zeros(3, dtype=float),
        K_omega_IK0=K_omega_IK0,
    )

    # junction
    r_OB0 = np.zeros(3, dtype=float)
    frame = Frame(r_OP=r_OB0, A_IK=A_IK0)
    joint = SphericalJoint(frame, beam, r_OB0, frame_ID2=(0,))

    # gravity beam
    g = np.array(-cross_section.line_density * 9.81 * e3, dtype=float)
    f_g_beam = DistributedForce1D(lambda t, xi: g, beam)

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame)
    model.add(joint)
    model.add(f_g_beam)
    model.assemble()

    t1 = 1.5
    # t1 = 0.25
    # t1 = 0.01
    dt = 5.0e-3
    # dt = 2.5e-2
    method = "RK45"
    rtol = 1.0e-6
    atol = 1.0e-6
    rho_inf = 0.5

    solver = ScipyIVP(
        model, t1, dt, method=method, rtol=rtol, atol=atol
    )  # this is no good idea for complement rotation vectors!
    # solver = GenAlphaFirstOrder(model, t1, dt, rho_inf=rho_inf, tol=atol)
    # solver = GenAlphaDAEAcc(model, t1, dt, rho_inf=rho_inf, newton_tol=atol)
    # dt = 5.0e-3
    # solver = Moreau(model, t1, dt)

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    ############################
    # Visualize tip displacement
    ############################
    elDOF = beam.qDOF[beam.elDOF[-1]]
    r_OP = np.array([beam.r_OP(ti, qi[elDOF], (1,)) for (ti, qi) in zip(t, q)])
    A_IK = np.array([beam.A_IK(ti, qi[elDOF], (1,)) for (ti, qi) in zip(t, q)])
    from scipy.spatial.transform import Rotation

    Euler = np.array(
        [Rotation.from_matrix(A_IKi).as_euler("zxz", degrees=False) for A_IKi in A_IK]
    )

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(t, r_OP[:, 0], "-k", label="x")
    ax[0].plot(t, r_OP[:, 1], "--k", label="y")
    ax[0].plot(t, r_OP[:, 2], "-.k", label="z")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("tip displacement")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, Euler[:, 0], "-k", label="alpha")
    ax[1].plot(t, Euler[:, 1], "--k", label="beta")
    ax[1].plot(t, Euler[:, 2], "-.k", label="gamma")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("Euler angles")
    ax[1].grid()
    ax[1].legend()

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam], L, show=True)


if __name__ == "__main__":
    # run(statics=True)
    # run(statics=False)
    # locking()
    # SE3_interpolation()
    # HelixIbrahimbegovic1997()
    HeavyTopMaekinen2006()
