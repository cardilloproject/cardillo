from distutils.log import error
from cardillo.math import e1, e2, e3, sqrt, sin, cos, pi, smoothstep2, A_IK_basic
from cardillo.beams.spatial import (
    UserDefinedCrossSection,
    CircularCrossSection,
    RectangularCrossSection,
    QuadraticCrossSection,
    ShearStiffQuadratic,
    Simo1986,
)
from cardillo.math.SE3 import SE3, se3
from cardillo.math.algebra import ax2skew, skew2ax, norm, inv3D, cross3
from cardillo.math.rotations import (
    inverse_tangent_map,
    rodriguez,
    rodriguez_inv,
    tangent_map,
    A_IK_basic,
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
from cardillo.forces import Force, K_Force, K_Moment, Moment, DistributedForce1D
from cardillo.model import Model
from cardillo.solver import (
    Newton,
    ScipyIVP,
    GenAlphaFirstOrder,
    Moreau,
    Riks,
)

from cardillo.solver.Fsolve import Fsolve

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
    v_Q0=np.zeros(3, dtype=float),
    xi_Q=0.0,
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
        Q, u0 = TimoshenkoAxisAngleSE3.initial_configuration(
            p_r,
            p_psi,
            nelements,
            L,
            r_OP0=r_OP0,
            A_IK0=A_IK0,
            v_P0=v_Q0,
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
        Q, u0 = TimoshenkoQuarternionSE3.initial_configuration(
            p_r,
            p_psi,
            nelements,
            L,
            r_OP0=r_OP0,
            A_IK0=A_IK0,
            v_Q0=v_Q0,
            xi_Q=xi_Q,
            K_omega_IK0=K_omega_IK0,
            basis=shape_functions,
        )
    else:
        raise NotImplementedError("")

    # Initial configuration coincides with reference configuration.
    # Note: This might be adapted.
    q0 = Q.copy()

    # extract cross section properties
    # TODO: Maybe we should pass this to the beam model itself?
    area = cross_section.area
    density = cross_section.density
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
    A_rho0 = density * area
    B_rho0 = density * first_moment
    C_rho0 = density * second_moment
    # TODO: I think this is Binet's inertia tensor!
    # TODO: See Mäkinen2006, (24) on page 1022 for a clarification of the
    # classical inertia tensor
    C_rho0 = np.zeros((3, 3))
    for a in range(1, 3):
        for b in range(1, 3):
            C_rho0[a, b] = density * second_moment[b, a]

    # This is the standard second moment of area weighted by a constant line
    # density
    K_S_rho0 = density * first_moment
    K_I_rho0 = density * second_moment

    ##################
    # build beam model
    ##################
    if Beam == TimoshenkoAxisAngle:
        beam = TimoshenkoAxisAngle(
            material_model,
            A_rho0,
            K_I_rho0,
            p_r,
            p_psi,
            nquadrature_points,
            nelements,
            Q=Q,
            q0=q0,
            # u0=u0,
            basis=shape_functions,
        )
    elif Beam == TimoshenkoAxisAngleSE3:
        beam = TimoshenkoAxisAngleSE3(
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
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
            K_I_rho0,
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
            K_I_rho0,
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
    g = np.array([0, 0, -cross_section.area * cross_section.density * 9.81 * 1.0e-1])
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


# def locking(case="quater_circle"):
# def locking(case="full_circle"):
def locking(case="helix"):
    """This example examines shear and membrande locking as done in Meier2015.

    References:
    ===========
    Meier2015: https://doi.org/10.1016/j.cma.2015.02.029
    """
    # Beam = TimoshenkoAxisAngle
    Beam = TimoshenkoAxisAngleSE3
    # Beam = TimoshenkoQuarternionSE3

    if case == "quater_circle":
        # number of elements
        nelements = 1

        # L = 1.0e3
        # # (slenderness, atol, n_load_steps
        # # triplet = (1.0e1, 1.0e-7, 2)
        # # triplet = (1.0e2, 1.0e-9, 2) #, 50)
        # # triplet = (1.0e3, 1.0e-10, 2) #, 100) # this is working with quaternions
        # triplet = (1.0e4, 1.0e-13, 2) #, 200)
        # # triplet = (1.0e5, 1.0e-14, 2) #, 200)

        L = 1.0e4
        # (slenderness, atol, n_load_steps
        # fmt: off
        # triplet = (1.0e1, 1.0e-6, 2)
        triplet = (1.0e2, 1.0e-8, 2)
        # triplet = (1.0e3, 1.0e-9, 2)
        # triplet = (1.0e4, 1.0e-12, 2)  # this can't be sovle using the quaternion implementation
        # triplet = (1.0e5, 1.0e-12, 4)
        # fmt: on

    elif case == "full_circle":
        # number of elements
        nelements = 1
        raise NotImplementedError
    elif case == "helix":
        # number of elements
        nelements = 3

        # (slenderness, atol, n_load_steps
        # triplet = (1.0e1, 1.0e-6, 50)
        triplet = (1.0e2, 1.0e-9, 2)  # , 50)
        # triplet = (1.0e3, 1.0e-10, 2)  # , 100) # this is working with quaternions
        # triplet = (1.0e4, 1.0e-13, 2) #, 200)
        # triplet = (1.0e5, 1.0e-14, 2) #, 200)
    else:
        raise NotImplementedError

    slenderness, atol, n_load_steps = triplet
    n_load_steps = 5

    # used polynomial degree
    # polynomial_degree = 1
    polynomial_degree = 2

    # number of quadrature points
    nquadrature_points = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
    print(f"nquadrature_points: {nquadrature_points}")

    # used shape functions for discretization
    # shape_functions = "B-spline"
    shape_functions = "Lagrange"

    # used cross section
    width = L / slenderness

    # cross section
    line_density = 1
    cross_section = QuadraticCrossSection(line_density, width)

    # Young's and shear modulus
    E = 1.0  # Meier2015
    G = 0.5  # Meier2015

    # build quadratic material model
    material_model = quadratic_beam_material(E, G, cross_section, Beam)
    print(f"Ei: {material_model.Ei}")
    print(f"Fi: {material_model.Fi}")

    # exit()

    # starting point and orientation of initial point, initial length
    r_OP = np.zeros(3)
    A_IK = np.eye(3)

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
    A_IK0 = np.eye(3)
    frame1 = Frame(r_OP=r_OB0, A_IK=A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, beam, r_OB0, frame_ID2=(0,))

    # moment at right end
    Fi = material_model.Fi
    if case == "quater_circle":
        M = lambda t: (e3 * Fi[2]) * t * 2 * np.pi / L * 0.25
    elif case == "full_circle":
        M = lambda t: (e3 * Fi[2]) * t * 2 * np.pi / L
        raise NotImplementedError
    elif case == "helix":
        M = lambda t: (e1 * Fi[0] + e3 * Fi[2]) * t * 2 * np.pi / L
    moment = K_Moment(M, beam, (1,))

    if case == "quater_circle":
        f = lambda t: t * 1e-4 * Fi[1] / L * e3
        force = Force(f, beam, frame_ID=(1,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    model.add(moment)
    if case == "quater_circle":
        model.add(force)
    model.assemble()

    # build different tolerances for static equilibrium and constraint forces
    # atol_u = np.ones(model.nu, dtype=float) * atol
    # atol_la_g = np.ones(model.nla_g, dtype=float) * 1.0e-12
    # atol_la_S = np.ones(model.nla_S, dtype=float) * 1.0e-12
    # atol = np.concatenate((atol_u, atol_la_g, atol_la_S))
    # atol = 1.0e-10
    # rtol = atol * 1e2
    rtol = 0
    # rtol = 1.0e-8

    # define constraint degrees of freedom
    if Beam == TimoshenkoAxisAngleSE3:
        cDOF_q = np.concatenate(
            [np.arange(3, dtype=int), np.arange(3, dtype=int) + beam.nq_r]
        )
        cDOF_u = cDOF_q
        cDOF_S = np.array([], dtype=int)
        b = lambda t: np.concatenate(
            [np.zeros(3, dtype=float), np.zeros(3, dtype=float)]
        )
    elif Beam == TimoshenkoQuarternionSE3:
        cDOF_q = np.concatenate(
            [np.arange(3, dtype=int), np.arange(4, dtype=int) + beam.nq_r]
        )
        cDOF_u = np.concatenate(
            [np.arange(3, dtype=int), np.arange(3, dtype=int) + beam.nu_r]
        )
        cDOF_S = np.array([0], dtype=int)
        b = lambda t: np.concatenate(
            [np.zeros(3, dtype=float), np.array([1, 0, 0, 0], dtype=float)]
        )
    else:
        cDOF_q = np.array([], dtype=int)
        cDOF_u = np.array([], dtype=int)
        cDOF_S = np.array([], dtype=int)
        b = lambda t: np.array([], dtype=float)
    cDOF_q = np.array([], dtype=int)
    cDOF_u = np.array([], dtype=int)
    cDOF_S = np.array([], dtype=int)
    b = lambda t: np.array([], dtype=float)

    solver = Newton(
        model,
        cDOF_q=cDOF_q,
        cDOF_u=cDOF_u,
        cDOF_S=cDOF_S,
        b=b,
        n_load_steps=n_load_steps,
        max_iter=100,
        atol=atol,
        rtol=rtol,
    )
    # solver = Fsolve(model, n_load_steps=n_load_steps, atol=atol, rtol=rtol)

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    ##################################
    # visualize nodal rotation vectors
    ##################################
    fig, ax = plt.subplots()

    # visualize tip rotation vector
    psi1 = q[:, beam.qDOF[beam.nodalDOF_psi[-1]]]
    for i, psi in enumerate(psi1.T):
        ax.plot(t, psi, label=f"psi1_{i}")

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
    Beam = TimoshenkoAxisAngleSE3
    # Beam = TimoshenkoDirectorDirac
    # Beam = TimoshenkoQuarternionSE3

    # fraction of 10 full rotations and the out of plane force
    # a corresponding fraction of 100 elements is chosen
    # # fraction = 0.05
    # fraction = 0.1  # 1 full rotations
    # fraction = 0.20  # 2 full rotations
    # fraction = 0.4  # 4 full rotations
    # fraction = 0.5  # 5 full rotations
    fraction = 1  # 10 full rotations

    # number of elements
    nelements_max = 30
    # nelements_max = 100 # Ibrahimbegovic1997
    nelements = max(3, int(fraction * nelements_max))
    print(f"nelemen: {nelements}")

    # used polynomial degree
    polynomial_degree = 1

    # number of quadrature points
    # TODO: We have to distinguish between integration of the mass matrix,
    #       gyroscopic forces and potential forces!
    nquadrature_points = int(np.ceil((polynomial_degree + 1) ** 2 / 2))

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
    n_load_steps = int(200 * 10 * fraction)  # works for all cases

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
        frame_ID = (1,)
        elDOF = beam.qDOF_P(frame_ID)
        r_OC_L = np.array(
            [beam.r_OP(ti, qi[elDOF], frame_ID) for (ti, qi) in zip(t, q)]
        )
        # export_data = np.vstack([t, *r_OC_L.T]).T
        export_data = np.vstack([np.arange(nt), *r_OC_L.T]).T
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
        # nframes = 17
        nframes = 37
        idxs = np.linspace(0, nt - 1, num=nframes, dtype=int)
        for i, idx in enumerate(idxs):
            ti = t[idx]
            qi = q[idx]
            r_OPs = beam.centerline(qi, n=500)

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


def HeavyTop():
    class RigidHeavyTopODE:
        def __init__(self, m, r, l, g):
            self.m = m
            self.r = r
            self.l = l
            self.g = g
            self.K_r_PS = np.array([l / 2, 0, 0], dtype=float)
            self.A = (1.0 / 2.0) * m * r**2
            self.B = (1.0 / 4.0) * m * r**2 + (1.0 / 12.0) * m * l**2
            self.K_Theta_S = np.diag([self.A, self.B, self.B])
            self.K_Theta_P = (
                self.K_Theta_S + m * ax2skew(self.K_r_PS) @ ax2skew(self.K_r_PS).T
            )
            self.K_Theta_P_inv = inv3D(self.K_Theta_P)

        def A_IK(self, t, q):
            alpha, beta, gamma = q
            # z, y, x
            # A_IB = A_IK_basic(alpha).z()
            # A_BC = A_IK_basic(beta).y()
            # A_CK = A_IK_basic(gamma).x()
            # return A_IB @ A_BC @ A_CK
            sa, ca = sin(alpha), cos(alpha)
            sb, cb = sin(beta), cos(beta)
            sg, cg = sin(gamma), cos(gamma)
            # fmt: off
            return np.array([
                [ca * cb, ca * sb  *sg - sa * cg, ca * sb * cg + sa * sg],
                [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
                [    -sb,                cb * sg,                cb * cg],
            ])
            # fmt: on

        def r_OP(self, t, q, K_r_SP=np.zeros(3, dtype=float)):
            A_IK = self.A_IK(t, q)
            r_OS = A_IK @ self.K_r_PS
            return r_OS + A_IK @ K_r_SP

        def __call__(self, t, x):
            dx = np.zeros(6, dtype=float)
            alpha, beta, gamma = x[:3]
            K_Omega = x[3:]

            m = self.m
            l = self.l
            g = self.g
            A = self.A
            B = self.B

            sb, cb = sin(beta), cos(beta)
            sg, cg = sin(gamma), cos(gamma)

            # z, y, x
            # fmt: off
            Q = np.array([
                [    -sb, 0.0, 1.0], 
                [cb * sg,  cg, 0.0], 
                [cb * cg, -sg, 0.0]
            ], dtype=float)
            # fmt: on

            f_gyr = cross3(K_Omega, self.K_Theta_P @ K_Omega)
            K_J_S = -ax2skew(self.K_r_PS)
            I_J_S = self.A_IK(t, x[:3]) @ K_J_S
            f_pot = I_J_S.T @ (-self.m * self.g * e3)
            h = f_pot - f_gyr

            dx[:3] = inv3D(Q) @ x[3:]
            dx[3:] = self.K_Theta_P_inv @ h

            return dx

    # Beam = TimoshenkoAxisAngle
    Beam = TimoshenkoAxisAngleSE3
    # Beam = TimoshenkoQuarternionSE3

    # number of elements
    nelements = 1

    # used polynomial degree
    polynomial_degree = 1

    # number of quadrature points
    nquadrature_points = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
    print(f"number of quadrature points: {nquadrature_points}")

    # used shape functions for discretization
    shape_functions = "Lagrange"

    ######################
    # nice locking results
    ######################
    g = 9.81
    l = 0.5
    r = 0.1
    omega_x = 2 * pi * 50
    E_stiff = 210e6  # steel (stiff beam)
    # E_soft = E_stiff * 1.0e-2 # soft beam
    # E_soft = E_stiff * 5.0e-3 # soft beam # looks good but maybe even softer
    # E_soft = E_stiff * 2.0e-3 # soft beam
    E_soft = E_stiff * 1.0e-3  # soft beam
    rho = 8000  # steel [kg/m^3]

    # ####################
    # # more beam like top
    # ####################
    # g = 9.81
    # l = 1.0
    # r = 0.1
    # omega_x = 400 * pi
    # E_stiff = 210e6 # steel [N/m^2] (stiff beam)
    # E_soft = E_stiff * 5.0e-3 # soft beam
    # rho = 8000 # steel [kg/m^3]

    # tip volume and mass
    V = l * pi * r**2
    m = rho * V
    print(f"total mass: {m}")
    cross_section = CircularCrossSection(rho, r)

    # initial angular velocity and orientation
    A = 0.5 * m * r**2
    # omega_pr = m * g * (0.5 * l) / (A * omega_x)
    omega_pr = g * l / (r**2 * omega_x)
    K_omega_IK0 = omega_x * e1 + omega_pr * e3  # perfect precession motion
    A_IK0 = np.eye(3, dtype=float)
    # A_IK0 = rodriguez(-pi / 10 * e2)
    from scipy.spatial.transform import Rotation

    angles0 = Rotation.from_matrix(A_IK0).as_euler("zyx")

    # starting point
    r_OP0 = np.zeros(3, dtype=float)

    ###################
    # solver parameters
    ###################
    t0 = 0.0
    t1 = 2.0 * pi / omega_pr
    # t1 *= 0.5
    # t1 *= 0.25
    # t1 *= 0.125
    # t1 *= 0.075
    t1 *= 0.01

    # nt = np.ceil(t1 / 1.0e-3)
    # dt = t1 * 1.0e-2
    dt = t1 * 1.0e-3
    rtol = 1.0e-8
    atol = 1.0e-8

    t_eval = np.arange(t0, t1 + dt, dt)

    def solve(E):
        nu = 1.0 / 3.0
        G = E / (2.0 * (1.0 + nu))  # TODO: Use G = E * 3/4 in paper?

        A = cross_section.area
        Ip, I2, I3 = np.diag(cross_section.second_moment)
        Ei = np.array([E * A, G * A, G * A])
        Fi = np.array([G * Ip, E * I2, E * I3])
        material_model = Simo1986(Ei, Fi)

        # build beam model
        beam = beam_factory(
            nelements,
            polynomial_degree,
            nquadrature_points,
            shape_functions,
            cross_section,
            material_model,
            Beam,
            l,
            r_OP0=r_OP0,
            A_IK0=A_IK0,
            v_Q0=np.zeros(3, dtype=float),
            K_omega_IK0=K_omega_IK0,
        )

        # junction
        r_OB0 = np.zeros(3, dtype=float)
        frame = Frame(r_OP=r_OB0, A_IK=A_IK0)
        joint = SphericalJoint(frame, beam, r_OB0, frame_ID2=(0,))

        # gravity beam
        vg = np.array(-cross_section.area * cross_section.density * g * e3, dtype=float)
        f_g_beam = DistributedForce1D(lambda t, xi: vg, beam)

        # assemble the model
        model = Model()
        model.add(beam)
        model.add(frame)
        model.add(joint)
        model.add(f_g_beam)
        model.assemble()

        sol = ScipyIVP(model, t1, dt, method="RK23", rtol=rtol, atol=atol).solve()
        # sol = ScipyIVP(model, t1, dt, method="RK45", rtol=rtol, atol=atol).solve()

        return beam, sol

    # compute reference solution using Euler top equations
    heavy_top = RigidHeavyTopODE(m, r, l, g)
    x0 = np.concatenate((angles0, K_omega_IK0))
    from scipy.integrate import solve_ivp

    ref = solve_ivp(
        heavy_top,
        (t_eval[0], t_eval[-1]),
        x0,
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    t_ref = ref.t
    angles_ref = ref.y[:3].T
    r_OP_ref = np.array(
        [
            heavy_top.r_OP(ti, qi, K_r_SP=np.array([l / 2, 0, 0]))
            for (ti, qi) in zip(t_ref, angles_ref)
        ]
    )

    # # debug rigid top solution
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot(*r_OP_ref.T)
    # plt.show()
    # exit()

    # sove for beam solutions
    beam_stiff, sol_stiff = solve(E_stiff)
    beam_soft, sol_soft = solve(E_soft)

    q_stiff = sol_stiff.q
    nt = len(q_stiff)
    t = sol_stiff.t[:nt]
    q_soft = sol_soft.q
    nt = len(q_soft)
    t = sol_soft.t[:nt]

    if True:
        # ###############################
        # # visualize tip rotation vector
        # ###############################
        # fig, ax = plt.subplots()

        # psi1 = q[:, beam.qDOF[beam.nodalDOF_psi[-1]]]
        # for i, psi in enumerate(psi1.T):
        #     ax.plot(t, psi, label=f"psi1_{i}")

        # ax.set_xlabel("t")
        # ax.set_ylabel("nodal rotation vectors")
        # ax.grid()
        # ax.legend()

        ############################
        # Visualize tip displacement
        ############################
        elDOF = beam_stiff.qDOF[beam_stiff.elDOF[-1]]
        r_OP_stiff = np.array(
            [beam_stiff.r_OP(ti, qi[elDOF], (1,)) for (ti, qi) in zip(t, q_stiff)]
        )
        elDOF = beam_soft.qDOF[beam_soft.elDOF[-1]]
        r_OP_soft = np.array(
            [beam_soft.r_OP(ti, qi[elDOF], (1,)) for (ti, qi) in zip(t, q_soft)]
        )

        fig = plt.figure(figsize=(10, 8))

        # 3D tracjectory of tip displacement
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.set_title("3D tip trajectory")
        ax.plot(*r_OP_ref.T, "-k", label="rigid body")
        ax.plot(*r_OP_stiff.T, "--r", label="stiff beam")
        ax.plot(*r_OP_soft.T, "--b", label="soft beam")
        ax.set_xlabel("x [-]")
        ax.set_ylabel("y [-]")
        ax.set_zlabel("z [-]")
        ax.grid()
        ax.legend()

        # tip displacement
        ax = fig.add_subplot(1, 2, 2)

        ax.set_title("tip displacement (components)")

        ax.plot(t, r_OP_ref[:, 0], "-k", label="x_ref")
        ax.plot(t, r_OP_ref[:, 1], "-k", label="y_ref")
        ax.plot(t, r_OP_ref[:, 2], "-k", label="z_ref")
        ax.plot(t, r_OP_stiff[:, 0], "--r", label="x stiff")
        ax.plot(t, r_OP_stiff[:, 1], "--r", label="y stiff")
        ax.plot(t, r_OP_stiff[:, 2], "--r", label="z stiff")
        ax.plot(t, r_OP_soft[:, 0], "--b", label="x soft")
        ax.plot(t, r_OP_soft[:, 1], "--b", label="y soft")
        ax.plot(t, r_OP_soft[:, 2], "--b", label="z soft")
        ax.set_xlabel("t")
        ax.set_ylabel("x, y")
        ax.grid()
        ax.legend()

        # ax2 = ax.twinx()
        # ax2.plot(t, r_OP_ref[:, 2], "-k", label="z_ref")
        # ax2.plot(t, r_OP_stiff[:, 2], "--r", label="z stiff")
        # ax2.plot(t, r_OP_soft[:, 2], "--b", label="z soft")
        # ax2.set_ylabel("z")
        # ax2.grid()
        # ax2.legend()

        fig.tight_layout()

        plt.show()

        # ###########
        # # animation
        # ###########
        # animate_beam(t, q, [beam], l, show=True)

    # exit()

    #######################
    # export tip deflection
    #######################
    # rigid body
    header = "t, x, y, z"
    export_data = np.vstack([t_ref, *r_OP_ref.T]).T
    np.savetxt(
        "results/HeavyTopRigidBody.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    # beam's
    frame_ID = (1,)
    elDOF = beam_stiff.qDOF_P(frame_ID)
    r_OC_L_soft = np.array(
        [beam_soft.r_OP(ti, qi[elDOF], frame_ID) for (ti, qi) in zip(t, q_soft)]
    )
    r_OC_L_stiff = np.array(
        [beam_stiff.r_OP(ti, qi[elDOF], frame_ID) for (ti, qi) in zip(t, q_stiff)]
    )
    header = "t, x_stiff, y_stiff, z_stiff, x_soft, y_soft, z_soft"
    export_data = np.vstack([t, *r_OC_L_stiff.T, *r_OC_L_soft.T]).T
    np.savetxt(
        "results/HeavyTopBeam.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )


def Dschanibekow():
    # Beam = TimoshenkoAxisAngle
    # Beam = TimoshenkoAxisAngleSE3
    Beam = TimoshenkoQuarternionSE3

    # number of elements
    nelements = 2
    # nelements = 1

    # used polynomial degree
    polynomial_degree = 1

    # number of quadrature points
    # TODO: We have to distinguish between integration of the mass matrix,
    #       gyroscopic forces and potential forces!
    nquadrature_points = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
    # nquadrature_points = polynomial_degree + 2
    # nquadrature_points = polynomial_degree + 1
    # nquadrature_points = polynomial_degree

    # used shape functions for discretization
    shape_functions = "B-spline"

    # beam parameters found in Section 4.3. Fast symmetrical top - Maekinen2006
    # TODO: We have to find good values for this. Maybe the stiff and soft
    #       values used by Maekinen2006?
    EA = GA = 1.0e6
    GJ = EI = 1.0e3

    # build quadratic material model
    Ei = np.array([EA, GA, GA], dtype=float)
    Fi = np.array([GJ, EI, EI], dtype=float)
    material_model = Simo1986(Ei, Fi)

    # cross section and inertia
    # K_Theta = np.diag([1.0, 0.5, 0.5])
    # A_rho0 = 13.5
    # # r = sqrt(2 / 13.5)
    # cross_section = UserDefinedCrossSection(
    #     A_rho0, A_rho0, np.zeros(3, dtype=float), K_Theta
    # )
    line_density = 1
    # This set works with K_omega_IK0 = 100 * e2 and t1 = 2
    # l = 2
    # w = 1.5
    # h = 1
    # This set works with K_omega_IK0 = 50 * e1 and t1 = 2
    l = 1.5
    w = 2
    h = 1
    cross_section = RectangularCrossSection(line_density, w, h)

    m = line_density * l * w * h
    Theta_xx = (m / 12.0) * (w**2 + h**2)
    Theta_yy = (m / 12.0) * (l**2 + h**2)
    Theta_zz = (m / 12.0) * (l**2 + w**2)
    print(f"Theta_xx: {Theta_xx}")
    print(f"Theta_yy: {Theta_yy}")
    print(f"Theta_zz: {Theta_zz}")

    # exit()

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = rodriguez(-pi / 2 * e2)
    # A_IK0 = rodriguez(0.3 * e1) @ rodriguez(-pi / 2 * e2)
    # A_IK0 = rodriguez(-pi / 10 * e2) # TODO: Find a good initial pertubation!
    # A_IK0 = rodriguez(pi / 20 * e1) # TODO: Find a good initial pertubation!
    # A_IK0 = np.eye(3, dtype=float)
    K_omega_IK0 = 50 * e1
    # K_omega_IK0 = 10 * e2 #+ 1. * e1
    # K_omega_IK0 = 100 * e2 #+ 1. * e1

    # build beam model
    beam = beam_factory(
        nelements,
        polynomial_degree,
        nquadrature_points,
        shape_functions,
        cross_section,
        material_model,
        Beam,
        l,
        r_OP0=r_OP0,
        A_IK0=A_IK0,
        v_Q0=np.zeros(3, dtype=float),
        xi_Q=0.5,
        K_omega_IK0=K_omega_IK0,
    )

    # assemble the model
    model = Model()
    model.add(beam)
    model.assemble()

    t1 = 2
    # t1 = 5
    dt = 1.0e-2
    # method = "RK45"
    method = "RK23"  # performs better (stiff beam example?)
    rtol = 1.0e-5
    atol = 1.0e-5
    rho_inf = 0.5

    solver = ScipyIVP(model, t1, dt, method=method, rtol=rtol, atol=atol)
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
        [
            # Rotation.from_matrix(A_IK0.T @ A_IKi).as_euler("xyz", degrees=False) for A_IKi in A_IK
            Rotation.from_matrix(A_IKi).as_euler("xyz", degrees=False)
            for A_IKi in A_IK
        ]  # Cardan angles
        # [Rotation.from_matrix(A_IKi).as_euler("zxz", degrees=False) for A_IKi in A_IK] # Euler angles
    )

    fig = plt.figure(figsize=(10, 8))

    # 3D tracjectory of tip displacement
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.set_title("3D tip trajectory")
    ax.plot(*r_OP.T, "-k")
    ax.set_xlabel("x [-]")
    ax.set_ylabel("y [-]")
    ax.set_zlabel("z [-]")
    ax.grid()

    # tip displacement
    ax = fig.add_subplot(2, 3, 3)
    ax.set_title("tip displacement (components)")
    ax.plot(t, r_OP[:, 0], "-k", label="x")
    ax.plot(t, r_OP[:, 1], "--k", label="y")
    ax.plot(t, r_OP[:, 2], "-.k", label="z")
    ax.set_xlabel("t")
    ax.grid()
    ax.legend()

    # Euler angles
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(t, Euler[:, 0], "-b")
    ax.set_xlabel("t")
    ax.set_ylabel("alpha")
    ax.grid()

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(t, Euler[:, 1], "-b")
    ax.set_xlabel("t")
    ax.set_ylabel("beta")
    ax.grid()

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(t, Euler[:, 2], "-b")
    ax.set_xlabel("t")
    ax.set_ylabel("gamma")
    ax.grid()

    fig.tight_layout()

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam], l, show=True)


def distributed_force():
    """This example examines the applicability of line force loads that should
    lead to non constant shear strains!
    """
    # Beam = TimoshenkoAxisAngle
    Beam = TimoshenkoAxisAngleSE3
    # Beam = TimoshenkoQuarternionSE3

    # number of elements
    nelements = 6

    # total beam length
    L = 1.0e3
    # L = 1.0e4

    # (slenderness, atol, n_load_steps
    triplet = (1.0e1, 1.0e-6, 10)
    # triplet = (1.0e2, 1.0e-8, 2)
    # triplet = (1.0e3, 1.0e-9, 2)
    # triplet = (1.0e4, 1.0e-12, 2) # this can't be sovle using the quaternion implementation
    # triplet = (1.0e5, 1.0e-12, 4)

    slenderness, atol, n_load_steps = triplet

    # used polynomial degree
    polynomial_degree = 1

    # number of quadrature points
    nquadrature_points = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
    print(f"nquadrature_points: {nquadrature_points}")

    # used shape functions for discretization
    # shape_functions = "B-spline"
    shape_functions = "Lagrange"

    # used cross section
    width = L / slenderness

    # cross section
    line_density = 1
    cross_section = QuadraticCrossSection(line_density, width)

    # Young's and shear modulus
    E = 1.0  # Meier2015
    G = 0.5  # Meier2015

    # build quadratic material model
    material_model = quadratic_beam_material(E, G, cross_section, Beam)
    print(f"Ei: {material_model.Ei}")
    print(f"Fi: {material_model.Fi}")

    # starting point and orientation of initial point, initial length
    r_OP = np.zeros(3)
    A_IK = np.eye(3)

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
    A_IK0 = np.eye(3)
    frame1 = Frame(r_OP=r_OB0, A_IK=A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, beam, r_OB0, frame_ID2=(0,))

    b = lambda t, xi: -t * e3 * 1.0e-1
    force = DistributedForce1D(b, beam)

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    model.add(force)
    model.assemble()

    # build different tolerances for static equilibrium and constraint forces
    # atol_u = np.ones(model.nu, dtype=float) * atol
    # atol_la_g = np.ones(model.nla_g, dtype=float) * 1.0e-12
    # atol_la_S = np.ones(model.nla_S, dtype=float) * 1.0e-12
    # atol = np.concatenate((atol_u, atol_la_g, atol_la_S))
    # atol = 1.0e-10
    # rtol = atol * 1e2
    rtol = 0
    # rtol = 1.0e-8

    # define constraint degrees of freedom
    if Beam == TimoshenkoAxisAngleSE3:
        cDOF_q = np.concatenate(
            [np.arange(3, dtype=int), np.arange(3, dtype=int) + beam.nq_r]
        )
        cDOF_u = cDOF_q
        cDOF_S = np.array([], dtype=int)
        b = lambda t: np.concatenate(
            [np.zeros(3, dtype=float), np.zeros(3, dtype=float)]
        )
    elif Beam == TimoshenkoQuarternionSE3:
        cDOF_q = np.concatenate(
            [np.arange(3, dtype=int), np.arange(4, dtype=int) + beam.nq_r]
        )
        cDOF_u = np.concatenate(
            [np.arange(3, dtype=int), np.arange(3, dtype=int) + beam.nu_r]
        )
        cDOF_S = np.array([0], dtype=int)
        b = lambda t: np.concatenate(
            [np.zeros(3, dtype=float), np.array([1, 0, 0, 0], dtype=float)]
        )
    else:
        cDOF_q = np.array([], dtype=int)
        cDOF_u = np.array([], dtype=int)
        cDOF_S = np.array([], dtype=int)
        b = lambda t: np.array([], dtype=float)
    cDOF_q = np.array([], dtype=int)
    cDOF_u = np.array([], dtype=int)
    cDOF_S = np.array([], dtype=int)
    b = lambda t: np.array([], dtype=float)

    solver = Newton(
        model,
        cDOF_q=cDOF_q,
        cDOF_u=cDOF_u,
        cDOF_S=cDOF_S,
        b=b,
        n_load_steps=n_load_steps,
        max_iter=100,
        atol=atol,
        rtol=rtol,
    )
    # solver = Fsolve(model, n_load_steps=n_load_steps, atol=atol, rtol=rtol)

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    ##################################
    # visualize nodal rotation vectors
    ##################################
    fig, ax = plt.subplots()

    # visualize tip rotation vector
    psi1 = q[:, beam.qDOF[beam.nodalDOF_psi[-1]]]
    for i, psi in enumerate(psi1.T):
        ax.plot(t, psi, label=f"psi1_{i}")

    ax.set_xlabel("t")
    ax.set_ylabel("nodal rotation vectors")
    ax.grid()
    ax.legend()

    ################################
    # visualize norm strain measures
    ################################
    nxi = 1000
    xis = np.linspace(0, 1, num=nxi)

    K_Gamma = np.zeros((3, nxi))
    K_Kappa = np.zeros((3, nxi))
    for i in range(nxi):
        frame_ID = (xis[i],)
        elDOF = beam.qDOF_P(frame_ID)
        qe = q[-1, beam.qDOF][elDOF]
        _, _, K_Gamma[:, i], K_Kappa[:, i] = beam.eval(qe, xis[i])

    fig, ax = plt.subplots(2, 3)

    for i in range(3):
        ax[0, i].set_title(f"K_Gamma{i}")
        ax[0, i].plot(xis, K_Gamma[i], "-k", label=f"K_Gamma{i}")
        ax[0, i].grid()

        ax[1, i].set_title(f"K_Kappa{i}")
        ax[1, i].plot(xis, K_Kappa[i], "-k", label=f"K_Kappa{i}")
        ax[1, i].grid()

    # ax[0].plot(xis, K_Gamma[1], "-g", label="K_Gamma1")
    # ax[0].plot(xis, K_Gamma[2], "-b", label="K_Gamma2")
    # ax[0].grid()
    # ax[0].legend()
    # ax[1].plot(xis, K_Kappa[0], "-r", label="K_Kappa0")
    # ax[1].plot(xis, K_Kappa[1], "-g", label="K_Kappa1")
    # ax[1].plot(xis, K_Kappa[2], "-b", label="K_Kappa2")
    # ax[1].grid()
    # ax[1].legend()

    # fig = plt.figure()

    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.set_title("axial and shear strains")
    # ax1.plot(xis, K_Gamma[0], "-b", label="K_Gamma0")
    # ax1.grid()

    # ax2 = ax1.twinx()
    # ax2.plot(xis, K_Gamma[1], "-k", label="K_Gamma1")
    # ax1.plot(xis, K_Gamma[2], "--k", label="K_Gamma2")

    # ax3 = ax1.twinx()
    # ax3.plot(xis, K_Gamma[2], "-.b", label="K_Gamma2")

    fig.tight_layout()

    # ############################
    # # Visualize potential energy
    # ############################
    # E_pot = np.array([model.E_pot(ti, qi) for (ti, qi) in zip(t, q)])

    # fig, ax = plt.subplots(1, 2)

    # ax[0].plot(t, E_pot)
    # ax[0].set_xlabel("t")
    # ax[0].set_ylabel("E_pot")
    # ax[0].grid()

    # # visualize final centerline projected in all three planes
    # r_OPs = beam.centerline(q[-1])
    # fig, ax = plt.subplots(1, 3)
    # ax[0].plot(r_OPs[0, :], r_OPs[1, :], label="x-y")
    # ax[1].plot(r_OPs[1, :], r_OPs[2, :], label="y-z")
    # ax[2].plot(r_OPs[2, :], r_OPs[0, :], label="z-x")
    # ax[0].grid()
    # ax[0].legend()
    # ax[0].set_aspect(1)
    # ax[1].grid()
    # ax[1].legend()
    # ax[1].set_aspect(1)
    # ax[2].grid()
    # ax[2].legend()
    # ax[2].set_aspect(1)

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam], L, show=True)


def objectivity_quater_circle():
    """This example examines shear and membrande locking as done in Meier2015.

    References:
    ===========
    Meier2015: https://doi.org/10.1016/j.cma.2015.02.029
    """
    # Beam = TimoshenkoAxisAngle
    Beam = TimoshenkoAxisAngleSE3
    # Beam = TimoshenkoQuarternionSE3

    # number of elements
    nelements = 1
    # nelements = 4
    # nelements = 8
    # nelements = 16
    # nelements = 32
    # nelements = 64

    # L = 1.0e3
    # # (slenderness, atol, n_load_steps
    # # triplet = (1.0e1, 1.0e-7, 2)
    # # triplet = (1.0e2, 1.0e-9, 2) #, 50)
    # # triplet = (1.0e3, 1.0e-10, 2) #, 100) # this is working with quaternions
    # triplet = (1.0e4, 1.0e-13, 2) #, 200)
    # # triplet = (1.0e5, 1.0e-14, 2) #, 200)

    L = 1.0e4
    # (slenderness, atol, n_load_steps
    # triplet = (1.0e1, 1.0e-7, 2)
    # triplet = (1.0e2, 1.0e-8, 2)
    # triplet = (1.0e3, 1.0e-9, 2)
    # triplet = (1.0e4, 1.0e-10, 2) # this can't be sovle using the quaternion implementation
    # triplet = (1.0e5, 1.0e-11, 4)

    # convergence for 64 elements
    # TODO: Maybe use L = 1.0e3 here since the slenderness = 1.0e1 solution
    # differs from the other ones?
    triplet = (1.0e1, 1.0e-5, 30)
    # triplet = (1.0e2, 1.0e-7, 5)
    # triplet = (1.0e3, 1.0e-9, 5)
    # triplet = (1.0e4, 1.0e-11, 5)
    # triplet = (1.0e5, 1.0e-12, 5)

    slenderness, atol, n_load_steps = triplet
    rtol = 0
    n = 10  # number of full rotations
    t_star = 0.1  # fraction of deformation pseudo time
    # t_star = 0.05  # fraction of deformation pseudo time
    # n_load_steps = 100
    # n_load_steps = 200
    n_load_steps = 500

    # used polynomial degree
    polynomial_degree = 1

    # number of quadrature points
    nquadrature_points = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
    print(f"nquadrature_points: {nquadrature_points}")

    # used shape functions for discretization
    shape_functions = "Lagrange"

    # used cross section
    width = L / slenderness

    # cross section
    line_density = 1
    cross_section = QuadraticCrossSection(line_density, width)

    # Young's and shear modulus
    E = 1.0  # Meier2015
    G = 0.5  # Meier2015

    # build quadratic material model
    material_model = quadratic_beam_material(E, G, cross_section, Beam)
    print(f"Ei: {material_model.Ei}")
    print(f"Fi: {material_model.Fi}")

    # starting point and orientation of initial point, initial length
    r_OB0 = np.zeros(3, dtype=float)

    def A_IK0(t):
        # return np.eye(3)
        phi = (
            n * np.heaviside(t - t_star, 1.0) * (t - t_star) / (1.0 - t_star) * 2.0 * pi
        )
        return A_IK_basic(phi).x()  # @ A_IK_basic(phi).y()

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
        r_OP0=r_OB0,
        A_IK0=A_IK0(0),
    )

    # junctions
    frame1 = Frame(r_OP=r_OB0, A_IK=A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, beam, r_OB0, frame_ID2=(0,))

    # moment at the beam's tip
    Fi = material_model.Fi

    def M(t):
        M_max = (e3 * Fi[2]) * 2 * np.pi / L * 0.25
        if t <= t_star:
            return t / t_star * M_max
        else:
            return M_max

    moment = K_Moment(M, beam, (1,))

    # force at the beam's tip
    def f(t):
        f_max = 1e-4 * Fi[2] / L * e3
        if t <= t_star:
            return t / t_star * f_max
        else:
            return f_max

    force = K_Force(f, beam, frame_ID=(1,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    model.add(moment)
    model.add(force)
    model.assemble()

    solver = Newton(
        model,
        n_load_steps=n_load_steps,
        max_iter=100,
        atol=atol,
        rtol=rtol,
    )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # compute tip displacement
    r_OB0 = beam.r_OP(t[-1], q[-1][beam.elDOF[-1]], frame_ID=(1,))
    print(f"r_OP(1): {r_OB0}")
    # print(f"r_OP(1) * slenderness: {r_OP * slenderness}")
    # print(f"r_OP(1) / slenderness: {r_OP / slenderness}")
    # print(f"r_OP(1) / Fi: {r_OP / Fi}")

    ##################################
    # visualize nodal rotation vectors
    ##################################
    fig, ax = plt.subplots()

    # visualize tip rotation vector
    psi1 = q[:, beam.qDOF[beam.nodalDOF_psi[-1]]]
    for i, psi in enumerate(psi1.T):
        ax.plot(t, psi, label=f"psi1_{i}")

    ax.set_xlabel("t")
    ax.set_ylabel("nodal rotation vectors")
    ax.grid()
    ax.legend()

    # ################################
    # # visualize norm strain measures
    # ################################
    # fig, ax = plt.subplots(1, 2)

    # nxi = 1000
    # xis = np.linspace(0, 1, num=nxi)

    # K_Gamma = np.zeros((3, nxi))
    # K_Kappa = np.zeros((3, nxi))
    # for i in range(nxi):
    #     frame_ID = (xis[i],)
    #     elDOF = beam.qDOF_P(frame_ID)
    #     qe = q[-1, beam.qDOF][elDOF]
    #     _, _, K_Gamma[:, i], K_Kappa[:, i] = beam.eval(qe, xis[i])
    # ax[0].plot(xis, K_Gamma[0], "-r", label="K_Gamma0")
    # ax[0].plot(xis, K_Gamma[1], "-g", label="K_Gamma1")
    # ax[0].plot(xis, K_Gamma[2], "-b", label="K_Gamma2")
    # ax[0].grid()
    # ax[0].legend()
    # ax[1].plot(xis, K_Kappa[0], "-r", label="K_Kappa0")
    # ax[1].plot(xis, K_Kappa[1], "-g", label="K_Kappa1")
    # ax[1].plot(xis, K_Kappa[2], "-b", label="K_Kappa2")
    # ax[1].grid()
    # ax[1].legend()

    ############################
    # Visualize potential energy
    ############################
    E_pot = np.array([model.E_pot(ti, qi) for (ti, qi) in zip(t, q)])

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(t, E_pot)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("E_pot")
    ax[0].grid()

    idx = np.where(t > t_star)[0]
    ax[1].plot(t[idx], E_pot[idx])
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("E_pot")
    ax[1].grid()

    ########################################
    # visualize final centerline projections
    ########################################
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

    ####################
    # export enery plots
    ####################

    # potential energy
    header = "t, E_pot"
    # export_data = np.vstack([t, E_pot]).T
    export_data = np.vstack([np.arange(len(t)), E_pot]).T
    np.savetxt(
        "results/ObjectivityEpot_full.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    # header = "t, E_pot"
    # export_data = np.vstack([t[idx], E_pot[idx]]).T
    # np.savetxt(
    #     "results/ObjectivityEpot_rotation.txt",
    #     export_data,
    #     delimiter=", ",
    #     header=header,
    #     comments="",
    # )

    # rotation vector
    header = "t, psi0, psi1, psi2, abs_psi"
    # export_data = np.vstack([t, *psi1.T, np.linalg.norm(psi1, axis=1)]).T
    export_data = np.vstack(
        [np.arange(len(t)), *psi1.T, np.linalg.norm(psi1, axis=1)]
    ).T
    np.savetxt(
        "results/ObjectivityRotationVector.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam], L, show=True)


def convergence_quater_circle():
    """This example examines shear and membrande locking as done in Meier2015.

    References:
    ===========
    Meier2015: https://doi.org/10.1016/j.cma.2015.02.029
    """
    # Beam = TimoshenkoAxisAngle
    Beam = TimoshenkoAxisAngleSE3
    # Beam = TimoshenkoQuarternionSE3

    ############
    # parameters
    ############
    L = 1.0e4
    slenderness = 1.0e3

    width = L / slenderness
    line_density = 1
    cross_section = QuadraticCrossSection(line_density, width)

    E = 1.0
    G = 0.5

    material_model = quadratic_beam_material(E, G, cross_section, Beam)
    print(f"Ei: {material_model.Ei}")
    print(f"Fi: {material_model.Fi}")

    atol = 1.0e-9
    rtol = 0
    n_load_steps = 10

    shape_functions = "Lagrange"
    polynomial_degree = 1
    nquadrature_points = int(np.ceil((polynomial_degree + 1) ** 2 / 2))
    print(f"nquadrature_points: {nquadrature_points}")

    # used parameters for the paper
    # nelements_list = np.array([1, 2, 4], dtype=int)
    # nelements_ref = 8
    nelements_list = np.array([1, 2, 4, 8, 16, 32, 64], dtype=int)
    nelements_ref = 256

    # starting point and orientation of initial point, initial length
    r_OP = np.zeros(3, dtype=float)
    A_IK = np.eye(3, dtype=float)

    def solve(nelements):

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
        A_IK0 = np.eye(3)
        frame1 = Frame(r_OP=r_OB0, A_IK=A_IK0)

        # left and right joint
        joint1 = RigidConnection(frame1, beam, r_OB0, frame_ID2=(0,))

        # moment at the beam's tip
        Fi = material_model.Fi
        M = lambda t: (e3 * Fi[2]) * t * 2 * np.pi / L * 0.25
        moment = K_Moment(M, beam, (1,))

        # force at the beam's tip
        f = lambda t: t * 1e-4 * Fi[2] / L * e3
        force = Force(f, beam, frame_ID=(1,))

        # assemble the model
        model = Model()
        model.add(beam)
        model.add(frame1)
        model.add(joint1)
        model.add(moment)
        model.add(force)
        model.assemble()

        solver = Newton(
            model,
            n_load_steps=n_load_steps,
            max_iter=100,
            atol=atol,
            rtol=rtol,
        )

        sol = solver.solve()

        return beam, sol

    beam_ref, sol_ref = solve(nelements_ref)

    # sample centerline deflection of reference solution
    num = 100
    xis = np.linspace(0, 1, num=num)
    r_OP_ref = beam_ref.centerline(sol_ref.q[-1], n=num)
    A_IK_ref = np.array(beam_ref.frames(sol_ref.q[-1], n=num)[1:])

    position_errors = []
    rotation_errors = []
    for nelements in nelements_list:
        beam, sol = solve(nelements)

        # centerline errors
        r_OPi = beam.centerline(sol.q[-1], n=num)
        diff = r_OPi - r_OP_ref
        error = sqrt(sum([d @ d for d in diff])) / num
        position_errors.append(error)

        # rotation errors
        A_IKi = np.array(beam.frames(sol.q[-1], n=num)[1:])
        diff = []
        for i in range(num):
            diff.append(rodriguez_inv(A_IKi[:, :, i].T @ A_IK_ref[:, :, i]))
        diff = np.array(diff)
        error = sqrt(sum([d @ d for d in diff]))
        rotation_errors.append(error)

    position_errors = np.array(position_errors)
    print(f"position_errors: {position_errors}")
    rotation_errors = np.array(rotation_errors)
    print(f"rotation_errors: {rotation_errors}")

    # export errors
    header = "nelements, position_error, rotation_errors"
    export_data = np.vstack([nelements_list, position_errors, rotation_errors]).T
    np.savetxt(
        "results/QuarterCircleConvergence.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    ##########################
    # plot rate of convergence
    ##########################
    fig, ax = plt.subplots()
    ax.loglog(nelements_list, position_errors, "-ok", label="e_r^100")
    ax.loglog(nelements_list, rotation_errors, "-sk", label="e_psi^100")
    ax.loglog(nelements_list, 90 / nelements_list, "--k", label="~1 / n_el")
    ax.loglog(nelements_list, 90 / nelements_list**2, "-.k", label="~1 / n_el^2")
    ax.grid()
    ax.legend()

    ###########################################
    # strain measures of the reference solution
    ###########################################
    fig, ax = plt.subplots(1, 2)

    nxi = 1000
    xis = np.linspace(0, 1, num=nxi)

    K_Gamma = np.zeros((3, nxi))
    K_Kappa = np.zeros((3, nxi))
    for i in range(nxi):
        frame_ID = (xis[i],)
        elDOF = beam_ref.qDOF_P(frame_ID)
        qe = sol_ref.q[-1, beam_ref.qDOF][elDOF]
        _, _, K_Gamma[:, i], K_Kappa[:, i] = beam_ref.eval(qe, xis[i])

    header = "xi, K_Gamma1, K_Gamma2, K_Gamma3, K_Kappa1, K_Kappa2, K_Kappa3"
    export_data = np.vstack([xis, *K_Gamma, *K_Kappa]).T
    np.savetxt(
        "results/QuarterCircleConvergenceStrainMeasures.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

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

    plt.show()


def Noor1981():
    """Large deflections of a clamped semicircular arch found in Noor1981 and Simo1985.

    References:
    ===========
    Noor1981:  https://doi.org/10.1002/nme.1620170409 \\
    Simo1985: https://doi.org/10.1016/0045-7825(86)90079-4
    """
    # Beam = TimoshenkoAxisAngle
    Beam = TimoshenkoAxisAngleSE3
    # Beam = TimoshenkoDirectorDirac
    # Beam = TimoshenkoQuarternionSE3

    # 1 elements per beam - 4 beams for the circle
    nelements = 1

    # used polynomial degree
    polynomial_degree = 1

    # number of quadrature points
    nquadrature_points = int(np.ceil((polynomial_degree + 1) ** 2 / 2))

    # used shape functions for discretization
    shape_functions = "Lagrange"

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
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    # radius of the circle
    R = 1

    ################################
    # first quadrant (R, 0) - (radius, R)
    ################################
    r_OP0 = np.array([radius, 0, 0], dtype=float)
    r_OP1 = np.array([0, radius, 0], dtype=float)
    # r_OP1 = np.array([0, 0, 0], dtype=float)
    psi0 = pi / 2 * e3
    # psi0 = pi * e3
    psi1 = pi * e3
    q0 = np.concatenate((r_OP0, r_OP1, psi0, psi1))

    beam0 = TimoshenkoAxisAngleSE3(
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree,
        polynomial_degree,
        nquadrature_points,
        nelements,
        q0,
        basis=shape_functions,
    )

    frame0 = Frame(r_OP=r_OP0)
    joint0 = RigidConnection(frame0, beam0, frame_ID1=(0,))
    # joint0 = SphericalJoint(frame0, beam0, r_OP0, frame_ID1=(0,))

    ##################################
    # second quadrant (0, R) - (-R, 0)
    ##################################
    r_OP0 = np.array([0, radius, 0], dtype=float)
    # r_OP0 = np.array([0, 0, 0], dtype=float)
    r_OP1 = np.array([-radius, 0, 0], dtype=float)
    psi0 = pi * e3
    # psi1 = pi * e3
    psi1 = 3 / 2 * pi * e3
    q0 = np.concatenate((r_OP0, r_OP1, psi0, psi1))

    beam1 = TimoshenkoAxisAngleSE3(
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree,
        polynomial_degree,
        nquadrature_points,
        nelements,
        q0,
    )

    frame1 = Frame(r_OP=r_OP1)
    # joint1 = RigidConnection(frame1, beam1, frame_ID2=(1,))
    joint1 = SphericalJoint(frame1, beam1, r_OP1, frame_ID2=(1,))

    #####################################
    # rigid connection between both beams
    #####################################
    joint2 = RigidConnection(beam0, beam1, frame_ID1=(1,), frame_ID2=(0,))

    # external force at the apex
    F_max = -3e3
    F = lambda t: F_max * e2 * t
    # force = Force(F, beam0, frame_ID=(1,)
    force = Force(F, beam1, frame_ID=(0,))

    # assemble the model
    model = Model()
    model.add(beam0)
    model.add(frame0)
    model.add(joint0)
    model.add(beam1)
    model.add(frame1)
    model.add(joint1)
    model.add(joint2)
    model.add(force)
    model.assemble()

    # n_load_steps = 2
    n_load_steps = 20

    # solver = Newton(
    #     model,
    #     n_load_steps=n_load_steps,
    #     max_iter=30,
    #     atol=1.0e-7,
    #     numerical_jacobian=False,
    # )
    solver = Riks(
        model,
        tol=1.0e-8,
        max_newton_iter=30,
        iter_goal=3,
        # la_arc0=5.0e-3,
        la_arc0=1.0e-3,
        la_arc_span=[-0.5, 1],
        scale_exponent=None,
    )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam0, beam1], radius, show=True)


def BucklingRightHingedFrame():
    """Buckling of a hinged right-angle frame under both fixed and follower point load - Simo1985.

    References:
    ===========
    Simo1985: https://doi.org/10.1016/0045-7825(86)90079-4
    """
    # elements per beam
    nelement_per_beam = 3

    # used polynomial degree
    polynomial_degree = 1

    # number of quadrature points
    nquadrature_points = int(np.ceil((polynomial_degree + 1) ** 2 / 2))

    # used shape functions for discretization
    shape_functions = "Lagrange"

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
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    # length of th eframe
    L = 2

    ############################
    # first beam (0, 0) - (0, L)
    ############################
    # r_OP0 = np.array([0, 0, 0], dtype=float)
    # r_OP1 = np.array([0, L, 0], dtype=float)
    # psi0 = pi / 2 * e3
    # psi1 = pi / 2 * e3
    # q0 = np.concatenate((r_OP0, r_OP1, psi0, psi1))

    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = rodriguez(pi / 2 * e3)
    q0 = TimoshenkoAxisAngleSE3.straight_configuration(
        polynomial_degree,
        polynomial_degree,
        nelement_per_beam,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
        basis=shape_functions,
    )

    beam0 = TimoshenkoAxisAngleSE3(
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree,
        polynomial_degree,
        nquadrature_points,
        nelement_per_beam,
        q0,
        basis=shape_functions,
    )

    frame0 = Frame(r_OP=r_OP0)
    joint0 = SphericalJoint(frame0, beam0, r_OP0, frame_ID1=(0,))

    ##################################
    # second quadrant (0, L) - (L, L)
    ##################################
    # r_OP0 = np.array([0, L, 0], dtype=float)
    # r_OP1 = np.array([L, L, 0], dtype=float)
    # psi0 = np.zeros(3, dtype=float)
    # psi1 = np.zeros(3, dtype=float)
    # q0 = np.concatenate((r_OP0, r_OP1, psi0, psi1))

    r_OP0 = np.array([0, L, 0], dtype=float)
    A_IK0 = np.eye(3, dtype=float)
    q0 = TimoshenkoAxisAngleSE3.straight_configuration(
        polynomial_degree,
        polynomial_degree,
        nelement_per_beam,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
        basis=shape_functions,
    )

    beam1 = TimoshenkoAxisAngleSE3(
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree,
        polynomial_degree,
        nquadrature_points,
        nelement_per_beam,
        q0,
    )

    r_OP1 = np.array([L, L, 0], dtype=float)
    frame1 = Frame(r_OP=r_OP1)
    joint1 = SphericalJoint(frame1, beam1, r_OP1, frame_ID2=(1,))

    #####################################
    # rigid connection between both beams
    #####################################
    joint2 = RigidConnection(beam0, beam1, frame_ID1=(1,), frame_ID2=(0,))

    # external force at the apex
    # F_max = -5e3
    F_max = -1e3
    F = lambda t: F_max * e2 * t
    force = Force(F, beam1, frame_ID=(1 / 3,))

    # assemble the model
    model = Model()
    model.add(beam0)
    model.add(frame0)
    model.add(joint0)
    model.add(beam1)
    model.add(frame1)
    model.add(joint1)
    model.add(joint2)
    model.add(force)
    model.assemble()

    # # n_load_steps = 2
    # n_load_steps = 10
    # solver = Newton(
    #     model,
    #     n_load_steps=n_load_steps,
    #     max_iter=30,
    #     atol=1.0e-7,
    #     numerical_jacobian=False,
    # )

    solver = Riks(
        model,
        tol=1.0e-8,
        max_newton_iter=30,
        # la_arc0=1.0e-1,
        la_arc0=1.0e-1,
        la_arc_span=[-0.5, 1],
        scale_exponent=None,
    )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    ###########
    # animation
    ###########
    animate_beam(t, q, [beam0, beam1], L, show=True)


if __name__ == "__main__":
    # run(statics=True)
    # run(statics=False)
    # locking(case="quater_circle")
    # locking(case="helix")
    # SE3_interpolation()
    # HelixIbrahimbegovic1997()
    # HeavyTop()
    # Dschanibekow()
    # distributed_force()
    # objectivity_quater_circle()
    # convergence_quater_circle()
    # Noor1981()
    BucklingRightHingedFrame()
