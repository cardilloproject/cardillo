from cardillo.math import e1, e2, e3, sin, pi, smoothstep2, A_IK_basic
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
    TimoshenkoQuaternion,
)
from cardillo.forces import Force, K_Moment, DistributedForce1D
from cardillo.contacts import Line2Line
from cardillo.model import Model
from cardillo.solver import (
    Newton,
    ScipyIVP,
    GenAlphaFirstOrderVelocityGGL,
    GenAlphaFirstOrderVelocity,
    GenAlphaDAEAcc,
    Moreau,
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def quadratic_beam_material(E, G, cross_section, Beam):
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])

    if Beam == Cable or Beam == CubicHermiteCable or Beam == Kirchhoff:
        return ShearStiffQuadratic(Ei[0], Fi)
    elif Beam == DirectorAxisAngle or Beam == TimoshenkoQuaternion:
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
        # p_psi = p_r
        p_psi = p_r - 1
        # p_psi = 1
        Q = DirectorAxisAngle.straight_configuration(
            p_r, p_psi, nelements, L, r_OP=r_OP, A_IK=A_IK, basis=shape_functions
        )
    elif Beam == TimoshenkoQuaternion:
        p_r = polynomial_degree
        # p_psi = p_r
        p_psi = p_r - 1
        Q = TimoshenkoQuaternion.straight_configuration(
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
    elif Beam == TimoshenkoQuaternion:
        beam = TimoshenkoQuaternion(
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
    # Beam = DirectorAxisAngle
    Beam = TimoshenkoQuaternion

    # number of elements
    nelements = 1
    # nelements = 2
    # nelements = 4
    # nelements = 8
    # nelements = 16
    # nelements = 32
    # nelements = 64

    # used polynomial degree
    # polynomial_degree = 1
    # polynomial_degree = 2
    polynomial_degree = 3
    # polynomial_degree = 5
    # polynomial_degree = 6

    # number of quadrature points
    # TODO: We have to distinguish between integration of the mass matrix,
    #       gyroscopic forces and potential forces!
    # nquadrature_points = int(np.ceil((polynomial_degree + 1)**2 / 2))
    nquadrature_points = polynomial_degree + 2
    # nquadrature_points = polynomial_degree + 1
    # nquadrature_points = polynomial_degree  # cures locking but has to be modified for mass matrix

    # working combinations
    # - Bspline shape functions: "Absolute rotation vector with Crisfield's  relative interpolation":
    #   p = 2,3,5; p_r = p; p_psi = p - 1, nquadrature = p (Gauss-Legendre),
    #   nelements = 1,2,4; slenderness = 1.0e3
    # - cubic Hermite shape functions: "Absolute rotation vector with Crisfield's  relative interpolation":
    #   p_r = 3; p_psi = 1, nquadrature = 3 (Gauss-Lobatto),
    #   nelements = 1,2,4; slenderness = 1.0e3

    # used shape functions for discretization
    # shape_functions = "B-spline"
    # shape_functions = "Lagrange"
    shape_functions = "Hermite"

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
    print(f"Ei: {material_model.Ei}")
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
    # q = model.q0
    # # q = np.random.rand(model.nq)

    # E_pot = model.E_pot(t, q)
    # print(f"E_pot: {E_pot}")
    # f_pot = model.f_pot(t, q)
    # print(f"f_pot:\n{f_pot}")
    # # f_pot_q = model.f_pot_q(t, q)
    # # print(f"f_pot_q:\n{f_pot_q}")

    # exit()

    # xis = np.linspace(0, 1, num=10)
    # for xi in xis:
    #     r_OC = beam.r_OC(t, q, frame_ID=(xi,))
    #     print(f"r_OC({xi}): {r_OC}")
    # for xi in xis:
    #     r_OC_xi = beam.r_OC_xi(t, q, frame_ID=(xi,))
    #     print(f"r_OC_xi({xi}): {r_OC_xi}")
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
        # phi = lambda t: n_circles * 2 * pi * smoothstep2(t, frac_deformation, 1.0) * 0.5
        # # phi2 = lambda t: pi / 4 * sin(2 * pi * smoothstep2(t, frac_deformation, 1.0))
        # # A_IK0 = lambda t: A_IK_basic(phi(t)).x()
        # # TODO: Get this strange rotation working with a full circle
        # # A_IK0 = lambda t: A_IK_basic(phi(t)).z()
        # A_IK0 = (
        #     lambda t: A_IK_basic(0.5 * phi(t)).z()
        #     @ A_IK_basic(0.5 * phi(t)).y()
        #     @ A_IK_basic(phi(t)).x()
        # )
        A_IK0 = lambda t: np.eye(3)
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
    elif Beam == Kirchhoff or Beam == DirectorAxisAngle or Beam == TimoshenkoQuaternion:
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
        # lambda t: (e3 * Fi[2])
        lambda t: (e1 * Fi[0] + e3 * Fi[2])
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
    # else:
    #     model.add(f_g_beam)
    model.assemble()

    if statics:
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
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # if nelements == 1:
    # visualize nodal rotation vectors
    fig, ax = plt.subplots()

    for i, nodalDOF_psi in enumerate(beam.nodalDOF_psi):
        psi = q[:, beam.qDOF[nodalDOF_psi]]
        ax.plot(t, np.linalg.norm(psi, axis=1), label=f"||psi{i}||")

    ax.set_xlabel("t")
    ax.set_ylabel("nodal rotation vectors")
    ax.grid()
    ax.legend()

    ########################################################
    # visualize norm of tangent vector and quadrature points
    ########################################################
    fig, ax = plt.subplots()

    nxi = 1000
    xis = np.linspace(0, 1, num=nxi)

    abs_r_xi = np.zeros(nxi)
    abs_r0_xi = np.zeros(nxi)
    for i in range(nxi):
        frame_ID = (xis[i],)
        elDOF = beam.qDOF_P(frame_ID)
        qe = q[-1, beam.qDOF][elDOF]
        abs_r_xi[i] = np.linalg.norm(beam.r_OC_xi(t[-1], qe, frame_ID))
        q0e = q[0, beam.qDOF][elDOF]
        abs_r0_xi[i] = np.linalg.norm(beam.r_OC_xi(t[0], q0e, frame_ID))
    ax.plot(xis, abs_r_xi, "-r", label="||r_xi||")
    ax.plot(xis, abs_r0_xi, "--b", label="||r0_xi||")
    ax.set_xlabel("xi")
    ax.set_ylabel("||r_xi||")
    ax.grid()
    ax.legend()

    # compute quadrature points
    for el in range(beam.nelement):
        elDOF = beam.elDOF[el]
        q0e = q[0, beam.qDOF][elDOF]
        for i in range(beam.nquadrature):
            xi = beam.qp[el, i]
            abs_r0_xi = np.linalg.norm(beam.r_OC_xi(t[0], q0e, (xi,)))
            ax.plot(xi, abs_r0_xi, "xr")

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


def animate_beam_with_contacts(
    t, q, beams, line2line, scale, scale_di=1, n_r=100, n_frames=10, show=True
):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="3d"))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=-scale, top=scale)

    # prepare data for animation
    frames = len(q) - 1
    target_frames = min(frames, 100)
    frac = max(1, int(frames / target_frames))
    animation_time = 1
    interval = animation_time * 1000 / target_frames

    t = t[::frac]
    q = q[::frac]

    # animated objects
    nodes = []
    center_lines = []
    d1s = []
    d2s = []
    d3s = []
    nodes_cpp = []

    # initialize beams
    for beam in beams:
        # beam nodes
        nodes.extend(ax.plot(*beam.nodes(q[0]), "--ob"))

        # beam centerline
        center_lines.extend(ax.plot(*beam.centerline(q[0], n=n_r), "-k"))

        # beam frames
        r, d1, d2, d3 = beam.frames(q[0], n=n_frames)
        d1 *= scale_di
        d2 *= scale_di
        d3 *= scale_di
        d1s.append(
            [
                ax.plot(*np.vstack((r[:, i], r[:, i] + d1[:, i])).T, "-r")[0]
                for i in range(n_frames)
            ]
        )
        d2s.append(
            [
                ax.plot(*np.vstack((r[:, i], r[:, i] + d2[:, i])).T, "-g")[0]
                for i in range(n_frames)
            ]
        )
        d3s.append(
            [
                ax.plot(*np.vstack((r[:, i], r[:, i] + d3[:, i])).T, "-b")[0]
                for i in range(n_frames)
            ]
        )

    # initialize closest points
    for _ in range(line2line.n_contact_points):
        nodes_cpp.append(ax.plot([], [], [], "--or")[0])

    def update(t, q):
        # update beams
        for i, beam in enumerate(beams):
            # beam nodes
            x, y, z = beam.nodes(q)
            nodes[i].set_data(x, y)
            nodes[i].set_3d_properties(z)

            # beam centerline
            if i == 0:
                x, y, z = beam.cover(q, line2line.R1)
            else:
                x, y, z = beam.cover(q, line2line.R2)
            # x, y, z = beam.centerline(q, n=n_r)
            center_lines[i].set_data(x, y)
            center_lines[i].set_3d_properties(z)

            # beam frames
            r, d1, d2, d3 = beam.frames(q, n=n_frames)
            d1 *= scale_di
            d2 *= scale_di
            d3 *= scale_di
            for j in range(n_frames):
                x, y, z = np.vstack((r[:, j], r[:, j] + d1[:, j])).T
                d1s[i][j].set_data(x, y)
                d1s[i][j].set_3d_properties(z)

                x, y, z = np.vstack((r[:, j], r[:, j] + d2[:, j])).T
                d2s[i][j].set_data(x, y)
                d2s[i][j].set_3d_properties(z)

                x, y, z = np.vstack((r[:, j], r[:, j] + d3[:, j])).T
                d3s[i][j].set_data(x, y)
                d3s[i][j].set_3d_properties(z)

        # update closest point
        points, active_contacts = line2line.contact_points(t, q)
        for i, p in enumerate(points):
            x, y, z = p
            nodes_cpp[i].set_data(x, y)
            nodes_cpp[i].set_3d_properties(z)
            if active_contacts[i]:
                nodes_cpp[i].set_color("red")
            else:
                nodes_cpp[i].set_color("blue")

    def animate(i):
        update(t[i], q[i])

    anim = FuncAnimation(
        fig, animate, frames=target_frames, interval=interval, blit=False
    )
    if show:
        plt.show()
    return fig, ax, anim


def run_contact():
    # used beam model
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
    L = 2 * pi
    r_OP0 = np.zeros(3)
    A_IK0 = np.eye(3)
    A_IK1 = A_IK_basic(pi / 3).z()
    center = np.array([L / 2, 0, 3 * radius])
    r_OP1 = center - A_IK1 @ np.array([L / 2, 0, 0])

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

    # contact between both beams
    eps_contact = 1.0e4
    line2line = Line2Line(eps_contact, radius, radius, beam0, beam1)

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
    model.add(line2line)
    model.assemble()

    # dynamic solver
    # t1 = 0.1
    t1 = 1.5
    tol = 1.0e-6

    # dt = 1.0e-3
    # solver = ScipyIVP(model, t1, dt, rtol=tol, atol=tol)

    # dt = 1.0e-2
    # solver = GenAlphaFirstOrderVelocity(model, t1, dt, rho_inf=0.5, tol=tol)

    dt = 2.5e-3
    solver = Moreau(model, t1, dt)

    sol = solver.solve()
    t = sol.t
    q = sol.q

    # ############################
    # # Visualize potential energy
    # ############################
    # E_pot = np.array([model.E_pot(ti, qi) for (ti, qi) in zip(t, q)])

    # fig, ax = plt.subplots()

    # ax.plot(t, E_pot)
    # ax.set_xlabel("t")
    # ax.set_ylabel("E_pot")
    # ax.grid()

    # # visualize final centerline projected in all three planes
    # r_OPs = beam0.centerline(q[-1])
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
    # fig, ax, anima = animate_beam(t, q, [beam0, beam1], L, show=True)
    fig, ax, anima = animate_beam_with_contacts(
        t, q, [beam0, beam1], line2line, L, show=True
    )


if __name__ == "__main__":
    run()
    # run_contact()
