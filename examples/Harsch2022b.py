from cardillo.math import e1, e2, e3, sqrt, sin, cos, pi
from cardillo.beams.spatial import (
    UserDefinedCrossSection,
    CircularCrossSection,
    RectangularCrossSection,
    QuadraticCrossSection,
    ShearStiffQuadratic,
    Simo1986,
)
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
    TimoshenkoAxisAngleSE3,
    TimoshenkoAxisAngle,
)
from cardillo.forces import Force, K_Force, K_Moment, Moment, DistributedForce1D
from cardillo.model import Model
from cardillo.solver import (
    Newton,
    ScipyIVP,
    Riks,
)

import numpy as np
import matplotlib.pyplot as plt


def locking_quater_circle():
    """This example examines shear and membrande locking as done in Meier2015.

    References:
    ===========
    Meier2015: https://doi.org/10.1016/j.cma.2015.02.029
    """
    # Young's and shear modulus
    E = 1.0  # Meier2015
    G = 0.5  # Meier2015
    nelements = 1
    polynomial_degree = 1
    rtol = 0

    L = 1.0e4
    # (slenderness, atol, n_load_steps
    triplets = [
        (1.0e1, 1.0e-6, 2),
        (1.0e2, 1.0e-8, 2),
        (1.0e3, 1.0e-9, 2),
        (1.0e4, 1.0e-10, 2),  # this can't be sovle using the quaternion implementation
        (1.0e5, 1.0e-11, 4),
    ]

    for triplet in triplets:
        slenderness, atol, n_load_steps = triplet

        # used cross section
        width = L / slenderness

        # cross section and quadratic beam material
        line_density = 1
        cross_section = QuadraticCrossSection(line_density, width)
        A_rho0 = line_density * cross_section.area
        K_S_rho0 = line_density * cross_section.first_moment
        K_I_rho0 = line_density * cross_section.second_moment
        A = cross_section.area
        Ip, I2, I3 = np.diag(cross_section.second_moment)
        Ei = np.array([E * A, G * A, G * A])
        Fi = np.array([G * Ip, E * I2, E * I3])
        material_model = Simo1986(Ei, Fi)

        # starting point and orientation of initial point, initial length
        r_OP0 = np.zeros(3, dtype=float)
        A_IK0 = np.eye(3, dtype=float)

        q0 = TimoshenkoAxisAngleSE3.straight_configuration(
            polynomial_degree,
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        beam = TimoshenkoAxisAngleSE3(
            polynomial_degree,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            nelements,
            q0,
        )

        frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)
        joint1 = RigidConnection(frame1, beam, r_OP0, frame_ID2=(0,))

        # moment at the beam's tip
        Fi = material_model.Fi
        M = lambda t: t * (e3 * Fi[2]) * 2 * np.pi / L * 0.25
        moment = K_Moment(M, beam, (1,))

        # force at the beam's tip
        f = lambda t: t * 1e-4 * Fi[2] / L * e3
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

    animate_beam(t, q, [beam], L, show=True)


def objectivity_quater_circle():
    """This example examines shear and membrande locking as done in Meier2015.

    References:
    ===========
    Meier2015: https://doi.org/10.1016/j.cma.2015.02.029
    """
    # Young's and shear modulus
    E = 1.0  # Meier2015
    G = 0.5  # Meier2015
    nelements = 1
    polynomial_degree = 1
    rtol = 0
    n = 10  # number of full rotations
    t_star = 0.1  # fraction of deformation pseudo time
    # n_load_steps = 200
    n_load_steps = 500  # used for the paper

    L = 1.0e4
    slenderness = 1.0e3
    atol = 1.0e-9

    # used cross section
    width = L / slenderness

    # cross section and quadratic beam material
    line_density = 1
    cross_section = QuadraticCrossSection(line_density, width)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)

    def A_IK0(t):
        phi = (
            n * np.heaviside(t - t_star, 1.0) * (t - t_star) / (1.0 - t_star) * 2.0 * pi
        )
        return A_IK_basic(phi).x()

    q0 = TimoshenkoAxisAngleSE3.straight_configuration(
        polynomial_degree,
        nelements,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0(0),
    )
    beam = TimoshenkoAxisAngleSE3(
        polynomial_degree,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
        q0,
    )

    frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)
    joint1 = RigidConnection(frame1, beam, r_OP0, frame_ID2=(0,))

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

    ####################
    # export enery plots
    ####################

    # potential energy
    header = "t, E_pot"
    export_data = np.vstack([np.arange(len(t)), E_pot]).T
    np.savetxt(
        "results/ObjectivityEpot_full.txt",
        export_data,
        delimiter=", ",
        header=header,
        comments="",
    )

    # rotation vector
    header = "t, psi0, psi1, psi2, abs_psi"
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
    # Young's and shear modulus
    E = 1.0  # Meier2015
    G = 0.5  # Meier2015
    nelements = 1
    polynomial_degree = 1
    rtol = 0
    n_load_steps = 10

    L = 1.0e4
    slenderness = 1.0e3
    atol = 1.0e-9

    # used cross section
    width = L / slenderness

    # cross section and quadratic beam material
    line_density = 1
    cross_section = QuadraticCrossSection(line_density, width)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

    # used parameters for the paper
    nelements_list = np.array([1, 2, 4], dtype=int)
    nelements_ref = 8
    # nelements_list = np.array([1, 2, 4, 8, 16, 32, 64], dtype=int)
    # nelements_ref = 256

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    def solve(nelements):
        q0 = TimoshenkoAxisAngleSE3.straight_configuration(
            polynomial_degree,
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        beam = TimoshenkoAxisAngleSE3(
            polynomial_degree,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            nelements,
            q0,
        )

        # junctions
        frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)

        # left and right joint
        joint1 = RigidConnection(frame1, beam, frame_ID2=(0,))

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


def HelixIbrahimbegovic1997():
    """Beam bent to a helical form - Section 5.2 of Ibrahimbegovic1997.

    References
    ==========
    Ibrahimbegovic1997: https://doi.org/10.1016/S0045-7825(97)00059-5
    """
    # fraction of 10 full rotations and the out of plane force
    # a corresponding fraction of 100 elements is chosen
    # # fraction = 0.05
    fraction = 0.1  # 1 full rotations
    # fraction = 0.20  # 2 full rotations
    # fraction = 0.4  # 4 full rotations
    # fraction = 0.5  # 5 full rotations
    # fraction = 1  # 10 full rotations

    # number of elements
    nelements_max = 30
    nelements = max(3, int(fraction * nelements_max))
    print(f"nelemen: {nelements}")

    # used polynomial degree
    polynomial_degree = 1

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

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    q0 = TimoshenkoAxisAngleSE3.straight_configuration(
        polynomial_degree,
        nelements,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
    )
    beam = TimoshenkoAxisAngleSE3(
        polynomial_degree,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelements,
        q0,
    )

    # junctions
    frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, beam, frame_ID2=(0,))

    # moment at right end
    Fi = material_model.Fi
    M = lambda t: (e3 * 10 * 2 * np.pi * Fi[2] / L * t * fraction)
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
    header = "t, x, y, z"
    frame_ID = (1,)
    elDOF = beam.qDOF_P(frame_ID)
    r_OC_L = np.array([beam.r_OP(ti, qi[elDOF], frame_ID) for (ti, qi) in zip(t, q)])
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

    # number of elements
    nelements = 1

    ######################
    # nice locking results
    ######################
    g = 9.81
    l = 0.5
    r = 0.1
    omega_x = 2 * pi * 50
    E_stiff = 210e6  # steel (stiff beam)
    E_soft = E_stiff * 1.0e-3  # soft beam
    rho = 8000  # steel [kg/m^3]

    # tip volume and mass
    V = l * pi * r**2
    m = rho * V
    print(f"total mass: {m}")
    cross_section = CircularCrossSection(rho, r)
    A_rho0 = rho * cross_section.area
    K_S_rho0 = rho * cross_section.first_moment
    K_I_rho0 = rho * cross_section.second_moment

    # initial angular velocity and orientation
    # A = 0.5 * m * r**2
    # omega_pr = m * g * (0.5 * l) / (A * omega_x)
    omega_pr = g * l / (r**2 * omega_x)
    K_omega_IK0 = omega_x * e1 + omega_pr * e3  # perfect precession motion
    A_IK0 = np.eye(3, dtype=float)
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
        G = E / (2.0 * (1.0 + nu))

        A = cross_section.area
        Ip, I2, I3 = np.diag(cross_section.second_moment)
        Ei = np.array([E * A, G * A, G * A])
        Fi = np.array([G * Ip, E * I2, E * I3])
        material_model = Simo1986(Ei, Fi)

        q0, u0 = TimoshenkoAxisAngleSE3.initial_configuration(
            nelements,
            l,
            r_OP0=r_OP0,
            A_IK0=A_IK0,
            v_P0=np.zeros(3, dtype=float),
            K_omega_IK0=K_omega_IK0,
        )
        beam = TimoshenkoAxisAngleSE3(
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            nelements,
            q0,
            u0=u0,
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


def BucklingRightHingedFrame(follower=False):
    # def BucklingRightHingedFrame(follower=True):
    """Buckling of a hinged right-angle frame under both fixed and follower
    point load - Simo1985.

    References:
    ===========
    Simo1985: https://doi.org/10.1016/0045-7825(86)90079-4
    """
    polynomial_degree = 1

    # elements per beam
    # nelement_per_beam = 2
    nelement_per_beam = 5
    # nelement_per_beam = 10

    # beam parameters found in Harsch2020
    L = 120
    E = 7.2e6
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    A = 6
    I = 2
    EA = E * A
    GA = G * A
    GJ = G * 2 * I
    EI = E * I

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

    ############################
    # first beam (0, 0) - (0, L)
    ############################
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = rodriguez(pi / 2 * e3)
    q0 = TimoshenkoAxisAngleSE3.straight_configuration(
        polynomial_degree,
        nelement_per_beam,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
    )
    beam0 = TimoshenkoAxisAngleSE3(
        polynomial_degree,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelement_per_beam,
        q0,
    )
    # q0 = TimoshenkoAxisAngle.straight_configuration(
    #     polynomial_degree,
    #     polynomial_degree,
    #     nelement_per_beam,
    #     L,
    #     r_OP=r_OP0,
    #     A_IK=A_IK0,
    # )
    # beam0 = TimoshenkoAxisAngle(
    #     material_model,
    #     A_rho0,
    #     K_I_rho0,
    #     polynomial_degree,
    #     polynomial_degree,
    #     int(np.ceil((polynomial_degree + 1) ** 2 / 2)),
    #     nelement_per_beam,
    #     q0,
    # )

    frame0 = Frame(r_OP=r_OP0)
    joint0 = SphericalJoint(frame0, beam0, r_OP0, frame_ID1=(0,))

    #############################
    # second beam (0, L) - (L, L)
    #############################
    r_OP0 = np.array([0, L, 0], dtype=float)
    A_IK0 = np.eye(3, dtype=float)
    q0 = TimoshenkoAxisAngleSE3.straight_configuration(
        polynomial_degree,
        nelement_per_beam,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
    )
    beam1 = TimoshenkoAxisAngleSE3(
        polynomial_degree,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        nelement_per_beam,
        q0,
    )
    # q0 = TimoshenkoAxisAngle.straight_configuration(
    #     polynomial_degree,
    #     polynomial_degree,
    #     nelement_per_beam,
    #     L,
    #     r_OP=r_OP0,
    #     A_IK=A_IK0,
    # )
    # beam1 = TimoshenkoAxisAngle(
    #     material_model,
    #     A_rho0,
    #     K_I_rho0,
    #     polynomial_degree,
    #     polynomial_degree,
    #     int(np.ceil((polynomial_degree + 1) ** 2 / 2)),
    #     nelement_per_beam,
    #     q0,
    # )

    r_OP1 = np.array([L, L, 0], dtype=float)
    frame1 = Frame(r_OP=r_OP1)
    joint1 = SphericalJoint(frame1, beam1, r_OP1, frame_ID2=(1,))

    #####################################
    # rigid connection between both beams
    #####################################
    joint2 = RigidConnection(beam0, beam1, frame_ID1=(1,), frame_ID2=(0,))

    # external force at the apex
    # F_max = -4e4  # Harsch2020
    F_max = -8e4  # Harsch2020
    F = lambda t: F_max * e2 * t
    if follower:
        force = K_Force(F, beam1, frame_ID=(0.2,))
    else:
        force = Force(F, beam1, frame_ID=(0.2,))

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
    # n_load_steps = 20
    # solver = Newton(
    #     model,
    #     n_load_steps=n_load_steps,
    #     max_iter=30,
    #     atol=1.0e-6,
    #     numerical_jacobian=False,
    # )

    solver = Riks(
        model,
        tol=1.0e-6,
        max_newton_iter=30,
        la_arc0=1.0e-1,  # works for constant force
        # la_arc0=5.0e-2,  # works for constant force
        # la_arc0=1.0e-2,  # works for constant force
        # la_arc0=5.0e-3,  # not working for follower force yet
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
    # locking_quater_circle()
    # objectivity_quater_circle()
    # convergence_quater_circle()
    HelixIbrahimbegovic1997()
    # HeavyTop()
    # BucklingRightHingedFrame()
