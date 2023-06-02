from cardillo.math import e1, e2, e3, ax2skew
from cardillo.beams import (
    CircularCrossSection,
    Simo1986,
)
from cardillo.discrete import Frame

from cardillo.beams import (
    K_R12_PetrovGalerkin_AxisAngle,
    K_R12_PetrovGalerkin_Quaternion,
    K_R12_PetrovGalerkin_R9,
)
from cardillo.beams import (
    K_SE3_PetrovGalerkin_AxisAngle,
    K_SE3_PetrovGalerkin_Quaternion,
    K_SE3_PetrovGalerkin_R9,
)
from cardillo.beams._fitting import fit_configuration

from cardillo.constraints import RigidConnection

from cardillo.discrete.shapes import Cylinder
from cardillo.discrete import RigidBodyQuaternion

from cardillo.forces import Force
from cardillo.forces import DistributedForce1DBeam

from cardillo import System
from cardillo.solver import (
    Newton,
    EulerBackward,
    ScipyIVP,
    RadauIIa,
    SimplifiedNonsmoothGeneralizedAlpha,
)
from cardillo.visualization import Export

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pathlib import Path

from cardillo.beams import animate_beam

###################
# R12 interpolation
###################
# Rod = K_R12_PetrovGalerkin_AxisAngle
Rod = K_R12_PetrovGalerkin_Quaternion

#####################
# SE(3)-interpolation
#####################
# Rod = K_SE3_PetrovGalerkin_AxisAngle
# Rod = K_SE3_PetrovGalerkin_Quaternion

# statics = True
statics = False

# ecc = 6.5e-3 # best eccentricity - n = 10
ecc = 7.25e-3  # best eccentricity - n = 20


def cylinder(R, h):
    rho = 7850  # [kg / m^3]; steel
    R = 18e-3  # radius
    h = 50e-3  # height
    V = np.pi * R**2 * h  # volume
    m = V * rho  # mass

    I11 = I22 = (1 / 4) * m * R**2 + (1 / 12) * m * h**2
    I33 = (1 / 2) * m * R**2
    Theta_S = np.diag([I11, I22, I33])  # intertia tensor
    return m, Theta_S


def discs(a, R):
    rho = 7850  # [kg / m^3]; steel
    b = 5.925e-3  # height
    r = a / 2
    r2 = r**2
    V = np.pi * b * r2  # volume
    m = V * rho  # mass

    I11 = 0.5 * m * r2
    I22 = I33 = (1 / 4) * m * r2 + (1 / 12) * m * b**2

    Theta_S13 = np.diag([I11, I22, I33])  # intertia tensor
    Theta_S24 = np.diag(
        [I22, I11, I33]
    )  # intertia tensor ( rotated with pi/2 around d3)

    r13 = np.array([R + ecc, 0, 0])
    r24 = np.array([0, R + ecc, 0])
    r13_tilde = ax2skew(r13)
    r24_tilde = ax2skew(r24)
    Theta_Steiner = m * (
        2 * r13_tilde.T @ r13_tilde + 2 * r24_tilde.T @ r24_tilde
    )  # Steiner part

    Theta_S = 2 * Theta_S13 + 2 * Theta_S24 + Theta_Steiner

    return 4 * m, Theta_S


def screws(R):
    rho = 7850  # [kg / m^3]; steel
    d = 5e-3  # diameter of the skrew
    r = d / 2
    r2 = r**2
    l = 33e-3  # length of the skrew
    V = np.pi * r2 * l  # volume
    m = V * rho  # mass

    I11 = 0.5 * m * r2
    I22 = I33 = (1 / 4) * m * r2 + (1 / 12) * m * l**2

    Theta_S13 = np.diag([I11, I22, I33])  # intertia tensor
    Theta_S24 = np.diag(
        [I22, I11, I33]
    )  # intertia tensor ( rotated with pi/2 around d3)

    r_13 = np.array([R + l / 2, 0, 0])
    r_24 = np.array([0, R + l / 2, 0])
    r_13_tilde = ax2skew(r_13)
    r_24_tilde = ax2skew(r_24)

    Theta_Steiner = m * (
        2 * r_13_tilde.T @ r_13_tilde + 2 * r_24_tilde.T @ r_24_tilde
    )  # Steiner part
    Theta_S = 2 * Theta_S13 + 2 * Theta_S24 + Theta_Steiner

    return 4 * m, Theta_S


def Wilberforce_bob(R, h, debug=True):
    ##########
    # cylinder
    ##########
    m1, Theta1_S = cylinder(R, h)
    if debug:
        print(f"mass cylinder = {m1};")
        # print(f'inertia cylinder =\n{Theta1}')

    ###########
    # 4 screw's
    ###########
    m2, Theta2_S = screws(R)
    if debug:
        print(f"mass 4 screw's = {m2}")
        # print(f"inertia 4 screw's  =\n{Theta2}")

    # compute total mass of cylinder and screw's
    # this can be measured
    if debug:
        print(
            f"mass cylinder & 4 screw's = {m1 + m2}; measured masses = {0.412}; error = {np.abs(m1 + m2 - 0.412) / 0.412}"
        )
        # print(f"inertia cylinder & 4 screw's  =\n{Theta1 + Theta2}")

    ##########
    # 4 disc's
    ##########
    a = 19e-3  # outer radius
    d = 5e-3  # inner radius
    m3_outer, Theta3_outer_S = discs(a, R)
    m3_inner, Theta3_inner_S = discs(d, R)
    m3 = m3_outer - m3_inner
    Theta3_S = Theta3_outer_S - Theta3_inner_S
    if debug:
        print(
            f"mass 4 disc's = {m3}; measured mass = {0.049}; error = {np.abs(m3 - 0.049) / 0.049}"
        )
        # print(f"inertia 4 disc's =\n{Theta3}")

    m = m1 + m2 + m3
    Theta = Theta1_S + Theta2_S + Theta3_S

    if debug:
        print(f"total mass = {m}")
        print(f"total inertia =\n{Theta}")

    return m, Theta


if __name__ == "__main__":
    ####################################################################
    # beam parameters, taken from
    # https://faraday.physics.utoronto.ca/PHY182S/WilberforceRefBerg.pdf
    ####################################################################

    # Federstahl nach EN 10270-1
    rho = 7850  # [kg / m^3]
    E = 206e9  # Pa
    G = 81.5e9  # Pa

    # 1mm cross sectional diameter
    d = 1e-3
    r = wire_radius = d / 2

    cross_section = CircularCrossSection(rho, r)
    A_rho0 = rho * cross_section.area
    K_S_rho0 = rho * cross_section.first_moment
    K_I_rho0 = rho * cross_section.second_moment
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

    ##########
    # gravity
    #########
    g = 9.81

    ###########################
    # discretization properties
    ###########################
    # nelements, nturns = 4, 1
    # nelements, nturns = 8, 2
    nelements, nturns = 16, 4
    # nelements, nturns = 32, 8
    # nelements, nturns = 64, 16
    # nelements, nturns = 80, 20
    # nelements = 16  # 2 turns
    # nelements = 32 # 5 turns
    # nelements = 64 # 10 turns
    # nelements = 128 # 20 turns

    ############################################
    # build rod with dummy initial configuration
    ############################################
    if Rod in [K_SE3_PetrovGalerkin_AxisAngle, K_SE3_PetrovGalerkin_Quaternion]:
        Q0 = Rod.straight_configuration(nelements, L=1)
        rod = Rod(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            nelements,
            Q0,
        )
    elif Rod in [K_R12_PetrovGalerkin_AxisAngle, K_R12_PetrovGalerkin_Quaternion]:
        p = 2
        basis = "Lagrange"
        Q0 = Rod.straight_configuration(
            p,
            p,
            basis,
            basis,
            nelements,
            L=1,
        )
        rod = Rod(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            polynomial_degree_r=p,
            polynomial_degree_psi=p,
            nelement=nelements,
            Q=Q0,
            basis_r=basis,
            basis_psi=basis,
        )
    else:
        raise NotImplementedError

    #############################
    # fit reference configuration
    #############################
    coil_diameter = 32.0e-3  # 32mm
    coil_radius = coil_diameter / 2
    pitch_unloaded = 1.0e-3  # 1mm
    c = pitch_unloaded / coil_radius

    # nturns = 1
    # # # turns = 0.5
    # # turns = 2
    # # # turns = 5
    # # # turns = 10
    # # # turns = 20 # used in the paper
    # # nxi = 1000
    # # # nxi = 10000 # used in the paper

    def r(xi, phi0=0.0):
        alpha = 2 * np.pi * nturns * xi
        return coil_radius * np.array(
            [np.sin(alpha + phi0), -np.cos(alpha + phi0), c * alpha]
        )

    def A_IK(xi, phi0=0.0):
        alpha = 2 * np.pi * nturns * xi
        sa = np.sin(alpha + phi0)
        ca = np.cos(alpha + phi0)

        e_x = np.array([ca, sa, c]) / np.sqrt(1 + c**2)
        e_y = np.array([-sa, ca, 0])
        e_z = np.array([-c * ca, -c * sa, 1]) / np.sqrt(1 + c**2)

        return np.vstack((e_x, e_y, e_z)).T

    # nxi = nturns * 25
    nxi = nturns * 15
    xis = np.linspace(0, 1, num=nxi)

    r_OPs = np.array([r(xi, phi0=np.pi) for xi in xis])
    A_IKs = np.array([A_IK(xi, phi0=np.pi) for xi in xis])

    Q0 = fit_configuration(rod, r_OPs, A_IKs, nodal_cDOF=[])
    rod.q0 = Q0.copy()

    #############
    # rod gravity
    #############
    if statics:
        f_g_rod = lambda t, xi: -t * m * g * e3
    else:
        f_g_rod = lambda t, xi: -m * g * e3
    force_rod = DistributedForce1DBeam(f_g_rod, rod)

    #############################################
    # joint between origin and top side of spring
    #############################################
    system = System()
    joint1 = RigidConnection(system.origin, rod, frame_ID2=(1,))

    ##############
    # pendulum bob
    ##############
    R = 18e-3  # radius of the main cylinder
    h = 50e-3  # height of the main cylinder
    m, K_Theta_S = Wilberforce_bob(R, h)
    # center of mass is shifted (wire starts out of the top cylinder surface)
    r_OS0 = np.array([0, 0, -h / 2 - wire_radius])
    p0 = np.array([1, 0, 0, 0], dtype=float)
    q0 = np.concatenate((r_OS0, p0))
    # # bob = Cylinder(RigidBodyQuaternion)(
    # #     length=h, radius=R, axis=2, mass=m, K_Theta_S=K_Theta_S, q0=q0
    # # )
    # bob = RigidBodyQuaternion(
    #     mass=m, K_Theta_S=K_Theta_S, q0=q0
    # )
    density = 7850  # [kg / m^3]; steel
    bob = Cylinder(RigidBodyQuaternion)(
        length=h, radius=R, axis=2, density=density, q0=q0
    )

    ########################
    # connect spring and bob
    ########################
    # joint2 = RigidConnection(bob, rod, frame_ID2=(0,))
    joint2 = RigidConnection(rod, bob, frame_ID1=(0,))

    ################
    # external force
    ################
    if statics:
        f_g_bob = lambda t: -t * A_rho0 * g * e3
    else:
        f_g_bob = lambda t: -m * A_rho0 * e3
    force_bob = Force(f_g_bob, bob)

    #####################
    # assemble the system
    #####################
    system.add(rod, joint1, force_rod)
    system.add(bob, joint2, force_bob)
    system.assemble()

    #####################
    # solve static system
    #####################
    if statics:
        n_load_steps = 30
        solver = Newton(
            system,
            n_load_steps=n_load_steps,
        )
    else:
        t1 = 3
        # dt = 1e-2
        dt = 5e-3

        solver = EulerBackward(system, t1, dt, method="index 3")
        # solver = ScipyIVP(system, t1, dt)
        # solver = RadauIIa(system, t1, dt, dae_index=2, max_step=dt)
        # solver = SimplifiedNonsmoothGeneralizedAlpha(system, t1, dt, rho_inf=0.8)

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    ################################
    # plot characteristic quantities
    ################################
    r_OS = np.array([bob.r_OP(ti, qi[bob.qDOF]) for (ti, qi) in zip(sol.t, sol.q)])

    # ordering = "zxz"
    # ordering = "xyz"
    ordering = "zyx"
    angles = np.array(
        [
            Rotation.from_matrix(bob.A_IK(ti, qi[bob.qDOF])).as_euler(ordering)
            for (ti, qi) in zip(sol.t, sol.q)
        ]
    )

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(t, r_OS[:, 0], label="x")
    ax[0].plot(t, r_OS[:, 1], label="y")
    ax[0].plot(t, r_OS[:, 2], label="z")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, angles[:, 0], label="alpha")
    ax[1].plot(t, angles[:, 1], label="beta")
    ax[1].plot(t, angles[:, 2], label="gamma")
    ax[1].legend()
    ax[1].grid()

    plt.show()

    ###########
    # animation
    ###########
    animate_beam(t, q, [rod], 0.05, scale_di=0.01, show=True)

    ############
    # VTK export
    ############
    path = Path(__file__)
    e = Export(path.parent, path.stem, True, 30, sol)
    # e.export_contr(rod, level="centerline + directors", num=20)
    e.export_contr(rod, level="volume", n_segments=nelements, num=3 * nelements)
    e.export_contr(bob)

    exit()

    # evaluate helixa at discrete points
    xi = np.linspace(0, turns, nxi)
    P, dP, ddP = helix3D(xi, coil_radius, pitch_unloaded)

    # compute directors using Serret-Frenet frame
    d1 = (dP.T / np.linalg.norm(dP, axis=-1)).T
    d2 = (ddP.T / np.linalg.norm(ddP, axis=-1)).T
    d3 = np.cross(d1, d2)

    exit()
    #############################
    # helix example of Harsch2023
    #############################

    # Young's and shear modulus
    E = 1.0
    G = 0.5

    # length od the rod
    L = 1.0e3

    # used cross section
    width = L / slenderness

    # cross section and quadratic beam material
    line_density = 1
    # cross_section = RectangularCrossSection(line_density, width, width)
    cross_section = CircularCrossSection(line_density, width)
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

    if Rod in [
        K_SE3_PetrovGalerkin_AxisAngle,
        K_SE3_PetrovGalerkin_Quaternion,
        K_SE3_PetrovGalerkin_R9,
    ]:
        q0 = Rod.straight_configuration(
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        rod = Rod(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            nelements,
            q0,
        )
    elif Rod in [
        K_R12_PetrovGalerkin_AxisAngle,
        K_R12_PetrovGalerkin_Quaternion,
        K_R12_PetrovGalerkin_R9,
    ]:
        q0 = Rod.straight_configuration(
            polynomial_degree,
            polynomial_degree,
            basis,
            basis,
            nelements,
            L,
            r_OP=r_OP0,
            A_IK=A_IK0,
        )
        rod = Rod(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            polynomial_degree,
            polynomial_degree,
            nelements,
            Q=q0,
            q0=q0,
            basis_r=basis,
            basis_psi=basis,
        )
    else:
        raise NotImplementedError

    frame1 = Frame(r_OP0, A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, rod, frame_ID2=(0,))

    # moment at the beam's tip
    Fi = material_model.Fi
    m = Fi[2] * 2 * np.pi / L * 0.25
    M = lambda t: t * e3 * m
    moment = K_Moment(M, rod, (1,))

    # force at the beam's tip
    f_g_rod = m / L
    F = lambda t: t * f_g_rod * e3
    print(f"f_max: {F(1)}")
    force_bob = Force(F, rod, frame_ID=(1,))

    # # moment at right end
    # Fi = material_model.Fi
    # # M = lambda t: (e3 * 2 * np.pi * Fi[2] / L * t) * 2
    # M = lambda t: 2 * np.pi / L * (e1 * Fi[0] + e3 * Fi[2]) * t * 2
    # # if statics:
    # #     M = lambda t: (e1 * Fi[0] + e3 * Fi[2]) * 1.0 * t * 2 * np.pi / L * 0.5
    # # else:
    # #     M = lambda t: (e1 * Fi[0] + e3 * Fi[2]) * 1.0 * 2 * np.pi / L * 0.05
    # moment = K_Moment(M, beam, (1,))

    # # force at the rght end
    # f = lambda t: t * e1 * 1.0e3
    # force = K_Force(f, beam, (1,))

    # # line distributed body force
    # if statics:
    #     l = lambda t, xi: t * (0.5 * e2 - e3) * 5e1
    # else:
    #     l = lambda t, xi: (0.5 * e2 - e3) * 5e0
    # line_force = DistributedForce1DBeam(l, beam)

    # assemble the system
    system = System()
    system.add(rod)
    system.add(frame1)
    system.add(joint1)
    system.add(moment)
    system.add(force_bob)
    system.assemble()

    if statics:
        n_load_steps = 10
        solver = Newton(
            system,
            n_load_steps=n_load_steps,
            max_iter=30,
            atol=atol,
        )
        # solver = Riks(
        #     system,
        #     atol=1e-8,
        # )
    else:
        t1 = 1
        dt = 2.5e-2
        solver = EulerBackward(system, t1, dt, method="index 1")
        # solver = EulerBackward(system, t1, dt, method="index 2")
        # solver = EulerBackward(system, t1, dt, method="index 3")
        # solver = EulerBackward(system, t1, dt, method="index 2 GGL")
        # solver = ScipyIVP(system, t1, dt, rtol=1.0e-2, atol=1.0e-2)
        # solver = ScipyIVP(system, t1, dt, rtol=1.0e-2, atol=1.0e-2, method="Radau")

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # ###########
    # # animation
    # ###########
    # animate_beam(t, q, [rod], L, show=True)

    ############
    # VTK export
    ############
    path = Path(__file__)
    e = Export(path.parent, path.stem, True, 30, sol)
    e.export_contr(rod, level="centerline + directors", num=20)
    e.export_contr(rod, level="volume", n_segments=5, num=50)
