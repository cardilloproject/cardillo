from cardillo.math import e1, e2, e3, ax2skew
from cardillo.beams import (
    CircularCrossSection,
    Simo1986,
)
from cardillo.discrete import Frame, PointMass

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

from cardillo.constraints import RigidConnection, Spherical

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
    GeneralizedAlphaFirstOrder,
    NPIRK,
)
from cardillo.solver._butcher_tableaus import RadauIIATableau
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
# Rod = K_R12_PetrovGalerkin_Quaternion

#####################
# SE(3)-interpolation
#####################
# Rod = K_SE3_PetrovGalerkin_AxisAngle
Rod = K_SE3_PetrovGalerkin_Quaternion

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
    ###########################
    # discretization properties
    ###########################
    # nelements, nturns = 4, 1
    # nelements, nturns = 8, 1
    nelements, nturns = 20, 1
    # nelements, nturns = 12, 2
    # nelements, nturns = 24, 4
    # nelements, nturns = 48, 8
    # nelements, nturns = 150, 10
    # nelements, nturns = 96, 16
    # nelements, nturns = 120, 20 # works for SE(3)-quaternion

    # nelements, nturns = 1000, 130

    # nelements = 16  # 2 turns
    # nelements = 32 # 5 turns
    # nelements = 64 # 10 turns
    # nelements = 128 # 20 turns

    ####################################################################
    # beam parameters, taken from
    # https://faraday.physics.utoronto.ca/PHY182S/WilberforceRefBerg.pdf
    ####################################################################

    # ############
    # # Harsch2021
    # ############
    # # Federstahl nach EN 10270-1
    # rho = 7850  # [kg / m^3]
    # # E = 206e9  # Pa
    # # G = 81.5e9  # Pa

    ##########
    # Berg1991
    ##########
    rho = 7850  # [kg / m^3]
    G = 8.1e10
    nu = 0.23
    E = 2 * G * (1 + nu)

    # 1mm cross sectional diameter
    wire_diameter = 1e-3
    wire_radius = wire_diameter / 2

    # helix parameter
    # coil_diameter = 32.0e-3  # 32mm
    # coil_radius = coil_diameter / 2
    coil_radius = 15.35e-3
    coil_diameter = 2 * coil_radius
    pitch_unloaded = 1.0e-3  # 1mm
    # pitch_unloaded = 0
    c = pitch_unloaded / (coil_radius * 2 * np.pi)

    # Berg1991
    k = G * wire_diameter**4 / (64 * nturns * coil_radius**3)
    # delta = ???
    print(f"k: {k}")

    # ############
    # # Marino2017
    # ############
    # rho = 7850  # [kg / m^3]
    # E = 1e11
    # nu = 0.2
    # G = E / (2 * (1 + nu))

    # wire_diameter = 1e-3
    # wire_radius = wire_diameter / 2
    # coil_diameter = 2e-2
    # coil_radius = coil_diameter / 2

    # pitch_unloaded = 5e-3
    # c = pitch_unloaded / (coil_radius * 2 * np.pi)

    # rod cross-section
    cross_section = CircularCrossSection(rho, wire_radius)
    A_rho0 = rho * cross_section.area
    K_S_rho0 = rho * cross_section.first_moment
    K_I_rho0 = rho * cross_section.second_moment
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

    print(f"Ei: {Ei}")
    print(f"Fi: {Fi}")

    ##########
    # gravity
    #########
    g = 9.81

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

    # #############
    # # rod gravity
    # #############
    # if statics:
    #     f_g_rod = lambda t, xi: -t * A_rho0 * g * e3
    # else:
    #     f_g_rod = -A_rho0 * g * e3
    # force_rod = DistributedForce1DBeam(f_g_rod, rod)

    #############################################
    # joint between origin and top side of spring
    #############################################
    system = System()
    joint1 = RigidConnection(system.origin, rod, frame_ID2=(1,))

    # #############################
    # # external fore at spring end
    # #############################
    # # pm = PointMass(1, q0=np.zeros(3))
    # # joint2 = Spherical(rod, pm, r_OB0=np.zeros(3), frame_ID1=(0,))

    # rb = RigidBodyQuaternion(
    #     mass=1, K_Theta_S=np.eye(3), q0=np.array([0, 0, 0, 1, 0, 0, 0])
    # )
    # joint2 = RigidConnection(rb, rod, frame_ID2=(0,))
    # assert statics
    # f_g = lambda t: -t * e3 * 100
    # force = Force(f_g, rb)

    ##############
    # pendulum bob
    ##############
    R = 18e-3  # radius of the main cylinder
    h = 50e-3  # height of the main cylinder
    m, K_Theta_S = Wilberforce_bob(R, h)
    # # TODO:
    # scale = 4
    # m *= scale
    # K_Theta_S *= scale
    # center of mass is shifted (wire starts out of the top cylinder surface)
    r_OS0 = np.array([0, 0, -h / 2 - wire_radius])
    p0 = np.array([1, 0, 0, 0], dtype=float)
    q0 = np.concatenate((r_OS0, p0))
    bob = Cylinder(RigidBodyQuaternion)(
        length=h, radius=R, axis=2, mass=m, K_Theta_S=K_Theta_S, q0=q0
    )
    # bob = RigidBodyQuaternion(
    #     mass=m, K_Theta_S=K_Theta_S, q0=q0
    # )
    # density = 7850  # [kg / m^3]; steel
    # bob = Cylinder(RigidBodyQuaternion)(
    #     length=h, radius=R, axis=2, density=density, q0=q0
    # )

    ########################
    # connect spring and bob
    ########################
    joint2 = RigidConnection(bob, rod, frame_ID2=(0,))

    ################
    # external force
    ################
    if statics:
        f_g_bob = lambda t: -t * m * g * e3
    else:
        f_g_bob = lambda t: -m * g * e3
    force_bob = Force(f_g_bob, bob)

    #####################
    # assemble the system
    #####################
    # system.add(rod, joint1, force_rod)
    system.add(rod, joint1)
    # system.add(rb, joint2, force)
    system.add(bob, joint2, force_bob)
    system.assemble()

    #####################
    # solve static system
    #####################
    if statics:
        # n_load_steps = 50
        n_load_steps = 500
        solver = Newton(
            system,
            n_load_steps=n_load_steps,
        )
    else:
        # t1 = 30
        # t1 = 10
        # t1 = 3
        # t1 = 1
        t1 = 0.5
        # t1 = 1e-1
        # dt = 1e-2
        # dt = 5e-3
        dt = 1e-3

        # solver = EulerBackward(system, t1, dt, method="index 3")
        # solver = ScipyIVP(system, t1, dt)
        # solver = ScipyIVP(system, t1, dt, method="RK23", rtol=1e-3, atol=1e-3)
        # solver = RadauIIa(system, t1, dt, dae_index=2, max_step=dt)
        # solver = SimplifiedNonsmoothGeneralizedAlpha(system, t1, dt, rho_inf=0.8)
        # solver = NPIRK(system, t1, dt, RadauIIATableau(2))
        solver = GeneralizedAlphaFirstOrder(
            system,
            t1,
            dt,
            rho_inf=0.8,
            atol=1e-6,
            method="index 3",
        )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # #####################
    # # spring dispalcement
    # #####################
    # # r_OP = np.array([
    # #     rod.r_OP(ti, qi[rod.qDOF][rod.local_qDOF_P((0,))], frame_ID=(0,)) for ti, qi in zip(sol.t, sol.q)
    # # ])
    # r_OP = np.array([rb.r_OP(ti, qi[rb.qDOF]) for ti, qi in zip(sol.t, sol.q)])

    # f = f_g(1)
    # delta = r_OP[-1] - r_OP[0]
    # k_rod = (f_g(t[-1]) - f_g(t[-2])) / (r_OP[-1] - r_OP[-2])
    # print(f"applied force: {f}")
    # print(f"spring displacement: {delta}")
    # print(f"k_rod: {k_rod}")

    # fig, ax = plt.subplots()
    # ax.plot(sol.t, r_OP[:, -1], label="z")
    # ax.grid()
    # ax.legend()

    # plt.show()

    # ############
    # # VTK export
    # ############
    # path = Path(__file__)
    # e = Export(path.parent, path.stem, True, 30, sol)
    # e.export_contr(rod, level="volume", n_segments=nelements, num=3 * nelements)

    # exit()

    ################################
    # plot characteristic quantities
    ################################
    r_OS = np.array([bob.r_OP(ti, qi[bob.qDOF]) for (ti, qi) in zip(sol.t, sol.q)])

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

    # plt.show()

    ###########
    # animation
    ###########
    _ = animate_beam(t, q, [rod], 0.05, scale_di=0.01, show=False)

    ############
    # VTK export
    ############
    path = Path(__file__)
    e = Export(path.parent, path.stem, True, 30, sol)
    e.export_contr(rod, level="volume", n_segments=nelements, num=3 * nelements)
    e.export_contr(bob)

    plt.show()
