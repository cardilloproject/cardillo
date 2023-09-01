from cardillo.math import e1, e2, e3, ax2skew
from cardillo.beams import (
    CircularCrossSection,
    Simo1986,
)
from cardillo.beams import (
    K_R12_PetrovGalerkin_AxisAngle,
    K_R12_PetrovGalerkin_Quaternion,
)
from cardillo.beams import (
    K_SE3_PetrovGalerkin_AxisAngle,
    K_SE3_PetrovGalerkin_Quaternion,
)
from cardillo.beams import K_PetrovGalerkinQuaternionInterpolation
from cardillo.beams._fitting import fit_configuration, fit_configuration_quaternion

from cardillo.constraints import RigidConnection, Spherical

from cardillo.discrete.shapes import Cylinder
from cardillo.discrete import RigidBodyQuaternion

from cardillo.forces import Force, Moment
from cardillo.forces import DistributedForce1DBeam

from cardillo import System
from cardillo.solver import (
    Newton,
    GeneralizedAlphaFirstOrder,
    EulerBackward,
    MoreauClassical,
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
Rod = K_R12_PetrovGalerkin_Quaternion

#####################
# SE(3)-interpolation
#####################
# Rod = K_SE3_PetrovGalerkin_AxisAngle
# Rod = K_SE3_PetrovGalerkin_Quaternion

# Rod = K_PetrovGalerkinQuaternionInterpolation

p = 1
# elements_per_turn = 25
# elements_per_turn = 20 # R12 + p1 + no volume correction
# elements_per_turn = 15
# elements_per_turn = 10 # R12 + p1 + volume correction
elements_per_turn = 15 # quaternion interpolation

# p = 2
# elements_per_turn = 6 # quaternion interpolation
# elements_per_turn = 6  # R12 + p2 + no volume correction
# elements_per_turn = 8 # SE(3)
nturns = 3
# # elements_per_turn = 12
# # nturns = 1

nelements = int(elements_per_turn * nturns)

##########
# Berg1991
##########
rho = 7850  # [kg / m^3]
G = 8.1e10
nu = 0.23
E = 2 * G * (1 + nu)
print(f"G: {G}; E: {E}")

# 1mm cross sectional diameter
wire_diameter = 1e-3
wire_radius = wire_diameter / 2

# helix parameter
# TODO: Should we 15/32 mm?
coil_diameter = 32.0e-3
coil_radius = coil_diameter / 2
# coil_radius = 15.35e-3
# coil_diameter = 2 * coil_radius
pitch_unloaded = 1.0e-3
# pitch_unloaded = 1.4e-2

alpha = np.arctan(pitch_unloaded / (2 * np.pi * coil_radius))
# alpha = 8.35 * np.pi / 180
c = pitch_unloaded / (coil_radius * 2 * np.pi)
k = G * wire_diameter**4 / (64 * nturns * coil_radius**3)
delta = k * coil_radius**2 * (1 + nu * np.cos(alpha) ** 2)
eps = k * coil_radius * nu * np.sin(2 * alpha)
# eps *= 450
# eps *= 50

# ##########################
# # Wahl1944, p. 48 eq. (52)
# ##########################
# spring_index = 2 * coil_radius / wire_diameter
# psi = (
#     np.cos(alpha) / (1 + (3 / 16) * np.cos(alpha) ** 4 / (spring_index**2 - 1))
#     + 2 * G * np.sin(alpha) * np.tan(alpha) / E
# )
# k /= psi
# delta = k * coil_radius**2 * (1 + nu * np.cos(alpha)**2)
# eps = k * coil_radius * nu * np.sin(2 * alpha)

print(f"k: {k}; delta: {delta}; eps: {eps}")

#########
# gravity
#########
g = 9.81

##############
# pendulum bob
##############
R = 23e-3  # radius of the main cylinder
h = 36e-3  # height of the main cylinder
# R = np.sqrt(2 * delta / k)
density = 7850  # [kg / m^3]; steel
r_OS0 = np.array([0, 0, -h / 2 - wire_radius])
p0 = np.array([1, 0, 0, 0], dtype=float)
q0 = np.concatenate((r_OS0, p0))
bob = Cylinder(RigidBodyQuaternion)(length=h, radius=R, axis=2, density=density, q0=q0)
omega_z = np.sqrt(k / bob.mass)
omega_theta = np.sqrt(delta / bob.K_Theta_S[-1, -1])
print(
    f"omega_z = np.sqrt(k / m): {omega_z}; omega_theta = np.sqrt(delta / Izz): {omega_theta}"
)
print(
    f"|omega_z - omega_theta| / |omega_z|: {np.abs(omega_z - omega_theta) / np.abs(omega_z)}"
)

omega = 28.6
omega_B = 2.56
eps_rod = 2 * omega * omega_B * np.sqrt(bob.mass * bob.K_Theta_S[-1, -1])
eps = eps_rod
print(f"eps_rod: {eps_rod}")

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
        # volume_correction=True,
        volume_correction=False,
    )
elif Rod is K_PetrovGalerkinQuaternionInterpolation:
    basis = "Lagrange"
    Q0 = Rod.straight_configuration(
        p,
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
        p,
        nelements,
        Q=Q0,
        q0=Q0,
        basis=basis,
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


# nxi = int(nturns * 15) # p = 1 quaternion
# nxi = int(nturns * 30) # p=2 quaternion
# nxi = int(nturns * elements_per_turn * 5) # p=2 quaternion
nxi = int(nturns * elements_per_turn) # p=1 quaternion
xis = np.linspace(0, 1, num=nxi)

r_OPs = np.array([r(xi, phi0=np.pi) for xi in xis])
A_IKs = np.array([A_IK(xi, phi0=np.pi) for xi in xis])

if Rod is K_PetrovGalerkinQuaternionInterpolation:
    Q0 = fit_configuration(rod, r_OPs, A_IKs, nodal_cDOF=[])
    # Q0 = fit_configuration_quaternion(rod, r_OPs, A_IKs, nodal_cDOF=[])
    # Q0 = fit_configuration_quaternion(rod, r_OPs, A_IKs, nodal_cDOF=[0, -1])
else:
    Q0 = fit_configuration(rod, r_OPs, A_IKs, nodal_cDOF=[])
rod.q0 = Q0.copy()


def run_FEM_statics(use_force=True):
    # def run_FEM_statics(use_force=False):

    # #############
    # # rod gravity
    # #############
    # f_g_rod = lambda t, xi: -t * A_rho0 * g * e3
    # force_rod = DistributedForce1DBeam(f_g_rod, rod)

    #############################################
    # joint between origin and top side of spring
    #############################################
    system = System()
    joint1 = RigidConnection(system.origin, rod, frame_ID2=(1,))

    #############################
    # external fore at spring end
    #############################
    joint2 = RigidConnection(bob, rod, frame_ID2=(0,))

    if use_force:
        f_t = lambda t: -t * e3 * 10
        force = Force(f_t, bob)
    else:
        f_t = lambda t: -t * e3 * 0.05
        moment = Moment(f_t, bob)

    #####################
    # assemble the system
    #####################
    # system.add(rod, joint1, force_rod)
    system.add(rod, joint1)
    system.add(bob, joint2)
    if use_force:
        system.add(force)
    else:
        system.add(moment)
    system.assemble()

    #####################
    # solve static system
    #####################
    n_load_steps = 10
    # n_load_steps = 50
    sol = Newton(
        system,
        n_load_steps=n_load_steps,
    ).solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

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

    # fig, ax = plt.subplots(2, 1)

    # ax[0].plot(t, r_OS[:, 0], label="x")
    # ax[0].plot(t, r_OS[:, 1], label="y")
    # ax[0].plot(t, r_OS[:, 2], label="z")
    # ax[0].legend()
    # ax[0].grid()

    # ax[1].plot(t, angles[:, 0], label="alpha")
    # ax[1].plot(t, angles[:, 1], label="beta")
    # ax[1].plot(t, angles[:, 2], label="gamma")
    # ax[1].legend()
    # ax[1].grid()

    # plt.show()

    #####################
    # spring dispalcement
    #####################
    f_max = f_t(1)

    if use_force:
        print(f"max. applied force: {f_max}")
        delta_z = r_OS[-1] - r_OS[0]
        k_rod = (f_t(t[0]) - f_t(t[1]))[-1] / (r_OS[0, -1] - r_OS[1, -1])
        print(f"delta_z: {delta_z}")
        print(f"k_rod: {k_rod}")
        print(f"|k - k_rod|: {np.abs(k - k_rod)}")
    else:
        print(f"max. applied moment: {f_max}")
        delta_alpha = angles[-1, 0] - angles[0, 0]
        delta_rod = (f_t(t[0]) - f_t(t[1]))[-1] / (angles[0, 0] - angles[1, 0])
        print(f"delta_alpha: {delta_alpha}")
        print(f"delta_rod: {delta_rod}")
        print(f"|delta - delta_rod|: {np.abs(delta - delta_rod)}")

    fs = np.array([f_t(ti) for ti in sol.t])
    Delta_z = -(r_OS[:, -1] - r_OS[0, -1])
    Delta_angle_z = angles[:, 0] - angles[0, 0]

    fig, ax = plt.subplots(2, 1)
    if use_force:
        ax[0].plot(Delta_z, -fs[:, -1], "-ob", label="rod")
        ax[0].plot(Delta_z, Delta_z * k, "--k", label="k z")
        ax[0].plot(
            Delta_z,
            Delta_z * k + 0.5 * Delta_angle_z * eps,
            ":r",
            label="k * z + eps * theta",
        )
        ax[0].set_xlabel("Delta z")
        ax[0].set_ylabel("f")
        ax[0].grid()
        ax[0].legend()

        # ax[1].plot(Delta_angle_z, -fs[:, -1], "-ob", label="rod")
        # ax[1].plot(Delta_angle_z, 0.5 * Delta_angle_z * eps, , "--k", label="0.5 * eps * theta")
        # ax[1].set_xlabel("Delta angle_z")
        # ax[1].set_ylabel("f")
        # ax[1].grid()
        # ax[1].legend()

    else:
        ax[1].plot(Delta_angle_z, -fs[:, -1], "-ob", label="rod")
        ax[1].plot(Delta_angle_z, -Delta_angle_z * delta, "--k", label="delta * theta")
        ax[1].plot(
            Delta_angle_z,
            -Delta_angle_z * delta - 0.5 * Delta_z * eps,
            ":r",
            label="delta * theta + 0.5 * eps * z",
        )
        ax[1].set_xlabel("Delta angle_z")
        ax[1].set_ylabel("f")
        ax[1].grid()
        ax[1].legend()

    # if use_force:
    #     pass
    #     # ax[1].plot(
    #     #     [Delta_angle_z[0], Delta_angle_z[-1]],
    #     #     [
    #     #         0.5 * Delta_angle_z[0] * eps + 0.5 * displacement_z[0] * k,
    #     #         0.5 * Delta_angle_z[-1] * eps - 0.5 * displacement_z[-1] * k
    #     #     ],
    #     #     "--k",
    #     #     label="linear",
    #     # )
    # else:
    #     ax[1].plot(
    #         [Delta_angle_z[0], Delta_angle_z[-1]],
    #         [Delta_angle_z[0] * delta, Delta_angle_z[-1] * delta],
    #         "--k",
    #         label="linear",
    #     )
    # ax[1].set_xlabel("Delta alpha")
    # ax[1].set_ylabel("f")
    # ax[1].grid()
    # ax[1].legend()

    plt.show()

    ############
    # VTK export
    ############
    path = Path(__file__)
    e = Export(path.parent, path.stem, True, 30, sol)
    e.export_contr(rod, level="volume", n_segments=nelements, num=3 * nelements)
    e.export_contr(bob)


def run_FEM_dynamics():
    #############################################
    # joint between origin and top side of spring
    #############################################
    system = System()
    joint1 = RigidConnection(system.origin, rod, frame_ID2=(1,))

    ########################
    # connect spring and bob
    ########################
    joint2 = RigidConnection(bob, rod, frame_ID2=(0,))

    ################
    # external force
    ################
    f_g_bob = lambda t: -bob.mass * g * e3
    force_bob = Force(f_g_bob, bob)

    #####################
    # assemble the system
    #####################
    # system.add(rod)
    system.add(rod, joint1)
    system.add(bob, joint2, force_bob)
    system.assemble()

    animate_beam(
        t=[0, 1],
        q=[Q0, Q0],
        beams=[rod],
        scale=max(R, h),
        scale_di=0.1 * R,
        show=True,
        n_frames=50,
    )
    # animate_beam(t=[0, 1], q=[Q0, Q0], beams=[rod], scale=0.1 * R, show=True)
    # exit()

    ##################
    # time integration
    ##################
    # nturns = 1
    # t1 = 3
    # dt = 1e-2

    # nturns = 3
    # t1 = 0.01
    # t1 = 1e-2
    t1 = 4
    # t1 = 8
    # t1 = 1e-2
    # dt = 5e-4
    dt = 5e-3  # nturns = 3, p=1 quaternion interpolation
    # dt = 5e-4  # nturns = 3
    # dt = 5e-4 # nturns = 3, p=2 quaternion interpolation

    # nturns = 10 # TODO: Not working yet!
    # t1 = 10
    # dt = 1e-4 # nturns = 10

    # solver = EulerBackward(system, t1, dt, method="index 3")
    # solver = MoreauClassical(system, t1, dt)
    solver = GeneralizedAlphaFirstOrder(
        system,
        t1,
        dt,
        rho_inf=0.8,
        atol=1e-8,
        method="index 3",
        max_iter=10,
    )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # ################################
    # # plot characteristic quantities
    # ################################
    r_OS = np.array([bob.r_OP(ti, qi[bob.qDOF]) for (ti, qi) in zip(sol.t, sol.q)])

    ordering = "zyx"
    angles = np.array(
        [
            Rotation.from_matrix(bob.A_IK(ti, qi[bob.qDOF])).as_euler(ordering)
            for (ti, qi) in zip(sol.t, sol.q)
        ]
    )

    # ########
    # # export
    # ########
    # np.savetxt(
    #     Path(__file__).parent
    #     / f"wilberforce_rigid_body_state_nturns_{nturns}_t1_{t1}_dt_{dt}.txt",
    #     np.hstack([t[:, None], r_OS, angles]),
    #     delimiter=", ",
    #     header="t, x, y, z, alpha, beta, gamma",
    #     comments="",
    # )

    ###############
    # visualization
    ###############
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

    ###########
    # animation
    ###########
    _ = animate_beam(t, q, [rod], 0.05, scale_di=0.01, show=False)
    # plt.show()

    ############
    # VTK export
    ############
    path = Path(__file__)
    e = Export(path.parent, path.stem, True, 30, sol)
    if Rod in [K_R12_PetrovGalerkin_AxisAngle, K_R12_PetrovGalerkin_Quaternion, K_PetrovGalerkinQuaternionInterpolation]:
        e.export_contr(
            rod,
            continuity="C0",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
        )
        # e.export_contr(rod, continuity="C1", level="volume", n_segments=nelements, num=3 * nelements)
    else:
        e.export_contr(
            rod,
            continuity="C1",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
        )
    e.export_contr(bob)

    plt.show()


def run_ODE():
    m = bob.mass
    Izz = bob.K_Theta_S[-1, -1]

    # Berg1991
    m = 0.5164
    Izz = 1.45e-4
    # k = 2.8
    # delta = 7.86e-4
    # eps = 9.27e-3

    def eqm(t, y):
        z, theta, z_dot, theta_dot = y
        y_dot = np.array(
            [
                z_dot,
                theta_dot,
                -(k * z + 0.5 * eps * theta + m * g) / m,
                # -(k * z + 0.5 * eps * theta) / m,
                -(delta * theta + 0.5 * eps * z) / Izz,
            ]
        )
        return y_dot

    y0 = np.zeros(4, dtype=float)
    # y0 = np.array([1.68, 0, 0, 0], dtype=float)
    t_span = (0, 3)

    from scipy.integrate import solve_ivp

    sol = solve_ivp(eqm, t_span, y0, max_step=1e-2)
    t = sol.t
    y = sol.y

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, y[0], label="z")
    ax[1].plot(t, y[1], label="theta")
    plt.show()

    print(f"")


if __name__ == "__main__":
    # run_FEM_statics()
    run_FEM_dynamics()
    # run_ODE()
