from cardillo.math import e1, e2, e3
from cardillo.beams import (
    CircularCrossSection,
    Simo1986,
)
from cardillo.beams.cosseratRodPGMixed import CosseratRodPG_SE3Mixed
from cardillo.beams import K_R12_PetrovGalerkin_Quaternion
from cardillo.beams import K_SE3_PetrovGalerkin_Quaternion
from cardillo.beams._fitting import fit_configuration

from cardillo.constraints import RigidConnection

from cardillo.forces import Force, Moment

from cardillo import System
from cardillo.solver import Newton
from cardillo.visualization import Export

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cardillo.beams import animate_beam

###################
# R12 interpolation
###################
# Rod = K_R12_PetrovGalerkin_Quaternion

#####################
# SE(3)-interpolation
#####################
# Rod = K_SE3_PetrovGalerkin_Quaternion
Rod = CosseratRodPG_SE3Mixed

nturns = 10

# Marino2017: nnodes = 151, p = 6
# => 150 two-node elements
# =>  75 three-node elements

# discretization
# p = 1
# # elements_per_turn = 10
# # elements_per_turn = 15 # Marino20171
# # elements_per_turn = 20

p = 2
elements_per_turn = 7.5

nelements = int(elements_per_turn * nturns)

############
# Marino2017
############
rho = 1  # [kg / m^3]
E = 1e11
# E = 1e5 # [g, mm, ms, N, MPa], see https://www.dynasupport.com/howtos/general/consistent-units
nu = 0.2
G = E / (2 * (1 + nu))

# 1mm cross sectional diameter
wire_diameter = 1e-3
# wire_diameter = 1 # [g, mm, ms, N, MPa], see https://www.dynasupport.com/howtos/general/consistent-units
wire_radius = wire_diameter / 2

# helix parameter
helix_height = 5e-2
coil_radius = 1e-2
# helix_height = 50 # [g, mm, ms, N, MPa], see https://www.dynasupport.com/howtos/general/consistent-units
# coil_radius = 10 # [g, mm, ms, N, MPa], see https://www.dynasupport.com/howtos/general/consistent-units
coil_diameter = 2 * coil_radius
pitch_unloaded = helix_height / nturns

alpha = np.arctan(pitch_unloaded / (2 * np.pi * coil_radius))
c = pitch_unloaded / (coil_radius * 2 * np.pi)
k = G * wire_diameter**4 / (64 * nturns * coil_radius**3)
delta = k * coil_radius**2 * (1 + nu * np.cos(alpha) ** 2)
eps = k * coil_radius * nu * np.sin(2 * alpha)

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
# exit()

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
# if Rod is K_SE3_PetrovGalerkin_Quaternion:
#     Q0 = Rod.straight_configuration(nelements, L=1)
#     rod = Rod(
#         cross_section,
#         material_model,
#         A_rho0,
#         K_S_rho0,
#         K_I_rho0,
#         nelements,
#         Q0,
#     )
# elif Rod is K_R12_PetrovGalerkin_Quaternion:
#     basis = "Lagrange"
#     Q0 = Rod.straight_configuration(
#         p,
#         p,
#         basis,
#         basis,
#         nelements,
#         L=1,
#     )
#     rod = Rod(
#         cross_section,
#         material_model,
#         A_rho0,
#         K_S_rho0,
#         K_I_rho0,
#         polynomial_degree_r=p,
#         polynomial_degree_psi=p,
#         nelement=nelements,
#         Q=Q0,
#         basis_r=basis,
#         basis_psi=basis,
#         volume_correction=False,
#     )
# else:
#     raise NotImplementedError

Q0 = Rod.straight_configuration(nelements, L=1, mixed=True,)
rod = Rod(
    cross_section,
    material_model,
    A_rho0,
    K_S_rho0,
    K_I_rho0,
    nelements,
    Q0,
    reduced_integration=False,
    mixed=True,
)


print(f"rod.nnodes_r: {rod.nnodes_r}")


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


nxi = nturns * 15
xis = np.linspace(0, 1, num=nxi)

r_OPs = np.array([r(xi, phi0=np.pi) for xi in xis])
A_IKs = np.array([A_IK(xi, phi0=np.pi) for xi in xis])

Q0 = fit_configuration(rod, r_OPs, A_IKs, nodal_cDOF=[])
rod.q0 = Q0.copy()
rod.Q = Q0.copy()


def run(case="force"):
    # def run(case="moment"):
    #############################################
    # joint between origin and top side of spring
    #############################################
    system = System()
    joint = RigidConnection(system.origin, rod, frame_ID2=(0,))
    system.add(rod, joint)

    ################
    # external force
    ################
    if case == "force":
        f_max = 100
        # f_max = 0
        f_t = lambda t: t * e3 * f_max
        force = Force(f_t, rod, frame_ID=(1,))
        system.add(force)
    elif case == "moment":
        f_max = 0.0445
        # f_max *= 1e3
        f_t = lambda t: t * e1 * f_max
        moment = Moment(f_t, rod, frame_ID=(1,))
        system.add(moment)
    else:
        raise NotImplementedError

    #####################
    # assemble the system
    #####################
    system.assemble()

    #####################
    # solve static system
    #####################
    # n_load_steps = 10
    # n_load_steps = 20
    # n_load_steps = 50
    # n_load_steps = 120
    n_load_steps = 150
    # n_load_steps = 300
    # n_load_steps = 400
    # n_load_steps = 500  # works with volume correction
    # n_load_steps = 700  # works without volume correction
    # n_load_steps = 1000  # works without volume correction
    sol = Newton(
        system,
        n_load_steps=n_load_steps,
    ).solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    ############
    # VTK export
    ############
    path = Path(__file__)
    e = Export(path.parent, path.stem, True, 30, sol)
    e.export_contr(
        rod, continuity="C0", level="volume", n_segments=nelements, num=3 * nelements
    )

    ################################
    # plot characteristic quantities
    ################################
    r_OPs = np.array(
        [
            rod.r_OP(ti, qi[rod.qDOF][rod.elDOF[-1]], frame_ID=(1,))
            for (ti, qi) in zip(sol.t, sol.q)
        ]
    )
    A_IKs = np.array(
        [
            rod.A_IK(ti, qi[rod.qDOF][rod.elDOF[-1]], frame_ID=(1,))
            for (ti, qi) in zip(sol.t, sol.q)
        ]
    )

    # applied forces/ moments and displacements
    fs = np.array([f_t(ti) for ti in sol.t])
    Delta_z = r_OPs[:, -1] - r_OPs[0, -1]

    #################
    # export solution
    #################
    np.savetxt(
        Path(__file__).parent / f"force_displacement.txt",
        np.hstack([t[:, None], r_OPs, fs, Delta_z[:, None]]),
        delimiter=", ",
        header="t, x, y, z, fx, fy, fz, Delta_z",
        comments="",
    )

    #####################
    # spring dispalcement
    #####################
    print(f"max. applied force/ moment: {f_max}")
    delta_z = r_OPs[-1] - r_OPs[0]
    k_rod = (f_t(t[0]) - f_t(t[1]))[-1] / (r_OPs[0, -1] - r_OPs[1, -1])
    print(f"delta_z: {delta_z}")
    print(f"k_rod: {k_rod}")
    print(f"|k - k_rod|: {np.abs(k - k_rod)}")

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(Delta_z, fs[:, -1], "-ob", label="rod")
    ax[0].plot(Delta_z, Delta_z * k, "--k", label="k z")
    ax[0].set_xlabel("Delta z")
    ax[0].set_ylabel("f/ m")
    ax[0].grid()
    ax[0].legend()

    plt.show()


if __name__ == "__main__":
    run()
