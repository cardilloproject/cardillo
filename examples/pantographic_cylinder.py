from cardillo.math import e3
from cardillo.beams import (
    RectangularCrossSection,
    Simo1986,
)
from cardillo.constraints import RigidConnection

# from cardillo.beams import K_R12_PetrovGalerkin_Quaternion as Rod

from cardillo.beams import K_R12_PetrovGalerkin_AxisAngle as Rod

# from cardillo.beams import K_SE3_PetrovGalerkin_AxisAngle as Rod
# from cardillo.beams import K_SE3_PetrovGalerkin_Quaternion as Rod
# from cardillo.beams import K_SE3_PetrovGalerkin_R9 as Rod
# from cardillo.beams import K_R3_SO3_PetrovGalerkin_AxisAngle as Rod
from cardillo.beams import animate_beam
# from cardillo.forces import K_Moment
from cardillo import System
from cardillo.solver import Newton
from cardillo.discrete import Frame
from cardillo.beams._fitting import fit_configuration

from cardillo.utility import Export

import numpy as np
import pickle

if __name__ == "__main__":
    # number of elements
    nelements = 8

    # used polynomial degree
    polynomial_degree = 2
    basis = "Lagrange"

    # Young's and shear modulus
    E = 1.0
    G = 0.5

    # length of the rod
    L = 1.0e3

    # slenderness and corresponding absolute tolerance for Newton-Raphson solver
    slenderness = 1.0e2
    atol = 1.0e-10

    # used cross section
    width = L / slenderness

    # cross section
    line_density = 1
    cross_section = RectangularCrossSection(line_density, width, width)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment
    A = cross_section.area

    # quadratic beam material
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

    # ccw rod
    Q0 = Rod.straight_configuration(
        polynomial_degree,
        polynomial_degree,
        basis,
        basis,
        nelements,
        L,
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
        Q=Q0,
        q0=Q0,
        basis_r=basis,
        basis_psi=basis,
    )

    # pivot
    hp = 1  # pivot height

    # helix
    n_coils = 2  # number of helix coils
    scale = 1.0e1
    RO = 1 * scale  # helix outer radius
    RI = 1 * scale - hp
    h = 2 * scale  # helix height
    cO = h / (RO * 2 * np.pi * n_coils)
    LO = np.sqrt(1 + cO**2) * RO * 2 * np.pi * n_coils
    cI = h / (RI * 2 * np.pi * n_coils)
    LI = np.sqrt(1 + cI**2) * RI * 2 * np.pi * n_coils
    # print(f"R0: {R0}")
    print(f"h: {h}")
    # print(f"c: {c}")
    print(f"n: {n_coils}")
    # print(f"L: {L}")

    # reference solution
    def r(xi, R=RO, phi0=0., dor=1, c=cO):
        alpha = dor * 2 * np.pi * n_coils * xi 
        return R * np.array([np.sin(alpha + phi0), -np.cos(alpha + phi0), dor * c * alpha])

    def A_IK(xi, phi0=0., dor=1, c=cO):
        alpha = dor * 2 * np.pi * n_coils * xi
        sa = np.sin(alpha + phi0)
        ca = np.cos(alpha + phi0)

        e_x = dor * np.array([ca, sa, dor * c]) / np.sqrt(1 + c**2)
        e_y = dor * np.array([-sa, ca, 0])
        e_z = np.array([-c * dor * ca, -dor * c * sa, 1]) / np.sqrt(1 + c**2)

        return np.vstack((e_x, e_y, e_z)).T

    nxi = 30
    xis = np.linspace(0, 1, num=nxi)

    import matplotlib.pyplot as plt

    # ax = plt.axes(projection="3d")
    # for xi in xis:
    #     ax.plot3D(*r(xi, dor=-1, R=RI, c=cI))
    #     ax.quiver(*r(xi, dor=-1, R=RI, c=cI), *A_IK(xi, dor=-1, c=cI).T[0])
    #     ax.quiver(*r(xi, dor=-1, R=RI, c=cI), *A_IK(xi, dor=-1, c=cI).T[1])
    #     ax.quiver(*r(xi, dor=-1, R=RI, c=cI), *A_IK(xi, dor=-1, c=cI).T[2])

    #     ax.plot3D(*r(xi, dor=1))
    #     ax.quiver(*r(xi, dor=1), *A_IK(xi, dor=1).T[0])
    #     ax.quiver(*r(xi, dor=1), *A_IK(xi, dor=1).T[1])
    #     ax.quiver(*r(xi, dor=1), *A_IK(xi, dor=1).T[2])
    # plt.show()

    # individual rods
    n_rod = 4  # number of rods per layer
    Q0_list = []
    rod_list = []
    joint_list = []

    # load config
    load_config = False
    from pathlib import Path
    import copy

    path = Path(__file__)
    # path.mkdir()
    filename = Path(path.parent, "initial_config")
    if load_config:
        Q0_list = pickle.load(open(filename, "rb"))
        for n in range(n_rod):
            rod_ccw = copy.deepcopy(rod)
            rod_ccw.q0 = Q0_list[2*n].copy()
            rod_ccw.set_initial_strains(rod_ccw.q0)
            
            rod_cw = copy.deepcopy(rod)
            rod_cw.q0 = Q0_list[2*n+1].copy()
            rod_cw.set_initial_strains(rod_cw.q0)
            
            rod_list.extend((rod_ccw,rod_cw))
    else:
        for n in range(n_rod):
            rod_ccw = copy.deepcopy(rod)
            phi0 = 2 * np.pi * n / (n_rod)
            r_OP_ccw = np.array([r(xi, R=RO, dor=1, c=cO, phi0=phi0) for xi in xis])
            A_IK_ccw = np.array([A_IK(xi, dor=1, phi0=phi0) for xi in xis])
            Q0_ccw = fit_configuration(rod_ccw, r_OP_ccw, A_IK_ccw)
            rod_ccw.q0 = Q0_ccw.copy()
            Q0_list.append(Q0_ccw)
            rod_list.append(rod_ccw)

            rod_cw = copy.deepcopy(rod)
            r_OP_cw = np.array([r(xi, R=RI, dor=-1, c=cI, phi0=phi0) for xi in xis])
            A_IK_cw = np.array([A_IK(xi, dor=-1, c=cI, phi0=phi0) for xi in xis])
            Q0_cw = fit_configuration(rod_cw, r_OP_cw, A_IK_cw)
            Q0_list.append(Q0_cw)
            rod_cw.q0 = Q0_cw.copy()
            rod_list.append(rod_cw)

        file = open(filename, "wb")
        pickle.dump(Q0_list, file)

    # joints between frames and rods
    system = System()
    Z_max = r(1)[-1]
    r_OP_top = lambda t: np.array([0, 0, Z_max - 2 * t])
    frame_top = Frame(r_OP=r_OP_top, A_IK=np.eye(3))
    for rod in rod_list:
        joint_bottom = RigidConnection(system.origin, rod, frame_ID2=(0,))
        joint_top = RigidConnection(frame_top, rod, frame_ID2=(1,))
        joint_list.extend((joint_bottom, joint_top))

    # moment at right beam's tip
    # Fi = material_model.Fi
    # m = Fi[2] * 2 * np.pi / (2 * L) * 0.25 * 0.1
    # M = lambda t: t * e3 * m
    # moment = K_Moment(M, rod, (1,))

    # assemble the system
    system.add(*rod_list)
    system.add(*joint_list)
    system.add(frame_top)
    # system.add(moment)
    system.assemble()

    # solve static system
    n_load_steps = 10
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        atol=atol,
        max_iter=50,
    )
    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # t = [0, 1]
    # q = [q0, q0]

    ###########
    # animation
    ###########
    # animate_beam(t, q, rod_list, 2 * scale, show=True)

    ###########
    # export
    ########### 
    e = Export(path.parent, path.stem, True, 10, sol)
    for rod in rod_list:
        e.export_contr(rod, level="centerline + directors", num=50)