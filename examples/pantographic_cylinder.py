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
from cardillo.forces import K_Moment
from cardillo import System
from cardillo.solver import Newton
from cardillo.discrete import Frame
from cardillo.beams._fitting import fit_configuration

import numpy as np
import pickle

if __name__ == "__main__":
    # number of elements
    nelements = 4

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

    # left rod
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
    hp = 5 # pivot height

    # helix
    n_coils = 1  # number of helix coils
    scale = 1.0e1
    RO = 1 * scale  # helix outer radius
    RI = 1 * scale - hp
    h = 1 * scale  # helix height
    cO = h / (RO * 2 * np.pi * n_coils)
    LO = np.sqrt(1 + cO**2) * RO * 2 * np.pi * n_coils
    cI = h / (RI * 2 * np.pi * n_coils)
    LI = np.sqrt(1 + cI**2) * RI * 2 * np.pi * n_coils
    # print(f"R0: {R0}")
    print(f"h: {h}")
    #print(f"c: {c}")
    print(f"n: {n_coils}")
    #print(f"L: {L}")

    # reference solution
    def r(xi, R=RO, phi0=0, dor=1, c=cO):
        alpha = dor * 2 * np.pi * n_coils * xi + phi0
        return R * np.array([np.sin(alpha), -np.cos(alpha), dor * c * alpha])

    def A_IK(xi, phi0=0, dor=1, c=cO):
        alpha = 2 * np.pi * n_coils * xi + phi0
        sa = np.sin(alpha)
        ca = np.cos(alpha)

        e_x = np.array([dor*ca, sa, c]) / np.sqrt(1 + c**2)
        e_y = np.array([-sa, dor*ca, 0])
        e_z = np.array([-dor*c * ca, -c * sa, 1]) / np.sqrt(1 + c**2)

        return np.vstack((e_x, e_y, e_z)).T



    nxi = 30
    xis = np.linspace(0, 1, num=nxi)

    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')
    for xi in xis:
        ax.plot3D(*r(xi,dor=-1,R=RI,c=cI))
        ax.quiver(*r(xi,dor=-1,R=RI,c=cI),*A_IK(xi,dor=-1,c=cI).T[0])
        ax.quiver(*r(xi,dor=-1,R=RI,c=cI),*A_IK(xi,dor=-1,c=cI).T[1])
        ax.quiver(*r(xi,dor=-1,R=RI,c=cI),*A_IK(xi,dor=-1,c=cI).T[2])

        ax.plot3D(*r(xi,dor=1))
        ax.quiver(*r(xi,dor=1),*A_IK(xi,dor=1).T[0])
        ax.quiver(*r(xi,dor=1),*A_IK(xi,dor=1).T[1])
        ax.quiver(*r(xi,dor=1),*A_IK(xi,dor=1).T[2])
    plt.show()

    # individual rods
    n_rod = 1 # number of rods per layer
    Q0_list = []
    rod_list = []
    joint_list = []

    # load config
    load_config = False
    from pathlib import Path
    path = Path(Path.cwd(), "examples/pantographic_cylinder")
    # path.mkdir()
    filename = Path(path, "initial_config")
    if load_config:
        Q0_list = pickle.load(open(filename, 'rb'))
    else:
        for n in range(n_rod):
            phi0 = 2 * np.pi * n / (n_rod + 1)
            r_OPs = np.array([r(xi, R=RO, dor=1, c=cO) for xi in xis])
            A_IKs = np.array([A_IK(xi, dor=1) for xi in xis])
            Q0_helix = fit_configuration(rod, r_OPs, A_IKs)
            Q0_list.append(Q0_helix)
            r_OPs = np.array([r(xi, R=RI, dor=-1, c=cI) for xi in xis])
            A_IKs = np.array([A_IK(xi, dor=-1, c=cI) for xi in xis])
            Q0_helix = fit_configuration(rod, r_OPs, A_IKs)
            Q0_list.append(Q0_helix)

    file = open(filename, 'wb')
    pickle.dump(Q0_list, file)

    import copy
    for Q0_helix in Q0_list:
        rod_list.append(copy.deepcopy(rod))
        rod_list[-1].q0 = Q0_helix.copy()

    # joints between frames and rods
    system = System()
    Z_max = r(1)[-1]
    r_OP_top = lambda t: np.array([0, 0, Z_max - 1*t])
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
    n_load_steps = 5
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        atol=atol,
        max_iter=10,
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
    animate_beam(t, q, rod_list, 2 * scale, show=True)
