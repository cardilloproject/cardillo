from cardillo.math import e3
from cardillo.beams import (
    CircularCrossSection,
    RectangularCrossSection,
    Simo1986,
)
from cardillo.constraints import RigidConnection

from cardillo.beams import K_R12_PetrovGalerkin_Quaternion as Rod

# from cardillo.beams import K_R12_PetrovGalerkin_AxisAngle as Rod

from cardillo.beams import animate_beam
from cardillo.forces import K_Moment
from cardillo import System
from cardillo.solver import Newton

from cardillo.beams._fitting import fit_configuration
from cardillo.visualization import Export

import numpy as np


if __name__ == "__main__":
    # number of elements
    nelements = 5

    # used polynomial degree
    polynomial_degree = 1
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
    density = 1
    cross_section = CircularCrossSection(density, 0.5 * width)
    # cross_section = RectangularCrossSection(density, width, width)
    A_rho0 = density * cross_section.area
    K_S_rho0 = density * cross_section.first_moment
    K_I_rho0 = density * cross_section.second_moment
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
        # A_IK=np.array([[-1,0,0],[0,-1,0],[0,0,1]],dtype=float)
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

    # helix
    n = 1  # number of helix coils
    scale = 1.0e1
    R0 = 1 * scale  # helix radius
    h = 5 * scale  # helix height
    c = h / (R0 * 2 * np.pi * n)
    L = np.sqrt(1 + c**2) * R0 * 2 * np.pi * n
    print(f"R0: {R0}")
    print(f"h: {h}")
    print(f"c: {c}")
    print(f"n: {n}")
    print(f"L: {L}")

    # reference solution
    def r(xi, phi0=0.0):
        alpha = 2 * np.pi * n * xi
        return R0 * np.array([np.sin(alpha + phi0), -np.cos(alpha + phi0), c * alpha])

    def A_IK(xi, phi0=0.0):
        alpha = 2 * np.pi * n * xi
        sa = np.sin(alpha + phi0)
        ca = np.cos(alpha + phi0)

        e_x = np.array([ca, sa, c]) / np.sqrt(1 + c**2)
        e_y = np.array([-sa, ca, 0])
        e_z = np.array([-c * ca, -c * sa, 1]) / np.sqrt(1 + c**2)

        return np.vstack((e_x, e_y, e_z)).T

    nxi = 100
    xis = np.linspace(0, 1, num=nxi)

    r_OPs = np.array([r(xi, phi0=np.pi) for xi in xis])
    A_IKs = np.array([A_IK(xi, phi0=np.pi) for xi in xis])

    Q0 = fit_configuration(rod, r_OPs, A_IKs)
    rod.q0 = Q0.copy()

    # joint between origin and left rod
    system = System()
    joint1 = RigidConnection(system.origin, rod, frame_ID2=(0,))

    # moment at right beam's tip
    Fi = material_model.Fi
    m = Fi[2] * 2 * np.pi / (2 * L) * 0.25 * 0.1
    M = lambda t: t * e3 * m
    moment = K_Moment(M, rod, (1,))

    # assemble the system
    system.add(rod, joint1)
    system.add(moment)
    system.assemble()

    # solve static system
    n_load_steps = 3
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        atol=atol,
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
    animate_beam(t, q, [rod], 2 * scale, show=True)

    ###########
    # export
    ###########
    from pathlib import Path

    path = Path(__file__)
    e = Export(path.parent, path.stem, True, nt, sol)
    e.export_contr(rod, level="volume", export_fields=["strains"], continuity="C0")
