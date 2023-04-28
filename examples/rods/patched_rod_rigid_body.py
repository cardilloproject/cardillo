from cardillo.math import e3
from cardillo.beams import (
    RectangularCrossSection,
    Simo1986,
)
from cardillo.constraints import RigidConnection

from cardillo.beams import K_R12_PetrovGalerkin_AxisAngle as Rod

# from cardillo.beams import K_R12_PetrovGalerkin_Quaternion as Rod
# from cardillo.beams import K_R12_PetrovGalerkin_R9 as Rod
from cardillo.beams import animate_beam

from cardillo.discrete import RigidBodyQuaternion as RigidBody

# from cardillo.discrete import RigidBodyAxisAngle as RigidBody
from cardillo.forces import K_Moment, Force
from cardillo import System
from cardillo.solver import Newton

import numpy as np


if __name__ == "__main__":
    # number of elements
    nelements = 2

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
    q0_left = Rod.straight_configuration(
        polynomial_degree,
        polynomial_degree,
        basis,
        basis,
        nelements,
        L,
    )
    rod_left = Rod(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree,
        polynomial_degree,
        nelements,
        Q=q0_left,
        q0=q0_left,
        basis_r=basis,
        basis_psi=basis,
    )

    # joint between origin and left rod
    system = System()
    joint1 = RigidConnection(system.origin, rod_left, frame_ID2=(0,))

    # intermediate rigid body
    rb = RigidBody(1, np.eye(3), np.array([L, 0, 0, 1, 0, 0, 0]))
    # rb = RigidBody(1, np.eye(3), np.array([0, 0, 0, 0, 0, 0]))
    joint2 = RigidConnection(rod_left, rb, frame_ID1=(1,))

    # right rod
    q0_right = Rod.straight_configuration(
        polynomial_degree,
        polynomial_degree,
        basis,
        basis,
        nelements,
        L,
        r_OP=np.array([L, 0, 0], dtype=float),
    )
    rod_right = Rod(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree,
        polynomial_degree,
        nelements,
        Q=q0_right,
        q0=q0_right,
        basis_r=basis,
        basis_psi=basis,
    )

    # joint between left and right rod
    joint3 = RigidConnection(rb, rod_right, frame_ID2=(0,))

    # moment at right beam's tip
    Fi = material_model.Fi
    m = Fi[2] * 2 * np.pi / (2 * L) * 0.25
    M = lambda t: t * e3 * m
    moment = K_Moment(M, rod_right, (1,))

    # force at right beam's tip
    f = m / (2 * L)
    F = lambda t: t * f * e3
    force = Force(F, rod_right, frame_ID=(1,))

    # assemble the system
    system.add(rod_left, joint1)
    system.add(rod_right, joint3)
    system.add(rb, joint2)
    system.add(moment)
    system.add(force)
    system.assemble()

    # solve static system
    n_load_steps = 10
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        atol=atol,
        # numerical_jacobian=True,
        numerical_jacobian=False,
    )
    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    ###########
    # animation
    ###########
    animate_beam(t, q, [rod_left, rod_right], 2 * L, show=True)
