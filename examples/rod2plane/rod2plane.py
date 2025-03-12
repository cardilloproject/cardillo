import numpy as np
from cardillo import System
from cardillo.rods import (
    CircularCrossSection,
    CrossSectionInertias,
    Simo1986,
    animate_beam,
)
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.rods.force_line_distributed import Force_line_distributed
from cardillo.contacts import Sphere2Plane
from cardillo.solver import SolverOptions, Moreau, MoreauThetaCompliance


nelements = 3
polynomial_degree = 2
length = 2 * np.pi
slenderness = 1.0e2
reduced_integration = True
g = 9.81

r_OP0 = np.array([0, 0, 0.1], dtype=float)

# Rod = make_CosseratRod(interpolation="Quaternion", mixed=False)
Rod = make_CosseratRod(interpolation="Quaternion", mixed=True)


if __name__ == "__main__":
    system = System()

    # cross section properties for visualization purposes
    radius = length / slenderness
    cross_section = CircularCrossSection(radius)

    # inertia properties
    density = 1
    cross_section_inertias = CrossSectionInertias(density, cross_section)
    A_rho0 = cross_section_inertias.A_rho0

    # material properties
    Ei = np.array([5, 1, 1])
    Fi = np.array([0.5, 2, 2])
    material_model = Simo1986(Ei, Fi)

    # compute straight initial configuration of cantilever
    q0 = Rod.straight_configuration(
        nelements,
        length,
        polynomial_degree=polynomial_degree,
        r_OP0=r_OP0,
    )
    # construct cantilever
    rod = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        reduced_integration=reduced_integration,
        cross_section_inertias=cross_section_inertias,
    )

    # gravity
    gravity = Force_line_distributed(np.array([0, 0, -g * A_rho0]), rod)

    # contacts
    contact_left = Sphere2Plane(
        system.origin, rod, mu=0.3, r=0, xi=0, name="contact left"
    )
    contact_right = Sphere2Plane(
        system.origin, rod, mu=0.3, r=0, xi=1, name="contact right"
    )

    # assemble the system
    system.add(rod)
    system.add(gravity)
    system.add(contact_left)
    system.add(contact_right)
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # solver
    t1 = 2
    dt = 1e-3
    dt = 1e-2
    # solver = Moreau(system, t1, dt)
    solver = MoreauThetaCompliance(
        system, t1, dt, theta=0.5, options=SolverOptions(prox_scaling=0.5)
    )
    sol = solver.solve()
    t = sol.t
    q = sol.q

    # animation
    animate_beam(t, q, [rod], scale=length)
