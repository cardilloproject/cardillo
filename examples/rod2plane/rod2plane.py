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
from cardillo.solver import SolverOptions, Moreau, DualStormerVerlet


nelements = 5
polynomial_degree = 1
length = 2 * np.pi
slenderness = 1e2
# slenderness = 2e2
# slenderness = 1e3
# slenderness = 1e4
reduced_integration = True
mixed = True

Rod = make_CosseratRod(
    interpolation="Quaternion",
    mixed=mixed,
    polynomial_degree=polynomial_degree,
    reduced_integration=reduced_integration,
)


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
    r_OP0 = np.array([0, 0, 0.1], dtype=float)
    q0 = Rod.straight_configuration(
        nelements,
        length,
        r_OP0=r_OP0,
    )
    # construct cantilever
    rod = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        cross_section_inertias=cross_section_inertias,
    )

    # gravity
    g = 9.81
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
    t1 = 10
    # dt = 5e-5
    # solver = Moreau(system, t1, dt, options=SolverOptions(prox_scaling=0.05))
    dt = 1e-1
    prox_scaling = 1
    solver = DualStormerVerlet(
        system,
        t1,
        dt,
        options=SolverOptions(
            prox_scaling=prox_scaling,
            newton_atol=1e-8,
            newton_rtol=1e-8,
            newton_max_iter=500,
        ),
    )
    sol = solver.solve()
    t = sol.t
    q = sol.q

    # animation
    animate_beam(t, q, [rod], scale=length)
