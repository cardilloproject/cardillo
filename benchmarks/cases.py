"""Short, deterministic benchmark cases. Each `case_*` returns the Solution."""

import numpy as np

from cardillo import System
from cardillo.constraints import RigidConnection, Spherical
from cardillo.contacts import Sphere2Plane, Sphere2Sphere
from cardillo.discrete import Box, Frame, PointMass, RigidBody, Sphere
from cardillo.force_laws import KelvinVoigtElement
from cardillo.forces import B_Moment, Force
from cardillo.interactions import TwoPointInteraction
from cardillo.math import A_IB_basic, e2, e3
from cardillo.rods import (
    CircularCrossSection,
    CrossSectionInertias,
    Harsch2021,
    RectangularCrossSection,
    Simo1986,
)
from cardillo.rods.force_line_distributed import Force_line_distributed
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.solver import (
    BackwardEuler,
    DualStormerVerlet,
    Moreau,
    Newton,
    Rattle,
    SolverOptions,
)


def case_balls(t1=0.5, nx=3, ny=3, nz=5):
    """Many rigid spheres with Sphere2Plane and Sphere2Sphere contacts, Moreau."""
    radius, mu, g = 0.05, 0.3, np.array([0, 0, -10])
    offset = 0.3 * radius
    system = System()
    r_OC0 = np.array([0, 0, 8 * radius])
    ox = np.array([offset + 2 * radius, 0, 0])
    oy = np.array([0, offset + 2 * radius, 0])
    oz = np.array([0, 0, offset + 2 * radius])
    balls = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                q0 = RigidBody.pose2q(r_OC0 + i * ox + j * oy + k * oz, np.eye(3))
                balls.append(
                    Sphere(RigidBody)(
                        radius=radius,
                        density=1,
                        subdivisions=3,
                        q0=q0,
                        u0=np.zeros(6),
                        name=f"ball_{i}{j}{k}",
                    )
                )
    system.add(*balls)
    floor = Box(Frame)(
        dimensions=[4.5, 4.5, 0.0001],
        name="floor",
        A_IB=A_IB_basic(np.deg2rad(10)).x @ A_IB_basic(np.deg2rad(10)).y,
    )
    system.add(floor)
    for ball in balls:
        system.add(Force(ball.mass * g, ball, name="gravity_" + ball.name))
        system.add(
            Sphere2Plane(
                floor, ball, mu=mu, radius=radius, e_N=0, e_F=0, name="f2" + ball.name
            )
        )
    idx = 0
    while balls:
        for ball in balls[1:]:
            idx += 1
            system.add(
                Sphere2Sphere(
                    balls[0], ball, radius, radius, mu, e_N=0, e_F=0, name=f"s2s{idx}"
                )
            )
        balls.pop(0)
    system.assemble()
    solver = Moreau(
        system,
        t1,
        1e-3,
        options=SolverOptions(fixed_point_max_iter=int(1e4), prox_scaling=0.5),
    )
    return solver.solve()


def case_chain(t1=2, n_particles=20):
    """Compliance-form spring chain of point masses, DualStormerVerlet."""
    m, l0, k, d = 1, 1, 1e8, 0
    system = System()
    particles = []
    for i in range(n_particles):
        p = Sphere(PointMass)(
            radius=l0 / 20,
            mass=m,
            q0=(i + 1) * np.array([l0, 0, 0]),
            u0=np.zeros(3),
            name=f"mass{i}",
        )
        system.add(p)
        particles.append(p)
        system.add(Force(m * np.array([0, 0, -10]), p, name=f"gravity{i}"))
    pairs = [(system.origin, particles[0])] + list(zip(particles[:-1], particles[1:]))
    for i, (a, b) in enumerate(pairs):
        system.add(
            KelvinVoigtElement(
                TwoPointInteraction(a, b),
                k,
                d,
                l_ref=l0,
                compliance_form=True,
                name=f"sd{i}",
            )
        )
    system.assemble()
    solver = DualStormerVerlet(
        system, t1, 1e-2, options=SolverOptions(fixed_point_max_iter=50)
    )
    return solver.solve()


def _cantilever(Rod, nelements, n_load_steps, constitutive_law=Simo1986):
    length = 2 * np.pi
    width = length / 1e2
    cross_section = RectangularCrossSection(width, width)
    material_model = constitutive_law(np.array([5, 1, 1]), np.array([0.5, 2, 2]))
    system = System()
    q0 = Rod.straight_configuration(nelements, length)
    cantilever = Rod(cross_section, material_model, nelements, Q=q0, q0=q0)
    system.add(cantilever)
    system.add(RigidConnection(system.origin, cantilever, xi2=(0,)))
    P = lambda t: material_model.Fi[2] * (10 * t) / length**2
    system.add(Force(lambda t: -P(t) * e2, cantilever, (1,)))
    system.add(B_Moment(lambda t: 2.5 * P(t) * e3, cantilever, (1,)))
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_max_iter=30, newton_atol=1e-8),
    )
    return solver.solve()


def case_rod_mixed(nelements=20, n_load_steps=10):
    return _cantilever(make_CosseratRod(mixed=True), nelements, n_load_steps)


def case_rod_constrained(nelements=20, n_load_steps=10):
    return _cantilever(
        make_CosseratRod(mixed=True, constraints=[0, 1, 2]), nelements, n_load_steps
    )


def case_rod_se3(nelements=20, n_load_steps=10):
    return _cantilever(
        make_CosseratRod(interpolation="SE3", mixed=True), nelements, n_load_steps
    )


def case_rod_displacement(nelements=20, n_load_steps=10):
    return _cantilever(
        make_CosseratRod(mixed=False),
        nelements,
        n_load_steps,
        constitutive_law=Harsch2021,
    )


def case_rod_r12(nelements=20, n_load_steps=10):
    return _cantilever(
        make_CosseratRod(interpolation="R12", mixed=True), nelements, n_load_steps
    )


def case_rod2plane(nelements=10, t1=3.0, dt=0.1):
    """Dynamic mixed rod falling onto a plane (two Sphere2Plane contacts)."""
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        mixed=True,
        polynomial_degree=1,
        reduced_integration=True,
    )
    length = 2 * np.pi
    cross_section = CircularCrossSection(length / 1e2)
    inertias = CrossSectionInertias(1, cross_section)
    material_model = Simo1986(np.array([5, 1, 1]), np.array([0.5, 2, 2]))
    q0 = Rod.straight_configuration(
        nelements, length, r_OP0=np.array([0, 0, 0.1], dtype=float)
    )
    system = System()
    rod = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        cross_section_inertias=inertias,
    )
    system.add(rod)
    system.add(Force_line_distributed(np.array([0, 0, -9.81 * inertias.A_rho0]), rod))
    system.add(Sphere2Plane(system.origin, rod, mu=0.3, radius=0, xi2=0, name="cl"))
    system.add(Sphere2Plane(system.origin, rod, mu=0.3, radius=0, xi2=1, name="cr"))
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))
    solver = DualStormerVerlet(
        system,
        t1,
        dt,
        options=SolverOptions(
            prox_scaling=1, newton_atol=1e-8, newton_rtol=1e-8, newton_max_iter=500
        ),
    )
    return solver.solve()


def _pendulum_chain(Solver, n_bodies=10, t1=1.0, dt=1e-2):
    """Chain of rigid bodies connected by spherical joints, hanging in gravity."""
    m, l, r = 1.0, 0.5, 0.05
    Theta = np.diag([m * r**2 / 2] * 3)
    system = System()
    prev = system.origin
    for i in range(n_bodies):
        r_OC = np.array([(i + 0.5) * l, 0, 0])
        body = RigidBody(m, Theta, RigidBody.pose2q(r_OC, np.eye(3)), np.zeros(6))
        system.add(body)
        system.add(Spherical(prev, body, r_OJ0=np.array([i * l, 0, 0])))
        system.add(Force(np.array([0, 0, -9.81 * m]), body))
        prev = body
    system.assemble()
    return Solver(system, t1, dt).solve()


def case_pendulum_rattle(**kw):
    kw = {"t1": 0.25, "dt": 5e-4, **kw}
    return _pendulum_chain(Rattle, **kw)


def case_pendulum_backward_euler(**kw):
    kw = {"t1": 2.0, "dt": 1e-2, **kw}
    return _pendulum_chain(BackwardEuler, **kw)


CASES = {
    "balls": case_balls,
    "chain": case_chain,
    "rod_mixed": case_rod_mixed,
    "rod_constrained": case_rod_constrained,
    "rod_se3": case_rod_se3,
    "rod_displacement": case_rod_displacement,
    "rod_r12": case_rod_r12,
    "rod2plane": case_rod2plane,
    "pendulum_rattle": case_pendulum_rattle,
    "pendulum_backward_euler": case_pendulum_backward_euler,
}
