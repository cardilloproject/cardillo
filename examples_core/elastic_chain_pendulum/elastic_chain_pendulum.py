import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import Sphere, PointMass, Frame
from cardillo.forces import Force

from cardillo.force_laws import KelvinVoigtElement as SpringDamper

from cardillo.interactions import TwoPointInteraction
from cardillo.solver import ScipyIVP, BackwardEuler, Rattle, SolverOptions

if __name__ == "__main__":
    # system parameters
    m = 1  # mass
    l0 = 1  # undeformed length of spring
    k = 1e8  # spring stiffness
    d = 0  # damping constant

    n_particles = 20  # number of particles

    gravity = np.array([0, 0, -10])  # gravity

    # simulation parameters
    compliance_form = True  # use compliance formulation for force element, i.e., force is Lagrange multiplier la_c.
    t0 = 0  # initial time
    t1 = 3  # final time

    # initialize system
    system = System(t0=t0)

    # particles
    particles = []
    offset = np.array([l0, 0, 0])

    u0 = np.zeros(3)
    for i in range(n_particles):
        q0 = (i + 1) * offset
        particle = Sphere(PointMass)(
            radius=l0 / 20, mass=m, q0=q0, u0=u0, name="mass" + str(i)
        )
        system.add(particle)
        particles.append(particle)
        system.add(Force(m * gravity, particle, name="gravity" + str(i)))

    # first force element
    system.add(
        SpringDamper(
            TwoPointInteraction(system.origin, particles[0]),
            k,
            d,
            l_ref=l0,
            compliance_form=compliance_form,
            name="spring_damper_0",
        )
    )

    for i in range(n_particles - 1):
        system.add(
            SpringDamper(
                TwoPointInteraction(particles[i], particles[i + 1]),
                k,
                d,
                l_ref=l0,
                compliance_form=compliance_form,
                name="spring_damper_" + str(i + 1),
            )
        )

    system.assemble()

    # simulation
    dt = 1e-2  # time step
    # solver = ScipyIVP(system, t1, dt)  # create solver
    solver = BackwardEuler(
        system, t1, dt, options=SolverOptions(newton_max_iter=50)
    )  # create solver
    # solver = Rattle(system, t1, dt)  # create solver
    sol = solver.solve()  # simulate system
    t = sol.t
    q = sol.q
    u = sol.u

    # VTK export
    path = Path(__file__)
    system.export(path.parent, "vtk", sol)
