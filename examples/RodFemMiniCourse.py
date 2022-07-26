import numpy as np

from cardillo.beams.spatial.material_models import Simo1986
from cardillo.beams.spatial.cross_section import CircularCrossSection
from cardillo.beams.spatial import DirectorAxisAngle
from cardillo.beams import animate_beam
from cardillo.model import frame

from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import RigidConnection

from cardillo.forces import Force, Moment
from cardillo.model import Model
from cardillo.solver import Newton, ScipyIVP

from cardillo.math import e1, e2, e3


def statics():
    nelements = 10
    polynomial_degree = 1

    # beam parameters
    L = 10
    EA = GA = 1.0e4
    GJ = EI = 1.0e2

    # build quadratic material model
    Ei = np.array([EA, GA, GA], dtype=float)
    Fi = np.array([GJ, EI, EI], dtype=float)
    material_model = Simo1986(Ei, Fi)

    # Note: This is never used in statics!
    line_density = 1.0
    radius = 1.0
    cross_section = CircularCrossSection(line_density, radius)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    Q = DirectorAxisAngle.straight_configuration(
        polynomial_degree,
        nelements,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
    )

    beam = DirectorAxisAngle(
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree,
        nelements,
        Q,
    )

    # junctions
    frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, beam, frame_ID2=(0,))

    # moment at right end
    Fi = material_model.Fi
    M = lambda t: t * 2 * np.pi * (Fi[0] * e1 + Fi[2] * e3) / L * 1.0
    moment = Moment(M, beam, (1,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    model.add(moment)
    model.assemble()

    n_load_steps = 20

    solver = Newton(
        model,
        n_load_steps=n_load_steps,
        max_iter=30,
        atol=1.0e-8,
    )
    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    animate_beam(t, q, [beam], L, show=True)


def dynamics():
    nelements = 10
    polynomial_degree = 1

    # beam parameters
    L = 1
    EA = GA = 1.0e2
    GJ = EI = 1.0e2

    # build quadratic material model
    Ei = np.array([EA, GA, GA], dtype=float)
    Fi = np.array([GJ, EI, EI], dtype=float)
    material_model = Simo1986(Ei, Fi)

    # Note: This is never used in statics!
    line_density = 1.0
    radius = 1.0
    cross_section = CircularCrossSection(line_density, radius)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment

    # starting point and orientation of initial point, initial length
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    Q = DirectorAxisAngle.straight_configuration(
        polynomial_degree,
        nelements,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
    )

    beam = DirectorAxisAngle(
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree,
        nelements,
        Q,
    )

    # junctions
    frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)

    # left and right joint
    joint1 = RigidConnection(frame1, beam, frame_ID2=(0,))

    # # moment at right end
    # Fi = material_model.Fi
    # M = lambda t: t**2 * 2 * np.pi * (Fi[0] * e1 + Fi[2] * e3) / L * 0.1
    # moment = Moment(M, beam, (1,))

    F = -1.0e1 * e3
    force = Force(F, beam, (1,))

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame1)
    model.add(joint1)
    # model.add(moment)
    model.add(force)
    model.assemble()

    t1 = 1
    dt = 1.0e-2
    method = "RK45"
    rtol = 1.0e-5
    atol = 1.0e-5

    solver = ScipyIVP(model, t1, dt, method=method, rtol=rtol, atol=atol)
    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    animate_beam(t, q, [beam], L, show=True)


if __name__ == "__main__":
    statics()
    # dynamics()
