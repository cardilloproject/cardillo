import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.solver import ScipyIVP
from cardillo.discrete import Frame, PointMass
from cardillo.forces import Force
from cardillo.scalar_force_interactions.force_laws import Linear_spring, Linear_damper
from cardillo.scalar_force_interactions import (
    Translational_f_pot,
    ScalarForceTranslational,
)

if __name__ == "__main__":
    m = 1
    g = 9.81
    l0 = 1
    k = 2.0e2
    d = 4

    frame = Frame()

    q0 = np.array([0, 0, -l0])
    mass = PointMass(m, q0=q0, u0=np.zeros(3))

    f_g = Force(lambda t: np.array([0, 0, -m * g]), mass)

    linear_spring = Linear_spring(k)
    spring_element = Translational_f_pot(linear_spring, frame, mass)

    linear_damper = Linear_damper(d)
    damping_element = ScalarForceTranslational(linear_damper, frame, mass)

    model = System()
    model.add(frame)
    model.add(mass)
    model.add(f_g)
    model.add(spring_element)
    model.add(damping_element)
    model.assemble()

    t0 = 0
    t1 = 2
    dt = 1.0e-2
    solver = ScipyIVP(model, t1, dt)
    sol = solver.solve()
    t = sol.t
    q = sol.q

    plt.plot(t, q[:, 0], "-r")
    plt.plot(t, q[:, 1], "--g")
    plt.plot(t, q[:, 2], "-.b")
    plt.show()
