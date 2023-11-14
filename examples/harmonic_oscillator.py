import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.solver import ScipyIVP, BackwardEuler
from cardillo.discrete import PointMass
from cardillo.forces import Force
from cardillo.forces import (
    LinearDamper,
    LinearSpring,
    ScalarForceTranslational,
)

if __name__ == "__main__":
    m = 1
    g = 9.81
    l0 = 1
    k = 2.0e2
    d = 4

    system = System()

    q0 = np.array([0, 0, -l0])
    mass = PointMass(m, q0=q0, u0=np.zeros(3))

    f_g = Force(lambda t: np.array([0, 0, -m * g]), mass)

    linear_spring = LinearSpring(k)
    # linear_spring = None
    linear_damper = LinearDamper(d)
    # linear_damper = None
    scalar_force_element = ScalarForceTranslational(
        system.origin, mass, linear_spring, linear_damper
    )
    system.add(mass)
    system.add(f_g)
    system.add(scalar_force_element)
    system.assemble()

    # get contributions from system
    contrs = system.get_contributions("force.Force")
    print(contrs)

    t0 = 0
    t1 = 2
    dt = 1.0e-2
    # solver = ScipyIVP(system, t1, dt)
    solver = BackwardEuler(system, t1, dt)
    sol = solver.solve()
    t = sol.t
    q = sol.q

    plt.plot(t, q[:, 0], "-r")
    plt.plot(t, q[:, 1], "--g")
    plt.plot(t, q[:, 2], "-.b")
    plt.show()
