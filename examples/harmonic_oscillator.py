import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.solver import ScipyIVP, BackwardEuler
from cardillo.discrete import PointMass
from cardillo.forces import Force
from cardillo.force_laws import KelvinVoigtElement
from cardillo.interactions import TwoPointInteraction

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

    tp_interaction = TwoPointInteraction(system.origin, mass)
    scalar_force_element = KelvinVoigtElement(tp_interaction, k, d)
    system.add(mass)
    system.add(f_g)
    system.add(tp_interaction)
    system.add(scalar_force_element)
    system.assemble()

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
