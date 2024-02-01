import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from cardillo import System
from cardillo.solver import ScipyIVP, BackwardEuler
from cardillo.discrete import Box, PointMass, Frame
from cardillo.force_laws import KelvinVoigtElement
from cardillo.interactions import TwoPointInteraction
from cardillo.visualization import Export

if __name__ == "__main__":
    m = 1
    l0 = 1
    stretch = 1.02
    
    width = l0 / 2
    height = depth = width / 2
    box_dim = np.array([width, height, depth])
    
    k = 1.0e2
    d = 2

    system = System()

    # mass 1
    q10 = np.array([0, 0, 0])
    u10 = np.zeros(3)
    mass1 = Box(PointMass)(dimensions=box_dim, mass=m, q0=q10, u0=u10, name="mass 1")

    # mass 2
    q20 = np.array([stretch * l0 + width, 0, 0])
    u20 = np.zeros(3)
    mass2 = Box(PointMass)(dimensions=box_dim, mass=m, q0=q20, u0=u20, name="mass 2")

    # spring-damper interaction
    tp_interaction = TwoPointInteraction(mass1, mass2, K_r_SP1=np.array([width / 2, 0, 0]), K_r_SP2=np.array([-width / 2, 0, 0]), name="spring_damper")
    spring_damper = KelvinVoigtElement(tp_interaction, k, d, l_ref=l0)

    # floor
    rectangle = Box(Frame)(dimensions=[5, 0.001, 5], name="floor", r_OP=np.array([0, -height / 2, 0]))

    # add contributions and assemble system
    # system.add(mass1, mass2, tp_interaction, spring_damper, rectangle)
    system.add(mass1, mass2, spring_damper, rectangle)
    system.assemble()

    t0 = 0
    t1 = 2
    dt = 1.0e-2
    solver = ScipyIVP(system, t1, dt, method="BDF")
    sol = solver.solve()
    t = sol.t
    q = sol.q


    plt.plot(t, q[:, 0], "-r")
    plt.plot(t, q[:, 3], "--g")
    plt.grid()

    plt.show()

        # VTK export
    path = Path(__file__)
    e = Export(path.parent, path.stem, True, 50, sol)
    for contr in system.contributions:
        if hasattr(contr, "export"):
            e.export_contr(contr, file_name=contr.name)