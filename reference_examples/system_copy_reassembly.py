import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.solver import ScipyIVP, Newton
from cardillo.discrete import Frame, PointMass
from cardillo.forces import Force
from cardillo.forces import (
    LinearDamper,
    LinearSpring,
    ScalarForceTranslational,
)
from cardillo.constraints import SphericalJoint
from cardillo.math import norm

case = "force"
# case = "constraint"

if __name__ == "__main__":
    m = 1
    g = 9.81
    L = 1
    k = 1.0e2
    d = 4

    frame = Frame()

    q0 = np.array([0.01, 0.01, -L * 0.9])
    mass = PointMass(m, q0=q0, u0=np.zeros(3))

    f_g_value = np.array([0, 0, -m * g])
    f_g_statics = Force(lambda t: t * f_g_value, mass)

    linear_spring = LinearSpring(k, g_ref=L)
    linear_damper = LinearDamper(d)
    scalar_force_element = ScalarForceTranslational(
        frame, mass, linear_spring, linear_damper
    )

    system = System()
    system.add(frame)
    system.add(mass)
    # if case == "force":
    system.add(f_g_statics)
    system.add(scalar_force_element)
    system.assemble()

    # solve for static equilibrium
    sol_statics = Newton(system, n_load_steps=10).solve()
    t = sol_statics.t
    q = sol_statics.q

    fig, ax = plt.subplots()
    ax.plot(t, q[:, 0], "-r", label="q1")
    ax.plot(t, q[:, 1], "--g", label="q1")
    ax.plot(t, q[:, 2], "-.b", label="q2")
    ax.plot(t, [norm(qi) for qi in q], "-ok", label="||q||")
    ax.grid()
    ax.legend()

    from copy import deepcopy

    system_statics = deepcopy(system)

    q0 = sol_statics.q[-1]
    # u0 = sol1.u[-1]

    for contr in system.contributions:
        if hasattr(contr, "nq"):
            contr.q0 = q0[contr.qDOF]
        # if hasattr(contr, "nu"):
        #     contr.u0 = u0[contr.uDOF]

    # replace static gravity contribution with dynamic one
    system.remove(f_g_statics)
    f_g_dynamics = Force(f_g_value, mass)
    system.add(f_g_dynamics)

    # reassemble the model
    system.assemble()

    print(f"system.q0: {system.q0}")
    print(f"system.contributions: {system.contributions}")

    t0 = 0
    t1 = 2
    dt = 1.0e-2
    sol_dynamics = ScipyIVP(system, t1, dt).solve()
    t = sol_dynamics.t
    q = sol_dynamics.q

    fig, ax = plt.subplots()
    ax.plot(t, q[:, 0], "-r", label="q1")
    ax.plot(t, q[:, 1], "--g", label="q1")
    ax.plot(t, q[:, 2], "-.b", label="q2")
    ax.plot(t, [norm(qi) for qi in q], "-ok", label="||q||")
    ax.grid()
    ax.legend()

    plt.show()
