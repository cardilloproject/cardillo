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

    system = System()

    q0 = np.array([0.01, 0.01, -L * 0.9])
    u0 = np.zeros(3, dtype=float)
    # u0 = np.array([0, 0, 1], dtype=float)
    mass = PointMass(m, q0=q0, u0=u0)

    f_g_value = np.array([0, 0, -m * g])
    f_g_statics = Force(lambda t: t * f_g_value, mass)

    r_OP = lambda t: np.array([0, 0, -L * (1 + 0.5 * t)])
    frame2 = Frame(r_OP=r_OP)
    joint = SphericalJoint(frame2, mass, r_OB0=r_OP(0))

    linear_spring = LinearSpring(k, g_ref=L)
    linear_damper = LinearDamper(d)
    scalar_force_element = ScalarForceTranslational(
        system.origin, mass, linear_spring, linear_damper
    )

    system.add(mass)
    if case == "force":
        system.add(f_g_statics)
    if case == "constraint":
        system.add(frame2)
        system.add(joint)
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

    system_statics = system.deepcopy(sol_statics)

    # replace static gravity contribution with dynamic one
    if case == "force":
        system.remove(f_g_statics)
        f_g_dynamics = Force(f_g_value, mass)
        system.add(f_g_dynamics)

    # remove joint contribution from the system
    if case == "constraint":
        system.remove(frame2)
        system.remove(joint)

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
