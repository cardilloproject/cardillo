import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from cardillo import System
from cardillo.discrete import PointMass
from cardillo.interactions import TwoPointInteraction
from cardillo.forces import Force
from cardillo.force_laws import MaxwellElement
from cardillo.solver import ScipyIVP

if __name__ == "__main__":
    mass = 1e-1
    stiffness = 1e1
    damping = 1
    l0 = 1
    x0 = 1
    x_D0 = 0.0
    x_dot0 = 0.0
    F_mag = mass * 10
    q0 = np.array([x0, x_D0], dtype=float)
    u0 = np.array([x_dot0], dtype=float)
    la_c0 = np.array([-5], dtype=float)

    system = System()
    pm = PointMass(mass, q0=np.array([x0, 0, 0]), u0=np.array([x_dot0, 0, 0]))
    pm.name = "point mass"
    system.add(pm)
    tpi = TwoPointInteraction(system.origin, pm)
    max = MaxwellElement(tpi, stiffness, damping, l_ref=l0, q0=np.array([x_D0]))
    max.name = "Maxwell-element"
    system.add(tpi, max)
    t0 = 0
    tM = 0.25
    t1 = 1
    F = lambda t: -F_mag * 0.5 * np.array([1 - np.sign(t - tM), 0, 0])
    f = Force(F, pm, name="force")

    system.add(f)
    system.assemble()

    t0 = 0
    tF = 1
    dt = 1e-3
    sol_ref = ScipyIVP(system, tF, dt).solve()
    t_ref, q_ref, u_ref = sol_ref.t, sol_ref.q, sol_ref.u

    sol1 = ScipyIVP(system, tM, dt).solve()
    t1, q1, u1 = sol1.t, sol1.q, sol1.u

    # system_old = system.deepcopy()
    # system.set_new_initial_state(q1[-1], u1[-1], t0=tM)
    # system.remove(system.contributions_map["force"])
    # system.assemble()
    # sol2 = ScipyIVP(system, tF, dt).solve()

    system_new = system.deepcopy()
    system_new.set_new_initial_state(q1[-1], u1[-1], t0=tM)
    system_new.remove(system_new.contributions_map["force"])
    system_new.assemble()
    sol2 = ScipyIVP(system_new, tF, dt).solve()

    t2, q2, u2 = sol2.t, sol2.q, sol2.u

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(t1, q1[:, 0], "-xr", label="x1")
    ax[0].plot(t2, q2[:, 0], "-xg", label="x2")
    ax[0].plot(t_ref, q_ref[:, 0], "--b", label="x_ref")
    ax[0].plot(t1, q1[:, -1], "-xr", label="x_D1")
    ax[0].plot(t2, q2[:, -1], "-xg", label="x_D2")
    ax[0].plot(t_ref, q_ref[:, -1], "--r", label="x_D_ref")
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(t1, u1[:, 0], "-xr", label="x_dot1")
    ax[1].plot(t2, u2[:, 0], "-xg", label="x_dot2")
    ax[1].plot(t_ref, u_ref[:, 0], "--b", label="x_dot_ref")
    ax[1].grid()

    ax[0].legend()
    ax[1].legend()
    plt.show()
