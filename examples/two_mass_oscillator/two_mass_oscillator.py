import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import Box, PointMass, Frame
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.interactions import TwoPointInteraction
from cardillo.solver import ScipyIVP

if __name__ == "__main__":
    ###################
    # system parameters
    ###################
    m = 1  # mass
    l0 = 1  # undeformed length of spring
    k = 100  # spring stiffness
    d = 2  # damping constant

    # dimensions of boxes
    width = l0
    height = depth = 0.5 * width
    box_dim = np.array([width, height, depth])

    #######################
    # simulation parameters
    #######################
    t0 = 0  # initial time
    t1 = 3  # final time

    # initial condition
    stretch = 1.9  # initial stretch of spring

    #################
    # assemble system
    #################

    # initialize system
    system = System(t0=t0)

    # mass 1
    q10 = np.array([-0.5 * (stretch * l0 + width), 0, 0])
    u10 = np.zeros(3)
    mass1 = Box(PointMass)(dimensions=box_dim, mass=m, q0=q10, u0=u10, name="mass 1")

    # mass 2
    q20 = np.array([0.5 * (stretch * l0 + width), 0, 0])
    u20 = np.zeros(3)
    mass2 = Box(PointMass)(dimensions=box_dim, mass=m, q0=q20, u0=u20, name="mass 2")

    # spring-damper interaction
    spring_damper = SpringDamper(
        TwoPointInteraction(
            mass1,
            mass2,
            B_r_CP1=np.array([0.5 * width, 0, 0]),
            B_r_CP2=np.array([-0.5 * width, 0, 0]),
        ),
        k,
        d,
        l_ref=l0,
        name="spring_damper",
    )

    # floor (only for visualization purposes)
    rectangle = Box(Frame)(
        dimensions=[5, 0.001, 2 * depth],
        r_OP=np.array([0, -0.5 * height, 0]),
        name="floor",
    )

    # add contributions and assemble system
    system.add(mass1, mass2, spring_damper, rectangle)
    system.assemble()

    ############
    # simulation
    ############

    dt = 1.0e-2  # time step
    solver = ScipyIVP(system, t1, dt)  # create solver
    sol = solver.solve()  # simulate system
    t = sol.t
    q = sol.q
    u = sol.u

    ##############
    # plot results
    ##############
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    # plot time evolution for x-coordinates
    x1 = [mass1.r_OP(ti, qi)[0] for ti, qi in zip(t, q[:, mass1.qDOF])]
    x2 = [mass2.r_OP(ti, qi)[0] for ti, qi in zip(t, q[:, mass2.qDOF])]
    ax[0, 0].plot(t, x1, "-r", label="$x_1$")
    ax[0, 0].plot(t, x2, "-g", label="$x_2$")
    ax[0, 0].set_title("Evolution of positions")
    ax[0, 0].set_xlabel("t")
    ax[0, 0].set_ylabel("x")
    ax[0, 0].legend()
    ax[0, 0].grid()

    # plot time evolution of elongation of SD-element
    l = [spring_damper.l(ti, qi) for ti, qi in zip(t, q[:, spring_damper.qDOF])]
    ax[0, 1].plot(t, l)
    ax[0, 1].set_title("Evolution of elongation of SD-element")
    ax[0, 1].set_xlabel("t")
    ax[0, 1].set_ylabel("length")
    ax[0, 1].grid()

    # plot time evolution of force of SD-element
    f = [
        spring_damper.force(ti, qi, ui)
        for ti, qi, ui in zip(t, q[:, spring_damper.qDOF], u[:, spring_damper.uDOF])
    ]
    ax[1, 0].plot(t, f)
    ax[1, 0].set_title("Evolution of scalar force of SD-element")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].set_ylabel("force")
    ax[1, 0].grid()

    # plot time evolution of energy
    # potential energy
    E_pot = np.array([system.E_pot(ti, qi) for ti, qi in zip(t, q)])
    # kinetic energy
    E_kin = np.array([system.E_kin(ti, qi, ui) for ti, qi, ui in zip(t, q, u)])
    ax[1, 1].plot(t, E_pot, label="$E_{pot}$")
    ax[1, 1].plot(t, E_kin, label="$E_{kin}$")
    ax[1, 1].plot(t, E_kin + E_pot, label="$E_{tot}$")
    ax[1, 1].set_title("Evolution of energies")
    ax[1, 1].set_xlabel("t")
    ax[1, 1].set_ylabel("energy")
    ax[1, 1].legend()
    ax[1, 1].grid()

    plt.tight_layout()
    plt.show()

    ############
    # VTK export
    ############
    path = Path(__file__)
    system.export(path.parent, "vtk", sol)
