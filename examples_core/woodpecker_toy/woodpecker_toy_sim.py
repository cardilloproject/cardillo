# This file implements the woodpecker toy example,
# see Glocker 1995 or Glocker 2001,

import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.solver import Moreau, SolverOptions

from WoodpeckerToy import WoodpeckerToy

if __name__ == "__main__":
    ############
    # parameters
    ############

    # initial condition
    t0 = 0
    q0 = np.array([0, -0.1036, -0.2788])
    u0 = np.array([-0.3411, 0, -7.4583])

    # simulation parameter
    t1 = 0.15  # final time
    dt = 1e-4  # time step

    #################
    # assemble system
    #################
    system = System(t0=t0)
    system.add(WoodpeckerToy(q0, u0))
    system.assemble(
        options=SolverOptions(compute_consistent_initial_conditions=False)
    )  # We use Moreau's scheme, which does not need initial accelerations or initial contact forces. Hence, we can omit the computation of consistent initial conditions. This enables us to implement less functions in the WoodpeckerToy class, i.e., g_N_ddot and gamma_F_dot are not needed.

    ############
    # simulation
    ############
    sol = Moreau(system, t1, dt).solve()

    t = sol.t
    q = sol.q
    u = sol.u
    P_N = sol.P_N
    P_F = sol.P_F

    #################
    # post-processing
    #################
    # plot trajectories
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 7))

    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel("$y$")
    ax[0].plot(t, q[:, 0])
    ax[0].grid()
    ax[0].set_title("time history of the coordinates")

    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel(r"$\varphi_M$")
    ax[1].plot(t, q[:, 1])
    ax[1].grid()

    ax[2].set_xlabel("$t$")
    ax[2].set_ylabel(r"$\varphi_S$")
    ax[2].plot(t, q[:, 2])
    ax[2].grid()

    plt.tight_layout()
    plt.show()

    # plot phase portraits
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    ax[0].set_xlabel(r"$\varphi_M$")
    ax[0].set_ylabel(r"$\dot{\varphi}_M$")
    ax[0].plot(q[:, 1], u[:, 1])
    ax[0].set_title(r"phase portrait of $\varphi_M$")

    ax[1].set_xlabel(r"$\varphi_S$")
    ax[1].set_ylabel(r"$\dot{\varphi}_S$")
    ax[1].plot(q[:, 2], u[:, 2])
    ax[1].set_title(r"phase portrait of $\varphi_S$")

    plt.tight_layout()
    plt.show()
