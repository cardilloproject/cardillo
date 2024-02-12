# This file implements the woodpecker toy example,
# see Glocker 1995 or Glocker 2001,

import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.solver import Moreau

from WoodpeckerToy import WoodpeckerToy


if __name__ == "__main__":
    q0 = np.array([0, -0.1036, -0.2788])
    u0 = np.array([-0.3411, 0, -7.4583])
    # q0 = np.array([-0.00910531, -0.06546076, -0.07842767])
    # u0 = np.array([-0.19510371, -7.53870682, -5.55658158])
    woodpecker_toy = WoodpeckerToy(q0, u0)

    t0 = 0
    t1 = 0.15
    dt = 1e-4
    system = System(t0=t0)
    system.add(woodpecker_toy)
    system.assemble()

    sol = Moreau(system, t1, dt).solve()
  
    t = sol.t
    q = sol.q
    u = sol.u
    P_N = sol.P_N
    P_F = sol.P_F

    fig, ax = plt.subplots(2, 1)

    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel("$y$")
    ax[0].plot(t, q[:, 0], "--r", label="moreau")
    ax[0].legend()

    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel("$u_y$")
    ax[1].plot(t, u[:, 0], "--r", label="moreau")
    ax[1].legend()

    ######
    fig, ax = plt.subplots(2, 1)

    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel(r"$\varphi_M$")
    ax[0].plot(t, q[:, 1], "--r", label="moreau")
    ax[0].legend()

    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel(r"$u_{\varphi_M}$")
    ax[1].plot(t, u[:, 1], "--r", label="moreau")
    ax[1].legend()


    ######
    fig, ax = plt.subplots(2, 1)

    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel(r"$\varphi_S$")
    ax[0].plot(t, q[:, 2], "--r", label="moreau")
    ax[0].legend()

    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel(r"$u_{\varphi_S}$")
    ax[1].plot(t, u[:, 2], "--r", label="moreau")
    ax[1].legend()

    ######
    fig, ax = plt.subplots(1, 2)

    ax[0].set_xlabel(r"$\varphi_M$")
    ax[0].set_ylabel(r"$u_{\varphi_M}$")
    ax[0].plot(q[:, 1], u[:, 1], "--r", label="moreau")
    ax[0].legend()

    ax[1].set_xlabel(r"$\varphi_S$")
    ax[1].set_ylabel(r"$u_{\varphi_S}$")
    ax[1].plot(q[:, 2], u[:, 2], "--r", label="moreau")
    ax[1].legend()

    plt.show()
