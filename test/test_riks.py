import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.solver import Riks, Newton, SolverOptions
from cardillo.math.approx_fprime import approx_fprime


class Truss2D:
    def __init__(self, force, stiffness, phi0, width):
        self.force = force
        self.stiffness = stiffness
        self.phi0 = phi0
        self.width = width
        self.l0 = self.width / np.cos(phi0)

        self.nu = self.nq = 1
        self.u0 = np.zeros(self.nu)
        self.q0 = np.array([phi0])

    def M(self, t, q):
        return np.eye(self.nu)

    def h(self, t, q, u):
        phi = q[0]
        sphi, cphi = np.sin(phi), np.cos(phi)
        f = self.force(t)
        return np.array(
            [
                f
                + 2
                * self.stiffness
                * (self.width / cphi - self.width / np.cos(self.phi0))
                * sphi
            ]
        )

    def h_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.h(t, q, u), method="cs")


if __name__ == "__main__":
    force = lambda t: t
    stiffness = 1.0
    phi0 = np.pi / 4
    width = 1.0
    truss = Truss2D(force, stiffness, phi0, width)

    system = System()
    system.add(truss)
    system.assemble()

    sol = Riks(
        system,
        la_arc_span=[-1, 1],
        la_arc0=1e-6,
        iter_goal=2,
        options=SolverOptions(newton_atol=1e-8, newton_rtol=1e-8),
    ).solve()
    # sol = Newton(system, n_load_steps=50).solve()
    t, q = sol.t, sol.q

    # compute force for given angle
    phi = q[:, 0]
    F = -2 * stiffness * (width / np.cos(phi) - width / np.cos(phi0)) * np.sin(phi)

    fig, ax = plt.subplots()
    ax.plot(phi, F, "-k", label="analytical")
    ax.plot(q, t, "--r", label="Riks")

    # plot directions
    u = np.diff(phi)
    v = np.diff(F)
    norm = np.sqrt(u**2 + v**2)
    n = int(len(u) / 31)
    ax.quiver(
        phi[:-1][::n], F[:-1][::n], (u / norm)[::n], (v / norm)[::n], color="black"
    )

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$F(\phi)$")
    ax.legend()
    ax.grid()
    plt.show()
