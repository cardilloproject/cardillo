import numpy as np
import pytest
import matplotlib.pyplot as plt
from scipy.integrate._ivp.tests.test_ivp import compute_error

from cardillo import System
from cardillo.solver import (
    ScipyIVP,
    ScipyDAE,
    BackwardEuler,
    Moreau,
    Rattle,
    DualStörmerVerlet,
    SolverOptions,
    Solution,
)
from cardillo.utility.convergence_analysis import convergence_analysis


solvers_and_kwargs = [
    # (ScipyDAE, {}),
    # (ScipyIVP, {}),
    # (BackwardEuler, {}),
    # (Moreau, {}),
    # (Rattle, {}),
    (DualStörmerVerlet, {"theta": 0.5}),
]


class KnifeEdge:
    def __init__(self):
        self.nq = 3
        self.nu = 3
        self.nla_gamma = 1
        self.Omega = 1

        self.q0 = np.zeros(3, dtype=float)
        self.u0 = np.array([0, 0, self.Omega], dtype=float)
        self.la_gamma0 = np.zeros(1, dtype=float)

        self.m = 1
        self.theta = 1
        self.grav = 9.81
        self.alpha = 35 / 180 * np.pi
        self.sa = np.sin(self.alpha)

    def __call__(self, t):
        """Evaluate true solution."""
        x = (
            self.grav
            * self.sa
            / self.Omega
            * (t / 2 - np.sin(2 * self.Omega * t) / (4 * self.Omega))
        )
        y = self.grav * self.sa / (2 * self.Omega**2) * np.sin(self.Omega * t) ** 2
        phi = self.Omega * t

        u = self.grav * self.sa / self.Omega**2 * np.sin(self.Omega * t) ** 2
        v = (
            self.grav
            * self.sa
            / self.Omega
            * np.sin(self.Omega * t)
            * np.cos(self.Omega * t)
        )
        omega = self.Omega * np.ones_like(t)

        vq = np.array([x, y, phi]).T
        vu = np.array([u, v, omega]).T

        return vq, vu

    def q_dot(self, t, q, u):
        return u

    def q_dot_u(self, t, q):
        return np.eye(self.nq)

    def M(self, t, q):
        return np.diag([self.m, self.m, self.theta])

    def h(self, t, q, u):
        x, y, phi = q
        return np.array([0, self.m * self.grav * self.sa, 0])

    def gamma(self, t, vq, vu):
        x, y, phi = vq
        u, v, omega = vu
        return np.array([v * np.sin(phi) - u * np.cos(phi)])

    def gamma_dot(self, t, vq, vu, vu_dot):
        x, y, phi = vq
        u, v, omega = vu
        u_dot, v_dot, omega_dot = vu_dot
        return np.array(
            [
                v_dot * np.sin(phi)
                + v * np.cos(phi) * omega
                - u_dot * np.cos(phi)
                + u * np.sin(phi) * omega
            ]
        )

    def gamma_q(self, t, vq, vu):
        x, y, phi = vq
        u, v, omega = vu
        return np.array(
            [
                [0, 0, v * np.cos(phi) + u * np.sin(phi)],
            ]
        )

    def gamma_u(self, t, q):
        return self.W_gamma(t, q).T

    def W_gamma(self, t, q):
        x, y, phi = q
        return np.array(
            [
                [-np.cos(phi)],
                [np.sin(phi)],
                [0],
            ]
        )

    def Wla_gamma_q(self, t, q, la_ga):
        x, y, phi = q
        # fmt: off
        return np.array([
            [0, 0, np.sin(phi) * la_ga[0]],
            [0, 0, np.cos(phi) * la_ga[0]],
            [0, 0, 0],
        ])
        # fmt: on


@pytest.mark.parametrize("Solver, kwargs", solvers_and_kwargs)
def test_index2_problem(Solver, kwargs, show=False):
    # create the system
    system = System()
    knife_edge = KnifeEdge()
    system.add(knife_edge)
    system.assemble()

    # call the solver
    t1 = 2 * np.pi / knife_edge.Omega
    # t1 *= 5
    t1 *= 0.1
    dt = 5e-2
    sol = Solver(system, t1, dt, **kwargs).solve()
    t = sol.t
    q = sol.q
    u = sol.u

    # compare with exact solution
    q_true, u_true = knife_edge(t)
    error_q = compute_error(q, q_true, rtol=1e-6, atol=1e-3)
    error_u = compute_error(u, u_true, rtol=1e-6, atol=1e-3)
    print(f"error q: {error_q}")
    print(f"error u: {error_u}")

    # visualization
    if show:
        fig, ax = plt.subplots(2, 1)

        ax[0].plot(t, q[:, 0], "-r", label="x")
        ax[0].plot(t, q_true[:, 0], "rx", label="x_true")
        ax[0].plot(t, q[:, 1], "-g", label="y")
        ax[0].plot(t, q_true[:, 1], "gx", label="y_true")
        ax[0].plot(t, q[:, 2], "-b", label="phi")
        ax[0].plot(t, q_true[:, 2], "bx", label="phi_true")
        ax[0].grid()
        ax[0].legend()

        ax[1].plot(t, u[:, 0], "-r", label="u")
        ax[1].plot(t, u_true[:, 0], "rx", label="u_true")
        ax[1].plot(t, u[:, 1], "-g", label="v")
        ax[1].plot(t, u_true[:, 1], "gx", label="v_true")
        ax[1].plot(t, u[:, 2], "-b", label="omega")
        ax[1].plot(t, u_true[:, 2], "bx", label="omega_true")
        ax[1].grid()
        ax[1].legend()

        plt.show()

    # convergence analysis
    global first
    first = True

    def get_solver(t_final, dt, atol):
        global first
        if first:
            first = False
            t_true = np.arange(0, t_final + dt, dt)
            q_true, u_true = knife_edge(t_true)
            return type(
                "Solver",
                (),
                {
                    "solve": lambda self: Solution(
                        system,
                        t=t_true,
                        q=q_true,
                        u=u_true,
                    )
                },
            )()
        else:
            return Solver(
                system,
                t_final,
                dt,
                options=SolverOptions(
                    fixed_point_atol=atol,
                    fixed_point_rtol=atol,
                    newton_atol=atol,
                    newton_rtol=atol,
                    reuse_lu_decomposition=False,
                ),
                **kwargs,
            )

    errors = convergence_analysis(
        get_solver,
        #
        # dt_ref=1.6e-3,
        # final_power=7,
        # power_span=(1, 5),
        #
        # dt_ref=8e-4,
        # final_power=8,
        # power_span=(1, 6),
        #
        # dt_ref=4e-4,
        # final_power=9,
        # power_span=(1, 7),
        #
        dt_ref=2e-4,
        final_power=10,
        power_span=(1, 8),
        #
        #############
        # other setup
        #############
        states=["q", "u"],
        measure="lp",
        visualize=show,
        export=True,
        # export_name="Rattle",
        kwargs={"p": 1},
    )


if __name__ == "__main__":
    for p in solvers_and_kwargs:
        test_index2_problem(*p, show=True)
