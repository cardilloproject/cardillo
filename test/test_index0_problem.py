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

# we assume m = 1 and r = 1
omega = 2 * np.pi


def PHI(t):
    """The time derivative of this function has to be phi_p(t)**2."""
    return omega**2 * (t / 2 + np.sin(2 * t) / 4)


def phi(t):
    return omega * np.sin(t)


def phi_p(t):
    return omega * np.cos(t)


# force = phi_pp
def phi_pp(t):
    return -omega * np.sin(t)


def sol_true(t):
    x = np.cos(phi(t))
    y = np.sin(phi(t))
    omega = phi_p(t)

    vq = np.array([x, y]).T
    vu = np.array([omega]).T

    return vq, vu


class ParticleOnCircularTrack:
    def __init__(self):
        self.nq = 2
        self.nu = 1
        self.q0, self.u0 = sol_true(0)
        # self.u0 = np.array([omega])

    def q_dot(self, t, q, u):
        x, y = q
        omega = u[0]
        return np.array([-y, x]) * omega

    def q_dot_u(self, t, q):
        x, y = q
        return np.array([[-y], [x]])

    def M(self, t, q):
        return np.eye(1)

    def h(self, t, q, u):
        force = phi_pp(t)
        return np.array([force])


@pytest.mark.parametrize("Solver, kwargs", solvers_and_kwargs)
def test_index0_problem(Solver, kwargs, show=False):
    # create the system
    system = System()
    particle = ParticleOnCircularTrack()
    system.add(particle)
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # call the solver
    t1 = 2 * np.pi
    t1 *= 0.1
    dt = 1e-2
    sol = Solver(system, t1, dt, **kwargs).solve()
    t = sol.t
    q = sol.q
    u = sol.u

    # compare with exact solution
    q_true, u_true = sol_true(t)
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
        ax[0].grid()
        ax[0].legend()

        ax[1].plot(t, u[:, 0], "-r", label="omega")
        ax[1].plot(t, u_true[:, 0], "rx", label="omega_true")
        ax[1].grid()
        ax[1].legend()

        plt.show()

    # exit()

    # convergence analysis
    global first
    first = True

    def get_solver(t_final, dt, atol):
        global first
        if first:
            first = False
            t_true = np.arange(0, t_final + dt, dt)
            q_true, u_true = sol_true(t_true)
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
        # P_g
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
        test_index0_problem(*p, show=True)
