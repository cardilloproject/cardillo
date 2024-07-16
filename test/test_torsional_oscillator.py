import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pytest
from itertools import product

from cardillo import System
from cardillo.solver import ScipyIVP, ScipyDAE, BackwardEuler, Moreau, Rattle
from cardillo.constraints import Revolute
from cardillo.discrete import RigidBody
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.math import Exp_SO3, axis_angle2quat, norm

# solver parameters
t_span = (0.0, 2.0)
t0, t1 = t_span
dt = 1.0e-3

# axis angle rotation axis
psi = np.random.rand(3)
A_IB0 = Exp_SO3(psi)
print(f"A_IB0:\n{A_IB0}")

# initial rotational velocity e_z^K axis
alpha_dot0 = 2
rotation_axis = 0

# parameters
k = 1e1
d = 0.05
g_ref = 2 * np.pi
l = 0.1
m = 1
r = 0.2
A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
C = 1 / 2 * m * r**2
B_Theta_C = np.diag(np.array([A, A, C]))


def run(Solver, solver_kwargs, show=False):
    ############################################################################
    #                   system setup
    ############################################################################

    ####################
    # initial conditions
    ####################
    r_OP0 = np.zeros(3)
    v_P0 = np.zeros(3)
    B_Omega0 = alpha_dot0 * np.eye(3)[rotation_axis]
    u0 = np.hstack((v_P0, B_Omega0))

    ######################
    # define contributions
    ######################
    system = System()

    n_psi = norm(psi)
    p = axis_angle2quat(psi / n_psi, n_psi)
    q0 = np.hstack((r_OP0, p))
    rigid_body = RigidBody(m, B_Theta_C, q0, u0)

    joint = Revolute(
        system.origin,
        rigid_body,
        rotation_axis,
        r_OJ0=np.zeros(3),
        A_IJ0=A_IB0,
    )

    spring = SpringDamper(joint, k, d, l_ref=g_ref)

    system.add(rigid_body, joint, spring)
    system.assemble()

    ############################################################################
    #                   DAE solution
    ############################################################################
    sol = Solver(system, t1, dt, **solver_kwargs).solve()
    t, q = sol.t, sol.q

    ############################################################################
    #                   plot
    ############################################################################
    if show:
        joint.reset()
        alpha_cmp = [joint.angle(ti, qi[joint.qDOF]) for ti, qi in zip(t, q)]

        Theta = B_Theta_C[rotation_axis, rotation_axis]

        def eqm(t, x):
            dx = np.zeros(2)
            dx[0] = x[1]
            dx[1] = -1 / Theta * (d * x[1] + k * (x[0] - g_ref))
            return dx

        x0 = np.array((0, alpha_dot0))
        ref = solve_ivp(eqm, [0, t1], x0, method="RK45", rtol=1e-8, atol=1e-12)
        x = ref.y
        t_ref = ref.t
        alpha_ref = x[0]

        fig, ax = plt.subplots(1, 1)

        ax.plot(t, alpha_cmp, "-k", label="alpha")
        ax.plot(t_ref, alpha_ref, "-.r", label="alpha_ref")
        ax.legend()

        plt.show()


################################################################################
# test setup
################################################################################
test_parameters = [
    (ScipyIVP, {}),
    (ScipyDAE, {}),
    (BackwardEuler, {}),
    (Moreau, {}),
    (Rattle, {}),
]


@pytest.mark.parametrize("Solver, kwargs", test_parameters)
def test_torsional_oscillator(Solver, kwargs, show=False):
    run(Solver, kwargs, show)


if __name__ == "__main__":
    for p in test_parameters:
        test_torsional_oscillator(*p, show=True)
