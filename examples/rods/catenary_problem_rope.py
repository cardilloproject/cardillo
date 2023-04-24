from cardillo.discrete import Frame
from cardillo.constraints import (
    Spherical,
)
from cardillo.beams import (
    Rope,
    animate_rope,
)
from cardillo.forces.scalar_force_laws import LinearSpring
from cardillo.beams.rope import QuadraticMaterial as QuadraticMaterialRope
from cardillo.forces import DistributedForce1DBeam
from cardillo import System
from cardillo.solver import (
    Newton,
    ScipyIVP,
)
from cardillo.math import e1, e2

import numpy as np

case = "statics"
# case = "dynamics"

# TODO: add inextensibility constraint and compare solution with catenary
if __name__ == "__main__":
    # rope parameters
    g = 9.81  # gravity constant
    L = 3.14  # reference length rope
    A_rho0 = 1.0e1  # reference density per unit length

    # material law rope
    k_e = 1.0e2  # extensional stiffness
    material_model = QuadraticMaterialRope(k_e)  # quadratic potential

    ############################################################################
    #                   rope + boundary constraints
    ############################################################################

    # discretization properties
    nelements = 5
    # polynomial_degree = 1
    # basis = "Lagrange"
    polynomial_degree = 3
    basis = "B-spline"

    # left boundary condition
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    # right boundary condition
    r_OP1 = L * e1

    # initial configuration: straight line
    q0 = Rope.straight_configuration(
        basis,
        polynomial_degree,
        nelements,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
    )

    # initial nodal/control point positions
    r0 = q0.copy().reshape(3, -1, order="C")

    # reference configuration: corresponds to initial configuration
    Q = q0

    # gravitational line force density
    __fg = -A_rho0 * g * e2

    if case == "statics":
        # increasing gravity
        fg = lambda t, xi: t * __fg

        # Manipulate initial configuration in order to overcome singular initial
        # configuration. Do not change first and last node, otherwise constraints
        # are violated!
        eps = 1.0e-4
        nn = r0.shape[1]
        for i in range(1, nn - 1):
            r0[:2, i] += eps * 0.5 * (2.0 * np.random.rand(2) - 1)
        q0 = r0.reshape(-1, order="C")
    elif case == "dynamics":
        fg = lambda t, xi: __fg

    rope = Rope(
        material_model,
        A_rho0,
        polynomial_degree,
        nelements,
        Q,
        q0=q0,
        basis=basis,
    )

    # left joint
    frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)
    joint1 = Spherical(frame1, rope, r_OP0, frame_ID2=(0,))

    # left joint
    frame2 = Frame(r_OP=r_OP1, A_IK=A_IK0)
    joint2 = Spherical(frame2, rope, r_OP1, frame_ID2=(1,))

    # gravitational line force density
    gravity = DistributedForce1DBeam(fg, rope)

    ############################################################################
    #                   model
    ############################################################################
    system = System()
    system.add(rope)
    system.add(frame1)
    system.add(joint1)
    system.add(frame2)
    system.add(joint2)
    system.add(gravity)
    system.assemble()

    # # show initial configuration
    # animate_rope([0], [q0], [rope], L, show=True)

    ############################################################################
    #                   solver
    ############################################################################
    # solver parameter
    if case == "statics":
        atol = 1.0e-8
        rtol = 0.0
        n_load_steps = 10
        max_iter = 20
        solver = Newton(
            system,
            n_load_steps=n_load_steps,
            atol=atol,
            max_iter=max_iter,
        )
    elif case == "dynamics":
        atol = 1.0e-8
        rtol = 1.0e-6
        t1 = 1
        dt = 1.0e-2
        method = "RK45"
        solver = ScipyIVP(
            system,
            t1,
            dt,
            method=method,
            rtol=rtol,
            atol=atol,
        )

    sol = solver.solve()
    q = sol.q
    t = sol.t

    ############################################################################
    #                   visualization
    ############################################################################

    animate_rope(t, q, [rope], L, show=True)
