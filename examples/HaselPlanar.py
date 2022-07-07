from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import SphericalJoint
from cardillo.beams import (
    Rope,
    animate_rope,
)
from cardillo.forces import DistributedForce1D
from cardillo.model import Model
from cardillo.solver import (
    Newton,
    ScipyIVP,
)
from cardillo.math import e1, e2, e3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


if __name__ == "__main__":
    # statics or dynamics?
    # statics = True
    statics = False

    # solver parameter
    if statics:
        atol = 1.0e-8
        rtol = 0.0
        n_load_steps = 10
        max_iter = 20
    else:
        atol = 1.0e-8
        rtol = 1.0e-6
        t1 = 1
        dt = 1.0e-2
        method = "RK45"

    # discretization properties
    nelements = 5
    polynomial_degree = 1

    # rope parameters
    g = 9.81
    L = 3.14
    k_e = 1.0e2
    A_rho0 = 1.0e0

    # starting point and corresponding orientation
    r_OP0 = np.zeros(3, dtype=float)
    A_IK0 = np.eye(3, dtype=float)

    # end point
    r_OP1 = L * e1

    # straight initial configuration
    Q = Rope.straight_configuration(
        polynomial_degree,
        nelements,
        L,
        r_OP=r_OP0,
        A_IK=A_IK0,
    )

    # Manipulate initial configuration in order to overcome singular initial
    # configuration. Do not change first and last node, otherwise constraints
    # are violated!
    eps = 1.0e-3
    q0 = Q.copy().reshape(-1, 3)
    nn = len(q0)
    for i in range(1, nn - 1):
        q0[i] += eps * 0.5 * (2.0 * np.random.rand(3) - 1)
    q0 = q0.reshape(-1)

    # build rope class
    rope = Rope(
        k_e,
        polynomial_degree,
        A_rho0,
        nelements,
        Q,
        q0=q0,
    )

    # left joint
    frame1 = Frame(r_OP=r_OP0, A_IK=A_IK0)
    joint1 = SphericalJoint(frame1, rope, r_OP0, frame_ID2=(0,))

    # left joint
    frame2 = Frame(r_OP=r_OP0, A_IK=A_IK0)
    joint2 = SphericalJoint(frame2, rope, r_OP1, frame_ID2=(1,))

    __fg = -A_rho0 * g * e3
    if statics:
        fg = lambda t, xi: t * __fg
    else:
        fg = lambda t, xi: __fg
    gravity = DistributedForce1D(fg, rope)

    # assemble the model
    model = Model()
    model.add(rope)
    model.add(frame1)
    model.add(joint1)
    model.add(frame2)
    model.add(joint2)
    model.add(gravity)
    model.assemble()

    if statics:
        solver = Newton(
            model,
            n_load_steps=n_load_steps,
            max_iter=max_iter,
            atol=atol,
            rtol=rtol,
        )
    else:
        solver = ScipyIVP(
            model,
            t1,
            dt,
            method=method,
            rtol=rtol,
            atol=atol,
        )

    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    animate_rope(t, q, [rope], L, show=True)
