import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.solver import ScipyIVP
from cardillo.discrete import Frame
from cardillo.constraints import RevoluteJoint
from cardillo.discrete import RigidBodyEuler
from cardillo.forces import (
    LinearSpring,
    LinearDamper,
    add_rotational_forcelaw,
    K_Force,
    K_Moment,
)


class RigidCylinder(RigidBodyEuler):
    def __init__(self, m, r, l, q0=None, u0=None):
        A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
        C = 1 / 2 * m * r**2
        K_theta_S = np.diag(np.array([A, A, C]))

        super().__init__(m, K_theta_S, q0=q0, u0=u0)


if __name__ == "__main__":
    m = 1
    r = 0.2
    l = 0.1
    A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
    C = 1 / 2 * m * r**2
    K_theta_S = np.diag(np.array([A, A, C]))

    k = 1
    d = 0.025
    alpha0 = np.pi / 10

    q0 = np.array([0, 0, 0, alpha0, 0, 0])
    u0 = np.zeros(6)
    alpha_dot0 = 10
    u0[5] = alpha_dot0

    rigid_body = RigidCylinder(m, r, l, q0, u0)

    origin = Frame()
    spring = LinearSpring(k)
    damper = LinearDamper(d)
    ActuatedRevoluteJoint = add_rotational_forcelaw(
        RevoluteJoint, force_law_spring=spring
    )
    # ActuatedRevoluteJoint = add_rotational_forcelaw(RevoluteJoint, force_law_damper=damper)
    # ActuatedRevoluteJoint = add_rotational_forcelaw(RevoluteJoint, spring, damper)
    joint = ActuatedRevoluteJoint(
        origin,
        rigid_body,
        np.zeros(3),
        np.eye(3),
    )

    F = K_Force(np.array([0, 0.2, 0]), rigid_body, K_r_SP=np.array([r, 0, 0]))
    # M = K_Moment(np.array([0, 0, -0.04]), rigid_body)
    # M = K_Moment(np.array([1, 0, 0]), rigid_body)
    # M = K_Moment(np.array([0, 1, 0]), rigid_body)
    M = K_Moment(np.array([0, 0, 1]), rigid_body)

    model = System()
    model.add(origin)
    model.add(rigid_body)
    model.add(joint)
    # model.add(F)
    # model.add(M)

    model.assemble()

    t0 = 0
    t1 = 2
    dt = 1.0e-2
    solver = ScipyIVP(model, t1, dt)
    sol = solver.solve()
    t = sol.t
    q = sol.q
    u = sol.u

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(t, q[:, 0], "-r", label="x")
    ax[0].plot(t, q[:, 1], "-g", label="y")
    ax[0].plot(t, q[:, 2], "-b", label="z")
    ax[0].legend()

    ax[1].plot(t, q[:, 3], "-r", label="alpha")
    ax[1].plot(t, q[:, 4], "-g", label="beta")
    ax[1].plot(t, q[:, 5], "-b", label="gamma")
    ax[1].legend()

    plt.show()
