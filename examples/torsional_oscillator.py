import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.solver import ScipyIVP, EulerBackward
from cardillo.constraints import RevoluteJoint
from cardillo.discrete import RigidBodyEuler
from cardillo.forces import (
    LinearSpring,
    LinearDamper,
    PDRotationalJoint,
    K_Force,
    K_Moment,
)


class RigidCylinder(RigidBodyEuler):
    def __init__(self, m, r, l, q0=None, u0=None):
        A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
        C = 1 / 2 * m * r**2
        K_theta_S = np.diag(np.array([A, A, C]))

        super().__init__(m, K_theta_S, q0=q0, u0=u0)


def run(Spring=LinearSpring, Damper=LinearDamper, alpha0=0, alpha_dot0=0, g_ref=0):
    m = 1
    r = 0.2
    l = 0.1

    k = 1
    d = 0.025
    g_ref = g_ref
    alpha0 = alpha0

    r_OP0 = np.zeros(3)
    p0_euler = np.array((alpha0, 0, 0))
    q0 = np.hstack((r_OP0, p0_euler))
    v_P0 = np.zeros(3)
    alpha_dot0 = alpha_dot0
    Omega0 = np.array((0, 0, alpha_dot0))
    u0 = np.hstack((v_P0, Omega0))

    system = System()

    rigid_body = RigidCylinder(m, r, l, q0, u0)

    joint = PDRotationalJoint(RevoluteJoint, Spring=Spring, Damper=Damper)(
        subsystem1=system.origin,
        subsystem2=rigid_body,
        r_OB0=np.zeros(3),
        A_IB0=np.eye(3),
        k=k,
        d=d,
        g_ref=g_ref,
    )

    # F = K_Force(np.array([0, 0.2, 0]), rigid_body, K_r_SP=np.array([r, 0, 0]))
    # # M = K_Moment(np.array([0, 0, -0.04]), rigid_body)
    # # M = K_Moment(np.array([1, 0, 0]), rigid_body)
    # # M = K_Moment(np.array([0, 1, 0]), rigid_body)
    # M = K_Moment(np.array([0, 0, 1]), rigid_body)

    system.add(rigid_body, joint)
    # system.add(F)
    # system.add(M)

    system.assemble()

    t1 = 2
    dt = 1.0e-2
    solver = ScipyIVP(system, t1, dt)
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


if __name__ == "__main__":
    alpha0 = np.pi / 10
    alpha_dot0 = 25
    # run(Spring=LinearSpring, Damper=None, alpha0=alpha0, alpha_dot0=alpha_dot0)
    # run(Spring=LinearSpring, Damper=LinearDamper, g_ref=-np.pi/2)
    run(Spring=LinearSpring, Damper=LinearDamper, g_ref=4 * np.pi)
