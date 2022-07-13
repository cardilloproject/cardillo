import numpy as np
import matplotlib.pyplot as plt

from PyPanto import Model
from PyPanto import Frame, RevoluteJoint
from PyPanto.solver import ScipyIVP
from PyPanto.rigid_bodies import RigidBodyEuler
from PyPanto.scalar_force_interactions import LinearSpring, add_rotational_forcelaw


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
    phi0 = np.pi / 2 * 0
    print(f"phi0: {phi0}")
    print(f"pi: {np.pi}")

    q0 = np.array([0, 0, 0, 0, phi0, 0])
    u0 = np.zeros(6)
    # alpha_dot0 = 0
    # alpha_dot0 = 22
    # alpha_dot0 = 23
    alpha_dot0 = 48
    u0[5] = alpha_dot0

    RB = RigidCylinder(m, r, l, q0, u0)

    origin = Frame()
    factor = 10
    spring = LinearSpring(k, g0=-factor * np.pi / 2)
    joint_with_spring = add_rotational_forcelaw(spring, RevoluteJoint)(
        origin, RB, np.zeros(3), np.eye(3)
    )

    model = Model()
    model.add(origin)
    model.add(RB)
    model.add(joint_with_spring)
    model.assemble()

    t0 = 0
    t1 = 5
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
    ax[1].plot(
        [t[0], t[-1]], [factor * 0.5 * np.pi, factor * 0.5 * np.pi], "--k", label="pi/2"
    )
    ax[1].plot([t[0], t[-1]], [factor * np.pi, factor * np.pi], "-k", label="pi")
    # ax[1].plot([t[0], t[-1]], [2 * np.pi, 2 * np.pi], "--k", label="2pi")
    # ax[1].plot([t[0], t[-1]], [2.1 * np.pi, 2.1 * np.pi], "-.r", label="2.1pi")
    # ax[1].plot([t[0], t[-1]], [4 * np.pi, 4 * np.pi], "-.k", label="4pi")
    ax[1].legend()

    plt.show()
