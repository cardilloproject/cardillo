import numpy as np
import matplotlib.pyplot as plt

from cardillo.model import System
from cardillo.math.algebra import A_IK_basic_z
from cardillo.solver import Generalized_alpha_1, Scipy_ivp
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import (
    Revolute_joint as Revolute_joint_implicit,
)
from cardillo.model.bilateral_constraints.explicit import (
    Revolute_joint as Revolute_joint_explicit,
)
from cardillo.model.rigid_body import Rigid_body_euler, Rigid_body_rel_kinematics
from cardillo.model.scalar_force_interactions.force_laws import (
    Linear_spring,
    Linear_damper,
    Linear_spring_damper,
)
from cardillo.model.scalar_force_interactions import add_rotational_forcelaw
from cardillo.model.force import K_Force
from cardillo.model.moment import K_Moment

implicit = True
# implicit = False


class Rigid_cylinder(Rigid_body_euler):
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

    if implicit:
        RB = Rigid_cylinder(m, r, l, q0, u0)

        Origin = Frame()
        spring = Linear_spring(k)
        rot_spring = add_rotational_forcelaw(spring, Revolute_joint_implicit)(
            Origin, RB, np.zeros(3), np.eye(3)
        )
        # damper = Linear_damper(d)
        # rot_damper = add_rotational_forcelaw(damper, Revolute_joint_implicit)(Origin, RB, np.zeros(3), np.eye(3))

    else:

        Origin = Frame()
        # spring = Linear_spring(k)
        # rot_spring = add_rotational_forcelaw(spring, Revolute_joint_explicit)(np.zeros(3), np.eye(3), q0=np.array([alpha0]))
        # RB = Rigid_body_rel_kinematics(m, K_theta_S, rot_spring, Origin, A_IK0=A_IK_basic_z(alpha0))
        # damper = Linear_damper(d)

        damper = Linear_spring_damper(k, d)
        rot_damper = add_rotational_forcelaw(damper, Revolute_joint_explicit)(
            np.zeros(3), np.eye(3), q0=np.array([alpha0]), u0=np.array([alpha_dot0])
        )
        RB = Rigid_body_rel_kinematics(
            m, K_theta_S, rot_damper, Origin, A_IK0=A_IK_basic_z(alpha0)
        )

    F = K_Force(np.array([0, 0.2, 0]), RB, K_r_SP=np.array([r, 0, 0]))
    # M = K_Moment(np.array([0, 0, -0.04]), RB)
    # M = K_Moment(np.array([1, 0, 0]), RB)
    # M = K_Moment(np.array([0, 1, 0]), RB)
    M = K_Moment(np.array([0, 0, 1]), RB)

    model = System()
    model.add(Origin)
    if implicit:
        model.add(RB)
        model.add(rot_spring)
    else:
        model.add(rot_damper)
        model.add(RB)

    # model.add(F)
    # model.add(M)

    model.assemble()

    t0 = 0
    t1 = 2
    dt = 1.0e-2
    # solver = Generalized_alpha_1(model, t1)
    solver = Scipy_ivp(model, t1, dt)
    sol = solver.solve()
    t = sol.t
    q = sol.q
    u = sol.u

    if implicit:
        fig, ax = plt.subplots(1, 2)

        ax[0].plot(t, q[:, 0], "-r", label="x")
        ax[0].plot(t, q[:, 1], "-g", label="y")
        ax[0].plot(t, q[:, 2], "-b", label="z")
        ax[0].legend()

        ax[1].plot(t, q[:, 3], "-r", label="alpha")
        ax[1].plot(t, q[:, 4], "-g", label="beta")
        ax[1].plot(t, q[:, 5], "-b", label="gamma")
        ax[1].legend()
    else:
        plt.plot(t, q[:, 0], "-r", label="alpha")
        plt.legend()

    plt.show()
