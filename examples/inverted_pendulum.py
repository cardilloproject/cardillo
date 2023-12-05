import numpy as np

from cardillo import System

from cardillo.discrete import RigidBody
from cardillo.constraints import Revolute
from cardillo.forces import Force
from cardillo.transmissions import RotationalTransmission
from cardillo.actuators import Motor

from cardillo.solver import Moreau

from cardillo.math import A_IK_basic, Spurrier, cross3

if __name__ == '__main__':

    l = 1
    m = 1
    theta_S = m * (l**2) / 12

    phi0 = 0
    phi_dot0 = 0

    system = System()

    r_OS0 = l * np.array([np.sin(phi0), -np.cos(phi0), 0])
    A_IK0 = A_IK_basic(phi0).z()
    K_Omega0 = np.array([0, 0, phi_dot0])
    v_S0 = cross3(K_Omega0, r_OS0) # I_Omega0 = K_Omega0

    q0 = RigidBody.pose2q(r_OS0, A_IK0)
    u0 = np.concatenate([v_S0, K_Omega0])
    pendulum = RigidBody(m, theta_S*np.eye(3), q0=q0, u0=u0)
    pendulum.name = "pendulum"

    joint = Revolute(system.origin, pendulum, axis=2, A_IB0=A_IK_basic(-np.pi/2).z())
    joint.name = "revolute joint"

    # TODO: do this with transmission?
    gravity = Force(np.array([0, -10 * m, 0]), pendulum)

    system.add(pendulum, gravity, joint)

    # add moment
    M = lambda t: 1
    moment = Motor(RotationalTransmission)(M, subsystem=joint)
    system.add(moment)

    system.assemble()


    t1 = 3
    dt = 1e-2
    sol = Moreau(system, t1, dt).solve()

    from matplotlib import pyplot as plt

    angle = []
    for ti, qi in zip(sol.t, sol.q):
        angle.append(joint.angle(ti, qi))

    plt.plot(sol.t, angle)
    plt.show()





