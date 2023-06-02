import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cardillo.discrete import Cuboid, RigidBodyQuaternion
from cardillo.math import axis_angle2quat, Log_SO3
from cardillo.system import System
from cardillo.constraints import Planarizer
from cardillo.solver import ScipyIVP, RadauIIa, EulerBackward
from cardillo.visualization import Export

planarize = True
# planarize = False

if __name__ == "__main__":
    l = 1
    w = 2
    h = 3
    r_OP0 = np.zeros(3)
    v_P0 = np.zeros(3)
    phi0 = 0
    phi_dot0 = 20
    K_Omega_disturbance = np.array((1e-10, 0, 0))  # disturbance is required
    K_Omega = np.array((0, phi_dot0, 0))
    q0 = np.concatenate((r_OP0, axis_angle2quat(np.array((0, 0, 1)), phi0)))
    u0 = np.concatenate((v_P0, K_Omega + K_Omega_disturbance))
    m = 5
    box = Cuboid(RigidBodyQuaternion)(l, w, h, q0, u0, mass=m)

    system = System()
    system.add(box)
    if planarize:
        system.add(Planarizer(system.origin, box, axis=1))
    system.assemble()

    t1 = 5
    dt = 1e-2
    # solver = RadauIIa(system, t1, dt, dae_index=3)
    solver = EulerBackward(system, t1, dt, method="index 3")
    # solver = ScipyIVP(system, t1, dt)
    sol = solver.solve()

    ###############
    # visualization
    ###############
    t, q = sol.t, sol.q
    r_OP = np.array([box.r_OP(ti, qi) for (ti, qi) in zip(t, q)])
    A_IK = np.array([box.A_IK(ti, qi) for (ti, qi) in zip(t, q)])
    Delta_angles = np.zeros((len(t), 3), dtype=float)
    for i in range(1, len(t)):
        Delta_angles[i] = Log_SO3(A_IK[i - 1].T @ A_IK[i])
    angles = np.cumsum(Delta_angles, axis=0)

    fig, ax = plt.subplots(2, 3)

    ax[0, 0].set_title("x")
    ax[0, 0].plot(t, r_OP[:, 0], "-k")

    ax[0, 1].set_title("y")
    ax[0, 1].plot(t, r_OP[:, 1], "-k")

    ax[0, 2].set_title("z")
    ax[0, 2].plot(t, r_OP[:, 2], "-k")

    ax[1, 0].set_title("psi0")
    ax[1, 0].plot(t, angles[:, 0], "-k")

    ax[1, 1].set_title("psi1")
    ax[1, 1].plot(t, angles[:, 1], "-k")

    ax[1, 2].set_title("psi2")
    ax[1, 2].plot(t, angles[:, 2], "-k")

    plt.show()

    ############
    # vtk export
    ############
    path = Path(__file__)
    e = Export(
        path=path.parent,
        folder_name=path.stem,
        overwrite=True,
        fps=100,
        solution=sol,
        system=system,
    )
    e.export_contr(box)
    e.export_contr(box, base_export=True)
