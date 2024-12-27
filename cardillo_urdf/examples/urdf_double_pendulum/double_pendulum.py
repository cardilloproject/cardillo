import numpy as np
from cardillo_urdf.urdf import load_urdf
from matplotlib import pyplot as plt

from cardillo.math import A_IB_basic
from pathlib import Path
from cardillo import System
from cardillo.actuators import PDcontroller
from cardillo.solver import Rattle, BackwardEuler
from cardillo.visualization import Export, Renderer
from cardillo.visualization.trimesh import animate_system, show_system

if __name__ == "__main__":
    from os import path

    dir_name = path.dirname(__file__)

    # Method 1
    initial_config = {}
    initial_config["joint1"] = np.pi / 3
    initial_config["joint2"] = np.pi / 4

    initial_vel = {}
    initial_vel["joint1"] = 0
    initial_vel["joint2"] = 0

    # Method 2
    # initial_config = (np.pi / 2, np.pi/2)
    # initial_vel = (1, 1)

    system = System()

    load_urdf(
        system,
        path.join(
            dir_name,
            "urdf",
            "double_pendulum.urdf",
        ),
        r_OC0=np.array([0, 0, 0]),
        A_IC0=A_IB_basic(0).y @ A_IB_basic(np.pi).x,
        v_C0=np.zeros(3),
        C0_Omega_0=np.array([0, 0, 0]),
        initial_config=initial_config,
        initial_vel=initial_vel,
        gravitational_acceleration=np.array([0, 0, -10]),
    )
    show_system(system, 0, system.q0)
    kp = 2
    kd = 1
    for joint_name in initial_config:
        motor = PDcontroller(system.contributions_map[joint_name], kp, kd, np.array([initial_config[joint_name], 0]))
        motor.name = "PD_" + system.contributions_map[joint_name].name
        system.add(motor)

    system.contributions_map["PD_joint1"].tau = lambda t: np.array([np.pi/2, 0])
    system.contributions_map["PD_joint2"].tau = lambda t: np.array([np.pi/4, 0])
    system.assemble()

    sol = BackwardEuler(system, 5, 1e-2).solve()
    t = sol.t
    q = sol.q
    u = sol.u
    render = Renderer(system, system.contributions)
    render.render_solution(sol, repeat=True)
    # show_system(system, sol.t[5], sol.q[5])
    # animate_system(system, sol.t, sol.q, t_factor=1)
    path = Path(__file__)
    e = Export(
        path=path.parent,
        folder_name=path.stem,
        overwrite=True,
        fps=30,
        solution=sol,
    )

    # Plotting #
    phi1 = [system.contributions_map["joint1"].angle(ti, qi[system.contributions_map["joint1"].qDOF])
        for ti, qi in zip(t, q)]
    phi2 = [system.contributions_map["joint2"].angle(ti, qi[system.contributions_map["joint2"].qDOF])
        for ti, qi in zip(t, q)]

    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    ax[0].plot(t, phi1, "-r", label="$\\varphi_1$")
    ax[0].plot(t, phi2, "-g", label="$\\varphi_2$")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("angle")
    ax[0].legend()
    ax[0].grid()
    
    phi1_dot = [
    system.contributions_map["joint1"].angle_dot(ti, qi[system.contributions_map["joint1"].qDOF], 
                                                 ui[system.contributions_map["joint1"].uDOF])
    for ti, qi, ui in zip(t, q, u)
    ]
    phi2_dot = [
        system.contributions_map["joint2"].angle_dot(ti, qi[system.contributions_map["joint2"].qDOF], 
                                                    ui[system.contributions_map["joint2"].uDOF])
        for ti, qi, ui in zip(t, q, u)
    ]
    
    print(system.contributions_map.keys())
    print(system.contributions_map["joint1"].qDOF)
    print(system.contributions_map["joint1"].uDOF)
    
    ax[1].plot(t, phi1_dot, "-r", label="$\\dot{\\varphi}_1$")
    ax[1].plot(t, phi2_dot, "-g", label="$\\dot{\\varphi}_2$")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("angular velocity")
    ax[1].legend()
    ax[1].grid()

    plt.show()

    for b in system.contributions:
        if hasattr(b, "export"):
            print(f"exporting {b.name}")
            print(f"b.r_OP(0, sol.q[0, b.qDOF]) = {b.r_OP(0, sol.q[1, b.qDOF])}")
            e.export_contr(b)
