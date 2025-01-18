from turtle import home
import numpy as np
from cardillo_urdf.urdf import load_urdf
from matplotlib import pyplot as plt
from cardillo.math import A_IB_basic, cross3
from pathlib import Path
from cardillo import System
from cardillo.actuators import PDcontroller
from cardillo.solver import Rattle, BackwardEuler, ScipyIVP, ScipyDAE
from cardillo.visualization import Export, Renderer
from cardillo.visualization.trimesh import animate_system, show_system
from cardillo.constraints import Revolute
from cardillo.discrete import RigidBody, Frame, Meshed
from cardillo.forces import Force

def double_pendulum_urdf():
    
    from os import path
    dir_name = Path(__file__).parent

    initial_config = {}
    initial_config["joint1"] = np.pi/1000
    initial_config["joint2"] = np.pi/100
    initial_vel = {}
    initial_vel["joint1"] = 0
    initial_vel["joint2"] = 0
    
    system = System()

    load_urdf(
        system,
        path.join(
            dir_name,
            "cardillo_urdf",
            "examples",
            "urdf_double_pendulum",
            "urdf",
            "double_pendulum.urdf",
        ),
        r_OC0=np.array([0, 0, 0]),
        A_IC0=A_IB_basic(0).y @ A_IB_basic(np.pi).x,
        v_C0=np.zeros(3),
        C0_Omega_0=np.array([0, 0, 0]),
        initial_config=initial_config,
        initial_vel=initial_vel,
        base_link_is_floating=False,
        gravitational_acceleration=np.array([0, 0, -10]),
    )


    dt = 1.0e-2
    ren = Renderer(system, system.contributions)
    sol = BackwardEuler(system, 5, dt).solve()
    ren.render_solution(sol, repeat=True)
    t1 = sol.t
    q = sol.q
    u = sol.u

    path = Path(__file__)
    e = Export(
        path=path.parent,
        folder_name=path.stem,
        overwrite=True,
        fps=30,
        solution=sol,
    )

    phi1 = [system.contributions_map["joint1"].angle(ti, qi[system.contributions_map["joint1"].qDOF])
        for ti, qi in zip(t1, q)]
    phi2 = [system.contributions_map["joint2"].angle(ti, qi[system.contributions_map["joint2"].qDOF])
        for ti, qi in zip(t1, q)]

    phi1_dot = [
    system.contributions_map["joint1"].angle_dot(ti, qi[system.contributions_map["joint1"].qDOF], 
                                                 ui[system.contributions_map["joint1"].uDOF])
    for ti, qi, ui in zip(t1, q, u)
    ]
    phi2_dot = [
        system.contributions_map["joint2"].angle_dot(ti, qi[system.contributions_map["joint2"].qDOF], 
                                                    ui[system.contributions_map["joint2"].uDOF])
        for ti, qi, ui in zip(t1, q, u)
    ]

    plt.show()

    for b in system.contributions:
        if hasattr(b, "export"):
            print(f"exporting {b.name}")
            print(f"b.r_OP(0, sol.q[0, b.qDOF]) = {b.r_OP(0, sol.q[1, b.qDOF])}")
            e.export_contr(b)            
    return phi1, phi2, phi1_dot, phi2_dot, t1

def double_pendulum_nonurdf():

    dir_name = Path(__file__).parent
    g = np.array([0, 0, -10])

    # initial conditions
    phi10 = np.pi/1000
    phi20 = np.pi/100
    phi1_dot0 = 0
    phi2_dot0 = 0

    # initialize system
    system = System()

    # create base link and add it to system
    base_link = Meshed(Frame)(
        mesh_obj=Path(dir_name, "examples", "double_pendulum","stl","base_link.stl"),
        B_r_CP=np.array([-0.0030299, -2.6279E-13, -0.02912]),
        A_BM=np.eye(3),
        r_OP=np.zeros(3),
        A_IB=np.eye(3),
        name="base_link",
    )
    system.add(base_link)

    ########
    # link 1
    ########

    # position of joint 1
    r_OJ1 = np.array([3.0573e-03, -2.6279e-13, 5.8800e-03])

    # kinematics and initial conditions of link 1
    B1_r_J1C1 = np.array([0.0086107, 2.1727E-06, 0.036012])
    A_IB1 = A_IB_basic(np.pi + phi10).x
    r_J1C1 = A_IB1 @ B1_r_J1C1
    r_OC1 = r_OJ1 + r_J1C1
    B1_Omega1 = np.array([phi1_dot0, 0, 0])
    v_C1 = A_IB1 @ cross3(B1_Omega1, B1_r_J1C1)

    q0 = RigidBody.pose2q(r_OC1, A_IB1)
    u0 = np.hstack([v_C1, B1_Omega1])

    # create link 1
    link1 = Meshed(RigidBody)(
        mesh_obj=Path(dir_name, "examples", "double_pendulum","stl","link1.stl"),
        B_r_CP=np.array([-0.0086107, -2.1727E-06, -0.036012]),
        A_BM=np.eye(3),
        mass=0.26703,
        B_Theta_C=np.array(
            [
                [0.00040827, 1.2675E-09, 1.8738e-05],
                [1.2675E-09, 0.00038791, 3.5443e-08],
                [1.8738e-05, 3.5443e-08, 3.6421e-05],
            ]
        ),
        q0=q0,
        u0=u0,
        name="link1",
    )

    # gravity for link 1
    gravity1 = Force(link1.mass * g, link1, name="gravity_link1")

    # add contributions to the system
    system.add(link1, gravity1)

    #########
    # joint 1
    #########

    joint1 = Revolute(
        base_link, link1, axis=0, r_OJ0=r_OJ1, angle0=phi10, name="joint1"
    )
    system.add(joint1)

    ########
    # link 2
    ########

    # position of joint 2
    B1_r_J1J2 = np.array([0.023, 0, 0.1])
    r_OJ2 = r_OJ1 + A_IB1 @ B1_r_J1J2
    v_J2 = A_IB1 @ cross3(B1_Omega1, B1_r_J1J2)

    # kinematics of link 2
    B2_r_J2C2 = np.array([-0.0050107, 1.9371E-10, 0.10088])
    A_IB2 = A_IB_basic(np.pi + phi20).x
    r_J2C2 = A_IB2 @ B2_r_J2C2
    r_OC2 = r_OJ2 + r_J2C2

    B_Omega2 = np.array([phi2_dot0, 0, 0])
    v_C2 = v_J2 + A_IB2 @ cross3(B_Omega2, B2_r_J2C2)

    q0 = RigidBody.pose2q(r_OC2, A_IB2)
    u0 = np.hstack([v_C2, B_Omega2])

    link2 = Meshed(RigidBody)(
        mesh_obj=Path(dir_name, "examples", "double_pendulum","stl","link2.stl"),
        B_r_CP=np.array([0.0050107, -1.9371E-10, -0.10088]),
        A_BM=np.eye(3),
        mass=0.33238,
        B_Theta_C=np.array(
            [
                [0.0011753, -3.8540e-13, -2.9304e-08],
                [-3.8540e-13, 0.0011666, -5.2365e-12],
                [-2.9304e-08, -5.2365e-12, 1.4553e-05],
            ]
        ),
        q0=q0,
        u0=u0,
        name="link2",
    )

    # gravity for link 2
    gravity2 = Force(link2.mass * g, link2, name="gravity_link2")

    # add contributions to the system
    system.add(link2, gravity2)

    #########
    # joint 2
    #########

    joint2 = Revolute(link1, link2, axis=0, r_OJ0=r_OJ2, angle0=phi20, name="joint2")
    system.add(joint2)

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    dt = 1.0e-2  # time step
    # solver = ScipyIVP(system, t1, dt)  # create solver
    ren = Renderer(system, system.contributions)
    solver = BackwardEuler(system, 5, dt)  # create solver
    sol = solver.solve()  # simulate system
    ren.render_solution(sol, repeat=True)


    # read solution
    t = sol.t
    q = sol.q
    u = sol.u

    phi1 = [joint1.angle(ti, qi) for ti, qi in zip(t, q[:, joint1.qDOF])]
    phi2 = [joint2.angle(ti, qi) for ti, qi in zip(t, q[:, joint2.qDOF])]

    # plot time evolution for angular velocities
    phi1_dot = [
        joint1.angle_dot(ti, qi, ui)
        for ti, qi, ui in zip(t, q[:, joint1.qDOF], u[:, joint1.uDOF])
    ]
    phi2_dot = [
        joint2.angle_dot(ti, qi, ui)
        for ti, qi, ui in zip(t, q[:, joint2.qDOF], u[:, joint2.uDOF])
    ]

    return phi1, phi2, phi1_dot, phi2_dot, t

if __name__ == "__main__":
    phi1_urdf, phi2_urdf, phi1_dot_urdf, phi2_dot_urdf, t1 = double_pendulum_urdf()
    phi1, phi2, phi1_dot, phi2_dot, t = double_pendulum_nonurdf()
        # Combined plot for angles and angular velocities


    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))
    ax[0].plot(t1, phi1_urdf, "-r", label="$\\varphi_1$")
    ax[0].plot(t1, phi1, "-g", label="$\\varphi_1$")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("angle phi1")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(t1, phi2_urdf, "-r", label="$\\varphi_2$")
    ax[1].plot(t1, phi2, "-g", label="$\\varphi_2$")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("angle phi2")
    ax[1].legend()
    ax[1].grid()
    ax[2].plot(t1, phi1_dot_urdf, "-r", label="$\\dot{\\varphi}_1$")
    ax[2].plot(t1, phi1_dot, "-g", label="$\\dot{\\varphi}_1$")
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("angular velocity phi1_dot")
    ax[2].legend()
    ax[2].grid()
    ax[3].plot(t1, phi2_dot_urdf, "-r", label="$\\dot{\\varphi}_2$")
    ax[3].plot(t1, phi2_dot, "-g", label="$\\dot{\\varphi}_2$")
    ax[3].set_xlabel("t")
    ax[3].set_ylabel("anglular velocity phi2_dot")
    ax[3].legend()
    ax[3].grid()


    plt.show()
