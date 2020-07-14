import numpy as np
from math import cos, sin, pi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from cardillo.math.algebra import A_IK_basic_z

from cardillo.model import Model
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.explicit import Revolute_joint, Rigid_connection
from cardillo.model.rigid_body import Rigid_body_rel_kinematics
from cardillo.model.force import Force
from cardillo.solver import Euler_forward, Scipy_ivp

from scipy.integrate import solve_ivp



if __name__ == "__main__":
    animate = True
    reference_solution = True

    m = 1
    r = 0.1
    l = 2
    g = 9.81

    A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
    C = 1 / 2 * m * r**2
    K_theta_S = np.diag(np.array([A, C, A]))

    alpha0 = pi / 2
    alpha_dot0 = 0
  
    r_OB1 = np.zeros(3)
    A_IB1 = np.eye(3)
    origin = Frame(r_OP=r_OB1, A_IK=A_IB1)
    joint1 = Revolute_joint(r_OB1, A_IB1, q0=np.array([alpha0]), u0=np.array([alpha_dot0]))
    A_IK10 = A_IK_basic_z(alpha0)
    r_OS10 = - 0.5 * l * A_IK10[:, 1]
    RB1 = Rigid_body_rel_kinematics(m, K_theta_S, joint1, origin, r_OS0=r_OS10, A_IK0=A_IK10)

    beta0 = 0
    beta_dot0 = 0
  
    r_OB2 = - l * A_IK10[:, 1]
    A_IB2 = A_IK10
    joint2 = Revolute_joint(r_OB2, A_IB2, q0=np.array([beta0]), u0=np.array([beta_dot0]))
    # joint2 = Rigid_connection(r_OB2, A_IB2)
    A_IK20 = A_IK10 @ A_IK_basic_z(beta0)
    r_OS20 = r_OB2 - 0.5 * l * A_IK20[:, 1]
    RB2 = Rigid_body_rel_kinematics(m, K_theta_S, joint2, RB1, r_OS0=r_OS20, A_IK0=A_IK20)

    model = Model()
    model.add(origin)
    model.add(RB1)
    model.add(RB2)
    model.add(Force(lambda t: np.array([0, -g * m, 0]), RB1))
    model.add(Force(lambda t: np.array([0, -g * m, 0]), RB2))

    model.assemble()

    t0 = 0
    t1 = 5
    dt = 1e-1
    # solver = Euler_forward(model, t1, dt)
    solver = Scipy_ivp(model, t1, dt)
    sol = solver.solve()
    t = sol.t
    q = sol.q





    if animate:

        # animate configurations
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        scale = 2 * l
        ax.set_xlim3d(left=-scale, right=scale)
        ax.set_ylim3d(bottom=-scale, top=scale)
        ax.set_zlim3d(bottom=-scale, top=scale)

        def init(t, q):
            x_0, y_0, z_0 = origin.r_OP(t)
            x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])
            x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF])
            
            A_IK1 = RB1.A_IK(t, q[RB1.qDOF])
            d11 = A_IK1[:, 0]
            d21 = A_IK1[:, 1]
            d31 = A_IK1[:, 2]

            A_IK2 = RB2.A_IK(t, q[RB2.qDOF])
            d12 = A_IK2[:, 0]
            d22 = A_IK2[:, 1]
            d32 = A_IK2[:, 2]

            # COM, = ax.plot([x_0, x_S1], [y_0, y_S1], [z_0, z_S1], '-ok')
            COM, = ax.plot([x_0, x_S1, x_S2], [y_0, y_S1, y_S2], [z_0, z_S1, z_S2], '-ok')
            d11_, = ax.plot([x_S1, x_S1 + d11[0]], [y_S1, y_S1 + d11[1]], [z_S1, z_S1 + d11[2]], '-r')
            d21_, = ax.plot([x_S1, x_S1 + d21[0]], [y_S1, y_S1 + d21[1]], [z_S1, z_S1 + d21[2]], '-g')
            d31_, = ax.plot([x_S1, x_S1 + d31[0]], [y_S1, y_S1 + d31[1]], [z_S1, z_S1 + d31[2]], '-b')
            d12_, = ax.plot([x_S2, x_S2 + d12[0]], [y_S2, y_S2 + d12[1]], [z_S2, z_S2 + d12[2]], '-r')
            d22_, = ax.plot([x_S2, x_S2 + d22[0]], [y_S2, y_S2 + d22[1]], [z_S2, z_S2 + d22[2]], '-g')
            d32_, = ax.plot([x_S2, x_S2 + d32[0]], [y_S2, y_S2 + d32[1]], [z_S2, z_S2 + d32[2]], '-b')

            return COM, d11_, d21_, d31_, d12_, d22_, d32_

        def update(t, q, COM, d11_, d21_, d31_, d12_, d22_, d32_):
        # def update(t, q, COM, d11_, d21_, d31_):
            x_0, y_0, z_0 = origin.r_OP(t)
            x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF], K_r_SP=np.array([0, -l / 2, 0]))
            x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF], K_r_SP=np.array([0, -l / 2, 0]))
            
            A_IK1 = RB1.A_IK(t, q[RB1.qDOF])
            d11 = A_IK1[:, 0]
            d21 = A_IK1[:, 1]
            d31 = A_IK1[:, 2]

            A_IK2 = RB2.A_IK(t, q[RB2.qDOF])
            d12 = A_IK2[:, 0]
            d22 = A_IK2[:, 1]
            d32 = A_IK2[:, 2]


            COM.set_data([x_0, x_S1, x_S2], [y_0, y_S1, y_S2])
            COM.set_3d_properties([z_0, z_S1, z_S2])
            # COM.set_data([x_0, x_S1], [y_0, y_S1])
            # COM.set_3d_properties([z_0, z_S1])

            d11_.set_data([x_S1, x_S1 + d11[0]], [y_S1, y_S1 + d11[1]])
            d11_.set_3d_properties([z_S1, z_S1 + d11[2]])

            d21_.set_data([x_S1, x_S1 + d21[0]], [y_S1, y_S1 + d21[1]])
            d21_.set_3d_properties([z_S1, z_S1 + d21[2]])

            d31_.set_data([x_S1, x_S1 + d31[0]], [y_S1, y_S1 + d31[1]])
            d31_.set_3d_properties([z_S1, z_S1 + d31[2]])

            d12_.set_data([x_S2, x_S2 + d12[0]], [y_S2, y_S2 + d12[1]])
            d12_.set_3d_properties([z_S2, z_S2 + d12[2]])

            d22_.set_data([x_S2, x_S2 + d22[0]], [y_S2, y_S2 + d22[1]])
            d22_.set_3d_properties([z_S2, z_S2 + d22[2]])

            d32_.set_data([x_S2, x_S2 + d32[0]], [y_S2, y_S2 + d32[1]])
            d32_.set_3d_properties([z_S2, z_S2 + d32[2]])

            return COM, d11_, d21_, d31_, d12_, d22_, d32_


        # COM, d11_, d21_, d31_ = init(0, q[0])
        COM, d11_, d21_, d31_, d12_, d22_, d32_ = init(0, q[0])

        def animate(i):
            update(t[i], q[i], COM, d11_, d21_, d31_, d12_, d22_, d32_)
            # update(t[i], q[i], COM, d11_, d21_, d31_)
        
        # compute naimation interval according to te - ts = frames * interval / 1000
        frames = len(t)
        interval = dt * 1000
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
        # fps = int(np.ceil(frames / (te - ts))) / 10
        # writer = animation.writers['ffmpeg'](fps=fps, bitrate=1800)
        # # anim.save('directorRigidBodyPendulum.mp4', writer=writer)

        plt.show()

    # #%% reference solution

    if reference_solution:

        def eqm(t,x):
            thetaA = A + 5 * m * (l ** 2) /4
            thetaB = A + m * (l ** 2) /4

            M = np.array([[thetaA, 0.5*m*l*l*cos(x[0]-x[1])], 
                        [0.5*m*l*l*cos(x[0]-x[1]), thetaB]])

            h = np.array([-0.5*m*l*l*(x[3]**2) * sin(x[0]-x[1]) - 1.5*m*l*g*sin(x[0]), \
                        0.5*m*l*l*(x[2]**2) * sin(x[0]-x[1]) - 0.5*m*l*g*sin(x[1])])
            
            dx = np.zeros(4)
            dx[:2] = x[2:]
            dx[2:] = np.linalg.inv(M) @ h
            return dx

        x0 = np.array([alpha0, alpha0 + beta0, alpha_dot0, alpha_dot0 + beta_dot0])
        ref = solve_ivp(eqm,[t0,t1],x0, method='RK45', rtol=1e-8, atol=1e-12) # MATLAB ode45
        x = ref.y
        t = ref.t

        # import matplotlib.pyplot as plt

        plt.plot(t,x[0], '-r')
        plt.plot(t,x[1], '-g')
        plt.plot(sol.t,sol.q[:, 0], 'xr')
        plt.plot(sol.t,sol.q[:, 0] + sol.q[:, 1], 'xg')
        plt.show()
