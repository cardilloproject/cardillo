import numpy as np

from math import sin, cos, sqrt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from cardillo.model import Model
from cardillo.model.rolling_disc import Rolling_disc
from cardillo.model.rigid_body import Rigid_disc, Rigid_disc_Lesaux
from cardillo.model.rolling_disc import Rolling_condition_I_frame, Rolling_condition_R_frame, Rolling_condition_I_frame_g_gamma
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints import Rod
from cardillo.model.force import Force
from cardillo.solver import Euler_backward
from cardillo.math.algebra import axis_angle2quat

def DMS():
    m = 1
    r = 1
    g = 9.81

    # q0 = np.zeros(5)
    # u0 = np.array([0, 0, 0])
    alpha0 = 0
    beta0 = np.pi / 4
    # q0 = np.array([0, 0, alpha0, beta0, 0])
    # u0 = np.array([5, 0, 0])

    #-----------
    # Initial condition for circular trajectory
    R = 5 # radius of trajectory
    rho = r/R
    gamma_dot = 4 * g * sin(beta0) / ((6 - 5 * rho * sin(beta0)) * rho * r * cos(beta0))
    gamma_dot = sqrt(gamma_dot)
    alpha_dot = -rho * gamma_dot
    
    u0 =   np.array([alpha_dot, 0, gamma_dot])
    q0 = np.array([0, R, alpha0, beta0, 0])
    #------------

    RD = Rolling_disc(m, r, 9.81, q0=q0, u0=u0)

    model = Model()
    model.add(RD)
    model.assemble()
    t0 = 0
    t1 = 10
    dt = 1e-2
    t_span = t0, t1
    solver = Euler_backward(model, t_span=t_span, dt=dt, newton_max_iter=50, numerical_jacobian=False, debug=False)
    t, q, u, la = solver.solve()

    # animate configurations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    scale = 10* r
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=0, top=2*scale)

    x_trace = []
    y_trace = []
    z_trace = []

    def init(t, q):
        x_S, y_S, z_S = RD.r_OS(t, q)
        
        A_IK = RD.A_IK(t, q)
        d1 = A_IK[:, 0]
        d2 = A_IK[:, 1]
        d3 = A_IK[:, 2]

        COM, = ax.plot([x_S], [y_S], [z_S], 'ok')
        bdry, = ax.plot([], [], [], '-k')
        trace, = ax.plot([], [], [], '-r')
        d1_, = ax.plot([x_S, x_S + d1[0]], [y_S, y_S + d1[1]], [z_S, z_S + d1[2]], '-r')
        d2_, = ax.plot([x_S, x_S + d2[0]], [y_S, y_S + d2[1]], [z_S, z_S + d2[2]], '-g')
        d3_, = ax.plot([x_S, x_S + d3[0]], [y_S, y_S + d3[1]], [z_S, z_S + d3[2]], '-b')
       
        return COM, bdry, trace, d1_, d2_, d3_

    def update(t, q, COM, bdry, trace, d1_, d2_, d3_):
        x_S, y_S, z_S = RD.r_OS(t, q)

        x_bdry, y_bdry, z_bdry = RD.boundary(t, q)

        x_t, y_t, z_t = RD.r_OA(t, q)

        x_trace.append(x_t)
        y_trace.append(y_t)
        z_trace.append(z_t)
        
        A_IK = RD.A_IK(t, q)
        d1 = A_IK[:, 0]
        d2 = A_IK[:, 1]
        d3 = A_IK[:, 2]


        COM.set_data([x_S], [y_S])
        COM.set_3d_properties([z_S])

        bdry.set_data(x_bdry, y_bdry)
        bdry.set_3d_properties(z_bdry)

        trace.set_data(x_trace, y_trace)
        trace.set_3d_properties(z_trace)

        d1_.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
        d1_.set_3d_properties([z_S, z_S + d1[2]])

        d2_.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])
        d2_.set_3d_properties([z_S, z_S + d2[2]])

        d3_.set_data([x_S, x_S + d3[0]], [y_S, y_S + d3[1]])
        d3_.set_3d_properties([z_S, z_S + d3[2]])

        return COM, bdry, trace, d1_, d2_, d3_


    COM, bdry, trace, d1_, d2_, d3_ = init(0, q[0])

    def animate(i):
        update(t[i], q[i], COM, bdry, trace, d1_, d2_, d3_)

    frames = len(t)
    interval = dt * 100
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)

    plt.show()

    plt.plot(x_trace, y_trace)
    plt.show()

def rolling_disc_velocity_constraints():
    g = 9.81
    m = 0.3048
    r = 3.75e-2
    R = 0.5 # radius of trajectory
    beta0 = 5 * np.pi / 180

    #-----------
    # Initial condition for circular trajectory
    rho = r / R

    r_S0 = np.array([0, R - r * sin(beta0), r * cos(beta0)])
    p0 = axis_angle2quat(np.array([1, 0, 0]), beta0)

    gamma_dot = 4 * g * sin(beta0) / ((6 - 5 * rho * sin(beta0)) * rho * r * cos(beta0))
    gamma_dot = sqrt(gamma_dot)
    alpha_dot = -rho * gamma_dot

    v_S0 = np.array([-R * alpha_dot + r * alpha_dot * sin(beta0), \
                     0, \
                     0])
    omega0 = np.array([0, alpha_dot * sin(beta0) + gamma_dot, alpha_dot * cos(beta0)])
    u0 = np.concatenate((v_S0, omega0))
    #------------

    # r_S0 = np.array([0, 0, r])
    # p0 = np.array([1, 0, 0, 0])

    # gamma_dot0 = 5
    # v_S0 = np.array([gamma_dot0 * r, 0, 0])
    # omega0 = np.array([0, gamma_dot0, 0])

    q0 = np.concatenate((r_S0, p0))
    u0 = np.concatenate((v_S0, omega0))

    # RD = Rigid_disc(m, r, q0=q0, u0=u0)
    RD = Rigid_disc_Lesaux(m, r, q0=q0, u0=u0)
    # RC = Rolling_condition_I_frame(RD)
    # RC = Rolling_condition_R_frame(RD)
    RC = Rolling_condition_I_frame_g_gamma(RD)
    f_g = Force(lambda t: np.array([0, 0, -m * g]), RD)

    model = Model()
    model.add(RD)
    model.add(RC)
    model.add(f_g)
    model.assemble()

    # m = 0.3048;
    # r = 3.75e-2;
    # thickness = 2.2e-3;
    # theta = diag([1.0716e-4,1.0716e-4,2.1433e-4]);
    # alpha = 0; %pi/20;
    # beta = 0.0873;%deg2rad(5);
    # R = 0.5;
    # gravity = 9.81;

    t0 = 0
    t1 = 2
    dt = 2.5e-3
    t_span = t0, t1
    solver = Euler_backward(model, t_span=t_span, dt=dt, numerical_jacobian=True, debug=False)
    t, q, u, la_g, la_gamma = solver.solve()

    # # animate configurations
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    # ax.set_zlabel('z [m]')
    # scale = 2* R
    # ax.set_xlim3d(left=-scale, right=scale)
    # ax.set_ylim3d(bottom=-scale, top=scale)
    # ax.set_zlim3d(bottom=0, top=2*scale)

    # x_trace = []
    # y_trace = []
    # z_trace = []

    # def init(t, q):
    #     x_S, y_S, z_S = RD.r_OP(t, q)
        
    #     A_IK = RD.A_IK(t, q)
    #     d1 = A_IK[:, 0]
    #     d2 = A_IK[:, 1]
    #     d3 = A_IK[:, 2]

    #     COM, = ax.plot([x_S], [y_S], [z_S], 'ok')
    #     bdry, = ax.plot([], [], [], '-k')
    #     trace, = ax.plot([], [], [], '-r')
    #     d1_, = ax.plot([x_S, x_S + d1[0]], [y_S, y_S + d1[1]], [z_S, z_S + d1[2]], '-r')
    #     d2_, = ax.plot([x_S, x_S + d2[0]], [y_S, y_S + d2[1]], [z_S, z_S + d2[2]], '-g')
    #     d3_, = ax.plot([x_S, x_S + d3[0]], [y_S, y_S + d3[1]], [z_S, z_S + d3[2]], '-b')
       
    #     return COM, bdry, trace, d1_, d2_, d3_

    # def update(t, q, COM, bdry, trace, d1_, d2_, d3_):
    #     x_S, y_S, z_S = RD.r_OP(t, q)

    #     x_bdry, y_bdry, z_bdry = RD.boundary(t, q)

    #     x_t, y_t, z_t = RD.r_OP(t, q) + RC.r_SA(t, q)

    #     x_trace.append(x_t)
    #     y_trace.append(y_t)
    #     z_trace.append(z_t)
        
    #     A_IK = RD.A_IK(t, q)
    #     d1 = A_IK[:, 0]
    #     d2 = A_IK[:, 1]
    #     d3 = A_IK[:, 2]


    #     COM.set_data([x_S], [y_S])
    #     COM.set_3d_properties([z_S])

    #     bdry.set_data(x_bdry, y_bdry)
    #     bdry.set_3d_properties(z_bdry)

    #     trace.set_data(x_trace, y_trace)
    #     trace.set_3d_properties(z_trace)

    #     d1_.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
    #     d1_.set_3d_properties([z_S, z_S + d1[2]])

    #     d2_.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])
    #     d2_.set_3d_properties([z_S, z_S + d2[2]])

    #     d3_.set_data([x_S, x_S + d3[0]], [y_S, y_S + d3[1]])
    #     d3_.set_3d_properties([z_S, z_S + d3[2]])

    #     return COM, bdry, trace, d1_, d2_, d3_


    # COM, bdry, trace, d1_, d2_, d3_ = init(0, q[0])

    # def animate(i):
    #     update(t[i], q[i], COM, bdry, trace, d1_, d2_, d3_)

    # frames = len(t)
    # interval = dt * 100
    # anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)

    # plt.show()

    x_trace = []
    y_trace = []
    z_trace = []
    for i, (t_i, q_i) in enumerate(zip(t, q)): 
        x_t, y_t, z_t = RD.r_OP(t_i, q_i) + RC.r_SA(t_i, q_i)
        x_trace.append(x_t)
        y_trace.append(y_t)
        z_trace.append(z_t)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(x_trace, y_trace, '-k')
    ax[1].plot(t, z_trace, '-b')
    ax[0].axis('equal')
    # ax[1].axis('equal')
    plt.show()

if __name__ == "__main__":
    # DMS()
    rolling_disc_velocity_constraints()    