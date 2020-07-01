import numpy as np

from math import sin, cos, sqrt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from cardillo.model import Model
from cardillo.model.rolling_disc import Rolling_disc
from cardillo.solver import Euler_backward

if __name__ == "__main__":
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