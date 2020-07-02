import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from cardillo.math.algebra import axis_angle2quat
from cardillo.model import Model
from cardillo.model.rigid_body import Rigid_body_quaternion
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints import Spherical_joint
from cardillo.model.force import Force
from cardillo.solver import Euler_backward

class Rigid_cylinder(Rigid_body_quaternion):
    def __init__(self, m, r, l, q0=None, u0=None):
        A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
        C = 1 / 2 * m * r**2
        K_theta_S = np.diag(np.array([A, A, C]))

        super().__init__(m, K_theta_S, q0=q0, u0=u0)

if __name__ == "__main__":
    m = 10
    r = 1
    l = 0.2

    r0 = np.array([0, r, 0])
    # p0 = np.array([1, 0, 0, 0])
    p0 = axis_angle2quat(np.array([1,0,0]), np.pi/2)
    q0 = np.concatenate((r0, p0))

    r0_t = np.array([0, 0, 0])
    omega = np.array([0, 0, 50])
    u0 = np.concatenate((r0_t, omega))

    RB = Rigid_cylinder(m, r, l,  q0, u0)
    # RB = Rigid_body_quaternion(m, 2/5*m*r**2*np.eye(3), q0, u0)
    frame = Frame()

    model = Model()
    model.add(RB)
    model.add(Force(lambda t: np.array([0, 0, -9.81 * m]), RB))
    model.add(frame)
    K_r_SP = np.array([0, 0, r])
    model.add( Spherical_joint(frame, RB, np.zeros(3)) )
    model.assemble()

    t0 = 0
    t1 = 5
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
    scale = 2 * r
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=-scale, top=scale)

    def init(t, q):
        x_0, y_0, z_0 = frame.r_OP(t)
        x_S, y_S, z_S = RB.r_OP(t, q)
        x_P, y_P, z_P = RB.r_OP(t, q, K_r_SP=K_r_SP)
        
        A_IK = RB.A_IK(t, q)
        d1 = A_IK[:, 0]
        d2 = A_IK[:, 1]
        d3 = A_IK[:, 2]

        COM, = ax.plot([x_0, x_P, x_S], [y_0, y_P, y_S], [z_0, z_P, z_S], '-ok')
        d1_, = ax.plot([x_S, x_S + d1[0]], [y_S, y_S + d1[1]], [z_S, z_S + d1[2]], '-r')
        d2_, = ax.plot([x_S, x_S + d2[0]], [y_S, y_S + d2[1]], [z_S, z_S + d2[2]], '-g')
        d3_, = ax.plot([x_S, x_S + d3[0]], [y_S, y_S + d3[1]], [z_S, z_S + d3[2]], '-b')

        return COM, d1_, d2_, d3_

    def update(t, q, COM, d1_, d2_, d3_):
        x_0, y_0, z_0 = frame.r_OP(t)
        x_S, y_S, z_S = RB.r_OP(t, q)
        x_P, y_P, z_P = RB.r_OP(t, q, K_r_SP=K_r_SP)

        A_IK = RB.A_IK(t, q)
        d1 = A_IK[:, 0]
        d2 = A_IK[:, 1]
        d3 = A_IK[:, 2]

        COM.set_data([x_0, x_P, x_S], [y_0, y_P, y_S])
        COM.set_3d_properties([z_0, z_P, z_S])

        d1_.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
        d1_.set_3d_properties([z_S, z_S + d1[2]])

        d2_.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])
        d2_.set_3d_properties([z_S, z_S + d2[2]])

        d3_.set_data([x_S, x_S + d3[0]], [y_S, y_S + d3[1]])
        d3_.set_3d_properties([z_S, z_S + d3[2]])

        return COM, d1_, d2_, d3_


    COM, d1_, d2_, d3_ = init(0, q0)

    def animate(i):
        update(t[i], q[i], COM, d1_, d2_, d3_)
    
    # compute naimation interval according to te - ts = frames * interval / 1000
    frames = len(t)
    interval = dt * 1000
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
    # fps = int(np.ceil(frames / (te - ts))) / 10
    # writer = animation.writers['ffmpeg'](fps=fps, bitrate=1800)
    # # anim.save('directorRigidBodyPendulum.mp4', writer=writer)

    plt.show()
