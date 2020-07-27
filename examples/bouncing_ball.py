import numpy as np
from math import cos, sin, pi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from cardillo.math.algebra import cross3, A_IK_basic_z

from cardillo.model import Model
from cardillo.model.rigid_body import Rigid_body2D
from cardillo.model.frame import Frame
from cardillo.model.force import Force
from cardillo.model.contacts import Sphere_to_plane
from cardillo.solver import Moreau

from scipy.integrate import solve_ivp

class Ball(Rigid_body2D):
    def __init__(self, m, r, q0=None, u0=None):
        theta = 2 / 5 * m * r**2 
        self.r = r
        super().__init__(m, theta, q0=q0, u0=u0)
        
    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.r * np.vstack([np.sin(phi), np.cos(phi), np.zeros(n)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP

if __name__ == "__main__":
    animate = True

    m = 1
    r = 0.1
    g = 9.81
    x0 = -0.3
    y0 = 1
    x_dot0 = 0
    y_dot0 = 0
    phi0 = 0
    phi_dot0 = 0
    r_OS0 = np.array([x0, y0, 0])
    vS0 = np.array([x_dot0, y_dot0, 0])
    q0 = np.array([r_OS0[0], r_OS0[1], phi0])
    u0 = np.array([vS0[0], vS0[1], phi_dot0])
    RB = Ball(m, r, q0, u0)

    e1, e2, e3 = np.eye(3)
    frame = Frame(A_IK=np.vstack( (e3, e1, e2) ).T )
    mu = 0.5
    r_N = 0.2
    e_N = 0.9 * 0
    plane = Sphere_to_plane(frame, RB, r, mu, prox_r_N=r_N, prox_r_T=r_N, e_N=e_N)

    alpha = pi/4
    e1, e2, e3 = A_IK_basic_z(alpha)
    frame1 = Frame(A_IK=np.vstack( (e3, e1, e2) ).T )
    mu = 0.3
    r_N = 0.2
    e_N = 0.9
    plane_left = Sphere_to_plane(frame1, RB, r, mu, prox_r_N=r_N, prox_r_T=r_N, e_N=e_N)

    beta = -pi/4
    e1, e2, e3 = A_IK_basic_z(beta)
    frame2 = Frame(A_IK=np.vstack( (e3, e1, e2) ).T )
    mu = 0
    r_N = 0.2
    e_N = 0.9 * 0
    plane_right = Sphere_to_plane(frame2, RB, r, mu, prox_r_N=r_N, prox_r_T=r_N, e_N=e_N)

    model = Model()
    model.add(RB)
    model.add(Force(lambda t: np.array([0, -g * m, 0]), RB))
    # model.add(plane)
    model.add(plane_left)
    model.add(plane_right)
    model.assemble()

    t0 = 0
    t1 = 10
    dt = 2.5e-2
    solver = Moreau(model, t1, dt)
    sol = solver.solve()
    t = sol.t
    q = sol.q
    u = sol.u
    la_N = sol.la_N

    fig, ax = plt.subplots(3, 1)

    ax[0].set_title('positions')
    ax[0].plot(t, q[:, 0], '-r', label='x(t)')
    ax[0].plot(t, q[:, 1], '-b', label='y(t)')
    ax[0].legend()

    ax[1].set_title('velocities')
    ax[1].plot(t, u[:, 0], '-r', label='x_dot(t)')
    ax[1].plot(t, u[:, 1], '-b', label='y_dot(t)')
    ax[1].legend()

    ax[2].set_title('contact forces')
    ax[2].plot(t, la_N[:, 0], '-r', label='P_N_left(t)')
    ax[2].plot(t, la_N[:, -1], '-b', label='P_N_right(t)')
    ax[2].legend()

    plt.show()

    if animate:

           # animate configurations
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.axis('equal')
        ax.set_xlim(-2 * y0, 2 * y0)
        ax.set_ylim(-2 * y0, 2 * y0)
        

        # prepare data for animation
        frames = len(t)
        target_frames = min(len(t), 200)
        frac = int(frames / target_frames)
        animation_time = 5
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t[::frac]
        q = q[::frac]

        # ax.plot([-2 * y0, 2 * y0], [0, 0], '-k')
        ax.plot([0, y0 * np.cos(alpha)], [0, y0 * np.sin(alpha)], '-k')
        ax.plot([0, - y0 * np.cos(beta)], [0, - y0 * np.sin(beta)], '-k')

        def create(t, q):
            x_S, y_S, _ = RB.r_OP(t, q)
            
            A_IK = RB.A_IK(t, q)
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
            # d3 = A_IK[:, 2] * r

            COM, = ax.plot([x_S], [y_S], 'ok')
            bdry, = ax.plot([], [],  '-k')
            d1_, = ax.plot([x_S, x_S + d1[0]], [y_S, y_S + d1[1]], '-r')
            d2_, = ax.plot([x_S, x_S + d2[0]], [y_S, y_S + d2[1]], '-g')
            return COM, bdry, d1_, d2_

        COM, bdry, d1_, d2_ = create(0, q[0])

        def update(t, q, COM, bdry,  d1_, d2_):

            x_S, y_S, _ = RB.r_OP(t, q)

            x_bdry, y_bdry, _ = RB.boundary(t, q)
            
            A_IK = RB.A_IK(t, q)
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
            # d3 = A_IK[:, 2] * r

            COM.set_data([x_S], [y_S])
            bdry.set_data(x_bdry, y_bdry)

            d1_.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
            d2_.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])

            return COM, bdry, d1_, d2_

        def animate(i):
            update(t[i], q[i], COM, bdry, d1_, d2_)

        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
        plt.show()