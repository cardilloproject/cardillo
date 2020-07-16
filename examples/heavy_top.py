import numpy as np 
from math import sin, cos, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from cardillo.math.algebra import inverse3D, A_IK_basic_x, A_IK_basic_y, A_IK_basic_z, cross3
from scipy.integrate import solve_ivp
from cardillo.model.rigid_body import Rigid_body_euler

class Heavy_top():
    def __init__(self, m, r, L):
        self.m = m
        self.r = r
        self.L = L
        self.A = 1 / 2 * m * r**2
        self.B = 1 / 4 * m * r**2

    def A_IK(self, t, q):
        A_IB = A_IK_basic_z(q[0])
        A_BC = A_IK_basic_y(-q[1])
        A_CK = A_IK_basic_x(q[2])
        return A_IB @ A_BC @ A_CK

    def r_OP(self, t, q, K_r_SP=np.zeros(3)):
        A_IK = self.A_IK(t, q) 
        r_OS = A_IK @ np.array([L, 0, 0])
        return r_OS + A_IK @ K_r_SP

    def eqm(self, t, x):
        dx = np.zeros(6)
        beta = x[1]
        gamma = x[2]
        omega_x = x[3]
        omega_y = x[4]
        omega_z = x[5]

        g = 9.81

        m = self.m
        L = self.L
        A = self.A
        B = self.B

        Q = np.array([[sin(beta), 0, 1], \
                        [cos(beta) * sin(gamma), -cos(gamma), 0], \
                        [cos(beta) * cos(gamma),  sin(gamma), 0]])

        M = np.diag(np.array([A, B + m * L**2, B + m * L**2]))

        h = np.array([0, \
                (B + m * L**2 - A) * omega_x * omega_z + m * g * L * cos(beta) * cos(gamma), \
               -(B + m * L**2 - A) * omega_x * omega_y - m * g * L * cos(beta) * sin(gamma)])

        dx[:3] = inverse3D(Q) @ x[3:]
        dx[3:] = inverse3D(M) @ h
        return dx

    def boundary(self, t, q, n=100):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        K_r_SP = self.r * np.vstack([np.zeros(n), np.sin(phi), np.cos(phi)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP

if __name__ == "__main__":

    animate = True
    plot_graphs = True

    m = 1
    L = 0.5
    g = 9.81
    r = 0.1

    heavy_top = Heavy_top(m, r, L)

    Omega = 1
    Omega_y = 0 #0.1 * Omega_x
    
    K_r_S0 = np.array([L, 0, 0])

    t0 = 0
    t1 = 5
    alpha0 = 0
    beta0 = pi / 4
    gamma0 = 0

    omega_x0 = Omega
    omega_y0 = 0
    omega_z0 = 0

    # A_IB0 = A_IK_basic_y([alpha0])
    # A_BC0 = A_IK_basic_z([beta0])
    # A_CK0 = A_IK_basic_x([gamma0])

    # r_OS0 = A_IB0 @ A_BC0 @ A_CK0 @ np.array([L, 0, 0])
    # K_Omega0 = np.array([omega_x0, omega_y0, omega_z0])
    # v_S0 = cross3(K_Omega0, r_OS0)
    
    # if rigid_body == 'Euler':
    #     p0 = np.array([0, beta0, 0])
    # elif rigid_body == 'Quaternion':
    #     p0 = axis_angle2quat(np.array([1, 0, 0]), beta0)
    # elif rigid_body == 'Director':
    #     R0 = A_IK_basic_x(beta0)
    #     p0 = np.concatenate((R0[:, 0], R0[:, 1], R0[:, 2]))

    # gamma_dot = 4 * g * sin(beta0) / ((6 - 5 * rho * sin(beta0)) * rho * r * cos(beta0))
    # gamma_dot = sqrt(gamma_dot)
    # alpha_dot = -rho * gamma_dot

    # v_S0 = np.array([-R * alpha_dot + r * alpha_dot * sin(beta0), \
    #                  0, \
    #                  0])
    # omega0 = np.array([0, alpha_dot * sin(beta0) + gamma_dot, alpha_dot * cos(beta0)])

    # if rigid_body == 'Euler' or rigid_body == 'Quaternion':
    #     u0 = np.concatenate((v_S0, omega0))
    # elif rigid_body == 'Director':
    #     omega0_tilde = R0 @ ax2skew(omega0) 
    #     u0 = np.concatenate((v_S0, omega0_tilde[:, 0], omega0_tilde[:, 1], omega0_tilde[:, 2]))
    # #------------

    # q0 = np.concatenate((r_S0, p0))
    

    
    # reference solution



    # reference solution
    dt = 0.001
    x0 = np.array([alpha0, beta0, gamma0, omega_x0, omega_y0, omega_z0])
    ref = solve_ivp(heavy_top.eqm, [t0, t1], x0, method='RK45', t_eval=np.arange(t0,t1 + dt,dt), rtol=1e-8, atol=1e-12) # MATLAB ode45
    x_ref = ref.y
    t = ref.t

    q = ref.y[:3].T

    if plot_graphs:
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(t, q[:, 0], '-r')
        ax[0].set(ylabel='alpha')
        ax[1].plot(t, q[:, 1], '-r')
        ax[1].set(ylabel='beta')
        ax[2].plot(t, q[:, 2], '-r')
        ax[2].set(ylabel='gamma')
        plt.xlabel('time')
        # ax.plot(t, 2 * np.arcsin(q[:, 3]), '-x')
        plt.show()

    if animate:
        # animate configurations
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        scale = L
        ax.set_xlim3d(left=-scale, right=scale)
        ax.set_ylim3d(bottom=-scale, top=scale)
        ax.set_zlim3d(bottom=-2*scale, top=2*scale)

        # from collections import deque
        # x_trace = deque([])
        # y_trace = deque([])
        # z_trace = deque([])

        # prepare data for animation
        frames = len(t)
        target_frames = 100
        frac = int(frames / target_frames)
        animation_time = 5
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t[::frac]
        q = q[::frac]

        def create(t, q):
            x_0, y_0, z_0 = np.zeros(3)
            x_S, y_S, z_S = heavy_top.r_OP(t, q)
            
            A_IK = heavy_top.A_IK(t, q)
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
            d3 = A_IK[:, 2] * r

            COM, = ax.plot([x_0, x_S], [y_0, y_S], [z_0, z_S], '-ok')
            bdry, = ax.plot([], [], [], '-k')
            trace, = ax.plot([], [], [], '--k')
            d1_, = ax.plot([x_S, x_S + d1[0]], [y_S, y_S + d1[1]], [z_S, z_S + d1[2]], '-r')
            d2_, = ax.plot([x_S, x_S + d2[0]], [y_S, y_S + d2[1]], [z_S, z_S + d2[2]], '-g')
            d3_, = ax.plot([x_S, x_S + d3[0]], [y_S, y_S + d3[1]], [z_S, z_S + d3[2]], '-b')
        
            return COM, bdry, d1_, d2_, d3_
            # return COM, bdry, trace, d1_, d2_, d3_

        # COM, bdry, trace, d1_, d2_, d3_ = create(0, q[0])

        COM, bdry, d1_, d2_, d3_ = create(0, q[0])

        def update(t, q, COM, bdry, d1_, d2_, d3_):

        # def update(t, q, COM, bdry, trace, d1_, d2_, d3_):
            global x_trace, y_trace, z_trace
            # if t == t0:
            #     x_trace = deque([])
            #     y_trace = deque([])
            #     z_trace = deque([])

            x_0, y_0, z_0 = np.zeros(3)
            x_S, y_S, z_S = heavy_top.r_OP(t, q)

            x_bdry, y_bdry, z_bdry = heavy_top.boundary(t, q)

            # x_t, y_t, z_t = RD.r_OP(t, q) + RC.r_SA(t, q)

            # x_trace.append(x_t)
            # y_trace.append(y_t)
            # z_trace.append(z_t)
            
            A_IK = heavy_top.A_IK(t, q)
            d1 = A_IK[:, 0] * r
            d2 = A_IK[:, 1] * r
            d3 = A_IK[:, 2] * r

            COM.set_data([x_0, x_S], [y_0, y_S])
            COM.set_3d_properties([z_0, z_S])

            bdry.set_data(x_bdry, y_bdry)
            bdry.set_3d_properties(z_bdry)

            # if len(x_trace) > 500:
            #     x_trace.popleft()
            #     y_trace.popleft()
            #     z_trace.popleft()
            # trace.set_data(x_trace, y_trace)
            # trace.set_3d_properties(z_trace)

            d1_.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
            d1_.set_3d_properties([z_S, z_S + d1[2]])

            d2_.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])
            d2_.set_3d_properties([z_S, z_S + d2[2]])

            d3_.set_data([x_S, x_S + d3[0]], [y_S, y_S + d3[1]])
            d3_.set_3d_properties([z_S, z_S + d3[2]])

            return COM, bdry, d1_, d2_, d3_
            # return COM, bdry, trace, d1_, d2_, d3_

        def animate(i):
            # update(t[i], q[i], COM, bdry, trace, d1_, d2_, d3_)
            update(t[i], q[i], COM, bdry, d1_, d2_, d3_)

        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
        plt.show()

        # x_trace = []
        # y_trace = []
        # z_trace = []
        # for i, (t_i, q_i) in enumerate(zip(t, q)): 
        #     x_t, y_t, z_t = RD.r_OP(t_i, q_i) + RC.r_SA(t_i, q_i)
        #     x_trace.append(x_t)
        #     y_trace.append(y_t)
        #     z_trace.append(z_t)

        # fig, ax = plt.subplots(2, 1)
        # ax[0].plot(x_trace, y_trace, '-k')
        # ax[1].plot(t, z_trace, '-b')
        # ax[0].axis('equal')
        # plt.show()
