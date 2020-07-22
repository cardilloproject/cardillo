import numpy as np
from math import cos, sin, pi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from cardillo.math.algebra import cross3, A_IK_basic_z

from cardillo.model import Model
from cardillo.model.rigid_body import Rigid_body2D
from cardillo.model.bilateral_constraints.explicit import Revolute_joint
from cardillo.model.rigid_body import Rigid_body_rel_kinematics
from cardillo.model.frame import Frame
from cardillo.model.force import Force
from cardillo.model.contacts import Sphere_to_plane
from cardillo.solver import Moreau

from scipy.integrate import solve_ivp

class Main_body(Rigid_body2D):
    def __init__(self, q0=None, u0=None):
        m = 14 # kg
        theta = 0.32 # kg m^2
        super().__init__(m, theta, q0=q0, u0=u0)

        self.a = 0.6 # m
        
    def boundary(self, t, q, n=100):
        xi = np.linspace(-self.a/2, self.a/2, n, endpoint=True)
        K_r_SP = np.vstack([xi, np.zeros(n), np.zeros(n)])
        return np.repeat(self.r_OP(t, q), n).reshape(3, n) + self.A_IK(t, q) @ K_r_SP

# unit system: kg, m, s
if __name__ == "__main__":
    animate = True

    g = 9.81

    # main body
    x_mb0 = 0
    y_mb0 = 1
    x_mb_dot0 = 0
    y_mb_dot0 = 0
    phi0 = 0
    phi_dot0 = 0
    A_IK_mb0 = A_IK_basic_z(phi0)

    r_mb0 = np.array([x_mb0, y_mb0, 0])
    v_mb0 = np.array([x_mb_dot0, y_mb_dot0, 0])
    q_mb0 = np.array([r_mb0[0], r_mb0[1], phi0])
    u_mb0 = np.array([v_mb0[0], v_mb0[1], phi_dot0])

    main_body = Main_body(q_mb0, u_mb0)

    # hind leg
    K_r_mbBh = np.array([-0.575 / 2, -0.137675, 0])
    r_OBh = r_mb0 + A_IK_mb0 @ K_r_mbBh
    alpha_h0 = -10 / 180 * np.pi
    alpha_h_dot0 = 0
    hip_h = Revolute_joint(r_OBh, A_IK_mb0, q0=np.array([alpha_h0]), u0=np.array([alpha_h_dot0]))
    A_ITh = A_IK_basic_z(phi0 + alpha_h0)

    K_r_ThBh = np.array([0, 0.0503, 0])
    K_r_ThCh = np.array([0, 0.0503 - 0.2, 0])
    r_Th = r_OBh - A_ITh @ K_r_ThBh
    m_Th = 0.7887
    K_theta_Th = 0.002207 * np.eye(3)
    
    thigh_h = Rigid_body_rel_kinematics(m_Th, K_theta_Th, hip_h, main_body, r_OS0=r_Th, A_IK0=A_ITh)

    # front leg
    K_r_mbBf = np.array([0.575 / 2, -0.137675, 0])
    r_OBf = r_mb0 + A_IK_mb0 @ K_r_mbBf
    alpha_f0 = 10 / 180 * np.pi
    alpha_f_dot0 = 0
    hip_f = Revolute_joint(r_OBf, A_IK_mb0, q0=np.array([alpha_f0]), u0=np.array([alpha_f_dot0]))
    A_ITf = A_IK_basic_z(phi0 + alpha_f0)

    K_r_ThBf = np.array([0, 0.0503, 0])
    K_r_ThCf = np.array([0, 0.0503 - 0.2, 0])
    r_Tf = r_OBf - A_ITf @ K_r_ThBf
    m_Tf = 0.7887
    K_theta_Tf = 0.002207 * np.eye(3)
    
    thigh_f = Rigid_body_rel_kinematics(m_Tf, K_theta_Tf, hip_f, main_body, r_OS0=r_Tf, A_IK0=A_ITf)

    # ground
    e1, e2, e3 = np.eye(3)
    frame = Frame(A_IK=np.vstack( (e3, e1, e2) ).T )
    mu = 0.25
    r_N = 0.2
    e_N = 0.0
    contact_mb = Sphere_to_plane(frame, main_body, 0, mu, prox_r_N=r_N, prox_r_T=r_N, e_N=e_N)
    contact_h = Sphere_to_plane(frame, thigh_h, 0, mu, prox_r_N=r_N, prox_r_T=r_N, e_N=e_N, K_r_SP=K_r_ThCh)
    contact_f = Sphere_to_plane(frame, thigh_f, 0, mu, prox_r_N=r_N, prox_r_T=r_N, e_N=e_N, K_r_SP=K_r_ThCf)
    # contact_h = Sphere_to_plane(frame, thigh_h, 0, mu, prox_r_N=r_N, prox_r_T=r_N, e_N=e_N)
    # contact_f = Sphere_to_plane(frame, thigh_f, 0, mu, prox_r_N=r_N, prox_r_T=r_N, e_N=e_N)

    model = Model()
    model.add(main_body)
    model.add(Force(lambda t: np.array([0, -g * main_body.m, 0]), main_body))

    model.add(thigh_f)
    model.add(Force(lambda t: np.array([0, -g * m_Tf, 0]), thigh_f))

    model.add(thigh_h)
    model.add(Force(lambda t: np.array([0, -g * m_Th, 0]), thigh_h))

    model.add(contact_mb)
    model.add(contact_h)
    model.add(contact_f)
    model.assemble()

    t0 = 0
    t1 = 1.5
    dt = 1e-3
    solver = Moreau(model, t1, dt)
    sol = solver.solve()
    t = sol.t
    q = sol.q
    u = sol.u
    la_N = sol.la_N

    plt.plot(t, q[:, thigh_h.qDOF[-1]], '-r', label='alpha_h')
    plt.plot(t, q[:, thigh_f.qDOF[-1]], '-g', label='alpha_f')
    plt.legend()
    plt.show()
    # exit()

    if animate:
        # animate configurations
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.axis('equal')
        scale = 1.2
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)

        # prepare data for animation
        frames = len(t)
        target_frames = min(len(t), 200)
        frac = int(frames / target_frames)
        animation_time = 5
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t[::frac]
        q = q[::frac]

        ground, = ax.plot([-100, 100], [0, 0], '-k')

        # COM, = ax.plot([], [], '-ok')
        bdry_mb, = ax.plot([], [],  '-ok')
        d1_mb, = ax.plot([], [], '-r')
        d2_mb, = ax.plot([], [], '-g')

        # COM_Th, = ax.plot([], [], 'ok')
        bdry_Th, = ax.plot([], [],  '-ok')
        bdry_Tf, = ax.plot([], [],  '-ok')

        # def update(t, q, COM, bdry, d1_, d2_):
        def update(t, q, *args):
            bdry_mb, d1_mb, d2_mb, bdry_Th, bdry_Tf = args

            # main body
            x_S, y_S, _ = main_body.r_OP(t, q)
            # x_bdry, y_bdry, _ = main_body.boundary(t, q)
            A_IK = main_body.A_IK(t, q)
            d1 = A_IK[:, 0] * main_body.a / 4
            d2 = A_IK[:, 1] * main_body.a / 4
            
            d1_mb.set_data([x_S, x_S + d1[0]], [y_S, y_S + d1[1]])
            d2_mb.set_data([x_S, x_S + d2[0]], [y_S, y_S + d2[1]])

            x1, y1, _ = main_body.r_OP(t, q, K_r_SP=K_r_mbBh)
            x2, y2, _ = main_body.r_OP(t, q, K_r_SP=K_r_mbBf)
            bdry_mb.set_data([x1, x_S, x2], [y1, y_S, y2])

            # hind leg
            x1, y1, _ = thigh_h.r_OP(t, q[thigh_h.qDOF], K_r_SP=K_r_ThBh)
            x_S, y_S, _ = thigh_h.r_OP(t, q[thigh_h.qDOF])
            x2, y2, _ = thigh_h.r_OP(t, q[thigh_h.qDOF], K_r_SP=K_r_ThCh)
            x_C, y_C, _ = contact_h.r_OP(t, q[contact_h.qDOF])
            bdry_Th.set_data([x1, x_S, x2, x_C], [y1, y_S, y2, y_C])

            # front leg
            x1, y1, _ = thigh_f.r_OP(t, q[thigh_f.qDOF], K_r_SP=K_r_ThBf)
            x_S, y_S, _ = thigh_f.r_OP(t, q[thigh_f.qDOF])
            x2, y2, _ = thigh_f.r_OP(t, q[thigh_f.qDOF], K_r_SP=K_r_ThCf)
            x_C, y_C, _ = contact_f.r_OP(t, q[contact_f.qDOF])
            bdry_Tf.set_data([x1, x_S, x2, x_C], [y1, y_S, y2, y_C])

            return bdry_mb, d1_mb, d2_mb, bdry_Th, bdry_Tf

        def animate(i):
            update(t[i], q[i], bdry_mb, d1_mb, d2_mb, bdry_Th, bdry_Tf)

        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
        plt.show()