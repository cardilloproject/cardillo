import numpy as np 
from math import sin, cos, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from cardillo.math.algebra import inverse2D, A_IK_basic_x, A_IK_basic_y, A_IK_basic_z, cross3, axis_angle2quat, ax2skew
from scipy.integrate import solve_ivp

from cardillo.model import Model
from cardillo.model.point_mass import Point_mass
from cardillo.model.bilateral_constraints.implicit import Rod
from cardillo.model.frame import Frame
from cardillo.model.force import Force
from cardillo.solver import Scipy_ivp, Generalized_alpha_1, Moreau_sym, Euler_backward, Moreau

class Mathematical_pendulum3D_excited():
    def __init__(self, m, L, e, e_t, e_tt):
        self.m = m
        self.L = L
        self.e = e
        self.e_t = e_t
        self.e_tt = e_tt

    def r_OP(self, t, q):
        L = self.L
        return A_IK_basic_y(q[0]) @ np.array([L * cos(q[1]), L * sin(q[1]) + self.e(t), 0])

    def v_P(self, t, q, u):
        L = self.L
        B_v_S = np.array([-L * sin(q[1]) * u[1], \
                           L * cos(q[1]) * u[1] + self.e_t(t), \
                           L * cos(q[1]) * u[0]])
        return  A_IK_basic_y(q[0]) @ B_v_S

    def eqm(self, t, x):
        dx = np.zeros(4)
        alpha = x[0]
        beta = x[1]
        alpha_dot = x[2]
        beta_dot = x[3]

        g = 9.81
        mL = self.m * self.L
        mL2 = self.m * self.L**2
        
        M = np.diag(np.array([mL2 * cos(beta)**2, mL2]))

        h = np.array([2 * mL2 * cos(beta) * sin(beta) * alpha_dot * beta_dot, \
            -mL * cos(beta) * self.e_tt(t) -mL2 * cos(beta) * sin(beta) * alpha_dot**2 - mL * cos(beta)* g])

        dx[:2] = x[2:]
        dx[2:] = inverse2D(M) @ h
        return dx

def comparison_mathematical_pendulum3D(t1=1, plot_graphs=True, animate=True, animate_ref=False):
    t0 = 0
    dt = 1e-4

    m = 0.1
    L = 0.2
    g = 9.81
    dim = 3

    omega = 10
    A = L / 10

    e = lambda t: A * np.sin(omega * t)
    e_t = lambda t: A * omega * np.cos(omega * t)
    e_tt = lambda t: -A * omega * omega * np.sin(omega * t)

    # e = lambda t: A * t
    # e_t = lambda t: A 
    # e_tt = lambda t: 0

    # e = lambda t: 0
    # e_t = lambda t: 0 
    # e_tt = lambda t: 0

    r_OP = lambda t: np.array([0, e(t),    0]) 
    v_P =  lambda t: np.array([0, e_t(t),  0]) 
    a_P =  lambda t: np.array([0, e_tt(t), 0]) 

    # reference solution
    pendulum = Mathematical_pendulum3D_excited(m, L, e, e_t, e_tt)

    alpha0 = 1
    alpha_dot0 = 0
    beta0 = 0
    beta_dot0 = 0

    if dim == 2:
        alpha0 = 0
        alpha_dot0 = 0
    

    x0 = np.array([alpha0, beta0, alpha_dot0, beta_dot0])
    ref = solve_ivp(pendulum.eqm, [t0, t1], x0, method='RK45', t_eval=np.arange(t0,t1 + dt,dt), rtol=1e-8, atol=1e-12)
    t_ref = ref.t
    q_ref = ref.y[:2].T

    # solutions with cardillo models
    r_OS0 = pendulum.r_OP(t0, np.array([alpha0, beta0]))
    v_S0 = pendulum.v_P(t0, np.array([alpha0, beta0]), np.array([alpha_dot0, beta_dot0]))

    PM = Point_mass(m, q0=r_OS0[:dim], u0=v_S0[:dim], dim=dim)

    origin = Frame(r_OP, r_OP_t=v_P, r_OP_tt=a_P)
    joint = Rod(origin, PM)
    model = Model()
    model.add(origin)
    model.add(PM)
    model.add(joint)
    model.add(Force(lambda t: np.array([0, -g * m, 0]), PM))

    model.assemble()

    # solver = Euler_backward(model, t1, dt, numerical_jacobian=False, debug=False)
    # solver = Moreau_sym(model, t1, dt)
    # solver = Moreau(model, t1, dt)
    # solver = Generalized_alpha_1(model, t1, dt, rho_inf=1, numerical_jacobian=False, debug=False)
    # solver = Scipy_ivp(model, t1, dt)
    solver = Scipy_ivp(model, t1, dt, rtol = 1e-6, atol=1.0e-7)

    sol = solver.solve()
    t = sol.t
    q = sol.q

    if plot_graphs:     
        x_ref_ = []
        y_ref_ = []
        for i, ti in enumerate(t_ref):
            x_ref_.append(pendulum.r_OP(ti, q_ref[i])[0])
            y_ref_.append(pendulum.r_OP(ti, q_ref[i])[1])

        x_ = []
        y_ = []
        for i, ti in enumerate(t):
            x_.append(PM.r_OP(ti, q[i])[0])
            y_.append(PM.r_OP(ti, q[i])[1])

        plt.plot(x_ref_, y_ref_, '-r')
        plt.plot(x_, y_, '--gx')
        scale_ = 1.2 * L
        plt.xlim(-scale_, scale_)
        plt.ylim(-scale_, scale_)
        plt.axis('equal')
        plt.xlabel('x_S [m]')
        plt.ylabel('y_S [m]')
        plt.show()

    if animate:
        if animate_ref:
            t = t_ref
            q = q_ref
            PM = pendulum

        # animate configurations
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel('z [m]')
        ax.set_ylabel('x [m]')
        ax.set_zlabel('y [m]')
        scale = L
        ax.set_xlim3d(left=-scale, right=scale)
        ax.set_ylim3d(bottom=-scale, top=scale)
        ax.set_zlim3d(bottom=-scale, top=scale)

        # prepare data for animation
        frames = len(t)
        target_frames = min(frames, 100)
        frac = int(frames / target_frames)
        animation_time = t1 - t0
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t[::frac]
        q = q[::frac]

        def create(t, q):
            x_0, y_0, z_0 = r_OP(t)
            x_S, y_S, z_S = PM.r_OP(t, q)

            COM, = ax.plot([z_0, z_S], [x_0, x_S], [y_0, y_S], '-ok')
                    
            return COM
        COM = create(0, q[0])

        def update(t, q, COM):
            x_0, y_0, z_0 = r_OP(t)
            x_S, y_S, z_S = PM.r_OP(t, q)

            COM.set_data([z_0, z_S], [x_0, x_S])
            COM.set_3d_properties([y_0, y_S])

            return COM 

        def animate(i):
            update(t[i], q[i], COM)

        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
        plt.show()

if __name__ == "__main__":
    comparison_mathematical_pendulum3D(t1=5, animate=True, animate_ref=False)
