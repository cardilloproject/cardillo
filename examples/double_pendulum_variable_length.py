import numpy as np
from math import pi
import logging
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.model import Model
from cardillo.model.pendulum_variable_length import Pendulum_variable_length
from cardillo.model.point_mass import Point_mass
from cardillo.model.bilateral_constraints import Rod
from cardillo.model.force import Force
from cardillo.solver import Euler_backward, Scipy_ivp

def double_pendulum():
    m = 1
    L = 2
    g = 9.81

    Fg = lambda t: np.array([0, -m * g, 0])

    omega = np.pi
    l = lambda t: L -0.5 * np.sin(omega * t)
    l_t = lambda t: -0.5 * omega * np.cos(omega * t)
    l_tt = lambda t: 0.5 * omega**2 * np.sin(omega * t)

    model = Model()

    q0 = np.array([L, 0])
    u0 = np.array([0])
    pendulum = Pendulum_variable_length(m, l, l_t, Fg, q0=q0, u0=u0)
    model.add(pendulum)

    pm = Point_mass(m, 2, np.array([2 * L, 0]), np.array([0, 0]))
    model.add(pm)

    gravity_force = Force(Fg, pm)
    model.add(gravity_force)

    rod = Rod(pendulum, pm, frame_ID1=(1,))
    model.add(rod)

    model.assemble()

    t1 = 1
    dt = 1e-3

    solver = Euler_backward(model, t1, dt)
    # solver = Scipy_ivp(model, t1, dt) # Rod does not implement g_dot_u
    sol = solver.solve()
    t = sol.t
    q = sol.q
    
    # fig, ax = plt.subplots()
    # ax.set_xlim([-2*L, 2*L])
    # ax.set_ylim([-2*L, 2*L])

    # pendulum_line, = ax.plot([], [], '-ok', label='')
    # point_mass_line, = ax.plot([], [], '-ob', label='')

    # def animate(i):
    #     # pendulum
    #     pendulum_x0, pendulum_y0, _ = pendulum.point(0).position(t[i], q[i, pendulum.qDOF])
    #     pendulum_x, pendulum_y, _ = pendulum.point(1).position(t[i], q[i, pendulum.qDOF])
    #     pendulum_line.set_data(([pendulum_x0, pendulum_x], [pendulum_y0, pendulum_y]))

    #     # point mass
    #     rod_x1, rod_y1, _ = rod.point1.position(t[i], q[i, pendulum.qDOF])
    #     point_mass_x, point_mass_y, _ = pm.point().position(t[i], q[i, pm.qDOF])
    #     point_mass_line.set_data(([rod_x1, point_mass_x], [rod_y1 , point_mass_y]))

    #     return pendulum_line, point_mass_line

    # anim = animation.FuncAnimation(fig, animate, frames=len(t))
    # plt.show()
    
    # reference solution
    def eqm(t,x):
        alpha, beta = x[:2]
        alpha_t, beta_t = x[2:]
        sab = np.sin(alpha - beta)
        cab = np.cos(alpha - beta)
        
        F = Fg(t)

        M = np.array([[2 * m * l(t)**2, m * l(t) * L * cab], 
                      [m * l(t) * L * cab, m * L**2]])

        f_gyr = np.array([4 * m * l(t) * l_t(t) * alpha_t + m * L * l(t) * sab * beta_t**2, \
                          sab * m * L * (l_tt(t) - l(t) * alpha_t**2) + 2 * m * L * cab * l_t(t) * alpha_t])

        f_pot = np.array([l(t) * np.cos(alpha) * 2 * F[0] + l(t) * np.sin(alpha) * 2 * F[1], \
                          L * np.cos(beta) * F[0] + L * np.sin(beta) * F[1]])
                          
        h = f_pot - f_gyr
        
        dx = np.zeros(4)
        dx[:2] = x[2:]
        dx[2:] = np.linalg.inv(M) @ h
        return dx

    alpha0 = np.arctan2(pendulum.q0[0], pendulum.q0[1])
    beta0 = np.arctan2( pm.q0[0] - pendulum.q0[0], -pm.q0[1] - pendulum.q0[1])
    
    x0 = np.array([alpha0, beta0, 0, 0])
    ref = solve_ivp(eqm, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-8, atol=1e-12)
    x = ref.y

    alpha = x[0]
    x_pendulum_ref = l(t) * np.sin(alpha)
    y_pendulum_ref = -l(t) * np.cos(alpha)

    beta = x[1]
    x_point_mass_ref = x_pendulum_ref + L * np.sin(beta)
    y_point_mass_ref = y_pendulum_ref - L * np.cos(beta)

    # fig, ax = plt.subplots()
    # ax.set_xlim([-2*L, 2*L])
    # ax.set_ylim([-2*L, 2*L])

    # pendulum_line, = ax.plot([], [], '-ok', label='')
    # point_mass_line, = ax.plot([], [], '-ok', label='')

    # pendulum_line_ref, = ax.plot([], [], '-xr', label='')
    # point_mass_line_ref, = ax.plot([], [], '-xr', label='')

    # def animate(i):
    #     # pendulum
    #     pendulum_x0, pendulum_y0, _ = pendulum.point(0).position(t[i], q[i, pendulum.qDOF])
    #     pendulum_x, pendulum_y, _ = pendulum.point(1).position(t[i], q[i, pendulum.qDOF])
    #     pendulum_line.set_data(([pendulum_x0, pendulum_x], [pendulum_y0, pendulum_y]))

    #     # point mass
    #     rod_x1, rod_y1, _ = rod.point1.position(t[i], q[i, pendulum.qDOF])
    #     point_mass_x, point_mass_y, _ = pm.point().position(t[i], q[i, pm.qDOF])
    #     point_mass_line.set_data(([rod_x1, point_mass_x], [rod_y1 , point_mass_y]))

    #     # reference solution
    #     pendulum_line_ref.set_data(([0, x_pendulum_ref[i]], [0, y_pendulum_ref[i]]))
    #     point_mass_line_ref.set_data(([x_pendulum_ref[i], x_point_mass_ref[i]], [y_pendulum_ref[i] , y_point_mass_ref[i]]))

    #     return pendulum_line, point_mass_line, pendulum_line_ref, point_mass_line_ref

    # anim = animation.FuncAnimation(fig, animate, frames=len(t))
    # plt.show()

    fig, ax = plt.subplots()
    ax.plot(t, x_pendulum_ref, '-k', label='x_pendulum_ref')
    ax.plot(t, y_pendulum_ref, '-b', label='y_pendulum_ref')
    ax.plot(t, x_point_mass_ref, '-g', label='x_point_mass_ref')
    ax.plot(t, y_point_mass_ref, '-r', label='y_point_mass_ref')

    ax.plot(t, q[:, 0], 'xk', label='x_pendulum')
    ax.plot(t, -q[:, 1], 'xb', label='y_pendulum')
    ax.plot(t, q[:, 2], 'xg', label='x_point_mass')
    ax.plot(t, q[:, 3], 'xr', label='y_point_mass')
    
    plt.show()

if __name__ == "__main__":
    double_pendulum()