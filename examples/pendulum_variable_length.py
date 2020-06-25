import numpy as np
from math import pi
import logging
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.model import Model
from cardillo.model.pendulum_variable_length import Pendulum_variable_length
from cardillo.solver import Euler_forward, Euler_backward

m = 1
L = 2
g = 9.81

F = lambda t: np.array([0, -m * g])

# l = lambda t: L + np.sin(t)
# l_t = lambda t: np.cos(t)
# l_tt = lambda t: -np.sin(t)

l = lambda t: L
l_t = lambda t: 0
l_tt = lambda t: 0

model = Model()

q0 = np.array([L, 0])
u0 = np.array([0])
pendulum = Pendulum_variable_length(m, l, l_t, F, q0=q0, u0=u0)
model.add(pendulum)

model.assemble()

tspan = [0, 10]
# dt = 1e-1
dt = 1e-2

# solver = Euler_forward(model, tspan, dt)
# t, q, u = solver.solve()

solver = Euler_backward(model, tspan, dt)
t, q, u, _ = solver.solve()


# fig, ax = plt.subplots()

# ax.plot(t, q[:,0], '-k', label='x')
# ax.plot(t, q[:,1], '-b', label='y')
# ax.plot(t, u[:,0], '-g', label='u')
# ax.legend()
# plt.show()
# exit()

##########################
# reference solution
def eqm(t,z):
    x, y, u = z
    dz = np.zeros(3)
    dz[0] = y * u + l_t(t) / l(t) * x
    dz[1] = -x * u + l_t(t) / l(t) * y
    dz[2] = -2 * l_t(t) / l(t) * u + (F(t)[0] * y + F(t)[1] * x) / (m * l(t)**2)
    return dz

z0 = np.concatenate([q0, u0])
# z0 = np.array([0, L, 0])
sol = solve_ivp(eqm, [t[0], t[-1]], z0, method='RK45', t_eval=t, rtol=1e-8, atol=1e-12)
z = sol.y[:3]
dz = np.zeros_like(z)
for i, (ti, zi) in enumerate(zip(t, z.T)):
    dz[:, i] = eqm(ti, zi)

fig, ax = plt.subplots(2, 1)

ax[0].plot(t, z[0], '-k', label='x')
ax[0].plot(t, z[1], '-b', label='y')
ax[0].plot(t, z[2], '-g', label='u')
ax[0].plot(t, q[:,0], 'xk', label='x')
ax[0].plot(t, q[:,1], 'xb', label='y')
ax[0].plot(t, u[:,0], 'xg', label='u')
ax[0].legend()

ax[1].plot(t, dz[0], '-k', label='dx')
ax[1].plot(t, dz[1], '-b', label='dy')
ax[1].plot(t, dz[2], '-g', label='du')
ax[1].legend()

fig, ax = plt.subplots()
ax.set_xlim([-2*L, 2*L])
ax.set_ylim([-2*L, 2*L])
line, = ax.plot([], [], '-ok', label='')

def animate(i):
    # x = np.array([0, z[0, i]])
    # y = np.array([0, -z[1, i]])
    x = np.array([0, q[i, 0]])
    y = np.array([0, -q[i, 1]])
    line.set_data((x, y))
    return line,

anim = animation.FuncAnimation(fig, animate, frames=len(t))

plt.show()

fig, ax = plt.subplots()
e_spy = z[0]**2 + z[1]**2 - l(t)**2
e = q[:,0]**2 + q[:,1]**2 - l(t)**2
ax.plot(t, e_spy, '-r')
ax.plot(t, e, 'xr')
plt.show()