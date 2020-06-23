import numpy as np
from math import pi, sin, cos, tan
import logging
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

m = 1
L = 2
g = 9.81

Fx = lambda t: 0
Fy = lambda t: -m * g

l = lambda t: L + sin(t)
l_t = lambda t: cos(t)
l_tt = lambda t: -sin(t)

# l = lambda t: L
# l_t = lambda t: 0
# l_tt = lambda t: 0

def eqm(t,z):
    x, y, u = z

    dz = np.zeros(3)
    dz[0] = y * u + l_t(t) / l(t) * x
    dz[1] = -x * u + l_t(t) / l(t) * y
    dz[2] = -2 * l_t(t) / l(t) * u + (Fx(t) * y + Fy(t) * x) / (m * l(t)**2)
    return dz

tspan = [0, 10]
nt = 100
z0 = np.array([L, 0, 0])
# z0 = np.array([0, L, 0])
sol = solve_ivp(eqm, tspan, z0, method='RK45', t_eval=np.linspace(tspan[0], tspan[-1], num=nt), rtol=1e-8, atol=1e-12)
z = sol.y[:3]
t = sol.t
dz = np.zeros_like(z)
for i, (ti, zi) in enumerate(zip(t, z.T)):
    dz[:, i] = eqm(ti, zi)

fig, ax = plt.subplots(2, 1)

ax[0].plot(t, z[0], '-k', label='x')
ax[0].plot(t, z[1], '-b', label='y')
ax[0].plot(t, z[2], '-g', label='u')
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
    x = np.array([0, z[0, i]])
    y = np.array([0, -z[1, i]])
    line.set_data((x, y))
    return line,

anim = animation.FuncAnimation(fig, animate, frames=len(t))

plt.show()
