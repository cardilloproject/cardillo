import numpy as np
import matplotlib.pyplot as plt

from cardillo.model import Model
from cardillo.solver import Euler_forward, Euler_backward, Moreau
from cardillo.model.frame import Frame
from cardillo.model.point_mass import Point_mass
from cardillo.model.force import Force
from cardillo.model.scalar_force_interactions.potential_force_laws import Linear_spring
from cardillo.model.scalar_force_interactions.nonpotential_force_laws import Linear_damper
from cardillo.model.scalar_force_interactions import Translational_f_pot, Translational_f_npot

if __name__ == "__main__":
    m = 1
    g = 9.81
    l0 = 1
    k = 2.0e2
    d = 4

    frame = Frame()

    q0 = np.array([0, 0, -l0])
    mass = Point_mass(m, q0=q0, u0=np.zeros(3))

    f_g = Force(lambda t: np.array([0, 0, -m * g]), mass)

    linear_spring = Linear_spring(k)
    spring_element = Translational_f_pot(linear_spring, frame, mass)
    # spring_element = Translational_f_pot(linear_spring, frame, mass, n=lambda t: np.array([0, 0, -1]))
    # spring_element = Translational_f_pot(linear_spring, frame, mass, n=np.array([0, 0, -1]))

    linear_damper = Linear_damper(d)
    damping_element = Translational_f_npot(linear_damper, frame, mass)
    damping_element = Translational_f_npot(linear_damper, frame, mass, n=np.array([0, 0, 1]))

    model = Model()
    model.add(frame)
    model.add(mass)
    model.add(f_g)
    model.add(spring_element)
    model.add(damping_element)
    model.assemble()

    t0 = 0
    t1 = 2
    t_span = (t0, t1)
    dt = 1.0e-2
    # solver = Euler_backward(model, t_span, dt, numerical_jacobian=False, debug=False)
    # t, q, u, la_g, la_gamma = solver.solve()
    solver = Moreau(model, t_span, dt)
    t, q, u, la_g, la_gamma = solver.solve()
    # solver = Euler_forward(model, t_span, dt)
    # t, q, u = solver.solve()

    plt.plot(t, q[:, 0], '-r')
    plt.plot(t, q[:, 1], '-g')
    plt.plot(t, q[:, 2], '-b')
    plt.show()