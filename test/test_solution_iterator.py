import numpy as np

from cardillo import System
from cardillo.discrete import RigidBody, Sphere, Box, Frame
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane
from cardillo.solver import Moreau, BackwardEuler, ScipyIVP, Rattle

############
# parameters
############

# radius of ball
radius = 0.05

# contact parameters
e_N = 0.75  # restitution coefficient in normal direction
e_F = 0.0  # restitution coefficient in tangent direction
mu = 0.5  # frictional coefficient

# density of ball
density = 1

# gravitational acceleration
g = np.array([0, 0, -10])

# initial conditions
r_OC0 = np.array([-0.75, 0, 8 * radius])  # initial position of c.o.m.
v_C0 = np.array([1, 0, 0])  # initial velocity of c.o.m.
B_Omega0 = np.array([0, -25, 0])  # initial angular velocity

# initialize system
system = System()

#################
# assemble system
#################

# create ball
q0 = RigidBody.pose2q(r_OC0, np.eye(3))
u0 = np.hstack([v_C0, B_Omega0])

ball = Sphere(RigidBody)(
    radius=radius,
    density=density,
    subdivisions=3,
    q0=q0,
    u0=u0,
    name="ball",
)
# gravitational force for ball
gravity = Force(ball.mass * g, ball, name="gravity")
# add ball and gravitational force to system
system.add(ball, gravity)

# create floor (Box only for visualization purposes)
floor = Box(Frame)(
    dimensions=[1.5, 4 * radius, 0.0001],
    name="floor",
)

# add contact between ball and floor
ball2plane = Sphere2Plane(floor, ball, mu=mu, r=radius, e_N=e_N, e_F=e_F)
system.add(floor, ball2plane)

# assemble system
system.assemble()


def run(Solver):
    ############
    # simulation
    ############
    dt = 2.0e-3  # time step
    t1 = 3 * dt
    solver = Solver(system, t1, dt)  # create solver
    sol = solver.solve()  # simulate system

    for sol_i in sol:
        assert True


if __name__ == "__main__":
    run(Moreau)
    run(BackwardEuler)
    run(ScipyIVP)
    run(Rattle)
