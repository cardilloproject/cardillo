import numpy as np
from math import pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.model import Model
from cardillo.model.rigid_body import Rigid_body_quaternion
from cardillo.solver import Euler_forward, Euler_backward

class Rigid_cylinder(Rigid_body_quaternion):
    def __init__(self, m, r, l, q0=None, u0=None):
        A = 1 / 4 * m * r**2 + 1 / 12 * m * l**2
        C = 1 / 2 * m * r**2
        K_theta_S = np.diag(np.array([A, A, C]))

        super().__init__(m, K_theta_S, q0=q0, u0=u0)

    def 

m = 1
R