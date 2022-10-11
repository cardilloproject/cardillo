import numpy as np


class Power_spring:
    def __init__(self, k, gamma, g0=None):
        self.k = k
        self.gamma = gamma
        self.g0 = g0

    def pot(self, t, g):
        return 0.5 * self.k * (abs(g - self.g0)) ** (self.gamma)

    def pot_g(self, t, g):
        return (
            0.5
            * self.k
            * self.gamma
            * (abs(g - self.g0)) ** (self.gamma - 1)
            * np.sign(g - self.g0)
        )

    def pot_gg(self, t, g):
        if np.isclose(g, self.g0):
            stiffness = (
                0.5
                * self.k
                * self.gamma
                * (self.gamma - 1)
                * (abs(1e-5)) ** (self.gamma - 2)
            )
        else:
            stiffness = (
                0.5
                * self.k
                * self.gamma
                * (self.gamma - 1)
                * (abs(g - self.g0)) ** (self.gamma - 2)
            )

        return stiffness
