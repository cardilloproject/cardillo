class Linear_spring:
    def __init__(self, k, g0=None):
        self.k = k
        self.g0 = g0

    def pot(self, t, g):
        return 0.5 * self.k * (g - self.g0) ** 2

    def pot_g(self, t, g):
        return self.k * (g - self.g0)

    def pot_gg(self, t, g):
        return self.k
