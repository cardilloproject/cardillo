class Linear_spring_damper:
    def __init__(self, k, d, g0=None):
        self.k = k
        self.d = d
        self.g0 = g0

    def pot(self, t, g):
        return 0.5 * self.k * (g - self.g0) ** 2

    def pot_g(self, t, g):
        return self.k * (g - self.g0)

    def pot_gg(self, t, g):
        return self.k

    def F(self, t, g, gamma):
        return self.d * gamma

    # def F_g(self, t, g, gamma):
    #     return 0

    def F_gamma(self, t, g, gamma):
        return self.d
