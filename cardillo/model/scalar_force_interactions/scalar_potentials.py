class QuadraticPotential:
    def __init__(self, k):
        self.k = k

    def pot(self, g, g0):
        return 0.5 * self.k * (g - g0) ** 2

    def pot_g(self, g, g0):
        return self.k * (g - g0)

    def pot_gg(self, g, g0):
        return self.k
