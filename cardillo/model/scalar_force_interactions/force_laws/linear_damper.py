class Linear_damper():
    def __init__(self, d):
        self.d = d
        self.g0 = 0

    def F(self, t, g, gamma):
        return self.d * gamma

    def F_g(self, t, g, gamma):
        return 0

    def F_gamma(self, t, g, gamma):
        return self.d