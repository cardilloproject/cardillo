class LinearSpring:
    def __init__(self, k, g_ref=None):
        self.k = k
        self.g_ref = g_ref

    def E_pot(self, t, g):
        return 0.5 * self.k * (g - self.g_ref) ** 2

    def la(self, t, g):
        return self.k * (g - self.g_ref)

    def la_g(self, t, g):
        return self.k


class LinearDamper:
    def __init__(self, d):
        self.d = d

    def la(self, t, gamma):
        return self.d * gamma

    def la_gamma(self, t, gamma):
        return self.d
