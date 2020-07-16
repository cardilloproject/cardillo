class Hooke(object):
    r"""Simple quadratic and uncoupled material model.
    """

    def __init__(self, EA, EI):
        self.EA = EA
        self.EI = EI
        
    def potential(self, lambda_, kappa, kappa_0):
        return 0.5 * self.EA * (lambda_ - 1)**2 + 0.5 * self.EI * (kappa - kappa_0)**2

    def n(self, lambda_, kappa, kappa_0):
        return self.EA * (lambda_ - 1)

    def m(self, lambda_, kappa, kappa_0):
        return self.EI * (kappa - kappa_0)

    def n_lambda(self, lambda_, kappa, kappa_0):
        return self.EA

    def n_kappa(self, lambda_, kappa, kappa_0):
        return 0

    def m_lambda(self, lambda_, kappa, kappa_0):
        return 0

    def m_kappa(self, lambda_, kappa, kappa_0):        
        return self.EI