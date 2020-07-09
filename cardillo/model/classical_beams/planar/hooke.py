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

    def dn_dlambda(self, lambda_, kappa, kappa_0):
        return self.EA

    def dn_dkappa(self, lambda_, kappa, kappa_0):
        return 0

    def dm_dlambda(self, lambda_, kappa, kappa_0):
        return 0

    def dm_dkappa(self, lambda_, kappa, kappa_0):        
        return self.EI