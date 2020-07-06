class Hooke(object):
    r"""Simple Hooke material model.
    
    Parameters
    ----------
    EA : float
        axial stiffness
    """
    def __init__(self, EA):
        self.EA = EA
        
    def potential(self, lambda_):
        return 0.5 * self.EA * (lambda_ - 1)**2

    def n(self, lambda_):
        return self.EA * (lambda_ - 1)

    def dn(self, lambda_):
        return self.EA