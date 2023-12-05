import numpy as np


def Motor(Transmission):
    class _Motor(Transmission):
        def __init__(self, force, **kwargs):
            self.force = force
            self.nla_tau = 1
            super().__init__(**kwargs)

            self.W_tau = self.W_l

        def la_tau(self, t, q, u):
            return self.force(t)

    return _Motor
