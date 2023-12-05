import numpy as np

def Motor(Transmission):
    class _Motor(Transmission):
        def __init__(self, force, **kwargs):
            self.force = force
            
            super().__init__(**kwargs)

        def la_tau(self, t, q, u):
            return self.force(t)
    return _Motor