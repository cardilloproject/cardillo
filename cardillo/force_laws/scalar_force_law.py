import numpy as np


def ScalarForceLaw(Transmission):
    class _ScalarForceLaw(Transmission):
        def __init__(self, force_law, **kwargs):
            self.force_law = force_law
            super().__init__(**kwargs)

        def h(self, t, q, u):
            return -self.force_law(t, self.l(t, q), self.l_dot(t, q, u)) * self.W_l(
                t, q
            ).reshape(self.subsystem._nu)

        def h_q(self, t, q, u):
            raise NotImplementedError

        def h_u(self, t, q, u):
            raise NotImplementedError

    return _ScalarForceLaw
