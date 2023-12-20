import numpy as np

from cardillo.force_laws._base import ScalarForceLaw
    

class Spring(ScalarForceLaw):
    def __init__(self, subsystem, k, l_ref=None):
        super().__init__(subsystem)
        self.k = k
        self.l_ref = l_ref

    def assembler_callback(self):
        super().assembler_callback()
        if self.l_ref is None:
            self.l_ref = self.subsystem.l(self.subsystem.t0, self.subsystem.q0)

    def la(self, t, l, l_dot):
        return self.k * (l - self.l_ref) 
    
    def la_l(self, t, l, l_dot):
        return self.k




# def ScalarForceLaw(Transmission):
#     class _ScalarForceLaw(Transmission):
#         def __init__(self, force_law, **kwargs):
#             self.force_law = force_law
#             super().__init__(**kwargs)

#         def h(self, t, q, u):
#             return -self.force_law(t, self.l(t, q), self.l_dot(t, q, u)) * self.W_l(
#                 t, q
#             ).reshape(self.subsystem._nu)

#         def h_q(self, t, q, u):
#             raise NotImplementedError

#         def h_u(self, t, q, u):
#             raise NotImplementedError

#     return _ScalarForceLaw
