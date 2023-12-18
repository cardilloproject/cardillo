import numpy as np

class ScalarForceLaw:
    def __init__(self, subsystem):
        self.subsystem = subsystem

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    def _E_pot(self, t, q):
        return self.subsystem.E_pot(t, q)

    def h(self, t, q, u):
        return -self.la(t, self.subsystem.l(t, q), self.subsystem.l_dot(t, q, u)) * self.subsystem.W_l(
            t, q
        ).reshape(self.subsystem._nu)

    def h_q(self, t, q, u):
        raise NotImplementedError

    def h_u(self, t, q, u):
        raise NotImplementedError
    

class SpringDamper(ScalarForceLaw):
    def __init__(self, subsystem, k, d, l_ref=None):
        super().__init__(subsystem)
        self.k = k
        self.d = d
        if l_ref is None:
            self.l_ref = self.subsystem.l(self.subsystem.t0, self.subsystem.q0)
        else:
            self.l_ref = l_ref

    def la(self, t, l, l_dot):
        return self.k * (l - self.l_ref) + self.d * l_dot



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
