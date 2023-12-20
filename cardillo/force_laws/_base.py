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
    
    def la(self, t, l, l_dot):
        return 0.0
    
    def la_l(self, t, l, l_dot):
        return 0.0
    
    def la_l_dot(self, t, l, l_dot):
        return 0.0
    


