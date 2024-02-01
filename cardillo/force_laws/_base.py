import numpy as np


class ScalarForceLaw:
    def __init__(self, subsystem):
        self.subsystem = subsystem

    def assembler_callback(self):
        self.subsystem.assembler_callback()
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    def _E_pot(self, t, q):
        return self.subsystem.E_pot(t, q)

    def h(self, t, q, u):
        return -self.la(
            t, self.subsystem.l(t, q), self.subsystem.l_dot(t, q, u)
        ) * self.subsystem.W_l(t, q).reshape(self.subsystem._nu)

    def h_q(self, t, q, u):
        W = self.subsystem.W_l(t, q).reshape(self.subsystem._nu)
        return (
            -self.la(t, self.subsystem.l(t, q), self.subsystem.l_dot(t, q, u))
            * self.subsystem.W_l_q(t, q)
            - self.la_l(t, self.subsystem.l(t, q), self.subsystem.l_dot(t, q, u))
            * np.outer(W, self.subsystem.l_q(t, q))
            - self.la_l_dot(t, self.subsystem.l(t, q), self.subsystem.l_dot(t, q, u))
            * np.outer(W, self.subsystem.l_dot_q(t, q, u))
        )

    def h_u(self, t, q, u):
        W = self.subsystem.W_l(t, q).reshape(self.subsystem._nu)
        return -self.la_l_dot(
            t, self.subsystem.l(t, q), self.subsystem.l_dot(t, q, u)
        ) * np.outer(W, self.subsystem.l_dot_u(t, q, u))

    def la(self, t, l, l_dot):
        return 0.0

    def la_l(self, t, l, l_dot):
        return 0.0

    def la_l_dot(self, t, l, l_dot):
        return 0.0
