import numpy as np

from ._base import ScalarForceLaw


class Spring(ScalarForceLaw):
    def __init__(self, subsystem, k, l_ref=None, name="spring"):
        super().__init__(subsystem)
        self.k = k
        self.l_ref = l_ref
        self.name = name

    def assembler_callback(self):
        super().assembler_callback()
        if self.l_ref is None:
            self.l_ref = self.subsystem.l(self.subsystem.t0, self.subsystem.q0)

    def e_pot(self, t, l):
        return 0.5 * self.k * (l - self.l_ref) ** 2

    def la(self, t, l, l_dot):
        return -self.k * (l - self.l_ref)

    def la_l(self, t, l, l_dot):
        return -self.k

    def la_l_dot(self, t, l, l_dot):
        return 0.0
