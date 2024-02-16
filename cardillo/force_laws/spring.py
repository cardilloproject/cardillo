import numpy as np

from ._base import ScalarForceLaw

class Spring(ScalarForceLaw):
    def __init__(self, subsystem, k, l_ref=None, compliance_form=True, name="linear_spring"):
        super().__init__(subsystem, compliance_form)
        self.k = k
        self.l_ref = l_ref
        self.name = name

    def assembler_callback(self):
        super().assembler_callback()
        if self.l_ref is None:
            self.l_ref = self.subsystem.l(self.subsystem.t0, self.subsystem.q0)

    def _E_pot(self, t, l):
        return 0.5 * self.k * (l - self.l_ref) ** 2

    def _la_c(self, t, l, l_dot):
        return -self.k * (l - self.l_ref)

    def _la_c_l(self, t, l, l_dot):
        return -self.k

    def _la_c_l_dot(self, t, l, l_dot):
        return 0.0

    def _c(self, t, l, l_dot, la_c):
        return la_c / self.k + (l - self.l_ref)

    def _c_l(self, t, l, l_dot, la_c):
        return 1.0

    def _c_l_dot(self, t, l, l_dot, la_c):
        return 0.0

    def c_la_c(self):
        return 1 / self.k
