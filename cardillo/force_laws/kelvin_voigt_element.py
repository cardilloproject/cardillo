import numpy as np

from ._base import ScalarForceLaw


class KelvinVoigtElement(ScalarForceLaw):
    def __init__(
        self,
        subsystem,
        k,
        d,
        l_ref=None,
        compliance_form=True,
        name="kelvin_voigt_element",
    ):
        super().__init__(subsystem, compliance_form)
        self.k = k
        self.d = d
        self.l_ref = l_ref
        self.name = name

    def assembler_callback(self):
        super().assembler_callback()
        if self.l_ref is None:
            self.l_ref = self.subsystem.l(self.subsystem.t0, self.subsystem.q0)

    def _E_pot(self, t, l):
        return 0.5 * self.k * (l - self.l_ref) ** 2

    def _la_c(self, t, l, l_dot):
        return -self.k * (l - self.l_ref) - self.d * l_dot

    def _la_c_l(self, t, l, l_dot):
        return -self.k

    def _la_c_l_dot(self, t, l, l_dot):
        return -self.d

    def _c(self, t, l, l_dot, la_c):
        return la_c / self.k + (l - self.l_ref) + (self.d / self.k) * l_dot

    def _c_l(self, t, l, l_dot, la_c):
        return 1

    def _c_l_dot(self, t, l, l_dot, la_c):
        return self.d / self.k

    def c_la_c(self):
        return 1 / self.k

