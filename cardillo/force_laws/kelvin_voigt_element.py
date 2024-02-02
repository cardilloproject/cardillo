import numpy as np

from cardillo.force_laws._base import ScalarForceLaw


class KelvinVoigtElement(ScalarForceLaw):
    def __init__(self, subsystem, k, d, l_ref=None, name="kelvin_voigt_element"):
        super().__init__(subsystem)
        self.k = k
        self.d = d
        self.l_ref = l_ref
        self.name = name

    def assembler_callback(self):
        super().assembler_callback()
        if self.l_ref is None:
            self.l_ref = self.subsystem.l(self.subsystem.t0, self.subsystem.q0)

    def la(self, t, l, l_dot):
        return self.k * (l - self.l_ref) + self.d * l_dot

    def la_l(self, t, l, l_dot):
        return self.k

    def la_l_dot(self, t, l, l_dot):
        return self.d
