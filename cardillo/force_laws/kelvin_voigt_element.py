import numpy as np

from ._base import ScalarForceLawComplianceForm


class KelvinVoigtElement(ScalarForceLawComplianceForm):
    r"""Scalar force law consisting of a linear spring and a linear damper acting in parallel.

    Parameters
    ----------
    subsystem : object
        Object providing the interface for a scalar force law, e.g., Revolute, TwoPointInteraction
    k : float
        Stiffness of spring.
    d : float
        Viscosity of linear damper.
    l_ref : float
        Undeformed elongation of linear spring.
    compliance_form : bool
        True : Scalar force is contemplated as Lagrange multiplier and force law as algebraic equation resulting in a DAE.
        False : Scalar force is contemplated as function of the system's state resulting in an ODE.
    name : str
        Name of contribution.
    """

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
