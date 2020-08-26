from cardillo.math import Numerical_derivative

class Force_distr3D():
    def __init__(self, force_distr3D, subsystem):
        if not callable(force_distr3D):
            self.force_distr3D = lambda t, xi, eta, zeta: force_distr3D
        else:
            self.force_distr3D = force_distr3D
        self.subsystem = subsystem

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    def E_pot(self, t, q):
        return self.subsystem.force_distr3D_pot(t, q, self.force_distr3D)

    def f_pot(self, t, q):
        return self.subsystem.force_distr3D(t, q, self.force_distr3D)

    def f_pot_q(self, t, q, coo):
        self.subsystem.force_distr3D_q(t, q, coo, self.force_distr3D)
        # dense = Numerical_derivative(self.f_pot)._x(t, q)
        # coo.extend(dense, (self.uDOF, self.qDOF))