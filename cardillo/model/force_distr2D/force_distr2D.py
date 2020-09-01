from cardillo.math import Numerical_derivative

class Force_distr2D():
    def __init__(self, force_distr2D, subsystem, srf_idx):
        if not callable(force_distr2D):
            self.force_distr2D = lambda t, xi, eta: force_distr2D
        else:
            self.force_distr2D = force_distr2D
        self.subsystem = subsystem
        self.srf_idx = srf_idx

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    def E_pot(self, t, q):
        return self.subsystem.force_distr2D_pot(t, q, self.force_distr2D, self.srf_idx)

    def f_pot(self, t, q):
        return self.subsystem.force_distr2D(t, q, self.force_distr2D, self.srf_idx)

    def f_pot_q(self, t, q, coo):
        self.subsystem.force_distr2D_q(t, q, coo, self.force_distr2D, self.srf_idx)
        # dense = Numerical_derivative(self.f_pot)._x(t, q)
        # coo.extend(dense, (self.uDOF, self.qDOF))