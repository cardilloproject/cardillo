from cardillo.math import Numerical_derivative

class Force_distr1D():
    def __init__(self, force_distr1D, subsystem, edge_idx):
        if not callable(force_distr1D):
            self.force_distr1D = lambda t, xi: force_distr1D
        else:
            self.force_distr1D = force_distr1D
        self.subsystem = subsystem
        self.edge_idx = edge_idx

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    def E_pot(self, t, q):
        return self.subsystem.force_distr1D_pot(t, q, self.force_distr1D, self.edge_idx)

    def f_pot(self, t, q):
        return self.subsystem.force_distr1D(t, q, self.force_distr1D, self.edge_idx)

    def f_pot_q(self, t, q, coo):
        self.subsystem.force_distr1D_q(t, q, coo, self.force_distr1D, self.edge_idx)
        # dense = Numerical_derivative(self.f_pot)._x(t, q)
        # coo.extend(dense, (self.uDOF, self.qDOF))