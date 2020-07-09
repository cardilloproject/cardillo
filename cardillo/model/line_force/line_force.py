from cardillo.math import Numerical_derivative

class Line_force():
    def __init__(self, line_force, subsystem):
        if not callable(line_force):
            self.line_force = lambda xi, t: line_force
        else:
            self.line_force = line_force
        self.subsystem = subsystem

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    def E_pot(self, t, q):
        return self.subsystem.body_force_pot(t, q, self.line_force)

    def f_pot(self, t, q):
        return self.subsystem.body_force(t, q, self.line_force)

    def f_pot_q(self, t, q, coo):
        self.subsystem.body_force_q(t, q, coo, self.line_force)
        # dense = Numerical_derivative(self.f_pot)._x(t, q)
        # coo.extend(dense, (self.uDOF, self.qDOF))