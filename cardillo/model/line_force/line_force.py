import numpy as np
from cardillo.math.algebra import norm3

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

    def potential(self, t, q):
        raise NotImplementedError('not implemented')

    def f_pot_el(self, t, N, qp, J0, qw):
        fe = np.zeros(self.subsystem.nq_el)

        for Ni, qpi, J0i, qwi in zip(N, qp, J0, qw):
            # build matrix of shape functions and derivatives
            NNi = np.kron(np.eye(self.subsystem.dim), Ni)
            
            # integrate elemente line force
            fe += NNi.T @ self.line_force(qpi, t) * J0i * qwi
        
        return fe

    def f_pot(self, t, q):
        f = np.zeros(self.subsystem.nq)

        for el in range(self.subsystem.nEl):
            # Freedom degree of element
            elDOF = self.subsystem.elDOF[el, :]
            
            f[elDOF] += self.f_pot_el(t, self.subsystem.N[el], self.subsystem.qp, self.subsystem.J0[el], self.subsystem.qw[el])
        
        return f

    def f_pot_q(self, t, q, coo):
        pass
