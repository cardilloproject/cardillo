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

    def f_pot_el(self, t, qe, Qe, N, dN, qp, qw):
        fe = np.zeros(self.subsystem.nq_el)

        for Ni, dNi, qpi, qwi in zip(N, dN, qp, qw):
            # build matrix of shape functions and derivatives
            NNi = np.kron(np.eye(self.subsystem.dim), Ni)
            dNNi = np.kron(np.eye(self.subsystem.dim), dNi)

            # reference tangential vector
            dr0 = dNNi @ Qe
            G = norm3(dr0)
            
            # integrate elemente line force
            fe += NNi.T @ self.line_force(qpi, t) * G * qwi
        
        return fe

    def f_pot(self, t, q):
        f = np.zeros(self.subsystem.nq)

        for el in range(self.subsystem.nEl):
            # Freedom degree of element
            elDOF = self.subsystem.elDOF[el, :]
            
            f[elDOF] += self.f_pot_el(t, q[elDOF], self.subsystem.Q[elDOF], self.subsystem.N[el], self.subsystem.dN[el], self.subsystem.qp, self.subsystem.qw[el])
        
        return f

    def f_pot_q(self, t, q):
        return np.zeros((self.subsystem.nq, self.subsystem.nq))
