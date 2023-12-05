from numpy import zeros


class RotationalTransmission:
    r"""Transmission for objects with scalar 'angle' """

    def __init__(self, subsystem=None):
        if subsystem is None:
            raise ValueError("No subsystem added to 'RotationalTransmission'.")
        self.subsystem = subsystem
        self.nla_tau = 1
        
        self.l = lambda t, q: self.subsystem.angle(t, q)
        self.l_dot = lambda t, q, u: self.subsystem.angle_dot(t, q, u)

        self.l_q = lambda t, q: self.subsystem.angle_q(t, q)
        self.l_dot_q = lambda t, q, u: self.subsystem.angle_dot_q(t, q, u)
        self.l_dot_u = lambda t, q, u: self.subsystem.angle_dot_u(t, q, u)

        self.W_tau = lambda t, q: self.subsystem.W_angle(t, q)
        self.W_tau_q = lambda t, q: self.subsystem.W_angle_q(t, q)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF


    


