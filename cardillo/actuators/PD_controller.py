import numpy as np


class PDcontroller:
    def __init__(self, subsystem, kp, kd, tau):
        self.subsystem = subsystem
        if not callable(tau):
            self.tau = lambda t: tau
        else:
            self.tau = tau
        self.nla_tau = 1
        self.ntau = 2

        self.kp = kp
        self.kd = kd
        self.W_tau = self.subsystem.W_l

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    def la_tau(self, t, q, u):
        return -(
            self.kp * (self.subsystem.l(t, q) - self.tau(t)[0])
            + self.kd * (self.subsystem.l_dot(t, q, u) - self.tau(t)[1])
        )
