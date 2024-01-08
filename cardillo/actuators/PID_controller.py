import numpy as np


class PIDcontroller:
    def __init__(self, subsystem, kp, ki, kd, tau):
        self.subsystem = subsystem
        if not callable(tau):
            self.tau = lambda t: tau
        else:
            self.tau = tau
        self.nla_tau = 1
        self.ntau = 2
        self.nq = 1
        self.q0 = np.zeros(1)

        self.kp = kp
        self.ki = ki
        self.kd = kd

    def assembler_callback(self):
        self.qDOF = np.concatenate([self.q_dotDOF, self.subsystem.qDOF])
        self.uDOF = self.subsystem.uDOF

    def q_dot(self, t, q, u):
        return (
            self.subsystem.l(t, q[1:]) - self.tau(t)[0]
        )  # TODO: wie machen wir das?? optionales tau argument für q_dot??

    def W_tau(self, t, q):
        return self.subsystem.W_l(t, q[1:])

    def la_tau(self, t, q, u):
        tau = self.tau(t)
        integral_error = q[0]
        return -(
            self.ki * integral_error
            + self.kp * (self.subsystem.l(t, q[1:]) - tau[0])
            + self.kd * (self.subsystem.l_dot(t, q[1:], u) - tau[1])
        )
