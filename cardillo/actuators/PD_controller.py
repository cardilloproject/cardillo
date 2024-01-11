import numpy as np
from cardillo.actuators._base import BaseActuator


class PDcontroller(BaseActuator):
    def __init__(self, subsystem, kp, kd, tau):
        super().__init__(subsystem, tau, nla_tau=1, ntau=2)

        self.kp = kp
        self.kd = kd
        self.W_tau = self.subsystem.W_l
        self.W_tau_q = self.subsystem.W_l_q

    def la_tau(self, t, q, u):
        return -np.array(
            [
                self.kp * (self.subsystem.l(t, q) - self.tau(t)[0])
                + self.kd * (self.subsystem.l_dot(t, q, u) - self.tau(t)[1])
            ]
        )

    def la_tau_q(self, t, q, u):
        return -np.array(
            [
                self.kp * self.subsystem.l_q(t, q)
                + self.kd * self.subsystem.l_dot_q(t, q, u)
            ]
        )

    def la_tau_u(self, t, q, u):
        return -np.array([self.kd * self.subsystem.l_dot_u(t, q, u)])
