import numpy as np
from cardillo.actuators._base import BaseActuator


class Motor(BaseActuator):
    def __init__(self, subsystem, tau):
        super().__init__(subsystem, tau, nla_tau=1, ntau=1)
        self.W_tau = self.subsystem.W_l
        self.W_tau_q = self.subsystem.W_l_q

    def la_tau(self, t, q, u):
        return np.array([self.tau(t)])
