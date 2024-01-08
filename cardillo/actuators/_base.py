import numpy as np


class BaseActuator:
    def __init__(self, subsystem, tau, nla_tau, ntau):
        self.subsystem = subsystem
        if not callable(tau):
            self.tau = lambda t: tau
        else:
            self.tau = tau
        self.nla_tau = nla_tau
        self.ntau = ntau

    def la_tau_q(self, t, q, u):
        return np.zeros((self.nla_tau, len(q)))

    def la_tau_u(self, t, q, u):
        return np.zeros((self.nla_tau, len(u)))

    def Wla_tau_q(self, t, q, u):
        return np.einsum(
            "ijk,j->ik", self.W_tau_q(t, q), self.la_tau(t, q, u)
        ) + self.W_tau(t, q) @ self.la_tau_q(t, q, u)

    def Wla_tau_u(self, t, q, u):
        return self.W_tau(t, q) @ self.la_tau_u(t, q, u)
