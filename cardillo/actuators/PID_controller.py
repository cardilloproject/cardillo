import numpy as np
from cardillo.actuators._base import BaseActuator


class PIDcontroller(BaseActuator):
    def __init__(self, subsystem, kp, ki, kd, tau):
        super().__init__(subsystem, tau, nla_tau=1, ntau=2)
        self.nq = 1
        self.q0 = np.zeros(1)

        self.kp = kp
        self.ki = ki
        self.kd = kd

    def assembler_callback(self):
        self.qDOF = np.concatenate([self.my_qDOF, self.subsystem.qDOF])
        self._nq = len(self.qDOF)
        self.uDOF = self.subsystem.uDOF
        self._nu = len(self.uDOF)

    def q_dot(self, t, q, u):
        return self.subsystem.l(t, q[1:]) - self.tau(t)[0]

    def q_dot_q(self, t, q, u):
        return np.array([0, *self.subsystem.l_q(t, q[1:])])

    def W_tau(self, t, q):
        return self.subsystem.W_l(t, q[1:])

    def W_tau_q(self, t, q):
        W_tau_q = np.zeros((self._nu, self.nla_tau, self._nq))
        W_tau_q[:, :, 1:] = self.subsystem.W_l_q(t, q[1:])
        return W_tau_q

    def la_tau(self, t, q, u):
        tau = self.tau(t)
        integral_error = q[0]
        return -np.array(
            [
                self.ki * integral_error
                + self.kp * (self.subsystem.l(t, q[1:]) - tau[0])
                + self.kd * (self.subsystem.l_dot(t, q[1:], u) - tau[1])
            ]
        )

    def la_tau_q(self, t, q, u):
        la_tau_q = np.zeros((self.nla_tau, self._nq))
        la_tau_q[0, 0] = -self.ki
        la_tau_q[0, 1:] = -np.array(
            [
                self.kp * self.subsystem.l_q(t, q[1:])
                + self.kd * self.subsystem.l_dot_q(t, q[1:], u)
            ]
        )
        return la_tau_q

    def la_tau_u(self, t, q, u):
        return -np.array([self.kd * self.subsystem.l_dot_u(t, q[1:], u)])
