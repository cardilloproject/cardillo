import numpy as np


class MaxwellElement:
    def __init__(self, subsystem, stiffness, viscosity, l_ref=None, q0=np.zeros(1)):
        self.subsystem = subsystem
        self.k = stiffness
        self.eta = viscosity
        self.l_ref = l_ref
        self.nq = 1  # l_damper
        self.q0 = q0

    def assembler_callback(self):
        self.qDOF = np.concatenate((self.subsystem.qDOF, self.q_dotDOF))
        self.uDOF = self.subsystem.uDOF
        self._nq = len(self.qDOF)
        self._nu = len(self.uDOF)
        if self.l_ref is None:
            self.l_ref = self.subsystem.l(self.subsystem.t0, self.subsystem.q0)

    def q_dot(self, t, q, u):
        l_d = q[-1]
        return (self.k / self.eta) * (self.subsystem.l(t, q[:-1]) - l_d - self.l_ref)

    def q_dot_q(self, t, q, u):
        q_dot_q = np.zeros(self._nq)
        q_dot_q[:-1] = (self.k / self.eta) * self.subsystem.l_q(t, q[:-1])
        q_dot_q[-1] -= self.k / self.eta
        return q_dot_q

    def q_dot_u(self, t, q):
        return np.zeros(self._nu)

    def q_ddot(self, t, q, u, u_dot):
        l_d_dot = self.q_dot(t, q, u)
        return (self.k / self.eta) * (self.subsystem.l_dot(t, q[:-1], u) - l_d_dot)

    def h(self, t, q, u):
        l_d = q[-1]
        return (
            -self.subsystem.W_l(t, q[:-1])
            * self.k
            * (self.subsystem.l(t, q[:-1]) - l_d - self.l_ref)
        )

    def h_q(self, t, q, u):
        l_d = q[-1]
        h_q = np.zeros((self._nu, self._nq))
        h_q[:, :-1] = -(
            self.subsystem.W_l_q(t, q[:-1])
            * self.k
            * (self.subsystem.l(t, q[:-1]) - l_d - self.l_ref)
            + np.outer(self.subsystem.W_l(t, q[:-1]), self.subsystem.l_q(t, q[:-1]))
            * self.k
        )
        h_q[:, -1] -= self.subsystem.W_l(t, q[:-1]) * self.k
        return h_q
