import numpy as np


class MaxwellElement:
    r"""Scalar force law consisting of a linear spring and a linear damper acting in series.

    Parameters
    ----------
    subsystem : object
        Object providing the interface for a scalar force law, e.g., Revolute, TwoPointInteraction
    stiffness : float
        Stiffness of spring.
    viscosity : float
        Viscosity of linear damper.
    l_ref : float
        Undeformed elongation of linear spring.
    q0 : np.ndarray(1,)
        Initial elongation of linear damper.
    name : str
        Name of contribution.
    """

    def __init__(
        self,
        subsystem,
        stiffness,
        viscosity,
        l_ref=None,
        q0=np.zeros(1),
        name="maxwell_element",
    ):
        self.subsystem = subsystem
        self.k = stiffness
        self.eta = viscosity
        self.l_ref = l_ref
        self.nq = 1  # l_damper
        self.q0 = q0
        self.name = name

    def assembler_callback(self):
        self.qDOF = np.concatenate((self.my_qDOF, self.subsystem.qDOF))
        self.uDOF = self.subsystem.uDOF
        self._nq = len(self.qDOF)
        self._nu = len(self.uDOF)
        if self.l_ref is None:
            self.l_ref = self.subsystem.l(self.subsystem.t0, self.subsystem.q0)

    def q_dot(self, t, q, u):
        l_d = q[0]
        return (self.k / self.eta) * (self.subsystem.l(t, q[1:]) - l_d - self.l_ref)

    def q_dot_q(self, t, q, u):
        q_dot_q = np.zeros(self._nq)
        q_dot_q[1:] = (self.k / self.eta) * self.subsystem.l_q(t, q[1:])
        q_dot_q[0] -= self.k / self.eta
        return q_dot_q

    def q_dot_u(self, t, q):
        return np.zeros(self._nu)

    def E_pot(self, t, q):
        l_d = q[0]
        return 0.5 * self.k * (self.subsystem.l(t, q[1:]) - l_d - self.l_ref) ** 2

    def force(self, t, q, u):
        l_d = q[0]
        return -self.k * (self.subsystem.l(t, q[1:]) - l_d - self.l_ref)

    def h(self, t, q, u):
        l_d = q[0]
        return self.force(t, q, u) * self.subsystem.W_l(t, q[1:])

    def h_q(self, t, q, u):
        l_d = q[0]
        qext = q[1:]
        h_q = np.zeros((self._nu, self._nq))
        h_q[:, 1:] = -(
            self.subsystem.W_l_q(t, qext)
            * self.k
            * (self.subsystem.l(t, qext) - l_d - self.l_ref)
            + np.outer(self.subsystem.W_l(t, qext), self.subsystem.l_q(t, qext))
            * self.k
        )
        h_q[:, 0] -= self.subsystem.W_l(t, q[1:]) * self.k
        return h_q
