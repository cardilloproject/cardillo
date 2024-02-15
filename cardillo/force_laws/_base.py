from abc import ABC, abstractmethod
import numpy as np


class ScalarForceLaw(ABC):
    def __init__(self, subsystem):
        self.subsystem = subsystem
        self.l = self.subsystem.l
        self.l_q = self.subsystem.l_q
        self.l_dot = self.subsystem.l_dot
        self.l_dot_q = self.subsystem.l_dot_q
        self.l_dot_u = self.subsystem.l_dot_u

        self.nla_c = 1

    def assembler_callback(self):
        self.subsystem.assembler_callback()
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    @abstractmethod
    def _E_pot(self, t, l, l_dot):
        ...

    @abstractmethod
    def _la_c(self, t, l, l_dot):
        ...

    @abstractmethod
    def _la_c_l(self, t, l, l_dot):
        ...

    @abstractmethod
    def _la_c_l_dot(self, t, l, l_dot):
        ...

    @abstractmethod
    def _c(self, t, l, l_dot, la_c):
        ...

    @abstractmethod
    def _c_l(self, t, l, l_dot, la_c):
        ...

    @abstractmethod
    def _c_l_dot(self, t, l, l_dot, la_c):
        ...

    @abstractmethod
    def c_la_c(self):
        ...

    def E_pot(self, t, q):
        return self._E_pot(t, self.l(t, q))

    def la_c(self, t, q, u):
        return self._la_c(t, self.l(t, q), self.l_dot(t, q, u))

    def c(self, t, q, u, la_c):
        return self._c(t, self.l(t, q), self.l_dot(t, q, u), la_c)

    def c_q(self, t, q, u, la_c):
        return self._c_l(t, self.l(t, q), self.l_dot(t, q, u), la_c) * self.l_q(
            t, q
        ) + self._c_l_dot(t, self.l(t, q), self.l_dot(t, q, u), la_c) * self.l_dot_q(
            t, q, u
        )

    def c_u(self, t, q, u, la_c):
        return self._c_l_dot(t, self.l(t, q), self.l_dot(t, q, u), la_c) * self.l_dot_u(
            t, q, u
        )

    def W_c(self, t, q):
        return self.subsystem.W_l(t, q).reshape(self.subsystem._nu, self.nla_c)

    def Wla_c_q(self, t, q, la_c):
        return la_c * self.subsystem.W_l_q(
            t, q
        )  

    def export(self, sol_i, **kwargs):
        return self.subsystem.export(sol_i, **kwargs)


class ScalarForceLaw_h(ABC):
    def __init__(self, subsystem):
        self.subsystem = subsystem
        self.l = self.subsystem.l
        self.l_dot = self.subsystem.l_dot

    def assembler_callback(self):
        self.subsystem.assembler_callback()
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    @abstractmethod
    def e_pot(self, t, l, l_dot):
        ...

    @abstractmethod
    def la(self, t, l, l_dot):
        ...

    @abstractmethod
    def la_l(self, t, l, l_dot):
        ...

    @abstractmethod
    def la_l_dot(self, t, l, l_dot):
        ...

    def E_pot(self, t, q):
        return self.e_pot(t, self.l(t, q))

    def force(self, t, q, u):
        return self.la(t, self.l(t, q), self.l_dot(t, q, u))

    def force_q(self, t, q, u):
        return self.la_l(t, self.l(t, q), self.l_dot(t, q, u)) * self.subsystem.l_q(
            t, q
        ) + self.la_l_dot(
            t, self.l(t, q), self.l_dot(t, q, u)
        ) * self.subsystem.l_dot_q(
            t, q
        )

    def force_u(self, t, q, u):
        return self.la_l_dot(
            t, self.l(t, q), self.l_dot(t, q, u)
        ) * self.subsystem.l_dot_q(t, q, u)

    def h(self, t, q, u):
        return self.force(t, q, u) * self.subsystem.W_l(t, q).reshape(
            self.subsystem._nu
        )

    def h_q(self, t, q, u):
        W = self.subsystem.W_l(t, q).reshape(self.subsystem._nu)
        return self.force(t, q, u) * self.subsystem.W_l_q(t, q) + np.outer(
            W, self.force_q(t, q, u)
        )

    def h_u(self, t, q, u):
        W = self.subsystem.W_l(t, q).reshape(self.subsystem._nu)
        return np.outer(W, self.force_u(t, q, u))

    def export(self, sol_i, **kwargs):
        return self.subsystem.export(sol_i, **kwargs)
