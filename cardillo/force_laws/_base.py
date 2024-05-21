from abc import ABC, abstractmethod
import numpy as np


class ScalarForceLawBase(ABC):
    def __init__(self, subsystem):
        self.subsystem = subsystem
        self.l = self.subsystem.l
        self.l_q = self.subsystem.l_q
        self.l_dot = self.subsystem.l_dot
        self.l_dot_q = self.subsystem.l_dot_q
        self.l_dot_u = self.subsystem.l_dot_u
        self.force = self.la_c  # TODO: Do we keep this?

    def assembler_callback(self):
        self.subsystem.assembler_callback()
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    @abstractmethod
    def _E_pot(self, t, l, l_dot): ...

    @abstractmethod
    def _la_c(self, t, l, l_dot): ...

    @abstractmethod
    def _la_c_l(self, t, l, l_dot): ...

    @abstractmethod
    def _la_c_l_dot(self, t, l, l_dot): ...

    def E_pot(self, t, q):
        return self._E_pot(t, self.l(t, q))

    def la_c(self, t, q, u):
        return self._la_c(t, self.l(t, q), self.l_dot(t, q, u))

    def la_c_q(self, t, q, u):
        return self._la_c_l(t, self.l(t, q), self.l_dot(t, q, u)) * self.l_q(
            t, q
        ) + self._la_c_l_dot(t, self.l(t, q), self.l_dot(t, q, u)) * self.l_dot_q(
            t, q, u
        )

    def la_c_u(self, t, q, u):
        return self._la_c_l_dot(t, self.l(t, q), self.l_dot(t, q, u)) * self.l_dot_u(
            t, q, u
        )

    def _h(self, t, q, u):
        return self.la_c(t, q, u) * self.subsystem.W_l(t, q).reshape(self.subsystem._nu)

    def _h_q(self, t, q, u):
        return self.la_c(t, q, u) * self.subsystem.W_l_q(t, q).reshape(
            self.subsystem._nu, self.subsystem._nq
        ) + np.outer(
            self.subsystem.W_l(t, q).reshape(self.subsystem._nu), self.la_c_q(t, q, u)
        )

    def _h_u(self, t, q, u):
        return np.outer(
            self.subsystem.W_l(t, q).reshape(self.subsystem._nu), self.la_c_u(t, q, u)
        )

    def export(self, sol_i, **kwargs):
        return self.subsystem.export(sol_i, **kwargs)


class ScalarForceLaw(ScalarForceLawBase):
    def __init__(self, subsystem):
        super().__init__(subsystem)
        self.h = self._h
        self.h_q = self._h_q
        self.h_u = self._h_u


class ScalarForceLawComplianceForm(ScalarForceLawBase):
    def __init__(self, subsystem, compliance_form=True):
        super().__init__(subsystem)
        if compliance_form:
            self.nla_c = 1
            self.c = self.__c
            self.c_q = self.__c_q
            self.c_u = self.__c_u
        else:
            self.h = self._h
            self.h_q = self._h_q
            self.h_u = self._h_u

    @abstractmethod
    def _c(self, t, l, l_dot, la_c): ...

    @abstractmethod
    def _c_l(self, t, l, l_dot, la_c): ...

    @abstractmethod
    def _c_l_dot(self, t, l, l_dot, la_c): ...

    @abstractmethod
    def c_la_c(self): ...

    def __c(self, t, q, u, la_c):
        return self._c(t, self.l(t, q), self.l_dot(t, q, u), la_c)

    def __c_q(self, t, q, u, la_c):
        return self._c_l(t, self.l(t, q), self.l_dot(t, q, u), la_c) * self.l_q(
            t, q
        ) + self._c_l_dot(t, self.l(t, q), self.l_dot(t, q, u), la_c) * self.l_dot_q(
            t, q, u
        )

    def __c_u(self, t, q, u, la_c):
        return self._c_l_dot(t, self.l(t, q), self.l_dot(t, q, u), la_c) * self.l_dot_u(
            t, q, u
        )

    def W_c(self, t, q):
        return self.subsystem.W_l(t, q).reshape(self.subsystem._nu, self.nla_c)

    def Wla_c_q(self, t, q, la_c):
        return la_c * self.subsystem.W_l_q(t, q).reshape(
            self.subsystem._nu, self.subsystem._nq
        )
