import numpy as np


def PDRotationalJoint(Joint, Spring=None, Damper=None):
    class _PDRotationalJoint(Joint):
        def __init__(self, *args, k=None, g_ref=None, d=None, **kwargs):
            super().__init__(*args, **kwargs)

            assert (Spring is not None) or (Damper is not None)

            if Spring is not None:
                self.spring = Spring(k, g_ref)

            if Damper is not None:
                self.damper = Damper(d)

            if Spring is not None:
                self.E_pot = lambda t, q: self.spring.E_pot(t, self.angle(t, q))
                # case spring and damper
                if Damper is not None:
                    self._h = lambda t, q, u: self.__f_spring(t, q) + self.__f_damper(
                        t, q, u
                    )
                    self._h_q = lambda t, q, u, coo: coo.extend(
                        self.__f_spring_q(t, q) + self.__f_damper_q(t, q, u),
                        (self.uDOF, self.qDOF),
                    )
                    self._h_u = lambda t, q, u, coo: self.__f_damper_u(t, q, u, coo)
                # case just a spring
                else:
                    self._h = lambda t, q, u: self.__f_spring(t, q)
                    self._h_q = lambda t, q, u, coo: coo.extend(
                        self.__f_spring_q(t, q), (self.uDOF, self.qDOF)
                    )
            # just a damper
            else:
                self._h = lambda t, q, u: self.__f_damper(t, q, u)
                self._h_q = lambda t, q, u, coo: coo.extend(
                    self.__f_damper_q(t, q, u), (self.uDOF, self.qDOF)
                )
                self.h_u = lambda t, q, u, coo: self.__f_damper_u(t, q, u, coo)

        def assembler_callback(self):
            super().assembler_callback()

            if (Spring is not None) and (self.spring.g_ref is None):
                self.spring.g_ref = 0

        def __f_spring(self, t, q):
            return -self.W_angle(t, q) * self.spring.la(t, self.angle(t, q))

        def __f_spring_q(self, t, q):
            angle = self.angle(t, q)
            dense = -self.spring.la(t, angle) * self.W_angle_q(t, q) - self.spring.la_g(
                t, angle
            ) * np.outer(self.W_angle(t, q), self.angle_q(t, q))
            return dense

        def __f_damper(self, t, q, u):
            gamma = self.angle_dot(t, q, u)
            return -self.W_angle(t, q) * self.damper.la(t, gamma)

        def __f_damper_q(self, t, q, u):
            gamma = self.angle_dot(t, q, u)
            gamma_q = self.angle_dot_q(t, q, u)
            W = self.W_angle(t, q)
            dense = -self.damper.la(t, gamma) * self.W_angle_q(
                t, q
            ) - self.damper.la_gamma(t, gamma) * np.outer(W, gamma_q)
            return dense

        def __f_damper_u(self, t, q, u, coo):
            gamma = self.angle_dot(t, q, u)
            gamma_u = self.angle_dot_u(t, q, u)
            W = self.W_angle(t, q)
            dense = -self.damper.la_gamma(t, gamma) * np.outer(W, W)
            coo.extend(dense, (self.uDOF, self.uDOF))

        def h(self, t, q, u):
            return self._h(t, q, u)

        def h_q(self, t, q, u, coo):
            self._h_q(t, q, u, coo)

        def h_u(self, t, q, u, coo):
            self._h_u(t, q, u, coo)

    return _PDRotationalJoint
