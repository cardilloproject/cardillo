import numpy as np


def PDRotational(Joint, Spring=None, Damper=None):
    class _PDRotational(Joint):
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
                    self._h_q = lambda t, q, u: self.__f_spring_q(
                        t, q
                    ) + self.__f_damper_q(t, q, u)
                    self.h_u = lambda t, q, u: self.__f_damper_u(t, q, u)
                # case just a spring
                else:
                    self._h = lambda t, q, u: self.__f_spring(t, q)
                    self._h_q = lambda t, q, u: self.__f_spring_q(t, q)
            # just a damper
            else:
                self._h = lambda t, q, u: self.__f_damper(t, q, u)
                self._h_q = lambda t, q, u: self.__f_damper_q(t, q, u)
                self.h_u = lambda t, q, u: self.__f_damper_u(t, q, u)

        def assembler_callback(self):
            super().assembler_callback()

            if (Spring is not None) and (self.spring.g_ref is None):
                self.spring.g_ref = 0

        def __f_spring(self, t, q):
            return -self.W_angle(t, q) * self.spring.la(t, self.angle(t, q))

        def __f_spring_q(self, t, q):
            angle = self.angle(t, q)
            return -self.spring.la(t, angle) * self.W_angle_q(t, q) - self.spring.la_g(
                t, angle
            ) * np.outer(self.W_angle(t, q), self.angle_q(t, q))

        def __f_damper(self, t, q, u):
            gamma = self.angle_dot(t, q, u)
            return -self.W_angle(t, q) * self.damper.la(t, gamma)

        def __f_damper_q(self, t, q, u):
            gamma = self.angle_dot(t, q, u)
            gamma_q = self.angle_dot_q(t, q, u)
            W = self.W_angle(t, q)
            return -self.damper.la(t, gamma) * self.W_angle_q(
                t, q
            ) - self.damper.la_gamma(t, gamma) * np.outer(W, gamma_q)

        def __f_damper_u(self, t, q, u):
            gamma = self.angle_dot(t, q, u)
            W = self.W_angle(t, q)
            return -self.damper.la_gamma(t, gamma) * np.outer(W, W)

        def h(self, t, q, u):
            return self._h(t, q, u)

        def h_q(self, t, q, u):
            return self._h_q(t, q, u)

        # E_pot and h_u defined in self.__init__ function if necessary

    return _PDRotational
