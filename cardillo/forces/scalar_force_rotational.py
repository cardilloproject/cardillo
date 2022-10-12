import numpy as np


def add_rotational_forcelaw(
    RevoluteJoint, force_law_spring=None, force_law_damper=None
):
    class __RevoluteJoint(RevoluteJoint):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.force_law_spring = force_law_spring
            self.force_law_damper = force_law_damper

            assert (self.force_law_spring is not None) or (
                self.force_law_damper is not None
            )

            if self.force_law_spring is not None:
                self.E_pot = lambda t, q: self.force_law_spring.E_pot(
                    t, self.angle(t, q)
                )
                if self.force_law_damper is not None:
                    self.h = lambda t, q, u: self.__f_spring(t, q) + self.__f_damper(
                        t, q, u
                    )
                    self.h_q = lambda t, q, u, coo: self.__f_spring_q(
                        t, q, coo
                    ) + self.__f_damper_q(t, q, u, coo)
                    self.h_u = lambda t, q, u, coo: self.__f_damper_u(t, q, u, coo)
                else:
                    self.h = lambda t, q, u: self.__f_spring(t, q)
                    self.h_q = lambda t, q, u, coo: self.__f_spring_q(t, q, coo)
            else:
                self.h = lambda t, q, u: self.__f_damper(t, q, u)
                self.h_q = lambda t, q, u, coo: self.__f_damper_q(t, q, u, coo)
                self.h_u = lambda t, q, u, coo: self.__f_damper_u(t, q, u, coo)

        def assembler_callback(self):
            super().assembler_callback()

            # TODO: What do we have to implement here?
            if (self.force_law_spring is not None) and (
                self.force_law_spring.g_ref is None
            ):
                self.force_law_spring.g_ref = self.angle0

        def __f_spring(self, t, q):
            return -self.W_angle(t, q) * self.force_law_spring.la(t, self.angle(t, q))

        def __f_spring_q(self, t, q, coo):
            angle = self.angle(t, q)
            dense = -self.force_law_spring.la(t, angle) * self.W_angle_q(
                t, q
            ) - self.force_law_spring.la_g(t, angle) * np.outer(
                self.W_angle(t, q), self.angle_q(t, q)
            )
            coo.extend(dense, (self.uDOF, self.qDOF))

        def __f_damper(self, t, q, u):
            gamma = self.angle_dot(t, q, u)
            return -self.W_angle(t, q) * self.force_law_damper.la(t, gamma)

        def __f_damper_q(self, t, q, u, coo):
            gamma = self.angle_dot(t, q, u)
            gamma_q = self.angle_dot_q(t, q, u)
            W = self.W_angle(t, q)
            dense = -self.force_law_damper.la(t, gamma) * self.W_angle_q(
                t, q
            ) - self.force_law_damper.la_gamma(t, gamma) * np.outer(W, gamma_q)
            coo.extend(dense, (self.uDOF, self.qDOF))

        def __f_damper_u(self, t, q, u, coo):
            gamma = self.angle_dot(t, q, u)
            W = self.W_angle(t, q)
            dense = -self.force_law_damper.la_gamma(t, gamma) * np.outer(W, W)
            coo.extend(dense, (self.uDOF, self.uDOF))

    return __RevoluteJoint
