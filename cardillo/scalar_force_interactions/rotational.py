import numpy as np


def add_rotational_forcelaw(force_law, Revolute_joint):
    class __Revolute_joint(Revolute_joint):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.force_law = force_law

            if hasattr(self.force_law, "pot"):
                self.pot = self.__pot

            if hasattr(self.force_law, "pot_g"):
                self.f_pot = self.__f_pot

                if hasattr(self.force_law, "pot_gg"):
                    self.f_pot_q = self.__f_pot_q
                else:
                    self.f_pot_q = self.__f_pot_q_reduced

            if hasattr(self.force_law, "F"):
                self.f_npot = self.__f_npot

                if hasattr(self.force_law, "F_gamma"):
                    self.f_npot_u = self.__f_npot_u

                    if hasattr(self.force_law, "F_g"):
                        self.f_npot_q = self.__f_npot_q
                    else:
                        self.f_npot_q = self.__f_npot_q_reduced

        def assembler_callback(self):
            super().assembler_callback()

            if self.force_law.g0 is None:
                self.force_law.g0 = self.angle0

        # potential forces implementation
        def __pot(self, t, q):
            return self.force_law.pot(t, self.angle(t, q))

        def __f_pot(self, t, q):
            return -self.W_angle(t, q) * self.force_law.pot_g(t, self.angle(t, q))

        def __f_pot_q(self, t, q, coo):
            angle = self.angle(t, q)
            dense = -self.W_angle_q(t, q) * self.force_law.pot_g(
                t, angle
            ) - self.force_law.pot_gg(t, angle) * np.outer(
                self.W_angle(t, q), self.angle_q(t, q)
            )
            coo.extend(dense, (self.uDOF, self.qDOF))

        def __f_pot_q_reduced(self, t, q, coo):
            angle = self.angle(t, q)
            dense = -self.W_angle_q(t, q) * self.force_law.pot_g(t, angle)
            coo.extend(dense, (self.uDOF, self.qDOF))

        # nonpotential forces implementation
        def __f_npot(self, t, q, u):
            g = self.angle(t, q)
            gamma = self.angle_dot(t, q, u)
            return -self.W_angle(t, q) * self.force_law.F(t, g, gamma)

        def __f_npot_q(self, t, q, u, coo):
            g = self.angle(t, q)
            gamma = self.angle_dot(t, q, u)
            g_q = self.angle_q(t, q)
            gamma_q = self.angle_dot_q(t, q, u)
            W = self.W_angle(t, q)
            dense = (
                -self.W_angle_q(t, q) * self.force_law.F(t, g, gamma)
                - self.force_law.F_g(t, g, gamma) * np.outer(W, g_q)
                - self.force_law.F_gamma(t, g, gamma) * np.outer(W, gamma_q)
            )
            coo.extend(dense, (self.uDOF, self.qDOF))

        def __f_npot_q_reduced(self, t, q, u, coo):
            g = self.angle(t, q)
            gamma = self.angle_dot(t, q, u)
            gamma_q = self.angle_dot_q(t, q, u)
            W = self.W_angle(t, q)
            dense = -self.W_angle_q(t, q) * self.force_law.F(
                t, g, gamma
            ) - self.force_law.F_gamma(t, g, gamma) * np.outer(W, gamma_q)
            coo.extend(dense, (self.uDOF, self.qDOF))

        def __f_npot_u(self, t, q, u, coo):
            g = self.angle(t, q)
            gamma = self.angle_dot(t, q, u)
            W = self.W_angle(t, q)
            dense = -self.force_law.F_gamma(t, g, gamma) * np.outer(W, W)
            coo.extend(dense, (self.uDOF, self.uDOF))

    return __Revolute_joint
