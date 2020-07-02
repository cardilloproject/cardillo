import numpy as np
from cardillo.math.algebra import norm3
from cardillo.model.bilateral_constraints import Revolute_joint

class Rotational_f_pot(Revolute_joint):
    def __init__(self, force_law, subsystem1, subsystem2, r_OB, A_IB, frame_ID1=np.zeros(3), frame_ID2=np.zeros(3), la_g0=None):

        super().__init__(subsystem1, subsystem2, r_OB, A_IB, frame_ID1, frame_ID2, la_g0)

        self.force_law = force_law

    def assembler_callback(self):

        super().assembler_callback()

        if self.force_law.g0 is None:
            self.force_law.g0 = self.angle(self.subsystem1.t0, np.concatenate((self.subsystem1.q0, self.subsystem2.q0)))

    def __g(self, t, q):
        return self.angle(t, q)

    def __g_q(self, t, q):
        return self.angle_q(t, q)

    def __W(self, t, q):
        n = np.array([0, 0, 1])
        K_J_R1 = self.K_J_R1(t, q) 
        K_J_R2 = self.K_J_R2(t, q)
        return np.concatenate([-K_J_R1.T @ n, K_J_R2.T @ n])

    def __W_q(self, t, q):
        nq1 = self.nq1
        nu1 = self.nu1
        n = np.array([0, 0, 1])
        K_J_R1_q1 = self.K_J_R1_q(t, q) 
        K_J_R2_q2 = self.K_J_R2_q(t, q)

        # dense blocks
        dense = np.zeros((self.nu, self.nq))
        dense[:nu1, :nq1] = np.einsum('i,ijk->jk', -n, K_J_R1_q1)
        dense[nu1:, nq1:] = np.einsum('i,ijk->jk',  n, K_J_R2_q2)

        return dense

    # public functions
    def pot(self, t, q):
        return self.force_law.pot(t, self.__g(t, q))

    def f_pot(self, t, q):
        g = self.__g(t, q)
        return -self.__W(t, q) * self.force_law.pot_g(t, g)

    def f_pot_q(self, t, q, coo):
        g = self.__g(t, q)
        dense = - self.__W_q(t, q) * self.force_law.pot_g(t, g) \
                - self.force_law.pot_gg(t, g) * np.outer( self.__W(t, q), self.__g_q(t, q) )
        coo.extend(dense, (self.uDOF, self.qDOF))