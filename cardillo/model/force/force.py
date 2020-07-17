from numpy import einsum, zeros
from cardillo.math import Numerical_derivative

class Force():
    r"""Force implementation."""

    def __init__(self, force, subsystem, frame_ID=zeros(3), K_r_SP=zeros(3)):
        if not callable(force):
            self.force = lambda t: force
        else:
            self.force = force
        self.subsystem = subsystem
        self.frame_ID = frame_ID
        self.r_OP  = lambda t, q: subsystem.r_OP(t, q, frame_ID, K_r_SP)
        self.J_P   = lambda t, q: subsystem.J_P(t, q, frame_ID, K_r_SP)
        self.J_P_q = lambda t, q: subsystem.J_P_q(t, q, frame_ID, K_r_SP)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.qDOF_P(self.frame_ID)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.uDOF_P(self.frame_ID)]
          
    def potential(self, t, q):
        return - ( self.force(t) @ self.r_OP(t, q) )

    def f_pot(self, t, q):
        return self.force(t) @ self.J_P(t, q)

    def f_pot_q(self, t, q, coo):
        f_q = einsum('i,ijk->jk', self.force(t), self.J_P_q(t, q))
        coo.extend(f_q, (self.uDOF, self.qDOF))

class K_Force():
    r"""Follower force implementation."""

    def __init__(self, force, subsystem, frame_ID=zeros(3), K_r_SP=zeros(3)):
        if not callable(force):
            self.force = lambda t: force
        else:
            self.force = force
        self.subsystem = subsystem
        self.frame_ID = frame_ID

        self.A_IK   = lambda t, q: subsystem.A_IK(t, q, frame_ID=frame_ID)
        self.A_IK_q = lambda t, q: subsystem.A_IK_q(t, q, frame_ID=frame_ID)
        self.r_OP   = lambda t, q: subsystem.r_OP(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP)
        self.J_P    = lambda t, q: subsystem.J_P(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP)
        self.J_P_q  = lambda t, q: subsystem.J_P_q(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.qDOF_P(self.frame_ID)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.uDOF_P(self.frame_ID)]

    def f_npot(self, t, q, u):
        return (self.A_IK(t, q) @ self.force(t)) @ self.J_P(t, q)

    def f_npot_q(self, t, q, u, coo):
        f_q = einsum('ijk,j,il->lk', self.A_IK_q(t, q), self.force(t), self.J_P(t, q)) \
            + einsum('i,ijk->jk', self.A_IK(t, q) @ self.force(t), self.J_P_q(t, q))
        coo.extend(f_q, (self.uDOF, self.qDOF))


