from numpy import einsum, zeros


class B_Moment:
    """Moment defined in the body-fixed basis of a subsystem."""

    def __init__(self, B_M, subsystem, xi=zeros(3)):
        if not callable(B_M):
            self.B_M = lambda t: B_M
        else:
            self.B_M = B_M
        self.subsystem = subsystem
        self.xi = xi

        self.B_J_R = lambda t, q: subsystem.B_J_R(t, q, xi=xi)
        self.B_J_R_q = lambda t, q: subsystem.B_J_R_q(t, q, xi=xi)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

    def h(self, t, q, u):
        return self.B_M(t) @ self.B_J_R(t, q)

    def h_q(self, t, q, u):
        return einsum("i,ijk->jk", self.B_M(t), self.B_J_R_q(t, q))


class Moment:
    """Moment defined in the inertial fixed."""

    def __init__(self, I_M, subsystem, xi=zeros(3)):
        if not callable(I_M):
            self.I_M = lambda t: I_M
        else:
            self.I_M = I_M
        self.subsystem = subsystem
        self.xi = xi

        self.A_IB = lambda t, q: subsystem.A_IB(t, q, xi=xi)
        self.A_IB_q = lambda t, q: subsystem.A_IB_q(t, q, xi=xi)
        self.B_J_R = lambda t, q: subsystem.B_J_R(t, q, xi=xi)
        self.B_J_R_q = lambda t, q: subsystem.B_J_R_q(t, q, xi=xi)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

    def h(self, t, q, u):
        return (self.I_M(t) @ self.A_IB(t, q)) @ self.B_J_R(t, q)

    def h_q(self, t, q, u):
        I_M = self.I_M(t)
        return einsum(
            "i,ijl,jk->kl", I_M, self.A_IB_q(t, q), self.B_J_R(t, q)
        ) + einsum("i,ijk->jk", I_M @ self.A_IB(t, q), self.B_J_R_q(t, q))
