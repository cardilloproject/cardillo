from numpy import einsum, zeros


class K_Moment:
    """Moment defined in the body fixed frame of a subsystem."""

    def __init__(self, K_M, subsystem, xi=zeros(3)):
        if not callable(K_M):
            self.K_M = lambda t: K_M
        else:
            self.K_M = K_M
        self.subsystem = subsystem
        self.xi = xi

        self.K_J_R = lambda t, q: subsystem.K_J_R(t, q, xi=xi)
        self.K_J_R_q = lambda t, q: subsystem.K_J_R_q(t, q, xi=xi)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

    def h(self, t, q, u):
        return self.K_M(t) @ self.K_J_R(t, q)

    def h_q(self, t, q, u):
        return einsum("i,ijk->jk", self.K_M(t), self.K_J_R_q(t, q))


class Moment:
    """Moment defined in the inertial fixed."""

    def __init__(self, I_M, subsystem, xi=zeros(3)):
        if not callable(I_M):
            self.I_M = lambda t: I_M
        else:
            self.I_M = I_M
        self.subsystem = subsystem
        self.xi = xi

        self.A_IK = lambda t, q: subsystem.A_IK(t, q, xi=xi)
        self.A_IK_q = lambda t, q: subsystem.A_IK_q(t, q, xi=xi)
        self.K_J_R = lambda t, q: subsystem.K_J_R(t, q, xi=xi)
        self.K_J_R_q = lambda t, q: subsystem.K_J_R_q(t, q, xi=xi)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

    def h(self, t, q, u):
        return (self.I_M(t) @ self.A_IK(t, q)) @ self.K_J_R(t, q)

    def h_q(self, t, q, u):
        I_M = self.I_M(t)
        return einsum(
            "i,ijl,jk->kl", I_M, self.A_IK_q(t, q), self.K_J_R(t, q)
        ) + einsum("i,ijk->jk", I_M @ self.A_IK(t, q), self.K_J_R_q(t, q))
