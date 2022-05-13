from numpy import einsum, zeros
from cardillo.math.numerical_derivative import approx_fprime


class K_Moment:
    """Moment defined in the body fixed frame of a subsystem."""

    def __init__(self, K_M, subsystem, frame_ID=zeros(3)):
        if not callable(K_M):
            self.K_M = lambda t: K_M
        else:
            self.K_M = K_M
        self.subsystem = subsystem
        self.frame_ID = frame_ID

        self.K_J_R = lambda t, q: subsystem.K_J_R(t, q, frame_ID=frame_ID)
        self.K_J_R_q = lambda t, q: subsystem.K_J_R_q(t, q, frame_ID=frame_ID)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.qDOF_P(self.frame_ID)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.uDOF_P(self.frame_ID)]

    def f_npot(self, t, q, u):
        return self.K_M(t) @ self.K_J_R(t, q)

    def f_npot_q(self, t, q, u, coo):
        f_q = einsum("i,ijk->jk", self.K_M(t), self.K_J_R_q(t, q))
        coo.extend(f_q, (self.uDOF, self.qDOF))


class Moment:
    """Moment defined in the inertial fixed."""

    def __init__(self, I_M, subsystem, frame_ID=zeros(3)):
        if not callable(I_M):
            self.I_M = lambda t: I_M
        else:
            self.I_M = I_M
        self.subsystem = subsystem
        self.frame_ID = frame_ID

        self.A_IK = lambda t, q: subsystem.A_IK(t, q, frame_ID=frame_ID)
        self.A_IK_q = lambda t, q: subsystem.A_IK_q(t, q, frame_ID=frame_ID)
        self.K_J_R = lambda t, q: subsystem.K_J_R(t, q, frame_ID=frame_ID)
        self.K_J_R_q = lambda t, q: subsystem.K_J_R_q(t, q, frame_ID=frame_ID)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.qDOF_P(self.frame_ID)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.uDOF_P(self.frame_ID)]

    def f_npot(self, t, q, u):
        return (self.I_M(t) @ self.A_IK(t, q)) @ self.K_J_R(t, q)

    def f_npot_q(self, t, q, u, coo):
        I_M = self.I_M(t)
        f_q = einsum("i,ijl,jk->kl", I_M, self.A_IK_q(t, q), self.K_J_R(t, q)) + einsum(
            "i,ijk->jk", I_M @ self.A_IK(t, q), self.K_J_R_q(t, q)
        )
        coo.extend(f_q, (self.uDOF, self.qDOF))

        # f_q_num = approx_fprime(q, lambda q: self.f_npot(t, q, u), method="3-point")
        # diff = f_q - f_q_num
        # from numpy.linalg import norm
        # error = norm(diff)
        # print(f"error f_q: {error}")
        # coo.extend(f_q_num, (self.uDOF, self.qDOF))
