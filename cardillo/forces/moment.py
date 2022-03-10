from numpy import einsum, zeros


class K_Moment:
    def __init__(self, moment, subsystem, frame_ID=zeros(3)):
        if not callable(moment):
            self.moment = lambda t: moment
        else:
            self.moment = moment
        self.subsystem = subsystem
        self.frame_ID = frame_ID

        self.K_J_R = lambda t, q: subsystem.K_J_R(t, q, frame_ID=frame_ID)
        self.K_J_R_q = lambda t, q: subsystem.K_J_R_q(t, q, frame_ID=frame_ID)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.qDOF_P(self.frame_ID)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.uDOF_P(self.frame_ID)]

    def f_npot(self, t, q, u):
        return self.moment(t) @ self.K_J_R(t, q)

    def f_npot_q(self, t, q, u, coo):
        f_q = einsum("i,ijk->jk", self.moment(t), self.K_J_R_q(t, q))
        coo.extend(f_q, (self.uDOF, self.qDOF))
