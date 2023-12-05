from numpy import zeros


class SpatialForceTransmission:
    r"""The force la_tau corresponds to a 3D force F represented w.r.t. the inertial frame, i.e., la_tau = I_F."""

    def __init__(self, subsystem, frame_ID=zeros(3), K_r_SP=zeros(3)):
        self.subsystem = subsystem
        self.frame_ID = frame_ID
        self.n_tau = 3
        self.r_OP = lambda t, q: subsystem.r_OP(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP)
        self.J_P = lambda t, q: subsystem.J_P(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP)
        self.J_P_q = lambda t, q: subsystem.J_P_q(
            t, q, frame_ID=frame_ID, K_r_SP=K_r_SP
        )

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.frame_ID)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.frame_ID)]

    def W_tau(self, t, q):
        return self.J_P(t, q).T
