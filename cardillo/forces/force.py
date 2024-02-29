from numpy import einsum, zeros


class Force:
    r"""Force represented w.r.t. I-basis

    Parameters
    ----------
    force : np.ndarray (3,)
        Force w.r.t. inertial I-basis as a callable function of time t.
    subsystem : object
        Object on which force acts.
    frame_ID : #TODO
    B_r_CP : np.ndarray (3,)
        Position vector of point of attack (P) w.r.t. center of mass (S) in body-fixed K-basis.
    name : str
        Name of contribution.
    """

    def __init__(
        self, force, subsystem, frame_ID=zeros(3), B_r_CP=zeros(3), name="force"
    ):
        if not callable(force):
            self.force = lambda t: force
        else:
            self.force = force
        self.subsystem = subsystem
        self.frame_ID = frame_ID
        self.name = name
        self.r_OP = lambda t, q: subsystem.r_OP(t, q, frame_ID, B_r_CP)
        self.J_P = lambda t, q: subsystem.J_P(t, q, frame_ID, B_r_CP)
        self.J_P_q = lambda t, q: subsystem.J_P_q(t, q, frame_ID, B_r_CP)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.frame_ID)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.frame_ID)]

    def E_pot(self, t, q):
        return -(self.force(t) @ self.r_OP(t, q))

    def h(self, t, q, u):
        return self.force(t) @ self.J_P(t, q)

    def h_q(self, t, q, u):
        return einsum("i,ijk->jk", self.force(t), self.J_P_q(t, q))

    def export(self, sol_i, **kwargs):
        points = [self.r_OP(sol_i.t, sol_i.q[self.qDOF])]
        cells = [("vertex", [[0]])]
        F = [self.force(sol_i.t)]
        cell_data = dict(F=[F])
        return points, cells, None, cell_data


class K_Force:
    r"""Force represented w.r.t. body-fixed K-basis

    Parameters
    ----------
    force : np.ndarray (3,)
        Force w.r.t. body-fixed K-basis as a callable function of time t.
    subsystem : object
        Object on which force acts.
    frame_ID : #TODO
    B_r_CP : np.ndarray (3,)
        Position vector of point of attack (P) w.r.t. center of mass (S) in body-fixed K-basis.
    name : str
        Name of contribution.
    """

    def __init__(self, force, subsystem, frame_ID=zeros(3), B_r_CP=zeros(3)):
        if not callable(force):
            self.force = lambda t: force
        else:
            self.force = force
        self.subsystem = subsystem
        self.frame_ID = frame_ID

        self.A_IK = lambda t, q: subsystem.A_IK(t, q, frame_ID=frame_ID)
        self.A_IK_q = lambda t, q: subsystem.A_IK_q(t, q, frame_ID=frame_ID)
        self.r_OP = lambda t, q: subsystem.r_OP(t, q, frame_ID=frame_ID, B_r_CP=B_r_CP)
        self.J_P = lambda t, q: subsystem.J_P(t, q, frame_ID=frame_ID, B_r_CP=B_r_CP)
        self.J_P_q = lambda t, q: subsystem.J_P_q(
            t, q, frame_ID=frame_ID, B_r_CP=B_r_CP
        )

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.frame_ID)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.frame_ID)]

    def h(self, t, q, u):
        return (self.A_IK(t, q) @ self.force(t)) @ self.J_P(t, q)

    def h_q(self, t, q, u):
        return einsum(
            "ijk,j,il->lk", self.A_IK_q(t, q), self.force(t), self.J_P(t, q)
        ) + einsum("i,ijk->jk", self.A_IK(t, q) @ self.force(t), self.J_P_q(t, q))
