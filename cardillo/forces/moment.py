from numpy import einsum, zeros
from vtk import VTK_VERTEX


class B_Moment:
    r"""Moment represented w.r.t. body-fixed B-basis

    Parameters
    ----------
    moment : np.ndarray (3,)
        Moment w.r.t. body-fixed B-basis as a callable function of time t.
    subsystem : object
        Object on which moment acts.
    xi : #TODO
    name : str
        Name of contribution.
    """

    def __init__(self, moment, subsystem, xi=zeros(3), name="moment"):
        if not callable(moment):
            self.moment = lambda t: moment
        else:
            self.moment = moment
        self.subsystem = subsystem
        self.xi = xi
        self.name = name

        self.B_J_R = lambda t, q: subsystem.B_J_R(t, q, xi=xi)
        self.B_J_R_q = lambda t, q: subsystem.B_J_R_q(t, q, xi=xi)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

    def h(self, t, q, u):
        return self.moment(t) @ self.B_J_R(t, q)

    def h_q(self, t, q, u):
        return einsum("i,ijk->jk", self.moment(t), self.B_J_R_q(t, q))

    def export(self, sol_i, **kwargs):
        r_OP = self.subsystem.r_OP(sol_i.t, sol_i.q[self.qDOF], xi=self.xi)
        A_IB = self.subsystem.A_IB(sol_i.t, sol_i.q[self.qDOF], xi=self.xi)
        I_M = A_IB @ self.moment(sol_i.t)

        points = [r_OP]
        cells = [(VTK_VERTEX, [0])]
        cell_data = dict(M=[I_M])
        return points, cells, None, cell_data


class Moment:
    r"""Moment represented w.r.t. I-basis

    Parameters
    ----------
    moment : np.ndarray (3,)
        Moment w.r.t. inertial I-basis as a callable function of time t.
    subsystem : object
        Object on which moment acts.
    xi : #TODO
    name : str
        Name of contribution.
    """

    def __init__(self, moment, subsystem, xi=zeros(3), name="moment"):
        if not callable(moment):
            self.moment = lambda t: moment
        else:
            self.moment = moment
        self.subsystem = subsystem
        self.xi = xi
        self.name = name

        self.A_IB = lambda t, q: subsystem.A_IB(t, q, xi=xi)
        self.A_IB_q = lambda t, q: subsystem.A_IB_q(t, q, xi=xi)
        self.B_J_R = lambda t, q: subsystem.B_J_R(t, q, xi=xi)
        self.B_J_R_q = lambda t, q: subsystem.B_J_R_q(t, q, xi=xi)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

    def h(self, t, q, u):
        return (self.moment(t) @ self.A_IB(t, q)) @ self.B_J_R(t, q)

    def h_q(self, t, q, u):
        I_M = self.moment(t)
        return einsum(
            "i,ijl,jk->kl", I_M, self.A_IB_q(t, q), self.B_J_R(t, q)
        ) + einsum("i,ijk->jk", I_M @ self.A_IB(t, q), self.B_J_R_q(t, q))

    def export(self, sol_i, **kwargs):
        r_OP = self.subsystem.r_OP(sol_i.t, sol_i.q[self.qDOF], xi=self.xi)
        I_M = self.moment(sol_i.t)

        points = [r_OP]
        cells = [(VTK_VERTEX, [0])]
        cell_data = dict(M=[I_M])
        return points, cells, None, cell_data
